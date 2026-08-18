# -*- coding: utf-8 -*-
'''
Convergence checks for the PM2 EcoRecover LCA dynamic simulation.

Run from the EXPOsan repository root with:

>>> python -m exposan.pm2_ecorecover_lca.convergence_check

This script tests whether selected dynamic PM2/design parameters reach similar
results at multiple simulation horizons. It reuses the uncertainty bounds from
``models.py`` so the checked low/high values stay synchronized with the Monte
Carlo model.
'''

import os
import numpy as np
import pandas as pd

from qsdsan import Model
from exposan.pm2_ecorecover_lca import results_path
from exposan.pm2_ecorecover_lca import ecorecover_lca as pmlca
from exposan.pm2_ecorecover_lca import models


DEFAULT_T_ENDS = (10, 20, 40, 60, 100) # t = 60, t_step = 0.5 is used in all simulations
DEFAULT_COMPONENTS = ('X_ALG', 'S_CO2', 'S_NH', 'S_NO', 'S_P')
DEFAULT_TRACKERS = (
    ('PBR20', 'unit', 'PBR20'),
    ('Effluent', 'stream', 'Effluent'),
)


def _relevant_parameter_data():
    '''
    Return parameter dictionaries that can affect PM2 convergence.

    The selected parameters are:
    - HRT for MIX, RET, MEM, and PBR1/PBR2/.../PBR20 volume synchronization.
    - PBR ``InD``, which the Monte Carlo setter maps to PM2 ``b_reactor``.
    - PBR ``light_intensity``, which rescales the exogenous light time series.
    '''
    selected = []
    for data in models._PARAMETERS:
        group = data.get('group')
        attr = data.get('attr')
        unit = data.get('unit')
        if attr == 'hrt' and unit in ('MIX', 'RET', 'MEM'):
            selected.append(data)
        elif group == 'Photobioreactor' and attr in (
                'hrt', 'InD', 'light_intensity'):
            selected.append(data)
    return tuple(selected)


def _parameter_label(data):
    '''
    Return a readable label for one convergence-test parameter.

    The parameter dictionaries use internal attribute names such as ``InD`` and
    ``light_intensity``. This helper converts them to labels that make the
    scenario and parameter tables easier to read.
    '''
    attr = data['attr']
    if attr == 'InD':
        return 'PM2 b_reactor via PBR InD'
    if attr == 'light_intensity':
        return 'PBR light intensity'
    unit = data.get('unit')
    if unit:
        return f'{unit} hrt'
    return 'PBR hrt'


def _build_scenarios(include_combined=True):
    '''
    Build baseline, one-factor low/high, and optionally combined low/high cases.
    '''
    parameters = _relevant_parameter_data()
    scenarios = [('baseline', {})]

    for data in parameters:
        label = _parameter_label(data)
        scenarios.append((f'{label} low', {id(data): data['low']}))
        scenarios.append((f'{label} high', {id(data): data['high']}))

    if include_combined:
        scenarios.append(
            ('all selected low', {id(data): data['low'] for data in parameters})
        )
        scenarios.append(
            ('all selected high', {id(data): data['high'] for data in parameters})
        )

    return parameters, scenarios


def _apply_scenario(system, parameters, scenario_values):
    '''
    Apply scenario parameter values through the same setter path used by Model.
    '''
    for data in parameters:
        value = scenario_values.get(id(data), data['expected'])
        models._parameter_setter(system, data)(value)


def _get_scope_object(system, kind, ID):
    '''
    Return a unit or stream scope object for convergence tracking.

    Convergence diagnostics read time-series data from QSDsan scopes. This
    helper finds the requested unit or stream and returns it only when scope
    data are available.
    '''
    registry = system.flowsheet.unit if kind == 'unit' else system.flowsheet.stream
    try:
        obj = getattr(registry, ID)
    except AttributeError:
        return None
    return obj if hasattr(obj, 'scope') else None


def _get_record(scope, component):
    '''
    Extract one component time series from a unit or stream scope.

    QSDsan unit and stream scopes store records in slightly different layouts.
    This helper hides that difference so convergence metrics can request
    components like ``S_P`` or ``X_ALG`` from either source.
    '''
    try:
        record = np.asarray(scope.record, dtype=float)
    except Exception:
        return None
    if record.size == 0:
        return None

    subject = getattr(scope, 'subject', None)

    # WasteStreamScope stores concentrations in component order followed by Q.
    if hasattr(subject, 'components'):
        if component == 'Q':
            idx = -1
        else:
            try:
                idx = subject.components.index(component)
            except Exception:
                idx = None
        if idx is not None:
            try:
                return record[:, idx]
            except Exception:
                return None

    # SanUnitScope stores state variables according to the scope header. Header
    # entries often look like ('PBR20', 'S_P') or ('Effluent', 'S_P [mg/L]').
    for i, header in enumerate(getattr(scope, 'header', ())):
        try:
            name = header[1]
        except Exception:
            name = str(header)
        if str(name).split(' ')[0] == component:
            try:
                return record[:, i]
            except Exception:
                return None

    return None


def _value_at_time(scope, component, target_time):
    '''
    Return the component value nearest to a requested simulation time.

    The convergence table reports values at selected horizons such as 10, 20,
    40, 60, and 100 days. This helper picks the recorded time point closest to
    each horizon.
    '''
    t = np.asarray(scope.time_series, dtype=float)
    y = _get_record(scope, component)
    if y is None or not len(t):
        return np.nan
    idx = int(np.argmin(np.abs(t - target_time)))
    return float(y[idx])


def _window_relative_change(scope, component, target_time, window):
    '''
    Return relative change across the final convergence-check window.

    This tells whether a concentration is still drifting near the end of the
    simulation horizon. Smaller values mean the endpoint is more stable over
    the selected window.
    '''
    t = np.asarray(scope.time_series, dtype=float)
    y = _get_record(scope, component)
    if y is None or not len(t):
        return np.nan
    mask = (t >= target_time - window) & (t <= target_time)
    recent = y[mask]
    if recent.size < 2:
        return np.nan
    return float(abs(recent[-1] - recent[0]) / max(abs(recent[-1]), 1e-6))


def _daily_periodic_relative_error(scope, component, target_time, period=1.):
    '''
    Compare the final day profile against the previous day profile.

    This is useful because the PM2 system may approach a daily periodic pattern
    rather than a flat steady state due to time-varying light and temperature.
    '''
    t = np.asarray(scope.time_series, dtype=float)
    y = _get_record(scope, component)
    if y is None or not len(t) or target_time < 2 * period:
        return np.nan

    previous = (t >= target_time - 2 * period) & (t <= target_time - period)
    current = (t >= target_time - period) & (t <= target_time)
    if previous.sum() < 2 or current.sum() < 2:
        return np.nan

    current_t = t[current] - (target_time - period)
    previous_t = t[previous] - (target_time - 2 * period)
    previous_y = np.interp(current_t, previous_t, y[previous])
    current_y = y[current]
    scale = max(np.nanmax(np.abs(current_y)), 1e-6)
    return float(np.nanmax(np.abs(current_y - previous_y)) / scale)


def _collect_state_results(system, scenario, t_ends, components, trackers, window):
    '''
    Build rows for the state-convergence output table.

    For each scenario, tracked unit/stream, component, and simulation horizon,
    this helper records the value, final-window drift, and daily periodic error.
    '''
    rows = []
    for location, kind, ID in trackers:
        obj = _get_scope_object(system, kind, ID)
        if obj is None:
            continue
        scope = obj.scope
        for t_end in t_ends:
            for component in components:
                rows.append({
                    'scenario': scenario,
                    't_end_d': t_end,
                    'location': location,
                    'component': component,
                    'value_mg_L': _value_at_time(scope, component, t_end),
                    'last_window_relative_change': _window_relative_change(
                        scope, component, t_end, window),
                    'daily_periodic_relative_error': (
                        _daily_periodic_relative_error(
                            scope, component, t_end, period=1.)
                    ),
                })
    return rows


def _collect_metric_results(system, scenario, t_ends):
    '''
    Collect process-level metrics after each dynamic run.

    These are final-state results from the finished simulation. They are repeated
    under the largest ``t_end`` when ``run_convergence_check`` simulates once to
    the maximum horizon.
    '''
    rows = []
    metric_model = Model(system)
    models._add_core_metrics(metric_model)
    metrics = metric_model.metrics
    for metric in metrics:
        try:
            value = metric()
        except Exception:
            value = np.nan
        rows.append({
            'scenario': scenario,
            't_end_d': max(t_ends),
            'metric': metric.name,
            'units': metric.units,
            'value': value,
        })
    return rows


def _parameter_rows(parameters, scenarios):
    '''
    Build rows describing the parameter values used in each scenario.

    The convergence workbook includes this table so low/high scenario results
    can be traced back to the exact expected, low, high, and reference values
    from the uncertainty parameter definitions.
    '''
    rows = []
    for scenario, scenario_values in scenarios:
        for data in parameters:
            rows.append({
                'scenario': scenario,
                'parameter': _parameter_label(data),
                'group': data.get('group'),
                'unit': data.get('unit', ''),
                'attr': data.get('attr'),
                'units': data.get('units'),
                'expected': data.get('expected'),
                'low': data.get('low'),
                'high': data.get('high'),
                'value_used': scenario_values.get(id(data), data['expected']),
                'reference': data.get('reference', ''),
            })
    return rows


def run_convergence_check(
        t_ends=DEFAULT_T_ENDS, t_step=0.5, method='RK23',
        components=DEFAULT_COMPONENTS, trackers=DEFAULT_TRACKERS,
        window=5, include_combined=True, output='pm2_convergence_check.xlsx',
        write_csv=True, print_t=False,
        ):
    '''
    Run convergence checks for PM2-sensitive uncertainty parameters.

    Parameters
    ----------
    t_ends : iterable(float)
        Simulation horizons to evaluate [d].
    t_step : float
        Dynamic simulation output interval [d].
    method : str
        ODE solver method passed to ``pm2_ecorecover_lca.run``.
    components : iterable(str)
        Component IDs to evaluate in time-series records.
    trackers : iterable(tuple)
        Tuples of ``(label, 'unit' or 'stream', ID)`` to inspect.
    window : float
        Look-back window for relative-change convergence checks [d].
    include_combined : bool
        If True, include combined all-low and all-high cases in addition to
        one-factor-at-a-time low/high cases.
    output : str
        Output Excel file path. Relative paths are written to ``results_path``.
    write_csv : bool
        Also write CSV versions of the state, metric, and parameter tables.
    print_t : bool
        Whether to print dynamic simulation time from QSDsan.

    Returns
    -------
    state_table : pandas.DataFrame
        Time-series concentration and convergence diagnostics.
    metric_table : pandas.DataFrame
        Core TEA/LCA metrics for the final simulation horizon.
    parameter_table : pandas.DataFrame
        Parameter values used by each scenario.
    '''
    t_ends = tuple(sorted(float(i) for i in t_ends))
    if not t_ends:
        raise ValueError('`t_ends` must include at least one simulation horizon.')

    parameters, scenarios = _build_scenarios(include_combined=include_combined)
    state_rows = []
    metric_rows = []

    for scenario, scenario_values in scenarios:
        print(f'Running convergence scenario: {scenario}')
        system = pmlca.create_system(set_global=True)
        _apply_scenario(system, parameters, scenario_values)
        pmlca.run(t=max(t_ends), t_step=t_step, method=method, print_t=print_t)
        state_rows.extend(_collect_state_results(
            system, scenario, t_ends, components, trackers, window))
        metric_rows.extend(_collect_metric_results(system, scenario, t_ends))

    state_table = pd.DataFrame(state_rows)
    metric_table = pd.DataFrame(metric_rows)
    parameter_table = pd.DataFrame(_parameter_rows(parameters, scenarios))

    if output:
        if not os.path.isabs(output):
            output = os.path.join(results_path, output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with pd.ExcelWriter(output) as writer:
            state_table.to_excel(writer, sheet_name='state convergence',
                                 index=False)
            metric_table.to_excel(writer, sheet_name='metrics', index=False)
            parameter_table.to_excel(writer, sheet_name='parameters',
                                     index=False)

        if write_csv:
            base, _ = os.path.splitext(output)
            state_table.to_csv(f'{base}_state_convergence.csv', index=False)
            metric_table.to_csv(f'{base}_metrics.csv', index=False)
            parameter_table.to_csv(f'{base}_parameters.csv', index=False)

    return state_table, metric_table, parameter_table


if __name__ == '__main__':
    run_convergence_check()
