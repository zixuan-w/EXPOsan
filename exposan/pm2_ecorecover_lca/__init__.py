# -*- coding: utf-8 -*-
'''
EXPOsan: Exposition of sanitation and resource recovery systems

This module is developed by:
    Ga-Yeong Kim <gayeong1225@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/EXPOsan/blob/main/LICENSE.txt
for license details.
'''

import os
import numpy as np
import qsdsan as qs
import biosteam as bst
from qsdsan import ImpactItem, StreamImpactItem
from qsdsan.utils import load_data
from exposan.utils import _init_modules
pm2_path = os.path.dirname(__file__)
module = os.path.split(pm2_path)[-1]
data_path, results_path, figures_path = \
    _init_modules(module, include_data_path=True, include_figures_path=True)

from ._components import *

# =============================================================================
# TEA and LCA settings
# =============================================================================

discount_rate = 0.05
tea_lifetime = 30
tea_uptime_ratio = 1

labor_wage_filename = 'US_labor_wage_state_M2022_dl.xlsx'
labor_wage_OCC_CODE = '51-8031'

def _labor_wage_file():
    '''Return the labor-wage workbook path.'''
    candidates = (
        os.path.join(data_path, labor_wage_filename),
        os.path.join(os.path.dirname(pm2_path), 'werf', 'data',
                     labor_wage_filename),
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f'Cannot find {labor_wage_filename!r} in pm2_ecorecover_lca/data '
        'or exposan/werf/data.'
    )


def load_labor_wage_data():
    '''
    Return state-level hourly median wages [USD/hr] for OCC_CODE 51-8031.

    Source workbook: U.S. Bureau of Labor Statistics OEWS May 2022 state data,
    ``US_labor_wage_state_M2022_dl.xlsx``. OCC_CODE 51-8031 is Water and
    Wastewater Treatment Plant and System Operators. Suppressed nonnumeric wage 
    cells are ignored.
    '''
    df = load_data(_labor_wage_file())
    mask = df['OCC_CODE'].astype(str).str.strip() == labor_wage_OCC_CODE
    if 'PRIM_STATE' in df:
        mask &= df['PRIM_STATE'].astype(str).str.strip() != 'PR'
    wages = []
    for value in df.loc[mask, 'H_MEDIAN']:
        try:
            wage = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(wage):
            wages.append(wage)
    if not wages:
        raise ValueError(
            f'No numeric H_MEDIAN values found for OCC_CODE '
            f'{labor_wage_OCC_CODE!r}.'
        )
    return np.asarray(wages, dtype=float)


def update_resource_settings():
    '''Return price and characterization-factor dictionaries.'''
    global price_dct, GWP_dct, EUTRO_dct
    price_dct = {
        # USD/kWh; 8.27 cents/kWh from EIA monthly Table 5.6.B, March 2025 YTD, U.S. Total industrial sector.
        'Electricity': 0.0827,
        'NaturalGas': 0.696, # USD per kg as of Dec. 2025 in the U.S. https://www.globalpetrolprices.com/natural_gas_prices/ ; converted from 0.048 USD/kWh
        'Labor': float(np.median(load_labor_wage_data())), #USD per hour in 2022
        'CitricAcid': 1.8595038517821776,
        'SodiumHypochlorite': 14.265088213315847,
        'CO2': 0.,
        'S_NH': 0.,
        'S_NO': 0.,
        'S_P': 0.,
        'COD': 0.,
        }
    GWP_dct = {
        'Electricity': 0.512,
        'NaturalGas': 0.59/0.735, #converted to per kg based on 0.735 kg/m3 density
        'CitricAcid': 5.56,
        'SodiumHypochlorite': 2.38,
        'CO2': 0.82,
        'S_NH': 0.,
        'S_NO': 0.,
        'S_P': 0.,
        'COD': 0.,
        }
    EUTRO_dct = {
        'Electricity': 0.000691,
        'NaturalGas': 0.000206/0.735, 
        'CitricAcid': 0.0205,
        'SodiumHypochlorite': 0.00189,
        'CO2': 0.000523,
        # Direct effluent eutrophication factors from U.S. EPA
        # https://www.epa.gov/system/files/documents/2023-06/life-cycle-nutrient-removal.pdf, p. 4-11.
        'S_NH': 1.,
        'S_NO': 1.05,
        'S_P': 7.29,
        'COD': 0.05,
        }
    # Keep BioSTEAM/QSDsan TEA electricity cost synchronized with this dictionary.
    bst.PowerUtility.price = price_dct['Electricity']
    return price_dct, GWP_dct, EUTRO_dct

update_resource_settings()

_components_loaded = False
def _load_components(reload=False):
    '''Load PM2 EcoRecover components and set thermo.'''
    global components, _components_loaded
    if not _components_loaded or reload:
        components = create_components()
        qs.set_thermo(components)
        _components_loaded = True
    return components


_impact_item_loaded = False
def _load_lca_data(reload=False):
    '''Load construction impact items and placeholder stream impact items.'''
    global _impact_item_loaded
    indicator_path = os.path.join(data_path, 'impact_indicators.csv')
    qs.ImpactIndicator.load_from_file(indicator_path)
    # impact_items.xlsx uses an ``EUT`` sheet; register it so the workbook can
    # load without changing the source data file.
    if qs.ImpactIndicator.get_indicator('EUT') is None:
        qs.ImpactIndicator(
            'EUT', alias='EUTRO', method='TRACI',
            category='environmental impact', unit='kg N-eq',
            description='Eutrophication potential placeholder indicator.',
            )

    if not _impact_item_loaded or reload:
        item_path = os.path.join(data_path, 'impact_items.xlsx')
        qs.ImpactItem.load_from_file(item_path)
        price_dct, GWP_dct, EUTRO_dct = update_resource_settings()

        def create_stream_impact_item(resource_ID):
            item_ID = f'{resource_ID}_item'
            if ImpactItem.get_item(item_ID) is None:
                StreamImpactItem(
                    ID=item_ID, functional_unit='kg',
                    GWP=GWP_dct[resource_ID],
                    EUT=EUTRO_dct[resource_ID],
                    )

        for resource_ID in (
                'NaturalGas', 'CitricAcid', 'SodiumHypochlorite', 'CO2',
            ):
            create_stream_impact_item(resource_ID)
        if ImpactItem.get_item('e_item') is None:
            ImpactItem(
                ID='e_item', functional_unit='kWh',
                price=price_dct['Electricity'],
                GWP=GWP_dct['Electricity'],
                EUT=EUTRO_dct['Electricity'],
                )
        if ImpactItem.get_item('EffluentEutrophication_item') is None:
            StreamImpactItem(
                ID='EffluentEutrophication_item', functional_unit='kg',
                GWP=0.,
                EUT=1.,
                )

        _impact_item_loaded = True

    return _impact_item_loaded

from . import _sanunits
from ._sanunits import *
from . import ecorecover_lca

batch_init = ecorecover_lca.batch_init
EcorecoverTEA = ecorecover_lca.EcorecoverTEA

def _sync_globals(sys=None):
    '''
    Mirror key objects from ``ecorecover_lca`` onto the package namespace.

    Users often work interactively with ``import exposan.pm2_ecorecover_lca as
    pl`` and expect objects like ``pl.eco`` or stream/unit IDs to be available
    after creating or running the system. This helper keeps those convenient
    aliases synchronized with the underlying system module.
    '''
    dct = globals()
    for name in (
            'cmps', 'resource_cmps', 'pm2', 'eco', 'bio_IDs',
            '_init_conds',
        ):
        if hasattr(ecorecover_lca, name):
            dct[name] = getattr(ecorecover_lca, name)
    sys = getattr(ecorecover_lca, 'eco', None) if sys is None else sys
    try:
        dct.update(sys.flowsheet.to_dict())
    except AttributeError:
        pass

def create_system(*args, **kwargs):
    '''
    Package-level wrapper for ``ecorecover_lca.create_system``.

    The wrapper returns the system and then exposes the new system, flowsheet,
    units, and streams on the package namespace for interactive use.
    '''
    sys = ecorecover_lca.create_system(*args, **kwargs)
    _sync_globals(sys)
    return sys


def run(*args, **kwargs):
    '''
    Package-level wrapper for the dynamic simulation run function.

    After running ``ecorecover_lca.run``, this wrapper refreshes package-level
    aliases so interactive commands such as ``pl.TE`` or ``pl.PBR1`` point to
    the latest flowsheet objects.
    '''
    output = ecorecover_lca.run(*args, **kwargs)
    _sync_globals()
    return output

from .models import create_model, run_uncertainty

__all__ = (
    'pm2_path',
    'data_path',
    'results_path',
    'figures_path',
    'create_components',
    '_load_components',
    '_load_lca_data',
    'discount_rate',
    'tea_lifetime',
    'tea_uptime_ratio',
    'price_dct',
    'GWP_dct',
    'EUTRO_dct',
    'batch_init',
    'create_system',
    'run',
    'create_model',
    'run_uncertainty',
    *_sanunits.__all__,
    # *system.__all__,
    # *model.__all__,
    )
