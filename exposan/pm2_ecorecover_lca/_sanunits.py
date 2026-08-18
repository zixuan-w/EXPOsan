#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPOsan: Exposition of sanitation and resource recovery systems

This module is developed by:

    Zixuan Wang <wyatt4428@gmail.com>
    Portia Ewing <portiae2@illinois.edu>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/EXPOsan/blob/main/LICENSE.txt
for license details.
"""
import biosteam as bst
import os, numpy as np
from qsdsan import SanUnit, Construction, WasteStream, System, unit_operations as su
from qsdsan.unit_operations import WWTpump
from qsdsan import processes as pc
from qsdsan.utils import auom, select_pipe, format_str, get_P_blower
from biosteam.units.design_tools import (
    CEPCI_by_year,
    compute_number_of_tanks_and_purchase_cost,
    mix_tank_purchase_cost_algorithms,
    storage_tank_purchase_cost_algorithms,
    vessel_material_factors,
)
from biosteam.units.compressor import IsothermalCompressor
from biosteam.units.decorators import cost
from warnings import warn

import math
from math import pi, ceil
__all__ = ('Photobioreactor',
           'Ecorecoverypump',
           'Mixtank',
           'HRTCSTR',
           'CO2Supply',
           'LCAResourceInput',
           'Ultrafiltration',
           'AlgaeCentrifuge',
           )
#%%
CSTR = su.CSTR
lb_to_kg = 0.453592
acre_to_sq_m = 4046.86
sq_feet_to_sq_m = 10.7639
m_to_feet = 3.28084
euro_to_usd = 1.16
CEPCI_by_year.update({
    2022: 816.0,
    2023: 797.9,
    2024: 798.0,   # https://reg.lub.lu.se/luur/download?func=downloadFile&recordOId=9209157&fileOId=9209158
    2025: 811.0,   # estimated as 2024 * 1.016
})
hours_per_year = 365 * 24


def _annual_to_hourly_cost(cost):
    '''
    Convert an annual OPEX estimate to QSDsan's hourly add_OPEX basis.

    The unit maintenance and labor assumptions are usually annual values, but
    BioSTEAM/QSDsan ``add_OPEX`` entries are hourly rates.
    '''
    return cost / hours_per_year

def _labor_wage():
    '''Return system-level labor wage [USD/hr].'''
    from exposan import pm2_ecorecover_lca as pmlca
    try:
        return pmlca.price_dct['Labor']
    except KeyError:
        return 0.


def _installed_capital(unit, IDs=None, include_parallel=True):
    '''
    Return installed capital for selected purchase-cost items in one unit.

    Maintenance assumptions are often a percentage of installed capital. This
    helper applies the unit's bare-module, design, pressure, material, and
    parallel factors consistently before calculating those percentages.
    '''
    C = unit.baseline_purchase_costs
    if IDs is None:
        IDs = tuple(C)
    total = 0.
    for ID in IDs:
        if ID not in C:
            continue
        total += (
            C[ID] * unit.F_BM.get(ID, 1.) * unit.F_D.get(ID, 1.)
            * unit.F_P.get(ID, 1.) * unit.F_M.get(ID, 1.)
        )
    if include_parallel:
        total *= unit.parallel.get('self', 1)
    return total


def _require_nonnegative(name, value):
    '''
    Validate a module-level numeric input that cannot be negative.

    Several small helper calculations use this shared check before class-level
    validators are available.
    '''
    if value < 0:
        raise ValueError(f'`{name}` must be non-negative.')
    return value


def _warn_if_flow_outside_NEIWPCC(unit, Q_m3_d, name):
    '''
    Warn when a NEIWPCC labor correlation is used outside its flow range.

    The UF and centrifuge labor-hour correlations were fitted for larger water
    treatment flows. This helper keeps the calculation running but flags when
    the current design flow is outside the cited range.
    '''
    if 0 < Q_m3_d < 1400 or Q_m3_d > 76000:
        warn(
            f'The NEIWPCC labor-hour relationship for {name} may not hold '
            f'for flow rates outside 1400-76000 m3/d; current flow is '
            f'{Q_m3_d:.3g} m3/d.',
            RuntimeWarning,
            stacklevel=3,
        )

def _initial_volume_from_hrt(hrt, sizing_flow=None, n_series=1,
                             fallback=1000.):
    '''
    Return an initial CSTR working volume [m3] from HRT and flow.

    QSDsan dynamic CSTRs need this initial volume before stream flows are
    solved.
    '''
    if hrt <= 0:
        raise ValueError('`hrt` must be positive.')
    if n_series <= 0:
        raise ValueError('`n_series` must be positive.')
    if sizing_flow is not None:
        if sizing_flow < 0:
            raise ValueError('`sizing_flow` must be non-negative.')
        if sizing_flow > 0:
            return hrt * sizing_flow / n_series
    return fallback

class HRTCSTR(CSTR):
    '''
    Dynamic CSTR wrapper that calculates ``V_max`` from hydraulic retention
    time and a user-specified sizing flow.

    QSDsan dynamic CSTRs need ``V_max`` when the unit is initialized, before the
    dynamic simulation has solved the stream flow rates. This helper class
    calculates that initial reactor volume as:

    ``V_max = hrt * sizing_flow / n_series``

    where ``hrt`` is in hr, ``sizing_flow`` is in m3/hr, and ``n_series`` is
    the number of equal CSTR blocks representing the same total hydraulic
    residence time. If ``sizing_flow`` is not provided or is zero, a fallback
    initial volume from :func:`_initial_volume_from_hrt` is used.

    This class is intended for PM2 CSTR blocks that should participate in the
    dynamic process model but do not need separate design, construction, or
    costing algorithms. In ``pm2_ecorecover_lca``, it is mainly used for
    PBR2-PBR20 so that all PBR blocks can update their working volume when
    PBR hydraulic retention time changes during uncertainty analysis.
    '''
    _units = {
        **CSTR._units,
        'Sizing flow': 'm3/hr',
        'Hydraulic retention time': 'hr',
        'Number of CSTRs in series': '',
    }

    def __init__(self, ID='', ins=None, outs=(), thermo=None,
                 init_with='WasteStream', split=None,
                 hrt=1., sizing_flow=None, n_series=1, **kwargs):
        V_init = _initial_volume_from_hrt(
            hrt, sizing_flow=sizing_flow, n_series=n_series,
        )
        CSTR.__init__(
            self, ID=ID, ins=ins, outs=outs, split=split, thermo=thermo,
            init_with=init_with, V_max=V_init, **kwargs,
        )
        self.hrt = hrt
        self.sizing_flow = sizing_flow
        self.n_series = n_series

    @property
    def hrt(self):
        return self._hrt

    @hrt.setter
    def hrt(self, value):
        if value <= 0:
            raise ValueError('`hrt` must be positive.')
        self._hrt = value

    @property
    def sizing_flow(self):
        return self._sizing_flow

    @sizing_flow.setter
    def sizing_flow(self, value):
        if value is not None and value < 0:
            raise ValueError('`sizing_flow` must be non-negative.')
        self._sizing_flow = value

class Mixtank(CSTR):
    '''
    Tank with design, BioSTEAM mixing-tank cost, and power estimation.

    The process model, aeration behavior, and dynamic mass balances are inherited
    from :class:`qsdsan.unit_operations.CSTR`. 

    Parameters
    ----------
    hrt : float
        Hydraulic retention time used to size tank working volume from the
        configured ``sizing_flow``, ``V_max = hrt * Q`` [hr]. The default is
        3.3 hr.
    sizing_flow : float, optional
        Initial flow basis used to initialize dynamic CSTR volume before stream
        flows are solved [m3/hr].
    V_wf : float
        Working liquid volume divided by total tank volume.
    vessel_type : str
        BioSTEAM mixing-tank purchase-cost algorithm.
    vessel_material : str
        Tank construction material used by the BioSTEAM material factor.
    include_aeration_power : bool
        Whether to include blower power.
    include_mixing_power : bool
        Whether to include mechanical mixing power.
    Q_air : float, optional
        Field airflow in [m3/d]. If omitted, airflow is read from a
        :class:`DiffusedAeration` object supplied through ``aeration``; if
        neither is available, it defaults to ``0.1 * V_max * 1440``.
    mixing_intensity : float, optional
        Velocity-gradient mixing intensity, ``G``, in [1/s]. When supplied, it
        takes precedence over ``kW_per_m3``.
    kW_per_m3 : float
        Specific mechanical mixing power in [kW/m3].
    unit_diffuser_flow_rate : float
        The flow rate a single diffuser provides [m3/d]. The default is 61.2 m3/d.
        source: https://www.aquariustechnologies.com/wp-content/uploads/2019/07/Quantaer-Diffused-Aeration-Systems-Web.pdf
    diffuser_unit_cost: float
        The cost of one diffusor [USD]. The default is 445 USD based on 
        Appendix 128-3 in https://cleanwaterservices.org/wp-content/uploads/2025/10/04-TM12_ForestGroveWRRFAerationEvaluation.pdf
    Labor wage is read from ``pm2_ecorecover_lca.price_dct['Labor']`` [USD/hr].
    Air compressor carbon steel follows the compressor weight relationship
        [kg] = 16.013 * aeration power [hp] + 75.813.
        source: https://us.kaeser.com/products-and-solutions/rotary-screw-compressors/3-hp.aspx
    
    References
    ----------
    [1] BioSTEAM Development Group. BioSTEAM Tank and MixTank Cost
        Algorithms.
        https://biosteam.readthedocs.io/en/latest/_modules/biosteam/units/tank.html.

    [2] BioSTEAM Development Group. BioSTEAM IsothermalCompressor Screw
        Compressor Cost Algorithm.
        https://biosteam.readthedocs.io/en/latest/_modules/biosteam/units/compressor.html#IsothermalCompressor.

    [3] QSDsan Development Group. Wastewater Treatment Design Utilities:
        Blower Power Estimation.
        https://github.com/QSD-Group/QSDsan.

    [4] QSDsan Development Group. Static Reactor Design Utilities:
        Mechanical Mixing Power from Specific Power or Velocity Gradient.
        https://github.com/QSD-Group/QSDsan.

    [5] Aquarius Technologies. Quantaer Diffused Aeration Systems.
        Diffuser flow capacity used as the default unit diffuser flow rate.
        https://www.aquariustechnologies.com/wp-content/uploads/2019/07/Quantaer-Diffused-Aeration-Systems-Web.pdf.

    [6] Clean Water Services. Forest Grove WRRF Aeration Evaluation,
        Appendix 128-3. Diffuser unit cost basis.
        https://cleanwaterservices.org/wp-content/uploads/2025/10/04-TM12_ForestGroveWRRFAerationEvaluation.pdf.

    [7] KAESER Compressors. Rotary Screw Compressors, 3 hp Product
        Specification. Compressor weight correlation basis.
        https://us.kaeser.com/products-and-solutions/rotary-screw-compressors/3-hp.aspx.

    [8] Gebreslassie, B. H.; Waymire, R.; You, F. Sustainable Design and
        Synthesis of Algae-Based Biorefinery for Simultaneous Hydrocarbon
        Biofuel Production and Carbon Sequestration.
        AIChE J. 2013, 59 (5), 1599-1621.
        https://doi.org/10.1002/aic.13952.

    [9] NEIWPCC. Northeast Guide for Estimating Staffing at Publicly and
        Privately Owned Wastewater Treatment Plants.
        https://neiwpcc.org/wp-content/uploads/2020/08/NEIWPCC-Northeast-Staffing-Guide.pdf.
    '''
    _F_BM_default = {
        'Tank': 2.3,
        'Air compressor': 2.15,
        'Diffusers': 1.,
    }
    _units = {
        **CSTR._units,
        'Influent flow': 'm3/hr',
        'Hydraulic retention time': 'hr',
        'Tank volume': 'm3',
        'Tank width': 'm',
        'Tank depth': 'm',
        'Tank length': 'm',
        'Volume of concrete wall': 'm3',
        'Volume of concrete slab': 'm3',
        'Total volume': 'm3',
        'Aeration power': 'kW',
        'Mechanical mixing power': 'kW',
        'Total power': 'kW',
        'Air flow rate': 'm3/d',
        'Air flow rate at compressor': 'cfm',
        'Number of air compressors': '',
        'Air compressor carbon steel': 'kg',
        'Number of diffusers': '',
        'Diffuser area': 'm2',
        'Diffuser stainless steel': 'kg',
    }
    purchase_cost_algorithms = {
        **mix_tank_purchase_cost_algorithms,
        **storage_tank_purchase_cost_algorithms,
    }

    def __init__(self, ID='', ins=None, outs=(), thermo=None,
                 init_with='WasteStream', split=None,
                 hrt=3.3,
                 sizing_flow=None,
                 W_tank=6.4, D_tank=3.65, freeboard=0.61,
                 t_wall=None, t_slab=None, aeration=2.0, DO_ID='S_O2',
                 suspended_growth_model=None, gas_stripping=False,
                 gas_IDs=None, stripping_kLa_min=None, K_Henry=None,
                 D_gas=None, p_gas_atm=None, isdynamic=True,
                 exogenous_vars=(), V_wf=0.8,
                 vessel_type='Conventional',
                 vessel_material='Stainless steel',
                 include_aeration_power=False,
                 include_mixing_power=True, Q_air=None,
                 mixing_intensity=None, kW_per_m3=0.0985,
                 blower_T=20, P_atm=101.325, P_inlet_loss=1,
                 P_diffuser_loss=7, h_submergance=5.18,
                 blower_efficiency=0.7, blower_K=0.283,
                 unit_diffuser_flow_rate=61.2, diffuser_unit_cost=445.,
                 **kwargs):
        V_init = _initial_volume_from_hrt(
            hrt, sizing_flow=sizing_flow,
        )
        CSTR.__init__(
            self, ID=ID, ins=ins, outs=outs, split=split, thermo=thermo,
            init_with=init_with, V_max=V_init, W_tank=W_tank,
            D_tank=D_tank, freeboard=freeboard, t_wall=t_wall,
            t_slab=t_slab, aeration=aeration, DO_ID=DO_ID,
            suspended_growth_model=suspended_growth_model,
            gas_stripping=gas_stripping, gas_IDs=gas_IDs,
            stripping_kLa_min=stripping_kLa_min, K_Henry=K_Henry,
            D_gas=D_gas, p_gas_atm=p_gas_atm, isdynamic=isdynamic,
            exogenous_vars=exogenous_vars, **kwargs,
        )
        self.W_tank = W_tank
        self.D_tank = D_tank
        self.freeboard = freeboard
        self.t_wall = t_wall
        self.t_slab = t_slab
        self.hrt = hrt
        self.sizing_flow = sizing_flow
        self.V_wf = V_wf
        self.vessel_type = vessel_type
        self.vessel_material = vessel_material
        self.include_aeration_power = bool(include_aeration_power)
        self.include_mixing_power = bool(include_mixing_power)
        self.Q_air = Q_air
        self.mixing_intensity = mixing_intensity
        self.kW_per_m3 = kW_per_m3
        self.blower_T = blower_T
        self.P_atm = P_atm
        self.P_inlet_loss = P_inlet_loss
        self.P_diffuser_loss = P_diffuser_loss
        self.h_submergance = h_submergance
        self.blower_efficiency = blower_efficiency
        self.blower_K = blower_K
        self.unit_diffuser_flow_rate = unit_diffuser_flow_rate
        self.diffuser_unit_cost = diffuser_unit_cost

    @staticmethod
    def _require_positive(name, value):
        if value <= 0:
            raise ValueError(f'`{name}` must be positive.')
        return value

    @staticmethod
    def _require_nonnegative(name, value):
        if value < 0:
            raise ValueError(f'`{name}` must be non-negative.')
        return value

    @property
    def W_tank(self):
        return self._W_tank

    @W_tank.setter
    def W_tank(self, value):
        self._W_tank = self._require_positive('W_tank', value)

    @property
    def D_tank(self):
        return self._D_tank

    @D_tank.setter
    def D_tank(self, value):
        self._D_tank = self._require_positive('D_tank', value)

    @property
    def freeboard(self):
        return self._freeboard

    @freeboard.setter
    def freeboard(self, value):
        self._freeboard = self._require_nonnegative('freeboard', value)

    @property
    def t_wall(self):
        if self._t_wall is not None:
            return self._t_wall
        depth_ft = self.D_tank * m_to_feet
        return 0.3048 + max(depth_ft - 12, 0) * 0.0254

    @t_wall.setter
    def t_wall(self, value):
        self._t_wall = (
            None if value is None
            else self._require_positive('t_wall', value)
        )

    @property
    def t_slab(self):
        if self._t_slab is not None:
            return self._t_slab
        return self.t_wall + 0.0508

    @t_slab.setter
    def t_slab(self, value):
        self._t_slab = (
            None if value is None
            else self._require_positive('t_slab', value)
        )

    @property
    def V_wf(self):
        return self._V_wf

    @V_wf.setter
    def V_wf(self, value):
        value = self._require_positive('V_wf', value)
        if value > 1:
            raise ValueError('`V_wf` cannot exceed 1.')
        self._V_wf = value

    @property
    def hrt(self):
        return self._hrt

    @hrt.setter
    def hrt(self, value):
        self._hrt = self._require_positive('hrt', value)

    def _get_hrt_sizing_flow(self):
        if self.sizing_flow and self.sizing_flow > 0:
            return self.sizing_flow
        if not self.ins:
            return 0.
        Q = getattr(self.ins[0], 'F_vol', 0.)
        return Q if Q > 0 else 0.

    def _get_total_inlet_flow(self):
        return self._get_hrt_sizing_flow()

    @property
    def sizing_flow(self):
        return self._sizing_flow

    @sizing_flow.setter
    def sizing_flow(self, value):
        self._sizing_flow = (
            None if value is None
            else self._require_nonnegative('sizing_flow', value)
        )

    @property
    def Q_air(self):
        return self._Q_air

    @Q_air.setter
    def Q_air(self, value):
        self._Q_air = (
            None if value is None
            else self._require_nonnegative('Q_air', value)
        )

    @property
    def mixing_intensity(self):
        return self._mixing_intensity

    @mixing_intensity.setter
    def mixing_intensity(self, value):
        self._mixing_intensity = (
            None if value is None
            else self._require_nonnegative('mixing_intensity', value)
        )

    @property
    def kW_per_m3(self):
        return self._kW_per_m3

    @kW_per_m3.setter
    def kW_per_m3(self, value):
        self._kW_per_m3 = self._require_nonnegative('kW_per_m3', value)

    @property
    def diffuser_unit_cost(self):
        return self._diffuser_unit_cost

    @diffuser_unit_cost.setter
    def diffuser_unit_cost(self, value):
        self._diffuser_unit_cost = self._require_nonnegative(
            'diffuser_unit_cost', value,
        )

    @property
    def unit_diffuser_flow_rate(self):
        return self._unit_diffuser_flow_rate

    @unit_diffuser_flow_rate.setter
    def unit_diffuser_flow_rate(self, value):
        self._unit_diffuser_flow_rate = self._require_positive(
            'unit_diffuser_flow_rate', value,
        )

    @property
    def vessel_type(self):
        return self._vessel_type

    @vessel_type.setter
    def vessel_type(self, value):
        try:
            algorithm = self.purchase_cost_algorithms[value]
        except KeyError:
            valid = ', '.join(self.purchase_cost_algorithms)
            raise ValueError(
                f'`vessel_type` must be one of: {valid}.'
            ) from None
        self._vessel_type = value
        self.purchase_cost_algorithm = algorithm

    @property
    def vessel_material(self):
        return self._vessel_material

    @vessel_material.setter
    def vessel_material(self, value):
        try:
            factor = vessel_material_factors[value]
        except KeyError:
            valid = ', '.join(vessel_material_factors)
            raise ValueError(
                f'No vessel material factor is available for {value!r}; '
                f'choose one of: {valid}.'
            ) from None
        self._vessel_material = value
        self.F_M['Tank'] = factor
    def _init_lca(self):
        self.include_construction = True
        self.construction = [
            Construction("stainless_steel", linked_unit=self,
                         item = "StainlessSteel", 
                         quantity_unit= "kg"),
            Construction("carbon_steel", linked_unit=self,
                         item = "LowCarbonSteel", 
                         quantity_unit= "kg"),
            ]
    def _design(self):
        D = self.design_results
        Q = self._get_hrt_sizing_flow()
        if Q > 0:
            self.V_max = self.hrt * Q
        D['Influent flow'] = Q
        D['Hydraulic retention time'] = self.hrt
        V = self.V_max
        W = self.W_tank
        depth = self.D_tank
        L = V / (W * depth)
        t_wall = self.t_wall
        t_slab = self.t_slab
        total_depth = depth + self.freeboard

        # Source: geometry in the commented QSDsan dynamic CSTR._design.
        D['Tank volume'] = V
        D['Tank width'] = W
        D['Tank depth'] = depth
        D['Tank length'] = L
        D['Volume of wall'] = (
            2 * (L + 2 * t_wall) * t_wall * total_depth
            + 2 * W * t_wall * total_depth
        )
        D['Volume of slab'] = (
            (L + 2 * t_wall) * (W + 2 * t_wall) * t_slab
        )
        self.construction[0].quantity = (D['Volume of wall']+D['Volume of slab'])*8000 # kg, assume 8000 kg/m3 stainless steel density

        D['Total volume'] = V / self.V_wf
        

        if self.include_aeration_power:
            Q_air = self._get_Q_air()
            Q_air_acfm = auom('m3/d').convert(Q_air, 'cfm')
            aeration_power = self._get_aeration_power()
            compressor_algorithm = (
                IsothermalCompressor.baseline_cost_algorithms['Screw']
            )
            max_acfm = compressor_algorithm.acfm_bounds[1]
            N_compressors = (
                ceil(Q_air_acfm / max_acfm) if Q_air_acfm > 0 else 0
            )
            N_diffusers = (
                ceil(Q_air / self.unit_diffuser_flow_rate)
                if Q_air > 0 else 0
            )
            # Source: full-floor diffuser coverage specified for this model.
            diffuser_area = W * L
        else:
            Q_air = Q_air_acfm = aeration_power = 0.
            N_compressors = 0
            N_diffusers = 0
            diffuser_area = 0.

        # Source: maximum ACFM from BioSTEAM's screw-compressor algorithm.
        D['Air flow rate'] = Q_air
        D['Air flow rate at compressor'] = Q_air_acfm
        D['Number of air compressors'] = N_compressors
        D['Number of diffusers'] = N_diffusers
        # Source: Kaeser rotary screw compressor specification,
        # https://us.kaeser.com/products-and-solutions/rotary-screw-compressors/3-hp.aspx
        aeration_power_hp = auom('kW').convert(aeration_power, 'hp')
        D['Air compressor carbon steel'] = self.construction[1].quantity = (
            16.013 * aeration_power_hp + 75.813
            if aeration_power > 0 else 0.
        ) #kg, fitted relationship

        D['Diffuser area'] = diffuser_area #m2
        D['Diffuser stainless steel'] = diffuser_area * 181 / 18.5 #kg
        self.construction[0].quantity += D['Diffuser stainless steel'] 

    def _get_Q_air(self):
        Q_air = self.Q_air
        if Q_air is None and isinstance(self.aeration, pc.DiffusedAeration):
            Q_air = self.aeration.Q_air
        if Q_air is None:
            # Default: 0.1 m3 air/(m3 tank*min), converted here to m3/d.
            Q_air = 0.1 * self.V_max * 1440
        return Q_air

    def _get_aeration_power(self):
        if not self.include_aeration_power:
            return 0.
        Q_air = self._get_Q_air()

        # Source: QSDsan wwt_design.get_P_blower; it expects airflow in m3/min.
        return get_P_blower(
            Q_air / 1440,
            T=self.blower_T,
            P_atm=self.P_atm,
            P_inlet_loss=self.P_inlet_loss,
            P_diffuser_loss=self.P_diffuser_loss,
            h_submergance=self.h_submergance,
            efficiency=self.blower_efficiency,
            K=self.blower_K,
        )

    def _get_mixing_power(self):
        if not self.include_mixing_power:
            return 0.
        if self.mixing_intensity is None:
            return self.kW_per_m3 * self.V_max

        # Source: QSDsan static Reactor.kW_per_m3 (P/V = mu*G^2/1000).
        self._mixed.mix_from(self.ins)
        specific_power = self._mixed.mu * self.mixing_intensity**2 / 1000
        return specific_power * self.V_max

    def _calc_replacement_cost(self):
        # Maintenance source: 6% of total installed capital,
        # https://doi.org/10.1016/j.bej.2017.08.006.
        return _annual_to_hourly_cost(0.06 * _installed_capital(self))

    def _calc_maintenance_labor_cost(self):
        # Tank labor source: NEIWPCC Northeast Staffing Guide,
        # https://neiwpcc.org/wp-content/uploads/2020/08/NEIWPCC-Northeast-Staffing-Guide.pdf.
        N_tanks = self.parallel.get('self', 1)
        hours = 36.5 * N_tanks
        if self.include_aeration_power:
            hours += 73 * self.design_results.get(
                'Number of air compressors', 0,
            )
        return _annual_to_hourly_cost(hours * _labor_wage())

    def _cost(self):
        D = self.design_results
        C = self.baseline_purchase_costs

        # Source: BioSTEAM Tank._cost used by BioSTEAM MixTank.
        N, Cp = compute_number_of_tanks_and_purchase_cost(
            D['Total volume'], self.purchase_cost_algorithm,
        )
        if not N:
            C.clear()
            self.parallel.pop('self', None)
            self.power_utility.rate = 0.
            self.add_OPEX['labor'] = 0.
            self.add_OPEX['maintenance and replacement'] = 0.
            return

        self.parallel['self'] = N
        default_material = self.purchase_cost_algorithm.material
        C['Tank'] = Cp / vessel_material_factors.get(default_material, 1.)

        aeration_power = self._get_aeration_power()
        mixing_power = self._get_mixing_power()
        total_power = aeration_power + mixing_power
        D['Aeration power'] = aeration_power
        D['Mechanical mixing power'] = mixing_power
        D['Total power'] = total_power

        if self.include_aeration_power:
            N_compressors = D['Number of air compressors']
            if N_compressors and aeration_power > 0:
                algorithm = (
                    IsothermalCompressor.baseline_cost_algorithms['Screw']
                )
                total_hp = auom('kW').convert(aeration_power, 'hp')
                hp_per_compressor = total_hp / N_compressors
                compressor_cost = (
                    N_compressors * CEPCI_by_year[2022] / algorithm.CE #assume basis year is 2022
                    * algorithm.cost(hp_per_compressor)
                )
            else:
                compressor_cost = 0.

            # BioSTEAM later scales all entries by parallel['self'].
            C['Air compressor'] = compressor_cost / N
            C['Diffusers'] = (
                D['Number of diffusers'] * self.diffuser_unit_cost / N
            )
            self.F_M['Air compressor'] = 2.5
        else:
            C.pop('Air compressor', None)
            C.pop('Diffusers', None)

        # BioSTEAM scales utilities by parallel['self'] after _cost().
        self.power_utility.rate = total_power / N
        self.add_OPEX['labor'] = self._calc_maintenance_labor_cost()
        self.add_OPEX['maintenance and replacement'] = (
            self._calc_replacement_cost()
        )

class CO2Supply(SanUnit):
    '''
    Pass-through unit that estimates purchased CO2 makeup and operating cost.

    The influent is copied to the effluent without adding CO2 to the process
    mass balance. Supply is estimated from the difference between a target and
    user-specified influent dissolved-CO2 concentration.

    Parameters
    ----------
    target_CO2 : float
        Target dissolved-CO2 concentration in [mg/L]. The default is 30.
    influent_CO2 : float
        Influent dissolved-CO2 concentration in [mg/L]. The default is 20.
    excess_fraction : float
        Fractional allowance for unmodeled CO2 losses. The default is 0.10.
    CO2_ID : str
        Component ID representing dissolved CO2. The default is 'S_CO2'.
    CO2_price : float
        CO2 price in 2016 [USD/metric tonne]. The default is 45.
    Labor wage is read from ``pm2_ecorecover_lca.price_dct['Labor']`` [USD/hr].

    References
    ----------
    CO2 purchase price:
        https://docs.nlr.gov/docs/fy19osti/72716.pdf
    CEPCI price escalation:
        BioSTEAM CEPCI_by_year, converted from 2016 to 2022 basis.
    Labor:
        https://www.reliantbevcarb.com/products/bulkco2
    '''
    _N_ins = 2
    _N_outs = 1
    _units = {
        'Influent CO2 concentration': 'mg/L',
        'Target CO2 concentration': 'mg/L',
        'Wastewater flow': 'L/hr',
        'Base CO2 makeup': 'kg/hr',
        'CO2 supply': 'kg/hr', 
        'Excess CO2 fraction': '',
    }

    def __init__(
            self, ID='', ins=None, outs=(), thermo=None,
            init_with='WasteStream', target_CO2=30., influent_CO2=20.,
            excess_fraction=0.10, CO2_ID='S_CO2',
            CO2_price=45., #source: https://docs.nlr.gov/docs/fy19osti/72716.pdf
        ):
        SanUnit.__init__(
            self, ID=ID, ins=ins, outs=outs, thermo=thermo,
            init_with=init_with,
        )
        self.target_CO2 = target_CO2
        self.influent_CO2 = influent_CO2
        self.excess_fraction = excess_fraction
        self.CO2_ID = CO2_ID
        self.CO2_price = CO2_price

    @staticmethod
    def _require_nonnegative(name, value):
        if value < 0:
            raise ValueError(f'`{name}` must be non-negative.')
        return value

    @property
    def target_CO2(self):
        return self._target_CO2

    @target_CO2.setter
    def target_CO2(self, value):
        self._target_CO2 = self._require_nonnegative('target_CO2', value)

    @property
    def influent_CO2(self):
        return self._influent_CO2

    @influent_CO2.setter
    def influent_CO2(self, value):
        self._influent_CO2 = self._require_nonnegative(
            'influent_CO2', value,
        )

    @property
    def excess_fraction(self):
        return self._excess_fraction

    @excess_fraction.setter
    def excess_fraction(self, value):
        self._excess_fraction = self._require_nonnegative(
            'excess_fraction', value,
        )

    @property
    def CO2_ID(self):
        return self._CO2_ID

    @CO2_ID.setter
    def CO2_ID(self, value):
        if value not in self.components.IDs:
            raise ValueError(
                f'`CO2_ID` must be one of the system components; '
                f'received {value!r}.'
            )
        self._CO2_ID = value

    @property
    def CO2_price(self):
        return self._CO2_price

    @CO2_price.setter
    def CO2_price(self, value):
        self._CO2_price = self._require_nonnegative('CO2_price', value)

    def _run(self):
        self.outs[0].copy_like(self.ins[0])

    @property
    def state(self):
        '''Dynamic state passed from the wastewater inlet to the wastewater outlet.'''
        if self._state is None:
            return None
        return dict(zip(list(self.components.IDs) + ['Q'], self._state))

    def _init_state(self):
        # Dynamic pass-through method follows QSDsan dynamic Pump.
        # The second inlet is LCA-only CO2 supply and is not part of the PM2
        # process state.
        self._state = self._ins_QC[0]
        self._dstate = self._state * 0.

    def _update_state(self):
        self._outs[0].state = self._state

    def _update_dstate(self):
        self._outs[0].dstate = self._dstate

    @property
    def AE(self):
        if self._AE is None:
            self._compile_AE()
        return self._AE

    def _compile_AE(self):
        _state = self._state
        _dstate = self._dstate
        _update_state = self._update_state
        _update_dstate = self._update_dstate
        def yt(t, QC_ins, dQC_ins):
            _state[:] = QC_ins[0]
            _dstate[:] = dQC_ins[0]
            _update_state()
            _update_dstate()
        self._AE = yt

    def _design(self):
        D = self.design_results
        influent = self.ins[0]
        supply_stream = self.ins[1]
        Q = influent.get_total_flow('L/hr')
        C_in = self.influent_CO2
        # mg/L * L/hr * 1e-6 = kg/hr.
        base_makeup = max(self.target_CO2 - C_in, 0.) * Q * 1e-6
        # User-settable allowance for unmodeled CO2 losses.
        supply = base_makeup * (1 + self.excess_fraction)
        supply_stream.empty()
        supply_stream.imass[self.CO2_ID] = supply

        D['Influent CO2 concentration'] = C_in
        D['Target CO2 concentration'] = self.target_CO2
        D['Wastewater flow'] = Q
        D['Base CO2 makeup'] = base_makeup
        D['CO2 supply'] = supply
        D['Excess CO2 fraction'] = self.excess_fraction

    def _calc_replacement_cost(self):
        # Maintenance defaults to 1.6% of the purchased CO2 supply cost.
        return 0.016 * self.add_OPEX.get('CO2 supply', 0.)

    def _calc_maintenance_labor_cost(self):
        # Labor source: Reliant BevCarb bulk CO2 refill basis,
        # https://www.reliantbevcarb.com/products/bulkco2.
        # Assumes 2 hr labor per 340 kg CO2 refill, rounded to 0.015 hr/kg.
        return (
            0.015 * self.design_results['CO2 supply'] * _labor_wage()
        )

    def _cost(self):
        # Convert 2016 USD/metric tonne to 2022 USD/kg.
        price_2022 = (
            self.CO2_price / 1000
            * CEPCI_by_year[2022] / CEPCI_by_year[2016]
        )
        self.add_OPEX['CO2 supply'] = (
            self.design_results['CO2 supply'] * price_2022
        )
        self.add_OPEX['labor'] = self._calc_maintenance_labor_cost()
        self.add_OPEX['maintenance and replacement'] = (
            self._calc_replacement_cost()
        )

class LCAResourceInput(SanUnit):
    '''
    LCA-only resource accounting unit linked to another process unit.

    This unit is for resource inputs, such as natural gas or membrane-cleaning
    chemicals, that should be included in LCA/impact accounting but should not
    enter PM2 process-stream mass balances. The unit reads a resource demand
    from ``linked_unit.design_results[design_key]``, writes that demand to its
    own feed stream, and copies it to a dummy outlet for flowsheet visibility.

    Parameters
    ----------
    ID : str
        Unit ID.
    ins : WasteStream
        Resource feed stream.
    outs : WasteStream
        Dummy outlet stream used only for flowsheet visualization.
    linked_unit : SanUnit
        Process unit that calculates the resource demand.
    design_key : str
        Key in ``linked_unit.design_results`` that stores the resource demand.
    component_ID : str
        Component ID to assign in the resource stream.
    flow_unit : str
        Unit of the resource demand. The default is 'kg/hr'.
    init_with : str
        Stream class used to initialize missing streams.
    '''
    _N_ins = 1
    _N_outs = 1
    _units = {
        'Resource flow': 'kg/hr',
    }

    def __init__(
            self, ID='', ins=None, outs=(), thermo=None, *,
            linked_unit=None, design_key='', component_ID='',
            flow_unit='kg/hr', init_with='WasteStream',
        ):
        SanUnit.__init__(
            self, ID=ID, ins=ins, outs=outs, thermo=thermo,
            init_with=init_with,
        )
        self.linked_unit = linked_unit
        self.design_key = design_key
        self.component_ID = component_ID
        self.flow_unit = flow_unit

    def _run(self):
        self.outs[0].copy_like(self.ins[0])

    @property
    def state(self):
        '''Dynamic placeholder state passed from resource inlet to dummy outlet.'''
        if self._state is None:
            return None
        return dict(zip(list(self.components.IDs) + ['Q'], self._state))

    def _init_state(self):
        # Dynamic pass-through method follows QSDsan dynamic Pump.
        # The stream is LCA-only and does not connect to PM2 units.
        self._state = self._ins_QC[0]
        self._dstate = self._state * 0.

    def _update_state(self):
        self._outs[0].state = self._state

    def _update_dstate(self):
        self._outs[0].dstate = self._dstate

    @property
    def AE(self):
        if self._AE is None:
            self._compile_AE()
        return self._AE

    def _compile_AE(self):
        _state = self._state
        _dstate = self._dstate
        _update_state = self._update_state
        _update_dstate = self._update_dstate
        def yt(t, QC_ins, dQC_ins):
            _state[:] = QC_ins[0]
            _dstate[:] = dQC_ins[0]
            _update_state()
            _update_dstate()
        self._AE = yt

    def _design(self):
        D = self.design_results
        linked_unit = self.linked_unit
        flow = 0.
        if linked_unit is not None and self.design_key:
            flow = linked_unit.design_results.get(self.design_key, 0.)
        self.ins[0].empty()
        if flow:
            self.ins[0].set_flow(flow, self.flow_unit, self.component_ID)
        self.outs[0].copy_like(self.ins[0])
        D['Resource flow'] = flow

    def _cost(self):
        pass


class Ultrafiltration(su.Splitter):
    '''
    Splitter-based ultrafiltration unit with membrane design and optional
    support equipment accounting.

    The material split is inherited from :class:`qsdsan.unit_operations.Splitter`;
    this class only adds design, purchase-cost, power, and chemical OPEX
    accounting.

    Parameters
    ----------
    ID : str
        Unit ID.
    ins : sequence[Stream], optional
        Inlet stream. The inlet flow is used to size the membrane area.
    outs : sequence[Stream], optional
        Outlet streams for the inherited splitter behavior.
    thermo : Thermo, optional
        Thermodynamic property package.
    split : float, array, or dict
        Split fraction passed to :class:`Splitter`; units are dimensionless.
    order : sequence[str], optional
        Component order for split arrays.
    init_with : str
        Stream class used to initialize missing streams.
    F_BM_default : float, optional
        Optional default bare-module factor override; units are dimensionless.
    isdynamic : bool
        Whether to initialize the unit as dynamic.
    R_t : float
        Total membrane resistance in [1/m].
    T : float
        Water temperature used in the viscosity equation in [deg C].
    TMP : float
        Transmembrane pressure in [Pa].
    capacity_factor : float
        Design redundancy factor applied to influent flow; units are
        dimensionless.
    include_tank : bool
        Whether to include UF tank volume and tank purchase cost.
    include_sparging : bool
        Whether to include air sparging flow, blower power, compressor cost,
        diffuser cost, and stainless-steel mass.
    include_chemical_cleaning : bool
        Whether to include citric-acid and sodium-hypochlorite cleaning usage
        and OPEX.
    hrt : float
        Hydraulic retention time used to size UF tank working volume from
        influent flow, ``V_max = hrt * Q`` [hr]. The default is 0.147 hr.
    sizing_flow : float, optional
        Initial flow basis used to initialize UF tank volume before stream flows
        are solved [m3/hr].
    V_wf : float
        Working volume fraction, i.e., working liquid volume divided by total
        tank volume; units are dimensionless.
    vessel_type : str
        BioSTEAM mixing-tank purchase-cost algorithm name.
    vessel_material : str
        Tank material used by the BioSTEAM material factor.
    specific_sparging_air_demand : float
        Sparging air demand per membrane area in [m3/m2/hr].
    blower_T : float
        Air temperature used for blower power estimation in [deg C].
    P_atm : float
        Atmospheric pressure used for blower power estimation in [kPa].
    P_inlet_loss : float
        Blower inlet pressure loss in [kPa].
    P_diffuser_loss : float
        Diffuser pressure loss in [kPa].
    h_submergance : float
        Diffuser submergence depth used for blower pressure in [m].
    blower_efficiency : float
        Blower efficiency; units are dimensionless.
    blower_K : float
        Blower compression constant used by :func:`qsdsan.utils.get_P_blower`;
        units are dimensionless.
    unit_diffuser_flow_rate : float
        Flow capacity of one diffuser in [m3/d]. The default is 61.2 m3/d.
        Source: Aquarius Technologies Quantaer diffused aeration systems.
    diffuser_unit_cost : float
        Diffuser purchase cost per diffuser in [USD/diffuser]. The default is
        445 USD.
    Air compressor carbon steel follows the compressor weight relationship
        [kg] = 16.013 * sparging power [hp] + 75.813.
        Source: https://us.kaeser.com/products-and-solutions/rotary-screw-compressors/3-hp.aspx.
    NaOCl_cleaning_frequency_full : float
        Full-cleaning frequency with 500 mg/L NaOCl in [1/d]. The default is
        0.0328767 1/d.
    NaOCl_cleaning_frequency_maintenance : float
        Maintenance-cleaning frequency with 120 mg/L NaOCl in [1/d]. The
        default is 0.285714 1/d.
    citric_acid_cleaning_frequency_maintenance : float
        Maintenance-cleaning frequency with 2000 mg/L citric acid in [1/d].
        The default is 0.087866927592955 1/d.
    citric_acid_unit_price : float
        Citric-acid price in [USD/kg]. The default is 1.8595038517821776
        USD/kg.
    sodium_hypochlorite_unit_price : float
        Sodium-hypochlorite price in [USD/kg/%_frac]. The default is
        14.265088213315847 USD/kg/%_frac.
    Labor wage is read from ``pm2_ecorecover_lca.price_dct['Labor']`` [USD/hr].

    References
    ----------
    [1] Li, Y.; et al. Life Cycle Assessment and Techno-Economic Analysis of
        Membrane-Based Wastewater Resource Recovery Systems. Industrial &
        Engineering Chemistry Research. Membrane bare-module factor basis.
        https://doi.org/10.1021/acs.iecr.2c00598.

    [2] Guo, T.; et al. Low-Cost Ceramic Membrane Development for
        Wastewater-Based Resource Recovery. Environmental Science &
        Technology. Membrane resistance and UF tank-volume basis.
        https://doi.org/10.1021/acs.est.3c10264.

    [3] Kovalus Separation Solutions. PURON Hollow Fiber Modules Product
        Information. Membrane-module flux/design reference.
        https://www.kovalus.com/wp-content/uploads/2020/10/puron-hf-modules.pdf.

    [4] Wang, Z.; et al. Techno-Economic and Environmental Assessment of
        Membrane-Based Algae Harvesting. Science of the Total Environment.
        Sparging demand and chemical-cleaning assumptions.
        https://doi.org/10.1016/j.scitotenv.2024.177273.

    [5] Panagopoulos, A. Process Simulation and Techno-Economic Assessment of
        Seawater Reverse Osmosis Desalination. Desalination. Water viscosity
        relationship.
        https://doi.org/10.1016/j.desal.2021.115409.

    [6] BioSTEAM Development Group. BioSTEAM Tank and MixTank Cost
        Algorithms.
        https://biosteam.readthedocs.io/en/latest/_modules/biosteam/units/tank.html.

    [7] BioSTEAM Development Group. BioSTEAM IsothermalCompressor Screw
        Compressor Cost Algorithm.
        https://biosteam.readthedocs.io/en/latest/_modules/biosteam/units/compressor.html#IsothermalCompressor.

    [8] Aquarius Technologies. Quantaer Diffused Aeration Systems.
        Diffuser flow capacity used as the default unit diffuser flow rate.
        https://www.aquariustechnologies.com/wp-content/uploads/2019/07/Quantaer-Diffused-Aeration-Systems-Web.pdf.

    [9] KAESER Compressors. Rotary Screw Compressors, 3 hp Product
        Specification. Compressor weight correlation basis.
        https://us.kaeser.com/products-and-solutions/rotary-screw-compressors/3-hp.aspx.

    [10] Li, Y.; et al. Environmental Impacts of Membrane Replacement in
         Wastewater Treatment. Journal of Cleaner Production. Membrane
         lifetime assumption.
         https://doi.org/10.1016/j.jclepro.2019.01.321.

    [11] Gebreslassie, B. H.; Waymire, R.; You, F. Sustainable Design and
         Synthesis of Algae-Based Biorefinery for Simultaneous Hydrocarbon
         Biofuel Production and Carbon Sequestration.
         AIChE J. 2013, 59 (5), 1599-1621.
         https://doi.org/10.1002/aic.13952.

    [12] NEIWPCC. Northeast Guide for Estimating Staffing at Publicly and
         Privately Owned Wastewater Treatment Plants.
         https://neiwpcc.org/wp-content/uploads/2020/08/NEIWPCC-Northeast-Staffing-Guide.pdf.
    '''
    _N_ins = 1
    _F_BM_default = {
        'Membrane': 3.2, #source: http://doi.org/10.1021/acs.iecr.2c00598
        'Tank': 2.3,
        'Air compressor': 2.15,
        'Diffusers': 1.,
    }
    _units = {
        'Influent flow': 'm3/d',
        'Designed flow': 'm3/d',
        'Tank sizing flow': 'm3/hr',
        'Hydraulic retention time': 'hr',
        'Water viscosity': 'Pa*s',
        'Membrane flux': 'm3/m2/s',
        'Membrane area': 'm2',
        'Membrane module area': 'm2',
        'Tank volume': 'm3',
        'Sparging air flow': 'm3/hr',
        'Sparging air flow at compressor': 'cfm',
        'Number of air compressors': '',
        'Sparging power': 'kW',
        'Number of diffusers': '',
        'Air compressor carbon steel': 'kg',
        'Diffuser area': 'm2',
        'Diffuser stainless steel': 'kg',
        'NaOCl full cleaning frequency': '1/d',
        'NaOCl maintenance cleaning frequency': '1/d',
        'Citric acid maintenance cleaning frequency': '1/d',
        'NaOCl full cleaning usage': 'kg/hr',
        'NaOCl maintenance cleaning usage': 'kg/hr',
        'Citric acid usage': 'kg/hr',
        'Sodium hypochlorite usage': 'kg/hr',
    }
    purchase_cost_algorithms = mix_tank_purchase_cost_algorithms

    def __init__(
            self, ID='', ins=None, outs=(), thermo=None, *, split,
            order=None, init_with='WasteStream', F_BM_default=None,
            isdynamic=False, R_t=2.12e12, #steady-state flux based on https://doi.org/10.1021/acs.est.3c10264; https://www.kovalus.com/wp-content/uploads/2020/10/puron-hf-modules.pdf
            T=25., TMP=15.4e3,
            capacity_factor=1.5, include_tank=False,
            include_sparging=False, include_chemical_cleaning=False,
            hrt=0.377,
            sizing_flow=None,
            V_wf=0.8, vessel_type='Conventional',
            vessel_material='Stainless steel',
            specific_sparging_air_demand=0.3, blower_T=20,
            P_atm=101.325, P_inlet_loss=1, P_diffuser_loss=7,
            h_submergance=5.18, blower_efficiency=0.7,
            blower_K=0.283, unit_diffuser_flow_rate=61.2,
            diffuser_unit_cost=445,
            NaOCl_cleaning_frequency_full=0.03287671232876712,
            NaOCl_cleaning_frequency_maintenance=0.2857142857142857,
            citric_acid_cleaning_frequency_maintenance=0.08786692759295499,
            citric_acid_unit_price=1.8595038517821776,
            sodium_hypochlorite_unit_price=14.265088213315847,
            membrane_lifetime = 5,
        ):
        lca_ins = ()
        process_ins = ins
        if isinstance(ins, (list, tuple)) and len(ins) > 1:
            process_ins = ins[:1]
            lca_ins = tuple(ins[1:])
        self._N_lca_ins = len(lca_ins)
        self._lca_ins = lca_ins
        su.Splitter.__init__(
            self, ID=ID, ins=process_ins, outs=outs, thermo=thermo, split=split,
            order=order, init_with=init_with, F_BM_default=F_BM_default,
            isdynamic=isdynamic,
        )
        for n, stream in enumerate(lca_ins, start=1):
            old = self.ins[n]
            old._sink = None
            self.ins._streams[n] = stream
            if stream._source is None:
                stream._source = self
            stream._sink = None
        self.R_t = R_t
        self.T = T
        self.TMP = TMP
        self.capacity_factor = capacity_factor
        self.include_tank = bool(include_tank)
        self.include_sparging = bool(include_sparging)
        self.include_chemical_cleaning = bool(include_chemical_cleaning)

        self.hrt = hrt
        self.sizing_flow = sizing_flow
        self.V_max = _initial_volume_from_hrt(
            hrt, sizing_flow=sizing_flow,
        )
        self.V_wf = V_wf
        self.vessel_type = vessel_type
        self.vessel_material = vessel_material
        self.specific_sparging_air_demand = specific_sparging_air_demand
        self.blower_T = blower_T
        self.P_atm = P_atm
        self.P_inlet_loss = P_inlet_loss
        self.P_diffuser_loss = P_diffuser_loss
        self.h_submergance = h_submergance
        self.blower_efficiency = blower_efficiency
        self.blower_K = blower_K
        self.unit_diffuser_flow_rate = unit_diffuser_flow_rate
        self.diffuser_unit_cost = diffuser_unit_cost
        self.NaOCl_cleaning_frequency_full = NaOCl_cleaning_frequency_full
        self.NaOCl_cleaning_frequency_maintenance = NaOCl_cleaning_frequency_maintenance
        self.citric_acid_cleaning_frequency_maintenance = citric_acid_cleaning_frequency_maintenance
        self.citric_acid_unit_price = citric_acid_unit_price
        self.sodium_hypochlorite_unit_price = sodium_hypochlorite_unit_price
        self.membrane_lifetime = membrane_lifetime

    @staticmethod
    def _require_positive(name, value):
        if value <= 0:
            raise ValueError(f'`{name}` must be positive.')
        return value

    @staticmethod
    def _require_nonnegative(name, value):
        if value < 0:
            raise ValueError(f'`{name}` must be non-negative.')
        return value

    @property
    def R_t(self):
        return self._R_t

    @R_t.setter
    def R_t(self, value):
        self._R_t = self._require_positive('R_t', value)

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if value <= -42.5:
            raise ValueError('`T` must be greater than -42.5 deg C.')
        self._T = value

    @property
    def TMP(self):
        return self._TMP

    @TMP.setter
    def TMP(self, value):
        self._TMP = self._require_positive('TMP', value)

    @property
    def capacity_factor(self):
        return self._capacity_factor

    @capacity_factor.setter
    def capacity_factor(self, value):
        self._capacity_factor = self._require_positive(
            'capacity_factor', value,
        )

    @property
    def hrt(self):
        return self._hrt

    @hrt.setter
    def hrt(self, value):
        self._hrt = self._require_positive('hrt', value)

    @property
    def V_max(self):
        return self._V_max

    @V_max.setter
    def V_max(self, value):
        self._V_max = self._require_positive('V_max', value)

    @property
    def sizing_flow(self):
        return self._sizing_flow

    @sizing_flow.setter
    def sizing_flow(self, value):
        self._sizing_flow = (
            None if value is None
            else self._require_nonnegative('sizing_flow', value)
        )

    @property
    def V_wf(self):
        return self._V_wf

    @V_wf.setter
    def V_wf(self, value):
        value = self._require_positive('V_wf', value)
        if value > 1:
            raise ValueError('`V_wf` cannot exceed 1.')
        self._V_wf = value

    @property
    def vessel_type(self):
        return self._vessel_type

    @vessel_type.setter
    def vessel_type(self, value):
        try:
            algorithm = self.purchase_cost_algorithms[value]
        except KeyError:
            valid = ', '.join(self.purchase_cost_algorithms)
            raise ValueError(
                f'`vessel_type` must be one of: {valid}.'
            ) from None
        self._vessel_type = value
        self.purchase_cost_algorithm = algorithm

    @property
    def vessel_material(self):
        return self._vessel_material

    @vessel_material.setter
    def vessel_material(self, value):
        try:
            factor = vessel_material_factors[value]
        except KeyError:
            valid = ', '.join(vessel_material_factors)
            raise ValueError(
                f'No vessel material factor is available for {value!r}; '
                f'choose one of: {valid}.'
            ) from None
        self._vessel_material = value
        self.F_M['Tank'] = factor

    @property
    def specific_sparging_air_demand(self):
        return self._specific_sparging_air_demand

    @specific_sparging_air_demand.setter
    def specific_sparging_air_demand(self, value):
        self._specific_sparging_air_demand = self._require_nonnegative(
            'specific_sparging_air_demand', value,
        )

    @property
    def blower_efficiency(self):
        return self._blower_efficiency

    @blower_efficiency.setter
    def blower_efficiency(self, value):
        value = self._require_positive('blower_efficiency', value)
        if value > 1:
            raise ValueError('`blower_efficiency` cannot exceed 1.')
        self._blower_efficiency = value

    @property
    def diffuser_unit_cost(self):
        return self._diffuser_unit_cost

    @diffuser_unit_cost.setter
    def diffuser_unit_cost(self, value):
        self._diffuser_unit_cost = self._require_nonnegative(
            'diffuser_unit_cost', value,
        )

    @property
    def unit_diffuser_flow_rate(self):
        return self._unit_diffuser_flow_rate

    @unit_diffuser_flow_rate.setter
    def unit_diffuser_flow_rate(self, value):
        self._unit_diffuser_flow_rate = self._require_positive(
            'unit_diffuser_flow_rate', value,
        )

    @property
    def NaOCl_cleaning_frequency_full(self):
        return self._NaOCl_cleaning_frequency_full

    @NaOCl_cleaning_frequency_full.setter
    def NaOCl_cleaning_frequency_full(self, value):
        self._NaOCl_cleaning_frequency_full = self._require_nonnegative(
            'NaOCl_cleaning_frequency_full', value,
        )

    @property
    def NaOCl_cleaning_frequency_maintenance(self):
        return self._NaOCl_cleaning_frequency_maintenance

    @NaOCl_cleaning_frequency_maintenance.setter
    def NaOCl_cleaning_frequency_maintenance(self, value):
        self._NaOCl_cleaning_frequency_maintenance = self._require_nonnegative(
            'NaOCl_cleaning_frequency_maintenance', value,
        )

    @property
    def citric_acid_cleaning_frequency_maintenance(self):
        return self._citric_acid_cleaning_frequency_maintenance

    @citric_acid_cleaning_frequency_maintenance.setter
    def citric_acid_cleaning_frequency_maintenance(self, value):
        self._citric_acid_cleaning_frequency_maintenance = self._require_nonnegative(
            'citric_acid_cleaning_frequency_maintenance', value,
        )

    @property
    def citric_acid_unit_price(self):
        return self._citric_acid_unit_price

    @citric_acid_unit_price.setter
    def citric_acid_unit_price(self, value):
        self._citric_acid_unit_price = self._require_nonnegative(
            'citric_acid_unit_price', value,
        )

    @property
    def sodium_hypochlorite_unit_price(self):
        return self._sodium_hypochlorite_unit_price

    @sodium_hypochlorite_unit_price.setter
    def sodium_hypochlorite_unit_price(self, value):
        self._sodium_hypochlorite_unit_price = self._require_nonnegative(
            'sodium_hypochlorite_unit_price', value,
        )

    def _get_membrane_design(self):
        Q = self.outs[0].get_total_flow('m3/d')
        Q_design = Q * self.capacity_factor
        # Source: https://doi.org/10.1016/j.scitotenv.2024.177273.
        mu = 497e-3 / (self.T + 42.5)**1.5 #relative viscoity, unit Pa x s, Source: https://doi.org/10.1016/j.desal.2021.115409
        J = self.TMP / mu / self.R_t # membrane flux [m3/m2/s]
        A = Q_design / 24 / 3600 / J if Q_design else 0. #m2
        return Q, Q_design, mu, J, A

    def _get_sparging_power(self, Q_air):
        if not self.include_sparging:
            return 0.

        # Source: QSDsan wwt_design.get_P_blower; Q_air m3/min.
        return get_P_blower(
            Q_air / 60,
            T=self.blower_T,
            P_atm=self.P_atm,
            P_inlet_loss=self.P_inlet_loss,
            P_diffuser_loss=self.P_diffuser_loss,
            h_submergance=self.h_submergance,
            efficiency=self.blower_efficiency,
            K=self.blower_K,
        )

    def _chemical_usage(self, concentration, frequency):#kg/hr
        return (
            self.V_max * 1000 * concentration * 1e-6
            * frequency / 24
        )

    def _calc_replacement_cost(self):
        C = self.baseline_purchase_costs
        annual = 0.
        # Membrane lifetime source:
        # https://doi.org/10.1016/j.jclepro.2019.01.321.
        annual += C.get('Membrane', 0.) / self.membrane_lifetime
        if self.include_tank:
            # Tank maintenance source:
            # https://doi.org/10.1016/j.bej.2017.08.006.
            annual += 0.06 * _installed_capital(
                self, ('Tank',), include_parallel=False,
            )
        if self.include_sparging:
            # Sparging equipment maintenance source:
            # https://doi.org/10.1016/j.bej.2017.08.006.
            annual += 0.06 * _installed_capital(
                self, ('Air compressor', 'Diffusers'),
                include_parallel=False,
            )
        return _annual_to_hourly_cost(annual)

    def _calc_maintenance_labor_cost(self):
        D = self.design_results
        Q = D.get('Influent flow', self.ins[0].get_total_flow('m3/d'))
        _warn_if_flow_outside_NEIWPCC(self, Q, 'ultrafiltration')
        # Membrane labor source: NEIWPCC Northeast Staffing Guide,
        # https://neiwpcc.org/wp-content/uploads/2020/08/NEIWPCC-Northeast-Staffing-Guide.pdf.
        hours = 5.524 * Q**0.37 if Q > 0 else 0.
        if self.include_sparging:
            hours += 73 * D.get('Number of air compressors', 0)
        if self.include_chemical_cleaning:
            hours += 365 * (
                self.NaOCl_cleaning_frequency_full
                + self.NaOCl_cleaning_frequency_maintenance
                + self.citric_acid_cleaning_frequency_maintenance
            )
        return _annual_to_hourly_cost(hours * _labor_wage())

    def _init_lca(self): 
        self.include_construction = True
        self.construction = [
            Construction('ultrafiltration', linked_unit=self, 
                            item='UltrafiltrationModule', 
                            quantity_unit='ea'),
            Construction('carbon_steel', linked_unit=self,
                            item='LowCarbonSteel',
                            quantity_unit='kg'),
            Construction('stainless_steel', linked_unit=self,
                            item='StainlessSteel',
                            quantity_unit='kg'),
            ]

    def _design(self):
        D = self.design_results
        Q, Q_design, mu, J, A = self._get_membrane_design()
        Q_tank = self.sizing_flow
        if Q_tank > 0:
            self.V_max = self.hrt * Q_tank

        D['Influent flow'] = Q
        D['Designed flow'] = Q_design
        D['Tank sizing flow'] = Q_tank
        D['Hydraulic retention time'] = self.hrt
        D['Water viscosity'] = mu
        D['Membrane flux'] = J
        D['Membrane area'] = A
        D['Membrane module area'] = A 
        self.construction[0].quantity = A/50 #Ecoinvent ultrafiltration unit is made of 50 m2 membrane area
        D['Tank volume'] = self.V_max if self.include_tank else 0.
        D['Sparging air flow'] = 0.
        D['Sparging air flow at compressor'] = 0.
        D['Number of air compressors'] = 0
        D['Sparging power'] = 0.
        D['Number of diffusers'] = 0
        D['Air compressor carbon steel'] = 0.
        D['Diffuser area'] = 0.
        D['Diffuser stainless steel'] = 0.
        self.construction[1].quantity = 0.
        self.construction[2].quantity = 0.
        D['NaOCl full cleaning frequency'] = 0.
        D['NaOCl maintenance cleaning frequency'] = 0.
        D['Citric acid maintenance cleaning frequency'] = 0.
        D['NaOCl full cleaning usage'] = 0.
        D['NaOCl maintenance cleaning usage'] = 0.
        D['Citric acid usage'] = 0.
        D['Sodium hypochlorite usage'] = 0.

        if self.include_sparging:
            # Source: specific demand from https://doi.org/10.1016/j.scitotenv.2024.177273.
            Q_air = self.specific_sparging_air_demand * A
            power = self._get_sparging_power(Q_air)
            algorithm = IsothermalCompressor.baseline_cost_algorithms['Screw']
            Q_air_acfm = auom('m3/hr').convert(Q_air, 'cfm')
            max_acfm = algorithm.acfm_bounds[1]
            N_compressors = (
                ceil(Q_air_acfm / max_acfm) if Q_air_acfm > 0 else 0
            )
            N_diffusers = (
                ceil(Q_air * 24 / self.unit_diffuser_flow_rate)
                if Q_air > 0 else 0
            )

            # Source: compressor sizing and diffuser mass method used in Tank.
            D['Sparging air flow'] = Q_air
            D['Sparging air flow at compressor'] = Q_air_acfm
            D['Number of air compressors'] = N_compressors
            D['Sparging power'] = power
            D['Number of diffusers'] = N_diffusers
            sparging_power_hp = auom('kW').convert(power, 'hp')
            D['Air compressor carbon steel'] = self.construction[1].quantity = (
                16.013 * sparging_power_hp + 75.813
                if power > 0 else 0.
            )
            diffuser_area = (
                D['Tank volume'] / self.h_submergance
                if self.include_tank and self.h_submergance > 0 else 0.
            )
            D['Diffuser area'] = diffuser_area
            D['Diffuser stainless steel'] = diffuser_area * 181 / 18.5
            self.construction[2].quantity = D['Diffuser stainless steel']

        if self.include_chemical_cleaning:
            # Source: cleaning frequency and concentrations from
            # Unit_process_assumptions_ecorecover_081426.xlsx, UF tab.
            D['NaOCl full cleaning frequency'] = self.NaOCl_cleaning_frequency_full
            D['NaOCl maintenance cleaning frequency'] = self.NaOCl_cleaning_frequency_maintenance
            D['Citric acid maintenance cleaning frequency'] = (
                self.citric_acid_cleaning_frequency_maintenance
            )
            D['Citric acid usage'] = self._chemical_usage(
                2000., self.citric_acid_cleaning_frequency_maintenance,
            ) #kg/hr
            D['NaOCl full cleaning usage'] = self._chemical_usage(
                500., self.NaOCl_cleaning_frequency_full,
            )
            D['NaOCl maintenance cleaning usage'] = self._chemical_usage(
                120., self.NaOCl_cleaning_frequency_maintenance,
            )
            D['Sodium hypochlorite usage'] = (
                D['NaOCl full cleaning usage']
                + D['NaOCl maintenance cleaning usage']
            )

    def _cost(self):
        D = self.design_results
        C = self.baseline_purchase_costs
        A = D['Membrane area']

        if A > 0:
            # Source: https://doi.org/10.1016/j.scitotenv.2024.177273.
            membrane_unit_cost = (
                -2.985 * np.log(D['Membrane module area']) + 77.759
            )
            C['Membrane'] = membrane_unit_cost * euro_to_usd * A
        else:
            C['Membrane'] = 0.

        if self.include_tank:
            # Source: BioSTEAM MixTank purchase-cost method used in Tank.
            N, Cp = compute_number_of_tanks_and_purchase_cost(
                self.V_max / self.V_wf, self.purchase_cost_algorithm,
            )
            default_material = self.purchase_cost_algorithm.material
            C['Tank'] = (
                N * Cp / vessel_material_factors.get(default_material, 1.)
            )
        else:
            C.pop('Tank', None)

        if self.include_sparging:
            N_compressors = D['Number of air compressors']
            sparging_power = D['Sparging power']
            if N_compressors and sparging_power > 0:
                # Source: BioSTEAM IsothermalCompressor screw-compressor
                # baseline algorithm, matching Tank.
                algorithm = (
                    IsothermalCompressor.baseline_cost_algorithms['Screw']
                )
                total_hp = auom('kW').convert(sparging_power, 'hp')
                hp_per_compressor = total_hp / N_compressors
                compressor_cost = (
                    N_compressors * CEPCI_by_year[2022] / algorithm.CE
                    * algorithm.cost(hp_per_compressor)
                )
            else:
                compressor_cost = 0.
            C['Air compressor'] = compressor_cost
            C['Diffusers'] = D['Number of diffusers'] * self.diffuser_unit_cost
            self.F_M['Air compressor'] = 2.5
        else:
            C.pop('Air compressor', None)
            C.pop('Diffusers', None)

        # Chemical purchase costs are accounted through priced LCA-only
        # resource streams, not through Ultrafiltration add_OPEX.
        self.add_OPEX.pop('Citric acid', None)
        self.add_OPEX.pop('Sodium hypochlorite', None)

        self.power_utility.rate = D['Sparging power']
        self.add_OPEX['labor'] = self._calc_maintenance_labor_cost()
        self.add_OPEX['maintenance and replacement'] = (
            self._calc_replacement_cost()
        )


class AlgaeCentrifuge(su.Splitter):
    '''
    Splitter-based algae centrifuge with design, purchase-cost, and energy
    accounting.

    The material split is inherited from :class:`qsdsan.unit_operations.Splitter`;
    this class only adds centrifuge design and cost accounting.

    Parameters
    ----------
    ID : str
        Unit ID.
    ins : sequence[Stream], optional
        Inlet stream. The inlet volumetric flow is used for design and cost.
    outs : sequence[Stream], optional
        Outlet streams for the inherited splitter behavior.
    thermo : Thermo, optional
        Thermodynamic property package.
    split : float, array, or dict
        Split fraction passed to :class:`Splitter`; units are dimensionless.
    order : sequence[str], optional
        Component order for split arrays.
    init_with : str
        Stream class used to initialize missing streams.
    F_BM_default : float, optional
        Optional default bare-module factor override; units are dimensionless.
    isdynamic : bool
        Whether to initialize the unit as dynamic.
    capacity_factor : float
        Design capacity factor applied to inlet flow for centrifuge sizing,
        energy intensity, weight, and purchase cost. The default is 4.
    algal_cell_diameter : float
        Algal particle diameter in [m].
    algal_particle_density : float
        Algal particle density in [kg/m3].
    water_density : float
        Water density in [kg/m3].
    T : float
        Water temperature used in the viscosity equation in [deg C].
    phi : float
        Solids volume fraction for hindered-settling correction; units are
        dimensionless.
    vgm : float
        Master-curve settling velocity in [m/s].
    UF_flow : float, optional
        Influent flow to the upstream ultrafiltration unit in [m3/d], used for
        centrifuge labor estimation.
    Labor wage is read from ``pm2_ecorecover_lca.price_dct['Labor']`` [USD/hr].

    References
    ----------
    [1] Cost follows BioSTEAM ``LiquidsCentrifuge``:
    ``C = 28100 * Q**0.574`` with ``Q`` in [m3/hr], CEPCI 525.4,
    upper-bound flow 100 m3/hr, and bare-module factor 2.03.

    [2] Energy equations follow Najjar and Abu-Shamleh, Algal Research 51
    (2020) 102046, http://doi.org/10.1016/j.algal.2020.102046.

    [3] Stainless-steel weight follows the Dolphin Centrifuge capacity
    correlation:
    ``weight = 1126.1*ln(Q) - 1204.8`` with ``Q`` in [m3/hr].
    '''
    _F_BM_default = {'Centrifuge': 2.03}
    _units = {
        'Inlet flow': 'm3/hr',
        'Capacity factor': '',
        'Influent flow': 'm3/hr',
        'Water viscosity': 'Pa*s',
        'Gravity settling velocity': 'm/s',
        'Effective settling velocity': 'm/s',
        'Master-curve flow': 'm3/hr',
        'Disc centrifuge energy intensity': 'kWh/m3',
        'Centrifuge stainless steel': 'kg',
        'Number of centrifuges': '',
    }

    def __init__(
            self, ID='', ins=None, outs=(), thermo=None, *, split,
            order=None, init_with='WasteStream', F_BM_default=None,
            isdynamic=False, capacity_factor=4., algal_cell_diameter=5e-6,
            algal_particle_density=1050., water_density=1000.,
            T=25., phi=0., vgm=0.1e-6, UF_flow=None,
        ):
        su.Splitter.__init__(
            self, ID=ID, ins=ins, outs=outs, thermo=thermo, split=split,
            order=order, init_with=init_with, F_BM_default=F_BM_default,
            isdynamic=isdynamic,
        )
        self.water_density = water_density
        self.algal_cell_diameter = algal_cell_diameter
        self.algal_particle_density = algal_particle_density
        self.capacity_factor = capacity_factor
        self.T = T
        self.phi = phi
        self.vgm = vgm
        self.UF_flow = UF_flow

    @staticmethod
    def _require_positive(name, value):
        if value <= 0:
            raise ValueError(f'`{name}` must be positive.')
        return value

    @staticmethod
    def _require_nonnegative(name, value):
        if value < 0:
            raise ValueError(f'`{name}` must be non-negative.')
        return value

    @property
    def capacity_factor(self):
        return self._capacity_factor

    @capacity_factor.setter
    def capacity_factor(self, value):
        self._capacity_factor = self._require_positive(
            'capacity_factor', value,
        )

    @property
    def algal_cell_diameter(self):
        return self._algal_cell_diameter

    @algal_cell_diameter.setter
    def algal_cell_diameter(self, value):
        self._algal_cell_diameter = self._require_positive(
            'algal_cell_diameter', value,
        )

    @property
    def water_density(self):
        return self._water_density

    @water_density.setter
    def water_density(self, value):
        self._water_density = self._require_positive('water_density', value)

    @property
    def algal_particle_density(self):
        return self._algal_particle_density

    @algal_particle_density.setter
    def algal_particle_density(self, value):
        if value <= self.water_density:
            raise ValueError(
                '`algal_particle_density` must be greater than '
                '`water_density`.',
            )
        self._algal_particle_density = value

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if value <= -42.5:
            raise ValueError('`T` must be greater than -42.5 deg C.')
        self._T = value

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        if not 0 <= value < 1:
            raise ValueError(
                '`phi` must be greater than or equal to 0 and less than 1.'
            )
        self._phi = value

    @property
    def vgm(self):
        return self._vgm

    @vgm.setter
    def vgm(self, value):
        self._vgm = self._require_positive('vgm', value)

    @property
    def UF_flow(self):
        return self._UF_flow

    @UF_flow.setter
    def UF_flow(self, value):
        self._UF_flow = (
            None if value is None
            else self._require_nonnegative('UF_flow', value)
        )

    def _get_mu(self):
        # Source: same viscosity relationship used in Ultrafiltration.
        return 497e-3 / (self.T + 42.5)**1.5

    def _get_gravity_settling_velocity(self, mu):
        # Source: Eq. 7 in Najjar and Abu-Shamleh (2020),
        # http://doi.org/10.1016/j.algal.2020.102046.
        return (
            (self.algal_particle_density - self.water_density)
            * self.algal_cell_diameter**2 * 9.8 / (18 * mu)
        )

    def _get_labor_flow(self):
        if self.UF_flow is not None:
            return self.UF_flow
        return self.ins[0].get_total_flow('m3/d')

    def _calc_replacement_cost(self):
        # Centrifuge maintenance source:
        # https://doi.org/10.1016/j.algal.2017.11.038.
        return _annual_to_hourly_cost(0.05 * _installed_capital(self))

    def _calc_maintenance_labor_cost(self):
        Q = self._get_labor_flow()
        _warn_if_flow_outside_NEIWPCC(self, Q, 'centrifuge')
        # Labor source: NEIWPCC Northeast Staffing Guide,
        # https://neiwpcc.org/wp-content/uploads/2020/08/NEIWPCC-Northeast-Staffing-Guide.pdf.
        hours_per_unit = 0.0018 * Q + 45.028 if Q > 0 else 0.
        N = self.design_results.get('Number of centrifuges', 0)
        return _annual_to_hourly_cost(
            hours_per_unit * N * _labor_wage()
        )

    def _init_lca(self):
        self.include_construction = True
        self.construction = [
            Construction('stainless_steel', linked_unit=self,
                         item='StainlessSteel',
                         quantity_unit='kg'),
            ]

    def _design(self):
        D = self.design_results
        Q_in_hr = self.ins[0].get_total_flow('m3/hr')
        Q_hr = Q_in_hr * self.capacity_factor
        mu = self._get_mu()
        vg = self._get_gravity_settling_velocity(mu)
        vg_eff = vg * (1 - self.phi)**4.65

        if Q_hr > 0:
            # Source: Eq. 15 in Najjar and Abu-Shamleh (2020),
            # converts actual flow to master-curve flow [m3/hr].
            Q_s = Q_hr / 3600
            Qm = Q_s * 3600 * self.vgm / vg_eff
            # Source: Eq. 16 in Najjar and Abu-Shamleh (2020),
            # disc centrifuge energy intensity [kWh/m3].
            E_disc = 1.447 * Qm**(-0.304)
            # Source: Dolphin Centrifuge capacity correlation;
            # clamp to zero outside the low-flow correlation range.
            weight = max(1126.1 * np.log(Q_hr) - 1204.8, 0.)
        else:
            Qm = E_disc = weight = 0.

        D['Inlet flow'] = Q_in_hr
        D['Capacity factor'] = self.capacity_factor
        D['Influent flow'] = Q_hr
        D['Water viscosity'] = mu
        D['Gravity settling velocity'] = vg
        D['Effective settling velocity'] = vg_eff
        D['Master-curve flow'] = Qm
        D['Disc centrifuge energy intensity'] = E_disc
        D['Centrifuge stainless steel'] = weight
        if self.include_construction:
            self.construction[0].quantity = weight
        D['Number of centrifuges'] = ceil(Q_hr / 100) if Q_hr > 0 else 0

    def _cost(self):
        D = self.design_results
        C = self.baseline_purchase_costs
        Q_hr = D['Influent flow']
        N = D['Number of centrifuges']
        if Q_hr > 0:
            Q_each = Q_hr / N
            # Source: BioSTEAM LiquidsCentrifuge cost correlation.
            C['Centrifuge'] = (
                N * CEPCI_by_year[2022] / 525.4
                * 28100 * Q_each**0.574
            )
        else:
            C['Centrifuge'] = 0.
        self.power_utility.rate = (
            D['Disc centrifuge energy intensity'] * Q_hr
        )
        self.add_OPEX['labor'] = self._calc_maintenance_labor_cost()
        self.add_OPEX['maintenance and replacement'] = (
            self._calc_replacement_cost()
        )

@cost(basis = 'Working volume', ID='Glass tube & fittings', units='m3',
      cost = 215897.8, S=74.8, CE=CEPCI_by_year[2022], n=1, BM=1) #ref: Roberts Project Cost Summary,Glass + 20% cost from pumps, piping, and pigging, linear scale up to volume
@cost(basis='Aerial footage', ID='Greenhouse',units='m2',cost= 203840, S=504, CE=CEPCI_by_year[2022], n=0.6, BM=1) #ref: Roberts Project Cost Summary, use 0.6 as exponential factor
@cost(basis='Working volume', ID='Support structure', units='m3',
      cost=178575, S=74.8, CE=CEPCI_by_year[2022], n=1, BM=1) #ref: Roberts Project Cost Summary
@cost(basis='Aerial footage', ID= 'LED system', units = 'm2', cost= 43050, S=504, CE=CEPCI_by_year[2022],n=1, BM=1) #reference: Clearas quote from vendor
class Photobioreactor(CSTR):
    '''
    Helical fence-type tubular photobioreactor.

    Parameters
    ----------
    ID : str
        ID for the reactor.
    ins : :class:`WasteStream`
        Influents to the reactor. Can be an array of up to 3 WasteStream objects by
        default, typically wastewater to be treated, recycled effluent, recycled
        activated sludge.
    outs : :class:`WasteStream`
        Treated effluent.
    split : iterable of float
        Volumetric splits of effluent flows if there are more than one effluent.
        The default is None.
    InD: float
        The designed inner diameter [m] of the tube. The default is 0.1 m.
    OD: float
        The designed outer diameter [m] of the tube. The default is 0.105 m.
    spt: float
        The designed single pass time [hr] for an micralgae cell to travel through one set of photobioreactor. The default is 1.67 hrs.
    hrt: float
        The designed hydraulic retention time [hr] for influent. The default is 4 hrs.
    spacing: float
        The designed average space [m] between each set of photobioreactors. The default is 1.05 m.
    support_spacing: float
        The designed average space [m] between each support structure for glass tubes. The default is 2.0 m.
    length: float
        The designed maximum length [m] for each row of photobioreactor. The default is 80 m.
    v: float
        The designed maximum flow velocity [m/s] in the tube. The default is 0.4 m/s. Note this flow is the combined flow from influent and recycled flow.
    design_flow_stream : :class:`WasteStream`, optional
        Influent stream used to size the PBR working volume [m3/hr].
        If not provided, the immediate PBR process inlet is used.
    sizing_flow : float, optional
        Initial flow basis used to initialize one dynamic PBR CSTR block before
        stream flows are solved [m3/hr].
    n_series : int
        Number of CSTRs used in series to represent the plug-flow PBR. The
        default is 20.
    rho: float
        The density of the mixed liquor [kg/m3]. The default is 1000.
    light_intensity: int
        The designed light intensity [umol/(m2·s)] over the illuminated area. The default is 100.
    annual_heating_days: int
        The average winter days per year that requires heaters, default to 150 days.
    heater_up_time_ratio: float
        The average daily up time ratio [0-1] of heaters, default to 0.15.
    LED_power_per_area_basis : float
        Installed LED power [kW] per ``lighting_area_basis`` [m2]. The default
        is 20 kW per 504 m2, based on two 10 kW power supplies in the Clearas
        lighting-system documentation.
    LED_up_time_ratio : float
        Average LED power-use fraction [0-1]. The default is 0.25 to represent
        supplemental lighting that is not on continuously and not always at
        maximum power.
    lighting_area_basis : float
        Greenhouse aerial-footage basis [m2] for LED and fan scaling. The
        default is 504 m2.
    fan_number_per_area_basis : float
        Number of ventilation fans per ``lighting_area_basis`` [ea]. The
        default is 2 fans per 504 m2 based on the Clearas lighting system.
    fan_power_per_fan : float
        Rated fan power [hp/fan]. The default is 0.75 hp/fan.
    fan_up_time_ratio : float
        Average fan operating fraction [0-1]. The default is 0.7.
    Labor wage is read from ``pm2_ecorecover_lca.price_dct['Labor']`` [USD/hr].
    glass_tube_lifetime : float
        Glass tube lifetime in [yr].
    greenhouse_maintenance_ratio : float
        Annual greenhouse maintenance cost as a fraction of installed capital.
    LED_lifetime : float
        LED system lifetime in [yr].
    support_structure_maintenance_ratio : float
        Annual support-structure maintenance cost as a fraction of installed
        capital.
    ===============================
    aeration : float or :class:`Process`, optional
        Aeration setting. Either specify a targeted dissolved oxygen concentration
        in [mg O2/L] or provide a :class:`Process` object to represent aeration,
        or None for no aeration. The default is 2.0.
    DO_ID : str, optional
        The :class:`Component` ID for dissolved oxygen, only relevant when the
        reactor is aerated. The default is 'S_O2'.
    suspended_growth_model : :class:`Processes`, optional
        The suspended growth biokinetic model. The default is None.
    exogenous_var : iterable[:class:`ExogenousDynamicVariable`], optional
        Any exogenous dynamic variables that affect the process mass balance,
        e.g., temperature, sunlight irradiance. Must be independent of state 
        variables of the suspended_growth_model (if has one).
    
    References
    ----------
     [1] Shoener, B. D.; Zhong, C.; Greiner, A. D.; Khunjar, W. O.; Hong, P.-Y.; Guest, J. S.
         Design of Anaerobic Membrane Bioreactors for the Valorization
         of Dilute Organic Carbon Waste Streams.
         Energy Environ. Sci. 2016, 9 (3), 1102-1112.
         https://doi.org/10.1039/C5EE03715H.

     [2] Roberts Project Cost Summary. Cost basis for glass tube and fittings,
         greenhouse, support structure, and photobioreactor hardware scaling.

     [3] Clearas Water Recovery. Vendor Quote and Documentation. LED-system
         cost basis and LED lifetime assumption.

     [4] Wang, C.; Lan, C. Q. Effects of Shear Stress on Microalgae -
         A Review. Biotechnology Advances 2018, 36 (4), 986-1002.
         Velocity/design range for photobioreactor operation.
         https://doi.org/10.1016/j.biotechadv.2018.03.001.

     [5] IMETRA. Borosilicate Glass Material Properties. Borosilicate glass
         density basis.
         https://www.imetra.com/borosilicate-glass-material-properties/.

     [6] California LightWorks. MegaDrive Linear 400 Product Information.
         Aluminum mass basis for LED fixtures.
         https://californialightworks.com/megadrive/megadrive-linear-400/.

     [7] Algae Foundation ATEC. Techno-Economic Analysis of Microalgae
         Production. Glass tube lifetime and Table 8 labor scaling basis.
         https://www.algaefoundationatec.org/aces/download/Techno-Economic%20Analysis.pdf.

     [8] Mississippi State University Extension. Greenhouse Tomato Handbook:
         Greenhouse Costs and Maintenance. Annual greenhouse and support
         maintenance-ratio basis.
         https://extension.msstate.edu/sites/default/files/publications/P2766.pdf.
    
    '''
    _units = {
        **CSTR._units,
        'Total flow': 'm3/hr',
        'Working volume': 'm3',
        'CSTR block working volume': 'm3',
        'Total length': 'm',
        'sets of PBR blocks': 'ea',
        'number of rows per set': 'ea',
        'Aerial footage-volume-to-area ratio': 'm3/m2',
        'Aerial footage': 'm2',
        'volume-to-area ratio': 'm3/m2',
        'Total glass weight': 'kg',
        'U-bends': 'ea',
        'Plates': 'ea',
        'anchor bolts': 'ea',
        'Plate mounting bolts': 'ea',
        'nuts': 'ea',
        'washers': 'ea',
        'Coupling': 'ea',
        'Hose clamps': 'ea',
        'Support structure': 'ea',
        'Greenhouse panel area': 'm2',
        'Number of Fans': 'ea',
        '250K BTU heater': 'ea',
        'natural gas': 'kg/hr',
        'LED light': 'ea',
        'LED electricity': 'kW',
        'Fan electricity': 'kW',
        'Total electricity': 'kW',
        'LED mounting structure': 'kg',
        'PIG assemblies PVC': 'kg',
        'PIG assemblies stainless steel': 'kg',
    }

    def __init__(self, ID='', ins=None, outs=(), thermo=None, init_with='WasteStream',
                split=None, W_tank=6.4, D_tank=3.65,
                freeboard=0.61, t_wall=None, t_slab=None, aeration=None, 
                DO_ID='S_O2', suspended_growth_model=None, 
                gas_stripping=False, gas_IDs=None, stripping_kLa_min=None, 
                K_Henry=None, D_gas=None, p_gas_atm=None,
                isdynamic=True, exogenous_vars=(), 
                InD=0.1, 
                OD =0.105,
                spt=1.67,
                hrt=4,
                spacing=1.05,
                support_spacing=2,
                length = 80,
                v= 0.8, #within range from Table 1 in https://doi.org/10.1016/j.biotechadv.2018.03.001
                design_flow_stream=None,
                sizing_flow=None,
                n_series=20,
                light_intensity = 100,
                annual_heating_days = 150,
                heater_up_time_ratio = 0.15,
                LED_power_per_area_basis=20.,
                LED_up_time_ratio=0.25,
                lighting_area_basis=504.,
                fan_number_per_area_basis=2.,
                fan_power_per_fan=0.75,
                fan_up_time_ratio=0.7,
                rho = 1000,
                include_construction= True,
                glass_tube_lifetime=30.,
                greenhouse_maintenance_ratio=0.016,
                LED_lifetime=12.,
                support_structure_maintenance_ratio=0.016,
                **kwargs):
        lca_ins = ()
        process_ins = ins
        if isinstance(ins, (list, tuple)) and len(ins) > 1:
            process_ins = ins[:1]
            lca_ins = tuple(ins[1:])
        self._N_lca_ins = len(lca_ins)
        self._lca_ins = lca_ins
        V_init = _initial_volume_from_hrt(
            hrt, sizing_flow=sizing_flow, n_series=n_series,
        )
        CSTR.__init__(self,ID=ID,ins=process_ins,outs=outs,split=None,V_max=V_init, W_tank = W_tank, D_tank = D_tank,
                freeboard = freeboard, t_wall = t_wall, t_slab = t_slab, aeration=aeration, 
                DO_ID=DO_ID, suspended_growth_model=suspended_growth_model, 
                gas_stripping=gas_stripping, gas_IDs=gas_IDs, stripping_kLa_min=stripping_kLa_min, 
                K_Henry=K_Henry, D_gas=D_gas, p_gas_atm=p_gas_atm,
                isdynamic=isdynamic, exogenous_vars=exogenous_vars, )
        for stream in lca_ins:
            self.ins.append(stream)
            if stream._source is None: stream._source = self
            stream._sink = None
        self.InD = InD
        self.OD = OD
        self.spt = spt
        self.hrt = hrt
        self.spacing = spacing
        self.support_spacing = support_spacing
        self.length = length
        self.v = v
        self.design_flow_stream = design_flow_stream
        self.sizing_flow = sizing_flow
        self.n_series = n_series
        self.light_intensity = light_intensity
        self.annual_heating_days = annual_heating_days
        self.heater_up_time_ratio = heater_up_time_ratio
        self.LED_power_per_area_basis = _require_nonnegative(
            'LED_power_per_area_basis', LED_power_per_area_basis,
        )
        self.LED_up_time_ratio = _require_nonnegative(
            'LED_up_time_ratio', LED_up_time_ratio,
        )
        if self.LED_up_time_ratio > 1:
            raise ValueError('`LED_up_time_ratio` cannot exceed 1.')

        if lighting_area_basis <= 0:
            raise ValueError('`lighting_area_basis` must be positive.')
        self.lighting_area_basis = lighting_area_basis
        self.fan_number_per_area_basis = _require_nonnegative(
            'fan_number_per_area_basis', fan_number_per_area_basis,
        )
        self.fan_power_per_fan = _require_nonnegative(
            'fan_power_per_fan', fan_power_per_fan,
        )
        self.fan_up_time_ratio = _require_nonnegative(
            'fan_up_time_ratio', fan_up_time_ratio,
        )
        if self.fan_up_time_ratio > 1:
            raise ValueError('`fan_up_time_ratio` cannot exceed 1.')
        self.rho = rho
        self.glass_tube_lifetime = _require_nonnegative(
            'glass_tube_lifetime', glass_tube_lifetime,
        )
        self.greenhouse_maintenance_ratio = _require_nonnegative(
            'greenhouse_maintenance_ratio', greenhouse_maintenance_ratio,
        )
        self.LED_lifetime = _require_nonnegative(
            'LED_lifetime', LED_lifetime,
        )
        self.support_structure_maintenance_ratio = _require_nonnegative(
            'support_structure_maintenance_ratio',
            support_structure_maintenance_ratio,
        )

    @property
    def _process_ins(self):
        N_lca = getattr(self, '_N_lca_ins', 0)
        if N_lca:
            return tuple(self.ins[:-N_lca])
        return tuple(self.ins)

    def _run(self):
        '''Only process inlets enter the PM2 mass balance; extra inlets are LCA-only.'''
        mixed = self._mixed
        mixed.mix_from(self._process_ins)
        Q = mixed.F_vol
        if self.split is None:
            self.outs[0].copy_like(mixed)
        else:
            for ws, spl in zip(self._outs, self.split):
                ws.copy_like(mixed)
                ws.set_total_flow(Q*spl, 'm3/hr')

    def _update_state(self):
        arr = self._state
        arr[arr < 1e-16] = 0.
        arr[-1] = sum(ws.state[-1] for ws in self._process_ins)
        if self.split is None:
            self._outs[0].state = arr
        else:
            for ws, spl in zip(self._outs, self.split):
                y = arr.copy()
                y[-1] *= spl
                ws.state = y
    
    def _init_lca(self):
        self.include_construction = True
        self.construction = [
            Construction('glasses', linked_unit=self, 
                         item='Borosilicate', 
                         quantity_unit='kg'),
            Construction('PVC', linked_unit=self, 
                         item='PVC', 
                         quantity_unit='kg'),
            Construction('HDPE', linked_unit=self, 
                         item='HDPE', 
                         quantity_unit='kg'),
            Construction("stainless_steel", linked_unit=self,
                         item = "StainlessSteel", 
                         quantity_unit= "kg"),
            Construction("aluminum", linked_unit=self,
                         item = "Aluminum", 
                         quantity_unit= "kg"),
            Construction("LED", linked_unit=self,
                         item = "LED", 
                         quantity_unit= "kg"),
            Construction("FRP", linked_unit=self,
                         item = "GFRPlastic", 
                         quantity_unit= "kg"),
            Construction("polycarbonate_glass", linked_unit=self,
                         item = "Polycarbonate", 
                         quantity_unit= "kg"),
            Construction('fan',linked_unit=self, 
                         item='Fan',
                         quantity_unit='ea'),
            Construction('gas_boiler',linked_unit=self, 
                         item='Fan',
                         quantity_unit='ea'),
            ]
        
    def _design(self):
        self._init_lca()
        self.design_results['Total flow'] = self.ins[0].F_vol #m3/hr
        ###Glass tube
        #working olume of glass tube
        design_flow = self.ins[0] if self.design_flow_stream is None else self.design_flow_stream
        design_Q = design_flow.F_vol or self.sizing_flow or 0.
        self.design_results['Working volume'] = design_Q*self.hrt #m3
        self.V_max = self.design_results['CSTR block working volume'] = self.design_results['Working volume'] / self.n_series
        #total length
        self.design_results['Total length'] = self.design_results['Working volume']/math.pi/(self.InD/2)**2 #m
        #the number of sets required
        tube_area = math.pi*(self.InD/2)**2
        N_blocks_float = self.design_results['Total flow']/3600/tube_area/self.v
        self.design_results['sets of PBR blocks'] = math.ceil(N_blocks_float)
        self.design_results['number of rows per set'] = math.ceil(self.design_results['Total length']/self.design_results['sets of PBR blocks']/self.length)
        self.design_results['Aerial footage'] = self.length*(self.design_results['sets of PBR blocks']+1)*self.spacing #m2
        self.design_results['volume-to-area ratio'] = self.design_results['Working volume'] / self.design_results['Aerial footage'] #m3/m2
        # self.design_results['Aerial footage-working-volume'] = self.design_results['Working volume']*self.design_results['Aerial footage'] #m5
        #total glass material weight
        rho_glass = 2230 #kg/m3, reference: https://www.imetra.com/borosilicate-glass-material-properties/
        self.design_results['Total glass weight'] = (
        math.pi * ((self.OD / 2) ** 2 - (self.InD / 2) ** 2)
        * self.design_results['Total length'])*rho_glass #kg
        self.construction[0].quantity = self.design_results['Total glass weight']
        self.design_results['U-bends'] = self.design_results['Total length']/self.length*2 #ea
        self.design_results['Plates'] = self.design_results['anchor bolts'] = self.design_results['U-bends']/2 #ea,based on Clearas estimation
        self.design_results['Plate mounting bolts'] =  self.design_results['nuts'] = self.design_results['washers']=self.design_results['Plates']*30 #ea,based on Clearas estimation
        self.design_results['Coupling'] = self.design_results['Total length']/3.1 #ea,based on Clearas estimation
        self.design_results['Hose clamps'] = self.design_results['Total length']/0.76 #ea,based on Clearas estimation
        
        self.construction[1].quantity = (self.design_results['U-bends']*1.4+ #U-bend, PVC
                                         self.design_results['Coupling']*0.91 #coupling, PVC
                                         )
        self.construction[4].quantity = self.design_results['Plates'] * 0.2 # aluminum plates
        self.construction[3].quantity = (self.design_results['Plate mounting bolts'] * 0.03 + 
                                         self.design_results['Hose clamps'] * 0.06 + #based on Clearas estimation
                                         self.design_results['anchor bolts'] * 0.7 +  #based on Clearas estimation
                                         self.design_results['nuts'] * 0.02 +
                                         self.design_results['washers'] * 0.005
                                         )
        self.construction[2].quantity = 424*lb_to_kg/79*self.design_results['sets of PBR blocks'] #kg HDPE pipe
        
        ###Support structure
        self.design_results['Support structure'] = math.ceil((self.length/self.support_spacing+1))*self.design_results['sets of PBR blocks'] #ea
        self.construction[3].quantity += self.design_results['Support structure']*50 #assume 50 kg stainless steel per support structure
        # self.design_results['Aerial footage-tube area per support area'] = (self.design_results['Aerial footage']*
        #                                                                 (math.pi*(self.InD/2)**2*
        #                                                              self.design_results['number of rows per set'] 
        #                                                              /self.support_spacing/self.spacing)) #m2
        
        ###Greenhouse
        self.design_results['Greenhouse panel area'] = 1.02*self.design_results['Aerial footage']+15.5*self.design_results['Aerial footage']**0.5 #m2, gable roof with 3m sidewall and 4m total height, aerial footage L:W= 4 
        panel_thickness = 0.008 #m,  http://www.unitedgreenhouse.com/accessories/multilayered-poly-coverings.php
        polycarbonate_density = 1200 #kg/m3
        self.construction[7].quantity = self.design_results['Greenhouse panel area']*panel_thickness*polycarbonate_density #kg
        area_ratio = self.design_results['Aerial footage'] / self.lighting_area_basis
        self.design_results['Number of Fans'] = (
            area_ratio * self.fan_number_per_area_basis
        ) #fan in ea, linear scale up from 2 fans per 504 m2 Clearas lighting-system basis
        self.construction[8].quantity = self.design_results['Number of Fans']
        self.design_results['250K BTU heater'] = math.ceil(self.design_results['Aerial footage']/6720*20) #ea, Modine HD125 model, number based on Clearas estimation, linear scale up from Waupun 6720 m2
        self.construction[9].quantity = self.design_results['250K BTU heater']*7 #ea 10 kW gas boiler, 250K BTU is equivalent to 7*10 kW gas boiler, assume linear relationship
        self.construction[6].quantity = self.design_results['Aerial footage']/6720*294*lb_to_kg #kg FRP

        self.design_results['natural gas'] = (
            self.design_results['250K BTU heater'] * self.heater_up_time_ratio
            * self.annual_heating_days / 365 * 2.5 * 1.9
        ) #kg/hr, 1.9 kg natural gas per therm, 2.5 therm/hr fuel input
        
        ###LED lighting
        self.design_results['LED light'] = self.light_intensity*self.design_results['Aerial footage']/1100 #ea, 1100 PPF per light based on Clearas estimation
        self.construction[4].quantity += self.design_results['LED light'] * 10 * lb_to_kg #kg, aluminum as the main component, https://californialightworks.com/megadrive/megadrive-linear-400/
        self.design_results['LED mounting structure'] = 32.3*self.design_results['LED light'] #kg, 32.3 kg each based on Clearas estimation
        self.construction[3].quantity += self.design_results['LED mounting structure'] #kg, stainless steel
        self.construction[5].quantity = self.design_results['LED light'] * 0.012 #kg LED, 12 g LED per fixture
        self.design_results['LED electricity'] = (
            area_ratio * self.LED_power_per_area_basis
            * self.LED_up_time_ratio
        ) #kW, 20 kW per 504 m2 Clearas lighting-system basis times average use fraction
        self.design_results['Fan electricity'] = (
            self.design_results['Number of Fans']
            * auom('hp').convert(self.fan_power_per_fan, 'kW')
            * self.fan_up_time_ratio
        ) #kW, 3/4 hp per fan times average use fraction
        self.design_results['Total electricity'] = (
            self.design_results['LED electricity']
            + self.design_results['Fan electricity']
        ) #kW
        
        ###Pigging & interconnects
        self.design_results['PIG assemblies PVC'] = 19907/6720*self.design_results['Aerial footage']*lb_to_kg #kg of PVC
        self.construction[1].quantity += self.design_results['PIG assemblies PVC']
        self.design_results['PIG assemblies stainless steel'] =1934/6720*self.design_results['Aerial footage']*lb_to_kg #kg of stainless steel
        self.construction[3].quantity += self.design_results['PIG assemblies stainless steel']

    def _cost(self):
        self._decorated_cost()
        self.power_utility.rate = self.design_results['Total electricity']
        self.add_OPEX['labor'] = self._calc_maintenance_labor_cost()
        self.add_OPEX['maintenance and replacement'] = (
            self._calc_replacement_cost()
        )

    def _calc_replacement_cost(self):
        # Glass tube lifetime source:
        # https://www.algaefoundationatec.org/aces/download/Techno-Economic%20Analysis.pdf.
        # Greenhouse/support annual maintenance source:
        # https://extension.msstate.edu/sites/default/files/publications/P2766.pdf.
        # LED lifetime source: Clearas documentation.
        annual = 0.
        if self.glass_tube_lifetime:
            annual += (
                _installed_capital(
                    self, ('Glass tube & fittings',),
                    include_parallel=False,
                ) / self.glass_tube_lifetime
            )
        annual += self.greenhouse_maintenance_ratio * _installed_capital(
            self, ('Greenhouse',), include_parallel=False,
        )
        if self.LED_lifetime:
            annual += (
                _installed_capital(
                    self, ('LED system',), include_parallel=False,
                ) / self.LED_lifetime
            )
        annual += self.support_structure_maintenance_ratio * _installed_capital(
            self, ('Support structure',), include_parallel=False,
        )
        return _annual_to_hourly_cost(annual)

    def _calc_maintenance_labor_cost(self):
        # Labor source: Table 8 in
        # https://www.algaefoundationatec.org/aces/download/Techno-Economic%20Analysis.pdf.
        V = self.design_results.get('Working volume', 0.)
        hours = (100 - 9) * 2000 * (V / 349000)**0.6 if V > 0 else 0.
        return _annual_to_hourly_cost(hours * _labor_wage())

#%%
class Ecorecoverypump(WWTpump):
    '''
    Pump subclass for Ecorecovery systems.

    Adds a feed_PBR pump type for the hydraulic design of fence-type tubular photobioreactors.

    Additional inputs for pump_type='feed_PBR'
    ------------------------------------------------
    InD : float
        Inner diameter of the PBR tube, [ft].
        Default is 0.1 m converted to ft.

    OD : float
        Outer diameter, or vertical spacing basis, of the PBR tube, [ft].

    hrt : float
        Hydraulic retention time based on influent flow, [hr].
        Default is 4 hr.

    length : float
        Maximum length of each PBR row, [ft].
        Default is 80 m converted to ft.

    v : float
        Designed flow velocity inside the PBR tube, [ft/s].
        This should represent the combined influent plus recirculation flow.
        Default is 0.4 m/s converted to ft/s.

    capacity_factor : float
        Design capacity factor for feed-PBR pump sizing, dimensionless.
        The pump inlet flow already includes process recirculation, so this
        factor is only for extra sizing capacity. The default is 1.

    # rr : float
    #     Recirculation ratio relative to influent flow, dimensionless.
    #     Default is 1.

    H_p : float
        Pressure head, [ft].
        Default is 0.

    L_s : float
        Suction pipe length, [ft].
        Default is 10 m converted to ft.

    N_ubends : int or None
        Number of U-bends per PBR set. If None, it is estimated as
        number of rows per set minus 1.
    Labor wage is read from ``pm2_ecorecover_lca.price_dct['Labor']`` [USD/hr].
    linked_unit : SanUnit, optional
        Unit that calculates an LCA resource demand for this pump.
    design_key : str
        Key in ``linked_unit.design_results`` that stores the resource demand.
    component_ID : str
        Component ID to assign in the resource stream.
    flow_unit : str
        Unit of the linked resource demand. The default is 'kg/hr'.
    '''
    _valid_pump_types = WWTpump._valid_pump_types + ('feed_PBR',)
    _ft_to_m = auom('ft').conversion_factor('m')
    _m_to_ft = 1 / _ft_to_m
    _g = 32.174  # gravitational acceleration, [ft/s2]
    # BioSTEAM Pump._F_BM_default uses 3.3 for both pump and motor.
    F_BM_pump = 3.3
    _lb_to_kg = auom('lb').conversion_factor('kg')
    default_F_BM = {
            'Pump': F_BM_pump,
            'Pump building': F_BM_pump,
            }
    default_equipment_lifetime = {
        'Pump': 15,
        'Pump pipe stainless steel': 15,
        'Pump stainless steel': 15,
        'Pump chemical storage HDPE': 30,
        }
    _feed_PBR_input_order = (
        'InD',      # [ft]
        'OD',       # [ft]
        'hrt',      # [hr]
        'length',   # [ft]
        'v',        # [ft/s]
        'capacity_factor', # dimensionless
        'H_p',      # [ft]
        'L_s',      # [ft]
        'N_ubends', # dimensionless, optional
    )

    def __init__(self, ID='', ins=None, outs=(), thermo=None,
                 init_with='WasteStream',
                 prefix='', pump_type='', Q_mgd=None, add_inputs=(),
                 InD=None, OD=None, hrt=4.,
                 length=None, v=None, 
                #  rr=1.,
                 H_p=0., L_s=None, N_ubends=None,
                 capacity_factor=4.,
                 include_pump_cost=True, include_building_cost=False,
                 include_OM_cost=False,
                 F_BM=default_F_BM,
                 lifetime=default_equipment_lifetime,
                 linked_unit=None,
                 design_key='',
                 component_ID='',
                 flow_unit='kg/hr',
                 **kwargs):

        super().__init__(
            ID=ID, ins=ins, outs=outs, thermo=thermo,
            init_with=init_with,
            prefix=prefix,
            pump_type=pump_type,
            Q_mgd=Q_mgd,
            add_inputs=add_inputs,
            capacity_factor=capacity_factor,
            include_pump_cost=include_pump_cost,
            include_building_cost=include_building_cost,
            include_OM_cost=include_OM_cost,
            F_BM=F_BM,
            lifetime=lifetime,
            **kwargs
        )

        # Store feed_PBR specific values separately so other WWTpump types
        # can still use WWTpump's default v, H_p, etc.
        self.feed_PBR_InD = 0.1 * self._m_to_ft if InD is None else InD
        self.feed_PBR_OD = 0.105 * self._m_to_ft if OD is None else OD
        self.feed_PBR_hrt = hrt
        self.feed_PBR_length = 80 * self._m_to_ft if length is None else length
        self.feed_PBR_v = 0.4 * self._m_to_ft if v is None else v
        self.feed_PBR_H_p = H_p
        self.feed_PBR_L_s = 10 * self._m_to_ft if L_s is None else L_s
        self.feed_PBR_N_ubends = N_ubends
        self.linked_unit = linked_unit
        self.design_key = design_key
        self.component_ID = component_ID
        self.flow_unit = flow_unit

    def _load_linked_resource_flow(self):
        linked_unit = getattr(self, 'linked_unit', None)
        design_key = getattr(self, 'design_key', '')
        component_ID = getattr(self, 'component_ID', '')
        if linked_unit is None or not design_key or not component_ID:
            return 0.
        flow = linked_unit.design_results.get(design_key, 0.)
        feed = self.ins[0]
        feed.empty()
        if flow:
            feed.set_flow(flow, self.flow_unit, component_ID)
        return flow

    def _run(self):
        self._load_linked_resource_flow()
        WWTpump._run(self)

    @property
    def state(self):
        '''Dynamic state passed from pump inlet to outlet.'''
        if self._state is None:
            return None
        return dict(zip(list(self.components.IDs) + ['Q'], self._state))

    def _init_state(self):
        # Dynamic pass-through method follows QSDsan dynamic Pump.
        self._state = self._ins_QC[0]
        self._dstate = self._state * 0.

    def _update_state(self):
        self._outs[0].state = self._state

    def _update_dstate(self):
        self._outs[0].dstate = self._dstate

    @property
    def AE(self):
        if self._AE is None:
            self._compile_AE()
        return self._AE

    def _compile_AE(self):
        _state = self._state
        _dstate = self._dstate
        _update_state = self._update_state
        _update_dstate = self._update_dstate
        def yt(t, QC_ins, dQC_ins):
            _state[:] = QC_ins[0]
            _dstate[:] = dQC_ins[0]
            _update_state()
            _update_dstate()
        self._AE = yt

    def _init_lca(self):
        self.include_construction = True
        self.construction = [
            Construction('stainless_steel', linked_unit=self,
                         item='StainlessSteel',
                         quantity_unit='kg'),
            ]

    def _design(self):
        self._load_linked_resource_flow()
        WWTpump._design(self)

    @property
    def pump_type(self):
        return self._pump_type

    @pump_type.setter
    def pump_type(self, i):
        i = i or ''
        i_lower = i.lower()
        i_lower = i_lower.replace('cstr', 'CSTR')
        i_lower = i_lower.replace('af', 'AF')
        i_lower = i_lower.replace('pbr', 'PBR')

        if i_lower not in self.valid_pump_types:
            raise ValueError(
                f'The given `pump_type` "{i}" is not valid, '
                'check `valid_pump_types` for acceptable pump types.'
            )

        self._pump_type = i_lower

    def _get_feed_PBR_inputs(self, InD=None, OD=None, hrt=None,
                             length=None, v=None, capacity_factor=None,
                             H_p=None, L_s=None, N_ubends=None):
        values = {
            'InD': self.feed_PBR_InD,
            'OD': self.feed_PBR_OD,
            'hrt': self.feed_PBR_hrt,
            'length': self.feed_PBR_length,
            'v': self.feed_PBR_v,
            'capacity_factor': self.capacity_factor,
            'H_p': self.feed_PBR_H_p,
            'L_s': self.feed_PBR_L_s,
            'N_ubends': self.feed_PBR_N_ubends,
        }

        # Allow add_inputs to follow the WWTpump style.
        for name, value in zip(self._feed_PBR_input_order, self.add_inputs):
            if value is not None:
                values[name] = value

        explicit_values = {
            'InD': InD,
            'OD': OD,
            'hrt': hrt,
            'length': length,
            'v': v,
            'capacity_factor': capacity_factor,
            'H_p': H_p,
            'L_s': L_s,
            'N_ubends': N_ubends,
        }

        for name, value in explicit_values.items():
            if value is not None:
                values[name] = value

        missing = [
            name for name in ('InD', 'OD', 'hrt', 'length', 'v', 'capacity_factor', 'H_p', 'L_s')
            if values[name] is None
        ]
        if missing:
            raise ValueError(
                "`pump_type='feed_PBR'` is missing required input(s): "
                f"{', '.join(missing)}."
            )

        if values['InD'] <= 0:
            raise ValueError('`InD` must be positive, [ft].')
        if values['OD'] <= 0:
            raise ValueError('`OD` must be positive, [ft].')
        if values['hrt'] <= 0:
            raise ValueError('`hrt` must be positive, [hr].')
        if values['length'] <= 0:
            raise ValueError('`length` must be positive, [ft].')
        if values['v'] <= 0:
            raise ValueError('`v` must be positive, [ft/s].')
        if values['capacity_factor'] <= 0:
            raise ValueError('`capacity_factor` must be positive.')
        if values['L_s'] < 0:
            raise ValueError('`L_s` must be non-negative, [ft].')

        return values

    def design_feed_PBR(self, Q_mgd=None, InD=None, OD=None, hrt=None,
                        length=None, v=None, capacity_factor=None,
                        H_p=None, L_s=None, N_ubends=None,
                        **kwargs):
        '''
        Design the feed pump for a fence-type tubular PBR.

        This method intentionally does not call WWTpump._design_generic().
        It follows the PBRpump hydraulic design structure, but all length,
        volume, and velocity calculations are in US units.
        '''

        # Allow additional overrides through kwargs.
        if kwargs:
            for k, val in kwargs.items():
                if k in self._feed_PBR_input_order:
                    locals()[k] = val
                else:
                    setattr(self, k, val)

        if Q_mgd is None:
            Q_mgd = self.Q_mgd
        self.Q_mgd = Q_mgd

        vals = self._get_feed_PBR_inputs(
            InD=InD, OD=OD, hrt=hrt, length=length,
            v=v, capacity_factor=capacity_factor, H_p=H_p, L_s=L_s, N_ubends=N_ubends,
        )

        InD = vals['InD']          # [ft]
        OD = vals['OD']            # [ft]
        hrt = vals['hrt']          # [hr]
        length = vals['length']    # [ft]
        v = vals['v']              # [ft/s]
        capacity_factor = vals['capacity_factor']  # [-]
        H_p = vals['H_p']          # [ft]
        L_s = vals['L_s']          # [ft]
        N_ubends = vals['N_ubends']

        D = self.design_results

        # Flow and PBR sizing, all in US units.
        Q_cfs = self.Q_cfs                         # [ft3/s], influent flow
        Q_cfh = Q_cfs * 3600                       # [ft3/hr]
        tube_area = pi / 4 * InD**2                # [ft2]

        if Q_cfs <= 0:
            self.N_pump = 0
            self._H_ts = self._H_sf = self._H_df = self._H_p = 0.
            D['Total flow'] = 0.
            D['Working volume'] = 0.
            D['Total length'] = 0.
            D['sets of PBR blocks'] = 0
            D['number of rows per set'] = 0
            D['U-bends'] = 0
            if self.include_construction:
                self.construction[0].quantity = 0.
            return 0., 0., 0.


        # Working volume = influent flow * HRT
        # Total length = working volume / tube cross sectional area
        working_volume = Q_cfh * hrt               # [ft3]
        total_length = working_volume / tube_area  # [ft]

        # Number of PBR sets, equivalent to N_pump in WWTpump logic.
        N_pump = ceil(Q_cfs * capacity_factor / tube_area / v)
        N_pump = max(N_pump, 1)
        self.N_pump = N_pump

        rows_per_set = ceil(total_length / N_pump / length)
        rows_per_set = max(rows_per_set, 1)

        if N_ubends is None:
            N_ubends = max(rows_per_set, 0) * 2
        else:
            N_ubends = ceil(N_ubends)

        D['Total flow'] = Q_cfh                         # [ft3/hr]
        D['Working volume'] = working_volume            # [ft3]
        D['Total length'] = total_length                # [ft]
        D['sets of PBR blocks'] = N_pump                # [-]
        D['number of rows per set'] = rows_per_set      # [-]
        D['U-bends'] = N_ubends                         # [-]

        # Store units for the additional feed_PBR design results.
        self._units['Total flow'] = 'ft3/hr'
        self._units['Working volume'] = 'ft3'
        self._units['Total length'] = 'ft'
        self._units['sets of PBR blocks'] = ''
        self._units['number of rows per set'] = ''
        self._units['U-bends'] = ''

        self._v = v
        C = self.C

        # Suction side pipe sizing.
        OD_s, t_s, ID_s = select_pipe(Q_cfs / N_pump, v)  # [in]

        # Static head.
        # Height simplifies to PBR vertical height

        self._H_ts = rows_per_set * OD  # [ft]

        # Suction friction head, Hazen-Williams form.
        self._H_sf = (
            3.02 * L_s * v**1.85 * C**(-1.85) * InD**(-1.17)
        )
        self._H_sf *= self.headloss_multiplication_factor

        # Discharge friction head along one PBR set.
        length_per_set = total_length / N_pump  # [ft]

        # Minor loss from U-bends.
        #https://www.engineeringtoolbox.com/minor-loss-coefficients-pipes-d_626.html
        H_bends = N_ubends * 0.2 * v**2 / (2 * self._g)  # [ft]

        self._H_df = (
            3.02 * length_per_set * v**1.85 * C**(-1.85) * InD**(-1.17)
            + H_bends
        )
        self._H_df *= self.headloss_multiplication_factor

        self._H_p = H_p

        # Stainless steel pipe mass.
        # only counts suction-side pipe SS.
        V_s = N_pump * pi / 4 * (OD_s**2 - ID_s**2) * (L_s * 12)  # [in3]
        M_SS_pipe = 0.29 * V_s * self._lb_to_kg          # [kg]

        # Stainless steel pump mass follows WWTpump's N_pump handling.
        M_SS_pump = N_pump * self.SS_per_pump                       # [kg]
        if self.include_construction:
            self.construction[0].quantity = M_SS_pipe + M_SS_pump

        return M_SS_pipe, M_SS_pump, 0.

    def _cost(self):
        super()._cost()
        C = self.baseline_purchase_costs
        start = self._format_key_start_with_prefix('P')
        pump_key = f'{start}ump'
        Q_mgd = self.Q_mgd
        Q_cost_mgd = Q_mgd * self.capacity_factor
        if self.include_pump_cost and Q_cost_mgd > 0:
            N_cost = ceil(Q_cost_mgd / 2.5)
            Q_per_pump = Q_cost_mgd / N_cost
            # Fitted curve; valid for per-pump flow <= 2.5 MGD.
            C[pump_key] = N_cost * (7116.1 * Q_per_pump + 1002.4)
        else:
            N_cost = 0
            Q_per_pump = 0.
            C[pump_key] = 0.
        self.design_results['Number of costed pumps'] = N_cost
        self.design_results['Cost capacity factor'] = self.capacity_factor
        self.design_results['Costed design flow'] = Q_cost_mgd
        self.design_results['Costed flow per pump'] = Q_per_pump
        self._units['Number of costed pumps'] = ''
        self._units['Cost capacity factor'] = ''
        self._units['Costed design flow'] = 'MGD'
        self._units['Costed flow per pump'] = 'MGD'
        self.add_OPEX['labor'] = self._calc_maintenance_labor_cost()
        self.add_OPEX['maintenance and replacement'] = (
            self._calc_replacement_cost()
        )

    def _calc_replacement_cost(self):
        # Pump lifetime assumption: 15 yr.
        pump_keys = tuple(
            k for k in self.baseline_purchase_costs
            if 'ump' in k and 'building' not in k
        )
        return _annual_to_hourly_cost(
            _installed_capital(self, pump_keys, include_parallel=False) / 15
        )

    def _calc_maintenance_labor_cost(self):
        # Labor source: NEIWPCC Northeast Staffing Guide,
        # https://neiwpcc.org/wp-content/uploads/2020/08/NEIWPCC-Northeast-Staffing-Guide.pdf.
        return _annual_to_hourly_cost(
            10 * getattr(self, 'N_pump', 0) * _labor_wage()
        )
