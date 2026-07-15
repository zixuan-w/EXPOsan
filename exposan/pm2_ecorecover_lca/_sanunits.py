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
from qsdsan.unit_operations.bst._pumping import Pump
from qsdsan import processes as pc
from qsdsan.utils import auom, select_pipe, format_str, get_P_blower
from biosteam.units.design_tools import (
    CEPCI_by_year,
    compute_number_of_tanks_and_purchase_cost,
    mix_tank_purchase_cost_algorithms,
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
           'CO2Supply',
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
    return cost / hours_per_year


def _installed_capital(unit, IDs=None, include_parallel=True):
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
    if value < 0:
        raise ValueError(f'`{name}` must be non-negative.')
    return value


def _warn_if_flow_outside_NEIWPCC(unit, Q_m3_d, name):
    if 0 < Q_m3_d < 1400 or Q_m3_d > 76000:
        warn(
            f'The NEIWPCC labor-hour relationship for {name} may not hold '
            f'for flow rates outside 1400-76000 m3/d; current flow is '
            f'{Q_m3_d:.3g} m3/d.',
            RuntimeWarning,
            stacklevel=3,
        )

class Mixtank(CSTR):
    '''
    Tank with design, BioSTEAM mixing-tank cost, and power estimation.

    The process model, aeration behavior, and dynamic mass balances are inherited
    from :class:`qsdsan.unit_operations.CSTR`. Tank geometry is reported for
    reference, but concrete is not included in the purchase cost.

    Parameters
    ----------
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
    labor_wage : float
        Labor wage in [USD/hr].
    Air compressor carbon steel follows the compressor weight relationship
        [kg] = 16.013 * aeration power [hp] + 75.813.
        source: https://us.kaeser.com/products-and-solutions/rotary-screw-compressors/3-hp.aspx
    
    References
    ----------
    # TODO: insert references from comments in this unit
    '''
    _F_BM_default = {
        'Tank': 2.3,
        'Air compressor': 2.15,
        'Diffusers': 1.,
    }
    _units = {
        **CSTR._units,
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
    purchase_cost_algorithms = mix_tank_purchase_cost_algorithms

    def __init__(self, ID='', ins=None, outs=(), thermo=None,
                 init_with='WasteStream', split=None, V_max=1000,
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
                 labor_wage=0.,
                 **kwargs):
        CSTR.__init__(
            self, ID=ID, ins=ins, outs=outs, split=split, thermo=thermo,
            init_with=init_with, V_max=V_max, W_tank=W_tank,
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
        self.labor_wage = self._require_nonnegative(
            'labor_wage', labor_wage,
        )

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

    def _design(self):
        D = self.design_results
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
        D['Volume of concrete wall'] = (
            2 * (L + 2 * t_wall) * t_wall * total_depth
            + 2 * W * t_wall * total_depth
        )
        D['Volume of concrete slab'] = (
            (L + 2 * t_wall) * (W + 2 * t_wall) * t_slab
        )
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
        D['Air compressor carbon steel'] = (
            16.013 * aeration_power_hp + 75.813
            if aeration_power > 0 else 0.
        ) #kg
        D['Diffuser area'] = diffuser_area #m2
        D['Diffuser stainless steel'] = diffuser_area * 181 / 18.5 #kg

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
        return _annual_to_hourly_cost(hours * self.labor_wage)

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
    influent dissolved-CO2 concentration.

    Parameters
    ----------
    target_CO2 : float
        Target dissolved-CO2 concentration in [mg/L].
    excess_fraction : float
        Fractional allowance for unmodeled CO2 losses.
    CO2_ID : str
        Component ID representing dissolved CO2.
    CO2_price : float
        CO2 price in 2016 [USD/metric tonne].
    labor_wage : float
        Labor wage in [USD/hr].

    References
    ----------
    # TODO: insert references from comments in this unit
    '''
    _N_ins = 1
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
            init_with='WasteStream', target_CO2=30.,
            excess_fraction=0.10, CO2_ID='S_CO2',
            CO2_price=45., #source: https://docs.nlr.gov/docs/fy19osti/72716.pdf
            labor_wage=0.,
        ):
        SanUnit.__init__(
            self, ID=ID, ins=ins, outs=outs, thermo=thermo,
            init_with=init_with,
        )
        self.target_CO2 = target_CO2
        self.excess_fraction = excess_fraction
        self.CO2_ID = CO2_ID
        self.CO2_price = CO2_price
        self.labor_wage = self._require_nonnegative(
            'labor_wage', labor_wage,
        )

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

    def _design(self):
        D = self.design_results
        influent = self.ins[0]
        Q = influent.get_total_flow('L/hr')
        C_in = (
            0. if Q == 0 else float(
                influent.get_mass_concentration(
                    'mg/L', IDs=(self.CO2_ID,),
                )[0]
            )
        )
        # mg/L * L/hr * 1e-6 = kg/hr.
        base_makeup = max(self.target_CO2 - C_in, 0.) * Q * 1e-6
        # User-settable allowance for unmodeled CO2 losses.
        supply = base_makeup * (1 + self.excess_fraction)

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
            0.015 * self.design_results['CO2 supply'] * self.labor_wage
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
    V_max : float
        UF tank working volume used for tank design and chemical cleaning in
        [m3].
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
    diffuser_unit_cost : float
        Diffuser purchase cost per diffuser area in [USD/m2].
    compressor_specific_mass : float
        Stainless-steel compressor mass per blower power in [kg/kW].
    chemical_cleaning_frequency : float
        Chemical cleaning frequency in [1/d].
    citric_acid_concentration : float
        Citric-acid cleaning concentration in [mg/L].
    sodium_hypochlorite_concentration : float
        Sodium-hypochlorite cleaning concentration in [mg/L].
    citric_acid_unit_price : float
        Citric-acid price in [USD/kg].
    sodium_hypochlorite_unit_price : float
        Sodium-hypochlorite price in [USD/kg].
    labor_wage : float
        Labor wage in [USD/hr].
    '''
    _F_BM_default = {
        'Membrane': 3.2, #source: http://doi.org/10.1021/acs.iecr.2c00598
        'Tank': 2.3,
        'Air compressor': 2.15,
        'Diffusers': 2.15,
    }
    _units = {
        'Influent flow': 'm3/d',
        'Designed flow': 'm3/d',
        'Water viscosity': 'Pa*s',
        'Membrane flux': 'm3/m2/s',
        'Membrane area': 'm2',
        'Membrane module area': 'm2',
        'Tank volume': 'm3',
        'Sparging air flow': 'm3/hr',
        'Sparging air flow at compressor': 'cfm',
        'Number of air compressors': '',
        'Sparging power': 'kW',
        'Air compressor stainless steel': 'kg',
        'Diffuser area': 'm2',
        'Diffuser stainless steel': 'kg',
        'Citric acid usage': 'kg/hr',
        'Sodium hypochlorite usage': 'kg/hr',
    }
    purchase_cost_algorithms = mix_tank_purchase_cost_algorithms

    def __init__(
            self, ID='', ins=None, outs=(), thermo=None, *, split,
            order=None, init_with='WasteStream', F_BM_default=None,
            isdynamic=False, R_t=5e12, #steady-state flux based on https://doi.org/10.1021/acs.est.3c10264; https://www.kovalus.com/wp-content/uploads/2020/10/puron-hf-modules.pdf
            T=25., TMP=15.4e3,
            capacity_factor=1.5, include_tank=False,
            include_sparging=False, include_chemical_cleaning=False,
            V_max=3.8*2, #source: https://doi.org/10.1021/acs.est.3c10264
            V_wf=0.8, vessel_type='Conventional',
            vessel_material='Stainless steel',
            specific_sparging_air_demand=0.7, blower_T=20,
            P_atm=101.325, P_inlet_loss=1, P_diffuser_loss=7,
            h_submergance=5.18, blower_efficiency=0.7,
            blower_K=0.283, diffuser_unit_cost=0.,
            compressor_specific_mass=0.,
            chemical_cleaning_frequency=1/30,# per day, source: https://doi.org/10.1016/j.scitotenv.2024.177273. change to 30 days for more conservative cleaning
            citric_acid_concentration=2000.,
            sodium_hypochlorite_concentration=2000.,
            citric_acid_unit_price=1.06*euro_to_usd,
            sodium_hypochlorite_unit_price=0.88/0.125*euro_to_usd,
            labor_wage=0.,
            membrane_lifetime = 5,
        ):
        su.Splitter.__init__(
            self, ID=ID, ins=ins, outs=outs, thermo=thermo, split=split,
            order=order, init_with=init_with, F_BM_default=F_BM_default,
            isdynamic=isdynamic,
        )
        self.R_t = R_t
        self.T = T
        self.TMP = TMP
        self.capacity_factor = capacity_factor
        self.include_tank = bool(include_tank)
        self.include_sparging = bool(include_sparging)
        self.include_chemical_cleaning = bool(include_chemical_cleaning)
        self.V_max = V_max
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
        self.diffuser_unit_cost = diffuser_unit_cost
        self.compressor_specific_mass = compressor_specific_mass
        self.chemical_cleaning_frequency = chemical_cleaning_frequency
        self.citric_acid_concentration = citric_acid_concentration
        self.sodium_hypochlorite_concentration = sodium_hypochlorite_concentration
        self.citric_acid_unit_price = citric_acid_unit_price
        self.sodium_hypochlorite_unit_price = sodium_hypochlorite_unit_price
        self.membrane_lifetime = membrane_lifetime
        self.labor_wage = self._require_nonnegative(
            'labor_wage', labor_wage,
        )

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
    def V_max(self):
        return self._V_max

    @V_max.setter
    def V_max(self, value):
        self._V_max = self._require_positive('V_max', value)

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
    def compressor_specific_mass(self):
        return self._compressor_specific_mass

    @compressor_specific_mass.setter
    def compressor_specific_mass(self, value):
        self._compressor_specific_mass = self._require_nonnegative(
            'compressor_specific_mass', value,
        )

    @property
    def chemical_cleaning_frequency(self):
        return self._chemical_cleaning_frequency

    @chemical_cleaning_frequency.setter
    def chemical_cleaning_frequency(self, value):
        self._chemical_cleaning_frequency = self._require_nonnegative(
            'chemical_cleaning_frequency', value,
        )

    @property
    def citric_acid_concentration(self):
        return self._citric_acid_concentration

    @citric_acid_concentration.setter
    def citric_acid_concentration(self, value):
        self._citric_acid_concentration = self._require_nonnegative(
            'citric_acid_concentration', value,
        )

    @property
    def sodium_hypochlorite_concentration(self):
        return self._sodium_hypochlorite_concentration

    @sodium_hypochlorite_concentration.setter
    def sodium_hypochlorite_concentration(self, value):
        self._sodium_hypochlorite_concentration = self._require_nonnegative(
            'sodium_hypochlorite_concentration', value,
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
        Q = self.ins[0].get_total_flow('m3/d')
        Q_design = Q * self.capacity_factor

        # Source: https://doi.org/10.1016/j.scitotenv.2024.177273.
        mu = 497e-3 / (self.T + 42.5)**1.5 #relative viscoity, unit Pa x s, Source: https://doi.org/10.1016/j.desal.2021.115409
        J = self.TMP / mu / self.R_t
        A = Q_design / 24 / 3600 / J if Q_design else 0.
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

    def _chemical_usage(self, concentration):#kg/hr
        return (
            self.V_max * 1000 * concentration * 1e-6
            * self.chemical_cleaning_frequency / 24
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
            hours += self.chemical_cleaning_frequency * 365
        return _annual_to_hourly_cost(hours * self.labor_wage)

    def _design(self):
        D = self.design_results
        Q, Q_design, mu, J, A = self._get_membrane_design()

        D['Influent flow'] = Q
        D['Designed flow'] = Q_design
        D['Water viscosity'] = mu
        D['Membrane flux'] = J
        D['Membrane area'] = A
        D['Membrane module area'] = A
        D['Tank volume'] = self.V_max if self.include_tank else 0.
        D['Sparging air flow'] = 0.
        D['Sparging air flow at compressor'] = 0.
        D['Number of air compressors'] = 0
        D['Sparging power'] = 0.
        D['Air compressor stainless steel'] = 0.
        D['Diffuser area'] = 0.
        D['Diffuser stainless steel'] = 0.
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

            # Source: compressor sizing and diffuser mass method used in Tank.
            D['Sparging air flow'] = Q_air
            D['Sparging air flow at compressor'] = Q_air_acfm
            D['Number of air compressors'] = N_compressors
            D['Sparging power'] = power
            D['Air compressor stainless steel'] = (
                power * self.compressor_specific_mass
            )
            D['Diffuser area'] = A
            D['Diffuser stainless steel'] = A * 181 / 18.5

        if self.include_chemical_cleaning:
            # Source: cleaning frequency and concentrations from
            # https://doi.org/10.1016/j.scitotenv.2024.177273.
            D['Citric acid usage'] = self._chemical_usage(
                self.citric_acid_concentration,
            ) #kg/hr
            D['Sodium hypochlorite usage'] = self._chemical_usage(
                self.sodium_hypochlorite_concentration,
            )

    def _cost(self):
        D = self.design_results
        C = self.baseline_purchase_costs
        A = D['Membrane area']

        if A > 0:
            # Source: https://doi.org/10.1016/j.scitotenv.2024.177273.
            membrane_unit_cost = (
                -2.985 * np.log(D['Membrane module area']) + 68.159
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
            C['Diffusers'] = D['Diffuser area'] * self.diffuser_unit_cost
            self.F_M['Air compressor'] = 2.5
        else:
            C.pop('Air compressor', None)
            C.pop('Diffusers', None)

        if self.include_chemical_cleaning:
            self.add_OPEX['Citric acid'] = (
                D['Citric acid usage'] * self.citric_acid_unit_price
            )
            self.add_OPEX['Sodium hypochlorite'] = (
                D['Sodium hypochlorite usage']
                * self.sodium_hypochlorite_unit_price
            )
        else:
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
    labor_wage : float
        Labor wage in [USD/hr].

    References
    ----------
    Cost follows BioSTEAM ``LiquidsCentrifuge``:
    ``C = 28100 * Q**0.574`` with ``Q`` in [m3/hr], CEPCI 525.4,
    upper-bound flow 100 m3/hr, and bare-module factor 2.03.

    Energy equations follow Najjar and Abu-Shamleh, Algal Research 51
    (2020) 102046, http://doi.org/10.1016/j.algal.2020.102046.

    Stainless-steel weight follows the Dolphin Centrifuge capacity
    correlation:
    ``weight = 1126.1*ln(Q) - 1204.8`` with ``Q`` in [m3/hr].
    '''
    _F_BM_default = {'Centrifuge': 2.03}
    _units = {
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
            isdynamic=False, algal_cell_diameter=5e-6,
            algal_particle_density=1050., water_density=1000.,
            T=25., phi=0., vgm=0.1e-6, UF_flow=None,
            labor_wage=0.,
        ):
        su.Splitter.__init__(
            self, ID=ID, ins=ins, outs=outs, thermo=thermo, split=split,
            order=order, init_with=init_with, F_BM_default=F_BM_default,
            isdynamic=isdynamic,
        )
        self.water_density = water_density
        self.algal_cell_diameter = algal_cell_diameter
        self.algal_particle_density = algal_particle_density
        self.T = T
        self.phi = phi
        self.vgm = vgm
        self.UF_flow = UF_flow
        self.labor_wage = self._require_nonnegative(
            'labor_wage', labor_wage,
        )

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
            hours_per_unit * N * self.labor_wage
        )

    def _design(self):
        D = self.design_results
        Q_hr = self.ins[0].get_total_flow('m3/hr')
        mu = self._get_mu()
        vg = self._get_gravity_settling_velocity(mu)
        vg_eff = vg * (1 - self.phi)**4.65

        if Q_hr > 0:
            # Source: Eq. 15 in Najjar and Abu-Shamleh (2020),
            # converts actual flow to master-curve flow [m3/hr].
            Q_s = self.ins[0].get_total_flow('m3/s')
            Qm = Q_s * 3600 * self.vgm / vg_eff
            # Source: Eq. 16 in Najjar and Abu-Shamleh (2020),
            # disc centrifuge energy intensity [kWh/m3].
            E_disc = 1.447 * Qm**(-0.304)
            # Source: Dolphin Centrifuge capacity correlation;
            # clamp to zero outside the low-flow correlation range.
            weight = max(1126.1 * np.log(Q_hr) - 1204.8, 0.)
        else:
            Qm = E_disc = weight = 0.

        D['Influent flow'] = Q_hr
        D['Water viscosity'] = mu
        D['Gravity settling velocity'] = vg
        D['Effective settling velocity'] = vg_eff
        D['Master-curve flow'] = Qm
        D['Disc centrifuge energy intensity'] = E_disc
        D['Centrifuge stainless steel'] = weight
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

@cost(basis = 'Aerial footage-volume-to-area ratio', ID='Glass tube & fittings', units='m3/m2',
      cost = 233240/acre_to_sq_m, S=0.029499829299047615, CE=CEPCI_by_year[2014], n=1, BM=1.1) #ref:https://docs.nrel.gov/docs/fy19osti/72716.pdf
@cost(basis='Aerial footage', ID='Greenhouse',units='m2',cost= 13*sq_feet_to_sq_m, S=1, CE=CEPCI_by_year[1994], n=1, BM=1) #conventional greenhouse in ref page 38: https://blog.uvm.edu/cwcallah/files/2021/03/NRAES-33_Web.pdf
@cost(basis='Aerial footage-tube area per support area', ID='Support structure', units='m2',
      cost=118937/acre_to_sq_m, S=0.00983, CE=CEPCI_by_year[2014], n=1, BM=1.1) #ref:https://docs.nrel.gov/docs/fy19osti/72716.pdf
@cost(basis='Aerial footage', ID= 'LED system', units = 'm2', cost= 11.27*sq_feet_to_sq_m, S=1, CE=CEPCI_by_year[2025],n=1, BM=1.1)

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
    rr: float
        The designed recirculation rate proportional to the influent flow rate. The default is 1.
    rho: float
        The density of the mixed liquor [kg/m3]. The default is 1000.
    light_intensity: int
        The designed light intensity [umol/(m2·s)] over the illuminated area. The default is 100.
    annual_heating_days: int
        The average winter days per year that requires heaters, default to 150 days.
    heater_up_time_ratio: float
        The average daily up time ratio [0-1] of heaters, default to 0.15.
    labor_wage : float
        Labor wage in [USD/hr].
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
    V_max : float
        Designed volume, in [m^3]. The default is 1000.
        
    # W_tank : float
    #     The design width of the tank, in [m]. The default is 6.4 m (21 ft). [1, Yalin's adaptation of code]
    # D_tank : float
    #     The design depth of the tank in [m]. The default is 3.65 m (12 ft). [1, Yalin's adaptation of code]
    # freeboard : float
    #     Freeboard added to the depth of the reactor tank, [m]. The default is 0.61 m (2 ft). [1, Yalin's adaptation of code]
        
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
    
    '''
    _units = {
        **CSTR._units,
        'Aerial footage-volume-to-area ratio': 'm3/m2',
        'Aerial footage': 'm2',
        'Aerial footage-tube area per support area': 'm2',
    }

    def __init__(self, ID='', ins=None, outs=(), thermo=None, init_with='WasteStream',
                split=None,V_max=1000, W_tank = 6.4, D_tank = 3.65,
                freeboard = 0.61, t_wall = None, t_slab = None, aeration=2.0, 
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
                v= 0.4,
                rr=1,
                light_intensity = 100,
                annual_heating_days = 150,
                heater_up_time_ratio = 0.15,
                rho = 1000,
                include_construction= True,
                labor_wage=0.,
                glass_tube_lifetime=30.,
                greenhouse_maintenance_ratio=0.016,
                LED_lifetime=12.,
                support_structure_maintenance_ratio=0.016,
                **kwargs):
        CSTR.__init__(self,ID=ID,ins=ins,outs=outs,split=None,V_max=V_max, W_tank = W_tank, D_tank = D_tank,
                freeboard = freeboard, t_wall = t_wall, t_slab = t_slab, aeration=aeration, 
                DO_ID=DO_ID, suspended_growth_model=suspended_growth_model, 
                gas_stripping=gas_stripping, gas_IDs=gas_IDs, stripping_kLa_min=stripping_kLa_min, 
                K_Henry=K_Henry, D_gas=D_gas, p_gas_atm=p_gas_atm,
                isdynamic=isdynamic, exogenous_vars=exogenous_vars, )
        SanUnit.__init__(self, ID, ins, outs, thermo, init_with,F_BM_default=None)
        self.InD = InD
        self.OD = OD
        self.spt = spt
        self.hrt = hrt
        self.spacing = spacing
        self.support_spacing = support_spacing
        self.length = length
        self.v = v
        self.rr = rr
        self.light_intensity = light_intensity
        self.annual_heating_days = annual_heating_days
        self.heater_up_time_ratio = heater_up_time_ratio
        self.rho = rho
        self.labor_wage = _require_nonnegative('labor_wage', labor_wage)
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
    
    def _init_lca(self):#TODO
        self.include_construction = True
        self.construction = [
            Construction('glasses', linked_unit=self, 
                         item='XXXX', 
                         quantity_unit='kg'),
            Construction('PVC', linked_unit=self, 
                         item='XXXX', 
                         quantity_unit='kg'),
            Construction('HDPE', linked_unit=self, 
                         item='XXXX', 
                         quantity_unit='kg'),
            Construction("stainless_steel", linked_unit=self,
                         item = "StainlessSteel", 
                         quantity_unit= "kg"),
            Construction("aluminum", linked_unit=self,
                         item = "XXXX", 
                         quantity_unit= "kg"),
            Construction("LED", linked_unit=self,
                         item = "XXXX", 
                         quantity_unit= "kg"),
            Construction("FRP", linked_unit=self,
                         item = "XXXX", 
                         quantity_unit= "kg"),
            Construction("polycarbonate_glass", linked_unit=self,
                         item = "XXXX", 
                         quantity_unit= "kg"),
            Construction('fan',linked_unit=self, 
                         item='Fan',
                         quantity_unit='kg'),
            ]
        
    def _design(self):
        self.design_results['Total flow'] = self.ins[0].F_vol #m3/hr
        ###Glass tube
        #vworking olume of glass tube
        self.design_results['Working volume'] = self.design_results['Total flow']*self.hrt #m3
        #total length
        self.design_results['Total length'] = self.design_results['Working volume']/math.pi/(self.InD/2)**2 #m
        #the number of sets required
        self.design_results['sets of PBR blocks'] = math.ceil(self.design_results['Total flow']/3600*
                                                              (1+self.rr)/math.pi/(self.InD/2)**2/self.v)
        self.design_results['number of rows per set'] = math.ceil(self.design_results['Total length']/self.design_results['sets of PBR blocks']/self.length)
        self.design_results['Aerial footage'] = self.length*(self.design_results['sets of PBR blocks']+1)*self.spacing #m2
        self.design_results['volume-to-area ratio'] = self.design_results['Working volume'] / self.design_results['Aerial footage'] #m3/m2
        self.design_results['Aerial footage-volume-to-area ratio'] = self.design_results['volume-to-area ratio']*self.design_results['Aerial footage']
        #total glass material weight
        rho_glass = 2230 #kg/m3, reference: https://www.imetra.com/borosilicate-glass-material-properties/
        self.design_results['Total glass weight'] = (
        math.pi * ((self.OD / 2) ** 2 - (self.InD / 2) ** 2)
        * self.design_results['Total length'])*rho_glass #kg
        self.construction[0].quantity = self.design_results['Total glass weight']
        self.design_results['U-bends'] = self.design_results['Total length']/self.length*2 #ea
        self.design_results['Plates'] = self.design_results['anchor bolts'] = self.design_results['U-bends']/2 #ea,based on Clearas estimation
        self.design_results['Plate mounting bolts'] =  self.design_results['nuts'] = self.design_results['washers']=self.design_results['plates']*30 #ea,based on Clearas estimation
        self.design_results['Coupling'] = self.design_results['Total length']/3.1 #ea,based on Clearas estimation
        self.design_results['Hose clamps'] = self.design_results['Total length']/0.76 #ea,based on Clearas estimation
        
        self.construction[1].quantity = (self.design_results['U-bends']*1.4+ #U-bend, PVC
                                         self.design_results['coupling']*0.91 #coupling, PVC
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
        self.design_results['Aerial footage-tube area per support area'] = (self.design_results['Aerial footage']*
                                                                        (math.pi*(self.InD/2)**2*
                                                                     self.design_results['number of rows per set'] 
                                                                     /self.support_spacing/self.spacing)) #m2
        
        ###Greenhouse
        self.design_results['Greenhouse panel area'] = 1.02*self.design_results['Aerial footage']+15.5*self.design_results['Aerial footage']**0.5 #m2, gable roof with 3m sidewall and 4m total height, aerial footage L:W= 4 
        panel_thickness = 0.008 #m,  http://www.unitedgreenhouse.com/accessories/multilayered-poly-coverings.php
        polycarbonate_density = 1200 #kg/m3
        self.construction[7].quantity = self.design_results['Greenhouse panel area']*panel_thickness*polycarbonate_density #kg
        self.construction[8].quantity = self.design_results['Fans weight']= self.design_results['Aerial footage']/504*722.98*lb_to_kg #fan in kg based on clearas estimation, linear scale up
        self.design_results['250K BTU heater'] = math.ceil(self.design_results['Aerial footage']/6720*20) #ea, Modine HD125 model, number based on Clearas estimation
        self.construction[3].quantity += self.design_results['250K BTU heater']*143* lb_to_kg #kg stainless steel
        self.construction[6].quantity = self.design_results['Aerial footage']/6720*294*lb_to_kg #kg FRP
        
        #TODO
        #need to convert natural gas into a stream
        self.design_results['natural gas'] = (self.design_results['250K BTU heater']*
                                              self.heater_up_time_ratio*
                                              self.annual_heating_days*2.5*1.9) #1.9 kg natural gas per therm, 2.5 therm per hr fuel input
        
        ###LED lighting
        self.design_results['LED light'] = self.light_intensity*self.design_results['Aerial footage']/1100 #ea, 1100 PPF per light based on Clearas estimation
        self.self.construction[4].quantity += self.design_results['LED light'] * 10 * lb_to_kg #kg, aluminum as the main component, https://californialightworks.com/megadrive/megadrive-linear-400/
        self.design_results['LED mounting structure'] = 32.3*self.design_results['LED light'] #kg, 32.3 kg each based on Clearas estimation
        self.construction[3].quantity += self.design_results['LED mounting structure'] #kg, stainless steel
        self.construction[5].quantity = self.design_results['LED light'] * 0.012 #kg LED, 12 g LED per fixture
        
        ###Pigging & interconnects
        self.design_results['PIG assemblies PVC'] = 19907/6720*self.design_results['Aerial footage']*lb_to_kg #kg of PVC
        self.construction[1].quantity += self.design_results['PIG assemblies PVC']
        self.design_results['PIG assemblies stainless steel'] =1934/6720*self.design_results['Aerial footage']*lb_to_kg #kg of stainless steel
        self.construction[3].quantity += self.design_results['PIG assemblies stainless steel']

        self._init_lca()
        inf, RAA=self.ins
        eff, = self.outs

    def _cost(self):
        self._decorated_cost()
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
        return _annual_to_hourly_cost(hours * self.labor_wage)

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

    rr : float
        Recirculation ratio relative to influent flow, dimensionless.
        Default is 1.

    H_p : float
        Pressure head, [ft].
        Default is 0.

    L_s : float
        Suction pipe length, [ft].
        Default is 10 m converted to ft.

    N_ubends : int or None
        Number of U-bends per PBR set. If None, it is estimated as
        number of rows per set minus 1.
    labor_wage : float
        Labor wage in [USD/hr].
    '''
    _valid_pump_types = WWTpump._valid_pump_types + ('feed_PBR',)
    _ft_to_m = auom('ft').conversion_factor('m')
    _m_to_ft = 1 / _ft_to_m
    _g = 32.174  # gravitational acceleration, [ft/s2]
    F_BM_pump = 1.18*(1+0.007/100)
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
        'rr',       # dimensionless
        'H_p',      # [ft]
        'L_s',      # [ft]
        'N_ubends', # dimensionless, optional
    )

    def __init__(self, ID='', ins=None, outs=(), thermo=None,
                 init_with='WasteStream',
                 prefix='', pump_type='', Q_mgd=None, add_inputs=(),
                 InD=None, OD=None, hrt=4.,
                 length=None, v=None, rr=1.,
                 H_p=0., L_s=None, N_ubends=None,
                 capacity_factor=1.,
                 include_pump_cost=True, include_building_cost=False,
                 include_OM_cost=False,
                 F_BM=default_F_BM,
                 lifetime=default_equipment_lifetime,
                 labor_wage=0.,
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
        self.feed_PBR_rr = rr
        self.feed_PBR_H_p = H_p
        self.feed_PBR_L_s = 10 * self._m_to_ft if L_s is None else L_s
        self.feed_PBR_N_ubends = N_ubends
        self.labor_wage = _require_nonnegative('labor_wage', labor_wage)

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

        if i_lower == 'feed_PBR':
            warn(
                "`pump_type='feed_PBR'` requires PBR-specific inputs. "
                "Pass them as keyword arguments or through `add_inputs` in this order: "
                "InD [ft], OD [ft], hrt [hr], length [ft], v [ft/s], "
                "rr [-], H_p [ft], L_s [ft], N_ubends [-]. ",
                RuntimeWarning,
                stacklevel=3,
            )

        self._pump_type = i_lower

    def _get_feed_PBR_inputs(self, InD=None, OD=None, hrt=None,
                             length=None, v=None, rr=None,
                             H_p=None, L_s=None, N_ubends=None):
        values = {
            'InD': self.feed_PBR_InD,
            'OD': self.feed_PBR_OD,
            'hrt': self.feed_PBR_hrt,
            'length': self.feed_PBR_length,
            'v': self.feed_PBR_v,
            'rr': self.feed_PBR_rr,
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
            'rr': rr,
            'H_p': H_p,
            'L_s': L_s,
            'N_ubends': N_ubends,
        }

        for name, value in explicit_values.items():
            if value is not None:
                values[name] = value

        missing = [
            name for name in ('InD', 'OD', 'hrt', 'length', 'v', 'rr', 'H_p', 'L_s')
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
        if values['rr'] < 0:
            raise ValueError('`rr` must be non-negative.')
        if values['L_s'] < 0:
            raise ValueError('`L_s` must be non-negative, [ft].')

        return values

    def design_feed_PBR(self, Q_mgd=None, InD=None, OD=None, hrt=None,
                        length=None, v=None, rr=None,
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
            v=v, rr=rr, H_p=H_p, L_s=L_s, N_ubends=N_ubends,
        )

        InD = vals['InD']          # [ft]
        OD = vals['OD']            # [ft]
        hrt = vals['hrt']          # [hr]
        length = vals['length']    # [ft]
        v = vals['v']              # [ft/s]
        rr = vals['rr']            # [-]
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
            return 0., 0., 0.


        # Working volume = influent flow * HRT
        # Total length = working volume / tube cross sectional area
        working_volume = Q_cfh * hrt               # [ft3]
        total_length = working_volume / tube_area  # [ft]

        # Number of PBR sets, equivalent to N_pump in WWTpump logic.
        # Uses combined influent plus recirculation flow.
        N_pump = ceil(Q_cfs * (1 + rr) / tube_area / v)
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
        M_SS_pipe = 0.29 * V_s * _lb_to_kg          # [kg]

        # Stainless steel pump mass follows WWTpump's N_pump handling.
        M_SS_pump = N_pump * self.SS_per_pump                       # [kg]

        return M_SS_pipe, M_SS_pump, 0.

    def _cost(self):
        super()._cost()
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
            10 * getattr(self, 'N_pump', 0) * self.labor_wage
        )
