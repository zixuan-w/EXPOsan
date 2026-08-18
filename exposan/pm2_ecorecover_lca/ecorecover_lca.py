# -*- coding: utf-8 -*-
'''
EXPOsan: Exposition of sanitation and resource recovery systems

This module is developed by:
    Ga-Yeong Kim <gayeong1225@gmail.com>
    Zixuan Wang <wyatt4428@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/QSDsan/blob/main/LICENSE.txt
for license details.
'''

import os, numpy as np
import biosteam as bst
from biosteam.units.design_tools import CEPCI_by_year
from qsdsan import unit_operations as su
from qsdsan import processes as pc
from qsdsan import WasteStream, System, TEA as QSDsanTEA, LCA, ImpactItem
from qsdsan.utils import (
    ospath,
    time_printer,
    load_data,
    get_SRT,
    auom,
    ExogenousDynamicVariable as EDV,
)

from exposan.pm2_ecorecover_lca import (
    data_path,
    results_path,
    create_components,
    _load_lca_data,
    discount_rate,
    price_dct,
    tea_lifetime,
    tea_uptime_ratio,
    update_resource_settings,
    )
from exposan.pm2_ecorecover_lca._sanunits import (
    AlgaeCentrifuge,
    CO2Supply,
    Ecorecoverypump,
    HRTCSTR,
    LCAResourceInput,
    Mixtank,
    Photobioreactor,
    Ultrafiltration,
    )

__all__ = (
    'batch_init',
    'create_system',
    'EcorecoverTEA',
    'sync_dynamic_volumes',
    'run',
    )

_system_write_state = System._write_state
_lca_only_streams_by_system = {}
_extra_lca_streams_by_system = {}

def _write_state_skip_pm2_lca_only(system):
    '''
    Write dynamic states back to process streams while skipping LCA-only streams.

    LCA resource streams such as natural gas and cleaning chemicals should
    keep their design-calculated flows and should not be overwritten by the PM2
    dynamic state writer. This function temporarily replaces BioSTEAM's normal
    state writer during simulation to protect those bookkeeping streams.
    '''
    lca_only_streams = _lca_only_streams_by_system.get(id(system), ())
    if not lca_only_streams:
        return _system_write_state(system)
    feeds = system.feeds
    for ws in system.streams:
        if ws in feeds:
            continue
        if ws in lca_only_streams:
            continue
        ws._state2flows()

def batch_init(path, sheet, units=None, init_conds=None):
    '''
    Initialize dynamic reactor concentrations from a spreadsheet.

    PM2 simulations need initial concentrations for each CSTR block. This
    helper reads those values from the named workbook sheet and applies them to
    the PM2 tanks/PBRs, optionally overriding selected concentrations with
    ``init_conds``.
    '''
    df = load_data(path, sheet)
    dct = df.to_dict('index')
    init_conds = {} if init_conds is None else dict(init_conds)

    if units is None:
        units = [globals()[ID] for ID in (
            'MIX', 'PBR1', 'PBR2', 'PBR3', 'PBR4', 'PBR5',
            'PBR6', 'PBR7', 'PBR8', 'PBR9', 'PBR10', 'PBR11',
            'PBR12', 'PBR13', 'PBR14', 'PBR15', 'PBR16', 'PBR17',
            'PBR18', 'PBR19', 'PBR20', 'MEV', 'RET',
            )]

    for k in units:
        dct_i = dict(dct[k._ID[:3]])
        dct_i.update(init_conds)
        k.set_init_conc(**dct_i)

def _get_dynamic_influent_flow_m3_hr(data_file):
    '''Return the mean dynamic influent flow rate [m3/hr].'''
    data = np.genfromtxt(data_file, names=True, delimiter='\t')
    try:
        Q = np.atleast_1d(data['Q']).astype(float)
    except (KeyError, ValueError):
        raise ValueError(f'No Q column found in {data_file!r}.') from None
    Q = Q[np.isfinite(Q)]
    if not Q.size or np.mean(Q) <= 0:
        raise ValueError(f'No positive dynamic influent flow in {data_file!r}.')
    return float(np.mean(Q)) / 24

class EcorecoverTEA(QSDsanTEA):
    '''
    TEA class that applies a system-level factor on top of unit installed costs 
    to account for electronic controls, consultation fees, site preparation, and 
    administration, etc. 

    This is intentionally separate from BioSTEAM/QSDsan ``lang_factor``. A
    ``lang_factor`` estimates installed cost from purchase cost and bypasses
    unit-specific bare-module factors. ``additional_cost_factor`` keeps the
    unit-level installed costs, including all unit ``F_BM`` values, and then
    multiplies their sum:

    installed_equipment_cost = additional_cost_factor * sum(unit.installed_cost)
    '''
    __slots__ = ('_additional_cost_factor',)

    def __init__(self, *args, additional_cost_factor=1., **kwargs):
        self.additional_cost_factor = additional_cost_factor
        super().__init__(*args, **kwargs)

    @property
    def additional_cost_factor(self):
        return self._additional_cost_factor

    @additional_cost_factor.setter
    def additional_cost_factor(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError('`additional_cost_factor` must be positive.')
        self._additional_cost_factor = value

    @property
    def installed_equipment_cost(self):
        if self._CAPEX:
            return self._CAPEX
        if self.lang_factor:
            return self.purchase_cost * self.lang_factor
        return self.additional_cost_factor * sum(
            u.installed_cost for u in self.units
        )

def create_system(ID='EcoRecovery', init_conds=None, set_global=True,
                  include_tea=True, include_lca=True,
                  lifetime=tea_lifetime, uptime_ratio=tea_uptime_ratio,
                  additional_cost_factor=2.):
    '''Create the PM2 EcoRecover LCA system.

    Parameters
    ----------
    ID : str
        System ID.
    init_conds : dict, optional
        Optional updates to the default PM2 initial concentrations [mg/L].
    set_global : bool
        Whether to expose created streams, units, ``pm2``, ``cmps``, and ``eco``
        as module-level globals for interactive workflows.
    include_tea : bool
        Whether to create and attach a QSDsan TEA object.
    include_lca : bool
        Whether to load LCA data, attach stream impact items, and create a
        QSDsan LCA object.
    lifetime : float
        TEA/LCA project lifetime [yr].
    uptime_ratio : float
        Fraction of time the system operates [0-1].
    additional_cost_factor : float
        System-level multiplier applied to the sum of unit installed costs in
        TEA. This keeps unit-specific ``F_BM`` values active.
    '''
    init_conds = {} if init_conds is None else dict(init_conds)
    if include_lca:
        _load_lca_data()

    #%%
    # =============================================================================
    # Phototrophic-Mixotrophic Process Model (PM2)
    # =============================================================================

    ############# load components #############
    cmps = pc.create_pm2_cmps()
    resource_cmps = create_components(set_thermo=False)

    ############# create WasteStream objects #################
    # Q = 449.06           # influent flowrate [m3/d]
    Temp = 286.08          # temperature [K]
    dynamic_influent_file = ospath.join(data_path, 'dynamic_influent.tsv')

    T, I = EDV.batch_init(os.path.join(data_path, 'exo_vars_dynamic_influent.xlsx'), 'linear')

    T_mix = T
    I_mix = EDV('light_I_mix', function=lambda t: 0)

    DYINF = WasteStream('Dynamic_influent', T=Temp)
    PHO = WasteStream('To_PBR', T=Temp)
    DYINF_MIX = WasteStream('Dynamic_influent_to_MIX', T=Temp)
    MIXED = WasteStream('Mixed_influent', T=Temp)
    CO2_ADJUSTED = WasteStream('CO2_adjusted_mixed_influent', T=Temp)
    CO2_SUPPLY = WasteStream('CO2_supply', T=Temp)
    ME = WasteStream('To_membrane', T=Temp)
    INT = WasteStream('Internal_recycle', T=Temp)
    ME_RAW = WasteStream('To_membrane_unpumped', T=Temp)
    INT_RAW = WasteStream('Internal_recycle_unpumped', T=Temp)
    TE = WasteStream('Effluent', T=Temp)
    RETEN = WasteStream('Retentate', T=Temp)
    TE_RAW = WasteStream('Effluent_unpumped', T=Temp)
    RE = WasteStream('To_return_tank', T=Temp)
    CE = WasteStream('To_centrifuge', T=Temp)
    RE_RAW = WasteStream('To_return_tank_unpumped', T=Temp)
    CE_RAW = WasteStream('To_centrifuge_unpumped', T=Temp)
    CEN = WasteStream('Centrate', T=Temp)
    ALG = WasteStream('Harvested_biomass', T=Temp)
    CEN_RAW = WasteStream('Centrate_unpumped', T=Temp)
    RAA = WasteStream('Return_activated_algae', T=Temp)
    RAA_RAW = WasteStream('Return_activated_algae_unpumped', T=Temp)
    CITRIC_ACID = WasteStream('Citric_acid', T=Temp, thermo=resource_cmps)
    SODIUM_HYPOCHLORITE = WasteStream('Sodium_hypochlorite', T=Temp, thermo=resource_cmps)
    NATURAL_GAS = WasteStream('Natural_gas', phase='g', T=Temp, thermo=resource_cmps)
    CITRIC_ACID_TO_MEM = WasteStream('Citric_acid_to_membrane', T=Temp, thermo=resource_cmps)
    SODIUM_HYPOCHLORITE_TO_MEM = WasteStream('Sodium_hypochlorite_to_membrane', T=Temp, thermo=resource_cmps)
    NATURAL_GAS_DUMMY = WasteStream('Natural_gas_dummy', phase='g', T=Temp, thermo=resource_cmps)
    CITRIC_ACID_DUMMY = WasteStream('Citric_acid_dummy', T=Temp, thermo=resource_cmps)
    SODIUM_HYPOCHLORITE_DUMMY = WasteStream('Sodium_hypochlorite_dummy', T=Temp, thermo=resource_cmps)
    ALG_RAW = WasteStream('Harvested_biomass_unpumped', T=Temp)
    extra_lca_streams = ()
    if include_lca:
        _attach_resource_stream_impact_items(ID, {
            NATURAL_GAS: 'NaturalGas',
            CITRIC_ACID: 'CitricAcid',
            SODIUM_HYPOCHLORITE: 'SodiumHypochlorite',
            CO2_SUPPLY: 'CO2',
            })
        _attach_effluent_eutrophication_item(ID, TE_RAW)
        extra_lca_streams = (TE_RAW,)

    #%%
    ############# load and tailor process models #############
    Q_in_hr = _get_dynamic_influent_flow_m3_hr(dynamic_influent_file)
    mix_hrt = 3.3
    pbr_hrt = 4.
    mem_hrt = 0.377
    ret_hrt = 0.334
    pbr_n_series = 20

    pm2 = pc.PM2(arr_e=6663.36141724313, K_P=6.06569854392092, f_CH_max=9.60813888591872, exponent=7.56541058257826, 
                 q_CH=1.92792246509906, q_LI=26.1535941900048, V_NH=0.150722549179019, V_P=0.540050768528713,
                  a_c=0.049, I_n=1500, arr_a=1.8e10, beta_1=2.90,
                  beta_2=3.50, b_reactor=0.03, I_opt=2000, k_gamma=1e-5,
                  K_N=0.1, K_A=6.3, K_G=6.3, rho=1.186, K_STO=1.566,
                  f_LI_max=3.249, m_ATP=10,
                  mu_max=1.969, Q_N_max=0.417, Q_N_min=0.082, Q_P_max=0.092, Q_P_min=0.0163,
                  V_NO=0.003, n_dark=0.7,
                  Y_ATP_PHO=55.073, Y_CH_PHO=0.754, Y_LI_PHO=0.901, Y_X_ALG_PHO=0.450,
                  Y_ATP_HET_ACE=39.623, Y_CH_NR_HET_ACE=0.625, Y_CH_ND_HET_ACE=0.600,
                  Y_LI_NR_HET_ACE=1.105, Y_LI_ND_HET_ACE=0.713, Y_X_ALG_HET_ACE=0.216,
                  Y_ATP_HET_GLU=58.114, Y_CH_NR_HET_GLU=0.917, Y_CH_ND_HET_GLU=0.880,
                  Y_LI_NR_HET_GLU=1.620, Y_LI_ND_HET_GLU=1.046, Y_X_ALG_HET_GLU=0.317)  # ecorecover_cali (optuna results) seed777

    #%%
    ############# create unit operations #####################

    # Dynamic influent
    SE = su.DynamicInfluent('SE', outs=[DYINF],
                            data_file=dynamic_influent_file)
    P_DYINF = Ecorecoverypump(
        'P_DYINF', ins=SE-0, outs=[DYINF_MIX], thermo=None,
        init_with='WasteStream', prefix='', pump_type='', Q_mgd=None,
        add_inputs=(1, 10, 10, 0, 0),
        InD=None, OD=None, hrt=4., length=None, v=None,
        H_p=0., L_s=None, N_ubends=None, capacity_factor=4.,
        include_pump_cost=True, include_building_cost=False,
        include_OM_cost=True, F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        )
    #    1,   # N_pump: number of pumps    
    #  10,  # L_s: suction pipe length [ft]
    #     10,  # L_d: discharge pipe length [ft]
    #     0,   # H_ts: total static head [ft]
    #     0,   # H_p: pressure head [ft]

    MIX = Mixtank(
        'MIX', ins=[P_DYINF-0, RAA], outs=[MIXED], thermo=None,
        init_with='WasteStream', split=None,
        hrt=mix_hrt,
        sizing_flow=Q_in_hr,
        W_tank=6.4, D_tank=3.65, freeboard=0.61,
        t_wall=None, t_slab=None, aeration=None, DO_ID='S_O2',
        suspended_growth_model=pm2, gas_stripping=False,
        gas_IDs=None, stripping_kLa_min=None, K_Henry=None,
        D_gas=None, p_gas_atm=None, isdynamic=True,
        exogenous_vars=(T_mix, I_mix), V_wf=0.8,
        vessel_type='Field erected', vessel_material='Stainless steel',
        include_aeration_power=True, include_mixing_power=True, Q_air=None,
        mixing_intensity=None, kW_per_m3=0.0985,
        blower_T=20, P_atm=101.325, P_inlet_loss=1,
        P_diffuser_loss=7, h_submergance=5.18,
        blower_efficiency=0.7, blower_K=0.283,
        unit_diffuser_flow_rate=61.2, diffuser_unit_cost=445.,
        )

    CO2 = CO2Supply(
        'CO2', ins=[MIX-0, CO2_SUPPLY], outs=[CO2_ADJUSTED], thermo=None,
        init_with='WasteStream', target_CO2=30., influent_CO2=20.,
        excess_fraction=0.10, CO2_ID='S_CO2', CO2_price=45.,
        )

    P_DYINF_TO_PBR = Ecorecoverypump(
        'P_DYINF_TO_PBR', ins=CO2-0, outs=[PHO], thermo=None,
        init_with='WasteStream', prefix='', pump_type='feed_PBR',
        Q_mgd=None, add_inputs=(), InD=None, OD=None, hrt=4.,
        length=None, v=None, H_p=0., L_s=None, N_ubends=None,
        capacity_factor=4., include_pump_cost=True,
        include_building_cost=False, include_OM_cost=True,
        F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        )
    
    PBR1 = Photobioreactor(
        'PBR1', ins=P_DYINF_TO_PBR-0, outs=(), thermo=None,
        init_with='WasteStream', split=None,
        aeration=None, DO_ID='S_O2',
        suspended_growth_model=pm2, gas_stripping=False,
        gas_IDs=None, stripping_kLa_min=None, K_Henry=None,
        D_gas=None, p_gas_atm=None, isdynamic=True, exogenous_vars=(T, I),
        InD=0.1, OD=0.105, spt=1.67, hrt=pbr_hrt,
        spacing=1.05, support_spacing=2, length=80, v=0.8,
        design_flow_stream=SE-0,
        sizing_flow=Q_in_hr,
        n_series=pbr_n_series,
        light_intensity=100, annual_heating_days=150,
        heater_up_time_ratio=0.4, rho=1000,
        include_construction=True,
        glass_tube_lifetime=30., greenhouse_maintenance_ratio=0.016,
        LED_lifetime=12., support_structure_maintenance_ratio=0.016,
        )

    PBR1_NATURAL_GAS = LCAResourceInput(
        'PBR1_NATURAL_GAS', ins=NATURAL_GAS, outs=[NATURAL_GAS_DUMMY],
        thermo=resource_cmps, linked_unit=PBR1,
        design_key='natural gas', component_ID='NaturalGas',
        flow_unit='kg/hr', init_with='WasteStream',
        )
    PBR2 = HRTCSTR('PBR2', ins=PBR1-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR3 = HRTCSTR('PBR3', ins=PBR2-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR4 = HRTCSTR('PBR4', ins=PBR3-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR5 = HRTCSTR('PBR5', ins=PBR4-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR6 = HRTCSTR('PBR6', ins=PBR5-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR7 = HRTCSTR('PBR7', ins=PBR6-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR8 = HRTCSTR('PBR8', ins=PBR7-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR9 = HRTCSTR('PBR9', ins=PBR8-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR10 = HRTCSTR('PBR10', ins=PBR9-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR11 = HRTCSTR('PBR11', ins=PBR10-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR12 = HRTCSTR('PBR12', ins=PBR11-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR13 = HRTCSTR('PBR13', ins=PBR12-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR14 = HRTCSTR('PBR14', ins=PBR13-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR15 = HRTCSTR('PBR15', ins=PBR14-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR16 = HRTCSTR('PBR16', ins=PBR15-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR17 = HRTCSTR('PBR17', ins=PBR16-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR18 = HRTCSTR('PBR18', ins=PBR17-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR19 = HRTCSTR('PBR19', ins=PBR18-0, hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    PBR20 = HRTCSTR('PBR20', ins=PBR19-0, outs=[ME_RAW, INT_RAW], split=[0.52, 0.48],
                  hrt=pbr_hrt, sizing_flow=Q_in_hr, n_series=pbr_n_series,
                  aeration=None, suspended_growth_model=pm2, exogenous_vars=(T, I))
    for PBR in (PBR2, PBR3, PBR4, PBR5, PBR6, PBR7, PBR8, PBR9, PBR10,
                PBR11, PBR12, PBR13, PBR14, PBR15, PBR16, PBR17, PBR18, PBR19, PBR20):
        PBR.hrt = pbr_hrt

    P_ME = Ecorecoverypump(
        'P_ME', ins=PBR20-0, outs=[ME], thermo=None,
        init_with='WasteStream', prefix='', pump_type='', Q_mgd=None,
        add_inputs=(1, 10, 10, 0, 0),
        InD=None, OD=None, hrt=4., length=None, v=None,
        H_p=0., L_s=None, N_ubends=None, capacity_factor=4.,
        include_pump_cost=True, include_OM_cost = True, include_building_cost=False,
        F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        )
    P_INT = Ecorecoverypump(
        'P_INT', ins=PBR20-1, outs=[INT], thermo=None,
        init_with='WasteStream', prefix='', pump_type='', Q_mgd=None,
        add_inputs=(1, 10, 10, 0, 0),
        InD=None, OD=None, hrt=4., length=None, v=None,
        H_p=0., L_s=None, N_ubends=None, capacity_factor=4.,
        include_pump_cost=True, include_building_cost=False,
        include_OM_cost=True, F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        )

    MEM = Ultrafiltration(
        'MEM', ins=P_ME-0,
        outs=[TE_RAW, RETEN], thermo=None, split=0.39*(1-cmps.x),
        order=None, init_with='WasteStream', F_BM_default=None,
        isdynamic=False, R_t=2.12e12, T=25., TMP=15.4e3,
        capacity_factor=1.5, include_tank=True,
        include_sparging=True, include_chemical_cleaning=True,
        hrt=mem_hrt,
        sizing_flow=Q_in_hr, V_wf=0.8, vessel_type='Conventional',
        vessel_material='Stainless steel',
        specific_sparging_air_demand=0.2, blower_T=20,
        P_atm=101.325, P_inlet_loss=1, P_diffuser_loss=7,
        h_submergance=5.18, blower_efficiency=0.7,
        blower_K=0.283, unit_diffuser_flow_rate=61.2,
        diffuser_unit_cost=445.,
        NaOCl_cleaning_frequency_full=0.03287671232876712,
        NaOCl_cleaning_frequency_maintenance=0.2857142857142857,
        citric_acid_cleaning_frequency_maintenance=0.08786692759295499,
        citric_acid_unit_price=1.8595038517821776,
        sodium_hypochlorite_unit_price=14.265088213315847,
        membrane_lifetime=5,
        )

    P_CITRIC_ACID = Ecorecoverypump(
        'P_CITRIC_ACID', ins=CITRIC_ACID, outs=[CITRIC_ACID_TO_MEM],
        thermo=resource_cmps, init_with='WasteStream', prefix='',
        pump_type='chemical', Q_mgd=None, add_inputs=(),
        InD=None, OD=None, hrt=4., length=None, v=None,
        H_p=0., L_s=None, N_ubends=None, capacity_factor=4.,
        include_pump_cost=True, include_building_cost=False,
        include_OM_cost=False, F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        linked_unit=MEM, design_key='Citric acid usage',
        component_ID='CitricAcid', flow_unit='kg/hr',
        )
    
    P_SODIUM_HYPOCHLORITE = Ecorecoverypump(
        'P_SODIUM_HYPOCHLORITE', ins=SODIUM_HYPOCHLORITE,
        outs=[SODIUM_HYPOCHLORITE_TO_MEM], thermo=resource_cmps,
        init_with='WasteStream', prefix='', pump_type='chemical',
        Q_mgd=None, add_inputs=(), InD=None, OD=None, hrt=4.,
        length=None, v=None, H_p=0., L_s=None, N_ubends=None,
        capacity_factor=4., include_pump_cost=True,
        include_building_cost=False, include_OM_cost=False,
        F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        linked_unit=MEM, design_key='Sodium hypochlorite usage',
        component_ID='SodiumHypochlorite', flow_unit='kg/hr',
        )

    MEM_CITRIC_ACID = LCAResourceInput(
        'MEM_CITRIC_ACID', ins=P_CITRIC_ACID-0, outs=[CITRIC_ACID_DUMMY],
        thermo=resource_cmps, linked_unit=MEM,
        design_key='Citric acid usage', component_ID='CitricAcid',
        flow_unit='kg/hr', init_with='WasteStream',
        )
    MEM_SODIUM_HYPOCHLORITE = LCAResourceInput(
        'MEM_SODIUM_HYPOCHLORITE', ins=P_SODIUM_HYPOCHLORITE-0,
        outs=[SODIUM_HYPOCHLORITE_DUMMY], thermo=resource_cmps,
        linked_unit=MEM, design_key='Sodium hypochlorite usage',
        component_ID='SodiumHypochlorite', flow_unit='kg/hr',
        init_with='WasteStream',
        )

    P_TE = Ecorecoverypump(
        'P_TE', ins=MEM-0, outs=[TE], thermo=None,
        init_with='WasteStream', prefix='', pump_type='permeate_cross-flow',
        Q_mgd=None, add_inputs=(1, MEM.V_max, auom('Pa').convert(15.4e3, 'psi'), False),
        InD=None, OD=None, hrt=4., length=None, v=None,
        H_p=0., L_s=None, N_ubends=None, capacity_factor=4.,
        include_pump_cost=True, include_building_cost=False,
        include_OM_cost=False, F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        )

    MEV = su.CSTR('MEV', ins=MEM-1, V_max=MEM.V_max, aeration=None,
                  suspended_growth_model=None)

    POST_MEM = su.Splitter('POST_MEM', MEV-0, outs=[RE_RAW, CE_RAW],
                           split=0.97)                    # changed compared to previous ver.

    P_RE = Ecorecoverypump(
        'P_RE', ins=POST_MEM-0, outs=[RE], thermo=None,
        init_with='WasteStream', prefix='', pump_type='', Q_mgd=None,
        add_inputs=(1, 10, 10, 0, 0),
        InD=None, OD=None, hrt=4., length=None, v=None,
        H_p=0., L_s=None, N_ubends=None, capacity_factor=4.,
        include_pump_cost=True, include_building_cost=False,
        include_OM_cost=False, F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        )
    P_CE = Ecorecoverypump(
        'P_CE', ins=POST_MEM-1, outs=[CE], thermo=None,
        init_with='WasteStream', prefix='', pump_type='sludge', Q_mgd=None,
        add_inputs=(), InD=None, OD=None, hrt=4., length=None, v=None,
        H_p=0., L_s=None, N_ubends=None, capacity_factor=4.,
        include_pump_cost=True, include_building_cost=False,
        include_OM_cost=False, F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        )
    CENT = AlgaeCentrifuge(
        'CENT', ins=P_CE-0, outs=[CEN_RAW, ALG_RAW], thermo=None,
        split={'X_CHL':0.33,
               'X_ALG':0.33,
               'X_PG':0.33,
               'X_TAG':0.33,
               'S_CO2':0.975,
               'S_A':0.975,
               'S_G':0.975,
               'S_O2':0.975,
               'S_NH':0.975,
               'S_NO':0.975,
               'S_P':0.975,
               'X_N_ALG':0.33,
               'X_P_ALG':0.33,
               'H2O':0.975},
        order=None, init_with='WasteStream', F_BM_default=None,
        isdynamic=False, capacity_factor=4., algal_cell_diameter=5e-6,
        algal_particle_density=1050., water_density=1000.,
        T=25., phi=0., vgm=0.1e-6, UF_flow=None,
        )

    P_ALG = Ecorecoverypump(
        'P_ALG', ins=CENT-1, outs=[ALG], thermo=None,
        init_with='WasteStream', prefix='', pump_type='sludge',
        Q_mgd=None, add_inputs=(), InD=None, OD=None, hrt=4.,
        length=None, v=None, H_p=0., L_s=None, N_ubends=None,
        capacity_factor=4., include_pump_cost=True,
        include_building_cost=False, include_OM_cost=False,
        F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        )

    P_CEN = Ecorecoverypump(
        'P_CEN', ins=CENT-0, outs=[CEN], thermo=None,
        init_with='WasteStream', prefix='', pump_type='', Q_mgd=None,
        add_inputs=(1, 10, 10, 0, 0),
        InD=None, OD=None, hrt=4., length=None, v=None,
        H_p=0., L_s=None, N_ubends=None, capacity_factor=4.,
        include_pump_cost=True, include_building_cost=False,
        include_OM_cost=False, F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        )

    RET = Mixtank(
        'RET', ins=[P_INT-0, P_RE-0, P_CEN-0], outs=[RAA_RAW], thermo=None,
        init_with='WasteStream', split=None,
        hrt=ret_hrt,
        sizing_flow=Q_in_hr,
        W_tank=6.4, D_tank=3.65, freeboard=0.61,
        t_wall=None, t_slab=None, aeration=None, DO_ID='S_O2',
        suspended_growth_model=None, gas_stripping=False,
        gas_IDs=None, stripping_kLa_min=None, K_Henry=None,
        D_gas=None, p_gas_atm=None, isdynamic=True,
        exogenous_vars=(), V_wf=0.8, vessel_type='Conventional',
        vessel_material='Stainless steel', include_aeration_power=False,
        include_mixing_power=True, Q_air=None, mixing_intensity=None,
        kW_per_m3=0.0985, blower_T=20, P_atm=101.325,
        P_inlet_loss=1, P_diffuser_loss=7, h_submergance=5.18,
        blower_efficiency=0.7, blower_K=0.283,
        unit_diffuser_flow_rate=61.2, diffuser_unit_cost=445.,
        )

    P_RAA = Ecorecoverypump(
        'P_RAA', ins=RET-0, outs=[RAA], thermo=None,
        init_with='WasteStream', prefix='', pump_type='recirculation_CSTR',
        Q_mgd=None, add_inputs=(RET.V_max,), InD=None, OD=None, hrt=4.,
        length=None, v=None, H_p=0., L_s=None, N_ubends=None,
        capacity_factor=4., include_pump_cost=True,
        include_building_cost=False, include_OM_cost=False,
        F_BM=Ecorecoverypump.default_F_BM,
        lifetime=Ecorecoverypump.default_equipment_lifetime,
        )

    #%%

    _init_conds = {
            'X_CHL':2.53,
            'X_ALG':505.97,
            'X_PG':22.99,
            'X_TAG':93.78,
            'S_CO2':30.0,
            'S_A':5.0,
            'S_G':5.0,
            'S_O2':5.0,
            'S_NH':35.80,
            'S_NO':0.7,
            'S_P':0.36,
            'X_N_ALG':3.23,
            'X_P_ALG':0.19,
        }

    batch_init(
        ospath.join(data_path, 'initial_conditions_pm2_dynamic_influent.xlsx'),
        'default',
        units=(MIX, PBR1, PBR2, PBR3, PBR4, PBR5, PBR6, PBR7, PBR8,
               PBR9, PBR10, PBR11, PBR12, PBR13, PBR14, PBR15, PBR16,
               PBR17, PBR18, PBR19, PBR20, MEV, RET),
        init_conds=init_conds,
        )
    if init_conds:
        _init_conds.update(init_conds)

    #%%
    eco = System(ID,
                 path=(SE, P_DYINF, MIX, CO2, P_DYINF_TO_PBR,
                       PBR1, PBR1_NATURAL_GAS, PBR2, PBR3, PBR4, PBR5, PBR6, PBR7, PBR8, PBR9, PBR10,
                       PBR11, PBR12, PBR13, PBR14, PBR15, PBR16, PBR17, PBR18, PBR19, PBR20,
                       P_ME, P_INT, MEM, P_CITRIC_ACID,
                       MEM_CITRIC_ACID, P_SODIUM_HYPOCHLORITE,
                       MEM_SODIUM_HYPOCHLORITE, P_TE, MEV, POST_MEM,
                       P_RE, P_CE, CENT, P_ALG, P_CEN, RET,
                       P_RAA),
                 recycle=(RAA,))

    eco.set_dynamic_tracker(SE, MIX, PBR1, PBR20, RET, TE, CEN, ALG)
    eco.set_tolerance(rmol=1e-6)
    bio_IDs = ('X_ALG',)
    if include_tea:
        EcorecoverTEA(system=eco, discount_rate=discount_rate, start_year=2022,
            lifetime=lifetime, uptime_ratio=uptime_ratio,
            lang_factor=None, annual_maintenance=0, annual_labor=0,
            additional_cost_factor=additional_cost_factor,
            simulate_system=False)
    if include_lca:
        def get_lifetime_electricity():
            return (
                sum(u.power_utility.rate for u in eco.units)
                * 24 * 365 * uptime_ratio * lifetime
            )
        lca = LCA(system=eco, lifetime=lifetime, lifetime_unit='yr',
                  uptime_ratio=uptime_ratio, annualize_construction=False,
                  simulate_system=False, e_item=get_lifetime_electricity)
        _extra_lca_streams_by_system[id(eco)] = extra_lca_streams
        _add_extra_lca_streams(lca, extra_lca_streams)

    if set_global:
        globals().update({
            k: v for k, v in locals().items()
            if k not in ('ID', 'init_conds', 'set_global',
                         'include_tea', 'include_lca')
            })
    return eco

def sync_dynamic_volumes(system=None):
    '''
    Update unit sizing during Monte Carlo simulation by synchronizing
    HRT-sized dynamic volumes before each PM2 simulation.

    Several units in this system calculate their dynamic working volume from a
    hydraulic retention time rather than from a fixed ``V_max``. This function
    updates those ``V_max`` values immediately before simulation so each Monte
    Carlo sample uses the current uncertain/design parameter values in the
    dynamic PM2 states.

    For each HRT-sized unit, the volume is updated as:

    ``V_max = hrt * Q / n_series``

    where ``Q`` is taken from the unit's ``_get_hrt_sizing_flow()`` method when
    available, otherwise from ``sizing_flow``, and finally from the current
    inlet volumetric flow. For single CSTR/tank units, ``n_series = 1``.

    The function currently updates:

    * ``MIX`` and ``RET`` tank volumes.
    * ``PBR1`` and PBR2-PBR20 volumes. PBR2-PBR20 inherit ``hrt``,
      ``sizing_flow``, and ``n_series`` from ``PBR1`` so the 20-CSTR plug-flow
      approximation stays internally consistent.
    * ``MEM`` tank volume, and ``MEV.V_max`` is matched to ``MEM.V_max`` when
      ``MEV`` exists.
    '''
    system = globals().get('eco') if system is None else system
    if system is None:
        return
    units = system.flowsheet.unit

    def _set_volume(unit, n_series=1):
        if hasattr(unit, '_get_hrt_sizing_flow'):
            Q = unit._get_hrt_sizing_flow()
        else:
            Q = getattr(unit, 'sizing_flow', None)
            if Q is None or Q <= 0:
                Q = sum(getattr(stream, 'F_vol', 0.) for stream in unit.ins)
        if Q and getattr(unit, 'hrt', None):
            unit.V_max = unit.hrt * Q / n_series

    for ID in ('MIX', 'RET'):
        if hasattr(units, ID):
            _set_volume(getattr(units, ID))

    if hasattr(units, 'PBR1'):
        PBR1 = units.PBR1
        n_series = getattr(PBR1, 'n_series', 20)
        _set_volume(PBR1, n_series=n_series)
        for n in range(2, n_series + 1):
            ID = f'PBR{n}'
            if not hasattr(units, ID):
                continue
            PBR = getattr(units, ID)
            PBR.hrt = PBR1.hrt
            PBR.sizing_flow = PBR1.sizing_flow
            PBR.n_series = n_series
            _set_volume(PBR, n_series=n_series)

    if hasattr(units, 'MEM'):
        _set_volume(units.MEM)
        if hasattr(units, 'MEV'):
            units.MEV.V_max = units.MEM.V_max

def _attach_resource_stream_impact_items(system_ID, stream_dct):
    '''Attach stream impact items to LCA-only resource feeds.'''
    price_dct, _, _ = update_resource_settings()
    for stream, resource_ID in stream_dct.items():
        source = ImpactItem.get_item(f'{resource_ID}_item')
        if source is None:
            raise RuntimeError(
                f'ImpactItem "{resource_ID}_item" is not loaded; '
                'call `_load_lca_data()` before creating LCA streams.'
            )
        item_ID = f'{system_ID}_{stream.ID}_item'
        item = ImpactItem.get_item(item_ID)
        if item is None:
            item = source.copy(item_ID, stream=stream, set_as_source=True)
        else:
            item.linked_stream = stream
        stream.price = price_dct.get(resource_ID, 0.)

def _component_element_mass_flow(stream, component_ID, element):
    '''Return elemental mass flow [kg/hr] for one component in a WasteStream.'''
    try:
        mass_flow = stream.imass[component_ID]
    except Exception:
        return 0.
    try:
        component = getattr(stream.components, component_ID)
    except Exception:
        return mass_flow
    factor = getattr(component, f'i_{element}', None)
    if factor is None:
        factor = 1. if getattr(component, 'measured_as', None) == element else 0.
    return mass_flow * factor


def _cod_mass_flow(stream):
    '''Return COD mass flow [kg COD/hr] from stream COD [mg/L] and F_vol [m3/hr].'''
    try:
        COD = stream.COD
        F_vol = stream.F_vol
    except Exception:
        return 0.
    if COD <= 0. or F_vol <= 0.:
        return 0.
    return COD * F_vol / 1000.


def _effluent_eutrophication_flow(stream, eutro_dct):
    '''
    Return direct effluent eutrophication load [kg N-eq/hr].

    Source: U.S. EPA 832-R-21-006/A, Life Cycle and Cost Assessments
    of Nutrient Removal Technologies in Wastewater Treatment Plants,
    p. 4-11.
    '''
    return (
        _component_element_mass_flow(stream, 'S_NH', 'N') * eutro_dct['S_NH']
        + _component_element_mass_flow(stream, 'S_NO', 'N') * eutro_dct['S_NO']
        + _component_element_mass_flow(stream, 'S_P', 'P') * eutro_dct['S_P']
        + _cod_mass_flow(stream) * eutro_dct['COD']
    )


def _attach_effluent_eutrophication_item(system_ID, stream):
    '''Attach an LCA-only eutrophication impact item to tertiary effluent.'''
    source = ImpactItem.get_item('EffluentEutrophication_item')
    if source is None:
        raise RuntimeError(
            'ImpactItem "EffluentEutrophication_item" is not loaded; '
            'call `_load_lca_data()` before creating LCA streams.'
        )
    item_ID = f'{system_ID}_{stream.ID}_eutrophication_item'
    item = ImpactItem.get_item(item_ID)
    if item is None:
        item = source.copy(item_ID, stream=stream, set_as_source=True)
    else:
        item.linked_stream = stream

    def flow_getter(ws):
        from exposan import pm2_ecorecover_lca as pmlca
        return _effluent_eutrophication_flow(ws, pmlca.EUTRO_dct)

    item.flow_getter = flow_getter
    stream.price = 0.


def _add_extra_lca_streams(lca, streams):
    '''Keep intermediate LCA-only streams visible to QSDsan LCA.'''
    if not streams:
        return
    lca_streams = set(lca.lca_streams)
    lca_streams.update(streams)
    lca._lca_streams = lca_streams


def _restore_lca_only_streams(system=None):
    '''Initialize standalone LCA-only resource units after dynamic reset.'''
    lca_only_streams = []

    system = globals().get('eco') if system is None else system
    if system is not None:
        for unit in system.units:
            if not isinstance(unit, LCAResourceInput):
                continue
            N = len(unit.components) + 1
            if hasattr(unit, '_ins_QC') and len(unit._ins_QC):
                unit._ins_QC[0, :] = 1.
                unit._ins_dQC[0, :] = 0.
            for stream in unit.ins:
                stream._state = np.ones(N)
                stream._dstate = np.zeros(N)
                lca_only_streams.append(stream)
            for stream in unit.outs:
                stream._state = np.ones(N)
                stream._dstate = np.zeros(N)
                lca_only_streams.append(stream)
        for attr in ('_feeds', '_products'):
            if hasattr(system, attr):
                delattr(system, attr)
        _lca_only_streams_by_system[id(system)] = tuple(lca_only_streams)


def _reset_dynamic_system():
    '''BioSTEAM reset hook that preserves LCA-only stream bookkeeping.'''
    eco.reset_cache()
    _restore_lca_only_streams(eco)

#%%

@time_printer
def run(t, t_step, method=None, print_t=False, **kwargs):
    '''
    Run the dynamic EcoRecover PM2 simulation and refresh TEA/LCA bookkeeping.

    This is the package-level simulation entry point. It synchronizes HRT-sized
    volumes, protects LCA-only streams from the dynamic state writer, runs the
    PM2 ODE simulation, resets LCA caches so construction/resource streams are
    current, and reports the estimated SRT.
    '''
    global eco
    try:
        eco
    except NameError:
        create_system()
    bst.CE = CEPCI_by_year[2022]
    sync_dynamic_volumes(eco)

    System._write_state = _write_state_skip_pm2_lca_only
    export_state_to = kwargs.pop('export_state_to', None)
    if method:
        try:
            simulate_kwargs = dict(kwargs)
            if export_state_to:
                simulate_kwargs['export_state_to'] = export_state_to
            eco.simulate(state_reset_hook=_reset_dynamic_system,
                          t_span=(0,t),
                          t_eval = np.arange(0, t+t_step, t_step),
                          method=method,
                          # rtol=1e-2,
                          # atol=1e-3,
                          print_t=print_t,
                          **simulate_kwargs)
        finally:
            System._write_state = _system_write_state
    else:
        try:
            eco.simulate(state_reset_hook=_reset_dynamic_system,
                          t_span=(0, t),
                          t_eval=np.arange(0, t+t_step/30, t_step/30),
                          method='LSODA',
                          print_msg=True,
                          print_t=print_t,
                          **kwargs)
        finally:
            System._write_state = _system_write_state
    # Reset LCA's cached construction/stream unit lists after simulation.
    # Some unit construction quantities are created or updated during _design(),
    # so LCA must rebuild these caches to include the latest construction items
    # and LCA-only resource streams.
    if hasattr(eco, '_LCA'):
        eco._LCA._construction_units = set()
        eco._LCA._transportation_units = set()
        eco._LCA._lca_streams = set()
        eco._LCA.system = eco
        _add_extra_lca_streams(
            eco._LCA, _extra_lca_streams_by_system.get(id(eco), ()))
    unit_IDs = [u.ID for u in eco.units if isinstance(u, su.CSTR)]
    srt = get_SRT(eco, bio_IDs, active_unit_IDs=unit_IDs)
    print(f'Estimated SRT assuming at steady state is {round(srt, 2)} days')

if __name__ == '__main__':
    t = 100
    t_step = 1
    # method = 'RK45'
    method = 'RK23'
    # method = 'DOP853'
    # method = 'Radau'
    # method = 'BDF'
    # method = 'LSODA'
    # method = None
    msg = f'Method {method}'
    print(f'\n{msg}\n{"-"*len(msg)}') # long live OCD!
    print(f'Time span 0-{t}d \n')
    run(t, t_step, method=method,
        print_t = True,
        )

# ALG.get_mass_concentration('mg/l','X_ALG')
# RET.scope.plot_time_series(('X_ALG'))

# simulation ends with "Estimated SRT assuming at steady state is 6.98 days"
