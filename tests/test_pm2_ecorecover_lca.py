#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from biosteam.units.compressor import IsothermalCompressor
from biosteam.units.design_tools import CEPCI_by_year
from qsdsan import WasteStream
from qsdsan import process_models as pc
from qsdsan.utils import auom, get_P_blower

from exposan.pm2_ecorecover_lca._sanunits import CO2Supply, Tank


@pytest.fixture
def tank():
    pc.create_pm2_cmps()
    feed = WasteStream('tank_feed', H2O=1000, units='kg/hr')
    return Tank(
        'T_test',
        ins=feed,
        outs='tank_effluent',
        V_max=100,
        W_tank=5,
        D_tank=4,
        freeboard=0.5,
        t_wall=0.3,
        t_slab=0.4,
        aeration=None,
        suspended_growth_model=None,
        isdynamic=False,
        include_aeration_power=False,
        include_mixing_power=False,
    )


@pytest.fixture
def co2_supply():
    pc.create_pm2_cmps()
    feed = WasteStream('co2_supply_feed')
    feed.set_flow_by_concentration(
        1000,
        {'S_CO2': 10},
        units=('L/hr', 'mg/L'),
    )
    return CO2Supply(
        'CO2_test',
        ins=feed,
        outs='co2_supply_effluent',
    )


def test_tank_design_geometry(tank):
    tank.simulate()
    D = tank.design_results
    assert D['Tank volume'] == pytest.approx(100)
    assert D['Tank length'] == pytest.approx(5)
    assert D['Volume of concrete wall'] == pytest.approx(
        2 * ((5 + 2 * 0.3) * 0.3 * 4.5) + 2 * (5 * 0.3 * 4.5)
    )
    assert D['Volume of concrete slab'] == pytest.approx(
        (5 + 2 * 0.3) * (5 + 2 * 0.3) * 0.4
    )
    assert D['Total volume'] == pytest.approx(125)


def test_tank_uses_biosteam_mix_tank_cost(tank):
    tank.simulate()
    assert tank.parallel['self'] >= 1
    assert tank.baseline_purchase_costs['Tank'] > 0


def test_tank_specific_mixing_power(tank):
    tank.include_mixing_power = True
    tank.kW_per_m3 = 0.01
    tank.simulate()
    assert tank.design_results['Mechanical mixing power'] == pytest.approx(1)
    assert tank.power_utility.rate == pytest.approx(1)


def test_tank_velocity_gradient_mixing_power(tank):
    tank.include_mixing_power = True
    tank.mixing_intensity = 100
    tank.simulate()
    expected = tank._mixed.mu * 100**2 / 1000 * tank.V_max
    assert tank.design_results['Mechanical mixing power'] == pytest.approx(expected)
    assert tank.power_utility.rate == pytest.approx(expected)


def test_tank_explicit_airflow_power(tank):
    tank.include_aeration_power = True
    tank.Q_air = 1440
    tank.simulate()
    expected = get_P_blower(1)
    assert tank.design_results['Aeration power'] == pytest.approx(expected)
    assert tank.power_utility.rate == pytest.approx(expected)


def test_tank_diffused_aeration_airflow_power(tank):
    aeration = pc.DiffusedAeration(
        'tank_aeration', 'S_O2', V=tank.V_max, Q_air=2880
    )
    tank.aeration = aeration
    tank.Q_air = None
    tank.include_aeration_power = True
    tank.simulate()
    assert tank.design_results['Aeration power'] == pytest.approx(get_P_blower(2))


def test_tank_combines_selected_power_terms(tank):
    tank.include_aeration_power = True
    tank.include_mixing_power = True
    tank.Q_air = 1440
    tank.kW_per_m3 = 0.01
    tank.simulate()
    assert tank.power_utility.rate == pytest.approx(get_P_blower(1) + 1)


def test_tank_can_disable_both_power_terms(tank):
    tank.simulate()
    assert tank.power_utility.rate == 0


def test_tank_defaults_airflow_from_tank_volume(tank):
    tank.include_aeration_power = True
    tank.simulate()
    expected_Q_air = 0.1 * tank.V_max * 1440
    assert tank.design_results['Aeration power'] == pytest.approx(
        get_P_blower(expected_Q_air / 1440)
    )


def test_tank_adds_aeration_equipment_when_enabled(tank):
    tank.include_aeration_power = True
    tank.Q_air = 1440
    tank.diffuser_unit_cost = 12
    tank.compressor_specific_mass = 5
    tank.simulate()

    D = tank.design_results
    power = get_P_blower(1)
    hp = auom('kW').convert(power, 'hp')
    algorithm = IsothermalCompressor.baseline_cost_algorithms['Screw']
    expected_compressor_cost = (
        CEPCI_by_year[2022] / algorithm.CE * algorithm.cost(hp)
    )
    expected_area = tank.W_tank * D['Tank length']

    assert D['Air flow rate'] == pytest.approx(1440)
    assert D['Air flow rate at compressor'] == pytest.approx(
        auom('m3/d').convert(1440, 'cfm')
    )
    assert D['Number of air compressors'] == 1
    assert D['Air compressor stainless steel'] == pytest.approx(power * 5)
    assert D['Diffuser area'] == pytest.approx(expected_area)
    assert D['Diffuser stainless steel'] == pytest.approx(
        expected_area * 181 / 18.5
    )
    assert tank.baseline_purchase_costs['Air compressor'] == pytest.approx(
        expected_compressor_cost
    )
    assert tank.baseline_purchase_costs['Diffusers'] == pytest.approx(
        expected_area * 12
    )
    assert tank.F_M['Air compressor'] == pytest.approx(2.5)
    assert tank.F_BM['Air compressor'] == pytest.approx(2.15)


def test_tank_sizes_parallel_air_compressors(tank):
    tank.include_aeration_power = True
    tank.Q_air = auom('cfm').convert(40001, 'm3/d')
    tank.simulate()
    assert tank.design_results['Number of air compressors'] == 3


def test_tank_omits_aeration_equipment_when_disabled(tank):
    tank.diffuser_unit_cost = 12
    tank.compressor_specific_mass = 5
    tank.simulate()
    D = tank.design_results
    assert D['Number of air compressors'] == 0
    assert D['Air compressor stainless steel'] == 0
    assert D['Diffuser area'] == 0
    assert D['Diffuser stainless steel'] == 0
    assert 'Air compressor' not in tank.baseline_purchase_costs
    assert 'Diffusers' not in tank.baseline_purchase_costs


def test_tank_handles_zero_airflow_without_compressor_cost(tank):
    tank.include_aeration_power = True
    tank.Q_air = 0
    tank.simulate()
    assert tank.design_results['Number of air compressors'] == 0
    assert tank.baseline_purchase_costs['Air compressor'] == 0


@pytest.mark.parametrize(
    'name',
    ('diffuser_unit_cost', 'compressor_specific_mass'),
)
def test_tank_rejects_negative_aeration_equipment_factors(tank, name):
    with pytest.raises(ValueError, match=name):
        setattr(tank, name, -1)


def test_tank_retains_dynamic_cstr_behavior():
    pc.create_pm2_cmps()
    feed = WasteStream('dynamic_tank_feed', H2O=1000, units='kg/hr')
    tank = Tank(
        'T_dynamic',
        ins=feed,
        outs='dynamic_tank_effluent',
        V_max=10,
        aeration=None,
        suspended_growth_model=None,
        include_aeration_power=False,
        include_mixing_power=False,
    )
    tank.simulate(t_span=(0, 0.01), method='BDF')
    assert tank._mock_dyn_sys.scope.sol.success


def test_co2_supply_passes_stream_and_calculates_default_makeup(co2_supply):
    feed = co2_supply.ins[0]
    co2_supply.simulate()
    effluent = co2_supply.outs[0]
    expected_base = (30 - 10) * 1000 * 1e-6
    expected_supply = expected_base * 1.10

    assert effluent.F_mass == pytest.approx(feed.F_mass)
    assert effluent.F_vol == pytest.approx(feed.F_vol)
    assert effluent.imass['S_CO2'] == pytest.approx(feed.imass['S_CO2'])
    assert co2_supply.design_results[
        'Influent CO2 concentration'
    ] == pytest.approx(10)
    assert co2_supply.design_results[
        'Target CO2 concentration'
    ] == pytest.approx(30)
    assert co2_supply.design_results['Wastewater flow'] == pytest.approx(1000)
    assert co2_supply.design_results['Base CO2 makeup'] == pytest.approx(
        expected_base
    )
    assert co2_supply.design_results['CO2 supply'] == pytest.approx(
        expected_supply
    )
    assert co2_supply.design_results['Excess CO2 fraction'] == pytest.approx(
        0.10
    )


def test_co2_supply_uses_user_excess_fraction(co2_supply):
    co2_supply.excess_fraction = 0.25
    co2_supply.simulate()
    assert co2_supply.design_results['CO2 supply'] == pytest.approx(
        (30 - 10) * 1000 * 1e-6 * 1.25
    )


def test_co2_supply_has_zero_makeup_above_target(co2_supply):
    co2_supply.ins[0].set_flow_by_concentration(
        1000,
        {'S_CO2': 35},
        units=('L/hr', 'mg/L'),
    )
    co2_supply.simulate()
    assert co2_supply.design_results['Base CO2 makeup'] == 0
    assert co2_supply.design_results['CO2 supply'] == 0


def test_co2_supply_cost_uses_2022_cepci(co2_supply):
    co2_supply.simulate()
    supply = co2_supply.design_results['CO2 supply']
    expected_price = (
        45 / 1000
        * CEPCI_by_year[2022] / CEPCI_by_year[2016]
    )
    assert co2_supply.add_OPEX['CO2 supply'] == pytest.approx(
        supply * expected_price
    )
    assert not co2_supply.baseline_purchase_costs


def test_co2_supply_handles_zero_flow():
    pc.create_pm2_cmps()
    feed = WasteStream('empty_co2_supply_feed')
    unit = CO2Supply(
        'CO2_empty',
        ins=feed,
        outs='empty_co2_supply_effluent',
    )
    unit.simulate()
    assert unit.design_results['Wastewater flow'] == 0
    assert unit.design_results['CO2 supply'] == 0
    assert unit.add_OPEX['CO2 supply'] == 0


@pytest.mark.parametrize(
    ('name', 'value'),
    (
        ('target_CO2', -1),
        ('excess_fraction', -0.1),
        ('CO2_price', -1),
    ),
)
def test_co2_supply_rejects_negative_inputs(co2_supply, name, value):
    with pytest.raises(ValueError, match=name):
        setattr(co2_supply, name, value)


def test_co2_supply_rejects_invalid_component():
    pc.create_pm2_cmps()
    feed = WasteStream('invalid_co2_supply_feed', H2O=1, units='kg/hr')
    with pytest.raises(ValueError, match='CO2_ID'):
        CO2Supply(
            'CO2_invalid',
            ins=feed,
            outs='invalid_co2_supply_effluent',
            CO2_ID='missing_CO2',
        )
