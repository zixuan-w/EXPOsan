#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
EXPOsan: Exposition of sanitation and resource recovery systems

This module is developed by:
    Zixuan Wang <wyatt4428@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/QSDsan/blob/main/LICENSE.txt
for license details.
'''

import os
import biosteam as bst
import numpy as np
import qsdsan as qs
from chaospy import distributions as shape
from qsdsan import ImpactItem, Metric, Model
from qsdsan.utils import auom, load_data, ospath, ExogenousDynamicVariable as EDV
from exposan.utils import run_uncertainty as run

from exposan import pm2_ecorecover_lca as pmlca
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

__all__ = ('create_model', 'run_uncertainty',)


_m_to_ft = auom('m').conversion_factor('ft')
_hp_to_kW = auom('hp').conversion_factor('kW')
_pump_motor_efficiency = None
_baseline_light_by_system = {}
_current_light_by_system = {}

# Uncertainty values transformed from
# Unit_process_assumptions_Ecorecover_081426.xlsx.
_PARAMETERS = (
    # mix tank tab. Applies to all Mixtank units (MIX and RET).
    # References and criteria are from the workbook columns.
    dict(group='Mixtank', name='kW_per_m3', attr='kW_per_m3',
         units='kW/m3', expected=5.5, low=3., high=8.,
         distribution='uniform',
         reference='expected: average of low & high; sources: '
                   'https://doi.org/10.2166/wst.2015.306; '
                   'ISBN: 9781260132274; '
                   'https://doi.org/10.1016/j.compag.2014.10.007; '
                   'criteria 2'),
    dict(group='Mixtank', name='mixing_intensity', attr='mixing_intensity',
         units='1/s', expected=65., low=50., high=80.,
         distribution='uniform',
         reference='expected: average of low & high; source: '
                   'https://doi.org/10.1016/j.compag.2014.10.007; '
                   'criteria 2'),
    dict(group='Mixtank', name='unit_diffuser_flow_rate',
         attr='unit_diffuser_flow_rate', units='m3/d/diffuser',
         expected=163., low=114.1, high=211.9, distribution='uniform',
         reference='source: '
                   'https://cleanwaterservices.org/wp-content/uploads/2025/10/04-TM12_ForestGroveWRRFAerationEvaluation.pdf; '
                   '+/-30%; criteria 3'),
    dict(group='Mixtank', name='diffuser_unit_cost',
         attr='diffuser_unit_cost', units='USD/diffuser',
         expected=447.7, low=313.4,
         high=582.1, distribution='uniform',
         reference='source: '
                   'https://cleanwaterservices.org/wp-content/uploads/2025/10/04-TM12_ForestGroveWRRFAerationEvaluation.pdf; '
                   '+/-30%; criteria 3'),
    dict(group='Mixtank', name='D_tank', attr='D_tank',
         units='m', expected=5.325, low=3.65, high=7.,
         distribution='uniform',
         reference='expected: average of low & high; expected source: '
                   'https://doi.org/10.1038/s41545-025-00537-4; '
                   'additional source: https://doi.org/10.2166/wst.1996.0333; '
                   'criteria 2'),
    dict(group='Mixtank', name='blower_efficiency',
         attr='blower_efficiency', units='-', expected=0.6, low=0.4,
         high=0.85, distribution='triangular',
         reference='sources: Compressors and Blowers: Maintenance, '
                   'Practical Guidance, Energy-Efficient; '
                   'https://www.algaefoundationatec.org/aces/download/Techno-Economic%20Analysis.pdf; '
                   'Clearas membrane documentation; criteria 4.2'),
    dict(group='Mixtank', unit='MIX', name='MIX hrt', attr='hrt',
         units='hr', expected=3.3, low=1.65, high=4.95,
         distribution='uniform',
         reference='MIX HRT varies +/-50% around the current system default; '
                   'dynamic volume is synchronized as V_max = hrt * sizing_flow.'),
    dict(group='Mixtank', unit='RET', name='RET hrt', attr='hrt',
         units='hr', expected=0.334, low=0.167, high=0.501,
         distribution='uniform',
         reference='RET HRT varies +/-50% around the current system default; '
                   'dynamic volume is synchronized as V_max = hrt * sizing_flow.'),

    # CO2 supply tab. Applies to the CO2Supply unit.
    dict(group='CO2Supply', name='target_CO2', attr='target_CO2',
         units='mg/L', expected=30., low=21., high=39.,
         distribution='uniform',
         reference='expected: https://doi.org/10.1038/s41545-025-00545-4; '
                   '+/-30%; criteria 3'),
    dict(group='CO2Supply', name='excess_fraction', attr='excess_fraction',
         units='-', expected=0.28, low=0.46, high=0.10,
         distribution='uniform',
         reference='expected: average of low & high; sources: '
                   'https://doi.org/10.1016/0196-8904(95)00340-1; '
                   'https://doi.org/10.1007/BF01025286; '
                   'https://www.osti.gov/servlets/purl/1613442; '
                   'criteria 2'),
    dict(group='CO2Supply', name='CO2_price', attr='CO2_price',
         units='USD/metric ton', expected=56.39381569965872,
         low=37.59587713310581, high=75.19175426621162,
         distribution='uniform',
         reference='expected: average of low & high; source: '
                   'https://doi.org/10.1002/ceat.201700210; criteria 2'),

    # UF tab. Applies to the Ultrafiltration unit (MEM).
    dict(group='Ultrafiltration', name='TMP', attr='TMP',
         units='Pa', expected=15400., low=2500., high=60000.,
         distribution='triangular',
         reference='expected: https://doi.org/10.1021/acs.est.3c10264; '
                   'additional sources: https://doi.org/10.1016/j.heliyon.2021.e07367; '
                   'https://doi.org/10.1016/j.scitotenv.2019.03.371; '
                   'https://doi.org/10.3390/pr14081206; criteria 4.2'),
    dict(group='Ultrafiltration', unit='MEM', name='MEM hrt', attr='hrt',
         units='hr', expected=0.377, low=0.1885, high=0.5655,
         distribution='uniform',
         reference='MEM HRT varies +/-50% around the current system default; '
                   'dynamic volume is synchronized as V_max = hrt * sizing_flow.'),
    dict(group='Ultrafiltration', name='specific_sparging_air_demand',
         attr='specific_sparging_air_demand', units='m3/m2/hr',
         expected=0.7, low=0.2, high=1.3, distribution='uniform',
         reference='expected: average of low & high; source: '
                   'https://doi.org/10.1016/j.scitotenv.2024.177273; '
                   'criteria 2'),
    dict(group='Ultrafiltration', name='NaOCl_cleaning_frequency_full',
         attr='NaOCl_cleaning_frequency_full', units='1/d',
         expected=0.03287671232876712, low=0.0136986301369863,
         high=0.0666666666666667, distribution='triangular',
         reference='UF tab full clean with NaOCl at 500 mg/L; expected source: '
                   'Clearas documentation, Page 14 of "Ultrafiltration System '
                   'for EcoRecover" (UF: Row 87); low source: '
                   'https://doi.org/10.1021/acsestengg.2c00184; high source: '
                   'https://doi.org/10.1016/j.jclepro.2025.145527; criteria 4.2'),
    dict(group='Ultrafiltration', name='NaOCl_cleaning_frequency_maintenance',
         attr='NaOCl_cleaning_frequency_maintenance', units='1/d',
         expected=0.2857142857142857, low=0.14285714285714285,
         high=6., distribution='triangular',
         reference='UF tab maintenance clean with NaOCl at 120 mg/L; expected '
                   'source: Clearas documentation, Page 14 of "Ultrafiltration '
                   'System for EcoRecover" (UF: Row 87); low/high sources: '
                   'https://doi.org/10.1080/19443994.2012.713739 and '
                   'https://doi.org/10.1016/j.jclepro.2025.145527; criteria 4.2'),
    dict(group='Ultrafiltration',
         name='citric_acid_cleaning_frequency_maintenance',
         attr='citric_acid_cleaning_frequency_maintenance', units='1/d',
         expected=0.08786692759295499, low=0.03287671232876712,
         high=0.14285714285714285, distribution='uniform',
         reference='UF tab maintenance clean with citric acid at 2000 mg/L; '
                   'expected is average of low & high; sources: '
                   'https://doi.org/10.3390/w17243453 and Clearas '
                   'documentation, Page 14 of "Ultrafiltration System for '
                   'EcoRecover" (UF: Row 87); criteria 2'),
    dict(group='Ultrafiltration', name='membrane_lifetime',
         attr='membrane_lifetime', units='yr', expected=5.,
         low=2., high=10., distribution='triangular',
         reference='sources: https://doi.org/10.1016/j.watres.2019.115212; '
                   'https://doi.org/10.1016/j.jclepro.2019.01.321; '
                   'https://doi.org/10.1021/acsestengg.2c00184; '
                   'https://doi.org/10.3390/membranes10080180; '
                   'https://doi.org/10.1016/j.scitotenv.2024.177273; '
                   'criteria 4.2'),
    dict(group='Ultrafiltration', name='blower_efficiency',
         attr='blower_efficiency', units='-', expected=0.6,
         low=0.4, high=0.85, distribution='triangular',
         reference='sources: Compressors and Blowers: Maintenance, '
                   'Practical Guidance, Energy-Efficient; '
                   'https://www.algaefoundationatec.org/aces/download/Techno-Economic%20Analysis.pdf; '
                   'Clearas membrane documentation; criteria 4.2'),
    dict(group='Ultrafiltration', name='diffuser_unit_cost',
         attr='diffuser_unit_cost', units='USD/diffuser',
         expected=447.7435265104809, low=313.4204685573366,
         high=582.0665844636252, distribution='uniform',
         reference='source: '
                   'https://cleanwaterservices.org/wp-content/uploads/2025/10/04-TM12_ForestGroveWRRFAerationEvaluation.pdf; '
                   '+/-30%; criteria 3'),

    # centrifuge tab. Applies to AlgaeCentrifuge.
    dict(group='AlgaeCentrifuge', name='algal_cell_diameter',
         attr='algal_cell_diameter', units='m', expected=6e-6,
         low=2e-6, high=1e-5, distribution='uniform',
         reference='expected: average of low & high; sources: '
                   'https://doi.org/10.1016/j.bej.2020.107741; '
                   'https://doi.org/10.1016/j.biotechadv.2006.11.002; '
                   'https://doi.org/10.1016/j.enconman.2016.06.060; '
                   'criteria 2'),
    dict(group='AlgaeCentrifuge', name='algal_particle_density',
         attr='algal_particle_density', units='kg/m3', expected=1080.,
         low=1020., high=1140., distribution='uniform',
         reference='expected: average of low & high; source: '
                   'https://doi.org/10.1016/j.watres.2007.11.039; '
                   'criteria 2'),
    dict(group='AlgaeCentrifuge', name='phi', attr='phi',
         units='-', expected=0.5, low=0.27, high=0.73,
         distribution='uniform',
         reference='expected: average of low & high; sources: '
                   'https://doi.org/10.1016/j.algal.2017.11.038; '
                   'https://doi.org/10.1016/j.algal.2020.102046; '
                   'https://doi.org/10.1016/j.algal.2015.12.007; '
                   'criteria 2'),
    dict(group='Photobioreactor', name='InD', attr='InD',
         units='m', expected=0.1, low=0.02, high=0.2,
         distribution='triangular',
         reference='expected: https://doi.org/10.1021/acs.est.3c10264; '
                   'additional sources: https://doi.org/10.1016/j.apenergy.2012.12.068; '
                   'https://doi.org/10.1016/j.jece.2026.121938; '
                   'criteria 4.2'),
    dict(group='Photobioreactor', name='length', attr='length',
         units='m', expected=80., low=11., high=120.,
         distribution='triangular',
         reference='expected: https://doi.org/10.1021/acs.est.3c10264; '
                   'additional sources: https://doi.org/10.1016/j.jece.2026.121938; '
                   'https://doi.org/10.1016/j.algal.2022.102679; '
                   'criteria 4.2'),
    dict(group='Photobioreactor', name='v', attr='v',
         units='m/s', expected=0.4, low=0.23, high=1.,
         distribution='triangular',
         reference='expected: https://doi.org/10.1021/acs.est.3c10264; '
                   'additional sources: https://doi.org/10.1016/S0009-2509(00)00521-2; '
                   'https://doi.org/10.1016/j.algal.2022.102679; '
                   'criteria 4.2'),
    dict(group='Photobioreactor', name='light_intensity',
         attr='light_intensity', units='umol/m2/s', expected=100.,
         low=50., high=200., distribution='triangular',
         reference='expected: Clearas designed light intensity documentation; '
                   'additional sources: https://doi.org/10.3390/en13225994; '
                   'https://research-portal.uws.ac.uk/en/publications/development-of-effective-approach-for-dewatering-microalgae/; '
                   'criteria 4.2'),
    dict(group='Photobioreactor', name='hrt', attr='hrt',
         units='hr', expected=5, low=4., high=48,
         distribution='triangular',
         reference='expected: average of low & high; '
                   'https://doi.org/10.1021/acs.est.3c10264;'
                   'https://doi.org/10.1021/acsestengg.5c00485'
                   'https://doi.org/10.1021/acs.est.1c02590'
                    'criteria 4.2'),
    dict(group='Photobioreactor', name='glass_tube_lifetime',
         attr='glass_tube_lifetime', units='yr', expected=40.,
         low=30., high=50., distribution='uniform',
         reference='expected: average of low & high; sources: '
                   'https://www.algaefoundationatec.org/aces/download/Techno-Economic%20Analysis.pdf; '
                   'https://doi.org/10.1007/s10098-021-02042-x; '
                   'criteria 2'),
    dict(group='Photobioreactor', name='LED_lifetime',
         attr='LED_lifetime', units='yr', expected=18.5,
         low=12., high=25., distribution='uniform',
         reference='expected: average of low & high; expected: Clearas '
                   'documentation and https://doi.org/10.1371/journal.pone.0099010; '
                   'additional sources: https://doi.org/10.1016/j.algal.2022.102679; '
                   'Clearas Documentation ("Lighting System"); criteria 2'),
    dict(group='Photobioreactor', name='heater_up_time_ratio',
         attr='heater_up_time_ratio', units='-', expected=0.39778420038535645,
         low=0.16666666666666666, high=0.6289017341040463,
         distribution='uniform',
         reference='expected: average of low & high; expected: heating 50% '
                   'of night assumption; source: https://doi.org/10.17930/AGL2016123; '
                   'criteria 2'),
    dict(group='Photobioreactor', name='annual_heating_days',
         attr='annual_heating_days', units='d/yr', expected=152.,
         low=31., high=273., distribution='uniform',
         reference='expected: average of low & high; sources: '
                   'https://www.weather.gov/wrh/climate?wfo=mfl&tab=cliplot&sid=MIAthr&view=yearly&year=2025&yearMode=calendar; '
                   'https://climatology.nelson.wisc.edu/wisconsin-historic-climate-data/statewide-climate-normals/; '
                   'criteria 2'),
    # Ecorecovery pump tab. Applies to Ecorecoverypump units.
    dict(group='Ecorecoverypump', name='N_ubends',
         attr='feed_PBR_N_ubends', units='-', expected=12.,
         low=0., high=24., distribution='uniform',
         reference='expected: average of low & high; source: '
                   'https://doi.org/10.1016/j.apenergy.2012.12.068; '
                   'criteria 2'),
    dict(group='Ecorecoverypump', name='length',
         attr='feed_PBR_length', units='m', expected=80.,
         low=11., high=120., distribution='triangular',
         reference='expected: https://doi.org/10.1021/acs.est.3c10264; '
                   'additional sources: https://doi.org/10.1016/j.jece.2026.121938; '
                   'https://doi.org/10.1016/j.algal.2022.102679; '
                   'criteria 4.2'),
    dict(group='Ecorecoverypump', name='motor_efficiency',
         attr='motor_efficiency_override', units='-', expected=0.7,
         low=0.5, high=0.9, distribution='uniform',
         reference='expected: average of low & high; sources: '
                   'https://doi.org/10.1061/(ASCE)0733-9496(2008)134:1(88); '
                   'https://doi.org/10.14512/tatup.21.1.54; '
                   'https://doi.org/10.1016/j.algal.2022.102679; '
                   'criteria 2'),
    dict(group='Ecorecoverypump', name='capacity_factor',
         attr='capacity_factor', units='-', expected=1.,
         low=0.7, high=1.3, distribution='uniform',
         reference='workbook parameter rr, mapped to capacity_factor following '
                   'project naming; https://doi.org/10.1021/acs.est.3c10264; '
                   '+/-30%; criteria 3'),
)


def _distribution(kind, low, expected, high):
    '''
    Build the Chaospy distribution requested by one parameter row.

    This helper keeps the parameter table compact: each entry stores only
    ``low``, ``expected``, ``high``, and a distribution name, while this
    function converts those values into the object QSDsan ``Model`` expects.
    '''
    low, high = sorted((float(low), float(high)))
    expected = min(max(float(expected), low), high)
    kind = str(kind).lower()
    if 'triangular' in kind:
        return shape.Triangle(lower=low, midpoint=expected, upper=high)
    return shape.Uniform(lower=low, upper=high)


def _unit_group(system, group):
    '''
    Return the unit objects affected by a parameter group.

    The uncertainty table is organized by process category, not always by a
    single unit ID. This helper maps category names such as ``Mixtank`` or
    ``Ecorecoverypump`` to the actual units in the current flowsheet.
    '''
    units = system.flowsheet.unit
    if group == 'Mixtank':
        return [u for u in system.units if isinstance(u, Mixtank)]
    if group == 'CO2Supply':
        return [units.CO2]
    if group == 'Ultrafiltration':
        return [units.MEM]
    if group == 'AlgaeCentrifuge':
        return [units.CENT]
    if group == 'Photobioreactor':
        return [units.PBR1]
    if group == 'Ecorecoverypump':
        return [u for u in system.units if isinstance(u, Ecorecoverypump)]
    return []

def _units_from_data(system, data):
    '''
    Return the specific unit list for one uncertainty-parameter definition.

    Some parameters apply to one named unit, such as ``MEM hrt``. Others apply
    to a whole category, such as all pumps. This helper chooses the named unit
    first when a ``unit`` key is present and otherwise falls back to
    :func:`_unit_group`.
    '''
    unit_ID = data.get('unit')
    if unit_ID:
        try:
            return [getattr(system.flowsheet.unit, unit_ID)]
        except AttributeError:
            return []
    return _unit_group(system, data['group'])

def _set_resource_price(system, resource_ID, value):
    '''
    Apply a sampled resource price to the project dictionaries and streams.

    Stream costs in TEA read prices from the resource streams themselves, while
    other code reads ``pmlca.price_dct``. This helper keeps both places
    synchronized during Monte Carlo sampling. Electricity also updates
    ``qs.PowerUtility.price`` and the electricity impact item price.
    '''
    pmlca.price_dct[resource_ID] = value
    if resource_ID == 'Electricity':
        qs.PowerUtility.price = value
        item = ImpactItem.get_item('e_item')
        if item: item.price = value
    stream_key = {
        'NaturalGas': 'Natural_gas',
        'CitricAcid': 'Citric_acid',
        'SodiumHypochlorite': 'Sodium_hypochlorite',
    }.get(resource_ID, resource_ID)
    for stream in system.streams:
        if stream_key in stream.ID:
            stream.price = value

def _set_resource_impact(system, resource_ID, indicator, value):
    '''
    Update one sampled GWP or eutrophication factor during uncertainty
    analysis.

    During Monte Carlo simulation, a resource impact factor such as natural-gas
    GWP or electricity eutrophication potential may be sampled away from its
    baseline value. This function pushes that sampled value into every place
    where the LCA calculation may read it:

    * ``pmlca.GWP_dct`` or ``pmlca.EUTRO_dct``, for calculations that read the
      project dictionaries directly.
    * The matching QSDsan ``ImpactItem`` such as ``e_item``,
      ``NaturalGas_item``, ``CitricAcid_item``, ``SodiumHypochlorite_item``, or
      ``CO2_item``.
    * Any per-stream copied impact item already attached to a system stream.

    Nutrient and COD discharge factors (``S_NH``, ``S_NO``, ``S_P``, and
    ``COD``) are handled only through ``pmlca.EUTRO_dct`` because their direct
    effluent impact is calculated from that dictionary in
    ``ecorecover_lca._effluent_eutrophication_flow`` rather than through normal
    resource-stream ``ImpactItem`` objects.
    '''
    dct = pmlca.GWP_dct if indicator == 'GlobalWarming' else pmlca.EUTRO_dct
    dct[resource_ID] = value

    if resource_ID == 'Electricity':
        item_IDs = ('e_item',)
    elif resource_ID in ('S_NH', 'S_NO', 'S_P', 'COD'):
        # Direct effluent eutrophication is calculated from pmlca.EUTRO_dct in
        # ecorecover_lca._effluent_eutrophication_flow.
        item_IDs = ()
    else:
        item_IDs = (f'{resource_ID}_item',)

    for item_ID in item_IDs:
        item = ImpactItem.get_item(item_ID)
        if item is not None:
            item.CFs[indicator] = value

    # Per-system stream items are normally linked to the source item. This loop
    # also catches any unlinked copies created during interactive work.
    for stream in getattr(system, 'streams', ()):
        item = getattr(stream, 'stream_impact_item', None)
        if item is None:
            continue
        if resource_ID == 'Electricity':
            continue
        if resource_ID in item.ID:
            try:
                item.CFs[indicator] = value
            except Exception:
                pass


def _uniform_relative_distribution(baseline, fraction):
    '''
    Create a uniform uncertainty range around a baseline value.

    This is used for resource impact factors and prices where the model uses a
    simple relative uncertainty assumption, such as +/-50% for electricity or
    +/-10% for most resource impact factors.
    '''
    return shape.Uniform(
        lower=float(baseline) * (1 - fraction),
        upper=float(baseline) * (1 + fraction),
    )


def _resource_price_data():
    '''
    Return stream price uncertainty data.

    Citric acid and sodium hypochlorite values are from the UF tab of
    Unit_process_assumptions_Ecorecover.xlsx. Reference links are from
    the UF-breakdown tab rows cited in that workbook.
    '''
    return (
        dict(
            ID='NaturalGas',
            units='USD/kg',
            baseline=pmlca.price_dct['NaturalGas'],
            distribution=_uniform_relative_distribution(
                pmlca.price_dct['NaturalGas'], 0.50),
            description='Baseline follows pmlca.price_dct["NaturalGas"]; '
                        '+/-50% uncertainty requested by user. Current '
                        'baseline source in __init__.py: '
                        'https://www.globalpetrolprices.com/natural_gas_prices/',
        ),
        dict(
            ID='CitricAcid',
            units='USD/kg',
            baseline=1.8595038517821776,
            distribution=shape.Triangle(
                lower=0.7670134102017797,
                midpoint=1.8595038517821776,
                upper=5.964547472256473),
            description='UF tab citric_acid_unit_price; triangular '
                        'distribution, criteria 4.2. Expected is average of '
                        'data; expected sources: '
                        'https://doi.org/10.1016/j.jwpe.2024.104892 and '
                        'https://doi.org/10.1021/acs.est.4c05389; low source: '
                        'https://doi.org/10.1021/acs.est.4c05389; high source: '
                        'https://doi.org/10.3390/w17243453.',
        ),
        dict(
            ID='SodiumHypochlorite',
            units='USD/kg/%_frac',
            baseline=14.265088213315847,
            distribution=shape.Triangle(
                lower=1.6480834782608698,
                midpoint=14.265088213315847,
                upper=38.66574435782045),
            description='UF tab sodium_hypochlorite_unit_price; triangular '
                        'distribution, criteria 4.2. Expected is average of '
                        'data; low source: '
                        'https://doi.org/10.1016/j.watres.2009.11.020; '
                        'high source: '
                        'https://www.chemworld.com/Sodium-HypoChlorite-p/cw56-55.htm.',
        ),
    )


def _resource_impact_data(indicator):
    '''
    Return uncertainty definitions for resource GWP or eutrophication factors.

    The baseline values come from ``pmlca.GWP_dct`` or ``pmlca.EUTRO_dct``.
    Nonzero factors become Monte Carlo parameters; zero factors are skipped
    because sampling around zero would still produce no impact signal.
    '''
    if indicator == 'GlobalWarming':
        dct = pmlca.GWP_dct
        units = 'kg CO2-eq/kg'
        electricity_units = 'kg CO2-eq/kWh'
    else:
        dct = pmlca.EUTRO_dct
        units = 'kg N-eq/kg'
        electricity_units = 'kg N-eq/kWh'
    rows = []
    for resource_ID, baseline in dct.items():
        if baseline == 0:
            continue
        fraction = 0.50 if resource_ID == 'Electricity' else 0.10
        rows.append(dict(
            ID=resource_ID,
            baseline=baseline,
            distribution=_uniform_relative_distribution(baseline, fraction),
            units=electricity_units if resource_ID == 'Electricity' else units,
            description=f'Baseline follows pmlca.'
                        f'{"GWP_dct" if indicator == "GlobalWarming" else "EUTRO_dct"}'
                        f'["{resource_ID}"]; +/-{fraction:.0%} uncertainty.',
        ))
    return tuple(rows)

def _pbr_dynamic_units(system):
    '''
    Return all PBR blocks that participate in the dynamic PM2 model.

    Light intensity and HRT changes must be applied consistently to PBR1
    through PBR20, so this helper collects those units in order when they exist
    in the flowsheet.
    '''
    units = system.flowsheet.unit
    pbrs = []
    for n in range(1, 21):
        ID = f'PBR{n}'
        if hasattr(units, ID):
            pbrs.append(getattr(units, ID))
    return pbrs


def _set_pm2_b_reactor(system, value):
    '''
    Apply the sampled PBR inner diameter to PM2's ``b_reactor`` parameter.

    PM2 uses ``b_reactor`` internally for light attenuation. The uncertainty
    parameter is stored on the Photobioreactor as ``InD``, so this helper
    translates the sampled unit attribute into the PM2 process-model parameter.
    '''
    pm2 = getattr(pmlca, 'pm2', None)
    if pm2 is None:
        try:
            pm2 = system.flowsheet.unit.PBR1._model
        except Exception:
            pm2 = None
    if pm2 is not None:
        pm2.set_parameters(b_reactor=value)

def _baseline_light_data(system):
    '''
    Return and cache the original PBR light time-series data.

    PM2 uses an exogenous light variable, ``I``, rather than one fixed light
    value. For uncertainty analysis, we need a stable baseline time series so
    each Monte Carlo sample can scale the same original day/night light pattern
    instead of repeatedly scaling an already-scaled series.

    This function reads the light variable from the first available PBR,
    stores its ID, time points, light values, and positive-value mean, and then
    caches those data for this system. The mean is used as the baseline average
    light intensity. If a sampled light intensity is later provided,
    :func:`_set_pbr_light_intensity` rescales the original light profile by
    ``sampled_value / baseline_mean``.
    '''
    key = id(system)
    if key in _baseline_light_by_system:
        return _baseline_light_by_system[key]
    pbrs = _pbr_dynamic_units(system)
    if not pbrs:
        raise RuntimeError('No PBR units are available for light uncertainty.')
    light = pbrs[0].exo_dynamic_vars[1]
    t = np.asarray(light.t_data, dtype=float)
    y = np.asarray(light.y_data, dtype=float)
    mean = float(np.mean(y[y > 0])) if np.any(y > 0) else float(np.mean(y))
    if mean <= 0:
        raise RuntimeError('Baseline exogenous light data must be positive.')
    data = (light.ID, t, y, mean)
    _baseline_light_by_system[key] = data
    return data


def _set_pbr_light_intensity(system, value):
    '''
    Apply a sampled average light intensity to all dynamic PBR blocks.

    The PM2 model needs a time-varying light profile, not just a scalar light
    intensity. This function keeps the original baseline light pattern from
    :func:`_baseline_light_data` but scales all light values so the profile's
    average intensity matches the sampled Monte Carlo value.

    The same scaled light variable is assigned to PBR1-PBR20, and the module
    globals ``pmlca.I`` and ``pmlca.ecorecover_lca.I`` are updated so later
    dynamic simulations use the sampled light condition consistently.
    '''
    ID, t, y, mean = _baseline_light_data(system)
    scaled_light = EDV(ID, t, y * value / mean, interpolator='linear')
    T = _pbr_dynamic_units(system)[0].exo_dynamic_vars[0]
    for unit in _pbr_dynamic_units(system):
        unit.exo_dynamic_vars = (T, scaled_light)
    pmlca.I = scaled_light
    if hasattr(pmlca, 'ecorecover_lca'):
        pmlca.ecorecover_lca.I = scaled_light
    _current_light_by_system[id(system)] = value
    try:
        system.flowsheet.unit.PBR1.light_intensity = value
    except Exception:
        pass


def _sync_dynamic_model_inputs(system):
    '''
    Push sampled dynamic parameters into PM2 before evaluating a sample.

    This function synchronizes the inputs that affect PM2 dynamics but are not
    ordinary static cost parameters: PBR diameter/light intensity and HRT-based
    reactor volumes. It is called by both static and dynamic model
    specifications so design/cost summaries use the current sampled values.
    '''
    try:
        PBR1 = system.flowsheet.unit.PBR1
    except Exception:
        PBR1 = None
    if PBR1 is not None:
        _set_pm2_b_reactor(system, PBR1.InD)
        if _current_light_by_system.get(id(system)) != PBR1.light_intensity:
            _set_pbr_light_intensity(system, PBR1.light_intensity)
    pmlca.ecorecover_lca.sync_dynamic_volumes(system)


def _set_pbr_attr(system, attr, value):
    '''
    Set one sampled Photobioreactor parameter and linked pump assumptions.

    PBR geometry affects both the PM2 process model and the feed-PBR pump
    hydraulic calculations. This helper keeps those related attributes in sync
    whenever Monte Carlo changes PBR diameter, length, velocity, HRT, or light.
    '''
    if attr == 'light_intensity':
        _set_pbr_light_intensity(system, value)
        return
    for unit in _unit_group(system, 'Photobioreactor'):
        setattr(unit, attr, value)
    if attr == 'InD':
        _set_pm2_b_reactor(system, value)
    if attr == 'hrt':
        PBR1 = system.flowsheet.unit.PBR1
        for unit in _pbr_dynamic_units(system):
            unit.hrt = PBR1.hrt
    # Keep feed-PBR pump hydraulic assumptions synchronized with PBR geometry.
    for unit in _unit_group(system, 'Ecorecoverypump'):
        if unit.pump_type != 'feed_PBR':
            continue
        if attr == 'InD':
            unit.feed_PBR_InD = value * _m_to_ft
        elif attr == 'length':
            unit.feed_PBR_length = value * _m_to_ft
        elif attr == 'v':
            unit.feed_PBR_v = value * _m_to_ft
        elif attr == 'hrt':
            unit.feed_PBR_hrt = value


def _set_pump_attr(system, attr, value):
    '''
    Set one sampled pump parameter across the relevant EcoRecovery pumps.

    Most pump uncertainty parameters apply to all Ecorecoverypump units. A few
    feed-PBR hydraulic parameters only apply to the pump feeding the PBR, and
    motor efficiency is stored as a module-level override because it is applied
    after pump design calculates brake horsepower.
    '''
    global _pump_motor_efficiency
    if attr == 'motor_efficiency_override':
        _pump_motor_efficiency = value
        return
    for unit in _unit_group(system, 'Ecorecoverypump'):
        if attr == 'feed_PBR_length':
            if unit.pump_type == 'feed_PBR':
                unit.feed_PBR_length = value * _m_to_ft
        elif attr == 'feed_PBR_N_ubends':
            if unit.pump_type == 'feed_PBR':
                unit.feed_PBR_N_ubends = value
        else:
            setattr(unit, attr, value)


def _set_mixtank_attr(system, attr, value, baseline):
    '''
    Set one sampled Mixtank parameter while preserving HRT scaling behavior.

    For HRT-like uncertainty, the sampled value scales each tank from its own
    baseline HRT instead of forcing MIX and RET to the same absolute value.
    Other Mixtank parameters are assigned directly to every Mixtank unit.
    '''
    units = _unit_group(system, 'Mixtank')
    if attr == 'hrt_ratio':
        for unit in units:
            if not hasattr(unit, '_baseline_hrt'):
                unit._baseline_hrt = unit.hrt
            unit.hrt = unit._baseline_hrt * value / baseline
    else:
        for unit in units:
            setattr(unit, attr, value)


def _parameter_setter(system, data):
    '''
    Build the setter function registered with QSDsan ``Model.parameter``.

    QSDsan calls the returned closure for every Monte Carlo sample. The closure
    knows whether a parameter should update one named unit, all units in a
    group, PM2 dynamic inputs, pump hydraulic assumptions, or tank HRT scaling.
    '''
    group = data['group']
    attr = data['attr']
    baseline = data['expected']
    def setter(value):
        if data.get('unit'):
            for unit in _units_from_data(system, data):
                setattr(unit, attr, value)
            return
        if group == 'Photobioreactor':
            _set_pbr_attr(system, attr, value)
        elif group == 'Ecorecoverypump':
            _set_pump_attr(system, attr, value)
        elif group == 'Mixtank':
            _set_mixtank_attr(system, attr, value, baseline)
        else:
            for unit in _unit_group(system, group):
                setattr(unit, attr, value)
    return setter

def _add_resource_price_parameters(model):
    '''
    Register system-level price and impact-factor uncertainty parameters.

    These parameters are not SanUnit attributes. They change electricity price,
    labor wage, resource stream prices, and resource impact factors used by TEA
    and LCA calculations.
    '''
    system = model.system
    param = model.parameter
    eia_df = load_data(
        ospath.join(pmlca.data_path, 'eia_electricity_price.xlsx'),
        header=[0, 1], index_col=0,
    )
    data = eia_df.loc[:, ('All Sectors', 'March 2025 YTD')].to_numpy()
    data = data[np.isfinite(data.astype(float))].astype(float)
    b = float(eia_df.loc['U.S. Total', ('Industrial', 'March 2025 YTD')])
    D = shape.TruncNormal(
        data.min() - 0.5, data.max() + 0.5, mu=b, sigma=data.std(),
    )
    param(
        setter=lambda value: _set_resource_price(
            system, 'Electricity', value / 100,
        ),
        name='Electricity price',
        element='System',
        kind='cost',
        units='cents/kWh',
        baseline=b,
        distribution=D,
        description='EIA electricity monthly Table 5.6.B, March 2025 YTD. '
                    'Baseline is U.S. Total industrial electricity price; '
                    'distribution uses all-sector state data. Source: '
                    'https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=table_5_06_b',
    )

    labor_data = pmlca.load_labor_wage_data()
    b = pmlca.price_dct['Labor']
    D = shape.TruncNormal(
        labor_data.min() - 0.5,
        labor_data.max() + 0.5,
        mu=b,
        sigma=labor_data.std(),
    )
    param(
        setter=lambda value: _set_resource_price(system, 'Labor', value),
        name='Labor wage',
        element='System',
        kind='cost',
        units='USD/hr',
        baseline=b,
        distribution=D,
        description='U.S. Bureau of Labor Statistics OEWS May 2022 state '
                    'hourly median wages for OCC_CODE 51-8031, Water and '
                    'Wastewater Treatment Plant and System Operators. Baseline '
                    'is the median of state H_MEDIAN values; suppressed nonnumeric wages are excluded.',
    )

    for data in _resource_price_data():
        resource_ID = data['ID']
        param(
            setter=lambda value, ID=resource_ID: _set_resource_price(
                system, ID, value,
            ),
            name=f'Resource - {resource_ID} price',
            element='Resource price',
            kind='cost',
            units=data['units'],
            baseline=data['baseline'],
            distribution=data['distribution'],
            description=data['description'],
        )

    for indicator, name in (
            ('GlobalWarming', 'GWP'),
            ('EUT', 'Eutrophication'),
        ):
        for data in _resource_impact_data(indicator):
            resource_ID = data['ID']
            param(
                setter=lambda value, ID=resource_ID, ind=indicator:
                    _set_resource_impact(system, ID, ind, value),
                name=f'Resource - {resource_ID} {name}',
                element='Resource impact',
                kind='LCA',
                units=data['units'],
                baseline=data['baseline'],
                distribution=data['distribution'],
                description=data['description'],
        )


def _add_parameters(model):
    '''
    Register all unit-level and resource-level uncertainty parameters.

    This function converts the project ``_PARAMETERS`` table into QSDsan model
    parameters, then appends price, labor, and impact-factor uncertainties that
    are not stored as SanUnit attributes.
    '''
    system = model.system
    param = model.parameter
    for data in _PARAMETERS:
        units = _units_from_data(system, data)
        element = units[0] if units else data['group']
        D = _distribution(
            data['distribution'], data['low'], data['expected'], data['high'],
        )
        param(
            setter=_parameter_setter(system, data),
            name=f"{data['group']} - {data['name']}",
            element=element, kind='coupled', units=data['units'],
            baseline=data['expected'], distribution=D,
            description=data['reference'],
        )
    _add_resource_price_parameters(model)


def _annual_treated_volume(system):
    '''
    Return the annual treated water volume used to normalize metrics.

    The model reports treatment cost and impacts per m3. This helper searches
    the influent/effluent streams for the first positive volumetric flow and
    multiplies it by operating hours. It raises an error instead of hardcoding
    a flow if no simulated flow is available.
    '''
    for ID in ('Dynamic_influent', 'Effluent_unpumped', 'Effluent'):
        try:
            F_vol = getattr(system.flowsheet.stream, ID).F_vol
        except Exception:
            F_vol = 0.
        if F_vol > 0:
            return F_vol * system.operating_hours
    raise RuntimeError(
        'No positive flow is available from Dynamic_influent, '
        'Effluent_unpumped, or Effluent; run the dynamic simulation before '
        'evaluating volume-normalized metrics.'
    )

def _normalize(system, value):
    '''
    Normalize an annual cost or impact by annual treated volume.

    This keeps all core TEA/LCA metrics on a common ``per m3 treated`` basis.
    '''
    Q = _annual_treated_volume(system)
    return value / Q if Q else float('nan')


def _impact(system, ID):
    '''
    Return one annual total LCA impact from the system LCA object.

    ``ID`` is the impact indicator name used by QSDsan, such as
    ``GlobalWarming`` or ``EUT``.
    '''
    lca = system.LCA
    return lca.get_total_impacts(annual=True).get(ID, 0.)

def _stream_concentration(system, stream_ID, component_ID):
    '''
    Return one stream component concentration for process-performance metrics.

    This is used for PM2 output metrics such as effluent phosphate
    concentration. It returns ``nan`` when the stream/component is unavailable
    so failed metrics are visible without crashing the full metric list.
    '''
    try:
        stream = getattr(system.flowsheet.stream, stream_ID)
        return float(
            stream.get_mass_concentration(
                'mg/L', IDs=(component_ID,),
            )[0]
        )
    except Exception:
        return float('nan')


def _apply_pump_motor_efficiency(system):
    '''
    Recalculate pump electricity after sampling motor efficiency.

    Pump design first calculates brake horsepower. If motor efficiency is an
    uncertainty parameter, this helper converts brake horsepower to electrical
    power using the sampled efficiency after unit design has run.
    '''
    if _pump_motor_efficiency is None:
        return
    for unit in _unit_group(system, 'Ecorecoverypump'):
        try:
            BHP = unit.BHP
        except Exception:
            continue
        unit.power_utility.rate = BHP / _pump_motor_efficiency * _hp_to_kW


def _refresh_lca(system):
    '''
    Reset LCA cached unit and stream lists after design/cost updates.

    Some construction quantities and LCA-only resource streams are created or
    updated during unit design and summary calculations. Clearing these caches
    forces QSDsan LCA to see the latest construction items and stream impacts.
    '''
    if not hasattr(system, '_LCA'):
        return
    system._LCA._construction_units = set()
    system._LCA._transportation_units = set()
    system._LCA._lca_streams = set()
    system._LCA.system = system
    add_extra = getattr(pmlca.ecorecover_lca, '_add_extra_lca_streams', None)
    extra_dct = getattr(pmlca.ecorecover_lca, '_extra_lca_streams_by_system', {})
    if add_extra:
        add_extra(system._LCA, extra_dct.get(id(system), ()))


def _summary_only(system):
    '''
    Recalculate design, cost, and LCA results without rerunning PM2 dynamics.

    This is the fast specification used for the static uncertainty model. It
    assumes the process stream states from a previous PM2 simulation are already
    available and that the sampled uncertain parameters do not require solving
    the PM2 ODEs again.

    The function first synchronizes dynamic model inputs such as HRT-based
    volumes, PBR light intensity, and PM2 ``b_reactor``. It then calls
    ``system._summary()`` to rerun unit design/cost calculations from the
    cached stream states, applies any sampled pump motor efficiency, and
    refreshes the LCA cached unit/stream lists.
    '''
    _sync_dynamic_model_inputs(system)
    system._summary()
    _apply_pump_motor_efficiency(system)
    _refresh_lca(system)


def _simulate_baseline(system, t, t_step, method, print_t):
    '''
    Run the full dynamic PM2 simulation for the baseline system.

    This is used when creating a model or when dynamic uncertainty sampling
    needs fresh PM2 state trajectories. It also syncs the module-level
    ``pmlca.eco`` reference so package-level commands point to the same system.
    '''
    _sync_dynamic_model_inputs(system)
    if system is not getattr(pmlca, 'eco', None):
        pmlca.ecorecover_lca.eco = system
        pmlca._sync_globals(system)
    pmlca.run(t=t, t_step=t_step, method=method, print_t=print_t)


def _dynamic_summary(system, t, t_step, method, print_t):
    '''
    Evaluate one Monte Carlo sample with a fresh PM2 dynamic simulation.

    The dynamic model kind uses this path when sampled parameters can affect
    PM2 states, such as HRT, light intensity, or reactor diameter. After the
    ODE simulation, it also refreshes design, cost, and LCA summaries.
    '''
    _sync_dynamic_model_inputs(system)
    _simulate_baseline(system, t=t, t_step=t_step, method=method,
                       print_t=print_t)
    _summary_only(system)


def _make_specification(system, model_kind='static', t=30, t_step=0.5,
                        method='RK23', print_t=False):
    '''
    Build the QSDsan model specification function.

    The returned function is called once per Monte Carlo sample. It chooses the
    fast static path or the slower dynamic path based on ``model_kind``.
    '''
    def specification():
        if model_kind == 'dynamic':
            _dynamic_summary(system, t=t, t_step=t_step, method=method,
                             print_t=print_t)
        else:
            _summary_only(system)
    return specification


def _add_core_metrics(model):
    '''
    Add the main treatment-cost, LCA, and effluent-quality metrics.

    These are the default high-level outputs for uncertainty analysis:
    USD/m3, kg CO2-eq/m3, kg N-eq/m3, and effluent soluble phosphorus.
    '''
    system = model.system
    metrics = [
        Metric('Treatment cost', lambda: _normalize(system, system.TEA.EAC),
               'USD/m3', 'TEA results'),
        Metric('GlobalWarming',
               lambda: _normalize(system, _impact(system, 'GlobalWarming')),
               'kg CO2-eq/m3', 'LCA results'),
        Metric('Eutrophication',
               lambda: _normalize(system, _impact(system, 'EUT')),
               'kg N-eq/m3', 'LCA results'),
        Metric('Effluent S_P concentration',
               lambda: _stream_concentration(system, 'Effluent_unpumped', 'S_P'),
               'mg/L', 'PM2 results'),
    ]
    model.metrics = [*model.metrics, *metrics]

# def _unit_purchase_cost(unit):
#     return sum(unit.purchase_costs.values())

def _unit_installed_cost(unit):
    '''
    Return installed equipment cost for one unit.

    Unit-category metrics use installed cost rather than purchase cost because
    the project TEA applies bare-module and system-level cost factors.
    '''
    return sum(unit.installed_costs.values())


def _unit_add_opex(unit, key=None):
    '''
    Return one unit's additional OPEX, optionally for one OPEX category.

    SanUnits store labor and maintenance/replacement costs in ``add_OPEX`` as
    hourly costs. Category metrics use this helper before annualizing them.
    '''
    if key is None:
        return sum(unit.add_OPEX.values())
    return unit.add_OPEX.get(key, 0.)


def _unit_electricity_cost(system, unit):
    '''
    Return annual electricity cost for one unit.

    The calculation uses the unit power draw, the current sampled electricity
    price, and annual operating hours.
    '''
    return unit.power_utility.rate * bst.PowerUtility.price * system.operating_hours

def _related_lca_resource_streams(unit):
    '''
    Return streams that can represent an LCAResourceInput resource demand.

    Chemical LCAResourceInput units receive pumped streams. The pumped streams
    carry the mass flow, but the original chemical feed stream upstream of the
    pump carries the price and stream impact item. This helper follows the
    inlet source unit one step upstream so unit metrics can use the accounting
    stream rather than the dummy outlet.
    '''
    if not isinstance(unit, LCAResourceInput):
        return ()

    streams = []
    for stream in unit.ins:
        streams.append(stream)
        source = getattr(stream, 'source', None)
        if source is not None:
            streams.extend(source.ins)
    streams.extend(unit.outs)

    unique = []
    seen = set()
    for stream in streams:
        if stream is None or id(stream) in seen:
            continue
        seen.add(id(stream))
        unique.append(stream)
    return tuple(unique)


def _lca_resource_accounting_stream(unit):
    '''
    Return one stream for LCAResourceInput cost/impact metrics.

    Preference order:
    1. Stream with an impact item and nonzero mass flow.
    2. Stream with nonzero stream cost.
    3. The LCAResourceInput inlet, as a last-resort flow placeholder.
    '''
    streams = _related_lca_resource_streams(unit)
    for stream in streams:
        if getattr(stream, 'stream_impact_item', None) and stream.F_mass:
            return stream
    for stream in streams:
        try:
            if stream.cost:
                return stream
        except Exception:
            pass
    return streams[0] if streams else None


def _unit_resource_stream_cost(system, unit):
    '''
    Return annual stream purchase cost for one LCAResourceInput unit.

    Natural gas and membrane-cleaning chemicals are priced as streams rather
    than as SanUnit add_OPEX. This helper pulls those stream costs into the
    unit-category O&M metrics.
    '''
    stream = _lca_resource_accounting_stream(unit)
    if stream is None:
        return 0.
    try:
        return stream.cost * system.operating_hours
    except Exception:
        return 0.


def _unit_resource_stream_impact(system, unit, indicator):
    '''
    Return annual stream impact for one LCAResourceInput unit.

    This lets resource-input units contribute operation GWP/eutrophication from
    their attached stream impact items, even though their dummy outlets do not
    participate in PM2 mass balances.
    '''
    stream = _lca_resource_accounting_stream(unit)
    if stream is None or not getattr(stream, 'stream_impact_item', None):
        return 0.
    return _stream_impacts(system, (stream,), indicator)


def _stream_impacts(system, streams, indicator):
    '''
    Return annual impacts for streams that have stream impact items.

    The helper filters out ordinary process streams without impact items so
    QSDsan LCA is only asked to evaluate streams that are meant to represent
    purchased resources or direct discharge factors.
    '''
    streams = [i for i in streams if getattr(i, 'stream_impact_item', None)]
    if not streams:
        return 0.
    return system.LCA.get_stream_impacts(stream_items=streams, annual=True).get(
        indicator, 0.,
    )


def _unit_construction_impact(system, unit, indicator):
    '''
    Return annualized construction impact for one unit.

    Unit-category LCA metrics use this helper to separate construction impacts
    from operation and electricity impacts.
    '''
    return system.LCA.get_construction_impacts([unit], annual=True).get(
        indicator, 0.,
    )


def _unit_operation_impact(system, unit, indicator):
    '''
    Return annual non-electric operation impact for one unit.

    For LCAResourceInput units, this means the linked resource stream impact.
    For regular process units, this checks inlet and outlet streams for any
    attached stream impact items.
    '''
    if isinstance(unit, LCAResourceInput):
        return _unit_resource_stream_impact(system, unit, indicator)
    return _stream_impacts(system, [*unit.ins, *unit.outs], indicator)


def _unit_electricity_impact(system, unit, indicator):
    '''
    Return annual electricity impact for one unit.

    This converts unit electricity use to GWP or eutrophication using the
    current ``e_item`` characterization factor, which may be sampled during
    uncertainty analysis.
    '''
    item = ImpactItem.get_item('e_item')
    if item is None:
        return 0.
    return (
        unit.power_utility.rate
        * system.operating_hours
        * item.CFs.get(indicator, 0.)
    )

def _unit_metric_categories(system):
    '''
    Return grouped unit categories for granular TEA/LCA metrics.

    Excluded by design: SE, PBR2-PBR20, MEV, and POST_MEM. CO2Supply is also
    omitted because it is not part of the user-defined category list here.
    '''
    units = system.flowsheet.unit
    categories = {
        'pump': [
            unit for unit in system.units
            if isinstance(unit, Ecorecoverypump)
        ],
        'photobioreactor': [
            units.PBR1,
            units.PBR1_NATURAL_GAS,
        ],
        'centrifuge': [
            unit for unit in system.units
            if isinstance(unit, AlgaeCentrifuge)
        ],
        'tank': [
            unit for unit in system.units
            if isinstance(unit, Mixtank)
        ],
        'ultrafiltration': [
            units.MEM,
            units.MEM_CITRIC_ACID,
            units.MEM_SODIUM_HYPOCHLORITE,
        ],
    }
    return {
        category: tuple(unit for unit in category_units if unit is not None)
        for category, category_units in categories.items()
    }


def _sum_installed_cost(units):
    '''
    Sum installed cost across a unit category.

    Category metrics group similar equipment, so this helper aggregates the
    installed-cost contribution for that group.
    '''
    return sum(_unit_installed_cost(unit) for unit in units)


def _sum_opex(system, units):
    '''
    Sum annual O&M cost across a unit category.

    The total includes SanUnit add_OPEX, electricity cost, and priced
    LCA-resource streams such as natural gas and cleaning chemicals.
    '''
    return sum(
        _unit_add_opex(unit) * system.operating_hours
        + _unit_electricity_cost(system, unit)
        + _unit_resource_stream_cost(system, unit)
        for unit in units
    )


def _sum_electricity_cost(system, units):
    '''
    Sum annual electricity cost across a unit category.

    This metric is separated from total O&M so electricity can be inspected as
    its own cost driver.
    '''
    return sum(_unit_electricity_cost(system, unit) for unit in units)


def _sum_labor_cost(system, units):
    '''
    Sum annual labor cost across a unit category.

    Labor is stored in each unit's ``add_OPEX['labor']`` as an hourly cost, so
    this helper annualizes and aggregates it.
    '''
    return sum(
        _unit_add_opex(unit, 'labor') * system.operating_hours
        for unit in units
    )


def _sum_maintenance_replacement_cost(system, units):
    '''
    Sum annual maintenance and replacement cost across a unit category.

    This separates equipment maintenance/replacement from other O&M items for
    granular TEA reporting.
    '''
    return sum(
        _unit_add_opex(unit, 'maintenance and replacement')
        * system.operating_hours
        for unit in units
    )


def _sum_construction_impact(system, units, indicator):
    '''
    Sum annualized construction impacts across a unit category.

    Used by granular LCA metrics to show which process category drives material
    and equipment embodied impacts.
    '''
    return sum(_unit_construction_impact(system, unit, indicator) for unit in units)


def _sum_operation_impact(system, units, indicator):
    '''
    Sum non-electric operation impacts across a unit category.

    This captures resource streams and direct stream impacts separately from
    construction and electricity.
    '''
    return sum(_unit_operation_impact(system, unit, indicator) for unit in units)


def _sum_electricity_impact(system, units, indicator):
    '''
    Sum electricity impacts across a unit category.

    The same function supports both GWP and eutrophication by accepting the
    requested impact indicator.
    '''
    return sum(_unit_electricity_impact(system, unit, indicator) for unit in units)


def _add_unit_metrics(model):
    '''
    Add granular TEA/LCA metrics by unit category.

    This optional metric set groups similar units, such as pumps, tanks, PBR,
    UF, and centrifuge, so uncertainty outputs can show which process category
    contributes to capital cost, O&M, electricity, labor, construction impacts,
    operation impacts, and electricity impacts.
    '''
    system = model.system
    metrics = list(model.metrics)
    for category, units in _unit_metric_categories(system).items():
        metrics.extend([
            Metric(f'{category} installed cost',
                   lambda us=units: _sum_installed_cost(us),
                   'USD', 'unit TEA'),
            Metric(f'{category} O&M',
                   lambda us=units: _normalize(system, _sum_opex(system, us)),
                   'USD/m3', 'unit TEA'),
            Metric(f'{category} electricity cost',
                   lambda us=units: _normalize(
                       system, _sum_electricity_cost(system, us),
                   ), 'USD/m3', 'unit TEA'),
            Metric(f'{category} labor cost',
                   lambda us=units: _normalize(
                       system, _sum_labor_cost(system, us),
                   ), 'USD/m3', 'unit TEA'),
            Metric(f'{category} maintenance and replacement cost',
                   lambda us=units: _normalize(
                       system, _sum_maintenance_replacement_cost(system, us),
                   ), 'USD/m3', 'unit TEA'),
        ])
        for indicator, unit_label in (
                ('GlobalWarming', 'kg CO2-eq/m3'),
                ('EUT', 'kg N-eq/m3'),
            ):
            metrics.extend([
                Metric(f'{category} construction {indicator}',
                       lambda us=units, i=indicator: _normalize(
                           system, _sum_construction_impact(system, us, i),
                       ), unit_label, 'unit LCA'),
                Metric(f'{category} operation {indicator}',
                       lambda us=units, i=indicator: _normalize(
                           system, _sum_operation_impact(system, us, i),
                       ), unit_label, 'unit LCA'),
                Metric(f'{category} electricity {indicator}',
                       lambda us=units, i=indicator: _normalize(
                           system, _sum_electricity_impact(system, us, i),
                       ), unit_label, 'unit LCA'),
            ])

    try:
        TE_RAW = system.flowsheet.stream.Effluent_unpumped
    except Exception:
        TE_RAW = None
    if TE_RAW is not None:
        metrics.extend([
            Metric('MEM direct discharge GlobalWarming',
                   lambda: _normalize(
                       system, _stream_impacts(system, [TE_RAW], 'GlobalWarming'),
                   ), 'kg CO2-eq/m3', 'unit LCA'),
            Metric('MEM direct discharge Eutrophication',
                   lambda: _normalize(
                       system, _stream_impacts(system, [TE_RAW], 'EUT'),
                   ), 'kg N-eq/m3', 'unit LCA'),
        ])
    model.metrics = metrics

def create_model(
        system=None, include_unit_metrics=False, simulate_baseline=True,
        model_kind='static', t=30, t_step=0.5, method='RK23', print_t=False,
        additional_cost_factor=None, **model_kwargs,
        ):
    '''
    Create a PM2 EcoRecover uncertainty model.

    Parameters
    ----------
    system : System, optional
        Existing system. If omitted, ``pm2_ecorecover_lca.create_system()`` is
        called.
    include_unit_metrics : bool
        Whether to add unit-level TEA/LCA metrics.
    model_kind : {'static', 'dynamic'}
        ``'static'`` keeps the previous fast behavior: run PM2 once, then
        recalculate design/cost/LCA from cached stream states for each sample.
        ``'dynamic'`` reruns PM2 for each Monte Carlo sample, so parameters
        such as HRT, light intensity, and PBR diameter can affect effluent
        concentrations.
    simulate_baseline : bool
        If True, run the PM2 dynamic simulation once during model creation.
    t : float
        Dynamic simulation time [d] for baseline and dynamic-sample runs.
    t_step : float
        Dynamic simulation time step [d].
    method : str, optional
        Dynamic solver method passed to ``pm2_ecorecover_lca.run``.
    print_t : bool
        Whether to print simulation time during baseline dynamic simulation.
    additional_cost_factor : float
        System-level multiplier applied to the sum of unit installed costs in
        TEA. If ``system`` is provided and this argument is None, the existing
        system TEA factor is preserved.
    '''
    model_kind = model_kind.lower()
    if model_kind not in ('static', 'dynamic'):
        raise ValueError("`model_kind` must be either 'static' or 'dynamic'.")

    if system is None:
        if additional_cost_factor is None:
            additional_cost_factor = 2.
        system = pmlca.create_system(
            additional_cost_factor=additional_cost_factor,
        )
    elif (
            additional_cost_factor is not None
            and hasattr(system, 'TEA')
            and hasattr(system.TEA, 'additional_cost_factor')
            ):
        system.TEA.additional_cost_factor = additional_cost_factor
    if simulate_baseline:
        _simulate_baseline(system, t=t, t_step=t_step, method=method,
                           print_t=print_t)
    model = Model(system, specification=_make_specification(
        system, model_kind=model_kind, t=t, t_step=t_step,
        method=method, print_t=print_t),
                  **model_kwargs)
    _add_parameters(model)
    _add_core_metrics(model)
    if include_unit_metrics:
        _add_unit_metrics(model)
    return model


def run_uncertainty(
        model, N=1000, rule='L', seed=None, path='',
        file_name='pm2_ecorecover_lca_uncertainty.xlsx', **kwargs,
        ):
    '''
    Run uncertainty analysis and export a full uncertainty workbook.

    Parameters
    ----------
    model : Model
        Model created by :func:`create_model`.
    N : int
        Number of samples.
    rule : str
        Sampling rule passed to ``Model.sample``; default ``'L'`` is Latin
        hypercube sampling.
    seed : int, optional
        Random seed passed to ``Model.sample`` when supported.
    path : str
        Output Excel path. If omitted, the file is saved in ``results_path``
        with ``xlsx_name``. If a directory is given, ``xlsx_name`` is appended.
    xlsx_name : str
        Output Excel file name when ``path`` is omitted or is a directory.
    kwargs
        Additional arguments passed to :func:`exposan.utils.run_uncertainty`,
        such as ``percentiles`` and ``print_time``.
    '''
    if path:
        if os.path.isdir(path):
            path = os.path.join(path, file_name)
    else:
        path = os.path.join(pmlca.results_path, file_name)
    run(model=model, seed=seed, N=N, rule=rule, path=path, **kwargs)
    return model.table
