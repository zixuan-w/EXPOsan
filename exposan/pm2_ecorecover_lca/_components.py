#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
EXPOsan: Exposition of sanitation and resource recovery systems

This module is developed by:

    Zixuan Wang <wyatt4428@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/EXPOsan/blob/main/LICENSE.txt
for license details.
'''

from qsdsan import Component, Components, set_thermo as qs_set_thermo
from qsdsan import processes as pc

__all__ = ('create_components',)


def create_components(set_thermo=True):
    '''
    Create PM2 process components plus LCA-only resource components.

    The dynamic process model needs the standard PM2 components, while the
    TEA/LCA model also needs components for purchased resources such as citric
    acid, sodium hypochlorite, natural gas, and CO2. This function combines
    them into one component set so resource streams can carry mass, price, and
    impact items without entering PM2 reactions.
    '''
    pm2_cmps = pc.create_pm2_cmps(set_thermo=False)

    # Source: PM2 components from QSDsan, with resource components added for
    # LCA-only input streams in this module.
    CitricAcid = Component('CitricAcid', formula='C6H8O7',
                           phase='s', particle_size='Soluble',
                           degradability='Readily', organic=True)

    SodiumHypochlorite = Component('SodiumHypochlorite', formula='NaClO',
                                   phase='s', particle_size='Soluble',
                                   degradability='Undegradable', organic=False)
    # Assume the same liquid volume model as water for soluble resource chemicals.
    CitricAcid.copy_models_from(pm2_cmps.H2O, ('V',))
    SodiumHypochlorite.copy_models_from(pm2_cmps.H2O, ('V',))

    # Natural gas is represented as methane for LCA/resource accounting.
    # This follows the EXPOsan reclaimer LPG-component convention of adding a
    # simple gas-phase fuel component for external fuel inputs.
    NaturalGas = Component('NaturalGas', search_ID='methane', formula='CH4',
                           phase='g', particle_size='Dissolved gas',
                           degradability='Slowly', organic=True)

    cmps = Components((*pm2_cmps, CitricAcid, SodiumHypochlorite, NaturalGas))

    for cmp in cmps:
        for attr in ('HHV', 'LHV', 'Hf'):
            if getattr(cmp, attr) is None:
                setattr(cmp, attr, 0)

    cmps.compile(ignore_inaccurate_molar_weight=True)

    if set_thermo:
        qs_set_thermo(cmps)

    return cmps
