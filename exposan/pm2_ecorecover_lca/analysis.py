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
from exposan import pm2_ecorecover_lca as pl
from exposan.pm2_ecorecover_lca.models import create_model, run_uncertainty

__all__ = ('run',)

def run(
        N=1000, seed=None, include_unit_metrics=False,
        t=100, t_step=0.1, method=None,
        additional_cost_factor=2.,
        path='', xlsx_name='pm2_ecorecover_lca_uncertainty.xlsx',
        print_t=False, **evaluate_kwargs,
        ):
    '''
    Run a basic uncertainty analysis for the PM2 EcoRecover LCA system.

    The PM2 dynamic simulation is run once while creating the model. Subsequent
    Monte Carlo samples recalculate design, cost, and LCA metrics from cached
    stream states.

    additional_cost_factor : float
        System-level multiplier applied to TEA installed equipment cost after
        unit-specific bare-module factors are applied.
    '''
    model = create_model(
        include_unit_metrics=include_unit_metrics,
        simulate_baseline=True,
        t=t,
        t_step=t_step,
        method=method,
        print_t=print_t,
        additional_cost_factor=additional_cost_factor,
    )

    path = path or os.path.join(pl.results_path, xlsx_name)
    table = run_uncertainty(
        model, N=N, seed=seed, path=path, xlsx_name=xlsx_name, **evaluate_kwargs,
    )
    return model, table


if __name__ == '__main__':
    run()
