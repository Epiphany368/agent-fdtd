"""Proposed constrained-segment design method for a future G7-01 revision.

This module is development code. The r3 judged case remains a failed result;
the proposed held-out r4 seeds must not run before a case revision is approved.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from torchfdtd.design_problem import DesignProblem
from torchfdtd.fabrication import measure_feature_sizes

from . import workflow as w
from .constrained_segments import ThreeRidgeDensity, logits_from_runs


def design_seed(seed, objective, settings, case, *, checkpoint=None):
    """Six Adam steps; choose initial or final hard design by the short objective."""
    started = time.perf_counter()
    if checkpoint is not None:
        raise ValueError('The six-step constrained design does not yet support checkpoint replay.')
    fixture = case['fixture']
    spec = fixture['three_ridge_segments']
    initial = logits_from_runs(tuple(spec['initial_runs']))
    jitter = float(spec['initial_logit_jitter_std']) * torch.randn(
        (6, 1), generator=torch.Generator().manual_seed(int(seed)))
    design = ThreeRidgeDensity(initial=initial+jitter, beta=float(fixture['projection_beta'][0]))
    optimizer = torch.optim.Adam(design.parameters(), lr=float(fixture['learning_rate']))
    problem = DesignProblem(design, objective, optimizer, name=f'g7-01r4-seed{seed}')
    starting_binary = problem.density(hard=True)
    starting_loss, starting_metrics = problem.evaluate(starting_binary)
    starting_runs = design.integer_runs()
    history = problem.run(int(fixture['iterations_per_beta']))
    final_binary = problem.density(hard=True)
    final_loss, final_metrics = problem.evaluate(final_binary)
    final_runs = design.integer_runs()
    choose_final = final_loss < starting_loss
    selected = final_binary if choose_final else starting_binary
    selected_loss, selected_metrics = ((final_loss, final_metrics) if choose_final
                                       else (starting_loss, starting_metrics))
    pixel_um = float(fixture['pixel_um'])
    fabrication = fixture['fabrication']
    raw = selected.bool().cpu().numpy()
    side = int(round(fabrication['min_linewidth_um']/pixel_um))
    processed = w.open_close(raw, side)
    sizes = measure_feature_sizes(processed, pixel_um,
                                  boundary=(fabrication['boundary'], 'extend'))
    violations = list(sizes.violations(
        min_linewidth_um=fabrication['min_linewidth_um'],
        min_gap_um=fabrication['min_gap_um']))
    if np.any(processed != raw):
        raise ValueError('The constrained segment design changed under the declared open-close.')
    return dict(seed=int(seed), iterations=problem.iteration,
                filter_radius_um=0., design_wall_seconds=time.perf_counter()-started,
                iteration_seconds=sum(row['elapsed_s'] for row in history),
                initial_logits=(initial+jitter)[:, 0].tolist(),
                final_logits=design.design.detach()[:, 0].tolist(),
                history=history, binary=processed[:, 0].astype(int).tolist(),
                smooth_density=problem.density()[:, 0].tolist(),
                selected_hard_design='final' if choose_final else 'initial',
                short_design_scores=dict(initial=-starting_loss, final=-final_loss,
                                         selected=-selected_loss),
                short_design_metrics=dict(initial=starting_metrics, final=final_metrics),
                initial_runs=list(starting_runs), final_runs=list(final_runs),
                design_mesh_binary=dict(objective=selected_loss, metrics=selected_metrics),
                fabrication=dict(feature_sizes=sizes.report(), violations=violations,
                                 satisfies_declared_constraints=not violations,
                                 declared=dict(min_linewidth_um=fabrication['min_linewidth_um'],
                                               min_gap_um=fabrication['min_gap_um'],
                                               boundary=[fabrication['boundary'], 'extend']),
                                 measured='the selected hard binary design after the open and close'),
                open_close=dict(side_pixels=side,
                                boundary=[fabrication['boundary'], 'extend'],
                                changed_pixels=int(np.count_nonzero(processed != raw)),
                                thresholded=raw[:, 0].astype(int).tolist(),
                                processed=processed[:, 0].astype(int).tolist()))
