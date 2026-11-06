"""Slab transmission and its thickness and permittivity derivatives against the Airy formula.

Runs benchmarks/gradient_mesh.py, the only check of a physical parameter
gradient (slab thickness in um and slab permittivity) against an analytic
derivative, so that it is collected as gate evidence. The physical domain,
90 fs window and 0.4 um PML are fixed across the joint mesh/width refinement;
the duration and PML controls are separate cases whose outputs must not move.
The 3 percent gradient limit is the program's physical-parameter gradient
threshold; the benchmark applies it after its regularization followup.
"""
import numpy as np
import pytest
import torch

from benchmarks.gradient_mesh import run_case, summarize

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
CASES = [('joint', .04, .16, 90., .4), ('joint', .02, .12, 90., .4), ('joint', .01, .08, 90., .4),
         ('regularization_followup', .01, .06, 90., .4), ('fixed_width', .04, .12, 90., .4),
         ('fixed_width', .01, .12, 90., .4), ('duration_control', .01, .06, 120., .4),
         ('pml_control', .01, .06, 90., .5)]


@CUDA
def test_slab_thickness_and_epsilon_derivatives_refine_toward_airy_with_controls():
    report = dict(precision='float32', device='cuda', cases=[])
    for group, h, w, duration, pml in CASES:
        row = run_case(h, w, 'cuda', duration, pml, improve=(group == 'regularization_followup'))
        row['group'] = group
        report['cases'].append(row)
    summarize(report)
    final = next(row for row in report['cases'] if row['group'] == 'regularization_followup')
    print({'joint': [(r['mesh_um'], r['width_um'], r['absolute_errors'][0], r['relative_gradient_errors'])
                     for r in report['cases'] if r['group'] == 'joint'],
           'followup': (final['mesh_um'], final['width_um'], final['absolute_errors'][0], final['relative_gradient_errors']),
           'analytic': final['analytic'], 'measured': [final['transmission'], final['thickness_derivative_per_um'], final['epsilon_derivative']],
           'conservation_max': max(r['conservation_error'] for r in report['cases']),
           'controls': {r['group']: r['changes_from_baseline'] for r in report['cases'] if 'changes_from_baseline' in r},
           'elapsed_seconds': sum(r['elapsed_seconds'] for r in report['cases']), 'checks': report['checks']})
    assert max(final['relative_gradient_errors']) < .03
    assert all(report['checks'].values()), report['checks']
