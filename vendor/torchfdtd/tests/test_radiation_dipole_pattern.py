"""Native FDTD dipole far-field pattern against sin^2(theta) under mesh refinement.

Runs benchmarks/radiation_dipole.py, the only check of the native time-domain
dipole radiation pattern against an analytic oracle, so that it is collected
as gate evidence. Physical box, PML thickness and simulation time are fixed
across the three meshes; the pattern error must decrease and end below the
benchmark's one percent limit.
"""
import pytest
import torch

from benchmarks.radiation_dipole import run

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


@CUDA
def test_native_dipole_pattern_error_decreases_with_mesh_and_ends_below_one_percent():
    report = run('cuda')
    cases = report['cases']
    assert [c['mesh_um'] for c in cases] == [.1, .075, .05]
    times = [c['physical_time_s'] for c in cases]
    assert max(times)-min(times) < 1.5*max(c['physical_time_s']/c['steps'] for c in cases)
    assert all(abs(c['pml_um']-.3) < 1e-12 for c in cases)
    errors = [c['pattern_relative_l2'] for c in cases]
    print({'mesh_um': [c['mesh_um'] for c in cases], 'shape': [c['shape'] for c in cases],
           'steps': [c['steps'] for c in cases], 'pattern_relative_l2': errors,
           'wall_s': [c['full_wall_s'] for c in cases], 'passed': report['passed']})
    assert errors[1] < errors[0] and errors[2] < errors[1]
    assert errors[-1] < .01
    assert report['passed']
