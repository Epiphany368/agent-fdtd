"""Birefringent slab transmission and index gradient against the scalar-channel transfer matrix.

Runs benchmarks/tensor_cpml_birefringent_slab.py under its pre-declared
criteria so that the only tensor-slab check against an analytic oracle is
collected as gate evidence. The oracle is the per-eigenpolarization
transfer-matrix transmission of docs/TENSOR_CPML_SLAB_ACCEPTANCE.md; the
benchmark's CRITERIA are not restated here.
"""
import numpy as np
import pytest
import torch

from benchmarks.tensor_cpml_birefringent_slab import CRITERIA, SPEC, oracle, run

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


def test_transfer_matrix_oracle_limits():
    matched = dict(SPEC, slab_indices=list(SPEC['exterior_indices']))
    assert np.allclose(oracle(matched)[0], [1, 1], rtol=0, atol=1e-12)
    weak = dict(SPEC, slab_indices=[n+1e-6 for n in SPEC['exterior_indices']])
    assert all(np.angle(v) < 0 for v in oracle(weak)[0])


@CUDA
def test_birefringent_slab_channels_refine_toward_transfer_matrix_and_index_vjp():
    torch.set_num_threads(1)
    coarse, fine = (run(dz, gradient=(i == 0)) for i, dz in enumerate(SPEC['normal_meshes_um']))
    checks = dict(
        coarse_accuracy=coarse['channel_vector_relative_error'] <= CRITERIA['coarse_channel_vector_relative_error_max'],
        fine_accuracy=fine['channel_vector_relative_error'] <= CRITERIA['fine_channel_vector_relative_error_max'],
        refinement=fine['channel_vector_relative_error'] <= CRITERIA['fine_to_coarse_error_ratio_max']*coarse['channel_vector_relative_error'],
        power=all(abs(a-b) <= CRITERIA['fine_channel_power_absolute_error_max']
                  for a, b in zip(fine['channel_power'], fine['oracle_channel_power'])),
        tail=all(r['tail_rms_to_peak'] <= CRITERIA['tail_rms_to_peak_max'] for row in (coarse, fine) for r in row['records']),
        memory=all(r['cuda_peak_allocated_bytes'] <= r['gpu_reservation_bytes'] for row in (coarse, fine) for r in row['records']))
    value = coarse['records'][1]['index_vjp_per_unit_index']
    expected = coarse['oracle_index_vjp_per_unit_index']
    checks['index_vjp'] = abs(value-expected)/max(abs(expected), 1e-12) <= CRITERIA['index_vjp_relative_error_max']
    print({'coarse_error': coarse['channel_vector_relative_error'], 'fine_error': fine['channel_vector_relative_error'],
           'fine_channel_power': fine['channel_power'], 'oracle_channel_power': fine['oracle_channel_power'],
           'index_vjp': value, 'oracle_index_vjp': expected, 'index_vjp_relative_error': abs(value-expected)/abs(expected),
           'steps': [coarse['steps'], fine['steps']], 'checks': checks})
    assert all(checks.values()), checks
