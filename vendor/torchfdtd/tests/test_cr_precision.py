import json

import numpy as np
import pytest
import torch

from test_cr_optimization import synthetic_inputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_default_fp32_against_fp64_full_objective_and_density_vjp(tmp_path, monkeypatch):
    from benchmarks.cr_spectral_objective import main
    synthetic_inputs(tmp_path)
    common = ['cr_spectral_objective', '--schedule',str(tmp_path/'schedule.json'),
        '--density',str(tmp_path/'seed.npy'),'--context',str(tmp_path/'context.pt'),
        '--mesh','.1','--steps','160','--pml-cells','6','--cpu-threads','1',
        '--reference-cache-mib','1','--gpu-budget-gib','1','--host-budget-gib','1',
        '--slab-width','4','--temporal-depth','4','--forward-kernel','fused','--backward-kernel','fused']
    expected = None
    for mode, precision in [('legacy','float64'),('legacy','float32'),('resident','float32'),('dram','float32')]:
        path = tmp_path/f'{mode}-{precision}.json'
        monkeypatch.setattr('sys.argv', common+['--output',str(path),'--execution-policy',mode,
            *(['--precision',precision] if precision == 'float64' else [])])
        main()
        record = json.loads(path.read_text())
        gradient = torch.from_numpy(np.load(path.with_suffix('.gradient.npy'), allow_pickle=False))
        actual = (torch.tensor(record['response'], dtype=torch.float64),
                  torch.tensor(record['weighted_bits_per_pixel'], dtype=torch.float64), gradient)
        assert gradient.dtype == getattr(torch, precision) and gradient.norm() > 0
        assert record['precision'] == precision
        assert record['restart_contract']['precision'] == precision
        assert record['restart_session']['computed_cases'] == 3
        if expected is None:
            expected = actual
        else:
            for value, reference, tolerance in zip(actual, expected, (1e-4, 1e-4, 1e-3)):
                relative = torch.linalg.vector_norm(value.double()-reference.double())/torch.linalg.vector_norm(reference)
                assert relative < tolerance


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_information_uses_response_precision_with_imported_fp64_context(tmp_path, dtype):
    from benchmarks.cr_spectral_objective import information_objective
    synthetic_inputs(tmp_path)
    context = torch.load(tmp_path/'context.pt', weights_only=True)['context']
    response = torch.full((4, 3), .2, dtype=dtype, requires_grad=True)
    result = information_objective(response, context)
    gradient, = torch.autograd.grad(result.weighted_bits_per_pixel, response)
    assert result.weighted_bits_per_pixel.dtype == result.bits_per_pixel.dtype == dtype
    assert gradient.dtype == dtype and bool(torch.isfinite(gradient).all()) and gradient.norm() > 0
    # Imported data remain unchanged, and a matched FP64 run provides an oracle.
    assert context['scene_covariance'].dtype == torch.float64
    reference = response.detach().double().requires_grad_()
    expected = information_objective(reference, context).weighted_bits_per_pixel
    expected_gradient, = torch.autograd.grad(expected, reference)
    torch.testing.assert_close(result.weighted_bits_per_pixel.double(), expected, rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(gradient.double(), expected_gradient, rtol=1e-4, atol=1e-7)
