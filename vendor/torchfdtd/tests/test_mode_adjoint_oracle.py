"""Mode-amplitude objectives through the checkpointed adjoint versus full Torch autograd.

The 600-step scene lets the two-cycle pulse pass the detector completely."""
import numpy as np
import pytest
import torch

from test_mode_branches import tiny_project, yee_epsilon, guide, CORE, CLAD
from torchfdtd import AdjointOptions, FieldMonitor, StreamedAdjointOptions
from torchfdtd.mode_branches import prepare_aperture_modal_launch
from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_s_parameters

FREQUENCY = [299792458/1.55e-6]


def scene(device, precision='float64', steps=600):
    p = tiny_project(precision, steps=steps)
    p.monitors = [FieldMonitor(id='out', normal='x', center=(.3, 0., 0.), size=(0., 1.2, .5))]
    launch = prepare_aperture_modal_launch(p, guide)
    base = yee_epsilon(p.region, lambda x, y, z: guide(y, z), device)
    mask = torch.zeros_like(base)
    mask[9:13, 4:10] = 1
    direction = torch.randn(base.shape, generator=torch.Generator().manual_seed(3), dtype=base.dtype).to(device)*mask
    return p, launch, base, direction


def objective(model, value, reference, launch):
    """A mode amplitude ratio and its squared modulus, both linear in the plane fields."""
    t = modal_s_parameters(model(value, FREQUENCY)['out'], reference, launch)['transmission'][0]
    return t.real+.3*t.imag+t.abs().square()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_checkpointed_modal_adjoint_matches_full_autograd_taylor_and_central_difference(device):
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA is unavailable')
        pytest.importorskip('cupy')
    p, launch, base, direction = scene(device)
    model = ModeInjectedPlaneSimulation(p, launch, AdjointOptions(checkpoints=3))
    with torch.no_grad():
        reference = model(base, FREQUENCY)['out']
    x = base.clone().requires_grad_()
    value = objective(model, x, reference, launch)
    gradient, = torch.autograd.grad(value, x)
    x_ref = base.clone().requires_grad_()
    t = modal_s_parameters(model.reference(x_ref, FREQUENCY)['out'], reference, launch)['transmission'][0]
    oracle, = torch.autograd.grad(t.real+.3*t.imag+t.abs().square(), x_ref)
    torch.testing.assert_close(value.detach(), torch.tensor(2., dtype=value.dtype, device=device), atol=1e-9, rtol=0)
    assert oracle.abs().max() > 1e-4
    torch.testing.assert_close(gradient, oracle, rtol=1e-9, atol=1e-16)
    # The frozen source neighbourhood carries exactly zero derivative.
    w = launch.electric_index
    assert torch.count_nonzero(gradient[w-1:w+2, 1:13]) == 0
    slope = float((gradient*direction).sum())
    residuals, central = [], []
    with torch.no_grad():
        for h in (2e-3, 1e-3, 5e-4):
            plus = float(objective(model, base+h*direction, reference, launch))
            minus = float(objective(model, base-h*direction, reference, launch))
            residuals.append(abs(plus-float(value)-h*slope))
            central.append((plus-minus)/(2*h))
    # Second-order Taylor remainder: halving the step quarters the residual.
    assert residuals[1] < .3*residuals[0] and residuals[2] < .3*residuals[1]
    assert abs(central[-1]-slope) < 1e-6*abs(slope)
    print({'device': device, 'gradient_vs_autograd': float((gradient-oracle).abs().max()/oracle.abs().max()),
           'taylor_residuals': residuals, 'central_difference': central[-1], 'directional_derivative': slope})


def test_full_autograd_oracle_rejects_streamed_and_large_problems():
    p, launch, base, _ = scene('cpu', 'float32', steps=12)
    streamed = ModeInjectedPlaneSimulation(p, launch, StreamedAdjointOptions(device='cpu', slab_width=4, temporal_depth=2, checkpoints=1))
    with pytest.raises(ValueError, match='resident'):
        streamed.reference(base, FREQUENCY)
    model = ModeInjectedPlaneSimulation(p, launch, AdjointOptions(checkpoints=1))
    model.model.project.region.steps = 10**7
    with pytest.raises(ValueError, match='estimated .* bytes of retained graph'):
        model.reference(base, FREQUENCY)
