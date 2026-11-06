from functools import partial
import pytest
import torch
from benchmarks.cr_spectral_objective import check_density_direction
from torchfdtd import (periodic_layer_response,spectral_pupil_response,PlaneReferenceCache,
    AdjointOptions,spectral_electron_model,exposure_target_information)


def test_fused_multicase_information_gradient_with_reference_cache():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    density=torch.tensor([[.2,.4],[.6,.3]],device='cuda',dtype=torch.float64,requires_grad=True)
    cache=PlaneReferenceCache(8*1024**2)
    def cases(backward):
        return [[partial(periodic_layer_response,spec=dict(wavelength_um=wl,background_index=1.4,
            design_index=n,period_um=(.8,.8),height_um=.2,detector_offset_um=.5,
            theta_inside_rad=theta,phi_rad=phi),mesh=.1,steps=320,pml_cells=6,
            quadrature_counts=(4,4),reference_cache=cache,forward_kernel='fused',
            options=AdjointOptions(checkpoints=3,backward_kernel=backward))
            for theta,phi in ((.08,.3),(.12,-.5))] for wl,n in ((.5,1.8),(.6,1.9))]
    weights=[.3,.6]
    def score(response):
        eye=torch.eye(2,dtype=response.dtype,device=response.device)
        model=spectral_electron_model(response,[500,600],[.7,.8],[1.,1.5],eye,calibration=1e-10)
        return exposure_target_information(model,eye,.6*eye,eye,exposure_scales=[.02,.08,.32],
            probabilities=[.2,.5,.3],read_noise_e_rms=1.5,raw_pixels=4).weighted_bits_per_pixel
    direct=torch.stack([sum(case(density).mean(0)*w for case,w in zip(row,weights))
        for row in cases('torch')],1).cpu()
    expected=score(direct);eg,=torch.autograd.grad(expected,density)
    replay_cases=cases('fused')
    def objective(d):
        return score(spectral_pupil_response(replay_cases,d,weights,replay_rtol=1e-10,replay_atol=1e-30))
    actual=objective(density);ag,=torch.autograd.grad(actual,density)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(ag,eg,rtol=1e-9,atol=1e-12)
    direction=density.new_tensor([[.2,-.3],[.4,.1]]);h=1e-4
    with torch.no_grad():fd=(objective(density+h*direction)-objective(density-h*direction))/(2*h)
    torch.testing.assert_close((ag*direction).sum().cpu(),fd,rtol=2e-4,atol=1e-9)
    assert ag.norm()>1e-5
    assert cache.misses==8 and cache.hits>=24
    assert 0<cache.tensor_bytes<=cache.budget_bytes
    sweep=check_density_direction(objective,density,ag,[.001,.0005],rtol=2e-4,atol=1e-9)
    assert sweep['passed'] and all(row['passed'] for row in sweep['rows'])
