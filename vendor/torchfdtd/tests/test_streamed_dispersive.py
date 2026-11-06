"""Complete ADE block Jacobians, including pole state and overlapping halos."""
from dataclasses import replace

import pytest
import torch

from torchfdtd import (BoundaryFace, DispersiveSimulation, DispersivePlaneSimulation, Region,
                         StreamedDispersiveSimulation, StreamedAdjointOptions)
from torchfdtd.streamed_dispersive import (_SlabDispersiveSystem, DispersiveSlabBlockOperator,
                                             _DispersiveExecution)
from torchfdtd.dispersive_adjoint import _DispersiveSystem
from test_dispersive_adjoint import project


def scene(bloch=False, steps=12, precision='float64'):
    p = project(steps=steps, precision=precision)
    if bloch:
        p.region.boundaries.x_min = BoundaryFace(kind='bloch')
        p.region.boundaries.x_max = BoundaryFace(kind='bloch')
        p.region.bloch_phase = (.43, 0, 0)
    return p


def inputs(p, layout, requires_grad=False):
    dtype = getattr(torch, p.region.precision)
    rng = torch.Generator().manual_seed(492)
    shapes = {
        'shared': [p.region.shape, (2,), (), (2,)],
        'spatial': [p.region.shape, (2,*p.region.shape), (2,), (2,*p.region.shape)],
        'diagonal': [(*p.region.shape,3), (2,*p.region.shape,3), (2,*p.region.shape,3), ()],
    }[layout]
    scales = (1., 1e30, 1e15, 1e14)
    return tuple(((1.5 if i == 0 else .2)+.2*torch.rand(s, generator=rng, dtype=dtype)).mul(scale).requires_grad_(requires_grad)
                 for i, (s, scale) in enumerate(zip(shapes, scales)))


@pytest.mark.parametrize('layout', ['shared', 'spatial', 'diagonal'])
@pytest.mark.parametrize('bloch,depth', [(False, 2), (True, 2), (True, 18)])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_complete_block_jacobian(layout, bloch, depth, device):
    if device == 'cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    p = scene(bloch, max(12,depth+1))
    if layout == 'spatial' and depth == 2:
        p.region = Region.model_validate({**p.region.model_dump(), 'dimension':'3d'})
    values = inputs(p, layout)
    parameters, specification = DispersiveSimulation(p)._pack(*values)
    host = _SlabDispersiveSystem(p, values[0], parameters, specification, prepare_updates=False)
    oracle = _DispersiveSystem(p, values[0], parameters, specification)
    torch.manual_seed(934)
    state = tuple(torch.randn_like(s)*.03 for s in host.state())
    endpoint = tuple(torch.randn_like(s)*.04 for s in host.state())
    original = tuple(s.clone() for s in state)
    weights = torch.randn(depth, len(host.monitors), dtype=host.field_dtype)
    operator = DispersiveSlabBlockOperator(host, 3, device, local_checkpoints=2)
    if depth == 18:
        assert any(len(d[2])>2*p.region.shape[0] for d in operator.tiles(depth))
    actual, signals = operator.forward(parameters, state, 1, depth)
    bars, gradient = operator.transpose(parameters, state, 1, depth, endpoint, weights)
    differentiable = parameters.clone().requires_grad_()
    initial = tuple(s.clone().requires_grad_() for s in host._pole_state(state))
    current, observed = initial, []
    for j in range(depth):
        current = oracle.reference_step(current, j+1, differentiable)
        observed.append(oracle.observe(current))
    observed = torch.stack(observed)
    objective = (observed.conj()*weights).real.sum()
    objective += sum((s.conj()*b).real.sum() for s,b in zip(current, host._pole_state(endpoint)))
    expected = torch.autograd.grad(objective, (*initial, differentiable))
    for a,b in zip(actual, host._slab_state(current)):torch.testing.assert_close(a,b,rtol=2e-10,atol=2e-12)
    for a,b in zip(bars, host._slab_state(expected[:-1])):torch.testing.assert_close(a,b,rtol=2e-9,atol=2e-11)
    torch.testing.assert_close(signals, observed, rtol=2e-10,atol=2e-12)
    torch.testing.assert_close(gradient, expected[-1], rtol=2e-9,atol=2e-11)
    for a,b in zip(state, original):torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('storage', ['host', 'disk'])
@pytest.mark.parametrize('checkpoints', [0, 2])
def test_public_geometry_and_material_vjp(tmp_path, storage, checkpoints):
    p = scene(True)
    variables = torch.tensor([1.6, .4, .8, .12], dtype=torch.float64, requires_grad=True)
    def arguments(v):
        return v[0].expand(p.region.shape), v[1:2]*1e30, v[2]*1e15, v[3]*1e15
    options = StreamedAdjointOptions(device='cpu', slab_width=3, temporal_depth=3,
        local_checkpoints=2, checkpoints=checkpoints, state_storage=storage,
        state_directory=tmp_path, disk_budget_bytes=64*1024**2)
    result = StreamedDispersiveSimulation(p, options)(*arguments(variables))
    got, = torch.autograd.grad(result.signals.abs().square().sum(), variables)
    reference = DispersiveSimulation(p).reference(*arguments(variables))
    want, = torch.autograd.grad(reference.abs().square().sum(), variables)
    torch.testing.assert_close(result.signals, reference, rtol=2e-11,atol=2e-12)
    torch.testing.assert_close(got, want, rtol=2e-10,atol=2e-12)
    assert result.report['material_state_bytes'] > 0
    assert result.report['host_initial_state_storage_bytes'] < 1024
    if storage == 'disk':
        assert result.report['backward_backing_store']['closed']
        assert result.report['backward_backing_store']['live_logical_file_bytes'] == 0
        assert not list(tmp_path.iterdir())


def test_admission_includes_poles_and_precedes_pack(tmp_path, monkeypatch):
    p = scene()
    values = inputs(p, 'spatial')
    parameters, layout = DispersiveSimulation(p)._pack(*values)
    execution = _DispersiveExecution(layout)
    options = StreamedAdjointOptions(device='cpu', slab_width=3, temporal_depth=2,
        state_storage='disk', state_directory=tmp_path/'scratch', disk_budget_bytes=64*1024**2)
    report = execution.reservation(p, parameters, options, None)
    from torchfdtd import estimate_streamed_dispersive_memory
    assert report == estimate_streamed_dispersive_memory(p,layout.shapes,options)
    host = execution.host(p, parameters, None)
    assert report['state_bytes'] == sum(s.numel()*s.element_size() for s in host.state())
    monkeypatch.setattr(torch, 'cat', lambda *a, **kw: pytest.fail('Packed before budget rejection'))
    with pytest.raises(ValueError,match='disk budget'):
        StreamedDispersiveSimulation(p, replace(options,disk_budget_bytes=report['disk_reservation_bytes']-1))(*values)
    with pytest.raises(ValueError,match='host budget'):
        StreamedDispersiveSimulation(p, replace(options,host_budget_bytes=1))(*values)
    assert not (tmp_path/'scratch').exists()


@pytest.mark.parametrize('device,transfers', [('cpu','sync'),('cuda','sync'),('cuda','async')])
def test_spectrum_and_plane_flux(tmp_path, device, transfers):
    if device == 'cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from test_adjoint_planes import scene as plane_scene
    p = plane_scene(); p.region.steps = 18
    options = StreamedAdjointOptions(device=device, slab_width=3, temporal_depth=3,
        tile_transfers=transfers, local_checkpoints=1, checkpoints=1,
        state_storage='disk', state_directory=tmp_path, disk_budget_bytes=128*1024**2)
    epsilon = torch.full(p.region.shape, 1.7, dtype=torch.float64)
    rate = torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    frequencies = [.03/p.region.time_step,.06/p.region.time_step]
    def loss(model, value):
        planes = model(epsilon, value[None]*1e30, 1.4e15, 2e14, frequencies)
        return sum(plane.flux().sum() for plane in planes.values())
    streamed = DispersivePlaneSimulation(p,options)
    resident = DispersivePlaneSimulation(p)
    actual, expected = loss(streamed,rate), loss(resident,rate)
    a, = torch.autograd.grad(actual,rate)
    b, = torch.autograd.grad(expected,rate)
    torch.testing.assert_close(actual,expected,rtol=2e-10,atol=1e-25)
    torch.testing.assert_close(a,b,rtol=2e-9,atol=1e-25)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('precision,bloch', [('float32',False),('float32',True),('float64',True)])
@pytest.mark.parametrize('device,transfers,reuse', [('cpu','sync',False),('cuda','sync',True),('cuda','async',True)])
def test_precision_spectral_taylor_and_repeat_backward(precision, bloch, device, transfers, reuse):
    if device == 'cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    p = scene(bloch, steps=15, precision=precision)
    dtype = getattr(torch,precision)
    v = torch.tensor([1.6,.45,.8,.13],dtype=dtype,requires_grad=True)
    options = StreamedAdjointOptions(device=device, slab_width=3, temporal_depth=4,
        tile_transfers=transfers, tile_buffers=3, reuse_tile_buffers=reuse, local_checkpoints=2)
    model = StreamedDispersiveSimulation(p,options)
    frequency = [.03/p.region.time_step,.07/p.region.time_step]
    def args(t):return t[0].expand(p.region.shape),t[1:2]*1e30,t[2]*1e15,t[3]*1e15
    def loss(t):
        r = model.spectrum(*args(t),frequency,block_size=3)
        return (r.fields.abs()/(p.region.steps*p.region.time_step)).square().sum()
    value = loss(v)
    gradient, = torch.autograd.grad(value,v,retain_graph=True)
    repeated, = torch.autograd.grad(value,v)
    torch.testing.assert_close(gradient,repeated,rtol=0,atol=0)
    ref = DispersiveSimulation(p).spectrum(*args(v),frequency,block_size=3)
    expected = (ref.fields.abs()/(p.region.steps*p.region.time_step)).square().sum()
    want, = torch.autograd.grad(expected,v)
    tol = 3e-5 if precision=='float32' else 2e-9
    torch.testing.assert_close(value,expected,rtol=tol,atol=1e-9)
    torch.testing.assert_close(gradient,want,rtol=tol,atol=1e-9)
    if precision=='float64':
        direction = v.new_tensor([.2,-.3,.4,.1]); h = 1e-4
        finite = (loss(v.detach()+h*direction)-loss(v.detach()-h*direction))/(2*h)
        torch.testing.assert_close(gradient@direction,finite,rtol=2e-6,atol=1e-10)


def test_file_cleanup_after_transpose_failure(tmp_path, monkeypatch):
    p = scene()
    values = inputs(p,'shared',True)
    options = StreamedAdjointOptions(device='cpu',slab_width=3,temporal_depth=3,
        state_storage='disk',state_directory=tmp_path,disk_budget_bytes=64*1024**2)
    model = StreamedDispersiveSimulation(p,options)
    result = model(*values)
    def fail(*args):raise RuntimeError('Injected ADE transpose failure')
    monkeypatch.setattr(_SlabDispersiveSystem,'transpose_step',fail)
    with pytest.raises(RuntimeError,match='Injected ADE'):
        result.signals.abs().square().sum().backward()
    assert result.report['backward_backing_store']['closed']
    assert result.report['backward_backing_store']['live_logical_file_bytes'] == 0
    assert not list(tmp_path.iterdir())


def test_cuda_allocation_scales_with_tile_not_domain():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    import gc
    peaks = []
    for nx in (24,48):
        p = scene(True,steps=10,precision='float32')
        data = p.region.model_dump(); data['size']=(nx*.1,*p.region.size[1:])
        p.region = Region.model_validate(data)
        values = inputs(p,'shared',True)
        model = StreamedDispersiveSimulation(p, StreamedAdjointOptions(device='cuda',
            slab_width=4,temporal_depth=2,checkpoints=1,local_checkpoints=1,tile_transfers='async'))
        gc.collect();torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        was_enabled=gc.isenabled();gc.disable()
        try:
            result=model(*values)
            result.signals.abs().square().sum().backward()
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated()-baseline)
            assert peaks[-1] <= result.report['gpu_reservation_bytes']
            del result,model,values
            assert torch.cuda.memory_allocated()==baseline
        finally:
            if was_enabled:gc.enable()
    assert peaks[1] <= peaks[0]*1.05+4096,peaks
