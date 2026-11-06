"""G4-02: every valid CUDA execution path reproduces the CPU float64 solution of one tiny problem.

Case file: docs/validation/cases/G4-02_cuda_valid_path_matrix.json. The forward
matrix is torch/fused Yee kernel x CUDA graph off/on x torch/fused plane DFT x
float32/float64 x real/complex (Bloch) fields x default/side stream; the adjoint
matrix is torch/fused forward x torch/fused backward x precision x fields x
stream with a scalar loss. Combinations the code does not support are asserted
as named errors, never as a silent fallback. Comparisons are on quantities
normalized by the reference maximum, at the program's layer-A tolerances.
"""
import contextlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from torchfdtd import (AdjointOptions, BoundaryFace, DifferentiableSimulation, FieldMonitor, Material, Monitor,
                       Project, Region, Simulation, Source, SpectrumSettings, Structure)
from torchfdtd.solver import voxelize

CASE = json.loads((Path(__file__).resolve().parents[1] / 'docs' / 'validation' / 'cases' / 'G4-02_cuda_valid_path_matrix.json').read_text(encoding='utf-8'))
LIMITS = {precision: (CASE['acceptance'][precision]['rtol'], CASE['acceptance'][precision]['atol']) for precision in ('float32', 'float64')}
F = CASE['fixture']
STEPS = F['steps']


def gpu():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')


def tiny(precision='float64', backend='cpu', fields='real', kernel='torch', monitor='torch', *, plane=True):
    """The one discrete problem of the case file, placed on a backend and kernel choice."""
    r = Region(dimension='3d', size=tuple(F['size_um']), mesh=F['mesh_um'], pml_cells=F['pml_cells'], steps=STEPS,
               precision=precision, backend=backend, cuda_kernel=kernel, cuda_monitor_kernel=monitor, snapshot_interval=10000)
    if fields == 'complex':
        r.boundaries.y_min = BoundaryFace(kind='bloch')
        r.boundaries.y_max = BoundaryFace(kind='bloch')
        r.bloch_phase = (0, F['bloch_phase_y_rad'], 0)
    monitors = [Monitor(center=tuple(F['point_monitor_um']), component='Ez')]
    if plane:
        monitors.append(FieldMonitor(id='plane', normal='x', center=(F['plane_x_um'], 0, 0), size=(0, *F['plane_size_um']),
                                     spectrum=SpectrumSettings(sampling='frequency', frequency_points=F['plane_frequency_points'], apodization='none')))
    return Project(region=r, materials=[Material(name='glass', index=F['sphere_index'])],
                   structures=[Structure(kind='sphere', center=tuple(F['sphere_center_um']), radius=F['sphere_radius_um'], material='glass')],
                   sources=[Source(center=tuple(F['source_um']), component='Ez', wavelength=F['wavelength_um'], pulse_cycles=1)],
                   monitors=monitors)


def stream_context(stream):
    if stream == 'default':
        return contextlib.nullcontext()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    return torch.cuda.stream(side)


def assert_normalized_close(got, expected, precision, label):
    """|got - expected| <= atol + rtol |expected| after dividing both by max |expected|."""
    rtol, atol = LIMITS[precision]
    got, expected = np.asarray(got, dtype=np.complex128), np.asarray(expected, dtype=np.complex128)
    scale = np.max(np.abs(expected))
    assert scale > 0, f'{label}: the reference is identically zero, which would make the comparison vacuous'
    np.testing.assert_allclose(got / scale, expected / scale, rtol=rtol, atol=atol, err_msg=label)


@pytest.fixture(scope='module')
def references():
    return {fields: Simulation(tiny('float64', 'cpu', fields)).run(cuda_graph=False) for fields in ('real', 'complex')}


def compare_forward(got, ref, precision):
    assert got.summary['steps'] == STEPS and got.summary['field_peak'] > 1e-3
    for key in ('electric', 'magnetic', 'signals'):
        assert_normalized_close(getattr(got, key), getattr(ref, key), precision, key)
    plane, expected = got.field_monitor('plane'), ref.field_monitor('plane')
    assert_normalized_close(plane['fields'], expected['fields'], precision, 'plane DFT fields')
    assert_normalized_close(plane['flux'], expected['flux'], precision, 'plane flux')


@pytest.mark.parametrize('stream', ['default', 'side'])
@pytest.mark.parametrize('fields', ['real', 'complex'])
@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('monitor', ['torch', 'fused'])
@pytest.mark.parametrize('graph', [False, True], ids=['eager', 'graph'])
@pytest.mark.parametrize('kernel', ['torch', 'fused'])
def test_forward_cell(kernel, graph, monitor, precision, fields, stream, references):
    gpu()
    p = tiny(precision, 'cuda', fields, kernel, monitor)
    unsupported = fields == 'complex' and (kernel == 'fused' or monitor == 'fused')
    with stream_context(stream):
        if unsupported:
            # Fused kernels and fused plane DFT are real-field paths; Bloch fields must be refused by name.
            expected = 'The fused CUDA kernel currently supports real fields' if kernel == 'fused' else 'Fused frequency monitors require real CUDA fields'
            with pytest.raises(ValueError, match=expected):
                Simulation(p).run(cuda_graph=graph)
            return
        got = Simulation(p).run(cuda_graph=graph)
    torch.cuda.synchronize()
    assert got.summary['backend'] == 'cuda' and got.summary['cuda_graph'] is graph
    assert got.summary['cuda_kernel'] == kernel and got.summary['cuda_monitor_kernel'] == monitor
    assert got.summary['complex_fields'] is (fields == 'complex') and got.summary['precision'] == precision
    compare_forward(got, references[fields], precision)


def adjoint_inputs(fields, precision, device, kernel='torch'):
    p = tiny(precision, 'cpu', fields, kernel, plane=False)
    epsilon, _ = voxelize(tiny('float64', 'cpu', fields, plane=False))
    dtype = torch.float64 if precision == 'float64' else torch.float32
    return p, torch.as_tensor(np.array(epsilon, dtype=np.float64), device=device, dtype=dtype).requires_grad_(True)


def scalar_loss(signals):
    return (signals.real.square() + signals.imag.square()).sum() if signals.is_complex() else signals.square().sum()


@pytest.fixture(scope='module')
def adjoint_references():
    out = {}
    for fields in ('real', 'complex'):
        p, epsilon = adjoint_inputs(fields, 'float64', 'cpu')
        result = DifferentiableSimulation(p, AdjointOptions(checkpoints=F['adjoint_checkpoints']))(epsilon)
        gradient, = torch.autograd.grad(scalar_loss(result.signals), epsilon)
        assert result.report['forward_backend'] == 'torch CPU' and result.report['backward_backend'] == 'torch explicit transpose'
        out[fields] = (result.signals.detach(), gradient)
    return out


@pytest.mark.parametrize('stream', ['default', 'side'])
@pytest.mark.parametrize('fields', ['real', 'complex'])
@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('backward', ['torch', 'fused'])
@pytest.mark.parametrize('kernel', ['torch', 'fused'])
def test_adjoint_cell(kernel, backward, precision, fields, stream, adjoint_references):
    gpu()
    p, epsilon = adjoint_inputs(fields, precision, 'cuda', kernel)
    with stream_context(stream):
        result = DifferentiableSimulation(p, AdjointOptions(checkpoints=F['adjoint_checkpoints'], backward_kernel=backward))(epsilon)
        gradient, = torch.autograd.grad(scalar_loss(result.signals), epsilon)
    torch.cuda.synchronize()
    # Documented contract (DIFFERENTIABLE_FDTD.md): real CUDA forward always runs the fused kernel;
    # complex fields select torch or fused by Region.cuda_kernel. The report must say which ran.
    forward = 'fused CUDA' if fields == 'real' or kernel == 'fused' else 'torch CUDA'
    transpose = ('fused CUDA complex transpose' if fields == 'complex' else 'fused CUDA transpose') if backward == 'fused' else 'torch explicit transpose'
    assert result.report['forward_backend'] == forward and result.report['backward_backend'] == transpose
    signals, expected = adjoint_references[fields]
    assert_normalized_close(result.signals.detach().cpu().numpy(), signals.numpy(), precision, 'signals')
    assert_normalized_close(gradient.cpu().numpy(), expected.numpy(), precision, 'epsilon gradient')
    assert float(expected.norm()) > 0


def test_adjoint_rejects_unsupported_combinations_by_name():
    p, epsilon = adjoint_inputs('real', 'float64', 'cpu')
    with pytest.raises(ValueError, match='fused backward requires a CUDA tensor'):
        DifferentiableSimulation(p, AdjointOptions(backward_kernel='fused'))(epsilon)
    with pytest.raises(ValueError, match='checkpoints require a CUDA tensor'):
        DifferentiableSimulation(p, AdjointOptions(checkpoint_transfers='async'))(epsilon)
    with pytest.raises(ValueError, match='backward_kernel must be'):
        AdjointOptions(backward_kernel='graph')
    gpu()
    from torchfdtd.boundaries import material_shape
    q = tiny('float64', 'cuda', plane=False)
    q.region.boundaries.z_max = BoundaryFace(kind='pmc')
    with pytest.raises(ValueError, match='does not implement PMC'):
        DifferentiableSimulation(q, AdjointOptions(backward_kernel='fused'))(torch.ones(material_shape(q.region), device='cuda', dtype=torch.float64))


def test_forward_rejects_unsupported_combinations_by_name():
    p = tiny('float64', 'cpu', kernel='fused')
    with pytest.raises(ValueError, match='requires backend="cuda" and a CUDA GPU'):
        Simulation(p).run()
    with pytest.raises(ValueError, match='cuda_graph_steps > 1 requires CUDA graph execution'):
        Simulation(tiny('float64', 'cpu')).run(cuda_graph_steps=4)
    gpu()
    with pytest.raises(ValueError, match='cuda_graph_steps > 1 requires CUDA graph execution'):
        Simulation(tiny('float64', 'cuda')).run(cuda_graph=False, cuda_graph_steps=4)
    with pytest.raises(ValueError, match='cuda_graph_steps must be an integer from 1 to 64'):
        Simulation(tiny('float64', 'cuda')).run(cuda_graph_steps=65)
