"""Fixed port basis normalization and actual reciprocal native FDTD network."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Source, Boundaries, BoundaryFace, AdjointOptions
from torchfdtd.adjoint_planes import DifferentiablePlaneResult
from torchfdtd.mode_network import FixedModePort, ModeNetwork, _decompose


def setup(modes=(0,), steps=600):
    boundaries = Boundaries(**{a+'_'+side: BoundaryFace(kind='periodic')
                              for a in 'yz' for side in ('min', 'max')})
    r = Region(dimension='3d', size=(8., 1., 1.), mesh=.2, pml_cells=5,
               steps=steps, material_sampling='yee', boundaries=boundaries, precision='float32')
    p = Project(region=r, sources=[Source(kind='plane', normal='x', center=(-2., 0., 0.),
                size=(0., 1., 1.), pulse_cycles=2)], monitors=[])
    ports = (FixedModePort('left', -1., -2., 1, modes), FixedModePort('right', 1., 2., -1, modes))
    return p, ports


def test_two_mode_directional_power_normalization_and_gram_rejection():
    p, ports = setup((0, 1), steps=10)
    network = ModeNetwork(p, ports, 2.25)
    launch = network._launches[0]
    y, z = np.meshgrid(-.5+(np.arange(5)+.5)*.2, -.5+(np.arange(5)+.5)*.2, indexing='ij')
    points = np.stack((np.full(25, -1.), y.ravel(), z.ravel()), -1)
    basis = []
    for item in network._launches[:2]:
        values = torch.tensor(item.detector_mode(-1.).sample_plane(points))
        power = .5*((values[:, 1]*values[:, 5].conj()-values[:, 2]*values[:, 4].conj()).real).sum()*4e-14
        basis.append(values/power.sqrt())
    amplitudes = torch.tensor([.7+.2j, -.1+.3j], dtype=torch.complex64, requires_grad=True)
    backward = basis[1].clone(); backward[:, [0, 4, 5]] *= -1
    fields = amplitudes[0]*basis[0]+amplitudes[1]*basis[1]+(.13-.11j)*backward
    plane = DifferentiablePlaneResult(fields[None], torch.tensor([299792458/1.55e-6]),
        torch.tensor(points, dtype=torch.float32), torch.full((25,), 4e-14), (5, 5), 'x', 'synthetic', {})
    forward, back = _decompose(plane, network._launches[:2], 1e-4)
    torch.testing.assert_close(forward[0], amplitudes, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(back[0], torch.tensor([0., .13-.11j]), atol=2e-6, rtol=2e-5)
    gradient, = torch.autograd.grad(forward.abs().square().sum(), amplitudes)
    torch.testing.assert_close(gradient, 2*amplitudes, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match='orthogonal'):
        _decompose(plane, (launch, launch), 1e-4)


def test_network_contracts_and_early_packet_budget():
    p, ports = setup(steps=10)
    with pytest.raises(ValueError, match='network byte budget'):
        ModeNetwork(p, ports, 2.25, network_budget_bytes=1)
    with pytest.raises(ValueError, match='ordered'):
        ModeNetwork(p, (replace(ports[0], source_coordinate_um=0.), ports[1]), 2.25)
    with pytest.raises(ValueError, match='opposing'):
        ModeNetwork(p, ports[:1], 2.25)
    network = ModeNetwork(p, ports, 2.25)
    epsilon = network.reference_epsilon()
    epsilon[0] += .1
    with pytest.raises(ValueError, match='exterior material'):
        network(epsilon)


def test_reciprocal_two_port_native_propagation_and_material_gradient():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        pytest.importorskip('cupy')
    p, ports = setup()
    network = ModeNetwork(p, ports, 2.25, AdjointOptions(checkpoints=4))
    epsilon = network.reference_epsilon(device=device)
    with torch.no_grad():
        baseline = network(epsilon)
    s = baseline.s
    expected = torch.tensor(np.exp(2j*network._launches[0].mode.beta_per_um), dtype=s.dtype)
    torch.testing.assert_close(s[1, 0], expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(s[0, 1], expected, atol=2e-3, rtol=2e-3)
    assert s.diagonal().abs().max() < 2e-5
    assert (s-s.T).abs().max() < 2e-3
    mask = torch.zeros_like(epsilon)
    mask[19:21] = 1
    parameter = torch.tensor(.35, device=device, requires_grad=True)
    result = network(epsilon+parameter*mask)
    # A complex S objective probes phase as well as magnitude.
    score = result.s[1, 0].real + .3*result.s[0, 1].imag
    derivative, = torch.autograd.grad(score, parameter)
    step = .004
    with torch.no_grad():
        high = network(epsilon+(parameter+step)*mask).s
        low = network(epsilon+(parameter-step)*mask).s
        finite = ((high[1, 0].real+.3*high[0, 1].imag)
                  -(low[1, 0].real+.3*low[0, 1].imag))/(2*step)
    torch.testing.assert_close(derivative.cpu(), finite, rtol=.01, atol=3e-4)
    assert (result.s-result.s.T).abs().max() < 3e-3
    assert (result.s.abs().square().sum(0)-1).abs().max() < .015
    print({'device': device, 'straight_phase_error': float((s[1, 0]-expected).abs()),
           'reciprocity_error': float((result.s-result.s.T).abs().max().detach()),
           'gradient': float(derivative), 'finite_difference': float(finite),
           'gradient_relative_error': float(abs((derivative.cpu()-finite)/finite)),
           's_perturbed': [[complex(x) for x in row] for row in result.s.detach().tolist()]})


def test_fixed_port_snapshot_and_one_case_replay_after_network_mutation(monkeypatch):
    import torchfdtd.mode_network as module
    indices = [0]
    port = FixedModePort('immutable', -1., -2., 1, indices)
    indices.append(1)
    assert port.mode_indices == (0,)
    p, ports = setup(steps=10)
    network = ModeNetwork(p, ports, 2.25)
    class Fake:
        def __init__(self, project, launch, options):
            assert project.region.steps == 10
        def __call__(self, value, frequency):
            a = value.mean().to(torch.complex64).reshape(1, 1, 1)
            fields = torch.cat((a, 2*a), -1)
            return {name: DifferentiablePlaneResult(fields, torch.tensor(frequency),
                torch.zeros(1, 3), torch.ones(1), (1, 1), 'x', 'stub', {})
                for name in ('left', 'right')}
    monkeypatch.setattr(module, 'ModeInjectedPlaneSimulation', Fake)
    monkeypatch.setattr(module, '_decompose', lambda plane, launches, tolerance:
                        (plane.fields[..., 0], plane.fields[..., 1]))
    base = network.reference_epsilon()
    mask = torch.zeros_like(base); mask[17:24] = 1
    parameter = torch.tensor(.1, requires_grad=True)
    result = network(base+parameter*mask)
    network.project.region.steps = 20
    network.ports = ()
    derivative, = torch.autograd.grad(result.s.real.sum(), parameter)
    torch.testing.assert_close(derivative, torch.tensor(.35), atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match='project changed'):
        network(base)


def test_native_four_channel_straight_guide_mode_mapping():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        pytest.importorskip('cupy')
    p, ports = setup((0, 1))
    network = ModeNetwork(p, ports, 2.25, AdjointOptions(checkpoints=2))
    with torch.no_grad():
        result = network(network.reference_epsilon(device=device))
    expected = torch.zeros_like(result.s)
    for mode in range(2):
        phase = np.exp(2j*network._launches[mode].mode.beta_per_um)
        expected[2+mode, mode] = phase
        expected[mode, 2+mode] = phase
    torch.testing.assert_close(result.s, expected, atol=2e-3, rtol=2e-3)
    print({'four_channel_error': float((result.s-expected).abs().max()),
           'channels': result.channels})


@pytest.mark.parametrize('change', ['port', 'axis'])
def test_reassigned_port_phase_plane_rejected_before_calibration(monkeypatch, change):
    import torchfdtd.mode_network as module
    p, ports = setup(steps=10)
    network = ModeNetwork(p, ports, 2.25)
    epsilon = network.reference_epsilon()
    if change == 'port':
        network.ports = (replace(ports[0], coordinate_um=-.8), ports[1])
    else:
        network.axis = 1
    def forbidden(*args, **kwargs):
        raise AssertionError('Calibration started after port metadata changed.')
    monkeypatch.setattr(module, 'ModeInjectedPlaneSimulation', forbidden)
    with pytest.raises(ValueError, match='configuration changed'):
        network(epsilon)
