"""N-port aperture mode networks: contracts, straight guide, Y-branch, crossing, GDS markers."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Source, AdjointOptions, StreamedAdjointOptions
from torchfdtd.gds import GDSPort
from torchfdtd.mode_branches import (ModePort, ModeBranchNetwork, prepare_aperture_modal_launch,
                                     branch_network_from_ports)
from torchfdtd.mode_injection import _frozen_launch_material
from torchfdtd.mode_network import FixedModePort, ModeNetwork

CORE, CLAD = 12., 2.1


def device_name():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        pytest.importorskip('cupy')
    return device


def yee_epsilon(region, fn, device='cpu'):
    """Sample fn(x, y, z) at the three electric Yee positions."""
    r = region
    out = np.empty(r.shape+(3,), dtype=np.float32 if r.precision == 'float32' else np.float64)
    for c in range(3):
        axes = [float(r.mesh_nodes[a][0])+(np.arange(r.shape[a])+(.5 if a == c else 0.))*r.axis_steps[a]
                for a in range(3)]
        out[..., c] = fn(*np.meshgrid(*axes, indexing='ij'))
    return torch.tensor(out, device=device)


def straight_project(steps=600, ysize=2.4):
    faces = {a+'_'+s: dict(kind='periodic') for a in 'yz' for s in ('min', 'max')}
    r = Region(dimension='3d', size=(4.8, ysize, .5), mesh=.1, pml_cells=8, steps=steps,
               material_sampling='yee', boundaries=faces, precision='float32')
    return Project(region=r, sources=[Source(kind='plane', normal='x', center=(-1.4, 0., 0.),
                   size=(0., ysize, .5), pulse_cycles=2)], monitors=[])


def guide(u, v):
    return np.where(np.abs(u) < .2-1e-9, CORE, CLAD)


def aperture_ports(size=(0., 1.2, .5)):
    return (ModePort('in', 'x', 1, (-1., 0., 0.), size, .4),
            ModePort('out', 'x', -1, (1., 0., 0.), size, .4))


def branch_project(steps=800):
    faces = {a+'_'+s: dict(kind='pml', layers=8) for a in 'xy' for s in ('min', 'max')}
    faces.update({'z_'+s: dict(kind='periodic') for s in ('min', 'max')})
    r = Region(dimension='3d', size=(4.8, 4.8, .5), mesh=.1, pml_cells=8, steps=steps,
               material_sampling='yee', boundaries=faces, precision='float32')
    return Project(region=r, sources=[Source(kind='plane', normal='x', center=(-1.4, 0., 0.),
                   size=(0., 1.2, .5), pulse_cycles=2)], monitors=[])


def ybranch_ports():
    return (ModePort('in', 'x', 1, (-1., 0., 0.), (0., 1.2, .5), .4),
            ModePort('up', 'x', -1, (1., .7, 0.), (0., 1.2, .5), .4),
            ModePort('down', 'x', -1, (1., -.7, 0.), (0., 1.2, .5), .4))


def ybranch_sections():
    return {'in': lambda y, z: np.where(np.abs(y) < .2-1e-9, CORE, CLAD),
            'up': lambda y, z: np.where(np.abs(y-.7) < .2-1e-9, CORE, CLAD),
            'down': lambda y, z: np.where(np.abs(y+.7) < .2-1e-9, CORE, CLAD)}


def ybranch_epsilon(region, device='cpu'):
    def fn(x, y, z):
        centre = np.clip((x+.4)/1., 0., 1.)*.7
        return np.where((np.abs(y-centre) < .2-1e-9) | (np.abs(y+centre) < .2-1e-9), CORE, CLAD)
    return yee_epsilon(region, fn, device)


def crossing_ports():
    return (ModePort('west', 'x', 1, (-1., 0., 0.), (0., 1.2, .5), .4),
            ModePort('east', 'x', -1, (1., 0., 0.), (0., 1.2, .5), .4),
            ModePort('south', 'y', 1, (0., -1., 0.), (1.2, 0., .5), .4),
            ModePort('north', 'y', -1, (0., 1., 0.), (1.2, 0., .5), .4))


def crossing_sections():
    horizontal = lambda y, z: np.where(np.abs(y) < .2-1e-9, CORE, CLAD)
    vertical = lambda z, x: np.where(np.abs(x) < .2-1e-9, CORE, CLAD)
    return {'west': horizontal, 'east': horizontal, 'south': vertical, 'north': vertical}


def tiny_project(precision='float64', steps=300, normal='x'):
    """Small x or y propagation scene with a narrow aperture across its cell."""
    if normal == 'x':
        faces = {a+'_'+s: dict(kind='periodic') for a in 'yz' for s in ('min', 'max')}
        size, center, span = (2.4, 1.4, .5), (-.5, 0., 0.), (0., 1.2, .5)
    else:
        faces = {a+'_'+s: dict(kind='periodic') for a in 'xz' for s in ('min', 'max')}
        size, center, span = (1.4, 2.4, .5), (0., -.5, 0.), (1.2, 0., .5)
    r = Region(dimension='3d', size=size, mesh=.1, pml_cells=5, steps=steps,
               material_sampling='yee', boundaries=faces, precision=precision)
    return Project(region=r, sources=[Source(kind='plane', normal=normal, center=center, size=span,
                   pulse_cycles=2)], monitors=[])


def test_port_and_network_contracts():
    with pytest.raises(ValueError, match='direction'):
        ModePort('a', 'x', 2, (0., 0., 0.), (0., 1., 1.), .4)
    with pytest.raises(ValueError, match='zero along the normal'):
        ModePort('a', 'x', 1, (0., 0., 0.), (1., 1., 1.), .4)
    with pytest.raises(ValueError, match='mode indices'):
        ModePort('a', 'x', 1, (0., 0., 0.), (0., 1., 1.), .4, (0, 0))
    p = straight_project(steps=10)
    with pytest.raises(ValueError, match='at least two'):
        ModeBranchNetwork(p, aperture_ports()[:1], guide)
    with pytest.raises(ValueError, match='cell boundaries'):
        ModeBranchNetwork(p, aperture_ports((0., 1.25, .5)), guide)
    with pytest.raises(ValueError, match='at least two cells'):
        ModeBranchNetwork(p, (replace(aperture_ports()[0], source_offset_um=.1), aperture_ports()[1]), guide)
    with pytest.raises(ValueError, match='E nodes'):
        ModeBranchNetwork(p, (replace(aperture_ports()[0], center_um=(-1.03, 0., 0.)), aperture_ports()[1]), guide)
    with pytest.raises(ValueError, match='overlap'):
        ModeBranchNetwork(p, (aperture_ports()[0], replace(aperture_ports()[1], direction=1, center_um=(-.8, 0., 0.))), guide)
    with pytest.raises(ValueError, match='exactly the port names'):
        ModeBranchNetwork(p, aperture_ports(), port_permittivities={'in': guide})
    with pytest.raises(ValueError, match='inside num_modes'):
        ModeBranchNetwork(p, (aperture_ports()[0], replace(aperture_ports()[1], mode_indices=(1,))), guide)
    with pytest.raises(ValueError, match='AdjointOptions'):
        ModeBranchNetwork(p, aperture_ports(), guide, options=object())
    with pytest.raises(ValueError, match='network byte budget'):
        ModeBranchNetwork(p, aperture_ports(), guide, network_budget_bytes=1)
    # A narrower aperture leaks the mode through its periodic edges.
    with pytest.raises(ValueError, match='confinement'):
        ModeBranchNetwork(p, aperture_ports((0., .6, .5)), guide)
    network = ModeBranchNetwork(p, aperture_ports(), guide)
    assert network.channels == (('in', 0), ('out', 0))
    with pytest.raises(ValueError, match='Unknown'):
        network.reference_epsilon(port='missing')
    epsilon = network.reference_epsilon(port='in')
    # Exterior footprints cover each aperture from one cell inside the phase
    # plane outward; material elsewhere, including beside the aperture, is free.
    assert network._footprint(0) == (slice(None, 16), slice(6, 18), slice(0, 5))
    assert network._footprint(1) == (slice(33, None), slice(6, 18), slice(0, 5))
    inside = epsilon.clone()
    inside[2, 12, 2, 1] += .1
    with pytest.raises(ValueError, match='Exterior material of port in'):
        network(inside)
    with pytest.raises(ValueError, match='output byte budget'):
        network(epsilon, output_budget_bytes=1)
    network.ports = network.ports[::-1]
    with pytest.raises(ValueError, match='configuration changed'):
        network(epsilon)


def test_aperture_launch_terms_confinement_and_frozen_neighborhood():
    p = tiny_project('float32', steps=12)
    launch = prepare_aperture_modal_launch(p, guide)
    assert launch.aperture == ((1, 13), (0, 5))
    assert launch.edge_energy_fraction < 1e-3
    assert len(launch.terms) == 8
    for name, loc, waveform, profile in launch.terms:
        assert loc[1:] == (slice(1, 13), slice(0, 5))
        assert profile.shape == (1, 12, 5) and waveform.shape == (12,)
    assert {t[1][0].start for t in launch.terms if t[0][0] == 'E'} == {launch.electric_index}
    with pytest.raises(ValueError, match='confinement'):
        prepare_aperture_modal_launch(p, guide, confinement_tolerance=1e-6)
    with pytest.raises(ValueError, match='byte budget'):
        prepare_aperture_modal_launch(p, guide, source_budget_bytes=1)
    epsilon = yee_epsilon(p.region, lambda x, y, z: guide(y, z))
    epsilon.requires_grad_()
    frozen, selection, _ = _frozen_launch_material(launch, epsilon)
    assert selection == (slice(launch.electric_index-1, launch.electric_index+2), slice(1, 13), slice(0, 5))
    cotangent, = torch.autograd.grad(frozen.sum(), epsilon)
    assert torch.count_nonzero(cotangent[selection]) == 0
    assert torch.count_nonzero(cotangent) == cotangent.numel()-cotangent[selection].numel()
    changed = epsilon.detach().clone()
    changed[launch.electric_index, 0, 0, 0] += .1  # outside the aperture, at the source plane
    _frozen_launch_material(launch, changed)
    changed[launch.electric_index, 6, 2, 0] += .1
    with pytest.raises(ValueError, match='Injection-neighborhood'):
        _frozen_launch_material(launch, changed)


def test_aperture_ports_match_full_cell_network_on_straight_guide():
    device = device_name()
    p = straight_project()
    options = AdjointOptions(checkpoints=4)
    narrow = ModeBranchNetwork(p, aperture_ports(), guide, options)
    mixed = ModeBranchNetwork(p, (replace(aperture_ports()[0], size_um=(0., 2.4, .5)), aperture_ports()[1]), guide, options)
    legacy = ModeNetwork(p, (FixedModePort('left', -1., -1.4, 1), FixedModePort('right', 1., 1.4, -1)),
                         guide, options, num_modes=1)
    with torch.no_grad():
        s_narrow = narrow(narrow.reference_epsilon(port='in', device=device)).s
        s_mixed = mixed(mixed.reference_epsilon(port='out', device=device)).s
        s_legacy = legacy(legacy.reference_epsilon(device=device)).s
    expected = torch.tensor(np.exp(2j*narrow._launches[0].mode.beta_per_um), dtype=s_narrow.dtype)
    assert (s_narrow-s_legacy).abs().max() < 1e-4
    assert (s_mixed-s_legacy).abs().max() < 1e-4
    torch.testing.assert_close(s_narrow[1, 0], expected, atol=1e-3, rtol=1e-3)
    assert s_narrow.diagonal().abs().max() < 1e-4
    power = s_narrow.abs().square().sum(0)
    assert (power-1).abs().max() < 1e-3
    assert narrow.apertures[0] == ((6, 18, False), (0, 5, True))
    assert mixed.apertures[0][0] == (0, 24, True)
    print({'device': device, 'narrow_vs_full': float((s_narrow-s_legacy).abs().max()),
           'mixed_vs_full': float((s_mixed-s_legacy).abs().max()),
           'straight_phase_error': float((s_narrow[1, 0]-expected).abs()), 'column_power': power.tolist()})


def test_y_branch_power_balance_reciprocity_symmetry_and_gradient():
    device = device_name()
    p = branch_project()
    network = ModeBranchNetwork(p, ybranch_ports(), options=AdjointOptions(checkpoints=4),
                                port_permittivities=ybranch_sections())
    assert network.channels == (('in', 0), ('up', 0), ('down', 0))
    epsilon = ybranch_epsilon(p.region, device)
    with torch.no_grad():
        result = network(epsilon)
    s = result.s
    power = torch.tensor(result.report['column_power'])
    # Passivity within the recorded numerical margin. Selected channels miss
    # radiation, so the crude staircase junction loses input power and a
    # launch from one arm loses about half into the antisymmetric field.
    assert power.max() < 1.002
    assert power[0] > .75 and power[1] > .5 and power[2] > .5
    assert (s-s.T).abs().max() < 5e-3
    assert abs(float(s[1, 0].abs()-s[2, 0].abs())) < 1e-3
    assert result.report['ports'][1]['normal'] == 'x' and result.report['ports'][1]['direction'] == -1
    mask = torch.zeros_like(epsilon)
    mask[22:26, 22:26] = 1
    parameter = torch.tensor(.5, device=device, requires_grad=True)
    perturbed = network(epsilon+parameter*mask)
    score = perturbed.s[1, 0].real+.3*perturbed.s[0, 2].imag
    derivative, = torch.autograd.grad(score, parameter)
    step = .01
    with torch.no_grad():
        high = network(epsilon+(parameter+step)*mask).s
        low = network(epsilon+(parameter-step)*mask).s
        finite = ((high[1, 0].real+.3*high[0, 2].imag)-(low[1, 0].real+.3*low[0, 2].imag))/(2*step)
    torch.testing.assert_close(derivative.cpu(), finite, rtol=1e-3, atol=1e-5)
    print({'device': device, 'S': [[complex(x) for x in row] for row in s.tolist()],
           'column_power': power.tolist(), 'reciprocity': float((s-s.T).abs().max()),
           'arm_symmetry': float(s[1, 0].abs()-s[2, 0].abs()),
           'gradient': float(derivative), 'finite_difference': float(finite),
           'gradient_relative_error': float(abs((derivative.cpu()-finite)/finite))})


def test_four_port_crossing_on_two_normals():
    device = device_name()
    p = branch_project()
    network = ModeBranchNetwork(p, crossing_ports(), options=AdjointOptions(checkpoints=4),
                                port_permittivities=crossing_sections())
    epsilon = yee_epsilon(p.region, lambda x, y, z: np.where((np.abs(y) < .2-1e-9) | (np.abs(x) < .2-1e-9), CORE, CLAD), device)
    with torch.no_grad():
        result = network(epsilon)
    s = result.s
    power = torch.tensor(result.report['column_power'])
    assert s.shape == (4, 4)
    assert power.max() < 1.002 and power.min() > .7
    assert (s-s.T).abs().max() < 1e-3
    # Four-fold rotation maps west->east onto south->north exactly on the
    # transposed Yee lattice; the y mirror does not, so it keeps a staircase gap.
    assert abs(float(s[1, 0].abs()-s[3, 2].abs())) < 1e-4
    assert abs(float(s[2, 0].abs()-s[3, 0].abs())) < 5e-3
    assert s[1, 0].abs().square() > .6 and s[2, 0].abs().square() > .04
    print({'device': device, 'column_power': power.tolist(), 'reciprocity': float((s-s.T).abs().max()),
           'rotation_gap': float(s[1, 0].abs()-s[3, 2].abs()), 'mirror_gap': float(s[2, 0].abs()-s[3, 0].abs())})


def test_branch_network_from_gds_markers_matches_explicit_ports():
    device = device_name()
    p = branch_project(steps=400)
    markers = (GDSPort('in', (-1., 0., 0.), (-1., 0., 0.), .4, .6, 10, 0),
               GDSPort('up', (1., .7, 0.), (1., 0., 0.), .4, .6, 10, 0),
               GDSPort('down', (1., -.7, 0.), (1., 0., 0.), .4, .6, 10, 0))
    built = branch_network_from_ports(markers, p, normal_convention='outward', wavelength_um=1.55,
        core_epsilon=CORE, cladding_epsilon=CLAD, source_offset_um=.4, aperture_um=(1.2, .5),
        options=AdjointOptions(checkpoints=2))
    manual = ModeBranchNetwork(p, ybranch_ports(), options=AdjointOptions(checkpoints=2),
                               port_permittivities=ybranch_sections())
    assert [(q.name, q.normal, q.direction, q.center_um, q.size_um) for q in built.ports] == \
        [(q.name, q.normal, q.direction, q.center_um, q.size_um) for q in manual.ports]
    assert [l.identity for l in built._launches] == [l.identity for l in manual._launches]
    epsilon = ybranch_epsilon(p.region, device)
    with torch.no_grad():
        s_built = built(epsilon).s
        s_manual = manual(epsilon).s
    assert (s_built-s_manual).abs().max() < 1e-6
    with pytest.raises(ValueError, match='overlap'):
        branch_network_from_ports(markers, p, normal_convention='inward', wavelength_um=1.55,
            core_epsilon=CORE, cladding_epsilon=CLAD, source_offset_um=.4, aperture_um=(1.2, .5))
    with pytest.raises(ValueError, match='z-normal'):
        branch_network_from_ports((markers[0], GDSPort('top', (0., 0., 0.), (0., 0., 1.), .4, .6, 10, 0)), p,
            normal_convention='outward', wavelength_um=1.55, core_epsilon=CORE, cladding_epsilon=CLAD,
            source_offset_um=.4, aperture_um=(1.2, .5))
    with pytest.raises(ValueError, match='every port name'):
        branch_network_from_ports(markers, p, normal_convention='outward', wavelength_um=1.55,
            core_epsilon=CORE, cladding_epsilon=CLAD, source_offset_um={'in': .4})
    with pytest.raises(ValueError, match='every port name'):
        branch_network_from_ports(markers, p, normal_convention='outward', wavelength_um=1.55,
            core_epsilon=CORE, cladding_epsilon=CLAD, source_offset_um=.4, mode_indices={'in': (0,)})
    with pytest.raises(ValueError, match='both be finite'):
        branch_network_from_ports(markers, p, normal_convention='outward', wavelength_um=1.55,
            core_epsilon=CORE, source_offset_um=.4)
    with pytest.raises(ValueError, match='not both'):
        branch_network_from_ports(markers, p, normal_convention='outward', wavelength_um=1.55,
            core_epsilon=CORE, cladding_epsilon=CLAD, source_offset_um=.4, permittivity=guide)
    with pytest.raises(ValueError, match='Provide core_epsilon'):
        branch_network_from_ports(markers, p, normal_convention='outward', wavelength_um=1.55, source_offset_um=.4)
    # Explicit sections in each port's own transverse coordinates, as the
    # two-port GDS helper passes them, replace the core/cladding rectangles.
    sectioned = branch_network_from_ports(markers, p, normal_convention='outward', wavelength_um=1.55,
        source_offset_um=.4, aperture_um=(1.2, .5), mode_indices=(0,), options=AdjointOptions(checkpoints=2),
        port_permittivities=ybranch_sections())
    assert [l.identity for l in sectioned._launches] == [l.identity for l in manual._launches]


def test_z_normal_ports_match_cyclic_relabelling_of_x_normal_ports():
    """Ports on the z faces: the scene relabelled (x, y, z) -> (z, x, y) keeps Yee handedness."""
    device = device_name()
    options = AdjointOptions(checkpoints=2)
    along_x = ModeBranchNetwork(tiny_project('float32', steps=400), (
        ModePort('in', 'x', 1, (-.3, 0., 0.), (0., 1.2, .5), .2),
        ModePort('out', 'x', -1, (.3, 0., 0.), (0., 1.2, .5), .2)), guide, options)
    faces = {a+'_'+s: dict(kind='periodic') for a in 'xy' for s in ('min', 'max')}
    r = Region(dimension='3d', size=(1.4, .5, 2.4), mesh=.1, pml_cells=5, steps=400,
               material_sampling='yee', boundaries=faces, precision='float32')
    p = Project(region=r, sources=[Source(kind='plane', normal='z', center=(0., 0., -.5), size=(1.2, .5, 0.),
                pulse_cycles=2)], monitors=[])
    along_z = ModeBranchNetwork(p, (ModePort('in', 'z', 1, (0., 0., -.3), (1.2, .5, 0.), .2),
                                    ModePort('out', 'z', -1, (0., 0., .3), (1.2, .5, 0.), .2)), guide, options)
    assert along_z.apertures == (((1, 13, False), (0, 5, True)),)*2
    assert along_z._footprint(0) == (slice(1, 13), slice(0, 5), slice(None, 11))
    relabel = lambda e: e.permute(1, 2, 0, 3)[..., [1, 2, 0]].contiguous()
    base = along_x.reference_epsilon(port='in', device=device)
    assert torch.equal(along_z.reference_epsilon(port='in', device=device), relabel(base))
    mask = torch.zeros_like(base)
    mask[11:14, 4:10] = 1
    parameter = torch.tensor(.4, device=device, requires_grad=True)
    sx = along_x(base+parameter*mask).s
    sz = along_z(relabel(base+parameter*mask)).s
    torch.testing.assert_close(sz, sx, rtol=1e-4, atol=1e-5)
    score = lambda s: s[1, 0].abs().square()+.3*s[0, 1].imag
    gx, = torch.autograd.grad(score(sx), parameter)
    gz, = torch.autograd.grad(score(sz), parameter)
    assert gx.abs() > 0
    torch.testing.assert_close(gz, gx, rtol=1e-3, atol=1e-6)
    print({'device': device, 's_difference': float((sz-sx).detach().abs().max()),
           'gradient': [float(gx), float(gz)]})


def test_branch_network_power_objective_taylor_and_central_difference_cpu():
    """|S21|^2 plus a phase term through the N-port wrapper, FP64 resident adjoint."""
    p = tiny_project('float64', steps=400)
    network = ModeBranchNetwork(p, (ModePort('in', 'x', 1, (-.3, 0., 0.), (0., 1.2, .5), .2),
                                    ModePort('out', 'x', -1, (.3, 0., 0.), (0., 1.2, .5), .2)),
                                guide, AdjointOptions(checkpoints=2))
    base = network.reference_epsilon(port='in')
    mask = torch.zeros_like(base)
    mask[11:14, 4:10] = 1
    direction = torch.randn(base.shape, generator=torch.Generator().manual_seed(5), dtype=base.dtype)*mask
    score = lambda s: s[1, 0].abs().square()+.3*s[0, 1].real
    x = base.clone().requires_grad_()
    value = score(network(x).s)
    gradient, = torch.autograd.grad(value, x)
    value = float(value.detach())
    for index in range(2):
        assert torch.count_nonzero(gradient[network._footprint(index)]) == 0
    slope = float((gradient*direction).sum())
    residuals, central = [], []
    with torch.no_grad():
        for h in (2e-3, 1e-3, 5e-4):
            plus = float(score(network(base+h*direction).s))
            minus = float(score(network(base-h*direction).s))
            residuals.append(abs(plus-value-h*slope))
            central.append((plus-minus)/(2*h))
    assert residuals[1] < .3*residuals[0] and residuals[2] < .3*residuals[1]
    assert abs(central[-1]-slope) < 1e-6*abs(slope)
    print({'value': value, 'directional_derivative': slope, 'central_difference': central[-1],
           'taylor_residuals': residuals})


def test_streamed_branch_network_matches_resident():
    # A discrete consistency check between executions; the short run truncates the pulse.
    p = tiny_project('float32', steps=400)
    ports = (ModePort('in', 'x', 1, (-.3, 0., 0.), (0., 1.2, .5), .2),
             ModePort('out', 'x', -1, (.3, 0., 0.), (0., 1.2, .5), .2))
    resident = ModeBranchNetwork(p, ports, guide, AdjointOptions(checkpoints=2))
    streamed = ModeBranchNetwork(p, ports, guide, StreamedAdjointOptions(device='cpu', slab_width=4,
                                                                          temporal_depth=3, checkpoints=2))
    epsilon = resident.reference_epsilon(port='in')
    mask = torch.zeros_like(epsilon)
    mask[11:14, 4:10] = 1
    parameter = torch.tensor(.4, requires_grad=True)
    dense = resident(epsilon+parameter*mask)
    tiled = streamed(epsilon+parameter*mask)
    assert tiled.report['execution'] == 'streamed' and dense.report['execution'] == 'resident'
    torch.testing.assert_close(tiled.s, dense.s, rtol=1e-5, atol=1e-6)
    score = lambda s: s[1, 0].real+.3*s[0, 1].imag+s[1, 0].abs().square()
    expected, = torch.autograd.grad(score(dense.s), parameter)
    actual, = torch.autograd.grad(score(tiled.s), parameter)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-7)
    assert expected.abs() > 0
    print({'s_difference': float((tiled.s-dense.s).detach().abs().max()), 'gradient': [float(expected), float(actual)]})
