"""G6-04: per-port mode tracking, normalization, reference planes, direction separation and
degenerate/weak-mode diagnostics on the analytic slab and a degenerate square guide, CPU only."""
import math
import warnings

import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Source, Boundaries, BoundaryFace, AdjointOptions, ModeNetwork, FixedModePort
from torchfdtd.adjoint_planes import DifferentiablePlaneResult
from torchfdtd.mode_ports import solve_waveguide_modes, C0
from torchfdtd.ports import (FixedPortSectionError, ModeTrackingWarning, WeakModeWarning, TrackedPortModes,
                             track_port_modes, port_diagnostics, degenerate_clusters, overlap_matrix,
                             confinement_factor, shift_reference_plane, deembed_s_matrix, separate_directions,
                             port_normalization, fixed_port_section)
from tests.test_mode_ports import slab_neff


def slab(u, v):
    return np.where(abs(u) < .25-1e-9, 4., np.where(abs(abs(u)-.25) < 1e-9, 3.125, 2.25))


SLAB = dict(shape=(80, 4), spacing_um=(.05, .05), normal='z')


def test_slab_tracking_follows_te0_and_tm0_across_the_band_with_reported_overlaps():
    wavelengths = (1.45, 1.5, 1.55, 1.6, 1.65)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        tracked = track_port_modes(slab, wavelengths, num_modes=2, candidates=4, minimum_overlap=.9, **SLAB)
    assert isinstance(tracked, TrackedPortModes)
    assert tracked.overlaps.shape == (4, 2) and tracked.minimum_overlap == tracked.overlaps.min()
    assert tracked.minimum_overlap > .99
    for row, wavelength in zip(tracked.neff, wavelengths):
        exact = [slab_neff('TE', wavelength=wavelength), slab_neff('TM', wavelength=wavelength)]
        np.testing.assert_allclose(row, exact, atol=2e-3)
    # Both tracks vary smoothly: one-sided differences of neff never change sign.
    steps = np.diff(tracked.neff, axis=0)
    assert (steps < 0).all()
    report = tracked.report()
    assert report['declared_minimum'] == .9 and report['normalization'] == port_normalization()
    assert tracked.solver_index.shape == (5, 2)


def test_tracking_warns_below_the_declared_minimum_and_rejects_bad_inputs():
    with pytest.warns(ModeTrackingWarning, match='below the declared minimum'):
        tracked = track_port_modes(slab, (1.55, 1.6), num_modes=2, candidates=3, minimum_overlap=.99999, **SLAB)
    assert tracked.minimum_overlap < .99999
    with pytest.raises(ValueError, match='candidates'):
        track_port_modes(slab, (1.55,), num_modes=2, candidates=1, **SLAB)
    with pytest.raises(ValueError, match='minimum_overlap'):
        track_port_modes(slab, (1.55,), num_modes=1, minimum_overlap=0., **SLAB)
    with pytest.raises(ValueError, match='wavelengths_um'):
        track_port_modes(slab, (), num_modes=1, **SLAB)


def test_design_variable_port_section_is_rejected_by_a_named_error():
    logits = torch.zeros((6, 6), requires_grad=True)

    def section(u, v):
        return 2.25+logits.sigmoid()
    with pytest.raises(FixedPortSectionError, match='fixed'):
        track_port_modes(section, (1.55,), shape=(6, 6), spacing_um=(.2, .2), num_modes=1)
    with pytest.raises(FixedPortSectionError):
        fixed_port_section(logits)
    assert issubclass(FixedPortSectionError, ValueError)
    # The same error reaches the network layer through its fixed-section check.
    boundaries = Boundaries(**{a+'_'+side: BoundaryFace(kind='periodic') for a in 'yz' for side in ('min', 'max')})
    region = Region(dimension='3d', size=(8., 1., 1.), mesh=.2, pml_cells=5, steps=10, material_sampling='yee',
                    boundaries=boundaries)
    project = Project(region=region, sources=[Source(kind='plane', normal='x', center=(-2., 0., 0.),
                      size=(0., 1., 1.), pulse_cycles=2)], monitors=[])
    ports = (FixedModePort('left', -1., -2., 1), FixedModePort('right', 1., 2., -1))
    trainable = torch.full((5, 5), 2.25, requires_grad=True)
    with pytest.raises(FixedPortSectionError):
        ModeNetwork(project, ports, lambda u, v: trainable, num_modes=1)


def test_degenerate_square_guide_reports_the_cluster_canonical_basis_and_overlap_matrix():
    def square(u, v):
        return np.where((abs(u) < .3) & (abs(v) < .3), 4., 2.25)
    modes = solve_waveguide_modes(square, shape=(16, 16), spacing_um=(.1, .1), wavelength_um=1.55, num_modes=2)
    diagnostics = port_diagnostics(modes, core=lambda u, v: (abs(u) < .3) & (abs(v) < .3), confinement_threshold=.5)
    assert diagnostics.clusters == ((0, 1),)
    assert math.isclose(modes[0].beta_per_um, modes[1].beta_per_um, rel_tol=1e-6)
    np.testing.assert_allclose(diagnostics.overlap, np.eye(2), atol=2e-4)
    # Canonical basis: track 0 is u polarized, track 1 v polarized.
    power = lambda mode, c: float((np.abs(mode.fields[..., c])**2).sum())
    assert power(modes[0], 1) > 100*power(modes[0], 2) and power(modes[1], 2) > 100*power(modes[1], 1)
    assert diagnostics.weak_modes == () and (diagnostics.confinement > .5).all()
    report = diagnostics.report()
    assert report['degenerate_clusters'] == [[0, 1]] and report['overlap_max_off_diagonal'] < 2e-4
    # A nondegenerate slab pair forms two singleton clusters.
    slab_modes = solve_waveguide_modes(slab, wavelength_um=1.55, num_modes=2, **SLAB)
    assert degenerate_clusters(slab_modes) == ((0,), (1,))
    assert overlap_matrix(slab_modes).shape == (2, 2)


def test_weak_mode_warning_uses_a_measured_confinement_factor():
    # A narrow low-contrast core at long wavelength: the fundamental mode
    # spreads far beyond the declared core of the aperture.
    def thin(u, v):
        return np.where(abs(u) < .05, 2.3, 2.25)
    modes = solve_waveguide_modes(thin, shape=(40, 4), spacing_um=(.1, .1), wavelength_um=1.55, normal='z', num_modes=1)
    core = lambda u, v: abs(u) < .05
    factor = confinement_factor(modes[0], core)
    assert 0 < factor < .5
    with pytest.warns(WeakModeWarning, match='below the declared threshold'):
        diagnostics = port_diagnostics(modes, core=core, confinement_threshold=.5)
    assert diagnostics.weak_modes == (0,)
    assert diagnostics.confinement[0] == pytest.approx(factor)
    # Strong confinement of the slab under the same threshold: no warning.
    strong = solve_waveguide_modes(slab, wavelength_um=1.55, num_modes=1, **SLAB)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        clean = port_diagnostics(strong, core=lambda u, v: abs(u) < .25, confinement_threshold=.5)
    assert clean.weak_modes == () and clean.confinement[0] > .7
    with pytest.raises(ValueError, match='core mask'):
        confinement_factor(strong[0], np.zeros((3, 3), dtype=bool))
    assert np.isnan(port_diagnostics(strong).confinement).all()


def test_reference_plane_shift_and_de_embedding_cancel_a_straight_guide_phase():
    mode = solve_waveguide_modes(2.25, shape=(5, 5), spacing_um=(.2, .2), wavelength_um=1.55, num_modes=1)[0]
    beta = mode.beta_per_um
    length = 1.
    forward = np.exp(1j*beta*length)
    assert shift_reference_plane(1., mode, length) == pytest.approx(forward)
    assert shift_reference_plane(forward, mode, -length) == pytest.approx(1.)
    assert shift_reference_plane(1., mode, length, direction='backward') == pytest.approx(np.conj(forward))
    amplitude = torch.tensor([.3+.4j], dtype=torch.complex64, requires_grad=True)
    shifted = shift_reference_plane(amplitude, mode, length)
    gradient, = torch.autograd.grad(shifted.abs().square().sum(), amplitude)
    torch.testing.assert_close(gradient, 2*amplitude.detach(), rtol=1e-5, atol=1e-6)
    # Straight guide of length 2L between phase planes at -L and +L: S21 = exp(2 i beta L).
    s = torch.tensor([[0., np.exp(2j*beta*length)], [np.exp(2j*beta*length), 0.]], dtype=torch.complex64)
    identity_like = deembed_s_matrix(s, (mode, mode), (length, length))
    torch.testing.assert_close(identity_like, torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex64), rtol=1e-5, atol=1e-6)
    numpy_version = deembed_s_matrix(s.numpy(), (mode, mode), (length, length))
    np.testing.assert_allclose(numpy_version, identity_like.numpy(), atol=1e-6)
    with pytest.raises(ValueError, match='one fixed mode'):
        deembed_s_matrix(s, (mode,), (length, length))


def test_direction_separation_recovers_synthetic_forward_and_backward_amplitudes():
    mode = solve_waveguide_modes(2.25, shape=(8, 6), spacing_um=(.1, .1), wavelength_um=1.55, num_modes=1)[0]
    u, v = np.meshgrid(-.4+(np.arange(8)+.5)*.1, -.3+(np.arange(6)+.5)*.1, indexing='ij')
    points = np.stack([np.zeros(u.size), u.ravel(), v.ravel()], axis=1)
    basis = torch.tensor(mode.sample_plane(points))
    reverse = torch.tensor(mode.backward().sample_plane(points))
    a = torch.tensor(1.7-.2j, dtype=torch.complex64, requires_grad=True)
    b = torch.tensor(.3+.5j, dtype=torch.complex64, requires_grad=True)
    scale = 1e-22
    plane = DifferentiablePlaneResult(fields=scale*(a*basis[None]+b*reverse[None]), frequency_hz=torch.tensor([C0/1.55e-6]),
        points_um=torch.tensor(points, dtype=torch.float32), weights=torch.full((len(points),), 1e-14),
        shape=(1, 8, 6), normal='x', run_signature='fixed', report={})
    forward, backward = separate_directions(plane, mode)
    torch.testing.assert_close(forward/scale, a.detach()[None], rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(backward/scale, b.detach()[None], rtol=2e-5, atol=1e-5)
    # |a|^2 is the reduced modal power in the plane units, here scale^2 |a|^2.
    # FP32 squares of 1e-22 amplitudes underflow, so the caller divides by a
    # reference scale before squaring, as normalized_mode_power does.
    assert float(forward.detach().abs()) == pytest.approx(scale*abs(complex(a.detach())), rel=1e-4)
    gradients = torch.autograd.grad(((forward/scale).abs().square()+(backward/scale).abs().square()).sum(), (a, b))
    torch.testing.assert_close(gradients[0], 2*a.detach(), rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(gradients[1], 2*b.detach(), rtol=2e-5, atol=1e-5)
    with pytest.raises(ValueError, match='single matching frequency'):
        separate_directions(DifferentiablePlaneResult(fields=plane.fields, frequency_hz=torch.tensor([C0/1.5e-6]),
            points_um=plane.points_um, weights=plane.weights, shape=plane.shape, normal='x', run_signature='fixed', report={}), mode)
