"""Unit tests for the 4 core numerical upgrades:
1. Conformal Variant 0 anisotropic subpixel permittivity in mode solvers, launches, and 3D waveguides.
2. Deterministic half-plane phase canonicalization for antisymmetric higher-order modes (TE1).
3. Broadband multi-frequency support in modal_plane_amplitudes and modal_s_parameters.
4. 3D space-time Yee dispersion predictor yee_3d_modal_beta and numerical 3D FDTD phase verification.
"""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch

from torchfdtd import (
    Project,
    Region,
    MeshRefinement,
    Source,
    FieldMonitor,
    AdjointOptions,
    solve_waveguide_modes,
    mode_power_overlap,
    prepare_modal_launch,
    prepare_aperture_modal_launch,
    ModeInjectedPlaneSimulation,
    modal_plane_amplitudes,
    modal_s_parameters,
    yee_3d_modal_beta,
    conformal_variant0_cross_section,
    make_conformal_variant0_section,
    conformal_variant0_strip_rib_3d,
)
from torchfdtd.adjoint_planes import DifferentiablePlaneResult
from torchfdtd.mode_ports import C0


def test_conformal_variant0_anisotropic_callable_and_array_equivalence():
    """solve_waveguide_modes accepts both 3-arg callable(u, v, comp) and (Nu, Nv, 3) array."""
    eps_si = 3.4777**2
    eps_sio2 = 1.444**2
    # Rib waveguide: slab [-1.2, 1.2] x [-0.11, -0.02] + ridge [-0.25, 0.25] x [-0.02, 0.11]
    slabs = [
        (-1.2, 1.2, -0.11, -0.02, eps_si),
        (-0.25, 0.25, -0.02, 0.11, eps_si),
    ]
    shape = (48, 24)
    spacing = (0.05, 0.025)
    origin = (-1.2, -0.3)

    section_fn = make_conformal_variant0_section(
        slabs=slabs, dy_um=spacing[0], dz_um=spacing[1], eps_bg=eps_sio2, normal='x'
    )
    modes_fn = solve_waveguide_modes(
        section_fn, shape=shape, spacing_um=spacing, origin_um=origin,
        wavelength_um=1.55, normal='x', num_modes=2, precision='float64'
    )

    # Build explicit (Nu, Nv, 3) array at each component's native Yee positions
    eps_array = np.empty((*shape, 3), dtype=np.float64)
    for c, comp in enumerate(('Ex', 'Ey', 'Ez')):
        u_ax, v_ax = modes_fn[0].component_axes(comp)
        uu, vv = np.meshgrid(u_ax, v_ax, indexing='ij')
        eps_array[..., c] = conformal_variant0_cross_section(
            uu, vv, comp, slabs=slabs, dy_um=spacing[0], dz_um=spacing[1], eps_bg=eps_sio2, normal='x'
        )

    # Verify anisotropy at sidewall cells (Ey harmonic != Ez arithmetic != Ex arithmetic)
    assert not np.allclose(eps_array[..., 1], eps_array[..., 2])
    assert not np.allclose(eps_array[..., 0], eps_array[..., 1])

    modes_arr = solve_waveguide_modes(
        eps_array, shape=shape, spacing_um=spacing, origin_um=origin,
        wavelength_um=1.55, normal='x', num_modes=2, precision='float64'
    )
    for m1, m2 in zip(modes_fn, modes_arr):
        assert math.isclose(m1.beta_per_um, m2.beta_per_um, rel_tol=1e-13, abs_tol=1e-13)
        np.testing.assert_allclose(m1.fields, m2.fields, rtol=1e-12, atol=1e-12)
        assert m1.maxwell_residual < 1e-9

    # Polygon cross-section matches analytical rectangle cross-section
    poly_slabs = [
        ([(-0.23, -0.11), (0.23, -0.11), (0.23, 0.11), (-0.23, 0.11)], eps_si),
    ]
    rect_slabs = [(-0.23, 0.23, -0.11, 0.11, eps_si)]
    u_ax, v_ax = modes_fn[0].component_axes('Ey')
    uu, vv = np.meshgrid(u_ax, v_ax, indexing='ij')
    for comp in ('Ex', 'Ey', 'Ez'):
        got_poly = conformal_variant0_cross_section(
            uu, vv, comp, slabs=poly_slabs, dy_um=spacing[0], dz_um=spacing[1], eps_bg=eps_sio2, n_sub=64
        )
        got_rect = conformal_variant0_cross_section(
            uu, vv, comp, slabs=rect_slabs, dy_um=spacing[0], dz_um=spacing[1], eps_bg=eps_sio2
        )
        np.testing.assert_allclose(got_poly, got_rect, rtol=5e-3, atol=5e-2)


def test_conformal_variant0_strip_rib_3d_matches_aperture_modal_launch():
    """conformal_variant0_strip_rib_3d produces a 3D tensor whose cross-section slice launches cleanly."""
    faces = {a + '_' + s: dict(kind='periodic') for a in 'yz' for s in ('min', 'max')}
    r = Region(
        dimension='3d', size=(2.0, 1.6, 0.8), mesh=0.1, pml_cells=4, steps=16,
        material_sampling='yee', boundaries=faces, precision='float32'
    )
    p = Project(
        region=r,
        sources=[Source(kind='plane', normal='x', center=(-0.4, 0.0, 0.0), size=(0.0, 1.4, 0.8), pulse_cycles=2)],
        monitors=[FieldMonitor(id='out', normal='x', center=(0.4, 0.0, 0.0), size=(0.0, 1.4, 0.8))],
    )
    eps_core, eps_bg = 12.0, 2.1
    # Width tapers smoothly in the middle while staying constant in the launch/detector leads
    def width_fn(x):
        return 0.42 if abs(x) >= 0.2 else 0.42 + 0.08 * math.cos(math.pi * x / 0.4)

    eps_3d = conformal_variant0_strip_rib_3d(
        r, eps_core=eps_core, eps_bg=eps_bg, width_fn=width_fn, z_bounds=(-0.11, 0.11)
    )
    assert eps_3d.shape == r.shape + (3,)
    assert eps_3d.dtype == np.float32

    # Pass a 2D slice of eps_3d over the aperture (y: 1..15, z: 0..8) directly into prepare_aperture_modal_launch
    port_slice = eps_3d[0, 1:15, 0:8, :]
    launch_from_slice = prepare_aperture_modal_launch(p, port_slice)

    # Also pass the 3-arg callable section and verify bit-identical launch
    section_fn = make_conformal_variant0_section(
        slabs=[(-0.21, 0.21, -0.11, 0.11, eps_core)], dy_um=0.1, dz_um=0.1, eps_bg=eps_bg
    )
    launch_from_fn = prepare_aperture_modal_launch(p, section_fn)
    np.testing.assert_allclose(launch_from_slice.epsilon, launch_from_fn.epsilon, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(launch_from_slice.mode.fields, launch_from_fn.mode.fields, rtol=1e-5, atol=1e-5)


def test_deterministic_half_plane_phase_canonicalization_for_te1():
    """Antisymmetric TE1 mode deterministically selects the +u lobe without 180-deg phase flips."""
    eps_core, eps_bg = 12.1, 2.085
    shape = (40, 20)
    spacing = (0.04, 0.04)
    # Wide multimode waveguide supporting TE0 and TE1
    section = make_conformal_variant0_section(
        slabs=[(-0.45, 0.45, -0.11, 0.11, eps_core)], dy_um=spacing[0], dz_um=spacing[1], eps_bg=eps_bg
    )
    modes = solve_waveguide_modes(
        section, shape=shape, spacing_um=spacing, wavelength_um=1.55,
        normal='x', num_modes=3, precision='float64'
    )
    # Find the antisymmetric TE1 mode (two symmetric lobes in Ey across y=0 with opposite sign)
    te1 = next(
        m for m in modes
        if np.max(np.abs(m.fields[:shape[0] // 2, :, 1])) > 0.5 * np.max(np.abs(m.fields[..., 1]))
        and np.real(
            np.vdot(m.fields[:shape[0] // 2, :, 1].ravel(), m.fields[shape[0] // 2:][::-1, :, 1].ravel())
        ) < 0
    )
    # The upper half-plane (+y, u_idx >= shape[0]//2) peak of Ey must be real and strictly positive
    upper_ey = te1.fields[shape[0] // 2:, :, 1]
    lower_ey = te1.fields[:shape[0] // 2, :, 1]
    upper_peak = upper_ey.ravel()[np.argmax(np.abs(upper_ey))]
    lower_peak = lower_ey.ravel()[np.argmax(np.abs(lower_ey))]
    assert upper_peak.real > 0.0 and abs(upper_peak.imag) < 1e-12 * abs(upper_peak.real)
    assert lower_peak.real < 0.0

    # Now perturb the lower half-plane permittivity by +1e-10 so the lower lobe is slightly larger in magnitude
    # than the upper lobe; without half-plane canonicalization, argmax would flip to the lower lobe (-180 deg)!
    def perturbed_section(u, v, comp='Ey'):
        base = section(u, v, comp)
        return np.where(u < 0, base * (1.0 + 1e-10), base)

    modes_perturbed = solve_waveguide_modes(
        perturbed_section, shape=shape, spacing_um=spacing, wavelength_um=1.55,
        normal='x', num_modes=3, precision='float64'
    )
    te1_perturbed = modes_perturbed[modes.index(te1)]
    overlap = mode_power_overlap(te1, te1_perturbed)
    # Overlap must be +1.0 (in-phase), NOT -1.0 (180-deg flipped)!
    assert overlap.real == pytest.approx(1.0, abs=1e-5)
    assert abs(overlap.imag) < 1e-6


def test_broadband_multi_frequency_modal_plane_amplitudes_and_s_parameters():
    """modal_plane_amplitudes and modal_s_parameters support multi-frequency PlaneDFT and Sequence[ModalLaunch]."""
    faces = {a + '_' + s: dict(kind='periodic') for a in 'yz' for s in ('min', 'max')}
    r = Region(
        dimension='3d', size=(2.0, 1.2, 0.6), mesh=0.1, pml_cells=4, steps=16,
        material_sampling='yee', boundaries=faces, precision='float32'
    )
    p = Project(
        region=r,
        sources=[Source(kind='plane', normal='x', center=(-0.4, 0.0, 0.0), size=(0.0, 1.2, 0.6),
                        wavelength=1.55, pulse_cycles=2)],
        monitors=[FieldMonitor(id='out', normal='x', center=(0.4, 0.0, 0.0), size=(0.0, 1.2, 0.6))],
    )
    guide = lambda y, z: np.where((np.abs(y) < 0.25) & (np.abs(z) < 0.15), 12.0, 2.25)
    launch = prepare_modal_launch(p, guide, num_modes=1)
    mode = launch.detector_mode(0.4)

    u, v = np.meshgrid(
        -0.6 + (np.arange(12) + 0.5) * 0.1,
        -0.3 + (np.arange(6) + 0.5) * 0.1,
        indexing='ij',
    )
    points = np.stack([np.full(u.size, 0.4), u.ravel(), v.ravel()], axis=1)
    basis = torch.tensor(mode.sample_plane(points), dtype=torch.complex64)
    reverse = torch.tensor(mode.backward().sample_plane(points), dtype=torch.complex64)

    wavelengths = np.array([1.50, 1.55, 1.60], dtype=np.float32)
    freqs_hz = torch.tensor(C0 / (wavelengths * 1e-6), dtype=torch.float32)
    a_fwd = torch.tensor([1.2 + 0.1j, 1.5 - 0.2j, 0.9 + 0.3j], dtype=torch.complex64, requires_grad=True)
    b_bwd = torch.tensor([0.1 - 0.05j, -0.2 + 0.1j, 0.15 + 0.05j], dtype=torch.complex64, requires_grad=True)
    inc = torch.tensor([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64)

    scale = 1e-20
    sample_fields = scale * (a_fwd[:, None, None] * basis[None, :, :] + b_bwd[:, None, None] * reverse[None, :, :])
    ref_fields = scale * (inc[:, None, None] * basis[None, :, :])

    common = dict(
        frequency_hz=freqs_hz,
        points_um=torch.tensor(points, dtype=torch.float32),
        points_um_geometry=torch.tensor(points, dtype=torch.float64),
        weights=torch.full((len(points),), 0.01 * 1e-12, dtype=torch.float32),
        shape=(3, 12, 6),
        normal='x',
        run_signature='broadband_test',
        report={},
    )
    sample_plane = DifferentiablePlaneResult(fields=sample_fields, **common)
    ref_plane = DifferentiablePlaneResult(fields=ref_fields, **common)

    # 1. Single carrier launch across all 3 frequencies
    amps = modal_plane_amplitudes(sample_plane, launch)
    assert amps['forward'].shape == (3,) and amps['backward'].shape == (3,)
    torch.testing.assert_close(amps['forward'] / scale, a_fwd.detach(), rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(amps['backward'] / scale, b_bwd.detach(), rtol=2e-5, atol=1e-5)

    s_params = modal_s_parameters(sample_plane, ref_plane, launch)
    torch.testing.assert_close(s_params['transmission'], a_fwd.detach(), rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(s_params['reflection'], b_bwd.detach(), rtol=2e-5, atol=1e-5)

    # Autograd VJP through broadband modal_s_parameters
    loss = (s_params['transmission'].abs().square() + s_params['reflection'].abs().square()).sum()
    grad_a, grad_b = torch.autograd.grad(loss, (a_fwd, b_bwd))
    torch.testing.assert_close(grad_a, 2 * a_fwd.detach(), rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(grad_b, 2 * b_bwd.detach(), rtol=2e-5, atol=1e-5)

    # 2. Sequence of frequency-dependent launches (one per frequency)
    launches = []
    for wl in wavelengths:
        p_wl = p.model_copy(deep=True)
        p_wl.sources[0].wavelength = float(wl)
        launches.append(prepare_modal_launch(p_wl, guide, num_modes=1))
    amps_seq = modal_plane_amplitudes(sample_plane, launches)
    assert amps_seq['forward'].shape == (3,)

    # 3. Single-frequency mismatch still raises ValueError as required by existing contracts
    bad_single = replace(sample_plane, fields=sample_fields[:1], frequency_hz=torch.tensor([C0 / 1.30e-6]))
    with pytest.raises(ValueError, match='carrier frequency'):
        modal_plane_amplitudes(bad_single, launch)


def test_yee_3d_modal_beta_matches_launch_and_3d_fdtd_straight_waveguide():
    """yee_3d_modal_beta predicts exact 3D space-time dispersion and matches 3D FDTD propagation phase."""
    faces = {a + '_' + s: dict(kind='periodic') for a in 'yz' for s in ('min', 'max')}
    r = Region(
        dimension='3d', size=(4.8, 2.4, 0.5), mesh=0.1, pml_cells=8, steps=600,
        material_sampling='yee', boundaries=faces, precision='float32'
    )
    p = Project(
        region=r,
        sources=[Source(kind='plane', normal='x', center=(-1.4, 0.0, 0.0), size=(0.0, 1.2, 0.5),
                        wavelength=1.55, pulse_cycles=2)],
        monitors=[
            FieldMonitor(id='near', normal='x', center=(-1.0, 0.0, 0.0), size=(0.0, 1.2, 0.5)),
            FieldMonitor(id='far', normal='x', center=(1.0, 0.0, 0.0), size=(0.0, 1.2, 0.5)),
        ],
    )
    eps_core, eps_bg = 12.0, 2.1
    section = make_conformal_variant0_section(
        slabs=[(-0.2, 0.2, -0.25, 0.25, eps_core)], dy_um=0.1, dz_um=0.1, eps_bg=eps_bg
    )
    launch = prepare_aperture_modal_launch(p, section)

    # 1. Verify yee_3d_modal_beta on launch directly
    assert yee_3d_modal_beta(launch) == pytest.approx(launch.beta_yee_spacetime_per_um, rel=1e-12)
    assert yee_3d_modal_beta(launch, dw_um=0.1) == pytest.approx(launch.beta_yee_spacetime_per_um, rel=1e-12)

    # 2. Verify yee_3d_modal_beta on homogeneous 1D/3D mode where analytic space-time formula is exact
    hom_mode = solve_waveguide_modes(
        2.25, shape=(8, 8), spacing_um=(0.1, 0.1), wavelength_um=1.55, normal='x', num_modes=1, precision='float64'
    )[0]
    dt = r.time_step
    dw = 0.1
    omega = 2.0 * math.pi * C0 / 1.55e-6
    kt = 2.0 * math.sin(0.5 * omega * dt) / (C0 * dt) * 1e-6
    k0 = 2.0 * math.pi / 1.55
    expected_beta = 2.0 * math.asin(0.5 * dw * math.sqrt(hom_mode.beta_per_um**2 + kt**2 - k0**2)) / dw
    assert yee_3d_modal_beta(hom_mode, wavelength_um=1.55, dw_um=dw, dt_s=dt) == pytest.approx(expected_beta, rel=1e-14)

    # 3. Run 3D FDTD on the straight waveguide and verify S21 phase exp(i * beta_FDTD * L)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps_3d = conformal_variant0_strip_rib_3d(
        r, eps_core=eps_core, eps_bg=eps_bg, width_fn=0.4, z_bounds=(-0.25, 0.25)
    )
    eps_tensor = torch.tensor(eps_3d, device=device)
    sim = ModeInjectedPlaneSimulation(p, launch, AdjointOptions(checkpoints=4))
    with torch.no_grad():
        planes = sim(eps_tensor, [C0 / 1.55e-6])
    amp_near = modal_plane_amplitudes(planes['near'], launch)['forward'][0]
    amp_far = modal_plane_amplitudes(planes['far'], launch)['forward'][0]
    s21_measured = complex((amp_far / amp_near).cpu())
    distance_um = 2.0  # from x = -1.0 to x = +1.0
    s21_predicted = cmath_exp_i(launch.beta_yee_spacetime_per_um * distance_um)
    assert abs(abs(s21_measured) - 1.0) < 2e-3
    phase_err_rad = abs(math.atan2((s21_measured / s21_predicted).imag, (s21_measured / s21_predicted).real))
    assert phase_err_rad < 5e-3


def cmath_exp_i(phi: float) -> complex:
    return complex(math.cos(phi), math.sin(phi))


def test_forward_v1_admission_on_graded_mesh(monkeypatch):
    """forward-v1 admission activates on mesh_type='graded' under torch.no_grad()."""
    from torchfdtd.adjoint_memory import _resident_reservation

    r = Region(
        dimension='3d',
        size=(2.0, 2.0, 1.0),
        mesh=0.05,
        mesh_max=0.1,
        mesh_type='graded',
        mesh_refinements=[MeshRefinement(center=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.4))],
        pml_cells=4,
        steps=20,
        material_sampling='yee',
        precision='float32',
        cuda_kernel='fused',
    )
    p = Project(
        region=r,
        sources=[Source(kind='plane', normal='x', center=(-0.4, 0.0, 0.0), size=(0.0, 0.8, 0.4),
                        wavelength=1.55, pulse_cycles=2)],
        monitors=[],
    )
    monkeypatch.setenv('TORCHFDTD_ADMISSION_MODEL', 'forward-v1')
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device=None: 64 * 1024**2)
    monkeypatch.setattr('torchfdtd.adjoint_memory.cuda_budget_limit', lambda *a: 8 * 1024**3)
    with torch.no_grad():
        plan = _resident_reservation(p, AdjointOptions(checkpoints=4, storage='device'), 'cuda')
    assert plan['forward_only'] is True
    assert plan['admission_model'] == 'forward-v1'
    assert plan['workspace_model'] == 'forward_cuda_allocations_v1'
    assert plan['device_checkpoint_reservation_bytes'] == 0
    assert plan['unused_slot_count'] == 4


def test_open_waveguide_modes_graded_mesh_and_high_contrast_silicon_complex128():
    """solve_open_waveguide_modes and prepare_open_modal_launch support graded meshes and FP64 Silicon solves."""
    from types import SimpleNamespace
    from torchfdtd.open_mode_ports import solve_open_waveguide_modes
    from torchfdtd.open_mode_injection import OpenPortOptions, prepare_open_modal_launch

    eps_si = 3.4777**2
    eps_sio2 = 1.444**2
    faces = {
        a + '_' + s: dict(kind='pml', layers=8 if a != 'x' else 4)
        for a in 'xyz'
        for s in ('min', 'max')
    }
    r = Region(
        dimension='3d',
        size=(2.0, 3.0, 2.6),
        mesh=0.05,
        mesh_max=0.08,
        mesh_type='graded',
        mesh_refinements=[
            MeshRefinement(center=(0.0, 0.0, 0.0), size=(2.0, 0.6, 0.4))
        ],
        steps=20,
        boundaries=faces,
        material_sampling='yee',
        precision='float32',
        backend='cpu',
    )
    y_bounds = r.interior_bounds(1)
    z_bounds = r.interior_bounds(2)
    p = Project(
        region=r,
        sources=[
            Source(
                kind='plane',
                normal='x',
                center=(float(r.mesh_nodes[0][6]), 0.0, 0.0),
                size=(0.0, y_bounds[1] - y_bounds[0], z_bounds[1] - z_bounds[0]),
                wavelength=1.55,
                pulse_cycles=2,
            )
        ],
        monitors=[],
    )

    def si_strip(y, z):
        return np.where((np.abs(y) < 0.22) & (np.abs(z) < 0.11), eps_si, eps_sio2)

    opts = OpenPortOptions(
        cladding_epsilon=eps_sio2,
        target_neff=2.4,
        confinement_tolerance=1e-3,
        mode_budget_bytes=6 * 1024**3,
    )
    modes = solve_open_waveguide_modes(
        si_strip,
        region=r,
        normal='x',
        wavelength_um=1.55,
        cladding_epsilon=eps_sio2,
        num_modes=1,
        target_neff=opts.target_neff,
        confinement_tolerance=opts.confinement_tolerance,
        mode_budget_bytes=opts.mode_budget_bytes,
    )
    mode = modes[0]
    assert mode.diagnostics['eigensolver_dtype'] == 'complex128'
    assert mode.maxwell_residual < 2e-4
    assert mode.eigenpair_residual < 2e-4
    assert mode.transverse_nodes_um is not None
    assert 2.0 < mode.neff.real < 3.48

    # Verify native nonuniform midpoint quadrature validation
    pts_axes = []
    w_axes = []
    for local_axis, (lo, hi) in enumerate(mode.physical_bounds_um):
        nd = np.asarray(mode.transverse_nodes_um[local_axis])
        edges = np.r_[lo, nd[(nd > lo + 1e-10) & (nd < hi - 1e-10)], hi]
        pts_axes.append(0.5 * (edges[:-1] + edges[1:]))
        w_axes.append(np.diff(edges))
    yy, zz = np.meshgrid(*pts_axes, indexing='ij')
    wy, wz = np.meshgrid(*w_axes, indexing='ij')
    pts = np.c_[np.zeros(yy.size), yy.ravel(), zz.ravel()]
    weights = (wy * wz).ravel() * 1e-12
    plane = SimpleNamespace(normal='x', points_um=pts, weights=weights)
    assert mode.validate_quadrature(plane) > 0

    launch = prepare_open_modal_launch(p, si_strip, options=opts)
    assert launch.normal_nodes_um is not None
    det = launch.detector_mode(float(r.mesh_nodes[0][8]))
    assert det.fields.shape == mode.fields.shape


def test_auto_shutoff_early_termination_and_matched_adjoint_horizon():
    """AutoShutoff terminates early after source pulse decay and matches backward replay steps."""
    from torchfdtd import AutoShutoff, DifferentiableSimulation, Monitor

    r = Region(
        dimension='3d',
        size=(2.4, 2.4, 2.4),
        mesh=0.2,
        pml_cells=3,
        steps=400,
        material_sampling='yee',
        precision='float32',
    )
    p = Project(
        region=r,
        sources=[
            Source(
                kind='plane',
                normal='x',
                center=(-0.2, 0.0, 0.0),
                size=(0.0, 0.4, 0.4),
                wavelength=1.55,
                pulse_cycles=2,
            )
        ],
        monitors=[Monitor(center=(0.2, 0.0, 0.0))],
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps = torch.full(r.shape + (3,), 2.25, dtype=torch.float32, device=device, requires_grad=True)
    sim = DifferentiableSimulation(p, AdjointOptions(checkpoints=4))

    with AutoShutoff(
        min_time_fs=5.0,
        check_interval=16,
        energy_tolerance=1e-2,
        spectral_tolerance=5e-2,
        consecutive=2,
    ) as shutoff:
        res = sim.spectrum(eps, frequency_hz=[C0 / 1.55e-6], block_size=16)
        loss = res.fields.abs().square().sum()
        (grad,) = torch.autograd.grad(loss, (eps,))

    assert res.report['tail_converged'] is True
    assert res.report['actual_steps'] < res.report['maximum_steps']
    assert len(shutoff.checks) >= 2
    assert shutoff.checks[-1]['stable_checks'] >= 2
    assert torch.isfinite(grad).all()

