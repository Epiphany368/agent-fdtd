"""Angular-spectrum sections, volumes and points: analytic beams, the Green function, gradients and a metalens."""
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from torchfdtd import (plane_spectrum, propagate_points, propagate_section, propagate_volume, project_nearzone,
                       volume_bytes, propagate_plane)
from benchmarks.angular_spectrum import (analytic_dipole, face, fdtd_comparison, gaussian_plane, huygens_sheet,
                                         paraxial_beam, SIX)

C0 = 299792458.


def test_tilted_plane_wave_phase_is_exact_in_section_volume_and_points():
    wavelength, spacing, count = 1., .25, 96
    plane, ug, vg = gaussian_plane(1e9, wavelength, spacing, count)
    ku, kv = 2 * math.pi * 3 / (count * spacing), 2 * math.pi * 2 / (count * spacing)
    k = 2 * math.pi / wavelength
    wave = torch.exp(1j * (ku * ug + kv * vg))
    plane.fields[0, :, :, 0] = wave
    plane.fields[0, :, :, 4] = 2 * wave
    distances = [7.3, 11.1]
    kn = math.sqrt(k**2 - ku**2 - kv**2)
    volume = propagate_volume(plane, distances, components=('Ex', 'Hy'), pad=1)
    section = propagate_section(plane, distances, section='xz', offset_um=float(plane.v_um[5]), components=('Ex', 'Hy'), pad=1)
    points = torch.stack([ug[3, 5], vg[3, 5], torch.tensor(distances[1], dtype=torch.float64)]).reshape(1, 3)
    direct = propagate_points(plane, points, components=('Ex', 'Hy'), pad=1)
    for i, distance in enumerate(distances):
        expected = wave * complex(math.cos(kn * distance), math.sin(kn * distance))
        torch.testing.assert_close(volume.fields[0, i, :, :, 0], expected, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(volume.fields[0, i, :, :, 1], 2 * expected, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(section.fields[0, i, :, 0], expected[:, 5], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(direct.fields[0, 0, 0], wave[3, 5] * complex(math.cos(kn * distances[1]), math.sin(kn * distances[1])), rtol=1e-9, atol=1e-9)
    assert volume.normal_um.tolist() == pytest.approx(distances) and section.axes == ('z', 'x')
    with pytest.raises(ValueError, match='nonnegative'):
        propagate_section(plane, [-1.], section='xz')
    with pytest.raises(ValueError, match='propagation side'):
        propagate_points(plane, [[0., 0., -1.]])
    with pytest.raises(ValueError, match='section must pair'):
        propagate_section(plane, [1.], section='xy')


def test_gaussian_beam_matches_closed_form_and_sampling_limit():
    wavelength = 1.
    # Waist series: the transform is exact, the remaining error is the paraxial one and falls fourfold per doubled waist.
    errors = []
    for w0 in (2., 4., 8.):
        plane, ug, vg = gaussian_plane(w0, wavelength, .25, 192)
        rayleigh = math.pi * w0**2 / wavelength
        section = propagate_section(plane, [rayleigh], section='xz', components=('Ex',), pad=2)
        u = plane.u_um
        expected = paraxial_beam(w0, wavelength, u, torch.zeros_like(u), torch.tensor(rayleigh))
        errors.append(float((section.fields[0, 0, :, 0] - expected).norm() / expected.norm()))
    assert errors[0] < 1e-2 and all(a / b > 3.5 for a, b in zip(errors, errors[1:]))
    # A beam tilted by 30 degrees: above the Nyquist spacing the carrier aliases, below it the error is the paraxial floor.
    w0, tilt = 3., 30.
    rayleigh = math.pi * w0**2 / wavelength
    sampled = {}
    for spacing in (1.25, .75, .25):
        plane, ug, vg = gaussian_plane(w0, wavelength, spacing, int(round(96 / spacing)), tilt_deg=tilt)
        section = propagate_section(plane, [rayleigh], section='xz', components=('Ex',), pad=2)
        u = plane.u_um
        expected = paraxial_beam(w0, wavelength, u, torch.zeros_like(u), torch.tensor(rayleigh), tilt_deg=tilt)
        sampled[spacing] = float((section.fields[0, 0, :, 0] - expected).norm() / expected.norm())
        assert section.report['max_angle_deg'][0] == pytest.approx(math.degrees(math.asin(min(1., wavelength / (2 * spacing)))))
    assert sampled[1.25] > .5 and sampled[.75] < .05 and sampled[.25] < 1e-2 and sampled[.25] <= sampled[.75]


def test_points_agree_with_green_function_for_a_directed_beam_and_not_for_a_truncated_dipole():
    waist, half, height = .8 * 1.55, 4 * 1.55, 3 * 1.55
    bounds = ((-half, half), (-half, half), (-height, height))
    faces = {d + ('_min' if side == 0 else '_max'): replace(p := face(d, bounds, side, 48, 1.55), fields=huygens_sheet(p.points_um, waist))
             for d in 'xyz' for side in (0, 1)}
    points = torch.tensor([[0, 0, height + .3], [0, 0, height + 1], [1., .5, height + 1], [2., 0, height + 3]], dtype=torch.float64)
    exact = huygens_sheet(points, waist)
    near = project_nearzone(faces, points, bounds_um=bounds, refractive_index=1.)
    asm = propagate_points(faces['z_max'], points, components=SIX, pad=2)
    beam = ((asm.fields - near.fields).norm(dim=-1) / near.fields.norm(dim=-1))[0]
    # Declared tolerance: 8e-3 between the single-plane ASM and the closed box, both within 5e-3 of the exact beam.
    assert float(beam.max()) < 8e-3 and float(((near.fields - exact).norm(dim=-1) / exact.norm(dim=-1)).max()) < 5e-3
    assert float(((asm.fields - exact).norm(dim=-1) / exact.norm(dim=-1)).max()) < 5e-3
    # A dipole 0.55 um below a 1.3 x 1.4 um face: the single plane cannot hold its 1/r field, the closed box can.
    bounds = ((-.65, .65), (-.7, .7), (-.6, .6))
    faces = {d + ('_min' if side == 0 else '_max'): replace(p := face(d, bounds, side, 28, 1.55), fields=analytic_dipole(p.points_um))
             for d in 'xyz' for side in (0, 1)}
    points = torch.tensor([[.12, -.08, .65], [.12, -.08, 1.4]], dtype=torch.float64)
    exact = analytic_dipole(points)
    near = project_nearzone(faces, points, bounds_um=bounds, refractive_index=1.3)
    asm = propagate_points(faces['z_max'], points, index=1.3, components=SIX, pad=4)
    dipole = ((asm.fields - exact).norm(dim=-1) / exact.norm(dim=-1))[0]
    assert float(dipole[0]) < 3e-2 and float(dipole[1]) > .2
    # The closed box stays accurate 0.8 um above the face (its midpoint rule needs finer faces for the nearer point).
    assert float(((near.fields - exact).norm(dim=-1) / exact.norm(dim=-1))[0, 1]) < 5e-3


def test_section_volume_points_and_propagate_plane_are_consistent():
    torch.manual_seed(3)
    plane, ug, vg = gaussian_plane(2.5, 1., .5, 24)
    plane.fields = plane.fields + .05 * torch.randn_like(plane.fields)
    distances = [.7, 1.9, 3.1, 4.3]
    volume = propagate_volume(plane, distances, components=('Ex', 'Ey', 'Hz'), pad=2, chunk=3)
    for section, axis, index in (('xz', 1, 7), ('yz', 0, 9)):
        offset = float((plane.v_um if section == 'xz' else plane.u_um)[index])
        cut = propagate_section(plane, distances, section=section, offset_um=offset, components=('Ex', 'Ey', 'Hz'), pad=2, chunk=64)
        expected = volume.fields[0, :, :, index] if section == 'xz' else volume.fields[0, :, index, :]
        torch.testing.assert_close(cut.fields[0], expected, rtol=1e-12, atol=1e-14)
    points = torch.tensor([[float(plane.u_um[4]), float(plane.v_um[6]), distances[2]]], dtype=torch.float64)
    direct = propagate_points(plane, points, components=('Ex', 'Ey', 'Hz'), pad=2)
    torch.testing.assert_close(direct.fields[0, 0], volume.fields[0, 2, 4, 6], rtol=1e-12, atol=1e-14)
    single = propagate_plane(plane, distances[1], 1., pad=2)
    torch.testing.assert_close(single.fields[0, :, :, [0, 1, 5]], volume.fields[0, 1], rtol=1e-12, atol=1e-14)
    # FP32 planes follow the same path at single precision.
    fp32 = replace(plane, fields=plane.fields.to(torch.complex64))
    volume32 = propagate_volume(fp32, distances, components=('Ex', 'Ey', 'Hz'), pad=2, chunk=3)
    assert volume32.fields.dtype == torch.complex64
    torch.testing.assert_close(volume32.fields.to(torch.complex128), volume.fields, rtol=1e-5, atol=1e-5 * float(volume.fields.abs().max()))
    assert volume.intensity().shape == (1, 4, 24, 24)
    spectrum = plane_spectrum(plane, 2)
    assert spectrum.report['padded_shape'] == (48, 48) and 0 < spectrum.report['evanescent_fraction'][0] < 1


def test_focal_objective_gradient_passes_a_taylor_check_and_matches_plain_autograd():
    torch.manual_seed(5)
    base, ug, vg = gaussian_plane(2., 1., .5, 16)
    base.fields = base.fields + .1 * torch.randn_like(base.fields)
    distances = torch.tensor([1.5, 3., 4.5, 6.])

    def objective(fields):
        section = propagate_section(replace(base, fields=fields), distances, section='xz', offset_um=.3, components=('Ex', 'Ey', 'Ez'), chunk=2)
        volume = propagate_volume(replace(base, fields=fields), distances[:2], components=('Ex', 'Ez'), chunk=1)
        focus = section.focus()
        return section.intensity()[0, 2, 6:10].sum() + .3 * volume.intensity()[0, 1, 5:9, 5:9].sum(), focus

    def naive(fields):
        spectrum = plane_spectrum(replace(base, fields=fields), 2)
        total = 0.
        for c in (0, 1, 2):
            s = spectrum.spectrum(0, c)
            real, imaginary = spectrum.normal_wavenumber(0)
            transfer = torch.polar(torch.exp(-imaginary * 4.5), torch.remainder(real * 4.5, 2 * math.pi))
            y = s * torch.exp(1j * spectrum.kv * (.3 - spectrum.v_origin))[None, :] / spectrum.pv
            row = torch.fft.ifft((y * transfer).sum(1))[spectrum.ou:spectrum.ou + spectrum.nu]
            total = total + row[6:10].abs().square().sum()
        for c in (0, 2):
            s = spectrum.spectrum(0, c)
            transfer = torch.polar(torch.exp(-imaginary * 3.), torch.remainder(real * 3., 2 * math.pi))
            field = torch.fft.ifft2(s * transfer)[spectrum.ou:spectrum.ou + spectrum.nu, spectrum.ov:spectrum.ov + spectrum.nv]
            total = total + .3 * field[5:9, 5:9].abs().square().sum()
        return total

    x = base.fields.clone().requires_grad_()
    value, focus = objective(x)
    gradient, = torch.autograd.grad(value, x)
    assert torch.isfinite(gradient).all() and math.isfinite(focus['fwhm_um']) and focus['fwhm_um'] > 0
    x2 = base.fields.clone().requires_grad_()
    reference, = torch.autograd.grad(naive(x2), x2)
    torch.testing.assert_close(gradient, reference, rtol=1e-10, atol=1e-12)
    # Taylor: the first-order prediction along a random complex direction is second-order accurate.
    direction = torch.randn_like(base.fields)
    predicted = (gradient.conj() * direction).real.sum()   # d/dt f(x + t d) for a real objective of complex fields
    residuals = []
    for step in (1e-2, 5e-3):
        with torch.no_grad():
            actual = objective(base.fields + step * direction)[0] - value.detach()
        residuals.append(abs(float(actual - step * predicted)))
    assert 3.5 < residuals[0] / residuals[1] < 4.5 and residuals[1] < .1 * abs(float(step * predicted))


def test_volume_budget_rejection_and_npz_chunks(tmp_path):
    plane, ug, vg = gaussian_plane(2., 1., .5, 16)
    distances = list(np.linspace(.5, 5., 7))
    estimate = volume_bytes(plane, distances, components=('Ex', 'Ey'), pad=2, chunk=3)
    assert estimate['output_bytes'] == 7 * 16 * 16 * 2 * 16 and estimate['planes'] == 7
    with pytest.raises(ValueError, match='above the budget'):
        propagate_volume(plane, distances, components=('Ex', 'Ey'), pad=2, chunk=3, budget_bytes=estimate['output_bytes'])
    with pytest.raises(ValueError, match='budget_bytes'):
        propagate_volume(plane, distances, budget_bytes=0)
    memory = propagate_volume(plane, distances, components=('Ex', 'Ey'), pad=2, chunk=3)
    path = tmp_path / 'volume.npz'
    written = propagate_volume(plane, distances, components=('Ex', 'Ey'), pad=2, chunk=3, output=path,
                               budget_bytes=estimate['working_bytes'] + 1)
    assert written.fields is None and written.path == str(path)
    with pytest.raises(ValueError, match='written to disk'):
        written.intensity()
    with np.load(path) as archive:
        chunks = [archive[f'fields_f0_z{start}'] for start in range(0, 7, 3)]
        assert list(archive['components']) == ['Ex', 'Ey'] and archive['x_um'].shape == (16,) and archive['z_um'].tolist() == pytest.approx(distances)
    torch.testing.assert_close(torch.as_tensor(np.concatenate(chunks)), memory.fields[0], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        propagate_volume(plane, distances, components=('Ex',), output=path)


def test_float32_plane_coordinates_far_from_the_origin_count_as_uniform():
    from torchfdtd.angular_spectrum import plane_grid
    from torchfdtd import DifferentiablePlaneResult
    # 0.4 um samples of a 120 um plane held in float32: the coordinate rounding
    # near 60 um is about 4e-6 um, above the old 1e-6 x spacing tolerance.
    axis = torch.arange(-59.8, 60., .4, dtype=torch.float64)
    points = torch.stack(torch.meshgrid(axis, axis, torch.zeros(1, dtype=torch.float64), indexing='ij'), -1).reshape(-1, 3)
    stored = points.to(torch.float32)
    steps = stored.reshape(len(axis), len(axis), 3)[:, 0, 0].double().diff()
    assert float((steps - .4).abs().max()) > 1e-6 * .4
    plane = DifferentiablePlaneResult(torch.zeros((1, len(axis)**2, 6), dtype=torch.complex64), torch.tensor([2e14], dtype=torch.float64),
                                      stored, torch.full((len(axis)**2,), .16e-12, dtype=torch.float32), (len(axis), len(axis), 1), 'z', 'sig', {})
    grid = plane_grid(plane)
    assert grid.spacing_um == pytest.approx((.4, .4), rel=1e-5) and grid.fields.shape == (1, len(axis), len(axis), 6)
    uneven = stored.clone().reshape(len(axis), len(axis), 3)
    uneven[::2, :, 0] += .01
    uneven = uneven.reshape(-1, 3)
    with pytest.raises(ValueError, match='uniformly'):
        plane_grid(replace(plane, points_um=uneven))


def test_small_metalens_fdtd_versus_asm_on_cpu():
    record = fdtd_comparison('2d', 'cpu', mesh=.1, aperture=4., focal=4., radii=[.06, .1, .14, .18, .22], empty=False,
                             distances_um=(.5, 1.5, 3., 4.5), extra_fs=30.)
    assert record['pillars'] == 8 and record['runs']['plane_agreement'] < 2e-2
    errors = [row['intensity_error'] for row in record['planes']]
    # Declared tolerance for this coarse 0.1 um mesh: 6% in intensity at every recorded plane.
    assert max(errors) < .06 and all(.7 < row['peak_ratio'] < 1.3 for row in record['planes'])
    section = record['section']
    fdtd, asm = section['focus_fdtd_from_recorded_planes'], section['focus_asm_at_that_plane']
    assert asm['z_um'] == fdtd['z_um'] and abs(asm['fwhm_um'] - fdtd['fwhm_um']) < .1 * fdtd['fwhm_um']
    assert abs(asm['peak_intensity'] - fdtd['peak_intensity']) < .1 * fdtd['peak_intensity']
    assert record['runs']['to_plane']['cells'] < record['runs']['through_focus']['cells'] / 2
