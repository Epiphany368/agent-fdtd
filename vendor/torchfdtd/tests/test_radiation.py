"""Analytic Rayleigh waves, closed-surface dipoles and a native FDTD VJP."""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from torchfdtd.adjoint_planes import DifferentiablePlaneResult
from torchfdtd.radiation import (C0, diffraction_orders, diffraction_efficiency,
    project_farfield, normalized_farfield_intensity, project_nearzone, farfield_at_points,
    spherical_directions, spherical_points, cartesian_plane_points, kspace_directions)


def plane(normal='z', bounds=((-1, 1), (-1, 1), (-1, 1)), side=1, count=16,
          wavelength=1.1, dtype=torch.float64):
    a = 'xyz'.index(normal)
    coords = []
    for d in range(3):
        lo, hi = bounds[d]
        coords.append(torch.tensor([bounds[d][side]], dtype=dtype) if d == a else
                      lo + (torch.arange(count, dtype=dtype) + .5) * (hi - lo) / count)
    points = torch.stack(torch.meshgrid(*coords, indexing='ij'), -1).reshape(-1, 3)
    area = math.prod(bounds[d][1] - bounds[d][0] for d in range(3) if d != a) * 1e-12
    return DifferentiablePlaneResult(torch.zeros((1, len(points), 6), dtype=torch.complex64 if dtype == torch.float32 else torch.complex128),
        torch.tensor([C0 / (wavelength * 1e-6)], dtype=dtype), points,
        torch.full((len(points),), area / len(points), dtype=dtype),
        tuple(len(c) for c in coords), normal, 'same-run', {})


def rayleigh(p, order, *, n=1.4, direction=1, bloch=(.17, -.13), amplitude=(.7 + .2j, .3 - .4j)):
    a = 'xyz'.index(p.normal)
    b, c = (a + 1) % 3, (a + 2) % 3
    k0 = 2 * math.pi * float(p.frequency_hz[0]) / C0 * 1e-6
    k = torch.zeros(3, dtype=p.fields.dtype)
    k[b], k[c] = bloch[0] + math.pi * order[0], bloch[1] + math.pi * order[1]
    k[a] = direction * torch.sqrt(torch.as_tensor((n * k0)**2, dtype=p.fields.dtype) - k[b]**2 - k[c]**2)
    e = torch.zeros(3, dtype=p.fields.dtype)
    e[b], e[c] = amplitude
    e[a] = -(k[b]*e[b] + k[c]*e[c]) / k[a]
    h = torch.linalg.cross(k, e) / k0
    origin = p.points_um.mean(0)
    phase = torch.exp(1j * ((p.points_um - origin).to(k.dtype) @ k))
    return phase[None, :, None] * torch.cat((e, h))[None, None, :], torch.cat((e, h))


@pytest.mark.parametrize('normal', ['x', 'y', 'z'])
def test_rayleigh_orders_bloch_direction_complex_phase_and_power_sum(normal):
    p = plane(normal)
    one, expected_one = rayleigh(p, (1, -1))
    two, expected_two = rayleigh(p, (-1, 0), direction=-1)
    p.fields = one + .4j * two
    result = diffraction_orders(p, [(1, -1), (-1, 0), (0, 0)], period_um=(2, 2), refractive_index=1.4,
                                bloch_wavevector_per_um=(.17, -.13))
    torch.testing.assert_close(result.forward_fields[0, 0], expected_one, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result.backward_fields[0, 1], .4j * expected_two, rtol=1e-12, atol=1e-12)
    assert result.backward_power[0, 0] < 1e-40
    assert result.forward_power[0, 1] < 1e-40
    torch.testing.assert_close((result.forward_power - result.backward_power).sum(-1), p.flux(), rtol=1e-12, atol=1e-26)


def test_evanescent_amplitudes_zero_power_and_cutoff_rejection():
    p = plane(wavelength=2.)
    p.fields, expected = rayleigh(p, (2, 0), n=1., bloch=(0., 0.))
    result = diffraction_orders(p, [(2, 0)], period_um=(2, 2))
    assert not bool(result.propagating[0, 0])
    torch.testing.assert_close(result.forward_fields[0, 0], expected, rtol=1e-12, atol=1e-12)
    assert result.forward_power[0, 0] == 0
    with pytest.raises(ValueError, match='cutoff'):
        diffraction_orders(p, [(1, 0)], period_um=(2, 2))


def test_fp32_efficiency_preserves_sample_and_reference_gradients_at_si_scales():
    base = plane(dtype=torch.float32)
    base.fields, _ = rayleigh(base, (0, 0), bloch=(0., 0.))
    a, r = [torch.tensor(x, requires_grad=True) for x in (1.7, .8)]
    sample = replace(base, fields=base.fields * a * 1e-22)
    reference = replace(base, fields=base.fields * r * 1e-22)
    ratio = diffraction_efficiency(sample, reference, [(0, 0)], period_um=(2, 2), refractive_index=1.4)
    torch.testing.assert_close(ratio, (a/r).square().reshape(1, 1), rtol=1e-6, atol=1e-6)
    got = torch.autograd.grad(ratio.sum(), (a, r))
    for actual, expected in zip(got, (2*a/r**2, -2*a*a/r**3)):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)


def test_fp32_grazing_cutoff_roundoff_is_rejected_for_multiple_indices():
    for n in (.7, 1., 1.1, 1.2, 1.3, 1.4, 1.7, 3.4):
        p = plane(dtype=torch.float32, wavelength=2*n)
        with pytest.raises(ValueError, match='cutoff'):
            diffraction_orders(p, [(1, 0)], period_um=(2, 2), refractive_index=n)


def analytic_dipole(points, *, n=1.3, shift=(.12, -.08, .05), mu=1., dipole=None):
    """Exact E/H of a vector dipole at 1.55 um, all near/intermediate/far terms, shape (1, P, 6).

    The closed forms hold for complex ``n`` (lossy exterior, ``k = 2 pi n / wavelength``)
    and relative permeability ``mu``: E carries ``mu / n^2 = 1 / eps_r``, H carries ``k^2 / n``.
    """
    dtype = points.dtype
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    dipole = torch.tensor([.3 + .1j, -.2j, 1.], dtype=cdtype) if dipole is None else dipole.to(cdtype)
    rvec = points - torch.tensor(shift, dtype=dtype)
    r = rvec.norm(dim=-1)
    unit = rvec.to(dipole.dtype) / r[:, None]
    dot = (unit * dipole).sum(-1, keepdim=True)
    transverse = dipole - unit * dot
    k = 2*math.pi*n/1.55
    wave = torch.exp(1j*k*r)[:, None]
    electric = wave*mu/n**2 * (k*k*transverse/r[:, None] + (3*unit*dot-dipole)*(1/r**3-1j*k/r**2)[:, None])
    magnetic = wave*k*k/n * torch.linalg.cross(unit, dipole.expand_as(unit)) * (1/r + 1j/(k*r*r))[:, None]
    return torch.cat((electric, magnetic), -1)[None], dipole


def dipole_faces(count, *, n=1.3, shift=(.12, -.08, .05), dtype=torch.float64, amplitude=1., mu=1.):
    bounds = ((-.65, .65), (-.7, .7), (-.6, .6))
    faces = {}
    for d in 'xyz':
        for side in (0, 1):
            p = plane(d, bounds, side, count, wavelength=1.55, dtype=dtype)
            fields, dipole = analytic_dipole(p.points_um, n=n, shift=shift, mu=mu)
            faces[d + ('_min' if side == 0 else '_max')] = replace(p, fields=amplitude*fields)
    return faces, bounds, dipole


def huygens_sheet(points, waist, spacing=.5*1.55):
    """Gaussian-apodized sheet of Huygens pairs (p = x, m = -y) in z = 0 radiating toward +z.

    The magnetic dipole fields follow from the electric ones by duality,
    (E, H) -> (H/n, -n E), in vacuum. Backward radiation cancels on axis and
    the sheet is truncated at 2.5 waists (amplitude 2e-3).
    """
    extent = 2.5*waist
    coords = torch.arange(-extent, extent + 1e-9, spacing, dtype=torch.float64)
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    amplitude = torch.exp(-(x**2 + y**2)/waist**2).reshape(-1)
    px = torch.tensor([1., 0, 0], dtype=torch.complex128)
    py = torch.tensor([0, 1., 0], dtype=torch.complex128)
    fields = torch.zeros((1, len(points), 6), dtype=torch.complex128)
    for xi, yi, a in zip(x.reshape(-1).tolist(), y.reshape(-1).tolist(), amplitude.tolist()):
        ex, _ = analytic_dipole(points, n=1., shift=(xi, yi, 0.), dipole=px)
        ey, _ = analytic_dipole(points, n=1., shift=(xi, yi, 0.), dipole=py)
        fields = fields + a*torch.cat((ex[..., :3] - ey[..., 3:], ex[..., 3:] + ey[..., :3]), -1)
    return fields


def beam_faces(waist, count, half=4*1.55, height=3*1.55):
    bounds = ((-half, half), (-half, half), (-height, height))
    faces = {}
    for d in 'xyz':
        for side in (0, 1):
            p = plane(d, bounds, side, count, wavelength=1.55)
            faces[d + ('_min' if side == 0 else '_max')] = replace(p, fields=huygens_sheet(p.points_um, waist))
    return faces, bounds


def test_closed_surface_dipole_complex_amplitude_converges_and_translation_phase():
    directions = torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [-1., 0, 0], [1., 2., 3.]], dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    errors = []
    for count in (14, 28, 56):
        faces, bounds, dipole = dipole_faces(count)
        result = project_farfield(faces, directions, bounds_um=bounds, refractive_index=1.3,
                                  direction_chunk=3, point_chunk=153)
        transverse = dipole - directions * (directions * dipole).sum(-1, keepdim=True)
        phase = torch.exp(-1j*(2*math.pi*1.3/1.55)*(directions @ torch.tensor([.12, -.08, .05], dtype=torch.float64)))
        expected = (2*math.pi/1.55)**2 * transverse * phase[:, None] * 1e-6
        errors.append(float((result.electric_amplitude[0] - expected).norm()/expected.norm()))
    assert errors[0]/errors[1] > 3.8 and errors[1]/errors[2] > 3.8
    assert errors[-1] < 2e-4
    assert (result.electric_amplitude * directions).sum(-1).abs().max() < 1e-19
    remote = result.fields_at_radius(.5)
    torch.testing.assert_close(remote[..., 3:], 1.3*torch.linalg.cross(directions.to(remote.dtype)[None], remote[..., :3]))


def test_closed_surface_near_flux_matches_sphere_integrated_far_power():
    faces, bounds, _ = dipole_faces(28)
    z, weight = np.polynomial.legendre.leggauss(12)
    phi = np.arange(24)*2*np.pi/24
    direction = np.stack([np.sqrt(1-z[:, None]**2)*np.cos(phi)[None],
                          np.sqrt(1-z[:, None]**2)*np.sin(phi)[None], np.broadcast_to(z[:, None], (12, 24))], -1).reshape(-1, 3)
    result = project_farfield(faces, direction, bounds_um=bounds, refractive_index=1.3)
    far = (result.intensity().reshape(12, 24)*torch.tensor(weight)[:, None]).sum()*2*np.pi/24
    near = sum(face.flux()[0] * (-1 if name.endswith('min') else 1) for name, face in faces.items())
    torch.testing.assert_close(far, near, rtol=5e-4, atol=1e-25)


def test_fp32_farfield_both_graphs_and_phase_origin():
    a, r = [torch.tensor(x, requires_grad=True) for x in (1.7, .8)]
    faces, bounds, _ = dipole_faces(16, dtype=torch.float32, amplitude=a*1e-22)
    reference = plane(dtype=torch.float32, wavelength=1.55)
    reference.fields, _ = rayleigh(reference, (0, 0), n=1.3, bloch=(0., 0.))
    reference.fields = reference.fields*r*1e-22
    ratio = normalized_farfield_intensity(faces, reference, [[1., 0, 0]], bounds_um=bounds, refractive_index=1.3)
    gradients = torch.autograd.grad(ratio.sum(), (a, r))
    torch.testing.assert_close(gradients[0], 2*ratio.sum()/a, rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(gradients[1], -2*ratio.sum()/r, rtol=2e-6, atol=1e-6)
    unscaled, bounds, _ = dipole_faces(12)
    direct = project_farfield(unscaled, [[1., 0, 0]], bounds_um=bounds, refractive_index=1.3)
    shifted = project_farfield(unscaled, [[1., 0, 0]], bounds_um=bounds, refractive_index=1.3, phase_origin_um=(.2, 0, 0))
    torch.testing.assert_close(shifted.electric_amplitude, direct.electric_amplitude*complex(np.exp(1j*2*np.pi*1.3/1.55*.2)))


def test_invalid_surface_quadrature_and_conventions_fail_explicitly():
    p = plane()
    with pytest.raises(ValueError, match='Nyquist'):
        diffraction_orders(p, [(8, 0)], period_um=(2, 2))
    with pytest.raises(ValueError, match='integer'):
        diffraction_orders(p, [(1.2, 0)], period_um=(2, 2))
    with pytest.raises(ValueError, match='midpoints'):
        diffraction_orders(p, [(0, 0)], period_um=(1., 2.))
    faces, bounds, _ = dipole_faces(4)
    with pytest.raises(ValueError, match='six'):
        project_farfield({'x_min': faces['x_min']}, [[1., 0, 0]], bounds_um=bounds)
    faces['z_max'] = replace(faces['z_max'], weights=faces['z_max'].weights*2)
    with pytest.raises(ValueError, match='weights'):
        project_farfield(faces, [[1., 0, 0]], bounds_um=bounds)


def test_fp32_farfield_macroscopic_radius_preserves_coherent_phase():
    from torchfdtd.radiation import FarFieldResult
    frequency = torch.tensor([C0/1.55e-6], dtype=torch.float32)
    amplitude = torch.tensor([[[0, 1., 0]]], dtype=torch.complex64, requires_grad=True)
    result = FarFieldResult(amplitude, torch.tensor([[1., 0, 0]]), frequency, 1.3, torch.zeros(3))
    got = result.fields_at_radius(.5)
    phase = complex(np.exp(1j*2*np.pi*float(frequency[0])/C0*1.3*.5)/.5)
    assert abs(complex(got[0, 0, 1].detach())-phase) < 2e-7
    assert got.dtype == torch.complex64
    gradient, = torch.autograd.grad(got.abs().square().sum(), amplitude)
    assert torch.isfinite(gradient).all()


def test_complex_radiation_geometry_is_rejected_without_silent_real_cast():
    with pytest.raises(ValueError, match='must be real'):
        diffraction_orders(plane(), [(0, 0)], period_um=(2, 2),
                           bloch_wavevector_per_um=torch.tensor([.2+1j, 0j]))
    faces, bounds, _ = dipole_faces(4)
    with pytest.raises(ValueError, match='must be real'):
        project_farfield(faces, torch.tensor([[1+1j, 0j, 0j]]), bounds_um=bounds)
    with pytest.raises(ValueError, match='must be real'):
        project_farfield(faces, [[1., 0, 0]], bounds_um=bounds, phase_origin_um=torch.tensor([1j, 0j, 0j]))


def test_nearzone_dipole_fields_converge_with_radial_component():
    # Every point is at least .55 um (about 24 count-56 cells) outside the
    # box. The midpoint rule needs the face spacing small against the
    # distance to the face and the wavelength; second order is then observed.
    points = torch.tensor([[1.2, .3, -.4], [0., -1.6, .2], [.5, .5, 1.3], [-2., .1, .1], [1., 1., 1.]], dtype=torch.float64)
    expected, _ = analytic_dipole(points)
    errors = []
    for count in (14, 28, 56):
        faces, bounds, _ = dipole_faces(count)
        got = project_nearzone(faces, points, bounds_um=bounds, refractive_index=1.3, point_chunk=97, observation_chunk=2)
        assert got.fields.shape == (1, 5, 6) and got.fields.dtype == torch.complex128
        errors.append(float((got.fields - expected).norm()/expected.norm()))
    assert errors[0]/errors[1] > 3.5 and errors[1]/errors[2] > 3.5
    assert errors[-1] < 5e-4
    unit = points - torch.tensor([.12, -.08, .05], dtype=torch.float64)
    unit = (unit / unit.norm(dim=-1, keepdim=True)).to(got.fields.dtype)
    radial = (got.fields[0, :, :3]*unit).sum(-1).abs() / got.fields[0, :, :3].norm(dim=-1)
    assert radial.max() > .2
    poynting = got.poynting()
    assert poynting.shape == (1, 5, 3) and poynting.dtype == torch.float64
    torch.testing.assert_close(poynting, .5*torch.linalg.cross(expected[..., :3], expected[..., 3:].conj()).real, rtol=2e-3, atol=0)
    # FP32 faces give the same fields at FP32 accuracy with an FP32 result.
    faces, bounds, _ = dipole_faces(28, dtype=torch.float32)
    single = project_nearzone(faces, points.float(), bounds_um=bounds, refractive_index=1.3)
    assert single.fields.dtype == torch.complex64
    assert float((single.fields - expected.to(torch.complex64)).norm()/expected.norm()) < 2e-3


def test_nearzone_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    faces, bounds, _ = dipole_faces(12, dtype=torch.float32)
    points = torch.tensor([[1.5, .2, .1], [0., 0., -2.]], dtype=torch.float32)
    host = project_nearzone(faces, points, bounds_um=bounds, refractive_index=1.3)
    moved = {name: replace(face, fields=face.fields.cuda(), frequency_hz=face.frequency_hz.cuda(),
                           points_um=face.points_um.cuda(), weights=face.weights.cuda()) for name, face in faces.items()}
    device = project_nearzone(moved, points.cuda(), bounds_um=bounds, refractive_index=1.3, observation_chunk=1)
    assert device.fields.device.type == 'cuda' and device.fields.dtype == torch.complex64
    torch.testing.assert_close(device.fields.cpu(), host.fields, rtol=1e-5, atol=0)
    far = farfield_at_points(moved, points.cuda(), bounds_um=bounds, refractive_index=1.3)
    torch.testing.assert_close(far.cpu(), farfield_at_points(faces, points, bounds_um=bounds, refractive_index=1.3), rtol=1e-5, atol=0)


def test_nearzone_converges_to_farfield_model_at_large_radius():
    faces, bounds, _ = dipole_faces(28)
    theta, phi = torch.linspace(.2, 2.9, 5), torch.linspace(0, 2*math.pi, 7)[:-1]
    differences, analytic = [], []
    for radius in (200., 2000.):
        points = spherical_points(theta, phi, radius)
        near = project_nearzone(faces, points, bounds_um=bounds, refractive_index=1.3)
        far = farfield_at_points(faces, points, bounds_um=bounds, refractive_index=1.3)
        expected, _ = analytic_dipole(points)
        differences.append(float((near.fields - far).norm()/far.norm()))
        analytic.append(float((near.fields - expected).norm()/expected.norm()))
    # The far-field model omits the box-size phase k a^2/(2r) and the 1/(kr)
    # terms, so the model difference falls as 1/r while the near-zone result
    # stays at the count-28 quadrature floor.
    assert differences[0]/differences[1] > 8 and differences[1] < 4e-4
    assert max(analytic) < 1e-3


def test_observation_grids_and_cartesian_far_field_are_consistent():
    theta = torch.tensor([0., math.pi/3, math.pi], dtype=torch.float64)
    phi = torch.tensor([0., math.pi/2, math.pi, 1.5*math.pi], dtype=torch.float64)
    directions = spherical_directions(theta, phi)
    assert torch.equal(directions, spherical_directions(theta.float(), phi.float()).to(directions.dtype)) or \
        float((directions - spherical_directions(theta.float(), phi.float())).abs().max()) < 1e-6
    t, p = np.meshgrid(theta.numpy(), phi.numpy(), indexing='ij')
    expected = np.stack((np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)), -1).reshape(-1, 3)
    np.testing.assert_allclose(directions.numpy(), expected, atol=1e-15)
    sphere = spherical_points(theta, phi, 3., origin_um=(.1, -.2, .3))
    torch.testing.assert_close(sphere, torch.tensor([.1, -.2, .3], dtype=torch.float64) + 3*directions)
    grid = cartesian_plane_points('x', 2.5, [-1., 1.], [-.5, 0., .5], origin_um=(.1, 0., 0.))
    assert grid.shape == (6, 3) and torch.equal(grid[:, 0], torch.full((6,), 2.6, dtype=torch.float64))
    torch.testing.assert_close(grid[:, 1], torch.tensor([-1., -1., -1., 1., 1., 1.], dtype=torch.float64))
    torch.testing.assert_close(grid[:, 2], torch.tensor([-.5, 0., .5]*2, dtype=torch.float64))
    faces, bounds, _ = dipole_faces(12)
    origin = (.1, 0., 0.)
    at_points = farfield_at_points(faces, grid, bounds_um=bounds, refractive_index=1.3, phase_origin_um=origin)
    offset = grid - torch.tensor(origin, dtype=torch.float64)
    radius = offset.norm(dim=-1)
    far = project_farfield(faces, offset/radius[:, None], bounds_um=bounds, refractive_index=1.3, phase_origin_um=origin)
    torch.testing.assert_close(at_points, far.fields_at_radius(radius*1e-6), rtol=0, atol=0)
    same = far.fields_at_radius(float(radius[0])*1e-6)
    torch.testing.assert_close(at_points[:, 0], same[:, 0], rtol=1e-12, atol=0)
    electric = at_points[0, :, :3]
    longitudinal = (electric*offset.to(electric.dtype)).sum(-1).abs() / (electric.norm(dim=-1)*radius)
    assert at_points.shape == (1, 6, 6) and float(longitudinal.max()) < 1e-12
    # Radial per-point far fields at one sphere reduce to the scalar radius form.
    points = spherical_points(theta, phi, 400.)
    sphere_fields = farfield_at_points(faces, points, bounds_um=bounds, refractive_index=1.3)
    torch.testing.assert_close(sphere_fields, project_farfield(faces, directions, bounds_um=bounds, refractive_index=1.3).fields_at_radius(400e-6), rtol=1e-12, atol=0)


def test_nearzone_and_observation_inputs_are_rejected_explicitly():
    faces, bounds, _ = dipole_faces(4)
    with pytest.raises(ValueError, match='strictly outside'):
        project_nearzone(faces, [[.2, .1, 0.], [3., 0., 0.]], bounds_um=bounds)
    with pytest.raises(ValueError, match='strictly outside'):
        farfield_at_points(faces, [[.65, 0., 0.]], bounds_um=bounds)
    with pytest.raises(ValueError, match='must be real'):
        project_nearzone(faces, torch.tensor([[3+1j, 0j, 0j]]), bounds_um=bounds)
    with pytest.raises(ValueError, match='fixed'):
        project_nearzone(faces, torch.tensor([[3., 0., 0.]], requires_grad=True), bounds_um=bounds)
    with pytest.raises(ValueError, match='chunk'):
        project_nearzone(faces, [[3., 0., 0.]], bounds_um=bounds, observation_chunk=0)
    with pytest.raises(ValueError, match='six'):
        project_nearzone({'x_min': faces['x_min']}, [[3., 0., 0.]], bounds_um=bounds)
    with pytest.raises(ValueError, match='phase origin'):
        farfield_at_points(faces, [[3., 0., 0.]], bounds_um=bounds, phase_origin_um=(3., 0., 0.))
    with pytest.raises(ValueError, match='Theta'):
        spherical_directions([3.2], [0.])
    with pytest.raises(ValueError, match='Radius'):
        spherical_points([1.], [0.], 0.)
    with pytest.raises(ValueError, match='normal'):
        cartesian_plane_points('w', 1., [0.], [0.])
    far = project_farfield(faces, [[1., 0, 0], [0, 1., 0]], bounds_um=bounds)
    with pytest.raises(ValueError, match='per direction'):
        far.fields_at_radius([1., 2., 3.])
    with pytest.raises(ValueError, match='per direction'):
        far.fields_at_radius(-1.)


@pytest.mark.parametrize('n,mu', [(1.3+.05j, 1.), (1.3+.05j, 1.1+.02j)])
def test_lossy_exterior_dipole_near_and_far_fields(n, mu):
    # Same closed forms with complex k = 2 pi n / wavelength; Im(k) = .2/um.
    points = torch.tensor([[1.2, .3, -.4], [0., -1.6, .2], [.5, .5, 1.3], [-2., .1, .1], [1., 1., 1.]], dtype=torch.float64)
    directions = torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [-1., 0, 0], [1., 2., 3.]], dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    expected, dipole = analytic_dipole(points, n=n, mu=mu)
    near_errors, far_errors = [], []
    for count in (14, 28, 56):
        faces, bounds, _ = dipole_faces(count, n=n, mu=mu)
        near = project_nearzone(faces, points, bounds_um=bounds, refractive_index=n, relative_permeability=mu)
        far = project_farfield(faces, directions, bounds_um=bounds, refractive_index=n, relative_permeability=mu)
        near_errors.append(float((near.fields - expected).norm()/expected.norm()))
        transverse = dipole - directions.to(dipole.dtype)*(directions.to(dipole.dtype)*dipole).sum(-1, keepdim=True)
        phase = torch.exp(-1j*(2*math.pi*n/1.55)*(directions @ torch.tensor([.12, -.08, .05], dtype=torch.float64)))
        amplitude = (2*math.pi/1.55)**2*mu*transverse*phase[:, None]*1e-6
        far_errors.append(float((far.electric_amplitude[0] - amplitude).norm()/amplitude.norm()))
    for errors in (near_errors, far_errors):
        assert errors[0]/errors[1] > 3.5 and errors[1]/errors[2] > 3.5 and errors[-1] < 5e-4
    assert near.fields.dtype == torch.complex128 and near.refractive_index == n and far.relative_permeability == mu
    remote = far.fields_at_radius(1e-3)
    e = remote[..., :3]
    torch.testing.assert_close(remote[..., 3:], (n/mu)*torch.linalg.cross(directions.to(e.dtype)[None].expand_as(e), e))
    torch.testing.assert_close(far.intensity(), .5*(n/mu).real*far.electric_amplitude.abs().square().sum(-1))
    # The finite-distance model still tends to the far-field model as 1/r while exp(ikr) decays.
    theta, phi = torch.linspace(.2, 2.9, 5), torch.linspace(0, 2*math.pi, 7)[:-1]
    differences = []
    for radius in (30., 60.):
        sphere = spherical_points(theta, phi, radius)
        nz = project_nearzone(faces, sphere, bounds_um=bounds, refractive_index=n, relative_permeability=mu)
        ff = farfield_at_points(faces, sphere, bounds_um=bounds, refractive_index=n, relative_permeability=mu)
        analytic, _ = analytic_dipole(sphere, n=n, mu=mu)
        differences.append(float((nz.fields - ff).norm()/ff.norm()))
        assert float((nz.fields - analytic).norm()/analytic.norm()) < 5e-4
    assert differences[0]/differences[1] > 1.8
    faces, bounds, _ = dipole_faces(28, n=n, mu=mu, dtype=torch.float32)
    single = project_nearzone(faces, points.float(), bounds_um=bounds, refractive_index=n, relative_permeability=mu)
    assert single.fields.dtype == torch.complex64
    assert float((single.fields - expected.to(torch.complex64)).norm()/expected.norm()) < 2e-3


def two_frequency_faces(count=10, dtype=torch.float64):
    """Two stored frequencies per face: the dipole fields and a rescaled conjugate copy."""
    faces, bounds, _ = dipole_faces(count, dtype=dtype)
    stacked = {}
    for name, face in faces.items():
        fields = torch.cat((face.fields, (.6 - .3j)*face.fields.conj()), 0)
        frequency = torch.cat((face.frequency_hz, 1.2*face.frequency_hz), 0)
        stacked[name] = replace(face, fields=fields, frequency_hz=frequency)
    return stacked, bounds


def frequency_slice(faces, index):
    return {name: replace(face, fields=face.fields[index:index+1], frequency_hz=face.frequency_hz[index:index+1])
            for name, face in faces.items()}


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_per_frequency_exterior_equals_single_frequency_calls(dtype):
    faces, bounds = two_frequency_faces(dtype=dtype)
    n = torch.tensor([1.3+.05j, 1.45+.05j], dtype=torch.complex128)  # Im(n^2/mu) > 0 at both frequencies
    mu = [1., 1.1+.02j]
    directions = torch.tensor([[1., 0, 0], [0, 0, 1.], [.6, 0, .8]], dtype=dtype)
    points = torch.tensor([[1.5, .2, .1], [0., 0., -2.]], dtype=dtype)
    tolerance = dict(rtol=1e-12, atol=0) if dtype == torch.float64 else dict(rtol=1e-5, atol=0)
    far = project_farfield(faces, directions, bounds_um=bounds, refractive_index=n, relative_permeability=mu)
    near = project_nearzone(faces, points, bounds_um=bounds, refractive_index=n, relative_permeability=mu)
    at = farfield_at_points(faces, points, bounds_um=bounds, refractive_index=n, relative_permeability=mu)
    assert far.electric_amplitude.shape == (2, 3, 3) and near.fields.shape == (2, 2, 6) and at.shape == (2, 2, 6)
    assert far.electric_amplitude.dtype == near.fields.dtype == at.dtype == faces['x_min'].fields.dtype
    assert torch.equal(far.refractive_index.cpu(), n) and near.relative_permeability.shape == (2,)
    for index in range(2):
        single = frequency_slice(faces, index)
        ni, mi = complex(n[index]), mu[index]
        one = project_farfield(single, directions, bounds_um=bounds, refractive_index=ni, relative_permeability=mi)
        torch.testing.assert_close(far.electric_amplitude[index], one.electric_amplitude[0], **tolerance)
        torch.testing.assert_close(far.intensity()[index], one.intensity()[0], **tolerance)
        torch.testing.assert_close(far.fields_at_radius(2e-3)[index], one.fields_at_radius(2e-3)[0], **tolerance)
        torch.testing.assert_close(near.fields[index], project_nearzone(single, points, bounds_um=bounds, refractive_index=ni, relative_permeability=mi).fields[0], **tolerance)
        torch.testing.assert_close(at[index], farfield_at_points(single, points, bounds_um=bounds, refractive_index=ni, relative_permeability=mi)[0], **tolerance)
    # A real per-frequency array keeps a real wavenumber and a real result index.
    real = project_farfield(faces, directions, bounds_um=bounds, refractive_index=[1.3, 1.45])
    assert not real.refractive_index.is_complex() and real.electric_amplitude.dtype == far.electric_amplitude.dtype
    torch.testing.assert_close(real.electric_amplitude[1], project_farfield(frequency_slice(faces, 1), directions, bounds_um=bounds, refractive_index=1.45).electric_amplitude[0], **tolerance)
    with pytest.raises(ValueError, match='growing exterior'):
        project_farfield(faces, directions, bounds_um=bounds, refractive_index=[1.3, 1.3-.05j])
    with pytest.raises(ValueError, match='not passive'):
        project_nearzone(faces, points, bounds_um=bounds, refractive_index=[1.3, 1.3], relative_permeability=[1., 1.1+.02j])
    with pytest.raises(ValueError, match=r'shape \(2,\)'):
        project_farfield(faces, directions, bounds_um=bounds, refractive_index=[1.3, 1.4, 1.5])
    with pytest.raises(ValueError, match=r'shape \(2,\)'):
        project_farfield(faces, directions, bounds_um=bounds, relative_permeability=torch.ones(2, 1))
    with pytest.raises(ValueError, match='design variable'):
        project_farfield(faces, directions, bounds_um=bounds, refractive_index=torch.tensor([1.3, 1.4], requires_grad=True))


def test_kspace_directions_match_spherical_directions():
    ux, uy = torch.tensor([-.6, 0., .3, .8]), torch.tensor([0., .5, -.25])
    directions = kspace_directions(ux, uy, 'z')
    assert directions.shape == (12, 3) and directions.dtype == torch.float64
    torch.testing.assert_close(directions.norm(dim=-1), torch.ones(12, dtype=torch.float64))
    assert bool((directions[:, 2] >= 0).all())
    ug, vg = torch.meshgrid(ux.double(), uy.double(), indexing='ij')
    torch.testing.assert_close(directions[:, 0], ug.reshape(-1))
    torch.testing.assert_close(directions[:, 1], vg.reshape(-1))
    for row, u, v in zip(directions, ug.reshape(-1).tolist(), vg.reshape(-1).tolist()):
        theta, phi = math.asin(math.hypot(u, v)), math.atan2(v, u)
        torch.testing.assert_close(row, spherical_directions([theta], [phi])[0])
    lower = kspace_directions(ux, uy, '-z')
    torch.testing.assert_close(lower[:, :2], directions[:, :2])
    torch.testing.assert_close(lower[:, 2], -directions[:, 2])
    sideways = kspace_directions(ux, uy, 'x')
    torch.testing.assert_close(sideways[:, 1], ug.reshape(-1))
    torch.testing.assert_close(sideways[:, 2], vg.reshape(-1))
    torch.testing.assert_close(sideways[:, 0], directions[:, 2])
    grazing = kspace_directions([1.], [0.], 'y')
    torch.testing.assert_close(grazing, torch.tensor([[0., 0., 1.]], dtype=torch.float64))
    faces, bounds, _ = dipole_faces(6)
    assert project_farfield(faces, kspace_directions(ux, uy), bounds_um=bounds, refractive_index=1.3).electric_amplitude.shape == (1, 12, 3)
    with pytest.raises(ValueError, match='evanescent'):
        kspace_directions([.8], [.7])
    with pytest.raises(ValueError, match='axis'):
        kspace_directions([0.], [0.], 'w')
    with pytest.raises(ValueError, match='finite'):
        kspace_directions([float('nan')], [0.])


def test_growing_or_per_frequency_exterior_is_rejected():
    faces, bounds, _ = dipole_faces(4)
    with pytest.raises(ValueError, match='growing exterior'):
        project_farfield(faces, [[1., 0, 0]], bounds_um=bounds, refractive_index=1.3-.05j)
    with pytest.raises(ValueError, match='growing exterior'):
        project_nearzone(faces, [[3., 0, 0]], bounds_um=bounds, relative_permeability=1.-.1j)
    with pytest.raises(ValueError, match='not passive'):
        project_farfield(faces, [[1., 0, 0]], bounds_um=bounds, refractive_index=1.3, relative_permeability=1.1+.02j)
    with pytest.raises(ValueError, match=r'shape \(1,\)'):
        project_farfield(faces, [[1., 0, 0]], bounds_um=bounds, refractive_index=torch.tensor([1.3, 1.4]))
    with pytest.raises(ValueError, match='positive real part'):
        project_farfield(faces, [[1., 0, 0]], bounds_um=bounds, refractive_index=.1j)
    with pytest.raises(ValueError, match='design variable'):
        project_farfield(faces, [[1., 0, 0]], bounds_um=bounds, refractive_index=torch.tensor(1.3, requires_grad=True))
    with pytest.raises(ValueError, match='lossless'):
        diffraction_orders(plane(), [(0, 0)], period_um=(2, 2), refractive_index=1.4+.1j)


def test_open_surface_converges_for_directional_emitter_and_not_for_dipole():
    # Fixed 8 x 8 x 6 wavelength box around a Gaussian Huygens sheet in z = 0.
    # Narrow waists diverge fast, so the beam still has a sizeable amplitude on
    # the side faces and the single z_max plane misses it; wider waists leave
    # negligible fields on the excluded faces and the plane result converges.
    directions = spherical_directions(torch.tensor([0., 10., 20., 30.])*math.pi/180, torch.tensor([0., 90., 180., 270.])*math.pi/180)
    points = torch.tensor([[0., 0., 3*1.55 + 4.], [1.55, -1.55, 3*1.55 + 3.]], dtype=torch.float64)
    single_errors, five_errors, near_errors, levels = [], [], [], []
    for waist in (.4*1.55, .6*1.55, .8*1.55):
        faces, bounds = beam_faces(waist, 64)
        closed = project_farfield(faces, directions, bounds_um=bounds)
        top = {'z_max': faces['z_max']}
        single = project_farfield(top, directions, bounds_um=bounds, open_surface=True)
        five = project_farfield({k: v for k, v in faces.items() if k != 'z_min'}, directions, bounds_um=bounds, open_surface=True)
        windowed = project_farfield(top, directions, bounds_um=bounds, open_surface=True, edge_window=(.2, .2))
        reference = closed.electric_amplitude.norm()
        single_errors.append(float((single.electric_amplitude - closed.electric_amplitude).norm()/reference))
        five_errors.append(float((five.electric_amplitude - closed.electric_amplitude).norm()/reference))
        assert float((windowed.electric_amplitude - closed.electric_amplitude).norm()/reference) < 3*single_errors[-1]
        near_closed = project_nearzone(faces, points, bounds_um=bounds)
        near_single = project_nearzone(top, points, bounds_um=bounds, open_surface=True)
        near_errors.append(float((near_single.fields - near_closed.fields).norm()/near_closed.fields.norm()))
        levels.append(max(float(v.fields.abs().max()/faces['z_max'].fields.abs().max()) for k, v in faces.items() if k != 'z_max'))
        assert closed.approximation is None and closed.surfaces == tuple(faces)
        assert single.approximation == near_single.approximation == 'open surface' and single.surfaces == near_single.surfaces == ('z_max',)
        assert five.surfaces == ('x_min', 'x_max', 'y_min', 'y_max', 'z_max')
    for errors in (single_errors, five_errors, near_errors, levels):
        assert errors[0] > errors[1] > errors[2]
    assert single_errors[0] > 3e-2 and single_errors[2] < 8e-3 and near_errors[2] < 3e-3
    assert all(five < single for five, single in zip(five_errors, single_errors))
    # A dipole leaves comparable fields on every face: the single plane is far
    # off and refining the face sampling does not help.
    dipole_errors = []
    for count in (16, 32):
        faces, bounds, _ = dipole_faces(count)
        closed = project_farfield(faces, directions, bounds_um=bounds, refractive_index=1.3)
        single = project_farfield({'z_max': faces['z_max']}, directions, bounds_um=bounds, refractive_index=1.3, open_surface=True)
        dipole_errors.append(float((single.electric_amplitude - closed.electric_amplitude).norm()/closed.electric_amplitude.norm()))
    assert min(dipole_errors) > .5 and abs(dipole_errors[0] - dipole_errors[1]) < .05
    with pytest.raises(ValueError, match='one to five'):
        project_farfield(faces, directions, bounds_um=bounds, refractive_index=1.3, open_surface=True)
    with pytest.raises(ValueError, match='six'):
        project_farfield({'z_max': faces['z_max']}, directions, bounds_um=bounds, refractive_index=1.3)
    with pytest.raises(ValueError, match='single open plane'):
        project_farfield({'z_max': faces['z_max'], 'z_min': faces['z_min']}, directions, bounds_um=bounds, refractive_index=1.3, open_surface=True, edge_window=(.1, 0.))
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        project_nearzone({'z_max': faces['z_max']}, points, bounds_um=bounds, refractive_index=1.3, open_surface=True, edge_window=(1.5, 0.))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_nearzone_objective_preserves_field_graph(dtype):
    a = torch.tensor(1.7, dtype=dtype, requires_grad=True)
    faces, bounds, _ = dipole_faces(10, dtype=dtype, amplitude=a)
    near = project_nearzone(faces, torch.tensor([[1.5, .2, .1], [0., 0., -2.]], dtype=dtype), bounds_um=bounds, refractive_index=1.3)
    objective = near.poynting().norm(dim=-1).sum()
    gradient, = torch.autograd.grad(objective, a)
    torch.testing.assert_close(gradient, 2*objective.detach()/a.detach(), rtol=2e-6 if dtype == torch.float32 else 1e-12, atol=0)


def test_native_fdtd_far_and_nearzone_objective_material_vjp_matches_difference():
    from torchfdtd import FieldMonitor, DifferentiablePlaneSimulation, AdjointOptions
    from test_differentiable import project
    p = project('3d', steps=35)
    p.monitors = []
    bounds = ((-.3, .3),)*3
    for d in range(3):
        for side in (0, 1):
            center, size = [0.]*3, [.6]*3
            center[d], size[d] = bounds[d][side], 0.
            p.monitors.append(FieldMonitor(id='xyz'[d]+('_min' if side == 0 else '_max'),
                center=tuple(center), size=tuple(size), normal='xyz'[d]))
    model = DifferentiablePlaneSimulation(p, AdjointOptions(checkpoints=2),
        quadrature_counts={m.id: (5, 5) for m in p.monitors})
    base = torch.ones(p.region.shape, dtype=torch.float64)
    mask = torch.zeros_like(base)
    mask[9:11, 6:9, 6:9] = 1
    frequency = torch.tensor([C0/1.1e-6], dtype=torch.float64)
    points = [[.5, .1, -.1], [-.4, .3, .2]]
    def objective(parameter):
        faces = model(base + parameter*mask, frequency)
        faces = {key: replace(face, fields=face.fields/p.region.time_step) for key, face in faces.items()}
        far = project_farfield(faces, [[1., 0, 0], [0., 1, 0]], bounds_um=bounds)
        near = project_nearzone(faces, points, bounds_um=bounds)
        return torch.stack((far.intensity().sum()*1e12, near.poynting()[..., 0].sum()))
    x = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    value = objective(x)
    gradient = torch.stack([torch.autograd.grad(v, x, retain_graph=True)[0] for v in value])
    h = 1e-4
    difference = (objective(x.detach()+h) - objective(x.detach()-h))/(2*h)
    assert bool((gradient.abs() > 1e-8).all())
    torch.testing.assert_close(gradient, difference, rtol=2e-6, atol=1e-10)
