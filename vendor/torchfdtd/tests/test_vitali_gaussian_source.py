"""LEG 48/49: CPU-only source API controls; no solver construction or stepping.

All numbers below are synthetic mathematical test inputs, not Vitali parameters.
The scalar phasor is tested independently via a complex beam parameter. Line
power means integral |E|^2 dt in micrometres for a unit waist field; it is not
the incident power of an FDTD run or evidence of paper-equivalent fibre input.
"""
import cmath
import math

import numpy as np
import pytest

from torchfdtd.models import Boundaries, Project, Region, Source
from torchfdtd.solver import field_axes, gaussian_source_profile, source_profile, source_slice


@pytest.fixture(autouse=True)
def no_simulation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('LEG 48 must not construct or advance a solver or initialize a device')

    monkeypatch.setattr('torchfdtd.solver.Simulation.__init__', forbidden)
    monkeypatch.setattr('torchfdtd.solver.Simulation.run', forbidden)
    monkeypatch.setattr('torchfdtd.solver.YeeGrid.__init__', forbidden)
    monkeypatch.setattr('fdtd.Grid.__init__', forbidden)
    monkeypatch.setattr('torch.cuda.init', forbidden)
    monkeypatch.setattr('torch.cuda._lazy_init', forbidden)


def source(**patch):
    data = dict(id='analytic-source', kind='plane', injection='soft', normal='x',
                direction='+', center=(0.5, -0.2, 0), size=(0, 2.0, 0), wavelength=1.1,
                gaussian=dict(waist_radius=0.7, waist_center=(-0.4, 0.1, 0),
                              tilt_deg=23, refractive_index=1.3))
    data.update(patch)
    return Source(**data)


def analytic_field(s, coordinate):
    """Independent complex-q formula at the declared reference plane.

    A rotation matrix maps the laboratory displacement onto the beam axes.
    (1 + i*z/zR)^(-1/2) supplies spreading and the 2-D Gouy phase, while
    exp[-u^2/(w0^2*(1+i*z/zR))] supplies decay and wavefront curvature.
    """
    b = s.gaussian
    a = 'xy'.index(s.normal)
    displacement = np.array([s.center[a]-b.waist_center[a], coordinate-b.waist_center[1-a]])
    angle = b.tilt_deg*math.pi/180
    sign = {'+': 1, '-': -1}[s.direction]
    rotation = np.array([[sign*math.cos(angle), math.sin(angle)],
                         [-sign*math.sin(angle), math.cos(angle)]])
    z, u = rotation @ displacement
    z_rayleigh = math.pi*b.refractive_index*b.waist_radius**2/s.wavelength
    q = complex(1, z/z_rayleigh)
    wave = cmath.exp(2j*math.pi*b.refractive_index*z/s.wavelength)
    return wave/cmath.sqrt(q)*cmath.exp(-u*u/(b.waist_radius**2*q))


@pytest.mark.parametrize('normal', ['x', 'y'])
@pytest.mark.parametrize('direction', ['+', '-'])
@pytest.mark.parametrize('tilt', [0, -31, 23])
def test_complex_profile_matches_independent_rotated_beam(normal, direction, tilt):
    s = source(normal=normal, direction=direction,
               size=(0, 2, 0) if normal == 'x' else (2, 0, 0),
               gaussian=dict(waist_radius=0.7, waist_center=(-0.4, 0.1, 0),
                             tilt_deg=tilt, refractive_index=1.3))
    axis = 1-'xy'.index(normal)
    coordinates = s.center[axis]+np.array([-0.9, -0.35, 0, 0.15, 0.8])
    expected = np.array([analytic_field(s, value) for value in coordinates])
    actual = gaussian_source_profile(s, coordinates)
    assert np.iscomplexobj(actual)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)


def test_declared_waist_and_reference_are_explicit_and_aperture_is_finite():
    s = source(center=(0.5, 0.1, 0),
               gaussian=dict(waist_radius=0.7, waist_center=(0.5, 0.1, 0),
                             tilt_deg=0, refractive_index=1.3))
    values = gaussian_source_profile(s, [0.1, 0.8, -1.01, 1.21])
    np.testing.assert_allclose(values[:2], [1, math.exp(-1)], rtol=2e-15, atol=0)
    np.testing.assert_array_equal(values[2:], [0j, 0j])
    moved = s.model_copy(update={'center': (0.9, 0.1, 0)})
    assert gaussian_source_profile(moved, [0.1])[0] != values[0]
    translated = source(center=(1.0, 0.6, 0),
                        gaussian=dict(waist_radius=0.7, waist_center=(0.1, 0.9, 0),
                                      tilt_deg=23, refractive_index=1.3))
    np.testing.assert_allclose(gaussian_source_profile(source(), [-0.5, 0.2]),
                               gaussian_source_profile(translated, [0.3, 1.0]), rtol=3e-14)


@pytest.mark.parametrize('direction', ['+', '-'])
@pytest.mark.parametrize('distance', [0, 1.7, -2.3])
def test_independent_finite_aperture_gaussian_power_integral(direction, distance):
    """Closed-form erf control for an untilted beam at/off its waist.

    On this plane |E|^2 = (w0/w)*exp[-2*(t-t0)^2/w^2]. Thus the integral
    over [a,b] is w0*sqrt(pi)/(2*sqrt(2)) times the difference of the two
    erf values. The bounds deliberately clip appreciable power asymmetrically.
    Gauss-Legendre quadrature of the production profile is the measured side;
    the expected side uses only scalar math and the analytic antiderivative.
    """
    w0, wavelength, index, t0 = 0.7, 1.1, 1.3, 0.1
    a, b = -0.8, 0.4
    z_rayleigh = math.pi*index*w0*w0/wavelength
    width = w0*math.sqrt(1+(distance/z_rayleigh)**2)
    expected = w0*math.sqrt(math.pi)/(2*math.sqrt(2))*(
        math.erf(math.sqrt(2)*(b-t0)/width)-math.erf(math.sqrt(2)*(a-t0)/width))
    s = source(direction=direction, center=(distance, (a+b)/2, 0), size=(0, b-a, 0),
               gaussian=dict(waist_radius=w0, waist_center=(0, t0, 0),
                             tilt_deg=0, refractive_index=index))
    nodes, weights = np.polynomial.legendre.leggauss(128)
    coordinates = (a+b)/2+(b-a)/2*nodes
    intensity = np.abs(gaussian_source_profile(s, coordinates))**2
    measured = (b-a)/2*np.dot(weights, intensity)
    assert measured == pytest.approx(expected, rel=5e-14, abs=0)
    infinite_aperture = w0*math.sqrt(math.pi/2)
    assert expected < 0.9*infinite_aperture  # catches a missing finite aperture
    wrong_field_radius = w0*math.sqrt(math.pi)/2*(
        math.erf((b-t0)/width)-math.erf((a-t0)/width))
    assert abs(measured-wrong_field_radius) > 0.01*expected


def test_tilt_and_polarization_theta_are_independent():
    original = source(theta=0)
    polarized = source(theta=90, phi=0)
    assert original.polarization_components == (('Ez', 1.0),)
    assert polarized.polarization_components == (('Ex', 1.0),)
    coordinates = np.array([-0.7, -0.2, 0.3])
    np.testing.assert_array_equal(gaussian_source_profile(original, coordinates),
                                  gaussian_source_profile(polarized, coordinates))
    untilted = source(theta=0, gaussian={**original.gaussian.model_dump(), 'tilt_deg': 0})
    assert untilted.polarization_components == original.polarization_components
    assert not np.allclose(gaussian_source_profile(original, coordinates),
                           gaussian_source_profile(untilted, coordinates))


@pytest.mark.parametrize('key', ['center', 'size', 'normal', 'direction', 'wavelength'])
def test_reference_inputs_cannot_fall_back_to_legacy_defaults(key):
    data = source().model_dump()
    del data[key]
    with pytest.raises(ValueError, match='explicit'):
        Source(**data)


@pytest.mark.parametrize('key', ['waist_radius', 'waist_center', 'tilt_deg', 'refractive_index'])
def test_all_beam_inputs_are_required(key):
    data = source().model_dump()
    del data['gaussian'][key]
    with pytest.raises(ValueError):
        Source(**data)


@pytest.mark.parametrize('patch', [dict(waist_radius=0), dict(waist_radius=-1),
    dict(waist_radius=float('nan')), dict(waist_radius=float('inf')),
    dict(tilt_deg=-90), dict(tilt_deg=90), dict(tilt_deg=float('nan')),
    dict(refractive_index=0), dict(refractive_index=float('inf')),
    dict(waist_center=(0, float('nan'), 0)), dict(waist_center=(0, 0, 1))])
def test_invalid_beam_inputs_are_rejected(patch):
    with pytest.raises(ValueError):
        source(gaussian={**source().gaussian.model_dump(), **patch})


@pytest.mark.parametrize('patch', [dict(kind='point'), dict(kind='tfsf'), dict(injection='oneway'),
    dict(normal='z'), dict(size=(1, 2, 0)), dict(size=(0, 0, 0)), dict(size=(0, -1, 0)),
    dict(size=(0, 2, 1)), dict(size=(0, float('inf'), 0)), dict(center=(0, 0, 1)),
    dict(center=(float('nan'), 0, 0)), dict(extend_through_pml=True)])
def test_unsupported_sheet_inputs_are_rejected(patch):
    with pytest.raises(ValueError):
        source(**patch)


def region(**patch):
    return Region(**{**dict(size=(6, 4, 2), mesh=0.1, pml_cells=3,
                           backend='cpu', complex_source_fields=True), **patch})


@pytest.mark.parametrize('normal', ['x', 'y'])
@pytest.mark.parametrize('sampling', ['cell', 'yee'])
def test_profile_is_connected_to_sheet_sampling_without_constructing_solver(normal, sampling):
    s = source(normal=normal, size=(0, 2, 0) if normal == 'x' else (2, 0, 0))
    r = region(material_sampling=sampling)
    loc = source_slice(s, r)
    transverse = 1-'xy'.index(normal)
    coords = (field_axes(r, s.component)[transverse] if sampling == 'yee' else
              (np.arange(r.shape[transverse])+0.5)*r.mesh-r.actual_size[transverse]/2)[loc[transverse]]
    expected = np.array([analytic_field(s, coordinate)
                         if abs(coordinate-s.center[transverse]) <= s.size[transverse]/2 else 0j
                         for coordinate in coords])
    actual = source_profile(s, loc, r)
    shape = [1, 1, 1]
    shape[transverse] = len(coords)
    assert actual.shape == tuple(shape)
    np.testing.assert_allclose(actual.reshape(-1), expected, rtol=3e-14, atol=3e-15)
    p = Project(region=r, sources=[s])
    assert p.region.complex_fields
    assert Source.model_validate_json(s.model_dump_json()) == s


@pytest.mark.parametrize('patch', [dict(complex_source_fields=False), dict(dimension='3d'),
    dict(boundaries=Boundaries(y_min={'kind': 'bloch'}, y_max={'kind': 'bloch'}))])
def test_incompatible_regions_are_rejected_before_execution(patch):
    r = region(**patch)
    s = source()
    with pytest.raises(ValueError, match='Gaussian'):
        Project(region=r, sources=[s])
    with pytest.raises(ValueError, match='Gaussian'):
        source_profile(s, source_slice(s, r), r)


def test_exact_legacy_source_defaults_and_serialization():
    expected = dict(id='legacy', name='source', center=(0, 0, 0), enabled=True,
        wavelength=1.55, pulse='gaussian', pulse_cycles=3, time_definition='cycles',
        pulse_length=20e-15, pulse_offset=50e-15, signal=None, wavelength_start=1.3,
        wavelength_stop=1.8, optimize_for_short_pulse=True, eliminate_discontinuities=False,
        chirp_bandwidth_hz=100e12, kind='point', injection='soft', normal='x', direction='+',
        incident_pml_cells=96, size=(0, 2, 0), component='Ez', theta=None, phi=0,
        amplitude=1, phase=0, use_global_source=False, extend_through_pml=False)
    s = Source(id='legacy')
    assert s.model_dump() == expected
    assert s.gaussian is None
    assert Source(**expected).model_dump_json() == s.model_dump_json()
    r = Region()
    assert r.complex_source_fields is False and r.complex_fields is False
    assert 'complex_source_fields' not in r.model_dump()
    assert 'complex_source_fields' not in r.model_dump_json()
    assert source_profile(s, source_slice(s, r), r) is None
    plane = Source(kind='plane')
    assert source_profile(plane, source_slice(plane, r), r) is None


@pytest.mark.parametrize('sampling', ['cell', 'yee'])
def test_legacy_bloch_profile_is_bit_identical(sampling):
    r = Region(size=(6, 4, 2), mesh=0.1, material_sampling=sampling,
               boundaries=Boundaries(y_min={'kind': 'bloch'}, y_max={'kind': 'bloch'}),
               bloch_phase=(0, 0.8, 0))
    s = Source(kind='plane', center=(0.5, -0.2, 0), size=(0, 2, 0))
    loc = source_slice(s, r)
    # The unmodified spatial expression, with the same floating-point order.
    n, span = r.shape[1], r.actual_size[1]
    coords = (field_axes(r, s.component)[1] if sampling == 'yee' else
              (np.arange(n)+0.5)*r.mesh-span/2)[loc[1]]
    expected = np.exp(1j*(0.0+(r.bloch_phase[1]*(coords-s.center[1])/span).reshape(1, -1, 1)))
    np.testing.assert_array_equal(source_profile(s, loc, r), expected)


# LEG 49: explicit wavelength under inheritance and one signed-frequency convention.

def test_inherited_gaussian_wavelength_must_equal_global_exactly():
    from torchfdtd.models import SourceTimeSettings
    r = region()
    s = source(use_global_source=True)
    with pytest.raises(ValueError, match='explicit wavelength equals global_source.wavelength'):
        Project(region=r, sources=[s])  # default global_source.wavelength is 1.55, not 1.1
    p = Project(region=r, sources=[s], global_source=SourceTimeSettings(wavelength=1.1, pulse_cycles=5))
    resolved = p.resolved_source(s)
    assert resolved.wavelength == 1.1 and resolved.pulse_cycles == 5
    np.testing.assert_array_equal(gaussian_source_profile(resolved, [0.1]), gaussian_source_profile(s, [0.1]))
    # Non-Gaussian inheritance keeps replacing the local wavelength, unchanged.
    plain = Source(id='plain', kind='plane', center=(0.5, -0.2, 0), wavelength=1.1, use_global_source=True)
    assert Project(region=r, sources=[plain]).resolved_source(plain).wavelength == 1.55


def legacy_time_signal(source, times):
    """Verbatim pre-LEG-49 source_time_signal, for a bit-identity control."""
    from torchfdtd.waveforms import gaussian_envelope, pulse_parameters
    times = np.asarray(times, dtype=np.float64)
    if source.pulse == 'sampled':
        signal = source.signal
        amplitude = np.interp(times, signal.time_s, signal.amplitude, left=0, right=0)
        phase = np.interp(times, signal.time_s, signal.phase_rad)+math.radians(source.phase)
        return source.amplitude*amplitude*np.sin(phase)
    p = pulse_parameters(source); x = times-p.offset_s; u = x/p.sigma_s
    omega = 2*math.pi*p.frequency_hz; phase = math.radians(source.phase)
    if source.pulse == 'continuous':
        envelope = 1-np.exp(-(times/p.sigma_s)**2); carrier = omega*times
    else:
        envelope = gaussian_envelope(u, source.eliminate_discontinuities)
        carrier = omega*times if source.time_definition == 'cycles' else -omega*x
        if p.chirped:
            inside = p.frequency_span_hz*p.sigma_s*(4*np.cos(np.pi*u/4)-2*np.pi)
            outside = -np.pi*p.frequency_span_hz*np.abs(x)
            carrier = carrier+np.where(np.abs(u) <= 2, inside, outside)
    return source.amplitude*envelope*np.sin(carrier+phase)


@pytest.mark.parametrize('patch', [dict(), dict(phase=37, amplitude=2.5), dict(pulse='continuous'),
    dict(time_definition='standard'), dict(eliminate_discontinuities=True),
    dict(pulse='broadband', time_definition='wavelength'), dict(pulse='broadband', time_definition='standard'),
    dict(pulse='sampled', signal=dict(time_s=[0, 1e-14, 3e-14], amplitude=[0, 1, .2], phase_rad=[0, 2, 5]))])
def test_legacy_time_signal_is_bit_identical(patch):
    from torchfdtd.waveforms import source_time_signal
    s = Source(id='legacy', **patch)
    times = np.linspace(0, 2e-13, 4001)
    np.testing.assert_array_equal(source_time_signal(s, times), legacy_time_signal(s, times))


def test_non_gaussian_source_terms_are_unchanged():
    from torchfdtd.injection import source_terms
    r = region(complex_source_fields=False, steps=50)
    s = Source(id='plain', kind='plane', center=(0.5, -0.2, 0))
    terms = list(source_terms(Project(region=r, sources=[s]), s))
    assert len(terms) == 1 and terms[0][3] is None
    times = np.arange(1, r.steps+1)*r.time_step
    np.testing.assert_array_equal(terms[0][2], legacy_time_signal(s, times))


def test_gaussian_rejects_sampled_waveform_without_carrier_sign():
    with pytest.raises(ValueError, match='frequency sign'):
        source(pulse='sampled', signal=dict(time_s=[0, 1e-14], amplitude=[1, 1], phase_rad=[0, 1]))


def complex_drive(s, times):
    from torchfdtd.waveforms import source_quadrature_signal, source_time_signal
    return source_time_signal(s, times)+1j*source_quadrature_signal(s, times)


@pytest.mark.parametrize('patch', [dict(), dict(time_definition='standard', phase=50),
                                   dict(pulse='broadband', time_definition='wavelength',
                                        wavelength_start=1.0, wavelength_stop=1.2)])
def test_complex_drive_has_only_positive_physical_frequency(patch):
    """Independent FFT control of exp(-i*omega*t): numpy's kernel is exp(-2i*pi*f*t),
    so a positive physical frequency appears at numpy frequency -f0."""
    from torchfdtd.waveforms import C0, source_time_signal
    s = source(**patch)
    f0 = C0/(s.wavelength*1e-6)
    times = np.arange(0, 300e-15, 1/(40*f0))
    freqs = np.fft.fftfreq(len(times), times[1]-times[0])
    drive = np.abs(np.fft.fft(complex_drive(s, times)))**2
    real = np.abs(np.fft.fft(source_time_signal(s, times)))**2
    assert drive[freqs > 0].sum() < 1e-8*drive.sum()  # only the t=0 truncation of the legacy pulse leaks
    assert abs(real[freqs > 0].sum()/real.sum()-0.5) < 1e-6  # a real carrier holds both signs equally
    assert abs(abs(freqs[np.argmax(drive)])-f0) < 0.05*f0 and freqs[np.argmax(drive)] < 0
    np.testing.assert_array_equal(complex_drive(s, times).real, source_time_signal(s, times))


@pytest.mark.parametrize('normal', ['x', 'y'])
@pytest.mark.parametrize('direction', ['+', '-'])
@pytest.mark.parametrize('tilt', [-31, 23])
def test_space_time_spectrum_selects_requested_direction(normal, direction, tilt):
    """Fourier control on the reference sheet: the 2-D (k_t, omega) spectrum of
    profile(t_coord)*drive(t) peaks at k_t=+k*sin(tilt) with physical omega>0, and the
    mirrored tilt (k_t=-k*sin(tilt) at omega>0, i.e. +k_t at omega<0) is suppressed.
    The normal wavenumber sign follows direction from a reference-plane shift."""
    from torchfdtd.waveforms import C0, source_time_signal
    a = 'xy'.index(normal)
    wide = dict(waist_radius=8.0, waist_center=(0.5, -0.2, 0), tilt_deg=tilt, refractive_index=1.3)
    s = source(normal=normal, direction=direction, center=(0.5, -0.2, 0), gaussian=wide,
               size=(0, 60, 0) if normal == 'x' else (60, 0, 0))
    k = 2*math.pi*1.3/s.wavelength
    dy = s.wavelength/16
    coords = s.center[1-a]+(np.arange(1024)-512)*dy
    profile = gaussian_source_profile(s, coords)
    f0 = C0/(s.wavelength*1e-6)
    times = np.arange(0, 120e-15, 1/(16*f0))
    kt = 2*np.pi*np.fft.fftfreq(len(coords), dy)
    omega = -2*np.pi*np.fft.fftfreq(len(times), times[1]-times[0])  # physical, exp(-i*omega*t)
    spectrum = np.abs(np.fft.fft2(np.outer(profile, complex_drive(s, times))))**2
    i, j = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    assert omega[j] > 0
    assert abs(kt[i]-k*math.sin(math.radians(tilt))) < 2*np.pi/(len(coords)*dy)+1e-12
    mirrored = spectrum[:, omega < 0].sum()
    assert mirrored < 1e-8*spectrum.sum()
    real = np.abs(np.fft.fft2(np.outer(profile, source_time_signal(s, times))))**2
    assert abs(real[:, omega < 0].sum()/real.sum()-0.5) < 1e-6  # the LEG 48 ambiguity
    # Normal wavenumber: phase of the on-axis sample vs reference-plane position.
    h = 1e-4
    shifted = [list(s.center) for _ in range(2)]
    shifted[0][a] += h; shifted[1][a] -= h
    plus, minus = (gaussian_source_profile(s.model_copy(update={'center': tuple(c)}), [s.center[1-a]])[0]
                   for c in shifted)
    kn = np.angle(plus/minus)/(2*h)
    assert abs(kn-(1 if direction == '+' else -1)*k*math.cos(math.radians(tilt))) < 1e-3*k


def test_gaussian_source_terms_carry_the_complex_drive():
    from torchfdtd.injection import source_terms
    r = region(steps=80)
    s = source()
    terms = list(source_terms(Project(region=r, sources=[s]), s))
    assert len(terms) == 2 and terms[0][1] == terms[1][1]
    times = np.arange(1, r.steps+1)*r.time_step
    combined = np.multiply.outer(terms[0][2], terms[0][3])+np.multiply.outer(terms[1][2], terms[1][3])
    expected = np.multiply.outer(complex_drive(s, times), source_profile(s, source_slice(s, r), r))
    np.testing.assert_allclose(combined, expected, rtol=1e-14, atol=1e-15)
