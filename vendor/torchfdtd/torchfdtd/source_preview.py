"""Preview exactly the temporal samples passed to the native field updates.

The preview also reports the realized spatial amplitude and phase on the source
cells, the polarization vector, the effective bandwidth of the waveform and the
incidence definition of a plane source (docs/SOURCES.md, docs/CONVENTIONS.md):
every oblique source of the solver is a fixed-k_parallel (Bloch) source, and a
request for a fixed-angle source is refused with the registry's message.
"""
import math
import numpy as np

from .solver import source_time_signal, source_slice, source_profile, field_axes, C0
from .waveforms import pulse_parameters
from .capabilities import INCIDENCE

BANDWIDTH_FRACTION = .01   # amplitude spectrum above 1 percent of its peak (-40 dB)
# The preview keeps every sample, so a longer run is previewed over its first PREVIEW_STEPS steps: its
# arrays, its JSON and the per-step incident line of a TFSF or one-way source stay bounded.
PREVIEW_STEPS = 100_000


def effective_bandwidth(frequency_hz, spectrum, *, fraction=BANDWIDTH_FRACTION):
    """The outermost frequencies whose single-sided amplitude exceeds ``fraction`` of the peak.

    Declared definition: |X(f)| >= fraction * max |X| over the DFT bins of the
    whole simulation interval, so the resolution is 1 / (steps dt). A zero
    spectrum (a disabled source) has no bandwidth and says why.
    """
    frequency = np.asarray(frequency_hz, dtype=float)
    magnitude = np.asarray(spectrum, dtype=float)
    definition = (f'band where the single-sided amplitude spectrum of the injected samples is at least {fraction:g} of its peak '
                  f'({20*math.log10(fraction):.0f} dB), evaluated on the unwindowed DFT bins of the whole simulation interval')
    resolution = float(frequency[1]-frequency[0]) if len(frequency) > 1 else None
    if not len(magnitude) or not np.isfinite(magnitude).all() or magnitude.max() <= 0:
        return dict(definition=definition, threshold_fraction=fraction, frequency_resolution_hz=resolution,
                    frequency_hz=None, wavelength_um=None, peak_frequency_hz=None,
                    reason='the injected samples are zero (disabled source or zero amplitude): no bandwidth')
    above = np.flatnonzero(magnitude >= fraction*magnitude.max())
    low, high = float(frequency[above[0]]), float(frequency[above[-1]])
    peak = float(frequency[int(np.argmax(magnitude))])
    return dict(definition=definition, threshold_fraction=fraction, frequency_resolution_hz=resolution,
                frequency_hz=[low, high], wavelength_um=[C0/high*1e6, C0/low*1e6], peak_frequency_hz=peak,
                center_wavelength_um=C0/peak*1e6, reason=None)


def _positions(region, component, axis, part):
    """Physical coordinates (um) of the source cells along one axis, as the solver places them."""
    if region.material_sampling == 'yee':
        coords = field_axes(region, component)[axis]
    else:
        n, span = region.shape[axis], region.actual_size[axis]
        coords = np.array([0.]) if n == 1 else (np.arange(n)+.5)*region.mesh-span/2
    return coords[part]


def _spatial_terms(project, source):
    """Amplitude and phase on the cells of every soft or one-way term of the source."""
    region = project.region
    terms = []
    if source.kind == 'tfsf':
        from .tfsf import tfsf_metadata
        box = tfsf_metadata(source, region)
        for component, weight in source.polarization_components:
            terms.append(dict(component=component, weight=float(weight), amplitude=abs(source.amplitude*weight),
                              kind='tfsf box', cells=[[lo, hi] for lo, hi in zip(box['lower_node_indices'], box['upper_node_indices'])],
                              lower_um=box['lower_um'], upper_um=box['upper_um'], phase_rad_by_axis={},
                              profile='sparse E/H corrections on the six faces of the staggered box, driven by one scalar incident Yee line; '
                                      'uniform incident amplitude over the box, no spatial phase (normal incidence)'))
        return terms
    if source.injection == 'oneway':
        from .injection import oneway_metadata
        plane = oneway_metadata(source, region)
        axis = 'xyz'.index(source.normal)
        for component, weight in source.polarization_components:
            cells = [[0, region.shape[a]] for a in range(3)]
            cells[axis] = [plane['electric_index'], plane['electric_index']+1]
            positions = {a: _positions(region, component, a, slice(*cells[a])).tolist() for a in range(3) if region.shape[a] > 1}
            terms.append(dict(component=component, weight=float(weight), amplitude=abs(source.amplitude*weight),
                              kind='one-way plane', cells=cells, positions_um={'xyz'[a]: v for a, v in positions.items()},
                              phase_rad_by_axis={}, magnetic_index=plane['magnetic_index'],
                              profile='paired E/H corrections on one Yee cut over the whole transverse cell; uniform amplitude, '
                                      'no spatial phase (normal incidence, zero Bloch phase required)'))
        return terms
    for component, weight in source.polarization_components:
        scalar = source.model_copy(update={'component': component, 'theta': None})
        loc = source_slice(scalar, region)
        parts = [slice(i, i+1) if isinstance(i, (int, np.integer)) else i for i in loc]
        profile = source_profile(scalar, tuple(parts), region)
        phase = {}
        if profile is not None:
            # The Bloch profile is separable by construction (solver.source_profile adds one term per axis),
            # so the line along each axis at the first cell of the others carries that axis's phase up to a
            # constant; it is reported relative to the first cell of the line.
            for a in range(3):
                if profile.shape[a] > 1:
                    index = [0, 0, 0]; index[a] = slice(None)
                    line = np.unwrap(np.angle(profile[tuple(index)]))
                    phase['xyz'[a]] = (line-line[0]).tolist()
        cells = [[p.start, p.stop] for p in parts]
        positions = {'xyz'[a]: _positions(region, component, a, parts[a]).tolist() for a in range(3) if region.shape[a] > 1 or a < 2}
        terms.append(dict(component=component, weight=float(weight), amplitude=abs(source.amplitude*weight),
                          kind='point' if source.kind == 'point' else 'soft sheet', cells=cells, positions_um=positions,
                          phase_rad_by_axis=phase, phase_reference='relative to the first cell of each axis line',
                          profile=('single Yee sample' if source.kind == 'point' else
                                   'uniform amplitude over the sheet cells' +
                                   ('; Bloch spatial phase exp(+i phi (x - x0) / L) about the sheet centre x0 on each Bloch axis'
                                    if phase else '; no spatial phase'))))
    return terms


def _incidence(project, source, band, requested):
    """The incidence definition of a plane source and its angle range across the band."""
    if requested not in (None, 'fixed_k_parallel', 'fixed_angle'):
        raise ValueError('incidence must be None, "fixed_k_parallel" or "fixed_angle".')
    if requested == 'fixed_angle':
        raise ValueError(INCIDENCE['fixed_angle']['message'])
    region = project.region
    if source.kind != 'plane':
        return dict(kind='not applicable', statement=f'{source.kind} sources have no incidence angle.', fixed_angle_supported=False)
    active = 2 if region.dimension == '2d' else 3
    # A soft sheet is oriented by its zero span; one-way planes and TFSF boxes carry an explicit normal.
    zero = [a for a in range(active) if source.size[a] == 0]
    normal = zero[0] if len(zero) == 1 and source.injection == 'soft' else 'xyz'.index(source.normal)
    k_parallel = {}
    for a in range(active):
        if a != normal and region.boundaries.pair(a)[0].kind == 'bloch' and region.bloch_phase[a] != 0:
            k_parallel['xyz'[a]] = region.bloch_phase[a]/region.actual_size[a]   # rad per um
    if not k_parallel:
        kind = 'normal'
        return dict(kind=kind, statement=INCIDENCE[kind]['description'], k_parallel_rad_per_um={}, angle_deg={},
                    fixed_angle_supported=False, code_path=INCIDENCE[kind]['code_path'])
    kind = 'fixed_k_parallel'
    magnitude = math.sqrt(sum(v*v for v in k_parallel.values()))
    n = region.background_index
    angles = {}
    samples = {'carrier': C0/(source.wavelength*1e-6)} if source.pulse != 'sampled' else {}
    if band['frequency_hz'] is not None:
        samples.update(band_low=band['frequency_hz'][0], band_peak=band['peak_frequency_hz'], band_high=band['frequency_hz'][1])
    for name, f in samples.items():
        ratio = magnitude/(n*2*math.pi*f/C0*1e-6)
        angles[name] = dict(frequency_hz=f, wavelength_um=C0/f*1e6,
                            angle_deg=math.degrees(math.asin(ratio)) if ratio <= 1 else None,
                            reason=None if ratio <= 1 else 'k_parallel exceeds n k0: evanescent, no propagation angle')
    return dict(kind=kind, statement=INCIDENCE[kind]['description'], k_parallel_rad_per_um=k_parallel,
                k_parallel_magnitude_rad_per_um=magnitude, exterior_index=n, angle_deg=angles,
                fixed_angle_supported=False, fixed_angle_message=INCIDENCE['fixed_angle']['message'],
                code_path=INCIDENCE[kind]['code_path'])


def preview_source(project, source_id, *, incidence=None):
    """Waveform, spectrum, spatial profile, polarization, bandwidth and incidence of one source.

    ``incidence`` may name the definition the caller expects: ``'fixed_k_parallel'``
    is what a Bloch cell realizes; ``'fixed_angle'`` is refused with the registry
    message (``torchfdtd.capabilities.INCIDENCE``) because no such source exists.
    """
    original = next((s for s in project.sources if s.id == source_id), None)
    if original is None:raise ValueError('Source not found in this project')
    steps = project.region.steps
    if steps > PREVIEW_STEPS:
        project = project.model_copy(update={'region': project.region.model_copy(update={'steps': PREVIEW_STEPS})})
    source = project.resolved_source(original)
    region = project.region
    times = np.arange(1, region.steps+1)*region.time_step+source.time_offset_steps*region.time_step
    values = source_time_signal(source, times).astype(region.precision)
    injections = []
    plane = None
    box=None
    if source.kind=='tfsf' and source.enabled:
        from .tfsf import incident_preview,tfsf_metadata
        values,h,far=incident_preview(source,region)
        box=tfsf_metadata(source,region)
        injections=[dict(field='incident E at entry',signal=values.tolist(),time_fs=(times*1e15).tolist()),
                    dict(field='incident H at entry',signal=h.tolist(),time_fs=((times+.5*region.time_step)*1e15).tolist()),
                    dict(field='incident E at exit',signal=far.tolist(),time_fs=(times*1e15).tolist())]
    elif source.injection == 'oneway' and source.enabled:
        from .injection import oneway_tables, source_terms, oneway_metadata
        plane = oneway_metadata(source,region)
        values = oneway_tables(source, region)[0].astype(region.precision)
        for field, loc, increment, _ in source_terms(project, source):
            offset = .5 if field.startswith('H') else 0.
            injections.append(dict(field=field,
                time_fs=((np.arange(1, region.steps+1)*region.time_step+offset*region.time_step)*1e15).tolist(),
                signal=increment.astype(region.precision).tolist(),
                slice_indices=[[part.start,part.stop] for part in loc]))
    if not source.enabled:values[:] = 0
    # Single-sided amplitude, no window, no source-power normalization.
    transform = np.fft.rfft(values)/len(values)
    transform[1:-1 if len(values)%2 == 0 else None] *= 2
    frequencies = np.fft.rfftfreq(len(values), region.time_step)
    spectrum = abs(transform[1:])
    bandwidth = effective_bandwidth(frequencies[1:], spectrum)
    declared = ([source.wavelength_start, source.wavelength_stop] if source.time_definition in ('wavelength', 'frequency') else None)
    spatial = _spatial_terms(project, source) if source.enabled else []
    family = source.component[0]
    vector = [0., 0., 0.]
    for component, weight in source.polarization_components:
        vector['xyz'.index(component[1].lower())] = float(weight)
    polarization = dict(family='electric' if family == 'E' else 'magnetic', vector=vector, components=dict(source.polarization_components),
                        theta_deg=source.theta, phi_deg=source.phi if source.theta is not None else None,
                        definition='unit vector (sin theta cos phi, sin theta sin phi, cos theta) in the E or H family; '
                                   'None theta keeps the single Cartesian component')
    # Keep every actual sample: downsampling oscillations can hide aliasing.
    return dict(name=source.name, id=source.id, enabled=source.enabled,
                inherited=original.use_global_source, dt_fs=region.time_step*1e15, steps=steps, preview_steps=region.steps,
                pulse_parameters=pulse_parameters(source).as_dict() if source.pulse in ('gaussian','broadband') else None,
                time_fs=(times*1e15).tolist(), signal=values.astype(float).tolist(),
                field_family=source.component[0],time_offset_steps=source.time_offset_steps,
                injection=source.injection, injections=injections, plane=plane,tfsf=box,
                polarization_components=dict(source.polarization_components),
                polarization=polarization, spatial=spatial,
                bandwidth=dict(bandwidth, declared_wavelength_um=declared),
                incidence=_incidence(project, source, bandwidth, incidence),
                frequency_thz=(frequencies[1:]*1e-12).tolist(),
                wavelength_um=(299792458/frequencies[1:]*1e6).tolist(),
                spectrum=spectrum.tolist(), spectrum_units='reduced injection',
                spectrum_settings={'sampling':'fft','apodization':'none'},
                time_label='Time (fs) · temporal injection (reduced units)',
                note=('TFSF incident E at the entry face before polarization weights. The preview also includes canonical incident H and exit-face E. '
                      'These are the live auxiliary-line fields, not the individual face correction amplitudes. '
                      'The eight-cell delay and numerical dispersion are retained. Amplitude scales the soft drive, not calibrated incident power.' if source.kind=='tfsf' else
                      'One-way E correction before polarization weights. Paired E/H correction arrays are included in this preview. '
                      'The incident Yee line retains an eight-cell propagation delay and discrete dispersion. '
                      'Amplitude scales its soft drive, not a calibrated incident power.' if source.injection=='oneway' else
                      'Actual mesh-time samples, including local amplitude and phase. '
                      'Magnetic injection uses the H half-step time. The plotted envelope precedes signed vector weights. '
                      'The sheet spatial phase is listed per axis under spatial. FFT uses the full simulation interval without a window.')
                     +(f' The preview covers the first {region.steps:,} of the {steps:,} steps; the FFT and the bandwidth use those steps.'
                       if steps > region.steps else ''))
