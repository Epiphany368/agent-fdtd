"""Immutable resolved simulation plan shared by every public entry point.

``resolve_plan(project)`` runs the existing resolvers once and freezes what
they return: the realized mesh nodes and Yee component coordinates, the time
step and half-step convention, the per-face boundary kinds with the CPML
coefficient arrays and Bloch phases, the ADE pole coefficients of every
dispersive material, the rasterization inputs of the sampled material, each
enabled source's realized support cells and sampled waveform, each monitor's
quadrature and interpolation weights and DFT settings, and the resource
estimate. Nothing here re-implements physics; the plan wraps ``mesh_nodes``,
``field_axes``, ``BoundaryDescription``, ``MaterialADE``, ``source_terms``,
``plane_plan``/``interpolation_map``, ``frequency_samples``, ``voxelize`` and
``estimate``.

``plan_hash`` is a SHA-256 over the canonical serialization of the physical
plan. Placement (backend, kernels, memory and execution mode, tiling,
precision) and the resource estimate, which depends on the precision, stay
outside the hash. The sampled material enters the hash through its
rasterization inputs (the enabled structures with their material parameters,
the background, the sampling mode and the interface method) on the hashed
nodes; the voxel arrays are a deterministic function of those inputs and are
materialized on demand through ``plan.material`` so that resolving a plan
for a large streamed scene does not allocate a full-volume array.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cached_property

import numpy as np

COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
# E is observed after its update at (n+1) dt and H half a step later.
SAMPLE_TIME_STEPS = {'E': 1., 'H': 1.5}
FOURIER_CONVENTION = 'exp(+2 pi i f t)'
PLACEMENT_FIELDS = ('backend', 'cuda_kernel', 'cuda_monitor_kernel', 'memory_mode',
                    'execution_mode', 'tiling', 'precision')
# Fields carried for lookups and display but left out of the canonical form:
# object ids and names, and the raw settings whose effect is already hashed
# through the sampled waveform, realized support, frequency samples and
# apodization they produce.
UNHASHED = {'hashed': False}
# Fields that exist only for some scenes: left out of the canonical and browser forms while None,
# so the hashes and JSON of every other scene stay what they were.
OPTIONAL = {'omit_none': True}


def _present(value, f):
    return not f.name.startswith('_') and not (f.metadata.get('omit_none') and getattr(value, f.name) is None)


class PlanInvalidated(ValueError):
    """The project behind a resolved plan changed after the plan was fixed."""


def _frozen(array):
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


def _canonical_array(array):
    array = np.asarray(array)
    if np.iscomplexobj(array):
        array = np.ascontiguousarray(array, dtype=np.complex128)
    elif array.dtype.kind == 'f':
        array = np.ascontiguousarray(array, dtype=np.float64)
    elif array.dtype.kind == 'b':
        array = np.ascontiguousarray(array, dtype=np.int64)
    else:
        array = np.ascontiguousarray(array, dtype=np.int64)
    return dict(dtype=str(array.dtype), shape=list(array.shape),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def _canonical(value):
    """JSON-able canonical form: arrays become dtype/shape/digest records."""
    if isinstance(value, np.ndarray):
        return _canonical_array(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _canonical(getattr(value, f.name)) for f in fields(value)
                if _present(value, f) and f.metadata.get('hashed', True)}
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, slice):
        return dict(slice=[value.start, value.stop])
    if isinstance(value, complex):
        return dict(real=float(value.real), imag=float(value.imag))
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f'Plan field of type {type(value).__name__} has no canonical form.')


def _json_value(value):
    """Browser form: one-dimensional arrays inline, larger arrays as digest records."""
    if isinstance(value, np.ndarray):
        if value.ndim <= 1:
            if np.iscomplexobj(value):
                return dict(real=value.real.tolist(), imag=value.imag.tolist())
            return value.tolist()
        return _canonical_array(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value) if _present(value, f)}
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return _canonical(value)


def _flatten(value, prefix, out):
    if isinstance(value, dict) and 'sha256' not in value:
        for k, v in value.items():
            _flatten(v, f'{prefix}.{k}' if prefix else str(k), out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _flatten(v, f'{prefix}[{i}]', out)
    else:
        out[prefix] = json.dumps(value, sort_keys=True)


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


@dataclass(frozen=True, eq=False)
class CPMLSegment:
    """One stretched-coordinate CPML segment of ``YeeGrid._prepare_boundaries``."""
    forward: bool
    axis: int
    component: int
    side: int
    start: int
    stop: int
    kappa: np.ndarray
    sigma: np.ndarray
    alpha: np.ndarray
    b: np.ndarray
    c: np.ndarray


@dataclass(frozen=True, eq=False)
class BoundaryPlan:
    faces: tuple
    bloch_phase: tuple
    courant_number: float
    wrap: dict
    pec_upper: dict
    pmc_lower: dict
    pmc_upper: dict
    cpml: tuple
    # Absorber faces, their per-axis (integer, half node) losses and the reference permittivity of each dispersive material.
    absorber: dict | None = field(default=None, metadata=OPTIONAL)


@dataclass(frozen=True, eq=False)
class ADEPlan:
    """Trapezoidal ADE coefficients of one dispersive material at the plan's dt."""
    material: str
    epsilon_inf: float
    oscillators: tuple
    a: np.ndarray
    d: np.ndarray
    k: np.ndarray


@dataclass(frozen=True, eq=False)
class SourceTerm:
    component: str
    location: tuple
    sample_times: np.ndarray
    samples: np.ndarray
    profile: np.ndarray | None


@dataclass(frozen=True, eq=False)
class SourcePlan:
    id: str = field(metadata=UNHASHED)
    name: str = field(metadata=UNHASHED)
    kind: str
    injection: str
    settings: dict = field(metadata=UNHASHED)
    polarization: tuple
    terms: tuple
    tfsf: dict | None
    tfsf_drive: np.ndarray | None
    oneway: dict | None


@dataclass(frozen=True, eq=False)
class MonitorPlan:
    id: str = field(metadata=UNHASHED)
    name: str = field(metadata=UNHASHED)
    kind: str
    components: tuple
    index: tuple | None
    normal: str | None
    points_um: np.ndarray | None
    weights: np.ndarray | None
    shape: tuple | None
    maps: tuple
    frequency_hz: np.ndarray | None
    apodization: dict
    time_downsample: int
    dft_precision: str | None
    spectrum: dict = field(metadata=UNHASHED)


@dataclass(frozen=True, eq=False)
class SampledMaterial:
    """Voxelized epsilon per Yee component and material ownership, as ``voxelize`` returns them."""
    epsilon: np.ndarray
    counts: dict
    ownership: np.ndarray
    interface: object


class _ADEDescription:
    """Coefficient-only stand-in for a grid, like ``BoundaryDescription``."""
    is_torch = False

    def __init__(self, time_step):
        self.time_step = time_step
        self.memory_states = []

    def _zeros(self, shape):
        return None

    def _coefficient(self, array):
        return np.asarray(array, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class SimulationPlan:
    dimension: str
    size: tuple
    shape: tuple
    nodes: tuple
    axes: dict
    time_step: float
    steps: int
    run_control: dict
    sample_time_steps: dict
    fourier_convention: str
    background_index: float
    material_sampling: str
    interface_method: str
    subpixel_quadrature: int
    boundaries: BoundaryPlan
    pml_dispersion: str
    structures: tuple
    materials: tuple
    ade: tuple
    sources: tuple
    monitors: tuple
    resources: dict
    placement: dict
    plan_hash: str
    _canonical: dict = field(repr=False, compare=False)
    _project: object = field(repr=False, compare=False)

    # ---- derived views ---------------------------------------------------------
    @cached_property
    def material(self):
        """The sampled material, materialized on first use with the same resolvers the solver runs."""
        from .solver import voxelize
        from .subpixel import prepare_interfaces
        interface = prepare_interfaces(self._project)
        epsilon, counts, ownership = voxelize(self._project, with_ownership=True, interface_plan=interface)
        return SampledMaterial(_frozen(epsilon), dict(counts), _frozen(ownership), interface)

    @property
    def source_terms(self):
        """(component, location, samples, profile) for every enabled soft/one-way term, in solver order."""
        return tuple((term.component, term.location, term.samples, term.profile)
                     for source in self.sources for term in source.terms)

    @property
    def sections(self):
        """A copy of the canonical serialization by section, the input of the identity conditions."""
        return copy.deepcopy(self._canonical)

    # ---- public API ------------------------------------------------------------
    def to_json(self):
        payload = {name: _json_value(getattr(self, name)) for name in
                   ('dimension', 'size', 'shape', 'nodes', 'axes', 'time_step', 'steps', 'run_control', 'sample_time_steps',
                    'fourier_convention', 'background_index', 'material_sampling', 'interface_method',
                    'subpixel_quadrature', 'boundaries', 'pml_dispersion', 'structures', 'materials', 'ade', 'sources',
                    'monitors', 'placement')}
        payload['resources'] = json.loads(json.dumps(self.resources, default=str))
        payload['plan_hash'] = self.plan_hash
        return payload

    def diff(self, other):
        """Dotted keys of the canonical plan and placement that differ from ``other``."""
        if not isinstance(other, SimulationPlan):
            raise TypeError('diff expects another SimulationPlan.')
        mine, theirs = {}, {}
        _flatten(dict(self._canonical, placement=_canonical(self.placement)), '', mine)
        _flatten(dict(other._canonical, placement=_canonical(other.placement)), '', theirs)
        return sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))

    def verify_grid(self, grid):
        """Raise PlanInvalidated when a prepared grid's time base or CPML/Bloch coefficients differ from the plan."""
        def host(value):
            return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
        if not np.isclose(grid.time_step, self.time_step, rtol=0, atol=0):
            raise PlanInvalidated('Grid time step differs from the plan.')
        if not np.isclose(grid.courant_number, self.boundaries.courant_number, rtol=0, atol=0):
            raise PlanInvalidated('Grid Courant number differs from the plan.')
        if {k: complex(v) for k, v in grid.wrap.items()} != self.boundaries.wrap:
            raise PlanInvalidated('Grid Bloch wrap phases differ from the plan.')
        expected = {}
        for segment in self.boundaries.cpml:
            expected.setdefault((segment.forward, segment.axis, segment.component), []).append(segment)
        actual_segments = {k: list(v) for k, v in grid.cpml.items() if v}
        if set(actual_segments) != set(expected) or any(len(actual_segments[k]) != len(expected[k]) for k in expected):
            raise PlanInvalidated('Grid CPML segment layout differs from the plan.')
        for key, segments in expected.items():
            for planned, actual in zip(segments, actual_segments[key]):
                for name in ('b', 'c'):
                    value = host(actual[name]).reshape(-1)
                    if not np.array_equal(value, getattr(planned, name).astype(value.dtype)):
                        raise PlanInvalidated(f'Grid CPML coefficient {name} differs from the plan on axis {key[1]}.')
                inv_k = host(actual['inv_k']).reshape(-1)
                if not np.array_equal(inv_k, (1/planned.kappa).astype(inv_k.dtype)):
                    raise PlanInvalidated(f'Grid CPML kappa differs from the plan on axis {key[1]}.')
        absorber = self.boundaries.absorber
        if (absorber is None) != (getattr(grid, 'absorber', None) is None):
            raise PlanInvalidated('Grid absorber faces differ from the plan.')
        if absorber is not None:
            if [list(face) for face in grid.absorber_faces] != absorber['faces']:
                raise PlanInvalidated('Grid absorber faces differ from the plan.')
            for axis, pair in absorber['profiles'].items():
                if any(not np.array_equal(host(actual), planned) for planned, actual in zip(pair, grid.absorber[axis])):
                    raise PlanInvalidated(f'Grid absorber profile differs from the plan on axis {axis}.')
            for name, value in getattr(grid, 'absorber_reference', {}).items():
                if value != absorber['reference_epsilon'].get(name):
                    raise PlanInvalidated(f'Grid absorber reference permittivity of {name} differs from the plan.')

    def verify_planes(self, planes):
        """Raise PlanInvalidated when prepared frequency planes differ from the plan's field monitors."""
        planned = [m for m in self.monitors if m.kind == 'field']
        if len(planned) != len(planes):
            raise PlanInvalidated('The number of prepared frequency planes differs from the plan.')
        for monitor, plane in zip(planned, planes):
            if (tuple(plane.components) != monitor.components
                    or not np.array_equal(np.asarray(plane.frequency), monitor.frequency_hz)
                    or not np.array_equal(plane.plan['points_um'], monitor.points_um)
                    or not np.array_equal(plane.plan['weights'], monitor.weights)):
                raise PlanInvalidated(f'{monitor.name}: prepared frequency plane differs from the plan.')


def project_snapshot(project):
    """What a differentiable model fixes at construction: the scene and its realized nodes."""
    return dict(project=project.model_dump(), nodes=tuple(a.tolist() for a in project.region.mesh_nodes))


def check_snapshot(project, snapshot, owner):
    if project_snapshot(project) != snapshot:
        raise PlanInvalidated(f'{owner}: the project or its realized mesh changed after the plan was fixed. '
                              'Rebuild the model instead of mutating its project during differentiation.')


def _run_control(region):
    """The run-control settings the completed step count depends on.

    With auto_shutoff the decay test decides when the run stops, so the settings it reads
    enter the time base. The divergence checks abort a run and never change a result.
    """
    c = region.run_control
    if not c.auto_shutoff:
        return dict(auto_shutoff=False)
    return dict(auto_shutoff=True, decay_threshold=float(c.decay_threshold), check_interval=int(c.check_interval),
                consecutive_checks=int(c.consecutive_checks), min_steps=int(c.min_steps),
                source_tail_amplitude=float(c.source_tail_amplitude), after_source_s=float(c.after_source_s))


def _boundary_plan(region, absorber_faces=(), reference_epsilon=None):
    from .boundaries import BoundaryDescription
    template = BoundaryDescription(region, absorber_faces)
    faces = []
    for axis in range(3):
        for side, face in enumerate(region.boundaries.pair(axis)):
            faces.append(dict(axis='xyz'[axis], side='min' if side == 0 else 'max', kind=face.kind,
                              layers=region.pml_layers(axis, side),
                              **{k: v for k, v in face.model_dump().items() if k not in ('kind', 'layers')}))
    segments = []
    for (forward, axis, component), items in template.cpml.items():
        for segment in items:
            sl = segment['slice'][axis]
            segments.append(CPMLSegment(bool(forward), int(axis), int(component), int(segment['side']),
                                        int(sl.start), int(sl.stop),
                                        *(_frozen(segment[name].reshape(-1)) for name in ('kappa', 'sigma', 'alpha', 'b', 'c'))))
    return BoundaryPlan(tuple(faces), tuple(float(v) for v in region.bloch_phase), float(template.courant_number),
                        {k: complex(v) for k, v in template.wrap.items()},
                        {k: float(v) for k, v in template.pec_upper.items()},
                        {k: float(v) for k, v in template.pmc_lower.items()},
                        {k: float(v) for k, v in template.pmc_upper.items()}, tuple(segments),
                        None if template.absorber is None else dict(
                            faces=[list(face) for face in template.absorber_faces],
                            profiles={axis: tuple(_frozen(v) for v in pair) for axis, pair in template.absorber.items()},
                            reference_epsilon=dict(reference_epsilon or {})))


def _absorber_references(project):
    """Real permittivity each active dispersive material's absorber loss is matched to (materials.configure_materials)."""
    from .materials import absorber_reference_epsilon, frozen_pml_frequency_hz
    r = project.region
    if r.pml_dispersion != 'absorber':
        return {}
    active = {s.material for s in project.structures if s.enabled}
    reference_hz = frozen_pml_frequency_hz(project)
    return {m.name: absorber_reference_epsilon(m, reference_hz, r.time_step) for m in project.materials
            if m.oscillators and m.name in active}


def _ade_plans(project):
    from .materials import MaterialADE
    r = project.region
    active = {s.material for s in project.structures if s.enabled}
    plans = []
    for material in project.materials:
        if not material.oscillators or material.name not in active:
            continue
        ade = MaterialADE(_ADEDescription(r.time_step), material, np.zeros(0, dtype=np.int64), r.material_sampling == 'yee')
        plans.append(ADEPlan(material.name, float(material.epsilon_inf), tuple(tuple(map(float, o)) for o in material.oscillators),
                             *(_frozen(np.asarray(getattr(ade, name), dtype=np.float64).reshape(-1)) for name in ('a', 'd', 'k'))))
    return tuple(plans)


def _source_plans(project):
    from .injection import source_terms, oneway_metadata
    from .waveforms import source_time_signal
    r = project.region
    times = np.arange(1, r.steps+1)*r.time_step
    plans = []
    for raw in project.sources:
        if not raw.enabled:
            continue
        s = project.resolved_source(raw)
        terms = []
        for component, location, samples, profile in source_terms(project, raw):
            # Injection times, formed as source_terms forms them: E terms enter at (n+1) dt, H terms at (n+1.5) dt.
            sample_times = times+(SAMPLE_TIME_STEPS[component[0]]-1.)*r.time_step
            terms.append(SourceTerm(component, tuple(location), _frozen(sample_times),
                                    _frozen(np.asarray(samples, dtype=np.float64)),
                                    None if profile is None else _frozen(profile)))
        tfsf = drive = oneway = None
        if s.kind == 'tfsf':
            from .tfsf import tfsf_metadata
            tfsf = {k: v for k, v in tfsf_metadata(s, r).items() if k != 'source_id'}
            drive = _frozen(source_time_signal(s, times))
        elif s.injection == 'oneway':
            oneway = {k: v for k, v in oneway_metadata(s, r).items() if k != 'source_id'}
        plans.append(SourcePlan(s.id, s.name, s.kind, s.injection,
                                s.model_dump(mode='json', exclude={'id', 'name', 'enabled', 'signal'}),
                                tuple((f, float(w)) for f, w in s.polarization_components),
                                tuple(terms), tfsf, drive, oneway))
    return tuple(plans)


def _monitor_plans(project):
    from .field_monitors import plane_plan, interpolation_map
    from .solver import index_at
    from .spectra import frequency_samples
    r = project.region
    plans = []
    for raw in project.monitors:
        if not raw.enabled:
            continue
        m = project.resolved_monitor(raw)
        spectrum = m.spectrum.model_dump(mode='json')
        apodization = dict(mode=m.spectrum.apodization)
        if m.spectrum.apodization in ('start', 'end', 'full'):
            apodization.update(center_s=m.spectrum.apodization_center, time_width_s=m.spectrum.apodization_time_width)
        if m.kind == 'field':
            plan = plane_plan(r, m)
            maps = tuple((c, *(_frozen(v) for v in interpolation_map(r, c, plan['points_um']))) for c in m.required_fields)
            plans.append(MonitorPlan(m.id, m.name, 'field', tuple(m.required_fields), None, m.normal,
                                     _frozen(plan['points_um']), _frozen(plan['weights']), tuple(plan['shape']), maps,
                                     _frozen(frequency_samples(m.spectrum)), apodization, m.time_downsample, m.dft_precision, spectrum))
        else:
            frequency = None if m.spectrum.sampling == 'fft' else _frozen(frequency_samples(m.spectrum))
            plans.append(MonitorPlan(m.id, m.name, 'point', (m.component,), tuple(int(i) for i in index_at(m.center, r, m.component)),
                                     None, None, None, None, (), frequency, apodization, m.time_downsample, None, spectrum))
    return tuple(plans)


def resolve_plan(project):
    """Resolve every physical input of ``project`` once and return the frozen plan."""
    from .models import Project
    from .solver import estimate, field_axes
    from .boundaries import absorber_faces
    from .field_monitors import plane_sizes
    project = Project.model_validate(project.model_dump())
    # The plane sample caps are counted before any plane or map is built.
    plane_sizes(project)
    r = project.region
    nodes = tuple(_frozen(a) for a in r.mesh_nodes)
    axes = {c: tuple(_frozen(a) for a in field_axes(r, c)) for c in COMPONENTS}
    active = {s.material for s in project.structures if s.enabled}
    structures = tuple(dict(s.model_dump(mode='json', exclude={'id', 'name', 'enabled'}), order=i)
                       for i, s in enumerate(project.structures) if s.enabled)
    # Effective material parameters: what the rasterizer and the ADE read, not fit provenance or colors.
    materials = tuple(dict(name=m.name, model=m.model,
                           epsilon=list(m.epsilon_tensor) if m.model == 'tensor' else m.instantaneous_epsilon,
                           oscillators=[list(o) for o in m.oscillators])
                      for m in project.materials if m.name in active)
    placement = {k: (r.tiling.model_dump(mode='json') if k == 'tiling' else getattr(r, k)) for k in PLACEMENT_FIELDS}
    values = dict(dimension=r.dimension, size=tuple(float(v) for v in r.size), shape=tuple(int(n) for n in r.shape),
                  nodes=nodes, axes=axes, time_step=float(r.time_step), steps=int(r.steps), run_control=_run_control(r),
                  sample_time_steps=dict(SAMPLE_TIME_STEPS), fourier_convention=FOURIER_CONVENTION,
                  background_index=float(r.background_index), material_sampling=r.material_sampling,
                  interface_method=r.interface_method, subpixel_quadrature=int(r.subpixel_quadrature),
                  boundaries=_boundary_plan(r, absorber_faces(project), _absorber_references(project)), pml_dispersion=r.pml_dispersion,
                  structures=structures,
                  materials=materials, ade=_ade_plans(project), sources=_source_plans(project),
                  monitors=_monitor_plans(project))
    # The PML dispersion mode changes the absorber update (docs/BOUNDARIES.md), so it is hashed with the exterior.
    canonical = dict(
        mesh={k: _canonical(values[k]) for k in ('dimension', 'size', 'shape', 'nodes', 'axes')},
        time={k: _canonical(values[k]) for k in ('time_step', 'steps', 'run_control', 'sample_time_steps', 'fourier_convention')},
        exterior=dict(background_index=values['background_index'], boundaries=_canonical(values['boundaries']),
                      pml_dispersion=values['pml_dispersion']),
        material={k: _canonical(values[k]) for k in ('material_sampling', 'interface_method', 'subpixel_quadrature',
                                                       'background_index', 'structures', 'materials', 'ade')},
        sources=_canonical(values['sources']),
        monitors=_canonical(values['monitors']))
    resources = estimate(project, endpoint_dispatch=False)
    return SimulationPlan(**values, resources=resources, placement=placement, plan_hash=_digest(canonical),
                          _canonical=canonical, _project=project)


def resources_copy(plan):
    """A mutable copy of the estimate record for a run's summary."""
    return copy.deepcopy(plan.resources)
