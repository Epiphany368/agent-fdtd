"""Fixed guided-mode sheets including their transverse CPML tails.

An ordinary Source remains a timing template on the physical aperture. Only
this validated packet supplies the complete computational-plane currents.
"""
from dataclasses import dataclass, replace
import cmath
import hashlib
import json
import math

import numpy as np

from .mode_injection import ModalLaunch, _profile
from .mode_ports import C0
from .waveforms import source_time_signal


def _immutable(array):
    value = np.ascontiguousarray(array)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class OpenPortOptions:
    cladding_epsilon: float
    mode_budget_bytes: int = 2*1024**3
    target_neff: float | None = None
    confinement_tolerance: float = 1e-4

    def __post_init__(self):
        if (isinstance(self.cladding_epsilon, bool) or not isinstance(self.cladding_epsilon,(int,float,np.floating))
                or not math.isfinite(self.cladding_epsilon) or self.cladding_epsilon < 1):
            raise ValueError('Open ports require finite cladding epsilon at least one.')
        if type(self.mode_budget_bytes) is not int or self.mode_budget_bytes < 1:
            raise ValueError('mode_budget_bytes must be a positive integer.')
        if self.target_neff is not None and (isinstance(self.target_neff,bool)
                or not isinstance(self.target_neff,(int,float,np.floating))
                or not math.isfinite(self.target_neff) or self.target_neff <= 0):
            raise ValueError('target_neff must be positive and finite.')
        if (isinstance(self.confinement_tolerance,bool) or not isinstance(self.confinement_tolerance,(int,float,np.floating))
                or not math.isfinite(self.confinement_tolerance) or not 0 < self.confinement_tolerance <= 1e-3):
            raise ValueError('confinement_tolerance must lie in (0,1e-3].')


@dataclass(frozen=True)
class OpenModalLaunch(ModalLaunch):
    cladding_epsilon: float
    fixed_transverse_slices: tuple

    def detector_mode(self, coordinate_um):
        q = (coordinate_um-self.normal_origin_um)/self.normal_step_um
        if not math.isclose(q, round(q), abs_tol=2e-5, rel_tol=0):
            raise ValueError('Validated modal detectors must lie on longitudinal E-node planes.')
        fields = self.mode.fields.copy()
        w = 'xyz'.index(self.mode.normal)
        factor = cmath.cos(self.mode.beta_per_um*self.normal_step_um/2)
        fields[..., w] *= factor
        for axis in ((w+1)%3, (w+2)%3):
            fields[..., 3+axis] *= factor
        return replace(self.mode, fields=_immutable(fields))


def _open_geometry(project):
    # Revalidation keeps the timing template and detector apertures within the
    # ordinary physical domain. The packet's tails have their own contract.
    from .models import Project
    project = Project.model_validate(project.model_dump())
    r = project.region
    active = [project.resolved_source(s) for s in project.sources if s.enabled]
    if len(active) != 1:
        raise ValueError('Open modal launch requires exactly one enabled timing source.')
    source = active[0]
    if (r.dimension != '3d' or r.mesh_type != 'uniform' or r.complex_fields
            or r.material_sampling != 'yee' or r.precision != 'float32'
            or r.interface_method != 'staircase'):
        raise ValueError('Open modal launch requires real FP32 uniform 3D staircase Yee sampling.')
    if any(m.model != 'dielectric' or m.oscillators for m in project.materials):
        raise ValueError('Open modal launch requires nondispersive isotropic materials.')
    if (source.kind != 'plane' or source.injection != 'soft' or source.pulse != 'gaussian'
            or source.time_definition != 'cycles' or source.pulse_cycles < 2):
        raise ValueError('Use a soft Gaussian plane timing template with at least two carrier cycles.')
    w = 'xyz'.index(source.normal)
    if source.size[w] != 0 or any(face.kind != 'pml' for face in r.boundaries.pair(w)):
        raise ValueError('Open modal propagation requires a zero-normal-span plane and longitudinal CPML pair.')
    fixed = []
    for axis in range(3):
        if axis == w:
            continue
        kinds = tuple(face.kind for face in r.boundaries.pair(axis))
        if kinds == ('pml', 'pml'):
            fixed.extend(((axis, 0, r.pml_layers(axis, 0)+1),
                          (axis, r.shape[axis]-r.pml_layers(axis, 1)-1, r.shape[axis])))
        elif kinds != ('periodic', 'periodic'):
            raise ValueError('Open modal transverse axes require CPML pairs or zero-phase periodic pairs.')
        lo, hi = r.interior_bounds(axis)
        if not (math.isclose(source.center[axis], (lo+hi)/2, abs_tol=1e-7, rel_tol=0)
                and math.isclose(source.size[axis], hi-lo, abs_tol=1e-7, rel_tol=0)):
            raise ValueError('Open modal timing template must cover the complete physical aperture.')
    if not fixed:
        raise ValueError('Use the periodic modal API when no transverse CPML is present.')
    nodes = r.mesh_nodes[w]
    k = int(np.argmin(abs(nodes[:-1]-source.center[w])))
    lo, hi = r.interior_bounds(w)
    if (k < 2 or k+2 >= len(nodes) or nodes[k-1] < lo or nodes[k+2] > hi
            or not math.isclose(source.center[w], float(nodes[k]), abs_tol=1e-7, rel_tol=0)):
        raise ValueError('Open modal source must lie on an E-node with two interior cells before CPML.')
    return project, source, w, k, 1 if source.direction == '+' else -1, tuple(fixed)


def prepare_open_modal_launch(project, permittivity, *, options, mode_index=0,
                              num_modes=1, source_budget_bytes=256*1024**2):
    """Prepare fixed bound-mode currents over the full CPML cross-section.

    The profile and CPML cladding collar remain fixed in material backward.
    Eigenmode derivatives, leaky modes and tapered source apertures are absent.
    """
    if not isinstance(options, OpenPortOptions):
        raise ValueError('Open modal launch requires explicit OpenPortOptions.')
    project, source, w, k, direction, fixed = _open_geometry(project)
    r = project.region
    u, v = (w+1)%3, (w+2)%3
    shape = (r.shape[u], r.shape[v])
    if type(source_budget_bytes) is not int or source_budget_bytes < 1:
        raise ValueError('source_budget_bytes must be a positive integer.')
    required = (8*r.steps+64*math.prod(shape))*4
    if required > source_budget_bytes:
        raise ValueError('Open modal source tables exceed the source byte budget.')
    if type(mode_index) is not int or not 0 <= mode_index < num_modes:
        raise ValueError('Invalid open mode index.')
    from .open_mode_ports import solve_open_waveguide_modes
    modes = solve_open_waveguide_modes(permittivity, region=r, normal=source.normal,
        wavelength_um=source.wavelength, cladding_epsilon=options.cladding_epsilon,
        num_modes=num_modes, target_neff=options.target_neff,
        mode_budget_bytes=options.mode_budget_bytes,
        confinement_tolerance=options.confinement_tolerance)
    mode = modes[mode_index]
    epsilon = mode.sampled_epsilon
    beta = mode.beta_per_um
    h, dt = r.axis_steps[w], r.time_step
    omega = 2*np.pi*C0/(source.wavelength*1e-6)
    kt = 2*np.sin(omega*dt/2)/(C0*dt)*1e-6
    amplitude_scale = float(np.max(np.abs(mode.fields[..., :3])))
    if not math.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError('Open modal electric profile must be finite and nonzero.')
    incident = mode if direction == 1 else mode.backward()
    fields = incident.fields/amplitude_scale
    courant = C0*dt/(h*1e-6)
    phase = np.exp(-1j*beta*h/2)
    loc_e, loc_h = [slice(None)]*3, [slice(None)]*3
    loc_e[w] = slice(k, k+1)
    hat = k-1 if direction == 1 else k
    loc_h[w] = slice(hat, hat+1)
    terms = []
    for component, partner, sign in ((u, v, -1), (v, u, 1)):
        electric = -direction*courant*sign*fields[..., 3+partner]*phase/epsilon[..., component]
        magnetic = direction*courant*sign*fields[..., partner]
        for family, location, profile, shift in (('E', tuple(loc_e), electric, .5),
                                                ('H', tuple(loc_h), magnetic, 1.)):
            times = (np.arange(r.steps)+shift)*dt
            for imaginary, offset in ((False, 90.), (True, 0.)):
                waveform = source_time_signal(source.model_copy(update={'phase':source.phase+offset}), times).astype(np.float32)
                spatial = _profile(profile.imag if imaginary else profile.real, source.normal).astype(np.float32)
                terms.append((family+'xyz'[component], location, _immutable(waveform), _immutable(spatial)))
    signature = json.dumps(r.model_dump(mode='json'), sort_keys=True)
    source_signature = json.dumps(source.model_dump(mode='json'), sort_keys=True)
    digest = hashlib.sha256(signature.encode()+source_signature.encode()+mode.fields.tobytes()+epsilon.tobytes()
                            +repr(options).encode()).hexdigest()
    return OpenModalLaunch(mode, epsilon, signature, source_signature,
        float(r.mesh_nodes[w][0]), h, k, direction, mode.beta_tilde_per_um,
        float(kt), tuple(terms), digest, float(options.cladding_epsilon), fixed)
