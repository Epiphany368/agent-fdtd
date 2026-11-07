"""N-port fixed-mode networks on cardinal apertures.

Every port is a fixed periodic-supercell eigenmode on an axis-aligned
rectangle that may be smaller than the domain cross-section. Ports may use any
of the six cardinal normals, their own aperture, exterior guide and matched
calibration. Mode profiles, eigenvalues, source packets and calibration remain
fixed; only interior material differentiates through the existing checkpointed
resident adjoint or the streamed X-slab adjoint.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np
import torch

from .differentiable import AdjointOptions
from .mode_injection import ApertureModalLaunch, ModeInjectedPlaneSimulation, _profile
from .mode_network import ModeNetworkResult, _decompose, _fixed_section
from .mode_ports import C0, solve_waveguide_modes, _sample_permittivity_component
from .models import FieldMonitor, Project
from .recomputed_batch import recompute_cases
from .streamed import StreamedAdjointOptions
from .waveforms import source_time_signal


@dataclass(frozen=True)
class ModePort:
    """One fixed-mode port on a cardinal plane.

    normal is the propagation axis and direction the inward propagation sign
    along it, so a port on the minimum face uses +1. center_um is the phase
    plane centre and size_um is zero along the normal and gives the transverse
    aperture, whose edges must lie on Yee cell boundaries. The source plane
    lies source_offset_um outward from the phase plane. mode_indices select
    fixed periodic eigenmodes of the aperture cross-section.
    """
    name: str
    normal: str
    direction: int
    center_um: tuple
    size_um: tuple
    source_offset_um: float
    mode_indices: tuple = (0,)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Port names must be nonempty strings.')
        if self.normal not in ('x', 'y', 'z'):
            raise ValueError('Port normal must be x, y or z.')
        if isinstance(self.direction, bool) or self.direction not in (1, -1):
            raise ValueError('Port direction must be +1 or -1 along its normal.')
        for field in ('center_um', 'size_um'):
            value = tuple(float(v) for v in getattr(self, field))
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError('Port centre and size need three finite coordinates in um.')
            object.__setattr__(self, field, value)
        if self.size_um[self.axis] != 0 or any(self.size_um[a] <= 0 for a in self.transverse_axes):
            raise ValueError('Port size must be zero along the normal and positive across it.')
        if not math.isfinite(self.source_offset_um) or self.source_offset_um <= 0:
            raise ValueError('source_offset_um must be a positive outward distance.')
        modes = tuple(self.mode_indices)
        if (not modes or len(set(modes)) != len(modes)
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in modes)):
            raise ValueError('Select unique nonnegative integer mode indices.')
        object.__setattr__(self, 'mode_indices', modes)

    @property
    def axis(self):
        return 'xyz'.index(self.normal)

    @property
    def transverse_axes(self):
        return (self.axis+1) % 3, (self.axis+2) % 3

    @property
    def source_coordinate_um(self):
        return self.center_um[self.axis]-self.direction*self.source_offset_um


def _aperture_cells(region, normal, center_um, size_um):
    """Yee cell index ranges (begin, end, full) of an aperture, in cyclic order."""
    w = 'xyz'.index(normal)
    ranges = []
    for axis in ((w+1) % 3, (w+2) % 3):
        if region.dimension == '2d' and axis == 2:
            # This entry describes storage only; z has no mesh/BC/aperture edge.
            ranges.append((0, 1, True))
            continue
        nodes = region.mesh_nodes[axis]
        n = region.shape[axis]
        lo, hi = center_um[axis]-size_um[axis]/2, center_um[axis]+size_um[axis]/2
        begin, end = int(np.argmin(abs(nodes-lo))), int(np.argmin(abs(nodes-hi)))
        if not (math.isclose(float(nodes[begin]), lo, abs_tol=1e-7, rel_tol=0)
                and math.isclose(float(nodes[end]), hi, abs_tol=1e-7, rel_tol=0)):
            raise ValueError('Port aperture edges must lie on transverse Yee cell boundaries.')
        if end-begin < 2:
            raise ValueError('Port apertures need at least two cells across each transverse axis.')
        kinds = tuple(face.kind for face in region.boundaries.pair(axis))
        if kinds == ('periodic', 'periodic'):
            if region.bloch_phase[axis] != 0:
                raise ValueError('Transverse periodic axes must have zero Bloch phase.')
        elif kinds == ('pml', 'pml'):
            inner = region.interior_bounds(axis)
            if lo < inner[0]-1e-7 or hi > inner[1]+1e-7:
                raise ValueError('Port apertures on a CPML axis must lie inside the physical region.')
        else:
            raise ValueError('Port transverse axes require periodic or CPML boundary pairs.')
        ranges.append((begin, end, begin == 0 and end == n))
    return tuple(ranges)


@dataclass(frozen=True)
class _GradedApertureModalLaunch(ApertureModalLaunch):
    """Uniform modal support inside a graded domain; no resampled operator."""
    normal_nodes_um: tuple

    def detector_mode(self, coordinate_um):
        from .mode_injection import ModalLaunch
        nodes = np.asarray(self.normal_nodes_um)
        k = int(np.argmin(abs(nodes-coordinate_um)))
        if not math.isclose(float(nodes[k]), coordinate_um, rel_tol=0, abs_tol=1e-7):
            raise ValueError('Graded modal detectors must lie on realised E nodes.')
        widths = np.diff(nodes)[max(0, k-1):k+1]
        if len(widths) != 2 or not np.allclose(widths, self.normal_step_um, rtol=1e-10, atol=0):
            raise ValueError('Graded modal detector needs the launch longitudinal spacing on both sides.')
        # The base method supplies the unchanged Yee half-cell phase correction.
        # Its lattice test is applied at a valid synthetic node, after the real
        # node and both local widths above have been checked.
        return ModalLaunch.detector_mode(self, self.normal_origin_um)


def prepare_aperture_modal_launch(project, permittivity, *, mode_index=0, num_modes=1,
                                  source_budget_bytes=256*1024**2, confinement_tolerance=1e-3,
                                  allow_graded=False):
    """Fixed periodic-supercell mode currents on the source's own rectangle.

    The sole project source fixes normal, direction, timing and the aperture
    through its transverse centre and size. The mode is solved on that
    rectangle with periodic transverse boundaries, so unless the rectangle is
    the complete periodic cell on an axis, its outermost cell ring must hold
    at most confinement_tolerance of the squared six-component amplitude.
    Injection and the fixed-material check cover only the rectangle.
    allow_graded explicitly admits a graded exterior only when the realised
    aperture and source neighbourhood retain uniform Yee spacing. A graded
    transverse eigenproblem is refused, never replaced by a nominal grid.
    """
    r = project.region
    active = [project.resolved_source(s) for s in project.sources if s.enabled]
    if len(active) != 1:
        raise ValueError('Aperture modal launch requires exactly one enabled source.')
    s = active[0]
    graded = r.mesh_type == 'graded' and allow_graded is True
    if (r.mesh_type != 'uniform' and not graded) or r.complex_fields or r.material_sampling != 'yee':
        raise ValueError('Aperture modal launch requires a real uniform Yee-sampled region.')
    if r.dimension == '2d':
        from .invariant_modes import validate_invariant_launch
        validate_invariant_launch(r, s)
    if r.interface_method != 'staircase' or any(m.oscillators for m in project.materials):
        raise ValueError('Only real nondispersive staircase materials are supported.')
    if s.kind != 'plane' or s.injection != 'soft' or s.pulse != 'gaussian' or s.time_definition != 'cycles':
        raise ValueError('Use one soft Gaussian plane source with cycle-based timing.')
    if s.pulse_cycles < 2:
        raise ValueError('Validated modal pulses require at least two carrier cycles.')
    if not math.isfinite(confinement_tolerance) or not 0 < confinement_tolerance < 1:
        raise ValueError('confinement_tolerance must lie in (0,1).')
    if type(source_budget_bytes) is not int or source_budget_bytes < 1:
        raise ValueError('source_budget_bytes must be a positive integer.')
    if isinstance(mode_index, bool) or not isinstance(mode_index, int) or not 0 <= mode_index < num_modes:
        raise ValueError('Invalid mode index.')
    w = 'xyz'.index(s.normal)
    u, v = (w+1) % 3, (w+2) % 3
    if s.size[w] != 0 or any(face.kind != 'pml' for face in r.boundaries.pair(w)):
        raise ValueError('The launch plane needs zero normal span and CPML at both ends of its normal.')
    nodes = r.mesh_nodes[w]
    k = int(np.argmin(abs(nodes[:-1]-s.center[w])))
    lo, hi = r.interior_bounds(w)
    if (k < 2 or k+2 >= len(nodes) or nodes[k-1] < lo or nodes[k+2] > hi
            or not math.isclose(s.center[w], float(nodes[k]), abs_tol=1e-7, rel_tol=0)):
        raise ValueError('The launch plane must lie on a longitudinal E node with two interior cells before CPML.')
    d = 1 if s.direction == '+' else -1
    ranges = _aperture_cells(r, s.normal, s.center, s.size)
    spacing = (r.axis_steps[u], r.axis_steps[v])
    h = r.axis_steps[w]
    if graded:
        from .mesh import local_uniform_step
        # These are the exact spacings of the operator's Yee samples, not
        # minimum/mean approximations to a nonuniform transverse section.
        spacing = tuple(r.axis_steps[a] if r.dimension == '2d' and a == 2
                        else local_uniform_step(r, a, begin, end)
                        for a, (begin, end, _) in zip((u, v), ranges))
        h = local_uniform_step(r, w, k-1, k+2)
    shape = tuple(end-begin for begin, end, _ in ranges)
    dtype = np.float32 if r.precision == 'float32' else np.float64
    required = (8*r.steps+40*math.prod(shape))*np.dtype(dtype).itemsize
    if required > source_budget_bytes:
        raise ValueError('Modal source tables exceed the source byte budget.')
    dt = r.time_step
    omega = 2*np.pi*C0/(s.wavelength*1e-6)
    kt = 2*np.sin(omega*dt/2)/(C0*dt)*1e-6
    origin = (float(r.mesh_nodes[u][ranges[0][0]]), float(r.mesh_nodes[v][ranges[1][0]]))
    mode_solver = solve_waveguide_modes
    if r.dimension == '2d':
        from .invariant_modes import solve_invariant_waveguide_modes
        mode_solver = solve_invariant_waveguide_modes
    modes = mode_solver(permittivity, shape=shape, spacing_um=spacing,
                                  origin_um=origin, wavelength_um=2*np.pi/kt, normal=s.normal,
                                  num_modes=num_modes, precision=r.precision)
    base = modes[mode_index]
    ratio = base.beta_per_um*h/2
    if not 0 < ratio < 1:
        raise ValueError('Mode exceeds the longitudinal Yee propagation band.')
    beta = 2*np.arcsin(ratio)/h
    mode = replace(base, wavelength_um=s.wavelength, beta_per_um=float(beta))
    energy = np.sum(np.abs(mode.fields.astype(np.complex128))**2, axis=-1)
    edge = np.zeros(shape, dtype=bool)
    for i, (_, _, full) in enumerate(ranges):
        if full:
            continue
        index = [slice(None)]*2
        index[i] = slice(0, 1)
        edge[tuple(index)] = True
        index[i] = slice(shape[i]-1, shape[i])
        edge[tuple(index)] = True
    fraction = float(energy[edge].sum()/energy.sum()) if edge.any() else 0.
    if not math.isfinite(fraction) or fraction > confinement_tolerance:
        raise ValueError(f'Aperture edge energy fraction {fraction:.3e} exceeds the confinement tolerance '
                         f'{confinement_tolerance:.1e}. Enlarge the aperture or select a bound mode.')
    epsilon = np.empty((*shape, 3), dtype=dtype)
    for c in range(3):
        coords = np.meshgrid(*mode.component_axes('E'+'xyz'[c]), indexing='ij')
        epsilon[..., c] = _sample_permittivity_component(permittivity, coords, c, shape, dtype)
    if np.min(epsilon) < 1:
        raise ValueError('Native FDTD CFL requires epsilon at least one.')
    amplitude_scale = float(np.max(np.abs(mode.fields[..., :3])))
    incident = mode if d == 1 else mode.backward()
    fields = incident.fields/amplitude_scale
    courant = C0*dt/(h*1e-6)
    phase = np.exp(-1j*beta*h/2)
    loc_e, loc_h = [slice(None)]*3, [slice(None)]*3
    loc_e[w] = slice(k, k+1)
    hat = k-1 if d == 1 else k
    loc_h[w] = slice(hat, hat+1)
    for axis, (begin, end, _) in zip((u, v), ranges):
        loc_e[axis] = loc_h[axis] = slice(begin, end)
    terms = []
    for c, partner, sign in ((u, v, -1), (v, u, 1)):
        # (w cross H)_u=-H_v and (w cross H)_v=H_u, as in prepare_modal_launch.
        electric = -d*courant*sign*fields[..., 3+partner]*phase/epsilon[..., c]
        magnetic = d*courant*sign*fields[..., partner]
        for family, component, loc, profile, tshift in (
                ('E', c, tuple(loc_e), electric, .5), ('H', c, tuple(loc_h), magnetic, 1.)):
            times = (np.arange(r.steps)+tshift)*dt
            for imag, offset in ((False, 90.), (True, 0.)):
                waveform = source_time_signal(s.model_copy(update={'phase': s.phase+offset}), times).astype(dtype)
                spatial = _profile(profile.imag if imag else profile.real, s.normal).astype(dtype)
                waveform.setflags(write=False)
                spatial.setflags(write=False)
                terms.append((family+'xyz'[component], loc, waveform, spatial))
    signature_region = r
    if graded and r.mesh_auto_refine:
        # Plane execution freezes these same refinements before replacing
        # field monitors with internal point observations.
        from .mesh import freeze_refinements
        signature_region = freeze_refinements(project).region
    region_signature = json.dumps(signature_region.model_dump(mode='json'), sort_keys=True)
    source_signature = json.dumps(s.model_dump(mode='json'), sort_keys=True)
    aperture = tuple((begin, end) for begin, end, _ in ranges)
    digest = hashlib.sha256(region_signature.encode()+source_signature.encode()+mode.fields.tobytes()
                            +epsilon.tobytes()+repr(aperture).encode()).hexdigest()
    epsilon.setflags(write=False)
    args = (mode, epsilon, region_signature, source_signature, float(nodes[0]), h, k, d,
            base.beta_per_um, float(kt), tuple(terms), digest, aperture, fraction)
    if graded:
        return _GradedApertureModalLaunch(*args, tuple(float(x) for x in nodes))
    return ApertureModalLaunch(*args)


class ModeBranchNetwork:
    """Complex S[outgoing channel, incident channel] for N fixed-mode ports.

    Ports may sit on any cardinal face with their own apertures and exterior
    guides. Each incident channel is one independent launch with a fresh
    matched straight-guide calibration, and sample columns are recomputed one
    at a time in backward. Material in every port's exterior footprint, its
    aperture times the cells from one cell inside its phase plane outward
    through the source and CPML, must equal that port's calibration guide and
    is frozen. Everything else differentiates. Mode profiles, eigenvalues,
    packets and calibration carry no derivatives. Selected guided channels do
    not account for radiation or omitted modes, so column powers below one
    are reported, not corrected.
    """
    def __init__(self, project, ports, permittivity=None, options=None, *, num_modes=1,
                 port_permittivities=None, network_budget_bytes=256*1024**2,
                 gram_tolerance=1e-4, confinement_tolerance=1e-3):
        self.project = Project.model_validate(project.model_dump())
        self.ports = tuple(ports)
        self.options = options or AdjointOptions(checkpoints=4)
        if not isinstance(self.options, (AdjointOptions, StreamedAdjointOptions)):
            raise ValueError('Branch networks require AdjointOptions or StreamedAdjointOptions.')
        if len(self.ports) < 2 or not all(isinstance(p, ModePort) for p in self.ports):
            raise ValueError('Provide at least two ModePort entries.')
        names = tuple(p.name for p in self.ports)
        if len(set(names)) != len(names):
            raise ValueError('Port names must be unique.')
        self._shared_section = port_permittivities is None
        if port_permittivities is None:
            if permittivity is None:
                raise ValueError('Provide permittivity or explicit port_permittivities.')
            sections = {name: permittivity for name in names}
        else:
            if permittivity is not None:
                raise ValueError('Use either permittivity or port_permittivities, not both.')
            if not isinstance(port_permittivities, Mapping) or set(port_permittivities) != set(names):
                raise ValueError('port_permittivities must map exactly the port names to fixed sections.')
            sections = dict(port_permittivities)
        self._sections = {name: _fixed_section(sections[name]) for name in names}
        if isinstance(num_modes, bool) or not isinstance(num_modes, int) or num_modes < 1:
            raise ValueError('num_modes must be a positive integer.')
        if any(i >= num_modes for p in self.ports for i in p.mode_indices):
            raise ValueError('Select mode indices inside num_modes.')
        if not math.isfinite(gram_tolerance) or not 0 < gram_tolerance <= .01:
            raise ValueError('gram_tolerance must lie in (0,.01].')
        if isinstance(network_budget_bytes, bool) or not isinstance(network_budget_bytes, int) or network_budget_bytes <= 0:
            raise ValueError('network_budget_bytes must be a positive integer.')
        active = [self.project.resolved_source(s) for s in self.project.sources if s.enabled]
        if len(active) != 1:
            raise ValueError('Use one Gaussian plane source as the launch timing template.')
        self.gram_tolerance, self.network_budget_bytes = gram_tolerance, network_budget_bytes
        self.confinement_tolerance = confinement_tolerance
        r = self.project.region
        indices, apertures = [], []
        for port in self.ports:
            nodes = r.mesh_nodes[port.axis]
            index = int(np.argmin(abs(nodes-port.center_um[port.axis])))
            if not math.isclose(float(nodes[index]), port.center_um[port.axis], rel_tol=0, abs_tol=1e-7):
                raise ValueError('Port phase planes must lie on E nodes.')
            offset = port.source_offset_um/r.axis_steps[port.axis]
            if not math.isclose(offset, round(offset), abs_tol=1e-6, rel_tol=0) or round(offset) < 2:
                raise ValueError('Source offsets must be a whole number of at least two cells.')
            indices.append(index)
            apertures.append(_aperture_cells(r, port.normal, port.center_um, port.size_um))
        self.indices, self.apertures = tuple(indices), tuple(apertures)
        boxes = [self._footprint_box(i) for i in range(len(self.ports))]
        for a in range(len(boxes)):
            for b in range(a+1, len(boxes)):
                if all(boxes[a][axis][0] < boxes[b][axis][1] and boxes[b][axis][0] < boxes[a][axis][1]
                       for axis in range(3)):
                    raise ValueError(f'Exterior footprints of ports {names[a]} and {names[b]} overlap.')
        self.channels = tuple((p.name, i) for p in self.ports for i in p.mode_indices)
        self._admit()
        projects, launches = [], []
        for port in self.ports:
            for mode_index in port.mode_indices:
                p = self.project.model_copy(deep=True)
                source = active[0].model_copy(deep=True)
                source.normal = port.normal
                source.direction = '+' if port.direction == 1 else '-'
                source.center = tuple(port.source_coordinate_um if a == port.axis else port.center_um[a]
                                      for a in range(3))
                source.size = port.size_um
                p.sources = [source]
                p.monitors = [FieldMonitor(id=q.name, normal=q.normal, center=q.center_um, size=q.size_um)
                              for q in self.ports]
                p = Project.model_validate(p.model_dump())
                launch = prepare_aperture_modal_launch(p, self._sections[port.name], mode_index=mode_index,
                    num_modes=num_modes, source_budget_bytes=network_budget_bytes,
                    confinement_tolerance=confinement_tolerance)
                projects.append(p)
                launches.append(launch)
        self._projects, self._launches = tuple(projects), tuple(launches)
        self._packet_bytes = sum(x.storage_bytes for x in launches)
        self._admit()
        # Check every port's signed basis on its actual detector rule before
        # any volume field exists.
        from types import SimpleNamespace
        from .field_monitors import plane_plan
        offset = 0
        for port in self.ports:
            monitor = next(m for m in projects[0].monitors if m.id == port.name)
            plan = plane_plan(r, monitor)
            plane = SimpleNamespace(normal=port.normal,
                points_um=torch.tensor(plan['points_um'], dtype=torch.float32),
                points_um_geometry=torch.tensor(plan['points_um'], dtype=torch.float64),
                weights=torch.tensor(plan['weights'], dtype=torch.float32),
                fields=torch.zeros((1, len(plan['weights']), 6), dtype=torch.complex64))
            count = len(port.mode_indices)
            _decompose(plane, launches[offset:offset+count], self.gram_tolerance)
            offset += count
        self._project_snapshot = self.project.model_dump()
        self._configuration_snapshot = self._configuration()
        self._prepared_snapshot = tuple(p.model_dump() for p in self._projects)

    def _configuration(self):
        return (self.ports, self.indices, self.apertures, self.channels, self.options,
                self.gram_tolerance, self.network_budget_bytes, self.confinement_tolerance,
                self._shared_section, tuple(launch.identity for launch in self._launches))

    def _guard_configuration(self):
        if self.project.model_dump() != self._project_snapshot:
            raise ValueError('Branch network project changed. Rebuild fixed ports and launches.')
        if (self._configuration() != self._configuration_snapshot
                or tuple(p.model_dump() for p in self._projects) != self._prepared_snapshot):
            raise ValueError('Branch network configuration changed. Rebuild fixed ports and launches.')

    def _footprint(self, index):
        port, i = self.ports[index], self.indices[index]
        selection = [slice(None)]*3
        selection[port.axis] = slice(None, i+2) if port.direction == 1 else slice(i-1, None)
        for axis, (begin, end, _) in zip(port.transverse_axes, self.apertures[index]):
            selection[axis] = slice(begin, end)
        return tuple(selection)

    def _footprint_box(self, index):
        return tuple(s.indices(n)[:2] for s, n in zip(self._footprint(index), self.project.region.shape))

    def _admit(self):
        r = self.project.region
        item = 8 if r.precision == 'float64' else 4
        cells = math.prod(r.shape)
        points = sum(math.prod(end-begin for begin, end, _ in aperture) for aperture in self.apertures)
        count = len(self.channels)
        # Fixed packets for all launches, parameter carrier and gradients,
        # reference epsilon, one plane-map construction and modal scratch.
        packet = getattr(self, '_packet_bytes', count*(8*r.steps+80*points)*item)
        required = packet+21*cells*item+3*cells*item+8192*points+128*count*points*item+16*count*count*item
        if required > self.network_budget_bytes:
            raise ValueError('Branch network packet/wrapper reservation exceeds network byte budget.')
        from .memory_profile import host_memory
        available = host_memory()['available_bytes']
        if available is not None and required > .8*available:
            raise ValueError('Branch network wrapper reservation exceeds available host memory.')
        return required

    def _port_profile(self, index, device):
        """The port's fixed section on the complete cross-section, w axis of length one."""
        port = self.ports[index]
        r = self.project.region
        u, v = port.transverse_axes
        section = self._sections[port.name]
        dtype = np.float32 if r.precision == 'float32' else np.float64
        cross = np.empty((r.shape[u], r.shape[v], 3), dtype=dtype)
        for c in range(3):
            axes = [float(r.mesh_nodes[a][0])+(np.arange(r.shape[a])+(.5 if a == c else 0.))*r.axis_steps[a]
                    for a in (u, v)]
            coords = np.meshgrid(*axes, indexing='ij')
            cross[..., c] = _sample_permittivity_component(section, coords, c, cross.shape[:2], dtype)
        return torch.tensor(np.stack([_profile(cross[..., c], port.normal) for c in range(3)], axis=-1), device=device)

    def reference_epsilon(self, *, port, device='cpu'):
        """The straight calibration guide of one port, not the branched device."""
        self._guard_configuration()
        self._admit()
        names = tuple(p.name for p in self.ports)
        if port not in names:
            raise ValueError('Unknown calibration port name.')
        return self._port_profile(names.index(port), device).expand(self.project.region.shape+(3,)).clone()

    def __call__(self, epsilon, *, output_device='cpu', output_budget_bytes=64*1024**2):
        self._guard_configuration()
        reserved = self._admit()
        r = self.project.region
        if isinstance(output_budget_bytes, bool) or not isinstance(output_budget_bytes, int) or output_budget_bytes <= 0:
            raise ValueError('output_budget_bytes must be a positive integer.')
        output_required = len(self.channels)**2*(16 if r.precision == 'float64' else 8)
        if output_required > output_budget_bytes:
            raise ValueError('S matrix exceeds output byte budget before calibration.')
        if not isinstance(epsilon, torch.Tensor) or epsilon.shape != r.shape+(3,):
            raise ValueError('Network epsilon must be explicit diagonal Yee material.')
        if epsilon.dtype != (torch.float64 if r.precision == 'float64' else torch.float32):
            raise ValueError('Network epsilon precision must match the region.')
        if epsilon.device.type not in ('cpu', 'cuda'):
            raise ValueError('Network epsilon requires CPU or CUDA.')
        streamed = isinstance(self.options, StreamedAdjointOptions)
        if streamed and epsilon.device.type != 'cpu':
            raise ValueError('Streamed branch networks require CPU epsilon.')
        fixed = []
        for index, port in enumerate(self.ports):
            selection = self._footprint(index)
            profile = self._port_profile(index, epsilon.device)
            expected = profile[tuple(slice(None) if a == port.axis else selection[a] for a in range(3))]
            if not torch.allclose(epsilon[selection], expected.expand_as(epsilon[selection]), rtol=2e-6, atol=1e-7):
                raise ValueError(f'Exterior material of port {port.name} must match its fixed calibration guide.')
            fixed.append(selection)
        material = epsilon.clone() if epsilon.requires_grad else epsilon
        if epsilon.requires_grad:
            for selection in fixed:
                material[selection] = material[selection].detach()
        projects = tuple(p.model_copy(deep=True) for p in self._projects)
        launches, ports = self._launches, self.ports
        counts = [len(p.mode_indices) for p in ports]
        offsets = np.cumsum([0]+counts)
        port_launches = tuple(launches[offsets[i]:offsets[i+1]] for i in range(len(ports)))
        frequency = [C0/(launches[0].mode.wavelength_um*1e-6)]
        tolerance, options = self.gram_tolerance, self.options
        cases, reports = [], []
        for column, (project, launch) in enumerate(zip(projects, launches)):
            port_index = int(np.searchsorted(offsets, column, side='right')-1)
            local_index = column-int(offsets[port_index])
            port = ports[port_index]
            reference = self.reference_epsilon(port=port.name, device=epsilon.device)
            model = ModeInjectedPlaneSimulation(project, launch, options)
            with torch.no_grad():
                planes = model(reference, frequency)
                plane = planes[port.name]
                scale = float(plane.fields.abs().max())
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError('Calibration has no finite incident field.')
                forward, backward = _decompose(replace(plane, fields=plane.fields/scale),
                                               port_launches[port_index], tolerance)
                incoming = forward if port.direction == 1 else backward
                baseline = backward if port.direction == 1 else forward
                normalization = incoming[0, local_index].detach().clone()
                if not bool(torch.isfinite(normalization)) or not bool(normalization.abs() > 1e-6*incoming.abs().max()):
                    raise ValueError('Calibration incident channel is unsupported.')
                incident = incoming[0].clone()
                incident[local_index] = 0
                if bool(incident.abs().max() > 1e-3*normalization.abs()):
                    raise ValueError('Calibration excites multiple incident channels. Select a validated basis.')
                reflected = baseline[0].detach().clone()
                reports.append(dict(incident_channel=self.channels[column], field_scale=scale,
                                    incident_amplitude=[float(normalization.real), float(normalization.imag)]))
            del planes, plane, forward, backward, incoming, baseline, model, reference

            def case(value, p=project, source=launch, input_port=port_index,
                     norm=normalization, reflection=reflected, field_scale=scale):
                current = ModeInjectedPlaneSimulation(p, source, options)(value, frequency)
                outgoing = []
                for i, (q, basis) in enumerate(zip(ports, port_launches)):
                    f, b = _decompose(replace(current[q.name], fields=current[q.name].fields/field_scale), basis, tolerance)
                    amplitude = (b if q.direction == 1 else f)[0]
                    if i == input_port:
                        amplitude = amplitude-reflection
                    outgoing.append(amplitude/norm)
                return torch.cat(outgoing)
            cases.append(case)
        columns = recompute_cases(cases, material, output_device=output_device,
                                  output_budget_bytes=output_budget_bytes)
        s = columns.transpose(0, 1)
        with torch.no_grad():
            column_power = s.abs().square().sum(0).cpu()
        return ModeNetworkResult(s, self.channels, tuple(p.center_um[p.axis] for p in ports),
            dict(scope='N-port fixed-mode branch network', execution='streamed' if streamed else 'resident',
                 ports=[dict(name=p.name, normal=p.normal, direction=p.direction, phase_plane_index=i,
                             aperture_cells=[list(item[:2]) for item in aperture], full_cell=[item[2] for item in aperture],
                             edge_energy_fraction=port_launches[n][0].edge_energy_fraction)
                        for n, (p, i, aperture) in enumerate(zip(ports, self.indices, self.apertures))],
                 column_power=column_power.tolist(), power_defect=(1-column_power).tolist(),
                 power_balance='sum over selected guided channels only; radiation and omitted modes are not included',
                 wrapper_reservation_bytes=reserved, mode_boundary='periodic aperture supercell',
                 calibration_volume_bytes=3*math.prod(r.shape)*epsilon.element_size(), calibration_volume_limit=1,
                 per_port_sections=not self._shared_section, calibration=reports,
                 calibration_policy='fresh matched-guide solve per column per call', case_graphs_retained=0,
                 backward_case_graph_limit=1,
                 phase_convention='outgoing at row phase plane / matched incident at column phase plane',
                 reflection_reference='subtract same-input-port matched-guide outgoing baseline only',
                 mode_profile_gradients=False, exterior_material_gradients='frozen'))


def branch_network_from_ports(ports, project, *, normal_convention, wavelength_um, source_offset_um,
                              core_epsilon=None, cladding_epsilon=None, aperture_um=None,
                              mode_indices=(0,), options=None, num_modes=1, **network_kwargs):
    """Build a ModeBranchNetwork from in-plane port markers such as GDSImport.ports.

    Each marker needs name, center_um, normal, width_um and height_um. Only
    cardinal x or y normals are accepted; normal_convention declares whether
    they point outward from or inward into the device. With core_epsilon and
    cladding_epsilon, a port's fixed guide is a rectangle of core_epsilon,
    width_um across the in-plane transverse axis and height_um along z, inside
    cladding_epsilon. Without them, pass ModeBranchNetwork's own permittivity
    or port_permittivities sections, each in that port's cyclic transverse
    coordinates. aperture_um gives the (in-plane, z) aperture extents shared
    by all ports; None uses the marker width and height themselves. Both must
    align with Yee cell boundaries. source_offset_um is one outward distance
    or a mapping by name. mode_indices is one index sequence for every port
    or a mapping by name. The project holds one Gaussian plane pulse template
    whose wavelength is replaced. Marker layers and geometry are not
    rasterized; the caller's runtime epsilon, built from reference_epsilon or
    native voxelization, stays authoritative. Remaining keyword arguments pass
    to ModeBranchNetwork.
    """
    if normal_convention not in ('outward', 'inward'):
        raise ValueError('Declare normal_convention as outward or inward.')
    markers = tuple(ports)
    if len(markers) < 2:
        raise ValueError('Provide at least two port markers.')
    names = [m.name for m in markers]
    if len(set(names)) != len(names):
        raise ValueError('Port marker names must be unique.')
    if isinstance(wavelength_um, bool) or not isinstance(wavelength_um, (int, float)) or not math.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError('Provide a finite positive wavelength_um.')
    rectangles = core_epsilon is not None or cladding_epsilon is not None
    if rectangles:
        for value in (core_epsilon, cladding_epsilon):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1:
                raise ValueError('core_epsilon and cladding_epsilon must both be finite scalars at least one.')
        if 'permittivity' in network_kwargs or 'port_permittivities' in network_kwargs:
            raise ValueError('Use either core/cladding rectangles or explicit sections, not both.')
    elif 'permittivity' not in network_kwargs and 'port_permittivities' not in network_kwargs:
        raise ValueError('Provide core_epsilon and cladding_epsilon, or permittivity or port_permittivities sections.')
    offsets = dict(source_offset_um) if isinstance(source_offset_um, Mapping) else {n: source_offset_um for n in names}
    if set(offsets) != set(names):
        raise ValueError('source_offset_um must be one distance or a mapping of every port name.')
    modes = dict(mode_indices) if isinstance(mode_indices, Mapping) else {n: tuple(mode_indices) for n in names}
    if set(modes) != set(names):
        raise ValueError('mode_indices must be one index sequence or a mapping of every port name.')
    if aperture_um is not None and (len(aperture_um) != 2 or any(not math.isfinite(a) or a <= 0 for a in aperture_um)):
        raise ValueError('aperture_um must be two positive (in-plane, z) extents.')
    close = lambda a, b: math.isclose(a, b, rel_tol=0, abs_tol=1e-8)
    built, sections = [], {}
    for m in markers:
        if len(m.normal) != 3 or len(m.center_um) != 3:
            raise ValueError('Port markers need three-coordinate centres and normals.')
        axis = next((a for a in (0, 1) if close(abs(m.normal[a]), 1)
                     and all(close(m.normal[b], 0) for b in range(3) if b != a)), None)
        if axis is None:
            raise ValueError('Only cardinal in-plane x/y marker normals are supported; use ModePort for z-normal ports.')
        sign = 1 if m.normal[axis] > 0 else -1
        direction = -sign if normal_convention == 'outward' else sign
        inplane = 1-axis
        in_size, z_size = (m.width_um, m.height_um) if aperture_um is None else aperture_um
        size = [0., 0., 0.]
        size[inplane], size[2] = in_size, z_size
        u = (axis+1) % 3

        def section(p, q, c_in=m.center_um[inplane], c_z=m.center_um[2], half_w=m.width_um/2,
                    half_h=m.height_um/2, swap=u == 2):
            in_coord, z_coord = (q, p) if swap else (p, q)
            inside = (np.abs(in_coord-c_in) < half_w-1e-9) & (np.abs(z_coord-c_z) < half_h-1e-9)
            return np.where(inside, core_epsilon, cladding_epsilon)
        built.append(ModePort(m.name, 'xyz'[axis], direction, tuple(m.center_um), tuple(size),
                              offsets[m.name], tuple(modes[m.name])))
        sections[m.name] = section
    copied = Project.model_validate(project.model_dump())
    active = [copied.resolved_source(s) for s in copied.sources if s.enabled]
    if len(active) != 1 or active[0].kind != 'plane':
        raise ValueError('Provide exactly one plane source as the pulse timing template.')
    source = active[0].model_copy(deep=True)
    source.wavelength = float(wavelength_um)
    copied.sources = [source]
    if rectangles:
        network_kwargs['port_permittivities'] = sections
    return ModeBranchNetwork(copied, built, options=options, num_modes=num_modes, **network_kwargs)
