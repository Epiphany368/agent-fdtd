"""Opposing fixed-mode ports with sequential, recomputed S-matrix columns.

Experimental two-port network on one shared propagation axis.
Physical port phases are retained. This is not arbitrary multi-branch routing.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
import torch

from .models import Project, FieldMonitor
from .differentiable import AdjointOptions
from .mode_injection import prepare_modal_launch, ModeInjectedPlaneSimulation, _profile
from .mode_ports import C0, _amplitudes, _uniform_cell_quadrature
from .recomputed_batch import recompute_cases


@dataclass(frozen=True)
class FixedModePort:
    """Complete transverse plane, modes and an exterior source plane in um.

    direction is the inward propagation sign along the shared normal.
    mode_indices select the fixed native eigensolver basis.
    """
    name: str
    coordinate_um: float
    source_coordinate_um: float
    direction: int
    mode_indices: tuple = (0,)

    def __post_init__(self):
        object.__setattr__(self, 'mode_indices', tuple(self.mode_indices))


@dataclass(frozen=True)
class ModeNetworkResult:
    s: torch.Tensor
    channels: tuple
    port_coordinates_um: tuple
    report: dict


def _decompose(plane, launches, gram_tolerance):
    """Project an orthogonal, power-normalized collocated fixed basis."""
    w = 'xyz'.index(plane.normal)
    bases, powers = [], []
    u, v = (w+1)%3, (w+2)%3
    for launch in launches:
        geometry = getattr(plane, 'points_um_geometry', None)
        coordinate = geometry[0, w] if geometry is not None else plane.points_um[0, w]
        mode = launch.detector_mode(float(coordinate))
        _uniform_cell_quadrature(plane, mode)
        basis = torch.tensor(mode.sample_plane(plane.points_um.detach().cpu().numpy()),
                             device=plane.fields.device, dtype=plane.fields.dtype)
        power = .5 * ((basis[:, u]*basis[:, 3+v].conj()
                      - basis[:, v]*basis[:, 3+u].conj()).real * plane.weights).sum()
        if not bool(torch.isfinite(power)) or not bool(power > 0):
            raise ValueError('Port basis has no positive sampled forward power.')
        bases.append(basis / power.sqrt())
        powers.append(power)
    # Both signed blocks are needed. The Hermitian forward Gram alone can
    # hide an anti-Hermitian E/H overlap that leaks into backward channels.
    gram, reverse_gram = [], []
    for first in bases:
        row, reverse_row = [], []
        for second in bases:
            electric = ((first[:, u]*second[:, 3+v].conj()
                         - first[:, v]*second[:, 3+u].conj()) * plane.weights).sum()
            magnetic = ((second[:, u].conj()*first[:, 3+v]
                         - second[:, v].conj()*first[:, 3+u]) * plane.weights).sum()
            row.append(.25*(electric+magnetic))
            reverse_row.append(.25*(electric-magnetic))
        gram.append(torch.stack(row))
        reverse_gram.append(torch.stack(reverse_row))
    gram = torch.stack(gram)
    if not torch.allclose(gram, torch.eye(len(bases), device=gram.device, dtype=gram.dtype),
                          rtol=gram_tolerance, atol=gram_tolerance):
        raise ValueError('Sampled port modes are not power orthogonal. Refine or select a validated basis.')
    reverse_gram = torch.stack(reverse_gram)
    if not torch.allclose(reverse_gram, torch.zeros_like(reverse_gram),
                          rtol=0., atol=gram_tolerance):
        raise ValueError('Sampled port modes fail signed forward/backward orthogonality. Refine or select a validated basis.')
    forward, backward = [], []
    for basis in bases:
        # Scale all products to avoid squared SI-area/DFT underflow in FP32.
        factor = basis.detach().abs().max()
        weights = plane.weights / plane.weights.detach().max()
        f, b = _amplitudes(plane.fields, basis/factor, weights, w)
        forward.append(f/factor)
        backward.append(b/factor)
    return torch.stack(forward, -1), torch.stack(backward, -1)


def _fixed_section(section):
    from .ports import fixed_port_section
    return fixed_port_section(section)


class ModeNetwork:
    """Complex S[outgoing channel, incident channel] from independent launches.

    Supports two opposing ports on project.sources[0].normal. Periodic ports
    cover the complete cell. Explicit OpenPortOptions use the physical CPML
    aperture and fixed homogeneous cladding, including source tails in CPML.
    Each port has a fixed straight calibration cross-section. Use permittivity
    for a shared section, or port_permittivities keyed by the two port names. All port
    and source exterior material remains fixed. Only the interior design
    differentiates. Mode profiles/eigenvalues and calibration do not.

    per-case AdjointOptions bound one solver. network_budget_bytes separately
    bounds retained packets and conservative wrapper working tensors, not the
    solver, eigensolver library, optimizer, or caller material-construction graph.
    """
    def __init__(self, project, ports, permittivity=None, options=None, *, num_modes=2,
                 port_permittivities=None,
                 network_budget_bytes=256*1024**2, gram_tolerance=1e-4, open_ports=None):
        self.project = Project.model_validate(project.model_dump())
        self.ports = tuple(ports)
        self.options = options or AdjointOptions(checkpoints=4)
        if open_ports is not None:
            from .open_mode_injection import OpenPortOptions
            if not isinstance(open_ports,OpenPortOptions):
                raise ValueError('open_ports must be explicit OpenPortOptions.')
        self.open_ports = open_ports
        if not isinstance(self.options, AdjointOptions):
            raise ValueError('Mode networks require resident AdjointOptions, not streaming.')
        from .adjoint_memory import _resident_contract
        _resident_contract(self.project.region, self.options)
        if len(self.ports) != 2 or not all(isinstance(p, FixedModePort) for p in self.ports):
            raise ValueError('Exactly two opposing fixed mode ports are supported.')
        self._shared_section = port_permittivities is None
        if port_permittivities is None:
            if permittivity is None:
                raise ValueError('Provide permittivity or explicit port_permittivities.')
            sections = {p.name: permittivity for p in self.ports}
        else:
            if permittivity is not None:
                raise ValueError('Use either permittivity or port_permittivities, not both.')
            if not isinstance(port_permittivities, dict) or set(port_permittivities) != {p.name for p in self.ports}:
                raise ValueError('port_permittivities must map exactly both port names to fixed sections.')
            sections = dict(port_permittivities)
        left, right = self.ports
        if len({p.name for p in self.ports}) != 2 or any(not p.name for p in self.ports):
            raise ValueError('Port names must be unique and nonempty.')
        if (left.direction, right.direction) != (1, -1):
            raise ValueError('Ports must be ordered left(+ inward), right(- inward).')
        if not left.source_coordinate_um < left.coordinate_um < right.coordinate_um < right.source_coordinate_um:
            raise ValueError('Source planes must lie outside ordered detector phase planes.')
        if any(not p.mode_indices or len(set(p.mode_indices)) != len(p.mode_indices)
               or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < num_modes for i in p.mode_indices)
               for p in self.ports):
            raise ValueError('Select unique mode indices inside num_modes.')
        if not math.isfinite(gram_tolerance) or not 0 < gram_tolerance <= .01:
            raise ValueError('gram_tolerance must lie in (0,.01].')
        if isinstance(network_budget_bytes, bool) or not isinstance(network_budget_bytes, int) or network_budget_bytes <= 0:
            raise ValueError('network_budget_bytes must be a positive integer.')
        active = [self.project.resolved_source(s) for s in self.project.sources if s.enabled]
        if len(active) != 1:
            raise ValueError('Use one Gaussian plane source as the launch timing template.')
        self.normal = active[0].normal
        self.axis = 'xyz'.index(self.normal)
        self.gram_tolerance, self.network_budget_bytes = gram_tolerance, network_budget_bytes
        r = self.project.region
        nodes = r.mesh_nodes[self.axis]
        self.indices = tuple(int(np.argmin(abs(nodes-p.coordinate_um))) for p in self.ports)
        for p, index in zip(self.ports, self.indices):
            if not math.isclose(float(nodes[index]), p.coordinate_um, rel_tol=0, abs_tol=1e-7):
                raise ValueError('Port phase planes must lie on E nodes.')
        if self.indices[1]-self.indices[0] < 4:
            raise ValueError('Leave at least four cells between phase planes.')
        self.channels = tuple((p.name, i) for p in self.ports for i in p.mode_indices)
        self._admit()
        projects, launches = [], []
        for port in self.ports:
            for mode_index in port.mode_indices:
                p = self.project.model_copy(deep=True)
                source = active[0].model_copy(deep=True)
                center = [0., 0., 0.]; center[self.axis] = port.source_coordinate_um
                size = list(r.actual_size); size[self.axis] = 0.
                if open_ports is not None:
                    for a in range(3):
                        if a != self.axis:
                            lo,hi = r.interior_bounds(a)
                            center[a],size[a] = (lo+hi)/2,hi-lo
                source.center, source.size = tuple(center), tuple(size)
                source.direction = '+' if port.direction == 1 else '-'
                p.sources = [source]
                p.monitors = [FieldMonitor(id=q.name, normal=self.normal,
                    center=tuple(q.coordinate_um if a == self.axis else center[a] for a in range(3)),
                    size=tuple(size)) for q in self.ports]
                if open_ports is None:
                    launch = prepare_modal_launch(p, _fixed_section(sections[port.name]), mode_index=mode_index, num_modes=num_modes,
                                                  source_budget_bytes=network_budget_bytes)
                else:
                    from .open_mode_injection import prepare_open_modal_launch
                    launch = prepare_open_modal_launch(p, _fixed_section(sections[port.name]),options=open_ports,
                        mode_index=mode_index,num_modes=num_modes,source_budget_bytes=network_budget_bytes)
                projects.append(p); launches.append(launch)
        self._projects, self._launches = tuple(projects), tuple(launches)
        self._packet_bytes = sum(x.storage_bytes for x in launches)
        self._admit()
        if open_ports is not None:
            # Check the complete signed basis on the actual detector rule
            # before creating any volume fields or running a calibration.
            from types import SimpleNamespace
            from .field_monitors import plane_plan
            offset = 0
            for port in self.ports:
                monitor = next(m for m in projects[0].monitors if m.id == port.name)
                plan = plane_plan(r, monitor)
                plane = SimpleNamespace(normal=self.normal,
                    points_um=torch.tensor(plan['points_um'], dtype=torch.float32),
                    points_um_geometry=torch.tensor(plan['points_um'], dtype=torch.float64),
                    weights=torch.tensor(plan['weights'], dtype=torch.float32),
                    fields=torch.zeros((1,len(plan['weights']),6), dtype=torch.complex64))
                count = len(port.mode_indices)
                _decompose(plane, launches[offset:offset+count], self.gram_tolerance)
                offset += count
        self._project_snapshot = self.project.model_dump()
        self._configuration_snapshot = self._configuration()
        self._prepared_snapshot = tuple(p.model_dump() for p in self._projects)

    def _configuration(self):
        return (self.ports, self.normal, self.axis, self.indices, self.channels, self.options,
                self.gram_tolerance, self.network_budget_bytes, self._shared_section,self.open_ports,
                tuple(launch.identity for launch in self._launches))

    def _guard_configuration(self):
        if self.project.model_dump() != self._project_snapshot:
            raise ValueError('Mode network project changed. Rebuild fixed ports and launches.')
        if (self._configuration() != self._configuration_snapshot
                or tuple(p.model_dump() for p in self._projects) != self._prepared_snapshot):
            raise ValueError('Mode network configuration changed. Rebuild fixed ports and launches.')

    def _admit(self):
        r = self.project.region
        item = 8 if r.precision == 'float64' else 4
        cells = math.prod(r.shape)
        points = math.prod(n for a, n in enumerate(r.shape) if a != self.axis)
        count = len(self.channels)
        # Fixed packets for all launches, parameter carrier and gradients,
        # reference epsilon, one plane-map construction and modal scratch.
        packet = getattr(self, '_packet_bytes', count*(8*r.steps+80*points)*item)
        calibration_bytes = 3*cells*item
        required = packet + 21*cells*item + calibration_bytes + 8192*points + 128*count*points*item + 16*count*count*item
        if required > self.network_budget_bytes:
            raise ValueError('Mode network packet/wrapper reservation exceeds network byte budget.')
        from .memory_profile import host_memory
        available = host_memory()['available_bytes']
        if available is not None and required > .8*available:
            raise ValueError('Mode network wrapper reservation exceeds available host memory.')
        return required

    def _port_profile(self, port_index, device):
        offset = 0 if port_index == 0 else len(self.ports[0].mode_indices)
        launch = self._launches[offset]
        cross = np.stack([_profile(launch.epsilon[..., c], self.normal) for c in range(3)], -1)
        return torch.tensor(cross, device=device)

    def reference_epsilon(self, *, port=None, device='cpu'):
        """A straight guide for one incident port, not the unequal-port device."""
        self._guard_configuration()
        self._admit()
        if port is None:
            if not self._shared_section:
                raise ValueError('Select port by name for a per-port calibration guide.')
            index = 0
        else:
            names = tuple(p.name for p in self.ports)
            if port not in names:
                raise ValueError('Unknown calibration port name.')
            index = names.index(port)
        # No full-volume NumPy temporary or retained reference per port.
        return self._port_profile(index, device).expand(self.project.region.shape+(3,)).clone()

    def __call__(self, epsilon, *, output_device='cpu', output_budget_bytes=64*1024**2):
        self._guard_configuration()
        reserved = self._admit()
        r = self.project.region
        if isinstance(output_budget_bytes, bool) or not isinstance(output_budget_bytes, int) or output_budget_bytes <= 0:
            raise ValueError('output_budget_bytes must be a positive integer.')
        output_required = len(self.channels)**2 * (16 if r.precision == 'float64' else 8)
        if output_required > output_budget_bytes:
            raise ValueError('S matrix exceeds output byte budget before calibration.')
        if not isinstance(epsilon, torch.Tensor) or epsilon.shape != r.shape+(3,):
            raise ValueError('Network epsilon must be explicit diagonal Yee material.')
        if epsilon.dtype != (torch.float64 if r.precision == 'float64' else torch.float32):
            raise ValueError('Network epsilon precision must match the region.')
        if epsilon.device.type not in ('cpu', 'cuda'):
            raise ValueError('Network epsilon requires CPU or CUDA.')
        # Freeze both exterior guides once, before recomputed launch cases.
        # No full-volume mask is retained, and all columns share this carrier.
        fixed = []
        for port_index, span in enumerate((slice(None, self.indices[0]+2), slice(self.indices[1]-1, None))):
            selection = [slice(None)]*3; selection[self.axis] = span
            selection = tuple(selection)
            if not torch.allclose(epsilon[selection], self._port_profile(port_index, epsilon.device).expand_as(epsilon[selection]), rtol=2e-6, atol=1e-7):
                raise ValueError('Port/source exterior material must match the fixed calibration guide.')
            fixed.append(selection)
        material = epsilon.clone() if epsilon.requires_grad else epsilon
        if epsilon.requires_grad:
            for selection in fixed:
                material[selection] = material[selection].detach()
        projects = tuple(p.model_copy(deep=True) for p in self._projects)
        launches, ports = self._launches, self.ports
        counts = [len(p.mode_indices) for p in ports]
        port_launches = (launches[:counts[0]], launches[counts[0]:])
        frequency = [C0/(launches[0].mode.wavelength_um*1e-6)]
        tolerance = self.gram_tolerance
        options = self.options
        cases = []
        reports = []
        for column, (project, launch) in enumerate(zip(projects, launches)):
            port_index = 0 if column < counts[0] else 1
            local_index = column if port_index == 0 else column-counts[0]
            reference = self.reference_epsilon(port=ports[port_index].name, device=epsilon.device)
            model = ModeInjectedPlaneSimulation(project, launch, options)
            with torch.no_grad():
                planes = model(reference, frequency)
                plane = planes[ports[port_index].name]
                scale = float(plane.fields.abs().max())
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError('Calibration has no finite incident field.')
                forward, backward = _decompose(replace(plane, fields=plane.fields/scale),
                    port_launches[port_index], tolerance)
                incoming = forward if ports[port_index].direction == 1 else backward
                baseline = backward if ports[port_index].direction == 1 else forward
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
                for i, (port, basis) in enumerate(zip(ports, port_launches)):
                    f, b = _decompose(replace(current[port.name], fields=current[port.name].fields/field_scale), basis, tolerance)
                    amplitude = (b if port.direction == 1 else f)[0]
                    if i == input_port:
                        amplitude = amplitude-reflection
                    outgoing.append(amplitude/norm)
                return torch.cat(outgoing)
            cases.append(case)
        columns = recompute_cases(cases, material, output_device=output_device,
                                  output_budget_bytes=output_budget_bytes)
        return ModeNetworkResult(columns.transpose(0, 1), self.channels,
            tuple(p.coordinate_um for p in ports),
            dict(scope='opposing two-port fixed-mode network', wrapper_reservation_bytes=reserved,
                 transverse_boundary='cpml' if self.open_ports is not None else 'periodic',
                 calibration_volume_bytes=3*math.prod(r.shape)*epsilon.element_size(),
                 calibration_volume_limit=1, per_port_sections=not self._shared_section,
                 calibration=reports, calibration_policy='fresh matched-guide solve per column per call', case_graphs_retained=0, backward_case_graph_limit=1,
                 phase_convention='outgoing at row phase plane / matched incident at column phase plane',
                 reflection_reference='subtract same-input-port matched-guide outgoing baseline only',
                 mode_profile_gradients=False, exterior_material_gradients='frozen'))
