"""Explicit CPU-design bridge to fixed-exterior recorded CPML plane adjoints."""
from dataclasses import replace
from types import SimpleNamespace
import copy
import math

import torch

from .reversible_cpml import ReversibleCPMLOptions
from .reversible_cpml_planes import ReversibleCPMLPlaneSimulation


def _background(value):
    if type(value) is not float or not math.isfinite(value) or value < 1:
        raise ValueError('Recorded execution requires a fixed finite scalar background epsilon >= 1.')
    return float(value)


def _settings(project, policy, background, *, dispersive=False):
    from .plane_execution import plane_mode
    options = getattr(policy, 'recorded', None)
    if not isinstance(options, ReversibleCPMLOptions):
        raise ValueError('Recorded execution requires ReversibleCPMLOptions.')
    if getattr(policy, 'resident', None) is not None or getattr(policy, 'streamed', None) is not None:
        raise ValueError('Recorded execution requires exactly one algorithm policy.')
    if dispersive or any(m.model != 'dielectric' or m.oscillators for m in project.materials):
        raise ValueError('Recorded execution supports nondispersive dielectric planes only.')
    if project.region.precision != 'float32' or not plane_mode(project):
        raise ValueError('Recorded execution requires FP32 fixed field planes.')
    device = torch.device(policy.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Recorded execution requires CPU or CUDA.')
    if type(policy.host_budget_bytes) is not int or policy.host_budget_bytes <= 0:
        raise ValueError('Unified host budget must be a positive integer.')
    return options, device, _background(background)


def _make_plane(project, policy, quadrature_counts):
    from .execution_tuning import _resident_project
    from .streamed_geometry import _plane_construction_preflight
    prepared = _resident_project(project, policy.recorded)
    limit = policy.host_budget_bytes
    if policy.recorded.host_budget_bytes is not None:
        limit = min(limit, policy.recorded.host_budget_bytes)
    if torch.device(policy.device).type == 'cpu' and policy.recorded.resident_budget_bytes is not None:
        limit = min(limit, policy.recorded.resident_budget_bytes)
    # Existing metadata-only bound rejects layout construction before dense
    # quadrature/interpolation arrays, independently of later solver admission.
    _plane_construction_preflight(prepared, SimpleNamespace(host_budget_bytes=limit), quadrature_counts)
    return ReversibleCPMLPlaneSimulation(prepared, policy.recorded,
                                         quadrature_counts=quadrature_counts)


def _recorded_reservation(project, shapes, policy, frequency_hz=None, window=None,
                          block_size=32, *, quadrature_counts=None, plane=None,
                          fixed_background_epsilon=None):
    """Complete core and CPU-interface admission before input/background copies."""
    from .memory_profile import host_memory
    from .cuda_memory import cuda_budget_limit
    options, device, background = _settings(project, policy, fixed_background_epsilon)
    shapes = tuple(tuple(s) for s in shapes)
    if len(shapes) != 1 or shapes[0] not in (project.region.shape, (*project.region.shape, 3)):
        raise ValueError('Recorded execution requires one scalar or diagonal epsilon shape matching the grid.')
    if frequency_hz is None:
        raise ValueError('Recorded fixed-plane execution requires frequency_hz.')
    if window is not None:
        raise ValueError('Recorded fixed-plane execution does not support a temporal window.')
    plane = _make_plane(project, policy, quadrature_counts) if plane is None else plane
    if not isinstance(plane, ReversibleCPMLPlaneSimulation):
        raise ValueError('Recorded planning requires a recorded CPML plane model.')
    from .execution_tuning import _resident_project
    if (plane.project.model_dump() != _resident_project(project, options).model_dump()
            or plane.model.options != options):
        raise ValueError('Recorded plane model does not match the project/policy snapshot.')
    result = plane.plan(frequency_hz, device=device,
        material_components=3 if len(shapes[0]) == 4 else 1, block_size=block_size)
    parameters = math.prod(shapes[0])*4
    output = result['plane_output_bytes']
    # Six complex components per point/frequency. Result metadata is real FP32
    # points/weights and one frequency vector per plane. Core already accounts
    # device copies of these; this bridge adds their CPU copies on CUDA only.
    points = sum(len(plan['weights']) for _, _, plan, _ in plane.plans)
    if not points or output % (6*8*points):
        raise ValueError('Invalid recorded plane output metadata accounting.')
    frequencies = output//(6*8*points)
    metadata = (4*points+len(plane.plans)*frequencies)*4
    cuda = device.type == 'cuda'
    gpu_transfer = 2*parameters if cuda else 0
    host_transfer = 2*parameters+2*output+metadata if cuda else 0
    generated_gpu = parameters if cuda else 0
    generated_host = 0 if cuda else parameters
    # Core already charges effective assembly/CopySlices; the generated fixed
    # input is a separate allocation, with no retained per-adapter GPU cache.
    extra_gpu = gpu_transfer+generated_gpu
    headroom = (extra_gpu+19)//20+4096 if cuda else 0
    gpu = result['gpu_reservation_bytes']+extra_gpu+headroom
    host = result['host_reservation_bytes']+host_transfer+generated_host
    active = gpu if cuda else host
    if host > policy.host_budget_bytes:
        raise ValueError('Recorded solver, background and copies exceed the unified host budget.')
    if options.host_budget_bytes is not None and host > options.host_budget_bytes:
        raise ValueError('Recorded solver, background and copies exceed the explicit host budget.')
    if options.resident_budget_bytes is not None and active > options.resident_budget_bytes:
        raise ValueError('Recorded solver, background and copies exceed the resident byte budget.')
    available = host_memory()['available_bytes']
    if available is not None and host > int(.8*available):
        raise ValueError('Recorded solver, background and copies exceed available host memory.')
    if cuda and gpu > cuda_budget_limit(device, gpu, options.gpu_budget_bytes):
        raise ValueError('Recorded solver, background and copies exceed the GPU budget.')
    return dict(result, memory_reservation_bytes=active, host_reservation_bytes=host,
        gpu_reservation_bytes=gpu, host_transfer_reservation_bytes=host_transfer,
        gpu_transfer_reservation_bytes=gpu_transfer, generated_background_bytes=parameters,
        generated_background_host_bytes=generated_host, generated_background_gpu_bytes=generated_gpu,
        bridge_allocation_headroom_bytes=headroom,
        allocation_headroom_bytes=result['allocation_headroom_bytes']+headroom,
        plane_result_metadata_bytes=metadata, fixed_background_epsilon=background,
        unified_execution='recorded_cpml', host_input_interface=True,
        owned_device_input_copy_bytes=parameters if cuda else 0,
        owned_device_incoming_gradient_bytes=parameters if cuda else 0,
        generated_background_cached=False)


class _RecordedPlanesFromHost(torch.nn.Module):
    def __init__(self, project, policy, dispersive=False, quadrature_counts=None, *,
                 fixed_background_epsilon=None):
        super().__init__()
        _, _, background = _settings(project, policy, fixed_background_epsilon,
                                      dispersive=dispersive)
        self.policy = policy
        self.fixed_background_epsilon = background
        self.quadrature_counts = copy.deepcopy(quadrature_counts)
        self.model = _make_plane(project, policy, self.quadrature_counts)
        self.project = self.model.project
        self._configuration = (copy.deepcopy(policy), background,
            copy.deepcopy(self.quadrature_counts), self.project.model_dump(),
            self.model.model.project.model_dump())

    def check_fixed(self):
        current = (self.policy, self.fixed_background_epsilon, self.quadrature_counts,
                   self.project.model_dump(), self.model.model.project.model_dump())
        if (current != self._configuration or self.project is not self.model.project
                or self.model.model.options != self.policy.recorded):
            raise ValueError('Recorded execution configuration changed. Rebuild the adapter.')

    def forward(self, epsilon, *, frequency_hz=None, block_size=32):
        self.check_fixed()
        if (not isinstance(epsilon, torch.Tensor) or epsilon.device.type != 'cpu'
                or epsilon.dtype != torch.float32 or epsilon.layout != torch.strided
                or not epsilon.is_contiguous() or epsilon.is_conj() or epsilon.is_neg()
                or tuple(epsilon.shape) not in (self.project.region.shape, (*self.project.region.shape, 3))):
            raise ValueError('Recorded execution requires a resolved contiguous real CPU FP32 epsilon map.')
        reservation = _recorded_reservation(self.project, (tuple(epsilon.shape),), self.policy,
            frequency_hz, block_size=block_size, plane=self.model,
            quadrature_counts=self.quadrature_counts,
            fixed_background_epsilon=self.fixed_background_epsilon)
        # Copies are differentiable. Only the explicit exterior scalar is fixed.
        copied = epsilon.to(self.policy.device)
        background = torch.full_like(copied, self.fixed_background_epsilon, requires_grad=False)
        planes = self.model(copied, frequency_hz, fixed_epsilon=background, block_size=block_size)
        output = {}
        for key, plane in planes.items():
            plane.report.update(unified_execution='recorded_cpml', host_input_interface=True,
                fixed_background_epsilon=self.fixed_background_epsilon,
                generated_background_cached=False, execution_reservation=reservation)
            output[key] = replace(plane, fields=plane.fields.cpu(), frequency_hz=plane.frequency_hz.cpu(),
                                  points_um=plane.points_um.cpu(), weights=plane.weights.cpu())
        return output
