"""Spatially streamed full-tensor Yee/CPML slabs on the nodal constitutive operator.

One tensor time step reaches two cells along x: the backward curl reads one
row below, the node assembly S scatters one row further, and the forward
curl of the magnetic update cancels one of them only for a pointwise update.
Every tile therefore carries a 2K-cell halo for K steps, twice the diagonal
halo, and the tile operator treats x as a finite axis whose end-row closure
differs from the resident operator only inside that halo. Node tensors have
no Bloch phase; fields and CPML memories keep the existing winding extension.
Material rows and their tensor VJP rows are gathered and reduced by index.
"""
from dataclasses import dataclass

import torch

from .anisotropy import TensorConstitutive, _TensorSystem, _TensorValidation, _pec_faces
from .boundaries import BoundaryDescription
from .cuda_memory import cuda_budget_limit
from .differentiable import DifferentiableResult
from .memory_profile import host_memory
from .spacetime import SlabBlockOperator
from .streamed import StreamedSimulation, StreamedAdjointOptions, _Streamed, _StreamedExecution, _reservation


def _tensor_reservation(project, epsilon, options, spectral=None):
    """Diagonal streamed reservation plus the doubled halo and tensor tile workspace."""
    if epsilon.ndim != 5:
        raise ValueError('Streamed tensor reservation needs node tensors of shape (Nx,Ny,Nz,3,3).')
    base = _reservation(project, epsilon, options, spectral)
    region = project.region
    item = epsilon.element_size()*(2 if region.complex_fields else 1)
    depth = min(options.temporal_depth, region.steps)
    periodic = 0 in BoundaryDescription(region).wrap
    def rows(halo):
        width = min(options.slab_width, region.shape[0])+2*halo
        return width if periodic else min(width, region.shape[0])
    tile_cells = rows(2*depth)*region.shape[1]*region.shape[2]
    if (6 if region.complex_fields else 3)*tile_cells >= 2**31:
        raise ValueError('A tensor CUDA tile exceeds the supported integer index range.')
    added_rows = (rows(2*depth)-rows(depth))*region.shape[1]*region.shape[2]
    buffers = options.tile_buffers if options.tile_transfers == 'async' else 1
    # Base workspace for the extra halo rows, then per tile cell: nine node
    # coefficients and their inverse, eight-triplet gather/scatter scratch,
    # the tensor VJP accumulator and its outer products.
    extra = buffers*(128*added_rows+192*tile_cells)*item
    # Cold CUDA linear-algebra/allocator allowance for the tile inverses, as in
    # the resident tensor path; not a measured peak.
    library = 64*1024**2 if torch.device(options.device).type == 'cuda' else 0
    host = base['host_reservation_bytes']+extra
    gpu = base['gpu_reservation_bytes']+(extra+library if torch.device(options.device).type == 'cuda' else 0)
    available = host_memory()['available_bytes']
    host_limit = min(options.host_budget_bytes, int(available*.8)) if available is not None else options.host_budget_bytes
    if host > host_limit:
        raise ValueError(f'Streamed tensor reservation exceeds the host budget: required={host} bytes, admissible={host_limit} bytes.')
    if torch.device(options.device).type == 'cuda':
        gpu_limit = cuda_budget_limit(options.device, gpu, options.gpu_budget_bytes)
        if gpu > gpu_limit:
            raise ValueError(f'Streamed tensor tile reservation exceeds the GPU budget: required={gpu} bytes, admissible={gpu_limit} bytes.')
    return dict(base, host_reservation_bytes=host, gpu_reservation_bytes=gpu, halo_cells_per_side=2*depth,
                max_extended_tile_cells=tile_cells, tensor_tile_reservation_bytes=extra,
                tensor_cuda_library_allowance_bytes=library)


def estimate_streamed_tensor_memory(project, options=None, *, frequency_hz=None, window=None):
    """Admission of a streamed full-tensor policy without domain allocation."""
    options = options or StreamedAdjointOptions()
    epsilon = torch.empty(project.region.shape+(3, 3), dtype=getattr(torch, project.region.precision), device='meta')
    if window is not None and frequency_hz is None:
        raise ValueError('A spectral window requires frequency_hz.')
    spectral = None
    if frequency_hz is not None:
        from .adjoint_spectrum import SpectralObservation
        scalar = torch.empty((), dtype=epsilon.dtype, device='cpu')
        spectral = SpectralObservation(scalar, project.region,
                                       [m.component for m in project.monitors if m.enabled], frequency_hz, window)
    reservation = _tensor_reservation(project, epsilon, options, spectral)
    if spectral is not None:
        reservation.update(spectral.reservation(min(options.temporal_depth, project.region.steps)))
    return reservation


class TensorSlabBlockOperator(SlabBlockOperator):
    """Slab blocks whose tiles run the explicit tensor update and its transpose."""
    def tiles(self, depth):
        n = self.host.region.shape[0]
        halo = 2*depth
        for lo in range(0, n, self.width):
            hi = min(lo+self.width, n)
            begin, end = lo-halo, hi+halo
            if 0 not in self.host.grid.wrap:
                begin, end = max(0, begin), min(n, end)
            coordinates = torch.arange(begin, end, dtype=torch.int64)
            yield lo, hi, coordinates.remainder(n), slice(lo-begin, hi-begin)

    def _new_local(self):
        return object.__new__(_TensorSystem)

    def _extra_payload(self, local, material, state, rows, phase, mapping, descriptor):
        _, _, indices, _ = descriptor
        lower, upper = _pec_faces(self.host.region)[0]
        n = self.host.region.shape[0]
        local.x_walls = (lower and int(indices[0]) == 0, upper and int(indices[-1]) == n-1)
        local.fixed_collar = False
        return ()

    def _prepare_kernel(self, local):
        pass

    def _prepare_permittivity(self, local):
        host = self.host
        walls = (local.x_walls, *_pec_faces(host.region)[1:])
        local.operator = TensorConstitutive(local.epsilon, (1., *(host.grid.wrap.get(a, 1.) for a in (1, 2))),
                                            (False, *(a in host.grid.wrap for a in (1, 2))), walls)

    def _backward(self, local, gradient, samples):
        return None


@dataclass(frozen=True)
class _TensorExecution(_StreamedExecution):
    operator_type = TensorSlabBlockOperator

    def host(self, project, value, spectral):
        return _TensorSystem(project, value, None if spectral is None else spectral.observers, prepare_updates=False)

    def reservation(self, project, value, options, spectral):
        return _tensor_reservation(project, value, options, spectral)


class StreamedTensorSimulation(_TensorValidation, StreamedSimulation):
    """Streamed first-order derivatives of node-sampled full permittivity tensors.

    Same input contract as TensorDielectricSimulation with cpml_material='tensor'
    (CPU node tensors of shape (Nx,Ny,Nz,3,3), eigenvalues at least one, the
    geometric criterion on every CPML face, periodic/Bloch/PEC/CPML faces, soft
    sources, point or spectral observations). Tiles run the Torch tensor update
    and its explicit transpose on the selected device with a 2K halo. The fixed
    isotropic collar contract is not offered here; it is a resident-only path.
    """
    _streamed = True

    def __init__(self, project, options=None, *, cpml_material='tensor'):
        if cpml_material != 'tensor':
            raise ValueError("Streamed tensor media support cpml_material='tensor' only.")
        self.cpml_material = cpml_material
        self.cpml_background_epsilon = None
        super().__init__(project, options)
        self._validate_project()

    def _run(self, epsilon, spectral):
        region = self.project.region
        if not isinstance(epsilon, torch.Tensor) or epsilon.device.type != 'cpu':
            raise ValueError('Streamed epsilon must be a CPU tensor to avoid full-volume VRAM allocation.')
        self._validate_input_shape(epsilon)
        options = self.streaming_options
        reservation = _tensor_reservation(self.project, epsilon, options, spectral)
        with torch.no_grad():
            self._validate_epsilon(epsilon)
        report = dict(experimental=True, spatial_streaming=True, full_time_autograd=False,
                      complex_fields=region.complex_fields, higher_order=False, slab_width=options.slab_width,
                      temporal_depth=options.temporal_depth, checkpoint_capacity=options.checkpoints,
                      local_checkpoint_capacity=options.local_checkpoints, state_storage=options.state_storage,
                      execution_device=options.device, tile_transfers=options.tile_transfers,
                      tile_buffers=options.tile_buffers if options.tile_transfers == 'async' else 1,
                      cuda_binding=options.cuda_binding, reuse_tile_buffers=options.reuse_tile_buffers,
                      policy='manual', precision=str(epsilon.dtype),
                      adjoint='space-time tiled full-tensor Yee/CPML explicit transpose',
                      forward_backend='torch '+torch.device(options.device).type.upper(),
                      backward_backend='torch explicit transpose',
                      tensor_sampling='common mesh nodes, normalized finite/periodic edge triplets',
                      cpml_contract='D-field composition; tensors extend into CPML where each face normal is a principal '
                                    'axis with a non-intermediate eigenvalue', cpml_collar_material_vjp='full tensor VJP in every node',
                      pec_walls=_pec_faces(region), tile_halo='2K cells per side for K steps', **reservation)
        report['observation_storage'] = 'time_history' if spectral is None else 'online_spectrum'
        if spectral is not None:
            report.update(spectral.reservation(min(options.temporal_depth, region.steps)))
        signals = _Streamed.apply(epsilon, self.project, options, report, spectral, _TensorExecution())
        if spectral is not None:
            return spectral.result(signals, report)
        return DifferentiableResult(signals, region.time_step,
                                    tuple(m.component for m in self.project.monitors if m.enabled), report)
