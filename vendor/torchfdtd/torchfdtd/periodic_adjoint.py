"""Budgeted periodic density responses through resident or streamed adjoints."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math

import torch

from .adjoint_batch import AdjointBatchOptions, AdjointCase, RecomputedAdjointBatch
from .density_layer import periodic_density_layer
from .detector_allocation import quadrant_intensity_allocation
from .execution_tuning import AdjointExecutionPolicy
from .memory_profile import host_memory
from .periodic_response import _periodic_project
from .polarization import calibrate_plane_polarization, mix_plane_fields
from .reference_cache import PlaneReferenceCache
from .response_cache import PeriodicResponseCache
from .solver import C0


class PeriodicLayerResponse(torch.nn.Module):
    """Fixed-frequency two-polarization response with one active solver graph.

    A real CPU density tensor of density_shape produces a CPU (2, 4) response
    in R/G2/G1/B order. Mesh, materials, incidence and detector geometry are
    fixed. Coherent source synthesis precedes intensity/power reduction.
    First-order density gradients flow through both source-basis solutions.

    policy selects resident CPU/CUDA or DRAM/file spatial execution. The host
    budget includes solver replay, material-map construction and this module's
    bounded reference cache. Caller optimizer/other graphs, process/runtime
    overhead and OS file cache are excluded. Streamed policies synthesize
    density materials by slab and reduce directly to the 2D density. Resident
    policies retain their dense material path. This is not concurrent adjoint
    batching or a converged CR optimization by itself.
    """
    def __init__(self, spec, *, density_shape, policy, batch_options,
                 mesh, steps, dtype=torch.float32, pml_cells=12,
                 quadrature_counts=(24, 24), pixel_origin='cell_edges',
                 reference_cache=None, response_cache=None, forward_kernel='fused'):
        super().__init__()
        if not isinstance(policy, AdjointExecutionPolicy) or not isinstance(batch_options, AdjointBatchOptions):
            raise ValueError('Provide an AdjointExecutionPolicy and shared AdjointBatchOptions.')
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('Density dtype must be real FP32 or FP64.')
        density_shape, quadrature_counts = tuple(density_shape), tuple(quadrature_counts)
        if len(density_shape) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in density_shape):
            raise ValueError('density_shape must contain two positive integer counts.')
        if len(quadrature_counts) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in quadrature_counts):
            raise ValueError('Use at least two detector quadrature points per axis.')
        if pixel_origin not in ('cell_edges', 'sample_centers'):
            raise ValueError('Unknown density pixel origin.')
        if forward_kernel not in ('torch', 'fused'):
            raise ValueError('forward_kernel must be torch or fused.')
        period = spec['period_um']
        if len(period) != 2 or any(isinstance(v, torch.Tensor) or not math.isfinite(v) or v <= 0 for v in period):
            raise ValueError('Period must contain two fixed positive finite lengths.')
        if any(isinstance(spec[k], torch.Tensor) or not math.isfinite(spec[k]) or spec[k] < 1
               for k in ('background_index', 'design_index')):
            raise ValueError('Fixed lossless indices must be finite and at least one.')
        if reference_cache is not None and not isinstance(reference_cache, PlaneReferenceCache):
            raise ValueError('Expected PlaneReferenceCache.')
        if response_cache is not None and not isinstance(response_cache, PeriodicResponseCache):
            raise ValueError('Expected PeriodicResponseCache.')
        recorded = policy.recorded is not None
        if recorded and dtype != torch.float32:
            raise ValueError('Recorded periodic responses require FP32 density and material maps.')
        self._response_cache = response_cache
        self._response_budget = 0 if response_cache is None else response_cache.budget_bytes
        self._response_entries = 0 if response_cache is None else response_cache.max_entries
        self._spec = deepcopy(spec)
        self._density_shape, self._dtype, self._pixel_origin = density_shape, dtype, pixel_origin
        self._cache = reference_cache if reference_cache is not None else PlaneReferenceCache()
        self._cache_budget = self._cache.budget_bytes
        self._host_budget = batch_options.host_budget_bytes
        self._project, self._kt = _periodic_project(self._spec, dtype=dtype,
            mesh=mesh, steps=steps, pml_cells=pml_cells, forward_kernel=forward_kernel,
            memory_mode='streamed')
        region = self._project.region
        lo, hi = region.interior_bounds(2)
        if -spec['height_um']/2 < lo or spec['height_um']/2 > hi:
            raise ValueError('Patterned layer must lie inside the non-PML region.')
        if recorded:
            from .density_layer import _layer_z_fraction
            from .reversible_cpml import _resolve_interval
            a, b = _resolve_interval(region, policy.recorded)
            scalar = torch.empty((), dtype=dtype, device='cpu')
            for component in ('Ex', 'Ey', 'Ez'):
                fraction = _layer_z_fraction(region, component, -spec['height_um']/2,
                                             spec['height_um']/2, scalar)
                support = torch.nonzero(fraction > 0, as_tuple=False).flatten()
                if support.numel() and (int(support[0]) < a or int(support[-1]) > b):
                    raise ValueError(f'{component} layer support [{int(support[0])}, {int(support[-1])}] '
                                     f'exceeds recorded reconstruction interval [{a}, {b}].')
        self._epsilon_shape = region.shape + (3,)
        self._streamed_density = policy.streamed is not None
        self._layer_settings = dict(bottom_um=-self._spec['height_um']/2,
            top_um=self._spec['height_um']/2,
            background_epsilon=self._spec['background_index']**2,
            design_epsilon=self._spec['design_index']**2, pixel_origin=self._pixel_origin)
        item = torch.empty((), dtype=dtype).element_size()
        parameter_bytes = math.prod(self._epsilon_shape)*item
        # Yee averaging constructs three component arrays and then stacks them.
        # The batch's own gradient-carrier reservation is additional to this
        # material-map/geometry allowance. Reference planes stay compact.
        nx, ny, nz = region.shape
        dx, dy = density_shape
        maps = 8*(nx*dx + ny*dy + nx*ny + nx + ny + nz)*item
        reference_copies = 4*4*(16*math.prod(quadrature_counts)+1)*item + 65536
        self._geometry_allowance = (8*dx*dy*item+64*(nx+ny+nz)*item+65536
            if self._streamed_density else 2*parameter_bytes+4*dx*dy*item+maps)
        self._reference_allowance = reference_copies + self._cache_budget
        self._response_allowance = self._response_budget + (4*dx*dy*item+1024*self._response_entries+1024 if self._response_budget else 0)
        self._outside_batch = self._geometry_allowance + self._reference_allowance + self._response_allowance
        if self._outside_batch >= self._host_budget:
            raise ValueError('Material map, geometry and reference cache exceed the host budget.')
        frequency = [C0/(spec['wavelength_um']*1e-6)]
        cases = []
        for component in ('Ex', 'Ey'):
            project = self._project.model_copy(deep=True)
            project.sources[0].component = component
            cases.append(AdjointCase(project, policy, frequency_hz=frequency,
                quadrature_counts={'incident':quadrature_counts, 'detector':quadrature_counts},
                **({'fixed_background_epsilon': self._spec['background_index']**2} if recorded else {})))
        batch_settings = replace(batch_options, host_budget_bytes=self._host_budget-self._outside_batch)
        if self._streamed_density:
            from .streamed_density import RecomputedDensityBatch
            self._batch = RecomputedDensityBatch(cases, batch_settings, layer=self._layer_settings)
        else:
            self._batch = RecomputedAdjointBatch(cases, batch_settings)
        policy_identity = asdict(policy)
        if not recorded:
            policy_identity.pop('recorded', None)  # Keep previous cache identities.
        payload = dict(version='budgeted-periodic-density-streaming-1' if self._streamed_density else 'budgeted-periodic-response-1',
            projects=[c.project.model_dump(mode='json') for c in cases],
            policy=policy_identity, frequency=frequency, quadrature_counts=quadrature_counts,
            dtype=str(dtype))
        if recorded:
            payload.update(version='recorded-periodic-response-1',
                           fixed_background_epsilon=self._spec['background_index']**2,
                           material_support='all-component-yee-overlap-inside-recorded-interval')
        # Placement/kernel policy is part of the key. Numerically different
        # execution paths must not silently reuse another path's reference.
        self._reference_key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        response_identity = dict(version='periodic-density-response-vjp-1', reference=self._reference_key,
            spec=self._spec, density_shape=density_shape, pixel_origin=pixel_origin)
        self._response_key = hashlib.sha256(json.dumps(response_identity, sort_keys=True).encode()).hexdigest()
        self._execution_device = torch.device(policy.device)
        self.last_report = None
        self._selection_report = None

    @classmethod
    def auto(cls, spec, *, density_shape, mesh, steps, gpu_budget_bytes,
             host_budget_bytes, device='cuda', checkpoints=2, max_slab_width=256,
             temporal_depth=8, state_directory=None, disk_budget_bytes=None,
             disk_free_reserve_bytes=100*1024**3, **settings):
        """Select an admitted memory tier without running calibration solves.

        Prefer resident execution, then asynchronous CUDA/DRAM slabs, then
        explicitly configured file banks. Tile width and temporal depth shrink
        together until the shared solver/geometry/reference budget is admitted.
        This capacity heuristic does not promise the fastest policy. Precision,
        physical mesh, duration and checkpoint count are never reduced.
        Inspect selection_report for the chosen policy and rejected plans.
        """
        from .differentiable import AdjointOptions
        from .streamed import StreamedAdjointOptions
        if any(key in settings for key in ('policy', 'batch_options')):
            raise ValueError('auto supplies policy and batch_options from the shared budgets.')
        if (state_directory is None) != (disk_budget_bytes is None):
            raise ValueError('File fallback requires both state_directory and disk_budget_bytes.')
        for name, value in (('max_slab_width', max_slab_width), ('temporal_depth', temporal_depth)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        device = str(torch.device(device))
        cuda = torch.device(device).type == 'cuda'
        batch = AdjointBatchOptions(host_budget_bytes=host_budget_bytes,
            gpu_budget_bytes=gpu_budget_bytes, disk_budget_bytes=disk_budget_bytes)
        resident = AdjointOptions(checkpoints=checkpoints,
            backward_kernel='fused' if cuda else 'torch', gpu_budget_bytes=gpu_budget_bytes,
            host_budget_bytes=host_budget_bytes,
            resident_budget_bytes=gpu_budget_bytes if cuda else host_budget_bytes)
        streamed = StreamedAdjointOptions(device=device, checkpoints=checkpoints,
            slab_width=max_slab_width, temporal_depth=temporal_depth,
            tile_transfers='async' if cuda else 'sync', gpu_budget_bytes=gpu_budget_bytes,
            host_budget_bytes=host_budget_bytes, state_directory=state_directory,
            disk_budget_bytes=disk_budget_bytes, disk_free_reserve_bytes=disk_free_reserve_bytes)
        attempts = []
        def admit(policy, mode):
            try:
                model = cls(spec, density_shape=density_shape, mesh=mesh, steps=steps,
                    policy=policy, batch_options=batch, **settings)
                plan = model.plan()
            except ValueError as exc:
                attempts.append(dict(mode=mode, policy=asdict(policy), admitted=False, reason=str(exc)))
                return None
            attempts.append(dict(mode=mode, policy=asdict(policy), admitted=True,
                reservation_bytes={key:plan[key] for key in (
                    'host_reservation_bytes', 'gpu_reservation_bytes', 'disk_reservation_bytes')}))
            model._selection_report = dict(mode=mode, policy=asdict(policy), attempts=attempts,
                calibration_solves=0, scope='Capacity-first heuristic using live admission. No performance optimality claim.')
            return model
        selected = admit(AdjointExecutionPolicy(resident=resident, device=device,
            host_budget_bytes=host_budget_bytes), 'resident')
        if selected is not None:
            return selected
        project, _ = _periodic_project(spec, dtype=settings.get('dtype', torch.float32),
            mesh=mesh, steps=steps, pml_cells=settings.get('pml_cells', 12),
            forward_kernel=settings.get('forward_kernel', 'fused'), memory_mode='streamed')
        width = min(max_slab_width, project.region.shape[0])
        tiles = []
        while True:
            # Keep halos bounded relative to useful width as tiles shrink.
            depth = min(temporal_depth, steps, max(1, width//2))
            tiles.append((width, depth))
            if width == 1:
                break
            width = max(1, width//2)
        for storage, mode in (('host', 'dram'), ('disk', 'file')):
            if storage == 'disk' and state_directory is None:
                continue
            for width, depth in tiles:
                options = replace(streamed, state_storage=storage, slab_width=width, temporal_depth=depth)
                selected = admit(AdjointExecutionPolicy(streamed=options, device=device,
                    host_budget_bytes=host_budget_bytes), mode)
                if selected is not None:
                    return selected
        raise ValueError('No periodic execution policy fits the shared budgets: '+json.dumps(attempts))

    @property
    def selection_report(self):
        """Automatic planning evidence, or None for an explicit policy."""
        return deepcopy(self._selection_report)

    @property
    def project(self):
        """Return an inspection copy. Rebuild the module to change the project."""
        return self._project.model_copy(deep=True)

    def plan(self):
        """Live admission without full material maps or field allocations."""
        if self._response_cache is not None and (self._response_cache.budget_bytes != self._response_budget
                or self._response_cache.max_entries != self._response_entries):
            raise ValueError('Response-cache budget or entry limit changed. Rebuild the response module.')
        if self._cache.budget_bytes != self._cache_budget:
            raise ValueError('Reference-cache budget changed. Rebuild the response module.')
        input_shape = self._density_shape if self._streamed_density else self._epsilon_shape
        logical = torch.zeros((), dtype=self._dtype).expand(input_shape)
        batch = self._batch.plan(logical)
        total = batch['host_reservation_bytes'] + self._outside_batch
        available = host_memory()['available_bytes']
        limit = self._host_budget if available is None else min(self._host_budget, int(.8*available))
        if total > limit:
            raise ValueError('Periodic solver, geometry and references exceed the shared host budget.')
        return dict(batch=batch, host_reservation_bytes=total,
            gpu_reservation_bytes=batch['gpu_reservation_bytes'],
            disk_reservation_bytes=batch['disk_reservation_bytes'],
            geometry_allowance_bytes=self._geometry_allowance,
            reference_allowance_bytes=self._reference_allowance,
            response_cache_allowance_bytes=self._response_allowance,
            density_shape=self._density_shape, epsilon_shape=self._epsilon_shape,
            material_input='streamed_density' if self._streamed_density else 'dense_epsilon',
            dense_material_storage_bytes=0 if self._streamed_density else math.prod(self._epsilon_shape)*torch.empty((),dtype=self._dtype).element_size(),
            scope='One invocation. Excludes caller optimizer/other graphs, runtime and OS cache.')

    def _references(self):
        def compute():
            background = (torch.zeros(self._density_shape, dtype=self._dtype) if self._streamed_density else
                torch.full(self._epsilon_shape, self._spec['background_index']**2, dtype=self._dtype))
            result = self._batch(background)
            return {f'{basis}/{name}':plane for basis, planes in enumerate(result.cases)
                    for name, plane in planes.items()}
        cached = self._cache._get(self._reference_key, 'cpu', compute)
        return [{name:cached[f'{basis}/{name}'] for name in ('incident', 'detector')} for basis in range(2)]

    def forward(self, density):
        if torch.is_inference_mode_enabled():
            raise ValueError('Use torch.no_grad(), not inference_mode, for periodic adjoints.')
        if not isinstance(density, torch.Tensor) or density.device.type != 'cpu' or density.dtype != self._dtype or tuple(density.shape) != self._density_shape:
            raise ValueError('Density must be a real CPU tensor matching the prepared shape and dtype.')
        if not bool(torch.isfinite(density).all()) or bool(((density < 0) | (density > 1)).any()):
            raise ValueError('Density must be finite in [0,1].')
        if self._response_cache is None or not self._response_budget:
            return self._compute_response(density)
        plan = self.plan()
        namespace = self._response_namespace()
        computed = [False]
        def compute(value):
            computed[0] = True
            return self._compute_response(value)
        result = self._response_cache._evaluate(namespace, density, compute, self._validate_response_cache)
        statistics = self._response_cache.statistics()
        if not computed[0]:
            self.last_report = dict(plan=plan, batch=None, selection=self.selection_report, skipped_forward_solver=True)
        self.last_report.update(response_cache_at_forward_return=statistics)
        return result

    def _response_namespace(self):
        device = self._execution_device
        index = torch.cuda.current_device() if device.type == 'cuda' and device.index is None else device.index
        return (self._response_key, device.type, index, torch.get_num_threads(),
            torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled())

    def _validate_response_cache(self, namespace):
        self.plan()
        if namespace != self._response_namespace():
            raise RuntimeError('Periodic execution context changed between cached forward and backward.')

    def _compute_response(self, density):
        plan = self.plan()
        with torch.no_grad():
            hits, misses = self._cache.hits, self._cache.misses
            refs = self._references()
            theta, phi = self._spec['theta_inside_rad'], self._spec['phi_rad']
            pvec = density.new_tensor([math.cos(theta)*math.cos(phi), math.cos(theta)*math.sin(phi)])
            svec = density.new_tensor([-math.sin(phi), math.cos(phi)])
            targets = [math.cos(phi)*pvec-math.sin(phi)*svec, math.sin(phi)*pvec+math.cos(phi)*svec]
            coefficients = [calibrate_plane_polarization([r['incident'] for r in refs], [self._kt], v[None])
                            for v in targets]
        # Revalidate after reference computation, before material production.
        self.plan()
        material = density if self._streamed_density else periodic_density_layer(
            density, self._project.region, **self._layer_settings)
        result = self._batch(material)
        planes = [p['detector'] for p in result.cases]
        responses = []
        for c in coefficients:
            sample = mix_plane_fields(planes, c)
            reference = mix_plane_fields([r['detector'] for r in refs], c)
            responses.append(quadrant_intensity_allocation(sample, sample.normalized_flux(reference))[0])
        self.last_report = dict(plan=plan, batch=result.report, selection=self.selection_report,
            reference_cache_hits=self._cache.hits-hits, reference_cache_misses=self._cache.misses-misses,
            reference_cache_tensor_bytes=self._cache.tensor_bytes)
        return torch.stack(responses)
