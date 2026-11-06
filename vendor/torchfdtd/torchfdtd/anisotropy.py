"""Experimental full-tensor Yee dielectric and discrete adjoint.

Node-sampled real symmetric epsilon >= I. Normalized incident edge triplets define
an SPD inverse constitutive operator. CPML takes either a fixed isotropic exterior or
tensors satisfying the geometric PML stability criterion on each face. PEC walls
use a nodal closure. Not anisotropic interface homogenization.
Soft sources are impressed field increments, not calibrated current sources.
"""
from dataclasses import replace
from itertools import product
import math

import torch

from .differentiable import (AdjointOptions, DifferentiableSimulation,
                            DifferentiableResult, _System, _FDTD, _Checkpoints)


def _shift(value, axis, forward, phase):
    """Read neighbor along an axis with the physical Bloch seam convention."""
    result = torch.roll(value, -1 if forward else 1, axis)
    edge = [slice(None)] * 3
    edge[axis] = -1 if forward else 0
    result[tuple(edge)] *= phase if forward else 1 / phase
    return result


def _cpml_faces(region):
    # CPML derivatives target the outer `layers` field rows. R reads an
    # incident edge at node n or n-1, so one additional node row contains
    # every tensor coefficient that can act on a CPML-supported curl.
    for axis in range(3):
        for side in range(2):
            layers = region.pml_layers(axis, side)
            if layers:
                index = [slice(None)]*3
                width = min(layers+1, region.shape[axis])
                index[axis] = slice(0, width) if side == 0 else slice(-width, None)
                yield axis, side, tuple(index)


def _cpml_collar_slices(region):
    for _, _, index in _cpml_faces(region):
        yield index


def cpml_face_admissible(epsilon, axis):
    """Geometric PML stability criterion of node tensors for a face normal to `axis`.

    Stretched-coordinate PML is unstable when a slowness sheet carries energy
    against its phase along the face normal (Becache, Fauqueux and Joly, JCP
    188, 2003). For a symmetric tensor that holds at the normal-incidence
    optic-axis cone unless the normal is a principal axis whose eigenvalue is
    not strictly between the two transverse eigenvalues. Returns a boolean
    node mask; ties are exact, nothing is clipped.
    """
    b, c = [k for k in range(3) if k != axis]
    aligned = (epsilon[..., axis, b] == 0) & (epsilon[..., axis, c] == 0)
    a_a, b_b, c_c, b_c = epsilon[..., axis, axis], epsilon[..., b, b], epsilon[..., c, c], epsilon[..., b, c]
    extreme = (a_a-b_b)*(a_a-c_c)-b_c*b_c >= 0
    return aligned & extreme


def _fold(value, axis):
    """Add a replicated ghost row back onto the last real row."""
    n = value.shape[axis]
    tail = value.narrow(axis, n-2, 1)+value.narrow(axis, n-1, 1)
    return torch.cat((value.narrow(axis, 0, n-2), tail), axis)


class TensorConstitutive:
    """Matrix-free S = sum R^dagger epsilon^-1 R / 8.

    epsilon has shape (Nx,Ny,Nz,3,3), at common mesh nodes. Caller must admit
    storage and validate epsilon first. No global sparse/dense matrix is built.
    pec[a] = (lower, upper) marks PEC walls on a finite axis. Wall-tangential
    components are zero states; each wall node keeps the admissible closure
    e_a e_a^T / epsilon_aa (its normal component), and a node on two or more
    walls contributes nothing. The upper wall is a replicated ghost node row
    whose fields are zero, so both walls share one closure.

    With inverse=False the node matrices are applied directly, giving the
    forward operator T = sum R^dagger chi R / 8 used for ADE coupling tensors.
    Its wall closure is the projection chi_aa e_a e_a^T.
    """
    def __init__(self, epsilon, phases=(1., 1., 1.), periodic=(True, True, True), pec=((False, False),)*3,
                 *, inverse=True):
        self.phases = tuple(phases)
        self.periodic = tuple(periodic)
        self.pec = tuple((bool(lo), bool(hi)) for lo, hi in pec)
        self.inverse_mode = bool(inverse)
        if any(self.periodic[a] and any(self.pec[a]) for a in range(3)):
            raise ValueError('PEC walls require a nonperiodic axis.')
        self.pad = tuple(int(self.pec[a][1]) for a in range(3))
        for axis, pad in enumerate(self.pad):
            if pad:
                epsilon = torch.cat((epsilon, epsilon.narrow(axis, epsilon.shape[axis]-1, 1)), axis)
        self.inverse = torch.linalg.inv(epsilon) if inverse else epsilon
        if any(any(p) for p in self.pec):
            shape = epsilon.shape[:3]
            walls = []
            for axis in range(3):
                index = torch.zeros(shape[axis], dtype=torch.bool, device=epsilon.device)
                index[0], index[-1] = self.pec[axis]
                view = [1, 1, 1]
                view[axis] = shape[axis]
                walls.append(index.reshape(view).expand(shape))
            count = sum(w.to(torch.int8) for w in walls)
            closure = torch.zeros_like(self.inverse)
            for axis in range(3):
                single = walls[axis] & (count == 1)
                diagonal = 1/epsilon[..., axis, axis] if inverse else epsilon[..., axis, axis]
                closure[..., axis, axis] = torch.where(single, diagonal, closure[..., axis, axis])
            self.inverse = torch.where((count > 0)[..., None, None], closure, self.inverse)

    def _normalize(self, field):
        # Terminal edges of a finite axis without an upper wall have half
        # coverage. Scaling both gather and scatter keeps S Hermitian, S <= I.
        scaled = [a for a in range(3) if not self.periodic[a] and not self.pec[a][1]]
        if not scaled:
            return field
        field = field.clone()
        for axis in scaled:
            edge = [slice(None)] * 4
            edge[axis], edge[3] = -1, axis
            field[tuple(edge)] *= math.sqrt(2.)
        return field

    def _neighbor(self, value, axis, forward):
        if self.periodic[axis]:
            return _shift(value, axis, forward, self.phases[axis])
        # Finite zero-extension and its exact transpose, never a wrap.
        result = torch.zeros_like(value)
        source, target = [slice(None)]*3, [slice(None)]*3
        source[axis] = slice(1, None) if forward else slice(None, -1)
        target[axis] = slice(None, -1) if forward else slice(1, None)
        result[tuple(target)] = value[tuple(source)]
        return result

    def _pad(self, field):
        for axis, pad in enumerate(self.pad):
            if pad:
                ghost = torch.zeros_like(field.narrow(axis, 0, 1))
                field = torch.cat((field, ghost), axis)
        return field

    def _crop(self, field):
        for axis, pad in enumerate(self.pad):
            if pad:
                field = field.narrow(axis, 0, field.shape[axis]-1)
        return field

    def gather(self, field, signs):
        field = self._normalize(field)
        return torch.stack([self._neighbor(field[..., a], a, False)
                            if signs[a] else field[..., a] for a in range(3)], -1)

    def scatter(self, field, signs):
        return self._normalize(torch.stack([self._neighbor(field[..., a], a, True)
                            if signs[a] else field[..., a] for a in range(3)], -1))

    def multiply(self, field):
        # Multiplication instead of a dtype-converting full coefficient copy.
        return sum(self.inverse[..., :, a] * field[..., a, None] for a in range(3))

    def apply(self, field):
        field = self._pad(field)
        result = torch.zeros_like(field)
        for signs in product((0, 1), repeat=3):
            result = result + self.scatter(self.multiply(self.gather(field, signs)), signs) / 8
        return self._crop(result)

    def epsilon_vjp(self, field, output_bar):
        field, output_bar = self._pad(field), self._pad(output_bar)
        gradient = torch.zeros_like(self.inverse)
        for signs in product((0, 1), repeat=3):
            right, left = self.gather(field, signs), self.gather(output_bar, signs)
            if self.inverse_mode:
                # dK = -K dEpsilon K
                right, left = self.multiply(right), self.multiply(left)
                gradient -= (left.conj()[..., :, None] * right[..., None, :]).real / 8
            else:
                gradient += (left.conj()[..., :, None] * right[..., None, :]).real / 8
        for axis, pad in enumerate(self.pad):
            if pad:
                gradient = _fold(gradient, axis)
        return (gradient + gradient.transpose(-1, -2)) / 2


def _pec_faces(region):
    return tuple(tuple(face.kind == 'pec' for face in region.boundaries.pair(a)) for a in range(3))


class _TensorSystem(_System):
    def __init__(self, project, epsilon, observation_monitors=None, *, fixed_collar=False, prepare_updates=True):
        carrier = epsilon.new_ones(()).expand(project.region.shape)
        super().__init__(project, carrier, prepare_kernels=False, prepare_permittivity=False,
                         prepare_updates=prepare_updates, observation_monitors=observation_monitors)
        self.epsilon = epsilon
        self.fixed_collar = fixed_collar
        self.operator = self.constitutive(epsilon, project.region.shape) if prepare_updates else None

    def constitutive(self, epsilon, shape, pec=None):
        return TensorConstitutive(epsilon, tuple(self.grid.wrap.get(a, 1.) for a in range(3)),
                                  tuple(a in self.grid.wrap for a in range(3)),
                                  _pec_faces(self.region) if pec is None else pec)

    def reference_step(self, state, step, epsilon):
        e, h, *psis = state
        curl, psis = self.curl(h, psis, False)
        e = self.inject(e + self.grid.courant_number * self.operator.apply(curl),
                        'E', step, functional=True)
        curl, psis = self.curl(e, psis, True)
        h = self.inject(h - self.grid.courant_number * curl, 'H', step, functional=True)
        return (e, h, *psis)

    def advance(self, start, end):
        for step in range(start, end):
            new = self.reference_step(self.state(), step, self.epsilon)
            for target, value in zip(self.state(), new):
                target.copy_(value)
        self.current_step = end

    def transpose_step(self, state, adjoint, signal_bar):
        e_bar, h_bar, *psi_bar = adjoint
        for target, (positions, indices) in zip(adjoint[:2], self.observation_maps):
            if indices.numel():
                target.reshape(-1).index_add_(0, indices, signal_bar.index_select(0, positions))
        courant = self.grid.courant_number
        contribution, psi_bar = self.curl_transpose(-courant * h_bar, psi_bar, True)
        e_bar = e_bar + contribution
        curl, _ = self.curl(state[1], state[2:], False)
        gradient = courant * self.operator.epsilon_vjp(curl, e_bar)
        if self.fixed_collar:
            for index in _cpml_collar_slices(self.region):
                gradient[index] = 0  # Fixed exterior is not a design variable.
        contribution, psi_bar = self.curl_transpose(courant * self.operator.apply(e_bar), psi_bar, False)
        return (e_bar, h_bar + contribution, *psi_bar), gradient


class _TensorValidation:
    """Project and node-tensor admission shared by the resident and streamed tensor paths."""
    _streamed = False

    def _validate_project(self):
        p, r = self.project, self.project.region
        if r.dimension != '3d' or r.mesh_type != 'uniform':
            raise ValueError('Tensor dielectric requires a uniform rectangular 3D grid.')
        if r.memory_mode == 'streamed' and not self._streamed:
            raise ValueError('Streamed tensor scenes use StreamedTensorSimulation.')
        kinds = {face.kind for a in range(3) for face in r.boundaries.pair(a)}
        if kinds & {'pmc', 'symmetric', 'antisymmetric'}:
            raise ValueError('PMC/symmetric/antisymmetric faces are not supported with tensor media; only PEC walls are.')
        if kinds - {'periodic', 'bloch', 'pml', 'pec'}:
            raise ValueError('Tensor dielectric supports periodic/Bloch, CPML or PEC faces.')
        if 'pml' in kinds:
            value = self.cpml_background_epsilon
            if self.cpml_material == 'isotropic':
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1:
                    raise ValueError("CPML requires explicit fixed cpml_background_epsilon >= 1, or cpml_material='tensor'.")
            elif value is not None:
                raise ValueError("cpml_material='tensor' takes no fixed cpml_background_epsilon.")
        if any(n < 2 for n in r.shape):
            raise ValueError('Tensor dielectric requires at least two cells on every axis.')
        if r.interface_method != 'staircase' or r.material_sampling != 'yee':
            raise ValueError('Tensor dielectric uses explicit node tensors and Yee fields, without subpixel geometry.')
        if any(m.oscillators for m in p.materials):
            raise ValueError('Full-tensor ADE is not implemented.')
        if any(s.enabled and (s.injection != 'soft' or s.kind == 'tfsf') for s in p.sources):
            raise ValueError('Only soft impressed-field sources are supported, not one-way/modal/current injection.')

    def _validate_cpml_collar(self, epsilon):
        if self.cpml_material == 'tensor':
            for axis, side, index in _cpml_faces(self.project.region):
                if not bool(cpml_face_admissible(epsilon[index], axis).all()):
                    raise ValueError(
                        'Tensor CPML face %s_%s: every node tensor in its PML layers plus one-node collar must have '
                        'the face normal as a principal axis with an eigenvalue not strictly between the two transverse '
                        'eigenvalues (geometric PML stability criterion).' % ('xyz'[axis], ('min', 'max')[side]))
            return
        if self.cpml_background_epsilon is None:
            return
        fixed = epsilon.new_tensor(self.cpml_background_epsilon)*torch.eye(3, dtype=epsilon.dtype, device=epsilon.device)
        for index in _cpml_collar_slices(self.project.region):
            if not bool((epsilon[index] == fixed).all()):
                raise ValueError('Tensor CPML requires fixed isotropic background in every PML layer plus one-node collar.')

    def _validate_input_shape(self, epsilon):
        self._validate_project()
        r = self.project.region
        if not isinstance(epsilon, torch.Tensor) or epsilon.dtype not in (torch.float32, torch.float64):
            raise ValueError('epsilon must be a real FP32/FP64 tensor.')
        if epsilon.device.type not in ('cpu', 'cuda'):
            raise ValueError('Tensor dielectric supports CPU/CUDA only.')
        if epsilon.shape != r.shape + (3, 3):
            raise ValueError('epsilon must have node-grid shape (Nx,Ny,Nz,3,3).')
        if (epsilon.dtype == torch.float64) != (r.precision == 'float64'):
            raise ValueError('epsilon precision must match the region.')

    @staticmethod
    def _eigenvalue_bounds(value):
        """Bounded-workspace extreme eigenvalues of a batch of symmetric 3x3 tensors."""
        # CUDA batched eigvalsh scratch grows far faster than the 3x3
        # coefficients. Fixed batches keep validation workspace bounded.
        lowest = torch.full((), math.inf, dtype=value.dtype, device=value.device)
        highest = torch.full((), -math.inf, dtype=value.dtype, device=value.device)
        for batch in value.reshape(-1, 3, 3).split(64):
            eigenvalues = torch.linalg.eigvalsh(batch)
            lowest = torch.minimum(lowest, eigenvalues.min())
            highest = torch.maximum(highest, eigenvalues.max())
        return lowest, highest

    def _validate_epsilon(self, epsilon):
        """Finite, exactly symmetric, eigenvalues >= 1 and any fixed collar. No inverse yet."""
        if not bool(torch.isfinite(epsilon).all()):
            raise ValueError('epsilon must be finite.')
        if not torch.equal(epsilon, epsilon.transpose(-1, -2)):
            raise ValueError('epsilon must be exactly symmetric. Construct it symmetrically.')
        lowest, _ = self._eigenvalue_bounds(epsilon)
        if bool(lowest < 1):
            raise ValueError('The conservative CFL requires eigenvalues of epsilon >= 1.')
        self._validate_cpml_collar(epsilon)


class TensorDielectricSimulation(_TensorValidation, DifferentiableSimulation):
    """Checkpointed epsilon-to-point-signals for node-sampled full tensors.

    CPU/CUDA FP32 (default) or FP64 diagnostics, uniform rectangular 3D grids, all
    axes periodic/Bloch, PEC or CPML. With cpml_material='isotropic' CPML layers
    plus one node row must equal the explicit cpml_background_epsilon times I
    and their VJP is zero. With cpml_material='tensor' they hold node tensors
    whose face normal is a principal axis with a non-intermediate eigenvalue,
    and every node has a VJP. Nondispersive tensors require eigenvalues >= 1.
    forward() and spectrum() return the native result types. First derivatives
    only. Caller optimizer/material-construction graphs are outside admission.
    host_budget_bytes bounds total host reservation for this API.
    """
    def __init__(self, project, options=None, *, cpml_background_epsilon=None, cpml_material='isotropic'):
        if cpml_material not in ('isotropic', 'tensor'):
            raise ValueError("cpml_material must be 'isotropic' (fixed collar) or 'tensor' (media extend into CPML).")
        self.cpml_background_epsilon = cpml_background_epsilon
        self.cpml_material = cpml_material
        options = options or AdjointOptions()
        if options.backward_kernel == 'fused':
            raise ValueError('Full-tensor fused kernels are not validated.')
        super().__init__(project, replace(options, backward_kernel='torch'))
        self._validate_project()
        # Region.valid_grid applies the resident size checks only to execution_mode='resident';
        # the resident contract is repeated here so that construction refuses before any allocation.
        from .adjoint_memory import _resident_contract
        _resident_contract(self.project.region, self.options)

    def reservation(self, spectral=None, *, device='cpu'):
        """Preallocation bound, including tensor coefficients and VJP workspace."""
        from .adjoint_memory import _resident_reservation
        from .memory_profile import host_memory
        device = torch.device(device)
        base = _resident_reservation(self.project, self.options, device, spectral)
        r = self.project.region
        # Caller tensor, detached inverse, inversion/eigenvalue validation,
        # material gradient and outer products, eight-triplet sequential scratch.
        # Native field/replay workspace and observation history are in base.
        extra = math.prod(r.shape) * (8 if r.precision == 'float64' else 4) * 192
        # Cold CUDA linear-algebra/allocator allowance, not a measured peak.
        library = 64 * 1024**2 if device.type == 'cuda' else 0
        extra += library
        host = base['host_reservation_bytes'] + (extra if device.type == 'cpu' else 0)
        active = base['memory_reservation_bytes'] + extra
        if self.options.host_budget_bytes is not None and host > self.options.host_budget_bytes:
            raise ValueError('Full-tensor reservation exceeds the explicit host byte budget.')
        if self.options.resident_budget_bytes is not None and active > self.options.resident_budget_bytes:
            raise ValueError('Full-tensor reservation exceeds the explicit resident byte budget.')
        available = host_memory()['available_bytes']
        if available is not None and host > .8 * available:
            raise ValueError('Full-tensor reservation exceeds available host memory.')
        if device.type == 'cuda':
            from .cuda_memory import cuda_budget_limit
            if active > cuda_budget_limit(device, active, self.options.gpu_budget_bytes):
                raise ValueError('Full-tensor reservation exceeds the GPU byte budget.')
        return dict(base, host_reservation_bytes=host,
                    memory_reservation_bytes=active,
                    gpu_reservation_bytes=active if device.type == 'cuda' else 0,
                    tensor_reservation_bytes=extra, tensor_cuda_library_allowance_bytes=library)

    def _run(self, epsilon, spectral):
        self._validate_input_shape(epsilon)
        report = self.reservation(spectral, device=epsilon.device)
        # No full-volume validation temporary or inverse before admission.
        with torch.no_grad():
            self._validate_epsilon(epsilon)
            system = _TensorSystem(self.project.model_copy(deep=True), epsilon,
                                   None if spectral is None else spectral.observers,
                                   fixed_collar=self.cpml_background_epsilon is not None)
        report.update(experimental=True, adjoint='full-tensor Yee explicit transpose',
                      higher_order=False, spatial_streaming=False, full_time_autograd=False,
                      forward_backend='torch ' + epsilon.device.type.upper(), backward_backend='torch explicit transpose',
                      tensor_sampling='common mesh nodes, normalized finite/periodic edge triplets',
                      cpml_contract='fixed isotropic exterior with one-node collar; tensor interior'
                      if self.cpml_background_epsilon is not None else
                      'D-field composition: CPML memories on the curl, S on the complete curl; tensors extend into CPML '
                      'where each face normal is a principal axis with a non-intermediate eigenvalue',
                      cpml_background_epsilon=self.cpml_background_epsilon,
                      cpml_collar_material_vjp='zero: fixed coefficients, not design variables'
                      if self.cpml_background_epsilon is not None else 'full tensor VJP in every node, including CPML',
                      pec_walls=_pec_faces(self.project.region),
                      wall_closure='wall nodes keep e_a e_a^T/epsilon_aa; tangential wall components are zero states',
                      source_contract='soft impressed-field increments',
                      steps=system.region.steps)
        admission = _Checkpoints(system, self.options, report, admission=True)
        admission.close()
        signals = _FDTD.apply(epsilon, system, self.options, report, spectral)
        if spectral is not None:
            return spectral.result(signals, report)
        return DifferentiableResult(signals, system.region.time_step,
                                    tuple(m.component for m in self.project.monitors if m.enabled), report)

    def reference(self, epsilon):
        raise NotImplementedError('Use the bounded tensor tests as an independent oracle, not the diagonal reference.')
