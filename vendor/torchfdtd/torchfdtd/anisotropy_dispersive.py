"""Full-tensor trapezoidal ADE on the nodal constitutive assembly.

Pole p has one scalar resonance omega0_p, one scalar damping gamma_p and a
node strength tensor chi_p >= 0 (rad/s)^2. Its coupling operator is the
forward nodal assembly X_p = sum R^dagger (chi_p dt^2) R / 8, symmetric
positive semidefinite like the inverse assembly S of epsilon_inf. The
isotropic trapezoidal ADE keeps its algebra with scalars replaced by these
operators: a_p = (omega0 dt)^2/2, d_p = 1 + gamma dt/2 + (omega0 dt)^2/4,
K_p = X_p/(4 d_p), response_p = (Q_p - a_p P_p)/d_p,
(S^-1 + K) E' = (S^-1 - K) E + C curl H - sum response_p, K = sum K_p,
delta_p = response_p + K_p (E' + E), P_p += delta_p, Q_p <- 2 delta_p - Q_p.
The implicit step is E' = E + S D with D = (I + K S)^-1 b and
b = C curl H - sum response_p - 2 K E, computed by a truncated Neumann series
whose length is fixed from the admitted bound on ||K S|| before execution.
That fixed polynomial is transposed exactly, so the material VJP includes
every S and X_p application. Pole coupling is a full symmetric tensor; the
scalar resonance and damping are shared by the three principal directions of
each pole (rotated diagonal dispersion uses one pole per principal axis).
"""
from dataclasses import dataclass
import math

import torch

from .anisotropy import (TensorConstitutive, TensorDielectricSimulation, _TensorSystem, _cpml_faces,
                         cpml_face_admissible)
from .differentiable import DifferentiableResult, _FDTD, _Checkpoints


@dataclass(frozen=True)
class _TensorPoleLayout:
    shape: tuple
    pole_count: int
    iterations: int

    def views(self, flat):
        cells = math.prod(self.shape)
        sizes = [9*cells, 9*cells*self.pole_count, self.pole_count, self.pole_count]
        epsilon, chi, frequency2, damping = flat.split(sizes)
        return (epsilon.reshape(self.shape+(3, 3)), chi.reshape((self.pole_count,)+self.shape+(3, 3)),
                frequency2, damping)

    def flatten(self, epsilon, chi, frequency2, damping):
        return torch.cat([epsilon.reshape(-1), chi.reshape(-1), frequency2.reshape(-1), damping.reshape(-1)])


def cpml_face_dispersive_admissible(epsilon, chi, axis):
    """Frequency-independent geometric criterion for dispersive node tensors.

    Where no pole has strength the nondispersive criterion applies. Where any
    pole is active, epsilon_inf must be axis-aligned (diagonal) and satisfy the
    nondispersive criterion, and every chi_p must be a nonnegative scalar
    multiple of epsilon_inf at that node. Then epsilon(omega) is a real scalar
    times epsilon_inf at every frequency: the same admissible ellipsoid when the
    scalar is positive, evanescent when it is negative, never a hyperbolic
    band. Aligned tensors also make the node assemblies pointwise, so the
    discrete scheme keeps this proportionality exactly; rotated or
    non-proportional dispersion inside a face is rejected because it is not.
    Proportionality is compared to within eight units of the input precision.
    """
    active = (chi != 0).any(-1).any(-1).any(0)
    diagonal = torch.ones(epsilon.shape[:-2], dtype=torch.bool, device=epsilon.device)
    for a in range(3):
        for b in range(a+1, 3):
            diagonal = diagonal & (epsilon[..., a, b] == 0)
            for pole in chi:
                diagonal = diagonal & (pole[..., a, b] == 0)
    tolerance = 8*torch.finfo(epsilon.dtype).eps
    proportional = diagonal
    for pole in chi:
        for a in range(3):
            b = (a+1) % 3
            left, right = pole[..., a, a]*epsilon[..., b, b], pole[..., b, b]*epsilon[..., a, a]
            proportional = proportional & ((left-right).abs() <= tolerance*torch.maximum(left.abs(), right.abs()))
    dispersive = proportional & cpml_face_admissible(epsilon, axis)
    return torch.where(active, dispersive, cpml_face_admissible(epsilon, axis))


class _TensorDispersiveSystem(_TensorSystem):
    def __init__(self, project, epsilon, parameters, layout, observation_monitors=None, *, prepare_updates=True):
        super().__init__(project, epsilon, observation_monitors, prepare_updates=prepare_updates)
        self.parameters = parameters.detach()
        self.layout = layout
        self.pole_count = layout.pole_count
        self.iterations = layout.iterations
        _, chi, _, _ = layout.views(self.parameters)
        self.poles = [self.constitutive(pole, project.region.shape, inverse=False) for pole in chi]
        shape = (self.pole_count, *self.grid.E.shape)
        if prepare_updates:
            self.P = torch.zeros(shape, dtype=self.field_dtype, device=self.device)
            self.Q = torch.zeros_like(self.P)
        else:
            self.P = torch.zeros((), dtype=self.field_dtype, device=self.device).expand(shape)
            self.Q = torch.zeros((), dtype=self.field_dtype, device=self.device).expand(shape)

    def constitutive(self, epsilon, shape, pec=None, *, inverse=True):
        from .anisotropy import _pec_faces
        return TensorConstitutive(epsilon, tuple(self.grid.wrap.get(a, 1.) for a in range(3)),
                                  tuple(a in self.grid.wrap for a in range(3)),
                                  _pec_faces(self.region) if pec is None else pec, inverse=inverse)

    def state(self):
        return (*super().state(), self.P, self.Q)

    def coefficients(self):
        _, _, frequency2, damping = self.layout.views(self.parameters)
        a = .5*frequency2
        d = 1+.5*damping+.25*frequency2
        return a, d, 1/(4*d)

    def _coupling(self, scale, field):
        """K field = sum_p X_p field / (4 d_p)."""
        result = torch.zeros_like(field)
        for s, pole in zip(scale, self.poles):
            result = result + s*pole.apply(field)
        return result

    def _electric(self, state):
        """Electric half of the step with every intermediate kept for the transpose."""
        e, h = state[:2]
        p, q = state[-2:]
        curl, psis = self.curl(h, state[2:-2], False)
        a, d, scale = self.coefficients()
        shape = (-1, 1, 1, 1, 1)
        response = (q-a.reshape(shape)*p)/d.reshape(shape)
        coupled = self._coupling(scale, e)
        b = self.grid.courant_number*curl-response.sum(0)-2*coupled
        iterates = [b]
        for _ in range(self.iterations):
            iterates.append(b-self._coupling(scale, self.operator.apply(iterates[-1])))
        new = e+self.operator.apply(iterates[-1])
        total = new+e
        delta = torch.stack([response[i]+scale[i]*pole.apply(total) for i, pole in enumerate(self.poles)])
        return dict(curl=curl, psis=psis, response=response, coupled=coupled, b=b, iterates=iterates,
                    new=new, total=total, delta=delta, a=a, d=d, scale=scale)

    def reference_step(self, state, step, parameters):
        parts = self._electric(state)
        e = self.inject(parts['new'], 'E', step, functional=True)
        curl, psis = self.curl(e, parts['psis'], True)
        h = self.inject(state[1]-self.grid.courant_number*curl, 'H', step, functional=True)
        return (e, h, *psis, state[-2]+parts['delta'], 2*parts['delta']-state[-1])

    def advance(self, start, end):
        for step in range(start, end):
            new = self.reference_step(self.state(), step, self.parameters)
            for target, value in zip(self.state(), new):
                target.copy_(value)
        self.current_step = end

    def transpose_step(self, state, adjoint, signal_bar):
        e_bar, h_bar = adjoint[:2]
        p_bar, q_bar = adjoint[-2:]
        psi_bar = adjoint[2:-2]
        for target, (positions, indices) in zip((e_bar, h_bar), self.observation_maps):
            if indices.numel():
                target.reshape(-1).index_add_(0, indices, signal_bar.index_select(0, positions))
        courant = self.grid.courant_number
        contribution, psi_bar = self.curl_transpose(-courant*h_bar, psi_bar, True)
        new_bar = e_bar+contribution
        parts = self._electric(state)
        a, d, scale = parts['a'], parts['d'], parts['scale']
        e, p = state[0], state[-2]
        epsilon_bar = torch.zeros(self.region.shape+(3, 3), dtype=self.dtype, device=self.device)
        chi_bar = [torch.zeros_like(epsilon_bar) for _ in self.poles]
        scale_bar = torch.zeros_like(scale)
        # Source injection follows the electric update and does not enter P or Q.
        delta_bar = p_bar+2*q_bar
        total_bar = torch.zeros_like(new_bar)
        for i, pole in enumerate(self.poles):
            applied = pole.apply(parts['total'])
            scale_bar[i] = scale_bar[i]+(delta_bar[i].conj()*applied).real.sum()
            total_bar = total_bar+scale[i]*pole.apply(delta_bar[i])
            chi_bar[i] = chi_bar[i]+scale[i]*pole.epsilon_vjp(parts['total'], delta_bar[i])
        new_bar = new_bar+total_bar
        old_bar = total_bar+new_bar
        # new = e + S D_J
        iterates = parts['iterates']
        d_bar = self.operator.apply(new_bar)
        epsilon_bar = epsilon_bar+self.operator.epsilon_vjp(iterates[-1], new_bar)
        b_bar = d_bar
        for j in range(self.iterations, 0, -1):
            # D_j = b - K S D_{j-1}
            through = self.operator.apply(iterates[j-1])
            k_bar = -d_bar
            for i, pole in enumerate(self.poles):
                applied = pole.apply(through)
                scale_bar[i] = scale_bar[i]+(k_bar.conj()*applied).real.sum()
                chi_bar[i] = chi_bar[i]+scale[i]*pole.epsilon_vjp(through, k_bar)
            through_bar = self._coupling(scale, k_bar)
            epsilon_bar = epsilon_bar+self.operator.epsilon_vjp(iterates[j-1], through_bar)
            d_bar = self.operator.apply(through_bar)
            b_bar = b_bar+d_bar
        # b = C curl - sum response - 2 K e
        curl_bar = courant*b_bar
        response_bar = -b_bar.expand(self.pole_count, *b_bar.shape)+delta_bar
        coupled_bar = -2*b_bar
        for i, pole in enumerate(self.poles):
            applied = pole.apply(e)
            scale_bar[i] = scale_bar[i]+(coupled_bar.conj()*applied).real.sum()
            chi_bar[i] = chi_bar[i]+scale[i]*pole.epsilon_vjp(e, coupled_bar)
        old_bar = old_bar+self._coupling(scale, coupled_bar)
        # response = (q - a p)/d, scale = 1/(4 d)
        shape = (-1, 1, 1, 1, 1)
        previous_q_bar = -q_bar+response_bar/d.reshape(shape)
        previous_p_bar = p_bar-a.reshape(shape)*response_bar/d.reshape(shape)
        a_bar = -(response_bar.conj()*p).real.sum((1, 2, 3, 4))/d
        d_bar_scalar = -(response_bar.conj()*parts['response']).real.sum((1, 2, 3, 4))/d-scale_bar*scale/d
        frequency2_bar = .5*a_bar+.25*d_bar_scalar
        damping_bar = .5*d_bar_scalar
        contribution, psi_bar = self.curl_transpose(curl_bar, psi_bar, False)
        gradient = self.layout.flatten(epsilon_bar, torch.stack(chi_bar), frequency2_bar, damping_bar)
        return (old_bar, h_bar+contribution, *psi_bar, previous_p_bar, previous_q_bar), gradient


class TensorDispersiveSimulation(TensorDielectricSimulation):
    """Checkpointed derivatives of node epsilon_inf and tensor Lorentz/Drude poles.

    ``model(epsilon_inf, strength, omega0, gamma)``: epsilon_inf has shape
    (Nx,Ny,Nz,3,3) with eigenvalues >= 1; strength has shape (P,Nx,Ny,Nz,3,3) in
    (rad/s)^2, exactly symmetric and positive semidefinite; omega0 and gamma
    have shape (P,) in rad/s and are nonnegative (omega0=0 is Drude). One
    resonance and damping per pole apply to all its principal directions.
    Same face contract as the nondispersive path, plus: where a pole is active
    in a CPML face region, epsilon_inf is axis-aligned and every strength tensor
    is a scalar multiple of epsilon_inf, so the anisotropy there is frequency
    independent. The implicit step uses a fixed Neumann series; the admitted
    coupling bound ||K S|| decides its length and rejects inputs that would
    need more than 64 terms.
    """
    def __init__(self, project, options=None, *, cpml_material='tensor'):
        if cpml_material != 'tensor':
            raise ValueError("Tensor dispersion supports cpml_material='tensor' only.")
        super().__init__(project, options, cpml_material='tensor')
        if any(s.enabled and s.injection != 'soft' for s in self.project.sources):
            raise ValueError('Tensor dispersion requires soft source injection.')

    def _validate_cpml_collar(self, epsilon, chi=None):
        if chi is None:
            return super()._validate_cpml_collar(epsilon)
        for axis, side, index in _cpml_faces(self.project.region):
            if not bool(cpml_face_dispersive_admissible(epsilon[index], chi[(slice(None),)+index], axis).all()):
                raise ValueError(
                    'Tensor CPML face %s_%s: where a pole is active in its PML layers plus one-node collar, '
                    'epsilon_inf must be axis-aligned with a non-intermediate normal eigenvalue and every strength '
                    'tensor a nonnegative scalar multiple of epsilon_inf.' % ('xyz'[axis], ('min', 'max')[side]))

    def _pack(self, epsilon, strength, omega0, gamma, admission=None):
        self._validate_input_shape(epsilon)
        r = self.project.region
        def rates(value, name):
            raw = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if raw.is_complex():
                raise ValueError(name+' must be real.')
            result = torch.as_tensor(value, device=epsilon.device, dtype=epsilon.dtype)
            if result.ndim != 1 or not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
                raise ValueError(name+' must be a finite nonnegative vector with one entry per pole.')
            return result
        if not isinstance(strength, torch.Tensor) or strength.dtype != epsilon.dtype or strength.device != epsilon.device:
            raise ValueError('strength must be a tensor with the epsilon_inf dtype and device.')
        if strength.ndim != 6 or strength.shape[1:] != epsilon.shape or not 1 <= strength.shape[0] <= 64:
            raise ValueError('strength must have shape (P,Nx,Ny,Nz,3,3) with 1 to 64 poles.')
        count = strength.shape[0]
        omega0, gamma = rates(omega0, 'omega0'), rates(gamma, 'gamma')
        if omega0.shape != (count,) or gamma.shape != (count,):
            raise ValueError('omega0 and gamma need one entry per pole.')
        dt = r.time_step
        with torch.no_grad():
            self._validate_epsilon(epsilon)
            if not bool(torch.isfinite(strength).all()):
                raise ValueError('strength must be finite.')
            if not torch.equal(strength, strength.transpose(-1, -2)):
                raise ValueError('strength must be exactly symmetric. Construct it symmetrically.')
            lowest, highest = self._eigenvalue_bounds(strength)
            # A singular PSD tensor assembled as R diag(0, ...) R^T carries a rounding
            # eigenvalue of either sign whose size depends on the LAPACK build.
            tolerance = 64*torch.finfo(strength.dtype).eps*max(float(highest.abs()), 1.)
            if bool(lowest < -tolerance):
                raise ValueError('Passive tensor poles require positive semidefinite strength.')
            self._validate_cpml_collar(epsilon, strength)
            d = 1+.5*gamma*dt+.25*(omega0*dt).square()
            # ||K S|| <= max_n sum_p lambda_max(chi_p(n) dt^2)/(4 d_p) with lambda_min(epsilon_inf) >= 1.
            # Bounded eigenvalue batches keep validation scratch small on CUDA.
            top = torch.cat([torch.linalg.eigvalsh(batch).max(-1).values
                             for batch in strength.reshape(-1, 3, 3).split(64)]).reshape(count, -1)
            bound = float((top*dt*dt/(4*d)[:, None]).sum(0).max())
            if not math.isfinite(bound) or bound >= 1:
                raise ValueError('Tensor ADE coupling bound ||K S|| must be below one; reduce the time step or strength.')
            tolerance = torch.finfo(epsilon.dtype).eps
            iterations = 0 if bound == 0 else max(0, math.ceil(math.log(tolerance)/math.log(bound))-1)
            if iterations > 64:
                raise ValueError('Tensor ADE Neumann series would need more than 64 terms; reduce the time step or strength.')
        layout = _TensorPoleLayout(tuple(epsilon.shape[:3]), count, iterations)
        if admission is not None:admission(layout)
        # Scale before packing; the large physical rates are never squared before dt.
        packed = layout.flatten(epsilon, strength*dt*dt, (omega0*dt).square(), gamma*dt)
        if not bool(torch.isfinite(packed).all()):
            raise ValueError('Normalized oscillator coefficients overflow this precision.')
        return packed, layout, bound

    def _evaluate(self, epsilon, strength, omega0, gamma, spectral):
        parameters, layout, bound = self._pack(epsilon, strength, omega0, gamma)
        count = layout.pole_count
        report = self.reservation(spectral, device=epsilon.device, pole_count=count, iterations=layout.iterations)
        with torch.no_grad():
            system = _TensorDispersiveSystem(self.project.model_copy(deep=True), epsilon, parameters, layout,
                                             None if spectral is None else spectral.observers)
        r = self.project.region
        report.update(experimental=True, adjoint='full-tensor Yee explicit transpose with trapezoidal tensor ADE',
                      higher_order=False, spatial_streaming=False, full_time_autograd=False,
                      forward_backend='torch '+epsilon.device.type.upper(), backward_backend='torch explicit transpose',
                      tensor_sampling='common mesh nodes, normalized finite/periodic edge triplets',
                      cpml_contract='D-field composition; tensors extend into CPML where each face normal is a principal '
                                    'axis with a non-intermediate eigenvalue; axis-aligned proportional dispersion where poles are active',
                      oscillator_count=count, neumann_terms=layout.iterations+1, coupling_bound=bound,
                      material_parameters='epsilon_inf, strength [(rad/s)^2 node tensors], omega0 [rad/s], gamma [rad/s]',
                      source_contract='soft impressed-field increments', steps=system.region.steps)
        admission = _Checkpoints(system, self.options, report, admission=True)
        admission.close()
        signals = _FDTD.apply(parameters, system, self.options, report, spectral)
        if spectral is not None:
            return spectral.result(signals, report)
        return DifferentiableResult(signals, r.time_step,
                                    tuple(m.component for m in self.project.monitors if m.enabled), report)

    def reservation(self, spectral=None, *, device='cpu', pole_count=1, iterations=1):
        from .adjoint_memory import _resident_reservation
        from .memory_profile import host_memory
        device = torch.device(device)
        r = self.project.region
        cells = math.prod(r.shape)
        base = _resident_reservation(self.project, self.options, device, spectral, pole_count=pole_count,
                                     parameter_elements=9*cells*(1+pole_count)+2*pole_count)
        # Inverse/forward coefficients, validation scratch, VJP accumulators and
        # outer products per operator, plus the retained Neumann iterates.
        extra = cells*(8 if r.precision == 'float64' else 4)*(192*(1+pole_count)+6*(iterations+2))
        library = 64*1024**2 if device.type == 'cuda' else 0
        extra += library
        host = base['host_reservation_bytes']+(extra if device.type == 'cpu' else 0)
        active = base['memory_reservation_bytes']+extra
        if self.options.host_budget_bytes is not None and host > self.options.host_budget_bytes:
            raise ValueError('Full-tensor ADE reservation exceeds the explicit host byte budget.')
        if self.options.resident_budget_bytes is not None and active > self.options.resident_budget_bytes:
            raise ValueError('Full-tensor ADE reservation exceeds the explicit resident byte budget.')
        available = host_memory()['available_bytes']
        if available is not None and host > .8*available:
            raise ValueError('Full-tensor ADE reservation exceeds available host memory.')
        if device.type == 'cuda':
            from .cuda_memory import cuda_budget_limit
            if active > cuda_budget_limit(device, active, self.options.gpu_budget_bytes):
                raise ValueError('Full-tensor ADE reservation exceeds the GPU byte budget.')
        return dict(base, host_reservation_bytes=host, memory_reservation_bytes=active,
                    gpu_reservation_bytes=active if device.type == 'cuda' else 0,
                    tensor_reservation_bytes=extra, tensor_cuda_library_allowance_bytes=library)

    def forward(self, epsilon_inf, strength, omega0, gamma):
        return self._evaluate(epsilon_inf, strength, omega0, gamma, None)

    def spectrum(self, epsilon_inf, strength, omega0, gamma, frequency_hz, *, window=None, block_size=32):
        from .adjoint_spectrum import SpectralObservation
        spectral = SpectralObservation(epsilon_inf, self.project.region,
                                       [m.component for m in self.project.monitors if m.enabled], frequency_hz, window, block_size)
        return self._evaluate(epsilon_inf, strength, omega0, gamma, spectral)

    def reference(self, epsilon_inf, strength, omega0, gamma, *, graph_budget_bytes=None):
        """Small-problem full-autograd oracle over the same discrete update.

        The retained graph is admitted by its estimated memory, which depends
        on the Neumann length fixed by packing; graph_budget_bytes caps it.
        """
        from .oracle_memory import admit_oracle, oracle_graph_bytes, oracle_source_terms
        r = self.project.region
        terms = oracle_source_terms(self.project)
        def admit(count, iterations):
            admit_oracle(oracle_graph_bytes(r, 'tensor_ade', pole_count=count, iterations=iterations, source_terms=terms),
                         strength.device, graph_budget_bytes)
        # Zero Neumann terms is the smallest graph for these poles, so this
        # refuses before any validation scratch; packing admits the exact
        # length before it allocates the packed parameters.
        if isinstance(strength, torch.Tensor) and strength.ndim == 6:
            admit(strength.shape[0], 0)
        parameters, layout, _ = self._pack(epsilon_inf, strength, omega0, gamma,
                                           admission=lambda layout: admit(layout.pole_count, layout.iterations))
        system = _TensorDispersiveSystem(self.project, epsilon_inf, parameters, layout)
        # The oracle differentiates through the operators built from the live parameters.
        epsilon, chi, frequency2, damping = layout.views(parameters)
        system.operator = system.constitutive(epsilon, r.shape)
        system.poles = [system.constitutive(pole, r.shape, inverse=False) for pole in chi]
        system.parameters = parameters
        state = tuple(torch.zeros_like(x) for x in system.state())
        signals = []
        for step in range(r.steps):
            state = system.reference_step(state, step, parameters)
            signals.append(system.observe(state))
        return torch.stack(signals)
