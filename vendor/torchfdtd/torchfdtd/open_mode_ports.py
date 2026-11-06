"""Fixed bound modes with native transverse CPML.

Material sampling is FP32 and sparse fields/operators are complex64. Sparse
shift-invert memory is admitted using a deliberately dense-fill engineering
bound, not an assertion about SciPy/BLAS process RSS. Profiles do not carry
material/eigenvalue derivatives. Leaky channels are intentionally rejected.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

from .mode_ports import COMPONENTS


def _axes(shape, spacing, origin, transverse, component):
    c = 'xyz'.index(component[1].lower())
    offsets = [(.5 if axis == c else 0.) if component[0] == 'E'
               else (0. if axis == c else .5) for axis in transverse]
    return tuple(origin[i]+(np.arange(shape[i])+offsets[i])*spacing[i] for i in range(2))


def _weights(shape, spacing, origin, transverse, bounds, periodic, component):
    axes = _axes(shape, spacing, origin, transverse, component)
    weights = []
    for axis, h, (lo, hi), wrap in zip(axes, spacing, bounds, periodic):
        weights.append(np.full(len(axis), h) if wrap else
                       np.maximum(0., np.minimum(axis+h/2, hi)-np.maximum(axis-h/2, lo)))
    return weights[0][:, None]*weights[1][None, :]*1e-12


def _power_pair(first, second, weights, transverse):
    u, v = transverse
    wu, wv = weights
    return .25*np.sum(wu*(first[..., u]*second[..., 3+v].conj()+second[..., u].conj()*first[..., 3+v])
                       -wv*(first[..., v]*second[..., 3+u].conj()+second[..., v].conj()*first[..., 3+u]))


@dataclass(frozen=True)
class OpenWaveguideMode:
    beta_per_um: complex
    beta_tilde_per_um: complex
    wavelength_um: float
    normal: str
    spacing_um: tuple
    origin_um: tuple
    fields: np.ndarray
    eigenpair_residual: float
    maxwell_residual: float
    region_signature: str
    physical_bounds_um: tuple
    sampled_epsilon: np.ndarray
    periodic_axes: tuple
    diagnostics: dict
    precision: str = 'float32'
    boundary: str = 'cpml'

    def __post_init__(self):
        # Bytes own the immutable payload, so callers cannot re-enable writes.
        for name, dtype in (('fields', np.complex64), ('sampled_epsilon', np.float32)):
            value = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            frozen = np.frombuffer(value.tobytes(), dtype=dtype).reshape(value.shape)
            object.__setattr__(self, name, frozen)

    @property
    def neff(self):
        return self.beta_per_um*self.wavelength_um/(2*math.pi)

    @property
    def transverse_axes(self):
        w = 'xyz'.index(self.normal)
        return (w+1)%3, (w+2)%3

    def component_axes(self, component):
        if component not in COMPONENTS:
            raise ValueError('Unknown field component.')
        return _axes(self.fields.shape[:2], self.spacing_um, self.origin_um,
                     self.transverse_axes, component)

    def _power_weights(self):
        return tuple(_weights(self.fields.shape[:2], self.spacing_um, self.origin_um,
            self.transverse_axes, self.physical_bounds_um, self.periodic_axes,
            'E'+'xyz'[axis]) for axis in self.transverse_axes)

    def power(self):
        """Signed physical-aperture reduced power, not stretched-PML power."""
        return float(_power_pair(self.fields, self.fields, self._power_weights(), self.transverse_axes).real)

    def backward(self):
        fields = self.fields.copy()
        fields[..., 'xyz'.index(self.normal)] *= -1
        for axis in self.transverse_axes:
            fields[..., 3+axis] *= -1
        fields.setflags(write=False)
        return replace(self, beta_per_um=-self.beta_per_um,
                       beta_tilde_per_um=-self.beta_tilde_per_um, fields=fields)

    def sample_plane(self, points_um):
        """Transverse interpolation without wrapping across an open axis.

        Co-phased profile only. Longitudinal injection/collocation factors are
        supplied separately by prepare_open_modal_launch and its detector.
        """
        points = np.asarray(points_um)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Plane points must be finite with shape (points,3).')
        w = 'xyz'.index(self.normal)
        if len(points) and not np.allclose(points[:, w], points[0, w], atol=1e-7, rtol=0):
            raise ValueError('Points must lie on one transverse plane.')
        result = np.zeros((len(points), 6), dtype=np.complex64)
        for c, name in enumerate(COMPONENTS):
            coords = self.component_axes(name)
            indices, fractions = [], []
            for i, axis in enumerate(self.transverse_axes):
                q = (points[:, axis]-coords[i][0])/self.spacing_um[i]
                if not self.periodic_axes[i]:
                    if np.any(q < -1e-6) or np.any(q > len(coords[i])-1+1e-6):
                        raise ValueError('Requested plane samples lie outside the nonperiodic Yee support.')
                    q = np.clip(q, 0, len(coords[i])-1)
                low = np.floor(q).astype(np.int64)
                if not self.periodic_axes[i]:
                    low = np.minimum(low, len(coords[i])-2)
                indices.append(low)
                fractions.append(q-low)
            for i in (0, 1):
                for j in (0, 1):
                    u, v = indices[0]+i, indices[1]+j
                    if self.periodic_axes[0]: u = u % self.fields.shape[0]
                    if self.periodic_axes[1]: v = v % self.fields.shape[1]
                    weight = (fractions[0] if i else 1-fractions[0])*(fractions[1] if j else 1-fractions[1])
                    result[:, c] += weight*self.fields[u, v, c]
        return result

    def validate_quadrature(self, plane):
        """Require the complete physical rectangle's midpoint tensor rule."""
        def array(value):
            if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
            return np.asarray(value)
        points, weights = array(plane.points_um), array(plane.weights)
        if plane.normal != self.normal or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Quadrature must match the mode normal and physical rectangle.')
        if weights.shape != (len(points),) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Physical quadrature weights must be finite and positive.')
        axes = [np.unique(points[:, axis]) for axis in self.transverse_axes]
        if math.prod(len(a) for a in axes) != len(points):
            raise ValueError('Quadrature must cover a full tensor-product rectangle.')
        expected_axes = []
        for axis, (lo, hi) in zip(axes, self.physical_bounds_um):
            expected = lo+(np.arange(len(axis))+.5)*(hi-lo)/len(axis)
            if not np.allclose(axis, expected, atol=1e-6, rtol=2e-6):
                raise ValueError('Quadrature does not cover the declared physical rectangle.')
            expected_axes.append(expected)
        pairs = points[:, self.transverse_axes]
        if len(np.unique(pairs, axis=0)) != len(points):
            raise ValueError('Quadrature has duplicate points.')
        area = math.prod(hi-lo for lo, hi in self.physical_bounds_um)*1e-12
        if not np.allclose(weights, area/len(points), rtol=2e-5, atol=0):
            raise ValueError('Quadrature weights do not match the physical rectangle.')
        sampled = self.sample_plane(points)
        u, v = self.transverse_axes
        complex_flux = .5*np.sum(weights*(sampled[:, u]*sampled[:, 3+v].conj()-sampled[:, v]*sampled[:, 3+u].conj()))
        power = float(complex_flux.real)
        if not np.isfinite(power) or power*self.power() <= 0:
            raise ValueError('Quadrature produces nonfinite or wrong-sign physical flux.')
        if abs(complex_flux.imag)/abs(power) > 1e-3:
            raise ValueError('Physical quadrature reactive flux exceeds 1e-3; forward/backward decomposition is unsupported.')
        return float(power)


def solve_open_waveguide_modes(permittivity, *, region, normal, wavelength_um,
        cladding_epsilon, num_modes=1, target_neff=None, mode_budget_bytes=2*1024**3,
        tolerance=2e-6, maxiter=3000, confinement_tolerance=1e-4):
    """Return fixed bound modes; reject nonconfined/near-cutoff PML box modes.

    One scalar isotropic material callable is sampled independently at Eu/Ev/Ew.
    Arrays/tensors representing an unspecific electric staggering are rejected.
    Sparse LU may use O(N²) fill. The preflight reserves eight dense complex
    matrices of eigensystem dimension plus vectors/sampling/operator overhead.
    This conservative admission is not a hard process-RSS guarantee.

    Multiple modes may be power-orthogonalized only when their relative native
    beta_t separation is <= min(4e-5, max(64*eps32, 4*tolerance)). A fresh
    eigenpair and six-field Maxwell residual must pass after the rotation.
    Rank-deficient clusters and nonorthogonal distinct-beta families fail
    explicitly instead of silently substituting another family. Fixed ARPACK
    start and phase normalization make repeated solves reproducible on the
    same numerical stack, not an invariant polarization basis across meshes.
    """
    from .open_mode_operators import plan_open_mode_operators, prepare_open_mode_operators
    if type(mode_budget_bytes) is not int or mode_budget_bytes <= 0:
        raise ValueError('mode_budget_bytes must be a positive integer.')
    if type(num_modes) is not int or num_modes < 1 or type(maxiter) is not int or maxiter < 1:
        raise ValueError('num_modes and maxiter must be positive integers.')
    if not math.isfinite(tolerance) or tolerance <= 0 or not 0 < confinement_tolerance < 1:
        raise ValueError('Positive finite eigensolver and confinement tolerances are required.')
    if type(cladding_epsilon) not in (float, int) or not math.isfinite(cladding_epsilon) or cladding_epsilon < 1:
        raise ValueError('cladding_epsilon must be fixed, finite and at least one.')
    if not callable(permittivity) and not np.isscalar(permittivity):
        raise ValueError('Supply a fixed scalar or isotropic material callable, not an unstaggered array.')
    from .models import Region
    region = Region.model_validate(region.model_dump())
    plan = plan_open_mode_operators(region, normal=normal, wavelength_um=wavelength_um)
    w = 'xyz'.index(normal)
    transverse = ((w+1)%3, (w+2)%3)
    shape = tuple(region.shape[a] for a in transverse)
    n, dimension = math.prod(shape), 2*math.prod(shape)
    if num_modes >= dimension-1:
        raise ValueError('Too many requested modes for the transverse eigensystem.')
    candidates = min(dimension-2, max(num_modes+6, 2*num_modes))
    ncv = min(dimension, max(4*candidates+1, 40))
    required = 8*dimension*dimension*8+8*dimension*(ncv+4*candidates)+2048*n
    required += int(plan.get('required_bytes', plan.get('workspace_bytes', 0)))
    if required > mode_budget_bytes:
        raise ValueError(f'Open mode dense-fill engineering reservation exceeds mode budget: {required} > {mode_budget_bytes}.')
    from .memory_profile import host_memory
    available = host_memory()['available_bytes']
    if available is not None and required > int(.8*available):
        raise ValueError(f'Open mode reservation exceeds 80% of available host memory: {required} > {int(.8*available)}.')
    spacing = tuple(region.axis_steps[a] for a in transverse)
    origin = tuple(float(region.mesh_nodes[a][0]) for a in transverse)
    periodic = tuple(region.boundaries.pair(a)[0].kind == 'periodic' for a in transverse)
    sampled = []
    collar = np.zeros(shape, dtype=bool)
    for i, axis in enumerate(transverse):
        if not periodic[i]:
            lo, hi = region.pml_layers(axis, 0)+1, region.pml_layers(axis, 1)+1
            mask = np.arange(shape[i]) < lo
            mask |= np.arange(shape[i]) >= shape[i]-hi
            collar |= mask[:, None] if i == 0 else mask[None, :]
    for axis in (*transverse, w):
        coords = np.meshgrid(*_axes(shape, spacing, origin, transverse, 'E'+'xyz'[axis]), indexing='ij')
        raw = permittivity(*coords) if callable(permittivity) else permittivity
        if getattr(raw, 'requires_grad', False):
            raise ValueError('Open mode profiles are fixed; trainable material samples are unsupported.')
        if np.iscomplexobj(raw):
            raise ValueError('Only real isotropic dielectric samples are supported.')
        value = np.broadcast_to(np.asarray(raw, dtype=np.float32), shape).copy()
        if not np.isfinite(value).all() or np.any(value < 1):
            raise ValueError('Material samples must be finite FP32 epsilon >= 1.')
        if not np.all(value[collar] == np.float32(cladding_epsilon)):
            raise ValueError('Material must equal fixed homogeneous cladding in CPML plus one-cell collar.')
        sampled.append(value.ravel())
    operators = prepare_open_mode_operators(region, normal=normal, wavelength_um=wavelength_um)
    up, um, vp, vm = operators.up, operators.um, operators.vp, operators.vm
    k = np.float32(operators.temporal_k_per_um)
    ident = sparse.eye(n, dtype=np.complex64, format='csr')
    eu, ev, ew = [sparse.diags(x.astype(np.complex64), format='csr') for x in sampled]
    iw = sparse.diags((1/sampled[2]).astype(np.complex64), format='csr')
    p = sparse.bmat([[-up@iw@vm/k, k*ident+up@iw@um/k],
                     [-k*ident-vp@iw@vm/k, vp@iw@um/k]], format='csr').astype(np.complex64)
    q = sparse.bmat([[um@vp/k, -k*ev-um@up/k],
                     [k*eu+vm@vp/k, -vm@up/k]], format='csr').astype(np.complex64)
    operator = (p@q).astype(np.complex64)
    maximum = math.sqrt(max(float(x.max()) for x in sampled))
    target = maximum*1.001 if target_neff is None else target_neff
    if not np.isfinite(target) or target <= math.sqrt(cladding_epsilon):
        raise ValueError('target_neff must be finite and above the cladding light line.')
    # An explicit target is the physical beta/k0, not beta_t/k_t.
    target_beta_t = float(k)*target if target_neff is None else 2*math.sin(
        math.pi*target*region.axis_steps[w]/wavelength_um)/region.axis_steps[w]
    shift = np.complex64(target_beta_t**2)
    start = np.random.default_rng(918).normal(size=dimension).astype(np.complex64)
    values, vectors = eigs(operator, k=candidates, sigma=shift, which='LM',
        tol=tolerance, maxiter=maxiter, ncv=ncv, v0=start)
    weights = tuple(_weights(shape, spacing, origin, transverse, operators.physical_bounds_um,
        periodic, 'E'+'xyz'[axis]) for axis in transverse)
    epsilon_global = np.empty((*shape, 3), dtype=np.float32)
    for local, global_axis in enumerate((*transverse, w)):
        epsilon_global[..., global_axis] = sampled[local].reshape(shape)
    epsilon_global.setflags(write=False)
    modes, failures = [], []
    hw = region.axis_steps[w]
    for index in np.argsort(np.abs(values-shift)):
        beta_t = complex(np.sqrt(values[index]))
        if beta_t.real <= 0: beta_t = -beta_t
        if beta_t.real <= float(k)*math.sqrt(cladding_epsilon)*(1+20*tolerance):
            failures.append('cladding light line'); continue
        if beta_t.real > float(k)*maximum*(1+100*tolerance) or abs(beta_t*hw/2) >= 1:
            failures.append('spurious or longitudinal band'); continue
        beta = complex(2*np.arcsin(beta_t*hw/2)/hw)
        if abs(beta.imag) > max(confinement_tolerance, 20*tolerance)*beta.real:
            failures.append('attenuating/leaky PML mode'); continue
        electric = vectors[:, index].astype(np.complex64)
        magnetic = (q@electric)/np.complex64(beta_t)
        elu, elv, hlu, hlv = electric[:n], electric[n:], magnetic[:n], magnetic[n:]
        elw = 1j*(iw@(um@hlv-vm@hlu))/k
        hlw = -1j*(up@elv-vp@elu)/k
        all_e, all_h = np.r_[elu, elv, elw], np.r_[hlu, hlv, hlw]
        fields = np.empty((*shape, 6), dtype=np.complex64)
        for local, global_axis in enumerate((*transverse, w)):
            fields[..., global_axis] = all_e.reshape(3, *shape)[local]
            fields[..., 3+global_axis] = all_h.reshape(3, *shape)[local]
        # Only a numerically degenerate eigenspace may be rotated. The cap
        # remains below the fresh residual gate, independent of loose user tol.
        cluster_tolerance = min(4e-5, max(64*np.finfo(np.float32).eps, 4*tolerance))
        cluster = [old for old in modes if abs(beta_t-old.beta_tilde_per_um)
                   <= cluster_tolerance*max(abs(beta_t), abs(old.beta_tilde_per_um))]
        if cluster:
            original_norm = np.linalg.norm(fields)
            # Twice-applied modified Gram-Schmidt in physical Hermitian flux.
            for _ in range(2):
                for old in cluster:
                    overlap = _power_pair(fields, old.fields, weights, transverse)
                    fields -= np.complex64(overlap)*old.fields
            if np.linalg.norm(fields) <= 1e-4*original_norm:
                raise ValueError('Degenerate eigenspace is rank deficient in physical power; increase solver resolution or change target.')
            elu, elv, elw = [fields[..., axis].ravel() for axis in (*transverse, w)]
            hlu, hlv, hlw = [fields[..., 3+axis].ravel() for axis in (*transverse, w)]
            electric = np.r_[elu, elv]
            all_e, all_h = np.r_[elu, elv, elw], np.r_[hlu, hlv, hlw]
        energy = np.sum(np.abs(fields.astype(np.complex128))**2, axis=-1)
        tail = float(energy[collar].sum()/energy.sum())
        if not np.isfinite(tail) or tail > confinement_tolerance:
            if cluster:
                raise ValueError('Degenerate eigenspace rotation violates confinement; refusing family substitution.')
            failures.append('unconfined collar/PML tail'); continue
        power = float(_power_pair(fields, fields, weights, transverse).real)
        if not math.isfinite(power) or power <= 0:
            if cluster:
                raise ValueError('Degenerate eigenspace rotation has nonpositive physical power.')
            failures.append('nonpositive physical power'); continue
        residual = np.linalg.norm(operator@electric-beta_t**2*electric)/(abs(beta_t)**2*np.linalg.norm(electric))
        ce = np.r_[vp@elw-1j*beta_t*elv, 1j*beta_t*elu-up@elw, up@elv-vp@elu]
        ch = np.r_[vm@hlw-1j*beta_t*hlv, 1j*beta_t*hlu-um@hlw, um@hlv-vm@hlu]
        eps_e = np.concatenate([sampled[i]*all_e[i*n:(i+1)*n] for i in range(3)])
        maxwell = max(np.linalg.norm(ce-1j*k*all_h)/(k*np.linalg.norm(all_h)),
                      np.linalg.norm(ch+1j*k*eps_e)/(k*np.linalg.norm(eps_e)))
        if not np.isfinite(residual) or not np.isfinite(maxwell) or max(residual, maxwell) > max(100*tolerance, 2e-4):
            if cluster:
                raise ValueError('Degenerate eigenspace rotation fails fresh eigenpair or six-field Maxwell residual.')
            failures.append('six-field Maxwell residual'); continue
        fields /= math.sqrt(power)
        pivot = fields[..., :3].reshape(-1)[np.argmax(np.abs(fields[..., :3]))]
        fields *= np.complex64(np.exp(-1j*np.angle(pivot)))
        if any(abs(_power_pair(fields, old.fields, weights, transverse)) > max(1e-3, 50*tolerance) for old in modes):
            raise ValueError('Distinct-beta modes have nonorthogonal physical power; refusing to substitute another mode family.')
        fields.setflags(write=False)
        modes.append(OpenWaveguideMode(beta, beta_t, wavelength_um, normal, spacing, origin,
            fields, float(residual), float(maxwell), operators.region_signature,
            tuple(operators.physical_bounds_um), epsilon_global, periodic,
            dict(collar_energy_fraction=tail, engineering_reservation_bytes=required,
                 eigensolver_dtype='complex64', material_dtype='float32',
                 physical_power_gram_tolerance=max(1e-3, 50*tolerance),
                 branch='positive-real beta_t; principal complex asin within propagation band',
                 degeneracy_relative_beta_t_tolerance=cluster_tolerance,
                 degenerate_predecessors=len(cluster), injectable=False)))
        if len(modes) == num_modes: return tuple(modes)
    raise ValueError('Insufficient admitted bound modes: '+', '.join(failures))
