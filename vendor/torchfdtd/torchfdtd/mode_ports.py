"""Fixed full-vector Yee modes in uniform periodic transverse supercells.

This CPU sparse eigensolver is a mode-port foundation, not a mode source.
Fields use exp(+i beta w - i omega t), matching the positive-time DFT in
adjoint_planes. Only propagating real-beta modes of positive isotropic epsilon
are returned. Eigenvalues and mode profiles are NOT differentiable.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

C0 = 299792458.
COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


def _difference(n, spacing, dtype):
    rows = np.arange(n)
    plus = sparse.coo_matrix((np.r_[-np.ones(n), np.ones(n)].astype(dtype),
        (np.r_[rows, rows], np.r_[rows, (rows+1)%n])), shape=(n,n)).tocsr()/spacing
    return plus, -plus.T


def _canonical_degenerate_basis(pairs, n, area, tol):
    """Rotate each cluster of equal-beta modes onto a platform-independent basis.

    ARPACK returns an arbitrary basis of a degenerate eigenspace, and which
    basis it returns depends on the BLAS build and the CPU. Within one cluster
    the power-orthonormal vectors are rotated so that they diagonalize the
    Hermitian overlap of their Eu components, ordered by descending Eu power.
    A uniform or square section therefore yields the u-polarized mode first and
    the v-polarized mode second on every machine. Nondegenerate modes are
    returned unchanged. A cluster whose Eu overlaps are themselves degenerate
    keeps its orthogonalized basis.
    """
    result = list(pairs)
    start = 0
    while start < len(result):
        end = start+1
        while end < len(result) and abs(result[end][0]-result[start][0])<max(10*tol,1e-8)*max(result[end][0],result[start][0]):
            end += 1
        if end-start > 1:
            cluster = [vector for _,vector in result[start:end]]
            eu = np.stack([vector[:n] for vector in cluster],axis=1)
            overlap = eu.conj().T@eu
            values,rotation = np.linalg.eigh(.5*(overlap+overlap.conj().T))
            spread = float(values.max()-values.min())
            if spread > max(100*tol,1e-4)*float(np.abs(values).max()):
                rotated = []
                for column in rotation[:,::-1].T:
                    vector = sum(coefficient*member for coefficient,member in zip(column,cluster))
                    power = float(_power_pair(vector,vector,area).real)
                    if not np.isfinite(power) or power<=0:
                        raise ValueError('Degenerate mode rotation produced a mode without positive power.')
                    vector /= math.sqrt(power)
                    pivot = vector[:2*n][np.argmax(np.abs(vector[:2*n]))]
                    vector *= np.exp(-1j*np.angle(pivot))
                    rotated.append(vector)
                for offset,vector in enumerate(rotated):
                    result[start+offset] = (result[start+offset][0],vector)
        start = end
    return result


def _power_pair(a, b, area):
    """Hermitian, discrete, signed power pairing for [Eu,Ev,Hu,Hv]."""
    n = a.size//4
    eu, ev, hu, hv = a.reshape(4,n)
    fu, fv, gu, gv = b.reshape(4,n)
    return area/4*np.sum(eu*gv.conj()-ev*gu.conj()+fu.conj()*hv-fv.conj()*hu)


@dataclass(frozen=True)
class WaveguideMode:
    beta_per_um: float
    wavelength_um: float
    normal: str
    spacing_um: tuple
    origin_um: tuple
    fields: np.ndarray  # Nu,Nv,6, global component order, own transverse Yee locations
    eigenpair_residual: float
    maxwell_residual: float
    precision: str
    boundary: str = 'periodic'
    invariant_axis: str | None = None

    @property
    def cell_measure(self):
        """SI area in 3D, SI length per invariant length in 2D."""
        if self.invariant_axis is None:
            return math.prod(self.spacing_um)*1e-12
        active = [i for i,a in enumerate(self.transverse_axes) if 'xyz'[a] != self.invariant_axis]
        return self.spacing_um[active[0]]*1e-6

    @property
    def neff(self):
        return self.beta_per_um*self.wavelength_um/(2*math.pi)

    @property
    def transverse_axes(self):
        w = 'xyz'.index(self.normal)
        return (w+1)%3, (w+2)%3

    def component_axes(self, component):
        """Two transverse axes with exactly the native field_axes staggering."""
        if component not in COMPONENTS:
            raise ValueError('Unknown field component.')
        c = 'xyz'.index(component[1].lower())
        offsets = [(.5 if axis == c else 0.) if component[0] == 'E'
                   else (0. if axis == c else .5) for axis in self.transverse_axes]
        return tuple(np.array([0.]) if 'xyz'[axis] == self.invariant_axis else
                     self.origin_um[a]+(np.arange(self.fields.shape[a])+offsets[a])*self.spacing_um[a]
                     for a,axis in enumerate(self.transverse_axes))

    def power(self):
        """Signed reduced E*H integral; SI area or length per invariant length."""
        u,v = self.transverse_axes
        f = self.fields
        density = .5*np.real(f[...,u]*f[...,3+v].conj()-f[...,v]*f[...,3+u].conj()).sum()
        return float(density*math.prod(self.spacing_um)*1e-12 if self.invariant_axis is None
                     else density*self.cell_measure)

    def backward(self):
        """Reciprocal backward mode at the same reference plane, signed P=-1."""
        fields = self.fields.copy()
        w = 'xyz'.index(self.normal)
        fields[...,w] *= -1
        for axis in self.transverse_axes:
            fields[...,3+axis] *= -1
        fields.setflags(write=False)
        return replace(self,beta_per_um=-self.beta_per_um,fields=fields)

    def sample_plane(self, points_um):
        """Periodic bilinear interpolation to co-phased collocated plane points.

        This is a reference-plane profile. No longitudinal propagation or
        half-cell phase for injecting raw staggered time-domain fields is added.
        """
        points = np.asarray(points_um)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Plane points must be finite with shape (points,3).')
        w = 'xyz'.index(self.normal)
        if len(points) and not np.allclose(points[:,w],points[0,w],rtol=0,atol=1e-7):
            raise ValueError('Points must lie on one transverse plane.')
        output = np.zeros((len(points),6),dtype=self.fields.dtype)
        for c,component in enumerate(COMPONENTS):
            axes = self.component_axes(component)
            q = [np.zeros(len(points)) if 'xyz'[axis] == self.invariant_axis else
                 (points[:,axis]-axes[i][0])/self.spacing_um[i]
                 for i,axis in enumerate(self.transverse_axes)]
            lower = [np.floor(value).astype(np.int64) for value in q]
            fractions = [value-index for value,index in zip(q,lower)]
            for i in (0,1):
                for j in (0,1):
                    weights = (fractions[0] if i else 1-fractions[0])*(fractions[1] if j else 1-fractions[1])
                    output[:,c] += weights*self.fields[(lower[0]+i)%self.fields.shape[0],
                                                       (lower[1]+j)%self.fields.shape[1],c]
        return output


def mode_power_overlap(first, second):
    """Independent Hermitian power overlap on native matching Yee samples."""
    if (first.normal,first.spacing_um,first.origin_um,first.fields.shape) != (
            second.normal,second.spacing_um,second.origin_um,second.fields.shape):
        raise ValueError('Mode sampling must match.')
    u,v = first.transverse_axes
    def transverse(mode):
        return np.concatenate([mode.fields[...,c].ravel() for c in (u,v,3+u,3+v)])
    if first.invariant_axis != second.invariant_axis:
        raise ValueError('Mode dimensions must match.')
    return _power_pair(transverse(first),transverse(second),first.cell_measure)


def solve_waveguide_modes(permittivity, *, shape, spacing_um, wavelength_um,
                          origin_um=None, normal='x', num_modes=2, target_neff=None,
                          precision='float32', tolerance=None, maxiter=3000):
    """Solve coupled full-vector modes in an explicitly periodic supercell.

    permittivity is a positive real scalar or callable epsilon(u_um,v_um).
    One scalar material function is sampled separately at Eu, Ev and Ew Yee
    positions. An arbitrary diagonal tensor is intentionally not accepted.
    Isolated guides require padding/box convergence and confinement checks.
    FP32 sparse matrices/ARPACK are default. FP64 is an explicit diagnostic.
    """
    if normal not in ('x','y','z') or precision not in ('float32','float64'):
        raise ValueError('Use normal x/y/z and precision float32/float64.')
    if len(shape)!=2 or any(isinstance(n,bool) or not isinstance(n,int) or n<2 for n in shape):
        raise ValueError('shape must contain two integers at least two.')
    if len(spacing_um)!=2 or any(not np.isfinite(h) or h<=0 for h in spacing_um):
        raise ValueError('spacing_um must contain two positive finite values.')
    if not np.isfinite(wavelength_um) or wavelength_um<=0:
        raise ValueError('wavelength_um must be positive and finite.')
    if isinstance(num_modes,bool) or not isinstance(num_modes,int) or not 1<=num_modes<2*math.prod(shape)-1:
        raise ValueError('Requested mode count must be below transverse eigensystem size minus one.')
    spacing_um = tuple(float(v) for v in spacing_um)
    origin_um = tuple(-n*h/2 for n,h in zip(shape,spacing_um)) if origin_um is None else tuple(origin_um)
    if len(origin_um)!=2 or not np.isfinite(origin_um).all():
        raise ValueError('origin_um needs two finite coordinates.')
    dtype = np.dtype(precision)
    cdtype = np.complex64 if precision=='float32' else np.complex128
    tol = (2e-6 if precision=='float32' else 1e-10) if tolerance is None else tolerance
    if not np.isfinite(tol) or tol<=0:
        raise ValueError('tolerance must be positive and finite.')
    sampled = []
    for offsets in ((.5,0.),(0.,.5),(0.,0.)):
        axes = [origin_um[a]+(np.arange(shape[a])+offsets[a])*spacing_um[a] for a in range(2)]
        coords = np.meshgrid(*axes,indexing='ij')
        value = permittivity(*coords) if callable(permittivity) else permittivity
        if np.iscomplexobj(value):
            raise ValueError('Only real positive isotropic permittivity is supported.')
        values = np.broadcast_to(np.asarray(value,dtype=dtype),shape).copy()
        if not np.isfinite(values).all() or np.min(values)<=0:
            raise ValueError('Permittivity must be finite and positive.')
        sampled.append(values.ravel())
    eu,ev,ew = [sparse.diags(v,format='csr') for v in sampled]
    inverse_ew = sparse.diags(1/sampled[2],format='csr')
    n = math.prod(shape)
    iu,iv = [sparse.eye(v,dtype=dtype,format='csr') for v in shape]
    up,um = _difference(shape[0],spacing_um[0],dtype)
    vp,vm = _difference(shape[1],spacing_um[1],dtype)
    up,um = sparse.kron(up,iv,format='csr'),sparse.kron(um,iv,format='csr')
    vp,vm = sparse.kron(iu,vp,format='csr'),sparse.kron(iu,vm,format='csr')
    ident = sparse.eye(n,dtype=dtype,format='csr')
    k = dtype.type(2*math.pi/wavelength_um)
    # beta Et = P Ht, beta Ht = Q Et. Plus/minus map node/half-cell Yee locations.
    p = sparse.bmat([[-up@inverse_ew@vm/k, k*ident+up@inverse_ew@um/k],
                     [-k*ident-vp@inverse_ew@vm/k, vp@inverse_ew@um/k]],format='csr')
    q = sparse.bmat([[um@vp/k,-k*ev-um@up/k],
                     [k*eu+vm@vp/k,-vm@up/k]],format='csr')
    operator = (p@q).astype(dtype)
    maximum_index = math.sqrt(max(float(v.max()) for v in sampled))
    target = maximum_index*1.001 if target_neff is None else target_neff
    if not np.isfinite(target) or target<=0:
        raise ValueError('target_neff must be positive and finite.')
    # Fixed start is reproducible without changing the caller NumPy RNG.
    start = np.random.default_rng(918).normal(size=2*n).astype(dtype)
    candidates = min(2*n-2,max(num_modes+4,2*num_modes))
    eigenvalues,vectors = eigs(operator,k=candidates,sigma=dtype.type((target*float(k))**2),
                              which='LM',tol=tol,maxiter=maxiter,v0=start,
                              ncv=min(2*n,max(4*candidates+1,40)))
    nearest = np.argsort(np.abs(eigenvalues-(target*float(k))**2))[:num_modes]
    order = nearest[np.argsort(-eigenvalues[nearest].real)]
    modes,previous = [],[]
    area = math.prod(spacing_um)*1e-12
    for index in order:
        value = eigenvalues[index]
        if value.real<=0 or abs(value.imag)>max(20*tol,1e-5)*abs(value.real):
            raise ValueError('Requested eigenspace contains evanescent/nonreal beta. Change target_neff or mode count.')
        beta = math.sqrt(float(value.real))
        electric = vectors[:,index].astype(cdtype)
        magnetic = (q@electric)/beta
        transverse = np.r_[electric,magnetic]
        # ARPACK bases of a degenerate eigenspace need not be power orthogonal.
        for old_beta,old in previous:
            if abs(beta-old_beta)<max(10*tol,1e-8)*max(beta,old_beta):
                transverse -= _power_pair(transverse,old,area)*old
        power = float(_power_pair(transverse,transverse,area).real)
        if not np.isfinite(power) or power<=np.finfo(dtype).tiny:
            raise ValueError('Mode has zero/negative power or a dependent degenerate eigenvector.')
        transverse /= math.sqrt(power)
        electric,magnetic = transverse[:2*n],transverse[2*n:]
        pivot = electric[np.argmax(np.abs(electric))]
        transverse *= np.exp(-1j*np.angle(pivot))
        previous.append((beta,transverse.copy()))
    previous = _canonical_degenerate_basis(previous,n,area,tol)
    for beta,transverse in previous:
        electric,magnetic = transverse[:2*n],transverse[2*n:]
        elu,elv = electric[:n],electric[n:]
        hlu,hlv = magnetic[:n],magnetic[n:]
        elw = 1j*(inverse_ew@(um@hlv-vm@hlu))/k
        hlw = -1j*(up@elv-vp@elu)/k
        residual = np.linalg.norm(operator@electric-beta**2*electric)/(beta**2*np.linalg.norm(electric))
        # Direct six Maxwell curl equations, independent of the squared operator residual.
        curl_e = np.r_[vp@elw-1j*beta*elv,1j*beta*elu-up@elw,up@elv-vp@elu]
        curl_h = np.r_[vm@hlw-1j*beta*hlv,1j*beta*hlu-um@hlw,um@hlv-vm@hlu]
        all_e,all_h = np.r_[elu,elv,elw],np.r_[hlu,hlv,hlw]
        epsilon_e = np.r_[sampled[0]*elu,sampled[1]*elv,sampled[2]*elw]
        maxwell = max(np.linalg.norm(curl_e-1j*k*all_h)/(k*np.linalg.norm(all_h)),
                      np.linalg.norm(curl_h+1j*k*epsilon_e)/(k*np.linalg.norm(epsilon_e)))
        if residual>max(100*tol,2e-4) or maxwell>max(100*tol,2e-4):
            raise RuntimeError('Computed mode fails the eigenpair/Maxwell residual criterion. Refine eigensolver tolerance or diagnose precision.')
        w = 'xyz'.index(normal)
        axes = ((w+1)%3,(w+2)%3,w)
        fields = np.empty((*shape,6),dtype=cdtype)
        for local,global_axis in enumerate(axes):
            fields[...,global_axis] = all_e.reshape(3,*shape)[local]
            fields[...,3+global_axis] = all_h.reshape(3,*shape)[local]
        fields.setflags(write=False)
        modes.append(WaveguideMode(beta,wavelength_um,normal,spacing_um,origin_um,fields,
                                   float(residual),float(maxwell),precision))
    return tuple(modes)


def _amplitudes(fields, basis, weights, normal):
    """Lorentz power overlaps for a fixed forward basis and collocated fields."""
    import torch
    u,v = (normal+1)%3,(normal+2)%3
    power = .5*((basis[:,u]*basis[:,3+v].conj()-basis[:,v]*basis[:,3+u].conj()).real*weights).sum()
    if not bool(torch.isfinite(power)) or not bool(power>0):
        raise ValueError('Sampled forward mode has nonpositive/nonfinite quadrature power.')
    electric = (fields[...,u]*basis[:,3+v].conj()-fields[...,v]*basis[:,3+u].conj())@weights.to(fields.dtype)
    magnetic = (basis[:,u].conj()*fields[...,3+v]-basis[:,v].conj()*fields[...,3+u])@weights.to(fields.dtype)
    return (electric+magnetic)/(4*power),(electric-magnetic)/(4*power)


def _uniform_cell_quadrature(plane, mode):
    """Require one complete, nonduplicated uniform midpoint tensor product."""
    if mode.boundary == 'cpml':
        return mode.validate_quadrature(plane)
    if mode.invariant_axis is not None:
        from .invariant_modes import validate_line_quadrature
        return validate_line_quadrature(plane, mode)
    points = plane.points_um.detach().cpu().numpy()
    weights = plane.weights.detach().cpu().numpy()
    if points.ndim!=2 or points.shape[1]!=3 or weights.shape!=(len(points),):
        raise ValueError('Mode quadrature needs points (N,3) and weights (N,).')
    if not np.isfinite(points).all() or not np.isfinite(weights).all() or not (weights>0).all():
        raise ValueError('Mode quadrature must be finite with positive weights.')
    u,v = mode.transverse_axes
    coordinates = [np.unique(points[:,axis]) for axis in (u,v)]
    counts = tuple(len(axis) for axis in coordinates)
    if min(counts)<2 or math.prod(counts)!=len(points):
        raise ValueError('Mode quadrature requires a complete uniform midpoint tensor product.')
    indices = np.searchsorted(coordinates[0],points[:,u])*counts[1]+np.searchsorted(coordinates[1],points[:,v])
    if len(np.unique(indices))!=len(points):
        raise ValueError('Mode quadrature cannot contain duplicate points.')
    for axis,coordinate in enumerate(coordinates):
        length = mode.fields.shape[axis]*mode.spacing_um[axis]
        expected = mode.origin_um[axis]+(np.arange(counts[axis])+.5)*length/counts[axis]
        if not np.allclose(coordinate,expected,rtol=0,atol=2e-6*length):
            raise ValueError('Mode quadrature must cover exactly its full supercell with uniform midpoints.')
    area = math.prod(mode.fields.shape[:2])*math.prod(mode.spacing_um)*1e-12
    if not np.allclose(weights,area/len(points),rtol=2e-5,atol=0):
        raise ValueError('Mode quadrature requires uniform SI cell-area weights.')


def normalized_mode_power(plane, reference, mode, *, direction='forward'):
    """Torch-differentiable directional mode power / reference forward mode power.

    The mode is fixed. Sample and reference plane fields retain their graphs.
    Both planes require a complete uniform midpoint tensor-product quadrature
    over the same periodic cell at one frequency, with uniform SI area weights.
    Arbitrary quadrature, cropped/duplicated points and nonuniform weights fail.
    This overlap has not yet been validated as an injected time-domain port.
    """
    import torch
    if direction not in ('forward','backward') or mode.beta_per_um.real<=0:
        raise ValueError('Use a forward mode and direction forward/backward.')
    if plane.normal!=mode.normal or reference.normal!=mode.normal or plane.run_signature!=reference.run_signature:
        raise ValueError('Mode normal and sample/reference configurations must match.')
    for name in ('frequency_hz','points_um','weights'):
        a,b = getattr(plane,name),getattr(reference,name)
        if a.device!=b.device or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):
            raise ValueError('Sample/reference frequency, points, weights and placement must match.')
    if plane.frequency_hz.numel()!=1 or not torch.isclose(plane.frequency_hz[0],
            plane.frequency_hz.new_tensor(C0/(mode.wavelength_um*1e-6)),rtol=1e-6,atol=0):
        raise ValueError('A fixed mode requires its single matching frequency.')
    _uniform_cell_quadrature(plane, mode)
    basis = torch.as_tensor(mode.sample_plane(plane.points_um.detach().cpu().numpy()),
                            dtype=plane.fields.dtype,device=plane.fields.device)
    basis_scale = basis.detach().abs().amax()
    if not bool(torch.isfinite(basis_scale)) or not bool(basis_scale>0):
        raise ValueError('Mode fields must be finite and nonzero.')
    basis = basis/basis_scale
    scale = reference.fields.detach().abs().amax()
    if not bool(torch.isfinite(scale)) or not bool(scale>0):
        raise ValueError('Reference fields must be finite and nonzero.')
    weight_scale = plane.weights.detach().abs().amax()
    weights = plane.weights/weight_scale
    forward,backward = _amplitudes(plane.fields/scale,basis,weights,'xyz'.index(mode.normal))
    incident,_ = _amplitudes(reference.fields/scale,basis,weights,'xyz'.index(mode.normal))
    # Relative support check is independent of SI field/area normalization.
    if not bool(torch.isfinite(incident).all()) or bool((incident.abs()<=torch.finfo(incident.real.dtype).eps).any()):
        raise ValueError('Reference forward modal amplitude is zero or numerically unsupported.')
    amplitude = forward if direction=='forward' else backward
    return (amplitude/incident).abs().square()
