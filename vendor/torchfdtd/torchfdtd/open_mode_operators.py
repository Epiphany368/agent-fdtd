"""Native discrete-CPML transverse derivatives for fixed open mode solves.

Fields use exp(-i omega t). This module builds no eigenproblem or factorization.
Sparse operators use C-order (u,v), with cyclic axes u=w+1 and v=w+2.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from scipy import sparse

from .models import Region
from .boundaries import BoundaryDescription


@dataclass(frozen=True)
class OpenModeOperators:
    shape: tuple
    spacing_um: tuple
    origin_um: tuple
    temporal_k_per_um: float
    up: sparse.csr_matrix
    um: sparse.csr_matrix
    vp: sparse.csr_matrix
    vm: sparse.csr_matrix
    physical_bounds_um: tuple
    transverse_axes: tuple
    dtype: np.dtype
    region_signature: str
    longitudinal_spacing_um: float
    boundary_kinds: tuple


def plan_open_mode_operators(region, normal, wavelength_um):
    """Validate metadata and bound operator preparation before sparse allocation.

    This reservation excludes epsilon, P/Q assembly, eigensolver and LU storage.
    The caller must admit its complete solver budget before prepare().
    """
    if not isinstance(region, Region):
        raise ValueError('region must be a native Region.')
    if normal not in ('x', 'y', 'z'):
        raise ValueError('normal must be x, y or z.')
    if (isinstance(wavelength_um, bool) or not isinstance(wavelength_um, (int, float))
            or not math.isfinite(wavelength_um) or wavelength_um <= 0):
        raise ValueError('wavelength_um must be a fixed positive finite scalar.')
    r = Region.model_validate(region.model_dump())
    if (r.dimension != '3d' or r.precision != 'float32' or r.mesh_type != 'uniform'
            or r.material_sampling != 'yee' or r.interface_method != 'staircase'
            or r.complex_fields):
        raise ValueError('Open modes require uniform 3D real FP32 Yee staircase metadata.')
    w = 'xyz'.index(normal)
    axes = ((w+1)%3, (w+2)%3)
    kinds = []
    for axis in axes:
        pair = tuple(face.kind for face in r.boundaries.pair(axis))
        if pair not in (('pml','pml'), ('periodic','periodic')) or r.bloch_phase[axis] != 0:
            raise ValueError('Each transverse axis requires a PML pair or zero-phase periodic pair.')
        kinds.append(pair[0])
    if 'pml' not in kinds or any(face.kind != 'pml' for face in r.boundaries.pair(w)):
        raise ValueError('At least one transverse axis and both longitudinal faces require PML.')
    omega_dt = 2*math.pi*299792458.*r.time_step/(wavelength_um*1e-6)
    if not 0 < omega_dt < math.pi:
        raise ValueError('Carrier frequency must be below temporal Nyquist.')
    shape = tuple(r.shape[a] for a in axes)
    spacing = tuple(float(r.axis_steps[a]) for a in axes)
    origin = tuple(-float(r.actual_size[a])/2 for a in axes)
    bounds = tuple((origin[i]+r.pml_layers(a,0)*spacing[i],
                    origin[i]+(shape[i]-r.pml_layers(a,1))*spacing[i])
                   for i,a in enumerate(axes))
    n = math.prod(shape)
    # Four 2N-stencil CSR matrices plus simultaneous COO/Kronecker conversion,
    # coefficients and conservative packing/index allocation allowance.
    required = 2048*n + 256*sum(r.shape) + 65536
    signature = hashlib.sha256(json.dumps(r.model_dump(mode='json'),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(n=n, shape=shape, spacing_um=spacing, origin_um=origin,
        physical_bounds_um=bounds, transverse_axes=axes, boundary_kinds=tuple(kinds),
        longitudinal_spacing_um=float(r.axis_steps[w]),
        temporal_k_per_um=2*math.sin(omega_dt/2)/(299792458.*r.time_step)*1e-6,
        omega_dt=omega_dt, region_signature=signature, dtype=np.dtype('complex64'),
        required_workspace_bytes=required, workspace_bytes=required,
        scope='Sparse derivative preparation only, excludes eigenproblem and factorization.')


def _axis_operator(region, description, axis, forward, omega_dt):
    n = region.shape[axis]
    periodic = region.boundaries.pair(axis)[0].kind == 'periodic'
    rows = np.arange(n if periodic else n-1, dtype=np.int64)
    if not forward and not periodic:
        rows += 1
    neighbor = (rows+1)%n if forward else (rows-1)%n
    gain = np.ones(n, dtype=np.complex64)
    component = (axis+1)%3  # CPML coefficients depend on target family/axis only.
    for segment in description.cpml[forward,axis,component]:
        target = segment['slice'][axis]
        offset = 0 if forward else 1
        loc = slice(target.start+offset,target.stop+offset)
        b,c,inv_k = (np.asarray(segment[key],dtype=np.float32).reshape(-1)
                     for key in ('b','c','inv_k'))
        # Round coefficients exactly as the native FP32 state preparation does.
        gain[loc] = (inv_k.astype(np.complex128)
                     +c.astype(np.complex128)/(1-b.astype(np.complex128)*np.exp(1j*omega_dt))).astype(np.complex64)
    scale = gain[rows]/np.float32(region.axis_steps[axis])
    diagonal = -scale if forward else scale
    other = scale if forward else -scale
    return sparse.coo_matrix((np.concatenate((diagonal,other)),
        (np.concatenate((rows,rows)),np.concatenate((rows,neighbor)))),shape=(n,n)).tocsr()


def _readonly(matrix):
    matrix = matrix.astype(np.complex64).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    for array in (matrix.data,matrix.indices,matrix.indptr):
        array.setflags(write=False)
    return matrix


def prepare_open_mode_operators(region, *, normal, wavelength_um):
    """Prepare independent staggered plus/minus CSR derivatives, no eigensolve."""
    plan = plan_open_mode_operators(region,normal,wavelength_um)
    r = Region.model_validate(region.model_dump())
    description = BoundaryDescription(r)
    axes = plan['transverse_axes']
    identity = [sparse.eye(n,dtype=np.complex64,format='csr') for n in plan['shape']]
    matrices = []
    for i,axis in enumerate(axes):
        for forward in (True,False):
            derivative = _axis_operator(r,description,axis,forward,plan['omega_dt'])
            matrices.append(_readonly(sparse.kron(derivative,identity[1],format='csr')
                if i == 0 else sparse.kron(identity[0],derivative,format='csr')))
    return OpenModeOperators(**{key:plan[key] for key in (
        'shape','spacing_um','origin_um','temporal_k_per_um','physical_bounds_um',
        'transverse_axes','dtype','region_signature','longitudinal_spacing_um','boundary_kinds')},
        up=matrices[0],um=matrices[1],vp=matrices[2],vm=matrices[3])
