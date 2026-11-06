"""Fixed spectral polarization calibration of two linear source responses."""
from dataclasses import replace
import hashlib
import torch


def _pair(planes):
    if len(planes)!=2:raise ValueError('Exactly two source-basis planes are required.')
    a,b=planes
    if a.normal not in ('x','y','z') or a.components!=('Ex','Ey','Ez','Hx','Hy','Hz'):
        raise ValueError('Calibration requires complete collocated E/H planes.')
    if a.fields.ndim!=3 or a.fields.shape[-1]!=6 or not a.fields.is_complex():
        raise ValueError('Calibration requires complex frequency, point, component fields.')
    if a.normal!=b.normal or a.components!=b.components or a.fields.shape!=b.fields.shape:
        raise ValueError('Source-basis planes must have matching components and shapes.')
    for name in ('frequency_hz','points_um','weights'):
        x,y=getattr(a,name),getattr(b,name)
        if x.dtype!=y.dtype or x.device!=y.device or not torch.equal(x,y):
            raise ValueError('Source-basis planes must share frequencies and quadrature.')
    if a.fields.dtype!=b.fields.dtype or a.fields.device!=b.fields.device:
        raise ValueError('Source-basis field dtype and device must match.')
    if not all(bool(torch.isfinite(p.fields).all()) for p in planes):
        raise ValueError('Source-basis fields must be finite.')
    if a.fields.shape[0]<1 or a.fields.shape[1]<1 or a.points_um.shape!=(a.fields.shape[1],3) or a.weights.shape!=(a.fields.shape[1],):
        raise ValueError('Source-basis quadrature must match nonempty field arrays.')
    if not bool(torch.isfinite(a.points_um).all() & torch.isfinite(a.weights).all()) or bool((a.weights<=0).any()):
        raise ValueError('Source-basis quadrature must be finite with positive weights.')
    return a,b


def calibrate_plane_polarization(planes, transverse_wavevector_per_um, target_electric,
                                *, max_condition=1e6):
    """Find fixed spectral coefficients for a desired tangential electric vector.

    planes are two independent homogeneous-reference source responses. Wavevector
    and target have shape (frequency, 2), in increasing transverse-axis order.
    Fields use exp(+i k.r), and target is defined at coordinate origin after
    removing transverse phase. The operation does not separate forward/backward
    waves. Reference planes must be downstream of the sources with negligible
    reflected waves, in a uniform isotropic medium. Verify convergence separately.
    """
    a,b=_pair(planes)
    if torch.as_tensor(transverse_wavevector_per_um).is_complex():
        raise ValueError('Transverse wavevector must be real.')
    k=torch.as_tensor(transverse_wavevector_per_um,device=a.fields.device,dtype=a.fields.real.dtype)
    target=torch.as_tensor(target_electric,device=a.fields.device,dtype=a.fields.dtype)
    shape=(a.fields.shape[0],2)
    if k.shape!=shape or target.shape!=shape or k.requires_grad or target.requires_grad:
        raise ValueError('Fixed wavevector and target must have shape (frequency, 2).')
    if any(p.fields.requires_grad for p in planes):
        raise ValueError('Calibrate on fixed reference fields under torch.no_grad().')
    if not bool(torch.isfinite(k).all() & torch.isfinite(target).all()) or not 1<=max_condition<float('inf'):
        raise ValueError('Calibration inputs and condition limit must be finite.')
    axes=[i for i in range(3) if i!='xyz'.index(a.normal)]
    phase=torch.exp(-1j*(k@a.points_um[:,axes].T))
    weights=a.weights/a.weights.sum()
    columns=[(p.fields[...,axes]*phase[...,None]*weights[None,:,None]).sum(1) for p in planes]
    matrix=torch.stack(columns,-1)
    condition=torch.linalg.cond(matrix)
    if not bool(torch.isfinite(condition).all()) or bool((condition>max_condition).any()):
        raise ValueError('Source basis is singular or ill-conditioned at a requested frequency.')
    return torch.linalg.solve(matrix,target[...,None])[...,0].detach()


def mix_plane_fields(planes, coefficients):
    """Coherently mix two source-basis solutions with fixed spectral coefficients.

    Solve both bases in the same physical structure, then mix fields before
    evaluating power or allocation. Reuse exactly the same coefficients for
    sample and reference. This is selected-frequency synthesis, not a broadband
    time-domain source or an incoherent average. Structure gradients propagate
    through both solves. Source coefficient gradients are intentionally excluded.
    """
    a,b=_pair(planes)
    c=torch.as_tensor(coefficients,device=a.fields.device,dtype=a.fields.dtype)
    if c.shape!=(a.fields.shape[0],2) or c.requires_grad or not bool(torch.isfinite(c).all()):
        raise ValueError('Mixing coefficients must be finite fixed (frequency, 2) values.')
    fields=a.fields*c[:,0,None,None]+b.fields*c[:,1,None,None]
    signature=hashlib.sha256((a.run_signature+'|'+b.run_signature).encode()+c.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return replace(a,fields=fields,run_signature=signature,
                   report={'polarization_synthesis':'two fixed spectral source responses',
                           'basis_reports':[a.report,b.report]})
