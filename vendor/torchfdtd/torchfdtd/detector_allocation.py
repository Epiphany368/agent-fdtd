"""Electric-intensity allocation proxies, distinct from local absorbed power."""
import torch


def quadrant_intensity_allocation(plane, total_transmission, *, split_um=(0.,0.)):
    """Return (frequency, R/G2/G1/B) transmission fractions for an xy plane.

    Split coordinates are physical x/y in the simulation's coordinate system.
    R is low-x/low-y, G2 high-x/low-y, G1 low-x/high-y, B high-x/high-y.
    Points on a split belong to its high side. Every quadrant must be sampled.
    Weights integrate electric intensity, whose relative shares are multiplied
    by the caller's nonnegative total transmission at each frequency. Both the
    fields and total transmission remain differentiable. No electron yield,
    absorption, local flux, pupil average or brightness renormalization is added.
    """
    fields,points,weights=plane.fields,plane.points_um,plane.weights
    if plane.normal!='z' or tuple(plane.components[:3])!=('Ex','Ey','Ez'):
        raise ValueError('Quadrant allocation requires an xy plane with Ex, Ey, Ez.')
    if fields.ndim!=3 or fields.shape[-1]<3 or not (fields.is_floating_point() or fields.is_complex()):
        raise ValueError('Plane fields must have frequency, point, component axes.')
    dtype=fields.real.dtype
    if points.shape!=(fields.shape[1],3) or weights.shape!=(fields.shape[1],):
        raise ValueError('Plane points and quadrature weights must match the field samples.')
    for value in (points,weights,total_transmission):
        if not isinstance(value,torch.Tensor) or value.device!=fields.device or value.dtype!=dtype:
            raise ValueError('Coordinates, weights and transmission must match the real field dtype and device.')
        if not bool(torch.isfinite(value).all()):raise ValueError('Allocation inputs must be finite.')
    if total_transmission.shape!=(fields.shape[0],) or bool((total_transmission<0).any()):
        raise ValueError('Total transmission must be nonnegative with one value per frequency.')
    if fields.shape[0]==0 or not bool(torch.isfinite(fields).all()) or bool((weights<=0).any()):
        raise ValueError('Fields must be nonempty and finite, with positive quadrature weights.')
    split=torch.as_tensor(split_um,device=fields.device,dtype=dtype)
    if split.shape!=(2,) or not bool(torch.isfinite(split).all()) or split.requires_grad:
        raise ValueError('Split must contain two finite fixed coordinates.')
    if points.requires_grad or weights.requires_grad:
        raise ValueError('Detector quadrature must be fixed during differentiation.')
    low_x=points[:,0]<split[0];low_y=points[:,1]<split[1]
    masks=torch.stack((low_x&low_y,~low_x&low_y,low_x&~low_y,~low_x&~low_y))
    if not bool(masks.any(dim=1).all()):raise ValueError('Every quadrant must contain samples.')
    # Scale fields and areas before squaring, as normalized_flux does. SI-scale
    # spectral fields times m^2 areas fall below the float32 normal range, which
    # loses the forward ratio and overflows its backward seed. Both detached
    # scales cancel from the ratio and leave its derivatives unchanged.
    field_scale=fields[...,:3].detach().abs().amax(dim=(1,2),keepdim=True)
    if bool((field_scale<=0).any()):
        raise ValueError('Integrated electric intensity must be finite and positive.')
    intensity=(fields[...,:3]/field_scale).abs().square().sum(-1)
    integrals=(intensity*(weights/weights.detach().amax()))@masks.to(dtype).T
    total=integrals.sum(-1,keepdim=True)
    if not bool(torch.isfinite(total).all()) or bool((total<=0).any()):
        raise ValueError('Integrated electric intensity must be finite and positive.')
    return integrals/total*total_transmission[:,None]
