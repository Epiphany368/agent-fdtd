"""Conservative periodic density transfer to a uniform Yee grid."""
import math
import torch
from .solver import field_axes


def _overlap(centers,width,count,period,like,origin):
    edges=torch.linspace(-period/2,period/2,count+1,device=like.device,dtype=like.dtype)
    if origin=='sample_centers':edges=edges-period/(2*count)
    centers=torch.tensor(centers,device=like.device,dtype=like.dtype)
    lo=centers[:,None]-width/2;hi=centers[:,None]+width/2
    result=like.new_zeros((centers.numel(),count))
    for shift in (-period,0.,period):
        result+=(torch.minimum(hi,edges[None,1:]+shift)-torch.maximum(lo,edges[None,:-1]+shift)).clamp_min(0)/width
    return result


def _layer_z_fraction(region, component, bottom_um, top_um, like):
    """Exact dtype/device-rounded Yee overlap shared by transfer and admission."""
    z = torch.tensor(field_axes(region, component)[2], device=like.device, dtype=like.dtype)
    width = region.axis_steps[2]
    return (torch.minimum(z+width/2, z.new_tensor(top_um))
            - torch.maximum(z-width/2, z.new_tensor(bottom_um))).clamp_min(0)/width


def periodic_density_layer(density,region,*,bottom_um,top_um,background_epsilon,design_epsilon,pixel_origin='cell_edges'):
    """Lift xy density pixels into an extruded layer with differentiable averages.

    Density axes are x,y, with piecewise-constant pixels spanning the full
    centered periodic cell. Each electric component receives volume-averaged
    permittivity over a cell-sized box centered at its own Yee location. This
    is arithmetic volume averaging, not a conformal interface prescription.
    Validate optical convergence before using a coarse transfer for design.
    The z interfaces and lossless selected-frequency material values are fixed.
    cell_edges places the first pixel's lower edge at the domain's lower edge.
    sample_centers places its center there, matching a nodal FFT phase origin.
    """
    if not isinstance(density,torch.Tensor) or density.dtype not in (torch.float32,torch.float64) or density.ndim!=2 or min(density.shape)<1:
        raise ValueError('Density must be a nonempty real floating x,y tensor.')
    if not bool(torch.isfinite(density).all()) or bool(((density<0)|(density>1)).any()):
        raise ValueError('Density must be finite in [0,1].')
    if pixel_origin not in ('cell_edges','sample_centers'):raise ValueError('Unknown density pixel origin.')
    if region.dimension!='3d' or region.mesh_type!='uniform' or region.material_sampling!='yee':
        raise ValueError('Density transfer requires a uniform 3D Yee-material grid.')
    if any(face.kind not in ('periodic','bloch') for axis in (0,1) for face in region.boundaries.pair(axis)):
        raise ValueError('Density transfer requires periodic/Bloch x and y boundaries.')
    values=(bottom_um,top_um,background_epsilon,design_epsilon)
    if any(isinstance(v,torch.Tensor) or not math.isfinite(v) for v in values):
        raise ValueError('Layer interfaces and material values must be fixed finite scalars.')
    if bottom_um>=top_um or min(background_epsilon,design_epsilon)<1:
        raise ValueError('Layer thickness must be positive and permittivities at least one.')
    lower,upper=region.interior_bounds(2)
    if bottom_um<lower or top_um>upper:raise ValueError('Patterned layer must lie inside the non-PML region.')
    step=region.axis_steps;period=region.actual_size
    fields=[]
    for component in ('Ex','Ey','Ez'):
        x,y,z=field_axes(region,component)
        wx=_overlap(x,step[0],density.shape[0],period[0],density,pixel_origin)
        wy=_overlap(y,step[1],density.shape[1],period[1],density,pixel_origin)
        xy=wx@density@wy.T
        fraction=_layer_z_fraction(region,component,bottom_um,top_um,density)
        fields.append(background_epsilon+(design_epsilon-background_epsilon)*xy[:,:,None]*fraction)
    return torch.stack(fields,-1)


def _box_overlap(centers, width, count, lower, upper, like):
    """Overlap of each cell-sized box with the pixels of a bounded axis, no wrap."""
    edges = torch.linspace(lower, upper, count+1, device=like.device, dtype=like.dtype)
    centers = torch.tensor(centers, device=like.device, dtype=like.dtype)
    lo = centers[:, None]-width/2
    hi = centers[:, None]+width/2
    return (torch.minimum(hi, edges[None, 1:])-torch.maximum(lo, edges[None, :-1])).clamp_min(0)/width


def bounded_density_layer(density, base_epsilon, region, *, bounds_um, bottom_um, top_um,
                          background_epsilon, design_epsilon):
    """Replace one axis-aligned box of a fixed base epsilon by a density layer.

    Density pixels tile the xy box bounds_um = (x_min, x_max, y_min, y_max)
    uniformly and are extruded from bottom_um to top_um. Each electric
    component receives the volume average, over a cell-sized box centred at
    its own Yee location, of background_epsilon + density * (design_epsilon
    - background_epsilon) inside the design box and of base_epsilon outside
    it, so a component whose box straddles the design boundary blends the two
    by its overlap fraction. This is the same arithmetic volume averaging as
    periodic_density_layer, restricted to a box inside a non-periodic domain;
    only the density carries a Torch graph. A layer that spans a periodic z
    axis exactly is treated as z invariant.
    """
    if not isinstance(density,torch.Tensor) or density.dtype not in (torch.float32,torch.float64) or density.ndim!=2 or min(density.shape)<1:
        raise ValueError('Density must be a nonempty real floating x,y tensor.')
    if not bool(torch.isfinite(density).all()) or bool(((density<0)|(density>1)).any()):
        raise ValueError('Density must be finite in [0,1].')
    if not isinstance(base_epsilon,torch.Tensor) or base_epsilon.shape!=region.shape+(3,) or base_epsilon.dtype!=density.dtype:
        raise ValueError('base_epsilon must be the explicit Yee diagonal epsilon of the region in the density precision.')
    if region.dimension!='3d' or region.mesh_type!='uniform' or region.material_sampling!='yee':
        raise ValueError('Density transfer requires a uniform 3D Yee-material grid.')
    values=(*bounds_um,bottom_um,top_um,background_epsilon,design_epsilon)
    if len(bounds_um)!=4 or any(isinstance(v,torch.Tensor) or not math.isfinite(v) for v in values):
        raise ValueError('Box bounds, layer interfaces and material values must be fixed finite scalars.')
    x0,x1,y0,y1=bounds_um
    if x0>=x1 or y0>=y1 or bottom_um>=top_um or min(background_epsilon,design_epsilon)<1:
        raise ValueError('Box spans and layer thickness must be positive and permittivities at least one.')
    for axis,(lo,hi) in enumerate(((x0,x1),(y0,y1),(bottom_um,top_um))):
        inner=region.interior_bounds(axis)
        if lo<inner[0]-1e-9 or hi>inner[1]+1e-9:
            raise ValueError('The design box must lie inside the non-PML region.')
    step=region.axis_steps
    # A layer spanning a periodic z axis is z invariant: no z fraction, so the
    # node on the periodic seam is not blended with its wrapped image.
    z_periodic=all(face.kind in ('periodic','bloch') for face in region.boundaries.pair(2))
    inner_z=region.interior_bounds(2)
    invariant=z_periodic and math.isclose(bottom_um,inner_z[0],abs_tol=1e-9) and math.isclose(top_um,inner_z[1],abs_tol=1e-9)
    fields=[]
    for c,component in enumerate(('Ex','Ey','Ez')):
        x,y,_=field_axes(region,component)
        wx=_box_overlap(x,step[0],density.shape[0],x0,x1,density)
        wy=_box_overlap(y,step[1],density.shape[1],y0,y1,density)
        weighted=wx@density@wy.T
        inside=wx.sum(1)[:,None]*wy.sum(1)[None,:]
        fz=(density.new_ones(region.shape[2]) if invariant
            else _layer_z_fraction(region,component,bottom_um,top_um,density))
        design=background_epsilon*inside+(design_epsilon-background_epsilon)*weighted
        base=base_epsilon[...,c]
        fields.append(base+fz[None,None,:]*(design[:,:,None]-base*inside[:,:,None]))
    return torch.stack(fields,-1)
