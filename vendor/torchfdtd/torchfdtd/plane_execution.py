"""Fixed-plane admission and proxy objectives for unified execution tuning."""
from types import SimpleNamespace
from dataclasses import replace
import math

import torch

from .adjoint_planes import DifferentiablePlaneSimulation
from .adjoint_memory import _resident_reservation, _material_shapes
from .memory_profile import host_memory
from .streamed import StreamedAdjointOptions, _reservation
from .streamed_tuning import _TuningWorkload, _DispersiveTuningWorkload


def plane_mode(project):
    active=[m for m in project.monitors if m.enabled]
    fields=[m.kind=='field' for m in active]
    if any(fields) and not all(fields):
        raise ValueError('Execution selection requires all point monitors or all fixed field planes.')
    return bool(fields) and all(fields)


def plane_model(project,options,dispersive,quadrature_counts=None):
    kind=DifferentiablePlaneSimulation
    if dispersive:
        from .dispersive_adjoint import DispersivePlaneSimulation
        kind=DispersivePlaneSimulation
    return kind(project,options,quadrature_counts=quadrature_counts)


def plane_reservation(model,shapes,options,frequency_hz,*,device='cpu',block_size=32):
    if frequency_hz is None:raise ValueError('Fixed-plane execution requires frequency_hz.')
    region=model.project.region
    if shapes[0] not in (region.shape,region.shape+(3,)):
        raise ValueError('epsilon shape must match the grid, optionally with three components.')
    poles=elements=0
    if len(shapes)==4:
        shapes,poles=_material_shapes(region,shapes)
        elements=sum(math.prod(s) for s in shapes)
    elif len(shapes)!=1:raise ValueError('Provide epsilon or four ADE material shapes.')
    scalar=torch.empty((),dtype=getattr(torch,region.precision),device='cpu')
    spectral=model._spectral(scalar,frequency_hz,block_size)
    internal=model.model.project
    if isinstance(options,StreamedAdjointOptions):
        epsilon=torch.empty(shapes[0],dtype=scalar.dtype,device='meta')
        result=_reservation(internal,epsilon,options,spectral,
            pole_count=poles,parameter_shapes=shapes if poles else None)
        depth=min(options.temporal_depth,region.steps)
    else:
        # This planner also admits batch replay before enable_grad. An
        # adjoint reservation must not depend on the caller's grad mode.
        result=_resident_reservation(internal,options,device,spectral,
            pole_count=poles,parameter_elements=elements,forward_admission=False)
        depth=spectral.block_size
    result.update(spectral.reservation(depth))
    layout=model.layout_reservation_bytes
    host=result['host_reservation_bytes']+layout
    available=host_memory()['available_bytes']
    if available is not None and host>int(available*.8):
        raise ValueError('Plane layout and solver reservation exceed available host memory.')
    if isinstance(options,StreamedAdjointOptions) and host>options.host_budget_bytes:
        raise ValueError('Plane layout and streamed solver reservation exceed the host budget.')
    if not isinstance(options,StreamedAdjointOptions) and torch.device(device).type=='cpu' and options.resident_budget_bytes is not None and host>options.resident_budget_bytes:
        raise ValueError('Plane layout and solver reservation exceed the resident byte budget.')
    points=spectral.point_count
    result.update(host_reservation_bytes=host,plane_layout_reservation_bytes=layout,
        plane_result_metadata_bytes=(4*points+len(model.plans)*spectral.frequency.numel())*scalar.element_size())
    return result


class PlaneTuningWorkload:
    def __init__(self,project,epsilon,materials,frequency_hz,window,quadrature_counts):
        if frequency_hz is None:raise ValueError('Plane tuning requires frequency_hz.')
        if window is not None:raise ValueError('Plane tuning does not yet support a temporal window.')
        self.project=project
        self.dispersive=bool(materials)
        self.quadrature_counts=quadrature_counts
        self.planner=plane_model(project,StreamedAdjointOptions(device='cpu'),self.dispersive,quadrature_counts)
        internal=self.planner.model.project
        self.base=(_DispersiveTuningWorkload(internal,epsilon,*materials)
            if materials else _TuningWorkload(internal,epsilon))
        self.spectral=self.planner._spectral(epsilon,frequency_hz,32)

    def __getattr__(self,name):return getattr(self.base,name)

    @property
    def parameter_bytes(self):
        # The tuning planner and prefix/reference models retain metadata too.
        # The bounded tuner's comparison overhead includes this allowance.
        return self.base.parameter_bytes+self.planner.layout_reservation_bytes

    def output_bytes(self,steps):return self.spectral.reservation(1)['plane_output_bytes']

    def reservation(self,options):
        return plane_reservation(self.planner,tuple(tuple(v.shape) for v in self.design),
            options,self.spectral.frequency)

    def execution_reservation(self,policy):
        from .execution_tuning import _resident_reservation as resident, _streamed_options
        if policy.streamed is not None:return self.reservation(_streamed_options(policy))
        return resident(self.project,tuple(tuple(v.shape) for v in self.design),policy,
            self.spectral.frequency,quadrature_counts=self.quadrature_counts)

    def execution_model(self,project,policy):
        return policy.simulation(project,dispersive=self.dispersive,quadrature_counts=self.quadrature_counts)

    def evaluate(self,model,length):
        planes=model(*self.design,frequency_hz=self.spectral.frequency)
        scale=length*self.project.region.time_step
        values=torch.cat([p.fields.reshape(-1) for p in planes.values()])/scale
        # Exercise both field-amplitude and E/H cross-product derivatives. The
        # flux term is area-normalized so SI quadrature weights cannot hide it.
        loss=values.abs().square().sum()
        for plane in planes.values():
            normalized=replace(plane,fields=plane.fields/scale,weights=plane.weights/plane.weights.sum())
            loss=loss+normalized.flux().sum()
        gradients=torch.autograd.grad(loss,self.design)
        report=next(iter(planes.values())).report
        return SimpleNamespace(report=report),values.detach(),gradients
