"""Fixed native geometry to common-node Cartesian tensor permittivity."""
import math
import numpy as np
import torch


def uses_tensor(project):
    active={s.material for s in project.structures if s.enabled}
    return any(m.name in active and m.model=='tensor' for m in project.materials)


def validate_tensor_project(project):
    """Metadata-only native scope admission, before scalar material preparation."""
    r=project.region
    if r.dimension!='3d' or r.precision!='float32' or r.mesh_type!='uniform':
        raise ValueError('Native tensor materials require FP32 material on uniform 3D grids.')
    if r.memory_mode!='resident' or r.run_control.auto_shutoff:
        raise ValueError('Native tensor materials require resident fixed-duration execution.')
    if r.interface_method!='staircase' or r.material_sampling!='yee':
        raise ValueError('Native tensor materials require staircase geometry and Yee fields.')
    if any(f.kind not in ('periodic','bloch','pml') for a in range(3) for f in r.boundaries.pair(a)):
        raise ValueError('Native tensor faces must be periodic/Bloch or isotropic-exterior PML.')
    if any(m.model not in ('dielectric','tensor') for m in project.materials):
        raise ValueError('Native tensor projects cannot mix ADE materials.')
    if any(s.enabled and (s.kind!='point' or s.injection!='soft' or not s.component.startswith('E')) for s in project.sources):
        raise ValueError('Native tensor sources must be point soft electric field increments.')
    if not any(m.enabled for m in project.monitors) or any(m.enabled and (m.kind!='point' or m.time_downsample!=1) for m in project.monitors):
        raise ValueError('Native tensor requires point monitors at every timestep.')


class TensorProject:
    """Fixed node ownership, differentiable material table, no shape/source VJP.

    material_parameters is FP32 [number of materials,6], in Cartesian order
    xx,yy,zz,xy,xz,yz. Every scalar material defaults to epsilon times identity.
    Caller optimizer graphs are outside the reservation.
    """
    def __init__(self,project,*,device='cpu',checkpoints=4,tensor_budget_bytes=None):
        from .models import Project
        from .anisotropy import TensorDielectricSimulation
        from .differentiable import AdjointOptions
        self.project=Project.model_validate(project.model_dump() if isinstance(project,Project) else project)
        validate_tensor_project(self.project)
        self.device=torch.device(device)
        if self.device.type=='cuda' and self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        if self.device.type not in ('cpu','cuda'):raise ValueError('Tensor adapter requires CPU or CUDA.')
        if tensor_budget_bytes is not None and (isinstance(tensor_budget_bytes,bool) or not isinstance(tensor_budget_bytes,int) or tensor_budget_bytes<=0):
            raise ValueError('tensor_budget_bytes must be a positive integer.')
        self.tensor_budget_bytes=tensor_budget_bytes
        cells=math.prod(self.project.region.shape)
        # Ownership CPU/device copies, gathered six entries and graph indices,
        # matrix assembly scratch, material table, bounded geometry coordinates.
        self.adapter_bytes=128*cells+512*min(cells,65536)+128*(len(self.project.materials)+1)
        remaining=None if tensor_budget_bytes is None else tensor_budget_bytes-self.adapter_bytes
        if remaining is not None and remaining<=0:raise ValueError('Tensor adapter preparation exceeds tensor budget.')
        options=AdjointOptions(checkpoints=checkpoints,resident_budget_bytes=remaining,backward_kernel='torch')
        self.simulation=TensorDielectricSimulation(self.project,options,cpml_background_epsilon=self.project.region.background_index**2)
        self._fingerprint=self.project.model_dump_json()
        self._ownership=None
        self._admit()

    def _admit(self):
        if self.project.model_dump_json()!=self._fingerprint or self.simulation.project.model_dump_json()!=self._fingerprint:
            raise ValueError('Tensor Project changed after preparation. Rebuild the adapter.')
        plan=self.simulation.reservation(device=self.device)
        from .memory_profile import host_memory
        host=plan['host_reservation_bytes']+self.adapter_bytes
        available=host_memory()['available_bytes']
        if available is not None and host>.8*available:raise ValueError('Tensor adapter exceeds available host memory.')
        total=plan['memory_reservation_bytes']+self.adapter_bytes
        if self.tensor_budget_bytes is not None and total>self.tensor_budget_bytes:raise ValueError('Tensor adapter exceeds tensor budget.')
        if self.device.type=='cuda':
            from .cuda_memory import cuda_budget_limit
            if total>cuda_budget_limit(self.device,total,None):raise ValueError('Tensor adapter exceeds available CUDA memory.')
        return dict(plan,memory_reservation_bytes=total,host_reservation_bytes=host,
                    gpu_reservation_bytes=total if self.device.type=='cuda' else 0,
                    adapter_preparation_bytes=self.adapter_bytes)

    def _owners(self):
        if self._ownership is None:
            from .geometry import contains
            p=self.project;shape=p.region.shape;axes=tuple(v[:-1] for v in p.region.mesh_nodes)
            count=math.prod(shape);owners=np.full(count,len(p.materials),dtype=np.int64)
            ids={m.name:i for i,m in enumerate(p.materials)}
            for start in range(0,count,65536):
                stop=min(start+65536,count);linear=np.arange(start,stop);xyz=[]
                for a in range(3):xyz.append(axes[a][(linear//math.prod(shape[a+1:]))%shape[a]])
                for solid in sorted(p.structures,key=lambda s:-s.mesh_order):
                    if solid.enabled:owners[start:stop][contains(solid,*xyz)]=ids[solid.material]
            self._ownership=torch.tensor(owners,device=self.device)
        return self._ownership

    def rasterize(self,material_parameters=None):
        self._admit();p=self.project
        if material_parameters is None:
            values=[]
            for m in p.materials:
                e=m.index**2
                values.append(m.epsilon_tensor if m.model=='tensor' else (e,e,e,0.,0.,0.))
            material_parameters=torch.tensor(values,dtype=torch.float32,device=self.device).reshape(len(values),6)
        table=material_parameters
        if not isinstance(table,torch.Tensor) or table.shape!=(len(p.materials),6) or table.dtype!=torch.float32 or table.device!=self.device or table.layout!=torch.strided:
            raise ValueError('Material parameters require FP32 [number of materials,6] on the configured device.')
        if not bool(torch.isfinite(table).all()):raise ValueError('Material parameters must be finite.')
        bg=p.region.background_index**2
        values=torch.cat((table,table.new_tensor([[bg,bg,bg,0.,0.,0.]])),0).index_select(0,self._owners())
        xx,yy,zz,xy,xz,yz=values.unbind(-1)
        epsilon=torch.stack((xx,xy,xz,xy,yy,yz,xz,yz,zz),-1).reshape(*p.region.shape,3,3)
        self.simulation._validate_cpml_collar(epsilon)
        return epsilon

    def validate_material(self,epsilon):
        """Shared bounded admission for native forward and differentiable calls."""
        self._admit()
        self.simulation._validate_input_shape(epsilon)
        if epsilon.device!=self.device or epsilon.layout!=torch.strided:
            raise ValueError('Node tensor must use the configured device and dense strided layout.')
        with torch.no_grad():
            if not bool(torch.isfinite(epsilon).all()) or not torch.equal(epsilon,epsilon.transpose(-1,-2)):
                raise ValueError('Node tensors must be finite and exactly symmetric.')
            invalid=torch.zeros((),dtype=torch.bool,device=self.device)
            for batch in epsilon.reshape(-1,3,3).split(64):
                invalid.logical_or_((torch.linalg.eigvalsh(batch)<1).any())
            if bool(invalid):raise ValueError('Node tensors require eigenvalues >= 1.')
            self.simulation._validate_cpml_collar(epsilon)

    def observations(self):
        from .solver import index_at,field_axes
        r=self.project.region
        return [dict(name=m.name,component=m.component,index=index_at(m.center,r,m.component),
                     sampled_um=[float(a[i]) for a,i in zip(field_axes(r,m.component),index_at(m.center,r,m.component))])
                for m in self.project.monitors if m.enabled]

    def plan(self):
        return dict(api='tensor-project',sampling='common mesh_nodes[:-1], normalized incident Yee edge triplets',
                    material_components=('xx','yy','zz','xy','xz','yz'),shape_gradients=False,
                    source_contract='fixed point soft electric field increments',
                    cpml_background_epsilon=self.project.region.background_index**2,
                    cpml_collar='fixed isotropic PML plus one node row, zero material VJP',
                    observations=self.observations(),memory=self._admit())

    def __call__(self,material_parameters=None):
        return self.simulation(self.rasterize(material_parameters))


def tensor_from_project(project,*,device='cpu',checkpoints=4,tensor_budget_bytes=None):
    return TensorProject(project,device=device,checkpoints=checkpoints,tensor_budget_bytes=tensor_budget_bytes)
