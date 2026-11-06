"""Native Project adapter for endpoint PEC/PMC and restricted CPML."""
import math
import numpy as np
import torch

from .models import Project
from .geometry import contains
from .waveforms import source_time_signal
from .differentiable import DifferentiableResult
from .pmc_simulation import EndpointSimulation, admit_budget


class EndpointProject:
    """Validated native geometry/source adapter with an optional closed-wall override.

    Without an override, use the native Project face labels. Explicit overrides
    remain separate from the native project. Physical positions use nearest Yee samples with lower-coordinate
    tie breaking. Out-of-domain and wall-constrained requests are rejected,
    never silently moved to the next active sample. Budgets left as None are
    derived at every admission for the bytes admitted: the tensor budget from
    free CUDA memory or available host memory (80%, as the solver does), the
    host preparation budget from 80% of available host memory.
    """
    def __init__(self,project,*,boundary_faces=None,device='cpu',checkpoints=4,
                 tensor_budget_bytes=None,host_preparation_budget_bytes=None):
        self.project=Project.model_validate(project.model_dump() if isinstance(project,Project) else project)
        p=self.project;r=p.region
        if r.dimension!='3d' or r.precision!='float32' or r.complex_fields:
            raise ValueError('Endpoint Project requires real FP32 3D fields.')
        if r.mesh_type not in ('uniform','explicit'):
            raise ValueError('Endpoint Project requires fixed uniform or explicit meshes; material-aware graded meshing is unsupported.')
        if r.memory_mode=='streamed' or r.run_control.auto_shutoff:
            raise ValueError('Endpoint Project requires resident fixed-duration execution.')
        if r.interface_method!='staircase' or r.material_sampling!='yee':
            raise ValueError('Endpoint Project requires staircase geometry and explicit Yee material sampling.')
        active={s.material for s in p.structures if s.enabled}
        if any(m.oscillators for m in p.materials if m.name in active):raise ValueError('Endpoint Project does not support dispersive materials.')
        if any(not s.enabled for s in p.sources) or any(not m.enabled for m in p.monitors):
            raise ValueError('Remove disabled sources/monitors explicitly before endpoint conversion.')
        if boundary_faces is None:
            boundary_faces=tuple(tuple(f.kind for f in r.boundaries.pair(a)) for a in range(3))
        aliases={'pec':'pec','antisymmetric':'pec','pmc':'pmc','symmetric':'pmc','pml':'pml'}
        try:faces=tuple(tuple(aliases[x.lower()] for x in pair) for pair in boundary_faces)
        except (KeyError,AttributeError) as exc:raise ValueError('Provide explicit PEC/PMC/PML boundary faces.') from exc
        if len(faces)!=3 or any(len(pair)!=2 for pair in faces):raise ValueError('Provide three lower/upper face pairs.')
        from .endpoint_native import endpoint_cpml_options
        if any('pml' in pair for pair in faces):
            if any(f.kind not in aliases for a in range(3) for f in r.boundaries.pair(a)):
                raise ValueError('CPML boundary overrides require native PEC/PMC/PML faces, not periodic or Bloch faces.')
            native=tuple(tuple(aliases[f.kind] for f in r.boundaries.pair(a)) for a in range(3))
            if faces!=native:raise ValueError('CPML boundary overrides must match native Project faces and parameters.')
            self.cpml_options=endpoint_cpml_options(r)
        else:self.cpml_options=None
        nodes=tuple(np.array(a,copy=True) for a in r.mesh_nodes)
        if host_preparation_budget_bytes is not None and (isinstance(host_preparation_budget_bytes,bool) or not isinstance(host_preparation_budget_bytes,int) or host_preparation_budget_bytes<=0):
            raise ValueError('A positive host_preparation_budget_bytes, or None to derive it from host memory, is required.')
        self.host_preparation_budget_bytes=host_preparation_budget_bytes
        self.source_records=[];self.observation_records=[];sources=[];observers=[];self._terms=[]
        def location(item,component):
            family=component[0];comp='xyz'.index(component[1].lower());indices=[];actual=[]
            for axis,(values,position) in enumerate(zip(nodes,item.center)):
                if not values[0]<=position<=values[-1]:raise ValueError(f'{item.name}: position lies outside physical mesh endpoints.')
                nodal=comp!=axis if family=='E' else comp==axis
                # Include the physical upper node even for PEC. Resolving that
                # node must reject it, rather than quietly snapping inward.
                coordinates=values if nodal else (values[:-1]+values[1:])/2
                index=int(np.argmin(abs(coordinates-position)))
                indices.append(index);actual.append(float(coordinates[index]))
            return comp,tuple(indices),dict(id=item.id,name=item.name,component=component,
                requested_um=list(item.center),sampled_um=actual,index=indices,
                displacement_um=[a-b for a,b in zip(actual,item.center)],
                sampling='nearest Yee coordinate, lower-coordinate ties')
        for raw in p.sources:
            source=p.resolved_source(raw)
            if source.kind!='point' or source.injection!='soft' or source.component[0]!='E':
                raise ValueError('Endpoint Project supports point soft electric sources only.')
            for component,weight in source.polarization_components:
                comp,index,record=location(source,component);sources.append((comp,index))
                record['polarization_weight']=weight;self.source_records.append(record);self._terms.append((source,weight))
        for monitor in p.monitors:
            if monitor.time_downsample!=1:raise ValueError('Endpoint Project returns every timestep; monitor time_downsample must be 1.')
            if monitor.kind!='point':raise ValueError('Endpoint Project supports point monitors only.')
            comp,index,record=location(monitor,monitor.component)
            observers.append((monitor.component[0],comp,index));self.observation_records.append(record)
        simulation=EndpointSimulation
        if self.cpml_options is not None:
            from .pmc_cpml import EndpointCPMLSimulation
            simulation=EndpointCPMLSimulation
        self.simulation=simulation(nodes,faces,dt_seconds=r.time_step,sources=sources,
            observations=observers,device=device,checkpoints=checkpoints,tensor_budget_bytes=tensor_budget_bytes,
            **(self.cpml_options or {}))
        self._project_fingerprint=self.project.model_dump_json()
        self._admit()

    def _admit(self):
        if self.project.model_dump_json()!=self._project_fingerprint:
            raise ValueError('Endpoint Project changed after preparation. Rebuild the adapter.')
        plan=self.simulation.memory_plan(self.project.region.steps)
        self.simulation.admit_tensors(plan['tensor_upper_bound_bytes'],'Endpoint Project duration exceeds the endpoint tensor budget')
        # Bounded chunk coordinate/membership scratch plus native FP64 pulse
        # arrays and stacking. Solver tensor payload is admitted separately.
        preparation=512*min(65536,self.simulation.topology.counts['E'])+self.project.region.steps*(8+16*len(self._terms))
        admit_budget(preparation,self.host_preparation_budget_bytes,'cpu','Endpoint Project exceeds host preparation byte budget',
                     'host_preparation_budget_bytes')
        return dict(plan,adapter_host_preparation_bytes=preparation)

    def plan(self):
        """JSON-serializable review record. Original Project boundaries are overridden."""
        cpml=None
        if self.cpml_options is not None:
            cpml=dict(self.cpml_options,physical_faces=self.simulation.physical_faces,
                native_sigma_scale=next(f.sigma_scale for a in range(3) for f in self.project.region.boundaries.pair(a) if f.kind=='pml'),
                kappa=1,alpha=0,polynomial=3,alpha_polynomial=0,
                profile='rho=max(1-distance/(layers*h),0) at each endpoint derivative target; b=exp(-rate*c0*dt), c=b-1',
                rate_per_um='40*sigma_scale/((layers+1)*h)*rho**3',
                collar='PML plus one electric-sample cell, exact fixed background epsilon; zero material VJP',
                scalar_yee_profile_equivalence=False)
        return dict(api='endpoint-project',boundary_faces=self.simulation.topology.faces,cpml=cpml,
            original_boundary_faces=[[f.kind for f in self.project.region.boundaries.pair(a)] for a in range(3)],
            nodes_um=[a.tolist() for a in self.simulation.topology.nodes],
            source_terms=self.source_records,observations=self.observation_records,
            disabled_structures_ignored=[s.id for s in self.project.structures if not s.enabled],
            material_sampling='analytic native solids at actual E coordinates including upper faces/edges',
            precedence='lower mesh order wins; later tree object wins ties',
            time_step_seconds=self.project.region.time_step,steps=self.project.region.steps,
            magnetic_units='impedance-scaled Z0 H',memory=self._admit())

    def waveforms(self):
        """Native pulse definitions, one column per reported polarization term."""
        self._admit();r=self.project.region
        times=np.arange(1,r.steps+1)*r.time_step
        array=np.column_stack([source_time_signal(source,times)*weight for source,weight in self._terms]) if self._terms else np.empty((r.steps,0))
        return torch.tensor(array,dtype=torch.float32,device=self.simulation.device)

    def rasterize(self,*,chunk_size=65536):
        """Rasterize native analytic geometry at original FP64 mesh coordinates.

        Chunk temporaries live on CPU. This is a fixed staircase material map,
        not a differentiable shape rasterizer. Pass sampled epsilon instead for
        a differentiable material/geometry parameterization.
        """
        self._admit()
        if isinstance(chunk_size,bool) or not isinstance(chunk_size,int) or not 1<=chunk_size<=65536:
            raise ValueError('Raster chunks must be integers in [1, 65536].')
        p=self.project;t=self.simulation.topology
        active={s.material for s in p.structures if s.enabled}
        material={m.name:m.instantaneous_epsilon for m in p.materials if m.name in active}
        result=torch.empty(t.counts['E'],dtype=torch.float32,device=self.simulation.device)
        for block in t.blocks['E']:
            for start in range(block.start,block.stop,chunk_size):
                stop=min(start+chunk_size,block.stop);offset=np.arange(start,stop)-block.start
                comp=offset%3 if block.component is None else np.full(len(offset),block.component)
                linear=offset//3 if block.component is None else offset;xyz=[]
                for axis,values in enumerate(t.nodes):
                    index=(linear//math.prod(block.shape[axis+1:3]))%block.shape[axis]
                    if axis in block.upper_axes:index=np.full(len(index),t.shape[axis])
                    xyz.append(np.where(comp!=axis,values[index],(values[index]+values[np.minimum(index+1,len(values)-1)])/2))
                epsilon=np.full(stop-start,p.region.background_index**2,dtype=np.float32)
                for obj in sorted(p.structures,key=lambda s:-s.mesh_order):
                    if obj.enabled:epsilon[contains(obj,*xyz)]=material[obj.material]
                result[start:stop]=torch.as_tensor(epsilon,device=self.simulation.device)
        return result

    def validate_material(self,epsilon,waveforms):
        """Shared native-loop input admission; the adapter call also checks these."""
        sim=self.simulation
        for value in (epsilon,waveforms):
            if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.device!=sim.device or not value.is_contiguous():
                raise ValueError('Endpoint inputs must be contiguous FP32 tensors on the configured device.')
            if not bool(torch.isfinite(value).all()):raise ValueError('Endpoint inputs must be finite.')
        if epsilon.shape!=(sim.topology.counts['E'],):raise ValueError('One epsilon per endpoint electric DOF is required.')
        if bool((epsilon<=0).any()):raise ValueError('Endpoint epsilon must be positive.')
        if self.cpml_options is not None and bool((epsilon<1).any()):raise ValueError('Endpoint CPML requires epsilon >= 1.')
        if sim.dt>math.sqrt(float(epsilon.detach().min()))*sim.topology.cfl_unit*(1+1e-7):
            raise ValueError('Endpoint material violates conservative Yee CFL.')
        if self.cpml_options is not None and not bool((epsilon[sim.collar]==sim.background_epsilon).all()):
            raise ValueError('CPML and its one-cell collar require exact fixed isotropic background epsilon.')

    def __call__(self,epsilon=None,waveforms=None):
        """Return native DifferentiableResult; optional tensors retain gradients."""
        self._admit()
        if epsilon is None:epsilon=self.rasterize()
        if waveforms is None:waveforms=self.waveforms()
        if not isinstance(waveforms,torch.Tensor) or waveforms.ndim!=2:
            raise ValueError('Waveforms must be a tensor with shape [steps, source terms].')
        if waveforms.shape[0]!=self.project.region.steps:raise ValueError('Waveform duration must match the fixed native Project.')
        self.validate_material(epsilon,waveforms)
        signals=self.simulation(epsilon,waveforms)
        report=self.plan();report['execution']=self.simulation.last_report
        return DifferentiableResult(signals,self.project.region.time_step,
            tuple(m.component for m in self.project.monitors),report)


def endpoint_from_project(project,*,boundary_faces=None,**options):
    """Create an endpoint adapter from a Project or its dict."""
    return EndpointProject(project,boundary_faces=boundary_faces,**options)
