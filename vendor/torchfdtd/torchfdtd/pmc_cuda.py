"""Experimental direct FP32 CUDA for exact-endpoint PEC/PMC fields.

No public simulation dispatch, ADE, sources, checkpoint or budget integration.
Only compact block descriptors and O(Nx+Ny+Nz) metric arrays are retained.
Explicit adjoints are provided. These calls do not construct an autograd graph.
"""
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import torch

from .boundaries import CURL_TERMS
from .cuda_kernels import _compile, _direct_cuda_view
from .pmc_reference import BoundaryBlock, ReferenceState


class CompactEndpointTopology:
    """Same packed ordering as EndpointTopology without per-DOF lookup tables."""
    def __init__(self,nodes,faces):
        self.nodes=tuple(np.array(x,dtype=float,copy=True) for x in nodes)
        self.faces=tuple(tuple(x) for x in faces)
        if len(self.nodes)!=3 or len(self.faces)!=3:
            raise ValueError('Three axes and three PEC/PMC face pairs are required.')
        if any(x.ndim!=1 or len(x)<3 or not np.isfinite(x).all() or np.any(np.diff(x)<=0) for x in self.nodes):
            raise ValueError('Coordinates require at least two positive finite cells per axis.')
        if any(len(p)!=2 or any(k not in ('pec','pmc') for k in p) for p in self.faces):
            raise ValueError('Only PEC/PMC face pairs are supported by this private kernel.')
        for x in self.nodes:x.setflags(write=False)
        self.shape=tuple(len(x)-1 for x in self.nodes)
        self.blocks={};self.counts={}
        for family in ('E','H'):
            count=3*math.prod(self.shape)
            blocks=[BoundaryBlock(family,None,(),(*self.shape,3),0,count)]
            for comp in range(3):
                axes=[a for a in range(3) if (comp!=a if family=='E' else comp==a) and self.faces[a][1]=='pmc']
                for length in range(1,len(axes)+1):
                    for upper in combinations(axes,length):
                        shape=tuple(1 if a in upper else n for a,n in enumerate(self.shape))
                        end=count+math.prod(shape)
                        blocks.append(BoundaryBlock(family,comp,upper,shape,count,end));count=end
            self.blocks[family]=tuple(blocks);self.counts[family]=count
        self.cfl_unit=1/math.sqrt(sum(1/np.diff(x).min()**2 for x in self.nodes))

    def state_bytes(self):return 4*sum(self.counts.values())


@dataclass(frozen=True)
class PreparedEndpointMaterial:
    epsilon: torch.Tensor
    minimum: float
    version: int
    owner: object


class EndpointCUDA:
    """Direct gathers for curls, transpose and two-stage forward/reverse updates.

    Prepare each positive sampled epsilon tensor once before its time loop.
    Ordinary in-place Torch edits invalidate preparation. Writes that bypass
    version counters (raw pointers or .data) remain the caller's responsibility.
    CUDA FP32 contiguous
    packed fields only. Geometry sampling and parameter sharing are external.
    """
    def __init__(self,topology,*,device='cuda',dtype=torch.float32):
        if dtype!=torch.float32:raise ValueError('EndpointCUDA supports real FP32 only.')
        device=torch.device(device)
        if device.type!='cuda' or not torch.cuda.is_available():raise ValueError('EndpointCUDA requires a CUDA device.')
        self.device=torch.device('cuda',torch.cuda.current_device() if device.index is None else device.index)
        # Copy only nodes/faces, never reference sparse operators or DOF maps.
        self.topology=CompactEndpointTopology(topology.nodes,topology.faces)
        import cupy
        self.cp=cupy
        self.metrics=[]
        for x in self.topology.nodes:
            widths=np.diff(x)
            self.metrics.extend(torch.tensor(v,device=self.device,dtype=torch.float32) for v in
                (1/widths,np.r_[2/widths[0],2/(widths[:-1]+widths[1:]),2/widths[-1]]))
        self.source=self._source()
        with cupy.cuda.Device(self.device.index),self._stream():
            first,self.module=_compile(self.source,self.device.index,cupy.cuda.Device(self.device.index).compute_capability,'pmc_curl')
            self.kernels={'pmc_curl':first}
            for name in ('pmc_e','pmc_h','pmc_et','pmc_ht'):
                self.kernels[name]=self.module.get_function(name)

    @property
    def metadata_bytes(self):return sum(x.numel()*x.element_size() for x in self.metrics)

    def _stream(self):return self.cp.cuda.ExternalStream(torch.cuda.current_stream(self.device).cuda_stream,device_id=self.device.index)

    def _tensor(self,value,family):
        if not isinstance(value,torch.Tensor) or value.device!=self.device or value.dtype!=torch.float32 or not value.is_contiguous() or value.shape!=(self.topology.counts[family],):
            raise ValueError('Expected a contiguous packed real FP32 field on the configured CUDA device.')
        if value.requires_grad:raise ValueError('EndpointCUDA uses an explicit adjoint, not an autograd graph. Detach inputs explicitly.')
        return value

    def prepare_epsilon(self,epsilon):
        self._tensor(epsilon,'E')
        if not bool(torch.isfinite(epsilon).all()):raise ValueError('Epsilon must be finite and positive.')
        minimum=float(epsilon.min())
        if minimum<=0:raise ValueError('Epsilon must be finite and positive.')
        try:version=epsilon._version
        except RuntimeError as exc:
            raise ValueError('Epsilon needs a Torch version counter. Inference-mode tensors are unsupported.') from exc
        return PreparedEndpointMaterial(epsilon,minimum,version,self)

    def _material(self,material,dt):
        if not isinstance(material,PreparedEndpointMaterial) or material.owner is not self:
            raise ValueError('Prepare epsilon on this EndpointCUDA instance before stepping.')
        if not math.isfinite(dt) or dt<=0 or dt>math.sqrt(material.minimum)*self.topology.cfl_unit*(1+1e-7):
            raise ValueError('The timestep must be positive and no greater than the conservative Yee CFL limit.')
        self._tensor(material.epsilon,'E')
        if material.epsilon._version!=material.version:
            raise ValueError('Epsilon changed after preparation. Call prepare_epsilon again after parameter updates.')
        return material.epsilon

    def zeros(self):return ReferenceState(*(torch.zeros(self.topology.counts[f],device=self.device,dtype=torch.float32) for f in ('E','H')))

    def _launch(self,name,count,args):
        with self.cp.cuda.Device(self.device.index),self._stream():
            values=[_direct_cuda_view(self.cp,x) if isinstance(x,torch.Tensor) else x for x in args]
            self.kernels[name](((count+255)//256,),(256,),tuple(values))

    def curl(self,value,forward,*,transpose=False):
        source,target=('E','H') if forward else ('H','E')
        if transpose:source,target=target,source
        self._tensor(value,source)
        result=torch.empty(self.topology.counts[target],device=self.device,dtype=torch.float32)
        self._launch('pmc_curl',result.numel(),(value,result,*self.metrics,np.int32(bool(forward)),np.int32(bool(transpose))))
        return result

    def step(self,state,material,dt):
        epsilon=self._material(material,dt)
        self._tensor(state.electric,'E');self._tensor(state.magnetic,'H')
        result=ReferenceState(torch.empty_like(state.electric),torch.empty_like(state.magnetic))
        self._launch('pmc_e',result.electric.numel(),(state.electric,state.magnetic,epsilon,result.electric,*self.metrics,np.float32(dt)))
        self._launch('pmc_h',result.magnetic.numel(),(state.magnetic,result.electric,result.magnetic,*self.metrics,np.float32(dt)))
        return result

    def transpose_step(self,state,seed,material,dt):
        epsilon=self._material(material,dt)
        self._tensor(state.electric,'E');self._tensor(state.magnetic,'H')
        self._tensor(seed.electric,'E');self._tensor(seed.magnetic,'H')
        result=ReferenceState(torch.empty_like(seed.electric),torch.empty_like(seed.magnetic))
        gradient=torch.empty_like(epsilon)
        self._launch('pmc_et',result.electric.numel(),(state.magnetic,seed.electric,seed.magnetic,epsilon,result.electric,gradient,*self.metrics,np.float32(dt)))
        self._launch('pmc_ht',result.magnetic.numel(),(seed.magnetic,result.electric,epsilon,result.magnetic,*self.metrics,np.float32(dt)))
        return result,gradient

    def _source(self):
        t=self.topology;nx,ny,nz=t.shape;code=[]
        for family in ('E','H'):
            code.append(f'__device__ long long idx_{family}(int c,int x,int y,int z) {{')
            code.append(f'if(x<0||y<0||z<0||x>{nx}||y>{ny}||z>{nz})return -1;')
            for a,n in enumerate(t.shape):
                if t.faces[a][0]=='pec':
                    comparison='!=' if family=='E' else '=='
                    code.append(f'if({"xyz"[a]}==0 && c{comparison}{a})return -1;')
            code.append(f'int mask=(x=={nx}?1:0)|(y=={ny}?2:0)|(z=={nz}?4:0);')
            code.append(f'if(mask==0)return 3*((((long long)x)*{ny}+y)*{nz}+z)+c;')
            for b in t.blocks[family][1:]:
                mask=sum(1<<a for a in b.upper_axes)
                q=['0' if a in b.upper_axes else 'xyz'[a] for a in range(3)]
                code.append(f'if(c=={b.component} && mask=={mask})return {b.start}LL+((long long)({q[0]})*{b.shape[1]}+({q[1]}))*{b.shape[2]}+({q[2]});')
            code.append('return -1;}')
            code.append(f'__device__ void decode_{family}(long long i,int &c,int &x,int &y,int &z) {{')
            code.append(f'if(i<{3*math.prod(t.shape)}LL){{c=i%3;i/=3;x=i/{ny*nz};y=(i/{nz})%{ny};z=i%{nz};return;}}')
            for b in t.blocks[family][1:]:
                q=[str(t.shape[a]) if a in b.upper_axes else [f'j/{b.shape[1]*b.shape[2]}',f'(j/{b.shape[2]})%{b.shape[1]}',f'j%{b.shape[2]}'][a] for a in range(3)]
                code.append(f'if(i<{b.stop}LL){{long long j=i-{b.start}LL;c={b.component};x={q[0]};y={q[1]};z={q[2]};return;}}')
            code.append('c=x=y=z=0;}')
            code.append(f'__device__ float read_{family}(const float* src,int c,int x,int y,int z,const float* epsilon,float gain){{long long i=idx_{family}(c,x,y,z);return i<0?0:gain*src[i]/(epsilon?epsilon[i]:1.0f);}}')
        metrics='const float* f0,const float* b0,const float* f1,const float* b1,const float* f2,const float* b2'
        names='f0,b0,f1,b1,f2,b2'
        for forward in (False,True):
            for transpose in (False,True):
                name=('ce' if forward else 'ch')+('t' if transpose else '')
                source,target=('E','H') if forward else ('H','E')
                if transpose:source,target=target,source
                code.append(f'__device__ float {name}(const float* src,int c,int x,int y,int z,{metrics},const float* epsilon=0,float gain=1){{')
                code.append(f'if(idx_{target}(c,x,y,z)<0)return 0;float total=0;')
                for axis,comp,out,sign in CURL_TERMS:
                    wanted=comp if transpose else out;readcomp=out if transpose else comp
                    coord='xyz'[axis];n=t.shape[axis]
                    def read(delta):
                        q=list('xyz');q[axis]=f'({coord}{delta:+d})' if delta else coord
                        return f'read_{source}(src,{readcomp},{",".join(q)},epsilon,gain)'
                    if not transpose:
                        expr=(f'f{axis}[{coord}]*({read(1)}-{read(0)})' if forward else
                              f'({coord}==0?b{axis}[0]*{read(0)}:({coord}=={n}?-b{axis}[{n}]*{read(-1)}:b{axis}[{coord}]*({read(0)}-{read(-1)})))')
                    else:
                        expr=(f'({coord}>0?f{axis}[{coord}-1]*{read(-1)}:0)-({coord}<{n}?f{axis}[{coord}]*{read(0)}:0)' if forward else
                              f'b{axis}[{coord}]*{read(0)}-b{axis}[{coord}+1]*{read(1)}')
                    code.append(f'if(c=={wanted})total+={sign}*({expr});')
                code.append('return total;}')
        ecount,hcount=t.counts['E'],t.counts['H']
        code.append(f'extern "C" __global__ void pmc_curl(const float* src,float* dst,{metrics},int forward,int transpose){{long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;int eout=(forward==transpose);if(i>=(eout?{ecount}LL:{hcount}LL))return;int c,x,y,z;if(eout)decode_E(i,c,x,y,z);else decode_H(i,c,x,y,z);dst[i]=forward?(transpose?cet(src,c,x,y,z,{names}):ce(src,c,x,y,z,{names})):(transpose?cht(src,c,x,y,z,{names}):ch(src,c,x,y,z,{names}));}}')
        code.append(f'extern "C" __global__ void pmc_e(const float* old,const float* h,const float* eps,float* dst,{metrics},float dt){{long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>={ecount}LL)return;int c,x,y,z;decode_E(i,c,x,y,z);dst[i]=idx_E(c,x,y,z)<0?0:old[i]+(dt/eps[i])*ch(h,c,x,y,z,{names});}}')
        code.append(f'extern "C" __global__ void pmc_h(const float* old,const float* e,float* dst,{metrics},float dt){{long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>={hcount}LL)return;int c,x,y,z;decode_H(i,c,x,y,z);dst[i]=idx_H(c,x,y,z)<0?0:old[i]-dt*ce(e,c,x,y,z,{names});}}')
        code.append(f'extern "C" __global__ void pmc_et(const float* primal_h,const float* seed_e,const float* seed_h,const float* eps,float* dst,float* gradient,{metrics},float dt){{long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>={ecount}LL)return;int c,x,y,z;decode_E(i,c,x,y,z);float value=idx_E(c,x,y,z)<0?0:seed_e[i]-dt*cet(seed_h,c,x,y,z,{names});dst[i]=value;gradient[i]=(-dt/(eps[i]*eps[i]))*value*ch(primal_h,c,x,y,z,{names});}}')
        code.append(f'extern "C" __global__ void pmc_ht(const float* seed_h,const float* ebar,const float* eps,float* dst,{metrics},float dt){{long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>={hcount}LL)return;int c,x,y,z;decode_H(i,c,x,y,z);dst[i]=idx_H(c,x,y,z)<0?0:seed_h[i]+cht(ebar,c,x,y,z,{names},eps,dt);}}')
        return '\n'.join(code)
