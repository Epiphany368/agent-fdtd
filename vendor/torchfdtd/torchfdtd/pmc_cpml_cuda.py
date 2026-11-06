"""Direct endpoint CPML gathers and transpose, compact slab descriptors.

Real FP32 only. Multiple launches per derivative are a correctness-oriented
CUDA path, not a fused throughput implementation. No CPU sparse topology.
"""
import math
import numpy as np
import torch

from .boundaries import CURL_TERMS
from .pmc_cuda import EndpointCUDA
from .cuda_kernels import _compile


class EndpointCPMLCUDA(EndpointCUDA):
    def __init__(self,closed,faces,layers,dt,background,reflection):
        # Reuse compact endpoint geometry and already admitted metric tensors.
        self.topology,self.device,self.cp=closed.topology,closed.device,closed.cp
        self.metrics=list(closed.metrics)
        self.forward_groups={False:[],True:[]}
        self.rectangles={};self.psi_lengths={};self.profiles={}
        t=self.topology
        code=closed.source
        metrics='const float* f0,const float* b0,const float* f1,const float* b1,const float* f2,const float* b2'
        for forward in (False,True):
            source,target=('E','H') if forward else ('H','E')
            for axis,component,out,sign in CURL_TERMS:
                key=len(self.psi_lengths);self.forward_groups[forward].append(key)
                rectangles=[];count=0
                for block in t.blocks[target]:
                    if block.component not in (None,out):continue
                    for side in (0,1):
                        if faces[axis][side]!='pml':continue
                        low=[];high=[]
                        for a,n in enumerate(t.shape):
                            node=(out!=a if target=='E' else out==a)
                            lo=n if a in block.upper_axes else int(node and t.faces[a][0]=='pec')
                            hi=n+1 if a in block.upper_axes else n
                            if a==axis:
                                if side==0:hi=min(hi,layers)
                                else:lo=max(lo,n-layers+(1 if node else 0))
                            low.append(lo);high.append(hi)
                        if any(a>=b for a,b in zip(low,high)):continue
                        shape=tuple(b-a for a,b in zip(low,high))
                        rectangles.append((count,tuple(low),tuple(high),shape))
                        count+=math.prod(shape)
                self.rectangles[key]=tuple(rectangles);self.psi_lengths[key]=count
                # Only one-dimensional physical CPML profiles, no volume maps.
                n=t.shape[axis];h=float(t.nodes[axis][1]-t.nodes[axis][0])
                node=(out!=axis if target=='E' else out==axis)
                x=np.arange(n+1)+(0 if node else .5)
                rho=np.zeros_like(x)
                for side in (0,1):
                    if faces[axis][side]=='pml':
                        rho=np.maximum(rho,np.maximum(1-(x if side==0 else n-x)/layers,0))
                rate=-2*math.log(reflection)/(layers*h*math.sqrt(background))*rho**3
                b=torch.tensor(np.exp(-dt*rate),dtype=torch.float32,device=self.device)
                c=torch.tensor(np.expm1(-dt*rate),dtype=torch.float32,device=self.device)
                self.profiles[key]=(b,c);self.metrics.extend((b,c))
                code+=f'\n__device__ long long pi{key}(int c,int x,int y,int z){{if(c!={out})return -1;'
                for start,low,high,shape in rectangles:
                    condition='&&'.join(f'{q}>={lo}&&{q}<{hi}' for q,lo,hi in zip('xyz',low,high))
                    code+=f'if({condition})return {start}LL+((long long)(x-{low[0]})*{shape[1]}+(y-{low[1]}))*{shape[2]}+(z-{low[2]});'
                code+='return -1;}'
                # Effective derivative cotangent includes future psi seed.
                code+=f'\n__device__ float qb{key}(const float* q,const float* p,const float* cp,int x,int y,int z){{long long i=idx_{target}({out},x,y,z);if(i<0)return 0;long long j=pi{key}({out},x,y,z);return q[i]+(j<0?0:cp[{"xyz"[axis]}]*(p[j]+q[i]));}}'
                coordinate='xyz'[axis];n=t.shape[axis]
                def read(delta,reverse=False):
                    q=list('xyz');q[axis]=f'({coordinate}{delta:+d})' if delta else coordinate
                    if reverse:return f'qb{key}(src,oldpsi,cp,{",".join(q)})'
                    return f'read_{source}(src,{component},{",".join(q)},0,1)'
                expr=(f'f{axis}[{coordinate}]*({read(1)}-{read(0)})' if forward else
                      f'({coordinate}==0?b{axis}[0]*{read(0)}:({coordinate}=={n}?-b{axis}[{n}]*{read(-1)}:b{axis}[{coordinate}]*({read(0)}-{read(-1)})))')
                common=f'const float* src,const float* oldpsi,float* dst,float* newpsi,const float* bp,const float* cp,{metrics}'
                code+=f'\nextern "C" __global__ void cf{key}({common}){{long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>={t.counts[target]}LL)return;int c,x,y,z;decode_{target}(i,c,x,y,z);if(c!={out}||idx_{target}(c,x,y,z)<0)return;float d={sign}*({expr});long long j=pi{key}(c,x,y,z);if(j>=0){{float p=bp[{coordinate}]*oldpsi[j]+cp[{coordinate}]*d;newpsi[j]=p;d+=p;}}dst[i]+=d;}}'
                expr=(f'({coordinate}>0?f{axis}[{coordinate}-1]*{read(-1,True)}:0)-({coordinate}<{n}?f{axis}[{coordinate}]*{read(0,True)}:0)' if forward else
                      f'b{axis}[{coordinate}]*{read(0,True)}-b{axis}[{coordinate}+1]*{read(1,True)}')
                code+=f'\nextern "C" __global__ void ct{key}({common}){{long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;int c,x,y,z;if(i<{t.counts[target]}LL){{decode_{target}(i,c,x,y,z);long long j=pi{key}(c,x,y,z);if(j>=0)newpsi[j]=bp[{coordinate}]*(oldpsi[j]+src[i]);}}if(i>={t.counts[source]}LL)return;decode_{source}(i,c,x,y,z);if(c!={component}||idx_{source}(c,x,y,z)<0)return;dst[i]+={sign}*({expr});}}'
        for family in ('E','H'):
            code+=f'\nextern "C" __global__ void project_{family}(const float* src,float* dst){{long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>={t.counts[family]}LL)return;int c,x,y,z;decode_{family}(i,c,x,y,z);dst[i]=idx_{family}(c,x,y,z)<0?0:src[i];}}'
        condition=[]
        for axis,n in enumerate(t.shape):
            # CPU collar uses physical <= and >= including its interface node.
            coordinate=f'({"xyz"[axis]}+(c=={axis}?0.5:0.0))'
            if faces[axis][0]=='pml':condition.append(f'{coordinate}<={layers+1}')
            if faces[axis][1]=='pml':condition.append(f'{coordinate}>={n-layers-1}')
        code+=f'\nextern "C" __global__ void collar(float* dst){{long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>={t.counts["E"]}LL)return;int c,x,y,z;decode_E(i,c,x,y,z);dst[i]=({"||".join(condition)});}}'
        self.source=code
        with self.cp.cuda.Device(self.device.index),self._stream():
            first,self.module=_compile(code,self.device.index,self.cp.cuda.Device(self.device.index).compute_capability,'cf0')
            names=[f'{p}{k}' for k in self.psi_lengths for p in ('cf','ct')]+['project_E','project_H','collar']
            self.kernels={name:self.module.get_function(name) for name in names}
        self.psi_count=sum(self.psi_lengths.values())

    def collar_mask(self):
        result=torch.empty(self.topology.counts['E'],device=self.device,dtype=torch.float32)
        self._launch('collar',len(result),(result,))
        return result.bool()

    def project(self,value,family):
        result=torch.empty_like(value)
        self._launch('project_'+family,len(result),(value,result))
        return result

    def curl(self,field,psi,forward):
        target='H' if forward else 'E'
        result=field.new_zeros(self.topology.counts[target]);updated=list(psi)
        for key in self.forward_groups[forward]:
            updated[key]=torch.empty_like(psi[key])
            self._launch('cf'+str(key),len(result),(field,psi[key],result,updated[key],
                *self.profiles[key],*self.metrics[:6]))
        return result,tuple(updated)

    def transpose(self,seed,psi_seed,forward):
        source='E' if forward else 'H'
        result=seed.new_zeros(self.topology.counts[source]);previous=list(psi_seed)
        for key in self.forward_groups[forward]:
            previous[key]=torch.empty_like(psi_seed[key])
            self._launch('ct'+str(key),max(len(seed),len(result)),(seed,psi_seed[key],result,
                previous[key],*self.profiles[key],*self.metrics[:6]))
        return result,tuple(previous)
