"""One live-line plus surface-correction CUDA launch per TFSF family/group.

E-phase reads old incident H and updates incident E. H-phase reads the new
incident E and updates incident H. Those operations share a launch without a
cross-block read/write dependency. Different boxes in one grid execute in
source order, while independent grids share the launch's second dimension.
"""
import math
import numpy as np
import torch

from .cuda_kernels import _compile


class CudaTfsfInjection:
    def __init__(self,groups,counter):
        import cupy
        self.cp=cupy;self.device=counter.device.index
        self.owners=[counter];self.launches={'E':[],'H':[]}
        real='double' if groups[0][0].e.dtype==torch.float64 else 'float'
        source=r'''
typedef REAL real;
extern "C" __global__ void tfsf_step(const long long* table, const real* factors,
                                      const long long* counter, int forward) {
    const long long* row=table+13*blockIdx.y;
    real* field=(real*)row[0];
    const long long* target=(const long long*)row[1];
    const long long* sample=(const long long*)row[2];
    const real* weight=(const real*)row[3];
    real* e=(real*)row[4]; real* h=(real*)row[5]; real* psi=(real*)row[6];
    const real* b=(const real*)row[7]; const real* c=(const real*)row[8];
    const real* drive=(const real*)row[9];
    long long count=row[10], tasks=row[11], drive_at=row[12];
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) {
        real diff=forward ? (i+1<count ? e[i+1]-e[i] : (real)0)
                          : (i>0 ? h[i]-h[i-1] : (real)0);
        real p=b[i]*psi[i]+c[i]*diff;
        psi[i]=p;
        real scale=factors[2*blockIdx.y+(forward ? 1 : 0)];
        if(forward) h[i]-=scale*(diff+p);
        else {
            e[i]-=scale*(diff+p);
            if(i==drive_at) e[i]+=drive[counter[0]];
        }
    } else if(i<count+tasks) {
        long long j=i-count;
        const real* incident=forward ? e : h;
        field[target[j]]+=weight[j]*incident[sample[j]];
    }
}
'''.replace('REAL',real)
        with cupy.cuda.Device(self.device),self._stream():
            self.kernel,self.module=_compile(source,self.device,
                cupy.cuda.Device(self.device).compute_capability,'tfsf_step')
            self.counter_view=cupy.from_dlpack(counter)
            for group in groups:
                factors=torch.tensor([[s.ce,s.c] for s in group],device=counter.device,dtype=group[0].e.dtype)
                self.owners.append(factors)
                for family in ('E','H'):
                    descriptors=[];length=0
                    for s in group:
                        target,sample,weight=s.maps[family]
                        field=s.grid.E if family=='E' else s.grid.H
                        psi=s.pe if family=='E' else s.ph
                        b,c=s.coefficients[int(family=='H')]
                        arrays=(field,target,sample,weight,s.e,s.h,psi,b,c,s.drive)
                        self.owners.extend(arrays)
                        descriptors.append([v.data_ptr() for v in arrays]+[s.count,target.numel(),s.drive_at])
                        length=max(length,s.count+target.numel())
                    table=torch.tensor(descriptors,device=counter.device,dtype=torch.int64)
                    self.owners.append(table)
                    self.launches[family].append((cupy.from_dlpack(table),cupy.from_dlpack(factors),
                                                 (math.ceil(length/256),len(group))))

    def _stream(self):
        return self.cp.cuda.ExternalStream(torch.cuda.current_stream(self.device).cuda_stream,device_id=self.device)

    def inject(self,family):
        with self.cp.cuda.Device(self.device),self._stream():
            for table,factors,blocks in self.launches[family]:
                self.kernel(blocks,(256,),(table,factors,self.counter_view,np.int32(family=='H')))
