"""One sparse interface launch per independent CUDA cohort, without atomics."""
import torch

from .cuda_kernels import _compile


class SubpixelCUDA:
    def __init__(self,grids):
        import cupy
        self.cp=cupy;self.device=grids[0].E.device.index
        self.batch_size=len(grids)
        self.count=max(len(g.subpixel.rows) for g in grids)
        if not self.count:return
        real='double' if grids[0].E.dtype==torch.float64 else 'float'
        source=r'''
extern "C" __global__ void subpixel_update(const long long* table) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long* t=table+6*blockIdx.y;
    if(i>=t[5])return;
    const long long* rows=(const long long*)t[0];
    const int* cols=(const int*)t[1];
    const REAL* coeff=(const REAL*)t[2];
    const REAL* curl=(const REAL*)t[3];
    REAL* field=(REAL*)t[4];
    REAL sum=0;
    for(int k=0;k<8;k++)sum+=coeff[8*i+k]*curl[cols[8*i+k]];
    field[rows[i]]+=(REAL)(COURANT)*sum;
}
'''.replace('REAL',real).replace('COURANT',f'{grids[0].courant_number:.17g}')
        self.table=torch.tensor([[g.subpixel.rows.data_ptr(),g.subpixel.columns.data_ptr(),
                                 g.subpixel.values.data_ptr(),g.subpixel.curl_buffer.data_ptr(),g.E.data_ptr(),len(g.subpixel.rows)]
                                for g in grids],device=grids[0].E.device,dtype=torch.int64)
        with cupy.cuda.Device(self.device),self.stream():
            self.kernel,self.module=_compile(source,self.device,cupy.cuda.Device(self.device).compute_capability,'subpixel_update')
            self.view=cupy.from_dlpack(self.table)

    def stream(self):return self.cp.cuda.ExternalStream(torch.cuda.current_stream(self.device).cuda_stream,device_id=self.device)

    def update(self):
        if not self.count:return
        with self.cp.cuda.Device(self.device),self.stream():
            self.kernel(((self.count+255)//256,self.batch_size),(256,),(self.view,))
