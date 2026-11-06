"""Two fused, read-only reductions for full-domain stopping diagnostics.

Arrays remain owned by Torch. Each block reduces one contiguous array segment
in double precision, then a second kernel combines partials. No atomics, field
copies, or whole-domain double-precision temporaries are needed.
"""
import math

import numpy as np
import torch

from .cuda_kernels import _compile


class CudaStateDiagnostics:
    def __init__(self, diagnostics):
        import cupy
        self.cp = cupy
        g = diagnostics.grid
        self.device = g.E.device.index
        self.arrays = []
        descriptors, factors, tasks = [], [], []

        def add(array, kind, weight=None, inverse=None, factor=1.):
            if not array.is_contiguous():
                raise ValueError('Diagnostic arrays must be contiguous.')
            self.arrays.append(array)
            for tensor in (weight, inverse):
                if tensor is not None:
                    self.arrays.append(tensor)
            d = len(descriptors)
            descriptors.append([array.data_ptr(), array.numel(), kind,
                                weight.data_ptr() if weight is not None else 0,
                                inverse.data_ptr() if inverse is not None else 0])
            factors.append(factor)
            # Four elements per thread amortize descriptor/reduction work.
            tasks.extend((d, start) for start in range(0, array.numel(), 1024))

        add(g.E, 0, diagnostics.volume, diagnostics.inverse)
        add(g.H, 1, diagnostics.volume)
        for state, weight in zip(g.material_states, diagnostics.material_weights):
            for j, (w0, strength, _) in enumerate(state.oscillators):
                p = state.P[j] if state.multiple else state.P
                q = state.Q[j] if state.multiple else state.Q
                add(q, 2, weight, factor=1/(g.time_step*math.sqrt(strength)))
                # Zero-frequency P is still checked for non-finite values.
                add(p, 2, weight, factor=w0/math.sqrt(strength))
        for array, weight in diagnostics.dispersive_terms:
            add(array, 2, weight)
        for segments in g.cpml.values():
            for segment in segments:
                add(segment['psi'], 3)
        for state in g.incident_states:
            add(state.e,2,state.norm_weight,factor=g.region.background_index)
            add(state.h,2,state.norm_weight)
            add(state.pe,3);add(state.ph,3)
        device = g.E.device
        self.descriptors = torch.tensor(descriptors, device=device, dtype=torch.int64)
        self.factors = torch.tensor(factors, device=device, dtype=torch.float64)
        self.tasks = torch.tensor(tasks, device=device, dtype=torch.int64)
        self.partial = torch.empty((len(tasks), 3), device=device, dtype=torch.float64)
        self.output = torch.empty(3, device=device, dtype=torch.float64)
        self.blocks = len(tasks)
        real = 'double' if g.E.real.dtype == torch.float64 else 'float'
        stride = 2 if g.E.is_complex() else 1
        source = r'''
typedef REAL real;
__device__ void reduce(double sum, double peak, double bad, double* out) {
    __shared__ double sums[256], peaks[256], bads[256];
    int t = threadIdx.x;
    sums[t] = sum; peaks[t] = peak; bads[t] = bad;
    __syncthreads();
    for (int s=128; s>0; s/=2) {
        if (t<s) {
            sums[t] += sums[t+s];
            peaks[t] = fmax(peaks[t], peaks[t+s]);
            bads[t] = fmax(bads[t], bads[t+s]);
        }
        __syncthreads();
    }
    if (t==0) { out[0]=sums[0]; out[1]=peaks[0]; out[2]=bads[0]; }
}
extern "C" __global__ void diagnostic_partial(
    const long long* desc, const double* factors, const long long* tasks, double* out) {
    long long d = tasks[2*blockIdx.x], start = tasks[2*blockIdx.x+1];
    const long long* a = desc+5*d;
    const real* data = (const real*)a[0];
    const real* weight = (const real*)a[3];
    const real* inverse = (const real*)a[4];
    long long n=a[1], kind=a[2];
    double sum=0, peak=0, bad=0;
    for (long long i=start+threadIdx.x; i<n && i<start+1024; i+=256) {
        double re = (double)data[STRIDE*i];
        double im = STRIDE==2 ? (double)data[STRIDE*i+1] : 0.;
        if (!isfinite(re) || !isfinite(im)) bad=1;
        if (kind==3) continue;
        double scale=factors[d];
        double x=re*scale, y=im*scale;
        double w=(double)weight[kind<2 ? i/3 : i];
        if (kind==0) w /= (double)inverse[i];
        sum += (x*x+y*y)*w;
        if (kind<2) peak=fmax(peak,hypot(re,im));
    }
    reduce(sum,peak,bad,out+3*blockIdx.x);
}
extern "C" __global__ void diagnostic_final(const double* partial, double* out, int blocks) {
    double sum=0, peak=0, bad=0;
    for (int i=threadIdx.x; i<blocks; i+=256) {
        sum += partial[3*i];
        peak=fmax(peak,partial[3*i+1]);
        bad=fmax(bad,partial[3*i+2]);
    }
    reduce(sum,peak,bad,out);
}
'''.replace('REAL', real).replace('STRIDE', str(stride))
        with cupy.cuda.Device(self.device), self._stream():
            self.kernel, self.module = _compile(source, self.device,
                cupy.cuda.Device(self.device).compute_capability, 'diagnostic_partial')
            self.finish = self.module.get_function('diagnostic_final')
            self.views = tuple(cupy.from_dlpack(t) for t in
                (self.descriptors, self.factors, self.tasks, self.partial, self.output))

    def _stream(self):
        return self.cp.cuda.ExternalStream(torch.cuda.current_stream(self.device).cuda_stream,
                                           device_id=self.device)

    def measure(self):
        desc, factors, tasks, partial, output = self.views
        with self.cp.cuda.Device(self.device), self._stream():
            self.kernel((self.blocks,), (256,), (desc, factors, tasks, partial))
            self.finish((1,), (256,), (partial, output, np.int32(self.blocks)))
        energy, peak, bad = self.output.cpu().tolist()
        if bad:
            raise FloatingPointError('Non-finite full-domain or auxiliary state detected.')
        return energy, peak
