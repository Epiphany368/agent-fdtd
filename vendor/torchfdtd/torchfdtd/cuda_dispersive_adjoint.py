"""Native coupled ADE updates and first-order material VJPs with compact inputs.

Shared material derivatives accumulate in one slot per CUDA block. A final
Torch reduction restores the original scalar/pole shapes without atomic adds
or a pole-by-cell material-gradient temporary. E/H and P/Q remain full states.
"""
import math

import torch

from .cuda_kernels import FusedYeeCUDA, _compile, _direct_cuda_view
from .cuda_complex import FusedComplexYeeCUDA
from .cuda_adjoint import FusedAdjointCUDA
from .cuda_complex_adjoint import FusedComplexAdjointCUDA, complex_definition


class _ParameterCode:
    def __init__(self, system):
        self.shapes = system.layout.shapes
        self.n = math.prod(system.region.shape)
        self.poles = system.pole_count
        self.offsets = []
        offset = 0
        self.shared = {}
        shared_count = 0
        for index, shape in enumerate(self.shapes):
            self.offsets.append(offset)
            count = math.prod(shape)
            offset += count
            if index and len(shape) <= 1:
                self.shared[index] = (shared_count, count)
                shared_count += count
        self.shared_count = shared_count

    def index(self, parameter, *, component='c'):
        shape = self.shapes[parameter]
        if parameter == 0:
            value = 'i' if len(shape) == 3 else f'3LL*i+({component})'
        elif len(shape) == 0:value = '0'
        elif len(shape) == 1:value = 'p'
        elif len(shape) == 4:value = f'(long long)p*{self.n}+i'
        else:value = f'3LL*((long long)p*{self.n}+i)+({component})'
        return f'({self.offsets[parameter]}LL+({value}))'

    def parameter(self, index, *, component='c'):
        return f'parameters[{self.index(index, component=component)}]'

    def coefficients(self, *, component='c'):
        s, a, g = [self.parameter(index, component=component) for index in (1,2,3)]
        return f'R a=R(.5)*{a}, d=R(1)+R(.5)*{g}+R(.25)*{a}, k={s}/(R(4)*d);'


class _ADEUpdate:
    def __init__(self, system, *, recompute=False, buffers=None):
        self.code = _ParameterCode(system)
        self.parameters = system.parameters
        self.P, self.Q = system.P, system.Q
        self.recompute = recompute
        self.write_cpml = not recompute
        self.output = (buffers.array('ade_recompute_E', system.grid.E.shape, system.field_dtype)
                       if buffers is not None else torch.empty_like(system.grid.E)) if recompute else system.grid.E
        # No strong system/grid reference, so a completed graph is releasable
        # without cyclic garbage collection.
        super().__init__(system.grid, direct_views=True, bindings_cache=buffers)

    def _update_statements(self, forward, inverse, argument, real, **kwargs):
        if forward:return super()._update_statements(forward,inverse,argument,real,**kwargs)
        complex_fields = self.grid.E.is_complex()
        argument('parameters', self.parameters)
        if complex_fields:
            argument('ade_p',self.P,True,not self.recompute)
            argument('ade_q',self.Q,True,not self.recompute)
            if self.recompute:argument('ade_new',self.output,True,True)
        else:
            argument('ade_p',self.P,not self.recompute)
            argument('ade_q',self.Q,not self.recompute)
            if self.recompute:argument('ade_new',self.output,True)
        def lane(index):return f'2*({index})+lane' if complex_fields else index
        lines=[f'using R={real};']
        for component in range(3):
            field = lane(f'3*i+{component}')
            material = lane('j')
            lines += ['{',f'const R old=dst[{field}], eps={self.code.parameter(0,component=str(component))};',
                      'R ksum=0,response=0;',
                      f'for(int p=0;p<{self.code.poles};++p){{',
                      f'const long long j=3LL*((long long)p*{self.code.n}+i)+{component};',
                      self.code.coefficients(component=str(component)),
                      f'ksum+=k; response+=(ade_q[{material}]-a*ade_p[{material}])/d;','}',
                      f'const R next=((eps-ksum)*old+R({self.grid.courant_number:.17g})*c{component}-response)/(eps+ksum);']
            if not self.recompute:
                lines += [f'for(int p=0;p<{self.code.poles};++p){{',
                          f'const long long j=3LL*((long long)p*{self.code.n}+i)+{component};',
                          self.code.coefficients(component=str(component)),
                          f'R q=ade_q[{material}], delta=(q-a*ade_p[{material}])/d+k*(next+old);',
                          f'ade_p[{material}]+=delta; ade_q[{material}]=-q+R(2)*delta;','}']
            destination = 'ade_new' if self.recompute else 'dst'
            lines += [f'{destination}[{field}]=next;','}']
        return lines


class _RealADE(_ADEUpdate, FusedYeeCUDA):pass
class _ComplexADE(_ADEUpdate, FusedComplexYeeCUDA):pass


def fused_ade_forward(system, *, recompute=False, buffers=None):
    implementation = _ComplexADE if system.grid.E.is_complex() else _RealADE
    return implementation(system, recompute=recompute, buffers=buffers)


class FusedDispersiveAdjointCUDA:
    """Compose H-curl transpose, material transpose and E-curl transpose."""
    def __init__(self, system, gradient, signal_bar, *, buffers=None):
        import cupy
        self.cp = cupy
        self.system = system
        self.gradient = gradient
        self.signal_bar = signal_bar.contiguous()
        self.code = _ParameterCode(system)
        self.blocks = (self.code.n+255)//256
        self.numerator_bar = buffers.array('ade_numerator_bar',system.grid.E.shape,system.field_dtype) if buffers is not None else torch.empty_like(system.grid.E)
        self.p_bar = buffers.zeros('ade_p_bar',system.P) if buffers is not None else torch.zeros_like(system.P)
        self.q_bar = buffers.zeros('ade_q_bar',system.Q) if buffers is not None else torch.zeros_like(system.Q)
        partial_shape = (self.code.shared_count,self.blocks)
        self.partials = buffers.array('ade_partials',partial_shape,system.dtype).zero_() if buffers is not None else torch.zeros(partial_shape,dtype=system.dtype,device=system.device)
        self.recompute = fused_ade_forward(system, recompute=True, buffers=buffers)
        kernel = FusedComplexAdjointCUDA if system.grid.E.is_complex() else FusedAdjointCUDA
        self.curl = kernel(system,gradient,self.signal_bar,direct_views=True,buffers=buffers,
                           material_gradient=False,electric_seed=self.numerator_bar,
                           curl_permittivity=torch.ones(1,dtype=system.dtype,device=system.device))
        source = self.source()
        self.device = system.device.index
        with cupy.cuda.Device(self.device), self.curl.stream():
            self.kernel, self.module = _compile(source,self.device,cupy.cuda.Device(self.device).compute_capability,'ade_transpose')
            tensors = (system.parameters,system.grid.E,self.recompute.output,system.P,system.Q,
                       self.curl.e_bar,self.p_bar,self.q_bar,self.numerator_bar,gradient,self.partials)
            self.arrays = tuple(_direct_cuda_view(cupy,t) for t in tensors)

    def source(self):
        code = self.code
        real = 'double' if self.system.dtype == torch.float64 else 'float'
        complex_fields = self.system.grid.E.is_complex()
        header = complex_definition(real) if complex_fields else ''
        header += f'using R={real}; using F={"C" if complex_fields else real};\n'
        header += ('__device__ R dot(F a,F b){return a.r*b.r+a.j*b.j;}\n' if complex_fields else
                   '__device__ R dot(F a,F b){return a*b;}\n')
        # Fixed reduction order, one partial per parameter/block. All 256
        # threads participate, including zero-valued threads in the last block.
        header += '''
__device__ R block_sum(R value,R* shared){
    for(int offset=16;offset>0;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(lane==0)shared[warp]=value;
    __syncthreads();
    value=threadIdx.x<8?shared[lane]:R(0);
    if(warp==0)for(int offset=16;offset>0;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
    __syncthreads();
    return value;
}
extern "C" __global__ void ade_transpose(
    const R* parameters,const F* old_e,const F* new_e,const F* old_p,const F* old_q,
    F* e_bar,F* p_bar,F* q_bar,F* numerator_bar,R* gradient,R* partials){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ R shared[8];
    F old[3],next[3],nb[3]; R denominator_bar[3];
    R epsilon_gradient=0;
'''
        lines = [f'const bool active=i<{code.n};', 'if(active){', 'for(int c=0;c<3;++c){',
                 'old[c]=old_e[3*i+c]; next[c]=new_e[3*i+c];',
                 'F common=0; R ksum=0;', f'for(int p=0;p<{code.poles};++p){{',
                 f'const long long j=3LL*((long long)p*{code.n}+i)+c;',
                 code.coefficients(), 'ksum+=k; common+=k*(p_bar[j]+R(2)*q_bar[j]);','}',
                 f'const R eps={code.parameter(0)};',
                 'F new_bar=e_bar[3*i+c]+common;',
                 'nb[c]=new_bar/(eps+ksum);',
                 'denominator_bar[c]=-dot(new_bar,next[c])/(eps+ksum);',
                 'e_bar[3*i+c]=common+(eps-ksum)*nb[c];',
                 'numerator_bar[3*i+c]=nb[c];',
                 'R value=dot(nb[c],old[c])+denominator_bar[c];']
        if len(code.shapes[0]) == 4:lines += [f'gradient[{code.index(0)}]+=value;']
        else:lines += ['epsilon_gradient+=value;']
        lines += ['}']
        if len(code.shapes[0]) == 3:lines += [f'gradient[{code.index(0)}]+=epsilon_gradient;']
        lines += ['}', f'for(int p=0;p<{code.poles};++p){{', 'R gs=0,ga=0,gg=0;',
                  'if(active){for(int c=0;c<3;++c){',
                  f'const long long j=3LL*((long long)p*{code.n}+i)+c;',
                  code.coefficients(),
                  'F delta_bar=p_bar[j]+R(2)*q_bar[j];',
                  'F response=(old_q[j]-a*old_p[j])/d;',
                  'F response_bar=delta_bar-nb[c];',
                  'R k_bar=dot(delta_bar,next[c]+old[c])-dot(nb[c],old[c])+denominator_bar[c];',
                  'R a_bar=-dot(response_bar,old_p[j])/d;',
                  'R d_bar=-dot(response_bar,response)/d-k_bar*k/d;',
                  'R vs=k_bar/(R(4)*d),va=R(.5)*a_bar+R(.25)*d_bar,vg=R(.5)*d_bar;',
                  'p_bar[j]-=a*response_bar/d; q_bar[j]=-q_bar[j]+response_bar/d;']
        for index, value, accumulator in ((1,'vs','gs'),(2,'va','ga'),(3,'vg','gg')):
            if len(code.shapes[index]) == 5:lines += [f'gradient[{code.index(index)}]+={value};']
            else:lines += [f'{accumulator}+={value};']
        lines += ['}}']
        for index, value in ((1,'gs'),(2,'ga'),(3,'gg')):
            if index in code.shared:
                start, _ = code.shared[index]
                slot = f'{start}+p' if len(code.shapes[index]) == 1 else str(start)
                lines += ['{',f'R sum=block_sum({value},shared);',
                          f'if(threadIdx.x==0)partials[({slot})*{self.blocks}LL+blockIdx.x]+=sum;','}']
            elif len(code.shapes[index]) == 4:
                lines += [f'if(active)gradient[{code.index(index)}]+={value};']
        return header+'\n'.join(lines)+'\n}}'

    def step(self,index,*,observation_index=None):
        system = self.system
        seed = self.signal_bar[index if observation_index is None else observation_index]
        for target,(positions,indices) in zip((self.curl.e_bar,self.curl.h_bar),system.observation_maps):
            if indices.numel():target.reshape(-1).index_add_(0,indices,seed.index_select(0,positions))
        # Re-evaluate only pre-source E. Primal P/Q and CPML restart states stay
        # immutable for the following replay or checkpoint save.
        self.recompute.update(False)
        with self.cp.cuda.Device(self.device), self.curl.stream():
            fn,arrays,_ = self.curl.launches[True,self.curl.phase]
            fn((self.blocks,),(256,),arrays)
            self.kernel((self.blocks,),(256,),self.arrays)
            fn,arrays,_ = self.curl.launches[False,self.curl.phase]
            fn((self.blocks,),(256,),arrays)
        self.curl.phase = 1-self.curl.phase

    def finalize(self,report):
        for index,(start,count) in self.code.shared.items():
            offset = self.code.offsets[index]
            self.gradient[offset:offset+count].add_(self.partials[start:start+count].sum(1))
        report['material_gradient_reduction_bytes'] = self.partials.numel()*self.partials.element_size()
        report['material_gradient_reduction'] = 'fixed CUDA block partials followed by compact Torch sums, no atomics'
