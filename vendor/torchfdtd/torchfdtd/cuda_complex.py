"""Fused complex Yee/CPML updates for real dielectric adjoint systems."""
import math
import torch
from .boundaries import CURL_TERMS
from .cuda_kernels import FusedYeeCUDA


class FusedComplexYeeCUDA(FusedYeeCUDA):
    """Two real lanes per cell, complex Bloch seams, real material coefficients."""
    complex_fields=True

    def __init__(self,grid,*,direct_views=False,bindings_cache=None,
                 cells_per_thread=1,block_size=None,report=None):
        if grid.E.dtype not in (torch.complex64,torch.complex128):
            raise ValueError('Complex fused updates require complex64/complex128 fields.')
        if grid.material_states or getattr(grid,'subpixel',None) is not None:
            raise ValueError('Complex fused updates currently require real diagonal dielectric coefficients.')
        super().__init__(grid,direct_views=direct_views,bindings_cache=bindings_cache,
                         cells_per_thread=cells_per_thread,block_size=block_size,report=report)

    @staticmethod
    def launch_count(grid,forward):
        # The unchanged complex forward body owns one real lane per entry.
        return 2*math.prod(grid.E.shape[:3])

    def _source(self,forward):
        g=self.grid
        shape=tuple(g.E.shape[:3]);strides=(shape[1]*shape[2],shape[2],1)
        real='double' if g.E.dtype==torch.complex128 else 'float'
        dtype=torch.float64 if real=='double' else torch.float32
        tensors=[];parameters=[]
        def argument(name,tensor,complex_value=False,write=False):
            expected=g.E.dtype if complex_value else dtype
            if tensor.dtype!=expected or tensor.device!=g.E.device or not tensor.is_contiguous():
                raise ValueError('Complex fused fields and real coefficients must have matching precision/device and contiguous storage.')
            tensors.append(tensor)
            parameters.append(f'{"" if write else "const "}{real}* __restrict__ {name}')
            return name
        argument('src',g.E if forward else g.H,True)
        argument('dst',g.H if forward else g.E,True,True)
        inverse=g.inverse_permeability if forward else g.inverse_permittivity
        argument('inverse',inverse)
        def field(index,comp,lane='lane'):
            return f'src[2*(3*({index})+{comp})+{lane}]'
        def phase_field(index,comp,phase):
            return f'(({real})({phase.real:.17g})*{field(index,comp)} + (lane ? ({real})({phase.imag:.17g}) : -({real})({phase.imag:.17g}))*{field(index,comp,"(1-lane)")})'
        lines=[f'const int tid=blockIdx.x*blockDim.x+threadIdx.x;',
               f'if(tid>={2*math.prod(shape)}) return;',
               'const int i=tid/2, lane=tid%2;',
               f'const int x=i/{strides[0]}, y=(i/{strides[1]})%{shape[1]}, z=i%{shape[2]};',
               f'{real} c0=0,c1=0,c2=0;']
        for term,(axis,comp,out,sign) in enumerate(CURL_TERMS):
            n,stride=shape[axis],strides[axis]
            if n==1:continue
            coord='xyz'[axis]
            guard=f'{coord}<{n-1}' if forward else f'{coord}>0'
            high=f'i+{stride}' if forward else 'i'
            low='i' if forward else f'i-{stride}'
            lines+=['{',f'{real} d=0;',f'if({guard}) {{',f'd={field(high,comp)}-{field(low,comp)};']
            metric=g.metric.get((forward,axis))
            if metric:
                name=argument(f'metric{term}',metric[0])
                lines.append(f'd*={name}[{coord}{"" if forward else "-1"}];')
            active_start=0 if forward else 1
            for si,seg in enumerate(g.cpml.get((forward,axis,comp),[])):
                lo=seg['slice'][axis].start+active_start;hi=seg['slice'][axis].stop+active_start
                psi=argument(f'psi{term}_{si}',seg['psi'],True,True)
                b,c,k=[argument(f'{key}{term}_{si}',seg[key]) for key in ('b','c','inv_k')]
                coords=['x','y','z'];coords[axis]=f'({coord}-{lo})'
                slab=list(shape);slab[axis]=hi-lo
                index=f'({coords[0]}*{slab[1]}+{coords[1]})*{slab[2]}+{coords[2]}'
                lines += [f'if({coord}>={lo} && {coord}<{hi}) {{',
                          f'const int p=2*({index})+lane,q={coord}-{lo};',
                          f'{real} memory={psi}[p]*{b}[q]+{c}[q]*d;',
                          *([f'{psi}[p]=memory;'] if getattr(self,'write_cpml',True) else []),f'd=d*{k}[q]+memory;','}']
            lines.append('}')
            if axis in g.wrap:
                phase=complex(g.wrap[axis])
                first=f'i-{(n-1)*stride}' if forward else 'i'
                last='i' if forward else f'i+{(n-1)*stride}'
                expression=f'{phase_field(first,comp,phase)}-{field(last,comp)}' if forward else f'{field(first,comp)}-{phase_field(last,comp,1/phase)}'
                lines+=['else {',f'd={expression};']
                if metric:lines.append(f'd*=({real})({metric[1]:.17g});')
                lines.append('}')
            elif forward and axis in getattr(g,'pec_upper',{}):
                factor=g.pec_upper[axis]
                lines+=['else {',f'd=-({real})({factor:.17g})*{field("i",comp)};','}']
            lines += [f'c{out}+={"-" if sign<0 else ""}d;','}']
        lines.extend(self._update_statements(forward,inverse,argument,real))
        return 'extern "C" __global__ void yee_update('+', '.join(parameters)+') {\n'+'\n'.join(lines)+'\n}',tensors

    def _update_statements(self,forward,inverse,argument,real):
        g=self.grid
        lines=[]
        for comp in range(3):
            index='0' if inverse.numel()==1 else str(comp) if inverse.numel()==3 else f'3*i+{comp}'
            lines.append(f'dst[2*(3*i+{comp})+lane] {"-=" if forward else "+="} (({real})({g.courant_number:.17g})*inverse[{index}])*c{comp};')
        return lines

    def update(self,forward):
        super().update(forward)
