"""Optional fused field updates for real FP32, periodic x/y and z-only CPML.

The expressions originated in the metalens prototype and retain the native
split Yee update's arithmetic order. E is advanced first, soft electric sources
are added, and H then uses those new E values. Fusion removes an intermediate
global-memory round trip without changing the time staggering or CPML recurrence.

Two forward layouts implement the same equations. FusedEHMarching reuses two
x planes in shared memory while advancing through an x chunk. FusedEHGather
recomputes the neighboring E values needed by each cell's H update and handles
z extents that exceed the marching layout's thread or shared-memory limits.
Each simulation owns its buffers and launch bindings. No device-model-specific
patches or process-wide kernel replacements are installed.

OnePassFieldAdjoint applies the H transpose followed by the E transpose in one
gather kernel. Separate old/new cotangent buffers prevent cross-thread hazards.
Both directions preserve curl-term order and use the native IEEE compiler flags
without fused multiply-add contraction or flush-to-zero. Unsupported physics
is checked before allocation, and the caller records use of the split fallback.
"""
from __future__ import annotations

import hashlib
import math
import weakref

import numpy as np
import torch

from . import cuda_kernels as _ck
from .boundaries import CURL_TERMS as _CURL
from .reversible_cuda_tuning import select_launch


def _compiled(source, device, name):
    import cupy
    with cupy.cuda.Device(device):
        return _ck._compile(source, device, cupy.cuda.Device(device).compute_capability, name)


def _bounded(source, name, block):
    return source.replace(f'void {name}(', f'void __launch_bounds__({block}) {name}(')


def eligibility_reason(system):
    g = system.grid
    shape = tuple(system.region.shape)
    if not g.E.is_cuda:
        return 'CUDA is required.'
    if system.field_dtype != torch.float32 or g.E.dtype != torch.float32:
        return 'Fused updates require real FP32 fields.'
    if system.pmc or getattr(g, 'pec_upper', {}):
        return 'Fused updates do not support mirror boundaries.'
    if g.material_states or getattr(g, 'subpixel', None) is not None:
        return 'Fused updates require nondispersive staircase materials.'
    if g.metric or dict(g.wrap) != {0: 1.0, 1: 1.0}:
        return 'Fused updates require uniform periodic x/y boundaries.'
    if min(shape) < 2 or 3 * math.prod(shape) >= 2**31:
        return 'Grid dimensions exceed the fused launch contract.'
    if g.inverse_permittivity.numel() != 3 * math.prod(shape) or g.inverse_permeability.numel() != 1:
        return 'Fused updates require diagonal electric and constant magnetic coefficients.'
    if any(segments for (forward, axis, comp), segments in g.cpml.items() if axis != 2):
        return 'Fused updates support CPML on z only.'
    return None


def forward_eligibility_reason(system):
    reason = eligibility_reason(system)
    if reason:
        return reason
    if system.sources['H'] or any(system.face_sources.values()):
        return 'Fused forward supports volume electric sources only.'
    for loc, comp, wave, profile in system.sources['E']:
        if profile is not None:
            return 'Fused forward does not support spatial source profiles.'
        if len(loc) != 3 or any(isinstance(part, slice) and part.step not in (None, 1) for part in loc):
            return 'Fused forward requires contiguous source support.'
    return None


def make_forward(system, requested, report):
    reason = forward_eligibility_reason(system)
    if reason:
        report['fallback_reason']['forward'] = reason
        return None
    nz = system.region.shape[2]
    zblock = 32 * ((nz + 31) // 32)
    try:
        if zblock * 6 <= 1024 and 4 * 6 * zblock * 3 * 4 <= 48 * 1024:
            engine = FusedEHMarching(system)
            report['forward_kernel_variant'] = 'marching'
            report.setdefault('cuda_launches', {})['forward_fused_eh_marching'] = dict(
                source_sha256=hashlib.sha256(engine.source.encode()).hexdigest(),
                block_size=list(engine.block), requested='fixed_marching_layout')
        else:
            engine = FusedEHGather(system, requested, report)
            report['forward_kernel_variant'] = 'gather'
    except RuntimeError as exc:
        if 'compilation failed' not in str(exc):
            raise
        report['fallback_reason']['forward'] = str(exc)
        return None
    report['forward_kernel_used'] = 'fused_eh'
    return engine


class _EHBase:
    """Shared bindings for both fused E/H layouts.

    E, H and E-CPML memories use old/new buffers because neighboring threads
    may still need their previous values. A halo calculation can read an old
    CPML memory but only its owning cell writes the new value. H-CPML memories
    can be updated in place because only their owner reads and writes them.
    The caller swaps parity after a complete launch, then rebinds observers
    and adjoint inputs to the newly current fields.
    """

    def _bind(self, system):
        g = system.grid
        nx, ny, nz = system.region.shape; self.shape = (nx, ny, nz); self.count = nx * ny * nz
        self.cn = f'((float)({g.courant_number:.17g}))'
        self.E = [g.E, torch.empty_like(g.E)]; self.H = [g.H, torch.empty_like(g.H)]
        self.esegs, self.hsegs = {}, {}                              # differentiated comp -> [(lo, hi, name)]
        self.epsi = []                                               # (segment dict, [psi parity 0, psi parity 1])
        params = ['const float* __restrict__ Eo', 'const float* __restrict__ Ho', 'float* __restrict__ En', 'float* __restrict__ Hn',
                  'const float* __restrict__ inv', 'const float* __restrict__ imu']
        fixed = {'inv': g.inverse_permittivity, 'imu': g.inverse_permeability}
        per_parity = []                                              # (name, pair, 0 = old / 1 = new)
        for forward, table in ((False, self.esegs), (True, self.hsegs)):
            for comp in (0, 1, 2):
                for k, seg in enumerate(g.cpml.get((forward, 2, comp), [])):
                    lo = seg['slice'][2].start + (0 if forward else 1); hi = seg['slice'][2].stop + (0 if forward else 1)
                    name = f'{"h" if forward else "e"}{comp}_{k}'
                    for key in ('b', 'c', 'inv_k'):
                        params.append(f'const float* __restrict__ {key}_{name}'); fixed[f'{key}_{name}'] = seg[key]
                    if forward:
                        params.append(f'float* __restrict__ psi_{name}'); fixed[f'psi_{name}'] = seg['psi']
                    else:
                        pair = [seg['psi'], torch.empty_like(seg['psi'])]; self.epsi.append((seg, pair))
                        params += [f'const float* __restrict__ po_{name}', f'float* __restrict__ pn_{name}']
                        per_parity += [(f'po_{name}', pair, 0), (f'pn_{name}', pair, 1)]
                    table.setdefault(comp, []).append((lo, hi, name))
        self.srcs = []
        for s_i, (loc, comp, wave, profile) in enumerate(system.sources["E"]):
            box = [sl.indices(size)[:2] if isinstance(sl, slice) else (sl, sl + 1) for sl, size in zip(loc, (nx, ny, nz))]
            params.append(f'const float* __restrict__ w{s_i}'); fixed[f'w{s_i}'] = wave; self.srcs.append((comp, box, f'w{s_i}'))
        return params, fixed, per_parity

    def _args(self, params, fixed, per_parity):
        names = [p.split()[-1] for p in params]
        self.args = []
        for parity in (0, 1):
            t = dict(fixed, Eo=self.E[parity], Ho=self.H[parity], En=self.E[1 - parity], Hn=self.H[1 - parity])
            for name, pair, which in per_parity: t[name] = pair[parity] if which == 0 else pair[1 - parity]
            tensors = [t[nm] for nm in names]
            self.args.append((tensors, tuple(_ck._direct_cuda_view(self.cp, x) for x in tensors)))

    def _e_code(self, var, out, j, X, Y, Z, own, h, declare=True):
        """The split E update of component `out` at cell j (coordinates X, Y, Z). h(axis, comp): the H operand at the cell (axis None)
        or at its -axis neighbour, wrapped across the periodic x/y seam (the split seam branch differences the same two values)."""
        ny = self.shape[1]
        L = ([f'float {var};'] if declare else []) + ['{', 'float cc=0;']
        for axis, comp, o, sign in _CURL:
            if o != out: continue
            coord = (X, Y, Z)[axis]
            L += ['{', 'float d=0;', f'if ({coord} > 0) {{', f'd = {h(None, comp)} - {h(axis, comp)};']
            for lo, hi, name in (self.esegs.get(comp, []) if axis == 2 else []):
                L += [f'if ({Z} >= {lo} && {Z} < {hi}) {{', f'const int p = (({X})*{ny}+({Y}))*{hi - lo}+(({Z})-{lo});', f'const int q = ({Z})-{lo};',
                      f'float memory = po_{name}[p]*b_{name}[q] + c_{name}[q]*d;', *([f'if ({own}) pn_{name}[p] = memory;'] if own else []),
                      f'd = d*inv_k_{name}[q] + memory;', '}']
            L.append('}')
            if axis != 2: L += ['else {', f'd = {h(None, comp)} - {h(axis, comp)};', '}']
            L += [f'cc += {"-" if sign < 0 else ""}d;', '}']
        L.append(f'{var} = Eo[3*({j})+{out}] + ({self.cn} * inv[3*({j})+{out}]) * cc;')
        for comp, box, w in self.srcs:
            if comp == out:
                (x0, x1), (y0, y1), (z0, z1) = box
                L.append(f'if (({X}) >= {x0} && ({X}) < {x1} && ({Y}) >= {y0} && ({Y}) < {y1} && ({Z}) >= {z0} && ({Z}) < {z1}) {var} = {var} + {w}[n];')
        L.append('}')
        return L

    def _h_code(self, e, hold, write, X, Y):
        """The split H update of the own cell (x, y, z); e(axis, comp): E_new at the cell (None) or its +axis neighbour."""
        nz, ny = self.shape[2], self.shape[1]
        L = ['float c0=0, c1=0, c2=0;']
        for axis, comp, out, sign in _CURL:
            L += ['{', 'float d=0;']
            if axis == 2:
                L += [f'if (z < {nz - 1}) {{', f'd = {e(2, comp)} - {e(None, comp)};']
                for lo, hi, name in self.hsegs.get(comp, []):
                    L += [f'if (z >= {lo} && z < {hi}) {{', f'const int p = (({X})*{ny}+({Y}))*{hi - lo}+(z-{lo});', f'const int q = z-{lo};',
                          f'float memory = psi_{name}[p]*b_{name}[q] + c_{name}[q]*d;', f'psi_{name}[p] = memory;', f'd = d*inv_k_{name}[q] + memory;', '}']
                L.append('}')
            else:
                L.append(f'd = {e(axis, comp)} - {e(None, comp)};')    # the periodic wrap branch differences the same two values
            L += [f'c{out} += {"-" if sign < 0 else ""}d;', '}']
        L += [f'{write(c)} = {hold(c)} - ({self.cn} * imu[0]) * c{c};' for c in range(3)]
        return L

    def _stream(self):
        return self.cp.cuda.ExternalStream(torch.cuda.current_stream(self.system.grid.E.device).cuda_stream, device_id=self.dev)

    def step(self, n):
        with self.cp.cuda.Device(self.dev), self._stream():
            self.fn(self.grid, self.block, self.args[self.parity][1] + (np.int32(n),))

    def swap(self):
        self.parity ^= 1; g = self.system.grid
        g.E, g.H = self.E[self.parity], self.H[self.parity]
        for seg, pair in self.epsi: seg['psi'] = pair[self.parity]


class FusedEHGather(_EHBase):
    """Gather layout with one thread owning each cell's E and H outputs.

    The thread recomputes E at its own cell and the +x/+y/+z neighbors needed
    by curl E. All reads use the old buffers, so no block-wide or grid-wide
    synchronization is required. Redundant neighbor arithmetic trades memory
    reuse for support of z extents outside the shared-memory marching layout.
    """

    def __init__(self, system, requested=None, report=None):
        import cupy
        self.cp = cupy; self.system = weakref.proxy(system); g = system.grid
        params, fixed, per_parity = self._bind(system)
        nx, ny, nz = self.shape; S = (ny * nz, nz, 1)
        wrap = {0: (nx - 1) * S[0], 1: (ny - 1) * S[1]}
        def hg(j, X, Y):
            def h(axis, comp):
                if axis is None: return f'Ho[3*({j})+{comp}]'
                if axis == 2: return f'Ho[3*({j}-1)+{comp}]'
                coord = (X, Y)[axis]
                return f'Ho[3*(({coord}) > 0 ? ({j})-{S[axis]} : ({j})+{wrap[axis]})+{comp}]'
            return h
        body = [f'const int i = blockIdx.x * blockDim.x + threadIdx.x;', f'if (i >= {self.count}) return;',
                f'const int x = i / {S[0]};', f'const int y = (i / {S[1]}) % {ny};', f'const int z = i % {nz};',
                f'const int xq = x == {nx - 1} ? 0 : x + 1;', f'const int xp = x == {nx - 1} ? i - {(nx - 1) * S[0]} : i + {S[0]};',
                f'const int yq = y == {ny - 1} ? 0 : y + 1;', f'const int yp = y == {ny - 1} ? i - {(ny - 1) * S[1]} : i + {S[1]};']
        for c in range(3): body += self._e_code(f'e{c}', c, 'i', 'x', 'y', 'z', 'true', hg('i', 'x', 'y'))
        body += self._e_code('ez_x', 2, 'xp', 'xq', 'y', 'z', None, hg('xp', 'xq', 'y')) + self._e_code('ey_x', 1, 'xp', 'xq', 'y', 'z', None, hg('xp', 'xq', 'y'))
        body += self._e_code('ez_y', 2, 'yp', 'x', 'yq', 'z', None, hg('yp', 'x', 'yq')) + self._e_code('ex_y', 0, 'yp', 'x', 'yq', 'z', None, hg('yp', 'x', 'yq'))
        body += ['float ey_z = 0, ex_z = 0;', f'if (z < {nz - 1}) {{', 'const int zq = z + 1;']
        body += self._e_code('ey_z', 1, 'i+1', 'x', 'y', 'zq', None, hg('i+1', 'x', 'y'), False) + self._e_code('ex_z', 0, 'i+1', 'x', 'y', 'zq', None, hg('i+1', 'x', 'y'), False) + ['}']
        nb = {0: {2: 'ez_x', 1: 'ey_x'}, 1: {2: 'ez_y', 0: 'ex_y'}, 2: {1: 'ey_z', 0: 'ex_z'}}
        body += self._h_code(lambda axis, comp: f'e{comp}' if axis is None else nb[axis][comp], lambda c: f'Ho[3*i+{c}]', lambda c: f'Hn[3*i+{c}]', 'x', 'y')
        body += [f'En[3*i+{c}] = e{c};' for c in range(3)]
        self.source = 'extern "C" __global__ void eh_update(' + ', '.join(params + ['const int n']) + ') {\n' + '\n'.join(body) + '\n}'
        self._args(params, fixed, per_parity)
        self.dev = g.E.device.index
        with cupy.cuda.Device(self.dev), self._stream():
            tensors, views = self.args[0]
            self.fn, self.module, bs = select_launch(self.source, 'eh_update',
                tensors + [None], views + (np.int32(0),), self.count, self.dev,
                requested, label='forward_fused_eh_gather', report=report)
        self.grid, self.block = ((self.count + bs - 1) // bs,), (bs,)
        self.parity = 0


class FusedEHMarching(_EHBase):
    """2.5D blocking: a block owns TY y-rows x all z (ZB >= nz threads per row, two extra thread rows for the y halo) and marches
    along an x-chunk. Per plane x it loads H_old(x+1) (rows y0-1 .. y0+TY) to shared memory, computes E_new(x+1) once for rows
    y0 .. y0+TY (the last row is the halo the H update's y difference needs; x+1 = x1 is the next chunk's first plane, computed
    here as a halo and written there), then H_new(x) of its own rows. Each E value is computed once per block; global traffic is
    the ~60 B/cell of one read and one write of E and H plus the eps table."""

    def __init__(self, system, TY=4, xchunks=None):
        import cupy
        self.cp = cupy; self.system = weakref.proxy(system); g = system.grid
        params, fixed, per_parity = self._bind(system)
        nx, ny, nz = self.shape
        ZB = 32 * ((nz + 31) // 32); R = TY + 2
        if ZB * R > 1024 or 4 * R * ZB * 3 * 4 > 48 * 1024: raise ValueError("z extent too large for the marching kernel")
        yblocks = (ny + TY - 1) // TY
        if xchunks is None:
            sms = torch.cuda.get_device_properties(g.E.device).multi_processor_count
            xchunks = max(1, min(nx, (8 * sms + yblocks - 1) // yblocks))
        CL = (nx + xchunks - 1) // xchunks; xchunks = (nx + CL - 1) // CL
        P = f'{R * ZB * 3}'
        def s(buf, row, zz, c): return f'{buf}[(({row})*{ZB}+({zz}))*3+{c}]'
        def hsm(row):                                                 # E update of plane x+1 at thread row `row`, column z
            def h(axis, comp):
                if axis is None: return s('Hn_', row, 'z', comp)
                if axis == 0: return s('Hc_', row, 'z', comp)          # H_old(x): the previous plane
                if axis == 1: return s('Hn_', f'{row}-1', 'z', comp)   # y-1 (wrapped) is the row above in the tile
                return s('Hn_', row, 'z-1', comp)
            return h
        body = [f'__shared__ float SH[2][{P}];', f'__shared__ float SE[2][{P}];',
                'const int z = threadIdx.x; const int ty = threadIdx.y;',
                f'const int y0 = blockIdx.x * {TY}; const int nrows = min({TY}, {ny} - y0);',
                f'const int x0 = blockIdx.y * {CL}; const int x1 = min(x0 + {CL}, {nx});',
                'if (x0 >= x1) return;',
                f'const int y = ((y0 - 1 + ty) % {ny} + {ny}) % {ny};',
                f'const bool zin = z < {nz};', 'const bool hrow = ty <= nrows + 1;', 'const bool erow = ty >= 1 && ty <= nrows + 1;',
                'const bool own = ty >= 1 && ty <= nrows;',
                'int cur = 0;',
                '{', f'const int xm = x0 == 0 ? {nx - 1} : x0 - 1;',
                f'if (hrow && zin) {{ const int j = (xm*{ny}+y)*{nz}+z; for (int c = 0; c < 3; ++c) SH[0][(ty*{ZB}+z)*3+c] = Ho[3*j+c]; }}', '}',
                'for (int x = x0 - 1; x < x1; ++x) {',
                'const int nxt = cur ^ 1;',
                f'const int xe = x + 1 == {nx} ? 0 : x + 1;',
                'float* Hn_ = SH[nxt]; float* Hc_ = SH[cur]; float* En_ = SE[nxt]; float* Ec_ = SE[cur];',
                f'if (hrow && zin) {{ const int j = (xe*{ny}+y)*{nz}+z; for (int c = 0; c < 3; ++c) Hn_[(ty*{ZB}+z)*3+c] = Ho[3*j+c]; }}',
                '__syncthreads();',
                'if (erow && zin) {',
                f'const int j = (xe*{ny}+y)*{nz}+z;',
                'const bool eown = own && x + 1 < x1;']
        for c in range(3): body += self._e_code(f'e{c}', c, 'j', 'xe', 'y', 'z', 'eown', hsm('ty'))
        body += [f'{s("En_", "ty", "z", c)} = e{c};' for c in range(3)]
        body += ['if (eown) {', *[f'En[3*j+{c}] = e{c};' for c in range(3)], '}', '}', '__syncthreads();',
                 'if (x >= x0 && own && zin) {', f'const int i = (x*{ny}+y)*{nz}+z;']
        def esm(axis, comp):
            if axis is None: return s('Ec_', 'ty', 'z', comp)
            if axis == 0: return s('En_', 'ty', 'z', comp)
            if axis == 1: return s('Ec_', 'ty+1', 'z', comp)
            return s('Ec_', 'ty', 'z+1', comp)
        body += self._h_code(esm, lambda c: s('Hc_', 'ty', 'z', c), lambda c: f'Hn[3*i+{c}]', 'x', 'y')
        body += ['}', '__syncthreads();', 'cur = nxt;', '}']
        self.source = 'extern "C" __global__ void eh_march(' + ', '.join(params + ['const int n']) + ') {\n' + '\n'.join(body) + '\n}'
        self._args(params, fixed, per_parity)
        self.dev = g.E.device.index
        self.fn, self.module = _compiled(_bounded(self.source, 'eh_march', ZB * R), self.dev, 'eh_march')
        self.grid, self.block = (yblocks, xchunks), (ZB, R)
        self.TY, self.xchunks = TY, xchunks
        self.parity = 0



class OnePassFieldAdjoint:
    """Gather the two field transposes without an intermediate global write.

    Reconstruct post-H-transpose E cotangents locally, including the neighbors
    needed by the E transpose, then write the new E/H cotangents once. CPML
    cotangents have distinct input/output buffers and only their owner writes
    each output. Material-gradient accumulation and field reconstruction stay
    in their existing callers. This optional path does not assume that fewer
    launches are faster for every grid or GPU.
    """
    def __init__(self, adj, requested=None, report=None):
        cn = f'((float)({adj.system.grid.courant_number:.17g}))'
        cupy = adj.cp; s = adj.system; self.adj = weakref.proxy(adj)
        nx, ny, nz = s.region.shape; S = (ny * nz, nz, 1); self.count = nx * ny * nz
        diagonal = adj.epsilon.shape[-1] == 3
        def sc(t, out):
            ep = f'eps[3*({t})+{out}]' if diagonal else f'eps[{t}]'
            return f'{cn}/{ep}'
        self.ebar = [adj.e_bar, torch.zeros_like(adj.e_bar)]; self.hbar = [adj.h_bar, torch.zeros_like(adj.h_bar)]
        params = ['const float* __restrict__ eo', 'const float* __restrict__ ho', 'float* __restrict__ en', 'float* __restrict__ hn',
                  'const float* __restrict__ eps']
        segs = {}                                                    # (forward, comp) -> [(k, lo, hi)]
        for (fw, axis, comp), idx in s.keys.items():
            for k in idx:
                seg = s.segments[k]; segs.setdefault((fw, comp), []).append((k, seg['slice'][2].start, seg['slice'][2].stop))
                params += [f'const float* __restrict__ b_{k}', f'const float* __restrict__ c_{k}', f'const float* __restrict__ inv_k_{k}',
                           f'const float* __restrict__ old_{k}', f'float* __restrict__ new_{k}']

        def terms(forward, term_list, j, X, Y, Z, write, acc, value, scale):
            """The split adjoint_update code of the given CURL_TERMS entries, in order; value(kind, target, out) / scale(...) give the
            bar operand and its factor (kind: 'lo' = the coord>0 edge, 'hi' = the coord<n-1 edge, 'w0'/'wn' = the two periodic seams)."""
            L = []
            for axis, comp, out, sign in term_list:
                n, stride, coord = (nx, ny, nz)[axis], S[axis], (X, Y, Z)[axis]
                r = acc[comp]
                def edge(kind, index, coordinate, wr):
                    target = index if forward else f'({index})+{stride}'
                    code = [f'float d=({"-" if sign < 0 else ""}(({scale(kind, target, out)})*{value(kind, target, out)}));']
                    for k, lo, hi in (segs.get((forward, comp), []) if axis == 2 else []):
                        code += [f'if(({coordinate})>={lo} && ({coordinate})<{hi}){{', f'const int p=(({X})*{ny}+({Y}))*{hi - lo}+(({coordinate})-{lo}),q=({coordinate})-{lo};',
                                 f'float b=old_{k}[p]+d;', f'd=d*inv_k_{k}[q]+c_{k}[q]*b;', *([f'new_{k}[p]=b_{k}[q]*b;'] if wr else []), '}']
                    return code
                L += [f'if({coord}>0){{', *edge('lo', f'{j}-{stride}', f'{coord}-1', False), f'{r}+=d;', '}',
                      f'if({coord}<{n - 1}){{', *edge('hi', j, coord, write), f'{r}-=d;', '}']
                if axis != 2:
                    for cv, action, kind in ((0, '+=', 'w0'), (n - 1, '-=', 'wn')):
                        target = f'{j}+{(n - 1 - cv) * stride}' if forward else f'{j}-{cv * stride}'
                        L.append(f'if({coord}=={cv}){r}{action}({"-" if sign < 0 else ""}(({scale(kind, target, out)})*{value(kind, target, out)}));')
            return L

        body = [f'const int i=blockIdx.x*blockDim.x+threadIdx.x;', f'if(i>={self.count})return;',
                f'const int x=i/{S[0]};', f'const int y=(i/{S[1]})%{ny};', f'const int z=i%{nz};',
                f'const int xq=x=={nx - 1}?0:x+1;', f'const int xp=x=={nx - 1}?i-{(nx - 1) * S[0]}:i+{S[0]};',
                f'const int yq=y=={ny - 1}?0:y+1;', f'const int yp=y=={ny - 1}?i-{(ny - 1) * S[1]}:i+{S[1]};']
        hval = lambda kind, target, out: f'ho[3*({target})+{out}]'
        hscale = lambda kind, target, out: f'-{cn}'
        def ebar_new(var, c, j, X, Y, Z, own, declare=True):          # post-H-transpose e_bar component c at cell j
            return ([f'float {var};'] if declare else []) + ['{', 'float rr=0;', *terms(True, [t for t in _CURL if t[1] == c], j, X, Y, Z, own, {c: 'rr'}, hval, hscale),
                                                              f'{var}=eo[3*({j})+{c}]+rr;', '}']
        # own e_bar: the split H-transpose accumulates r0, r1, r2 over all terms, each term touching only its own r
        body += ['float r0=0,r1=0,r2=0;', *terms(True, _CURL, 'i', 'x', 'y', 'z', True, {0: 'r0', 1: 'r1', 2: 'r2'}, hval, hscale)]
        body += [f'const float e{c}=eo[3*i+{c}]+r{c};' for c in range(3)]
        body += ebar_new('e1_x', 1, 'xp', 'xq', 'y', 'z', False) + ebar_new('e2_x', 2, 'xp', 'xq', 'y', 'z', False)
        body += ebar_new('e0_y', 0, 'yp', 'x', 'yq', 'z', False) + ebar_new('e2_y', 2, 'yp', 'x', 'yq', 'z', False)
        body += ['float e0_z=0,e1_z=0;', f'if(z<{nz - 1}){{', 'const int zq=z+1;',
                 *ebar_new('e0_z', 0, 'i+1', 'x', 'y', 'zq', False, False), *ebar_new('e1_z', 1, 'i+1', 'x', 'y', 'zq', False, False), '}']
        nb_index = {0: 'xp', 1: 'yp', 2: '(i+1)'}
        def eval_(axis):
            def value(kind, target, out):
                return f'e{out}' if kind in ('lo', 'w0') else f'e{out}_{"xyz"[axis]}'
            def scale(kind, target, out):
                return sc('i', out) if kind in ('lo', 'w0') else sc(nb_index[axis], out)
            return value, scale
        body.append('float s0=0,s1=0,s2=0;')
        for term in _CURL:                                           # E-transpose: every term in the split order, each into s{comp}
            value, scale = eval_(term[0])
            body += terms(False, [term], 'i', 'x', 'y', 'z', True, {0: 's0', 1: 's1', 2: 's2'}, value, scale)
        body += [f'en[3*i+{c}]=e{c};' for c in range(3)] + [f'hn[3*i+{c}]=ho[3*i+{c}]+s{c};' for c in range(3)]
        self.source = 'extern "C" __global__ void adj_one(' + ','.join(params) + '){\n' + '\n'.join(body) + '\n}'
        names = [p.split()[-1] for p in params]
        self.args = []
        for parity in (0, 1):
            t = dict(eo=self.ebar[parity], ho=self.hbar[parity], en=self.ebar[1 - parity], hn=self.hbar[1 - parity], eps=adj.epsilon)
            for k, seg in enumerate(s.segments):
                t.update({f'b_{k}': seg['b'], f'c_{k}': seg['c'], f'inv_k_{k}': seg['inv_k'],
                          f'old_{k}': adj.psi_bars[parity][k], f'new_{k}': adj.psi_bars[1 - parity][k]})
            tensors = [t[nm] for nm in names]
            self.args.append((tensors, tuple(_ck._direct_cuda_view(cupy, x) for x in tensors)))
        self.obs = None
        if adj.observer is not None:
            fn, arrays, module, count = adj.observer
            self.obs = (fn, count, [(_ck._direct_cuda_view(cupy, self.ebar[p]), _ck._direct_cuda_view(cupy, self.hbar[p])) + tuple(arrays[2:]) for p in (0, 1)])
        tensors, views = self.args[0]
        self.fn, self.module, self.bs = select_launch(self.source, 'adj_one', tensors, views, self.count, adj.device, requested, label='adjoint_one_pass', report=report)
        self.parity = 0

    def step(self, row):
        adj = self.adj; p = self.parity
        with adj.cp.cuda.Device(adj.device), adj.stream():
            if self.obs is not None:
                fn, count, arrays = self.obs
                fn(((count + 127) // 128,), (128,), (*arrays[p], np.int32(row)))
            self.fn(((self.count + self.bs - 1) // self.bs,), (self.bs,), self.args[p][1])
        self.parity ^= 1
        adj.e_bar, adj.h_bar = self.ebar[self.parity], self.hbar[self.parity]
        adj.phase = 1 - adj.phase

