"""Exact speedups of the reversible-CPML tile adjoint for the B200 build (FF_B200_KERNELS=1, hook in ff_opt.py).

torchfdtd_src stays byte-identical: every change below replaces a method or module attribute of the vendored torchfdtd in this
process only, like the FF_FAST patches in ff_opt.py. No approximation anywhere: each item is either bit-identical to the vendored
solver or (item 1) the same discrete adjoint evaluated on a smaller reconstruction interval (float32 round-off only).

1. FF_B200_INTERVAL=pil (default; =full keeps the solver's interval). The reversible adjoint reconstructs the primal E/H backwards
   in time on an inclusive z interval [a, b] and records, every step, E_x,y at b+1 and H_x,y at a-1. The solver takes the whole
   lossless interior (z 13..111 of 125); the reverse update inside [a, b] needs only those two halo planes, so any interval that
   covers the layers where epsilon can depend on the design gives the same gradient. ff_opt's eps_of (and loop_levelset10's) varies
   only on the pillar layers `pil`, so the interval becomes pil (35 layers). Sources and monitors outside the interval are carried
   by the recorded halos (forward) and by the full-domain adjoint (backward): the solver's own source check still runs against the
   full interior, and _undo_sources skips sources that do not touch [a, b] (their fields there are never read). The terminal copy,
   drift check and memory reservation follow the interval. Every call checks epsilon == fixed_epsilon on the dropped layers and
   falls back to the full interval (once-per-model message) if not, so the forward is always the vendored forward.
   The returned d/d(epsilon) is zero on the dropped layers; eps_of does not depend on the design there, so d/d(design) is unchanged.
   Accuracy (3060, vs the float64 gradient of the same discrete model, 5.2 um period, 1200 steps): checkpointed float32 2.885e-6,
   reversible full interval 2.888e-6, pil 2.892e-6 rel L2 (pil vs full 2.3e-7); vs the checkpointed float32 gradient at 6.4 um,
   1200 steps: full 2.08e-7, pil 2.11e-7. The reversal round-off stays trapped between the
   two exact halo planes, which shows only on tiny short runs (1.2-3.6 um period, 300 steps: pil 2-4x the full interval, <= 2e-5).
2. FF_B200_FUSE_EG=1 (default). The backward step ran the reverse E update ('e': E -= cn/eps curl H) before the full-domain adjoint
   and the material VJP ('g': grad += -cn e_bar.curl H / eps^2) after it; both compute curl H of the same reconstructed H. The adjoint
   step (material_gradient=False) reads neither primal E nor H, so 'e' moves after it and one kernel computes curl H once and does
   both updates with the same expressions (bit-identical).
3. FF_B200_HYGIENE=1 (default): reverse interior kernels with __restrict__/const pointers and 32-bit index math without modulo
   (when 3*cells < 2^31); __launch_bounds__ on these, the fused Yee kernels and the adjoint kernels; FF_B200_AUTOTUNE=1 (default)
   times block sizes 128/256/512/1024 once per kernel source on scratch outputs and keeps the fastest (a thread computes only its
   own cell, so the block size cannot change any result). FF_B200_BLOCK=<n> forces one block size. The adjoint E-transpose reads
   fl(cn/epsilon) from a table built once per backward by the same NVRTC division instead of dividing 12 times per cell (same bytes;
   B200's FP32 rate grew much less than its bandwidth). NVRTC runs with --fmad=false, so unchanged expression trees give
   bit-identical floats.
Phase 2 (sections at the end, bit-identical, default off): FF_B200_FUSED_EH=1 one-pass E+H forward for the no-grad and the recorded
forward (2.5D marching kernel; =naive: one thread per cell with recomputed neighbours), FF_B200_FUSED_ADJ=1 one-pass adjoint step.
FF_B200_VERBOSE=1 prints every autotune table. Complex (Bloch) fields, PMC faces, subpixel and dispersive grids keep the vendored code.
Phase 3 (2026-09-30, bit-identical, default off):
4. FF_FAST_OBS=1: the monitor tables of a solve are built once per observer tuple of a model and reused. _System.prepare_observations
   (every forward) and FusedAdjointCUDA.observer_kernel (every backward) loop in Python over each of the ~6e5 plane monitors:
   A100, E2 1040^2 tile: 1.42 s per forward and 2.70 s per backward, independent of the step count (the 2.8-3.3 s per unit
   that B200/H200 spent outside the adjoint kernels). The cached index tensors and observer layout are the arrays the first call
   built (read-only in every consumer), keyed by the identity of the model's observer tuple (held alive), shape and device.
5. FF_B200_ADJ_FAST=1: adj_Ht / adj_Et get a branch-free path for cells with 1 <= x, y <= n-2 and z (and z-1) outside every z-CPML
   segment: the vendored per-term expressions in the vendored order (lo edge +=, hi edge -=), without the edge, CPML and seam
   tests that are all false there. Other cells take the unchanged vendored body. Same floats (--fmad=false, same expression
   trees); only the instruction count per cell drops (B200 ran these two kernels at 34-41 % of HBM bandwidth).
"""
import hashlib, math, os
import numpy as np
import torch
import torchfdtd.reversible_cpml as _rc
import torchfdtd.reversible_cpml_kernels as _rk
import torchfdtd.cuda_kernels as _ck
import torchfdtd.cuda_adjoint as _ca
from torchfdtd.cuda_bootstrap import prepare_cuda_kernels

CANDIDATES = (128, 256, 512, 1024)
CFG = {}
_ORIG = {}
_PIL = None                     # (first, last, nz) of the pillar layers
_TUNED = {}                     # sha1(kernel source) -> dict(label, bs, ms, count)
_KERNELS = {}                   # (source, device) -> (function, module); bypasses the 64-entry lru_cache of _compile
_WARNED = set()


def _env(name, default):
    return os.environ.get(name, default)


def _say(*a):
    if os.environ.get("FF_MULTI_RANK", "0") == "0" or os.environ.get("FF_MULTI_VERBOSE", "0") == "1":
        print("[b200_kernels]", *a, flush=True)


def configure(**kw):
    """Set options (interval 'pil'|'full', fuse_eg, hygiene, autotune, block, verbose); unspecified keys keep their value."""
    unknown = set(kw) - {"interval", "fuse_eg", "hygiene", "autotune", "block", "verbose", "fused_eh", "fused_adj", "fast_obs", "adj_fast"}
    if unknown: raise ValueError(f"unknown b200_kernels options {sorted(unknown)}")
    CFG.update(kw)
    if CFG["interval"] not in ("pil", "full"): raise ValueError("FF_B200_INTERVAL must be pil or full")
    if CFG["block"] is not None and CFG["block"] not in CANDIDATES: raise ValueError(f"FF_B200_BLOCK must be one of {CANDIDATES}")


def set_pil(pil):
    """Pillar layers (bool over z, one contiguous run) = the reconstruction interval of item 1."""
    global _PIL
    p = np.asarray(pil.detach().cpu() if isinstance(pil, torch.Tensor) else pil).astype(bool).ravel()
    idx = np.flatnonzero(p)
    if idx.size < 2 or not (np.diff(idx) == 1).all(): raise ValueError("pil must be one contiguous run of at least two z layers")
    _PIL = (int(idx[0]), int(idx[-1]), int(p.size))


def enable(g=None, **kw):
    """Install the patches (idempotent). g: the caller's globals (ff_opt.py / zonec_multi.py), which supply `pil`."""
    blk = _env("FF_B200_BLOCK", "")
    configure(interval=_env("FF_B200_INTERVAL", "pil"), fuse_eg=_env("FF_B200_FUSE_EG", "1") == "1",
              hygiene=_env("FF_B200_HYGIENE", "1") == "1", autotune=_env("FF_B200_AUTOTUNE", "1") == "1",
              block=int(blk) if blk else None, verbose=_env("FF_B200_VERBOSE", "0") == "1", fused_eh={"0": False, "naive": "naive"}.get(_env("FF_B200_FUSED_EH", "0"), "march"),
              fused_adj=_env("FF_B200_FUSED_ADJ", "0") == "1", fast_obs=_env("FF_FAST_OBS", "0") == "1", adj_fast=_env("FF_B200_ADJ_FAST", "0") == "1")
    configure(**kw)
    if CFG["fast_obs"]: _install_fast_obs()
    if g is not None and g.get("pil") is not None: set_pil(g["pil"])
    if CFG["interval"] == "pil" and _PIL is None: raise ValueError("FF_B200_INTERVAL=pil needs the pillar layers: enable(globals()) after `pil` is defined")
    if not _ORIG:
        _ORIG.update(snapshot=_rc.ReversibleCPMLSimulation._snapshot, run=_rc.ReversibleCPMLSimulation._run,
                     interior=_rk._InteriorCUDA, undo=_rk.InteriorReconstruction._undo_sources,
                     yee_init=_ck.FusedYeeCUDA.__init__, yee_update=_ck.FusedYeeCUDA.update,
                     adj_init=_ca.FusedAdjointCUDA.__init__, adj_step=_ca.FusedAdjointCUDA.step)
        _rc.ReversibleCPMLSimulation._snapshot = _snapshot
        _rc.ReversibleCPMLSimulation._run = _run
        _rk._InteriorCUDA = _Interior
        _rk.InteriorReconstruction._undo_sources = _undo_sources
        _ck.FusedYeeCUDA.__init__ = _yee_init; _ck.FusedYeeCUDA.update = _yee_update
        _ca.FusedAdjointCUDA.__init__ = _adj_init; _ca.FusedAdjointCUDA.step = _adj_step
        _ORIG["advance"] = _rc._advance_recorded; _rc._advance_recorded = _advance_recorded
    if g is not None and callable(g.get("_forward_only")) and g["_forward_only"] is not _forward_only:
        _ORIG["forward_only"] = (g, g["_forward_only"]); g["_forward_only"] = _forward_only          # ff_opt's no-grad path (phase 2)
    _say(f"enabled: interval {CFG['interval']}{' z %d..%d' % _PIL[:2] if CFG['interval'] == 'pil' else ''}, fuse_eg {int(CFG['fuse_eg'])}, "
         f"hygiene {int(CFG['hygiene'])}, block {CFG['block'] or ('autotune' if CFG['autotune'] else 'default')}, fused_eh {CFG['fused_eh'] or 0}, fused_adj {int(CFG['fused_adj'])}"
         f", fast_obs {int(CFG['fast_obs'])}, adj_fast {int(CFG['adj_fast'])}")


def disable():
    """Restore the vendored methods (objects built while enabled keep their kernels)."""
    if not _ORIG: return
    _rc.ReversibleCPMLSimulation._snapshot = _ORIG["snapshot"]; _rc.ReversibleCPMLSimulation._run = _ORIG["run"]
    _rk._InteriorCUDA = _ORIG["interior"]; _rk.InteriorReconstruction._undo_sources = _ORIG["undo"]
    _ck.FusedYeeCUDA.__init__ = _ORIG["yee_init"]; _ck.FusedYeeCUDA.update = _ORIG["yee_update"]
    _ca.FusedAdjointCUDA.__init__ = _ORIG["adj_init"]; _ca.FusedAdjointCUDA.step = _ORIG["adj_step"]
    _rc._advance_recorded = _ORIG["advance"]
    if "forward_only" in _ORIG: g, f = _ORIG["forward_only"]; g["_forward_only"] = f
    _ORIG.clear()


def tune_report():
    """Autotune decisions so far: list of dict(label, bs, ms={bs: fastest ms}, count)."""
    return [dict(v) for v in _TUNED.values()]


# ---------------- 1. reconstruction interval = pillar layers ----------------
def _restricted(self, project, full):
    if CFG.get("interval") != "pil" or _PIL is None or getattr(self, "_b200_full", False): return full
    lo, hi, nz = _PIL
    if project.region.shape[2] != nz or not full[0] <= lo < hi <= full[1]: return full
    return (lo, hi)


def _snapshot(self):
    project, full = _ORIG["snapshot"](self)                 # vendored validation, sources checked against the full interior
    return project, _restricted(self, project, full)


def _run(self, epsilon, spectral, *, fixed_epsilon):
    """The current _run (ff_opt's _run_fast or the vendored one), on the full interval when epsilon differs from fixed_epsilon
    on a layer the restricted interval would take from fixed_epsilon."""
    if CFG.get("interval") == "pil" and _PIL is not None and isinstance(epsilon, torch.Tensor) and isinstance(fixed_epsilon, torch.Tensor) \
            and epsilon.shape == fixed_epsilon.shape and epsilon.device == fixed_epsilon.device and epsilon.dim() >= 3:
        project, full = _ORIG["snapshot"](self)
        a, b = _restricted(self, project, full)
        if (a, b) != full:
            with torch.no_grad():
                same = torch.equal(epsilon[:, :, full[0]:a], fixed_epsilon[:, :, full[0]:a]) and \
                       torch.equal(epsilon[:, :, b + 1:full[1] + 1], fixed_epsilon[:, :, b + 1:full[1] + 1])
            if not same:
                if id(self) not in _WARNED:
                    _WARNED.add(id(self)); _say(f"epsilon differs from fixed_epsilon outside z {a}..{b}: this model uses the full interval {full}")
                self._b200_full = True
                try: return _ORIG["run"](self, epsilon, spectral, fixed_epsilon=fixed_epsilon)
                finally: self._b200_full = False
    return _ORIG["run"](self, epsilon, spectral, fixed_epsilon=fixed_epsilon)


def _undo_sources(self, family, field, n):
    """Vendored undo, minus sources whose z extent misses [a, b]: their cells are poisoned exterior or a halo plane that is
    overwritten from the trace before its next read, so subtracting there is never observed."""
    nz = self.system.region.shape[2]
    for loc, component, wave, profile in reversed(self.system.sources[family]):
        z = loc[2]
        lo, hi = (z.indices(nz)[0], z.indices(nz)[1] - 1) if isinstance(z, slice) else (z, z)
        if hi < self.a or lo > self.b: continue
        field[loc + (component,)] -= wave[n] if profile is None else wave[n] * profile


# ---------------- compile, bind, autotune ----------------
def _compiled(source, device, name):
    import cupy
    key = (source, device)
    if key not in _KERNELS:
        with cupy.cuda.Device(device):
            _KERNELS[key] = _ck._compile.__wrapped__(source, device, cupy.cuda.Device(device).compute_capability, name)
    return _KERNELS[key]


def _bounded(source, name, bs):
    head = f'extern "C" __global__ void {name}('
    assert source.count(head) == 1, f"kernel head {head!r} not found once"
    return source.replace(head, f'extern "C" __global__ void __launch_bounds__({bs}) {name}(')


def _written(source, name):
    """Per parameter: True when the kernel may write it (declared without const)."""
    params = source[source.index(f"{name}(") + len(name) + 1: source.index(")", source.index(f"{name}("))].split(",")
    return [not p.strip().startswith("const") for p in params]


def _best_ms(fn, grid, block, args, device, warm=2, reps=7):
    """Fastest of `reps` timed launches (the minimum is the robust statistic when the GPU is shared)."""
    import cupy
    stream = torch.cuda.current_stream(device)
    with cupy.cuda.Device(device), cupy.cuda.ExternalStream(stream.cuda_stream, device_id=device):
        for _ in range(warm): fn(grid, block, args)
        out = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(stream); fn(grid, block, args); e.record(stream); e.synchronize(); out.append(s.elapsed_time(e))
    return min(out)


def _select(label, source, name, tensors, arrays, count, device, default):
    """(function, module, block size) for this kernel source; autotuned once per source on scratch copies of the written arguments."""
    import cupy
    if CFG["block"]: bs = CFG["block"]
    elif not CFG["autotune"]: bs = default
    else:
        key = hashlib.sha1(f"{device}|{count}|{source}".encode()).hexdigest()
        if key not in _TUNED:
            scratch = [torch.zeros_like(t) if w else None for t, w in zip(tensors, _written(source, name))]
            args = tuple(_ck._direct_cuda_view(cupy, s) if s is not None else a for s, a in zip(scratch, arrays))
            fns = {b: _compiled(_bounded(source, name, b), device, name)[0] for b in CANDIDATES}
            ms = {b: math.inf for b in CANDIDATES}
            for _ in range(2):                                      # two interleaved sweeps
                for b in CANDIDATES: ms[b] = min(ms[b], _best_ms(fns[b], ((count + b - 1) // b,), (b,), args, device))
            del args, scratch
            best = min(ms, key=ms.get)
            _TUNED[key] = dict(label=label, bs=best, ms=ms, count=count)
            msg = f"autotune {label}: block {best} ({ms[best]:.3f} ms; " + ", ".join(f"{b}:{t:.3f}" for b, t in ms.items()) + ")"
            if CFG["verbose"] or len(_TUNED) <= 12: _say(msg)
        bs = _TUNED[key]["bs"]
    fn, module = _compiled(_bounded(source, name, bs), device, name)
    return fn, module, bs


# ---------------- 2+3. reverse interior kernels ----------------
class _Interior:
    """Drop-in for reversible_cpml_kernels._InteriorCUDA (real fields). run('h') reverses H, run('e') reverses E, run('g') adds the
    material VJP. With fuse_eg, run('e') is a no-op and run('g') runs the fused kernel (reverse E + VJP from one curl H); the
    InteriorReconstruction.step order (h, undo E sources, e, adjoint step, g) then puts the E update after the adjoint step, which
    reads no primal field. Expression strings are the vendored ones, so the floats are identical."""

    def __init__(self, s, a, b, gradient, eb):
        self.s = s; self.cp = prepare_cuda_kernels(); self.a = a; self.b = b
        nx, ny, nz = s.region.shape; cn = s.grid.courant_number
        diagonal = s.epsilon.ndim == 4
        depth = b - a + 1; self.count = nx * ny * depth
        hyg = CFG["hygiene"]; self.fuse = CFG["fuse_eg"]
        if hyg and 3 * nx * ny * nz < 2 ** 31:
            common = f'''const int q=blockIdx.x*blockDim.x+threadIdx.x;
 if(q>={self.count})return;
 const int r=q/{depth};const int z=q-r*{depth}+{a};const int y=r%{ny};const int x=r/{ny};
 const int i=(x*{ny}+y)*{nz}+z;
 const int xp=x=={nx - 1}?i-{(nx - 1) * ny * nz}:i+{ny * nz};
 const int xm=x==0?i+{(nx - 1) * ny * nz}:i-{ny * nz};
 const int yp=y=={ny - 1}?i-{(ny - 1) * nz}:i+{nz};
 const int ym=y==0?i+{(ny - 1) * nz}:i-{nz};
 '''
        else:                                                       # the vendored index math
            common = f'''const long long q=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 const long long depth={b - a + 1};if(q>={nx * ny * (b - a + 1)})return;
 const long long z=q%depth+{a},y=(q/depth)%{ny},x=q/(depth*{ny});
 const long long i=(x*{ny}+y)*{nz}+z;
 const long long xp=(((x+1)%{nx})*{ny}+y)*{nz}+z;
 const long long xm=(((x+{nx}-1)%{nx})*{ny}+y)*{nz}+z;
 const long long yp=(x*{ny}+(y+1)%{ny})*{nz}+z;
 const long long ym=(x*{ny}+(y+{ny}-1)%{ny})*{nz}+z;
 '''
        def curl(forward):
            diffs = {}
            for axis, plus, minus in [('x', 'xp', 'xm'), ('y', 'yp', 'ym'), ('z', 'i+1', 'i-1')]:
                for c in range(3):
                    diffs[axis, c] = f'(f[3*({plus})+{c}]-f[3*i+{c}])' if forward else f'(f[3*i+{c}]-f[3*({minus})+{c}])'
            curls = [diffs['y', 2] + '-' + diffs['z', 1], diffs['z', 0] + '-' + diffs['x', 2], diffs['x', 1] + '-' + diffs['y', 0]]
            return ''.join(f'const float c{c}={v};\n' for c, v in enumerate(curls))
        def update(forward, out='out'):
            body = ''
            for c in range(3):
                ep = f'eps[3*i+{c}]' if diagonal else 'eps[i]'
                scale = f'((float){cn:.17g})' if forward else f'(-((float){cn:.17g})/{ep})'
                body += f'{out}[3*i+{c}]+={scale}*c{c};'
            return body
        def vjp(out='out'):
            if diagonal:
                return ''.join(f'{out}[3*i+{c}]+=(-((float){cn:.17g})*bar[3*i+{c}]*c{c})/(eps[3*i+{c}]*eps[3*i+{c}]);' for c in range(3))
            return f'{out}[i]+=(-((float){cn:.17g})*((bar[3*i]*c0+bar[3*i+1]*c1)+bar[3*i+2]*c2))/(eps[i]*eps[i]);'
        R = ' __restrict__' if hyg else ''                        # hygiene off: the vendored head and index math exactly
        head = f'extern "C" __global__ void interior(const float*{R} f,float*{R} out,const float*{R} eps,const float*{R} bar'
        E, H = s.grid.E, s.grid.H
        plan = {'h': (common + '\n' + curl(True) + update(True), (E, H, s.epsilon, eb)),
                'e': (common + '\n' + curl(False) + update(False), (H, E, s.epsilon, eb)),
                'g': (common + '\n' + curl(False) + vjp(), (H, gradient, s.epsilon, eb))}
        if self.fuse:
            plan = {'h': plan['h'], 'eg': (common + '\n' + curl(False) + update(False) + vjp('grad'), (H, E, s.epsilon, eb, gradient))}
        self.kernels = {}
        dev = s.device.index
        with self.cp.cuda.Device(dev), self.cp.cuda.ExternalStream(torch.cuda.current_stream(s.device).cuda_stream, device_id=dev):
            for name, (body, tensors) in plan.items():
                source = head + (f',float*{R} grad' if name == 'eg' else '') + '){' + body + '}'
                arrays = tuple(_ck._direct_cuda_view(self.cp, t) for t in tensors)
                if hyg:
                    fn, module, bs = _select(f"rev_{name}", source, 'interior', tensors, arrays, self.count, dev, 128)
                else:
                    fn, module = _compiled(source, dev, 'interior'); bs = 128
                self.kernels[name] = (fn, arrays, module, bs)
        # one-pass adjoint (phase 2): e_bar alternates between two buffers; 'g'/'eg' read the current one
        ref = getattr(eb, "_b200_adj", None); self.adj = ref() if ref is not None else None
        self.bar_variants = {}
        if self.adj is not None:
            for name, (fn, arrays, module, bs) in self.kernels.items():
                k = 3                                                   # position of `bar` in every interior kernel
                self.bar_variants[name] = {t.data_ptr(): arrays[:k] + (_ck._direct_cuda_view(self.cp, t),) + arrays[k + 1:] for t in self.adj._b200_one.ebar}

    def run(self, name):
        if self.fuse:
            if name == 'e': return
            if name == 'g': name = 'eg'
        fn, arrays, _, bs = self.kernels[name]
        if self.adj is not None: arrays = self.bar_variants[name][self.adj.e_bar.data_ptr()]
        dev = self.s.device.index
        with self.cp.cuda.Device(dev), self.cp.cuda.ExternalStream(torch.cuda.current_stream(self.s.device).cuda_stream, device_id=dev):
            fn(((self.count + bs - 1) // bs,), (bs,), arrays)


# ---------------- 3. fused Yee and adjoint kernels: launch bounds + block size ----------------
def _yee_init(self, grid, *, direct_views=False, bindings_cache=None):
    _ORIG["yee_init"](self, grid, direct_views=direct_views, bindings_cache=bindings_cache)
    self._b200 = None
    if (not CFG.get("hygiene") or type(self) is not _ck.FusedYeeCUDA or bindings_cache is not None or self.interface_update is not None
            or grid.material_states or getattr(grid, 'pmc_lower', None) or getattr(grid, 'pmc_upper', None)):
        return
    tuned = {}
    for forward in (False, True):
        _, arrays, _ = self.launches[forward]
        source, tensors = self._source(forward)
        count = self.launch_count(grid, forward)
        fn, module, bs = _select("yee_" + ("H" if forward else "E"), source, 'yee_update', tensors, arrays, count, self.device, 256)
        tuned[forward] = (fn, arrays, module, count, bs)
    self._b200 = tuned


def _yee_update(self, forward):
    tuned = getattr(self, "_b200", None)
    if tuned is None: return _ORIG["yee_update"](self, forward)
    fn, arrays, _, count, bs = tuned[forward]
    with self.cp.cuda.Device(self.device), self._stream():
        fn(((count + bs - 1) // bs,), (bs,), arrays)


def _adj_init(self, system, gradient, signal_bar, *, direct_views=False, buffers=None, material_gradient=True, electric_seed=None, curl_permittivity=None):
    _ORIG["adj_init"](self, system, gradient, signal_bar, direct_views=direct_views, buffers=buffers, material_gradient=material_gradient,
                      electric_seed=electric_seed, curl_permittivity=curl_permittivity)
    self._b200 = None
    if not CFG.get("hygiene") or type(self) is not _ca.FusedAdjointCUDA or buffers is not None: return
    tuned = {}
    cn = f'((float)({self.system.grid.courant_number:.17g}))'
    with self.cp.cuda.Device(self.device), self.stream():
        for (forward, phase), (_, arrays, _) in self.launches.items():
            code, tensors = self.source(forward, phase)
            if CFG.get("adj_fast") and not self.material_gradient: code = _adj_fast(self, code, forward, cn)
            if not forward and not self.material_gradient and f'{cn}/epsilon[' in code and 'epsilon[' not in code.replace(f'{cn}/epsilon[', ''):
                # every epsilon read of the E-transpose is the scale cn/epsilon[j]: read fl(cn/epsilon) from a table computed once by
                # the same NVRTC division (correctly rounded both ways), in the epsilon slot -> same bytes, no per-term division
                if getattr(self, "_b200_scale", None) is None: self._b200_scale = _scale_table(self.epsilon, cn, self.device)
                code = code.replace(f'{cn}/epsilon[', 'epsilon[')
                k = next(n for n, t in enumerate(tensors) if t is self.epsilon)
                tensors = list(tensors); tensors[k] = self._b200_scale
                arrays = tuple(_ck._direct_cuda_view(self.cp, self._b200_scale) if n == k else a for n, a in enumerate(arrays))
            fn, module, bs = _select("adj_" + ("Ht" if forward else "Et"), code, 'adjoint_update', tensors, arrays, self.count, self.device, 256)
            tuned[forward, phase] = (fn, arrays, module, bs)
    self._b200 = tuned
    self._b200_one = None
    if CFG.get("fused_adj") and _one_eligible(self):
        self._b200_one = _OnePassAdjoint(self, cn)
        import weakref
        for t in self._b200_one.ebar: t._b200_adj = weakref.ref(self)


def _adj_fast(adj, code, forward, cn):
    """Item 5: the vendored adjoint_update source with a branch-free block for the interior cells inserted after `r0=r1=r2=0`.
    Returns `code` unchanged when the grid has anything the block does not reproduce (metric, PEC wall, non-periodic x/y, x/y CPML)."""
    s = adj.system; g = s.grid; nx, ny, nz = s.region.shape; strides = (ny * nz, nz, 1)
    anchor = 'float r0=0,r1=0,r2=0;'
    if (g.E.dtype != torch.float32 or g.metric or getattr(g, "pec_upper", {}) or dict(g.wrap) != {0: 1.0, 1: 1.0} or min(nx, ny, nz) < 3
            or code.count(anchor) != 1 or any(idx for (fw, axis, _), idx in s.keys.items() if axis != 2)):
        return code
    bad = set()
    for (fw, axis, _), idx in s.keys.items():
        if fw == forward and axis == 2:
            for k in idx: sl = s.segments[k]['slice'][2]; bad.update(range(sl.start, sl.stop))
    ok = [z for z in range(1, nz - 1) if z not in bad and z - 1 not in bad]
    if not ok: return code
    zlo, zhi = ok[0], ok[-1]
    if ok != list(range(zlo, zhi + 1)): return code                         # the free z range must be one run
    diagonal = adj.epsilon.shape[-1] == 3
    def eps(index, component):
        return 'epsilon[0]' if adj.epsilon.numel() == 1 else f'epsilon[{"3*(" + index + ")+" + str(component) if diagonal else index}]'
    L = [f'if(x>=1 && x<={nx - 2} && y>=1 && y<={ny - 2} && z>={zlo} && z<={zhi}){{']
    for axis, comp, out, sign in _CURL:                                     # vendored order: per term the lo edge (+=), then the hi edge (-=)
        stride = strides[axis]
        for index, op in ((f'i-{stride}', '+='), ('i', '-=')):
            target = index if forward else f'({index})+{stride}'
            scale = f'-{cn}' if forward else f'{cn}/{eps(target, out)}'
            L.append(f'{{float d=({"-" if sign < 0 else ""}(({scale})*bar[3*({target})+{out}]));r{comp}{op}d;}}')
    L += [f'target[3*i+{c}]+=r{c};' for c in range(3)] + ['return;', '}']
    return code.replace(anchor, anchor + '\n' + '\n'.join(L), 1)


# ---------------- item 4 (FF_FAST_OBS=1): monitor tables once per observer tuple ----------------
_OBS = {}                                                                   # key -> (observer tuple kept alive, cached tables)


def _obs_key(system, kind):
    src = getattr(system, "_fa_obs_src", None)
    g = system.grid
    if not isinstance(src, tuple) or len(system.monitors) != len(src) or any(g.pmc_blocks.get(f) for f in ('E', 'H')): return None
    return (kind, id(src), tuple(system.region.shape), str(system.device))


def _install_fast_obs():
    import torchfdtd.differentiable as _df
    if "sys_init" in _ORIG_OBS: return
    _ORIG_OBS.update(sys_init=_df._System.__init__, prep=_df._System.prepare_observations, obsk=_ca.FusedAdjointCUDA.observer_kernel)

    def sys_init(self, project, epsilon, *a, observation_monitors=None, **k):
        self._fa_obs_src = observation_monitors if isinstance(observation_monitors, tuple) else None
        return _ORIG_OBS["sys_init"](self, project, epsilon, *a, observation_monitors=observation_monitors, **k)

    def prep(self):
        key = _obs_key(self, "prep")
        if key is None: return _ORIG_OBS["prep"](self)
        hit = _OBS.get(key)
        if hit is None:
            _ORIG_OBS["prep"](self)
            _OBS[key] = (self._fa_obs_src, (list(self.observation_maps), list(self.face_observation_maps)))
            return
        self.observation_maps, self.face_observation_maps = list(hit[1][0]), list(hit[1][1])

    def obsk(self):
        key = _obs_key(self.system, "observer") if self.buffers is None and self.direct_views else None
        if key is None: return _ORIG_OBS["obsk"](self)
        hit = _OBS.get(key)
        if hit is None:
            out = _ORIG_OBS["obsk"](self)
            if out is not None: _OBS[key] = (self.system._fa_obs_src, (out[0], out[2], out[3], out[1][-1], self.observer_layout))
            return out
        fn, module, count, nmon, layout = hit[1]
        self.observer_layout = layout
        arrays = tuple(self.view(t) for t in (self.e_bar, self.h_bar, self.signal_bar)) + (self.view(layout),)
        return fn, (*arrays, np.int32(count), nmon), module, count

    _df._System.__init__ = sys_init; _df._System.prepare_observations = prep; _ca.FusedAdjointCUDA.observer_kernel = obsk


_ORIG_OBS = {}


def _scale_table(epsilon, cn, device):
    """fl(cn / epsilon) elementwise, compiled with the solver's NVRTC options (IEEE division, no FMA, no FTZ)."""
    import cupy
    out = torch.empty_like(epsilon)
    n = epsilon.numel()
    fn, _ = _compiled('extern "C" __global__ void scale_table(const float* __restrict__ e, float* __restrict__ s, const long long n) {'
                      f'const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) s[i] = {cn}/e[i];' '}', device, 'scale_table')
    with cupy.cuda.Device(device), cupy.cuda.ExternalStream(torch.cuda.current_stream(epsilon.device).cuda_stream, device_id=device):
        fn(((n + 255) // 256,), (256,), (_ck._direct_cuda_view(cupy, epsilon), _ck._direct_cuda_view(cupy, out), np.int64(n)))
    return out


def _adj_step(self, index, *, observation_index=None):
    tuned = getattr(self, "_b200", None)
    if tuned is None: return _ORIG["adj_step"](self, index, observation_index=observation_index)
    one = getattr(self, "_b200_one", None)
    if one is not None: return one.step(index if observation_index is None else observation_index)
    with self.cp.cuda.Device(self.device), self.stream():
        if self.observer is not None:
            fn, arrays, _, count = self.observer
            fn(((count + 127) // 128,), (128,), (*arrays, np.int32(index if observation_index is None else observation_index)))
        for forward in (True, False):
            fn, arrays, _, bs = tuned[forward, self.phase]
            fn(((self.count + bs - 1) // bs,), (bs,), arrays)
    self.phase = 1 - self.phase


# ---------------- phase 2 (FF_B200_FUSED_EH=1, default off): one-pass E+H forward ----------------
# One kernel per step instead of update_E, inject, update_H. E, H and the E-CPML memories are double-buffered, so no thread reads
# a value another thread writes. Each thread recomputes the E components its H update differences (own cell, +x, +y, +z) with the
# vendored E-update expression (same terms, same order, --fmad=false), applies the soft E sources to every such value (fl(E + w),
# as torch's inject), writes its own E and E-CPML memory, then updates its own H (H-CPML in place: only the owner touches it).
# After the launch the buffers swap (grid.E/H and the segment psi entries point at the new state), so observe, traces, the
# terminal copy and the backward see exactly the vendored state. Scope: real FP32 fields, periodic x/y, CPML on z only, uniform
# mesh, dielectric, E sources that are boxes with profile None, no PMC/subpixel/ADE; anything else keeps the vendored loop.
from torchfdtd.boundaries import CURL_TERMS as _CURL


def _eh_eligible(system):
    g = system.grid; shape = tuple(system.region.shape)
    if system.field_dtype != torch.float32 or g.E.dtype != torch.float32 or not g.E.is_cuda or system.pmc: return False
    if g.material_states or getattr(g, "subpixel", None) is not None or g.metric or dict(g.wrap) != {0: 1.0, 1: 1.0}: return False
    if min(shape) < 2 or 3 * math.prod(shape) >= 2 ** 31: return False
    if g.inverse_permittivity is None or g.inverse_permittivity.numel() != 3 * math.prod(shape) or g.inverse_permeability.numel() != 1: return False
    if any(segs for (fw, axis, comp), segs in g.cpml.items() if axis != 2): return False
    if system.sources["H"] or any(system.face_sources.values()): return False
    for loc, comp, wave, profile in system.sources["E"]:
        if profile is not None or len(loc) != 3 or any(isinstance(s, slice) and s.step not in (None, 1) for s in loc): return False
    return True


class _EHBase:
    """Double-buffered one-pass E+H forward step. Binding shared by the two kernel layouts below: E, H and the E-CPML memories are
    double-buffered (old -> new, then swap), the H-CPML memories are updated in place (only the owner thread touches them)."""

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
        """The vendored E update of component `out` at cell j (coordinates X, Y, Z). h(axis, comp): the H operand at the cell (axis None)
        or at its -axis neighbour, wrapped across the periodic x/y seam (the vendored seam branch differences the same two values)."""
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
        """The vendored H update of the own cell (x, y, z); e(axis, comp): E_new at the cell (None) or its +axis neighbour."""
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


class _FusedEH(_EHBase):
    """One thread per cell; the E values its H update differences (own, +x, +y, +z) are recomputed by that thread."""

    def __init__(self, system):
        import cupy
        self.cp = cupy; self.system = system; g = system.grid
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
            if CFG["hygiene"]:
                tensors, views = self.args[0]
                self.fn, self.module, bs = _select("fused_EH", self.source, 'eh_update', tensors + [torch.zeros(1)], views + (np.int32(0),), self.count, self.dev, 256)
            else:
                (self.fn, self.module), bs = _compiled(_bounded(self.source, 'eh_update', 256), self.dev, 'eh_update'), 256
        self.grid, self.block = ((self.count + bs - 1) // bs,), (bs,)
        self.parity = 0


class _MarchEH(_EHBase):
    """2.5D blocking: a block owns TY y-rows x all z (ZB >= nz threads per row, two extra thread rows for the y halo) and marches
    along an x-chunk. Per plane x it loads H_old(x+1) (rows y0-1 .. y0+TY) to shared memory, computes E_new(x+1) once for rows
    y0 .. y0+TY (the last row is the halo the H update's y difference needs; x+1 = x1 is the next chunk's first plane, computed
    here as a halo and written there), then H_new(x) of its own rows. Each E value is computed once per block; global traffic is
    the ~60 B/cell of one read and one write of E and H plus the eps table."""

    def __init__(self, system, TY=4, xchunks=None):
        import cupy
        self.cp = cupy; self.system = system; g = system.grid
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


def _eh_engine(system):
    """FF_B200_FUSED_EH=1 (or march): the marching kernel; =naive: one thread per cell with recomputed neighbours."""
    if CFG.get("fused_eh") != "naive":
        try: return _MarchEH(system)
        except ValueError: pass                                      # z extent beyond one 1024-thread tile
    return _FusedEH(system)


def _advance_recorded(system, step, frame, a, b):
    """reversible_cpml._advance_recorded; with fused_eh the one-pass kernel and the same two trace planes."""
    eng = getattr(system, "_b200_eh", None)
    if eng is None:
        if not (CFG.get("fused_eh") and step == 0 and system.kernel is not None and _eh_eligible(system)):
            return _ORIG["advance"](system, step, frame, a, b)
        eng = system._b200_eh = _eh_engine(system)
    eng.step(step)
    frame[0].copy_(eng.E[1 - eng.parity][:, :, b + 1, :2])          # post-E-update, post-injection E at b+1
    frame[1].copy_(eng.H[eng.parity][:, :, a - 1, :2])              # pre-H-update H at a-1
    eng.swap()
    system.current_step = step + 1


def _forward_only(epsilon, project, options, interval, report, spectral):
    """ff_opt._forward_only with the one-pass E+H kernel (same loop, same observation and accumulation order)."""
    g, orig = _ORIG["forward_only"]
    if not (CFG.get("fused_eh") and epsilon.is_cuda): return orig(epsilon, project, options, interval, report, spectral)
    import time
    system = _rc._System(project, epsilon.detach(), prepare_kernels=False, observation_monitors=None if spectral is None else spectral.observers)
    if not _eh_eligible(system):
        del system; return orig(epsilon, project, options, interval, report, spectral)
    eng = _eh_engine(system)
    steps = project.region.steps; block_size = steps if spectral is None else spectral.block_size
    samples = system.grid.E.new_empty((block_size, len(system.monitors))); signals = samples if spectral is None else spectral.zeros()
    started = time.perf_counter()
    for step in range(steps):
        eng.step(step); eng.swap()
        system.current_step = step + 1
        row = step % block_size; samples[row] = system.observe(system.state())
        if spectral is not None and (row + 1 == block_size or step + 1 == steps): spectral.accumulate(signals, samples[:row + 1], step - row)
    rf = g["_require_finite_once"]
    rf(signals, 'Forward-only CPML observations became nonfinite.', 0)
    e, h, *_ = system.state(); rf(e, 'Forward-only E became nonfinite.', 0); rf(h, 'Forward-only H became nonfinite.', 0)
    torch.cuda.synchronize(epsilon.device); report.update(forward_seconds=time.perf_counter() - started, forward_only=True, fused_eh=True)
    return signals


# ---------------- phase 2 (FF_B200_FUSED_ADJ=1, default off): one-pass adjoint step ----------------
# The reversible backward's adjoint step is the observer, the H-transpose (e_bar += gather of -cn h_bar) and the E-transpose
# (h_bar += gather of fl(cn/eps) e_bar). One kernel does both transposes: e_bar and h_bar are double-buffered (the CPML adjoint
# memories already are), each thread recomputes the post-H-transpose e_bar values its E-transpose gathers (own cell, +x, +y, +z)
# with the vendored per-term code (same branches in the same order, so every accumulation rounds the same way), writes its own
# e_bar and h_bar and, as the owner, its CPML memories. The observer still runs first on the current buffers; the interior
# 'eg' kernel reads the new e_bar (the post-H-transpose value it read before).
def _one_eligible(adj):
    s = adj.system; g = s.grid; shape = tuple(s.region.shape)
    if type(adj) is not _ca.FusedAdjointCUDA or adj.material_gradient or adj.buffers is not None or adj.electric_seed is not adj.e_bar: return False
    if g.E.dtype != torch.float32 or g.metric or dict(g.wrap) != {0: 1.0, 1: 1.0} or getattr(g, "pec_upper", {}) or adj.phase != 0: return False
    if min(shape) < 2 or 3 * math.prod(shape) >= 2 ** 31 or getattr(adj, "_b200_scale", None) is None: return False
    return not any(idx for (fw, axis, comp), idx in s.keys.items() if axis != 2)


class _OnePassAdjoint:
    def __init__(self, adj, cn):
        cupy = adj.cp; s = adj.system; self.adj = adj
        nx, ny, nz = s.region.shape; S = (ny * nz, nz, 1); self.count = nx * ny * nz
        diagonal = adj.epsilon.shape[-1] == 3
        def sc(t, out): return f'sc[3*({t})+{out}]' if diagonal else f'sc[{t}]'
        self.ebar = [adj.e_bar, torch.zeros_like(adj.e_bar)]; self.hbar = [adj.h_bar, torch.zeros_like(adj.h_bar)]
        params = ['const float* __restrict__ eo', 'const float* __restrict__ ho', 'float* __restrict__ en', 'float* __restrict__ hn',
                  'const float* __restrict__ sc']
        segs = {}                                                    # (forward, comp) -> [(k, lo, hi)]
        for (fw, axis, comp), idx in s.keys.items():
            for k in idx:
                seg = s.segments[k]; segs.setdefault((fw, comp), []).append((k, seg['slice'][2].start, seg['slice'][2].stop))
                params += [f'const float* __restrict__ b_{k}', f'const float* __restrict__ c_{k}', f'const float* __restrict__ inv_k_{k}',
                           f'const float* __restrict__ old_{k}', f'float* __restrict__ new_{k}']

        def terms(forward, term_list, j, X, Y, Z, write, acc, value, scale):
            """The vendored adjoint_update code of the given CURL_TERMS entries, in order; value(kind, target, out) / scale(...) give the
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
        # own e_bar: the vendored H-transpose accumulates r0, r1, r2 over all terms, each term touching only its own r
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
        for term in _CURL:                                           # E-transpose: every term in the vendored order, each into s{comp}
            value, scale = eval_(term[0])
            body += terms(False, [term], 'i', 'x', 'y', 'z', True, {0: 's0', 1: 's1', 2: 's2'}, value, scale)
        body += [f'en[3*i+{c}]=e{c};' for c in range(3)] + [f'hn[3*i+{c}]=ho[3*i+{c}]+s{c};' for c in range(3)]
        self.source = 'extern "C" __global__ void adj_one(' + ','.join(params) + '){\n' + '\n'.join(body) + '\n}'
        names = [p.split()[-1] for p in params]
        self.args = []
        for parity in (0, 1):
            t = dict(eo=self.ebar[parity], ho=self.hbar[parity], en=self.ebar[1 - parity], hn=self.hbar[1 - parity], sc=adj._b200_scale)
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
        self.fn, self.module, self.bs = _select("adj_one", self.source, 'adj_one', tensors, views, self.count, adj.device, 256)
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

