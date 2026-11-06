"""Non-symmetric D200 topology optimization (E2 RGB hologram, E3 volume hologram) and the common full-wave evaluator, on N GPUs.

Design: one density on the 10 nm Cartesian grid of the whole circular aperture (20000 x 20000 px over [-100, 100] um, no symmetry
replication), filter + tanh projection + Adam exactly as loop_zonec.py (FA_NS branch; the same EZC_* recipe knobs). FDTD: the
D200 contract of ff_opt.py (20 nm Yee lattice, z 2.5 um, 1200 steps, SiN 700 nm n 2.0 on SiO2 n 1.444, FP32, periodic tile faces
+ z CPML, reversible adjoint), on every tile of a square core grid that carries an aperture pupil cell. Each FDTD E component gets
the 2 x 2 design-px mean centred on it (loop_levelset10.tile_q10; FF_L9_ANISO=1 adds the Meep diagonal fill with fixed normals), assembled per tile window of core + 2 OVER;
the gradient is the exact transpose of that window map, margins included (FF_OWNGRAD=0 semantics), summed on the global grid.
Work unit = one (tile, polarization) solve (FA_POLS=x for E2/E3, xy for scoring E1 designs); rank 0 receives unit fields and
gradients over NCCL in canonical tile order, so the numbers do not depend on the GPU count.
usage: python -m torch.distributed.run --standalone --nproc_per_node=N ns_run.py <widths.npy> <out_dir> [steps] [--eval-only <density.npz>]
       widths.npy: the 800 x 800 bookkeeping map ff_opt.py reads (uniform 240, not used by the physics)
density files (--eval-only, EZC_INIT=density:<npz>): x-major arrays on the 10 nm grid, px (i, j) centred at (-100 + (i + 0.5) 0.01,
       -100 + (j + 0.5) 0.01) um: key density (float/uint8/bool N x N), d10_bits + d10_shape (packbits), density20 (20 nm grid, each
       value covers 2 x 2 px), or d10_quad + d10_shape (the D4 quadrant of loop_zonec step files, unfolded by the mirror symmetry).
       Values outside the aperture are set to 0, as the optimizer does.
"""
import os, sys, time, json, math, hashlib, traceback
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch, torch.distributed as dist

_NS_HERE = Path(os.path.abspath(__file__)).parent
EVAL_FILE = None
if "--eval-only" in sys.argv:                                              # strip before ff_opt.py reads argv
    _i = sys.argv.index("--eval-only"); EVAL_FILE = sys.argv[_i + 1]; del sys.argv[_i:_i + 2]
TRACE_BENCH = "--trace-bench" in sys.argv                                  # device-trace vs kept host-trace solve of one unit, interleaved (timing)
if TRACE_BENCH: sys.argv.remove("--trace-bench")
FD_CHECK = "--fd-check" in sys.argv                                        # directional finite difference of the assembled model (local check)
if FD_CHECK: sys.argv.remove("--fd-check")
ANISO = os.environ.get("FF_L9_ANISO", "0") == "1"                         # 1: Meep diagonal fill with fixed interface normals (loop_levelset10.tile_q10 / eps_of)
if os.environ.get("FF_OWNGRAD", "0") != "0": raise SystemExit("ns_run.py sums the margin gradients (FF_OWNGRAD=0) only")
_BACKEND = os.environ.get("FF_DIST_BACKEND", "nccl" if torch.cuda.is_available() and sys.platform != "win32" else "gloo")
_local = int(os.environ.get("LOCAL_RANK", "0")) % max(torch.cuda.device_count(), 1); torch.cuda.set_device(_local)
if int(os.environ.get("WORLD_SIZE", "1")) > 1:                            # one process: no process group (all exchanges are local)
    dist.init_process_group(_BACKEND, timeout=timedelta(seconds=int(os.environ.get("FF_NCCL_TIMEOUT_S", "7200"))),
                            **({"device_id": torch.device("cuda", _local)} if _BACKEND == "nccl" else {}))
RANK, WORLD = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
os.environ["LOCAL_RANK"] = str(_local); os.environ["FF_MULTI_RANK"] = str(RANK)
if RANK != 0 and os.environ.get("FF_MULTI_VERBOSE", "0") != "1":
    def print(*a, **k): pass
_T_SETUP = time.time()
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False   # the filter is rebuilt after the solves: same floats

# ---------------- ff_opt.py: head A (tile project, solver options), B (reference), P (solver patches, b200 kernels) ----------------
_SRC = open(_NS_HERE / "ff_opt.py", encoding="utf-8").read()
_MARK = ["# ---------------- reference: homogeneous SiO2", "# ---------------- design parameters (octant)",
         "# ---------------- solver diagnostic: one reduction per large chunk", "# ---------------- optimization loop ----------------"]
for _m in _MARK: assert _SRC.count(_m) == 1, f"ff_opt.py anchor not found exactly once: {_m!r}"
_i1, _i2, _i3, _i4 = (_SRC.index(_m) for _m in _MARK)
def _exec(a, b, name): exec(compile("\n" * _SRC.count("\n", 0, a) + _SRC[a:b], name, "exec"), globals())
if RANK == 0: _exec(0, _i1, "ff_opt.py[head A]"); _t_ref = time.time(); _exec(_i1, _i2, "ff_opt.py[head B]"); _t_ref = time.time() - _t_ref
if WORLD > 1: dist.barrier()
if RANK != 0: _exec(0, _i1, "ff_opt.py[head A]"); _t_ref = time.time(); _exec(_i1, _i2, "ff_opt.py[head B]"); _t_ref = time.time() - _t_ref
_exec(_i3, _i4, "ff_opt.py[patches]")
import fa_runtime as _NR, fa_dist as _D
PL = _NR.phase_log(OUT, RANK); PL.event(None, "reference", _t_ref)
Fn = torch.nn.functional

# ---------------- geometry: 10 nm Cartesian design grid, aperture, tiles ----------------
SUB = 2; N10 = NPX * SUB; MESH10 = MESH / SUB; H10 = N10                   # H10: the saved-shape convention of loop_zonec (d10_shape)
_xn = (torch.arange(NPX, device=dev, dtype=torch.float64) + 0.5) * (MESH * 1e3) + XG0 * 1e3   # 20 nm px centres (nm), as the D4 neutral cache
AP20 = torch.empty((NPX, NPX), dtype=torch.bool, device=dev)
SQUARE = bool(LENS) and LENS.get("SHAPE") == "square"                 # 2026-09-30: square aperture |x|, |y| <= R_AP (R_AP = half-width)
for _r in range(0, NPX, 1000):
    AP20[_r:_r + 1000] = ((_xn[_r:_r + 1000, None].abs() <= R_AP * 1e3) & (_xn[None, :].abs() <= R_AP * 1e3)) if SQUARE else (torch.hypot(_xn[_r:_r + 1000, None], _xn[None, :]) <= R_AP * 1e3)
AP10 = AP20.repeat_interleave(SUB, 0).repeat_interleave(SUB, 1); del AP20, _xn
KZ = 1; NRMAX = AMAX = N10; VALID = AP10[None]                             # loop_zonec cells = design px
NS_POLS = tuple(os.environ.get("FA_POLS", "x")); assert NS_POLS in (("x",), ("x", "y")), "FA_POLS must be x or xy"
JC_NS = tuple(k for p in NS_POLS for k in (("xx", "yx") if p == "x" else ("xy", "yy")))
L_MIN = int(os.environ.get("FF_L9_MIN", "6")); L_GAP = int(os.environ.get("FF_L9_GAP", "8"))   # recorded in history only
_cc = (np.arange(NCELL) - NCELL // 2 + 0.5) * PITCH
_apc = ((np.abs(_cc)[:, None] <= R_AP) & (np.abs(_cc)[None, :] <= R_AP)) if SQUARE else (_cc[:, None] ** 2 + _cc[None, :] ** 2 <= R_AP ** 2)
def _grid(off):                                                           # cores [x0 -+ c/2) x [y0 -+ c/2) with an aperture pupil-cell centre
    n = int(math.ceil((R_AP + TILE) / TILE)) + 1; out = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x0, y0 = off + i * TILE, off + j * TILE
            inx = (_cc >= x0 - TILE / 2) & (_cc < x0 + TILE / 2); iny = (_cc >= y0 - TILE / 2) & (_cc < y0 + TILE / 2)
            if (_apc & inx[:, None] & iny[None, :]).any(): out.append((float(x0), float(y0)))
    return out
_GRID_MODE = os.environ.get("FA_TILE_GRID", "auto")
tiles = (_grid(TILE / 2) if _GRID_MODE == "corner" else _grid(0.0) if _GRID_MODE == "centre" else _grid((-R_AP + TILE / 2) % TILE) if _GRID_MODE == "edge"
         else min(_grid(TILE / 2), _grid(0.0), key=len))                  # edge: a tile edge on -R_AP (square aperture, e.g. 8 x 8 cores of 19 um over 150 um)
tiles = sorted(tiles)
def tile_window(x0, y0): return int(round((x0 - L / 2 - XG0) / MESH)), int(round((y0 - L / 2 - XG0) / MESH))
_cov = np.zeros((NCELL, NCELL), bool)
for (x0, y0) in tiles: _cov |= ((_cc >= x0 - TILE / 2) & (_cc < x0 + TILE / 2))[:, None] & ((_cc >= y0 - TILE / 2) & (_cc < y0 + TILE / 2))[None, :]
assert (_cov | ~_apc).all(), "tile cores do not cover every aperture pupil cell"
UNITS = [(key, p) for key in tiles for p in NS_POLS]                      # canonical order: tiles sorted, x before y
def owners(units):                                                        # contiguous split, remainder to the LAST ranks (rank 0 fewest)
    q, m = divmod(len(units), WORLD); own, i = {}, 0
    for r in range(WORLD):
        n = q + (1 if r >= WORLD - m else 0)
        for u in units[i:i + n]: own[u] = r
        i += n
    return own
OWN = owners(UNITS)

# ---------------- design window -> Yee E-component fill, solver, pupil cells ----------------
_EPS_S = N_SIN ** 2
N_SIG = float(os.environ.get("FF_L9_NORMAL_SIGMA", "1")); PADN = int(math.ceil(3 * N_SIG)) + 1 if ANISO else 0   # as loop_levelset10
def _gblur(x, s):
    if s <= 0: return x.float()
    r = int(math.ceil(3 * s)); t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32); k = torch.exp(-t ** 2 / (2 * s * s)); k = k / k.sum()
    y = Fn.conv2d(x.float()[None, None], k.view(1, 1, -1, 1), padding=(r, 0))
    return Fn.conv2d(y, k.view(1, 1, 1, -1), padding=(0, r))[0, 0]
def tile_q(Dg, i0, j0):
    """(nx, ny, 3) = (f_Ex, f_Ey, f_Ez): the 2 x 2 design-px means centred on each E component of the 20 nm tile window at lattice
    (i0, j0), zero outside the design grid (loop_levelset10.tile_q10 channels 0..2); with FF_L9_ANISO=1 (nx, ny, 5) plus the squared
    interface normals nx^2 at Ex, ny^2 at Ey from the sigma-N_SIG blurred window (channels 3, 4, used detached)."""
    nx, ny = shape[0], shape[1]; r0, c0 = 2 * i0 - 1 - PADN, 2 * j0 - 1 - PADN; h, w = 2 * nx + 2 + 2 * PADN, 2 * ny + 2 + 2 * PADN
    W = torch.zeros((h, w), dtype=torch.float32, device=dev)
    a0, a1, b0, b1 = max(r0, 0), min(r0 + h, N10), max(c0, 0), min(c0 + w, N10)
    if a0 < a1 and b0 < b1: W[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = Dg[a0:a1, b0:b1]
    P = PADN
    fx = Fn.avg_pool2d(W[P + 1:P + 1 + 2 * nx, P:P + 2 * ny][None, None], 2)[0, 0]
    fy = Fn.avg_pool2d(W[P:P + 2 * nx, P + 1:P + 1 + 2 * ny][None, None], 2)[0, 0]
    fz = Fn.avg_pool2d(W[P:P + 2 * nx, P:P + 2 * ny][None, None], 2)[0, 0]
    if not ANISO: return torch.stack([fx, fy, fz], -1)
    Ws = _gblur(W, N_SIG)
    def grads(oa, ob):
        sb = Ws[oa: oa + 2 * nx, ob: ob + 2 * ny]
        gx = Fn.avg_pool2d((sb[1::2, :] - sb[0::2, :])[None, None], (1, 2))[0, 0]; gy = Fn.avg_pool2d((sb[:, 1::2] - sb[:, 0::2])[None, None], (2, 1))[0, 0]
        return gx, gy
    gxx, gyx = grads(1 + P, P); gxy, gyy = grads(P, 1 + P)
    nx2 = gxx ** 2 / (gxx ** 2 + gyx ** 2 + 1e-12); ny2 = gyy ** 2 / (gxy ** 2 + gyy ** 2 + 1e-12)
    return torch.stack([fx, fy, fz, nx2, ny2], -1)
def tile_q_T(g, i0, j0, G):
    """G += the exact transpose of tile_q for the window gradient g (nx, ny, 3): each design px gets 1/4 of every component window it
    lies in (loop_levelset10.region_gradient10), clipped to the grid."""
    nx, ny = shape[0], shape[1]; buf = torch.zeros((2 * nx + 2, 2 * ny + 2), device=dev)
    up = lambda c: g[..., c].float().repeat_interleave(2, 0).repeat_interleave(2, 1) / 4
    buf[1: 1 + 2 * nx, 0: 2 * ny] += up(0); buf[0: 2 * nx, 1: 1 + 2 * ny] += up(1); buf[0: 2 * nx, 0: 2 * ny] += up(2)
    r0, c0 = 2 * i0 - 1, 2 * j0 - 1; a0, a1, b0, b1 = max(r0, 0), min(r0 + 2 * nx + 2, N10), max(c0, 0), min(c0 + 2 * ny + 2, N10)
    if a0 < a1 and b0 < b1: G[a0:a1, b0:b1] += buf[a0 - r0:a1 - r0, b0 - c0:b1 - c0]
def eps_of_ns(q):                                                         # (nx, ny, 3|5) -> diagonal Yee eps (nx, ny, nz, 3), as loop_levelset10.eps_of
    def comp(f, n2):
        ea = 1.0 + f * (_EPS_S - 1.0); ih = (1.0 - f) + f / _EPS_S
        return 1.0 / (n2 * ih + (1.0 - n2) / ea) if ANISO else ea
    if ANISO: lay = torch.stack([comp(q[..., 0], q[..., 3].detach()), comp(q[..., 1], q[..., 4].detach()), 1.0 + q[..., 2] * (_EPS_S - 1.0)], -1)
    else: lay = torch.stack([1.0 + q[..., 0] * (_EPS_S - 1.0), 1.0 + q[..., 1] * (_EPS_S - 1.0), 1.0 + q[..., 2] * (_EPS_S - 1.0)], -1)
    eps = torch.ones(shape + (3,), device=dev, dtype=torch.float32)
    eps = torch.where(sub[None, None, :, None], torch.full_like(eps, N_SIO2 ** 2), eps)
    eps = torch.where(pil[None, None, :, None], lay[:, :, None, :].expand(shape + (3,)), eps)
    return eps.contiguous()
project_y = project.model_copy(deep=True); project_y.sources[0].component = "Ey"; project_y = Project.model_validate(project_y.model_dump())
def ns_model(key, pol, options=None):
    global project
    if pol == "x": p, counts = tile_project(*key)
    else:
        p0 = project; project = project_y
        try: p, counts = tile_project(*key)
        finally: project = p0
    return shared_model(p, counts, 3, options)
LO = torch.as_tensor(lam_order.copy(), device=dev); FA_LAST_REPORT = None
def ns_solve(q, key, pol, options=None):                                  # exit-plane (Ex, Ey) for a pol-polarized plane wave, LAM_NM ascending
    global FA_LAST_REPORT
    eps = eps_of_ns(q)                                                    # fixed exterior = this eps itself (detached): outside the pillar layers it equals the
    det = ns_model(key, pol, options)(eps, FREQ_T.to(dev), fixed_epsilon=eps.detach(), block_size=32)["xy"]; FA_LAST_REPORT = det.report   #  background exactly, and no 2 GB background map stays resident
    return det.fields[:, :, 0][LO], det.fields[:, :, 1][LO], torch.as_tensor(det.points_um, device=dev, dtype=torch.float32)
def cell_ids(pts, x0, y0):
    gx = pts[:, 0] + x0; gy = pts[:, 1] + y0
    keep = (gx >= x0 - TILE / 2) & (gx < x0 + TILE / 2) & (gy >= y0 - TILE / 2) & (gy < y0 + TILE / 2)
    ci = torch.floor(gx / PITCH).long() + NCELL // 2; cj = torch.floor(gy / PITCH).long() + NCELL // 2
    keep &= (ci >= 0) & (ci < NCELL) & (cj >= 0) & (cj < NCELL)
    return keep, ci, cj
def tile_cells(key, Ex, pts):                                              # ff_opt.tile_cells with the full-plane core cut
    x0, y0 = key; keep, ci, cj = cell_ids(pts, x0, y0); flat = (ci * NCELL + cj)[keep]
    acc = torch.zeros((len(LAM_NM), NCELL * NCELL), dtype=torch.complex64, device=dev).index_add(1, flat, Ex[:, keep]).view(len(LAM_NM), NCELL, NCELL)
    cnt = torch.zeros(NCELL * NCELL, device=dev).index_add(0, flat, torch.ones(int(keep.sum()), device=dev)).view(NCELL, NCELL)
    return acc, cnt
def _norm(a, cnt): return a / cnt.clamp(min=1)[None] / E_ref[:, None, None] * JONES_SCALE
_JK = {"x": ("xx", "yx"), "y": ("xy", "yy")}

# ---------------- units and phases ----------------
import fa_keep_trace as _KT                                              # FA_KEEP_TRACE=1: one recorded forward per unit and step (fa_keep_trace.py)
if _KT.ENABLED: _KT.install()
KEEP = _KT.Keeper(dev)
def unit_fwd(u, Dg, keep=False):
    (x0, y0), pol = u; i0, j0 = tile_window(x0, y0)
    if keep:                                                              # recorded forward with a live graph; the adjoint phase runs only the backward
        m = ns_model(u[0], pol, ADJ_KEEP)
        if KEEP.admit(KEEP.reservation(m, FREQ_T), math.prod(shape)):
            before = torch.cuda.memory_allocated(dev)
            with torch.no_grad(): leaf = tile_q(Dg, i0, j0)
            leaf.requires_grad_(True)
            a, b, pts = ns_solve(leaf, u[0], pol, ADJ_KEEP); KEEP.store(u, leaf, a, b, pts, before)
            return torch.view_as_real(torch.stack([a.detach(), b.detach()])).contiguous(), pts.contiguous()
    with torch.no_grad():
        try: a, b, pts = ns_solve(tile_q(Dg, i0, j0), u[0], pol)
        except ValueError as e:                                           # the solver refused for lack of free GPU memory: drop the newest kept unit (its adjoint then
            if "CUDA budget" not in str(e) or not KEEP.held: raise        #  recomputes it, as for a declined unit) and retry
            KEEP.held.popitem(); import gc; gc.collect(); torch.cuda.empty_cache()
            a, b, pts = ns_solve(tile_q(Dg, i0, j0), u[0], pol)
    return torch.view_as_real(torch.stack([a, b])).contiguous(), pts.contiguous()
def unit_grad_kept(u, g, cnt):                                            # backward of a kept unit: the unit_grad arithmetic on the stored graph
    ka, kb = _JK[u[1]]; leaf, fa, fb, pts = KEEP.pop(u)
    oa = _norm(tile_cells(u[0], fa, pts)[0], cnt); ob = _norm(tile_cells(u[0], fb, pts)[0], cnt)
    torch.autograd.backward([oa, ob], [g[JC_NS.index(ka)], g[JC_NS.index(kb)]])
    out = leaf.grad.detach()[..., :3].contiguous(); del fa, fb, oa, ob, leaf   # normals are detached: channels 3, 4 carry no gradient
    return out
def unit_grad(u, Dg, g, cnt):
    (x0, y0), pol = u; i0, j0 = tile_window(x0, y0); ka, kb = _JK[pol]
    with torch.no_grad(): leaf = tile_q(Dg, i0, j0)
    leaf.requires_grad_(True)
    fa, fb, pts = ns_solve(leaf, u[0], pol)
    oa = _norm(tile_cells(u[0], fa, pts)[0], cnt); ob = _norm(tile_cells(u[0], fb, pts)[0], cnt)
    torch.autograd.backward([oa, ob], [g[JC_NS.index(ka)], g[JC_NS.index(kb)]])
    out = leaf.grad.detach()[..., :3].contiguous(); del fa, fb, oa, ob, leaf   # normals are detached: channels 3, 4 carry no gradient
    return out
def run_phase(kind, step, Dg, g=None, cnt=None):
    """This rank's units of one phase; rank 0 returns every unit's tensors (canonical order), the others {} (None on failure)."""
    mine = [u for u in UNITS if OWN[u] == RANK]; local = {}; ok = True
    if kind == "forward": KEEP.clear()                                    # a new design: graphs of an earlier step are stale
    if kind == "adjoint": mine = [u for u in mine if u in KEEP.held] + [u for u in mine if u not in KEEP.held]   # kept units first (frees their memory)
    try:
        for u in mine:
            t = time.perf_counter()
            if kind == "adjoint":
                kept = u in KEEP.held
                local[u] = (unit_grad_kept(u, g, cnt) if kept else unit_grad(u, Dg, g, cnt),); torch.cuda.synchronize(dev); wall = time.perf_counter() - t
                rep = FA_LAST_REPORT or {}; rf = 0.0 if kept else float(rep.get("forward_seconds", 0.0))
                if not kept: PL.event(step, "recorded_forward", rf, tile=list(u[0]), pol=u[1])
                PL.event(step, "adjoint", wall - rf, tile=list(u[0]), pol=u[1], solver_backward_s=(rep.get("last_backward") or {}).get("seconds"), kept_trace=kept)
            else:
                nk = len(KEEP.held); local[u] = unit_fwd(u, Dg, keep=_KT.ENABLED and kind == "forward"); kept = len(KEEP.held) > nk; torch.cuda.synchronize(dev)
                PL.event(step, "recorded_forward" if kept else kind, time.perf_counter() - t, tile=list(u[0]), pol=u[1],
                         solver_forward_s=(FA_LAST_REPORT or {}).get("forward_seconds"), kept_trace=kept,
                         mem_gib=[round(x / 2**30, 2) for x in (torch.cuda.memory_allocated(dev), torch.cuda.memory_reserved(dev), torch.cuda.mem_get_info(dev)[0])])
            if os.environ.get("FA_EMPTY_CACHE", "1") == "1": torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc(); ok = False
    t = time.perf_counter(); bad = _D.all_reduce(torch.tensor([0.0 if ok else 1.0], device=dev)); PL.event(step, "idle", time.perf_counter() - t, part=kind)
    if float(bad[0]) > 0:
        if RANK == 0: raise RuntimeError(f"[ns] {int(bad[0])} rank(s) failed during {kind} (traceback on that rank's stderr)")
        return None
    t = time.perf_counter(); out = _D.gather_units(UNITS, OWN, local, RANK, dev); torch.cuda.synchronize(dev)
    PL.event(step, "nccl", time.perf_counter() - t, part=f"gather {kind}")
    return out

OP_STOP, OP_FWD, OP_GRAD = 0, 1, 2
def _send(op, code, step):
    _D.broadcast(torch.tensor([op, code, step, 0], dtype=torch.int64, device=dev), 0)
def _bcast_design(Dg):                                                    # code 1 float design, 2 binary (uint8 on the wire); the solves read the design
    buf = (Dg.to(torch.uint8) if Dg.dtype == torch.bool else Dg).to(dev).contiguous(); _D.broadcast(buf, 0); del buf   #  windows from host memory, so no
_HOSTD = {}                                                               #  1.6 GB design grid stays on the GPU through the FDTD phases
def _recv_design(code):
    buf = torch.empty((N10, N10), dtype=torch.uint8 if code == 2 else torch.float32, device=dev); _D.broadcast(buf, 0)
    h = _HOSTD.get(code)
    if h is None: h = _HOSTD[code] = torch.empty((N10, N10), dtype=buf.dtype, pin_memory=True)
    h.copy_(buf); del buf; torch.cuda.empty_cache()
    return h.bool() if code == 2 else h
def serve():                                                              # ranks > 0: solve units on command
    Dg = None
    while True:
        t = time.perf_counter(); cmd = _D.broadcast(torch.zeros(4, dtype=torch.int64, device=dev), 0); op, code, step = (int(v) for v in cmd[:3])
        PL.event(step if op != OP_STOP else None, "idle", time.perf_counter() - t, part="wait for rank 0 (serial phase)")
        if op == OP_STOP: return
        if op == OP_FWD:
            Dg = None; Dg = _recv_design(code); run_phase("binary_evaluation" if code == 2 else "forward", step, Dg)
        elif op == OP_GRAD:
            gr = _D.broadcast(torch.empty((len(JC_NS), len(LAM_NM), NCELL, NCELL, 2), dtype=torch.float32, device=dev), 0)
            cn = _D.broadcast(torch.empty((NCELL, NCELL), dtype=torch.float32, device=dev), 0)
            run_phase("adjoint", step, Dg, torch.view_as_complex(gr), cn); del gr, cn
        PL.telemetry(step, dev)

# ---------------- rank-0 API used by loop_zonec.py (FA_NS) ----------------
FA_LAST_T = {}
def _step(): return int(globals().get("k", -1))
def _assemble(got):
    acc = {k: torch.zeros((len(LAM_NM), NCELL, NCELL), dtype=torch.complex64, device=dev) for k in JC_NS}; cnt = torch.zeros((NCELL, NCELL), device=dev)
    for key in tiles:
        for pol in NS_POLS:
            f, pts = got[(key, pol)]; f = torch.view_as_complex(f.contiguous())
            for kk, fld in zip(_JK[pol], (f[0], f[1])):
                a, c = tile_cells(key, fld, pts); acc[kk] = acc[kk] + a
        cnt = cnt + c
    return {k: _norm(v, cnt) for k, v in acc.items()}, cnt
def _forward(Dg, kind):
    step = _step(); code = 2 if Dg.dtype == torch.bool else 1
    _send(OP_FWD, code, step); _bcast_design(Dg)
    return step, run_phase(kind, step, Dg)
def forward_score():
    """Grey forward of the global Dq: objective I, dI/dT (stacked like JC_NS), cell counts."""
    t0 = time.time(); step, got = _forward(Dq, "forward"); t1 = time.time()
    T, cnt = _assemble(got); del got
    leaf = {k: v.detach().requires_grad_(True) for k, v in T.items()}
    FA_LAST_T.clear(); FA_LAST_T.update({k: v.detach() for k, v in leaf.items()})
    I = fa_objective()(leaf); I.backward(); g = torch.stack([leaf[k].grad for k in JC_NS], 0)
    PL.event(step, "asm", time.time() - t1, part="assembly+objective+dI/dT")
    print(f"[ns] forward: {len(UNITS)} units on {WORLD} ranks {t1 - t0:.0f}s, assembly + objective {time.time() - t1:.1f}s", flush=True)
    return float(I.detach()), g, cnt, None
def region_gradient10(keys, g, cnt):
    """dI/dD on the 10 nm grid: every unit's window gradient through the exact transpose of tile_q, tiles in canonical order."""
    t0 = time.time(); step = _step(); assert list(keys) == tiles
    _send(OP_GRAD, 0, step); tb = time.perf_counter()
    _D.broadcast(torch.view_as_real(g.contiguous()).contiguous(), 0); _D.broadcast(cnt.float().contiguous(), 0)
    PL.event(step, "nccl", time.perf_counter() - tb, part="broadcast dI/dT")
    got = run_phase("adjoint", step, Dq, g, cnt); t1 = time.time()
    G = torch.zeros((N10, N10), dtype=torch.float32, device=dev)
    for key in tiles:
        gw = got.pop((key, NS_POLS[0]))[0]
        for pol in NS_POLS[1:]: gw = gw + got.pop((key, pol))[0]          # x then y, as autograd accumulates a shared leaf
        tile_q_T(gw, *tile_window(*key), G)
    PL.event(step, "asm", time.time() - t1, part="gradient transpose")
    print(f"[ns] adjoint: {len(UNITS)} units on {WORLD} ranks {t1 - t0:.0f}s, transpose {time.time() - t1:.1f}s", flush=True)
    return G
def tmaps(q):
    """Forward of a (binary) design: (pupil Jones maps, cell counts, objective)."""
    global Dq
    Dq = q; t0 = time.time(); step, got = _forward(q, "binary_evaluation" if q.dtype == torch.bool else "forward"); t1 = time.time()
    T, cnt = _assemble(got); del got
    with torch.no_grad(): I = float(fa_objective()(T))
    PL.event(step, "asm", time.time() - t1, part="binary assembly+objective")
    print(f"[ns] binary forward: {len(UNITS)} units on {WORLD} ranks {t1 - t0:.0f}s, objective {I:.6g}", flush=True)
    return T, cnt, I
def view20(Q): return (Fn.avg_pool2d(Q.to(dev).float()[None, None], 2)[0, 0] >= 0.5).cpu().numpy()
def pack_q(Q): return np.packbits(Q.cpu().numpy())
def _jones_expand(T): return T

def load_density(path):
    """Density file -> (N10, N10) float32 in [0, 1] on the host, aperture applied (formats in the module docstring)."""
    z = np.load(path)
    if "density" in z.files: d = torch.tensor(np.asarray(z["density"]).astype(np.float32))
    elif "d10_bits" in z.files:
        sh = tuple(int(v) for v in z["d10_shape"]); d = torch.tensor(np.unpackbits(z["d10_bits"])[:int(np.prod(sh))].reshape(sh).astype(np.float32))
    elif "d10_full" in z.files:                                            # fa-baselines format (2026-09-30): packbits of the full x-major 10 nm grid
        sh = tuple(int(v) for v in z["d10_full_shape"]); d = torch.tensor(np.unpackbits(z["d10_full"])[:int(np.prod(sh))].reshape(sh).astype(np.float32))
    elif "density20" in z.files: d = torch.tensor(np.asarray(z["density20"]).astype(np.float32)).repeat_interleave(2, 0).repeat_interleave(2, 1)
    elif "d10_quad" in z.files:                                            # D4 quadrant (x, y >= 0) -> full plane by the two axis mirrors
        sh = tuple(int(v) for v in z["d10_shape"]); Q = torch.tensor(np.unpackbits(z["d10_quad"])[:int(np.prod(sh))].reshape(sh).astype(np.float32))
        assert sh == (N10 // 2, N10 // 2), f"D4 quadrant {sh} does not match the {N10} px grid"
        k = torch.arange(N10); qi = torch.where(k >= N10 // 2, k - N10 // 2, N10 // 2 - 1 - k); d = Q[qi[:, None], qi[None, :]]
    else: raise ValueError(f"{path}: no density / d10_bits / density20 / d10_quad array")
    assert tuple(d.shape) == (N10, N10), f"{path}: density {tuple(d.shape)}, expected {(N10, N10)} (10 nm grid over the D200 aperture)"
    assert float(d.min()) >= 0 and float(d.max()) <= 1, f"{path}: density outside [0, 1]"
    return d * AP10.cpu()

# ---------------- startup: admission margin, manifest ----------------
_resv = ns_model(tiles[0], NS_POLS[0]).plan(FREQ_T, device="cuda", material_components=3, block_size=32)["gpu_reservation_bytes"]
_free, _total = torch.cuda.mem_get_info(dev)
PL.event(None, "setup", time.time() - _T_SETUP, synchronized=False, reservation_bytes=int(_resv), free_bytes=int(_free), total_bytes=int(_total),
         admission_limit_bytes=int(0.8 * _free), units=sum(1 for v in OWN.values() if v == RANK))
_objname = os.environ.get("FA_OBJECTIVE", "")
if RANK == 0:
    print(f"[ns] {WORLD} ranks ({_BACKEND}); core {TILE} um + 2 x {OVER} um -> {shape[0]} x {shape[1]} x {shape[2]} cells, {len(tiles)} tiles "
          f"({_GRID_MODE} grid), {len(UNITS)} units ({''.join(NS_POLS)}), per rank {[sum(1 for v in OWN.values() if v == r) for r in range(WORLD)]}; "
          f"{int(AP10.sum())} aperture px of 10 nm; tile reservation {_resv / 2**30:.1f} GiB, free {_free / 2**30:.1f} GiB "
          f"(admission limit {0.8 * _free / 2**30:.1f} GiB)", flush=True)
    _geo = dict(diameter_um=2 * R_AP, shape=[int(v) for v in shape], tiles=len(tiles), units=len(UNITS), polarizations=len(NS_POLS), symmetry="none",
                tile_grid=_GRID_MODE, mesh_um=MESH, steps=STEPS, core_um=TILE, overlap_um=OVER, overlap_factor=((TILE + 2 * OVER) / TILE) ** 2, pupil_pitch_um=PITCH,
                pupil_n=NCELL, QPC=QPC, monitor_point_um=PITCH / QPC, trace_storage=ADJ.trace_storage,
                reconstruction_layers=int(pil.sum().item()) if os.environ.get("FF_B200_KERNELS") == "1" else None, wavelengths_nm=LAM_NM,
                source_wavelength_um=_SRC_BAND, layer=dict(n_design=N_SIN, n_substrate=N_SIO2, height_um=H), time_steps=STEPS, latent="Cartesian 10 nm, whole aperture", independent_cells=int(AP10.sum()),
                tile_list=[list(t) for t in tiles], tile_reservation_bytes=int(_resv))
    _objinfo = dict(name=_objname)
    if _objname: _objinfo.update(__import__("fa_objective_loader").info(fa_objective()))
    _NR.write_manifest(OUT, geometry=_geo, objective=_objinfo, source_dir=_NS_HERE, world=WORLD,
                       extra=dict(driver="ns_run.py", mode="eval-only" if EVAL_FILE else "optimize", eval_file=EVAL_FILE))

# ---------------- run ----------------
if RANK != 0:                                                              # only rank 0 needs the aperture grid (loop, density files)
    del VALID, AP10; torch.cuda.empty_cache()
if RANK == 0:
    try:
        if EVAL_FILE:                                                     # forward only: pupil Jones maps, raw exit fields, objective and metrics
            ed = OUT / f"eval_{Path(EVAL_FILE).stem}"; ed.mkdir(parents=True, exist_ok=True); t0 = time.time()
            Dg = load_density(EVAL_FILE); binary = bool(((Dg == 0) | (Dg == 1)).all())      # host tensor: tile_q reads its windows
            if binary: Dg = Dg > 0.5
            k = -1; step, got = _forward(Dg, "binary_evaluation" if binary else "forward")
            raw = {f"{i:04d}_{pol}": torch.view_as_complex(got[(key, pol)][0].contiguous()).cpu().numpy() for i, key in enumerate(tiles) for pol in NS_POLS}
            pts0 = got[(tiles[0], NS_POLS[0])][1].cpu().numpy()
            T, cnt = _assemble(got); del got
            obj = fa_objective() if _objname else None
            with torch.no_grad(): value = float(obj(T)) if obj is not None else float("nan")
            met, met_arr = _NR.metrics(obj, T, arrays=True) if obj is not None else (None, {})
            if met_arr: np.savez(ed / "metrics_arrays.npz", **{k: np.asarray(v) for k, v in met_arr.items()})
            np.savez(ed / "pupil_jones.npz", wavelength_nm=np.array(LAM_NM), cnt=cnt.cpu().numpy(), pitch_um=PITCH, E_ref=E_ref.cpu().numpy(),
                     jones_scale=JONES_SCALE.reshape(-1).cpu().numpy(), **{f"T_{kk}": v.cpu().numpy() for kk, v in T.items()})
            if os.environ.get("FA_EVAL_RAW", "1") == "1":
                np.savez(ed / "exit_fields_raw.npz", tiles=np.array(tiles), points_um_local=pts0, components=np.array(["Ex", "Ey"]), wavelength_nm=np.array(LAM_NM), **raw)
            rec = dict(density_file=str(EVAL_FILE), density_sha256=_NR.sha256_file(EVAL_FILE), binary=binary, fill=float(Dg.float()[AP10.cpu()].mean()),
                       objective=_objname, objective_value=value, metrics=met, polarizations="".join(NS_POLS), wavelengths_nm=LAM_NM, tiles=len(tiles),
                       core_um=TILE, overlap_um=OVER, seconds=time.time() - t0, source_sha256=json.loads((OUT / "run_manifest.json").read_text())["source_sha256"])
            _NR.write_json(ed / "metrics.json", rec); PL.merge_ranks()
            print(f"[ns] eval {EVAL_FILE}: objective {value:.6g}, fill {rec['fill']:.3f}, {rec['seconds']:.0f} s -> {ed}", flush=True)
            print("FF_DONE (eval only)", flush=True)
        elif TRACE_BENCH:
            u = UNITS[0]; i0, j0 = tile_window(*u[0]); ka, kb = _JK[u[1]]; rows = []
            Dg = (torch.rand((N10, N10), generator=torch.Generator().manual_seed(1)) * AP10.cpu())
            gcot = torch.randn((len(JC_NS), len(LAM_NM), NCELL, NCELL), dtype=torch.complex64, device=dev); cntb = torch.ones((NCELL, NCELL), device=dev)
            for rep_i in range(int(os.environ.get("FA_TRACE_BENCH_REPS", "3"))):
                for label, opts in (("device", None), ("keep_host", ADJ_KEEP)):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    with torch.no_grad(): leaf = tile_q(Dg, i0, j0)
                    leaf.requires_grad_(True); fa, fb, pts = ns_solve(leaf, u[0], u[1], opts); torch.cuda.synchronize(); t1 = time.perf_counter()
                    oa = _norm(tile_cells(u[0], fa, pts)[0], cntb); ob = _norm(tile_cells(u[0], fb, pts)[0], cntb)
                    torch.autograd.backward([oa, ob], [gcot[JC_NS.index(ka)], gcot[JC_NS.index(kb)]]); torch.cuda.synchronize(); t2 = time.perf_counter()
                    rp = FA_LAST_REPORT or {}; g1 = leaf.grad.detach().clone()
                    rows.append(dict(rep=rep_i, mode=label, forward_s=t1 - t0, backward_s=t2 - t1, solver_forward_s=rp.get("forward_seconds"),
                                     solver_backward_s=(rp.get("last_backward") or {}).get("seconds"), grad_sum=float(g1.double().sum())))
                    print("[trace-bench]", json.dumps(rows[-1]), flush=True); del fa, fb, oa, ob, leaf; torch.cuda.empty_cache()
            _NR.write_json(OUT / "trace_bench.json", dict(rows=rows, shape=[int(v) for v in shape], pinned_pool=dict(count=_KT.POOL.count, bytes=_KT.POOL.bytes)))
            print("FF_DONE (trace bench)", flush=True)
        elif FD_CHECK:                                                     # tile_q / tile_q_T dot-product test, then a directional FD of I(D) against the adjoint
            gen = torch.Generator().manual_seed(int(os.environ.get("FA_FD_SEED", "20260930"))); rec = {}
            def _smooth(n, s):
                x = torch.randn((1, 1, n, n), generator=gen).to(dev); r = int(3 * s); t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
                w = torch.exp(-t ** 2 / (2 * s * s)); w = w / w.sum()
                x = Fn.conv2d(Fn.pad(x, (r, r, 0, 0)), w.view(1, 1, 1, -1)); x = Fn.conv2d(Fn.pad(x, (0, 0, r, r)), w.view(1, 1, -1, 1)); return x[0, 0]
            dots = []
            for key in tiles[:3]:
                i0, j0 = tile_window(*key); Dr = torch.rand((N10, N10), generator=gen).to(dev); gr = torch.randn(shape[:2] + (3,), generator=gen).to(dev)
                Gt = torch.zeros((N10, N10), device=dev); tile_q_T(gr, i0, j0, Gt)
                a_ = float((tile_q(Dr, i0, j0)[..., :3].double() * gr.double()).sum()); b_ = float((Dr.double() * Gt.double()).sum()); dots.append(abs(a_ - b_) / (abs(a_) + abs(b_)))
            rec["transpose_dot_rel"] = dots; print(f"[fd] tile_q / tile_q_T dot-product relative mismatch {dots}", flush=True)
            s0 = _smooth(N10, float(os.environ.get("FA_FD_SIGMA_PX", "8"))); s0 = s0 / s0.abs().max()
            D0 = ((0.5 + 0.3 * s0).clamp(0.05, 0.95) * AP10).float()
            v = _smooth(N10, float(os.environ.get("FA_FD_SIGMA_PX", "8"))); v = (v / v.abs().max() * AP10).float()
            k = -1; Dq = D0; I0, g, cnt, _ = forward_score(); G = region_gradient10(tiles, g, cnt)
            adj = float((G.double() * v.double()).sum()); rec.update(I0=I0, adjoint=adj, fd={})
            for delta in [float(x) for x in os.environ.get("FA_FD_DELTAS", "0.02").split(",")]:
                Ip = tmaps((D0 + delta * v).clamp(0, 1))[2]; Im = tmaps((D0 - delta * v).clamp(0, 1))[2]; fd = (Ip - Im) / (2 * delta)
                rel = abs(fd - adj) / (abs(fd) + abs(adj) + 1e-30); rec["fd"][str(delta)] = dict(I_plus=Ip, I_minus=Im, fd=fd, relative_error=rel)
                print(f"[fd] delta {delta}: FD {fd:.9e}  adjoint {adj:.9e}  relative error {rel:.3e}", flush=True)
            rec.update(pols="".join(NS_POLS), wavelengths_nm=LAM_NM, tiles=len(tiles), core_um=TILE, overlap_um=OVER, objective=_objname,
                       b200_kernels=os.environ.get("FF_B200_KERNELS"), trace=ADJ.trace_storage)
            rec.update(peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(dev), peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(dev),
                       tile_reservation_bytes=int(_resv))
            _NR.write_json(OUT / "fd_check.json", rec); PL.telemetry(-1, dev); PL.merge_ranks(); print("FF_DONE (fd check)", flush=True)
        else:
            FA_NS = True; FA_OFFLOAD_NAMES = ("theta", "opt_m", "opt_v", "D", "Dq")   # FA_OPT_OFFLOAD=1: also the design grid (read from host by tile_q)
            exec(compile(open(_NS_HERE / "loop_zonec.py", encoding="utf-8").read(), "loop_zonec.py", "exec"), globals())
    finally:
        _send(OP_STOP, 0, -1)
else:
    serve()
if WORLD > 1: dist.destroy_process_group()
