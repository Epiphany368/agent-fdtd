"""Multi-GPU launcher for the zone density topology optimization (FF_ALGO=zonec): same numbers as `python ff_opt.py`, tile FDTD on N GPUs.

Rank 0 runs the unchanged single-GPU program: ff_opt.py's head (tile solver, patches) and loop_zonec.py (filter, projection, Adam,
adaptive beta, saving, resume). Only the three calls that run tile FDTD are replaced, and each replacement ends in the ORIGINAL
function, so the assembly, the objective and the gradient accumulation are the single-GPU code in the single-GPU tile order:
  forward_score() / tmaps(q):  rank 0 broadcasts the design Dq; every rank solves its share of the (tile, polarization) units;
                               rank 0 receives every unit's exit fields over NCCL in the canonical unit order and runs the original
                               forward_score / tmaps, whose refresh() is answered from those fields (exact float32 copies).
  region_gradient10(keys,..):  rank 0 broadcasts dI/dT (g_oct) and the cell counts; every rank runs the per-polarization half of the
                               original tile_grad() on its units; rank 0 receives the unit gradients in canonical order, adds the two
                               polarizations of a tile (x then y, as autograd accumulated them) and runs the original region_gradient10.
2026-09-30 (FA package): work unit = one (tile, polarization) solve instead of one tile with both polarizations (balanced queues:
32 units of the D200 core-19 layout are 4 per rank on 8 GPUs, 8 on 4), fields and gradients move over NCCL instead of nf_cache /
_mp files (no per-step disk round trip), one solver model per local geometry, phase JSONL per rank (fa_runtime.py).
Units are split contiguously, remainder to the LAST ranks, so rank 0 (objective, optimizer, saving) has the fewest.
launch: python -m torch.distributed.run --standalone --nproc_per_node=<N> zonec_multi.py <widths.npy> <out_dir> [steps] [k]
env:    FF_NCCL_TIMEOUT_S (7200) collective timeout; FF_DIST_BACKEND (nccl on CUDA, else gloo); FF_HEAD_SRC (ff_opt.py);
        FF_MULTI_VERBOSE=1 lets the other ranks print.
"""
import os, sys, time, json, math, traceback
from datetime import timedelta
from pathlib import Path
import torch, torch.distributed as dist

_MP_HERE = Path(os.path.abspath(__file__)).parent
if os.environ.get("FF_ALGO") != "zonec": raise SystemExit("zonec_multi.py runs FF_ALGO=zonec only (use ff_opt.py for the other loops)")
if os.environ.get("EZC_HTDIAG"): raise SystemExit("EZC_HTDIAG is a single-GPU diagnostic: run ff_opt.py")
if os.environ.get("FF_L9_DUALPOL", "0") != "1": raise SystemExit("zonec needs FF_L9_DUALPOL=1 (tmaps reads the Jones tile fields)")
_MP_BACKEND = os.environ.get("FF_DIST_BACKEND", "nccl" if torch.cuda.is_available() and torch.cuda.device_count() > 0 and sys.platform != "win32" else "gloo")
_mp_local = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    _mp_local %= torch.cuda.device_count(); torch.cuda.set_device(_mp_local)
if int(os.environ.get("WORLD_SIZE", "1")) > 1:                            # one process: no process group (all exchanges are local)
    dist.init_process_group(_MP_BACKEND, timeout=timedelta(seconds=int(os.environ.get("FF_NCCL_TIMEOUT_S", "7200"))),
                            **({"device_id": torch.device("cuda", _mp_local)} if _MP_BACKEND == "nccl" else {}))
MP_RANK, MP_WORLD = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
os.environ["LOCAL_RANK"] = str(_mp_local); os.environ["FF_MULTI_RANK"] = str(MP_RANK)   # ff_opt.py builds the objective only where it is evaluated
if MP_RANK != 0 and os.environ.get("FF_MULTI_VERBOSE", "0") != "1":
    def print(*a, **k): pass                                               # exec'd code resolves print() here; tracebacks still reach stderr
_T_SETUP = time.time()

# ---------------- ff_opt.py head, in three parts ----------------
_MP_SRC = open(os.environ.get("FF_HEAD_SRC", str(_MP_HERE / "ff_opt.py")), encoding="utf-8").read()
_MP_MARK = ["# ---------------- reference: homogeneous SiO2", "# ---------------- design parameters (octant)", "# ---------------- optimization loop ----------------"]
for _m in _MP_MARK: assert _MP_SRC.count(_m) == 1, f"ff_opt.py anchor not found exactly once: {_m!r}"
_i1, _i2, _i3 = (_MP_SRC.index(_m) for _m in _MP_MARK)
def _mp_exec(src, start, name):                                            # keep ff_opt.py line numbers in tracebacks
    exec(compile("\n" * _MP_SRC.count("\n", 0, start) + src, name, "exec"), globals())
# A: tile project, solver options. B: the homogeneous reference (reference.npz). Rank 0 goes first, so run-dir files (geometry.json,
# reference.npz) have one writer; the other ranks then load the reference instead of re-solving it.
if MP_RANK == 0:
    _mp_exec(_MP_SRC[:_i1], 0, "ff_opt.py[head A]"); _t_ref = time.time(); _mp_exec(_MP_SRC[_i1:_i2], _i1, "ff_opt.py[head B]"); _t_ref = time.time() - _t_ref
if MP_WORLD > 1: dist.barrier()
if MP_RANK != 0:
    _mp_exec(_MP_SRC[:_i1], 0, "ff_opt.py[head A]")
    _t_ref = time.time(); _mp_exec(_MP_SRC[_i1:_i2], _i1, "ff_opt.py[head B]"); _t_ref = time.time() - _t_ref
_mp_exec(_MP_SRC[_i2:_i3], _i2, "ff_opt.py[head C]")                      # design parameters, tiles, cache, FF_FAST / b200 patches

# ---------------- loop_zonec.py: prelude on every rank (helpers, zones, state), the loop on rank 0 ----------------
_MP_ZSRC = open(_MP_HERE / "loop_zonec.py", encoding="utf-8").read()
_MP_ANCHOR = "\nfor k in range(k0, EZC_OUTER + 1):\n"
assert _MP_ZSRC.count(_MP_ANCHOR) == 1, "loop_zonec.py: main loop anchor not found exactly once"
_i4 = _MP_ZSRC.index(_MP_ANCHOR) + 1
exec(compile(_MP_ZSRC[:_i4], "loop_zonec.py", "exec"), globals())
assert DUALPOL and "model_y_for" in globals(), "the dual-polarization tile solver of loop_levelset10.py is not active"
_mp_forward_score_local, _mp_tmaps_local, _mp_region_gradient10_local = forward_score, tmaps, region_gradient10
import fa_runtime as _NR, fa_dist as _D
_MP_PL = _NR.phase_log(OUT, MP_RANK)
_MP_PL.event(None, "reference", _t_ref, synchronized=True)

MP_STOP, MP_FWD, MP_GRAD = 0, 1, 2
POLS = ("x", "y")
_mp_tile_index = {k: i for i, k in enumerate(tiles)}
_MP_FIELDS = {}                                                            # rank 0: tile key -> (Jones field dict, points) of the current design

def _mp_units(keys): return [(key, pol) for key in keys for pol in POLS]   # canonical order: tiles in list order, x before y

def _mp_owners(units):
    """Contiguous split of the units over the ranks; the remainder goes to the LAST ranks, so rank 0 has the fewest."""
    q, m = divmod(len(units), MP_WORLD); own, i = {}, 0
    for r in range(MP_WORLD):
        n = q + (1 if r >= MP_WORLD - m else 0)
        for u in units[i:i + n]: own[u] = r
        i += n
    return own

def _mp_step(): return int(globals().get("k", -1))                         # loop_zonec's step counter (rank 0)

def _mp_all_ok(ok, what, step):
    """Every rank reports; the wait here is the rank's idle time before the exchange."""
    t = time.perf_counter(); bad = torch.tensor([0.0 if ok else 1.0], device=dev); _D.all_reduce(bad)
    _MP_PL.event(step, "idle", time.perf_counter() - t, part=what)
    if float(bad[0]) > 0 and MP_RANK == 0:
        raise RuntimeError(f"[multi] {int(bad[0])} rank(s) failed during {what} (traceback on that rank's stderr)")
    return float(bad[0]) == 0

def _mp_send(op, keys, code=0, step=-1):
    cmd = torch.tensor([op, code, len(keys), step], dtype=torch.int64, device=dev); _D.broadcast(cmd, 0)
    if keys: _D.broadcast(torch.tensor([_mp_tile_index[k] for k in keys], dtype=torch.int64, device=dev), 0)

def _mp_bcast_design():                                                   # rank 0 -> all: the design the solver reads (global Dq)
    assert Dq.dtype in (torch.float32, torch.bool) and tuple(Dq.shape) == (H10, H10), f"unexpected design {Dq.dtype} {tuple(Dq.shape)}"
    _D.broadcast(Dq.to(torch.uint8).contiguous() if Dq.dtype == torch.bool else Dq.contiguous(), 0)

def _mp_recv_design(code):
    buf = torch.empty((H10, H10), dtype=torch.uint8 if code == 2 else torch.float32, device=dev); _D.broadcast(buf, 0)
    return buf.bool() if code == 2 else buf

import fa_keep_trace as _KT                                              # FA_KEEP_TRACE=1: one recorded forward per unit and step (fa_keep_trace.py)
if _KT.ENABLED: _KT.install()
_KEEP = _KT.Keeper(dev)

def _keep_model(key, pol):                                                 # the unit's model with the host-trace options of the kept path
    global project
    if pol == "x": p, counts = tile_project(*key)
    else:
        p0 = project; project = project_y
        try: p, counts = tile_project(*key)
        finally: project = p0
    return shared_model(p, counts, 3, ADJ_KEEP)

def _mp_fwd_unit(unit, keep=False):
    key, pol = unit; i0, j0 = tile_window(*key)
    if keep:                                                              # recorded forward with a live graph; the adjoint phase runs only the backward
        global FA_LAST_REPORT
        m = _keep_model(key, pol)
        if _KEEP.admit(_KEEP.reservation(m, FREQ_T), math.prod(shape)):
            before = torch.cuda.memory_allocated(dev)
            with torch.no_grad(): leaf = rho_tile_of(theta, i0, j0).clone()
            leaf.requires_grad_(True)
            det = m(eps_of(leaf), FREQ_T.to(dev), fixed_epsilon=eps_background(), block_size=32)["xy"]; FA_LAST_REPORT = det.report
            lo = torch.as_tensor(lam_order.copy(), device=dev)
            a, b, pts = det.fields[:, :, 0][lo], det.fields[:, :, 1][lo], torch.as_tensor(det.points_um, device=dev, dtype=torch.float32)
            del det; _KEEP.store(unit, leaf, a, b, pts, before)
            return torch.view_as_real(torch.stack([a.detach(), b.detach()])).contiguous(), pts.contiguous()
    with torch.no_grad():
        q = rho_tile_of(theta, i0, j0); a, b, pts = _solve_pol(q, key, pol)
    return torch.view_as_real(torch.stack([a, b])).contiguous(), pts.contiguous()

def _mp_grad_kept(unit, g_oct, cnt):                                      # backward of a kept unit: same tile_grad arithmetic on the stored graph
    key, pol = unit; ka, kb = ("xx", "yx") if pol == "x" else ("xy", "yy")
    leaf, fa, fb, pts = _KEEP.pop(unit)
    oa = _norm(tile_cells(key, fa, pts)[0], cnt); ob = _norm(tile_cells(key, fb, pts)[0], cnt)
    torch.autograd.backward([oa, ob], [g_oct[JC.index(ka)], g_oct[JC.index(kb)]])
    g = leaf.grad.detach()[..., :3].contiguous()
    del fa, fb, oa, ob, leaf
    return g

def _mp_grad_unit(unit, g_oct, cnt):                                      # one polarization of the original tile_grad (same code)
    key, pol = unit; i0, j0 = tile_window(*key); ka, kb = ("xx", "yx") if pol == "x" else ("xy", "yy")
    with torch.no_grad(): leaf = rho_tile_of(theta, i0, j0).clone()
    leaf.requires_grad_(True)
    fa, fb, pts = _solve_pol(leaf, key, pol)
    oa = _norm(tile_cells(key, fa, pts)[0], cnt); ob = _norm(tile_cells(key, fb, pts)[0], cnt)
    torch.autograd.backward([oa, ob], [g_oct[JC.index(ka)], g_oct[JC.index(kb)]])
    g = leaf.grad.detach()[..., :3].contiguous()                           # channels 3, 4 (interface normals) are detached in eps_of: zero
    del fa, fb, oa, ob, leaf
    return g

def _mp_phase(kind, keys, step, g_oct=None, cnt=None):
    """Solve this rank's units (kind 'forward' | 'binary_evaluation' | 'adjoint'); rank 0 returns every unit's tensors."""
    units = _mp_units(keys); own = _mp_owners(units); mine = [u for u in units if own[u] == MP_RANK]; local = {}; ok = True
    if kind == "forward": _KEEP.clear()                                    # a new design: graphs of an earlier step are stale
    if kind == "adjoint": mine = [u for u in mine if u in _KEEP.held] + [u for u in mine if u not in _KEEP.held]   # kept units first (frees their memory)
    try:
        for u in mine:
            t = time.perf_counter()
            if kind == "adjoint":
                kept = u in _KEEP.held
                local[u] = (_mp_grad_kept(u, g_oct, cnt) if kept else _mp_grad_unit(u, g_oct, cnt),)
                torch.cuda.synchronize(dev); wall = time.perf_counter() - t; rep = globals().get("FA_LAST_REPORT") or {}
                rf = 0.0 if kept else float(rep.get("forward_seconds", 0.0)); bw = (rep.get("last_backward") or {}).get("seconds")
                _MP_PL.event(step, "recorded_forward", rf, tile=list(u[0]), pol=u[1])
                _MP_PL.event(step, "adjoint", wall - rf, tile=list(u[0]), pol=u[1], solver_backward_s=bw, kept_trace=kept)
            else:
                nk = len(_KEEP.held); local[u] = _mp_fwd_unit(u, keep=_KT.ENABLED and kind == "forward"); kept = len(_KEEP.held) > nk
                torch.cuda.synchronize(dev); _MP_PL.event(step, "recorded_forward" if kept else kind, time.perf_counter() - t, tile=list(u[0]), pol=u[1],
                                                          solver_forward_s=(globals().get("FA_LAST_REPORT") or {}).get("forward_seconds"), kept_trace=kept)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc(); ok = False
    if not _mp_all_ok(ok, kind, step): return None
    t = time.perf_counter(); out = _D.gather_units(units, own, local, MP_RANK, dev); torch.cuda.synchronize(dev)
    _MP_PL.event(step, "nccl", time.perf_counter() - t, part=f"gather {kind}")
    return out

def _mp_tile_fields(got, keys):                                            # rank 0: unit tensors -> the (Jones dict, points) cache entry of each tile
    out = {}
    for key in keys:
        fx, px = got[(key, "x")]; fy, py = got[(key, "y")]
        assert torch.equal(px, py), "x and y models of a tile disagree on the monitor points"
        fx = torch.view_as_complex(fx.contiguous()); fy = torch.view_as_complex(fy.contiguous())
        out[key] = ({"xx": fx[0], "yx": fx[1], "xy": fy[0], "yy": fy[1]}, px)
    return out

def refresh(keys):                                                         # rank 0: the solved fields of the current design (no disk round trip)
    for key in keys: cache[key] = _MP_FIELDS[key]

def _mp_forward(kind):
    step = _mp_step(); code = 2 if Dq.dtype == torch.bool else 1
    _mp_send(MP_FWD, tiles, code, step); _mp_bcast_design()
    got = _mp_phase(kind, tiles, step); _MP_FIELDS.clear(); _MP_FIELDS.update(_mp_tile_fields(got, tiles))
    return step

def forward_score():                                                      # rank 0: parallel unit forwards, then the original function
    t0 = time.time(); step = _mp_forward("forward"); t1 = time.time()
    out = _mp_forward_score_local(); _MP_PL.event(step, "asm", time.time() - t1, part="assembly+objective+dI/dT")
    print(f"[multi] forward: {len(tiles)} tiles x 2 pol on {MP_WORLD} ranks {t1 - t0:.0f}s, assembly + objective {time.time() - t1:.1f}s", flush=True)
    return out

def tmaps(q):                                                             # rank 0: binary evaluation, same pattern
    global Dq
    Dq = q; t0 = time.time(); step = _mp_forward("binary_evaluation"); t1 = time.time()
    out = _mp_tmaps_local(q); _MP_PL.event(step, "asm", time.time() - t1, part="binary assembly+objective")
    print(f"[multi] binary forward: {len(tiles)} tiles x 2 pol on {MP_WORLD} ranks {t1 - t0:.0f}s, assembly + objective {time.time() - t1:.1f}s", flush=True)
    return out

def region_gradient10(keys, g_oct, cnt):                                  # rank 0: parallel unit adjoints, then the original accumulation
    global tile_grad
    t0 = time.time(); keys = list(keys); step = _mp_step()
    assert g_oct.dtype == torch.complex64 and tuple(g_oct.shape) == (len(JC), len(LAM_NM), NCELL, NCELL)
    _mp_send(MP_GRAD, keys, 0, step)
    tb = time.perf_counter(); _D.broadcast(torch.view_as_real(g_oct.contiguous()).contiguous(), 0); _D.broadcast(cnt.float().contiguous(), 0)
    _MP_PL.event(step, "nccl", time.perf_counter() - tb, part="broadcast dI/dT")
    got = _mp_phase("adjoint", keys, step, g_oct, cnt); t1 = time.time()
    per = {}
    for key in keys:                                                      # x then y: the order in which autograd accumulated leaf.grad
        g = got[(key, "x")][0] + got[(key, "y")][0]
        per[key] = torch.cat([g, torch.zeros(g.shape[:2] + (2,), dtype=g.dtype, device=g.device)], -1)
    saved = tile_grad; tile_grad = lambda key, g, c: per.pop(key)
    try: G = _mp_region_gradient10_local(keys, g_oct, cnt)
    finally: tile_grad = saved
    _MP_PL.event(step, "asm", time.time() - t1, part="gradient accumulation")
    print(f"[multi] adjoint: {len(keys)} tiles x 2 pol on {MP_WORLD} ranks {t1 - t0:.0f}s, accumulation {time.time() - t1:.1f}s", flush=True)
    return G

def _mp_serve():                                                          # ranks > 0
    global Dq
    while True:
        t = time.perf_counter(); cmd = torch.zeros(4, dtype=torch.int64, device=dev); _D.broadcast(cmd, 0); op, code, n, step = (int(v) for v in cmd)
        _MP_PL.event(step if op != MP_STOP else None, "idle", time.perf_counter() - t, part="wait for rank 0 (serial phase)")
        if op == MP_STOP: return
        idx = torch.zeros(n, dtype=torch.int64, device=dev); _D.broadcast(idx, 0); keys = [tiles[i] for i in idx.tolist()]
        if op == MP_FWD:
            Dq = _mp_recv_design(code); _mp_phase("binary_evaluation" if code == 2 else "forward", keys, step)
        elif op == MP_GRAD:
            gr = torch.empty((len(JC), len(LAM_NM), NCELL, NCELL, 2), dtype=torch.float32, device=dev); _D.broadcast(gr, 0)
            cn = torch.empty((NCELL, NCELL), dtype=torch.float32, device=dev); _D.broadcast(cn, 0)
            _mp_phase("adjoint", keys, step, torch.view_as_complex(gr), cn); del gr, cn
        else:
            raise RuntimeError(f"unknown command {op}")
        _MP_PL.telemetry(step, dev)

# ---------------- startup: admission margin of this rank's tile model, manifest ----------------
_mp_units_all = _mp_units(tiles); _own = _mp_owners(_mp_units_all)
_resv = model_for(tiles[0]).plan(FREQ_T, device="cuda", material_components=3, block_size=32)["gpu_reservation_bytes"]
_free, _total = torch.cuda.mem_get_info(dev)
_MP_PL.event(None, "setup", time.time() - _T_SETUP, synchronized=False, reservation_bytes=int(_resv), free_bytes=int(_free), total_bytes=int(_total),
             admission_limit_bytes=int(0.8 * _free), units=sum(1 for v in _own.values() if v == MP_RANK))
if MP_RANK == 0:
    print(f"[multi] {MP_WORLD} ranks ({_MP_BACKEND}); {len(tiles)} tiles, {len(_mp_units_all)} (tile, pol) units, per rank "
          f"{[sum(1 for v in _own.values() if v == r) for r in range(MP_WORLD)]}; tile reservation {_resv / 2**30:.1f} GiB, "
          f"rank-0 free {_free / 2**30:.1f} GiB (admission limit {0.8 * _free / 2**30:.1f} GiB); collective timeout {os.environ.get('FF_NCCL_TIMEOUT_S', '7200')} s", flush=True)
    _geo = dict(diameter_um=2 * R_AP, shape=[int(v) for v in shape], tiles=len(tiles), units=len(_mp_units_all), polarizations=2, symmetry="D4 octant tiles",
                mesh_um=MESH, steps=STEPS, core_um=TILE, overlap_um=OVER, overlap_factor=((TILE + 2 * OVER) / TILE) ** 2, pupil_pitch_um=PITCH, pupil_n=NCELL,
                QPC=QPC, monitor_point_um=PITCH / QPC, trace_storage=ADJ.trace_storage, reconstruction_layers=int(pil.sum().item()) if os.environ.get("FF_B200_KERNELS") == "1" else None,
                wavelengths_nm=LAM_NM, source_wavelength_um=_SRC_BAND, layer=dict(n_design=N_SIN, n_substrate=N_SIO2, height_um=H), time_steps=STEPS, nominal_NA=R_AP / float(os.environ.get("FF_FOCAL_UM", LENS["F_UM"] if LENS else 2 * R_AP / 0.6)),
                focal_um=float(os.environ.get("FF_FOCAL_UM", LENS["F_UM"] if LENS else 2 * R_AP / 0.6)), tile_list=[list(t) for t in tiles],
                latent="polar octant 10 nm" if EZ_POLAR else "zones", independent_cells=int(VALID.sum()), tile_reservation_bytes=int(_resv))
    _objinfo = dict(name=FA_OBJECTIVE or os.environ.get("FF_NEW_OBJECTIVE"))
    if FA_OBJECTIVE: _objinfo.update(__import__("fa_objective_loader").info(fa_objective()))
    _NR.write_manifest(OUT, geometry=_geo, objective=_objinfo, source_dir=_MP_HERE, world=MP_WORLD, extra=dict(driver="zonec_multi.py"))
    try:
        exec(compile("\n" * _MP_ZSRC.count("\n", 0, _i4) + _MP_ZSRC[_i4:], "loop_zonec.py", "exec"), globals())
    finally:
        _mp_send(MP_STOP, [])                                             # also after an exception: release the solver ranks
else:
    _mp_serve()
if MP_WORLD > 1: dist.destroy_process_group()
