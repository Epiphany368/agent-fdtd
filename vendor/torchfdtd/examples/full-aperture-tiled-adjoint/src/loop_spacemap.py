# ---------------- space mapping on a random-texture carrier (FF_ALGO=spacemap) ----------------
# I_tar depends on the design only through the 0.29 um pupil-cell Jones maps T (forward_score builds them; dI/dT is free).
# The structure -> T map is local but strongly nonlinear in pixel space (full sign steps: measured/predicted 0.04-0.14), so the
# large moves are taken in T space with a cheap surrogate and FDTD only checks and corrects:
#   design  = fab_repair( g > tau(x) ),  g a fixed Gaussian random field (corr. SM_ELL nm), tau from a per-cell nominal fill f
#   library = one FDTD forward of a composite with one nominal fill per tile -> mean isotropic t_bar(f, lambda)
#   k = 0   : f_0 = argmax I( t_bar(f) )                                  (surrogate only, no FDTD)
#   k >= 1  : d_k = T_fdtd(f_k) - t_bar(f_k)  (per cell and Jones element); f_{k+1} = argmax I( t_bar(f) + d_k ), |f - f_k| <= Delta
#             realised, repaired, one FDTD forward; accepted if I_tar rises; Delta x1.5 / x0.5 by the measured / predicted ratio
# Posts (low fill) and holes (high fill) follow from the fill map; the texture itself stays random.
# Every design passed to the solver is fab-legal (two monotone repairs of random_seed.py v2, per-site minimal edit, torch/GPU).
_src = open(Path(os.path.abspath(__file__)).with_name("loop_levelset10.py"), encoding="utf-8").read().split("\nwhile step < N_STEPS:")[0]
exec(compile(_src, "loop_levelset10.py(helpers)", "exec"))
import scipy.ndimage as ndi
SM_ELL = float(os.environ.get("SM_ELL", "120")); SM_SEED = int(os.environ.get("SM_SEED", "20260925"))
SM_F0, SM_F1, SM_NLIB = float(os.environ.get("SM_F0", "0.10")), float(os.environ.get("SM_F1", "0.90")), len(tiles)
SM_ITERS = int(os.environ.get("SM_ITERS", "300")); SM_LR = float(os.environ.get("SM_LR", "0.05")); SM_OUTER = int(os.environ.get("SM_OUTER", "8"))
SM_DELTA0 = float(os.environ.get("SM_DELTA0", "0.15")); SM_MARGIN = float(os.environ.get("SM_MARGIN_UM", "1.2"))
print(f"[spacemap] carrier corr {SM_ELL} nm seed {SM_SEED}; library {SM_NLIB} fills {SM_F0}-{SM_F1}; surrogate {SM_ITERS} Adam its lr {SM_LR}; "
      f"outer {SM_OUTER}, Delta0 {SM_DELTA0}; margin {SM_MARGIN} um", flush=True)
assert DUALPOL, "spacemap needs FF_L9_DUALPOL=1 (full Jones maps)"

# ---- carrier and fill <-> threshold ----
if os.environ.get("SM_NO_CARRIER", "0") != "1":                          # loop_elem.py reuses the helpers below without the carrier
    t0 = time.time()
    _rng = np.random.default_rng(SM_SEED)
    G = ndi.gaussian_filter(_rng.standard_normal((H10, H10)).astype(np.float32), SM_ELL / 20.0, mode="reflect")
    G = np.where(np.arange(H10)[:, None] >= np.arange(H10)[None, :], G, G.T); del _rng
    AP_np = AP10.cpu().numpy()
    QLEV = np.linspace(0.0, 1.0, 2001); QTAU = np.quantile(G[AP_np], 1.0 - QLEV)      # nominal fill f -> threshold tau (P(g > tau) = f)
    Gt = torch.tensor(G, device=dev); del G
    print(f"[spacemap] carrier ready in {time.time() - t0:.0f} s", flush=True)
def tau_of_f(f): return torch.tensor(np.interp(f.detach().cpu().numpy().ravel(), QLEV, QTAU).reshape(f.shape), device=dev, dtype=torch.float32)
CELL_Q = NCELL // 2                                                        # quadrant cells: global cell index 360..719
def fill_px(fq):                                                          # (360,360) quadrant-cell nominal fill -> (H10,H10) threshold, bilinear on cell centres
    tq = tau_of_f(fq)
    u = (torch.arange(H10, device=dev, dtype=torch.float32) + 0.5) * (MESH10 / PITCH) - 0.5
    i0 = u.floor().clamp(0, CELL_Q - 1).long(); i1 = (i0 + 1).clamp(max=CELL_Q - 1); w = (u - i0.float()).clamp(0, 1)
    rows = tq[i0] * (1 - w)[:, None] + tq[i1] * w[:, None]                  # (H10, 360)
    return rows[:, i0] * (1 - w)[None, :] + rows[:, i1] * w[None, :]

# ---- fab repair (torch port of seed_library/fab_repair.py; D4 image via d4q, exact-anchor opening _open_disk) ----
P_ = 16
def _mpad(q): return d4q(q, H10 - P_, N10 + P_, H10 - P_, N10 + P_)
def _crop(b): return b[P_:P_ + H10, P_:P_ + H10]
def _thin(q):
    X = _mpad(q); xf = X.float()
    return _crop(X & ~(_open_disk(xf, L_MIN) > 0.5)) & AP10, _crop(~X & ~(_open_disk(1 - xf, L_GAP) > 0.5)) & AP10
def _dil_disk(m, k):
    K = _disk(k); p0 = k // 2
    return Fn.conv2d(Fn.pad(m.float()[None, None], (p0, k - 1 - p0, p0, k - 1 - p0)), K[None, None])[0, 0] > 0.5
_RIM = AP10 & ~(Fn.conv2d(Fn.pad(AP10.float()[None, None], (8, 8, 8, 8), value=1.0), _disk(17)[None, None])[0, 0] >= float(_disk(17).sum()) - 0.5)
def _repair_side(q, side, maxit=400):
    q = q.clone()
    for it in range(maxit):
        ts, ta = _thin(q); ns, na = int(ts.sum()), int(ta.sum())
        if ns == 0 and na == 0: return q, it
        if side == "A":
            if it == 0 and na: qn = q | ta
            elif ns: qn = q & ~ts
            else: qn = q & ~_crop(_dil_disk(_mpad(ta), L_GAP))
        else:
            if it == 0 and ns: qn = q & ~ts
            elif na: qn = q | ta
            elif bool((ts & _RIM).any()): qn = q & ~_crop(_dil_disk(_mpad(ts & _RIM), L_GAP))
            else: qn = (q | _crop(_dil_disk(_mpad(ts), L_MIN))) & AP10
        qn = symmetrize(qn)
        if bool((qn == q).all()): raise RuntimeError(f"repair {side}: no change at it {it} ({ns} thin SiN, {na} thin air)")
        q = qn
    raise RuntimeError(f"repair {side} did not converge")
def fab_repair(q0):
    qa, ia = _repair_side(q0, "A"); qs, is_ = _repair_side(q0, "S")
    diff = (qa != qs).cpu().numpy()
    if not diff.any(): return qa, dict(itA=ia, itS=is_, sites=0, sin_side=0, seam_it=0)
    sites, n = ndi.label(ndi.binary_dilation(diff, iterations=12), structure=np.ones((3, 3), int)); idx = np.arange(1, n + 1)
    q0n = q0.cpu().numpy(); cA = ndi.sum((qa.cpu().numpy() != q0n).astype(np.int32), sites, idx); cS = ndi.sum((qs.cpu().numpy() != q0n).astype(np.int32), sites, idx)
    pick = np.zeros(n + 1, bool); pick[1:] = cS < cA
    qc = symmetrize(torch.where(torch.tensor(pick[sites], device=dev), qs, qa)); q, itf = _repair_side(qc, "A")
    return q, dict(itA=ia, itS=is_, sites=int(n), sin_side=int((cS < cA).sum()), seam_it=itf)
def realise(fq):
    q0 = symmetrize((Gt > fill_px(fq)) & AP10)
    q, info = fab_repair(q0); ts, ta = _thin(q)
    assert int(ts.sum()) == 0 and int(ta.sum()) == 0
    info.update(repair_px=int((q != q0).sum()), fill=float(q[AP10].float().mean()))
    return q, info

# ---- FDTD Jones maps of a design (octant-valid dict, cnt) ----
def tmaps(q):
    global Dq
    Dq = q; cache.clear(); refresh(tiles)
    acc_ = {k: torch.zeros((len(LAM_NM), NCELL, NCELL), dtype=torch.complex64, device=dev) for k in JC}; cnt = torch.zeros((NCELL, NCELL), device=dev)
    for key in tiles:
        fl, pts = cache[key]
        for k in JC:
            a, c = tile_cells(key, fl[k], pts); acc_[k] = acc_[k] + a
        cnt = cnt + c
    T = {k: _norm(v, cnt) for k, v in acc_.items()}
    with torch.no_grad(): I = float(i_tar_jones(_jones_expand(T)))
    return T, cnt, I

# ---- pupil-cell geometry ----
cc = (torch.arange(NCELL, device=dev, dtype=torch.float32) - NCELL // 2 + 0.5) * PITCH           # cell centres (um)
CR = torch.hypot(cc[:, None], cc[None, :]); ci_ = torch.arange(NCELL, device=dev)
OCTC = (ci_[:, None] >= ci_[None, :]) & (ci_[None, :] >= NCELL // 2)                              # octant cells (x index >= y index >= centre)
INSIDE = OCTC & (CR <= R_AP)                                                                        # cells whose fill is a variable
def q_cells(fo):                                                          # (NCELL,NCELL) octant-valid -> (360,360) quadrant cells, transpose-symmetric
    a = fo[CELL_Q:, CELL_Q:]; lo = torch.arange(CELL_Q, device=dev)
    return torch.where(lo[:, None] >= lo[None, :], a, a.T)
SMD = OUT / "spacemap"; SMD.mkdir(exist_ok=True)

# ---- library: one nominal fill per tile core, measured once ----
lib_path = SMD / "library.npz"
def tile_core(key, margin):
    x0, y0 = key; inx = (cc >= x0 - TILE / 2 + margin) & (cc < x0 + TILE / 2 - margin); iny = (cc >= y0 - TILE / 2 + margin) & (cc < y0 + TILE / 2 - margin)
    return inx[:, None] & iny[None, :]
if os.environ.get("SM_EVAL_FILE"):                                       # evaluate a given design (Q_bits quadrant): per-cell Jones maps of every tile core
    _z = np.load(os.environ["SM_EVAL_FILE"]); q = symmetrize(torch.tensor(np.unpackbits(_z["Q_bits"])[:H10 * H10].reshape(H10, H10).astype(bool), device=dev) & AP10)
    ts, ta = _thin(q); print(f"[spacemap] eval {os.environ['SM_EVAL_FILE']}: fill {float(q[AP10].float().mean()):.3f}, thin SiN {int(ts.sum())}, thin air {int(ta.sum())}", flush=True)
    T, cnt, I = tmaps(q); print(f"[spacemap] eval FDTD done: I_tar {I:.5f}", flush=True)
    tile_of_cell = torch.full((NCELL, NCELL), -1, dtype=torch.long, device=dev)
    for ti, key in enumerate(tiles): tile_of_cell[tile_core(key, SM_MARGIN)] = ti
    use = OCTC & (CR <= R_AP - SM_MARGIN) & (cnt >= cnt.max() - 0.5) & (tile_of_cell >= 0); sel = use.nonzero(as_tuple=True)
    qf = Fn.avg_pool2d(Fn.pad(q.float()[None, None], (0, CELL_Q * 29 - H10, 0, CELL_Q * 29 - H10)), 29)[0, 0]
    fcell = torch.zeros((NCELL, NCELL), device=dev); fcell[CELL_Q:, CELL_Q:] = qf
    np.savez_compressed(SMD / "eval_cells.npz", I_tar=I, tiles=np.array(tiles), tile_of_cell=tile_of_cell[sel].cpu().numpy(), ci=sel[0].cpu().numpy(), cj=sel[1].cpu().numpy(),
                        freal=fcell[sel].cpu().numpy(), **{f"T_{k}": T[k][:, sel[0], sel[1]].cpu().numpy() for k in JC}, source=os.environ["SM_EVAL_FILE"])
    print(f"FF_DONE (eval only, {len(sel[0])} cells saved)", flush=True); raise SystemExit(0)
if not lib_path.exists():
    levels = np.linspace(SM_F0, SM_F1, SM_NLIB)
    usable = [int((tile_core(k, SM_MARGIN) & OCTC & (CR <= R_AP - SM_MARGIN)).sum()) for k in tiles]
    by_area = np.argsort(usable)[::-1]                                     # largest usable area first
    ends = [i for pair in zip(range(SM_NLIB // 2), range(SM_NLIB - 1, SM_NLIB // 2 - 1, -1)) for i in pair] + ([SM_NLIB // 2] if SM_NLIB % 2 else [])
    perm = np.zeros(SM_NLIB, int); perm[by_area] = ends[:SM_NLIB]         # the extreme fills go to the largest tiles
    fo = torch.full((NCELL, NCELL), 0.5, device=dev); tile_of_cell = torch.full((NCELL, NCELL), -1, dtype=torch.long, device=dev)
    for ti, key in enumerate(tiles):
        m = tile_core(key, 0.0); fo[m] = float(levels[perm[ti]]); tile_of_cell[m] = ti
    print("[spacemap] library assignment (tile: nominal fill, usable cells): " + ", ".join(f"{k[0]:.0f},{k[1]:.0f}: {levels[perm[i]]:.3f} ({usable[i]})" for i, k in enumerate(tiles)), flush=True)
    q, info = realise(q_cells(fo)); print(f"[spacemap] library design realised: {info}", flush=True)
    T, cnt, I = tmaps(q); print(f"[spacemap] library FDTD done: I_tar {I:.5f}", flush=True)
    core = torch.zeros_like(OCTC)
    for key in tiles: core |= tile_core(key, SM_MARGIN)
    use = core & OCTC & (CR <= R_AP - SM_MARGIN) & (cnt >= cnt.max() - 0.5)
    # realised fill per cell (10 nm design averaged over the cell)
    qf = Fn.avg_pool2d(Fn.pad(q.float()[None, None], (0, CELL_Q * 29 - H10, 0, CELL_Q * 29 - H10)), 29)[0, 0]    # 29 px = 0.29 um
    fcell = torch.zeros((NCELL, NCELL), device=dev); fcell[CELL_Q:, CELL_Q:] = qf
    sel = use.nonzero(as_tuple=True)
    np.savez_compressed(lib_path, levels=levels, perm=perm, tiles=np.array(tiles), I_tar=I, tile_of_cell=tile_of_cell[sel].cpu().numpy(),
                        fnom=fo[sel].cpu().numpy(), freal=fcell[sel].cpu().numpy(), ci=sel[0].cpu().numpy(), cj=sel[1].cpu().numpy(),
                        **{f"T_{k}": T[k][:, sel[0], sel[1]].cpu().numpy() for k in JC}, T_air=T["xx"][:, NCELL - 1, NCELL // 2].cpu().numpy(),
                        d10_quad=pack_q(q), d10_shape=np.array([H10, H10]), repair=json.dumps(info))
    del T
L_ = np.load(lib_path); fn = L_["fnom"]
ncl = np.array([int(np.isclose(fn, lv).sum()) for lv in L_["levels"]]); LEV = L_["levels"][ncl >= 100]
print(f"[spacemap] library cells per level: {dict(zip(np.round(L_['levels'], 3).tolist(), ncl.tolist()))}; kept {len(LEV)} levels with >= 100 cells", flush=True)
Tbar = np.stack([0.5 * (L_["T_xx"][:, np.isclose(fn, lv)].mean(1) + L_["T_yy"][:, np.isclose(fn, lv)].mean(1)) for lv in LEV], 0)    # (nlev, Nl)
spread = np.array([np.abs(np.concatenate([L_["T_xx"][:, np.isclose(fn, lv)], L_["T_yy"][:, np.isclose(fn, lv)]], 1) - Tbar[i][:, None]).mean() for i, lv in enumerate(LEV)])
off = np.array([np.abs(np.concatenate([L_["T_xy"][:, np.isclose(fn, lv)], L_["T_yx"][:, np.isclose(fn, lv)]], 1)).mean() for lv in LEV])
ph = np.unwrap(np.angle(Tbar), axis=0)
print("[spacemap] library (nominal fill: |t_bar| at 420/540/670 nm, phase range over fills, mean |t - t_bar|, mean |off-diagonal|):", flush=True)
for i, lv in enumerate(LEV): print(f"  f {lv:.3f}: |t| {abs(Tbar[i, 0]):.3f} {abs(Tbar[i, 4]):.3f} {abs(Tbar[i, 8]):.3f}  phase(540) {ph[i, 4]:+.2f}  scatter {spread[i]:.3f}  offdiag {off[i]:.3f}", flush=True)
print(f"[spacemap] phase coverage over the library (rad, 2pi = {2 * math.pi:.2f}): " + " ".join(f"{LAM_NM[l]}:{ph[:, l].max() - ph[:, l].min():.2f}" for l in range(len(LAM_NM))), flush=True)
_inc = np.array([np.abs(0.5 * (L_["T_xx"][4, np.isclose(fn, lv)] + L_["T_yy"][4, np.isclose(fn, lv)])).mean() for lv in LEV])
_fr = np.array([L_["freal"][np.isclose(fn, lv)].std() for lv in LEV])
print("[spacemap] coherent vs incoherent at 540 nm (|t_bar| / mean |t|) and per-cell realised-fill std: " +
      " ".join(f"{lv:.2f}:{abs(Tbar[i, 4]):.2f}/{_inc[i]:.2f}/{_fr[i]:.2f}" for i, lv in enumerate(LEV)), flush=True)
if os.environ.get("SM_LIB_ONLY", "0") == "1":                            # library measurement only (carrier tests)
    print("FF_DONE (library only)", flush=True); raise SystemExit(0)
LEVt = torch.tensor(LEV, device=dev, dtype=torch.float32); TBt = torch.tensor(Tbar, device=dev, dtype=torch.complex64)
T_AIR = torch.tensor(L_["T_air"], device=dev, dtype=torch.complex64)
def tbar_of(f):                                                           # piecewise-linear complex t_bar(f) -> (Nl, NCELL, NCELL)
    j = torch.searchsorted(LEVt, f.contiguous()).clamp(1, len(LEV) - 1); w = ((f - LEVt[j - 1]) / (LEVt[j] - LEVt[j - 1])).clamp(0, 1)
    return (TBt[j - 1] * (1 - w)[..., None] + TBt[j] * w[..., None]).permute(2, 0, 1)
def surrogate_T(f, d):                                                    # isotropic t_bar(f) inside, air outside, plus residual d
    tb = torch.where(INSIDE[None], tbar_of(f), T_AIR[:, None, None]); z = torch.zeros_like(tb)
    base = {"xx": tb, "yy": tb, "xy": z, "yx": z}
    return {k: base[k] + (d[k] if d is not None else 0) for k in JC}
def optimise(f_start, d, f_center=None, delta=None, iters=SM_ITERS, tag=""):
    lo_, hi_ = float(LEV[0]), float(LEV[-1])
    if delta is None:
        u = torch.logit(((f_start - lo_) / (hi_ - lo_)).clamp(1e-3, 1 - 1e-3)).detach().requires_grad_(True)
        fmap = lambda u: lo_ + (hi_ - lo_) * torch.sigmoid(u)
    else:
        u = torch.zeros_like(f_start).requires_grad_(True)
        fmap = lambda u: (f_center + delta * torch.tanh(u)).clamp(lo_, hi_)
    opt = torch.optim.Adam([u], lr=SM_LR); best = (-1e30, None); t1 = time.time()
    for it in range(iters):
        opt.zero_grad(); f = fmap(u); I = i_tar_jones(_jones_expand(surrogate_T(f, d))); (-I).backward()
        with torch.no_grad(): u.grad[~INSIDE] = 0
        opt.step()
        if float(I) > best[0]: best = (float(I), f.detach().clone())
        if it % 25 == 0 or it == iters - 1: print(f"  [surrogate{tag}] it {it}: I_sur {float(I):.5f} (best {best[0]:.5f}) {time.time() - t1:.0f} s", flush=True)
    del opt, u; _jresp.J = None                                         # free the last surrogate graph before the FDTD (see loop_elem.py)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best[1], best[0]

# ---- outer loop ----
state_path = SMD / "sm_state.npz"; hist = []
if state_path.exists():
    S = np.load(state_path, allow_pickle=True); k0 = int(S["k"]) + 1; f_acc = torch.tensor(S["f_acc"], device=dev); I_acc = float(S["I_acc"]); delta = float(S["delta"])
    d_acc = {k: torch.tensor(S[f"d_{k}"], device=dev) for k in JC}; hist = json.loads(str(S["hist"])); Dq_acc = torch.tensor(np.unpackbits(S["d10"])[:H10 * H10].reshape(H10, H10).astype(bool), device=dev)
    print(f"[spacemap] resumed after outer iteration {k0 - 1}: I_acc {I_acc:.5f}, Delta {delta}", flush=True)
else:
    k0 = 0; f_acc = None; I_acc = -1e30; delta = SM_DELTA0; d_acc = None; Dq_acc = None
def save_step(k, q, I, gate, rec):
    hard = view20(q)
    np.savez(OUT / f"step_{k:04d}.npz", hard_eval_oct=np.packbits(hard), hard_shape=np.array(hard.shape), d10_quad=pack_q(q), d10_shape=np.array([H10, H10]),
             I_tar=I, I_accepted=I_acc)
    if gate != "reject":
        np.savez(OUT / "best.npz", hard_oct=np.packbits(hard), hard_shape=np.array(hard.shape), d10_quad=pack_q(q), d10_shape=np.array([H10, H10]), I_tar=I, step=k)
    hist.append(rec); (OUT / "history.json").write_text(json.dumps(hist, indent=1))
for k in range(k0, SM_OUTER + 1):
    t0 = time.time()
    if k == 0:                                                            # surrogate only, from a uniform half fill
        f_new, I_pred = optimise(torch.full((NCELL, NCELL), 0.5, device=dev), None, tag=" k0")
        pred_gain = float("nan")
    else:
        f_new, I_pred = optimise(f_acc, d_acc, f_center=f_acc, delta=delta, tag=f" k{k}")
        with torch.no_grad(): I_here = float(i_tar_jones(_jones_expand(surrogate_T(f_acc, d_acc))))
        pred_gain = I_pred - I_here; _jresp.J = None                      # corrected-surrogate gain (I_here = I_acc by construction of d)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    t_sur = time.time() - t0
    q, info = realise(q_cells(f_new)); t_rep = time.time() - t0 - t_sur
    T, cnt, I = tmaps(q); t_fdtd = time.time() - t0 - t_sur - t_rep
    ratio = (I - I_acc) / pred_gain if k > 0 and pred_gain > 0 else float("nan")
    gate = "first" if k == 0 else ("accept" if I > I_acc else "reject")
    changed = int((q != Dq_acc).sum()) if Dq_acc is not None else 0
    if gate != "reject":
        if k > 0: delta = min(delta * 1.5, 0.8) if ratio > 0.5 else (max(delta * 0.5, 0.01) if ratio < 0.2 else delta)
        with torch.no_grad():
            tb = surrogate_T(f_new, None); d_acc = {kk: (T[kk] - tb[kk]).detach() for kk in JC}
        f_acc, I_acc, Dq_acc = f_new, I, q
    else:
        delta = max(delta * 0.5, 0.01)
    rec = {"step": k, "I_tar": I, "I_accepted": I_acc, "gate": gate, "algo": "spacemap", "region": f"SM{k}", "I_surrogate": I_pred, "pred_gain": pred_gain,
           "tr_ratio": ratio, "delta_next": delta, "flipped": changed, "fill": info["fill"], "repair": info, "t_surrogate_s": t_sur, "t_repair_s": t_rep,
           "t_forward_s": t_fdtd, "t_step_s": time.time() - t0, "ls_min_px": L_MIN, "ls_gap_px": L_GAP}
    save_step(k, q, I, gate, rec)
    np.savez(state_path, k=k, f_acc=f_acc.cpu().numpy(), I_acc=I_acc, delta=delta, hist=json.dumps(hist), d10=pack_q(Dq_acc),
             **{f"d_{kk}": d_acc[kk].cpu().numpy() for kk in JC})
    print(f"step {k:3d}  I_tar(binary) {I:.5f}  accepted {I_acc:.5f}  gate {gate:6s}  region SM{k}  I_surrogate {I_pred:.5f}  pred {pred_gain:+.2e}  ratio {ratio:+.2f}  "
          f"Delta next {delta:.3f}  flipped {changed}  repair {info['repair_px']} px ({info['sites']} sites)  fill {info['fill']:.3f}  "
          f"{time.time() - t0:.0f}s (surrogate {t_sur:.0f}s, repair {t_rep:.0f}s, forward {t_fdtd:.0f}s)", flush=True)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
print(f"FF_DONE (best accepted I_tar {I_acc:.5f})", flush=True)
