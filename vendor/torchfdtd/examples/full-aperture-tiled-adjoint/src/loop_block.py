# ---------------- optimization loop (block-coordinate binary trust region) ----------------
# Every scored structure is binary. Each step works inside one REGION (a 2x2 block of tiles, two staggered partitions so
# every pixel is interior to some region). The region's density gradient comes from the adjoints of the region's tiles
# only, taken at the accepted design with the exact pupil sensitivity dI/dT of the full forward; it is exact for pixels
# whose every covering tile window lies in the region (overlap strips shared with outside tiles are excluded). A proposal
# flips the top-K first-order gains there (boundary pixels, plus a small share of interior births), so its forward
# re-solves only the region's changed tiles and an accept costs only the region's adjoints: ~6 tiles of work per step
# instead of 44. Trust region per region on K (ratio rule); tabu on small failed sets; region change after R_TRIES rejects.
K0 = int(os.environ.get("FF_K0", "4000")); KMIN = int(os.environ.get("FF_KMIN", "1")); KMAX = int(os.environ.get("FF_KMAX", "400000"))
NO_ISLAND = os.environ.get("FF_NO_ISLAND", "0") == "1"                   # polish: also reject flips that create a new isolated SiN island
import scipy.ndimage as _ndi
FAB_K = int(os.environ.get("FF_FAB_K", "1"))                           # >1: reject flips that create a new SiN line / air gap narrower than FAB_K px (polish phase: 2 = 40 nm)
FAB_KG = int(os.environ.get("FF_FAB_K_GAP", str(FAB_K)))                 # air-gap minimum when it differs from the SiN one (curvilinear polish: SiN 4 = 80 nm, gap 2 = 40 nm)
CELL_CAP = int(os.environ.get("FF_CELL_CAP", "0"))                     # >0: at most this many flips per 0.29 um pupil cell per proposal (off in the running sweep)
KPROBE = os.environ.get("FF_KPROBE", "0") == "1"                        # probe 2K by measured gain (mock: worse, 548 vs 577) - off by default
BIRTH = float(os.environ.get("FF_BIRTH_FRAC", "0.1")); R_TRIES = int(os.environ.get("FF_R_TRIES", "3")); R_STAY = int(os.environ.get("FF_R_STAY", "4"))
STAY_RULE = os.environ.get("FF_STAY_RULE", "count")                    # count: leave after R_STAY accepts | rate: stay while gain/s >= STAY_FRAC x the visit's first accept
STAY_FRAC = float(os.environ.get("FF_STAY_FRAC", "0.4")); R_STAY_MAX = int(os.environ.get("FF_R_STAY_MAX", "12"))
STALE = os.environ.get("FF_STALE", "0") == "1"                          # after a fresh-gradient accept, one proposal from the previous gradient (no adjoint)
hist_path = OUT / "history.json"
MAX_HOURS = float(os.environ.get("FF_MAX_HOURS", "0")); T_START = time.time()
ii = torch.arange(NCELL, device=dev); a_ = torch.where(ii >= NCELL // 2, ii, NCELL - 1 - ii)
A_ = a_[:, None].expand(NCELL, NCELL); B_ = a_[None, :].expand(NCELL, NCELL); REP = (torch.maximum(A_, B_) * NCELL + torch.minimum(A_, B_)).reshape(-1)

def window_reps(k):                                                       # octant representatives read by tile k's window
    i0, j0 = tile_window(*k)
    ti = torch.arange(i0, i0 + shape[0], device=dev)[:, None].expand(shape[0], shape[1]); tj = torch.arange(j0, j0 + shape[1], device=dev)[None, :].expand(shape[0], shape[1])
    inside = (ti >= 0) & (ti < NPX) & (tj >= 0) & (tj < NPX)
    hi, lo = oct_index(ti[inside], tj[inside]); return hi, lo, inside
def coverage(keys):
    c = torch.zeros((NPX, NPX), dtype=torch.int16, device=dev)
    for k in keys:
        hi, lo, _ = window_reps(k); c.index_put_((hi, lo), torch.ones_like(hi, dtype=torch.int16), accumulate=True)
    return c
def forward_score():
    cache.clear(); refresh(tiles); t_map, cnt = assemble()
    t_leaf = t_map.detach().requires_grad_(True); I = i_tar_of(t_leaf); I.backward()
    g_oct = torch.zeros((len(LAM_NM), NCELL * NCELL), dtype=t_leaf.grad.dtype, device=dev).index_add(1, REP, t_leaf.grad.reshape(len(LAM_NM), -1)).view(len(LAM_NM), NCELL, NCELL)
    return float(I.detach()), g_oct, cnt, t_map[4].abs().cpu().numpy()
def region_gradient(keys, g_oct, cnt):                                    # sum of the region tiles' density VJPs on the representatives
    G = torch.zeros((NPX, NPX), device=dev)
    for k in keys:
        g = tile_grad(k, g_oct, cnt); hi, lo, inside = window_reps(k)
        G.index_put_((hi, lo), g[inside].float(), accumulate=True); torch.cuda.empty_cache()
    return G
def d4_full(m):
    ii_ = torch.arange(NPX, device=dev); a = torch.where(ii_ >= half, ii_, NPX - 1 - ii_)
    A2 = a[:, None].expand(NPX, NPX); B2 = a[None, :].expand(NPX, NPX)
    return m[torch.maximum(A2, B2), torch.minimum(A2, B2)]
def boundary(mf):
    b = torch.zeros_like(mf)
    b[1:, :] |= mf[1:, :] != mf[:-1, :]; b[:-1, :] |= mf[:-1, :] != mf[1:, :]
    b[:, 1:] |= mf[:, 1:] != mf[:, :-1]; b[:, :-1] |= mf[:, :-1] != mf[:, 1:]
    return b
def select_capped(gain, pool, K):
    """Top-K positive first-order gains in pool, at most CELL_CAP per 0.29 um pupil cell (agentA: uncapped top-K piles flips
    into the same cell edge, whose t then over-shoots; a per-cell cap keeps the first-order prediction honest)."""
    cand = torch.where(pool, gain, torch.full_like(gain, -1.0)).reshape(-1); npool = int(pool.sum())
    if npool == 0 or K <= 0: return torch.zeros(0, dtype=torch.long, device=dev)
    M = min(npool, (8 * K + 1000) if CELL_CAP > 0 else K)
    v, ix = torch.topk(cand, M)                                           # sorted by gain, descending
    if CELL_CAP > 0:
        i, j = ix // NPX, ix % NPX
        ci = torch.floor((XG0 + (i.double() + 0.5) * MESH) / PITCH).long(); cj = torch.floor((XG0 + (j.double() + 0.5) * MESH) / PITCH).long()
        cell = ci * 100003 + cj
        cs, order = torch.sort(cell, stable=True)                         # stable: gain order kept inside each cell
        start = torch.ones_like(cs, dtype=torch.bool); start[1:] = cs[1:] != cs[:-1]
        pos = torch.arange(cs.numel(), device=dev); gstart = torch.cummax(torch.where(start, pos, torch.zeros_like(pos)), 0).values
        rank = torch.empty_like(pos); rank[order] = pos - gstart
        ix = ix[rank < CELL_CAP]
    return ix[:K]
def _open(x, k):                                                          # binary opening with a k x k square (float in, float out)
    e = -torch.nn.functional.max_pool2d(-x[None, None], k, stride=1)
    return torch.nn.functional.max_pool2d(torch.nn.functional.pad(e, (k - 1, k - 1, k - 1, k - 1)), k, stride=1)[0, 0]
def thin_features(m, k, kg=None):                                         # SiN narrower than k px, and air gaps narrower than kg px (default k)
    kg = k if kg is None else kg
    x = m.float(); o = _open(x, k) > 0.5; c = ~(_open(1 - x, kg) > 0.5)
    return (m & ~o) | (~m & c)
def fab_filter(mf, idx, rounds=6):
    """Drop flips that create a NEW feature (SiN line or air gap) narrower than FAB_K px; features already present are left.
    Checked on the full D4 image around the proposal, so mirrored neighbours across the axes and the diagonal count."""
    i, j = idx // NPX, idx % NPX
    lo_ = int(torch.minimum(i.min(), j.min())) - 3 * FAB_K; hi_ = int(torch.maximum(i.max(), j.max())) + 3 * FAB_K + 1
    lo_, hi_ = max(lo_, 0), min(hi_, NPX)
    base = thin_features(mf[lo_:hi_, lo_:hi_], FAB_K, FAB_KG); sin0 = mf[lo_:hi_, lo_:hi_].cpu().numpy()
    def violations(pf):                                                   # new sub-FAB_K features, plus (FF_NO_ISLAND) new SiN islands
        new = thin_features(pf, FAB_K, FAB_KG) & ~base
        if NO_ISLAND:                                                     # a SiN component (8-conn) sharing no pixel with the accepted SiN and not
            p_ = pf.cpu().numpy(); lab, n = _ndi.label(p_, structure=np.ones((3, 3), int))   # reaching the window edge
            keep = np.zeros(n + 1, bool); keep[np.unique(lab[p_ & sin0])] = True
            keep[np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))] = True; keep[0] = True
            if not keep.all(): new = new | torch.from_numpy(~keep[lab]).to(dev)
        new[:FAB_K] = False; new[-FAB_K:] = False; new[:, :FAB_K] = False; new[:, -FAB_K:] = False
        return new
    for _ in range(rounds):
        if idx.numel() == 0: break
        th = theta.detach().clone().view(-1); th[idx] = -th[idx]
        pf = d4_full(th.view(NPX, NPX) >= 0)[lo_:hi_, lo_:hi_]
        new = violations(pf)
        if not bool(new.any()): return idx
        near = torch.nn.functional.max_pool2d(new.float()[None, None], 3, stride=1, padding=1)[0, 0] > 0.5
        ri, rj = torch.nonzero(near, as_tuple=True); hi2, lo2 = oct_index(ri + lo_, rj + lo_)
        bad = torch.zeros(NPX * NPX, dtype=torch.bool, device=dev); bad[hi2 * NPX + lo2] = True
        idx = idx[~bad[idx]]
    th = theta.detach().clone().view(-1); th[idx] = -th[idx]
    new = violations(d4_full(th.view(NPX, NPX) >= 0)[lo_:hi_, lo_:hi_])
    return idx if not bool(new.any()) else torch.zeros(0, dtype=torch.long, device=dev)
def solved_tiles():                                                       # tiles whose mask hash has no cached field yet (= will be re-solved)
    n = 0
    with torch.no_grad():
        for k in tiles:
            if not cache_path(k, tile_hash(rho_tile_of(theta, *tile_window(*k)))).exists(): n += 1
    return n

# regions: 2x2 blocks of the octant tile grid, two partitions offset by one tile
tile_ab = {k: (int(round((k[0] - TILE / 2) / TILE)), int(round((k[1] - TILE / 2) / TILE))) for k in tiles}
partitions = []
for s in (0, 1):
    groups = {}
    for k, (a, b) in tile_ab.items(): groups.setdefault(((a + s) // 2, (b + s) // 2), []).append(k)
    partitions.append([sorted(v) for _, v in sorted(groups.items())])
COV_ALL = coverage(tiles)
print(f"[block] regions per partition: {[len(p) for p in partitions]}; tiles per region: {[[len(r) for r in p] for p in partitions]}", flush=True)

# state (resume from acc_state.pt when present: it holds the accepted latent, score and, if valid, a gradient)
log = json.loads(hist_path.read_text()) if hist_path.exists() and os.environ.get("FF_FRESH", "0") != "1" else []
ACC_PATH = OUT / "acc_state.pt"
acc = None
if ACC_PATH.exists():
    _s = torch.load(ACC_PATH, map_location=dev, weights_only=False)
    with torch.no_grad(): theta.copy_(_s["theta"].to(dev))
    acc = {"theta": theta.detach().clone(), "I": float(_s["I"]), "t540": _s["t540"], "cnt": _s["cnt"].to(dev), "g_oct": _s.get("g_oct"),
           "G": _s.get("G"), "G_for": _s.get("G_for", "all" if _s.get("G") is not None else None)}
    if acc["G"] is not None: acc["G"] = acc["G"].to(dev)
    if acc["g_oct"] is not None: acc["g_oct"] = acc["g_oct"].to(dev)
    print(f"[block] resumed accepted design (I_tar {acc['I']:.5f}, gradient valid for {acc['G_for']})", flush=True); del _s
INIT_MASK = os.environ.get("FF_INIT_MASK")                              # polish: start from a saved accepted mask (best.npz hard_oct) instead of the warm start
if acc is None and INIT_MASK:
    _d = np.load(INIT_MASK); _q = np.unpackbits(_d["hard_oct"])[:int(np.prod(_d["hard_shape"]))].reshape(tuple(_d["hard_shape"])).astype(bool)
    with torch.no_grad():
        _mq = torch.zeros((NPX, NPX), dtype=torch.bool, device=dev); _mq[half:, half:] = torch.tensor(_q, device=dev)
        _mf = d4_full(_mq); CK = int(os.environ.get("FF_CLEANUP_K", "1"))
        if CK > 1:                                                        # open (drop SiN narrower than CK px), then close (fill air gaps narrower than CK px)
            _x = (_open(_mf.float(), CK) > 0.5).float(); _mf = ~(_open(1 - _x, CK) > 0.5)
        _oa = (torch.arange(NPX, device=dev)[:, None] >= half) & (torch.arange(NPX, device=dev)[None, :] >= half) & (torch.arange(NPX, device=dev)[:, None] >= torch.arange(NPX, device=dev)[None, :])
        theta.copy_(torch.where(_mf, torch.full_like(theta, 0.5), torch.full_like(theta, -0.5)))
        theta[_oa & ~oct_mask] = -0.5                                    # outside the aperture stays air
    print(f"[block] start mask from {INIT_MASK} (step {int(_d['step'])}, recorded I_tar {float(_d['I_tar']):.5f}); cleanup k={CK}", flush=True)
    del _d, _q, _mq, _mf
def save_acc():
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in acc.items() if k != "theta"} | {"theta": acc["theta"].cpu()}, OUT / "acc_state.tmp")
    (OUT / "acc_state.tmp").replace(ACC_PATH)

Kreg = {}; Kstate = {}; tabu = torch.zeros((NPX, NPX), dtype=torch.bool, device=dev)
part = 0; reg_i = 0; tries = 0; stay = 0; step = len(log)
for h in reversed(log):                                                   # resume the region sweep where it stopped (and that region's K)
    if h.get("algo") == "block" and h.get("region", "-") != "-":
        part, reg_i = int(h["region"][1]), int(h["region"].split("r")[1]); Kreg[h["region"]] = int(h.get("K_next", K0))
        print(f"[block] resuming the sweep at region {h['region']} (K {Kreg[h['region']]})", flush=True); break
best_I = max([h["I_accepted"] for h in log] + ([acc["I"]] if acc else []), default=-1e30)
stale_G = None; stale_rid = None; rate0 = None; stale_fail = 0             # stale_fail: consecutive useless stale steps (2 -> stale off for the run)
def next_region():
    global reg_i, part, tries, stay, stale_G, rate0
    tries = 0; stay = 0; tabu.zero_(); stale_G = None; rate0 = None
    reg_i += 1
    if reg_i >= len(partitions[part]): reg_i = 0; part = 1 - part
    if K_RESET: Kreg.pop(f"p{part}r{reg_i}", None)                      # a new visit has a fresh gradient: start from K0, not the K the last visit shrank to
K_RESET = os.environ.get("FF_K_RESET", "0") == "1"
if os.environ.get("FF_RESUME_NEXT") == "1" and log:                      # resume at the region after the one the last record names
    next_region(); print(f"[block] FF_RESUME_NEXT: starting at region p{part}r{reg_i}", flush=True)
while step < N_STEPS:
    t0 = time.time(); gate = "-"; pred = float("nan"); ratio = float("nan"); nflip = 0; t_adj = 0.0; n_solved = 0; rid = "-"; was_stale = False
    if acc is None:                                                       # score the warm start once (all tiles, no adjoint yet)
        n_solved = solved_tiles(); I, g_oct, cnt, t540 = forward_score()
        acc = {"theta": theta.detach().clone(), "I": I, "g_oct": g_oct, "cnt": cnt, "t540": t540, "G": None, "G_for": None}; save_acc(); gate = "first"
    else:
        region = partitions[part][reg_i]; rid = f"p{part}r{reg_i}"
        if acc["g_oct"] is None:                                          # resumed without the pupil sensitivity: rebuild it (cached forward)
            _, acc["g_oct"], acc["cnt"], _ = forward_score()
        was_stale = stale_G is not None and stale_rid == rid
        if was_stale: Guse = stale_G; stale_G = None                     # the gradient of the previous accepted design; just-flipped pixels now have negative gain
        else:
            stale_G = None
            if acc["G_for"] not in ("all", rid):
                t1 = time.time(); acc["G"] = region_gradient(region, acc["g_oct"], acc["cnt"]); acc["G_for"] = rid; t_adj = time.time() - t1; save_acc()
            Guse = acc["G"]
        with torch.no_grad():
            elig = oct_mask & (coverage(region) == COV_ALL) & (COV_ALL > 0) & ~tabu
            m = theta >= 0; mf = d4_full(m); gain = Guse * (1 - 2 * m.float())
            bnd = boundary(mf)
            K = Kreg.get(rid, K0)
            pool = elig & (gain > 0) & (bnd if BIRTH <= 0 else torch.ones_like(bnd))   # births off by default: a lone new pixel breaks the 40 nm rule anyway
            idx = select_capped(gain, pool, K)
            n_pre = int(idx.numel()); n_fab = 0
            if FAB_K > 1 and n_pre: idx = fab_filter(mf, idx); n_fab = n_pre - int(idx.numel())
            nflip = int(idx.numel()); pred = float(gain.view(-1)[idx].sum()) if nflip else float("nan")
        if nflip == 0:
            print(f"step {step:3d}  region {rid}: no candidate with positive first-order gain -> next region", flush=True); next_region(); continue
        with torch.no_grad():
            flat = theta.view(-1); flat[idx] = torch.where(flat[idx] >= 0, torch.full_like(flat[idx], -0.5), torch.full_like(flat[idx], 0.5))
        n_solved = solved_tiles(); I, g_oct, cnt, t540 = forward_score()
        ratio = (I - acc["I"]) / pred if pred > 0 else float("nan")
        if I > acc["I"]:
            gate = "accept"; stay += 1; tries = 0; tabu.zero_(); gain = I - acc["I"]
            if STALE and not was_stale and stale_fail < 2: stale_G = acc["G"]; stale_rid = rid   # keep this gradient for one adjoint-free proposal
            if was_stale: stale_fail = 0 if ratio >= 0.15 else stale_fail + 1
            rate = gain / max(time.time() - t0, 1.0); rate0 = rate if rate0 is None else max(rate0, rate)   # best rate of this visit (a near-zero first accept must not pin the visit)
            acc = {"theta": theta.detach().clone(), "I": I, "g_oct": g_oct, "cnt": cnt, "t540": t540, "G": None, "G_for": None}; save_acc()
            # K by measured gain, not only by the ratio (which sits at 0.2-0.4 here, so the ratio rule never grows K):
            # after a plain accept, probe 2K once; keep 2K if it measured more than K did, otherwise go back to K and stop probing here
            st = Kstate.setdefault(rid, {"probe": False, "cool": not KPROBE, "gain": 0.0})
            if was_stale: Kreg[rid] = K                                   # a stale step's ratio reflects the old gradient, not the trust radius
            elif not KPROBE: Kreg[rid] = min(K * 2, KMAX) if ratio > 0.75 else (max(K // 2, KMIN) if ratio < 0.25 else K)   # ratio rule (default)
            elif st["probe"]:
                if gain > st["gain"]: Kreg[rid] = min(K * 2, KMAX); st["probe"] = True          # the larger set paid off: try doubling again
                else: Kreg[rid] = max(K // 2, KMIN); st["probe"] = False; st["cool"] = True   # it did not: back to the smaller K
            elif ratio < 0.1: Kreg[rid] = max(K // 2, KMIN)
            elif not st["cool"]: Kreg[rid] = min(K * 2, KMAX); st["probe"] = True
            else: Kreg[rid] = K
            st["gain"] = gain
            leave = stay >= R_STAY if STAY_RULE == "count" else (stay >= R_STAY_MAX or rate < STAY_FRAC * rate0)
            if leave: next_region()
        elif was_stale:                                                   # a stale proposal failed: no K cut, no try counted; the next step takes a fresh gradient
            gate = "reject"; stale_fail += 1
            with torch.no_grad(): theta.copy_(acc["theta"])
        else:
            gate = "reject"; tries += 1
            with torch.no_grad():
                if K <= 8: tabu.view(-1)[idx] = True                           # a small set that still fails: do not propose it again here
                theta.copy_(acc["theta"])
            Kreg[rid] = max(K // 2, KMIN)
            st = Kstate.setdefault(rid, {"probe": False, "cool": False, "gain": 0.0}); st["probe"] = False; st["cool"] = True
            if tries >= R_TRIES: next_region()
    best_I = max(best_I, acc["I"])
    with torch.no_grad():
        if gate == "reject":
            prop = acc["theta"].clone().view(-1); prop[idx] = -prop[idx]; hard_eval = (prop.view(NPX, NPX)[half:, half:] >= 0).cpu().numpy()
        else:
            hard_eval = (theta[half:, half:] >= 0).cpu().numpy()
            np.savez(OUT / "best.npz", hard_oct=np.packbits(hard_eval), hard_shape=np.array(hard_eval.shape), I_tar=acc["I"], step=step)
    t2 = time.time()
    rec = {"step": step, "I_tar": I, "I_accepted": acc["I"], "binary": True, "algo": "block", "gate": gate, "region": rid,
           "K_next": Kreg.get(rid, K0), "flipped": nflip, "pred_gain": pred, "tr_ratio": ratio, "tiles_solved": n_solved,
           "t_step_s": t2 - t0, "t_adjoint_s": t_adj, "fill": float(hard_eval.mean()), "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
           "lr": 0.0, "lr_scale": 1.0, "beta": 0.0, "stale": bool(gate != "first" and rid != "-" and was_stale)}
    log.append(rec); hist_path.write_text(json.dumps(log, indent=1))
    np.savez(OUT / f"step_{step:04d}.npz", hard_eval_oct=np.packbits(hard_eval), hard_shape=np.array(hard_eval.shape), I_tar=I, I_accepted=acc["I"],
             t_abs_540=t540, cnt=cnt.detach().cpu().numpy())
    print(f"step {step:3d}  I_tar(binary) {I:.5f}  accepted {acc['I']:.5f}  gate {gate:6s}  region {rec['region']}  flipped {nflip}  pred {pred:+.2e}  ratio {ratio:+.2f}  "
          f"tiles solved {n_solved}  {t2-t0:.0f}s (adjoint {t_adj:.0f}s)  fill {rec['fill']:.3f}  peak {rec['peak_gib']:.1f} GiB", flush=True)
    torch.cuda.empty_cache(); step += 1
    if MAX_HOURS and (time.time() - T_START) / 3600 > MAX_HOURS:
        print(f"wall-clock budget {MAX_HOURS} h reached after step {step - 1} (best {best_I:.5f})", flush=True); break
print(f"FF_DONE (best accepted I_tar {best_I:.5f})", flush=True)
