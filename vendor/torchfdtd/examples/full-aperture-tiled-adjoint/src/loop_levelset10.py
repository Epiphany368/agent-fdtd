# ---------------- optimization loop v9: level set on a 10 nm design lattice, FDTD on the 20 nm lattice ----------------
# Design: a D4-symmetric binary mask Dq on the 10 nm lattice (quadrant array H10 x H10, kept transpose-symmetric).
# FDTD: unchanged 20 nm Yee lattice with DIAGONAL anisotropic permittivity (Meep-style subpixel averaging, diagonal part):
# each E component (Ex at the x-edge midpoint, Ey at the y-edge midpoint, Ez at the node, TorchFDTD field_axes) gets the SiN
# fraction f of the 20 nm window (2x2 design px) centred on it and the interface normal n from the blurred design there:
#   1/eps_xx = nx^2 <1/eps> + (1 - nx^2) / <eps>,  1/eps_yy likewise with ny,  eps_zz = <eps>  (side walls are vertical).
# The tile adjoint returns dI/df per component; a design px gets 1/4 of the derivative of each component window it lies in
# (normals held fixed), summed over tiles and D4 images -> G10 on the 10 nm representatives.
# Start: the initial width map (FF_INIT_DESIGN) rasterised on the 10 nm lattice, corners rounded (sigma FF_L9_SIGMA px), then the fab
# repair (gaps < FF_L9_GAP px filled, SiN necks FF_L9_NECK..FF_L9_MIN px widened, narrower SiN removed).
# Step: region + K trust region, 1 seed per pupil cell, the boundary moved by FF_L9_A0 px on seed patches, the patches
# re-smoothed, NEW violations repaired next to them, residual seam violations dropped with the minimal-radius rule.
_src = open(Path(os.path.abspath(__file__)).with_name("loop_block.py"), encoding="utf-8").read().split("\nKreg = {}; Kstate")[0]
exec(compile(_src, "loop_block.py(helpers)", "exec"))                      # helpers, regions, tiles, forward_score, coverage
Fn = torch.nn.functional
SUB = 2; N10 = NPX * SUB; H10 = half * SUB; MESH10 = MESH / SUB
L_MIN = int(os.environ.get("FF_L9_MIN", "6")); L_GAP = int(os.environ.get("FF_L9_GAP", "8")); L_NECK = int(os.environ.get("FF_L9_NECK", "3"))
L_SIG = float(os.environ.get("FF_L9_SIGMA", "2")); L_VSIG = float(os.environ.get("FF_L9_VSIGMA", "4")); L_BAND = int(os.environ.get("FF_L9_BAND", "6"))
L_SEED_R = int(os.environ.get("FF_L9_SEED_R", "4")); L_SEED_CAP = int(os.environ.get("FF_L9_SEED_CAP", "1")); A0 = float(os.environ.get("FF_L9_A0", "1"))
L_K0 = int(os.environ.get("FF_L9_K0", "800")); L_KMIN = int(os.environ.get("FF_L9_KMIN", "100")); PHI_CLIP = 16.0
N_SIG = float(os.environ.get("FF_L9_NORMAL_SIGMA", "1")); ANISO = os.environ.get("FF_L9_ANISO", "1") == "1"
EL_ERODE = int(os.environ.get("FF_L9_ELIG_ERODE", "4"))                  # 10 nm px kept away from the interior edge (normal-blur reach)
VANISH = os.environ.get("FF_L9_VANISH", "0") == "1"                        # SiN pieces with no L_MIN-wide part vanish (mirror of the gap fill)
NECK_CUT = os.environ.get("FF_L9_NECKCUT", "0") == "1"                     # SiN necks are cut (not widened) where the first-order gain says so
CUT_TRUST = float(os.environ.get("FF_L9_CUT_TRUST", "1"))                  # weight of cut pixels in the trust-ratio prediction (measured cut real/pred ~0.1-0.15); gate is unaffected
LAST_CUT = None                                                           # cut mask of the last repair9 call (crop coords)
START_RING = os.environ.get("FF_L9_START", "") == "ring"                  # bull's-eye seed: the width map as area-fill matched concentric rings
FILL_MATCH = os.environ.get("FF_L9_FILLMATCH", "0") == "1"                  # seed: restore the pre-repair local fill after gap filling
START_POLAR = os.environ.get("FF_L9_START", "") == "polar"                # polar-lattice seed: area-matched circular posts on rings, ~PITCH spacing
START_FILE = os.environ.get("FF_L9_START", "") == "file"                  # rule-based seed (seed_library/rule_seed.py): packed 10 nm quadrant FF_L9_SEED_FILE, fab-clean by construction
SEAM_ROUNDS = int(os.environ.get("FF_L9_SEAM_ROUNDS", "12"))              # seam-cleanup rounds before a proposal is dropped (K halved)
TOPO = os.environ.get("FF_L9_TOPO", "0") == "1"                          # v10 R2/R3: two-way topology by first-order gain (widen/drop, widen/fill, nucleation)
NUC_MAX = int(os.environ.get("FF_L9_NUC_MAX", "64")); NUC_TRUST = float(os.environ.get("FF_L9_NUC_TRUST", os.environ.get("FF_L9_CUT_TRUST", "1")))
STAY_RULE9 = os.environ.get("FF_L9_STAY_RULE", "count"); STAY_FRAC9 = float(os.environ.get("FF_L9_STAY_FRAC", "0.5")); STAY_MAX9 = int(os.environ.get("FF_L9_STAY_MAX", "12"))
TOPO_STATS = {}
SMOOTH = os.environ.get("FF_L9_SMOOTH", "blur"); L_OC = int(os.environ.get("FF_L9_OC", "4"))   # step smoothing: blur (gblur sigma L_SIG + threshold 0.5) | openclose (L_OC px disc)
A_TRUST = os.environ.get("FF_L9_ATRUST", "0") == "1"                      # v10c: the trust region also sets the boundary displacement A (px)
A_MAX = float(os.environ.get("FF_L9_A_MAX", str(L_BAND)))                  # A <= the velocity band (v is zero beyond it)
SHARE_TR = os.environ.get("FF_L9_SHARE_TR", "0") == "1"
PHASE1 = os.environ.get("FF_L9_PHASE1", "0") == "1"                        # v10d phase 1: bang-bang sign step, no fab rules (see patch_v10d.py)
P1_SIG = float(os.environ.get("FF_L9_P1_SIGMA", "3"))                      # gradient low-pass (10 nm px) before the sign                    # v10c: first visit of a region starts from the last trust state (K, A)
DUALPOL = os.environ.get("FF_L9_DUALPOL", "0") == "1"                      # Ex AND Ey tiles, full Jones per pupil cell, unpolarized objective (2x solver cost)
M20 = int(math.ceil((3 * max(L_SIG, L_VSIG) + 2 * L_MIN + L_BAND + (max(L_SEED_R, int(math.ceil(A_MAX)) + 1) if A_TRUST else L_SEED_R) + 8) / SUB)) + 2
print(f"[levelset10] v10b rules: two-way topology {TOPO} (nucleation stamps compete with boundary seeds in one K pool, trust {NUC_TRUST}), region stay rule {STAY_RULE9}" + (f" (leave below {STAY_FRAC9} x recent gain rate, max {STAY_MAX9})" if STAY_RULE9 == "rate" else f" ({R_STAY} accepts)"), flush=True)
if PHASE1: print(f"[levelset10] v10d PHASE 1: sign step on the gradient filtered at sigma {P1_SIG} px, top-K of all positive-gain flips, NO fab rules", flush=True)
if A_TRUST or SHARE_TR: print(f"[levelset10] v10c trust region: displacement A in it {A_TRUST} (A0 {A0} px, max {A_MAX} px), shared first-visit state {SHARE_TR}", flush=True)
print(f"[levelset10] step smoothing: {SMOOTH}" + (f" ({L_OC} px disc)" if SMOOTH == "openclose" else f" (sigma {L_SIG} px)"), flush=True)
print(f"[levelset10] design 10 nm / FDTD 20 nm, {'anisotropic (Meep diagonal)' if ANISO else 'isotropic'} averaging; SiN >= {L_MIN} px, "
      f"gap >= {L_GAP} px, neck >= {L_NECK} px (10 nm px); blur {L_SIG} px, velocity blur {L_VSIG} px, band {L_BAND} px, move {A0} px, "
      f"seed patch r {L_SEED_R} px, {L_SEED_CAP} seed/cell, K0 {L_K0}, pillars < {L_MIN} px {'vanish' if VANISH else 'are widened'}, necks {'cut or widened by gain' if NECK_CUT else 'widened'}", flush=True)

# ---- image operators (10 nm) ----
def gblur(x, s):
    if s <= 0: return x.float()
    r = int(math.ceil(3 * s)); t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32); k = torch.exp(-t ** 2 / (2 * s * s)); k = k / k.sum()
    y = Fn.conv2d(x.float()[None, None], k.view(1, 1, -1, 1), padding=(r, 0))
    return Fn.conv2d(y, k.view(1, 1, 1, -1), padding=(0, r))[0, 0]
def _disk(k):                                                             # k=3 plus sign, 4: 4x4 minus corners, 6/8 round
    t = torch.arange(k, device=dev, dtype=torch.float32) - (k - 1) / 2
    return ((t[:, None] ** 2 + t[None, :] ** 2) <= (k / 2) ** 2 - 0.5).float()
def _open_disk(x, k):                                                     # binary opening, exact anchor (float in, count out; > 0.5 kept)
    K = _disk(k); s_ = float(K.sum())
    e = (Fn.conv2d(x[None, None], K[None, None]) >= s_ - 0.5).float()
    return Fn.conv2d(Fn.pad(e, (k - 1, k - 1, k - 1, k - 1)), K.flip(0, 1)[None, None])[0, 0]
def _dil(m, r): return Fn.max_pool2d(m.float()[None, None], 2 * r + 1, stride=1, padding=r)[0, 0] > 0.5 if r > 0 else m.clone()
def _ero(m, r): return ~_dil(~m, r) if r > 0 else m.clone()
def boundary10(mf):
    b = torch.zeros_like(mf)
    b[1:, :] |= mf[1:, :] != mf[:-1, :]; b[:-1, :] |= mf[:-1, :] != mf[1:, :]; b[:, 1:] |= mf[:, 1:] != mf[:, :-1]; b[:, :-1] |= mf[:, :-1] != mf[:, 1:]
    return b
def thin9(m):                                                             # SiN narrower than L_MIN, air gaps narrower than L_GAP
    x = m.float(); return (m & ~(_open_disk(x, L_MIN) > 0.5)) | (~m & ~(_open_disk(1 - x, L_GAP) > 0.5))
def drop_small9(x):                                                       # remove SiN components (8-conn) that contain no L_MIN-wide part
    import scipy.ndimage as ndi
    wide = (_open_disk(x.float(), L_MIN) > 0.5).cpu().numpy()
    lab, n = ndi.label(x.cpu().numpy(), structure=np.ones((3, 3), int))
    has = np.zeros(n + 1, bool); has[np.unique(lab[wide])] = True; has[0] = False
    return torch.tensor(has[lab], device=x.device)
def cut9(x, neck, dI):                                                    # SiN to remove where cutting a neck beats widening it (first-order dI/dD)
    import scipy.ndimage as ndi                                           # cut set = neck stamped with a (L_GAP - 1) px disk, so the merged air waist is >= L_GAP wide
    K = _disk(L_GAP - 1); r = (L_GAP - 2) // 2
    R = x & (Fn.conv2d(neck.float()[None, None], K[None, None], padding=r)[0, 0] > 0.5)
    lab, n = ndi.label(R.cpu().numpy(), structure=np.ones((3, 3), int))
    if n == 0: return torch.zeros_like(neck)
    d = dI.float().cpu().numpy(); nk = neck.cpu().numpy(); ring = (_dil(neck, 1) & ~x).cpu().numpy(); idx = np.arange(1, n + 1)
    cutg = -ndi.sum(d, lab, idx); widg = ndi.sum(d * ring, ndi.grey_dilation(np.where(nk, lab, 0), size=(3, 3)), idx)
    has_neck = ndi.sum(nk, lab, idx) > 0
    return torch.tensor(np.isin(lab, idx[has_neck & (cutg > widg)]), device=neck.device)
def _disk_np(k):
    t = np.arange(k) - (k - 1) / 2; return (t[:, None] ** 2 + t[None, :] ** 2) <= (k / 2) ** 2 - 0.5
def topo_choice9(x, dI, sin):
    """v10 R3: a sub-minimum part goes the way its first-order gain says, not one fixed direction.
    sin=True : SiN components with no L_MIN-wide part -> widened into air (1..3 px) until an L_MIN disc fits, or removed.
    sin=False: air narrower than L_GAP (small holes, pinched gaps) -> widened into SiN (1..3 px) until an L_GAP disc fits, or filled.
    dI = dI/dD with D = 1 for SiN: adding a SiN px gains +dI, removing one gains -dI."""
    import scipy.ndimage as ndi
    ph = x if sin else ~x; lm = L_MIN if sin else L_GAP
    wide = _open_disk(ph.float(), lm) > 0.5; phn = ph.cpu().numpy()
    if sin:                                                               # whole components only; necks of wider parts stay with cut9 / widening
        lab0, n0 = ndi.label(phn, structure=np.ones((3, 3), int))
        has = np.zeros(n0 + 1, bool); has[np.unique(lab0[wide.cpu().numpy()])] = True; has[0] = True; T = ~has[lab0]
    else:
        T = phn & ~wide.cpu().numpy()
    lab, n = ndi.label(T, structure=np.ones((3, 3), int))
    if n == 0: return x
    key = "sin" if sin else "air"; ids = np.arange(1, n + 1); d = dI.float().cpu().numpy(); sg = 1.0 if sin else -1.0
    g_drop = -sg * ndi.sum(d, lab, ids)                                   # post removed: -sum dI ; thin air filled: +sum dI
    best_k = np.zeros(n, int); g_wid = np.full(n, -np.inf)
    for k in (1, 2, 3):
        Lk = ndi.grey_dilation(lab, footprint=_disk_np(2 * k + 1)); grow = (Lk > 0) & ~phn
        ok = (_open_disk(torch.tensor(phn | grow, device=x.device).float(), lm) > 0.5).cpu().numpy()
        legal = ndi.maximum(ok.astype(np.int8), lab, ids) > 0
        g = sg * ndi.sum(d * grow, Lk, ids); new = legal & (best_k == 0); best_k[new] = k; g_wid[new] = g[new]
    widen = (best_k > 0) & (g_wid > g_drop)
    out = phn.copy(); out[np.isin(lab, ids[~widen])] = False
    for k in (1, 2, 3):
        sel = ids[widen & (best_k == k)]
        if len(sel): out |= ndi.grey_dilation(np.where(np.isin(lab, sel), lab, 0), footprint=_disk_np(2 * k + 1)) > 0
    TOPO_STATS[key + "_widen"] = TOPO_STATS.get(key + "_widen", 0) + int(widen.sum())
    TOPO_STATS[key + "_drop"] = TOPO_STATS.get(key + "_drop", 0) + int((~widen).sum())
    res = torch.tensor(out, device=x.device)
    return res if sin else ~res
def nuc_peaks9(q, flipg, allowed, sin):
    """Candidate centres of a minimum post (sin) / hole (not sin) on the crop mask q (numpy bool): fab-legal clearance, positive
    summed first-order gain, peaks >= one stamp + clearance apart. Returns (rows, cols, stamp gain, stamp size k)."""
    import scipy.ndimage as ndi
    k, clr = (L_MIN, L_GAP) if sin else (L_GAP, L_MIN)
    free = ~q if sin else q
    dist = torch.tensor(ndi.distance_transform_edt(free), device=flipg.device, dtype=torch.float32)
    K = _disk(k); p0 = k // 2
    f = flipg.float() * torch.tensor(free, device=flipg.device).float()
    gsum = Fn.conv2d(Fn.pad(f[None, None], (p0, k - 1 - p0, p0, k - 1 - p0)), K[None, None])[0, 0]
    okc = (dist >= k / 2 + clr + 1.5) & _ero(allowed, k) & (gsum > 0)
    R = k + clr
    mx = Fn.max_pool2d(torch.where(okc, gsum, torch.full_like(gsum, -1e30))[None, None], 2 * R + 1, stride=1, padding=R)[0, 0]
    pi, pj = torch.nonzero(okc & (gsum >= mx), as_tuple=True)
    return pi, pj, gsum[pi, pj], k
def nucleate9(cs, flipg, allowed, nmax=None):
    """v10 R2: new minimum posts in open air and new minimum holes in solid SiN where the stamp's summed first-order gain is
    positive (boundary moves alone never create a feature). Fab-legal by construction: an L_MIN px post >= L_GAP px from all SiN,
    an L_GAP px hole with >= L_MIN px of SiN around it, stamps >= one stamp + clearance apart; at most NUC_MAX of each per step."""
    import scipy.ndimage as ndi
    out = cs.clone(); nuc = torch.zeros_like(cs); q = cs.cpu().numpy(); counts = []
    for ii_, sin in enumerate((True, False)):
        pi, pj, gv, k = nuc_peaks9(q, flipg, allowed, sin); p0 = k // 2; K = _disk(k)
        cap = NUC_MAX if nmax is None else int(nmax[ii_])
        if pi.numel() > cap: top = torch.topk(gv, cap).indices; pi, pj = pi[top], pj[top]
        st = K > 0.5
        for a, b in zip(pi.tolist(), pj.tolist()):
            sl = (slice(a - p0, a - p0 + k), slice(b - p0, b - p0 + k))
            if sin: out[sl] |= st
            else: out[sl] &= ~st
            nuc[sl] |= st
        counts.append(int(pi.numel()))
    return out, nuc, tuple(counts)
def repair9(m, gain=None, dI=None):                                       # fill gaps < L_GAP; [VANISH: drop pillars < L_MIN]; [NECK_CUT+dI: cut necks that gain]; widen other SiN necks L_NECK..L_MIN; drop narrower SiN
    if TOPO and dI is not None:                                           # v10: thin air widens or fills, sub-L_MIN posts widen or drop, by gain
        x = topo_choice9(topo_choice9(m, dI, sin=False), dI, sin=True)
    else:
        x = ~(_open_disk((~m).float(), L_GAP) > 0.5)
        if VANISH: x = drop_small9(x)
    thin = x & ~(_open_disk(x.float(), L_MIN) > 0.5); neck = thin & (_open_disk(x.float(), L_NECK) > 0.5)
    if NECK_CUT and dI is not None and bool(neck.any()):
        global LAST_CUT
        cut = cut9(x, neck, dI); x = x & ~cut; neck = neck & ~cut; LAST_CUT = cut if LAST_CUT is None else (LAST_CUT | cut)
    if bool(neck.any()):
        ring = _dil(neck, 1) & ~x
        if gain is not None:
            x = x | (ring & (gain > 0)); still = neck & ~(_open_disk(x.float(), L_MIN) > 0.5); x = x | (_dil(still, 1) & ~x)
        else:
            x = x | ring
    x = x & (_open_disk(x.float(), L_MIN) > 0.5)
    return ~(_open_disk((~x).float(), L_GAP) > 0.5)
def openclose9(m, k):                                                     # shape-preserving smoothing: opening then closing with a k px disc. Discs of
    x = _open_disk(m.float(), k) > 0.5                                    # diameter >= k are unchanged (blur + threshold shrinks a 80 nm post to 64 nm,
    return ~(_open_disk((~x).float(), k) > 0.5)                           # a 100 nm post to 93 nm per application; measured 2026-09-25)
def sdf10(m):                                                             # clipped signed distance (px) of a crop mask (CPU EDT)
    import scipy.ndimage as ndi
    q = m.cpu().numpy(); sd = np.where(q, ndi.distance_transform_edt(q), -ndi.distance_transform_edt(~q)) - 0.5 * np.where(q, 1, -1)
    return torch.tensor(np.clip(sd, -PHI_CLIP, PHI_CLIP), dtype=torch.float32, device=dev)

# ---- lattices and symmetry ----
def qidx(k): return torch.where(k >= H10, k - H10, H10 - 1 - k)           # 10 nm global index -> quadrant index
def d4q(Q, I0, I1, J0, J1, fill=0):                                       # D4 image of the symmetric quadrant Q on global rows I0:I1, cols J0:J1
    ka = torch.arange(I0, I1, device=dev); kb = torch.arange(J0, J1, device=dev)
    out = Q[qidx(ka.clamp(0, N10 - 1))[:, None], qidx(kb.clamp(0, N10 - 1))[None, :]]
    ok = ((ka >= 0) & (ka < N10))[:, None] & ((kb >= 0) & (kb < N10))[None, :]
    return torch.where(ok, out, torch.full_like(out, fill))
def _ax20(r): return torch.where(r >= half, r, NPX - 1 - r)
def d4c20(x, i0, i1, j0, j1):                                             # D4 image of a 20 nm representative array
    a = _ax20(torch.arange(i0, i1, device=dev))[:, None]; b = _ax20(torch.arange(j0, j1, device=dev))[None, :]
    return x[torch.maximum(a, b), torch.minimum(a, b)]
kq = torch.arange(H10, device=dev)
def symmetrize(Q): return torch.where(kq[:, None] >= kq[None, :], Q, Q.T)
C4 = os.environ.get("FF_SYM", "D4").upper() == "C4"                     # FF_SYM=C4 (2026-09-27): quadrant design, lens = its 90 deg rotations
if C4:
    def c4rep(u, v):                                                      # px / cell offsets from the centre (u = x index, v = y index, <0 allowed) ->
        u, v = torch.broadcast_tensors(u, v)                              #  (a, b) in the first quadrant with p = R^k (a, b), R = +90 deg about z,
        q2 = (u < 0) & (v >= 0); q3 = (u < 0) & (v < 0); q4 = (u >= 0) & (v < 0)   #  (x, y) -> (-y, x); odd = k odd (Q2: k 1, Q3: 2, Q4: 3)
        a = torch.where(q2, v, torch.where(q3, -1 - u, torch.where(q4, -1 - v, u)))
        b = torch.where(q2, -1 - u, torch.where(q3, -1 - v, torch.where(q4, u, v)))
        return a, b, q2 | q4
    def d4q(Q, I0, I1, J0, J1, fill=0):                                   # (C4) rotation image of the quadrant Q on global rows I0:I1, cols J0:J1
        ka = torch.arange(I0, I1, device=dev); kb = torch.arange(J0, J1, device=dev)
        a, b, _ = c4rep((ka.clamp(0, N10 - 1) - H10)[:, None], (kb.clamp(0, N10 - 1) - H10)[None, :])
        out = Q[a, b]; ok = ((ka >= 0) & (ka < N10))[:, None] & ((kb >= 0) & (kb < N10))[None, :]
        return torch.where(ok, out, torch.full_like(out, fill))
    def symmetrize(Q): return Q                                           # (C4) no transpose symmetry
_oq = oct_mask[half:, half:]; AP10 = (_oq | _oq.T).repeat_interleave(SUB, 0).repeat_interleave(SUB, 1); del _oq
MULTQ = None                                                              # D4 multiplicity of a quadrant representative (4 diagonal, 8 else), on demand
Dq = None

# ---- the FDTD view of the design: per-component fraction and normal on each tile window ----
PADN = int(math.ceil(3 * N_SIG)) + 1
def tile_q10(i0, j0):
    """(nx, ny, 5) = (f_Ex, f_Ey, f_Ez, nx^2 at Ex, ny^2 at Ey) of the 20 nm tile window starting at lattice (i0, j0)."""
    nx, ny = shape[0], shape[1]; A = 2 * i0 - 1 - PADN; B = 2 * j0 - 1 - PADN
    D = d4q(Dq, A, 2 * (i0 + nx) + 1 + PADN, B, 2 * (j0 + ny) + 1 + PADN).float(); Ds = gblur(D, N_SIG)
    def win(oa, ob):
        sub = D[oa: oa + 2 * nx, ob: ob + 2 * ny]; s = Ds[oa: oa + 2 * nx, ob: ob + 2 * ny]
        f = Fn.avg_pool2d(sub[None, None], 2)[0, 0]
        gx = Fn.avg_pool2d((s[1::2, :] - s[0::2, :])[None, None], (1, 2))[0, 0]; gy = Fn.avg_pool2d((s[:, 1::2] - s[:, 0::2])[None, None], (2, 1))[0, 0]
        return f, gx, gy
    fx, gxx, gyx = win(1 + PADN, PADN); fy, gxy, gyy = win(PADN, 1 + PADN); fz, _, _ = win(PADN, PADN)
    nx2 = gxx ** 2 / (gxx ** 2 + gyx ** 2 + 1e-12); ny2 = gyy ** 2 / (gxy ** 2 + gyy ** 2 + 1e-12)
    return torch.stack([fx, fy, fz, nx2, ny2], -1)
def rho_tile_of(theta_, i0, j0): return tile_q10(i0, j0)                   # (patched) what the solver and the cache hash see
_EPS_S = globals().get("N_SIN", 2.0) ** 2
def eps_of(q):                                                            # (patched) (nx,ny,5) -> diagonal Yee eps (nx,ny,nz,3)
    if q.dim() == 2: q = torch.stack([q, q, q, torch.full_like(q, 0.5), torch.full_like(q, 0.5)], -1)
    def comp(f, n2):
        ea = 1.0 + f * (_EPS_S - 1.0); ih = (1.0 - f) + f / _EPS_S
        return 1.0 / (n2 * ih + (1.0 - n2) / ea) if ANISO else ea
    lay = torch.stack([comp(q[..., 0], q[..., 3].detach()), comp(q[..., 1], q[..., 4].detach()), 1.0 + q[..., 2] * (_EPS_S - 1.0)], -1)   # (nx,ny,3)
    eps = torch.ones(shape + (3,), device=dev, dtype=torch.float32)
    eps = torch.where(sub[None, None, :, None], torch.full_like(eps, N_SIO2 ** 2), eps)
    eps = torch.where(pil[None, None, :, None], lay[:, :, None, :].expand(shape + (3,)), eps)
    return eps.contiguous()
if "ReversibleCPMLPlaneSimulation" in globals():                         # real run: diagonal material maps (3 components)
    models.clear(); EPS_BG = None
    def model_for(key):
        if key not in models:
            p, counts = tile_project(*key); models[key] = shared_model(p, counts, 3)   # 2026-09-30: one diagonal-eps model per local geometry (ff_opt.py)
        return models[key]
JC = ("xx", "yx", "xy", "yy")                                            # T_ab: b-polarized input -> a-polarized output
def _jones_expand(Toct):
    """Toct: dict xx,xy,yx,yy -> (Nl,NCELL,NCELL) valid on octant cells (x index >= y index >= centre); returns full-pupil dict.
    Axis mirror x->-x or y->-y flips the off-diagonal sign; the diagonal mirror swaps x<->y (xx<->yy, xy<->yx)."""
    ii = torch.arange(NCELL, device=dev); c = NCELL // 2
    a = torch.where(ii >= c, ii, NCELL - 1 - ii); sg = torch.where(ii >= c, 1.0, -1.0).to(dev)
    A = a[:, None].expand(NCELL, NCELL); B = a[None, :].expand(NCELL, NCELL); sw = A < B
    hi, lo = torch.maximum(A, B), torch.minimum(A, B); s_off = (sg[:, None] * sg[None, :])
    g = {k: v[:, hi, lo] for k, v in Toct.items()}
    return {"xx": torch.where(sw, g["yy"], g["xx"]), "yy": torch.where(sw, g["xx"], g["yy"]),
            "xy": torch.where(sw, g["yx"], g["xy"]) * s_off, "yx": torch.where(sw, g["xy"], g["yx"]) * s_off}
if C4:
    def _jones_expand(Toct):
        """(C4) Toct valid on the first-quadrant cells (x, y index >= centre) -> full pupil. A C4-invariant structure at normal incidence:
        E_{R e}(R p) = R E_e(p), so the Jones matrix T = [E_x-in | E_y-in] obeys T(R p) = R T(p) R^-1 with R = [[0, -1], [1, 0]]:
        k odd (Q2, Q4): xx' = yy, yy' = xx, xy' = -yx, yx' = -xy; k = 2 (Q3): T unchanged. No mirror is used."""
        ii = torch.arange(NCELL, device=dev); c = NCELL // 2
        a, b, odd = c4rep((ii - c)[:, None], (ii - c)[None, :]); a = a + c; b = b + c
        g = {k: v[:, a, b] for k, v in Toct.items()}
        return {"xx": torch.where(odd, g["yy"], g["xx"]), "yy": torch.where(odd, g["xx"], g["yy"]),
                "xy": torch.where(odd, -g["yx"], g["xy"]), "yx": torch.where(odd, -g["xy"], g["yx"])}
if DUALPOL and "ReversibleCPMLPlaneSimulation" in globals():            # real run: Ex and Ey tiles, full Jones objective
    def i_tar_jones(Tfull):
        if FA_OBJECTIVE:                                                 # 2026-09-30: FA objective; the loop saves the Jones maps (FA_LAST_T), not the objective
            FA_LAST_T.clear(); FA_LAST_T.update({k: v.detach() for k, v in Tfull.items()})
            return fa_objective()(Tfull)
        from jones_objectives import optical_objective
        return optical_objective(Tfull,kind=os.environ['FF_NEW_OBJECTIVE'],lam_nm=LAM_NM,pitch=PITCH,out=OUT)
    FA_LAST_T = {}
    project_y = project.model_copy(deep=True); project_y.sources[0].component = "Ey"; project_y = Project.model_validate(project_y.model_dump())
    models_y = {}
    def model_y_for(key):
        global project
        if key not in models_y:
            p0 = project; project = project_y                                # every Ey model is kept, like the Ex models (was rebuilt per tile)
            try: p, counts = tile_project(*key)
            finally: project = p0
            models_y[key] = shared_model(p, counts, 3)                    # 2026-09-30: shared per local geometry (Ey source is part of the key)
        return models_y[key]
    _LO = None
    def _solve_pol(q, key, pol):                                          # exit-plane (Ex, Ey) for a pol-polarized plane wave, LAM_NM ascending
        global _LO, FA_LAST_REPORT
        if _LO is None: _LO = torch.as_tensor(lam_order.copy(), device=dev)
        m = model_for(key) if pol == "x" else model_y_for(key)
        det = m(eps_of(q), FREQ_T.to(dev), fixed_epsilon=eps_background(), block_size=32)["xy"]; FA_LAST_REPORT = det.report   # solver timings (phase log)
        return det.fields[:, :, 0][_LO], det.fields[:, :, 1][_LO], torch.as_tensor(det.points_um, device=dev, dtype=torch.float32)
    def cache_path(key, h): return CACHE_DIR / f"tile_{key[0]:.0f}_{key[1]:.0f}_{h}{globals().get('NF_TAG', '')}_dp.npz"
    def refresh(keys):
        for key in keys:
            i0, j0 = tile_window(*key)
            with torch.no_grad():
                q = rho_tile_of(theta, i0, j0); f = cache_path(key, tile_hash(q))
                if f.exists():
                    d = np.load(f); cache[key] = ({k: torch.tensor(d[k], device=dev) for k in JC}, torch.tensor(d["pts"], device=dev)); continue
                xx, yx, pts = _solve_pol(q, key, "x"); xy, yy, _ = _solve_pol(q, key, "y")
                fl = {"xx": xx, "yx": yx, "xy": xy, "yy": yy}; cache[key] = (fl, pts)
                np.savez(f, pts=pts.cpu().numpy(), **{k: v.cpu().numpy() for k, v in fl.items()})
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    def _norm(a, cnt): return a / cnt.clamp(min=1)[None] / E_ref[:, None, None] * JONES_SCALE
    def forward_score():
        cache.clear(); refresh(tiles)
        acc_ = {k: torch.zeros((len(LAM_NM), NCELL, NCELL), dtype=torch.complex64, device=dev) for k in JC}; cnt = torch.zeros((NCELL, NCELL), device=dev)
        for key in tiles:
            fl, pts = cache[key]
            for k in JC:
                a, c = tile_cells(key, fl[k], pts); acc_[k] = acc_[k] + a
            cnt = cnt + c
        leaf = {k: _norm(v, cnt).detach().requires_grad_(True) for k, v in acc_.items()}
        I = i_tar_jones(_jones_expand(leaf)); I.backward()
        g_oct = torch.stack([leaf[k].grad for k in JC], 0)                 # (4, Nl, NCELL, NCELL), octant cells only
        t540 = _jones_expand({k: v.detach() for k, v in leaf.items()})["xx"][4].abs().cpu().numpy()
        return float(I.detach()), g_oct, cnt, t540
    def tile_grad(key, g_oct, cnt):
        """dI/d(tile window) by two reversible adjoints (Ex source: T_xx, T_yx; Ey source: T_xy, T_yy)."""
        i0, j0 = tile_window(*key)
        with torch.no_grad(): leaf = rho_tile_of(theta, i0, j0).clone()
        leaf.requires_grad_(True)
        for pol, (ka, kb) in (("x", ("xx", "yx")), ("y", ("xy", "yy"))):
            fa, fb, pts = _solve_pol(leaf, key, pol)
            oa = _norm(tile_cells(key, fa, pts)[0], cnt); ob = _norm(tile_cells(key, fb, pts)[0], cnt)
            torch.autograd.backward([oa, ob], [g_oct[JC.index(ka)], g_oct[JC.index(kb)]])
            del fa, fb, oa, ob
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return leaf.grad.detach()
    print("[levelset10] DUAL POLARIZATION: Ex and Ey tiles, full Jones (unpolarized) objective", flush=True)
def region_gradient10(keys, g_oct, cnt):
    """dI/dD of each 10 nm representative (all D4 images, all tiles): each design px gets 1/4 of dI/df of the Ex, Ey and Ez windows it lies in."""
    G = torch.zeros((H10, H10), device=dev)
    for k in keys:
        g = tile_grad(k, g_oct, cnt); i0, j0 = tile_window(*k); nx, ny = shape[0], shape[1]
        if "own_grad" in globals(): g = own_grad(g)                        # FF_OWNGRAD=1: margin px zeroed (ff_opt.py; identity by default)
        if g.dim() == 2: g = torch.stack([g, g, g], -1) / 3                # mock: one density
        buf = torch.zeros((2 * nx + 2, 2 * ny + 2), device=dev)              # 10 nm px rows 2*i0-1 .. 2*(i0+nx), same for cols
        up = lambda c: g[..., c].float().repeat_interleave(2, 0).repeat_interleave(2, 1) / 4
        buf[1: 1 + 2 * nx, 0: 2 * ny] += up(0); buf[0: 2 * nx, 1: 1 + 2 * ny] += up(1); buf[0: 2 * nx, 0: 2 * ny] += up(2)
        ka = torch.arange(2 * i0 - 1, 2 * (i0 + nx) + 1, device=dev); kb = torch.arange(2 * j0 - 1, 2 * (j0 + ny) + 1, device=dev)
        ok = ((ka >= 0) & (ka < N10))[:, None] & ((kb >= 0) & (kb < N10))[None, :]
        qa = qidx(ka.clamp(0, N10 - 1))[:, None].expand_as(buf); qb = qidx(kb.clamp(0, N10 - 1))[None, :].expand_as(buf)
        hi, lo = torch.maximum(qa, qb)[ok], torch.minimum(qa, qb)[ok]
        G.index_put_((hi, lo), buf[ok], accumulate=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return G                                                              # valid on hi >= lo (quadrant lower triangle)
if C4:
    def region_gradient10(keys, g_oct, cnt):
        """(C4) as above, each global 10 nm px of a tile window goes to its first-quadrant representative by rotation (whole quadrant valid)."""
        G = torch.zeros((H10, H10), device=dev)
        for k in keys:
            g = tile_grad(k, g_oct, cnt); i0, j0 = tile_window(*k); nx, ny = shape[0], shape[1]
            if "own_grad" in globals(): g = own_grad(g)
            if g.dim() == 2: g = torch.stack([g, g, g], -1) / 3
            buf = torch.zeros((2 * nx + 2, 2 * ny + 2), device=dev)
            up = lambda c: g[..., c].float().repeat_interleave(2, 0).repeat_interleave(2, 1) / 4
            buf[1: 1 + 2 * nx, 0: 2 * ny] += up(0); buf[0: 2 * nx, 1: 1 + 2 * ny] += up(1); buf[0: 2 * nx, 0: 2 * ny] += up(2)
            ka = torch.arange(2 * i0 - 1, 2 * (i0 + nx) + 1, device=dev); kb = torch.arange(2 * j0 - 1, 2 * (j0 + ny) + 1, device=dev)
            ok = ((ka >= 0) & (ka < N10))[:, None] & ((kb >= 0) & (kb < N10))[None, :]
            qa, qb, _ = c4rep((ka.clamp(0, N10 - 1) - H10)[:, None], (kb.clamp(0, N10 - 1) - H10)[None, :])
            G.index_put_((qa[ok], qb[ok]), buf[ok], accumulate=True)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return G

def ring_seed_q10(Wnm):
    """Bull's-eye seed: concentric SiN rings on the PITCH radial period; ring k's width is area-fill matched to the width map's
    azimuthal mean w_k in that annulus (w_ring = w_k^2 / PITCH), clipped to the fab window [L_MIN, PITCH - L_GAP] (10 nm px)."""
    n = Wnm.shape[0]; c = (n - 1) / 2; ii = np.arange(n) - c; rr = np.hypot(ii[:, None], ii[None, :]) * PITCH
    kk = np.floor(rr / PITCH).astype(int); nk = int(kk.max()) + 1
    wk = np.bincount(kk.ravel(), Wnm.ravel().astype(np.float64), nk) / np.maximum(np.bincount(kk.ravel(), None, nk), 1)
    wr = np.clip(wk ** 2 / (PITCH * 1e3), 10.0 * L_MIN, PITCH * 1e3 - 10.0 * L_GAP) * 1e-3
    wr_t = torch.tensor(wr, device=dev, dtype=torch.float64)
    xs = XG0 + (torch.arange(H10, N10, device=dev, dtype=torch.float64) + 0.5) * MESH10
    Q = torch.zeros((H10, H10), dtype=torch.bool, device=dev)
    for r0 in range(0, H10, 1500):
        r1 = min(r0 + 1500, H10); r = torch.hypot(xs[r0:r1][:, None], xs[None, :]); k = torch.floor(r / PITCH).long().clamp(max=nk - 1)
        Q[r0:r1] = (r - (k.double() + 0.5) * PITCH).abs() <= wr_t[k] / 2
    return Q
def polar_seed_q10(Wnm):
    """Polar-lattice seed: circular SiN posts on rings r_k = k * PITCH (one post at the centre), n_k posts per
    ring = the multiple of 4 nearest 2 pi r_k / PITCH (D4-compatible, azimuthal spacing ~ PITCH), post k,j at angle j 2 pi / n_k;
    diameter d = 2 w / sqrt(pi) (same area, i.e. same SiN fill, as the square post of width w of the width map at that point).
    Each 10 nm px is tested against its nearest post (nearest ring, nearest angle), which is exact while d < PITCH."""
    n = Wnm.shape[0]; c = (n - 1) / 2; Wt = torch.tensor(Wnm * 1e-3, dtype=torch.float64, device=dev)
    xs = XG0 + (torch.arange(H10, N10, device=dev, dtype=torch.float64) + 0.5) * MESH10
    Q = torch.zeros((H10, H10), dtype=torch.bool, device=dev)
    for r0 in range(0, H10, 1000):
        r1 = min(r0 + 1000, H10); X = xs[r0:r1][:, None].expand(-1, H10); Y = xs[None, :].expand(r1 - r0, -1)
        r = torch.hypot(X, Y); ph = torch.atan2(Y, X)
        k = torch.round(r / PITCH); rk = k * PITCH
        nk = torch.clamp(torch.round(2 * math.pi * k / 4) * 4, min=4)
        j = torch.round(ph * nk / (2 * math.pi)); pj = j * 2 * math.pi / nk
        cx = torch.where(k > 0, rk * torch.cos(pj), torch.zeros_like(rk)); cy = torch.where(k > 0, rk * torch.sin(pj), torch.zeros_like(rk))
        ci = torch.round(cx / PITCH + c).long().clamp(0, n - 1); cj = torch.round(cy / PITCH + c).long().clamp(0, n - 1)
        f = (Wt[ci, cj] / PITCH) ** 2                                     # SiN fill of the square post of the width map at that site
        dist = torch.hypot(X - cx, Y - cy)
        d_post = 2 * PITCH * torch.sqrt(f / math.pi); d_hole = 2 * PITCH * torch.sqrt((1 - f).clamp(min=0) / math.pi)
        Q[r0:r1] = torch.where(f <= 0.5, dist <= d_post / 2, dist > d_hole / 2)   # fill > 1/2: a circular air hole at the site instead
    return Q
def webcut9(x, rounds=6):
    """Seed cleanup: SiN webs thinner than L_MIN left between merged posts are cut with an (L_GAP - 1) px disk stamp (the merged
    air is then >= L_GAP wide), then repair9 again; gap fill would otherwise re-close the web (fixed-point oscillation)."""
    K = _disk(L_GAP - 1); r = (L_GAP - 2) // 2
    for _ in range(rounds):
        th = thin9(x)
        if not bool(th.any()): break
        stamp = Fn.conv2d((th & x).float()[None, None], K[None, None], padding=r)[0, 0] > 0.5
        x = repair9(x & ~stamp)
    return x
def _box(v, k):                                                           # k x k box mean, same size (edge-replicated)
    p = k // 2; return Fn.avg_pool2d(Fn.pad(v.float()[None, None], (p, p, p, p), mode="replicate"), k, stride=1)[0, 0]
def fill_match9(x, target, iters=3, cell=29):
    """Restore the local SiN fill (box mean over one pupil cell) of the pre-repair design: gap filling adds SiN, so the boundary is
    moved into the SiN by delta = excess fill / boundary density (px), i.e. holes grow and posts thin; then repair9 + webcut9."""
    Ft = _box(target, cell)
    for _ in range(iters):
        F = _box(x, cell); per = _box(boundary10(x).float() * 0.5, cell)
        delta = torch.clamp((F - Ft) / per.clamp(min=1e-3), -4.0, 4.0)
        x = webcut9(repair9(sdf10(x) > delta))
    return x
def start_design():
    """The initial width map (FF_INIT_DESIGN) rasterised on the 10 nm quadrant (FF_L9_START=ring: as concentric rings), corners rounded, fab repair;
    mock: the warm mask x2."""
    if START_FILE:
        _z = np.load(os.environ["FF_L9_SEED_FILE"]); assert int(_z["H10"]) == H10, "seed lattice mismatch"
        Q = torch.tensor(np.unpackbits(_z["Q_bits"])[:H10 * H10].reshape(H10, H10).astype(bool), device=dev); del _z
    elif "W_INIT_NM" in globals() and START_RING:
        Q = ring_seed_q10(W_INIT_NM)
    elif "W_INIT_NM" in globals() and START_POLAR:
        Q = polar_seed_q10(W_INIT_NM)
    elif "W_INIT_NM" in globals():
        Wt = torch.tensor(W_INIT_NM * 1e-3, dtype=torch.float32, device=dev); n = W_INIT_NM.shape[0]; c = (n - 1) / 2
        xs = XG0 + (torch.arange(H10, N10, device=dev, dtype=torch.float64) + 0.5) * MESH10
        ci = torch.round(xs / PITCH + c).long(); inb = (ci >= 0) & (ci < n); cic = ci.clamp(0, n - 1)
        dx = (xs - (ci.double() - c) * PITCH).abs().float()
        Q = torch.zeros((H10, H10), dtype=torch.bool, device=dev)
        for r0 in range(0, H10, 1500):
            r1 = min(r0 + 1500, H10); w = Wt[cic[r0:r1][:, None], cic[None, :]]
            Q[r0:r1] = (dx[r0:r1][:, None] <= w / 2) & (dx[None, :] <= w / 2) & inb[r0:r1][:, None] & inb[None, :]
    else:
        Q = (theta.detach()[half:, half:] >= 0); Q = (Q | Q.T).repeat_interleave(SUB, 0).repeat_interleave(SUB, 1)
    Q &= AP10; raw = Q.clone(); r = int(math.ceil(3 * max(L_SIG, 1.0))) + L_GAP
    x = Q; x = torch.cat([x[:r].flip(0), x], 0); x = torch.cat([x[:, :r].flip(1), x], 1)         # mirror images across the two axes
    x = torch.cat([x, torch.zeros((r, x.shape[1]), dtype=torch.bool, device=dev)], 0); x = torch.cat([x, torch.zeros((x.shape[0], r), dtype=torch.bool, device=dev)], 1)
    x0raw = x.clone()
    if L_SIG > 0 and not START_FILE: x = gblur(x, L_SIG) >= 0.5            # corner rounding at 10 nm (the file seed is already circular)
    x = repair9(x)                                                        # gaps < L_GAP filled, necks widened / removed
    if START_POLAR or START_RING: x = webcut9(x)                           # webs between merged posts: cut, not re-closed
    if FILL_MATCH: x = fill_match9(x, x0raw)                              # holes grow / posts thin back to the pre-repair local fill
    Q = symmetrize(x[r: r + H10, r: r + H10] & AP10)
    print(f"[levelset10] start: {int(raw.sum())} -> {int(Q.sum())} SiN px (quadrant), rounding + repair changed {int((Q != raw).sum())} px", flush=True)
    return Q

# ---- state ----
if acc is not None and ACC_PATH.exists():
    _s = torch.load(ACC_PATH, map_location="cpu", weights_only=False)
    if _s.get("Dq_bits") is not None:
        Dq = torch.tensor(np.unpackbits(_s["Dq_bits"])[:H10 * H10].reshape(H10, H10).astype(bool), device=dev); acc["Dq_bits"] = _s["Dq_bits"]
        print("[levelset10] resumed the 10 nm design from acc_state", flush=True)
    del _s
if Dq is None: acc = None
SELq = torch.zeros((H10, H10), dtype=torch.bool, device=dev); FLq = torch.zeros((H10, H10), dtype=torch.bool, device=dev)
def pack_q(Q): return np.packbits(Q.cpu().numpy())
Areg = {}; part = 0; reg_i = 0; tries = 0; stay = 0; idle = 0; step = len(log); Dreg = {}; TR_LAST = None
for h in reversed(log):
    if h.get("algo") == "levelset10" and h.get("region", "-") != "-":
        part, reg_i = int(h["region"][1]), int(h["region"].split("r")[1]); Areg[h["region"]] = int(h.get("K_next", L_K0)); Dreg[h["region"]] = float(h.get("A_next", A0))
        _acc_seen = False                                                 # restore this region visit's accept / reject counters (they were reset by a relaunch)
        for h2 in reversed(log):
            if h2.get("algo") != "levelset10" or h2.get("region", "-") == "-": continue
            if h2["region"] != h["region"]: break
            if h2.get("gate") == "accept": stay += 1; _acc_seen = True
            elif h2.get("gate") == "reject" and not _acc_seen: tries += 1
        print(f"[levelset10] resuming the sweep at region {h['region']} (K {Areg[h['region']]}, {stay} accepts / {tries} trailing rejects in this visit)", flush=True); break
_acc_tr = [(int(h["K_next"]), float(h.get("A_next", A0))) for h in log if h.get("algo") == "levelset10" and h.get("gate") == "accept" and "K_next" in h][-5:]
if _acc_tr:                                                               # v10c: shared first-visit state = median of the last 5 ACCEPTED steps
    TR_LAST = (int(np.median([k for k, _ in _acc_tr])), float(np.median([a for _, a in _acc_tr])))
best_I = max([h["I_accepted"] for h in log if h.get("algo") == "levelset10"] + ([acc["I"]] if acc else []), default=-1e30)
RATES = []                                                                # v10 R6: gain per second of accepted steps (restored from the log)
_prevI = None
for h in log:
    if h.get("algo") == "levelset10" and h.get("gate") == "accept" and _prevI is not None and h.get("t_step_s"): RATES.append((h["I_accepted"] - _prevI) / max(h["t_step_s"], 1.0))
    if h.get("algo") == "levelset10" and h.get("gate") in ("first", "accept"): _prevI = h["I_accepted"]
def next_region():
    global reg_i, part, tries, stay
    tries = 0; stay = 0; reg_i += 1
    if reg_i >= len(partitions[part]): reg_i = 0; part = 1 - part
if stay >= (STAY_MAX9 if STAY_RULE9 == "rate" else R_STAY) or tries >= R_TRIES:                                    # the resumed visit had already used up its accepts / tries
    print(f"[levelset10] resumed region visit already complete ({stay} accepts, {tries} rejects) -> next region", flush=True); next_region()
def cap_select_crop(score, K, cap, I0, J0):                               # top-K positive scores of a crop, <= cap per pupil cell
    W_ = score.shape[1]; flat = score.reshape(-1); npool = int((flat > 0).sum())
    if npool == 0: return torch.zeros(0, dtype=torch.long, device=dev)
    v_, ix = torch.topk(flat, min(npool, 8 * K + 1000)); ix = ix[v_ > 0]
    gi = ix // W_ + I0; gj = ix % W_ + J0
    ci = torch.floor((XG0 + (gi.double() + 0.5) * MESH10) / PITCH).long(); cj = torch.floor((XG0 + (gj.double() + 0.5) * MESH10) / PITCH).long()
    cs, order = torch.sort(ci * 100003 + cj, stable=True)
    start = torch.ones_like(cs, dtype=torch.bool); start[1:] = cs[1:] != cs[:-1]
    pos = torch.arange(cs.numel(), device=dev); gstart = torch.cummax(torch.where(start, pos, torch.zeros_like(pos)), 0).values
    rank = torch.empty_like(pos); rank[order] = pos - gstart
    return ix[rank < cap][:K]
def set_q(Q, gi, gj, val):                                                # write quadrant reps given by global coords (gi >= gj >= H10) and the transpose
    a = gi - H10; b = gj - H10; Q[a, b] = val; Q[b, a] = val
def tr_set(rid, K, A, grow, shrink, ncand, share=True):                                # v10c trust region over (K, A); with A_TRUST off: K x2 / K /2 / K as before
    global TR_LAST
    if grow:
        if A_TRUST and K >= ncand and A < A_MAX: Dreg[rid] = min(2 * A, A_MAX); Areg[rid] = K
        else: Areg[rid] = K * 2; Dreg[rid] = A
    elif shrink:
        if A_TRUST and A > A0: Dreg[rid] = max(A / 2, A0); Areg[rid] = K
        else: Areg[rid] = max(K // 2, L_KMIN); Dreg[rid] = A
    else: Areg[rid] = K; Dreg[rid] = A
    if share: TR_LAST = (Areg[rid], Dreg[rid])                            # only accepted steps set the shared first-visit state (a corner
                                                                          # region's rejects had shrunk K to 800 for the next region, 2026-09-25)
def view20(Q): return (Fn.avg_pool2d(Q.float()[None, None], 2)[0, 0] >= 0.5).cpu().numpy()   # 20 nm view (SiN fraction >= 1/2) for the old tools

while step < N_STEPS:
    t0 = time.time(); gate = "-"; pred = float("nan"); ratio = float("nan"); nflip = 0; t_adj = 0.0; n_solved = 0; rid = "-"; n_seam = 0; n_rep = 0; K = 0; seam_fail = False
    A = A0; n_cutpx = 0; pred_cut = 0.0; t_prop = 0.0; t_fwd = 0.0; n_nuc = (0, 0); n_nucpx = 0; pred_nuc = 0.0; TOPO_STATS.clear()
    if acc is None:
        with torch.no_grad(): Dq = start_design()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        n_solved = solved_tiles(); I, g_oct, cnt, t540 = forward_score()
        acc = {"theta": theta.detach().clone(), "I": I, "g_oct": g_oct, "cnt": cnt, "t540": t540, "G": None, "G_for": None, "Dq_bits": pack_q(Dq)}
        save_acc(); gate = "first"
    else:
        region = partitions[part][reg_i]; rid = f"p{part}r{reg_i}"
        if acc["g_oct"] is None: _, acc["g_oct"], acc["cnt"], _ = forward_score()
        if acc["G_for"] not in ("all", rid):
            t1 = time.time(); acc["G"] = region_gradient10(region, acc["g_oct"], acc["cnt"]); acc["G_for"] = rid; t_adj = time.time() - t1; save_acc()
        tp0 = time.time()
        with torch.no_grad():
            elig = oct_mask & (coverage(region) == COV_ALL) & (COV_ALL > 0)
            ri, rj = torch.nonzero(elig, as_tuple=True)
        if ri.numel() == 0:
            print(f"step {step:3d}  region {rid}: no interior pixels -> next region", flush=True); next_region(); continue
        with torch.no_grad():
            i0, i1 = max(int(ri.min()) - M20, 0), min(int(ri.max()) + M20 + 1, NPX); j0, j1 = max(int(rj.min()) - M20, 0), min(int(rj.max()) + M20 + 1, NPX)
            I0, I1, J0, J1 = SUB * i0, SUB * i1, SUB * j0, SUB * j1
            gI = torch.arange(I0, I1, device=dev)[:, None]; gJ = torch.arange(J0, J1, device=dev)[None, :]
            rep10 = (gI >= gJ) & (gJ >= H10)
            el10 = _ero(d4c20(elig, i0, i1, j0, j1).repeat_interleave(SUB, 0).repeat_interleave(SUB, 1), EL_ERODE)
            Gs = symmetrize(acc["G"]); mq = torch.where(kq[:, None] == kq[None, :], 4.0, 8.0)
            g10 = d4q(Gs / mq, I0, I1, J0, J1); del Gs                   # per-image dI/dD (for the velocity field)
            Grep = d4q(symmetrize(acc["G"]), I0, I1, J0, J1)              # per-representative dI/dD (for the prediction)
            m_c = d4q(Dq, I0, I1, J0, J1)
            flipg = Grep * (1 - 2 * m_c.float())                          # first-order gain of flipping a representative
            band = _dil(boundary10(m_c), L_BAND)
            v = gblur(g10 * band, L_VSIG) * el10; vmax = float(v.abs().max())
            K = Areg.get(rid, TR_LAST[0] if (SHARE_TR and TR_LAST) else L_K0)
            A = Dreg.get(rid, TR_LAST[1] if (SHARE_TR and TR_LAST) else A0) if A_TRUST else A0
            cand = v.abs() * (el10 & rep10 & boundary10(m_c)); ncand = int((cand > 0).sum())
            flip_g = None
            Kb = K; nuc_n = (0, 0)
            if TOPO:                                                      # v10b: one pool, K decides; stamp score = trust x gain / (px x 8 D4 images)
                qn = m_c.cpu().numpy(); sc_n = []
                for sin in (True, False):
                    _, _, gv, kk = nuc_peaks9(qn, flipg, el10 & rep10, sin); sc_n.append(NUC_TRUST * gv / (float(_disk(kk).sum()) * 8.0))
                allsc = torch.cat([cand[cand > 0].flatten(), sc_n[0], sc_n[1]])
                thr = float(torch.topk(allsc, K).values[-1]) if allsc.numel() > K else 0.0
                nuc_n = (int((sc_n[0] >= thr).sum()), int((sc_n[1] >= thr).sum())); Kb = max(K - nuc_n[0] - nuc_n[1], 0)
            if PHASE1:                                                    # v10d phase 1: flip the top-K positive-gain representatives, nothing else
                sc1 = (gblur(Grep, P1_SIG) if P1_SIG > 0 else Grep) * (1 - 2 * m_c.float())
                ok1 = el10 & rep10 & (sc1 > 0); ncand = int(ok1.sum()); K = min(K, ncand)
                if K > 0:
                    fr = ok1 & (sc1 >= float(torch.topk(sc1[ok1], K).values[-1])) if ncand > K else ok1
                    a_, b_ = torch.nonzero(fr, as_tuple=True); gi = a_ + I0; gj = b_ + J0; nflip = int(gi.numel())
                    if nflip: pred = float(flipg[a_, b_].sum()); flip_g = (gi, gj, ~m_c[a_, b_])
            elif vmax > 0 and ncand > 0 and (Kb > 0 or sum(nuc_n) > 0):
                six = cap_select_crop(cand, Kb, L_SEED_CAP, I0, J0) if Kb > 0 else torch.zeros(0, dtype=torch.long, device=dev); Wc = J1 - J0
                set_q(SELq, six // Wc + I0, six % Wc + J0, True); sel_c = d4q(SELq, I0, I1, J0, J1); set_q(SELq, six // Wc + I0, six % Wc + J0, False)
                patch = _dil(sel_c, max(L_SEED_R, int(math.ceil(A)) + 1) if A_TRUST else L_SEED_R)
                p_new = torch.clamp(sdf10(m_c) + A * torch.sign(v) * patch, -PHI_CLIP, PHI_CLIP)
                raw = openclose9(p_new >= 0, L_OC) if SMOOTH == "openclose" else gblur((p_new >= 0).float(), L_SIG) >= 0.5
                zone = _dil(patch, L_MIN) & el10
                cs_ = torch.where(zone, raw, m_c); nucm = None
                if TOPO and sum(nuc_n) > 0: cs_, nucm, n_nuc = nucleate9(cs_, flipg, el10 & rep10, nuc_n)
                base = thin9(m_c); newv = thin9(cs_) & ~base; LAST_CUT = None; cutm = None
                if bool(newv.any()):
                    rp = repair9(cs_, flipg, Grep if NECK_CUT else None); fz = _dil(newv, 6 if NECK_CUT else 2) & el10   # a neck cut reaches ~L_GAP/2 px past the violation
                    n_rep = int(((rp != cs_) & fz).sum()); cs_ = torch.where(fz, rp, cs_)
                    if LAST_CUT is not None: cutm = LAST_CUT & fz
                fr = el10 & rep10 & (cs_ != m_c)
                a_, b_ = torch.nonzero(fr, as_tuple=True); gi = a_ + I0; gj = b_ + J0
                r_, last_ = 0, None
                for _ in range(SEAM_ROUNDS):
                    if gi.numel() == 0: break
                    set_q(FLq, gi, gj, True); comp = m_c ^ d4q(FLq, I0, I1, J0, J1); set_q(FLq, gi, gj, False)
                    new = thin9(comp) & ~base
                    new[:L_GAP] = False; new[-L_GAP:] = False; new[:, :L_GAP] = False; new[:, -L_GAP:] = False
                    c_ = int(new.sum())
                    if c_ == 0: break
                    if last_ is not None and c_ >= last_: r_ += 1
                    last_ = c_
                    na, nb = torch.nonzero(_dil(new, r_), as_tuple=True); qa = qidx(na + I0); qb = qidx(nb + J0)
                    bad = torch.zeros((H10, H10), dtype=torch.bool, device=dev); bad[torch.maximum(qa, qb), torch.minimum(qa, qb)] = True
                    keep = ~bad[gi - H10, gj - H10]; n_seam += int((~keep).sum()); gi, gj = gi[keep], gj[keep]
                else:
                    gi = gi[:0]; gj = gj[:0]; seam_fail = True                # seam cleanup did not settle in SEAM_ROUNDS rounds: whole proposal dropped
                nflip = int(gi.numel())
                if nflip:
                    li, lj = gi - I0, gj - J0; pred = float(flipg[li, lj].sum()); flip_g = (gi, gj, ~m_c[li, lj])
                    if cutm is not None:                                  # cut pixels among the flips: their first-order gain is over-optimistic
                        ic = cutm[li, lj]; n_cutpx = int(ic.sum()); pred_cut = float(flipg[li, lj][ic].sum())
                        pred = pred - (1.0 - CUT_TRUST) * pred_cut
                    if nucm is not None:                                  # nucleated px: first-order gain of a whole new post / hole, discounted like cuts
                        inn = nucm[li, lj]; n_nucpx = int(inn.sum()); pred_nuc = float(flipg[li, lj][inn].sum())
                        pred = pred - (1.0 - NUC_TRUST) * pred_nuc
        t_prop = time.time() - tp0
        if nflip == 0:
            idle += 1
            if idle > 4 * sum(len(p_) for p_ in partitions):
                print(f"step {step:3d}: no admissible boundary move in any region -> stop", flush=True); break
            if seam_fail and K > L_KMIN:                                  # too many seeds for the seam cleanup: fewer, not more
                tr_set(rid, K, A, False, True, ncand, share=False); print(f"step {step:3d}  region {rid}: K {K} A {A:g} seam cleanup did not settle -> K {Areg[rid]} A {Dreg[rid]:g}", flush=True); continue
            if vmax > 0 and K < ncand and not seam_fail:
                Areg[rid] = K * 2; print(f"step {step:3d}  region {rid}: K {K} moves no pixel -> K x2", flush=True); continue
            print(f"step {step:3d}  region {rid}: no admissible boundary move -> next region", flush=True); next_region(); continue
        if not pred > 0:
            tries += 1; tr_set(rid, K, A, False, True, ncand, share=False)
            print(f"step {step:3d}  region {rid}: K {K} A {A:g} predicts {pred:+.2e} -> K {Areg[rid]} A {Dreg[rid]:g} (no solve)", flush=True)
            if tries >= R_TRIES or K <= L_KMIN: next_region()
            continue
        idle = 0
        with torch.no_grad(): gi, gj, newval = flip_g; set_q(Dq, gi, gj, newval)
        tf0 = time.time(); n_solved = solved_tiles(); I, g_oct, cnt, t540 = forward_score(); t_fwd = time.time() - tf0
        ratio = (I - acc["I"]) / pred if pred > 0 else float("nan"); I_prev_acc = acc["I"]
        if I > acc["I"]:
            gate = "accept"; stay += 1; tries = 0
            acc = {"theta": theta.detach().clone(), "I": I, "g_oct": g_oct, "cnt": cnt, "t540": t540, "G": None, "G_for": None, "Dq_bits": pack_q(Dq)}; save_acc()
            tr_set(rid, K, A, ratio > 0.75, ratio < 0.25, ncand)
            if STAY_RULE9 == "rate":                                      # v10 R6: leave when this region's gain per second falls below STAY_FRAC x the recent average
                rate = (I - I_prev_acc) / max(time.time() - t0, 1.0); RATES.append(rate); ref = float(np.mean(RATES[-20:]))
                if stay >= STAY_MAX9 or (stay >= 2 and rate < STAY_FRAC9 * ref): next_region()
            elif stay >= R_STAY: next_region()
        else:
            gate = "reject"; tries += 1; prop_bits = pack_q(Dq); prop_view = view20(Dq)
            with torch.no_grad(): set_q(Dq, gi, gj, ~newval)
            tr_set(rid, K, A, False, True, ncand, share=False)
            if tries >= R_TRIES or K <= L_KMIN: next_region()
    best_I = max(best_I, acc["I"])
    with torch.no_grad():
        d10_bits = prop_bits if gate == "reject" else acc["Dq_bits"]; hard_eval = prop_view if gate == "reject" else view20(Dq)
        fill10 = float(np.unpackbits(d10_bits).mean()) if gate == "reject" else float(Dq.float().mean())
        if gate != "reject":
            np.savez(OUT / "best.npz", hard_oct=np.packbits(hard_eval), hard_shape=np.array(hard_eval.shape), d10_quad=acc["Dq_bits"], d10_shape=np.array([H10, H10]),
                     I_tar=acc["I"], step=step)
    t2 = time.time()
    rec = {"step": step, "I_tar": I, "I_accepted": acc["I"], "binary": True, "algo": "levelset10", "gate": gate, "region": rid,
           "alpha": A, "K": K, "K_next": Areg.get(rid, L_K0), "A_next": Dreg.get(rid, A), "flipped": nflip, "seam_dropped": n_seam, "repaired": n_rep, "pred_gain": pred, "tr_ratio": ratio,
           "tiles_solved": n_solved, "t_step_s": t2 - t0, "t_adjoint_s": t_adj, "t_propose_s": t_prop, "t_forward_s": t_fwd, "fill": fill10,
           "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0,
           "design_mesh_nm": round(1000 * MESH10, 3), "aniso": ANISO, "ls_min_px": L_MIN, "ls_gap_px": L_GAP, "ls_neck_px": L_NECK, "ls_sigma_px": L_SIG, "ls_smooth": SMOOTH, "ls_oc_px": L_OC,
           "ls_seed_r": L_SEED_R, "ls_seed_cap": L_SEED_CAP, "ls_vanish": VANISH, "ls_neck_cut": NECK_CUT, "cut_trust": CUT_TRUST, "cut_px": n_cutpx, "pred_cut_raw": pred_cut, "topo": TOPO, "nuc_posts": n_nuc[0], "nuc_holes": n_nuc[1], "nuc_px": n_nucpx, "pred_nuc_raw": pred_nuc, "topo_stats": dict(TOPO_STATS), "stay_rule": STAY_RULE9, "ls_version": 10.1, "lr": 0.0, "lr_scale": 1.0, "beta": 0.0}
    log.append(rec); hist_path.write_text(json.dumps(log, indent=1))
    np.savez(OUT / f"step_{step:04d}.npz", hard_eval_oct=np.packbits(hard_eval), hard_shape=np.array(hard_eval.shape), d10_quad=d10_bits, d10_shape=np.array([H10, H10]),
             I_tar=I, I_accepted=acc["I"], t_abs_540=t540, cnt=cnt.detach().cpu().numpy())
    if TOPO: print(f"  topology: nucleated posts {n_nuc[0]} holes {n_nuc[1]} ({n_nucpx} px kept, raw pred {pred_nuc:+.2e}), sub-minimum decisions {dict(TOPO_STATS)}", flush=True)
    print(f"step {step:3d}  I_tar(binary) {I:.5f}  accepted {acc['I']:.5f}  gate {gate:6s}  region {rid}  K {K}{f' A {A:g}px' if A_TRUST else ''}  flipped {nflip}  repaired {n_rep}  seam-dropped {n_seam}  "
          f"pred {pred:+.2e}  ratio {ratio:+.2f}  tiles solved {n_solved}  {t2-t0:.0f}s (adjoint {t_adj:.0f}s, propose {t_prop:.0f}s, forward {t_fwd:.0f}s)  fill {fill10:.3f}  peak {rec['peak_gib']:.1f} GiB", flush=True)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    step += 1
    if MAX_HOURS and (time.time() - T_START) / 3600 > MAX_HOURS:
        print(f"wall-clock budget {MAX_HOURS} h reached after step {step - 1} (best {best_I:.5f})", flush=True); break
print(f"FF_DONE (best accepted I_tar {best_I:.5f})", flush=True)


