# ---------------- zone-periodic freeform optimization (FF_ALGO=zone), ZDA-style zones ----------------
# Zones after Sun, ..., Johnson, Lin, ACS Photonics 12,
# 3163 (2025) (ZDA): wide radial zones, one discrete rotational order n per zone, sub-wavelength azimuthal period. Here: zone j = [Z_j, Z_j+1)
# (2 um steps to 8 um, then widths 25 % of the radius -> inner/outer azimuthal period ratio >= 0.8), n_j (multiple of 4) gives a ~EZ_P nm
# azimuthal period at the zone's OUTER radius; the design variable of zone j is one mirror-symmetric half wedge C[j, rho, a]
# (rho = 10 nm radial rows over the whole zone width, a = 10 nm of arc at the outer radius), copied n_j times around the circle -> radial
# merging / splitting inside a zone keeps the periodicity; n changes only at the ~16 zone edges (lens-level fab repair there).
# Step: one full-lens adjoint -> pixel gradient summed onto the wedge px (all copies); K wedge-boundary flips + nucleation seeds with the
# largest first-order gain; realise (raster + lens fab repair); FDTD; accept if I_tar rises; trust region on the realised-design prediction.
# Start: plan B's ring elements (EZ_START elem npz: ring_r_nm, ring_hole, ring_d_nm) moved onto the zone lattice with the fill kept
# (d' = d sqrt(P'/P), P / P' = the ring's azimuthal pitch before / in the zone), or EZ_START = a d10 design npz (majority projection).
os.environ["SM_NO_CARRIER"] = "1"
_src = open(Path(os.path.abspath(__file__)).with_name("loop_spacemap.py"), encoding="utf-8").read().split("\n# ---- library: one nominal fill per tile core")[0]
exec(compile(_src, "loop_spacemap.py(helpers)", "exec"))
EZ_P = float(os.environ.get("EZ_P", "290")); EZ_START = os.environ["EZ_START"]; EZ_K0 = int(os.environ.get("EZ_K0", "2000"))
EZ_OUTER = int(os.environ.get("EZ_OUTER", "80")); EZ_KMIN = int(os.environ.get("EZ_KMIN", "50")); EZ_NN0 = int(os.environ.get("EZ_NN0", "16"))
EZ_NUC = os.environ.get("EZ_NUC", "1") == "1"
EZ_TABU = os.environ.get("EZ_TABU", "1") == "1"                           # 2026-09-26: 36 % of step 3's flips undid step 2 -> tabu + line search
EZ_LS = [float(x) for x in os.environ.get("EZ_LS", "1,0.25,0.0625").split(",")]
LASTFLIP = None
EZ_FREE = float(os.environ.get("EZ_FREE", "4")) * 1e3                    # zone 0 = r < EZ_FREE: plain D4 pixels (the wedges get too thin
def zone_edges():                                                         #  near the centre), then 1 um zones to 8 um (inner/outer period >= 0.8)
    if os.environ.get("EZ_EDGES"): return np.array([float(x) * 1e3 for x in os.environ["EZ_EDGES"].split(",")])
    e = [0.0, EZ_FREE, EZ_FREE + 1e3, EZ_FREE + 2e3, EZ_FREE + 3e3, 8e3] if EZ_FREE == 4e3 else [0.0, EZ_FREE, 8e3]
    while e[-1] < R_AP * 1e3:
        nxt = e[-1] * 1.25
        if R_AP * 1e3 - nxt < 0.4 * (nxt - e[-1]): nxt = R_AP * 1e3 + 30.0
        e.append(min(nxt, R_AP * 1e3 + 30.0))
    if os.environ.get("EZ_SNAP", "0") == "1":                             # edges at fixed radii cut the plan-B posts in half -> snap every
        P_ = float(os.environ.get("EZ_SNAP_P", "290"))                     #  inner edge to the nearest ring-lattice midpoint (k + 1/2) P, between the
        e = [e[0]] + [(round(x / P_ - 0.5) + 0.5) * P_ for x in e[1:-1]] + [e[-1]]   #  posts (radius <= 115 nm around k P)
    return np.array(e)
ZE = zone_edges(); KZ = len(ZE) - 1; ZO = ZE[1:]; ZI = ZE[:-1]
EZ_SUPER = int(os.environ.get("EZ_SUPER", "1"))                          # repeat unit = EZ_SUPER sub-periods of ~EZ_P (does a 290 x n
NZB = np.array([max(4 * EZ_SUPER, int(round(2 * math.pi * zo / EZ_P / (4 * EZ_SUPER))) * 4 * EZ_SUPER) for zo in ZO])   #  supercell stay 290-periodic?)
NZ = NZB // EZ_SUPER; NZ[0] = 4; NZB[0] = 4                                 # NZB: sub-period count (multiple of 4 EZ_SUPER), NZ: copies of the wedge
if os.environ.get("EZ_OCTANT", "0") == "1": NZ = np.full_like(NZ, 4)      # every zone's wedge = the whole 45 deg arc, only D4
NR = np.round((ZO - ZI) / 10.0).astype(int); NRMAX = int(NR.max())
AZ = np.array([max(1, int(math.ceil(math.pi * zo / n / 10.0))) for zo, n in zip(ZO, NZ)]); AZ[0] = NR[0]; AMAX = int(AZ.max())   # zone 0: (x, y) px
EZ_C4 = os.environ.get("FF_SYM", "D4").upper() == "C4"                   # FF_SYM=C4 (2026-09-27): every zone's wedge = the whole 90 deg arc of the
if EZ_C4:                                                                 #  quadrant, no mirror fold (only C4 left); zone 0 = the whole quarter disc of (x, y) px
    NZ = np.full_like(NZ, 4); AZ = np.array([max(1, int(math.ceil(0.5 * math.pi * zo / 10.0))) for zo in ZO]); AZ[0] = NR[0]; AMAX = int(AZ.max())
EZ_CART = os.environ.get("EZ_CART", "0") == "1"                          # EZ_CART=1 (2026-09-27): no zones; ONE Cartesian density on the 10 nm quadrant,
if EZ_CART:                                                               #  cell (0, i, j) = quadrant px (i, j), i >= j (D4 octant), VALID = octant px in
    assert not EZ_C4, "EZ_CART is the D4 octant (no FF_SYM=C4)"           #  the aperture; the mirror px (j, i) maps to the same cell
    KZ = 1; NR = np.array([H10]); AZ = np.array([H10]); NRMAX = AMAX = H10; NZ = np.array([4]); NZB = NZ.copy()
    ZE = np.array([0.0, H10 * 10.0]); ZI = ZE[:-1]; ZO = ZE[1:]
EZ_POLAR = os.environ.get("EZ_POLAR", "0") == "1"                        # EZ_POLAR=1 (2026-09-28): no zones; ONE density
if EZ_POLAR:                                                              #  on a ragged polar grid of the D4 octant: row k = r in [10 k, 10 k + 10) nm (r_k =
    assert not EZ_C4 and not EZ_CART, "EZ_POLAR is the D4 octant (no FF_SYM=C4, no EZ_CART)"   #  (k + 1/2) 10 nm), PN[k] = max(1, round(pi/4 r_k / 10 nm)) arc
    _nr = int(math.ceil((R_AP * 1e3 + 10.0) / 10.0))                      #  cells over 0..45 deg (~10 nm of arc at every radius, one cell at the centre);
    PN = np.maximum(1, np.round(0.25 * math.pi * (np.arange(_nr) + 0.5))).astype(int)   #  cell (0, k, a), a < PN[k], padded to (1, NR, max PN); rows
    KZ = 1; NR = np.array([_nr]); AZ = np.array([int(PN.max())]); NRMAX = _nr; AMAX = int(PN.max()); NZ = np.array([4]); NZB = NZ.copy()   #  to R_AP + 10 nm
    ZE = np.array([0.0, _nr * 10.0]); ZI = ZE[:-1]; ZO = ZE[1:]           #  (rows / cells without an aperture px leave VALID by the valid map below)
ZD = OUT / "zone"; ZD.mkdir(exist_ok=True)
VALID = torch.zeros((KZ, NRMAX, AMAX), dtype=torch.bool, device=dev)
for j in range(1, KZ): VALID[j, :NR[j], :AZ[j]] = True
_i0 = torch.arange(NR[0], device=dev)[:, None]; _j0 = torch.arange(NR[0], device=dev)[None, :]
if not EZ_POLAR: VALID[0, :NR[0], :NR[0]] = (_i0 >= _j0) & (torch.hypot((_i0 + 0.5) * 10.0, (_j0 + 0.5) * 10.0) < ZO[0])   # lower triangle of the central disc
if EZ_C4: VALID[0, :NR[0], :NR[0]] = torch.hypot((_i0 + 0.5) * 10.0, (_j0 + 0.5) * 10.0) < ZO[0]   # (C4) the whole quarter disc
if EZ_CART: VALID[0] = (_i0 >= _j0) & AP10.bool()                         # (CART) octant px inside the solver's aperture
if EZ_POLAR: VALID[0] = torch.arange(AMAX, device=dev)[None, :] < torch.tensor(PN, device=dev)[:, None]   # (POLAR) a < PN[k]; unmapped cells removed below
print(f"[zone] {KZ} zones (zone 0 = D4 pixels r < {EZ_FREE / 1e3:g} um), supercell x{EZ_SUPER}, edges (um) {np.round(ZE / 1e3, 2).tolist()}, n {NZ.tolist()}, azimuthal period at inner/outer edge "
      f"{np.round(2 * math.pi * ZI / NZ).astype(int).tolist()} / {np.round(2 * math.pi * ZO / NZ).astype(int).tolist()} nm, wedge <= {NRMAX} x {AMAX} px, "
      f"{int(VALID.sum())} wedge px; start {EZ_START}; K0 {EZ_K0}, seeds {EZ_NN0 if EZ_NUC else 0}", flush=True)
_fa_cache_path=Path(os.environ.get('FA_NEUTRAL_CACHE','__missing__'))
if EZ_POLAR and _fa_cache_path.exists():
    _fa_geo=np.load(_fa_cache_path)
    assert int(_fa_geo['H10'])==H10 and int(_fa_geo['NRMAX'])==NRMAX and int(_fa_geo['AMAX'])==AMAX
    assert np.array_equal(_fa_geo['PN'],PN)
    IDX=torch.tensor(_fa_geo['IDX'],device=dev,dtype=torch.int32)
    VALID=torch.tensor(_fa_geo['VALID'],device=dev,dtype=torch.bool)
    print('[neutral geometry] loaded shared IDX/VALID',flush=True)
else:
    xs10 = (torch.arange(H10, device=dev, dtype=torch.float64) + 0.5) * 10.0
    def build_index():
        IDX = torch.empty((H10, H10), dtype=torch.int32, device=dev)
        Et = torch.tensor(ZE, device=dev); Nt = torch.tensor(NZ, device=dev, dtype=torch.float64); Ot = torch.tensor(ZO, device=dev)
        At = torch.tensor(AZ, device=dev); Rt = torch.tensor(NR, device=dev)
        for r0 in range(0, H10, 700):
            r1 = min(r0 + 700, H10); X = xs10[r0:r1][:, None].expand(-1, H10); Y = xs10[None, :].expand(r1 - r0, -1)
            r = torch.hypot(X, Y); phi = torch.atan2(Y, X)
            j = (torch.searchsorted(Et, r.contiguous(), right=True) - 1).clamp(0, KZ - 1)
            rho = torch.minimum(torch.floor((r - Et[j]) / 10.0).long().clamp(min=0), Rt[j] - 1)
            dphi = 2 * math.pi / Nt[j]; pl = torch.remainder(phi, dphi); pl = torch.where(pl > dphi / 2, dphi - pl, pl)
            a = torch.minimum(torch.floor(pl * Ot[j] / 10.0).long(), At[j] - 1)
            ii = torch.arange(r0, r1, device=dev)[:, None].expand(-1, H10); jj = torch.arange(H10, device=dev)[None, :].expand(r1 - r0, -1)
            fr = j == 0; rho = torch.where(fr, torch.maximum(ii, jj).clamp(max=NRMAX - 1), rho); a = torch.where(fr, torch.minimum(ii, jj).clamp(max=AMAX - 1), a)
            IDX[r0:r1] = (j * NRMAX * AMAX + rho * AMAX + a).int()
        return IDX
    if EZ_C4:
        def build_index():                                                    # (C4) arc cell a = floor(phi * Z_outer / 10 nm), phi in [0, 90 deg], no fold;
            IDX = torch.empty((H10, H10), dtype=torch.int32, device=dev)      #  zone 0: rho = x px, a = y px (whole quarter disc)
            Et = torch.tensor(ZE, device=dev); Ot = torch.tensor(ZO, device=dev); At = torch.tensor(AZ, device=dev); Rt = torch.tensor(NR, device=dev)
            for r0 in range(0, H10, 700):
                r1 = min(r0 + 700, H10); X = xs10[r0:r1][:, None].expand(-1, H10); Y = xs10[None, :].expand(r1 - r0, -1)
                r = torch.hypot(X, Y); phi = torch.atan2(Y, X)
                j = (torch.searchsorted(Et, r.contiguous(), right=True) - 1).clamp(0, KZ - 1)
                rho = torch.minimum(torch.floor((r - Et[j]) / 10.0).long().clamp(min=0), Rt[j] - 1)
                a = torch.minimum(torch.floor(phi * Ot[j] / 10.0).long(), At[j] - 1)
                ii = torch.arange(r0, r1, device=dev)[:, None].expand(-1, H10); jj = torch.arange(H10, device=dev)[None, :].expand(r1 - r0, -1)
                fr = j == 0; rho = torch.where(fr, ii.clamp(max=NRMAX - 1), rho); a = torch.where(fr, jj.clamp(max=AMAX - 1), a)
                IDX[r0:r1] = (j * NRMAX * AMAX + rho * AMAX + a).int()
            return IDX
    if EZ_CART:
        def build_index():                                                    # (CART) px (i, j) -> octant cell (max, min): i H10 + j, transpose-symmetric
            IDX = torch.empty((H10, H10), dtype=torch.int32, device=dev)
            for r0 in range(0, H10, 700):
                r1 = min(r0 + 700, H10); ii = torch.arange(r0, r1, device=dev)[:, None]; jj = torch.arange(H10, device=dev)[None, :]
                IDX[r0:r1] = torch.where(ii >= jj, ii * H10 + jj, jj * H10 + ii).int()
            return IDX
        print(f"[zone] EZ_CART: one Cartesian octant density, {H10} x {H10} quadrant px, {int(VALID.sum())} octant px in the aperture", flush=True)
    if EZ_POLAR:
        def build_index():                                                    # (POLAR) px -> polar cell: k = floor(r / 10 nm), a = floor(phi' / 45 deg PN[k]) with
            IDX = torch.empty((H10, H10), dtype=torch.int32, device=dev)      #  phi' = atan2(min, max) (octant fold -> IDX transpose-symmetric, diagonal px ->
            Pt = torch.tensor(PN, device=dev)                                 #  last cell); px past the last row (outside the aperture) clamp to it
            for r0 in range(0, H10, 700):
                r1 = min(r0 + 700, H10); X = xs10[r0:r1][:, None].expand(-1, H10); Y = xs10[None, :].expand(r1 - r0, -1)
                k = torch.floor(torch.hypot(X, Y) / 10.0).long().clamp(max=NRMAX - 1); n = Pt[k]
                a = torch.minimum(torch.floor(torch.atan2(torch.minimum(X, Y), torch.maximum(X, Y)) / (0.25 * math.pi) * n).long(), n - 1)
                IDX[r0:r1] = (k * AMAX + a).int()
            return IDX
        print(f"[zone] EZ_POLAR: one polar octant density, {NRMAX} rows of 10 nm, {int(PN[0])}..{AMAX} arc cells per row, {int(VALID.sum())} cells", flush=True)
    IDX = build_index()
    if os.environ.get("EZ_VALIDMAP", "0") == "1" or EZ_POLAR:                          # 2026-09-27: wedge cells that no 10 nm px of the aperture maps to (arc
        _cnt = torch.zeros(KZ * NRMAX * AMAX, dtype=torch.int32, device=dev)  #  spacing < 10 nm on a zone's inner rows) have no gradient but still
        for r0 in range(0, H10, 1500):                                        #  entered the filter as frozen values -> drop them from VALID
            _ii = IDX[r0:r0 + 1500][AP10[r0:r0 + 1500].bool()].long(); _cnt.index_add_(0, _ii, torch.ones_like(_ii, dtype=torch.int32))
        _nv = int(VALID.sum()); VALID &= (_cnt > 0).reshape(KZ, NRMAX, AMAX); print(f"[zone] EZ_VALIDMAP: {_nv - int(VALID.sum())} unmapped wedge px removed from VALID", flush=True)
    def raster(C):
        flat = C.reshape(-1); q = torch.zeros((H10, H10), dtype=torch.bool, device=dev)
        for r0 in range(0, H10, 1500): q[r0:r0 + 1500] = flat[IDX[r0:r0 + 1500].long()]
        return symmetrize(q & AP10)
    EZ_FAB = os.environ.get("EZ_FAB", "1") == "1"                             # 0: no fab rules at all;
    

def realise_c(C):                                                         #  the raster is scored as is
    q0 = raster(C)
    if not EZ_FAB:
        ts, ta = _thin(q0); return q0, dict(repair_px=0, fill=float(q0[AP10].float().mean()), thin_sin_px=int(ts.sum()), thin_air_px=int(ta.sum()))
    q, info = fab_repair(q0); ts, ta = _thin(q)
    assert int(ts.sum()) == 0 and int(ta.sum()) == 0
    info.update(repair_px=int((q != q0).sum()), fill=float(q[AP10].float().mean())); return q, info
GPIX = None
def cell_gradient(tag):
    global GPIX
    gf = ZD / f"G_{tag}.npy"
    if gf.exists(): G = torch.tensor(np.load(gf), device=dev); print(f"[zone] pixel gradient loaded from {gf.name}", flush=True)
    else:
        _, g_oct, cnt_, _ = forward_score(); G = region_gradient10(tiles, g_oct, cnt_); np.save(gf, G.float().cpu().numpy())
    GPIX = G
    Gc = torch.zeros(KZ * NRMAX * AMAX, dtype=torch.float64, device=dev)
    for r0 in range(0, H10, 1500): Gc.index_add_(0, IDX[r0:r0 + 1500].reshape(-1).long(), G[r0:r0 + 1500].reshape(-1).double())
    return Gc.view(KZ, NRMAX, AMAX).float() * VALID
def cell_boundary(C):
    b = torch.zeros_like(C)
    b[:, 1:, :] |= C[:, 1:, :] != C[:, :-1, :]; b[:, :-1, :] |= C[:, :-1, :] != C[:, 1:, :]
    b[:, :, 1:] |= C[:, :, 1:] != C[:, :, :-1]; b[:, :, :-1] |= C[:, :, :-1] != C[:, :, 1:]
    return b & VALID
NUC = {"post": (3, 8), "hole": (4, 6)} if os.environ.get("EZ_FAB", "1") == "1" else {"post": (1, 2), "hole": (1, 2)}   # no fab: 30 nm seeds, 20 nm clearance                                    # seed radius / clearance (10 nm px): 60 nm post, 80 nm hole, gap 80 / wall 60 nm
def _disc(R): y = torch.arange(-R, R + 1, device=dev).float(); return (y[:, None] ** 2 + y[None, :] ** 2) <= (R + 0.5) ** 2
def _fold(a, A): a = torch.where(a < 0, -1 - a, a); return torch.where(a > A - 1, 2 * A - 1 - a, a)
def seed_pixels(j, r, a, R):
    d = _disc(R); rr, aa = torch.nonzero(d, as_tuple=True); rr = rr + r - R; aa = _fold(aa + a - R, int(AZ[j]))
    m = (rr >= 0) & (rr < int(NR[j])); return torch.unique(torch.stack([rr[m], aa[m]], 1), dim=0)
def nucleation(C, Gc, nmax):                                              # one seed per zone and 64-row slab per step (zones are wide)
    gainS = (Gc * (1 - 2 * C.float())).float(); cands = []
    for kind, (R, M) in NUC.items():
        L = R + M; region = (C if kind == "hole" else ~C) & VALID; bad = (~region).float()[:, None]
        bad = Fn.pad(bad, (L, L, 0, 0), mode="reflect"); bad = Fn.pad(bad, (0, 0, L, L), value=1.0)
        ok = Fn.conv2d(bad, _disc(L).float()[None, None])[:, 0] == 0
        g = Fn.pad(gainS[:, None], (R, R, 0, 0), mode="reflect"); g = Fn.pad(g, (0, 0, R, R))
        sc = Fn.conv2d(g, _disc(R).float()[None, None])[:, 0]; sc = torch.where(ok & VALID, sc, torch.full_like(sc, -1e30))
        for j in range(KZ):
            for r0 in range(0, int(NR[j]), 64):
                blk = sc[j, r0:r0 + 64]; v, i = blk.reshape(-1).max(0)
                if float(v) > 0: rr_, aa_ = divmod(int(i), AMAX); cands.append((float(v), j, r0 + rr_, aa_, kind))
    cands.sort(reverse=True); out, used = [], []
    for c in cands:
        if any(u[0] == c[1] and abs(u[1] - c[2]) < 2 * (NUC[c[4]][0] + NUC[c[4]][1]) for u in used): continue
        px = seed_pixels(c[1], c[2], c[3], NUC[c[4]][0]); gx = float(gainS[c[1], px[:, 0], px[:, 1]].sum())
        if gx > 0: out.append((gx, c[1], c[2], c[3], c[4])); used.append((c[1], c[2]))
        if len(out) >= nmax: break
    return out
def start_cells():
    z = np.load(EZ_START, allow_pickle=True); C = torch.zeros((KZ, NRMAX, AMAX), dtype=torch.bool, device=dev)
    if "ring_r_nm" not in z.files:                                         # a d10 design: majority vote over every copy of a wedge px
        qd = torch.tensor(np.unpackbits(z["d10_quad"] if "d10_quad" in z.files else z["Q_bits"])[:H10 * H10].reshape(H10, H10).astype(bool), device=dev)
        cnt = torch.zeros(KZ * NRMAX * AMAX, device=dev); s = torch.zeros_like(cnt)
        for r0 in range(0, H10, 1500):
            m = AP10[r0:r0 + 1500]; idx = IDX[r0:r0 + 1500][m].long(); v = qd[r0:r0 + 1500][m].float(); cnt.index_add_(0, idx, torch.ones_like(v)); s.index_add_(0, idx, v)
        return ((s / cnt.clamp(min=1)) > 0.5).view(KZ, NRMAX, AMAX) & VALID
    rr_, hh, dd = z["ring_r_nm"], z["ring_hole"].astype(bool), z["ring_d_nm"].astype(float); P0 = float(rr_[1] - rr_[0])
    n_ring = np.array([1 if r == 0 else max(4, int(round(2 * math.pi * r / P0 / 4)) * 4) for r in rr_])
    rho_nm = torch.arange(NRMAX, device=dev, dtype=torch.float64)[:, None] * 10.0 + 5.0; a_nm = torch.arange(AMAX, device=dev, dtype=torch.float64)[None, :] * 10.0 + 5.0
    stat = {"posts_dropped": 0, "holes_dropped": 0}
    qb = torch.tensor(np.unpackbits(z["Q_bits"])[:H10 * H10].reshape(H10, H10).astype(bool), device=dev) if "Q_bits" in z.files else None
    if qb is not None: C[0, :NR[0], :NR[0]] = qb[:NR[0], :NR[0]]           # centre: plan B's own pixels
    for j in range(1, KZ):
        r = ZI[j] + rho_nm; s = a_nm * r / ZO[j]                            # physical radius and arc of every wedge px
        i = np.clip(np.round((ZI[j] + (np.arange(NRMAX) + 0.5) * 10.0) / P0).astype(int), 0, len(rr_) - 1)   # nearest ring per row
        it = torch.tensor(i, device=dev)
        Pp = 2 * math.pi * rr_ / NZB[j]; P = np.where(n_ring > 1, 2 * math.pi * rr_ / np.maximum(n_ring, 1), 1e9)
        d2 = dd * np.sqrt(np.clip(Pp / P, 0, 4)); d2 = np.where(hh, np.clip(d2, 0, Pp - 60.0), np.clip(d2, 0, Pp - 80.0))
        small = (hh & (d2 < 80)) | (~hh & (d2 < 60)); stat["holes_dropped"] += int((small & hh & (rr_ >= ZI[j]) & (rr_ < ZO[j])).sum())
        stat["posts_dropped"] += int((small & ~hh & (rr_ >= ZI[j]) & (rr_ < ZO[j])).sum()); d2 = np.where(small, 0.0, d2)
        rc = torch.tensor(rr_, device=dev)[it][:, None]; rad = torch.tensor(d2 / 2, device=dev)[it][:, None]; hole = torch.tensor(hh, device=dev)[it][:, None]
        Pb = 2 * math.pi * r / NZB[j]; t = torch.abs(torch.remainder(s + Pb / 2, Pb) - Pb / 2)   # arc folded into the sub-period: EZ_SUPER identical
        inside = ((r - rc) ** 2 + t ** 2) <= rad ** 2                     #  sub-cells at the start
        C[j] = torch.where(hole, ~inside, inside)
    print(f"[zone] start from plan B elements (fill kept, d' = d sqrt(P'/P)): {stat}", flush=True)
    return C & VALID
# ---- outer loop ----
st_path = ZD / "zone_state.npz"; hist = []
if st_path.exists():
    S = np.load(st_path, allow_pickle=True); k0 = int(S["k"]) + 1
    C_acc = torch.tensor(np.unpackbits(S["C"])[:KZ * NRMAX * AMAX].reshape(KZ, NRMAX, AMAX).astype(bool), device=dev)
    I_acc = float(S["I_acc"]); K = int(S["K"]); NNUC = int(S["NNUC"]); hist = json.loads(str(S["hist"])); Gc = None
    Dq_acc = torch.tensor(np.unpackbits(S["d10"])[:H10 * H10].reshape(H10, H10).astype(bool), device=dev)
    if os.environ.get("EZ_KOVR"): K = int(os.environ["EZ_KOVR"])          # K override on resume
    if "LASTFLIP" in S.files: LASTFLIP = torch.tensor(np.unpackbits(S["LASTFLIP"])[:KZ * NRMAX * AMAX].reshape(KZ, NRMAX, AMAX).astype(bool), device=dev)
    print(f"[zone] resumed after step {k0 - 1}: I_acc {I_acc:.5f}, K {K}, seeds {NNUC}", flush=True)
else:
    k0 = 0; C_acc = None; I_acc = -1e30; K = EZ_K0; NNUC = EZ_NN0; Gc = None; Dq_acc = None
def save_state(k):
    np.savez(st_path, k=k, C=np.packbits(C_acc.cpu().numpy()), I_acc=I_acc, K=K, NNUC=NNUC, hist=json.dumps(hist), d10=pack_q(Dq_acc), **({"LASTFLIP": np.packbits(LASTFLIP.cpu().numpy())} if LASTFLIP is not None else {}))
for k in range(k0, EZ_OUTER + 1):
    t0 = time.time(); t_adj = 0.0; nnuc = 0; kinds = {}; PP = {}
    trials = []                                                           # (I, q, info, C, pred, pred_q, nflip, nnuc, kinds, PP, size)
    if k == 0:
        C_new = start_cells(); t_prop = time.time() - t0; q, info = realise_c(C_new)
        T, cnt, I = tmaps(q); trials.append((I, q, info, C_new, float("nan"), float("nan"), 0, 0, {}, {}, 0))
    else:
        if Gc is None:
            _t = time.time(); tmaps(Dq_acc); Gc = cell_gradient(f"step{max(h['step'] for h in hist if h['gate'] in ('first', 'accept')):04d}"); t_adj += time.time() - _t
        gain = Gc * (1 - 2 * C_acc.float()); cand = cell_boundary(C_acc) & (gain > 0); ntabu = 0
        if EZ_TABU and LASTFLIP is not None: ntabu = int((cand & LASTFLIP).sum()); cand &= ~LASTFLIP   # no flip-back of the last accepted step
        nc = int(cand.sum()); gs_sorted = torch.sort(gain[cand], descending=True).values
        seeds = nucleation(C_acc, Gc, NNUC) if EZ_NUC else []
        top = max(1, min(K, nc)); sizes = sorted({max(1, int(round(top * f))) for f in EZ_LS}, reverse=True)
        t_prop = time.time() - t0
        for si, size in enumerate(sizes):                                 # line search over the move size: forward only, best one wins
            thr = float(gs_sorted[size - 1]) if nc > 0 else 0.0
            sel = cand & (gain >= thr); Cn = C_acc ^ sel; PPt = {"flip": float(gain[sel].sum())}; kt = {}; nn = 0
            for gx, j, r_, a_, kind in seeds[:max(1, int(round(len(seeds) * size / top)))] if seeds else []:
                px = seed_pixels(j, r_, a_, NUC[kind][0]); Cn[j, px[:, 0], px[:, 1]] = (kind == "post"); PPt["seed"] = PPt.get("seed", 0.0) + gx
                nn += 1; kt[kind] = kt.get(kind, 0) + 1
            qt, it_ = realise_c(Cn); pq = 0.0
            for r0 in range(0, H10, 1500): pq += float((GPIX[r0:r0 + 1500].double() * (qt[r0:r0 + 1500].double() - Dq_acc[r0:r0 + 1500].double())).sum())
            import gc as _gc; _gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            T, cnt, It = tmaps(qt); trials.append((It, qt, it_, Cn, sum(PPt.values()), pq, int(sel.sum()), nn, kt, PPt, size))
            print(f"   line search {si + 1}/{len(sizes)}: flips {int(sel.sum())} (of {nc} candidates, {ntabu} tabu), seeds {nn}, I {It:.5f} "
                  f"(accepted {I_acc:.5f}), pred {pq:+.2e}", flush=True)
    best = max(range(len(trials)), key=lambda i: trials[i][0])
    I, q, info, C_new, pred, pred_q, nflip, nnuc, kinds, PP, size_b = trials[best]
    t_rep = 0.0; t_fdtd = time.time() - t0 - t_prop - t_adj
    ratio_cell = (I - I_acc) / pred if k > 0 and pred > 0 else float("nan")
    ratio = (I - I_acc) / pred_q if k > 0 and pred_q > 0 else ratio_cell
    gate = "first" if k == 0 else ("accept" if I > I_acc else "reject")
    ls_I = [round(t[0], 5) for t in trials]; ls_sizes = [t[10] for t in trials]
    changed = int((q != Dq_acc).sum()) if Dq_acc is not None else 0
    if gate != "reject":
        if k > 0:                                                         # next top size: the chosen one, doubled if it was the largest and predicted well
            K = max(int(size_b * (2 if (best == 0 and ratio > 0.75) else 1)), EZ_KMIN)
            if nnuc: NNUC = max(int(NNUC * (2 if (best == 0 and ratio > 0.75) else (1 if best == 0 else 0.5))), 1)
        LASTFLIP = (C_new ^ C_acc) if C_acc is not None else None
        C_acc, I_acc, Dq_acc = C_new, I, q
        _t = time.time(); Gc = cell_gradient(f"step{k:04d}"); t_adj += time.time() - _t
    else:
        K = max(min(ls_sizes[-1] if k > 0 else K, max(nflip, 1)) // 2, EZ_KMIN)   # every size lost: half of the smallest tried
        if nnuc: NNUC = max(NNUC // 2, 1)
    hard = view20(q)
    np.savez(OUT / f"step_{k:04d}.npz", hard_eval_oct=np.packbits(hard), hard_shape=np.array(hard.shape), d10_quad=pack_q(q), d10_shape=np.array([H10, H10]),
             I_tar=I, I_accepted=I_acc, cells=np.packbits(C_new.cpu().numpy()), cell_shape=np.array([KZ, NRMAX, AMAX]), zone_edges_nm=ZE, zone_n=NZ)
    if gate != "reject":
        np.savez(OUT / "best.npz", hard_oct=np.packbits(hard), hard_shape=np.array(hard.shape), d10_quad=pack_q(q), d10_shape=np.array([H10, H10]), I_tar=I, step=k)
    rec = {"step": k, "I_tar": I, "I_accepted": I_acc, "gate": gate, "algo": "zone", "region": f"Z{k}", "K": K, "flipped": nflip, "nucleated": nnuc,
           "nucleated_kinds": kinds, "NNUC": NNUC, "pred_gain": pred, "pred_parts": PP, "line_search_I": ls_I, "line_search_sizes": ls_sizes, "chosen_size": size_b, "pred_gain_realised": pred_q, "tr_ratio": ratio, "tr_ratio_cells": ratio_cell,
           "changed_px": changed, "fill": info["fill"], "repair": info, "t_propose_s": t_prop, "t_adjoint_s": t_adj, "t_repair_s": t_rep, "t_forward_s": t_fdtd,
           "t_step_s": time.time() - t0, "ls_min_px": L_MIN, "ls_gap_px": L_GAP}
    hist.append(rec); (OUT / "history.json").write_text(json.dumps(hist, indent=1)); save_state(k)
    print(f"step {k:3d}  I_tar(binary) {I:.5f}  accepted {I_acc:.5f}  gate {gate:6s}  region Z{k}  K {K}  flipped {nflip} wedge px, seeds {nnuc} {kinds}  "
          f"pred {pred:+.2e} {dict((a_, round(b_, 4)) for a_, b_ in PP.items())} (realised {pred_q:+.2e})  ratio {ratio:+.2f} (cells {ratio_cell:+.2f})  changed {changed} px  "
          f"repair {info['repair_px']} px  fill {info['fill']:.3f}  {time.time() - t0:.0f}s (adjoint {t_adj:.0f}s, repair {t_rep:.0f}s, forward {t_fdtd:.0f}s)", flush=True)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
print(f"FF_DONE (best accepted I_tar {I_acc:.5f})", flush=True)

