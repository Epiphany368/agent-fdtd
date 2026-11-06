# ---------------- zone density topology optimization (FF_ALGO=zonec): continuous density + filter + projection ----------------
# Continuous density + filter + projection, adopted after the binary wedge flips oscillated (36 % of step 3's flips undid step 2; the undone
# flips were boundary px with the largest predicted gains, 88 % SiN removals opening sub-grid slits -> the density gradient at a binary
# point mispredicts a whole 0 -> 1 flip at the high-contrast SiN/air boundary). Standard density TO instead (Christiansen & Sigmund,
# JOSA B 2021; ZDA: conic filter + binarization): the same ZDA-style zones (loop_zone.py geometry: D4 pixels r < 4 um, one n per zone,
# 290 nm azimuthal period at the zone's outer edge), per wedge px a latent theta in [0, 1];
#   rho~ = Gaussian filter of theta in wedge coordinates (radius EZC_RF px, mirror in arc, replicate at zone edges: sets the length scale)
#   rho^ = tanh projection (beta, eta = 0.5), beta doubled every EZC_BSTEP steps from EZC_B0 to EZC_BMAX (continuation to binary)
#   design D = rho^ copied over the zone (float 10 nm quadrant) -> FDTD with the fill-fraction (anisotropic) averaging the solver already uses
# One forward + adjoint per step (forward_score): pixel gradient -> summed onto the wedge px -> autograd through projection and filter ->
# Adam on theta. Every EZC_BINEVAL steps the thresholded binary design (rho^ > 0.5) is scored too (what a fabricated device would be).
FA_NS = globals().get("FA_NS", False)                                  # 2026-09-30: ns_run.py defines the non-symmetric Cartesian geometry, the tile solver
if not FA_NS:                                                            #  calls and pack_q/view20 before exec'ing this file; the D4 helpers below are skipped
    os.environ["SM_NO_CARRIER"] = "1"
    _src = open(Path(os.path.abspath(__file__)).with_name("loop_spacemap.py"), encoding="utf-8").read().split("\n# ---- library: one nominal fill per tile core")[0]
    exec(compile(_src, "loop_spacemap.py(helpers)", "exec"))
    _zsrc = open(Path(os.path.abspath(__file__)).with_name("loop_zone.py"), encoding="utf-8").read()
    os.environ.setdefault("EZ_FAB", "0"); os.environ.setdefault("EZ_START", os.environ.get("EZC_START", ""))
    exec(compile("EZ_P = " + _zsrc.split("EZ_P = ")[1].split("def realise_c")[0], "loop_zone.py(geometry)", "exec"))
    exec(compile("def start_cells" + _zsrc.split("def start_cells")[1].split("# ---- outer loop ----")[0], "loop_zone.py(start)", "exec"))
else:
    EZ_C4 = EZ_CART = EZ_POLAR = False
EZC_RF = float(os.environ.get("EZC_RF", "2.0")); EZC_B0 = float(os.environ.get("EZC_B0", "4")); EZC_BMAX = float(os.environ.get("EZC_BMAX", "64"))
EZC_BSTEP = int(os.environ.get("EZC_BSTEP", "6")); EZC_LR = float(os.environ.get("EZC_LR", "0.05")); EZC_OUTER = int(os.environ.get("EZC_OUTER", "60"))
EZC_SCHED = os.environ.get("EZC_SCHED", "")                               # "meep": beta stages
EZC_BETAS = [float(x) for x in os.environ.get("EZC_BETAS", "8,16,32,64").split(",")]; EZC_SLEN = int(os.environ.get("EZC_SLEN", "18"))   #  of EZC_SLEN
EZC_SK0 = int(os.environ.get("EZC_SK0", "0"))                              #  steps from step EZC_SK0, Adam reset per stage, lr EZC_LR (last stage EZC_LRLAST)
EZC_LRLAST = float(os.environ.get("EZC_LRLAST", "0.02"))
EZC_GW = float(os.environ.get("EZC_GW", "0")); EZC_GWRAMP = int(os.environ.get("EZC_GWRAMP", "30"))   # grey penalty J = I - w <4 rho (1 - rho)>,
#  w ramped 0 -> EZC_GW over EZC_GWRAMP steps from EZC_SK0 (pamping needs loss, which the reversible lossless-interior adjoint cannot take)
EZC_LR0 = float(os.environ.get("EZC_LR0", "0")); EZC_LRDECAY = int(os.environ.get("EZC_LRDECAY", "30"))   # lr: EZC_LR0 -> EZC_LR exponentially over EZC_LRDECAY steps
EZC_BINEVAL = int(os.environ.get("EZC_BINEVAL", "4")); EZC_INIT = os.environ.get("EZC_INIT", "")
EZC_ADAPT = os.environ.get("EZC_ADAPT", "1") == "1"                       # adaptive continuation
EZC_STALL = float(os.environ.get("EZC_STALL", "0.001")); EZC_SMIN = int(os.environ.get("EZC_SMIN", "5")); EZC_SMAX = int(os.environ.get("EZC_SMAX", "15"))
#  beta doubles when the grey I rose < EZC_STALL over the last 3 steps at this beta (>= EZC_SMIN, <= EZC_SMAX steps per stage)
EZC_BINRATE = float(os.environ.get("EZC_BINRATE", "0")); EZC_BINW = float(os.environ.get("EZC_BINW", "1")); EZC_BINK0 = int(os.environ.get("EZC_BINK0", "0"))
#  2026-09-30 binarization-rate constraint (after Ballew et al. 2023, here a soft one for Adam): from step EZC_BINK0 the grey measure g = <4 rho (1 - rho)>
#  must fall at least EZC_BINRATE per step below its value at EZC_BINK0; J = I - EZC_BINW relu(g - g_target)^2 (0: off)
CD = OUT / "zonec"; CD.mkdir(exist_ok=True)
_kr = int(math.ceil(3 * EZC_RF)); _t = torch.arange(-_kr, _kr + 1, device=dev, dtype=torch.float32); _g = torch.exp(-_t ** 2 / (2 * EZC_RF ** 2)); _g = _g / _g.sum()
def _conv(x):                                                            # separable Gaussian, zero outside the array
    y = Fn.conv2d(Fn.pad(x[:, None], (_kr, _kr, 0, 0)), _g.view(1, 1, 1, -1)); y = Fn.conv2d(Fn.pad(y, (0, 0, _kr, _kr)), _g.view(1, 1, -1, 1)); return y[:, 0]
_VF = None
def filt(x):                                                              # normalized convolution over the valid wedge px only (2026-09-26 fix: the plain
    global _VF                                                            #  filter mixed the zeros outside the wedge / above the centre diagonal in and
    if _VF is None: _VF = _conv(VALID.float()).clamp(min=1e-6)           #  darkened every wedge edge and the diagonal, seen in the grey figure)
    return _conv(x * VALID) / _VF * VALID
def proj(x, beta, eta=0.5):
    return (math.tanh(beta * eta) + torch.tanh(beta * (x - eta))) / (math.tanh(beta * eta) + math.tanh(beta * (1 - eta)))
def raster_f(rho):                                                        # float wedge -> float 10 nm quadrant (transpose-symmetric)
    flat = rho.reshape(-1); q = torch.zeros((H10, H10), dtype=torch.float32, device=dev)
    for r0 in range(0, H10, 1500): q[r0:r0 + 1500] = flat[IDX[r0:r0 + 1500].long()]
    q = q * AP10; return torch.where(kq_[:, None] >= kq_[None, :], q, q.T)
kq_ = torch.arange(H10, device=dev)
if EZ_C4:                                                                 # FF_SYM=C4 (2026-09-27): whole-quadrant wedges, no transpose symmetrization,
    def raster_f(rho):                                                    #  and the filter continues across the quadrant seams by C4 (below)
        flat = rho.reshape(-1); q = torch.zeros((H10, H10), dtype=torch.float32, device=dev)
        for r0 in range(0, H10, 1500): q[r0:r0 + 1500] = flat[IDX[r0:r0 + 1500].long()]
        return q * AP10
    def _c4pad_index():                                                   # gather map of the filter input padded by _kr px: flat wedge index, n = zero
        n = KZ * NRMAX * AMAX; P = torch.full((KZ, NRMAX + 2 * _kr, AMAX + 2 * _kr), n, dtype=torch.long, device=dev)
        R_ = torch.arange(-_kr, NRMAX + _kr, device=dev)[:, None]; A_ = torch.arange(-_kr, AMAX + _kr, device=dev)[None, :]
        for j in range(1, KZ):                                            # zones >= 1: the arc is periodic (90 deg arc -> next quadrant = this arc by C4,
            okr = (R_ >= 0) & (R_ < int(NR[j]))                           #  AZ[j] cells; the last cell is < 10 nm short), rows outside the zone: zero
            P[j] = torch.where(okr, j * NRMAX * AMAX + R_.clamp(0, NRMAX - 1) * AMAX + torch.remainder(A_, int(AZ[j])), torch.full_like(P[j], n))
        a, b, _ = c4rep(R_, A_); ok0 = (a < int(NR[0])) & (b < int(NR[0]))   # zone 0 (x, y px): the C4 image of the quarter disc across both axes
        P[0] = torch.where(ok0, a.clamp(max=NRMAX - 1) * AMAX + b.clamp(max=AMAX - 1), torch.full_like(P[0], n))
        return P
    _PIDX = _c4pad_index()
    def _conv(x):                                                         # (C4) separable Gaussian on the C4-continued wedges (no zero padding)
        xp = torch.cat([x.reshape(-1), x.new_zeros(1)])[_PIDX]
        y = Fn.conv2d(xp[:, None], _g.view(1, 1, 1, -1)); y = Fn.conv2d(y, _g.view(1, 1, -1, 1)); return y[:, 0]
if EZ_CART:                                                               # EZ_CART=1 (2026-09-27, zone seams): cells = 10 nm octant px (loop_zone.py); the
    _APQ = AP10.float()[None]                                             #  filter acts on the D4-symmetric quadrant (octant mirrored at the diagonal) with
    def _conv(x):                                                         #  half-sample mirror padding at both axes (px i <-> -1 - i) = the whole-lens filter;
        xp = torch.cat([x[:, :_kr].flip(1), x, x.new_zeros(x.shape[0], _kr, x.shape[2])], 1)   #  zero beyond the far edge (outside the aperture)
        xp = torch.cat([xp[:, :, :_kr].flip(2), xp, xp.new_zeros(xp.shape[0], xp.shape[1], _kr)], 2)
        y = Fn.conv2d(xp[:, None], _g.view(1, 1, 1, -1)); y = Fn.conv2d(y, _g.view(1, 1, -1, 1)); return y[:, 0]
    _VF = _conv(_APQ).clamp(min=1e-6)                                     # normalized over the aperture px of the whole lens
    def filt(x): return _conv(torch.where(kq_[:, None] >= kq_[None, :], x, x.transpose(1, 2)) * _APQ) / _VF * VALID
    print(f"[zonec] EZ_CART: filter on the D4-symmetric {H10}^2 quadrant (mirror at axes and diagonal), sigma {EZC_RF} px", flush=True)
if EZ_POLAR:                                                              # EZ_POLAR=1 (2026-09-28): cells = ragged polar grid of the octant (loop_zone.py). Separable
    assert (os.environ.get("EZC_INIT", "") == "grey" or os.environ.get("EZC_INIT", "").startswith("rhoquad:")) and float(os.environ.get("EZC_RMAX", "0")) == 0, \
        "EZ_POLAR: EZC_INIT grey or rhoquad:<npz> only (plan B start_cells / EZC_RMAX assume zones)"   #  Gaussian with the physical sigma 10 EZC_RF nm:
    _PN = torch.tensor(PN, device=dev); _PA = torch.arange(AMAX, device=dev); _PRC = max(1, (1 << 23) // AMAX)   # (1) along the arc of each row, weights
    _ps = 0.25 * math.pi * (torch.arange(NRMAX, device=dev, dtype=torch.float64) + 0.5) * 10.0 / _PN   #  exp(-(t s_k)^2 / 2 sigma^2), s_k = the row's arc cell
    _pka = int(math.ceil(3 * 10.0 * EZC_RF / float(_ps[_PN >= 2].min())))  #  (nm), half-sample mirror at 0 and 45 deg (both D4 mirror lines); rows with
    _pw = torch.exp(-(torch.arange(-_pka, _pka + 1, device=dev, dtype=torch.float64)[None, :] * _ps[:, None]) ** 2 / (2 * (10.0 * EZC_RF) ** 2)).float()   #  one cell
    def _pfold(u, n): u = torch.remainder(u, 2 * n); return torch.where(u >= n, 2 * n - 1 - u, u)   #  fold every tap onto it; (2) along the radius: row k + d
    def _paz(x, T):                                                       #  (weight _g, 10 nm rows) sampled at the same angle by linear interpolation of
        xf = x.reshape(-1); out = torch.zeros_like(xf)                    #  its arc index (a + 1/2) PN[k + d] / PN[k] - 1/2 (same mirror); row k + d < 0 =
        for k0 in range(0, NRMAX, _PRC):                                  #  row -1 - (k + d) at the same angle (r -> -r is a 180 deg rotation, in D4),
            k1 = min(k0 + _PRC, NRMAX); n = _PN[k0:k1, None]; m = _PA[None, :] < n   #  rows past the last: zero. T = transpose (adjoint) pass.
            xs = xf[k0 * AMAX:k1 * AMAX].view(k1 - k0, AMAX); o = out[k0 * AMAX:k1 * AMAX].view(k1 - k0, AMAX)
            if T: xs = xs * m
            for t in range(-_pka, _pka + 1):
                j = _pfold(_PA[None, :] + t, n); w = _pw[k0:k1, t + _pka, None]
                if T: o.scatter_add_(1, j, w * xs)
                else: o += w * torch.gather(xs, 1, j)
            if not T: o *= m
        return out.view_as(x)
    def _prad(x, T):
        xf = x.reshape(-1); out = torch.zeros_like(xf)
        for k0 in range(0, NRMAX, _PRC):
            k1 = min(k0 + _PRC, NRMAX); kk = torch.arange(k0, k1, device=dev); n = _PN[k0:k1]; m = _PA[None, :] < n[:, None]
            if T: gs = xf[k0 * AMAX:k1 * AMAX].view(k1 - k0, AMAX) * m
            else: acc = torch.zeros((k1 - k0, AMAX), dtype=xf.dtype, device=dev)
            for d in range(-_kr, _kr + 1):
                kd = kk + d; kd = torch.where(kd < 0, -1 - kd, kd); ok = kd < NRMAX; kd = kd.clamp(max=NRMAX - 1); nd = _PN[kd]
                f = (_PA[None, :].double() + 0.5) * (nd.double() / n.double())[:, None] - 0.5; i0 = torch.floor(f); tf = (f - i0).to(xf.dtype); i0 = i0.long()
                b = (kd * AMAX)[:, None]; j0 = b + _pfold(i0, nd[:, None]); j1 = b + _pfold(i0 + 1, nd[:, None]); w = _g[d + _kr] * ok.to(xf.dtype)[:, None]
                if T: out.index_add_(0, j0.reshape(-1), (w * (1 - tf) * gs).reshape(-1)); out.index_add_(0, j1.reshape(-1), (w * tf * gs).reshape(-1))
                else: acc += w * ((1 - tf) * xf[j0] + tf * xf[j1])
            if not T: out[k0 * AMAX:k1 * AMAX] = (acc * m).reshape(-1)
        return out.view_as(x)
    class _PolarL(torch.autograd.Function):                               # L = radial o azimuthal pass; backward = L^T from the same index maps (recomputed per
        @staticmethod                                                     #  row chunk, nothing stored: ~8e7 cells at 208 um)
        def forward(ctx, x): return _prad(_paz(x, False), False)
        @staticmethod
        def backward(ctx, g): return _paz(_prad(g.contiguous(), True), True)
    _VF = _PolarL.apply(VALID.float()).clamp(min=1e-6)                    # normalized over the valid (mapped) cells
    def filt(x): return _PolarL.apply(x * VALID) / _VF * VALID
    print(f"[zonec] EZ_POLAR: polar filter sigma {10 * EZC_RF:.0f} nm, {2 * _pka + 1} arc taps x {2 * _kr + 1} rows, {NRMAX} rows x <= {AMAX} cells", flush=True)
def cell_grad(G):
    G=G*AP10  # Exact VJP of the aperture multiplication in raster_f.
    Gc = torch.zeros(KZ * NRMAX * AMAX, dtype=torch.float64, device=dev)
    for r0 in range(0, H10, 1500): Gc.index_add_(0, IDX[r0:r0 + 1500].reshape(-1).long(), G[r0:r0 + 1500].reshape(-1).double())
    return (Gc.view(KZ, NRMAX, AMAX) * VALID).float()
if FA_NS:                                                                # 2026-09-30 (ns_run.py): one density on the 10 nm grid of the whole aperture, cell
    def filt(x):                                                          #  (0, i, j) = px (i, j); the same separable Gaussian normalized over the aperture
        return _conv(x * VALID) / _conv(VALID.float()).clamp(min=1e-6) * VALID   #  px (normalizer rebuilt per call: no 1.6 GB tensor held through the FDTD)
    def raster_f(rho): return rho[0] * VALID[0]                           # design grid = cell grid: only the aperture multiplication
    def cell_grad(G): return (G * VALID[0])[None]                         # its exact VJP
    print(f"[zonec] NS: Cartesian 10 nm density {NRMAX} x {AMAX}, {int(VALID.sum())} aperture px, filter sigma {10 * EZC_RF:.0f} nm", flush=True)
# ---- state ----
# half-tone of a grey rho_cells: binary texture with local duty = rho
def _halftone(_R, mode):                                                  #  forward (tmaps). Zone 0 (x, y px): 100 nm cells with a centred square post
    B = torch.zeros_like(_R)                                          #  of area = cell-mean rho (all modes except "thr").
    for j in range(KZ):
        nr, az = int(NR[j]), int(AZ[j]); V = VALID[j, :nr, :az].float(); r = _R[j, :nr, :az]
        if mode == "thr": B[j, :nr, :az] = (r > 0.5).float(); continue
        b = torch.zeros_like(r)
        if j == 0:
            for i0 in range(0, nr, 10):
                for j0 in range(0, az, 10):
                    v = V[i0:i0 + 10, j0:j0 + 10]
                    if v.sum() == 0: continue
                    d = float((r[i0:i0 + 10, j0:j0 + 10] * v).sum() / v.sum()); s = int(round(10 * d ** 0.5)); o = (10 - s) // 2
                    b[i0 + o:i0 + o + s, j0 + o:j0 + o + s] = 1.0
        else:
            a = torch.arange(az, device=dev, dtype=torch.float32)[None, :] + 0.5; dr = r.mean(1, keepdim=True)   # row duty (azimuth-averaged rho)
            if mode == "az1": b = (a < dr * az).float()               # one spoke per 290 nm period, width = duty x period (wedge = half period, mirrored)
            elif mode == "az2": b = ((a - az / 2).abs() < dr * az / 2).float()   # two spokes per period (145 nm)
            elif mode == "rad100":                                    # radial rings, 100 nm period, duty = 10-row block mean
                for i0 in range(0, nr, 10):
                    d = float(r[i0:i0 + 10].mean()); b[i0:i0 + int(round(10 * d))] = 1.0
                    b[i0 + int(round(10 * d)):i0 + 10] = 0.0
            elif mode == "post290":                                   # posts on a 290 nm x period lattice: area fraction = cell-mean rho
                for i0 in range(0, nr, 29):
                    blk = r[i0:i0 + 29]; d = float(blk.mean()); h = blk.shape[0]; sr = int(round(h * d ** 0.5)); o = (h - sr) // 2
                    b[i0 + o:i0 + o + sr, :] = (a[0] < d ** 0.5 * az).float()[None, :]
        B[j, :nr, :az] = b * V
    return B
st_path = CD / "zonec_state.pt"; hist = []
_rf = os.environ.get("FA_RESUME_FROM", "")                               # 2026-10-01: resume from a kept state (path, or step number in zonec/hist/)
_rs_path = (CD / "hist" / f"state_step{int(_rf):04d}.pt") if _rf.isdigit() else Path(_rf) if _rf else st_path
if _rf: print(f"[zonec] FA_RESUME_FROM: resuming from {_rs_path} with the current env", flush=True)
if _rs_path.exists():
    S = torch.load(_rs_path, map_location=dev, weights_only=False); theta = S["theta"]; opt_m, opt_v = S["m"], S["v"]; k0 = S["k"] + 1; hist = S["hist"]
    BETA = S.get("beta", EZC_B0); STAGE0 = S.get("stage0", 0); FINAL = S.get("final", -1); BIN_G0 = S.get("bin_g0")
    best_bin = S["best_bin"]; print(f"[zonec] resumed after step {k0 - 1}", flush=True)
else:
    k0 = 0; BIN_G0 = None
    if FA_NS and EZC_INIT not in ("grey",) and not EZC_INIT.startswith("density:"):
        raise SystemExit(f"NS geometry: EZC_INIT grey or density:<npz> only (got {EZC_INIT!r})")
    if EZC_INIT == "grey":                                                # grey start: uniform 0.5 + small noise (breaks the azimuthal
        _gen = torch.Generator().manual_seed(int(os.environ.get("EZC_SEED", "20260926")))   #  symmetry the gradient alone would keep), beta from 1
        C0 = None; theta = (0.5 + float(os.environ.get("EZC_NOISE", "0.02")) * (2 * torch.rand((KZ, NRMAX, AMAX), generator=_gen) - 1)).to(dev) * VALID
    elif FA_NS and EZC_INIT.startswith("density:"):                      # 2026-09-30 (NS): warm start from a density map (ns_run.load_density), theta =
        _D0 = load_density(EZC_INIT.split(":", 1)[1]).to(dev)             #  the inverse projection at EZC_B0 (as rhoquad; the filter is ~identity on it)
        C0 = None; theta = ((0.5 + torch.atanh(((2 * _D0 - 1) * math.tanh(EZC_B0 / 2)).clamp(-0.999999, 0.999999)) / EZC_B0).clamp(0, 1)[None] * VALID)
        del _D0
    elif EZC_INIT.startswith("rhoquad:"):                                 # warm start from a grey design of another zone geometry (e.g.
        _D0 = torch.tensor(np.load(EZC_INIT.split(":", 1)[1])["q"].astype(np.float32), device=dev)   #  ff_d50_A step 14 -> supercell zones): float 10 nm
        _s = torch.zeros(KZ * NRMAX * AMAX, dtype=torch.float64, device=dev); _c = torch.zeros_like(_s)   #  quadrant rho -> mean per new wedge cell ->
        for r0 in range(0, H10, 1500):                                    #  theta = inverse projection at EZC_B0 (the filter is ~identity on a smooth
            _ii = IDX[r0:r0 + 1500].reshape(-1).long(); _m = AP10[r0:r0 + 1500].reshape(-1).bool()   #  grey field), so step 0 reproduces the source rho
            _s.index_add_(0, _ii[_m], _D0[r0:r0 + 1500].reshape(-1)[_m].double()); _c.index_add_(0, _ii[_m], torch.ones_like(_s[:1]).expand(int(_m.sum())))
        _cm = (_s / _c.clamp(min=1)).float().reshape(KZ, NRMAX, AMAX); _tb = math.tanh(EZC_B0 / 2)
        # 2026-09-27 fix: wedge cells that no 10 nm px maps to (arc spacing < 10 nm on the inner rows of a zone) kept rho 0 and the filter
        # pulled the grey down (octant: mean |drho| 0.083) -> fill them with the nearest mapped cell along the arc (previous, else next)
        _ok = (_c > 0).reshape(KZ, NRMAX, AMAX); _ar = torch.arange(AMAX, device=dev).expand(KZ, NRMAX, AMAX); _m1 = torch.full_like(_ar, -1)
        _fi = torch.cummax(torch.where(_ok, _ar, _m1), -1).values                                          # last mapped index <= a (-1: none)
        _ni = (AMAX - 1) - torch.flip(torch.cummax(torch.where(_ok.flip(-1), _ar, _m1), -1).values, [-1])   # first mapped index >= a
        _cm = torch.where(_ok, _cm, torch.where(_fi >= 0, torch.gather(_cm, -1, _fi.clamp(min=0)), torch.gather(_cm, -1, _ni.clamp(0, AMAX - 1))))
        C0 = None; theta = (0.5 + torch.atanh(((2 * _cm - 1) * _tb).clamp(-0.999999, 0.999999)) / EZC_B0).clamp(0, 1) * VALID
        _nzq = float(os.environ.get("EZC_RQNOISE", "0"))                  # 2026-09-27 (C4 from a D4 warm start): the on-axis unpolarized objective is mirror
        if _nzq > 0:                                                      #  symmetric, so a mirror-symmetric start keeps a mirror-symmetric gradient and C4
            _gq = torch.Generator().manual_seed(int(os.environ.get("EZC_SEED", "20260926")))   #  would just repeat D4 -> uniform noise on theta breaks it
            theta = (theta + _nzq * (2 * torch.rand(theta.shape, generator=_gq) - 1).to(dev)).clamp(0, 1) * VALID
    elif EZC_INIT.startswith("halftone:"):                                # start from the half-tone (EZC_INIT=halftone:<mode>) of the
        _Rh = torch.tensor(np.load(os.environ["EZC_HTSRC"])["rho_cells"].astype(np.float32), device=dev) * VALID   #  grey rho of EZC_HTSRC
        C0 = _halftone(_Rh, EZC_INIT.split(":", 1)[1]) > 0.5                #  (d50_A step 14: post290 scored 1.526 vs threshold 0.461)
    elif EZC_INIT == "greybin":                                           # binarized grey start: the grey + noise
        _gen = torch.Generator().manual_seed(int(os.environ.get("EZC_SEED", "20260926")))   #  start thresholded at its rho = 0.5 contour (the filtered
        _t = (0.5 + float(os.environ.get("EZC_NOISE", "0.02")) * (2 * torch.rand((KZ, NRMAX, AMAX), generator=_gen) - 1)).to(dev) * VALID   #  field, as the
        with torch.no_grad(): C0 = (filt(_t) > 0.5) & VALID.bool()        #  binarized step-0 figure shows) -> a random binary texture, then the soft start
    elif EZC_INIT:                                                        # a binary wedge design of loop_zone (step npz "cells"), same zones
        z = np.load(EZC_INIT); sh = tuple(int(v) for v in z["cell_shape"]); C0 = torch.zeros((KZ, NRMAX, AMAX), dtype=torch.bool, device=dev)
        C0[:sh[0], :sh[1], :sh[2]] = torch.tensor(np.unpackbits(z["cells"])[:int(np.prod(sh))].reshape(sh).astype(bool), device=dev)
    else: C0 = start_cells()
    if C0 is not None:                                                    # EZC_SOFT s: theta = 0.5 + s (C0 - 0.5) (soft B-lineage start, beta 1)
        _sft = float(os.environ.get("EZC_SOFT", "0.5")); theta = (0.5 + _sft * ((C0 & VALID).float() - 0.5)) * VALID
        _nz = float(os.environ.get("EZC_NOISE0", "0"))                    # noise on the posts start breaks each post's circular symmetry
        if _nz > 0:
            _gn = torch.Generator().manual_seed(int(os.environ.get("EZC_SEED", "20260926")))
            theta = (theta + _nz * (2 * torch.rand(theta.shape, generator=_gn) - 1).to(dev)).clamp(0, 1) * VALID
    opt_m = torch.zeros_like(theta); opt_v = torch.zeros_like(theta); best_bin = -1e30; BETA = EZC_B0; STAGE0 = 0; FINAL = -1
RAMP0 = S.get("ramp0", -1) if _rs_path.exists() else -1                  # EZC_SMOOTH: step at which the smooth beta ramp started (-1: not yet)
EZC_SMOOTH = float(os.environ.get("EZC_SMOOTH", "0"))                     # EZC_ADAPT + EZC_SMOOTH d > 0: smooth beta ramp, one doubling per d steps
EZC_FINALPOLISH = int(os.environ.get("EZC_FINALPOLISH", "0"))             # once beta sits at EZC_BMAX and the grey I
print(f"[zonec] {KZ} zones, filter sigma {EZC_RF} px ({10 * EZC_RF:.0f} nm), beta {EZC_B0} -> {EZC_BMAX} (x2 every {EZC_BSTEP} steps), Adam lr {EZC_LR}, "
      f"binary check every {EZC_BINEVAL} steps, init {EZC_INIT or 'plan B elements'}", flush=True)
_BTAB = json.loads(Path(os.environ["EZC_BETA_TABLE"]).read_text())["beta"] if os.environ.get("EZC_BETA_TABLE") else None   # EZC_BETA_TABLE: JSON whose
if _BTAB is not None:                                                     #  "beta" list gives the beta of every step (index = step); replaces the EZC_ADAPT rules
    print(f"[zonec] EZC_BETA_TABLE {os.environ['EZC_BETA_TABLE']}: beta of each step from the table ({len(_BTAB)} entries)", flush=True)
EZC_RMAX = float(os.environ.get("EZC_RMAX", "0"))                         # small-scale test: optimize only the
if EZC_RMAX > 0:                                                          #  zones inside r < EZC_RMAX um, the rest stays plan B's binary design; adjoint only on
    OPT = torch.zeros_like(VALID)                                         #  the tiles that overlap the optimized disc (the other tiles hit the near-field cache)
    for j in range(KZ):
        if ZO[j] <= EZC_RMAX * 1e3 + 1: OPT[j] = VALID[j]
    C_FIX = start_cells().float()                                         # fixed zones = plan B binary, for any start inside (grey or soft)
    GT = [t for t in tiles if math.hypot(max(t[0] - TILE / 2, 0.0), max(t[1] - TILE / 2, 0.0)) < EZC_RMAX]
    print(f"[zonec] small-scale test: optimized zones {[j for j in range(KZ) if bool(OPT[j].any())]} (r < {EZC_RMAX} um), adjoint tiles {GT}", flush=True)
else: OPT = VALID; C_FIX = None; GT = tiles
EZC_LOGN = int(os.environ.get("EZC_LOGN", "30")); EZC_LAST = int(os.environ.get("EZC_LAST", "5"))   # EZC_SCHED=log: beta B0 -> BMAX log-spaced over LOGN steps
_HT = os.environ.get("EZC_HTDIAG", "")                                    # half-tone diagnostic.
if _HT:                                                                   #  The grey rho of a saved step (EZC_HTDIAG = step npz) is turned into binary
    _R = torch.tensor(np.load(_HT)["rho_cells"].astype(np.float32), device=dev) * VALID   #  textures whose local duty = rho, each scored by one binary
    _res = {"src": _HT}
    for _m in os.environ.get("EZC_HTMODES", "thr,az1,az2,rad100,post290").split(","):
        _t0 = time.time(); _D = raster_f(_halftone(_R, _m)); _qb = _D > 0.5; _Ib = 0.0 if os.environ.get("EZC_HTDRY") else tmaps(_qb)[2]   # EZC_HTDRY: CPU test, no FDTD
        _res[_m] = {"I_tar": float(_Ib), "fill": float(_qb[AP10.bool()].float().mean()), "s": round(time.time() - _t0, 1)}
        np.savez(OUT / f"ht_{_m}.npz", d10_quad=pack_q(_qb), d10_shape=np.array([H10, H10]), I_tar=float(_Ib))
        print(f"[htdiag] {_m}: I_tar {_Ib:.5f}, fill {_res[_m]['fill']:.3f} ({_res[_m]['s']} s)", flush=True)
        (OUT / "htdiag.json").write_text(json.dumps(_res, indent=1))
    EZC_OUTER = k0 - 1                                                    # no optimization steps in the diagnostic
_MAXH = float(os.environ.get("FF_MAX_HOURS", "0")); _T_RUN = time.time()   # wall-clock budget of this launch (2026-09-27, H200 x2 24 h slot)
import fa_runtime as _NR                                                  # 2026-09-30: phase JSONL, background saves, STOP file / deadline, optimizer offload
_PL = _NR.phase_log(OUT); _SAVER = _NR.AsyncSaver(); _STOPC = _NR.StopControl(OUT)
FA_RHO_EVERY = int(os.environ.get("FA_RHO_EVERY", "10")); FA_GRAD_EVERY = int(os.environ.get("FA_GRAD_EVERY", "10"))   # grey rho / gradient export cadence (0: off)
FA_METRICS_EVERY = int(os.environ.get("FA_METRICS_EVERY", "20"))        # objective metrics (PSF, images, ...) of the binary design; the maps are saved anyway
FA_TKEEP_EVERY = int(os.environ.get("FA_TKEEP_EVERY", "10"))            # pupil Jones-map snapshots (T_grey_latest.npz is rewritten every step)
def _rho_of(t, beta):                                                     # projected density; built without autograd before the solves and rebuilt with it
    r = proj(filt(t), beta).clamp(0, 1) * VALID                           #  after them (same floats), so rank 0 holds no graph through the FDTD phases
    return torch.where(OPT, r, C_FIX * VALID) if C_FIX is not None else r #  (clamp: a -1e-8 rounding value made eps < 1 and stopped the CFL check, 2026-09-26)
def _full_T(T): return T if FA_NS else _jones_expand(T)                  # pupil Jones maps for saving (D4: octant cells -> full pupil)
for k in range(k0, EZC_OUTER + 1):
    t0 = time.time()
    if _MAXH > 0 and (t0 - _T_RUN) / 3600 > _MAXH:
        print(f"[zonec] wall-clock budget {_MAXH} h reached before step {k}; state saved after step {k - 1}, relaunch to resume", flush=True); break
    _why = _STOPC.check([h["t_step_s"] for h in hist])                     # STOP file, or FA_DEADLINE_UNIX closer than the expected step time
    if _why:
        print(f"[zonec] stop before step {k}: {_why}; state saved after step {k - 1}, relaunch to resume", flush=True); break
    if FINAL >= 0 and k > FINAL + EZC_FINALPOLISH: break                  #  stalls again, re-binarize once (theta <- soft(rho > 0.5), EZC_SOFT) and polish
    if _BTAB is not None:                                                 # recorded schedule: beta of step k (the last entry beyond the table)
        BETA = float(_BTAB[min(k, len(_BTAB) - 1)]); beta = BETA
    elif EZC_ADAPT:                                                       #  EZC_FINALPOLISH more steps, then stop
        n_st = k - STAGE0; hI = [h["I_tar"] for h in hist if h["step"] >= STAGE0]
        relative_gain=(hI[-1]-hI[-4])/max(abs(hI[-4]),1e-12) if len(hI)>=4 else float('inf')
        _smin = int(os.environ["EZC_SMIN0"]) if os.environ.get("EZC_SMIN0") and BETA <= EZC_B0 else EZC_SMIN   # 2026-10-01: EZC_SMIN0 for the grey (first) stage
        _sto = float(os.environ.get("EZC_SMOOTH_TO", "0"))                # 2026-10-01: smooth ramp only up to this beta, then x EZC_BFAC per stall
        stall=n_st>=_smin and len(hI)>=4 and (relative_gain<float(os.environ.get('EZC_STALL_REL','.01')) if os.environ.get('EZC_STALL_MODE','rel')!='abs'
                                                 else (hI[-1]-hI[-4])<EZC_STALL)   # EZC_STALL_MODE=abs: the absolute rule of the earlier 208 um runs (EZC_STALL)
        if RAMP0<0 and int(os.environ.get('EZC_MAX_GREY','0'))>0 and n_st>=int(os.environ['EZC_MAX_GREY']): stall=True   # grey-phase cap: off unless set (the earlier 208 um runs had none)
        _bgs = float(os.environ.get("EZC_BINGAIN_STOP", "0"))   # at BMAX, stop once a binary check gains <= _bgs over the previous one
        if _bgs > 0 and BETA >= EZC_BMAX:
            _ab2 = [h for h in hist if h.get("I_binary") is not None and h["I_binary"] == h["I_binary"]]
            if len(_ab2) >= 2 and _ab2[-1].get("beta", 0) >= EZC_BMAX and _ab2[-1]["I_binary"] - _ab2[-2]["I_binary"] <= _bgs * abs(_ab2[-2]["I_binary"]):
                print(f"[zonec] step {k}: binary gain {_ab2[-1]['I_binary'] - _ab2[-2]['I_binary']:.5f} <= {_bgs:g} x previous at beta {BETA:g}, stop", flush=True); break
        _gs = float(os.environ.get("EZC_GAPSTOP", "0"))               # stop at BMAX once grey/binary differ <= _gs twice
        if _gs > 0 and BETA >= EZC_BMAX:
            _hb = [h for h in hist if h.get("I_binary") is not None and h["I_binary"] == h["I_binary"] and h.get("beta", 0) >= EZC_BMAX]
            _bw = int(os.environ.get("EZC_BINSTALL", "0")); _br = float(os.environ.get("EZC_BINSTALL_REL", "0.002"))   # also require
            _ab = [h for h in hist if h.get("I_binary") is not None and h["I_binary"] == h["I_binary"]]                #  no binary gain > _br over the last _bw steps
            _old = [h["I_binary"] for h in _ab if h["step"] <= k - 1 - _bw]; _new = [h["I_binary"] for h in _ab if h["step"] > k - 1 - _bw]
            _stl = _bw <= 0 or (bool(_old) and bool(_new) and max(_new) <= max(_old) * (1 + _br))
            if _stl and len(_hb) >= 2 and all(abs(h["I_tar"] - h["I_binary"]) <= _gs * abs(h["I_tar"]) for h in _hb[-2:]):
                print(f"[zonec] step {k}: grey/binary gap <= {_gs:g} at beta {BETA:g} in the last two binary checks, stop", flush=True); break
        _bc = float(os.environ.get("EZC_BCONT", "0")); _bcf = int(os.environ.get("EZC_BCONT_FROM", "0"))   # beta x _bc every step from step _bcf
        if _bc > 1 and k >= _bcf and BETA < EZC_BMAX and (BETA > EZC_B0 or _bcf > 0):
            BETA = min(_bc * BETA, EZC_BMAX); STAGE0 = k
        elif EZC_SMOOTH > 0 and not (_sto > 0 and BETA >= _sto):            # beta stays B0 until the
            if RAMP0 < 0 and stall: RAMP0 = k                             #  first stall (grey phase), then rises every step, x2 per EZC_SMOOTH steps
            if RAMP0 >= 0 and BETA < EZC_BMAX:                            #  (ff_d50_S4: beta 4 -> 256 in steps 35 -> 88, one doubling per 8.8 steps)
                BETA = min(EZC_B0 * 2 ** ((k - RAMP0 + 1) / EZC_SMOOTH), EZC_BMAX)
                if _sto > 0 and BETA >= _sto: BETA = _sto; STAGE0 = k    # hand over to the stall-triggered rule
                if BETA >= EZC_BMAX: STAGE0 = k                           # at BMAX: stop on the next stall (no re-binarization)
            elif BETA >= EZC_BMAX and stall:
                print(f"[zonec] step {k}: grey I stalled at beta {BETA:g}, stop", flush=True); break
        elif (int(os.environ.get("EZC_RAISE_ON_PEAK", "0")) and BETA < EZC_BMAX and k > STAGE0                # after a drop at this beta,
              and [h for h in hist if h["step"] < STAGE0] and [h for h in hist if h["step"] >= STAGE0]          #  raise again the moment the grey I passes
              and min(h["I_tar"] for h in hist if h["step"] >= STAGE0) < max(h["I_tar"] for h in hist if h["step"] < STAGE0)   #  the peak reached before
              and hist[-1]["I_tar"] > max(h["I_tar"] for h in hist if h["step"] < STAGE0)):                  #  this beta (no stall wait)
            BETA = min(float(os.environ.get("EZC_BFAC", "2")) * BETA, EZC_BMAX); STAGE0 = k
            print(f"[zonec] step {k}: grey I passed the previous peak, beta -> {BETA:g}", flush=True)
        elif BETA < EZC_BMAX and ((stall and (not int(os.environ.get("EZC_BFREEZE", "0")) or hist[-1]["I_tar"] >= 0.999 * max(h["I_tar"] for h in hist))) or n_st >= EZC_SMAX):   # EZC_BFREEZE=1: raise beta only on a stall AT the running peak of the grey I; after a drop beta stays frozen until I passes that peak
             BETA = min(float(os.environ.get("EZC_BFAC", "2")) * BETA, EZC_BMAX); STAGE0 = k   # EZC_BFAC (1.3)
        elif EZC_FINALPOLISH > 0 and FINAL < 0 and BETA >= EZC_BMAX and stall:
            with torch.no_grad():
                _b = (proj(filt(theta), BETA).clamp(0, 1) > 0.5).float()
                theta = (0.5 + float(os.environ.get("EZC_SOFT", "0.5")) * (_b - 0.5)) * VALID
                theta = torch.where(OPT, theta, (C_FIX if C_FIX is not None else theta))
            opt_m = torch.zeros_like(theta); opt_v = torch.zeros_like(theta); STAGE0 = k; FINAL = k
            print(f"[zonec] step {k}: final re-binarization at beta {BETA:g}, polishing {EZC_FINALPOLISH} steps", flush=True)
        beta = BETA
    elif EZC_SCHED == "log":
        beta = EZC_B0 * (EZC_BMAX / EZC_B0) ** min(1.0, k / max(EZC_LOGN, 1)); si = 0
    elif EZC_SCHED == "meep":
        si = min(max(k - EZC_SK0, 0) // EZC_SLEN, len(EZC_BETAS) - 1); beta = EZC_BETAS[si]
        if k > EZC_SK0 and (k - EZC_SK0) % EZC_SLEN == 0 and (k - EZC_SK0) // EZC_SLEN < len(EZC_BETAS):
            opt_m = torch.zeros_like(theta); opt_v = torch.zeros_like(theta); STAGE0 = k      # Adam reset at every beta stage (Meep restarts MMA)
        if k == EZC_SK0: opt_m = torch.zeros_like(theta); opt_v = torch.zeros_like(theta); STAGE0 = k
    else: beta = min(EZC_B0 * 2 ** (k // EZC_BSTEP), EZC_BMAX)
    _rb = int(os.environ.get("EZC_REBIN", "0"))                          # EZC_REBIN: periodic re-binarization:
    if _rb > 0 and k > 0 and k % _rb == 0:                                #  every EZC_REBIN steps theta <- soft(binarized design), the same soft start
        with torch.no_grad():                                             #  as step 0 (EZC_SOFT), Adam reset; this step's forward scores the re-binarized design
            _b = (proj(filt(theta), beta).clamp(0, 1) > 0.5).float()
            theta = (0.5 + float(os.environ.get("EZC_SOFT", "0.5")) * (_b - 0.5)) * VALID
            theta = torch.where(OPT, theta, (C_FIX if C_FIX is not None else theta))
        opt_m = torch.zeros_like(theta); opt_v = torch.zeros_like(theta); STAGE0 = k
    with torch.no_grad(): D = raster_f(_rho_of(theta, beta))
    Dq = D                                                                # the solver reads the global Dq (fill-fraction averaging handles grey)
    import gc as _gc; _gc.collect()
    _NR.offload(globals())                                                # FA_OPT_OFFLOAD=1: theta and the Adam moments wait in pinned host memory
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    I, g_oct, cnt_, _ = forward_score(); T_grey = dict(FA_LAST_T); G = region_gradient10(GT, g_oct, cnt_); t_fa = time.time() - t0
    _NR.reload(globals(), dev); t_opt0 = time.time()
    th = theta.clone().requires_grad_(True); rho = _rho_of(th, beta)
    Gc = cell_grad(G); gw = EZC_GW * min(1.0, max(k - EZC_SK0, 0) / max(EZC_GWRAMP, 1)) if EZC_GW > 0 else 0.0
    obj = (rho * Gc).sum() - gw * (4 * rho * (1 - rho) * OPT).sum() / OPT.sum()   # d/dtheta of I - w <grey> (I enters through Gc)
    bin_target = float("nan"); bin_pen = 0.0
    if EZC_BINRATE > 0 and k >= EZC_BINK0:                                # binarization-rate constraint: grey <= grey(EZC_BINK0) - EZC_BINRATE (k - EZC_BINK0 + 1)
        g_now = (4 * rho * (1 - rho) * OPT).sum() / OPT.sum()
        if BIN_G0 is None: BIN_G0 = float(g_now.detach())
        bin_target = max(0.0, BIN_G0 - EZC_BINRATE * (k - EZC_BINK0 + 1))
        _pen = torch.relu(g_now - bin_target) ** 2; bin_pen = float(_pen); obj = obj - EZC_BINW * _pen
    obj.backward(); gth = th.grad * VALID
    if os.environ.get("EZC_RESTART", "0") == "1":                         # per-px momentum restart (O'Donoghue & Candes 2015 adaptive restart): B4 centre
        opt_m = torch.where(opt_m * gth < 0, torch.zeros_like(opt_m), opt_m) #  gradient + -> - -> + over 8 steps while rho 0.53 -> 0.77 -> 0.53 (momentum overshoot)
    opt_m = 0.9 * opt_m + 0.1 * gth; opt_v = 0.999 * opt_v + 0.001 * gth ** 2
    ta = (k - STAGE0 + 1) if (EZC_SCHED == "meep" or _rb > 0) else (k + 1)
    mh = opt_m / (1 - 0.9 ** ta); vh = opt_v / (1 - 0.999 ** ta)
    if EZC_SCHED == "meep": lr = EZC_LRLAST if si == len(EZC_BETAS) - 1 else EZC_LR
    elif EZC_SCHED == "log": lr = EZC_LRLAST if k >= EZC_OUTER - EZC_LAST + 1 else EZC_LR
    else: lr = EZC_LR0 * (EZC_LR / EZC_LR0) ** min(1.0, k / max(EZC_LRDECAY, 1)) if EZC_LR0 > 0 else EZC_LR   # EZC_LR0 > 0: lr decays from EZC_LR0 to EZC_LR
    if float(os.environ.get("EZC_LRBETA", "0")) > 0: lr = min(lr, float(os.environ["EZC_LRBETA"]) / beta)   # after the schedule (2026-09-27 fix: was wedged between if/elif) rec.: lr = min(lr, c / beta) per beta stage
    _w = int(os.environ.get("EZC_WARM", "0"))                             # linear lr warm-up (EZC_REBIN: restarts after every re-binarization) over the first EZC_WARM steps (config I: Adam's first
    if _w > 0: lr = lr * min(1.0, ((k - STAGE0 if (_rb > 0 or os.environ.get("EZC_WARMSTAGE") == "1") else k) + 1) / _w)   #  step moved every px by ~lr = 0.5 and flipped the fill 0.48 -> 0.85)
    theta = (theta + lr * mh / (vh.sqrt() + 1e-12 * float(vh.max().sqrt() + 1e-30))).clamp(0, 1) * VALID
    theta = torch.where(OPT, theta, (C_FIX if C_FIX is not None else theta))
    rv = rho.detach()[OPT]; grey = float((4 * rv * (1 - rv)).mean()); fill = float(rv.mean()); del th, gth, mh, vh, obj, Gc
    t_opt = time.time() - t_opt0; _PL.event(k, "optimizer", t_opt)
    I_bin = float("nan"); qb = None; bin_metrics = None
    if k % EZC_BINEVAL == 0 or k == EZC_OUTER or (FINAL >= 0 and k == FINAL + EZC_FINALPOLISH):   # final polished step always scored
        qb = (D > 0.5); T_bin, _, I_bin = tmaps(qb); best_bin = max(best_bin, I_bin)
        if FA_OBJECTIVE and (k == EZC_OUTER or (FA_METRICS_EVERY > 0 and k % FA_METRICS_EVERY == 0)): bin_metrics = _NR.metrics(fa_objective(), _full_T(T_bin))
    t_sv0 = time.time()
    qsave = qb if qb is not None else (D > 0.5); hard = view20(qsave)
    _SAVER.save_npz(OUT / f"step_{k:04d}.npz", hard_eval_oct=np.packbits(hard), hard_shape=np.array(hard.shape), d10_quad=pack_q(qsave), d10_shape=np.array([H10, H10]),
                    I_tar=I, I_accepted=I, I_binary=I_bin, rho_cells=rho.detach().cpu().numpy().astype(np.float16) if FA_RHO_EVERY > 0 and k % FA_RHO_EVERY == 0 else np.empty(0,dtype=np.float16),
                    cell_shape=np.array([KZ, NRMAX, AMAX]), beta=beta)
    if qb is not None and I_bin >= best_bin:
        _SAVER.save_npz(OUT / 'best_binary.npz',d10_quad=pack_q(qb),d10_shape=np.array([H10,H10]),objective_value=I_bin,step=k,objective_kind=os.environ.get('FF_NEW_OBJECTIVE') or FA_OBJECTIVE)
    _SAVER.save_npz(OUT / "latest_mask.npz", hard_oct=np.packbits(hard), hard_shape=np.array(hard.shape), d10_quad=pack_q(qsave), d10_shape=np.array([H10, H10]), I_tar=I, step=k)
    if FA_GRAD_EVERY > 0 and k % FA_GRAD_EVERY == 0:                    # figure's gradient panel: float16 export (as grad_export.py writes)
        _sc = float(G.abs().max()) or 1.0; _SAVER.save_npz(OUT / f"grad_step{k:04d}.npz", compressed=True, G16=(G.float() / _sc).half().cpu().numpy(), scale=_sc)
    if T_grey:                                                            # pupil Jones maps: grey every step, snapshots every FA_TKEEP_EVERY, binary when scored
        _tg = {f"T_{kk}": v.cpu().numpy() for kk, v in T_grey.items()}
        _SAVER.save_npz(OUT / "optical" / "T_grey_latest.npz", compressed=False, step=k, wavelength_nm=np.array(LAM_NM), objective_value=I, **_tg)
        if FA_TKEEP_EVERY > 0 and k % FA_TKEEP_EVERY == 0: _SAVER.save_npz(OUT / "optical" / f"T_grey_step{k:04d}.npz", compressed=False, step=k, wavelength_nm=np.array(LAM_NM), objective_value=I, **_tg)
        if qb is not None:
            _SAVER.save_npz(OUT / "optical" / f"T_binary_step{k:04d}.npz", compressed=False, step=k, wavelength_nm=np.array(LAM_NM), objective_value=I_bin,
                            **{f"T_{kk}": v.cpu().numpy() for kk, v in _full_T(T_bin).items()})
    rec = {"step": k, "I_tar": I, "I_accepted": max([h["I_tar"] for h in hist] + [I]), "gate": "first" if k == 0 else "accept", "algo": "zonec", "region": f"C{k}",
           "I_binary": I_bin, "best_binary": best_bin, "beta": beta, "lr": lr, "grey": grey, "grey_weight": gw, "fill": fill, "t_forward_adjoint_s": t_fa,
           "t_optimizer_s": t_opt, "bin_target": bin_target, "bin_penalty": bin_pen, "binary_metrics": bin_metrics,
           "ls_min_px": L_MIN, "ls_gap_px": L_GAP, "rebin": bool(_rb > 0 and k > 0 and k % _rb == 0), "final_rebin": bool(FINAL == k)}
    _ck = {"theta": theta.detach().cpu(), "m": opt_m.detach().cpu(), "v": opt_v.detach().cpu()}   # host copies owned by the background writer
    rec["t_step_s"] = time.time() - t0; hist.append(rec)
    _st = dict(_ck, k=k, hist=[dict(h) for h in hist], best_bin=best_bin, beta=BETA if EZC_ADAPT else beta, stage0=STAGE0, final=FINAL, ramp0=RAMP0, bin_g0=BIN_G0)
    _SAVER.torch_save(_st, st_path)                                       # per-step checkpoint (tmp file + rename)
    _ke = int(os.environ.get("FA_STATE_KEEP_EVERY", "0"))                # 2026-10-01: resumable state history zonec/hist/state_stepXXXX.pt
    if _ke > 0 and k % _ke == 0: _SAVER.torch_save(_st, CD / "hist" / f"state_step{k:04d}.pt")
    del _ck, _st
    t_save = time.time() - t_sv0; _PL.event(k, "save", t_save); _NR.write_json(OUT / "history.json", hist)
    _PL.telemetry(k, dev); _PL.merge_ranks()
    print(f"step {k:3d}  optical_score(grey) {I:.5f}  binary {I_bin:.5f}  beta {beta:g}  grey {grey:.3f}  fill {fill:.3f}  "
          f"{rec['t_step_s']:.0f}s (forward+adjoint {t_fa:.0f}s, optimizer {t_opt:.1f}s, save queue {t_save:.1f}s)", flush=True)
    del rho
    if torch.cuda.is_available(): torch.cuda.empty_cache()
_SAVER.close(); _PL.merge_ranks()
print(f"FF_DONE (best binary I_tar {best_bin:.5f})", flush=True)
