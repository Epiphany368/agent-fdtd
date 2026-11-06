"""Tile solver of the full-aperture drivers (zonec_multi.py, ns_run.py execute it in parts): TorchFDTD reversible-CPML tile adjoints.

Design: density on the global 10 nm lattice (FDTD on 20 nm), SiN 700 nm extrusion (n=2.0) on SiO2. Each tile is solved with periodic
lateral faces and a margin FF_OVER around its core; the exit fields are sampled on the pupil cells and divided by a homogeneous-substrate
reference to give the local transmission (pupil Jones maps). The objective (fa_objectives.py) propagates these maps, and its gradient
flows back through the maps -> tile fields -> TorchFDTD plane adjoint -> density. I_tar is the name of the objective value.
"""
import sys, os, time, math, json, random
from pathlib import Path
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np, torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
from torchfdtd import (Project, Region, Source, FieldMonitor, SpectrumSettings, Boundaries, BoundaryFace, ReversibleCPMLPlaneSimulation, ReversibleCPMLOptions)
from torchfdtd import field_monitors as _fm, solver as _solver
def _mm(project):
    size = 0
    for raw in project.monitors:
        if not raw.enabled or raw.kind != "field": continue
        m = project.resolved_monitor(raw); n = len(_fm.plane_plan(project.region, m)["weights"]); nf = len(_fm.frequency_samples(m.spectrum)); nc = len(m.required_fields)
        size += n * nf * nc * 8 * 4 + n * nc * 8 * 24 + project.region.steps * 16
    return size
_fm.monitor_memory = _mm; _solver.monitor_memory = _mm
_ADMIT = float(os.environ.get("FA_ADMIT_FREE_FRAC", "0.8"))               # 2026-09-30: fraction of the free CUDA memory a solve may reserve (solver rule 0.8,
if _ADMIT != 0.8:                                                          #  unchanged by default; same logic as torchfdtd.cuda_memory.cuda_budget_limit)
    import torchfdtd.cuda_memory as _cm, torchfdtd.adjoint_memory as _am
    def _budget_limit(device, required, budget=None):
        device = torch.device(device); free, total = torch.cuda.mem_get_info(device); cap = total if budget is None else budget
        limit = min(cap, int(free * _ADMIT))
        if required <= limit or required > min(cap, int(total * _ADMIT)): return limit
        unused = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
        if unused and required <= min(cap, int(min(total, free + unused) * _ADMIT)):
            with torch.cuda.device(device): torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device); limit = min(cap, int(free * _ADMIT))
        return limit
    _cm.cuda_budget_limit = _budget_limit; _am.cuda_budget_limit = _budget_limit



INIT = os.environ.get("FF_INIT_DESIGN", "info")                      # FF_INIT_DESIGN (the examples set w240)
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
W_NM = np.load(sys.argv[1])                                           # bookkeeping width map of the pupil grid (nm); not used by the physics
N_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 120               # argv[4] (tiles per step) is ignored: every step evaluates all tiles
PITCH, H, OVER, MESH, STEPS, PMLC = 0.29, float(os.environ.get("FA_H_UM", "0.7")), float(os.environ.get("FF_OVER", "1.2")), 0.02, int(os.environ.get("FA_STEPS", "1200")), 12   # FF_OVER (2026-09-27): tile margin um (default 1.2); FA_H_UM / FA_STEPS (2026-09-30): layer height, time steps
assert abs(OVER / MESH - round(OVER / MESH)) < 1e-6 and OVER > 0, f"FF_OVER {OVER} um must be a positive multiple of the {MESH} um mesh"
TILE = float(os.environ.get("FF_TILE", "16.0"))                       # 16 um fits 48 GB (reservation 20 GB); 24 um for an 80 GB H100 (~45 GB)
GPU_BUDGET_GIB = float(os.environ.get("FF_GPU_GIB", "44")); HOST_BUDGET_GIB = float(os.environ.get("FF_HOST_GIB", "110"))   # lateral faces periodic (reversible adjoint): 1.2 um margin absorbs the wrap-around coupling
Z0 = 0.35; ZM = H + 0.5; R_AP = 104.0; N_SIN, N_SIO2 = float(os.environ.get("FA_N_DESIGN", "2.0")), float(os.environ.get("FA_N_SUB", "1.444"))          # domain z -1.25..1.25 -> physical -0.9..1.6          # monitor 0.5 um above the posts: lattice harmonics decayed, field band-limited
QPC = 2                                                                    # quadrature points per 0.29 um pupil cell and axis
LAM_NM = [int(v) for v in os.environ.get("FA_LAM_NM", "420,450,470,510,540,570,600,635,670").split(",")]   # FA_LAM_NM (2026-09-30): E2 450,540,635, E3 532
assert LAM_NM == sorted(LAM_NM), "FA_LAM_NM must be ascending"
L = TILE + 2 * OVER; LI = L                                      # periodic lateral faces: no lateral PML
NPX = int(round(210.0 / MESH)); XG0 = -105.0                     # global lattice: 10500 px, centre of px i at XG0 + (i+0.5)*MESH
NCELL = 720; LR = float(os.environ.get("FF_LR", "0.03"))                 # maximum of the cosine schedule
import lens_cfg; LENS = lens_cfg.get()                                    # FF_LENS (e.g. d50f80): other lens geometry; unset = the 208 um lens above
if LENS:
    R_AP, NPX, XG0, NCELL = LENS["R_AP"], LENS["NPX"], LENS["XG0"], LENS["PUPIL_GRID"]
    PITCH=LENS['PITCH_UM']
    assert W_NM.shape == (NCELL, NCELL), f"widths file {sys.argv[1]} is {W_NM.shape}, lens {LENS['name']} needs {NCELL}^2"
    lens_cfg.write_run_json(OUT, LENS); print(f"[ff_opt] lens {LENS['name']}: R_AP {R_AP} um, F {LENS['F_UM']} um, z_obj {LENS['OBJ_DIST_UM']} um, "
                                              f"pupil {NCELL}, lattice {NPX} px from {XG0} um", flush=True)
# binary design of the legacy loops (binary warm start): latent -> sigmoid -> tanh projection -> hard mask,
# straight-through; lr 0.03 cosine to 0.2x; beta 9 -> 20 geometric, no hold; accept gate every step
BETA_MIN = float(os.environ.get("FF_BETA_MIN", "9")); BETA_MAX = 20.0; BETA_HOLD = float(os.environ.get("FF_BETA_HOLD", "0"))
BETA = {"v": BETA_MIN}
dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))); torch.manual_seed(0); random.seed(0)

# Only new optical objectives are allowed in this research package.
IS_MAIN = os.environ.get('FF_MULTI_RANK', '0') == '0'
def i_tar_of(*args): raise RuntimeError('Information objective is forbidden')
FA_OBJECTIVE = os.environ.get("FA_OBJECTIVE", "")                         # 2026-09-30: e1_achromat | e2_rgb_holo | e3_volume_holo (fa_objectives.make_objective),
_FA_OBJ = []                                                             #  built on first use, i.e. only on the rank that evaluates it
def fa_objective():
    if not _FA_OBJ:
        import fa_objective_loader
        _FA_OBJ.append(fa_objective_loader.load(FA_OBJECTIVE, lam_nm=LAM_NM, pitch_um=PITCH, pupil_n=NCELL, aperture_um=2 * R_AP, device=dev, out_dir=OUT,
                                                  focal_um=float(os.environ.get("FF_FOCAL_UM", LENS["F_UM"] if LENS else 2 * R_AP / 0.6))))
    return _FA_OBJ[0]
# ---------------- tile solver (one project, epsilon supplied per tile) ----------------
_SRC_BAND = [float(v) for v in os.environ.get("FA_SRC_BAND_UM", "0.40,0.70").split(",")]   # 2026-09-30: broadband pulse band (um); must contain FA_LAM_NM
assert _SRC_BAND[0] <= min(LAM_NM) * 1e-3 and max(LAM_NM) * 1e-3 <= _SRC_BAND[1], f"FA_LAM_NM {LAM_NM} outside the source band {_SRC_BAND} um"
assert 0.0 < H <= 0.85, "FA_H_UM: the exit monitor (H + 0.5 um) must stay below the upper CPML (from 1.36 um in the 2.5 um tile domain)"
spec = SpectrumSettings(sampling="custom", custom_frequencies_hz=sorted(299792458.0 / (l * 1e-9) for l in LAM_NM), apodization="none")
project = Project(name="ff tile",
    region=Region(dimension="3d", size=(L, L, 2.5), mesh=MESH, mesh_type="uniform", mesh_auto_refine=False,
                  boundaries=Boundaries(x_min=BoundaryFace(kind="periodic"), x_max=BoundaryFace(kind="periodic"), y_min=BoundaryFace(kind="periodic"), y_max=BoundaryFace(kind="periodic")),
                  steps=STEPS, pml_cells=PMLC, backend="cuda", precision="float32", material_sampling="yee", interface_method="staircase",
                  memory_mode="budgeted", snapshot_interval=10000),
    structures=[], sources=[Source(name="plane", kind="plane", injection="soft", normal="z", direction="+", center=(0, 0, -0.55 - Z0), size=(LI, LI, 0),
                                   component="Ex", pulse="broadband", time_definition="wavelength", wavelength_start=_SRC_BAND[0], wavelength_stop=_SRC_BAND[1])],
    monitors=[FieldMonitor(id="xy", name="exit", normal="z", center=(0, 0, ZM - Z0), size=(LI, LI, 0), spectrum=spec, spatial_interpolation="specified",
                           record_fields=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], record_poynting=[], record_flux=False)])
project = Project.model_validate(project.model_dump())
def tile_project(x0, y0):
    """Tile project whose monitor spans whole pupil cells (global cell edges at multiples of 0.29 um) inside the non-PML region."""
    lo_x = math.floor((x0 - TILE / 2) / PITCH) * PITCH; hi_x = math.ceil((x0 + TILE / 2) / PITCH) * PITCH
    lo_y = math.floor((y0 - TILE / 2) / PITCH) * PITCH; hi_y = math.ceil((y0 + TILE / 2) / PITCH) * PITCH
    # periodic lateral faces: the whole tile width is physical; the monitor stays 0.5 um clear of the wrap seam
    p = project.model_copy(deep=True); m = p.monitors[0]
    m.center = ((lo_x + hi_x) / 2 - x0, (lo_y + hi_y) / 2 - y0, ZM - Z0); m.size = (hi_x - lo_x, hi_y - lo_y, 0)
    assert abs(m.center[0]) + m.size[0] / 2 <= L / 2 - 0.5 and abs(m.center[1]) + m.size[1] / 2 <= L / 2 - 0.5
    counts = (int(round((hi_x - lo_x) / PITCH)) * QPC, int(round((hi_y - lo_y) / PITCH)) * QPC)
    return Project.model_validate(p.model_dump()), counts
region = project.region; shape = tuple(region.shape); nodes = [np.asarray(v) for v in region.mesh_nodes]
zc = 0.5 * (nodes[2][:-1] + nodes[2][1:]) + Z0; xc = 0.5 * (nodes[0][:-1] + nodes[0][1:])
assert shape[0] == shape[1] == int(round(L / MESH)) and np.allclose(np.diff(xc), MESH, atol=1e-6)
sub = torch.tensor(zc < 0.0, device=dev); pil = torch.tensor((zc >= 0.0) & (zc < H), device=dev)
ADJ = ReversibleCPMLOptions(trace_storage=os.environ.get("FF_TRACE", "cpu"), trace_transfers="async" if os.environ.get("FF_TRACE", "cpu") == "cpu" else "sync", trace_chunk_steps=32,
                            resident_budget_bytes=int(GPU_BUDGET_GIB * 1024 ** 3), host_budget_bytes=int(HOST_BUDGET_GIB * 1024 ** 3))
ADJ_KEEP = ReversibleCPMLOptions(trace_storage="cpu", trace_transfers="async", trace_chunk_steps=32,   # FA_KEEP_TRACE=1: kept units (fa_keep_trace.py)
                                 resident_budget_bytes=int(GPU_BUDGET_GIB * 1024 ** 3), host_budget_bytes=int(HOST_BUDGET_GIB * 1024 ** 3))
FREQ_T = torch.tensor(sorted(299792458.0 / (l * 1e-9) for l in LAM_NM), dtype=torch.float32)
models = {}; MODELS_BY_GEOM = {}                                          # 2026-09-30: a model holds no tile position (only the local plan), so tiles
def shared_model(p, counts, components, options=None):                    #  whose local project is identical (same monitor offset/size) share one model
    import hashlib as _hl
    options = ADJ if options is None else options
    h = _hl.sha256(f"{json.dumps(p.model_dump(mode='json'), sort_keys=True)}|{counts}|{components}|{options!r}".encode()).hexdigest()
    if h not in MODELS_BY_GEOM:
        t0 = time.time(); m = ReversibleCPMLPlaneSimulation(p, options, quadrature_counts={"xy": counts})
        m.plan(FREQ_T, device="cuda", material_components=components, block_size=32); MODELS_BY_GEOM[h] = m
        print(f"  built reversible plane model ({components} eps component(s), {p.sources[0].component} source, {counts[0]}x{counts[1]} pts), interior_z {m.interior_z}, "
              f"in {time.time()-t0:.0f} s", flush=True)
    return MODELS_BY_GEOM[h]
def model_for(key):
    if key not in models:
        p, counts = tile_project(*key); models[key] = shared_model(p, counts, 1)
    return models[key]
freqs = sorted(299792458.0 / (l * 1e-9) for l in LAM_NM); lam_order = np.argsort(-np.array(freqs))    # freqs ascending -> lambda descending; LAM_NM ascending
def eps_of(rho_tile):                                                       # (nx,ny) -> (nx,ny,nz)
    eps = torch.ones(shape, device=dev, dtype=torch.float32)
    eps = torch.where(sub[None, None, :], torch.full_like(eps, N_SIO2 ** 2), eps)
    eps = torch.where(pil[None, None, :], (1.0 + rho_tile[:, :, None] * (N_SIN ** 2 - 1.0)).expand(shape), eps)
    return eps.contiguous()                                                # scalar map (nx,ny,nz); the solver admits material_components=1
EPS_BG = None
def eps_background():
    global EPS_BG
    if EPS_BG is None: EPS_BG = eps_of(torch.zeros(shape[:2], device=dev)).detach()
    return EPS_BG
def run_tile(rho_tile, key):
    """Exit-plane Ex (Nl ascending in LAM_NM, Npts) at the tile's quadrature points, plus local point coordinates."""
    det = model_for(key)(eps_of(rho_tile), FREQ_T.to(dev), fixed_epsilon=eps_background(), block_size=32)["xy"]
    Ex = det.fields[:, :, 0]; Ex = Ex[torch.as_tensor(lam_order.copy(), device=Ex.device)]         # freqs ascending -> LAM_NM ascending
    return Ex, torch.as_tensor(det.points_um, device=dev, dtype=torch.float32)

# ---------------- reference: homogeneous SiO2 (incident field at the monitor plane) ----------------
ref_path = OUT / "reference.npz"
if ref_path.exists():
    dref=np.load(ref_path)
    if 'incident_flux' not in dref: raise RuntimeError('Reference lacks six-field incident flux. Use a new versioned run directory.')
    E_ref=torch.tensor(dref['E_ref'],device=dev)
    P_ref=torch.tensor(dref['incident_flux'],device=dev)
else:
    with torch.no_grad():
        m=model_for((8.0,8.0))
        det=m(torch.full(shape,N_SIO2**2,device=dev),FREQ_T.to(dev),fixed_epsilon=torch.full(shape,N_SIO2**2,device=dev),block_size=32)['xy']
        fields=det.fields[torch.as_tensor(lam_order.copy(),device=dev)]
        pts=torch.as_tensor(det.points_um,device=dev)
        keep=(pts[:,0].abs()<8)&(pts[:,1].abs()<8)
        E_ref=fields[:,keep,0].mean(1)
        P_ref=(.5*(fields[:,keep,0]*fields[:,keep,4].conj()-fields[:,keep,1]*fields[:,keep,3].conj()).real).mean(1)
    if not bool(torch.isfinite(P_ref).all() and (P_ref>0).all()): raise RuntimeError('Invalid measured incident Poynting calibration')
    np.savez(ref_path,E_ref=E_ref.cpu().numpy(),incident_flux=P_ref.cpu().numpy(),fields=fields.cpu().numpy(),points_um=pts.cpu().numpy(),wavelength_nm=LAM_NM,convention='exp(+2 pi i f t)',units='reduced spectral Poynting; efficiency ratios dimensionless')
    print('Measured reference flux',P_ref.cpu().tolist(),flush=True)
JONES_SCALE=torch.sqrt(E_ref.abs().square()/(2*P_ref)).to(torch.float32)[:,None,None]

# ---------------- design parameters (octant) and D4 expansion ----------------
theta=torch.zeros((),device=dev)
print('Independent grey density initialization, no design seed',flush=True)
STEP0 = 0
if os.environ.get("FF_INIT"):                                             # resume: octant parameters from a saved step (Adam state starts fresh)
    _d = np.load(os.environ["FF_INIT"]); STEP0 = int(os.path.basename(os.environ["FF_INIT"]).split("_")[1].split(".")[0]) + 1
    with torch.no_grad(): theta[NPX // 2:, NPX // 2:] = torch.tensor(_d["theta_oct"].astype(np.float32), device=dev)
    print(f"resumed octant parameters from {os.environ['FF_INIT']} -> starting at step {STEP0}", flush=True)
_fa_cache_path=Path(os.environ.get('FA_NEUTRAL_CACHE','__missing__'))
if _fa_cache_path.exists():
    _fa_geo=np.load(_fa_cache_path)
    assert int(_fa_geo['H10'])==NPX
    oct_mask=torch.tensor(_fa_geo['oct_mask'],device=dev,dtype=torch.bool)
    ii=torch.arange(NPX,device=dev);half=NPX//2
else:
    oct_mask = torch.zeros((NPX, NPX), dtype=torch.bool, device=dev); ii = torch.arange(NPX, device=dev)
    half = NPX // 2; oct_mask[(ii[:, None] >= half) & (ii[None, :] >= half) & (ii[:, None] - half >= ii[None, :] - half)] = True
    _xc = XG0 + (ii.float() + 0.5) * MESH
    oct_mask &= torch.hypot(_xc[:, None], _xc[None, :]) <= R_AP                  # design region: octant pixels inside the 104 um aperture; outside stays air
    del _xc
theta.requires_grad_(True)
def oct_index(i, j):                                                       # D4 representative of lattice index (i, j) in the octant
    a = torch.where(i >= half, i, NPX - 1 - i); b = torch.where(j >= half, j, NPX - 1 - j)
    hi = torch.maximum(a, b); lo = torch.minimum(a, b); return hi, lo
def tanh_project(r):                                                        # tanh projection of a density, eta = 0.5
    b = BETA["v"]; t = math.tanh(0.5 * b)
    return (t + torch.tanh(b * (r - 0.5))) / (2 * t)
def rho_tile_of(theta, i0, j0):                                             # tile lattice window: latent -> sigmoid -> projection -> hard (straight-through)
    ti = torch.arange(i0, i0 + shape[0], device=dev)[:, None].expand(shape[0], shape[1]); tj = torch.arange(j0, j0 + shape[1], device=dev)[None, :].expand(shape[0], shape[1])
    inside = (ti >= 0) & (ti < NPX) & (tj >= 0) & (tj < NPX)                      # lattice covers the aperture; outside it rho = 0
    hi, lo = oct_index(ti.clamp(0, NPX - 1), tj.clamp(0, NPX - 1)); r = torch.where(inside, torch.sigmoid(theta[hi, lo]), torch.zeros((), device=dev))
    soft = tanh_project(r); hard = (soft >= 0.5).to(soft.dtype)
    return hard + (soft - soft.detach())                                  # value = the hard mask exactly; gradient through the projection

# tiles: octant centres X0 >= Y0 > 0 that intersect the aperture
c0 = (math.ceil(R_AP / TILE) - 0.5) * TILE
tiles = [(x0, y0) for x0 in np.arange(TILE / 2, c0 + 0.1, TILE) for y0 in np.arange(TILE / 2, c0 + 0.1, TILE)
         if x0 >= y0 and math.hypot(max(x0 - TILE / 2, 0), max(y0 - TILE / 2, 0)) <= R_AP + 0.5]
SYM = os.environ.get("FF_SYM", "D4").upper()                              # FF_SYM=C4 (2026-09-27): design = the whole quadrant x, y >= 0, lens = its
if SYM not in ("D4", "C4"): raise SystemExit(f"unknown FF_SYM {SYM}")     #  0/90/180/270 deg rotations (no mirror); default D4 unchanged
if SYM == "C4":                                                           # C4: every quadrant tile (not only x0 >= y0); Q2..Q4 come from rotation
    assert os.environ.get("FF_ALGO") == "zonec" and os.environ.get("FF_L9_DUALPOL") == "1", "FF_SYM=C4 is implemented for FF_ALGO=zonec + FF_L9_DUALPOL=1 only"
    tiles = [(x0, y0) for x0 in np.arange(TILE / 2, c0 + 0.1, TILE) for y0 in np.arange(TILE / 2, c0 + 0.1, TILE)
             if math.hypot(max(x0 - TILE / 2, 0), max(y0 - TILE / 2, 0)) <= R_AP + 0.5]
def tile_window(x0, y0):                                                    # lattice index of the tile's first px (tile spans x0 +- L/2)
    return int(round((x0 - L / 2 - XG0) / MESH)), int(round((y0 - L / 2 - XG0) / MESH))
def aperture_fraction(key):
    x0, y0 = key; xs_ = np.linspace(x0 - TILE / 2, x0 + TILE / 2, 81); ys_ = np.linspace(y0 - TILE / 2, y0 + TILE / 2, 81)
    GX_, GY_ = np.meshgrid(xs_, ys_, indexing="ij"); return float((np.hypot(GX_, GY_) <= R_AP).mean())      # fraction of the tile area inside the aperture
tile_frac = {k: aperture_fraction(k) for k in tiles}
def aperture_cells(key):                                                   # pupil cells of the tile core (octant part for D4) whose centre lies in the aperture
    x0, y0 = key; cc_ = (np.arange(NCELL) - NCELL // 2 + 0.5) * PITCH
    inx = (cc_ >= x0 - TILE / 2) & (cc_ < x0 + TILE / 2); iny = (cc_ >= y0 - TILE / 2) & (cc_ < y0 + TILE / 2)
    m = (cc_[:, None] ** 2 + cc_[None, :] ** 2 <= R_AP ** 2) & inx[:, None] & iny[None, :] & (cc_[:, None] >= 0) & (cc_[None, :] >= 0)
    if SYM == "D4": m &= cc_[:, None] >= cc_[None, :]
    return int(m.sum())
if os.environ.get("FA_TILE_SELECT", "frac") == "cells":                  # 2026-09-30: keep exactly the tiles that carry an aperture pupil cell (the 2 % area
    tiles = [k for k in tiles if aperture_cells(k) > 0]                   #  cut left rim cells without a tile, i.e. dark in the assembled pupil)
else:
    tiles = [k for k in tiles if tile_frac[k] > 0.02]
print(f"{len(tiles)} octant tiles; lattice {NPX}^2; optimizing {int(oct_mask.sum())} octant px", flush=True)
print("aperture fraction per tile:", {f"{k[0]:.0f},{k[1]:.0f}": round(v, 2) for k, v in tile_frac.items()}, flush=True)

# ---------------- pupil-cell assembly from tile fields ----------------
cell_counts = torch.zeros((NCELL, NCELL), device=dev)
def cell_ids(pts, x0, y0):
    gx = pts[:, 0] + x0; gy = pts[:, 1] + y0
    keep = (gx >= x0 - TILE / 2) & (gx < x0 + TILE / 2) & (gy >= y0 - TILE / 2) & (gy < y0 + TILE / 2) & (gx >= 0) & (gy >= 0) & (gx >= gy)   # octant px only
    ci = torch.floor(gx / PITCH).long() + NCELL // 2; cj = torch.floor(gy / PITCH).long() + NCELL // 2
    keep &= (ci >= 0) & (ci < NCELL) & (cj >= 0) & (cj < NCELL)
    return keep, ci, cj
def d4_expand_cells(t_oct):                                                 # (Nl,720,720) filled on octant cells (i>=j>=centre) -> full pupil
    ii = torch.arange(NCELL, device=dev); a = torch.where(ii >= NCELL // 2, ii, NCELL - 1 - ii)
    A = a[:, None].expand(NCELL, NCELL); B = a[None, :].expand(NCELL, NCELL)
    return t_oct[:, torch.maximum(A, B), torch.minimum(A, B)]
if SYM == "C4":
    def cell_ids(pts, x0, y0):                                             # C4: every quadrant point of the tile core (no gx >= gy cut)
        gx = pts[:, 0] + x0; gy = pts[:, 1] + y0
        keep = (gx >= x0 - TILE / 2) & (gx < x0 + TILE / 2) & (gy >= y0 - TILE / 2) & (gy < y0 + TILE / 2) & (gx >= 0) & (gy >= 0)
        ci = torch.floor(gx / PITCH).long() + NCELL // 2; cj = torch.floor(gy / PITCH).long() + NCELL // 2
        keep &= (ci >= 0) & (ci < NCELL) & (cj >= 0) & (cj < NCELL)
        return keep, ci, cj

cache = {}                                                                  # tile key -> (Ex.detach(), pts) at current theta
def tile_cells(key, Ex, pts):
    x0, y0 = key; keep, ci, cj = cell_ids(pts, x0, y0); flat = (ci * NCELL + cj)[keep]
    acc = torch.zeros((len(LAM_NM), NCELL * NCELL), dtype=torch.complex64, device=dev).index_add(1, flat, Ex[:, keep]).view(len(LAM_NM), NCELL, NCELL)
    cnt = torch.zeros(NCELL * NCELL, device=dev).index_add(0, flat, torch.ones(int(keep.sum()), device=dev)).view(NCELL, NCELL)
    return acc, cnt
CACHE_DIR = OUT / "nf_cache"; CACHE_DIR.mkdir(exist_ok=True)
import hashlib
def tile_hash(rho_tile):
    return hashlib.sha1(np.ascontiguousarray(rho_tile.detach().cpu().numpy().astype(np.float16).astype(np.float32).astype(np.float16))).hexdigest()[:16]
NF_TAG = "" if OVER == 1.2 else f"_ov{int(round(OVER / MESH))}"            # FF_OVER in the near-field cache key (default names unchanged)
def cache_path(key, h): return CACHE_DIR / f"tile_{key[0]:.0f}_{key[1]:.0f}_{h}{NF_TAG}.npz"
def refresh(keys):
    """Cached exit fields survive a process kill: each tile's field is stored on disk keyed by the hash of its rho window."""
    for key in keys:
        i0, j0 = tile_window(*key)
        with torch.no_grad():
            rt = rho_tile_of(theta, i0, j0); h = tile_hash(rt); f = cache_path(key, h)
            if f.exists():
                d = np.load(f); cache[key] = (torch.tensor(d["Ex"], device=dev), torch.tensor(d["pts"], device=dev)); continue
            Ex, pts = run_tile(rt, key); cache[key] = (Ex.detach(), pts)
            np.savez(f, Ex=Ex.detach().cpu().numpy(), pts=pts.cpu().numpy())
def assemble():
    """t_map (Nl,720,720) from the cache (no grad) and the per-cell sample counts."""
    acc = torch.zeros((len(LAM_NM), NCELL, NCELL), dtype=torch.complex64, device=dev); cnt = torch.zeros((NCELL, NCELL), device=dev)
    for key in tiles:
        a, c = tile_cells(key, *cache[key]); acc = acc + a; cnt = cnt + c
    t_oct = acc / cnt.clamp(min=1)[None] / E_ref[:, None, None] * JONES_SCALE
    return d4_expand_cells(t_oct), cnt
def tile_grad(key, grad_t_oct, cnt):
    """dI/d(hard density of this tile) by the reversible adjoint. Depends only on the hard mask and grad_t_oct, so it is reusable."""
    i0, j0 = tile_window(*key)
    with torch.no_grad(): leaf = rho_tile_of(theta, i0, j0).clone()
    leaf.requires_grad_(True)
    Ex, pts = run_tile(leaf, key)
    a, _ = tile_cells(key, Ex, pts)
    (a / cnt.clamp(min=1)[None] / E_ref[:, None, None] * JONES_SCALE).backward(gradient=grad_t_oct)
    return leaf.grad.detach()
OWNGRAD = os.environ.get("FF_OWNGRAD", "0") == "1"                         # FF_OWNGRAD=1 (2026-09-27, tile seams): each tile's design gradient is kept
if OWNGRAD:                                                                #  on its own TILE x TILE core only (margin px zeroed), so every lens px takes
    assert os.environ.get("FF_ALGO") in ("zonec", "zone", "levelset10"), "FF_OWNGRAD is applied in region_gradient10 (FF_ALGO zonec / zone / levelset10)"   #  its gradient from the one tile that owns it
    print(f"[ff_opt] FF_OWNGRAD: tile gradient masked to the {TILE} um core (margin {OVER} um zeroed)", flush=True)
def own_grad(g):                                                           # (nx, ny[, c]) tile-window gradient -> margin px zeroed (FF_OWNGRAD=1), else g itself
    if not OWNGRAD: return g
    n0 = int(round(OVER / MESH)); n1 = n0 + int(round(TILE / MESH)); assert tuple(g.shape[:2]) == (n1 + n0, n1 + n0), f"tile gradient {tuple(g.shape)} is not the {n1 + n0} px window"
    m = torch.zeros(g.shape[:2], dtype=g.dtype, device=g.device); m[n0:n1, n0:n1] = 1
    return g * (m[..., None] if g.dim() == 3 else m)
def chain_to_theta(key, g_rho):                                            # straight-through VJP at the current beta; no solver call
    rho_tile_of(theta, *tile_window(*key)).backward(gradient=own_grad(g_rho))

import copy
# ---------------- solver diagnostic: one reduction per large chunk instead of per 64 Ki-lane block ----------------
# reversible_cpml._interior_scale (peak / L2 of the interior E,H every 64 steps, used only for the reconstruction-drift
# check) splits a 920x920x~100x3 field into ~7,700 blocks and runs ~8 small ops on each: ~1.2e6 ops per tile forward,
# which the A100 profile showed as the largest CPU cost. Same peak exactly; L2 differs only by float64 summation order.
import torchfdtd.reversible_cpml as _rc
_ORIG_INTERIOR_SCALE = _rc._interior_scale
def _interior_scale_chunked(system, a, b, lanes_per_chunk=1 << 24):
    square = torch.zeros((), dtype=torch.float64, device=system.device); peak = torch.zeros((), dtype=torch.float32, device=system.device)
    for field in system.state()[:2]:
        x = field[:, :, a:b + 1]; per = max(1, lanes_per_chunk // (x[0].numel() * (2 if x.is_complex() else 1)))
        for r in range(0, x.shape[0], per):
            blk = x[r:r + per]; peak = torch.maximum(peak, blk.abs().amax())
            lanes = torch.view_as_real(blk) if blk.is_complex() else blk
            square.add_(lanes.to(torch.float64).square().sum())
    maximum, norm = float(peak), math.sqrt(float(square))
    if not math.isfinite(maximum) or not math.isfinite(norm): raise RuntimeError('Reversible interior fields became nonfinite.')
    return maximum, norm
if os.environ.get("FF_FAST_DIAG", "1") == "1": _rc._interior_scale = _interior_scale_chunked
# The installed solver's _require_finite reads one GPU bool per 64 Ki elements: ~62,000 host syncs on the 4e9-element
# boundary trace of every forward. Same checks, results accumulated on the GPU and read once per tensor.
def _require_finite_once(value, message, chunk, lanes=1 << 28):
    flat = value.reshape(-1); per = max(1, lanes // 2) if value.is_complex() else lanes; ok = torch.ones((), dtype=torch.bool, device=flat.device)
    for s in range(0, flat.numel(), per): ok &= torch.isfinite(flat[s:s + per]).all()
    if not bool(ok): raise RuntimeError(message)
_ORIG_RUN = _rc.ReversibleCPMLSimulation._run; _ORIG_REQUIRE_FINITE = _rc._require_finite
def _forward_only(epsilon, project, options, interval, report, spectral):
    """No-grad forward with the recorded path's exact update order (update_E, inject E, update_H, inject H, observe,
    32-step spectrum accumulate) minus what only the adjoint needs: boundary trace copies, 64-step reconstruction scale,
    terminal interior copies. Returns None when the fused CUDA kernel is unavailable (caller falls back)."""
    if not epsilon.is_cuda: return None
    system = _rc._System(project, epsilon.detach(), prepare_kernels=False, observation_monitors=None if spectral is None else spectral.observers)
    from torchfdtd.cuda_kernels import FusedYeeCUDA
    from torchfdtd.cuda_complex import FusedComplexYeeCUDA
    system.kernel = (FusedComplexYeeCUDA if system.field_dtype == torch.complex64 else FusedYeeCUDA)(system.grid, direct_views=True)
    steps = project.region.steps; block_size = steps if spectral is None else spectral.block_size
    samples = system.grid.E.new_empty((block_size, len(system.monitors))); signals = samples if spectral is None else spectral.zeros()
    started = time.perf_counter()
    for step in range(steps):
        e, h, *_ = system.state()
        system.kernel.update_E(); system.inject(e, 'E', step)
        system.kernel.update_H(); system.inject(h, 'H', step)
        system.current_step = step + 1
        row = step % block_size; samples[row] = system.observe(system.state())
        if spectral is not None and (row + 1 == block_size or step + 1 == steps): spectral.accumulate(signals, samples[:row + 1], step - row)
    _require_finite_once(signals, 'Forward-only CPML observations became nonfinite.', 0)
    e, h, *_ = system.state(); _require_finite_once(e, 'Forward-only E became nonfinite.', 0); _require_finite_once(h, 'Forward-only H became nonfinite.', 0)
    torch.cuda.synchronize(epsilon.device); report.update(forward_seconds=time.perf_counter() - started, forward_only=True)
    return signals
def _run_fast(self, epsilon, spectral, *, fixed_epsilon):
    """_run with the material checks read once per map, and the forward-only path when no gradient can be requested."""
    project, interval = self._snapshot()
    for name, value in (('epsilon', epsilon), ('fixed_epsilon', fixed_epsilon)):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type not in ('cpu', 'cuda') or value.layout != torch.strided
                or not value.is_contiguous() or value.is_conj() or value.is_neg() or tuple(value.shape) not in (project.region.shape, (*project.region.shape, 3))):
            raise ValueError(f'{name} must be a contiguous resolved scalar FP32 or diagonal FP32 CPU/CUDA tensor matching the region shape.')
    if fixed_epsilon.requires_grad: raise ValueError('fixed_epsilon must not require gradients. Only the explicit reconstruction interior is differentiable.')
    if fixed_epsilon.device != epsilon.device or fixed_epsilon.shape != epsilon.shape: raise ValueError('epsilon and fixed_epsilon must share device and shape.')
    from torchfdtd.reversible_cpml_memory import _cpml_reversible_reservation
    reservation = _cpml_reversible_reservation(project, self.options, epsilon.device, interval, material_components=3 if epsilon.ndim == 4 else 1, spectral=spectral)
    for value in (epsilon, fixed_epsilon):
        ok = torch.ones((), dtype=torch.bool, device=value.device); flat = value.reshape(-1)
        for s in range(0, flat.numel(), 1 << 26): blk = flat[s:s + (1 << 26)]; ok &= torch.isfinite(blk).all() & (blk >= 1).all()
        if not bool(ok): raise ValueError('The recorded CPML CFL contract requires finite epsilon >= 1 in both maps.')
    if epsilon.is_cuda:
        from torchfdtd.cuda_bootstrap import prepare_cuda_kernels
        prepare_cuda_kernels()
    a, b = interval; effective = fixed_epsilon.clone(); effective[:, :, a:b + 1] = epsilon[:, :, a:b + 1]
    report = dict(experimental=True, adjoint='recorded-interface CPML reconstruction', reconstruction_interval_z=[a, b],
                  reconstruction_tolerance=self.options.reconstruction_tolerance, backend='fused CUDA' if epsilon.is_cuda else 'torch CPU', **reservation)
    signals = None
    if FAST_NOGRAD and not (torch.is_grad_enabled() and epsilon.requires_grad):
        with torch.no_grad(): signals = _forward_only(effective, project, self.options, interval, report, spectral)
    if signals is None: signals = _rc._RecordedCPML.apply(effective, project, self.options, interval, report, spectral)
    if spectral is not None: return spectral.result(signals, report)
    return _rc.DifferentiableResult(signals, project.region.time_step, tuple(m.component for m in project.monitors if m.enabled), report)
FAST_NOGRAD = os.environ.get("FF_FAST_NOGRAD", "1") == "1"
if os.environ.get("FF_FAST_SYNC", "1") == "1":
    _rc._require_finite = _require_finite_once; _rc.ReversibleCPMLSimulation._run = _run_fast
if os.environ.get("FF_FAST_SPEC", "1") == "1":                              # spectral DFT blocks: cached index tensors, bit-identical (fast_spec_patch.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import fast_spec_patch; fast_spec_patch.enable(True)
if os.environ.get("FF_B200_KERNELS", "0") == "1":                          # 2026-09-30: b200_kernels.py (reconstruction on the pillar layers, fused reverse E +
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import b200_kernels; b200_kernels.enable(globals())   #  VJP, launch bounds, block autotune);
                                                                          #  A100 16 um tile: adjoint 1.36x, gradient rel. L2 4e-7
# ---------------- optimization loop ----------------
# FF_ALGO=block (default): block-coordinate binary trust region, region adjoints only (loop_block.py).
# FF_ALGO=greedy: whole-lens boundary-greedy trust region on the binary mask (loop_greedy.py).
# FF_ALGO=cr: straight-through Adam with an accept gate (loop_cr.py).
# FF_ALGO=cont: relaxed-density continuation after Kim et al. Sci. Adv. 2024 (loop_cont.py); binary scored every step.
_loop = {"block": "loop_block.py", "greedy": "loop_greedy.py", "cr": "loop_cr.py", "cont": "loop_cont.py", "probe": "loop_probe.py", "verify": "loop_verify.py", "levelset": "loop_levelset.py", "levelset10": "loop_levelset10.py", "polcheck": "loop_polcheck.py", "score10": "loop_score10.py", "spacemap": "loop_spacemap.py", "elem": "loop_elem.py", "band": "loop_band.py", "zone": "loop_zone.py", "zonec": "loop_zonec.py"}[os.environ.get("FF_ALGO", "block")]
exec(compile(open(Path(os.path.abspath(__file__)).with_name(_loop), encoding="utf-8").read(), _loop, "exec"))




