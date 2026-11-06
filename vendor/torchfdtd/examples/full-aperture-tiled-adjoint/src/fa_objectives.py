"""Optical objectives of the FA examples E1 (achromatic lens), E2 (RGB hologram), E3 (volumetric point hologram).

    import fa_objectives
    obj = fa_objectives.make_objective('e1_achromat')   # 'e2_rgb_holo' | 'e3_volume_holo'; make_objective(name, cfg) overrides DEFAULTS
    J = obj(T)                                           # 0-dim float64 tensor, maximized; J.backward() fills T[k].grad
    rep = obj.metrics(T)                                 # no grad: rep['scalars'] (JSON-safe), rep['arrays'] (numpy, np.savez)

Input T, the same layout jones_objectives.optical_objective takes (`_jones_expand(leaf)` of the zonec/levelset10 assembly
for the D4 lens, or the full-pupil assembly without symmetry copies for E2/E3):
    T['xx'], T['yx']  exit-plane Ex, Ey for an x-polarized incident plane wave
    T['xy'], T['yy']  exit-plane Ex, Ey for a y-polarized incident plane wave (read only if 'y' in cfg['pols'], i.e. E1)
    each (Nl, m, m) complex64/complex128, any device. Axis 0 = wavelength in cfg['lam_nm'] order (or pass
    obj(T, lam_nm_T=[...]) when T carries more wavelengths), axis 1 = x, axis 2 = y, cell (i, j) centred at
    ((i - m/2 + 1/2) p, (j - m/2 + 1/2) p) um with p = cfg['pitch_um'] (m = 800, p = 0.25 um for D200). Flux normalization
    of ff_opt (JONES_SCALE): |T|^2 = 1 is full transmission of the incident flux density. Cells with centre radius
    > cfg['radius_um'] are masked here, as in jones_objectives. Incident power = pi R^2.

Propagation: the vector angular spectrum of jones_objectives.optical_objective (plane API exp(+i w t), ASM +k). T * pupil is
zero padded to n = pad * m, A = fft2, H = exp(+2 pi i z s_z / lam) [s^2 < s_cut^2], E_z = -(s_x E_x + s_y E_y) / s_z,
H = s x E, S_z = Re(E_x H_y* - E_y H_x*) (a unit plane wave carries S_z = 1). The inverse transform is evaluated only at the
requested points, by the exact inverse DFT (separable matrices) instead of ifft2 and crop. On the FFT grid both agree to
round-off. pad = 3 (600 um period) and every
evaluated point is checked at build time to be free of periodic wrap-around of components up to s_cut.

E1 'e1_achromat' (D4 lens, f = 333.333 um, 9 wavelengths, x and y polarization):
    S(lam, pol) = S_z(0, 0, f) / S_z_ideal(0, 0, f; lam). Ideal = lossless aberration-free lens of the same aperture on the
    same grid, T = exp(-i k (sqrt(r^2 + f^2) - f)) in the pupil (full incident flux transmitted), so S = Strehl x
    transmission (incident-flux normalized). S_bar(lam) = mean over the two polarizations (unpolarized).
    J1 = [ mean_lam S_bar^q ]^(1/q), q = -8: soft-min of log S_bar with temperature 1/8. min S_bar <= J1 <= 9^(1/8) min S_bar
    = 1.32 min S_bar. Gradient weight ~ S_bar^(q-1): a wavelength 10 % (20 %) below the others gets 2.6x (7.5x) their
    weight, scale free, so the same q holds from the grey start (S ~ 1e-4) to the end.
    Metrics: S, Strehl S / transmission, transmission, focusing efficiency = power in a disk of radius
    1.5 x 0.61 lam / NA (NA = sin atan(R / f) = 0.2873) / incident power (ideal lens: Airy 0.893), focal-plane PSF,
    on-axis S_z(z) for f +- 100 um and its peak position.

E2 'e2_rgb_holo' (150 x 150 um SQUARE aperture, aperture='square', radius_um = 75 half-width, image plane z = 300 um
    (image-side NA sin atan(75/300) = 0.2425 as before), pupil 600 x 600 cells of 0.25 um; no symmetry, x polarization, 450 -> B, 540 -> G, 635 -> R channel of targets/e2_parrot_rgb.npz, a procedural
    macaw-style parrot on black (default); e2_flower_rgb.npz and e2_starry_night_rgb.npz are kept as alternatives):
    z = 400 um, window 120 x 120 um centred on axis, 96 x 96 pixels of 1.25 um, 4 x 4 midpoint samples per pixel.
    Image-side NA = sin atan(100 / 400) = 0.2425, coherent resolution lam / (2 NA) = 0.93 / 1.11 / 1.31 um at 450 / 540 /
    635 nm: 1.25 um pixels are resolved at 450 and 540 nm and within 5 % of the 635 nm limit. Window edge rays reach
    sin = 0.36, far inside s_cut = 0.65 and the 0.25 um pupil Nyquist.
    P_c(j) = (integral of S_z over pixel j) / (pi R^2), eta_c = sum_j P_c(j) (efficiency into the window).
    Shat(P, t) = [cos(P+, t) - cos(1, t)] / [1 - cos(1, t)], cos(a, b) = <a, b> / (|a| |b|), P+ = max(P, 0):
    1 only for P proportional to t, 0 for a uniform window, lowered by any uniform background (unlike Pearson
    correlation), never above 1 (no gain from contrast stretching).
    J_c = eta_c [Shat(P_c, t_c) - w_x sum_{c' != c} max(0, Shat(P_c, t_c') - Shat(t_c, t_c'))], w_x = 0.5: a wavelength is
    penalized only for resembling another channel more than its own target does (the painting's channels are correlated).
    J2 = -tau log[ mean_c exp(-J_c / tau) ], tau = 0.02 (J2 within tau log 3 = 0.022 of min_c J_c).
    Metrics: eta, Shat matrix, Pearson, PSNR and SSIM (images scaled to the target sum), cross-talk (unmixing) matrix,
    white-balanced composite.

E3 'e3_volume_holo' (no symmetry, 532 nm, x polarization, 71 points of a DNA double helix, targets/e3_dna_double_helix_points.npz,
    axis along x, radius 25 um about z = 400 um, 7 planes z = 375-425 um):
    eta_p = S_z(x_p, y_p, z_p) / S_z_ideal_p, ideal = lossless lens focusing the full aperture onto point p (same grid).
    Background samples: on each target plane, the bounding box of that plane's points +- 10 um at 0.5 um, minus disks of
    radius 1.5 r_A(z) around every target point whose focal volume reaches the plane (|z_p - z| < 1.2 x 2 lam / NA^2, which
    includes points of the planes 3.35 um away), r_A = 0.61 lam / NA(z), NA(z) = sin atan(R / z); iota = S_z / S_z_ideal(0, 0, z).
    J3 = N M_{-8}(eta) + w_e sum_p eta_p - w_b N M_{+8}(iota_bg), M_q = power mean, w_e = 0.5, w_b = 1.
    N M_{-8}(eta) is efficiency x uniformity (= sum eta when uniform). M_{+8} over ~1e5 background samples is a smooth
    soft-max: it targets ghost spots, the smooth defocused cones of the other points stay near its floor.
    Metrics: eta_p, point efficiency (power in 1.5 r_A disks), uniformity, peak-to-background, plane images.
    Curve mode (target NPZ with mode = 'curve', e.g. targets/e3_dna_curve.npz): dense samples along the strands and rungs
    (0.75 um arc spacing), each scored on its nearest evaluation plane (planes every 5 um, finer than the axial first zero),
    eta normalized by the on-axis ideal peak of that plane, tube = 1 Airy radius around the in-slab samples:
    J3c = A M_{-8}(eta) + w_e E_tube - w_b A M_{+8}(iota_bg), A = curve length / (2 r_A) (number of resolvable spots, so
    A eta plays the role of N eta), E_tube = mean over planes of the power inside the tube / incident power.

E3 'e3_sliced_object' (production E3, So et al. Adv. Mater. 2023 style): default target targets/e3_sliced_b737true_K18_z72-162.npz
    a Boeing 737-800-like airplane in TRUE proportions (2.4 um per m on all axes), nose at z = 70 um, 18 cross sections
    at their real depths z = 72.9-162.2 um (spacing >= 1.37 axial DOF), 100 x 100 px of 1.0 um; the NPZ's recommended_cfg sets s_cut 0.95
    and sub 6 (NA up to 0.81) unless the caller overrides them. Other variants: e3_sliced_b737true_K8/K9/K12_*.npz, e3_sliced_airplane_*.npz.
    Per plane: J_k = (A_mean / A_k) e_k (1 + rho_kk) / 2 exp(-w_x sum_j max(0, rho_kj - rho0_kj)), rho = Pearson, e_k = power
    inside the slice support, w_x = 1, J = M_{-8}(J_k). pad 4 and s_cut 0.5 (the window
    needs sin < 0.44 at z = 300 um) keep planes out to 1000 um free of wrap-around.
"""
import json, math, hashlib
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TARGETS = HERE / "targets"
NAMES = ("e1_achromat", "e2_rgb_holo", "e3_volume_holo", "e3_sliced_object", "e4_dot_projector", "e3_polcam", "e3_dot_projector", "e3_pol_holo")
_GEOM = dict(radius_um=100.0, pitch_um=0.25, s_cut=0.65, pad=3)          # D200 pupil; s_cut as in jones_objectives
DEFAULTS = {
    "e1_achromat": dict(_GEOM, lam_nm=[420, 450, 470, 510, 540, 570, 600, 635, 670], pols=["x", "y"], focal_um=200.0 / 0.6,
                        softmin_q=-8.0, disk_airy=1.5, psf_half_um=6.0, psf_step_um=0.05, axial_half_um=100.0, axial_step_um=2.5),
    "e2_rgb_holo": dict(_GEOM, lam_nm=[450, 540, 635], pols=["x"], aperture="square", radius_um=75.0, z_um=300.0, window_um=120.0, sub=4, w_x=0.5, tau=0.02,
                        target="e2_parrot_rgb.npz"),                 # alternatives: e2_flower_rgb.npz, e2_starry_night_rgb.npz
    "e3_volume_holo": dict(_GEOM, lam_nm=[532], pols=["x"], softmin_q=-8.0, softmax_q=8.0, w_e=0.5, w_b=1.0,
                           bg_margin_um=10.0, bg_step_um=0.5, excl_airy=1.5, disk_airy=1.5, disk_step_um=0.1,
                           image_step_um=0.25, target="e3_dna_double_helix_points.npz"),
    "e3_pol_holo": dict(_GEOM, lam_nm=[540], pols=["x", "y"], aperture="square", radius_um=50.0, z_um=200.0, window_um=80.0, sub=4,
                        w_x=0.5, tau=0.02, target="e3_cnu_phy_sq100.npz"),         # E3: x -> CNU, y -> PHY, 540 nm, 100 um square, z 200 um
    "e3_dot_projector": dict(_GEOM, radius_um=57.447, lam_nm=[540], pols=["x"], pad=2, s_cut=1.0, window_airy=1.22, w_z=5.0, softmin_q=-8.0,
                             target="e3_dots_N5000_fov120_540nm.npz"),   # E3: dot projector, scaled units, physical 940 nm / 200 um = x 1.7407
    "e3_polcam": dict(_GEOM, radius_um=57.45, lam_nm=[540], pols=["x", "y"], pad=3, disk_airy=1.5, disk_step_um=0.1,
                      target="e3_polcam_target.npz"),                            # scaled units: physical 940 nm = x 1.7407
    "e4_dot_projector": dict(_GEOM, lam_nm=[940], pols=["x"], pad=2, s_cut=1.0, window_airy=1.22, w_z=5.0, softmin_q=-8.0,
                             target="e4_dots_N5000_fov120_940nm.npz"),   # Si at 940 nm
    "e3_sliced_object": dict(_GEOM, lam_nm=[532], pols=["x"], pad=4, s_cut=0.5, sub=5, w_x=1.0, softmin_q=-8.0,
                             target="e3_sliced_b737true_K18_z72-162.npz"),   # true-proportion 737, 18 stations; recommended_cfg sets s_cut 0.95, sub 6
}
_POL = {"x": ("xx", "yx"), "y": ("xy", "yy")}


def default_cfg(name):
    return json.loads(json.dumps(DEFAULTS[name]))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_target(fname):
    """Canonical target NPZ from targets/, verified against targets/manifest.json."""
    path = TARGETS / fname
    rec = json.loads((TARGETS / "manifest.json").read_text(encoding="utf-8"))["files"][fname]
    sha = _sha256(path)
    if sha != rec["sha256"]:
        raise RuntimeError(f"{path} sha256 {sha} does not match targets/manifest.json {rec['sha256']}")
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}, sha


def power_mean(x, q):
    """(mean x^q)^(1/q) in the log domain; x > 0 (clamped at 1e-30)."""
    lx = x.clamp_min(1e-30).log() * q
    return torch.exp((torch.logsumexp(lx, 0) - math.log(x.numel())) / q)


def airy_radius_um(lam_um, na):
    return 0.61 * lam_um / na


class VectorASM:
    """Grid and vector angular spectrum of jones_objectives.optical_objective, with outputs at chosen points."""

    def __init__(self, m, cfg, device):
        p = float(cfg["pitch_um"]); n = int(cfg["pad"]) * m
        assert m % 2 == 0 and (n - m) % 2 == 0, (m, n)
        self.m, self.n, self.p, self.o, self.dev = m, n, p, (n - m) // 2, device
        self.radius, self.s_cut = float(cfg["radius_um"]), float(cfg["s_cut"])
        c = (torch.arange(m, device=device, dtype=torch.float64) - m / 2 + 0.5) * p
        self.coord = c
        self.square = cfg.get("aperture", "circle") == "square"                 # square: |x|, |y| <= radius_um (half-width)
        self.pupil = ((c.abs()[:, None] <= self.radius) & (c.abs()[None, :] <= self.radius) if self.square
                      else (c[:, None] ** 2 + c[None, :] ** 2 <= self.radius ** 2)).to(torch.float64)
        self.f = torch.fft.fftfreq(n, d=p, device=device, dtype=torch.float64)
        self.x0 = (-n / 2 + 0.5) * p                                          # coordinate of padded index 0
        self.area = (2 * self.radius) ** 2 if self.square else math.pi * self.radius ** 2

    def check_alias(self, z_um, half_um):
        """Components with s < s_cut travel at most z s / s_z sideways; their periodic copies must miss |x|, |y| <= half_um."""
        travel = z_um * self.s_cut / math.sqrt(1 - self.s_cut ** 2)
        if travel >= self.n * self.p - self.radius - half_um:
            raise ValueError(f"wrap-around: z {z_um:.1f} um, half-width {half_um:.1f} um needs pad > {self.n // self.m}")

    def spectrum(self, e):
        buf = torch.zeros((self.n, self.n), dtype=torch.complex128, device=self.dev)
        buf[self.o:self.o + self.m, self.o:self.o + self.m] = e.to(torch.complex128) * self.pupil
        return torch.fft.fft2(buf)

    def _s(self, lam_um):
        sx = (lam_um * self.f)[:, None]; sy = (lam_um * self.f)[None, :]; s2 = sx * sx + sy * sy
        return sx, sy, torch.sqrt((1 - s2).clamp_min(1e-9)), s2 < self.s_cut ** 2

    def plane(self, ax, ay, lam_um, z_um):
        """Spectra of (Ex, Ey, Hx, Hy) at distance z (same formulas as jones_objectives)."""
        sx, sy, sz, keep = self._s(lam_um)
        h = torch.exp(2j * math.pi * z_um / lam_um * sz) * keep
        bx = ax * h; by = ay * h; bz = -(sx * bx + sy * by) / sz
        return bx, by, sy * bz - sz * by, sz * bx - sx * bz

    def rows(self, xs):
        xs = torch.as_tensor(np.asarray(xs, dtype=np.float64), device=self.dev).reshape(-1)
        return torch.exp(2j * math.pi * self.f[None, :] * (xs[:, None] - self.x0)) / self.n

    def grid(self, spec, rx, ry):
        """S_z on the grid rows(xs) x rows(ys)."""
        ex, ey, hx, hy = [rx @ b @ ry.T for b in spec]
        return (ex * hy.conj() - ey * hx.conj()).real

    def points(self, spec, rx, ry):
        """S_z at the points (xs[i], ys[i])."""
        ex, ey, hx, hy = [((rx @ b) * ry).sum(1) for b in spec]
        return (ex * hy.conj() - ey * hx.conj()).real

    def power(self, ax, ay, lam_um):
        """Propagating power through any plane z > 0 / incident power (Parseval)."""
        sx, sy, sz, keep = self._s(lam_um)
        az = -(sx * ax + sy * ay) / sz
        return ((ax.abs() ** 2 + ay.abs() ** 2 + az.abs() ** 2) * sz * keep).sum() * self.p ** 2 / self.n ** 2 / self.area

    def converging(self, lam_um, x_um, y_um, z_um):
        """Ideal lossless lens onto (x, y, z): unit amplitude, phase -k (distance - z) (converging for +k propagation)."""
        k = 2 * math.pi / lam_um; c = self.coord
        d = torch.sqrt((c[:, None] - x_um) ** 2 + (c[None, :] - y_um) ** 2 + z_um ** 2)
        return torch.exp(-1j * k * (d - z_um))


class FAObjective:
    def __init__(self, name, cfg, user_keys=()):
        self.name, self.cfg = name, cfg
        self.target, self.target_sha256 = (None, None)
        if isinstance(cfg.get("target"), str):
            self.target, self.target_sha256 = load_target(cfg["target"])
            if "recommended_cfg" in self.target:                               # target-specific settings (e.g. s_cut, sub) unless set by the caller
                for k, v in json.loads(str(self.target["recommended_cfg"])).items():
                    if k not in user_keys: cfg[k] = v
        elif isinstance(cfg.get("target"), dict):                            # tests: arrays passed directly
            self.target = {k: np.asarray(v) for k, v in cfg["target"].items()}
        self._key = None

    @property
    def info(self):
        cfg = {k: v for k, v in self.cfg.items() if not isinstance(v, dict)}
        return dict(name=self.name, cfg=cfg, target_sha256=self.target_sha256, module_sha256=_sha256(__file__))

    def _ready(self, T, lam_nm_T):
        sample = T[_POL[self.cfg["pols"][0]][0]]
        m = int(sample.shape[-1]); key = (str(sample.device), m)
        if self._key != key:
            self.asm = VectorASM(m, self.cfg, sample.device); self._build(); self._key = key
        lam_T = self.cfg["lam_nm"] if lam_nm_T is None else list(lam_nm_T)
        if sample.shape[0] != len(lam_T):
            raise ValueError(f"T has {sample.shape[0]} wavelengths, expected {len(lam_T)} ({lam_T})")
        idx = []
        for lam in self.cfg["lam_nm"]:
            hit = [i for i, l in enumerate(lam_T) if abs(float(l) - float(lam)) < 1e-6]
            if not hit: raise ValueError(f"wavelength {lam} nm not in T ({lam_T})")
            idx.append(hit[0])
        return idx

    def _spectra(self, T, i, pol):
        kx, ky = _POL[pol]
        return self.asm.spectrum(T[kx][i]), self.asm.spectrum(T[ky][i])

    def __call__(self, T, lam_nm_T=None):
        return self._value(T, self._ready(T, lam_nm_T))

    def metrics(self, T, lam_nm_T=None):
        idx = self._ready(T, lam_nm_T)
        with torch.no_grad():
            T = {k: v.detach() for k, v in T.items()}
            return self._metrics(T, idx)


class AchromatE1(FAObjective):
    def _build(self):
        a, c = self.asm, self.cfg; f = float(c["focal_um"])
        self.na = math.sin(math.atan(a.radius / f))
        a.check_alias(f, c["psf_half_um"]); a.check_alias(f + c["axial_half_um"], 0.0)
        self.r0 = a.rows([0.0])
        ideal = torch.zeros((len(c["lam_nm"]), len(c["pols"])), dtype=torch.float64, device=a.dev)
        with torch.no_grad():
            for li, lam in enumerate(c["lam_nm"]):
                t = a.converging(lam * 1e-3, 0.0, 0.0, f); z = torch.zeros_like(t); A = a.spectrum(t); Z = a.spectrum(z)
                for pi, pol in enumerate(c["pols"]):
                    spec = a.plane(A, Z, lam * 1e-3, f) if pol == "x" else a.plane(Z, A, lam * 1e-3, f)
                    ideal[li, pi] = a.points(spec, self.r0, self.r0)[0]
        self.ideal = ideal

    def _strehl(self, T, idx):
        c = self.cfg; S = []
        for li, lam in enumerate(c["lam_nm"]):
            for pi, pol in enumerate(c["pols"]):
                ax, ay = self._spectra(T, idx[li], pol)
                spec = self.asm.plane(ax, ay, lam * 1e-3, float(c["focal_um"]))
                S.append(self.asm.points(spec, self.r0, self.r0)[0] / self.ideal[li, pi])
        return torch.stack(S).reshape(len(c["lam_nm"]), len(c["pols"]))

    def _value(self, T, idx):
        return power_mean(self._strehl(T, idx).mean(1), self.cfg["softmin_q"])

    def _metrics(self, T, idx):
        a, c = self.asm, self.cfg; f = float(c["focal_um"]); nl, npol = len(c["lam_nm"]), len(c["pols"])
        S = self._strehl(T, idx)
        hw, st = c["psf_half_um"], c["psf_step_um"]; xs = (np.arange(int(round(2 * hw / st))) + 0.5) * st - hw
        rw = a.rows(xs); X = torch.as_tensor(xs, device=a.dev)
        zs = f + (np.arange(int(round(2 * c["axial_half_um"] / c["axial_step_um"])) + 1) * c["axial_step_um"] - c["axial_half_um"])
        trans = torch.zeros(nl, npol, dtype=torch.float64, device="cpu"); eff = torch.zeros(nl, npol, dtype=torch.float64, device="cpu")
        psf = torch.zeros(nl, npol, len(xs), len(xs), dtype=torch.float64, device="cpu"); axial = torch.zeros(nl, npol, len(zs), dtype=torch.float64, device="cpu")
        r_disk = np.array([c["disk_airy"] * airy_radius_um(l * 1e-3, self.na) for l in c["lam_nm"]])
        for li, lam in enumerate(c["lam_nm"]):
            disk = (X[:, None] ** 2 + X[None, :] ** 2 <= r_disk[li] ** 2).to(torch.float64)
            for pi, pol in enumerate(c["pols"]):
                ax, ay = self._spectra(T, idx[li], pol)
                trans[li, pi] = a.power(ax, ay, lam * 1e-3).cpu()
                sz = a.grid(a.plane(ax, ay, lam * 1e-3, f), rw, rw)
                psf[li, pi] = (sz / self.ideal[li, pi]).cpu(); eff[li, pi] = ((sz * disk).sum() * st ** 2 / a.area).cpu()
                for zi, z in enumerate(zs):
                    axial[li, pi, zi] = (a.points(a.plane(ax, ay, lam * 1e-3, float(z)), self.r0, self.r0)[0] / self.ideal[li, pi]).cpu()
        S = S.cpu(); J = power_mean(S.mean(1), c["softmin_q"])
        scal = dict(objective=float(J), lam_nm=c["lam_nm"], pols=c["pols"], S=S.tolist(), S_unpolarized=S.mean(1).tolist(),
                    transmission=trans.tolist(), strehl=(S / trans).tolist(), focusing_efficiency=eff.tolist(),
                    disk_radius_um=r_disk.tolist(), na_geometric=self.na, focal_um=f,
                    axial_peak_um=[[float(zs[int(axial[li, pi].argmax())]) for pi in range(npol)] for li in range(nl)])
        arrs = dict(S=S.numpy(), transmission=trans.numpy(), focusing_efficiency=eff.numpy(), psf_norm=psf.numpy().astype(np.float32),
                    psf_x_um=xs, axial_norm=axial.numpy(), axial_z_um=zs, ideal_peak=self.ideal.cpu().numpy(),
                    lam_nm=np.array(c["lam_nm"], dtype=np.float64), disk_radius_um=r_disk)
        return dict(scalars=scal, arrays=arrs)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-300)


def shat(P, t, c0):
    cos = (P * t).sum() / (P.norm() * t.norm() + 1e-300)
    return (cos - c0) / (1 - c0)


class RGBHoloE2(FAObjective):
    def _build(self):
        a, c = self.asm, self.cfg
        tg = np.asarray(self.target["targets"], dtype=np.float64)          # (3, n_px, n_px), x-y indexed, cfg['lam_nm'] order
        assert tg.shape[0] == len(c["lam_nm"]) and tg.shape[1] == tg.shape[2], tg.shape
        self.npx = tg.shape[1]; W = float(c["window_um"]); self.ds = W / (self.npx * c["sub"])
        a.check_alias(float(c["z_um"]), W / 2)
        xs = (np.arange(self.npx * c["sub"]) + 0.5) * self.ds - W / 2
        self.rw = a.rows(xs)
        self.tsum = tg.sum((1, 2))                                            # linear channel sums: white balance of the composite
        t = torch.as_tensor(tg / tg.sum((1, 2), keepdims=True), device=a.dev)
        self.t = t
        self.c0 = [float(tc.sum() / (math.sqrt(tc.numel()) * tc.norm())) for tc in t]
        self.S0 = torch.tensor([[float(shat(t[i], t[j], self.c0[j])) for j in range(3)] for i in range(3)], dtype=torch.float64, device=a.dev)

    def _images(self, T, idx):
        a, c = self.asm, self.cfg; P = []
        for li, lam in enumerate(c["lam_nm"]):
            ax, ay = self._spectra(T, idx[li], "x")
            sz = a.grid(a.plane(ax, ay, lam * 1e-3, float(c["z_um"])), self.rw, self.rw)
            P.append(sz.reshape(self.npx, c["sub"], self.npx, c["sub"]).sum((1, 3)) * self.ds ** 2 / a.area)
        return torch.stack(P)

    def _terms(self, P):
        n = len(P); Pp = P.clamp_min(0)
        Sm = torch.stack([torch.stack([shat(Pp[i], self.t[j], self.c0[j]) for j in range(n)]) for i in range(n)])
        excess = (Sm - self.S0).clamp_min(0) * (1 - torch.eye(n, dtype=torch.float64, device=P.device))
        eta = P.sum((1, 2))
        Jc = eta * (Sm.diagonal() - self.cfg["w_x"] * excess.sum(1))
        tau = self.cfg["tau"]
        return -tau * (torch.logsumexp(-Jc / tau, 0) - math.log(n)), Jc, eta, Sm, excess

    def _value(self, T, idx):
        return self._terms(self._images(T, idx))[0]

    def _metrics(self, T, idx):
        P = self._images(T, idx); J, Jc, eta, Sm, excess = self._terms(P)
        P, t = P.cpu().numpy(), self.t.cpu().numpy(); n = len(P)
        Ps = P / P.sum((1, 2), keepdims=True) * t.sum((1, 2), keepdims=True)          # scaled to the target sum
        pear = [float(np.corrcoef(Ps[i].ravel(), t[i].ravel())[0, 1]) for i in range(n)]
        psnr = [float(10 * np.log10(t[i].max() ** 2 / np.mean((Ps[i] - t[i]) ** 2))) for i in range(n)]
        ssim = [ssim_gauss(Ps[i], t[i], data_range=float(t[i].max() - t[i].min())) for i in range(n)]
        X = np.concatenate([np.ones((t[0].size, 1)), t.reshape(n, -1).T], 1)            # P_c ~ a + sum_c' M_cc' t_c'
        M = np.linalg.lstsq(X, P.reshape(n, -1).T, rcond=None)[0][1:].T
        slot = np.asarray(self.target.get("rgb_slot", [2, 1, 0]))                     # R, G, B slot of each wavelength
        comp = np.zeros(P.shape[1:] + (3,))                                           # white balance: linear channel sums of the target
        for i in range(n): comp[..., int(slot[i])] = P[i] / P[i].sum() * self.tsum[i]
        scal = dict(objective=float(J), lam_nm=self.cfg["lam_nm"], J_channel=Jc.tolist(), efficiency=eta.tolist(),
                    similarity=Sm.tolist(), similarity_targets=self.S0.tolist(), crosstalk_excess=excess.tolist(),
                    pearson=pear, psnr_db=psnr, ssim=ssim, unmixing=M.tolist(),
                    unmixing_rownorm=(M / np.diag(M)[:, None]).tolist())
        arrs = dict(P=P.astype(np.float32), target=t.astype(np.float32), lam_nm=np.array(self.cfg["lam_nm"], dtype=np.float64),
                    composite_linear_xy=comp.astype(np.float32))
        return dict(scalars=scal, arrays=arrs)


def ssim_gauss(x, y, data_range, sigma=1.5, K1=0.01, K2=0.03):
    """SSIM (Wang et al. 2004): 11 x 11 Gaussian window, sigma 1.5, valid region only."""
    r = np.arange(-5, 6); g = np.exp(-r ** 2 / (2 * sigma ** 2)); g /= g.sum()
    def filt(a):
        a = np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 0, a)
        return np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 1, a)
    mx, my = filt(x), filt(y); sxx = filt(x * x) - mx ** 2; syy = filt(y * y) - my ** 2; sxy = filt(x * y) - mx * my
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    return float(np.mean((2 * mx * my + C1) * (2 * sxy + C2) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))))


class VolumeHoloE3(FAObjective):
    def _build(self):
        a, c = self.asm, self.cfg; lam = float(np.asarray(self.target["lam_nm"]).ravel()[0])
        assert len(c["lam_nm"]) == 1 and abs(lam - c["lam_nm"][0]) < 1e-6, (lam, c["lam_nm"])
        lu = lam * 1e-3; pts = np.asarray(self.target["points_um"], dtype=np.float64); self.pts = pts; self.N = len(pts)
        self.curve = str(self.target.get("mode", "points")) == "curve"          # curve: dense samples, each scored on its nearest plane
        zp = np.round(np.asarray(self.target["plane_um"], dtype=np.float64) if self.curve else pts[:, 2], 6); self.planes = np.unique(zp)
        self.on = [np.flatnonzero(zp == z) for z in self.planes]
        self.on_t = [torch.as_tensor(o, device=a.dev) for o in self.on]
        self.na = np.sin(np.arctan(a.radius / self.planes)); self.rA = airy_radius_um(lu, self.na)
        self.prow, self.bgrow, self.bgmask, self.bgxy, self.tube = [], [], [], [], []
        for j, z in enumerate(self.planes):
            q = pts[self.on[j]]
            self.prow.append((a.rows(q[:, 0]), a.rows(q[:, 1])))
            xs, ys = self._box(q, c["bg_margin_um"], c["bg_step_um"])
            a.check_alias(float(z), max(np.abs(xs).max(), np.abs(ys).max()))
            self.bgmask.append(torch.as_tensor(self._bg(xs, ys, z), device=a.dev))
            self.bgrow.append((a.rows(xs), a.rows(ys))); self.bgxy.append((xs, ys))
            if self.curve:                                                    # tube cross-section: 1 Airy radius around the in-slab samples
                d2 = (xs[:, None, None] - q[None, None, :, 0]) ** 2 + (ys[None, :, None] - q[None, None, :, 1]) ** 2
                self.tube.append(torch.as_tensor((d2 <= self.rA[j] ** 2).any(-1), device=a.dev))
        if self.curve: self.A = float(self.target["curve_weight"])             # curve length / (2 Airy radii): resolvable spots
        ip = torch.zeros(self.N, dtype=torch.float64, device=a.dev); iz = torch.zeros(len(self.planes), dtype=torch.float64, device=a.dev)
        with torch.no_grad():
            Z = a.spectrum(torch.zeros((a.m, a.m), dtype=torch.complex128, device=a.dev))
            for j, z in enumerate(self.planes):
                r0 = a.rows([0.0])
                iz[j] = a.points(a.plane(a.spectrum(a.converging(lu, 0.0, 0.0, float(z))), Z, lu, float(z)), r0, r0)[0]
                for i in ([] if self.curve else self.on[j]):
                    rx, ry = a.rows([pts[i, 0]]), a.rows([pts[i, 1]])
                    ip[i] = a.points(a.plane(a.spectrum(a.converging(lu, pts[i, 0], pts[i, 1], float(z))), Z, lu, float(z)), rx, ry)[0]
        if self.curve: ip = iz[torch.as_tensor(np.searchsorted(self.planes, zp), device=a.dev)]
        self.ideal_p, self.ideal_axis = ip, iz

    def _bg(self, xs, ys, z):
        """Background samples of plane z: outside the disks (radius excl_airy Airy radii) around every target point whose focal
        volume reaches z (|z_p - z| < 1.2 axial first zeros 2 lam / NA^2), so a nearby plane's in-focus spot is not background."""
        lu = self.cfg["lam_nm"][0] * 1e-3; na = np.sin(np.arctan(self.asm.radius / self.pts[:, 2]))
        q = self.pts[np.abs(self.pts[:, 2] - z) < 1.2 * 2 * lu / na ** 2]
        r = self.cfg["excl_airy"] * 0.61 * lu / np.sin(np.arctan(self.asm.radius / z))
        d2 = (xs[:, None, None] - q[None, None, :, 0]) ** 2 + (ys[None, :, None] - q[None, None, :, 1]) ** 2
        return (d2 > r ** 2).all(-1)

    @staticmethod
    def _box(q, margin, step):
        out = []
        for k in (0, 1):
            lo, hi = q[:, k].min() - margin, q[:, k].max() + margin; K = int(math.ceil((hi - lo) / step))
            out.append((lo + hi) / 2 + (np.arange(K) - (K - 1) / 2) * step)
        return out

    def _fields(self, T, idx):
        a, lu = self.asm, self.cfg["lam_nm"][0] * 1e-3
        ax, ay = self._spectra(T, idx[0], "x")
        eta = torch.zeros(self.N, dtype=torch.float64, device=a.dev); bg = []; tube = []
        for j, z in enumerate(self.planes):
            spec = a.plane(ax, ay, lu, float(z))
            eta = eta.index_copy(0, self.on_t[j], a.points(spec, *self.prow[j]) / self.ideal_p[self.on_t[j]])
            g = a.grid(spec, *self.bgrow[j]); bg.append((g / self.ideal_axis[j])[self.bgmask[j]])
            if self.curve: tube.append(g[self.tube[j]].sum() * self.cfg["bg_step_um"] ** 2 / a.area)
        self._tube = torch.stack(tube) if tube else None
        return eta, bg, ax, ay

    def _value(self, T, idx):
        c = self.cfg; eta, bg, _, _ = self._fields(T, idx)
        if self.curve:                                                        # J3c = A M_-8(eta) + w_e E_tube - w_b A M_+8(iota_bg)
            return self.A * power_mean(eta, c["softmin_q"]) + c["w_e"] * self._tube.mean() - c["w_b"] * self.A * power_mean(torch.cat(bg), c["softmax_q"])
        return self.N * power_mean(eta, c["softmin_q"]) + c["w_e"] * eta.sum() - c["w_b"] * self.N * power_mean(torch.cat(bg), c["softmax_q"])

    def _metrics(self, T, idx):
        a, c, lu = self.asm, self.cfg, self.cfg["lam_nm"][0] * 1e-3
        eta, bg, ax, ay = self._fields(T, idx); J = self._value(T, idx)
        eff = np.zeros(self.N); images = {}; per_plane = []
        for j, z in enumerate(self.planes):
            spec = a.plane(ax, ay, lu, float(z)); rd = c["disk_airy"] * self.rA[j]; st = c["disk_step_um"]
            u = (np.arange(int(math.ceil(2 * rd / st))) + 0.5) * st - int(math.ceil(2 * rd / st)) * st / 2
            for i in ([] if self.curve else self.on[j]):
                sz = a.grid(spec, a.rows(self.pts[i, 0] + u), a.rows(self.pts[i, 1] + u)).cpu().numpy()
                eff[i] = float((sz * ((u[:, None] ** 2 + u[None, :] ** 2) <= rd ** 2)).sum() * st ** 2 / a.area)
            q = self.pts[self.on[j]]; xs, ys = self._box(q, c["bg_margin_um"], c["image_step_um"])
            img = (a.grid(spec, a.rows(xs), a.rows(ys)) / self.ideal_axis[j]).cpu().numpy()
            m_bg = self._bg(xs, ys, z)
            images[f"plane_{j:02d}"] = img.astype(np.float32); images[f"plane_{j:02d}_x_um"] = xs; images[f"plane_{j:02d}_y_um"] = ys
            e_j = eta[self.on_t[j]].cpu().numpy()
            per_plane.append(dict(z_um=float(z), n_points=len(q), eta_min=float(e_j.min()), eta_mean=float(e_j.mean()),
                                  bg_max=float(img[m_bg].max()), bg_mean=float(img[m_bg].mean()),
                                  pbr_min_over_max=float(e_j.min() / img[m_bg].max()), pbr_mean_over_mean=float(e_j.mean() / img[m_bg].mean())))
        e = eta.cpu().numpy(); bgall = torch.cat(bg).cpu().numpy(); tube = None if self._tube is None else self._tube.cpu().numpy()
        scal = dict(objective=float(J), N=self.N, eta_min=float(e.min()), eta_mean=float(e.mean()), eta_sum=float(e.sum()),
                    point_efficiency_sum=float(eff.sum()), uniformity_eff=float(1 - (eff.max() - eff.min()) / (eff.max() + eff.min())),
                    uniformity_eta=float(1 - (e.max() - e.min()) / (e.max() + e.min())), cv_eta=float(e.std() / e.mean()),
                    pbr_min_over_max=float(e.min() / bgall.max()), pbr_mean_over_mean=float(e.mean() / bgall.mean()),
                    softmin_term=float(self.N * power_mean(eta, c["softmin_q"])), efficiency_term=float(c["w_e"] * eta.sum()),
                    background_term=float(c["w_b"] * self.N * power_mean(torch.cat(bg), c["softmax_q"])),
                    transmission=float(a.power(ax, ay, lu)), per_plane=per_plane, mode="curve" if self.curve else "points",
                    eta_min_over_mean=float(e.min() / e.mean()), eta_p05_over_mean=float(np.percentile(e, 5) / e.mean()))
        if self.curve:
            scal.update(tube_efficiency_per_plane=tube.tolist(), tube_efficiency_mean=float(tube.mean()), A=self.A,
                        softmin_term=float(self.A * power_mean(eta, c["softmin_q"])), efficiency_term=float(c["w_e"] * tube.mean()),
                        background_term=float(c["w_b"] * self.A * power_mean(torch.cat(bg), c["softmax_q"])))
        arrs = dict(eta=e, point_efficiency=eff, points_um=self.pts, planes_um=self.planes, airy_radius_um=self.rA, **images)
        return dict(scalars=scal, arrays=arrs)


class SlicedObjectE3(FAObjective):
    """E3 sliced 3D object (So et al. style): K slab images t_k on DOF-graded planes z_k, one window per plane.
    P_k = pixel power / incident power, e_k = power inside the slice support / incident power, rho = Pearson correlation
    (offset tolerant: the defocused light of the other slices is an unavoidable background in every window),
    J_k = (A_mean / A_k) e_k (1 + rho(P_k, t_k)) / 2 exp(-w_x sum_{j != k} max(0, rho(P_k, t_j) - rho(t_k, t_j))) >= 0,
    J = M_{-8}(J_k) (power-mean soft-min as in E1). Non-negative per-plane scores, so dimming a plane never raises J
    (the first version, e_k x baseline-corrected cosine, did reward dimming while the similarity was negative)."""
    def _build(self):
        a, c = self.asm, self.cfg; tg = self.target
        assert abs(float(np.asarray(tg["lam_nm"]).ravel()[0]) - c["lam_nm"][0]) < 1e-6
        sl = np.asarray(tg["slices"], dtype=np.float64); self.K, self.npx = sl.shape[0], sl.shape[1]
        self.planes = np.asarray(tg["planes_um"], dtype=np.float64); W = float(tg["window_um"]); self.ds = W / (self.npx * c["sub"])
        for z in self.planes: a.check_alias(float(z), W / 2)
        xs = (np.arange(self.npx * c["sub"]) + 0.5) * self.ds - W / 2; self.rw = a.rows(xs)
        self.t = torch.as_tensor(sl / sl.sum((1, 2), keepdims=True), device=a.dev)
        self.mask = torch.as_tensor(np.asarray(tg["masks"], dtype=bool), device=a.dev)
        area = self.mask.sum((1, 2)).double(); self.wt = area.mean() / area
        self.S0 = torch.stack([torch.stack([pearson(self.t[i], self.t[j]) for j in range(self.K)]) for i in range(self.K)])

    def _images(self, T, idx):
        a, c = self.asm, self.cfg; lu = c["lam_nm"][0] * 1e-3
        ax, ay = self._spectra(T, idx[0], "x"); P = []
        for z in self.planes:
            sz = a.grid(a.plane(ax, ay, lu, float(z)), self.rw, self.rw)
            P.append(sz.reshape(self.npx, c["sub"], self.npx, c["sub"]).sum((1, 3)) * self.ds ** 2 / a.area)
        return torch.stack(P), ax, ay

    def _terms(self, P):
        K = self.K; Pp = P.clamp_min(0)
        Sm = torch.stack([torch.stack([pearson(Pp[i], self.t[j]) for j in range(K)]) for i in range(K)])
        excess = (Sm - self.S0).clamp_min(0) * (1 - torch.eye(K, dtype=torch.float64, device=P.device))
        sig = (P * self.mask).sum((1, 2)); Jk = self.wt * sig.clamp_min(0) * (1 + Sm.diagonal()) / 2 * torch.exp(-self.cfg["w_x"] * excess.sum(1))
        return power_mean(Jk, self.cfg["softmin_q"]), Jk, sig, Sm, excess

    def _value(self, T, idx):
        return self._terms(self._images(T, idx)[0])[0]

    def _metrics(self, T, idx):
        P, ax, ay = self._images(T, idx); J, Jk, sig, Sm, excess = self._terms(P)
        Pn, t, K = P.cpu().numpy(), self.t.cpu().numpy(), self.K
        Ps = Pn / Pn.sum((1, 2), keepdims=True)
        pear = [float(np.corrcoef(Ps[i].ravel(), t[i].ravel())[0, 1]) for i in range(K)]
        psnr = [float(10 * np.log10(t[i].max() ** 2 / np.mean((Ps[i] - t[i]) ** 2))) for i in range(K)]
        ssim = [ssim_gauss(Ps[i], t[i], data_range=float(t[i].max() - t[i].min())) for i in range(K)]
        win = Pn.sum((1, 2))
        scal = dict(objective=float(J), K=K, planes_um=self.planes.tolist(), J_plane=Jk.tolist(), signal_efficiency=sig.tolist(),
                    signal_efficiency_sum=float(sig.sum()), window_power=win.tolist(), signal_fraction_of_window=(sig.cpu().numpy() / win).tolist(),
                    pearson=pear, psnr_db=psnr, ssim=ssim, similarity=Sm.tolist(), crosstalk_excess=excess.tolist(),
                    transmission=float(self.asm.power(ax, ay, self.cfg["lam_nm"][0] * 1e-3)))
        return dict(scalars=scal, arrays=dict(P=Pn.astype(np.float32), target=t.astype(np.float32), planes_um=self.planes))


class DotProjectorE4(FAObjective):
    """E4 structured-light dot projector (940 nm, silicon, x-pol, far field). Far field = angular spectrum of the pupil field on the
    pad x zero-padded grid (bin lam / (pad D)); each bin carries the exact propagating power
    s_z (|Ax|^2 + |Ay|^2) + |sx Ax + sy Ay|^2 / s_z (vector ASM Parseval, same as VectorASM.power), after dividing the spectrum by
    cos(pi fx p / 2) cos(pi fy p / 2): the D200 pupil cells are means of 2 x 2 point samples at +-p/4 (QPC2), so this is the exact
    cell-average transfer (cell_average="box" uses sinc instead). far_model="scalar_thin" (scalar feasibility of thin pixelated masks):
    bin power |A|^2 sinc^2(fx p) sinc^2(fy p), energy conserving. P_d = power in the bins within window_airy x 0.61...
    precisely within r = window_airy lam / D of dot d, / incident power; Z = same at the zero order.
    J = E U - w_z Z, E = sum_d P_d, U = M_q(P_d) / mean(P_d) (q = -8)."""
    def _build(self):
        a, c = self.asm, self.cfg; tg = self.target
        assert abs(float(np.asarray(tg["lam_nm"]).ravel()[0]) - c["lam_nm"][0]) < 1e-6 and int(tg["pad"]) == int(c["pad"]) and int(tg["m"]) == a.m
        lu = c["lam_nm"][0] * 1e-3; n = a.n; binsz = lu / (n * a.p); cell = lu / (2 * a.radius); r = c["window_airy"] * cell
        k = int(np.ceil(r / binsz)); di, dj = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing="ij")
        ok = (di ** 2 + dj ** 2) * binsz ** 2 <= r ** 2; di, dj = di[ok], dj[ok]
        bi = np.asarray(tg["bin_index"], dtype=np.int64); self.N = len(bi)
        self.idx = torch.as_tensor(((bi[:, 0:1] + di[None]) % n) * n + (bi[:, 1:2] + dj[None]) % n, device=a.dev)
        self.idx0 = torch.as_tensor((di % n) * n + dj % n, device=a.dev)
        f = a.f; fm = c.get("far_model", "vector")
        if fm == "vector":                                                   # FDTD cells = mean of 2 x 2 point samples at +-p/4 (QPC2): 1 / cos(pi f p / 2) per axis
            sc = torch.cos(math.pi * f * a.p / 2) if c.get("cell_average", "qpc2") == "qpc2" else torch.sinc(f * a.p)
            self.boxcorr = 1.0 / (sc[:, None] * sc[None, :])
        else:                                                                # thin pixelated mask: piecewise-constant field, far field x sinc
            sc = torch.sinc(f * a.p); self.boxcorr = sc[:, None] * sc[None, :]
        self.win_bins = int(ok.sum())

    def _binpower(self, T, idx):
        a, lu = self.asm, self.cfg["lam_nm"][0] * 1e-3
        ax, ay = self._spectra(T, idx[0], "x"); ax = ax * self.boxcorr; ay = ay * self.boxcorr
        sx, sy, sz, keep = a._s(lu)
        if self.cfg.get("far_model", "vector") == "scalar_thin":             # energy-conserving scalar thin-element model (feasibility only)
            w = (ax.abs() ** 2 + ay.abs() ** 2) * keep
        else:
            w = (sz * (ax.abs() ** 2 + ay.abs() ** 2) + (sx * ax + sy * ay).abs() ** 2 / sz) * keep
        return w.reshape(-1) * a.p ** 2 / a.n ** 2 / a.area

    def _terms(self, W):
        P = W[self.idx].sum(1); Z = W[self.idx0].sum(); E = P.sum(); U = power_mean(P, self.cfg["softmin_q"]) / P.mean()
        return E * U - self.cfg["w_z"] * Z, P, E, U, Z

    def _value(self, T, idx):
        return self._terms(self._binpower(T, idx))[0]

    def _metrics(self, T, idx):
        W = self._binpower(T, idx); J, P, E, U, Z = self._terms(W); tot = float(W.sum()); Pn = P.cpu().numpy()
        n = self.asm.n; Wm = torch.fft.fftshift(W.reshape(n, n)).cpu().numpy().astype(np.float32)
        scal = dict(objective=float(J), N=self.N, efficiency=float(E), uniformity_softmin=float(U), uniformity_min_over_max=float(Pn.min() / Pn.max()),
                    sigma_over_mean=float(Pn.std() / Pn.mean()), zero_order=float(Z), zero_order_over_mean_dot=float(Z / P.mean()),
                    transmitted=tot, background_fraction=float((tot - float(E) - float(Z)) / tot), window_bins=self.win_bins,
                    efficiency_excl_zero=float(E), zero_order_fraction_incident=float(Z), zero_order_fraction_transmitted=float(Z) / tot,
                    sota_eff_ge_50=bool(float(E) >= 0.5), sota_sigma_le_20=bool(Pn.std() / Pn.mean() <= 0.2), sota_zero_le_1=bool(float(Z) <= 0.01))
        scal["sota_all"] = scal["sota_eff_ge_50"] and scal["sota_sigma_le_20"] and scal["sota_zero_le_1"]
        return dict(scalars=scal, arrays=dict(dot_power=Pn, farfield_power_shifted=Wm))


def stokes_rows(Q):
    """Hermitian 2 x 2 (x, y basis) -> Stokes analyzer row with P = a . S, S3 = 2 Im(Ex* Ey)."""
    return torch.stack([(Q[0, 0] + Q[1, 1]).real / 2, (Q[0, 0] - Q[1, 1]).real / 2, Q[0, 1].real, -Q[0, 1].imag])


class PolCamE3(FAObjective):
    """E3 full-Stokes polarization camera (Rubin et al. Science 2019 type), scaled units (physical = x target scale). A lens of focal
    length f sends any input polarization into 4 spots on a common focal plane. Spot k power for input Jones vector E is
    E^H Q_k E, Q_k = Hermitian part of G_ab = integral over the 1.5 Airy-radius disk of (E^b_x H^a_y* - E^b_y H^a_x*) for the
    x- and y-input exit fields (T['xx'], T['yx']) and (T['xy'], T['yy']). Rows a_k = stokes_rows(Q_k) form the instrument
    matrix A (P = A S). J = eff x sqrt(3) / cond(A), eff = sum_k A[k, 0] (power in the 4 spots averaged over input
    polarizations / incident); sqrt(3) is the condition number of the regular-tetrahedron optimum."""
    def _build(self):
        a, c = self.asm, self.cfg; tg = self.target; lu = c["lam_nm"][0] * 1e-3; f = float(tg["focal_um"])
        assert abs(float(np.asarray(tg["lam_nm"]).ravel()[0]) - c["lam_nm"][0]) < 1e-6 and abs(float(tg["radius_um"]) - a.radius) < 1e-6
        self.f = f; self.spots = np.asarray(tg["spots_um"], dtype=np.float64); self.na = math.sin(math.atan(a.radius / f))
        rd = c["disk_airy"] * airy_radius_um(lu, self.na); st = c["disk_step_um"]; K = int(math.ceil(2 * rd / st))
        u = (np.arange(K) + 0.5) * st - K * st / 2; self.st = st; self.rd = rd
        self.disk = torch.as_tensor((u[:, None] ** 2 + u[None, :] ** 2 <= rd ** 2).astype(np.float64), device=a.dev)
        a.check_alias(f, float(np.abs(self.spots).max() + rd))
        self.rows_k = [(a.rows(x + u), a.rows(y + u)) for x, y in self.spots]; self.r0 = [(a.rows([x]), a.rows([y])) for x, y in self.spots]
        with torch.no_grad():                                                          # ideal lossless lens onto spot k (x input): peak for Strehl
            Z = a.spectrum(torch.zeros((a.m, a.m), dtype=torch.complex128, device=a.dev))
            self.ideal = torch.stack([a.points(a.plane(a.spectrum(a.converging(lu, x, y, f)), Z, lu, f), *self.r0[k])[0] for k, (x, y) in enumerate(self.spots)])

    def _fields(self, T, idx):
        a, lu = self.asm, self.cfg["lam_nm"][0] * 1e-3
        return [a.plane(*self._spectra(T, idx[0], pol), lu, self.f) for pol in ("x", "y")]

    def _A(self, specs):
        a = self.asm; rows = []
        for k in range(4):
            rx, ry = self.rows_k[k]
            F = [[rx @ b @ ry.T for b in sp] for sp in specs]                         # F[input][Ex, Ey, Hx, Hy]
            G = torch.stack([torch.stack([((F[b][0] * F[aa][3].conj() - F[b][1] * F[aa][2].conj()) * self.disk).sum() for b in range(2)]) for aa in range(2)])
            Q = (G + G.conj().T) / 2 * self.st ** 2 / a.area
            rows.append(stokes_rows(Q))
        return torch.stack(rows)

    def _terms(self, A):
        sv = torch.linalg.svdvals(A); kappa = sv[0] / sv[-1]; eff = A[:, 0].sum()
        return eff * math.sqrt(3) / kappa, eff, kappa

    def _value(self, T, idx):
        return self._terms(self._A(self._fields(T, idx)))[0]

    def _metrics(self, T, idx):
        specs = self._fields(T, idx); A = self._A(specs); J, eff, kappa = self._terms(A); An = A.cpu().numpy()
        ewv = float(np.trace(np.linalg.inv(An.T @ An)))
        A0 = float(eff) / 4 * np.array(self.target["tetra_stokes"]); ewv0 = float(np.trace(np.linalg.inv(A0.T @ A0)))
        g = np.random.default_rng(5); v = g.normal(size=(2000, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True); S = np.concatenate([np.ones((2000, 1)), v], 1)
        P = S @ An.T; noise = g.normal(size=P.shape) * 0.01 * P.mean(); Sr = (P + noise) @ np.linalg.inv(An).T
        err = float(np.sqrt(np.mean(np.sum((Sr[:, 1:] / Sr[:, :1] - S[:, 1:]) ** 2, 1))))
        a = self.asm
        peaks = [[float(a.points(specs[i], *self.r0[k])[0] / (self.ideal[k] / 4)) for k in range(4)] for i in range(2)]
        sc = float(self.target["scale_to_physical"])
        scal = dict(objective=float(J), efficiency=float(eff), condition_number=float(kappa), frame_quality=float(math.sqrt(3) / kappa), A=An.tolist(),
                    ewv=ewv, ewv_over_tetra_same_eff=ewv / ewv0, stokes_rms_error_1pct_noise=err,
                    spot_peak_over_quarter_ideal=dict(x_input=peaks[0], y_input=peaks[1]), spot_power_x_input=(An[:, 0] + An[:, 1]).tolist(),
                    spot_power_y_input=(An[:, 0] - An[:, 1]).tolist(),
                    physical=dict(lam_nm=float(self.target["lam_phys_nm"]), focal_um=self.f * sc, aperture_um=2 * a.radius * sc,
                                  spots_um=(self.spots * sc).tolist(), disk_radius_um=self.rd * sc))
        return dict(scalars=scal, arrays=dict(A=An))


class PolHoloE3(RGBHoloE2):
    """E3 polarization-switched hologram: one wavelength (540 nm), default 100 x 100 um square aperture (pupil 400 x 400), z 200 um
    (image NA 0.243), 80 um window, 64 x 64 px of 1.25 um (the 150 um / z 300 / 96 px variant is e3_cnu_phy.npz). Channel 0 = x input (Ex, Ey) = (T_xx, T_yx) must show target 0 ("CNU"), channel 1 = y input (T_xy, T_yy)
    target 1 ("PHY"); image = total S_z (no analyzer). Same terms as E2: J_c = eta_c [Shat(P_c, t_c) - w_x max(0, Shat(P_c, t_c') -
    Shat(t_c, t_c'))], J = LSE soft-min (tau). Extinction = power of the input in the other word's letter pixels / power in its own."""
    CH = ("x", "y")

    def _build(self):
        a, c = self.asm, self.cfg
        tg = np.asarray(self.target["targets"], dtype=np.float64); assert tg.shape[0] == 2 and tg.shape[1] == tg.shape[2], tg.shape
        self.npx = tg.shape[1]; W = float(c["window_um"]); self.ds = W / (self.npx * c["sub"])
        a.check_alias(float(c["z_um"]), W / 2)
        xs = (np.arange(self.npx * c["sub"]) + 0.5) * self.ds - W / 2; self.rw = a.rows(xs)
        self.tsum = tg.sum((1, 2)); t = torch.as_tensor(tg / tg.sum((1, 2), keepdims=True), device=a.dev); self.t = t
        self.masks = torch.as_tensor(tg >= 0.5, device=a.dev)
        self.c0 = [float(tc.sum() / (math.sqrt(tc.numel()) * tc.norm())) for tc in t]
        self.S0 = torch.tensor([[float(shat(t[i], t[j], self.c0[j])) for j in range(2)] for i in range(2)], dtype=torch.float64, device=a.dev)

    def _images(self, T, idx):
        a, c = self.asm, self.cfg; lu = c["lam_nm"][0] * 1e-3; P = []
        for pol in self.CH:
            ax, ay = self._spectra(T, idx[0], pol)
            sz = a.grid(a.plane(ax, ay, lu, float(c["z_um"])), self.rw, self.rw)
            P.append(sz.reshape(self.npx, c["sub"], self.npx, c["sub"]).sum((1, 3)) * self.ds ** 2 / a.area)
        return torch.stack(P)

    def _metrics(self, T, idx):
        rep = super()._metrics(T, idx); P = torch.as_tensor(rep["arrays"]["P"], dtype=torch.float64, device=self.masks.device)
        only = [self.masks[1] & ~self.masks[0], self.masks[0] & ~self.masks[1]]
        own = [(P[0] * self.masks[0]).sum(), (P[1] * self.masks[1]).sum()]; wrong = [(P[0] * only[0]).sum(), (P[1] * only[1]).sum()]
        rep["scalars"].update(channels=["x -> " + str(self.target["words"][0]), "y -> " + str(self.target["words"][1])],
                              extinction_wrong_over_own=[float(wrong[i] / own[i]) for i in range(2)],
                              own_word_power=[float(v) for v in own], wrong_word_power=[float(v) for v in wrong])
        rep["arrays"].pop("composite_linear_xy", None)
        return rep


_CLASSES = {"e1_achromat": AchromatE1, "e2_rgb_holo": RGBHoloE2, "e3_volume_holo": VolumeHoloE3, "e3_sliced_object": SlicedObjectE3, "e4_dot_projector": DotProjectorE4, "e3_polcam": PolCamE3, "e3_dot_projector": DotProjectorE4, "e3_pol_holo": PolHoloE3}


def make_objective(name, cfg=None):
    """name in NAMES; cfg: overrides of DEFAULTS[name]. Returns a callable obj(T, lam_nm_T=None) -> 0-dim float64 tensor."""
    if name not in _CLASSES: raise ValueError(f"unknown objective {name}, expected one of {NAMES}")
    full = default_cfg(name); full.update(cfg or {})
    return _CLASSES[name](name, full, tuple(cfg or {}))


def metrics(name, T, cfg=None, lam_nm_T=None):
    return make_objective(name, cfg).metrics(T, lam_nm_T)
