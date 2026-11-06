"""G7-01: the metagrating application workflow of docs/G7_WORKFLOWS.md, case G7-01r3.

A density of 100 pixels (0.02 um) across the 2.0 um period of the metagrating fixture
(examples/meep_comparison/metagrating/geometry.json: grid, absorbers, pulse and DFT lines, with 3 um
more air above and substrate below) is extruded through the 0.5 um silicon layer on SiO2 and designed
with DesignProblem from three seeds at the 0.01 um mesh over 1120 fs. The design is robust: eroded,
nominal and dilated projections of the same filtered density are evaluated and the smallest of their
mean +1 transmitted order efficiencies at 1.50, 1.55 and 1.60 um (TE, E along the ridges, normal
incidence from the substrate) is maximized; the threshold shift is chosen first, on development seeds
only. Every seed's nominal design is binarized, opened and closed by 3 pixels, checked for the declared
linewidth and gap, evaluated over 2240 fs at 0.01 and 0.005 um for TE and TM at normal incidence and at
a fixed Bloch wavevector over 41 wavelengths, and checked against TORCWA 0.1.4.2 (rcwa_check.py, run in
the interpreter that holds TORCWA). The fixture's two-ridge baseline is evaluated in the same path. The
case file docs/validation/cases/G7-01r3.json supplies the declared quantities and criteria.

    python -m examples.g7.metagrating.workflow --stage select-run --shift 0.1 --seed 11 --selection-dir <dev dir> --out <records>
    python -m examples.g7.metagrating.workflow --stage select-decide --selection-dir <dev dir> --out <records>
    python -m examples.g7.metagrating.workflow --stage seed --seed 1 --skip-rcwa --out <records>   (one per declared seed, in parallel)
    python -m examples.g7.metagrating.workflow --stage baseline --out <records>          (the two-ridge baseline in the judged path)
    python -m examples.g7.metagrating.workflow --stage rcwa --out <records>              (one TORCWA process for every seed)
    python -m examples.g7.metagrating.workflow --stage judge --out <records>
    python -m examples.g7.metagrating.workflow --small --out <dir>                           (CPU-sized development run)

The full run needs CUDA; on the shared workstation every stage that uses the GPU runs under
D:/TorchFDTD/.local/gpu_lock.py.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time

import numpy as np
import torch

from torchfdtd import (AdjointOptions, Boundaries, BoundaryFace, DifferentiablePlaneSimulation, FieldMonitor, Project, Region,
                       Source, SpectrumSettings)
from torchfdtd.density_layer import _overlap
from torchfdtd.design_parameterization import DensityParameterization
from torchfdtd.design_problem import Continuation, DesignProblem
from torchfdtd.fabrication import measure_feature_sizes, morphological_open
from torchfdtd.solver import field_axes

C0 = 299792458.0
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASE_PATH = ROOT/'docs'/'validation'/'cases'/'G7-01r3.json'
GEOMETRY_PATH = ROOT/'examples'/'meep_comparison'/'metagrating'/'geometry.json'
RCWA_SCRIPT = HERE/'rcwa_check.py'
SELECTION = 'threshold-selection.json'
BASELINE = 'baseline.json'
REALIZATIONS = ('eroded', 'nominal', 'dilated')
ORDERS = (-1, 0, 1)
ALL_ORDERS = tuple(range(-3, 4))           # every order that can propagate in the band, for the energy balance
DESIGN_WAVELENGTHS_UM = (1.50, 1.55, 1.60)
ANGLE_DEG = 10.
ANGLE_WAVELENGTH_UM = 1.55
POLARIZATIONS = ('TE', 'TM')
INCIDENCES = ('normal', 'bloch')
TWO_RIDGE_PIXELS = (range(2, 6), range(29, 40))   # the fixture's ridges as pixels (Ez columns 2-5 and 29-39)
COURANT = .99/math.sqrt(2)


@dataclass(frozen=True)
class Settings:
    """Parameters of one run. Settings.declared() reads the declared run from the case file and the
    threshold selection; Settings.for_selection() is one development design of the selection; small_run()
    is a CPU development size that judges nothing."""
    design_mesh_um: float = .01
    optimization_mesh_um: float | None = None  # r4 optimizes on a short coarse grid, then judges at design_mesh_um
    fine_mesh_um: float = .005
    physical_time_fs: float = 1120.      # of every design run
    evaluation_time_fs: float = 2240.    # of every evaluation, of the designs, the baseline and the bare substrate
    extension_um: float = 3.             # air above the layer and substrate below it, added to the fixture's cell
    absorber_um: float = .4
    filter_radius_um: float | None = None
    threshold_shift: float | None = None  # robust projection at eta 0.5 +- shift; None designs the nominal projection alone
    open_close: bool = False             # periodic open-then-close of every thresholded design before its checks
    iterations_per_beta: int = 20
    betas: tuple = (8., 16., 32., 64.)
    learning_rate: float = .1
    band_points: int = 41
    seeds: tuple = (1, 2, 3)
    backend: str = 'cuda'
    checkpoints: int = 64
    # TORCWA applies the Laurent rule to TM too, whose error falls as 1/N; the count rises until the
    # first-order error estimate is at most rcwa_error_target.
    rcwa_harmonics: tuple = (40, 80, 160, 240, 320, 400, 480, 560, 640, 720, 800)
    rcwa_error_target: float = 3e-3
    rcwa_device: str = 'cuda'
    anomaly_exclusion_um: float = .02
    small: bool = False

    @classmethod
    def declared(cls, case, selection):
        """The declared run of the case file, with the threshold shift of the recorded selection."""
        f, p = case['fixture'], case_parameters(case)
        evaluation = f['evaluation']
        if 'three_ridge_segments' in f:
            fine = [m for m in evaluation['meshes_um'] if not math.isclose(m, evaluation['judged_mesh_um'])]
            if len(fine) != 1 or not math.isclose(f['design_mesh_um'], .02):
                raise ValueError('the constrained design must optimize at 0.02 um and judge at the two declared meshes')
            return cls(design_mesh_um=evaluation['judged_mesh_um'], optimization_mesh_um=f['design_mesh_um'],
                       fine_mesh_um=fine[0], physical_time_fs=p['design_time_fs'],
                       evaluation_time_fs=p['evaluation_time_fs'], extension_um=p['extension_um'],
                       absorber_um=p['absorber_um'], filter_radius_um=0., threshold_shift=None,
                       open_close=True, iterations_per_beta=f['iterations_per_beta'],
                       betas=tuple(float(beta) for beta in f['projection_beta']),
                       learning_rate=f['learning_rate'], band_points=evaluation['wavelengths_um'][2],
                       seeds=tuple(declared_seeds(case)), rcwa_error_target=p['rcwa_error_target'],
                       anomaly_exclusion_um=p['exclusion_um'])
        if not math.isclose(f['design_mesh_um'], evaluation['judged_mesh_um']):
            raise ValueError('the case file judges at a mesh other than the design mesh')
        if selection.get('chosen_threshold_shift') is None:
            raise ValueError('the threshold selection chose no shift; the declared run cannot start')
        fine = [m for m in evaluation['meshes_um'] if not math.isclose(m, evaluation['judged_mesh_um'])]
        return cls(design_mesh_um=f['design_mesh_um'], fine_mesh_um=fine[0], physical_time_fs=p['design_time_fs'],
                   evaluation_time_fs=p['evaluation_time_fs'], extension_um=p['extension_um'], absorber_um=p['absorber_um'],
                   filter_radius_um=f['filter_radius_um'], threshold_shift=selection['chosen_threshold_shift'], open_close=True,
                   iterations_per_beta=f['iterations_per_beta'], betas=tuple(float(b) for b in f['projection_beta']),
                   learning_rate=f['learning_rate'], band_points=evaluation['wavelengths_um'][2], seeds=tuple(declared_seeds(case)),
                   rcwa_error_target=p['rcwa_error_target'], anomaly_exclusion_um=p['exclusion_um'])

    @classmethod
    def for_selection(cls, case, threshold_shift, seed):
        """One development design of the threshold selection: the declared schedule at the selection mesh and
        time, thresholded without the open and close (the selection judges the robust design alone)."""
        p = case_parameters(case)
        return replace(cls.declared(case, dict(chosen_threshold_shift=threshold_shift)), design_mesh_um=p['selection_mesh_um'],
                       physical_time_fs=p['selection_time_fs'], evaluation_time_fs=p['selection_time_fs'], open_close=False, seeds=(seed,))

    @classmethod
    def small_run(cls):
        return cls(design_mesh_um=.04, fine_mesh_um=.02, physical_time_fs=56., evaluation_time_fs=56., extension_um=.2, filter_radius_um=.06,
                   iterations_per_beta=1, band_points=5, seeds=(1,), backend='cpu', checkpoints=4, rcwa_harmonics=(10, 20), rcwa_error_target=1.,
                   rcwa_device='cpu', small=True)

    @property
    def iterations(self):
        return self.iterations_per_beta*len(self.betas)


def load_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw.decode('utf-8')), hashlib.sha256(raw).hexdigest()


def declared():
    """The case file and the fixture geometry, with the shared quantities checked against each other."""
    case, case_sha = load_json(CASE_PATH)
    g, geometry_sha = load_json(GEOMETRY_PATH)
    f = case['fixture']
    bottom, top = f['design_layer_um']
    checks = dict(period=math.isclose(f['period_um'], g['period_um']), substrate=math.isclose(f['substrate_index'], g['substrate_index']),
                  ridge=math.isclose(f['ridge_index'], g['ridge_index']), layer_bottom=math.isclose(bottom, g['substrate_top_y_um']),
                  layer_height=math.isclose(top-bottom, g['ridge_height_um']), pixels=math.isclose(f['pixels']*f['pixel_um'], g['period_um']),
                  mesh=math.isclose(f['pixel_um'], g['mesh_um']), band=(f['evaluation']['wavelengths_um'] == [g['spectrum']['wavelength_start_um'],
                  g['spectrum']['wavelength_stop_um'], g['spectrum']['points']]),
                  absorber=math.isclose(case_parameters(case)['absorber_um'], g['pml_cells']*g['mesh_um']))
    if not all(checks.values()):
        raise ValueError(f'the case file and geometry.json disagree: {checks}')
    return case, g, dict(case=CASE_PATH.relative_to(ROOT).as_posix(), case_sha256=case_sha, geometry_sha256=geometry_sha)


def case_parameters(case):
    """The quantities of the case file that the workflow reads: the threshold-shift candidates, their
    fallback and development seeds, the selection mesh and time, the design and evaluation times, the
    anomaly exclusion, the TORCWA TM error target, the absorber, the cell extension and the 0.02 limit of
    (c), (d) and (e); the ones stated in prose are parsed and checked."""
    f = case['fixture']
    robust = f.get('robust_projection')

    def search(pattern, text):
        found = re.search(pattern, text)
        if found is None:
            raise ValueError(f'the case file no longer states {pattern!r}: {text}')
        return found
    extension = f['cell_extension_um']
    if extension['air_above_layer'] != extension['substrate_below_layer'] or 'every design and evaluation run' not in extension['applies_to']:
        raise ValueError('the workflow adds the same extension above and below in every run')
    for key in ('rcwa_agreement', 'mesh', 'energy_balance'):
        search(r'0\.02', case['acceptance'][key])
    search(r'larger of 0\.7706 and', case['acceptance']['performance'])
    search(r'3 pixels is applied to every binary design', f['open_close'])
    if robust is None:
        segment = f['three_ridge_segments']
        selection = dict(shift_candidates=[], shift_fallback=None,
                         development_seeds=[int(seed) for seed in segment['development_seeds']],
                         selection_mesh_um=float(f['design_mesh_um']),
                         selection_time_fs=float(f['physical_time_fs']['design']))
    else:
        selection = dict(shift_candidates=[float(v) for v in robust['threshold_shift_candidates']],
                         shift_fallback=float(search(r'if every candidate is rejected, (\d+\.\d+)', robust['threshold_shift_rule']).group(1)),
                         development_seeds=[int(s) for s in robust['development_seeds']], selection_mesh_um=float(robust['selection_mesh_um']),
                         selection_time_fs=float(robust['selection_time_fs']))
    return dict(**selection, design_time_fs=float(f['physical_time_fs']['design']),
                evaluation_time_fs=float(f['physical_time_fs']['evaluation']),
                exclusion_um=float(search(r'at least (\d+\.\d+) um from every Rayleigh anomaly', f['rayleigh_anomaly_exclusion']).group(1)),
                rcwa_error_target=float(search(r'at most (\d+\.\d+)', f['torcwa_tm_convergence']).group(1)),
                absorber_um=float(search(r'(\d+\.\d+) um CPML', f['absorber']).group(1)), extension_um=extension['air_above_layer'], limit=.02)


def declared_seeds(case):
    """The seeds named by the case file ('logits 0.5 * randn with seeds 1, 2, 3')."""
    return [int(s) for s in case['seed'].split('seeds', 1)[1].replace('and', ',').split(',') if s.strip()]


def pixel_count(g):
    """100 pixels of the fixture mesh (0.02 um) across the period."""
    return int(round(g['period_um']/g['mesh_um']))


def open_close(binary, side):
    """Periodic open-then-close of a thresholded pixel design by a line of `side` pixels: the opening removes every
    line narrower than `side`, the closing then fills every gap narrower than `side`, so both meet `side` pixels."""
    array = np.asarray(binary, dtype=bool).reshape(-1, 1)
    opened = morphological_open(array, side, boundary=('periodic', 'extend'))
    return ~morphological_open(~opened, side, boundary=('periodic', 'extend'))


def bloch_kx(g):
    """The declared fixed Bloch wavevector (per um): 10 degrees in the substrate at 1.55 um."""
    return 2*math.pi*g['substrate_index']*math.sin(math.radians(ANGLE_DEG))/ANGLE_WAVELENGTH_UM


def band_wavelengths(g, settings):
    s = g['spectrum']
    return np.linspace(s['wavelength_start_um'], s['wavelength_stop_um'], settings.band_points)


def rayleigh_anomalies(g, kx0):
    """Wavelengths (um) at which order m (|m| <= 3) grazes in a medium of index n: n L / |k_x0 L / (2 pi) + m|.
    Higher orders graze below 0.8 um, far outside the band."""
    q = kx0*g['period_um']/(2*math.pi)
    return [dict(order=m, medium=medium, wavelength_um=index*g['period_um']/abs(q+m))
            for m in ALL_ORDERS for medium, index in (('air', 1.), ('substrate', g['substrate_index'])) if abs(q+m) > 1e-12]


def anomaly_distance(g, kx0, wavelength_um):
    anomalies = np.array([a['wavelength_um'] for a in rayleigh_anomalies(g, kx0)])
    return np.min(abs(np.asarray(wavelength_um, dtype=float)[:, None]-anomalies[None, :]), axis=1)


def fixture_time_fs(g):
    """The fixture's duration: its steps at its own mesh (560.4 fs)."""
    return g['steps']*COURANT*g['mesh_um']*1e-6/C0*1e15


def build_project(g, mesh_um, polarization, incidence, settings, time_fs, *, monitors=('reflection', 'transmission')):
    """The fixture scene without structures: the permittivity is supplied explicitly by layer_epsilon.

    The cell grows by settings.extension_um of substrate below and of air above, so the source, the DFT
    lines and the layer keep their positions; the absorber keeps its physical thickness at every mesh
    (set per face, since Region.pml_cells is capped at 50) and the run lasts time_fs.
    TE drives Ez (E along the ridges), TM drives Hz (H along the ridges); the Bloch incidence sets the
    declared k_x on both periodic faces, and the plane source carries its phase exp(i k_x x).
    """
    lx, ly = g['cell_size_um']
    kind = 'bloch' if incidence == 'bloch' else 'periodic'
    phase = bloch_kx(g)*g['period_um'] if incidence == 'bloch' else 0.
    absorber = int(round(settings.absorber_um/mesh_um))
    cuda = settings.backend == 'cuda'
    region = Region(dimension='2d', size=(lx, ly+2*settings.extension_um, 1), mesh=mesh_um,
                    steps=int(round(time_fs*1e-15/(COURANT*mesh_um*1e-6/C0))), pml_cells=min(absorber, 50), courant_factor=.99,
                    backend=settings.backend, precision='float32', material_sampling='yee', cuda_kernel='fused' if cuda else 'torch',
                    cuda_monitor_kernel='fused' if cuda else 'torch',
                    boundaries=Boundaries(x_min=BoundaryFace(kind=kind), x_max=BoundaryFace(kind=kind), y_min=BoundaryFace(layers=absorber),
                                          y_max=BoundaryFace(layers=absorber)), bloch_phase=(phase, 0, 0))
    wave = g['source']['waveform']
    source = Source(id='source', name='source', kind='plane', component='Ez' if polarization == 'TE' else 'Hz', normal='y',
                    center=(0, g['source']['y_um'], 0), size=(lx, 0, 0), wavelength=wave['wavelength_um'],
                    pulse_cycles=wave['pulse_cycles'], amplitude=wave['amplitude'])
    lines = [FieldMonitor(id=name, name=name, normal='y', center=(0, g['monitors'][name+'_y_um'], 0), size=(lx, 0, 1),
                          spectrum=SpectrumSettings(sampling='frequency', apodization='none')) for name in monitors]
    return Project(name=f'g7-01-{polarization}-{incidence}-{mesh_um:g}um', region=region, sources=[source], monitors=lines)


def layer_epsilon(density, region, g):
    """Yee-component permittivity (Nx, Ny, 1, 3) of the substrate, the extruded density layer and air.

    Pixel k is centred on the Ez node x = -L/2 + k p, so its edges lie midway between the Ez nodes
    of the 0.02 um grid, as the fixture's ridges do. Every component receives the arithmetic mean
    of eps = 1 + (n_Si^2 - 1) rho (layer), n_sub^2 (below the layer) or 1 (above) over a cell-sized
    box centred at its own Yee position, the rule of periodic_density_layer: on the 0.02 um grid the
    Ez boxes coincide with the pixels and the layer rows, so a binary density is an exact staircase,
    and on the finer grids a tangential component on an interface receives the mean of both sides.
    """
    rho = density.reshape(-1)
    period, n_sub2, n_si2 = g['period_um'], g['substrate_index']**2, g['ridge_index']**2
    bottom = g['substrate_top_y_um']
    top = bottom+g['ridge_height_um']
    dx, dy = region.axis_steps[:2]
    fields = []
    for component in ('Ex', 'Ey', 'Ez'):
        x, y, _ = field_axes(region, component)
        row = _overlap(np.asarray(x), dx, rho.numel(), period, rho, 'sample_centers')@rho
        lo, hi = np.asarray(y)-dy/2, np.asarray(y)+dy/2
        substrate = np.clip(np.minimum(hi, bottom)-lo, 0, dy)/dy
        layer = np.clip(np.minimum(hi, top)-np.maximum(lo, bottom), 0, dy)/dy
        background = torch.as_tensor(substrate*n_sub2+(1-substrate), dtype=rho.dtype, device=rho.device)
        fields.append(background[None, :]+(n_si2-1)*row[:, None]*torch.as_tensor(layer, dtype=rho.dtype, device=rho.device)[None, :])
    return torch.stack(fields, -1)[:, :, None, :]


def order_wavevectors(wavelength_um, index, kx0, period_um, orders, device=None):
    """k_0 (F, 1), k_x (M,), k_y (F, M) in 1/um and the propagating mask of every order in a medium of `index`."""
    k0 = (2*math.pi/torch.as_tensor(np.asarray(wavelength_um), dtype=torch.float64, device=device))[:, None]
    kx = kx0+2*math.pi*torch.as_tensor(orders, dtype=torch.float64, device=device)/period_um
    ky2 = (index*k0)**2-kx[None, :]**2
    propagating = ky2 > 0
    return k0, kx, torch.sqrt(torch.where(propagating, ky2, torch.ones_like(ky2))), propagating


def order_branches(plane, wavelength_um, index, kx0, period_um, polarization, orders):
    """Forward and backward complex amplitudes of the diffraction orders on one DFT line.

    Order m has k_x = k_x0 + 2 pi m / L; its amplitude is the weighted Fourier mean of the line
    fields times exp(-i k_x x). The plane spectra use exp(+2 pi i f t) (exp(-i omega t) phasors), so a
    +y wave is exp(+i k_y y). TE: e = Ez, h = Hx, a +y wave has Hx = (k_y/k_0) Ez, e+- = (e +- (k_0/k_y) h)/2,
    branch power 0.5 (k_y/k_0) |e+-|^2 (the compare.py decomposition of the fixture). TM: h = Hz, e = Ex,
    a +y wave has Ex = -(k_y/(k_0 n^2)) Hz, h+- = (h -+ (k_0 n^2/k_y) e)/2, branch power
    0.5 (k_y/(k_0 n^2)) |h+-|^2. Returns (plus, minus, factor, ky) in complex128/float64 with branch
    power 0.5 factor |amplitude|^2 and factor 0 for an evanescent order.
    """
    fields, x, w = plane.fields.to(torch.complex128), plane.points_um[:, 0].to(torch.float64), plane.weights.to(torch.float64)
    k0, kx, ky, propagating = order_wavevectors(wavelength_um, index, kx0, period_um, orders, x.device)
    phase = torch.exp(-1j*kx[:, None]*x[None, :])*(w/w.sum())[None, :]
    if polarization == 'TE':
        ae, ah = fields[..., 2]@phase.T, fields[..., 3]@phase.T
        ratio = k0/ky
        plus, minus, factor = (ae+ratio*ah)/2, (ae-ratio*ah)/2, ky/k0
    else:
        ah, ae = fields[..., 5]@phase.T, fields[..., 0]@phase.T
        ratio = index**2*k0/ky
        plus, minus, factor = (ah-ratio*ae)/2, (ah+ratio*ae)/2, ky/(k0*index**2)
    return plus, minus, torch.where(propagating, factor, torch.zeros_like(factor)), ky


def incident(reference, wavelength_um, g, kx0, polarization):
    """Incident amplitude and branch-power factor: order 0, forward, reflection line of the bare-substrate run."""
    plus, _, factor, _ = order_branches(reference['reflection'], wavelength_um, g['substrate_index'], kx0, g['period_um'], polarization, (0,))
    return plus[:, 0], factor[:, 0]


def transmitted_efficiency(plane, incident_amplitude, incident_factor, wavelength_um, g, kx0, polarization, orders):
    plus, _, factor, _ = order_branches(plane, wavelength_um, 1., kx0, g['period_um'], polarization, orders)
    return factor*plus.abs()**2/(incident_factor*incident_amplitude.abs()**2)[:, None]


def fresnel_reflectance(g, wavelength_um, kx0, polarization):
    """Power reflectance of the bare substrate-air interface for the incident order."""
    n1, n2 = g['substrate_index'], 1.
    k0 = 2*np.pi/np.asarray(wavelength_um)
    c1 = np.sqrt(1-(kx0/(n1*k0))**2)
    c2 = np.sqrt(1-(kx0/(n2*k0))**2+0j)
    r = (n1*c1-n2*c2)/(n1*c1+n2*c2) if polarization == 'TE' else (n2*c1-n1*c2)/(n2*c1+n1*c2)
    return abs(r)**2


def order_table(sample, reference, wavelength_um, g, kx0, polarization):
    """Efficiencies of every order in ALL_ORDERS, complex amplitudes of ORDERS and the energy balance.

    Efficiencies are branch powers over the incident power of the bare-substrate reference. An
    amplitude a satisfies |a|^2 = efficiency (TE: E field, TM: H field); its phase is referenced to
    x = 0, to the incident wave at the substrate top (y = 0.01 um), to the layer top (y = 0.51 um)
    for transmitted and to the substrate top for reflected orders, with exp(-i omega t) phasors.
    """
    n_sub, period = g['substrate_index'], g['period_um']
    bottom = g['substrate_top_y_um']
    top = bottom+g['ridge_height_um']
    y_r, y_t = g['monitors']['reflection_y_um'], g['monitors']['transmission_y_um']
    ref_plus, _, ref_factor, ref_ky = order_branches(reference['reflection'], wavelength_um, n_sub, kx0, period, polarization, (0,))
    a_inc, f_inc = ref_plus[:, 0], ref_factor[:, 0]
    p_inc = f_inc*a_inc.abs()**2
    t_plus, _, t_factor, t_ky = order_branches(sample['transmission'], wavelength_um, 1., kx0, period, polarization, ALL_ORDERS)
    _, r_minus, r_factor, r_ky = order_branches(sample['reflection'], wavelength_um, n_sub, kx0, period, polarization, ALL_ORDERS)
    T = (t_factor*t_plus.abs()**2/p_inc[:, None]).cpu().numpy()
    R = (r_factor*r_minus.abs()**2/p_inc[:, None]).cpu().numpy()
    a_bottom = a_inc*torch.exp(1j*ref_ky[:, 0]*(bottom-y_r))
    t = (torch.sqrt(t_factor/f_inc[:, None])*t_plus*torch.exp(1j*t_ky*(top-y_t))/a_bottom[:, None]).cpu().numpy()
    r = (torch.sqrt(r_factor/f_inc[:, None])*r_minus*torch.exp(-1j*r_ky*(bottom-y_r))/a_bottom[:, None]).cpu().numpy()
    total = T.sum(axis=1)+R.sum(axis=1)
    declared_total = sum(T[:, ALL_ORDERS.index(m)]+R[:, ALL_ORDERS.index(m)] for m in ORDERS)

    def amplitudes(values):
        return {str(m): dict(re=values[:, ALL_ORDERS.index(m)].real.tolist(), im=values[:, ALL_ORDERS.index(m)].imag.tolist()) for m in ORDERS}
    return dict(T={str(m): T[:, k].tolist() for k, m in enumerate(ALL_ORDERS)}, R={str(m): R[:, k].tolist() for k, m in enumerate(ALL_ORDERS)},
                amplitudes=dict(t=amplitudes(t), r=amplitudes(r)), energy_sum=total.tolist(), declared_order_sum=declared_total.tolist(),
                max_abs_energy_residual=float(np.max(abs(1-total))),
                propagating_orders=dict(T=[m for k, m in enumerate(ALL_ORDERS) if bool((t_factor[:, k] > 0).any())],
                                        R=[m for k, m in enumerate(ALL_ORDERS) if bool((r_factor[:, k] > 0).any())]))


def reference_diagnostics(reference, wavelength_um, g, kx0, polarization):
    """Bare-substrate checks: order-0 T and R against Fresnel, power in any other order."""
    n_sub, period = g['substrate_index'], g['period_um']
    plus, minus, factor, _ = order_branches(reference['reflection'], wavelength_um, n_sub, kx0, period, polarization, ALL_ORDERS)
    t_plus, _, t_factor, _ = order_branches(reference['transmission'], wavelength_um, 1., kx0, period, polarization, ALL_ORDERS)
    i0 = ALL_ORDERS.index(0)
    p_inc = factor[:, i0]*plus[:, i0].abs()**2
    T0 = (t_factor[:, i0]*t_plus[:, i0].abs()**2/p_inc).cpu().numpy()
    R0 = (factor[:, i0]*minus[:, i0].abs()**2/p_inc).cpu().numpy()
    fresnel = fresnel_reflectance(g, wavelength_um, kx0, polarization)
    others = [k for k in range(len(ALL_ORDERS)) if k != i0]
    stray = ((factor*(plus.abs()**2+minus.abs()**2)+t_factor*t_plus.abs()**2)[:, others]/p_inc[:, None]).cpu().numpy()
    return dict(T0=T0.tolist(), R0=R0.tolist(), fresnel_R0=fresnel.tolist(), max_abs_T0_minus_fresnel=float(np.max(abs(T0-(1-fresnel)))),
                max_abs_R0_minus_fresnel=float(np.max(abs(R0-fresnel))), max_other_order_power_over_incident=float(stray.max()))


class CaseEvaluator:
    """One evaluated case (mesh, polarization, incidence): its bare-substrate reference once, then any design."""
    def __init__(self, g, settings, mesh_um, polarization, incidence):
        self.g, self.mesh_um, self.polarization, self.incidence = g, mesh_um, polarization, incidence
        self.kx0 = bloch_kx(g) if incidence == 'bloch' else 0.
        self.device = torch.device(settings.backend)
        self.wavelength = band_wavelengths(g, settings)
        self.frequency = C0/(self.wavelength*1e-6)
        self.project = build_project(g, mesh_um, polarization, incidence, settings, settings.evaluation_time_fs)
        self.model = DifferentiablePlaneSimulation(self.project, AdjointOptions(checkpoints=0))
        started = _clock(self.device)
        self.reference = self.planes(np.zeros(pixel_count(g)))
        self.reference_seconds = _clock(self.device)-started
        self.diagnostics = reference_diagnostics(self.reference, self.wavelength, g, self.kx0, polarization)

    @torch.no_grad()
    def planes(self, density):
        epsilon = layer_epsilon(torch.as_tensor(np.asarray(density), dtype=torch.float32, device=self.device), self.project.region, self.g)
        return self.model(epsilon, self.frequency)

    def evaluate(self, density):
        started = _clock(self.device)
        table = order_table(self.planes(density), self.reference, self.wavelength, self.g, self.kx0, self.polarization)
        return dict(self.describe(), wall_seconds=_clock(self.device)-started, **table)

    def describe(self):
        r = self.project.region
        return dict(mesh_um=self.mesh_um, polarization=self.polarization, incidence=self.incidence, kx_per_um=self.kx0,
                    bloch_phase_rad=r.bloch_phase[0], cells=list(r.shape[:2]), cell_um=list(r.size[:2]), steps=r.steps, dt_s=r.time_step,
                    physical_time_fs=r.steps*r.time_step*1e15, absorber_cells=r.pml_layers(1, 0),
                    source_component=self.project.sources[0].component, wavelength_um=self.wavelength.tolist())


def _clock(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


class TransmissionObjective:
    """DesignProblem objective: minus the mean T+1 (TE, normal incidence) at the design wavelengths.

    The transmission line of the design-mesh scene is a differentiable plane; the incident power
    comes once from the reflection line of the bare-substrate run of the same scene.
    """
    def __init__(self, g, settings):
        self.g = g
        self.device = torch.device(settings.backend)
        self.wavelength = np.asarray(DESIGN_WAVELENGTHS_UM)
        self.frequency = C0/(self.wavelength*1e-6)
        time_fs = settings.physical_time_fs
        project = build_project(g, settings.optimization_mesh_um or settings.design_mesh_um, 'TE', 'normal', settings, time_fs, monitors=('transmission',))
        self.region = project.region
        self.model = DifferentiablePlaneSimulation(project, AdjointOptions(checkpoints=settings.checkpoints))
        reference = DifferentiablePlaneSimulation(build_project(g, settings.optimization_mesh_um or settings.design_mesh_um, 'TE', 'normal', settings, time_fs,
                                                                monitors=('reflection',)), AdjointOptions(checkpoints=0))
        with torch.no_grad():
            planes = reference(layer_epsilon(torch.zeros(pixel_count(g), device=self.device), self.region, g), self.frequency)
        self.incident = incident(planes, self.wavelength, g, 0., 'TE')

    def efficiencies(self, density):
        epsilon = layer_epsilon(density.to(self.device), self.region, self.g)
        plane = self.model(epsilon, self.frequency)['transmission']
        return transmitted_efficiency(plane, *self.incident, self.wavelength, self.g, 0., 'TE', ORDERS)

    def __call__(self, density):
        """Minus the mean T+1 of `density`, or of the smallest of its three realizations when it carries them.

        A RobustDensity output carries its eroded and dilated projections: all three are evaluated
        without gradient, and the loss and gradient are those of the realization with the smallest mean
        T+1 (the gradient of the minimum). A fixed density, as in DesignProblem.evaluate, is evaluated alone.
        """
        realizations = getattr(density, 'realizations', None)
        extra = {}
        if realizations is not None:
            designs = dict(eroded=realizations['eroded'], nominal=density, dilated=realizations['dilated'])
            with torch.no_grad():
                values = {name: float(self.efficiencies(design.detach())[:, ORDERS.index(1)].mean()) for name, design in designs.items()}
            active = min(REALIZATIONS, key=lambda name: values[name])
            extra = {f'T+1 mean {name}': value for name, value in values.items()}
            extra['active realization'] = float(REALIZATIONS.index(active))
            density = designs[active]
        T = self.efficiencies(density)
        plus = T[:, ORDERS.index(1)]
        metrics = {f'T+1 {w:.2f} um': plus[i] for i, w in enumerate(self.wavelength)}
        metrics.update({f'T{m:+d} mean': T[:, k].mean() for k, m in enumerate(ORDERS)}, **extra)
        return -plus.mean(), metrics


def project(filtered, beta, eta):
    """The tanh projection of DensityParameterization at threshold eta, applied to a filtered density."""
    beta, eta = filtered.new_tensor(beta), filtered.new_tensor(eta)
    lower = torch.tanh(beta*eta)
    return ((lower+torch.tanh(beta*(filtered-eta)))/(lower+torch.tanh(beta*(1-eta)))).clamp(0, 1)


class RobustDensity(DensityParameterization):
    """A DensityParameterization whose soft output, the nominal projection (eta 0.5) of the filtered density,
    carries the eroded (eta 0.5 + shift) and dilated (eta 0.5 - shift) projections of the same filtered
    density in its `realizations` attribute, for the robust objective. The hard output is the nominal
    design thresholded at 0.5, as for the parent class."""
    def __init__(self, *args, threshold_shift, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < threshold_shift < float(self.eta) or not float(self.eta)+threshold_shift < 1:
            raise ValueError('the threshold shift must keep both shifted thresholds inside (0, 1)')
        self.threshold_shift = float(threshold_shift)
        self._config['threshold_shift'] = self.threshold_shift

    def filtered(self):
        """The filtered density: the parent's output with its projection switched off (beta 0) for this call."""
        beta = float(self.beta)
        with torch.no_grad():
            self.beta.fill_(0.)
        try:
            return super().forward()
        finally:
            with torch.no_grad():
                self.beta.fill_(beta)

    def forward(self, *, hard=False, straight_through=False):
        if hard or straight_through:
            return super().forward(hard=hard, straight_through=straight_through)
        filtered, beta, eta = self.filtered(), float(self.beta), float(self.eta)
        nominal = project(filtered, beta, eta)
        nominal.realizations = dict(eroded=project(filtered, beta, eta+self.threshold_shift),
                                    dilated=project(filtered, beta, eta-self.threshold_shift))
        return nominal


def design_seed(seed, objective, settings, case, *, checkpoint=None):
    """One start: 0.5 randn logits, conic filter, tanh projection continuation (robust with eroded and dilated
    projections when settings.threshold_shift is set), Adam; the nominal design binarized at 0.5."""
    f = case['fixture']
    pixels, pixel_um = f['pixels'], f['pixel_um']
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    initial = .5*torch.randn((pixels, 1), generator=generator)
    options = dict(spacing_um=(pixel_um, f['design_layer_um'][1]-f['design_layer_um'][0]), initial=initial, mode='logits',
                   filter_radius_um=settings.filter_radius_um, boundary='periodic', beta=settings.betas[0], eta=.5)
    design = (DensityParameterization((pixels, 1), **options) if settings.threshold_shift is None
              else RobustDensity((pixels, 1), threshold_shift=settings.threshold_shift, **options))
    optimizer = torch.optim.Adam(design.parameters(), lr=settings.learning_rate)
    problem = DesignProblem(design, objective, optimizer, name=f'g7-01-seed{seed}',
                            continuation=Continuation(every=settings.iterations_per_beta, factor=2., maximum=settings.betas[-1]))
    history = problem.run(settings.iterations, checkpoint=checkpoint, resume=checkpoint is not None)
    fab = f['fabrication']
    fabrication = problem.fabrication(spacing_um=pixel_um, min_linewidth_um=fab['min_linewidth_um'], min_gap_um=fab['min_gap_um'],
                                      perturbation_um=pixel_um, boundary=(fab['boundary'], 'extend'))
    binary = problem.density(hard=True)
    extra = {}
    if settings.threshold_shift is not None:
        with torch.no_grad():
            nominal = problem.parameterization()
            realizations = dict(nominal.realizations, nominal=nominal)
        patterns = {name: (value >= .5).cpu().numpy() for name, value in realizations.items()}
        extra['robust'] = dict(threshold_shift=settings.threshold_shift, eta={name: .5+d for name, d in
                                                                              zip(REALIZATIONS, (settings.threshold_shift, 0., -settings.threshold_shift))},
                               thresholded={name: pattern[:, 0].astype(int).tolist() for name, pattern in patterns.items()},
                               feature_sizes={name: measure_feature_sizes(pattern, pixel_um, boundary=(fab['boundary'], 'extend')).report()
                                              for name, pattern in patterns.items()})
    if settings.open_close:
        # The optimizer never sees the open and close; every check below uses the processed design.
        side = int(round(fab['min_linewidth_um']/pixel_um))
        if side != int(round(fab['min_gap_um']/pixel_um)):
            raise ValueError('one open-then-close side serves the declared linewidth and gap only when they are equal')
        thresholded = binary.bool().cpu().numpy()
        processed = open_close(thresholded, side)
        sizes = measure_feature_sizes(processed, pixel_um, boundary=(fab['boundary'], 'extend'))
        violations = list(sizes.violations(min_linewidth_um=fab['min_linewidth_um'], min_gap_um=fab['min_gap_um']))
        loss_before, metrics_before = problem.evaluate(binary)
        binary = torch.as_tensor(processed, dtype=binary.dtype)
        extra['open_close'] = dict(side_pixels=side, boundary=[fab['boundary'], 'extend'], changed_pixels=int((thresholded != processed).sum()),
                                   thresholded=thresholded[:, 0].astype(int).tolist(), processed=processed[:, 0].astype(int).tolist(),
                                   thresholded_design_mesh=dict(objective=loss_before, metrics=metrics_before))
        fabrication = dict(feature_sizes=sizes.report(), violations=violations, satisfies_declared_constraints=not violations,
                           declared=dict(min_linewidth_um=fab['min_linewidth_um'], min_gap_um=fab['min_gap_um'], boundary=[fab['boundary'], 'extend']),
                           measured='the design after the open and close', thresholded=fabrication)
    loss, metrics = problem.evaluate(binary)
    return dict(seed=seed, iterations=problem.iteration, filter_radius_um=settings.filter_radius_um, history=history,
                initial_logits=initial[:, 0].tolist(), final_logits=design.design.detach()[:, 0].tolist(),
                smooth_density=problem.density()[:, 0].tolist(), binary=binary[:, 0].int().tolist(),
                design_mesh_binary=dict(objective=loss, metrics=metrics), fabrication=fabrication, **extra,
                design_wall_seconds=time.perf_counter()-started, iteration_seconds=sum(h['elapsed_s'] for h in history))


def two_ridge_density(g):
    rho = np.zeros(pixel_count(g))
    for ridge in TWO_RIDGE_PIXELS:
        rho[list(ridge)] = 1
    return rho


def fixture_reproduction(g, backend='cuda'):
    """The fixture's two ridges in the fixture's own scene (0.02 um, 560 fs, no extension) through this
    workflow's forward, against the committed native-solver record of that scene."""
    path = ROOT/'docs'/'validation'/'meep_comparison'/'metagrating_comparison.json'
    committed, _ = load_json(path)
    recorded = committed['efficiencies']['torchfdtd']
    settings = replace(Settings(), evaluation_time_fs=fixture_time_fs(g), extension_um=0., backend=backend)
    e = CaseEvaluator(g, settings, g['mesh_um'], 'TE', 'normal').evaluate(two_ridge_density(g))
    difference = max(float(np.max(abs(np.asarray(e[kind][str(m)])-np.asarray(recorded[kind][str(m)])))) for kind in ('T', 'R') for m in ORDERS)
    return dict(record=path.relative_to(ROOT).as_posix(), steps=e['steps'], band_mean_T_plus1=float(np.mean(e['T']['1'])),
                recorded_band_mean_T_plus1=float(np.mean(recorded['T']['1'])), max_abs_order_efficiency_difference=difference)


def rcwa_job(g, settings, designs):
    """Every design, polarization and incidence over the band, with the harmonic sweep at the design wavelengths."""
    kx = bloch_kx(g)
    return dict(period_um=g['period_um'], substrate_index=g['substrate_index'], ridge_index=g['ridge_index'],
                thickness_um=g['ridge_height_um'], samples_per_pixel=200, orders=list(ALL_ORDERS), harmonics=list(settings.rcwa_harmonics),
                error_target=settings.rcwa_error_target, device=settings.rcwa_device, designs=designs,
                cases=[dict(design=name, polarization=p, incidence=i, kx_per_um=kx if i == 'bloch' else 0.,
                            wavelengths_um=band_wavelengths(g, settings).tolist(), convergence_wavelengths_um=list(DESIGN_WAVELENGTHS_UM))
                       for name in designs for p in POLARIZATIONS for i in INCIDENCES])


def run_rcwa(job, python, directory):
    job_path, out = Path(directory)/'rcwa-job.json', Path(directory)/'rcwa-result.json'
    job_path.write_text(json.dumps(job), encoding='utf-8')
    subprocess.run([python, str(RCWA_SCRIPT), '--job', str(job_path), '--out', str(out)], check=True)
    return json.loads(out.read_text(encoding='utf-8'))


def find(record, mesh_um, polarization, incidence):
    matches = [e for e in record['evaluations']
               if math.isclose(e['mesh_um'], mesh_um) and e['polarization'] == polarization and e['incidence'] == incidence]
    if len(matches) != 1:
        raise ValueError(f"seed {record['seed']}: {len(matches)} evaluations of {mesh_um} um {polarization} {incidence}")
    return matches[0]


def worst_difference(evaluation, reference, keep=None):
    """Largest |TorchFDTD - TORCWA| over every order efficiency (T and R) and the kept wavelengths, with its location."""
    keep = np.ones(len(evaluation['wavelength_um']), bool) if keep is None else np.asarray(keep)
    worst = (-1., None, None)
    for kind in ('T', 'R'):
        for m in ALL_ORDERS:
            diff = np.where(keep, abs(np.asarray(evaluation[kind][str(m)])-np.asarray(reference[kind][str(m)])), -1.)
            i = int(np.argmax(diff))
            if diff[i] > worst[0]:
                worst = (float(diff[i]), evaluation['wavelength_um'][i], f'{kind}{m:+d}')
    return dict(value=worst[0], at_wavelength_um=worst[1], at_order=worst[2])


def judge(records, case, settings, g, baseline=None):
    """Every acceptance criterion of the case file, from the seed records and the baseline record alone.

    (b) is judged against the larger of the declared baseline and the baseline design evaluated in the
    judged path (a missing baseline record fails it). (c) and (e) keep the wavelengths at least
    settings.anomaly_exclusion_um from every Rayleigh anomaly of the evaluated k_x and report the excluded
    ones; the values over all wavelengths are recorded beside them.
    """
    f = case['fixture']
    limit = case_parameters(case)['limit']
    design, fine = settings.design_mesh_um, settings.fine_mesh_um
    expected = {(m, p, i) for m in (design, fine) for p in POLARIZATIONS for i in INCIDENCES}
    complete = {r['seed']: dict(history=len(r['history']) == settings.iterations, design=len(r['binary']) == f['pixels'],
                                evaluations={(e['mesh_um'], e['polarization'], e['incidence']) for e in r['evaluations']} == expected
                                and all(len(e['wavelength_um']) == settings.band_points for e in r['evaluations']),
                                rcwa={(c['polarization'], c['incidence']) for c in r.get('rcwa') or []}
                                == {(p, i) for p in POLARIZATIONS for i in INCIDENCES}, environment='environment' in r)
                for r in records}
    criteria = dict(all_seeds=dict(value=sorted(complete), declared=declared_seeds(case), complete=complete,
                                   passed=sorted(complete) == declared_seeds(case) and all(all(v.values()) for v in complete.values())))

    scores = {r['seed']: float(np.mean(find(r, design, 'TE', 'normal')['T']['1'])) for r in records}
    best = max(scores, key=scores.get)
    declared_baseline = case['baseline']['band_mean_T_plus1']
    judged_baseline = None if baseline is None else baseline['band_mean_T_plus1']
    bound = max(declared_baseline, judged_baseline if judged_baseline is not None else declared_baseline)
    criteria['performance'] = dict(value=scores[best], limit_min=bound, declared_baseline=declared_baseline, judged_path_baseline=judged_baseline,
                                   best_seed=best, per_seed=scores, passed=judged_baseline is not None and scores[best] >= bound,
                                   statistic=f'band-mean T+1, TE, normal incidence, binary design after the open and close, {design:g} um, '
                                             f'{settings.evaluation_time_fs:g} fs, against the larger of the declared baseline and the '
                                             'baseline design in the same path')

    wavelength = band_wavelengths(g, settings)
    keep = {i: anomaly_distance(g, bloch_kx(g) if i == 'bloch' else 0., wavelength) >= settings.anomaly_exclusion_um-1e-12 for i in INCIDENCES}
    exclusion = {i: dict(anomalies=rayleigh_anomalies(g, bloch_kx(g) if i == 'bloch' else 0.), excluded_um=wavelength[~keep[i]].round(6).tolist(),
                         kept=int(keep[i].sum())) for i in INCIDENCES}
    judged, other = [], []
    for r in records:
        for c in r.get('rcwa') or []:
            for mesh in (design, fine):
                e = find(r, mesh, c['polarization'], c['incidence'])
                row = dict(seed=r['seed'], polarization=c['polarization'], incidence=c['incidence'], mesh_um=mesh, rcwa_harmonics=c['harmonics'],
                           rcwa_converged=c['converged'], rcwa_error_estimate=c['first_order_error_estimate'],
                           **worst_difference(e, c, keep[c['incidence']]), all_wavelengths=worst_difference(e, c))
                (judged if math.isclose(mesh, design) else other).append(row)
    worst = max(judged, key=lambda row: row['value']) if judged else None
    criteria['rcwa_agreement'] = dict(
        value=None if worst is None else worst['value'], limit_max=limit, worst=worst, judged=judged, not_judged=other, exclusion=exclusion,
        rcwa_converged=bool(judged) and all(row['rcwa_converged'] for row in judged),
        statistic=f'max |eta_TorchFDTD - eta_TORCWA| over every propagating order (T and R), TE and TM at {design:g} um, at the '
                  f'wavelengths at least {settings.anomaly_exclusion_um:g} um from every Rayleigh anomaly of the evaluated k_x',
        passed=(len(judged) == len(records)*len(POLARIZATIONS)*len(INCIDENCES)
                and all(row['rcwa_converged'] for row in judged)
                and worst is not None and worst['value'] <= limit))

    mesh_change = {}
    for r in records:
        coarse, refined = find(r, design, 'TE', 'normal'), find(r, fine, 'TE', 'normal')
        diff = abs(np.asarray(refined['T']['1'])-np.asarray(coarse['T']['1']))
        mesh_change[r['seed']] = dict(value=float(diff.max()), at_wavelength_um=coarse['wavelength_um'][int(np.argmax(diff))])
    value = max(v['value'] for v in mesh_change.values())
    criteria['mesh'] = dict(value=value, limit_max=limit, per_seed=mesh_change, passed=value <= limit,
                            statistic=f'max over wavelengths of |T+1({fine:g} um) - T+1({design:g} um)|, TE, normal incidence, every seed')

    energy = []
    for r in records:
        for e in r['evaluations']:
            residual = abs(1-np.asarray(e['energy_sum']))
            kept = np.where(keep[e['incidence']], residual, -1.)
            energy.append(dict(seed=r['seed'], mesh_um=e['mesh_um'], polarization=e['polarization'], incidence=e['incidence'],
                               value=float(kept.max()), at_wavelength_um=e['wavelength_um'][int(np.argmax(kept))],
                               all_wavelengths=float(residual.max()), all_at_wavelength_um=e['wavelength_um'][int(np.argmax(residual))],
                               declared_orders_only=float(np.max(abs(1-np.asarray(e['declared_order_sum'])))), propagating_orders=e['propagating_orders']))
    worst = max(energy, key=lambda row: row['value'])
    criteria['energy_balance'] = dict(value=worst['value'], limit_max=limit, worst=worst, cases=energy, exclusion=exclusion, passed=worst['value'] <= limit,
                                      statistic='max |1 - sum R - sum T| over every propagating order of every evaluated case, at the wavelengths '
                                                f'at least {settings.anomaly_exclusion_um:g} um from every Rayleigh anomaly of the evaluated k_x')

    violations = {r['seed']: r['fabrication']['violations'] for r in records}
    criteria['fabrication'] = dict(value=violations[best], best_seed=best, per_seed=violations, passed=not violations[best],
                                   feature_sizes={r['seed']: r['fabrication']['feature_sizes'] for r in records},
                                   declared=dict(min_linewidth_um=f['fabrication']['min_linewidth_um'], min_gap_um=f['fabrication']['min_gap_um']))
    return criteria


def environment(device):
    def git(*args):
        try:
            return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    import importlib.metadata
    import torchfdtd
    version = getattr(torchfdtd, '__version__', None)
    return dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda,
                numpy=np.__version__, torchfdtd=dict(file=torchfdtd.__file__, version=version or importlib.metadata.version('torchfdtd'),
                                                     version_source='__version__' if version else 'distribution metadata'),
                device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                commit=git('rev-parse', 'HEAD'), tracked_changes=bool(git('status', '--porcelain', '--untracked-files=no')))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=1, allow_nan=False)+'\n').encode('utf-8'))


def open_close_side(case):
    """The declared minimum linewidth and gap in pixels, the side of the open and close."""
    f = case['fixture']
    fab = f['fabrication']
    side = int(round(fab['min_linewidth_um']/f['pixel_um']))
    if side != int(round(fab['min_gap_um']/f['pixel_um'])):
        raise ValueError('one open-then-close side serves the declared linewidth and gap only when they are equal')
    return side


def open_close_block(case, evaluator, thresholded):
    """The open and close of a thresholded design with its changed pixels, the feature sizes of the processed
    design and the band-mean T+1 of both patterns through `evaluator`; the evaluation of the processed one too."""
    f = case['fixture']
    fab = f['fabrication']
    thresholded = np.asarray(thresholded, dtype=bool).reshape(-1, 1)
    processed = open_close(thresholded, open_close_side(case))
    sizes = measure_feature_sizes(processed, f['pixel_um'], boundary=(fab['boundary'], 'extend'))
    before = evaluator.evaluate(thresholded[:, 0].astype(float))
    changed = int((thresholded != processed).sum())
    after = evaluator.evaluate(processed[:, 0].astype(float)) if changed else before
    block = dict(side_pixels=open_close_side(case), changed_pixels=changed, thresholded=thresholded[:, 0].astype(int).tolist(),
                 processed=processed[:, 0].astype(int).tolist(), linewidth_pixels=sizes.linewidth_pixels, gap_pixels=sizes.gap_pixels,
                 violations=list(sizes.violations(min_linewidth_um=fab['min_linewidth_um'], min_gap_um=fab['min_gap_um'])),
                 mesh_um=evaluator.mesh_um, physical_time_fs=after['physical_time_fs'],
                 band_mean_T_plus1=dict(thresholded=float(np.mean(before['T']['1'])), processed=float(np.mean(after['T']['1']))))
    return block, after


def select_run(threshold_shift, seed, directory):
    """One development design of the threshold selection, kept in `directory` (outside the repository).

    The robust design runs at the selection mesh and time and records the feature sizes of its nominal
    design thresholded at 0.5 (the robust design alone). The declared open and close is then applied,
    with its changed pixels, the feature sizes of the processed pattern and the band-mean T+1 (TE,
    normal incidence) of both patterns in the judged path. A design already in `directory` is reused, so
    only the open-and-close block is computed again.
    """
    case, g, provenance = declared()
    if seed not in case_parameters(case)['development_seeds'] or seed in declared_seeds(case):
        raise ValueError(f'seed {seed} is not a development seed of the case file')
    path = Path(directory)/f'selection-d{threshold_shift:g}-seed{seed}.json'
    if path.exists():
        run = load_json(path)[0]
    else:
        settings = Settings.for_selection(case, threshold_shift, seed)
        record = design_seed(seed, TransmissionObjective(g, settings), settings, case)
        e = CaseEvaluator(g, settings, settings.design_mesh_um, 'TE', 'normal').evaluate(record['binary'])
        sizes = record['fabrication']['feature_sizes']
        run = dict(threshold_shift=threshold_shift, seed=seed, mesh_um=settings.design_mesh_um, physical_time_fs=e['physical_time_fs'],
                   linewidth_pixels=sizes['linewidth_pixels'], gap_pixels=sizes['gap_pixels'], violations=record['fabrication']['violations'],
                   smooth_objective=-record['history'][-1]['objective'], binary_objective=-record['design_mesh_binary']['objective'],
                   band_mean_T_plus1=float(np.mean(e['T']['1'])), binary=record['binary'],
                   robust_feature_sizes={name: dict(linewidth_pixels=v['linewidth_pixels'], gap_pixels=v['gap_pixels'])
                                         for name, v in record['robust']['feature_sizes'].items()},
                   seconds=record['design_wall_seconds'], record=record, evaluation=e, settings=asdict(settings), provenance=provenance,
                   environment=environment(torch.device(settings.backend)))
    judged = Settings.declared(case, dict(chosen_threshold_shift=threshold_shift))
    evaluator = CaseEvaluator(g, judged, judged.design_mesh_um, 'TE', 'normal')
    block, _ = open_close_block(case, evaluator, run['binary'])
    run['open_close'] = dict(block, environment=environment(evaluator.device))
    write_json(path, run)
    row = {k: v for k, v in run.items() if k not in ('binary', 'record', 'evaluation', 'settings', 'provenance', 'environment')}
    summary = {k: v for k, v in row['open_close'].items() if k not in ('thresholded', 'processed', 'environment')}
    print(json.dumps(dict(row, open_close=summary)), flush=True)
    return row


def select_decide(directory):
    """The selection record of the development designs in `directory`.

    Candidates in increasing order: the smallest threshold shift whose nominal designs of every development
    seed, thresholded at 0.5, meet the declared linewidth and gap without the open and close. A candidate is
    rejected at its first violating design; the result is None, with the candidate still to run, until a
    candidate passes; if every candidate is rejected, the declared fallback.
    """
    case, _, provenance = declared()
    p = case_parameters(case)
    runs = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(Path(directory).glob('selection-d*-seed*.json'))]
    rows = [{k: v for k, v in run.items() if k not in ('record', 'evaluation', 'settings', 'provenance', 'environment')} for run in runs]
    commits = {run['environment']['commit'] for run in runs} | {run['open_close']['environment']['commit'] for run in runs if 'open_close' in run}
    for row in rows:
        if 'open_close' in row:
            row['open_close'] = {k: v for k, v in row['open_close'].items() if k != 'environment'}
    chosen, pending, rejected = None, [], []
    for shift in sorted(p['shift_candidates']):
        done = {row['seed']: row for row in rows if math.isclose(row['threshold_shift'], shift)}
        if any(row['violations'] for row in done.values()):
            rejected.append(shift)
            continue
        if set(done) != set(p['development_seeds']) or not all('open_close' in row for row in done.values()):
            pending.append(shift)
            break
        chosen = shift
        break
    fallback = chosen is None and not pending
    if fallback:
        chosen = p['shift_fallback']
    return dict(schema='torchfdtd-g7-01-threshold-selection-v1', task='G7-01', provenance=provenance,
                rule=case['fixture']['robust_projection']['threshold_shift_rule'], candidates=sorted(p['shift_candidates']),
                development_seeds=p['development_seeds'], judged_seeds_used=False, selection_mesh_um=p['selection_mesh_um'],
                selection_time_fs=p['selection_time_fs'], extension_um=p['extension_um'], chosen_threshold_shift=chosen,
                chosen_by_fallback=fallback, rejected=rejected, pending=pending, open_close=True, open_close_side_pixels=open_close_side(case),
                runs=rows, commits=sorted(commits), development_records='kept outside the repository (g7_archive/G7-01/r3-selection)')


def seed_stage(settings, seed, out, *, rcwa_python, selection=None, checkpoint=None, skip_rcwa=False):
    """Design, fabrication check and every evaluation of one seed, and its TORCWA check unless skipped; writes seed<N>.json."""
    case, g, provenance = declared()
    device = torch.device(settings.backend)
    started = time.perf_counter()
    if 'three_ridge_segments' in case['fixture']:
        from .r4_design import design_seed as constrained_design_seed
        record = constrained_design_seed(seed, TransmissionObjective(g, settings), settings, case,
                                        checkpoint=checkpoint)
    else:
        record = design_seed(seed, TransmissionObjective(g, settings), settings, case, checkpoint=checkpoint)
    print(json.dumps(dict(seed=seed, smooth=-record['history'][-1]['objective'], binary=record['design_mesh_binary']['metrics'],
                          violations=record['fabrication']['violations'], changed_pixels=record.get('open_close', {}).get('changed_pixels'),
                          seconds=round(record['design_wall_seconds'], 1))), flush=True)
    wall = dict(design=time.perf_counter()-started)
    evaluations, references = [], []
    for mesh in (settings.design_mesh_um, settings.fine_mesh_um):
        for polarization in POLARIZATIONS:
            for incidence in INCIDENCES:
                evaluator = CaseEvaluator(g, settings, mesh, polarization, incidence)
                references.append(dict(evaluator.describe(), reference_wall_seconds=evaluator.reference_seconds, **evaluator.diagnostics))
                evaluations.append(evaluator.evaluate(record['binary']))
                if settings.open_close and (mesh, polarization, incidence) == (settings.design_mesh_um, 'TE', 'normal'):
                    before = (evaluations[-1] if record['open_close']['changed_pixels'] == 0
                              else evaluator.evaluate(record['open_close']['thresholded']))
                    record['open_close']['band_mean_T_plus1'] = dict(thresholded=float(np.mean(before['T']['1'])),
                                                                     processed=float(np.mean(evaluations[-1]['T']['1'])),
                                                                     mesh_um=mesh, polarization=polarization, incidence=incidence)
                print(json.dumps(dict(seed=seed, mesh_um=mesh, polarization=polarization, incidence=incidence,
                                      band_mean_T_plus1=round(float(np.mean(evaluations[-1]['T']['1'])), 4),
                                      energy=round(evaluations[-1]['max_abs_energy_residual'], 4))), flush=True)
                del evaluator
    wall['evaluation'] = time.perf_counter()-started-wall['design']
    record.update(evaluations=evaluations, references=references, schema='torchfdtd-g7-01-seed-v3', task='G7-01', settings=asdict(settings),
                  provenance=provenance, environment=environment(device),
                  selection=None if selection is None else dict(chosen_threshold_shift=selection['chosen_threshold_shift'],
                                                                open_close=selection['open_close']))
    wall['total'] = time.perf_counter()-started
    record['wall_seconds'] = wall
    if not skip_rcwa:
        add_rcwa([record], settings, rcwa_python)
    write_json(Path(out)/f'seed{seed}.json', record)
    return record


def baseline_stage(settings, out):
    """The fixture's two-ridge baseline in the judged path (design mesh, evaluation time, TE, normal incidence,
    the open and close applied); writes baseline.json."""
    case, g, provenance = declared()
    device = torch.device(settings.backend)
    started = time.perf_counter()
    evaluator = CaseEvaluator(g, settings, settings.design_mesh_um, 'TE', 'normal')
    block, evaluation = open_close_block(case, evaluator, two_ridge_density(g))
    record = dict(schema='torchfdtd-g7-01-baseline-v1', task='G7-01', design='the two ridges of geometry.json (pixels 2-5 and 29-39)',
                  source=case['baseline']['source'], declared_band_mean_T_plus1=case['baseline']['band_mean_T_plus1'],
                  band_mean_T_plus1=float(np.mean(evaluation['T']['1'])), open_close=block, evaluation=evaluation,
                  reference=dict(evaluator.describe(), **evaluator.diagnostics), settings=asdict(settings), provenance=provenance,
                  environment=environment(device), wall_seconds=time.perf_counter()-started)
    write_json(Path(out)/BASELINE, record)
    print(json.dumps(dict(baseline_band_mean_T_plus1=record['band_mean_T_plus1'], changed_pixels=block['changed_pixels'])), flush=True)
    return record


def add_rcwa(records, settings, rcwa_python):
    """One TORCWA process for every record's binary design; its cases and run description go into each record."""
    _, g, _ = declared()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='g7-01-rcwa-') as directory:
        rcwa = run_rcwa(rcwa_job(g, settings, {f"seed{r['seed']}": r['binary'] for r in records}), rcwa_python, directory)
    seconds = time.perf_counter()-started
    run = dict(interpreter=rcwa_python, threads=os.environ.get('OMP_NUM_THREADS'), designs=[r['seed'] for r in records],
               **{k: v for k, v in rcwa.items() if k != 'cases'})
    for record in records:
        record.update(rcwa=[c for c in rcwa['cases'] if c['design'] == f"seed{record['seed']}"], rcwa_run=run)
        record['wall_seconds']['rcwa_process'] = seconds
    return records


def rcwa_stage(settings, out, *, rcwa_python):
    """The TORCWA check of every seed record in `out` in one process, written back into the records."""
    paths = [Path(out)/f'seed{seed}.json' for seed in settings.seeds]
    records = add_rcwa([load_json(path)[0] for path in paths], settings, rcwa_python)
    for path, record in zip(paths, records):
        write_json(path, record)
    return records


def judge_stage(settings, out, *, selection=None):
    """Judge every criterion from the seed and baseline records in `out` and write summary.json."""
    case, g, provenance = declared()
    records = [load_json(Path(out)/f'seed{seed}.json')[0] for seed in settings.seeds]
    baseline = load_json(Path(out)/BASELINE)[0] if (Path(out)/BASELINE).exists() else None
    criteria = judge(records, case, settings, g, baseline)
    summary = dict(
        schema='torchfdtd-g7-01-summary-v3', task='G7-01', date=time.strftime('%Y-%m-%d'), declaration='docs/G7_WORKFLOWS.md',
        provenance=provenance, settings=asdict(settings), selection=selection, environment=environment(torch.device('cpu')),
        seed_environments={r['seed']: r['environment'] for r in records},
        baseline=None if baseline is None else {k: baseline[k] for k in ('design', 'band_mean_T_plus1', 'declared_band_mean_T_plus1', 'environment')},
        rcwa={r['seed']: r.get('rcwa_run') for r in records}, wall_seconds={r['seed']: r['wall_seconds'] for r in records},
        seeds={r['seed']: dict(band_mean_T_plus1_TE_normal_design_mesh=criteria['performance']['per_seed'][r['seed']],
                               objective_binary_design_mesh=r['design_mesh_binary']['metrics'], violations=r['fabrication']['violations'],
                               feature_sizes=r['fabrication']['feature_sizes'], binary=r['binary'],
                               open_close=None if 'open_close' not in r else {k: r['open_close'].get(k) for k in ('changed_pixels', 'band_mean_T_plus1')},
                               robust=None if 'robust' not in r else {k: r['robust'][k] for k in ('threshold_shift', 'feature_sizes')})
               for r in records},
        references={r['seed']: r['references'] for r in records[:1]},
        conventions=dict(
            epsilon='arithmetic mean of the permittivity over a cell-sized box at each Yee component; pixel k centred on the Ez node '
                    'x = -1 + 0.02 k um',
            cell=f'the fixture cell with {settings.extension_um:g} um more substrate below and air above; absorbers of '
                 f'{settings.absorber_um:g} um at every mesh; design runs over {settings.physical_time_fs:g} fs, evaluations over '
                 f'{settings.evaluation_time_fs:g} fs',
            robust=(None if 'three_ridge_segments' in case['fixture'] else
                    'eroded, nominal and dilated tanh projections of the same filtered density at eta 0.5 + shift, 0.5 and 0.5 - shift; '
                    'the smallest of their mean T+1 at the design wavelengths is maximized; the judged design is the nominal one '
                    'thresholded at 0.5, then opened and closed'),
            constrained_segments=(None if 'three_ridge_segments' not in case['fixture'] else
                                  'six alternating ridge/gap runs of at least three pixels, initialized from the disclosed '
                                  'TORCWA-assisted development design with seed-specific logit jitter; six Adam steps '
                                  'on the 0.02 um/560 fs short objective, with the better initial or final hard design '
                                  'selected before 0.01/0.005 um/2240 fs evaluation'),
            efficiency='order branch power over the forward order-0 power at the reflection line of the bare-substrate run of the same case',
            amplitudes='|a|^2 = efficiency; TE amplitudes of E, TM amplitudes of H; exp(-i omega t); phase referenced to x = 0, to the '
                       'incident wave at y = 0.01 um, to y = 0.51 um for transmitted and y = 0.01 um for reflected orders',
            bloch=f'k_x = 2 pi n_sub sin({ANGLE_DEG:g} deg)/{ANGLE_WAVELENGTH_UM} um on both periodic faces; the incidence angle in the substrate '
                  'is asin(k_x lambda/(2 pi n_sub)) and follows the wavelength',
            energy_balance='sum over every propagating order in ALL_ORDERS (-3..3); at the Bloch wavevector R-2 propagates in the substrate over '
                           'the whole band and T+1 propagates in air below 1.511 um; declared_order_sum keeps orders -1, 0, +1 only'),
        criteria=criteria, all_passed=all(c['passed'] for c in criteria.values()))
    write_json(Path(out)/'summary.json', summary)
    for name, c in criteria.items():
        print(f"{name}: value {c['value']} {'pass' if c['passed'] else 'FAIL'}", flush=True)
    return summary


def run(settings, out, *, rcwa_python, selection=None, checkpoint_dir=None, skip_rcwa=False):
    """Every seed in turn, the baseline, one TORCWA process for all seeds, then the judgement (the declared run
    starts the seed stages in parallel instead)."""
    for seed in settings.seeds:
        seed_stage(settings, seed, out, rcwa_python=rcwa_python, selection=selection, skip_rcwa=True,
                   checkpoint=None if checkpoint_dir is None else Path(checkpoint_dir)/f'seed{seed}.pt')
    baseline_stage(settings, out)
    if not skip_rcwa:
        rcwa_stage(settings, out, rcwa_python=rcwa_python)
    return judge_stage(settings, out, selection=selection)


def declared_run(out):
    """The declared settings and the recorded threshold selection of the records directory `out`."""
    case, _, _ = declared()
    if 'three_ridge_segments' in case['fixture']:
        return Settings.declared(case, None), None
    path = Path(out)/SELECTION
    if not path.exists():
        raise FileNotFoundError(f'{path} is missing: run the threshold selection (stages select-run and select-decide) first')
    selection = load_json(path)[0]
    return Settings.declared(case, selection), selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True,
                        help='directory of the records: threshold-selection.json, seed<N>.json, baseline.json, summary.json')
    parser.add_argument('--stage', choices=('all', 'select-run', 'select-decide', 'seed', 'baseline', 'rcwa', 'judge'), default='all')
    parser.add_argument('--seed', type=int, help='the seed of a seed or select-run stage')
    parser.add_argument('--shift', type=float, help='the candidate threshold shift of a select-run stage')
    parser.add_argument('--selection-dir', type=Path, help='directory of the development selection records (outside the repository)')
    parser.add_argument('--small', action='store_true', help='CPU development size: coarser meshes, shorter runs, one seed, few iterations')
    parser.add_argument('--checkpoint-dir', type=Path, help='save the design state after every iteration and resume from it')
    parser.add_argument('--rcwa-python', default=os.environ.get('TORCHFDTD_RCWA_PYTHON', 'C:/anaconda3/python.exe'),
                        help='interpreter that holds TORCWA 0.1.4.2')
    parser.add_argument('--skip-rcwa', action='store_true', help='development only: no TORCWA check (criterion c then fails)')
    args = parser.parse_args(argv)
    if args.stage == 'select-run':
        return select_run(args.shift, args.seed, args.selection_dir)
    if args.stage == 'select-decide':
        selection = select_decide(args.selection_dir)
        write_json(args.out/SELECTION, selection)
        print(json.dumps({k: selection[k] for k in ('chosen_threshold_shift', 'chosen_by_fallback', 'rejected', 'pending')}), flush=True)
        return selection
    settings, selection = (Settings.small_run(), None) if args.small else declared_run(args.out)
    checkpoint = None if args.checkpoint_dir is None else args.checkpoint_dir/f'seed{args.seed}.pt'
    if args.stage == 'seed':
        return seed_stage(settings, args.seed, args.out, rcwa_python=args.rcwa_python, selection=selection, checkpoint=checkpoint,
                          skip_rcwa=args.skip_rcwa)
    if args.stage == 'baseline':
        return baseline_stage(settings, args.out)
    if args.stage == 'rcwa':
        return rcwa_stage(settings, args.out, rcwa_python=args.rcwa_python)
    if args.stage == 'judge':
        return judge_stage(settings, args.out, selection=selection)
    return run(settings, args.out, rcwa_python=args.rcwa_python, selection=selection, checkpoint_dir=args.checkpoint_dir,
               skip_rcwa=args.skip_rcwa)


if __name__ == '__main__':
    main()
