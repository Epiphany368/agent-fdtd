"""G7-02r2: refine the 2D silicon-ridge metalens from three deterministic starts and validate every final design.

The fixed declaration is docs/G7_WORKFLOWS.md (section G7-02 and its revision) and docs/validation/cases/G7-02r2.json.
The lens, grid, source and DFT lines are those of the Meep comparison (examples/meep_comparison/metalens/geometry.json);
the lens and line builders, the observables and the focusing-efficiency definition are reused from
torchfdtd_metalens.py and metalens_common.py, which compare.py applies to the comparison records.

Design model. Each ridge is a DifferentiableSolid box whose width is a Torch parameter; smooth_geometry_epsilon
rasterizes the 33 boxes with a fixed physical transition width (TRANSITION_UM) and its bounded shape VJP carries
the adjoint epsilon gradient back to the widths. The transition width is part of the design: every validation
run rasterizes the same regularized boxes, so the 0.0125 um run refines the mesh of one physical permittivity
profile. Widths are kept inside the library range after each Adam step.

Time rule (G7-02r2). Every forward run, of the lens and of the bare cell, lasts until the total field energy in the
cell falls below 1e-3 of its peak after the source has ended, checked every 20 steps and capped at 10 times the
declared 303.6 fs window (and at the Region limit of 100000 steps). The energy is the native StateDiagnostics measure,
sum over cells of volume * (E^2 / inverse permittivity + H^2) in float64; find_stop evaluates it on the adjoint's own
Yee system (torchfdtd.differentiable._System, the same kernels as the runs it times), because the native Simulation
and its RunControl shutoff accept structures only, not the regularized permittivity, and the differentiable models
run a fixed step count. The run of the found length then follows. In each refinement iteration a stop pass on the
current widths fixes the step count of that iteration's value and gradient; (d) runs lens and bare cell for twice
their stops, within the same cap.

Runs. The objective is |Ez|^2 at the declared focal point (x = 0, y = focal line) at 1.55 um, relative to the
bare-cell incident intensity, through DifferentiableSimulation with an online DFT. Validation runs use
DifferentiablePlaneSimulation forwards (no gradient) with the recorded incident, focal and axis lines, whose
interpolation and time convention are those of the native plane monitors; the staircase library design is also run
through the native Simulation to tie this path to the recorded comparison.

Stages. 'common' (the native tie, the native shutoff cross-check and the gradient check) and 'start' (one start's
refinement and validation) write common.json and start-<name>.json; 'summary' judges them into summary.json. run()
executes them in this process, or, given a launcher prefix such as a GPU lock command, as parallel processes.

    python examples/g7/metalens/metalens_workflow.py --output-dir docs/validation/g7/G7-02
    python examples/g7/metalens/metalens_workflow.py --stage start --start wider --output-dir docs/validation/g7/G7-02
    python examples/g7/metalens/metalens_workflow.py --reduced --backend cpu --iterations 1 --output-dir <dir>

The first form is the declared run (CUDA, float32); the judged test in tests/test_g7_metalens.py runs the same
function. --reduced builds a five-ridge lens on a coarse short grid with a looser stop for the fast tests; its numbers
have no meaning beyond exercising the same code. The repository-only modules (examples/, the comparison helpers) are
found by appending their directories to the end of sys.path, so an installed torchfdtd is never shadowed by this
checkout. Every record names the imported torchfdtd file, its version, whether it came from the installed
distribution or the checkout, the local wheel it was installed from with that file's SHA-256, and the commit of this
checkout. The judged run imports the installed wheel from a working directory outside the checkout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib import metadata
import json
import math
import platform
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LENS = REPO / 'examples' / 'meep_comparison' / 'metalens'
for path in (str(REPO), str(LENS)):
    if path not in sys.path:
        sys.path.append(path)
import metalens_common as mc  # noqa: E402
import torchfdtd_metalens as tm  # noqa: E402

import torch  # noqa: E402
import torchfdtd  # noqa: E402
from torchfdtd import (AdjointOptions, Boundaries, BoundaryFace, DifferentiablePlaneSimulation,  # noqa: E402
                       DifferentiableSimulation, Monitor, Project, Region, RunControl, Simulation)
from torchfdtd.angular_spectrum import propagate_points  # noqa: E402
from torchfdtd.differentiable import _System  # noqa: E402
from torchfdtd.differentiable_geometry import DifferentiableSolid, smooth_geometry_epsilon  # noqa: E402
from torchfdtd.run_control import source_end_time  # noqa: E402
from torchfdtd.solver import voxelize  # noqa: E402

CASE_PATH = REPO / 'docs' / 'validation' / 'cases' / 'G7-02r2.json'
DESIGN_PATH = LENS / 'design_2d.json'
STARTS = ('library', 'wider', 'narrower')
TRANSITION_UM = .05          # regularized-fill transition width: two cells of the design grid, four of the fine grid
CHECKPOINTS = 32             # device checkpoints of the adjoint replay
ANGULAR_PAD = 4              # zero padding of the aperture line for the angular spectrum
MAX_PML_CELLS = 50           # Region.pml_cells limit; thicker absorbers are set per face
MAX_SNAPSHOT_INTERVAL = 10000  # Region.snapshot_interval limit
MAX_STEPS = 100000           # Region.steps limit
SCHEMA = 'torchfdtd-g7-02r2-v1'
# The time rule of docs/validation/cases/G7-02r2.json (tests/test_g7_metalens.py checks that the text carries it).
STOP_RULE = dict(threshold=1e-3, check_steps=20, cap_factor=10)
# Numeric limits of the acceptance text of the case (tests/test_g7_metalens.py checks that the text carries them).
LIMITS = dict(efficiency_min=.5571, mesh_efficiency=.02, mesh_axis_peak_um=.1, time_efficiency=.005, propagation_relative_l2=.05, fwhm_um=1.76)


def load_case():
    return json.loads(CASE_PATH.read_text(encoding='utf-8'))


def library_widths():
    return [row['width_um'] for row in json.loads(DESIGN_PATH.read_text(encoding='utf-8'))['library']]


def start_widths(design_widths, library, start):
    """The three declared starts: the library design, and every ridge one library step wider or narrower where one exists."""
    library = list(library)
    indices = [int(np.argmin(np.abs(np.asarray(library) - w))) for w in design_widths]
    assert all(abs(library[i] - w) < 1e-9 for i, w in zip(indices, design_widths)), 'design widths must be library widths'
    step = {'library': 0, 'wider': 1, 'narrower': -1}[start]
    return [library[min(max(i + step, 0), len(library) - 1)] for i in indices]


# ----------------------------------------------------------------------------
# Specifications: the declared lens, its grid variants and the reduced test lens
# ----------------------------------------------------------------------------
def declared_spec():
    spec = mc.load_geometry('2d')
    spec['pml_um'] = spec['pml_cells'] * spec['mesh_um']
    spec['transition_um'] = TRANSITION_UM
    spec['window_steps'] = spec['steps']      # the declared 303.6 fs window of this grid, the unit of the cap
    spec['stop'] = dict(STOP_RULE)
    return spec


def reduced_spec(ridges=5, mesh=.1, margin=1.5, run_time_fs=150.):
    """A small lens with the declared structure (central ridges of the library design, one-cycle pulse) on a short coarse grid."""
    spec = declared_spec()
    pitch = spec['pitch_um']
    half = (ridges // 2) * pitch
    aperture = ridges * pitch
    pml = .5
    sx, sy = aperture + 2 * margin + 2 * pml, 6.
    y_min = -sy / 2
    bottom = y_min + pml + .5
    top = bottom + spec['ridge_height_um']
    incident = round(y_min + round((top + .125 - y_min) / mesh) * mesh, 6)
    focal = round(y_min + round((1.2 - y_min) / mesh) * mesh, 6)
    courant = spec['courant_factor'] / math.sqrt(2)
    dt = courant * mesh * 1e-6 / mc.C0
    wave = dict(spec['source']['waveform'], pulse_cycles=1)
    spec.update(mesh_um=mesh, cell_um=[sx, sy], pml_um=pml, pml_cells=int(round(pml / mesh)), courant_number=courant, dt_s=dt,
                steps=int(round(run_time_fs * 1e-15 / dt)), aperture_um=aperture, ridges=[r for r in spec['ridges'] if abs(r['x_um']) <= half + 1e-9],
                ridge_bottom_um=bottom, ridge_top_um=top, ridge_y_center_um=bottom + spec['ridge_height_um'] / 2,
                source=dict(spec['source'], y_um=round(y_min + pml + .25, 6), x_span_um=aperture, waveform=wave),
                monitors=dict(incident=dict(y_um=incident, x_span_um=aperture), focal=dict(y_um=focal, x_span_um=sx - 2 * pml),
                              axis=dict(x_um=0., y_range_um=[incident, sy / 2 - pml])))
    spec['run_time_fs'] = spec['steps'] * dt * 1e15
    spec['window_steps'] = spec['steps']
    spec['transition_um'] = 2 * mesh
    spec['stop'] = dict(threshold=3e-2, check_steps=20, cap_factor=3)   # looser, for test speed only
    spec['reduced'] = True
    return spec


def grid_variant(spec, *, mesh=None, time_factor=1.):
    """The same lens, lines and absorber thickness on another mesh and/or for a longer physical time."""
    out = copy.deepcopy(spec)
    mesh = spec['mesh_um'] if mesh is None else mesh
    ratio = spec['mesh_um'] / mesh
    out['mesh_um'] = mesh
    out['pml_cells'] = int(round(spec['pml_um'] / mesh))
    out['steps'] = int(round(spec['steps'] * ratio * time_factor))
    out['window_steps'] = int(round(spec['window_steps'] * ratio))
    out['dt_s'] = spec['courant_number'] * mesh * 1e-6 / mc.C0
    out['run_time_fs'] = out['steps'] * out['dt_s'] * 1e15
    assert abs(out['pml_cells'] * mesh - spec['pml_um']) < 1e-9 and abs(spec['steps'] * ratio * time_factor - out['steps']) < 1e-6
    return out


def with_steps(spec, steps):
    """The same grid run for a given step count."""
    out = copy.deepcopy(spec)
    out['steps'] = int(steps)
    out['run_time_fs'] = out['steps'] * spec['courant_number'] * spec['mesh_um'] * 1e-6 / mc.C0 * 1e15
    return out


def cap_steps(spec):
    """The cap of the time rule on this grid: cap_factor declared windows, within the Region step limit."""
    return min(spec['stop']['cap_factor'] * spec['window_steps'], MAX_STEPS)


def build(spec, *, backend, precision='float32', monitors='all'):
    """tm.build_2d without ridges; the absorber keeps its physical thickness when it needs more than 50 cells.

    The snapshot interval (unused by the adjoint path, at most 10000 in the Region model) is capped for long runs.
    """
    cells, steps = spec['pml_cells'], spec['steps']
    project = tm.build_2d(dict(spec, pml_cells=min(cells, MAX_PML_CELLS), steps=min(steps, MAX_SNAPSHOT_INTERVAL)), with_lens=False, backend=backend,
                          monitors=monitors)
    region = project.region.model_dump()
    region.update(precision=precision, steps=steps, snapshot_interval=min(steps, MAX_SNAPSHOT_INTERVAL))
    if cells > MAX_PML_CELLS:
        face = BoundaryFace(layers=cells)
        region['boundaries'] = Boundaries(x_min=face, x_max=face, y_min=face, y_max=face).model_dump()
    project = Project.model_validate(dict(project.model_dump(), region=Region(**region).model_dump()))
    assert project.region.steps == steps and all(project.region.pml_layers(axis, side) == cells for axis in (0, 1) for side in (0, 1))
    return project


def native_project(spec, steps, *, with_lens, backend, run_control=None):
    """tm.build_2d of the staircase lens or bare cell for any step count (snapshot interval capped), optionally with a RunControl."""
    project = tm.build_2d(with_steps(spec, min(steps, MAX_SNAPSHOT_INTERVAL)), with_lens=with_lens, backend=backend)
    region = project.region.model_dump()
    region.update(steps=steps, snapshot_interval=min(steps, MAX_SNAPSHOT_INTERVAL))
    if run_control is not None:
        region['run_control'] = run_control.model_dump()
    return Project.model_validate(dict(project.model_dump(), region=region))


def focal_point(spec):
    return (spec['monitors']['axis']['x_um'], spec['monitors']['focal']['y_um'], 0.)


def frequencies(spec):
    return mc.frequencies_hz(spec['frequencies'])


def centre_index(spec, wavelength=1.55):
    return spec['frequencies']['wavelengths_um'].index(wavelength)


# ----------------------------------------------------------------------------
# Geometry: ridge widths to a regularized Yee permittivity through the shape VJP
# ----------------------------------------------------------------------------
def ridge_solids(widths, spec):
    epsilon = spec['material']['index'] ** 2
    height, yc = spec['ridge_height_um'], spec['ridge_y_center_um']
    return [DifferentiableSolid.box((widths[i], height, 2.), epsilon=epsilon, center=(r['x_um'], yc, 0.))
            for i, r in enumerate(spec['ridges'])]


def epsilon_of(region, widths, spec):
    return smooth_geometry_epsilon(region, ridge_solids(widths, spec), background=1., width=spec['transition_um'])


def device_of(backend):
    return torch.device('cuda' if backend == 'cuda' else 'cpu')


def dtype_of(precision):
    return torch.float64 if precision == 'float64' else torch.float32


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


# ----------------------------------------------------------------------------
# The time rule
# ----------------------------------------------------------------------------
def find_stop(spec, epsilon, *, backend, precision='float32', record=False):
    """First check step after the source end at which the cell energy is at most threshold x its running peak.

    Advances the adjoint's Yee system for this epsilon in blocks of check_steps up to the cap. Returns the stop step
    (the cap when the threshold is not reached), whether it was reached, the relative energy there and, with record,
    the energy after every check.
    """
    rule, cap = spec['stop'], cap_steps(spec)
    project = build(with_steps(spec, cap), backend=backend, precision=precision, monitors='none')
    project.monitors = [Monitor(id='focus', name='focus', component='Ez', center=focal_point(spec))]
    dt = project.region.time_step
    source_end = source_end_time(project)
    device = device_of(backend)
    started = time.perf_counter()
    energies, peak, stop, energy = [], 0., None, None
    with torch.no_grad():
        system = _System(project, epsilon)
        weight = system.eps4.double()
        for begin in range(0, cap, rule['check_steps']):
            end = min(begin + rule['check_steps'], cap)
            system.advance(begin, end)
            energy = float((system.grid.E.double().square() * weight).sum() + system.grid.H.double().square().sum())
            peak = max(peak, energy)
            if record:
                energies.append((end, energy))
            if end * dt >= source_end and energy <= rule['threshold'] * peak:
                stop = end
                break
        synchronize(device)
        del system, weight
    steps = stop if stop is not None else cap
    out = dict(steps=steps, reached=stop is not None, relative_energy=energy / peak, threshold=rule['threshold'], check_steps=rule['check_steps'],
               cap_steps=cap, cap_limited_by_region=rule['cap_factor'] * spec['window_steps'] > MAX_STEPS, window_factor=steps / spec['window_steps'],
               time_fs=steps * dt * 1e15, source_end_step=int(math.ceil(source_end / dt)), seconds=time.perf_counter() - started)
    if record:
        out['energy'] = energies
    return out


class Grid:
    """One mesh of the lens: the bare cell's stop, bare runs by step count and lens runs under the time rule."""

    def __init__(self, spec, *, backend, log=print):
        self.spec, self.backend, self.log = spec, backend, log
        self.device = device_of(backend)
        self.region = build(spec, backend=backend, monitors='none').region
        self.ones = torch.ones((*self.region.shape, 3), device=self.device, dtype=torch.float32)
        self.cap = cap_steps(spec)
        self.bare_stop = find_stop(spec, self.ones, backend=backend)
        self._bare = {}
        log(f"  grid {spec['mesh_um']} um: bare-cell stop {self.bare_stop['steps']} steps ({self.bare_stop['window_factor']:.2f} x)")

    def lines(self, steps):
        return LineModel(with_steps(self.spec, steps), backend=self.backend)

    def bare(self, steps):
        if steps not in self._bare:
            self._bare[steps] = self.lines(steps).run(self.ones)[0]
        return self._bare[steps]

    def epsilon(self, widths):
        with torch.no_grad():
            return epsilon_of(self.region, torch.as_tensor(np.asarray(widths), device=self.device, dtype=torch.float32), self.spec)

    def evaluate(self, epsilon, *, stop=None, factor=1):
        """Lens and bare cell each for factor x its own stop (capped); the lens stop is found unless given."""
        stop = stop if stop is not None else find_stop(self.spec, epsilon, backend=self.backend)
        steps, bare_steps = min(factor * stop['steps'], self.cap), min(factor * self.bare_stop['steps'], self.cap)
        planes, seconds = self.lines(steps).run(epsilon)
        bare = self.bare(bare_steps)
        obs = line_observables(planes, bare, with_steps(self.spec, steps))
        return dict(planes=planes, bare=bare, obs=obs, stop=stop, steps=steps, bare_steps=bare_steps, seconds=seconds)


def incident_intensity(bare, spec):
    k = centre_index(spec)
    f, c, _, _, _ = tm.plane_arrays(bare.field_monitor('incident'))
    return float(np.mean(np.abs(f[k, :, c.index('Ez')]) ** 2))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class FocalObjective:
    """|Ez|^2 at the declared focal point at 1.55 um through the discrete adjoint, relative to the bare incident intensity."""

    def __init__(self, spec, *, backend, precision='float32', checkpoints=CHECKPOINTS):
        project = build(spec, backend=backend, precision=precision, monitors='none')
        project.monitors = [Monitor(id='focus', name='focus', component='Ez', center=focal_point(spec))]
        self.model = DifferentiableSimulation(project, AdjointOptions(checkpoints=checkpoints))
        self.region = self.model.project.region
        self.spec = spec
        self.device, self.dtype = device_of(backend), dtype_of(precision)
        self.frequency = [frequencies(spec)[centre_index(spec)]]
        self.reference = None   # bare incident intensity at 1.55 um, set by the caller

    def __call__(self, widths):
        epsilon = epsilon_of(self.region, widths, self.spec)
        field = self.model.spectrum(epsilon, self.frequency).fields[0, 0]
        value = field.abs().square()
        return value / self.reference if self.reference is not None else value

    def value_and_gradient(self, widths):
        w = widths.detach().clone().requires_grad_(True)
        value = self(w)
        value.backward()
        return float(value.detach()), w.grad.detach().clone()


class _Planes:
    """Adapter from DifferentiablePlaneResult to the native field-monitor dicts that tm.observables_2d reads."""

    def __init__(self, planes, model):
        self.planes = planes
        self.plans = {identifier: plan for identifier, _, plan, _ in model.plans}

    def field_monitor(self, name):
        plane, plan = self.planes[name], self.plans[name]
        return dict(fields=plane.fields.detach().cpu().numpy().astype(np.complex128), components=list(plane.components),
                    points_um=np.asarray(plan['points_um']), weights=np.asarray(plan['weights']), shape=list(plan['shape']))


class LineModel:
    """The recorded incident, focal and axis lines of one grid and step count, for any epsilon (forward only)."""

    def __init__(self, spec, *, backend, precision='float32'):
        self.spec = spec
        self.project = build(spec, backend=backend, precision=precision, monitors='all')
        self.model = DifferentiablePlaneSimulation(self.project, AdjointOptions(checkpoints=0))
        self.region = self.model.project.region
        self.device, self.dtype = device_of(backend), dtype_of(precision)

    def run(self, epsilon):
        synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            planes = self.model(epsilon, frequencies(self.spec))
        synchronize(self.device)
        return _Planes(planes, self.model), time.perf_counter() - started

    def bare(self):
        epsilon = torch.ones((*self.region.shape, 3), device=self.device, dtype=self.dtype)
        return self.run(epsilon)

    def lens(self, widths):
        with torch.no_grad():
            epsilon = epsilon_of(self.region, torch.as_tensor(widths, device=self.device, dtype=self.dtype), self.spec)
        return self.run(epsilon)


# ----------------------------------------------------------------------------
# Observables
# ----------------------------------------------------------------------------
def side_lobe_ratio(x, intensity):
    """Largest focal-line intensity outside the main lobe (bounded by the first minima on each side of the peak) over the peak.

    The descent from the peak continues through equal samples, so a peak shared by two samples stays inside the main lobe.
    """
    values = np.asarray(intensity, dtype=np.float64)
    i = int(np.argmax(values))
    left = i
    while left > 0 and values[left - 1] <= values[left]:
        left -= 1
    right = i
    while right < len(values) - 1 and values[right + 1] <= values[right]:
        right += 1
    outside = np.r_[values[:left], values[right + 1:]]
    return dict(ratio=float(outside.max() / values[i]) if len(outside) else 0., main_lobe_um=[float(x[left]), float(x[right])])


def line_observables(lens, bare, spec):
    """tm.observables_2d (the comparison's reductions) plus the side-lobe ratio, at every wavelength."""
    obs = tm.observables_2d(lens, bare, spec)
    for k, row in enumerate(obs['summary']):
        row['side_lobe'] = side_lobe_ratio(obs['focal']['x_um'], np.asarray(obs['focal']['intensity'])[k])
    return obs


def centre_row(obs, spec):
    return obs['summary'][centre_index(spec)]


def compact_lines(obs, spec):
    """Focal-line and axial intensity at 1.55 um only (the full-band arrays are summarized, not stored)."""
    k = centre_index(spec)
    return dict(focal_x_um=obs['focal']['x_um'], focal_intensity=obs['focal']['intensity'][k], focal_poynting_y=obs['focal']['poynting_y'][k],
                axis_y_um=obs['axis']['y_um'], axis_intensity=obs['axis']['intensity'][k])


def wide_line_spec(spec):
    """The same grid with the aperture line widened to the whole interior (information: the effect of truncating the line)."""
    out = copy.deepcopy(spec)
    out['monitors']['incident']['x_span_um'] = spec['monitors']['focal']['x_span_um']
    return out


def angular_spectrum_check(lens, bare, obs, spec):
    """Propagate the lens aperture-line Ez to the focal and axis lines; bare and obs are those of the recorded lines."""
    k = centre_index(spec)
    plane = lens.planes['incident']
    device = plane.fields.device
    incident = incident_intensity(bare, spec)
    focal_x = np.asarray(obs['focal']['x_um'])
    axis_y = np.asarray(obs['axis']['y_um'])
    focal_points = np.stack([focal_x, np.full_like(focal_x, spec['monitors']['focal']['y_um']), np.zeros_like(focal_x)], -1)
    axis_points = np.stack([np.full_like(axis_y, spec['monitors']['axis']['x_um']), axis_y, np.zeros_like(axis_y)], -1)
    started = time.perf_counter()
    with torch.no_grad():
        values = {}
        for name, points in (('focal', focal_points), ('axis', axis_points)):
            result = propagate_points(plane, torch.as_tensor(points, device=device), components=('Ez',), pad=ANGULAR_PAD, point_chunk=64)
            values[name] = (result.fields[k, :, 0].abs().square().double().cpu().numpy() / incident)
    seconds = time.perf_counter() - started
    direct_focal = np.asarray(obs['focal']['intensity'])[k]
    direct_axis = np.asarray(obs['axis']['intensity'])[k]
    def rel_l2(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))
    y_prop, _ = mc.peak_parabolic(axis_y, values['axis'])
    y_direct, _ = mc.peak_parabolic(axis_y, direct_axis)
    return dict(method=f'angular spectrum (torchfdtd.angular_spectrum.propagate_points) of the complex Ez on the aperture line y = '
                       f"{spec['monitors']['incident']['y_um']} um (the recorded incident line, {spec['monitors']['incident']['x_span_um']} um wide) "
                       f'from the design-grid lens run, zero padding {ANGULAR_PAD}, air; intensities relative to the bare incident intensity',
                wavelength_um=spec['frequencies']['wavelengths_um'][k], pad=ANGULAR_PAD, seconds=seconds,
                focal_relative_l2=rel_l2(values['focal'], direct_focal), axis_relative_l2=rel_l2(values['axis'], direct_axis),
                axis_peak_y_um=dict(propagated=y_prop, direct=y_direct, difference=abs(y_prop - y_direct)),
                focal_peak=dict(propagated=float(values['focal'].max()), direct=float(direct_focal.max())),
                propagated_focal_intensity=mc.as_list(values['focal']), propagated_axis_intensity=mc.as_list(values['axis']))


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------
def environment(device, output_dir=None):
    """Host, packages and provenance; tracked_changes ignores the record directory, which a run rewrites."""
    try:
        exclude = []
        if output_dir is not None and REPO in Path(output_dir).resolve().parents:
            exclude = [f':(exclude){Path(output_dir).resolve().relative_to(REPO).as_posix()}']
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no', '--', '.', *exclude], cwd=REPO, capture_output=True,
                                    text=True, timeout=30, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__, platform=platform.platform(),
                device=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or 'cpu',
                gpu=mc.nvidia_smi() if device.type == 'cuda' else None, **provenance(commit, dirty))


def provenance(commit, dirty):
    """Which torchfdtd ran and which checkout held this example; copied into every record.

    torchfdtd_import is 'installed' when the package was imported from the installed distribution's directory,
    'checkout' when from this checkout and 'other' otherwise (an editable install of another checkout, for example).
    """
    try:
        dist = metadata.distribution('torchfdtd')
    except metadata.PackageNotFoundError:
        dist = None
    imported = Path(torchfdtd.__file__).resolve()
    installed = dist is not None and Path(dist.locate_file('')).resolve() in imported.parents
    return dict(torchfdtd_file=str(imported), torchfdtd_version=getattr(torchfdtd, '__version__', None) or (dist.version if dist else None),
                torchfdtd_version_source=('torchfdtd.__version__' if getattr(torchfdtd, '__version__', None) else
                                          'importlib.metadata.version("torchfdtd"): the installed distribution (the package defines no __version__)'),
                checkout_import=REPO in imported.parents,
                torchfdtd_import='checkout' if REPO in imported.parents else 'installed' if installed else 'other',
                torchfdtd_wheel=installed_wheel(dist), commit=commit, tracked_changes=dirty)


def installed_wheel(dist):
    """The local wheel file pip installed the distribution from (its direct_url.json) and that file's SHA-256 now; None otherwise."""
    text = dist.read_text('direct_url.json') if dist is not None else None
    info = json.loads(text) if text else {}
    if 'archive_info' not in info or not info.get('url', '').startswith('file:'):
        return None
    path = Path(urllib.request.url2pathname(urllib.parse.urlparse(info['url']).path))
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return dict(url=info['url'], path=str(path), sha256=sha256)


PROVENANCE_KEYS = ('torchfdtd_file', 'torchfdtd_version', 'torchfdtd_version_source', 'checkout_import', 'torchfdtd_import', 'torchfdtd_wheel',
                   'commit', 'tracked_changes')


def design_summary(row):
    keys = ('axis_peak_y_um', 'axis_peak_intensity', 'focal_plane_peak_x_um', 'focal_plane_peak_intensity', 'fwhm_um', 'efficiency', 'transmission')
    out = {k: row[k] for k in keys}
    out['side_lobe_ratio'] = row['side_lobe']['ratio']
    return out


def stop_summary(evaluation):
    stop = evaluation['stop']
    return dict(lens_stop={k: stop[k] for k in ('steps', 'reached', 'relative_energy', 'window_factor', 'time_fs', 'seconds')},
                lens_steps=evaluation['steps'], bare_steps=evaluation['bare_steps'])


# ----------------------------------------------------------------------------
# Refinement and checks
# ----------------------------------------------------------------------------
def refine(grid, widths, *, reference, iterations, learning_rate, bounds, log=print):
    """Adam on the ridge widths (um), maximizing the objective; each iteration first finds the stop of the current design.

    Widths are clipped to the library range after each step.
    """
    spec, backend = grid.spec, grid.backend
    parameters = torch.nn.Parameter(torch.as_tensor(widths, device=grid.device, dtype=torch.float32).clone())
    optimizer = torch.optim.Adam([parameters], lr=learning_rate)
    history = []
    for iteration in range(iterations):
        synchronize(grid.device)
        started = time.perf_counter()
        stop = find_stop(spec, grid.epsilon(parameters.detach().cpu().numpy()), backend=backend)
        objective = FocalObjective(with_steps(spec, stop['steps']), backend=backend)
        objective.reference = reference
        optimizer.zero_grad()
        value = objective(parameters)
        (-value).backward()
        gradient = parameters.grad.detach().cpu().double().numpy()
        before = parameters.detach().cpu().double().numpy()
        optimizer.step()
        with torch.no_grad():
            parameters.clamp_(*bounds)
        synchronize(grid.device)
        history.append(dict(iteration=iteration, objective=float(value.detach()), steps=stop['steps'], stop_reached=stop['reached'],
                            stop_relative_energy=stop['relative_energy'], widths_um=mc.as_list(before, 9), gradient_per_um=mc.as_list(gradient, 7),
                            stop_seconds=stop['seconds'], seconds=time.perf_counter() - started))
        log(f"  iteration {iteration}: focal intensity {history[-1]['objective']:.5f} at {stop['steps']} steps ({history[-1]['seconds']:.1f} s)")
        del objective, value
    return parameters.detach(), history


def finite_difference_check(objective, widths, *, steps):
    """Adjoint derivative of the objective against central differences for the ridge with the largest adjoint derivative.

    The cubic transition makes the central difference carry a truncation error of order (step / transition)^2, and float32
    rounding of the objective grows as the step shrinks, so several steps are reported.
    """
    base = torch.as_tensor(widths, device=objective.device, dtype=objective.dtype)
    value, gradient = objective.value_and_gradient(base)
    ridge = int(torch.argmax(gradient.abs()))
    adjoint = float(gradient[ridge])
    rows = []
    for step in steps:
        values = []
        for sign in (1., -1.):
            shifted = base.clone()
            shifted[ridge] += sign * step
            with torch.no_grad():
                values.append(float(objective(shifted)))
        central = (values[0] - values[1]) / (2 * step)
        rows.append(dict(step_um=step, central_difference_per_um=central, relative_difference=abs(adjoint - central) / abs(central)))
    return dict(ridge=ridge, x_um=objective.spec['ridges'][ridge]['x_um'], width_um=float(base[ridge]), objective=value, adjoint_per_um=adjoint,
                precision=str(objective.dtype).replace('torch.', ''), steps=objective.spec['steps'], central_differences=rows)


def fine_spec(spec, case):
    return grid_variant(spec, mesh=round(case['fixture']['validation']['fine_mesh_um'] * spec['mesh_um'] / case['fixture']['mesh_um'], 12))


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def run_common(spec, *, backend='cuda', output_dir=None, fd_steps=(.004, .002, .001), log=print):
    """The native tie, the native shutoff cross-check of the stop rule and the gradient check; writes common.json."""
    case = load_case()
    device = device_of(backend)
    env = environment(device, output_dir)
    started = time.perf_counter()
    grid = Grid(spec, backend=backend, log=log)
    staircase = torch.as_tensor(voxelize(tm.build_2d(spec, with_lens=True, backend=backend))[0], device=device, dtype=torch.float32)
    stop = find_stop(spec, staircase, backend=backend)
    log(f"library staircase: stop {stop['steps']} steps; native Simulation and DifferentiablePlaneSimulation")
    plane = grid.evaluate(staircase, stop=stop)
    native_lens = Simulation(native_project(spec, plane['steps'], with_lens=True, backend=backend)).run()
    native_bare = Simulation(native_project(spec, plane['bare_steps'], with_lens=False, backend=backend)).run()
    native_row = centre_row(line_observables(native_lens, native_bare, with_steps(spec, plane['steps'])), spec)
    plane_row = centre_row(plane['obs'], spec)
    consistency = dict(note='staircase library design (voxelize) on the design grid at 1.55 um, lens and bare cell each run to its 1e-3 stop: '
                            'native Simulation against the DifferentiablePlaneSimulation forward used for every validation',
                       lens_steps=plane['steps'], bare_steps=plane['bare_steps'], native=design_summary(native_row), plane_forward=design_summary(plane_row),
                       efficiency_difference=abs(native_row['efficiency'] - plane_row['efficiency']),
                       axis_peak_difference_um=abs(native_row['axis_peak_y_um'] - plane_row['axis_peak_y_um']),
                       recorded_comparison_efficiency=case['baseline']['focusing_efficiency'])
    log(f"  native efficiency {native_row['efficiency']:.6f}, plane forward {plane_row['efficiency']:.6f}")
    # The native RunControl shutoff on the same staircase lens (peak frozen at the source end, two consecutive checks).
    rule = spec['stop']
    control = RunControl(auto_shutoff=True, decay_threshold=rule['threshold'], check_interval=rule['check_steps'], consecutive_checks=2)
    native = Simulation(native_project(spec, grid.cap, with_lens=True, backend=backend, run_control=control)).run()
    shutoff = dict(note='native Simulation with RunControl(auto_shutoff=True, consecutive_checks=2) on the staircase lens; find_stop uses one check '
                        'and the running peak', native_steps=native.summary['steps'], termination_reason=native.summary.get('termination_reason'),
                   find_stop_steps=stop['steps'], find_stop_relative_energy=stop['relative_energy'],
                   native_last_check=native.summary['diagnostics'][-1] if native.summary.get('diagnostics') else None)
    log(f"  native shutoff at {shutoff['native_steps']} steps, find_stop at {stop['steps']}")
    log('finite-difference check of the shape gradient (library start)')
    library = start_widths([r['width_um'] for r in spec['ridges']], library_widths(), 'library')
    start_stop = find_stop(spec, grid.epsilon(library), backend=backend)
    objective = FocalObjective(with_steps(spec, start_stop['steps']), backend=backend)
    objective.reference = incident_intensity(grid.bare(grid.bare_stop['steps']), spec)
    gradient_check = finite_difference_check(objective, library, steps=fd_steps)
    log(f"  ridge {gradient_check['ridge']}: adjoint {gradient_check['adjoint_per_um']:.6g}, central "
        + ', '.join(f"{r['central_difference_per_um']:.6g} (step {r['step_um']})" for r in gradient_check['central_differences']))
    result = finite_or_none(dict(schema=SCHEMA, case_id='G7-02r2', stage='common', reduced=bool(spec.get('reduced')), environment=env,
                                 bare_stop=grid.bare_stop, staircase_stop=stop, consistency=consistency, native_shutoff=shutoff,
                                 gradient_check=gradient_check, seconds=time.perf_counter() - started))
    if output_dir is not None:
        write_json(Path(output_dir) / 'common.json', result)
    return result


def run_start(spec, start, *, backend='cuda', iterations=None, learning_rate=None, output_dir=None, log=print):
    """One start: refinement under the time rule, then the validation of its final design; writes start-<name>.json."""
    case = load_case()
    refinement = case['fixture']['refinement']
    iterations = refinement['iterations'] if iterations is None else iterations
    learning_rate = refinement['learning_rate_um'] if learning_rate is None else learning_rate
    device = device_of(backend)
    env = environment(device, output_dir)   # the commit and tree state the run loaded, before any later edit
    started = time.perf_counter()
    library = library_widths()
    bounds = (min(library), max(library))
    initial = start_widths([r['width_um'] for r in spec['ridges']], library, start)
    grids = dict(design_grid=Grid(spec, backend=backend, log=log), fine_mesh=Grid(fine_spec(spec, case), backend=backend, log=log))
    design = grids['design_grid']
    reference = incident_intensity(design.bare(design.bare_stop['steps']), spec)
    log(f'start {start}')
    t0 = time.perf_counter()
    final, history = refine(design, initial, reference=reference, iterations=iterations, learning_rate=learning_rate, bounds=bounds, log=log)
    refine_seconds = time.perf_counter() - t0
    final_widths = final.cpu().double().numpy()
    t0 = time.perf_counter()
    evaluations = dict(design_grid=design.evaluate(design.epsilon(final_widths)))
    evaluations['longer_time'] = design.evaluate(design.epsilon(final_widths), stop=evaluations['design_grid']['stop'], factor=2)
    evaluations['fine_mesh'] = grids['fine_mesh'].evaluate(grids['fine_mesh'].epsilon(final_widths))
    for name, e in evaluations.items():
        row = centre_row(e['obs'], spec)
        log(f"  {name}: efficiency {row['efficiency']:.5f}, axis peak {row['axis_peak_y_um']:.4f} um, lens {e['steps']} / bare {e['bare_steps']} steps")
    objective = FocalObjective(with_steps(spec, evaluations['design_grid']['steps']), backend=backend)
    objective.reference = reference
    with torch.no_grad():
        final_objective = float(objective(torch.as_tensor(final_widths, device=device, dtype=torch.float32)))
    t = evaluations['design_grid']
    propagation = angular_spectrum_check(t['planes'], t['bare'], t['obs'], spec)
    wide_spec = wide_line_spec(with_steps(spec, t['steps']))
    wide_planes, _ = LineModel(wide_spec, backend=backend).run(design.epsilon(final_widths))
    propagation['full_width_line'] = angular_spectrum_check(wide_planes, t['bare'], t['obs'], wide_spec)
    for key in ('propagated_focal_intensity', 'propagated_axis_intensity'):
        propagation['full_width_line'].pop(key)
    propagation['full_width_line']['note'] = 'information only: the aperture line widened to the interior, direct lines and normalization unchanged'
    log(f"  angular spectrum: focal-line relative L2 {propagation['focal_relative_l2']:.4f} "
        f"(full-width line {propagation['full_width_line']['focal_relative_l2']:.4f})")
    initial_eval = design.evaluate(design.epsilon(initial))
    rows = {name: centre_row(e['obs'], spec) for name, e in evaluations.items()}
    record = dict(schema=SCHEMA, case_id='G7-02r2', start=start, reduced=bool(spec.get('reduced')), provenance={k: env[k] for k in PROVENANCE_KEYS},
                  device=env['device'], iterations=iterations, learning_rate_um=learning_rate, width_bounds_um=list(bounds),
                  transition_um=spec['transition_um'], stop_rule=spec['stop'],
                  bare_stops={name: g.bare_stop for name, g in grids.items()}, objective_reference_incident_intensity=reference,
                  ridges_x_um=[r['x_um'] for r in spec['ridges']], initial_widths_um=list(initial), final_widths_um=mc.as_list(final_widths, 9),
                  widths_at_bounds=int(np.sum((final_widths <= bounds[0] + 1e-7) | (final_widths >= bounds[1] - 1e-7))),
                  history=history, final_objective=final_objective,
                  initial_design=dict(design_grid=design_summary(centre_row(initial_eval['obs'], spec)), **stop_summary(initial_eval)),
                  final_design={name: design_summary(row) for name, row in rows.items()},
                  final_stops={name: stop_summary(e) for name, e in evaluations.items()},
                  final_design_all_wavelengths={name: dict(summary=e['obs']['summary'], transmission=e['obs']['incident']['transmission'])
                                                for name, e in evaluations.items()},
                  checks=dict(mesh_efficiency_change=abs(rows['fine_mesh']['efficiency'] - rows['design_grid']['efficiency']),
                              mesh_axis_peak_change_um=abs(rows['fine_mesh']['axis_peak_y_um'] - rows['design_grid']['axis_peak_y_um']),
                              time_efficiency_change=abs(rows['longer_time']['efficiency'] - rows['design_grid']['efficiency']),
                              propagation_focal_relative_l2=propagation['focal_relative_l2']),
                  propagation=propagation, lines_1550={name: compact_lines(e['obs'], spec) for name, e in evaluations.items()},
                  seconds=dict(refinement=refine_seconds, validation=time.perf_counter() - t0, total=time.perf_counter() - started))
    record = finite_or_none(record)
    if output_dir is not None:
        write_json(Path(output_dir) / f'start-{start}.json', record)
    return record


def summarize(spec, output_dir, *, starts=STARTS, common=None, records=None):
    """Judge the stage records into summary.json."""
    case = load_case()
    output_dir = Path(output_dir) if output_dir is not None else None
    if common is None:
        common = json.loads((output_dir / 'common.json').read_text(encoding='utf-8'))
    if records is None:
        records = [json.loads((output_dir / f'start-{s}.json').read_text(encoding='utf-8')) for s in starts]
    summary = judge(records, case, spec)
    summary.update(schema=SCHEMA, case_id='G7-02r2', reduced=bool(spec.get('reduced')), date=time.strftime('%Y-%m-%d'),
                   environment=environment(torch.device('cpu'), output_dir),
                   design=dict(transition_um=spec['transition_um'], width_bounds_um=records[0]['width_bounds_um'], checkpoints=CHECKPOINTS,
                               optimizer='torch.optim.Adam, default betas', stop_rule=spec['stop'],
                               objective=f'|Ez|^2 at (x, y) = {focal_point(spec)[:2]} um, 1.55 um, relative to the bare-cell mean incident |Ez|^2 '
                                         f"({records[0]['objective_reference_incident_intensity']:.6g} reduced units) through DifferentiableSimulation.spectrum, "
                                         'run for the stop of the current design'),
                   grids=dict(design_grid=dict(mesh_um=spec['mesh_um'], window_steps=spec['window_steps'], cap_steps=cap_steps(spec)),
                              fine_mesh=dict(mesh_um=fine_spec(spec, case)['mesh_um'], window_steps=fine_spec(spec, case)['window_steps'],
                                             cap_steps=cap_steps(fine_spec(spec, case)))),
                   stops={r['start']: r['final_stops'] for r in records},
                   common='common.json', consistency=common['consistency'], native_shutoff=common['native_shutoff'], gradient_check=common['gradient_check'],
                   stage_provenance=dict(common={k: common['environment'][k] for k in PROVENANCE_KEYS}, **{r['start']: r['provenance'] for r in records}),
                   wall_seconds=dict(common=common['seconds'], **{f"start_{r['start']}": r['seconds']['total'] for r in records}),
                   records=[f"start-{r['start']}.json" for r in records])
    summary = finite_or_none(summary)
    if output_dir is not None:
        write_json(output_dir / 'summary.json', summary)
    return summary


def run(*, spec, backend='cuda', iterations=None, learning_rate=None, starts=STARTS, output_dir=None, launcher=None, log=print):
    """All stages: in this process, or with launcher (a command prefix such as a GPU lock) as parallel processes."""
    if launcher is None:
        common = run_common(spec, backend=backend, output_dir=output_dir, log=log)
        records = [run_start(spec, s, backend=backend, iterations=iterations, learning_rate=learning_rate, output_dir=output_dir, log=log) for s in starts]
        return summarize(spec, output_dir, starts=starts, common=common, records=records), records
    if output_dir is None or spec.get('reduced'):
        raise ValueError('parallel stages need an output directory and the declared lens')
    prefix = shlex.split(launcher) if isinstance(launcher, str) else list(launcher)
    base = [*prefix, sys.executable, str(Path(__file__).resolve()), '--backend', backend, '--output-dir', str(output_dir)]
    if iterations is not None:
        base += ['--iterations', str(iterations)]
    commands = [base + ['--stage', 'common']] + [base + ['--stage', 'start', '--start', s] for s in starts]
    processes = [subprocess.Popen(command) for command in commands]
    codes = [p.wait() for p in processes]
    if any(codes):
        raise RuntimeError(f'stage processes failed with exit codes {codes}: {commands}')
    records = [json.loads((Path(output_dir) / f'start-{s}.json').read_text(encoding='utf-8')) for s in starts]
    return summarize(spec, output_dir, starts=starts, records=records), records


def judge(records, case, spec):
    """Every declared criterion with its value, limit and verdict; (c) to (e) apply to every start's final design."""
    acc = case['acceptance']
    assert LIMITS['efficiency_min'] == case['baseline']['focusing_efficiency']
    limits = dict(efficiency=LIMITS['efficiency_min'], mesh_efficiency=LIMITS['mesh_efficiency'], mesh_axis_peak_um=LIMITS['mesh_axis_peak_um'],
                  time_efficiency=LIMITS['time_efficiency'], propagation=LIMITS['propagation_relative_l2'], fwhm_um=LIMITS['fwhm_um'])
    by_start = {r['start']: r for r in records}
    best = max(records, key=lambda r: -math.inf if r['final_design']['design_grid']['efficiency'] is None else r['final_design']['design_grid']['efficiency'])

    def at_most(value, limit):
        return value is not None and value <= limit

    def at_least(value, limit):
        return value is not None and value >= limit
    rows = [
        dict(id='a', criterion=acc['all_starts'], value=sorted(by_start), limit=sorted(STARTS), passed=sorted(by_start) == sorted(STARTS)),
        dict(id='b', criterion=acc['performance'], value=best['final_design']['design_grid']['efficiency'], best_start=best['start'], limit=limits['efficiency'],
             comparison='>=', passed=at_least(best['final_design']['design_grid']['efficiency'], limits['efficiency']),
             all_starts={s: r['final_design']['design_grid']['efficiency'] for s, r in by_start.items()}),
        dict(id='c', criterion=acc['mesh'], value=dict(efficiency={s: r['checks']['mesh_efficiency_change'] for s, r in by_start.items()},
                                                       axis_peak_um={s: r['checks']['mesh_axis_peak_change_um'] for s, r in by_start.items()}),
             limit=dict(efficiency=limits['mesh_efficiency'], axis_peak_um=limits['mesh_axis_peak_um']), comparison='<=',
             passed=all(at_most(r['checks']['mesh_efficiency_change'], limits['mesh_efficiency'])
                        and at_most(r['checks']['mesh_axis_peak_change_um'], limits['mesh_axis_peak_um']) for r in records)),
        dict(id='d', criterion=acc['time'], value={s: r['checks']['time_efficiency_change'] for s, r in by_start.items()}, limit=limits['time_efficiency'],
             comparison='<=', passed=all(at_most(r['checks']['time_efficiency_change'], limits['time_efficiency']) for r in records)),
        dict(id='e', criterion=acc['propagation'], value={s: r['checks']['propagation_focal_relative_l2'] for s, r in by_start.items()}, limit=limits['propagation'],
             comparison='<=', passed=all(at_most(r['checks']['propagation_focal_relative_l2'], limits['propagation']) for r in records)),
        dict(id='f', criterion=acc['psf'], value=best['final_design']['design_grid']['fwhm_um'], best_start=best['start'], limit=limits['fwhm_um'], comparison='<=',
             passed=at_most(best['final_design']['design_grid']['fwhm_um'], limits['fwhm_um']),
             all_starts={s: r['final_design']['design_grid']['fwhm_um'] for s, r in by_start.items()}),
        dict(id='scope', criterion=acc['scope'], value=f'every reported number comes from a full-aperture {len(spec["ridges"])}-ridge run; '
                                                         'the unit-cell library only supplies the start widths', passed=True),
    ]
    return dict(criteria=rows, all_pass=all(r['passed'] for r in rows), best_start=best['start'])


def finite_or_none(value):
    """Records hold JSON numbers only: a non-finite value (a profile that never crosses half maximum) is recorded as null and fails its criterion."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, indent=1, allow_nan=False) + '\n').encode('utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--reduced', action='store_true', help='five-ridge lens on a coarse short grid (tests only)')
    parser.add_argument('--iterations', type=int, default=None, help='Adam iterations (default: the declared 20)')
    parser.add_argument('--stage', choices=('all', 'common', 'start', 'summary'), default='all')
    parser.add_argument('--start', choices=STARTS, help='the start of --stage start')
    parser.add_argument('--launcher', default=None, help='with --stage all: command prefix that runs each stage as a parallel process')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args(argv)
    spec = reduced_spec() if args.reduced else declared_spec()
    tag = args.stage if args.stage != 'start' else f'start {args.start}'
    log = lambda *items: print(f'[{tag}]', *items, flush=True)  # noqa: E731
    if args.stage == 'common':
        return run_common(spec, backend=args.backend, output_dir=args.output_dir, log=log)
    if args.stage == 'start':
        if args.start is None:
            parser.error('--stage start needs --start')
        return run_start(spec, args.start, backend=args.backend, iterations=args.iterations, output_dir=args.output_dir, log=log)
    if args.stage == 'summary':
        summary = summarize(spec, args.output_dir)
    else:
        summary, _ = run(spec=spec, backend=args.backend, iterations=args.iterations, output_dir=args.output_dir, launcher=args.launcher, log=log)
    print(json.dumps(dict(all_pass=summary['all_pass'], criteria=[{k: r.get(k) for k in ('id', 'value', 'limit', 'passed')} for r in summary['criteria']]), indent=1))
    return summary


if __name__ == '__main__':
    main()
