"""G7-03: fabrication-aware density design of the offset-guide mode coupler from three declared seeds.

The device is the coupler of examples/design_mode_coupler.py: two slab guides of epsilon 4 in a
cladding of 2.25, 0.6 um wide and offset by 0.6 um, a fixed-mode port on each guide (ModeNetwork),
and its 1.2 by 2.0 um design box, here as 12 by 20 pixels of 0.1 um under a 180-degree rotation
symmetry. The declaration is docs/G7_WORKFLOWS.md (section G7-03) with the revised case
docs/validation/cases/G7-03r2.json; the fixed quantities below are copied from there. The filter
radius is chosen first, on development seeds only (develop(), select_radius()). For every seed the
workflow

  1. checks the adjoint derivative of |S21|^2 at 1.55 um with respect to the design logits against
     central finite differences at three pixels of the start design;
  2. maximizes |S21|^2 at 1.55 um at the 0.05 um mesh from logits 0.5 randn(seed);
  3. thresholds the design, measures its minimum linewidth and gap against the declared 0.4 um, and
     evaluates the design eroded and dilated by 0.1 um on a 0.05 um sub-pixel grid;
  4. exports the binary design as native rectangles and GDS, re-imports both, and evaluates the
     full 2 x 2 modal S matrix with phases at 1.50, 1.55 and 1.60 um for the binary density, the
     pre-export rectangles and the re-imported GDS polygons (1.50 um is held out, never optimized);
  5. evaluates the re-imported GDS design again at the 0.025 um mesh over the same physical time.

judge() evaluates the seven acceptance criteria of the case from the seed records. same_mesh_baseline() is a
non-judged diagnostic: G6's three coupler designs through the same GDS round trip and meshes. Every record
names the torchfdtd it imported (file, version, checkout or installed) and the repository commit.

Development selection (CUDA; one process per radius and seed, development seeds only):
    python -m examples.g7.coupler.workflow --output-dir docs/validation/g7/G7-03 --develop 0.4 --seeds 11
Judged run (CUDA, float32):
    python -m examples.g7.coupler.workflow --output-dir docs/validation/g7/G7-03 --baseline
    python -m examples.g7.coupler.workflow --output-dir docs/validation/g7/G7-03
Reduced mechanics run (CPU, a few minutes, never judged):
    python -m examples.g7.coupler.workflow --reduced --seeds 1 --output-dir <scratch>
"""
import argparse
import cmath
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
# examples/ ships with the repository, not with the wheel. Appending the root (never prepending it) keeps an
# installed torchfdtd in charge while the repository-only examples stay importable.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from examples import design_mode_coupler as base  # noqa: E402
import torchfdtd  # noqa: E402
from torchfdtd.design_parameterization import DensityParameterization  # noqa: E402
from torchfdtd.design_problem import Continuation, DesignProblem  # noqa: E402
from torchfdtd.fabrication import binary_structures, fabrication_perturbation, measure_feature_sizes  # noqa: E402
from torchfdtd.gds import export_gds, GDSLayer, import_gds  # noqa: E402

CASE = 'docs/validation/cases/G7-03r2.json'
SEEDS = (1, 2, 3)
DEVELOPMENT_SEEDS = (11, 12, 13)   # the filter radius is chosen on these only, never on SEEDS
WAVELENGTHS_UM = (1.5, 1.55, 1.6)
DESIGN_UM, HOLDOUT_UM = base.WAVELENGTH_UM, base.HOLDOUT_UM
PIXELS, PIXEL_UM = (12, 20), .1    # G7-03r2: 0.1 um pixels in the example's unchanged 1.2 by 2.0 um box
MIN_FEATURE_UM = .4                # declared minimum linewidth and minimum gap (4 pixels)
EROSION_UM = .1                    # declared erosion and dilation
SUBPIXEL = 2                       # the 0.1 um perturbation acts on a grid of 0.05 um, half a pixel
# Filter radii tried on the development seeds, ascending in one-pixel steps from the declared minimum feature;
# the first whose designs meet the rule for every development seed is the smallest, so larger ones need no run.
RADIUS_CANDIDATES_UM = (.4, .5, .6, .7, .8)
# (b) compares with the best G6 design through this pipeline at 0.05 um (the case's baseline value).
LIMITS = dict(performance=.6025, passivity=1.01, reciprocity=1e-3, gradient=.02, gds=.005, mesh=.02)


@dataclass(frozen=True)
class Settings:
    """Workflow parameters. mesh_um and check_mesh_um are declared; the others are workflow choices
    fixed before any run of the declared seeds. steps keep the physical time of G6's 500 steps at
    0.2 um (at 0.05 um, 2000 against 3000 steps changed |S21|^2 by 2.7e-4 on a random density). The
    filter radius is select_radius() of the development records in docs/validation/g7/G7-03/development:
    0.4 and 0.5 um left features below 4 pixels, 0.6 um met the rule for development seeds 11, 12, 13."""
    mesh_um: float = .05
    steps: int = 2000
    check_mesh_um: float = .025
    check_steps: int = 4000
    iterations: int = 50
    beta: float = 4.
    continuation_every: int = 10
    beta_maximum: float = 64.
    learning_rate: float = .1
    filter_radius_um: float | None = .6
    fd_step: float = .02
    fd_pixels: int = 3
    device: str = 'cuda'
    reduced: bool = False


# G6's mesh and physical time; exercises every stage on the CPU, judges nothing (any radius will do).
REDUCED = Settings(mesh_um=.2, steps=500, check_mesh_um=.1, check_steps=1000, iterations=2, filter_radius_um=.4, device='cpu',
                   reduced=True)


class Coupler(base.CouplerForward):
    """The two-port network of the coupler at one mesh, step count and wavelength, on one device.

    The design density stays wherever the parameterization lives and moves to the network device
    inside epsilon(), so the parameterization, fabrication checks and export remain on the CPU.
    """
    def __init__(self, mesh_um, steps, wavelength_um, device='cpu'):
        super().__init__(mesh_um, steps, wavelength_um)
        self.base = self.base.to(device)
        self.wavelength_um = wavelength_um

    def epsilon(self, density):
        return super().epsilon(density.to(self.base.device))

    @torch.no_grad()
    def s_density(self, density):
        return self.network(self.epsilon(density)).s

    @torch.no_grad()
    def s_structures(self, structures):
        return self.network(self.structure_epsilon(structures)).s


def s_record(s):
    """The 2 x 2 modal S[out, in] (port 1 = left guide, port 2 = right guide) with its checks.

    Each entry is recorded as real and imaginary part, power and phase (radians, at the port phase
    planes of ModeNetwork). column_power holds |S11|^2 + |S21|^2 and |S22|^2 + |S12|^2 (passivity),
    reciprocity is |S21 - S12|.
    """
    values = s.detach().cpu().to(torch.complex128).numpy()
    if values.shape != (2, 2) or not np.isfinite(values).all():
        raise ValueError('Expected one finite 2 x 2 S matrix.')
    record = {}
    for name, (row, column) in dict(S11=(0, 0), S21=(1, 0), S12=(0, 1), S22=(1, 1)).items():
        value = complex(values[row, column])
        record[name] = dict(real=value.real, imag=value.imag, power=abs(value)**2, phase_rad=cmath.phase(value))
    record['column_power'] = [float(np.sum(np.abs(values[:, 0])**2)), float(np.sum(np.abs(values[:, 1])**2))]
    record['reciprocity'] = float(abs(values[1, 0]-values[0, 1]))
    record['transmission'] = record['S21']['power']
    return record


def build_problem(seed, settings, objective):
    """The declared start (logits 0.5 randn from torch.Generator().manual_seed(seed), as in G6) and schedule."""
    if settings.filter_radius_um is None:
        raise ValueError('Choose the filter radius on the development seeds first (develop, select_radius).')
    initial = .5*torch.randn(PIXELS, generator=torch.Generator().manual_seed(seed))
    design = DensityParameterization(PIXELS, spacing_um=PIXEL_UM, initial=initial, mode='logits',
                                     filter_radius_um=settings.filter_radius_um, boundary='truncate', symmetry='rotate180',
                                     beta=settings.beta, eta=.5)
    optimizer = torch.optim.Adam(design.parameters(), lr=settings.learning_rate)
    return DesignProblem(design, objective, optimizer, name=f'g7-coupler-seed{seed}',
                         continuation=Continuation(every=settings.continuation_every, factor=2., maximum=settings.beta_maximum))


def orbit(pixel):
    """Representative of a pixel's orbit under the 180-degree rotation of the design box."""
    image = (PIXELS[0]-1-pixel[0], PIXELS[1]-1-pixel[1])
    return min(tuple(pixel), image)


def gradient_check(problem, model, settings):
    """Adjoint derivative of |S21|^2 against central finite differences at the current design logits.

    The pixels are the settings.fd_pixels largest adjoint magnitudes taken from distinct rotation
    orbits (a fixed rule: a relative comparison needs derivatives above the float32 floor). The
    logits are restored exactly afterwards and the gradients cleared.
    """
    design = problem.parameterization
    logits = design.design.detach().clone()

    def transmission():
        return model.network(model.epsilon(design())).s[1, 0].abs().square()
    started = time.perf_counter()
    design.zero_grad(set_to_none=True)
    value = transmission()
    value.backward()
    adjoint = design.design.grad.detach().clone()
    design.zero_grad(set_to_none=True)
    order = torch.argsort(adjoint.abs().reshape(-1), descending=True).tolist()
    pixels, orbits = [], set()
    for flat in order:
        pixel = divmod(flat, PIXELS[1])
        if orbit(pixel) not in orbits:
            orbits.add(orbit(pixel))
            pixels.append(pixel)
        if len(pixels) == settings.fd_pixels:
            break
    rows = []
    for pixel in pixels:
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                design.design.copy_(logits)
                design.design[pixel] += sign*settings.fd_step
                values.append(float(transmission()))
        with torch.no_grad():
            design.design.copy_(logits)
        finite = (values[0]-values[1])/(2*settings.fd_step)
        derivative = float(adjoint[pixel])
        rows.append(dict(pixel=list(pixel), adjoint=derivative, finite_difference=finite, transmission_plus=values[0],
                         transmission_minus=values[1], relative_error=abs(derivative-finite)/abs(finite)))
    assert torch.equal(design.design.detach(), logits)
    return dict(variable='design logits (optimization variables) at the start design', quantity='|S21|^2 at 1.55 um',
                step=settings.fd_step, beta=float(design.beta), transmission=float(value.detach()), pixel_rule=(
                    f'the {settings.fd_pixels} largest adjoint magnitudes from distinct rotation orbits'),
                pixels=rows, max_relative_error=max(r['relative_error'] for r in rows), wall_time_s=time.perf_counter()-started)


def perturbed_designs(binary):
    """The binary design eroded and dilated by EROSION_UM on the sub-pixel grid (extended box edges)."""
    fine = np.kron(np.asarray(binary, dtype=bool), np.ones((SUBPIXEL, SUBPIXEL), dtype=bool))
    eroded, dilated, realized = fabrication_perturbation(fine, EROSION_UM, PIXEL_UM/SUBPIXEL, boundary='extend')
    return fine, eroded, dilated, realized


def structures_of(fine, name):
    return binary_structures(fine, origin_um=(base.BOX_UM[0], base.BOX_UM[2]), spacing_um=PIXEL_UM/SUBPIXEL,
                             z_min_um=-base.SLAB_UM/2, z_max_um=base.SLAB_UM/2, material=base.MATERIAL, id_prefix=name)


def lf_line_endings(*paths):
    """Rewrite text files written in the platform's text mode (CRLF on Windows) with LF, content unchanged."""
    for path in paths:
        data = Path(path).read_bytes()
        if b'\r\n' in data:
            Path(path).write_bytes(data.replace(b'\r\n', b'\n'))


def _sync(settings):
    if torch.device(settings.device).type == 'cuda':
        torch.cuda.synchronize()


class Models:
    """The networks of every wavelength at the design mesh and at the check mesh, shared by all seeds."""
    def __init__(self, settings):
        started = time.perf_counter()
        self.design = {w: Coupler(settings.mesh_um, settings.steps, w, settings.device) for w in WAVELENGTHS_UM}
        self.check = {w: Coupler(settings.check_mesh_um, settings.check_steps, w, settings.device) for w in WAVELENGTHS_UM}
        self.wall_time_s = time.perf_counter()-started


def run_seed(seed, settings, models, export_dir):
    """Gradient check, optimization, fabrication, export and every evaluation of one declared seed."""
    started = time.perf_counter()
    times = {}
    problem = build_problem(seed, settings, models.design[DESIGN_UM].objective)
    gradient = gradient_check(problem, models.design[DESIGN_UM], settings)
    _sync(settings)
    times['gradient_check'] = gradient['wall_time_s']
    mark = time.perf_counter()
    history = problem.run(settings.iterations)
    _sync(settings)
    times['optimization'] = time.perf_counter()-mark

    smooth, binary = problem.density(), problem.density(hard=True)
    sizes = measure_feature_sizes(binary, PIXEL_UM, boundary='extend')
    violations = sizes.violations(min_linewidth_um=MIN_FEATURE_UM, min_gap_um=MIN_FEATURE_UM)
    exported = problem.export(Path(export_dir)/f'seed{seed}', origin_um=(base.BOX_UM[0], base.BOX_UM[2]),
                              spacing_um=PIXEL_UM, z_min_um=-base.SLAB_UM/2, z_max_um=base.SLAB_UM/2,
                              material=base.MATERIAL)
    lf_line_endings(exported['structure_path'], exported['sidecar_path'], exported['binary_path'])
    reimported = problem.reimport(exported)
    fine, eroded, dilated, realized = perturbed_designs(binary.numpy())
    shapes = dict(eroded=structures_of(eroded, f'seed{seed}-eroded'), dilated=structures_of(dilated, f'seed{seed}-dilated'))

    mark = time.perf_counter()
    evaluations = {name: {} for name in ('smooth_density', 'binary_density', 'structures', 'gds', 'eroded', 'dilated',
                                         'gds_check_mesh')}
    for wavelength in WAVELENGTHS_UM:
        key = f'{wavelength:.2f}'
        model = models.design[wavelength]
        evaluations['smooth_density'][key] = s_record(model.s_density(smooth))
        evaluations['binary_density'][key] = s_record(model.s_density(binary))
        evaluations['structures'][key] = s_record(model.s_structures(exported['structures']))
        evaluations['gds'][key] = s_record(model.s_structures(reimported['gds_structures']))
        for name, structures in shapes.items():
            evaluations[name][key] = s_record(model.s_structures(structures))
        evaluations['gds_check_mesh'][key] = s_record(models.check[wavelength].s_structures(reimported['gds_structures']))
    _sync(settings)
    times['evaluation'] = time.perf_counter()-mark
    design_key = f'{DESIGN_UM:.2f}'
    transmission = {name: {key: value['transmission'] for key, value in stage.items()} for name, stage in evaluations.items()}
    nominal = transmission['structures'][design_key]
    fabrication = dict(
        feature_sizes=sizes.report(), declared=dict(min_linewidth_um=MIN_FEATURE_UM, min_gap_um=MIN_FEATURE_UM,
                                                    erosion_dilation_um=EROSION_UM, boundary='extend'),
        violations=list(violations), satisfies_declared_constraints=not violations,
        enforcement=f'conic filter radius {settings.filter_radius_um} um with tanh projection to beta {settings.beta_maximum}, '
                    'then this morphological check of the thresholded design',
        perturbation=dict(realized_um=realized, grid_um=PIXEL_UM/SUBPIXEL, operation='digital-square erosion and dilation',
                          nominal_transmission=nominal, solid_fraction=float(fine.mean()),
                          eroded=dict(transmission=transmission['eroded'][design_key],
                                      change=transmission['eroded'][design_key]-nominal, solid_fraction=float(eroded.mean())),
                          dilated=dict(transmission=transmission['dilated'][design_key],
                                       change=transmission['dilated'][design_key]-nominal, solid_fraction=float(dilated.mean()))))
    differences = {key: dict(gds_round_trip=transmission['gds'][key]-transmission['structures'][key],
                             smoothing=transmission['structures'][key]-transmission['binary_density'][key],
                             thresholding=transmission['binary_density'][key]-transmission['smooth_density'][key],
                             check_mesh=transmission['gds_check_mesh'][key]-transmission['gds'][key]) for key in transmission['gds']}
    gds = Path(exported['gds_path'])
    times['total'] = time.perf_counter()-started
    return dict(
        task='G7-03', case=CASE, seed=seed, iterations=problem.iteration, settings=asdict(settings),
        start=f'logits 0.5 * torch.randn({PIXELS}, generator=torch.Generator().manual_seed(seed))',
        history=history, gradient_check=gradient, final_density=smooth.tolist(), binary=binary.int().tolist(),
        fabrication=fabrication, evaluations=evaluations, transmission=transmission, differences=differences,
        holdout=dict(wavelength_um=HOLDOUT_UM, gds_transmission=transmission['gds'][f'{HOLDOUT_UM:.2f}']),
        export=dict(gds=gds.relative_to(Path(export_dir).parent).as_posix(), gds_sha256=hashlib.sha256(gds.read_bytes()).hexdigest(), structure_count=len(exported['structures']),
                    reimported_structure_count=len(reimported['structures']), gds_structure_count=len(reimported['gds_structures']),
                    gds_maximum_rounding_um=exported['sidecar']['maximum_rounding_um']),
        conventions=dict(s='S[out, in], port 1 = left guide (x = -1 um), port 2 = right guide (x = +1 um), phases at the port phase planes',
                         binary_density='thresholded pixels through bounded_density_layer (volume average per Yee component)',
                         smooth_density='final continuous density (last beta) through the same transfer',
                         structures='pre-export native rectangles, inclusive staircase voxelization',
                         gds='polygons read back by import_gds from the exported file, same voxelization',
                         gds_check_mesh='the re-imported GDS polygons at check_mesh_um over the same physical time',
                         eroded_dilated='binary design on the 0.05 um grid eroded or dilated by 0.1 um, as rectangles',
                         differences='after minus before: gds_round_trip = gds - structures, smoothing = structures - binary_density, '
                                     'thresholding = binary_density - smooth_density, check_mesh = gds_check_mesh - gds'),
        wall_time_s=times)


def develop(radius_um, seed, settings):
    """One development run of the filter-radius selection: the judged optimization and threshold on a development seed.

    Records the measured linewidth and gap (pixels of 0.1 um; the rule is 4) and |S21|^2 at 1.55 um of the last
    iterate, the binary density and the staircase rectangles at the design mesh.
    """
    if seed in SEEDS:
        raise ValueError('The declared seeds never enter the development selection.')
    settings = replace(settings, filter_radius_um=radius_um)
    started = time.perf_counter()
    model = Coupler(settings.mesh_um, settings.steps, DESIGN_UM, settings.device)
    problem = build_problem(seed, settings, model.objective)
    history = problem.run(settings.iterations)
    binary = problem.density(hard=True)
    sizes = measure_feature_sizes(binary, PIXEL_UM, boundary='extend')
    rectangles = binary_structures(binary.numpy().astype(bool), origin_um=(base.BOX_UM[0], base.BOX_UM[2]), spacing_um=PIXEL_UM,
                                   z_min_um=-base.SLAB_UM/2, z_max_um=base.SLAB_UM/2, material=base.MATERIAL,
                                   id_prefix=f'development-seed{seed}')
    return dict(radius_um=radius_um, seed=seed, settings=asdict(settings), linewidth_px=sizes.linewidth_pixels,
                gap_px=sizes.gap_pixels, feature_sizes=sizes.report(),
                violations=list(sizes.violations(min_linewidth_um=MIN_FEATURE_UM, min_gap_um=MIN_FEATURE_UM)),
                binary=binary.int().tolist(),
                transmission=dict(last_iterate=history[-1]['metrics']['transmission'],
                                  binary_density=float(model.s_density(binary)[1, 0].abs().square()),
                                  structures=float(model.s_structures(rectangles)[1, 0].abs().square())),
                trace=[h['metrics']['transmission'] for h in history], environment=environment(settings),
                wall_time_s=time.perf_counter()-started)


def finite_difference_probe(seed, settings, steps=(.01, .02, .05)):
    """Development check of the central-difference step: the gradient check of a development seed at several steps."""
    if seed in SEEDS:
        raise ValueError('The declared seeds never enter the development selection.')
    model = Coupler(settings.mesh_um, settings.steps, DESIGN_UM, settings.device)
    problem = build_problem(seed, settings, model.objective)
    rows = {f'{step:g}': gradient_check(problem, model, replace(settings, fd_step=step)) for step in steps}
    return dict(seed=seed, settings=asdict(settings), max_relative_error={key: row['max_relative_error'] for key, row in rows.items()},
                checks=rows, environment=environment(settings))


def select_radius(records):
    """The smallest candidate radius whose binary designs meet the declared linewidth and gap for every development seed.

    Candidates are taken in ascending order; a candidate not yet run for every development seed stops the scan,
    so None means that no complete smaller candidate met the rule yet.
    """
    rows = {}
    for record in records:
        rows.setdefault(round(record['radius_um'], 6), {})[record['seed']] = record
    for radius in RADIUS_CANDIDATES_UM:
        runs = rows.get(round(radius, 6), {})
        if set(runs) != set(DEVELOPMENT_SEEDS):
            return None
        if all(not run['violations'] for run in runs.values()):
            return radius
    return None


def development_selection(output_dir):
    """The recorded filter-radius selection (and step probe) from output_dir/development, for the summary."""
    folder = Path(output_dir)/'development'
    records = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(folder.glob('radius-*.json'))]
    probes = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(folder.glob('fd-step-*.json'))]
    return dict(rule='smallest candidate filter radius whose binary designs meet the declared 0.4 um (4-pixel) linewidth and gap '
                     'for every development seed', candidates_um=list(RADIUS_CANDIDATES_UM), development_seeds=list(DEVELOPMENT_SEEDS),
                judged_seeds_used=any(r['seed'] in SEEDS for r in records+probes), selected_um=select_radius(records),
                runs=[dict(radius_um=r['radius_um'], seed=r['seed'], linewidth_px=r['linewidth_px'], gap_px=r['gap_px'],
                           violations=r['violations'], transmission=r['transmission'], commit=r['environment']['commit'])
                      for r in records],
                finite_difference_step=[dict(seed=p['seed'], radius_um=p['settings']['filter_radius_um'],
                                             max_relative_error=p['max_relative_error']) for p in probes])


G6_EXPORT = ROOT/'docs/validation/g6/export/coupler'


def same_mesh_baseline(models, settings):
    """Non-judged diagnostic: G6's three coupler designs through this workflow's GDS round trip and meshes.

    Each G6 binary design (docs/validation/g6/export/coupler/seedN, optimized by G6 at 0.2 um and never
    optimized here) becomes the same rectangles and export_gds file as a judged seed, is re-imported with
    import_gds and evaluated at the design mesh and the check mesh at every wavelength; its linewidth and
    gap are measured against the declared rule. The re-export must voxelize like G6's committed GDS.
    """
    import tempfile
    started = time.perf_counter()
    layer = GDSLayer(layer=1, datatype=0, z_min=-base.SLAB_UM/2, z_max=base.SLAB_UM/2, material=base.MATERIAL)
    designs = {}
    with tempfile.TemporaryDirectory() as scratch:
        for seed in SEEDS:
            folder = G6_EXPORT/f'seed{seed}'
            pixels = np.array(json.loads((folder/f'coupler-seed{seed}-binary.json').read_text(encoding='utf-8'))['pixels'], dtype=bool)
            sizes = measure_feature_sizes(pixels, base.PIXEL_UM, boundary='extend')
            name = f'g6-coupler-seed{seed}'
            structures = binary_structures(pixels, origin_um=(base.BOX_UM[0], base.BOX_UM[2]), spacing_um=base.PIXEL_UM,
                                           z_min_um=-base.SLAB_UM/2, z_max_um=base.SLAB_UM/2, material=base.MATERIAL, id_prefix=name)
            path = Path(scratch)/f'{name}.gds'
            export_gds(path, structures, layers={s.id: (1, 0) for s in structures}, cell='DESIGN')
            polygons = import_gds(path, cell='DESIGN', layers=[layer]).structures
            committed = import_gds(folder/f'coupler-seed{seed}.gds', cell='DESIGN', layers=[layer]).structures
            forward = models.design[DESIGN_UM]
            identical = bool(torch.equal(forward.structure_epsilon(polygons), forward.structure_epsilon(committed)))
            evaluations = {stage: {} for stage in ('structures', 'gds', 'gds_check_mesh')}
            for wavelength in WAVELENGTHS_UM:
                key = f'{wavelength:.2f}'
                evaluations['structures'][key] = s_record(models.design[wavelength].s_structures(structures))
                evaluations['gds'][key] = s_record(models.design[wavelength].s_structures(polygons))
                evaluations['gds_check_mesh'][key] = s_record(models.check[wavelength].s_structures(polygons))
            g6 = json.loads((ROOT/f'docs/validation/g6/coupler-seed{seed}.json').read_text(encoding='utf-8'))
            designs[seed] = dict(
                source=f'docs/validation/g6/export/coupler/seed{seed}', binary=pixels.astype(int).tolist(),
                feature_sizes=sizes.report(), violations=list(sizes.violations(min_linewidth_um=MIN_FEATURE_UM, min_gap_um=MIN_FEATURE_UM)),
                transmission={stage: {key: value['transmission'] for key, value in values.items()} for stage, values in evaluations.items()},
                g6_recorded_fine_gds_transmission=g6['final_evaluation']['stages']['fine_gds']['transmission'],
                g6_recorded_fine_mesh_um=g6['declared']['fine_mesh_um'], committed_gds_voxelizes_identically=identical,
                evaluations=evaluations)
    return dict(diagnostic='same-mesh baseline', judged=False,
                note='G6 designs re-evaluated with the G7 pipeline; they were not optimized in G7, and this block enters no '
                     'criterion of the case',
                mesh_um=settings.mesh_um, steps=settings.steps, check_mesh_um=settings.check_mesh_um, check_steps=settings.check_steps,
                designs=designs, environment=environment(settings), wall_time_s=time.perf_counter()-started)


def _criterion(name, value, limit, passed, **details):
    return dict(criterion=name, value=value, limit=limit, passed=bool(passed), **details)


def judge(records):
    """The seven acceptance criteria of G7-03 from the seed records; every seed enters, none is dropped."""
    records = sorted(records, key=lambda r: r['seed'])
    seeds = [r['seed'] for r in records]
    design_key = f'{DESIGN_UM:.2f}'
    complete = all(r['iterations'] == r['settings']['iterations'] and len(r['history']) == r['iterations']
                   and all(len(stage) == len(WAVELENGTHS_UM) for stage in r['evaluations'].values()) for r in records)
    per_seed = {r['seed']: dict(gds_transmission=r['transmission']['gds'][design_key],
                                holdout_gds_transmission=r['holdout']['gds_transmission'],
                                violations=r['fabrication']['violations'],
                                linewidth_um=r['fabrication']['feature_sizes']['min_linewidth_um'],
                                gap_um=r['fabrication']['feature_sizes']['min_gap_um']) for r in records}
    best = max(records, key=lambda r: r['transmission']['gds'][design_key])
    cases = [(r['seed'], stage, key, value) for r in records for stage, values in r['evaluations'].items() for key, value in values.items()]
    passivity = max(cases, key=lambda c: max(c[3]['column_power']))
    reciprocity = max(cases, key=lambda c: c[3]['reciprocity'])
    gradients = [(r['seed'], row) for r in records for row in r['gradient_check']['pixels']]
    gradient = max(gradients, key=lambda g: g[1]['relative_error'])
    round_trips = [(r['seed'], key, abs(d['gds_round_trip'])) for r in records for key, d in r['differences'].items()]
    round_trip = max(round_trips, key=lambda g: g[2])
    mesh = best['differences'][design_key]['check_mesh']
    criteria = [
        _criterion('a_all_seeds', seeds, list(SEEDS), seeds == list(SEEDS) and complete,
                   note='every declared seed with its full history and every evaluation'),
        _criterion('b_performance', best['transmission']['gds'][design_key], LIMITS['performance'],
                   best['transmission']['gds'][design_key] >= LIMITS['performance'] and not best['fabrication']['violations'],
                   best_seed=best['seed'], best_seed_violations=best['fabrication']['violations'], per_seed=per_seed,
                   note='|S21|^2 at 1.55 um of the re-imported GDS at the design mesh; the best seed must also meet the '
                        'declared 0.4 um linewidth and gap'),
        _criterion('c_passivity', max(passivity[3]['column_power']), LIMITS['passivity'],
                   max(passivity[3]['column_power']) <= LIMITS['passivity'], worst=dict(seed=passivity[0], stage=passivity[1],
                   wavelength_um=float(passivity[2])), cases=len(cases)),
        _criterion('d_reciprocity', reciprocity[3]['reciprocity'], LIMITS['reciprocity'],
                   reciprocity[3]['reciprocity'] <= LIMITS['reciprocity'], worst=dict(seed=reciprocity[0], stage=reciprocity[1],
                   wavelength_um=float(reciprocity[2])), cases=len(cases)),
        _criterion('e_gradient', gradient[1]['relative_error'], LIMITS['gradient'], gradient[1]['relative_error'] <= LIMITS['gradient'],
                   worst=dict(seed=gradient[0], pixel=gradient[1]['pixel']), checked=len(gradients)),
        _criterion('f_gds_round_trip', round_trip[2], LIMITS['gds'], round_trip[2] <= LIMITS['gds'],
                   worst=dict(seed=round_trip[0], wavelength_um=float(round_trip[1])),
                   note='|T(re-imported GDS) - T(pre-export rectangles)| at every seed and wavelength, same mesh and voxelization'),
        _criterion('g_mesh', abs(mesh), LIMITS['mesh'], abs(mesh) <= LIMITS['mesh'], best_seed=best['seed'], signed=mesh,
                   per_seed={r['seed']: r['differences'][design_key]['check_mesh'] for r in records},
                   note='|T(GDS, check mesh) - T(GDS, design mesh)| at 1.55 um for the best seed'),
    ]
    return dict(task='G7-03', case=CASE, seeds=seeds, best_seed=best['seed'], criteria=criteria,
                passed=all(c['passed'] for c in criteria), settings=records[0]['settings'])


def environment(settings):
    """Where torchfdtd came from (checkout or installed wheel), its version and the repository commit."""
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        from importlib.metadata import PackageNotFoundError, version
        distribution = version('torchfdtd')
    except (ImportError, PackageNotFoundError):
        distribution = None
    source = Path(torchfdtd.__file__).resolve()
    device = torch.device(settings.device)
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, os=platform.platform(),
                device=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or 'cpu',
                commit=commit, torchfdtd_file=str(source), torchfdtd_version=getattr(torchfdtd, '__version__', None),
                torchfdtd_distribution_version=distribution,
                torchfdtd_import='checkout' if ROOT in source.parents else 'installed')


def write_json(path, payload):
    """Write through a temporary file, so seeds running in parallel never read a partial record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(payload, indent=1, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def summarize(output_dir):
    """Judge every seed record in output_dir (one settings set) and write summary.json next to them."""
    output_dir = Path(output_dir)
    records = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(output_dir.glob('seed*.json'))]
    if not records:
        raise ValueError(f'No seed records in {output_dir}.')
    if len({json.dumps(r['settings'], sort_keys=True) for r in records}) != 1:
        raise ValueError('The seed records were produced with different settings; judge one run only.')
    summary = judge(records)
    seeds = tuple(summary['seeds'])
    summary.update(judged=not records[0]['settings']['reduced'] and seeds == SEEDS,
                   environment={r['seed']: r['environment'] for r in records},
                   wall_time_s=dict(per_seed={r['seed']: r['wall_time_s'] for r in records},
                                    network_setup={r['seed']: r['network_setup_s'] for r in records},
                                    total=sum(r['wall_time_s']['total']+r['network_setup_s'] for r in records)),
                   development_selection=development_selection(output_dir))
    summary['development_selection']['radius_used_um'] = records[0]['settings']['filter_radius_um']
    baseline = output_dir/'baseline.json'
    if baseline.exists():
        summary['diagnostics_same_mesh_baseline'] = json.loads(baseline.read_text(encoding='utf-8'))
    write_json(output_dir/'summary.json', summary)
    return summary


def run_baseline(settings, output_dir, models=None):
    """Write the non-judged same-mesh baseline diagnostic to output_dir/baseline.json."""
    record = same_mesh_baseline(models or Models(settings), settings)
    write_json(Path(output_dir)/'baseline.json', record)
    return record


def run(seeds, settings, output_dir):
    """Run the seeds, write one record per seed, then judge every seed record in output_dir."""
    output_dir = Path(output_dir)
    models = Models(settings)
    records = []
    for index, seed in enumerate(seeds):
        record = run_seed(seed, settings, models, output_dir/'export')
        record.update(environment=environment(settings), network_setup_s=models.wall_time_s if index == 0 else 0.)
        write_json(output_dir/f'seed{seed}.json', record)
        records.append(record)
        print(json.dumps(dict(seed=seed, gds_transmission=record['transmission']['gds'][f'{DESIGN_UM:.2f}'],
                              violations=record['fabrication']['violations'],
                              gradient_error=record['gradient_check']['max_relative_error'],
                              wall_time_s=round(record['wall_time_s']['total'], 1))), flush=True)
    return records, summarize(output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', help='declared seeds by default; development seeds with --develop')
    parser.add_argument('--reduced', action='store_true', help='CPU mechanics run at a coarse mesh; never judged')
    parser.add_argument('--device', help='override the settings device')
    parser.add_argument('--summary-only', action='store_true', help='judge the seed records already in --output-dir')
    parser.add_argument('--baseline', action='store_true', help='write only the non-judged same-mesh baseline diagnostic')
    parser.add_argument('--develop', type=float, metavar='RADIUS_UM', help='development run of one candidate filter radius')
    parser.add_argument('--fd-probe', type=float, metavar='RADIUS_UM', help='development check of the central-difference step')
    args = parser.parse_args(argv)
    settings = REDUCED if args.reduced else Settings()
    if args.device:
        settings = replace(settings, device=args.device)
    if args.develop is not None or args.fd_probe is not None:
        for seed in args.seeds or DEVELOPMENT_SEEDS:
            if args.develop is not None:
                record = develop(args.develop, seed, settings)
                write_json(args.output_dir/'development'/f'radius-{args.develop:g}-seed{seed}.json', record)
                print(json.dumps({key: record[key] for key in ('radius_um', 'seed', 'linewidth_px', 'gap_px', 'violations', 'transmission')}),
                      flush=True)
            else:
                record = finite_difference_probe(seed, replace(settings, filter_radius_um=args.fd_probe))
                write_json(args.output_dir/'development'/f'fd-step-seed{seed}.json', record)
                print(json.dumps(record['max_relative_error']), flush=True)
        return record
    if args.seeds is None:
        args.seeds = list(SEEDS)
    if args.baseline:
        record = run_baseline(settings, args.output_dir)
        print(json.dumps({seed: dict(violations=d['violations'], linewidth_um=d['feature_sizes']['min_linewidth_um'],
                                     gap_um=d['feature_sizes']['min_gap_um'], **{stage: values[f'{DESIGN_UM:.2f}'] for stage, values in d['transmission'].items()})
                          for seed, d in record['designs'].items()}, indent=1))
        return record
    summary = summarize(args.output_dir) if args.summary_only else run(args.seeds, settings, args.output_dir)[1]
    print(json.dumps({c['criterion']: dict(value=c['value'], limit=c['limit'], passed=c['passed']) for c in summary['criteria']}, indent=1))
    return summary


if __name__ == '__main__':
    main()
