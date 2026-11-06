#!/usr/bin/env python3
"""Leg 20, `O1`: an opt-in time-step bound for graded grids, read from the realised mesh.

The rule under test is `Region.time_step_rule = 'realised_axis_minimum'` in the fork:

    dt <= courant_factor / (c * sqrt(sum over active axes of 1 / h_min(axis)**2))

with `h_min(axis)` the smallest realised node spacing on that axis. It is a BOUND on the time step.
It is not a description of how the vendor selects its time step; that mechanism stays class X.

Three modes, each its own run through `./tools/run.sh configs/leg20-dt-rule.json`:

  --arithmetic   C1 and the spectral margin. Every frozen P0-b slab grid: the per-axis-minimum form
                 in SI against the stored vendor dt (gated at 2e-16), the fork's opt-in rule on the
                 same nodes in micrometres (reported), the excluded global-minimum form (reported),
                 and the exact largest eigenvalue of the fork's vacuum curl-curl operator on those
                 realised widths (gated: the Courant ratio at the stored dt is below 1). Also a
                 fork-native graded mesh, where the nominal and realised rules differ. No GPU.
  --equivalence  The default path, frozen solver against this snapshot's solver, in two child
                 processes. Every compared number must be bit-identical. No GPU.
  --stability    C2. The registered graded fixture on the GPU at the opt-in bound, and a negative
                 control above the operator limit that must be caught as divergence.

The frozen coordinates and stored time steps are copied into the config with their source hashes.
No reference repository is read or imported at run time.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal

from parity import (C0, ROOT, Project, Region, Material, Structure, Source, FieldMonitor,  # noqa: F401
                    Simulation, write_json)

EXIT_FAILED = 53


# ----------------------------------------------------------------------------------------------
# Arithmetic

def axis_minimum_bound_si(nodes_m, factor):
    """The per-axis-minimum form in SI, on the stored coordinates, with no unit round trip."""
    minima = [float(np.diff(np.asarray(a, dtype=np.float64)).min()) for a in nodes_m]
    if any(not math.isfinite(h) or h <= 0 for h in minima):
        raise ValueError('Realised spacings must be finite and positive')
    return factor / (C0 * math.sqrt(sum(1 / h**2 for h in minima))), minima


def global_minimum_bound_si(nodes_m, factor):
    """The excluded alternative: one global minimum spacing, isotropic form."""
    h = min(float(np.diff(np.asarray(a, dtype=np.float64)).min()) for a in nodes_m)
    return factor * h / (C0 * math.sqrt(len(nodes_m)))


def relative(a, b):
    # Subtraction before division keeps a sub-ulp ratio meaningful.
    return abs(a - b) / abs(b)


def largest_eigenvalue(widths, periodic):
    """Largest eigenvalue of the fork's 1-D vacuum operator D_dual^T W_primal^-1 D on these widths.

    Primal widths w_i are the realised cells; dual widths are means of adjacent primal widths, the
    construction in `torchfdtd.boundaries.YeeGrid._prepare_boundaries` (the periodic wrap dual is
    (w_0 + w_last)/2). The operator is symmetrised by the dual weights. On a rectilinear 2-D grid
    the vacuum operator is separable, so the 2-D largest eigenvalue is the sum over axes.
    Dirichlet ends are a truncation; the periodic form is the fixture's y boundary.
    """
    w = np.asarray(widths, dtype=np.float64)
    a = 1 / w
    if periodic:
        n = len(w)
        dual = (w + np.roll(w, 1)) / 2           # dual at node i sits between cells i-1 and i
        s = 1 / np.sqrt(dual)
        m = np.zeros((n, n))
        idx = np.arange(n)
        m[idx, idx] = (a + np.roll(a, 1)) * s * s
        m[idx, (idx + 1) % n] -= a * s * np.roll(s, -1)
        m[(idx + 1) % n, idx] -= a * s * np.roll(s, -1)
        return float(np.linalg.eigvalsh(m)[-1])
    dual = (w[1:] + w[:-1]) / 2
    diag = (a[:-1] + a[1:]) / dual
    off = -a[1:-1] / np.sqrt(dual[:-1] * dual[1:])
    n = len(diag)
    return float(eigh_tridiagonal(diag, off, eigvals_only=True, select='i', select_range=(n - 1, n - 1))[0])


def courant_ratio(dt, lam):
    """c dt sqrt(lambda_max) / 2. Leapfrog on this operator is stable if and only if it is <= 1."""
    return C0 * dt * math.sqrt(lam) / 2


def fork_region_um(grid, rule):
    """The same frozen nodes as a fork explicit Region, converted to micrometres node by node."""
    um = [[float(v) * 1e6 for v in grid['nodes_m'][a]] for a in ('x', 'y')]
    span = [u[-1] - u[0] for u in um]
    return Region(dimension='2d', size=(span[0], span[1], 0.1), mesh_type='explicit',
                  mesh_coordinates=(tuple(um[0]), tuple(um[1]), (-0.05, 0.05)),
                  material_sampling='yee', pml_cells=8, time_step_rule=rule,
                  boundaries=dict(y_min=dict(kind='periodic'), y_max=dict(kind='periodic')))


def native_graded(case):
    """A fork-native graded mesh: the case where nominal and realised rules can differ."""
    project = Project.model_validate(case['project'])
    nominal = project.region
    opt = Project.model_validate(dict(case['project'], region=dict(case['project']['region'],
                                      time_step_rule='realised_axis_minimum'))).region
    assert all(np.array_equal(a, b) for a, b in zip(nominal.mesh_nodes, opt.mesh_nodes))
    widths = [np.diff(n) for n in opt.mesh_nodes[:2]]
    return dict(id=case['id'], shape=list(nominal.shape),
                nominal_axis_steps_um=list(nominal.axis_steps[:2]),
                realised_axis_minima_um=list(opt.realised_axis_minima),
                realised_axis_maxima_um=[float(w.max()) for w in widths],
                minimum_below_nominal_relative=[(n - r) / n for n, r in
                                                zip(nominal.axis_steps[:2], opt.realised_axis_minima)],
                nominal_rule_dt_s=nominal.time_step, realised_rule_dt_s=opt.time_step,
                realised_over_nominal_minus_one=opt.time_step / nominal.time_step - 1,
                spectral_courant_ratio_nominal_dt=courant_ratio(
                    nominal.time_step, sum(largest_eigenvalue(w * 1e-6, False) for w in widths)),
                spectral_courant_ratio_realised_dt=courant_ratio(
                    opt.time_step, sum(largest_eigenvalue(w * 1e-6, False) for w in widths)))


def arithmetic(cfg, run):
    rows, cache = [], {}
    tol = cfg['relative_tolerance']
    for grid in cfg['frozen_grids']:
        nodes = [grid['nodes_m'][a] for a in ('x', 'y')]
        bound, minima = axis_minimum_bound_si(nodes, cfg['courant_factor'])
        vendor = grid['vendor_dt_s']
        key = tuple(len(a) for a in nodes) + (hashlib.sha256(json.dumps(nodes).encode()).hexdigest(),)
        if key not in cache:
            widths = [np.diff(np.asarray(a, dtype=np.float64)) for a in nodes]
            cache[key] = dict(dirichlet=largest_eigenvalue(widths[0], False) + largest_eigenvalue(widths[1], False),
                              fixture_bc=largest_eigenvalue(widths[0], False) + largest_eigenvalue(widths[1], True))
        lam = cache[key]
        opt, default = fork_region_um(grid, 'realised_axis_minimum'), fork_region_um(grid, 'nominal')
        row = dict(id=grid['id'], layout=grid['layout'], vendor_dt_s=vendor,
                   realised_minimum_m=minima,
                   nominal_override_m=[grid['nominal_override_m']['dx'], grid['nominal_override_m']['dy']],
                   si_bound_s=bound, si_relative_residual=relative(bound, vendor),
                   c1_passed=relative(bound, vendor) <= tol,
                   fork_opt_in_dt_s=opt.cfl_time_step,
                   fork_opt_in_relative_residual=relative(opt.cfl_time_step, vendor),
                   fork_realised_minima_um=list(opt.realised_axis_minima),
                   fork_default_equals_opt_in=default.cfl_time_step == opt.cfl_time_step,
                   global_minimum_relative_residual=relative(global_minimum_bound_si(nodes, cfg['courant_factor']), vendor),
                   spectral_courant_ratio_dirichlet=courant_ratio(vendor, lam['dirichlet']),
                   spectral_courant_ratio_fixture_bc=courant_ratio(vendor, lam['fixture_bc']))
        row['spectral_passed'] = max(row['spectral_courant_ratio_dirichlet'], row['spectral_courant_ratio_fixture_bc']) < 1
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('id', 'si_relative_residual', 'fork_opt_in_relative_residual',
                                              'global_minimum_relative_residual', 'spectral_courant_ratio_fixture_bc',
                                              'c1_passed', 'spectral_passed')}), flush=True)
    graded = [native_graded(case) for case in cfg['native_graded_cases']]
    for g in graded:
        print(json.dumps(g), flush=True)
    c1 = all(r['c1_passed'] for r in rows)
    spectral = all(r['spectral_passed'] for r in rows) and all(
        g['spectral_courant_ratio_realised_dt'] < 1 for g in graded)
    summary = dict(grids=len(rows), c1_passed=c1, spectral_passed=spectral,
                   si_relative_residual_max=max(r['si_relative_residual'] for r in rows),
                   fork_opt_in_relative_residual_max=max(r['fork_opt_in_relative_residual'] for r in rows),
                   global_minimum_relative_residual_range=[min(r['global_minimum_relative_residual'] for r in rows),
                                                           max(r['global_minimum_relative_residual'] for r in rows)],
                   spectral_courant_ratio_range=[min(min(r['spectral_courant_ratio_dirichlet'], r['spectral_courant_ratio_fixture_bc']) for r in rows),
                                                 max(max(r['spectral_courant_ratio_dirichlet'], r['spectral_courant_ratio_fixture_bc']) for r in rows)],
                   fork_default_equals_opt_in_on_every_explicit_grid=all(r['fork_default_equals_opt_in'] for r in rows))
    print(json.dumps(summary), flush=True)
    write_json(run / 'leg20-arithmetic.json', dict(summary=summary, grids=rows, native_graded=graded,
                                                   grating=cfg['grating'], assumptions=cfg['assumptions'],
                                                   rests_on=cfg['rests_on']))
    return c1 and spectral


# ----------------------------------------------------------------------------------------------
# Default-path equivalence, frozen solver against this snapshot's solver

PROBE = r'''
import hashlib, json, math, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import torchfdtd
from torchfdtd import Project, Region
cases = json.loads(open(sys.argv[2]).read())
out = {'solver_file': torchfdtd.__file__, 'cases': []}
for case in cases:
    if case['kind'] == 'region':
        r = Region.model_validate(case['value']); dump = r.model_dump()
    else:
        p = Project.model_validate(case['value']); r = p.region; dump = p.model_dump()
    nodes = b''.join(np.ascontiguousarray(a).tobytes() for a in r.mesh_nodes)
    out['cases'].append(dict(id=case['id'], time_step=r.time_step.hex(), cfl_time_step=r.cfl_time_step.hex(),
        rectangular_courant=float(r.rectangular_courant).hex(), shape=list(r.shape),
        dump_sha256=hashlib.sha256(json.dumps(dump, sort_keys=True).encode()).hexdigest(),
        dump_has_time_step_rule='time_step_rule' in dump,
        nodes_sha256=hashlib.sha256(nodes).hexdigest()))
print(json.dumps(out))
'''


def equivalence_cases(cfg):
    cases = [dict(c) for c in cfg['equivalence']['cases']]
    for grid in cfg['frozen_grids']:
        um = [[float(v) * 1e6 for v in grid['nodes_m'][a]] for a in ('x', 'y')]
        cases.append(dict(id='frozen:' + grid['id'], kind='region', value=dict(
            dimension='2d', size=[um[0][-1] - um[0][0], um[1][-1] - um[1][0], 0.1], mesh_type='explicit',
            mesh_coordinates=[um[0], um[1], [-0.05, 0.05]], material_sampling='yee', pml_cells=8,
            time_step_override=grid['vendor_dt_s'],
            boundaries=dict(y_min=dict(kind='periodic'), y_max=dict(kind='periodic')))))
    for case in cfg['native_graded_cases']:
        cases.append(dict(id='native:' + case['id'], kind='project', value=case['project']))
    return cases


def equivalence(cfg, run):
    frozen = os.environ.get('SOLVER_FROZEN')
    if not frozen or not Path(frozen, 'torchfdtd/models.py').is_file():
        raise SystemExit('SOLVER_FROZEN is not set to a solver copy; source env/refs.env before the run')
    snapshot_solver = Path(os.environ['PARITY_SNAPSHOT']) / 'src/solver'
    cases = equivalence_cases(cfg)
    case_file = run / 'leg20-equivalence-cases.json'
    write_json(case_file, cases)
    probe = run / 'leg20_probe.py'
    probe.write_text(PROBE)
    results = {}
    for role, root in (('frozen', Path(frozen)), ('snapshot', snapshot_solver)):
        out = subprocess.run([sys.executable, '-B', '-I', str(probe), str(root), str(case_file)],
                             capture_output=True, text=True, check=True)
        results[role] = json.loads(out.stdout.strip().splitlines()[-1])
    a, b = results['frozen']['cases'], results['snapshot']['cases']
    fields = [k for k in a[0] if k != 'id']
    mismatches = [dict(id=x['id'], field=k, frozen=x[k], snapshot=y[k])
                  for x, y in zip(a, b) for k in fields if x[k] != y[k]]
    compared = len(a) * len(fields)
    summary = dict(cases=len(a), fields_per_case=len(fields), compared_values=compared,
                   mismatches=len(mismatches), frozen_solver=results['frozen']['solver_file'],
                   snapshot_solver=results['snapshot']['solver_file'],
                   passed=not mismatches and len(a) == len(b) and len(a) > 0)
    print(json.dumps(summary), flush=True)
    write_json(run / 'leg20-equivalence.json', dict(summary=summary, mismatches=mismatches,
                                                    frozen=results['frozen'], snapshot=results['snapshot']))
    return summary['passed']


# ----------------------------------------------------------------------------------------------
# Stability on the registered graded fixture

def fixture_project(cfg, rule, source_kind='plane'):
    """The frozen 375.3 fs coarse staircase resolution as a fork Project.

    The field mapping follows `$PARITY_REF/src/vendor_adapter.py` (descriptive_audit and
    candidate_project), re-implemented here, not imported: realised nodes converted node by node to
    micrometres, x PML with the frozen layer count and stretching parameters, periodic y, one
    nondispersive n = 3.5 rectangle, a broadband Ez source over the frozen 0.4-0.7 um object band.
    `plane` is the frozen one-way plane source. `point` is a soft point source at the same position,
    which breaks the y symmetry. The native CPML sigma is NOT the vendor's (that conversion is
    unverified, see vendor_adapter); this fixture tests stability, not parity.
    """
    raw = cfg['fixture_source']['project']
    r = raw['region']
    um = [[float(v) * 1e6 for v in a] for a in r['mesh_coordinates']]
    size = [a[-1] - a[0] for a in um]
    pml = raw['pml']
    layers = int(pml['layers'])
    faces = {f'x_{s}': dict(kind='pml', layers=layers, kappa=pml['kappa'], alpha=pml['alpha'],
                            polynomial=pml['polynomial'], alpha_polynomial=pml['alpha_polynomial'])
             for s in ('min', 'max')}
    faces.update({f'y_{s}': dict(kind='periodic') for s in ('min', 'max')})
    st = cfg['stability']
    region = dict(dimension='2d', size=size, mesh_type='explicit', mesh_coordinates=um,
                  material_sampling='yee', interface_method='staircase', pml_cells=layers,
                  boundaries=faces, backend='cuda', cuda_kernel='fused', cuda_monitor_kernel='fused',
                  precision='float64', snapshot_interval=10000, time_step_rule=rule,
                  run_control=dict(auto_shutoff=False, divergence_check=True,
                                   check_interval=st['check_interval'], growth_limit=st['growth_limit']))
    materials = [dict(name=m['id'], model='dielectric', index=m['index']) for m in raw['materials']]
    structures = [dict(id=o['id'], kind='rectangle', material=o['material'],
                       center=[(o['x_min'] + o['x_max']) * .5e6, (o['y_min'] + o['y_max']) * .5e6, 0],
                       size=[(o['x_max'] - o['x_min']) * 1e6, (o['y_max'] - o['y_min']) * 1e6, size[2]])
                  for o in raw['structures']]
    s = raw['sources'][0]
    band = dict(component='Ez', amplitude=s['amplitude'], center=[s['x'] * 1e6, s['y'] * 1e6, 0],
                pulse='broadband', time_definition='wavelength', wavelength_start=s['wavelength_start'] * 1e6,
                wavelength_stop=s['wavelength_stop'] * 1e6, optimize_for_short_pulse=True,
                eliminate_discontinuities=True)
    if source_kind == 'plane':
        source = dict(id=s['id'], kind='plane', injection='oneway', normal='x', direction='+',
                      size=[0, size[1], size[2]], **band)
    elif source_kind == 'point':
        source = dict(id=s['id'], kind='point', injection='soft', **band)
    else:
        raise ValueError(f'Unknown fixture source {source_kind!r}')
    project = Project.model_validate(dict(name='leg20-graded-fixture-' + source_kind, region=region,
                                          materials=materials, structures=structures, sources=[source],
                                          monitors=[]))
    project.region.steps = math.ceil(st['duration_s'] / project.region.time_step)
    return Project.model_validate(project.model_dump())


def history(result):
    summary = result.summary if isinstance(result.summary, dict) else json.loads(result.summary)
    return summary, summary.get('diagnostics') or []


def stability_case(cfg, case, steps):
    import torch
    st = cfg['stability']
    project = fixture_project(cfg, 'realised_axis_minimum', case['source'])
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = Simulation(project).run(cuda_graph=True, cuda_graph_steps=32)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    summary, hist = history(result)
    norms = [h['relative_norm'] for h in hist if h['source_finished']]
    state = [h['state_norm'] for h in hist]
    finite = all(math.isfinite(v) for v in state)
    peak_index = int(np.argmax(state)) if state else -1
    q = len(hist) // 4
    out = dict(id=case['id'], source=case['source'], seconds=seconds, dt_s=project.region.time_step,
               completed_steps=summary.get('steps'), requested_steps=summary.get('requested_steps'),
               termination_reason=summary.get('termination_reason'), samples=len(hist),
               finite=finite, peak_state_norm=max(state) if state else None,
               peak_time_s=hist[peak_index]['time_s'] if hist else None,
               post_source_samples=len(norms), post_source_max_relative=max(norms) if norms else None,
               final_relative=norms[-1] if norms else None,
               third_quarter_max_relative=max(h['relative_norm'] for h in hist[2 * q:3 * q]) if q else None,
               last_quarter_max_relative=max(h['relative_norm'] for h in hist[3 * q:]) if q else None)
    limit = case.get('post_source_ratio_limit', st['post_source_ratio_limit'])
    checks = dict(completed=out['completed_steps'] == steps and out['termination_reason'] == 'max_steps',
                  finite=finite,
                  no_post_source_growth=bool(norms) and out['post_source_max_relative'] <= limit,
                  decayed=bool(norms) and out['final_relative'] <= st['final_ratio_limit'])
    out.update(checks=checks, gated=case['gated'], post_source_ratio_limit=limit,
               passed=all(checks[k] for k in case['gated']))
    print(json.dumps({k: v for k, v in out.items()}), flush=True)
    return out, hist


def negative_case(cfg, case, over):
    """The fixture above the operator's own limit. The solver re-validates the project in
    Simulation() and again in resolve_plan, so the CFL guard is lifted in this process for this
    control only, by replacing the property, and restored afterwards. No solver file changes; the
    Yee update is the solver's own, with dt taken from the override."""
    neg = cfg['negative_control']
    control = fixture_project(cfg, 'realised_axis_minimum', case['source'])
    control.region.steps = neg['steps']
    control.region.time_step_override = over
    guard = Region.cfl_time_step
    caught, message, completed = False, None, None
    try:
        Region.cfl_time_step = property(lambda self: math.inf)
        summary, _ = history(Simulation(control).run(cuda_graph=True, cuda_graph_steps=32))
        completed = summary.get('steps')
    except FloatingPointError as error:
        caught, message = True, str(error)
    finally:
        Region.cfl_time_step = guard
    out = dict(id=case['id'], source=case['source'], dt_s=over, spectral_courant_ratio=neg['courant_ratio'],
               steps=neg['steps'], caught=caught, message=message, completed_steps=completed,
               must_be_caught=case['must_be_caught'],
               passed=caught if case['must_be_caught'] else None)
    print(json.dumps(out), flush=True)
    return out


def stability(cfg, run):
    project = fixture_project(cfg, 'realised_axis_minimum')
    r = project.region
    nodes = r.mesh_nodes
    widths_m = [np.diff(n) * 1e-6 for n in nodes[:2]]
    lam_x, lam_y = largest_eigenvalue(widths_m[0], False), largest_eigenvalue(widths_m[1], True)
    vendor_dt = cfg['fixture_source']['project']['region']['time_step_override']
    info = dict(shape=list(r.shape), steps=r.steps, dt_s=r.time_step, vendor_dt_s=vendor_dt,
                dt_relative_to_vendor=relative(r.time_step, vendor_dt),
                default_rule_dt_s=fixture_project(cfg, 'nominal').region.time_step,
                realised_minimum_um=list(r.realised_axis_minima),
                realised_maximum_um=[float(np.diff(n).max()) for n in nodes[:2]],
                nominal_override_um=[cfg['fixture_source']['project']['region']['mesh_override'][k] * 1e6 for k in ('dx', 'dy')],
                spectral_courant_ratio=courant_ratio(r.time_step, lam_x + lam_y),
                spectral_courant_ratio_x_only=courant_ratio(r.time_step, lam_x))
    print(json.dumps(dict(fixture=info)), flush=True)
    cases, histories = [], {}
    for case in cfg['stability']['cases']:
        out, hist = stability_case(cfg, case, r.steps)
        cases.append(out)
        histories[case['id']] = hist
    over = cfg['negative_control']['courant_ratio'] * 2 / (C0 * math.sqrt(lam_x + lam_y))
    negatives = [negative_case(cfg, case, over) for case in cfg['negative_control']['cases']]
    assert fixture_project(cfg, 'realised_axis_minimum').region.cfl_time_step == r.time_step, 'guard not restored'
    for n in negatives:
        n['dt_over_bound'] = over / r.time_step
        n['x_only_courant_ratio'] = courant_ratio(over, lam_x)
    passed = all(c['passed'] for c in cases) and all(n['passed'] for n in negatives if n['must_be_caught'])
    write_json(run / 'leg20-stability.json', dict(fixture=info, cases=cases, negative_controls=negatives,
                                                  histories=histories, project=project.model_dump(),
                                                  passed=passed))
    return passed


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('arithmetic', 'equivalence', 'stability'):
        mode.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    if 'RUN_DIR_ABS' not in os.environ:
        raise RuntimeError('Use tools/run.sh')
    cfg = json.loads(Path(os.environ['PARITY_CONFIG']).read_text())
    run = Path(os.environ['RUN_DIR_ABS'])
    name = next(n for n in ('arithmetic', 'equivalence', 'stability') if getattr(args, n))
    passed = dict(arithmetic=arithmetic, equivalence=equivalence, stability=stability)[name](cfg, run)
    write_json(os.environ['EFFECTIVE_JSON'], dict(config_policy=cfg['policy'], mode=name,
                                                  simulation_executed=name == 'stability', passed=passed))
    (run / 'verdict').write_text(f'{name}: {"passed" if passed else "failed"}\n')
    print(f'leg20 {name}: {"passed" if passed else "FAILED"}', flush=True)
    return 0 if passed else EXIT_FAILED


if __name__ == '__main__':
    sys.exit(main())
