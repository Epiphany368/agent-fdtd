#!/usr/bin/env python3
"""Leg 14, `V1-slab-mode`: the modal launch's own effective index on a symmetric slab.

`launch_neff` is the number `torchfdtd.mode_branches.prepare_aperture_modal_launch` returns for
the launched mode. This driver builds the N8 launch geometry (x-normal soft plane source, `Ey`,
the 3.6 micrometre aperture centred at y = 0.12, five periodic z cells of 0.04, the leg-1 core and
cladding indices) with ONE change that puts it inside the law's scope: the cross-section is a
SYMMETRIC slab with sharp interfaces (cladding on both sides of the 0.22 micrometre core). In
the N8 aperture the cross-section is already cladding / core / cladding; the N8 builder smooths
the interfaces over `transition_nm`, which this case does not.

Two modes of operation, both through `./tools/run.sh`:

  floor   (`configs/leg14-slab-floor.json`)   a mesh ladder. Measures the discretisation floor
          from the solver's OWN self-convergence. The closed-form law is never evaluated here, so
          the floor cannot be tuned to the law.
  check   (`configs/leg14-analytic-cases.json`, `slab` block)  the fixture mesh, judged by
          `analytic_cases` against the tolerance declared from the floor run, plus the
          asymmetric-stack negative control.

Sharp interface, sampled: a Yee sample that falls exactly on an interface takes the mean of the
two permittivities (the arithmetic sub-pixel value of a node bisected by the interface); every
other sample takes its own layer's value. No smoothing width.
"""
import json
import math
import sys
import time
from pathlib import Path

from parity import *  # noqa: F401,F403  (Region, Project, Source, Boundaries, load_run, write_json)
import numpy as np
from torchfdtd import Boundaries, BoundaryFace
from torchfdtd.mode_branches import prepare_aperture_modal_launch

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'leg14_slab.section (symmetric or asymmetric sharp three-layer stack)',
    'F2-mesh': 'leg14_slab.project_for (uniform Region, the N8 boundary kinds)',
    'F3-source': 'torchfdtd.mode_branches.prepare_aperture_modal_launch',
    'F4-monitors': 'none: the quantity is the launch eigenmode itself',
    'F5-run': 'no time stepping: the launch mode is solved, the field is never advanced',
    'F6-readout': 'launch.mode.neff, the same expression leg1_objective.py records',
}
CORE_Y = (0.0, 0.22)


def section(core_eps, lower_eps, upper_eps):
    """eps(y, z) of a sharp three-layer stack; a sample on an interface takes the mean."""
    lo, hi = CORE_Y

    def eps(y, z):
        y = np.broadcast_to(np.asarray(y, dtype=np.float64), np.broadcast(y, z).shape)
        out = np.where(y < lo, lower_eps, np.where(y > hi, upper_eps, core_eps)).astype(np.float64)
        on_lo = np.isclose(y, lo, rtol=0, atol=1e-9)
        on_hi = np.isclose(y, hi, rtol=0, atol=1e-9)
        out = np.where(on_lo, 0.5 * (core_eps + lower_eps), out)
        return np.where(on_hi, 0.5 * (core_eps + upper_eps), out)
    return eps


def project_for(cfg, dx, dy):
    """The N8 launch geometry (run_b9_native_shape_n8.scene), source only, short x span."""
    nz, zspan = cfg['z_cells'], cfg['z_cells'] * cfg['dz_um']
    span = cfg['x_span_um']
    boundaries = Boundaries(
        # Native CPML caps layers at 100. The mode is solved on the aperture with periodic
        # transverse walls, so the absorber depth does not enter `launch_neff`.
        x_min=BoundaryFace(kind='pml', layers=min(round(.64 / dx), 50)),
        x_max=BoundaryFace(kind='pml', layers=min(round(.64 / dx), 50)),
        y_min=BoundaryFace(kind='pml', layers=min(round(.4 / dy), 100)),
        y_max=BoundaryFace(kind='pml', layers=min(round(.4 / dy), 100)),
        z_min=BoundaryFace(kind='periodic'), z_max=BoundaryFace(kind='periodic'))
    region = Region(dimension='3d', size=(span, cfg['y_span_um'], zspan), mesh=dx,
                    mesh_steps=(dx, dy, cfg['dz_um']), pml_cells=min(round(.64 / dx), 50),
                    precision='float64', material_sampling='yee', interface_method='staircase',
                    boundaries=boundaries, steps=cfg['steps'])
    source = Source(id='source', kind='plane', normal='x', direction='+', component='Ey',
                    center=(0.0, cfg['aperture_center_y_um'], 0),
                    size=(0, cfg['aperture_um'], zspan), wavelength=cfg['wavelength_um'],
                    pulse_cycles=4)
    assert nz == round(zspan / cfg['dz_um'])
    return Project(name='leg14-slab', region=region, sources=[source], monitors=[])


def launch_neff(cfg, dx, dy, stack):
    """The launch the way leg1_objective.py selects it: two modes, the Ey-dominant one."""
    core, lower, upper = (cfg[key] ** 2 for key in stack)
    project = project_for(cfg, dx, dy)
    eps = section(core, lower, upper)
    started = time.perf_counter()
    modes = [prepare_aperture_modal_launch(project, eps, mode_index=i, num_modes=2)
             for i in range(2)]
    scores = [float(np.sum(abs(m.mode.fields[..., 1]) ** 2)
                    / np.sum(abs(m.mode.fields[..., [1, 2]]) ** 2)) for m in modes]
    launch = modes[int(np.argmax(scores))]
    if max(scores) <= 1 - 1e-8 or launch.mode.maxwell_residual >= 1e-5:
        raise SystemExit('The TM launch eigenmode is not resolvable on this cross-section')
    return dict(launch_neff=float(launch.mode.neff),
                residual=float(launch.mode.maxwell_residual),
                score=max(scores), shape=list(project.region.shape),
                elapsed_seconds=time.perf_counter() - started)


def floor_main(cfg, run):
    rows, values = [], []
    for dx, dy in cfg['ladder_dx_dy_um']:
        result = launch_neff(cfg, dx, dy, ('core_index', 'cladding_index', 'cladding_index'))
        values.append(result['launch_neff'])
        rows.append(dict(id=f'slab-tm0-dx{dx:g}-dy{dy:g}', shape=result['shape'],
                         elapsed_seconds=result['elapsed_seconds'],
                         measured=dict(launch_neff=result['launch_neff'],
                                       residual=result['residual'])))
        print(json.dumps(dict(dx=dx, dy=dy, **result)), flush=True)
    # Self-convergence only. Differences between successive meshes, each mesh halving.
    diffs = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    orders = [math.log2(abs(diffs[i] / diffs[i + 1])) for i in range(len(diffs) - 1)]
    fixture = [list(pair) for pair in cfg['ladder_dx_dy_um']].index(list(cfg['fixture_dx_dy_um']))
    # Richardson estimate of the fixture's own discretisation error, with the order observed on
    # the two finer pairs: e(h) = (n(h) - n(h/2)) * 2^p / (2^p - 1).
    p = orders[fixture] if fixture < len(orders) else orders[-1]
    estimate = abs(diffs[fixture]) * 2 ** p / (2 ** p - 1)
    extrapolated = values[-1] - diffs[-1] / (2 ** orders[-1] - 1)
    analysis = dict(values=values, differences=diffs, observed_orders=orders,
                    fixture_index=fixture, fixture_error_estimate=estimate,
                    safety_factor=cfg['safety_factor'],
                    floor=cfg['safety_factor'] * estimate, richardson_extrapolated=extrapolated)
    rows.append(dict(id='slab-tm0-floor', measured=dict(absolute_change=estimate),
                     recorded=dict(ladder=analysis)))
    print(json.dumps(analysis), flush=True)
    return rows, analysis


def check_main(cfg, run):
    dx, dy = cfg['fixture_dx_dy_um']
    rows = []
    for case_id, stack in (('slab-tm0-symmetric', ('core_index', 'cladding_index',
                                                     'cladding_index')),
                           ('slab-tm0-asymmetric-control', ('core_index', 'cladding_index',
                                                            'upper_index_control'))):
        result = launch_neff(cfg, dx, dy, stack)
        rows.append(dict(id=case_id, shape=result['shape'],
                         elapsed_seconds=result['elapsed_seconds'],
                         measured=dict(launch_neff=result['launch_neff'],
                                       residual=result['residual'])))
        print(json.dumps(dict(case=case_id, dx=dx, dy=dy, **result)), flush=True)
    return rows, None


def main():
    cfg, run = load_run()
    mode = cfg['slab']['mode']
    rows, analysis = (floor_main if mode == 'floor' else check_main)(cfg['slab'], run)
    readout = dict(policy=cfg['policy'], status='completed', simulation_executed=False,
                   fixture=('N8 launch geometry, symmetric sharp slab (cladding / 0.22 um core / '
                            'cladding), TM0 launch mode; mode solve only, no time stepping'),
                   stages=list(STAGES), stage_source=dict(STAGE_SOURCE), cases=rows)
    write_json(Path(run) / 'readout.json', readout)
    if mode == 'floor':
        return 0
    import analytic_cases
    checks = analytic_cases.evaluate_analytic_cases(readout, cfg)
    for check in checks:
        print(json.dumps(check), flush=True)
    return 0 if checks and all(c['status'] == 'passed' for c in checks) else 52


if __name__ == '__main__':
    sys.exit(main())
