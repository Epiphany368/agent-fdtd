#!/usr/bin/env python3
"""Leg 14, `V1-grating-zero-order`: the grating readout on a zero-modulation grating.

The leg-9 driver reports `zero_order_efficiency`, `zero_order_efficiency_a/_b` and
`zero_order_forward_power` through `torchfdtd.radiation.diffraction_orders` and
`diffraction_efficiency` on a Bloch cell. No closed form exists for the leg-9 binary grating
(see `analytic_cases.LEG14_DROPPED`). A closed form DOES exist when the modulation is zero: the
cell then holds one uniform film, only the (0,0) order propagates, and its efficiency is the
oblique p-polarised film transmittance (Born and Wolf, section 1.6.4).

This driver keeps everything of leg 9 (the same `leg7_campaign.build_project` Bloch cell, the
same source, monitors, order list, readout calls and the same two angles 0.4002 degrees apart)
and changes only the structure: a uniform film of eps 4 and thickness 0.6 micrometres, the
leg-9 tooth's permittivity and slab height, filling the whole period.

The film edges sit on z nodes. Tangential components (Ex, Ey) take the arithmetic mean of the
cell they straddle and the normal component (Ez) the harmonic mean, the standard sub-pixel
rule for a planar interface (Farjadpour et al., Opt. Lett. 31, 2972 (2006)).

Run:  ./tools/run.sh configs/leg14-grating-zero-order.json src/leg14_grating.py
"""
import math
import time

from parity import *  # noqa: F401,F403
import numpy as np
import torch
from torchfdtd import AdjointOptions, DifferentiablePlaneSimulation
from torchfdtd.radiation import diffraction_orders, diffraction_efficiency
from torchfdtd.solver import C0, field_axes

import leg7_campaign as l7

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'leg14_grating.film_epsilon (a zero-modulation grating: one uniform film)',
    'F2-mesh': 'leg7_campaign.build_project (uniform Region, as leg 9)',
    'F3-source': 'leg7_campaign.build_project (soft plane source, Ex, Bloch phase set per angle)',
    'F4-monitors': 'leg7_campaign.build_project (transmitted and reflected planes)',
    'F5-run': 'torchfdtd.DifferentiablePlaneSimulation',
    'F6-readout': 'torchfdtd.radiation.diffraction_orders and diffraction_efficiency (as leg 9)',
}
ORDERS = [[-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0]]
ZERO = 2


def _fill(z, h, half):
    """Fraction of the cell [z - h/2, z + h/2] inside |z| <= half."""
    return np.clip(np.minimum(z + h / 2, half) - np.maximum(z - h / 2, -half), 0, None) / h


def film_epsilon(project, film, h):
    eps = torch.ones(tuple(project.region.shape) + (3,), dtype=torch.float64)
    half = film['thickness_um'] / 2
    for index, component in enumerate(('Ex', 'Ey', 'Ez')):
        z = np.asarray(field_axes(project.region, component)[2], dtype=np.float64)
        f = _fill(z, h, half)
        if component == 'Ez':
            value = 1.0 / ((1.0 - f) + f / film['eps'])
        else:
            value = 1.0 + (film['eps'] - 1.0) * f
        eps[..., index] = torch.as_tensor(value, dtype=torch.float64)[None, None, :]
    return eps


def run(cfg, phase_rad, epsilon, frequency):
    project = l7.build_project(cfg, phase_rad)
    quadrature = (round(cfg['period_um'] / cfg['mesh_um']), round(cfg['slab_um'] / cfg['mesh_um']))
    model = DifferentiablePlaneSimulation(project, AdjointOptions(checkpoints=4),
                                          quadrature_counts={'transmitted': quadrature,
                                                             'reflected': quadrature})
    started = time.perf_counter()
    with torch.no_grad():
        planes = model(epsilon, [frequency])
    torch.cuda.synchronize()
    return planes, time.perf_counter() - started, project


def phase_for(cfg, angle_deg):
    k0 = 2.0 * math.pi / cfg['wavelength_um']
    return k0 * math.sin(math.radians(angle_deg)) * cfg['period_um']


def angle_pair(cfg):
    frequency = C0 / (cfg['wavelength_um'] * 1e-6)
    out = {}
    for label, angle in (('a', cfg['angle_a_deg']), ('b', cfg['angle_a_deg'] + cfg['drift_deg'])):
        phase = phase_for(cfg, angle)
        project = l7.build_project(cfg, phase)
        background = torch.ones(tuple(project.region.shape) + (3,), dtype=torch.float64)
        film = film_epsilon(project, cfg['film'], cfg['mesh_um'])
        empty, empty_seconds, _ = run(cfg, phase, background, frequency)
        sample, sample_seconds, project = run(cfg, phase, film, frequency)
        kw = dict(period_um=(cfg['period_um'], cfg['slab_um']),
                  bloch_wavevector_per_um=(phase / cfg['period_um'], 0.0),
                  refractive_index=cfg['exterior_index'])
        forward = diffraction_orders(sample['transmitted'], ORDERS, **kw).forward_power
        forward = forward.detach().cpu().numpy().reshape(-1)
        efficiency = diffraction_efficiency(sample['transmitted'], empty['transmitted'], ORDERS,
                                            direction='forward', **kw)
        efficiency = efficiency.detach().cpu().numpy().reshape(-1)
        out[label] = dict(angle_deg=angle, zero_order_efficiency=float(efficiency[ZERO]),
                          zero_order_forward_power=float(forward[ZERO]),
                          other_orders_efficiency=float(np.sum(np.abs(efficiency))
                                                        - abs(efficiency[ZERO])),
                          shape=list(project.region.shape), steps=project.region.steps,
                          seconds=empty_seconds + sample_seconds)
        print(json.dumps(dict(mesh_um=cfg['mesh_um'], label=label, **out[label])), flush=True)
    return out


def main():
    cfg, run_dir = load_run()
    ladder = {}
    fixture = None
    for mesh in cfg['ladder_mesh_um']:
        local = dict(cfg, mesh_um=mesh, steps=int(round(cfg['steps_at_0p02'] * 0.02 / mesh)))
        ladder[f'{mesh:g}'] = angle_pair(local)
        if mesh == cfg['fixture_mesh_um']:
            fixture = ladder[f'{mesh:g}']
    a, b = fixture['a'], fixture['b']
    readout = dict(
        policy=cfg['policy'], status='completed', simulation_executed=True,
        fixture=('Zero-modulation grating (one uniform film, eps 4, 0.6 um) inside the leg-9 '
                 'Bloch cell, period 0.9 um, p polarisation, two angles 0.4002 degrees apart'),
        stages=list(STAGES), stage_source=dict(STAGE_SOURCE),
        cases=[dict(id='film-grating-angle-pair', ppw=round(cfg['wavelength_um']
                                                            / (2.0 * cfg['fixture_mesh_um'])),
                    shape=a['shape'], steps=a['steps'], seconds=a['seconds'] + b['seconds'],
                    measured=dict(zero_order_efficiency=a['zero_order_efficiency'],
                                  zero_order_efficiency_a=a['zero_order_efficiency'],
                                  zero_order_efficiency_b=b['zero_order_efficiency'],
                                  zero_order_forward_power=a['zero_order_forward_power']),
                    recorded=dict(ladder=ladder))],
        note=('zero_order_forward_power is reported, NOT judged: it is an absolute power in solver '
              'units and its normalisation is S4\'s question. Only the efficiencies are judged.'))
    write_json(Path(run_dir) / 'readout.json', readout)
    import analytic_cases
    checks = analytic_cases.evaluate_analytic_cases(readout, cfg)
    for check in checks:
        print(json.dumps(check), flush=True)
    return 0 if checks and all(c['status'] == 'passed' for c in checks) else 52


if __name__ == '__main__':
    sys.exit(main())
