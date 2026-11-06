"""Leg 9 part 2: the grating's OWN angle sensitivity, with a run.

Part 1 measured the sensitivity of a closed-form resonant slab and found it small (`2.40e-03`
absolute, inside every tolerance here). That result is **not** transferred to a grating: a grating
coupler is sharper in angle. This driver runs the grating at two incidence angles `0.4002` degrees
apart — the drift leg 7 measured over `1.50`-`1.60` µm — and reports the change in the zero-order
efficiency.

Run:  ./tools/run.sh configs/leg9-grating-angle.json src/leg9_grating.py
"""
import math
import time

from parity import *
import numpy as np
import torch
from torchfdtd import AdjointOptions, DifferentiablePlaneSimulation
from torchfdtd.radiation import diffraction_orders, diffraction_efficiency
from torchfdtd.solver import C0, field_axes

import leg7_campaign as l7

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'leg9_grating.grating_epsilon (a binary grating inside the Bloch cell)',
    'F2-mesh': 'leg7_campaign.build_project (uniform Region)',
    'F3-source': 'leg7_campaign.build_project (soft plane source, Bloch phase set per angle)',
    'F4-monitors': 'leg7_campaign.build_project (transmitted and reflected planes)',
    'F5-run': 'torchfdtd.DifferentiablePlaneSimulation',
    'F6-readout': 'torchfdtd.radiation.diffraction_orders and diffraction_efficiency',
}
ORDERS = [[-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0]]
ZERO = 2  # the index of order (0, 0) inside ORDERS


def grating_epsilon(project, grating, step):
    """A binary grating: tooth of eps_tooth, gap of vacuum, inside a z slab."""
    period = float(np.ptp(np.asarray(field_axes(project.region, 'Ex')[0])))
    eps = torch.ones(tuple(project.region.shape) + (3,), dtype=torch.float64)
    for component_index, component in enumerate(('Ex', 'Ey', 'Ez')):
        x = np.asarray(field_axes(project.region, component)[0], dtype=np.float64)
        fill = np.zeros_like(x)
        for cell in range(int(round(period / grating['fill_period_um']))):
            start = -period / 2 + cell * grating['fill_period_um']
            fill = fill + np.clip(np.minimum(x + step / 2, start + grating['tooth_um'])
                                  - np.maximum(x - step / 2, start), 0, None) / step
        z = np.asarray(field_axes(project.region, component)[2], dtype=np.float64)
        half = grating['slab_z_um'] / 2
        zmask = ((z >= -half) & (z <= half)).astype(np.float64)
        eps[..., component_index] = torch.as_tensor(
            1.0 + (grating['eps_tooth'] - 1.0) * fill[:, None, None] * zmask[None, None, :],
            dtype=torch.float64)
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
    return planes, time.perf_counter() - started


def phase_for(cfg, angle_deg):
    k0 = 2.0 * math.pi / cfg['wavelength_um']
    return k0 * math.sin(math.radians(angle_deg)) * cfg['period_um']


def main():
    cfg, run_dir = load_run()
    frequency = C0 / (cfg['wavelength_um'] * 1e-6)
    angle_a = cfg['angle_a_deg']
    angle_b = angle_a + cfg['drift_deg']
    results = {}
    for label, angle in (('a', angle_a), ('b', angle_b)):
        phase = phase_for(cfg, angle)
        project = l7.build_project(cfg, phase)
        background = torch.ones(tuple(project.region.shape) + (3,), dtype=torch.float64)
        grating = grating_epsilon(project, cfg['grating'], cfg['mesh_um'])
        empty_planes, empty_seconds = run(cfg, phase, background, frequency)
        grating_planes, grating_seconds = run(cfg, phase, grating, frequency)
        kw = dict(period_um=(cfg['period_um'], cfg['slab_um']),
                  bloch_wavevector_per_um=(phase / cfg['period_um'], 0.0),
                  refractive_index=cfg['exterior_index'])
        forward = diffraction_orders(grating_planes['transmitted'], ORDERS, **kw).forward_power
        forward = forward.detach().cpu().numpy().reshape(-1)
        efficiency = diffraction_efficiency(
            grating_planes['transmitted'], empty_planes['transmitted'], ORDERS,
            direction='forward', **kw).detach().cpu().numpy().reshape(-1)
        results[label] = dict(angle_deg=angle, phase_rad=phase,
                              k_parallel_per_um=phase / cfg['period_um'],
                              zero_order_forward_power=float(forward[ZERO]),
                              zero_order_efficiency=float(efficiency[ZERO]),
                              forward_total=float(forward.sum()),
                              empty_seconds=empty_seconds, grating_seconds=grating_seconds)
        print(json.dumps(results[label]), flush=True)

    a, b = results['a'], results['b']
    separation = b['angle_deg'] - a['angle_deg']
    change = b['zero_order_efficiency'] - a['zero_order_efficiency']
    relative = abs(change) / max(abs(a['zero_order_efficiency']), 1e-30)
    verdict = ('below analytic tolerance' if abs(change) <= cfg['tolerance_analytic']
               else 'below the known discrepancy only' if abs(change) <= cfg['tolerance_known_discrepancy']
               else 'above both')
    readout = dict(
        policy=cfg['policy'], status='completed', simulation_executed=True,
        fixture=('Binary grating inside a Bloch cell, period 0.9 micrometres, mesh 0.02, at two '
                 'incidence angles 0.4002 degrees apart.'),
        stages=list(STAGES), stage_source=dict(STAGE_SOURCE),
        cases=[dict(id='grating-angle-pair', ppw=round(cfg['period_um'] / cfg['mesh_um']),
                    steps=cfg['steps'], dt_s=0.0, shape=[len(ORDERS)],
                    measured=dict(angle_a_deg=a['angle_deg'], angle_b_deg=b['angle_deg'],
                                  realised_separation_deg=separation,
                                  zero_order_efficiency_a=a['zero_order_efficiency'],
                                  zero_order_efficiency_b=b['zero_order_efficiency'],
                                  absolute_change=change, relative_change=relative,
                                  tolerance_analytic=cfg['tolerance_analytic'],
                                  tolerance_known_discrepancy=cfg['tolerance_known_discrepancy']))],
        blocks=[a, b], verdict=verdict,
        note=('Part 2 of leg 9. Zero order only. The closed-form slab result from part 1 is NOT '
              'transferred here; this is the grating\'s own measurement.'))
    write_json(Path(run_dir) / 'readout.json', readout)
    print(f'realised separation = {separation:.6f} deg')
    print(f'zero-order efficiency: {a["zero_order_efficiency"]:.6f} -> '
          f'{b["zero_order_efficiency"]:.6f}; change = {change:+.6e} absolute, '
          f'{relative:.4e} relative')
    print(f'verdict: {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
