"""Leg 7, part 2 of S1: launch the M Bloch cells, combine them, and check the result.

The feasibility study (part 1) sized the Array Scanning Method: `M` Bloch runs per waist, combined
by the midpoint rule over the Brillouin zone,

    U(phi_xi, n) = (1/M) * sum_m A(k_m) U_m(xi) exp(i k_m n d),

with `A(k) = exp(-k^2 w0^2 / 4)` the beam's plane-wave spectrum. This driver runs it on the
homogeneous Bloch cell, where the answer is known: the reconstruction must be the Gaussian itself.

It also measures the frequency drift of a fixed Bloch wavevector, because that drift is the defect
the route exists to address, and it checks that the forward plus backward power does not depend on
the injected angle.

Run:  ./tools/run.sh configs/leg7-asm-campaign.json src/leg7_campaign.py
"""
import math
import time

from parity import *
import numpy as np
import torch
from torchfdtd import AdjointOptions, Boundaries, BoundaryFace, DifferentiablePlaneSimulation, \
    FieldMonitor, Project, Region, Source
from torchfdtd.radiation import diffraction_orders
from torchfdtd.solver import C0

import oblique_asm

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'leg7_campaign.build_project (the homogeneous Bloch cell)',
    'F2-mesh': 'leg7_campaign.build_project (uniform Region)',
    'F3-source': 'leg7_campaign.build_project (soft plane source in a Bloch cell)',
    'F4-monitors': 'leg7_campaign.build_project (transmitted and reflected planes)',
    'F5-run': 'torchfdtd.DifferentiablePlaneSimulation',
    'F6-readout': 'torchfdtd.radiation.diffraction_orders, plus oblique_asm for the ASM weights',
}
ORDERS = [[-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0]]


def build_project(cfg, phase_rad):
    period, slab, half_z = cfg['period_um'], cfg['slab_um'], cfg['half_z_um']
    mesh = cfg['mesh_um']
    transverse = BoundaryFace(kind='bloch')
    boundaries = Boundaries(x_min=transverse, x_max=transverse,
                            y_min=BoundaryFace(kind='periodic'), y_max=BoundaryFace(kind='periodic'),
                            z_min=BoundaryFace(kind='pml', layers=round(cfg['pml_um'] / mesh)),
                            z_max=BoundaryFace(kind='pml', layers=round(cfg['pml_um'] / mesh)))
    region = Region(dimension='3d', size=(period, slab, 2 * half_z), mesh=mesh,
                    material_sampling='yee', boundaries=boundaries,
                    bloch_phase=(phase_rad, 0.0, 0.0), precision='float64', steps=cfg['steps'])
    return Project(name='asm', region=region,
                   sources=[Source(id='plane', kind='plane', normal='z', size=(period, slab, 0),
                                   center=(0, 0, cfg['source_z_um']), component='Ex',
                                   wavelength=cfg['wavelength_um'], pulse_cycles=cfg['pulse_cycles'])],
                   monitors=[FieldMonitor(id='transmitted', normal='z', size=(period, slab, 0),
                                          center=(0, 0, cfg['monitor_z_um'])),
                             FieldMonitor(id='reflected', normal='z', size=(period, slab, 0),
                                          center=(0, 0, -cfg['monitor_z_um']))])


def kw(cfg, phase_rad):
    return dict(period_um=(cfg['period_um'], cfg['slab_um']),
                bloch_wavevector_per_um=(phase_rad / cfg['period_um'], 0.),
                refractive_index=cfg['exterior_index'])


def run_one(cfg, phase_rad, frequency):
    project = build_project(cfg, phase_rad)
    quadrature = (round(cfg['period_um'] / cfg['mesh_um']), round(cfg['slab_um'] / cfg['mesh_um']))
    model = DifferentiablePlaneSimulation(project, AdjointOptions(checkpoints=4),
                                          quadrature_counts={'transmitted': quadrature,
                                                             'reflected': quadrature})
    shape = tuple(project.region.shape) + (3,)
    epsilon = torch.full(shape, cfg['exterior_index'] ** 2, dtype=torch.float64, device='cuda')
    started = time.perf_counter()
    with torch.no_grad():
        planes = model(epsilon, [frequency])
    torch.cuda.synchronize()
    return planes, time.perf_counter() - started


def profile_of(plane):
    """The transverse field profile on the monitor, as a numpy array."""
    fields = plane.fields
    return fields.detach().cpu().numpy().reshape(fields.shape[0], -1)[:, 0]


def main():
    cfg, run = load_run()
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    frequency = C0 / (cfg['wavelength_um'] * 1e-6)
    wavelength_um = cfg['wavelength_um']
    d = cfg['period_um']
    k0 = 2.0 * math.pi / wavelength_um
    w0 = cfg['w0_um']

    # FIX after the first failed attempt: the phase grid is set by ONE M (the registered count),
    # not by the sweep of M values. The first attempt reused each swept M as its own grid size,
    # which produced three samples of three different grids instead of M midpoints of one.
    total = cfg['registered_M']
    rows, weights, phases, profiles = [], [], [], []
    for i in range(1, total + 1):
        m = i
        phase = -math.pi + (i - 0.5) * (2.0 * math.pi / total)
        planes, seconds = run_one(cfg, phase, frequency)
        transmitted, reflected = planes['transmitted'], planes['reflected']
        forward = diffraction_orders(transmitted, ORDERS, **kw(cfg, phase)).forward_power
        backward = diffraction_orders(reflected, ORDERS, **kw(cfg, phase)).backward_power
        forward_total = float(forward.detach().cpu().numpy().sum())
        backward_total = float(backward.detach().cpu().numpy().sum())
        k_m = phase / d
        weight = math.exp(-(k_m ** 2) * w0 ** 2 / 4.0)
        weights.append(weight)
        phases.append(phase)
        # Normalise the response to the power this angle actually delivered. A soft source does
        # not inject the same power at every angle (measured here: 39 percent spread), so without
        # this the sum is weighted by the source's angle response as well as by the beam spectrum.
        # The ASM sum needs the BLOCH-PERIODIC part u_k(xi), not the total field. The monitor
        # returns U = exp(i k x) u_k(x), so the intra-cell Bloch phase exp(i k xi) must be removed
        # before summing; only the CELL-index phase exp(i k n d) belongs in the reconstruction.
        # Summing the total field instead leaves an extra position-dependent phase per k, which no
        # amount of summing over k can turn into a beam. This is the defect the first campaign had.
        profile = profile_of(transmitted)
        xi = np.linspace(-d / 2, d / 2, len(profile), endpoint=False)
        periodic_part = profile * np.exp(-1j * k_m * xi)
        amplitude = periodic_part / math.sqrt(forward_total + backward_total)
        profiles.append(amplitude)
        rows.append(dict(m=m, phase_rad=phase, k_per_um=k_m, weight=weight, seconds=seconds,
                         forward_total=forward_total, backward_total=backward_total,
                         delivered_total=forward_total + backward_total))
        print(json.dumps(rows[-1]), flush=True)

    # FIX after the first failed attempt: the ASM sum carries a phase per CELL INDEX. The first
    # attempt summed only the in-cell profile (cell index zero), so it never built a beam.
    n_points = len(profiles[0])
    xi_um = np.linspace(-d / 2, d / 2, n_points, endpoint=False)
    half_cells = int(math.ceil(2.0 * w0 / d))
    grid_x, grid_u = [], []
    for n in range(-half_cells, half_cells + 1):
        for j in range(n_points):
            value = sum(weights[idx] * profiles[idx][j] * np.exp(1j * phases[idx] * n)
                        for idx in range(len(rows))) / len(rows)
            grid_x.append(n * d + xi_um[j])
            grid_u.append(value)
    order = np.argsort(grid_x)
    grid_x = np.asarray(grid_x)[order]
    grid_u = np.asarray(grid_u)[order]
    reference = np.exp(-(grid_x ** 2) / w0 ** 2)
    reference = reference / np.abs(reference).max()
    measured = np.abs(grid_u) / (np.abs(grid_u).max() + 1e-300)
    reconstruction_error = float(
        np.sqrt(np.mean((measured - reference) ** 2)) / np.sqrt(np.mean(reference ** 2)))
    profile_points = len(grid_x)

    delivered = [row['delivered_total'] for row in rows]
    energy_spread = max(delivered) - min(delivered)
    energy_spread_relative = energy_spread / (sum(delivered) / len(delivered))

    # The frequency drift of a FIXED Bloch wavevector: the defect the route exists to address.
    k_fixed = phases[len(phases) // 2] / d
    band = cfg['band_um']
    angles = [math.degrees(math.asin(min(1.0, k_fixed / (2.0 * math.pi / lam)))) for lam in band]
    drift = dict(
        k_fixed_per_um=k_fixed, band_um=band, angle_deg=angles,
        drift_deg=angles[1] - angles[0],
        statement=('A fixed Bloch wavevector holds k_parallel constant, so the propagation angle '
                   'falls as the wavelength rises. The ASM sum uses k-independent weights, so it '
                   'does NOT remove this drift: it only makes the beam finite. The drift is the '
                   'route\'s error bound across the band and must be compensated by a '
                   'frequency-dependent weight, or accepted.'))

    readout = dict(
        policy=cfg['policy'], status='completed', simulation_executed=True,
        fixture=('Homogeneous Bloch cell, period 0.9 micrometres, normal z launch, coarse mesh. '
                 'The Array Scanning Method sum over the Brillouin zone.'),
        stages=list(STAGES), stage_source=dict(STAGE_SOURCE),
        cases=[dict(id='asm-campaign', ppw=round(d / cfg['mesh_um']), steps=cfg['steps'],
                    dt_s=0.0, shape=[profile_points], measured=dict(
                        registered_M=len(rows), reconstruction_error=reconstruction_error,
                        tolerance_reconstruction=cfg['tolerance_reconstruction'],
                        energy_spread=energy_spread, energy_spread_relative=energy_spread_relative,
                        profile_points=profile_points,
                        tolerance_energy=cfg['tolerance_energy'],
                        peak_weight=max(weights), min_weight=min(weights),
                        drift_deg=drift['drift_deg']))],
        blocks=rows, drift=drift,
        note=('Part 2 of leg 7. The reconstruction is compared in the monitor plane, not on a far '
              'plane: propagating to a far plane is separate work and is not claimed here.'))
    write_json(Path(run) / 'readout.json', readout)
    print(f'reconstruction error = {reconstruction_error:.4e} '
          f'(tolerance {cfg["tolerance_reconstruction"]:g})', flush=True)
    print(f'energy spread over the swept phases = {energy_spread:.4e} absolute, '
          f'{energy_spread_relative:.4e} relative (tolerance {cfg["tolerance_energy"]:g} relative)',
          flush=True)
    print(f'fixed-wavevector drift over {band[0]}-{band[1]} um = {drift["drift_deg"]:.4f} deg',
          flush=True)
    failed = (reconstruction_error > cfg['tolerance_reconstruction']
              or energy_spread_relative > cfg['tolerance_energy'])
    print(f'leg7 part 2: {"FAILED" if failed else "passed"}', flush=True)
    return 55 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
