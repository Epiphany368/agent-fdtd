"""Metagrating deflector designed through DesignProblem, CPU only.

A one-dimensional density (30 pixels of 50 nm across a 1.5 um period) of a
0.5 um high TiO2-like layer (epsilon 4) in air deflects a normally incident
plane wave of 1.0 um into the +1 transmitted order. The cell is a thin
periodic slab of five cells along y, so the problem is two-dimensional. The
objective is the +1 efficiency at 1.0 um; the efficiency at the holdout
wavelength 0.95 um is recorded from the same run but never optimized. The
final evaluation rebuilds the forward at half the mesh and twice the steps
and evaluates the smooth density, the thresholded density, the exported
rectangles and the GDS polygons read back through import_gds.

Run: python -m examples.design_metagrating --seed 1 --iterations 24 --output results/metagrating-seed1.json
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from torchfdtd import (AdjointOptions, Boundaries, BoundaryFace, DifferentiablePlaneSimulation, FieldMonitor, Material,
                       Project, Region, Source, periodic_density_layer)
from torchfdtd.design_parameterization import DensityParameterization
from torchfdtd.design_problem import Continuation, DesignProblem
from torchfdtd.radiation import diffraction_efficiency
from torchfdtd.solver import C0, voxelize

PERIOD_UM = 1.5
SLAB_UM = .25          # thin invariant y extent, five coarse cells
HEIGHT_UM = .5
WAVELENGTH_UM = 1.
HOLDOUT_UM = .95
DESIGN_EPSILON = 4.
PIXELS = 30
PIXEL_UM = PERIOD_UM/PIXELS
SOURCE_Z_UM, MONITOR_Z_UM, HALF_Z_UM = -1.2, 1.2, 1.8
ORDERS = [[1, 0], [0, 0], [-1, 0]]
FREQUENCIES_HZ = [C0/(WAVELENGTH_UM*1e-6), C0/(HOLDOUT_UM*1e-6)]
MATERIAL = 'tio2-like'


def build_project(mesh_um, steps):
    boundaries = Boundaries(**{axis+'_'+side: BoundaryFace(kind='periodic') for axis in 'xy' for side in ('min', 'max')})
    return Project(region=Region(dimension='3d', size=(PERIOD_UM, SLAB_UM, 2*HALF_Z_UM), mesh=mesh_um, pml_cells=round(.4/mesh_um),
                                 steps=steps, material_sampling='yee', boundaries=boundaries, precision='float32'),
                   materials=[Material(name=MATERIAL, index=math.sqrt(DESIGN_EPSILON))],
                   sources=[Source(id='plane', kind='plane', normal='z', size=(PERIOD_UM, SLAB_UM, 0), center=(0, 0, SOURCE_Z_UM),
                                   component='Ex', wavelength=WAVELENGTH_UM, pulse_cycles=1)],
                   monitors=[FieldMonitor(id='transmitted', normal='z', size=(PERIOD_UM, SLAB_UM, 0), center=(0, 0, MONITOR_Z_UM))])


class GratingForward:
    """Plane-wave forward with a cached background reference at one mesh."""
    def __init__(self, mesh_um, steps, quadrature):
        self.project = build_project(mesh_um, steps)
        self.model = DifferentiablePlaneSimulation(self.project, AdjointOptions(checkpoints=4),
                                                   quadrature_counts={'transmitted': quadrature})
        with torch.no_grad():
            background = torch.ones(self.project.region.shape+(3,))
            self.reference = self.model(background, FREQUENCIES_HZ)['transmitted']

    def epsilon(self, density):
        return periodic_density_layer(density, self.project.region, bottom_um=-HEIGHT_UM/2, top_um=HEIGHT_UM/2,
                                      background_epsilon=1., design_epsilon=DESIGN_EPSILON)

    def efficiencies(self, epsilon):
        plane = self.model(epsilon, FREQUENCIES_HZ)['transmitted']
        return diffraction_efficiency(plane, self.reference, ORDERS, period_um=(PERIOD_UM, SLAB_UM), refractive_index=1.)

    def metrics(self, efficiencies):
        return dict(efficiency=efficiencies[0, 0], holdout_efficiency=efficiencies[1, 0],
                    zero_order=efficiencies[0, 1], minus_one=efficiencies[0, 2])

    def objective(self, density):
        efficiencies = self.efficiencies(self.epsilon(density))
        return -efficiencies[0, 0], self.metrics(efficiencies)


class FineEvaluator:
    """Independent forward at half the mesh and twice the steps."""
    def __init__(self, mesh_um, steps):
        self.forward = GratingForward(mesh_um, steps, (round(PERIOD_UM/mesh_um), round(SLAB_UM/mesh_um)))

    @torch.no_grad()
    def evaluate_density(self, density):
        return self.forward.metrics(self.forward.efficiencies(self.forward.epsilon(density)))

    @torch.no_grad()
    def evaluate_structures(self, structures):
        project = self.forward.project.model_copy(deep=True)
        project.structures = list(structures)
        epsilon, _ = voxelize(project)
        return self.forward.metrics(self.forward.efficiencies(torch.as_tensor(np.ascontiguousarray(epsilon), dtype=torch.float32)))


def build_problem(seed, *, mesh_um=.05, steps=800, learning_rate=.1, beta=4., continuation_every=4):
    forward = GratingForward(mesh_um, steps, (PIXELS, round(SLAB_UM/mesh_um)))
    generator = torch.Generator().manual_seed(seed)
    initial = .5*torch.randn((PIXELS, 1), generator=generator)
    design = DensityParameterization((PIXELS, 1), spacing_um=(PIXEL_UM, SLAB_UM), initial=initial, mode='logits',
                                     filter_radius_um=2.5*PIXEL_UM, boundary='periodic', beta=beta, eta=.5)
    optimizer = torch.optim.Adam(design.parameters(), lr=learning_rate)
    problem = DesignProblem(design, forward.objective, optimizer, name=f'metagrating-seed{seed}',
                            continuation=Continuation(every=continuation_every, factor=2., maximum=64.))
    return problem, forward


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=24)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--mesh', type=float, default=.05)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export-dir', type=Path, default=Path('results/metagrating'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-fine', action='store_true', help='Skip the finer-mesh final evaluation.')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.iterations < 0 or args.steps < 10 or args.mesh <= 0:
        parser.error('Use nonnegative iterations, at least ten steps and a positive mesh.')
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    problem, forward = build_problem(args.seed, mesh_um=args.mesh, steps=args.steps)
    history = problem.run(args.iterations, checkpoint=args.checkpoint, resume=args.resume)
    fabrication = problem.fabrication(spacing_um=PIXEL_UM, min_linewidth_um=2*PIXEL_UM, min_gap_um=2*PIXEL_UM,
                                      perturbation_um=PIXEL_UM, boundary=('periodic', 'extend'))
    exported = problem.export(args.export_dir/f'seed{args.seed}', origin_um=(-PERIOD_UM/2, -SLAB_UM/2),
                              spacing_um=(PIXEL_UM, SLAB_UM), z_min_um=-HEIGHT_UM/2, z_max_um=HEIGHT_UM/2, material=MATERIAL)
    record = dict(example='metagrating', seed=args.seed, iterations=problem.iteration,
                  declared=dict(period_um=PERIOD_UM, slab_um=SLAB_UM, height_um=HEIGHT_UM, wavelength_um=WAVELENGTH_UM,
                                holdout_um=HOLDOUT_UM, design_epsilon=DESIGN_EPSILON, pixels=PIXELS, pixel_um=PIXEL_UM,
                                coarse_mesh_um=args.mesh, coarse_steps=args.steps, fine_mesh_um=args.mesh/2, fine_steps=2*args.steps,
                                filter_radius_um=2.5*PIXEL_UM, learning_rate=.1, beta_initial=4., continuation='x2 every 4 steps to 64',
                                objective='+1 transmitted order efficiency at 1.0 um, maximized',
                                holdout='+1 transmitted order efficiency at 0.95 um, recorded and never optimized'),
                  history=history, fabrication=fabrication, binary=problem.density(hard=True).squeeze(1).int().tolist(),
                  export=dict(gds=str(exported['gds_path']), structures=str(exported['structure_path']),
                              structure_count=len(exported['structures'])))
    if not args.skip_fine:
        evaluator = FineEvaluator(args.mesh/2, 2*args.steps)
        record['final_evaluation'] = problem.final_evaluation(evaluator, exported)
    record['wall_time_s'] = time.perf_counter()-started
    record['device'] = 'cpu'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=1), encoding='utf-8')
    summary = dict(seed=args.seed, iterations=record['iterations'],
                   last_objective=history[-1]['objective'] if history else None,
                   fine_gds=record.get('final_evaluation', {}).get('stages', {}).get('fine_gds'),
                   violations=fabrication['violations'], wall_time_s=round(record['wall_time_s'], 1))
    print(json.dumps(summary))
    return record


if __name__ == '__main__':
    main()
