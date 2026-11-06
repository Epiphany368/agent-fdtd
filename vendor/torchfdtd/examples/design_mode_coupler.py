"""Mode-port offset-guide coupler designed through DesignProblem, CPU only.

Two opposing fixed-mode ports (ModeNetwork) sit on two slab guides of
epsilon 4 in a cladding of 2.25, 0.6 um wide, offset by 0.6 um from each
other (the left guide spans y in (0, 0.6) um, the right guide y in
(-0.6, 0) um), in a thin periodic slab of five cells along z, so the
problem is two-dimensional. Between the ports a design box of 6 by 10 pixels
of 200 nm must couple the two guides; its density selects cladding or core
material per pixel under a 180-degree rotation symmetry. The objective is the
modal transmission |S21|^2 at 1.55 um; the transmission at the holdout
wavelength 1.50 um is evaluated once on the final GDS design and never
optimized. The final evaluation rebuilds the network at half the mesh and
twice the steps.

Run: python -m examples.design_mode_coupler --seed 1 --iterations 16 --output results/coupler-seed1.json
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from torchfdtd import (AdjointOptions, Boundaries, BoundaryFace, FixedModePort, Material, ModeNetwork, Project, Region,
                       Source)
from torchfdtd.density_layer import bounded_density_layer
from torchfdtd.design_parameterization import DensityParameterization
from torchfdtd.design_problem import Continuation, DesignProblem
from torchfdtd.solver import voxelize

LENGTH_UM, PERIOD_UM, SLAB_UM = 6.4, 2.4, 1.
CORE_EPSILON, CLADDING_EPSILON = 4., 2.25
GUIDE_WIDTH_UM = .6
GUIDE_OFFSET_UM = .3          # left guide centred at +0.3 um, right guide at -0.3 um
WAVELENGTH_UM, HOLDOUT_UM = 1.55, 1.5
PORT_UM, SOURCE_UM = 1., 1.8
BOX_UM = (-.6, .6, -1., 1.)   # x_min, x_max, y_min, y_max of the design box
PIXELS = (6, 10)
PIXEL_UM = .2
MATERIAL = 'core'


def section(centre_um):
    """Fixed port cross-section of one offset guide in the port's cyclic (y, z) coordinates."""
    return lambda u, v: np.where(abs(u-centre_um) < GUIDE_WIDTH_UM/2-1e-9, CORE_EPSILON, CLADDING_EPSILON)


SECTIONS = dict(left=section(GUIDE_OFFSET_UM), right=section(-GUIDE_OFFSET_UM))


def build_project(mesh_um, steps, wavelength_um):
    boundaries = Boundaries(**{axis+'_'+side: BoundaryFace(kind='periodic') for axis in 'yz' for side in ('min', 'max')})
    return Project(region=Region(dimension='3d', size=(LENGTH_UM, PERIOD_UM, SLAB_UM), mesh=mesh_um, pml_cells=round(1./mesh_um),
                                 steps=steps, material_sampling='yee', boundaries=boundaries, precision='float32',
                                 background_index=CLADDING_EPSILON**.5),
                   materials=[Material(name=MATERIAL, index=CORE_EPSILON**.5)],
                   sources=[Source(id='plane', kind='plane', normal='x', center=(-SOURCE_UM, 0., 0.), size=(0., PERIOD_UM, SLAB_UM),
                                   wavelength=wavelength_um, pulse_cycles=2)], monitors=[])


class CouplerForward:
    """Two-port network at one mesh and wavelength with its straight-guide reference."""
    def __init__(self, mesh_um, steps, wavelength_um=WAVELENGTH_UM):
        self.project = build_project(mesh_um, steps, wavelength_um)
        ports = (FixedModePort('left', -PORT_UM, -SOURCE_UM, 1), FixedModePort('right', PORT_UM, SOURCE_UM, -1))
        self.network = ModeNetwork(self.project, ports, options=AdjointOptions(checkpoints=4), num_modes=1,
                                   port_permittivities=SECTIONS)
        region = self.project.region
        # The fixed exterior is the left guide before the design box and the right guide after it.
        self.base = self.network.reference_epsilon(port='left')
        split = int(round((0.-region.mesh_nodes[0][0])/mesh_um))
        self.base[split:] = self.network.reference_epsilon(port='right')[split:]
        self.box = tuple(slice(int(round((lo-region.mesh_nodes[axis][0])/mesh_um)), int(round((hi-region.mesh_nodes[axis][0])/mesh_um)))
                         for axis, (lo, hi) in enumerate((BOX_UM[:2], BOX_UM[2:])))

    def epsilon(self, density):
        return bounded_density_layer(density, self.base, self.project.region, bounds_um=BOX_UM, bottom_um=-SLAB_UM/2,
                                     top_um=SLAB_UM/2, background_epsilon=CLADDING_EPSILON, design_epsilon=CORE_EPSILON)

    def structure_epsilon(self, structures):
        project = self.project.model_copy(deep=True)
        project.structures = list(structures)
        voxels, _ = voxelize(project)
        epsilon = self.base.clone()
        x, y = self.box
        epsilon[x, y] = torch.as_tensor(np.ascontiguousarray(voxels[x, y]), dtype=epsilon.dtype)
        return epsilon

    def metrics(self, result):
        s = result.s
        return dict(transmission=s[1, 0].abs().square(), reflection=s[0, 0].abs().square(),
                    reverse_transmission=s[0, 1].abs().square())

    def objective(self, density):
        result = self.network(self.epsilon(density))
        return -result.s[1, 0].abs().square(), self.metrics(result)


class FineEvaluator:
    """Independent networks at half the mesh and twice the steps, at the design and holdout wavelengths."""
    def __init__(self, mesh_um, steps):
        self.forward = CouplerForward(mesh_um, steps)
        self.holdout = CouplerForward(mesh_um, steps, HOLDOUT_UM)

    @torch.no_grad()
    def evaluate_density(self, density):
        return self.forward.metrics(self.forward.network(self.forward.epsilon(density)))

    @torch.no_grad()
    def evaluate_structures(self, structures):
        return self.forward.metrics(self.forward.network(self.forward.structure_epsilon(structures)))

    @torch.no_grad()
    def evaluate_holdout(self, structures):
        metrics = self.holdout.metrics(self.holdout.network(self.holdout.structure_epsilon(structures)))
        return {'holdout_'+key: value for key, value in metrics.items()}


def build_problem(seed, *, mesh_um=.2, steps=500, learning_rate=.1, beta=4., continuation_every=4):
    forward = CouplerForward(mesh_um, steps)
    generator = torch.Generator().manual_seed(seed)
    initial = .5*torch.randn(PIXELS, generator=generator)
    design = DensityParameterization(PIXELS, spacing_um=PIXEL_UM, initial=initial, mode='logits',
                                     filter_radius_um=2*PIXEL_UM, boundary='truncate', symmetry='rotate180', beta=beta, eta=.5)
    optimizer = torch.optim.Adam(design.parameters(), lr=learning_rate)
    problem = DesignProblem(design, forward.objective, optimizer, name=f'coupler-seed{seed}',
                            continuation=Continuation(every=continuation_every, factor=2., maximum=64.))
    return problem, forward


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--mesh', type=float, default=.2)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export-dir', type=Path, default=Path('results/coupler'))
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
                                      perturbation_um=PIXEL_UM, boundary='extend')
    exported = problem.export(args.export_dir/f'seed{args.seed}', origin_um=(BOX_UM[0], BOX_UM[2]), spacing_um=PIXEL_UM,
                              z_min_um=-SLAB_UM/2, z_max_um=SLAB_UM/2, material=MATERIAL)
    record = dict(example='coupler', seed=args.seed, iterations=problem.iteration,
                  declared=dict(length_um=LENGTH_UM, period_um=PERIOD_UM, slab_um=SLAB_UM, core_epsilon=CORE_EPSILON,
                                cladding_epsilon=CLADDING_EPSILON, guide_width_um=GUIDE_WIDTH_UM, guide_offset_um=GUIDE_OFFSET_UM,
                                wavelength_um=WAVELENGTH_UM,
                                holdout_um=HOLDOUT_UM, port_um=PORT_UM, source_um=SOURCE_UM, box_um=list(BOX_UM), pixels=list(PIXELS),
                                pixel_um=PIXEL_UM, coarse_mesh_um=args.mesh, coarse_steps=args.steps, fine_mesh_um=args.mesh/2,
                                fine_steps=2*args.steps, filter_radius_um=2*PIXEL_UM, symmetry='rotate180', learning_rate=.1,
                                beta_initial=4., continuation='x2 every 4 steps to 64',
                                objective='modal transmission |S21|^2 at 1.55 um, maximized',
                                holdout='|S21|^2 at 1.50 um of the final GDS design, never optimized'),
                  history=history, fabrication=fabrication, binary=problem.density(hard=True).int().tolist(),
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
    final = record.get('final_evaluation', {})
    summary = dict(seed=args.seed, iterations=record['iterations'],
                   last_objective=history[-1]['objective'] if history else None,
                   fine_gds=final.get('stages', {}).get('fine_gds'), holdout=final.get('holdout'),
                   violations=fabrication['violations'], wall_time_s=round(record['wall_time_s'], 1))
    print(json.dumps(summary))
    return record


if __name__ == '__main__':
    main()
