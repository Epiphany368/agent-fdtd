"""Leg 10: the true 2-D path, in this repository, for the first time.

Every driver here so far builds a **3-D region with one layer or a few periodic layers** in the
invariant axis. That is a *fake* 2-D: it makes a two-dimensional problem look three-dimensional, at
the cost of the extra cells. The fork has a **true 2-D** path (commit `2cfb82b`, a native
invariant-axis Yee eigensolve), and this repository has never used it.

The reference project already established that the two agree: true 2-D against its own five-cell 3-D
arm at **1e-13** for a z-invariant fixture, with the 3-D path bit-identical. **What this repository
lacks is the path, not the evidence.** This leg brings the path here and checks it against this
repository's own recorded 3-D number.

Run:  ./tools/run.sh configs/leg10-true-2d.json src/leg10_true_2d.py
"""
from parity import *
import numpy as np
import torch
from torchfdtd import AdjointOptions, Boundaries, BoundaryFace, DifferentiablePlaneSimulation, \
    Monitor, Project, Region, Source
from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_plane_amplitudes
from torchfdtd.mode_branches import prepare_aperture_modal_launch

from run_b9_native_shape_n8 import scene

STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
STAGE_SOURCE = {
    'F1-structure': 'run_b9_native_shape_n8.scene (the N8 grating, z-invariant by construction)',
    'F2-mesh': 'leg10_true_2d.to_true_2d (the same x/y mesh, the invariant axis reduced)',
    'F3-source': 'torchfdtd.mode_branches.prepare_aperture_modal_launch (the 2-D branch)',
    'F4-monitors': 'run_b9_native_shape_n8.scene (unchanged)',
    'F5-run': 'torchfdtd.mode_injection.ModeInjectedPlaneSimulation (the 2-D path)',
    'F6-readout': 'torchfdtd.mode_injection.modal_plane_amplitudes',
}


def to_true_2d(project, time_step):
    """Reduce the invariant axis before anything else runs.

    The recipe the reference project uses: switch the dimension, pin the time step so the physical
    duration is unchanged, and set the invariant-axis boundaries to PML (a 2-D region refuses a
    non-PML or Bloch phase on that axis). The retained z span is metadata; nothing integrates it as
    a thickness.
    """
    dump = project.model_dump()
    region = dump['region']
    region.update(dimension='2d', time_step_override=time_step)
    region['boundaries'].update(z_min=dict(kind='pml'), z_max=dict(kind='pml'))
    reduced = Project.model_validate(dump)
    return reduced, reduced.region


def launch_for(project, section):
    modes = [prepare_aperture_modal_launch(project, section, mode_index=i, num_modes=2)
             for i in range(2)]
    scores = [float(np.sum(abs(m.mode.fields[..., 1]) ** 2)
                    / np.sum(abs(m.mode.fields[..., [1, 2]]) ** 2)) for m in modes]
    launch = modes[int(np.argmax(scores))]
    if max(scores) <= 1 - 1e-8 or launch.mode.maxwell_residual >= 1e-5:
        raise SystemExit('the launch eigenmode is not resolvable in true 2-D')
    return launch


SOURCE_COMBINATIONS = (
    ('point', 'soft'), ('plane', 'soft'), ('plane', 'oneway'), ('tfsf', 'oneway'))


def homogeneous_pair(cfg, transverse='periodic', width_um=None, pml_cells=32):
    """The minimal cell, built by MIRRORING `parity.project_for` rather than by hand.

    The first attempt hand-built a Region and the validator refused it four times, which read like
    a 2-D source refusal and was not one. This version copies the parameter shape of the
    construction this repository already runs: a 1 micrometre invariant extent carrying exactly one
    cell, `mesh_steps=(dx, dx, 1)`, Yee sampling, staircase interfaces, periodic transverse faces.
    """
    dx = 0.02
    width = 6 * dx if width_um is None else width_um
    span = 4.0
    region = Region(size=(span, width, 1), mesh=dx, mesh_steps=(dx, dx, 1),
                    material_sampling='yee', interface_method='staircase', pml_cells=pml_cells,
                    backend='cuda', precision='float64', cuda_kernel='fused',
                    cuda_monitor_kernel='fused', snapshot_interval=10000,
                    boundaries=Boundaries(y_min=BoundaryFace(kind=transverse),
                                          y_max=BoundaryFace(kind=transverse)))
    region.steps = 2500
    return region, dx, width, span


def build_source(kind, injection, dx, width, span):
    """Mirrors the source shape `parity.project_for` uses, varied only in kind and injection."""
    snap = lambda x: round(x / dx) * dx
    # A TFSF box needs two interior cells between itself and the PML, and in a 2-D region it must
    # also span the invariant axis, so it takes the full z extent.
    size = (0.2, 0.2, 1.0) if kind == 'tfsf' else (0, width, 0)
    return Source(id='s', kind=kind, injection=injection, normal='x', component='Ez',
                  center=(snap(-span / 2 + 1.0), 0.0, 0.0), size=size,
                  wavelength=1.5, pulse_cycles=4)


def source_sweep(cfg):
    """Run every schema source kind in true 2-D, and compare against the fake 2-D arm."""
    rows = []
    frequency = [C0 / (1.55e-6)]
    for kind, injection in SOURCE_COMBINATIONS:
        label = f'{kind}/{injection}'
        try:
            # A TFSF box requires PML on every active face, so its cell takes PML transverse
            # faces. That is a property of the source, not of the dimension.
            # A TFSF box requires PML on every active face, so its cell is wider and uses thinner
            # PML to leave the five interior cells the validator demands.
            if kind == 'tfsf':
                region, dx, width, span = homogeneous_pair(cfg, transverse='pml',
                                                           width_um=2.0, pml_cells=25)
            else:
                region, dx, width, span = homogeneous_pair(cfg)
            snap = lambda x: round(x / dx) * dx
            project_3d = Project(name=f'{kind}-{injection}',
                                 region=region,
                                 sources=[build_source(kind, injection, dx, width, span)],
                                 monitors=[FieldMonitor(id='m', name='m', normal='x',
                                                        center=(snap(span / 2 - 1.0), 0.0, 0.0),
                                                        size=(0, width, 1))])
            shape = tuple(region.shape) + (3,)
            epsilon = torch.full(shape, 1.0, dtype=torch.float64, device='cuda')
            model_3d = DifferentiablePlaneSimulation(project_3d, AdjointOptions(checkpoints=4))
            with torch.no_grad():
                planes_3d = model_3d(epsilon, frequency)
            fields_3d = planes_3d['m'].fields.detach().cpu().numpy()

            project_2d, region_2d = to_true_2d(project_3d, region.time_step)
            shape_2d = tuple(region_2d.shape) + (3,)
            epsilon_2d = torch.full(shape_2d, 1.0, dtype=torch.float64, device='cuda')
            model_2d = DifferentiablePlaneSimulation(project_2d, AdjointOptions(checkpoints=4))
            with torch.no_grad():
                planes_2d = model_2d(epsilon_2d, frequency)
            fields_2d = planes_2d['m'].fields.detach().cpu().numpy()

            finite = bool(np.isfinite(fields_2d).all())
            if fields_3d.shape != fields_2d.shape:
                rows.append(dict(source=label, ran=True, finite=finite,
                                 shapes_differ=[list(fields_3d.shape), list(fields_2d.shape)],
                                 relative_difference=None))
                continue
            scale = float(np.abs(fields_3d).max())
            difference = float(np.abs(fields_2d - fields_3d).max()) / scale if scale else float('nan')
            rows.append(dict(source=label, ran=True, finite=finite,
                             shapes_differ=None, relative_difference=difference,
                             two_d_shape=list(region_2d.shape), three_d_cells=int(np.prod(region.shape)),
                             two_d_cells=int(np.prod(region_2d.shape))))
        except Exception as error:  # noqa: BLE001 - a refusal is a result
            # Report WHICH layer refused. The first attempt truncated to the first line, which made
            # a Region-level defect read like a source-level refusal.
            text = ' '.join(str(error).split())
            layer = 'Region' if 'Region' in text else ('Project' if 'Project' in text else 'other')
            rows.append(dict(source=label, ran=False, refused=True, refused_by=layer,
                             message=text[:400]))
    return rows


def tensor_probe(cfg):
    """Criterion C3, third rule: a tensor material in a 2-D region must be refused.

    The first version of this probe never reached the rule because the fixture had no tensor
    material and it returned "admitted" -- a false green. This one builds a project that DOES
    carry `epsilon_tensor`, satisfies every other tensor requirement, and then asks for 2-D.
    """
    from torchfdtd.tensor_project import validate_tensor_project
    from torchfdtd.models import Material
    region = Region(dimension='3d', size=(1.0, 1.0, 1.0), mesh=0.1, precision='float32',
                    memory_mode='resident', interface_method='staircase', material_sampling='yee',
                    boundaries=Boundaries(x_min=BoundaryFace(kind='periodic'), x_max=BoundaryFace(kind='periodic'),
                                          y_min=BoundaryFace(kind='periodic'), y_max=BoundaryFace(kind='periodic'),
                                          z_min=BoundaryFace(kind='periodic'), z_max=BoundaryFace(kind='periodic')))
    region.steps = 20
    material = Material(name='tensor', model='tensor',
                        epsilon_tensor=(2.25, 2.25, 2.25, 0.0, 0.0, 0.0))
    project = Project(name='tensor2d', region=region, materials=[material],
                      sources=[Source(id='s', kind='point', injection='soft', component='Ez',
                                      center=(0.0, 0.0, 0.0), size=(0, 0, 0), wavelength=1.55)],
                      monitors=[Monitor(id='m', name='m', component='Ez',
                                        center=(0.0, 0.0, 0.0), time_downsample=1)])
    validate_tensor_project(project)          # 3-D: must pass, so the probe proves the 2-D refusal
    dump = project.model_dump()
    dump['region'].update(dimension='2d')
    try:
        validate_tensor_project(Project.model_validate(dump))
    except Exception as error:  # noqa: BLE001
        return dict(probe='anisotropic/tensor material in 2-D', refused=True, three_d_admitted=True,
                    message=str(error).split('\n')[0][:220])
    return dict(probe='anisotropic/tensor material in 2-D', refused=False, three_d_admitted=True,
                message='NO REFUSAL: the 2-D tensor project was admitted')


def refusal_probes(cfg):
    """Reproduce the fork's own 2-D refusals, and record them as results."""
    probes = []

    def probe(name, expected_reason, fn, exercises=True):
        if not exercises:
            # A probe that cannot reach the rule is not a check. It is reported as invalid rather
            # than as a pass, because a green result here would claim coverage this leg does not have.
            probes.append(dict(probe=name, refused=None, reason=expected_reason,
                               message='PROBE INVALID: this fixture does not contain the feature the '
                                       'rule guards, so the rule was never reached.'))
            return
        try:
            fn()
        except Exception as error:  # noqa: BLE001 - the message is the result
            message = str(error).split('\n')[0][:220]
            probes.append(dict(probe=name, refused=True, reason=expected_reason, message=message))
        else:
            probes.append(dict(probe=name, refused=False, reason=expected_reason,
                               message='NO REFUSAL: the combination was admitted'))

    # A PMC or symmetric face requires a 3-D region (rule `pmc_2d_schema`).
    def pmc_2d():
        r = Region(dimension='2d', size=(1.0, 0.4, 0.2), mesh=0.02, pml_cells=10,
                   boundaries=Boundaries(x_min=BoundaryFace(kind='pml'), x_max=BoundaryFace(kind='pml'),
                                         y_min=BoundaryFace(kind='symmetric'), y_max=BoundaryFace(kind='symmetric'),
                                         z_min=BoundaryFace(kind='pml'), z_max=BoundaryFace(kind='pml')))
        Project(name='pmc2d', region=r, sources=[Source(id='s', kind='plane', normal='x',
                                                        size=(0, 0.4, 0.2), center=(-0.3, 0, 0),
                                                        wavelength=1.55)], monitors=[])
    probe('a symmetric face in a 2-D region is refused', 'pmc_2d_schema', pmc_2d)

    # A 2-D modal launch propagating along the invariant axis must be refused.
    def modal_along_z():
        dump = scene(cfg, dict(cfg['case']))['p'].model_dump()
        dump['region'].update(dimension='2d')
        dump['region']['boundaries'].update(z_min=dict(kind='pml'), z_max=dict(kind='pml'))
        sources = dump['sources']
        sources[0].update(normal='z', size=(sources[0]['size'][0], sources[0]['size'][1], 0.0),
                          center=(sources[0]['center'][0], sources[0]['center'][1], 0.0))
        project = Project.model_validate(dump)
        prepare_aperture_modal_launch(project, scene(cfg, dict(cfg['case']))['section'], num_modes=1)
    probe('a 2-D modal launch along the invariant axis is refused', 'validate_invariant_launch',
          modal_along_z)

    return probes


def main():
    cfg, run_dir = load_run()
    case = cfg['case']
    sc = scene(cfg, case)
    project_3d, region_3d = sc['p'], sc['r']
    project_2d, region_2d = to_true_2d(project_3d, region_3d.time_step)

    launch = launch_for(project_2d, sc['section'])
    width = torch.tensor(cfg['width_um'], dtype=torch.float64, device='cuda')
    with torch.no_grad():
        epsilon_3d = sc['epsilon'](width)
    # The 3-D epsilon was built against the 3-D region, so its shape does not match. Taking one z
    # slice is legitimate ONLY because this fixture is z-invariant by construction -- every solid in
    # the N8 scene is a box of full z extent. If the structure varied along the invariant axis, this
    # slice would be wrong and the 2-D reduction would be invalid.
    shape_2d = tuple(region_2d.shape)
    if len(shape_2d) == 2:
        epsilon = epsilon_3d[:, :, 0, :]
    elif len(shape_2d) == 3 and shape_2d[2] == 1:
        epsilon = epsilon_3d[:, :, :1, :]
    else:
        raise SystemExit(f'cannot slice a 3-D epsilon to the 2-D shape {shape_2d}')
    if tuple(epsilon.shape) != shape_2d + (3,):
        raise SystemExit(f'epsilon shape {tuple(epsilon.shape)} does not match {shape_2d + (3,)}')
    model = ModeInjectedPlaneSimulation(project_2d, launch,
                                        AdjointOptions(checkpoints=32,
                                                       gpu_budget_bytes=32 * 1024 ** 3))
    frequency = [C0 / (cfg['wavelength_um'] * 1e-6)]
    with torch.no_grad():
        planes = model(epsilon, frequency)
    amps = {key: modal_plane_amplitudes(planes[key], launch) for key in ('near', 'far', 'outside')}
    measured = dict(
        T=float((amps['far']['forward'] / amps['near']['forward']).abs().square().sum()),
        R=float((amps['near']['backward'] / amps['near']['forward']).abs().square().sum()))
    recorded = cfg['recorded_3d']
    errors = {key: abs(measured[key] - recorded[key]) / abs(recorded[key]) for key in ('T', 'R')}

    probes = refusal_probes(cfg)
    probes.append(tensor_probe(cfg))
    sweep = source_sweep(cfg)
    readout = dict(
        policy=cfg['policy'], status='completed', simulation_executed=True,
        fixture=('The N8 grating in a TRUE 2-D region: the invariant axis carries no mesh, no '
                 'boundary and no variation. The 3-D five-cell arm is the fake 2-D.'),
        stages=list(STAGES), stage_source=dict(STAGE_SOURCE),
        cases=[dict(id=case['id'], ppw=round(cfg['wavelength_um'] / case['dx_um']),
                    steps=region_2d.steps, dt_s=region_2d.time_step, shape=list(region_2d.shape),
                    measured=dict(**measured, **{f'{k}_relative_error': v for k, v in errors.items()},
                                  tolerance_relative=cfg['tolerance_relative']))],
        dimension=dict(three_d_shape=list(region_3d.shape), two_d_shape=list(region_2d.shape),
                       three_d_cells=int(np.prod(region_3d.shape)),
                       two_d_cells=int(np.prod(region_2d.shape)),
                       cell_ratio=float(np.prod(region_3d.shape)) / float(np.prod(region_2d.shape)),
                       time_step_pinned=bool(region_2d.time_step_override)),
        refusal_probes=probes, source_sweep=sweep,
        note=('First true-2-D run in this repository. The flux-unit trap applies downstream: a 2-D '
              'plane carries flux per invariant length, a 3-D plane carries flux over an area, so '
              'any absolute power carried across without conversion is wrong by a factor of the '
              'invariant extent.'))
    write_json(Path(run_dir) / 'readout.json', readout)

    print(f'3-D shape {region_3d.shape} ({int(np.prod(region_3d.shape))} cells) -> '
          f'2-D shape {region_2d.shape} ({int(np.prod(region_2d.shape))} cells), '
          f'ratio {np.prod(region_3d.shape) / np.prod(region_2d.shape):.2f}x')
    print(f'true 2-D: T = {measured["T"]!r}, R = {measured["R"]!r}')
    print(f'recorded 3-D: T = {recorded["T"]!r}, R = {recorded["R"]!r}')
    for key, value in errors.items():
        print(f'  {key} relative error = {value:.3e} '
              f'({"inside" if value <= cfg["tolerance_relative"] else "OUTSIDE"} '
              f'{cfg["tolerance_relative"]:g})')
    for probe_row in probes:
        print(f'  probe {probe_row["probe"]}: refused={probe_row["refused"]}')
    for row in sweep:
        if row.get('ran'):
            print(f'  source {row["source"]}: ran=True finite={row["finite"]} '
                  f'2D-vs-fake-2D relative difference={row["relative_difference"]}')
        else:
            print(f'  source {row["source"]}: REFUSED by {row.get("refused_by")}: {row.get("message")[:180]}')
    # The summary must be driven by EVERY section. The previous version inspected only the T/R
    # comparison and printed "passed" over a failed source sweep -- a false green.
    failures = []
    for key, value in errors.items():
        if value > cfg['tolerance_relative']:
            failures.append(f'{key} relative error {value:.3e} exceeds {cfg["tolerance_relative"]:g}')
    for row in sweep:
        if not row.get('ran'):
            failures.append(f'source {row["source"]} refused by {row.get("refused_by")}: '
                            f'{row.get("message", "")[:120]}')
        elif not row.get('finite'):
            failures.append(f'source {row["source"]} produced non-finite fields')
    for probe_row in probes:
        if probe_row.get('refused') is not True:
            failures.append(f'probe {probe_row["probe"]}: refused={probe_row.get("refused")}')
    for line in failures:
        print(f'SECTION FAILURE: {line}')
    print(f'leg10: {"FAILED" if failures else "passed"} '
          f'(every section checked: {len(errors)} comparisons, {len(sweep)} sources, '
          f'{len(probes)} probes)')
    return 56 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
