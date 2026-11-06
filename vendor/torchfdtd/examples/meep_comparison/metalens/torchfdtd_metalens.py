"""TorchFDTD side of the metalens comparison (Part A: 2D ridge lens, Part B: 3D pillar lens).

Run inside the WSL2 distribution torchfdtd-bench from the worktree root:
    PYTHONPATH=$PWD python \
        examples/meep_comparison/metalens/torchfdtd_metalens.py --part 2d
    ... --part 3d
Each part writes docs/validation/meep_comparison/metalens_<part>_torchfdtd.json. The lens run and the
bare-cell normalisation run use the same grid, source, monitors and step count; every reported quantity
is a ratio of the two. `--timing` repeats the timed lens solve three times after one warm-up.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # the repository that holds this example
sys.path.insert(0, str(HERE))
# Import torchfdtd from the checkout that holds this example, as the microring script does.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import metalens_common as mc  # noqa: E402

import torch  # noqa: E402
import torchfdtd  # noqa: E402
from torchfdtd import FieldMonitor, Material, Project, Region, Simulation, Source, Structure  # noqa: E402
from torchfdtd.solver import field_axes, hardware, source_slice, voxelize  # noqa: E402
from torchfdtd.waveforms import source_time_signal  # noqa: E402


def require_checkout_import():
    """The records name the commit of the checkout that holds this example; an installed wheel would misattribute them."""
    imported = Path(torchfdtd.__file__).resolve()
    if not str(imported).startswith(str(REPO)):
        raise SystemExit(f'torchfdtd was imported from {imported}, not from the checkout {REPO} that holds this example. '
                         f'Run the script from that checkout: cd {REPO} && python {HERE.relative_to(REPO).as_posix()}/{Path(__file__).name} --part 2d')
PACKAGES = ('torchfdtd', 'torch', 'cupy-cuda12x', 'numpy', 'scipy')


def spectrum(spec):
    return dict(sampling='custom', custom_frequencies_hz=mc.frequencies_hz(spec['frequencies']), apodization='none')


def source_settings(spec):
    wave = spec['source']['waveform']
    return dict(wavelength=wave['wavelength_um'], pulse_cycles=wave['pulse_cycles'], time_definition='cycles', pulse='gaussian')


def region_settings(spec, backend):
    return dict(mesh=spec['mesh_um'], steps=spec['steps'], pml_cells=spec['pml_cells'], backend=backend, precision='float32',
                material_sampling='yee', interface_method='staircase', cuda_kernel='fused' if backend == 'cuda' else 'torch',
                cuda_monitor_kernel='fused' if backend == 'cuda' else 'torch', snapshot_interval=spec['steps'],
                courant_factor=spec['courant_factor'])


# ----------------------------------------------------------------------------
# Part A: 2D cylindrical lens of silicon ridges, Ez polarisation, propagation along +y
# ----------------------------------------------------------------------------
def build_2d(spec, *, with_lens=True, backend='cuda', monitors='all'):
    s = spec
    sx, sy = s['cell_um']
    region = Region(dimension='2d', size=(sx, sy, 1.), **region_settings(s, backend))
    src = s['source']
    half = src['x_span_um'] / 2
    dx = s['mesh_um']
    # Meep weights the two end nodes of a volume source by 1/2 (linear interpolation of the volume onto the
    # grid); the same sheet here is the 799 interior nodes at full amplitude plus two half-amplitude points.
    sources = [Source(id='sheet', kind='plane', normal='y', component='Ez', center=(0., src['y_um'], 0.), size=(src['x_span_um'] - 2 * dx, 0., 0.),
                      **source_settings(s)),
               Source(id='end_neg', kind='point', component='Ez', center=(-half, src['y_um'], 0.), amplitude=.5, **source_settings(s)),
               Source(id='end_pos', kind='point', component='Ez', center=(half, src['y_um'], 0.), amplitude=.5, **source_settings(s))]
    m = s['monitors']
    field_monitors = []
    if monitors in ('all', 'axis'):
        y0, y1 = m['axis']['y_range_um']
        field_monitors.append(FieldMonitor(id='axis', name='axis', normal='x', center=(m['axis']['x_um'], (y0 + y1) / 2, 0.), size=(0., y1 - y0, 1.),
                                           record_fields=('Ez',), record_poynting=(), record_flux=False, spectrum=spectrum(s)))
    if monitors == 'all':
        for name in ('incident', 'focal'):
            field_monitors.append(FieldMonitor(id=name, name=name, normal='y', center=(0., m[name]['y_um'], 0.), size=(m[name]['x_span_um'], 0., 1.),
                                               record_fields=('Ez', 'Hx', 'Hy'), record_poynting=('y',), record_flux=True, spectrum=spectrum(s)))
    structures = [Structure(id=f'ridge{i}', kind='rectangle', center=(r['x_um'], s['ridge_y_center_um'], 0.), size=(r['width_um'], s['ridge_height_um'], 1.),
                            material='Si') for i, r in enumerate(s['ridges'])] if with_lens else []
    return Project(name='meep comparison metalens 2d', region=region, materials=[Material(name='Air', index=1.), Material(name='Si', index=s['material']['index'])],
                   structures=structures, sources=sources, monitors=field_monitors)


def unit_cell_2d(period, width, height, y_center, *, mesh, index, steps, pml_cells, source_y, monitor_y, cell_y, courant_factor, waveform, wavelengths_um, backend='cuda'):
    """One periodic ridge cell lit from below; the transmitted Ez line lies above the ridge."""
    from torchfdtd import Boundaries, BoundaryFace
    region = Region(dimension='2d', size=(period, cell_y, 1.), mesh=mesh, steps=steps, pml_cells=pml_cells, backend=backend, precision='float32',
                    material_sampling='yee', cuda_kernel='fused' if backend == 'cuda' else 'torch', cuda_monitor_kernel='fused' if backend == 'cuda' else 'torch',
                    snapshot_interval=steps, courant_factor=courant_factor,
                    boundaries=Boundaries(x_min=BoundaryFace(kind='periodic'), x_max=BoundaryFace(kind='periodic')))
    freqs = [mc.C0 / (w * 1e-6) for w in wavelengths_um]
    source = Source(id='sheet', kind='plane', normal='y', component='Ez', center=(0., source_y, 0.), size=(period, 0., 0.),
                    wavelength=waveform['wavelength_um'], pulse_cycles=waveform['pulse_cycles'], time_definition='cycles')
    monitor = FieldMonitor(id='out', name='out', normal='y', center=(0., monitor_y, 0.), size=(period, 0., 1.), record_fields=('Ez',), record_poynting=(),
                           record_flux=False, spectrum=dict(sampling='custom', custom_frequencies_hz=freqs, apodization='none'))
    structures = [Structure(id='ridge', kind='rectangle', center=(0., y_center, 0.), size=(width, height, 1.), material='Si')] if width > 0 else []
    return Project(name='unit cell', region=region, materials=[Material(name='Air', index=1.), Material(name='Si', index=index)],
                   structures=structures, sources=[source], monitors=[monitor])


# ----------------------------------------------------------------------------
# Part B: 3D pillar lens, Ex polarisation, propagation along +z
# ----------------------------------------------------------------------------
def build_3d(spec, *, with_lens=True, backend='cuda', monitors='all'):
    s = spec
    region = Region(dimension='3d', size=tuple(s['cell_um']), **region_settings(s, backend))
    src = s['source']
    dx = s['mesh_um']
    z = src['z_um']
    # Meep weights the samples on the boundary of a volume source by 1/2 (linear interpolation of the volume onto the grid). The
    # Meep sheet is (x_span - dx) by y_span so that its x boundaries fall on Ex samples (x = +-2.975 um) and its y boundaries on
    # nodes (y = +-3 um). The same weights here: the interior sheet at full amplitude, the four boundary lines at 1/2 and the four
    # corner samples at 1/4.
    hx, hy = (src['x_span_um'] - dx) / 2, src['y_span_um'] / 2       # boundary sample positions: x = +-hx, y = +-hy
    inner_x, inner_y = src['x_span_um'] - 3 * dx, src['y_span_um'] - 2 * dx  # spans of the interior samples (|x| <= hx - dx, |y| <= hy - dx)
    settings = dict(component='Ex', **source_settings(s))
    sources = [Source(id='sheet', kind='plane', normal='z', center=(0., 0., z), size=(inner_x, inner_y, 0.), **settings)]
    for sign in (-1., 1.):
        sources.append(Source(id=f'edge_x{sign:+.0f}', kind='plane', normal='z', center=(sign * hx, 0., z), size=(0., inner_y, 0.), amplitude=.5, **settings))
        sources.append(Source(id=f'edge_y{sign:+.0f}', kind='plane', normal='z', center=(0., sign * hy, z), size=(inner_x, 0., 0.), amplitude=.5, **settings))
    for sx in (-1., 1.):
        for sy in (-1., 1.):
            sources.append(Source(id=f'corner_x{sx:+.0f}_y{sy:+.0f}', kind='point', center=(sx * hx, sy * hy, z), amplitude=.25, **settings))
    m = s['monitors']
    field_monitors = []
    z0, z1 = m['sections']['z_range_um']
    if monitors in ('all', 'axis'):
        (x0, x1) = m['sections']['x_range_um']
        field_monitors.append(FieldMonitor(id='xz', name='xz', normal='y', center=((x0 + x1) / 2, 0., (z0 + z1) / 2), size=(x1 - x0, 0., z1 - z0),
                                           record_fields=('Ex', 'Ey', 'Ez'), record_poynting=(), record_flux=False, spectrum=spectrum(s)))
        field_monitors.append(FieldMonitor(id='yz', name='yz', normal='x', center=(0., (x0 + x1) / 2, (z0 + z1) / 2), size=(0., x1 - x0, z1 - z0),
                                           record_fields=('Ex', 'Ey', 'Ez'), record_poynting=(), record_flux=False, spectrum=spectrum(s)))
    if monitors == 'all':
        for name in ('incident', 'focal'):
            span = m[name]['span_um']
            field_monitors.append(FieldMonitor(id=name, name=name, normal='z', center=(0., 0., m[name]['z_um']), size=(span, span, 0.),
                                               record_fields=('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'), record_poynting=('z',), record_flux=True, spectrum=spectrum(s)))
    structures = [Structure(id=f'pillar{i}', kind='circle', center=(p['x_um'], p['y_um'], s['pillar_z_center_um']), radius=p['radius_um'],
                            size=(1., 1., s['pillar_height_um']), material='Si') for i, p in enumerate(s['pillars'])] if with_lens else []
    return Project(name='meep comparison metalens 3d', region=region, materials=[Material(name='Air', index=1.), Material(name='Si', index=s['material']['index'])],
                   structures=structures, sources=sources, monitors=field_monitors)


# ----------------------------------------------------------------------------
# Runs, checks and the record
# ----------------------------------------------------------------------------
def check_grid(project, spec):
    r = project.region
    assert list(r.shape[:len(spec['cell_um'])]) == mc.cells(spec), (r.shape, mc.cells(spec))
    assert math.isclose(r.time_step, mc.time_step_s(spec), rel_tol=1e-12), (r.time_step, mc.time_step_s(spec))
    assert r.steps == spec['steps']
    times = np.arange(1, r.steps + 1) * r.time_step
    native = source_time_signal(project.sources[0], times)
    shared = mc.waveform(spec['source']['waveform'], times)
    assert np.allclose(native, shared, rtol=0, atol=1e-15), 'shared waveform differs from the torchfdtd waveform'


def source_support(project):
    out = {}
    for src in project.sources:
        loc = source_slice(src, project.region)
        out[src.id] = dict(cell_slices=[[sl.start, sl.stop] if isinstance(sl, slice) else [int(sl), int(sl) + 1] for sl in loc], amplitude=src.amplitude)
    return out


def silicon_cells(project):
    eps, counts = voxelize(project)
    per_component = {c: int(np.count_nonzero(eps[..., i] > 1.5)) for i, c in enumerate(('Ex', 'Ey', 'Ez'))}
    return dict(per_component=per_component, per_structure_max=counts)


def timed_run(project):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    started = time.perf_counter()
    result = Simulation(project).run()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    full = time.perf_counter() - started
    return result, dict(setup_seconds=result.summary['setup_seconds'], stepping_seconds=result.summary['seconds'], full_seconds=full,
                        steps=result.summary['steps'], mcells_per_second=result.summary['mcells_per_second'])


def plane_arrays(monitor):
    """(freq, point, comp) complex fields plus point coordinates and quadrature weights in m or m^2."""
    fields = np.asarray(monitor['fields'])
    comps = list(monitor['components'])
    return fields, comps, np.asarray(monitor['points_um']), np.asarray(monitor['weights']), tuple(monitor['shape'])


def observables_2d(lens, bare, spec):
    wl = spec['frequencies']['wavelengths_um']
    out = dict(wavelengths_um=wl)
    # Incident power and intensity through the aperture-wide line just above the ridge tops (bare cell).
    f, c, pts, w, _ = plane_arrays(bare.field_monitor('incident'))
    ez, hx = f[..., c.index('Ez')], f[..., c.index('Hx')]
    inc_density = mc.poynting_2d(ez, hx)
    inc_power = inc_density @ w
    inc_intensity = np.mean(np.abs(ez) ** 2, axis=1)
    f, c, pts, w, _ = plane_arrays(lens.field_monitor('incident'))
    trans_power = mc.poynting_2d(f[..., c.index('Ez')], f[..., c.index('Hx')]) @ w
    out['incident'] = dict(x_um=mc.as_list(pts[:, 0]), weight_m=mc.uniform_weight(w), bare_power=mc.as_list(inc_power), lens_power=mc.as_list(trans_power),
                           transmission=mc.as_list(trans_power / inc_power), bare_mean_intensity=mc.as_list(inc_intensity))
    # Focal line: intensity relative to the incident intensity, Poynting density relative to the incident power.
    f, c, pts, w, _ = plane_arrays(lens.field_monitor('focal'))
    ez, hx = f[..., c.index('Ez')], f[..., c.index('Hx')]
    intensity = np.abs(ez) ** 2 / inc_intensity[:, None]
    density = mc.poynting_2d(ez, hx) / inc_power[:, None]
    out['focal'] = dict(y_um=spec['monitors']['focal']['y_um'], x_um=mc.as_list(pts[:, 0]), weight_m=mc.uniform_weight(w),
                        intensity=mc.as_list(intensity), poynting_y=mc.as_list(density))
    # Axis line: intensity relative to the incident intensity.
    f, c, pts, w, _ = plane_arrays(lens.field_monitor('axis'))
    axis_i = np.abs(f[..., c.index('Ez')]) ** 2 / inc_intensity[:, None]
    out['axis'] = dict(x_um=spec['monitors']['axis']['x_um'], y_um=mc.as_list(pts[:, 1]), intensity=mc.as_list(axis_i))
    out.update(mc.summary_2d(out, spec))
    return out


def observables_3d(lens, bare, spec):
    wl = spec['frequencies']['wavelengths_um']
    out = dict(wavelengths_um=wl)
    f, c, pts, w, shape = plane_arrays(bare.field_monitor('incident'))
    e = [f[..., c.index(n)] for n in ('Ex', 'Ey', 'Hx', 'Hy')]
    inc_power = mc.poynting_3d_z(*e) @ w
    inc_intensity = np.mean(np.abs(f[..., c.index('Ex')]) ** 2 + np.abs(f[..., c.index('Ey')]) ** 2 + np.abs(f[..., c.index('Ez')]) ** 2, axis=1)
    f, c, pts, w, shape = plane_arrays(lens.field_monitor('incident'))
    e = [f[..., c.index(n)] for n in ('Ex', 'Ey', 'Hx', 'Hy')]
    trans_power = mc.poynting_3d_z(*e) @ w
    out['incident'] = dict(shape=list(shape), bare_power=mc.as_list(inc_power), lens_power=mc.as_list(trans_power), transmission=mc.as_list(trans_power / inc_power),
                           bare_mean_intensity=mc.as_list(inc_intensity))
    f, c, pts, w, shape = plane_arrays(lens.field_monitor('focal'))
    e = [f[..., c.index(n)] for n in ('Ex', 'Ey', 'Hx', 'Hy')]
    intensity = mc.intensity(f[..., [c.index(n) for n in ('Ex', 'Ey', 'Ez')]]) / inc_intensity[:, None]
    density = mc.poynting_3d_z(*e) / inc_power[:, None]
    out['focal'] = dict(z_um=spec['monitors']['focal']['z_um'], shape=list(shape), x_um=mc.as_list(pts[:, 0].reshape(shape)[:, 0, 0]),
                        y_um=mc.as_list(pts[:, 1].reshape(shape)[0, :, 0]), weight_m2=mc.uniform_weight(w), intensity=mc.as_list(intensity), poynting_z=mc.as_list(density))
    for name, axis in (('xz', 0), ('yz', 1)):
        f, c, pts, w, shape = plane_arrays(lens.field_monitor(name))
        inten = mc.intensity(f[..., [c.index(n) for n in ('Ex', 'Ey', 'Ez')]]) / inc_intensity[:, None]
        grid = pts.reshape(*shape, 3)
        transverse = grid[..., axis].reshape(-1, shape[2])[:, 0] if axis == 0 else grid[0, :, 0, 1]
        out[name] = dict(shape=list(shape), transverse_um=mc.as_list(transverse), z_um=mc.as_list(grid.reshape(-1, shape[2], 3)[0, :, 2]),
                         intensity=[mc.as_list(inten[k].reshape(-1, shape[2])) for k in range(len(wl))])
    out.update(mc.summary_3d(out, spec))
    return mc.trim_3d(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('2d', '3d'), required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--timing', action='store_true', help='one warm-up, then three timed lens solves; medians are reported')
    parser.add_argument('--backend', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    require_checkout_import()
    spec = mc.load_geometry(args.part)
    build = build_2d if args.part == '2d' else build_3d
    lens_project = build(spec, with_lens=True, backend=args.backend)
    bare_project = build(spec, with_lens=False, backend=args.backend)
    check_grid(lens_project, spec)
    load_note = mc.host_load_note('timing' if args.timing else 'development')
    print('bare cell run', flush=True)
    bare, bare_timing = timed_run(bare_project)
    samples = []
    repeats = 4 if args.timing else 1
    for i in range(repeats):
        print(f'lens run {i}', flush=True)
        lens, timing = timed_run(lens_project)
        print(json.dumps(timing), flush=True)
        samples.append(timing)
    timed = samples[1:] if args.timing else samples
    observables = (observables_2d if args.part == '2d' else observables_3d)(lens, bare, spec)
    r = lens_project.region
    record = dict(schema='torchfdtd-meep-comparison-v1', solver='torchfdtd', example='metalens', part=args.part, geometry_file=Path(spec['_path']).name,
                  geometry_sha256=spec['_sha256'], date=time.strftime('%Y-%m-%d'),
                  environment=mc.environment(dict(packages=mc.package_versions(PACKAGES), torch_hardware=hardware(), torch_cuda=torch.version.cuda,
                                                  torchfdtd_file=torchfdtd.__file__)),
                  method=('Yee grid, staircase materials sampled at each E component position (material_sampling="yee"), fused CUDA kernels, float32 fields, '
                          'CPML of the shared thickness, soft additive current sheet with the shared waveform, plane DFT monitors with trilinear collocation '
                          'of E and H at cell-centred points (H carries the half-step phase), complex64 accumulation. Bare-cell normalisation run on the same '
                          'grid with the same monitors.'),
                  grid=dict(shape=list(r.shape), cells=int(math.prod(r.shape)), mesh_um=r.mesh, dt_s=r.time_step, courant_number=r.rectangular_courant,
                            steps=lens.summary['steps'], pml_cells=spec['pml_cells'], precision='float32', backend=lens.summary['backend'],
                            cuda_kernel=lens.summary['cuda_kernel'], cuda_graph=lens.summary['cuda_graph'], device=lens.summary['gpu']),
                  pml=dict(formulation='stretched-coordinate CPML, cubic sigma profile sigma=40*rho^3/(L+1) in Courant units, kappa=1, alpha=1e-8',
                           layers=spec['pml_cells'], thickness_um=spec['pml_cells'] * spec['mesh_um']),
                  source=dict(waveform=spec['source']['waveform'], support=source_support(lens_project), injection='soft additive E (no scaling)'),
                  silicon_cells=silicon_cells(lens_project),
                  monitors={m.id: dict(points=len(lens.field_monitor(m.id)['weights']), shape=list(lens.field_monitor(m.id)['shape']), normal=m.normal)
                            for m in lens_project.monitors},
                  observables=observables,
                  timing=dict(timing_mode='timing' if args.timing else 'development', samples=timed, warmup=samples[0] if args.timing else None,
                              bare_run=bare_timing, host_load_note=load_note, **mc.timing_summary(timed)))
    out = Path(args.out) if args.out else mc.RECORDS / f'metalens_{args.part}_torchfdtd.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(record, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    print('wrote', out, flush=True)
    print(json.dumps(observables['summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()
