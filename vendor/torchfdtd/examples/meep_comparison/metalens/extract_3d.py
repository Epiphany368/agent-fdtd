"""Geometry step for Part B (TorchFDTD only, not part of the comparison): writes geometry_3d.json.

The pillar list is the 112-cylinder lens of benchmarks/angular_spectrum.py: its `metalens()` builder is
called with the pillar library recorded in docs/validation/angular-spectrum-3060.json (6 um aperture,
f = 6 um, 0.5 um period, 0.6 um tall silicon cylinders, 0.05 um mesh), and the (x, y, radius) of every
structure is copied. The cell, source and monitors of the comparison are declared here; one TorchFDTD run
with the two section monitors locates the focus and the focal-plane monitor is declared at that node.

    PYTHONPATH=$PWD python \
        examples/meep_comparison/metalens/extract_3d.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import metalens_common as mc  # noqa: E402
import torchfdtd_metalens as tm  # noqa: E402
from torchfdtd import Simulation  # noqa: E402
from benchmarks import angular_spectrum  # noqa: E402

RECORD = HERE.parents[2] / 'docs' / 'validation' / 'angular-spectrum-3060.json'
MESH = .05
INDEX = 3.48
HEIGHT = .6
WAVELENGTHS = [1.6, 1.55, 1.5]
WAVE = dict(type='gaussian_cycles', wavelength_um=1.55, pulse_cycles=3, amplitude=1.0)
COURANT_FACTOR = .99
CELL = [8., 8., 6.7]                     # x, y: 6 um aperture + 2 x 0.5 um margin + 2 x 0.5 um PML; z: see the layout
PML_CELLS = 10
Z_MIN = -CELL[2] / 2                     # -3.35
SOURCE_Z = Z_MIN + 17 * MESH             # -2.5, 0.35 um above the lower PML
PILLAR_BOTTOM = Z_MIN + 24.75 * MESH     # -2.1125: quarter-cell offset, so neither the z nodes (Ex, Ey) nor the z half cells (Ez) touch a pillar face
INCIDENT_Z = Z_MIN + 39 * MESH           # -1.4, 0.1125 um above the pillar tops
TOP_INTERIOR = CELL[2] / 2 - PML_CELLS * MESH   # 2.85
INTERIOR_XY = CELL[0] - 2 * PML_CELLS * MESH    # 7.0
STEPS = 2500                             # 238 fs: pulse end (155 fs) + 5.35 um source-to-top-PML path (18 fs) + ringing


def pillars_from_benchmark():
    record = json.loads(RECORD.read_text(encoding='utf-8'))['metalens_3d']
    library = record['library']
    project, centre = angular_spectrum.metalens('3d', library, aperture=record['aperture_um'], focal=record['focal_um'], mesh=record['mesh_um'],
                                                height=record['height_um'], index=record['index'], period=record['period_um'], through_focus=False,
                                                backend='cpu', distances_um=())
    assert len(project.structures) == record['pillars'] == 112, len(project.structures)
    pillars = [dict(x_um=float(s.center[0]), y_um=float(s.center[1]), radius_um=float(s.radius)) for s in project.structures]
    return pillars, dict(source_record=RECORD.name, library=library, aperture_um=record['aperture_um'], focal_um=record['focal_um'], period_um=record['period_um'],
                         benchmark_focus_from_pillar_top_um=record['section']['focus_fdtd']['normal_um'])


def tie_count(pillars):
    """Yee sample points that lie exactly on a pillar surface (a staircase ambiguity between solvers)."""
    nodes = [-CELL[a] / 2 + np.arange(int(round(CELL[a] / MESH))) * MESH for a in range(3)]
    halves = [v + MESH / 2 for v in nodes]
    ties = 0
    for comp in range(3):
        grids = [halves[a] if a == comp else nodes[a] for a in range(2)]
        x, y = np.meshgrid(*grids, indexing='ij')
        zs = halves[2] if comp == 2 else nodes[2]
        z_faces = np.sum(np.isclose(zs, PILLAR_BOTTOM, atol=1e-9) | np.isclose(zs, PILLAR_BOTTOM + HEIGHT, atol=1e-9))
        for p in pillars:
            d2 = (x - p['x_um']) ** 2 + (y - p['y_um']) ** 2
            ties += int(np.count_nonzero(np.isclose(d2, p['radius_um'] ** 2, rtol=1e-9, atol=0)))
        ties += int(z_faces)
    return ties


def geometry(pillars, provenance, focal_plane_z):
    dt = COURANT_FACTOR / math.sqrt(3) * MESH * 1e-6 / mc.C0
    return dict(
        name='metalens', part='3d', units='um, s',
        description='3D metalens: 112 silicon cylinders (n = 3.48, 0.6 um tall) in air on a 0.5 um square grid, 6 um aperture, hyperbolic phase for f = 6 um, '
                    'x-polarised plane-wave pulse from below (+z). Pillar list copied from benchmarks/angular_spectrum.py.',
        mesh_um=MESH, courant_factor=COURANT_FACTOR, courant_number=COURANT_FACTOR / math.sqrt(3), dt_s=dt, steps=STEPS, run_time_fs=STEPS * dt * 1e15,
        cell_um=CELL, pml_cells=PML_CELLS, pml_um=PML_CELLS * MESH, background_index=1.0, material=dict(name='Si', index=INDEX), polarization='Ex',
        aperture_um=provenance['aperture_um'], focal_length_um=provenance['focal_um'], pitch_um=provenance['period_um'],
        pillar_height_um=HEIGHT, pillar_bottom_um=PILLAR_BOTTOM, pillar_top_um=PILLAR_BOTTOM + HEIGHT, pillar_z_center_um=PILLAR_BOTTOM + HEIGHT / 2,
        pillars=pillars,
        source=dict(component='Ex', z_um=SOURCE_Z, x_span_um=provenance['aperture_um'], y_span_um=provenance['aperture_um'], waveform=WAVE,
                    note='Sheet over the aperture. Meep sheet of (x_span - mesh) by y_span so that its boundaries lie on Ex sample positions (weight 1/2, corners 1/4); TorchFDTD injects the same weights (see the README).'),
        frequencies=dict(wavelengths_um=WAVELENGTHS, note='DFT over the full run, no window'),
        monitors=dict(incident=dict(z_um=INCIDENT_Z, span_um=provenance['aperture_um'], note='aperture-size plane 0.1125 um above the pillar tops: incident power from the bare run'),
                      focal=dict(z_um=focal_plane_z, span_um=INTERIOR_XY, note='declared focal plane (node): map, peak, FWHM along x and y, efficiency'),
                      sections=dict(x_range_um=[-INTERIOR_XY / 2, INTERIOR_XY / 2], z_range_um=[INCIDENT_Z, TOP_INTERIOR],
                                    note='xz section at y = 0 and yz section at x = 0: on-axis intensity versus z, focus location')),
        provenance=dict(source_record=provenance['source_record'], benchmark_focus_from_pillar_top_um=provenance['benchmark_focus_from_pillar_top_um'],
                        library=provenance['library'], focal_plane_note='focal plane = mesh node nearest the on-axis intensity peak of the TorchFDTD run at 1.55 um',
                        surface_ties=tie_count(pillars)))


def main():
    started = time.perf_counter()
    pillars, provenance = pillars_from_benchmark()
    print(f'{len(pillars)} pillars, radii {min(p["radius_um"] for p in pillars):.4f} to {max(p["radius_um"] for p in pillars):.4f} um, '
          f'surface ties {tie_count(pillars)}', flush=True)
    path = mc.GEOMETRY_FILES['3d']
    provisional = round(PILLAR_BOTTOM + HEIGHT + provenance['focal_um'], 6)
    path.write_bytes((json.dumps(geometry(pillars, provenance, provisional), indent=1, allow_nan=False) + '\n').encode('utf-8'))
    spec = mc.load_geometry('3d')
    project = tm.build_3d(spec, with_lens=True, backend='cuda', monitors='axis')
    tm.check_grid(project, spec)
    result = Simulation(project).run()
    xz = result.field_monitor('xz')
    shape = tuple(xz['shape'])
    pts = np.asarray(xz['points_um']).reshape(*shape, 3)
    fields = np.asarray(xz['fields']).reshape(len(xz['frequency_hz']), *shape, -1)
    k = WAVELENGTHS.index(1.55)
    comps = [xz['components'].index(c) for c in ('Ex', 'Ey', 'Ez')]
    intensity = np.sum(np.abs(fields[k][:, 0, :, :][..., comps]) ** 2, axis=-1)   # (nx, nz)
    x = pts[:, 0, 0, 0]
    z = pts[0, 0, :, 2]
    i0 = int(np.argmin(np.abs(x)))
    z_peak, _ = mc.peak_parabolic(z, intensity[i0])
    node = round(Z_MIN + round((z_peak - Z_MIN) / MESH) * MESH, 6)
    path.write_bytes((json.dumps(geometry(pillars, provenance, node), indent=1, allow_nan=False) + '\n').encode('utf-8'))
    print(f'focus at z = {z_peak:.4f} um ({z_peak - PILLAR_BOTTOM - HEIGHT:.3f} um above the pillar tops; benchmark record: '
          f'{provenance["benchmark_focus_from_pillar_top_um"]:.3f}), declared plane {node}; run {result.summary["seconds"]:.1f} s', flush=True)
    print('wrote', path, f'{time.perf_counter() - started:.0f} s', flush=True)


if __name__ == '__main__':
    main()
