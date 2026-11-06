"""Design step for Part A (TorchFDTD only, not part of the comparison): writes design_2d.json and geometry.json.

1. Bloch unit-cell sweep of the transmission phase of one silicon ridge versus its width (period 0.6 um,
   Ez polarisation, 1.55 um). Widths are odd multiples of the mesh so that the ridge edges fall on
   half cells: every staircase-realisable width appears once in the library.
2. Hyperbolic phase profile for a 20 um aperture and a 15 um focal length; each ridge takes the library
   width whose phase is closest to the target modulo 2 pi.
3. One lens run with the on-axis line monitor locates the focus; the focal-plane monitor of the
   comparison is declared at that node. The comparison scripts read geometry.json only.

    PYTHONPATH=$PWD python \
        examples/meep_comparison/metalens/design_2d.py --height 1.0
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
sys.path.insert(0, str(HERE))
import metalens_common as mc  # noqa: E402
import torchfdtd_metalens as tm  # noqa: E402
from torchfdtd import Simulation  # noqa: E402

MESH = .025
PERIOD = .6
INDEX = 3.48
WAVELENGTHS = [1.6, 1.55, 1.5]   # frequency-increasing order (custom DFT samples must increase)
WAVE = dict(type='gaussian_cycles', wavelength_um=1.55, pulse_cycles=3, amplitude=1.0)
COURANT_FACTOR = .99
APERTURE, FOCAL = 20., 15.
PML_CELLS = 40
CELL = [24., 21.5]                      # x: aperture + 2 x 1 um margin + 2 x 1 um PML; y: see the layout below
Y_MIN = -CELL[1] / 2                    # -10.75
SOURCE_Y = Y_MIN + 50 * MESH            # -9.5, 0.25 um above the lower PML
RIDGE_BOTTOM = Y_MIN + 70 * MESH - MESH / 2   # -9.0125: ridge edges on half cells (Ez nodes are never on an edge)
TOP_INTERIOR = CELL[1] / 2 - PML_CELLS * MESH   # 9.75
STEPS = 5200                            # 303 fs: pulse end (155 fs) + 18 um edge-to-focus path (60 fs) + top-PML exit and ringing


def sweep(height, backend):
    """Unit cell: y from -2.5 to 2.5, 1 um PML, source at -1.25, ridge bottom on the half cell -0.7625, Ez line at 0.5."""
    bottom = -.7625
    monitor_y = .5
    assert bottom + height < monitor_y - .05, 'the ridge top must stay below the monitor line'
    common = dict(mesh=MESH, index=INDEX, steps=4300, pml_cells=PML_CELLS, source_y=-1.25, monitor_y=monitor_y, cell_y=5.,
                  courant_factor=COURANT_FACTOR, waveform=WAVE, wavelengths_um=WAVELENGTHS, backend=backend)
    y_center = bottom + height / 2
    ref = Simulation(tm.unit_cell_2d(PERIOD, 0., height, y_center, **common)).run().field_monitor('out')
    ez = ref['components'].index('Ez')
    base = np.mean(ref['fields'][:, :, ez], axis=1)
    rows = []
    widths = [round((2 * m + 1) * MESH, 6) for m in range(1, int(round(PERIOD / MESH / 2)))]   # 0.075 ... 0.575
    for width in widths:
        plane = Simulation(tm.unit_cell_2d(PERIOD, width, height, y_center, **common)).run().field_monitor('out')
        t = np.mean(plane['fields'][:, :, ez], axis=1) / base
        rows.append(dict(width_um=width, nodes=int(round(width / MESH)), phase_rad=[float(v) for v in np.angle(t)], amplitude=[float(v) for v in np.abs(t)]))
        print(f'width {width:.3f}: phase {rows[-1]["phase_rad"][1]:+.3f} rad, |t| {rows[-1]["amplitude"][1]:.3f}', flush=True)
    k = WAVELENGTHS.index(1.55)
    unwrapped = np.unwrap([r['phase_rad'][k] for r in rows])
    for row, phase in zip(rows, unwrapped):
        row['phase_unwrapped_rad'] = float(phase)
    return rows, dict(cell_y_um=5., pml_cells=PML_CELLS, source_y_um=-1.25, ridge_bottom_um=bottom, monitor_y_um=monitor_y, steps=common['steps'])


def choose_widths(library):
    phases = np.array([r['phase_unwrapped_rad'] for r in library]) - library[0]['phase_unwrapped_rad']
    widths = np.array([r['width_um'] for r in library])
    k = 2 * math.pi / 1.55
    count = int(math.floor(APERTURE / PERIOD))
    start = -(count - 1) * PERIOD / 2
    ridges = []
    for i in range(count):
        x = round(start + PERIOD * i, 6)
        target = (k * (math.sqrt((APERTURE / 2) ** 2 + FOCAL ** 2) - math.sqrt(x ** 2 + FOCAL ** 2))) % (2 * math.pi)
        error = np.angle(np.exp(1j * (phases - target)))
        j = int(np.argmin(np.abs(error)))
        ridges.append(dict(x_um=x, width_um=float(widths[j]), target_phase_rad=float(target), library_index=j, phase_error_rad=float(error[j])))
    return ridges


def incident_y(height):
    return round(RIDGE_BOTTOM + height + 4.5 * MESH, 6)   # node 0.1125 um above the ridge top


def geometry(height, ridges, library, focal_plane_y):
    ridge_top = RIDGE_BOTTOM + height
    y_inc = incident_y(height)
    return dict(
        name='metalens', part='2d', units='um, s',
        description='2D cylindrical metalens: silicon ridges (n = 3.48) in air on a 0.6 um pitch, Ez (out-of-plane) polarisation, plane-wave pulse from below (+y).',
        mesh_um=MESH, courant_factor=COURANT_FACTOR, courant_number=COURANT_FACTOR / math.sqrt(2), dt_s=COURANT_FACTOR / math.sqrt(2) * MESH * 1e-6 / mc.C0,
        steps=STEPS, run_time_fs=STEPS * COURANT_FACTOR / math.sqrt(2) * MESH * 1e-6 / mc.C0 * 1e15,
        cell_um=CELL, pml_cells=PML_CELLS, pml_um=PML_CELLS * MESH, background_index=1.0, material=dict(name='Si', index=INDEX), polarization='Ez',
        aperture_um=APERTURE, focal_length_um=FOCAL, pitch_um=PERIOD, ridge_height_um=height, ridge_bottom_um=RIDGE_BOTTOM, ridge_top_um=ridge_top,
        ridge_y_center_um=RIDGE_BOTTOM + height / 2, ridges=[dict(x_um=r['x_um'], width_um=r['width_um']) for r in ridges],
        source=dict(component='Ez', y_um=SOURCE_Y, x_span_um=APERTURE, waveform=WAVE,
                    note='Sheet over the aperture; the two end nodes carry half amplitude in both solvers (Meep volume-source interpolation).'),
        frequencies=dict(wavelengths_um=WAVELENGTHS, note='DFT over the full run, no window'),
        monitors=dict(incident=dict(y_um=y_inc, x_span_um=APERTURE, note='aperture-wide line 0.1125 um above the ridge tops: incident power from the bare run'),
                      focal=dict(y_um=focal_plane_y, x_span_um=CELL[0] - 2 * PML_CELLS * MESH, note='declared focal plane (node): profile, FWHM, efficiency'),
                      axis=dict(x_um=0., y_range_um=[y_inc, TOP_INTERIOR], note='on-axis intensity versus y: focus location')),
        design=dict(file='design_2d.json', library_widths_um=[r['width_um'] for r in library], phase_range_rad=float(library[-1]['phase_unwrapped_rad'] - library[0]['phase_unwrapped_rad']),
                    focal_plane_note='focal plane = mesh node nearest the on-axis intensity peak of the TorchFDTD design run at 1.55 um'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height', type=float, default=1.)
    parser.add_argument('--backend', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    height = args.height
    assert abs(height / MESH - round(height / MESH)) < 1e-9, 'ridge height must be a whole number of cells'
    started = time.perf_counter()
    library, cell_info = sweep(height, args.backend)
    coverage = library[-1]['phase_unwrapped_rad'] - library[0]['phase_unwrapped_rad']
    print(f'phase coverage {coverage:.3f} rad ({coverage / (2 * math.pi):.3f} x 2 pi)', flush=True)
    ridges = choose_widths(library)
    provisional = round(RIDGE_BOTTOM + height + FOCAL, 6)
    spec = geometry(height, ridges, library, provisional)
    path = mc.GEOMETRY_FILES['2d']
    path.write_bytes((json.dumps(spec, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    # Locate the focus with the axis monitor only, then declare the focal plane at the nearest node.
    spec = mc.load_geometry('2d')
    project = tm.build_2d(spec, with_lens=True, backend=args.backend, monitors='axis')
    tm.check_grid(project, spec)
    result = Simulation(project).run()
    axis = result.field_monitor('axis')
    ez = axis['components'].index('Ez')
    y = np.asarray(axis['points_um'])[:, 1]
    k = WAVELENGTHS.index(1.55)
    intensity = np.abs(axis['fields'][k, :, ez]) ** 2
    y_peak, _ = mc.peak_parabolic(y, intensity)
    node = round(Y_MIN + round((y_peak - Y_MIN) / MESH) * MESH, 6)
    spec = geometry(height, ridges, library, node)
    path.write_bytes((json.dumps(spec, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    design = dict(height_um=height, library=library, unit_cell=cell_info, phase_coverage_rad=coverage, ridges=ridges,
                  focus_scan=dict(y_um=y.tolist(), intensity_arbitrary=intensity.tolist(), peak_y_um=y_peak, peak_from_ridge_top_um=y_peak - (RIDGE_BOTTOM + height),
                                  declared_focal_plane_y_um=node, nominal_focal_plane_y_um=provisional),
                  seconds=time.perf_counter() - started, note='TorchFDTD design step; not part of the comparison')
    (HERE / 'design_2d.json').write_bytes((json.dumps(design, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    print(f'focus at y = {y_peak:.4f} um ({y_peak - (RIDGE_BOTTOM + height):.3f} um above the ridge top), declared plane {node}', flush=True)
    print('wrote', path, 'and design_2d.json', f'{time.perf_counter() - started:.0f} s', flush=True)


if __name__ == '__main__':
    main()
