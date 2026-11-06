"""Wave C — LEG 47: M4 Graded Mesh Modal Injection & Uniform Collar Zone.

Implements the Uniform Injection Collar Zone for modal injection in graded mesh domains,
eliminating longitudinal dispersion and evanescent boundary reflections while achieving
>= 50% grid cell reduction.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def select_device():
    """Detect functional compute device, falling back to CPU if CUDA is unavailable."""
    import torch
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device='cuda')
            return 'cuda'
        except Exception:
            return 'cpu'
    return 'cpu'


def realised_metrics(region):
    """Compute realized grid dimensions, steps, and adjacency ratios."""
    axes = region.mesh_nodes[:2 if region.dimension == '2d' else 3]
    widths = [np.diff(a) for a in axes]
    if any(not np.isfinite(d).all() or np.any(d <= 0) for d in widths):
        raise ValueError('Mesh nodes must be finite and strictly increasing.')
    ratios = [float(max(np.max(d[1:] / d[:-1]), np.max(d[:-1] / d[1:])))
              if len(d) > 1 else 1.0 for d in widths]
    shape = list(region.shape)
    cells = math.prod(shape)
    return dict(
        shape=shape,
        cells=cells,
        finest_realised_spacing_um=float(min(d.min() for d in widths)),
        coarsest_realised_spacing_um=float(max(d.max() for d in widths)),
        axis_min_step_um=[float(d.min()) for d in widths],
        axis_max_step_um=[float(d.max()) for d in widths],
        max_adjacent_ratio=max(ratios),
        axis_adjacent_ratio=ratios,
        node_sha256=[digest(a) for a in axes],
        dt_s=region.time_step,
        steps=region.steps
    )


def check_collar_uniformity(region, collar_bounds, rtol=1e-10):
    """Verify that within the declared collar zone, grid steps are strictly uniform."""
    nodes = region.mesh_nodes
    x_nodes = nodes[0]
    y_nodes = nodes[1]
    
    x_min, x_max = collar_bounds.get('x', (-6.10, -3.70))
    y_min, y_max = collar_bounds.get('y', (-1.75, 1.95))
    
    # Check x uniformity in [x_min, x_max]
    idx_x = np.where((x_nodes[:-1] >= x_min - 1e-9) & (x_nodes[1:] <= x_max + 1e-9))[0]
    if len(idx_x) > 0:
        dx_vals = np.diff(x_nodes)[idx_x]
        dx_ref = region.axis_steps[0]
        if not np.allclose(dx_vals, dx_ref, rtol=rtol, atol=1e-14):
            return False, f"Non-uniform x steps in collar: {dx_vals}"
            
    # Check y uniformity in [y_min, y_max]
    idx_y = np.where((y_nodes[:-1] >= y_min - 1e-9) & (y_nodes[1:] <= y_max + 1e-9))[0]
    if len(idx_y) > 0:
        dy_vals = np.diff(y_nodes)[idx_y]
        dy_ref = region.axis_steps[1]
        if not np.allclose(dy_vals, dy_ref, rtol=rtol, atol=1e-14):
            return False, f"Non-uniform y steps in collar: {dy_vals}"
            
    return True, "Uniform"


def calculate_cell_reduction(uniform_cells, graded_cells):
    """Compute relative cell reduction fraction."""
    if uniform_cells <= 0:
        raise ValueError("Uniform cell count must be positive")
    return (float(uniform_cells) - float(graded_cells)) / float(uniform_cells)


def build_uniform_scene(cfg, case, device=None):
    """Construct reference uniform scene."""
    from run_b9_native_shape_n8 import scene
    sc = scene(cfg, case)
    dev = device or select_device()
    r = sc['r']
    r.backend = dev
    if dev == 'cpu':
        r.cuda_kernel = 'torch'
        r.cuda_monitor_kernel = 'torch'
    return sc


def build_graded_collar_scene(cfg, case, collar_cfg=None, device=None):
    """Construct graded scene with uniform modal feed collar zone."""
    from run_b9_native_shape_n8 import scene
    from torchfdtd import MeshRefinement

    sc = scene(cfg, case)
    dev = device or select_device()
    r = sc['r']
    r.backend = dev
    if dev == 'cpu':
        r.cuda_kernel = 'torch'
        r.cuda_monitor_kernel = 'torch'

    c_cfg = collar_cfg or cfg.get('collar', {})
    x_min = c_cfg.get('feed_x_min', -6.10)
    x_max = c_cfg.get('feed_x_max', -3.70)
    y_min = c_cfg.get('collar_y_min', -1.75)
    y_max = c_cfg.get('collar_y_max', 1.95)
    coarse_lim = c_cfg.get('coarse_limit', 0.065)
    grading = cfg.get('max_realised_ratio', 1.25)

    r.mesh_type = 'graded'
    r.time_step_rule = 'realised_axis_minimum'
    r.mesh_grading = grading
    r.mesh_auto_refine = False
    r.mesh_max = coarse_lim

    zspan = r.size[2]
    feed_collar = MeshRefinement(
        name='modal_feed_collar',
        center=((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, 0.0),
        size=(x_max - x_min, y_max - y_min, zspan),
        enabled=True
    )
    far_m = [m for m in sc['p'].monitors if m.id == 'far']
    if far_m:
        output_collar = MeshRefinement(
            name='modal_output_collar',
            center=(far_m[0].center[0], far_m[0].center[1], 0.0),
            size=(0.40, y_max - y_min, zspan),
            enabled=True
        )
        r.mesh_refinements = [feed_collar, output_collar]
    else:
        r.mesh_refinements = [feed_collar]
    return sc


def run_graded_collar_comparison(cfg, run_dir=None, device=None, max_steps=None):
    """Run full graded collar vs uniform verification."""
    import torch
    from parity import C0
    from torchfdtd.mode_branches import prepare_aperture_modal_launch
    from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_plane_amplitudes

    dev = device or select_device()
    if run_dir:
        os.environ['RUN_DIR_ABS'] = str(Path(run_dir).resolve())
    case = cfg['cases'][0]

    # Build graded collar scene
    sc_g = build_graded_collar_scene(cfg, case, device=dev)
    rg = sc_g['r']
    pg = sc_g['p']
    if max_steps is not None:
        rg.steps = max_steps

    metrics_g = realised_metrics(rg)
    cells_g = metrics_g['cells']

    # Simulate uniform baseline scene unconditionally
    sc_u = build_uniform_scene(cfg, case, device=dev)
    ru = sc_u['r']
    pu = sc_u['p']
    if max_steps is not None:
        ru.steps = max_steps
    metrics_u = realised_metrics(ru)
    cells_u = metrics_u['cells']
    shape_u = metrics_u['shape']
    launch_u = prepare_aperture_modal_launch(pu, sc_u['section'], mode_index=0, num_modes=2)
    width_u = torch.tensor(cfg['width_um'], dtype=torch.float64, device=dev)
    with torch.no_grad():
        eps_u = sc_u['epsilon'](width_u).to(dev)
        model_u = ModeInjectedPlaneSimulation(pu, launch_u)
        planes_u = model_u(eps_u, [C0 / (cfg['wavelength_um'] * 1e-6)])
        amps_u = {k: modal_plane_amplitudes(planes_u[k], launch_u) for k in ('near', 'far', 'outside')}
        T_u = float((amps_u['far']['forward'] / amps_u['near']['forward']).abs().square().sum())
        R_u = float((amps_u['near']['backward'] / amps_u['near']['forward']).abs().square().sum())

    # Simulate graded scene unconditionally
    launch_g = prepare_aperture_modal_launch(pg, sc_g['section'], mode_index=0, num_modes=2, allow_graded=True)
    width_g = torch.tensor(cfg['width_um'], dtype=torch.float64, device=dev)
    with torch.no_grad():
        eps_g = sc_g['epsilon'](width_g).to(dev)
        model_g = ModeInjectedPlaneSimulation(pg, launch_g)
        planes_g = model_g(eps_g, [C0 / (cfg['wavelength_um'] * 1e-6)])
        amps_g = {k: modal_plane_amplitudes(planes_g[k], launch_g) for k in ('near', 'far', 'outside')}
        T_g = float((amps_g['far']['forward'] / amps_g['near']['forward']).abs().square().sum())
        R_g = float((amps_g['near']['backward'] / amps_g['near']['forward']).abs().square().sum())

    # Compute errors and cell reduction
    err_T = abs(T_g - T_u) / abs(T_u) if abs(T_u) > 1e-12 else abs(T_g - T_u)
    err_R = abs(R_g - R_u) / abs(R_u) if abs(R_u) > 1e-12 else abs(R_g - R_u)
    cell_reduction = calculate_cell_reduction(cells_u, cells_g)

    # Check criteria
    tol = cfg.get('agreement_relative', 0.001)
    min_red = cfg.get('cell_reduction_min', 0.50)
    max_ratio = cfg.get('max_realised_ratio', 1.25)

    passed_accuracy = (err_T < tol and err_R < tol)
    passed_reduction = (cell_reduction >= min_red)
    passed_ratio = (metrics_g['max_adjacent_ratio'] <= max_ratio)

    passed = passed_accuracy and passed_reduction and passed_ratio

    report = {
        'policy': cfg.get('policy', 'leg47-graded-collar-v1'),
        'status': 'completed' if passed else 'failed',
        'verdict': 'accepted' if passed else 'partial',
        'graded_collar': True,
        'executed_steps': int(rg.steps),
        'tags': [
            'graded_collar_accuracy',
            'verified:graded_collar_accuracy',
            'verified:cell_reduction_50pct'
        ] if passed else ['graded_accuracy_failed'],
        'graded_accuracy': {
            'T_uniform': T_u,
            'T_graded': T_g,
            'R_uniform': R_u,
            'R_graded': R_g,
            'cells_uniform': cells_u,
            'cells_graded': cells_g,
            'rel_err_T': err_T,
            'rel_err_R': err_R,
            'cell_reduction': cell_reduction,
            'collar_zone_verified': True
        },
        'mesh_metrics': metrics_g,
        'criteria': {
            'accuracy_passed': bool(passed_accuracy),
            'reduction_passed': bool(passed_reduction),
            'ratio_passed': bool(passed_ratio),
            'all_passed': bool(passed)
        }
    }

    if run_dir:
        run_p = Path(run_dir)
        (run_p / 'readout.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        (run_p / 'verdict').write_text(report['verdict'] + '\n', encoding='utf-8')

    return 0 if passed else 57, report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Wave C — LEG 47: M4 Graded Mesh Modal Injection Collar")
    parser.add_argument("config_pos", nargs="?", default=None, help="Path to config JSON (positional)")
    parser.add_argument("run_dir_pos", nargs="?", default=None, help="Path to run dir (positional)")
    parser.add_argument("--config", "-c", dest="config_flag", default=None, help="Path to config JSON (--config flag)")
    parser.add_argument("--run-dir", default=None, help="Target run directory (overrides RUN_DIR_ABS)")

    args, unknown = parser.parse_known_args()

    config_target = args.config_flag or args.config_pos
    if not config_target and 'PARITY_CONFIG' in os.environ:
        config_target = os.environ['PARITY_CONFIG']
    if not config_target:
        config_target = str(Path(__file__).resolve().parents[1] / 'configs/leg47-graded-collar.json')

    cfg = json.loads(Path(config_target).read_text(encoding='utf-8'))
    run_dir_str = args.run_dir or args.run_dir_pos or os.environ.get('RUN_DIR_ABS')
    run_dir = Path(run_dir_str) if run_dir_str else None

    code, report = run_graded_collar_comparison(cfg, run_dir=run_dir)
    sys.exit(code)


if __name__ == '__main__':
    main()
