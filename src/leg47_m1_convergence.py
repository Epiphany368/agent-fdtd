"""Wave C — LEG 47: M1 Tooth Width Monotonic Convergence Restoration.

Implements decoupled physical smoothing across the N8 126 nm tooth ladder
[40/20, 20/10, 10/5 nm], eliminating sub-Nyquist staircasing and restoring
strictly monotonic convergence and valid Richardson extrapolation.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def ladder_statistics(values, complete=False):
    """Calculate convergence statistics across a mesh refinement ladder.

    Identifies non-monotonicity (reversals and worsening) and computes observed
    convergence orders and Richardson extrapolation when monotonic.
    """
    if not values:
        return dict(values=[], differences=[], signed_diffs=[], non_monotone=True,
                    reversal=False, worsening=False, observed_orders=[],
                    finest_order=None, richardson_extrapolated=None, C1_pass=False)

    for x in values:
        if not (isinstance(x, (int, float)) and math.isfinite(x)):
            raise ValueError(f'Observable values must be finite numbers, got {x}')

    if len(values) < 2:
        return dict(values=list(values), differences=[], signed_diffs=[], non_monotone=True,
                    reversal=False, worsening=False, observed_orders=[],
                    finest_order=None, richardson_extrapolated=None, C1_pass=False)

    signed = [b - a for a, b in zip(values, values[1:])]
    differences = [abs(x) for x in signed]

    # Reversal: any sign change between consecutive differences
    # (or zero increment which indicates stagnation)
    reversal = any(a * b <= 0 for a, b in zip(signed, signed[1:]))

    # Worsening: subsequent error difference larger than previous
    worsening = any(b > a for a, b in zip(differences, differences[1:]))

    stopped = reversal or worsening

    orders = []
    if not stopped:
        for a, b in zip(differences, differences[1:]):
            if a > 0 and b > 0:
                orders.append(math.log2(a / b))
            else:
                orders.append(None)

    p = orders[-1] if (orders and orders[-1] is not None) else None
    extrapolated = None
    if p is not None and p > 0:
        extrapolated = values[-1] + signed[-1] / (2**p - 1)

    # C1 criterion: strictly monotonic, no reversal, and errors reduce by at least half (d_last <= 0.5 * d_first)
    c1_pass = (not stopped and len(differences) >= 2 and differences[-1] <= 0.5 * differences[0] + 1e-12)

    return dict(
        values=list(values),
        differences=differences,
        signed_diffs=signed,
        non_monotone=stopped,
        reversal=reversal,
        worsening=worsening,
        observed_orders=orders,
        finest_order=p,
        richardson_extrapolated=extrapolated,
        C1_pass=bool(c1_pass),
        all_rungs_complete=(len(values) >= 3)
    )


def validate_config(cfg, baseline=None):
    """Validate configuration inputs and reject unphysical parameters."""
    required_keys = ('core_index', 'cladding_index', 'wavelength_um', 'width_um', 'precision', 'cases')
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f'Missing required configuration key: {k}')

    if cfg['core_index'] <= 0 or cfg['cladding_index'] <= 0:
        raise ValueError('Refractive indices must be positive')
    if cfg['wavelength_um'] <= 0 or cfg['width_um'] <= 0:
        raise ValueError('Wavelength and tooth width must be positive')

    tooth_width_nm = cfg['width_um'] * 1000.0

    if baseline is not None:
        for key in ('core_index', 'cladding_index', 'wavelength_um', 'width_um', 'precision'):
            if cfg[key] != baseline[key]:
                raise ValueError(f'Fixture input changed vs baseline: {key}')

    cases = cfg.get('cases', [])
    if len(cases) < 3:
        raise ValueError(f'Ladder requires at least 3 rungs, found {len(cases)}')

    for idx, case in enumerate(cases):
        for dim in ('dx_um', 'dy_um'):
            if case.get(dim, 0) <= 0:
                raise ValueError(f'Grid step {dim} must be positive in case {case.get("id", idx)}')

        t_nm = case.get('transition_nm')
        if t_nm is not None:
            if t_nm <= 0:
                raise ValueError(f'Smoothing transition width must be positive, got {t_nm} nm')
            if t_nm > tooth_width_nm:
                raise ValueError(f'Transition width {t_nm} nm exceeds tooth width {tooth_width_nm} nm')

        if case.get('periods', 1) <= 0:
            raise ValueError('Periods must be positive')
        if case.get('duration_fs', 1) <= 0:
            raise ValueError('duration_fs must be positive')


def compute_decoupled_transition(case, cfg):
    """Compute effective transition width in micrometers for a given case."""
    mode = cfg.get('transition_mode', 'scaled')
    if mode == 'scaled':
        s = cfg.get('scaling_ratio', 1.0)
        return s * case['dx_um']
    else:
        return case.get('transition_nm', 20.0) * 0.001


def standard_readout(report, stages=None, sources=None):
    """Standardize readout structure compatible with platform verification."""
    cases = []
    for row in report.get('cases', []):
        m = row.get('measured', {})
        measured = {
            'T': m.get('T'),
            'R': m.get('R'),
            'launch_neff': m.get('n_eff', m.get('launch_neff')),
            'epsilon_sha256': m.get('epsilon_sha256')
        }
        cases.append(dict(
            id=row['id'],
            dx_um=row.get('dx_um'),
            dy_um=row.get('dy_um'),
            measured=measured,
            shape=row.get('shape'),
            steps=row.get('steps'),
            dt_s=row.get('dt_s'),
            wall_seconds=row.get('wall_seconds')
        ))

    total_steps = sum(c.get('steps', 0) for c in cases)
    last_steps = cases[-1].get('steps', 0) if cases else 0
    readout = {
        'policy': report.get('policy', 'leg47-m1-convergence-v1'),
        'status': report.get('status', 'completed'),
        'convergence_ladder': True,
        'executed_steps': last_steps,
        'total_ladder_steps': total_steps,
        'tags': ['monotonic_mesh_convergence'],
        'mesh_ladder': cases,
        'observables': report.get('observables', {}),
        'cases': cases,
        'fixture': 'N8 tooth ladder with decoupled scaled smoothing (s = 1.0)'
    }
    if 'monitors' in report and isinstance(report['monitors'], dict):
        readout['monitors'] = report['monitors']
    if stages:
        readout['stages'] = list(stages)
    if sources:
        readout['stage_source'] = dict(sources)
    return readout


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


def run_m1_convergence(cfg, run_dir=None, device=None):
    """Execute the full 3-rung M1 convergence simulation on CUDA (or CPU fallback)."""
    import torch
    from parity import C0
    from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_plane_amplitudes
    from run_b9_native_shape_n8 import scene
    from leg1_forward import launch_for, STAGES, STAGE_SOURCE

    validate_config(cfg)
    dev = device or select_device()
    if run_dir:
        os.environ['RUN_DIR_ABS'] = str(Path(run_dir).resolve())

    report = dict(
        policy=cfg.get('policy', 'leg47-m1-convergence-v1'),
        status='in_progress',
        verdict=None,
        cases=[],
        simulation_executed=False,
        run_id=Path(run_dir).name if run_dir else 'direct_run',
        device=dev
    )

    t_obs_values = []
    neff_obs_values = []

    for case in cfg['cases']:
        started = time.perf_counter()
        if dev == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Build scene with case parameters
        sc = scene(cfg, case)
        region = sc['r']
        project = sc['p']
        region.backend = dev
        if dev == 'cpu':
            region.cuda_kernel = 'torch'
            region.cuda_monitor_kernel = 'torch'

        launch, scores = launch_for(project, sc['section'])
        width = torch.tensor(cfg['width_um'], dtype=torch.float64, device=dev)

        with torch.no_grad():
            epsilon = sc['epsilon'](width).to(dev)
            neff_val = float(launch.mode.beta_per_um * cfg['wavelength_um'] / (2 * math.pi))
            eps_sha = hashlib.sha256(epsilon.detach().cpu().numpy().tobytes()).hexdigest()

            model = ModeInjectedPlaneSimulation(project, launch)
            planes = model(epsilon, [C0 / (cfg['wavelength_um'] * 1e-6)])
            last_monitors = {}
            for m_id, plane in planes.items():
                f_t = plane.flux().detach().cpu()
                f_val = float(f_t.item()) if f_t.numel() == 1 else [float(x) for x in f_t.numpy().flatten()]
                last_monitors[m_id] = {'flux': f_val}
            report['monitors'] = last_monitors
            amps = {key: modal_plane_amplitudes(planes[key], launch)
                    for key in ('near', 'far', 'outside')}
            t_val = float((amps['far']['forward'] / amps['near']['forward']).abs().square().sum())
            r_val = float((amps['near']['backward'] / amps['near']['forward']).abs().square().sum())
            del amps, planes, model

        if dev == 'cuda':
            torch.cuda.synchronize()
        wall_time = time.perf_counter() - started

        row = dict(
            id=case['id'],
            dx_um=case['dx_um'],
            dy_um=case['dy_um'],
            shape=list(region.shape),
            cells=math.prod(region.shape),
            dt_s=region.time_step,
            steps=region.steps,
            wall_seconds=wall_time,
            measured=dict(
                T=t_val,
                R=r_val,
                n_eff=neff_val,
                epsilon_sha256=eps_sha
            )
        )
        report['cases'].append(row)
        t_obs_values.append(t_val)
        neff_obs_values.append(neff_val)

        del epsilon, width, launch, sc, project, region
        if dev == 'cuda':
            torch.cuda.empty_cache()

    report['simulation_executed'] = True

    # Compute ladder statistics
    stats_t = ladder_statistics(t_obs_values, complete=True)
    stats_neff = ladder_statistics(neff_obs_values, complete=True)

    report['observables'] = {
        'T': stats_t,
        'n_eff': stats_neff
    }

    passed = stats_t['C1_pass'] and not stats_t['non_monotone'] and not stats_neff['non_monotone']
    report['status'] = 'completed' if passed else 'stopped-non-monotone'
    report['verdict'] = 'accepted' if passed else 'rejected'

    if run_dir:
        run_path = Path(run_dir)
        (run_path / 'tooth-convergence.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8')
        readout = standard_readout(report, STAGES, STAGE_SOURCE)
        (run_path / 'readout.json').write_text(
            json.dumps(readout, indent=2) + '\n', encoding='utf-8')
        (run_path / 'verdict').write_text(report['verdict'] + '\n', encoding='utf-8')

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Wave C — LEG 47: M1 Tooth Width Monotonic Convergence Restoration")
    parser.add_argument("config_pos", nargs="?", default=None, help="Path to config JSON (positional)")
    parser.add_argument("--config", "-c", dest="config_flag", default=None, help="Path to config JSON (--config flag)")
    parser.add_argument("--run-dir", default=None, help="Target run directory (overrides RUN_DIR_ABS)")

    args, unknown = parser.parse_known_args()

    config_target = args.config_flag or args.config_pos
    if not config_target and 'PARITY_CONFIG' in os.environ:
        config_target = os.environ['PARITY_CONFIG']

    if config_target:
        config_path = Path(config_target)
    else:
        config_path = Path(__file__).resolve().parents[1] / 'configs/leg47-m1-convergence.json'

    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    run_dir_str = args.run_dir or os.environ.get('RUN_DIR_ABS')
    run_dir = Path(run_dir_str) if run_dir_str else Path.cwd()

    report = run_m1_convergence(cfg, run_dir=run_dir)
    print(f'[leg47_m1_convergence] Completed run: status={report["status"]}, verdict={report["verdict"]}')
    if report['verdict'] in ('accepted', 'passed'):
        sys.exit(0)
    else:
        sys.exit(53)


if __name__ == '__main__':
    main()
