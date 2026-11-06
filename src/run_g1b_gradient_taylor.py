# ---------------------------------------------------------------------------
# COPIED, not invented. Source: the reference project 20261010_torchfdtd-lumerical-parity
#   src/run_g1b_gradient_taylor.py
#   at commit 9b0f61a41b026342420422b7d208a046d7b6e45b
#   sha256(source) = d0f0d15f3da4f47a8fc835ec0a474c5229e87661193cf7f9bb7dcbb3284beaeb
#   copied 2026-10-18 by leg 1 (F0+F8). This project owns this copy now.
# ---------------------------------------------------------------------------
"""G1b/G2: upward-power gradient machinery and one-adjoint cost, plan v2.1.

Run only through tools/run.sh. No optimizer, vendor data, or solver changes.
Parameters are absolute edge x positions in cells (physical x = dx * p).
N8 retains the existing eight narrow and eight wide teeth: 32 edges.
"""
import os
import sys
from pathlib import Path

# Execute the solver copied by launch.py, rather than an editable live install.
if os.environ.get('PARITY_SNAPSHOT'):
    sys.path.insert(0, str(Path(os.environ['PARITY_SNAPSHOT']) / 'src/solver'))

from parity import *
import contextlib
import hashlib
import signal
import torch
import torchfdtd
import torchfdtd.differentiable as differentiable
from torchfdtd import AdjointOptions
from torchfdtd.solver import field_axes
from torchfdtd.mode_injection import ModeInjectedPlaneSimulation, modal_plane_amplitudes
from torchfdtd.mode_branches import prepare_aperture_modal_launch
from run_b9_native_shape_n8 import scene


FAILURE_RULE = ('A direction with abs(4 (g.v)) > S and no eps^2 window, or a '
                'slope <= 1.2 in that direction: the first-order grid-phase term '
                'dominates the remainder. Stage 4 stops and the next line is '
                'value-level discretisation accuracy, not optimisation.')
INSUFFICIENCY_RULE = ('If no direction is resolvable at eps = 4, the window is too '
                     'short. Escalate once to eps = 8 and re-judge; if still '
                     'unresolvable, the gradient is below the grid-phase scale '
                     'and the loop is premature.')
G2_STOP_RULE = ('If the N30 adjoint exceeds 30 min or does not fit in 16 GiB, '
                'speed becomes binding and adjoint-path acceleration precedes '
                'the closed loop.')


@contextlib.contextmanager
def replay_counter():
    """Observe the existing scheduler; each replay calls restore exactly once.

    Includes zero-step replay calls. Replayed time steps are separately supplied
    by the solver. This temporary wrapper changes no state or computation.
    """
    counts = dict(checkpoint_replay_calls=0)
    original = differentiable._Checkpoints.restore
    def restore(owner, key):
        counts['checkpoint_replay_calls'] += 1
        return original(owner, key)
    differentiable._Checkpoints.restore = restore
    try:
        yield counts
    finally:
        differentiable._Checkpoints.restore = original


@contextlib.contextmanager
def adjoint_deadline(seconds):
    def stop(signum, frame):
        raise TimeoutError('One-adjoint wall time exceeded the registered 30 min stop')
    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def fixture(cfg, case, device='cuda', *, scene_factory=scene):
    """Reuse B9's Region/Project, B6's aperture and arithmetic-cell coefficients."""
    sc = scene_factory(cfg, dict(case, transition_nm=case['dx_um']*1000))
    r = sc['r']; dx = case['dx_um']; ec = sc['ec']; eb = sc['eb']
    def section(y, z):
        value = np.where((y < -2-1e-9) | ((y > 1e-9) & (y < .22-1e-9)),
                         ec, np.where(y > 2.2+1e-9, 1., eb))
        for boundary, average in ((-2, .5*(ec+eb)), (0, .5*(ec+eb)),
                                  (.22, .5*(ec+eb)), (2.2, .5*(eb+1))):
            value = np.where(abs(y-boundary) < 1e-9, average, value)
        return value
    edges = np.array([(k*.9+a, k*.9+a+w) for k in range(sc['N'])
                      for a, w in ((.26325, .126), (.6525, .2475))]).reshape(-1)
    p0 = torch.tensor(edges/dx, dtype=torch.float64, device=device)
    geometry = []
    for component in ('Ex', 'Ey', 'Ez'):
        axes = field_axes(r, component); x = axes[0]+sc['xc']; y = axes[1]
        baseline = np.broadcast_to(section(y[None, :, None], 0), r.shape).copy()
        cy = np.where((y > 1e-9) & (y < .22-1e-9), 1., 0.)
        cy = np.where((abs(y) < 1e-9) | (abs(y-.22) < 1e-9), .5, cy)
        geometry.append(tuple(torch.tensor(a, dtype=torch.float64, device=device)
                              for a in (x, cy, baseline)))
    def build(parameters, whole_geometry_shift_cells=0.):
        # Fixed source, mesh and monitors. Whole-geometry shifts include both
        # semi-infinite guide ends, in addition to all teeth; y layers are invariant.
        pairs = parameters.reshape(-1, 2)*dx
        leads = torch.stack((parameters.new_tensor([-1e6, whole_geometry_shift_cells*dx]),
                             parameters.new_tensor([sc['end']+whole_geometry_shift_cells*dx, 1e6])))
        intervals = torch.cat((leads, pairs))
        intervals = intervals[torch.argsort(intervals[:, 0])]
        parts = []
        for x, cy, baseline in geometry:
            fill = torch.zeros_like(x)
            rightmost = x.new_tensor(-1e6)
            for a, b in intervals:
                # Exact union overlap. Inverted teeth have zero width; overlapping
                # teeth are counted once. Whole-cell checks can change topology.
                b = torch.maximum(a, b)
                left = torch.maximum(a, rightmost)
                fill = fill + torch.clamp(torch.minimum(x+dx/2, b)
                                         - torch.maximum(x-dx/2, left), min=0)/dx
                rightmost = torch.maximum(rightmost, b)
            mixed = eb+(ec-eb)*fill
            parts.append(baseline-(ec-mixed[:, None, None])*cy[None, :, None])
        return torch.stack(parts, dim=-1)
    return sc, section, p0, build


def judge(rows, dots, gradient_norm, null_norm, scale):
    """Apply G1a thresholds verbatim. Only resolvable points vote."""
    directions = {}
    failures = []
    for name, dot in dots.items():
        points = sorted([r for r in rows if r['direction'] == name], key=lambda r: r['eps_cells'])
        for point in points:
            eps = point['eps_cells']
            point['resolvable'] = abs(eps*dot) > scale and point['r2'] > scale
            point['C2_sign_equal'] = bool(np.sign(point['secant']) == np.sign(dot))
        windows = []
        # The only registered sequences are (1,2,4) and, after escalation, (2,4,8).
        for start in range(len(points)-2):
            window = points[start:start+3]
            if all(p['resolvable'] for p in window):
                x = np.log([p['eps_cells'] for p in window])
                y = np.log([p['r2'] for p in window])
                slope, intercept = np.polyfit(x, y, 1)
                slope = float(slope)
                residuals = y-(slope*x+intercept)
                windows.append(dict(eps_cells=[p['eps_cells'] for p in window],
                                    least_squares_log_slope=slope, log_intercept=float(intercept),
                                    log_residuals=residuals.tolist(),
                                    log_residual_rmse=float(np.sqrt(np.mean(residuals**2))),
                                    log_residual_sum_squares=float(residuals@residuals),
                                    C1_pass=1.8 <= slope <= 2.2))
        has_window = any(w['C1_pass'] for w in windows)
        informative = abs(4*dot) > scale
        first_order = any(w['least_squares_log_slope'] <= 1.2 for w in windows)
        failure = informative and (not has_window or first_order)
        if failure:
            failures.append(name)
        voters = [p for p in points if p['resolvable']]
        directions[name] = dict(g_dot_v=dot, abs_g_dot_v=abs(dot), windows=windows,
                                resolvable_steps=[p['eps_cells'] for p in voters],
                                C1=('pass' if has_window and not first_order else
                                    'fail' if failure else 'insufficient'),
                                C2=('pass' if all(p['C2_sign_equal'] for p in voters) else 'fail')
                                   if voters else 'insufficient', registered_failure=failure)
    null_dot = dots['rigid shift']
    ratio = abs(null_dot)/(gradient_norm*null_norm) if gradient_norm > 0 else None
    active = [d for d in directions.values() if d['resolvable_steps']]
    c1 = 'fail' if failures else ('pass' if active and all(d['C1'] == 'pass' for d in active) else 'insufficient')
    c2 = 'fail' if any(d['C2'] == 'fail' for d in active) else ('pass' if active else 'insufficient')
    return dict(S=scale, gradient_norm=gradient_norm, directions=directions,
                C1=c1, C2=c2, C3=('pass' if ratio <= .05 else 'fail') if ratio is not None else 'insufficient',
                C3_normalized_null=ratio, failure_directions=failures,
                no_direction_resolvable_at_4=not any(4 in d['resolvable_steps'] for d in directions.values()),
                failure_rule=FAILURE_RULE, insufficiency_rule=INSUFFICIENCY_RULE,
                stage4_stops=bool(failures or c2 == 'fail' or ratio is None or ratio > .05))


def main(*, fixture_factory=fixture, capability_probe=None, refusal_handler=None):
    cfg, run = load_run()
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    report = dict(policy=cfg['policy'], run_id=run.name, status='in_progress', cases=[],
                  objective='J = top upward flux / (near forward modal amplitude squared * unit-mode flux)',
                  parameter_units='cells; x_um = dx_um * p', hardware=torch.cuda.get_device_name(),
                  solver_import=str(Path(torchfdtd.__file__).relative_to(Path(os.environ['PARITY_SNAPSHOT']))),
                  acceptance_granted=False, physical_mapping_certified=False, G2_stop_rule=G2_STOP_RULE)
    effective['conventions'].update(objective=report['objective'], parameter_units=report['parameter_units'],
                                    denominator='differentiable near forward modal power times launch-basis flux')
    def save():
        write_json(run/'readout.json', report)
        write_json(os.environ['EFFECTIVE_JSON'], effective)
    save()
    if capability_probe is not None:
        report['capability'] = capability_probe(cfg)
        save()
        if report['capability']['status'] == 'refused':
            refusal_handler(cfg, report, effective, save)
            return
    for case_cfg in cfg['cases']:
        case_started = time.perf_counter()
        case = dict(id=case_cfg['id'], config=case_cfg, evaluations=[], taylor_points=[], status='in_progress')
        report['cases'].append(case); save()
        sc, section, p0, build = fixture_factory(cfg, case_cfg)
        r = sc['r']; p = sc['p']; n = p0.numel()
        modes = [prepare_aperture_modal_launch(p, section, mode_index=i, num_modes=2) for i in range(2)]
        scores = [float(np.sum(abs(m.mode.fields[..., 1])**2)/np.sum(abs(m.mode.fields[..., [1, 2]])**2)) for m in modes]
        launch = modes[int(np.argmax(scores))]
        assert max(scores) > 1-1e-8 and launch.mode.maxwell_residual < 1e-5
        options = AdjointOptions(checkpoints=32, gpu_budget_bytes=16*1024**3,
                                 backward_kernel='fused', **case_cfg.get('checkpoint_options', {}))
        model = ModeInjectedPlaneSimulation(p, launch, options)
        write_json(run/(case_cfg['id']+'-project.json'), p.model_dump(mode='json'))
        case.update(shape=list(r.shape), dt_s=r.time_step, steps=r.steps, edge_count=n,
                    base_edges_um=(p0*case_cfg['dx_um']).cpu().tolist(),
                    checkpoint_slots=32, gpu_budget_bytes=16*1024**3,
                    checkpoint_options=case_cfg.get('checkpoint_options', {}),
                    launch_identity=launch.identity, launch_neff=launch.mode.neff,
                    top_plane_height_um=1.5, launch_mode_residual=launch.mode.maxwell_residual)
        effective['resolved_cases'].append(dict(id=case_cfg['id'], shape=list(r.shape), steps=r.steps,
            dt_s=r.time_step, sampling=case_cfg['sampling'], edge_count=n, parameter_units='cells',
            checkpoint_options=case_cfg.get('checkpoint_options', {}),
            source_and_monitors_fixed=True, whole_geometry_shift_includes_guide_ends=True,
            float64=True, forward_and_backward_kernels='fused', launch_identity=launch.identity))
        save()

        def evaluate(parameters, label, adjoint=False, shift_cells=0.):
            torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            raw = []
            parameter = parameters.detach().clone().requires_grad_(adjoint)
            if adjoint:
                def project_gradient(value):
                    raw.append(value.detach().clone())
                    return value-value.mean()
                parameter.register_hook(project_gradient)
            limits = adjoint_deadline(1800) if adjoint and sc['N'] == 30 else contextlib.nullcontext()
            with limits, replay_counter() as count, torch.set_grad_enabled(adjoint):
                base = build(parameter, shift_cells)
                geometry_done = time.perf_counter()
                planes = model(base, [C0/(cfg['wavelength_um']*1e-6)])
                amps = {k: modal_plane_amplitudes(planes[k], launch) for k in ('near', 'far', 'outside')}
                near = planes['near']
                coordinate = near.points_um_geometry if near.points_um_geometry is not None else near.points_um
                basis = torch.as_tensor(launch.detector_mode(float(coordinate[0, 0])).sample_plane(
                    near.points_um.detach().cpu().numpy()), device='cuda')
                unit_flux = .5*(basis[..., 1]*basis[..., 5].conj()-basis[..., 2]*basis[..., 4].conj()).real @ near.weights
                incident = amps['near']['forward'].abs().square().reshape(-1)[0]*unit_flux
                upward = planes['top'].flux()[0]
                objective = upward/incident
                assert float(incident.detach()) > 0 and bool(torch.isfinite(objective))
                torch.cuda.synchronize(); forward_done = time.perf_counter()
                gradient = torch.autograd.grad(objective, parameter)[0] if adjoint else None
                torch.cuda.synchronize(); finished = time.perf_counter()
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
            row = dict(id=label, adjoint=adjoint, J=float(objective.detach()),
                       P_up_reduced=float(upward.detach()), P_in_reduced=float(incident.detach()),
                       unit_mode_flux_reduced=float(unit_flux.detach()), whole_geometry_shift_cells=shift_cells,
                       wall_seconds=finished-started, geometry_seconds=geometry_done-started,
                       forward_and_objective_seconds=forward_done-geometry_done,
                       backward_seconds=finished-forward_done, gpu_peak_allocated_bytes=peak_allocated,
                       gpu_peak_reserved_bytes=peak_reserved, checkpoint_replay_calls=count['checkpoint_replay_calls'],
                       replayed_steps=planes['near'].report.get('replayed_steps'), solver_report=dict(planes['near'].report),
                       epsilon_sha256=hashlib.sha256(base.detach().cpu().numpy().tobytes()).hexdigest(),
                       edge_positions_um=(parameter.detach()*case_cfg['dx_um']).cpu().tolist(),
                       inverted_or_zero_teeth=int(((parameter[1::2]-parameter[::2]) <= 0).sum()))
            arrays = {}
            for name, plane in planes.items():
                for key in ('fields', 'weights', 'points_um', 'frequency_hz'):
                    arrays[name+'_'+key] = getattr(plane, key).detach().cpu().numpy()
            for name, values in amps.items():
                for key, value in values.items():
                    arrays[name+'_'+key] = value.detach().cpu().numpy()
            row['finite_fields'] = all(bool(np.isfinite(a).all()) for a in arrays.values())
            np.savez_compressed(run/(case_cfg['id']+'-'+label+'-raw.npz'), **arrays)
            result_gradient = None
            if adjoint:
                result_gradient = gradient.detach().cpu().numpy()
                row.update(gradient_projected_per_cell=result_gradient.tolist(),
                           gradient_raw_per_cell=raw[0].cpu().tolist(),
                           gradient_norm=float(np.linalg.norm(result_gradient)),
                           gradient_raw_norm=float(raw[0].norm()), gradient_raw_null=float(raw[0].sum()))
            case['evaluations'].append(row); save()
            print(json.dumps({k: v for k, v in row.items() if k not in
                              ('solver_report', 'edge_positions_um', 'gradient_raw_per_cell', 'gradient_projected_per_cell')}), flush=True)
            del objective, upward, incident, planes, amps, base, parameter, gradient, raw, arrays
            torch.cuda.empty_cache()
            return row, result_gradient

        try:
            baseline, g = evaluate(p0, 'baseline-adjoint', True)
            if capability_probe is not None:
                case['normal_harmonic_derivative_available'] = bool(np.isfinite(g).all())
                report['capability']['executed_adjoint_cases'] = [c['id'] for c in report['cases']
                    if c.get('normal_harmonic_derivative_available')]
                report['capability']['status'] = 'available_on_executed_meshes'
            case['adjoint_cost'] = {k: baseline[k] for k in ('wall_seconds', 'forward_and_objective_seconds',
                'backward_seconds', 'gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes',
                'checkpoint_replay_calls', 'replayed_steps')}
            case['G2_binding'] = baseline['wall_seconds'] > 1800 or baseline['gpu_peak_reserved_bytes'] > 16*1024**3
            save()
            if case_cfg.get('taylor', False):
                assert n == 32
                rng = np.random.default_rng(cfg['random_seed'])
                directions = {'sign-pattern': np.where(g >= 0, 1., -1.),
                              'random +-1': rng.choice([-1., 1.], size=n),
                              'uniform width': np.tile([-1., 1.], n//2), 'rigid shift': np.ones(n)}
                # G7: the direction a gradient step actually takes. Additive; absent key keeps
                # every earlier configuration byte-identical. |v| = 1, so g.v = |g|.
                if case_cfg.get('gradient_direction', False):
                    norm_g = float(np.linalg.norm(g))
                    assert norm_g > 0
                    directions['gradient unit'] = g/norm_g
                dots = {name: float(g @ v) for name, v in directions.items()}
                case['directions'] = {name: dict(v_cells=v.tolist(), norm=float(np.linalg.norm(v)),
                                               g_dot_v=dots[name]) for name, v in directions.items()}
                # S is three independent forwards; the adjoint forward is not substituted.
                phase_rows = [evaluate(p0+phase, f'phase-{i}', shift_cells=phase)[0]
                              for i, phase in enumerate((0., 1/3, 2/3))]
                scale = max(row['J'] for row in phase_rows)-min(row['J'] for row in phase_rows)
                case['grid_phase'] = dict(shifts_cells=[0., 1/3, 2/3], J=[row['J'] for row in phase_rows], S=scale)
                def steps(eps_set):
                    for name, v in directions.items():
                        for eps in eps_set:
                            result, _ = evaluate(p0+eps*torch.as_tensor(v, device=p0.device),
                                name.replace(' ', '-')+f'-eps{eps}', shift_cells=eps if name == 'rigid shift' else 0.)
                            delta = result['J']-baseline['J']
                            case['taylor_points'].append(dict(direction=name, eps_cells=eps,
                                evaluation_id=result['id'], J=result['J'], delta_J=delta, secant=delta/eps,
                                g_dot_v=dots[name], abs_eps_g_dot_v=abs(eps*dots[name]), r2=abs(delta-eps*dots[name])))
                            save()
                # Additive: absent keys keep the registered (1, 2, 4) ladder and the
                # escalation to 8, so every earlier configuration reproduces unchanged.
                steps(tuple(case_cfg.get('taylor_steps', (1, 2, 4))))
                verdict = judge(case['taylor_points'], dots, float(np.linalg.norm(g)), math.sqrt(n), scale)
                escalated = verdict['no_direction_resolvable_at_4'] and case_cfg.get('taylor_escalate', True)
                if escalated:
                    steps((8,))
                    verdict = judge(case['taylor_points'], dots, float(np.linalg.norm(g)), math.sqrt(n), scale)
                verdict['escalated_once_to_8'] = escalated
                verdict['gradient_below_grid_phase_scale'] = escalated and not any(
                    8 in d['resolvable_steps'] for d in verdict['directions'].values())
                if verdict['gradient_below_grid_phase_scale']:
                    verdict['stage4_stops'] = True
                case['verdicts'] = verdict
                case['C3_null_pair'] = dict(projected_g_dot_v_null=dots['rigid shift'],
                    finite_difference_null_per_cell={str(row['eps_cells']): row['secant']
                        for row in case['taylor_points'] if row['direction'] == 'rigid shift'},
                    raw_unprojected_g_dot_v_null=baseline['gradient_raw_null'])
            case['status'] = 'completed'
        except (TimeoutError, torch.OutOfMemoryError, ValueError) as error:
            case.update(status='stopped', error_type=type(error).__name__, error=str(error),
                        incomplete_adjoint_wall_seconds=time.perf_counter()-case_started,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            case['G2_binding'] = isinstance(error, (TimeoutError, torch.OutOfMemoryError)) or 'budget' in str(error).lower()
            report['status'] = 'incomplete'; save()
            if refusal_handler is not None and isinstance(error, ValueError) and not case['G2_binding']:
                report['capability'].update(status='refused', error_type=type(error).__name__,
                                             error=str(error), refused_case=case_cfg['id'])
                refusal_handler(cfg, report, effective, save)
                return
            raise
        case['case_wall_seconds'] = time.perf_counter()-case_started
        save()
        del model, modes, launch, build, p0, sc
        torch.cuda.empty_cache()
    report.update(status='completed', simulation_executed=True)
    effective.update(stage='completed', simulation_executed=True)
    save()


if __name__ == '__main__':
    main()
