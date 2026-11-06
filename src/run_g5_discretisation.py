# ---------------------------------------------------------------------------
# COPIED, not invented. Source: the reference project 20261010_torchfdtd-lumerical-parity
#   src/run_g5_discretisation.py
#   at commit 9b0f61a41b026342420422b7d208a046d7b6e45b
#   sha256(source) = 6e7dacc0839a9e2d3b7c93d85d6c6e5522bffbf8eb774278b4fe4e473cf2b9b5
#   copied 2026-10-18 by leg 1 (F0+F8). This project owns this copy now.
# ---------------------------------------------------------------------------
"""Forward-only G5 measurement, using the unchanged G1b N8 fixture.

Execute via tools/run.sh. Nine matrix cells, three rigid shifts per cell.
No derivative, Taylor test, or optimization. Criteria are plan.md G5 verbatim.
"""
import os
import sys
from pathlib import Path

if os.environ.get('PARITY_SNAPSHOT'):
    sys.path.insert(0, str(Path(os.environ['PARITY_SNAPSHOT']) / 'src/solver'))

from run_g1b_gradient_taylor import (
    fixture, np, torch, torchfdtd, field_axes, AdjointOptions,
    ModeInjectedPlaneSimulation, modal_plane_amplitudes,
    prepare_aperture_modal_launch, load_run, write_json, json, time, C0, hashlib,
)

BRIDGES = ('arithmetic-cell', 'normal-harmonic', 'yee-point')
MESHES = ((.04, .02), (.02, .01), (.01, .005))
SHIFTS = (0., 1/3, 2/3)
SECOND_ORDER_RULE = ('p_S >= 1.7 for some bridge and S/J small enough that 5 S is a '
    'resolvable delta_J: the objective is second-order-class; the next line is a '
    'step-size / bridge fix for the gradient, not a solver change. G1b may be '
    're-registered at the selected bridge.')
BLOCK_RULE = ('No bridge reaches p_S >= 1.7: the objective\'s value is first-order '
    'in the mesh, no gradient can drive it at whole-cell steps, and the optimisation '
    'route is blocked at the discretisation level; the line becomes solver-order '
    'work (e.g. sub-cell or higher-order interface treatment), which is a different '
    'project scope.')


def g5_fixture(cfg, case, device='cuda', *, scene_factory=None):
    """Reuse G1b verbatim for arithmetic; other policies are the B7 rules.

    Only x-interface mixing changes. The y-layer completion, section used by
    the launch, source, all monitors, and 32 absolute edges stay those of G1b.
    normal-harmonic mixes Ex harmonically, Ey/Ez arithmetically. yee-point
    samples at component Yee nodes, with the existing half-value boundary tie.
    """
    kwargs = {} if scene_factory is None else dict(scene_factory=scene_factory)
    sc, section, p0, arithmetic_build = fixture(cfg, case, device=device, **kwargs)
    sampling = case['sampling']
    if sampling not in BRIDGES:
        raise ValueError('Unregistered G5 bridge')
    if sampling == 'arithmetic-cell':
        return sc, section, p0, arithmetic_build
    dx = case['dx_um']; ec = sc['ec']; eb = sc['eb']; r = sc['r']
    geometry = []
    for component in ('Ex', 'Ey', 'Ez'):
        axes = field_axes(r, component); x = axes[0]+sc['xc']; y = axes[1]
        baseline = np.broadcast_to(section(y[None, :, None], 0), r.shape).copy()
        cy = np.where((y > 1e-9) & (y < .22-1e-9), 1., 0.)
        cy = np.where((abs(y) < 1e-9) | (abs(y-.22) < 1e-9), .5, cy)
        geometry.append(tuple(torch.tensor(a, dtype=torch.float64, device=device)
                              for a in (x, cy, baseline)))

    def build(parameters, whole_geometry_shift_cells=0.):
        pairs = parameters.reshape(-1, 2)*dx
        leads = torch.stack((parameters.new_tensor([-1e6, whole_geometry_shift_cells*dx]),
            parameters.new_tensor([sc['end']+whole_geometry_shift_cells*dx, 1e6])))
        intervals = torch.cat((leads, pairs))
        intervals = intervals[torch.argsort(intervals[:, 0])]
        parts = []
        for i, (x, cy, baseline) in enumerate(geometry):
            fill = torch.zeros_like(x)
            rightmost = x.new_tensor(-1e6)
            for a, b in intervals:
                if sampling == 'yee-point':
                    fill = torch.maximum(fill, torch.where((x > a+1e-9) & (x < b-1e-9), 1.,
                        torch.where((abs(x-a) < 1e-9) | (abs(x-b) < 1e-9), .5, 0.)).to(x))
                else:
                    b = torch.maximum(a, b)
                    left = torch.maximum(a, rightmost)
                    fill = fill + torch.clamp(torch.minimum(x+dx/2, b)
                        - torch.maximum(x-dx/2, left), min=0)/dx
                    rightmost = torch.maximum(rightmost, b)
            mixed = (1/(fill/ec+(1-fill)/eb) if sampling == 'normal-harmonic' and i == 0
                     else eb+(ec-eb)*fill)
            parts.append(baseline-(ec-mixed[:, None, None])*cy[None, :, None])
        return torch.stack(parts, dim=-1)
    return sc, section, p0, build


def fit_spread(cases):
    """OLS in natural logs; residuals are observed minus fitted log(S)."""
    ordered = sorted(cases, key=lambda c: c['config']['dx_um'], reverse=True)
    if len(ordered) != 3 or any(c.get('S', 0) <= 0 for c in ordered):
        return dict(status='not_established', reason='Three positive finite spreads required for log fit')
    x = np.log([c['config']['dx_um'] for c in ordered])
    y = np.log([c['S'] for c in ordered])
    if not np.isfinite(y).all():
        return dict(status='not_established', reason='Nonfinite spread')
    slope, intercept = np.polyfit(x, y, 1)
    residual = y-(slope*x+intercept)
    return dict(status='reported', p_S=float(slope), intercept=float(intercept),
        spacing_um=[c['config']['dx_um'] for c in ordered],
        log_residuals=residual.tolist(), log_residual_rmse=float(np.sqrt(np.mean(residual**2))),
        log_residual_sum_squares=float(residual@residual),
        order_class=('second-order class' if slope >= 1.7 else
                     'first-order class' if slope <= 1.3 else 'unresolved'))


def judge(cases):
    expected = {(dx, dy, bridge) for dx, dy in MESHES for bridge in BRIDGES}
    completed = [c for c in cases if c['status'] == 'completed']
    found = {(c['config']['dx_um'], c['config']['dy_um'], c['config']['sampling']) for c in completed}
    coverage = (len(cases) == len(completed) == 9 and found == expected and
                all(np.isfinite([c['J'], c['S']]).all() for c in completed))
    fits = {b: fit_spread([c for c in completed if c['config']['sampling'] == b]) for b in BRIDGES}
    finest = [c for c in completed if c['config']['dx_um'] == .01]
    ratios = {c['config']['sampling']: dict(S_over_J=c['S']/c['J'],
              required_relative_gain_5S_over_J=5*c['S']/c['J']) for c in finest}
    candidate = min(finest, key=lambda c: c['S'])['config']['sampling'] if len(finest) == 3 else None
    any_second = any(f.get('p_S', -np.inf) >= 1.7 for f in fits.values())
    blocked = coverage and not any_second
    return dict(V1='pass' if coverage else 'incomplete',
        V2=dict(status='reported' if coverage and all(f['status'] == 'reported' for f in fits.values())
                else 'incomplete', bridges=fits),
        V3=dict(status='not_established', finest=ratios,
            minimum_S_over_J_over_matrix=min((c['S']/c['J'] for c in completed), default=None),
            reason='G4(a) registers delta_J >= 5 S but no achievable relative-gain bound; '
                   'a forward baseline cannot establish a resolvable design gain. No bound is invented.'),
        V4=dict(status='reported' if candidate else 'incomplete', candidate_objective_carrier=candidate,
                derivative_bridge='arithmetic-cell', selection_made=False),
        verdict=('blocked_at_discretisation_level' if blocked else
                 'second_order_class_gain_resolvability_not_established' if coverage and any_second else 'incomplete'),
        optimisation_route_blocked_at_discretisation_level=blocked if coverage else None,
        optimisation_authorized=False, acceptance_granted=False,
        registered_second_order_rule=SECOND_ORDER_RULE, registered_block_rule=BLOCK_RULE)


def main():
    cfg, run = load_run()
    effective = json.loads(Path(os.environ['EFFECTIVE_JSON']).read_text())
    report = dict(policy=cfg['policy'], run_id=run.name, status='in_progress', cases=[],
        objective='J = P_up / P_in; top upward flux / (near forward modal amplitude squared * unit-mode flux)',
        hardware=torch.cuda.get_device_name(), precision='float64',
        solver_import=str(Path(torchfdtd.__file__).relative_to(Path(os.environ['PARITY_SNAPSHOT']))),
        acceptance_granted=False, physical_mapping_certified=False,
        adjoint_executed=False, optimisation_executed=False, simulation_executed=False)
    effective['conventions'].update(objective=report['objective'], normal='+y for top',
        denominator='near forward modal power times launch-basis flux',
        whole_geometry_shift_includes_guide_ends=True, source_and_monitors_fixed=True)
    # Register every cell before execution: refused/missing cells remain visible.
    report['cases'] = [dict(id=c['id'], config=c, run_id=run.name, status='pending', evaluations=[])
                       for c in cfg['cases']]
    def save():
        write_json(run/'readout.json', report)
        write_json(os.environ['EFFECTIVE_JSON'], effective)
    save()
    for case in report['cases']:
        c = case['config']; started = time.perf_counter()
        case['status'] = 'in_progress'; save()
        try:
            sc, section, p0, build = g5_fixture(cfg, c)
            r = sc['r']; p = sc['p']
            assert p0.numel() == 32 and c['periods'] == 8 and c['duration_fs'] == 500
            modes = [prepare_aperture_modal_launch(p, section, mode_index=i, num_modes=2) for i in range(2)]
            scores = [float(np.sum(abs(m.mode.fields[..., 1])**2)/np.sum(abs(m.mode.fields[..., [1, 2]])**2)) for m in modes]
            launch = modes[int(np.argmax(scores))]
            assert max(scores) > 1-1e-8 and launch.mode.maxwell_residual < 1e-5
            # The existing API admits checkpoint capacity even with no_grad.
            # Zero slots avoid reserving an unused adjoint at the finest mesh;
            # this does not alter forward updates, sources, or observations.
            assert cfg['forward_checkpoint_slots'] == 0
            model = ModeInjectedPlaneSimulation(p, launch, AdjointOptions(
                checkpoints=cfg['forward_checkpoint_slots'], gpu_budget_bytes=16*1024**3,
                backward_kernel='fused'))
            write_json(run/(c['id']+'-project.json'), p.model_dump(mode='json'))
            case.update(shape=list(r.shape), dt_s=r.time_step, steps=r.steps, edge_count=p0.numel(),
                base_edges_um=(p0*c['dx_um']).cpu().tolist(), launch_identity=launch.identity,
                launch_neff=launch.mode.neff, launch_mode_residual=launch.mode.maxwell_residual,
                top_plane_height_um=1.5, forward_checkpoint_slots=0)
            effective['resolved_cases'].append(dict(id=c['id'], shape=list(r.shape), steps=r.steps,
                dt_s=r.time_step, sampling=c['sampling'], edge_count=32,
                float64=True, forward_kernel='fused', monitor_kernel='fused',
                source_and_monitors_fixed=True, whole_geometry_shift_includes_guide_ends=True,
                forward_checkpoint_slots=0,
                launch_identity=launch.identity))
            save()
            for i, phase in enumerate(SHIFTS):
                torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                with torch.no_grad():
                    parameter = p0+phase
                    base = build(parameter, phase)
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
                    assert float(incident) > 0 and bool(torch.isfinite(objective))
                    torch.cuda.synchronize()
                arrays = {name+'_'+key: getattr(plane, key).detach().cpu().numpy()
                          for name, plane in planes.items()
                          for key in ('fields', 'weights', 'points_um', 'frequency_hz')}
                arrays.update({name+'_'+key: value.detach().cpu().numpy()
                               for name, values in amps.items() for key, value in values.items()})
                finite = all(bool(np.isfinite(a).all()) for a in arrays.values())
                row = dict(id=f'phase-{i}', adjoint=False, whole_geometry_shift_cells=phase,
                    whole_geometry_shift_um=phase*c['dx_um'], J=float(objective),
                    P_up_reduced=float(upward), P_in_reduced=float(incident),
                    unit_mode_flux_reduced=float(unit_flux), finite_fields=finite,
                    wall_seconds=time.perf_counter()-t, geometry_seconds=geometry_done-t,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    solver_report=dict(near.report),
                    epsilon_sha256=hashlib.sha256(base.cpu().numpy().tobytes()).hexdigest(),
                    edge_positions_um=(parameter*c['dx_um']).cpu().tolist())
                assert finite, 'Nonfinite monitor fields'
                np.savez_compressed(run/(c['id']+f'-phase-{i}-raw.npz'), **arrays)
                case['evaluations'].append(row)
                report['simulation_executed'] = True
                save()
                print(json.dumps(dict(case=c['id'], phase=phase, J=row['J'],
                                      wall_seconds=row['wall_seconds'])), flush=True)
                del objective, upward, incident, planes, amps, base, parameter, arrays, near, basis, unit_flux
            js = [row['J'] for row in case['evaluations']]
            case.update(J=js[0], S=max(js)-min(js), grid_phase=dict(shifts_cells=list(SHIFTS), J=js),
                        status='completed')
            del model, modes, launch, build, p0, sc
        except Exception as error:
            case.update(status='failed', error_type=type(error).__name__, error=str(error))
            print(json.dumps(dict(case=c['id'], status='failed', error=str(error))), flush=True)
        finally:
            case['case_wall_seconds'] = time.perf_counter()-started
            torch.cuda.empty_cache()
            save()
    report['verdicts'] = judge(report['cases'])
    complete = report['verdicts']['V1'] == 'pass'
    report['status'] = 'completed' if complete else 'incomplete'
    effective.update(stage=report['status'], simulation_executed=report['simulation_executed'])
    save()
    if not complete:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
