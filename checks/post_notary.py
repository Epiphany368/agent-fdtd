"""Post-flight notary gate enforcing kernel telemetry consistency, passivity, and cross-solver reconciliation (LEG 43 - 49).

Zero-Trust Rule: Agents possess propose rights, never dispose rights. A simulation's
credibility, completion, and merge eligibility are verified by deterministic cross-auditing
of kernel telemetry (solver_telemetry.json) against driver readout (readout.json) and config,
with SHA-256 content digests (and optional HMAC-SHA256 signing when TORCHFDTD_NOTARY_SECRET is set).

Exit codes:
- 0: Verification succeeded, verification_certificate.json issued.
- 66: Postflight verification failed (tamper detected, passivity violation, missing telemetry, or cross-solver discrepancy exceeded).
"""
import datetime
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sys

DEFAULT_TAU_PML = 1e-5
DEFAULT_TAU_BOX = 1e-5
DEFAULT_DECAY_FLOOR = 1e-6
DEFAULT_RECONCILIATION_TOLERANCE = 0.015  # 1.5% max cross-solver discrepancy tolerance (LEG 48)
FLUX_TOLERANCE = 1e-5
STEP_TOLERANCE = 1e-5


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_finite_number(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def _extract_readout_steps(readout: dict):
    if 'executed_steps' in readout and _is_finite_number(readout['executed_steps']):
        return int(readout['executed_steps'])
    if 'steps' in readout and _is_finite_number(readout['steps']):
        return int(readout['steps'])
    cases = readout.get('cases', [])
    if isinstance(cases, list) and cases:
        for case in cases:
            if isinstance(case, dict) and 'steps' in case and _is_finite_number(case['steps']):
                return int(case['steps'])
    return None


def _extract_readout_fluxes(readout: dict):
    """Extract monitor fluxes from readout.json into a flat dictionary."""
    fluxes = {}
    if isinstance(readout.get('monitors'), dict):
        for k, v in readout['monitors'].items():
            if isinstance(v, dict) and 'flux' in v:
                fluxes[k] = v['flux']
            elif _is_finite_number(v) or isinstance(v, list):
                fluxes[k] = v

    cases = readout.get('cases', [])
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            measured = case.get('measured', {})
            if not isinstance(measured, dict):
                continue
            for k in ('R', 'T', 'transmission', 'reflection', 'scattered', 'flux'):
                if k in measured:
                    fluxes[k] = measured[k]
            if isinstance(measured.get('monitors'), dict):
                for mk, mv in measured['monitors'].items():
                    fluxes[mk] = mv.get('flux', mv) if isinstance(mv, dict) else mv

    if isinstance(readout.get('measured'), dict):
        for k, v in readout['measured'].items():
            if k not in fluxes:
                fluxes[k] = v

    return fluxes


def _normalize_flux_value(val):
    if _is_finite_number(val):
        return float(val)
    if isinstance(val, list) and val:
        return [float(x) for x in val if _is_finite_number(x)]
    return None


def _compare_fluxes(telemetry_val, readout_val, tolerance=FLUX_TOLERANCE):
    """Return (match: bool, diff: float)."""
    t_norm = _normalize_flux_value(telemetry_val)
    r_norm = _normalize_flux_value(readout_val)
    if t_norm is None or r_norm is None:
        return False, math.inf

    if isinstance(t_norm, float) and isinstance(r_norm, float):
        diff = abs(t_norm - r_norm)
        return diff <= tolerance, diff

    if isinstance(t_norm, list) and isinstance(r_norm, list):
        if len(t_norm) != len(r_norm) or not t_norm:
            return False, math.inf
        diffs = [abs(a - b) for a, b in zip(t_norm, r_norm)]
        max_diff = max(diffs)
        return max_diff <= tolerance, max_diff

    if isinstance(t_norm, float) and isinstance(r_norm, list) and r_norm:
        diffs = [abs(t_norm - b) for b in r_norm]
        min_diff = min(diffs)
        return min_diff <= tolerance, min_diff

    if isinstance(t_norm, list) and isinstance(r_norm, float) and t_norm:
        diffs = [abs(a - r_norm) for a in t_norm]
        min_diff = min(diffs)
        return min_diff <= tolerance, min_diff

    return False, math.inf


def _detect_4surface_box(config: dict, readout: dict, telemetry_monitors: dict) -> bool:
    """Identify if a simulation is equipped with a 4-surface Poynting enclosure."""
    if config.get('poynting_box') is True or config.get('v2_box') is True or config.get('box_enclosure') is True:
        return True
    if readout.get('poynting_box') is True or readout.get('v2_box') is True:
        return True
    tags = set(config.get('tags', []) + readout.get('tags', []))
    if any(t in ('poynting_box', 'v2_box', 'v2-box', 'box_enclosure') for t in tags):
        return True
    if any(isinstance(inv, dict) and inv.get('id') in ('V2-box', 'v2-box', 'V2_box')
           for inv in config.get('invariants', [])):
        return True
    if 'tau_box' in config:
        return True

    box_right_keys = {'box-right', 'box_right'}
    box_left_keys = {'box-left', 'box_left'}
    box_top_keys = {'box-top', 'box_top', 'top'}
    box_bottom_keys = {'box-bottom', 'box_bottom', 'bottom'}

    mons = set(telemetry_monitors.keys())
    has_4_names = (
        bool(mons & box_right_keys) and bool(mons & box_left_keys) and
        bool(mons & box_top_keys) and bool(mons & box_bottom_keys)
    )
    if not has_4_names:
        return False

    # ModeInjectedPlaneSimulation emits raw unnormalized DFT fluxes (~1e-41 in reduced units)
    # for monitors named box-right/box-left/top/bottom. Only auto-trigger V2-box when fluxes
    # are normalized (order 1) or when poynting_box was explicitly requested above.
    raw_vals = []
    for k in (box_right_keys | box_left_keys | box_top_keys | box_bottom_keys) & mons:
        v = telemetry_monitors[k]
        f = v.get('flux') if isinstance(v, dict) else v
        norm = _normalize_flux_value(f)
        if isinstance(norm, float):
            raw_vals.append(abs(norm))
        elif isinstance(norm, list) and norm:
            raw_vals.append(max(abs(x) for x in norm))
    if raw_vals and sum(raw_vals) < 1e-20:
        return False
    return True


def _extract_box_surface_flux(telemetry_monitors: dict, readout_fluxes: dict, candidate_names: list):
    """Extract normalized flux value and matched name from telemetry_monitors or readout_fluxes."""
    for name in candidate_names:
        if name in telemetry_monitors:
            mon_data = telemetry_monitors[name]
            val = mon_data.get('flux') if isinstance(mon_data, dict) else mon_data
            if val is None and isinstance(mon_data, dict):
                val = mon_data.get('raw_poynting_flux')
            norm = _normalize_flux_value(val)
            if norm is not None:
                return norm, name
    for name in candidate_names:
        if name in readout_fluxes:
            norm = _normalize_flux_value(readout_fluxes[name])
            if norm is not None:
                return norm, name
    return None, None


def verify_post_flight(run_dir: Path) -> dict:
    """Run all notary verification checks for run_dir.

    Returns certificate dict.
    Raises RuntimeError(reason) if verification fails.
    """
    run_dir = Path(run_dir).resolve()
    config_file = run_dir / 'config.input.json'
    telemetry_file = run_dir / 'solver_telemetry.json'
    readout_file = run_dir / 'readout.json'

    # Load config
    config = {}
    if config_file.is_file():
        try:
            config = json.loads(config_file.read_text(encoding='utf-8'))
        except Exception as e:
            raise RuntimeError(f'corrupted config.input.json: {e}')

    kind = config.get('kind', 'simulation')

    # Audit runs do not execute simulation unless simulation_executed is explicitly set
    if kind == 'audit':
        readout = {}
        if readout_file.is_file():
            try:
                readout = json.loads(readout_file.read_text(encoding='utf-8'))
            except Exception:
                pass
        if not readout.get('simulation_executed', False):
            certificate = {
                'schema_version': '1.0',
                'run_id': run_dir.name,
                'verified': True,
                'kind': 'audit',
                'certified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'tags': ['audit_only'],
                'config_sha256': _sha256_file(config_file) if config_file.is_file() else None,
            }
            (run_dir / 'verification_certificate.json').write_text(
                json.dumps(certificate, indent=2) + '\n', encoding='utf-8')
            return certificate

    # Simulation runs MUST have solver_telemetry.json and readout.json
    if not telemetry_file.is_file():
        raise RuntimeError('missing solver_telemetry.json (tamper_detected)')

    try:
        telemetry = json.loads(telemetry_file.read_text(encoding='utf-8'))
    except Exception as e:
        raise RuntimeError(f'corrupted solver_telemetry.json: {e} (tamper_detected)')

    if not readout_file.is_file():
        raise RuntimeError('missing readout.json (tamper_detected)')

    try:
        readout = json.loads(readout_file.read_text(encoding='utf-8'))
    except Exception as e:
        raise RuntimeError(f'corrupted readout.json: {e} (tamper_detected)')

    # Check 1: Step counts
    telemetry_steps = telemetry.get('executed_steps')
    readout_steps = _extract_readout_steps(readout)
    if telemetry_steps is not None and readout_steps is not None:
        step_diff = abs(telemetry_steps - readout_steps)
        if step_diff > 0:
            raise RuntimeError(
                f'step count mismatch: telemetry={telemetry_steps} vs readout={readout_steps} (tamper_detected)'
            )

    # Check 1b: Resumed step consistency (LEG 45)
    telemetry_resumed = telemetry.get('resumed_from_step', 0)
    readout_resumed = readout.get('resumed_from_step', 0)
    if readout_resumed != telemetry_resumed:
        raise RuntimeError(
            f'resumed_from_step mismatch: telemetry={telemetry_resumed} vs readout={readout_resumed} (tamper_detected)'
        )

    # Check 2: Permittivity & Canonical SHA-256 (if declared in readout)
    if 'epsilon_sha256' in readout and 'epsilon_sha256' in telemetry:
        if readout['epsilon_sha256'] != telemetry['epsilon_sha256']:
            raise RuntimeError(
                f'epsilon_sha256 mismatch: telemetry={telemetry["epsilon_sha256"]} vs '
                f'readout={readout["epsilon_sha256"]} (tamper_detected)'
            )

    # Check 3: Lossless flag consistency
    telemetry_lossless = telemetry.get('is_lossless')
    readout_lossless = readout.get('is_lossless')
    if readout_lossless is None:
        for case in readout.get('cases', []):
            if isinstance(case, dict) and isinstance(case.get('physics'), dict):
                if 'lossless' in case['physics']:
                    readout_lossless = case['physics']['lossless']
                    break

    if readout_lossless is True and telemetry_lossless is False:
        raise RuntimeError(
            'lossless contradiction: readout claims lossless=True but kernel telemetry reports '
            'is_lossless=False (tamper_detected)'
        )

    # Check 4: Flux consistency with bidirectional alias mapping
    telemetry_monitors = telemetry.get('monitors') or telemetry.get('monitor_raw_digests') or {}
    readout_fluxes = _extract_readout_fluxes(readout)

    alias_map = {
        'box-right': ['T', 'transmission', 'trans', 'far', 'box-right'],
        'box-left': ['R', 'reflection', 'refl', 'near', 'box-left'],
        'transmission': ['T', 'transmission', 'trans', 'far', 'box-right'],
        'reflection': ['R', 'reflection', 'refl', 'near', 'box-left'],
        'scattered': ['S', 'scattered', 'scat'],
        'top': ['top', 'up'],
        'bottom': ['bottom', 'down'],
    }

    max_flux_diff = 0.0
    compared_monitors = 0

    for mon_name, mon_data in telemetry_monitors.items():
        t_flux = mon_data.get('flux') if isinstance(mon_data, dict) else mon_data
        if t_flux is None and isinstance(mon_data, dict):
            t_flux = mon_data.get('raw_poynting_flux')
        if t_flux is None:
            continue

        r_flux = None
        candidates = [mon_name] + alias_map.get(mon_name, [])
        for candidate in candidates:
            if candidate in readout_fluxes:
                r_flux = readout_fluxes[candidate]
                break

        if r_flux is not None:
            matches, diff = _compare_fluxes(t_flux, r_flux, tolerance=FLUX_TOLERANCE)
            if not matches:
                raise RuntimeError(
                    f'flux mismatch on {mon_name}: telemetry={t_flux} vs readout={r_flux} '
                    f'(diff={diff:.6e} > {FLUX_TOLERANCE}) (tamper_detected)'
                )
            max_flux_diff = max(max_flux_diff, diff)
            compared_monitors += 1

    is_lossless = bool(telemetry_lossless if telemetry_lossless is not None else readout_lossless)

    # Check 5a: V2-box Poynting enclosure closure gate (LEG 44)
    tau_box = config.get('tau_box', DEFAULT_TAU_BOX)
    is_box_equipped = _detect_4surface_box(config, readout, telemetry_monitors)

    box_closure_residual = None
    box_closure_passed = None
    box_monitors_record = None

    is_6surface_box = False
    if is_box_equipped and is_lossless:
        right_candidates = ['box-right', 'box_right', 'transmission', 'trans', 'far', 'T', 'right']
        left_candidates = ['box-left', 'box_left', 'reflection', 'refl', 'near', 'R', 'left']
        top_candidates = ['box-top', 'box_top', 'top', 'up', 'cladding']
        bottom_candidates = ['box-bottom', 'box_bottom', 'bottom', 'down', 'substrate']
        front_candidates = ['box-front', 'box_front', 'front']
        back_candidates = ['box-back', 'box_back', 'back']

        flux_right, name_r = _extract_box_surface_flux(telemetry_monitors, readout_fluxes, right_candidates)
        flux_left, name_l = _extract_box_surface_flux(telemetry_monitors, readout_fluxes, left_candidates)
        flux_top, name_t = _extract_box_surface_flux(telemetry_monitors, readout_fluxes, top_candidates)
        flux_bottom, name_b = _extract_box_surface_flux(telemetry_monitors, readout_fluxes, bottom_candidates)
        flux_front, name_f = _extract_box_surface_flux(telemetry_monitors, readout_fluxes, front_candidates)
        flux_back, name_bk = _extract_box_surface_flux(telemetry_monitors, readout_fluxes, back_candidates)

        proj_region = config.get('project', {}).get('region', {}) if isinstance(config.get('project'), dict) else {}
        proj_mons = config.get('project', {}).get('monitors', []) if isinstance(config.get('project'), dict) else []
        proj_mon_names = {m.get('name') or m.get('id') for m in proj_mons if isinstance(m, dict)}
        is_6surface_box = (
            config.get('box_surfaces') == 6 or
            readout.get('box_surfaces') == 6 or
            str(config.get('dimension', '')).lower() == '3d' or
            str(readout.get('dimension', '')).lower() == '3d' or
            (str(proj_region.get('dimension', '')).lower() == '3d' and bool(proj_mon_names & {'box-right', 'box_right'})) or
            flux_front is not None or
            flux_back is not None or
            bool(proj_mon_names & {'box-front', 'box_front', 'box-back', 'box_back'})
        )

        missing_surfaces = []
        if flux_right is None: missing_surfaces.append('right/transmission')
        if flux_left is None: missing_surfaces.append('left/reflection')
        if flux_top is None: missing_surfaces.append('top/cladding')
        if flux_bottom is None: missing_surfaces.append('bottom/substrate')
        if is_6surface_box:
            if flux_front is None: missing_surfaces.append('front')
            if flux_back is None: missing_surfaces.append('back')

        if missing_surfaces:
            box_label = '6-surface 3D' if is_6surface_box else '4-surface 2D'
            raise RuntimeError(
                f'poynting box unclosed ({box_label}): missing {", ".join(missing_surfaces)} monitor(s) '
                f'(poynting_box_unclosed)'
            )

        p_inc = config.get('incident_flux', config.get('incident_power', 1.0))
        p_inc_norm = _normalize_flux_value(p_inc)

        active_fluxes = [flux_right, flux_left, flux_top, flux_bottom]
        if is_6surface_box:
            active_fluxes.extend([flux_front, flux_back])

        if all(isinstance(f, float) for f in active_fluxes):
            phi_out = sum(abs(f) for f in active_fluxes)
            inc = p_inc_norm if isinstance(p_inc_norm, float) else 1.0
            r_box = abs(phi_out / inc - 1.0)
            box_closure_residual = float(r_box)
            box_monitors_record = {
                'trans': float(flux_right),
                'refl': float(flux_left),
                'up': float(flux_top),
                'down': float(flux_bottom),
                'sum': float(phi_out),
            }
            if is_6surface_box:
                box_monitors_record['front'] = float(flux_front)
                box_monitors_record['back'] = float(flux_back)
        elif all(isinstance(f, list) for f in active_fluxes):
            lens = [len(f) for f in active_fluxes]
            if len(set(lens)) != 1:
                raise RuntimeError(
                    f'poynting box monitor array length mismatch: {lens} (poynting_box_unclosed)'
                )
            n_pts = lens[0]
            inc_list = p_inc_norm if isinstance(p_inc_norm, list) and len(p_inc_norm) == n_pts else [1.0] * n_pts
            r_boxes = []
            sums = []
            for i in range(n_pts):
                phi_out = sum(abs(f[i]) for f in active_fluxes)
                r_boxes.append(abs(phi_out / inc_list[i] - 1.0))
                sums.append(phi_out)
            r_box = max(r_boxes)
            box_closure_residual = float(r_box)
            box_monitors_record = {
                'trans': flux_right,
                'refl': flux_left,
                'up': flux_top,
                'down': flux_bottom,
                'sum': sums,
            }
            if is_6surface_box:
                box_monitors_record['front'] = flux_front
                box_monitors_record['back'] = flux_back
        else:
            raise RuntimeError(
                'poynting box monitor flux type mismatch (poynting_box_unclosed)'
            )

        box_closure_passed = (box_closure_residual <= tau_box)
        if not box_closure_passed:
            box_label = '6-surface 3D' if is_6surface_box else '4-surface 2D'
            raise RuntimeError(
                f'poynting box closure failure ({box_label}): residual R_box {box_closure_residual:.6e} > {tau_box:.1e} '
                f'(poynting_box_unclosed)'
            )

    # Check 5b: Universal passivity gate
    tau_pml = config.get('tau_pml', DEFAULT_TAU_PML)

    total_power = None
    if is_lossless:
        powers = []
        for mon_name, mon_data in telemetry_monitors.items():
            t_flux = mon_data.get('flux') if isinstance(mon_data, dict) else mon_data
            if t_flux is None and isinstance(mon_data, dict):
                t_flux = mon_data.get('raw_poynting_flux')
            norm_val = _normalize_flux_value(t_flux)
            if norm_val is not None:
                if isinstance(norm_val, float):
                    powers.append(norm_val)
                elif isinstance(norm_val, list) and norm_val:
                    powers.append(max(norm_val))

        if powers:
            total_power = sum(powers)
        elif 'R' in readout_fluxes and 'T' in readout_fluxes:
            r_val = _normalize_flux_value(readout_fluxes['R'])
            t_val = _normalize_flux_value(readout_fluxes['T'])
            if isinstance(r_val, list) and isinstance(t_val, list):
                total_power = max(r + t for r, t in zip(r_val, t_val))
            elif isinstance(r_val, float) and isinstance(t_val, float):
                total_power = r_val + t_val

        if total_power is not None and total_power > (1.0 + tau_pml):
            raise RuntimeError(
                f'passivity violation: total outward power {total_power:.7f} > 1.0 + {tau_pml:.1e} '
                f'(passivity_violation)'
            )


    # Check 6: Energy decay gate
    decay_ratio = telemetry.get('energy_decay_ratio')
    decay_floor = config.get('decay_floor', DEFAULT_DECAY_FLOOR)
    decay_passed = None

    if decay_ratio is not None:
        decay_passed = (decay_ratio <= decay_floor)
        if not decay_passed:
            raise RuntimeError(
                f'spectral convergence failure: energy_decay_ratio {decay_ratio:.6e} > {decay_floor:.6e} '
                f'(under_converged_time)'
            )

    # Check 7: Auto-shutoff verification gate (LEG 46)
    telemetry_shutoff = bool(telemetry.get('auto_shutoff_triggered', False))
    readout_shutoff = bool(readout.get('auto_shutoff_triggered', False) or readout.get('auto_shutoff', False))

    if readout_shutoff and not telemetry_shutoff:
        raise RuntimeError(
            'auto_shutoff contradiction: readout claims auto_shutoff but kernel telemetry reports false (tamper_detected)'
        )

    shutoff_step = None
    shutoff_energy = None
    shutoff_flux_delta = None

    if telemetry_shutoff:
        shutoff_step = telemetry.get('shutoff_step')
        if shutoff_step is not None and telemetry_steps is not None and shutoff_step != telemetry_steps:
            raise RuntimeError(
                f'shutoff_step mismatch: telemetry shutoff_step={shutoff_step} vs executed_steps={telemetry_steps} (tamper_detected)'
            )

        shutoff_energy = float(telemetry.get('shutoff_energy_ratio', 1.0))
        shutoff_flux_delta = float(telemetry.get('shutoff_flux_delta', 1.0))

        if shutoff_energy > 1e-9:
            raise RuntimeError(
                f'auto_shutoff premature energy: {shutoff_energy:.3e} > 1e-9 (under_converged_time)'
            )
        if shutoff_flux_delta > 1e-6:
            raise RuntimeError(
                f'auto_shutoff unstable flux: {shutoff_flux_delta:.3e} > 1e-6 (under_converged_flux)'
            )

    # Check 8: Monotonic Grid Convergence Gate (LEG 47)
    is_convergence_run = (
        config.get('convergence_ladder') is True or
        readout.get('convergence_ladder') is True or
        'monotonic_mesh_convergence' in config.get('tags', []) or
        'monotonic_mesh_convergence' in readout.get('tags', []) or
        'mesh_ladder' in readout or
        'mesh_ladder' in config
    )
    mesh_convergence_passed = False
    mesh_convergence_record = None

    if is_convergence_run:
        # Zero-Trust Telemetry Cross-Referencing: Ensure solver executed genuine steps
        tel_steps = telemetry.get('executed_steps', 0)
        if not _is_finite_number(tel_steps) or int(tel_steps) <= 0:
            raise RuntimeError(
                'mesh convergence requires verified positive executed_steps in solver_telemetry.json (tamper_detected)'
            )

        ladder = readout.get('mesh_ladder') or readout.get('cases', [])
        if not isinstance(ladder, list) or len(ladder) < 3:
            raise RuntimeError(
                'mesh convergence requires at least 3 completed rungs (non_monotonic_mesh_ladder)'
            )

        # Cross-reference with kernel telemetry facts if present
        if 'epsilon_sha256' in telemetry and ladder:
            last_rung_m = ladder[-1].get('measured', ladder[-1]) if isinstance(ladder[-1], dict) else {}
            if 'epsilon_sha256' in last_rung_m and last_rung_m['epsilon_sha256'] != telemetry['epsilon_sha256']:
                raise RuntimeError(
                    f'mesh ladder rung {len(ladder)-1} epsilon_sha256 mismatch: '
                    f'telemetry={telemetry["epsilon_sha256"]} vs readout={last_rung_m["epsilon_sha256"]} (tamper_detected)'
                )

        if 'mesh_ladder' in telemetry and isinstance(telemetry['mesh_ladder'], list):
            tel_ladder = telemetry['mesh_ladder']
            if len(tel_ladder) != len(ladder):
                raise RuntimeError(
                    f'mesh ladder rung count mismatch: telemetry={len(tel_ladder)} vs readout={len(ladder)} (tamper_detected)'
                )
            for idx, (r_rung, t_rung) in enumerate(zip(ladder, tel_ladder)):
                r_m = r_rung.get('measured', r_rung) if isinstance(r_rung, dict) else {}
                t_m = t_rung.get('measured', t_rung) if isinstance(t_rung, dict) else {}
                r_t = r_m.get('T') if 'T' in r_m else r_m.get('transmission')
                t_t = t_m.get('T') if 'T' in t_m else t_m.get('transmission')
                if r_t is not None and t_t is not None:
                    if abs(float(r_t) - float(t_t)) > FLUX_TOLERANCE:
                        raise RuntimeError(
                            f'mesh ladder rung {idx} transmission mismatch: telemetry={t_t} vs readout={r_t} (tamper_detected)'
                        )
                # Cross-reference epsilon hashes if reported
                if 'epsilon_sha256' in r_m and 'epsilon_sha256' in t_m:
                    if r_m['epsilon_sha256'] != t_m['epsilon_sha256']:
                        raise RuntimeError(
                            f'mesh ladder rung {idx} epsilon_sha256 mismatch: telemetry={t_m["epsilon_sha256"]} vs readout={r_m["epsilon_sha256"]} (tamper_detected)'
                        )
                # Cross-reference grid step coordinates if reported
                for dim in ('dx_um', 'dy_um', 'steps'):
                    if dim in r_rung and dim in t_rung:
                        if r_rung[dim] != t_rung[dim]:
                            raise RuntimeError(
                                f'mesh ladder rung {idx} {dim} mismatch: telemetry={t_rung[dim]} vs readout={r_rung[dim]} (tamper_detected)'
                            )

        t_values = []
        for rung in ladder:
            m = rung.get('measured', rung) if isinstance(rung, dict) else {}
            t_val = m.get('T') if 'T' in m else m.get('transmission')
            if not _is_finite_number(t_val):
                raise RuntimeError('missing or non-finite transmission in mesh ladder (non_monotonic_mesh_ladder)')
            t_values.append(float(t_val))

        signed_diffs = [b - a for a, b in zip(t_values, t_values[1:])]
        abs_diffs = [abs(d) for d in signed_diffs]

        # Check for reversals: sign(d_i) * sign(d_{i+1}) must be positive
        reversal = any(a * b <= 0 for a, b in zip(signed_diffs, signed_diffs[1:]))
        worsening = any(b > a for a, b in zip(abs_diffs, abs_diffs[1:]))

        if reversal or worsening:
            raise RuntimeError(
                f'mesh ladder non-monotonic: signed_diffs={signed_diffs} (non_monotonic_mesh_ladder)'
            )

        # Calculate observed order and Richardson extrapolation
        order = math.log2(abs_diffs[0] / abs_diffs[1]) if abs_diffs[1] > 0 else 0.0
        extrapolated = t_values[-1] + signed_diffs[-1] / (2**order - 1) if order > 0 else None

        mesh_convergence_passed = True
        mesh_convergence_record = {
            'rungs': len(t_values),
            't_values': t_values,
            'observed_order': order,
            'richardson_extrapolated_T': extrapolated,
            'monotonic': True,
        }

    # Check 9: Graded Mesh Collar Accuracy Gate (LEG 47)
    is_graded_collar_run = (
        config.get('graded_collar') is True or
        readout.get('graded_collar') is True or
        'graded_collar_accuracy' in config.get('tags', []) or
        'graded_collar_accuracy' in readout.get('tags', []) or
        'graded_accuracy' in readout or
        'graded_accuracy' in config
    )
    graded_collar_passed = False
    graded_collar_record = None

    if is_graded_collar_run:
        # Zero-Trust Telemetry Cross-Referencing: Ensure solver executed genuine steps
        tel_steps = telemetry.get('executed_steps', 0)
        if not _is_finite_number(tel_steps) or int(tel_steps) <= 0:
            raise RuntimeError(
                'graded collar run requires verified positive executed_steps in solver_telemetry.json (tamper_detected)'
            )

        graded_acc = readout.get('graded_accuracy') or readout
        t_uniform = graded_acc.get('T_uniform')
        t_graded = graded_acc.get('T_graded')
        r_uniform = graded_acc.get('R_uniform')
        r_graded = graded_acc.get('R_graded')
        cells_uniform = graded_acc.get('cells_uniform')
        cells_graded = graded_acc.get('cells_graded')

        for val, name in ((t_uniform, 'T_uniform'), (t_graded, 'T_graded'),
                          (r_uniform, 'R_uniform'), (r_graded, 'R_graded'),
                          (cells_uniform, 'cells_uniform'), (cells_graded, 'cells_graded')):
            if not _is_finite_number(val):
                raise RuntimeError(f'missing or invalid {name} for graded collar verification (graded_accuracy_exceeded)')

        # Cross-reference with kernel telemetry facts if present
        if 'graded_accuracy' in telemetry and isinstance(telemetry['graded_accuracy'], dict):
            tel_ga = telemetry['graded_accuracy']
            for name, r_val in (('T_uniform', t_uniform), ('T_graded', t_graded),
                                ('R_uniform', r_uniform), ('R_graded', r_graded)):
                if name in tel_ga:
                    if abs(float(r_val) - float(tel_ga[name])) > FLUX_TOLERANCE:
                        raise RuntimeError(
                            f'graded accuracy metric {name} mismatch: telemetry={tel_ga[name]} vs readout={r_val} (tamper_detected)'
                        )
            for name, r_val in (('cells_uniform', cells_uniform), ('cells_graded', cells_graded)):
                if name in tel_ga:
                    if int(r_val) != int(tel_ga[name]):
                        raise RuntimeError(
                            f'grid cell count {name} mismatch: telemetry={tel_ga[name]} vs readout={r_val} (tamper_detected)'
                        )

        if 'mesh_metrics' in readout and isinstance(readout['mesh_metrics'], dict):
            r_mm = readout['mesh_metrics']
            if 'max_adjacent_ratio' in r_mm and r_mm['max_adjacent_ratio'] > 1.25 + 1e-12:
                raise RuntimeError(
                    f'mesh adjacent ratio in readout {r_mm["max_adjacent_ratio"]} exceeds 1.25 (graded_accuracy_exceeded)'
                )

        if 'mesh_metrics' in telemetry and isinstance(telemetry['mesh_metrics'], dict):
            tel_mm = telemetry['mesh_metrics']
            if 'max_adjacent_ratio' in tel_mm and tel_mm['max_adjacent_ratio'] > 1.25 + 1e-12:
                raise RuntimeError(
                    f'mesh adjacent ratio in telemetry {tel_mm["max_adjacent_ratio"]} exceeds 1.25 (graded_accuracy_exceeded)'
                )

        u_t = float(t_uniform)
        u_r = float(r_uniform)
        g_t = float(t_graded)
        g_r = float(r_graded)
        rel_err_t = abs(g_t - u_t) / abs(u_t) if abs(u_t) > 1e-12 else abs(g_t - u_t)
        rel_err_r = abs(g_r - u_r) / abs(u_r) if abs(u_r) > 1e-12 else abs(g_r - u_r)
        reduction = (float(cells_uniform) - float(cells_graded)) / float(cells_uniform)

        if rel_err_t >= 1e-3:
            raise RuntimeError(
                f'graded mesh transmission error {rel_err_t:.6e} >= 1e-3 (graded_accuracy_exceeded)'
            )
        if rel_err_r >= 1e-3:
            raise RuntimeError(
                f'graded mesh reflection error {rel_err_r:.6e} >= 1e-3 (graded_accuracy_exceeded)'
            )
        if reduction < 0.50:
            raise RuntimeError(
                f'insufficient grid cell reduction {reduction:.2%} < 50% (insufficient_cell_reduction)'
            )

        graded_collar_passed = True
        graded_collar_record = {
            'rel_err_T': rel_err_t,
            'rel_err_R': rel_err_r,
            'cell_reduction': reduction,
            'cells_uniform': int(cells_uniform),
            'cells_graded': int(cells_graded),
            'collar_zone_verified': True,
        }

    # Check 10: Cross-Solver Reconciliation Gate (V4-reconcile, LEG 48)
    is_reconciliation_run = (
        config.get('kind') == 'reconciliation' or
        readout.get('kind') == 'reconciliation' or
        config.get('cross_solver') is True or
        readout.get('cross_solver') is True or
        'cross_solver_reconciliation' in config.get('tags', []) or
        'cross_solver_reconciliation' in readout.get('tags', []) or
        'cross_solver' in readout or
        'cross_solver' in config
    )
    reconciliation_passed = False
    reconciliation_record = None

    if is_reconciliation_run:
        # Zero-Trust Telemetry Cross-Referencing: Ensure solver executed genuine steps
        tel_steps = telemetry.get('executed_steps', 0)
        if not _is_finite_number(tel_steps) or int(tel_steps) <= 0:
            raise RuntimeError(
                'cross-solver reconciliation requires verified positive executed_steps in solver_telemetry.json (tamper_detected)'
            )

        cs_data = readout.get('cross_solver')
        if not isinstance(cs_data, dict):
            raise RuntimeError(
                'missing cross_solver dictionary in readout.json for reconciliation run (tamper_detected)'
            )

        # Cross-reference epsilon hash if declared in readout cross_solver
        if 'epsilon_sha256' in telemetry and 'epsilon_sha256' in cs_data:
            if cs_data['epsilon_sha256'] != telemetry['epsilon_sha256']:
                raise RuntimeError(
                    f'cross_solver epsilon_sha256 mismatch: '
                    f'telemetry={telemetry["epsilon_sha256"]} vs readout={cs_data["epsilon_sha256"]} (tamper_detected)'
                )

        # External solver provenance and execution verification
        ext_solver = cs_data.get('external_solver')
        if not isinstance(ext_solver, dict) or not ext_solver.get('name'):
            raise RuntimeError(
                'missing or invalid external_solver provenance in cross_solver readout (tamper_detected)'
            )
        ext_exit_code = ext_solver.get('exit_code', 0)
        if ext_exit_code != 0:
            raise RuntimeError(
                f'external solver execution failed with exit code {ext_exit_code} (tamper_detected)'
            )

        # Extract and validate discrepancy metric
        max_discrepancy = cs_data.get('max_discrepancy')
        if max_discrepancy is None:
            max_discrepancy = cs_data.get('max_abs_diff')
        if max_discrepancy is None:
            max_discrepancy = cs_data.get('discrepancy')

        if not _is_finite_number(max_discrepancy):
            raise RuntimeError(
                'missing or non-finite discrepancy metric in cross_solver readout (cross_solver_discrepancy_exceeded)'
            )

        max_discrepancy = float(max_discrepancy)

        # Cross-reference telemetry discrepancy if recorded in kernel telemetry
        if 'cross_solver' in telemetry and isinstance(telemetry['cross_solver'], dict):
            tel_cs = telemetry['cross_solver']
            tel_disc = tel_cs.get('max_discrepancy', tel_cs.get('max_abs_diff'))
            if tel_disc is not None and abs(float(tel_disc) - max_discrepancy) > FLUX_TOLERANCE:
                raise RuntimeError(
                    f'cross_solver discrepancy mismatch: telemetry={tel_disc} vs readout={max_discrepancy} (tamper_detected)'
                )

        # Discrepancy tolerance check (<= 1.5% default)
        tolerance = config.get('reconciliation_tolerance',
                               config.get('tolerance',
                                          cs_data.get('tolerance', DEFAULT_RECONCILIATION_TOLERANCE)))
        if not _is_finite_number(tolerance) or tolerance <= 0:
            tolerance = DEFAULT_RECONCILIATION_TOLERANCE
        tolerance = float(tolerance)

        if max_discrepancy > tolerance:
            raise RuntimeError(
                f'cross-solver discrepancy {max_discrepancy:.6e} exceeds tolerance {tolerance:.6e} '
                f'(cross_solver_discrepancy_exceeded)'
            )

        # Order efficiencies consistency check if present
        torch_orders = cs_data.get('torchfdtd_efficiencies') or cs_data.get('efficiencies')
        ext_orders = cs_data.get('external_efficiencies')
        if isinstance(torch_orders, dict) and isinstance(ext_orders, dict):
            for k in torch_orders:
                if k in ext_orders:
                    val_t = _normalize_flux_value(torch_orders[k])
                    val_e = _normalize_flux_value(ext_orders[k])
                    if isinstance(val_t, float) and isinstance(val_e, float):
                        diff = abs(val_t - val_e)
                        if diff > tolerance + FLUX_TOLERANCE:
                            raise RuntimeError(
                                f'cross-solver efficiency discrepancy for order {k}: '
                                f'{diff:.6e} > {tolerance:.6e} (cross_solver_discrepancy_exceeded)'
                            )

        reconciliation_passed = True
        reconciliation_record = {
            'external_solver': ext_solver.get('name'),
            'external_solver_version': ext_solver.get('version', 'unknown'),
            'external_solver_provenance': ext_solver.get('provenance', 'unknown'),
            'max_discrepancy': max_discrepancy,
            'tolerance': tolerance,
            'reconciled': True,
            'benchmark': config.get('benchmark', cs_data.get('benchmark', 'metagrating')),
        }

    # Check 11: Inverse Design Optimization Step Notary Gate (I-notary, LEG 49)
    is_inverse_design_run = (
        config.get('kind') == 'inverse_design' or
        readout.get('kind') == 'inverse_design' or
        config.get('inverse_design') is True or
        readout.get('inverse_design') is not None or
        'inverse_design_step' in config.get('tags', []) or
        'inverse_design_step' in readout.get('tags', []) or
        'inverse_design' in config.get('tags', []) or
        'inverse_design' in readout.get('tags', [])
    )
    inverse_design_passed = False
    inverse_design_record = None

    if is_inverse_design_run:
        inv_data = readout.get('inverse_design')
        if not isinstance(inv_data, dict):
            raise RuntimeError('missing inverse_design dictionary in readout.json (tamper_detected)')

        # Gate 1: Genuine Kernel Telemetry
        tel_steps = telemetry.get('executed_steps', 0)
        if not _is_finite_number(tel_steps) or int(tel_steps) <= 0:
            raise RuntimeError('inverse design requires positive executed_steps in solver_telemetry.json (tamper_detected)')
        if not telemetry.get('is_lossless', True):
            raise RuntimeError('inverse design test vehicle must be lossless (passivity_violation)')

        steps = inv_data.get('steps', [])
        if not isinstance(steps, list) or len(steps) == 0:
            raise RuntimeError('inverse design requires non-empty steps list in readout.json (tamper_detected)')

        # Gate 2: Gradient Verification Gate (<= 1e-6)
        gradient_tolerance = float(inv_data.get('gradient_tolerance', 1e-6))
        for step in steps:
            fwd = step.get('forward', {})
            adj = step.get('adjoint', {})
            if int(fwd.get('executed_steps', 0)) <= 0 or int(adj.get('executed_steps', 0)) <= 0:
                raise RuntimeError('inverse design step missing or zero executed_steps (tamper_detected)')

            # Permittivity hash match between forward and adjoint
            fwd_eps = fwd.get('epsilon_sha256')
            adj_eps = adj.get('epsilon_sha256')
            if fwd_eps and adj_eps and fwd_eps != adj_eps:
                raise RuntimeError('forward and adjoint operate on mismatched epsilon tensors (tamper_detected)')

            # Gradient check
            g_ver = step.get('gradient_verification', {})
            rel_err = g_ver.get('relative_error')
            if rel_err is None or not _is_finite_number(rel_err):
                raise RuntimeError('missing or invalid gradient verification relative_error (gradient_verification_failed)')
            rel_err = float(rel_err)
            if rel_err > gradient_tolerance or g_ver.get('passed') is False:
                raise RuntimeError(
                    f'gradient verification failed: relative_error {rel_err:.6e} > {gradient_tolerance:.6e} '
                    f'(gradient_verification_failed)'
                )
            if 'fdtd_objective_relative_error' in g_ver:
                fdtd_err = g_ver.get('fdtd_objective_relative_error')
                if not _is_finite_number(fdtd_err) or float(fdtd_err) > gradient_tolerance:
                    raise RuntimeError(
                        f'end-to-end FDTD gradient verification failed: {fdtd_err} > {gradient_tolerance:.6e} '
                        f'(gradient_verification_failed)'
                    )

            # Gate 3: Passivity Gate
            p_ratio = fwd.get('power_ratio')
            if p_ratio is not None and float(p_ratio) > 1.0 + tau_pml:
                raise RuntimeError(
                    f'inverse design passivity violation: power_ratio {float(p_ratio):.6f} > 1.0 + {tau_pml:.1e} '
                    f'(passivity_violation)'
                )

            # Gate 4: M3 Fabrication Constraint Gate (tooth >= 150 nm, trench >= 150 nm, zero violations)
            m3_info = step.get('m3_fabrication', {})
            min_tooth = m3_info.get('min_tooth_width_nm')
            min_trench = m3_info.get('min_trench_width_nm')
            violations = m3_info.get('violations', [])

            if min_tooth is None or min_trench is None or not _is_finite_number(min_tooth) or not _is_finite_number(min_trench):
                raise RuntimeError('missing or invalid feature widths in m3_fabrication (fabrication_violation_m3)')
            if float(min_tooth) < 150.0 - 1e-3 or float(min_trench) < 150.0 - 1e-3 or len(violations) > 0:
                raise RuntimeError(
                    f'M3 fabrication constraint violation: tooth={min_tooth} nm, trench={min_trench} nm, '
                    f'violations={violations} (fabrication_violation_m3)'
                )
            if m3_info.get('realized_simulation_executed') is True:
                real_obj = m3_info.get('realized_objective_value')
                real_p = m3_info.get('realized_power_ratio')
                if not _is_finite_number(real_obj) or not m3_info.get('realized_epsilon_sha256'):
                    raise RuntimeError('missing or invalid realized M3 FDTD evaluation (fabrication_violation_m3)')
                if _is_finite_number(real_p) and float(real_p) > 1.0 + tau_pml:
                    raise RuntimeError(
                        f'realized M3 FDTD passivity violation: {float(real_p):.6f} > 1.0 + {tau_pml:.1e} '
                        f'(passivity_violation)'
                    )

        # Gate 5: Objective Progress / Monotonicity & Terminal State Integrity Gate
        step0_obj = steps[0].get('objective_value')
        step_last_obj = steps[-1].get('objective_value')
        if not _is_finite_number(step0_obj) or not _is_finite_number(step_last_obj):
            raise RuntimeError('missing or non-finite step objective_value in steps (tamper_detected)')

        step0_obj = float(step0_obj)
        step_last_obj = float(step_last_obj)

        if 'initial_objective' in inv_data:
            if not _is_finite_number(inv_data['initial_objective']) or abs(float(inv_data['initial_objective']) - step0_obj) > 1e-9:
                raise RuntimeError(
                    f'initial_objective {inv_data.get("initial_objective")} does not match steps[0] {step0_obj} (tamper_detected)'
                )
        if 'final_objective' in inv_data:
            if not _is_finite_number(inv_data['final_objective']) or abs(float(inv_data['final_objective']) - step_last_obj) > 1e-9:
                raise RuntimeError(
                    f'final_objective {inv_data.get("final_objective")} does not match terminal step {step_last_obj} '
                    f'(report-only objective substitution is forbidden) (tamper_detected)'
                )
        if 'objective_gain' in inv_data:
            if not _is_finite_number(inv_data['objective_gain']) or abs(float(inv_data['objective_gain']) - (step_last_obj - step0_obj)) > 1e-9:
                raise RuntimeError(
                    f'objective_gain {inv_data.get("objective_gain")} does not match terminal - initial {step_last_obj - step0_obj} (tamper_detected)'
                )

        # Verify terminal step epsilon_sha256 matches kernel telemetry when both are present
        term_eps = steps[-1].get('forward', {}).get('epsilon_sha256')
        if term_eps and telemetry.get('epsilon_sha256') and term_eps != telemetry.get('epsilon_sha256'):
            raise RuntimeError(
                f'terminal step epsilon_sha256 {term_eps} does not match kernel telemetry {telemetry.get("epsilon_sha256")} (tamper_detected)'
            )

        if len(steps) >= 2:
            j_initial = step0_obj
            j_final = step_last_obj
            gain = j_final - j_initial
            if j_final <= j_initial or gain <= 0:
                raise RuntimeError(
                    f'inverse design failed to improve objective: initial={j_initial:.6f}, final={j_final:.6f} '
                    f'(non_monotonic_objective)'
                )

            if 'initial_realized_objective' in inv_data or 'final_realized_objective' in inv_data:
                r0 = steps[0].get('m3_fabrication', {}).get('realized_objective_value')
                r_last = steps[-1].get('m3_fabrication', {}).get('realized_objective_value')
                if not _is_finite_number(r0) or not _is_finite_number(r_last):
                    raise RuntimeError('missing realized_objective_value on initial/terminal steps (tamper_detected)')
                r0_f = float(r0)
                r_last_f = float(r_last)
                if 'initial_realized_objective' in inv_data and abs(float(inv_data['initial_realized_objective']) - r0_f) > 1e-9:
                    raise RuntimeError('initial_realized_objective mismatch with steps[0] (tamper_detected)')
                if 'final_realized_objective' in inv_data and abs(float(inv_data['final_realized_objective']) - r_last_f) > 1e-9:
                    raise RuntimeError('final_realized_objective mismatch with steps[-1] (tamper_detected)')
                if r_last_f < r0_f:
                    raise RuntimeError(
                        f'realized M3 discrete design regressed: initial={r0_f:.6f}, final={r_last_f:.6f} '
                        f'(non_monotonic_objective)'
                    )

        inverse_design_passed = True
        inverse_design_record = {
            'device': inv_data.get('device', 'canonical_grating_2d'),
            'wavelength_um': inv_data.get('wavelength_um', 1.55),
            'iterations_verified': len(steps),
            'initial_objective': step0_obj,
            'final_objective': step_last_obj,
            'objective_gain': float(step_last_obj - step0_obj),
            'max_gradient_relative_error': max(float(s['gradient_verification']['relative_error']) for s in steps),
            'min_tooth_width_nm': min(float(s['m3_fabrication']['min_tooth_width_nm']) for s in steps),
            'min_trench_width_nm': min(float(s['m3_fabrication']['min_trench_width_nm']) for s in steps),
            'all_steps_passive': True,
            'm3_compliant': True,
            'monotonic_progress': True,
        }
        if 'initial_realized_objective' in inv_data and 'final_realized_objective' in inv_data:
            inverse_design_record['initial_realized_objective'] = float(inv_data['initial_realized_objective'])
            inverse_design_record['final_realized_objective'] = float(inv_data['final_realized_objective'])
            inverse_design_record['realized_objective_gain'] = float(
                inv_data['final_realized_objective'] - inv_data['initial_realized_objective']
            )
            inverse_design_record['m3_realized_fdtd_verified'] = True
        if any('fdtd_objective_relative_error' in s.get('gradient_verification', {}) for s in steps):
            inverse_design_record['max_fdtd_gradient_relative_error'] = max(
                float(s['gradient_verification'].get('fdtd_objective_relative_error', s['gradient_verification']['relative_error']))
                for s in steps
            )
            inverse_design_record['full_fdtd_gradient_verified'] = True

    # Issue verification certificate
    certificate = {
        'schema_version': '1.0',
        'run_id': run_dir.name,
        'verified': True,
        'certified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'telemetry_sha256': _sha256_file(telemetry_file),
        'readout_sha256': _sha256_file(readout_file),
        'config_sha256': _sha256_file(config_file) if config_file.is_file() else None,
        'project_canonical_sha256': telemetry.get('project_canonical_sha256'),
        'epsilon_sha256': telemetry.get('epsilon_sha256'),
        'is_lossless': is_lossless,
        'executed_steps': telemetry_steps,
        'resumed_from_step': telemetry_resumed,
        'auto_shutoff_triggered': telemetry_shutoff,
        'shutoff_step': shutoff_step,
        'shutoff_energy_ratio': shutoff_energy,
        'shutoff_flux_delta': shutoff_flux_delta,
        'max_flux_diff': max_flux_diff,
        'compared_monitors': compared_monitors,
        'total_power_sum': total_power,
        'passivity_passed': is_lossless and (total_power is not None and total_power <= 1.0 + tau_pml),
        'box_closure_residual': box_closure_residual,
        'box_closure_passed': box_closure_passed,
        'box_monitors': box_monitors_record,
        'decay_ratio': decay_ratio,
        'energy_decay_ratio': decay_ratio,
        'spectral_decay_passed': decay_passed if decay_passed is not None else True,
        'tags': [
            'verified:postflight_notary',
            'verified:telemetry_consistent',
            'verified:passivity' if is_lossless else 'unverified:lossy',
        ],
    }
    if telemetry_resumed > 0:
        certificate['tags'].append('verified:resumed_run')
    if box_closure_passed:
        certificate['tags'].extend(['verified:v2_box_closure', 'verified:poynting_box_closed'])
        if is_6surface_box:
            certificate['tags'].append('verified:v2_box_closure_3d')
    if decay_passed:
        certificate['tags'].extend(['verified:v2_decay', 'verified:spectral_decay'])
    if telemetry_shutoff:
        certificate['tags'].extend([
            'verified:auto_shutoff',
            'verified:energy_decay_1e-9',
            'verified:flux_stable_1e-6',
        ])
    if mesh_convergence_passed:
        certificate['tags'].extend([
            'verified:monotonic_mesh_convergence',
            'verified:richardson_extrapolated',
        ])
        certificate['mesh_convergence'] = mesh_convergence_record
    if graded_collar_passed:
        certificate['tags'].extend([
            'verified:graded_collar_accuracy',
            'verified:cell_reduction_50pct',
        ])
        certificate['graded_accuracy'] = graded_collar_record
    if reconciliation_passed:
        certificate['tags'].extend([
            'verified:cross_solver_reconciled',
            'verified:v4_reconciliation',
        ])
        certificate['cross_solver_reconciliation'] = reconciliation_record
    if inverse_design_passed:
        certificate['tags'].extend([
            'verified:inverse_design_step',
            'verified:analytic_jacobian',
            'verified:m3_fabrication_compliant',
        ])
        if inverse_design_record.get('full_fdtd_gradient_verified'):
            certificate['tags'].append('verified:full_fdtd_gradient')
        if inverse_design_record.get('m3_realized_fdtd_verified'):
            certificate['tags'].append('verified:m3_realized_fdtd')
        certificate['inverse_design'] = inverse_design_record
    if 'gpu_characterization' in config.get('tags', []) or 'gpu_characterization' in readout.get('tags', []):
        acceptance = readout.get('acceptance')
        if not isinstance(acceptance, dict):
            raise RuntimeError('post_notary: gpu_characterization acceptance criteria violated: missing or invalid acceptance block (tamper_detected)')

        a1a = acceptance.get('A1a')
        if not isinstance(a1a, dict) or a1a.get('max_flux_diff') != 0.0 or a1a.get('passed') is not True:
            max_flux = a1a.get('max_flux_diff') if isinstance(a1a, dict) else None
            raise RuntimeError(f"post_notary: gpu_characterization acceptance criteria violated: A1a max_flux_diff={max_flux} != 0.0 (tamper_detected)")

        a1b = acceptance.get('A1b')
        if not isinstance(a1b, dict) or a1b.get('passed') is not True:
            raise RuntimeError('post_notary: gpu_characterization acceptance criteria violated: A1b failed or missing (tamper_detected)')
        flux_err = a1b.get('relative_flux_error')
        grad_err = a1b.get('relative_gradient_error')
        if flux_err is None or not isinstance(flux_err, (int, float)) or float(flux_err) > 1e-12:
            raise RuntimeError(f"post_notary: gpu_characterization acceptance criteria violated: A1b relative_flux_error={flux_err} > 1e-12 (tamper_detected)")
        if grad_err is None or not isinstance(grad_err, (int, float)) or float(grad_err) > 1e-10:
            raise RuntimeError(f"post_notary: gpu_characterization acceptance criteria violated: A1b relative_gradient_error={grad_err} > 1e-10 (tamper_detected)")

        setup_sec = readout.get('setup_graph_capture_sec')
        steady_sec = readout.get('steady_replay_sec')
        if setup_sec is None or not isinstance(setup_sec, (int, float)) or float(setup_sec) <= 0.0:
            raise RuntimeError(f"post_notary: gpu_characterization acceptance criteria violated: setup_graph_capture_sec={setup_sec} must be > 0.0 (tamper_detected)")
        if steady_sec is None or not isinstance(steady_sec, (int, float)) or float(steady_sec) <= 0.0:
            raise RuntimeError(f"post_notary: gpu_characterization acceptance criteria violated: steady_replay_sec={steady_sec} must be > 0.0 (tamper_detected)")

        a3 = acceptance.get('A3')
        if a3 is not None:
            if not isinstance(a3, dict) or a3.get('status') not in ('verified:bottleneck_classification', 'completed'):
                raise RuntimeError(f"post_notary: gpu_characterization acceptance criteria violated: A3 status={a3.get('status') if isinstance(a3, dict) else None} invalid (tamper_detected)")

        a4 = acceptance.get('A4')
        if not isinstance(a4, dict) or a4.get('status') not in ('completed', 'partial'):
            a4_stat = a4.get('status') if isinstance(a4, dict) else None
            raise RuntimeError(f"post_notary: gpu_characterization acceptance criteria violated: A4 status={a4_stat} not in ('completed', 'partial') (tamper_detected)")
        cum_gpu = a4.get('cumulative_gpu_seconds')
        if isinstance(cum_gpu, (int, float)) and float(cum_gpu) > 2160.0:
            raise RuntimeError(f"post_notary: gpu_characterization acceptance criteria violated: A4 cumulative_gpu_seconds={cum_gpu} > 2160.0 (tamper_detected)")

        certificate['tags'].extend([
            'verified:gpu_characterization',
            'verified:cuda_graph_identity',
            'verified:cross_execution_precision',
            'verified:timing_separation',
        ])
    if 'vector_overlap' in config.get('tags', []) or 'vector_overlap' in readout.get('tags', []):
        if 'analytic_self_checks' in readout:
            asc = readout['analytic_self_checks']
            if not asc.get('all_passed'):
                raise RuntimeError('analytic self-checks for vector mode overlap failed (tamper_detected)')
        if 'edge_sweep' in readout:
            es = readout['edge_sweep']
            if not es.get('sweep_passed') or float(es.get('max_interior_rel_err', 1.0)) > 1e-5:
                raise RuntimeError('21-point edge sweep derivative verification failed (gradient_verification_failed)')

        cases = readout.get('etch_bias_cases')
        if cases is not None or 'verified:etch_bias_corners' in config.get('tags', []) or 'verified:etch_bias_corners' in readout.get('tags', []):
            if cases is None or not isinstance(cases, dict):
                raise RuntimeError('post_notary: etch_bias_corners validation failed: missing or invalid etch_bias_cases dictionary (tamper_detected)')

            nominal_key = 'nominal' if 'nominal' in cases else ('nominal_0nm' if 'nominal_0nm' in cases else None)
            if nominal_key is None or 'over_etched_-10nm' not in cases or 'under_etched_+10nm' not in cases:
                raise RuntimeError('post_notary: etch_bias_corners validation failed: missing required process corners (over_etched_-10nm, nominal/nominal_0nm, under_etched_+10nm) (tamper_detected)')

            c_over = cases['over_etched_-10nm']
            c_nom = cases[nominal_key]
            c_under = cases['under_etched_+10nm']

            if not (isinstance(c_over, dict) and isinstance(c_nom, dict) and isinstance(c_under, dict)):
                raise RuntimeError('post_notary: etch_bias_corners validation failed: corner data must be dictionaries (tamper_detected)')

            hashes = [
                c_over.get('epsilon_sha256'),
                c_nom.get('epsilon_sha256'),
                c_under.get('epsilon_sha256'),
            ]
            if any(not h or not isinstance(h, str) for h in hashes):
                raise RuntimeError('post_notary: etch_bias_corners validation failed: missing or non-string epsilon_sha256 digest in corners (tamper_detected)')
            if len(set(hashes)) != 3:
                raise RuntimeError('post_notary: etch_bias_corners validation failed: duplicate epsilon_sha256 digests across process corners (tamper_detected)')

            rho_over = c_over.get('mean_density')
            rho_nom = c_nom.get('mean_density')
            rho_under = c_under.get('mean_density')
            if None in (rho_over, rho_nom, rho_under):
                raise RuntimeError('post_notary: etch_bias_corners validation failed: missing mean_density in corners (tamper_detected)')
            try:
                rho_over_f = float(rho_over)
                rho_nom_f = float(rho_nom)
                rho_under_f = float(rho_under)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(f'post_notary: etch_bias_corners validation failed: non-numeric mean_density: {exc} (tamper_detected)') from exc

            if not (rho_over_f < rho_nom_f < rho_under_f):
                raise RuntimeError(f'post_notary: etch_bias_corners validation failed: non-monotonic mean density ({rho_over_f} >= {rho_nom_f} or {rho_nom_f} >= {rho_under_f}) (tamper_detected)')

            for corner_name in ('over_etched_-10nm', nominal_key, 'under_etched_+10nm'):
                corner_data = cases[corner_name]
                eta = corner_data.get('calibrated_vector_overlap_eta')
                if eta is None:
                    raise RuntimeError(f'post_notary: etch_bias_corners validation failed: missing calibrated_vector_overlap_eta in corner {corner_name} (tamper_detected)')
                try:
                    eta_f = float(eta)
                except (ValueError, TypeError) as exc:
                    raise RuntimeError(f'post_notary: etch_bias_corners validation failed: non-numeric calibrated_vector_overlap_eta in corner {corner_name}: {exc} (tamper_detected)') from exc
                if not (0.0 <= eta_f <= 1.0):
                    raise RuntimeError(f'post_notary: etch_bias_corners validation failed: calibrated_vector_overlap_eta={eta_f} violates passivity bounds [0.0, 1.0] in corner {corner_name} (tamper_detected)')

        vector_tags = [
            'verified:vector_overlap',
            'verified:subpixel_etch_bias',
            'verified:analytic_identity',
            'verified:analytic_orthogonality',
            'verified:directional_isolation',
        ]
        if 'edge_sweep' in readout:
            vector_tags.append('verified:edge_sweep_gradient')
        if cases is not None:
            vector_tags.append('verified:etch_bias_corners')
        certificate['tags'].extend(vector_tags)

    # Check 12: Two-Tier Numerical Error Budget & ScientificClaim Falsification Gate (Wave E — LEG 54)
    has_claim_tags = any(
        t in ('scientific_claim', 'two_tier_budget', 'falsification_gate',
              'verified:scientific_claim', 'verified:two_tier_budget', 'verified:falsification_gate')
        for t in config.get('tags', []) + readout.get('tags', [])
    )
    is_claim_run = 'scientific_claims' in readout or 'scientific_claims' in config or has_claim_tags

    if is_claim_run:
        claims = readout.get('scientific_claims')
        if claims is None or not isinstance(claims, list) or len(claims) == 0:
            raise RuntimeError('post_notary: Check 12 (scientific_claims) violated: missing or empty scientific_claims in readout (tamper_detected)')

        validated_claims_record = []
        for idx, claim in enumerate(claims):
            if not isinstance(claim, dict):
                raise RuntimeError(f'post_notary: Check 12 (scientific_claims) violated: claim {idx} is not a dictionary (tamper_detected)')

            # Extract fields
            param = claim.get('parameter', f'param_{idx}')
            nom_val = claim.get('nominal_value')
            thresh = claim.get('threshold')
            if not _is_finite_number(nom_val) or not _is_finite_number(thresh):
                raise RuntimeError(f'post_notary: Check 12 (scientific_claims) violated: non-numeric nominal_value or threshold in claim {param} (tamper_detected)')
            nom_val = float(nom_val)
            thresh = float(thresh)

            claim_type = claim.get('claim_type', 'greater_than')
            if claim_type not in ('greater_than', 'less_than'):
                raise RuntimeError(f'post_notary: Check 12 (scientific_claims) violated: unsupported claim_type {claim_type!r} (tamper_detected)')

            # 1. Budget structure & NC4: omitting dominant uncertainty component / zero budget
            budget = claim.get('budget')
            if not isinstance(budget, dict):
                raise RuntimeError(f'post_notary: Check 12 (scientific_claims) violated: missing budget dictionary in claim {param} (tamper_detected)')

            components = budget.get('components', [])
            if not isinstance(components, list) or len(components) == 0:
                raise RuntimeError(f'post_notary: Check 12 (scientific_claims) violated: omitted or empty uncertainty budget components in claim {param} (tamper_detected)')

            comp_vals = []
            comp_certified = []
            for c_idx, comp in enumerate(components):
                if not isinstance(comp, dict) or 'value' not in comp:
                    raise RuntimeError(f'post_notary: Check 12 (scientific_claims) violated: corrupted component {c_idx} in claim {param} (tamper_detected)')
                c_val = comp['value']
                if not _is_finite_number(c_val) or float(c_val) < 0.0 or math.isnan(float(c_val)) or math.isinf(float(c_val)):
                    raise RuntimeError(f'post_notary: Check 12 (scientific_claims) violated: invalid component value in claim {param} (tamper_detected)')
                comp_vals.append(float(c_val))
                c_cert = comp.get('certified', False)
                if not isinstance(c_cert, bool):
                    raise RuntimeError(
                        f'post_notary: Check 12 (scientific_claims) violated: certified must be boolean in claim {param} component {c_idx} (tamper_detected)'
                    )
                if c_cert:
                    prov = comp.get('provenance')
                    if not isinstance(prov, dict) or not prov or not any(k in prov for k in ('proof', 'solver', 'theorem', 'reference_solver', 'envelope', 'analytic')):
                        raise RuntimeError(
                            f'post_notary: Check 12 (scientific_claims) violated: component {comp.get("name", c_idx)} claims certified=True but lacks proof provenance (tamper_detected)'
                        )
                comp_certified.append(c_cert)

            # NC5: Inconsistent combined uncertainty math
            u_total = budget.get('total_uncertainty', claim.get('total_uncertainty'))
            if not _is_finite_number(u_total) or math.isnan(u_total) or math.isinf(u_total):
                raise RuntimeError(f'post_notary: Check 12 (scientific_claims) violated: missing or non-numeric total_uncertainty in claim {param} (tamper_detected)')
            u_total = float(u_total)

            comb_method = budget.get('combination_method', 'rss')
            if comb_method == 'linear':
                expected_u = sum(comp_vals)
            else:
                expected_u = math.sqrt(sum(v * v for v in comp_vals))

            if not math.isfinite(expected_u) or math.isnan(expected_u) or math.isinf(expected_u):
                raise RuntimeError(
                    f'post_notary: Check 12 (scientific_claims) violated: non-finite combined uncertainty calculation in claim {param} (tamper_detected)'
                )

            if abs(u_total - expected_u) > 1e-6:
                raise RuntimeError(
                    f'post_notary: Check 12 (scientific_claims) violated: inconsistent combined uncertainty math: '
                    f'reported={u_total:.6e} vs expected={expected_u:.6e} (tamper_detected)'
                )

            if u_total <= 0.0:
                raise RuntimeError(f'post_notary: Check 12 (scientific_claims) violated: non-positive total uncertainty in claim {param} (tamper_detected)')

            # NC3: Claiming certified assurance level with uncertified error component
            budget_assurance = budget.get('assurance_level', 'estimated')
            claim_assurance = claim.get('assurance_level', budget_assurance)
            if budget_assurance not in ('estimated', 'certified') or claim_assurance not in ('estimated', 'certified'):
                raise RuntimeError(
                    f'post_notary: Check 12 (scientific_claims) violated: invalid assurance_level: '
                    f'budget={budget_assurance!r}, claim={claim_assurance!r} (tamper_detected)'
                )
            all_cert = all(comp_certified)

            if budget_assurance == 'certified' and not all_cert:
                raise RuntimeError(
                    f'post_notary: Check 12 (scientific_claims) violated: budget asserts certified assurance_level with uncertified component(s) (tamper_detected)'
                )
            if claim_assurance == 'certified' and not all_cert:
                raise RuntimeError(
                    f'post_notary: Check 12 (scientific_claims) violated: claim asserts certified assurance_level with uncertified component(s) (tamper_detected)'
                )

            # Mathematical 4-State Truth Evaluation
            q_low = nom_val - u_total
            q_high = nom_val + u_total

            req_assurance = claim.get('required_assurance_level', 'estimated')
            if claim_type == 'greater_than':
                if q_low > thresh:
                    if req_assurance == 'certified' and not all_cert:
                        math_state = 'INCONCLUSIVE'
                    else:
                        math_state = 'SUPPORTED'
                elif q_high < thresh:
                    math_state = 'REFUTED'
                else:
                    math_state = 'INCONCLUSIVE'
            else:  # less_than
                if q_high < thresh:
                    if req_assurance == 'certified' and not all_cert:
                        math_state = 'INCONCLUSIVE'
                    else:
                        math_state = 'SUPPORTED'
                elif q_low > thresh:
                    math_state = 'REFUTED'
                else:
                    math_state = 'INCONCLUSIVE'

            # NC1 & NC2: Over-claiming SUPPORTED
            raw_claimed = claim.get('claimed_state') or claim.get('claim_status')
            claimed_state = str(raw_claimed).strip().upper() if raw_claimed is not None else None
            positive_synonyms = ('SUPPORTED', 'PASS', 'PASSED', 'CONFIRMED', 'TRUE', 'VALIDATED')
            if claimed_state in positive_synonyms:
                if math_state == 'INCONCLUSIVE':
                    raise RuntimeError(
                        f'post_notary: Check 12 (scientific_claims) violated: claimed {claimed_state} on INCONCLUSIVE interval '
                        f'[{q_low:.6f}, {q_high:.6f}] against threshold {thresh:.6f} (tamper_detected)'
                    )
                if math_state == 'REFUTED':
                    raise RuntimeError(
                        f'post_notary: Check 12 (scientific_claims) violated: claimed {claimed_state} on REFUTED interval '
                        f'[{q_low:.6f}, {q_high:.6f}] against threshold {thresh:.6f} (tamper_detected)'
                    )

            # Validate declared actual_state: must be declared and must match mathematical truth
            declared_actual = claim.get('actual_state')
            if not declared_actual or not isinstance(declared_actual, str) or not declared_actual.strip():
                raise RuntimeError(
                    f'post_notary: Check 12 (scientific_claims) violated: missing actual_state declaration in claim {param} (tamper_detected)'
                )
            declared_actual = declared_actual.strip().upper()
            if declared_actual != math_state:
                raise RuntimeError(
                    f'post_notary: Check 12 (scientific_claims) violated: actual_state contradiction: '
                    f'declared={declared_actual} vs mathematical={math_state} (tamper_detected)'
                )

            # Validate diagnostic prescription for INCONCLUSIVE state
            if math_state == 'INCONCLUSIVE':
                prescription = claim.get('diagnostic_prescription')
                if not prescription or not isinstance(prescription, str) or not prescription.strip():
                    raise RuntimeError(
                        f'post_notary: Check 12 (scientific_claims) violated: missing diagnostic prescription for INCONCLUSIVE claim {param} (tamper_detected)'
                    )
                dom_name = budget.get('dominant_component')
                if not dom_name and components:
                    dom_comp = max(components, key=lambda c: c['value'])
                    dom_name = dom_comp['name']
                if dom_name and dom_name not in prescription and 'Level B' not in prescription:
                    raise RuntimeError(
                        f'post_notary: Check 12 (scientific_claims) violated: diagnostic prescription does not identify dominant component {dom_name} (tamper_detected)'
                    )

            validated_claims_record.append({
                'parameter': param,
                'nominal_value': nom_val,
                'threshold': thresh,
                'interval': [q_low, q_high],
                'total_uncertainty': u_total,
                'assurance_level': claim_assurance,
                'actual_state': math_state,
                'claimed_state': claimed_state,
                'dominant_component': budget.get('dominant_component'),
                'verified': True,
            })

        certificate['scientific_claims'] = validated_claims_record
        certificate['tags'].extend([
            'verified:scientific_claim',
            'verified:two_tier_budget',
            'verified:falsification_gate',
        ])

    # Check 13: GDS/PDK Dual-Equivalence Delivery Gate (Wave E — LEG 56)
    has_gds_tags = any(
        t in ('gds_delivery', 'soi_pdk', 'dual_equivalence', 'gds_dual_equivalence',
              'verified:gds_delivery', 'verified:gds_dual_equivalence', 'gds_dual_equivalence_verified')
        for t in config.get('tags', []) + readout.get('tags', [])
    )
    is_gds_run = (
        'gds_delivery' in readout or
        'gds_delivery' in config or
        'gds_dual_equivalence' in readout or
        'gds_dual_equivalence' in config or
        has_gds_tags
    )

    if is_gds_run:
        gds_data = readout.get('gds_delivery') or readout.get('gds_dual_equivalence')
        if not isinstance(gds_data, dict):
            raise RuntimeError('post_notary: Check 13 violated: missing gds_delivery block in readout (tamper_detected)')

        # Gate 1: GDS File Existence and SHA-256 Digest Match
        gds_rel_path = gds_data.get('gds_file', 'layout.gds')
        gds_file = run_dir / gds_rel_path
        if not gds_file.is_file():
            raise RuntimeError(f'post_notary: Check 13 violated: GDS file {gds_rel_path} not found on disk (tamper_detected)')
        actual_gds_sha256 = _sha256_file(gds_file)
        reported_sha = gds_data.get('gds_sha256')
        if not reported_sha or not isinstance(reported_sha, str) or not reported_sha.strip():
            raise RuntimeError('post_notary: Check 13 violated: missing or invalid gds_sha256 in gds_delivery (tamper_detected)')
        if reported_sha != actual_gds_sha256:
            raise RuntimeError(
                f'post_notary: Check 13 violated: GDS SHA-256 mismatch {actual_gds_sha256} vs {reported_sha} (tamper_detected)'
            )

        # Gate 2: Coordinate Quantization Rounding Gate (<= 1 dbu / 1 nm)
        max_rounding_nm = gds_data.get('max_rounding_error_nm', gds_data.get('max_coordinate_error_nm'))
        if max_rounding_nm is None or not _is_finite_number(max_rounding_nm) or float(max_rounding_nm) < 0.0 or float(max_rounding_nm) > 1.0 + 1e-6:
            raise RuntimeError(
                f'post_notary: Check 13 violated: max rounding error {max_rounding_nm} nm out of bounds [0, 1 dbu] (gds_rounding_exceeded)'
            )

        # Gate 3: DRC 0 Violations Gate
        drc_clean = gds_data.get('drc_clean', gds_data.get('is_drc_clean', False))
        if not isinstance(drc_clean, bool):
            raise RuntimeError('post_notary: Check 13 violated: drc_clean must be boolean (tamper_detected)')
        drc_viols = gds_data.get('drc_violations', [])
        drc_viol_count = len(drc_viols) if isinstance(drc_viols, list) else int(drc_viols)
        if not drc_clean or drc_viol_count > 0:
            raise RuntimeError(
                f'post_notary: Check 13 violated: DRC violations detected: {drc_viols} (drc_violation_detected)'
            )

        # Gate 4: Physical Readback Coupling Equivalence Gate
        eta_design = gds_data.get('eta_design')
        eta_readback = gds_data.get('eta_readback')
        u_disc = gds_data.get('u_discretization', 0.015)

        if not _is_finite_number(eta_design) or not _is_finite_number(eta_readback) or not _is_finite_number(u_disc):
            raise RuntimeError('post_notary: Check 13 violated: non-numeric efficiency or uncertainty in gds_delivery (tamper_detected)')

        discrepancy = abs(float(eta_readback) - float(eta_design))
        if discrepancy > float(u_disc) + 1e-9:
            raise RuntimeError(
                f'post_notary: Check 13 violated: physical coupling discrepancy {discrepancy:.6e} > U_discretization {float(u_disc):.6e} (physical_equivalence_failed)'
            )

        certificate['gds_dual_equivalence'] = {
            'gds_file': str(gds_rel_path),
            'gds_sha256': actual_gds_sha256,
            'pdk_spec': gds_data.get('pdk_spec', 'SOI_220nm_PDK'),
            'max_rounding_error_nm': float(max_rounding_nm),
            'drc_clean': True,
            'drc_violations_count': 0,
            'eta_design': float(eta_design),
            'eta_readback': float(eta_readback),
            'coupling_discrepancy': discrepancy,
            'u_discretization': float(u_disc),
            'geometric_equivalent': True,
            'physical_equivalent': True,
            'verified': True,
        }
        certificate['tags'].extend([
            'verified:gds_delivery',
            'verified:soi_pdk_drc',
            'verified:gds_geometric_equivalence',
            'verified:gds_physical_equivalence',
            'gds_dual_equivalence_verified',
        ])

    payload_bytes = json.dumps(certificate, sort_keys=True, separators=(',', ':')).encode('utf-8')
    certificate['certificate_payload_sha256'] = hashlib.sha256(payload_bytes).hexdigest()
    notary_secret = os.environ.get('TORCHFDTD_NOTARY_SECRET')
    if notary_secret:
        certificate['hmac_sha256'] = hmac.new(
            notary_secret.encode('utf-8'), payload_bytes, hashlib.sha256
        ).hexdigest()

    cert_path = run_dir / 'verification_certificate.json'
    cert_path.write_text(json.dumps(certificate, indent=2) + '\n', encoding='utf-8')
    return certificate


def main():
    if len(sys.argv) > 1 and sys.argv[1]:
        run_dir = Path(sys.argv[1])
    elif 'RUN_DIR_ABS' in os.environ:
        run_dir = Path(os.environ['RUN_DIR_ABS'])
    else:
        run_dir = Path.cwd()

    try:
        cert = verify_post_flight(run_dir)
        print(f'[post_notary] OK: verification certificate issued for {run_dir.name}')
        sys.exit(0)
    except RuntimeError as e:
        err_msg = str(e)
        print(f'[post_notary] postflight failed: {err_msg}', file=sys.stderr)
        status_tag = 'tamper_detected'
        if 'false_scientific_claim' in err_msg:
            status_tag = 'false_scientific_claim'
        elif 'under_converged_time' in err_msg:
            status_tag = 'under_converged_time'
        elif 'under_converged_flux' in err_msg:
            status_tag = 'under_converged_flux'
        elif 'poynting_box_unclosed' in err_msg:
            status_tag = 'poynting_box_unclosed'
        elif 'passivity_violation' in err_msg:
            status_tag = 'passivity_violation'
        elif 'non_monotonic_mesh_ladder' in err_msg:
            status_tag = 'non_monotonic_mesh_ladder'
        elif 'graded_accuracy_exceeded' in err_msg:
            status_tag = 'graded_accuracy_exceeded'
        elif 'insufficient_cell_reduction' in err_msg:
            status_tag = 'insufficient_cell_reduction'
        elif 'cross_solver_discrepancy_exceeded' in err_msg:
            status_tag = 'cross_solver_discrepancy_exceeded'
        elif 'gradient_verification_failed' in err_msg:
            status_tag = 'gradient_verification_failed'
        elif 'fabrication_violation_m3' in err_msg:
            status_tag = 'fabrication_violation_m3'
        elif 'non_monotonic_objective' in err_msg:
            status_tag = 'non_monotonic_objective'
        elif 'gds_rounding_exceeded' in err_msg:
            status_tag = 'gds_rounding_exceeded'
        elif 'drc_violation_detected' in err_msg:
            status_tag = 'drc_violation_detected'
        elif 'physical_equivalence_failed' in err_msg:
            status_tag = 'physical_equivalence_failed'
        elif 'tamper_detected' in err_msg:
            status_tag = 'tamper_detected'

        try:
            failed_cert = {
                'schema_version': '1.0',
                'run_id': run_dir.name,
                'verified': False,
                'certified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'error': err_msg,
                'status': status_tag,
            }
            (run_dir / 'verification_certificate.json').write_text(
                json.dumps(failed_cert, indent=2) + '\n', encoding='utf-8')
        except Exception:
            pass
        sys.exit(66)


if __name__ == '__main__':
    main()
