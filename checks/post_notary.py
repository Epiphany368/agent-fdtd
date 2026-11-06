"""Post-flight notary gate enforcing kernel telemetry consistency, passivity, and cross-solver reconciliation (LEG 43 - 48).

Zero-Trust Rule: Agents possess propose rights, never dispose rights. A simulation's
credibility, completion, and merge eligibility are verified by unforgeable kernel telemetry
(solver_telemetry.json) compared against driver readout (readout.json) and config.

Exit codes:
- 0: Verification succeeded, verification_certificate.json issued.
- 66: Postflight verification failed (tamper detected, passivity violation, missing telemetry, or cross-solver discrepancy exceeded).
"""
import datetime
import hashlib
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
    return bool(mons & box_right_keys) and bool(mons & box_left_keys) and \
           bool(mons & box_top_keys) and bool(mons & box_bottom_keys)


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

    if is_box_equipped and is_lossless:
        right_candidates = ['box-right', 'box_right', 'transmission', 'trans', 'far', 'T', 'right']
        left_candidates = ['box-left', 'box_left', 'reflection', 'refl', 'near', 'R', 'left']
        top_candidates = ['box-top', 'box_top', 'top', 'up', 'cladding']
        bottom_candidates = ['box-bottom', 'box_bottom', 'bottom', 'down', 'substrate']

        flux_right, name_r = _extract_box_surface_flux(telemetry_monitors, readout_fluxes, right_candidates)
        flux_left, name_l = _extract_box_surface_flux(telemetry_monitors, readout_fluxes, left_candidates)
        flux_top, name_t = _extract_box_surface_flux(telemetry_monitors, readout_fluxes, top_candidates)
        flux_bottom, name_b = _extract_box_surface_flux(telemetry_monitors, readout_fluxes, bottom_candidates)

        missing_surfaces = []
        if flux_right is None: missing_surfaces.append('right/transmission')
        if flux_left is None: missing_surfaces.append('left/reflection')
        if flux_top is None: missing_surfaces.append('top/cladding')
        if flux_bottom is None: missing_surfaces.append('bottom/substrate')

        if missing_surfaces:
            raise RuntimeError(
                f'poynting box unclosed: missing {", ".join(missing_surfaces)} monitor(s) '
                f'(poynting_box_unclosed)'
            )

        p_inc = config.get('incident_flux', config.get('incident_power', 1.0))
        p_inc_norm = _normalize_flux_value(p_inc)

        if isinstance(flux_right, float) and isinstance(flux_left, float) and \
           isinstance(flux_top, float) and isinstance(flux_bottom, float):
            phi_out = abs(flux_right) + abs(flux_left) + abs(flux_top) + abs(flux_bottom)
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
        elif all(isinstance(f, list) for f in (flux_right, flux_left, flux_top, flux_bottom)):
            lens = [len(f) for f in (flux_right, flux_left, flux_top, flux_bottom)]
            if len(set(lens)) != 1:
                raise RuntimeError(
                    f'poynting box monitor array length mismatch: {lens} (poynting_box_unclosed)'
                )
            n_pts = lens[0]
            inc_list = p_inc_norm if isinstance(p_inc_norm, list) and len(p_inc_norm) == n_pts else [1.0] * n_pts
            r_boxes = []
            sums = []
            for i in range(n_pts):
                phi_out = abs(flux_right[i]) + abs(flux_left[i]) + abs(flux_top[i]) + abs(flux_bottom[i])
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
        else:
            raise RuntimeError(
                'poynting box monitor flux type mismatch (poynting_box_unclosed)'
            )

        box_closure_passed = (box_closure_residual <= tau_box)
        if not box_closure_passed:
            raise RuntimeError(
                f'poynting box closure failure: residual R_box {box_closure_residual:.6e} > {tau_box:.1e} '
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
            if rel_err > gradient_tolerance:
                raise RuntimeError(
                    f'gradient verification failed: relative_error {rel_err:.6e} > {gradient_tolerance:.6e} '
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

        # Gate 5: Objective Progress / Monotonicity Gate
        if len(steps) >= 2:
            j_initial = float(inv_data.get('initial_objective', steps[0]['objective_value']))
            j_final = float(inv_data.get('final_objective', steps[-1]['objective_value']))
            gain = float(inv_data.get('objective_gain', j_final - j_initial))
            if j_final <= j_initial or gain <= 0:
                raise RuntimeError(
                    f'inverse design failed to improve objective: initial={j_initial:.6f}, final={j_final:.6f} '
                    f'(non_monotonic_objective)'
                )

        inverse_design_passed = True
        inverse_design_record = {
            'device': inv_data.get('device', 'canonical_grating_2d'),
            'wavelength_um': inv_data.get('wavelength_um', 1.55),
            'iterations_verified': len(steps),
            'initial_objective': float(inv_data.get('initial_objective', steps[0]['objective_value'])),
            'final_objective': float(inv_data.get('final_objective', steps[-1]['objective_value'])),
            'objective_gain': float(inv_data.get('objective_gain', steps[-1]['objective_value'] - steps[0]['objective_value'])),
            'max_gradient_relative_error': max(float(s['gradient_verification']['relative_error']) for s in steps),
            'min_tooth_width_nm': min(float(s['m3_fabrication']['min_tooth_width_nm']) for s in steps),
            'min_trench_width_nm': min(float(s['m3_fabrication']['min_trench_width_nm']) for s in steps),
            'all_steps_passive': True,
            'm3_compliant': True,
            'monotonic_progress': True,
        }

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
        certificate['inverse_design'] = inverse_design_record

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
        if 'under_converged_time' in err_msg:
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
