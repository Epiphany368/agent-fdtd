"""V3: the convergence ladder and the coarse-mesh coincidence guard.

Why this module exists:

1. A tolerance must be earned, not borrowed. Leg 3 declared `0.01` absolute as `INHERITED` from
   the reference project's canary. A ladder measured against a closed-form law is how this project
   replaces that with its own number.
2. A single grid can agree by coincidence. A sibling project recorded `35.5 percent` on a coarse
   grid and `41.3 percent` on a fine grid for the same quantity: a `5.7` point gap. Under rule
   `V-0`, a coarse-only agreement must never become `verified`.

Rules encoded here, all pre-declared in `plan.md` (leg 4, criteria C1-C6):

* every rung's error is measured against the closed form, so a rung is never judged only against
  another rung of ours;
* the order is fitted over the rungs, with every consecutive ratio and the fit residual reported;
* a non-monotone ladder is a **stop**, not a footnote: no slope is fitted;
* a case that is inside tolerance at the coarse rung but not at the finest rung is flagged
  **coincidence**, and no point from it may carry `verified`.
"""
import math

from analytic_cases import evaluate_analytic_cases


def _positive(values):
    return all(isinstance(value, (int, float)) and value > 0 and math.isfinite(value)
               for value in values)


def monotone_decreasing(errors):
    """True when each rung's error is strictly smaller than the previous rung's."""
    return all(later < earlier for earlier, later in zip(errors, errors[1:]))


def consecutive_orders(errors, refinement=2.0):
    """Implied order between consecutive rungs, assuming h is refined by `refinement` each rung."""
    return [math.log(earlier / later) / math.log(refinement)
            for earlier, later in zip(errors, errors[1:]) if later > 0]


def fit_order(errors, refinement=2.0):
    """Least-squares fit of log(error) = p log(h) + c, with h = refinement**-rung.

    Returns the order, the standard-error-like residual, and the individual implied orders.
    A single rung yields no fit.
    """
    if len(errors) < 2 or not _positive(errors):
        return dict(order=None, residual=None, implied=None,
                    reason='Fewer than two usable rungs, or a non-positive error: no fit.')
    xs = [-rung * math.log(refinement) for rung in range(len(errors))]
    ys = [math.log(error) for error in errors]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    residual = math.sqrt(sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys)) / n)
    return dict(order=slope, residual=residual,
                implied=consecutive_orders(errors, refinement),
                reason='Least-squares fit over the rungs vs 1/ppw.')


def coincidence(errors, tolerance, coarse_rungs=1):
    """The guard: inside tolerance on the coarse side, outside it at the finest rung."""
    if not errors or tolerance is None:
        return False
    coarse_inside = any(error <= tolerance for error in errors[:coarse_rungs])
    return bool(coarse_inside and errors[-1] > tolerance)


def evaluate_ladders(readout, config):
    """Recompute every declared ladder. No stored per-rung error or pass flag is trusted."""
    rows = []
    declared = config.get('ladders') if isinstance(config, dict) else None
    if not isinstance(declared, list) or not isinstance(readout, dict):
        return rows
    per_case = {row['case_id']: row for row in evaluate_analytic_cases(readout, config)}
    for entry in declared:
        if not isinstance(entry, dict):
            continue
        case_ids = entry.get('case_ids') or []
        rung_rows = [per_case.get(case_id) for case_id in case_ids]
        if any(row is None or row.get('error') is None for row in rung_rows):
            rows.append(dict(id=entry.get('id'), status='not-executed', case_ids=case_ids,
                             errors=None, order=None, residual=None, monotone=None,
                             coincidence=False, reason='A declared rung has no measured error.'))
            continue
        errors = [row['error'] for row in rung_rows]
        tolerance = entry.get('tolerance')
        refinement = entry.get('refinement', 2.0)
        decreasing = monotone_decreasing(errors)
        flag = coincidence(errors, tolerance, entry.get('coarse_rungs', 1))
        fit = fit_order(errors, refinement) if decreasing else dict(
            order=None, residual=None, implied=None,
            reason='Non-monotone ladder: criterion C3 stops the fit.')
        if not decreasing:
            status = 'stopped-non-monotone'
        elif flag:
            status = 'coincidence'
        elif errors[-1] <= tolerance:
            status = 'converged'
        else:
            status = 'not-converged'
        rows.append(dict(
            id=entry.get('id'), status=status, case_ids=list(case_ids), ppw=list(entry.get('ppw', [])),
            errors=errors, refinement=refinement, tolerance=tolerance,
            monotone=decreasing, coincidence=flag,
            implied_orders=fit['implied'], order=fit['order'], residual=fit['residual'],
            fit_reason=fit['reason'],
            worst_rung=case_ids[errors.index(max(errors))],
            reason=(entry.get('reason') or '')))
    return rows


def apply_coincidence_guard(headlines, rows):
    """Criterion C4: a case flagged `coincidence` may not carry `verified` anywhere.

    The guard is applied after the labels exist, so it can only remove a `verified` label, never
    create one. Returns the same list, with the offending rows rewritten.
    """
    flagged = {case_id for row in rows if row.get('coincidence') for case_id in (row.get('case_ids') or [])}
    if not flagged:
        return headlines
    for headline in headlines:
        if headline.get('case_id') in flagged and headline.get('label') == 'verified':
            headline['label'] = 'unchecked'
            headline['invariants'] = []
            headline['reason'] = (
                'Coarse-mesh coincidence: this case is inside tolerance at a coarse rung and '
                'outside it at the finest rung, so rule V-0 forbids `verified`. The earlier reason '
                'was: ' + str(headline.get('reason')))
    return headlines


def earned_floor(rows, registered_ppw, inherited_tolerance=None):
    """Criterion C6: the measured error at the registered rung, and its margin.

    An extrapolation to the continuum is only offered when the fit residual is small relative to
    the fitted order. Otherwise the function says so instead of guessing.
    """
    report = []
    for row in rows:
        if row.get('status') in (None, 'not-executed') or row.get('errors') is None:
            continue
        ppw = row.get('ppw') or []
        if registered_ppw not in ppw:
            continue
        index = ppw.index(registered_ppw)
        measured = row['errors'][index]
        entry = dict(ladder=row['id'], registered_ppw=registered_ppw, measured_error=measured,
                     tolerance=row.get('tolerance'), order=row.get('order'),
                     residual=row.get('residual'), status=row.get('status'))
        if isinstance(inherited_tolerance, (int, float)) and measured > 0:
            entry['margin_vs_inherited'] = inherited_tolerance / measured
            entry['inherited_tolerance'] = inherited_tolerance
        entry['extrapolation'] = ('supported' if (row.get('order') and row.get('residual') is not None
                                                  and abs(row['order']) > 0
                                                  and row['residual'] <= 0.1 * abs(row['order']))
                                  else 'not-supported')
        report.append(entry)
    return report
