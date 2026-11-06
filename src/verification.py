"""F7: conservative, CPU-only readout decisions under plan.md rule V-0.

Headlines are all numeric leaves outside recorded/relative_error (comparison
evidence), plus measured SHA-256 digests. Booleans are status, not numbers.
Metadata is included so an untested timestep or timing cannot escape a label.
There are no registered V1/V2/V4 checks here: this module cannot grant verified.
Archived config comparisons establish reproduction only, never physical validity.
"""
import math
import re
from collections.abc import Mapping


STAGES = ('F1-structure', 'F2-mesh', 'F3-source', 'F4-monitors', 'F5-run', 'F6-readout')
LABELS = ('verified', 'consistent', 'unchecked')
_MISSING = object()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value):
    if not _number(value):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _leaves(value, path=()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _leaves(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaves(child, path + (index,))
    elif _number(value) or (path and str(path[-1]).endswith('_sha256')):
        yield path, value


def _lookup(value, path):
    try:
        for part in path:
            value = value[part]
        return value
    except (KeyError, IndexError, TypeError):
        return _MISSING


def _path(parts):
    return ''.join(f'[{part}]' if isinstance(part, int) else
                   ('.' if index else '') + str(part)
                   for index, part in enumerate(parts))


def _comparison(value, path, case, readout, config):
    """Recompute from the archived expectation; ignore the driver's pass flags."""
    if not isinstance(config, Mapping):
        return 'unchecked', 'No archived config: no registered comparison tolerance or target.'
    if not config.get('policy') or config['policy'] != readout.get('policy'):
        return 'unchecked', 'Archived config policy does not match this readout.'
    expected = config.get('expected', {})
    if not isinstance(expected, Mapping):
        return 'unchecked', 'No valid archived expected-value declaration.'
    source = expected.get('source')
    if not isinstance(source, str) or not source.strip():
        return 'unchecked', 'No declared historical source for the comparison.'
    tolerance = expected.get('tolerance_relative')
    if not _finite(tolerance) or tolerance < 0:
        return 'unchecked', 'No finite, nonnegative archived relative tolerance.'
    reference = _lookup(expected, ('cases', case.get('id')) + path)
    if reference is _MISSING:
        return 'unchecked', 'No historical target for this headline in the archived config.'
    stored = _lookup(case.get('recorded', {}), path)
    if stored is not _MISSING and (type(stored) is not type(reference) or stored != reference):
        return 'unchecked', 'Readout recorded target contradicts the archived config.'
    if str(path[-1]).endswith('_sha256'):
        if not all(isinstance(item, str) and re.fullmatch(r'[0-9a-f]{64}', item)
                   for item in (value, reference)):
            return 'unchecked', 'Invalid SHA-256 digest; no comparison passed.'
        if value != reference:
            return 'unchecked', 'SHA-256 digest differs from the historical target.'
        return 'consistent', f'SHA-256 exactly reproduces {source}; identity only, not a physics check.'
    if not _finite(value) or not _finite(reference):
        return 'unchecked', 'Measured value or historical target is not a finite number.'
    error = abs(value - reference) / abs(reference) if reference else abs(value)
    convention = 'absolute error (zero target)' if reference == 0 else 'relative error'
    if not math.isfinite(error) or error > tolerance:
        return 'unchecked', (f'Reproduction failed: {convention} {error:.17g} exceeds '
                             f'tolerance {tolerance!r}; source: {source}.')
    return 'consistent', (f'Recomputed {convention} {error:.17g} <= tolerance '
                          f'{tolerance!r} against {source}; solver reproduction only, '
                          'no registered V1/V2/V4 physics check.')


def verify_readout(readout, config=None):
    """Return a labelled row and reason for every headline, without mutating input.

    ``config`` is optional. Without its archived comparison declaration measured
    values are unchecked, even if readout claims zero error or a passing check.
    Unknown measured numbers and numeric metadata are discovered recursively.
    """
    if not isinstance(readout, Mapping) or readout.get('stages') != list(STAGES):
        raise ValueError('Readout must declare the ordered F1-F6 stage list.')
    sources = readout.get('stage_source', {})
    if not isinstance(sources, Mapping) or any(
            not isinstance(sources.get(stage), str) or not sources[stage].strip()
            for stage in STAGES):
        raise ValueError('Every declared stage must name its implementation source.')
    cases = readout.get('cases')
    if not isinstance(cases, list) or any(not isinstance(case, Mapping) for case in cases):
        raise ValueError('Readout cases must be a list of objects.')
    ids = [case.get('id') for case in cases]
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Readout case IDs must be nonempty and unique.')
    if any(not isinstance(case.get('measured'), Mapping) for case in cases):
        raise ValueError('Every case must contain a measured object.')

    headlines = []
    for parts, value in _leaves({key: value for key, value in readout.items()
                                if key != 'cases'}):
        headlines.append(dict(path=_path(parts), case_id=None, value=value,
                              label='unchecked', reason='No registered check for this run metadata.'))
    for index, case in enumerate(cases):
        metadata = {key: value for key, value in case.items()
                    if key not in ('measured', 'recorded', 'relative_error')}
        for parts, value in _leaves(metadata):
            headlines.append(dict(path=_path(('cases', index) + parts), case_id=case['id'],
                                  value=value, label='unchecked',
                                  reason='No registered check for this case metadata.'))
        for parts, value in _leaves(case.get('measured', {})):
            label, reason = _comparison(value, parts, case, readout, config)
            headlines.append(dict(path=_path(('cases', index, 'measured') + parts),
                                  case_id=case['id'], value=value, label=label, reason=reason))
    return headlines
