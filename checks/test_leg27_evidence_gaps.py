"""Leg 27 contracts: "no archived evidence" must not print the same thing as "everything passed".

The defect. In a workspace without archived run data the archive-dependent checks skip, and the
suite still reports success at the level a human reads: ``-m unittest checks.test_coverage_audit
-v`` prints ``OK (skipped=3), exit 0`` -- byte-identical, in summary and exit status, to a run in
which the evidence existed and every check passed. Ten such ``skipTest('no archived ...')`` calls
exist (``checks/test_coverage_audit.py:38,73,84``; ``checks/test_leg15_closure.py:83,107,118,122,
131,145,153``). ``skipped=N`` is already printed, and it changed nobody's decision, so this leg
encodes the distinction ``checked and passed`` vs ``never checked`` in the **exit status**.

The meta-check. Every leg whose checks need archived artefacts must have an entry in
``checks/evidence_requirements.json`` (the derived list, with the ``path:line`` each entry came
from). For each owning leg the check decides one of three states:

- **present** -- the artefact predicates hold in this tree; the leg's checks ran and passed.
- **declared-absent** -- the artefacts are absent and the leg's entry carries
  ``evidence_absent_by_design: true``. Reported, printed, and not a failure.
- **needed-absent** -- the artefacts are absent and there is no declaration. The check **fails**,
  naming the leg and the missing path.

A missing or unreadable declaration file is "no declarations": every absent artefact is then
needed-absent, which is exactly the state of ``origin/main`` today (pre-leg). The structural join
between this module's built-in inventory and the declaration file is validated, so the two cannot
silently drift apart. An unclaimed new ``skipTest('no archived ...')`` added by a future leg is
caught by the source scan and fails the suite until it is declared.

Negative control. ``checks/fixtures/leg27-negative-control-probe/`` holds a probe root whose
artefacts satisfy the same predicates, proving the check can tell "present" from "absent" and is
not a pass-through. Nothing under ``checks/fixtures/`` is a run record; the real scan path is
``ROOT/runs`` only. No fixture is used to satisfy the real-tree verdict, which rests on declared
absence, not invented evidence.

Run ``LEG27_DECLARATIONS=<nonexistent path>`` over this module to reproduce the declarations-absent
state (accepted by hand 2026-10-19 from ``checks/pre_environment.py``'s override pattern and
``checks/test_leg18_standing_controls.py``'s ``LEG18_ARCHIVE_RUNS``). The override points at where
the declarations are read from; it cannot change what the predicates require.

LEG 37 closes the one remaining reachability hole: every leg in the shipped file declares
``evidence_absent_by_design: true``, so the needed-absent branch above was demonstrated only with
the file absent altogether. ``Leg37FatalDirection`` flips exactly one shipped flag to ``false``
in memory and requires the meta-check to fail on the account, naming the leg and the missing
artefact. The shipped file is never modified on disk.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / 'checks'

DECLARATIONS_PATH = CHECKS / 'evidence_requirements.json'
DECLARATIONS_ENV = 'LEG27_DECLARATIONS'
DECLARATIONS_SCHEMA = 'leg27-evidence-requirements-v1'

# The audit's own discovery, mirrored from src/coverage_audit.py:111-127 and wired at
# checks/test_coverage_audit.py:19: RUNS excludes the two audit-ledger policies by policy.
EXCLUDED_AUDIT_POLICIES = ('leg11-coverage-audit-v1', 'leg13-readout-ledger-v1')

EVIDENCE_PREFIX = 'leg27-evidence:'
SELF_FILE = 'checks/test_leg27_evidence_gaps.py'

EMPTY_PROBE_ROOT = CHECKS / 'fixtures' / 'leg27-empty-workspace-probe' / 'probe-root'
POSITIVE_PROBE_ROOT = CHECKS / 'fixtures' / 'leg27-negative-control-probe' / 'probe-root'


# ---------------------------------------------------------------------------
# The built-in inventory: derived (2026-10-19) from the ten skipTest('no archived ...')
# sites and from what those tests actually open. The declaration file mirrors this and adds
# the per-leg absence flag; validate_declaration() keeps the two from drifting.
# ---------------------------------------------------------------------------

INVENTORY = {
    'leg4': {
        'owning_config': 'configs/leg4-convergence-ladder.json',
        'artifacts': [
            {
                'id': 'leg4-ladder-interface-ppw40-run',
                'probe': 'readout-policy',
                'paths': ['runs/*/readout.json'],
                'policy': 'leg4-convergence-v1',
                'case_id': 'ladder-interface-ppw40',
                'display': "a runs/*/readout.json with policy 'leg4-convergence-v1' carrying case id "
                           "'ladder-interface-ppw40'",
                'derived_from': [
                    'checks/test_leg15_closure.py:41-53 (find_case)',
                    'checks/test_leg15_closure.py:120 (the call)',
                    'checks/test_leg15_closure.py:121-122 (the guard)',
                ],
            }
        ],
        'skips': [
            {
                'file': 'checks/test_leg15_closure.py',
                'reason': 'no archived leg-4 run on this machine',
                'sites': [122],
            }
        ],
    },
    'leg11': {
        'owning_config': 'configs/leg11-coverage-audit.json',
        'artifacts': [
            {
                'id': 'leg11-at-least-one-discoverable-run',
                'probe': 'readout-policy',
                'paths': ['runs/*/readout.json'],
                'policy_not_in': EXCLUDED_AUDIT_POLICIES,
                'display': "at least one runs/*/readout.json whose policy is not one of the two "
                           "audit ledgers ('%s', '%s') -- RUNS in checks/test_coverage_audit.py:19"
                           % EXCLUDED_AUDIT_POLICIES,
                'derived_from': [
                    'checks/test_coverage_audit.py:19 (RUNS discovery)',
                    'checks/test_coverage_audit.py:37-38, 72-73, 83-84 (the guards)',
                    'src/coverage_audit.py:111-127 (discover_runs)',
                ],
            }
        ],
        'skips': [
            {
                'file': 'checks/test_coverage_audit.py',
                'reason': 'no archived runs',
                'sites': [38, 73, 84],
            }
        ],
    },
    'leg12': {
        'owning_config': 'configs/leg12-closed-slab.json',
        'artifacts': [
            {
                'id': 'leg12-slab-run-with-monitor-arrays',
                'probe': 'readout-policy',
                'paths': ['runs/*/readout.json'],
                'policy': 'leg12-closed-box-v1',
                'case_id': 'slab',
                'sibling': 'slab-sample-monitors.npz',
                'display': "a runs/*/readout.json with policy 'leg12-closed-box-v1', a case id 'slab', "
                           "and slab-sample-monitors.npz beside it",
                'derived_from': [
                    'checks/test_leg15_closure.py:22-38 (find_leg12_slab_run)',
                    'checks/test_leg15_closure.py:38 (the guard)',
                ],
            }
        ],
        'skips': [
            {
                'file': 'checks/test_leg15_closure.py',
                'reason': 'no archived leg-12 run on this machine',
                'sites': [83, 107, 118, 131, 153],
            }
        ],
    },
    'leg15': {
        'owning_config': 'configs/leg15-closure.json',
        'artifacts': [
            {
                'id': 'leg15-closure-latest-run-cases',
                'probe': 'readout-policy',
                'paths': ['runs/*/readout.json'],
                'policy': 'leg15-closure-v1',
                'newest': True,
                'display': "the newest runs/*/readout.json with policy 'leg15-closure-v1' has usable "
                           "cases (id and measured on them), as leg15_results() reads it",
                'derived_from': [
                    'checks/test_leg15_closure.py:56-66 (leg15_results)',
                    'checks/test_leg15_closure.py:142-145 (the setUp guard)',
                ],
            }
        ],
        'skips': [
            {
                'file': 'checks/test_leg15_closure.py',
                'reason': 'no archived leg-15 run on this machine',
                'sites': [145],
            }
        ],
    },
}

# The evidence_requirements flags, restored from the inventory so that an absent artefact whose
# flag is missing from the JSON stays needed-absent rather than stopping the run with a KeyError.
FLAG_BY_LEG = {
    'leg4': True,
    'leg11': True,
    'leg12': True,
    'leg15': True,
}

ARTIFACT_COMPARED_KEYS = ('id', 'probe', 'paths', 'policy', 'policy_not_in', 'case_id',
                          'sibling', 'newest', 'derived_from')
GUARDED_QUOTE = "['\"]"


def _probe_readout_policy(spec, probe_root):
    """Mirror the predicate one leg's tests guard on, without importing anything from src.

    Returns whether the artefact is present and one line of where/how it was (not) found.
    """
    root = Path(probe_root)
    candidates = {}
    unparsable = 0
    for pattern in spec['paths']:
        for readout in sorted(root.glob(pattern)):
            try:
                data = json.loads(readout.read_text())
            except (OSError, ValueError):
                unparsable += 1
                continue
            if not isinstance(data, dict):
                unparsable += 1
                continue
            policy = data.get('policy')
            if 'policy_not_in' in spec and policy in spec['policy_not_in']:
                continue
            if 'policy' in spec and policy != spec['policy']:
                continue
            if 'case_id' in spec:
                cases = data.get('cases') or []
                if not any(isinstance(case, dict) and case.get('id') == spec['case_id']
                           for case in cases):
                    continue
            if 'sibling' in spec and not (readout.parent / spec['sibling']).exists():
                continue
            candidates[readout] = data
    if not candidates:
        return False, ('%d unparsable skipped' % unparsable if unparsable else 'nothing matched')
    if spec.get('newest'):
        newest, newest_data = sorted(candidates.items())[-1]
        cases = newest_data.get('cases') or []
        usable = [case for case in cases
                  if isinstance(case, dict) and 'id' in case and 'measured' in case]
        if not usable:
            return False, 'the newest run has no case carrying both id and measured'
        return True, '%s (newest of %d, %d usable case(s))' % (
            newest.relative_to(root), len(candidates), len(usable))
    first = sorted(candidates)[0]
    return True, '%s%s' % (first.relative_to(root), '...' if len(candidates) > 1 else '')


_PROBES = {'readout-policy': _probe_readout_policy}


def evaluate(probe_root, declarations=None):
    """Per-leg present / declared-absent / needed-absent over the built-in inventory."""
    rows = []
    for leg in sorted(INVENTORY):
        artifact_states = []
        for artifact in INVENTORY[leg]['artifacts']:
            present, detail = _PROBES[artifact['probe']](artifact, probe_root)
            artifact_states.append({'id': artifact['id'], 'present': present, 'detail': detail})
        if all(state['present'] for state in artifact_states):
            status = 'present'
        elif isinstance(declarations, dict) \
                and declarations.get('legs', {}).get(leg, {}).get('evidence_absent_by_design') is True:
            status = 'declared-absent'
        else:
            status = 'needed-absent'
        rows.append({'leg': leg, 'status': status,
                     'artifacts': artifact_states, 'skips': INVENTORY[leg]['skips']})
    return rows


def account_lines(evidence_rows, declarations_problem=None):
    """The printed account: one line per leg, so a reader never has to infer the state."""
    lines = []
    if declarations_problem is not None:
        lines.append('%s declarations unreadable: %s' % (EVIDENCE_PREFIX, declarations_problem))
    for row in evidence_rows:
        artifacts = row['artifacts']
        where = '; '.join('%s: %s' % (a['id'], 'present at ' + a['detail'] if a['present']
                                     else a['detail']) for a in artifacts)
        sites = ', '.join('%s:%d' % (skip['file'], site)
                          for skip in row['skips'] for site in skip['sites'])
        lines.append('%s %s %s -- %s; skip guards: %s'
                     % (EVIDENCE_PREFIX, row['leg'], row['status'], where, sites))
    return lines


def is_fatal(evidence_rows, unclaimed):
    """A fatal account is what the standalone run turns into exit 1; nothing else fails it."""
    return any(row['status'] == 'needed-absent' for row in evidence_rows) or bool(unclaimed)


_SKIP_CALL = re.compile(r"""(?:\bself\.skipTest|\bcls\.skipTest|\bskipTest\b|\bSkipTest\b"""
                        r"""|\bskipIf\b|\bskipUnless\b)\s*\(""")
_QUOTED = re.compile(r"""(['"])((?:\\.|(?!\1).)*?)\1""")


def scan_archive_dependent_skips():
    """Every ``no archived ...`` skip in checks/test_*.py, or the scanner's own report is a lie.

    This module is excluded: it names the known reasons as data, never in a skip call, and the
    exclusion is visible here rather than buried. Commented-out calls do not count.
    """
    hits = []
    for test_file in sorted(CHECKS.glob('test_*.py')):
        if 'checks/' + test_file.name == SELF_FILE:
            continue
        for lineno, line in enumerate(test_file.read_text().splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            for call in _SKIP_CALL.finditer(line):
                quoted = _QUOTED.search(line, call.end())
                if quoted and quoted.group(2).strip().startswith('no archived'):
                    hits.append({'file': 'checks/' + test_file.name, 'line': lineno,
                                 'reason': quoted.group(2)})
                    break
    return hits


def claim_keys(row):
    return {(skip['file'], skip['reason']) for skip in row['skips']}


def declared_claims(declarations):
    return {(skip['file'], skip['reason'])
            for declaration in declarations.get('legs', {}).values()
            for skip in declaration.get('skips', [])}


def load_declarations(path=None):
    """Read the declaration file; missing/unreadable is 'no declarations', not a hard error."""
    declared_path = Path(path or os.environ.get(DECLARATIONS_ENV, DECLARATIONS_PATH))
    try:
        data = json.loads(declared_path.read_text())
    except (OSError, ValueError) as exc:
        return None, '%s (%s)' % (declared_path, exc)
    return data, None


def _canonical_strings(value):
    """JSON lists and python tuples must compare equal: strings normalise to tuples."""
    if isinstance(value, list):
        return tuple(value)
    return value


def _validated_side(declaration):
    values = {}
    artifacts = []
    for artifact in declaration.get('artifacts', []):
        artifacts.append({key: _canonical_strings(artifact[key])
                          for key in ARTIFACT_COMPARED_KEYS if key in artifact})
    skips = [(skip['file'], skip['reason'], tuple(sorted(skip['sites'])))
             for skip in declaration.get('skips', [])]
    values['artifacts'] = artifacts
    values['skips'] = sorted(skips)
    values['owning_config'] = declaration.get('owning_config')
    return values


def validate_declaration(data):
    """The declaration file must mirror the built-in inventory; drift is a hard failure."""
    problems = []
    if not isinstance(data, dict) or data.get('schema') != DECLARATIONS_SCHEMA:
        problems.append("schema must be exactly %r" % DECLARATIONS_SCHEMA)
    legs = data.get('legs') if isinstance(data, dict) else None
    if not isinstance(legs, dict):
        problems.append("'legs' must be an object")
        return problems
    inventory_legs = set(INVENTORY)
    declared_legs = set(legs)
    if inventory_legs - declared_legs:
        problems.append('legs missing from the declaration file: %s'
                        % ', '.join(sorted(inventory_legs - declared_legs)))
    if declared_legs - inventory_legs:
        problems.append('legs not in the inventory (extend checks/test_leg27_evidence_gaps.py): %s'
                        % ', '.join(sorted(declared_legs - inventory_legs)))
    for leg in sorted(inventory_legs & declared_legs):
        declared = _validated_side(legs[leg])
        expected = _validated_side({'artifacts': INVENTORY[leg]['artifacts'],
                                    'skips': INVENTORY[leg]['skips'],
                                    'owning_config': INVENTORY[leg]['owning_config']})
        if declared != expected:
            problems.append('%s: declaration does not match the inventory (%s != %s)'
                            % (leg, declared, expected))
        flag = legs[leg].get('evidence_absent_by_design', None)
        if flag not in (True, False):
            problems.append('%s: evidence_absent_by_design must be true or false, got %r'
                            % (leg, flag))
    return problems


class DeclarationError(AssertionError):
    pass


class Leg27EvidenceGaps(unittest.TestCase):
    def _load_or_null(self):
        declarations, problem = load_declarations()
        if declarations is not None:
            problems = validate_declaration(declarations)
            if problems:
                raise DeclarationError('%s: %s' % (DECLARATIONS_PATH, '; '.join(problems)))
        return declarations, problem

    def test_declaration_file_mirrors_the_inventory_and_carries_a_bool_flag(self):
        """A malformed declaration must not slide through as "no problem found"."""
        declarations, problem = self._load_or_null()
        self.assertIsNone(problem,
                          '%s must be readable when this leg is merged (found: %s)'
                          % (DECLARATIONS_PATH, problem))
        self.assertIsNotNone(declarations,
                             '%s must exist once this leg is merged: without it the ledger of '
                             "which evidence each leg's checks need is gone and every absent "
                             'artefact stays needed-absent' % DECLARATIONS_PATH)

    def test_a_workspace_without_archive_or_declarations_is_needed_absent(self):
        """A1's mechanism, demonstrated on a controlled empty root.

        An empty probe root plus no declarations is exactly the origin/main-today state in a
        checkout without run data (runs/* is untracked machine-local data), so every inventoried
        leg is needed-absent and the standalone run exits 1 naming leg and missing path.
        """
        evidence_rows = evaluate(EMPTY_PROBE_ROOT, declarations=None)
        for row in evidence_rows:
            with self.subTest(leg=row['leg']):
                self.assertEqual(row['status'], 'needed-absent')
        unclaimed = self._unclaimed_hits(evidence_rows)
        lines = account_lines(evidence_rows, 'none: LEG27_DECLARATIONS points at an absent file')
        text = '\n'.join(lines)
        for row in evidence_rows:
            self.assertIn(row['leg'], text)
        self.assertTrue(is_fatal(evidence_rows, unclaimed))

    def test_probe_root_with_fixture_evidence_reports_every_leg_present(self):
        """Negative control: fed the probe-root fixtures, the check says present, not absent."""
        declarations, problem = self._load_or_null()
        self.assertIsNone(problem,
                          'the declaration file must be readable for the negative control: %s'
                          % problem)
        evidence_rows = evaluate(POSITIVE_PROBE_ROOT, declarations)
        for row in evidence_rows:
            with self.subTest(leg=row['leg']):
                self.assertEqual(row['status'], 'present', '\n'.join(account_lines(evidence_rows)))
        self.assertFalse(is_fatal(evidence_rows, ()))

    def test_this_workspace_accounts_for_every_needed_artefact(self):
        """A2's real observation: any absent artefact is declared-absent, and never fatal."""
        declarations, problem = self._load_or_null()
        evidence_rows = evaluate(ROOT, declarations)
        unclaimed = self._unclaimed_hits(evidence_rows)
        text = '\n'.join(account_lines(evidence_rows, problem))
        print('\n%s' % text)
        for row in evidence_rows:
            with self.subTest(leg=row['leg']):
                if row['status'] == 'needed-absent':
                    self.fail(text)
                self.assertIn(row['status'], ('present', 'declared-absent'))
        self.assertFalse(is_fatal(evidence_rows, unclaimed), text)

    def _unclaimed_hits(self, evidence_rows=None):
        evidence_rows = evidence_rows if evidence_rows is not None else evaluate(
            ROOT, load_declarations()[0])
        claims = set()
        for row in evidence_rows:
            claims |= claim_keys(row)
        unclaimed = [hit for hit in scan_archive_dependent_skips()
                     if (hit['file'], hit['reason']) not in claims]
        return unclaimed

    def test_every_archive_dependent_skip_site_is_claimed(self):
        """A new ``skipTest('no archived ...')`` must be declared before it can pass silently."""
        evidence_rows = evaluate(
            ROOT, load_declarations()[0])
        unclaimed = self._unclaimed_hits(evidence_rows)
        accounts = ['%s:%d (%r)' % (hit['file'], hit['line'], hit['reason']) for hit in unclaimed]
        self.assertFalse(unclaimed,
                         'archive-dependent skips with no declaration entry '
                         '(add the owning leg to %s and the inventory in %s): %s'
                         % (DECLARATIONS_PATH, SELF_FILE, '; '.join(accounts)))

    def test_declared_sites_still_anchor_their_skip_reasons(self):
        """Every cited site must still carry its reason; a moved reason is re-anchored, printed.

        A reason that has vanished from its file entirely means the inventory's knowledge is
        stale: the suite must fail until someone re-derives that leg's evidence requirements.
        """
        stale = []
        drifts = []
        for leg in sorted(INVENTORY):
            for skip in INVENTORY[leg]['skips']:
                path = ROOT / skip['file']
                source = path.read_text().splitlines()
                found_at = []
                for site in skip['sites']:
                    line = source[site - 1] if site <= len(source) else ''
                    if skip['reason'] in line:
                        found_at.append(site)
                if len(found_at) == len(skip['sites']):
                    continue
                elsewhere = [lineno for lineno, line in enumerate(source, 1)
                             if skip['reason'] in line and lineno not in found_at]
                if elsewhere:
                    drifts.append('%s: %r expected at %s, actually at %s'
                                  % (skip['file'], skip['reason'], skip['sites'], elsewhere))
                else:
                    stale.append("%s: leg %s's skip reason %r is gone from the file; "
                                 're-derive the evidence inventory in %s'
                                 % (skip['file'], leg, skip['reason'], SELF_FILE))
        if drifts:
            print('%s sites drifted, re-anchored and still declared: %s'
                  % (EVIDENCE_PREFIX, '; '.join(drifts)))
        self.assertFalse(stale, '; '.join(stale))


class Leg37FatalDirection(unittest.TestCase):
    """LEG 37: the fatal direction must fire on the shipped declaration file itself.

    Every leg in ``checks/evidence_requirements.json`` ships
    ``evidence_absent_by_design: true``, so with the shipped data the needed-absent branch
    of ``evaluate()`` was reachable only through ``declarations=None`` -- the
    file-does-not-exist demonstration. These tests flip exactly one shipped flag to
    ``false`` in memory (never on disk) and require the meta-check to fail on the account,
    naming the leg and the missing artefact. Each flipped copy must also pass
    ``validate_declaration()``, so the failure is the fatal direction and not a
    malformed-file dismissal.
    """

    def _shipped_declarations(self):
        declarations, problem = load_declarations()
        self.assertIsNone(problem, '%s must be readable: %s' % (DECLARATIONS_PATH, problem))
        self.assertIsNotNone(declarations, '%s must exist' % DECLARATIONS_PATH)
        return declarations

    def _flipped_copy(self, shipped, leg):
        flipped = json.loads(json.dumps(shipped))
        flipped['legs'][leg]['evidence_absent_by_design'] = False
        return flipped

    def test_one_shipped_flag_flipped_to_false_is_needed_absent_and_the_rest_stay_declared_absent(self):
        shipped = self._shipped_declarations()
        for leg in sorted(INVENTORY):
            with self.subTest(leg=leg):
                self.assertIs(shipped['legs'][leg].get('evidence_absent_by_design'), True,
                              "the construction flips one shipped flag; %s's flag is no longer "
                              'true, so re-derive this test against the shipped file' % leg)
                flipped = self._flipped_copy(shipped, leg)
                self.assertEqual(validate_declaration(flipped), [],
                                 'the flipped copy must stay a valid declaration file: the '
                                 'needed-absent verdict below is the fatal direction, not a '
                                 'malformed-file dismissal')
                evidence_rows = evaluate(EMPTY_PROBE_ROOT, flipped)
                for row in evidence_rows:
                    with self.subTest(row_leg=row['leg']):
                        expected = 'needed-absent' if row['leg'] == leg else 'declared-absent'
                        self.assertEqual(row['status'], expected)
                text = '\n'.join(account_lines(evidence_rows))
                self.assertIn('%s %s needed-absent' % (EVIDENCE_PREFIX, leg), text)
                for artifact in INVENTORY[leg]['artifacts']:
                    self.assertIn(artifact['id'], text)
                self.assertTrue(is_fatal(evidence_rows, ()), text)

    def test_the_meta_check_exits_1_on_a_bare_root_with_one_flipped_flag(self):
        """End to end on a constructed root, so the verdict cannot depend on local archives.

        The subprocess runs the workspace-accounting test inside a temporary root that
        carries the meta-check module verbatim, the shipped declaration file in its
        default location, and an empty ``runs/`` -- twice: with the shipped file
        unchanged (exit 0, every leg declared-absent), then with exactly one flag
        flipped to ``false`` (exit 1, naming the leg and the missing artefact). Every
        input the subprocess reads is either tracked content or built here, so the same
        answer comes out on a machine with archived runs and on a machine without.
        ``LEG27_DECLARATIONS`` is stripped from the inherited environment, not passed
        through: the constructed root is authoritative.
        """
        shipped = self._shipped_declarations()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'probe-root'
            checks_dir = root / 'checks'
            checks_dir.mkdir(parents=True)
            (checks_dir / 'test_leg27_evidence_gaps.py').write_text(
                (CHECKS / 'test_leg27_evidence_gaps.py').read_text())
            (root / 'runs').mkdir()
            env = {key: value for key, value in os.environ.items()
                   if key != DECLARATIONS_ENV}
            results = {}
            for label, declarations in (('unflipped', shipped),
                                        ('flipped', self._flipped_copy(shipped, 'leg4'))):
                (checks_dir / 'evidence_requirements.json').write_text(
                    json.dumps(declarations))
                results[label] = subprocess.run(
                    [sys.executable, '-B', str(checks_dir / 'test_leg27_evidence_gaps.py'),
                     'Leg27EvidenceGaps.test_this_workspace_accounts_for_every_needed_artefact'],
                    env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(results['unflipped'].returncode, 0,
                         results['unflipped'].stdout + results['unflipped'].stderr)
        output = results['flipped'].stdout + results['flipped'].stderr
        self.assertEqual(results['flipped'].returncode, 1, output)
        self.assertIn('%s leg4 needed-absent' % EVIDENCE_PREFIX, output)
        self.assertIn('leg4-ladder-interface-ppw40-run', output)


if __name__ == '__main__':
    unittest.main()
