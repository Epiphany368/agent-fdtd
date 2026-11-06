"""Leg 30: classification is an exit-status gate, with finite archive exceptions.

All generated artefacts live in temporary roots. No archived evidence is changed,
and the driver test mocks its workers rather than executing a simulation.
"""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import coverage_audit as ca
from verification import STAGES, verify_readout


def pipeline():
    return dict(policy='leg30-synthetic-pipeline', stages=list(STAGES),
                stage_source={stage: 'synthetic CPU fixture' for stage in STAGES},
                cases=[dict(id='sample', measured=dict(T=0.5))])


def diagnostics():
    return dict(policy='leg30-synthetic-diagnostics', readout_kind='diagnostics',
                readout_reason='Worker comparisons, not a staged pipeline.', cases={})


class ClassificationGate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_run(self, name, data):
        run = self.root / 'runs' / name
        run.mkdir(parents=True, exist_ok=True)
        (run / 'readout.json').write_text(json.dumps(data) + '\n')
        return run

    def cli(self):
        return subprocess.run([sys.executable, str(ROOT / 'src/coverage_audit.py'), str(self.root)],
                              text=True, capture_output=True, check=False)

    def test_a2_unclassifiable_is_a_real_failed_exit_naming_directory_and_field(self):
        run = self.write_run('unclassifiable-negative-control', dict(cases=[]))
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(str(run), result.stderr)
        self.assertIn('policy', result.stderr)
        self.assertNotIn('Runs read:', result.stdout)
        with self.assertRaisesRegex(ca.ReadoutClassificationError, 'policy'):
            ca.audit([run])

    def test_a2_diagnostics_are_listed_excluded_and_exit_zero(self):
        data = diagnostics()
        run = self.write_run('diagnostics-positive-control', data)
        found = ca.discover_runs(self.root)
        self.assertEqual(found, [])
        self.assertEqual([row['run'] for row in found.diagnostics], [run])
        text = ca.render(found)
        self.assertIn('Runs read: 0;', text)
        self.assertIn('Diagnostics excluded from pipeline inventory: 1', text)
        self.assertIn(str(run), text)
        self.assertIn(data['readout_reason'], text)
        self.assertIn(data['policy'], text)
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, text)
        # Explicit diagnostics do not relax the pipeline verifier at all.
        with self.assertRaisesRegex(ValueError, 'ordered F1-F6'):
            verify_readout(data)

    def test_pipeline_without_policy_is_not_hidden_even_with_valid_stages(self):
        data = pipeline()
        del data['policy']
        run = self.write_run('pipeline-forgot-policy', data)
        with self.assertRaisesRegex(ca.ReadoutClassificationError, 'policy') as caught:
            ca.discover_runs(self.root)
        self.assertIn(str(run), str(caught.exception))

    def test_diagnostics_require_both_policy_and_nonempty_reason(self):
        for field, invalid_values in (('policy', [None, '', ' ', 24, [], {}]),
                                      ('readout_reason', [None, '', '\t', 24, [], {}])):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    data = diagnostics()
                    data[field] = value
                    run = self.write_run('incomplete-diagnostic', data)
                    with self.assertRaisesRegex(ca.ReadoutClassificationError, field):
                        ca.classify_readout(run)
            data = diagnostics()
            del data[field]
            with self.assertRaisesRegex(ca.ReadoutClassificationError, field):
                ca.classify_readout(self.write_run('incomplete-diagnostic', data))

    def test_pipeline_requires_exact_stage_order_and_known_kind(self):
        for stages in (None, [], list(reversed(STAGES)), list(STAGES) + ['F7-verify'], 'F1-F6'):
            with self.subTest(stages=stages):
                data = pipeline()
                data['stages'] = stages
                run = self.write_run('invalid-stages', data)
                with self.assertRaisesRegex(ca.ReadoutClassificationError, 'stages'):
                    ca.classify_readout(run)
        for kind in (None, 'diagnostic', 'unknown', [], {}):
            with self.subTest(kind=kind):
                data = pipeline()
                data['readout_kind'] = kind
                with self.assertRaisesRegex(ca.ReadoutClassificationError, 'readout_kind'):
                    ca.classify_readout(self.write_run('invalid-kind', data))

    def test_pipeline_inventory_and_headlines_are_unchanged(self):
        data = pipeline()
        run = self.write_run('pipeline', data)
        expected = ca.audit_run(run)
        self.write_run('diagnostic', diagnostics())
        inventory = ca.discover_runs(self.root)
        self.assertEqual(inventory, [run])
        self.assertEqual(ca.audit_run(inventory[0]), expected)
        self.assertEqual(expected[1], verify_readout(data))
        self.assertEqual(ca.audit(inventory)[1], len(expected[1]))
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Runs read: 1;', result.stdout)
        self.assertIn('Diagnostics excluded from pipeline inventory: 1', result.stdout)
        # Existing drivers filter the inventory with a plain list comprehension.
        self.assertEqual(ca.render(list(inventory)), ca.render(inventory))

    def test_plain_empty_driver_list_still_reports_diagnostics(self):
        run = self.write_run('diagnostic-only', diagnostics())
        own = self.root / 'runs' / 'own-audit'
        with patch.dict(os.environ, {'RUN_DIR_ABS': str(own)}):
            text = ca.render([])
        self.assertIn('Runs read: 0;', text)
        self.assertIn('Diagnostics excluded from pipeline inventory: 1', text)
        self.assertIn(str(run), text)

    def test_audit_policy_exclusion_cannot_hide_a_malformed_readout(self):
        data = pipeline()
        data['policy'] = ca.AUDIT_POLICIES[0]
        run = self.write_run('prior-audit', data)
        self.assertEqual(ca.discover_runs(self.root, ca.AUDIT_POLICIES), [])
        del data['stages']
        self.write_run('prior-audit', data)
        with self.assertRaisesRegex(ca.ReadoutClassificationError, 'stages') as caught:
            ca.discover_runs(self.root, ca.AUDIT_POLICIES)
        self.assertIn(str(run), str(caught.exception))

    def test_pipeline_stage_source_contract_is_still_enforced(self):
        data = pipeline()
        del data['stage_source']
        run = self.write_run('pipeline-forgot-source', data)
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(str(run), result.stderr)
        self.assertIn('implementation source', result.stderr)

    def test_unreadable_and_nonobject_readouts_fail_instead_of_disappearing(self):
        run = self.write_run('broken-readout', {})
        for raw in ('{', '[]', 'null', '42', '"string"'):
            with self.subTest(raw=raw):
                (run / 'readout.json').write_text(raw)
                result = self.cli()
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(str(run), result.stderr)
                self.assertIn('readout.json', result.stderr)

    def test_a_diagnostic_changed_after_discovery_cannot_disappear(self):
        run = self.write_run('diagnostic-changed-after-discovery', diagnostics())
        inventory = ca.discover_runs(self.root)
        self.assertEqual(inventory, [])
        (run / 'readout.json').write_text('{}')
        for operation in (ca.audit, ca.render):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(ca.ReadoutClassificationError, 'policy'):
                    operation(inventory)

    def test_legacy_exception_is_finite_and_pins_all_three_files(self):
        self.assertEqual(len(ca.LEGACY_DIAGNOSTICS), 6)
        for hashes in ca.LEGACY_DIAGNOSTICS.values():
            self.assertEqual(len(hashes), 3)
            for digest in hashes:
                self.assertRegex(digest, r'^[0-9a-f]{64}$')
        run_id = next(iter(ca.LEGACY_DIAGNOSTICS))
        run = self.write_run(run_id, dict(leg=24, cases={}))
        (run / 'config.input.json').write_text(json.dumps(dict(policy='leg24-graded-modal-v1')))
        (run / 'manifest.json').write_text(json.dumps(dict(run_id=run_id, script='src/leg24_graded_modal.py')))
        names = ('readout.json', 'config.input.json', 'manifest.json')
        originals = {name: (run / name).read_bytes() for name in names}
        hashes = tuple(hashlib.sha256(originals[name]).hexdigest() for name in names)
        # Pin a synthetic triplet in a temporary registry. The production hashes
        # are independently checked against real, unedited archives in A4.
        with patch.dict(ca.LEGACY_DIAGNOSTICS, {run_id: hashes}, clear=True):
            self.assertEqual(ca.classify_readout(run)[:2], ('diagnostics', 'leg24-graded-modal-v1'))
            self.assertIn(hashes[0], ca.render(ca.discover_runs(self.root)))
            for filename in names:
                with self.subTest(tamper=filename):
                    (run / filename).write_bytes(originals[filename] + b' ')
                    with self.assertRaisesRegex(ca.ReadoutClassificationError, filename):
                        ca.discover_runs(self.root)
                    (run / filename).write_bytes(originals[filename])
            for filename in names[1:]:
                with self.subTest(missing=filename):
                    (run / filename).unlink()
                    with self.assertRaisesRegex(ca.ReadoutClassificationError, filename):
                        ca.discover_runs(self.root)
                    (run / filename).write_bytes(originals[filename])
            renamed = run.with_name('seventh-unregistered-copy')
            run.rename(renamed)
            with self.assertRaisesRegex(ca.ReadoutClassificationError, 'policy') as caught:
                ca.discover_runs(self.root)
            self.assertIn(str(renamed), str(caught.exception))


class DriverDeclaration(unittest.TestCase):
    def test_leg24_declares_diagnostics_in_incremental_and_final_readouts(self):
        import leg24_graded_modal as driver
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            effective = run / 'effective.json'
            effective.write_text('{}')
            cfg = dict(policy='leg24-graded-modal-v1', mode='uniform-baseline',
                       cases=[dict(id='synthetic')], expected=dict(cases={
                           'synthetic': dict(T=.5, R=.1, epsilon_sha256='synthetic')}))
            row = dict(measured=cfg['expected']['cases']['synthetic'], field_sha256={'near': 'same'},
                       mode_sha256='same', launch_identity='same', mesh={}, simulation_executed=False)
            saved = []

            def write_json(path, data):
                Path(path).write_text(json.dumps(data))
                if Path(path).name == 'readout.json':
                    saved.append(copy.deepcopy(data))

            def fake_worker(args, **kwargs):
                Path(args[-1]).write_text(json.dumps(row))
                return SimpleNamespace(returncode=0)

            fake_parity = SimpleNamespace(load_run=lambda: (cfg, run), write_json=write_json)
            env = dict(PARITY_SNAPSHOT=str(run / 'snapshot'), PARITY_CONFIG=str(run / 'config'),
                       SOLVER_FROZEN=str(run / 'frozen'), EFFECTIVE_JSON=str(effective))
            with patch.dict(sys.modules, {'parity': fake_parity}), patch.dict(os.environ, env), \
                    patch.object(driver.subprocess, 'run', side_effect=fake_worker), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(driver.main(), 0)
            self.assertEqual(len(saved), 3)
            for report in saved:
                self.assertEqual(report['policy'], cfg['policy'])
                self.assertEqual(report['readout_kind'], 'diagnostics')
                self.assertTrue(report['readout_reason'].strip())
            self.assertEqual(ca.classify_readout(run)[0], 'diagnostics')


if __name__ == '__main__':
    unittest.main()
