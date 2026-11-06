"""Reserved-peak negative controls and the real leg-19 OOM stop in isolation."""
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
import leg23_relation as gate
import leg19_admission as legacy


class ReservedRelation(unittest.TestCase):
    def row(self, prediction=120, reserved=100, allocated=80, admitted=True):
        return dict(id='synthetic', prediction_bytes=prediction,
                    gpu_peak_reserved_bytes=reserved, gpu_peak_allocated_bytes=allocated,
                    admitted=admitted)

    def test_negative_control_reserved_underprediction_fails_printed(self):
        # Allocated passes while reserved fails: this catches the wrong metric.
        row = gate.relation(self.row(prediction=99))
        self.assertFalse(row['relation_holds'])
        self.assertEqual(row['reserved_margin_bytes'], -1)
        self.assertEqual(gate.gate_exit([row]), 57)
        print('NEGATIVE CONTROL: allocated=80 <= prediction=99 < reserved=100; '
              'reserved margin=-1; gate FAIL; exit=57', flush=True)

    def test_equality_passes_and_reports_zero_margin(self):
        row = gate.relation(self.row(prediction=100))
        self.assertTrue(row['relation_holds'])
        self.assertEqual(row['reserved_margin_bytes'], 0)
        self.assertEqual(gate.gate_exit([row]), 0)

    def test_every_row_is_checked_including_earlier_or_later_rows(self):
        for index in range(len(gate.JOBS)):
            with self.subTest(index=index):
                rows = [self.row() for _ in gate.JOBS]
                rows[index] = self.row(prediction=99)
                self.assertEqual(gate.gate_exit(rows), 57)

    def test_refused_row_is_not_silently_skipped(self):
        with self.assertRaisesRegex(ValueError, 'refused rows'):
            gate.gate_exit([self.row(), self.row(prediction=99, admitted=False)])

    def test_missing_or_invalid_measurements_do_not_pass(self):
        for prediction in (True, -1, 1.5, float('nan'), None):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                gate.relation(self.row(prediction=prediction))
        with self.assertRaises(ValueError):
            gate.gate_exit([])
        with self.assertRaises(KeyError):
            gate.gate_exit([dict(admitted=True, prediction_bytes=120)])
        with self.assertRaises(ValueError):
            gate.relation(self.row(reserved=70))

    def test_refusal_uses_measured_peak_and_explicit_base_requirement(self):
        raw = dict(case_id='synthetic', refused=True, refusal='required_bytes=99; limit_bytes=50;')
        telemetry = dict(gpu_peak_reserved_bytes=100, gpu_peak_allocated_bytes=80,
            reservation_calls=[dict(capacity_bypass_report=dict(memory_reservation_bytes=99))])
        row = gate.measured_relation(raw, telemetry, 'occupied', 2)
        self.assertEqual(row['reserved_margin_bytes'], -1)
        self.assertFalse(row['admitted'])
        self.assertIn('excludes modal', row['prediction_scope'])

    def test_refusal_incremental_requirement_is_not_process_peak_prediction(self):
        # W3 changes the required_bytes diagnostic, retaining the prediction's
        # process-peak meaning. The relation must still read that prediction.
        raw = dict(case_id='synthetic', refused=True, refusal='required_bytes=10; limit_bytes=5;')
        telemetry = dict(gpu_peak_reserved_bytes=100, gpu_peak_allocated_bytes=80,
            reservation_calls=[dict(capacity_bypass_report=dict(memory_reservation_bytes=120))])
        row = gate.measured_relation(raw, telemetry, 'occupied', 2)
        self.assertEqual(row['prediction_bytes'], 120)
        self.assertEqual(row['reserved_margin_bytes'], 20)
        self.assertEqual(gate.gate_exit([row]), 0)


class RevertBranch(unittest.TestCase):
    def test_actual_oom_branch_preserves_temporary_repository(self):
        # This executes legacy.main's actual return-59 branch. Git state is real;
        # only child workers are mocked. No solver import or live fork lookup.
        with tempfile.TemporaryDirectory(prefix='leg23-revert-') as tmp:
            root = Path(tmp)
            fork = root / 'temporary-solver'
            source = fork / 'torchfdtd/adjoint_memory.py'
            source.parent.mkdir(parents=True)
            original = b'# pre-leg-19 temporary sentinel\n'
            source.write_bytes(original)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(fork), *args], text=True).strip()
            git('init', '-q')
            git('add', 'torchfdtd/adjoint_memory.py')
            git('-c', 'user.name=Leg23 Test', '-c', 'user.email=leg23@example.invalid',
                'commit', '-qm', 'Temporary pre-leg source')
            commit = git('rev-parse', 'HEAD')
            source.write_bytes(b'# temporary candidate that must be reverted\n')
            (root / 'env').mkdir()
            (root / 'env/refs.env').write_text('')
            run = root / 'run'
            run.mkdir()
            config = root / 'config.json'
            config.write_text(json.dumps(dict(policy='mock-revert-only',
                historical_full_device={}, cost_valve_seconds=60, pre_leg_fork_commit=commit)))
            called = []
            real_run = subprocess.run
            def worker(command, **kwargs):
                if '--worker' not in command:
                    return real_run(command, **kwargs)
                variant, index = command[-2:]
                called.append((variant, int(index)))
                if variant == 'new':
                    return SimpleNamespace(returncode=59)
                (run / f'{variant}-{index}.json').write_text(json.dumps(dict(
                    variant=variant, measured={'same': True}, plane_sha256={'same': 'hash'})))
                return SimpleNamespace(returncode=0)
            with patch.dict(os.environ, dict(SOLVER_FORK=str(fork), PARITY_CONFIG=str(config),
                                             RUN_DIR_ABS=str(run))), \
                    patch.object(legacy.Path, 'cwd', return_value=root), \
                    patch.object(legacy.subprocess, 'run', side_effect=worker):
                self.assertEqual(legacy.main(), 59)
            self.assertEqual(called, [('old', 2), ('frozen', 2), ('new', 0)])
            self.assertEqual(source.read_bytes(), b'# temporary candidate that must be reverted\n')
            self.assertEqual(git('rev-parse', 'HEAD'), commit)
            self.assertEqual(git('status', '--porcelain'), 'M torchfdtd/adjoint_memory.py')
            report = json.loads((run / 'admission.json').read_text())
            self.assertEqual(report['status'], 'stopped-worker-failure')
            self.assertNotIn('C4_reversion', report)
            print('OOM STOP CONTROL: real legacy OOM branch returned 59; temporary source '
                  'preserved byte-for-byte; no retry; live fork never referenced.', flush=True)


if __name__ == '__main__':
    unittest.main()
