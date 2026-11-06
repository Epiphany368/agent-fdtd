"""Negative control tests for tools/launch.py git upward traversal hardening (LEG 42)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'tools' / 'launch.py'


class LaunchGitHardeningTests(unittest.TestCase):
    def setUp(self):
        self.ledger_file = ROOT / 'runs' / 'ledger.jsonl'
        self.orig_ledger = self.ledger_file.read_text(encoding='utf-8') if self.ledger_file.is_file() else None
        self.created_run_dirs = []

    def tearDown(self):
        if self.orig_ledger is not None and self.ledger_file.is_file():
            self.ledger_file.write_text(self.orig_ledger, encoding='utf-8')
        for run_dir in self.created_run_dirs:
            if run_dir.is_dir():
                import shutil
                shutil.rmtree(run_dir, ignore_errors=True)

    def _track_run_dir(self, output):
        for line in output.splitlines():
            if 'run_dir=' in line:
                for token in line.split():
                    if token.startswith('run_dir='):
                        path = Path(token.split('=', 1)[1])
                        self.created_run_dirs.append(path)

    def test_launch_rejects_unrepro_solver_with_exit_65(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / 'config.json'
            cfg.write_text(json.dumps({'task': 'launch test', 'kind': 'audit'}))
            rel_script = ROOT / 'src' / '_test_dummy_launch.py'
            rel_script.write_text('import sys; sys.exit(0)\n')
            try:
                non_repo = Path(td) / 'unrepo_solver'
                non_repo.mkdir()

                env = dict(os.environ, PARITY_SOLVER_ROOT=str(non_repo))
                res = subprocess.run(
                    [sys.executable, '-B', str(LAUNCH), str(cfg), str(rel_script.relative_to(ROOT))],
                    cwd=str(ROOT),
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
                self._track_run_dir(res.stdout)
                self.assertEqual(res.returncode, 65, res.stdout)
                self.assertIn('exit=65', res.stdout)
            finally:
                if rel_script.is_file():
                    rel_script.unlink()

    def test_launch_rejects_upward_traversal_solver_with_exit_65(self):
        with tempfile.TemporaryDirectory() as td:
            outer = Path(td) / 'outer_repo'
            outer.mkdir()
            subprocess.run(['git', 'init', str(outer)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(outer), '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                            'commit', '--allow-empty', '-m', 'init'],
                           check=True, capture_output=True)
            nested = outer / 'nested_solver'
            nested.mkdir()

            cfg = Path(td) / 'config.json'
            cfg.write_text(json.dumps({'task': 'launch test upward', 'kind': 'audit'}))
            rel_script = ROOT / 'src' / '_test_dummy_launch_upward.py'
            rel_script.write_text('import sys; sys.exit(0)\n')
            try:
                env = dict(os.environ, PARITY_SOLVER_ROOT=str(nested))
                res = subprocess.run(
                    [sys.executable, '-B', str(LAUNCH), str(cfg), str(rel_script.relative_to(ROOT))],
                    cwd=str(ROOT),
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
                self._track_run_dir(res.stdout)
                self.assertEqual(res.returncode, 65, res.stdout)
                self.assertIn('exit=65', res.stdout)
            finally:
                if rel_script.is_file():
                    rel_script.unlink()


if __name__ == '__main__':
    unittest.main()
