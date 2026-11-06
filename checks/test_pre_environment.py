"""Negative control tests for preflight git upward traversal hardening (LEG 42).

Verifies that:
1. An un-repo'd fork directory is rejected with exit 65.
2. A fork directory that traverses upward to an ancestor repository is rejected with exit 65.
3. An upstream directory that traverses upward is rejected with exit 65 when required.
4. An upstream directory that traverses upward in read-only audit mode is marked unavailable
   instead of crashing on AssertionError, allowing audit to proceed.
5. check_solver_git unit contracts behave strictly per specification.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / 'checks' / 'pre_environment.py'
REFUSAL_EXIT = 65


def run_preflight(config, env_overrides=None):
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / 'config.json'
        cfg.write_text(json.dumps(config))
        env = {
            k: v for k, v in os.environ.items()
            if k not in ('PARITY_CONFIG', 'PARITY_SOLVER_ROOT',
                         'PARITY_SOLVER_UPSTREAM', 'PARITY_SOLVER_ROLE')
        }
        env['PARITY_CONFIG'] = str(cfg)
        if env_overrides:
            env.update(env_overrides)
        return subprocess.run(
            [sys.executable, '-B', str(PREFLIGHT)],
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )


class PreEnvironmentGitHardeningTests(unittest.TestCase):
    def test_fork_unrepro_directory_rejected_with_exit_65(self):
        with tempfile.TemporaryDirectory() as td:
            non_repo = Path(td) / 'plain_dir'
            non_repo.mkdir()
            res = run_preflight(
                {'task': 'test unrepro fork', 'kind': 'audit'},
                env_overrides={'PARITY_SOLVER_ROOT': str(non_repo)}
            )
            self.assertEqual(res.returncode, REFUSAL_EXIT, res.stdout)
            self.assertIn('solver fork is unavailable:', res.stdout)
            self.assertIn('no simulation was run', res.stdout)

    def test_fork_upward_traversal_rejected_with_exit_65(self):
        with tempfile.TemporaryDirectory() as td:
            outer = Path(td) / 'outer_repo'
            outer.mkdir()
            subprocess.run(['git', 'init', str(outer)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(outer), '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                            'commit', '--allow-empty', '-m', 'init'],
                           check=True, capture_output=True)
            sub = outer / 'nested_solver'
            sub.mkdir()
            res = run_preflight(
                {'task': 'test upward traversal fork', 'kind': 'audit'},
                env_overrides={'PARITY_SOLVER_ROOT': str(sub)}
            )
            self.assertEqual(res.returncode, REFUSAL_EXIT, res.stdout)
            self.assertIn('solver fork is unavailable:', res.stdout)
            self.assertIn('Git upward traversal detected', res.stdout)
            self.assertIn('no simulation was run', res.stdout)

    def test_upstream_upward_traversal_rejected_with_exit_65_when_required(self):
        with tempfile.TemporaryDirectory() as td:
            outer = Path(td) / 'outer_repo'
            outer.mkdir()
            subprocess.run(['git', 'init', str(outer)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(outer), '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                            'commit', '--allow-empty', '-m', 'init'],
                           check=True, capture_output=True)
            sub = outer / 'nested_upstream'
            sub.mkdir()
            res = run_preflight(
                {'task': 'test required upstream', 'kind': 'canary'},
                env_overrides={'PARITY_SOLVER_UPSTREAM': str(sub), 'PARITY_SOLVER_ROLE': 'upstream'}
            )
            self.assertEqual(res.returncode, REFUSAL_EXIT, res.stdout)
            self.assertIn('frozen upstream baseline is unavailable:', res.stdout)
            self.assertIn('no simulation was run', res.stdout)

    def test_upstream_upward_traversal_in_audit_mode_reports_unavailable_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            outer = Path(td) / 'outer_repo'
            outer.mkdir()
            subprocess.run(['git', 'init', str(outer)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(outer), '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                            'commit', '--allow-empty', '-m', 'init'],
                           check=True, capture_output=True)
            sub = outer / 'nested_upstream'
            sub.mkdir()

            fork_repo = Path(td) / 'valid_fork'
            fork_repo.mkdir()
            subprocess.run(['git', 'init', str(fork_repo)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(fork_repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                            'commit', '--allow-empty', '-m', 'init'],
                           check=True, capture_output=True)

            res = run_preflight(
                {'task': 'test audit upstream', 'kind': 'audit'},
                env_overrides={
                    'PARITY_SOLVER_UPSTREAM': str(sub),
                    'PARITY_SOLVER_ROOT': str(fork_repo),
                    'PARITY_SOLVER_ROLE': 'fork',
                }
            )
            self.assertEqual(res.returncode, 0, res.stdout)
            self.assertIn('preflight: frozen upstream baseline unavailable', res.stdout)
            self.assertIn('read-only audit does not read it', res.stdout)
            self.assertNotIn('AssertionError: FROZEN UPSTREAM MOVED', res.stdout)

    def test_check_solver_git_unit_contract(self):
        sys.path.insert(0, str(ROOT / 'checks'))
        import importlib
        pre_env = importlib.import_module('pre_environment')
        check_solver_git = getattr(pre_env, 'check_solver_git')

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / 'repo'
            repo.mkdir()
            subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                            'commit', '--allow-empty', '-m', 'init'],
                           check=True, capture_output=True)
            info = check_solver_git(repo)
            self.assertIn('commit', info)
            self.assertIn('dirty_lines', info)
            self.assertEqual(info['toplevel'], str(repo.resolve()))

            sub = repo / 'sub'
            sub.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'Git upward traversal detected'):
                check_solver_git(sub)

            non_repo = Path(td) / 'plain'
            non_repo.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'is not a git repository'):
                check_solver_git(non_repo)


if __name__ == '__main__':
    unittest.main()
