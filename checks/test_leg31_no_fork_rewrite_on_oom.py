"""Drive the real worker-exit decision without a solver, CUDA, or a git fixture."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import leg19_admission as admission

JOBS = [('old', 2), ('frozen', 2), ('new', 0), ('new', 1), ('new', 2),
        ('low-budget', 2), ('occupied', 2)]


class NoForkRewrite(unittest.TestCase):
    def check_failure(self, exit_status, failed_job):
        with tempfile.TemporaryDirectory(prefix='leg31-decision-') as tmp:
            root = Path(tmp)
            # This path never exists. Observe the decision to write, rather
            # than whether a fabricated repository happens to be restored.
            fork = root / 'solver-must-not-be-written'
            (root / 'env').mkdir()
            (root / 'env/refs.env').write_text('')
            run = root / 'run'
            run.mkdir()
            config = root / 'config.json'
            cfg = json.loads((ROOT / 'configs/leg19-admission.json').read_text())
            # A stale external config must not re-enable the removed policy.
            cfg['pre_leg_fork_commit'] = 'ad436c9'
            config.write_text(json.dumps(cfg))
            called, fork_writes = [], []
            real_open = Path.open

            def observe_open(path, mode='r', *args, **kwargs):
                if path.is_relative_to(fork) and any(flag in mode for flag in 'wax+'):
                    fork_writes.append((str(path.relative_to(fork)), mode))
                    # Contain the old defect while letting main finish, so
                    # the regression fails on the no-write assertion itself.
                    return io.BytesIO() if 'b' in mode else io.StringIO()
                return real_open(path, mode, *args, **kwargs)

            def worker(command, **kwargs):
                self.assertEqual(command[1:4], ['-B', admission.__file__, '--worker'])
                job = (command[-2], int(command[-1]))
                called.append(job)
                if job == failed_job:
                    return SimpleNamespace(returncode=exit_status)
                admission.dump(run / f'{job[0]}-{job[1]}.json', dict(
                    variant=job[0], measured={'same': True}, plane_sha256={'same': 'hash'}))
                return SimpleNamespace(returncode=0)

            with patch.dict(os.environ, dict(SOLVER_FORK=str(fork),
                    PARITY_CONFIG=str(config), RUN_DIR_ABS=str(run))), \
                    patch.object(admission.sys, 'argv', ['leg19_admission.py']), \
                    patch.object(admission.Path, 'cwd', return_value=root), \
                    patch.object(admission.Path, 'open', observe_open), \
                    patch.object(admission.subprocess, 'run', side_effect=worker), \
                    patch.object(admission.subprocess, 'check_output',
                                 return_value=b'old source must never be written\n') as git_read:
                actual_exit = admission.main()

            report = json.loads((run / 'admission.json').read_text())
            print(f'worker={failed_job}; simulated exit={exit_status}; '
                  f'driver exit={actual_exit}; fork writes={len(fork_writes)}', flush=True)
            self.assertEqual(actual_exit, exit_status)
            self.assertEqual(called, JOBS[:JOBS.index(failed_job) + 1])
            self.assertEqual(report['status'], 'stopped-worker-failure')
            self.assertEqual(report['worker_exit_statuses'][-1], dict(
                variant=failed_job[0], index=failed_job[1], exit_status=exit_status))
            self.assertEqual(fork_writes, [], 'worker failure must never write solver source')
            git_read.assert_not_called()
            self.assertNotIn('C4_reversion', report)
            self.assertFalse(fork.exists())

    def test_oom_stops_without_fork_write_or_retry(self):
        for job in JOBS:
            with self.subTest(job=job):
                self.check_failure(59, job)

    def test_non_oom_failure_stops_without_fork_write_or_retry(self):
        for job in JOBS:
            with self.subTest(job=job):
                self.check_failure(57, job)


if __name__ == '__main__':
    unittest.main()
