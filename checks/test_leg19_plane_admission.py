"""Leg 22: real plane batch admission, without CUDA execution or allocation."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from parity import C0
import torch
from torchfdtd import AdjointOptions
from torchfdtd import adjoint_batch as batch
from torchfdtd import adjoint_memory as memory
from torchfdtd import execution_tuning as execution
from torchfdtd import plane_execution as planes
from run_b9_native_shape_n8 import scene


class PlaneBatchAdmission(unittest.TestCase):
    def setUp(self):
        cfg = json.loads((ROOT/'configs/leg19-admission.json').read_text())
        self.project = scene(cfg, cfg['cases'][0])['p']
        env = patch.dict(os.environ, {'TORCHFDTD_ADMISSION_MODEL': 'forward-v1'})
        env.start()
        self.addCleanup(env.stop)
        # Keep all model/layout/spectral/transfer arithmetic real. Mock only
        # external capacity and allocation queries; all tensors remain on CPU.
        for module in (memory, execution):
            capacity = patch.object(module, 'cuda_budget_limit', return_value=2**60)
            capacity.start()
            self.addCleanup(capacity.stop)
        for module in (memory, execution, planes, batch):
            host = patch.object(module, 'host_memory', return_value={'available_bytes': 2**60})
            host.start()
            self.addCleanup(host.stop)
        live = patch.object(torch.cuda, 'memory_allocated', return_value=0)
        self.live = live.start()
        self.addCleanup(live.stop)
        capability = patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0))
        capability.start()
        self.addCleanup(capability.stop)
        policy = execution.AdjointExecutionPolicy(resident=AdjointOptions(), device='cuda')
        spec = batch.AdjointCase(self.project, policy,
                                frequency_hz=[C0/(cfg['wavelength_um']*1e-6)])
        self.case = batch._PreparedCase(spec)
        self.parameters = (torch.empty(self.project.region.shape, dtype=torch.float64,
                                       device='cpu', requires_grad=True),)

    def test_plane_batch_no_grad_keeps_full_adjoint_bound_and_default_report(self):
        self.assertTrue(self.case.planes)
        with torch.enable_grad():
            full = self.case.reservation(self.parameters)
        with torch.no_grad():
            replay = self.case.reservation(self.parameters)
            with patch.dict(os.environ):
                os.environ.pop('TORCHFDTD_ADMISSION_MODEL', None)
                default = self.case.reservation(self.parameters)
        print('Plane batch reservation (bytes): full={}, no_grad_opt_in={}, unset={}, '
              'full_checkpoint={}, replay_checkpoint={}'.format(
                  full['gpu_reservation_bytes'], replay['gpu_reservation_bytes'],
                  default['gpu_reservation_bytes'], full['device_checkpoint_reservation_bytes'],
                  replay['device_checkpoint_reservation_bytes']), flush=True)
        self.assertEqual(replay, full)
        self.assertEqual(replay, default)
        self.assertNotIn('forward_only', replay)
        self.assertEqual(replay['device_checkpoint_reservation_bytes'],
                         4*replay['restart_state_bytes'])
        self.live.assert_not_called()

    def test_replay_rejects_shared_budget_below_full_bound_before_execution(self):
        with torch.enable_grad():
            full = self.case.reservation(self.parameters)
            run = batch._BatchRun([self.case], batch.AdjointBatchOptions(
                gpu_budget_bytes=full['gpu_reservation_bytes']), self.parameters)
        # One byte below the real full bound must refuse, including at the
        # actual _replay entry, which admits before entering enable_grad.
        run.options = replace(run.options, gpu_budget_bytes=full['gpu_reservation_bytes']-1)
        with patch.object(self.case.model, 'forward', side_effect=AssertionError(
                'Shared batch budget admitted a plane replay below its full adjoint bound')) as execute:
            for selected in (True, False):
                with self.subTest(opt_in=selected), patch.dict(os.environ):
                    execute.reset_mock()
                    if not selected:
                        os.environ.pop('TORCHFDTD_ADMISSION_MODEL', None)
                    with torch.no_grad(), self.assertRaisesRegex(ValueError, 'shared batch GPU budget'):
                        batch._replay(run, self.case, self.parameters, (), ())
                    execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
