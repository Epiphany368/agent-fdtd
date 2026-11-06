"""CPU-only regression for forward-v1's live-allocation double count.

The physical fixture is LEG 19's unchanged new-2 case. The occupied live
allocation is the recorded LEG 19 value, not an allocated CUDA tensor.
"""
import json
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import parity
import torch
from torchfdtd import AdjointOptions, adjoint_memory as memory
from run_b9_native_shape_n8 import scene


class OwnRequired(unittest.TestCase):
    def setUp(self):
        cfg = json.loads((ROOT/'configs/leg19-admission.json').read_text())
        self.project = scene(cfg, cfg['cases'][2])['p']

    def reservation(self, live, limit=2**60, **options):
        with torch.no_grad(), patch.dict(os.environ, {'TORCHFDTD_ADMISSION_MODEL':'forward-v1'}), \
                patch.object(torch.cuda, 'memory_allocated', return_value=live), \
                patch.object(memory, 'cuda_budget_limit', return_value=limit):
            return memory._resident_reservation(self.project, AdjointOptions(**options), 'cuda')

    def test_occupied_required_equals_new_2_own_required(self):
        new_2 = self.reservation(203_555_840)
        with self.assertRaises(ValueError) as caught:
            self.reservation(12_908_102_656, limit=406_637_772)
        refusal = str(caught.exception)
        required = int(re.search(r'\brequired_bytes=(\d+)', refusal)[1])
        # Assert the number first, so the baseline fails on the reproduced
        # double count rather than merely on a newly missing report key.
        subtotal = new_2['memory_reservation_bytes'] - new_2['allocation_headroom_bytes']
        own_subtotal = subtotal - new_2['workspace_components_bytes']['live_inputs_and_external_allocations']
        expected = own_subtotal + 8*1024**2 + (own_subtotal+19)//20
        print(f'occupied required_bytes={required}; new-2 own expectation={expected}', flush=True)
        self.assertEqual(required, expected)
        self.assertEqual(required, new_2['own_required_bytes'])
        self.assertIn('live_allocated_bytes=12908102656;', refusal)

    def test_own_demand_is_independent_of_live_process_peak(self):
        new_2 = self.reservation(203_555_840)
        occupied = self.reservation(12_908_102_656)
        self.assertEqual(new_2['own_required_bytes'], occupied['own_required_bytes'])
        self.assertEqual(new_2['required_bytes'], new_2['own_required_bytes'])
        self.assertEqual(occupied['live_allocated_bytes'], 12_908_102_656)
        self.assertGreater(occupied['memory_reservation_bytes'], new_2['memory_reservation_bytes'])
        self.assertEqual(occupied['memory_reservation_bytes'], occupied['gpu_reservation_bytes'])
        subtotal = occupied['workspace_reservation_bytes'] + sum(occupied[k] for k in (
            'device_checkpoint_reservation_bytes', 'device_staging_reservation_bytes',
            'history_reservation_bytes', 'observation_index_bytes',
            'material_packing_reservation_bytes', 'spectral_library_reservation_bytes'))
        self.assertEqual(occupied['memory_reservation_bytes'], subtotal+8*1024**2+(subtotal+19)//20)

    def test_free_capacity_uses_own_demand_and_keeps_boundary(self):
        baseline = self.reservation(203_555_840)
        own = baseline['own_required_bytes']
        self.assertEqual(self.reservation(12_908_102_656, limit=own)['own_required_bytes'], own)
        with self.assertRaisesRegex(ValueError, f'required_bytes={own};'):
            self.reservation(12_908_102_656, limit=own-1)
        with torch.no_grad(), patch.dict(os.environ, {'TORCHFDTD_ADMISSION_MODEL':'forward-v1'}), \
                patch.object(torch.cuda, 'memory_allocated', return_value=12_908_102_656), \
                patch.object(memory, 'cuda_budget_limit', return_value=2**60) as capacity:
            memory._resident_reservation(self.project, AdjointOptions(), 'cuda')
        self.assertEqual(capacity.call_args.args[1], own)


if __name__ == '__main__':
    unittest.main()
