"""Leg 19 admission safety boundaries; no CUDA allocation in these unit tests."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from parity import C0
import torch
from torchfdtd import AdjointOptions, Monitor
from torchfdtd import adjoint_memory as memory
from run_b9_native_shape_n8 import scene

SPEC = importlib.util.spec_from_file_location('leg19', ROOT/'src/leg19_admission.py')
LEG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEG)


class ForwardAdmission(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'configs/leg19-admission.json').read_text())
        self.project = scene(self.cfg,self.cfg['cases'][0])['p']
        self.env = patch.dict(os.environ, {'TORCHFDTD_ADMISSION_MODEL':'forward-v1'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.budget = patch.object(memory,'cuda_budget_limit',return_value=2**60)
        self.budget.start()
        self.addCleanup(self.budget.stop)
        live_patch = patch.object(torch.cuda,'memory_allocated',return_value=0)
        self.live = live_patch.start()
        self.addCleanup(live_patch.stop)

    def reservation(self, options=None, **kwargs):
        return memory._resident_reservation(self.project,options or AdjointOptions(),'cuda',**kwargs)

    def test_physical_inputs_are_exact_copies(self):
        base=json.loads((ROOT/'configs/leg16-tooth-convergence.json').read_text())
        for key in ('core_index','cladding_index','wavelength_um','width_um','precision'):
            self.assertEqual(self.cfg[key],base[key])
        self.assertEqual(self.cfg['cases'],base['cases'][:3])

    def test_grad_enabled_preserves_original_bound_and_all_report_keys(self):
        with torch.enable_grad():
            enabled=self.reservation()
            with patch.dict(os.environ,{'TORCHFDTD_ADMISSION_MODEL':''}):
                default=self.reservation()
        self.assertEqual(enabled,default)
        self.live.assert_not_called()

    def test_default_off_is_independent_of_grad_mode(self):
        with patch.dict(os.environ,{'TORCHFDTD_ADMISSION_MODEL':''}):
            with torch.enable_grad():
                old=self.reservation()
            with torch.no_grad():
                forward=self.reservation()
        self.assertEqual(old,forward)
        self.live.assert_not_called()

    def test_adjoint_estimator_under_no_grad_keeps_full_bound_with_opt_in(self):
        # The estimator supports points; retain the first fixture monitor's
        # point metadata without changing the measured plane fixture.
        monitor=self.project.monitors[0]
        point=Monitor(**{key: getattr(monitor,key) for key in Monitor.model_fields if key!='kind'})
        project=self.project.model_copy(update={'monitors':[point]})
        with torch.enable_grad():
            full=memory.estimate_adjoint_memory(project,device='cuda')
        with torch.no_grad():
            estimate=memory.estimate_adjoint_memory(project,device='cuda')
            with patch.dict(os.environ,{'TORCHFDTD_ADMISSION_MODEL':''}):
                default=memory.estimate_adjoint_memory(project,device='cuda')
        self.assertEqual(estimate,full)
        self.assertEqual(estimate,default)
        self.assertNotIn('forward_only',estimate)
        self.assertEqual(estimate['device_checkpoint_reservation_bytes'],4*estimate['restart_state_bytes'])
        self.live.assert_not_called()

    def test_unused_slots_are_omitted_only_with_explicit_forward_opt_in(self):
        with torch.no_grad():
            result=self.reservation(AdjointOptions(checkpoints=32))
        self.assertEqual(result['unused_slot_reservation_bytes'],0)
        self.assertEqual(result['omitted_unused_slot_bytes'],32*result['restart_state_bytes'])
        self.assertEqual(result['device_checkpoint_reservation_bytes'],0)
        self.assertEqual(result['workspace_model'],'forward_cuda_allocations_v1')

    def test_external_live_tensors_increase_requirement(self):
        with torch.no_grad():
            baseline=self.reservation()
            with patch.object(torch.cuda,'memory_allocated',return_value=2**30):
                pressured=self.reservation()
        self.assertGreater(pressured['gpu_reservation_bytes'],baseline['gpu_reservation_bytes'])
        self.assertEqual(pressured['workspace_components_bytes']['live_inputs_and_external_allocations'],2**30)

    def test_gpu_refusal_is_self_contained(self):
        with torch.no_grad(),patch.object(memory,'cuda_budget_limit',return_value=1):
            with self.assertRaises(ValueError) as caught:
                self.reservation()
        text=str(caught.exception)
        for field in ('required_bytes=','limit_bytes=1','unused_slot_reservation_bytes=0','omitted_unused_slot_bytes=','unused_slot_count=4','reason=grad_disabled'):
            self.assertIn(field,text)

    def test_explicit_resident_budget_keeps_same_diagnostics(self):
        with torch.no_grad(), self.assertRaisesRegex(ValueError,'required_bytes=.*unused_slot_reservation_bytes=0'):
            self.reservation(AdjointOptions(resident_budget_bytes=1))

    def test_tensor_backward_option_retains_conservative_bound(self):
        with torch.no_grad():
            result=self.reservation(AdjointOptions(backward_kernel='torch'))
        self.assertEqual(result['workspace_model'],'conservative_tensor_bound')
        self.assertNotIn('forward_only',result)

    def test_unmeasured_checkpoint_tiers_are_not_reduced(self):
        with torch.no_grad():
            result=self.reservation(AdjointOptions(storage='host'))
        self.assertNotIn('forward_only',result)
        self.assertGreater(result['host_checkpoint_reservation_bytes'],0)

    def test_spectral_groups_reserve_the_largest_live_gather_and_cast(self):
        spectral=SimpleNamespace(components=('Ex',)*4+('Hx',)*2,block_size=32,
            groups={'E':list(range(4)),'H':[4,5]},
            reservation=lambda depth:dict(spectral_output_bytes=0,spectral_reservation_bytes=6144))
        with torch.no_grad(),patch.object(memory,'_spectral_library_reservation',return_value=0):
            result=self.reservation(spectral=spectral)
        self.assertEqual(result['forward_spectral_block_bytes'],4608)
        self.assertEqual(result['omitted_spectral_group_overcount_bytes'],1536)
        self.assertGreaterEqual(result['history_reservation_bytes'],4608)

    def test_all_electric_spectral_samples_retain_the_full_block_bound(self):
        spectral=SimpleNamespace(components=('Ex',)*6,block_size=32,
            groups={'E':list(range(6)),'H':[]},
            reservation=lambda depth:dict(spectral_output_bytes=0,spectral_reservation_bytes=6144))
        with torch.no_grad(),patch.object(memory,'_spectral_library_reservation',return_value=0):
            result=self.reservation(spectral=spectral)
        self.assertEqual(result['omitted_spectral_group_overcount_bytes'],0)
        self.assertEqual(result['forward_spectral_block_bytes'],6144)

    def test_missing_full_device_context_is_not_an_extra_gate(self):
        self.assertEqual(LEG.verdict({'criteria':{'C1':False,'C2':True}}),'rejected')
        self.assertEqual(LEG.verdict({'criteria':{'C1':True,'C2':True},'historical_full_device':self.cfg['historical_full_device']}),'partial')
        self.assertIsNone(self.cfg['historical_full_device']['memory_ratio_after'])
        self.assertIsNone(self.cfg['historical_full_device']['time_ratio_after'])

    def test_reporting_envelope_is_accepted_by_shared_verification(self):
        from verification import verify_readout,STAGES
        report=LEG.readout_envelope(dict(policy=self.cfg['policy'],status='completed',
                                        verdict='partial',simulation_executed=True))
        self.assertEqual(report['stages'],list(STAGES))
        self.assertIsInstance(verify_readout(report,self.cfg),list)


if __name__=='__main__':
    unittest.main()
