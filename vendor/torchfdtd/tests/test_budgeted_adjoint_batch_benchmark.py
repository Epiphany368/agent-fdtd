import pytest
import torch

from benchmarks.budgeted_adjoint_batch import main


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dispersive',[False,True])
def test_coupled_capacity_driver_matches_independent_finite_cone(tmp_path,device,dispersive):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    threads=torch.get_num_threads()
    try:
        report=main(['--output',str(tmp_path/'record.json'),'--execute','--smoke','--size','64',
            '--cases','2','--steps','10','--device',device,'--cpu-threads','2',
            *(['--dispersive'] if dispersive else [])])
    finally:torch.set_num_threads(threads)
    assert report['stage']=='forward_backward_validated' and report['driver_smoke']
    assert report['batch']['replayed_cases']==2
    assert max(report['relative_l2_output_and_gradients'])<1e-4
    assert report['global_epsilon_gradient_norm']>0
