import pytest
import torch

from benchmarks.budgeted_resident import main


@pytest.mark.parametrize('dispersive',[False,True])
def test_near_pml_driver_matches_full_autograd(tmp_path,dispersive):
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        result=main(['--output',str(tmp_path/'result.json'),'--execute','--smoke','--size','64',
            '--device','cpu','--precision','float64',*(['--dispersive'] if dispersive else [])])
    finally:torch.set_num_threads(old)
    assert result['stage']=='forward_backward_validated'
    assert result['driver_smoke'] and not result['exceeds_workbench_cell_guard']
    assert result['oracle_cpml_norm']>0
    assert max(result['relative_l2_output_and_gradients'])<2e-9


def test_small_driver_cannot_claim_large_resident_capacity(tmp_path):
    with pytest.raises(ValueError,match='eight million'):
        main(['--output',str(tmp_path/'absent.json'),'--size','64'])
    assert not (tmp_path/'absent.json').exists()
