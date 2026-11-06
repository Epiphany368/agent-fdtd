"""Policy claims must use full-duration evidence and include calibration cost."""
import json

import pytest
import torch

from benchmarks.streamed_policy import main, selection_metrics


@pytest.fixture(autouse=True)
def restore_thread_count():
    previous=torch.get_num_threads()
    yield
    torch.set_num_threads(previous)


def test_policy_metrics_include_rejections_regressions_and_calibration_overlap():
    tuning=dict(selected_index=2,calibration_steps=[12,24],full_steps=64)
    result=selection_metrics(tuning,[None,10.,8.,5.],25.)
    assert result['baseline_index']==1 and result['best_measured_index']==3
    assert result['selected_full_over_best']==1.6
    assert result['tuning_payback_iterations']==13
    assert result['held_out_duration']
    tuning.update(selected_index=3,calibration_steps=[32,64])
    result=selection_metrics(tuning,[None,4.,8.,5.],25.)
    assert result['baseline_over_selected']==.8
    assert result['tuning_payback_iterations'] is None
    assert not result['held_out_duration']


def test_dispersive_spectral_policy_driver_checks_full_gradients(tmp_path):
    output=tmp_path/'result.json'
    report=main(['--output',str(output),'--device','cpu','--nx','12','--ny','12','--nz','12',
        '--steps','26','--probe-steps','10','--repeats','1','--cpu-threads','1',
        '--dispersive','--spectrum','--complex-bloch','--require-held-out'])
    assert report['stage']=='complete' and report['held_out_duration']
    assert len(report['reference_gradient_norms'])==4
    assert all(v>0 for v in report['reference_gradient_norms'])
    assert len(report['full_records'])==3
    for records in report['full_records']:
        assert len(records)==1
        assert max(records[0]['relative_l2_output_and_gradients'])<1e-8
    assert report['tuning_wall_seconds']>=report['tuning']['tuning_seconds']
    assert json.loads(output.read_text())['stage']=='complete'
    assert 'benchmarks/streamed_policy.py' in report['source_sha256']


def test_holdout_rejected_before_running_calibration(tmp_path,monkeypatch):
    monkeypatch.setattr('benchmarks.streamed_policy.tune_streamed',
        lambda *a,**kw:pytest.fail('Ran an overlapping calibration'))
    output=tmp_path/'result.json'
    with pytest.raises(ValueError,match='Calibration can reach'):
        main(['--output',str(output),'--device','cpu','--steps','12','--probe-steps','10',
              '--nx','12','--ny','12','--require-held-out'])
    assert not output.exists()


@pytest.mark.parametrize('extra',[[],['--dispersive']])
def test_real_fp32_time_history_driver(tmp_path,extra):
    report=main(['--output',str(tmp_path/'time.json'),'--device','cpu','--nx','12','--ny','12',
        '--steps','26','--probe-steps','10','--repeats','1','--cpu-threads','1',
        '--precision','float32','--require-held-out',*extra])
    assert report['stage']=='complete' and report['held_out_duration']
    assert report['tuning']['observation']=='time_history'
    assert len(report['reference_gradient_norms'])==(4 if extra else 1)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_dispersive_policy_driver(tmp_path):
    report=main(['--output',str(tmp_path/'cuda.json'),'--device','cuda','--nx','12','--ny','12',
        '--steps','26','--probe-steps','10','--repeats','1','--cpu-threads','1',
        '--dispersive','--spectrum','--complex-bloch','--require-held-out'])
    assert report['stage']=='complete' and report['held_out_duration']
    assert len(report['full_records'])==4
    assert report['tuning']['candidates'][-1]['policy']['tile_transfers']=='async'
    assert all(rows[0]['peak_cuda_allocated_bytes']>0 for rows in report['full_records'])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_unified_driver_keeps_resident_and_streamed_full_validation(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    report=main(['--output',str(tmp_path/'unified.json'),'--device',device,'--nx','12','--ny','12',
        '--steps','26','--probe-steps','10','--repeats','1','--cpu-threads','1',
        '--dispersive','--spectrum','--complex-bloch','--require-held-out','--unified'])
    assert report['stage']=='complete' and report['held_out_duration'] and report['unified_selection']
    policies=[row['policy'] for row in report['tuning']['candidates']]
    assert any(p['resident'] is not None for p in policies)
    assert any(p['streamed'] is not None for p in policies)
    for rows in report['full_records']:
        assert len(rows)==1
        assert max(rows[0]['relative_l2_output_and_gradients'])<1e-8


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('extra',[['--precision','float32'],['--dispersive','--complex-bloch']])
def test_unified_plane_driver_full_fields_and_flux_vjp(tmp_path,device,extra,monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    # Reproduce the Windows replacement failure observed after a complete CPU
    # VJP. Saving may retry the rename, but must retain the numerical verdict.
    from pathlib import Path
    original_replace = Path.replace
    output = tmp_path/'planes.json'
    injected = []
    def replace(source,destination):
        if device=='cpu' and '--dispersive' in extra and destination==output and not injected:
            if json.loads(source.read_text())['stage']=='complete':
                injected.append(True)
                error=PermissionError('injected complete-record replacement failure')
                error.winerror=5
                raise error
        return original_replace(source,destination)
    monkeypatch.setattr(Path,'replace',replace)
    report=main(['--output',str(tmp_path/'planes.json'),'--device',device,'--nx','12','--ny','12',
        '--steps','26','--probe-steps','10','--repeats','1','--cpu-threads','1',
        '--require-held-out','--unified','--planes','--plane-stride','2',*extra])
    assert report['stage']=='complete' and report['held_out_duration'] and report['fixed_planes']
    assert report['tuning']['observation']=='fixed_plane_spectrum_and_flux'
    assert len(report['reference_gradient_norms'])==(4 if '--dispersive' in extra else 1)
    if device=='cpu' and '--dispersive' in extra:
        assert injected == [True]
    assert json.loads(output.read_text())['stage']=='complete'
    for rows in report['full_records']:
        assert len(rows)==1
        assert rows[0]['report']['execution_reservation']['plane_output_bytes']>0
        assert max(rows[0]['relative_l2_output_and_gradients'])<(1e-4 if report['precision']=='float32' else 1e-8)
