import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks.cr_optimization import ProjectedAdamRun


def start(path, resume=False, dtype=torch.float64, **kwargs):
    return ProjectedAdamRun(path, torch.tensor([[.1,.9],[.7,.2]], dtype=dtype),
        {'physics':'test'}, learning_rate=.03, resume=resume, **kwargs)


def update(run):
    score = -(run.density-.45).square().sum()
    return run.step(score, {'response':'synthetic quadratic'})


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_restart_preserves_density_adam_and_evaluated_best(tmp_path, dtype):
    with start(tmp_path/'uninterrupted', dtype=dtype) as whole:
        for _ in range(4):
            update(whole)
        expected = whole.density.detach().clone()
        moments = whole.optimizer.state[whole.density]
    with start(tmp_path/'split', dtype=dtype) as split:
        update(split)
        update(split)
        initial_best = split.best['record']['weighted_bits_per_pixel']
    # An interrupted archive write cannot replace the committed checkpoint.
    (tmp_path/'split/checkpoint.pt.tmp').write_bytes(b'partial')
    with start(tmp_path/'split', resume=True, dtype=dtype) as resumed:
        assert resumed.updates == 2
        for _ in range(2):
            update(resumed)
        torch.testing.assert_close(resumed.density, expected, rtol=0, atol=0)
        for key in moments:
            torch.testing.assert_close(resumed.optimizer.state[resumed.density][key], moments[key], rtol=0, atol=0)
        assert resumed.best['record']['weighted_bits_per_pixel'] > initial_best
        assert resumed.best['record']['update'] == 3  # Updated density 4 is not evaluated yet.
        with torch.no_grad():
            resumed.observe(-(resumed.density-.45).square().sum(), {'final':True})
        resumed.save()
        assert resumed.last_evaluation['update'] == 4
    with pytest.raises(RuntimeError, match='closed'):
        resumed.save()


def test_failed_gradient_and_low_disk_keep_last_commit(tmp_path, monkeypatch):
    with start(tmp_path) as run:
        update(run)
        committed = run.path.read_bytes()
        with pytest.raises(ValueError, match='finite density gradient'):
            run.step((run.density*float('nan')).sum(), {})
        assert run.path.read_bytes() == committed
        monkeypatch.setattr('benchmarks.cr_optimization.shutil.disk_usage', lambda _:SimpleNamespace(free=0))
        with pytest.raises(OSError, match='free-space reserve'):
            update(run)
        assert run.path.read_bytes() == committed
        assert run.committed_updates == 1
        with pytest.raises(RuntimeError, match='commit failed'):
            update(run)
    monkeypatch.undo()
    with start(tmp_path, resume=True) as run:
        assert run.updates == 1


def test_restart_rejects_changed_contract_corruption_and_concurrent_writer(tmp_path):
    with start(tmp_path) as run:
        update(run)
        with pytest.raises(RuntimeError, match='Another process'):
            start(tmp_path, resume=True)
    with pytest.raises(ValueError, match='contract mismatch'):
        ProjectedAdamRun(tmp_path, torch.zeros((2,2), dtype=torch.float64),
            {'physics':'changed'}, learning_rate=.03, resume=True)
    payload = torch.load(tmp_path/'checkpoint.pt', weights_only=True)
    payload['density'][0,0] += .01
    torch.save(payload, tmp_path/'checkpoint.pt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        start(tmp_path, resume=True)


def test_projection_and_failed_initial_archive(tmp_path, monkeypatch):
    monkeypatch.setattr('benchmarks.cr_optimization.shutil.disk_usage', lambda _:SimpleNamespace(free=0))
    with pytest.raises(OSError, match='free-space reserve'):
        start(tmp_path)
    assert not (tmp_path/'ownership.json').exists()
    monkeypatch.undo()
    with start(tmp_path) as run:
        for _ in range(5):
            run.step(100*run.density.sum(), {})
        assert float(run.density.detach().max()) == 1.
        assert bool((run.density >= 0).all())


def synthetic_inputs(tmp_path):
    import hashlib
    import numpy as np
    from test_periodic_adjoint import SPEC
    density = tmp_path/'seed.npy'
    np.save(density, np.array([[0.,1.],[1.,0.]]))
    wavelengths = [450.,540.,650.]
    schedule = tmp_path/'schedule.json'
    schedule.write_text(json.dumps(dict(density_sha256=hashlib.sha256(density.read_bytes()).hexdigest(),
        wavelengths_nm=wavelengths, ray_weights=[.7],
        cases=[[dict(SPEC, wavelength_um=w/1000)] for w in wavelengths])))
    eye = torch.eye(3, dtype=torch.float64)
    context = tmp_path/'context.pt'
    torch.save(dict(context=dict(wavelengths_nm=wavelengths, sampled_qe=[.5,.8,.6],
        source_spectrum=[1.,2.,1.], scene_spectral_basis=eye, electron_calibration=1e-10,
        scene_covariance=eye, scene_target_cross_covariance_xyz=.6*eye, target_covariance_xyz=eye,
        exposure_weighted_cfa_green_e=[20.,80.,320.], reference_weighted_cfa_green_e=100.,
        exposure_probabilities=[.2,.5,.3], read_noise_e_rms=1.5)), context)
    return ['--schedule',str(schedule),'--density',str(density),'--context',str(context),
        '--mesh','.1','--steps','160','--pml-cells','6','--cpu-threads','1',
        '--reference-cache-mib','1','--gpu-budget-gib','1','--host-budget-gib','1',
        '--checkpoint-free-reserve-gib','0','--learning-rate','.001',
        '--slab-width','4','--temporal-depth','4']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('precision', ['float64', 'float32'])
def test_full_torch_cr_optimization_resume_and_streaming(tmp_path, precision):
    from benchmarks.cr_inverse_design import main
    args = synthetic_inputs(tmp_path)+([] if precision == 'float32' else ['--precision',precision])
    whole = main(args+['--output-directory',str(tmp_path/'whole'),'--iterations','2'])
    first = main(args+['--output-directory',str(tmp_path/'split'),'--iterations','1'])
    resumed = main(args+['--output-directory',str(tmp_path/'split'),'--iterations','2','--resume'])
    for a, b in zip(whole['history'], resumed['history']):
        for key in ('update','weighted_bits_per_pixel','density_sha256','next_density_sha256','loss_gradient_l2'):
            assert a[key] == b[key]
        for key in ('response','bits_per_pixel'):
            assert a['metadata'][key] == b['metadata'][key]
    assert resumed['history'][1]['metadata']['restored_forward_cases'] == 3
    assert whole['history'][1]['metadata']['restored_forward_cases'] == 0
    assert whole['final'] == resumed['final']
    assert whole['artifacts']['latest-density']['sha256'] == resumed['artifacts']['latest-density']['sha256']
    assert whole['final']['weighted_bits_per_pixel'] > whole['history'][0]['weighted_bits_per_pixel']
    assert whole['best']['weighted_bits_per_pixel'] >= whole['final']['weighted_bits_per_pixel']
    streamed = main(args+['--output-directory',str(tmp_path/'dram'),'--iterations','1','--execution-policy','dram'])
    a = torch.load(tmp_path/'split/checkpoint.pt', weights_only=True)
    assert a['updates'] == 2 and len(a['history']) == 2
    assert a['density'].dtype == getattr(torch, precision)
    state = next(iter(a['optimizer']['state'].values()))
    assert state['exp_avg'].dtype == state['exp_avg_sq'].dtype == a['density'].dtype
    assert a['contract']['inputs']['precision'] == precision
    field_rtol, information_atol = (2e-7, 1e-10) if precision == 'float64' else (2e-4, 1e-5)
    torch.testing.assert_close(torch.tensor(streamed['final']['metadata']['response']),
        torch.tensor(first['final']['metadata']['response']), rtol=field_rtol, atol=information_atol)
    assert abs(streamed['final']['weighted_bits_per_pixel']-first['final']['weighted_bits_per_pixel']) < information_atol
    assert all(row['loss_gradient_l2'] > 0 for row in whole['history'])
    assert json.loads((tmp_path/'whole/plan.json').read_text())['cases'] == 3
    assert json.loads((tmp_path/'whole/progress.json').read_text())['stage'] == 'requested_updates_and_final_forward_complete'
    changed = 'float32' if precision == 'float64' else 'float64'
    with pytest.raises(ValueError, match='contract mismatch'):
        main(args+['--precision',changed,'--output-directory',str(tmp_path/'split'),
                   '--iterations','3','--resume'])
