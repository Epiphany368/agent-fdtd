import json
import threading

import numpy as np
import pytest
import torch

from torchfdtd import tune_tensor_batch, run_tensor_batch, Simulation, optimize, FieldMonitor, SpectrumSettings
from torchfdtd.solver import estimate
from test_tensor_batch import cases


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')


def test_measured_tuning_checks_outputs_and_saves_complete_cost(tmp_path):
    pytest.importorskip('cupy')
    projects=cases(3,'float64')
    for p in projects:
        p.monitors.append(FieldMonitor(id='plane',center=(.8,0,0),size=(0,1,1),
            spectrum=SpectrumSettings(sampling='frequency',frequency_points=3,apodization='none')))
    before=[p.model_dump() for p in projects]
    progress=[]
    tuning=tune_tensor_batch(projects,candidates=(1,2,3,16),repeats=2,progress=progress.append,cuda_graph_steps=8)
    assert tuning.cuda_graph_steps==8
    assert len(tuning.candidates)==3 and len(progress)==9
    best=min(tuning.candidates,key=lambda r:(r['median_wall_seconds'],r['cohort_size']))
    assert tuning.cohort_size==best['cohort_size']
    assert all(c['native_bitwise_identical'] and len(c['samples'])==2 for c in tuning.candidates)
    assert tuning.seconds>=sum(c['warmup_seconds']+sum(s['wall_seconds'] for s in c['samples']) for c in tuning.candidates)
    tuning.save(tmp_path/'tuning.json')
    assert json.loads((tmp_path/'tuning.json').read_text())==tuning.as_dict()
    with pytest.raises(FileExistsError):tuning.save(tmp_path/'tuning.json')
    assert before==[p.model_dump() for p in projects]
    actual=run_tensor_batch(projects,cohort_size=tuning.cohort_size)
    for item,p in zip(actual.items,projects):
        np.testing.assert_array_equal(item.load().electric,Simulation(p).run().electric)


def test_tuning_memory_rejection_and_cancellation():
    pytest.importorskip('cupy')
    projects=cases(2)
    free,_=torch.cuda.mem_get_info()
    fraction=1.5*estimate(projects[0])['estimated_memory_mb']*2**20/free
    tuning=tune_tensor_batch(projects,candidates=(1,2),repeats=1,memory_fraction=fraction)
    assert tuning.cohort_size==1
    assert tuning.candidates[1]['status']=='memory_rejected'
    assert not tuning.candidates[1]['samples']
    with pytest.raises(ValueError,match='No candidate fits'):
        tune_tensor_batch(projects,candidates=(1,),repeats=1,memory_fraction=1e-12)
    event=threading.Event();event.set()
    with pytest.raises(InterruptedError):tune_tensor_batch(projects,cancel=event)
    with pytest.raises(ValueError,match='fixed-duration'):
        p=projects[0].model_copy(deep=True);p.region.run_control.auto_shutoff=True
        tune_tensor_batch([p],candidates=(1,))


def test_tensor_design_matches_independent_forward_design_and_rejects_resume(tmp_path):
    pytest.importorskip('cupy')
    p=cases(1,'float64')[0]
    def objective(r):return float((np.linalg.norm(r.signals)-.2)**2)
    # A native serial runner isolates optimizer semantics from cohort scheduling.
    class Serial:
        def run(self,cases,objective,**kwargs):
            from torchfdtd import BatchItem,BatchReport
            return BatchReport([BatchItem(c.id,'completed',c.parameters,
                metrics={'objective':objective(Simulation(c.project).run())}) for c in cases],0,{})
    options=dict(population=4,generations=2,seed=7)
    ref=optimize(p,{'sources.0.amplitude':(.1,2)},objective,runner=Serial(),**options)
    got=optimize(p,{'sources.0.amplitude':(.1,2)},objective,execution='tensor',cohort_size=2,
                 output_dir=tmp_path/'design',cuda_graph_steps=8,**options)
    assert got.as_dict()==ref.as_dict()
    assert (tmp_path/'design/generation-0002/tensor-batch.json').exists()
    with pytest.raises(ValueError,match='resume'):
        optimize(p,{'sources.0.amplitude':(.1,2)},objective,execution='tensor',resume=True)
    with pytest.raises(ValueError,match='process runner'):
        optimize(p,{'sources.0.amplitude':(.1,2)},objective,execution='tensor',runner=Serial())
    with pytest.raises(ValueError,match='Unsupported tensor options'):
        optimize(p,{'sources.0.amplitude':(.1,2)},objective,execution='tensor',max_workers=2)
