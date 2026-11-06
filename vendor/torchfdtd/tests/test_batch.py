import json
from functools import partial
import multiprocessing as mp
import threading
import numpy as np
import pytest

from torchfdtd import (Project, Region, Source, Monitor, Simulation, Result, BatchRunner,
                         parameter_sweep, parameter_case, optimize)


def small_project():
    return Project(region=Region(size=(3,2,1),mesh=.1,pml_cells=4,steps=150,backend='cpu',precision='float64'),
                   sources=[Source(id='src',center=(-.5,0,0))],monitors=[Monitor(id='out',center=(.5,0,0))])


def trace_energy(result):
    return {'energy':float(np.sum(abs(result.signals)**2))}


def synchronized_trace_energy(result, *, barrier):
    # A serial executor cannot pass this rendezvous. Small simulations alone
    # may finish between OS scheduling slices even when two workers are ready.
    barrier.wait(timeout=60)
    return trace_energy(result)


def invalid_objective(result):return float('nan')


def amplitude_objective(result):
    # A physical inverse problem: recover the source amplitude that gives the
    # calibrated time-domain field trace, not an algebraic parameter objective.
    return float((np.linalg.norm(result.signals)-.2)**2)


def test_parallel_serial_agreement_resume_and_safe_paths(tmp_path):
    base=small_project()
    cases=parameter_sweep(base,{'sources.0.amplitude':[.5,1,1.5,2]})
    reference=Simulation(base).run()
    with mp.get_context('spawn').Manager() as manager, BatchRunner(backend='cpu',max_workers=2) as runner:
        objective=partial(synchronized_trace_energy,barrier=manager.Barrier(2))
        report=runner.run(cases,objective=objective,objective_key='energy-v1',output_dir=tmp_path,keep_results=True)
        report.raise_for_errors()
        assert len({item.pid for item in report.items})==2
        # Independent simulation/objective jobs overlap in distinct processes.
        assert any(a.pid!=b.pid and max(a.started,b.started)<min(a.finished,b.finished)
                   for a in report.items for b in report.items)
        for amplitude,item in zip((.5,1,1.5,2),report.items):
            np.testing.assert_allclose(item.load().signals,reference.signals*amplitude,rtol=1e-12,atol=1e-14)
            loaded=Result.load(item.output)
            np.testing.assert_array_equal(loaded.electric,item.result.electric)
            assert item.metrics['energy']==pytest.approx(np.sum(abs(reference.signals)**2)*amplitude**2)
        resumed=runner.run(cases,objective=trace_energy,objective_key='energy-v1',output_dir=tmp_path,resume=True)
        assert all(x.resumed for x in resumed.items)
        with pytest.raises(FileExistsError):runner.run(cases,output_dir=tmp_path)
        changed=parameter_case(base,{'sources.0.amplitude':3},cases[0].id)
        with pytest.raises(ValueError,match='fingerprint'):runner.run([changed],output_dir=tmp_path,resume=True)
        with pytest.raises(ValueError,match='objective_key'):runner.run(cases,objective=trace_energy,output_dir=tmp_path,resume=True)
    assert base.sources[0].amplitude==1
    with pytest.raises(ValueError,match='parameter path'):parameter_case(base,{'region.not_a_setting':2},'invalid')
    moved=parameter_case(base,{'sources.0.center.0':-.4},'moved')
    assert moved.project.sources[0].center[0]==-.4


def test_failure_cancellation_and_memory_admission(tmp_path):
    p=small_project()
    with BatchRunner(backend='cpu',max_workers=2) as runner:
        report=runner.run([p,p],objective=invalid_objective,output_dir=tmp_path,objective_key='invalid-v1')
        assert not report.successful
        assert all(x.status=='failed' and 'finite' in x.error for x in report.items)
        for item in report.items:
            record=json.loads((tmp_path/(item.id+'.json')).read_text())
            assert record['item']['metrics']=={} and record['item']['status']=='failed'
        with pytest.raises(RuntimeError):report.raise_for_errors()
        stop=threading.Event();stop.set()
        cancelled=runner.run([p,p],cancel=stop)
        assert all(x.status=='cancelled' and x.pid is None for x in cancelled.items)
    with BatchRunner(backend='cpu',memory_limit_mb=.01) as runner:
        with pytest.raises(ValueError,match='batch memory'):runner.run([p])


def test_black_box_design_improves_physical_objective_and_resumes(tmp_path):
    p=small_project()
    with BatchRunner(backend='cpu',max_workers=2) as runner:
        result=optimize(p,{'sources.0.amplitude':(.1,2)},amplitude_objective,runner=runner,
                        population=4,generations=3,seed=7,output_dir=tmp_path,objective_key='amplitude-v1')
        assert result.objective < result.history[0]['best']
        assert result.evaluations==16
        repeated=optimize(p,{'sources.0.amplitude':(.1,2)},amplitude_objective,runner=runner,
                          population=4,generations=3,seed=7,output_dir=tmp_path,resume=True,objective_key='amplitude-v1')
        assert repeated.as_dict()==result.as_dict()
        with pytest.raises(ValueError,match='configuration'):
            optimize(p,{'sources.0.amplitude':(.1,3)},amplitude_objective,runner=runner,
                     population=4,generations=3,seed=7,output_dir=tmp_path,resume=True,objective_key='amplitude-v1')
