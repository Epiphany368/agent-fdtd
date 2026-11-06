"""CPU-only dependency lifetime and concurrent initialization tests."""
import gc
import threading
import time
import types
import weakref
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
import torchfdtd.cuda_bootstrap as bootstrap


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(bootstrap,'_module',None)
    monkeypatch.setattr(bootstrap,'_failure',None)


def test_retained_dependency_traceback_excludes_existing_caller_tensor(monkeypatch):
    retained=[];module=types.ModuleType('fake_cupy')
    def importer(name):
        assert name=='cupy'
        try:raise ImportError('missing optional test dependency')
        except ImportError as error:retained.append(error)
        return module
    monkeypatch.setattr(bootstrap,'import_module',importer)
    def caller():
        epsilon=torch.ones(4096);owner=weakref.ref(epsilon)
        assert bootstrap.prepare_cuda_kernels() is module
        return owner
    owner=caller();gc.collect()
    assert owner() is None
    frame=retained[0].__traceback__.tb_frame;names=[]
    while frame is not None:names.append(frame.f_code.co_name);frame=frame.f_back
    assert 'caller' not in names and '_import_worker' in names


def test_failed_import_actionable_transactional_and_retry(monkeypatch):
    owners=[]
    class Marker:pass
    def failing(name):
        marker=Marker();owners.append(weakref.ref(marker))
        raise ModuleNotFoundError('No module named cupy')
    monkeypatch.setattr(bootstrap,'import_module',failing)
    with pytest.raises(RuntimeError,match=r'torchfdtd\[cuda-kernels\].*ModuleNotFoundError') as raised:
        bootstrap.prepare_cuda_kernels()
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    gc.collect();assert all(owner() is None for owner in owners)
    assert isinstance(bootstrap._failure,str) and bootstrap._module is None
    module=types.ModuleType('fake_cupy')
    monkeypatch.setattr(bootstrap,'import_module',lambda name:module)
    assert bootstrap.prepare_cuda_kernels() is module


def test_concurrent_callers_share_one_zero_argument_import(monkeypatch):
    calls=[];module=types.ModuleType('fake_cupy');caller=threading.get_ident()
    def importer(name):
        calls.append(threading.get_ident());time.sleep(.02)
        return module
    monkeypatch.setattr(bootstrap,'import_module',importer)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda _:bootstrap.prepare_cuda_kernels(),range(16)))
    assert all(value is module for value in results)
    assert len(calls)==1 and calls[0]!=caller


CUDA_CHILD = r"""
import gc,importlib.abc,importlib.util,json,sys,weakref
import torch
# Reproduce the real optional-test-dependency failure even on a dev machine
# where pytest is installed. This affects this fresh child process only.
class MissingPytest(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in ('pytest','_pytest') or fullname.startswith('_pytest.'):
            raise ModuleNotFoundError('pytest intentionally absent in isolated lifetime test')
sys.meta_path.insert(0,MissingPytest())
if len(sys.argv)>1:
    spec=importlib.util.spec_from_file_location('private_bootstrap',sys.argv[1])
    bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)
else:
    import torchfdtd.cuda_bootstrap as bootstrap
from torchfdtd.models import Project,Region,Source,Monitor
from torchfdtd.differentiable import _System,_FDTD,_Checkpoints,AdjointOptions
assert torch.cuda.is_available()
references={}
r=Region(dimension='3d',size=(.6,.6,.6),mesh=.1,steps=12,backend='cuda',material_sampling='yee',boundaries={a+'_'+side:{'kind':'periodic'} for a in 'xyz' for side in ('min','max')})
p=Project(region=r,sources=[Source(component='Ex',center=(0,0,0),pulse='continuous')],monitors=[Monitor(component='Ex',center=(.1,0,0))])
torch.cuda.init();torch.cuda.synchronize();before=torch.cuda.memory_allocated()
def caller():
    cpu_epsilon=torch.ones(4096)
    epsilon=torch.ones(r.shape,device='cuda',requires_grad=True)
    references.update(cpu_epsilon=weakref.ref(cpu_epsilon),epsilon=weakref.ref(epsilon))
    if len(sys.argv)>1:bootstrap.prepare_cuda_kernels()
    with torch.no_grad():system=_System(p.model_copy(deep=True),epsilon)
    references.update(system=weakref.ref(system),electric=weakref.ref(system.grid.E),magnetic=weakref.ref(system.grid.H))
    options=AdjointOptions(checkpoints=2,backward_kernel='fused');report={}
    check=_Checkpoints(system,options,report,admission=True);check.close();del check
    signals=_FDTD.apply(epsilon,system,options,report,None)
    gradient,=torch.autograd.grad(signals.square().sum(),epsilon)
    assert torch.isfinite(gradient).all() and gradient.abs().max()>0
    torch.cuda.synchronize()
caller();gc.collect();torch.cuda.synchronize()
released={key:value() is None for key,value in references.items()}
import cupy.testing._pytest_impl as retained
frame=retained._error.__traceback__.tb_frame;frames=[]
while frame is not None:
    frames.append(frame.f_code.co_name);frame=frame.f_back
record=dict(released=released,allocated_delta=torch.cuda.memory_allocated()-before,retained_caller='caller' in frames,retained_worker='_import_worker' in frames)
print(json.dumps(record))
assert all(released.values()) and record['allocated_delta']==0
assert not record['retained_caller'] and record['retained_worker']
"""


@pytest.mark.long
def test_real_cuda_cold_subprocess_ownership():
    import os
    import subprocess
    import sys
    if os.environ.get('TORCHFDTD_RUN_CUDA_BOOTSTRAP_TEST')!='1':
        pytest.skip('Opt-in isolated CUDA lifetime test, requires an idle GPU.')
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    subprocess.run([sys.executable,'-c',CUDA_CHILD],check=True,capture_output=True,text=True,timeout=180)
