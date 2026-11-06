import json
import math
from types import SimpleNamespace

import pytest
import torch

from benchmarks.beyond_vram import field_storage_bytes, main, relative_error, tensor_stats


def test_fp32_capacity_counts_real_field_storage():
    shape=(1152,1024,2048)
    assert field_storage_bytes(shape)==54*1024**3
    assert field_storage_bytes(shape)>51_526_500_352
    assert field_storage_bytes((1024,1024,576),'float64',True)==54*1024**3
    assert field_storage_bytes((1024,1024,576),'float32',False)==13.5*1024**3


def test_bounded_gradient_statistics_reject_nonfinite_and_tiny_wrong_groups():
    x=torch.zeros(100,dtype=torch.float32);x[2]=3e-30;x[80]=4e-30
    norm,count=tensor_stats(x,chunk_elements=16)
    assert count==2 and math.isclose(norm,5e-30,rel_tol=1e-7)
    assert relative_error(x,x*2)==.5
    x[99]=float('nan')
    with pytest.raises(AssertionError,match='Nonfinite'):tensor_stats(x,chunk_elements=16)
    with pytest.raises(AssertionError,match='nonzero'):relative_error(torch.ones(2),torch.zeros(2))


def test_real_fp32_metadata_admission_without_full_fields(tmp_path,monkeypatch):
    # Synthetic capacities only. This is not a GPU execution/capacity result.
    gib=1024**3
    monkeypatch.setattr('torchfdtd.streamed.host_memory',lambda:dict(available_bytes=120*gib))
    monkeypatch.setattr('torchfdtd.streamed.cuda_budget_limit',lambda device,required,budget:budget)
    monkeypatch.setattr('torchfdtd.state_store.disk_free',lambda path:410*gib)
    monkeypatch.setattr('benchmarks.beyond_vram.psutil.virtual_memory',lambda:SimpleNamespace(available=120*gib))
    monkeypatch.setattr('benchmarks.beyond_vram.shutil.disk_usage',lambda path:SimpleNamespace(free=410*gib))
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(46*gib,51_526_500_352))
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'Synthetic metadata test')
    def forbidden(*args,**kwargs):raise AssertionError('No field allocation during planning')
    monkeypatch.setattr('torchfdtd.streamed._System',forbidden)
    monkeypatch.setattr(torch,'full',forbidden)
    output=tmp_path/'plan.json'
    main(['--output',str(output),'--scratch',str(tmp_path/'scratch')])
    record=json.loads(output.read_text())
    assert record['stage']=='admitted_not_executed'
    assert record['precision']=='float32' and not record['complex_fields']
    assert record['eh_bytes']==54*gib and record['physical_vram_exceeded']
    assert record['reservation']['host_reservation_bytes']<=88*gib
    assert record['reservation']['disk_reservation_bytes']<=280*gib
    assert record['reservation']['gpu_reservation_bytes']<=32*gib
    assert not list((tmp_path/'scratch').iterdir())
