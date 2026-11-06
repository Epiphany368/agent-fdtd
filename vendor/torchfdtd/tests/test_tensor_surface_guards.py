"""Tensor surface guards, without commercial runtimes or GPU execution."""
from types import SimpleNamespace
import pytest
import torch
from torchfdtd.models import Material
from torchfdtd.session import FDTD
from torchfdtd.endpoint_project import endpoint_from_project
from torchfdtd.tensor_project import TensorProject


def test_tensor_session_edit_is_explicit_and_transactional():
    from test_endpoint_project import project
    p=project();p.materials.append(Material(name='unused tensor',model='tensor'))
    editor=FDTD(p);before=p.model_dump_json()
    with pytest.raises(ValueError,match='Unsupported tensor'):editor.setmaterial('unused tensor','Permittivity',3.)
    assert p.model_dump_json()==before
    with pytest.raises(ValueError,match='eigenvalues'):editor.setmaterial('unused tensor','Permittivity tensor',(1,1,1,2,0,0))
    assert p.model_dump_json()==before
    value=(2.4,2.6,2.8,.1,.2,.15)
    editor.setmaterial('unused tensor','Permittivity tensor',value)
    assert p.materials[-1].epsilon_tensor==value


def test_inactive_tensor_does_not_break_closed_pmc_rasterization():
    from test_endpoint_project import project
    p=project();expected=endpoint_from_project(p,boundary_faces=[('pmc','pmc')]*3).rasterize()
    p.materials.append(Material(name='unused tensor',model='tensor'))
    adapter=endpoint_from_project(p,boundary_faces=[('pmc','pmc')]*3)
    torch.testing.assert_close(adapter.rasterize(),expected)
    assert torch.isfinite(adapter().signals).all()


def test_fsp_reports_unused_tensor_without_rewriting_file():
    from test_fsp_geometry_write import imported,shape_fixture
    from torchfdtd.fsp_geometry import write_fsp_geometry
    document,p=imported(shape_fixture('rectangle'))
    p.materials.append(Material(name='unused tensor',model='tensor'))
    output,report=write_fsp_geometry(document,p)
    assert output.data==document.data
    assert any('unused tensor' in item and 'Cartesian tensor' in item for item in report['native_only_settings'])
    p.structures[0].material='unused tensor'
    with pytest.raises(ValueError):write_fsp_geometry(document,p)


def test_gpu_reservation_reporting_includes_adapter_without_gpu(monkeypatch):
    import torchfdtd.memory_profile as memory
    import torchfdtd.cuda_memory as cuda_memory
    monkeypatch.setattr(memory,'host_memory',lambda:{'available_bytes':100000})
    monkeypatch.setattr(cuda_memory,'cuda_budget_limit',lambda *args:100000)
    adapter=TensorProject.__new__(TensorProject)
    snapshot=SimpleNamespace(model_dump_json=lambda:'fixed')
    adapter.project=snapshot;adapter._fingerprint='fixed'
    adapter.device=torch.device('cuda:0');adapter.adapter_bytes=200;adapter.tensor_budget_bytes=2000
    adapter.simulation=SimpleNamespace(project=snapshot,reservation=lambda **kw:dict(host_reservation_bytes=100,memory_reservation_bytes=1000,gpu_reservation_bytes=1000))
    plan=adapter._admit()
    assert plan['gpu_reservation_bytes']==plan['memory_reservation_bytes']==1200
    assert plan['host_reservation_bytes']==300
