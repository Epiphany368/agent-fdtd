"""PEC facade, JSON validation, export and execution surface."""
import time
import pytest
from fastapi.testclient import TestClient
from torchfdtd import FDTD, Project, Source, Monitor
from torchfdtd.server import create_app
from test_pec_boundaries import scene


def test_familiar_pec_aliases_preserve_independent_faces(tmp_path):
    f=FDTD()
    f.set("x min bc","PEC")
    assert [x.kind for x in f.project.region.boundaries.pair(0)]==["pec","pml"]
    f.set("x max bc","Anti-Symmetric")
    assert [x.kind for x in f.project.region.boundaries.pair(0)]==["pec","antisymmetric"]
    f.set("x min bc","PML")
    assert [x.kind for x in f.project.region.boundaries.pair(0)]==["pml","antisymmetric"]
    f.set("x min bc","Bloch");f.set("bloch phase x",.3)
    f.set("x max bc","PEC")
    assert [x.kind for x in f.project.region.boundaries.pair(0)]==["pec","pec"]
    assert f.project.region.bloch_phase[0]==0
    f.save(tmp_path/"walls.json")
    assert Project.load(tmp_path/"walls.json").region.boundaries.x_max.kind=="pec"
    f.set("x max bc","PMC")
    assert f.project.region.boundaries.x_max.kind=="pmc"
    f.set("x max bc","Symmetric")
    assert f.project.region.boundaries.x_max.kind=="symmetric"
    # The editing facade permits sequential region edits. Save/run must still
    # reject this unsupported two-dimensional PMC configuration before execution.
    with pytest.raises(ValueError,match="3D region"):
        f.save(tmp_path/"unsupported.json")


def test_pec_api_validates_exports_and_runs(tmp_path):
    p=scene(dimension="2d")
    p.region.boundaries.x_max.kind="antisymmetric"
    p.region.backend="cpu"
    p.sources=[Source(center=(.3,0,0),pulse="continuous",wavelength=1.)]
    p.monitors=[Monitor(center=(.4,0,0))]
    with TestClient(create_app(tmp_path)) as client:
        payload=p.model_dump()
        assert client.post("/api/validate",json=payload).status_code==200
        exported=client.post("/api/python",json=payload)
        assert exported.status_code==200
        assert "'kind': 'pec'" in exported.text and "'kind': 'antisymmetric'" in exported.text
        response=client.post("/api/jobs",json=payload)
        assert response.status_code==202,response.text
        key=response.json()["id"]
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            job=client.get("/api/jobs/"+key).json()
            if job["status"] in ("completed","failed"):break
            time.sleep(.02)
        assert job["status"]=="completed",job
        assert job["summary"]["boundaries"]["x_min"]["kind"]=="pec"
        assert job["summary"]["boundaries"]["x_max"]["kind"]=="antisymmetric"
        assert any(value != 0 for value in job["monitors"][0]["signal"])
        payload["region"]["boundaries"]["x_max"]["kind"]="pmc"
        rejected=client.post("/api/validate",json=payload)
        assert rejected.status_code==422
        assert "periodic or Bloch mixing" in rejected.text


def test_familiar_pmc_facade_runs_and_preserves_endpoint_results(tmp_path):
    from torchfdtd.models import demo_project
    f=FDTD(demo_project('pmc'))
    f.set('x min bc','PEC')
    f.set('y max bc','Symmetry')
    f.set('z min bc','PMC')
    f.save(tmp_path/'closed.json')
    f.load(tmp_path/'closed.json')
    assert f.project.region.boundaries.y_max.kind=='symmetric'
    result=f.run()
    assert result.summary['backend']=='cpu'
    assert result.endpoint_fields is not None
    assert result.summary['steps']==160 and abs(result.signals).max()>0
    with pytest.raises(RuntimeError,match='switchtolayout'):
        f.set('z max bc','PEC')
    f.switchtolayout()
    f.set('z max bc','PML')
    # The endpoint forward enforces its CPML contract at run time.
    with pytest.raises(ValueError,match='alpha=0'):
        f.run()
