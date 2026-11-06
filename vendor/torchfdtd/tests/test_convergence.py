import json
import threading

import numpy as np
import pytest

from torchfdtd import mesh_convergence, mesh_refinement_projects, normalize_flux
from test_solver import small
from test_field_monitors import slab_project


def test_planner_keeps_physical_duration_domain_pml_and_inputs():
    p=small();original=p.model_dump()
    projects=mesh_refinement_projects(p,[.1,.05,.025])
    duration=p.region.steps*p.region.time_step
    assert [q.region.steps for q in projects]==[180,360,720]
    assert p.model_dump()==original
    for q in projects:
        assert q.region.actual_size==p.region.actual_size
        assert q.region.pml_layers(0,0)*q.region.mesh==pytest.approx(.5)
        assert q.region.steps*q.region.time_step==pytest.approx(duration)
        assert q.sources==p.sources and q.structures==p.structures
        assert not q.region.run_control.auto_shutoff
    with pytest.raises(ValueError,match='divisible'):
        mesh_refinement_projects(p,[.12,.06,.03])
    with pytest.raises(ValueError,match='decreasing'):
        mesh_refinement_projects(p,[.1,.1])
    with pytest.raises(ValueError,match='ten steps'):
        mesh_refinement_projects(p,[.1,.05],duration_s=1e-18)


def test_real_slab_spectrum_convergence_and_analytic_error(tmp_path):
    p=slab_project();p.region.size=(8,.8,1);p.region.steps=1600
    p.materials[1].index=1.5;p.structures[0].center=(.013,0,0)
    p.structures[0].size=(.2,.8,1);p.sources[0].size=(0,.8,0)
    for m in p.monitors:m.size=(0,.8,1)
    reference=p.model_copy(deep=True);reference.structures=[]
    def transmission(sample,air):
        spec=normalize_flux(sample.field_monitor('transmission'),air.field_monitor('transmission'))
        assert spec['valid'].all()
        return spec['ratio']
    report=mesh_convergence(p,[.05,.025,.0125],transmission,reference_project=reference,
                            atol=.006,rtol=0,output_dir=tmp_path)
    assert report.status=='converged'
    assert len(report.levels)==3
    wavelength=np.linspace(299792458/1.8e-6,299792458/1.3e-6,31)
    wavelength=299792458/wavelength*1e6
    analytic=1/(1+((1.5**2-1)/3)**2*np.sin(2*np.pi*1.5*.2/wavelength)**2)
    errors=[np.max(abs(np.array(level['values']['observable'])-analytic)) for level in report.levels]
    assert errors[-1]<errors[0]
    assert errors[-1]<.001
    assert json.loads((tmp_path/'convergence.json').read_text())['status']=='converged'
    assert (tmp_path/'level-02-reference.npz').is_file()


def test_invalid_observables_and_cancellation_do_not_report_success():
    p=small();p.region.steps=10;p.structures=[]
    with pytest.raises(ValueError,match='finite'):
        mesh_convergence(p,[.1,.05],lambda result:float('nan'),consecutive=1)
    with pytest.raises(ValueError,match='names and array shapes'):
        mesh_convergence(p,[.1,.05],lambda result:result.signals,consecutive=1)
    with pytest.raises(ValueError,match='real observables'):
        mesh_convergence(p,[.1,.05],lambda result:1j,consecutive=1)
    event=threading.Event();event.set()
    report=mesh_convergence(p,[.1,.05],lambda result:1.,consecutive=1,cancel=event)
    assert report.status=='cancelled' and not report.levels


def test_strict_tolerance_reports_nonconvergence():
    p=small();p.region.steps=30
    report=mesh_convergence(p,[.1,.05],lambda result:np.max(abs(result.electric)),
                            consecutive=1,rtol=0,atol=1e-15)
    assert report.status=='not_converged'
    assert not report.levels[-1]['comparison']['observable']['passed']


def test_graded_reference_uses_frozen_sample_nodes():
    p=slab_project();p.region.steps=80;p.region.size=(8,.8,1)
    p.sources[0].size=(0,.8,0)
    for monitor in p.monitors:monitor.size=(0,.8,1)
    ref=p.model_copy(deep=True);ref.structures=[]
    def matching(sample,reference):
        for a,b in zip(sample.project.region.mesh_nodes,reference.project.region.mesh_nodes):
            np.testing.assert_array_equal(a,b)
        assert sample.frequency_fields[0]['run_signature']==reference.frequency_fields[0]['run_signature']
        return float(np.max(abs(sample.electric)))
    report=mesh_convergence(p,[.05,.025],matching,reference_project=ref,mesh_type='graded',consecutive=1)
    assert len(report.levels)==2
