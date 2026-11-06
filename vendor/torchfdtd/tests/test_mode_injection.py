"""Modal source contracts and actual native propagation acceptance suite."""
import json
import numpy as np
import pytest
import torch
from torchfdtd.mode_injection import prepare_modal_launch,ModeInjectedPlaneSimulation
from torchfdtd import AdjointOptions
from benchmarks.mode_injection import make_case,run_physical_validation


def test_discrete_dispersion_and_staggered_modal_source_terms():
    p,epsilon=make_case(steps=12)
    launch=prepare_modal_launch(p,epsilon)
    h=launch.normal_step_um
    assert np.isclose(2*np.sin(launch.mode.beta_per_um*h/2)/h,launch.beta_tilde_per_um,rtol=1e-7)
    assert np.isclose(launch.beta_tilde_per_um,1.5*launch.temporal_k_per_um,rtol=2e-5)
    assert len(launch.terms)==8
    electric=[t for t in launch.terms if t[0][0]=='E']
    magnetic=[t for t in launch.terms if t[0][0]=='H']
    assert {t[1][0].start for t in electric}=={launch.electric_index}
    assert {t[1][0].start for t in magnetic}=={launch.electric_index-1}
    with pytest.raises(ValueError,match='E-node'):
        launch.detector_mode(.03)


def test_modal_source_unsupported_contracts_and_preallocation_budget():
    p,epsilon=make_case(steps=12)
    with pytest.raises(ValueError,match='byte budget'):
        prepare_modal_launch(p,epsilon,source_budget_bytes=1)
    p.sources[0].pulse='continuous'
    with pytest.raises(ValueError,match='Gaussian'):
        prepare_modal_launch(p,epsilon)
    p,epsilon=make_case(steps=12)
    p.region.material_sampling='cell'
    with pytest.raises(ValueError,match='Yee-sampled'):
        prepare_modal_launch(p,epsilon)


def test_launch_material_mismatch_and_changed_region_are_rejected():
    p,epsilon=make_case(steps=12)
    launch=prepare_modal_launch(p,epsilon)
    model=ModeInjectedPlaneSimulation(p,launch,AdjointOptions(checkpoints=1))
    base=torch.tensor(np.broadcast_to(launch.epsilon[None],p.region.shape+(3,)).copy())
    base[launch.electric_index,0,0,0]+=.1
    with pytest.raises(ValueError,match='Injection-neighborhood'):
        model(base,[299792458/1.55e-6])
    model.model.project.region.steps=13
    with pytest.raises(ValueError,match='configuration changed'):
        model(base,[299792458/1.55e-6])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Physical acceptance requires local CUDA')
def test_physical_native_modal_propagation_scattering_and_gradient():
    pytest.importorskip('cupy')
    report=run_physical_validation('cuda')
    assert len(report['cases'])==5


def test_modal_host_budget_includes_extra_before_material_allocation(monkeypatch):
    from dataclasses import replace
    from torchfdtd.adjoint_memory import _resident_reservation
    import torchfdtd.mode_injection as injection
    p,epsilon=make_case(steps=12)
    launch=prepare_modal_launch(p,epsilon)
    model=ModeInjectedPlaneSimulation(p,launch,AdjointOptions(checkpoints=1)).model
    material=torch.tensor(np.broadcast_to(launch.epsilon[None],p.region.shape+(3,)).copy(),requires_grad=True)
    base=_resident_reservation(model.project,model.options,material.device,None)
    extra=2*launch.storage_bytes+material.numel()*material.element_size()
    limit=base['host_reservation_bytes']+extra-1
    model.options=replace(model.options,host_budget_bytes=limit)
    # The existing base reservation accepts this limit. Only the modal packet
    # and gradient-freezing carrier make the full reservation exceed it.
    _resident_reservation(model.project,model.options,material.device,None)
    def forbidden(*args,**kwargs):
        raise AssertionError('Material allocation occurred before host-budget rejection.')
    monkeypatch.setattr(injection,'_profile',forbidden)
    monkeypatch.setattr(torch.Tensor,'clone',forbidden)
    with pytest.raises(ValueError,match='host byte budget'):
        model._run(material,None)
