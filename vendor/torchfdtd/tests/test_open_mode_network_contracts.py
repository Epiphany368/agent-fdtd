"""Open-port configuration and replay admission without any FDTD execution."""
from copy import copy
from dataclasses import FrozenInstanceError, replace
import numpy as np
import pytest
from torchfdtd import Project, Region, Source, AdjointOptions
from torchfdtd.mode_network import FixedModePort, ModeNetwork
from torchfdtd.open_mode_injection import OpenPortOptions, prepare_open_modal_launch
from torchfdtd.open_mode_ports import solve_open_waveguide_modes


def scene():
    faces={a+'_'+s:dict(kind='pml',layers=8 if a=='x' else 3)
           for a in 'xyz' for s in ('min','max')}
    for s in ('min','max'): faces['y_'+s]=dict(kind='periodic')
    r=Region(dimension='3d',size=(6.4,.5,1.6),mesh=.1,steps=10,
        boundaries=faces,material_sampling='yee',precision='float32',backend='cpu')
    p=Project(region=r,sources=[Source(kind='plane',normal='z',center=(0,0,-.3),
        size=(4.8,.5,0),wavelength=1.55,pulse_cycles=2)],monitors=[])
    ports=(FixedModePort('left',-.2,-.3,1,(0,)),FixedModePort('right',.2,.3,-1,(0,)))
    return p,ports


def material(x,y): return np.where(np.abs(x)<.4-1e-8,1.8**2,1.4**2)


def forbidden(*args,**kwargs): raise AssertionError('unexpected material/field allocation')


@pytest.fixture(scope='module')
def packet_network():
    p,ports=scene()
    # Exactly one real eigensolve is reused by both incident-port packets.
    mode=solve_open_waveguide_modes(material,region=p.region,normal='z',
        wavelength_um=1.55,cladding_epsilon=1.4**2)[0]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr('torchfdtd.open_mode_ports.solve_open_waveguide_modes',lambda *a,**k:(mode,))
        patch.setattr('torchfdtd.differentiable._System.__init__',forbidden)
        network=ModeNetwork(p,ports,material,AdjointOptions(checkpoints=1),num_modes=1,
                            open_ports=OpenPortOptions(1.4**2))
    return network


@pytest.mark.parametrize('kwargs,reason',[
    ({'open_ports':{}},'explicit OpenPortOptions'),
    ({'open_ports':OpenPortOptions(1.96),'network_budget_bytes':1},'network byte budget'),
    ({'open_ports':OpenPortOptions(1.96),'port_permittivities':{'left':material,'right':material}},'either permittivity'),
])
def test_network_rejections_precede_material_and_fields(monkeypatch,kwargs,reason):
    p,ports=scene()
    monkeypatch.setattr('torchfdtd.open_mode_ports.solve_open_waveguide_modes',forbidden)
    monkeypatch.setattr('torchfdtd.differentiable._System.__init__',forbidden)
    with pytest.raises(ValueError,match=reason): ModeNetwork(p,ports,forbidden,num_modes=1,**kwargs)


def test_source_budget_precedes_mode_or_field_allocation(monkeypatch):
    p,_=scene()
    monkeypatch.setattr('torchfdtd.open_mode_ports.solve_open_waveguide_modes',forbidden)
    with pytest.raises(ValueError,match='source byte budget'):
        prepare_open_modal_launch(p,forbidden,options=OpenPortOptions(1.96),source_budget_bytes=1)


def test_network_live_host_guard_precedes_sampling(monkeypatch):
    p,ports=scene()
    monkeypatch.setattr('torchfdtd.memory_profile.host_memory',lambda:dict(available_bytes=1))
    monkeypatch.setattr('torchfdtd.open_mode_ports.solve_open_waveguide_modes',forbidden)
    with pytest.raises(ValueError,match='available host memory'):
        ModeNetwork(p,ports,forbidden,num_modes=1,open_ports=OpenPortOptions(1.96))


def test_true_immutable_launch_payload_alias_and_packet_accounting(packet_network):
    network=packet_network
    for launch in network._launches:
        assert launch.epsilon is launch.mode.sampled_epsilon
        arrays=[launch.epsilon,launch.mode.fields,launch.detector_mode(-.2).fields]
        arrays += [a for _,_,wave,profile in launch.terms for a in (wave,profile)]
        for array in arrays:
            assert not array.flags.writeable
            with pytest.raises(ValueError): array.setflags(write=True)
        actual=launch.epsilon.nbytes+launch.mode.fields.nbytes+sum(
            wave.nbytes+profile.nbytes for _,_,wave,profile in launch.terms)
        assert launch.storage_bytes==actual
        assert len(launch.terms)==8
        assert all(wave.dtype==np.float32 and profile.dtype==np.float32
                   for _,_,wave,profile in launch.terms)
        with pytest.raises(FrozenInstanceError): launch.identity='changed'
    assert network._packet_bytes==sum(item.storage_bytes for item in network._launches)
    assert network._admit()>network._packet_bytes


@pytest.mark.parametrize('mutation',['project','prepared_source','options','ports'])
def test_recomputed_configuration_guard_before_reference_allocation(packet_network,monkeypatch,mutation):
    network=copy(packet_network)
    if mutation=='project':
        network.project=network.project.model_copy(deep=True);network.project.region.steps+=1
    elif mutation=='prepared_source':
        network._projects=tuple(p.model_copy(deep=True) for p in network._projects)
        network._projects[0].sources[0].phase+=1
    elif mutation=='options': network.open_ports=replace(network.open_ports,cladding_epsilon=2.)
    else: network.ports=(replace(network.ports[0],coordinate_um=-.1),network.ports[1])
    monkeypatch.setattr('torchfdtd.mode_network.torch.tensor',forbidden)
    with pytest.raises(ValueError,match='changed'): network.reference_epsilon()
    # Same guard is the first entry point for recomputed network calls.
    with pytest.raises(ValueError,match='changed'): network(None)


def test_output_admission_precedes_calibration_and_material(packet_network,monkeypatch):
    monkeypatch.setattr('torchfdtd.mode_network.ModeInjectedPlaneSimulation',forbidden)
    monkeypatch.setattr('torchfdtd.mode_network.torch.tensor',forbidden)
    with pytest.raises(ValueError,match='output byte budget before calibration'):
        packet_network(None,output_budget_bytes=1)


@pytest.mark.parametrize('which',['region','source'])
def test_launch_replay_signature_guards_precede_system_or_material(packet_network,monkeypatch,which):
    from torchfdtd.mode_injection import _ModalSimulation
    model=object.__new__(_ModalSimulation)
    model.project=packet_network._projects[0].model_copy(deep=True)
    model.launch=packet_network._launches[0]
    if which=='region': model.project.region.steps+=1
    else: model.project.sources[0].phase+=1
    monkeypatch.setattr('torchfdtd.mode_injection._System',forbidden)
    with pytest.raises(ValueError,match='Modal launch '+which+' changed'):
        model._run(None,None)
