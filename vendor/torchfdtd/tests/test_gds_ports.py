"""Explicit metadata admission and a real CPU GDS scatterer network."""
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from torchfdtd import Project, Region, Source, Material, AdjointOptions
from torchfdtd.gds import GDSImport, GDSPort, GDSLayer, GDSPortLayer, import_gds
from torchfdtd.gds_ports import prepare_gds_mode_network


def scene():
    r = Region(dimension='3d', size=(8., 1., 1.), mesh=.2, steps=400,
               pml_cells=5, material_sampling='yee', background_index=1.5,
               boundaries={a+'_'+s: dict(kind='periodic') for a in 'yz' for s in ('min','max')})
    return Project(region=r, materials=[Material(name='slab', index=1.7)],
                   sources=[Source(kind='plane', normal='x', center=(0,0,0), size=(0,1,1), pulse_cycles=2)], monitors=[])


def markers():
    return GDSImport((), (GDSPort('west', (-1.,0.,0.), (-1.,0.,0.), 1.,1.,10,0),
                          GDSPort('east', (1.,0.,0.), (1.,0.,0.), 1.,1.,11,0)), {})


def prepare(imported, project=None, **changes):
    args = dict(port_names=('east','west'), normal_convention='outward',
                source_offsets_um={'west':1.,'east':1.},
                mode_indices={'west':(0,), 'east':(0,)}, wavelength_um=1.55,
                permittivity=2.25, options=AdjointOptions(checkpoints=2))
    args.update(changes)
    return prepare_gds_mode_network(imported, project or scene(), **args)


def test_explicit_name_mapping_without_mutating_input(monkeypatch):
    import torchfdtd.gds_ports as module
    captured = {}
    def capture(project, ports, permittivity, options, **kwargs):
        captured.update(project=project, ports=ports, permittivity=permittivity)
        return captured
    monkeypatch.setattr(module, 'ModeNetwork', capture)
    p=scene(); original=p.model_dump()
    result=prepare(markers(),p,mode_indices={'west':[1,0], 'east':[0]})
    assert [(q.name,q.coordinate_um,q.source_coordinate_um,q.direction,q.mode_indices)
            for q in result['ports']] == [('west',-1.,-2.,1,(1,0)),('east',1.,2.,-1,(0,))]
    assert p.model_dump()==original
    assert result['project'].sources[0].wavelength==1.55


def test_y_axis_inward_contract_updates_complete_source_plane(monkeypatch):
    import torchfdtd.gds_ports as module
    from torchfdtd import Boundaries
    p=scene()
    data=p.model_dump()
    data['region'].update(size=(1.,8.,1.),boundaries=Boundaries(**{
        a+'_'+s: dict(kind='periodic') for a in 'xz' for s in ('min','max')}).model_dump())
    data['sources'][0].update(normal='y',size=(1.,0.,1.))
    p=Project.model_validate(data)
    imported=markers()
    imported=replace(imported,ports=tuple(replace(q,center_um=(0.,q.center_um[0],0.),
        normal=(0.,-q.normal[0],0.)) for q in imported.ports))
    def capture(project,ports,*args,**kwargs):
        validated=Project.model_validate(project.model_dump())
        assert validated.sources[0].normal=='y'
        assert validated.sources[0].size==(1.,0.,1.)
        assert validated.sources[0].center==(0.,-2.,0.)
        return ports
    monkeypatch.setattr(module,'ModeNetwork',capture)
    ports=prepare(imported,p,normal_convention='inward')
    assert tuple(q.direction for q in ports)==(1,-1)


@pytest.mark.parametrize('change,match',[
    ({'width_um':.4},'full transverse'),
    ({'height_um':.4},'full transverse'),
    ({'center_um':(-1.,.1,0.)},'centers'),
    ({'normal':(-.8,.6,0.)},'cardinal'),
    ({'normal':(1.,0.,0.)},'opposing'),
    ({'marker_record':'BOUNDARY'},'TEXT'),
])
def test_reject_metadata_before_network_allocation(change,match,monkeypatch):
    import torchfdtd.gds_ports as module
    monkeypatch.setattr(module,'ModeNetwork',lambda *a,**kw: pytest.fail('allocated network'))
    data=markers()
    data=replace(data,ports=(replace(data.ports[0],**change),data.ports[1]))
    with pytest.raises(ValueError,match=match):prepare(data)


@pytest.mark.parametrize('changes,match',[
    ({'port_names':('west','absent')},'map exactly'),
    ({'normal_convention':'auto'},'normal_convention'),
    ({'source_offsets_um':{'west':0.,'east':1.}},'positive'),
    ({'wavelength_um':float('nan')},'wavelength'),
])
def test_explicit_selection_arguments(changes,match):
    with pytest.raises(ValueError,match=match):prepare(markers(),**changes)


def test_synthetic_gds_to_native_cpu_scatterer_and_material_vjp(tmp_path):
    gdstk=pytest.importorskip('gdstk')
    lib=gdstk.Library(); cell=lib.new_cell('TOP')
    cell.add(gdstk.rectangle((-.2,-.5),(.2,.5),layer=1))
    cell.add(gdstk.Label('west',(-1.,0.),layer=10),gdstk.Label('east',(1.,0.),layer=11))
    with tempfile.TemporaryDirectory(prefix='.gds-mode-fixture-',dir='.') as directory:
        path=Path(directory)/'layout.gds';lib.write_gds(path)
        raw=path.read_bytes()
    path=tmp_path/'layout.gds';path.write_bytes(raw)
    imported=import_gds(path,cell='TOP',layers=[GDSLayer(1,0,-.5,.5,'slab')],
        port_layers=[GDSPortLayer(10,0,-.5,.5,1.,(-1.,0.)),GDSPortLayer(11,0,-.5,.5,1.,(1.,0.))])
    network=prepare(imported)
    from torchfdtd.solver import voxelize
    epsilon_np,counts=voxelize(network.project)
    assert len(network.project.structures)==1 and sum(counts.values())>0
    epsilon=torch.from_numpy(epsilon_np)
    reference=network.reference_epsilon()
    mask=(epsilon-reference).abs()>1e-6
    assert mask.any() and not mask[:15].any() and not mask[25:].any()
    with torch.no_grad():baseline=network(reference)
    parameter=torch.tensor(0.,requires_grad=True)
    result=network(epsilon+parameter*mask)
    loss=result.s[1,0].real+.3*result.s[0,1].imag
    gradient,=torch.autograd.grad(loss,parameter)
    assert result.channels == (('west',0),('east',0))
    assert torch.isfinite(gradient) and gradient.abs()>1e-4
    assert (result.s-baseline.s).abs().max()>.05
    assert (result.s-result.s.T).abs().max()<.01
    expected=np.exp(2j*network._launches[0].mode.beta_per_um)
    assert abs(complex(baseline.s[1,0])-expected)<.01
    bad=epsilon.clone();bad[0]+=.1
    with pytest.raises(ValueError,match='exterior material'):network(bad)
    print(dict(device='cpu',slab_voxels=int(mask.sum()),gradient=float(gradient),
               scattering_change=float((result.s-baseline.s).abs().max().detach())))


def test_one_call_two_port_from_straight_waveguide(tmp_path):
    gdstk=pytest.importorskip('gdstk')
    from torchfdtd.gds_ports import GDSTwoPort, prepare_gds_two_port
    lib=gdstk.Library(); cell=lib.new_cell('TOP')
    cell.add(gdstk.rectangle((-4.,-.3),(4.,.3),layer=1))
    cell.add(gdstk.Label('west',(-1.,0.),layer=10),gdstk.Label('east',(1.,0.),layer=11))
    with tempfile.TemporaryDirectory(prefix='.gds-two-port-fixture-',dir='.') as directory:
        path=Path(directory)/'layout.gds';lib.write_gds(path)
        raw=path.read_bytes()
    path=tmp_path/'layout.gds';path.write_bytes(raw)
    imported=import_gds(path,cell='TOP',layers=[GDSLayer(1,0,-.3,.3,'slab')],
        port_layers=[GDSPortLayer(10,0,-.5,.5,1.,(-1.,0.)),GDSPortLayer(11,0,-.5,.5,1.,(1.,0.))])
    project=scene().model_copy(update={'sources':[]})
    with pytest.raises(ValueError,match='wavelength_um'):prepare_gds_two_port(imported,project)
    two_port=prepare_gds_two_port(imported,project,wavelength_um=1.55,options=AdjointOptions(checkpoints=2))
    assert isinstance(two_port,GDSTwoPort)
    network=two_port.network
    assert network.channels==(('west',0),('east',0))
    assert [(q.coordinate_um,q.source_coordinate_um) for q in network.ports]==[(-1.,-1.8),(1.,1.8)]
    assert network.project.sources[0].wavelength==1.55
    assert tuple(two_port.epsilon.shape)==project.region.shape+(3,) and two_port.epsilon.dtype==torch.float32
    # The default modal section is the imported guide itself: the straight
    # calibration guide equals the sampled runtime geometry everywhere.
    assert torch.allclose(network.reference_epsilon(),two_port.epsilon)
    assert float(two_port.epsilon.max())==pytest.approx(1.7**2,rel=1e-6)
    result=two_port.run()
    s=result.s
    assert s.shape==(2,2) and torch.isfinite(s).all()
    assert abs(complex(s[1,0]))>.5 and abs(complex(s[0,1]))>.5
    assert (s-s.T).abs().max()<.01
    expected=np.exp(2j*network._launches[0].mode.beta_per_um)
    assert abs(complex(s[1,0])-expected)<.05
    # A supplied template source and its wavelength are the defaults instead.
    defaults=prepare_gds_two_port(imported,scene(),options=AdjointOptions(checkpoints=2))
    assert defaults.network.project.sources[0].wavelength==scene().sources[0].wavelength
    with pytest.raises(ValueError,match='port_names'):
        prepare_gds_two_port(replace(imported,ports=imported.ports[:1]),scene(),wavelength_um=1.55)
