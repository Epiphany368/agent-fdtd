import pytest
from torchfdtd import FDTD, Project


def test_familiar_commands_use_si_and_shared_project(tmp_path):
    f=FDTD()
    f.addrect(name='core',x_span=4e-6,y_span=.5e-6,z_span=.4e-6,index=2)
    f.adddipole(name='source',x=-1e-6,wavelength=1.55e-6)
    f.addtime(name='out',x=1e-6)
    f.setnamed('FDTD','backend','cpu');f.setnamed('FDTD','time steps',100)
    f.save(tmp_path/'scene.json');p=Project.load(tmp_path/'scene.json')
    assert p.structures[0].size==pytest.approx((4,.5,.4))
    result=f.run();assert result.summary['steps']==100
    assert f.getresult('out')['time_fs']
    with pytest.raises(RuntimeError):f.addrect()
    f.switchtolayout();f.select('core');f.delete()
    assert not f.project.structures
    with pytest.raises(ValueError):f.set('silently unsupported',True)


def test_familiar_boundary_commands_pair_cyclic_faces():
    f=FDTD()
    f.set('y min bc','Bloch')
    f.set('bloch phase y',.7)
    assert all(face.kind=='bloch' for face in f.project.region.boundaries.pair(1))
    assert f.project.region.bloch_phase[1]==.7
    f.set('y max bc','Periodic')
    assert all(face.kind=='periodic' for face in f.project.region.boundaries.pair(1))
    assert f.project.region.bloch_phase[1]==0


def test_plane_monitor_and_global_settings_python_only(tmp_path):
    import numpy as np
    from torchfdtd import Result
    f=FDTD()
    f.set('backend','cpu');f.set('time steps',100)
    f.adddipole(name='source')
    f.setglobalmonitor('custom frequencies',[180e12,200e12])
    f.addpower(name='flux',x=1e-6,y_span=.8e-6)
    f.set('use global monitor settings',True)
    result=f.run()
    data=f.getresult('flux','all')
    assert data['fields'].shape[0]==2
    np.testing.assert_array_equal(f.getglobalmonitor('custom frequencies'),[180e12,200e12])
    result.save(tmp_path/'fields.npz')
    loaded=Result.load(tmp_path/'fields.npz')
    np.testing.assert_array_equal(loaded.field_monitor('flux')['fields'],data['fields'])
    assert f.getresult('flux','E')['E'].shape[-1]==3
    with pytest.raises(ValueError):f.getresult('flux','T')
