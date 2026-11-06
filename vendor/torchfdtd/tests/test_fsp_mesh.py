"""Synthetic saved grids only, no vendor layout or calculation data."""
import numpy as np
import pytest
from torchfdtd.fsp_binary import FspDocument
from torchfdtd.fsp_native import convert_fsp
from test_fsp_native import fixture


def settings(mode=0,periodic=False):
    # Independent, centered tensor mesh. Its four PML cells touch ±2 um.
    inner=np.r_[np.linspace(-2,-.5,11),np.linspace(-.5,.5,21)[1:],np.linspace(.5,2,11)[1:]]
    grid=np.r_[[-2.6,-2.45,-2.3,-2.15],inner,[2.15,2.3,2.45,2.6]]*1e-6
    data=dict(customGrid=mode,dt=.8*.05e-6/(299792458*np.sqrt(3)),
              xGrid=grid,yGrid=grid,zGrid=grid)
    if periodic:
        data.update(xGrid=np.r_[inner[0]-(inner[-1]-inner[-2]),inner,inner[-1]+(inner[1]-inner[0])]*1e-6,
                    BCType0=1,BCType1=1)
    return data


@pytest.mark.parametrize('mode',[0,1])
@pytest.mark.parametrize('periodic',[False,True])
def test_nonuniform_saved_nodes_and_dt_are_retained_without_regeneration(mode,periodic):
    fields=settings(mode,periodic);raw=fixture(region_overrides=fields);doc=FspDocument(raw)
    report=convert_fsp(doc);assert report.project is not None,report.issues
    r=report.project.region;assert r.mesh_type=='explicit' and r.material_sampling=='yee'
    for a,axis in enumerate('xyz'):
        want=fields[axis+'Grid'][1:-1] if periodic and a==0 else fields[axis+'Grid']
        np.testing.assert_allclose(r.mesh_nodes[a],want*1e6,rtol=0,atol=1e-15)
        assert r.interior_bounds(a)==pytest.approx((-2,2))
    assert r.time_step==fields['dt'] and doc.data==raw
    old=[a.copy() for a in r.mesh_nodes];report.project.structures[0].radius*=2
    for a,b in zip(r.mesh_nodes,old):np.testing.assert_array_equal(a,b)
    assert any(i['code']=='frozen_rectilinear_mesh' for i in report.issues)


def test_uniform_anisotropic_saved_grid_and_conservative_smaller_dt():
    spacing=np.array([.1,.125,.2])*1e-6
    data=dict(customGrid=2,dt=.7/(299792458*np.sqrt(np.sum(1/spacing**2)))/2)
    for a,h in zip('xyz',spacing):
        data['d'+a]=h;count=round(4e-6/h)+8;data[a+'Grid']=np.linspace(-2e-6-4*h,2e-6+4*h,count+1)
    report=convert_fsp(FspDocument(fixture(region_overrides=data)))
    assert report.project is not None,report.issues
    assert report.project.region.shape==(48,40,28)
    assert report.project.region.time_step==data['dt']


@pytest.mark.parametrize('change',[dict(customGrid=3),dict(meshRefineDesired=0),dict(dt=1e-14),
    dict(GUIwidth=3.9e-6),dict(xGrid=np.arange(50)[::-1]*1e-6)])
def test_unverified_or_stale_mesh_settings_do_not_create_partial_scenes(change):
    report=convert_fsp(FspDocument(fixture(region_overrides=settings()|change)))
    assert report.project is None and any(i['severity']=='error' for i in report.issues)
