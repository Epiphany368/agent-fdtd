"""Self-authored layout records for the independently decoded ellipse subset."""
import struct
import numpy as np
import pytest
from torchfdtd.fsp_binary import FspDocument
from torchfdtd.fsp_native import convert_fsp,ZERO_UUID
from torchfdtd.solver import voxelize
from torchfdtd.geometry import contains
from test_fsp_native import fixture,items
from test_fsp_binary import u,string,mapping


def ellipse_fixture(kind='sphere',radius2=.21e-6,radius3=.13e-6):
    original=fixture();doc=FspDocument(original);old=next(n for n in doc.nodes() if n.legacy)
    sphere=kind=='sphere';code=8 if sphere else 4
    uid='{23046316-141b-4111-aa2f-9e18de790b6c}' if sphere else '{921e6d99-3bcb-4063-b513-216a3bb757d9}'
    fields=(.3e-6,0.,0.,0.) if sphere else (.3e-6,0.,0.,-.2e-6,.2e-6)
    tail=bytearray(167 if sphere else 158)
    radii=b'\0'+u(1)+b'\1'+struct.pack('<d',radius2)+(b'\1'+struct.pack('<d',radius3) if sphere else b'')
    tail[:len(radii)]=radii;n=len(radii)
    tail[n:n+15]=(b'\0'+struct.pack('<i',-1))*3
    tail[n+15:n+68]=u(3)+b'\0'+u(3)+u(2)+u(3)+u(1)+u(1)+u(1)+bytes(24)
    tail[-56:-52]=u(1);tail[-51:-47]=u(2);tail[-9:-5]=u(1)
    body=u(code)+u(25)+struct.pack('<i'+'d'*(len(fields)+1),-1,2.,*fields)+string('2')+bytes(8)+string(kind)
    body+=tail+mapping(items(dict(materialuuid=ZERO_UUID,use_relative_coordinates=1,gridAttributeName='')))+u(0)
    replacement=u(1000)+u(38)+uid.encode('ascii')+body
    return original[:old.start]+replacement+original[old.end:]


@pytest.mark.parametrize('kind',['sphere','circle'])
def test_unrotated_ellipse_radii_and_original_bytes_preserved(kind,monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp,'load_api',lambda:pytest.fail('Vendor runtime accessed'))
    raw=ellipse_fixture(kind);doc=FspDocument(raw);report=convert_fsp(doc,backend='cpu')
    assert report.project is not None,report.issues
    p=report.project;s=p.structures[0]
    assert s.make_ellipsoid and s.radius_2==pytest.approx(.21)
    if kind=='sphere':assert s.radius_3==pytest.approx(.13)
    eps,counts=voxelize(p);assert counts[s.id]>0
    axes=[(np.arange(n)+.5)*p.region.mesh-span/2 for n,span in zip(p.region.shape,p.region.actual_size)]
    x,y,z=np.meshgrid(*axes,indexing='ij',sparse=True)
    expected=(x/.3)**2+(y/.21)**2
    expected=expected+(z/.13)**2<=1 if kind=='sphere' else (expected<=1)&(abs(z)<=.2)
    np.testing.assert_array_equal(eps==4,expected)
    assert doc.data==raw and doc.fingerprint()==p.import_provenance.source_sha256


def test_ellipse_actual_support_rejects_pml_intersection_and_invalid_radius():
    for radius in (2.2e-6,0.,-1e-6):
        report=convert_fsp(FspDocument(ellipse_fixture(radius2=radius)))
        assert report.project is None
        assert any(i['severity']=='error' for i in report.issues)
