"""Synthetic GDS fixtures are generated independently with gdstk primitives."""
import json
import math
import struct
import tempfile
from pathlib import Path
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

gdstk=pytest.importorskip('gdstk')
from torchfdtd.gds import GDSLayer,GDSPortLayer,GDSLimits,import_gds,export_gds
from torchfdtd import Project,Region,Material,Source,Monitor,Simulation,Structure
from torchfdtd.solver import voxelize,index_at


def write(tmp_path,lib,name='layout.gds'):
    path=tmp_path/name
    with tempfile.TemporaryDirectory(prefix='.synthetic-gds-',dir='.') as directory:
        temporary=Path(directory)/'independent.gds'
        lib.write_gds(temporary,max_points=0,timestamp=datetime(2020,1,1))
        path.write_bytes(temporary.read_bytes())
    return path


def rectangle_file(tmp_path):
    lib=gdstk.Library();cell=lib.new_cell('TOP')
    cell.add(gdstk.rectangle((-.5,-.3),(.5,.3),layer=1,datatype=2))
    return write(tmp_path,lib)


def world_points(obj):return np.asarray(obj.vertices)+obj.center[:2]


def test_units_nested_transform_array_and_stack(tmp_path):
    lib=gdstk.Library(unit=1e-9,precision=1e-12)
    child=lib.new_cell('CHILD');child.add(gdstk.rectangle((0,0),(1000,500),layer=7,datatype=4))
    middle=lib.new_cell('MIDDLE')
    middle.add(gdstk.Reference(child,origin=(2000,3000),rotation=math.pi/2,magnification=2,x_reflection=True,
                               columns=2,rows=2,spacing=(4000,3000)))
    top=lib.new_cell('TOP');top.add(gdstk.Reference(middle,origin=(-1000,2000),rotation=math.pi))
    path=write(tmp_path,lib)
    layer=GDSLayer(7,4,-.2,.4,'core')
    result=import_gds(path,cell='TOP',layers=[layer,replace(layer,z_min=.8,z_max=1.,material='cladding')])
    assert len(result.structures)==8
    # Reference array spacing rotates/reflects but is not multiplied by MAG.
    expected=[]
    for col in range(2):
        for row in range(2):
            lower=np.array([2+3*row,3+4*col]);upper=lower+[1,2]
            expected.append((tuple(np.array([-1,2])-upper),tuple(np.array([-1,2])-lower)))
    actual=[]
    for obj in result.structures:
        if obj.material=='core':actual.append((tuple(world_points(obj).min(0)),tuple(world_points(obj).max(0))))
    np.testing.assert_allclose(np.array(sorted(actual)),np.array(sorted(expected)),atol=1e-9)
    assert result.report['file_unit_m']==pytest.approx(1e-9)
    assert result.report['visited_cells']==['CHILD','MIDDLE','TOP']
    assert result.report==import_gds(path,cell='TOP',layers=[layer,replace(layer,z_min=.8,z_max=1.,material='cladding')]).report
    json.dumps(result.report,allow_nan=False)


def test_real_gds_path_and_unmapped_layer_admission(tmp_path):
    lib=gdstk.Library();top=lib.new_cell('TOP')
    top.add(gdstk.FlexPath([(-1,0),(1,0)],.2,simple_path=True,layer=2,datatype=3))
    top.add(gdstk.rectangle((2,2),(3,3),layer=99))
    path=write(tmp_path,lib)
    with pytest.raises(ValueError,match='Unmapped'):import_gds(path,cell='TOP',layers=[GDSLayer(2,3,0,.2,'core')])
    result=import_gds(path,cell='TOP',layers=[GDSLayer(2,3,0,.2,'core')],unmapped='report')
    assert result.report['expanded_path_count']==1
    assert result.report['ignored_geometry']==[dict(layer=99,datatype=0,polygons=1)]
    np.testing.assert_allclose(world_points(result.structures[0]).min(0),[-1,-.1],atol=1e-8)
    np.testing.assert_allclose(world_points(result.structures[0]).max(0),[1,.1],atol=1e-8)


def test_port_texttype_nested_transform_and_repeated_names(tmp_path):
    lib=gdstk.Library();child=lib.new_cell('CHILD')
    child.add(gdstk.rectangle((0,0),(1,1),layer=1),gdstk.Label('in',(1,2),rotation=math.pi/4,layer=10,texttype=5))
    middle=lib.new_cell('MIDDLE');middle.add(gdstk.Reference(child,origin=(2,3),rotation=math.pi/2,magnification=2,x_reflection=True))
    top=lib.new_cell('TOP');top.add(gdstk.Reference(middle,origin=(-1,2),rotation=math.pi,magnification=.5))
    path=write(tmp_path,lib)
    ports=[GDSPortLayer(10,5,-.1,.3,.6,(1.,0.))]
    result=import_gds(path,cell='TOP',layers=[GDSLayer(1,0,0,.2,'core')],port_layers=ports)
    port=result.ports[0]
    # Two independent affine transforms: child(1,2) -> middle(6,5) -> top(-4,-.5).
    np.testing.assert_allclose(port.center_um,[-4,-.5,.1],atol=1e-8)
    np.testing.assert_allclose(port.normal,[-2**-.5,-2**-.5,0],atol=1e-8)
    assert port.width_um==pytest.approx(.6)
    assert port.marker_record=='TEXT' and port.datatype==5
    top.add(gdstk.Reference(child,columns=2,rows=1,spacing=(5,0)))
    path=write(tmp_path,lib,'array.gds')
    with pytest.raises(ValueError,match='unique'):
        import_gds(path,cell='TOP',layers=[GDSLayer(1,0,0,.2,'core')],port_layers=ports)


def test_self_intersections_unsupported_records_cycles_and_bounds_are_not_silent(tmp_path):
    lib=gdstk.Library();cell=lib.new_cell('TOP')
    cell.add(gdstk.Polygon([(-2,-2),(2,2),(2,-2),(-2,2)],layer=1))
    path=write(tmp_path,lib)
    with pytest.raises(ValueError,match='simple polygon'):
        import_gds(path,cell='TOP',layers=[GDSLayer(1,0,0,.2,'core')])
    path=rectangle_file(tmp_path)
    with pytest.raises(ValueError,match='XY bounds'):
        import_gds(path,cell='TOP',layers=[GDSLayer(1,2,0,.2,'core')],xy_bounds_um=(0,0,1,1))
    with pytest.raises(ValueError,match='max_file_bytes'):
        import_gds(path,cell='TOP',layers=[GDSLayer(1,2,0,.2,'core')],limits=GDSLimits(max_file_bytes=20))
    data=path.read_bytes();bad=tmp_path/'unsupported.gds';bad.write_bytes(data[:-4]+struct.pack('>HBB',4,0x15,0)+data[-4:])
    with pytest.raises(ValueError,match='Unsupported GDS record'):
        import_gds(bad,cell='TOP',layers=[GDSLayer(1,2,0,.2,'core')])
    cyclic=gdstk.Library();a=cyclic.new_cell('A');b=cyclic.new_cell('B');a.add(gdstk.Reference(b));b.add(gdstk.Reference(a))
    path=write(tmp_path,cyclic,'cyclic.gds')
    with pytest.raises(ValueError,match='Cyclic'):
        import_gds(path,cell='A',layers=[GDSLayer(1,0,0,.2,'core')])


def test_hierarchy_budget_is_checked_before_flattening(tmp_path):
    lib=gdstk.Library();child=lib.new_cell('C');child.add(gdstk.rectangle((0,0),(1,1),layer=1))
    top=lib.new_cell('TOP');top.add(gdstk.Reference(child,columns=100,rows=100,spacing=(2,2)))
    path=write(tmp_path,lib)
    with pytest.raises(ValueError,match='hierarchy exceeds'):
        import_gds(path,cell='TOP',layers=[GDSLayer(1,0,0,.2,'core')],limits=GDSLimits(max_instances=50))


def test_imported_geometry_is_used_by_native_material_sampling_and_simulation(tmp_path):
    result=import_gds(rectangle_file(tmp_path),cell='TOP',layers=[GDSLayer(1,2,-.2,.2,'core')])
    project=Project(region=Region(size=(4,3,1),mesh=.1,pml_cells=3,steps=60,backend='cpu',precision='float32',material_sampling='yee'),
                    materials=[Material(name='core',index=2.)],sources=[Source(center=(-.8,0,0),pulse='continuous',wavelength=1.)],
                    monitors=[Monitor(center=(.8,0,0))])
    project=result.add_to(project)
    eps,counts=voxelize(project)
    assert counts[result.structures[0].id]>0
    assert eps[index_at((0,0,0),project.region,'Ez')+(2,)]==4
    actual=Simulation(project).run()
    assert np.isfinite(actual.electric).all() and np.max(np.abs(actual.signals))>0


def test_export_supported_xy_geometry_and_explicit_stack_sidecar(tmp_path):
    original=Structure(id='core',kind='polygon',vertices=((-1,-.5),(1,-.5),(1,.5),(-1,.5)),
                       center=(1,2,.3),size=(2,1,.4),rotation=90,material='core')
    path=tmp_path/'native.gds'
    report=export_gds(path,[original],layers={'core':(7,3)})
    data=path.read_bytes()
    again=export_gds(path,[original],layers={'core':(7,3)})
    assert data==path.read_bytes() and report==again
    assert report['layer_stack'][0]['z_min']==pytest.approx(.1)
    imported=import_gds(path,cell='TOP',layers=[GDSLayer(7,3,.1,.5,'core')])
    np.testing.assert_allclose(world_points(imported.structures[0]).min(0),[.5,1],atol=1e-8)
    np.testing.assert_allclose(world_points(imported.structures[0]).max(0),[1.5,3],atol=1e-8)
    tilted=original.model_copy(update={'rotation_axes':('x','none','none'),'rotation_angles':(20,0,0)})
    with pytest.raises(ValueError,match='Tilted'):export_gds(tmp_path/'bad.gds',[tilted],layers={'core':(7,3)})


def test_stack_order_unknown_material_unicode_and_limits(tmp_path):
    path=rectangle_file(tmp_path)
    unicode_path=tmp_path/'layout_µ.gds';unicode_path.write_bytes(path.read_bytes())
    layers=[GDSLayer(1,2,-.2,.2,'z_first'),GDSLayer(1,2,-.2,.2,'a_second')]
    result=import_gds(unicode_path,cell='TOP',layers=layers)
    assert [s.material for s in result.structures]==['z_first','a_second']
    with pytest.raises(ValueError,match='material'):
        result.add_to(Project())
    with pytest.raises(ValueError,match='max_structures'):
        import_gds(path,cell='TOP',layers=layers,limits=GDSLimits(max_structures=1))
    with pytest.raises(ValueError,match='existing GDS cell'):
        import_gds(path,cell='MISSING',layers=layers)
    with pytest.raises(ValueError,match='unit direction'):
        GDSPortLayer(10,0,0,.2,.5,(2,0))


def test_absolute_transform_flags_are_rejected_before_dependency_read(tmp_path):
    lib=gdstk.Library();child=lib.new_cell('C');child.add(gdstk.rectangle((0,0),(1,1),layer=1))
    top=lib.new_cell('TOP');top.add(gdstk.Reference(child,rotation=.2))
    path=write(tmp_path,lib);data=bytearray(path.read_bytes());offset=0;found=False
    while offset<len(data):
        length,kind,_=struct.unpack_from('>HBB',data,offset)
        if kind==0x1a:
            struct.pack_into('>H',data,offset+4,0x0004);found=True;break
        offset+=length
    assert found
    path.write_bytes(data)
    with pytest.raises(ValueError,match='STRANS'):
        import_gds(path,cell='TOP',layers=[GDSLayer(1,0,0,.2,'core')])


def test_export_rejects_precision_collapse_and_unmapped_geometry(tmp_path):
    tiny=Structure(id='tiny',kind='rectangle',size=(.0001,.0001,.2),material='core')
    with pytest.raises(ValueError,match='simple polygon'):
        export_gds(tmp_path/'collapsed.gds',[tiny],layers={'tiny':(1,0)},precision_m=1e-9)
    with pytest.raises(ValueError,match='explicit layer'):
        export_gds(tmp_path/'unmapped.gds',[tiny],layers={})
    sphere=Structure(id='sphere',kind='sphere')
    with pytest.raises(ValueError,match='polygons and rectangles'):
        export_gds(tmp_path/'sphere.gds',[sphere],layers={'sphere':(1,0)})


def test_empty_reference_arrays_and_repeated_stack_vertices_are_bounded(tmp_path):
    lib=gdstk.Library();empty=lib.new_cell('EMPTY');top=lib.new_cell('TOP')
    top.add(gdstk.Reference(empty,columns=100,rows=100,spacing=(1,1)))
    path=write(tmp_path,lib)
    with pytest.raises(ValueError,match='hierarchy exceeds'):
        import_gds(path,cell='TOP',layers=[GDSLayer(1,0,0,.2,'core')],limits=GDSLimits(max_instances=30))
    path=rectangle_file(tmp_path)
    with pytest.raises(ValueError,match='max_total_vertices'):
        import_gds(path,cell='TOP',layers=[GDSLayer(1,2,0,.2,'core'),GDSLayer(1,2,.3,.5,'core')],
                   limits=GDSLimits(max_total_vertices=6))


def test_magnified_absolute_width_path_keeps_physical_width(tmp_path):
    lib=gdstk.Library();child=lib.new_cell('C')
    child.add(gdstk.FlexPath([(-.5,0),(.5,0)],.2,simple_path=True,scale_width=False,layer=1))
    top=lib.new_cell('TOP');top.add(gdstk.Reference(child,origin=(1,2),rotation=math.pi/2,magnification=3))
    path=write(tmp_path,lib)
    result=import_gds(path,cell='TOP',layers=[GDSLayer(1,0,0,.2,'core')])
    points=world_points(result.structures[0])
    np.testing.assert_allclose(points.min(0),[.9,.5],atol=1e-8)
    np.testing.assert_allclose(points.max(0),[1.1,3.5],atol=1e-8)
