"""Authored layouts, independent scene-list and material/EM checks."""
import struct

import numpy as np
import pytest

from torchfdtd import Material, Project, Structure, Simulation, run_tensor_batch
from torchfdtd.fsp_binary import FspDocument
from torchfdtd.fsp_geometry import write_fsp_scene, write_fsp_geometry
from torchfdtd.fsp_native import convert_fsp, FDTD, identity
from torchfdtd.solver import voxelize
from test_fsp_native import fixture
from test_fsp_geometry_write import imported, shape_fixture, assert_untouched
from test_fsp_settings_write import settings_fixture


def added(kind,material,*,name=None):
    return Structure(id='new-'+kind,name=name or 'New '+kind,kind=kind,material=material,
        center=(.21,-.19,.13),size=(.73,.51,.49),radius=.38,radius_2=.23,radius_3=.29,
        inner_radius=.13,inner_radius_2=.09,make_ellipsoid=True,
        theta_start=315,theta_stop=105,rotation_axes=('y','z','x'),rotation_angles=(13,-23,17),
        vertices=((-0.3,-.2),(.3,-.2),(.3,.1),(0.,.1),(0.,.3),(-.3,.3)),mesh_order=4)


def roundtrip(doc,p):
    before=doc.data
    out,report=write_fsp_scene(doc,p);converted=convert_fsp(out,backend='cpu')
    assert converted.project is not None,converted.issues
    q=converted.project
    assert doc.data==before and report['source_sha256']==doc.fingerprint()
    assert_untouched(doc.data,out.data,report)
    # Structural edits move intact bytes. Check every reported retained slice,
    # not merely the prefix/suffix outside the replaced child-list envelope.
    for segment in report['preserved_segments'] or []:
        a,b=segment['original_range'];c,d=segment['output_range']
        assert doc.data[a:b]==out.data[c:d]
    assert [s.name for s in p.structures]==[s.name for s in q.structures]
    assert [report['id_mapping'][s.id] for s in p.structures]==[s.id for s in q.structures]
    assert [report['id_mapping'][s.id] for s in p.sources]==[s.id for s in q.sources]
    np.testing.assert_array_equal(voxelize(p)[0],voxelize(q)[0])
    return out,q,report


@pytest.mark.parametrize('kind',['rectangle','sphere','circle','ring','polygon'])
@pytest.mark.parametrize('dimension',['2d','3d'])
def test_add_primitive_to_file_without_its_class(kind,dimension,monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp,'load_api',lambda:pytest.fail('Commercial runtime loaded'))
    from test_fsp_mesh_write import authored_2d
    doc,p=imported(authored_2d() if dimension=='2d' else fixture());p.materials.append(Material(name='new glass',index=1.73))
    s=added(kind,'new glass',name='New μ 光 '+kind);p.structures.append(s)
    out,q,report=roundtrip(doc,p)
    assert report['structure_list']['added']==[s.id]
    assert report['structure_list']['external_reader_verified'] is False
    assert report['structure_list']['new_record_metadata']=='authored default drawing fields'
    assert len(out.root.children)==len(doc.root.children)+1
    assert struct.unpack_from('<I',out.data,out.root.child_count_offset)[0]==len(out.root.children)
    node=next(n for n in out.root.children if identity(n)['id']==q.structures[-1].id)
    assert node.legacy['index']==1.73 and node.legacy['override_mesh_order']
    assert node.legacy['mesh_order']==4
    np.testing.assert_allclose(node.legacy['rotation_angles'],[13,-23,17],atol=1e-14)
    if kind=='polygon':
        # Stored unrotated contour is global in metres, independent of the
        # native conversion's local-coordinate/pivot implementation.
        np.testing.assert_allclose(node.legacy['vertices_global'],(np.array(s.vertices)+s.center[:2])*1e-6,atol=1e-21)
    assert out.data[:doc.root.start]==doc.data[:doc.root.start]
    # The untouched original solid, source and monitor records remain exact.
    for loc in report['object_offsets']:
        if loc['original'] is None:continue
        old=next(n for n in doc.root.children if n.start==loc['original'])
        new=next(n for n in out.root.children if n.start==loc['output'])
        assert doc.data[old.start:old.end]==out.data[new.start:new.end]


def test_delete_all_then_add_again_and_repeat_edit():
    doc,p=imported(shape_fixture());removed=p.structures[0].id;p.structures=[]
    empty,q,report=roundtrip(doc,p)
    assert report['structure_list']['removed']==[removed] and not q.structures
    q.structures=[added('sphere',q.materials[0].name)]
    more,r,_=roundtrip(empty,q)
    r.structures[0].radius=.29;r.structures[0].name='Second edit'
    final,s,_=roundtrip(more,r)
    assert s.structures[0].radius==pytest.approx(.29)
    assert write_fsp_scene(final,s)[0].data==final.data


def test_duplicate_names_offsets_and_priority_order_are_unambiguous():
    doc,p=imported(fixture());first=p.structures[0]
    first.name='same';first.radius=.44
    p.materials.append(Material(name='higher index',index=2.7))
    duplicate=first.model_copy(deep=True);duplicate.id='copy';duplicate.material='higher index'
    duplicate.radius=.36;p.structures.append(duplicate)
    initial,q,_=roundtrip(doc,p)
    original_eps=voxelize(q)[0]
    q.structures.reverse()
    reordered,r,report=roundtrip(initial,q)
    assert report['structure_list']['changed'] and not report['structure_list']['added']
    assert not report['structure_list']['removed']
    assert np.count_nonzero(voxelize(r)[0]!=original_eps)>0  # equal priority, later wins
    assert len(set(s.id for s in r.structures))==2
    assert len(report['preserved_segments'])>3


def test_interleaved_source_and_monitor_keep_order_and_settings_after_removal():
    # Move the existing authored solid after its source before editing the
    # topology. This independently authored ordering exercises mixed records.
    base=FspDocument(settings_fixture());children=base.root.children
    raw=base.data[:children[0].start]+b''.join(base.data[n.start:n.end] for n in (children[0],children[2],children[1],children[3]))+base.data[base.root.end:]
    doc,p=imported(raw)
    p.structures=[added('circle',p.materials[0].name),added('rectangle',p.materials[0].name)]
    p.sources[0].phase=57;p.sources[0].name='Moved source'
    p.monitors[0].spectrum.frequency_points=11;p.monitors[0].spectrum.apodization='full'
    p.monitors[0].spectrum.apodization_center=5e-15;p.monitors[0].spectrum.apodization_time_width=9e-15
    _,q,report=roundtrip(doc,p)
    assert q.sources[0].phase==57 and q.monitors[0].spectrum.frequency_points==11
    assert report['id_mapping'][p.monitors[0].id]==q.monitors[0].id
    a,b=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_allclose(a.electric,b.electric,atol=2e-7,rtol=2e-6)
    np.testing.assert_allclose(a.frequency_fields[0]['fields'],b.frequency_fields[0]['fields'],atol=1e-28,rtol=2e-6)


def test_shared_point_monitor_channels_are_verified_after_topology_change():
    doc,p=imported(fixture(monitor_overrides=dict(outputE=np.array([1,0,1,0,0,0]))))
    p.structures.append(added('rectangle',p.materials[0].name))
    # Reorder actual output component names while retaining their native IDs.
    p.monitors[0].component='Hy';p.monitors[0].name='monitor Hy'
    p.monitors[1].component='Ex';p.monitors[1].name='monitor Ex'
    _,q,report=roundtrip(doc,p)
    byid={m.id:m for m in q.monitors}
    for m in p.monitors:assert byid[report['id_mapping'][m.id]].component==m.component


def test_topology_edits_preserve_opaque_original_fields():
    doc,p=imported(shape_fixture('ring'))
    original=next(n for n in doc.root.children if n.legacy)
    p.structures.append(added('sphere',p.materials[0].name));p.sources[0].phase=91
    out,_,report=roundtrip(doc,p)
    row=next(r for r in report['object_offsets'] if r['original']==original.start)
    retained=next(n for n in out.root.children if n.start==row['output'])
    assert doc.data[original.start:original.end]==out.data[retained.start:retained.end]
    assert retained.properties['unknown_unicode'].value=='preserve μ 和 광학'


def test_existing_geometry_api_rejects_structure_list_changes():
    doc,p=imported(fixture());p.structures.append(added('polygon',p.materials[0].name))
    with pytest.raises(ValueError,match='Adding, deleting'):write_fsp_geometry(doc,p)


def test_new_unknown_material_and_changed_class_are_rejected_without_output():
    doc,p=imported(fixture());before=doc.data
    p.materials.append(Material(name='new Drude',model='drude'))
    p.structures.append(added('sphere','new Drude'))
    with pytest.raises(ValueError,match='original database'):write_fsp_scene(doc,p)
    p.structures.pop();p.structures[0].kind='circle'
    with pytest.raises(ValueError,match='primitive type'):write_fsp_scene(doc,p)
    assert doc.data==before


@pytest.mark.parametrize('kind',['drude','lorentz'])
def test_new_primitive_can_reuse_unchanged_authored_database_material(kind):
    from test_fsp_native import items
    from test_fsp_binary import mapping,u
    base=FspDocument(fixture());uid='{11111111-2222-3333-4444-555555555555}'
    record=dict(materialuuid=uid,name='Authored oscillator',anisotropy=0,priority=7,
                type=2 if kind=='drude' else 4,permittivity=np.eye(3)*2.25)
    if kind=='drude':record.update(omegaplasma=np.eye(3)*1.2e15,nuplasma=np.eye(3)*1e14)
    else:record.update(omegalorentz=np.eye(3)*1.1e15,deltalorentz=np.eye(3)*.8e14,epsilonlorentz=np.eye(3)*.7)
    raw=base.data[:base.root.start-4]+u(1)+mapping(items(record))+base.data[base.root.start:]
    doc,p=imported(raw)
    from torchfdtd.fsp_native import convert_material
    mat,_=convert_material(doc.materials,uid);p.materials.append(mat)
    s=added('rectangle',mat.name);s.mesh_order=3;p.structures.append(s)
    out,q,_=roundtrip(doc,p)
    material=next(m for m in q.materials if m.name==q.structures[-1].material)
    assert material.model==kind and material.oscillators==mat.oscillators
    assert q.structures[-1].mesh_order==3  # object priority overrides database's 7
    assert out.data[:doc.root.start]==doc.data[:doc.root.start]
    a,b=Simulation(p).run(),Simulation(q).run()
    np.testing.assert_allclose(a.electric,b.electric,rtol=2e-6,atol=2e-7)


def test_simultaneous_mesh_and_topology_edit_cpu_cuda_batch():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA not available')
    doc,p=imported(settings_fixture());p.structures=[]
    p.materials.append(Material(name='inserted n',index=1.61))
    p.structures=[added(k,'inserted n') for k in ('rectangle','sphere','circle','ring','polygon')]
    data=p.model_dump();data['region'].update(mesh=.12,mesh_steps=(.12,.16,.2),material_sampling='yee',steps=70)
    p=Project.model_validate(data)
    _,q,_=roundtrip(doc,p)
    cpu=Simulation(p).run();q.region.backend='cuda';q.region.cuda_kernel='fused';q.region.cuda_monitor_kernel='fused'
    gpu=Simulation(q).run();batch=run_tensor_batch([q,q.model_copy(deep=True)]);batch.raise_for_errors()
    np.testing.assert_allclose(cpu.electric,gpu.electric,atol=3e-6,rtol=3e-5)
    np.testing.assert_allclose(cpu.magnetic,gpu.magnetic,atol=3e-6,rtol=3e-5)
    scale=np.max(abs(cpu.frequency_fields[0]['fields']));assert scale>0
    np.testing.assert_allclose(cpu.frequency_fields[0]['fields']/scale,gpu.frequency_fields[0]['fields']/scale,rtol=4e-5,atol=4e-5)
    for item in batch.items:
        np.testing.assert_array_equal(item.result.electric,gpu.electric)
        np.testing.assert_array_equal(item.result.magnetic,gpu.magnetic)
        np.testing.assert_array_equal(item.result.frequency_fields[0]['fields'],gpu.frequency_fields[0]['fields'])


def test_cli_structure_export_keeps_original_and_refuses_overwrite(tmp_path):
    import json,subprocess,sys
    doc,p=imported(fixture());p.structures.append(added('polygon',p.materials[0].name))
    original=tmp_path/'original.fsp';original.write_bytes(doc.data)
    scene=tmp_path/'scene.json';p.save(scene)
    target=tmp_path/'export.fsp';report=tmp_path/'report.json'
    command=[sys.executable,'-m','torchfdtd.cli','fsp-write-scene',str(original),str(scene),'--output',str(target),'--report',str(report)]
    run=subprocess.run(command,capture_output=True,text=True);assert run.returncode==0,run.stderr
    data=json.loads(report.read_text());assert data['scope']=='primitive scene and supported settings'
    assert data['structure_list']['added']==['new-polygon']
    assert len(convert_fsp(FspDocument.load(target)).project.structures)==2
    assert original.read_bytes()==doc.data
    assert subprocess.run(command,capture_output=True).returncode!=0
