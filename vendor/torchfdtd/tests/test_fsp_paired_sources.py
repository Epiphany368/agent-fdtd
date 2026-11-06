"""Independently authored layout bytes. No vendor runtime, project or field data."""
import math
import sys
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch

from torchfdtd import Simulation,Source,run_tensor_batch
from torchfdtd.fsp_binary import FspDocument,Value
from torchfdtd.fsp_native import PLANE,TFSF,FDTD,convert_fsp,paired_polarization
from test_fsp_native import fixture


def paired_fixture(kind='tfsf',axis=0,direction='+',*,polarization=31.,phi=17.,extra=None,region_extra=None):
    codes=(3,4,9) if kind=='tfsf' else (1,2,5)
    source=dict(type=codes[axis],direction=int(direction=='+'),amplitude=.71,phase=19.,
                polarizationAngle=polarization,phi=phi,theta=0.,unfold=np.zeros((6,1)),
                xcoord=0.,ycoord=0.,zcoord=0.)
    if kind=='tfsf':
        source.update(left=-.8e-6,bottom=-.8e-6,z1=-.8e-6,z2=.8e-6,width=1.6e-6,height=1.6e-6)
        region={}
    else:
        source.update(planeWaveSource=1,planeWaveType=0,angleDefinition=0,polarizationDefinition=0,
                      useIFT=0,additionalDelay=0.,useCustomPupilFunction=0,
                      GUIxspan=4e-6,GUIyspan=4e-6,GUIzspan=4e-6)
        source['xyz'[axis]+'coord']=-1e-6 if direction=='+' else 1e-6
        # Periodic files retain one extra plane on each side, removed on import.
        region={f'BCType{2*a+side}':1 for a in range(3) if a!=axis for side in (0,1)}
        region.update({a+'Grid':np.linspace(-2.1e-6,2.1e-6,43) for a in 'xyz' if a!='xyz'[axis]})
    source.update(extra or {});region.update(region_extra or {})
    return fixture(source,region,source_class=TFSF if kind=='tfsf' else PLANE)


@pytest.mark.parametrize('kind',['plane','tfsf'])
@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('direction',['+','-'])
def test_paired_import_native_fields_and_original_bytes(kind,axis,direction,monkeypatch):
    from torchfdtd import fsp
    monkeypatch.setattr(fsp,'load_api',lambda:pytest.fail('Independent import loaded vendor runtime'))
    raw=paired_fixture(kind,axis,direction);doc=FspDocument(raw)
    report=convert_fsp(doc,backend='cpu');assert report.project is not None,report.issues
    p=report.project;s=p.sources[0]
    assert (s.kind,s.normal,s.direction,s.amplitude,s.phase)==(kind,'xyz'[axis],direction,.71,19.)
    a=np.deg2rad(48.)
    expected=np.array([(0,np.cos(a),np.sin(a)),(np.sin(a),0,np.cos(a)),(np.cos(a),np.sin(a),0)][axis])
    components=dict(s.polarization_components)
    np.testing.assert_allclose([components.get('E'+c,0) for c in 'xyz'],expected,atol=1e-15,rtol=1e-14)
    assert components.get('E'+'xyz'[axis],0)==0
    assert doc.data==raw and p.import_provenance.source_sha256==doc.fingerprint()
    codes={i['code'] for i in report.issues}
    assert {'paired_source_units','incident_line_delay'}<=codes
    p.region.steps=140;p.region.precision='float64'
    result=Simulation(p).run()
    # Independent assembly: two Cartesian sources with signed phase weights.
    ref=p.model_copy(deep=True);ref.sources=[]
    for j,w in enumerate(expected):
        if w==0:continue
        data=s.model_dump();data.update(id='cartesian-'+str(j),theta=None,phi=0.,component='E'+'xyz'[j],
                                        amplitude=.71*abs(w),phase=19.+(180 if w<0 else 0))
        ref.sources.append(Source(**data))
    independently_assembled=Simulation(ref).run()
    np.testing.assert_allclose(result.electric,independently_assembled.electric,atol=2e-13,rtol=2e-12)
    np.testing.assert_allclose(result.magnetic,independently_assembled.magnetic,atol=2e-13,rtol=2e-12)
    assert np.max(abs(result.electric))>1e-4


def test_cardinal_and_signed_polarization_vectors_do_not_create_longitudinal_fields():
    for axis in range(3):
        for angle in (0,90,180,270,360,-90,31,211,1e-9):
            theta,phi=paired_polarization(axis,angle)
            s=Source(kind='tfsf',normal='xyz'[axis],component='Ex' if axis==2 else 'Ez',theta=theta,phi=phi)
            weights=dict(s.polarization_components)
            assert weights.get('E'+'xyz'[axis],0)==0
            assert sum(w*w for w in weights.values())==pytest.approx(1.)
            if angle%90==0:assert len(weights)==1


@pytest.mark.parametrize('kind,key,value',[('plane','planeWaveSource',0),('plane','planeWaveType',1),
    ('plane','polarizationDefinition',1),('plane','angleDefinition',1),('plane','useIFT',1),
    ('plane','additionalDelay',1e-15),('plane','useCustomPupilFunction',1),
    ('plane','GUIyspan',1e-6),('plane','type',3),('tfsf','type',1),('tfsf','theta',13.),
    ('tfsf','direction',2),('tfsf','left',-.7e-6),('tfsf','unfold',np.ones((6,1)))])
def test_unsupported_source_settings_fail_without_silent_substitution(kind,key,value):
    report=convert_fsp(FspDocument(paired_fixture(kind,extra={key:value})))
    assert report.project is None and not report.as_dict()['native_execution_allowed']
    assert any(i['code']=='object_mapping' and i['severity']=='error' for i in report.issues)


def test_source_material_shell_is_rejected_during_import_and_periodic_support_is_clipped():
    doc=FspDocument(paired_fixture())
    next(n for n in doc.root.children if n.legacy).legacy['x']=.8e-6
    report=convert_fsp(doc)
    assert report.project is None
    assert any(i['code']=='source_environment' and 'source' in i['object_id'] for i in report.issues)
    report=convert_fsp(FspDocument(paired_fixture('plane',extra={'GUIyspan':6e-6,'ycoord':.2e-6})))
    assert report.project is not None,report.issues
    assert report.project.sources[0].center[1]==0
    assert report.project.sources[0].size[1]==pytest.approx(4.)
    assert any(i['code']=='plane_cell_support' for i in report.issues)


def test_source_global_local_settings_sampled_signal_and_invalid_global_diagnostics():
    local={'local::pulseTypeActual':0,'local::defineSourceBy':0,'local::frequency':190e12,
           'local::mEliminateDC':0,'local::pulseLength':2e-15,'local::offset':4e-15,
           'local::optimizeForShortPulse':0,'local::eliminateDiscontinuities':0}
    global_={'frequencyEnvelopeType':0,'sourcePreference':0,'globalFrequency':210e12,'globalEliminateDC':0,
             'pulseLength':23e-15,'offset':67e-15,'optimizeForShortPulse':0,'eliminateDiscontinuities':0}
    for kind in ('plane','tfsf'):
        raw=paired_fixture(kind,extra=dict(local,useGlobalSource=1,frequency=210e12,pulseLength=23e-15,offset=67e-15),region_extra=global_)
        doc=FspDocument(raw);report=convert_fsp(doc)
        assert report.project is not None,report.issues
        p=report.project;s=p.sources[0]
        assert s.pulse_length==2e-15 and p.resolved_source(s).pulse_length==23e-15
        node=next(n for n in doc.root.children if n.uid==FDTD)
        node.properties['globalFrequency'].value=-1
        report=convert_fsp(doc)
        assert report.project is None
        assert any(i['code']=='global_source_mapping' and i['severity']=='error' for i in report.issues)
        raw=paired_fixture(kind,extra=dict(frequencyEnvelopeType=2,userTime=np.array([0.,1e-15,2e-15]),
                                          userAmp=np.array([0.,1.,0.]),userPhs=np.zeros(3)))
        p=convert_fsp(FspDocument(raw)).project
        assert p is not None and p.sources[0].pulse=='sampled'
        assert p.sources[0].signal.amplitude==[0.,1.,0.]


def test_2d_polarization_mapping_is_not_assumed_from_3d():
    dt=.8*.1e-6/299792458/math.sqrt(2)
    raw=paired_fixture(region_extra=dict(dimension=0,dt=dt,MaxSimTime=30*dt))
    report=convert_fsp(FspDocument(raw))
    assert report.project is None
    assert any('2D orientation' in i['message'] for i in report.issues)


def test_imported_paired_sources_cuda_graph_and_tensor_match_independent_runs():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    for kind in ('plane','tfsf'):
        projects=[convert_fsp(FspDocument(paired_fixture(kind,direction=d)),backend='cuda').project for d in ('+','-')]
        for p in projects:
            p.region.steps=140;p.region.precision='float64';p.region.cuda_kernel='fused'
        outputs=run_tensor_batch(projects)
        for p,item in zip(projects,outputs.items):
            single=Simulation(p).run();batch=item.load()
            for field in ('electric','magnetic','signals'):
                np.testing.assert_array_equal(getattr(single,field),getattr(batch,field))


def test_cli_preserves_original_and_reports_conversion_scope(tmp_path):
    raw=paired_fixture();source=tmp_path/'source.fsp';source.write_bytes(raw)
    target=tmp_path/'native.json';report=tmp_path/'conversion.json'
    result=subprocess.run([sys.executable,'-m','torchfdtd.cli','fsp-convert',str(source),
                           '--output',str(target),'--report',str(report),'--backend','cpu'],capture_output=True)
    assert result.returncode==0,result.stderr
    assert source.read_bytes()==raw
    import json
    assert json.loads(target.read_text())['sources'][0]['kind']=='tfsf'
    assert json.loads(report.read_text())['requires_lumerical'] is False
