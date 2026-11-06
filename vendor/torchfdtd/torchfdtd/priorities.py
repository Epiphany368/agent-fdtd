"""Product priorities for the native photonics workflow, independent of property counts."""
from collections import Counter

# Priority is product importance. Rank is dependency-aware delivery order.
WORKSTREAMS = {
    'reliability': (0,1,'required','실행 신뢰성·낭비 방지','자동 종료, 전체 장 발산 검사, 취소·저장 결과의 구분은 모든 계산의 기반이다.'),
    'convergence': (0,2,'required','정확도·수렴 검증','같은 물리량과 물리적 시간으로 메시를 비교해야 정확도 대비 비용을 판단할 수 있다.'),
    'release': (0,3,'required','독립 배포 패키지','공개할 코드·데이터·라이선스·자격증명 제외 목록을 검증해야 한다.'),
    'ports': (1,6,'required','소스·모드·정규화','단방향 여기, 모드 포트와 검증된 전송 목적함수가 실제 소자 설계에 필요하다.'),
    'materials': (1,4,'required','수동 분산 재료','다중 공진과 측정 광학상수의 수동 피팅이 파장 범위의 정확도를 결정한다.'),
    'interfaces': (1,5,'required','인터페이스·메시 효율','subpixel 및 독립 축 간격은 같은 오차에서 셀 수를 줄이는 수단이다. 수렴 증명이 먼저다.'),
    'adjoint': (0,4,'required','Torch 이산 adjoint·자동미분','주요 개발 목표. 형상·유전율부터 loss.backward와 optimizer까지 연결하고 물리 gradient·시간·메모리를 검증한다.'),
    'memory': (0,5,'required','계층형 메모리·대규모 실행','주요 개발 목표. VRAM·DRAM·저장장치 checkpoint와 공간·시간 분할, 전송·재계산·batch 비용을 함께 검증한다.'),
    'tensor_batch': (1,8,'required','GPU tensor batch','같은 격자의 독립 구조물을 함께 처리하고 동일 결과와 실제 cases/s 향상을 검증해야 한다.'),
    'observables': (1,9,'required','관측량·주파수 공간장','전송·흡수·회절·방사와 성분 선택은 장 그림을 물리적 설계 지표로 바꾼다.'),
    'geometry': (1,10,'required','핵심 CAD·설계 영역','다각형, 회전과 공간 재료 표현은 실제 소자와 topology 설계에 필요하다.'),
    'workflow': (2,11,'required','사용성·자동화','결과 탐색, GUI sweep, 그룹과 checkpoint는 연구 생산성을 높인다.'),
    'special_physics': (2,12,'conditional','특수 물성·실험 조건','비선형·자기·열·전하·박막 모델은 대상 연구가 요구할 때 추가한다.'),
    'distributed': (1,13,'required','단일 문제 multi-GPU·클러스터','FDTDX 동등성 목표에 따라 단일 도메인 분할·halo transpose와 실제 여러 GPU의 정확도·성능 검증이 필요하다.'),
    'special_cad': (2,14,'conditional','특수 CAD·제조 포맷','곡면·STL·공정 builder는 구체적인 사용 사례와 형상 검증이 있을 때 추가한다.'),
    'fsp_compatibility': (2,15,'conditional','FSP 추가 호환','기존 연구 프로젝트 이전에 필요한 형식부터 검증한다. 모든 과거 버전·내부 결과 형식의 완전 재현은 필수 구현에 앞서 진행하지 않는다.'),
    'compatibility': (3,16,'omit','제품별 내부 구현의 복제','전용 스크립트 언어 전체와 제품 고유 내부 알고리즘의 동일 구현은 필요하지 않다. 필요한 FSP 작업 이전은 별도로 유지한다.'),
    'cosmetic': (3,17,'omit','세부 표시 옵션의 일대일 복제','계산과 결과 해석에 영향 없는 표시 속성은 동일하게 복제할 필요가 없다.'),
}

# Keep the two main research workstreams ahead of remaining product features.
for _group,_rank in dict(materials=6,interfaces=7,ports=8,tensor_batch=9,observables=10,
                        geometry=11,workflow=12,special_physics=13,distributed=14,
                        special_cad=15,fsp_compatibility=16,compatibility=17,cosmetic=18).items():
    _value=WORKSTREAMS[_group]
    WORKSTREAMS[_group]=(_value[0],_rank,*_value[2:])
del _group,_rank,_value

EXACT = {
    'reliability': 'workflow.shutoff workflow.divergence native.cpu native.cuda native.precision native.cancel native.python native.npz native.cuda_fused native.batch_resume',
    'convergence': 'native.convergence',
    'release': 'native.release',
    'ports': 'source.oneway source.tfsf source.mode source.port source.angle source.gaussian source.import source.calibration analysis.normalization analysis.sparameters workflow.sparam_sweep',
    'materials': 'material.sampled_fit material.nk material.multipole material.sellmeier material.diagonal material.tensor',
    'interfaces': 'mesh.subpixel mesh.axis_step mesh.override_step mesh.user_nodes mesh.metal_mesh mesh.symmetric_mesh boundary.pec boundary.pmc boundary.symmetry boundary.antisymmetry native.mesh_preview',
    'adjoint': 'workflow.adjoint',
    'memory': 'workflow.memory_hierarchy workflow.out_of_core native.memory_profile',
    'tensor_batch': 'workflow.fused_batch',
    'observables': 'monitor.plane monitor.global monitor.custom monitor.chebyshev monitor.selection monitor.volume_dft monitor.time_space monitor.index monitor.time_sampling analysis.diffraction analysis.nearfar analysis.absorption analysis.cross_sections analysis.resonance analysis.mode_area analysis.directivity',
    'geometry': 'cad.polygon cad.triangle cad.ellipse cad.rotation3 cad.arrays cad.spatial_index cad.image_geometry cad.gds',
    'workflow': 'cad.groups cad.group_script cad.assembly workflow.checkpoint workflow.sweep workflow.nested_sweep workflow.optimization workflow.monte_carlo workflow.datasets workflow.ui_tabs monitor.movie monitor.averaging',
    'special_physics': 'material.conductive material.debye material.pec material.magnetic material.nonlinear material.raman material.gain material.graphene material.temperature material.plugin source.magnetic source.vector source.dc source.pupil source.incoherence monitor.pml analysis.impulse boundary.bfast',
    'distributed': 'workflow.multi_gpu workflow.distributed',
    'special_cad': 'cad.arc cad.pyramid cad.solid cad.path cad.surface cad.sheet_geometry cad.layer_builder cad.roughness cad.stl mesh.subgrid',
    'fsp_compatibility': 'workflow.fsp_write workflow.fsp_results workflow.fsp_versions',
    'compatibility': 'mesh.conformal mesh.auto_accuracy boundary.vendor_pml workflow.script workflow.library material.library',
}
EXACT = {id:group for group,ids in EXACT.items() for id in ids.split()}
OBJECTS = {
    'addfdtd':'reliability', 'addmesh':'interfaces',
    **{k:'geometry' for k in 'addrect addcircle addsphere addpoly addtriangle addimport'.split()},
    **{k:'special_cad' for k in 'addring addpyramid addwaveguide addsurface addcustom addplanarsolid addlayerbuilder add2drect add2dpoly'.split()},
    **{k:'ports' for k in 'addplane addgaussian addmode addport addimportedsource addtfsf addmodeexpansion'.split()},
    'adddipole':'ports',
    **{k:'observables' for k in 'addpower addprofile addtime addindex'.split()},
    **{k:'workflow' for k in 'addmovie addgroup addstructuregroup addanalysisgroup'.split()},
}


def classify(row):
    id,name,category=row['id'],row['name'].lower(),row['category']
    group=EXACT.get(id)
    if group is None and category.startswith('Material: '):
        group='materials' if category.split(': ',1)[1] in ('Sampled data','Lorentz','Plasma','Dielectric','Sellmeier','(n,k) Material') else 'special_physics'
    if group is None:
        group=OBJECTS.get(category)
        if group is None:raise ValueError('Unclassified feature: '+id)
        if any(s in name for s in ('opacity','wireframe','render detail','override color','color opacity','render type','render mode')):
            group='cosmetic'
        elif any(s in name for s in ('express mode','legacy conformal','mesh accuracy','conformal variant')):
            group='compatibility'
        elif category=='addfdtd':
            if any(s in name for s in ('mesh','dx','dy','dz','grading','refinement')):group='interfaces'
            elif any(s in name for s in ('global source','bloch','angle')):group='ports'
            elif any(s in name for s in ('global monitor','down sample')):group='observables'
            elif any(s in name for s in ('thread','process','resource','mpi')):group='distributed'
            elif any(s in name for s in ('checkpoint','restart')):group='workflow'
            elif not any(s in name for s in ('shutoff','diverg','stability','simulation time')):group='workflow'
    priority,rank,decision,title,reason=WORKSTREAMS[group]
    return dict(product_priority=f'P{priority}',delivery_rank=rank,decision=decision,
                workstream=group,workstream_title=title,priority_reason=reason,
                remaining=row['native']!='implemented' or row['ui'] not in ('implemented','n/a'))


def annotate_inventory(inventory):
    for row in inventory['features']:row.update(classify(row))
    inventory['priority_labels']={'P0':'P0 · 기반 필수','P1':'P1 · 핵심 필수','P2':'P2 · 후속·조건부','P3':'P3 · 복제 제외'}
    inventory['decision_labels']={'required':'필요','conditional':'사용 사례에 따라','omit':'목표에서 제외'}
    inventory['remaining_priority_counts']=dict(Counter(r['product_priority'] for r in inventory['features'] if r['remaining']))
    inventory['priority_note']='선형 photonics, Python 자동화, 단일 CUDA GPU와 inverse design을 기준으로 분류. 속성 행 수는 기능 수나 완성률이 아님. 필수 항목부터 구현. 조건부 항목은 구체적인 사용 사례가 생길 때만 착수하며 제외 항목은 새로 구현하지 않음. 이미 구현된 기능은 유지.'
    return inventory
