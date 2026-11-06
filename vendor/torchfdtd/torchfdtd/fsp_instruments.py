"""Plan top-level source/monitor edits using decoded input properties only.

New generic records have authored property defaults. Split monitor records
reuse the original template while patching recognized controls. External reader
acceptance and the semantics of opaque references remain unverified.
"""
from __future__ import annotations

import struct
import numpy as np

from .fsp_binary import Node
from .fsp_native import DIPOLE,PLANE,TFSF,TIME,DFT


class Values:
    def __init__(self,initial=None):self.values=dict(initial or {})
    def set(self,node,key,value):self.values[key]=value
    def update(self,node,values):self.values.update(values)


def generic_record(uid,values):
    from .fsp_settings import _string,_value
    u=lambda n:struct.pack('<I',n)
    return u(1001)+u(38)+uid.encode('ascii')+u(len(values))+b''.join(_string(k)+_value(v) for k,v in values.items())+u(0)+u(0)


def template_record(document,node,values):
    from .fsp_settings import PropertyPlan
    plan=PropertyPlan(document);plan.update(node,values)
    chunks=[];cursor=node.start
    for start,end,data,_,_ in sorted(plan.patches()):
        if start<cursor or end>node.end:raise ValueError('Invalid monitor template edit.')
        chunks.extend((document.data[cursor:start],data));cursor=end
    chunks.append(document.data[cursor:node.end])
    return b''.join(chunks)


def source_record(source,base,project,origin,native_only):
    from .fsp_settings import _source
    if source.kind=='point':
        uid=DIPOLE;defaults=dict(sourceType=0)
    elif source.injection=='oneway' and source.kind in ('plane','tfsf'):
        if project.region.dimension!='3d':raise ValueError('FSP paired source records are currently mapped for 3D only.')
        uid=TFSF if source.kind=='tfsf' else PLANE
        defaults=dict(theta=0.,unfold=np.zeros((6,1)))
        if uid==PLANE:
            defaults.update(planeWaveSource=1,planeWaveType=0,angleDefinition=0,polarizationDefinition=0,
                            useIFT=0,additionalDelay=0.,useCustomPupilFunction=0,GUIxspan=0.,GUIyspan=0.,GUIzspan=0.)
    else:raise ValueError('New FSP sources require an electric dipole or a mapped 3D one-way plane/TFSF source.')
    node=Node(uid,0,{})
    values=Values(defaults)
    _source(values,node,source,source,base,project,origin,native_only,fresh=True)
    return generic_record(uid,values.values)


def monitor_values(monitor,uid,base,project,origin,native_only):
    from .fsp_settings import _monitor
    defaults=dict(recordInPML=0,simulationType=0,monitorShape=0,spatialAveraging=1,
                  outputPower=0,enabled=int(monitor.enabled))
    if uid==TIME:
        defaults.update(startTime=0.,stopMethod=0,downsampleT=1,outputP=np.zeros(3),outputE=np.zeros(6))
    else:
        defaults.update(standardDFT=1,partialSpectralAverage=0,totalSpectralAverage=0,
                        unfold=np.zeros((6,1)),useGlobalDFT=0,
                        **{'output'+c:0 for c in ('Ex','Ey','Ez','Hx','Hy','Hz','Px','Py','Pz')})
    values=Values(defaults);node=Node(uid,0,{})
    _monitor(values,node,[monitor],[monitor],base,project,origin,native_only,fresh=True)
    if monitor.kind=='field' and project.region.dimension=='2d' and monitor.size[2]!=1:
        native_only.append(monitor.id+'.size[2] (invariant 2D monitor display span, imported as 1 um)')
    return values.values


def _shared_group(monitors):
    """Can one FSP record retain these native outputs and their exact order?"""
    if len(monitors)<2:return True
    first=monitors[0]
    if any(m.kind!='point' for m in monitors):return False
    keys=('center','enabled','time_downsample','use_global_monitor','inherit_apodization','spectrum')
    if any(any(getattr(m,k)!=getattr(first,k) for k in keys) for m in monitors[1:]):return False
    names={m.name.removesuffix(' '+m.component) for m in monitors}
    if len(names)!=1:return False
    ranks=[('Ex','Ey','Ez','Hx','Hy','Hz').index(m.component) for m in monitors]
    return ranks==sorted(set(ranks))


def plan_instruments(plan,document,base,project,conversion,native_only):
    from .fsp_settings import _source,_monitor
    origin=np.asarray(conversion.origin_m)
    by_offset={n.start:n for n in document.root.children}
    # Analysis-group members are imported for execution only; their records sit inside the group.
    members={key for row in conversion.mappings if row.get('script_generated') for key in row['native_ids']}
    if members&({s.id for s in project.sources}|{m.id for m in project.monitors}):
        raise ValueError('Analysis-group members imported from the FSP are not written back. Remove them from the scene or edit the group in Lumerical; the geometry export keeps them unchanged.')
    mapped={key:(by_offset[row['record_offset']],row['native_ids']) for row in conversion.mappings for key in row['native_ids'] if not row.get('script_generated')}
    old_sources={s.id:s for s in base.sources};old_monitors={m.id:m for m in base.monitors}
    nodes={};new_records={};source_keys=[];monitor_keys=[];monitor_groups=[];splits=[]
    source_changed=[s.id for s in base.sources]!=[s.id for s in project.sources]
    monitor_changed=[m.id for m in base.monitors]!=[m.id for m in project.monitors]
    for source in project.sources:
        key=source.id;source_keys.append(key)
        if key in old_sources:
            node,_=mapped[key];nodes[key]=node
            _source(plan,node,old_sources[key],source,base,project,origin,native_only)
        else:new_records[key]=source_record(source,base,project,origin,native_only)
    index=0;used=set()
    while index<len(project.monitors):
        monitor=project.monitors[index];key=monitor.id
        if key not in old_monitors:
            uid=TIME if monitor.kind=='point' and project.resolved_monitor(monitor).spectrum.sampling=='fft' else DFT
            new_records[key]=generic_record(uid,monitor_values(monitor,uid,base,project,origin,native_only))
            monitor_keys.append(key);monitor_groups.append([key]);index+=1;continue
        node,original_ids=mapped[key]
        group=[monitor];next_index=index+1
        while next_index<len(project.monitors) and project.monitors[next_index].id in original_ids:
            group.append(project.monitors[next_index]);next_index+=1
        # Preserve a complete shared record only when channels remain adjacent,
        # compatible and in the order emitted by the native FSP reader.
        selected=[m for m in project.monitors if m.id in original_ids]
        keep_shared=(node.start not in used and len(group)==len(selected) and _shared_group(group))
        if keep_shared:
            nodes[key]=node
            _monitor(plan,node,[old_monitors[k] for k in original_ids],group,base,project,origin,native_only)
        else:
            group=[monitor];next_index=index+1;monitor_changed=True
            values=monitor_values(monitor,node.uid,base,project,origin,native_only)
            if node.start not in used:
                nodes[key]=node;plan.update(node,values)
            else:new_records[key]=template_record(document,node,values)
            splits.append(dict(input_id=key,original_record=node.start,template_reused=True))
        used.add(node.start)
        monitor_keys.append(key);monitor_groups.append([m.id for m in group]);index=next_index
    return dict(nodes=nodes,new_records=new_records,sources=source_keys,monitors=monitor_keys,
                monitor_groups=monitor_groups,splits=splits,
                source_changed=source_changed,monitor_changed=monitor_changed,
                source_added=[s.id for s in project.sources if s.id not in old_sources],
                source_removed=[s.id for s in base.sources if s.id not in source_keys],
                monitor_added=[m.id for m in project.monitors if m.id not in old_monitors],
                monitor_removed=[m.id for m in base.monitors if m.id not in {n.id for n in project.monitors}])
