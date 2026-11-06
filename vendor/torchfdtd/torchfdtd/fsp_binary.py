"""Independent, loss-preserving reader for observed FSP 1.1 records.

Derived from controlled files written by installed Lumerical v241, not vendor
code. Unknown versions/record layouts fail explicitly. Opaque legacy drawing
fields remain byte-for-byte in the document. This is not native solver mapping.
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class FspFormatError(ValueError):
    pass


LEGACY_CLASSES={'{75954e61-0067-4a0e-8d16-1ee8d371e892}':6,
                '{23046316-141b-4111-aa2f-9e18de790b6c}':8,
                '{921e6d99-3bcb-4063-b513-216a3bb757d9}':4,
                '{b4a87699-109a-4252-b2c4-06be606f82f3}':5,
                '{49651cff-e643-43b3-b70d-0b6846faa79c}':11,
                '{b1c063bf-9b6d-49e1-b1dc-2c5cddef5d3c}':13}


@dataclass
class Value:
    value: object
    kind: int
    start: int
    end: int
    payload: int


@dataclass
class Node:
    uid: str
    start: int
    properties: dict[str, Value]
    children: list = field(default_factory=list)
    legacy: dict = field(default_factory=dict)
    end: int = 0
    legacy_fields: dict = field(default_factory=dict)
    child_count_offset: int = 0

    @property
    def name(self):
        return str(self.properties['name'].value) if 'name' in self.properties else self.legacy.get('name', self.uid)


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def fail(self, message):
        raise FspFormatError(f'{message} at byte {self.pos}')

    def take(self, n):
        if n < 0 or n > len(self.data)-self.pos:
            self.fail('Truncated or excessive record length')
        start=self.pos; self.pos+=n
        return self.data[start:self.pos]

    def u32(self):return struct.unpack('<I', self.take(4))[0]
    def i32(self):return struct.unpack('<i', self.take(4))[0]
    def f64(self):return struct.unpack('<d', self.take(8))[0]

    def string(self):
        n=self.u32()
        if not 1 <= n <= 4_000_000:self.fail('Invalid string length')
        raw=self.take(n)
        if raw[-1:] != b'\0':self.fail('Missing string terminator')
        try:return raw[:-1].decode('utf-8')
        except UnicodeDecodeError:self.fail('Unsupported string encoding')

    def value(self):
        start=self.pos;kind=self.u32()
        if kind==0:
            payload=self.pos+4; value=self.string()
        elif kind in (1,2):
            count=self.u32();payload=self.pos
            if count>1_000_000:self.fail('Excessive scalar array')
            dtype='<f8' if kind==1 else '<i4'
            array=np.frombuffer(self.take(count*(8 if kind==1 else 4)),dtype=dtype).copy()
            value=array[0].item() if count==1 else array
        elif kind in (3,5):
            flags=self.u32();count=self.u32();nd=self.u32()
            if flags!=0 or nd>8 or count>16_000_000:self.fail('Unsupported matrix header')
            shape=tuple(self.u32() for _ in range(nd));real=self.u32();storage=self.u32()
            if real not in (0,1) or storage not in (0,1) or (math.prod(shape) if shape else 0)!=count:
                self.fail('Invalid matrix dimensions or storage')
            payload=self.pos;dtype='<f8' if kind==3 else '<i4';size=8 if kind==3 else 4
            value=np.frombuffer(self.take(count*size),dtype=dtype).copy()
            if not real:
                if kind!=3:self.fail('Unsupported complex integer matrix')
                value=value+1j*np.frombuffer(self.take(count*size),dtype=dtype)
            value=value.reshape(shape or (0,),order='F')
        else:self.fail(f'Unsupported value type {kind}')
        return Value(value,kind,start,self.pos,payload)

    def mapping(self):
        count=self.u32()
        if count>10000:self.fail('Excessive property count')
        values={}
        for _ in range(count):
            key=self.string()
            if key in values:self.fail('Duplicate property name')
            values[key]=self.value()
        return values

    def legacy_matrix(self):
        kind=self.u32();flags=self.take(1)
        count=self.u32();nd=self.u32()
        if kind!=3 or flags!=b'\0' or count>16_000_000 or nd>8:self.fail('Unsupported legacy matrix')
        shape=tuple(self.u32() for _ in range(nd))
        if self.u32()!=1 or self.u32()!=1 or math.prod(shape)!=count:self.fail('Unsupported legacy matrix storage')
        return np.frombuffer(self.take(count*8),dtype='<f8').copy().reshape(shape,order='F')

    def tagged_legacy_double(self):
        if self.take(1)!=b'\x01':self.fail('Unsupported legacy expression')
        return self.f64()

    def legacy_tail(self,size):
        raw=self.take(size)
        enabled=struct.unpack_from('<I',raw,len(raw)-9)[0]
        if enabled not in (0,1):self.fail('Unexpected legacy enabled flag')
        return bool(enabled)

    def shape_metadata(self, raw, kind, fields=None, base=0):
        """Decode independently varied rotation/ellipsoid/priority fields.

        Unrecognized expression forms remain readable/preservable, but must
        not be mistaken for decoded defaults by a native scene converter.
        """
        r = Reader(raw)
        result = {'transform_metadata_decoded': False}
        positions={}
        def locate(key,start,end,codec):positions[key]=(base+start,base+end,codec)
        try:
            if kind in (4, 8):
                if r.take(1) != b'\0':return result
                ellipsoid = r.u32()
                if ellipsoid not in (0, 1):return result
                result['make ellipsoid'] = bool(ellipsoid)
                locate('make ellipsoid',1,5,'i32')
                result['radius 2'] = r.tagged_legacy_double()
                locate('radius 2',r.pos-8,r.pos,'f64')
                if kind == 8:
                    result['radius 3'] = r.tagged_legacy_double()
                    locate('radius 3',r.pos-8,r.pos,'f64')
            axes = []
            for _ in range(3):
                if r.take(1) != b'\0':return result
                code = r.i32()
                if code not in (-1, 0, 1, 2):return result
                axes.append({-1:'none', 0:'x', 1:'y', 2:'z'}[code])
                locate('rotation axis '+str(4-len(axes)),r.pos-4,r.pos,'axis')
            angles = r.legacy_matrix().reshape(-1)
            if len(angles) != 3 or not np.isfinite(angles).all():return result
            for i in range(3):locate('rotation '+str(i+1),r.pos-8*(i+1),r.pos-8*i,'f64')
            db_position=88 if kind==11 else len(raw)-56
            order_position=db_position+5
            if raw[db_position-1] != 0 or raw[order_position-1] != 0:return result
            use_database = struct.unpack_from('<I', raw, db_position)[0]
            order = struct.unpack_from('<i', raw, order_position)[0]
            if use_database not in (0, 1) or order < 1:return result
            locate('use database order',db_position,db_position+4,'i32')
            locate('mesh order',order_position,order_position+4,'i32')
            if kind==11:
                r.pos=97;start=r.pos;pivot=r.legacy_matrix()
                if pivot.shape!=(2,1) or not np.isfinite(pivot).all():return result
                result.update(pivot_x=float(pivot[0,0]),pivot_y=float(pivot[1,0]))
                locate('polygon pivot',start,r.pos,'matrix')
            result.update(rotation_axes=axes[::-1], rotation_angles=angles[::-1].tolist(),
                          mesh_order=order, override_mesh_order=not bool(use_database),
                          transform_metadata_decoded=True)
            if fields is not None:fields.update(positions)
        except FspFormatError:
            pass
        return result

    def group_metadata(self, raw):
        """Rotation axes/angles lead the group drawing block; the rest stays opaque."""
        r = Reader(raw)
        result = {'transform_metadata_decoded': False}
        try:
            axes = []
            for _ in range(3):
                if r.take(1) != b'\0':return result
                code = r.i32()
                if code not in (-1, 0, 1, 2):return result
                axes.append({-1:'none', 0:'x', 1:'y', 2:'z'}[code])
            angles = r.legacy_matrix().reshape(-1)
            if len(angles) != 3 or not np.isfinite(angles).all():return result
            result.update(rotation_axes=axes[::-1], rotation_angles=angles[::-1].tolist(), transform_metadata_decoded=True)
        except FspFormatError:
            pass
        return result

    def legacy_geometry(self):
        kind=self.u32();version=self.u32()
        fields={}
        def scalar(key,tagged=False):
            v=self.tagged_legacy_double() if tagged else self.f64()
            fields[key]=(self.pos-8,self.pos,'f64');return v
        def text(key):
            start=self.pos;v=self.string();fields[key]=(start,self.pos,'string');return v
        def tail(size):
            start=self.pos;enabled=self.legacy_tail(size)
            fields['enabled']=(self.pos-9,self.pos-5,'i32')
            return enabled,start
        if (kind,version) in ((11,22),(13,18)):
            geometry={}
            if kind==11:
                if self.take(1)!=b'\0':self.fail('Unsupported polygon material expression')
                old_material=self.i32();index=scalar('index',True)
                start=self.pos;geometry['vertices_global']=self.legacy_matrix().T
                fields['vertices_global']=(start,self.pos,'matrix')
                geometry['z min']=scalar('z min',True);geometry['z max']=scalar('z max',True)
            else:
                geometry.update(self.group_metadata(self.take(97)))  # remaining drawing fields preserved verbatim
                if self.take(1)!=b'\0':self.fail('Unsupported group material expression')
                old_material=self.i32();index=scalar('index',True)
            if self.take(1)!=b'\x02':self.fail('Unsupported legacy index expression')
            expression=text('index_expression')
            if self.take(1)!=b'\0':self.fail('Unsupported legacy enabled expression')
            self.take(8)  # legacy expression metadata, not the enabled flag
            if self.take(2)!=b'\0\x02':self.fail('Unsupported legacy name expression')
            name=text('name')
            if kind==11:
                enabled,start=tail(189)
                geometry.update(self.shape_metadata(self.data[start:self.pos],kind,fields,start))
            else:
                for key in ('x','y','z'):geometry[key]=self.tagged_legacy_double()
                # Setup script and user properties precede the enabled block.
                # An empty script and no properties is the fixed 30-byte tail.
                self.take(4)
                if self.take(1)!=b'\x02':self.fail('Unsupported group script expression')
                geometry['script']=text('script')
                if self.take(1)!=b'\0':self.fail('Unsupported group property list')
                count=self.u32()
                if count>10000:self.fail('Excessive group property count')
                geometry['user_properties']=[]
                for _ in range(count):
                    if self.take(1)!=b'\x02':self.fail('Unsupported group property name')
                    key=self.string()
                    if self.take(1)!=b'\0':self.fail('Unsupported group property type')
                    code=self.u32()
                    if self.take(1)!=b'\x02':self.fail('Unsupported group property value')
                    geometry['user_properties'].append((key,code,self.string()))
                enabled,_=tail(15)
            map_start=self.pos;properties=self.mapping()
            fields['__properties__']=(map_start,self.pos,'mapping')
            if 'use_relative_coordinates' not in properties:self.fail('Unexpected legacy extension')
            return properties,dict(kind=kind,version=version,name=name,enabled=bool(enabled),index=index,
                                   index_expression=expression,legacy_material=old_material,**geometry),fields
        # Known fixed-format revisions. Their unexposed drawing/material fields
        # are preserved, never interpreted as native solver settings.
        formats={6:(32,('y min','x min','y max','x max','z min','z max'),192),
                 8:(25,('radius','x','y','z'),167),
                 4:(25,('radius','x','y','z min','z max'),158),
                 5:(26,('inner radius','outer radius','x','y','theta start','theta stop','z min','z max'),144)}
        if kind not in formats or version!=formats[kind][0]:self.fail(f'Unsupported legacy geometry {kind}/{version}')
        _, keys, tail_bytes=formats[kind]
        old_material=self.i32();index=scalar('index')
        geometry={key:scalar(key) for key in keys}
        index_expression=text('index_expression');self.take(8);name=text('name')
        tail_start=self.pos
        enabled,_=tail(tail_bytes)
        geometry.update(self.shape_metadata(self.data[tail_start:self.pos], kind,fields,tail_start))
        map_start=self.pos;properties=self.mapping()
        fields['__properties__']=(map_start,self.pos,'mapping')
        if 'materialuuid' not in properties or 'use_relative_coordinates' not in properties:
            self.fail('Unexpected legacy property extension')
        return properties,dict(kind=kind,version=version,name=name,enabled=bool(enabled),index=index,
                               index_expression=index_expression,legacy_material=old_material,**geometry),fields

    def node(self, depth=0):
        if depth>64:self.fail('Object tree nesting exceeds limit')
        start=self.pos;tag=self.u32();length=self.u32()
        if tag not in (1000,1001) or length!=38:self.fail('Unsupported object record')
        raw=self.take(length)
        try:uid=raw.decode('ascii')
        except UnicodeDecodeError:self.fail('Invalid object class identifier')
        if not (uid.startswith('{') and uid.endswith('}')):self.fail('Invalid object class identifier')
        legacy={};legacy_fields={}
        if tag==1001:
            map_start=self.pos;properties=self.mapping()
            legacy_fields['__properties__']=(map_start,self.pos,'mapping')
        else:
            if uid not in LEGACY_CLASSES:self.fail('Unsupported legacy object class')
            properties,legacy,legacy_fields=self.legacy_geometry()
            if legacy['kind']!=LEGACY_CLASSES[uid]:self.fail('Legacy class and geometry type disagree')
        children=[]
        if legacy.get('kind')==13:
            # Script-generated primitives follow the group headerless, kind first.
            count=self.u32()
            if count>100000:self.fail('Excessive generated object count')
            classes={kind:cls for cls,kind in LEGACY_CLASSES.items()}
            for _ in range(count):
                record=self.pos;p,g,f=self.legacy_geometry()
                if g['kind']==13:self.fail('Nested script-generated groups are not decoded yet')
                g['script_generated']=True
                children.append(Node(classes[g['kind']],record,p,[],g,self.pos,f))
        elif tag==1001:
            extra=self.u32()
            if extra!=0:self.fail('Result/dataset records are not decoded yet')
        child_count_offset=self.pos
        count=self.u32()
        if count>10000:self.fail('Excessive object count')
        children+=[self.node(depth+1) for _ in range(count)]
        return Node(uid,start,properties,children,legacy,self.pos,legacy_fields,child_count_offset)


class FspDocument:
    def __init__(self, data: bytes):
        if len(data)>128*1024*1024:raise FspFormatError('FSP exceeds the current 128 MiB limit')
        self.data=bytes(data);r=Reader(self.data)
        header=b'LUMERICAL file version 1.1\0'
        if r.take(len(header))!=header:r.fail('Unsupported FSP version')
        # Observed layout-mode header and empty table of contents only.
        header_fields=r.take(44)
        if header_fields not in tuple(struct.pack('<5I',mode,10,4,8,1)+bytes(24) for mode in (1,3)):
            r.fail('Unsupported FSP header or result-bearing file')
        if r.take(40)!=b'table of contents\0end table of contents\0':r.fail('Nonempty table of contents is not decoded yet')
        marker=b'Lumerical material data file version 2.0\0'
        if r.take(len(marker))!=marker:r.fail('Unsupported material section')
        count=r.u32()
        if count>10000:r.fail('Excessive material count')
        self.materials=[r.mapping() for _ in range(count)]
        self.root=r.node()
        self.footer_start=r.pos
        # Footer decoding is separate from the simulation object tree.
        self.footer=r.mapping()
        # Two zero words and a sweep/optimization record count end the file.
        # Those records are retained verbatim, never decoded.
        if r.u32()!=0 or r.u32()!=0:r.fail('Unsupported footer records')
        self.sweep_records=r.u32()
        self.trailing=r.take(len(self.data)-r.pos)
        if bool(self.sweep_records)!=bool(self.trailing):r.fail('Unsupported footer records')

    @classmethod
    def load(cls,path):
        path=Path(path)
        if path.stat().st_size>128*1024*1024:raise FspFormatError('FSP exceeds the current 128 MiB limit')
        return cls(path.read_bytes())

    def nodes(self):
        def walk(n):
            yield n
            for c in n.children:yield from walk(c)
        return list(walk(self.root))

    def fingerprint(self):return hashlib.sha256(self.data).hexdigest()

    def save(self,path):
        # No edits means exactly the original bytes, including opaque fields.
        with Path(path).open('xb') as f:f.write(self.data)

    def inspect(self):
        # Reuse only the public JSON value codec, never the vendor API loader.
        from .fsp import encode_value
        def describe(node):
            return {'record_offset':node.start,'class_id':node.uid,'name':node.name,
                    'legacy_geometry':encode_value(node.legacy),
                    'properties':{k:encode_value(v.value) for k,v in node.properties.items()},
                    'children':[describe(n) for n in node.children]}
        return {'format':'FSP 1.1 / observed v241 layouts','requires_lumerical':False,
                'source_bytes':len(self.data),'source_sha256':self.fingerprint(),
                'materials':[{k:encode_value(v.value) for k,v in m.items()} for m in self.materials],
                'root':describe(self.root),'native_execution_allowed':False,
                'limitations':['Only the explicitly recognized layout-mode record revisions are decoded.',
                               'Legacy drawing fields and footer bytes are retained but not fully interpreted.',
                               'This raw inspection is not a native scene; use fsp-convert for the supported independent mapping.']}

    def with_monitor_edits(self, patches):
        """Fixed-width independent edits of validated generic monitor settings.

        A record offset, not an ambiguous object name, identifies each target.
        No resizing, renaming, script execution or whole-file reconstruction.
        """
        allowed={'apodizationCenter','apodizationWidth','apodizationType','nfreqDesired',
                 'useWavelengthSpacing','useGlobalDFT','useSourceLimits','userF1','userF2',
                 'startTime','stopTime'}
        data=bytearray(self.data);by_offset={n.start:n for n in self.nodes()};changed=[];seen=set()
        for patch in patches:
            if set(patch)!={'record_offset','property','value'}:raise ValueError('Patch requires record_offset, property and value only')
            node=by_offset.get(patch['record_offset']);key=patch['property'];value=patch['value']
            identity=(patch['record_offset'],key)
            if identity in seen:raise ValueError('Duplicate patch target')
            seen.add(identity)
            if node is None or key not in allowed or key not in node.properties:
                raise ValueError('Unsupported monitor edit target')
            if node.uid not in ('{15879b8f-3efb-45dd-a50f-acd1106124bc}', '{3cbd368f-83d1-45a9-8b2b-154d82400a51}'):
                raise ValueError('Editing this monitor class is not validated')
            if 'apodizationType' not in node.properties and 'startTime' not in node.properties:
                raise ValueError('Target is not a recognized monitor record')
            stored=node.properties[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Monitor edit requires a finite number')
            if stored.kind not in (1,2) or not isinstance(stored.value,(int,float)):
                raise ValueError('Only scalar monitor settings are editable')
            if key in ('apodizationCenter','startTime','stopTime') and value<0:raise ValueError('Time must be nonnegative')
            if key in ('apodizationWidth','userF1','userF2') and value<=0:raise ValueError('Width and frequency must be positive')
            if key=='apodizationType' and value not in (0,1,2,3):raise ValueError('Unsupported apodization enum')
            if key in ('useWavelengthSpacing','useGlobalDFT','useSourceLimits') and value not in (0,1):raise ValueError('Toggle requires zero or one')
            if key=='nfreqDesired' and not 1<=value<=100000:raise ValueError('Frequency count outside supported range')
            if stored.kind==2 and value!=int(value):raise ValueError('Integer property requires integer value')
            packed=struct.pack('<d' if stored.kind==1 else '<i',value if stored.kind==1 else int(value))
            if stored.payload+len(packed)!=stored.end:raise ValueError('Unexpected scalar storage')
            data[stored.payload:stored.end]=packed;changed.append((stored.payload,stored.end))
        result=FspDocument(bytes(data))
        edited_nodes={patch['record_offset'] for patch in patches}
        for node in result.nodes():
            if node.start not in edited_nodes:continue
            for low,high in (('userF1','userF2'),('startTime','stopTime')):
                if low in node.properties and high in node.properties and node.properties[low].value>node.properties[high].value:
                    raise ValueError(f'{low} must not exceed {high}')
        # Same byte positions, object tree and all unedited content are retained.
        if [(n.start,n.end,n.name) for n in result.nodes()]!=[(n.start,n.end,n.name) for n in self.nodes()]:
            raise ValueError('An edit changed the object layout')
        result.changed_ranges=changed
        return result
