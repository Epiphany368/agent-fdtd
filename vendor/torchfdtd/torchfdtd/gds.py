"""Bounded GDSII geometry conversion through the optional gdstk dependency.

GDS provides XY geometry only. Z extents, materials and port-marker semantics
are explicit caller contracts. Nothing here creates sources or detectors.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import math
import re
import struct
import tempfile
import os
from contextlib import contextmanager

import numpy as np

from .models import Structure
from .geometry import validate_polygon, validate_polygon_holes, polygon_contains, rotation_matrix


def _gdstk():
    try:import gdstk
    except ImportError as exc:
        raise ImportError('GDS workflows require the optional dependency: pip install torchfdtd[gds]') from exc
    return gdstk


@contextmanager
def _gds_io():
    # gdstk 0.9 uses narrow Windows fopen and cannot open Unicode absolute
    # paths. A private, transient ASCII relative path avoids changing cwd.
    options={'dir':'.'} if os.name=='nt' else {}
    with tempfile.TemporaryDirectory(prefix='.torchfdtd-gds-',**options) as directory:
        yield Path(directory)/'layout.gds'


def _pair(layer,datatype):
    if any(isinstance(x,bool) or not isinstance(x,int) or not 0<=x<=65535 for x in (layer,datatype)):
        raise ValueError('GDS layer/datatype must be integers in [0, 65535].')


def _span(low,high):
    if not math.isfinite(low) or not math.isfinite(high) or low>=high:
        raise ValueError('Z bounds must be finite, increasing micrometre coordinates.')


@dataclass(frozen=True)
class GDSLayer:
    """One extruded layer/datatype pair.

    ``etch_by`` lists (layer, datatype) pairs whose polygons are subtracted
    from this pair by a boolean NOT before extrusion (etched holes, vias).
    Etch pairs count as mapped geometry and are not extruded by this entry.

    ``sidewall_angle_deg`` is measured from vertical. The drawn polygon is
    the footprint at ``z_min``; a positive angle shrinks the footprint toward
    ``z_max`` (a regular etch), a negative angle grows it (re-entrant). The
    taper is staircased: every z slice of ``import_gds(sidewall_z_nodes_um)``
    receives the polygon offset evaluated at its own centre.
    """
    layer: int
    datatype: int
    z_min: float
    z_max: float
    material: str
    mesh_order: int = 2
    etch_by: tuple[tuple[int,int],...] = ()
    sidewall_angle_deg: float = 0.

    def __post_init__(self):
        _pair(self.layer,self.datatype);_span(self.z_min,self.z_max)
        if not isinstance(self.material,str) or not self.material or isinstance(self.mesh_order,bool) or not isinstance(self.mesh_order,int) or not 1<=self.mesh_order<=100:
            raise ValueError('Provide a material name and mesh_order in [1, 100].')
        if isinstance(self.sidewall_angle_deg,bool) or not isinstance(self.sidewall_angle_deg,(int,float)) or not math.isfinite(self.sidewall_angle_deg) or not abs(self.sidewall_angle_deg)<90:
            raise ValueError('sidewall_angle_deg must be a finite angle from vertical inside (-90, 90).')
        etch=tuple(tuple(pair) for pair in self.etch_by)
        if any(len(pair)!=2 for pair in etch):raise ValueError('etch_by entries must be (layer, datatype) pairs.')
        for pair in etch:_pair(*pair)
        if (self.layer,self.datatype) in etch or len(set(etch))!=len(etch):raise ValueError('etch_by pairs must be unique and differ from the etched pair.')
        object.__setattr__(self,'etch_by',etch)


@dataclass(frozen=True)
class GDSPortLayer:
    """TEXT layer/TEXTTYPE pair, explicitly named datatype in this contract.

    Label text is the port name. Label transforms act on normal_xy and width.
    Width and Z bounds are in micrometres before label/reference transforms.
    Z bounds are physical stack coordinates and are not magnified.
    """
    layer: int
    datatype: int
    z_min: float
    z_max: float
    width_um: float
    normal_xy: tuple[float,float]

    def __post_init__(self):
        _pair(self.layer,self.datatype);_span(self.z_min,self.z_max)
        if not math.isfinite(self.width_um) or self.width_um<=0:
            raise ValueError('Port width must be finite and positive.')
        if len(self.normal_xy)!=2 or not all(math.isfinite(v) for v in self.normal_xy) or not math.isclose(math.hypot(*self.normal_xy),1,rel_tol=1e-8,abs_tol=1e-8):
            raise ValueError('Port normal_xy must be a finite unit direction vector.')


@dataclass(frozen=True)
class GDSPort:
    name: str
    center_um: tuple[float,float,float]
    normal: tuple[float,float,float]
    width_um: float
    height_um: float
    layer: int
    datatype: int
    marker_record: str = 'TEXT'

    def __post_init__(self):
        if not self.name or not all(math.isfinite(x) for x in (*self.center_um,*self.normal,self.width_um,self.height_um)):
            raise ValueError('Port metadata must have a name and finite coordinates.')
        if min(self.width_um,self.height_um)<=0 or not math.isclose(math.sqrt(sum(x*x for x in self.normal)),1,rel_tol=1e-8):
            raise ValueError('Port dimensions must be positive and its direction must be a unit vector.')


@dataclass(frozen=True)
class GDSLimits:
    max_file_bytes: int = 64*1024*1024
    max_instances: int = 10000
    max_structures: int = 1000
    # None, the Python-API default, admits any vertex count; the workbench server
    # passes an explicit limit (gds_service.SERVER_GDS_LIMITS).
    max_total_vertices: int | None = None
    max_depth: int = 64
    max_span_um: float = 10000.
    max_absolute_um: float = 1000000.

    def __post_init__(self):
        for name in ('max_file_bytes','max_instances','max_structures','max_total_vertices','max_depth'):
            value=getattr(self,name)
            if name=='max_total_vertices' and value is None:continue
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('Admission limits must be positive integers.')
        if any(not math.isfinite(x) or x<=0 for x in (self.max_span_um,self.max_absolute_um)):
            raise ValueError('Coordinate limits must be finite and positive.')

    @property
    def vertex_limit(self):
        return math.inf if self.max_total_vertices is None else self.max_total_vertices


@dataclass(frozen=True)
class GDSImport:
    structures: tuple[Structure,...]
    ports: tuple[GDSPort,...]
    report: dict

    def add_to(self,project):
        """Return a validated project copy. Material names must already exist."""
        from .models import Project
        data=project.model_dump()
        data['structures'].extend(s.model_dump() for s in self.structures)
        return Project.model_validate(data)


def _audit_records(data):
    # Reject records whose geometry/transform meaning is not admitted. GDS
    # properties and non-geometric library presentation metadata are reported.
    known={0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,
           0x0c,0x0d,0x0e,0x0f,0x10,0x11,0x12,0x13,0x16,0x17,0x19,0x1a,
           0x1b,0x1c,0x1f,0x20,0x21,0x22,0x23,0x26,0x2b,0x2c,0x2f,0x30,0x31}
    ignored={0x17,0x1f,0x20,0x22,0x23,0x26,0x2b,0x2c,0x2f}
    counts={};offset=0;ended=False
    while offset<len(data):
        if offset+4>len(data):raise ValueError('Truncated GDS record header.')
        length,kind,encoding=struct.unpack_from('>HBB',data,offset)
        if length<4 or length%2 or offset+length>len(data):raise ValueError('Invalid or truncated GDS record length.')
        if ended:raise ValueError('Unexpected data after GDS ENDLIB.')
        if kind not in known:raise ValueError(f'Unsupported GDS record 0x{kind:02x}; no geometry is silently discarded.')
        if kind==0x1a:
            if length!=6 or struct.unpack_from('>H',data,offset+4)[0]&0x7fff:
                raise ValueError('Absolute or unsupported GDS STRANS flags are not supported.')
        if kind in ignored:counts[f'0x{kind:02x}']=counts.get(f'0x{kind:02x}',0)+1
        if kind==0x04:ended=True
        offset+=length
    if not ended:raise ValueError('GDS ENDLIB is missing.')
    return dict(sorted(counts.items()))


def _canonical(points):
    points=np.asarray(points,dtype=float)
    if len(points)>1 and np.array_equal(points[0],points[-1]):points=points[:-1]
    if not np.isfinite(points).all():raise ValueError('GDS polygon coordinates must be finite.')
    try:validate_polygon(points)
    except ValueError as exc:
        raise ValueError('GDS contour is not an admitted simple polygon (self-intersections, touching edges or excessive vertices are unsupported): '+str(exc)) from exc
    shifted=points-points[0]
    area=np.sum(shifted[:,0]*np.roll(shifted[:,1],-1)-shifted[:,1]*np.roll(shifted[:,0],-1))
    if area<0:points=points[::-1]
    first=min(range(len(points)),key=lambda i:tuple(points[i]))
    return np.roll(points,-first,axis=0)


def _collapse(points):
    """Drop cyclic consecutive duplicates and X,Y,X back-tracks left by cut bridges."""
    while len(points)>=3:
        n=len(points);cut=None
        for i in range(n):
            if points[i]==points[(i+1)%n]:cut={(i+1)%n};break
            if points[(i-1)%n]==points[(i+1)%n]:cut={i,(i+1)%n};break
        if cut is None:break
        for k in sorted(cut,reverse=True):del points[k]
    return points


def _strip_collinear(points):
    """Remove vertices lying on the straight segment between their neighbours."""
    points=np.asarray(points,dtype=float);span=np.ptp(points,axis=0).max();tol=64*np.finfo(float).eps*max(span,1e-300)
    keep=[]
    for i,p in enumerate(points):
        a=points[i-1]-p;b=points[(i+1)%len(points)]-p
        if not (abs(a[0]*b[1]-a[1]*b[0])<=tol*span and a@b<0):keep.append(i)
    return points[keep]


def _split_contour(points):
    """Split one closed GDS contour at repeated vertices into simple loops.

    gdstk and other GDS tools store holes as a zero-width bridge whose
    endpoints appear twice in the contour. A simple contour is returned as
    is. Otherwise the repeated pair with the smallest span is cut out first,
    bridge remnants collapse, and every recovered loop drops collinear
    attachment vertices before simple-polygon admission.
    """
    points=np.asarray(points,dtype=float)
    if not np.isfinite(points).all():raise ValueError('GDS polygon coordinates must be finite.')
    remaining=[tuple(map(float,p)) for p in points]
    if len(set(remaining))==len(remaining):return [points]
    loops=[]
    while True:
        remaining=_collapse(remaining);seen={};pair=None
        for j,p in enumerate(remaining):
            if p in seen and (pair is None or j-seen[p]<pair[1]-pair[0]):pair=(seen[p],j)
            seen[p]=j
        if pair is None:break
        i,j=pair;loops.append(remaining[i:j]);remaining=remaining[:i]+remaining[j:]
    loops.append(remaining)
    loops=[_strip_collinear(loop) for loop in loops if len(loop)>=3]
    if not loops:raise ValueError('GDS contour collapses to a degenerate bridge.')
    return loops


def _polygons_with_holes(loops):
    """Group simple loops by even-odd nesting depth into (outer, holes) pairs."""
    loops=[_canonical(loop) for loop in loops]
    parents=[[j for j,b in enumerate(loops) if i!=j and np.all(polygon_contains(a[:,0],a[:,1],b))] for i,a in enumerate(loops)]
    result=[];assigned=0
    for i,a in enumerate(loops):
        if len(parents[i])%2:continue
        holes=[loops[j] for j in range(len(loops)) if len(parents[j])==len(parents[i])+1 and i in parents[j]]
        result.append((a,holes));assigned+=1+len(holes)
    if assigned!=len(loops):raise ValueError('GDS contour loops have ambiguous nesting (coincident or overlapping loops).')
    return result


def _gds_polygons(polygon):
    """Native (outer, holes) pairs of one GDS polygon, canonical and admitted."""
    pairs=_polygons_with_holes(_split_contour(polygon.points))
    for outer,holes in pairs:
        try:validate_polygon_holes(outer,holes)
        except ValueError as exc:raise ValueError('GDS hole contour is not admitted: '+str(exc)) from exc
    return pairs


def _admit_xy(points,limits,bounds):
    points=np.asarray(points)
    if not np.isfinite(points).all() or np.max(np.abs(points))>limits.max_absolute_um:
        raise ValueError('GDS coordinates exceed the finite absolute-coordinate admission limit.')
    if np.any(np.ptp(points,axis=0)>limits.max_span_um):raise ValueError('GDS XY span exceeds the admission limit.')
    if bounds is not None:
        xmin,ymin,xmax,ymax=bounds
        if np.any(points.min(axis=0)<[xmin,ymin]) or np.any(points.max(axis=0)>[xmax,ymax]):
            raise ValueError('GDS geometry or port lies outside requested XY bounds.')


def _preflight(cell,limits):
    memo={};visited=set()
    def visit(current,stack):
        if current.name in stack:raise ValueError('Cyclic GDS cell references are unsupported.')
        if len(stack)>limits.max_depth:raise ValueError('GDS hierarchy exceeds max_depth.')
        if current.name in memo:return memo[current.name]
        count=0;vertices=0;labels=0;paths=0;instances=1
        for polygon in current.polygons:
            repeat=max(1,polygon.repetition.size);count+=repeat;vertices+=repeat*len(polygon.points)
        for path in current.paths:
            # Loaded GDS paths are polygonized at the explicitly selected
            # tolerance. Native simple-polygon admission happens after flatten.
            if np.max(np.abs(path.widths()))>limits.max_span_um:raise ValueError('GDS path width exceeds admission limit.')
            polys=path.to_polygons();repeat=max(1,path.repetition.size)
            count+=len(polys)*repeat;vertices+=sum(len(p.points) for p in polys)*repeat;paths+=repeat
        labels+=sum(max(1,label.repetition.size) for label in current.labels)
        for ref in current.references:
            if isinstance(ref.cell,str):raise ValueError(f'Unresolved GDS cell reference: {ref.cell}')
            child=visit(ref.cell,(*stack,current.name));repeat=max(1,ref.repetition.size)
            count+=child[0]*repeat;vertices+=child[1]*repeat;labels+=child[2]*repeat;paths+=child[3]*repeat;instances+=child[4]*repeat
            if count+labels+instances>limits.max_instances or vertices>limits.vertex_limit:
                raise ValueError('Expanded GDS hierarchy exceeds instance/vertex admission limits.')
        if count+labels+instances>limits.max_instances or vertices>limits.vertex_limit:
            raise ValueError('Expanded GDS hierarchy exceeds instance/vertex admission limits.')
        visited.add(current.name);memo[current.name]=(count,vertices,labels,paths,instances)
        return memo[current.name]
    totals=visit(cell,())
    return totals,sorted(visited)


def _sidewall_slices(layer,nodes):
    """(z0, z1, offset_um) per native z slice of a tapered layer, base footprint at z_min."""
    edges=sorted({layer.z_min,layer.z_max,*(float(z) for z in nodes if layer.z_min<z<layer.z_max)})
    slope=math.tan(math.radians(layer.sidewall_angle_deg))
    return [(a,b,-((a+b)/2-layer.z_min)*slope) for a,b in zip(edges,edges[1:])]


def import_gds(path,*,cell,layers,port_layers=(),unmapped='error',limits=None,
               xy_bounds_um=None,path_tolerance_um=.001,sidewall_z_nodes_um=None):
    """Selected GDS cell -> explicit physical layer stack and marker metadata.

    ``unmapped='error'`` rejects unselected BOUNDARY/PATH layer pairs.
    ``'report'`` explicitly admits their omission and records counts. Labels
    outside port contracts are always counted as ignored descriptive metadata.
    ``sidewall_z_nodes_um`` (typically ``project.region.mesh_nodes[2]``) sets
    the z slices of every layer with a nonzero ``sidewall_angle_deg``.
    """
    gdstk=_gdstk();limits=GDSLimits() if limits is None else limits
    layers=tuple(layers);port_layers=tuple(port_layers)
    if not layers or not all(isinstance(x,GDSLayer) for x in layers):raise ValueError('Provide an explicit nonempty GDSLayer stack.')
    if len(layers)>limits.max_structures:raise ValueError('Layer stack exceeds max_structures.')
    if any(x.sidewall_angle_deg for x in layers):
        nodes=None if sidewall_z_nodes_um is None else np.asarray(sidewall_z_nodes_um,dtype=float).reshape(-1)
        if nodes is None or len(nodes)<2 or not np.isfinite(nodes).all() or np.any(np.diff(nodes)<=0):
            raise ValueError('Tapered sidewalls require sidewall_z_nodes_um: finite, increasing native z mesh nodes.')
    if not all(isinstance(x,GDSPortLayer) for x in port_layers):raise ValueError('Port contracts must be GDSPortLayer instances.')
    if unmapped not in ('error','report'):raise ValueError('unmapped must be error or report.')
    if not math.isfinite(path_tolerance_um) or path_tolerance_um<1e-6:raise ValueError('Path tolerance must be at least 1e-6 um.')
    if xy_bounds_um is not None:
        if len(xy_bounds_um)!=4 or not all(math.isfinite(v) for v in xy_bounds_um) or xy_bounds_um[0]>=xy_bounds_um[2] or xy_bounds_um[1]>=xy_bounds_um[3]:
            raise ValueError('XY bounds must be finite (xmin, ymin, xmax, ymax).')
    for layer in (*layers,*port_layers):
        if max(abs(layer.z_min),abs(layer.z_max))>limits.max_absolute_um or layer.z_max-layer.z_min>limits.max_span_um:
            raise ValueError('Stack Z bounds exceed coordinate admission limits.')
    path=Path(path)
    if path.stat().st_size>limits.max_file_bytes:raise ValueError('GDS file exceeds max_file_bytes.')
    with path.open('rb') as stream:data=stream.read(limits.max_file_bytes+1)
    if len(data)>limits.max_file_bytes:raise ValueError('GDS file exceeds max_file_bytes.')
    ignored_records=_audit_records(data)
    with _gds_io() as input_path:
        input_path.write_bytes(data)
        unit,precision=gdstk.gds_units(input_path)
        if not all(math.isfinite(x) and x>0 for x in (unit,precision)):raise ValueError('GDS unit and precision must be finite and positive.')
        library=gdstk.read_gds(input_path,unit=1e-6,tolerance=path_tolerance_um)
    names=[c.name for c in library.cells]
    if len(set(names))!=len(names):raise ValueError('Duplicate GDS cell names are ambiguous.')
    selected=[c for c in library.cells if c.name==cell]
    if len(selected)!=1:raise ValueError(f'Select one existing GDS cell explicitly. Available: {sorted(names)}')
    totals,visited=_preflight(selected[0],limits)
    # Polygonizing before reference transforms would incorrectly magnify GDS
    # negative-WIDTH paths. Transform paths first, then polygonize them.
    polygons=selected[0].get_polygons(apply_repetitions=True,include_paths=False)
    for flat_path in selected[0].get_paths(apply_repetitions=True):
        if np.max(np.abs(flat_path.widths()))>limits.max_span_um:
            raise ValueError('Transformed GDS path width exceeds admission limit.')
        polygons.extend(flat_path.to_polygons())
    if len(polygons)>limits.max_instances or sum(len(p.points) for p in polygons)>limits.vertex_limit:
        raise ValueError('Flattened GDS geometry exceeds instance/vertex admission limits.')
    stack={(entry.layer,entry.datatype) for entry in layers}
    etch_pairs={pair for entry in layers for pair in entry.etch_by}
    ports_by_pair={}
    for entry in port_layers:
        key=(entry.layer,entry.datatype)
        if key in ports_by_pair:raise ValueError('Each port TEXT layer/type requires one unambiguous contract.')
        ports_by_pair[key]=entry
    groups={};ignored={}
    for polygon in polygons:
        key=(polygon.layer,polygon.datatype)
        if key in stack or key in etch_pairs:groups.setdefault(key,[]).append(polygon)
        else:ignored[key]=ignored.get(key,0)+1
    if ignored and unmapped=='error':raise ValueError(f'Unmapped GDS geometry layer/datatype pairs: {sorted(ignored)}. Map them or explicitly choose unmapped="report".')
    layer_counts={key:len(group) for key,group in groups.items() if key in stack}
    converted=[];converted_vertices=0;hole_count=0;etched=[];tapered=[];plain={}
    def admit(key,pieces,layer,z0,z1):
        nonlocal converted_vertices,hole_count
        for outer,holes in pieces:
            _admit_xy(outer,limits,xy_bounds_um)
            if len(converted)>=limits.max_structures:raise ValueError('Converted stack exceeds max_structures.')
            converted_vertices+=len(outer)+sum(len(h) for h in holes);hole_count+=len(holes)
            if converted_vertices>limits.vertex_limit:raise ValueError('Converted stack exceeds max_total_vertices.')
            converted.append((key,tuple(map(tuple,outer)),tuple(tuple(map(tuple,h)) for h in holes),layer,(z0,z1)))
    for layer in layers:
        key=(layer.layer,layer.datatype);source=groups.get(key,[])
        if layer.etch_by:
            etch=[p for pair in layer.etch_by for p in groups.get(pair,[])]
            # Boolean NOT on the file precision grid, as any GDS tool applies it.
            if source and etch:source=gdstk.boolean(source,etch,'not',precision=precision/1e-6)
        if layer.sidewall_angle_deg:
            # Exact polygon offset (mitered joins) per native z slice; erosion may split or
            # remove a footprint, dilation may merge footprints. Nothing is rasterized here.
            slices=_sidewall_slices(layer,nodes);count=0
            for z0,z1,offset in slices:
                pieces=[pair for polygon in gdstk.offset(source,offset,use_union=True,precision=precision/1e-6) for pair in _gds_polygons(polygon)]
                admit(key,pieces,layer,z0,z1);count+=len(pieces)
            tapered.append(dict(layer=key[0],datatype=key[1],sidewall_angle_deg=layer.sidewall_angle_deg,slices=len(slices),polygons=count))
            if layer.etch_by:etched.append(dict(layer=key[0],datatype=key[1],etch_by=[list(pair) for pair in layer.etch_by],polygons=count))
            continue
        if layer.etch_by:
            pieces=[pair for polygon in source for pair in _gds_polygons(polygon)]
            etched.append(dict(layer=key[0],datatype=key[1],etch_by=[list(pair) for pair in layer.etch_by],polygons=len(pieces)))
        else:
            if key not in plain:plain[key]=[pair for polygon in source for pair in _gds_polygons(polygon)]
            pieces=plain[key]
        admit(key,pieces,layer,layer.z_min,layer.z_max)
    if not converted:raise ValueError('The selected cell has no geometry on the requested layer stack.')
    positions={id(layer):i for i,layer in enumerate(layers)}
    converted.sort(key=lambda item:(positions[id(item[3])],item[4],item[0],item[1],item[2]))
    structures=[];all_points=[]
    for i,(key,points,holes,layer,(z0,z1)) in enumerate(converted):
        points=np.asarray(points);lower=points.min(axis=0);upper=points.max(axis=0);center=(lower+upper)/2
        # Vertical layers keep the identity they had before sidewall support.
        record={k:v for k,v in asdict(layer).items() if k!='sidewall_angle_deg' or v}
        if layer.sidewall_angle_deg:record['z_slice']=(z0,z1)
        token=hashlib.sha256(repr((key,tuple(map(tuple,points)),holes,record)).encode()).hexdigest()[:12]
        structures.append(Structure(id=f'gds-{token}-{i}',name=f'{cell}:{key[0]}/{key[1]}:{i}'[:100],kind='polygon',
            center=(*center,(z0+z1)/2),size=(*(upper-lower),z1-z0),
            vertices=tuple(map(tuple,points-center)),holes=tuple(tuple(map(tuple,np.asarray(h)-center)) for h in holes),
            material=layer.material,mesh_order=layer.mesh_order))
        all_points.extend(points)
    _admit_xy(all_points,limits,xy_bounds_um)
    ports=[];ignored_labels=0
    for label in selected[0].get_labels(apply_repetitions=True):
        key=(label.layer,label.texttype)
        if key not in ports_by_pair:ignored_labels+=1;continue
        rule=ports_by_pair[key]
        if not re.fullmatch(r'[A-Za-z0-9_.:+/-]{1,100}',label.text):raise ValueError('Port marker names must be 1-100 ASCII letters, digits or _ . : + / - characters.')
        normal=np.array(rule.normal_xy,dtype=float)
        if label.x_reflection:normal[1]*=-1
        c,s=math.cos(label.rotation),math.sin(label.rotation)
        normal=np.array([[c,-s],[s,c]])@normal
        width=rule.width_um*label.magnification
        if not math.isfinite(width) or not 0<width<=limits.max_span_um:raise ValueError('Transformed port width exceeds admission limits.')
        tangent=np.array([-normal[1],normal[0]])
        _admit_xy([np.asarray(label.origin)-width/2*tangent,np.asarray(label.origin)+width/2*tangent],limits,xy_bounds_um)
        ports.append(GDSPort(label.text,(*map(float,label.origin),(rule.z_min+rule.z_max)/2),(*map(float,normal),0.),width,rule.z_max-rule.z_min,*key))
    ports.sort(key=lambda p:(p.name,p.center_um,p.normal))
    if len({p.name for p in ports})!=len(ports):raise ValueError('Expanded port marker names must be unique. Rename markers in repeated instances.')
    report=dict(format='gdsii',source_sha256=hashlib.sha256(data).hexdigest(),cell=cell,
        file_unit_m=unit,file_precision_m=precision,coordinate_unit='um',path_tolerance_um=path_tolerance_um,
        visited_cells=visited,expanded_polygon_count=len(polygons),expanded_path_count=totals[3],
        expanded_cell_instances=totals[4],structures=len(structures),native_vertex_count=converted_vertices,hole_count=hole_count,
        layer_stack=[asdict(x) for x in layers],mapped_layers=[dict(layer=k[0],datatype=k[1],polygons=v) for k,v in sorted(layer_counts.items())],
        etched_layers=etched,tapered_layers=tapered,ignored_geometry=[dict(layer=k[0],datatype=k[1],polygons=v) for k,v in sorted(ignored.items())],
        ignored_label_count=ignored_labels,ignored_metadata_records=ignored_records,ports=[asdict(p) for p in ports],
        bounds_um=[[*np.min(all_points,axis=0),min(z[0] for *_,z in converted)],
                   [*np.max(all_points,axis=0),max(z[1] for *_,z in converted)]],
        limitations=['Polygon extrusions with explicit holes; tapered sidewalls are staircased per native z slice.',
                     'Contours touching their holes are rejected.',
                     'Separate contours on one layer are unioned as in GDS, not treated as holes; use etch_by for that.',
                     'Port metadata does not create a mode source or detector; see gds_ports.prepare_gds_two_port.',
                     'GDS properties and presentation metadata do not define simulation physics.'])
    return GDSImport(tuple(structures),tuple(ports),report)


def export_gds(path,structures,*,layers,cell='TOP',unit_m=1e-6,precision_m=1e-9):
    """Export supported XY polygon/rectangle geometry with a returned Z sidecar.

    ``layers`` maps every enabled structure ID to (layer, datatype). GDS cannot
    store extrusion/material physics: returned stack metadata must be retained.
    No ports, sources, detectors or simulation-region settings are exported.
    """
    from datetime import datetime
    gdstk=_gdstk();structures=tuple(structures)
    if not cell or len(cell.encode('ascii',errors='replace'))>32 or not cell.isascii():raise ValueError('Use a nonempty ASCII GDS cell name of at most 32 characters.')
    if not all(math.isfinite(v) and v>0 for v in (unit_m,precision_m)) or precision_m>unit_m:
        raise ValueError('Require finite positive precision_m <= unit_m.')
    active=[s for s in structures if s.enabled]
    if not active:raise ValueError('No enabled geometry to export.')
    if len({s.id for s in active})!=len(active) or set(layers)!={s.id for s in active}:
        raise ValueError('Provide one explicit layer/datatype pair for every unique enabled structure ID.')
    lib=gdstk.Library(unit=unit_m,precision=precision_m);target=lib.new_cell(cell);stack=[]
    for obj in sorted(active,key=lambda x:x.id):
        pair=layers[obj.id]
        if len(pair)!=2:raise ValueError('Export layers require (layer, datatype).')
        _pair(*pair)
        if obj.kind not in ('polygon','rectangle'):raise ValueError('GDS export supports native polygons and rectangles only.')
        matrix=rotation_matrix(obj)
        if not np.allclose(matrix[2,:2],0,atol=1e-12) or not np.allclose(matrix[:2,2],0,atol=1e-12):
            raise ValueError('Tilted extrusions cannot be represented by an XY GDS polygon.')
        if obj.kind=='polygon':points=np.asarray(obj.vertices)
        else:
            x,y=np.asarray(obj.size[:2])/2;points=np.array([[-x,-y],[x,-y],[x,y],[-x,y]])
        def rounded(local):
            world=_canonical(np.asarray(local)@matrix[:2,:2].T+np.asarray(obj.center[:2]))
            integer=np.rint(world*1e-6/precision_m)
            if np.max(np.abs(integer))>2147483647:raise ValueError('Coordinates exceed GDS signed 32-bit precision range.')
            return _canonical(integer*precision_m/1e-6)
        outer=rounded(points);holes=[rounded(h) for h in obj.holes]
        if holes:
            # Holes are written the way GDS tools store them: one bridged contour.
            try:validate_polygon_holes(outer,holes)
            except ValueError as exc:raise ValueError('Rounded polygon holes are no longer admitted: '+str(exc)) from exc
            scale=1e-6/unit_m
            target.add(*gdstk.boolean(gdstk.Polygon(outer*scale),[gdstk.Polygon(h*scale) for h in holes],'not',
                                      precision=precision_m/unit_m,layer=pair[0],datatype=pair[1]))
        else:target.add(gdstk.Polygon(outer*1e-6/unit_m,layer=pair[0],datatype=pair[1]))
        stack.append(dict(structure_id=obj.id,layer=pair[0],datatype=pair[1],material=obj.material,
                          z_min=obj.center[2]-obj.size[2]/2,z_max=obj.center[2]+obj.size[2]/2,mesh_order=obj.mesh_order))
    path=Path(path)
    with _gds_io() as output_path:
        lib.write_gds(output_path,max_points=199,timestamp=datetime(2000,1,1))
        output=output_path.read_bytes()
    path.write_bytes(output)
    return dict(format='gdsii',cell=cell,unit_m=unit_m,precision_m=precision_m,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),structures=len(active),
                omitted_disabled=len(structures)-len(active),layer_stack=stack,
                maximum_rounding_um=precision_m/1e-6/2,
                limitations=['GDS contains XY geometry only. Retain this Z/material stack sidecar.',
                             'Sources, detectors, ports and solver settings are not exported.'])
