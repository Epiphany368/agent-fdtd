"""Analytic local-coordinate solids and ordered rigid transformations.

Display tessellation never determines material membership. Lengths are in um.
"""
import math
import numpy as np


def axis_rotation(axis,angle):
    if axis=='none':return np.eye(3)
    angle%=360
    c,s={0:(1.,0.),90:(0.,1.),180:(-1.,0.),270:(0.,-1.)}.get(angle,(math.cos(math.radians(angle)),math.sin(math.radians(angle))))
    a='xyz'.index(axis);b=(a+1)%3;d=(a+2)%3;r=np.eye(3)
    r[b,b]=r[d,d]=c;r[b,d]=-s;r[d,b]=s
    return r


def rotation_matrix(obj):
    """Local -> world. Legacy z angle first, then fixed world axes 1, 2, 3."""
    # Legacy membership evaluated the unreduced angle. Match that arithmetic
    # in support bounds as well, including very large finite legacy angles.
    angle=math.radians(obj.rotation);c,s=math.cos(angle),math.sin(angle)
    r=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    for axis,angle in zip(obj.rotation_axes,obj.rotation_angles):r=axis_rotation(axis,angle)@r
    return r


def radii(obj):
    return (obj.radius,obj.radius_2 if obj.make_ellipsoid else obj.radius,
            obj.radius_3 if obj.make_ellipsoid else obj.radius)


def validate_polygon(vertices):
    """A single simple contour, CW or CCW. Reject crossings and touching edges."""
    v=np.asarray(vertices,dtype=float)
    if v.ndim!=2 or v.shape[1]!=2 or not 3<=len(v)<=2048:raise ValueError('Polygon requires 3 to 2048 (x,y) vertices.')
    span=np.ptp(v,axis=0).max();tol=64*np.finfo(float).eps*max(span,1e-300)
    edges=np.roll(v,-1,axis=0)-v
    if np.any(np.linalg.norm(edges,axis=1)<=tol):raise ValueError('Polygon has a repeated vertex or zero-length edge. Do not repeat the closing vertex.')
    w=v-v[0];area=np.sum(w[:,0]*np.roll(w[:,1],-1)-w[:,1]*np.roll(w[:,0],-1))
    if abs(area)<=tol*span:raise ValueError('Polygon area must be nonzero.')
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    previous=np.roll(edges,1,axis=0)
    if np.any((abs(cross(previous,edges))<=tol*span)&((previous*edges).sum(axis=1)<0)):
        raise ValueError('Adjacent polygon edges must not overlap or reverse direction.')
    for i,(a,b) in enumerate(zip(v,np.roll(v,-1,axis=0))):
        ids=np.arange(i+2,len(v));ids=ids[ids!=(i-1)%len(v)]
        if not len(ids):continue
        c=v[ids];d=v[(ids+1)%len(v)]
        f=cross(b-a,c-a);g=cross(b-a,d-a);h=cross(d-c,a-c);k=cross(d-c,b-c)
        eps=tol*span
        boxes=(np.maximum(np.minimum(a,b),np.minimum(c,d))<=np.minimum(np.maximum(a,b),np.maximum(c,d))+tol).all(axis=1)
        straddle=((f<=eps)&(g>=-eps)|(g<=eps)&(f>=-eps))&((h<=eps)&(k>=-eps)|(k<=eps)&(h>=-eps))
        if np.any(boxes&straddle):raise ValueError('Polygon edges must not cross or touch nonadjacent edges.')


def _edges_cross(p,q):
    """True when an edge of contour p crosses or touches an edge of contour q."""
    span=max(np.ptp(p,axis=0).max(),np.ptp(q,axis=0).max());tol=64*np.finfo(float).eps*max(span,1e-300);eps=tol*span
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    c=q[None];d=np.roll(q,-1,axis=0)[None]
    for start in range(0,len(p),256):
        a=p[start:start+256,None];b=np.roll(p,-1,axis=0)[start:start+256,None]
        f=cross(b-a,c-a);g=cross(b-a,d-a);h=cross(d-c,a-c);k=cross(d-c,b-c)
        boxes=(np.maximum(np.minimum(a,b),np.minimum(c,d))<=np.minimum(np.maximum(a,b),np.maximum(c,d))+tol).all(axis=-1)
        straddle=((f<=eps)&(g>=-eps)|(g<=eps)&(f>=-eps))&((h<=eps)&(k>=-eps)|(k<=eps)&(h>=-eps))
        if np.any(boxes&straddle):return True
    return False


def validate_polygon_holes(vertices,holes):
    """Simple hole contours strictly inside the outer contour and strictly apart from each other."""
    if len(holes)>1024:raise ValueError('A polygon admits at most 1024 holes.')
    outer=np.asarray(vertices,dtype=float);loops=[]
    for hole in holes:
        validate_polygon(hole);loop=np.asarray(hole,dtype=float)
        inside,edge=_polygon_parity(loop[:,0],loop[:,1],outer)
        if not np.all(inside&~edge) or _edges_cross(outer,loop):
            raise ValueError('Polygon holes must lie strictly inside the outer contour without touching it.')
        for other in loops:
            if (np.any(np.logical_or(*_polygon_parity(loop[:,0],loop[:,1],other)))
                    or np.any(np.logical_or(*_polygon_parity(other[:,0],other[:,1],loop))) or _edges_cross(loop,other)):
                raise ValueError('Polygon holes must not overlap, nest or touch each other.')
        loops.append(loop)


def _polygon_parity(x,y,vertices):
    """Crossing parity and boundary membership of one closed contour."""
    shape=np.broadcast_shapes(x.shape,y.shape);inside=np.zeros(shape,bool);boundary=np.zeros(shape,bool)
    v=np.asarray(vertices);span=np.ptp(v,axis=0).max();tol=64*np.finfo(float).eps*max(span,1e-300)
    for (ax,ay),(bx,by) in zip(v,np.roll(v,-1,axis=0)):
        cross=(bx-ax)*(y-ay)-(by-ay)*(x-ax)
        boundary|=(abs(cross)<=tol*max(abs(bx-ax),abs(by-ay)))&(x>=min(ax,bx)-tol)&(x<=max(ax,bx)+tol)&(y>=min(ay,by)-tol)&(y<=max(ay,by)+tol)
        if by!=ay:inside^=((ay>y)!=(by>y))&(x<(bx-ax)*(y-ay)/(by-ay)+ax)
    return inside,boundary


def polygon_contains(x,y,vertices,holes=()):
    """Even-odd membership of the outer contour and its holes. Every contour boundary is material."""
    inside,boundary=_polygon_parity(x,y,vertices)
    for hole in holes:
        parity,edge=_polygon_parity(x,y,hole);inside^=parity;boundary|=edge
    return inside|boundary


def contains(obj,x,y,z):
    delta=[x-obj.center[0],y-obj.center[1],z-obj.center[2]]
    isotropic=not obj.make_ellipsoid
    only_z=all(axis in ('none','z') or angle%360==0 for axis,angle in zip(obj.rotation_axes,obj.rotation_angles))
    invariant=(obj.kind=='sphere' and isotropic) or (obj.kind in ('circle','ring') and isotropic and only_z and (obj.kind!='ring' or obj.angular_span==360))
    if invariant or (obj.rotation==0 and not any(obj.rotation_angles)):
        u,v,w=delta
    elif not any(obj.rotation_angles):
        # Preserve existing z-only operation ordering for legacy projects.
        a=math.radians(obj.rotation);c,s=math.cos(a),math.sin(a)
        u,v,w=c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1],delta[2]
    else:
        matrix=rotation_matrix(obj)
        u,v,w=[sum(matrix[a,b]*delta[a] for a in range(3) if matrix[a,b]!=0) for b in range(3)]
    if obj.kind=='rectangle':return (abs(u)<=obj.size[0]/2)&(abs(v)<=obj.size[1]/2)&(abs(w)<=obj.size[2]/2)
    if obj.kind=='polygon':return polygon_contains(u,v,obj.vertices,obj.holes)&(abs(w)<=obj.size[2]/2)
    rx,ry,rz=radii(obj)
    if obj.kind=='sphere':
        return (u/rx)**2+(v/ry)**2+(w/rz)**2<=1 if obj.make_ellipsoid else u*u+v*v+w*w<=rx*rx
    mask=((u/rx)**2+(v/ry)**2<=1 if obj.make_ellipsoid else u*u+v*v<=rx*rx)&(abs(w)<=obj.size[2]/2)
    if obj.kind=='ring':
        if obj.inner_radius>0:
            mask&=((u/obj.inner_radius)**2+(v/(obj.inner_radius_2 if obj.make_ellipsoid else obj.inner_radius))**2>=1
                   if obj.make_ellipsoid else u*u+v*v>=obj.inner_radius**2)
        if obj.angular_span<360:
            angle=(np.degrees(np.arctan2(v,u))-obj.theta_start)%360
            mask&=(angle<=obj.angular_span+1e-12)|((u==0)&(v==0))
    return mask


def object_bounds(obj):
    """Exact support bounds for boxes/ellipsoids/cylinders, conservative for arcs."""
    r=rotation_matrix(obj);center=np.asarray(obj.center)
    if obj.kind=='polygon':
        points=np.array([[x,y,z] for x,y in obj.vertices for z in (-obj.size[2]/2,obj.size[2]/2)])@r.T+center
        low,high=points.min(axis=0),points.max(axis=0)
        return tuple((low+high)/2),tuple(high-low)
    if obj.kind=='rectangle':half=np.abs(r)@(np.array(obj.size)/2)
    elif obj.kind=='sphere':half=np.sqrt((r*r)@(np.array(radii(obj))**2))
    else:
        rx,ry,_=radii(obj)
        half=np.sqrt((r[:,0]*rx)**2+(r[:,1]*ry)**2)+abs(r[:,2])*obj.size[2]/2
    return tuple(center),tuple(2*half)
