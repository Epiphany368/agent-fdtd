"""Independent geometric line integrals for dielectric interface quadrature.

Every primitive boundary contributes analytic line roots. Final membership
between roots resolves overlaps with the same material order as voxelization.
Face averages integrate these exact piecewise-constant line integrals with
Gauss-Legendre quadrature. No display tessellation participates.
"""
from itertools import product

import numpy as np

from .geometry import contains, object_bounds, radii, rotation_matrix


class DielectricGeometry:
    """Values default to the instantaneous permittivity (epsilon-infinity for
    dispersive objects). ``value(material)``, and ``value(None)`` for the
    background, substitutes other piecewise-constant values such as labels."""
    def __init__(self, project, value=None):
        self.region=project.region
        self.dim=2 if self.region.dimension=='2d' else 3
        self.periodic=[a for a in range(self.dim) if self.region.boundaries.pair(a)[0].kind in ('periodic','bloch')]
        self.span=np.array(self.region.actual_size)
        self.background=self.region.background_index**2 if value is None else value(None)
        materials={m.name:m for m in project.materials}
        self.objects=[]
        for obj in sorted(project.structures,key=lambda s:-s.mesh_order):
            if not obj.enabled:continue
            m=materials[obj.material]
            self.objects.append((obj,m.instantaneous_epsilon if value is None else value(m)))
        values=[self.background]+[v for _,v in self.objects]
        self.epsilon_bounds=min(values),max(values)

    def wrap(self, points):
        p=np.array(points,copy=True,dtype=float)
        for a in self.periodic:p[...,a]=(p[...,a]+self.span[a]/2)%self.span[a]-self.span[a]/2
        if self.dim==2:p[...,2]=0
        return p

    def epsilon(self, points):
        p=self.wrap(points);result=np.full(p.shape[:-1],self.background,dtype=float)
        for obj,value in self.objects:
            center,size=object_bounds(obj)
            bound=np.all(abs(p-np.array(center))<=np.array(size)/2+1e-12,axis=-1)
            ids=np.flatnonzero(bound.reshape(-1))
            if not len(ids):continue
            v=p.reshape(-1,3)[ids];mask=contains(obj,v[:,0],v[:,1],v[:,2])
            result.reshape(-1)[ids[mask]]=value
        return result

    @staticmethod
    def roots(obj, centers, axis, length):
        r=rotation_matrix(obj);u=(centers-np.asarray(obj.center))@r;v=r[axis]*length
        roots=[]
        def plane(normal,offset):
            denominator=float(np.dot(v,normal))
            if abs(denominator)>np.finfo(float).tiny:
                roots.append((offset-u@normal)/denominator)
        def quad(scale):
            q=u/scale;d=v/scale;a=np.dot(d,d)
            if a==0:return
            b=2*q@d;c=np.sum(q*q,axis=1)-1;disc=b*b-4*a*c
            t=-.5*(b+np.where(b>=0,1.,-1.)*np.sqrt(np.maximum(disc,0)))
            first=t/a;second=np.divide(c,t,out=np.zeros_like(c),where=t!=0)
            roots.extend((np.where(disc>=0,first,np.inf),np.where(disc>=0,second,np.inf)))
        if obj.kind=='rectangle':
            for a in range(3):
                for side in (-1,1):plane(np.eye(3)[a],side*obj.size[a]/2)
        elif obj.kind=='sphere':quad(np.array(radii(obj)))
        else:
            for side in (-1,1):plane(np.array([0.,0.,1.]),side*obj.size[2]/2)
            if obj.kind=='polygon':
                for contour in (obj.vertices,*obj.holes):
                    vertices=np.asarray(contour)
                    for p,q in zip(vertices,np.roll(vertices,-1,axis=0)):
                        n=np.array([q[1]-p[1],p[0]-q[0],0.]);plane(n,float(n[:2]@p))
            else:
                rx,ry,_=radii(obj);quad(np.array([rx,ry,np.inf]))
                if obj.kind=='ring':
                    if obj.inner_radius>0:quad(np.array([obj.inner_radius,obj.inner_radius_2 if obj.make_ellipsoid else obj.inner_radius,np.inf]))
                    if obj.angular_span<360:
                        for angle in (obj.theta_start,obj.theta_start+obj.angular_span):
                            a=np.deg2rad(angle);plane(np.array([-np.sin(a),np.cos(a),0.]),0.)
        return roots

    def line_average(self, centers, axis, length, *, inverse=False, chunk=256, table=None):
        """Average epsilon or its inverse over centered, axis-aligned edges.

        With ``table`` the values are integer labels and rows of the table are averaged."""
        centers=np.asarray(centers,dtype=float).reshape(-1,3)
        if axis>=self.dim or length==0:
            eps=self.epsilon(centers)
            return table[eps.astype(int)] if table is not None else 1/eps if inverse else eps
        result=np.empty(len(centers)) if table is None else np.empty((len(centers),table.shape[1]))
        for start in range(0,len(centers),chunk):
            p=self.wrap(centers[start:start+chunk]);m=len(p)
            knots=[np.full(m,-.5),np.full(m,.5)]
            shifts=(-self.span[axis],0.,self.span[axis]) if axis in self.periodic else (0.,)
            if axis in self.periodic:
                for sign in (-1,1):knots.append((sign*self.span[axis]/2-p[:,axis])/length)
            for obj,_ in self.objects:
                center,size=object_bounds(obj);half=np.array(size)/2;half[axis]+=length/2
                for shift in shifts:
                    q=p.copy();q[:,axis]+=shift
                    if not np.any(np.all(abs(q-np.array(center))<=half+1e-12,axis=1)):continue
                    knots.extend(self.roots(obj,q,axis,length))
            t=np.sort(np.clip(np.stack(knots,axis=1),-.5,.5),axis=1)
            width=np.diff(t,axis=1);mid=(t[:,1:]+t[:,:-1])/2
            points=np.broadcast_to(p[:,None,:],(*mid.shape,3)).copy();points[:,:,axis]+=length*mid
            eps=self.epsilon(points)
            if table is None:result[start:start+m]=np.sum(width*(1/eps if inverse else eps),axis=1)
            else:result[start:start+m]=np.sum(width[...,None]*table[eps.astype(int)],axis=1)
        return result

    def face_average(self, centers, axis, steps, order, normal=None, table=None):
        transverse=[a for a in range(self.dim) if a!=axis]
        if len(transverse)==1:return self.line_average(centers,transverse[0],steps[transverse[0]],table=table)
        a,b=transverse;nodes,weights=np.polynomial.legendre.leggauss(order)
        result=np.zeros(len(centers) if table is None else (len(centers),table.shape[1]))
        # Integrate the direction of strongest interface variation analytically.
        # This gives exact fractions for every axis-aligned layer, including
        # a face whose discontinuity would otherwise lie across Gauss nodes.
        swap=np.zeros(len(centers),bool) if normal is None else abs(normal[:,b])>abs(normal[:,a])
        for mask,inner,outer in ((~swap,a,b),(swap,b,a)):
            if not np.any(mask):continue
            value=np.zeros((np.count_nonzero(mask),)+result.shape[1:])
            for x,w in zip(nodes,weights):
                points=np.array(centers[mask],copy=True);points[:,outer]+=x*steps[outer]/2
                value+=w/2*self.line_average(points,inner,steps[inner],table=table)
            result[mask]=value
        return result

    def volume_average(self, centers, steps, order, normal, table):
        """Label-table average over the axis-aligned cell of ``steps`` centred at each point.

        Gauss-Legendre nodes run along the axis of weakest normal variation and
        face_average integrates the other two, analytically along the strongest."""
        if self.dim==2:return self.face_average(centers,2,steps,order,normal,table)
        nodes,weights=np.polynomial.legendre.leggauss(order)
        result=np.zeros((len(centers),table.shape[1]));weakest=np.argmin(abs(normal),axis=1)
        for axis in range(3):
            mask=weakest==axis
            if not np.any(mask):continue
            for x,w in zip(nodes,weights):
                points=np.array(centers[mask],copy=True);points[:,axis]+=x*steps[axis]/2
                result[mask]+=w/2*self.face_average(points,axis,steps,order,normal[mask],table)
        return result

    def normals_and_candidates(self, points, radius):
        """Closest analytic boundary normal, with 2D normals projected to XY.

        Sharp intersections do not have a unique normal. The constitutive
        construction remains bounded there, while accuracy needs refinement.
        Radius is a conservative enclosing ball of the integration supports.
        """
        points=self.wrap(points);best=np.full(len(points),np.inf);normals=np.zeros_like(points);candidate=np.zeros(len(points),bool)
        shifts=list(product(*[(-self.span[a],0.,self.span[a]) if a in self.periodic else (0.,) for a in range(3)]))
        for obj,_ in self.objects:
            center,size=object_bounds(obj);r=rotation_matrix(obj)
            for shift in shifts:
                p=points+np.asarray(shift)
                ids=np.flatnonzero(np.all(abs(p-np.asarray(center))<=np.asarray(size)/2+radius,axis=1))
                if not len(ids):continue
                u=(p[ids]-np.asarray(obj.center))@r;dist=[];normal=[]
                def surface(distance,n,location):
                    world=location@r.T+np.asarray(obj.center);world_normal=n@r.T
                    norm=np.linalg.norm(world_normal,axis=1)
                    unit=np.divide(world_normal,norm[:,None],out=np.zeros_like(world_normal),where=norm[:,None]>0)
                    valid=np.ones(len(ids),bool)
                    for a in self.periodic:valid&=abs(world[:,a])<=self.span[a]/2+1e-12
                    active=np.ones(len(ids),bool) if self.dim==3 else (np.linalg.norm(world_normal[:,:2],axis=1)>1e-14)|(norm==0)
                    candidate[ids]|=valid&active&(distance<=radius)
                    # Only an exposed surface of the final, clipped unit cell
                    # defines an interface. Oversized periodic objects must not
                    # introduce normals from surfaces outside the unit cell.
                    near=valid&(distance<=radius)
                    exposed=np.zeros(len(ids),bool)
                    if np.any(near):
                        delta=256*np.finfo(float).eps*max(1.,float(np.max(self.span)))
                        exposed[near]=self.epsilon(world[near]+delta*unit[near])!=self.epsilon(world[near]-delta*unit[near])
                    dist.append(np.where(exposed,distance,np.inf));normal.append(n)
                def face(a,extent):
                    n=np.zeros_like(u);n[:,a]=np.where(u[:,a]>=0,1.,-1.)
                    location=u.copy();location[:,a]=n[:,a]*extent
                    surface(abs(abs(u[:,a])-extent),n,location)
                def radial(scale):
                    q=np.sqrt(np.sum((u/scale)**2,axis=1))
                    location=u.copy();finite=np.isfinite(scale)
                    location[:,finite]=np.divide(u[:,finite],q[:,None],out=np.zeros_like(u[:,finite]),where=q[:,None]>0)
                    surface(abs(q-1)*np.min(scale),u/scale**2,location)
                if obj.kind=='rectangle':
                    for a in range(3):face(a,obj.size[a]/2)
                elif obj.kind=='sphere':radial(np.asarray(radii(obj)))
                else:
                    face(2,obj.size[2]/2)
                    if obj.kind=='polygon':
                        for contour in (obj.vertices,*obj.holes):
                            vertices=np.asarray(contour)
                            for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
                                v=b-a;t=np.clip((u[:,:2]-a)@v/(v@v),0,1);delta=u[:,:2]-a-t[:,None]*v
                                n=np.zeros_like(u);n[:,:2]=np.array([v[1],-v[0]])
                                location=u.copy();location[:,:2]=a+t[:,None]*v
                                surface(np.linalg.norm(delta,axis=1),n,location)
                    else:
                        rx,ry,_=radii(obj);radial(np.array([rx,ry,np.inf]))
                        if obj.kind=='ring':
                            if obj.inner_radius>0:radial(np.array([obj.inner_radius,obj.inner_radius_2 if obj.make_ellipsoid else obj.inner_radius,np.inf]))
                            if obj.angular_span<360:
                                for angle in (obj.theta_start,obj.theta_start+obj.angular_span):
                                    a=np.deg2rad(angle);n=np.array([-np.sin(a),np.cos(a),0.]);signed=u@n
                                    surface(abs(signed),np.broadcast_to(n,u.shape),u-signed[:,None]*n)
                ds=np.stack(dist,axis=1);ns=np.stack(normal,axis=1)@r.T
                if self.dim==2:
                    ns[:,:,2]=0;ds=np.where(np.linalg.norm(ns,axis=2)>1e-14,ds,np.inf)
                nearest=np.argmin(ds,axis=1);d=ds[np.arange(len(ids)),nearest];n=ns[np.arange(len(ids)),nearest]
                candidate[ids]|=d<=radius
                better=d<best[ids];normals[ids[better]]=n[better];best[ids[better]]=d[better]
        length=np.linalg.norm(normals,axis=1);normals=np.divide(normals,length[:,None],out=np.zeros_like(normals),where=length[:,None]>0)
        # The unit-cell seam may be a material boundary even where an analytic
        # object has no surface. Detect its actual repeated material jump.
        for a in self.periodic:
            near=abs(points[:,a]+self.span[a]/2)<radius
            ids=np.flatnonzero(near)
            if not len(ids):continue
            left=points[ids].copy();right=left.copy();left[:,a]=-self.span[a]/2-radius/8;right[:,a]=-self.span[a]/2+radius/8
            jump=self.epsilon(left)!=self.epsilon(right);choose=ids[jump];candidate[choose]=True
            normals[choose]=np.eye(3)[a]
        return normals,candidate
