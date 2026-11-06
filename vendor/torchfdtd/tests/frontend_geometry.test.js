import assert from 'node:assert/strict';
import test from 'node:test';
import {Vector3} from 'three';
import {structureGeometry,rotationMatrix,hitProjected} from '../frontend/src/geometry.js';

const shape=(extra)=>({kind:'polygon',size:[1,1,.6],radius:.5,inner_radius:.3,center:[0,0,0],vertices:[[-.5,-.4],[.5,-.4],[.5,0],[0,0],[0,.4],[-.5,.4]],...extra});
function volume(g){const f=g.index?g.toNonIndexed():g,p=f.attributes.position;let total=0;for(let i=0;i<p.count;i+=3){const a=new Vector3().fromBufferAttribute(p,i),b=new Vector3().fromBufferAttribute(p,i+1),c=new Vector3().fromBufferAttribute(p,i+2);total+=a.dot(b.cross(c))/6;}return Math.abs(total);}
test('noncommuting world rotations preserve the declared order',()=>{
 const p=new Vector3(1,0,0).applyMatrix4(rotationMatrix({rotation_axes:['z','x','none'],rotation_angles:[90,90,37]}));
 assert.ok(p.distanceTo(new Vector3(0,0,1))<1e-15);
});
test('concave CAD contour, thickness and hit testing preserve the missing corner',()=>{
 const p=shape({}),g=structureGeometry(p);assert.ok(Math.abs(volume(g)-.36)<1e-7);
 assert.equal(hitProjected(p,[.25,.2],[0,1],.001),false);
 assert.equal(hitProjected(p,[-.25,.2],[0,1],.001),true);
 assert.equal(hitProjected(p,[.25,-.2],[0,1],.001),true);
});
test('ellipse and sector display volumes agree with analytic solids',()=>{
 const samples=[
  [shape({kind:'sphere',make_ellipsoid:true,radius_2:.3,radius_3:.2}),4/3*Math.PI*.5*.3*.2],
  [shape({kind:'circle',make_ellipsoid:true,radius_2:.3}),Math.PI*.5*.3*.6],
  [shape({kind:'ring',theta_start:315,theta_stop:90}),135/360*Math.PI*(.5**2-.3**2)*.6]];
 for(const [p,exact] of samples)assert.ok(Math.abs(volume(structureGeometry(p))/exact-1)<.008);
 const ring=shape({kind:'ring',theta_start:0,theta_stop:90});
 assert.equal(hitProjected(ring,[.28,.28],[0,1],.001),true);
 assert.equal(hitProjected(ring,[-.28,.28],[0,1],.001),false);
 assert.equal(hitProjected(ring,[0,0],[0,1],.001),false);
});
