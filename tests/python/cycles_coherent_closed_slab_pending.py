#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent primary-TT closed parallel slab oracle and pending real fixture.
No renderer implementation and no render calls. --author OUTPUT inside Blender.
"""
import math,json,sys
from pathlib import Path

def ray(r,d=(.01,.01,.01),n=(1.,1.5,1.)):
 def radius(p):return sum(z*p/math.sqrt(v*v-p*p) for z,v in zip(d,n))
 lo,hi=0.,math.nextafter(min(n),0.)
 for _ in range(100):
  p=(lo+hi)/2
  if radius(p)<r:lo=p
  else:hi=p
 p=(lo+hi)/2;derivative=sum(z*v*v/(v*v-p*p)**1.5 for z,v in zip(d,n));D=sum(z/v for z,v in zip(d,n));jac=1/D**2 if r==0 else p/(math.sqrt(1-p*p)*r*derivative)
 cos=[math.sqrt(1-(p/v)**2) for v in n];Ts=Tp=1.
 for i in range(2):
  a,b=n[i]*cos[i],n[i+1]*cos[i+1];rs=(a-b)/(a+b);rp=(n[i+1]*cos[i]-n[i]*cos[i+1])/(n[i+1]*cos[i]+n[i]*cos[i+1]);Ts*=1-rs*rs;Tp*=1-rp*rp
 return {'p':p,'OPL':sum(v*z/c for v,z,c in zip(n,d,cos)),'Jacobian':jac,'Ts':Ts,'Tp':Tp,'single_radiance':.001/(4*math.pi**2)*jac*(Ts+Tp)/2}

def check():
 for r in [0.,.003,.01,.03]:
  q=ray(r);h=1e-6;p=q['p'];d=(.01,.01,.01);n=(1.,1.5,1.);radius=lambda x:sum(z*x/math.sqrt(v*v-x*x) for z,v in zip(d,n));fd=(radius(p+h)-radius(p-h))/(2*h)
  exact=sum(z*v*v/(v*v-p*p)**1.5 for z,v in zip(d,n));assert abs(fd/exact-1)<1e-8
  if r>0:
   annulus=(math.sqrt(1-(p-h)**2)-math.sqrt(1-(p+h)**2))*2/(radius(p+h)**2-radius(p-h)**2);assert abs(annulus/q['Jacobian']-1)<1e-7
  # Homogeneous medium independently reduces to cos(theta)/distance^2.
  u=ray(r,n=(1.,1.,1.));expected=.03/(.03**2+r*r)**1.5;assert abs(u['Jacobian']/expected-1)<1e-12
  # Fermat: transverse displacement derivative of OPL equals conserved p.
  if r>0:assert abs((ray(r+h)['OPL']-ray(r-h)['OPL'])/(2*h)-p)<1e-8
 assert abs(ray(0)['Jacobian']-1/(.01+.01/1.5+.01)**2)<1e-10
 return {'checks':'Snell conserved transverse p, central normal limit, independent homogeneous inverse-square limit, finite-difference area Jacobian and Fermat OPL derivative','off_axis':ray(.01),'scope':'Primary TT through closed slab, max2interfaces; internal reflection histories excluded by declared budget. Pending production; not actual acceptance.'}

def author(output):
 import bpy,numpy as np
 from mathutils import Vector
 sys.path.insert(0,str(Path(__file__).resolve().parent))
 from cycles_coherent_mirror_acceptance import configure_scene,set_diffuse_material
 from cycles_coherent_polarizer_scene import pane,glass
 out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);scene,*_=configure_scene(True)
 for o in list(bpy.data.objects):bpy.data.objects.remove(o,do_unlink=True)
 scene.cycles.coherent_transport_mode='FACET_SINGLE_REFLECTION';scene.cycles.use_coherent_specular_connections=True;scene.cycles.coherent_max_interface_events=2;scene.cycles.coherent_polarization_mode='VECTOR';scene.cycles.diffuse_bounces=0;scene.cycles.transmission_bounces=2;scene.cycles.glossy_bounces=2;scene.cycles.bdpt_max_bounces=3;scene.cycles.max_bounces=3;scene.cycles.device='CPU';scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False;scene.render.resolution_x=scene.render.resolution_y=16
 mat,_=glass('Physical n1.5 closed slab',1.5);bpy.ops.mesh.primitive_cube_add(size=1,location=(0,0,.015));slab=bpy.context.object;slab.name='One closed Glass object entry and exit';slab.dimensions=(.1,.1,.01);slab.data.materials.append(mat);slab.cycles.coherent_interface='GLASS';bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
 receiver=np.array([.01,0,.03]);detector=pane('Unit detector',receiver,(1,0,0),(0,-1,0),.002,set_diffuse_material('Unit detector',(1,1,1)),'DETECTOR')
 sources=[]
 for i in range(2):
  bpy.ops.object.light_add(type='POINT',location=(0,0,0));l=bpy.context.object;l.data.energy=.001;l.data.shadow_soft_size=0;l.data.cycles.coherence_group=1;l.data.cycles.coherence_wavelength_nm=550;l.data.cycles.coherence_length_m=1;sources.append(l)
 bpy.ops.object.camera_add(location=receiver+[0,0,-.004]);camera=bpy.context.object;camera.rotation_euler=(Vector(receiver)-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=.000002;camera.data.clip_start=.001;scene.camera=camera
 bpy.context.view_layer.update();world=[slab.matrix_world@v.co for v in slab.data.vertices];zmin=min(v.z for v in world);zmax=max(v.z for v in world);receiver=np.array(detector.location,dtype=float);thickness=(zmin,zmax-zmin,float(receiver[2])-zmax);power_scale=float(sources[0].data.energy)/.001
 report=check();report['specular_connections']=True;report['scope']='Pending streamed closedconvex exterior-air primaryTT; opposite-side slab geometry excludes direct/exteriorR/RR';report['pending_transport_api']=True;report['diffuse_bounces']=0;report['glossy_bounces']=2;report['transmission_bounces']=2;report['maximum_interface_events']=2;report['gates_declared_before_render']={'absolute_mean_error':1e-5,'absolute_rmse':2e-5};report['exact_saved_layer_thickness_m']=thickness;report['source_power_W']=float(sources[0].data.energy);report['reference_sampling']='2x2BOX exactsavedcamera ray-plane sampling; independent4x4 comparison; strict prospective gates preregistered';report['expected_host_status']='Unsupported declared Glass in streamed mirror mode; rejection before kernel, not physics failure';report['variants']={};arr={}
 for name,phase,groups,factor in [('phase_0',0.,[1,1],4.),('phase_pi',math.pi,[1,1],0.),('distinct',0.,[1,2],2.)]:
  for l,g in zip(sources,groups):l.data.cycles.coherence_group=g
  sources[1].data.cycles.coherence_phase=phase;actual=float(sources[1].data.cycles.coherence_phase);factor=2+2*math.cos(actual) if groups==[1,1] else 2
  matrix=np.array([list(row) for row in camera.matrix_world]);right=matrix[:3,0];up=matrix[:3,1];width=float(camera.data.ortho_scale)
  def pixels(order,wrong=None):
   result=np.zeros((16,16));offsets=(np.arange(order)+.5)/order
   for row in range(16):
    for col in range(16):
     for ox in offsets:
      for oy in offsets:
       point=receiver+((col+ox)/16-.5)*width*right+((row+oy)/16-.5)*width*up;r=math.hypot(point[0],point[1]);q=ray(r,d=thickness,n=(1.,1.,1.) if wrong=='IOR1' else (1.,1.5,1.));value=q['single_radiance']
       if wrong=='vacuumJacobian':value*=ray(r,d=thickness,n=(1.,1.,1.))['Jacobian']/q['Jacobian']
       result[row,col]+=factor*power_scale*value/order**2
   return result
  arr[name]=pixels(2);high=pixels(4);wrongior=pixels(2,'IOR1');wrongjac=pixels(2,'vacuumJacobian');report.setdefault('prospective_controls',{})[name]={'4x4_vs2x2_rmse':float(np.sqrt(np.mean((high-arr[name])**2))),'omittedTT_rmse':float(np.sqrt(np.mean(arr[name]**2))),'wrongIOR1_rmse':float(np.sqrt(np.mean((wrongior-arr[name])**2))),'vacuumJacobian_rmse':float(np.sqrt(np.mean((wrongjac-arr[name])**2)))}
  p=out/(name+'.blend');bpy.ops.wm.save_as_mainfile(filepath=str(p));report['variants'][name]={'blend':str(p),'path':str(p),'sha256':__import__('hashlib').sha256(p.read_bytes()).hexdigest(),'source_phases_rad':[0.,actual],'factor':factor,'phase':actual,'groups':groups}
 rows,cols=np.indices((16,16));roi=(rows>=2)&(rows<14)&(cols>=2)&(cols<14);np.savez_compressed(out/'primary_TT_reference.npz',roi_mask=roi,phase_0_radiance=arr['phase_0'],phase_pi_radiance=arr['phase_pi'],incoherent_radiance=arr['distinct']);(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
 if '--author' in sys.argv:author(sys.argv[sys.argv.index('--author')+1])
 else:print(json.dumps(check(),indent=2))
