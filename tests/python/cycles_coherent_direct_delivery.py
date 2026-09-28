# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Actual coherent point-light controls, compared to an independent scalar reference.

This tests direct source interference on a diffuse detector, not reflected-path
interference, finite-aperture wave diffraction, or polarization transport.
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--transport',choices=['pt','bdpt','guided','bdpt_guided'],default='pt')
p.add_argument('--samples',type=int,default=128)
p.add_argument('--guiding-training-samples',type=int)
p.add_argument('--case',choices=['all','phase_0','phase_pi','incoherent','partial','red','occluded','three_sources','unit_scale','off'],default='all')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=a.samples;s.cycles.seed=11
s.cycles.use_adaptive_sampling=False;s.cycles.use_denoising=False
s.cycles.use_bidirectional_path_tracing=a.transport in ('bdpt','bdpt_guided');s.cycles.use_guiding=a.transport in ('guided','bdpt_guided')
if a.guiding_training_samples is not None:s.cycles.guiding_training_samples=a.guiding_training_samples
s.cycles.max_bounces=4;s.cycles.sample_clamp_direct=0;s.cycles.sample_clamp_indirect=0
s.cycles.pixel_filter_type='BOX';s.cycles.filter_width=1
s.render.resolution_x=512;s.render.resolution_y=256;s.render.resolution_percentage=100
s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32'
s.view_settings.view_transform='Standard'
pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='METAL';pref.get_devices()
for d in pref.devices:d.use=d.type=='METAL'
assert any(d.use for d in pref.devices);s.cycles.device='GPU'
s.world=bpy.data.worlds.new('Black environment');s.world.use_nodes=True
s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0
bpy.ops.mesh.primitive_plane_add(size=1,location=(0,1,0),rotation=(math.pi/2,0,0))
detector=bpy.context.object;detector.name='Scalar diffuse detector';detector.scale=(.08,.04,1)
m=bpy.data.materials.new('White diffuse');m.use_nodes=True;n=m.node_tree.nodes;n.clear()
d=n.new('ShaderNodeBsdfDiffuse');d.inputs['Color'].default_value=(1,1,1,1)
o=n.new('ShaderNodeOutputMaterial');m.node_tree.links.new(d.outputs[0],o.inputs['Surface']);detector.data.materials.append(m)
sources=[]
for label,x in [('A',-50e-6),('B',50e-6)]:
 bpy.ops.object.light_add(type='POINT',location=(x,0,0));ob=bpy.context.object;ob.name='Coherent source '+label
 ob.data.energy=10;ob.data.shadow_soft_size=0;ob.data.cycles.coherence_group=1;sources.append(ob)
bpy.ops.object.camera_add(location=(0,.5,0));s.camera=bpy.context.object
s.camera.rotation_euler=Vector((0,1,0)).to_track_quat('-Z','Y').to_euler();s.camera.data.type='ORTHO';s.camera.data.ortho_scale=.04
s.camera.data.clip_start=.001;s.camera.data.clip_end=10
cases={'phase_0':(0,532,1),'phase_pi':(math.pi,532,1),'incoherent':(0,532,0),'partial':(0,532,1e-6),'red':(0,633,1),'occluded':(0,532,1),'three_sources':(0,532,1),'unit_scale':(0,532,1),'off':(0,532,0)}
manifest={'binary':bpy.app.binary_path,'sha256':hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
 'scope':'Actual direct scalar coherent point-source controls; not general coherent multipath',
 'transport':a.transport,'samples':a.samples,'adaptive_sampling':False,'denoising':False,
 'use_bidirectional_path_tracing':s.cycles.use_bidirectional_path_tracing,
 'use_guiding':s.cycles.use_guiding,'guiding_training_samples':s.cycles.guiding_training_samples,'runs':{}}
failed=False
for name,(phase,wavelength_nm,lc) in cases.items():
 if a.case!='all' and name!=a.case:continue
 s.unit_settings.scale_length=.5 if name=='unit_scale' else 1
 blocker=None;third=None
 if name=='occluded':
  bpy.ops.mesh.primitive_cube_add(size=1,location=(-50e-6,.001,0))
  blocker=bpy.context.object;blocker.name='Opaque mask blocking source A';blocker.dimensions=(60e-6,10e-6,60e-6)
 if name=='three_sources':
  bpy.ops.object.light_add(type='POINT',location=(0,0,0));third=bpy.context.object;third.name='Coherent source C'
  third.data.energy=10;third.data.shadow_soft_size=0;third.data.cycles.coherence_group=1;sources.append(third)
 for i,ob in enumerate(sources):
  ob.data.cycles.coherence_group=0 if name=='off' else 1
  ob.data.cycles.coherence_phase=phase if i else 0
  ob.data.cycles.coherence_wavelength_nm=wavelength_nm
  ob.data.cycles.coherence_length_m=lc
 s.render.filepath=str(a.output/(name+'.exr'));bpy.ops.wm.save_as_mainfile(filepath=str(a.output/(name+'.blend')))
 t=time.monotonic();bpy.ops.render.render(write_still=True);elapsed=time.monotonic()-t
 im=bpy.data.images.load(s.render.filepath,check_existing=False);pixels=np.array(im.pixels[:],dtype=np.float64).reshape((256,512,im.channels));bpy.data.images.remove(im)
 assert np.isfinite(pixels).all()
 # Independent float64 scalar reference, box-filtered by 4x4 subpixel quadrature.
 # L_i=P_i*cos(theta_i)/(4*pi^2*r_i^2) for a unit Lambertian detector.
 expected=np.zeros((256,512));dx=.04/512;dz=.02/256
 positions=[np.array(tuple(ob.location),dtype=np.float64) for ob in sources]
 for u in (.125,.375,.625,.875):
  for v in (.125,.375,.625,.875):
   x=(np.arange(512)+u-256)*dx;z=(np.arange(256)+v-128)*dz
   X,Z=np.meshgrid(x,z)
   radii=[np.sqrt((X-A[0])**2+(1-A[1])**2+(Z-A[2])**2) for A in positions]
   irradiances=[10*(1-A[1])/(4*math.pi**2*r**3) for A,r in zip(positions,radii)]
   if name=='occluded':irradiances[0]=np.zeros_like(X)
   intensity=sum(irradiances)
   unit=float(s.unit_settings.scale_length);wave=wavelength_nm*1e-9/unit;length=lc/unit
   for ia,A in enumerate(positions):
    for ib in range(ia+1,len(positions)):
     B=positions[ib]
     delta=((B[0]-A[0])*(2*X-A[0]-B[0])+(B[1]-A[1])*(2-A[1]-B[1])+(B[2]-A[2])*(2*Z-A[2]-B[2]))/(radii[ia]+radii[ib])
     gamma=np.exp(-.5*(delta/length)**2) if length else np.zeros_like(delta)
     phase_difference=(phase if ia else 0)-(phase if ib else 0)
     intensity+=2*np.sqrt(irradiances[ia]*irradiances[ib])*gamma*np.cos(2*math.pi*delta/wave+phase_difference)
   expected+=intensity/16
 actual=pixels[:,:,:3].mean(axis=2);error=actual-expected
 peak=float(expected.max());rmse=float(np.sqrt(np.mean(error**2)));mae=float(np.mean(np.abs(error)))
 # Predeclared absolute radiance gates account for stochastic pixel filtering.
 passed=rmse<.006 and abs(float(actual.mean()-expected.mean()))<.003
 entry={'phase_radians':phase,'wavelength_nm':wavelength_nm,'coherence_length_m':lc,
 'scene_unit_scale':float(s.unit_settings.scale_length),'source_count':len(sources),'opaque_mask':name=='occluded',
 'render_seconds':elapsed,'all_finite':True,'reference_mean':float(expected.mean()),'render_mean':float(actual.mean()),
 'absolute_rmse':rmse,'absolute_mae':mae,'maximum_absolute_error':float(np.abs(error).max()),'reference_peak':peak,
 'gate':'RMSE<0.006 and absolute mean error<0.003; independent scalar Lambertian reference, no fitted brightness scale',
 'passed':passed}
 np.savez_compressed(a.output/(name+'_reference.npz'),reference=expected,render=actual)
 s.render.image_settings.file_format='PNG';s.render.image_settings.color_depth='8'
 bpy.data.images['Render Result'].save_render(str(a.output/(name+'.png')),scene=s)
 s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32'
 manifest['runs'][name]=entry;(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(name,json.dumps(entry),flush=True);failed|=not passed
 if blocker is not None:bpy.data.objects.remove(blocker,do_unlink=True)
 if third is not None:sources.pop();bpy.data.objects.remove(third,do_unlink=True)
if failed:raise RuntimeError('Coherent direct numerical acceptance failed; inspect preserved manifest and raw images')
