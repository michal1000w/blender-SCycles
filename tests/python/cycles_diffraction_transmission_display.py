# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Transmission grating with a white slit and first-order wavelength markers.

The position references use the exact grating equation at the image centreline;
they are not predictions of order efficiency. No coherent cross-object transport.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import bpy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--quality',choices=['FAST','REALISTIC'],default='FAST')
p.add_argument('--transport',choices=['pt','bdpt','guided','bdpt_guided'],default='pt')
p.add_argument('--samples',type=int,default=4096)
p.add_argument('--render',action='store_true')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
if a.samples<1:p.error('Samples must be positive')
a.output.mkdir(parents=True,exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene;scene.render.engine='CYCLES'
scene.cycles.device='GPU';scene.cycles.samples=a.samples;scene.cycles.seed=11
scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False
scene.cycles.time_limit=0;scene.cycles.use_layer_samples='IGNORE'
scene.cycles.use_bidirectional_path_tracing=a.transport in ['bdpt','bdpt_guided']
scene.cycles.use_guiding=a.transport in ['guided','bdpt_guided']
scene.cycles.sample_clamp_direct=0;scene.cycles.sample_clamp_indirect=0
scene.cycles.max_bounces=8
scene.render.resolution_x=960;scene.render.resolution_y=540;scene.render.resolution_percentage=100
scene.view_settings.view_transform='AgX';scene.view_settings.exposure=0;scene.view_settings.gamma=1
world=bpy.data.worlds.new('Dark background');world.use_nodes=True
world.node_tree.nodes['Background'].inputs['Strength'].default_value=0;scene.world=world

def emission(name,value):
    m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;n.clear()
    e=n.new('ShaderNodeEmission');e.inputs['Color'].default_value=(value,value,value,1)
    o=n.new('ShaderNodeOutputMaterial');m.node_tree.links.new(e.outputs[0],o.inputs['Surface'])
    return m

def rectangle(name,x,y,z,width,height,material):
    bpy.ops.mesh.primitive_plane_add(size=2,location=(x,y,z));o=bpy.context.object;o.name=name
    o.scale=(width/2,height/2,1);o.data.materials.append(material);return o

def text(body,x,y,size):
    curve=bpy.data.curves.new(body,'FONT');curve.body=body;curve.size=size
    o=bpy.data.objects.new(body,curve);scene.collection.objects.link(o)
    o.location=(x,y,.003);o.data.materials.append(label)

white=emission('White slit radiance',3)
frame=emission('Neutral frame',.035);label=emission('Neutral labels',.35)
camera_distance=.6;source_distance=.05;pitch=740;depth=550
rectangle('White source slit',0,0,-source_distance,.003,.10,white)
m=bpy.data.materials.new('Transmission grating');m.use_nodes=True;n=m.node_tree.nodes;n.clear()
node=n.new('ShaderNodeBsdfDiffraction');node.name='Transmission grating';node.quality=a.quality
node.pitch=pitch;node.depth=depth;node.duty_cycle=.5
node.incident_ior=1;node.ridge_ior=1.5;node.ridge_extinction=0
node.groove_ior=1;node.substrate_ior=1;node.substrate_extinction=0
t=n.new('ShaderNodeCombineXYZ');t.inputs['X'].default_value=1
o=n.new('ShaderNodeOutputMaterial');m.node_tree.links.new(t.outputs[0],node.inputs['Tangent'])
m.node_tree.links.new(node.outputs[0],o.inputs['Surface'])
rectangle('Grating panel',0,0,0,.24,.09,m)
for y in [-.047,.047]:rectangle('Horizontal frame',0,y,.002,.246,.004,frame)
for x in [-.122,.122]:rectangle('Vertical frame',x,0,.002,.004,.098,frame)
text('Transmission grating',-.12,.061,.007)
text(f'740 nm pitch / 550 nm relief / {a.quality}',-.12,.052,.004)
text('Centreline wavelength (nm)',-.12,-.061,.004)
references=[]
for wavelength in [450,550,650]:
    lo,hi=0.,.5
    for _ in range(80):
        x=(lo+hi)/2
        momentum=x/math.hypot(x,camera_distance)+x/math.hypot(x,source_distance)
        if momentum<wavelength/pitch:lo=x
        else:hi=x
    x=(lo+hi)/2
    references.append(dict(wavelength_nm=wavelength,positive_order_x_m=-x,negative_order_x_m=x))
    rectangle('Wavelength marker',x,-.052,.002,.0005,.005,label)
    text(str(wavelength),x-.003,-.061,.004)
bpy.ops.object.camera_add(location=(0,0,camera_distance),rotation=(0,0,0))
camera=bpy.context.object;camera.name='Pinhole camera';camera.data.type='PERSP'
camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=36
camera.data.lens=camera_distance*36/.26;camera.data.dof.use_dof=False;scene.camera=camera
scene['test_scope']=__doc__;scene['diffraction_quality']=a.quality
bpy.data.texts.new('READ ME - transmission test').write(__doc__+'\n'+json.dumps(references,indent=2))
prefix=a.output/f'transmission_{a.quality.lower()}_{a.transport}'
scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32'
scene.render.filepath=str(prefix.with_suffix('.exr').resolve())
bpy.ops.wm.save_as_mainfile(filepath=str(prefix.with_suffix('.blend').resolve()))
bpy.ops.wm.open_mainfile(filepath=str(prefix.with_suffix('.blend').resolve()))
scene=bpy.context.scene
assert bpy.data.materials['Transmission grating'].node_tree.nodes['Transmission grating'].quality==a.quality
assert scene.cycles.samples==a.samples and not scene.cycles.use_adaptive_sampling
assert scene.cycles.use_bidirectional_path_tracing==(a.transport in ['bdpt','bdpt_guided'])
assert scene.cycles.use_guiding==(a.transport in ['guided','bdpt_guided'])
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
report=dict(status='saved_not_rendered',scope=__doc__,quality=a.quality,transport=a.transport,
            samples=a.samples,adaptive_sampling=False,denoising=False,seed=11,
            camera_distance_m=camera_distance,source_distance_m=source_distance,
            slit_width_m=.003,slit_height_m=.10,pitch_nm=pitch,depth_nm=depth,duty_cycle=.5,
            first_order_positions=references,serialization_verified=True,
            blend_sha256=digest(prefix.with_suffix('.blend')),script_sha256=digest(Path(__file__)))
prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if a.render:
    preferences=bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type='METAL';preferences.get_devices()
    for d in preferences.devices:d.use=d.type=='METAL'
    if not any(d.use for d in preferences.devices):raise RuntimeError('No Metal device')
    start=time.perf_counter();bpy.ops.render.render(write_still=True)
    report['wall_seconds_including_preparation']=time.perf_counter()-start
    image=bpy.data.images.load(str(prefix.with_suffix('.exr').resolve()),check_existing=False)
    pixels=tuple(image.pixels[:])
    if not pixels or not all(math.isfinite(v) for v in pixels):raise RuntimeError('Invalid EXR pixels')
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_depth='8'
    image.save_render(str(prefix.with_suffix('.preview.png').resolve()),scene=scene)
    report.update(status='rendered_requires_review',exr_sha256=digest(prefix.with_suffix('.exr')),
                  preview_sha256=digest(prefix.with_suffix('.preview.png')),
                  binary_sha256=digest(Path(bpy.app.binary_path)))
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
