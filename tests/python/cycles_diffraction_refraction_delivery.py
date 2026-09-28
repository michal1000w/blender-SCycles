# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Transmission-only grating windows over white line sources; fixed-sample integration fixture."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import hashlib
import bpy
from mathutils import Vector

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--samples',type=int,default=512)
p.add_argument('--transport',choices=['pt','bdpt','guided'],default='pt')
p.add_argument('--osl',action='store_true')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=a.samples;s.cycles.seed=11
s.cycles.use_adaptive_sampling=False;s.cycles.use_denoising=False;s.cycles.time_limit=0
s.cycles.use_bidirectional_path_tracing=a.transport=='bdpt';s.cycles.use_guiding=a.transport=='guided'
s.cycles.shading_system=a.osl;s.cycles.max_bounces=8
s.cycles.sample_clamp_direct=0;s.cycles.sample_clamp_indirect=0
if not a.osl:
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
    for d in prefs.devices:d.use=d.type=='METAL'
    assert any(d.use for d in prefs.devices);s.cycles.device='GPU'
s.render.resolution_x=900;s.render.resolution_y=500;s.render.resolution_percentage=100
s.world=bpy.data.worlds.new('Dim neutral surround');s.world.use_nodes=True
s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.08
s.view_settings.view_transform='AgX'

def emission(name,color,strength):
    m=bpy.data.materials.new(name);m.use_nodes=True;m.node_tree.nodes.clear()
    e=m.node_tree.nodes.new('ShaderNodeEmission');e.inputs['Color'].default_value=(*color,1);e.inputs['Strength'].default_value=strength
    out=m.node_tree.nodes.new('ShaderNodeOutputMaterial');m.node_tree.links.new(e.outputs[0],out.inputs['Surface'])
    return m

board=emission('Neutral line source',(1,1,1),2)
nodes=board.node_tree.nodes;tex=nodes.new('ShaderNodeTexWave');tex.wave_type='BANDS';tex.bands_direction='X';tex.wave_profile='SIN'
tex.inputs['Scale'].default_value=3
coords=nodes.new('ShaderNodeTexCoord');board.node_tree.links.new(coords.outputs['Generated'],tex.inputs['Vector'])
threshold=nodes.new('ShaderNodeMath');threshold.operation='GREATER_THAN';threshold.inputs[1].default_value=.8
board.node_tree.links.new(tex.outputs['Fac'],threshold.inputs[0]);board.node_tree.links.new(threshold.outputs[0],nodes.get('Emission').inputs['Color'])
label_mat=emission('Label',(.7,.75,.8),1)
specs=[('GGX smooth','GGX',0,1),('GGX rough','GGX',.16,1),('Beckmann half coverage','BECKMANN',.16,.5)]
for x,(name,distribution,roughness,coverage) in zip((-2.6,0,2.6),specs):
    bpy.ops.mesh.primitive_plane_add(size=2.2,location=(x,0,1.7),rotation=(math.pi/2,0,0))
    obj=bpy.context.object;obj.name=name
    m=bpy.data.materials.new(name);m.use_nodes=True;m.node_tree.nodes.clear()
    node=m.node_tree.nodes.new('ShaderNodeBsdfRefraction');node.distribution=distribution
    for key,value in {'Roughness':roughness,'IOR':1.5,'Diffraction Weight':coverage,'Diffraction Pitch':1200,'Diffraction Depth':400,'Diffraction Duty Cycle':.41}.items():node.inputs[key].default_value=value
    node.inputs['Color'].default_value=(1,1,1,1)
    t=m.node_tree.nodes.new('ShaderNodeCombineXYZ');t.inputs['X'].default_value=1
    m.node_tree.links.new(t.outputs[0],node.inputs['Tangent'])
    out=m.node_tree.nodes.new('ShaderNodeOutputMaterial');m.node_tree.links.new(node.outputs[0],out.inputs['Surface']);obj.data.materials.append(m)
    bpy.ops.mesh.primitive_plane_add(size=2.2,location=(x,2,1.7),rotation=(math.pi/2,0,0))
    bpy.context.object.name=name+' line source';bpy.context.object.data.materials.append(board)
    bpy.ops.object.text_add(location=(x,-.1,.3),rotation=(math.pi/2,0,0))
    text=bpy.context.object;text.data.body=name;text.data.align_x='CENTER';text.data.size=.14;text.data.materials.append(label_mat)
bpy.ops.object.camera_add(location=(0,-9,2.5));s.camera=bpy.context.object
s.camera.rotation_euler=(Vector((0,0,1.6))-s.camera.location).to_track_quat('-Z','Y').to_euler()
s.camera.data.type='ORTHO';s.camera.data.ortho_scale=8.3
notes=bpy.data.texts.new('READ ME - transmission grating test')
notes.write('Each window is a single Refraction BSDF interface with IOR 1.5, not a complete glass sheet.\nThe source is neutral white; colors arise from spectral order displacement.\nNo Fresnel weighting; unavailable orders retain a null event rather than being renormalized.\nUse Glass for a coupled reflective/transmissive interface.\n')
s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32';s.render.filepath=str(a.output/'refraction.exr')
bpy.ops.wm.save_as_mainfile(filepath=str(a.output/'refraction.blend'))
start=time.monotonic();bpy.ops.render.render(write_still=True);seconds=time.monotonic()-start
image=bpy.data.images.load(str(a.output/'refraction.exr'),check_existing=False);pixels=tuple(image.pixels[:]);assert pixels and all(math.isfinite(v) for v in pixels)
mean=[sum(pixels[i::image.channels])/(len(pixels)//image.channels) for i in range(3)];bpy.data.images.remove(image)
s.render.image_settings.file_format='PNG';s.render.image_settings.color_depth='8';bpy.data.images['Render Result'].save_render(str(a.output/'refraction.png'),scene=s)
report={'samples':a.samples,'adaptive_sampling':False,'denoising':False,'transport':a.transport,'osl':a.osl,'render_seconds':seconds,'all_pixels_finite':True,'mean_rgb':mean,'materials':specs,'binary_sha256':hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),'scope':'Single-interface Refraction node integration; not a full glass sheet or Maxwell comparison.'}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
