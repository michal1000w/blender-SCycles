# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Known-zero relief must preserve native carriers, including Multi-GGX."""
import argparse,hashlib,json,sys
from pathlib import Path
import bpy
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['CPU','METAL'],default='CPU');p.add_argument('--osl',action='store_true');a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=16;s.cycles.seed=11;s.cycles.use_adaptive_sampling=False;s.cycles.use_denoising=False;s.cycles.shading_system=a.osl;s.cycles.device='CPU';s.render.resolution_x=32;s.render.resolution_y=32;s.render.resolution_percentage=100
if a.device=='METAL':
 assert not a.osl
 pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='METAL';pref.get_devices()
 for d in pref.devices:d.use=d.type=='METAL'
 assert any(d.use for d in pref.devices);s.cycles.device='GPU'
s.world=bpy.data.worlds.new('Unit environment');s.world.use_nodes=True;s.world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=1
bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12);obj=bpy.context.object
bpy.ops.object.camera_add(location=(0,0,4));s.camera=bpy.context.object;s.camera.data.type='ORTHO';s.camera.data.ortho_scale=2.5
mat=bpy.data.materials.new('Native flat control');mat.use_nodes=True;obj.data.materials.append(mat)
s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32'
report={'binary_sha256':hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),'device':a.device,'osl':a.osl,'samples':16,'adaptive_sampling':False,'denoising':False,'scope':'Exact native zero-depth limit, not nonzero MultiGGX diffraction','cases':{}}
for name,kind in [('glossy','ShaderNodeBsdfAnisotropic'),('metallic','ShaderNodeBsdfMetallic'),('glass','ShaderNodeBsdfGlass'),('principled','ShaderNodeBsdfPrincipled'),('refraction','ShaderNodeBsdfRefraction')]:
 mat.node_tree.nodes.clear();node=mat.node_tree.nodes.new(kind);node.distribution='GGX' if name=='refraction' else 'MULTI_GGX';node.inputs['Roughness'].default_value=.55
 out=mat.node_tree.nodes.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(node.outputs[0],out.inputs['Surface']);images=[]
 for variant in ['disabled','zero_depth','zero_depth_linked_weight']:
  node.inputs['Diffraction Depth'].default_value=150 if variant=='disabled' else 0;node.inputs['Diffraction Weight'].default_value=0 if variant=='disabled' else 1
  if variant=='zero_depth_linked_weight':
   noise=mat.node_tree.nodes.new('ShaderNodeTexNoise');mat.node_tree.links.new(noise.outputs['Fac'],node.inputs['Diffraction Weight'])
  stem=name+'_'+variant;s.render.filepath=str(a.output/(stem+'.exr'));bpy.ops.render.render(write_still=True)
  im=bpy.data.images.load(s.render.filepath,check_existing=False);pixels=np.array(im.pixels[:],dtype=np.float64);bpy.data.images.remove(im);assert np.isfinite(pixels).all();images.append(pixels)
 errors=[float(np.max(np.abs(v-images[0]))) for v in images[1:]];report['cases'][name]={'max_absolute_errors':errors,'gate':1e-6,'passed':all(e<1e-6 for e in errors)}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
assert all(v['passed'] for v in report['cases'].values()),report
print(json.dumps(report,indent=2))
