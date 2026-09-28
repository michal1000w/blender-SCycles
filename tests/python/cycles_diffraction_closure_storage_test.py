# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Lossless furnace with eight Fast closures: verifies graph storage accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',choices=['CPU','METAL'],default='CPU')
p.add_argument('--transport',choices=['pt','bdpt','guided','bdpt_guided'],default='pt')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
a.output.mkdir(parents=True,exist_ok=False)
bpy.ops.wm.open_mainfile(filepath=str(a.scene.resolve()))
scene=bpy.context.scene;scene.cycles.device='CPU';scene.cycles.shading_system=False
scene.cycles.samples=256;scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False
scene.cycles.time_limit=0;scene.cycles.use_layer_samples='IGNORE'
scene.cycles.use_guiding=a.transport in ('guided','bdpt_guided')
scene.cycles.use_bidirectional_path_tracing=a.transport in ('bdpt','bdpt_guided')
if a.device=='METAL':
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='METAL';prefs.get_devices()
    for device in prefs.devices:device.use=device.type=='METAL'
    assert any(d.use for d in prefs.devices);scene.cycles.device='GPU'
material=bpy.data.materials['Physical grating measurement'];tree=material.node_tree
original=next(n for n in tree.nodes if n.bl_idname=='ShaderNodeBsdfDiffraction')
props={name:getattr(original,name) for name in ('pitch','depth','duty_cycle','incident_ior',
 'ridge_ior','ridge_extinction','groove_ior','substrate_ior','substrate_extinction')}
assert props['substrate_ior']==1 and props['substrate_extinction']==0
assert props['ridge_extinction']==0 and original.quality=='FAST'
tangent=original.inputs['Tangent'].links[0].from_socket
output=next(n for n in tree.nodes if n.bl_idname=='ShaderNodeOutputMaterial')
tree.nodes.remove(original)
level=[]
for i in range(8):
    node=tree.nodes.new('ShaderNodeBsdfDiffraction');node.quality='FAST'
    for name,value in props.items():setattr(node,name,value)
    # Distinct pitches prevent graph deduplication; every closure is lossless.
    node.pitch=props['pitch']+i*10
    tree.links.new(tangent,node.inputs['Tangent']);level.append(node.outputs[0])
while len(level)>1:
    next_level=[]
    for i in range(0,len(level),2):
        mix=tree.nodes.new('ShaderNodeMixShader');mix.inputs[0].default_value=.5
        tree.links.new(level[i],mix.inputs[1]);tree.links.new(level[i+1],mix.inputs[2])
        next_level.append(mix.outputs[0])
    level=next_level
tree.links.new(level[0],output.inputs['Surface'])
scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32'
scene.render.filepath=str((a.output/'furnace.exr').resolve())
bpy.ops.wm.save_as_mainfile(filepath=str((a.output/'furnace.blend').resolve()))
bpy.ops.render.render(write_still=True)
image=bpy.data.images.load(scene.render.filepath,check_existing=False);pixels=tuple(image.pixels[:])
mean=[sum(pixels[c::4])/(len(pixels)//4) for c in range(3)]
error=max(abs(x-1) for x in mean)
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
report=dict(status='passed' if error<.01 else 'failed',scope=__doc__,device=a.device,transport=a.transport,
 closure_count=8,required_storage_slots=16,samples=256,adaptive_sampling=False,denoising=False,
 raw_mean_rgb=mean,maximum_unit_energy_error=error,tolerance=.01,
 binary_sha256=digest(Path(bpy.app.binary_path)),script_sha256=digest(Path(__file__)),
 blend_sha256=digest(a.output/'furnace.blend'),exr_sha256=digest(a.output/'furnace.exr'))
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if error>=.01:raise RuntimeError('Lossless closure mixture lost energy')
