# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Embedded optical-constant table, serialization, spectral furnace and refresh checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import bpy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--csv',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',choices=['CPU','METAL'],default='CPU')
p.add_argument('--quality',choices=['FAST','REALISTIC'],default='FAST')
p.add_argument('--transport',choices=['pt','bdpt','guided','bdpt_guided'],default='pt')
p.add_argument('--depth',type=float,default=150)
p.add_argument('--samples',type=int,default=1024)
p.add_argument('--osl',action='store_true')
p.add_argument('--refresh-test',action='store_true')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
if a.samples<1 or a.depth<0:p.error('Invalid samples or depth')
if a.osl and a.device!='CPU':p.error('OSL test requires CPU')
if a.quality=='REALISTIC' and a.depth!=0:p.error('Fresnel reference applies to flat Realistic conductors')
a.output.mkdir(parents=True,exist_ok=False)
bpy.ops.wm.open_mainfile(filepath=str(a.scene.resolve()))
scene=bpy.context.scene;scene.cycles.samples=a.samples;scene.cycles.seed=11
scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False
scene.cycles.time_limit=0;scene.cycles.use_layer_samples='IGNORE'
scene.cycles.shading_system=a.osl;scene.cycles.device='CPU'
scene.cycles.use_guiding=a.transport in ('guided','bdpt_guided')
scene.cycles.use_bidirectional_path_tracing=a.transport in ('bdpt','bdpt_guided')
scene.render.use_persistent_data=True
if a.device=='METAL':
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='METAL';prefs.get_devices()
    for d in prefs.devices:d.use=d.type=='METAL'
    assert any(d.use for d in prefs.devices);scene.cycles.device='GPU'
material=bpy.data.materials['Physical grating measurement'];tree=material.node_tree
node=next(n for n in tree.nodes if n.bl_idname=='ShaderNodeBsdfDiffraction')
node.quality=a.quality;node.depth=a.depth
contents=a.csv.read_text()
table=bpy.data.texts.new('Embedded conductor optical constants');table.write(contents)
node.optical_constants=table
reference=json.loads(a.reference.read_text())
expected=reference['runs'][-1]['rgb']
quadrature=max(abs(x-y) for x,y in zip(expected,reference['runs'][-2]['rgb']))
assert quadrature<1e-6
scene['test_scope']=__doc__;scene['optical_constants_source']=a.csv.name
bpy.ops.wm.save_as_mainfile(filepath=str((a.output/'furnace.blend').resolve()))
bpy.ops.wm.open_mainfile(filepath=str((a.output/'furnace.blend').resolve()))
scene=bpy.context.scene;material=bpy.data.materials['Physical grating measurement'];tree=material.node_tree
node=next(n for n in tree.nodes if n.bl_idname=='ShaderNodeBsdfDiffraction')
assert node.optical_constants.as_string()==contents and node.quality==a.quality
assert node.optical_constants.users>0
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
report=dict(scope=__doc__,status='running',quality=a.quality,device=a.device,osl=a.osl,
 transport=a.transport,depth_nm=a.depth,samples=a.samples,adaptive_sampling=False,denoising=False,
 serialization_verified=True,quadrature_difference=quadrature,
 csv_sha256=sha(a.csv),reference_sha256=sha(a.reference),script_sha256=sha(Path(__file__)),
 binary_sha256=sha(Path(bpy.app.binary_path)),blend_sha256=sha(a.output/'furnace.blend'),runs=[])
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
def render(name,target):
    assert scene.cycles.samples==a.samples and not scene.cycles.use_adaptive_sampling
    scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32'
    scene.render.filepath=str((a.output/(name+'.exr')).resolve())
    bpy.ops.render.render(write_still=True)
    image=bpy.data.images.load(scene.render.filepath,check_existing=False);pixels=tuple(image.pixels[:])
    assert pixels and all(math.isfinite(x) for x in pixels)
    mean=[sum(pixels[c::4])/(len(pixels)//4) for c in range(3)]
    error=max(abs(x-y) for x,y in zip(mean,target))
    report['runs'].append(dict(name=name,mean_rgb=mean,reference_rgb=target,maximum_error=error,
      tolerance=.002,exr_sha256=sha(Path(scene.render.filepath)),passed=error<.002))
    save()
    if error>=.002:raise RuntimeError('Spectral furnace differs from independent reference')
save();render('furnace',expected)
if a.refresh_test:
    node.optical_constants.clear();node.optical_constants.write('wavelength_nm,n,k\n380,0.9,6\n780,0.9,6\n')
    area=bpy.context.screen.areas[0];area.type='NODE_EDITOR'
    space=area.spaces.active;space.tree_type='ShaderNodeTree';space.pin=True;space.node_tree=tree
    tree.nodes.active=node
    with bpy.context.temp_override(area=area,node=node):
        result=bpy.ops.node.diffraction_table_update()
    assert result=={'FINISHED'}
    # The constant-table target uses independent Fresnel reflectance. The 0.002
    # smoke tolerance includes the known finite-sample white spectral residual.
    R=((.9-1)**2+6**2)/((.9+1)**2+6**2)
    render('refreshed_constant',[R]*3)
    report['refresh_operator_verified']=True
    bpy.ops.wm.save_as_mainfile(filepath=str((a.output/'refreshed.blend').resolve()))
    report['refreshed_blend_sha256']=sha(a.output/'refreshed.blend')
report['status']='passed';save();print(json.dumps(report))
