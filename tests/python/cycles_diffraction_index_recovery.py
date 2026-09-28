# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Invalid embedded optical constants must fail visibly, then recover after refresh."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
a.output.mkdir(parents=True,exist_ok=False)
bpy.ops.wm.open_mainfile(filepath=str(a.scene.resolve()))
scene=bpy.context.scene;scene.cycles.device='CPU';scene.cycles.shading_system=False
scene.cycles.samples=64;scene.cycles.seed=11;scene.cycles.use_adaptive_sampling=False
scene.cycles.use_denoising=False;scene.cycles.use_guiding=False
scene.cycles.use_bidirectional_path_tracing=False;scene.cycles.time_limit=0
scene.cycles.use_layer_samples='IGNORE';scene.render.use_persistent_data=True
material=bpy.data.materials['Physical grating measurement'];tree=material.node_tree
node=next(n for n in tree.nodes if n.bl_idname=='ShaderNodeBsdfDiffraction')
node.quality='FAST';original=node.optical_constants.as_string()
area=bpy.context.screen.areas[0];area.type='NODE_EDITOR'
space=area.spaces.active;space.tree_type='ShaderNodeTree';space.pin=True;space.node_tree=tree
tree.nodes.active=node
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
report=dict(scope=__doc__,status='running',binary_sha256=sha(Path(bpy.app.binary_path)),
 script_sha256=sha(Path(__file__)),source_scene_sha256=sha(a.scene),samples=64,
 adaptive_sampling=False,denoising=False)
def refresh(text):
    node.optical_constants.clear();node.optical_constants.write(text)
    with bpy.context.temp_override(area=area,node=node):
        assert bpy.ops.node.diffraction_table_update()=={'FINISHED'}
def render(name):
    path=a.output/(name+'.exr');scene.render.filepath=str(path.resolve())
    scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32'
    bpy.ops.render.render(write_still=True)
    image=bpy.data.images.load(str(path.resolve()),check_existing=False);pixels=tuple(image.pixels[:])
    return [sum(pixels[c::4])/(len(pixels)//4) for c in range(3)]
before=render('valid_before')
refresh('wavelength_um,n,k\n0.38,1,2\n0.78,2,3\n')
try:
    render('invalid_must_not_exist')
except RuntimeError as error:
    message=str(error)
    assert 'optical constants' in message.lower(), message
    report['expected_error']=message
else:
    raise RuntimeError('Invalid optical constants were silently accepted')
assert not (a.output/'invalid_must_not_exist.exr').exists()
refresh(original);after=render('valid_after')
error=max(abs(x-y) for x,y in zip(before,after))
assert error<1e-4
report.update(status='passed',before_mean_rgb=before,after_mean_rgb=after,
 maximum_recovery_difference=error,before_exr_sha256=sha(a.output/'valid_before.exr'),
 after_exr_sha256=sha(a.output/'valid_after.exr'))
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
