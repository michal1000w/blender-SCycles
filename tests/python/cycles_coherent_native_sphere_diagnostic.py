#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One tiny native sphere emission-presence diagnostic, separate from oracle."""
import hashlib,json,sys,time
from pathlib import Path
import bpy
from mathutils import Vector
root=Path(__file__).resolve().parents[2];output=root/'build/tests/python/cycles_coherent_native_sphere_presence_v33b';output.mkdir(parents=True,exist_ok=False)
fixture=root/'build/tests/python/cycles_coherent_sphere_acceptance_v33_v2/phase_0.blend';bpy.ops.wm.open_mainfile(filepath=str(fixture))
scene=bpy.context.scene;sphere=bpy.data.objects['Native analytic mirror sphere'];camera=scene.camera
for obj in list(scene.objects):
 if obj not in (sphere,camera):bpy.data.objects.remove(obj,do_unlink=True)
scene.cycles.use_coherent_specular_connections=False;scene.cycles.use_bidirectional_path_tracing=False;scene.cycles.use_guiding=False
scene.cycles.samples=1;scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False;scene.cycles.device='GPU'
scene.render.resolution_x=scene.render.resolution_y=16;scene.render.resolution_percentage=100
camera.location=(-.8,0,0);camera.rotation_euler=(Vector((0,0,0))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.ortho_scale=.7
material=sphere.data.materials[0];nodes=material.node_tree.nodes;nodes.clear();emit=nodes.new('ShaderNodeEmission');emit.inputs['Color'].default_value=(1,1,1,1);emit.inputs['Strength'].default_value=1;out=nodes.new('ShaderNodeOutputMaterial');material.node_tree.links.new(emit.outputs[0],out.inputs['Surface'])
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
for d in prefs.devices:d.use=d.type=='METAL'
assert any(d.use for d in prefs.devices)
scene.render.image_settings.media_type='IMAGE';scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32';scene.render.filepath=str(output/'native_sphere.exr')
bpy.ops.wm.save_as_mainfile(filepath=str(output/'native_sphere.blend'));start=time.monotonic();bpy.ops.render.render(write_still=True)
report={'binary':bpy.app.binary_path,'binary_sha256':hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),'original_fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'output':scene.render.filepath,'samples':1,'resolution':[16,16],'connector_enabled':False,'elapsed_seconds_including_initialization':time.monotonic()-start,'scope':'Native analytic GN sphere presence via unit emission; changed camera/material diagnostic only, no reference changes and no transport acceptance claim'}
(output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
