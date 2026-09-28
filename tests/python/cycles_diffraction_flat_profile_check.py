# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Two fixed-sample renders of a flat-profile Glass fixture, preserving raw pixels."""
import argparse, array, hashlib, json, math, sys, time
from pathlib import Path
import bpy
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2]
bpy.ops.wm.open_mainfile(filepath=str(root/'tests/output/diffraction/glass_delivery_pt_1024_v1/glass.blend'))
s=bpy.context.scene;s.cycles.samples=512;s.cycles.seed=11;s.cycles.use_adaptive_sampling=False;s.cycles.use_denoising=False;s.cycles.time_limit=0;s.render.use_persistent_data=True
s.cycles.use_bidirectional_path_tracing=False;s.cycles.use_guiding=False
for m in bpy.data.materials:
 if m.use_nodes:
  for n in m.node_tree.nodes:
   if 'Diffraction Depth' in n.inputs:n.inputs['Diffraction Depth'].default_value=0
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
for d in prefs.devices:d.use=d.type=='METAL'
assert any(d.use for d in prefs.devices);s.cycles.device='GPU'
bpy.ops.wm.save_as_mainfile(filepath=str(a.output/'flat.blend'))
times=[]
for i in range(2):
 s.render.filepath=str(a.output/f'flat_{i}.exr');s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32'
 t=time.perf_counter();bpy.ops.render.render(write_still=True);times.append(time.perf_counter()-t)
im=bpy.data.images.load(s.render.filepath,check_existing=False);pixels=array.array('f',im.pixels[:]);assert pixels and all(math.isfinite(v) for v in pixels)
with (a.output/'pixels.f32').open('wb') as f:pixels.tofile(f)
s.render.image_settings.file_format='PNG';s.render.image_settings.color_depth='8';bpy.data.images['Render Result'].save_render(str(a.output/'flat.png'),scene=s)
(a.output/'report.json').write_text(json.dumps({'samples':512,'adaptive_sampling':False,'denoising':False,'seed':11,'times_seconds':times,'all_pixels_finite':True,'binary_sha256':hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),'scope':'Two renders in one isolated process; second is warm. Exact zero relief; not representative of nonzero gratings.'},indent=2))
