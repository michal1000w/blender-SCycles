# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import bpy, hashlib, json, math, time
out=Path(__file__).resolve().parents[2]/'tests/output/diffraction/bounded_package_cd_v1';out.mkdir(exist_ok=False)
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=1024;s.cycles.use_adaptive_sampling=False;s.cycles.use_denoising=False
s.cycles.use_bidirectional_path_tracing=False;s.cycles.use_guiding=False
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
for d in prefs.devices:d.use=d.type=='METAL'
assert any(d.use for d in prefs.devices)
s.cycles.device='GPU';s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32';s.render.filepath=str(out/'physical_discs_pt.exr')
bpy.ops.wm.save_as_mainfile(filepath=str(out/'physical_discs_pt.blend'))
start=time.monotonic();bpy.ops.render.render(write_still=True);elapsed=time.monotonic()-start
im=bpy.data.images.load(s.render.filepath,check_existing=False);pixels=tuple(im.pixels[:]);assert pixels and all(math.isfinite(v) for v in pixels);bpy.data.images.remove(im)
s.render.image_settings.file_format='PNG';s.render.image_settings.color_depth='8';bpy.data.images['Render Result'].save_render(str(out/'physical_discs_pt.png'),scene=s)
r={'binary':str(Path(bpy.app.binary_path).resolve()),'sha256':hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),'samples':1024,'adaptive_sampling':False,'denoising':False,'device':'METAL','transport':'PT','render_seconds':elapsed,'all_pixels_finite':True,'source_scene':'delivery_package_cd_v1/physical_discs_pt.blend','scope':'Fresh presentation render from selected bounded package, not a new broad benchmark'}
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
