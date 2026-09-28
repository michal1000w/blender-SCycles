#!/usr/bin/env python3
"""Diagnostic-only cloned package stage counters, not a physical render or acceptance."""
import hashlib,json,time
from pathlib import Path
import bpy,numpy as np
root=Path(__file__).resolve().parents[2]
out=root/'build/tests/python/cycles_coherent_sphere_stages_diagnostic_v34';out.mkdir(exist_ok=False)
fixture=root/'build/tests/python/cycles_coherent_sphere_acceptance_v33_v2/phase_0.blend'
assert Path(bpy.app.binary_path).resolve()==(root/'build/diagnostic_coherent_sphere_stages_v34/Blender.app/Contents/MacOS/Blender').resolve()
bpy.ops.wm.open_mainfile(filepath=str(fixture));s=bpy.context.scene
assert s.cycles.use_coherent_specular_connections and s.cycles.coherent_polarization_mode=='VECTOR'
s.cycles.coherent_polarization_mode='SCALAR';s.cycles.samples=4;s.render.resolution_x=s.render.resolution_y=16
s.cycles.device='GPU';s.cycles.use_adaptive_sampling=False;s.cycles.use_denoising=False
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
for d in prefs.devices:d.use=d.type=='METAL'
assert any(d.use for d in prefs.devices)
s.render.image_settings.media_type='IMAGE';s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32';s.render.filepath=str(out/'stages.exr')
bpy.ops.wm.save_as_mainfile(filepath=str(out/'stages.blend'));start=time.monotonic();bpy.ops.render.render(write_still=True)
im=bpy.data.images.load(s.render.filepath,check_existing=False);a=np.array(im.pixels[:],dtype=float).reshape(16,16,im.channels)[:,:,:3]
np.savez_compressed(out/'stages_raw.npz',rgb=a)
r={'binary_sha256':hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'mode':s.cycles.coherent_polarization_mode,'bidirectional':s.cycles.use_bidirectional_path_tracing,'samples':4,'resolution':[16,16],'seconds_including_initialization':time.monotonic()-start,'finite':bool(np.isfinite(a).all()),'mean_rgb':a.mean((0,1)).tolist(),'min_rgb':a.min((0,1)).tolist(),'max_rgb':a.max((0,1)).tolist(),'nonzero_pixels':int(np.any(a!=0,axis=2).sum()),'scope':'DIAGNOSTIC ONLY: RGB eligible/solved/visible sphere candidate counts divided by2; physical field accumulation replaced. Not radiometric acceptance','diagnostic_clone_manifest':str(root/'build/diagnostic_coherent_sphere_stages_v34/manifest.json')}
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
