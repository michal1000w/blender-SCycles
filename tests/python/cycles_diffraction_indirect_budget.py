# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-sample indirect-light budget experiment; preserve raw independent-seed images."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import bpy
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--light-paths',type=int,required=True)
p.add_argument('--samples',type=int,default=256)
p.add_argument('--resolution',type=int,default=480)
p.add_argument('--transport',choices=['bdpt','bdpt_guided'],default='bdpt')
p.add_argument('--seeds',type=int,nargs='+',default=[17,29])
p.add_argument('--presentation',action='store_true',help='Render one selected seed without an extra warm-up; not a benchmark.')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
if not 1024<=a.light_paths<=4194304 or a.samples<1 or a.resolution<16:
    p.error('Invalid light-path, sample or resolution budget')
if len(set(a.seeds))!=len(a.seeds) or any(not 0<=s<=2147483647 for s in a.seeds):
    p.error('Seeds must be distinct nonnegative signed integers')
if a.presentation and len(a.seeds)!=1:p.error('Presentation mode requires exactly one seed')
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(a.scene.resolve()))
s=bpy.context.scene
s.cycles.use_bidirectional_path_tracing=True
s.cycles.use_guiding=a.transport=='bdpt_guided'
s.cycles.bdpt_light_paths=a.light_paths
s.cycles.bdpt_update_samples=4
s.cycles.samples=a.samples
s.cycles.use_adaptive_sampling=False
s.cycles.use_denoising=False
s.cycles.time_limit=0
s.cycles.use_layer_samples='IGNORE'
s.render.resolution_y=round(a.resolution*s.render.resolution_y/s.render.resolution_x)
s.render.resolution_x=a.resolution;s.render.resolution_percentage=100
s.render.use_persistent_data=True
prefs=bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type='METAL';prefs.get_devices()
for device in prefs.devices:device.use=device.type=='METAL'
assert any(device.use for device in prefs.devices)
s.cycles.device='GPU'
assert any(n.bl_idname=='ShaderNodeBsdfDiffraction' and n.quality=='FAST'
           for m in bpy.data.materials if m.use_nodes for n in m.node_tree.nodes)
def settings():
    return dict(samples=s.cycles.samples,light_paths=s.cycles.bdpt_light_paths,
        update_samples=s.cycles.bdpt_update_samples,adaptive=s.cycles.use_adaptive_sampling,
        denoising=s.cycles.use_denoising,time_limit=s.cycles.time_limit,
        layer_samples=s.cycles.use_layer_samples,bdpt=s.cycles.use_bidirectional_path_tracing,
        guiding=s.cycles.use_guiding,resolution=[s.render.resolution_x,s.render.resolution_y])
expected=settings();assert expected['light_paths']==a.light_paths
report=dict(scope=__doc__,status='running',settings=expected,transport=a.transport,
    mode='presentation' if a.presentation else 'noise_pilot',
    source_scene_sha256=digest(a.scene),binary_sha256=digest(Path(bpy.app.binary_path)),
    script_sha256=digest(Path(__file__)),devices=[d.name for d in prefs.devices if d.use],
    color_space='linear BT.709',normalized=False,runs=[])
for index,seed in enumerate(a.seeds if a.presentation else [11]+a.seeds):
    s.cycles.seed=seed;warmup=index==0 and not a.presentation
    prefix=a.output/('warmup' if warmup else 'seed_'+str(seed))
    s.render.filepath=str(prefix.with_suffix('.exr'))
    s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32'
    bpy.ops.wm.save_as_mainfile(filepath=str(prefix.with_suffix('.blend')))
    assert settings()==expected
    start=time.perf_counter();bpy.ops.render.render(write_still=True);elapsed=time.perf_counter()-start
    assert settings()==expected
    image=bpy.data.images.load(s.render.filepath,check_existing=False)
    pixels=np.asarray(image.pixels[:],dtype=np.float32).reshape(image.size[1],image.size[0],4)[...,:3]
    assert np.isfinite(pixels).all()
    np.save(prefix.with_suffix('.npy'),pixels,allow_pickle=False)
    s.render.image_settings.file_format='PNG';s.render.image_settings.color_depth='8'
    image.save_render(str(prefix.with_suffix('.png')),scene=s)
    report['runs'].append(dict(seed=seed,warmup=warmup,render_seconds=elapsed,
        mean_rgb=pixels.mean(axis=(0,1),dtype=np.float64).tolist(),
        artifacts={ext:dict(file=prefix.with_suffix('.'+ext).name,sha256=digest(prefix.with_suffix('.'+ext)))
                   for ext in ['blend','exr','png','npy']}))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    bpy.data.images.remove(image)
report.update(status='completed',median_measured_seconds=statistics.median(r['render_seconds'] for r in report['runs'] if not r['warmup']),
    timing_scope=('Presentation render includes preparation and EXR writing; not a benchmark.' if a.presentation else
                  'Render calls include EXR writing. First warm-up excluded; not GPU timestamps or an equal-error performance verdict.'))
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
