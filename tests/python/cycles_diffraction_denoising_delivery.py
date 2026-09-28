# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Metal/OIDN integration fixture; noisy and denoised outputs stay separate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--template',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--samples',type=int,default=1024)
p.add_argument('--transport',choices=['pt','bdpt','guided'],default='pt')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
bpy.ops.wm.open_mainfile(filepath=str(a.template.resolve()))
s=bpy.context.scene;s.render.engine='CYCLES'
s.cycles.samples=a.samples;s.cycles.seed=11;s.cycles.use_adaptive_sampling=False
s.cycles.use_denoising=True;s.cycles.denoiser='OPENIMAGEDENOISE'
s.cycles.denoising_use_gpu=True;s.cycles.denoising_prefilter='ACCURATE'
s.cycles.denoising_quality='HIGH';s.cycles.denoising_input_passes='RGB_ALBEDO_NORMAL'
s.cycles.use_bidirectional_path_tracing=a.transport=='bdpt'
s.cycles.use_guiding=a.transport=='guided'
s.view_layers[0].cycles.use_denoising=True
s.view_layers[0].cycles.denoising_store_passes=True
prefs=bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type='METAL';prefs.get_devices()
for device in prefs.devices:device.use=device.type=='METAL'
assert any(device.use for device in prefs.devices)
s.cycles.device='GPU';s.render.use_compositing=True
tree=bpy.data.node_groups.new('Diffraction denoising evidence','CompositorNodeTree')
s.compositing_node_group=tree
tree.interface.new_socket(name='Image',in_out='OUTPUT',socket_type='NodeSocketColor')
layers=tree.nodes.new('CompositorNodeRLayers');result=tree.nodes.new('NodeGroupOutput')
tree.links.new(layers.outputs['Image'],result.inputs['Image'])
passes={'noisy':('Noisy Image','RGBA'),'albedo':('Denoising Albedo','RGBA'),
        'specular_albedo':('Denoising Specular Albedo','RGBA'),
        'normal':('Denoising Normal','VECTOR'),'roughness':('Denoising Roughness','FLOAT'),
        'depth':('Denoising Depth','FLOAT')}
for name,(socket,kind) in passes.items():
    output=tree.nodes.new('CompositorNodeOutputFile')
    output.file_output_items.new(kind,name)
    output.directory=str(a.output/'passes');output.file_name=name
    output.format.media_type='IMAGE';output.format.file_format='OPEN_EXR';output.format.color_depth='32'
    tree.links.new(layers.outputs[socket],output.inputs[0])
s.render.image_settings.media_type='IMAGE'
assert s.render.image_settings.media_type=='IMAGE'
s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32'
s.render.filepath=str(a.output/'denoised.exr')
bpy.ops.wm.save_as_mainfile(filepath=str(a.output/'denoising.blend'))
start=time.monotonic();bpy.ops.render.render(write_still=True);elapsed=time.monotonic()-start
report={'binary':str(Path(bpy.app.binary_path).resolve()),
        'sha256':hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
        'samples':a.samples,'adaptive_sampling':False,'denoising':True,'denoiser':'OPENIMAGEDENOISE',
        'gpu_denoising_requested':True,'render_device':'METAL','transport':a.transport,
        'render_and_denoise_seconds':elapsed,'passes':{},
        'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'output_media_type':'IMAGE',
        'scope':'Denoising integration/presentation check, not a speed benchmark or physical reference'}
paths={'denoised':a.output/'denoised.exr'}
for name in passes:
    matches=list((a.output/'passes').glob(name+'*.exr'))
    assert len(matches)==1,(name,matches)
    paths[name]=matches[0]
arrays={}
for name,path in paths.items():
    image=bpy.data.images.load(str(path),check_existing=False)
    pixels=np.array(image.pixels[:],dtype=np.float32).reshape(-1,image.channels)
    assert pixels.size and np.isfinite(pixels).all(),name
    rgb=pixels[:,:min(3,image.channels)]
    if name=='normal':assert np.max(abs(rgb))<=1.001
    report['passes'][name]={'file':str(path),'finite':True,'channels':image.channels,
                          'minimum':float(rgb.min()),'maximum':float(rgb.max()),'mean':float(rgb.mean(dtype=np.float64))}
    if name in ['noisy','denoised']:
        arrays[name]=pixels.copy()
        s.render.image_settings.file_format='PNG';s.render.image_settings.color_depth='8'
        image.save_render(str(a.output/f'{name}.png'),scene=s)
    bpy.data.images.remove(image)
assert arrays['noisy'].shape==arrays['denoised'].shape
delta=float(np.mean(abs(arrays['noisy']-arrays['denoised'])))
assert delta>0,'Denoiser did not change the image'
report['noisy_to_denoised_mean_absolute_difference']=delta
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
