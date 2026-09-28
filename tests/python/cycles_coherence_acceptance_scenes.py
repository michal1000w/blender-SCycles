# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Save cross-object coherence reference fixtures and measure the unsupported control.

reference_* properties are fixture metadata, NOT renderer controls. No reference
texture is used in the scene. Current Cycles intentionally cannot pass the phase
response acceptance requirement; the script records that fact rather than hiding it.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils import Vector

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
a.output=a.output.resolve();a.reference=a.reference.resolve()
a.output.mkdir(parents=True,exist_ok=False)
reference=json.loads(a.reference.read_text())
bpy.ops.wm.read_factory_settings(use_empty=True)
s=bpy.context.scene
s.render.engine='CYCLES'
s.cycles.samples=64
s.cycles.seed=11
s.cycles.use_adaptive_sampling=False
s.cycles.use_denoising=False
s.cycles.use_bidirectional_path_tracing=False
s.cycles.use_guiding=False
preferences=bpy.context.preferences.addons['cycles'].preferences
preferences.compute_device_type='METAL'
preferences.get_devices()
for device in preferences.devices: device.use=device.type=='METAL'
assert any(device.use for device in preferences.devices)
s.cycles.device='GPU'
s.render.resolution_x=512;s.render.resolution_y=256;s.render.resolution_percentage=100
s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32'
s.view_settings.view_transform='Standard'
s.world=bpy.data.worlds.new('Black environment');s.world.use_nodes=True
s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0
bpy.ops.mesh.primitive_plane_add(size=1,location=(0,1,0),rotation=(math.pi/2,0,0))
detector=bpy.context.object;detector.name='Diffuse detector — scalar reference uses no detector BSDF'
detector.scale=(.042,.022,1)
material=bpy.data.materials.new('White diffuse detector');material.use_nodes=True
nodes=material.node_tree.nodes;nodes.clear()
diffuse=nodes.new('ShaderNodeBsdfDiffuse');diffuse.inputs['Color'].default_value=(1,1,1,1)
out=nodes.new('ShaderNodeOutputMaterial');material.node_tree.links.new(diffuse.outputs[0],out.inputs['Surface'])
detector.data.materials.append(material)
sources=[]
for name,x in [('A',-50e-6),('B',50e-6)]:
    bpy.ops.object.light_add(type='POINT',location=(x,0,0))
    source=bpy.context.object;source.name=f'Separate source {name}'
    source.data.energy=10;source.data.shadow_soft_size=0
    sources.append(source)
bpy.ops.object.camera_add(location=(0,.5,0))
s.camera=bpy.context.object;s.camera.name='Detector readout camera'
s.camera.rotation_euler=Vector((0,1,0)).to_track_quat('-Z','Y').to_euler()
s.camera.data.type='ORTHO';s.camera.data.ortho_scale=.04
s.camera.data.clip_start=.001;s.camera.data.clip_end=10
s['reference_status']='UNIMPLEMENTED COHERENT TRANSPORT; metadata below is not consumed by Cycles'
s['reference_detector_model']='Analytic scalar intensity; native render is a diffuse-detector incoherent control'
manifest={'binary':str(Path(bpy.app.binary_path).resolve()),'sha256':hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
          'renderer_coherent_support':False,'samples':64,'adaptive_sampling':False,'denoising':False,
          'scope':'Two fixed-seed native incoherent controls; not a coherent rendering benchmark',
          'reference_file':str(a.reference.resolve()),'scenes':[],'renders':{}}
arrays={}
for case in reference['cases']:
    name=case['name']
    sources[0]['reference_phase_radians']=0.0
    sources[1]['reference_phase_radians']=case['phase_radians']
    for source in sources:
        source['reference_wavelength_m']=case['wavelength_m']
        source['reference_source_group']='shared_scalar_source_pair'
    s['reference_correlation']=case['correlation']
    s['reference_coherence_length_m']=case['coherence_length_m'] or -1.0
    s['reference_case_json']=json.dumps(case)
    s.render.filepath=str(a.output/f'{name}.exr')
    bpy.ops.wm.save_as_mainfile(filepath=str(a.output/f'{name}.blend'))
    manifest['scenes'].append(name)
    if name not in ['coherent_phase_0','coherent_phase_pi']: continue
    start=time.monotonic();bpy.ops.render.render(write_still=True);elapsed=time.monotonic()-start
    image=bpy.data.images.load(s.render.filepath,check_existing=False)
    pixels=np.array(image.pixels[:],dtype=np.float32)
    assert pixels.size and np.isfinite(pixels).all()
    arrays[name]=pixels;bpy.data.images.remove(image)
    s.render.image_settings.file_format='PNG';s.render.image_settings.color_depth='8'
    bpy.data.images['Render Result'].save_render(str(a.output/f'{name}.png'),scene=s)
    s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32'
    manifest['renders'][name]={'render_seconds':elapsed,'finite':True}
delta=float(np.max(abs(arrays['coherent_phase_0']-arrays['coherent_phase_pi'])))
manifest['phase_pair_max_pixel_difference']=delta
manifest['phase_response_acceptance']='FAIL: reference phase metadata has no renderer transport implementation'
manifest['reference_central_intensity_phase0_to_pi']='4/r^2 to 0 for equal coherent scalar sources'
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('UNIMPLEMENTED coherence control: phase-pair pixel difference',delta)
