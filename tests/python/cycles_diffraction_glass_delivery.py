# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Representative Glass grating material scene; fixed samples, no adaptive sampling."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import bpy
from mathutils import Vector

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--samples', type=int, default=256)
p.add_argument('--transport', choices=['pt', 'bdpt', 'guided'], default='pt')
p.add_argument('--osl', action='store_true')
p.add_argument('--coverage', type=float, default=1.0)
p.add_argument('--resolution', type=int, default=720)
p.add_argument('--multiggx', action='store_true', help='Neutral, tinted, and coated two-sided Multi-GGX Glass')
p.add_argument('--denoise', action='store_true', help='Presentation only: denoised PNG plus raw multilayer EXR passes')
p.add_argument('--save-only', action='store_true', help='Save the editable fixture without rendering')
a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
if not 0 <= a.coverage <= 1 or a.samples < 1 or a.resolution < 32:
    p.error('Invalid coverage, sample count or resolution')
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
s = bpy.context.scene
s.render.engine = 'CYCLES'
s.cycles.samples = a.samples
s.cycles.seed = 11
s.cycles.use_adaptive_sampling = False
s.cycles.use_denoising = a.denoise
if a.denoise:
    s.view_layers[0].cycles.denoising_store_passes = True
s.cycles.use_bidirectional_path_tracing = a.transport == 'bdpt'
s.cycles.use_guiding = a.transport == 'guided'
s.cycles.shading_system = a.osl
s.cycles.use_coherent_specular_connections = False
s.cycles.max_bounces = 12
s.cycles.sample_clamp_direct = 0
s.cycles.sample_clamp_indirect = 0
if not a.osl and not a.save_only:
    pref = bpy.context.preferences.addons['cycles'].preferences
    pref.compute_device_type = 'METAL'
    pref.get_devices()
    for d in pref.devices:
        d.use = d.type == 'METAL'
    assert any(d.use for d in pref.devices)
    s.cycles.device = 'GPU'
elif not a.osl:
    s.cycles.device = 'GPU'
s.render.resolution_x = a.resolution
s.render.resolution_y = round(a.resolution * 5 / 9)
s.render.resolution_percentage = 100
s.world = bpy.data.worlds.new('Low fill')
s.world.use_nodes = True
s.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.16,.18,.22,1)
s.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .12
s.view_settings.view_transform = 'AgX'

def mat(name, roughness, distribution, film, weight=1, color=(1,1,1,1)):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = m.node_tree.nodes
    n.clear()
    glass = n.new('ShaderNodeBsdfGlass')
    glass.distribution = distribution
    glass.inputs['Color'].default_value = color
    for key,value in {'Roughness':roughness, 'IOR':1.5,
                      'Diffraction Weight':weight * a.coverage, 'Diffraction Pitch':1200,
                      'Diffraction Depth':320, 'Diffraction Duty Cycle':.41,
                      'Thin Film Thickness':film, 'Thin Film IOR':1.33}.items():
        glass.inputs[key].default_value = value
    tangent = n.new('ShaderNodeCombineXYZ')
    tangent.inputs['X'].default_value = 1
    m.node_tree.links.new(tangent.outputs[0], glass.inputs['Tangent'])
    out = n.new('ShaderNodeOutputMaterial')
    m.node_tree.links.new(glass.outputs[0], out.inputs['Surface'])
    return m

specs = [('GGX smooth grating',.0,'GGX',0),
         ('GGX rough coated grating',.18,'GGX',420),
         ('Beckmann rough grating',.18,'BECKMANN',0)]
if a.multiggx:
    specs = [('Multi-GGX neutral grating',.30,'MULTI_GGX',0,1,(1,1,1,1)),
             ('Multi-GGX local-tint grating',.35,'MULTI_GGX',0,1,(.38,.72,.95,1)),
             ('Multi-GGX lossless-film grating',.30,'MULTI_GGX',420,1,(1,1,1,1))]
for x,spec in zip((-2.3,0,2.3), specs):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, location=(x,0,1.15))
    obj=bpy.context.object
    obj.name=spec[0]
    obj.data.materials.append(mat(*spec))
    for poly in obj.data.polygons:
        poly.use_smooth=True
bpy.ops.mesh.primitive_plane_add(size=200)
floor=bpy.context.object
m=bpy.data.materials.new('Neutral matte floor')
m.use_nodes=True
m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(.13,.13,.13,1)
m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value=.8
floor.data.materials.append(m)
for loc,power,size in [((0,-3,7),1600,3),((0,3,4),2200,1.5)]:
    bpy.ops.object.light_add(type='AREA', location=loc)
    light=bpy.context.object
    light.data.energy=power
    light.data.shape='DISK'
    light.data.size=size
    light.rotation_euler=(Vector((0,0,1))-light.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add(location=(0,-11,5.4))
s.camera=bpy.context.object
s.camera.rotation_euler=(Vector((0,0,1))-s.camera.location).to_track_quat('-Z','Y').to_euler()
s.camera.data.type='ORTHO'
s.camera.data.ortho_scale=8.4
s.render.image_settings.media_type='MULTI_LAYER_IMAGE' if a.denoise else 'IMAGE'
s.render.image_settings.file_format='OPEN_EXR_MULTILAYER' if a.denoise else 'OPEN_EXR'
s.render.image_settings.color_depth='32'
s.render.filepath=str(a.output/'glass.exr')
bpy.ops.wm.save_as_mainfile(filepath=str(a.output/'glass.blend'))
manifest={'samples':a.samples,'adaptive_sampling':False,'denoising':a.denoise,
          'raw_passes':'Noisy Image, Denoising Normal, Denoising Albedo' if a.denoise else None,
          'coverage':a.coverage,'resolution':[s.render.resolution_x,s.render.resolution_y],
          'transport':a.transport,'osl':a.osl,'materials':specs,'multiggx':a.multiggx,
          'binary_sha256':hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scene_sha256':hashlib.sha256((a.output/'glass.blend').read_bytes()).hexdigest(),
          'scope':'Denoised visual presentation; raw passes preserved; not a numeric gate.' if a.denoise else 'Appearance and node integration smoke; not convergence or physical equivalence proof.',
          'tint_model':'One tint per completed return; not per-bounce absorption.' if a.multiggx else None}
(a.output/'scene_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if not a.save_only:
    start=time.monotonic()
    bpy.ops.render.render(write_still=True)
    seconds=time.monotonic()-start
    combined_path=a.output/'glass.exr'
    if a.denoise:
        # Preserve the multilayer output, then save Combined for pixel QA and display.
        s.render.image_settings.media_type='IMAGE'
        s.render.image_settings.file_format='OPEN_EXR'
        combined_path=a.output/'glass_combined.exr'
        bpy.data.images['Render Result'].save_render(str(combined_path),scene=s)
    image=bpy.data.images.load(str(combined_path),check_existing=False)
    pixels=tuple(image.pixels[:])
    if not pixels or not all(math.isfinite(v) for v in pixels):
        raise RuntimeError('Missing or nonfinite rendered pixels')
    mean_rgb=[sum(pixels[c::image.channels])/(len(pixels)//image.channels) for c in range(3)]
    bpy.data.images.remove(image)
    s.render.image_settings.media_type='IMAGE'
    s.render.image_settings.file_format='PNG'
    s.render.image_settings.color_depth='8'
    bpy.data.images['Render Result'].save_render(str(a.output/'glass.png'),scene=s)
    report=dict(manifest,render_seconds=seconds,mean_rgb=mean_rgb,all_pixels_finite=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
