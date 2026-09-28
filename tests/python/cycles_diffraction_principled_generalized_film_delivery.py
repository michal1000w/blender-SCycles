# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Representative transmitting Principled grating material scene; fixed samples, no adaptive sampling."""
import argparse
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
a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
if not 0 <= a.coverage <= 1 or a.samples < 1 or a.resolution < 32:
    p.error('Invalid coverage, sample count or resolution')
a.output.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
s = bpy.context.scene
s.render.engine = 'CYCLES'
s.cycles.samples = a.samples
s.cycles.seed = 11
s.cycles.use_adaptive_sampling = False
s.cycles.use_denoising = False
s.cycles.use_bidirectional_path_tracing = a.transport == 'bdpt'
s.cycles.use_guiding = a.transport == 'guided'
s.cycles.shading_system = a.osl
s.cycles.max_bounces = 12
s.cycles.sample_clamp_direct = 0
s.cycles.sample_clamp_indirect = 0
if not a.osl:
    pref = bpy.context.preferences.addons['cycles'].preferences
    pref.compute_device_type = 'METAL'
    pref.get_devices()
    for d in pref.devices:
        d.use = d.type == 'METAL'
    assert any(d.use for d in pref.devices)
    s.cycles.device = 'GPU'
s.render.resolution_x = a.resolution
s.render.resolution_y = round(a.resolution * 5 / 9)
s.render.resolution_percentage = 100
s.world = bpy.data.worlds.new('Low fill')
s.world.use_nodes = True
s.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.16,.18,.22,1)
s.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .12
s.view_settings.view_transform = 'AgX'

def mat(name, roughness, distribution, film, weight=1):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = m.node_tree.nodes
    n.clear()
    glass = n.new('ShaderNodeBsdfPrincipled')
    glass.distribution = distribution
    for key,value in {'Roughness':roughness, 'IOR':1.5,
                      'Diffraction Weight':weight * a.coverage, 'Diffraction Pitch':1200,
                      'Diffraction Depth':320, 'Diffraction Duty Cycle':.41,
                      'Thin Film Thickness':film, 'Thin Film IOR':1.35, 'Transmission Weight':1, 'Specular Tint':(.25,.65,1,1), 'Transmission Dispersion Scale':1, 'Transmission Dispersion Abbe Number':10, 'Base Color':((.25,.7,.9,1) if film else (1,1,1,1))}.items():
        glass.inputs[key].default_value = value
    tangent = n.new('ShaderNodeCombineXYZ')
    tangent.inputs['X'].default_value = 1
    m.node_tree.links.new(tangent.outputs[0], glass.inputs['Tangent'])
    out = n.new('ShaderNodeOutputMaterial')
    m.node_tree.links.new(glass.outputs[0], out.inputs['Surface'])
    return m

specs = [('Tinted dispersive 420nm film',.0,'GGX',420),
         ('Tinted dispersive rough 260nm film',.18,'GGX',260),
         ('Tinted dispersive 640nm film half coverage',.18,'GGX',640,.5)]
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
s.render.image_settings.file_format='OPEN_EXR'
s.render.image_settings.color_depth='32'
s.render.filepath=str(a.output/'principled_generalized_film.exr')
bpy.ops.wm.save_as_mainfile(filepath=str(a.output/'principled_generalized_film.blend'))
start=time.monotonic()
bpy.ops.render.render(write_still=True)
seconds=time.monotonic()-start
image=bpy.data.images.load(str(a.output/'principled_generalized_film.exr'),check_existing=False)
pixels=tuple(image.pixels[:])
if not pixels or not all(math.isfinite(v) for v in pixels):
    raise RuntimeError('Missing or nonfinite rendered pixels')
mean_rgb=[sum(pixels[c::image.channels])/(len(pixels)//image.channels) for c in range(3)]
bpy.data.images.remove(image)
s.render.image_settings.file_format='PNG'
s.render.image_settings.color_depth='8'
bpy.data.images['Render Result'].save_render(str(a.output/'principled_generalized_film.png'),scene=s)
report={'render_seconds':seconds,'samples':a.samples,'adaptive_sampling':False,
        'denoising':False,'coverage':a.coverage,'mean_rgb':mean_rgb,'all_pixels_finite':True,
        'resolution':[s.render.resolution_x,s.render.resolution_y],'transport':a.transport,'osl':a.osl,'materials':specs,
        'scope':'Appearance and node integration smoke; not convergence or physical equivalence proof.'}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
