# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Saved, independent diffraction pipeline fixtures; fixed samples, no denoising."""
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
p.add_argument('--template', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--feature', choices=['motion', 'dof', 'volume', 'aov'], required=True)
p.add_argument('--transport', choices=['pt', 'bdpt', 'guided'], required=True)
p.add_argument('--samples', type=int, default=128)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
bpy.ops.wm.open_mainfile(filepath=str(a.template.resolve()))
s = bpy.context.scene
s.cycles.samples = a.samples
s.cycles.seed = 11
s.cycles.use_adaptive_sampling = False
s.cycles.use_denoising = False
s.cycles.time_limit = 0
s.cycles.use_layer_samples = 'IGNORE'
s.cycles.use_bidirectional_path_tracing = a.transport == 'bdpt'
s.cycles.use_guiding = a.transport == 'guided'
s.cycles.shading_system = False
s.cycles.device = 'GPU'
s.cycles.sample_clamp_direct = 0
s.cycles.sample_clamp_indirect = 0
s.render.resolution_x = 720
s.render.resolution_y = 400
s.render.resolution_percentage = 100
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'METAL'
prefs.get_devices()
for device in prefs.devices:
    device.use = device.type == 'METAL'
assert any(d.use for d in prefs.devices)
objects = sorted([o for o in s.objects if o.type == 'MESH' and len(o.data.polygons) > 100],
                 key=lambda o: o.location.x)
assert len(objects) == 3
features = {}
if a.feature == 'motion':
    obj = objects[0]
    position = obj.location.copy()
    for frame, offset in [(0, -1.2), (2, 1.2)]:
        obj.location = position + Vector((offset, 0, 0))
        obj.keyframe_insert('location', frame=frame)
    s.render.use_motion_blur = True
    s.render.motion_blur_shutter = 1.0
    s.frame_set(1)
    features = {'moving_object': obj.name, 'keyframes': [0, 2], 'shutter': 1.0}
elif a.feature == 'dof':
    # Separate object depths so both sides of the focal plane are represented.
    objects[0].location.y = -2.5
    objects[2].location.y = 3.0
    s.camera.data.type = 'PERSP'
    s.camera.data.lens = 42
    s.camera.data.dof.use_dof = True
    s.camera.data.dof.focus_object = objects[1]
    s.camera.data.dof.aperture_fstop = .8
    features = {'focus_object': objects[1].name, 'fstop': .8, 'lens_mm': 42}
elif a.feature == 'volume':
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 3))
    fog = bpy.context.object
    fog.name = 'Finite scattering volume surrounding diffraction surfaces'
    fog.scale = (5, 5, 4)
    mat = bpy.data.materials.new('Homogeneous scattering volume')
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    scatter = mat.node_tree.nodes.new('ShaderNodeVolumeScatter')
    scatter.inputs['Density'].default_value = .045
    scatter.inputs['Anisotropy'].default_value = .3
    output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(scatter.outputs[0], output.inputs['Volume'])
    fog.data.materials.append(mat)
    s.cycles.volume_bounces = 4
    features = {'density': .045, 'anisotropy': .3, 'volume_bounces': 4}
else:
    aov = s.view_layers[0].aovs.add()
    aov.name = 'DiffractionCoverageTest'
    aov.type = 'VALUE'
    for obj in objects:
        node = obj.active_material.node_tree.nodes.new('ShaderNodeOutputAOV')
        node.aov_name = aov.name
        node.inputs['Value'].default_value = .75
    tree = bpy.data.node_groups.new('Diffraction AOV compositor', 'CompositorNodeTree')
    s.compositing_node_group = tree
    tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    layers = tree.nodes.new('CompositorNodeRLayers')
    output = tree.nodes.new('NodeGroupOutput')
    tree.links.new(layers.outputs['Image'], output.inputs['Image'])
    file = tree.nodes.new('CompositorNodeOutputFile')
    file.file_output_items.new('FLOAT', 'Coverage')
    file.directory = str(a.output / 'aov')
    file.file_name = 'coverage'
    file.format.media_type = 'IMAGE'
    file.format.file_format = 'OPEN_EXR'
    file.format.color_depth = '32'
    tree.links.new(layers.outputs[aov.name], file.inputs[0])
    features = {'aov': aov.name, 'interior_value': .75}
notes = bpy.data.texts.new('READ ME - pipeline fixture')
notes.write(f'Independent {a.feature} fixture using spectral diffraction.\n'
            'Fixed samples, no adaptive sampling, no denoising or brightness normalization.\n'
            'Finite pixels and visual inspection are integration checks, not full physical validation.\n')
s.render.image_settings.file_format = 'OPEN_EXR'
s.render.image_settings.color_depth = '32'
s.render.filepath = str(a.output / 'pipeline.exr')
bpy.ops.wm.save_as_mainfile(filepath=str(a.output / 'pipeline.blend'))
start = time.monotonic()
bpy.ops.render.render(write_still=True)
seconds = time.monotonic() - start
image = bpy.data.images.load(str(a.output / 'pipeline.exr'), check_existing=False)
pixels = tuple(image.pixels[:])
assert pixels and all(math.isfinite(v) for v in pixels)
mean = [sum(pixels[i::image.channels]) / (len(pixels) // image.channels) for i in range(3)]
bpy.data.images.remove(image)
s.render.image_settings.file_format = 'PNG'
s.render.image_settings.color_depth = '8'
bpy.data.images['Render Result'].save_render(str(a.output / 'pipeline.png'), scene=s)
report = {'feature': a.feature, 'transport': a.transport, 'features': features,
          'samples': a.samples, 'adaptive_sampling': False, 'denoising': False,
          'all_pixels_finite': True, 'mean_rgb': mean, 'render_seconds': seconds,
          'binary_sha256': hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
          'template_sha256': hashlib.sha256(a.template.read_bytes()).hexdigest(),
          'scope': 'Pipeline integration check; not estimator convergence or physical equivalence.'}
if a.feature == 'aov':
    files = list((a.output / 'aov').rglob('*.exr'))
    assert len(files) == 1, files
    image = bpy.data.images.load(str(files[0]), check_existing=False)
    values = tuple(image.pixels[:])[0::image.channels]
    assert values and all(math.isfinite(v) for v in values)
    report['aov_minmax'] = [min(values), max(values)]
    assert abs(max(values) - .75) < 1e-5 and min(values) >= -1e-6
    bpy.data.images.remove(image)
(a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
