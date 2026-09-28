#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Purpose-built visual diagnostics for the CURRENT provisional grating closure.

Run with Blender --background --python this_file -- --output DIRECTORY.
These scenes do not validate the unintegrated Maxwell cache or coherent transport.
Each saves a .blend, unclamped linear EXR/NumPy data, display PNG and metadata.
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys
import time
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cycles_diffraction_scene import aim, annulus, build

CASES = {
    'pitch': 'Angular order spacing and wavelength cutoffs as grating pitch changes.',
    'roughness': 'Broadening and loss of visible color contrast as facet roughness increases.',
    'depth_duty': 'Redistribution between zero and nonzero orders with relief depth and duty cycle.',
    'orientation': 'Rotation of diffraction bands through a linked shader tangent node.',
    'weight': 'Feature-off endpoint and continuous mixing of ordinary and diffractive reflection.',
    'studio': 'CD/DVD radial tangent fields and macroscopic disc appearance under a strip source.',
    'furnace': 'Uniform unit-radiance illumination: energy-gain regression and the feature-off control.',
    'indirect': 'A restricted spectral reflection path into diffuse walls: multiple bounces and transport estimators.',
    'covered': 'A CD reflector embedded in a 1.2 mm dielectric cover, including refraction and MNEE.',
}


def matte(name, color, emission=False):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    shader = nodes.new('ShaderNodeEmission' if emission else 'ShaderNodeBsdfDiffuse')
    shader.inputs['Color'].default_value = (*color, 1)
    if emission:
        shader.inputs['Strength'].default_value = 1
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(shader.outputs[0], output.inputs['Surface'])
    return material


def label(body, location, size, material):
    curve = bpy.data.curves.new(body, 'FONT')
    curve.body = body
    curve.size = size
    curve.align_x = 'CENTER'
    obj = bpy.data.objects.new(body, curve)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(material)
    for attribute in ['visible_diffuse', 'visible_glossy', 'visible_transmission',
                      'visible_volume_scatter', 'visible_shadow']:
        setattr(obj, attribute, False)
    return obj


def grating(template, name, **parameters):
    material = template.copy()
    material.name = name
    node = next(n for n in material.node_tree.nodes if 'Diffraction Pitch' in n.inputs)
    sockets = {'pitch': 'Diffraction Pitch', 'depth': 'Diffraction Depth',
               'duty': 'Diffraction Duty Cycle', 'weight': 'Diffraction Weight',
               'roughness': 'Roughness', 'medium_ior': 'Diffraction Medium IOR'}
    for key, value in parameters.items():
        node.inputs[sockets[key]].default_value = value
    material['test_parameters'] = json.dumps(parameters, sort_keys=True)
    return material, node


def setup(args):
    settings = SimpleNamespace(scene='plane', pitch=1600.0, depth=150.0, duty=0.5,
                               medium_ior=1.0, roughness=0.075, weight=1.0,
                               samples=args.samples, seed=args.seed, resolution=args.resolution,
                               device=args.device, osl=False, guiding=False, bdpt=False,
                               cover=False, mnee=False, world_scale=1.0)
    scene, devices = build(settings)
    template = bpy.data.materials['Spectral binary grating']
    for obj in list(scene.objects):
        if obj.type != 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.025
    scene.cycles.use_guiding = args.transport in ['guided', 'bdpt_guided']
    scene.cycles.use_bidirectional_path_tracing = args.transport in ['bdpt', 'bdpt_guided']
    scene['validation_status'] = 'PROVISIONAL CLOSURE; physical cache and cross-object coherence not integrated'
    return scene, devices, template


def floor(size=2.0, z=-0.002):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, z))
    obj = bpy.context.object
    obj.name = 'Neutral diffuse backing'
    obj.data.materials.append(matte('Neutral backing', (0.045, 0.055, 0.07)))


def chart_world(scene):
    width, height = 1024, 256
    pixels = np.full((height, width, 4), 0.035, dtype=np.float32)
    pixels[..., 3] = 1
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    source = (np.abs(u[None, :] - 0.65) < 0.006) & (v[:, None] > 0.20) & (v[:, None] < 0.42)
    pixels[source, :3] = 40
    image = bpy.data.images.new('Analytic distant white strip / linear radiance',
                                width=width, height=height, float_buffer=True)
    image.colorspace_settings.name = 'Non-Color'
    image.pixels.foreach_set(pixels.ravel())
    image.pack()
    nodes, links = scene.world.node_tree.nodes, scene.world.node_tree.links
    texture = nodes.new('ShaderNodeTexEnvironment')
    texture.image = image
    links.new(texture.outputs['Color'], nodes['Background'].inputs['Color'])
    nodes['Background'].inputs['Strength'].default_value = 1.0
    scene.cycles.max_bounces = 1
    scene['lighting_definition'] = 'Uniform radiance 0.035 plus a linear-radiance-40 equirectangular strip; direct response only'


def charts(args, case):
    scene, devices, template = setup(args)
    if case == 'furnace':
        entries = [(f'w={w:g} / rough={r:g}', {'weight': w, 'roughness': r})
                   for r in [0, 0.2] for w in [0, 1]]
        columns, rows, title = 4, 1, 'UNIT WHITE FURNACE / ENERGY REGRESSION'
    elif case == 'pitch':
        entries = [(f'{p} nm', {'pitch': p}) for p in [400, 740, 1000, 1600, 3200]]
        columns, rows, title = 5, 1, 'PITCH / ANGULAR DIFFRACTION'
    elif case == 'roughness':
        entries = [(f'roughness {r:g}', {'roughness': r}) for r in [0, 0.025, 0.075, 0.15, 0.3]]
        columns, rows, title = 5, 1, 'ROUGHNESS / BAND BROADENING'
    elif case == 'weight':
        entries = [(f'weight {w:g}', {'weight': w}) for w in [0, 0.25, 0.5, 0.75, 1]]
        columns, rows, title = 5, 1, 'DIFFRACTION WEIGHT / OFF TO ON'
    elif case == 'orientation':
        entries = [(f'tangent {a} deg', {'angle': a}) for a in [0, 45, 90, 135]]
        columns, rows, title = 4, 1, 'LINKED TANGENT / ORIENTATION'
    else:
        entries = [(f'{d} nm / duty {f:g}', {'depth': d, 'duty': f})
                   for f in [0.25, 0.5, 0.75] for d in [0, 75, 150, 300]]
        columns, rows, title = 4, 3, 'RELIEF DEPTH / DUTY CYCLE'
    spacing = 0.1
    width, height = columns * spacing + 0.04, rows * 0.12 + 0.09
    scene.render.resolution_y = round(args.resolution * height / width)
    scene.camera.location = (0, 0, 1)
    aim(scene.camera, (0, 0, 0))
    scene.camera.data.ortho_scale = width
    text = matte('Chart lettering', (0, 0, 0) if case == 'furnace' else (0.7, 0.78, 0.9), emission=True)
    label(title, (0, height / 2 - 0.025, 0.0), 0.010, text)
    label('PROVISIONAL SHADER / ' + ('UNIT WHITE FIELD' if case == 'furnace' else 'IDENTICAL DISTANT WHITE STRIP / DIRECT RESPONSE'),
          (0, -height / 2 + 0.014, 0.0), 0.0045, text)
    floor()
    light = bpy.data.lights.new('Parallel white source', 'SUN')
    light.energy, light.angle = 2.5, 0.025
    obj = bpy.data.objects.new(light.name, light)
    bpy.context.collection.objects.link(obj)
    obj.location = (-0.45, 0.4, 1.0)
    aim(obj, (0, 0, 0))
    bpy.data.objects.remove(obj, do_unlink=True)
    backing = bpy.data.objects['Neutral diffuse backing']
    if case == 'furnace':
        bpy.data.objects.remove(backing, do_unlink=True)
        scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
    else:
        backing.data.materials.clear()
        backing.data.materials.append(matte('Camera-only chart backing', (0.025, 0.033, 0.045), emission=True))
        for attribute in ['visible_diffuse', 'visible_glossy', 'visible_transmission',
                          'visible_volume_scatter', 'visible_shadow']:
            setattr(backing, attribute, False)
        chart_world(scene)
    parameters = []
    for index, (caption, values) in enumerate(entries):
        values = values.copy()
        angle = values.pop('angle', 0)
        material, node = grating(template, caption, **values)
        if case == 'furnace':
            node.inputs['Color'].default_value = (1, 1, 1, 1)
        if case == 'orientation':
            vector = material.node_tree.nodes.new('ShaderNodeCombineXYZ')
            vector.label = 'Controlled world-space grating tangent'
            vector.inputs['X'].default_value = math.cos(math.radians(angle))
            vector.inputs['Y'].default_value = math.sin(math.radians(angle))
            material.node_tree.links.new(vector.outputs['Vector'], node.inputs['Tangent'])
        else:
            node.inputs['Tangent'].default_value = (1, 0, 0)
        x = (index % columns - (columns - 1) / 2) * spacing
        y = ((rows - 1) / 2 - index // columns) * 0.12 + 0.008
        bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=0.038,
                                            location=(x, y, 0.038))
        obj = bpy.context.object
        obj.name = caption
        # Each chart swatch measures the same isolated angular response.
        # Other swatches must not reflect in it or occlude its source.
        for attribute in ['visible_diffuse', 'visible_glossy', 'visible_transmission',
                          'visible_volume_scatter', 'visible_shadow']:
            setattr(obj, attribute, False)
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        label(caption, (x, y - 0.051, 0.0), 0.0055, text)
        parameters.append(dict(caption=caption, tangent_degrees=angle,
                               projected_center=[x, y], radius=0.038, **values))
    return scene, devices, parameters


def discs(args, covered):
    scene, devices, template = setup(args)
    scene.render.resolution_y = round(args.resolution * 0.72)
    scene.camera.location = (0, -0.31, 0.52)
    aim(scene.camera, (0, 0, 0))
    scene.camera.data.ortho_scale = 0.17 if covered else 0.30
    floor(z=-0.0014)
    text = matte('Disc lettering', (0.65, 0.75, 0.9), emission=True)
    plastic = matte('Disc carrier edge', (0.13, 0.14, 0.16))
    parameters = []
    for name, pitch, x in ([('CD / CLEAR COVER', 1600, 0)] if covered else
                           [('CD / 1600 nm', 1600, -0.071), ('DVD / 740 nm', 740, 0.071)]):
        material, node = grating(template, name, pitch=pitch, medium_ior=1.58 if covered else 1.0)
        coordinates = material.node_tree.nodes.new('ShaderNodeTexCoord')
        transform = material.node_tree.nodes.new('ShaderNodeVectorTransform')
        transform.vector_type = 'VECTOR'
        transform.convert_from, transform.convert_to = 'OBJECT', 'WORLD'
        material.node_tree.links.new(coordinates.outputs['Object'], transform.inputs['Vector'])
        material.node_tree.links.new(transform.outputs['Vector'], node.inputs['Tangent'])
        reflector = annulus(name, material, inner_radius=0.024, outer_radius=0.058)
        reflector.location.x = x
        reflector.cycles.is_caustics_receiver = covered
        base = annulus('Carrier / ' + name, plastic, bottom=-0.0012, top=-0.000002)
        base.location.x = x
        hub = annulus('Ungrooved hub / ' + name, plastic, inner_radius=0.0075, outer_radius=0.024)
        hub.location.x = x
        rim = annulus('Ungrooved rim / ' + name, plastic, inner_radius=0.058, outer_radius=0.06)
        rim.location.x = x
        if covered:
            glass = bpy.data.materials.new('Clear dielectric / IOR 1.58')
            glass.use_nodes = True
            nodes, links = glass.node_tree.nodes, glass.node_tree.links
            nodes.clear()
            shader = nodes.new('ShaderNodeBsdfGlass')
            shader.inputs['IOR'].default_value = 1.58
            shader.inputs['Roughness'].default_value = 0
            out = nodes.new('ShaderNodeOutputMaterial')
            links.new(shader.outputs['BSDF'], out.inputs['Surface'])
            cover = annulus('1.2 mm enclosing cover', glass, bottom=-0.000001, top=0.0012)
            cover.location.x = x
            cover.cycles.is_caustics_caster = True
        label(name, (x, -0.078, 0), 0.006, text)
        parameters.append(dict(name=name, pitch_nm=pitch, cover_mm=1.2 if covered else 0,
                               incident_ior=1.58 if covered else 1.0, grating_radius_mm=[24, 58]))
    light = bpy.data.lights.new('White studio strip', 'AREA')
    light.energy, light.shape, light.size, light.size_y = 0.5, 'RECTANGLE', 0.04, 0.24
    light.cycles.is_caustics_light = covered
    obj = bpy.data.objects.new(light.name, light)
    bpy.context.collection.objects.link(obj)
    obj.location = (-0.18, 0.20, 0.45)
    aim(obj, (0, 0, 0))
    return scene, devices, parameters


def indirect(args):
    scene, devices, template = setup(args)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
    scene.render.resolution_y = round(args.resolution * 0.85)
    scene.camera.data.type = 'PERSP'
    scene.camera.data.lens = 48
    scene.camera.location = (0, -0.9, 0.32)
    aim(scene.camera, (0, 0.03, 0.22))
    white = matte('Achromatic room', (0.75, 0.75, 0.75))
    def wall(name, location, scale):
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        obj = bpy.context.object
        obj.name, obj.scale = name, scale
        obj.data.materials.append(white)
    wall('Floor', (0, 0, -0.005), (0.5, 0.5, 0.01))
    wall('Back / indirect receiver', (0, 0.255, 0.25), (0.5, 0.01, 0.5))
    wall('Left / indirect receiver', (-0.255, 0, 0.25), (0.01, 0.5, 0.5))
    wall('Right / indirect receiver', (0.255, 0, 0.25), (0.01, 0.5, 0.5))
    wall('Ceiling / indirect receiver', (0, 0, 0.505), (0.5, 0.5, 0.01))
    material, node = grating(template, 'Indirect spectral reflector', roughness=0.075)
    node.inputs['Tangent'].default_value = (1, 0, 0)
    bpy.ops.mesh.primitive_plane_add(size=0.34, location=(0, 0, 0.0001))
    bpy.context.object.name = 'Grating tile'
    bpy.context.object.data.materials.append(material)
    wall('Occluder', (0.10, 0.13, 0.065), (0.065, 0.065, 0.13))
    bpy.context.object.rotation_euler.z = 0.3
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.055,
                                        location=(-0.11, 0.13, 0.055))
    bpy.context.object.data.materials.append(white)
    for polygon in bpy.context.object.data.polygons:
        polygon.use_smooth = True
    light = bpy.data.lights.new('Restricted white source', 'SPOT')
    light.energy, light.spot_size, light.spot_blend, light.shadow_soft_size = 30, math.radians(20), 0.15, 0.005
    obj = bpy.data.objects.new(light.name, light)
    bpy.context.collection.objects.link(obj)
    obj.location = (0, -0.07, 0.48)
    aim(obj, (0, 0.03, 0))
    return scene, devices, [dict(pitch_nm=1600, roughness=0.075, room_size_m=0.5,
                                 source_cone_degrees=20, surrounding_materials='achromatic')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', *CASES], default='all')
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--seed', type=int, default=11,
                        help='Independent render seed; use a separate output directory per seed.')
    parser.add_argument('--resolution', type=int, default=960)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--transport', choices=['pt', 'bdpt', 'guided', 'bdpt_guided'], default='pt')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if not 0 <= args.seed <= 2147483647:
        parser.error('--seed must be in [0, 2147483647]')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = pathlib.Path(bpy.app.binary_path)
    with binary.open('rb') as stream:
        binary_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    scripts = {name: hashlib.sha256(pathlib.Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ['cycles_diffraction_suite.py', 'cycles_diffraction_scene.py']}
    snapshot = args.output / 'sources' / scripts['cycles_diffraction_suite.py'][:12]
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in scripts:
        (snapshot / name).write_text(pathlib.Path(__file__).with_name(name).read_text())
    manifest = []
    for case in CASES if args.case == 'all' else [args.case]:
        scene, devices, parameters = (indirect(args) if case == 'indirect' else
                                      discs(args, case == 'covered') if case in ['studio', 'covered'] else
                                      charts(args, case))
        scene['test_purpose'] = CASES[case]
        stem = args.output / (case if args.transport == 'pt' else case + '_' + args.transport)
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.filepath = str(stem.with_suffix('.exr'))
        bpy.ops.wm.save_as_mainfile(filepath=str(stem.with_suffix('.blend')))
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        seconds = time.perf_counter() - start
        import OpenImageIO as oiio
        image = oiio.ImageInput.open(scene.render.filepath)
        rgb = np.asarray(image.read_image(format=oiio.FLOAT))[..., :3]
        image.close()
        if not np.isfinite(rgb).all():
            raise RuntimeError('Nonfinite render: ' + case)
        np.save(stem.with_suffix('.npy'), rgb)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '8'
        bpy.data.images['Render Result'].save_render(str(stem.with_suffix('.png')), scene=scene)
        report = dict(case=case, purpose=CASES[case], parameters=parameters, samples=args.samples,
                      transport=args.transport, seed=scene.cycles.seed, camera_only_annotations=True,
                      isolated_swatches=case not in ['studio', 'covered', 'indirect'],
                      resolution=[scene.render.resolution_x, scene.render.resolution_y],
                      devices=devices, seconds=seconds, mean=float(rgb.mean()), peak=float(rgb.max()),
                      binary_sha256=binary_hash, script_sha256=scripts, physical_cache_integrated=False,
                      coherent_transport_integrated=False, status='provisional_shader_visual_diagnostic')
        stem.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        manifest.append(report)
        print('DIFFRACTION_SCENE ' + json.dumps(report), flush=True)
    (args.output / ('manifest_' + args.case + '_' + args.transport + '.json')).write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
