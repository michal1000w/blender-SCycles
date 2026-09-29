#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Author streamed closed-convex Glass fixtures (Blender only, no render calls).

Run: blender --background --factory-startup --python THIS -- OUTPUT_DIR

Two layouts, each with saved editable variants:
  prism_TT: a 60-degree Glass prism (8 triangles) deviating a beam through two
    nonparallel faces (entry/exit TT), interfering with the unobstructed direct
    path. Variants: phase_0, phase_pi, distinct groups, partial coherence,
    negatively scaled object (flipped winding) and a linked mesh instance.
  cube_R: a rotated Glass cube (12 triangles) and a finite mirror: direct,
    exterior Fresnel reflection on Glass, mirror reflection and the ordered
    mirror -> Glass exterior reflection pair.
Each variant writes geometry.json with the exact saved world geometry, sources
and camera mapping. References are computed separately, before rendering, by
cycles_coherent_streamed_glass_plan.py with the independent oracle.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycles_coherent_mirror_acceptance import configure_scene, set_diffuse_material
from cycles_coherent_polarizer_scene import glass, pane

RESOLUTION = 32


def prism_mesh(half, height, depth):
    """Local triangular prism, axis along y, apex +z, centroid-free outward winding."""
    a, b, t = np.array([-half, 0, 0]), np.array([half, 0, 0]), np.array([0, 0, height])
    y0, y1 = np.array([0, -depth / 2, 0]), np.array([0, depth / 2, 0])
    verts = [a + y0, b + y0, t + y0, a + y1, b + y1, t + y1]
    faces = [(0, 2, 1), (3, 4, 5), (0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4), (2, 0, 3), (2, 3, 5)]
    inside = np.mean(verts, axis=0)
    for i, (p, q, r) in enumerate(faces):
        n = np.cross(verts[q] - verts[p], verts[r] - verts[p])
        if np.dot(n, (verts[p] + verts[q] + verts[r]) / 3 - inside) < 0:
            faces[i] = (p, r, q)  # Outward winding.
    return [tuple(map(float, v)) for v in verts], faces


def cube_mesh(size):
    d = size / 2
    verts = [(-d, -d, -d), (d, -d, -d), (d, d, -d), (-d, d, -d),
             (-d, -d, d), (d, -d, d), (d, d, d), (-d, d, d)]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return verts, faces


def mesh_object(name, verts, faces, material, tag, location=(0, 0, 0), rotation=None,
                scale=(1, 1, 1), data=None):
    if data is None:
        data = bpy.data.meshes.new(name)
        data.from_pydata(verts, [], faces)
        data.materials.append(material)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    if rotation is not None:
        obj.rotation_mode = 'AXIS_ANGLE'
        obj.rotation_axis_angle = rotation
    obj.scale = scale
    obj.cycles.coherent_interface = tag
    return obj


def world_triangles(obj):
    matrix = obj.matrix_world
    return [[list(map(float, matrix @ obj.data.vertices[i].co)) for i in poly.vertices]
            for poly in obj.data.polygons]


def outward_triangles(obj):
    """World triangles in native outward orientation (negative scale reverses)."""
    tris = world_triangles(obj)
    if obj.matrix_world.determinant() < 0:
        tris = [[t[0], t[2], t[1]] for t in tris]
    return tris


def add_sources(positions, wavelength_nm, coherence_m):
    sources = []
    for position in positions:
        bpy.ops.object.light_add(type='POINT', location=position)
        light = bpy.context.object
        light.data.energy = 0.001
        light.data.shadow_soft_size = 0
        light.data.cycles.coherence_group = 1
        light.data.cycles.coherence_wavelength_nm = wavelength_nm
        light.data.cycles.coherence_length_m = coherence_m
        light.data.cycles.coherence_phase = 0.0
        sources.append(light)
    return sources


def add_camera(scene, receiver, normal, footprint):
    bpy.ops.object.camera_add(location=Vector(receiver) + Vector(normal) * 0.004)
    camera = bpy.context.object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = footprint
    camera.data.clip_start = 0.001
    camera.rotation_euler = (Vector(receiver) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera
    return camera


def base_scene():
    scene, *_ = configure_scene(True)
    mirror_material = bpy.data.materials['Unit perfect mirror']
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.cycles.coherent_transport_mode = 'FACET_SINGLE_REFLECTION'
    scene.cycles.use_coherent_specular_connections = True
    scene.cycles.coherent_polarization_mode = 'VECTOR'
    scene.cycles.coherent_max_interface_events = 2
    # Direct, all one-event and two-event routes; nothing longer natively.
    scene.cycles.max_bounces = 3
    scene.cycles.bdpt_max_bounces = 3
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.diffuse_bounces = 0
    scene.cycles.transparent_max_bounces = 0
    scene.cycles.samples = 512
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = RESOLUTION
    return scene, mirror_material


def record(folder, name, scene, objects, detector, sources, camera, max_events):
    bpy.context.view_layer.update()
    path = folder / (name + '.blend')
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    matrix = np.array([list(row) for row in camera.matrix_world])
    kinds = {'GLASS': 'glass', 'MIRROR': 'mirror', 'DETECTOR': 'detector'}
    geometry = {
        'variant': name,
        'blend': str(path),
        'blend_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'max_events': max_events,
        'objects': [],
        'sources': [],
        'detector_object': detector.name,
        'detector_albedo': 1.0,
        'detector_normal': list(map(float, (detector.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized())),
        'camera': {'detector_center': list(map(float, detector.matrix_world.translation)),
                   'right': matrix[:3, 0].tolist(), 'up': matrix[:3, 1].tolist(),
                   'ortho_scale': float(camera.data.ortho_scale), 'resolution': RESOLUTION},
        'budgets': {k: getattr(scene.cycles, k) for k in (
            'max_bounces', 'glossy_bounces', 'transmission_bounces', 'diffuse_bounces',
            'bdpt_max_bounces', 'coherent_max_interface_events')},
    }
    for obj in objects + [detector]:
        entry = {'name': obj.name, 'kind': kinds[obj.cycles.coherent_interface],
                 'triangles': outward_triangles(obj),
                 'negative_scale': obj.matrix_world.determinant() < 0,
                 'linked_users': obj.data.users}
        if entry['kind'] == 'glass':
            node = [n for n in obj.data.materials[0].node_tree.nodes if n.type == 'BSDF_GLASS'][0]
            entry['ior'] = float(node.inputs['IOR'].default_value)
        geometry['objects'].append(entry)
    for light in sources:
        c = light.data.cycles
        geometry['sources'].append({'position': list(map(float, light.matrix_world.translation)),
                                    'power_W': float(light.data.energy),
                                    'phase_rad': float(c.coherence_phase),
                                    'group': int(c.coherence_group),
                                    'wavelength_m': float(c.coherence_wavelength_nm) * 1e-9,
                                    'coherence_length_m': float(c.coherence_length_m)})
    (folder / (name + '_geometry.json')).write_text(json.dumps(geometry, indent=1) + '\n')
    return geometry


def author_prism(output):
    folder = output / 'prism_TT'
    folder.mkdir()
    variants = ['phase_0', 'phase_pi', 'distinct', 'partial_coherence', 'negative_scale', 'linked_instance']
    for name in variants:
        scene, _ = base_scene()
        material, _ = glass('Ideal n1.5 prism', 1.5)
        verts, faces = prism_mesh(0.008, 0.0139, 0.02)
        scale = (-1, 1, 1) if name == 'negative_scale' else (1, 1, 1)
        prism = mesh_object('Glass prism', verts, faces, material, 'GLASS', location=(0, 0, 0),
                            rotation=(0.0, 0.0, 0.0, 1.0), scale=scale)
        objects = [prism]
        if name == 'linked_instance':
            # A second user of the same mesh far below: an instanced Glass volume
            # whose facets the oracle includes, and which blocks nothing here.
            objects.append(mesh_object('Glass prism linked copy', None, None, None, 'GLASS',
                                       location=(0.0, 0.3, -0.2), rotation=(0.4, 0.0, 1.0, 0.3),
                                       data=prism.data))
        receiver = np.array([0.03, 0.0, -0.01])
        towards = np.array([-0.03, 0.0, 0.004]) - receiver
        towards = towards / np.linalg.norm(towards) + np.array([0.0, 0.0, 0.25])
        normal = towards / np.linalg.norm(towards)
        u = np.cross(normal, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        detector = pane('Unit Lambertian detector', receiver, u, v, 0.002,
                        set_diffuse_material('Unit detector', (1, 1, 1)), 'DETECTOR')
        coherence = 0.0025 if name == 'partial_coherence' else 10.0
        sources = add_sources([(-0.03, 0.0, 0.004), (-0.03, 0.0006, 0.0046)], 100000.0, coherence)
        if name == 'phase_pi':
            sources[1].data.cycles.coherence_phase = math.pi
        if name == 'distinct':
            sources[1].data.cycles.coherence_group = 2
        camera = add_camera(scene, receiver, normal, 0.0003)
        record(folder, name, scene, objects, detector, sources, camera, 2)


def author_cube(output):
    folder = output / 'cube_R'
    folder.mkdir()
    for name in ['phase_0', 'phase_pi', 'distinct']:
        scene, mirror_material = base_scene()
        material, _ = glass('Ideal n1.5 cube', 1.5)
        verts, faces = cube_mesh(0.012)
        axis = np.array([0.3, 0.2, 1.0])
        axis /= np.linalg.norm(axis)
        cube = mesh_object('Glass cube', verts, faces, material, 'GLASS', location=(0, 0, 0),
                           rotation=(math.radians(35.0), *axis))
        h = 0.012
        mirror = mesh_object('Finite mirror', [(-h, -h, 0.03), (h, -h, 0.03), (h, h, 0.03), (-h, h, 0.03)],
                             [(0, 1, 2), (0, 2, 3)], mirror_material, 'MIRROR')
        receiver = np.array([0.015, -0.001, 0.012])
        source = np.array([-0.015, 0.002, 0.02])
        towards = source - receiver
        towards = towards / np.linalg.norm(towards) + np.array([0.0, 0.0, 0.3])
        normal = towards / np.linalg.norm(towards)
        u = np.cross(normal, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        detector = pane('Unit Lambertian detector', receiver, u, v, 0.002,
                        set_diffuse_material('Unit detector', (1, 1, 1)), 'DETECTOR')
        sources = add_sources([tuple(source), tuple(source + np.array([0.0004, -0.0003, 0.0002]))],
                              100000.0, 10.0)
        if name == 'phase_pi':
            sources[1].data.cycles.coherence_phase = math.pi
        if name == 'distinct':
            sources[1].data.cycles.coherence_group = 2
        camera = add_camera(scene, receiver, normal, 0.0003)
        record(folder, name, scene, [cube, mirror], detector, sources, camera, 2)


def main():
    output = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    author_prism(output)
    author_cube(output)
    print('authored', output)


if __name__ == '__main__':
    main()
