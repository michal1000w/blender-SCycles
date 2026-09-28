#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Author real sphere-entry/exit interference scenes; never render or invent reference pixels.

These fixtures exercise the curved-transmission work in progress. The v37
renderer deliberately rejects Glass point spheres; the manifest is not a pass.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycles_coherent_mirror_acceptance import configure_scene, set_diffuse_material
from cycles_coherent_sphere_acceptance import validate_native_points


def main():
    output = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene, *_ = configure_scene(True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.cycles.use_coherent_specular_connections = True
    scene.cycles.coherent_polarization_mode = 'VECTOR'
    scene.cycles.coherent_max_interface_events = 2
    scene.cycles.max_bounces = 3
    scene.cycles.bdpt_max_bounces = 3
    scene.cycles.transmission_bounces = 2
    scene.cycles.glossy_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.samples = 128
    scene.render.resolution_x = scene.render.resolution_y = 256

    mesh = bpy.data.meshes.new('Exact negative-X detector')
    mesh.from_pydata([(.4,-.04,-.04),(.4,-.04,.04),(.4,.04,.04),(.4,.04,-.04)],
                    [], [(0,1,2),(0,2,3)])
    detector = bpy.data.objects.new('Unit first-encounter detector', mesh)
    scene.collection.objects.link(detector)
    mesh.materials.append(set_diffuse_material('Unit Lambertian', (1,1,1)))
    detector.cycles.coherent_interface = 'DETECTOR'

    material = bpy.data.materials.new('Ideal lossless sphere n=1.5')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.distribution = 'GGX'
    glass.inputs['Color'].default_value = (1,1,1,1)
    glass.inputs['Roughness'].default_value = 0
    glass.inputs['IOR'].default_value = 1.5
    out = nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(glass.outputs[0], out.inputs['Surface'])
    mesh = bpy.data.meshes.new('One native sphere center')
    mesh.from_pydata([(0,0,0)], [], [])
    mesh.materials.append(material)
    sphere = bpy.data.objects.new('Native analytic Glass sphere', mesh)
    sphere.cycles.coherent_interface = 'GLASS'
    scene.collection.objects.link(sphere)
    modifier = sphere.modifiers.new('Actual point-sphere geometry', 'NODES')
    group = bpy.data.node_groups.new('Glass sphere point component', 'GeometryNodeTree')
    modifier.node_group = group
    group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    inp = group.nodes.new('NodeGroupInput')
    out = group.nodes.new('NodeGroupOutput')
    points = group.nodes.new('GeometryNodeMeshToPoints')
    points.name = 'NativeSphere'
    points.mode = 'VERTICES'
    points.inputs['Radius'].default_value = .25
    mat = group.nodes.new('GeometryNodeSetMaterial')
    mat.inputs['Material'].default_value = material
    group.links.new(inp.outputs['Geometry'], points.inputs['Mesh'])
    group.links.new(points.outputs['Points'], mat.inputs['Geometry'])
    group.links.new(mat.outputs['Geometry'], out.inputs['Geometry'])
    sources = []
    for y in (-1e-5, 1e-5):
        bpy.ops.object.light_add(type='POINT', location=(-1,y,.05))
        light = bpy.context.object
        light.data.energy = 100
        light.data.shadow_soft_size = 0
        light.data.cycles.coherence_group = 1
        light.data.cycles.coherence_length_m = 1e-4
        light.data.cycles.coherence_wavelength_nm = 550
        light.data.cycles.coherence_phase = 0
        sources.append(light)
    bpy.ops.object.camera_add(location=(.3,0,0))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((.4,0,0))-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.clip_start = .001
    camera.data.clip_end = 10
    camera.data.ortho_scale = .06
    scene.camera = camera
    bpy.context.view_layer.update()
    native = validate_native_points(sphere)
    variants = {}
    for name, phase, group_id in [('phase_0',0,1),('phase_pi',math.pi,1),
                                  ('incoherent_connector_control',0,2)]:
        sources[1].data.cycles.coherence_phase = phase
        sources[1].data.cycles.coherence_group = group_id
        path = output / (name+'.blend')
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        variants[name] = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {'status':'Authored fixtures only; curved transmission not yet accepted',
                'rendered':False,'reference_generated':False,'native_geometry_validation':native,
                'binary':bpy.app.binary_path,'variants':variants,
                'sources_m':[list(s.location) for s in sources],
                'radius_m':.25,'sphere_ior':1.5,'detector_x_m':float(detector.data.vertices[0].co.x),
                'resolution':[256,256],'samples':128,'adaptive':False,'denoising':False,
                'path_inventory':'Two transmissions through one analytic sphere; glossy/diffuse continuations disabled; direct source rays blocked by sphere.',
                'acceptance_required':'Independent complete stationary-root inventory, Jones fields, curvature spreading, compensated phase, visibility and Metal PT/BDPT/guiding comparisons. No PT brightness fitting.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(output),'rendered':False,'variants':len(variants)}))


if __name__ == '__main__':
    main()
