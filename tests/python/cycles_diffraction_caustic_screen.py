# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Isolate a reflected diffraction caustic on a separate diffuse receiver.

The finite area emitter broadens the ideal normal-incidence prediction. The
prediction is a geometric reference, not an absolute radiance acceptance test.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--transport',choices=('pt','bdpt','guided','bdpt_guided'),default='bdpt')
    p.add_argument('--samples',type=int,default=1024)
    p.add_argument('--render',action='store_true')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if a.samples<1:p.error('Samples must be positive')
    a.output=a.output.resolve()
    if a.output.exists():p.error('Use a new output directory')
    a.output.mkdir(parents=True)
    bpy.context.preferences.filepaths.file_preview_type='NONE'
    scene=bpy.context.scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene.render.engine='CYCLES'
    scene.cycles.samples=a.samples
    scene.cycles.seed=11
    scene.cycles.use_adaptive_sampling=False
    scene.cycles.use_denoising=False
    scene.cycles.time_limit=0
    scene.cycles.use_layer_samples='IGNORE'
    scene.cycles.use_bidirectional_path_tracing=a.transport in ('bdpt','bdpt_guided')
    scene.cycles.use_guiding=a.transport in ('guided','bdpt_guided')
    scene.cycles.max_bounces=8
    scene.cycles.diffuse_bounces=4
    scene.cycles.glossy_bounces=4
    scene.cycles.transmission_bounces=4
    scene.cycles.sample_clamp_direct=0
    scene.cycles.sample_clamp_indirect=0
    scene.render.resolution_x=768
    scene.render.resolution_y=768
    scene.render.resolution_percentage=100
    scene.render.film_transparent=False
    scene.view_settings.view_transform='AgX'
    scene.view_settings.exposure=0
    scene.view_settings.gamma=1
    world=bpy.data.worlds.new('Black environment')
    world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=0
    scene.world=world

    def plane(name,location,rotation,scale,material):
        bpy.ops.mesh.primitive_plane_add(size=1,location=location,rotation=rotation)
        obj=bpy.context.object
        obj.name=name
        obj.scale=scale
        obj.data.materials.append(material)
        return obj

    grating=bpy.data.materials.new('740 nm physical lamellar conductor')
    grating.use_nodes=True
    nodes=grating.node_tree.nodes
    nodes.clear()
    output=nodes.new('ShaderNodeOutputMaterial')
    bsdf=nodes.new('ShaderNodeBsdfDiffraction')
    bsdf.quality='REALISTIC'
    bsdf.name='Physical grating'
    properties=dict(pitch=740,depth=150,duty_cycle=.41,incident_ior=1,ridge_ior=.9,
                    ridge_extinction=6,groove_ior=1,substrate_ior=.9,substrate_extinction=6)
    for name,value in properties.items():setattr(bsdf,name,value)
    tangent=nodes.new('ShaderNodeCombineXYZ')
    tangent.inputs['X'].default_value=1
    grating.node_tree.links.new(tangent.outputs[0],bsdf.inputs['Tangent'])
    grating.node_tree.links.new(bsdf.outputs[0],output.inputs['Surface'])
    tile=plane('Narrow grating strip',(0,0,0),(0,0,0),(.001,.025,1),grating)

    receiver=bpy.data.materials.new('Neutral diffuse receiver')
    receiver.use_nodes=True
    nodes=receiver.node_tree.nodes
    nodes.clear()
    output=nodes.new('ShaderNodeOutputMaterial')
    diffuse=nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.inputs['Color'].default_value=(.8,.8,.8,1)
    receiver.node_tree.links.new(diffuse.outputs[0],output.inputs['Surface'])
    screen=plane('Caustic screen',(.04,0,.04),(0,-math.pi/2,0),(.08,.07,1),receiver)

    light_data=bpy.data.lights.new('Restricted white area source','AREA')
    light_data.energy=5
    light_data.shape='SQUARE'
    light_data.size=.0005
    light_data.spread=math.radians(12)
    light=bpy.data.objects.new('Restricted white area source',light_data)
    scene.collection.objects.link(light)
    light.location=(0,0,.2)
    # Local -Z points down toward the strip. The source cone cannot reach the screen.
    light.rotation_euler=(0,0,0)
    camera_data=bpy.data.cameras.new('Camera')
    camera=bpy.data.objects.new('Camera',camera_data)
    scene.collection.objects.link(camera)
    camera.location=(-.075,-.115,.095)
    camera.rotation_euler=(Vector((.025,0,.035))-camera.location).to_track_quat('-Z','Y').to_euler()
    camera_data.type='ORTHO'
    camera_data.ortho_scale=.115
    camera_data.clip_start=.001
    camera_data.clip_end=10
    scene.camera=camera
    ideal=[dict(wavelength_nm=w,screen_height_m=.04*math.sqrt((740/w)**2-1))
           for w in (380,450,550,650,700,730)]
    notes=('A 1 x 25 mm grating strip at z=0 illuminates a diffuse screen at x=40 mm.\n'
           'The restricted finite area source is at z=200 mm and points downward.\n'
           'Its direct cone misses the screen; reflected +1 order transport is isolated.\n'
           'For a central normally incident ray, z_screen = 0.04*sqrt((740/lambda_nm)^2-1).\n'
           'Finite source size and strip width broaden this geometric prediction.\n'
           'Generic constant-index lamellar conductor, no pits, cover or cross-object coherence.\n'
           'The two separate surfaces test radiometric caustic transport, not coherent interference.\n'
           'A brighter BDPT result is not a failure merely because PT misses rare light paths.\n')
    bpy.data.texts.new('READ ME - caustic measurement').write(notes)
    prefix=a.output/('caustic_screen_'+a.transport)
    scene.render.image_settings.file_format='OPEN_EXR'
    scene.render.image_settings.color_depth='32'
    scene.render.filepath=str(prefix.with_suffix('.exr'))
    bpy.context.view_layer.update()
    normal=screen.matrix_world.to_3x3()@Vector((0,0,1))
    assert normal.normalized().dot(Vector((-1,0,0)))>.999999
    assert (camera.location-screen.location).dot(normal)>0
    direct_cone_radius=.00025+.2*math.tan(light_data.spread/2)
    assert direct_cone_radius<.04
    material_name=grating.name
    bpy.ops.wm.save_as_mainfile(filepath=str(prefix.with_suffix('.blend')))
    bpy.ops.wm.open_mainfile(filepath=str(prefix.with_suffix('.blend')))
    scene=bpy.context.scene
    restored=bpy.data.materials[material_name].node_tree.nodes['Physical grating']
    assert restored.inputs['Tangent'].is_linked and abs(restored.pitch-740)<1e-6
    assert scene.cycles.samples==a.samples and not scene.cycles.use_adaptive_sampling
    assert scene.cycles.use_bidirectional_path_tracing==(a.transport in ('bdpt','bdpt_guided'))
    assert scene.cycles.use_guiding==(a.transport in ('guided','bdpt_guided'))
    report=dict(status='saved_not_rendered',transport_requested=a.transport,samples=a.samples,
                description=notes,properties={name:getattr(restored,name) for name in properties},
                ideal_normal_incidence_screen_positions=ideal,
                direct_source_cone_maximum_radius_m=direct_cone_radius,
                screen_x_m=.04,serialization_verified=True,coherent_transport=False,
                binary_sha256=hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                blend_sha256=hashlib.sha256(prefix.with_suffix('.blend').read_bytes()).hexdigest())
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if a.render:
        preferences=bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type='METAL'
        preferences.get_devices()
        for device in preferences.devices:device.use=device.type=='METAL'
        if not any(d.use for d in preferences.devices):raise RuntimeError('No Metal device')
        scene.cycles.device='GPU'
        bpy.ops.render.render(write_still=True)
        image=bpy.data.images.load(str(prefix.with_suffix('.exr')),check_existing=False)
        pixels=tuple(image.pixels[:])
        if not pixels or not all(math.isfinite(v) for v in pixels):raise RuntimeError('Invalid render')
        scene.render.image_settings.file_format='PNG'
        scene.render.image_settings.color_depth='8'
        image.save_render(str(prefix.with_suffix('.preview.png')),scene=scene)
        report.update(status='rendered_pending_numerical_and_visual_review',
                      exr_sha256=hashlib.sha256(prefix.with_suffix('.exr').read_bytes()).hexdigest(),
                      preview_sha256=hashlib.sha256(prefix.with_suffix('.preview.png').read_bytes()).hexdigest())
        prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
