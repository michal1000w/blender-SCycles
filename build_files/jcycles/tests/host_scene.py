# Runs in an official Blender with the JCycles add-on installed: builds a test scene that uses
# fork-only settings and nodes, saves it and renders it with JCycles.
#   blender -b --factory-startup --python host_scene.py -- <out_dir> <device CPU|GPU>
import os
import sys

import addon_utils
import bpy

out_dir, device = sys.argv[sys.argv.index("--") + 1:][:2]
os.makedirs(out_dir, exist_ok=True)
mod = addon_utils.enable("JCycles", default_set=True)
assert mod is not None, "JCycles add-on failed to enable"

bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("JCycles", default_set=True)
scene = bpy.context.scene
scene.render.engine = 'JCYCLES'
scene.render.resolution_x, scene.render.resolution_y = 160, 120
scene.render.resolution_percentage = 100
cs = scene.jcycles
cs.device = device
cs.samples = 24
cs.use_denoising = False
cs.seed = 7

bpy.ops.mesh.primitive_plane_add(size=6)
plane = bpy.context.object
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 1))
sphere = bpy.context.object
bpy.ops.object.shade_smooth()
bpy.ops.mesh.primitive_cube_add(size=1, location=(2, 0.5, 0.5))
cube = bpy.context.object

def material(name):
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    return mat

# Fork node: Diffraction BSDF (only exists in the fork).
mat = material("Diffraction")
nt = mat.node_tree
out = next(n for n in nt.nodes if n.bl_idname == 'ShaderNodeOutputMaterial')
for n in list(nt.nodes):
    if n != out:
        nt.nodes.remove(n)
d = nt.nodes.new("JCyclesShaderNodeBsdfDiffraction")
d.inputs["Roughness"].default_value = 0.1
nt.links.new(d.outputs[0], out.inputs["Surface"])
sphere.data.materials.append(mat)

# Extended node: Glass with the fork's diffraction inputs, value driven by a texture link.
mat2 = material("GlassFork")
nt = mat2.node_tree
out = next(n for n in nt.nodes if n.bl_idname == 'ShaderNodeOutputMaterial')
for n in list(nt.nodes):
    if n != out:
        nt.nodes.remove(n)
g = nt.nodes.new("JCyclesShaderNodeBsdfGlass")
g.inputs["Diffraction Weight"].default_value = 0.8
noise = nt.nodes.new("ShaderNodeTexNoise")
nt.links.new(noise.outputs["Color"], g.inputs["Color"])
nt.links.new(g.outputs[0], out.inputs["Surface"])
cube.data.materials.append(mat2)

bpy.ops.object.light_add(type='AREA', location=(2, -2, 4))
bpy.context.object.data.energy = 600
bpy.ops.object.camera_add(location=(6, -6, 4), rotation=(1.1, 0, 0.785))
scene.camera = bpy.context.object
world = bpy.data.worlds.new("W")
scene.world = world
world.color = (0.05, 0.05, 0.08)

vl = scene.view_layers[0]
vl.use_pass_diffuse_direct = True
vl.use_pass_z = True
vl.use_pass_cryptomatte_object = True
vl.jcycles.use_pass_volume_direct = True
second = scene.view_layers.new("Second")
second.use_pass_normal = True

if hasattr(scene.render.image_settings, 'media_type'):
    scene.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
scene.render.image_settings.color_depth = '32'
scene.render.filepath = os.path.join(out_dir, "host_####")
scene.render.use_compositing = False
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "scene.blend"))
result = bpy.ops.render.render(write_still=True)
print("HOST_RENDER", result)
