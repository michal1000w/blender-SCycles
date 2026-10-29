# Runs in an official Blender with JCycles installed. Feature tests on the scene from host_scene.py:
#   blender -b --factory-startup <scene.blend> --python host_features.py -- <out_dir> <osl|anim|bake>
import os
import sys

import addon_utils
import bpy

out_dir, mode = sys.argv[sys.argv.index("--") + 1:][:2]
addon_utils.enable("JCycles", default_set=True)
scene = bpy.context.scene
scene.render.engine = 'JCYCLES'
settings = scene.render.image_settings
if hasattr(settings, "media_type"):
    settings.media_type = 'MULTI_LAYER_IMAGE'
settings.file_format = 'OPEN_EXR_MULTILAYER'
settings.color_depth = '32'


def report(kind, message):
    print("REPORT", kind, message)


if mode == "osl":
    scene.jcycles.shading_system = True
    text = bpy.data.texts.new("checker.osl")
    text.write('shader checker(float Scale = 4.0, color A = color(0.9, 0.2, 0.1), output color Out = 0)\n'
               '{ point p = P * Scale; int c = (int(floor(p[0])) + int(floor(p[1]))) & 1; Out = c ? A : color(0.1); }\n')
    mat = bpy.data.objects["Plane"].active_material or bpy.data.materials.new("PlaneMat")
    if mat.node_tree is None:
        mat.use_nodes = True
    bpy.data.objects["Plane"].data.materials.clear()
    bpy.data.objects["Plane"].data.materials.append(mat)
    nt = mat.node_tree
    script = nt.nodes.new("ShaderNodeScript")
    script.mode = 'INTERNAL'
    script.script = text
    from JCycles import osl
    ok = osl.update_script_node(script, report)
    print("OSL compiled", ok, "inputs", [s.name for s in script.inputs], "outputs", [s.name for s in script.outputs],
          "bytecode", len(script.bytecode))
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    nt.links.new(script.outputs["Out"], bsdf.inputs["Base Color"])
    scene.render.filepath = os.path.join(out_dir, "osl_####")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "osl.blend"))
    print("OSL_RENDER", bpy.ops.render.render(write_still=True))

elif mode == "anim":
    sphere = bpy.data.objects["Sphere"]
    sphere.location.x = -1.5
    sphere.keyframe_insert("location", frame=1)
    sphere.location.x = 1.5
    sphere.keyframe_insert("location", frame=3)
    scene.frame_start, scene.frame_end = 1, 3
    scene.render.filepath = os.path.join(out_dir, "anim_####")
    from JCycles import jcy_engine
    loads = []
    original_ensure = jcy_engine.ensure_loaded

    def counting(worker, snapshot, should_cancel=None):
        if worker.loaded_token != snapshot[0]:
            loads.append(snapshot[0])
        return original_ensure(worker, snapshot, should_cancel)

    jcy_engine.ensure_loaded = counting
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "anim.blend"))
    print("ANIM_RENDER", bpy.ops.render.render(animation=True), "scene loads in worker:", len(loads))

elif mode == "bake":
    plane = bpy.data.objects["Plane"]
    mat = bpy.data.materials.new("BakeMat")
    if mat.node_tree is None:
        mat.use_nodes = True
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    image = bpy.data.images.new("BakeTarget", 128, 128, float_buffer=True)
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    mat.node_tree.nodes.active = node
    for obj in scene.objects:
        obj.select_set(obj == plane)
    bpy.context.view_layer.objects.active = plane
    scene.render.bake.margin = 4
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "bake.blend"))
    result = bpy.ops.object.bake(type='COMBINED')
    print("BAKE", result)
    image.filepath_raw = os.path.join(out_dir, "bake_host.exr")
    image.file_format = 'OPEN_EXR'
    image.save()
