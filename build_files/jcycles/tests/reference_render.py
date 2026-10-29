# Runs in this repository's Blender: opens the file saved by host_scene.py, prepares it the way
# the JCycles worker does, renders with CYCLES directly and writes a multilayer EXR.
#   blender -b --factory-startup --python reference_render.py -- <addon_dir> <scene.blend> <out.exr>
import importlib.util
import os
import sys

import bpy

addon_dir, blend, output = sys.argv[sys.argv.index("--") + 1:][:3]
sys.argv = [sys.argv[0], "--", "--addon-dir", addon_dir, "--session-dir", os.path.dirname(output)]
spec = importlib.util.spec_from_file_location("jcy_worker_ref", os.path.join(addon_dir, "jcy_worker", "worker_main.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker._args.update(worker.parse_args())
bpy.ops.wm.open_mainfile(filepath=blend, load_ui=False)
for owner in worker.all_property_owners():
    worker.copy_property_groups(owner)
worker.register_node_classes()
print("REF converted nodes:", worker.convert_nodes())
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
if scene.cycles.device == 'GPU':
    import _cycles
    types = dict(zip(("CUDA", "OPTIX", "HIP", "METAL", "ONEAPI", "HIPRT"), _cycles.get_device_types()))
    gpu = next((t for t in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI") if types.get(t)), None)
    worker.apply_device_settings({"preferences": {"compute_device_type": gpu}, "devices": []})
    prefs = bpy.context.preferences.addons["cycles"].preferences
    print("REF devices", [(d.type, d.name, d.use) for d in prefs.devices])
for vl in scene.view_layers:
    vl.use = True
scene.render.use_compositing = False
worker.set_multilayer_exr(scene)
bpy.ops.render.render(scene=scene.name)
bpy.data.images["Render Result"].save_render(output, scene=scene)
print("REF_RENDER", output, "samples", scene.cycles.samples, "device", scene.cycles.device)
