# Runs in this repository's Blender: the same bake as host_features.py bake, done natively.
#   blender -b --factory-startup --python reference_bake.py -- <addon_dir> <bake.blend> <out.exr>
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
worker.convert_nodes()
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
print("REF_BAKE", bpy.ops.object.bake(type='COMBINED'))
image = bpy.data.images["BakeTarget"]
image.filepath_raw = output
image.file_format = 'OPEN_EXR'
image.save()
