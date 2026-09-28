#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare eight bounded Thin Wall unit-world gates; no rendering.

Reuse the existing Principled furnace scene construction. Uniform unit world
radiance is an independent lossless white-sheet reference, not a PT comparison.
The first seven fixtures exercise nonzero diffraction, and zero_depth_front is
an exact native-route control. Fixed absolute mean tolerance is 0.02 per RGB.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--")+1:]
output = Path(args[0]).resolve()
output.mkdir(parents=True, exist_ok=False)
helper = Path(__file__).with_name("cycles_diffraction_principled_multiggx_furnace.py")
original_argv = sys.argv[:]
sys.argv = [str(helper), "--", str(output / "reused_base_fixture")]
runpy.run_path(str(helper), run_name="__main__")
sys.argv = original_argv
base = output / "reused_base_fixture" / "two_sided_front.blend"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

specs = (
    ("rough1_front", 1.0, 1.0, 125.0, (0.6,0.35,3.0)),
    ("rough1_back", 1.0, 1.0, 125.0, (0.6,0.35,-3.0)),
    ("rough06_front", 0.6, 1.0, 125.0, (0.6,0.35,3.0)),
    ("rough06_back", 0.6, 1.0, 125.0, (0.6,0.35,-3.0)),
    ("smooth_front", 0.0, 1.0, 125.0, (0.6,0.35,3.0)),
    ("grazing_front", 0.6, 1.0, 125.0, (2.5,0.1,0.55)),
    ("partial_front", 0.6, 0.5, 125.0, (0.6,0.35,3.0)),
    ("zero_depth_front", 0.6, 1.0, 0.0, (0.6,0.35,3.0)),
)
scenes = {}
for name, roughness, coverage, depth, camera_location in specs:
    bpy.ops.wm.open_mainfile(filepath=str(base))
    scene = bpy.context.scene
    node = bpy.data.materials["Neutral white Principled diffraction"].node_tree.nodes.get("Principled BSDF")
    node.distribution = "GGX"
    node.inputs["Thin Wall"].default_value = True
    node.inputs["Metallic"].default_value = 0.0
    node.inputs["Transmission Weight"].default_value = 1.0
    node.inputs["Roughness"].default_value = roughness
    node.inputs["Diffraction Weight"].default_value = coverage
    node.inputs["Diffraction Depth"].default_value = depth
    camera = scene.camera
    camera.location = camera_location
    camera.rotation_euler = (-Vector(camera_location)).to_track_quat("-Z", "Y").to_euler()
    side = name.rsplit("_",1)[1]
    scene["furnace_model"] = name.rsplit("_",1)[0]
    scene["furnace_side"] = side
    scene["furnace_scope"] = "white Thin Wall, ordinary GGX; completed cached nonzero phase screen or exact native zero-depth control"
    scene["furnace_expected_unit_radiance"] = 1.0
    assert scene.cycles.samples == 256 and not scene.cycles.use_adaptive_sampling and not scene.cycles.use_denoising
    scene.render.filepath = "//"+name
    bpy.context.view_layer.update()
    path = output/(name+".blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    scenes[name] = {"path":str(path),"sha256":digest(path),"roughness":roughness,"diffraction_weight":coverage,
                    "depth_nm":depth,"side":side,"camera_location":list(camera_location),
                    "expected_linear_rgb_radiance":[1.0,1.0,1.0],"energy_tolerance_absolute_each_rgb":0.02}
manifest = {"scope":"eight independent unit-world Thin Wall numerical cases; fixed64x64/256, adaptive and denoise OFF",
            "binary":str(Path(bpy.app.binary_path).resolve()),"binary_sha256":digest(bpy.app.binary_path),
            "script":str(Path(__file__).resolve()),"script_sha256":digest(__file__),
            "reused_constructor":str(helper.resolve()),"reused_constructor_sha256":digest(helper),
            "energy_tolerance_absolute_each_rgb":0.02,"front_back_pairs":[["rough1_front","rough1_back"],["rough06_front","rough06_back"]],
            "theory":"uniform unit environment is preserved by a lossless complete sheet after both exterior-air ports",
            "scenes":scenes}
(output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("THIN_WALL_FURNACE "+str(output),flush=True)
