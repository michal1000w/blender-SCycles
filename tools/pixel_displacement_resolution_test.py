"""Metal render regression: run Blender --background --factory-startup --python this_file.

Uses a small UV domain to exercise 16K sampling density within the cache memory budget.
"""

import os
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pixel_displacement_benchmark import create_scene


def main():
    assert not bpy.context.scene.cycles.use_pixel_displacement_resolution_clamp
    os.environ["PIXEL_DISPLACEMENT_BENCH_DEVICE"] = "GPU"
    create_scene(grid_size=2, texture_size=64, samples=1, resolution=64)
    prefs = bpy.context.preferences.addons["cycles"].preferences
    assert any(device.type == "METAL" and device.use for device in prefs.devices), "Metal required"

    scene = bpy.context.scene
    scene.cycles.seed = 0
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    obj = bpy.data.objects["pixel_displacement_grid"]
    for uv in obj.data.uv_layers.active.data:
        uv.uv /= 128.0
    # Preserve the original texture coordinates while varying the micromesh density.
    tree = obj.data.materials[0].node_tree
    texture = next(node for node in tree.nodes if node.type == "TEX_IMAGE")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    scale = tree.nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs[3].default_value = 128.0
    tree.links.new(coordinates.outputs["UV"], scale.inputs[0])
    tree.links.new(scale.outputs[0], texture.inputs["Vector"])

    with tempfile.TemporaryDirectory(prefix="pixel-resolution-test-") as directory:
        def render(enabled, resolution, label):
            scene.cycles.use_pixel_displacement_resolution_clamp = enabled
            scene.cycles.pixel_displacement_resolution = resolution
            assert scene.cycles.pixel_displacement_resolution == resolution
            scene.render.filepath = str(Path(directory) / (label + ".exr"))
            print(f"RESOLUTION_TEST_START {label}", flush=True)
            bpy.ops.render.render(write_still=True)
            result = bpy.data.images.load(scene.render.filepath, check_existing=False)
            pixels = np.empty(len(result.pixels), dtype=np.float32)
            result.pixels.foreach_get(pixels)
            bpy.data.images.remove(result)
            assert np.isfinite(pixels).all(), label
            assert pixels.size and pixels.max() > 0, label
            print(f"RESOLUTION_TEST_RENDER {label}", flush=True)
            return pixels

        unclamped_low = render(False, 64, "unclamped_low")
        unclamped_high = render(False, 16384, "unclamped_high")
        np.testing.assert_array_equal(unclamped_low, unclamped_high)
        clamped_low = render(True, 2048, "clamped_2k")
        clamped_high = render(True, 16384, "clamped_16k")
        assert np.max(np.abs(clamped_high - clamped_low)) > 1e-4, "16K must affect rendered detail"
        unclamped_again = render(False, 16384, "unclamped_again")
        np.testing.assert_array_equal(unclamped_high, unclamped_again)
        clamped_again = render(True, 16384, "clamped_again")
        np.testing.assert_array_equal(clamped_high, clamped_again)

    print("PASS: resolution ignored when disabled, 16K detail, and cache invalidation in both directions")


if __name__ == "__main__":
    main()
