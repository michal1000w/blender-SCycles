#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Procedural volume guiding fixtures with no external assets.

Uses the common guiding-room CLI. Select --scene volume, volume_dense,
volume_heterogeneous, or volume_backward. Every variant uses identical geometry
and transport settings on CPU and GPU; only the participating medium changes.
"""

import importlib.util
import pathlib
import sys

import bpy


PROFILES = {
    "volume": (0.12, 0.65, False),
    "volume_dense": (0.6, 0.65, False),
    "volume_heterogeneous": (0.4, 0.65, True),
    "volume_backward": (0.25, -0.65, False),
}


def main():
    base_path = pathlib.Path(__file__).with_name("cycles_metal_guiding_scene.py")
    spec = importlib.util.spec_from_file_location("cycles_guiding_volume_room", base_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    arguments = sys.argv[sys.argv.index("--") + 1:]
    index = arguments.index("--scene") + 1
    variant = arguments[index]
    density, anisotropy, heterogeneous = PROFILES[variant]
    arguments[index] = "volume"
    sys.argv = [str(base_path), "--", *arguments]
    original_box, original_build = generator.box, generator.build_scene

    def box(name, location, scale, mat):
        if name == "Medium boundary":
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            scatter = next(node for node in nodes if node.type == "VOLUME_SCATTER")
            scatter.inputs["Density"].default_value = density
            scatter.inputs["Anisotropy"].default_value = anisotropy
            if heterogeneous:
                coordinates = nodes.new("ShaderNodeTexCoord")
                noise = nodes.new("ShaderNodeTexNoise")
                noise.inputs["Scale"].default_value = 2.0
                noise.inputs["Detail"].default_value = 2.0
                ramp = nodes.new("ShaderNodeMapRange")
                ramp.inputs["To Min"].default_value = 0.02
                ramp.inputs["To Max"].default_value = density
                links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
                links.new(noise.outputs["Fac"], ramp.inputs["Value"])
                links.new(ramp.outputs["Result"], scatter.inputs["Density"])
            scene = bpy.context.scene
            scene["volume_guiding_fixture"] = variant
            scene["volume_guiding_density"] = density
            scene["volume_guiding_anisotropy"] = anisotropy
            # Make the saved demo open directly on its camera composition.
            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type == "VIEW_3D":
                        area.spaces.active.region_3d.view_perspective = "CAMERA"
        return original_box(name, location, scale, mat)

    def build_scene(settings):
        settings.volume_profile = variant
        settings.volume_density = density
        settings.volume_anisotropy = anisotropy
        settings.volume_heterogeneous = heterogeneous
        return original_build(settings)

    generator.box = box
    generator.build_scene = build_scene
    generator.main()


if __name__ == "__main__":
    main()
