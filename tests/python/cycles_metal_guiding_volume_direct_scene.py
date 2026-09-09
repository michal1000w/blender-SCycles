#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic room with one real scattering event on every contender.

This isolates the emitter/NEE/sensor partition from longer volume paths. It is
not a replacement for the full-transport acceptance fixtures. Use --scene
volume for forward scattering or volume_backward for backward scattering.
"""

import importlib.util
import pathlib
import sys

import bpy


def main():
    base_path = pathlib.Path(__file__).with_name("cycles_metal_guiding_scene.py")
    spec = importlib.util.spec_from_file_location("cycles_guiding_direct_volume_room", base_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    arguments = sys.argv[sys.argv.index("--") + 1:]
    index = arguments.index("--scene") + 1
    variant = arguments[index]
    profiles = {"volume": (0.12, 0.65), "volume_backward": (0.25, -0.65)}
    density, anisotropy = profiles[variant]
    arguments[index] = "volume"
    sys.argv = [str(base_path), "--", *arguments]
    original_box, original_build = generator.box, generator.build_scene

    def box(name, location, scale, mat):
        if name == "Medium boundary":
            cycles = bpy.context.scene.cycles
            # Cycles adds one internally: UI zero still evaluates direct light
            # at the first real event and its emitter-hit continuation.
            cycles.max_bounces = 0
            cycles.bdpt_max_bounces = 1
            cycles.volume_bounces = 0
            scatter = next(node for node in mat.node_tree.nodes if node.type == "VOLUME_SCATTER")
            scatter.inputs["Density"].default_value = density
            scatter.inputs["Anisotropy"].default_value = anisotropy
        return original_box(name, location, scale, mat)

    def build_scene(settings):
        settings.volume_profile = variant
        settings.volume_density = density
        settings.volume_anisotropy = anisotropy
        settings.diagnostic_max_bounces = 0
        settings.diagnostic_bdpt_max_bounces = 1
        return original_build(settings)

    generator.box = box
    generator.build_scene = build_scene
    generator.main()


if __name__ == "__main__":
    main()
