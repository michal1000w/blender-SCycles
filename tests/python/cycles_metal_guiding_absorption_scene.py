#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Absorption-only control for BDPT surface connections inside a medium.

Accepts the common guiding-room CLI with --scene volume. Removing scattering
isolates medium-stack/transmittance correctness from phase guiding and volume
connection MIS. This is a transport control, not a volume-guiding benchmark.
"""

import importlib.util
import pathlib


def main():
    path = pathlib.Path(__file__).with_name("cycles_metal_guiding_scene.py")
    spec = importlib.util.spec_from_file_location("cycles_absorption_room", path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    original_box, original_build = generator.box, generator.build_scene

    def box(name, location, scale, mat):
        if name == "Medium boundary":
            nodes = mat.node_tree.nodes
            nodes.clear()
            output = nodes.new("ShaderNodeOutputMaterial")
            absorption = nodes.new("ShaderNodeVolumeAbsorption")
            absorption.inputs["Density"].default_value = 0.3
            absorption.inputs["Color"].default_value = (0.7, 0.8, 1.0, 1.0)
            mat.node_tree.links.new(absorption.outputs[0], output.inputs["Volume"])
            mat.name = "Absorption-only medium transport control"
        return original_box(name, location, scale, mat)

    def build_scene(settings):
        if settings.scene != "volume":
            raise ValueError("The absorption control requires --scene volume")
        settings.transport_control = "pure_absorption"
        return original_build(settings)

    generator.box, generator.build_scene = box, build_scene
    generator.main()


if __name__ == "__main__":
    main()
