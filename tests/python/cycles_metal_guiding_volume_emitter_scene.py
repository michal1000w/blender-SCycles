#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Volume emitter-MIS controls using mesh emission and an environment.

The null-sheet variant is physically identical to the mesh-emitter scene, but
also exercises transparent continuation after a phase sample. All assets are
procedural. Accepts the common room CLI with the scene names below.
"""

import importlib.util
import pathlib
import sys

import bpy


def main():
    path = pathlib.Path(__file__).with_name("cycles_metal_guiding_scene.py")
    spec = importlib.util.spec_from_file_location("cycles_volume_emitter_room", path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    arguments = sys.argv[sys.argv.index("--") + 1:]
    index = arguments.index("--scene") + 1
    variant = arguments[index]
    if variant not in {"volume_mesh", "volume_null_mesh", "volume_world"}:
        raise ValueError("Unknown volume emitter fixture: " + variant)
    arguments[index] = "volume"
    sys.argv = [str(path), "--", *arguments]
    original_box, original_build = generator.box, generator.build_scene

    def box(name, location, scale, mat):
        if name == "Medium boundary":
            scene = bpy.context.scene
            lamp = scene.objects["Hidden area light"]
            transform = lamp.matrix_world.copy()
            bpy.data.objects.remove(lamp, do_unlink=True)
            if variant == "volume_world":
                background = scene.world.node_tree.nodes.get("Background")
                background.inputs["Color"].default_value = (0.8, 0.9, 1.0, 1.0)
                background.inputs["Strength"].default_value = 2.0
            else:
                emission = bpy.data.materials.new("Mesh emitter")
                emission.use_nodes = True
                nodes = emission.node_tree.nodes
                nodes.clear()
                output = nodes.new("ShaderNodeOutputMaterial")
                shader = nodes.new("ShaderNodeEmission")
                shader.inputs["Strength"].default_value = 1000.0
                emission.node_tree.links.new(shader.outputs[0], output.inputs["Surface"])
                bpy.ops.mesh.primitive_circle_add(vertices=64, radius=0.375, fill_type="NGON")
                emitter = bpy.context.object
                emitter.name = "Hidden emissive disk"
                emitter.matrix_world = transform
                emitter.data.materials.append(emission)
                if variant == "volume_null_mesh":
                    transparent = bpy.data.materials.new("Unit-transmission null sheet")
                    transparent.use_nodes = True
                    nodes = transparent.node_tree.nodes
                    nodes.clear()
                    output = nodes.new("ShaderNodeOutputMaterial")
                    shader = nodes.new("ShaderNodeBsdfTransparent")
                    transparent.node_tree.links.new(shader.outputs[0], output.inputs["Surface"])
                    original_box("Null sheet", (0, 2.5, 2), (6, 0.001, 4), transparent)
            scene["volume_emitter_fixture"] = variant
        return original_box(name, location, scale, mat)

    def build_scene(settings):
        settings.volume_emitter_fixture = variant
        return original_build(settings)

    generator.box, generator.build_scene = box, build_scene
    generator.main()


if __name__ == "__main__":
    main()
