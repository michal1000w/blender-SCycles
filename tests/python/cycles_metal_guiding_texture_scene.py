#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Guiding room with tiled, mipmapped image textures and view-dependent glass.

Accepts the shared room arguments. The generated .tx asset is saved beside the
render, so saved scenes remain reproducible. This exercises real texture-cache
requests, including reciprocal shading, rather than procedural texture nodes.
"""

import importlib.util
import pathlib
import sys

import bpy
import numpy as np
import OpenImageIO as oiio


def main():
    path = pathlib.Path(__file__).with_name('cycles_metal_guiding_scene.py')
    spec = importlib.util.spec_from_file_location('cycles_guiding_texture_room', path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    arguments = sys.argv[sys.argv.index('--') + 1:]
    persistent = '--persistent-textures' in arguments
    if persistent:
        arguments.remove('--persistent-textures')
    sys.argv = [str(path), '--', *arguments]
    output = pathlib.Path(arguments[arguments.index('--output') + 1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    texture_path = output.with_suffix('.tx')
    size = 512
    y, x = np.mgrid[:size, :size]
    checker = ((x // 16 + y // 16) % 2).astype(np.float32)
    pixels = np.stack((.15 + .65 * checker,
                       .2 + .5 * x / (size - 1),
                       .2 + .5 * y / (size - 1)), axis=-1).astype(np.float32)
    source = oiio.ImageBuf(oiio.ImageSpec(size, size, 3, oiio.FLOAT))
    if not source.set_pixels(oiio.ROI(0, size, 0, size, 0, 1, 0, 3), pixels):
        raise RuntimeError(source.geterror())
    config = oiio.ImageSpec()
    config.tile_width = config.tile_height = 32
    config.attribute('maketx:oiio_options', 1)
    if not oiio.ImageBufAlgo.make_texture(oiio.MakeTxTexture, source, str(texture_path), config):
        raise RuntimeError(oiio.geterror())

    original_material, original_box = generator.material, generator.box
    image = None

    def texture(material):
        nonlocal image
        if image is None:
            image = bpy.data.images.load(str(texture_path))
            image.colorspace_settings.name = 'Non-Color'
        node = material.node_tree.nodes.new('ShaderNodeTexImage')
        node.image = image
        return node

    def material(name, color, roughness=.6, metallic=0):
        bpy.context.scene.render.use_persistent_data = persistent
        result = original_material(name, color, roughness, metallic)
        if name == 'Diffuse ivory':
            node = texture(result)
            shader = result.node_tree.nodes.get('Principled BSDF')
            result.node_tree.links.new(node.outputs['Color'], shader.inputs['Base Color'])
        return result

    def box(name, location, scale, mat):
        if name == 'Tall block':
            mat = bpy.data.materials.new('Textured reciprocal glass')
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            material_output = nodes.new('ShaderNodeOutputMaterial')
            glass = nodes.new('ShaderNodeBsdfGlass')
            glass.inputs['IOR'].default_value = 1.45
            tex = texture(mat)
            facing = nodes.new('ShaderNodeLayerWeight')
            roughness = nodes.new('ShaderNodeMapRange')
            roughness.inputs['To Min'].default_value = .12
            roughness.inputs['To Max'].default_value = .4
            links = mat.node_tree.links
            links.new(tex.outputs['Color'], glass.inputs['Color'])
            links.new(facing.outputs['Facing'], roughness.inputs['Value'])
            links.new(roughness.outputs['Result'], glass.inputs['Roughness'])
            links.new(glass.outputs[0], material_output.inputs['Surface'])
            sphere = bpy.data.objects['Guiding test sphere']
            sphere.data.materials.clear()
            sphere.data.materials.append(mat)
            if persistent and '--warmup' in arguments:
                def capture_first(scene):
                    bpy.app.handlers.render_post.remove(capture_first)
                    cold = output.with_name(output.name + '_cold').with_suffix('.exr')
                    bpy.data.images['Render Result'].save_render(str(cold), scene=scene)
                    image_input = oiio.ImageInput.open(str(cold))
                    if image_input is None:
                        raise RuntimeError('Cold texture render was not saved')
                    rgb = np.asarray(image_input.read_image(format=oiio.FLOAT))[..., :3]
                    image_input.close()
                    np.save(cold.with_suffix('.npy'), rgb)
                bpy.app.handlers.render_post.append(capture_first)
        return original_box(name, location, scale, mat)

    generator.material, generator.box = material, box
    generator.main()
    if persistent and '--warmup' in arguments:
        cold = output.with_name(output.name + '_cold').with_suffix('.npy')
        if not cold.exists():
            raise RuntimeError('The persistent texture comparison did not capture its first render')


if __name__ == '__main__':
    main()
