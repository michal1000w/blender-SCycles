#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare cold/warm tiled volume renders without guiding or denoising.

An emissive medium with textured extinction exercises partial integration before
a tile request. The identical seeded second render must not lose extra emission
that was incorrectly accumulated on cold-cache retries. Run inside Blender.
"""

import argparse
import hashlib
import json
import pathlib
import runpy
import sys

import bpy
import numpy as np
import OpenImageIO as oiio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--ray-marching", action="store_true")
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path(__file__).with_name("cycles_metal_guiding_scene.py")
    sys.argv = [str(base), "--", "--scene", "volume", "--device", "GPU", "--build-only",
                "--samples", str(args.samples), "--resolution", "64", "--seed", "103",
                "--output", str(args.output / "scene")]
    runpy.run_path(str(base), run_name="__main__")
    scene = bpy.context.scene
    scene.render.use_persistent_data = True
    scene.render.use_texture_cache = True
    scene.render.use_auto_generate_texture_cache = False
    scene.cycles.volume_biased = args.ray_marching
    scene.view_layers[0].cycles.denoising_store_passes = True

    y, x = np.mgrid[:1024, :1024]
    density = (0.1 + 0.8 * ((x // 29 + y // 37) % 2)).astype(np.float32)
    pixels = np.repeat(density[..., None], 3, axis=2)
    source = oiio.ImageBuf(oiio.ImageSpec(1024, 1024, 3, oiio.FLOAT))
    assert source.set_pixels(oiio.ROI(0, 1024, 0, 1024, 0, 1, 0, 3), pixels)
    config = oiio.ImageSpec()
    config.tile_width = config.tile_height = 32
    texture = args.output / "volume-density.tx"
    assert oiio.ImageBufAlgo.make_texture(oiio.MakeTxTexture, source, str(texture), config)
    image = bpy.data.images.load(str(texture))
    image.colorspace_settings.name = "Non-Color"
    material = scene.objects["Medium boundary"].data.materials[0]
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    medium = nodes.new("ShaderNodeVolumePrincipled")
    medium.inputs["Anisotropy"].default_value = 0.65
    medium.inputs["Emission Strength"].default_value = 0.5
    medium.inputs["Emission Color"].default_value = (0.8, 0.9, 1.0, 1.0)
    coordinates = nodes.new("ShaderNodeTexCoord")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    links.new(coordinates.outputs["Generated"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], medium.inputs["Density"])
    links.new(medium.outputs["Volume"], output.inputs["Volume"])
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "scene.blend"))

    renders = []
    for label in ("cold", "warm"):
        path = args.output / (label + ".exr")
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        reader = oiio.ImageInput.open(str(path))
        assert reader is not None
        rgb = np.asarray(reader.read_image(format=oiio.FLOAT))[..., :3]
        reader.close()
        assert np.isfinite(rgb).all() and rgb.max() > 0
        np.save(path.with_suffix(".npy"), rgb)
        renders.append(rgb)
    cold, warm = renders
    delta = cold.astype(float) - warm.astype(float)
    passed = bool(np.allclose(cold, warm, rtol=2e-5, atol=2e-5))
    report = dict(passed=passed, max_absolute_difference=float(np.abs(delta).max()),
                  mse=float((delta ** 2).mean()), cold_mean=float(cold.mean()),
                  warm_mean=float(warm.mean()), samples=args.samples,
                  ray_marching=args.ray_marching, guiding=False, denoising=False,
                  denoising_passes=True,
                  texture_sha256=hashlib.sha256(texture.read_bytes()).hexdigest())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("VOLUME_CACHE_REGRESSION " + json.dumps(report), flush=True)
    assert passed, "Cold-cache volume integration differs from identical warm-cache render"


if __name__ == "__main__":
    main()
