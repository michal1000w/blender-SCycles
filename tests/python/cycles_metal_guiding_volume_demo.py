#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Save editable Metal PT and BDPT fog demos; optionally reopen and render both.

Run inside repository Blender. Arguments follow --. Saved scenes use 512x512,
1024 samples, and AgX. Verification renders use 256x256 and 256 samples with
denoising and adaptive termination disabled, without changing the saved files.
"""

import argparse
import hashlib
import json
import pathlib
import runpy
import sys

import bpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = pathlib.Path(__file__).with_name("cycles_metal_guiding_volume_scene.py")
    sys.argv = [str(fixture), "--", "--scene", "volume_dense", "--device", "GPU",
                "--guiding", "--build-only", "--samples", "1024", "--resolution", "512",
                "--output", str(args.output / "scene")]
    runpy.run_path(str(fixture), run_name="__main__")
    scene = bpy.context.scene
    scene.view_settings.view_transform = "AgX"
    scene.cycles.guiding_training_samples = 128
    from cycles import ui
    assert ui.CYCLES_RENDER_PT_sampling_path_guiding.poll(bpy.context)
    files = {}
    for mode in ("PT", "BDPT"):
        scene.cycles.use_bidirectional_path_tracing = mode == "BDPT"
        path = args.output / ("metal_volume_guiding_" + mode + ".blend")
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        files[mode] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    if args.verify:
        import numpy as np
        import OpenImageIO as oiio
        for mode, metadata in files.items():
            bpy.ops.wm.open_mainfile(filepath=metadata["path"])
            scene = bpy.context.scene
            cycles = scene.cycles
            assert cycles.device == "GPU" and cycles.use_guiding
            assert cycles.use_volume_guiding and cycles.use_surface_guiding
            assert cycles.use_bidirectional_path_tracing == (mode == "BDPT")
            assert cycles.samples == 1024 and scene.render.resolution_x == 512
            assert not cycles.use_denoising and not cycles.use_adaptive_sampling
            cycles.samples = 256
            scene.render.resolution_x = scene.render.resolution_y = 256
            scene.render.filepath = str(args.output / (mode + "_preview.exr"))
            bpy.ops.render.render(write_still=True)
            image = oiio.ImageInput.open(scene.render.filepath)
            assert image is not None
            pixels = np.asarray(image.read_image(format=oiio.FLOAT))[..., :3]
            image.close()
            assert pixels.shape == (256, 256, 3)
            assert np.isfinite(pixels).all() and float(pixels.max()) > 0
            np.save(args.output / (mode + "_preview.npy"), pixels)
            metadata["verification"] = dict(samples=256, resolution=256,
                                             mean=float(pixels.mean()), peak=float(pixels.max()))
            saved_digest = hashlib.sha256(pathlib.Path(metadata["path"]).read_bytes()).hexdigest()
            assert saved_digest == metadata["sha256"]

    (args.output / "report.json").write_text(json.dumps(files, indent=2) + "\n")
    print("VOLUME_DEMOS_VERIFIED" if args.verify else "VOLUME_DEMOS_SAVED", flush=True)


if __name__ == "__main__":
    main()
