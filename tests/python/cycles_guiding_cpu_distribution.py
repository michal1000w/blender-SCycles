#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Isolate the CPU guiding representation on saved benchmark scenes.

This is a diagnostic, not a replacement for the default CPU-guided reference.
Every run uses the same sample count and preserves the saved scene's render settings.
Run inside Blender with arguments following --.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time

import bpy
import numpy as np
import OpenImageIO as oiio


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', nargs='+', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 211, 307])
    parser.add_argument('--distributions', nargs='+', choices=['VMM', 'PARALLAX_AWARE_VMM'],
                        default=['VMM'])
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for scene_path in args.scenes:
        scene_path = scene_path.resolve()
        bpy.ops.wm.open_mainfile(filepath=str(scene_path))
        # CPU representation controls are intentionally developer-only in Cycles sync.
        bpy.context.preferences.view.show_developer_ui = True
        bpy.context.preferences.experimental.use_cycles_debug = True
        scene = bpy.context.scene
        assert not scene.cycles.use_denoising and not scene.cycles.use_adaptive_sampling
        assert not scene.cycles.use_bidirectional_path_tracing
        scene.cycles.device = 'CPU'
        scene.cycles.use_guiding = True
        scene.cycles.samples = args.samples
        scene.cycles.time_limit = 0
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        for distribution in args.distributions:
            scene.cycles.guiding_distribution_type = distribution
            for seed in args.seeds:
                scene.cycles.seed = seed
                name = f'{scene_path.stem}_{distribution}_{seed}'
                prefix = args.output / name
                scene.render.filepath = str(prefix.with_suffix('.exr'))
                print('CPU_DISTRIBUTION_BEGIN ' + name, flush=True)
                start = time.perf_counter()
                bpy.ops.render.render(write_still=True)
                elapsed = time.perf_counter() - start
                image = oiio.ImageInput.open(scene.render.filepath)
                if image is None:
                    raise RuntimeError('Missing rendered EXR')
                pixels = np.asarray(image.read_image(format=oiio.FLOAT))[..., :3]
                image.close()
                assert np.isfinite(pixels).all() and pixels.max() > 0
                np.save(prefix.with_suffix('.npy'), pixels)
                record = dict(name=name, scene=str(scene_path), scene_sha256=digest(scene_path),
                              binary_sha256=digest(pathlib.Path(bpy.app.binary_path)),
                              distribution=distribution, developer_controls_enabled=True, seed=seed, samples=args.samples,
                              training_samples=scene.cycles.guiding_training_samples,
                              seconds=elapsed, mean=float(pixels.mean()),
                              timing_comparisons_valid=False)
                prefix.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
                records.append(record)
                (args.output / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
                print('CPU_DISTRIBUTION_END ' + name, flush=True)


if __name__ == '__main__':
    main()
