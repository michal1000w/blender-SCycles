# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Render times of the volume index of refraction demo scenes in every integrator configuration.

  blender -b --factory-startup --python tools/volume_ior/bench_matrix.py -- \
      --output DIRECTORY [--scenes a,b] [--devices METAL,CPU] [--configs pt,pt_guiding,bdpt,bdpt_guiding] \
      [--samples N] [--resolution WIDTH] [--rounds N] [--modes on,off]

Every scene of `tests/volume_ior/` is rendered with the refraction and with it switched off in
the render settings (`on`, `off`). A render with one sample comes first, to compile and load the
kernels, and is not timed. Appends lines to `DIRECTORY/times.jsonl` and saves the images as
`DIRECTORY/<device>_<config>_<mode>/<scene>.exr`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene-dir", type=Path, default=ROOT / "tests" / "volume_ior")
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--devices", default="METAL,CPU")
    parser.add_argument("--configs", default="pt,pt_guiding,bdpt,bdpt_guiding")
    parser.add_argument("--modes", default="on,off")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=960)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="")
    return parser.parse_args(argv)


def enable_device(device):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device == "METAL":
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for entry in prefs.devices:
            entry.use = entry.type == "METAL"
    bpy.context.scene.cycles.device = "GPU" if device == "METAL" else "CPU"


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = sorted(path.stem for path in args.scene_dir.glob("*.blend"))
    if args.scenes != "all":
        names = [name for name in args.scenes.split(",") if name in names]

    with open(args.output / "times.jsonl", "a") as log:
        for name in names:
            for device in args.devices.split(","):
                for config in args.configs.split(","):
                    for mode in args.modes.split(","):
                        bpy.ops.wm.open_mainfile(filepath=str(args.scene_dir / (name + ".blend")))
                        scene = bpy.context.scene
                        enable_device(device)
                        cycles = scene.cycles
                        scene.render.resolution_x = args.resolution
                        scene.render.resolution_y = int(round(args.resolution * 9 / 16))
                        scene.render.resolution_percentage = 100
                        scene.render.image_settings.file_format = "OPEN_EXR"
                        scene.render.image_settings.color_depth = "32"
                        cycles.use_adaptive_sampling = False
                        cycles.use_denoising = False
                        cycles.seed = args.seed
                        cycles.use_bidirectional_path_tracing = config.startswith("bdpt")
                        cycles.use_guiding = config.endswith("_guiding")
                        cycles.use_volume_ior = mode != "off"

                        cycles.samples = 1
                        bpy.ops.render.render(write_still=False)

                        cycles.samples = args.samples
                        directory = args.output / ("%s_%s_%s" % (device, config, mode))
                        scene.render.filepath = str(directory / name)
                        seconds = []
                        for _ in range(args.rounds):
                            start = time.time()
                            bpy.ops.render.render(write_still=True)
                            seconds.append(time.time() - start)
                        seconds.sort()
                        median = seconds[len(seconds) // 2]
                        entry = {"scene": name, "device": device, "config": config, "mode": mode,
                                 "seconds": median, "samples": args.samples,
                                 "resolution": args.resolution, "label": args.label}
                        log.write(json.dumps(entry) + "\n")
                        log.flush()
                        print("VOLUME_IOR_BENCH %-14s %-6s %-13s %-3s %7.2f s" % (
                            name, device, config, mode, median), flush=True)


if __name__ == "__main__":
    main()
