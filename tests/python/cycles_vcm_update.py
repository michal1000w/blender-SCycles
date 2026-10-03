# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Change the vertex merging settings between renders of one persistent session.

  blender -b --factory-startup --python tests/python/cycles_vcm_update.py -- \
      --output DIRECTORY [--device CPU|METAL] [--integrator pt|bdpt]

Renders the same scene without merging, with merging, with another radius, with the other
integrator and without merging again. Switching the setting has to rebuild what depends on it:
both renders without merging must match, and so must the first render with merging and a last
one with the first settings. Prints VCM_UPDATE lines and exits non-zero on failure.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import bpy
import numpy as np

spec = importlib.util.spec_from_file_location(
    "cycles_vcm_stress", Path(__file__).with_name("cycles_vcm_stress.py"))
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)
scenes = stress.scenes


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="CPU", choices=("METAL", "CPU"))
    parser.add_argument("--integrator", default="pt", choices=("pt", "bdpt"))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=96)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    scene = scenes.scene_caustic_glass(stress.scene_args(args, False, args.samples))
    scene.render.use_persistent_data = True
    cycles = scene.cycles
    cycles.use_bidirectional_path_tracing = args.integrator == "bdpt"
    cycles.bdpt_max_bounces = 8

    def render(name):
        path = stress.render(args, scene, name)
        image = stress.combined(stress.load(path))
        valid = bool(np.isfinite(image).all() and (image >= 0.0).all())
        print("VCM_UPDATE %-6s %-5s %-14s mean %.5f %s" % (
            args.device, args.integrator, name, image.mean(), "" if valid else "INVALID"),
            flush=True)
        return image, valid

    images = {}
    valid = True
    steps = (
        ("off_first", dict(use_vertex_merging=False)),
        ("on_first", dict(use_vertex_merging=True, vcm_radius=0.0)),
        ("on_radius", dict(vcm_radius=0.03)),
        ("on_other", dict(use_bidirectional_path_tracing=args.integrator != "bdpt")),
        ("on_last", dict(use_bidirectional_path_tracing=args.integrator == "bdpt",
                         vcm_radius=0.0)),
        ("off_last", dict(use_vertex_merging=False)),
    )
    for name, settings in steps:
        for key, value in settings.items():
            setattr(cycles, key, value)
        images[name], ok = render(name)
        valid = valid and ok

    def difference(a, b):
        return float(np.abs(images[a] - images[b]).mean() / max(images[a].mean(), 1e-12))

    # The CPU repeats a render exactly; Metal accumulates in a different order every time.
    tolerance = 1.0e-5 if args.device == "CPU" else 0.05
    off = difference("off_first", "off_last")
    on = difference("on_first", "on_last")
    changed = difference("on_first", "on_radius")
    ok = valid and off <= tolerance and on <= tolerance and changed > 1.0e-4
    print("VCM_UPDATE %-6s %-5s off/off %.2g on/on %.2g radius change %.2g %s" % (
        args.device, args.integrator, off, on, changed, "ok" if ok else "FAILED"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
