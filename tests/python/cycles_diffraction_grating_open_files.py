# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Open .blend files saved by another build (for example the Al-Toto/blender branch, whose
Diffraction BSDF has no Model setting and no node storage), check that every Diffraction BSDF
loads as the Grating model with its settings, and render each file to EXR.

  blender -b --factory-startup --python-exit-code 1 \
      --python tests/python/cycles_diffraction_grating_open_files.py -- \
      --input DIRECTORY --output DIRECTORY [--device CPU|METAL] [--expect-json FILE]

--expect-json is a file written by --dump in the saving build: {blend: [[profile, grating type,
{socket: value}], ...]}. Without it the check only requires the Grating model.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="CPU", choices=("CPU", "METAL", "CUDA", "OPTIX"))
    parser.add_argument("--dump", type=Path, help="Write the node settings of the files and exit")
    parser.add_argument("--expect-json", type=Path)
    return parser.parse_args(argv)


def diffraction_nodes():
    for material in bpy.data.materials:
        if material.node_tree:
            for node in material.node_tree.nodes:
                if node.bl_idname == "ShaderNodeBsdfDiffraction":
                    yield material.name, node


def node_settings(node):
    values = {}
    for socket in node.inputs:
        if socket.name in {"Color", "Roughness", "Pitch", "Groove Depth", "Duty Cycle",
                           "Rotation"}:
            value = socket.default_value
            values[socket.name] = list(value) if hasattr(value, "__len__") else value
    return [node.profile, node.grating_type, values]


def main():
    args = parse_args()
    files = sorted(args.input.glob("*.blend"))
    if args.dump:
        dump = {}
        for path in files:
            bpy.ops.wm.open_mainfile(filepath=str(path))
            dump[path.name] = {name: node_settings(node) for name, node in diffraction_nodes()}
        args.dump.write_text(json.dumps(dump, indent=1))
        return
    expected = json.loads(args.expect_json.read_text()) if args.expect_json else {}
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    for path in files:
        bpy.ops.wm.open_mainfile(filepath=str(path))
        found = 0
        for name, node in diffraction_nodes():
            found += 1
            if getattr(node, "quality", "GRATING") != "GRATING":
                errors.append(f"{path.name}/{name}: model {node.quality}")
            want = expected.get(path.name, {}).get(name)
            if want is not None:
                have = node_settings(node)
                if have[0] != want[0] or have[1] != want[1]:
                    errors.append(f"{path.name}/{name}: enums {have[:2]} != {want[:2]}")
                for socket, value in want[2].items():
                    got = have[2].get(socket)
                    flat_w = value if isinstance(value, list) else [value]
                    flat_g = got if isinstance(got, list) else [got]
                    if got is None or any(abs(a - b) > 1e-6 for a, b in zip(flat_g, flat_w)):
                        errors.append(f"{path.name}/{name}: {socket} {got} != {value}")
        if found == 0:
            errors.append(f"{path.name}: no Diffraction BSDF")
        scene = bpy.context.scene
        scene.cycles.device = "CPU" if args.device == "CPU" else "GPU"
        if args.device != "CPU":
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = args.device
            prefs.get_devices()
            for device in prefs.devices:
                device.use = device.type == args.device
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.filepath = str(args.output / (path.stem + ".exr"))
        bpy.ops.render.render(write_still=True)
        print("FILE", path.name, "nodes", found, flush=True)
    for error in errors:
        print("ERROR", error, flush=True)
    if errors:
        sys.exit(1)
    print("ALL FILES OK", flush=True)


if __name__ == "__main__":
    main()
