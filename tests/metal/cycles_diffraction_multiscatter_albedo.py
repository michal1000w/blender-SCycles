#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare and optionally run the bounded native GGX grating albedo Metal probe."""

import argparse
import pathlib
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true", help="Expand source only; no compilation or GPU work")
    action.add_argument("--compile-only", action="store_true", help="Compile Metal pipeline; no GPU dispatch")
    action.add_argument("--run", action="store_true", help="Dispatch six profiles and sixteen wavelength slices together, each 128 nodes × 512 samples")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    cycles = root / "intern/cycles"
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    once = set()

    def expand(path, ancestors=()):
        path = path.resolve()
        data = path.read_text()
        if "#pragma once" in data:
            if path in once:
                return ""
            once.add(path)
        if path in ancestors:
            raise RuntimeError(f"Include cycle: {path}")

        def include(match):
            name = match.group(1)
            target = path.parent / name
            if not target.is_file():
                target = cycles / name
            if not target.is_file():
                return match.group(0)
            return expand(target, (*ancestors, path))

        data = re.sub(r'^[ \t]*#[ \t]*include[ \t]+"([^"]+)"[^\n]*$',
                      include, data, flags=re.MULTILINE)
        return re.sub(r'^\s*#\s*pragma\s+once\s*$', '', data, flags=re.MULTILINE)

    source = output / "cycles_diffraction_multiscatter_albedo_expanded.metal"
    source.write_text(expand(pathlib.Path(__file__).with_suffix(".metal")))
    if args.prepare_only:
        print(source)
        return
    binary = output / "cycles_diffraction_multiscatter_albedo"
    subprocess.run([
        "clang++", "-std=c++20", "-O2", "-fobjc-arc", "-I" + str(cycles),
        "-I" + str(root / "lib/macos_arm64/tbb/include"),
        "-DCCL_NAMESPACE_BEGIN=namespace ccl {", "-DCCL_NAMESPACE_END=}",
        "-I" + str(root / "lib/macos_arm64/eigen/include/eigen3"),
        "-framework", "Foundation", "-framework", "Metal",
        str(pathlib.Path(__file__).with_suffix(".mm")),
        str(cycles / "scene/diffraction.cpp"), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary), str(source), *(["--compile-only"] if args.compile_only else [])],
                   check=True)


if __name__ == "__main__":
    main()
