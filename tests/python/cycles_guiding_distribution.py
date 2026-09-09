#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Build and run the shared guiding-distribution tests independently of a full Blender build.

Run with system Python. --metal additionally compiles and executes the same sampling header
on a real Apple GPU; lack of a Metal device is an error, never a silently passing skip.
"""

import argparse
import pathlib
import subprocess
import tempfile


def run(command):
    subprocess.run([str(item) for item in command], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metal", action="store_true")
    parser.add_argument("--transport", action="store_true",
                        help="Also check reciprocal transport using the actual Cycles microfacet BSDF")
    parser.add_argument("--tbb-include", type=pathlib.Path,
                        help="TBB header directory for --transport; defaults to bundled dependencies")
    parser.add_argument("--build-dir", type=pathlib.Path)
    parser.add_argument("--kernel-include", type=pathlib.Path,
                        help="Optional header overlay used consistently by host and Metal tests")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    temporary = None if args.build_dir else tempfile.TemporaryDirectory(prefix="cycles-guiding-tests-")
    build = args.build_dir.resolve() if args.build_dir else pathlib.Path(temporary.name)
    build.mkdir(parents=True, exist_ok=True)
    source = root / "intern/cycles/test"
    overlay = ["-I", args.kernel_include.resolve()] if args.kernel_include else []
    if args.kernel_include and not (args.kernel_include / "kernel").is_dir():
        parser.error("--kernel-include must contain a kernel directory")
    shared = ["clang++", "-std=c++20", "-O2", "-DCCL_NAMESPACE_BEGIN=namespace ccl {",
              "-DCCL_NAMESPACE_END=}", *overlay, "-I", root / "intern/cycles",
              "-I", root / "intern/atomic"]
    run([*shared, "-I", root / "extern/gtest/include", "-I", root / "extern/gtest",
         source / "kernel_guiding_distribution_test.cpp", source / "kernel_guiding_resampling_test.cpp",
         source / "kernel_guiding_parallax_test.cpp", source / "kernel_guiding_mixture_fit_test.cpp",
         root / "extern/gtest/src/gtest-all.cc",
         root / "extern/gtest/src/gtest_main.cc", "-o", build / "distribution_test"])
    run([build / "distribution_test"])
    if args.transport:
        include = args.tbb_include
        if include is None:
            include = next((path for path in sorted((root / "lib").glob("*/tbb/include"))
                            if (path / "tbb/spin_mutex.h").exists()), None)
        if include is None:
            parser.error("--transport needs bundled TBB headers or --tbb-include")
        run([*shared, "-I", include, "-I", root / "extern/gtest/include", "-I", root / "extern/gtest",
             source / "kernel_bidirectional_pdf_test.cpp", root / "extern/gtest/src/gtest-all.cc",
             root / "extern/gtest/src/gtest_main.cc", "-o", build / "transport_test"])
        run([build / "transport_test"])
    if args.metal:
        run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.0",
             f"-fmodules-cache-path={build / 'module-cache'}", *overlay, "-I", root / "intern/cycles",
             "-c", source / "guiding_distribution.metal", "-o", build / "distribution.air"])
        run(["xcrun", "-sdk", "macosx", "metallib", build / "distribution.air",
             "-o", build / "distribution.metallib"])
        run([*shared, "-fobjc-arc", "-framework", "Metal", "-framework", "Foundation",
             source / "guiding_distribution_metal.mm", "-o", build / "metal_test"])
        run([build / "metal_test", build / "distribution.metallib"])
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
