#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run Cycles diffraction math on the macOS Metal GPU (outside the sandbox).

Uses runtime Metal compilation, as Cycles does, so an optional Xcode command-line
Metal toolchain download is unnecessary. This tests the production kernel header;
it does not substitute for image and integration tests in Blender.
"""

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--sample-orders", action="store_true", help="Sample computed cache power columns")
    parser.add_argument("--ray-lookup", action="store_true", help="Derive full-cache lookup and boundary inputs from rays")
    parser.add_argument("--cache-directory", type=pathlib.Path, help="Test an exported full cache")
    parser.add_argument("--cache-queries", type=int, default=512)
    parser.add_argument("--host-only", action="store_true", help="Validate cache loading and CPU reference preparation without using Metal")
    parser.add_argument("--reference", action="store_true", help="Test and benchmark reference-port matching")
    parser.add_argument("--chart-cell", action="store_true", help="Use chart interpolation in --cell tests")
    parser.add_argument("--inplace-chart", action="store_true", help="Test destructive chart matching")
    parser.add_argument("--quadratic-cell", action="store_true", help="Use quadratic chart controls")
    parser.add_argument("--cell", action="store_true", help="Test packed hybrid cell evaluation")
    parser.add_argument("--tensor-header", type=pathlib.Path, help="Isolated tensor header candidate for scene tests")
    parser.add_argument("--evaluate-header", type=pathlib.Path, help="Isolated evaluator header candidate for tensor scene tests")
    parser.add_argument("--reference-header", type=pathlib.Path, help="Isolated matrix solver header candidate for tensor scene tests")
    parser.add_argument("--tensor-normal-incidence", action="store_true")
    parser.add_argument("--tensor-sample-only", action="store_true", help="Profile one sampler call without additional probability API checks")
    parser.add_argument("--tensor-scene-cache", type=pathlib.Path, help="Run scene sampler on a native tensor cache export")
    parser.add_argument("--scene-sample", action="store_true", help="Execute the combined scene sampler on Metal")
    parser.add_argument("--cache-view", action="store_true", help="Test multi-cache offsets and mirror lookup")
    parser.add_argument("--beckmann", action="store_true", help="Use Beckmann local math in --rough-dielectric")
    parser.add_argument("--coated", action="store_true", help="Test the continuous coated proposal in --rough-dielectric")
    parser.add_argument("--surface-mixture", action="store_true", help="Include renderer surface-mixture checks in --rough-dielectric")
    parser.add_argument("--rough-dielectric", action="store_true", help="Run combined rough dielectric math on Metal")
    parser.add_argument("--thin-film", action="store_true", help="Test spectral dielectric film on Metal")
    parser.add_argument("--transmission-measure", action="store_true", help="Integrate transmitted facet geometry on Metal")
    parser.add_argument("--direction", action="store_true", help="Test reflected and transmitted order directions")
    parser.add_argument("--fast-interface", action="store_true", help="Test the experimental scalar reflective/transmissive sampler")
    parser.add_argument("--coordinates", action="store_true", help="Test ray-to-cache coordinate mapping")
    parser.add_argument("--boundary", action="store_true", help="Test GPU exterior boundary construction")
    parser.add_argument("--construct-boundaries", action="store_true", help="Construct exterior coefficients inside the matching kernel")
    parser.add_argument("--evaluations", type=int, default=262144)
    args = parser.parse_args()
    if sum((args.rough_dielectric, args.thin_film, args.transmission_measure, args.fast_interface, args.tensor_scene_cache is not None, args.scene_sample, args.reference, args.boundary, args.cell, args.coordinates, args.direction, args.cache_view, args.cache_directory is not None)) > 1:
        parser.error("Choose one of --reference, --boundary, --cell or --cache-directory")
    if (args.tensor_header or args.evaluate_header or args.reference_header or args.tensor_normal_incidence or args.tensor_sample_only) and not args.tensor_scene_cache:
        parser.error("Tensor overrides require --tensor-scene-cache")
    if args.sample_orders:
        args.ray_lookup = True
    if args.ray_lookup:
        if not args.cache_directory:
            parser.error("--ray-lookup requires --cache-directory")
        args.construct_boundaries = True
    if args.host_only and not args.cache_directory:
        parser.error("--host-only requires --cache-directory")
    if not 1 <= args.cache_queries <= 65536:
        parser.error("--cache-queries must be between 1 and 65536")
    if args.inplace_chart and not (args.chart_cell or args.cache_directory):
        parser.error("--inplace-chart requires --chart-cell or --cache-directory")
    if args.quadratic_cell and not args.chart_cell:
        parser.error("--quadratic-cell requires --chart-cell")
    if args.chart_cell and not args.cell:
        parser.error("--chart-cell requires --cell")
    if args.construct_boundaries and not (args.reference or args.cache_directory):
        parser.error("--construct-boundaries requires --reference or --cache-directory")
    if args.beckmann and not args.rough_dielectric:
        parser.error("--beckmann requires --rough-dielectric")
    if args.coated and not args.rough_dielectric:
        parser.error("--coated requires --rough-dielectric")
    if args.surface_mixture and not args.rough_dielectric:
        parser.error("--surface-mixture requires --rough-dielectric")
    root = pathlib.Path(__file__).resolve().parents[2]
    cycles = root / "intern/cycles"
    dependencies = {}
    pragma_onced = set()
    processed = {}

    def expand(path, ancestors=()):
        path = path.resolve()
        if args.tensor_header and path == (cycles / "kernel/util/diffraction_tensor.h").resolve():
            path = args.tensor_header.resolve()
        if args.evaluate_header and path == (cycles / "kernel/util/diffraction_evaluate.h").resolve():
            path = args.evaluate_header.resolve()
        if args.reference_header and path == (cycles / "kernel/util/diffraction_reference.h").resolve():
            path = args.reference_header.resolve()
        source = path.read_text()
        dependencies[str(path)] = hashlib.sha256(source.encode()).hexdigest()
        # Match Cycles path_source_replace_includes: only pragma-once headers
        # are single-inclusion. Macro templates such as data_template.h must
        # expand at every inclusion with their current macro definitions.
        if "#pragma once" in source:
            if path in pragma_onced:
                return ""
            pragma_onced.add(path)
        if path in processed:
            return processed[path]
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

        source = re.sub(r'^[ \t]*#[ \t]*include[ \t]+"([^"]+)"[^\n]*$',
                        include, source, flags=re.MULTILINE)
        source = re.sub(r'^\s*#\s*pragma\s+once\s*$', '', source, flags=re.MULTILINE)
        processed[path] = source
        return source

    stem = "cycles_diffraction_reference" if args.reference else "cycles_diffraction"
    if args.boundary:
        stem = "cycles_diffraction_boundary"
    if args.scene_sample:
        stem = "cycles_diffraction_scene_sample"
    if args.cache_view:
        stem = "cycles_diffraction_cache_view"
    if args.direction:
        stem = "cycles_diffraction_direction"
    if args.transmission_measure:
        stem = "cycles_diffraction_transmission_measure"
    if args.thin_film:
        stem = "cycles_diffraction_thin_film"
    if args.rough_dielectric:
        stem = "cycles_diffraction_rough_dielectric"
    if args.fast_interface:
        stem = "cycles_diffraction_fast_interface"
    if args.coordinates:
        stem = "cycles_diffraction_coordinates"
    if args.cell:
        stem = "cycles_diffraction_cell"
    if args.cache_directory:
        stem = "cycles_diffraction_cache"
        for name in ("cache.json", "cache.bin"):
            path = args.cache_directory.resolve() / name
            dependencies[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.tensor_scene_cache:
        stem = "cycles_diffraction_tensor_scene"
        path = args.tensor_scene_cache.resolve()
        dependencies[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    expanded = expand(root / "tests/metal" / (stem + ".metal"))
    if args.beckmann:
        expanded = "#define DIFFRACTION_TEST_BECKMANN 1\n" + expanded
    if args.coated:
        expanded = "#define DIFFRACTION_TEST_COATED 1\n" + expanded
    if args.surface_mixture:
        expanded = "#define DIFFRACTION_TEST_SURFACE_MIXTURE 1\n" + expanded
    if args.tensor_sample_only:
        expanded = "#define DIFFRACTION_SCENE_SAMPLE_ONLY 1\n" + expanded
    if args.construct_boundaries:
        expanded = "#define DIFFRACTION_CONSTRUCT_BOUNDARIES 1\n" + expanded
    if args.sample_orders:
        expanded = "#define DIFFRACTION_SAMPLE_ORDERS 1\n" + expanded
    if args.inplace_chart:
        expanded = "#define DIFFRACTION_INPLACE_CHART 1\n" + expanded
    if args.ray_lookup:
        expanded = "#define DIFFRACTION_RAY_LOOKUP 1\n" + expanded
    with tempfile.TemporaryDirectory(prefix="cycles-diffraction-metal-") as temporary:
        directory = pathlib.Path(temporary)
        source = directory / "diffraction.metal"
        executable = directory / "diffraction"
        source.write_text(expanded)
        host = root / "tests/metal" / (stem + ".mm")
        dependencies[str(host.relative_to(root))] = hashlib.sha256(host.read_bytes()).hexdigest()
        if args.cache_directory:
            helper = root / "tests/performance/diffraction_packed_audit.h"
            dependencies[str(helper.relative_to(root))] = hashlib.sha256(helper.read_bytes()).hexdigest()
        solver = cycles / "scene/diffraction.cpp"
        for path in (solver, cycles / "scene/diffraction.h",
                     cycles / "test/diffraction_reference_fixture.h",
                     cycles / "test/diffraction_boundary_fixture.h"):
            dependencies[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        subprocess.run(["clang++", "-std=c++20", "-O2", "-fobjc-arc", "-I" + str(cycles),
                        *(["-DDIFFRACTION_TEST_BECKMANN=1"] if args.beckmann else []),
                        *(["-DDIFFRACTION_TEST_COATED=1"] if args.coated else []),
                        *(["-DDIFFRACTION_TEST_SURFACE_MIXTURE=1"] if args.surface_mixture else []),
                        *(["-DDIFFRACTION_SAMPLE_ORDERS=1"] if args.sample_orders else []),
                        *(["-DDIFFRACTION_RAY_LOOKUP=1"] if args.ray_lookup else []),
                        *(["-DDIFFRACTION_CHART_CELL=1"] if args.chart_cell else []),
                        *(["-DDIFFRACTION_QUADRATIC_CELL=1"] if args.quadratic_cell else []),
                        *(["-DDIFFRACTION_INPLACE_CHART=1"] if args.inplace_chart else []),
                        *(["-DDIFFRACTION_CONSTRUCT_BOUNDARIES=1"] if args.construct_boundaries else []),
                        "-I" + str(root / "lib/macos_arm64/tbb/include"),
                        "-DCCL_NAMESPACE_BEGIN=namespace ccl {", "-DCCL_NAMESPACE_END=}",
                        "-I" + str(root / "lib/macos_arm64/eigen/include/eigen3"),
                        "-framework", "Foundation", "-framework", "Metal", str(host), str(solver),
                        "-o", str(executable)], check=True)
        options = [str(args.evaluations)] if args.reference or args.boundary or args.cell or args.coordinates or args.direction else []
        if args.cache_directory:
            options = [str(args.cache_directory.resolve()), str(args.cache_queries)]
            if args.host_only:
                options.append("--host-only")
        if args.scene_sample:
            optical = root / "tests/scenes/diffraction/optical_constants"
            manifest = json.loads((optical / "provenance.json").read_text())
            data = optical / "Al_Rakic_1995_nm.csv"
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            if digest != manifest['derived'][data.name]['sha256']:
                raise ValueError("Aluminum fixture hash does not match provenance")
            dependencies[str(data.relative_to(root))] = digest
            dependencies[str((optical / "provenance.json").relative_to(root))] = hashlib.sha256((optical / "provenance.json").read_bytes()).hexdigest()
            options.append(str(data))
        if args.tensor_scene_cache:
            options = [str(args.tensor_scene_cache.resolve())]
            if args.tensor_normal_incidence:
                options.append("--normal")
        result = subprocess.run([str(executable), str(source), *options], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stderr, end="")
        print(result.stdout, end="")
        report = dict(returncode=result.returncode, surface_mixture=args.surface_mixture,
                      surface_mixture_setups=256 if args.surface_mixture else 0,
                      surface_mixture_trials_per_setup=4 if args.surface_mixture else 0,
                      tensor_normal_incidence=args.tensor_normal_incidence,
                      tensor_sample_only=args.tensor_sample_only, source_sha256=dependencies,
                      stdout=result.stdout, stderr=result.stderr)
        if result.returncode == 0:
            report.update(json.loads(result.stdout))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
