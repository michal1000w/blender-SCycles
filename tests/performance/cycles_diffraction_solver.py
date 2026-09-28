#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Measure host RCWA response-generation time, not render time.

The complete block includes substrate incidence for lossless materials, whereas
separate solves cover upper-medium incidence only. Each uses the same modal
count but recentering a separate solve changes its finite Fourier window.
Consequently these timings do not imply equal approximation error. Numerical
correctness is tested separately by scene_diffraction_test.cpp.
"""

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--compiler', default='clang++')
    parser.add_argument('--cache-resolution', type=int, help='Audit cache accuracy instead of solve timing')
    parser.add_argument('--queries', type=int, default=256)
    parser.add_argument('--grazing-extension', action='store_true')
    parser.add_argument('--wavelength-samples', type=int, default=9)
    parser.add_argument('--reference-cache', action='store_true')
    parser.add_argument('--hybrid-cache', action='store_true')
    parser.add_argument('--cell-chart', action='store_true')
    parser.add_argument('--probe-cell', type=pathlib.Path, help='JSON cell description from cache inspector')
    parser.add_argument('--quadratic-cell', action='store_true')
    parser.add_argument('--quadratic-cache', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--mirror-symmetry', action='store_true')
    parser.add_argument('--intensity-only', action='store_true')
    parser.add_argument('--full-cache', action='store_true')
    parser.add_argument('--cache-directory', type=pathlib.Path)
    parser.add_argument('--reference-cache-mib', type=int, default=64)
    parser.add_argument('--geometric-splits', action='store_true')
    parser.add_argument('--curvature-splits', action='store_true')
    parser.add_argument('--complex-tolerance', type=float, default=0.0)
    parser.add_argument('--max-nodes', type=int, default=256)
    parser.add_argument('--adaptive-cache', action='store_true')
    parser.add_argument('--modal-query', type=float, nargs=3, metavar=('BLOCH','KY','WAVELENGTH'), help='Evaluate one specified modal-convergence query')
    parser.add_argument('--modal-convergence', action='store_true')
    parser.add_argument('--tolerance', type=float, default=0.002)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--seed', type=int, default=617923)
    parser.add_argument('--half-orders', type=int, default=16)
    parser.add_argument('--reference-orders', type=int, default=32)
    parser.add_argument('--cutoff-margin', type=float, default=0.0)
    parser.add_argument('--eigen', type=pathlib.Path)
    args = parser.parse_args()
    if args.modal_query and not args.modal_convergence:
        parser.error("--modal-query requires --modal-convergence")
    root = pathlib.Path(__file__).resolve().parents[2]
    eigen = args.eigen or root / 'lib/macos_arm64/eigen/include/eigen3'
    audit = bool(args.cache_resolution or args.adaptive_cache or args.modal_convergence or args.full_cache or args.cell_chart or args.quadratic_cell or args.probe_cell)
    benchmark = ('cycles_diffraction_cell_probe.cpp' if args.probe_cell else
                 'cycles_diffraction_quadratic.cpp' if args.quadratic_cell else
                 'cycles_diffraction_cell_chart.cpp' if args.cell_chart else
                 'cycles_diffraction_full_cache.cpp' if args.full_cache else
                 'cycles_diffraction_convergence.cpp' if args.modal_convergence else
                 'cycles_diffraction_adaptive.cpp' if args.adaptive_cache else
                 'cycles_diffraction_cache.cpp' if args.cache_resolution else 'cycles_diffraction_solver.cpp')
    sources = [root / 'intern/cycles/scene/diffraction.cpp', root / 'tests/performance' / benchmark]
    recorded = [*sources, root / 'intern/cycles/scene/diffraction.h',
                root / 'intern/cycles/kernel/util/diffraction_grid.h',
                root / 'intern/cycles/kernel/util/diffraction_table.h', pathlib.Path(__file__)]
    if args.full_cache:
        recorded.extend([root / 'tests/performance/diffraction_packed_audit.h',
                         root / 'tests/performance/diffraction_cache_export.h',
                         root / 'intern/cycles/kernel/util/diffraction_cache.h',
                         root / 'intern/cycles/kernel/util/diffraction_reference.h'])
    if args.probe_cell:
        recorded.append(args.probe_cell.resolve())
    hashes = {str(p.relative_to(root) if p.is_relative_to(root) else p): hashlib.sha256(p.read_bytes()).hexdigest() for p in recorded}
    with tempfile.TemporaryDirectory(prefix='cycles-grating-benchmark-') as temporary:
        executable = pathlib.Path(temporary) / 'solver'
        command = [args.compiler, '-std=c++20', '-O2',
                   '-DCCL_NAMESPACE_BEGIN=namespace ccl {', '-DCCL_NAMESPACE_END=}',
                   '-I' + str(root / 'intern/cycles'), '-I' + str(eigen),
                   *map(str, sources), '-o', str(executable)]
        subprocess.run(command, check=True)
        options = ([str(args.cache_resolution), str(args.queries), str(int(args.grazing_extension)),
                    str(args.wavelength_samples), str(int(args.reference_cache)), str(int(args.hybrid_cache)), str(args.cutoff_margin)] if args.cache_resolution
                   else [str(args.repeats)])
        if args.adaptive_cache:
            options = [str(args.queries), str(args.tolerance), str(args.max_depth), str(args.cutoff_margin), str(args.seed),
                       str(args.half_orders), str(args.reference_orders)]
        if args.modal_convergence:
            options = [str(args.queries), str(args.half_orders), str(args.reference_orders), str(args.seed)]
            if args.modal_query:
                options.extend(map(str,args.modal_query))
        if args.full_cache:
            options = [str(args.max_nodes), str(args.tolerance), str(args.max_depth),
                       str(args.half_orders), str(args.cutoff_margin), str(int(not args.geometric_splits)), str(args.reference_cache_mib), str(int(not args.intensity_only)), str(int(args.mirror_symmetry)), str(args.workers), str(int(args.curvature_splits)), str(args.complex_tolerance), str(int(args.quadratic_cache)), str(args.queries), str(args.seed), str(args.cache_directory.resolve()) if args.cache_directory else ""]
        if args.cell_chart:
            options = [str(args.cutoff_margin)]
        if args.quadratic_cell:
            options = []
        if args.probe_cell:
            cell = json.loads(args.probe_cell.read_text())
            if cell['degree'] != 2:
                parser.error('--probe-cell currently diagnoses quadratic cells')
            p = cell['profile']
            if (p['depth_nm'], p['duty'], p['incident_ior'], p['ridge_ior'], p['groove_ior'], p['substrate_ior']) != (150, 0.41, 1, [0.9, 6], [1, 0], [0.9, 6]):
                parser.error('Cell profile does not match the diagnostic fixture')
            options = list(map(str, [p['pitch_nm'], *cell['lower'], *cell['upper'], *cell['query']]))
        result = subprocess.run([str(executable), *options], text=True, stdout=subprocess.PIPE)
        result.check_returncode()
    report = json.loads(result.stdout)
    report.update(platform=platform.platform(), machine=platform.machine(), source_sha256=hashes,
                  compiler=subprocess.check_output([args.compiler, '--version'], text=True),
                  compile_command=command, execution_options=options, measurement='host precomputation; not GPU render time')
    if args.full_cache and args.cache_directory:
        report['cache_artifacts'] = []
        for case in report['cases']:
            if not case.get('cache_exported'):
                continue
            directory = args.cache_directory.resolve() / str(int(case['pitch_nm']))
            files = {}
            for name in ('cache.json', 'cache.bin'):
                path = directory / name
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                files[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest)
            report['cache_artifacts'].append(dict(pitch_nm=case['pitch_nm'], files=files,
                                                 held_out_validation=case.get('held_out_validation')))
    for case in report['cases'] if not audit else []:
        case['block_median_ms'] = statistics.median(case['block_ms'])
        case['separate_median_ms'] = statistics.median(case['separate_top_incidence_ms'])
        case['separate_over_block'] = case['separate_median_ms'] / case['block_median_ms']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if audit:
        for case in report['cases']:
            print({key: value for key, value in case.items() if key != 'probes'})
        return
    for case in report['cases']:
        print(f"metal={case['metal']} pitch={case['pitch_nm']} N={case['half_orders']}: "
              f"block {case['block_median_ms']:.3f} ms, "
              f"separate {case['separate_median_ms']:.3f} ms, "
              f"ratio {case['separate_over_block']:.2f}")


if __name__ == '__main__':
    main()
