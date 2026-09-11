#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Check independent BDPT batches using canonical per-sample caches and linear pixels.

Guiding is disabled only in this transport-equivalence test, to avoid unrelated
atomic training-order variation. Volume scattering-probability guiding remains
active independently and reads online optical-depth estimates from the film.
Consequently volume renders need statistical quality comparison unless that
learning is isolated in an explicitly diagnostic-only build. Exact cache checks
still apply. Performance/quality acceptance must use all saved guiding features.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', type=pathlib.Path, required=True)
    parser.add_argument('--baseline', type=pathlib.Path)
    parser.add_argument('--scene', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=range(2, 5), default=4)
    parser.add_argument('--default-batch', action='store_true',
                        help='Exercise the application default without a batch-size environment override')
    parser.add_argument('--cache-only', action='store_true',
                        help='Check exact light caches; report but do not certify pixels with online VSPG')
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--percentage', type=int, default=25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    script = pathlib.Path(__file__).with_name('cycles_metal_guiding_file_benchmark.py').resolve()
    variants = [('single', args.blender, 1), ('batch', args.blender, args.batch_size)]
    if args.baseline:
        variants.insert(0, ('original', args.baseline, 1))
    outputs = {}
    for name, app, batch_size in variants:
        folder = (args.output / name).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        if any(folder.glob('cache_*.vertices')):
            raise RuntimeError(f'Use a fresh output directory; stale cache snapshots exist in {folder}')
        prefix = folder / 'render'
        env = {k: v for k, v in os.environ.items() if not k.startswith('CYCLES_')}
        env.update(CYCLES_METAL_BDPT_BATCH_SIZE=str(batch_size), CYCLES_BDPT_DIAGNOSTICS='1',
                   CYCLES_BDPT_CACHE_DUMP=str(folder / 'cache'))
        if name == 'batch' and args.default_batch:
            env.pop('CYCLES_METAL_BDPT_BATCH_SIZE')
        command = [str(app.resolve()), '--background', '--factory-startup', '--disable-autoexec',
                   '--debug-cycles', '--log-level', 'debug', '--python-exit-code', '1',
                   '--python', str(script), '--', '--scene', str(args.scene.resolve()),
                   '--output', str(prefix), '--integrator', 'bdpt', '--no-guiding', '--raw',
                   '--fixed-samples', '--samples', str(args.samples), '--percentage', str(args.percentage)]
        print('RUN', name, flush=True)
        with prefix.with_suffix('.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=1800, check=True)
        log = prefix.with_suffix('.log').read_text()
        batches = [(int(s), int(e), int(c)) for s, e, c in re.findall(
            r'BDPT diagnostics: sample=(\d+) emitted=(\d+) camera_samples=(\d+)', log)]
        expected = 0
        for sample, emitted, cameras in batches:
            if sample != expected or cameras <= 0 or emitted <= 0:
                raise RuntimeError(f'{name}: incomplete sample/light coverage')
            expected += cameras
        if expected != args.samples:
            raise RuntimeError(f'{name}: expected {args.samples} samples, got {expected}')
        maximum_batch = max(c for _, _, c in batches)
        if name == 'batch' and maximum_batch < args.batch_size:
            raise RuntimeError('The requested batch size was not exercised; increase samples '
                               'or check the scene cache capacity')
        caches = {i: (folder / f'cache_{i}.vertices').read_bytes() for i in range(args.samples)}
        pixels = np.load(prefix.with_suffix('.npy'))
        if not np.isfinite(pixels).all():
            raise RuntimeError(f'{name}: nonfinite pixels')
        outputs[name] = (caches, pixels, sum(e for _, e, _ in batches), maximum_batch)

    reference_name = variants[0][0]
    reference_caches, reference_pixels, reference_emitted, _ = outputs[reference_name]
    report = {'reference': reference_name, 'samples': args.samples, 'variants': {},
              'performance_or_guiding_quality_accepted': False,
              'requested_batch_size': args.batch_size,
              'batch_environment_override': not args.default_batch,
              'pixel_comparison_required': not args.cache_only}
    pixel_failures = []
    for name, (caches, pixels, emitted, maximum_batch) in outputs.items():
        if caches != reference_caches:
            differing = [i for i in caches if caches[i] != reference_caches[i]]
            raise RuntimeError(f'{name}: canonical light caches differ at samples {differing}')
        if emitted != reference_emitted:
            raise RuntimeError(f'{name}: emitted light-path budget changed')
        difference = np.abs(pixels.astype(np.float64) - reference_pixels)
        pixels_equivalent = bool(np.allclose(pixels, reference_pixels, rtol=1e-5, atol=1e-6))
        if not pixels_equivalent:
            pixel_failures.append(name)
        report['variants'][name] = {'canonical_caches_identical': True,
                                    'pixels_within_accumulation_tolerance': pixels_equivalent,
                                    'emitted_light_paths': emitted,
                                    'maximum_observed_batch_size': maximum_batch,
                                    'max_pixel_difference': float(difference.max()),
                                    'mean_pixel_difference': float(difference.mean())}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if pixel_failures and not args.cache_only:
        raise RuntimeError(f'Pixel differences exceed accumulation tolerance: {pixel_failures}')


if __name__ == '__main__':
    main()
