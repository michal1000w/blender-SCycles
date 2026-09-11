#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Serial, resumable comparisons on existing blend scenes; never run timed GPUs concurrently."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=pathlib.Path, required=True)
    parser.add_argument('--candidate', type=pathlib.Path, required=True)
    parser.add_argument('--scenes', type=pathlib.Path, nargs='+', required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 47])
    parser.add_argument('--samples', type=int)
    parser.add_argument('--percentage', type=int)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--fixed-samples', action='store_true')
    parser.add_argument('--integrator', choices=['saved', 'pt', 'bdpt'], default='saved')
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--warmup', choices=['all', 'first', 'none'], default='all',
                        help='Full warmup per seed, per scene/variant, or no warmup')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    script = pathlib.Path(__file__).with_name('cycles_metal_guiding_file_benchmark.py').resolve()
    apps = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    hashes = {name: hashlib.sha256(app.read_bytes()).hexdigest() for name, app in apps.items()}
    kernel_hashes = {}
    for name, app in apps.items():
        digest = hashlib.sha256()
        source = app.parents[1] / 'Resources' / '5.3/scripts/addons_core/cycles/source'
        if not source.is_dir():
            raise RuntimeError(f'Missing installed Cycles source: {source}')
        for path in sorted(source.rglob('*')):
            if path.is_file():
                digest.update(str(path.relative_to(source)).encode() + b'\0')
                digest.update(path.read_bytes())
        kernel_hashes[name] = digest.hexdigest()
    env = dict(os.environ)
    for key in ['CYCLES_METAL_PROFILING', 'CYCLES_KERNEL_PATH', 'CYCLES_METAL_DEBUG']:
        env.pop(key, None)
    env['CYCLES_BDPT_DIAGNOSTICS'] = '1'
    exclusion = args.output / 'TIMING_NOT_ACCEPTED.json'
    report = {'cases': {}, 'profiling': False, 'quality_accepted': False,
              'timing_exclusion': json.loads(exclusion.read_text()) if exclusion.exists() else None}
    for scene in args.scenes:
        case = report['cases'][scene.stem] = {'variants': {}, 'scene': str(scene.resolve())}
        pixels = {'baseline': [], 'candidate': []}
        for seed_index, seed in enumerate(args.seeds):
            for name in (['baseline', 'candidate'] if seed_index % 2 == 0 else ['candidate', 'baseline']):
                prefix = (args.output / f'{scene.stem}-{name}-{seed}').resolve()
                command = [str(apps[name]), '--background', '--factory-startup', '--disable-autoexec',
                           '--debug-cycles', '--log-level', 'debug', '--python-exit-code', '1',
                           '--python', str(script), '--',
                           '--scene', str(scene.resolve()), '--output', str(prefix), '--seed', str(seed),
                           '--integrator', args.integrator]
                if args.warmup == 'all' or (args.warmup == 'first' and seed_index == 0):
                    command.append('--warmup')
                for option in ['samples', 'percentage']:
                    if getattr(args, option) is not None:
                        command += ['--' + option, str(getattr(args, option))]
                for option in ['raw', 'fixed_samples']:
                    if getattr(args, option):
                        command.append('--' + option.replace('_', '-'))
                config = {'command': command, 'binary': hashes[name],
                          'kernel_sources': kernel_hashes[name],
                          'scene': hashlib.sha256(scene.read_bytes()).hexdigest(),
                          'script': hashlib.sha256(script.read_bytes()).hexdigest(),
                          'cycles_environment': {key: value for key, value in env.items()
                                                 if key.startswith('CYCLES_')}}
                stamp = prefix.with_suffix('.config.json')
                cached = stamp.exists() and json.loads(stamp.read_text()) == config
                if not (cached and prefix.with_suffix('.json').exists() and prefix.with_suffix('.npy').exists()):
                    print('RENDER', scene.stem, name, seed, flush=True)
                    with prefix.with_suffix('.log').open('w') as log:
                        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=args.timeout)
                    if result.returncode:
                        raise RuntimeError(f'{prefix}: Blender exit {result.returncode}')
                    stamp.write_text(json.dumps(config, indent=2))
                metrics = json.loads(prefix.with_suffix('.json').read_text())
                render_log = prefix.with_suffix('.log').read_text()
                counts = re.findall(r'Rendered (\d+) samples in', render_log)
                metrics['rendered_samples'] = int(counts[-1]) if counts else None
                settings = metrics['cycles']
                actual = metrics['rendered_samples']
                if actual is None or not 0 < actual <= settings['samples']:
                    raise RuntimeError(f'{prefix}: actual sample count was not verified')
                if (settings['time_limit'] == 0 and not settings['use_adaptive_sampling']
                        and actual != settings['samples']):
                    raise RuntimeError(f'{prefix}: fixed sample budget was not completed')
                if settings['use_bidirectional_path_tracing']:
                    batches = [dict(sample=int(sample), emitted=int(emitted), camera_samples=int(camera))
                               for sample, emitted, camera in re.findall(
                                   r'BDPT diagnostics: sample=(\d+) emitted=(\d+) camera_samples=(\d+)',
                                   render_log)]
                    starts = [i for i, batch in enumerate(batches) if batch['sample'] == 0]
                    measured = batches[starts[-1]:] if starts else []
                    expected = 0
                    for batch in measured:
                        if batch['sample'] != expected or batch['camera_samples'] <= 0:
                            raise RuntimeError(f'{prefix}: invalid BDPT sample coverage')
                        expected += batch['camera_samples']
                    if expected != metrics['rendered_samples']:
                        raise RuntimeError(f'{prefix}: missing BDPT sample coverage')
                    metrics['bdpt_camera_samples_verified'] = expected
                    metrics['bdpt_emitted_light_paths'] = sum(b['emitted'] for b in measured)
                prefix.with_suffix('.json').write_text(json.dumps(metrics, indent=2))
                rgb = np.load(prefix.with_suffix('.npy'))
                if not np.isfinite(rgb).all():
                    raise RuntimeError(f'{prefix}: nonfinite image')
                pixels[name].append(rgb)
                case['variants'].setdefault(name, []).append(metrics)
                print('RESULT', scene.stem, name, seed, metrics['seconds'], flush=True)
                (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        baseline = np.array([m['seconds'] for m in case['variants']['baseline']])
        candidate = np.array([m['seconds'] for m in case['variants']['candidate']])
        case['median_speedup'] = float(np.median(baseline) / np.median(candidate))
        case['paired_speedups'] = (baseline / candidate).tolist()
        # Diagnostic only. Candidate/baseline differences cannot establish error without a reference.
        case['paired_image_mse'] = [float(np.mean((a.astype(float) - b)**2))
                                    for a, b in zip(pixels['baseline'], pixels['candidate'])]
        print('CASE', scene.stem, 'speedup', case['median_speedup'], flush=True)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
