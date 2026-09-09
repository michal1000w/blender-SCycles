#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Serial equal-sample guided CPU/Metal PT/Metal BDPT material benchmark.

Run with system Python; analysis is dispatched through the tested Blender binary
for its OpenImageIO/NumPy dependencies. Requires existing CPU-guided reference
arrays and their per-scene CPU/device/sample provenance JSON. Runtime is diagnostic,
not the budget. BDPT emitted work and camera coverage are checked independently.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

SCENES = ('rough_glass', 'transmission')
MODES = ('cpu_guided', 'metal_pt', 'metal_bdpt')


def training_counts(log):
    cpu = [tuple(map(int, row)) for row in re.findall(
        r'CPU guiding training: sample_start=(\d+) camera_samples=(\d+) '
        r'iteration_before=(\d+) iteration_after=(\d+) limit=(\d+) observations=(\d+)', log)]
    gpu = list(map(int, re.findall(r'Metal guiding publish: trained_samples=(\d+)', log)))
    return {'cpu_batches': cpu,
            'cpu_field_updates': max((row[3] for row in cpu), default=None),
            'cpu_training_camera_samples': sum(row[1] for row in cpu if row[4] == 0 or row[2] < row[4])
                                           if cpu else None,
            'metal_publications': gpu,
            'metal_training_camera_samples': max(gpu, default=None)}


def analyze(output):
    import numpy as np
    import OpenImageIO as oiio
    manifest = json.loads((output / 'manifest.json').read_text())
    work = json.loads((output / 'work_counts.json').read_text())
    expected = {f'{scene}_{mode}_{seed}' for scene in SCENES for mode in MODES
                for seed in manifest['seeds']}
    assert set(work) == expected and all(x['valid'] for x in work.values())
    result = {'scope': 'Equal camera samples, CPU-guided high-SPP primary reference. '
                       'Instrumented runtime is separate. Multiple seeds do not prove unbiasedness.',
              'manifest': manifest, 'scenes': {}}
    references = pathlib.Path(manifest['references'])
    for scene in SCENES:
        ref_meta = json.loads((references / f'{scene}_reference.json').read_text())
        assert ref_meta['device'] == 'CPU' and ref_meta['guiding']
        assert ref_meta['samples_per_seed'] > manifest['samples']
        reference = np.load(references / f'{scene}_reference.npy').astype(np.float64)
        images, groups = {}, {}
        for mode in MODES:
            for seed in manifest['seeds']:
                name = f'{scene}_{mode}_{seed}'
                metadata = json.loads((output / f'{name}.json').read_text())
                assert metadata['samples'] == manifest['samples']
                assert metadata['seed'] == seed and metadata['time_limit'] == 0
                source = oiio.ImageInput.open(str(output / f'{name}.exr'))
                assert source is not None, name
                rgb = np.asarray(source.read_image(format=oiio.FLOAT))[..., :3].astype(np.float64)
                source.close()
                assert rgb.shape == reference.shape and np.isfinite(rgb).all(), name
                mse = float(np.mean((rgb - reference) ** 2))
                regions = {}
                for region, (x0, y0, x1, y1) in metadata['object_regions'].items():
                    a, b = rgb[y0:y1, x0:x1], reference[y0:y1, x0:x1]
                    regions[region] = {'mse': float(np.mean((a-b)**2)),
                                       'relative_mean_error': float(a.mean()/b.mean()-1)}
                images[name] = {'mse': mse, 'relative_mean_error': float(rgb.mean()/reference.mean()-1),
                                'regions': regions, 'work': work[name]}
                groups.setdefault(mode, []).append(mse)
        summary = {mode: {'mean_mse': float(np.mean(values)),
                          'mse_sample_stddev': float(np.std(values, ddof=1)),
                          'per_seed_mse': values} for mode, values in groups.items()}
        for item in summary.values():
            item['ratio_to_cpu_guided'] = item['mean_mse']/summary['cpu_guided']['mean_mse']
        result['scenes'][scene] = {'reference_metadata': ref_meta, 'images': images, 'summary': summary}
    (output / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v['summary'] for k, v in result['scenes'].items()}, indent=2), flush=True)


def main():
    args = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    root = pathlib.Path(__file__).resolve().parents[2]
    parser.add_argument('--blender', type=pathlib.Path,
                        default=root / 'install/Blender.app/Contents/MacOS/Blender')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--references', type=pathlib.Path,
                        default=root / 'build/metal-guiding-tests/cache-order-material-bdpt')
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 211, 307])
    parser.add_argument('--mis-exponent', type=int, choices=(1, 2), default=2)
    parser.add_argument('--analyze', action='store_true')
    options = parser.parse_args(args)
    output = options.output.resolve()
    if options.analyze:
        analyze(output)
        return
    assert options.samples > 128 and len(set(options.seeds)) == len(options.seeds) >= 2
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError('Use an empty output directory to preserve existing evidence')
    blender = options.blender.resolve()
    headers = list((blender.parents[1] / 'Resources').glob(
        '*/scripts/addons_core/cycles/source/kernel/integrator/bidirectional_transport.h'))
    assert len(headers) == 1, headers
    source = headers[0].read_bytes()
    assert f'using BDPTMISWeight = BDPTMISWeightT<{options.mis_exponent}>;'.encode() in source
    manifest = {'samples': options.samples, 'seeds': options.seeds, 'mis_exponent': options.mis_exponent,
                'references': str(options.references.resolve()), 'blender': str(blender),
                'installed_mis_header_sha256': hashlib.sha256(source).hexdigest(),
                'light_paths_per_four_camera_samples': 16384, 'training_samples': 128,
                'resolution': 256, 'memory_mb': 256,
                'cpu_training_limit_unit': 'field updates',
                'metal_training_limit_unit': 'camera samples'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    work = {}
    for scene in SCENES:
        for seed in options.seeds:
            for mode in MODES:
                name = f'{scene}_{mode}_{seed}'
                command = [str(blender), '--background', '--factory-startup', '--python-exit-code', '1',
                           '--debug-cycles', '--log-level', 'debug', '--python',
                           str(root / 'tests/python/cycles_metal_guiding_material_scene.py'), '--',
                           '--scene', scene, '--device', 'CPU' if mode == 'cpu_guided' else 'GPU',
                           '--guiding', '--samples', str(options.samples), '--resolution', '256',
                           '--seed', str(seed), '--training-samples', '128', '--memory-mb', '256',
                           '--light-paths', '16384', '--output', str(output / name)]
                if mode == 'metal_bdpt':
                    command.append('--bdpt')
                (output / f'{name}.command.json').write_text(json.dumps(command, indent=2)+'\n')
                env = os.environ.copy()
                env['CYCLES_BDPT_DIAGNOSTICS'] = '1'
                with (output / f'{name}.log').open('w') as log:
                    process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                log = (output / f'{name}.log').read_text()
                counts = re.findall(r'Rendered (\d+) samples in ([\d.]+) seconds', log)
                batches = [tuple(map(int, x)) for x in re.findall(
                    r'BDPT diagnostics: sample=(\d+) emitted=(\d+) camera_samples=(\d+)', log)]
                covered = [i for start, _, count in batches for i in range(start, start+count)]
                emitted = sum(x[1] for x in batches)
                valid = process.returncode == 0 and counts and int(counts[-1][0]) == options.samples
                if mode == 'metal_bdpt':
                    valid = valid and covered == list(range(options.samples))
                    valid = valid and emitted == options.samples * 4096
                    valid = valid and 'BDPT image cache miss' not in log
                else:
                    valid = valid and not batches
                work[name] = {'valid': bool(valid), 'returncode': process.returncode,
                              'actual_spp': int(counts[-1][0]) if counts else None,
                              'instrumented_seconds': float(counts[-1][1]) if counts else None,
                              'emitted_light_paths': emitted, 'batches': batches,
                              'training': training_counts(log)}
                (output / 'work_counts.json').write_text(json.dumps(work, indent=2)+'\n')
                print(name, 'passed' if valid else 'FAILED', flush=True)
                if not valid:
                    raise RuntimeError(f'{name}: render or work-count check failed')
    subprocess.run([str(blender), '--background', '--factory-startup', '--python-exit-code', '1',
                    '--python', str(pathlib.Path(__file__).resolve()), '--', '--analyze',
                    '--output', str(output)], check=True)


if __name__ == '__main__':
    main()
