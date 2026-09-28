# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify a completed physical Blender transport batch and compare paired seeds.

Uncertainty is across independent render seeds, never across neighboring pixels.
No finite-sample agreement is automatically labeled proof of correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def estimate(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError('Need at least two finite independent seed observations')
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def analyze(directory, mask_directory=None):
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'rendered_pending_analysis':
        raise ValueError('Batch is not complete; refusing partial convergence statistics')
    seeds, transports = manifest['seeds'], manifest['transports']
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or len(set(transports)) != len(transports):
        raise ValueError('Invalid seed or transport list')
    if 'pt' not in transports:
        raise ValueError('A matched PT baseline is required')
    expected = {(mode, seed) for mode in transports for seed in seeds}
    found = set()
    images = {mode: {} for mode in transports}
    shape = None
    for run in manifest['runs']:
        key = (run['transport'], run['seed'])
        if key not in expected or key in found or run['samples'] != manifest['samples']:
            raise ValueError('Unexpected, duplicate or mismatched run')
        found.add(key)
        if run['status'] != 'rendered_pending_convergence_analysis' or run['array_origin'] != 'bottom_left':
            raise ValueError('Run is incomplete or has an unsupported pixel origin')
        prefix = directory / f'{key[0]}_seed{key[1]}'
        if json.loads(prefix.with_suffix('.json').read_text()) != run:
            raise ValueError('Per-run report differs from the batch manifest')
        for extension in ['blend', 'exr', 'npy', 'png']:
            if sha256(prefix.with_suffix('.' + extension)) != run['sha256'][extension]:
                raise ValueError('Artifact hash mismatch: ' + str(prefix))
        rgb = np.load(prefix.with_suffix('.npy'), allow_pickle=False)
        if (rgb.shape != (run['resolution'][1], run['resolution'][0], 3) or
                not np.isfinite(rgb).all() or (shape is not None and rgb.shape != shape)):
            raise ValueError('Invalid or inconsistent image dimensions/pixels')
        shape = rgb.shape
        mean = rgb.mean(axis=(0, 1), dtype=np.float64)
        if not np.allclose(mean, run['raw_mean_rgb'], rtol=0, atol=1e-12):
            raise ValueError('Reported mean differs from raw pixels')
        images[key[0]][key[1]] = rgb.astype(np.float64)
    if found != expected:
        raise ValueError('Missing runs in supposedly complete batch')
    results = dict(scope='Paired-seed whole-image and optional receiver observations; not a convergence proof.',
                   comparison_reference='PT is a diagnostic estimator, not physical ground truth.',
                   absolute_correctness='Neither agreement nor disagreement establishes physical correctness.',
                   manifest_sha256=sha256(manifest_path), seeds=seeds,
                   samples=manifest['samples'], groups={}, comparisons={},
                   uncertainty='Standard error across independent seeds; four seeds alone do not establish asymptotic normality.')
    baseline = np.stack([images['pt'][seed] for seed in seeds])
    regions = {}
    if mask_directory is not None:
        mask_report = json.loads((mask_directory / 'receiver_masks.json').read_text())
        mask_path = mask_directory / 'receiver_masks.npz'
        if (mask_report['input_sha256'] != manifest['input_sha256'] or
                mask_report['array_origin'] != 'bottom_left' or
                mask_report['mask_sha256'] != sha256(mask_path)):
            raise ValueError('Receiver masks do not match the batch input/provenance')
        with np.load(mask_path, allow_pickle=False) as archive:
            for name in archive.files:
                mask = archive[name]
                if (mask.shape != shape[:2] or mask.dtype != np.bool_ or
                        int(mask.sum()) != mask_report['pixels'][name]):
                    raise ValueError('Invalid receiver mask')
                if mask.any():
                    regions[name] = mask
        results['receiver_mask_sha256'] = sha256(mask_path)
        results['receiver_comparisons'] = {}
    maps = {}
    for mode in transports:
        stack = np.stack([images[mode][seed] for seed in seeds])
        means = stack.mean(axis=(1, 2))
        mean, stderr = estimate(means)
        results['groups'][mode] = dict(seed_mean_rgb=means.tolist(), mean_rgb=mean.tolist(),
                                       standard_error_rgb=stderr.tolist())
        if mode == 'pt':
            continue
        paired = stack - baseline
        difference, error = estimate(paired.mean(axis=(1, 2)))
        map_mean, map_error = estimate(paired)
        maps[mode + '_difference'] = map_mean
        maps[mode + '_standard_error'] = map_error
        results['comparisons'][mode + '_minus_pt'] = dict(mean_difference_rgb=difference.tolist(),
                                                         standard_error_rgb=error.tolist())
        for name, mask in regions.items():
            difference, error = estimate(paired[:, mask, :].mean(axis=1))
            results['receiver_comparisons'].setdefault(name, {})[mode + '_minus_pt'] = dict(
                pixels=int(mask.sum()), mean_difference_rgb=difference.tolist(),
                standard_error_rgb=error.tolist())
    return results, maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--masks', type=Path)
    parser.add_argument('--baseline', type=Path, help='Completed batch before the candidate change')
    args = parser.parse_args()
    results, maps = analyze(args.directory, args.masks)
    if args.baseline is not None:
        previous, _ = analyze(args.baseline, args.masks)
        current_manifest = json.loads((args.directory / 'manifest.json').read_text())
        previous_manifest = json.loads((args.baseline / 'manifest.json').read_text())
        for key in ('input_sha256', 'binary_sha256', 'script_sha256', 'seeds', 'samples',
                    'transports', 'emitters', 'bdpt_light_paths', 'bdpt_update_samples',
                    'caustics_reflective', 'caustics_refractive', 'persistent_data'):
            if current_manifest[key] != previous_manifest[key]:
                raise ValueError('Baseline configuration mismatch: ' + key)
        old_runs = {(r['transport'], r['seed']): r for r in previous_manifest['runs']}
        same_pixels = {}
        for run in current_manifest['runs']:
            key = (run['transport'], run['seed'])
            same_pixels.setdefault(key[0], []).append(
                run['sha256']['npy'] == old_runs[key]['sha256']['npy'])
        changes = {}
        for mode in current_manifest['transports']:
            values = (np.asarray(results['groups'][mode]['seed_mean_rgb']) -
                      np.asarray(previous['groups'][mode]['seed_mean_rgb']))
            mean, error = estimate(values)
            changes[mode] = dict(mean_difference_rgb=mean.tolist(),
                                 standard_error_rgb=error.tolist(),
                                 identical_raw_arrays_by_seed=same_pixels[mode])
        results['candidate_minus_baseline'] = dict(
            baseline_manifest_sha256=previous['manifest_sha256'],
            current_kernel=current_manifest['kernel_source_fingerprint'],
            baseline_kernel=previous_manifest['kernel_source_fingerprint'],
            comparisons=changes)
    map_path = args.directory / 'paired_seed_maps.npz' 
    np.savez_compressed(map_path, **maps)
    results['pixel_maps_sha256'] = sha256(map_path)
    results['pixel_maps_origin'] = 'bottom_left'
    (args.directory / 'verified_comparison.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results['comparisons'], indent=2))


if __name__ == '__main__':
    main()
