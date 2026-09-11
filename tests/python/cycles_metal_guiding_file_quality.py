#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare independent-seed renders against a separately rendered linear reference.

Reference noise contributes to both errors. This report does not automatically certify
equivalence: inspect reference convergence, per-seed results, and images as well.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=pathlib.Path, nargs='+', required=True)
    parser.add_argument('--candidate', type=pathlib.Path, nargs='+', required=True)
    parser.add_argument('--reference', type=pathlib.Path, nargs='+', required=True)
    parser.add_argument('--mask-reference', type=pathlib.Path, nargs='+',
                        help='Independent renders used only to choose luminance regions')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if len(args.baseline) != len(args.candidate):
        parser.error('Use equal numbers of baseline and candidate seeds')

    def metadata(paths):
        return [json.loads(path.with_suffix('.json').read_text()) for path in paths]

    base_metadata, candidate_metadata, reference_metadata = map(
        metadata, [args.baseline, args.candidate, args.reference])
    mask_metadata = metadata(args.mask_reference or [])
    test_seeds = set()
    scene_hash = base_metadata[0]['scene_sha256']
    for baseline_run, candidate_run in zip(base_metadata, candidate_metadata):
        if baseline_run['cycles'] != candidate_run['cycles']:
            raise ValueError('Baseline and candidate render settings differ')
        if baseline_run['resolution'] != candidate_run['resolution']:
            raise ValueError('Baseline and candidate resolutions differ')
        if not baseline_run['cycles']['use_guiding']:
            raise ValueError('The comparison must retain path guiding')
        seed = baseline_run['cycles']['seed']
        if seed in test_seeds:
            raise ValueError('Repeated test seeds do not measure independent variation')
        test_seeds.add(seed)
    reference_seeds = [run['cycles']['seed'] for run in reference_metadata + mask_metadata]
    if len(set(reference_seeds)) != len(reference_seeds) or test_seeds.intersection(reference_seeds):
        raise ValueError('Reference seeds must be independent of the compared renders')
    if any(run['scene_sha256'] != scene_hash
           for run in base_metadata + candidate_metadata + reference_metadata + mask_metadata):
        raise ValueError('Scene contents differ')
    reference_settings = dict(base_metadata[0]['cycles'])
    # References use fixed, higher sample counts and linear, undenoised output.
    # All transport and guiding-training settings must still match the test.
    reference_overrides = ['samples', 'seed', 'use_adaptive_sampling', 'use_denoising', 'time_limit']
    for key in reference_overrides:
        reference_settings.pop(key, None)
    for run in reference_metadata + mask_metadata:
        settings = dict(run['cycles'])
        if (settings['use_adaptive_sampling'] or settings['use_denoising'] or settings['time_limit'] != 0
                or settings['samples'] < base_metadata[0]['cycles']['samples']):
            raise ValueError('References must use at least the test budget, fixed samples, and no denoising')
        for key in reference_overrides:
            settings.pop(key, None)
        if settings != reference_settings or run['resolution'] != base_metadata[0]['resolution']:
            raise ValueError('Reference transport, training, or image settings differ')

    def read(paths):
        values = [np.load(path).astype(np.float64) for path in paths]
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError('Nonfinite pixels')
        return np.stack(values)

    baseline, candidate, references = map(read, [args.baseline, args.candidate, args.reference])
    reference = references.mean(axis=0)
    if baseline.shape[1:] != reference.shape or candidate.shape != baseline.shape:
        raise ValueError('Image dimensions differ')
    mask_reference = read(args.mask_reference).mean(axis=0) if args.mask_reference else reference
    if mask_reference.shape != reference.shape:
        raise ValueError('Region reference dimensions differ')
    luminance = mask_reference @ np.array([0.2126, 0.7152, 0.0722])
    positive = luminance[luminance > 0]
    if not positive.size:
        raise ValueError('A black reference cannot validate guiding image quality')
    thresholds = np.quantile(positive, [0.5, 0.9, 0.99])
    masks = {'all': np.ones(luminance.shape, dtype=bool),
             'lower_half': luminance <= thresholds[0],
             'brightest_10_percent': luminance >= thresholds[1],
             'brightest_1_percent': luminance >= thresholds[2]}
    result = {'baseline': [str(p) for p in args.baseline],
              'candidate': [str(p) for p in args.candidate],
              'references': [str(p) for p in args.reference],
              'mask_references': [str(p) for p in args.mask_reference or []],
              'region_mask_independent': bool(args.mask_reference),
              'independent_reference_count': len(references),
              'quality_accepted': False, 'regions': {},
              'analysis_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              'mse_regression_margin': 0.05,
              'uncertainty_note': 'Student-t intervals across independent test seeds; reference-noise '
                                  'correction is estimated and requires convergence/visual review. '
                                  'Luminance-selected regions need an independent mask reference '
                                  'to avoid conditioning their errors on the same reference noise.'}
    for name, mask in masks.items():
        errors = [(np.square(images - reference)[:, mask]).mean(axis=(1, 2))
                  for images in [baseline, candidate]]
        difference = errors[1] - errors[0]
        reference_noise = (float(references[:, mask].var(axis=0, ddof=1).mean() / len(references))
                           if len(references) > 1 else None)
        corrected_means = ([float(error.mean() - reference_noise) for error in errors]
                           if reference_noise is not None else None)
        # A noisy reference adds a common error floor. Correct the 5% comparison
        # so this floor cannot make a worse candidate appear equivalent.
        margin_difference = (errors[1] - 1.05 * errors[0] + 0.05 * reference_noise
                             if reference_noise is not None else None)
        # Two-sided 95% Student-t critical values (df 1..30); df=30 is
        # conservative for larger test ensembles. Never infer uncertainty from
        # the number of pixels: the independent units here are render seeds.
        critical_values = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
                           2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
                           2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
                           2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
        margin_upper = None
        if margin_difference is not None and len(difference) > 1:
            critical = critical_values[min(len(difference) - 2, len(critical_values) - 1)]
            margin_upper = float(margin_difference.mean() + critical *
                                 margin_difference.std(ddof=1) / np.sqrt(len(difference)))
        result['regions'][name] = {
            'pixels': int(mask.sum()),
            'baseline_mse': errors[0].tolist(), 'candidate_mse': errors[1].tolist(),
            'mean_mse_ratio': float(errors[1].mean() / errors[0].mean()),
            'paired_mse_difference': float(difference.mean()),
            'paired_difference_standard_error': (float(difference.std(ddof=1) / np.sqrt(len(difference)))
                                                 if len(difference) > 1 else None),
            'baseline_mean_rgb': baseline[:, mask].mean(axis=(0, 1)).tolist(),
            'candidate_mean_rgb': candidate[:, mask].mean(axis=(0, 1)).tolist(),
            'reference_mean_rgb': reference[mask].mean(axis=0).tolist(),
            'reference_mean_estimated_noise_mse': reference_noise,
            'noise_corrected_mean_mse': corrected_means,
            'noise_corrected_mse_ratio': (corrected_means[1] / corrected_means[0]
                                          if corrected_means is not None and
                                          min(corrected_means) > 0 else None),
            'corrected_5_percent_margin_difference': (float(margin_difference.mean())
                                                     if margin_difference is not None else None),
            'corrected_5_percent_margin_upper_95_two_sided': margin_upper,
            'mse_ratio_against_each_reference': [
                float(np.square(candidate[:, mask] - ref[mask]).mean() /
                      np.square(baseline[:, mask] - ref[mask]).mean()) for ref in references],
        }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
