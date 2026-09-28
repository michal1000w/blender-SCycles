# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify saved control artifacts and measure paired-seed PT/BDPT differences."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(directories, require_provenance=False):
    settings = None
    source_record = None
    provenance_records = []
    seeds = set()
    observations = []
    for directory in directories:
        provenance_path = directory / 'provenance.json'
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            if (provenance['status'] != 'complete' or
                    provenance['before'] != provenance['after']):
                raise ValueError(f'{directory}: incomplete or changed render provenance')
            if source_record is None:
                source_record = provenance['before']
            if provenance['before'] != source_record:
                raise ValueError('input, binary, script or source differs across seeds')
            provenance_records.append(hashlib.sha256(provenance_path.read_bytes()).hexdigest())
        elif require_provenance:
            raise ValueError(f'{directory}: render provenance is required')
        reports = json.loads((directory / 'manifest.json').read_text())
        if len(reports) != 2 or {r['transport'] for r in reports} != {'pt', 'bdpt'}:
            raise ValueError(f'{directory}: require exactly one PT and BDPT render')
        means = {}
        seed = reports[0]['seed']
        if seed in seeds:
            raise ValueError(f'duplicate seed {seed}')
        seeds.add(seed)
        for report in reports:
            if report['seed'] != seed or report['diffraction_enabled'] is not False:
                raise ValueError('inconsistent seed or non-control material')
            current = {key: report[key] for key in (
                'kind', 'emitter', 'persistent_data', 'samples', 'resolution', 'bounce_limits')}
            if settings is None:
                settings = current
            if current != settings:
                raise ValueError('control settings differ between renders')
            prefix = directory / (report['kind'] + '_' + report['transport'])
            for extension in ('blend', 'exr', 'npy', 'png'):
                artifact = prefix.with_suffix('.' + extension)
                if hashlib.sha256(artifact.read_bytes()).hexdigest() != report['sha256'][extension]:
                    raise ValueError(f'artifact hash mismatch: {artifact}')
            image = np.load(prefix.with_suffix('.npy'), allow_pickle=False)
            width, height = report['resolution']
            if image.shape != (height, width, 3) or not np.isfinite(image).all():
                raise ValueError(f'invalid image: {prefix}')
            mean = image.mean(axis=(0, 1), dtype=np.float64)
            if not np.allclose(mean, report['raw_mean_rgb'], rtol=0, atol=1e-10):
                raise ValueError(f'reported mean mismatch: {prefix}')
            means[report['transport']] = mean
        observations.append(dict(seed=seed, directory=str(directory.resolve()),
                                 pt=means['pt'].tolist(), bdpt=means['bdpt'].tolist(),
                                 difference=(means['bdpt'] - means['pt']).tolist()))
    if len(seeds) < 2:
        raise ValueError('at least two independent seeds are required')
    differences = np.array([r['difference'] for r in observations])
    return dict(status='artifacts_verified_observational_comparison', settings=settings,
                comparison_reference='PT is a diagnostic estimator, not physical ground truth.',
                absolute_correctness='Neither agreement nor disagreement establishes physical correctness.',
                observations=observations, seed_count=len(seeds),
                provenance_record_sha256=provenance_records,
                complete_matching_provenance_records=len(provenance_records) == len(directories),
                mean_bdpt_minus_pt=differences.mean(axis=0).tolist(),
                paired_seed_standard_error=(differences.std(axis=0, ddof=1) /
                                            np.sqrt(len(seeds))).tolist(),
                scope='Whole-image means; recorded provenance checked when present; '
                      'historical source files are not independently reconstructed. Not a convergence pass, '
                      'regional correctness test or isolated performance benchmark.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-provenance', action='store_true')
    args = parser.parse_args()
    result = analyze(args.directories, args.require_provenance)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
