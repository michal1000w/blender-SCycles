# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare enclosure renders with the analytic radiance, never with PT as truth."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(directories):
    if len(directories) < 2:
        raise ValueError('At least two independent seeds are required')
    groups = {'pt': [], 'bdpt': []}
    seeds = set()
    settings = source = None
    for directory in directories:
        reports = json.loads((directory / 'manifest.json').read_text())
        if len(reports) != 2 or {r['transport'] for r in reports} != set(groups):
            raise ValueError('Expected one completed PT and BDPT render')
        seed = reports[0]['seed']
        if seed in seeds:
            raise ValueError('Duplicate seed')
        seeds.add(seed)
        for r in reports:
            if r['seed'] != seed or r['status'] != 'rendered_pending_analysis':
                raise ValueError('Incomplete or mismatched report')
            current = {k: r[k] for k in ('kind', 'user_bounces', 'allowed_scattering_events',
                                         'samples', 'expected_rgb', 'infinite_depth_rgb')}
            current['total_bounces'] = r.get('total_bounces', r['user_bounces'])
            if settings is None:
                settings = current
            if settings != current:
                raise ValueError('Reference/settings differ across runs')
            before, after = r['provenance']['before'], r['provenance']['after']
            if before != after:
                raise ValueError('Render inputs changed during run')
            current_source = {k: v for k, v in before.items() if k != 'input_sha256'}
            if source is None:
                source = current_source
            if current_source != source:
                raise ValueError('Different code/binary across seeds')
            stem = directory / (r['kind'] + '_' + r['transport'])
            for extension in ('blend', 'exr', 'npy'):
                path = stem.with_suffix('.' + extension)
                if hashlib.sha256(path.read_bytes()).hexdigest() != r['sha256'][extension]:
                    raise ValueError('Artifact hash mismatch')
            rgb = np.load(stem.with_suffix('.npy'), allow_pickle=False)
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
                raise ValueError('Invalid pixel array')
            mean = rgb.mean((0, 1), dtype=np.float64)
            if not np.allclose(mean, r['mean_rgb'], rtol=0, atol=1e-12):
                raise ValueError('Reported mean mismatch')
            groups[r['transport']].append(mean)
    result = dict(settings=settings, source=source, seeds=sorted(seeds), groups={},
                  reference='Analytic L=E*sum(rho^k, k=0..N), E=0.25, rho=0.5; PT is not reference.',
                  scope='Independent-seed absolute-radiance observations; no automatic pass threshold.')
    for mode, means in groups.items():
        a = np.array(means)
        result['groups'][mode] = dict(mean_rgb=a.mean(0).tolist(),
            error_from_analytic_rgb=(a.mean(0)-settings['expected_rgb']).tolist(),
            standard_error_rgb=(a.std(0, ddof=1)/np.sqrt(len(a))).tolist())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directories)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
