#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Validate the matched fast-disc benchmark artifacts and summarize all transports."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--pt-root', type=Path, required=True)
p.add_argument('--transport-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    p.error('Refusing to overwrite an analysis')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

kernel_hashes = json.loads((a.pt_root/'kernel_hashes.json').read_text())
if kernel_hashes != json.loads((a.transport_root/'kernel_hashes.json').read_text()):
    raise ValueError('Benchmarks used different kernel source manifests')
if not kernel_hashes:
    raise ValueError('Missing kernel provenance')
for directory, count in ((a.pt_root, 2), (a.transport_root, 6)):
    manifest = json.loads((directory/'manifest.json').read_text())
    if len(manifest) != count or any(x['returncode'] != 0 for x in manifest):
        raise ValueError('Benchmark batch did not complete successfully')

rows = []
identity = None
for transport in ('pt', 'bdpt', 'guided', 'bdpt_guided'):
    reports = {}
    for material in ('diffraction', 'undiffracted'):
        directory = a.pt_root if transport == 'pt' else a.transport_root
        name = material if transport == 'pt' else transport+'_'+material
        path = directory/(name+'.json')
        r = json.loads(path.read_text())
        if r['status'] != 'completed' or r['material'] != material or r['transport'] != transport:
            raise ValueError('Wrong benchmark identity or incomplete result: '+name)
        settings = r['sampling_settings']
        if (settings['samples'] != 1024 or settings['adaptive_sampling'] or
            settings['denoising'] or settings['time_limit'] != 0 or
            settings['use_layer_samples'] != 'IGNORE' or
            settings['bidirectional'] != (transport in ('bdpt', 'bdpt_guided')) or
            settings['guiding'] != (transport in ('guided', 'bdpt_guided'))):
            raise ValueError('Unexpected sampling or transport settings: '+name)
        current_identity = {k:r[k] for k in ('blender_binary_sha256', 'script_sha256',
                                             'input_scene_sha256', 'resolution', 'devices')}
        current_identity['resolution_percentage'] = settings['resolution_percentage']
        if identity is None:
            identity = current_identity
        elif identity != current_identity:
            raise ValueError('Unmatched benchmark inputs: '+name)
        if digest(directory/(name+'.blend')) != r['benchmark_scene_sha256']:
            raise ValueError('Benchmark scene changed: '+name)
        runs = r['runs']
        if len(runs) != 4 or [x['warmup'] for x in runs] != [True, False, False, False]:
            raise ValueError('Expected one warm-up and three measured renders')
        if [x['seed'] for x in runs] != [11, 12, 13, 14]:
            raise ValueError('Unmatched random seeds')
        if any(x['sampling_settings'] != settings or not (0 < x['seconds'] < float('inf'))
               for x in runs):
            raise ValueError('Per-run settings or timing invalid')
        measured = [x['seconds'] for x in runs[1:]]
        if (statistics.median(measured) != r['median_seconds'] or
            min(measured) != r['minimum_seconds'] or max(measured) != r['maximum_seconds']):
            raise ValueError('Incorrect timing aggregate')
        disabled = r['disabled_diffraction_nodes']
        if material == 'diffraction' and disabled:
            raise ValueError('Diffraction accidentally disabled')
        if material == 'undiffracted' and (len(disabled) != 2 or
            {x['material'] for x in disabled} != {'CD / 1600 nm', 'DVD / 740 nm'} or
            any(x['original_weight'] != 1 for x in disabled)):
            raise ValueError('Unexpected matched control changes')
        reports[material] = dict(median_seconds=r['median_seconds'],
                                 measured_seconds=measured,
                                 warmup_seconds=runs[0]['seconds'],
                                 report_sha256=digest(path))
    rows.append(dict(transport=transport, **reports,
                     overhead_percent=100*(reports['diffraction']['median_seconds']/
                                           reports['undiffracted']['median_seconds']-1)))

result = dict(status='completed_artifact_checked_workload_comparison', identity=identity,
              kernel_manifest_sha256=digest(a.pt_root/'kernel_hashes.json'), rows=rows,
              scope='Elapsed warmed persistent-data render calls. Fixed 1,024 samples; adaptive '
                    'sampling and denoising off. Disabling diffraction changes paths, so these '
                    'are workload comparisons, not equal-image or equal-noise benchmarks. '
                    'Three measured runs do not establish statistical significance. This '
                    'scalar reflection result does not certify transmission or coherence.')
a.output.mkdir(parents=True)
(a.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
lines = ['# Fast disc workload comparison', '', result['scope'], '',
         '| Transport | Diffraction (s) | Disabled (s) | Difference |',
         '|---|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['transport']} | {row['diffraction']['median_seconds']:.3f} | "
                 f"{row['undiffracted']['median_seconds']:.3f} | {row['overhead_percent']:+.2f}% |")
(a.output/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result, indent=2))
