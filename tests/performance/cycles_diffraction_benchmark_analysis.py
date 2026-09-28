#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare completed, matching repeat-render benchmark configurations."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--diffraction',type=Path,required=True)
p.add_argument('--mirror',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
d,m=[json.loads(x.read_text()) for x in (a.diffraction,a.mirror)]
for path,r in ((a.diffraction,d),(a.mirror,m)):
    if r['status']!='completed':raise ValueError('Benchmark incomplete')
    if r.get('benchmark_schema')!=2:
        raise ValueError('Legacy report lacks explicit effective sampling settings; retain its archived analysis')
    settings=r['sampling_settings']
    if settings['adaptive_sampling'] is not False or settings['denoising'] is not False:
        raise ValueError('Benchmark requires adaptive sampling and denoising disabled')
    if settings.get('time_limit')!=0 or settings.get('use_layer_samples')!='IGNORE':
        raise ValueError('Benchmark requires no time limit or layer sample overrides')
    if settings['samples']!=r['samples'] or any(x['sampling_settings']!=settings for x in r['runs']):
        raise ValueError('Sampling settings changed during benchmark')
    if hashlib.sha256(path.with_suffix('.blend').read_bytes()).hexdigest()!=r['benchmark_scene_sha256']:
        raise ValueError('Benchmark scene changed')
    measured=[x for x in r['runs'] if not x['warmup']]
    if len(measured)<3 or any(x['seconds']<=0 for x in measured):raise ValueError('Invalid measurements')
    if statistics.median(x['seconds'] for x in measured)!=r['median_seconds']:
        raise ValueError('Incorrect median')
for key in ('sampling_settings','samples','resolution','devices','blender_binary_sha256','script_sha256','input_scene_sha256'):
    if d[key]!=m[key]:raise ValueError('Configuration mismatch: '+key)
if d['material']!='diffraction' or m['material']!='mirror':raise ValueError('Unexpected baseline')
report=dict(diffraction_median_seconds=d['median_seconds'],mirror_median_seconds=m['median_seconds'],
            sampling_settings=d['sampling_settings'],
            ratio=d['median_seconds']/m['median_seconds'],
            added_seconds=d['median_seconds']-m['median_seconds'],samples=d['samples'],resolution=d['resolution'],
            scope='Host-observed repeat-render calls after warmup. Ideal mirror is a one-bounce workload baseline, not an equivalent grating or a pre-change regression baseline.',
            reports_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.diffraction,a.mirror)})
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
