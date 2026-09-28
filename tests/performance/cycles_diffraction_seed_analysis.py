#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare independent render means to spectral references without renormalization.

Between-seed standard errors describe sampling variation, not modal, interpolation,
or reference integration error. This report does not certify physical convergence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--renders',type=Path,required=True)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():p.error('Refusing to overwrite a previous comparison')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
report_path=a.renders/'report.json'
r=json.loads(report_path.read_text())
scene_report=json.loads(a.scene.with_suffix('.json').read_text())
if r.get('status')!='completed_requires_reference_comparison':raise ValueError('Seed run incomplete')
if digest(a.scene)!=r['input_scene_sha256'] or digest(a.scene)!=scene_report['blend_sha256']:
    raise ValueError('Input scene mismatch')
transport=scene_report['transport_requested']
if (r['use_bdpt']!=(transport in ('bdpt','bdpt_guided')) or
    r['use_guiding']!=(transport in ('guided','bdpt_guided')) or not r['device_names']):
    raise ValueError('Render transport or device evidence does not match the fixture')
settings=r['sampling_settings']
if (settings['adaptive_sampling'] or settings['denoising'] or settings['time_limit']!=0 or
    settings['use_layer_samples']!='IGNORE' or settings['samples']!=r['samples']):
    raise ValueError('Invalid fixed-sample settings')
seeds=[run['seed'] for run in r['runs']]
if len(seeds)<3 or len(set(seeds))!=len(seeds):raise ValueError('Need independent distinct seeds')
for run in r['runs']:
    if run['sampling_settings']!=settings:raise ValueError('Per-run sampling settings changed')
    if len(run['mean_rgb'])!=3 or not all(math.isfinite(x) for x in run['mean_rgb']):
        raise ValueError('Invalid mean')
    for ext in ('blend','exr'):
        if digest(a.renders/f'seed_{run["seed"]}.{ext}')!=run[ext+'_sha256']:
            raise ValueError('Render artifact changed')
mean=[statistics.fmean(run['mean_rgb'][c] for run in r['runs']) for c in range(3)]
sem=[statistics.stdev(run['mean_rgb'][c] for run in r['runs'])/math.sqrt(len(seeds)) for c in range(3)]
if mean!=r['mean_rgb'] or sem!=r['between_seed_standard_error_rgb']:
    raise ValueError('Incorrect aggregate statistics')
expected_profile=dict(pitch=740,depth=150,duty_cycle=float.fromhex('0x1.a3d70a0000000p-2'),
                      incident_ior=1,ridge_ior=1.5,ridge_extinction=0,groove_ior=1,
                      substrate_ior=1,substrate_extinction=0)
if scene_report['properties']!=expected_profile:raise ValueError('Unexpected reference material')
ref=json.loads(a.reference.read_text())
if ref['color_space']!='linear BT.709':raise ValueError('Unexpected color space')
key=scene_report['illumination'],scene_report['angular_region']
refs={(run['half_orders'],run['subdivision']):run for run in ref['runs']}
def value(run):
    g=next(x['rgb'] for x in run['regions'] if (x['side'],x['region'])==key)
    if not scene_report.get('mix_mirror',False):return g
    mirror=run['neutral_rgb'] if key==('reflection','central') else [0.,0.,0.]
    return [.5*(x+y) for x,y in zip(g,mirror)]
comparisons=[]
for resolution in ((16,8),(64,4),(64,8)):
    target=value(refs[resolution]);error=[x-y for x,y in zip(mean,target)]
    comparisons.append(dict(half_orders=resolution[0],subdivision=resolution[1],reference_rgb=target,
                            error_rgb=error,error_in_standard_errors=[e/s if s else None for e,s in zip(error,sem)]))
out=dict(scope=__doc__,status='completed_statistical_diagnostic',seeds=seeds,samples=r['samples'],
         sampling_settings=settings,use_bdpt=r['use_bdpt'],use_guiding=r['use_guiding'],
         mean_rgb=mean,between_seed_standard_error_rgb=sem,comparisons=comparisons,
         modal_difference_rgb=[x-y for x,y in zip(value(refs[64,8]),value(refs[16,8]))],
         quadrature_difference_rgb=[x-y for x,y in zip(value(refs[64,8]),value(refs[64,4]))],
         report_sha256=digest(report_path),reference_sha256=digest(a.reference),
         scene_sha256=digest(a.scene),binary_sha256=r['binary_sha256'])
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
