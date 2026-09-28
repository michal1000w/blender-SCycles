#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare saved angular-control means with direct spectral Maxwell references.

Single-seed differences are diagnostics, not statistical acceptance tests.
Signed linear RGB values are retained without clipping or rescaling.
"""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--suite',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
reference=json.loads(a.reference.read_text())
if reference['color_space']!='linear BT.709':raise ValueError('Unsupported reference color space')
runs={(r['half_orders'],r['subdivision']):r for r in reference['runs']}
def regions(run):return {(r['side'],r['region']):r['rgb'] for r in run['regions']}
def expected_rgb(run, key, mix_mirror):
    grating=regions(run)[key]
    if not mix_mirror:return grating
    # At normal incidence the mirror reflects into the central region only.
    mirror=run['neutral_rgb'] if key==('reflection','central') else [0.,0.,0.]
    return [.5*(g+m) for g,m in zip(grating,mirror)]
manifest_path=a.suite/'manifest.json'
manifest=json.loads(manifest_path.read_text())
rows=[]
for run in manifest['runs']:
    report=run.get('scene_report',{})
    if report.get('angular_region','all')=='all':continue
    if run['returncode'] or report['status']!='rendered_unvalidated':
        raise ValueError('Angular render did not complete')
    if report['transport_requested'] not in ('pt','bdpt','guided','bdpt_guided') or report['device_requested']!='METAL':
        raise ValueError('Unexpected estimator or device request')
    if report['case']!='relief_dielectric':raise ValueError('Unexpected material')
    expected=dict(pitch=740,depth=150,duty_cycle=float.fromhex('0x1.a3d70a0000000p-2'),
                  incident_ior=1,ridge_ior=1.5,ridge_extinction=0,groove_ior=1,
                  substrate_ior=1,substrate_extinction=0)
    if report['properties']!=expected:raise ValueError('Material differs from reference')
    key=report['illumination'],report['angular_region']
    mixed=report.get('mix_mirror',False)
    prefix=a.suite/run['name']/('relief_dielectric_'+key[0]+'_'+key[1]+('_mirror_mix' if mixed else ''))
    for suffix in ('.blend','.exr'):
        with prefix.with_suffix(suffix).open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=report[suffix[1:]+'_sha256']:raise ValueError('Artifact hash mismatch')
    measured=report['raw_mean_rgb']
    fine=expected_rgb(runs[64,8],key,mixed)
    coarse=expected_rgb(runs[64,4],key,mixed)
    low=expected_rgb(runs[16,8],key,mixed)
    rows.append(dict(name=run['name'],transport=report['transport_requested'],mix_mirror=mixed,
      measured_rgb=measured,reference_rgb=fine,
      error_rgb=[x-y for x,y in zip(measured,fine)],
      quadrature_difference_rgb=[x-y for x,y in zip(fine,coarse)],
      modal_difference_rgb=[x-y for x,y in zip(fine,low)],
      status='single_seed_comparison_requires_convergence'))
report=dict(scope=__doc__,reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            reference_neutral_rgb=runs[64,8]['neutral_rgb'],comparisons=rows,
            angular_scenes_compared=len(rows),angular_scenes_expected=6)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
