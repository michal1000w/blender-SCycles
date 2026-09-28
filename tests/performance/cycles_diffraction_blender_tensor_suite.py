#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Render saved physical-node transport and angular controls on Metal."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=root/'tests/output/diffraction/tensor_blender_controls')
parser.add_argument('--provenance',type=Path,default=root/'tests/output/diffraction/tensor_dielectric_material_provenance.json')
parser.add_argument('--mix-mirror',action='store_true')
parser.add_argument('--angular-only',action='store_true')
parser.add_argument('--angular-transport',choices=('pt','bdpt','guided','bdpt_guided'),default='pt')
parser.add_argument('--samples',type=int,default=1024)
args=parser.parse_args()
if args.samples<=0:parser.error('Samples must be positive')
output=args.output.resolve()
output.mkdir(parents=True,exist_ok=True)
if (output/'manifest.json').exists():raise RuntimeError('Refusing to overwrite prior manifest')
provenance=json.loads(args.provenance.read_text())
env=os.environ.copy()
env.update(BLENDER_USER_RESOURCES='/tmp/cycles-diffraction-blender-user',
           DYLD_LIBRARY_PATH=str(root/'install/Blender.app/Contents/Resources/lib'),
           BLENDER_SYSTEM_RESOURCES=str(root/'install/Blender.app/Contents/Resources/5.3'),
           CYCLES_KERNEL_PATH=str(root/'intern/cycles'))
binary=root/'build/macos_arm64_Release/bin/Blender.app/Contents/MacOS/Blender'
jobs=[] if args.angular_only else [(transport,'both','all') for transport in ('bdpt','guided','bdpt_guided')]
jobs += [(args.angular_transport,side,region) for side in ('reflection','transmission')
         for region in ('central','middle','outer')]
manifest={'scope':'Saved Blender physical-node controls; angular renders require independent radiometric validation.',
          'source_sha256':provenance,'runs':[]}
for transport,side,region in jobs:
    for relative,expected in provenance.items():
        with (root/relative).open('rb') as source:
            if hashlib.file_digest(source,'sha256').hexdigest()!=expected:
                raise RuntimeError(f'Source changed during suite: {relative}')
    name=f'{transport}_{side}_{region}'
    directory=output/name
    if directory.exists():raise RuntimeError(f'Refusing to overwrite prior experiment: {directory}')
    print(f'Starting {name}',flush=True)
    command=[str(binary),'--background','--factory-startup','--python-exit-code','1','--threads','2',
             '--python',str(root/'tests/python/cycles_physical_diffraction_scene.py'),'--',
             '--output',str(directory),'--case','relief_dielectric','--illumination',side,
             '--angular-region',region,'--device','METAL','--transport',transport,'--samples',str(args.samples),'--render']
    if args.mix_mirror:command.append('--mix-mirror')
    start=time.monotonic()
    with (output/(name+'.log')).open('w') as log:
        run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    record={'name':name,'returncode':run.returncode,'total_process_seconds':time.monotonic()-start,
            'timing_scope':'Includes cache construction, device initialization, kernel compilation and rendering.'}
    report=directory/('relief_dielectric_'+side+('' if region=='all' else '_'+region)+
                      ('_mirror_mix' if args.mix_mirror else '')+'.json')
    if report.exists():record['scene_report']=json.loads(report.read_text())
    manifest['runs'].append(record)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if run.returncode:raise SystemExit(run.returncode)
