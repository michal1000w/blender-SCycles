#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Sequential isolated benchmarks of integrated Fast diffraction and a control.

Run only when no other GPU work is active. Timings measure warmed render calls,
not equivalent-image convergence. Sources are frozen and checked around each job.
The control either removes relief or replaces optical tables with the saved
constant indices. Neither control is an ordinary non-spectral material baseline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--samples',type=int,default=1024)
p.add_argument('--baseline',choices=['flat-grating','constant-index'],default='flat-grating')
p.add_argument('--transport',choices=['pt','bdpt','guided','bdpt_guided'],action='append')
a=p.parse_args()
if a.output.exists():p.error('Use a new output directory')
if a.samples<1:p.error('Samples must be positive')
scene=a.scene.resolve();out=a.output.resolve()
binary=root/'build/macos_arm64_Release/bin/Blender.app/Contents/MacOS/Blender'
script=root/'tests/python/cycles_diffraction_tensor_benchmark.py'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
files=[binary,script,Path(__file__).resolve(),scene]
files += [p for p in (root/'intern/cycles').rglob('*') if p.is_file()]
provenance={str(p):digest(p) for p in files}
def verify():
    for name,value in provenance.items():
        if digest(Path(name))!=value:raise RuntimeError('Frozen input changed: '+name)
out.mkdir(parents=True)
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
env=os.environ.copy()
env.update(BLENDER_USER_RESOURCES='/tmp/cycles-diffraction-blender-user',
 DYLD_LIBRARY_PATH=str(root/'install/Blender.app/Contents/Resources/lib'),
 BLENDER_SYSTEM_RESOURCES=str(root/'install/Blender.app/Contents/Resources/5.3'),
 CYCLES_KERNEL_PATH=str(root/'intern/cycles'))
transports=a.transport or ['pt','bdpt','guided','bdpt_guided']
manifest=dict(scope=__doc__,status='running',baseline=a.baseline,transports=transports,runs=[])
def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
save()
for index,transport in enumerate(transports):
    # Alternate treatment order to reduce a consistent warm-up/thermal ordering bias.
    materials=('diffraction',a.baseline) if index%2==0 else (a.baseline,'diffraction')
    for material in materials:
        verify();name=transport+'_'+material
        command=[str(binary),'--background','--factory-startup','--python-exit-code','1',
          '--threads','2','--python',str(script),'--','--scene',str(scene),
          '--output',str(out/(name+'.json')),'--material',material,
          '--transport',transport,'--samples',str(a.samples),'--runs','3']
        print('Starting '+name,flush=True)
        with (out/(name+'.log')).open('w') as log:
            process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        verify()
        manifest['runs'].append(dict(name=name,command=command,returncode=process.returncode))
        save()
        if process.returncode:
            manifest['status']='stopped_on_failure';save();raise SystemExit(process.returncode)
rows=[];identity=None
for transport in transports:
    pair={}
    for material in ('diffraction',a.baseline):
        name=transport+'_'+material;path=out/(name+'.json');r=json.loads(path.read_text())
        assert r['status']=='completed' and r['material']==material and r['transport']==transport
        s=r['sampling_settings']
        assert s['samples']==a.samples and not s['adaptive_sampling'] and not s['denoising']
        assert s['time_limit']==0 and s['use_layer_samples']=='IGNORE'
        assert s['bidirectional']==(transport in ('bdpt','bdpt_guided'))
        assert s['guiding']==(transport in ('guided','bdpt_guided'))
        current={k:r[k] for k in ('blender_binary_sha256','script_sha256','input_scene_sha256','resolution','devices')}
        if identity is None:identity=current
        assert identity==current
        assert digest(out/(name+'.blend'))==r['benchmark_scene_sha256']
        runs=r['runs'];assert len(runs)==4
        assert [x['seed'] for x in runs]==[11,12,13,14]
        assert [x['warmup'] for x in runs]==[True,False,False,False]
        assert all(x['sampling_settings']==s and 0<x['seconds']<float('inf') for x in runs)
        measured=[x['seconds'] for x in runs[1:]]
        assert statistics.median(measured)==r['median_seconds']
        disabled=r['disabled_diffraction_nodes']
        cleared=r.get('cleared_optical_tables',[])
        assert (bool(disabled) if material=='flat-grating' else not disabled)
        assert (bool(cleared) if material=='constant-index' else not cleared)
        if disabled:assert all(x['original_depth_nm']>0 for x in disabled)
        pair[material]=dict(median_seconds=r['median_seconds'],measured_seconds=measured,
                            warmup_seconds=runs[0]['seconds'],report_sha256=digest(path))
    rows.append(dict(transport=transport,**pair,difference_percent=100*(
      pair['diffraction']['median_seconds']/pair[a.baseline]['median_seconds']-1)))
manifest.update(status='completed_artifact_checked',rows=rows,identity=identity)
save();print(json.dumps(rows,indent=2))
