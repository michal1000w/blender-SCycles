#!/usr/bin/env python3
"""Bounded, sequential native-material package integration checks."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--blender',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.blender=a.blender.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2]
env=os.environ.copy()
for key in ['BLENDER_SYSTEM_RESOURCES','DYLD_LIBRARY_PATH','CYCLES_KERNEL_PATH','CYCLES_SHADER_PATH']:env.pop(key,None)
manifest={'binary':str(a.blender),'sha256':hashlib.sha256(a.blender.read_bytes()).hexdigest(),'scope':'PT package integration; not convergence or equal-variance benchmarking','runs':{}}
for name in ['glass','glossy','metallic','principled']:
    output=a.output/name
    command=[str(a.blender),'--background','--factory-startup','--python-exit-code','1','--threads','2','--python',str(root/f'tests/python/cycles_diffraction_{name}_delivery.py'),'--','--output',str(output),'--samples','128']
    with (a.output/f'{name}.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    report=json.loads((output/'report.json').read_text())
    assert report['all_pixels_finite'] and report['samples']==128 and not report['adaptive_sampling'] and not report['denoising']
    manifest['runs'][name]=report
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(name,report['render_seconds'],flush=True)
