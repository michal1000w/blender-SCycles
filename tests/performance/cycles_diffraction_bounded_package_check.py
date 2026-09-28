#!/usr/bin/env python3
"""Sequential fixed-sample checks of bounded high-order native diffraction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary',type=Path,default=root/'build/diffraction_delivery_20260927_bounded/Blender.app/Contents/MacOS/Blender')
parser.add_argument('--output',type=Path,default=root/'tests/output/diffraction/bounded_package_v1')
parser.add_argument('--cases',nargs='+',choices=['after_100um','pt_1mm','bdpt_1mm','guided_1mm','osl_1mm'])
args=parser.parse_args()
binary=args.binary.resolve()
output=args.output.resolve()
output.mkdir(exist_ok=False)
env=os.environ.copy()
for key in ['BLENDER_SYSTEM_RESOURCES','DYLD_LIBRARY_PATH','CYCLES_KERNEL_PATH','CYCLES_SHADER_PATH']:
    env.pop(key,None)
manifest={'binary':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
          'scope':'Bounded high-order integration checks; not an equal-error method comparison','runs':{}}
for name,transport,pitch,osl in [('after_100um','pt',100000,False),('pt_1mm','pt',1000000,False),
                                ('bdpt_1mm','bdpt',1000000,False),('guided_1mm','guided',1000000,False),
                                ('osl_1mm','pt',1000000,True)]:
    if args.cases and name not in args.cases:continue
    command=[str(binary),'--background','--factory-startup','--python-exit-code','1','--threads','2',
             '--python',str(root/'tests/python/cycles_diffraction_large_pitch_delivery.py'),
             '--','--output',str(output/name),'--pitch',str(pitch),'--transport',transport,
             '--samples','32','--resolution','128']
    if osl:command.append('--osl')
    with (output/f'{name}.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240,check=True)
    report=json.loads((output/name/'report.json').read_text())
    assert report['all_pixels_finite'] and not report['adaptive_sampling'] and not report['denoising']
    manifest['runs'][name]=report
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(name,report['render_times_seconds'],flush=True)
