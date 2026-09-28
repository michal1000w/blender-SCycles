#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Run the twelve diffraction pipeline fixtures sequentially with a per-job time limit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--blender', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--principled-scene', type=Path,
               default=root/'tests/output/diffraction/principled_delivery_v2/pt/principled.blend')
p.add_argument('--glass-scene', type=Path,
               default=root/'tests/output/diffraction/glass_delivery_pt_1024_v1/glass.blend')
p.add_argument('--timeout', type=float, default=300,
               help='Maximum seconds per job; timeout stops the suite and preserves its log.')
a = p.parse_args()
if a.timeout <= 0:
    p.error('Timeout must be positive')
script = root/'tests/python/cycles_diffraction_pipeline_delivery.py'
for path in [a.blender,a.principled_scene,a.glass_scene,script]:
    if not path.is_file():
        p.error('Missing input: '+str(path))
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
for key in ['BLENDER_SYSTEM_RESOURCES','DYLD_LIBRARY_PATH','CYCLES_KERNEL_PATH','CYCLES_SHADER_PATH']:
    env.pop(key, None)
manifest = {'scope':__doc__, 'runs':[], 'timeout_seconds':a.timeout,
            'input_sha256':{str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [a.blender,a.principled_scene,a.glass_scene,script]}}
for feature in ['aov','motion','dof','volume']:
    for mode in ['pt','bdpt','guided']:
        name = feature+'_'+mode
        template = a.glass_scene if feature == 'volume' else a.principled_scene
        command = [str(a.blender.resolve()),'--background','--factory-startup',
                   '--python-exit-code','1','--threads','2','--python',str(script),'--',
                   '--template',str(template.resolve()),'--output',str(a.output/name),
                   '--feature',feature,'--transport',mode,'--samples','128']
        timed_out = False
        with (a.output/(name+'.log')).open('w') as log:
            try:
                result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                        timeout=a.timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
                timed_out = True
        report = a.output/name/'report.json'
        manifest['runs'].append({'name':name,'returncode':code,'timed_out':timed_out,
                                 'report':json.loads(report.read_text()) if report.exists() else None})
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(name,code,flush=True)
        if code:
            raise SystemExit(code)
