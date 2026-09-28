#!/usr/bin/env python3
"""Bounded sequential checks of the standalone dispersion delivery package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--feature', choices=['dispersion', 'film', 'generalized_film'], default='dispersion')
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
binary = root / f'build/diffraction_delivery_20260927_{args.feature}/Blender.app/Contents/MacOS/Blender'
output = root / f'tests/output/diffraction/principled_{args.feature}_delivery_v1'
output.mkdir(exist_ok=False)
env = os.environ.copy()
for key in ['BLENDER_SYSTEM_RESOURCES', 'DYLD_LIBRARY_PATH', 'CYCLES_KERNEL_PATH', 'CYCLES_SHADER_PATH']:
    env.pop(key, None)
manifest = {'binary': str(binary), 'sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'scope': 'Integration checks, not equal-error transport benchmarks', 'runs': {}}
for name, samples in [('pt', 256), ('bdpt', 128), ('guided', 128), ('osl', 32)]:
    command = [str(binary), '--background', '--factory-startup', '--python-exit-code', '1',
               '--threads', '2', '--python',
               str(root / f'tests/python/cycles_diffraction_principled_{args.feature}_delivery.py'),
               '--', '--output', str(output / name), '--samples', str(samples),
               '--transport', 'pt' if name == 'osl' else name]
    if name == 'osl':
        command.append('--osl')
    with (output / f'{name}.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=240, check=True)
    report = json.loads((output / name / 'report.json').read_text())
    assert report['all_pixels_finite'] and not report['adaptive_sampling'] and not report['denoising']
    manifest['runs'][name] = report
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(name, report['render_seconds'], flush=True)
