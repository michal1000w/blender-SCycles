#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Run current physical-node scenes sequentially against a frozen source manifest.

Timings include preparation and are diagnostics, not isolated benchmarks.
Rendering success does not certify physical accuracy or unsupported features.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--provenance', type=Path, required=True)
p.add_argument('--samples', type=int, default=1024)
p.add_argument('--resolution', type=int, default=960)
p.add_argument('--quality', choices=('FAST', 'REALISTIC'), default='REALISTIC')
p.add_argument('--optical-constants', type=Path,
               help='Embedded conductor table for appearance fixtures; analytic controls retain their reference indices.')
p.add_argument('--group', choices=('controls', 'appearance', 'all'), default='all')
p.add_argument('--transport', choices=('pt', 'bdpt', 'guided', 'bdpt_guided'), action='append')
a = p.parse_args()
if a.samples < 1 or a.resolution < 16:
    p.error('Invalid sample count or resolution')
out = a.output.resolve()
if out.exists():
    p.error('Use a new output directory; previous experiments are immutable')
provenance = json.loads(a.provenance.read_text())
if a.optical_constants:
    a.optical_constants = a.optical_constants.resolve()
    table_name = str(a.optical_constants.relative_to(root))
    if table_name not in provenance:
        p.error('The optical-constant table must be included in the frozen provenance')
def verify_sources():
    for name, expected in provenance.items():
        with (root/name).open('rb') as source:
            if hashlib.file_digest(source, 'sha256').hexdigest() != expected:
                raise RuntimeError('Frozen source changed: '+name)
verify_sources()
out.mkdir(parents=True)
env = os.environ.copy()
env.update(BLENDER_USER_RESOURCES='/tmp/cycles-diffraction-blender-user',
           DYLD_LIBRARY_PATH=str(root/'install/Blender.app/Contents/Resources/lib'),
           BLENDER_SYSTEM_RESOURCES=str(root/'install/Blender.app/Contents/Resources/5.3'),
           CYCLES_KERNEL_PATH=str(root/'intern/cycles'))
binary = root/'build/macos_arm64_Release/bin/Blender.app/Contents/MacOS/Blender'
jobs = []
for transport in a.transport or ('pt', 'bdpt', 'guided', 'bdpt_guided'):
    if a.group in ('controls', 'all'):
        for case in ('flat_metal', 'relief_metal', 'relief_dielectric'):
            jobs.append((f'{transport}_{case}_furnace', 'cycles_physical_diffraction_scene.py',
                         ['--case', case, '--illumination', 'both', '--device', 'METAL',
                          '--transport', transport]))
        for mixed in (False, True):
            for side in ('reflection', 'transmission'):
                for region in ('central', 'middle', 'outer'):
                    options = ['--case', 'relief_dielectric', '--illumination', side,
                               '--angular-region', region, '--device', 'METAL', '--transport', transport]
                    if mixed:
                        options.append('--mix-mirror')
                    jobs.append((f'{transport}_{side}_{region}'+('_mirror_mix' if mixed else ''),
                                 'cycles_physical_diffraction_scene.py', options))
    if a.group in ('appearance', 'all'):
        for case in ('discs', 'covered', 'indirect'):
            jobs.append((f'{transport}_{case}', 'cycles_physical_diffraction_discs.py',
                         ['--case', case, '--device', 'GPU', '--transport', transport,
                          '--resolution', str(a.resolution), '--lighting', 'broad',
                          '--indirect-source-radius-mm', '50']))
            if a.optical_constants:
                jobs[-1][2].extend(['--optical-constants', str(a.optical_constants)])
manifest = dict(scope=__doc__, source_sha256=provenance, planned_jobs=[j[0] for j in jobs],
                quality=a.quality,
                optical_constants=str(a.optical_constants) if a.optical_constants else None,
                runs=[], status='running',
                limitations=['Cross-object coherent transport is not implemented.',
                             ('Fast scalar efficiency is approximate; conservation does not imply Maxwell agreement.'
                              if a.quality == 'FAST' else
                              'Current automatic material caches use fixed modal truncation.'),
                             'Completion of this runner is not a physical convergence verdict.'])
def save():
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
save()
for name, script, options in jobs:
    verify_sources()
    command = [str(binary), '--background', '--factory-startup', '--python-exit-code', '1',
               '--threads', '2', '--python', str(root/'tests/python'/script),
               '--python', str(root/'tests/python/cycles_diffraction_suite_preview.py'), '--',
               '--output', str(out/name), '--samples', str(a.samples), '--quality', a.quality, '--render']+options
    print('Starting '+name, flush=True)
    start = time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    verify_sources()
    reports = {str(path.relative_to(out)): json.loads(path.read_text())
               for path in (out/name).glob('*.json')}
    manifest['runs'].append(dict(name=name, returncode=process.returncode, command=command,
                                 total_process_seconds=time.monotonic()-start, reports=reports))
    save()
    # A failed pipeline must be fixed before spending hours on dependent scenes.
    if process.returncode:
        manifest['status'] = 'stopped_on_failure'
        save()
        raise SystemExit(process.returncode)
manifest['status'] = 'renders_completed_pending_analysis'
save()
