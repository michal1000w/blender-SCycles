#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded, sequential acceptance of the newly integrated features."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--job', action='append', help='Run only a named job; repeat to select several.')
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
a.binary = a.binary.resolve()
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
for key in ['BLENDER_SYSTEM_RESOURCES', 'DYLD_LIBRARY_PATH', 'CYCLES_KERNEL_PATH', 'CYCLES_SHADER_PATH']:
    env.pop(key, None)
jobs = []
# Compile/preparation costs accumulate when feature masks change between cases.
# Give every case its own bounded process and retain its outcome.
for transport in ['pt', 'bdpt', 'guided', 'bdpt_guided']:
    for case in ['phase_0', 'phase_pi', 'incoherent', 'partial', 'red', 'occluded',
                 'three_sources', 'unit_scale', 'off']:
        options = ['--transport', transport, '--samples', '128', '--case', case]
        if transport == 'bdpt_guided':
            options += ['--guiding-training-samples', '32']
        jobs.append(('coherence_' + transport + '_' + case,
                     'cycles_coherent_direct_delivery.py', options))
for transport in ['pt', 'bdpt', 'guided', 'osl']:
    opts = ['--samples', '128', '--resolution', '480']
    opts += ['--osl'] if transport == 'osl' else ['--transport', transport]
    jobs.append(('thin_wall_' + transport, 'cycles_diffraction_thin_wall_delivery.py', opts))
for label, opts in [('flat', ['--depth', '0']), ('uncovered', ['--coverage', '0'])]:
    jobs.append(('thin_wall_' + label, 'cycles_diffraction_thin_wall_delivery.py',
                 ['--samples', '128', '--resolution', '480'] + opts))
for transport in ['pt', 'bdpt', 'guided', 'osl']:
    opts = ['--ashikhmin', '--samples', '128', '--resolution', '480']
    opts += ['--osl'] if transport == 'osl' else ['--transport', transport]
    jobs.append(('ashikhmin_' + transport, 'cycles_diffraction_glossy_delivery.py', opts))
all_job_names = [job[0] for job in jobs]
if a.job:
    unknown = set(a.job) - set(all_job_names)
    if unknown:
        p.error('Unknown jobs: ' + ', '.join(sorted(unknown)))
    jobs = [job for job in jobs if job[0] in a.job]
manifest = {'binary': str(a.binary), 'sha256': hashlib.sha256(a.binary.read_bytes()).hexdigest(),
            'available_jobs': all_job_names, 'scheduled_jobs': [job[0] for job in jobs],
            'partial_selection': bool(a.job),
            'adaptive_sampling': False, 'denoising': False,
            'scope': 'Coherence has analytic absolute-radiance gates; material scenes are finite-output transport and appearance checks, not convergence proof.',
            'runs': {}}
for name, script, options in jobs:
    command = [str(a.binary), '--background', '--factory-startup', '--python-exit-code', '1',
               '--threads', '2', '--python', str(root / 'tests/python' / script), '--',
               '--output', str(a.output / name), *options]
    entry = {'command': command, 'status': 'running',
             'script_sha256': hashlib.sha256((root / 'tests/python' / script).read_bytes()).hexdigest()}
    manifest['runs'][name] = entry
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    start = time.monotonic()
    with (a.output / (name + '.log')).open('w') as log:
        try:
            run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=360)
            entry['exit_code'] = run.returncode
            entry['status'] = 'passed' if run.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            entry['timeout_seconds'] = 360
            entry['status'] = 'timeout'
    entry['process_seconds'] = time.monotonic() - start
    manifest['runs'][name] = entry
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(name, entry.get('exit_code', 'TIMEOUT'), flush=True)
    if entry.get('exit_code') != 0:
        raise SystemExit(1)
