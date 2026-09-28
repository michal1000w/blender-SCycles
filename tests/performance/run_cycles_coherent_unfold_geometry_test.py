#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU strict/fast checks for one sphere block and separate planar mirrors."""
import hashlib, json, subprocess, tempfile
from pathlib import Path
root = Path(__file__).resolve().parents[2]
source = root / 'tests/performance/cycles_coherent_unfold_geometry_test.cpp'
files = [source, Path(__file__), *(root / 'intern/cycles/kernel/light' / name for name in (
    'coherent_unfold_geometry.h', 'coherent_curved_geometry.h',
    'coherent_sphere_transmit_geometry.h', 'coherent_geometry.h'))]
report = {'scope': 'one sphere R or contiguous TT with planar reflection prefixes/suffixes',
          'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
          'cases': []}
with tempfile.TemporaryDirectory(prefix='cycles-unfold-') as temp:
    for mode, flags in [('strict', []), ('fast', ['-ffast-math'])]:
        exe = Path(temp) / mode
        command = ['c++', '-std=c++20', '-O2', *flags,
                   '-DCCL_NAMESPACE_BEGIN=namespace ccl {', '-DCCL_NAMESPACE_END=}',
                   '-Iintern/cycles', '-Iintern/cycles/kernel', str(source), '-o', str(exe)]
        build = subprocess.run(command, cwd=root, text=True, capture_output=True)
        run = subprocess.run([str(exe)], text=True, capture_output=True) if build.returncode == 0 else None
        report['cases'].append({'mode': mode, 'command': command, 'build_exit': build.returncode,
                                'build_stderr': build.stderr, 'run_exit': run.returncode if run else None,
                                'stdout': run.stdout if run else ''})
        print(mode, run.stdout.strip() if run else build.stderr)
output = root / 'build/tests/performance/coherent_unfold'
output.mkdir(parents=True, exist_ok=True)
(output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
raise SystemExit(0 if all(c['build_exit'] == 0 and c['run_exit'] == 0 for c in report['cases']) else 1)
