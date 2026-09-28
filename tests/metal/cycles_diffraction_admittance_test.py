#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Test direct-admittance matching on Metal using frozen float CPU fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--expected', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
hashes = {}
def digest(path):
    hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
def expand(path, ancestors=()):
    path = path.resolve()
    if path in ancestors:
        raise ValueError('Include cycle')
    digest(path)
    source = path.read_text()
    def include(match):
        target = path.parent / match[1]
        if not target.is_file():
            target = root / 'intern/cycles' / match[1]
        return expand(target, (*ancestors, path)) if target.is_file() else match[0]
    source = re.sub(r'^[ \t]*#[ \t]*include[ \t]+"([^"]+)"[^\n]*$', include, source, flags=re.M)
    source = re.sub(r'^\s*#\s*pragma\s+once\s*$', '', source, flags=re.M)
    guard = 'ADMITTANCE_TEST_' + hashlib.sha256(str(path).encode()).hexdigest()
    return f'#ifndef {guard}\n#define {guard}\n{source}\n#endif\n'
a.output.parent.mkdir(parents=True, exist_ok=True)
host = root / 'tests/metal/cycles_diffraction_anchor_match.mm'
for path in (Path(__file__),host,a.input,a.expected): digest(path)
with tempfile.TemporaryDirectory(prefix='diffraction-admittance-') as temporary:
    temporary = Path(temporary)
    source = temporary / 'kernel.metal'
    source.write_text(expand(root/'tests/metal/cycles_diffraction_admittance_match.metal'))
    executable = temporary / 'test'
    subprocess.run(['clang++','-O2','-std=c++20','-fobjc-arc','-framework','Foundation',
                    '-framework','Metal',str(host),'-o',str(executable)],check=True)
    run = subprocess.run([str(executable),str(source),str(a.input),str(a.expected),
                          str(a.output.with_suffix('.bin'))],capture_output=True,text=True)
    report = dict(returncode=run.returncode,source_sha256=hashes,stdout=run.stdout,stderr=run.stderr)
    if run.returncode == 0:
        report.update(json.loads(run.stdout))
        report['scope'] = 'Direct-admittance matching only; CPU interpolation and exterior momentum; excludes rendering'
        report['passed'] = report['maximum_component_error'] <= 2e-6
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
    raise SystemExit(run.returncode or (0 if report['passed'] else 1))
