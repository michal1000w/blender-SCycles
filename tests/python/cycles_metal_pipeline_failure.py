#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a missing generic Metal entry point fails the render instead of waiting forever.

Run with system Python. Only an isolated copy of kernel source is modified.
The normal source-override mechanism must compile that copy; a successful render
would indicate that the faulty source was ignored and is a test failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', type=Path,
                        default=root/'install/Blender.app/Contents/MacOS/Blender')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not any(output.iterdir()), 'Preserve previous test evidence; use an empty directory'
    blender = args.blender.resolve()
    sources = list((blender.parents[1]/'Resources').glob('*/scripts/addons_core/cycles/source'))
    assert len(sources) == 1, sources
    source = output/'source'
    shutil.copytree(sources[0], source)
    header = source/'kernel/device/gpu/kernel.h'
    original = header.read_bytes()
    entry = b'ccl_gpu_kernel_signature(guiding_begin_update, const uint work_size)'
    assert original.count(entry) == 1
    patched = original.replace(entry, entry.replace(b'guiding_begin_update',
                                                   b'guiding_begin_update_fault'))
    header.write_bytes(patched)
    command = [str(blender), '--background', '--factory-startup', '--python-exit-code', '1',
               '--debug-cycles', '--log-level', 'debug', '--python',
               str(root/'tests/python/cycles_metal_guiding_scene.py'), '--', '--guiding',
               '--samples', '4', '--resolution', '32', '--output', str(output/'render')]
    manifest = {'command': command, 'original_sha256': hashlib.sha256(original).hexdigest(),
                'patched_sha256': hashlib.sha256(patched).hexdigest(),
                'source_override': str(source), 'expected_missing_kernel': 'guiding_begin_update'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    env = os.environ.copy()
    env['CYCLES_KERNEL_PATH'] = str(source)
    env['CYCLES_METAL_DEBUG'] = '1'
    with (output/'render.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=600)
    log = (output/'render.log').read_text()
    expected = 'Failed to load required Metal kernel guiding_begin_update'
    report = {'returncode': result.returncode, 'expected_error_present': expected in log,
              'source_unchanged': header.read_bytes() == patched}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    assert result.returncode != 0 and expected in log and report['source_unchanged'], report
    print('METAL_FAILED_GENERIC_PIPELINE_PASSED', flush=True)


if __name__ == '__main__':
    main()
