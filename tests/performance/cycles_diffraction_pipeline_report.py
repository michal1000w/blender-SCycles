#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Freeze and present the bounded diffraction pipeline integration suite."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--suite', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
suite = a.suite.resolve()
out = a.output.resolve()
if out.exists():
    p.error('Refusing to overwrite a review')
manifest = json.loads((suite / 'manifest.json').read_text())
expected = {f'{feature}_{mode}' for feature in ['aov','motion','dof','volume']
            for mode in ['pt','bdpt','guided']}
assert len(manifest['runs']) == 12
assert {r['name'] for r in manifest['runs']} == expected
binary_hashes = set()
records = []
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for run in manifest['runs']:
    name = run['name']
    report = run['report']
    assert run['returncode'] == 0 and report['all_pixels_finite']
    assert report['samples'] == 128
    assert not report['adaptive_sampling'] and not report['denoising']
    assert sum(report['mean_rgb']) > 1e-6
    assert 'Too many parameters in function' not in (suite / (name + '.log')).read_text()
    if report['feature'] == 'aov':
        assert abs(report['aov_minmax'][1] - .75) < 1e-5
        assert report['aov_minmax'][0] >= -1e-6
    binary_hashes.add(report['binary_sha256'])
    artifacts = {}
    for ext in ['blend','exr','png']:
        path = suite / name / ('pipeline.' + ext)
        assert path.is_file()
        artifacts[ext] = {'path':os.path.relpath(path, out),'sha256':sha(path)}
    records.append({'name':name, 'report':report, 'artifacts':artifacts})
assert len(binary_hashes) == 1
out.mkdir(parents=True)
review = {'scope':'12 pipeline integration checks. Not full convergence or all-feature certification.',
          'status':'integration_checks_passed_visual_review_separate',
          'suite_manifest_sha256':sha(suite/'manifest.json'),
          'binary_sha256':next(iter(binary_hashes)), 'runs':records}
(out/'review.json').write_text(json.dumps(review,indent=2)+'\n')
parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Diffraction pipeline integration</title><style>body{max-width:1500px;margin:32px auto;padding:0 24px;background:#171a20;color:#e8edf5;font:16px/1.5 system-ui}a{color:#8fcaff}.grid{display:grid;grid-template-columns:repeat(3,minmax(250px,1fr));gap:18px}article{background:#252b34;padding:14px;border-radius:8px}img{width:100%}h1{font-size:32px}small{color:#b9c8dc}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style><h1>Diffraction pipeline integration</h1><p>Apple M5 Metal · 128 fixed samples · adaptive sampling OFF · denoising OFF</p><p>Four independent fixtures across PT, BDPT and guiding. Motion blur, depth of field, participating volume and an exported shader AOV. The AOV has a numerical expected value of 0.75. Other checks verify finite, nonempty output; visual review is separate. No PT brightness normalization and no claim of complete physical convergence.</p><p>The GPU material-function parameter capacity was increased from 37 to 41 to accommodate the new Principled inputs. Earlier failed fixture attempts are preserved in pipeline_delivery_v1 and v2.</p><div class="grid">''']
for row in records:
    r=row['report'];f=row['artifacts'];e=html.escape
    parts.append(f'<article><h2>{e(row["name"])}</h2><img loading="lazy" src="{e(f["png"]["path"])}" alt="{e(row["name"])}"><p>{r["render_seconds"]:.2f} s including preparation; diagnostic timing.</p><p><a href="{e(f["blend"]["path"])}">Saved scene</a> · <a href="{e(f["exr"]["path"])}">Linear EXR</a></p></article>')
parts.append('</div><p>General cross-object coherence and the other material-model gaps remain unimplemented. This suite does not certify those features.</p></html>')
(out/'index.html').write_text(''.join(parts))
print(out/'index.html')
