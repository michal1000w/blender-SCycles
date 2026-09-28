#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Catalog saved experiments without promoting them to final validation evidence."""
import argparse
import hashlib
import html
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scenes', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.scenes.resolve()
output = args.output.resolve()
if output.exists():
    parser.error('Refusing to overwrite an existing gallery')
output.mkdir(parents=True)
rows = []
cards = []
for scene in sorted(root.rglob('*.blend')):
    prefix = scene.with_suffix('')
    artifacts = {ext: prefix.with_suffix(ext) for ext in ('.blend', '.json', '.exr', '.png')}
    artifacts = {ext: path for ext, path in artifacts.items() if path.is_file()}
    report = {}
    if '.json' in artifacts:
        try:
            report = json.loads(artifacts['.json'].read_text())
        except (ValueError, UnicodeError):
            report = {'status': 'unreadable_report'}
    if not isinstance(report, dict):
        report = {'status': 'non_object_report'}
    name = str(scene.relative_to(root))
    status = report.get('status', 'no_explicit_status')
    row = dict(scene=name, recorded_status=status,
               scope='Archived experiment; not a current-version certification.',
               artifacts={ext: str(path) for ext, path in artifacts.items()},
               scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest())
    rows.append(row)
    links = ' · '.join(f'<a href="{html.escape(path.as_uri())}">{html.escape(ext[1:].upper())}</a>'
                       for ext, path in artifacts.items())
    image = (f'<img loading="lazy" src="{html.escape(artifacts[".png"].as_uri())}" '
             f'alt="Archived preview of {html.escape(name)}">' if '.png' in artifacts else
             '<p class="missing">No matching PNG preview. Use the EXR or scene link.</p>')
    cards.append(f'<article><h2>{html.escape(name)}</h2>{image}'
                 f'<p>Recorded status: {html.escape(str(status))}</p><p>{links}</p></article>')
(output/'inventory.json').write_text(json.dumps(dict(scope=__doc__, scenes=rows), indent=2)+'\n')
(output/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Diffraction experiment archive</title><style>
body{font:16px system-ui;background:#17191d;color:#eee;margin:32px}a{color:#9cd0ff}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}
article{background:#23272e;padding:16px;border-radius:8px;overflow-wrap:anywhere}
h2{font-size:15px}img{width:100%;height:260px;object-fit:contain;background:#111}
.missing{height:220px;color:#bbb}header{max-width:950px;margin-bottom:28px}
</style><header><h1>Diffraction experiment archive</h1>
<p>Historical saved scenes and matching artifacts. These previews come from different
versions and include failed, provisional and unvalidated experiments. A saved scene,
a rendered image and a physical correctness pass are separate milestones.
This archive is not the final current-version test suite.</p>
<p>Image previews retain their original display transforms. Numerical comparisons must
use linear EXR data and the corresponding report; brightness differences alone do not
establish PT or BDPT correctness.</p></header><main>'''+''.join(cards)+'</main></html>')
print(json.dumps(dict(scenes=len(rows), gallery=str(output/'index.html'))))
