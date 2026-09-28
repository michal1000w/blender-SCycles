#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Audit frozen-suite artifacts and compare angular means without PT normalization."""
import argparse
import hashlib
import html
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--suite', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--metal-reference', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
suite = a.suite.resolve()
if a.output.exists():
    p.error('Refusing to overwrite an analysis snapshot')
manifest_path = suite/'manifest.json'
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
reference = json.loads(a.reference.read_text())
assert reference['color_space'] == 'linear BT.709'
runs = {(r['half_orders'], r['subdivision']): r for r in reference['runs']}
metal_reference = json.loads(a.metal_reference.read_text()) if a.metal_reference else None
metal_runs = {}
if metal_reference:
    if metal_reference['color_space'] != 'linear BT.709':
        raise ValueError('Unsupported metal-reference color space')
    metal_runs = {(r['half_orders'], r['subdivision']): r for r in metal_reference['runs']}
    if set(metal_runs) != {(16,4),(64,4),(128,4),(128,8),(256,4)}:
        raise ValueError('Metal reference is incomplete or has unexpected resolutions')
def expected(run, report):
    g = next(r['rgb'] for r in run['regions']
             if (r['side'], r['region']) == (report['illumination'], report['angular_region']))
    if not report.get('mix_mirror', False):
        return g
    mirror = run['neutral_rgb'] if (report['illumination'], report['angular_region']) == ('reflection', 'central') else [0., 0., 0.]
    return [.5*(x+y) for x,y in zip(g, mirror)]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
rows = []
for job in manifest['runs']:
    row = dict(name=job['name'], returncode=job['returncode'])
    reports = [(suite/path, report) for path, report in job['reports'].items()
               if not path.endswith('.preview.json')]
    if len(reports) != 1:
        row['status'] = 'missing_or_ambiguous_scene_report'
        rows.append(row)
        continue
    report_path, report = reports[0]
    prefix = report_path.with_suffix('')
    row['recorded_status'] = report['status']
    row['quality'] = report.get('quality', 'REALISTIC')
    row['status'] = 'failed' if job['returncode'] else 'rendered_requires_physical_review'
    row['artifacts'] = {}
    preview = prefix.with_suffix('.preview.json')
    info = json.loads(preview.read_text()) if preview.exists() else None
    for ext in ('.blend', '.exr', '.png'):
        path = prefix.with_suffix(ext)
        if path.exists():
            value = digest(path)
            if ext[1:]+'_sha256' in report and value != report[ext[1:]+'_sha256']:
                # The first frozen suite exported fixture PNGs and then regenerated
                # previews from EXR at the same path. Preserve that provenance defect
                # explicitly; never waive a source EXR or blend hash mismatch.
                if (ext != '.png' or not info or info.get('preview_file') or value != info['png_sha256'] or
                    info['exr_sha256'] != report.get('exr_sha256') or
                    digest(prefix.with_suffix('.exr')) != info['exr_sha256']):
                    raise ValueError('Artifact changed: '+str(path))
                row['preview_provenance'] = dict(
                    status='regenerated_from_verified_exr_original_png_superseded',
                    original_png_sha256=report['png_sha256'],
                    displayed_png_sha256=value,
                    original_png_available=False,
                    scope='Current PNG is the recorded EXR-derived preview, not the original fixture PNG.')
            row['artifacts'][ext[1:]] = str(path)
    if not job['returncode']:
        if info is None:
            raise ValueError('Missing preview report')
        settings = info['settings']
        if settings['adaptive_sampling'] or settings['denoising'] or settings['time_limit'] != 0:
            raise ValueError('Invalid fixed-sample configuration')
        if settings['use_layer_samples'] != 'IGNORE' and any(settings['layer_samples'].values()):
            raise ValueError('Layer overrides present')
        preview_path = prefix.with_suffix('.png')
        if info.get('preview_file'):
            preview_path = prefix.with_suffix('.preview.png')
            if info['preview_file'] != preview_path.name:
                raise ValueError('Unexpected preview filename')
        for path, key in ((preview_path, 'png_sha256'),
                          (prefix.with_suffix('.exr'), 'exr_sha256')):
            if digest(path) != info[key]:
                raise ValueError('Preview/source mismatch')
        if info.get('preview_file'):
            if 'png' in row['artifacts']:
                row['artifacts']['original_png'] = row['artifacts']['png']
            row['artifacts']['png'] = str(preview_path)
        row['settings'] = settings
    if 'analytic_neutral_reference' in report:
        error = max(abs(v-report['analytic_neutral_reference']) for v in report['raw_mean_rgb'])
        if abs(error-report['max_mean_error']) > 1e-12:
            raise ValueError('Incorrect analytic error')
        row.update(analytic_reference=report['analytic_neutral_reference'],
                   measured_rgb=report['raw_mean_rgb'], maximum_error=error,
                   status='analytic_smoke_pass' if error < report['smoke_tolerance'] else 'analytic_smoke_fail')
    elif report.get('angular_region', 'all') != 'all' and 'raw_mean_rgb' in report:
        properties = report['properties']
        target = dict(pitch=740, depth=150, duty_cycle=float.fromhex('0x1.a3d70a0000000p-2'),
                      incident_ior=1, ridge_ior=1.5, ridge_extinction=0, groove_ior=1,
                      substrate_ior=1, substrate_extinction=0)
        if properties != target:
            raise ValueError('Material does not match angular reference')
        fine, coarse, low = [expected(runs[k], report) for k in ((64,8), (64,4), (16,8))]
        row.update(status=('fast_approximation_reference_comparison' if row['quality']=='FAST' else
                           'single_seed_comparison_requires_convergence'),
                   transport=report['transport_requested'], mix_mirror=report.get('mix_mirror', False),
                   illumination=report['illumination'], angular_region=report['angular_region'],
                   measured_rgb=report['raw_mean_rgb'], reference_rgb=fine,
                   error_rgb=[x-y for x,y in zip(report['raw_mean_rgb'], fine)],
                   quadrature_difference_rgb=[x-y for x,y in zip(fine, coarse)],
                   modal_difference_rgb=[x-y for x,y in zip(fine, low)])
    elif (metal_reference and report.get('case') == 'relief_metal' and
          report.get('angular_region', 'all') == 'all' and report.get('illumination') == 'both' and
          not report.get('mix_mirror', False) and 'raw_mean_rgb' in report):
        props = report['properties']
        expected_profile = dict(pitch=metal_reference['pitch_nm'], depth=metal_reference['depth_nm'],
                                duty_cycle=metal_reference['duty'], incident_ior=1,
                                ridge_ior=metal_reference['metal_n'], ridge_extinction=metal_reference['metal_k'],
                                groove_ior=1, substrate_ior=metal_reference['metal_n'],
                                substrate_extinction=metal_reference['metal_k'])
        if props != expected_profile:
            raise ValueError('Metal material does not match spectral reference')
        fine = metal_runs[256,4]['rgb']
        row.update(status=('fast_approximation_reference_comparison' if row['quality']=='FAST' else
                           'metal_modal_accuracy_diagnostic_requires_convergence'),
                   measured_rgb=report['raw_mean_rgb'], reference_rgb=fine,
                   error_rgb=[x-y for x,y in zip(report['raw_mean_rgb'],fine)],
                   same_N16_reference_rgb=metal_runs[16,4]['rgb'],
                   same_N16_error_rgb=[x-y for x,y in zip(report['raw_mean_rgb'],metal_runs[16,4]['rgb'])],
                   modal_differences_rgb={f'{low}_to_{high}': [x-y for x,y in zip(metal_runs[high,4]['rgb'],metal_runs[low,4]['rgb'])]
                                          for low,high in ((16,64),(64,128),(128,256))},
                   quadrature_difference_rgb=[x-y for x,y in zip(metal_runs[128,8]['rgb'],metal_runs[128,4]['rgb'])])
    rows.append(row)
partitions = []
for transport in ('pt', 'bdpt', 'guided', 'bdpt_guided'):
    for mixed in (False, True):
        subset = [r for r in rows if r.get('transport') == transport and r.get('mix_mirror') == mixed]
        keys = {(r['illumination'], r['angular_region']) for r in subset}
        complete = {(side, region) for side in ('reflection', 'transmission') for region in ('central', 'middle', 'outer')}
        if len(subset) != 6 or keys != complete:
            continue
        total = [sum(r['measured_rgb'][c] for r in subset) for c in range(3)]
        neutral = runs[64,8]['neutral_rgb']
        partitions.append(dict(transport=transport, mix_mirror=mixed, sum_rgb=total,
                               neutral_reference_rgb=neutral,
                               error_rgb=[x-y for x,y in zip(total,neutral)],
                               status='single_seed_partition_diagnostic',
                               scope='Disjoint reflected/transmitted angular regions must sum to the lossless furnace; not a statistical acceptance test.'))
result = dict(scope=__doc__, suite_status=manifest['status'], planned=len(manifest['planned_jobs']),
              completed_jobs=len(rows), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
              reference_sha256=digest(a.reference),
              metal_reference_sha256=digest(a.metal_reference) if a.metal_reference else None, comparisons=rows, angular_partitions=partitions,
              limitations=manifest['limitations'],
              timing_scope='No isolated benchmark is inferred from suite process durations.')
a.output.mkdir(parents=True)
(a.output/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
cards = []
for row in rows:
    artifacts = row.get('artifacts', {})
    image = ('<img loading="lazy" src="'+html.escape(Path(artifacts['png']).as_uri())+'">'
             if 'png' in artifacts else '<p>No completed preview.</p>')
    metrics = {k:row[k] for k in ('measured_rgb','reference_rgb','analytic_reference','maximum_error','error_rgb') if k in row}
    links = ' · '.join('<a href="'+html.escape(Path(v).as_uri())+'">'+k.upper()+'</a>' for k,v in artifacts.items())
    notice = ('<p>EXR-derived preview; original fixture PNG was superseded by preview export.</p>'
              if 'preview_provenance' in row else '')
    cards.append('<article><h2>'+html.escape(row['name'])+'</h2>'+image+'<p>'+html.escape(row['status'])+'</p>'+notice+'<pre>'+html.escape(json.dumps(metrics,indent=2))+'</pre><p>'+links+'</p></article>')
(a.output/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Diffraction candidate results</title>
<style>body{font:16px system-ui;background:#17191d;color:#eee;margin:28px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}article{background:#252932;padding:18px;overflow-wrap:anywhere}img{width:100%;height:250px;object-fit:contain}h2{font-size:16px}a{color:#9cd0ff}pre{white-space:pre-wrap;font-size:12px}</style>
<h1>Frozen diffraction candidate: recorded results</h1><p>'''+f'{len(rows)} / {result["planned"]} jobs recorded. Suite state: {html.escape(result["suite_status"])}.'+'''</p><p>Previews preserve scene display transforms. Numerical comparisons use unscaled linear RGB. Analytic smoke passes are limited controls; single-seed angular comparisons are not convergence certification. No fastest-version claim is made by this report.</p><main>'''+''.join(cards)+'</main></html>')
print(json.dumps(dict(completed=len(rows),planned=result['planned'],status=result['suite_status'])))
