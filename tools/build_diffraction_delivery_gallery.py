#!/usr/bin/env python3
"""Arrange unmodified renderer outputs and provenance into a delivery gallery."""
import argparse
import html
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--presentation', type=Path, required=True)
p.add_argument('--features', type=Path, required=True)
p.add_argument('--coherence', type=Path, help='Per-case combined-mode coherence suite, if separate from features')
p.add_argument('--multiggx', type=Path,
               help='Parent of multiggx_glossy_presentation_pt_v3b and '
                    'multiggx_glossy_presentation_combined_v3 outputs')
a = p.parse_args()
a.presentation = a.presentation.resolve()
a.features = a.features.resolve()
manifest = json.loads((a.presentation / 'manifest.json').read_text())
font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 22)
names = {'cd_dvd': 'CD / DVD', 'glass': 'Glass: roughness and film',
         'glossy': 'Glossy gratings', 'metallic': 'Metallic: film and coverage',
         'principled_film': 'Principled: tint, dispersion and film',
         'refraction': 'Refraction', 'covered_disc': 'Disc under dielectric cover',
         'indirect': 'Indirect diffraction (BDPT; noise remains)',
         'thin_wall': 'Thin Wall: roughness, film and dispersion',
         'ashikhmin': 'Glossy distribution comparison incl. Ashikhmin'}
style = 'body{background:#171b20;color:#eee;font:16px/1.5 system-ui;max-width:1400px;margin:40px auto;padding:20px}a{color:#9bd2ff}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px}article{background:#242a32;padding:18px}img{width:100%}strong{color:#ffce88}'
parts = [f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Cycles diffraction delivery</title><style>{style}</style>',
         '<h1>Cycles diffraction — selected Fast build</h1>',
         '<p>Real Metal renders. Adaptive sampling OFF. Material previews use OIDN; raw images, EXRs and scenes are linked. Interference checks use no denoising. No brightness normalization between transports.</p>',
         '<p><strong>Experimental development delivery: general coherent multipath, native MultiGGX diffraction beyond constant-input reflective Glossy GGX, and arbitrary-profile Realistic certification remain incomplete. Thin Wall is approximate; indirect noise remains.</strong></p>',
         f'<p>Executable SHA-256: <code>{manifest["sha256"]}</code>. <a href="manifest.json">Presentation manifest</a> · <a href="{os.path.relpath(a.features / "manifest.json", a.presentation)}">Feature suite manifest</a></p>',
         '<p><a href="../delivery_20260927/index.html">Historical numerical and pipeline gallery</a> · <a href="scenes/README.txt">Editable scene instructions</a></p>',
         '<div class="grid">']
tiles = []
for name, entry in manifest['runs'].items():
    assert entry.get('exit_code') == 0, (name, entry)
    folder = a.presentation / name
    report = json.loads((folder / 'report.json').read_text())
    title = names[name]
    caption = f'{report["transport"].upper()}, {report["samples"]} fixed samples, OIDN'
    tiles.append((folder / 'denoised.png', title, caption))
    parts.append(f'<article><h2>{html.escape(title)}</h2><img src="{name}/denoised.png" alt="{html.escape(title)}"><p>{caption}; {report["render_and_denoise_seconds"]:.2f} s including preparation and denoising. Presentation, not a physical reference.</p><p><a href="{name}/noisy.png">Raw preview</a> · <a href="{name}/passes/noisynoisy.exr">Raw linear EXR</a> · <a href="scenes/{name}.blend">Blender scene</a> · <a href="{name}/report.json">Pass checks</a></p></article>')
coherent_cases = {}
if a.coherence:
    root = a.coherence.resolve()
    suite = json.loads((root / 'manifest.json').read_text())
    assert suite['sha256'] == manifest['sha256']
    for job, result in suite['runs'].items():
        assert result.get('exit_code') == 0
        m = json.loads((root / job / 'manifest.json').read_text())
        assert m['sha256'] == manifest['sha256']
        for name, report in m['runs'].items():
            assert report['passed']
            coherent_cases[name] = (root / job / (name + '.png'), report)
    mode = 'BDPT and guiding'
else:
    coherent = json.loads((a.features / 'coherence_pt/manifest.json').read_text())
    for name, report in coherent['runs'].items():
        coherent_cases[name] = (a.features / 'coherence_pt' / (name + '.png'), report)
    mode = 'PT'
parts.append('</div><h2>Direct scalar interference — Metal ' + mode + ', 128 fixed samples, no denoising</h2><div class="grid">')
coherent_tiles = []
for name, (path, report) in coherent_cases.items():
    relative = os.path.relpath(path, a.presentation)
    label = {'red': '633 nm phase wavelength', 'unit_scale': 'Scene unit scale 0.5',
             'off': 'Coherence disabled', 'occluded': 'One source blocked',
             'three_sources': 'Three coherent sources'}.get(name, name.replace('_', ' ').title())
    coherent_tiles.append((path, label, f'Absolute radiance RMSE {report["absolute_rmse"]:.6g}'))
    parts.append(f'<article><h3>{label}</h3><img src="{relative}" alt="{label}"><p>Independent absolute-radiance RMSE {report["absolute_rmse"]:.6g}; passed: {report["passed"]}. <a href="scenes/coherence_{name}.blend">Scene</a> · <a href="{relative[:-4]}.exr">Raw EXR</a></p></article>')
feature_manifest = json.loads((a.features / 'manifest.json').read_text())
parts.append('</div><h2>Raw transport comparisons and controls</h2>')
if feature_manifest['sha256'] != manifest['sha256']:
    parts.append('<p>Historical controls from the earlier build identified in the feature manifest; '
                 'these images were not regenerated by the presentation build.</p>')
parts.append('<ul>')
for folder in sorted(a.features.iterdir()):
    if not folder.is_dir() or not folder.name.startswith(('thin_wall_', 'ashikhmin_')):
        continue
    for image_path in sorted(folder.glob('*.png')):
        relative = os.path.relpath(image_path, a.presentation)
        parts.append(f'<li><a href="{relative}">{html.escape(folder.name)}</a> · <a href="{relative[:-4]}.exr">Linear EXR</a> · <a href="{relative[:-4]}.blend">Original test scene</a></li>')
parts.append('</ul>')
multiggx_tiles = []
if a.multiggx:
    multiggx_root = a.multiggx.resolve()
    entries = [
        ('multiggx_glossy_presentation_pt_v3b', 'Glossy MultiGGX — PT', 'pt'),
        ('multiggx_glossy_presentation_combined_v3',
         'Glossy MultiGGX — BDPT and guiding', 'bdpt_guided'),
    ]
    parts.append('<h2>Constant-input reflective Glossy MultiGGX appendix</h2>')
    parts.append('<p>Saved native GGX grating appearance checks on Metal. These images show the '
                 'single-scatter control, a rough multiscatter grating, and an anisotropic '
                 'half-coverage grating. They do not certify arbitrary materials or render speed.</p>')
    parts.append('<div class="grid">')
    for folder_name, title, transport in entries:
        folder = multiggx_root / folder_name
        report = json.loads((folder / 'report.json').read_text())
        assert report['multiggx'] and report['transport'] == transport
        assert report['all_pixels_finite'] and not report['adaptive_sampling']
        assert not report['denoising'] and report['samples'] == 512
        image = folder / 'glossy.png'
        assert image.is_file() and (folder / 'glossy.exr').is_file()
        assert (folder / 'glossy.blend').is_file()
        relative = os.path.relpath(folder, a.presentation)
        scene_label = 'multiggx_pt' if transport == 'pt' else 'multiggx_bdpt_guided'
        multiggx_tiles.append((image, title, f'{transport.upper()}, 512 fixed samples'))
        parts.append(f'<article><h3>{html.escape(title)}</h3>'
                     f'<img src="{relative}/glossy.png" alt="{html.escape(title)}">'
                     '<p>Raw renderer output, fixed 512 samples, denoising and adaptive '
                     'sampling off. Appearance and node integration check, not a physical '
                     f'reference. <a href="{relative}/glossy.exr">Linear EXR</a> · '
                     f'<a href="scenes/{scene_label}.blend">Editable scene</a> · '
                     f'<a href="{relative}/report.json">Render report</a></p></article>')
    parts.append('</div>')
parts.append('</html>')
(a.presentation / 'index.html').write_text('\n'.join(parts))

def contact_sheet(items, output, columns, width, height):
    canvas = Image.new('RGB', (columns * width, ((len(items) + columns - 1) // columns) * height), '#171b20')
    draw = ImageDraw.Draw(canvas)
    for i, (path, title, caption) in enumerate(items):
        x, y = (i % columns) * width, (i // columns) * height
        source = Image.open(path).convert('RGB')
        preview = ImageOps.contain(source, (width - 20, height - 72))
        canvas.paste(preview, (x + (width - preview.width) // 2, y + 62))
        draw.text((x + 12, y + 8), title, fill='white', font=font)
        draw.text((x + 12, y + 34), caption, fill='#b9c6d6', font=font)
    canvas.save(output)

contact_sheet(tiles, a.presentation / 'material_overview.png', 2, 720, 470)
contact_sheet(coherent_tiles, a.presentation / 'coherence_overview.png', 3, 540, 330)
if multiggx_tiles:
    contact_sheet(multiggx_tiles, a.presentation / 'multiggx_overview.png', 2, 720, 470)
print(a.presentation / 'index.html')
