# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Export editable scene copies with output paths separate from test evidence."""
import argparse
import json
from pathlib import Path
import sys
import bpy

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--presentation', type=Path, required=True)
p.add_argument('--features', type=Path, required=True)
p.add_argument('--coherence', type=Path, help='Per-case coherence suite, if separate from features')
p.add_argument('--multiggx', type=Path, help='Parent directory of the validated Multi-GGX appearance scenes')
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
a.presentation = a.presentation.resolve()
a.features = a.features.resolve()
out = a.presentation / 'scenes'
out.mkdir(exist_ok=False)
sources = {}
for name, entry in json.loads((a.presentation / 'manifest.json').read_text())['runs'].items():
    assert entry.get('exit_code') == 0
    sources[name] = a.presentation / name / 'denoising.blend'
if a.coherence:
    root = a.coherence.resolve()
    suite = json.loads((root / 'manifest.json').read_text())
    for job, result in suite['runs'].items():
        assert result.get('exit_code') == 0
        m = json.loads((root / job / 'manifest.json').read_text())
        for name, result in m['runs'].items():
            assert result['passed']
            sources['coherence_' + name] = root / job / (name + '.blend')
else:
    for name in json.loads((a.features / 'coherence_pt/manifest.json').read_text())['runs']:
        sources['coherence_' + name] = a.features / 'coherence_pt' / (name + '.blend')
if a.multiggx:
    for label, folder in [('multiggx_pt', 'multiggx_glossy_presentation_pt_v3b'),
                          ('multiggx_bdpt_guided', 'multiggx_glossy_presentation_combined_v3')]:
        sources[label] = a.multiggx.resolve() / folder / 'glossy.blend'
for name, source in sources.items():
    bpy.ops.wm.open_mainfile(filepath=str(source))
    scene = bpy.context.scene
    assert not scene.cycles.use_adaptive_sampling
    settings = (scene.cycles.samples, scene.cycles.use_denoising, scene.cycles.use_guiding, scene.cycles.use_bidirectional_path_tracing)
    scene.render.filepath = '//renders/' + name + '/render.exr'
    tree = scene.compositing_node_group
    if tree:
        for node in tree.nodes:
            if node.bl_idname == 'CompositorNodeOutputFile':
                node.directory = '//renders/' + name + '/passes'
    bpy.ops.wm.save_as_mainfile(filepath=str(out / (name + '.blend')))
    bpy.ops.wm.open_mainfile(filepath=str(out / (name + '.blend')))
    scene = bpy.context.scene
    assert not scene.cycles.use_adaptive_sampling
    assert (scene.cycles.samples, scene.cycles.use_denoising, scene.cycles.use_guiding, scene.cycles.use_bidirectional_path_tracing) == settings
    assert scene.render.filepath == '//renders/' + name + '/render.exr'
(out / 'README.txt').write_text(
    'Open these files with the selected diffraction Blender application.\n'
    'These are copies of the tested scenes; only output destinations were changed.\n'
    'New renders go to scenes/renders/<scene>/ and do not overwrite acceptance evidence.\n'
    'Material scenes use labeled OIDN presentation settings; coherence scenes remain raw.\n'
    'Adaptive sampling is disabled in every scene. See ../index.html and the delivery report.\n')
print('Exported', len(sources), 'editable scene copies')
