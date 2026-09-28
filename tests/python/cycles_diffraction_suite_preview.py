# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Run after a scene fixture to export its existing EXR without rerendering."""
import hashlib
import json
from pathlib import Path
import bpy

scene = bpy.context.scene
source = Path(scene.render.filepath)
if source.suffix.lower() != '.exr' or not source.is_file():
    raise RuntimeError('Expected completed linear EXR render')
settings = dict(samples=scene.cycles.samples,
                adaptive_sampling=scene.cycles.use_adaptive_sampling,
                denoising=scene.cycles.use_denoising,
                time_limit=scene.cycles.time_limit,
                use_layer_samples=scene.cycles.use_layer_samples,
                layer_samples={layer.name: layer.samples for layer in scene.view_layers},
                bidirectional=scene.cycles.use_bidirectional_path_tracing,
                guiding=scene.cycles.use_guiding,
                view_transform=scene.view_settings.view_transform,
                exposure=scene.view_settings.exposure,
                gamma=scene.view_settings.gamma)
if settings['adaptive_sampling'] or settings['denoising'] or settings['time_limit'] != 0:
    raise RuntimeError('Final suite requires fixed samples, no denoising or time limit')
if settings['use_layer_samples'] != 'IGNORE' and any(settings['layer_samples'].values()):
    raise RuntimeError('Layer overrides invalidate the fixed scene sample count')
image = bpy.data.images.load(str(source), check_existing=False)
preview = source.with_suffix('.preview.png')
report_path = source.with_suffix('.preview.json')
if preview.exists() or report_path.exists():
    raise RuntimeError('Refusing to overwrite an existing preview or its provenance')
# Preserve the fixture's chosen display transform, exposure and gamma.
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_depth = '8'
image.save_render(str(preview), scene=scene)
report = dict(scope='Display preview of the completed linear EXR; no image normalization.',
              preview_file=preview.name,
              settings=settings, size=list(image.size),
              exr_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              png_sha256=hashlib.sha256(preview.read_bytes()).hexdigest())
report_path.write_text(json.dumps(report, indent=2)+'\n')
