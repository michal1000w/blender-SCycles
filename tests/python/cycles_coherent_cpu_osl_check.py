# SPDX-License-Identifier: Apache-2.0
"""CPU OSL variant of coherent acceptance cases: same scene, reference and gate as a plan job.

  blender --background --factory-startup --python THIS -- PLAN_JSON CASE_LABEL... -- REPORT
"""
import json, sys
from pathlib import Path
import bpy, numpy as np
args = sys.argv[sys.argv.index('--') + 1:]
plan = json.loads(Path(args[0]).read_text()); labels = args[1:-1]; report = Path(args[-1])
jobs = {j['case']: j for j in plan['jobs']}
results = {}
for label in labels:
    job = jobs[label]; a = job['arguments']
    blend = a[a.index('--blend') + 1]; ref = np.load(a[a.index('--reference') + 1])
    key = {'phase_0': 'phase_0_radiance', 'phase_pi': 'phase_pi_radiance'}.get(a[a.index('--case') + 1], 'incoherent_radiance')
    bpy.ops.wm.open_mainfile(filepath=blend); s = bpy.context.scene
    s.cycles.device = 'CPU'; s.cycles.shading_system = True; s.cycles.samples = 128; s.cycles.seed = 19
    s.cycles.use_bidirectional_path_tracing = False; s.cycles.use_guiding = False
    s.cycles.use_adaptive_sampling = False; s.cycles.use_denoising = False
    s.cycles.pixel_filter_type = 'BOX'; s.cycles.filter_width = 1.0
    out = report.parent / (label + '_osl.exr'); out.parent.mkdir(parents=True, exist_ok=True)
    s.render.filepath = str(out); s.render.image_settings.file_format = 'OPEN_EXR'; s.render.image_settings.color_depth = '32'
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(out)); w, h = img.size
    px = np.asarray(img.pixels[:], dtype=np.float64).reshape(h, w, img.channels)[:, :, :3].mean(axis=2)
    roi = ref['roi_mask'].astype(bool); e = px[roi] - ref[key][roi]
    m = {'absolute_mean_error': float(abs(e.mean())), 'absolute_rmse': float(np.sqrt((e * e).mean()))}
    results[label] = {'metrics': m, 'gate': job['gate'],
                      'passed': m['absolute_mean_error'] <= job['gate']['absolute_mean_error'] and m['absolute_rmse'] <= job['gate']['absolute_rmse']}
    print(label, results[label])
report.write_text(json.dumps({'scope': 'CPU OSL shading variant', 'passed': all(r['passed'] for r in results.values()), 'cases': results}, indent=1) + '\n')
