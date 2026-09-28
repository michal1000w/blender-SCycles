# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Recompute report statistics from unchanged EXRs using float64 reductions."""
import json
from pathlib import Path
import sys
import bpy
import numpy as np

root = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
manifest = json.loads((root / 'manifest.json').read_text())
for name, entry in manifest['runs'].items():
    assert entry.get('exit_code') == 0
    report_path = root / name / 'report.json'
    report = json.loads(report_path.read_text())
    for stats in report['passes'].values():
        image = bpy.data.images.load(stats['file'], check_existing=False)
        values = np.asarray(image.pixels[:], dtype=np.float64).reshape(-1, image.channels)
        assert np.isfinite(values).all()
        rgb = values[:, :min(3, image.channels)]
        stats.update(minimum=float(rgb.min()), maximum=float(rgb.max()), mean=float(rgb.mean()))
        bpy.data.images.remove(image)
    report['statistics_accumulator'] = 'float64; background depth sentinels can overflow a float32 sum'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    entry['report'] = report
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
print('Verified finite pixels and refreshed statistics for', len(manifest['runs']), 'scenes')
