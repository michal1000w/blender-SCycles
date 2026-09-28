# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Check saved linear outputs without rendering or fitting exposure."""
import argparse
import json
from pathlib import Path
import sys
import bpy
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--features', type=Path, required=True)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])

def read(folder):
    paths = list((a.features / folder).glob('*.exr'))
    assert len(paths) == 1, paths
    image = bpy.data.images.load(str(paths[0].resolve()), check_existing=False)
    values = np.array(image.pixels[:], dtype=np.float64).reshape(-1, image.channels)[:, :3]
    bpy.data.images.remove(image)
    assert np.isfinite(values).all()
    return values

flat = read('thin_wall_flat')
uncovered = read('thin_wall_uncovered')
relief = read('thin_wall_pt')
flat_error = float(np.max(abs(flat - uncovered)))
relief_difference = float(np.sqrt(np.mean((relief - flat) ** 2)))
result = {'thin_wall': {
    'flat_vs_uncovered_max_absolute': flat_error,
    'gate': 'Native flat and uncovered controls must agree within 1e-5 per linear RGB channel',
    'passed': flat_error < 1e-5,
    'relief_vs_flat_rmse': relief_difference,
    'relief_effect_present': relief_difference > 0.001,
}, 'coherence': {}}
for transport in ['pt', 'bdpt', 'guided']:
    folder = a.features / ('coherence_' + transport)
    zero = np.load(folder / 'incoherent_reference.npz')['render']
    off = np.load(folder / 'off_reference.npz')['render']
    error = float(np.max(abs(zero - off)))
    result['coherence'][transport] = {'zero_length_vs_off_max_absolute': error,
        'gate': 'Equivalent ordinary transport within 1e-5 linear radiance', 'passed': error < 1e-5}
result['passed'] = (result['thin_wall']['passed'] and result['thin_wall']['relief_effect_present'] and
                    all(x['passed'] for x in result['coherence'].values()))
(a.features / 'image_controls.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if not result['passed']:
    raise RuntimeError('Final image controls failed; preserved values must be investigated')
