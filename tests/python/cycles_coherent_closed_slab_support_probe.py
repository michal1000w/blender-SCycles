#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record the exact current support boundary; never count rejection as physics acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy

parser = argparse.ArgumentParser()
parser.add_argument('--fixtures', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
scene_path = args.fixtures / 'phase_0.blend'
bpy.ops.wm.open_mainfile(filepath=str(scene_path.resolve()))
bpy.context.scene.cycles.device = 'GPU'
preferences = bpy.context.preferences.addons['cycles'].preferences
preferences.compute_device_type = 'METAL'
preferences.get_devices()
devices = [device for device in preferences.devices if device.type == 'METAL']
for device in preferences.devices:
    device.use = device.type == 'METAL'
expected = 'Streamed Mirror Facets supports flat triangle Mirror objects and Lambertian detectors only'
result = {'scene_sha256': hashlib.sha256(scene_path.read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'metal_devices': [device.name for device in devices],
          'physics_acceptance': False}
try:
    if not devices:
        raise RuntimeError('No Metal device available')
    result['render_status'] = sorted(bpy.ops.render.render())
    result['status'] = 'UNEXPECTED_RENDER_SUPPORT'
except Exception as error:
    result['message'] = str(error)
    result['status'] = ('EXPECTED_UNSUPPORTED_HOST_REJECTION'
                        if expected in str(error) else 'UNEXPECTED_ERROR')
(args.fixtures / 'unsupported_host_probe.json').write_text(json.dumps(result, indent=2) + '\n')
if result['status'] != 'EXPECTED_UNSUPPORTED_HOST_REJECTION':
    raise RuntimeError(f"Unexpected support-probe result: {result}")
print(json.dumps(result))
