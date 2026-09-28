#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Measure raw furnace and zero-depth controls from cycles_diffraction_suite.py.

Spatial pixel variation is not an independent-seed Monte Carlo error estimate.
This reports observations; it does not turn a noisy image into an energy proof.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    result = {}
    for case in ['furnace', 'depth_duty']:
        array_path = args.directory / (case + '.npy')
        metadata_path = args.directory / (case + '.json')
        data = np.load(array_path)
        metadata = json.loads(metadata_path.read_text())
        if not np.isfinite(data).all():
            raise ValueError('Nonfinite linear image: ' + case)
        if not metadata.get('isolated_swatches'):
            raise ValueError('Control measurements require isolated swatches')
        if metadata.get('camera_type', 'ORTHO') != 'ORTHO':
            raise ValueError('These measurements require an orthographic camera')
        height, width, _ = data.shape
        # The first suite revision used the known four-column 0.44 m camera
        # scale but did not serialize it. Later revisions record it explicitly.
        ortho = metadata.get('camera_ortho_scale', 0.44)
        fit = metadata.get('camera_sensor_fit', 'AUTO')
        denominator = width if fit == 'HORIZONTAL' else height if fit == 'VERTICAL' else max(width, height)
        pixel_scale = ortho / denominator
        yy, xx = np.mgrid[:height, :width]
        px = (xx + 0.5 - width / 2) * pixel_scale
        py = (height / 2 - yy - 0.5) * pixel_scale
        swatches = []
        for parameters in metadata['parameters']:
            if case == 'depth_duty' and parameters['depth'] != 0:
                continue
            cx, cy = parameters['projected_center']
            radius = parameters['radius'] * 0.92
            mask = (px - cx) ** 2 + (py - cy) ** 2 < radius ** 2
            values = data[mask].astype(np.float64)
            swatches.append(dict(caption=parameters['caption'], pixels=len(values),
                                 mean_rgb=values.mean(0).tolist(), spatial_std_rgb=values.std(0).tolist(),
                                 minimum=float(values.min()), maximum=float(values.max())))
        result[case] = dict(samples=metadata['samples'], swatches=swatches,
                            scope='Interior 92% of projected radius; descriptive statistics, not a global energy proof',
                            input_sha256={array_path.name: digest(array_path), metadata_path.name: digest(metadata_path)})
    output = args.output or args.directory / 'control_measurements.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
