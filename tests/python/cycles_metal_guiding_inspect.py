#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect a CYCLES_METAL_GUIDING_DUMP snapshot without changing its distribution.

The prefix includes the trained-sample suffix, e.g. /tmp/guide_128. Reports spatial
coverage and normalized directional entropy; optional points show the actual query cells.
"""

import argparse
import json
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=pathlib.Path)
    parser.add_argument('--point', nargs=3, type=float, action='append', default=[])
    args = parser.parse_args()
    metadata = json.loads(args.prefix.with_suffix('.json').read_text())
    dtype = np.dtype([('children', '<u4'), ('parent', '<u4'), ('axis', '<u4'),
                      ('split', '<f4'), ('depth', '<u4'), ('visits', '<u4')])
    nodes = np.fromfile(args.prefix.with_suffix('.nodes'), dtype=dtype)
    sampling = np.fromfile(args.prefix.with_suffix('.sampling'), dtype='<f4').reshape(
        metadata['nodes'], metadata.get('fields', 4), metadata.get('sampling_size', metadata['tree_size']))
    trees = sampling[:, :, :metadata['tree_size']]
    assert len(nodes) == metadata['nodes']
    assert np.isfinite(sampling).all(), 'Nonfinite sampling field'
    gaussians = None
    conditioned = None
    parallax = None
    if sampling.shape[2] > metadata['tree_size']:
        components = metadata.get('gaussian_components', 16)
        stride = (sampling.shape[2] - metadata['tree_size']) // components
        assert stride in (5, 18, 19), 'Unknown Gaussian component layout'
        gaussians = sampling[:, :, metadata['tree_size']:].reshape(*sampling.shape[:2], components, stride)
        assert np.all(gaussians[..., 0] >= 0), 'Negative Gaussian mass'
        assert np.allclose(gaussians[..., 0].sum(axis=2), trees[:, :, 0] > 0, atol=1e-5), \
            'Gaussian mixture does not conserve mass'
        assert np.all((gaussians[..., 1] >= 0) & (gaussians[..., 1] <= 16384)), \
            'Invalid Gaussian concentration'
        conditioned = np.zeros(gaussians.shape[:-1], dtype=bool)
        parallax = np.zeros(gaussians.shape[:-1], dtype=bool)
        if stride >= 18:
            assert np.isin(gaussians[..., 17], [0, 1, 2]).all(), 'Invalid conditioning flag'
            conditioned = gaussians[..., 17] == 1
            parallax = gaussians[..., 17] == 2
            assert np.all(gaussians[..., 14:17][parallax] >= 0), 'Invalid parallax metric extent'
            if stride == 19:
                assert np.all(gaussians[..., 18] >= 0), 'Invalid prior strength/support ratio'
            # Conditioned entries store the raw mean direction, not an already normalized axis.
            assert np.all(np.linalg.norm(gaussians[..., 2:5][conditioned], axis=-1) <= 1.0001), \
                'Invalid conditional mean direction'
        assert np.allclose(np.linalg.norm(gaussians[..., 2:5][~conditioned], axis=-1), 1, atol=2e-6), \
            'Nonunit Gaussian axis'
    assert ((nodes['children'] == 0) | (nodes['children'] + 1 < len(nodes))).all(), 'Invalid child index'
    assert (nodes['parent'] < len(nodes)).all(), 'Invalid parent index'
    leaves = np.flatnonzero(nodes['children'] == 0)
    bins = metadata['bins']
    weights = trees[:, :, -bins:].astype(float)
    assert np.allclose(weights.sum(axis=2), trees[:, :, 0], rtol=5e-5, atol=1e-7), \
        'Directional tree does not conserve mass'
    weights /= np.maximum(weights.sum(axis=2, keepdims=True), 1e-30)
    entropy_bins = np.exp(-(weights * np.log(np.maximum(weights, 1e-30))).sum(axis=2))
    report = dict(metadata=metadata, spatial_leaves=len(leaves),
                  depth_histogram=np.bincount(nodes['depth'][leaves]).tolist(),
                  visit_quantiles=np.quantile(nodes['visits'][leaves], [0, .25, .5, .75, 1]).tolist(),
                  fields={})
    names = ['surface_radiance', 'volume_radiance', 'surface_importance', 'volume_importance']
    if metadata.get('fields') == 14:
        names = [f'surface_radiance_{axis}' for axis in ['+x', '-x', '+y', '-y', '+z', '-z']] + [
            'volume_radiance'] + [f'surface_importance_{axis}' for axis in ['+x', '-x', '+y', '-y', '+z', '-z']] + [
            'volume_importance']
    for field, name in enumerate(names):
        valid = leaves[trees[leaves, field, 0] > 0]
        report['fields'][name] = dict(populated_leaves=len(valid),
            effective_bin_quantiles=np.quantile(entropy_bins[valid, field], [0, .25, .5, .75, 1]).tolist()
            if len(valid) else [])
        if gaussians is not None:
            report['fields'][name].update(
                conditioned_components=int(conditioned[leaves, field].sum()),
                parallax_components=int(parallax[leaves, field].sum()),
                parallax_mixture_mass_mean=float(
                    (gaussians[valid, field, :, 0] * parallax[valid, field]).sum(axis=1).mean())
                if len(valid) else 0.0,
                conditioned_mixture_mass_mean=float(
                    (gaussians[valid, field, :, 0] * conditioned[valid, field]).sum(axis=1).mean())
                if len(valid) else 0.0)
    resolution = int(np.sqrt(bins))
    levels = resolution.bit_length() - 1
    directions = []
    for index in range(bins):
        x = sum(((index >> (2 * bit)) & 1) << bit for bit in range(levels))
        y = sum(((index >> (2 * bit + 1)) & 1) << bit for bit in range(levels))
        phi, z = 2 * np.pi * (x + .5) / resolution, 2 * (y + .5) / resolution - 1
        r = np.sqrt(1 - z * z)
        directions.append([r * np.cos(phi), r * np.sin(phi), z])
    directions = np.asarray(directions)
    report['queries'] = []
    for point in args.point:
        node = 0
        lower, upper = np.array(metadata['bounds_min']), np.array(metadata['bounds_max'])
        while nodes[node]['children']:
            axis, split = int(nodes[node]['axis']), float(nodes[node]['split'])
            right = point[axis] >= split
            (lower if right else upper)[axis] = split
            node = int(nodes[node]['children']) + int(right)
        fields = {}
        for field, name in enumerate(names):
            top = np.argsort(weights[node, field])[-5:][::-1]
            fields[name] = dict(effective_bins=float(entropy_bins[node, field]),
                direction_mean=(weights[node, field] @ directions).tolist(),
                top_directions=[dict(direction=directions[i].tolist(), mass=float(weights[node, field, i]))
                                for i in top])
            if gaussians is not None:
                entries = gaussians[node, field].astype(float)
                axes = entries[:, 2:5].copy()
                concentrations = entries[:, 1].copy()
                active = conditioned[node, field]
                physical = parallax[node, field]
                query = (np.asarray(point) - metadata['bounds_min']) / np.maximum(
                    np.asarray(metadata['bounds_max']) - metadata['bounds_min'], 1e-8)
                if np.any(active):
                    delta = query - entries[active, 5:8]
                    axes[active] += np.einsum('nij,nj->ni', entries[active, 8:17].reshape(-1, 3, 3), delta)
                    lengths = np.linalg.norm(axes[active], axis=1)
                    axes[active] = np.divide(axes[active], lengths[:, None],
                                             out=np.zeros_like(axes[active]), where=lengths[:, None] > 1e-6)
                    degenerate = np.flatnonzero(active)[lengths <= 1e-6]
                    for index in degenerate:
                        mean_length = np.linalg.norm(entries[index, 2:5])
                        axes[index] = entries[index, 2:5] / mean_length if mean_length > 0 else [0, 0, 1]
                for index in np.flatnonzero(physical):
                    entry = entries[index]
                    delta = entry[5:8] - query * entry[14:17]
                    squared_distance = float(delta @ delta)
                    concentrations[index] = 0
                    if squared_distance > 1e-12 and np.isfinite(squared_distance):
                        axes[index] = delta / np.sqrt(squared_distance)
                        xx, xy, xz, yy, yz, zz = entry[8:14]
                        covariance = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
                        variance = max(float(np.trace(covariance) - axes[index] @ covariance @ axes[index]), 0)
                        r = min(np.sqrt(max(1 - variance / squared_distance, 0)), 1 - 1e-6)
                        concentrations[index] = min(r * (3 - r * r) / (1 - r * r), 16384)
                fields[name]['gaussians'] = [dict(mass=float(entry[0]), concentration=float(k),
                    axis=axis.tolist(), conditioned=bool(flag), parallax=bool(source))
                    for entry, axis, k, flag, source in zip(entries, axes, concentrations, active, physical)
                    if entry[0] > 0]
        report['queries'].append(dict(point=point, node=node, bounds_min=lower.tolist(),
                                      bounds_max=upper.tolist(), fields=fields))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
