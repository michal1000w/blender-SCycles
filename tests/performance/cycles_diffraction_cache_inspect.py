#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Inspect an exported diffraction audit cache without rebuilding its response."""
import argparse
import json
import pathlib
import numpy as np


def inspect(directory, query):
    manifest = json.loads((directory / 'cache.json').read_text())
    if (manifest.get('schema'), manifest.get('version'), manifest.get('byte_order'),
            manifest.get('binary')) != ('cycles-diffraction-audit', 1, 'little', 'cache.bin'):
        raise ValueError('Unsupported audit cache format')
    binary = directory / 'cache.bin'
    if binary.stat().st_size != manifest['binary_bytes']:
        raise ValueError('Truncated or mismatched cache binary')
    buffers = {}
    end = 0
    for name, section in manifest['buffers'].items():
        dtype = {'int32': '<i4', 'float32': '<f4'}[section['type']]
        offset, count, components = (section[k] for k in ('offset', 'count', 'components'))
        size = count * components * 4
        if (min(offset, count) < 0 or components not in (1, 2, 4) or offset % 16 or
                offset < end or size != section['bytes'] or offset + size > manifest['binary_bytes']):
            raise ValueError('Invalid buffer extent: ' + name)
        buffers[name] = (np.memmap(binary, mode='r', dtype=dtype, offset=offset,
                                  shape=(count, components)) if count else
                         np.empty((0, components), dtype=dtype))
        end = offset + size
    point = np.asarray(query, dtype=np.float32)
    if not np.isfinite(point).all():
        raise ValueError('Nonfinite query')
    if manifest['mirror_symmetry']:
        point[:2] = -np.abs(point[:2])
    if np.any(point < manifest['lower']) or np.any(point > manifest['upper']):
        raise ValueError('Query outside cache domain')
    nodes = buffers['nodes']
    index = 0
    for _ in range(len(nodes)):
        if not 0 <= index < len(nodes):
            raise ValueError('Invalid tree child')
        axis, left, right, split_bits = map(int, nodes[index])
        if axis == -1:
            leaf = left
            break
        if axis not in (0, 1, 2):
            raise ValueError('Invalid tree axis')
        split = np.array(split_bits, dtype=np.int32).view(np.float32).item()
        index = left if point[axis] < split else right
    else:
        raise ValueError('Cyclic tree')
    if leaf < 0:
        return dict(query=point.tolist(), empty_leaf=True, profile=manifest['profile'])
    layout, bounds = buffers['cell_layout'], buffers['cell_bounds']
    if 2 * leaf + 1 >= len(layout) or 2 * leaf + 1 >= len(bounds):
        raise ValueError('Invalid cell index')
    matrix, port, active, degree = map(int, layout[2 * leaf])
    ports, active_count, _, _ = map(int, layout[2 * leaf + 1])
    if degree not in (0, 1, 2) or ports <= 0 or min(matrix, port, active, active_count) < 0:
        raise ValueError('Invalid cell layout')
    controls = 27 if degree == 2 else 8
    matrix_end = matrix + controls * (2 * ports) ** 2
    if (matrix_end > len(buffers['matrices']) or port + ports > len(buffers['ports']) or
            active + active_count > len(buffers['active_ports'])):
        raise ValueError('Cell payload exceeds buffer')
    lower, upper = bounds[2 * leaf], bounds[2 * leaf + 1]
    if np.any(upper[:3] <= lower[:3]) or np.any(point < lower[:3]) or np.any(point > upper[:3]):
        raise ValueError('Cell bounds disagree with lookup')
    values = buffers['matrices'][matrix:matrix_end]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite cell matrix')
    return dict(query=point.tolist(), leaf=leaf, degree=degree,
                lower=lower[:3].tolist(), upper=upper[:3].tolist(),
                coordinate=((point-lower[:3])/(upper[:3]-lower[:3])).tolist(),
                rotation=[float(lower[3]), float(upper[3])],
                ports=buffers['ports'][port:port+ports].tolist(),
                active_ports=buffers['active_ports'][active:active+active_count, 0].tolist(),
                matrix_float2_offset=matrix, matrix_float2_count=matrix_end-matrix,
                profile=manifest['profile'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=pathlib.Path)
    parser.add_argument('--query', type=float, nargs=3, required=True,
                        metavar=('BLOCH', 'KY', 'WAVELENGTH_NM'))
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    result = json.dumps(inspect(args.directory, args.query), indent=2) + '\n'
    if args.output:
        args.output.write_text(result)
    print(result, end='')


if __name__ == '__main__':
    main()
