#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare raw modal audits without treating the highest resolution as truth."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, nargs='+', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
groups, failures, hashes = {}, [], {}
profile = None
for path in a.input:
    report = json.loads(path.read_text())
    current = (report['pitch_nm'], report['depth_nm'])
    if profile is not None and profile != current:
        raise ValueError('Geometry mismatch')
    profile = current
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in report['runs']:
        key = (row['angle'], row['azimuth'], row['wavelength_nm'])
        if 'error' in row:
            failures.append(row)
            continue
        orders = {x['order']: x for x in row['orders']}
        if len(orders) != len(row['orders']):
            raise ValueError('Duplicate diffraction order')
        if any(not math.isfinite(x[field]) for x in orders.values() for field in ('R','T')):
            raise ValueError('Nonfinite order power')
        modes = groups.setdefault(key, {})
        previous = modes.get(row['half_orders'])
        if previous is not None and previous != row:
            raise ValueError('Conflicting repeated modal result')
        modes[row['half_orders']] = row
comparisons = []
for key, modes in sorted(groups.items()):
    for low, high in itertools.combinations(sorted(modes), 2):
        left = {x['order']: x for x in modes[low]['orders']}
        right = {x['order']: x for x in modes[high]['orders']}
        item = dict(angle=key[0], azimuth=key[1], wavelength_nm=key[2], lower=low, higher=high)
        for field, name in [('R','reflection_l1'),('T','substrate_entering_flux_l1')]:
            item[name] = sum(abs(left.get(k, {}).get(field, 0)-right.get(k, {}).get(field, 0))
                             for k in left.keys() | right.keys())
        comparisons.append(item)
result = dict(input_sha256=hashes, unsupported_cases=failures, comparisons=comparisons,
              scope='Successive modal differences, not an absolute error bound. For absorbing substrates T is entering flux, not far-field transmission. Energy balance with residual-defined absorption is not independent validation.')
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(comparisons=len(comparisons), unsupported_cases=len(failures))))
