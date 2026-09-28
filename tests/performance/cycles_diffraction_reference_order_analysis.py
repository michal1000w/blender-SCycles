#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare reference-port operators at successive truncations, without declaring convergence."""
import argparse
import hashlib
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('fixtures', nargs='+', type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
groups = {}
keys = ('ridge', 'groove', 'substrate', 'incident_ior', 'depth_nm', 'ky',
        'pitch_nm', 'wavelength_nm', 'duty', 'kx', 'retained_half_orders')
for path in args.fixtures:
    for case in json.loads(path.read_text()):
        key = json.dumps({k: case[k] for k in keys}, sort_keys=True)
        order = case['half_orders']
        if order in groups.setdefault(key, {}):
            raise ValueError('Duplicate profile/order')
        groups[key][order] = case
comparisons = []
for key, cases in groups.items():
    orders = sorted(cases)
    for low, high in zip(orders, orders[1:]):
        a, b = cases[low], cases[high]
        if a['ports'] != b['ports'] or len(a['matrix']) != len(b['matrix']):
            raise ValueError('Reference basis mismatch')
        x = [complex(*z) for z in a['matrix']]
        y = [complex(*z) for z in b['matrix']]
        differences = [abs(u-v) for u, v in zip(x, y)]
        if not all(math.isfinite(z) for z in differences):
            raise ValueError('Nonfinite difference')
        comparisons.append(dict(profile=json.loads(key), lower=low, upper=high,
                                max_component_change=max(differences),
                                frobenius_change=sum(z*z for z in differences)**0.5,
                                reference_frobenius=sum(abs(z)**2 for z in y)**0.5))
args.output.write_text(json.dumps(dict(
    scope='Reference-port operator differences, including evanescent channels; not a physical power or render error metric and not a convergence certificate',
    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.fixtures},
    comparisons=comparisons), indent=2))
