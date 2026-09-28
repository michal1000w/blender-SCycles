#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare dielectric modal solver accuracy and isolated CPU preparation times."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    data = json.loads(path.read_text())
    if data['scope'] != 'CPU modal preparation':
        raise ValueError('Unexpected benchmark scope')
    rows = {}
    for row in data['runs']:
        key = (row['half_orders'], row['ky'], row['repeat'])
        if key in rows or 'error' in row:
            raise ValueError('Duplicate or failed benchmark case')
        if row['warmup'] != (row['repeat'] == 0):
            raise ValueError('Incorrect warmup classification')
        if not math.isfinite(row['seconds']) or row['seconds'] <= 0:
            raise ValueError('Invalid duration')
        count = len(row['ports'])
        matrix = [complex(*v) for v in row['matrix']]
        if not count or len(matrix) != 4 * count * count or any(
                not math.isfinite(v.real) or not math.isfinite(v.imag) for v in matrix):
            raise ValueError('Invalid physical scattering operator')
        for field in ('minimum_power_gain', 'maximum_power_gain'):
            if not math.isfinite(row[field]):
                raise ValueError('Nonfinite power gain')
        rows[key] = (row, matrix)
    expected = {(n, ky, repeat) for n in (64, 128, 256)
                for ky in (0., -0.16666666666666663) for repeat in range(5)}
    if rows.keys() != expected:
        raise ValueError('Incomplete benchmark matrix')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--numeric-only', action='store_true',
                        help='Suppress timing comparisons for runs with competing jobs')
    args = parser.parse_args()
    baseline, candidate = read(args.baseline), read(args.candidate)
    differences = []
    for key in sorted(baseline):
        left, a = baseline[key]
        right, b = candidate[key]
        if left['ports'] != right['ports']:
            raise ValueError('Physical port topology changed')
        count = len(left['ports'])
        columns = []
        for c in range(count):
            column = 0.
            for r in range(count):
                indices = [(2*r+i)*2*count+2*c+j for i in range(2) for j in range(2)]
                column += abs(.5*sum(abs(a[i])**2-abs(b[i])**2 for i in indices))
            columns.append(column)
        differences.append(dict(half_orders=key[0], ky=key[1], repeat=key[2],
                                maximum_complex_difference=max(abs(x-y) for x, y in zip(a, b)),
                                maximum_power_column_l1=max(columns)))
    timings = []
    for modes, ky in ([] if args.numeric_only else sorted({key[:2] for key in baseline})):
        medians = [statistics.median(rows[(modes, ky, repeat)][0]['seconds']
                                    for repeat in range(1, 5))
                   for rows in (baseline, candidate)]
        timings.append(dict(half_orders=modes, ky=ky, baseline_seconds=medians[0],
                            candidate_seconds=medians[1], speedup=medians[0]/medians[1]))
    passed = all(row['maximum_complex_difference'] <= 1e-9 and
                 row['maximum_power_column_l1'] <= 1e-9 for row in differences)
    report = dict(passed=passed, differences=differences, timings=timings,
                  timing_comparison_enabled=not args.numeric_only,
                  complex_tolerance=1e-9, power_column_l1_tolerance=1e-9,
                  scope='Same-resolution operator equivalence and CPU preparation; '
                        'not modal convergence or GPU render performance.',
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (args.baseline, args.candidate)})
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
