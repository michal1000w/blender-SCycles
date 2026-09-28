#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize a completed diagnostic tensor-cache refinement trace."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--log', type=Path, required=True)
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
a = parser.parse_args()
result = json.loads(a.result.read_text())
nodes = {}
for line in a.log.read_text().splitlines():
    match = re.fullmatch(r'TRACE prepare node=(\d+) depth=(\d+) prepared=([01]) rank=(-?\d+) bounds=([^:]+):([^ ]+) reason=(.*)', line)
    if match:
        index, depth, prepared, rank, lo, hi, reason = match.groups()
        nodes[int(index)] = dict(index=int(index), depth=int(depth), prepared=bool(int(prepared)),
                                rank=int(rank), lower=list(map(float, lo.split(','))),
                                upper=list(map(float, hi.split(','))), reason=reason)
        continue
    match = re.fullmatch(r'TRACE validation node=(\d+) maximum=([^ ]+) complex=([^ ]+)', line)
    if match:
        index, power, amplitude = match.groups()
        nodes[int(index)].update(power_error=float(power), complex_error=float(amplitude))
        continue
    match = re.fullmatch(r'TRACE split node=(\d+) axis=(\d+) curvature=([^ ]+)', line)
    if match:
        index, axis, curvature = match.groups()
        nodes[int(index)].update(split_axis=int(axis), curvature=float(curvature))
if len(nodes) != result['nodes']:
    raise ValueError('Incomplete trace or unclassified node failure')
report = dict(result=result, nodes=list(nodes.values()),
              split_counts=dict(Counter(str(n['split_axis']) for n in nodes.values() if 'split_axis' in n)),
              preparation_failures=dict(Counter(n['reason'] for n in nodes.values() if not n['prepared'])),
              deepest_nodes=sorted(nodes.values(), key=lambda n:n['depth'], reverse=True)[:10],
              input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.log,a.result)})
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('nodes','input_sha256')}, indent=2))
