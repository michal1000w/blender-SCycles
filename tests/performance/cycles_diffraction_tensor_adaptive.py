# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded adaptive tensor experiment for the lossless 740 nm N16 profile.

This is an offline investigation, not a production cache. Random validation is
not a worst-case error bound. An unfinished tree is never marked complete.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sampler', type=Path, required=True)
    p.add_argument('--validator', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--maximum-nodes', type=int, default=31)
    p.add_argument('--maximum-depth', type=int, default=18)
    p.add_argument('--maximum-bytes', type=int, default=256*1024*1024)
    p.add_argument('--tolerance', type=float, default=.001)
    args = p.parse_args()
    if (args.maximum_nodes < 1 or args.maximum_depth < 0 or args.maximum_bytes < 1 or
            not 0 < args.tolerance < 1):
        p.error('Invalid resource or accuracy limit')
    args.output.mkdir(parents=True, exist_ok=False)
    fit = Path(__file__).with_name('cycles_diffraction_tensor_fit.py')
    sources = [Path(__file__), fit, args.sampler, args.validator]
    hashes = {str(x.resolve()): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    lower = np.array([-.5, -1, 380.])
    upper = np.array([.5, 1, 780.])
    nodes = [dict(lower=lower.tolist(), upper=upper.tolist(), depth=0, status='pending')]
    pending = [0]
    used = 0
    accepted_volume = 0.
    start = time.monotonic()
    stopped = None
    # Breadth first avoids spending the entire budget in one corner of the domain.
    while pending:
        if sum(n['status'] != 'pending' for n in nodes) >= args.maximum_nodes:
            stopped = 'node budget'
            break
        index = pending.pop(0)
        node = nodes[index]
        lo, hi = np.array(node['lower']), np.array(node['upper'])
        directory = args.output / f'node_{index:05d}'
        directory.mkdir()
        samples = directory / 'samples.json'
        with samples.open('w') as out, (directory/'sample.log').open('w') as log:
            subprocess.run([str(args.sampler.resolve()), str(190731 + index),
                            *map(lambda x: format(x, '.17g'), [*lo, *hi])],
                           stdout=out, stderr=log, check=True)
        report, predictions, model = [directory/name for name in ('fit.json', 'predictions.txt', 'model.npz')]
        with (directory/'fit.log').open('w') as log:
            result = subprocess.run([sys.executable, str(fit), '--precision', 'float32',
                                     '--degree', '5', '--compression-tolerance', '1e-7',
                                     '--input', str(samples), '--output', str(report),
                                     '--predictions', str(predictions), '--model', str(model)],
                                    stdout=log, stderr=log)
        accepted = False
        if result.returncode == 0:
            with predictions.open() as inp, (directory/'power.json').open('w') as out:
                subprocess.run([str(args.validator.resolve())], stdin=inp, stdout=out, check=True)
            power = json.loads((directory/'power.json').read_text())
            node['power'] = power
            # Empty sampled directions alone cannot certify an entire empty cell.
            accepted = (power['physical_queries'] > 0 and
                        power['maximum_power_column_l1_error'] <= args.tolerance)
        else:
            node['fit_failed'] = True
        if accepted:
            with np.load(model) as packed:
                size = sum(packed[k].nbytes for k in packed.files)
            if used + size > args.maximum_bytes:
                node['status'] = 'memory_limit'
                stopped = 'matrix memory budget'
                break
            used += size
            node.update(status='accepted', model=str(model.relative_to(args.output)), bytes=size)
            accepted_volume += float(np.prod((hi-lo)/(upper-lower)))
        else:
            if node['depth'] >= args.maximum_depth:
                node['status'] = 'depth_limit'
                stopped = 'depth limit'
                break
            data = json.loads(samples.read_text())
            raw = np.asarray([s['matrix'] for s in data['samples'] if s['fitting_sample']])
            values = (raw[...,0] + 1j*raw[...,1]).reshape(7,7,7,-1)
            # Variation only selects the split; physical validation decides acceptance.
            curvature = [float(np.linalg.norm(np.diff(values, n=2, axis=2-a))) for a in range(3)]
            axis = max(range(3), key=lambda a: (curvature[a], (hi[a]-lo[a])/(upper[a]-lower[a])))
            middle = float(np.float32((lo[axis]+hi[axis])/2))
            if not lo[axis] < middle < hi[axis]:
                node['status'] = 'coordinate_limit'
                stopped = 'float coordinate limit'
                break
            left_hi, right_lo = hi.copy(), lo.copy()
            left_hi[axis] = right_lo[axis] = middle
            children = [len(nodes), len(nodes)+1]
            nodes.extend([dict(lower=lo.tolist(), upper=left_hi.tolist(), depth=node['depth']+1, status='pending'),
                          dict(lower=right_lo.tolist(), upper=hi.tolist(), depth=node['depth']+1, status='pending')])
            pending.extend(children)
            node.update(status='split', axis=axis, middle=middle, children=children, curvature=curvature)
        print(json.dumps(dict(node=index, status=node['status'], bytes=used,
                              accepted_domain_fraction=accepted_volume)), flush=True)
    after = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in hashes}
    if after != hashes:
        raise RuntimeError('Source or tool binary changed during experiment')
    complete = stopped is None and not pending
    result = dict(scope=__doc__, complete=complete, stop_reason=stopped, nodes=nodes,
                  accepted_domain_fraction=accepted_volume, model_bytes=used,
                  tolerance=args.tolerance, seconds=time.monotonic()-start, source_hashes=hashes,
                  limitations=['One lossless profile and N16 truncation',
                               '128 random validation points per region, not a worst-case guarantee',
                               'No scene-manager or production device integration'])
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('nodes', 'source_hashes')}))


if __name__ == '__main__':
    main()
