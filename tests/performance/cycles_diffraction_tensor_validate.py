# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Fresh-query validation of frozen matrix-anchored tensor cells; no refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from cycles_diffraction_tensor_fit import basis, unpack_hermitian
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--cache',type=Path,required=True)
p.add_argument('--sampler',type=Path,required=True)
p.add_argument('--validator',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed-base',type=int,default=910237)
args=p.parse_args()
manifest=json.loads((args.cache/'manifest.json').read_text())
if not 0 <= args.seed_base <= 2**32-1-len(manifest['nodes']):raise ValueError('Invalid seed range')
if not manifest['complete']:raise ValueError('Cannot validate an incomplete cache as complete')
args.output.mkdir(parents=True,exist_ok=False)
reports=[]
for index,node in enumerate(manifest['nodes']):
    if node['status']!='accepted':continue
    directory=args.output/f'node_{index:05d}';directory.mkdir()
    samples=directory/'samples.json'
    with samples.open('w') as out,(directory/'sample.log').open('w') as log:
        subprocess.run([str(args.sampler.resolve()),str(args.seed_base+index),
                        *map(lambda x:format(x,'.17g'),node['lower']+node['upper'])],stdout=out,stderr=log,check=True)
    data=json.loads(samples.read_text())
    point=np.asarray([s['query'] for s in data['samples'] if not s['fitting_sample']])
    model_path=args.cache/node['model']
    model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    m=np.load(model_path)
    normalized=((point-np.asarray(node['lower']))/(np.asarray(node['upper'])-node['lower'])).astype(np.float32)
    matrices=[];identity=np.eye(20,dtype=np.complex64)
    for q in normalized:
        bx,by,bz=[basis(q[a],m['nodes'],m['weights']) for a in range(3)]
        c=np.einsum('i,j,k,ijkr->r',bz,by,bx,m['coefficients'],optimize=True)
        h=unpack_hermitian(m['mean']+c@m['directions'],20)
        matrices.append(m['anchor']@np.linalg.solve(identity+1j*h,identity-1j*h))
    predictions=directory/'predictions.txt'
    with predictions.open('w') as f:
        f.write(str(len(matrices))+'\n')
        for q,s in zip(point,matrices):
            f.write(f'{q[2]:.17g} {q[0]:.17g} {q[1]:.17g} 400\n')
            for z in s.flat:f.write(f'{z.real:.17g} {z.imag:.17g}\n')
    with predictions.open() as inp:
        run=subprocess.run([str(args.validator.resolve())],stdin=inp,capture_output=True,text=True,check=True)
    report=json.loads(run.stdout)
    report.update(node=index,model_sha256=model_hash)
    if model_hash!=hashlib.sha256(model_path.read_bytes()).hexdigest():raise RuntimeError('Model changed')
    reports.append(report);print(json.dumps(report),flush=True)
maximum=max(r['maximum_power_column_l1_error'] for r in reports)
result=dict(scope=__doc__,cells=reports,maximum_power_error=maximum,
            seed_base=args.seed_base,
            tolerance=manifest['tolerance'],within_tolerance=maximum<=manifest['tolerance'],
            limitations='Same N16 modal truncation, random probes; no rigorous worst-case guarantee')
(args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
