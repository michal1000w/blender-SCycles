# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare fixed-chart and direct-operator interpolation; no production promotion.

Direct scattering interpolation generally fails passivity. It is included only
to distinguish chart-coordinate singularities from physical response variation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from cycles_diffraction_tensor_fit import basis

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory',type=Path,required=True)
p.add_argument('--validator',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
reports=[]
for index in [0,4,9,17,30]:
    directory=args.directory/f'node_{index:05d}'
    data=json.loads((directory/'samples.json').read_text())
    model=np.load(directory/'model.npz')
    raw=np.asarray([s['matrix'] for s in data['samples']])
    matrices=(raw[...,0]+1j*raw[...,1]).reshape(-1,20,20)
    training=np.asarray([s['fitting_sample'] for s in data['samples']])
    query=np.asarray([s['query'] for s in data['samples']])
    normalized=(query-data['lower'])/(np.asarray(data['upper'])-data['lower'])
    condition=np.linalg.cond(np.eye(20)+matrices/model['phase'])
    grid=matrices[training].reshape(7,7,7,20,20)
    predictions=[]
    for point in normalized[~training]:
        bx,by,bz=[basis(point[a],model['nodes'].astype(float),model['weights'].astype(float)) for a in range(3)]
        predictions.append(np.einsum('i,j,k,ijkab->ab',bz,by,bx,grid,optimize=True))
    predictions=np.asarray(predictions)
    path=args.output/f'node_{index:05d}_predictions.txt'
    with path.open('w') as f:
        f.write(f'{len(predictions)}\n')
        for point,matrix in zip(query[~training],predictions):
            f.write(f'{point[2]:.17g} {point[0]:.17g} {point[1]:.17g} 400\n')
            for z in matrix.flat:f.write(f'{z.real:.17g} {z.imag:.17g}\n')
    with path.open() as f:
        result=subprocess.run([str(args.validator.resolve())],stdin=f,capture_output=True,text=True,check=True)
    power=json.loads(result.stdout)
    reports.append(dict(node=index,maximum_training_chart_condition=float(condition[training].max()),
                        maximum_validation_chart_condition=float(condition[~training].max()),
                        direct_operator_error=float(np.max(np.linalg.norm(predictions-matrices[~training],axis=(1,2)))),
                        chart_operator_error=json.loads((directory/'fit.json').read_text())['maximum_held_out_frobenius_error'],
                        direct_physical_validation=power))
report=dict(scope=__doc__,nodes=reports)
(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
