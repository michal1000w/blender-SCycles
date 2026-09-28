# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Systematic frozen-cache checks near Rayleigh cutoffs and adaptive cell faces.

Queries are rounded to float32 before both cache evaluation and direct reference
validation, so coordinate rounding is not confused with interpolation error.
This finite set is not a rigorous worst-case bound or modal convergence test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from cycles_diffraction_tensor_fit import basis,unpack_hermitian
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--cache',type=Path,required=True)
p.add_argument('--validator',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
manifest=json.loads((a.cache/'manifest.json').read_text())
if not manifest['complete']:raise ValueError('Incomplete tree')
a.output.mkdir(parents=True,exist_ok=False)
nodes=manifest['nodes'];lo=np.array(nodes[0]['lower']);hi=np.array(nodes[0]['upper'])
queries={}
def add(point,kind):
    q=np.asarray(point,dtype=np.float32)
    if np.all(q>=lo) and np.all(q<=hi):queries.setdefault(tuple(float(x) for x in q),set()).add(kind)
for wavelength in np.linspace(380,780,9):
    for b in np.linspace(-.5,.5,9):
        for order in range(-2,3):
            x=(b+order)*wavelength/740
            if abs(x)>1:continue
            y=np.sqrt(max(0,1-x*x))
            for sign in [-1,1]:
                center=np.float32(sign*y)
                for value in [np.nextafter(center,np.float32(-np.inf)),center,
                              np.nextafter(center,np.float32(np.inf)),
                              sign*y-1e-5,sign*y+1e-5,sign*y-1e-3,sign*y+1e-3]:
                    add([b,value,wavelength],'rayleigh')
for node in nodes:
    if node['status']!='accepted':continue
    l=np.array(node['lower']);h=np.array(node['upper'])
    for axis in range(3):
        other=[k for k in range(3) if k!=axis]
        for side in [l[axis],h[axis]]:
            for u in [.211324865405187,.5,.788675134594813]:
                for v in [.211324865405187,.5,.788675134594813]:
                    point=(l+h)/2
                    point[other[0]]=l[other[0]]+u*(h[other[0]]-l[other[0]])
                    point[other[1]]=l[other[1]]+v*(h[other[1]]-l[other[1]])
                    f=np.float32(side)
                    for coordinate in [np.nextafter(f,np.float32(-np.inf)),f,np.nextafter(f,np.float32(np.inf))]:
                        point[axis]=coordinate;add(point,'cell_face')
models={i:np.load(a.cache/n['model']) for i,n in enumerate(nodes) if n['status']=='accepted'}
hashes={str(a.cache/nodes[i]['model']):hashlib.sha256((a.cache/nodes[i]['model']).read_bytes()).hexdigest() for i in models}
identity=np.eye(20,dtype=np.complex64)
metadata=[]
# Validator caps each input at 10,000 queries; preserve smaller deterministic batches.
items=list(queries.items());reports=[]
for start in range(0,len(items),4096):
    batch=items[start:start+4096];path=a.output/f'predictions_{start:05d}.txt'
    with path.open('w') as stream:
        stream.write(str(len(batch))+'\n')
        for point,kinds in batch:
            index=0
            while nodes[index]['status']=='split':
                n=nodes[index];index=n['children'][int(point[n['axis']]>=n['middle'])]
            m=models[index];q=np.asarray(point,dtype=np.float32)
            t=(q-m['lower'])/(m['upper']-m['lower'])
            if np.any(t<0) or np.any(t>1):raise ValueError('Tree selected wrong cell')
            bx,by,bz=[basis(t[k],m['nodes'],m['weights']) for k in range(3)]
            c=np.einsum('i,j,k,ijkr->r',bz,by,bx,m['coefficients'],optimize=True)
            h=unpack_hermitian(m['mean']+c@m['directions'],20)
            s=m['anchor']@np.linalg.solve(identity+1j*h,identity-1j*h)
            if not np.isfinite(s).all():raise ValueError('Nonfinite prediction')
            stream.write(f'{point[2]:.17g} {point[0]:.17g} {point[1]:.17g} 400\n')
            for z in s.flat:stream.write(f'{z.real:.17g} {z.imag:.17g}\n')
            metadata.append(dict(query=point,kinds=sorted(kinds),node=index))
    with path.open() as inp:
        result=subprocess.run([str(a.validator.resolve())],stdin=inp,capture_output=True,text=True,check=True)
    reports.append(json.loads(result.stdout));print(json.dumps(reports[-1]),flush=True)
for path,digest in hashes.items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Model changed')
(a.output/'queries.json').write_text(json.dumps(metadata)+'\n')
maximum=max(r['maximum_power_column_l1_error'] for r in reports)
report=dict(scope=__doc__,queries=len(items),batches=reports,maximum_power_error=maximum,
            tolerance=manifest['tolerance'],within_tolerance=maximum<=manifest['tolerance'],model_hashes=hashes)
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
