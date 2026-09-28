# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare frozen matrix-anchor data and expand the production Metal matcher."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/performance'))
from cycles_diffraction_tensor_fit import basis,unpack_hermitian
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--samples',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=np.load(a.model);data=json.loads(a.samples.read_text())
if not np.array_equal(m['lower'],data['lower']) or not np.array_equal(m['upper'],data['upper']):
    raise ValueError('Model and validation domain differ')
points=np.asarray([s['query'] for s in data['samples'] if not s['fitting_sample']])
q=((points-data['lower'])/(np.asarray(data['upper'])-data['lower'])).astype(np.float32)
with (a.output/'input.txt').open('w') as f:
    f.write(str(len(points))+'\n')
    for z in m['anchor'].flat:f.write(f'{z.real:.17g} {z.imag:.17g}\n')
    for point,t in zip(points,q):
        bx,by,bz=[basis(t[i],m['nodes'],m['weights']) for i in range(3)]
        c=np.einsum('i,j,k,ijkr->r',bz,by,bx,m['coefficients'],optimize=True)
        h=unpack_hermitian(m['mean']+c@m['directions'],20)
        f.write(' '.join(format(v,'.17g') for v in point)+'\n')
        for z in h.flat:f.write(f'{z.real:.17g} {z.imag:.17g}\n')
hashes={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in [a.model,a.samples,Path(__file__),root/'tests/performance/cycles_diffraction_tensor_fit.py']}
def expand(path,ancestors=()):
    path=path.resolve()
    if path in ancestors:raise RuntimeError('Include cycle')
    source=path.read_text();hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    def include(match):
        target=path.parent/match[1]
        if not target.is_file():target=root/'intern/cycles'/match[1]
        return expand(target,(*ancestors,path)) if target.is_file() else match[0]
    source=re.sub(r'^\s*#\s*include\s+"([^"]+)"[^\n]*$',include,source,flags=re.M)
    source=re.sub(r'^\s*#\s*pragma\s+once\s*$','',source,flags=re.M)
    guard='ANCHOR_TEST_'+hashlib.sha256(str(path).encode()).hexdigest()
    return f'#ifndef {guard}\n#define {guard}\n{source}\n#endif\n'
(a.output/'kernel.metal').write_text(expand(root/'tests/metal/cycles_diffraction_anchor_match.metal'))
(a.output/'sources.json').write_text(json.dumps(hashes,indent=2)+'\n')
