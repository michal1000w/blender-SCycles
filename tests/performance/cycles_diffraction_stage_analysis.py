#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Localize response errors using exact captured GPU inputs; diagnostic only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cycles_diffraction_boundary_cascade import modal_reference
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('reference',type=Path)
args=parser.parse_args()
files={}
def read(name):
    path=args.directory/(name+'.bin');files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    shape=np.fromfile(path,dtype='<u4',count=2)
    data=np.fromfile(path,dtype='<c8',offset=8)
    return data.reshape(tuple(shape)).astype(np.complex128)
p,q,u,l,s,actual,retained=[read(name) for name in ('transformed_p','transformed_q','upper','lower','propagation','exterior','retained')]
m=p.shape[0];identity=np.eye(m);zero=np.zeros_like(identity)
c=np.block([[(identity+u)*.5,zero],[zero,(identity+l)*.5]])
d=np.block([[(identity-u)*.5,zero],[zero,(identity-l)*.5]])
def boundary(scattering):
    return np.linalg.solve(c-scattering@d,scattering@c-d)
# Match the float thickness supplied by the GPU host code.
thickness=float(np.float32(2*np.pi*750/600))
modal=modal_reference(p,q,thickness)
boundary_on_gpu=boundary(s)
all_double=boundary(modal)
count=m//2;half=(count-1)//2
indices=np.array([polarization*count+order+half for order in range(-6,7) for polarization in (0,1)])
def select(a):return a[np.ix_(indices,indices)]
reference=json.loads(args.reference.read_text())[0]['matrix']
reference=np.array([complex(*z) for z in reference]).reshape(retained.shape)
def metric(a,b):
    delta=np.abs(a-b)
    if not np.all(np.isfinite(delta)):raise ValueError('Nonfinite comparison')
    return dict(max_component=float(np.max(delta)),relative_frobenius=float(np.linalg.norm(a-b)/np.linalg.norm(b)))
report=dict(scope='Stage-local comparison on captured float inputs, not physical convergence or a production CPU fallback',
    propagation=metric(s,modal),boundary_arithmetic=metric(actual,boundary_on_gpu),
    retained_boundary_arithmetic=metric(select(actual),select(boundary_on_gpu)),
    retained_propagation_effect=metric(select(boundary_on_gpu),select(all_double)),
    total_gpu_to_original=metric(retained,reference),
    double_stages_on_captured_inputs_to_original=metric(select(all_double),reference),sources=files)
step_paths=list(args.directory.glob('propagation_step_*.bin'))
if step_paths:
    steps=max(int(path.stem.rsplit('_',1)[1]) for path in step_paths)
    report['doubling_stages']=[]
    for step in range(steps+1):
        captured=read('propagation_step_'+str(step))
        expected=modal_reference(p,q,thickness*2.0**(step-steps))
        report['doubling_stages'].append(dict(step=step,**metric(captured,expected)))
(args.directory/'analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
