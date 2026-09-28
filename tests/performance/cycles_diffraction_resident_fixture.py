#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Export transformed RCWA matrices for resident propagation validation."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
import cycles_diffraction_boundary_cascade as core
import cycles_diffraction_boundary_basis as basis
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
if args.output.exists():p.error('Refusing overwrite')
args.output.mkdir(parents=True)
cases=[]
with (args.output/'fixture.bin').open('wb') as f:
 f.write(struct.pack('<I',12))
 for n in (8,16,32,64):
  for ridge,index,depth,ky in [(1.5+0j,1.,600.,.27),(.9+6j,1.,1200.,0.),(.9+6j,1.58,150.,.27)]:
   p,q=core.matrices(n,740.,580.,ridge,complex(index),.41,-.2286,ky)
   kx=-.2286+(np.arange(2*n+1)-n)*580./740.
   y=basis.admittance(np.sqrt(.41*ridge**2+.59*index**2),kx,ky)
   p=(p@y).astype('<c8');q=np.linalg.solve(y,q).astype('<c8')
   thickness=float(np.float32(2*np.pi*depth/580.))
   expected=core.modal_reference(p.astype(complex),q.astype(complex),thickness).astype('<c8')
   f.write(struct.pack('<If',p.shape[0],thickness));f.write(p.tobytes());f.write(q.tobytes());f.write(expected.tobytes())
   cases.append(dict(half_orders=n,ridge=[ridge.real,ridge.imag],incident_ior=index,depth_nm=depth,ky=ky))
paths=[Path(__file__),Path(core.__file__),Path(basis.__file__),args.output/'fixture.bin']
(args.output/'manifest.json').write_text(json.dumps(dict(scope=__doc__,cases=cases,sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
