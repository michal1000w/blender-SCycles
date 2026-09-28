#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Capture actual linear systems from candidate diffraction propagation."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
import cycles_diffraction_boundary_cascade as reference
import cycles_diffraction_boundary_basis as basis
import cycles_diffraction_boundary_precision as precision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing overwrite')
    args.output.mkdir(parents=True)
    core=precision.load(Path(reference.__file__),'capture_core','wide_step',False)
    records=[]
    metadata=[]
    fixtures=[(8,1.5+0j,1.,600.,.27),(16,.9+6j,1.,1200.,0.),(32,.9+6j,1.58,150.,.27)]
    for fixture,(n,ridge,index,depth,ky) in enumerate(fixtures):
        p,q=reference.matrices(n,740.,580.,ridge,complex(index),.41,-.2286,ky)
        kx=-.2286+(np.arange(2*n+1)-n)*580./740.
        internal=basis.admittance(np.sqrt(.41*ridge**2+.59*index**2),kx,ky)
        def capture(a,b):
            assert a.dtype==np.complex64 and b.dtype==np.complex64
            expected=np.linalg.solve(a.astype(complex),b.astype(complex)).astype(np.complex64)
            records.append((np.concatenate((a,b),axis=1),expected))
            metadata.append(dict(fixture=fixture,rows=a.shape[0],rhs_columns=b.shape[1],
                                 condition_number=float(np.linalg.cond(a.astype(complex)))))
            return np.linalg.solve(a,b)
        core.diagnostic_solve=capture
        pp=core.diagnostic_matmul(p.astype(np.complex64),internal.astype(np.complex64))
        qq=capture(internal.astype(np.complex64),q.astype(np.complex64))
        core.eigen_free(pp,qq,2*np.pi*depth/580.,np.complex64)
    binary=args.output/'systems.bin'
    with binary.open('wb') as stream:
        stream.write(struct.pack('<I',len(records)))
        for (system,expected),meta in zip(records,metadata):
            stream.write(struct.pack('<II',meta['rows'],meta['rhs_columns']))
            stream.write(system.astype('<c8').tobytes())
            stream.write(expected.astype('<c8').tobytes())
    paths=[Path(__file__),Path(reference.__file__),Path(basis.__file__),Path(precision.__file__),binary]
    report=dict(scope=__doc__,records=metadata,fixtures=[dict(half_orders=n,ridge=[r.real,r.imag],
                incident_ior=i,depth_nm=d,ky=k) for n,r,i,d,k in fixtures],
                sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Captured {len(records)} systems; maximum condition {max(m["condition_number"] for m in metadata):.6g}')


if __name__=='__main__': main()
