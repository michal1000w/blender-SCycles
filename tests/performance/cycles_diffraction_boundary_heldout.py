#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Fresh parameter coverage for compensated-product-equivalent propagation.

CPU double product accumulation rounded to float, not a GPU execution. Gates
are unchanged. Fixed mean-permittivity internal basis; no oracle selection.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import cycles_diffraction_boundary_precision as precision
import cycles_diffraction_boundary_basis as basis
import cycles_diffraction_boundary_cascade as reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=928417)
    parser.add_argument('--cases',type=int,default=128)
    parser.add_argument('--wide-step',action='store_true')
    parser.add_argument('--half-orders',type=int,nargs='+',default=[2,4,8,16])
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing overwrite')
    if not args.half_orders or any(n<1 or n>128 for n in args.half_orders):
        parser.error('Half-orders must be between 1 and 128')
    root=Path(__file__).resolve().parent
    core=precision.load(root/'cycles_diffraction_boundary_cascade.py','heldout_core','wide_step' if args.wide_step else True,False)
    candidate_basis=precision.load(root/'cycles_diffraction_boundary_basis.py','heldout_basis',True,False)
    rng=np.random.default_rng(args.seed)
    cases=[]
    for i in range(args.cases):
        n=int(rng.choice(args.half_orders))
        pitch=float(rng.uniform(600,1800))
        wavelength=float(rng.uniform(380,780))
        depth=float(rng.uniform(20,1200))
        duty=float(rng.uniform(.2,.8))
        kx0=float(rng.uniform(-.35,.35))
        ky=float(rng.uniform(-.6,.6))
        ridge=(complex(rng.uniform(1.2,2.2)) if i%2==0 else
               complex(rng.uniform(.3,1.5),rng.uniform(3,8)))
        p,q=reference.matrices(n,pitch,wavelength,ridge,1.+0j,duty,kx0,ky)
        kx=kx0+(np.arange(2*n+1)-n)*wavelength/pitch
        air=basis.admittance(1.+0j,kx,ky)
        internal=basis.admittance(np.sqrt(duty*ridge**2+1-duty),kx,ky)
        thickness=2*np.pi*depth/wavelength
        expected=reference.modal_reference(p@air,np.linalg.solve(air,q),thickness)
        electric,outgoing=basis.flux_ports(kx,ky)
        expected_jones=outgoing@expected@electric
        row=dict(half_orders=n,pitch_nm=pitch,wavelength_nm=wavelength,depth_nm=depth,
                 duty=duty,kx=kx0,ky=ky,ridge=[ridge.real,ridge.imag])
        for name,dtype in [('double',np.complex128),('float',np.complex64)]:
            pp=core.diagnostic_matmul(p.astype(dtype),internal.astype(dtype))
            qq=np.linalg.solve(internal.astype(dtype),q.astype(dtype))
            s,steps=core.eigen_free(pp,qq,thickness,dtype)
            result=candidate_basis.exterior(s,internal,air,dtype)
            jones=outgoing@result@electric
            row[name]=dict(error=float(np.max(np.abs(result-expected))),
                           jones_error=float(np.max(np.abs(jones-expected_jones))),
                           power_column_l1=float(np.max(np.sum(np.abs(np.abs(jones)**2-np.abs(expected_jones)**2),axis=0))),
                           finite=bool(np.isfinite(result).all()),doublings=steps)
        cases.append(row)
    gates=dict(double=1e-9,float=3e-4)
    summary={name:dict(failures=sum(not c[name]['finite'] or c[name]['error']>=gate for c in cases),
                       max_error=max(c[name]['error'] for c in cases),
                       max_power_column_l1=max(c[name]['power_column_l1'] for c in cases))
             for name,gate in gates.items()}
    paths=[Path(__file__),Path(precision.__file__),Path(basis.__file__),Path(reference.__file__)]
    report=dict(scope=__doc__,half_orders=args.half_orders,wide_step=args.wide_step,seed=args.seed,gates=gates,cases=cases,summary=summary,
                source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    raise SystemExit(int(any(s['failures'] for s in summary.values())))


if __name__=='__main__': main()
