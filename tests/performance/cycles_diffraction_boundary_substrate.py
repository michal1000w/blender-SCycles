#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Metal-substrate/covered-metal validation of the eigen-free CPU diagnostic.

No GPU solver claim. Comparison uses same finite Fourier truncation, with the
production-style decaying modal boundary equations as a separate reference.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import cycles_diffraction_boundary_cascade as algebra
import cycles_diffraction_boundary_basis as basis
import cycles_diffraction_boundary_precision as precision


def reference(p,q,upper,lower,thickness):
    m=p.shape[0]
    values,w=np.linalg.eig(p@q)
    z=np.sqrt(values.astype(complex))
    z=np.where(z.imag<0,-z,z)
    v=(q@w)/z[None,:]
    x=np.diag(np.exp(1j*thickness*z))
    uw,lw=upper@w,lower@w
    boundary=np.block([[v+uw,(-v+uw)@x],[(v-lw)@x,-v-lw]])
    zero=np.zeros_like(upper)
    rhs=np.block([[2*upper,zero],[zero,-2*lower]])
    solution=np.linalg.solve(boundary,rhs)
    return np.block([[w,w@x],[w@x,w]])@solution-np.eye(2*m)


def convert(s,internal,upper,lower,dtype,matmul):
    m=internal.shape[0]
    eye=np.eye(m,dtype=dtype)
    zero=np.zeros_like(eye)
    ratios=[np.linalg.solve(internal.astype(dtype),y.astype(dtype)) for y in (upper,lower)]
    cs=[.5*(eye+r) for r in ratios]
    ds=[.5*(eye-r) for r in ratios]
    c=np.block([[cs[0],zero],[zero,cs[1]]])
    d=np.block([[ds[0],zero],[zero,ds[1]]])
    return np.linalg.solve(c-matmul(s,d),matmul(s,c)-d)


def top_flux(kx,ky,index):
    count=len(kx)
    selected=np.flatnonzero(kx*kx+ky*ky<index*index)
    e=np.zeros((2*count,2*len(selected)),complex)
    for port,m in enumerate(selected):
        z=np.sqrt(index*index-kx[m]**2-ky**2)
        tangent=np.array([kx[m],ky])/index
        frame=(np.eye(2)-np.outer(tangent,tangent)/(1+z/index))/np.sqrt(z)
        e[np.ix_([m,count+m],[2*port,2*port+1])]=frame
    return e,np.linalg.pinv(e)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing overwrite')
    core=precision.load(Path(algebra.__file__),'substrate_core','wide_step',False)
    cases=[]
    for n in (16,32):
        for index in (1.,1.58):
            for pitch in (740.,1600.):
                for wavelength in (450.,580.,700.):
                    for ky in (0.,.27):
                        ridge=.9+6j
                        depth=150.
                        kx=-.2286+(np.arange(2*n+1)-n)*wavelength/pitch
                        p,q=algebra.matrices(n,pitch,wavelength,ridge,complex(index),.41,-.2286,ky)
                        upper=basis.admittance(complex(index),kx,ky)
                        lower=basis.admittance(ridge,kx,ky)
                        internal=basis.admittance(np.sqrt(.41*ridge**2+.59*index**2),kx,ky)
                        thickness=2*np.pi*depth/wavelength
                        expected=reference(p,q,upper,lower,thickness)
                        e,l=top_flux(kx,ky,index)
                        m=p.shape[0]
                        ref_jones=l@expected[:m,:m]@e
                        row=dict(half_orders=n,incident_ior=index,pitch_nm=pitch,wavelength_nm=wavelength,ky=ky)
                        for label,dtype in [('double',np.complex128),('float',np.complex64)]:
                            pp=core.diagnostic_matmul(p.astype(dtype),internal.astype(dtype))
                            qq=np.linalg.solve(internal.astype(dtype),q.astype(dtype))
                            s,steps=core.eigen_free(pp,qq,thickness,dtype)
                            actual=convert(s,internal,upper,lower,dtype,core.diagnostic_matmul)
                            jones=l@actual[:m,:m]@e
                            row[label]=dict(error=float(np.max(np.abs(actual-expected))),
                                            reflected_jones_error=float(np.max(np.abs(jones-ref_jones))),
                                            reflected_power_column_l1=float(np.max(np.sum(np.abs(np.abs(jones)**2-np.abs(ref_jones)**2),axis=0))),
                                            max_reflected_power_gain=float(np.linalg.svd(jones,compute_uv=False)[0]**2),
                                            finite=bool(np.isfinite(actual).all()),doublings=steps)
                        cases.append(row)
    gates=dict(double=1e-9,float=3e-4)
    summary={label:dict(failures=sum(not c[label]['finite'] or c[label]['error']>=gate for c in cases),
                        max_error=max(c[label]['error'] for c in cases),
                        max_reflected_power_column_l1=max(c[label]['reflected_power_column_l1'] for c in cases))
             for label,gate in gates.items()}
    sources=[Path(__file__),Path(algebra.__file__),Path(basis.__file__),Path(precision.__file__)]
    args.output.write_text(json.dumps(dict(scope=__doc__,cases=cases,gates=gates,summary=summary,
                       source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    raise SystemExit(int(any(s['failures'] for s in summary.values())))


if __name__=='__main__': main()
