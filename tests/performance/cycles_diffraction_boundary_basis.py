#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare internal admittances with all responses converted back to air ports.

CPU NumPy diagnostic only. Includes evanescent ports; no physical flux or
GPU performance claim. Reference and candidate use the same Fourier truncation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import cycles_diffraction_boundary_cascade as core


def admittance(index, kx, ky):
    epsilon = index*index
    z = np.sqrt((epsilon-kx*kx-ky*ky).astype(complex))
    z = np.where(z.imag < 0, -z, z)
    return np.block([[np.diag(-kx*ky/z),np.diag((kx*kx-epsilon)/z)],
                     [np.diag((epsilon-ky*ky)/z),np.diag(kx*ky/z)]])


def exterior(s, internal, external, dtype):
    internal, external = internal.astype(dtype), external.astype(dtype)
    ratio = np.linalg.solve(internal, external)
    eye = np.eye(ratio.shape[0], dtype=dtype)
    c, d = .5*(eye+ratio), .5*(eye-ratio)
    zero = np.zeros_like(c)
    c, d = np.block([[c,zero],[zero,c]]), np.block([[d,zero],[zero,d]])
    return np.linalg.solve(c-s@d, s@c-d)


def flux_ports(kx, ky):
    count=len(kx)
    selected=np.flatnonzero(kx*kx+ky*ky<1.)
    electric=np.zeros((2*count,2*len(selected)),complex)
    for port,m in enumerate(selected):
        z=np.sqrt(1-kx[m]**2-ky**2)
        tangent=np.array([kx[m],ky])
        field=(np.eye(2)-np.outer(tangent,tangent)/(1+z))/np.sqrt(z)
        electric[np.ix_([m,count+m],[2*port,2*port+1])]=field
    zero=np.zeros_like(electric)
    electric=np.block([[electric,zero],[zero,electric]])
    return electric, np.linalg.pinv(electric)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing overwrite')
    cases=[]
    for n in (2,4,8,16):
        for ridge in (1.5+0j,.9+6j):
            for ky in (0.,.27):
                for depth in (0.,150.,1200.):
                    p,q=core.matrices(n,740.,580.,ridge,1.+0j,.41,-.2286,ky)
                    kx=-.2286+(np.arange(2*n+1)-n)*580./740.
                    air=admittance(1.+0j,kx,ky)
                    thickness=2*np.pi*depth/580.
                    expected=core.modal_reference(p@air,np.linalg.solve(air,q),thickness)
                    electric, outgoing=flux_ports(kx,ky)
                    expected_jones=outgoing@expected@electric
                    row=dict(half_orders=n,ridge=[ridge.real,ridge.imag],ky=ky,depth_nm=depth,bases={})
                    for name,index in [('air',1.+0j),('weak_absorption',1.+.2j),
                                       ('absorbing',1.+1j),('mean_epsilon',np.sqrt(.41*ridge**2+.59+0j))]:
                        basis=admittance(index,kx,ky)
                        entry={}
                        for label,dtype in [('double',np.complex128),('float',np.complex64)]:
                            # Include the basis transforms in candidate precision.
                            pp=p.astype(dtype)@basis.astype(dtype)
                            qq=np.linalg.solve(basis.astype(dtype),q.astype(dtype))
                            s,steps=core.eigen_free(pp,qq,thickness,dtype)
                            result=exterior(s,basis,air,dtype)
                            jones=outgoing@result@electric
                            physical=dict(max_jones_error=float(np.max(np.abs(jones-expected_jones))),
                                          max_power_column_l1=float(np.max(np.sum(np.abs(np.abs(jones)**2-np.abs(expected_jones)**2),axis=0))),
                                          max_power_gain=float(np.linalg.svd(jones,compute_uv=False)[0]**2))
                            entry[label]=dict(physical=physical,error=float(np.max(np.abs(result-expected))),
                                              finite=bool(np.isfinite(result).all()),doublings=steps)
                        row['bases'][name]=entry
                    cases.append(row)
    gates=dict(double=1e-9,float=3e-4)
    summary={name:{label:dict(max_error=max(c['bases'][name][label]['error'] for c in cases),
                             max_jones_error=max(c['bases'][name][label]['physical']['max_jones_error'] for c in cases),
                             max_power_column_l1=max(c['bases'][name][label]['physical']['max_power_column_l1'] for c in cases),
                             failures=sum(not c['bases'][name][label]['finite'] or
                                          c['bases'][name][label]['error']>=gate for c in cases))
                   for label,gate in gates.items()} for name in cases[0]['bases']}
    report=dict(scope=__doc__,gates=gates,cases=cases,summary=summary,
                sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__),Path(core.__file__))})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    raise SystemExit(0 if any(all(v['failures']==0 for v in entry.values()) for entry in summary.values()) else 1)


if __name__=='__main__': main()
