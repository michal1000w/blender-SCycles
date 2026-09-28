#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Convert absorbing-substrate reference reflection to physical flux channels."""
import argparse
import json
from pathlib import Path
import numpy as np

J=np.array([[0.,-1.],[1.,0.]],dtype=complex)
def admittance(index,kx,ky):
    epsilon=index*index
    kz=np.sqrt(complex(epsilon-kx*kx-ky*ky))
    if kz.imag<0:kz=-kz
    if abs(kz)<1e-12:raise ValueError('Exact grazing port requires a limiting calculation')
    return np.array([[-kx*ky,kx*kx-epsilon],[epsilon-ky*ky,kx*ky]],complex)/kz

def physical(reference,index,kx,ky):
    count=len(kx);ratio=np.zeros(np.shape(reference),dtype=complex)
    indices=np.broadcast_to(np.asarray(index,dtype=float),(count,))
    if np.any(indices<=0):raise ValueError("Exterior indices must be positive and lossless")
    metrics=[];propagating=[]
    for i,x in enumerate(kx):
        medium=indices[i]
        y=admittance(medium,x,ky);g=-J@y
        ratio[2*i:2*i+2,2*i:2*i+2]=g
        metrics.append(g)
        if x*x+ky*ky<medium*medium:propagating.append(i)
    identity=np.eye(2*count);c=(identity+ratio)*.5;d=(identity-ratio)*.5
    result=np.linalg.solve(c-reference@d,reference@c-d)
    normalized=np.zeros((2*len(propagating),2*len(propagating)),complex)
    for oi,o in enumerate(propagating):
        vo,uo=np.linalg.eigh(metrics[o]);root=(uo*np.sqrt(vo))@uo.conj().T
        for ii,i in enumerate(propagating):
            vi,ui=np.linalg.eigh(metrics[i]);inverse=(ui/np.sqrt(vi))@ui.conj().T
            normalized[2*oi:2*oi+2,2*ii:2*ii+2]=root@result[2*o:2*o+2,2*i:2*i+2]@inverse
    power=np.array([[.5*np.linalg.norm(normalized[2*o:2*o+2,2*i:2*i+2])**2
                     for i in range(len(propagating))] for o in range(len(propagating))])
    return power,normalized,propagating

def fresnel_check():
    worst=0.
    for x in (0.,.3,.8):
        substrate=.9+6j;y=admittance(substrate,x,0)
        reference=np.linalg.solve(J+y,J-y)
        power,_,_=physical(reference,1.,[x],0.)
        z=np.sqrt(1-x*x);zs=np.sqrt(substrate*substrate-x*x)
        rs=(z-zs)/(z+zs);rp=(substrate*substrate*z-zs)/(substrate*substrate*z+zs)
        worst=max(worst,abs(power[0,0]-.5*(abs(rs)**2+abs(rp)**2)))
    # A zero-thickness reference interface swaps the two J-basis ports.
    swap=np.block([[np.zeros((2,2)),np.eye(2)],[np.eye(2),np.zeros((2,2))]])
    for x in (0.,.3,.8):
        power,normalized,_=physical(swap,[1.,1.5],[x,x],0.)
        z=np.sqrt(1-x*x);zs=np.sqrt(2.25-x*x)
        rs=(z-zs)/(z+zs);rp=(2.25*z-zs)/(2.25*z+zs)
        expected=.5*(rs*rs+rp*rp)
        worst=max(worst,abs(power[0,0]-expected),abs(power[1,0]-(1-expected)),
                  float(np.max(np.abs(normalized.conj().T@normalized-np.eye(4)))))
    if worst>1e-12:raise ValueError('Fresnel conversion check failed')
    return worst

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path);parser.add_argument('fixture',type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    f=json.loads(args.fixture.read_text())[0]
    if complex(*f['substrate']).imag<=0:raise ValueError('This diagnostic requires an absorbing substrate')
    orders=[p[0] for p in f['ports']]
    if any(p[1] for p in f['ports']):raise ValueError('Unexpected lower reference ports')
    shape=tuple(np.fromfile(args.capture,dtype='<u4',count=2))
    actual=np.fromfile(args.capture,dtype='<c8',offset=8).reshape(shape).astype(complex)
    reference=np.array([complex(*z) for z in f['matrix']]).reshape(shape)
    kx=f['kx']+np.array(orders)*f['wavelength_nm']/f['pitch_nm']
    a,an,ports=physical(actual,f['incident_ior'],kx,f['ky'])
    b,bn,_=physical(reference,f['incident_ior'],kx,f['ky'])
    if not np.all(np.isfinite(a)):raise ValueError('Nonfinite power')
    report=dict(scope='Single absorbing-substrate profile, unpolarized propagating-order reflection; not modal convergence',
        fresnel_conversion_max_error=fresnel_check(),orders=[orders[i] for i in ports],
        max_order_power_error=float(np.max(np.abs(a-b))),
        max_total_reflection_error=float(np.max(np.abs(a.sum(axis=0)-b.sum(axis=0)))),
        gpu_reflected_power_by_incident_order=a.sum(axis=0).tolist(),
        reference_reflected_power_by_incident_order=b.sum(axis=0).tolist(),
        gpu_max_coherent_reflected_fraction=float(np.linalg.norm(an,2)**2),
        reference_max_coherent_reflected_fraction=float(np.linalg.norm(bn,2)**2))
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
