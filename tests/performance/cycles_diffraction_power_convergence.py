#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Physical power changes with modal truncation; no automatic convergence claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cycles_diffraction_reflection_power import physical,fresnel_check
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('fixtures',nargs='+',type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
groups={};checks=[]
keys=('ridge','groove','substrate','incident_ior','depth_nm','ky','pitch_nm','wavelength_nm','duty','kx','retained_half_orders')
for path in args.fixtures:
    for f in json.loads(path.read_text()):
        key=json.dumps({k:f[k] for k in keys},sort_keys=True)
        ports=f['ports'];size=2*len(ports)
        matrix=np.array([complex(*z) for z in f['matrix']]).reshape(size,size)
        substrate=complex(*f['substrate'])
        media=[substrate.real if side else f['incident_ior'] for order,side in ports]
        kx=[f['kx']+order*f['wavelength_nm']/f['pitch_nm'] for order,side in ports]
        power,normalized,selected=physical(matrix,media,kx,f['ky'])
        if not np.all(np.isfinite(power)):raise ValueError('Nonfinite power')
        labels=[ports[i] for i in selected]
        if f['half_orders'] in groups.setdefault(key,{}):raise ValueError('Duplicate profile/order')
        groups[key][f['half_orders']]=(power,labels)
        lossless=all(complex(*f[k]).imag==0 for k in ('ridge','groove','substrate'))
        gram=normalized.conj().T@normalized
        checks.append(dict(profile=json.loads(key),half_orders=f['half_orders'],
            lossless=lossless,max_output_power=float(power.sum(axis=0).max()),
            max_singular_power=float(np.linalg.norm(normalized,2)**2),
            unitarity_max_defect=float(np.max(np.abs(gram-np.eye(len(gram))))) if lossless else None))
comparisons=[]
for key,cases in groups.items():
    orders=sorted(cases)
    for lo,hi in zip(orders,orders[1:]):
        a,ports=cases[lo];b,other=cases[hi]
        if ports!=other:raise ValueError('Physical port mismatch')
        # Reflection/transmission totals distinguished for either incident side.
        worst_total=0.
        for i,(_,side) in enumerate(ports):
            reflected=[j for j,(_,s) in enumerate(ports) if s==side]
            transmitted=[j for j,(_,s) in enumerate(ports) if s!=side]
            for selected in (reflected,transmitted):
                worst_total=max(worst_total,abs(float(a[selected,i].sum()-b[selected,i].sum())))
        comparisons.append(dict(profile=json.loads(key),lower=lo,upper=hi,
            max_order_power_change=float(np.max(np.abs(a-b))),max_reflection_or_transmission_change=worst_total))
report=dict(scope='Production-double finite-order physical power diagnostics; no GPU speed or automatic convergence claim',
    fresnel_check=fresnel_check(),checks=checks,comparisons=comparisons,
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.fixtures})
args.output.write_text(json.dumps(report,indent=2))
