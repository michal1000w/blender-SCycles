#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare captured GPU reference operators in flux-normalized physical ports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cycles_diffraction_reflection_power import physical,fresnel_check
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path);parser.add_argument('fixtures',type=Path)
args=parser.parse_args();cases=[];hashes={}
for i,f in enumerate(json.loads(args.fixtures.read_text())):
    path=args.directory/f'case_{i}.bin';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    shape=tuple(np.fromfile(path,dtype='<u4',count=2));actual=np.fromfile(path,dtype='<c8',offset=8).reshape(shape).astype(complex)
    reference=np.array([complex(*z) for z in f['matrix']]).reshape(shape)
    ports=f['ports'];substrate=complex(*f['substrate'])
    media=[substrate.real if side else f['incident_ior'] for order,side in ports]
    kx=[f['kx']+order*f['wavelength_nm']/f['pitch_nm'] for order,side in ports]
    a,normalized,selected=physical(actual,media,kx,f['ky'])
    b,_,other=physical(reference,media,kx,f['ky'])
    if selected!=other or not np.all(np.isfinite(a)):raise ValueError('Invalid physical comparison')
    labels=[ports[j] for j in selected];total_error=0.
    for col,(_,side) in enumerate(labels):
        for reflected in (True,False):
            rows=[row for row,(_,out_side) in enumerate(labels) if (out_side==side)==reflected]
            total_error=max(total_error,abs(float(a[rows,col].sum()-b[rows,col].sum())))
    lossless=all(complex(*f[k]).imag==0 for k in ('ridge','groove','substrate'))
    gram=normalized.conj().T@normalized
    cases.append(dict(case=i,half_orders=f['half_orders'],lossless=lossless,
        max_order_power_error=float(np.max(np.abs(a-b))),max_rt_total_error=total_error,
        max_singular_power=float(np.linalg.norm(normalized,2)**2),
        max_total_power_defect=float(np.max(np.abs(a.sum(axis=0)-1))) if lossless else None,
        unitarity_defect=float(np.max(np.abs(gram-np.eye(len(gram))))) if lossless else None))
report=dict(scope='Same-truncation GPU vs production-double physical power; not modal convergence',
    fresnel_check=fresnel_check(),cases=cases,
    max_order_power_error=max(c['max_order_power_error'] for c in cases),
    max_rt_total_error=max(c['max_rt_total_error'] for c in cases),
    max_singular_power=max(c['max_singular_power'] for c in cases),
    max_lossless_unitarity_defect=max(c['unitarity_defect'] for c in cases if c['lossless']),
    captures=hashes,fixture_sha256=hashlib.sha256(args.fixtures.read_bytes()).hexdigest())
(args.directory/'power_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('cases','captures')},indent=2))
