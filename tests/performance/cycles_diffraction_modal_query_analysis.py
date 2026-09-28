#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare complete physical operators for one modal-convergence query."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--compare-to',type=Path,
               help='Compare each completed resolution with another solver at the same query')
a=p.parse_args()
d=json.loads(a.input.read_text())
runs={}
errors=[]
for row in d['runs']:
    if 'error' in row:
        errors.append(row)
        continue
    count=len(row['ports'])
    matrix=[complex(*v) for v in row['matrix']]
    if len(matrix)!=4*count*count or any(not math.isfinite(v.real) or not math.isfinite(v.imag) for v in matrix):
        raise ValueError('Invalid scattering matrix')
    powers=[[sum(abs(matrix[(2*r+i)*2*count+2*c+j])**2 for i in range(2) for j in range(2))*.5
             for c in range(count)] for r in range(count)]
    runs[row['half_orders']] = dict(row=row,matrix=matrix,powers=powers)
comparisons=[]
for lo,hi in itertools.combinations(sorted(runs),2):
    left,right=runs[lo],runs[hi]
    if left['row']['ports']!=right['row']['ports']:
        raise ValueError('Modal truncation changed physical port topology')
    ports=left['row']['ports']
    columns=[sum(abs(left['powers'][r][c]-right['powers'][r][c]) for r in range(len(ports)))
             for c in range(len(ports))]
    worst=max(range(len(ports)),key=columns.__getitem__) if ports else None
    comparisons.append(dict(lower=lo,higher=hi,maximum_power_column_l1=max(columns,default=0),
                            worst_incident_port=ports[worst] if worst is not None else None,
                            complex_frobenius_difference=math.sqrt(sum(abs(x-y)**2 for x,y in zip(left['matrix'],right['matrix'])))))
report=dict(wavelength_nm=d['wavelength_nm'],kx=d['kx'],ky=d['ky'],comparisons=comparisons,
            failed_resolutions=errors,
            input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
            scope='Complete physical-operator differences at one query; no absolute truncation-error bound.')
if a.compare_to:
    other=json.loads(a.compare_to.read_text())
    if any(d[k]!=other[k] for k in ('wavelength_nm','kx','ky')):
        raise ValueError('Solver comparison query mismatch')
    others={r['half_orders']:r for r in other['runs']}
    if len(others)!=len(other['runs']) or set(others)!=set(r['half_orders'] for r in d['runs']):
        raise ValueError('Solver comparison resolution mismatch')
    solver_comparisons=[]
    for modes,left in sorted(runs.items()):
        right=others[modes]
        if 'error' in right or left['row']['ports']!=right['ports']:
            raise ValueError('Solver comparison failure or port topology mismatch')
        matrix=[complex(*v) for v in right['matrix']]
        count=len(right['ports'])
        if len(matrix)!=len(left['matrix']) or any(
                not math.isfinite(v.real) or not math.isfinite(v.imag) for v in matrix):
            raise ValueError('Invalid comparison scattering matrix')
        columns=[sum(abs(left['powers'][r][c]-.5*sum(
                     abs(matrix[(2*r+i)*2*count+2*c+j])**2
                     for i in range(2) for j in range(2))) for r in range(count))
                 for c in range(count)]
        solver_comparisons.append(dict(half_orders=modes,
            maximum_complex_difference=max((abs(x-y) for x,y in zip(left['matrix'],matrix)),default=0),
            maximum_power_column_l1=max(columns,default=0)))
    report['solver_comparisons']=solver_comparisons
    report['comparison_sha256']=hashlib.sha256(a.compare_to.read_bytes()).hexdigest()
    report['solver_comparison_tolerance']=1e-9
    report['solver_comparison_passed']=not errors and all(
        r['maximum_complex_difference']<=1e-9 and r['maximum_power_column_l1']<=1e-9
        for r in solver_comparisons)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if a.compare_to and not report['solver_comparison_passed']:
    raise SystemExit(1)
