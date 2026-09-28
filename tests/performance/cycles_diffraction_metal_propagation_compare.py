#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare Metal results with CPU float propagation and double material construction."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import cycles_diffraction_boundary_cascade as algebra
import cycles_diffraction_boundary_basis as basis
import cycles_diffraction_boundary_substrate as substrate
import cycles_diffraction_boundary_precision as precision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing overwrite')
    report=json.loads((args.suite/'report.json').read_text())
    core=precision.load(Path(algebra.__file__),'cpu_core','wide_step',False)
    rows=[]
    for i,c in enumerate(report['cases']):
        artifact=args.suite/f'case_{i}.npz'
        assert hashlib.sha256(artifact.read_bytes()).hexdigest()==report['artifact_sha256'][artifact.name]
        arrays=np.load(artifact)
        n=c['half_orders']; ridge=complex(*c['ridge']); index=c['incident_ior'];ky=c['ky']
        pitch=c.get('pitch_nm',740.); wavelength=c.get('wavelength_nm',580.)
        duty=c.get('duty',.41); base=c.get('kx',-.2286)
        p,q=algebra.matrices(n,pitch,wavelength,ridge,complex(index),duty,base,ky)
        kx=base+(np.arange(2*n+1)-n)*wavelength/pitch
        upper=basis.admittance(complex(index),kx,ky)
        lower=basis.admittance(ridge,kx,ky)
        internal=basis.admittance(np.sqrt(duty*ridge**2+(1-duty)*index**2),kx,ky)
        pp=core.diagnostic_matmul(p.astype(np.complex64),internal.astype(np.complex64))
        qq=np.linalg.solve(internal.astype(np.complex64),q.astype(np.complex64))
        s,_=core.eigen_free(pp,qq,2*np.pi*c['depth_nm']/wavelength,np.complex64)
        cpu=substrate.convert(s,internal,upper,lower,np.complex64,core.diagnostic_matmul)
        rows.append(dict(case=i,half_orders=n,depth_nm=c['depth_nm'],incident_ior=index,
                         gpu_error=c['error'],cpu_error=float(np.max(np.abs(cpu-arrays['reference']))),
                         gpu_cpu_error=float(np.max(np.abs(cpu-arrays['actual'])))))
    sources=[Path(__file__),Path(algebra.__file__),Path(basis.__file__),Path(substrate.__file__),Path(precision.__file__),args.suite/'report.json']
    args.output.write_text(json.dumps(dict(scope=__doc__,gpu_constitutive_in_source=report.get('gpu_constitutive',False),cases=rows,sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
