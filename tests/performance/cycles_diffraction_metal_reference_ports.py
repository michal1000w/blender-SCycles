#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Metal cache-reference ports against production C++ cutoff fixtures.

Every candidate complex64 matrix product and linear solve executes on Metal;
its result feeds subsequent steps. Material assembly, scalar/elementwise array
operations and orchestration remain on CPU. Not a GPU cache builder/benchmark.
"""
import argparse,hashlib,json,struct,subprocess
from pathlib import Path
import numpy as np
import cycles_diffraction_boundary_cascade as algebra
import cycles_diffraction_boundary_basis as basis
import cycles_diffraction_boundary_substrate as substrate
import cycles_diffraction_boundary_precision as precision


def reference_admittance(index,kx,ky,n,retained):
    epsilon=index*index
    selected=np.abs(np.arange(len(kx))-n)<=retained
    z=np.sqrt((epsilon-kx*kx-ky*ky).astype(complex))
    z=np.where(selected,1.+0j,z)
    y=np.block([[np.diag(-kx*ky/z),np.diag((kx*kx-epsilon)/z)],
                [np.diag((epsilon-ky*ky)/z),np.diag(kx*ky/z)]])
    count=len(kx)
    for m in np.flatnonzero(selected):
        y[m,m]=y[count+m,count+m]=0
        y[m,count+m]=-1
        y[count+m,m]=1
    return y


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--shader',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spectral-random',action='store_true')
    parser.add_argument('--random-cases',type=int,default=0)
    parser.add_argument('--seed',type=int,default=841291)
    parser.add_argument('--gpu-constitutive',action='store_true')
    parser.add_argument('--refinement',type=int,default=0)
    parser.add_argument('--half-orders',type=int,nargs='+',default=[8,16])
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing overwrite')
    if args.refinement<0 or args.refinement>4: parser.error('Refinement must be in 0..4')
    if any(n<1 or n>128 for n in args.half_orders): parser.error('Half-orders must be in 1..128')
    args.output.mkdir(parents=True)
    paths=[Path(__file__),Path(algebra.__file__),Path(basis.__file__),Path(substrate.__file__),
           Path(precision.__file__),args.executable,args.shader,
           args.shader.with_suffix('.mm'),args.fixtures]
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output/'provenance.json').write_text(json.dumps(hashes,indent=2)+'\n')
    stderr=(args.output/'device.log').open('wb')
    process=subprocess.Popen([str(args.executable),str(args.shader)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr)
    counts=[0,0,0]
    def request(operation,a,b,initial=None):
        assert a.dtype==np.complex64 and b.dtype==np.complex64
        assert a.ndim==b.ndim==2 and a.shape[1]==b.shape[0]
        header=struct.pack('<4I',operation,a.shape[0],a.shape[1],b.shape[1])
        process.stdin.write(header+a.astype('<c8').tobytes()+b.astype('<c8').tobytes())
        if initial is not None:
            process.stdin.write(initial.astype('<c8').tobytes())
        process.stdin.flush()
        length=4+a.shape[0]*b.shape[1]*8
        data=bytearray()
        while len(data)<length:
            part=process.stdout.read(length-len(data))
            if not part: raise RuntimeError('GPU process ended before response; see device.log')
            data.extend(part)
        code=struct.unpack_from('<I',data)[0]
        if code!=1: raise RuntimeError(f'GPU matrix operation failed with status {code}')
        counts[operation]+=1
        return np.frombuffer(data,offset=4,dtype='<c8').reshape(a.shape[0],b.shape[1]).copy()
    def matmul(a,b):
        if np.result_type(a,b)==np.complex64: return request(0,a,b)
        raise TypeError('Candidate matrix product unexpectedly promoted beyond float')
    def solve(a,b):
        if np.result_type(a,b)==np.complex64:
            x=request(1,a,b)
            for _ in range(args.refinement):
                residual=request(2,a,x,b)
                x=x+request(1,a,residual)
            return x
        raise TypeError('Candidate linear solve unexpectedly promoted beyond float')
    def material_matrices(n,ridge,index,kx,ky,duty):
        count=2*n+1
        identity=np.eye(count,dtype=np.complex64)
        order=np.arange(count)
        indicator=(duty*np.sinc((order[:,None]-order[None,:])*duty)).astype(np.float32)
        er,eg=np.complex64(ridge*ridge),np.complex64(index*index)
        epsilon=(er-eg)*indicator+eg*identity
        inverse_e=solve(epsilon,identity)
        inverse_profile=(np.complex64(1)/er-np.complex64(1)/eg)*indicator+identity/eg
        normal_e=solve(inverse_profile,identity)
        k=np.diag(kx.astype(np.complex64))
        ki=matmul(k,inverse_e)
        ik=matmul(inverse_e,k)
        p=np.block([[np.float32(ky)*ki,identity-matmul(ki,k)],
                    [np.float32(ky*ky)*inverse_e-identity,-np.float32(ky)*ik]])
        q=np.block([[-np.float32(ky)*k,matmul(k,k)-epsilon],
                    [normal_e-np.float32(ky*ky)*identity,np.float32(ky)*k]])
        assert p.dtype==q.dtype==np.complex64
        return p,q
    core=precision.load(Path(algebra.__file__),'gpu_core','wide_step',False)
    conversion=precision.load(Path(substrate.__file__),'gpu_conversion',True,False)
    core.diagnostic_matmul=matmul
    core.diagnostic_solve=solve
    conversion.diagnostic_solve=solve
    cases=[]
    fixtures=json.loads(args.fixtures.read_text())
    profiles=[(1.5+0j,1.,600.,.27),(.9+6j,1.,1200.,0.),(.9+6j,1.58,150.,.27)]
    if args.random_cases:
        if args.random_cases<1 or args.random_cases>1024: parser.error('Random cases must be in 1..1024')
        rng=np.random.default_rng(args.seed)
        profiles=[]
        for i in range(args.random_cases):
            ridge=(complex(rng.uniform(1.2,2.2)) if i%2==0 else
                   complex(rng.uniform(.3,1.5),rng.uniform(3,8)))
            profiles.append((ridge,float(rng.uniform(1,1.6)),float(rng.uniform(20,1200)),float(rng.uniform(-.6,.6))))
    spectral_rng=np.random.default_rng(args.seed+1)
    profiles=[(*profile, *(tuple(float(v) for v in
              [spectral_rng.uniform(600,1800),spectral_rng.uniform(380,780),spectral_rng.uniform(.2,.8),spectral_rng.uniform(-.35,.35)])
              if args.spectral_random else (740.,580.,.41,-.2286))) for profile in profiles]
    try:
        for fixture in fixtures:
            n=fixture['half_orders']; ridge=complex(*fixture['ridge'])
            index=fixture['incident_ior'];depth=fixture['depth_nm'];ky=fixture['ky']
            pitch=fixture['pitch_nm'];wavelength=fixture['wavelength_nm']
            duty=fixture['duty'];base=fixture['kx'];retained=fixture['retained_half_orders']
            p,q=algebra.matrices(n,pitch,wavelength,ridge,complex(index),duty,base,ky)
            kx=base+(np.arange(2*n+1)-n)*wavelength/pitch
            upper=reference_admittance(complex(index),kx,ky,n,retained)
            lower=reference_admittance(ridge,kx,ky,n,retained) if ridge.imag==0 else basis.admittance(ridge,kx,ky)
            internal=basis.admittance(np.sqrt(duty*ridge**2+(1-duty)*index**2),kx,ky)
            dimension=2*len(fixture['ports'])
            expected=np.array([complex(*v) for v in fixture['matrix']]).reshape(dimension,dimension)
            if args.gpu_constitutive:
                p,q=material_matrices(n,ridge,index,kx,ky,duty)
            pp=matmul(p.astype(np.complex64),internal.astype(np.complex64))
            qq=solve(internal.astype(np.complex64),q.astype(np.complex64))
            s,steps=core.eigen_free(pp,qq,2*np.pi*depth/wavelength,np.complex64)
            actual=conversion.convert(s,internal,upper,lower,np.complex64,matmul)
            modes=p.shape[0];count=2*n+1
            selected=[side*modes+component*count+order+n
                      for order,side in fixture['ports'] for component in (0,1)]
            actual=actual[np.ix_(selected,selected)]
            error=float(np.max(np.abs(actual-expected)))
            entry=dict(pitch_nm=pitch,wavelength_nm=wavelength,duty=duty,kx=base,half_orders=n,ridge=[ridge.real,ridge.imag],incident_ior=index,depth_nm=depth,ky=ky,
                       error=error,reference_ports=fixture['ports'],
                       finite=bool(np.isfinite(actual).all()),doublings=steps)
            cases.append(entry)
            np.savez(args.output/f'case_{len(cases)-1}.npz',actual=actual,reference=expected)
            print(json.dumps(entry),flush=True)
    finally:
        process.stdin.close()
        code=process.wait(timeout=60)
        stderr.close()
    if code: raise RuntimeError(f'GPU process exit {code}')
    for name,h in hashes.items(): assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    failures=sum(not c['finite'] or c['error']>=3e-4 for c in cases)
    report=dict(scope=__doc__,spectral_random=args.spectral_random,seed=args.seed,random_cases_per_order=args.random_cases,cases=cases,failures=failures,component_gate=3e-4,
                gpu_constitutive=args.gpu_constitutive,refinement_steps=args.refinement,gpu_residual_calls=counts[2],gpu_product_calls=counts[0],gpu_solve_calls=counts[1],sources_verified=True,
                artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.npz')})
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(1 if failures else 0)


if __name__=='__main__': main()
