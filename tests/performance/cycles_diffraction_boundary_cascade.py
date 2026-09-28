#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Isolated eigen-free propagation experiment. Not a production/GPU cache builder.

Construct the same factorized isotropic RCWA P,Q matrices as the host solver.
Compare complete complex scattering operators in an air-admittance port basis;
this is an algebraic check, not a flux-normalized physical accuracy test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def matrices(n, pitch, wavelength, ridge, groove, duty, base, ky):
    count = 2*n+1
    eye = np.eye(count, dtype=complex)
    order = np.arange(count)
    indicator = duty*np.sinc((order[:, None]-order[None, :])*duty)
    er, eg = ridge**2, groove**2
    epsilon = (er-eg)*indicator+eg*eye
    inverse_e = np.linalg.solve(epsilon, eye)
    normal_e = np.linalg.solve((1/er-1/eg)*indicator+eye/eg, eye)
    k = np.diag(base+(order-n)*wavelength/pitch)
    p = np.block([[ky*k@inverse_e, eye-k@inverse_e@k],
                  [ky*ky*inverse_e-eye, -ky*inverse_e@k]])
    q = np.block([[-ky*k, k@k-epsilon],
                  [normal_e-ky*ky*eye, ky*k]])
    return p, q


def cascade(s, t):
    # Eliminate both shared port amplitudes; solve instead of forming inverses.
    a,b,c,d = s
    e,f,g,h = t
    eye = np.eye(a.shape[0], dtype=a.dtype)
    x = np.linalg.solve(eye-e@d, np.concatenate((e@c, f), axis=1))
    xc, xf = np.split(x, 2, axis=1)
    return a+b@xc, b@xf, g@(c+d@xc), h+g@d@xf


def eigen_free(p, q, thickness, dtype=np.complex128, trace=None):
    original_p, original_q = p, q
    p, q = p.astype(dtype), q.astype(dtype)
    m = p.shape[0]
    eye = np.eye(m, dtype=dtype)
    # In the artificial port basis a=(E+H)/2, b=(E-H)/2.
    generator = .5j*thickness*np.block([[p+q, q-p], [p-q, -p-q]])
    norm = float(np.linalg.norm(generator, 1))
    steps = max(0, int(np.ceil(np.log2(max(norm/1.0, 1)))))
    g = generator/(2**steps)
    term = np.eye(2*m, dtype=dtype)
    increment = np.zeros_like(term)
    for k in range(1, 21):
        term = term@g/k
        increment += term
    aa,b = increment[:m,:m], increment[:m,m:]
    c,dd = increment[m:,:m], increment[m:,m:]
    x = np.linalg.solve(eye+dd, np.concatenate((c, -dd), axis=1))
    dc, db = np.split(x, 2, axis=1)
    # Store transmission minus identity throughout; adding tiny increments to
    # identity at each thin slice destroys float precision after many doublings.
    a, b, c, d = -dc, db, aa-b@dc, b@(eye+db)
    regular = False
    for step in range(steps):
        if trace is not None:
            actual = np.block([[a,b if regular else eye+b],[c if regular else eye+c,d]])
            expected = modal_reference(original_p, original_q, thickness*2**(step-steps))
            trace.append(dict(completed_doublings=step, regular=regular,
                              max_error=float(np.max(np.abs(actual-expected))),
                              feedback_condition=float(np.linalg.cond(eye-a@d))))
        if regular or max(np.linalg.norm(b, 1), np.linalg.norm(c, 1)) > .25:
            if not regular:
                b, c = eye+b, eye+c
                regular = True
            a,b,c,d = cascade((a,b,c,d),(a,b,c,d))
            continue
        z = np.linalg.solve(eye-a@d, a@d)
        xc = (eye+z)@a@(eye+c)
        xf = (eye+z)@(eye+b)
        a,b,c,d = (a+(eye+b)@xc,
                   2*b+b@b+(eye+b)@z@(eye+b),
                   2*c+c@c+(eye+c)@d@xc,
                   d+(eye+c)@d@xf)
    return np.block([[a,b if regular else eye+b],[c if regular else eye+c,d]]), steps



def modal_reference(p, q, thickness):
    m = p.shape[0]
    values, w = np.linalg.eig(p@q)
    propagation = np.sqrt(values.astype(complex))
    propagation = np.where(propagation.imag < 0, -propagation, propagation)
    v = (q@w)/propagation[None,:]
    x = np.diag(np.exp(1j*thickness*propagation))
    # Decaying amplitudes are anchored at opposite boundaries, as in production.
    boundary = np.block([[w+v, (w-v)@x], [(w-v)@x, w+v]])
    solution = np.linalg.solve(boundary, 2*np.eye(2*m))
    output = np.block([[w-v, (w+v)@x], [(w+v)@x, w-v]])
    return .5*output@solution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite experiment')
    cases=[]
    for n in (2, 4, 8, 16):
        for ridge in (1.5+0j, .9+6j):
            for ky in (0., .27):
                for depth in (0., 150., 1200.):
                    p,q = matrices(n, 740., 580., ridge, 1.+0j, .41, -.2286, ky)
                    # Use air-wave admittance rather than an arbitrary identity
                    # magnetic basis. This aligns propagating/evanescent ports.
                    order = np.arange(2*n+1)-n
                    kx = -.2286+order*580./740.
                    z = np.sqrt((1-kx*kx-ky*ky).astype(complex))
                    y = np.block([[np.diag(-kx*ky/z),np.diag((kx*kx-1)/z)],
                                  [np.diag((1-ky*ky)/z),np.diag(kx*ky/z)]])
                    p,q = p@y, np.linalg.solve(y,q)
                    thickness = 2*np.pi*depth/580.
                    reference = modal_reference(p,q,thickness)
                    row = dict(half_orders=n, ridge=[ridge.real,ridge.imag],ky=ky,depth_nm=depth)
                    for name,dtype in [('double',np.complex128),('float',np.complex64)]:
                        result, steps = eigen_free(p,q,thickness,dtype)
                        error = float(np.max(np.abs(result-reference)))
                        row[name] = dict(max_complex_error=error, doublings=steps,
                                         finite=bool(np.isfinite(result).all()))
                    rounded_reference = modal_reference(p.astype(np.complex64).astype(complex),
                                                        q.astype(np.complex64).astype(complex), thickness)
                    if n == 16 and ridge.imag > 0 and ky == 0 and depth == 1200:
                        trace=[]
                        eigen_free(p,q,thickness,np.complex64,trace)
                        row['float_doubling_trace']=trace
                    row['input_quantization_error'] = float(np.max(np.abs(rounded_reference-reference)))
                    row['float_vs_rounded_modal_error'] = float(np.max(np.abs(result-rounded_reference)))
                    cases.append(row)
    report = dict(scope=__doc__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  numpy_version=np.__version__,cases=cases,
                  double_gate=1e-9,float_gate=3e-4)
    report['passed'] = all(c['double']['finite'] and c['float']['finite'] and
                           c['double']['max_complex_error']<report['double_gate'] and
                           c['float']['max_complex_error']<report['float_gate'] for c in cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],cases=len(cases),
                         maximum_double_error=max(c['double']['max_complex_error'] for c in cases),
                         maximum_float_error=max(c['float']['max_complex_error'] for c in cases))))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':
    main()
