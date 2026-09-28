#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU prototype of float-only compensated complex matrix multiplication.

No FP64 arithmetic in compensated_matmul. Assumes finite normal-range inputs;
production would need overflow/subnormal guards and a validated Metal port.
"""
import numpy as np


def two_sum(a,b):
    s=a+b
    v=s-a
    return s,(a-(s-v))+(b-v)


def product(a,b):
    # Dekker split for binary32. A Metal FMA implementation could obtain the
    # product residual more cheaply, but has not been implemented/tested here.
    split=np.float32(4097.)
    ca=split*a
    cb=split*b
    ah=ca-(ca-a)
    bh=cb-(cb-b)
    al=a-ah
    bl=b-bh
    p=a*b
    error=((ah*bh-p)+ah*bl+al*bh)+al*bl
    return p,error


def accumulate(high,low,p,error):
    s,e=two_sum(high,p)
    return two_sum(s,low+(e+error))


def compensated_matmul(a,b):
    assert a.dtype==np.complex64 and b.dtype==np.complex64
    assert a.ndim==2 and b.ndim==2 and a.shape[1]==b.shape[0]
    rh=np.zeros((a.shape[0],b.shape[1]),np.float32)
    rl=np.zeros_like(rh)
    ih=np.zeros_like(rh)
    il=np.zeros_like(rh)
    for k in range(a.shape[1]):
        ar,ai=a[:,k:k+1].real,a[:,k:k+1].imag
        br,bi=b[k:k+1,:].real,b[k:k+1,:].imag
        p,e=product(ar,br)
        rh,rl=accumulate(rh,rl,p,e)
        p,e=product(ai,bi)
        rh,rl=accumulate(rh,rl,-p,-e)
        p,e=product(ar,bi)
        ih,il=accumulate(ih,il,p,e)
        p,e=product(ai,br)
        ih,il=accumulate(ih,il,p,e)
    result=np.empty(rh.shape,np.complex64)
    result.real=rh+rl
    result.imag=ih+il
    return result
