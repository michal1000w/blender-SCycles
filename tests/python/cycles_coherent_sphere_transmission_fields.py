# SPDX-License-Identifier: Apache-2.0
"""Independent vector fields for isolated double-reference sphere TT paths.

No renderer imports, fitted brightness, image clamps or caustic regularization.
The source is the same three independent world-axis dipole ensemble as the
accepted planar tests. Geometric phase is -pi/2 per negative Hessian eigenvalue.
"""
import math
import numpy as np


def field(path, receiver_normal, source_power=1.0):
    from cycles_coherent_sphere_transmission_reference import spreading
    directions = path['directions']
    # One row per independent world-axis launch mode.
    modes = (np.eye(3)-np.outer(directions[0],directions[0])) / math.sqrt(2)
    ior = path['ior']
    for index, (n_in,n_out) in enumerate(((1.0,ior),(ior,1.0))):
        incoming,outgoing = directions[index:index+2]
        normal = path['normals'][index]
        s = np.cross(incoming,normal)
        if np.linalg.norm(s)<1e-12:
            s = path['frames'][index][:,0].copy()
            s -= incoming*np.dot(s,incoming)
        s = s/np.linalg.norm(s)
        p_in = np.cross(s,incoming)
        p_out = np.cross(s,outgoing)
        ci,ct = abs(np.dot(incoming,normal)),abs(np.dot(outgoing,normal))
        flux = math.sqrt(n_out*ct/(n_in*ci))
        ts = 2*n_in*ci/(n_in*ci+n_out*ct)*flux
        tp = 2*n_in*ci/(n_out*ci+n_in*ct)*flux
        modes = (modes@s)[:,None]*(ts*s)+(modes@p_in)[:,None]*(tp*p_out)
    geometric = spreading(path,receiver_normal)
    return modes.astype(complex)*math.sqrt(source_power*geometric/(4*math.pi**2))*np.exp(-.5j*math.pi*path['morse'])


def radiance(paths_by_source, receiver_normal, powers, phases, groups,
             wavelength=550e-9, coherence_length=1e-4, ior=1.5):
    completed=[]
    for source,paths in enumerate(paths_by_source):
        for path in paths:
            p=dict(path,ior=ior)
            completed.append((field(p,receiver_normal,powers[source]),p['opl'],
                              phases[source],groups[source]))
    value=sum(float(np.vdot(f,f).real) for f,_,_,_ in completed)
    for i,(a,la,pa,ga) in enumerate(completed):
        for b,lb,pb,gb in completed[i+1:]:
            if ga!=gb:continue
            opd=la-lb
            phase=2*math.pi*math.remainder(opd/wavelength,1.0)+pa-pb
            gamma=math.exp(-.5*(opd/coherence_length)**2)
            value+=2*gamma*float((np.sum(a*np.conj(b))*np.exp(1j*phase)).real)
    return value
