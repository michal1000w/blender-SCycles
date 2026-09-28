# SPDX-License-Identifier: Apache-2.0
"""Independent full visible native Glass sphere direct/R/TT/TRT/TRRT fields."""
import math
import numpy as np
from cycles_coherent_sphere_reference import sphere_path,spreading as r_spreading
from cycles_coherent_sphere_transmission_reference import basis
from cycles_coherent_sphere_internal_reference import inventory,field


def outside_segment(a,b,radius):
    d=b-a;A=np.dot(d,d);B=2*np.dot(a,d);C=np.dot(a,a)-radius*radius;disc=B*B-4*A*C
    if disc<=0:return True
    return not any(1e-7<t<1-1e-7 for t in ((-B-math.sqrt(disc))/(2*A),(-B+math.sqrt(disc))/(2*A)))


def all_fields(source,receiver,normal,radius,ior,max_internal,power):
    completed=[]
    direction=(receiver-source)/np.linalg.norm(receiver-source)
    if np.dot(direction,normal)<0 and outside_segment(source,receiver,radius):
        distance=np.linalg.norm(receiver-source);spread=-np.dot(direction,normal)/distance**2;modes=(np.eye(3)-np.outer(direction,direction))/math.sqrt(2)
        completed.append((modes.astype(complex)*math.sqrt(power*spread/(4*math.pi**2)),distance,'direct'))
    try:
        p=sphere_path(source,receiver,np.zeros(3),radius)
        if np.dot(p['outgoing'],normal)<0:
            incoming=p['incident'];outgoing=p['outgoing'];n=p['normal'];s=np.cross(incoming,n)
            if np.linalg.norm(s)<1e-12:s=basis(incoming)[:,0]
            s=s/np.linalg.norm(s);pi=np.cross(s,incoming);po=np.cross(s,outgoing);ci=abs(np.dot(incoming,n));ct=math.sqrt(1-(1-ci*ci)/(ior*ior));rs=(ci-ior*ct)/(ci+ior*ct);rp=(ior*ci-ct)/(ior*ci+ct)
            modes=(np.eye(3)-np.outer(incoming,incoming))/math.sqrt(2);modes=(modes@s)[:,None]*(rs*s)+(modes@pi)[:,None]*(rp*po)
            spread=float(r_spreading(p,*basis(normal).T,radius));completed.append((modes.astype(complex)*math.sqrt(power*spread/(4*math.pi**2)),float(p['optical_length']),'exterior_R'))
    except ValueError:pass
    for m in range(max_internal+1):
        for p in inventory(source,receiver,np.zeros(3),radius,ior,m):
            if np.dot(p['directions'][-1],normal)>=0:continue
            assert outside_segment(source,p['points'][0],radius) and outside_segment(p['points'][-1],receiver,radius)
            completed.append((field(p,normal,power),p['opl'],'T'+('R'*m)+'T'))
    return completed


def radiance(terms_by_source,phases,groups):
    terms=[(f,opl,label,phases[k],groups[k]) for k,source in enumerate(terms_by_source) for f,opl,label in source]
    result=sum(float(np.vdot(f,f).real) for f,*_ in terms)
    for j,(a,la,_,pa,ga) in enumerate(terms):
        for b,lb,_,pb,gb in terms[j+1:]:
            if ga!=gb:continue
            opd=la-lb;phase=2*math.pi*math.remainder(opd/550e-9,1)+pa-pb
            result+=2*math.exp(-.5*(opd/1e-4)**2)*float(np.real(np.sum(a*np.conj(b))*np.exp(1j*phase)))
    return result


