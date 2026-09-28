#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent double-precision finite-triangle direct + one-R field inventory.

World dipole labels remain common across paths. Mirrors use Householder field
transport. Pair intensity uses the declared Gaussian gamma, without sampling.
"""
import math
import numpy as np


def geometry():
    source=np.array([-.02,0.,.025]);receiver=np.array([.02,0.,.03]);triangles=[]
    for y in np.linspace(-.014,.014,8):
        for x in np.linspace(-.016,.016,10):
            center=np.array([x,y,0.]);normal=(source-center)/np.linalg.norm(source-center)+(receiver-center)/np.linalg.norm(receiver-center);normal/=np.linalg.norm(normal)
            u=np.cross(normal,[0.,1.,0.]);u/=np.linalg.norm(u);v=np.cross(normal,u);half=.0007
            triangles.append([center+half*(-u-v),center+half*(u-v),center+half*(2*v)])
    return np.array(triangles),source,receiver


def frames(triangles):
    a=triangles[:,0];e=triangles[:,1]-a;f=triangles[:,2]-a;n=np.cross(e,f);n/=np.linalg.norm(n,axis=1)[:,None]
    ee=np.einsum('ij,ij->i',e,e);ef=np.einsum('ij,ij->i',e,f);ff=np.einsum('ij,ij->i',f,f)
    return a,e,f,n,ee,ef,ff,ee*ff-ef*ef


def inside(points,frame):
    a,e,f,n,ee,ef,ff,den=frame;delta=points-a
    de=np.einsum('...ij,ij->...i',delta,e);df=np.einsum('...ij,ij->...i',delta,f)
    u=(ff*de-ef*df)/den;v=(ee*df-ef*de)/den
    return (u>1e-9)&(v>1e-9)&(u+v<1-1e-9)


def visibility(starts,ends,frame):
    a,e,f,n,*_=frame;delta=ends-starts
    denominator=delta@n.T;numerator=np.einsum('pfi,fi->pf',a[None]-starts[:,None],n)
    with np.errstate(divide='ignore',invalid='ignore'):t=numerator/denominator
    points=starts[:,None]+t[:,:,None]*delta[:,None]
    hit=(t>1e-7)&(t<1-1e-7)&inside(points,frame)&np.isfinite(t)
    return ~hit.any(axis=1)


def paths(source,receiver,triangles,detector_normal=np.array([0.,0.,-1.])):
    frame=frames(triangles);a,e,f,n,*_=frame;images=source-2*np.einsum('ij,ij->i',source-a,n)[:,None]*n
    ray=receiver-images;den=np.einsum('ij,ij->i',ray,n)
    t=np.einsum('ij,ij->i',a-images,n)/den;points=images+t[:,None]*ray
    in_dir=points-source;out_dir=receiver-points;li=np.linalg.norm(in_dir,axis=1);lo=np.linalg.norm(out_dir,axis=1);in_dir/=li[:,None];out_dir/=lo[:,None]
    good=(t>0)&(t<1)&inside(points,frame)&(np.einsum('ij,ij->i',source-a,n)*np.einsum('ij,ij->i',receiver-a,n)>0)
    good &= visibility(np.broadcast_to(source,points.shape),points,frame)&visibility(points,np.broadcast_to(receiver,points.shape),frame)
    cosine=-out_dir@detector_normal;good &= cosine>0
    terms=[]
    for index in np.flatnonzero(good):
        opl=li[index]+lo[index];virtual_length=np.linalg.norm(receiver-images[index]);assert abs(opl-virtual_length)<1e-12
        reflected=in_dir[index]-2*np.dot(in_dir[index],n[index])*n[index];assert np.linalg.norm(reflected-out_dir[index])<1e-11
        modes=(np.eye(3)-np.outer(in_dir[index],in_dir[index]))/math.sqrt(2)
        modes=modes@(np.eye(3)-2*np.outer(n[index],n[index]))
        terms.append({'facet':int(index),'point':points[index],'opl':float(opl),'modes':modes,'spread':float(cosine[index]/opl**2)})
    delta=receiver-source;length=np.linalg.norm(delta);direction=delta/length;cosine=-np.dot(direction,detector_normal)
    if cosine>0 and visibility(source[None],receiver[None],frame)[0]:
        terms.insert(0,{'facet':None,'point':None,'opl':float(length),'modes':(np.eye(3)-np.outer(direction,direction))/math.sqrt(2),'spread':float(cosine/length**2)})
    return terms


def pair_intensity(inventories,powers,phases,groups,wavelength=550e-9,coherence_length=.02):
    terms=[(p['modes']*math.sqrt(power*p['spread']/(4*math.pi**2)),p['opl'],phase,group) for ps,power,phase,group in zip(inventories,powers,phases,groups) for p in ps]
    fields=np.array([p[0].reshape(-1) for p in terms]);lengths=np.array([p[1] for p in terms]);angles=np.array([math.tau*math.remainder((p[1]-lengths[0])/wavelength,1)+p[2] for p in terms]);groups=np.array([p[3] for p in terms]);delta=lengths[:,None]-lengths
    gram=fields@fields.T;gamma=np.exp(-.5*(delta/coherence_length)**2)*(groups[:,None]==groups)
    return float(np.sum(gram*gamma*np.cos(angles[:,None]-angles))),terms


def self_check():
    triangles,source,receiver=geometry();inventory=paths(source,receiver,triangles);assert len(inventory)==81
    value,terms=pair_intensity([inventory],[.001],[0],[1]);assert value>=0
    # Independent Gaussian quadrature of shared per-group phase, not pair gamma.
    z,w=np.polynomial.hermite.hermgauss(80);sampled=0.;anchor=terms[0][1]
    for x,weight in zip(z,w):
        field=sum(f*np.exp(1j*(math.tau*math.remainder((length-anchor)/550e-9,1)+math.sqrt(2)*x*(length-anchor)/.02)) for f,length,phase,group in terms)
        sampled+=weight/math.sqrt(math.pi)*float(np.vdot(field,field).real)
    assert abs(value-sampled)<1e-10,(value,sampled)
    print('80 mirror routes + direct; explicit pair and shared-Gaussian phase agree',value,abs(value-sampled))


if __name__=='__main__':self_check()
