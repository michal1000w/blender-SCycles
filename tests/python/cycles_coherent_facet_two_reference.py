#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent double unfolding for all visible direct/1R/ordered2R triangles."""
import itertools,math
import numpy as np
from cycles_coherent_facet_reference import frames,inside,visibility,pair_intensity


def geometry(reverse=False):
    # Finite positive-quadrant inner corner. Separate objects have two triangles each.
    x=np.array([[0,0,-.02],[0,.06,-.02],[0,.06,.04],[0,0,.04]],float)
    y=np.array([[0,0,-.02],[.06,0,-.02],[.06,0,.04],[0,0,.04]],float)
    triangles=np.array([x[[0,1,2]],x[[0,2,3]],y[[0,1,2]],y[[0,2,3]]])
    source=np.array([.025,.03,.01]);receiver=np.array([.035,.025,.015])
    if reverse:source=source[[1,0,2]];receiver=receiver[[1,0,2]]
    return triangles,source,receiver


def inventory(source,receiver,triangles,max_events=2):
    source=np.asarray(source,dtype=float);receiver=np.asarray(receiver,dtype=float);triangles=np.asarray(triangles,dtype=float)
    frame=frames(triangles);a,e,f,n,*_=frame;terms=[];normal=np.array([0.,0.,-1.])
    for count in range(max_events+1):
        for sequence in itertools.product(range(len(triangles)),repeat=count):
            if any(i==j for i,j in zip(sequence,sequence[1:])):continue
            images=[source]
            for index in sequence:images.append(images[-1]-2*np.dot(images[-1]-a[index],n[index])*n[index])
            current=receiver;points=[];valid=True
            for step in reversed(range(count)):
                index=sequence[step];virtual=images[step+1];delta=current-virtual;den=np.dot(delta,n[index])
                if abs(den)<1e-14:valid=False;break
                t=np.dot(a[index]-virtual,n[index])/den
                if not 0<t<1:valid=False;break
                point=virtual+t*delta
                if not inside(point[None],tuple(v[index:index+1] for v in frame))[0]:valid=False;break
                points.insert(0,point);current=point
            if not valid:continue
            vertices=np.array([source,*points,receiver]);lengths=np.linalg.norm(np.diff(vertices,axis=0),axis=1)
            if np.any(lengths<1e-10) or not visibility(vertices[:-1],vertices[1:],frame).all():continue
            directions=np.diff(vertices,axis=0)/lengths[:,None];cosine=-np.dot(directions[-1],normal)
            if cosine<=0:continue
            modes=(np.eye(3)-np.outer(directions[0],directions[0]))/math.sqrt(2)
            for step,index in enumerate(sequence):
                H=np.eye(3)-2*np.outer(n[index],n[index]);assert np.linalg.norm(directions[step]@H-directions[step+1])<1e-11;modes=modes@H
            opl=float(lengths.sum());assert abs(opl-np.linalg.norm(receiver-images[-1]))<1e-12
            terms.append({'facet':sequence,'points':points,'opl':opl,'modes':modes,'spread':float(cosine/opl**2)})
    return terms


def self_check():
    for reverse in [False,True]:
        triangles,s,r=geometry(reverse);paths=inventory(s,r,triangles);orders=[tuple(i//2 for i in p['facet']) for p in paths if len(p['facet'])==2];assert orders==[(1,0) if reverse else (0,1)],orders;assert len(paths)==4
        s0=s+[0,-.00001,0];s1=s+[0,.00001,0];both=[inventory(s0,r,triangles),inventory(s1,r,triangles)]
        for phase in [0,math.pi]:
            full,_=pair_intensity(both,[.001,.001],[0,phase],[1,1],coherence_length=1.)
            omitted,_=pair_intensity([[p for p in ps if len(p['facet'])<2] for ps in both],[.001,.001],[0,phase],[1,1],coherence_length=1.)
            print('reverse',reverse,'order',orders,'phase',phase,'full',full,'omit2R',omitted,'response',full-omitted)


if __name__=='__main__':self_check()
