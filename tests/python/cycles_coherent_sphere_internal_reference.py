# SPDX-License-Identifier: Apache-2.0
"""Independent all-winding scalar roots and OPL Hessian for sphere T-R^m-T.

Exterior endpoints, IOR>=1, isolated roots only. No renderer utility imports.
Grazing, folds and axial rings reject the inventory instead of dropping roots.
"""
import math
import numpy as np
from cycles_coherent_sphere_transmission_reference import basis,bisect


def inventory(source,receiver,center,radius,ior,reflections):
    source,receiver,center=(np.asarray(x,dtype=float) for x in (source,receiver,center))
    s=source-center;r=receiver-center;a=np.linalg.norm(s);b=np.linalg.norm(r)
    if reflections not in (0,1,2) or ior<1 or radius<=0 or min(a,b)<=radius:raise ValueError('invalid domain')
    e0=s/a;rd=r/b;cosine=np.clip(np.dot(e0,rd),-1,1);orth=rd-cosine*e0;sine=np.linalg.norm(orth);theta=math.atan2(sine,cosine)
    e1=orth/sine if sine>1e-12 else basis(e0)[:,0];A=radius/a;B=radius/b;k=reflections+1
    def delta(t):return k*math.pi+2*math.asin(t)-math.asin(A*t)-math.asin(B*t)-2*k*math.asin(t/ior)
    def derivative(t):
        v=max(0.,1-t*t)
        last=2*k if ior==1 else 2*k/ior*math.sqrt(v/(1-(t/ior)**2))
        return 2-A*math.sqrt(v/(1-(A*t)**2))-B*math.sqrt(v/(1-(B*t)**2))-last
    critical=bisect(derivative,0,1-1e-15) if derivative(0)<0 and derivative(1-1e-15)>0 else None
    bounds=[-1.,1.] if critical is None else [-1.,-critical,critical,1.]
    values=[delta(t) for t in bounds];lo=math.ceil((min(values)-theta)/(2*math.pi));hi=math.floor((max(values)-theta)/(2*math.pi));roots=[]
    for winding in range(lo,hi+1):
        def F(t):return delta(t)-theta-2*math.pi*winding
        for l,h in zip(bounds,bounds[1:]):
            if abs(F(l))<1e-12 or abs(F(h))<1e-12:raise ValueError('grazing or fold boundary')
            if F(l)*F(h)<0:roots.append((bisect(F,l,h),winding))
    if sine<1e-10 and any(abs(t)>1e-8 for t,w in roots):raise ValueError('axial ring')
    paths=[]
    for t,winding in sorted(roots):
        phi=math.asin(t)-math.asin(A*t);step=math.pi-2*math.asin(t/ior)
        normals=[e0*math.cos(phi+j*step)+e1*math.sin(phi+j*step) for j in range(k+1)]
        points=[center+radius*n for n in normals];vertices=[source,*points,receiver]
        lengths=np.array([np.linalg.norm(y-x) for x,y in zip(vertices,vertices[1:])]);directions=[(y-x)/l for x,y,l in zip(vertices,vertices[1:],lengths)];weights=np.array([1,*([ior]*k),1]);frames=[basis(n) for n in normals]
        gradients=[weights[j]*directions[j]-weights[j+1]*directions[j+1] for j in range(k+1)]
        residual=max(np.linalg.norm(f.T@g) for f,g in zip(frames,gradients))
        if residual>1e-9:raise AssertionError(('Snell/reflection reconstruction',t,winding,residual))
        P=[np.eye(3)-np.outer(d,d) for d in directions];H=np.zeros((2*(k+1),2*(k+1)))
        for j in range(k+1):
            index=slice(2*j,2*j+2);H[index,index]=frames[j].T@(weights[j]*P[j]/lengths[j]+weights[j+1]*P[j+1]/lengths[j+1])@frames[j]-np.eye(2)*np.dot(gradients[j],normals[j])/radius
            if j<k:
                following=slice(2*j+2,2*j+4);H[index,following]=-weights[j+1]/lengths[j+1]*frames[j].T@P[j+1]@frames[j+1];H[following,index]=H[index,following].T
        eigen=np.linalg.eigvalsh(H)
        if min(abs(eigen))/max(abs(eigen))<1e-9:raise ValueError('caustic singular Hessian')
        paths.append(dict(t=t,winding=winding,points=points,normals=normals,frames=frames,length=lengths,directions=directions,weights=weights,H=H,morse=int((eigen<0).sum()),opl=float(np.dot(weights,lengths)),snell_residual=residual,reflections=reflections))
    return paths


def spreading(path,receiver_normal):
    d=path['directions'];l=path['length'];frames=path['frames'];n=len(frames);rb=basis(receiver_normal);rhs=np.zeros((2*n,2));rhs[-2:]=frames[-1].T@(np.eye(3)-np.outer(d[-1],d[-1]))@rb/l[-1];q=np.linalg.solve(path['H'],rhs);D=(np.eye(3)-np.outer(d[0],d[0]))@frames[0]@q[:2]/l[0]
    return float(np.linalg.norm(np.cross(D[:,0],D[:,1])))


def field(path,receiver_normal,source_power=1.):
    """Flux-normalized dielectric Jones transport, including signed internal R."""
    directions=path['directions'];m=path['reflections'];ior=float(path['weights'][1])
    modes=(np.eye(3)-np.outer(directions[0],directions[0]))/math.sqrt(2)
    for j,normal in enumerate(path['normals']):
        incoming,outgoing=directions[j:j+2];s=np.cross(incoming,normal)
        if np.linalg.norm(s)<1e-12:s=path['frames'][j][:,0]
        s=s/np.linalg.norm(s);pi=np.cross(s,incoming);po=np.cross(s,outgoing)
        ci=abs(np.dot(incoming,normal))
        if j in (0,m+1):
            ni,no=path['weights'][j:j+2];ct=abs(np.dot(outgoing,normal));flux=math.sqrt(no*ct/(ni*ci))
            fs=2*ni*ci/(ni*ci+no*ct)*flux;fp=2*ni*ci/(no*ci+ni*ct)*flux
        else:
            # The exterior-entry angular momentum bounds sin(theta_inside)<=1/n;
            # therefore isolated exterior branches cannot have internal TIR.
            ct=math.sqrt(max(0.,1-ior*ior*(1-ci*ci)))
            fs=(ior*ci-ct)/(ior*ci+ct);fp=(ci-ior*ct)/(ci+ior*ct)
        modes=(modes@s)[:,None]*(fs*s)+(modes@pi)[:,None]*(fp*po)
    return modes.astype(complex)*math.sqrt(source_power*spreading(path,receiver_normal)/(4*math.pi**2))*np.exp(-.5j*math.pi*path['morse'])


def checks():
    results=[]
    for m in (1,2):
        for theta in (.4,1.1,2.4,3.05):
            source=np.array([2.,0,.01]);receiver=np.array([2*math.cos(theta),2*math.sin(theta),.03]);normal=-receiver/np.linalg.norm(receiver)
            ps=inventory(source,receiver,np.zeros(3),1.,1.5,m);errors=[];h=1e-5
            for index,p in enumerate(ps):
                deriv=[]
                for direction in basis(normal).T:
                    a=inventory(source,receiver+h*direction,np.zeros(3),1.,1.5,m);b=inventory(source,receiver-h*direction,np.zeros(3),1.,1.5,m)
                    assert len(a)==len(b)==len(ps)
                    deriv.append((a[index]['directions'][0]-b[index]['directions'][0])/(2*h))
                fd=np.linalg.norm(np.cross(*deriv));analytic=spreading(p,normal);errors.append(abs(fd/analytic-1))
            assert max(errors,default=0)<2e-6,(m,theta,errors)
            assert all(np.isfinite(field(p,normal)).all() for p in ps)
            results.append(dict(reflections=m,theta=theta,roots=len(ps),windings=[p['winding'] for p in ps],morse=[p['morse'] for p in ps],max_law_residual=max([p['snell_residual'] for p in ps],default=0),max_fd_spreading_relative=max(errors,default=0)))
    return results

if __name__=='__main__':
    import json
    print(json.dumps(checks(),indent=2))
