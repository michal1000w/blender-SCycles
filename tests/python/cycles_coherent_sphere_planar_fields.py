# SPDX-License-Identifier: Apache-2.0
"""Independent finite-mirror/sphere visibility and full vector field oracle."""
import itertools, math
import numpy as np
from cycles_coherent_sphere_reference import sphere_path, spreading
from cycles_coherent_sphere_transmission_reference import inventory, spreading as tt_spreading, basis
from cycles_coherent_sphere_planar_reference import reflect, intersection
N=np.array([1.,0.,0.])


def setup(kind):
    if kind=='TT':
        return dict(kind=kind,radius=.01,ior=1.5,source=[.012,0,.0001],receiver=[-.012,.00023,0],
                    mirrors=[{'x':.016,'y':0.,'z':0.,'half':.009},{'x':-.016,'y':0.,'z':0.,'half':.009}],crop=.00007)
    return dict(kind=kind,radius=.003,ior=1.5,source=[-.005,.001,.0004],receiver=[-.006,-.002,-.0003],
                mirrors=[{'x':-.01,'y':.00052,'z':.00023,'half':.001},
                         {'x':-.009,'y':-.00139,'z':-.00020,'half':.001}],crop=.00012)


def plane(m):return np.array([m['x'],0.,0.])
def in_patch(p,m):return abs(p[1]-m['y'])<=m['half'] and abs(p[2]-m['z'])<=m['half']


def visible(vertices,weights,cfg):
    eps=2e-7
    for a,b,w in zip(vertices,vertices[1:],weights):
        delta=b-a
        for m in cfg['mirrors']:
            if abs(delta[0])<1e-15:continue
            t=(m['x']-a[0])/delta[0]
            if eps<t<1-eps and in_patch(a+t*delta,m):return False
        if w!=1:continue
        A=np.dot(delta,delta);B=2*np.dot(a,delta);C=np.dot(a,a)-cfg['radius']**2
        disc=B*B-4*A*C
        if disc>0:
            for t in ((-B-math.sqrt(disc))/(2*A),(-B+math.sqrt(disc))/(2*A)):
                if eps<t<1-eps:return False
    return True


def finish(source,receiver,points,normals,events,weights,spread,morse,cfg,label):
    vertices=[source,*points,receiver]
    if not visible(vertices,weights,cfg):return None
    lengths=np.array([np.linalg.norm(b-a) for a,b in zip(vertices,vertices[1:])])
    directions=[(b-a)/l for a,b,l in zip(vertices,vertices[1:],lengths)]
    if np.dot(directions[-1],N)>=0:return None
    return dict(points=points,normals=normals,events=events,weights=weights,directions=directions,
                opl=float(np.dot(weights,lengths)),spreading=float(spread),morse=morse,label=label)


def paths(source,receiver,cfg):
    completed=[];radius=cfg['radius'];mirrors=cfg['mirrors'];ior=cfg['ior']
    # Sphere chain with zero or one plane on each exterior side: all <=4-event candidates.
    for prefix,suffix in itertools.product((None,*range(len(mirrors))),repeat=2):
        if prefix is not None and prefix==suffix:continue
        vs=source if prefix is None else reflect(source,plane(mirrors[prefix]),N)
        vr=receiver if suffix is None else reflect(receiver,plane(mirrors[suffix]),N)
        try:
            branches=inventory(vs,vr,np.zeros(3),radius,ior) if cfg['kind']=='TT' else [sphere_path(vs,vr,np.zeros(3),radius)]
        except ValueError:continue
        for branch in branches:
            points=list(branch['points']) if cfg['kind']=='TT' else [branch['point']]
            normals=list(branch['normals']) if cfg['kind']=='TT' else [branch['normal']]
            events=['T','T'] if cfg['kind']=='TT' else ['R']
            weights=[1,ior,1] if cfg['kind']=='TT' else [1,1]
            vn=N if suffix is None else reflect(N,np.zeros(3),N)
            spread=tt_spreading(branch,vn) if cfg['kind']=='TT' else spreading(branch,*basis(vn).T,radius)
            try:
                if prefix is not None:
                    p=intersection(vs,points[0],plane(mirrors[prefix]),N)
                    if not in_patch(p,mirrors[prefix]):continue
                    points.insert(0,p);normals.insert(0,N);events.insert(0,'R');weights.insert(0,1)
                if suffix is not None:
                    p=intersection(points[-1],vr,plane(mirrors[suffix]),N)
                    if not in_patch(p,mirrors[suffix]):continue
                    points.append(p);normals.append(N);events.append('R');weights.append(1)
            except ValueError:continue
            p=finish(source,receiver,points,normals,events,weights,spread,branch.get('morse',0),cfg,f'{prefix}:{cfg["kind"]}:{suffix}')
            if p:completed.append(p)
    # Pure planar/direct families (all sequences <=4, no consecutive self reflection).
    for count in range(5):
        for sequence in itertools.product(range(len(mirrors)),repeat=count):
            if any(a==b for a,b in zip(sequence,sequence[1:])):continue
            target=receiver.copy()
            for index in reversed(sequence):target=reflect(target,plane(mirrors[index]),N)
            current=source.copy();direction=(target-current)/np.linalg.norm(target-current);points=[]
            valid=True
            for index in sequence:
                m=mirrors[index]
                t=(m['x']-current[0])/direction[0]
                if t<=0:valid=False;break
                point=current+t*direction
                if not in_patch(point,m):valid=False;break
                points.append(point);current=point;direction=reflect(direction,np.zeros(3),N)
            if not valid or np.linalg.norm(np.cross(receiver-current,direction))>1e-9 or np.dot(receiver-current,direction)<=0:continue
            length=sum(np.linalg.norm(b-a) for a,b in zip([source,*points],[*points,receiver]))
            spread=abs(np.dot(direction,N))/length**2
            p=finish(source,receiver,points,[N]*count,['R']*count,[1]*(count+1),spread,0,cfg,'planes:'+str(sequence))
            if p:completed.append(p)
    return completed


def field(path,power=1):
    directions=path['directions'];modes=(np.eye(3)-np.outer(directions[0],directions[0]))/math.sqrt(2)
    for index,(normal,event) in enumerate(zip(path['normals'],path['events'])):
        incoming,outgoing=directions[index:index+2];s=np.cross(incoming,normal)
        if np.linalg.norm(s)<1e-12:s=basis(incoming)[:,0]
        s=s/np.linalg.norm(s);pi=np.cross(s,incoming);po=np.cross(s,outgoing)
        ts=1.;tp=-1. if event=='R' else 1.
        if event=='T':
            ni,no=path['weights'][index:index+2];ci=abs(np.dot(incoming,normal));ct=abs(np.dot(outgoing,normal));flux=math.sqrt(no*ct/(ni*ci))
            ts=2*ni*ci/(ni*ci+no*ct)*flux;tp=2*ni*ci/(no*ci+ni*ct)*flux
        modes=(modes@s)[:,None]*(ts*s)+(modes@pi)[:,None]*(tp*po)
    return modes.astype(complex)*math.sqrt(power*path['spreading']/(4*math.pi**2))*np.exp(-.5j*math.pi*path['morse'])


def radiance(paths_by_source,phases,groups,wavelength=550e-9,coherence_length=1e-4,powers=(1.,1.)):
    terms=[(field(p,powers[k]),p['opl'],phases[k],groups[k]) for k,ps in enumerate(paths_by_source) for p in ps]
    result=sum(float(np.vdot(f,f).real) for f,*_ in terms)
    for i,(a,la,pa,ga) in enumerate(terms):
        for b,lb,pb,gb in terms[i+1:]:
            if ga!=gb:continue
            opd=la-lb;phase=2*math.pi*math.remainder(opd/wavelength,1)+pa-pb
            result+=2*math.exp(-.5*(opd/coherence_length)**2)*float(np.real(np.sum(a*np.conj(b))*np.exp(1j*phase)))
    return result
