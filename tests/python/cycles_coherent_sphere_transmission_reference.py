"""Independent double complete TT sphere branches; no production solver imports."""
import math
import numpy as np

def unit(v): return v/np.linalg.norm(v)
def basis(n):
 u=unit(np.cross(n,[0.,0.,1.] if abs(n[2])<.9 else [0.,1.,0.]))
 return np.stack((u,np.cross(n,u)),axis=1)
def bisect(f,a,b):
 fa=f(a)
 for _ in range(80):
  m=(a+b)/2;fm=f(m)
  if fa*fm<=0:b=m
  else:a=m;fa=fm
 return (a+b)/2

def inventory(source,receiver,center,radius,ior):
 s=np.asarray(source)-center;r=np.asarray(receiver)-center;a=np.linalg.norm(s);b=np.linalg.norm(r)
 if radius<=0 or ior<1 or min(a,b)<=radius:raise ValueError('outside endpoints and positive sphere/IOR>=1 required')
 e0=s/a;e1=r/b-e0*np.dot(e0,r/b);sine=np.linalg.norm(e1);theta=math.atan2(sine,np.dot(e0,r/b))
 e1=e1/sine if sine>1e-12 else basis(e0)[:,0]
 A=radius/a;B=radius/b
 def F(t):return math.pi+2*math.asin(t)-math.asin(A*t)-math.asin(B*t)-2*math.asin(t/ior)-theta
 def D(t):
  v=max(0.,1-t*t)
  return 2-A*math.sqrt(v/(1-(A*t)**2))-B*math.sqrt(v/(1-(B*t)**2))-2/ior*math.sqrt(v/(1-(t/ior)**2)) if t<1 or ior>1 else 2-A*0-B*0-2
 critical=bisect(D,0,1-1e-15) if D(0)<0 else None
 bounds=[-1,1] if critical is None else [-1,-critical,critical,1]
 roots=[]
 for l,h in zip(bounds,bounds[1:]):
  if abs(F(l))<1e-12 and abs(l)<1:raise ValueError('caustic fold')
  if F(l)*F(h)<0:roots.append(bisect(F,l,h))
  elif F(l)==0:roots.append(l)
 if sine<1e-10 and any(abs(t)>1e-8 for t in roots):raise ValueError('axial nonisolated ring')
 paths=[]
 for t in roots:
  if abs(t)>=1:raise ValueError('grazing boundary')
  phi=math.asin(t)-math.asin(A*t);psi=phi+math.pi-2*math.asin(t/ior)
  normals=[e0*math.cos(phi)+e1*math.sin(phi),e0*math.cos(psi)+e1*math.sin(psi)]
  points=[center+radius*n for n in normals];v=[points[0]-source,points[1]-points[0],receiver-points[1]];length=np.array([np.linalg.norm(x) for x in v]);d=[x/l for x,l in zip(v,length)]
  frames=[basis(n) for n in normals];P=[np.eye(3)-np.outer(x,x) for x in d]
  grad=[d[0]-ior*d[1],ior*d[1]-d[2]]
  H=np.zeros((4,4));H[:2,:2]=frames[0].T@(P[0]/length[0]+ior*P[1]/length[1])@frames[0]-np.eye(2)*np.dot(grad[0],normals[0])/radius
  H[2:,2:]=frames[1].T@(ior*P[1]/length[1]+P[2]/length[2])@frames[1]-np.eye(2)*np.dot(grad[1],normals[1])/radius
  H[:2,2:]=-ior/length[1]*frames[0].T@P[1]@frames[1];H[2:,:2]=H[:2,2:].T
  eigen=np.linalg.eigvalsh(H)
  if min(abs(eigen))/max(abs(eigen))<1e-9:raise ValueError('caustic singular Hessian')
  paths.append({'t':t,'points':points,'normals':normals,'frames':frames,'length':length,'directions':d,'H':H,'morse':int((eigen<0).sum()),'opl':math.fsum([length[0],ior*length[1],length[2]]),'snell_residual':max(np.linalg.norm(frames[i].T@grad[i]) for i in range(2))})
 return paths

def spreading(path,receiver_normal):
 rb=basis(receiver_normal);B=path['frames'];l=path['length'];d=path['directions']
 rhs=np.zeros((4,2));rhs[2:]=B[1].T@(np.eye(3)-np.outer(d[2],d[2]))@rb/l[2]
 q=np.linalg.solve(path['H'],rhs);D=(np.eye(3)-np.outer(d[0],d[0]))@B[0]@q[:2]/l[0]
 return np.linalg.norm(np.cross(D[:,0],D[:,1]))
