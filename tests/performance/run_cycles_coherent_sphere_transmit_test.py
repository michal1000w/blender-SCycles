#!/usr/bin/env python3
"""Independent double branch, Hessian, Jacobian and split-OPL checks (CPU only)."""
import hashlib,json,math,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/python'))
from cycles_coherent_sphere_transmission_reference import inventory,spreading,basis,unit,bisect
out=ROOT/'build/tests/performance/coherent_sphere_tt';out.mkdir(parents=True,exist_ok=True)
source=ROOT/'tests/performance/cycles_coherent_sphere_transmit_probe.cpp'
header=ROOT/'intern/cycles/kernel/light/coherent_sphere_transmit_geometry.h'
Q,_=np.linalg.qr(np.array([[.3,.7,.4],[.8,-.2,.3],[.1,.6,-.9]]))
cases=[]
for scale in [.01,1,100]:
 for rotate in [False,True]:
  for n,theta,a,b in [(1,3.,2,2),(1.5,2.9,2,2),(1.5,3.,2,2),(1.5,3.1,2,2),(1.5,3.13,2,2),(1.5,math.pi,4,4)]:
   c=np.array([1.7,-2.3,.8])*scale if rotate else np.zeros(3)
   s=np.array([a,0,0])*scale;r=np.array([b*math.cos(theta),b*math.sin(theta),0])*scale
   if rotate:s=Q@s;r=Q@r
   normal=unit(-r)
   cases.append(np.array([*(c+s),*(c+r),*c,scale,n,*normal],dtype=np.float32).astype(float))
# Actual proposed off-axis Blender geometry and a nearby source for stable OPD.
for y in [-1e-5,1e-5]:
 for dy,dz in [(.015,.025),(.025,-.015),(-.025,.03)]:
  cases.append(np.array([-1,y,.05,.4,dy,dz,0,0,0,.25,1.5,-1,0,0],dtype=np.float32).astype(float))
metrics={'spread_relative':0.,'opl_absolute_m':0.,'opl_scale_relative':0.,'point_scale_relative':0.,'snell_double':0.,'hessian_fd_relative':0.,'jacobian_fd_relative':0.}
refs=[]
for row in cases:
 ps=inventory(row[:3],row[3:6],row[6:9],row[9],row[10]);refs.append(ps)
 for p in ps:
  metrics['snell_double']=max(metrics['snell_double'],p['snell_residual'])
# Independent receiver finite differences, preserving ordered isolated branches.
for row,ps in zip(cases[:12],refs[:12]):
 if len(ps)==0:continue
 rb=basis(row[11:]);h=1e-5*row[9]
 neighbors=[[inventory(row[:3],row[3:6]+sign*h*rb[:,j],row[6:9],row[9],row[10]) for sign in [-1,1]] for j in range(2)]
 for k,p in enumerate(ps):
  D=np.stack([(neighbors[j][1][k]['directions'][0]-neighbors[j][0][k]['directions'][0])/(2*h) for j in range(2)],axis=1)
  fd=np.linalg.norm(np.cross(D[:,0],D[:,1]));exact=spreading(p,row[11:]);metrics['jacobian_fd_relative']=max(metrics['jacobian_fd_relative'],abs(fd/exact-1))
# Curvature Hessian against second differences of independently parameterized OPL.
for p in refs[2]:
 row=cases[2];h=2e-5*row[9]
 def objective(q):
  points=[row[6:9]+row[9]*unit(p['normals'][i]+p['frames'][i]@q[2*i:2*i+2]/row[9]) for i in range(2)]
  return math.fsum([np.linalg.norm(points[0]-row[:3]),row[10]*np.linalg.norm(points[1]-points[0]),np.linalg.norm(row[3:6]-points[1])])
 H=np.zeros((4,4));zero=np.zeros(4);f=objective(zero)
 for i in range(4):
  ei=np.eye(4)[i]*h;H[i,i]=(objective(ei)-2*f+objective(-ei))/(h*h)
  for j in range(i):
   ej=np.eye(4)[j]*h;H[i,j]=H[j,i]=(objective(ei+ej)-objective(ei-ej)-objective(-ei+ej)+objective(-ei-ej))/(4*h*h)
 metrics['hessian_fd_relative']=max(metrics['hessian_fd_relative'],float(np.max(np.abs(H-p['H']))/np.max(np.abs(p['H']))))
A=B=.5;n=1.5
tc=bisect(lambda t:2-A*math.sqrt((1-t*t)/(1-A*A*t*t))-B*math.sqrt((1-t*t)/(1-B*B*t*t))-2/n*math.sqrt((1-t*t)/(1-t*t/n/n)),0,1-1e-15)
foldtheta=math.pi+2*math.asin(tc)-math.asin(A*tc)-math.asin(B*tc)-2*math.asin(tc/n)
negatives=[([.1,0,0,-2,.1,0,0,0,0,1,1.5,1,0,0],2),([2,0,0,-2,0,0,0,0,0,1,1.5,1,0,0],4),([2,0,0,2,0,0,0,0,0,1,1.5,1,0,0],1),([2,0,0,2*math.cos(foldtheta),2*math.sin(foldtheta),0,0,0,0,1,1.5,1,0,0],3)]
report={'scope':'complete isolated exterior TT inventory; no BVH/integrator integration; excludes caustics, axial rings and grazing boundary','independent_checks':metrics,'cases':[]}
with tempfile.TemporaryDirectory(prefix='sphere-tt-') as tmp:
 for mode,flags in [('strict',[]),('optimized_fast',['-ffast-math'])]:
  exe=Path(tmp)/mode;cmd=['c++','-std=c++20','-O2',*flags,'-DCCL_NAMESPACE_BEGIN=namespace ccl {','-DCCL_NAMESPACE_END=}','-Iintern/cycles','-Iintern/cycles/kernel',str(source),'-o',str(exe)]
  subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True)
  inputs=cases+[np.array(x,dtype=np.float32).astype(float) for x,_ in negatives]
  run=subprocess.run([str(exe)],input='\n'.join(' '.join(map(str,x)) for x in inputs),text=True,capture_output=True,check=True)
  (out/(mode+'.txt')).write_text(run.stdout)
  checks=0;errors=[];actual=[];m=dict(metrics)
  for index,(row,line) in enumerate(zip(inputs,run.stdout.splitlines())):
   data=list(map(float,line.split()));status,count=map(int,data[:2]);checks+=1
   if index>=len(cases):
    if status!=negatives[index-len(cases)][1] or count!=0:errors.append(f'status case{index}: {status}/{count}')
    continue
   ps=refs[index]
   if status!=0 or count!=len(ps):errors.append(f'root inventory case{index}: {status}/{count} vs {len(ps)}');continue
   branchvalues=[]
   for k,p in enumerate(ps):
    h,spread,opl,morse,*points=data[2+10*k:12+10*k];checks+=4;branchvalues.append(opl)
    er=abs(spread/spreading(p,row[11:])-1);ep=abs(opl-p['opl']);ex=np.max(np.abs(np.array(points).reshape(2,3)-p['points']))/row[9]
    m['spread_relative']=max(m['spread_relative'],er);m['opl_absolute_m']=max(m['opl_absolute_m'],ep);m['opl_scale_relative']=max(m['opl_scale_relative'],ep/row[9]);m['point_scale_relative']=max(m['point_scale_relative'],ex)
    if er>3e-4 or ep/row[9]>2e-10 or ex>1e-5 or int(morse)!=p['morse']:errors.append(f'branch case{index}/{k}: spread{er}, opl{ep/row[9]}, point{ex}, Morse{morse}/{p["morse"]}')
   actual.append(branchvalues)
  # Pairwise OPD between the two proposed source positions, all branch combinations.
  opd=0.
  for j in range(3):
   for k in range(len(refs[-6+j])):
    for l in range(len(refs[-3+j])):
     opd=max(opd,abs((actual[-6+j][k]-actual[-3+j][l])-(refs[-6+j][k]['opl']-refs[-3+j][l]['opl'])))
  m['pair_opd_error_m']=opd
  if opd>5e-11:errors.append(f'pair OPD {opd}')
  if metrics['hessian_fd_relative']>2e-5 or metrics['jacobian_fd_relative']>2e-5:errors.append('independent differential check')
  report['cases'].append({'mode':mode,'command':cmd,'checks':checks,'metrics':m,'errors':errors})
  print(mode,checks,'checks',m,errors)
files=[source,header,Path(__file__),ROOT/'tests/python/cycles_coherent_sphere_transmission_reference.py',ROOT/'intern/cycles/kernel/light/coherent_geometry.h']
report['sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(1 if any(c['errors'] for c in report['cases']) else 0)
