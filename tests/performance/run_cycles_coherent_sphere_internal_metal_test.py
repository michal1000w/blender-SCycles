#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual Metal internal sphere all-winding inventory vs independent doubles."""
import json,subprocess,tempfile,sys,math
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tests/python'))
from cycles_coherent_sphere_internal_reference import inventory,spreading
out=root/'build/tests/performance/coherent_sphere_internal';out.mkdir(parents=True,exist_ok=True)
fixtures=[]
for scale in (.01,1,100):
 for m,theta in ((1,.3),(1,.9),(1,1.2),(2,.6),(2,1.2),(2,2.4)):
  s=np.array([2*scale,0,.01*scale],dtype=np.float32);r=np.array([2*scale*math.cos(theta),2*scale*math.sin(theta),.03*scale],dtype=np.float32);normal=np.asarray(-r/np.linalg.norm(r),dtype=np.float32);radius=float(np.float32(scale));ior=1.5
  refs=inventory(s,r,[0,0,0],radius,ior,m)
  fixtures.append((s,r,normal,radius,ior,m,refs))
report={'scope':'actual Metal T-R-T/T-R-R-T roots, OPL, full curved spreading and Morse','cases':[]}
with tempfile.TemporaryDirectory(prefix='sphere-internal-metal-') as temp:
 temp=Path(temp);inputs=temp/'inputs.txt'
 inputs.write_text('\n'.join(' '.join(format(float(x),'.9g') for x in [*s,*r,0,0,0,R,n,*normal,m]) for s,r,normal,R,n,m,refs in fixtures)+'\n')
 runner=temp/'probe';subprocess.run(['clang++','-std=c++17','-framework','Foundation','-framework','Metal',str(root/'tests/metal/cycles_coherent_sphere_internal_probe.mm'),'-o',str(runner)],check=True)
 for mode,flag in [('strict','-fno-fast-math'),('fast','-ffast-math')]:
  air=temp/(mode+'.air');lib=temp/(mode+'.metallib')
  subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.1',flag,'-fmodules-cache-path='+str(temp/'cache'),'-I'+str(root/'intern/cycles'),'-I'+str(root/'intern/cycles/kernel'),'-c',str(root/'tests/metal/cycles_coherent_sphere_internal_probe.metal'),'-o',str(air)],check=True)
  subprocess.run(['xcrun','-sdk','macosx','metallib',str(air),'-o',str(lib)],check=True)
  result=subprocess.run([str(runner),str(lib),str(inputs)],text=True,capture_output=True,check=True);data=json.loads(result.stdout)
  checks=[]
  for (s,r,normal,R,n,m,refs),case in zip(fixtures,data['cases']):
   rows=case['rows'];checks.append({'type':'inventory','reflections':m,'radius':R,'ok':int(rows[0][0])==(0 if refs else 1) and int(rows[0][1])==len(refs)})
   if int(rows[0][1])!=len(refs):continue
   for k,p in enumerate(refs):
    row=rows[1+k*6];opl=row[2]+row[3];expected=spreading(p,normal)
    error=abs(opl-p['opl'])/R;relative=abs(row[1]/expected-1)
    checks.append({'type':'branch','reflections':m,'radius':R,'branch':k,'opl_error_per_radius':error,'spreading_relative':relative,'selected_vs_full':rows[6+k*6],'ok':max(abs(v) for v in rows[6+k*6])<1e-7 and error<3e-10 and relative<2e-4 and int(rows[2+k*6][3])==p['morse'] and int(rows[4+k*6][3])==p['winding']})
  passed=all(c['ok'] for c in checks);report['cases'].append({'mode':mode,'device':data['device'],'checks':checks,'passed':passed,'raw':data});print(mode,'PASS' if passed else 'FAIL',len(checks),'checks')
(out/'metal_results.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if all(c['passed'] for c in report['cases']) else 1)
