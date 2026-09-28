#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import hashlib,json,subprocess,tempfile,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/python'))
from cycles_coherent_sphere_internal_reference import inventory,spreading
source=root/'tests/performance/cycles_coherent_sphere_internal_geometry_test.cpp'
files=[source,Path(__file__),root/'intern/cycles/kernel/light/coherent_sphere_internal_geometry.h',root/'intern/cycles/kernel/light/coherent_unfold_internal_geometry.h',root/'tests/python/cycles_coherent_sphere_internal_reference.py']
report={'scope':'complete isolated T-R-T/T-R-R-T sphere inventories and planar mirror prefix','sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'cases':[]}
with tempfile.TemporaryDirectory(prefix='sphere-internal-') as temp:
 for mode,flags in [('strict',[]),('fast',['-ffast-math'])]:
  exe=Path(temp)/mode
  cmd=['c++','-std=c++20','-O2',*flags,'-DCCL_NAMESPACE_BEGIN=namespace ccl {','-DCCL_NAMESPACE_END=}','-Iintern/cycles','-Iintern/cycles/kernel',str(source),'-o',str(exe)]
  build=subprocess.run(cmd,cwd=root,text=True,capture_output=True);run=subprocess.run([str(exe)],text=True,capture_output=True) if build.returncode==0 else None
  compare=[]
  if run:
   for line in run.stdout.splitlines():
    if not line.startswith('m'):continue
    import re
    hit=re.match(r'm(\d+) theta([^ ]+) branch(\d+) t([^ ]+) opl([^ ]+) spread([^ ]+) morse(\d+) rx([^ ]+) ry([^ ]+)',line)
    m,theta,k,t,opl,spread,morse,rx,ry=hit.groups();m=int(m);k=int(k);theta=np.float32(theta)
    # C++ sinf/cosf float endpoints are the scene input, not ideal decimal angles.
    receiver=np.array([np.float32(rx),np.float32(ry),0.],dtype=float)
    refs=inventory([2,0,0],receiver,[0,0,0],1,1.5,m);ref=refs[k]
    opl_error=abs(float(opl)-ref['opl']);spread_error=abs(float(spread)/spreading(ref,-receiver/np.linalg.norm(receiver))-1)
    compare.append({'m':m,'theta':float(theta),'branch':k,'opl_error':opl_error,'spreading_relative':spread_error,'morse_matches':int(morse)==ref['morse']})
  case={'mode':mode,'build_exit':build.returncode,'build_stderr':build.stderr,'run_exit':run.returncode if run else None,'stdout':run.stdout if run else '', 'oracle':compare}
  report['cases'].append(case);print(mode,case['stdout']);print('max OPL',max((c['opl_error'] for c in compare),default=0),'max spreading',max((c['spreading_relative'] for c in compare),default=0))
output=root/'build/tests/performance/coherent_sphere_internal';output.mkdir(parents=True,exist_ok=True);(output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if all(c['build_exit']==0 and c['run_exit']==0 and all(x['opl_error']<2e-10 and x['spreading_relative']<8e-5 and x['morse_matches'] for x in c['oracle']) for c in report['cases']) else 1)
