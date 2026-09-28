#!/usr/bin/env python3
"""Run the tiny precise Cauchy/unit-conversion bit probe on an Apple GPU."""
from pathlib import Path
import subprocess,json,hashlib
root=Path(__file__).resolve().parents[2];out=root/'build/tests/performance/dielectric_dispersion_metal_v28';out.mkdir(exist_ok=True,parents=True)
def run(a):
 r=subprocess.run(a,cwd=root,capture_output=True,text=True);return {'argv':a,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
result={'sources':{},'cases':[]}
for p in [root/'intern/cycles/kernel/util/dielectric_dispersion.h',root/'tests/metal/cycles_dielectric_dispersion_probe.metal',root/'tests/metal/cycles_dielectric_dispersion_probe.mm']:result['sources'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for flag in ['-fno-fast-math','-ffast-math']:
 name='fast' if flag=='-ffast-math' else 'safe';host=out/('host_'+name);air=out/(name+'.air');lib=out/(name+'.metallib')
 stages=[]
 stages.append(run(['clang++','-std=c++20','-x','objective-c++','-O2',flag,'-fobjc-arc','-DCCL_NAMESPACE_BEGIN=namespace ccl {','-DCCL_NAMESPACE_END=}', '-I',str(root/'intern/cycles'),'-framework','Foundation','-framework','Metal',str(root/'tests/metal/cycles_dielectric_dispersion_probe.mm'),'-o',str(host)]))
 stages.append(run(['xcrun','-sdk','macosx','metal','-c','-std=metal3.1',flag,'-fmodules-cache-path='+str(out/'module_cache'),'-I',str(root/'intern/cycles'),'-I',str(root/'intern/cycles/kernel'),str(root/'tests/metal/cycles_dielectric_dispersion_probe.metal'),'-o',str(air)]))
 if all(s['exit']==0 for s in stages):stages.append(run(['xcrun','-sdk','macosx','metallib',str(air),'-o',str(lib)]))
 if all(s['exit']==0 for s in stages):stages.append(run([str(host),str(lib)]))
 result['cases'].append({'name':name,'stages':stages});print(name,[(s['exit'],s['stderr'][-1500:]) for s in stages],stages[-1]['stdout'],flush=True)
result['passed']=all(len(c['stages'])==4 and all(s['exit']==0 for s in c['stages']) for c in result['cases']);(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')

raise SystemExit(0 if result["passed"] else 1)
