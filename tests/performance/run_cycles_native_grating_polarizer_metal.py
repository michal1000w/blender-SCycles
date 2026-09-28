#!/usr/bin/env python3
"""Actual shipping straight-grating payload/Mueller probe; --run requires GPU ownership."""
import argparse, hashlib, json, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--skip-compile',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=root/'build/tests/performance/native_grating_polarizer_metal_v43';out.mkdir(parents=True,exist_ok=True)
src=root/'tests/metal/cycles_native_grating_polarizer_probe.metal';host=root/'tests/metal/cycles_native_grating_polarizer_probe.mm'
record={'scope':'24 typed straight/coated-grating actual-payload/Mueller cases each strict/fast; no renderer or LUT substitutes','executed':a.run,'commands':[]}
def run(cmd):
 t=time.perf_counter();r=subprocess.run(cmd,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);record['commands'].append({'argv':list(map(str,cmd)),'exit':r.returncode,'seconds':time.perf_counter()-t,'output':r.stdout});return r
if not a.skip_compile:
 for mode,flag in [('strict','-fno-fast-math'),('fast','-ffast-math')]:
  run(['xcrun','-sdk','macosx','metal','-c','-std=metal3.1',flag,'-fmodules-cache-path='+str(out/'module_cache'),'-I',str(root/'intern/cycles'),'-I',str(root/'intern/cycles/kernel'),str(src),'-o',str(out/(mode+'.air'))])
  run(['xcrun','-sdk','macosx','metallib',str(out/(mode+'.air')),'-o',str(out/(mode+'.metallib'))])
 run(['clang++','-std=c++17','-O2','-fobjc-arc',str(host),'-framework','Metal','-framework','Foundation','-o',str(out/'probe')])
if a.run:
 record['modes']={}
 for mode in ['strict','fast']:
  r=run([str(out/'probe'),str(out/(mode+'.metallib'))]);(out/(mode+'.json')).write_text(r.stdout)
  try:record['modes'][mode]=json.loads(r.stdout)
  except json.JSONDecodeError:record['modes'][mode]={'error':r.stdout}
files=[src,host,root/'intern/cycles/kernel/closure/bsdf_diffraction_dielectric.h',root/'intern/cycles/kernel/integrator/polarization_surface.h',root/'intern/cycles/kernel/integrator/polarization_state.h',root/'intern/cycles/kernel/light/polarization_math.h',out/'probe',out/'strict.metallib',out/'fast.metallib']
record['hashes']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()}
record['passed']=a.run and all(c['exit']==0 for c in record['commands']) and all(v.get('failures',1)==0 for v in record.get('modes',{}).values())
(out/'results.json').write_text(json.dumps(record,indent=2)+'\n');print(out/'results.json')
if a.run and not record['passed']:raise SystemExit(1)
