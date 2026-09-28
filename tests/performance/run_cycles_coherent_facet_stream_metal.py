#!/usr/bin/env python3
"""Actual Metal stream arithmetic versus independent double sums; GPU only with --run."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--skip-compile',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parents[2];out=root/'build/tests/performance/coherent_facet_stream_metal_v44';out.mkdir(parents=True,exist_ok=True)
record={'scope':'83 streamed routes plus exact HDR cancellation, shared Gaussian draw, signed half-cross and metre-OPD tiny-Lc phase vs independent double; no scene acceptance','commands':[],'modes':{}}
def run(cmd):
 t=time.perf_counter();r=subprocess.run(cmd,cwd=root,capture_output=True,text=True);record['commands'].append({'argv':cmd,'exit':r.returncode,'seconds':time.perf_counter()-t,'stdout':r.stdout,'stderr':r.stderr});return r
if not args.skip_compile:
 for mode,flag in [('strict','-fno-fast-math'),('fast','-ffast-math')]:
  run(['xcrun','-sdk','macosx','metal','-c','-std=metal3.1',flag,'-fmodules-cache-path='+str(out/'module_cache'),'-I',str(root/'intern/cycles'),'-I',str(root/'intern/cycles/kernel'),str(root/'tests/metal/cycles_coherent_facet_stream_probe.metal'),'-o',str(out/(mode+'.air'))])
  run(['xcrun','-sdk','macosx','metallib',str(out/(mode+'.air')),'-o',str(out/(mode+'.metallib'))])
 run(['clang++','-std=c++17','-O2','-fobjc-arc',str(root/'tests/metal/cycles_coherent_facet_stream_probe.mm'),'-framework','Metal','-framework','Foundation','-o',str(out/'probe')])
if args.run and all(c['exit']==0 for c in record['commands']):
 for mode in ['strict','fast']:
  r=run([str(out/'probe'),str(out/(mode+'.metallib'))]);(out/(mode+'.json')).write_text(r.stdout)
  try:record['modes'][mode]=json.loads(r.stdout)
  except json.JSONDecodeError:record['modes'][mode]={'error':r.stdout+r.stderr}
files=[root/'intern/cycles/kernel/light/coherent_facet_stream.h',root/'tests/metal/cycles_coherent_facet_stream_probe.metal',root/'tests/metal/cycles_coherent_facet_stream_probe.mm',Path(__file__),out/'strict.metallib',out/'fast.metallib']
record['hashes']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()}
record['passed']=args.run and len(record['modes'])==2 and all(c['exit']==0 for c in record['commands']) and all(v.get('failures',1)==0 for v in record['modes'].values())
(out/'results.json').write_text(json.dumps(record,indent=2)+'\n');print(out/'results.json')
if args.run and not record['passed']:raise SystemExit(1)
