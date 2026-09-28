#!/usr/bin/env python3
"""Independent projector/slab and ownership regression, with source provenance."""
import hashlib,json,pathlib,subprocess,tempfile
root=pathlib.Path(__file__).resolve().parents[2]
out=root/'build/tests/performance/coherent_polarizer_v41'
out.mkdir(parents=True,exist_ok=True)
files=['intern/cycles/kernel/light/coherent_polarizer.h','intern/cycles/kernel/light/coherent_path_field.h','intern/cycles/kernel/light/coherent_history.h','tests/performance/cycles_coherent_polarizer_test.cpp','tests/performance/cycles_coherent_history_test.cpp']
report={'sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},'cases':[]}
with tempfile.TemporaryDirectory(prefix='coherent-polarizer-') as d:
 for source in files[-2:]:
  for mode,flags in [('strict',[]),('fast',['-ffast-math'])]:
   binary=pathlib.Path(d)/(pathlib.Path(source).stem+mode)
   cmd=['clang++','-std=c++20','-O2',*flags,'-DCCL_NAMESPACE_BEGIN=namespace ccl {','-DCCL_NAMESPACE_END=}','-Iintern/cycles','-Iintern/cycles/kernel',source,'-o',str(binary)]
   build=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
   run=subprocess.run([str(binary)],capture_output=True,text=True) if build.returncode==0 else None
   report['cases'].append({'source':source,'mode':mode,'command':cmd,'build_exit':build.returncode,'build_stderr':build.stderr,'run_exit':run.returncode if run else None,'stdout':run.stdout if run else ''})
   print(report['cases'][-1]['stdout'] or build.stderr)
report['passed']=all(c['build_exit']==0 and c['run_exit']==0 for c in report['cases'])
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['passed'] else 1)
