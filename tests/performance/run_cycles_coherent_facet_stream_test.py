#!/usr/bin/env python3
"""Focused CPU tests for streamed facets and source ownership; no GPU execution."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=root/'build/tests/performance/coherent_facet_stream_v45';out.mkdir(parents=True,exist_ok=True)
report={'scope':'Actual streaming geometry/field arithmetic and typed facet ownership helpers; renderer acceptance separate','cases':[]}
for name,source in [('core','cycles_coherent_facet_stream_test.cpp'),('history','cycles_coherent_facet_history_test.cpp')]:
 for mode,flags in [('strict',[]),('fast',['-ffast-math'])]:
  binary=out/(name+'_'+mode)
  cmd=['clang++','-std=c++20','-O2',*flags,'-DCCL_NAMESPACE_BEGIN=namespace ccl {','-DCCL_NAMESPACE_END=}',
       '-Iintern/cycles','-Iintern/cycles/kernel','-Iintern/glew-mx','-Iintern/guardedalloc','-I.',
       'tests/performance/'+source,'-o',str(binary)]
  built=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
  run=subprocess.run([str(binary)],cwd=root,capture_output=True,text=True) if built.returncode==0 else None
  item={'name':name+'_'+mode,'command':cmd,'compile_exit':built.returncode,'compile_output':built.stdout+built.stderr,
        'run_exit':run.returncode if run else None,'output':run.stdout+run.stderr if run else None}
  report['cases'].append(item);print(item['name'],item['run_exit'],item['output'])
files=[root/'intern/cycles/kernel/light'/f for f in ['coherent_facet_stream.h','coherent_facet_integrator.h','coherent_history.h','coherent_history_kernel.h','coherent_patch_membership.h']]
files+=[root/'tests/performance'/f for f in ['cycles_coherent_facet_stream_test.cpp','cycles_coherent_facet_history_test.cpp',Path(__file__).name]]
report['sha256']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
report['passed']=all(c['compile_exit']==0 and c['run_exit']==0 for c in report['cases'])
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['passed'] else 1)
