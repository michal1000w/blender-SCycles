#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU root/policy checks and release CPU/Metal syntax; never dispatch GPU work."""
import concurrent.futures,hashlib,json,pathlib,shlex,subprocess
r=pathlib.Path(__file__).resolve().parents[2];out=r/'build/tests/performance/coherent_glass_point_v38';out.mkdir(parents=True,exist_ok=True);db=json.load(open(r/'build/macos_arm64_Release/compile_commands.json'))
e=next(a for a in db if a['file'].endswith('/kernel/device/cpu/kernel.cpp'))
c=shlex.split(e['command']);c[c.index('-c')+1]=str(r/'tests/performance/cycles_coherent_glass_point_intersection_test.cpp');c[c.index('-o')+1]=str(out/'test.o')
subprocess.run(c,cwd=e['directory'],check=True)
subprocess.run(['/usr/bin/c++',str(out/'test.o'),'-o',str(out/'test')],check=True)
jobs=[]
for suffix,label in [('/kernel/device/cpu/kernel.cpp','cpu_kernel'),('/scene/object.cpp','host_object')]:
 e=next(a for a in db if a['file'].endswith(suffix));c=shlex.split(e['command']);i=c.index('-o');del c[i:i+2];c.remove('-c');c.append('-fsyntax-only');jobs.append((label,c,e['directory']))
raw=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(r/'build/macos_arm64_Release'),'-t','commands','cycles_kernel_metal_precompile'],text=True)
for line in raw.splitlines():
 if ' metal -c ' not in line:continue
 cwd,cmd=line.split(' && ',1);c=shlex.split(cmd);c.remove('-c');i=c.index('-o');name=pathlib.Path(c[i+1]).stem;del c[i:i+2];c.append('-fsyntax-only');jobs.append((name,c,shlex.split(cwd)[1]))
def run(j):
 label,c,cwd=j;p=subprocess.run(c,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(out/(label+'.log')).write_text(p.stdout);return label,{'exit':p.returncode,'command':c,'log':str(out/(label+'.log'))}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: results=dict(pool.map(run,jobs))
results['unit_test']=json.loads(subprocess.check_output([str(out/'test')],text=True))
paths=['intern/cycles/kernel/geom/point_intersect.h','intern/cycles/kernel/geom/point_intersect_policy.h','intern/cycles/kernel/bvh/util.h','intern/cycles/kernel/bvh/traversal.h','intern/cycles/kernel/bvh/shadow_all.h','intern/cycles/kernel/bvh/intersect_filter.h','intern/cycles/kernel/device/cpu/bvh.h','intern/cycles/kernel/device/metal/kernel.metal','intern/cycles/kernel/types.h','intern/cycles/scene/object.cpp','tests/performance/cycles_coherent_glass_point_intersection_test.cpp']
results['source_sha256']={p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in paths};(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({k:v.get('exit',v) for k,v in results.items() if k!='source_sha256'},indent=2))
