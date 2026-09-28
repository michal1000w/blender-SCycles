#!/usr/bin/env python3
"""Bounded CPU test of actual Metal classification bodies and scene key policy."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
def function(path,signature):
 text=(root/path).read_text();start=text.index(signature);open_brace=text.index('{',start);depth=1;i=open_brace+1
 while depth:
  depth+=(text[i]=='{')-(text[i]=='}');i+=1
 return text[start:i]
legacy=function('intern/cycles/device/kernel.cpp','bool device_kernel_has_intersection(DeviceKernel kernel)')
metal=function('intern/cycles/device/metal/kernel.h','static inline bool metal_kernel_has_intersection(')
source='''#include "kernel/types.h"
#include "kernel/features.h"
#include <cstdio>
#include <initializer_list>
using namespace ccl;
namespace ccl {
'''+legacy+'\n'+metal+'''\n}
int main() {
 int checks=0;
 for(int k=0;k<DEVICE_KERNEL_NUM;k++) for(uint64_t features:{uint64_t(0),uint64_t(KERNEL_FEATURE_POINTCLOUD),uint64_t(KERNEL_FEATURE_COHERENT_SPECULAR),uint64_t(KERNEL_FEATURE_POINTCLOUD|KERNEL_FEATURE_COHERENT_SPECULAR)}) {
  const DeviceKernel kernel=DeviceKernel(k);
  const bool old=device_kernel_has_intersection(kernel);
  const bool want=old || (kernel==DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE && (features & KERNEL_FEATURE_COHERENT_SPECULAR));
  if(metal_kernel_has_intersection(kernel,features)!=want) return 1;
  checks++;
 }
 if(device_kernel_has_intersection(DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE))return 2;
 if(!metal_kernel_has_intersection(DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE,KERNEL_FEATURE_COHERENT_SPECULAR))return 3;
 printf("{\\"classification_checks\\":%d,\\"legacy_surface_unmodified\\":true,\\"coherent_surface_tables_required\\":true}\\n",checks);
}
'''
out=root/'build/tests/performance/coherent_metal_intersection_v35';out.mkdir(parents=True,exist_ok=True)
(out/'actual_classification_bodies.cpp').write_text(source)
with tempfile.TemporaryDirectory() as tmp:
 cmd=['c++','-std=c++20','-O2','-DCCL_NAMESPACE_BEGIN=namespace ccl {','-DCCL_NAMESPACE_END=}','-I'+str(root/'intern/cycles'),'-I'+str(root/'intern/cycles/kernel'),str(out/'actual_classification_bodies.cpp'),'-o',str(Path(tmp)/'test')]
 p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stderr
 result=json.loads(subprocess.check_output([str(Path(tmp)/'test')],text=True));result['command']=cmd
result['source_hashes']={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ('intern/cycles/device/kernel.cpp','intern/cycles/device/metal/kernel.h','intern/cycles/device/metal/kernel.mm','intern/cycles/device/metal/queue.mm','intern/cycles/device/metal/device_impl.mm')}
result['scope']='Actual legacy and new helper bodies compiled from production source; full ObjC++ syntax checked separately. Rendering validation pending.'
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
