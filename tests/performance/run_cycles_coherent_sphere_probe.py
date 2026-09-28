#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict/optimized-fast production core vs independent double angle-root oracle."""
import hashlib,io,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/python'))
from cycles_coherent_sphere_reference import sphere_path,spreading,unit,checks
output=ROOT/'build/tests/performance/coherent_sphere_v33';output.mkdir(parents=True,exist_ok=True)
report={'independent_double_checks':checks(),'cases':{},'source_hashes':{}}
for p in ('tests/performance/cycles_coherent_sphere_probe.cpp','tests/python/cycles_coherent_sphere_reference.py','intern/cycles/kernel/light/coherent_curved_geometry.h','intern/cycles/scene/coherent_sphere_host.h'):
 report['source_hashes'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory() as td:
 for mode,flags in (('strict',[]),('optimized_fast',['-ffast-math'])):
  exe=str(Path(td)/mode);cmd=['c++','-std=c++20','-O2',*flags,'-DCCL_NAMESPACE_BEGIN=namespace ccl {','-DCCL_NAMESPACE_END=}','-I'+str(ROOT/'intern/cycles'),'-I'+str(ROOT/'intern/cycles/kernel'),str(ROOT/'tests/performance/cycles_coherent_sphere_probe.cpp'),'-o',exe]
  subprocess.run(cmd,check=True);raw=subprocess.check_output([exe],text=True);(output/(mode+'.csv')).write_text(raw)
  rows=np.genfromtxt(io.StringIO(raw),delimiter=',',names=True);operrs=[];gerrs=[];pairs=[]
  for row in rows:
   center=np.array([row[k] for k in ('cx','cy','cz')]);source=np.array([row[k] for k in ('sx','sy','sz')]);receiver=np.array([row[k] for k in ('rx','ry','rz')]);normal=unit(np.array([row[k] for k in ('nx','ny','nz')]))
   assert row['solved']==1
   tangent=unit(np.cross(normal,[0.,0.,1.]));second=np.cross(normal,tangent)
   ref=sphere_path(source,receiver,center,row['radius']);g=spreading(ref,tangent,second,row['radius'])
   operrs.append(abs(row['opl']-ref['optical_length']));gerrs.append(abs(row['spreading']-g)/g)
   pairs.append((row['opl'],ref['optical_length']))
  opd=max(abs((pairs[i][0]-pairs[i+1][0])-(pairs[i][1]-pairs[i+1][1])) for i in range(0,len(pairs),2))
  result={'count':len(rows),'max_optical_length_error_m':max(operrs),'max_source_pair_OPD_error_m':opd,'max_spreading_relative_error':max(gerrs),'command':cmd}
  assert max(operrs)<1e-9 and opd<1e-9 and max(gerrs)<3e-5,result
  report['cases'][mode]=result
(output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['cases'],indent=2))
