#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded sequential Metal presentation suite; not a speed/physical benchmark."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--samples',type=int,default=512)
p.add_argument('--feature-output',type=Path,help='Include completed Thin Wall and Ashikhmin feature scenes')
p.add_argument('--indirect-bdpt-samples',type=int,help='Use BDPT and this fixed sample count for the indirect presentation')
a=p.parse_args();a.binary=a.binary.resolve();a.output=a.output.resolve()
a.output.mkdir(parents=True,exist_ok=False)
fixtures={
 'cd_dvd':'bounded_package_cd_v1/physical_discs_pt.blend',
 'glass':'glass_delivery_pt_1024_v1/glass.blend',
 'glossy':'glossy_delivery_v1/pt/glossy.blend',
 'metallic':'metallic_delivery_v2/pt/metallic.blend',
 'principled_film':'principled_generalized_film_delivery_v1/pt/principled_generalized_film.blend',
 'refraction':'refraction_delivery_v1/pt/refraction.blend',
 'covered_disc':'fast_delivery_suite_v4/pt_covered/physical_covered_pt.blend',
 'indirect':'fast_delivery_suite_v4/pt_indirect/physical_indirect_pt.blend',
}
if a.feature_output:
 fixtures['thin_wall']=str((a.feature_output/'thin_wall_pt/thin_wall_d320_c1_f180_s1.blend').resolve())
 fixtures['ashikhmin']=str((a.feature_output/'ashikhmin_pt/glossy.blend').resolve())
env=os.environ.copy()
for k in ['BLENDER_SYSTEM_RESOURCES','DYLD_LIBRARY_PATH','CYCLES_KERNEL_PATH','CYCLES_SHADER_PATH']:env.pop(k,None)
manifest={'binary':str(a.binary),'sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),
 'scope':'Fixed-sample denoised presentation suite; noisy EXRs retained, not performance/physical-reference evidence.', 'runs':{}}
for name,relative in fixtures.items():
 template=root/'tests/output/diffraction'/relative
 cmd=[str(a.binary),'--background','--factory-startup','--python-exit-code','1','--threads','2',
      '--python',str(root/'tests/python/cycles_diffraction_denoising_delivery.py'),
      '--','--template',str(template),'--output',str(a.output/name),'--samples',str(a.samples)]
 if name=='indirect' and a.indirect_bdpt_samples:
  cmd[cmd.index('--samples')+1]=str(a.indirect_bdpt_samples)
  cmd+=['--transport','bdpt']
 entry={'template':str(template)}
 with (a.output/(name+'.log')).open('w') as log:
  try:
   run=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=480)
   entry['exit_code']=run.returncode
   if run.returncode==0:entry['report']=json.loads((a.output/name/'report.json').read_text())
  except subprocess.TimeoutExpired:entry['timeout_seconds']=480
 manifest['runs'][name]=entry
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(name,entry.get('exit_code','TIMEOUT'),flush=True)
 if entry.get('exit_code')!=0:raise SystemExit(1)
