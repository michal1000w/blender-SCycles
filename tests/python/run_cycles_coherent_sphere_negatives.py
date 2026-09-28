#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Four bounded native-sphere host rejections; explicit --run authorizes launch.

Each preserved scene must fail with its own descriptive host-validation error.
A normal render return, another error, or missing report fails the gate.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

EXPECTED={
 'reject_internal_source':'sources must lie strictly outside every declared sphere',
 'reject_glass_sphere':'exterior ideal Mirror reflection only',
 'reject_ellipsoid':'require an exact axis-aligned uniform transform',
 'reject_multiple_events':'require exactly one interface event',
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as source:
  for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def worker(fixtures,output):
 import bpy
 binary=bpy.app.binary_path
 report={'binary':binary,'binary_sha256':sha(binary),'cases':{},'scope':'Expected host rejections; not positive rendering evidence'}
 prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
 for d in prefs.devices:d.use=d.type=='METAL'
 assert any(d.use for d in prefs.devices),'Metal device unavailable'
 for name,message in EXPECTED.items():
  scene_path=fixtures/(name+'.blend');bpy.ops.wm.open_mainfile(filepath=str(scene_path));scene=bpy.context.scene
  assert scene.cycles.use_coherent_specular_connections,'Negative fixture did not enable connector'
  scene.cycles.device='GPU';scene.cycles.samples=1;scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False
  scene.render.resolution_x=scene.render.resolution_y=8
  scene.render.filepath=str(output/(name+'_unexpected.exr'))
  error=None;start=time.monotonic()
  try:bpy.ops.render.render(write_still=True)
  except RuntimeError as exc:error=str(exc)
  passed=error is not None and message in error and not (output/(name+'_unexpected.exr')).exists()
  report['cases'][name]={'scene_sha256':sha(scene_path),'expected_substring':message,'error':error,'passed':passed,'accidental_success':error is None,'elapsed_seconds':time.monotonic()-start}
  (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 report['all_passed']=all(case['passed'] for case in report['cases'].values())
 (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 if not report['all_passed']:raise RuntimeError('Native sphere expected-rejection gate failed')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--blender',type=Path);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');p.add_argument('--worker',action='store_true')
 a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None);fixtures=a.fixtures.resolve();output=a.output.resolve()
 for name in EXPECTED:assert (fixtures/(name+'.blend')).is_file()
 if a.worker:return worker(fixtures,output)
 if not a.run:p.error('GPU launch requires explicit --run')
 assert a.blender and a.blender.is_file();output.mkdir(parents=True,exist_ok=False)
 env=os.environ.copy()
 for key in ('BLENDER_SYSTEM_RESOURCES','DYLD_LIBRARY_PATH','CYCLES_KERNEL_PATH','CYCLES_SHADER_PATH'):env.pop(key,None)
 command=[str(a.blender.resolve()),'--background','--factory-startup','--python-exit-code','73','--threads','2','--python',str(Path(__file__).resolve()),'--','--worker','--fixtures',str(fixtures),'--output',str(output)]
 with (output/'validation.log').open('w') as log:completed=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 result={'command':command,'exit_code':completed.returncode,'binary_sha256':sha(a.blender),'runner_sha256':sha(__file__),'log':str(output/'validation.log')}
 path=output/'report.json'
 result['all_passed']=completed.returncode==0 and path.is_file() and json.loads(path.read_text()).get('all_passed',False)
 (output/'execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 return 0 if result['all_passed'] else 1

if __name__=='__main__':sys.exit(main())
