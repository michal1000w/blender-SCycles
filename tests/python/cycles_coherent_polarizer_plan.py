#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare complete batch JSON and prospective gate mapping; never renders."""
import argparse,json,hashlib
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];worker=Path(__file__).with_name('cycles_coherent_mirror_render_worker.py').resolve();jobs=[]
    declared=json.loads((a.fixtures/'pre_gpu_gates.json').read_text())
    for mode in ['pt','bdpt','pt-guiding']:
        for group in ['malus','brewster_glare','camera_filters']:
            manifest_path=a.fixtures/group/'manifest.json';manifest=json.loads(manifest_path.read_text());assert not manifest['pending_shader_api'] and manifest['specular_connections']
            gates=declared['brewster_glare' if group=='brewster_glare' else 'malus']
            for name,entry in manifest['variants'].items():
                scene=Path(entry['path']);ref=Path(entry['reference_npz']);assert digest(scene)==entry['sha256'];label=mode+'_'+group+'_'+name;folder=a.output.resolve()/label
                arguments=['--blend',str(scene),'--reference',str(ref),'--render',str(folder/'phase_0.exr'),'--report',str(folder/'phase_0_worker.json'),'--case','phase_0','--require-specular','--scene-budgets']
                if mode=='pt':arguments.append('--pt')
                if mode=='pt-guiding':arguments.append('--pt-guiding')
                mean_gate=gates['absolute_mean_error'];leak_gate=None
                if name in ['crossed','object_rotated_90'] and group!='brewster_glare':
                    mean_gate=gates['crossed_absolute_mean_error'];leak_gate=1e-8
                if group=='brewster_glare' and name in ['pass_p_glare_removed','object_rotated_90']:mean_gate=gates['pass_p_absolute_mean_error']
                jobs.append({'case':label,'arguments':arguments,'stdout':str(a.output.resolve()/(label+'_stdout.log')),'stderr':str(a.output.resolve()/(label+'_stderr.log')),
                    'gate':{'all_finite':True,'absolute_mean_error':mean_gate,'absolute_rmse':gates['absolute_rmse'],'absolute_render_mean_max':leak_gate},
                    'scene_sha256':digest(scene),'reference_sha256':digest(ref),'manifest_sha256':digest(manifest_path)})
    plan={'scope':'All16 active coherent variants in PT/BDPT/guiding; unchanged raw-EXR worker, prospective physical gates. Photon mapping unsupported.',
        'worker':str(worker),'state':str(a.output.resolve()/'batch_state.json'),'jobs':jobs,'declared_gates_sha256':digest(a.fixtures/'pre_gpu_gates.json'),
        'persistent_off_on_off':{'script':str(Path(__file__).with_name('cycles_coherent_polarizer_persistent.py').resolve()),'fixtures':str(a.fixtures.resolve()),'modes':['PT','BDPT','GUIDING'],'groups':['malus/single','camera_filters/parallel'],'states':['OFF','ON','OFF'],'gates':{'absolute_mean_error':1e-4,'absolute_rmse':2e-4},'separate_output_required':True},
        'fixed_worker_settings':{'samples':128,'seed':19,'adaptive':False,'denoise':False,'device':'METAL','reference_key':'phase_0_radiance','roi_key':'roi_mask'}}
    a.plan.write_text(json.dumps(plan,indent=2)+'\n');print(str(a.plan.resolve()),len(jobs),'serialized jobs plus18 persistent-state renders')


if __name__=='__main__':main()
