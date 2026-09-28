#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare nine active two-reflection acceptance jobs, never renders."""
import argparse,json,hashlib
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();jobs=[];worker=Path(__file__).with_name('cycles_coherent_mirror_render_worker.py').resolve()
    for mode in ['pt','bdpt','pt-guiding']:
        for layout in ['corner_xy']:
            folder=a.fixtures/layout;manifest=json.loads((folder/'manifest.json').read_text());assert not manifest['pending_transport_api'];assert manifest['diffuse_bounces']==0;assert manifest['glossy_bounces']==2;assert manifest['minimum_visible_path_count']==4
            destination=a.output.resolve()/(mode+'_'+layout);reference=folder/'virtual_source_reference.npz'
            for case in ['phase_0','phase_pi','incoherent_connector_control']:
                scene=Path(manifest['variants'][case]['path']);assert digest(scene)==manifest['variants'][case]['sha256'];label=mode+'_'+layout+'_'+case
                args=['--blend',str(scene),'--reference',str(reference.resolve()),'--render',str(destination/(case+'.exr')),'--report',str(destination/(case+'_worker.json')),'--case',case,'--require-specular','--scene-budgets','--minimum-glossy-bounces','2']
                if case=='phase_pi':args+=['--phase0-data',str(destination/'phase_0.npz')]
                if mode=='pt':args+=['--pt']
                if mode=='pt-guiding':args+=['--pt-guiding']
                jobs.append({'case':label,'arguments':args,'stdout':str(a.output.resolve()/(label+'_stdout.log')),'stderr':str(a.output.resolve()/(label+'_stderr.log')),'gate':manifest['gates_declared_before_render'],'phase_gate':{'absolute_phase_mean_error':.0004,'phase_difference_rmse':.0016} if case=='phase_pi' else None,'scene_sha256':digest(scene),'reference_sha256':digest(reference),'manifest_sha256':digest(folder/'manifest.json')})
    request={'scope':'Streamed flat mirror facets: four visible direct/1R/ordered2R routes, nine first-receiver checks; no refraction or smooth curvature claim.','worker':str(worker),'state':str(a.output.resolve()/'batch_state.json'),'jobs':jobs,'fixed_worker_settings':{'samples':128,'seed':19,'device':'METAL','adaptive':False,'denoise':False,'preserve_saved_budgets':True,'diffuse_bounces':0},'phase_gates_declared_before_actual_render':True}
    a.plan.write_text(json.dumps(request,indent=2)+'\n');print(len(jobs),'prepared jobs',a.plan.resolve())


if __name__=='__main__':main()
