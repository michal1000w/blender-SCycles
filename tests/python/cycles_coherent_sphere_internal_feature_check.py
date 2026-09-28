#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Non-render actual raw EXR checks against independently predeclared internal gates."""
import bpy,sys,json,hashlib
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_raw(path):
    before=sha(path);image=bpy.data.images.load(str(path),check_existing=False);width,height=map(int,image.size);channels=image.channels;pixels=np.array(image.pixels[:],dtype=float).reshape(height,width,channels);bpy.data.images.remove(image);assert before==sha(path);return pixels[:,:,:3].mean(axis=2),before


def main():
    args=sys.argv[sys.argv.index('--')+1:];renders,fixtures,output=map(lambda p:Path(p).resolve(),args[:3]);gates_file=fixtures/'pre_gpu_feature_gates.json';gates=json.loads(gates_file.read_text());control_file=fixtures/'independent_paired_TRRT_response.npz';controls=np.load(control_file);roi=controls['roi_mask'].astype(bool);actual={};report={'gates_sha256':sha(gates_file),'control_sha256':sha(control_file),'scope':'Raw immutable EXR feature presence checks. No render, fit, rescaling or post-GPU gate adjustment.','raw':{},'checks':{}}
    for kind in ('TRT','TRRT'):
        actual[kind]={}
        for case in ('phase_pi','incoherent_connector_control'):
            path=renders/kind/(case+'.exr');worker=json.loads((renders/kind/(case+'_worker.json')).read_text());assert worker['worker_status']=='completed' and worker['coherent_specular_enabled'];data,digest=load_raw(path);assert np.isfinite(data[roi]).all();actual[kind][case]=data;report['raw'][kind+'_'+case]={'path':str(path),'sha256':digest,'samples':worker['samples']}
    for case,key in (('phase_pi','phase_pi_response'),('incoherent_connector_control','distinct_group_response')):
        assert report['raw']['TRT_'+case]['samples']==report['raw']['TRRT_'+case]['samples']
        observed=(actual['TRRT'][case]-actual['TRT'][case])[roi];expected=controls[key][roi];difference=observed-expected;mean_error=float(abs(observed.mean()-expected.mean()));rmse=float(np.sqrt(np.mean(difference*difference)));limit=gates['trrt_presence']['absolute_mean_response_tolerance'];passed=mean_error<=limit
        if case=='phase_pi':passed=passed and rmse<=gates['trrt_presence']['response_rmse_tolerance']
        report['checks']['paired_TRRT_'+case]={'actual_response_mean':float(observed.mean()),'independent_response_mean':float(expected.mean()),'absolute_mean_response_error':mean_error,'response_rmse':rmse,'pass':passed}
    reference=np.load(fixtures/'TRT'/'virtual_source_reference.npz');expected=reference['phase_pi_radiance'][roi];observed=actual['TRT']['phase_pi'][roi];mean_error=float(abs(observed.mean()-expected.mean()));report['checks']['TRT_presence']={'actual_phase_pi_mean':float(observed.mean()),'independent_phase_pi_mean':float(expected.mean()),'absolute_mean_error':mean_error,'pass':mean_error<=gates['trt_presence']['absolute_mean_tolerance']}
    report['all_feature_presence_checks_pass']=all(x['pass'] for x in report['checks'].values());output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
