#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read immutable raw EXRs and reassess against both independent Jones versions.

No rendering, pixel modification, normalization, fitted scales or changed gates.
Run in Blender --factory-startup -b --python this.py -- render_dir old_ref_dir new_ref_dir report [--only-R].
"""
import bpy,sys,json,hashlib
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(actual,expected):
    difference=actual-expected
    return {'absolute_mean_error':float(abs(actual.mean()-expected.mean())),'absolute_rmse':float(np.sqrt(np.mean(difference*difference))),'maximum_absolute_error':float(np.max(abs(difference))),'actual_mean':float(actual.mean()),'reference_mean':float(expected.mean())}


def main():
    args=sys.argv[sys.argv.index('--')+1:];renders,old,new,output=map(lambda p:Path(p).resolve(),args[:4]);only_r='--only-R' in args
    original_results=renders/'results.json';baseline=json.loads(original_results.read_text());gates=baseline['gates'];report={'status':'partial_R_only' if only_r else 'terminal_full_reassessment','scope':'Same unmodified raw actual EXRs reassessed against original and independently corrected ideal mirror Jones reference. No rerender, scaling, fitting, or altered gates.','original_results':str(original_results),'original_results_sha256_at_read':sha(original_results),'gates':gates,'reference_correction_provenance':str(new/'reference_correction_provenance.json'),'cases':{},'phase_checks':{}}
    cases=('phase_0','phase_pi','incoherent_connector_control');keys=('phase_0_radiance','phase_pi_radiance','incoherent_radiance')
    for kind in (('R',) if only_r else ('R','TT')):
        original=old/kind/'virtual_source_reference.npz';corrected=new/kind/'virtual_source_reference.npz';refs={'original':np.load(original),'corrected':np.load(corrected)};roi=refs['original']['roi_mask'].astype(bool);assert np.array_equal(roi,refs['corrected']['roi_mask']);actuals={}
        for case,key in zip(cases,keys):
            path=renders/kind/(case+'.exr');worker=renders/kind/(case+'_worker.json');assert json.loads(worker.read_text())['worker_status']=='completed';before=sha(path);image=bpy.data.images.load(str(path),check_existing=False);width,height=map(int,image.size);channels=image.channels;data=np.array(image.pixels[:],dtype=float).reshape(height,width,channels);bpy.data.images.remove(image);actual=np.mean(data[:,:,:3],axis=2)[roi];assert before==sha(path) and np.isfinite(actual).all();actuals[case]=actual
            entry={'raw_exr':str(path),'raw_exr_sha256':before,'worker_report_sha256':sha(worker),'original_reference_sha256':sha(original),'corrected_reference_sha256':sha(corrected)}
            for version,reference in refs.items():
                m=metrics(actual,reference[key][roi]);m['pass']=m['absolute_mean_error']<=gates['absolute_mean_error_max_radiance'] and m['absolute_rmse']<=gates['absolute_rmse_max_radiance'];entry[version]=m
            # Independently reloaded EXR must reproduce archived worker extraction.
            archived=np.load(path.with_suffix('.npz'))['actual_roi'];entry['raw_reload_vs_worker_array_max_error']=float(np.max(abs(actual-archived)));assert entry['raw_reload_vs_worker_array_max_error']==0
            report['cases'][kind+'_'+case]=entry
        phase=actuals['phase_0']-actuals['phase_pi'];phase_entry={}
        for version,reference in refs.items():
            expected=(reference['phase_0_radiance']-reference['phase_pi_radiance'])[roi];m=metrics(phase,expected);m['pass']=m['absolute_mean_error']<=gates['phase_difference_mean_error_max_radiance'] and m['absolute_rmse']<=gates['phase_difference_rmse_max_radiance'];phase_entry[version]=m
        report['phase_checks'][kind]=phase_entry
    report['all_corrected_checks_pass']=all(c['corrected']['pass'] for c in report['cases'].values()) and all(p['corrected']['pass'] for p in report['phase_checks'].values());output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
