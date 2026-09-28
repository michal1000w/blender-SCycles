#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Non-render immutable raw EXR extraction for the coherent evidence gallery."""
import bpy,sys,json,hashlib
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    repo=Path(__file__).resolve().parents[2];output=Path(sys.argv[sys.argv.index('--')+1]).resolve();output.mkdir(parents=True,exist_ok=False);manifest={'scope':'Unmodified actual raw EXR readback and independent reference arrays; fixed display EV is applied later only to gallery PNGs.','features':{}}
    specs=[('R','coherent_mixed_sphere_render_v39','coherent_sphere_planar_v39_1mw_jones',-2),('TT','coherent_mixed_sphere_render_v39','coherent_sphere_planar_v39_1mw_jones',-1),('TRT','coherent_sphere_internal_render_v40','coherent_sphere_internal_v40_1mw',2),('TRRT','coherent_sphere_internal_render_v40','coherent_sphere_internal_v40_1mw',2)]
    for feature,render_dir,ref_dir,ev in specs:
        folder=output/feature;folder.mkdir();renders=repo/'build/tests/python'/render_dir/feature;fixtures=repo/'tests/output/diffraction'/ref_dir/feature;reference_path=fixtures/'virtual_source_reference.npz';reference=np.load(reference_path);roi=reference['roi_mask'].astype(bool);m={};arrays={'roi_mask':roi}
        for case,key in zip(('phase_0','phase_pi','incoherent_connector_control'),('phase_0_radiance','phase_pi_radiance','incoherent_radiance')):
            path=renders/(case+'.exr');before=sha(path);image=bpy.data.images.load(str(path),check_existing=False);w,h=map(int,image.size);channels=image.channels;actual=np.array(image.pixels[:],dtype=float).reshape(h,w,channels)[:,:,:3];bpy.data.images.remove(image);assert before==sha(path);gray=actual.mean(axis=2);expected=reference[key];difference=(gray-expected)[roi];arrays[case+'_actual_rgb']=actual;arrays[case+'_reference']=expected;m[case]={'actual_exr':str(path),'actual_sha256':before,'reference_sha256':sha(reference_path),'roi_rmse':float(np.sqrt(np.mean(difference*difference))),'roi_absolute_mean_error':float(abs(difference.mean())),'editable_blend':json.loads((renders/(case+'_worker.json')).read_text())['blend']}
        np.savez_compressed(folder/'immutable_readback.npz',**arrays);manifest['features'][feature]={'display_EV':ev,'samples':128,'reference_version':'corrected ideal Mirror Jones' if feature in ('R','TT') else 'all-winding Fresnel/Jones/Morse','reference_npz':str(reference_path),'metrics':m,'scope':'Native sphere plus finite mirror' if feature in ('R','TT') else 'Native Glass sphere including all direct/exterior/internal histories through '+feature}
    (output/'gallery_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'output':str(output),'features':list(manifest['features'])}))

if __name__=='__main__':main()
