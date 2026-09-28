#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-GPU independent feature omission controls; reads saved geometry, never renders."""
import bpy,sys,json,math,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cycles_coherent_sphere_internal_fields import all_fields,radiance


def main():
    root=Path(sys.argv[sys.argv.index('--')+1]).resolve();arrays={};report={'computed_without_rendered_pixels':True,'cases':{}}
    variants=[([0,0],[1,1]),([0,math.pi],[1,1]),([0,0],[1,2])]
    for name,m in (('TRT',1),('TRRT',2)):
        folder=root/name;bpy.ops.wm.open_mainfile(filepath=str(folder/'phase_0.blend'));scene=bpy.context.scene;assert scene.cycles.use_coherent_specular_connections
        lights=sorted([o for o in scene.objects if o.type=='LIGHT'],key=lambda o:o.location.y);sources=[np.array(tuple(l.location)) for l in lights];powers=[float(l.data.energy) for l in lights]
        detector=next(o for o in scene.objects if o.cycles.coherent_interface=='DETECTOR');stored=np.array([tuple(v.co) for v in detector.data.vertices]);normal=np.cross(stored[1]-stored[0],stored[2]-stored[0]);normal/=np.linalg.norm(normal);camera=scene.camera;matrix=np.array([list(row) for row in camera.matrix_world]);right,up,forward=matrix[:3,0],matrix[:3,1],-matrix[:3,2];width=float(camera.data.ortho_scale);n=32;radius=float(np.float32(.01))
        def pixel(row,col,ox,oy):
            origin=matrix[:3,3]+((col+ox)/n-.5)*width*right+((row+oy)/n-.5)*width*up
            return origin+np.dot(stored[0]-origin,normal)/np.dot(forward,normal)*forward
        result=np.zeros((3,3,n,n));start=time.perf_counter()
        for row in range(n):
            for col in range(n):
                for ox in (.25,.75):
                    for oy in (.25,.75):
                        r=pixel(row,col,ox,oy);full=[all_fields(s,r,normal,radius,1.5,m,p) for s,p in zip(sources,powers)]
                        sets=[full,[[t for t in pp if t[2] not in ('TRT','TRRT')] for pp in full],[[t for t in pp if t[2]!='TRRT'] for pp in full]]
                        for control,terms in enumerate(sets):
                            for variant,(phases,groups) in enumerate(variants):result[control,variant,row,col]+=radiance(terms,phases,groups)/4
        original=np.load(folder/'virtual_source_reference.npz');roi=original['roi_mask'].astype(bool);expected=np.array([original[k] for k in ('phase_0_radiance','phase_pi_radiance','incoherent_radiance')]);error=float(np.max(abs(expected-result[0])));assert error<1e-12,error
        arrays[name]=result;metrics={}
        for index,label in enumerate(('phase_0','phase_pi','distinct_groups')):
            metrics[label]={}
            for control,label_control in ((1,'omit_all_internal'),(2,'omit_TRRT')):
                delta=(result[0,index]-result[control,index])[roi];metrics[label][label_control]={'response_mean':float(delta.mean()),'omission_rmse':float(np.sqrt(np.mean(delta*delta))),'response_max_absolute':float(np.max(abs(delta)))}
        np.savez_compressed(folder/'independent_feature_controls.npz',full=result[0],omit_all_internal=result[1],omit_TRRT=result[2],roi_mask=roi)
        report['cases'][name]={'metrics':metrics,'exact_primary_reference_reproduction_max_error':error,'seconds':time.perf_counter()-start}
    roi=np.load(root/'TRRT'/'virtual_source_reference.npz')['roi_mask'].astype(bool);paired=arrays['TRRT'][0]-arrays['TRT'][0];np.savez_compressed(root/'independent_paired_TRRT_response.npz',phase_0_response=paired[0],phase_pi_response=paired[1],distinct_group_response=paired[2],roi_mask=roi)
    report['paired_TRRT_presence']={label:{'response_mean':float(paired[j][roi].mean()),'zero_response_omission_rmse':float(np.sqrt(np.mean(paired[j][roi]**2)))} for j,label in enumerate(('phase_0','phase_pi','distinct_groups'))}
    (root/'independent_feature_controls.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
