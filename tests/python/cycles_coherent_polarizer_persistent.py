#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Persistent coherent source/camera polarizer OFF/ON/OFF, without scene reload.

Authorization to render must precede invocation. Standard phase_0 reference
arrays and roi_mask are read prospectively; no brightness fitting.
"""
import argparse,json,sys,traceback,time
from pathlib import Path
import bpy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cycles_native_polarizer_render import digest,sockets


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['CPU','METAL'],default='METAL')
    p.add_argument('--modes',nargs='+',choices=['PT','BDPT','GUIDING'],default=['PT','BDPT','GUIDING'])
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if a.device!='METAL':p.error('Coherent specular acceptance requires Metal; CPU is an unsupported backend, not a physics acceptance test')
    a.output.mkdir(parents=True,exist_ok=False)
    report={'scope':'Persistent coherent source-side single and camera-side parallel filters; OFF/ON/OFF, no reload between states','cases':{},'passed':True,'binary_sha256':digest(bpy.app.binary_path),'script_sha256':digest(__file__),'gates':{'mean_error':1e-4,'rmse':2e-4}}
    for mode in a.modes:
        for group,on_name in [('malus','single'),('camera_filters','parallel')]:
            folder=a.fixtures/group;manifest=json.loads((folder/'manifest.json').read_text());assert not manifest['pending_shader_api']
            entry=manifest['variants'][on_name];assert digest(entry['path'])==entry['sha256'];bpy.ops.wm.open_mainfile(filepath=entry['path'])
            nodes=[n for m in bpy.data.materials if m.use_nodes for n in m.node_tree.nodes if n.type=='BSDF_GLASS' and n.inputs['Polarizer'].default_value]
            assert len(nodes)==(1 if group=='malus' else 2),sockets()
            scene=bpy.context.scene;assert scene.cycles.use_coherent_specular_connections
            scene.cycles.use_bidirectional_path_tracing=mode=='BDPT';scene.cycles.use_guiding=mode=='GUIDING';scene.cycles.samples=256;scene.cycles.seed=19;scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False;scene.cycles.min_light_bounces=8;scene.cycles.min_transparent_bounces=8;scene.cycles.max_bounces=12;scene.cycles.bdpt_max_bounces=12
            scene.cycles.device='CPU' if a.device=='CPU' else 'GPU'
            if a.device=='METAL':
                prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices();assert any(d.type=='METAL' for d in prefs.devices)
                for d in prefs.devices:d.use=d.type=='METAL'
            for index,on in enumerate([False,True,False]):
                key=mode+'_'+group+'_'+str(index)+('_ON' if on else '_OFF')
                try:
                    for node in nodes:node.inputs['Polarizer'].default_value=on
                    bpy.context.view_layer.update();name=on_name if on else 'clear';refpath=folder/(name+'_reference.npz')
                    with np.load(refpath) as data:reference=data['phase_0_radiance'];roi=data['roi_mask'].astype(bool)
                    path=a.output/(key+'.exr');scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32';scene.render.filepath=str(path.resolve());start=time.perf_counter();assert 'FINISHED' in bpy.ops.render.render(write_still=True)
                    image=bpy.data.images.load(str(path.resolve()),check_existing=False);rgb=np.asarray(image.pixels[:],dtype=float).reshape(image.size[1],image.size[0],image.channels)[...,:3];bpy.data.images.remove(image)
                    error=rgb[roi]-reference[roi,None];mean_error=float(abs(error.mean()));rmse=float(np.sqrt(np.mean(error*error)));finite=bool(np.isfinite(rgb).all());passed=finite and mean_error<=1e-4 and rmse<=2e-4
                    report['cases'][key]={'passed':passed,'finite':finite,'mean_error':mean_error,'rmse':rmse,'raw_exr':str(path.resolve()),'raw_sha256':digest(path),'reference_sha256':digest(refpath),'live_sockets':sockets(),'seconds':time.perf_counter()-start,'samples':256,'seed':19,'denoise':False}
                    report['passed'] &= passed
                except Exception as exc:
                    report['cases'][key]={'passed':False,'exception':repr(exc),'traceback':traceback.format_exc()};report['passed']=False
                (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise RuntimeError('Persistent coherent polarization gate failed; raw outputs retained')


if __name__=='__main__':main()
