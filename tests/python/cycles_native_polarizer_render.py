#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual native polarization suite. Run only after explicit render authorization.

Blender -b --python this.py -- --fixtures ACTIVE --output NEW --device CPU|METAL
No photon-mapping claim. Constant-world gates were declared before GPU testing.
"""
import argparse, hashlib, json, math, sys, time, traceback
from pathlib import Path
import bpy
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sockets():
    return sorted([(m.name, n.name, bool(n.inputs['Polarizer'].default_value),
                    float(n.inputs['Polarizer Angle'].default_value))
                   for m in bpy.data.materials if m.use_nodes
                   for n in m.node_tree.nodes if n.type == 'BSDF_GLASS'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['CPU','METAL'],default='CPU')
    p.add_argument('--photo',action='store_true',help='Also render matched-exposure qualitative photographic triplet')
    p.add_argument('--photo-only',action='store_true',help='Separate photographic run, after numeric acceptance')
    p.add_argument('--photo-bdpt',action='store_true',help='Append the same three photographic scenes through BDPT with separate provenance')
    p.add_argument('--modes',nargs='+',choices=['PT','BDPT','GUIDING'],default=['PT','BDPT','GUIDING'])
    p.add_argument('--osl',action='store_true',help='CPU-only Open Shading Language validation')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output.mkdir(parents=True,exist_ok=False)
    assert not a.osl or a.device=='CPU','OSL suite requires CPU; Metal OSL unsupported'
    manifest=json.loads((a.fixtures/'manifest.json').read_text());assert not manifest['pending_shader_api']
    gates=json.loads((a.fixtures/'pre_gpu_native_gates.json').read_text())
    report={'scope':'Native PT/BDPT/PT guiding ideal-film Malus; persistent OFF/ON/OFF. Photon mapping unsupported.',
            'binary_sha256':digest(bpy.app.binary_path),'script_sha256':digest(__file__),
            'gates_sha256':digest(a.fixtures/'pre_gpu_native_gates.json'),'gates':gates,'cases':{},'passed':True,'osl':a.osl,'modes':a.modes}
    def render_impl(key,expected,mode,photo=False):
        scene=bpy.context.scene;assert not scene.cycles.use_coherent_specular_connections
        assert scene.cycles.min_light_bounces>=8 and scene.cycles.min_transparent_bounces>=8
        scene.cycles.use_bidirectional_path_tracing=mode=='BDPT';scene.cycles.use_guiding=mode=='GUIDING'
        scene.cycles.bdpt_max_bounces=12
        scene.cycles.shading_system=a.osl
        scene.cycles.seed=19;scene.cycles.samples=256;scene.cycles.use_adaptive_sampling=False
        scene.cycles.use_denoising=False;scene.view_settings.exposure=0
        scene.cycles.device='CPU' if a.device=='CPU' else 'GPU'
        if a.device=='METAL':
            prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
            assert any(d.type=='METAL' for d in prefs.devices)
            for d in prefs.devices:d.use=d.type=='METAL'
        scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32'
        path=a.output/(key+'.exr');scene.render.filepath=str(path.resolve());start=time.perf_counter()
        assert 'FINISHED' in bpy.ops.render.render(write_still=True)
        image=bpy.data.images.load(str(path.resolve()),check_existing=False)
        rgb=np.asarray(image.pixels[:],dtype=np.float64).reshape(image.size[1],image.size[0],image.channels)[...,:3]
        record={'raw_exr':str(path.resolve()),'raw_sha256':digest(path),'seconds':time.perf_counter()-start,
                'mode':mode,'serialized_or_live_sockets':sockets(),'mean_rgb':rgb.mean(axis=(0,1)).tolist(),
                'finite':bool(np.isfinite(rgb).all()),'exposure':0,'samples':256,'min_light_bounces':8,
                'raw_denoised':False,'osl':bool(scene.cycles.shading_system)}
        if photo:
            # Blender applies the same saved AgX transform and fixed EV0 to every preview.
            scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_depth='8'
            image.save_render(str((a.output/(key+'.png')).resolve()),scene=scene)
            scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32'
            record['scope']='Qualitative finite-area photographic image; no central-ray pixel gate'
            record['passed']=record['finite']
        else:
            error=rgb-expected;record.update(expected_radiance=expected,mean_error=float(abs(error.mean())),
                rmse=float(np.sqrt(np.mean(error*error))),max_error=float(np.max(np.abs(error))))
            record['passed']=record['finite'] and record['mean_error']<=gates['native_malus_absolute_mean_error'] and record['rmse']<=gates['native_malus_absolute_rmse']
            if expected==0:record['passed'] &= float(np.max(rgb))<=gates['crossed_leakage_max']
        bpy.data.images.remove(image);report['cases'][key]=record;report['passed'] &= record['passed']
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def attempt(key,operation):
        # Preserve every completed/failed case before proceeding to the next.
        try:operation()
        except Exception as exc:
            report['cases'][key]={'passed':False,'exception':repr(exc),'traceback':traceback.format_exc()}
            report['passed']=False
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def render(key,expected,mode,photo=False):
        attempt(key,lambda:render_impl(key,expected,mode,photo))
    for mode in ([] if a.photo_only else a.modes):
        for group in ['native_malus','native_closed_slab']:
            for name,entry in manifest['cases'][group]['variants'].items():
                key=mode+'_'+group+'_'+name
                def serialized_case():
                    path=Path(entry['path']);assert digest(path)==entry['sha256']
                    bpy.ops.wm.open_mainfile(filepath=str(path));state=sockets()
                    # Check actual serialized sockets rather than pending custom properties.
                    on=sum(v[2] for v in state)
                    expected_on={'clear':0,'single':1,'parallel':2,'angle_30':2,'angle_45':2,'crossed':2,
                                 'three_0_45_90':3,'object_rotated_90':2,'polarized_front_back_axis_0_37':1}[name]
                    assert on==expected_on,(name,state)
                    render_impl(key,entry['expected_radiance'],mode)
                attempt(key,serialized_case)
        # Same scene and renderer process, no reload between state changes.
        bpy.ops.wm.open_mainfile(filepath=manifest['cases']['native_malus']['variants']['single']['path'])
        node=next(n for m in bpy.data.materials if m.use_nodes for n in m.node_tree.nodes
                  if n.type=='BSDF_GLASS' and n.inputs['Polarizer'].default_value)
        for index,on in enumerate([False,True,False]):
            node.inputs['Polarizer'].default_value=on;bpy.context.view_layer.update()
            render(mode+'_persistent_'+str(index)+('_ON' if on else '_OFF'),.5 if on else 1.,mode)
    if a.photo or a.photo_only or a.photo_bdpt:
        for mode in (['PT','BDPT'] if a.photo_bdpt else ['PT']):
            for name in ['unfiltered','pass_s','pass_p_glare_removed']:
                key=('photo_' if mode=='PT' else 'photo_bdpt_')+name
                def photographic_case():
                    entry=manifest['cases']['native_brewster_photo']['variants'][name];path=Path(entry['path'])
                    assert digest(path)==entry['sha256'];bpy.ops.wm.open_mainfile(filepath=str(path))
                    render_impl(key,None,mode,True)
                attempt(key,photographic_case)
    if not report['passed']:raise RuntimeError('Native physical gate failed; preserved report and raw EXRs')


if __name__=='__main__':main()
