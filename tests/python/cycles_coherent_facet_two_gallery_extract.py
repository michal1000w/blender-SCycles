#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Immutable actual/ref readback after all prospective facet gates pass; no render."""
import argparse,json,sys,hashlib
from pathlib import Path
import bpy,numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--gates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);plan=json.loads(a.plan.read_text());gates=json.loads(a.gates.read_text());assert gates['passed'] and gates['complete'] and len(gates['cases'])==len(plan['jobs']);a.output.mkdir(parents=True,exist_ok=False)
    scene=bpy.context.scene;scene.view_settings.view_transform='Standard';scene.view_settings.look='None';scene.view_settings.exposure=-1;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_depth='8';items=[]
    for job in plan['jobs']:
        entry=gates['cases'][job['case']];args=job['arguments'];raw=Path(entry['raw_exr']);ref=Path(args[args.index('--reference')+1]);blend=Path(args[args.index('--blend')+1]);case=args[args.index('--case')+1];before=digest(raw);assert before==entry['raw_sha256'];assert digest(ref)==job['reference_sha256'] and digest(blend)==job['scene_sha256']
        image=bpy.data.images.load(str(raw),check_existing=False);actual=a.output/(job['case']+'_actual.png');image.save_render(str(actual.resolve()),scene=scene);bpy.data.images.remove(image);assert digest(raw)==before
        key={'phase_0':'phase_0_radiance','phase_pi':'phase_pi_radiance','incoherent_connector_control':'incoherent_radiance'}[case]
        with np.load(ref) as data:values=data[key]
        rgba=np.ones((*values.shape,4),dtype=np.float32);rgba[...,:3]=values[...,None];image=bpy.data.images.new('Independent reference',width=values.shape[1],height=values.shape[0],float_buffer=True);image.pixels.foreach_set(rgba.ravel());expected=a.output/(job['case']+'_reference.png');image.save_render(str(expected.resolve()),scene=scene);bpy.data.images.remove(image)
        items.append({'label':job['case'],'actual_png':str(actual.resolve()),'reference_png':str(expected.resolve()),'raw_exr':str(raw),'raw_sha256':before,'editable_blend':str(blend),'scene_sha256':job['scene_sha256'],'reference_sha256':job['reference_sha256'],'metrics':entry['metrics'],'phase_check':entry.get('phase_check',{})})
    (a.output/'gallery_manifest.json').write_text(json.dumps({'scope':'All actual Metal two-reflection cases and independent references; Standard fixedEV−1 for every image, no normalization/fitting; raw hashes unchanged','gates':str(a.gates.resolve()),'gates_sha256':digest(a.gates),'plan_sha256':digest(a.plan),'items':items},indent=2)+'\n')


if __name__=='__main__':main()
