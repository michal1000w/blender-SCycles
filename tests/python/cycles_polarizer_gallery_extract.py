#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read immutable actual native EXRs into fixed-exposure gallery PNGs; no render."""
import argparse,json,sys,hashlib
from pathlib import Path
import bpy
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--numeric',type=Path,required=True);p.add_argument('--photo',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--in-progress',action='store_true');a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output.mkdir(parents=True,exist_ok=False)
    numeric=json.loads(a.numeric.read_text());photo=json.loads(a.photo.read_text()) if a.photo else {'cases':{},'passed':None};scene=bpy.context.scene;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_depth='8';scene.view_settings.exposure=0;scene.view_settings.look='None';items=[]
    keys=['PT_native_malus_'+v for v in ['clear','single','parallel','angle_45','crossed','three_0_45_90']]
    photo_keys=['photo_unfiltered','photo_pass_s','photo_pass_p_glare_removed'] if a.photo else []
    photo_keys += [key for key in ['photo_bdpt_unfiltered','photo_bdpt_pass_s','photo_bdpt_pass_p_glare_removed'] if key in photo['cases']]
    for family,report,selected,transform in [('numeric',numeric,keys,'Standard'),('photo',photo,photo_keys,'AgX')]:
        scene.view_settings.view_transform=transform
        for key in selected:
            entry=report['cases'][key];raw=Path(entry['raw_exr']);before=digest(raw);assert before==entry['raw_sha256'];assert entry['raw_denoised'] is False
            image=bpy.data.images.load(str(raw),check_existing=False);roi_stats=None
            if family=='photo':
                # Blender pixel buffers are bottom-origin; requested photographic ROI is top-origin.
                assert tuple(image.size)==(640,480)
                rgb=np.asarray(image.pixels[:],dtype=float).reshape(480,640,image.channels)[::-1,:,:3]
                roi_stats={}
                for label,(x0,x1,y0,y1) in {'glare':(270,370,190,290),'adjacent_checker':(170,250,190,290)}.items():
                    values=rgb[y0:y1,x0:x1];luminance=values@np.array([.2126,.7152,.0722])
                    roi_stats[label]={'top_origin_xy_bounds':[x0,x1,y0,y1],'mean_linear_rgb':values.mean(axis=(0,1)).tolist(),'mean_linear_luminance':float(luminance.mean())}
            path=a.output/(key+'.png');image.save_render(str(path.resolve()),scene=scene);bpy.data.images.remove(image);assert digest(raw)==before
            items.append({'key':key,'family':family,'image':str(path.resolve()),'raw_exr':str(raw.resolve()),'raw_sha256':before,'display_transform':transform,'fixed_EV':0,'raw_denoised':False,'metrics':entry,'descriptive_raw_roi':roi_stats})
    result={'scope':'Actual immutable raw EXR gallery readback; Standard EV0 numeric, AgX EV0 photographic; no fitting or image-specific normalization','report_in_progress':a.in_progress,'numeric_report':str(a.numeric.resolve()),'numeric_report_sha256':digest(a.numeric),'photo_report':str(a.photo.resolve()) if a.photo else None,'photo_report_sha256':digest(a.photo) if a.photo else None,'numeric_passed':numeric['passed'],'photo_finite':photo['passed'],'items':items}
    result['descriptive_photo_ratios']={}
    for prefix in ['photo_','photo_bdpt_']:
        selected={item['key']:item for item in items if item['family']=='photo'}
        baseline=selected.get(prefix+'unfiltered')
        if baseline:
            ratios={}
            for name in ['pass_s','pass_p_glare_removed']:
                if prefix+name in selected:
                    ratios[name]={roi:selected[prefix+name]['descriptive_raw_roi'][roi]['mean_linear_luminance']/baseline['descriptive_raw_roi'][roi]['mean_linear_luminance'] for roi in ['glare','adjacent_checker']}
            result['descriptive_photo_ratios'][prefix]={'scope':'Descriptive raw linear-luminance ratios to same-mode OFF; fixed pixel ROIs, no gate or fitting. OFF uses legacy scalar dielectric transport; ON enables polarization scene-wide, so multiple-Glass-bounce background changes are not isolated analyzer transmission.','ratios':ratios}
    fixtures=Path(__file__).resolve().parents[1]/'output/diffraction/native_polarizer_v41'
    fixture_manifest=json.loads((fixtures/'manifest.json').read_text())
    for item in items:
        group='native_malus' if item['family']=='numeric' else 'native_brewster_photo'
        name=item['key'].removeprefix('PT_native_malus_').removeprefix('photo_bdpt_').removeprefix('photo_')
        source=fixture_manifest['cases'][group]['variants'][name];assert digest(source['path'])==source['sha256']
        item['editable_blend']=source['path'];item['editable_blend_sha256']=source['sha256']
    (a.output/'gallery_manifest.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
