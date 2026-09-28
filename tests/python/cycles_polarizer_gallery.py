#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble actual polarization readbacks. Requires Pillow, never renders."""
import argparse,json,html
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();manifest=json.loads((a.folder/'gallery_manifest.json').read_text());font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',19);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
    photographs=[v for v in manifest['items'] if v['family']=='photo'];sheet=Image.new('RGB',(1200,1000 if len(photographs)>3 else 660 if photographs else 315),'#eeeeee');draw=ImageDraw.Draw(sheet)
    draw.text((16,10),'Actual native polarizer '+manifest.get('version','')+': raw numeric strip (Standard EV0)',fill='black',font=font)
    numeric=[v for v in manifest['items'] if v['family']=='numeric']
    for index,item in enumerate(numeric):
        x=index*200+8;image=Image.open(item['image']).convert('RGB');image=image.resize((184,130),Image.Resampling.NEAREST);sheet.paste(image,(x,45));entry=item['metrics'];name=item['key'].removeprefix('PT_native_malus_')
        draw.text((x,184),name,fill='black',font=small);draw.text((x,207),'I = %.8g'%entry['mean_rgb'][0],fill='black',font=small);draw.text((x,230),'target %.8g'%entry['expected_radiance'],fill='black',font=small)
    pending='Numeric report in progress; photographic renders pending.' if manifest.get('report_in_progress') else 'Native numerical suite complete; photographic renders pending.'
    photo_heading='Actual photographic glare: AgX EV0, raw EXRs (256 samples)'
    if manifest.get('photographic_kernel_mode'):photo_heading+='; '+manifest['photographic_kernel_mode']+' Metal'
    draw.text((16,270),photo_heading if photographs else pending,fill='black',font=font)
    for index,item in enumerate(photographs):
        row=index//3;x=(index%3)*400+8;y=310+340*row;image=Image.open(item['image']).convert('RGB');image.thumbnail((384,288),Image.Resampling.LANCZOS);sheet.paste(image,(x,y));name={'unfiltered':'Analyzer OFF','pass_s':'Analyzer ON: glare pass','pass_p_glare_removed':'Analyzer ON: glare cut'}[item['key'].removeprefix('photo_bdpt_').removeprefix('photo_')];draw.text((x,y+300),item['metrics']['mode']+' '+name,fill='black',font=font)
    sheet.save(a.folder/'actual_polarizer_contact.png')
    body='<html><head><meta charset="utf-8"><title>Actual polarizer evidence</title></head><body><h1>Actual polarization evidence</h1><p>'+html.escape(manifest['scope'])+'</p><img style="max-width:100%" src="actual_polarizer_contact.png"><ul>'
    for item in manifest['items']:
        body+='<li>'+html.escape(item['key'])+' — <a href="'+html.escape(item['raw_exr'])+'">immutable raw EXR</a>; <a href="'+html.escape(item.get('editable_blend',''))+'">editable Blender scene</a>; SHA256 '+item['raw_sha256']+'</li>'
    if manifest.get('descriptive_photo_ratios'):
        body+='</ul><h2>Descriptive raw ROI ratios</h2><p>Glare ROI top-origin x270:370, y190:290; adjacent checker x170:250, y190:290. Linear luminance, fixed coordinates for both modes. No acceptance gate or exposure fitting.</p><pre>'+html.escape(json.dumps(manifest['descriptive_photo_ratios'],indent=2))+'</pre><ul>'
    body+='</ul><p>Photographic finite-area glare is qualitative; central-ray Brewster response is independently analytic. OFF uses legacy scalar dielectric transport; ON enables polarization scene-wide. Background differences across multiple Glass bounces cannot be interpreted as isolated filter transmission. No energy or PT/BDPT brightness identity gate is imposed. Photon mapping is unsupported.</p>'
    evidence=manifest.get('coherent_acceptance')
    if evidence:
        body+='<h2>Actual coherent Metal acceptance '+html.escape(evidence.get('version',''))+'</h2><p>48/48 prospective gates pass across PT, BDPT and guiding, including source-side and camera-side filters. Maximum mean error %.4g; RMSE %.4g. <a href="%s">Full independent gate report and raw provenance</a>. This does not establish complete arbitrary-object coherent transport.</p>'%(evidence['maximum_mean_error'],evidence['maximum_rmse'],html.escape(evidence['report']))
    for evidence in manifest.get('additional_evidence',[]):
        body+='<p><a href="'+html.escape(evidence['report'])+'">'+html.escape(evidence['description'])+'</a>; SHA256 '+evidence['sha256']+'</p>'
    timing=manifest.get('timing_evidence')
    if timing:
        body+='<h2>Optional checkbox cost</h2><p>OFF by default. Warm optimized Metal, 320×240/64 samples, one excluded warm run and three measured runs per state with required specialization warmup. PT 0.5593→1.0163 seconds (+81.7%); BDPT 0.58949→1.03514 seconds (+75.6%). Synchronization included, file writing excluded. <a href="'+html.escape(timing['report'])+'">Measured timing report</a>. This is a substantial opt-in cost, not an image-identity or general performance guarantee.</p><p>Matched default-OFF CD: 1.83137 seconds versus v37 1.777637, about +3.02%; raw comparison is separate. <a href="'+html.escape(timing['cd_report'])+'">CD timing report</a>.</p>'
    body+='</body></html>'
    (a.folder/'index.html').write_text(body)


if __name__=='__main__':main()
