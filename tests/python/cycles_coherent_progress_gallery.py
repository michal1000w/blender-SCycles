#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fixed-exposure evidence gallery from immutable EXR readback, no normalization."""
import json,sys,html
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    folder=Path(sys.argv[1]).resolve();manifest=json.loads((folder/'gallery_source_manifest.json').read_text());fontpath='/System/Library/Fonts/Supplemental/Arial.ttf';font=ImageFont.truetype(fontpath,16);small=ImageFont.truetype(fontpath,13);title=ImageFont.truetype(fontpath,24)
    panel=156;gap=12;left=140;top=118;rowheight=225;width=left+6*(panel+gap)+20;height=top+4*rowheight+62;canvas=Image.new('RGB',(width,height),(19,24,31));draw=ImageDraw.Draw(canvas);draw.text((24,20),'Coherent native optics — actual Metal images and independent fields',font=title,fill=(237,242,248));draw.text((24,56),'Same fixed EV within each feature. ROI 26×26 from raw 32×32; actual 128 samples, adaptive OFF, denoise OFF.',font=small,fill=(188,202,217));draw.text((24,78),'Display-only sRGB conversion; raw EXRs, reference versions and numerical reports stay unchanged.',font=small,fill=(188,202,217))
    headers=('Actual phase 0','Reference phase 0','Actual phase π','Reference phase π','Actual distinct groups','Reference distinct groups')
    for index,header in enumerate(headers):draw.text((left+index*(panel+gap),top-24),header,font=small,fill=(226,234,244))
    html_sections=[]
    for row,(feature,info) in enumerate(manifest['features'].items()):
        values=np.load(folder/feature/'immutable_readback.npz');roi=values['roi_mask'];ys,xs=np.where(roi);region=(slice(ys.min(),ys.max()+1),slice(xs.min(),xs.max()+1));y=top+row*rowheight;draw.text((20,y+8),feature,font=title,fill=(131,196,255));draw.text((20,y+43),f'EV {info["display_EV"]:+g}',font=font,fill=(218,226,236));draw.text((20,y+71),'Actual / oracle',font=small,fill=(184,199,217));draw.text((20,y+92),'shared display',font=small,fill=(184,199,217));tiles=[]
        for caseindex,case in enumerate(('phase_0','phase_pi','incoherent_connector_control')):
            meta=info['metrics'][case]
            for reference in (False,True):
                raw=values[case+'_reference'] if reference else values[case+'_actual_rgb'];raw=raw[region]
                if raw.ndim==2:raw=np.repeat(raw[:,:,None],3,axis=2)
                linear=np.clip(raw*2.**info['display_EV'],0,1);display=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055);pixels=np.rint(display*255).astype(np.uint8);image=Image.fromarray(pixels,'RGB');name=feature+'_'+case+('_reference' if reference else '_actual')+'.png';image.save(folder/name)
                x=left+(2*caseindex+int(reference))*(panel+gap);canvas.paste(image.resize((panel,panel),Image.Resampling.NEAREST),(x,y));draw.rectangle((x,y,x+panel,y+panel),outline=(67,80,96));caption='Independent field' if reference else f'RMSE {meta["roi_rmse"]:.3g}';draw.text((x,y+panel+9),caption,font=small,fill=(196,213,232));tiles.append(f'<figure><img src="{name}" alt="{feature} {case} {"independent reference" if reference else "actual Metal"}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        draw.text((left,y+panel+33),info['reference_version'],font=small,fill=(163,187,213))
        links=' · '.join(f'<a href="{html.escape(meta["editable_blend"],quote=True)}">{case} editable .blend</a>' for case,meta in info['metrics'].items())
        html_sections.append(f'<section><h2>{feature} — fixed EV {info["display_EV"]:+g}</h2><p>{html.escape(info["scope"])}. {html.escape(info["reference_version"])}.</p><div class="grid">{"".join(tiles)}</div><p>{links}</p><details><summary>Raw EXR paths, hashes and radiance errors</summary><pre>{html.escape(json.dumps(info["metrics"],indent=2))}</pre></details></section>')
    draw.text((24,height-39),'Native sphere R / TT with finite mirror; Glass sphere TRT / TRRT full visible path inventories. Polarizer scenes remain pending.',font=small,fill=(185,201,218));canvas.save(folder/'coherent_actual_reference_contact.png')
    page='<!doctype html><html><meta charset="utf-8"><title>Coherent optics evidence</title><style>body{margin:32px;background:#13181f;color:#e2eaf3;font:16px Arial;max-width:1280px}h1{font-size:28px}h2{color:#83c4ff;margin-top:38px}.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}figure{margin:0}img{width:100%;image-rendering:pixelated}figcaption{font-size:13px;color:#bdd0e5;margin-top:8px}a{color:#9bc9ff}pre{overflow:auto;font-size:12px}p{line-height:1.5}aside{padding:14px;background:#223041;border-left:3px solid #83c4ff}</style><h1>Coherent optics: actual Metal renders / independent references</h1><p>Four features, phases 0 / π and distinct source groups. Fixed EV per feature; no image-specific normalization. Display uses the same sRGB transfer and the predeclared ROI. Original raw EXRs and reports remain unchanged.</p><p><a href="coherent_actual_reference_contact.png">Open full contact sheet</a> · <a href="gallery_source_manifest.json">Evidence manifest and raw hashes</a></p>' + ''.join(html_sections) + '<aside><strong>Separate physical illustration:</strong> R/TT labeled wide ordinary previews are authored at 640×480, 256 samples, denoise ON. They use separate noncoherent area/background/ground illumination and are not acceptance references. Native and coherent polarizer scenes are staged pending the new shader API/state integration; they have not passed GPU acceptance.</aside></html>'
    (folder/'index.html').write_text(page)
    print(folder/'coherent_actual_reference_contact.png')

if __name__=='__main__':main()
