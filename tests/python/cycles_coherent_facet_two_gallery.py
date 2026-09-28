#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble all actual facet cases, no rendering or image-specific exposure."""
import argparse,json,html
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();m=json.loads((a.folder/'gallery_manifest.json').read_text());font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14);sheet=Image.new('RGB',(1080,40+205*((len(m['items'])+2)//3)),'#eeeeee');draw=ImageDraw.Draw(sheet);draw.text((8,8),'Actual two-reflection cases / independent references; fixed Standard EV−1',fill='black',font=font);cards=[]
    for i,item in enumerate(m['items']):
        row=i//3;pair=i%3;x=pair*360;y=40+row*205
        for j,(key,title) in enumerate([('actual_png','actual'),('reference_png','reference')]):
            im=Image.open(item[key]).convert('RGB').resize((160,160),Image.Resampling.NEAREST);sheet.paste(im,(x+j*180+8,y+20));draw.text((x+j*180+8,y),title,fill='black',font=font)
        short=item['label'].replace('incoherent_connector_control','distinct');draw.text((x+8,y+182),short+' RMSE %.3g'%item['metrics']['absolute_rmse'],fill='black',font=font)
        cards.append('<article><h3>'+html.escape(item['label'])+'</h3><img width="240" src="'+Path(item['actual_png']).name+'"><img width="240" src="'+Path(item['reference_png']).name+'"><p>Actual / independent reference. <a href="'+html.escape(item['raw_exr'])+'">Raw EXR</a> · <a href="'+html.escape(item['editable_blend'])+'">Editable Blender scene</a>; RMSE '+str(item['metrics']['absolute_rmse'])+'</p></article>')
    sheet.save(a.folder/'actual_reference_contact.png');(a.folder/'index.html').write_text('<html><meta charset="utf-8"><h1>Actual streamed mirror facets</h1><p>'+html.escape(m['scope'])+'</p><p>Two finite ideal-white mirror objects with four triangles, direct and all visible one/two-reflection paths, two1mW sources, first Lambertian receiver only. No coherent refraction, smooth-wave curvature or chains beyond two reflections.</p><img width="1080" src="actual_reference_contact.png">'+''.join(cards)+'</html>')


if __name__=='__main__':main()
