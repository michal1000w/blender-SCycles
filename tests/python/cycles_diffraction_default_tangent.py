# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare implicit Glass tangent with an explicit Geometry Tangent connection."""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
a.output.mkdir(parents=True,exist_ok=False)
results=[]
for explicit in (False,True):
    bpy.ops.wm.open_mainfile(filepath=str(a.scene.resolve()))
    s=bpy.context.scene
    s.cycles.samples=128
    s.cycles.seed=11
    s.cycles.use_adaptive_sampling=False
    s.cycles.use_denoising=False
    s.render.resolution_x=360
    s.render.resolution_y=200
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='METAL'
    prefs.get_devices()
    for d in prefs.devices:
        d.use=d.type=='METAL'
    assert any(d.use for d in prefs.devices)
    s.cycles.device='GPU'
    count=0
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        for node in list(m.node_tree.nodes):
            if node.bl_idname != 'ShaderNodeBsdfGlass':
                continue
            count+=1
            for link in list(node.inputs['Tangent'].links):
                m.node_tree.links.remove(link)
            if explicit:
                geom=m.node_tree.nodes.new('ShaderNodeNewGeometry')
                m.node_tree.links.new(geom.outputs['Tangent'],node.inputs['Tangent'])
    assert count==3
    name='explicit' if explicit else 'implicit'
    s.render.image_settings.file_format='OPEN_EXR'
    s.render.image_settings.color_depth='32'
    s.render.filepath=str(a.output/(name+'.exr'))
    bpy.ops.wm.save_as_mainfile(filepath=str(a.output/(name+'.blend')))
    bpy.ops.render.render(write_still=True)
    im=bpy.data.images.load(s.render.filepath,check_existing=False)
    pixels=tuple(im.pixels[:])
    assert pixels and all(math.isfinite(x) for x in pixels)
    results.append(pixels)
    bpy.data.images.remove(im)
errors=[abs(x-y) for x,y in zip(*results)]
reference=sum(abs(x) for x in results[1])/len(errors)
mae=sum(errors)/len(errors)
report={'samples':128,'adaptive_sampling':False,'denoising':False,
        'max_absolute_error':max(errors),'mean_absolute_error':mae,
        'reference_mean_absolute_value':reference,
        'passed':mae<=1e-5*max(1,reference),
        'scope':'Same-seed implicit versus explicit geometry tangent, three Glass materials on Metal.'}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if not report['passed']:
    raise RuntimeError('Implicit and explicit Glass tangent renders differ')
