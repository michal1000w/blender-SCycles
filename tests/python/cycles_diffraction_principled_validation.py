# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Validate supported Principled diffraction combinations and explicit limits."""
import json
from pathlib import Path
import sys
import bpy
out=Path(sys.argv[sys.argv.index('--')+1]);out.mkdir(parents=True,exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=1
s.cycles.use_adaptive_sampling=False;s.cycles.use_denoising=False
s.render.resolution_x=s.render.resolution_y=8
bpy.ops.mesh.primitive_cube_add()
m=bpy.data.materials.new('Validation');m.use_nodes=True;bpy.context.object.data.materials.append(m)
n=m.node_tree.nodes.get('Principled BSDF')
bpy.ops.object.camera_add(location=(0,0,5));s.camera=bpy.context.object
results=[]
for name,coverage,distribution,transmission,overrides,expected_error in [
    ('ordinary_default',0,'MULTI_GGX',0,{},False),
    ('unsupported_multiscatter',1,'MULTI_GGX',0,{},True),
    ('tiny_unsupported_multiscatter',1e-7,'MULTI_GGX',0,{},True),
    ('tiny_standard_transmission',1e-7,'GGX',.5,{},False),
    ('standard_transmission',1,'GGX',.5,{},False),
    ('tinted_transmission',1,'GGX',.5,{'Specular Tint':(.5,.7,1,1)},False),
    ('procedural_tinted_transmission',1,'GGX',.5,{'_linked_tint':True},False),
    ('physical_transmission_film',1,'GGX',.5,{'Thin Film Thickness':300},False),
    ('linked_physical_film',1,'GGX',.5,{'_linked_film':True},False),
    ('linked_tinted_film',1,'GGX',.5,{'_linked_film':True,'Specular Tint':(.5,.7,1,1)},False),
    ('tinted_film',1,'GGX',.5,{'Thin Film Thickness':300,'Specular Tint':(.5,.7,1,1)},False),
    ('dispersive_film',1,'GGX',.5,{'Thin Film Thickness':300,'Transmission Dispersion Scale':1},False),
    ('dispersive_transmission',1,'GGX',.5,{'Transmission Dispersion Scale':1},False),
    ('thin_wall',1,'GGX',.5,{'Thin Wall':True},False),
    ('thin_wall_film_dispersion',1,'GGX',.5,
     {'Thin Wall':True,'Thin Film Thickness':300,'Transmission Dispersion Scale':1},False),
]:
    for socket in ['Specular Tint', 'Thin Film Thickness']:
        for link in list(n.inputs[socket].links):
            m.node_tree.links.remove(link)
    n.inputs['Diffraction Weight'].default_value=coverage
    n.distribution=distribution
    n.inputs['Transmission Weight'].default_value=transmission
    for key,value in {'Specular Tint':(1,1,1,1),'Thin Film Thickness':0,
                      'Transmission Dispersion Scale':0,'Thin Wall':False,**overrides}.items():
        if not key.startswith('_'):
            n.inputs[key].default_value=value
    if overrides.get('_linked_tint'):
        noise=m.node_tree.nodes.new('ShaderNodeTexNoise')
        m.node_tree.links.new(noise.outputs['Color'],n.inputs['Specular Tint'])
    if overrides.get('_linked_film'):
        value=m.node_tree.nodes.new('ShaderNodeValue')
        value.outputs[0].default_value=300
        m.node_tree.links.new(value.outputs[0],n.inputs['Thin Film Thickness'])
    message=None
    try:
        bpy.ops.render.render()
    except RuntimeError as error:
        message=str(error)
    passed=(message is None) if not expected_error else bool(message and 'Principled diffraction' in message)
    results.append({'case':name,'passed':passed,'error':message})
(out/'report.json').write_text(json.dumps(results,indent=2)+'\n')
if not all(r['passed'] for r in results):
    raise RuntimeError('Principled support validation failed')
