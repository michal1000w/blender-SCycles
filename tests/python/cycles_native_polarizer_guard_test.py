# SPDX-License-Identifier: Apache-2.0
"""CPU-only diagnostics for unsupported native polarization combinations."""
import argparse,json,sys
from pathlib import Path
import bpy
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output.parent.mkdir(parents=True,exist_ok=True)
r={'scope':'explicit Photon Mapping and rough matched-index thin-film guards, CPU only','cases':{},'passed':True}
for name in ['photon_mapping','rough_matched_film','linked_ior_rough_film','constant_link_rough_film']:
 bpy.ops.wm.open_mainfile(filepath=str(a.fixture.resolve()))
 s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.shading_system=False;s.cycles.samples=1
 s.render.threads_mode='FIXED';s.render.threads=1
 films=[(m,n) for m in bpy.data.materials if m.use_nodes for n in m.node_tree.nodes if n.type=='BSDF_GLASS' and n.inputs['Polarizer'].default_value]
 assert len(films)==1
 m,n=films[0]
 if name=='photon_mapping':
  s.cycles.use_photon_mapping=True;expected='Photon Mapping does not transport polarization'
 else:
  n.inputs['Roughness'].default_value=.2;n.inputs['IOR'].default_value=1
  n.inputs['Thin Film Thickness'].default_value=300
  if name.startswith('linked'):
   v=m.node_tree.nodes.new('ShaderNodeNewGeometry');m.node_tree.links.new(v.outputs['Position'],n.inputs['IOR'])
  if name.startswith('constant_link'):
   v=m.node_tree.nodes.new('ShaderNodeValue');v.outputs[0].default_value=1.5;m.node_tree.links.new(v.outputs[0],n.inputs['IOR'])
  expected=None if name.startswith('constant_link') else 'integrated transmission atom'
 try:
  bpy.ops.render.render()
  error='';passed=expected is None
 except RuntimeError as e:
  error=str(e);passed=expected is not None and expected in error
 r['cases'][name]={'expected_diagnostic':expected,'actual_error':error,'passed':passed};r['passed'] &= passed
 a.output.write_text(json.dumps(r,indent=2)+'\n')
assert r['passed'],r
