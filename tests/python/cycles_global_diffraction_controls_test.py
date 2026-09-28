#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU constant-world filter ON/OFF/ON and nondestructive saved controls.
Run only against the new installed build; historical fixtures are read-only.
"""
import bpy,json,sys,math,hashlib
from pathlib import Path
import numpy as np

def main():
 args=sys.argv[sys.argv.index('--')+1:];fixture=Path(args[0]).resolve();out=Path(args[1]).resolve();out.mkdir(parents=True,exist_ok=False);before=hashlib.sha256(fixture.read_bytes()).hexdigest();bpy.ops.wm.open_mainfile(filepath=str(fixture));s=bpy.context.scene
 props=['use_diffraction_effects','use_material_diffraction','use_polarization','use_coherent_interference'];assert all(getattr(s.cycles,p) is True for p in props)
 s.cycles.device='CPU';s.cycles.use_bidirectional_path_tracing=False;s.cycles.use_guiding=False;s.cycles.samples=1;s.cycles.use_adaptive_sampling=False;s.cycles.use_denoising=False;s.render.resolution_x=s.render.resolution_y=8
 glass=[n for m in bpy.data.materials if m.use_nodes for n in m.node_tree.nodes if n.bl_idname=='ShaderNodeBsdfGlass' and n.inputs['Polarizer'].default_value][0];assert glass.inputs['Polarizer'].default_value;glass.inputs['Polarizer Angle'].default_value=.37
 value=glass.id_data.nodes.new('ShaderNodeValue');value.outputs[0].default_value=.73;glass.id_data.links.new(value.outputs[0],glass.inputs['Diffraction Weight']);glass.inputs['Diffraction Depth'].default_value=0
 bpy.ops.object.light_add(type='POINT',location=(10,10,10));source=bpy.context.object;source.hide_render=True;source.data.cycles.coherence_group=7
 checks=[]
 for name,changes,expected in [('all_on',{},.5),('master_off',{'use_diffraction_effects':False},1.),('master_restored',{'use_diffraction_effects':True},.5),('polarization_off',{'use_polarization':False},1.),('polarization_restored',{'use_polarization':True},.5),('material_diffraction_off',{'use_material_diffraction':False},.5),('material_diffraction_restored',{'use_material_diffraction':True},.5),('coherent_interference_off',{'use_coherent_interference':False},.5),('coherent_interference_restored',{'use_coherent_interference':True},.5)]:
  for key,v in changes.items():setattr(s.cycles,key,v)
  s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32';s.render.filepath=str(out/(name+'.exr'));bpy.ops.render.render(write_still=True);im=bpy.data.images.load(s.render.filepath,check_existing=False);a=np.array(im.pixels[:],dtype=float).reshape(-1,4)[:,:3];bpy.data.images.remove(im);mean=float(a.mean());assert np.isfinite(a).all() and abs(mean-expected)<1e-5,(name,mean,expected)
  assert glass.inputs['Polarizer'].default_value and abs(glass.inputs['Polarizer Angle'].default_value-.37)<1e-7 and glass.inputs['Diffraction Weight'].is_linked and source.data.cycles.coherence_group==7
  checks.append({'case':name,'expected':expected,'mean':mean,'passed':True,'settings':{p:getattr(s.cycles,p) for p in props}});(out/'report.json').write_text(json.dumps({'complete':False,'checks':checks},indent=2))
 for p in props:setattr(s.cycles,p,False)
 saved=out/'all_disabled_preserved_material.blend';bpy.ops.wm.save_as_mainfile(filepath=str(saved));bpy.ops.wm.open_mainfile(filepath=str(saved));s=bpy.context.scene;assert all(getattr(s.cycles,p) is False for p in props)
 glass=[n for m in bpy.data.materials if m.use_nodes for n in m.node_tree.nodes if n.bl_idname=='ShaderNodeBsdfGlass' and n.inputs['Polarizer'].default_value][0];assert glass.inputs['Polarizer'].default_value and glass.inputs['Diffraction Weight'].is_linked;assert any(l.cycles.coherence_group==7 for l in bpy.data.lights)
 for p in props:setattr(s.cycles,p,True)
 s.render.filepath=str(out/'reload_restored.exr');bpy.ops.render.render(write_still=True);im=bpy.data.images.load(s.render.filepath,check_existing=False);mean=float(np.array(im.pixels[:]).reshape(-1,4)[:,:3].mean());bpy.data.images.remove(im);assert abs(mean-.5)<1e-5;checks.append({'case':'reload_restored','expected':.5,'mean':mean,'passed':True})
 # Dedicated Realistic node with nonzero relief must bypass to ordinary Glass
 # under masterOFF; this checks native nonblack output rather than ON solver.
 s.cycles.use_diffraction_effects=False;s.cycles.samples=16
 tree=glass.id_data;output=next(n for n in tree.nodes if n.bl_idname=='ShaderNodeOutputMaterial');diff=tree.nodes.new('ShaderNodeBsdfDiffraction');diff.quality='REALISTIC';diff.depth=1.2e-7;diff.inputs['Color'].default_value=(.8,.7,.6,1.);tree.links.new(diff.outputs[0],output.inputs['Surface'])
 def read_render(name):
  s.render.filepath=str(out/(name+'.exr'));bpy.ops.render.render(write_still=True);im=bpy.data.images.load(s.render.filepath,check_existing=False);a=np.array(im.pixels[:]).reshape(-1,4)[:,:3].copy();bpy.data.images.remove(im);return a
 disabled=read_render('dedicated_diffraction_master_off');native=tree.nodes.new('ShaderNodeBsdfGlass');native.inputs['IOR'].default_value=diff.substrate_ior/diff.incident_ior;native.inputs['Roughness'].default_value=0;native.inputs['Color'].default_value=(.8,.7,.6,1.);tree.links.new(native.outputs[0],output.inputs['Surface']);reference=read_render('ordinary_glass_reference');error=float(np.max(np.abs(disabled-reference)));assert np.isfinite(disabled).all() and disabled.mean()>0 and error<1e-6,(disabled.mean(),error)
 checks.append({'case':'dedicated_realistic_master_off_native_glass','mean':float(disabled.mean()),'maximum_absolute_difference_from_native_glass':error,'passed':True,'scope':'MasterOFF nonzero-relief Realistic dedicated node native fallback; no ON solver claim'})
 assert hashlib.sha256(fixture.read_bytes()).hexdigest()==before
 (out/'report.json').write_text(json.dumps({'complete':True,'passed':True,'backend':'CPU SVM','scope':'Constant-world ideal IOR1 filter, ten incremental same-process renders, all-disabled save/load; original socket/link/source data preserved. Zero-depth weight link proves nondestructive authoring, not nonzero-relief grating physics. No GPU/benchmark.','checks':checks,'save_load_all_disabled_passed':True,'saved_scene':str(saved),'historical_fixture_sha256':before},indent=2)+'\n')

if __name__=='__main__':main()
