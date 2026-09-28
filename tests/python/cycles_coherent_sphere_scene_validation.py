#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read saved sphere authoring/material controls without rendering."""
import hashlib,json,sys
from pathlib import Path
import bpy

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=Path(sys.argv[sys.argv.index('--')+1]);report={}
for name in ('phase_0','phase_pi','incoherent_connector_control','reject_internal_source','reject_glass_sphere','reject_ellipsoid','reject_multiple_events'):
 path=root/(name+'.blend');bpy.ops.wm.open_mainfile(filepath=str(path));scene=bpy.context.scene
 sphere=bpy.data.objects['Native analytic mirror sphere'];detector=bpy.data.objects['Flat white detector']
 material=sphere.data.materials[0];glossy=next(n for n in material.node_tree.nodes if n.bl_idname in ('ShaderNodeBsdfGlossy','ShaderNodeBsdfAnisotropic'))
 assert tuple(glossy.inputs['Color'].default_value)==(1.,1.,1.,1.)
 assert glossy.inputs['Roughness'].default_value==0 and glossy.distribution=='GGX'
 for socket in ('Color','Roughness','Normal','Anisotropy','Diffraction Weight'):assert not glossy.inputs[socket].is_linked
 assert glossy.inputs['Anisotropy'].default_value==0 and glossy.inputs['Diffraction Weight'].default_value==0
 assert len(material.node_tree.nodes)==2 and not material.node_tree.nodes.get('Material Output').inputs['Displacement'].is_linked
 diffuse=next(n for n in detector.data.materials[0].node_tree.nodes if n.type=='BSDF_DIFFUSE')
 assert tuple(diffuse.inputs['Color'].default_value)==(1.,1.,1.,1.) and diffuse.inputs['Roughness'].default_value==0
 assert not diffuse.inputs['Normal'].is_linked and all(not p.use_smooth for p in detector.data.polygons)
 assert detector.cycles.coherent_interface=='DETECTOR'
 expected_mode='GLASS' if name=='reject_glass_sphere' else 'MIRROR';assert sphere.cycles.coherent_interface==expected_mode
 group=sphere.modifiers[0].node_group;points=group.nodes['NativeSphere'];assert points.mode=='VERTICES' and points.inputs['Radius'].default_value==.25
 output=next(n for n in group.nodes if n.bl_idname=='NodeGroupOutput');assert output.inputs['Geometry'].links[0].from_node.bl_idname=='GeometryNodeSetMaterial'
 assert scene.cycles.use_coherent_specular_connections and scene.cycles.coherent_polarization_mode=='VECTOR'
 assert scene.cycles.coherent_max_interface_events==(2 if name=='reject_multiple_events' else 1)
 assert not scene.cycles.use_adaptive_sampling and not scene.cycles.use_denoising and scene.cycles.samples==128
 assert scene.cycles.diffuse_bounces==0 and scene.cycles.glossy_bounces==1 and scene.cycles.transmission_bounces==0
 assert scene.cycles.sample_clamp_direct==scene.cycles.sample_clamp_indirect==0
 assert scene.cycles.pixel_filter_type=='BOX' and scene.cycles.filter_width==1
 assert scene.render.resolution_x==scene.render.resolution_y==256
 lights=sorted((o for o in scene.objects if o.type=='LIGHT'),key=lambda o:o.name)
 assert len(lights)==2 and all(o.data.type=='POINT' and o.data.energy==100 and o.data.shadow_soft_size==0 for o in lights)
 assert all(o.data.cycles.coherence_length_m>0 and o.data.cycles.coherence_wavelength_nm==550 for o in lights)
 groups=sorted(o.data.cycles.coherence_group for o in lights);assert groups==([1,2] if name=='incoherent_connector_control' else [1,1])
 report[name]={'saved_scene_sha256':sha(path),'white_unit_mirror_verified':True,'native_GN_points_radius_m':.25,'settings_verified':True,'sphere_mode':expected_mode,'source_groups':groups}
(root/'saved_scene_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'saved_scenes_verified':len(report),'rendered':False}))
