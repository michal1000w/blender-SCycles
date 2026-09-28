#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Native incoherent PT Malus and photographic glare scenes, staged or active.

World/visible area emission illuminates delta paths. There are no coherent
point sources, terminal detectors, fitted pixels, or acceptance-render calls.
"""
import bpy,sys,math,json,hashlib
from pathlib import Path
import numpy as np
from mathutils import Vector,Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cycles_coherent_mirror_acceptance import configure_scene
from cycles_coherent_polarizer_scene import glass,pane,set_filter


def emitter(name,strength,color=(1,1,1)):
    mat=bpy.data.materials.new(name);mat.use_nodes=True;nodes=mat.node_tree.nodes;nodes.clear();em=nodes.new('ShaderNodeEmission');em.inputs['Color'].default_value=(*color,1);em.inputs['Strength'].default_value=strength;out=nodes.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(em.outputs[0],out.inputs['Surface']);return mat


def setup():
    scene,*_=configure_scene(False)
    for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
    scene.cycles.device='CPU';scene.cycles.use_coherent_specular_connections=False;scene.cycles.use_bidirectional_path_tracing=False;scene.cycles.use_guiding=False;scene.cycles.samples=256;scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False;scene.cycles.max_bounces=12;scene.cycles.min_light_bounces=8;scene.cycles.min_transparent_bounces=8;scene.cycles.transmission_bounces=10;scene.cycles.glossy_bounces=8;scene.cycles.diffuse_bounces=3;scene.cycles.transparent_max_bounces=8;scene.render.resolution_x=scene.render.resolution_y=64;scene.view_settings.view_transform='Standard';scene.view_settings.exposure=0
    return scene


def triangle_film(obj,half):
    # One finite planar primitive; the tested ROI lies far from every edge.
    # This avoids the archived legacy IOR1 coplanar-quad diagonal self-hit.
    obj.data.clear_geometry();obj.data.from_pydata([(-half,-half,0),(half,-half,0),(0,2*half,0)],[],[(0,1,2)])
    return obj


def camera(scene,location,target,ortho=None):
    bpy.ops.object.camera_add(location=location);obj=bpy.context.object;obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler();obj.data.clip_start=.001;obj.data.clip_end=100
    if ortho is not None:obj.data.type='ORTHO';obj.data.ortho_scale=ortho
    scene.camera=obj;return obj


def save(path):
    bpy.context.view_layer.update();bpy.ops.wm.save_as_mainfile(filepath=str(path));return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    args=sys.argv[sys.argv.index('--')+1:];output=Path(args[0]).resolve();output.mkdir(parents=True,exist_ok=False);legacy='--stage-legacy' in args;master={'scope':'Native ordinary incoherent PT polarization; world/area emission, coherenceOFF. Pending legacy scenes cannot validate polarization.','cases':{},'pending_shader_api':False}
    folder=output/'native_malus';folder.mkdir();scene=setup();scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=1
    filters=[];nodes=[]
    for z in (1.5,1.,.5):
        mat,node=glass('Native IOR1 ideal film '+str(z),1);filters.append(triangle_film(pane('Finite physical film '+str(z),(0,0,z),(1,0,0),(0,1,0),2,mat,'OFF'),2));nodes.append(node)
    camera(scene,(0,0,2),(0,0,0),.5);original=[obj.rotation_euler.to_quaternion() for obj in filters]
    variants=[('clear',(False,False,False),(0,0,0),1.),('single',(True,False,False),(0,0,0),.5),('parallel',(True,True,False),(0,0,0),.5),('angle_30',(True,True,False),(0,math.pi/6,0),.375),('angle_45',(True,True,False),(0,math.pi/4,0),.25),('crossed',(True,True,False),(0,math.pi/2,0),0.),('three_0_45_90',(True,True,True),(0,math.pi/4,math.pi/2),.125),('object_rotated_90',(True,True,False),(0,0,0),0.)]
    cases={};pending=False
    for name,enabled,angles,expected in variants:
        for obj,q in zip(filters,original):obj.rotation_euler=q.to_euler()
        if name=='object_rotated_90':filters[1].rotation_euler=(original[1]@Quaternion((0,0,1),math.pi/2)).to_euler()
        for node,on,angle in zip(nodes,enabled,angles):pending=set_filter(node,on,angle,legacy) or pending
        path=folder/(name+'.blend');entry=save(path);entry.update(expected_radiance=expected,pending_shader_api=pending);cases[name]=entry
        roi=np.ones((64,64),dtype=bool);ref=np.full((64,64),expected);np.savez_compressed(folder/(name+'_reference.npz'),roi_mask=roi,phase_0_radiance=ref,phase_pi_radiance=ref,incoherent_radiance=ref)
    master['cases']['native_malus']={'variants':cases,'world_radiance':1,'coherence':False,'pending_shader_api':pending,'reference':'Native unpolarized white World1: oneidealfilm.5,parallel.5,relativeangle.5cos²θ,three0→45→90.125,physicallyrotatedcrossed0'};master['pending_shader_api']|=pending
    # Closed slab: opposing physical face normals, same nonzero material axis.
    folder=output/'native_closed_slab';folder.mkdir();scene=setup();scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=1
    mat,node=glass('Closed IOR1 slab with one material axis',1);bpy.ops.mesh.primitive_cube_add(size=2,location=(0,0,1));slab=bpy.context.object;slab.name='Closed actual front/back Glass slab';slab.scale=(2,2,.1);slab.data.materials.append(mat)
    camera(scene,(.2,0,2),(0,0,0),.5);cases={};pending=False
    for name,on,expected in [('clear',False,1.),('polarized_front_back_axis_0_37',True,.5)]:
        pending=set_filter(node,on,.37,legacy) or pending;entry=save(folder/(name+'.blend'));entry.update(expected_radiance=expected,pending_shader_api=pending);cases[name]=entry
        ref=np.full((64,64),expected);np.savez_compressed(folder/(name+'_reference.npz'),roi_mask=np.ones((64,64),dtype=bool),phase_0_radiance=ref,phase_pi_radiance=ref,incoherent_radiance=ref)
    master['cases']['native_closed_slab']={'variants':cases,'coherence':False,'pending_shader_api':pending,'reference':'Oppositeface normals with the same object-localaxisθ.37 atoblique incidence: P²=P, unpolarizedworld transmission.5; never.25'};master['pending_shader_api']|=pending
    # Ordinary photographic arrangement: finite source angular extent is explicit.
    folder=output/'native_brewster_photo';folder.mkdir();scene=setup();scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0;scene.view_settings.view_transform='AgX';scene.render.resolution_x=640;scene.render.resolution_y=480;scene.cycles.use_denoising=True
    mat,water_node=glass('Ordinary nonpolarizing n1.5 Glass slab',1.5);set_filter(water_node,False,0,legacy);bpy.ops.mesh.primitive_cube_add(size=2,location=(0,0,-.0005));slab=bpy.context.object;slab.name='Physical Glass slab over checker floor';slab.scale=(.055,.055,.0005);slab.data.materials.append(mat)
    floor=bpy.data.materials.new('Colored checker floor');floor.use_nodes=True;ns=floor.node_tree.nodes;principled=ns.get('Principled BSDF');principled.inputs['Roughness'].default_value=.8;checker=ns.new('ShaderNodeTexChecker');checker.inputs['Color1'].default_value=(.04,.12,.65,1);checker.inputs['Color2'].default_value=(.8,.3,.025,1);checker.inputs['Scale'].default_value=10;floor.node_tree.links.new(checker.outputs['Color'],principled.inputs['Base Color']);pane('Actual colored checker floor',(0,0,-.003),(1,0,0),(0,1,0),.07,floor,'OFF')
    source=np.array([-.06,0,.04]);source_normal=-source/np.linalg.norm(source);u=np.array([0.,1.,0.]);v=np.cross(source_normal,u);pane('Visible finite unpolarized area emitter',source,u,v,.012,emitter('White area radiance20',20),'OFF')
    cam_position=np.array([.06,0,.04]);direction=cam_position/np.linalg.norm(cam_position);cam=camera(scene,cam_position,(0,0,0));cam.data.lens=48
    mat,node=glass('Photographic linear analyzer IOR1',1);analyzer=triangle_film(pane('Rotate this photographic polarizer',cam_position*.85,u,np.cross(direction,u),.027,mat,'OFF'),.027);base=analyzer.rotation_euler.to_quaternion();cases={};pending=False
    for name,on,angle,rotate in [('unfiltered',False,0,False),('pass_s',True,0,False),('angle_45',True,math.pi/4,False),('pass_p_glare_removed',True,math.pi/2,False),('object_rotated_90',True,0,True)]:
        analyzer.rotation_euler=(base@Quaternion((0,0,1),math.pi/2)).to_euler() if rotate else base.to_euler();pending=set_filter(node,on,angle,legacy) or pending;entry=save(folder/(name+'.blend'));entry.update(pending_shader_api=pending,angle_rad=angle,physical_object_rotated_90=rotate);cases[name]=entry
    master['cases']['native_brewster_photo']={'variants':cases,'coherence':False,'pending_shader_api':pending,'scope':'Qualitative native ordinary PT photograph; finite24mm source angular averaging and real n1.5slab+diffuse coloredfloor. No finite-source image claimed equalto central ray.','central_ray_analytic':{'brewster_angle_degrees':math.degrees(math.atan(1.5)),'unpolarized_top_surface_R_fraction':.07396449704142011,'ideal_pass_p_top_glare_fraction':0,'area_source_radiance':20},'samples':256,'resolution':[640,480],'denoise':True};master['pending_shader_api']|=pending
    (output/'manifest.json').write_text(json.dumps(master,indent=2)+'\n');(output/'pre_gpu_native_gates.json').write_text(json.dumps({'declared_before_native_polarizer_gpu':True,'native_malus_absolute_mean_error':1e-5,'native_malus_absolute_rmse':2e-5,'crossed_leakage_max':1e-8,'closed_slab_absolute_mean_error':1e-5,'reason':'WhiteunpolarizedWorld constant1 and deltaIOR1 paths have no geometric or pixelquadrature uncertainty; exactJones/Malus factors from independentprimary-source oracle. Missingfilter/independent50%perface scalar model fail. Nativephotographicfinite-area glaredemo qualitative only; centralBrewster ray separatelyvalidated analytically.'},indent=2)+'\n');print(json.dumps(master))

if __name__=='__main__':main()
