#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Author editable Malus films and a Brewster-glare photographic filter demo.

Use a polarizer-enabled Blender build, or --stage-legacy to author pending
geometry in an older build. Pending scenes are explicitly not GPU-ready.
"""
import bpy,sys,json,math,time,hashlib
from pathlib import Path
import numpy as np
from mathutils import Vector,Matrix,Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cycles_coherent_mirror_acceptance import configure_scene,set_diffuse_material
from cycles_coherent_polarizer_reference import projector,unpolarized_modes,power


def glass(name,ior):
    material=bpy.data.materials.new(name);material.use_nodes=True;nodes=material.node_tree.nodes;nodes.clear();node=nodes.new('ShaderNodeBsdfGlass');node.distribution='GGX';node.inputs['IOR'].default_value=ior;node.inputs['Roughness'].default_value=0;node.inputs['Color'].default_value=(1,1,1,1);out=nodes.new('ShaderNodeOutputMaterial');material.node_tree.links.new(node.outputs[0],out.inputs['Surface']);return material,node


def pane(name,center,u,v,half,material,tag):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata([(-half,-half,0),(half,-half,0),(half,half,0),(-half,half,0)],[],[(0,1,2),(0,2,3)]);obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj);obj.location=center
    u=np.asarray(u);v=np.asarray(v);normal=np.cross(u,v);obj.rotation_euler=Matrix(np.stack((u,v,normal),axis=1).tolist()).to_euler();mesh.materials.append(material);obj.cycles.coherent_interface=tag;return obj


def set_filter(node,enabled,angle,legacy):
    if 'Polarizer' in node.inputs and 'Polarizer Angle' in node.inputs:
        node.inputs['Polarizer'].default_value=enabled;node.inputs['Polarizer Angle'].default_value=angle;return False
    if not legacy:raise RuntimeError('This Blender build lacks Glass Polarizer/Polarizer Angle sockets. Build the feature or use --stage-legacy; pending scenes are not acceptance-ready.')
    node['pending_polarizer_enabled']=enabled;node['pending_polarizer_angle_rad']=angle;return True


def world_axis(obj,angle):
    matrix=np.array([list(row) for row in obj.matrix_world]);return matrix[:3,:3]@np.array([math.cos(angle),math.sin(angle),0.])


def main():
    args=sys.argv[sys.argv.index('--')+1:];output=Path(args[0]).resolve();output.mkdir(parents=True,exist_ok=False);legacy='--stage-legacy' in args;master={'pending_shader_api':False,'cases':{}}
    for experiment in ('malus','brewster_glare'):
        folder=output/experiment;folder.mkdir();scene,*_=configure_scene(True)
        for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
        scene.cycles.device='CPU';scene.cycles.use_bidirectional_path_tracing=False;scene.cycles.use_coherent_specular_connections=True;scene.cycles.coherent_polarization_mode='VECTOR';scene.cycles.coherent_max_interface_events=4;scene.cycles.max_bounces=5;scene.cycles.bdpt_max_bounces=5;scene.cycles.glossy_bounces=2;scene.cycles.transmission_bounces=4;scene.cycles.diffuse_bounces=0;scene.cycles.samples=256;scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False;scene.render.resolution_x=scene.render.resolution_y=32
        filters=[];nodes=[]
        if experiment=='malus':
            source=np.array([0.,0.,0.]);receiver=np.array([.03,0,0]);normal=np.array([-1.,0,0]);u=np.array([0.,1.,0]);v=np.array([0.,0.,1.])
            for x in (.01,.02):
                material,node=glass('Index matched polarizing film '+str(x),1.);filters.append(pane('Physical polarizing film '+str(x),(x,0,0),u,v,.008,material,'GLASS'));nodes.append(node)
            variants=[('clear',(False,False),(0.,0.)),('single',(True,False),(0.,0.)),('parallel',(True,True),(0.,0.)),('angle_30',(True,True),(0.,math.pi/6)),('angle_45',(True,True),(0.,math.pi/4)),('crossed',(True,True),(0.,math.pi/2)),('object_rotated_90',(True,True),(0.,0.))]
        else:
            source=np.array([-.015,0,.01]);receiver=np.array([.015,0,.01]);direction=receiver/np.linalg.norm(receiver);normal=-direction;u=np.array([0.,1.,0]);v=np.cross(direction,u)
            material,reflector_node=glass('Physical n1.5 Glass glare reflector',1.5);pane('Glass reflection surface',(0,0,0),(1,0,0),(0,1,0),.01,material,'GLASS');set_filter(reflector_node,False,0,legacy)
            material,node=glass('Index matched photographic analyzer',1.);filters=[pane('Photographic polarizer on reflected ray',receiver*.5,u,v,.007,material,'GLASS')];nodes=[node]
            # The mask blocks direct source-to-detector rays; reflected rays cross x0 at z0.
            maskmat=set_diffuse_material('Opaque direct glare mask',(0,0,0));pane('Physical direct-ray mask',(0,0,.01),(0,1,0),(0,0,1),.002,maskmat,'OFF')
            variants=[('unfiltered',(False,),(0.,)),('pass_s',(True,),(0.,)),('angle_45',(True,),(math.pi/4,)),('pass_p_glare_removed',(True,),(math.pi/2,)),('object_rotated_90',(True,),(0.,))]
        detector=pane('Unit first encounter detector',receiver,u,np.cross(normal,u),.0008,set_diffuse_material('Unit detector',(1,1,1)),'DETECTOR')
        bpy.ops.object.light_add(type='POINT',location=source);light=bpy.context.object;light.data.energy=.001;light.data.shadow_soft_size=0;light.data.cycles.coherence_group=1;light.data.cycles.coherence_wavelength_nm=550;light.data.cycles.coherence_length_m=1e-4;light.data.cycles.coherence_phase=0
        bpy.ops.object.camera_add(location=receiver+normal*.003);camera=bpy.context.object;camera.data.type='ORTHO';camera.data.ortho_scale=.0002;camera.data.clip_start=.001;camera.data.clip_end=10;camera.rotation_euler=(Vector(receiver)-camera.location).to_track_quat('-Z','Y').to_euler();scene.camera=camera;bpy.context.view_layer.update()
        source=np.array(tuple(light.location),dtype=float);source_power=float(light.data.energy);matrix=np.array([list(row) for row in camera.matrix_world]);right,up,forward=matrix[:3,0],matrix[:3,1],-matrix[:3,2];width=float(camera.data.ortho_scale);points=np.array([tuple(detector.matrix_world@vertex.co) for vertex in detector.data.vertices]);dn=np.cross(points[1]-points[0],points[2]-points[0]);dn/=np.linalg.norm(dn);n=32
        def pixel(row,col,ox,oy):
            origin=matrix[:3,3]+((col+ox)/n-.5)*width*right+((row+oy)/n-.5)*width*up
            return origin+np.dot(points[0]-origin,dn)/np.dot(forward,dn)*forward
        paths={};arrays={};pending=False;start=time.perf_counter();original_rotations=[obj.rotation_euler.to_quaternion() for obj in filters]
        for name,enabled,angles in variants:
            for obj,rotation in zip(filters,original_rotations):obj.rotation_euler=rotation.to_euler()
            if name=='object_rotated_90':filters[-1].rotation_euler=(original_rotations[-1]@Quaternion((0,0,1),math.pi/2)).to_euler()
            for node,on,angle in zip(nodes,enabled,angles):pending=set_filter(node,on,angle,legacy) or pending
            actual_angles=[float(node.inputs['Polarizer Angle'].default_value) if 'Polarizer Angle' in node.inputs else float(np.float32(angle)) for node,angle in zip(nodes,angles)]
            bpy.context.view_layer.update();axes=[world_axis(obj,angle) for obj,angle in zip(filters,actual_angles)];path=folder/(name+'.blend');bpy.ops.wm.save_as_mainfile(filepath=str(path));paths[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'enabled':enabled,'angles_rad':actual_angles,'world_pass_axes':[axis.tolist() for axis in axes]}
            array=np.zeros((n,n))
            for row in range(n):
                for col in range(n):
                    values=[]
                    for ox in (.25,.75):
                        for oy in (.25,.75):
                            r=pixel(row,col,ox,oy)
                            if experiment=='malus':
                                delta=r-source;length=np.linalg.norm(delta);d=delta/length;modes=unpolarized_modes(d)
                                for on,axis in zip(enabled,axes):
                                    if on:modes=modes@projector(d,axis)
                                spread=-np.dot(d,dn)/length**2
                            else:
                                virtual=source*np.array([1.,1.,-1.]);delta=r-virtual;length=np.linalg.norm(delta);point=virtual+(-virtual[2]/delta[2])*delta;incoming=(point-source)/np.linalg.norm(point-source);d=(r-point)/np.linalg.norm(r-point);sn=np.array([0.,0.,1.]);s=np.cross(incoming,sn);s/=np.linalg.norm(s);pi=np.cross(s,incoming);po=np.cross(s,d);ci=abs(incoming[2]);ct=math.sqrt(1-(1-ci*ci)/1.5**2);rs=(ci-1.5*ct)/(ci+1.5*ct);rp=(1.5*ci-ct)/(1.5*ci+ct);modes=unpolarized_modes(incoming);modes=(modes@s)[:,None]*(rs*s)+(modes@pi)[:,None]*(rp*po)
                                for on,axis in zip(enabled,axes):
                                    if on:modes=modes@projector(d,axis)
                                spread=-np.dot(d,dn)/length**2
                            values.append(source_power*spread*power(modes)/(4*math.pi**2))
                    array[row,col]=sum(values)/4
            arrays[name]=array
        rows,cols=np.indices((n,n));roi=(rows>=3)&(rows<n-3)&(cols>=3)&(cols<n-3);np.savez_compressed(folder/'independent_polarizer_reference.npz',roi_mask=roi,**arrays)
        for name,array in arrays.items():
            # Compatible with the existing unchanged raw-EXR render worker's phase_0 case.
            np.savez_compressed(folder/(name+'_reference.npz'),roi_mask=roi,phase_0_radiance=array,phase_pi_radiance=array,incoherent_radiance=array)
            paths[name]['reference_npz']=str(folder/(name+'_reference.npz'))
        ref={'status':'Independent Jones/Malus/Fresnel analytic reference; not rendered pixels','source_power_W':source_power,'source_position_m':source.tolist(),'mean_radiances':{name:float(array[roi].mean()) for name,array in arrays.items()},'seconds':time.perf_counter()-start,'geometry':'Matched IOR1 ideal transmission films; object-localXY pass axis, world transformed then projected transverse. Brewster n1.5 reflector plus physical direct mask and returned-ray analyzer.','pixel_filter':'2x2BOX','pending_shader_api':pending}
        (folder/'independent_polarizer_reference.json').write_text(json.dumps(ref,indent=2)+'\n');case={'specular_connections':True,'scope':experiment,'variants':paths,'reference':ref,'samples':256,'adaptive':False,'denoise':False,'resolution':[32,32],'pending_shader_api':pending,'rendered':False};(folder/'manifest.json').write_text(json.dumps(case,indent=2)+'\n');master['cases'][experiment]=case;master['pending_shader_api']=master['pending_shader_api'] or pending;print(json.dumps({'case':experiment,'reference':ref}))
    (output/'manifest.json').write_text(json.dumps(master,indent=2)+'\n')
    (output/'pre_gpu_gates.json').write_text(json.dumps({'declared_without_gpu_pixels':True,'malus':{'absolute_mean_error':1e-4,'absolute_rmse':2e-4,'crossed_absolute_mean_error':1e-8,'reason':'Unit unpolarized ideal film gives50% once,50% for parallel pair,25% for45deg pair,0 crossed; declared1mW source yields clear.0281 and single.0141, so these gates reject omitted/wrong scalar filters.'},'brewster_glare':{'absolute_mean_error':1e-5,'absolute_rmse':2e-5,'pass_p_absolute_mean_error':5e-8,'reason':'Finite off-axis crop predicts7e-9 residual glare at n1.5 Brewster, versus unfiltered.0014412. Absolute gates reject a filter that does not rotate away glare.'},'limits':'Pending legacy scenes cannot be GPU accepted until the actual shader sockets are set in a feature-enabled binary. No rendered-image normalization or fitted brightness.'},indent=2)+'\n')

if __name__=='__main__':main()
