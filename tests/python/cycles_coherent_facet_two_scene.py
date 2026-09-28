#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Author finite two-mirror two-reflection fixtures and exact-float independent refs.

No render calls. --stage-legacy saves geometry pending the new transport enum.
"""
import bpy,sys,json,math,hashlib,time
from pathlib import Path
import numpy as np
from mathutils import Matrix,Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cycles_coherent_facet_two_reference import geometry,inventory,pair_intensity
from cycles_coherent_mirror_acceptance import configure_scene,set_diffuse_material
from cycles_coherent_polarizer_scene import pane


def mesh_object(name,vertices,faces,material):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(vertices,[],faces);mesh.materials.append(material);obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj);obj.cycles.coherent_interface='MIRROR';return obj


def main():
    args=sys.argv[sys.argv.index('--')+1:];output=Path(args[0]).resolve();output.mkdir(parents=True,exist_ok=False);legacy='--stage-legacy' in args;master={'scope':'Two finite nonparallel mirror objects; direct, all visible 1R and ordered 2R; first Lambertian receiver','cases':{}}
    for layout in ['corner_xy']:
        folder=output/layout;folder.mkdir();scene,det,mirror,*_=configure_scene(True);material=mirror.data.materials[0]
        for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
        pending=legacy
        if legacy:scene['pending_streamed_two_reflection_activation']=True
        try:scene.cycles.coherent_transport_mode='FACET_SINGLE_REFLECTION'
        except (AttributeError,TypeError):
            if not legacy:raise
            scene['pending_coherent_transport_mode']='FACET_SINGLE_REFLECTION';pending=True
        scene.cycles.use_coherent_specular_connections=True;scene.cycles.coherent_polarization_mode='VECTOR';scene.cycles.coherent_max_interface_events=2;scene.cycles.samples=1024;scene.cycles.glossy_bounces=2;scene.cycles.max_bounces=3;scene.cycles.bdpt_max_bounces=3;scene.cycles.diffuse_bounces=0;scene.cycles.device='CPU';scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False;scene.render.resolution_x=scene.render.resolution_y=16
        triangles,source,receiver=geometry();objects=[]
        objects=[mesh_object('Finite corner mirror '+str(i),triangles[2*i:2*i+2].reshape(-1,3).tolist(),[(0,1,2),(3,4,5)],material) for i in range(2)]
        detector=pane('Actual unit Lambertian detector',receiver,(1,0,0),(0,-1,0),.001,set_diffuse_material('Unit detector',(1,1,1)),'DETECTOR')
        sources=[]
        for i,y in enumerate([-.00001,.00001]):
            bpy.ops.object.light_add(type='POINT',location=source+np.array([0.,y,0.]));light=bpy.context.object;light.data.energy=.001;light.data.shadow_soft_size=0;light.data.cycles.coherence_group=1;light.data.cycles.coherence_wavelength_nm=550;light.data.cycles.coherence_length_m=1.;light.data.cycles.coherence_phase=0;sources.append(light)
        bpy.ops.object.camera_add(location=receiver+np.array([0,0,-.004]));camera=bpy.context.object;camera.data.type='ORTHO';camera.data.ortho_scale=.000002;camera.data.clip_start=.001;camera.rotation_euler=(Vector(receiver)-camera.location).to_track_quat('-Z','Y').to_euler();scene.camera=camera;bpy.context.view_layer.update()
        exact=[]
        for obj in objects:
            for poly in obj.data.polygons:exact.append([tuple(obj.matrix_world@obj.data.vertices[i].co) for i in poly.vertices])
        exact=np.array(exact);assert exact.shape==(4,3,3);source_positions=[np.array(light.location) for light in sources];powers=[float(light.data.energy) for light in sources];receiver=np.array(detector.location);normal=np.array([0.,0.,-1.]);matrix=np.array([list(row) for row in camera.matrix_world]);right=matrix[:3,0];up=matrix[:3,1];width=float(camera.data.ortho_scale);start=time.perf_counter();variants={};counts=[]
        (folder/'saved_camera_mapping.json').write_text(json.dumps({'matrix_world':matrix.tolist(),'ortho_scale':width},indent=2)+'\n')
        pixel_paths={}
        for row in range(16):
            for col in range(16):
                for ox in [.25,.75]:
                    for oy in [.25,.75]:
                        point=receiver+((col+ox)/16-.5)*width*right+((row+oy)/16-.5)*width*up
                        inventories=[inventory(s,point,exact) for s in source_positions];counts.extend(len(ps) for ps in inventories);pixel_paths[row,col,ox,oy]=inventories
        arrays={};omission={}
        for name,phase,groups in [('phase_0',0,[1,1]),('phase_pi',math.pi,[1,1]),('incoherent_connector_control',0,[1,2])]:
            for light,g in zip(sources,groups):light.data.cycles.coherence_group=g
            sources[1].data.cycles.coherence_phase=phase;phases=[float(light.data.cycles.coherence_phase) for light in sources];coherence=float(sources[0].data.cycles.coherence_length_m);wave=float(sources[0].data.cycles.coherence_wavelength_nm)*1e-9
            array=np.zeros((16,16));truncated=np.zeros((16,16))
            for row in range(16):
                for col in range(16):
                    for ox in [.25,.75]:
                        for oy in [.25,.75]:
                            inventories=pixel_paths[row,col,ox,oy];array[row,col]+=pair_intensity(inventories,powers,phases,groups,wave,coherence)[0]/4
                            capped=[[p for p in ps if len(p['facet'])<2] for ps in inventories];truncated[row,col]+=pair_intensity(capped,powers,phases,groups,wave,coherence)[0]/4
            arrays[name]=array;omission[name]=truncated;path=folder/(name+'.blend');bpy.ops.wm.save_as_mainfile(filepath=str(path));variants[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_phases_rad':phases,'groups':groups}
        rows,cols=np.indices((16,16));roi=(rows>=2)&(rows<14)&(cols>=2)&(cols<14);np.savez_compressed(folder/'virtual_source_reference.npz',roi_mask=roi,phase_0_radiance=arrays['phase_0'],phase_pi_radiance=arrays['phase_pi'],incoherent_radiance=arrays['incoherent_connector_control']);np.savez_compressed(folder/'physical_geometry.npz',triangles=exact,source_positions=source_positions,source_powers=powers,receiver=receiver);np.savez_compressed(folder/'two_reflection_omission_control.npz',roi_mask=roi,**omission)
        case={'scope':master['scope'],'specular_connections':True,'pending_transport_api':pending,'facet_count':4,'mirror_object_count':len(objects),'linked_mesh_datablocks':len({obj.data.as_pointer() for obj in objects}),'variants':variants,'minimum_visible_path_count':min(counts),'maximum_visible_path_count':max(counts),'source_power_W':powers,'reference':'Double image-source stationary points + physical OPL + every finite triangle segment occlusion; signed Householder world-dipole transport + explicit double pair Gaussian gamma','mean_radiance':{k:float(v[roi].mean()) for k,v in arrays.items()},'author_seconds':time.perf_counter()-start,'pixel_quadrature':'2x2BOX; geometric path inventory recomputed at every quadrature point','gates_declared_before_render':{'absolute_mean_error':.0002,'absolute_rmse':.0008},'rendered':False}
        case['glossy_bounces']=2;case['maximum_interface_events']=2;case['coherence_length_m']=coherence;case['diffuse_bounces']=0;case['receiver_scope']='First Lambertian receiver only; diffuse0 avoids native detector→mirror→detector returns outside this reference';case['two_reflection_omission_rmse']={name:float(np.sqrt(np.mean((arrays[name][roi]-omission[name][roi])**2))) for name in arrays};case['center_OPL_range_m']=[min(p['opl'] for p in pixel_paths[8,8,.25,.25][0]),max(p['opl'] for p in pixel_paths[8,8,.25,.25][0])]
        (folder/'manifest.json').write_text(json.dumps(case,indent=2)+'\n');master['cases'][layout]=case;print(layout,min(counts),max(counts),case['mean_radiance'],case['two_reflection_omission_rmse'])
    (output/'manifest.json').write_text(json.dumps(master,indent=2)+'\n')


if __name__=='__main__':main()
