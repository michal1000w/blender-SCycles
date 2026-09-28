#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Native Glass sphere scenes with all visible histories through TRT/TRRT.

Independent direct/exterior R/TT/TRT/TRRT vector fields, fixed declared 1mW.
No renderer brightness oracle or fitted normalization; Blender authoring CPU.
"""
import bpy,sys,math,json,time,hashlib
from pathlib import Path
import numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cycles_coherent_mirror_acceptance import configure_scene,set_diffuse_material
from cycles_coherent_sphere_planar_scene import sphere
from cycles_coherent_sphere_acceptance import validate_native_points
from cycles_coherent_sphere_reference import sphere_path,spreading as r_spreading
from cycles_coherent_sphere_transmission_reference import basis
from cycles_coherent_sphere_internal_reference import inventory,field


from cycles_coherent_sphere_internal_fields import all_fields,radiance


def main():
    output=Path(sys.argv[sys.argv.index('--')+1]).resolve();output.mkdir(parents=True,exist_ok=False);master={}
    for max_internal in (1,2):
        name='TRT' if max_internal==1 else 'TRRT';folder=output/name;folder.mkdir();scene,*_=configure_scene(True)
        for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
        scene.cycles.device='CPU';scene.cycles.use_bidirectional_path_tracing=False;scene.cycles.use_coherent_specular_connections=True;scene.cycles.coherent_polarization_mode='VECTOR';scene.cycles.coherent_max_interface_events=max_internal+2
        scene.cycles.max_bounces=max_internal+3;scene.cycles.bdpt_max_bounces=max_internal+3;scene.cycles.glossy_bounces=max_internal+1;scene.cycles.transmission_bounces=2;scene.cycles.diffuse_bounces=0;scene.cycles.samples=256;scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False;scene.render.resolution_x=scene.render.resolution_y=32
        radius=float(np.float32(.01));native=sphere('TT',radius);geometry=validate_native_points(native,radius)
        receiver=np.array([.02*math.cos(1.1),.02*math.sin(1.1),.0003]);normal=-receiver/np.linalg.norm(receiver);frame=basis(normal);half=.0003
        vertices=[receiver+half*(u*frame[:,0]+v*frame[:,1]) for u,v in ((-1,-1),(1,-1),(1,1),(-1,1))];mesh=bpy.data.meshes.new('Actual detector plane');mesh.from_pydata(vertices,[],[(0,1,2),(0,2,3)]);detector=bpy.data.objects.new('Unit first encounter detector',mesh);scene.collection.objects.link(detector);mesh.materials.append(set_diffuse_material('Unit diffuse detector',(1,1,1)));detector.cycles.coherent_interface='DETECTOR'
        stored=np.array([tuple(v.co) for v in mesh.vertices]);normal=np.cross(stored[1]-stored[0],stored[2]-stored[0]);normal/=np.linalg.norm(normal)
        lights=[]
        for y in (-1e-5,1e-5):
            bpy.ops.object.light_add(type='POINT',location=(.02,y,.0001));light=bpy.context.object;light.data.energy=.001;light.data.shadow_soft_size=0;light.data.cycles.coherence_group=1;light.data.cycles.coherence_phase=0;light.data.cycles.coherence_wavelength_nm=550;light.data.cycles.coherence_length_m=1e-4;lights.append(light)
        bpy.ops.object.camera_add(location=receiver+normal*.003);camera=bpy.context.object;camera.data.type='ORTHO';camera.data.ortho_scale=.00007;camera.data.clip_start=.001;camera.data.clip_end=10;camera.rotation_euler=(Vector(receiver)-camera.location).to_track_quat('-Z','Y').to_euler();scene.camera=camera;bpy.context.view_layer.update()
        sources=[np.array(tuple(l.location),dtype=float) for l in lights];powers=[float(l.data.energy) for l in lights];matrix=np.array([list(row) for row in camera.matrix_world]);right,up,forward=matrix[:3,0],matrix[:3,1],-matrix[:3,2];width=float(camera.data.ortho_scale);n=32
        def pixel(row,col,ox,oy):
            origin=matrix[:3,3]+((col+ox)/n-.5)*width*right+((row+oy)/n-.5)*width*up
            return origin+np.dot(stored[0]-origin,normal)/np.dot(forward,normal)*forward
        variants={}
        for variant,phase,group in [('phase_0',0,1),('phase_pi',math.pi,1),('incoherent_connector_control',0,2)]:
            lights[1].data.cycles.coherence_phase=phase;lights[1].data.cycles.coherence_group=group;path=folder/(variant+'.blend');bpy.ops.wm.save_as_mainfile(filepath=str(path));variants[variant]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        counts={};arrays=np.zeros((3,n,n));start=time.perf_counter()
        def value(r):
            fields=[all_fields(s,r,normal,radius,1.5,max_internal,power) for s,power in zip(sources,powers)]
            for source in fields:
                for f,opl,label in source:counts[label]=counts.get(label,0)+1
            return np.array([radiance(fields,p,g) for p,g in [([0,0],[1,1]),([0,math.pi],[1,1]),([0,0],[1,2])]])
        for row in range(n):
            for col in range(n):arrays[:,row,col]=sum(value(pixel(row,col,x,y)) for x in (.25,.75) for y in (.25,.75))/4
        quadrature=0
        for row,col in ((5,5),(15,15),(26,26)):
            fine=sum(value(pixel(row,col,x,y)) for x in (.125,.375,.625,.875) for y in (.125,.375,.625,.875))/16
            quadrature=max(quadrature,float(np.max(abs(fine-arrays[:,row,col]))))
        assert np.isfinite(arrays).all() and arrays.min()>=0
        rows,cols=np.indices((n,n));roi=(rows>=3)&(rows<n-3)&(cols>=3)&(cols<n-3);coords=np.array([[pixel(r,c,.5,.5) for c in range(n)] for r in range(n)])
        np.savez_compressed(folder/'virtual_source_reference.npz',x_m=coords[:,:,0],y_m=coords[:,:,1],roi_mask=roi,phase_0_radiance=arrays[0],phase_pi_radiance=arrays[1],incoherent_radiance=arrays[2])
        ref={'source_positions_m':[s.tolist() for s in sources],'source_power_each_W':powers,'radius_m':radius,'ior':1.5,'detector_normal':normal.tolist(),'labels_seen':counts,'mean_radiances':[float(a[roi].mean()) for a in arrays],'max_checked_2x2_vs_4x4_absolute':quadrature,'seconds':time.perf_counter()-start,'path_model':'All geometrically visible direct, Fresnel exterior R and all-winding TT/TRT/TRRT through declared event cap, dielectric flux/Jones and Morse, no PT fit; 1mW declared sources.'}
        (folder/'virtual_source_reference.json').write_text(json.dumps(ref,indent=2)+'\n');case={'specular_connections':True,'scope':'Native Glass sphere full visible path inventory through '+name,'scene_variants':variants,'variants':variants,'reference':ref,'native_geometry':geometry,'rendered':False,'samples':256,'adaptive':False,'denoise':False,'resolution':[32,32],'camera_clip_start':.001};(folder/'manifest.json').write_text(json.dumps(case,indent=2)+'\n');master[name]=case;print(json.dumps({'case':name,'reference':ref}))
    (output/'manifest.json').write_text(json.dumps(master,indent=2)+'\n')

if __name__=='__main__':main()
