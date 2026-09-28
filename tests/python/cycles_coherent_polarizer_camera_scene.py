#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded coherent Lambertian detector → camera Malus fixture.

Two finite films lie on an oblique camera ray and outside all source rays.
Detector reradiation is unpolarized, so clear/parallel/crossed are 1/.5/0.
Use an active coherent polarizer fixture's malus/clear.blend as the input.
"""
import bpy,sys,json,math,hashlib
from pathlib import Path
import numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cycles_coherent_polarizer_scene import glass,pane,set_filter


def main():
    args=sys.argv[sys.argv.index('--')+1:];source_blend=Path(args[0]).resolve();folder=Path(args[1]).resolve();folder.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(source_blend));scene=bpy.context.scene
    assert all('Polarizer' in n.inputs for m in bpy.data.materials if m.use_nodes for n in m.node_tree.nodes if n.type=='BSDF_GLASS')
    receiver=np.array([.03,0.,0.]);outward=np.array([-math.sqrt(.75),.5,0.]);u=np.array([0.,0.,1.]);v=np.cross(outward,u)
    camera=scene.camera;camera.location=receiver+outward*.008;camera.rotation_euler=(Vector(receiver)-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.clip_start=.0001
    nodes=[]
    for distance in [.003,.005]:
        material,node=glass('Camera-side ideal film '+str(distance),1.)
        obj=pane('Finite off-source camera film '+str(distance),receiver+outward*distance,u,v,.0006,material,'OFF')
        obj.data.clear_geometry();obj.data.from_pydata([(-.0006,-.0006,0),(.0006,-.0006,0),(0,.0012,0)],[],[(0,1,2)])
        nodes.append(node)
    # Film centers have y>=.0015, their closest edge y>.00098. Source rays
    # reach a detector only .0008 wide and never intersect either film.
    bpy.context.view_layer.update();matrix=np.array([list(r) for r in camera.matrix_world]);right=matrix[:3,0];up=matrix[:3,1];forward=-matrix[:3,2];width=float(camera.data.ortho_scale);n=32
    light=next(o for o in bpy.data.objects if o.type=='LIGHT');source=np.array(light.location);energy=float(light.data.energy);normal=np.array([-1.,0.,0.]);reference=np.zeros((n,n))
    for row in range(n):
        for col in range(n):
            for ox in [.25,.75]:
                for oy in [.25,.75]:
                    origin=matrix[:3,3]+((col+ox)/n-.5)*width*right+((row+oy)/n-.5)*width*up
                    point=origin+(receiver[0]-origin[0])/forward[0]*forward;delta=point-source;length=np.linalg.norm(delta)
                    reference[row,col]+=energy*(-np.dot(delta/length,normal))/length**2/(4*math.pi**2)/4
    rows,cols=np.indices((n,n));roi=(rows>=3)&(rows<29)&(cols>=3)&(cols<29);variants={}
    for name,enabled,angle,factor in [('clear',(False,False),0,1),('single',(True,False),0,.5),('parallel',(True,True),0,.5),('crossed',(True,True),math.pi/2,0)]:
        for index,node in enumerate(nodes):set_filter(node,enabled[index],angle if index else 0,False)
        path=folder/(name+'.blend');bpy.ops.wm.save_as_mainfile(filepath=str(path))
        np.savez_compressed(folder/(name+'_reference.npz'),roi_mask=roi,phase_0_radiance=reference*factor,phase_pi_radiance=reference*factor,incoherent_radiance=reference*factor)
        variants[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'reference_factor':factor,'reference_npz':str(folder/(name+'_reference.npz'))}
    manifest={'scope':'Coherent source → Lambertian unpolarized detector → ordinary camera-side two ideal films; no source-film intersection','specular_connections':True,'pending_shader_api':False,'variants':variants,'source_power_W':energy,'gates':{'absolute_mean_error':1e-4,'absolute_rmse':2e-4,'crossed_leakage_mean':1e-8},'declared_before_actual_render':True}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
