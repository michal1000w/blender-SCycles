#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Author an isolated three-branch crop and an independent pixel-integrated oracle."""
import json,hashlib,math,sys,time
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cycles_coherent_sphere_transmission_reference import inventory
from cycles_coherent_sphere_transmission_fields import radiance


def main():
    args=sys.argv[sys.argv.index('--')+1:]
    base,output=map(lambda x:Path(x).resolve(),args)
    output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(base))
    scene=bpy.context.scene
    scene.cycles.use_coherent_specular_connections = True
    scene.cycles.max_bounces = 3
    scene.cycles.bdpt_max_bounces = 3
    sources=sorted([o for o in scene.objects if o.type=='LIGHT'],key=lambda o:o.location.y)
    for s in sources:s.data.energy=1
    scene.cycles.samples=256
    scene.render.resolution_x=scene.render.resolution_y=64
    camera=scene.camera;camera.location=(.3,.012,-.018)
    camera.rotation_euler=(Vector((.4,.012,-.018))-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.ortho_scale=.009
    camera.data.clip_start=.001
    camera.data.clip_end=10
    bpy.context.view_layer.update()
    variants={}
    for name,phase,group in [('phase_0',0,1),('phase_pi',math.pi,1),('incoherent_connector_control',0,2)]:
        sources[1].data.cycles.coherence_phase=phase;sources[1].data.cycles.coherence_group=group
        path=output/(name+'.blend');bpy.ops.wm.save_as_mainfile(filepath=str(path))
        variants[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    positions=[np.array(tuple(s.location),dtype=float) for s in sources]
    detector=next(o for o in scene.objects if o.cycles.coherent_interface=='DETECTOR')
    detector_x=float(detector.data.vertices[0].co.x)
    matrix=np.array([list(row) for row in camera.matrix_world],dtype=float)
    right,up,forward=matrix[:3,0],matrix[:3,1],-matrix[:3,2]
    n=64;width=float(camera.data.ortho_scale);normal=np.array([-1.,0.,0.])
    def receiver(row,col,ox,oy):
        origin=matrix[:3,3]+((col+ox)/n-.5)*width*right+((row+oy)/n-.5)*width*up
        return origin+(detector_x-origin[0])/forward[0]*forward
    arrays=np.zeros((3,n,n));counts=set();max_snell=0.;start=time.perf_counter()
    def value(r):
        nonlocal max_snell
        paths=[inventory(s,r,np.zeros(3),.25,1.5) for s in positions]
        counts.update(len(p) for p in paths)
        max_snell=max(max_snell,max(p['snell_residual'] for pp in paths for p in pp))
        return np.array([radiance(paths,normal,[1,1],phases,groups) for phases,groups in
                         [([0,0],[1,1]),([0,math.pi],[1,1]),([0,0],[1,2])]])
    for row in range(n):
        for col in range(n):
            arrays[:,row,col]=sum(value(receiver(row,col,ox,oy)) for ox in (.25,.75) for oy in (.25,.75))/4
    # Independent finer quadrature at dispersed pixels bounds the chosen pixel integration.
    quadrature_error=0.
    for row in (5,31,58):
        for col in (5,31,58):
            fine=sum(value(receiver(row,col,ox,oy)) for ox in (.125,.375,.625,.875) for oy in (.125,.375,.625,.875))/16
            quadrature_error=max(quadrature_error,float(np.max(abs(fine-arrays[:,row,col]))))
    rows,cols=np.indices((n,n));roi=(rows>=4)&(rows<n-4)&(cols>=4)&(cols<n-4)
    coords=np.array([[receiver(row,col,.5,.5) for col in range(n)] for row in range(n)])
    assert counts=={3},counts
    assert np.isfinite(arrays).all() and arrays.min()>0
    assert quadrature_error<.002,quadrature_error
    np.savez_compressed(output/'virtual_source_reference.npz',x_m=coords[:,:,1],y_m=coords[:,:,2],roi_mask=roi,
                        phase_0_radiance=arrays[0],phase_pi_radiance=arrays[1],incoherent_radiance=arrays[2])
    reference={'status':'Independent double sphere TT/Jones/Morse reference, not a render',
               'branches_per_source':sorted(counts),'source_power_each_W':1,'max_snell_residual':max_snell,
               'pixel_filter':'2x2 BOX quadrature; independent 4x4 checks at nine dispersed pixels',
               'max_checked_quadrature_difference':quadrature_error,'mean_radiances':[float(a[roi].mean()) for a in arrays],
               'seconds':time.perf_counter()-start,'path_model':'All isolated two-transmission branches, exact flux Fresnel, three world-axis dipole modes, full vector overlap, Morse phase; 1e-4 m Gaussian coherence length.'}
    (output/'virtual_source_reference.json').write_text(json.dumps(reference,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps({'variants':variants,'scene_variants':variants,'reference':reference,
        'rendered':False,'specular_connections':True,'resolution':[64,64],'samples':256,'adaptive':False,'denoise':False,
        'scope':'Three isolated curved-transmission branches; crop excludes fold/ring singularities. Wide fixture retained separately.'},indent=2)+'\n')
    print(json.dumps(reference))

if __name__=='__main__':main()
