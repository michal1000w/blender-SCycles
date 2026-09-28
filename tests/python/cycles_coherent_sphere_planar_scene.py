#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Author mixed native sphere/finite mirror fixtures and independent pixel fields.

Invoke Blender CPU background --python this_file -- output_dir [--preview].
No mixed-connector acceptance renders occur here; v38b can author the fixtures.
"""
import bpy,sys,json,math,time,hashlib
from pathlib import Path
import numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cycles_coherent_mirror_acceptance import configure_scene,set_diffuse_material
from cycles_coherent_sphere_acceptance import validate_native_points
from cycles_coherent_sphere_planar_fields import setup,paths,radiance


def rectangle(name,x,y,z,hy,hz,material,tag):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata([(x,y-hy,z-hz),(x,y-hy,z+hz),(x,y+hy,z+hz),(x,y+hy,z-hz)],[],[(0,1,2),(0,2,3)])
    obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj);mesh.materials.append(material)
    obj.cycles.coherent_interface=tag
    return obj


def specular_material(kind):
    material=bpy.data.materials.new('Unit lossless '+kind);material.use_nodes=True;nodes=material.node_tree.nodes;nodes.clear()
    node=nodes.new('ShaderNodeBsdfGlass' if kind=='TT' else 'ShaderNodeBsdfGlossy');node.distribution='GGX';node.inputs['Roughness'].default_value=0;node.inputs['Color'].default_value=(1,1,1,1)
    if kind=='TT':node.inputs['IOR'].default_value=1.5
    output=nodes.new('ShaderNodeOutputMaterial');material.node_tree.links.new(node.outputs[0],output.inputs['Surface']);return material


def sphere(kind,radius):
    material=specular_material(kind);mesh=bpy.data.meshes.new('Native center');mesh.from_pydata([(0,0,0)],[],[]);mesh.materials.append(material)
    obj=bpy.data.objects.new('Native analytical sphere '+kind,mesh);bpy.context.scene.collection.objects.link(obj);obj.cycles.coherent_interface='GLASS' if kind=='TT' else 'MIRROR'
    mod=obj.modifiers.new('Real point sphere','NODES');group=bpy.data.node_groups.new('Native sphere geometry','GeometryNodeTree');mod.node_group=group
    for direction in ('INPUT','OUTPUT'):group.interface.new_socket(name='Geometry',in_out=direction,socket_type='NodeSocketGeometry')
    inp=group.nodes.new('NodeGroupInput');out=group.nodes.new('NodeGroupOutput');points=group.nodes.new('GeometryNodeMeshToPoints');points.mode='VERTICES';points.inputs['Radius'].default_value=radius
    mat=group.nodes.new('GeometryNodeSetMaterial');mat.inputs['Material'].default_value=material
    group.links.new(inp.outputs['Geometry'],points.inputs['Mesh']);group.links.new(points.outputs['Points'],mat.inputs['Geometry']);group.links.new(mat.outputs['Geometry'],out.inputs['Geometry'])
    return obj


def main():
    args=sys.argv[sys.argv.index('--')+1:];output=Path(args[0]).resolve();output.mkdir(parents=True,exist_ok=False);preview='--preview' in args
    manifest={'status':'Authored actual Blender scenes and independent double fields; no mixed acceptance render yet','binary':bpy.app.binary_path,'cases':{}}
    for kind in ('R','TT'):
        dest=output/kind;dest.mkdir();cfg=setup(kind);cfg['mirrors']=cfg['mirrors'][:1]
        scene,*_=configure_scene(True)
        for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
        scene.cycles.device='CPU';scene.cycles.use_bidirectional_path_tracing=False;scene.cycles.use_coherent_specular_connections=True
        scene.cycles.coherent_polarization_mode='VECTOR';scene.cycles.coherent_max_interface_events=4
        scene.cycles.max_bounces=5;scene.cycles.bdpt_max_bounces=5;scene.cycles.glossy_bounces=4;scene.cycles.transmission_bounces=2;scene.cycles.diffuse_bounces=0
        scene.cycles.samples=256;scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False;scene.render.resolution_x=scene.render.resolution_y=32
        native=sphere(kind,cfg['radius']);geometry=validate_native_points(native, float(np.float32(cfg["radius"])))
        for i,m in enumerate(cfg['mirrors']):rectangle('Finite mirror '+str(i),m['x'],m['y'],m['z'],m['half'],m['half'],specular_material('R'),'MIRROR')
        center=np.array(cfg['receiver']);detector=rectangle('Unit terminal detector',*center,cfg['crop']*4,cfg['crop']*4,set_diffuse_material('Unit detector',(1,1,1)),'DETECTOR')
        # Rectangle order currently faces negative X; reverse to face incoming paths from +X.
        for polygon in detector.data.polygons:polygon.flip()
        lights=[]
        for dy in (-1e-5,1e-5):
            position=np.array(cfg['source']);position[1]+=dy;bpy.ops.object.light_add(type='POINT',location=position)
            light=bpy.context.object;light.name='Coherent physical source '+str(dy);light.data.energy=.001;light.data.shadow_soft_size=0
            light.data.cycles.coherence_group=1;light.data.cycles.coherence_length_m=1e-4;light.data.cycles.coherence_wavelength_nm=550;light.data.cycles.coherence_phase=0;lights.append(light)
        bpy.ops.object.camera_add(location=center+np.array([.002,0,0]));camera=bpy.context.object;camera.rotation_euler=(Vector(center)-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=cfg['crop'];camera.data.clip_start=.001;camera.data.clip_end=10;scene.camera=camera
        bpy.context.view_layer.update()
        # All model coordinates passed through Blender's float storage before the independent reference.
        positions=[np.array(tuple(light.location),dtype=float) for light in lights]
        cfg['radius']=float(np.float32(cfg['radius']));cfg['mirrors']=[{k:float(np.float32(v)) for k,v in m.items()} for m in cfg['mirrors']]
        detector_x=float(detector.data.vertices[0].co.x);matrix=np.array([list(row) for row in camera.matrix_world]);right=matrix[:3,0];up=matrix[:3,1];forward=-matrix[:3,2];width=float(camera.data.ortho_scale);n=32
        def receiver(row,col,ox,oy):
            origin=matrix[:3,3]+((col+ox)/n-.5)*width*right+((row+oy)/n-.5)*width*up
            return origin+(detector_x-origin[0])/forward[0]*forward
        variants={}
        for name,phase,group in [('phase_0',0,1),('phase_pi',math.pi,1),('incoherent_connector_control',0,2)]:
            lights[1].data.cycles.coherence_phase=phase;lights[1].data.cycles.coherence_group=group;path=dest/(name+'.blend');bpy.ops.wm.save_as_mainfile(filepath=str(path));variants[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        arrays=np.zeros((3,n,n));counts={};start=time.perf_counter()
        def value(r):
            ps=[paths(s,r,cfg) for s in positions]
            for source in ps:
                for p in source:counts[p['label']]=counts.get(p['label'],0)+1
            return np.array([radiance(ps,phase,groups,powers=[float(l.data.energy) for l in lights]) for phase,groups in [([0,0],[1,1]),([0,math.pi],[1,1]),([0,0],[1,2])]])
        for row in range(n):
            for col in range(n):arrays[:,row,col]=sum(value(receiver(row,col,ox,oy)) for ox in (.25,.75) for oy in (.25,.75))/4
        quadrature=0
        for row,col in [(5,5),(15,15),(26,26)]:
            fine=sum(value(receiver(row,col,ox,oy)) for ox in (.125,.375,.625,.875) for oy in (.125,.375,.625,.875))/16
            quadrature=max(quadrature,float(np.max(abs(fine-arrays[:,row,col]))/max(np.max(fine),1e-30)))
        rows,cols=np.indices((n,n));roi=(rows>=3)&(rows<n-3)&(cols>=3)&(cols<n-3);coords=np.array([[receiver(r,c,.5,.5) for c in range(n)] for r in range(n)])
        assert np.isfinite(arrays).all() and arrays.min()>=0
        np.savez_compressed(dest/'virtual_source_reference.npz',x_m=coords[:,:,1],y_m=coords[:,:,2],roi_mask=roi,phase_0_radiance=arrays[0],phase_pi_radiance=arrays[1],incoherent_radiance=arrays[2])
        ref={'cfg':cfg,'source_positions_m':[s.tolist() for s in positions],'source_power_each_W':[float(l.data.energy) for l in lights],'labels_seen':counts,'mean_radiances':[float(a[roi].mean()) for a in arrays],'max_checked_2x2_vs_4x4_relative':quadrature,'seconds':time.perf_counter()-start,'model':'Independent stationary sphere branches, finite mirror intersections, analytic segment visibility, Fresnel flux/Jones, world-axis dipole ensemble, OPL and Morse, Gaussian coherence. Includes clear direct/pure-planar alternatives; no PT fit.'}
        (dest/'virtual_source_reference.json').write_text(json.dumps(ref,indent=2)+'\n')
        case={'specular_connections':True,'scope':'Mixed native analytical sphere '+kind+' and finite planar mirror; independently summed supported visible families, two1mW sources, four-event budget','scene_variants':variants,'variants':variants,'reference':ref,'native_geometry':geometry,'rendered':False,'samples':256,'adaptive':False,'denoise':False,'resolution':[32,32],'camera_clip_start':.001}
        (dest/'manifest.json').write_text(json.dumps(case,indent=2)+'\n');manifest['cases'][kind]=case
        # Wide editable ordinary camera makes the physical arrangement inspectable.
        scene.cycles.use_coherent_specular_connections=False;scene.cycles.samples=32;scene.cycles.diffuse_bounces=2;scene.cycles.max_bounces=6;scene.render.resolution_x=200;scene.render.resolution_y=150
        scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5
        camera.data.type='PERSP';camera.data.lens=45;camera.location=(.045,-.055,.035) if kind=='TT' else (-.035,-.035,.025)
        camera.rotation_euler=(Vector((0,0,0) if kind=='TT' else (-.005,0,0))-camera.location).to_track_quat('-Z','Y').to_euler();scene.view_settings.exposure=0;scene.view_settings.view_transform='AgX'
        for light in lights:light.data.cycles.coherence_group=0;light.data.energy=.0001
        bpy.ops.mesh.primitive_plane_add(size=.07,location=(0,0,-cfg['radius']))
        ground=bpy.context.object;ground.name='Ordinary preview ground';ground.data.materials.append(set_diffuse_material('Gray ordinary ground',(.2,.24,.3)))
        bpy.ops.object.light_add(type='AREA',location=(.01,-.015,.045));area=bpy.context.object;area.name='Ordinary preview area illumination';area.data.energy=.02;area.data.shape='DISK';area.data.size=.035
        area.rotation_euler=(Vector((0,0,0))-area.location).to_track_quat('-Z','Y').to_euler()
        case['ordinary_preview_scope']='Separate noncoherent area/background/ground illustration of physical arrangement; not a detector acceptance reference.'
        preview_path=dest/'ordinary_physical_preview.blend';bpy.ops.wm.save_as_mainfile(filepath=str(preview_path));case['ordinary_preview_scene']=str(preview_path)
        if preview:
            scene.render.image_settings.file_format='PNG';scene.render.filepath=str(dest/'ordinary_physical_preview.png');bpy.ops.render.render(write_still=True);case['ordinary_preview_png']=scene.render.filepath
        print(json.dumps({'kind':kind,'reference':ref}))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
