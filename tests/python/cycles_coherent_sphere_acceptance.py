#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Editable native MeshToPoints mirror sphere; no rendering in this generator.

One exterior reflection, two coherent point sources, flat Lambertian detector.
Independent double angular-root/curvature reference includes the three global
projected dipole modes and full vector receiver overlap. Direct rays are exactly
in the detector tangent plane (zero Lambertian throughput); no fitted texture.
"""
import hashlib,json,math,sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cycles_coherent_mirror_acceptance import configure_scene,set_diffuse_material
from cycles_coherent_sphere_reference import sphere_path,spreading,modes,checks


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reference(sources,detector,sphere,camera,output):
    resolution=256
    positions=[np.array(tuple(o.location),dtype=np.float64) for o in sources]
    center=np.array(tuple(sphere.location),dtype=np.float64)
    radius=float(sphere.modifiers[0].node_group.nodes['NativeSphere'].inputs['Radius'].default_value)
    x=float(detector.data.vertices[0].co.x)
    matrix=np.array([list(row) for row in camera.matrix_world],dtype=np.float64)
    rows,cols=np.indices((resolution,resolution),dtype=np.float64)
    right,up,forward=matrix[:3,0],matrix[:3,1],-matrix[:3,2]
    width=float(camera.data.ortho_scale)
    arrays=[np.zeros_like(rows) for _ in range(3)]
    offsets=(.125,.375,.625,.875)
    def pixel(ox,oy):
        origin=matrix[:3,3]+((cols+ox)/resolution-.5)[...,None]*width*right+((rows+oy)/resolution-.5)[...,None]*width*up
        return origin+((x-origin[...,0])/forward[0])[...,None]*forward
    receiver_center=pixel(.5,.5)
    for ox in offsets:
        for oy in offsets:
            receiver=pixel(ox,oy)
            paths=[sphere_path(p,receiver,center,radius) for p in positions]
            g=[spreading(p,np.array([0.,1.,0.]),np.array([0.,0.,1.]),radius) for p in paths]
            field=[modes(p) for p in paths]
            factors=[float(source.data.energy)*s/(4*math.pi**2) for source,s in zip(sources,g)]
            diagonal=factors[0]+factors[1]
            opd=paths[0]['optical_length']-paths[1]['optical_length']
            overlap=np.sum(field[0]*field[1],axis=(-2,-1))
            cross=2*np.sqrt(factors[0]*factors[1])*overlap*np.exp(-.5*(opd/1e-4)**2)*np.cos(2*math.pi*opd/550e-9)
            arrays[0]+=(diagonal+cross)/16;arrays[1]+=(diagonal-cross)/16;arrays[2]+=diagonal/16
    roi=(abs(receiver_center[...,1])<.09)&(abs(receiver_center[...,2]+.25)<.09)
    np.savez_compressed(output/'virtual_source_reference.npz',x_m=receiver_center[...,1],y_m=receiver_center[...,2],roi_mask=roi,phase_0_radiance=arrays[0],phase_pi_radiance=arrays[1],incoherent_radiance=arrays[2])
    result={'status':'Independent double angle-root and curvature-Jacobian reference, not a render','source_positions_m':[p.tolist() for p in positions],'center_m':center.tolist(),'radius_m':radius,'detector_x_m':x,'wavelength_m':550e-9,'coherence_length_m':1e-4,'pixel_filter':'4x4 independent BOX quadrature','source_ensemble':'Three independent equal world-axis dipole modes, projected transverse with sqrt(1/2) amplitude; same mode correlated across same-group sources','receiver':'Full world-vector inner product at coherent unit Lambertian receiver','path_inventory':'Direct has exactly zero detector cosine. Exterior convex sphere has one visible reflection; no interior/refraction/multiple-interface/caustic scope. Diffuse continuation0 isolates first detector encounter.','checks':checks(),'roi_means':[float(a[roi].mean()) for a in arrays]}
    (output/'virtual_source_reference.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def validate_native_points(sphere, expected_radius=.25):
    # Evaluate the actual point-cloud node output by converting that component
    # temporarily to vertices. This is a CPU authoring check, not mesh fitting
    # or the representation subsequently used by the renderer.
    group=sphere.modifiers[0].node_group
    output=next(n for n in group.nodes if n.bl_idname=='NodeGroupOutput')
    original=output.inputs['Geometry'].links[0].from_socket
    store=group.nodes.new('GeometryNodeStoreNamedAttribute');store.data_type='FLOAT';store.domain='POINT'
    store.inputs['Name'].default_value='checked_native_radius'
    field=group.nodes.new('GeometryNodeInputRadius')
    convert=group.nodes.new('GeometryNodePointsToVertices')
    group.links.new(original,store.inputs['Geometry']);group.links.new(field.outputs['Radius'],store.inputs['Value'])
    group.links.new(store.outputs['Geometry'],convert.inputs['Points']);group.links.new(convert.outputs['Mesh'],output.inputs['Geometry'])
    try:
        bpy.context.view_layer.update()
        evaluated=sphere.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh=evaluated.to_mesh()
        count=len(mesh.vertices)
        centers=[list(v.co) for v in mesh.vertices]
        radii=[v.value for v in mesh.attributes['checked_native_radius'].data]
        assert count==1 and centers==[[0.,0.,0.]] and radii==[expected_radius],(count,centers,radii)
        evaluated.to_mesh_clear()
        return {'actual_evaluated_point_count':count,'local_centers_m':centers,'radii_m':radii,'validation':'Actual point component converted temporarily to vertices for CPU readback; saved renderer output remains native Points'}
    finally:
        group.links.new(original,output.inputs['Geometry'])
        for node in (convert,store,field):group.nodes.remove(node)
        bpy.context.view_layer.update()


def main():
    output=Path(sys.argv[sys.argv.index('--')+1]).resolve();output.mkdir(parents=True,exist_ok=False)
    scene,*_=configure_scene(True)
    for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
    scene.cycles.use_coherent_specular_connections=True;scene.cycles.coherent_polarization_mode='VECTOR'
    scene.cycles.coherent_max_interface_events=1;scene.cycles.samples=128;scene.cycles.diffuse_bounces=0
    scene.render.resolution_x=scene.render.resolution_y=256
    # Explicit vertices give exact flat +X detector normal and represented x.
    x=float(np.float32(-.65));mesh=bpy.data.meshes.new('Exact flat detector geometry')
    mesh.from_pydata([(x,-.15,-.4),(x,.15,-.4),(x,.15,-.1),(x,-.15,-.1)],[],[(0,1,2),(0,2,3)])
    detector=bpy.data.objects.new('Flat white detector',mesh);scene.collection.objects.link(detector)
    mesh.materials.append(set_diffuse_material('Unit Lambertian', (1,1,1)));detector.cycles.coherent_interface='DETECTOR'
    material=bpy.data.materials.new('Unit ideal sphere mirror');material.use_nodes=True;n=material.node_tree.nodes;n.clear()
    glossy=n.new('ShaderNodeBsdfGlossy');glossy.distribution='GGX';glossy.inputs['Color'].default_value=(1,1,1,1);glossy.inputs['Roughness'].default_value=0
    out=n.new('ShaderNodeOutputMaterial');material.node_tree.links.new(glossy.outputs[0],out.inputs['Surface'])
    pointmesh=bpy.data.meshes.new('One explicit sphere center');pointmesh.from_pydata([(0,0,0)],[],[])
    sphere=bpy.data.objects.new('Native analytic mirror sphere',pointmesh);scene.collection.objects.link(sphere)
    pointmesh.materials.append(material);sphere.cycles.coherent_interface='MIRROR'
    modifier=sphere.modifiers.new('Native sphere from one point','NODES');group=bpy.data.node_groups.new('Native analytic sphere authoring','GeometryNodeTree');modifier.node_group=group
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry');group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    nodes=group.nodes;inp=nodes.new('NodeGroupInput');out=nodes.new('NodeGroupOutput');points=nodes.new('GeometryNodeMeshToPoints');points.name='NativeSphere';points.mode='VERTICES';points.inputs['Radius'].default_value=.25
    mat=nodes.new('GeometryNodeSetMaterial');mat.inputs['Material'].default_value=material
    group.links.new(inp.outputs['Geometry'],points.inputs['Mesh']);group.links.new(points.outputs['Points'],mat.inputs['Geometry']);group.links.new(mat.outputs['Geometry'],out.inputs['Geometry'])
    sources=[]
    for y in (-1e-5,1e-5):
        bpy.ops.object.light_add(type='POINT',location=(x,y,.35));source=bpy.context.object;source.data.energy=100;source.data.shadow_soft_size=0
        source.data.cycles.coherence_group=1;source.data.cycles.coherence_length_m=1e-4;source.data.cycles.coherence_wavelength_nm=550;source.data.cycles.coherence_phase=0;sources.append(source)
    bpy.ops.object.camera_add(location=(-.4,0,-.25));camera=bpy.context.object;camera.rotation_euler=(Vector((x,0,-.25))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=.24;scene.camera=camera
    bpy.context.view_layer.update();native_points=validate_native_points(sphere);ref=reference(sources,detector,sphere,camera,output)
    variants={}
    def save(name):
        path=output/(name+'.blend');bpy.ops.wm.save_as_mainfile(filepath=str(path));variants[name]={'path':str(path),'sha256':sha(path)}
    for name,phase,distinct in (('phase_0',0,False),('phase_pi',math.pi,False),('incoherent_connector_control',0,True)):
        sources[1].data.cycles.coherence_phase=phase;sources[1].data.cycles.coherence_group=2 if distinct else 1;save(name)
    sources[1].data.cycles.coherence_phase=0;sources[1].data.cycles.coherence_group=1
    old=sources[0].location.copy();sources[0].location=(0,0,0);save('reject_internal_source');sources[0].location=old
    sphere.scale=(1,1,1.01);save('reject_ellipsoid');sphere.scale=(1,1,1)
    scene.cycles.coherent_max_interface_events=2;scene.cycles.max_bounces=3;save('reject_multiple_events');scene.cycles.coherent_max_interface_events=1;scene.cycles.max_bounces=2
    sphere.cycles.coherent_interface='GLASS';save('reject_glass_sphere');sphere.cycles.coherent_interface='MIRROR'
    manifest={'binary':bpy.app.binary_path,'binary_sha256':sha(bpy.app.binary_path),'reference':ref,'native_geometry_validation':native_points,'specular_connections':True,'scene_variants':variants,'variants':variants,'rendered':False,'resolution':[256,256],'samples':128,'adaptive':False,'denoise':False,'scope':'First smooth geometry increment: native analytic exterior convex Mirror R only, one event; not general curved coherent transport'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'output':str(output),'scenes':len(variants),'rendered':False}))

if __name__=='__main__':main()
