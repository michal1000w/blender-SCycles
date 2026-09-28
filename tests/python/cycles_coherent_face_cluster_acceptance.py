#!/usr/bin/env python3
"""Prepare face-cluster fixtures without rendering; independent scalar unfolding.

The slab reuses the existing double Snell/Jones oracle. Folded mirrors use
finite rectangles, explicit segment visibility and image-source isometries.
Short Gaussian coherence suppresses cross-family terms in the render fixture;
long-coherence cross-family algebra is separately checked at selected points.
"""
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import hashlib
import shutil
import numpy as np

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None

RESOLUTION = 256
LC = 1e-4
WAVELENGTH = 550e-9
SOURCE_OFFSET = 10e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(point, patch):
    a, b = patch['tangents']
    valid = ((point[..., a] >= patch['bounds'][0][0]) &
             (point[..., a] <= patch['bounds'][0][1]) &
             (point[..., b] >= patch['bounds'][1][0]) &
             (point[..., b] <= patch['bounds'][1][1]))
    for lo, hi in patch.get('holes', []):
        valid &= ~((point[..., a] > lo[0]) & (point[..., a] < hi[0]) &
                   (point[..., b] > lo[1]) & (point[..., b] < hi[1]))
    return valid


def reflect(point, patch):
    result = np.array(point, copy=True)
    result[..., patch['axis']] = 2 * patch['plane'] - result[..., patch['axis']]
    return result


def path_geometry(source, receiver, patches, sequence):
    images = [np.asarray(source, dtype=np.float64)]
    for index in sequence:
        images.append(reflect(images[-1], patches[index]))
    start = receiver
    points = []
    valid = np.ones(receiver.shape[:-1], dtype=bool)
    for step in range(len(sequence) - 1, -1, -1):
        patch = patches[sequence[step]]
        axis = patch['axis']
        delta = images[step + 1] - start
        denominator = delta[..., axis]
        t = np.divide(patch['plane'] - start[..., axis], denominator,
                      out=np.zeros_like(denominator), where=np.abs(denominator) > 1e-14)
        hit = start + t[..., None] * delta
        valid &= (np.abs(denominator) > 1e-14) & (t > 1e-10) & (t < 1 - 1e-10)
        valid &= inside(hit, patch)
        points.append(hit)
        start = hit
    physical = [np.broadcast_to(source, receiver.shape), *reversed(points), receiver]
    for index, patch_index in enumerate(sequence):
        patch = patches[patch_index]
        axis = patch['axis']
        incident = physical[index][..., axis] - patch['plane']
        outgoing = physical[index + 2][..., axis] - patch['plane']
        valid &= incident * outgoing > 0
    # Independent finite-face occlusion: only an intended segment endpoint may
    # lie on a mirror. A same-object different plane is still a blocker.
    for begin, end in zip(physical[:-1], physical[1:]):
        delta = end - begin
        for patch in patches:
            denominator = delta[..., patch['axis']]
            t = np.divide(patch['plane'] - begin[..., patch['axis']], denominator,
                          out=np.zeros_like(denominator), where=np.abs(denominator) > 1e-14)
            hit = begin + t[..., None] * delta
            blocked = (np.abs(denominator) > 1e-14) & (t > 1e-8) & (t < 1 - 1e-8) & inside(hit, patch)
            valid &= ~blocked
    virtual = images[-1]
    radius = np.linalg.norm(receiver - virtual, axis=-1)
    length = sum(np.linalg.norm(end - begin, axis=-1)
                 for begin, end in zip(physical[:-1], physical[1:]))
    if np.any(valid):
        assert np.max(np.abs(length[valid] - radius[valid])) < 2e-14
    return valid, virtual, radius, physical


def all_fields(sources, receiver, patches):
    fields = []
    sequences = [()] + [(i,) for i in range(len(patches))]
    sequences += [pair for pair in itertools.product(range(len(patches)), repeat=2) if pair[0] != pair[1]]
    for light, source in enumerate(sources):
        for sequence in sequences:
            valid, virtual, radius, points = path_geometry(source, receiver, patches, sequence)
            cosine = (receiver[..., 0] - virtual[0]) / radius  # detector normal -X
            power = np.where(valid & (cosine > 0), 10.0 * cosine / (4 * math.pi**2 * radius**2), 0.0)
            fields.append((light, sequence, virtual, radius, power))
    return fields


def intensity(fields, receiver, phase, coherence_length=LC, separate_groups=False):
    result = sum(field[4] for field in fields)
    for left, right in itertools.combinations(fields, 2):
        if separate_groups and left[0] != right[0]:
            continue
        # Exact difference of squared ranges, factored before division. This
        # avoids subtracting meter-scale paths to obtain source-scale OPD.
        a, b = left[2], right[2]
        difference = np.sum((b - a) * (2 * receiver - a - b), axis=-1) / (left[3] + right[3])
        gamma = np.exp(-0.5 * (difference / coherence_length)**2)
        source_phase = (left[0] - right[0]) * phase
        result += 2 * np.sqrt(left[4] * right[4]) * gamma * np.cos(2 * math.pi * difference / WAVELENGTH + source_phase)
    assert np.isfinite(result).all() and np.min(result) >= -1e-10
    return result


def patches_for(variant):
    patches = [{'axis': 2, 'plane': 0.0, 'tangents': [0, 1], 'bounds': [[0.0, 2.0], [-.4, .4]]},
               {'axis': 0, 'plane': 0.0, 'tangents': [1, 2], 'bounds': [[-.4, .4], [0.0, 2.0]]}]
    if variant == 'gap':
        patches[1]['holes'] = [([-.07, .005], [.07, .08])]
    if variant == 'wrong_face':
        patches.append({'axis': 0, 'plane': .15, 'tangents': [1, 2], 'bounds': [[-.1, .1], [.02, .3]]})
    # Every descriptor is derived from values Blender stores in mesh float3.
    for patch in patches:
        patch['plane'] = float(np.float32(patch['plane']))
        patch['bounds'] = np.asarray(patch['bounds'], dtype=np.float32).astype(float).tolist()
        patch['holes'] = [(np.asarray(lo,dtype=np.float32).astype(float).tolist(),
                           np.asarray(hi,dtype=np.float32).astype(float).tolist()) for lo,hi in patch.get('holes',[])]
    return patches


def self_checks():
    sources = np.asarray([[.3,-SOURCE_OFFSET,.2],[.3,SOURCE_OFFSET,.2]],dtype=np.float32).astype(float)
    receiver = np.asarray([[1,0,.8],[1,.05,.7],[1,-.08,.9]],dtype=np.float32).astype(float)
    fields = all_fields(sources, receiver, patches_for('folded'))
    counts = [int(sum(field[4][i] > 0 for field in fields)) for i in range(len(receiver))]
    assert all(count == 8 for count in counts)  # four families per source
    long = intensity(fields, receiver, .37, 1e9)
    # Independent explicit complex sum, with a common reference path removed.
    anchor = fields[0]
    value = np.zeros(len(receiver),dtype=np.complex128)
    for light, sequence, virtual, radius, power in fields:
        difference = np.sum((anchor[2]-virtual)*(2*receiver-virtual-anchor[2]),axis=-1)/(radius+anchor[3])
        value += np.sqrt(power)*np.exp(1j*(2*math.pi*difference/WAVELENGTH + light*.37))
    error = float(np.max(np.abs(long - np.abs(value)**2)))
    assert error < 1e-8
    gap = all_fields(sources,receiver,patches_for('gap'))
    assert sum(field[4][0] > 0 for field in gap) < counts[0]
    wrong = all_fields(sources,receiver,patches_for('wrong_face'))
    assert sum(field[4][0] > 0 and field[1] == (0,1) for field in wrong) == 0
    return {'passed':True,'visible_paths_at_samples':counts,'long_coherence_complex_sum_max_error':error,
            'gap_eliminates_finite_hits':True,'different_plane_blocks_original_two_bounce_path':True}


def module(name):
    path = Path(__file__).with_name('cycles_coherent_'+name+'_acceptance.py')
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def rectangles(patch):
    # Partition a rectangular hole into a shared-vertex 3x3 ring. Bounds stay
    # unchanged, so accepting the plane rectangle instead of actual primitives
    # would demonstrably reintroduce absent paths.
    bounds = patch['bounds']
    if not patch['holes']:
        return [bounds]
    lo,hi=patch['holes'][0];a,b=bounds
    grid_a=[a[0],lo[0],hi[0],a[1]];grid_b=[b[0],lo[1],hi[1],b[1]]
    return [[[grid_a[i],grid_a[i+1]],[grid_b[j],grid_b[j+1]]]
            for i,j in itertools.product(range(3),repeat=2) if (i,j)!=(1,1)]


def mesh_object(name,patches,material,mode):
    vertices=[];faces=[];vertex_indices={}
    for patch in patches:
        for bounds in rectangles(patch):
            face=[]
            for a,b in ((bounds[0][0],bounds[1][0]),(bounds[0][1],bounds[1][0]),
                        (bounds[0][1],bounds[1][1]),(bounds[0][0],bounds[1][1])):
                point=[0.,0.,0.];point[patch['axis']]=patch['plane'];point[patch['tangents'][0]]=a;point[patch['tangents'][1]]=b
                key=tuple(point)
                if key not in vertex_indices:
                    vertex_indices[key]=len(vertices);vertices.append(point)
                face.append(vertex_indices[key])
            if patch.get('reverse'): face.reverse()
            faces.append(face)
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(vertices,[],faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material);obj.cycles.coherent_interface=mode
    return obj


def save_phases(scene,sources,output):
    output.mkdir();files={}
    for name,phase in [('phase_0',0.0),('phase_pi',math.pi)]:
        for index,source in enumerate(sources):source.data.cycles.coherence_phase=index*phase
        path=output/(name+'.blend');scene.render.filepath='//'+name
        assert scene.cycles.use_coherent_specular_connections and not scene.cycles.use_adaptive_sampling
        bpy.ops.wm.save_as_mainfile(filepath=str(path));files[name]={'path':str(path),'sha256':sha(path)}
    return files


def worker_manifest(folder, scenes, reference, scope):
    target=folder/'virtual_source_reference.npz'
    if reference != target:shutil.copy2(reference,target)
    (folder/'manifest.json').write_text(json.dumps({'specular_connections':True,
        'scene_variants':scenes,'scope':scope,'reference_sha256':sha(target)},indent=2)+'\n')


def prepare_slab(output):
    slab=module('slab');slab.RESOLUTION=RESOLUTION
    scene,detector,old_faces,sources,camera=slab.configure_scene()
    scene.cycles.use_coherent_specular_connections=True;scene.cycles.samples=128
    for source,y in zip(sources,[-SOURCE_OFFSET,SOURCE_OFFSET]):source.location.y=y
    material=old_faces[0].data.materials[0]
    detector_x=float(detector.location.x);detector_material=detector.data.materials[0]
    bpy.data.objects.remove(detector,do_unlink=True)
    detector=mesh_object('Exact axis-aligned slab detector',[{'axis':0,'plane':detector_x,
        'tangents':[1,2],'bounds':[[-float(np.float32(.2)),float(np.float32(.2))]]*2,
        'holes':[],'reverse':True}],detector_material,'DETECTOR')
    patches=[]
    for old in old_faces:
        patches.append({'axis':0,'plane':float(old.location.x),'tangents':[1,2],
                        'bounds':[[-float(np.float32(.4)),float(np.float32(.4))]]*2,'holes':[],
                        'reverse':old.location.x < .5})
        bpy.data.objects.remove(old,do_unlink=True)
    separate=[mesh_object('Separate slab face '+str(i),[p],material,'GLASS') for i,p in enumerate(patches)]
    bpy.context.view_layer.update()
    slab.AIR_BEFORE_M=patches[0]['plane'];slab.SLAB_THICKNESS_M=patches[1]['plane']-patches[0]['plane'];slab.AIR_AFTER_M=detector_x-patches[1]['plane']
    common=output/'slab_reference';common.mkdir();slab.make_reference(sources,camera,common)
    result={'separate':save_phases(scene,sources,output/'slab_separate')}
    for obj in separate:bpy.data.objects.remove(obj,do_unlink=True)
    mesh_object('One object, both slab interfaces',patches,material,'GLASS')
    result['joined']=save_phases(scene,sources,output/'slab_joined')
    for name,scenes in result.items():
        worker_manifest(output/('slab_'+name),scenes,common/'snell_slab_reference.npz','two-transmission '+name+' object slab')
    return {'scenes':result,'reference':str(common/'snell_slab_reference.npz'),'planes':patches,
            'scope':'same exact saved float triangles and existing independent double Snell/Jones reference; two transmissions only'}


def prepare_folded(output,variant):
    mirror=module('mirror');slab=module('slab');slab.RESOLUTION=RESOLUTION
    scene,old_detector,old_interface,masks,sources,camera=mirror.configure_scene(True)
    material=old_interface.data.materials[0];diffuse=old_detector.data.materials[0]
    for obj in [old_detector,old_interface,*masks]:bpy.data.objects.remove(obj,do_unlink=True)
    mesh_object('Flat Lambertian detector',[{'axis':0,'plane':1.,'tangents':[1,2],
        'bounds':[[-.2,.2],[.6,1.]],'holes':[],'reverse':True}],diffuse,'DETECTOR')
    patches=patches_for(variant);mesh_object('Folded ideal mirror '+variant,patches,material,'MIRROR')
    for source,y in zip(sources,[-SOURCE_OFFSET,SOURCE_OFFSET]):
        source.location=(.3,y,.2);source.data.cycles.coherence_length_m=LC
    camera.location=(.8,0,.8);camera.rotation_euler=Vector((1.,0.,0.)).to_track_quat('-Z','Y').to_euler()
    scene.cycles.use_coherent_specular_connections=True;scene.cycles.coherent_max_interface_events=2
    scene.cycles.samples=128;scene.cycles.max_bounces=3;scene.cycles.glossy_bounces=2
    scene.render.resolution_x=scene.render.resolution_y=RESOLUTION
    bpy.context.view_layer.update()
    source_positions=np.asarray([list(s.location) for s in sources],dtype=np.float64)
    assert camera.location.x < 1.0  # view the arriving fields' Lambertian side
    folder=output/variant;scenes=save_phases(scene,sources,folder)
    row,col=np.indices((RESOLUTION,RESOLUTION),dtype=np.float64)
    cy,cz=slab.camera_world_at_detector(camera,1.,col+.5,row+.5)
    roi=(np.abs(cy)<.15)&(cz>.65)&(cz<.95)
    arrays={name:np.zeros_like(cy) for name in ['phase_0_radiance','phase_pi_radiance','incoherent_radiance']}
    for oy in (.125,.375,.625,.875):
        for ox in (.125,.375,.625,.875):
            y,z=slab.camera_world_at_detector(camera,1.,col+ox,row+oy)
            receiver=np.stack((np.ones_like(y),y,z),axis=-1)
            fields=all_fields(source_positions,receiver,patches)
            arrays['phase_0_radiance']+=intensity(fields,receiver,0.0)/16
            arrays['phase_pi_radiance']+=intensity(fields,receiver,math.pi)/16
            arrays['incoherent_radiance']+=intensity(fields,receiver,0.0,separate_groups=True)/16
    np.savez_compressed(folder/'virtual_source_reference.npz',**arrays,roi_mask=roi,x_m=cy,y_m=cz)
    worker_manifest(folder,scenes,folder/'virtual_source_reference.npz',variant+' finite-face mirror')
    return {'scenes':scenes,'reference':str(folder/'virtual_source_reference.npz'),'patches':patches,
            'source_positions':source_positions.tolist(),'camera_matrix_world':[list(r) for r in camera.matrix_world],
            'scope':'scalar direct/one/two reflection families with independent finite-face visibility; Gaussian Lc1e-4 suppresses cross-family interference, two-source phase retained'}


def main():
    checks=self_checks()
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    if args==['--self-test']:
        print(json.dumps(checks));return
    if bpy is None:raise RuntimeError('Fixture generation requires Blender')
    if args[0]=='--isolated-first-receiver-from':
        source=Path(args[1]).resolve();output=Path(args[2]).resolve()
        output.mkdir(parents=True,exist_ok=False)
        previous=json.loads((source/'manifest.json').read_text())
        result={'scope':'isolated first Lambertian receiver; diffuse continuation disabled',
                'source_fixture':str(source),'source_manifest_sha256':sha(source/'manifest.json'),
                'binary':bpy.app.binary_path,'binary_sha256':sha(bpy.app.binary_path),
                'script_sha256':sha(__file__),'unchanged_gates':previous['gates'],
                'design_note':'v4 diffuse_bounces1 permits receiver-wall-receiver illumination outside first-receiver oracle; v5 diffuse_bounces0 isolates the declared reference. Sources/triangles/reference arrays unchanged; v4 renders preserved.'}
        for group in ['folded','gap','wrong_face']:
            folder=output/group;folder.mkdir();files={}
            for name in ['phase_0','phase_pi']:
                bpy.ops.wm.open_mainfile(filepath=previous[group]['scenes'][name]['path'])
                scene=bpy.context.scene;scene.cycles.diffuse_bounces=0
                assert scene.cycles.samples==128 and scene.cycles.max_bounces==3 and scene.cycles.glossy_bounces==2
                path=folder/(name+'.blend');scene.render.filepath='//'+name
                scene['coherence_acceptance_scope']='isolated first receiver, no diffuse continuation'
                bpy.ops.wm.save_as_mainfile(filepath=str(path));files[name]={'path':str(path),'sha256':sha(path)}
            original=source/group/'virtual_source_reference.npz'
            worker_manifest(folder,files,original,group+' isolated first receiver')
            result[group]={'scenes':files,'reference_sha256':sha(original),'reference_hash_reused_exactly':sha(folder/'virtual_source_reference.npz')==sha(original)}
        (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return
    output=Path(args[0]).resolve();output.mkdir(parents=True,exist_ok=False)
    result={'self_checks':checks,'binary':bpy.app.binary_path,'binary_sha256':sha(bpy.app.binary_path),
            'script_sha256':sha(__file__),'prepared_only':True,'samples':128,'adaptive':False,'denoise':False,
            'resolution':[RESOLUTION,RESOLUTION],
            'physical_source_baseline_m':2*SOURCE_OFFSET,
            'sampling_design':'20um baseline chosen before GPU so 256px/128fixed BOX sampling resolves fringes under unchanged absolute gates; original100um v3 unrendered preserved',
            'gates':{'mean_max':.005,'rmse_max':.012,'phase_rmse_max':.012,'negative_roundoff':1e-6},
            'slab':prepare_slab(output)}
    for variant in ['folded','gap','wrong_face']:result[variant]=prepare_folded(output,variant)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
