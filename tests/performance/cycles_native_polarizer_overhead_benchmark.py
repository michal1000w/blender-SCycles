# SPDX-License-Identifier: Apache-2.0
"""Optional native polarizer overhead; prepare/run separately from acceptance.

Blender --factory-startup -b --python this.py -- --fixture PASS_S --output NEW
Use --device METAL only when the parent task grants the sole GPU slot.
One warmup and three measured renders per PT/BDPT checkbox state. No image
comparison between states: enabling the physical filter changes radiance.
"""
import argparse,hashlib,json,statistics,struct,sys,time
from pathlib import Path
import bpy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def glass_nodes():
    return [(m,n) for m in bpy.data.materials if m.use_nodes
            for n in m.node_tree.nodes if n.type=='BSDF_GLASS']


def sockets():
    return [{'material':m.name,'node':n.name,
             'enabled':bool(n.inputs['Polarizer'].default_value),
             'enabled_linked':n.inputs['Polarizer'].is_linked,
             'angle':float(n.inputs['Polarizer Angle'].default_value),
             'angle_hex':float(n.inputs['Polarizer Angle'].default_value).hex(),
             'angle_linked':n.inputs['Polarizer Angle'].is_linked}
            for m,n in glass_nodes()]


def geometry_digest():
    """Fingerprint geometry/transforms independently of the checkbox values."""
    h=hashlib.sha256()
    for o in sorted(bpy.context.scene.objects,key=lambda o:o.name):
        h.update(o.name.encode());h.update(o.type.encode())
        for row in o.matrix_world:
            for value in row:h.update(struct.pack('<d',float(value)))
        if o.type=='MESH':
            for v in o.data.vertices:
                for value in v.co:h.update(struct.pack('<d',float(value)))
            for f in o.data.polygons:
                h.update(struct.pack('<I',len(f.vertices)))
                for index in f.vertices:h.update(struct.pack('<I',index))
                h.update(struct.pack('<I',f.material_index))
        h.update(json.dumps([m.name if m else None for m in getattr(o.data,'materials',[])],sort_keys=True).encode())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['CPU','METAL'],default='METAL')
    p.add_argument('--modes',nargs='+',choices=['PT','BDPT'],default=['PT','BDPT'])
    p.add_argument('--warmup',action='store_true',required=True,
                   help='Required: Metal waits for specialization before measured renders')
    p.add_argument('--save-images',action='store_true',help='Write one final EXR per state outside timing')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    a.output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(a.fixture.resolve()))
    scene=bpy.context.scene
    assert not scene.cycles.use_coherent_specular_connections
    targets=[(m,n) for m,n in glass_nodes() if n.inputs['Polarizer'].default_value]
    assert len(targets)==1,'Expected the single camera analyzer in native_brewster_photo/pass_s.blend'
    assert all(not n.inputs['Polarizer'].is_linked and not n.inputs['Polarizer Angle'].is_linked
               for _,n in glass_nodes()),'Constant checkbox/angles required for this benchmark'
    original_angles={(m.name,n.name):float(n.inputs['Polarizer Angle'].default_value).hex()
                     for m,n in glass_nodes()}
    original_sockets=sockets();geometry=geometry_digest()
    scene.render.engine='CYCLES';scene.render.resolution_x=320;scene.render.resolution_y=240
    scene.render.resolution_percentage=100;scene.render.use_persistent_data=True
    scene.cycles.samples=64;scene.cycles.seed=19
    scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False
    scene.cycles.use_guiding=False;scene.cycles.use_photon_mapping=False
    scene.cycles.shading_system=False
    scene.cycles.bdpt_max_bounces=12
    scene.cycles.device='GPU' if a.device=='METAL' else 'CPU'
    devices=[]
    if a.device=='METAL':
        prefs=bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type='METAL';prefs.get_devices()
        assert any(d.type=='METAL' for d in prefs.devices)
        for d in prefs.devices:
            d.use=d.type=='METAL'
            devices.append({'name':d.name,'type':d.type,'enabled':bool(d.use)})
    report={'scope':'optional native photographic analyzer checkbox overhead; geometry unchanged',
            'fixture':str(a.fixture.resolve()),'fixture_sha256':digest(a.fixture),
            'binary':bpy.app.binary_path,'binary_sha256':digest(bpy.app.binary_path),
            'script_sha256':digest(__file__),'blender_version':bpy.app.version_string,
            'device':a.device,'devices':devices,'geometry_sha256':geometry,
            'original_sockets':original_sockets,
            'settings':{'width':320,'height':240,'samples':64,'seed':19,
                        'adaptive_sampling':False,'denoising':False,'persistent_data':True,
                        'warmups_per_state':1,'measured_runs_per_state':3,
                        'metal_specialization_warmup_cli':a.warmup,
                        'timing_scope':'bpy.ops.render.render including synchronization; file writing excluded',
                        'bdpt_max_bounces':12},
            'cases':{},'ratios':{},'completed':False,
            'interpretation':'Warm timings quantify optional-checkbox cost, not image equality, statistical significance, or default-OFF correctness.'}
    def save():
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for mode in a.modes:
        scene.cycles.use_bidirectional_path_tracing=mode=='BDPT'
        for enabled in [False,True]:
            for _,n in targets:n.inputs['Polarizer'].default_value=enabled
            assert geometry_digest()==geometry
            assert {(m.name,n.name):float(n.inputs['Polarizer Angle'].default_value).hex()
                    for m,n in glass_nodes()}==original_angles
            key=mode+('_ON' if enabled else '_OFF')
            current=sockets()
            record={'mode':mode,'polarizer_enabled':enabled,'sockets':current,
                    'feature_state':{'expected_polarization_feature_enabled':any(x['enabled'] or x['enabled_linked'] for x in current),
                                     'bidirectional':mode=='BDPT','guiding':False,'photon_mapping':False,
                                     'coherent_connections':False,
                                     'note':'Expected feature activation from shader inputs; internal kernel mask is not exposed by bpy.'},
                    'geometry_sha256':geometry,'runs':[]}
            report['cases'][key]=record;save()
            for iteration in range(4):
                start=time.perf_counter();result=bpy.ops.render.render();elapsed=time.perf_counter()-start
                assert 'FINISHED' in result and elapsed>0
                assert geometry_digest()==geometry
                record['runs'].append({'iteration':iteration,'warmup':iteration==0,'seconds':elapsed})
                print('POLARIZER_BENCHMARK',key,iteration,elapsed,flush=True);save()
            measured=[x['seconds'] for x in record['runs'][1:]]
            record.update(warmup_seconds=record['runs'][0]['seconds'],measured_seconds=measured,
                          median_seconds=statistics.median(measured),minimum_seconds=min(measured),maximum_seconds=max(measured))
            if a.save_images:
                scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32'
                image=a.output/(key+'.exr');bpy.data.images['Render Result'].save_render(str(image.resolve()),scene=scene)
                record.update(image=str(image.resolve()),image_sha256=digest(image))
            save()
        off=report['cases'][mode+'_OFF']['median_seconds'];on=report['cases'][mode+'_ON']['median_seconds']
        report['ratios'][mode]={'on_over_off_median':on/off,'on_minus_off_median_seconds':on-off}
        save()
    for _,n in targets:n.inputs['Polarizer'].default_value=True
    assert sockets()==original_sockets
    report['completed']=True;save()


if __name__=='__main__':main()
