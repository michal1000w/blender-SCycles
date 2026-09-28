#!/usr/bin/env python3
"""Author or CPU/Metal validate bounded closed-Glass and grating filter plumbing."""
import argparse,json,sys,re,hashlib,time,math
from pathlib import Path
import bpy,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cycles_native_polarizer_scene import setup,camera,save,triangle_film
from cycles_coherent_polarizer_scene import glass,pane,set_filter

def spectral_variance():
    text=(Path(__file__).resolve().parents[2]/'intern/cycles/kernel/tables.h').read_text()
    def table(name):
        body=text.split(name,1)[1].split('= {',1)[1].split('};',1)[0]
        return np.array([float(v) for v in re.findall(r'([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)f',body)])
    cie=table('cie_color_match').reshape(-1,3);spd=table('cie_d65_spd')
    u=(np.arange(200000)+.5)/200000;f=.9707633294863183*u+.021659159132699574
    lam=-np.log(1/f-1)/21.71348444564851+.5554867905834258
    grid=np.linspace(.38,.78,81);xyz=np.stack([np.interp(lam,grid,cie[:,k]) for k in range(3)],1)
    xyz*=np.interp(lam,grid,spd)[:,None]*.03785528242588043
    rgb=xyz@np.array([[3.2404542,-.969266,.0556434],[-1.5371385,1.8760108,-.2040259],[-.4985314,.041556,1.0572252]])
    rgb/= (21.71348444564851*f*(1-f)*.4/.9707633294863183)[:,None]
    return {'sensor_mean_rgb':rgb.mean(0).tolist(),'sensor_variance_rgb':rgb.var(0).tolist(),'quadrature_samples':len(u),'physical_target':.5,'systematic_sensor_table_margin':.002}

def main():
    p=argparse.ArgumentParser();p.add_argument('--author',action='store_true');p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--device',choices=['CPU','METAL'],default='CPU');p.add_argument('--modes',nargs='+',default=['PT','BDPT']);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if a.author:
        a.fixtures.mkdir(parents=True,exist_ok=False);cases={}
        for name,ior,depth,on in [('closed_clear',1.5,None,False),('closed_filter',1.5,None,True),('zero_depth_filter',1.,0.,True),('positive_depth_filter',1.,150.,True)]:
            s=setup();s.cycles.samples=1024;s.cycles.max_bounces=32;s.cycles.transmission_bounces=30;s.cycles.glossy_bounces=30;s.cycles.min_light_bounces=30
            s.world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=1
            mat,node=glass(name,ior);set_filter(node,on,.37,False)
            if depth is None:
                bpy.ops.mesh.primitive_cube_add(size=2,location=(0,0,1));obj=bpy.context.object;obj.scale=(2,2,.1);obj.rotation_euler.z=.37;obj.data.materials.append(mat)
                # Closed triangular prism: each viewed front/back face is one primitive,
                # avoiding the archived coplanar quad seam without masking pixels.
                obj.data.clear_geometry();obj.data.from_pydata([(-2,-2,-1),(2,-2,-1),(0,4,-1),(-2,-2,1),(2,-2,1),(0,4,1)],[],[(0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)])
            else:
                node.inputs['Diffraction Weight'].default_value=1;node.inputs['Diffraction Depth'].default_value=depth
                triangle_film(pane(name,(0,0,1),(1,0,0),(0,1,0),2,mat,'OFF'),2)
            camera(s,(.11,.073,3),(.11,.073,0),.5)
            entry=save(a.fixtures/(name+'.blend'));entry.update(expected=1 if not on else .52 if depth is None else .5,variance=.0096 if name=='closed_filter' else 0,spectral=depth==150)
            cases[name]=entry
        manifest={'samples':1024,'pixels':4096,'seed':19,'cases':cases,'spectral':spectral_variance(),'gate':'6 sigma mean; 2 sigma per-pixel RMS; sensor systematic margin .002; deterministic margin2e-6','scope':'Closed slab first R .04 unaffected, first T .96 filtered .5 gives .52, variance .0096. ZeroDepth host folds grating OFF. Positive default depth with exact matched IOR has zero optical contrast, true identity grating atom with payload; not nontrivial diffraction spectrum. Closed triangular prism single front/back primitives avoid archived coplanar quad seam; all pixels retained. CPU BDPT unsupported and skipped.'}
        (a.fixtures/'manifest.json').write_text(json.dumps(manifest,indent=2));return
    a.output.mkdir(parents=True,exist_ok=False);m=json.loads((a.fixtures/'manifest.json').read_text());report={'manifest':m,'binary':bpy.app.binary_path,'cases':{},'passed':True,'skipped':[]}
    for mode in a.modes:
        if a.device=='CPU' and mode=='BDPT':report['skipped'].append('CPU BDPT unsupported: backend requires Metal');continue
        for name,e in m['cases'].items():
            bpy.ops.wm.open_mainfile(filepath=e['path']);s=bpy.context.scene;s.cycles.seed=19;s.cycles.use_bidirectional_path_tracing=mode=='BDPT';s.cycles.bdpt_max_bounces=32;s.cycles.device='CPU' if a.device=='CPU' else 'GPU'
            if a.device=='METAL':
                prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
                for d in prefs.devices:d.use=d.type=='METAL'
            out=a.output/(mode+'_'+name+'.exr');s.render.image_settings.file_format='OPEN_EXR';s.render.image_settings.color_depth='32';s.render.filepath=str(out.resolve());start=time.perf_counter();bpy.ops.render.render(write_still=True)
            im=bpy.data.images.load(str(out.resolve()),check_existing=False);rgb=np.array(im.pixels[:]).reshape(im.size[1],im.size[0],im.channels)[...,:3];bpy.data.images.remove(im)
            variance=np.array(m['spectral']['sensor_variance_rgb'])*.25 if e['spectral'] else np.full(3,e['variance']);margin=.002 if e['spectral'] else 2e-6
            mean_gate=6*np.sqrt(variance/(1024*4096))+margin;rmse_gate=2*np.sqrt(variance/1024)+margin
            err=rgb-e['expected'];mean_error=np.abs(err.mean((0,1)));rmse=np.sqrt(np.mean(err*err,(0,1)));passed=bool(np.isfinite(rgb).all() and np.all(mean_error<=mean_gate) and np.all(rmse<=rmse_gate))
            report['cases'][mode+'_'+name]={'mean_rgb':rgb.mean((0,1)).tolist(),'mean_error_rgb':mean_error.tolist(),'rmse_rgb':rmse.tolist(),'mean_gate_rgb':mean_gate.tolist(),'rmse_gate_rgb':rmse_gate.tolist(),'passed':passed,'seconds':time.perf_counter()-start};report['passed'] &=passed
            (a.output/'report.json').write_text(json.dumps(report,indent=2))
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:raise RuntimeError('Shader integration physical gates failed')
if __name__=='__main__':main()
