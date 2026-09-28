#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Factory-created stock-compatible IOR1 seam repro; CPU tiny renders only."""
import bpy,sys,json
from pathlib import Path
import numpy as np


def main():
    output=Path(sys.argv[sys.argv.index('--')+1]).resolve();output.mkdir(parents=True,exist_ok=False);report={'binary':bpy.app.binary_path,'version':bpy.app.version_string,'scope':'Minimal native ordinary GlassIOR1/World1 CPU test; factory created, no custom coherent flags or polarizer API','cases':{}}
    for label,count,triangle,shift in [('quad1',1,False,0),('quad2',2,False,0),('quad3',3,False,0),('quad3_shifted',3,False,.071),('triangle3',3,True,0)]:
        bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=1;scene.cycles.seed=19;scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=False;scene.cycles.max_bounces=12;scene.cycles.min_light_bounces=8;scene.cycles.min_transparent_bounces=8;scene.cycles.transmission_bounces=10;scene.cycles.glossy_bounces=8;scene.cycles.diffuse_bounces=3;scene.cycles.transparent_max_bounces=8;scene.cycles.pixel_filter_type='BOX';scene.cycles.filter_width=1;scene.render.resolution_x=scene.render.resolution_y=64;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32';scene.render.film_transparent=False
        world=bpy.data.worlds.new('White unpolarized World1');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=1;scene.world=world
        material=bpy.data.materials.new('Ordinary native index matched Glass');material.use_nodes=True;nodes=material.node_tree.nodes;nodes.clear();glass=nodes.new('ShaderNodeBsdfGlass');glass.distribution='GGX';glass.inputs['Color'].default_value=(1,1,1,1);glass.inputs['Roughness'].default_value=0;glass.inputs['IOR'].default_value=1;out=nodes.new('ShaderNodeOutputMaterial');material.node_tree.links.new(glass.outputs[0],out.inputs['Surface'])
        for j in range(count):
            z=1.5-.5*j;mesh=bpy.data.meshes.new('One physical '+('triangle' if triangle else 'quad'));vertices=[(-2+shift,-2,z),(2+shift,-2,z),(shift,4,z)] if triangle else [(-2+shift,-2,z),(2+shift,-2,z),(2+shift,2,z),(-2+shift,2,z)];mesh.from_pydata(vertices,[],[(0,1,2)] if triangle else [(0,1,2,3)]);obj=bpy.data.objects.new('Finite film'+str(j),mesh);scene.collection.objects.link(obj);mesh.materials.append(material)
        bpy.ops.object.camera_add(location=(0,0,2));camera=bpy.context.object;camera.data.type='ORTHO';camera.data.ortho_scale=.5;camera.data.clip_start=.001;camera.data.clip_end=100;scene.camera=camera;bpy.context.view_layer.update();blend=output/(label+'.blend');bpy.ops.wm.save_as_mainfile(filepath=str(blend));scene.render.filepath=str(output/(label+'.exr'));bpy.ops.render.render(write_still=True);image=bpy.data.images.load(scene.render.filepath,check_existing=False);a=np.array(image.pixels[:]).reshape(image.size[1],image.size[0],image.channels)[:,:,:3].mean(axis=2);ys,xs=np.where(a<.1);report['cases'][label]={'black_pixels':len(xs),'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),'all_black_on_main_diagonal':bool(len(xs)>0 and np.all(xs==ys)),'black_coordinates':list(zip(xs.tolist(),ys.tolist())),'blend':str(blend),'raw_exr':scene.render.filepath}
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
