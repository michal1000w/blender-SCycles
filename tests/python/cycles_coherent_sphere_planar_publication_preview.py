#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Author labeled wide physical previews; acceptance optical scenes stay untouched."""
import bpy,sys,json
from pathlib import Path
from mathutils import Vector


def emission(name,color):
    mat=bpy.data.materials.new(name);mat.use_nodes=True;nodes=mat.node_tree.nodes;nodes.clear();shader=nodes.new('ShaderNodeEmission');shader.inputs['Color'].default_value=(*color,1);shader.inputs['Strength'].default_value=1;out=nodes.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(shader.outputs[0],out.inputs['Surface']);return mat


def main():
    root=Path(sys.argv[sys.argv.index('--')+1]).resolve();manifest={}
    for kind in ('R','TT'):
        folder=root/kind;bpy.ops.wm.open_mainfile(filepath=str(folder/'ordinary_physical_preview.blend'));scene=bpy.context.scene;camera=scene.camera
        scene.render.resolution_x=640;scene.render.resolution_y=480;scene.cycles.samples=256;scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=True
        camera.data.type='ORTHO';camera.data.ortho_scale=.058 if kind=='TT' else .027;camera.data.clip_start=.001
        scene.view_settings.view_transform='AgX';scene.view_settings.exposure=0
        blue=emission('Preview source marker',(1,.22,.08));white=emission('Preview annotation',(.8,.9,1))
        lights=[o for o in scene.objects if o.type=='LIGHT' and o.data.type=='POINT']
        for light in lights:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=.00045 if kind=='TT' else .00015,location=light.location)
            marker=bpy.context.object;marker.name='Illustrative point source marker';marker.data.materials.append(blue)
        source=Vector(lights[0].location);detector=next(o for o in scene.objects if o.cycles.coherent_interface=='DETECTOR');receiver=Vector(detector.data.vertices[0].co)
        if kind=='TT':labels=[('Glass sphere',(0,0,.013)),('Finite mirror',(.016,0,.011)),('2 coherent sources',tuple(source+Vector((0,-.011,.002)))),('Detector crop',(-.014,-.008,.004))]
        else:labels=[('Mirror sphere',(0,0,.004)),('Finite mirror',(-.010,0,.003)),('2 coherent sources',tuple(source+Vector((0,-.004,.002)))),('Detector crop',(-.006,-.004,.001))]
        for text,position in labels:
            curve=bpy.data.curves.new('Annotation '+text,'FONT');curve.body=text;curve.size=.0015 if kind=='TT' else .00065;curve.align_x='CENTER';curve.align_y='CENTER';obj=bpy.data.objects.new('Preview label '+text,curve);scene.collection.objects.link(obj);obj.location=position;obj.rotation_euler=camera.rotation_euler;curve.materials.append(white)
        scene.render.image_settings.file_format='PNG';scene.render.filepath=str(folder/'publication_physical_preview.png');path=folder/'publication_physical_preview.blend';bpy.ops.wm.save_as_mainfile(filepath=str(path))
        manifest[kind]={'scene':str(path),'output':scene.render.filepath,'resolution':[640,480],'samples':256,'denoise':True,'rendered':False,'scope':'Labeled ordinary physical arrangement preview with source markers, noncoherent area/background/ground illumination. Illustration only; separate saved detector crop acceptance files remain unchanged.'}
    (root/'publication_preview_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))

if __name__=='__main__':main()
