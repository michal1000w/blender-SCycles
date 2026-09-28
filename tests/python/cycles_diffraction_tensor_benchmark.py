# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Measure repeated Metal renders after preparation, retaining warm-up timings.

The mirror is a one-bounce workload baseline, not a physically equivalent
material. Run with no other GPU jobs; process time is not GPU-only time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import bpy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--material',choices=['diffraction','mirror','undiffracted','flat-grating','constant-index'],required=True)
p.add_argument('--samples',type=int,default=1024)
p.add_argument('--runs',type=int,default=3)
p.add_argument('--transport',choices=['pt','bdpt','guided','bdpt_guided'],default='pt')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
if a.samples<=0 or a.runs<3:p.error('Use positive samples and at least three measured runs')
if a.output.exists():raise RuntimeError('Refusing to overwrite benchmark')
a.output.parent.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(a.scene.resolve()))
scene=bpy.context.scene
scene.render.use_persistent_data=True
scene.cycles.samples=a.samples
scene.cycles.use_adaptive_sampling=False
scene.cycles.use_denoising=False
scene.cycles.time_limit=0
scene.cycles.use_layer_samples='IGNORE'
scene.cycles.use_bidirectional_path_tracing=a.transport in ('bdpt','bdpt_guided')
scene.cycles.use_guiding=a.transport in ('guided','bdpt_guided')
preferences=bpy.context.preferences.addons['cycles'].preferences
preferences.compute_device_type='METAL'
preferences.get_devices()
for device in preferences.devices:device.use=device.type=='METAL'
if not any(d.use for d in preferences.devices):raise RuntimeError('No Metal device')
scene.cycles.device='GPU'
disabled_diffraction_nodes=[]
cleared_optical_tables=[]
if a.material=='constant-index':
    for material in bpy.data.materials:
        if not material.use_nodes:continue
        for node in material.node_tree.nodes:
            if node.bl_idname!='ShaderNodeBsdfDiffraction' or node.optical_constants is None:continue
            if node.quality!='FAST':raise RuntimeError('This table benchmark requires Fast nodes')
            cleared_optical_tables.append(dict(material=material.name,node=node.name,
                text_sha256=hashlib.sha256(node.optical_constants.as_string().encode()).hexdigest(),
                fallback_ridge_index=[node.ridge_ior,node.ridge_extinction],
                fallback_substrate_index=[node.substrate_ior,node.substrate_extinction]))
            node.optical_constants=None
    if not cleared_optical_tables:raise RuntimeError('No optical tables to disable')
if a.material=='flat-grating':
    # Keep the integrated node and its optical constants, graph and tangent.
    # A zero relief has only order zero. This changes paths and appearance;
    # it measures the workload cost, not time to an equivalent image.
    for material in bpy.data.materials:
        if not material.use_nodes:continue
        for node in material.node_tree.nodes:
            if node.bl_idname!='ShaderNodeBsdfDiffraction':continue
            if node.quality!='FAST':
                raise RuntimeError('Flat control requires integrated Fast nodes')
            if node.depth>0:
                disabled_diffraction_nodes.append(dict(material=material.name,node=node.name,
                                                       original_depth_nm=node.depth))
                node.depth=0
    if not disabled_diffraction_nodes:
        raise RuntimeError('Flat control requires at least one non-flat Fast grating')
if a.material=='undiffracted':
    # Matched workload control: retain colour, roughness, normals and links.
    # Removing diffraction changes paths, so this is not equal-image timing.
    for material in bpy.data.materials:
        if not material.use_nodes:continue
        for node in material.node_tree.nodes:
            if node.bl_idname!='ShaderNodeBsdfAnisotropic':continue
            socket=node.inputs.get('Diffraction Weight')
            if socket is None:continue
            if socket.is_linked:
                raise RuntimeError('Linked diffraction weights require an explicit baseline scene')
            if socket.default_value>0:
                disabled_diffraction_nodes.append(dict(material=material.name,node=node.name,
                                                       original_weight=socket.default_value))
                socket.default_value=0
    if not disabled_diffraction_nodes:
        raise RuntimeError('Undiffracted control requires at least one active scalar diffraction node')
if a.material=='mirror':
    found=0
    for material in bpy.data.materials:
        if not material.use_nodes:continue
        nodes=material.node_tree.nodes
        for node in list(nodes):
            if node.bl_idname!='ShaderNodeBsdfDiffraction':continue
            outputs=[link.to_socket for link in node.outputs['BSDF'].links]
            mirror=nodes.new('ShaderNodeBsdfAnisotropic')
            mirror.inputs['Color'].default_value=(1,1,1,1)
            mirror.inputs['Roughness'].default_value=0
            for target in outputs:material.node_tree.links.new(mirror.outputs[0],target)
            nodes.remove(node);found+=1
    if found!=1:raise RuntimeError('Expected one grating material')
scene_path=a.output.with_suffix('.blend')
bpy.ops.wm.save_as_mainfile(filepath=str(scene_path))
def sampling_settings():
    return dict(samples=scene.cycles.samples,
                adaptive_sampling=scene.cycles.use_adaptive_sampling,
                denoising=scene.cycles.use_denoising,
                time_limit=scene.cycles.time_limit,
                use_layer_samples=scene.cycles.use_layer_samples,
                bidirectional=scene.cycles.use_bidirectional_path_tracing,
                guiding=scene.cycles.use_guiding,
                resolution_percentage=scene.render.resolution_percentage)

settings=sampling_settings()
if settings['adaptive_sampling'] or settings['denoising']:
    raise RuntimeError('Benchmark requires fixed samples without denoising')
if settings['time_limit']!=0 or settings['use_layer_samples']!='IGNORE':
    raise RuntimeError('Benchmark requires unrestricted scene sample counts')
report=dict(benchmark_schema=2,sampling_settings=settings,
            blender_binary_sha256=hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            material=a.material,transport=a.transport,samples=a.samples,resolution=[scene.render.resolution_x,scene.render.resolution_y],
            devices=[d.name for d in preferences.devices if d.use],
            input_scene_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),
            benchmark_scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            disabled_diffraction_nodes=disabled_diffraction_nodes,
            cleared_optical_tables=cleared_optical_tables,
            scope=__doc__,runs=[],status='running')
for run in range(a.runs+1):
    scene.cycles.seed=11+run
    if sampling_settings()!=settings:
        raise RuntimeError('Benchmark sampling settings changed')
    started=time.perf_counter()
    bpy.ops.render.render(write_still=False)
    elapsed=time.perf_counter()-started
    if sampling_settings()!=settings:
        raise RuntimeError('Render changed benchmark sampling settings')
    report['runs'].append(dict(index=run,warmup=run==0,seconds=elapsed,seed=scene.cycles.seed,
                              sampling_settings=sampling_settings()))
    a.output.write_text(json.dumps(report,indent=2)+'\n')
measured=[r['seconds'] for r in report['runs'] if not r['warmup']]
report.update(status='completed',median_seconds=statistics.median(measured),
              minimum_seconds=min(measured),maximum_seconds=max(measured))
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
