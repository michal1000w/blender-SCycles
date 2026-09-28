# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Copy verified scenes with separate render destinations, preserving evidence."""
import argparse, hashlib, json, sys
from pathlib import Path
import bpy
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evidence',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--extra-scene',type=Path,action='append',default=[])
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output=a.output.resolve()
a.output.mkdir(parents=True,exist_ok=False)
sources={f.parent.name+'_'+f.stem:f for f in sorted(a.evidence.resolve().glob('*/*.blend'))}
for f in a.extra_scene:sources[f.stem]=f.resolve()
records={}
for name,source in sources.items():
    bpy.ops.wm.open_mainfile(filepath=str(source))
    s=bpy.context.scene
    assert s.render.engine=='CYCLES' and not s.cycles.use_adaptive_sampling
    settings=(s.cycles.samples,s.cycles.use_denoising,s.cycles.use_guiding,s.cycles.use_bidirectional_path_tracing)
    s.render.filepath='//renders/'+name+'/render.exr'
    tree=s.compositing_node_group
    if tree:
        for node in tree.nodes:
            if node.bl_idname=='CompositorNodeOutputFile':node.directory='//renders/'+name+'/passes'
    target=a.output/(name+'.blend')
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    bpy.ops.wm.open_mainfile(filepath=str(target))
    s=bpy.context.scene
    assert s.render.filepath=='//renders/'+name+'/render.exr'
    assert (s.cycles.samples,s.cycles.use_denoising,s.cycles.use_guiding,s.cycles.use_bidirectional_path_tracing)==settings
    records[name]={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'copy_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'settings_preserved':True}
(a.output/'manifest.json').write_text(json.dumps({'scope':'Editable copies; only output destinations changed; not new renders','scenes':records},indent=2)+'\n')
print('Exported and reopened',len(records),'scenes')
