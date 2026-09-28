# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Check diffraction quality defaults, serialization and legacy scene loading."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--legacy-scene',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
a.output.mkdir(parents=True,exist_ok=False)
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
legacy_hash=digest(a.legacy_scene)
bpy.ops.wm.read_factory_settings(use_empty=True)
material=bpy.data.materials.new('Diffraction quality serialization')
material.use_nodes=True
nodes=material.node_tree.nodes
fast=nodes.new('ShaderNodeBsdfDiffraction');fast.name='Fast default'
assert fast.quality=='FAST'
realistic=nodes.new('ShaderNodeBsdfDiffraction');realistic.name='Realistic explicit'
realistic.quality='REALISTIC'
material.use_fake_user=True
saved=a.output/'quality_roundtrip.blend'
bpy.ops.wm.save_as_mainfile(filepath=str(saved.resolve()))
bpy.ops.wm.open_mainfile(filepath=str(saved.resolve()))
nodes=bpy.data.materials['Diffraction quality serialization'].node_tree.nodes
assert nodes['Fast default'].quality=='FAST'
assert nodes['Realistic explicit'].quality=='REALISTIC'
bpy.ops.wm.open_mainfile(filepath=str(a.legacy_scene.resolve()))
legacy_nodes=[n for m in bpy.data.materials if m.use_nodes for n in m.node_tree.nodes
              if n.bl_idname=='ShaderNodeBsdfDiffraction']
assert legacy_nodes, 'The legacy fixture must contain physical diffraction nodes'
assert all(n.quality=='REALISTIC' for n in legacy_nodes)
assert digest(a.legacy_scene)==legacy_hash
report=dict(status='passed',new_node_default='FAST',roundtrip_modes=['FAST','REALISTIC'],
            legacy_node_count=len(legacy_nodes),legacy_loaded_mode='REALISTIC',
            legacy_scene_sha256=legacy_hash,roundtrip_scene_sha256=digest(saved),
            binary_sha256=digest(Path(bpy.app.binary_path)),script_sha256=digest(Path(__file__)),
            scope='DNA/RNA and file compatibility checks; no rendering or performance claim')
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
