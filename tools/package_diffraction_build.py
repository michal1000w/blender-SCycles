#!/usr/bin/env python3
"""Package the local diffraction build with matching kernel resources."""
import argparse, hashlib, json, platform, shlex, shutil, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base-app',type=Path,required=True);p.add_argument('--output-app',type=Path,required=True);a=p.parse_args()
r=Path(__file__).resolve().parents[1];build=r/'build/macos_arm64_Release';out=a.output_app.resolve()
assert not out.exists(), 'Refuse to overwrite an existing delivery'
out.parent.mkdir(parents=True,exist_ok=True)
if platform.system() == 'Darwin':
 # APFS clones preserve immutable earlier packages without duplicating the
 # nearly 1 GB base app. They are copy-on-write, unlike hard links.
 subprocess.run(['cp', '-cR', str(a.base_app.resolve()), str(out)], check=True)
else:
 shutil.copytree(a.base_app.resolve(),out,symlinks=True)
c=out/'Contents/Resources/5.3/scripts/addons_core/cycles';pairs=[]
def copy(src,dst):
 if platform.system() == 'Darwin' and src.stat().st_size >= 1024*1024:
  # Fresh clone then atomic replacement preserves source/base bytes and avoids
  # allocating another binary/metallib copy on a nearly full APFS volume.
  temporary=dst.with_name(dst.name+'.apfs-clone-temp')
  subprocess.run(['cp','-c','-p',str(src),str(temporary)],check=True)
  temporary.replace(dst)
 else:
  shutil.copy2(src,dst)
 pairs.append((src,dst))
copy(build/'bin/Blender.app/Contents/MacOS/Blender',out/'Contents/MacOS/Blender')
for name in ['kernel','util']:
 dst=c/'source'/name
 if dst.exists():shutil.rmtree(dst)
 shutil.copytree(r/'intern/cycles'/name,dst)
 for src in (r/'intern/cycles'/name).rglob('*'):
  if src.is_file():pairs.append((src,dst/src.relative_to(r/'intern/cycles'/name)))
# Python property/UI registration must match the new binary as well.
for src in (r/'intern/cycles/blender/addon').rglob('*.py'):
 dst=c/src.relative_to(r/'intern/cycles/blender/addon')
 dst.parent.mkdir(parents=True,exist_ok=True)
 copy(src,dst)
for src in (build/'intern/cycles/kernel/osl/shaders').glob('*.oso'):copy(src,c/'shader'/src.name)
libs=list((build/'intern/cycles/kernel/device/metal').glob('*.metallib'));assert len(libs)==3
for src in libs:copy(src,c/'lib'/src.name)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
manifest=[]
for src,dst in pairs:
 h=sha(src);assert h==sha(dst)
 manifest.append({'source':str(src),'package_path':str(dst.relative_to(out)),'sha256':h})
(out.parent/'resource_manifest.json').write_text(json.dumps(manifest,indent=2))
launcher=out.parent/'Launch Blender.command'
launcher.write_text('''#!/bin/sh
set -eu
unset BLENDER_SYSTEM_RESOURCES DYLD_LIBRARY_PATH CYCLES_KERNEL_PATH CYCLES_SHADER_PATH
package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$package_dir"/''' + shlex.quote(out.name) + '/Contents/MacOS/Blender "$@"\n')
launcher.chmod(0o755)
print(out,flush=True)
