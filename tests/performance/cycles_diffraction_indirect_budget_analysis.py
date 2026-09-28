# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Independent-seed noise pilot; not an absolute radiance or convergence reference."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directories',nargs='+',type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
if a.output.exists():raise RuntimeError('Refusing to overwrite analysis')
rows=[];shared=None
for directory in a.directories:
 r=json.loads((directory/'report.json').read_text());assert r['status']=='completed'
 measured=[x for x in r['runs'] if not x['warmup']]
 assert len(measured)>=2 and len({x['seed'] for x in measured})==len(measured)
 identity={k:r[k] for k in ['source_scene_sha256','binary_sha256','script_sha256','transport','color_space']}
 identity['settings']={k:v for k,v in r['settings'].items() if k!='light_paths'}
 identity['seeds']=[x['seed'] for x in measured]
 if shared is None:shared=identity
 assert identity==shared
 images=[]
 for run in measured:
  for record in run['artifacts'].values():
   assert hashlib.sha256((directory/record['file']).read_bytes()).hexdigest()==record['sha256']
  image=np.load(directory/run['artifacts']['npy']['file'],allow_pickle=False).astype(np.float64)
  assert image.shape==(r['settings']['resolution'][1],r['settings']['resolution'][0],3)
  assert np.isfinite(image).all()
  assert np.allclose(image.mean(axis=(0,1)),run['mean_rgb'],rtol=0,atol=1e-12)
  images.append(image)
 stack=np.stack(images)
 variance=float(np.var(stack,axis=0,ddof=1).mean())
 luma=stack@np.array([.2126,.7152,.0722])
 luma_variance=float(np.var(luma,axis=0,ddof=1).mean())
 time=r['median_measured_seconds']
 assert time==statistics.median(run['render_seconds'] for run in measured)
 seed_means=stack.mean(axis=(1,2))
 rows.append(dict(directory=str(directory.resolve()),light_paths=r['settings']['light_paths'],
  measured_seeds=identity['seeds'],median_seconds=time,mean_rgb=stack.mean(axis=(0,1,2)).tolist(),
  mean_rgb_standard_error=(seed_means.std(axis=0,ddof=1)/np.sqrt(len(measured))).tolist(),
  single_seed_rgb_variance=variance,single_seed_luminance_variance=luma_variance,
  rgb_variance_time=variance*time,luminance_variance_time=luma_variance*time,
  report_sha256=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()))
result=dict(scope=__doc__,normalized=False,shared=shared,rows=rows,
 analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 limitations=['Independent-seed variation measures sampling noise, not bias or physical accuracy.',
 'Small seed counts and spatially correlated light splats limit statistical certainty.',
 'Variance-time is an empirical pilot metric, not a proof of equal-error performance.',
 'Render-call times include EXR writing; the first warm-up is excluded.'])
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(rows,indent=2))
