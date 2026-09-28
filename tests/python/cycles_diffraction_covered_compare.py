# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare preserved covered-grating EXRs against the no-culling control."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
if a.output.exists():raise RuntimeError('Refusing to overwrite an existing comparison')
def read(path):
 image=bpy.data.images.load(str(path.resolve()),check_existing=False)
 return tuple(image.size),np.asarray(image.pixels[:],dtype=np.float64).reshape(-1,4)[:,:3]
shape,ref=read(a.reference);candidate_shape,value=read(a.candidate)
assert shape==candidate_shape and np.isfinite(ref).all() and np.isfinite(value).all()
difference=value-ref
relative_l1=float(np.abs(difference).sum()/max(np.abs(ref).sum(),1e-20))
mean_error=float(np.max(np.abs(difference.mean(axis=0))))
report=dict(scope=__doc__,size=shape,relative_l1=relative_l1,
 max_pixel_absolute_error=float(np.max(np.abs(difference))),
 rmse=float(np.sqrt(np.mean(difference*difference))),max_mean_rgb_error=mean_error,
 normalized=False,passed=relative_l1<1e-4 and mean_error<1e-6,
 reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
 candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest())
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not report['passed']:raise RuntimeError('Covered-grating regression exceeds tolerance')
