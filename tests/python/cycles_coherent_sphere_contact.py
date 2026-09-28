#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual native-sphere EXR contact; common fixed exposure, no rendering."""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--renders',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 ref=np.load(a.reference);roi=ref['roi_mask'].astype(bool);tiles=[];records={};ev=-1
 for name,key,label in (('phase_0','phase_0_radiance','phase 0'),('phase_pi','phase_pi_radiance','phase pi'),('incoherent_connector_control','incoherent_radiance','distinct groups')):
  path=a.renders/(name+'.exr');raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
  assert raw is not None and np.isfinite(raw).all(),path
  expected=ref[key];rgb=raw[::-1,:,:3][:,:,::-1].astype(np.float64);actual=rgb.mean(axis=-1);error=actual[roi]-expected[roi]
  mean=float(actual[roi].mean());rmse=float(np.sqrt(np.mean(error**2)));worker=json.loads((a.renders/(name+'_worker.json')).read_text())
  assert abs(mean-worker['metrics']['render_mean_radiance'])<1e-8 and abs(rmse-worker['metrics']['absolute_rmse'])<1e-8
  exposed=np.maximum(raw[:,:,:3].astype(np.float64)*2**ev,0);display=np.where(exposed<=.0031308,12.92*exposed,1.055*exposed**(1/2.4)-.055);image=np.rint(np.clip(display,0,1)*255).astype(np.uint8)
  tile=np.full((366,320,3),24,dtype=np.uint8);tile[32:352]=cv2.resize(image,(320,320),interpolation=cv2.INTER_AREA)
  cv2.putText(tile,'native sphere / '+label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
  cv2.putText(tile,'RMSE '+format(rmse,'.2e'),(8,363),cv2.FONT_HERSHEY_SIMPLEX,.4,(230,230,230),1,cv2.LINE_AA);tiles.append(tile)
  records[name]={'exr':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'raw_mean':mean,'raw_rmse':rmse,'raw_min_roi':float(actual[roi].min()),'raw_max_error':float(abs(error).max()),'fixed_exposure_ev':ev}
 footer=np.full((45,960,3),24,dtype=np.uint8);cv2.putText(footer,'ACTUAL Metal EXRs | same -1 EV + sRGB | no normalization | independent double reference',(8,27),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
 a.output.mkdir(parents=True,exist_ok=True);png=a.output/'actual_native_sphere_fixed_exposure.png';cv2.imwrite(str(png),np.concatenate((np.concatenate(tiles,axis=1),footer),axis=0))
 (a.output/'raw_metric_verification.json').write_text(json.dumps({'actual_render_only':True,'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'raw_metrics_match_workers':True,'cases':records},indent=2)+'\n');print(png.resolve())

if __name__=='__main__':main()
