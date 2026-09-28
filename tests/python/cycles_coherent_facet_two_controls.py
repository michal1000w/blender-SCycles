#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prospective test-only quadrature and shared-Gaussian phase noise audit."""
import json,math,sys
from pathlib import Path
import numpy as np
from cycles_coherent_facet_two_reference import inventory as paths,pair_intensity


def main():
    root=Path(sys.argv[1]);out={}
    for layout in ['corner_xy']:
        folder=root/layout;manifest=json.loads((folder/'manifest.json').read_text());data=np.load(folder/'physical_geometry.npz');triangles=data['triangles'];sources=data['source_positions'];powers=data['source_powers'];receiver=data['receiver'];reference=np.load(folder/'virtual_source_reference.npz');roi=reference['roi_mask'];report={}
        mapping=json.loads((folder/'saved_camera_mapping.json').read_text());matrix=np.array(mapping['matrix_world']);right=matrix[:3,0];up=matrix[:3,1];width=mapping['ortho_scale'];offsets=(np.arange(4)+.5)/4;cached={}
        for row,col in zip(*np.where(roi)):
            for ox in offsets:
                for oy in offsets:
                    point=receiver+((col+ox)/16-.5)*width*right+((row+oy)/16-.5)*width*up;cached[row,col,ox,oy]=[paths(s,point,triangles) for s in sources]
        for name,entry in manifest['variants'].items():
            phases=entry['source_phases_rad'];groups=entry['groups'];inventories=[paths(s,receiver,triangles) for s in sources];value,terms=pair_intensity(inventories,powers,phases,groups,coherence_length=1.)
            z,w=np.polynomial.hermite.hermgauss(80);variance=0.;quad_mean=0.
            for group in set(groups):
                selected=[p for p in terms if p[3]==group];anchor=selected[0][1];fields=np.array([p[0].reshape(-1) for p in selected]);lengths=np.array([p[1]-anchor for p in selected]);angles=np.array([math.tau*math.remainder(d/550e-9,1)+p[2] for d,p in zip(lengths,selected)])
                sampled=np.exp(1j*(angles[None]+np.sqrt(2)*z[:,None]*lengths[None]))@fields;intensity=np.sum(np.abs(sampled)**2,axis=1);mean=np.dot(w,intensity)/math.sqrt(math.pi);variance+=np.dot(w,(intensity-mean)**2)/math.sqrt(math.pi);quad_mean+=mean
            assert abs(value-quad_mean)<1e-9
            high=np.zeros((16,16))
            for row,col in zip(*np.where(roi)):
                for ox in offsets:
                    for oy in offsets:
                        high[row,col]+=pair_intensity(cached[row,col,ox,oy],powers,phases,groups,coherence_length=1.)[0]/16
            key={'phase_0':'phase_0_radiance','phase_pi':'phase_pi_radiance','incoherent_connector_control':'incoherent_radiance'}[name];error=high[roi]-reference[key][roi]
            report[name]={'center_exact_pair':value,'center_gaussian_phase_variance':float(variance),'center_MC_sigma_128':math.sqrt(variance/128),'center_MC_sigma_1024':math.sqrt(variance/1024),'4x4_vs_2x2_quadrature_rmse':float(np.sqrt(np.mean(error**2))),'4x4_vs_2x2_quadrature_mean_error':float(abs(error.mean()))}
        out[layout]=report
    (root/'pre_render_numerical_audit.json').write_text(json.dumps({'scope':'Prospective quadrature and exact shared-Gaussian phase moments; no actual render inputs or gate fitting','layouts':out},indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
