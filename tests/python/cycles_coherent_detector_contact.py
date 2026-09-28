#!/usr/bin/env python3
"""Read six detector checker EXRs, verify raw metrics, export common-scale contact."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'build/tests/python'
OUTPUT = BASE / 'cycles_coherent_detector_contact_v30_final'
EV = 3.0


def main():
    OUTPUT.mkdir(exist_ok=True)
    inputs = {}
    rows = []
    for material in ('diffuse', 'principled'):
        folder = BASE / f'cycles_coherent_detector_{material}_checker_render_v30_final'
        run = json.loads((folder / 'results.json').read_text())
        ref = np.load(run['reference_npz'])
        roi = ref['roi_mask'].astype(bool)
        tiles = []
        for case, label in (('phase_0', 'phase 0'), ('phase_pi', 'phase pi'),
                            ('incoherent_connector_control', 'distinct source groups')):
            path = folder / f'{case}.exr'
            raw_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw_bgr is None or not np.isfinite(raw_bgr).all():
                raise RuntimeError(f'Missing/nonfinite EXR {path}')
            # OpenCV is top-down BGR; Blender worker reads bottom-up RGB.
            rgb = raw_bgr[::-1, :, :3][:, :, ::-1].astype(np.float64)
            key = {'phase_0': 'phase_0_radiance', 'phase_pi': 'phase_pi_radiance',
                   'incoherent_connector_control': 'incoherent_radiance'}[case]
            expected = ref[key]
            error = rgb[roi] - expected[roi]
            rmse = np.sqrt(np.mean(error * error, axis=0))
            mean = float(np.mean(rgb[roi]))
            black = roi & np.all(expected == 0.0, axis=2)
            blackmax = float(np.max(np.abs(rgb[black])))
            worker = json.loads((folder / f'{case}_worker.json').read_text())['metrics']
            if abs(mean - worker['render_mean_radiance']) > 1e-9 or np.max(
                np.abs(rmse - np.array(worker['rgb_rmse']))) > 1e-9:
                raise RuntimeError(f'EXR metric mismatch {material}/{case}')
            if blackmax != worker['black_max_absolute_radiance']:
                raise RuntimeError('Black metric mismatch')
            exposed = np.maximum(raw_bgr[:, :, :3].astype(np.float64) * 2**EV, 0.0)
            srgb = np.where(exposed <= .0031308, 12.92 * exposed,
                            1.055 * exposed**(1/2.4) - .055)
            image = np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)
            tile = np.full((382, 320, 3), 24, dtype=np.uint8)
            tile[40:360] = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f'{material} / {label}', (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(tile, f'RGB RMSE max {max(rmse):.2e}; black {blackmax:g}', (8, 376),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (230, 230, 230), 1, cv2.LINE_AA)
            tiles.append(tile)
            inputs[f'{material}/{case}'] = {'exr': str(path),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'rgb_rmse': rmse.tolist(), 'render_mean_radiance': mean,
                'black_pixels': int(np.count_nonzero(black)), 'black_max': blackmax}
        rows.append(np.concatenate(tiles, axis=1))
    footer = np.full((42, 960, 3), 24, dtype=np.uint8)
    cv2.putText(footer, 'ACTUAL Metal BDPT EXRs | fixed +3 EV + sRGB | 128 samples | no per-image normalization',
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .49, (245, 245, 245), 1, cv2.LINE_AA)
    png = OUTPUT / 'actual_checker_fixed_ev+3.png'
    cv2.imwrite(str(png), np.concatenate((*rows, footer), axis=0))
    (OUTPUT / 'raw_metric_verification.json').write_text(json.dumps(
        {'exposure_ev': EV, 'display': 'sRGB', 'actual_render_only': True,
         'raw_metrics_match_workers': True, 'inputs': inputs}, indent=2) + '\n')
    print(png)


if __name__ == '__main__':
    main()
