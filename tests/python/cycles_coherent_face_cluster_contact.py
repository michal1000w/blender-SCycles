#!/usr/bin/env python3
"""Read actual face-cluster acceptance EXRs; fixed exposures, no rendering."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'build/tests/python'
OUTPUT = BASE / 'cycles_coherent_face_cluster_contact_v32'
CASES = (
    ('folded / phase 0', 'cycles_coherent_face_cluster_folded_first_receiver_render_v32_retry', 'phase_0', -2),
    ('folded / phase pi', 'cycles_coherent_face_cluster_folded_first_receiver_render_v32_retry', 'phase_pi', -2),
    ('finite hole / phase 0', 'cycles_coherent_face_cluster_gap_render_v32', 'phase_0', -2),
    ('wrong-face blocker / phase 0', 'cycles_coherent_face_cluster_wrong_face_render_v32', 'phase_0', -2),
    ('same-object slab / phase 0', 'cycles_coherent_face_cluster_slab_joined_render_v32', 'phase_0', -2),
    ('checker regression / phase 0', 'cycles_coherent_detector_diffuse_checker_regression_v32', 'phase_0', 3),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, help='Aggregate persistent-suite results.json')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text()) if args.suite else None
    entries = []
    if suite:
        labels = {
            'folded_phase0': 'folded / phase 0',
            'folded_phasepi': 'folded / phase pi',
            'gap_phase0': 'finite hole / phase 0',
            'wrong_face_phase0': 'wrong-face blocker / phase 0',
            'joined_slab_phase0': 'same-object slab / phase 0',
            'diffuse_checker_phase0': 'checker regression / phase 0',
        }
        if len(suite['cases']) != 6:
            raise RuntimeError('This fixed contact expects six planar regression cases')
        for name, entry in suite['cases'].items():
            if not entry.get('passed'):
                raise RuntimeError('Suite case did not pass: ' + name)
            entries.append((labels.get(name, name), Path(entry['render_path']),
                            entry['case'], 3 if 'checker' in name else -2,
                            Path(entry['worker_report']), entry['reference_npz'],
                            entry['reference_npz_sha256']))
    else:
        for label, folder, case, ev in CASES:
            path = BASE / folder / f'{case}.exr'
            run = json.loads(path.with_name('results.json').read_text())
            entries.append((label, path, case, ev,
                            path.with_name(f'{case}_worker.json'),
                            run['reference_npz'], run['reference_npz_sha256']))
    rows = []
    records = {}
    tiles = []
    for label, path, case, ev, worker_path, reference, reference_hash in entries:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or not np.isfinite(raw).all():
            raise RuntimeError(f'Missing/nonfinite EXR {path}')
        worker = json.loads(worker_path.read_text())
        if hashlib.sha256(Path(reference).read_bytes()).hexdigest() != reference_hash:
            raise RuntimeError('Reference hash mismatch: ' + str(reference))
        ref = np.load(reference)
        key = {'phase_0': 'phase_0_radiance', 'phase_pi': 'phase_pi_radiance'}[case]
        expected = ref[key]
        rgb = raw[::-1, :, :3][:, :, ::-1].astype(np.float64)
        actual = rgb if expected.ndim == 3 else rgb.mean(axis=2)
        roi = ref['roi_mask'].astype(bool)
        error = actual[roi] - expected[roi]
        mean = float(np.mean(actual[roi]))
        rmse = float(np.sqrt(np.mean(error*error)))
        if abs(mean - worker['metrics']['render_mean_radiance']) > 1e-8 or abs(
                rmse - worker['metrics']['absolute_rmse']) > 1e-8:
            raise RuntimeError(f'Independent raw EXR metric mismatch: {path}')
        exposed = np.maximum(raw[:, :, :3].astype(np.float64) * 2**ev, 0)
        srgb = np.where(exposed <= .0031308, 12.92 * exposed,
                        1.055 * exposed**(1/2.4) - .055)
        display = np.rint(np.clip(srgb, 0, 1)*255).astype(np.uint8)
        tile = np.full((386, 320, 3), 24, dtype=np.uint8)
        tile[40:360] = cv2.resize(display, (320, 320), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(tile, f'{ev:+} EV | raw RMSE {rmse:.2e}', (8, 377),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (230, 230, 230), 1, cv2.LINE_AA)
        tiles.append(tile)
        records[label] = {'exr': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'exposure_ev': ev, 'raw_mean': mean, 'raw_rmse': rmse,
                          'reference_sha256': reference_hash}
    rows = [np.concatenate(tiles[i:i+3], axis=1) for i in (0, 3)]
    footer = np.full((50, 960, 3), 24, dtype=np.uint8)
    cv2.putText(footer, 'ACTUAL Metal EXRs | face cases fixed -2 EV; checker fixed +3 EV | sRGB | no autoscale',
                (10, 29), cv2.FONT_HERSHEY_SIMPLEX, .5, (245, 245, 245), 1, cv2.LINE_AA)
    args.output.mkdir(parents=True, exist_ok=True)
    png = args.output / 'actual_planar_faces_fixed_exposure.png'
    cv2.imwrite(str(png), np.concatenate((*rows, footer), axis=0))
    (args.output / 'raw_metric_verification.json').write_text(json.dumps(
        {'actual_render_only': True, 'raw_metrics_match_workers': True, 'cases': records,
         'suite': str(args.suite.resolve()) if args.suite else None,
         'binary_sha256': suite['binary_sha256'] if suite else None},
        indent=2) + '\n')
    print(png)


if __name__ == '__main__':
    main()
