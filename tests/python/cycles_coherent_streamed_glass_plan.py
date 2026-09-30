#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent references, prospective gates and render plans for the streamed
closed-convex Glass fixtures. Runs before any render; pure Python.

  python3 cycles_coherent_streamed_glass_plan.py FIXTURE_DIR OUTPUT_DIR

For every saved variant (geometry.json written by the Blender author script):
* reference radiance by 4x4 midpoint quadrature per pixel (independent oracle);
* 2x2 quadrature to bound the quadrature error;
* a Monte Carlo noise model for the fixed 128 samples per pixel: within-pixel
  radiance variance and, for finite coherence, the variance of the shared
  per-sample Gaussian phase estimator (Gauss-Hermite over the Gaussian);
* omission controls (drop TT, exterior Glass R, two-event reflections, or set
  the Glass IOR to 1) that the gates must be able to detect.
Gates are fixed from these numbers only, then plan.json lists the render jobs
for cycles_coherent_mirror_render_worker.py (Metal PT/BDPT/guiding; the CPU
plan is derived with cycles_coherent_cpu_plan.py).
"""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cycles_coherent_streamed_glass_reference as oracle

SAMPLES = 128  # Fixed by the render worker.
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimator_moments(scene, point, max_events, nodes=48):
    """Mean and second moment of the per-sample streamed field estimator.

    The renderer shares one standard normal Z per group and sample and weights
    each path by exp(i Z L / Lc). Expectation over Z gives the Gaussian mutual
    coherence; the second moment gives the added Monte Carlo variance."""
    paths = []
    for index, source in enumerate(scene.sources):
        for p in oracle.detector_paths(scene, source, point, max_events):
            paths.append(p)
    z, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    groups = sorted({p['group'] for p in paths})
    mean = 0.0
    second = 0.0
    # Groups use independent Gaussians; X = sum_g X_g.
    group_mean = {}
    group_second = {}
    for g in groups:
        members = [p for p in paths if p['group'] == g]
        values = []
        for zi in z:
            total = 0.0
            for m in range(3):
                field = np.zeros(3, dtype=complex)
                for p in members:
                    angle = (2 * math.pi * p['L'] / p['wavelength'] + p['phase'] +
                             zi * p['L'] / p['coherence_length'])
                    field += p['amplitude'] * p['fields'][m] * np.exp(1j * angle)
                total += float(np.vdot(field, field).real)
            values.append(total * scene.albedo / math.pi)
        values = np.array(values)
        group_mean[g] = float(np.dot(w, values))
        group_second[g] = float(np.dot(w, values * values))
    mean = sum(group_mean.values())
    variance = sum(group_second[g] - group_mean[g] ** 2 for g in groups)
    second = variance + mean * mean
    return mean, second


_WORKER_SCENE = {}
TIME_NODES = 6  # Gauss-Legendre nodes over the shutter for moving objects.


def time_scenes(data):
    """(weight, geometry) pairs over the shutter. Rigid motion: each moving
    object's triangles are translated linearly from its shutter-open to its
    shutter-close offset, as Cycles interpolates a pure translation."""
    motion = data.get('motion')
    if not motion:
        return [(1.0, data)]
    x, w = np.polynomial.legendre.leggauss(TIME_NODES)
    result = []
    for xi, wi in zip(x, w):
        t = 0.5 * (xi + 1.0)
        moved = copy.deepcopy(data)
        moved.pop('motion')
        moved.pop('route_keys', None)
        for obj in moved['objects']:
            if obj['name'] in motion:
                start = np.asarray(motion[obj['name']]['offset_open'])
                end = np.asarray(motion[obj['name']]['offset_close'])
                offset = start + t * (end - start)
                obj['triangles'] = (np.asarray(obj['triangles']) + offset).tolist()
        result.append((0.5 * wi, moved))
    return result


def _row(arguments):
    data, row, order, omit, moments = arguments
    scenes = []
    for weight, geometry in time_scenes(data):
        key = json.dumps(geometry, sort_keys=True)
        if key not in _WORKER_SCENE:
            _WORKER_SCENE[key] = oracle.Scene(geometry)
        scenes.append((weight, _WORKER_SCENE[key]))
    camera = data['camera']
    means, seconds = [], []
    for col in range(int(camera['resolution'])):
        values, second = [], []
        for x in oracle.pixel_points(camera, row, col, order):
            m = s = 0.0
            for weight, scene in scenes:
                if moments:
                    mi, si = estimator_moments(scene, x, data['max_events'])
                else:
                    mi = oracle.radiance(scene, x, data['max_events'], omit)[0]
                    si = mi * mi
                m += weight * mi
                s += weight * si  # Second moment over time and position.
            values.append(m)
            second.append(s)
        means.append(np.mean(values))
        seconds.append(np.mean(second))
    return means, seconds


def reference_image(data, order, omit=(), moments=False):
    """Pixel means and within-pixel estimator variance, rows in parallel."""
    import multiprocessing
    res = int(data['camera']['resolution'])
    with multiprocessing.Pool() as pool:
        rows = pool.map(_row, [(data, row, order, tuple(omit), moments) for row in range(res)])
    mean = np.array([r[0] for r in rows])
    second = np.array([r[1] for r in rows])
    return mean, second - mean * mean


def with_ior(data, ior):
    changed = copy.deepcopy(data)
    changed.pop('route_keys', None)  # Different optics: prefilter again.
    for obj in changed['objects']:
        if obj['kind'] == 'glass':
            obj['ior'] = ior
    return changed


def analyse(geometry_path):
    data = json.loads(Path(geometry_path).read_text())
    scene = oracle.Scene(data)
    if not data.get('motion'):
        data['route_keys'] = scene.route_keys()  # Prefilter once, reuse in workers.
    camera = data['camera']
    events = data['max_events']
    finite = any(s['coherence_length_m'] < 1.0 for s in data['sources'])
    ref4, var4 = reference_image(data, 4, moments=finite)
    ref2, _ = reference_image(data, 2)
    center = np.asarray(camera['detector_center'], dtype=float)
    _, paths = oracle.radiance(scene, center, events)
    families = sorted({p['family'] for p in paths})
    controls = {}
    for label, omit in [('omit_TT', ('TT',)), ('omit_glass_R', ('glass_R', 'RR_GM', 'RR_MG', 'RR_GG')),
                        ('omit_two_event_reflections', ('RR',)),
                        ('omit_three_and_four_event_routes', ('long',))]:
        present = any(p['family'] in omit or p['family'].split('_')[0] in omit for p in paths)
        if present:
            controls[label] = reference_image(data, 2, omit=omit)[0]
    if any(o['kind'] == 'glass' for o in data['objects']):
        controls['glass_IOR_1'] = reference_image(with_ior(data, 1.0), 2)[0]
    if data.get('motion'):
        static = copy.deepcopy(data)
        static.pop('motion')
        controls['ignore_motion'] = reference_image(static, 2)[0]
    return data, ref4, var4, ref2, families, controls, finite


def main():
    fixtures = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    res = None
    manifest = {'scope': 'Streamed closed convex Glass: independent oracle references and '
                         'prospective gates, fixed before rendering',
                'oracle': str(ROOT / 'tests/python/cycles_coherent_streamed_glass_reference.py'),
                'oracle_sha256': digest(ROOT / 'tests/python/cycles_coherent_streamed_glass_reference.py'),
                'plan_script_sha256': digest(__file__), 'samples': SAMPLES, 'layouts': {}}
    jobs = []
    for layout in sorted(p for p in fixtures.iterdir() if p.is_dir()):
        results = {}
        for geometry in sorted(layout.glob('*_geometry.json')):
            name = geometry.name[:-len('_geometry.json')]
            data, ref4, var4, ref2, families, controls, finite = analyse(geometry)
            res = ref4.shape[0]
            rows, cols = np.indices(ref4.shape)
            roi = (rows >= 2) & (rows < res - 2) & (cols >= 2) & (cols < res - 2)
            sigma_rmse = float(np.sqrt(np.mean(var4[roi] / SAMPLES)))
            quadrature = float(np.sqrt(np.mean((ref4[roi] - ref2[roi]) ** 2)))
            floor = 1e-6 * float(np.mean(np.abs(ref4[roi])))
            gate = {'absolute_mean_error': 4.0 * sigma_rmse / math.sqrt(roi.sum()) + quadrature + floor,
                    'absolute_rmse': 2.0 * sigma_rmse + 2.0 * quadrature + floor}
            control_rmse = {k: float(np.sqrt(np.mean((v[roi] - ref2[roi]) ** 2))) for k, v in controls.items()}
            control_mean = {k: float(abs(np.mean(v[roi]) - np.mean(ref2[roi]))) for k, v in controls.items()}
            results[name] = {'data': data, 'ref': ref4, 'gate': gate, 'roi': roi}
            manifest['layouts'].setdefault(layout.name, {})[name] = {
                'blend': data['blend'], 'blend_sha256': data['blend_sha256'],
                'geometry_sha256': digest(geometry), 'families_at_center': families,
                'finite_coherence_estimator_variance': finite,
                'mean_radiance': float(np.mean(ref4[roi])),
                'predicted_mc_rmse': sigma_rmse, 'quadrature_4x4_vs_2x2_rmse': quadrature,
                'gates_declared_before_render': gate,
                'omission_control_rmse': control_rmse, 'omission_control_mean_error': control_mean,
                'controls_detectable': {k: control_rmse[k] > gate['absolute_rmse'] or
                                        control_mean[k] > gate['absolute_mean_error']
                                        for k in control_rmse},
            }
        # Reference files, one per rendered variant, in the worker's key layout.
        for name, r in results.items():
            ref_path = output / layout.name / f'{name}_reference.npz'
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            base = results.get('phase_0', r)['ref']
            pi = results.get('phase_pi', r)['ref'] if name in ('phase_0', 'phase_pi') else r['ref']
            np.savez_compressed(ref_path, roi_mask=r['roi'],
                                phase_0_radiance=base if name == 'phase_pi' else r['ref'],
                                phase_pi_radiance=pi if name == 'phase_pi' else r['ref'],
                                incoherent_radiance=r['ref'])
            r['reference_path'] = ref_path
        for transport in ('pt', 'bdpt', 'pt-guiding'):
            if transport == 'bdpt' and 'scattering' in layout.name:
                # BDPT also renders light scattered by the medium (native, incoherent),
                # which this ballistic coherent reference intentionally excludes.
                continue
            for name, r in results.items():
                render = output / 'render' / f'{transport}_{layout.name}' / f'{name}.exr'
                render.parent.mkdir(parents=True, exist_ok=True)
                case = 'phase_pi' if name == 'phase_pi' else 'phase_0'
                arguments = ['--blend', r['data']['blend'], '--reference', str(r['reference_path']),
                             '--render', str(render), '--report', str(render.with_name(name + '_worker.json')),
                             '--case', case, '--require-specular', '--scene-budgets']
                if transport != 'bdpt':
                    arguments.append('--' + transport)
                if name == 'phase_pi':
                    arguments += ['--phase0-data', str(render.with_name('phase_0.npz'))]
                label = f'{transport}_{layout.name}_{name}'
                phase_gate = None
                if name == 'phase_pi':
                    g = results['phase_0']['gate']
                    phase_gate = {'absolute_phase_mean_error': g['absolute_mean_error'] + r['gate']['absolute_mean_error'],
                                  'phase_difference_rmse': g['absolute_rmse'] + r['gate']['absolute_rmse']}
                jobs.append({'case': label, 'arguments': arguments,
                             'stdout': str(output / (label + '_stdout.log')),
                             'stderr': str(output / (label + '_stderr.log')),
                             'gate': r['gate'], 'phase_gate': phase_gate,
                             'scene_sha256': r['data']['blend_sha256'],
                             'reference_sha256': digest(r['reference_path'])})
    plan = {'scope': manifest['scope'],
            'worker': str(ROOT / 'tests/python/cycles_coherent_mirror_render_worker.py'),
            'state': str(output / 'batch_state.json'), 'jobs': jobs}
    (output / 'plan.json').write_text(json.dumps(plan, indent=1) + '\n')
    (output / 'prospective_gates.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({layout: {name: {k: v[k] for k in ('families_at_center', 'mean_radiance', 'predicted_mc_rmse',
                                                          'quadrature_4x4_vs_2x2_rmse', 'gates_declared_before_render',
                                                          'omission_control_rmse', 'controls_detectable')}
                               for name, v in cases.items()} for layout, cases in manifest['layouts'].items()},
                     indent=1))


if __name__ == '__main__':
    main()
