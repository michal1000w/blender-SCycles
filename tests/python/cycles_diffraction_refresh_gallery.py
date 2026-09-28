#!/usr/bin/env python3
"""Refresh saved presentation scenes sequentially in one Blender process.

No scene art/material changes. Rendering requires --render; ordinary Python
--plan-only lists exact inputs without GPU work. Raw/OIDN EXRs and editable
copies are retained. These are appearance checks, not physical certification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('cd_dvd', 'glass', 'glossy', 'metallic', 'principled_film', 'refraction',
         'covered_disc', 'indirect', 'thin_wall', 'ashikhmin')
TEMPLATES = {name: ROOT / 'tests/output/diffraction/presentation_multiggx_v3/scenes' / (name + '.blend') for name in NAMES}
TEMPLATES.update({
    'two_sided_glass': ROOT / 'build/tests/python/cycles_diffraction_glass_multiggx_presentation_v27_denoised_512/glass.blend',
    'generalized_dispersion': ROOT / 'build/tests/python/cycles_diffraction_principled_dispersion_presentation_v28_denoised_512/principled_dispersion.blend',
    'coated_generalized_furnace': ROOT / 'build/tests/python/cycles_diffraction_principled_multiggx_coated_furnace_v29/two_sided_front.blend',
})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    for path in TEMPLATES.values():
        if not path.is_file():
            raise RuntimeError('Missing existing editable template: ' + str(path))
    original = json.loads((ROOT / 'tests/output/diffraction/presentation_multiggx_v3/manifest.json').read_text())['runs']
    counts = {name: entry['report']['samples'] for name, entry in original.items()}
    counts.update(two_sided_glass=512, generalized_dispersion=512, coated_generalized_furnace=256)
    plan = {name: {'template': str(path), 'template_sha256': sha(path),
                  'samples': counts[name], 'transport': 'bdpt' if name == 'indirect' else 'pt'}
            for name, path in TEMPLATES.items()}
    if args.plan_only:
        print(json.dumps({'scope': __doc__, 'runs': plan, 'fixed_counts': 'unchanged from original validated presentation',
                          'rendered': False}, indent=2))
        return
    if not args.render:
        parser.error('Choose --plan-only or explicit --render')
    import bpy
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    manifest = {'binary': bpy.app.binary_path, 'sha256': sha(Path(bpy.app.binary_path)),
                'scope': 'Current-binary fixed-sample Metal appearance refresh; OIDN previews, raw EXRs retained; not an energy/reciprocity benchmark.',
                'settings': {'samples': 'per-case original fixed count', 'seed': 11, 'adaptive': False,
                             'denoiser': 'OPENIMAGEDENOISE', 'geometry_camera_resolution': 'unchanged from saved template'},
                'runs': {}}
    if args.resume:
        prior = json.loads((output / 'manifest.json').read_text())
        assert prior['sha256'] == manifest['sha256'], 'Resume requires the identical executable'
        manifest = prior
    worker = ROOT / 'tests/python/cycles_diffraction_denoising_delivery.py'
    worker_hash = sha(worker)
    for name, entry in plan.items():
        folder = output / name
        if name in manifest['runs']:
            saved = manifest['runs'][name]
            report = json.loads((folder / 'report.json').read_text())
            assert saved['exit_code'] == 0 and report['sha256'] == manifest['sha256']
            assert saved['template_sha256'] == entry['template_sha256']
            assert report['samples'] == entry['samples'] and report['transport'] == entry['transport']
            assert report['adaptive_sampling'] is False
            assert all(item['finite'] and Path(item['file']).is_file() for item in report['passes'].values())
            print('Resume: preserving completed case', name, flush=True)
            continue
        if folder.exists():
            failed = output / (name + '_failed_setup_process1')
            assert not failed.exists(), 'Failed setup archive already exists'
            folder.rename(failed)
        # Existing worker retains raw passes and editable scene, checks finite
        # pixels, and saves PNGs with the template's existing display transform.
        sys.argv = [str(worker), '--', '--template', entry['template'], '--output',
                    str(output / name), '--samples', str(entry['samples']), '--transport', entry['transport']]
        runpy.run_path(str(worker), run_name='__main__')
        report = json.loads((output / name / 'report.json').read_text())
        assert report['sha256'] == manifest['sha256']
        assert report['adaptive_sampling'] is False
        assert all(item['finite'] for item in report['passes'].values())
        manifest['runs'][name] = {**entry, 'exit_code': 0, 'worker_sha256': worker_hash, 'report': report}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'completed': len(manifest['runs']), 'manifest': str(output / 'manifest.json')}))


if __name__ == '__main__':
    main()
