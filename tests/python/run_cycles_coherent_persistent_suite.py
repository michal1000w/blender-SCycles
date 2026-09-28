#!/usr/bin/env python3
"""Run a declared cross-fixture suite with unchanged coherence workers/gates."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

from run_cycles_coherent_mirror_acceptance import ROOT, GATES, case_passed, render_environment, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    blender, output = args.blender.resolve(), args.output_dir.resolve()
    if not blender.is_file():
        parser.error('Missing Blender executable')
    plan = json.loads(args.plan.read_text())
    worker = Path(__file__).with_name('cycles_coherent_mirror_render_worker.py').resolve()
    batch = worker.with_name('cycles_coherent_mirror_batch_worker.py')
    labels, phase0_folders, jobs, entries = set(), set(), [], {}
    for item in plan['jobs']:
        label, case, folder = item['label'], item['case'], item['folder']
        transport = item.get('transport', 'bdpt')
        if transport not in ('bdpt', 'pt', 'pt-guiding'):
            parser.error('Unsupported per-job transport: ' + transport)
        if label in labels or Path(folder).name != folder:
            parser.error('Job labels must be unique and output folders simple names')
        labels.add(label)
        fixtures = (ROOT / item['fixtures']).resolve()
        scene, reference = fixtures / (case + '.blend'), fixtures / 'virtual_source_reference.npz'
        manifest_path = fixtures / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get('specular_connections') or not scene.is_file() or not reference.is_file():
            parser.error('Missing scene/reference or connector declaration for ' + label)
        if case == 'phase_pi' and folder not in phase0_folders:
            parser.error('phase_pi requires preceding phase_0 in the same output folder')
        if case == 'phase_0':
            phase0_folders.add(folder)
        destination = output / folder
        arguments = ['--blend', str(scene), '--reference', str(reference),
                     '--render', str(destination / (case + '.exr')),
                     '--report', str(destination / (case + '_worker.json')),
                     '--case', case, '--require-specular']
        if case == 'phase_pi':
            arguments += ['--phase0-data', str(destination / 'phase_0.npz')]
        if item.get('scene_budgets', False):
            arguments.append('--scene-budgets')
        if transport == 'pt':
            arguments.append('--pt')
        elif transport == 'pt-guiding':
            arguments.append('--pt-guiding')
        jobs.append({'case': label, 'arguments': arguments,
                     'stdout': str(output / (label + '_stdout.log')),
                     'stderr': str(output / (label + '_stderr.log'))})
        entries[label] = {'case': case, 'transport': transport, 'fixture_directory': str(fixtures),
                          'fixture_manifest_sha256': sha256(manifest_path),
                          'scene': str(scene), 'scene_sha256': sha256(scene),
                          'reference_npz': str(reference), 'reference_npz_sha256': sha256(reference),
                          'worker_arguments': arguments,
                          'render_path': str(destination / (case + '.exr')),
                          'worker_report': str(destination / (case + '_worker.json')),
                          'preserve_saved_bounce_budgets': item.get('scene_budgets', False)}
    result = {'binary': str(blender), 'binary_sha256': sha256(blender),
              'worker_script': str(worker), 'worker_script_sha256': sha256(worker),
              'batch_worker_sha256': sha256(batch), 'suite_runner_sha256': sha256(Path(__file__)),
              'plan_sha256': sha256(args.plan), 'gates': GATES,
              'fixed_settings': {'samples': 128, 'seed': 19, 'adaptive_sampling': False,
                                 'denoising': False, 'device': 'METAL',
                                 'transport': 'BDPT' if all(e['transport'] == 'bdpt' for e in entries.values()) else 'per-job; see cases'},
              'execution': 'sequential unchanged workers in one Blender process', 'cases': entries}
    if args.plan_only:
        print(json.dumps(result, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    for item in plan['jobs']:
        (output / item['folder']).mkdir(exist_ok=True)
    jobs_path, state_path = output / 'batch_jobs.json', output / 'batch_state.json'
    jobs_path.write_text(json.dumps({'worker': str(worker), 'state': str(state_path), 'jobs': jobs}, indent=2) + '\n')
    command = [str(blender), '--background', '--factory-startup', '--python-exit-code', '73',
               '--threads', '2', '--python', str(batch), '--', str(jobs_path)]
    result['shared_process'] = {'command': command, 'status': 'running', 'timeout_seconds': 360 * len(jobs)}
    result['started_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = output / 'results.json'
    report.write_text(json.dumps(result, indent=2) + '\n')
    try:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                 env=render_environment(), timeout=360 * len(jobs), check=False)
        shared_exit = process.returncode
        result['shared_process'].update(status='completed', process_exit_code=shared_exit,
                                        stdout=process.stdout, stderr=process.stderr)
    except subprocess.TimeoutExpired as error:
        shared_exit = None
        result['shared_process'].update(status='timeout', process_exit_code=None,
                                        stdout=str(error.stdout or ''), stderr=str(error.stderr or ''))
    state = json.loads(state_path.read_text()).get('jobs', {}) if state_path.is_file() else {}
    result['shared_process']['case_execution'] = state
    for label, entry in entries.items():
        worker_path, image = Path(entry['worker_report']), Path(entry['render_path'])
        try:
            worker_result = json.loads(worker_path.read_text()) if worker_path.is_file() else {}
        except json.JSONDecodeError as error:
            worker_result = {}
            entry['worker_report_error'] = str(error)
        entry['worker_result'] = worker_result
        entry['execution_state'] = state.get(label, {'status': 'not_completed'})
        if image.is_file():
            entry['render_sha256'] = sha256(image)
            entry['render_size_bytes'] = image.stat().st_size
        entry['passed'] = state.get(label, {}).get('exit_code') == 0 and case_passed(entry['case'], worker_result)
        entry['status'] = 'passed' if entry['passed'] else 'failed'
        print(label + ': ' + entry['status'], flush=True)
    result['passed'] = shared_exit == 0 and all(entry['passed'] for entry in entries.values())
    result['completed_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
