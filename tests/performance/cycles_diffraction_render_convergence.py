#!/usr/bin/env python3
"""Independent-seed physical diffraction render comparison on Metal.

Run outside the sandbox. Error bars use independent render means, never pixels.
Timing is deliberately not reported as a rendering benchmark: each invocation
also prepares caches, initializes the device and integrates a CPU reference.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 29, 47, 83])
    parser.add_argument('--samples', nargs='+', type=int, default=[256, 4096])
    parser.add_argument('--fixture', choices=['mixed', 'transmission'], default='mixed')
    parser.add_argument('--illumination', choices=['both', 'reflection', 'transmission'], default='both')
    parser.add_argument('--reference-samples', type=int, default=1024)
    parser.add_argument('--transports', nargs='+', choices=['pt', 'bdpt', 'guided', 'bdpt_guided'],
                        default=['pt', 'bdpt_guided'])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error('Use at least two distinct seeds')
    if any(x < 0 for x in args.seeds) or any(x <= 0 for x in args.samples):
        parser.error('Seeds must be nonnegative and sample counts positive')
    if args.reference_samples <= 0:
        parser.error('Reference sample count must be positive')
    if args.fixture != 'transmission' and args.illumination != 'both':
        parser.error('Hemisphere illumination requires the transmission fixture')
    if len(set(args.samples)) != len(args.samples) or len(set(args.transports)) != len(args.transports):
        parser.error('Sample counts and transport modes must be distinct')
    root = Path(__file__).resolve().parents[2]
    binary = root / 'build/macos_arm64_Release/bin/cycles_diffraction_device_test'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['DYLD_LIBRARY_PATH'] = str(root / 'install/Blender.app/Contents/Resources/lib')
    environment['CYCLES_KERNEL_PATH'] = str(root / 'intern/cycles')
    manifest = {'binary_sha256': sha256(binary), 'runs': [], 'groups': [],
                'fixture': args.fixture, 'illumination': args.illumination,
                'reference_samples': args.reference_samples,
                'scope': 'Normal-incidence fixture with uniform or hemispherical illumination. '
                         'Standard errors describe between-seed variability only.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for samples in args.samples:
        for transport in args.transports:
            means = []
            references = []
            for seed in args.seeds:
                prefix = output / f'{transport}_{samples}_seed{seed}'
                command = [str(binary), '--render-' + args.fixture, '--metal', '--transport', transport,
                           '--seed', str(seed), '--samples', str(samples), '--output', str(prefix),
                           '--reference-samples', str(args.reference_samples)]
                if args.fixture == 'transmission':
                    command.extend(['--illumination', args.illumination])
                print(f'Starting {prefix.name}', flush=True)
                with prefix.with_suffix('.log').open('w') as log:
                    result = subprocess.run(command, cwd=root, env=environment,
                                            stdout=log, stderr=subprocess.STDOUT)
                record = {'name': prefix.name, 'command': command, 'exit_code': result.returncode}
                manifest['runs'].append(record)
                (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                if result.returncode:
                    raise RuntimeError(f'Render failed: {prefix.with_suffix(".log")}')
                report = json.loads(prefix.with_suffix('.json').read_text())
                assert report['seed'] == seed and report['samples'] == samples
                assert report['transport'] == transport and report['device_type'] == 'METAL'
                assert report['illumination'] == args.illumination
                assert report['quadrature_count'] == args.reference_samples
                with prefix.with_suffix('.pfm').open('rb') as image:
                    assert image.readline() == b'PF\n'
                    width, height = map(int, image.readline().split())
                    scale = float(image.readline())
                    assert abs(scale) == 1
                    pixels = struct.unpack(('<' if scale < 0 else '>') + str(width*height*3)+'f',
                                           image.read())
                assert all(math.isfinite(x) for x in pixels)
                mean = [statistics.fmean(pixels[c::3]) for c in range(3)]
                assert max(abs(a-b) for a,b in zip(mean, report['mean_rgb'])) < 1e-6
                means.append(mean)
                references.append(report['reference_rgb'])
                record['mean_rgb'] = mean
                record['pfm_sha256'] = sha256(prefix.with_suffix('.pfm'))
                record['report_sha256'] = sha256(prefix.with_suffix('.json'))
                print(f'Completed {prefix.name}: {mean}', flush=True)
            average = [statistics.fmean(row[c] for row in means) for c in range(3)]
            reference = [statistics.fmean(row[c] for row in references) for c in range(3)]
            group = {'transport': transport, 'samples': samples, 'seeds': args.seeds,
                     'mean_rgb': average, 'reference_rgb': reference,
                     'mean_error_rgb': [a-b for a,b in zip(average, reference)],
                     'standard_error_rgb': [statistics.stdev(row[c] for row in means) /
                                            math.sqrt(len(means)) for c in range(3)]}
            manifest['groups'].append(group)
            assert sha256(binary) == manifest['binary_sha256'], 'Binary changed during experiment'
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(json.dumps(group), flush=True)


if __name__ == '__main__':
    main()
