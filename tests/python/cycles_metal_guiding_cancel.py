#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Cancel an active Metal guiding render, then verify a fresh render in the same session.

Run with system Python, --blender and --output. The controller sends one SIGINT to
its own Blender child after observing completed rendering work. Blender maps this
to an internal render break. A render-cancel handler confirms cancellation; the
same Python session then changes lighting and renders again with a finite limit.
"""

import argparse
import json
import os
import pathlib
import queue
import re
import runpy
import signal
import subprocess
import sys
import threading
import time


def inside_blender(output, bdpt=False):
    import bpy
    import numpy as np
    import OpenImageIO as oiio

    scene_script = pathlib.Path(__file__).with_name('cycles_metal_guiding_scene.py')
    sys.argv = [str(scene_script), '--', '--build-only', '--guiding', '--resolution', '32',
                '--samples', '1048576', '--training-samples', '0', '--memory-mb', '16',
                '--output', str(output / 'scene')]
    if bdpt:
        sys.argv.append('--bdpt')
    runpy.run_path(str(scene_script), run_name='__main__')
    scene = bpy.context.scene
    scene.render.use_persistent_data = True
    cancelled = []

    def on_cancel(_scene):
        cancelled.append(True)

    bpy.app.handlers.render_cancel.append(on_cancel)
    print('GUIDING_CANCEL_BEGIN', flush=True)
    try:
        bpy.ops.render.render()
    except (RuntimeError, KeyboardInterrupt) as error:
        print('GUIDING_CANCEL_EXCEPTION ' + str(error), flush=True)
    finally:
        bpy.app.handlers.render_cancel.remove(on_cancel)
    assert cancelled, 'No render-cancel event was delivered'
    scene.cycles.samples = 32
    scene.cycles.guiding_training_samples = 5
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .1
    scene.render.filepath = str(output / 'restarted.exr')
    print('GUIDING_CANCEL_RESTART', flush=True)
    bpy.ops.render.render(write_still=True)
    image = oiio.ImageInput.open(scene.render.filepath)
    assert image is not None
    pixels = np.asarray(image.read_image(format=oiio.FLOAT))[..., :3]
    image.close()
    assert np.isfinite(pixels).all() and pixels.max() > 0, 'Invalid restarted render'
    (output / 'image.json').write_text(json.dumps(dict(mean=float(pixels.mean()),
                                                     peak=float(pixels.max())), indent=2) + '\n')
    print('GUIDING_CANCEL_PASSED', flush=True)


def controller(blender, output, bdpt=False):
    output.mkdir(parents=True, exist_ok=True)
    command = [str(blender), '--background', '--factory-startup', '--debug-cycles',
               '--log-level', 'debug', '--python-exit-code', '1', '--python', str(pathlib.Path(__file__).resolve()),
               '--', '--inside-blender', '--output', str(output)]
    if bdpt:
        command.append('--bdpt')
    env = dict(os.environ, BLENDER_USER_RESOURCES=str(output / 'blender-user'))
    child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = queue.Queue()

    def read_output():
        for line in child.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    transcript = []
    sent = False
    ready_at = None
    sent_at = None
    restart_at = None
    deadline = time.monotonic() + 900
    try:
        with (output / 'render.log').open('w') as log:
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Cancellation regression timed out')
                # Let the next scheduled batch enter GPU work, rather than cancelling exactly
                # at the outer batch boundary that already had a cancellation check.
                if not sent and ready_at is not None and time.monotonic() - ready_at >= .25:
                    child.send_signal(signal.SIGINT)
                    sent = True
                    sent_at = time.monotonic()
                try:
                    line = lines.get(timeout=.02 if ready_at is not None and not sent else 1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                transcript.append(line)
                log.write(line)
                log.flush()
                if ready_at is None and re.search(r'Rendered \d+ samples in', line):
                    ready_at = time.monotonic()
                if 'GUIDING_CANCEL_RESTART' in line:
                    restart_at = time.monotonic()
            result = child.wait(timeout=10)
        text = ''.join(transcript)
        assert sent and result == 0 and 'GUIDING_CANCEL_PASSED' in text, \
            'Cancel/restart failed; inspect render.log'
        restart = text.split('GUIDING_CANCEL_RESTART', 1)[1]
        publications = [int(value) for value in re.findall(
            r'Metal guiding publish: trained_samples=(\d+)', restart)]
        assert publications == [1, 2, 4, 5], 'Restart reused stale training state'
        latency = restart_at - sent_at
        assert latency < 5, f'Cancellation was delayed for {latency:.3f} seconds'
        (output / 'report.json').write_text(json.dumps(dict(cancelled=True, restarted=True,
            cancel_latency_seconds=latency, training_publications=publications,
            integrator='bdpt' if bdpt else 'pt'), indent=2) + '\n')
        print('GUIDING_CANCEL_PASSED ' + str(output))
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--inside-blender', action='store_true')
    parser.add_argument('--bdpt', action='store_true')
    arguments = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    output = args.output.resolve()
    if args.inside_blender:
        inside_blender(output, args.bdpt)
    else:
        if args.blender is None:
            parser.error('--blender is required')
        controller(args.blender.resolve(), output, args.bdpt)


if __name__ == '__main__':
    main()
