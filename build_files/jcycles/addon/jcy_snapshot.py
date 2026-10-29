# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Writing the current scene state to a .blend file the render worker can open.

Blender data may only be written from the main thread while the user can edit. Final renders run
in a render thread, so they hand the work to the main thread through `run_on_main_thread`
(a queue polled by a persistent timer) and wait for it.
"""

import itertools
import os
import queue
import threading
import time

import bpy

_jobs = queue.Queue()
_counter = itertools.count(1)
_timer_running = [False]


class _Job:
    def __init__(self, function):
        self.function = function
        self.done = threading.Event()
        self.result = None
        self.error = None


def _process_jobs():
    while True:
        try:
            job = _jobs.get_nowait()
        except queue.Empty:
            break
        try:
            job.result = job.function()
        except BaseException as ex:
            job.error = ex
        job.done.set()
    return 0.05


def start():
    if not _timer_running[0]:
        bpy.app.timers.register(_process_jobs, first_interval=0.05, persistent=True)
        _timer_running[0] = True


def stop():
    if _timer_running[0]:
        try:
            bpy.app.timers.unregister(_process_jobs)
        except ValueError:
            pass
        _timer_running[0] = False


def run_on_main_thread(function, should_cancel=None, timeout=None):
    if threading.current_thread() is threading.main_thread():
        return function()
    job = _Job(function)
    _jobs.put(job)
    deadline = None if timeout is None else time.monotonic() + timeout
    while not job.done.wait(0.1):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("Cancelled")
        if deadline is not None and time.monotonic() > deadline:
            raise RuntimeError("Blender's main thread did not respond (is the interface blocked?)")
    if job.error is not None:
        raise job.error
    return job.result


def _flush_edit_modes():
    for obj in bpy.data.objects:
        try:
            if obj.mode == 'EDIT' and hasattr(obj, "update_from_editmode"):
                obj.update_from_editmode()
        except Exception:
            pass


def _link_dir(source, link):
    if os.path.lexists(link):
        return
    try:
        os.symlink(source, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(source, link)
        except Exception:
            pass


def _mirror_derived_paths(blend_path, snapshot_dir, stem):
    """Point caches and simulation bakes without an explicit path live next to the .blend file
    and are named after it. Make those names resolve from the snapshot directory too."""
    if not blend_path:
        return
    source_dir = os.path.dirname(blend_path)
    try:
        entries = os.listdir(source_dir)
    except OSError:
        return
    for name in entries:
        full = os.path.join(source_dir, name)
        if not os.path.isdir(full):
            continue
        if name == "blendcache_" + stem or name.startswith(stem + "_") or name in {"bakes", "cache", "caches"}:
            _link_dir(full, os.path.join(snapshot_dir, name))


def _export_unsaved_images(directory):
    """Images painted or generated in memory: pass their pixels to the worker."""
    overrides = {}
    for image in bpy.data.images:
        try:
            if image.type not in {'IMAGE', 'UV_TEST'} or image.source not in {'FILE', 'GENERATED'}:
                continue
            if not (image.is_dirty and image.has_data):
                continue
            width, height = image.size
            if width == 0 or height == 0:
                continue
            import numpy as np
            pixels = np.empty(width * height * image.channels, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            pixels = pixels.reshape(height, width, image.channels)
            index = len(overrides)
            path = os.path.join(directory, "image_{:d}.exr".format(index) if image.is_float
                                else "image_{:d}.png".format(index))
            import OpenImageIO as oiio
            if image.is_float:
                spec = oiio.ImageSpec(width, height, image.channels, oiio.FLOAT)
                data = pixels[::-1]
            else:
                spec = oiio.ImageSpec(width, height, image.channels, oiio.UINT16 if image.depth > 32 else oiio.UINT8)
                data = pixels[::-1]
            out = oiio.ImageOutput.create(path)
            if out is None or not out.open(path, spec):
                continue
            out.write_image(np.ascontiguousarray(data))
            out.close()
            overrides[image.name] = {"path": path}
        except Exception as ex:
            print("JCycles: cannot pass unsaved image {!r} to the render: {:s}".format(image.name, str(ex)))
    return overrides


def write_snapshot(base_dir):
    """Main thread only. Returns (token, path, image overrides)."""
    _flush_edit_modes()
    token = "s{:d}_{:d}".format(os.getpid(), next(_counter))
    directory = os.path.join(base_dir, token)
    os.makedirs(directory, exist_ok=True)
    blend_path = bpy.data.filepath
    stem = os.path.splitext(os.path.basename(blend_path))[0] if blend_path else "untitled"
    path = os.path.join(directory, stem + ".blend")
    ids = set(bpy.data.scenes)
    try:
        bpy.data.libraries.write(path, ids, path_remap='ABSOLUTE', fake_user=True, compress=False)
    except TypeError:
        bpy.data.libraries.write(path, ids, path_remap='ABSOLUTE', fake_user=True)
    _mirror_derived_paths(blend_path, directory, stem)
    overrides = _export_unsaved_images(directory)
    return token, path, overrides


def remove_snapshot(path):
    import shutil
    directory = os.path.dirname(path)
    if os.path.basename(directory).startswith("s") and os.path.isdir(directory):
        # Remove the mirrored links first so the originals are never touched.
        for name in os.listdir(directory):
            full = os.path.join(directory, name)
            if os.path.islink(full):
                os.unlink(full)
            elif os.name == "nt" and os.path.isdir(full) and _is_junction(full):
                os.rmdir(full)
        shutil.rmtree(directory, ignore_errors=True)


def _is_junction(path):
    try:
        return bool(os.readlink(path))
    except (OSError, ValueError, AttributeError):
        return False
