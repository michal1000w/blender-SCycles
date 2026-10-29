# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Locating and preparing the fork's Blender build that ships inside the add-on."""

import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
import threading

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(ADDON_DIR, "jcy_worker", "worker_main.py")

_build_info = [None]
_prepared = set()
_lock = threading.Lock()
_session_dir = [None]


class RuntimeMissing(RuntimeError):
    pass


def build_info():
    if _build_info[0] is None:
        path = os.path.join(ADDON_DIR, "jcycles_build.json")
        try:
            with open(path, encoding="utf-8") as fh:
                _build_info[0] = json.load(fh)
        except OSError:
            _build_info[0] = {"cycles": {}, "runtimes": {}}
    return _build_info[0]


def node_specs():
    path = os.path.join(ADDON_DIR, "node_specs.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def platform_tag():
    system = platform.system()
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        machine = "x64"
    elif machine in {"aarch64", "arm64"}:
        machine = "arm64"
    name = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(system, system.lower())
    return "{:s}-{:s}".format(name, machine)


def executable_relpath(tag):
    if tag.startswith("macos"):
        return os.path.join("Blender.app", "Contents", "MacOS", "Blender")
    if tag.startswith("windows"):
        return "blender.exe"
    return "blender"


def runtime_dir(tag=None):
    return os.path.join(ADDON_DIR, "runtime", tag or platform_tag())


def available_runtimes():
    root = os.path.join(ADDON_DIR, "runtime")
    try:
        return sorted(name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)))
    except OSError:
        return []


def executable():
    tag = platform_tag()
    path = os.path.join(runtime_dir(tag), executable_relpath(tag))
    if not os.path.isfile(path):
        others = available_runtimes()
        raise RuntimeMissing(
            "This JCycles package has no render runtime for {:s} (it contains: {:s}). "
            "Build JCycles on a {:s} machine with compileCycles.".format(
                tag, ", ".join(others) or "none", tag.split("-")[0]))
    prepare(tag, path)
    return path


def _make_executable(path):
    try:
        mode = os.stat(path).st_mode
        wanted = mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        if mode != wanted:
            os.chmod(path, wanted)
    except OSError:
        pass


def prepare(tag, exe):
    """Undo what installing from a zip loses: executable bits (and quarantine on macOS)."""
    with _lock:
        if exe in _prepared:
            return
        root = runtime_dir(tag)
        if not tag.startswith("windows"):
            _make_executable(exe)
            if tag.startswith("macos"):
                macos_dir = os.path.dirname(exe)
                for name in os.listdir(macos_dir):
                    _make_executable(os.path.join(macos_dir, name))
            # Helper executables inside the bundled Python (used by pip and some modules).
            for dirpath, dirnames, filenames in os.walk(root):
                if os.path.basename(dirpath) == "bin":
                    for name in filenames:
                        _make_executable(os.path.join(dirpath, name))
            if tag.startswith("linux"):
                for name in os.listdir(root):
                    full = os.path.join(root, name)
                    if os.path.isfile(full) and name.startswith("blender") and "." not in name:
                        _make_executable(full)
        if tag.startswith("macos"):
            app = os.path.join(root, "Blender.app")
            marker = os.path.join(root, ".jcycles_unquarantined")
            if not os.path.exists(marker):
                try:
                    subprocess.run(["xattr", "-dr", "com.apple.quarantine", app],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                    with open(marker, "w") as fh:
                        fh.write("1")
                except Exception:
                    pass
        _prepared.add(exe)


def data_dir():
    """Persistent per-user directory, shared by all official Blender versions."""
    import bpy
    try:
        base = os.path.dirname(bpy.utils.resource_path('USER'))
    except Exception:
        base = os.path.join(os.path.expanduser("~"), ".jcycles")
    info = build_info()
    build_id = (info.get("fork", {}).get("build_hash") or "unknown")[:12]
    path = os.path.join(base, "JCycles", build_id)
    os.makedirs(path, exist_ok=True)
    return path


def worker_home():
    path = os.path.join(data_dir(), "worker_home")
    os.makedirs(path, exist_ok=True)
    return path


def session_dir():
    """Temporary directory for this Blender session (snapshots, render results)."""
    with _lock:
        if _session_dir[0] is None or not os.path.isdir(_session_dir[0]):
            _session_dir[0] = tempfile.mkdtemp(prefix="jcycles_")
        return _session_dir[0]


def cleanup_session():
    import shutil
    path = _session_dir[0]
    _session_dir[0] = None
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def worker_env():
    env = dict(os.environ)
    # Never let the host's Python or Blender environment leak into the fork's Blender.
    for key in list(env):
        if key.startswith(("PYTHON", "BLENDER_")) or key in {"VIRTUAL_ENV"}:
            del env[key]
    # Own user directory: no user add-ons, preferences or startup scripts of any installed
    # Blender, and a persistent place for kernel caches.
    env["BLENDER_USER_RESOURCES"] = worker_home()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def system_info():
    info = build_info()
    fork = info.get("fork", {})
    return "JCycles runtime: Blender {:s} ({:s}), platform {:s}, executable {:s}".format(
        fork.get("version_string", "?"), fork.get("build_hash", "?"), platform_tag(),
        os.path.join(runtime_dir(), executable_relpath(platform_tag())))


def python_info():
    return sys.version
