# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Stands in for Cycles' C++ module `_cycles` inside an official Blender.

The Cycles add-on code of the fork (properties, UI, operators) is reused unchanged except that
`import _cycles` points here. Compile-time features come from the fork build (recorded at package
time), devices are asked from the render worker once and cached, and functions that need Cycles
itself (OSL compiler, denoiser, merging) run in the worker.
"""

import collections
import json
import os
import threading
import time

from . import jcy_runtime

_DEVICE_TYPES = ('CUDA', 'OPTIX', 'HIP', 'METAL', 'ONEAPI')

_devices = [None]
_devices_lock = threading.Lock()
_fetch_thread = [None]
_fetch_error = [None]


def __getattr__(name):
    features = jcy_runtime.build_info().get("cycles", {})
    if name in features:
        value = features[name]
        return tuple(value) if isinstance(value, list) else value
    if name.startswith("with_"):
        return False
    raise AttributeError("JCycles has no Cycles attribute " + repr(name))


# -----------------------------------------------------------------------------
# Devices


def _cache_path():
    return os.path.join(jcy_runtime.data_dir(), "devices.json")


def _load_cache():
    try:
        with open(_cache_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("platform") == jcy_runtime.platform_tag():
            return data
    except Exception:
        pass
    return None


def _fetch_devices():
    from . import jcy_client
    try:
        reply = jcy_client.one_shot({"cmd": "devices"}, timeout=300.0)
        data = {"platform": jcy_runtime.platform_tag(), "time": time.time(),
                "device_types": reply["device_types"], "devices": reply["devices"]}
        with open(_cache_path(), "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with _devices_lock:
            _devices[0] = data
        _fetch_error[0] = None
    except Exception as ex:
        _fetch_error[0] = str(ex)
        print("JCycles: cannot list render devices:", ex)


def prefetch_devices(force=False):
    """Start asking the worker for devices in the background (no-op when cached)."""
    with _devices_lock:
        if not force:
            if _devices[0] is None:
                _devices[0] = _load_cache()
            if _devices[0] is not None:
                return
        thread = _fetch_thread[0]
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(target=_fetch_devices, name="JCyclesDevices", daemon=True)
        _fetch_thread[0] = thread
        thread.start()


def refresh_devices_blocking(timeout=300.0):
    prefetch_devices(force=True)
    thread = _fetch_thread[0]
    if thread is not None:
        thread.join(timeout)


def device_status():
    thread = _fetch_thread[0]
    if thread is not None and thread.is_alive():
        return "Detecting render devices..."
    if _fetch_error[0]:
        return "Device detection failed: " + _fetch_error[0].splitlines()[0]
    return ""


def _device_data(wait=10.0):
    with _devices_lock:
        data = _devices[0]
    if data is None:
        prefetch_devices()
        thread = _fetch_thread[0]
        if thread is not None and wait:
            thread.join(wait)
        with _devices_lock:
            data = _devices[0]
    return data


def get_device_types():
    data = _device_data()
    if data and data.get("device_types"):
        return tuple(bool(v) for v in data["device_types"])
    build = jcy_runtime.build_info().get("cycles", {}).get("build_device_types")
    if build and jcy_runtime.platform_tag() == jcy_runtime.build_info().get("platform"):
        return tuple(bool(v) for v in build)
    return (False, False, False, False, False, False)


# Fields of `_cycles.DeviceInfo`, in the order of `device_info_fields` in `blender/python.cpp`.
# The Cycles add-on reads them by name (`device.type`) and some code by index.
_DEVICE_INFO_FIELDS = (
    "name",
    "type",
    "id",
    "has_peer_memory",
    "use_hardware_raytracing",
    "has_openimagedenoise",
    "has_optix_denoiser",
    "has_execution_optimization",
    "meets_driver_requirement",
    "has_dlss",
    "has_metalfx",
)
DeviceInfo = collections.namedtuple("DeviceInfo", _DEVICE_INFO_FIELDS)


def _device_info(values):
    values = list(values)[:len(_DEVICE_INFO_FIELDS)]
    # A runtime with fewer fields does not support the missing features.
    values += [False] * (len(_DEVICE_INFO_FIELDS) - len(values))
    return DeviceInfo(*values)


def available_devices(device_type=None):
    data = _device_data()
    if not data:
        return ()
    devices = data.get("devices", {})
    if device_type is None or device_type == 'NONE':
        device_type = 'CPU'
    # The worker stores exactly what `_cycles.available_devices(type)` returned there.
    return tuple(_device_info(device) for device in devices.get(device_type, []))


# -----------------------------------------------------------------------------
# Functions that run in the worker


def osl_compile(input_path, output_path):
    from . import jcy_client
    try:
        jcy_client.one_shot({"cmd": "osl_compile", "input": input_path, "output": output_path})
        return True
    except Exception as ex:
        print("JCycles: OSL compilation failed:", ex)
        return False


def merge(input, output):
    from . import jcy_client
    jcy_client.one_shot({"cmd": "merge", "input": list(input), "output": output}, timeout=3600.0)
    return True


def denoise(preferences, scene, view_layer, input, output):
    """`scene` and `view_layer` are pointers in the Cycles API; JCycles resolves them by name."""
    import bpy
    from . import jcy_engine
    scene_obj = next((s for s in bpy.data.scenes if s.as_pointer() == scene), bpy.context.scene)
    layer_obj = next((v for v in scene_obj.view_layers if v.as_pointer() == view_layer), None)
    if layer_obj is None:
        layer_obj = bpy.context.view_layer
    jcy_engine.run_scene_tool({"cmd": "denoise", "scene": scene_obj.name, "view_layer": layer_obj.name,
                               "input": list(input), "output": list(output)})
    return True


def debug_flags_update(scene):
    # Debug flags are read by the worker from the scene settings.
    return None


def debug_flags_reset():
    return None


def system_info():
    return jcy_runtime.system_info()


def enable_print_stats():
    return None


def set_device_override(device):
    return True


def maketx(*args, **kwargs):
    raise RuntimeError("maketx is not available in JCycles; use the Cycles build directly")
