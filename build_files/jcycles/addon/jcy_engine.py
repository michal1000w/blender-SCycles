# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""The JCycles render engine: final renders, viewport, baking and OSL go to the fork's Cycles."""

import os
import re
import threading
import time
import traceback

import bpy

from . import jcy_client, jcy_runtime, jcy_snapshot

ADDON_ID = __package__

_SAMPLE_RE = re.compile(r"Sample (\d+)/(\d+)")
_TILE_RE = re.compile(r"Rendered (\d+)/(\d+) Tiles")
_MEM_RE = re.compile(r"Mem:\s*([\d.]+)M(?:, Peak:\s*([\d.]+)M)?")

_final = {"worker": None, "lock": threading.Lock()}
_job = {"id": 0, "snapshot": None, "scene": None, "active": False}


# -----------------------------------------------------------------------------
# Settings passed to the worker


def addon_preferences():
    addon = bpy.context.preferences.addons.get(ADDON_ID)
    return addon.preferences if addon else None


def device_settings():
    prefs = addon_preferences()
    settings = {"preferences": {}, "devices": [], "experimental": {}}
    if prefs is not None:
        for prop in prefs.bl_rna.properties:
            identifier = prop.identifier
            if identifier in {"rna_type", "bl_idname", "devices"} or prop.is_readonly:
                continue
            if prop.type in {'BOOLEAN', 'INT', 'FLOAT', 'STRING', 'ENUM'} and not getattr(prop, "array_length", 0):
                if prop.type == 'ENUM' and prop.is_enum_flag:
                    continue
                try:
                    settings["preferences"][identifier] = getattr(prefs, identifier)
                except Exception:
                    pass
        for device in prefs.devices:
            settings["devices"].append([device.type, device.id, bool(device.use)])
    try:
        settings["experimental"]["use_cycles_debug"] = bool(bpy.context.preferences.experimental.use_cycles_debug)
        settings["experimental"]["show_developer_ui"] = bool(bpy.context.preferences.view.show_developer_ui)
    except Exception:
        pass
    return settings


def autoexec_enabled():
    try:
        return bool(bpy.context.preferences.filepaths.use_scripts_auto_execute) and not bpy.app.autoexec_fail
    except Exception:
        return False


def echo_worker_output():
    prefs = addon_preferences()
    return bool(getattr(prefs, "jcycles_echo_worker", False)) if prefs else False


# -----------------------------------------------------------------------------
# Render job tracking (one snapshot per F12 / animation render)


@bpy.app.handlers.persistent
def _on_render_init(*_args):
    _job["id"] += 1
    _job["snapshot"] = None
    _job["scene"] = None
    _job["active"] = True


@bpy.app.handlers.persistent
def _on_render_end(*_args):
    _job["active"] = False
    snapshot = _job["snapshot"]
    _job["snapshot"] = None
    keep_worker = False
    try:
        scene = bpy.data.scenes.get(_job["scene"] or "")
        keep_worker = bool(scene and scene.render.use_persistent_data)
    except Exception:
        pass

    def cleanup():
        if snapshot:
            jcy_snapshot.remove_snapshot(snapshot[1])
        if not keep_worker:
            with _final["lock"]:
                worker = _final["worker"]
                _final["worker"] = None
            if worker is not None:
                worker.stop()

    threading.Thread(target=cleanup, daemon=True).start()


def register_handlers():
    for handlers, function in ((bpy.app.handlers.render_init, _on_render_init),
                               (bpy.app.handlers.render_complete, _on_render_end),
                               (bpy.app.handlers.render_cancel, _on_render_end)):
        if function not in handlers:
            handlers.append(function)


def unregister_handlers():
    for handlers, function in ((bpy.app.handlers.render_init, _on_render_init),
                               (bpy.app.handlers.render_complete, _on_render_end),
                               (bpy.app.handlers.render_cancel, _on_render_end)):
        while function in handlers:
            handlers.remove(function)


def final_worker():
    with _final["lock"]:
        worker = _final["worker"]
        autoexec = autoexec_enabled()
        if worker is not None and (not worker.is_alive() or worker.autoexec != autoexec):
            worker.stop(kill=not worker.is_alive())
            worker = None
        if worker is None:
            worker = jcy_client.Worker("final", autoexec=autoexec, echo=echo_worker_output())
            worker.start()
            _final["worker"] = worker
        return worker


def drop_final_worker():
    with _final["lock"]:
        worker = _final["worker"]
        _final["worker"] = None
    if worker is not None:
        worker.stop(kill=True)


def ensure_loaded(worker, snapshot, should_cancel=None):
    token, path, overrides = snapshot
    if worker.loaded_token == token:
        return
    worker.call({"cmd": "load", "path": path, "token": token, "autoexec": worker.autoexec,
                 "image_overrides": overrides}, should_cancel=should_cancel)
    worker.loaded_token = token


def run_scene_tool(command):
    """Snapshot the current file and run a scene command (denoise, ...) in a fresh worker."""
    snapshot = jcy_snapshot.write_snapshot(os.path.join(jcy_runtime.session_dir(), "snapshots"))
    worker = jcy_client.Worker("tool", autoexec=autoexec_enabled())
    try:
        worker.start()
        ensure_loaded(worker, snapshot)
        command = dict(command)
        command["token"] = snapshot[0]
        command["device"] = device_settings()
        return worker.call(command)
    finally:
        worker.stop()
        jcy_snapshot.remove_snapshot(snapshot[1])


def _report_error(engine, message):
    print("JCycles:", message)
    try:
        engine.report({'ERROR'}, message.splitlines()[0][:500])
    except Exception:
        pass
    error_set = getattr(engine, "error_set", None)
    if error_set is not None:
        try:
            error_set(message.splitlines()[0][:500])
        except Exception:
            pass


# -----------------------------------------------------------------------------
# The render engine


class JCyclesRender(bpy.types.RenderEngine):
    bl_idname = 'JCYCLES'
    bl_label = "JCycles"
    bl_use_eevee_viewport = True
    bl_use_preview = False
    bl_use_exclude_layers = True
    bl_use_spherical_stereo = True
    bl_use_shading_nodes_custom = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._jcy_key = None
        self._jcy_exr = None
        self._jcy_header = None
        self._jcy_view = None
        self._jcy_bake = None

    def __del__(self):
        view = getattr(self, "_jcy_view", None)
        if view is not None:
            view.free()
            self._jcy_view = None

    # Final render -----------------------------------------------------------

    def update(self, data, depsgraph):
        pass

    def render(self, depsgraph):
        try:
            scene = depsgraph.scene
            layer = depsgraph.view_layer.name
            key = (_job["id"], scene.name, scene.frame_current, round(scene.frame_subframe, 6))
            if getattr(self, "_jcy_failed", None) == key:
                return
            if self._jcy_key != key or self._jcy_exr is None:
                self._jcy_exr = None
                self._render_frame(depsgraph)
                self._jcy_key = key
            if self._jcy_exr is not None:
                self._load_layer(layer)
        except jcy_client.WorkerCancelled:
            self._jcy_key = None
            self._jcy_exr = None
            with _final["lock"]:
                _final["worker"] = None
        except Exception as ex:
            traceback.print_exc()
            _report_error(self, "JCycles render failed: {:s}".format(str(ex)))
            self._jcy_failed = key if "key" in locals() else None
            self._jcy_key = None
            self._jcy_exr = None

    def _snapshot_for_job(self, scene_name):
        snapshot = _job["snapshot"]
        if snapshot is None or _job["scene"] != scene_name or not _job["active"]:
            snapshot = jcy_snapshot.run_on_main_thread(
                lambda: jcy_snapshot.write_snapshot(os.path.join(jcy_runtime.session_dir(), "snapshots")),
                should_cancel=self.test_break)
            if _job["active"]:
                _job["snapshot"] = snapshot
                _job["scene"] = scene_name
        return snapshot

    def _render_frame(self, depsgraph):
        scene = depsgraph.scene
        render = scene.render
        current = depsgraph.view_layer.name
        if render.use_single_layer:
            layers = [current]
        else:
            layers = [vl.name for vl in scene.view_layers if vl.use]
            if current not in layers:
                layers.append(current)
        camera = self.camera_override or scene.camera

        self.update_stats("", "JCycles | Preparing scene")
        snapshot = self._snapshot_for_job(scene.name)
        if self.test_break():
            return
        self.update_stats("", "JCycles | Starting render process")
        worker = final_worker()
        self.update_stats("", "JCycles | Loading scene")
        ensure_loaded(worker, snapshot, should_cancel=self.test_break)

        out_dir = os.path.join(jcy_runtime.session_dir(), "frames")
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, "{:s}_{:d}_{:d}.exr".format(
            snapshot[0], scene.frame_current, int(round(scene.frame_subframe * 1000))))
        command = {
            "cmd": "render",
            "token": snapshot[0],
            "scene": scene.name,
            "frame": scene.frame_current,
            "subframe": scene.frame_subframe,
            "camera": camera.name if camera else None,
            "layers": layers,
            "output": output,
            "device": device_settings(),
        }
        worker.call(command, on_event=self._on_event, should_cancel=self.test_break)
        from . import jcy_exr
        self._jcy_exr = output
        try:
            self._jcy_header = jcy_exr.read_header(output)
        except Exception as ex:
            print("JCycles: cannot read render header:", ex)
            self._jcy_header = {"channels": [], "attributes": {}}

    def _on_event(self, message):
        if message.get("event") != "stats":
            return
        text = message.get("text", "")
        self.update_stats("", "JCycles | " + text)
        match = _SAMPLE_RE.search(text) or _TILE_RE.search(text)
        if match:
            total = int(match.group(2))
            if total > 0:
                self.update_progress(min(1.0, int(match.group(1)) / total))
        mem = _MEM_RE.search(text)
        if mem:
            used = float(mem.group(1))
            peak = float(mem.group(2)) if mem.group(2) else used
            try:
                self.update_memory_stats(memory_used=used, memory_peak=peak)
            except Exception:
                pass

    def _load_layer(self, layer):
        header = self._jcy_header or {"channels": [], "attributes": {}}
        result = self.begin_result(0, 0, 1 << 20, 1 << 20, layer=layer)
        if result is None:
            return
        try:
            result.load_from_file(self._jcy_exr)
            for key, value in header["attributes"].items():
                if key.startswith("cryptomatte/"):
                    try:
                        result.stamp_data_add_field(key, value)
                    except Exception:
                        pass
        finally:
            self.end_result(result)
        self.update_progress(1.0)

    # Baking -------------------------------------------------------------------

    def bake(self, depsgraph, obj, pass_type, pass_filter, width, height):
        try:
            self._bake(depsgraph, obj, pass_type, width, height)
        except jcy_client.WorkerCancelled:
            with _final["lock"]:
                _final["worker"] = None
        except Exception as ex:
            traceback.print_exc()
            _report_error(self, "JCycles bake failed: {:s}".format(str(ex)))

    def _bake(self, depsgraph, obj, pass_type, width, height):
        import numpy as np
        scene = depsgraph.scene
        name = obj.original.name if obj.original else obj.name
        state = self._jcy_bake
        if state is None or state["object"] != name or state["pass_type"] != pass_type:
            snapshot = jcy_snapshot.run_on_main_thread(
                lambda: jcy_snapshot.write_snapshot(os.path.join(jcy_runtime.session_dir(), "snapshots")),
                should_cancel=self.test_break)
            worker = final_worker()
            ensure_loaded(worker, snapshot, should_cancel=self.test_break)
            target = scene.render.bake.target
            reply = worker.call({
                "cmd": "bake", "token": snapshot[0], "scene": scene.name,
                "view_layer": depsgraph.view_layer.name, "object": name,
                "frame": scene.frame_current, "subframe": scene.frame_subframe,
                "camera": scene.camera.name if scene.camera else None,
                "pass_type": pass_type, "target": target, "device": device_settings(),
            }, on_event=self._on_event, should_cancel=self.test_break)
            state = {"object": name, "pass_type": pass_type, "outputs": reply["outputs"], "next": 0}
            self._jcy_bake = state
        index = state["next"]
        state["next"] += 1
        outputs = state["outputs"]
        match = None
        if index < len(outputs) and outputs[index].get("width", -1) * outputs[index].get("height", 1) == width * height:
            match = outputs[index]
        else:
            for output in outputs:
                if output.get("width", -1) * output.get("height", 1) == width * height or \
                        output.get("count", -1) == width * height:
                    match = output
                    break
        if match is None:
            raise RuntimeError("No baked data of {:d}x{:d} pixels for {!r}".format(width, height, name))
        pixels = np.fromfile(match["path"], dtype=np.float32)
        if pixels.size != width * height * 4:
            raise RuntimeError("Baked data has an unexpected size")
        # An explicit layer name: official Blender crashes on an empty one in its bake path.
        result = self.begin_result(0, 0, width, height, layer="JCyclesBake", view="")
        try:
            combined = result.layers[0].passes.find_by_name("Combined", "") if hasattr(
                result.layers[0].passes, "find_by_name") else result.layers[0].passes["Combined"]
            combined.rect = pixels.reshape(width * height, 4)
        finally:
            self.end_result(result)

    # Viewport -------------------------------------------------------------------

    def view_update(self, context, depsgraph):
        try:
            if self._jcy_view is None:
                from . import jcy_viewport
                self._jcy_view = jcy_viewport.ViewSession(self)
            self._jcy_view.update(context, depsgraph)
        except Exception:
            traceback.print_exc()

    def view_draw(self, context, depsgraph):
        try:
            if self._jcy_view is None:
                from . import jcy_viewport
                self._jcy_view = jcy_viewport.ViewSession(self)
                self._jcy_view.update(context, depsgraph)
            self._jcy_view.draw(context, depsgraph)
        except Exception:
            traceback.print_exc()

    # Scripts and passes ------------------------------------------------------

    def update_script_node(self, node):
        from . import osl
        osl.update_script_node(node, self.report)

    def update_custom_camera(self, cam):
        from . import osl
        osl.update_custom_camera_shader(cam, self.report)

    def update_render_passes(self, scene, srl):
        from . import engine
        engine.register_passes(self, scene, srl)


classes = (JCyclesRender,)


def register():
    jcy_snapshot.start()
    register_handlers()
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    unregister_handlers()
    jcy_snapshot.stop()
    drop_final_worker()
