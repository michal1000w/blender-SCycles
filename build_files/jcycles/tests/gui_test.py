# Runs in an official Blender WITH its interface (not -b):
#   blender --factory-startup <scene.blend> --python gui_test.py -- <out_dir>
# Rendered viewport (free view, then camera view), then F12 through the UI (render job thread).
import json
import os
import sys
import time

import addon_utils
import bpy

out_dir = sys.argv[sys.argv.index("--") + 1]
addon_utils.enable("JCycles", default_set=True)
from JCycles import jcy_viewport  # noqa: E402

status = {"steps": []}
state = {"phase": "start", "t": time.monotonic(), "start": time.monotonic()}


def log(text):
    status["steps"].append([round(time.monotonic() - state["start"], 2), text])
    print("GUI", text, flush=True)


def view3d():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                region = next(r for r in area.regions if r.type == 'WINDOW')
                return window, area, region
    return None, None, None


def session_state():
    sessions = list(jcy_viewport._sessions)
    if not sessions:
        return None
    s = sessions[-1]
    return {"samples": s.frame.get("samples") if s.frame else None, "error": s.error,
            "size": [s.frame["width"], s.frame["height"]] if s.frame else None}


def finish():
    with open(os.path.join(out_dir, "gui_status.json"), "w") as fh:
        json.dump(status, fh, indent=1)
    bpy.ops.wm.quit_blender()


@bpy.app.handlers.persistent
def on_complete(*_args):
    state["rendered"] = True


bpy.app.handlers.render_complete.append(on_complete)


def tick():
    now = time.monotonic()
    elapsed = now - state["t"]
    window, area, region = view3d()
    if now - state["start"] > 300:
        log("TIMEOUT in phase " + state["phase"])
        finish()
        return None
    phase = state["phase"]
    if phase == "start":
        bpy.context.scene.render.engine = 'JCYCLES'
        space = area.spaces.active
        space.region_3d.view_perspective = 'PERSP'
        space.shading.type = 'RENDERED'
        state.update(phase="free_view", t=now)
        log("rendered shading on (free view)")
    elif phase == "free_view":
        s = session_state()
        if s and (s["error"] or (s["samples"] or 0) >= 16):
            log("free view: {!r}".format(s))
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.screen.screenshot_area(filepath=os.path.join(out_dir, "viewport_free.png"))
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
            state.update(phase="camera_view", t=now, prev=s)
        elif elapsed > 120:
            log("free view timeout: {!r}".format(s))
            state.update(phase="camera_view", t=now)
    elif phase == "camera_view":
        s = session_state()
        if elapsed > 1.0 and s and (s["error"] or ((s["samples"] or 0) >= 16 and s["size"] != state.get("prev", {}).get("size"))):
            log("camera view: {!r}".format(s))
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.screen.screenshot_area(filepath=os.path.join(out_dir, "viewport_camera.png"))
            area.spaces.active.shading.type = 'SOLID'
            state.update(phase="f12", t=now)
        elif elapsed > 120:
            log("camera view timeout: {!r}".format(s))
            state.update(phase="f12", t=now)
    elif phase == "f12":
        if elapsed > 1.0:
            with bpy.context.temp_override(window=window, area=area, region=region):
                result = bpy.ops.render.render('INVOKE_DEFAULT')
            log("F12 invoked: {!r}".format(result))
            state.update(phase="wait_render", t=now)
    elif phase == "wait_render":
        if state.get("rendered"):
            image = bpy.data.images["Render Result"]
            path = os.path.join(out_dir, "f12.exr")
            scene = bpy.context.scene
            scene.render.image_settings.media_type = 'MULTI_LAYER_IMAGE' if hasattr(scene.render.image_settings, 'media_type') else None
            scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
            scene.render.image_settings.color_depth = '32'
            image.save_render(path, scene=scene)
            log("F12 finished in {:.2f}s, saved {:s}".format(elapsed, path))
            finish()
            return None
        if elapsed > 180:
            log("F12 timeout")
            finish()
            return None
    return 0.2


bpy.app.timers.register(tick, first_interval=2.0)
