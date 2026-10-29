# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Viewport (Rendered shading) for JCycles.

The fork's Cycles runs in the worker as progressive final renders of what the viewport shows:
the viewport camera (or the scene camera in camera view), viewport visibility, modifiers and the
viewport sampling settings. Moving the view only re-renders; editing the scene sends a new
snapshot of the file after a short pause.
"""

import os
import time
import weakref

import bpy

from . import jcy_client, jcy_engine, jcy_runtime, jcy_snapshot

SNAPSHOT_DELAY = 0.25
START_RESOLUTION = 64

_sessions = weakref.WeakSet()


def _tag_rendered_viewports():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type == 'VIEW_3D' and space.shading.type == 'RENDERED':
                    area.tag_redraw()
                    break


class ViewSession:
    def __init__(self, engine):
        self.engine_ref = weakref.ref(engine)
        self.worker = None
        self.snapshot = None
        self.snapshot_pending = True
        self.snapshot_due = 0.0
        self.loading = False
        self.load_request = None
        self.last_params = None
        self.view_id = 0
        self.sent_view_id = -1
        self.frame = None
        self.texture = None
        self.texture_key = None
        self.message = "Starting JCycles..."
        self.error = None
        self.scene_name = None
        self.view_layer_name = None
        self.alive = True
        self.old_snapshots = []
        _sessions.add(self)
        bpy.app.timers.register(self._tick, first_interval=0.05)

    # Lifetime -------------------------------------------------------------------

    def free(self):
        if not self.alive:
            return
        self.alive = False
        worker = self.worker
        self.worker = None
        snapshots = self.old_snapshots + ([self.snapshot] if self.snapshot else [])
        self.texture = None

        def cleanup():
            if worker is not None:
                worker.stop()
            for snapshot in snapshots:
                jcy_snapshot.remove_snapshot(snapshot[1])

        import threading
        threading.Thread(target=cleanup, daemon=True).start()

    # Scene changes ----------------------------------------------------------------

    def update(self, context, depsgraph):
        self.scene_name = depsgraph.scene.name
        self.view_layer_name = depsgraph.view_layer.name
        if self.snapshot is None or len(depsgraph.updates):
            self.snapshot_pending = True
            self.snapshot_due = time.monotonic() + (0.0 if self.snapshot is None else SNAPSHOT_DELAY)

    def _ensure_worker(self):
        autoexec = jcy_engine.autoexec_enabled()
        if self.worker is not None and (not self.worker.is_alive() or self.worker.autoexec != autoexec):
            self.worker.stop(kill=True)
            self.worker = None
        if self.worker is None:
            self.message = "Starting JCycles render process..."
            self.worker = jcy_client.Worker("viewport", autoexec=autoexec, echo=jcy_engine.echo_worker_output())
            self.worker.start()
            self.loading = False
            self.load_request = None
            self.sent_view_id = -1
        return self.worker

    def _take_snapshot(self):
        self.snapshot_pending = False
        worker = self._ensure_worker()
        if self.snapshot is not None:
            self.old_snapshots.append(self.snapshot)
        self.snapshot = jcy_snapshot.write_snapshot(os.path.join(jcy_runtime.session_dir(), "viewport"))
        token, path, overrides = self.snapshot
        self.load_request = worker.send({"cmd": "load", "path": path, "token": token, "autoexec": worker.autoexec,
                                         "image_overrides": overrides})
        self.loading = True
        self.message = "Loading scene..."
        # Force a new render once loaded.
        self.sent_view_id = -1

    # Main-thread timer: messages from the worker --------------------------------------

    def _tick(self):
        if not self.alive:
            return None
        try:
            now = time.monotonic()
            if self.snapshot_pending and now >= self.snapshot_due and not self.loading:
                self._take_snapshot()
            if self.worker is not None:
                if not self.worker.is_alive():
                    self.error = "JCycles render process stopped:\n" + self.worker.tail()
                    self.worker = None
                    _tag_rendered_viewports()
                else:
                    self._handle_messages()
            if not self.loading and self.last_params is not None and self.sent_view_id != self.view_id:
                self._send_view()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            self.error = str(ex)
            _tag_rendered_viewports()
        return 0.03

    def _handle_messages(self):
        redraw = False
        for message in self.worker.poll_messages():
            kind = message.get("type")
            if kind == "reply":
                if message.get("id") == self.load_request:
                    self.loading = False
                    self.load_request = None
                    if message.get("ok"):
                        self.worker.loaded_token = self.snapshot[0] if self.snapshot else None
                        self.error = None
                        for old in self.old_snapshots:
                            jcy_snapshot.remove_snapshot(old[1])
                        self.old_snapshots = []
                    else:
                        self.error = message.get("error", "Loading the scene failed")
                    redraw = True
                elif not message.get("ok", True):
                    self.error = message.get("error", "Viewport render failed")
                    redraw = True
            elif kind == "event":
                name = message.get("event")
                if name == "view_frame":
                    previous = self.frame
                    self.frame = message
                    self.error = None
                    if previous and previous.get("path") != message.get("path"):
                        try:
                            os.remove(previous["path"])
                        except OSError:
                            pass
                    redraw = True
                elif name in {"view_error", "view_warning"}:
                    if name == "view_error":
                        self.error = message.get("text")
                    else:
                        print("JCycles:", message.get("text"))
                    redraw = True
                elif name == "stats":
                    self.message = message.get("text", "")
        if redraw:
            engine = self.engine_ref()
            if engine is not None:
                try:
                    engine.tag_redraw()
                except Exception:
                    pass
            _tag_rendered_viewports()

    # View parameters --------------------------------------------------------------

    def _view_params(self, context, depsgraph):
        region = context.region
        rv3d = context.region_data
        space = context.space_data
        scene = depsgraph.scene
        params = {
            "scene": scene.name,
            "view_layer": depsgraph.view_layer.name,
            "region": [region.width, region.height],
            "render_pass": 'COMBINED',
            "start_resolution": START_RESOLUTION,
        }
        shading = getattr(space.shading, "jcycles", None)
        if shading is not None:
            params["render_pass"] = shading.render_pass
        visible = []
        for obj in depsgraph.objects:
            original = obj.original
            try:
                if original.visible_get(viewport=space):
                    visible.append(original.name)
            except Exception:
                visible.append(original.name)
        params["visible_objects"] = sorted(set(visible))

        camera = scene.camera
        if rv3d.view_perspective == 'CAMERA' and camera is not None:
            from bpy_extras import view3d_utils
            frame = [camera.matrix_world @ v for v in camera.original.data.view_frame(scene=scene.original)]
            points = [view3d_utils.location_3d_to_region_2d(region, rv3d, p) for p in frame]
            if all(p is not None for p in points):
                xs = [p.x for p in points]
                ys = [p.y for p in points]
                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                width = max(1, int(round(x1 - x0)))
                height = max(1, int(round(y1 - y0)))
                params.update(mode='CAMERA', camera=camera.original.name, width=width, height=height,
                              rect=[x0, y0, x1 - x0, y1 - y0])
                render = scene.render
                if render.use_border:
                    params["border"] = [render.border_min_x, render.border_min_y,
                                        render.border_max_x, render.border_max_y]
                    params["rect"] = [x0 + (x1 - x0) * render.border_min_x,
                                      y0 + (y1 - y0) * render.border_min_y,
                                      (x1 - x0) * (render.border_max_x - render.border_min_x),
                                      (y1 - y0) * (render.border_max_y - render.border_min_y)]
                return params
        params.update(
            mode='VIEW', width=region.width, height=region.height, rect=[0, 0, region.width, region.height],
            window_matrix=[list(row) for row in rv3d.window_matrix],
            view_matrix=[list(row) for row in rv3d.view_matrix],
            is_perspective=bool(rv3d.is_perspective),
            clip_start=space.clip_start, clip_end=space.clip_end,
        )
        return params

    def _send_view(self):
        worker = self.worker
        if worker is None or not worker.is_alive() or worker.loaded_token is None:
            return
        scene = bpy.data.scenes.get(self.last_params["scene"])
        if scene is not None and getattr(getattr(scene, "jcycles", None), "preview_pause", False):
            self.message = "Paused"
            self.sent_view_id = self.view_id
            return
        command = dict(self.last_params)
        command.pop("rect", None)
        command.pop("region", None)
        command.update(cmd="view", token=worker.loaded_token, view_id=self.view_id,
                       frame=scene.frame_current if scene else 1,
                       subframe=scene.frame_subframe if scene else 0.0,
                       device=jcy_engine.device_settings())
        worker.send(command)
        self.sent_view_id = self.view_id

    # Drawing ----------------------------------------------------------------------

    def draw(self, context, depsgraph):
        params = self._view_params(context, depsgraph)
        compare = dict(params)
        if params != self.last_params:
            changed_render = self.last_params is None or \
                {k: v for k, v in compare.items() if k != "rect"} != \
                {k: v for k, v in self.last_params.items() if k != "rect"}
            self.last_params = params
            if changed_render:
                self.view_id += 1

        self._draw_image(context, depsgraph, params)
        status = self.error or ("" if self.frame else self.message)
        engine = self.engine_ref()
        if engine is not None:
            if self.frame is not None and not self.error:
                samples = self.frame.get("samples", 0)
                text = "Sample {:d}".format(samples) if samples else "Preview"
                engine.update_stats("", "JCycles | " + text)
            elif status:
                engine.update_stats("", "JCycles | " + status.splitlines()[0])
        if self.error:
            self._draw_text(context, self.error.splitlines()[0][:200])
        elif self.frame is None:
            self._draw_text(context, self.message)

    def _draw_image(self, context, depsgraph, params):
        frame = self.frame
        if frame is None:
            return
        import gpu
        import numpy as np
        key = frame["path"]
        if self.texture_key != key:
            try:
                pixels = np.fromfile(frame["path"], dtype=np.float32)
            except OSError:
                pixels = None
            if pixels is not None and pixels.size == frame["width"] * frame["height"] * 4:
                buffer = gpu.types.Buffer('FLOAT', pixels.size, pixels)
                self.texture = gpu.types.GPUTexture((frame["width"], frame["height"]), format='RGBA16F', data=buffer)
                self.texture_key = key
        if self.texture is None:
            return
        from gpu_extras.presets import draw_texture_2d
        x, y, w, h = params["rect"]
        engine = self.engine_ref()
        gpu.state.blend_set('ALPHA_PREMULT')
        engine.bind_display_space_shader(depsgraph.scene)
        draw_texture_2d(self.texture, (x, y), w, h)
        engine.unbind_display_space_shader()
        gpu.state.blend_set('NONE')

    def _draw_text(self, context, text):
        try:
            import blf
            font = 0
            blf.size(font, 13 * context.preferences.system.ui_scale)
            blf.color(font, 1.0, 1.0, 1.0, 0.9)
            blf.position(font, 20, context.region.height - 60, 0)
            blf.draw(font, text)
        except Exception:
            pass


def free_all():
    for session in list(_sessions):
        session.free()
