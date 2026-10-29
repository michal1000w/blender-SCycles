# Runs in an official Blender with JCycles installed: drives the worker's viewport mode directly.
#   blender -b --factory-startup <scene.blend> --python worker_view_test.py -- <render_pass> <denoise 0|1>
import os
import sys
import time

import addon_utils
import bpy
import numpy as np

render_pass, denoise = sys.argv[sys.argv.index("--") + 1:][:2]
module = addon_utils.enable("JCycles", default_set=True)
from JCycles import jcy_client, jcy_engine, jcy_runtime, jcy_snapshot  # noqa: E402

scene = bpy.context.scene
scene.render.engine = 'JCYCLES'
scene.jcycles.preview_samples = 16
scene.jcycles.use_preview_denoising = denoise == "1"
scene.jcycles.preview_denoiser = 'OPENIMAGEDENOISE'
scene.jcycles.preview_denoising_start_sample = 1
depsgraph = bpy.context.evaluated_depsgraph_get()
cam = scene.camera
width, height = 320, 240
window = cam.calc_matrix_camera(depsgraph, x=width, y=height)
view = cam.matrix_world.inverted()

snapshot = jcy_snapshot.write_snapshot(os.path.join(jcy_runtime.session_dir(), "viewport"))
worker = jcy_client.Worker("viewport")
t0 = time.monotonic()
worker.start()
worker.call({"cmd": "load", "path": snapshot[1], "token": snapshot[0], "image_overrides": snapshot[2]})
worker.loaded_token = snapshot[0]
print("VIEW load seconds", round(time.monotonic() - t0, 2))
command = {
    "cmd": "view", "token": snapshot[0], "view_id": 1, "scene": scene.name,
    "view_layer": scene.view_layers[0].name, "render_pass": render_pass, "start_resolution": 64,
    "visible_objects": sorted(o.name for o in scene.objects), "mode": 'VIEW',
    "width": width, "height": height, "window_matrix": [list(r) for r in window],
    "view_matrix": [list(r) for r in view], "is_perspective": True,
    "clip_start": cam.data.clip_start, "clip_end": cam.data.clip_end,
    "frame": 1, "subframe": 0.0, "device": jcy_engine.device_settings(),
}
t0 = time.monotonic()
worker.send(command)
frames = []
warnings = []
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    for message in worker.poll_messages():
        if message.get("type") == "event" and message.get("event") == "view_frame":
            frames.append((round(time.monotonic() - t0, 3), message["samples"], message["width"], message["height"]))
            last = message
        elif message.get("type") == "event" and message.get("event") in {"view_error", "view_warning"}:
            warnings.append(message["text"])
        elif message.get("type") == "reply" and not message.get("ok"):
            warnings.append(message.get("error"))
    if frames and frames[-1][1] >= 16 or any("error" in w.lower() for w in warnings):
        break
    time.sleep(0.02)
print("VIEW frames (seconds, samples, w, h):", frames)
print("VIEW warnings:", warnings)
pixels = np.fromfile(last["path"], dtype=np.float32).reshape(last["height"], last["width"], 4)
print("VIEW final mean", pixels[:, :, :3].mean().round(4), "nonzero", bool(np.any(pixels[:, :, :3] > 0)),
      "finite", bool(np.isfinite(pixels).all()))
np.save(os.path.join(os.path.dirname(bpy.data.filepath), "view_{:s}_{:s}.npy".format(render_pass, denoise)), pixels)
worker.stop()
