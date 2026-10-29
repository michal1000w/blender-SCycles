# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
JCycles render worker.

Runs inside the Blender build of the blender-SCycles fork that is bundled with the add-on:

  blender -b --factory-startup --python worker_main.py -- --addon-dir DIR --session-dir DIR

The official Blender that hosts the JCycles add-on sends one JSON command per line on stdin.
Replies and events are single stdout lines starting with MESSAGE_PREFIX; any other output is
regular Blender / Cycles logging. Everything that is rendered here is rendered by the real
Cycles of the fork, with the real `scene.cycles` settings and the real fork shader nodes.
"""

import json
import os
import queue
import sys
import threading
import time
import traceback

import bpy

MESSAGE_PREFIX = "@@JCYCLES@@"

# Python-defined property groups of the JCycles add-on and the Cycles groups they stand for.
PROPERTY_GROUP_MAP = {
    "jcycles": "cycles",
    "jcycles_curves": "cycles_curves",
    "jcycles_visibility": "cycles_visibility",
    "jcycles_custom": "cycles_custom",
}

_stdout_lock = threading.Lock()
_commands = queue.Queue()
_args = {}
_state = {
    "loaded_token": None,
    "loaded_path": None,
    "node_specs": None,
    "view": None,
    "stats_time": 0.0,
}


# -----------------------------------------------------------------------------
# Messaging


def send(message):
    line = MESSAGE_PREFIX + json.dumps(message) + "\n"
    with _stdout_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def reply(command, **data):
    data["type"] = "reply"
    data["id"] = command.get("id")
    send(data)


def event(command, name, **data):
    data["type"] = "event"
    data["event"] = name
    data["id"] = command.get("id") if command else None
    send(data)


def stdin_reader():
    stream = sys.stdin
    while True:
        line = stream.readline()
        if not line:
            # Host closed the pipe (host exited or crashed): quit.
            _commands.put({"cmd": "quit", "id": None})
            return
        line = line.strip()
        if not line:
            continue
        try:
            _commands.put(json.loads(line))
        except Exception as ex:
            send({"type": "log", "level": "ERROR", "text": "Invalid command: {!r} ({:s})".format(line[:200], str(ex))})


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv):
            args[argv[i][2:].replace("-", "_")] = argv[i + 1]
            i += 2
        else:
            i += 1
    return args


# -----------------------------------------------------------------------------
# Progress reporting during renders


_current_command = [None]


@bpy.app.handlers.persistent
def render_stats_handler(text, *_args_unused):
    # Called from Cycles' session thread, keep it light.
    now = time.monotonic()
    if now - _state["stats_time"] < 0.1:
        return
    _state["stats_time"] = now
    try:
        event(_current_command[0], "stats", text=str(text))
    except Exception:
        pass


def install_handlers():
    handlers = bpy.app.handlers.render_stats
    if render_stats_handler not in handlers:
        handlers.append(render_stats_handler)


# -----------------------------------------------------------------------------
# Loading a snapshot of the host file


def system_properties(owner):
    getter = getattr(owner, "bl_system_properties_get", None)
    if getter is None:
        return None
    try:
        return getter(do_create=False)
    except TypeError:
        return getter()


def copy_property_groups(owner):
    """Make the Cycles settings of `owner` exactly the JCycles settings saved by the host.

    Both groups are defined by the same `properties.py` of the fork, so their raw storage has the
    same meaning. Settings never changed in JCycles are absent and therefore take their default,
    so leftovers of the official Cycles add-on of the host never leak into the render."""
    for storage in (system_properties(owner), owner):
        if storage is None:
            continue
        try:
            keys = set(storage.keys())
        except Exception:
            continue
        for src, dst in PROPERTY_GROUP_MAP.items():
            if src not in keys and dst not in keys:
                continue
            if dst in keys:
                del storage[dst]
            if src in keys:
                storage[dst] = storage[src]


def all_property_owners():
    collections = (
        bpy.data.scenes, bpy.data.objects, bpy.data.materials, bpy.data.lights, bpy.data.worlds,
        bpy.data.meshes, bpy.data.curves, bpy.data.metaballs, bpy.data.cameras,
    )
    for collection in collections:
        for owner in collection:
            if owner.library is not None and not owner.is_editable:
                # Linked data keeps the settings of its library file, which was saved by the host
                # too, so convert it as well (in memory only).
                pass
            yield owner
    for scene in bpy.data.scenes:
        for view_layer in scene.view_layers:
            yield view_layer


def load_node_specs():
    if _state["node_specs"] is None:
        path = os.path.join(_args["addon_dir"], "node_specs.json")
        with open(path, encoding="utf-8") as fh:
            _state["node_specs"] = json.load(fh)
    return _state["node_specs"]


def import_addon_module(name):
    import importlib.util
    path = os.path.join(_args["addon_dir"], name + ".py")
    module_name = "jcycles_worker_" + name
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_node_classes_registered = [False]


def register_node_classes():
    if _node_classes_registered[0]:
        return
    nodedefs = import_addon_module("jcy_nodedefs")
    specs = load_node_specs()
    for cls in nodedefs.make_node_classes(specs, with_ui=False):
        bpy.utils.register_class(cls)
    _node_classes_registered[0] = True


def convert_nodes():
    nodedefs = import_addon_module("jcy_nodedefs")
    specs = load_node_specs()
    converted = 0
    trees = []
    for collection in (bpy.data.materials, bpy.data.worlds, bpy.data.lights):
        for owner in collection:
            tree = getattr(owner, "node_tree", None)
            if tree is not None:
                trees.append(tree)
    for tree in bpy.data.node_groups:
        if tree.bl_idname == 'ShaderNodeTree':
            trees.append(tree)
    for tree in trees:
        converted += nodedefs.convert_tree_to_native(tree, specs)
    return converted


def replace_images(overrides):
    for name, info in overrides.items():
        image = bpy.data.images.get(name)
        if image is None:
            continue
        try:
            new_image = bpy.data.images.load(info["path"], check_existing=False)
            new_image.colorspace_settings.name = image.colorspace_settings.name
            new_image.alpha_mode = image.alpha_mode
            if hasattr(new_image, "use_half_precision"):
                new_image.use_half_precision = image.use_half_precision
            image.user_remap(new_image)
        except Exception as ex:
            send({"type": "log", "level": "WARNING",
                  "text": "Could not use the unsaved pixels of image {!r}: {:s}".format(name, str(ex))})


def cmd_load(command):
    path = command["path"]
    token = command["token"]
    t0 = time.monotonic()
    _state["view"] = None
    bpy.ops.wm.open_mainfile(filepath=path, load_ui=False, use_scripts=bool(command.get("autoexec")))
    install_handlers()
    for owner in all_property_owners():
        copy_property_groups(owner)
    register_node_classes()
    converted = convert_nodes()
    replace_images(command.get("image_overrides", {}))
    _state["loaded_token"] = token
    _state["loaded_path"] = path
    reply(command, ok=True, converted_nodes=converted, seconds=time.monotonic() - t0,
          scenes=[scene.name for scene in bpy.data.scenes])


# -----------------------------------------------------------------------------
# Devices


DEVICE_TYPES = ('CUDA', 'OPTIX', 'HIP', 'METAL', 'ONEAPI')


def cycles_preferences():
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        import addon_utils
        addon_utils.enable("cycles", default_set=True)
        addon = bpy.context.preferences.addons["cycles"]
    return addon.preferences


def cmd_devices(command):
    import _cycles
    result = {"device_types": list(_cycles.get_device_types()), "devices": {}}
    for device_type in ('CPU',) + DEVICE_TYPES:
        try:
            result["devices"][device_type] = [list(device) for device in _cycles.available_devices(device_type)]
        except Exception as ex:
            result["devices"][device_type] = []
            result.setdefault("errors", []).append("{:s}: {:s}".format(device_type, str(ex)))
    reply(command, ok=True, **result)


def apply_device_settings(settings):
    if not settings:
        return
    prefs = cycles_preferences()
    for key, value in settings.get("preferences", {}).items():
        if key in {"devices", "rna_type", "bl_idname"}:
            continue
        try:
            setattr(prefs, key, value)
        except Exception:
            pass
    try:
        prefs.refresh_devices()
    except Exception:
        pass
    use = {(entry[0], entry[1]): entry[2] for entry in settings.get("devices", [])}
    for entry in prefs.devices:
        key = (entry.type, entry.id)
        if key in use:
            entry.use = bool(use[key])
    experimental = settings.get("experimental", {})
    try:
        bpy.context.preferences.experimental.use_cycles_debug = bool(experimental.get("use_cycles_debug", False))
        bpy.context.preferences.view.show_developer_ui = bool(experimental.get("show_developer_ui", False))
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Output format


def set_multilayer_exr(scene):
    settings = scene.render.image_settings
    if hasattr(settings, "media_type"):
        try:
            settings.media_type = 'MULTI_LAYER_IMAGE'
        except Exception:
            pass
    settings.file_format = 'OPEN_EXR_MULTILAYER'
    settings.color_depth = '32'
    for codec in ('NONE', 'ZIP'):
        try:
            settings.exr_codec = codec
            break
        except Exception:
            continue
    if hasattr(settings, "use_preview"):
        settings.use_preview = False


def require_loaded(command):
    if _state["loaded_token"] is None or command.get("token") not in (None, _state["loaded_token"]):
        raise RuntimeError("Scene snapshot {!r} is not loaded".format(command.get("token")))


def prepare_scene_for_render(command):
    scene = bpy.data.scenes[command["scene"]]
    render = scene.render
    render.engine = 'CYCLES'
    apply_device_settings(command.get("device"))
    scene.frame_set(int(command["frame"]), subframe=float(command.get("subframe", 0.0)))
    camera = command.get("camera")
    if camera:
        scene.camera = bpy.data.objects[camera]
    render.use_compositing = False
    render.use_sequencer = False
    render.use_stamp = False
    if hasattr(render, "use_lock_interface"):
        render.use_lock_interface = False
    return scene


# -----------------------------------------------------------------------------
# Final render


def cmd_render(command):
    require_loaded(command)
    scene = prepare_scene_for_render(command)
    render = scene.render
    layers = set(command["layers"])
    for view_layer in scene.view_layers:
        view_layer.use = view_layer.name in layers
    render.use_single_layer = False
    # The host's render result has the size of the border region.
    render.use_crop_to_border = True
    set_multilayer_exr(scene)

    output = command["output"]
    _current_command[0] = command
    t0 = time.monotonic()
    try:
        result = bpy.ops.render.render(scene=scene.name, write_still=False)
    finally:
        _current_command[0] = None
    if 'CANCELLED' in result:
        raise RuntimeError("Render was cancelled or failed, see the worker log")
    image = bpy.data.images["Render Result"]
    image.save_render(output, scene=scene)
    if not os.path.exists(output):
        raise RuntimeError("The render result was not written to " + output)
    reply(command, ok=True, output=output, seconds=time.monotonic() - t0)


# -----------------------------------------------------------------------------
# Viewport render (progressive)

# Viewport render passes of the Cycles add-on: how to enable them and the pass to show.
VIEW_PASSES = {
    'EMISSION': (("vl", "use_pass_emit"), "Emission"),
    'BACKGROUND': (("vl", "use_pass_environment"), "Environment"),
    'AO': (("vl", "use_pass_ambient_occlusion"), "Ambient Occlusion"),
    'SHADOW_CATCHER': (("cvl", "use_pass_shadow_catcher"), "Shadow Catcher"),
    'DIFFUSE_DIRECT': (("vl", "use_pass_diffuse_direct"), "Diffuse Direct"),
    'DIFFUSE_INDIRECT': (("vl", "use_pass_diffuse_indirect"), "Diffuse Indirect"),
    'DIFFUSE_COLOR': (("vl", "use_pass_diffuse_color"), "Diffuse Color"),
    'GLOSSY_DIRECT': (("vl", "use_pass_glossy_direct"), "Glossy Direct"),
    'GLOSSY_INDIRECT': (("vl", "use_pass_glossy_indirect"), "Glossy Indirect"),
    'GLOSSY_COLOR': (("vl", "use_pass_glossy_color"), "Glossy Color"),
    'TRANSMISSION_DIRECT': (("vl", "use_pass_transmission_direct"), "Transmission Direct"),
    'TRANSMISSION_INDIRECT': (("vl", "use_pass_transmission_indirect"), "Transmission Indirect"),
    'TRANSMISSION_COLOR': (("vl", "use_pass_transmission_color"), "Transmission Color"),
    'VOLUME_DIRECT': (("cvl", "use_pass_volume_direct"), "Volume Direct"),
    'VOLUME_INDIRECT': (("cvl", "use_pass_volume_indirect"), "Volume Indirect"),
    'POSITION': (("vl", "use_pass_position"), "Position"),
    'NORMAL': (("vl", "use_pass_normal"), "Normal"),
    'UV': (("vl", "use_pass_uv"), "UV"),
    'MIST': (("vl", "use_pass_mist"), "Mist"),
    'DENOISING_ALBEDO': (("cvl", "denoising_store_passes"), "Denoising Albedo"),
    'DENOISING_SPECULAR_ALBEDO': (("cvl", "denoising_store_passes"), "Denoising Specular Albedo"),
    'DENOISING_NORMAL': (("cvl", "denoising_store_passes"), "Denoising Normal"),
    'DENOISING_ROUGHNESS': (("cvl", "denoising_store_passes"), "Denoising Roughness"),
    'DENOISING_DEPTH': (("cvl", "denoising_store_passes"), "Denoising Depth"),
    'DENOISING_BACKWARD_MOTION': (("cvl", "denoising_store_passes"), "Denoising Backward Motion"),
    'DENOISING_SPECULAR_MOTION': (("cvl", "denoising_store_passes"), "Denoising Specular Motion"),
    'SAMPLE_COUNT': (("scene", "pass_debug_sample_count"), "Debug Sample Count"),
}

# Settings that Cycles reads differently for viewport renders: final setting <- viewport setting.
VIEW_SETTINGS = (
    ("samples", "preview_samples"),
    ("use_adaptive_sampling", "use_preview_adaptive_sampling"),
    ("adaptive_threshold", "preview_adaptive_threshold"),
    ("adaptive_min_samples", "preview_adaptive_min_samples"),
    ("dicing_rate", "preview_dicing_rate"),
    ("volume_step_rate", "volume_preview_step_rate"),
    ("texture_limit_render", "texture_limit"),
    ("ao_bounces_render", "ao_bounces"),
    ("texture_resolution_render", "texture_resolution"),
    ("denoiser", "preview_denoiser"),
    ("denoising_prefilter", "preview_denoising_prefilter"),
    ("denoising_quality", "preview_denoising_quality"),
    ("denoising_input_passes", "preview_denoising_input_passes"),
    ("denoising_use_gpu", "preview_denoising_use_gpu"),
)


def apply_viewport_evaluation(scene, view_layer, visible_objects):
    """Make the final render evaluate the scene like the 3D viewport does."""
    cscene = scene.cycles
    for final, preview in VIEW_SETTINGS:
        if hasattr(cscene, final) and hasattr(cscene, preview):
            try:
                setattr(cscene, final, getattr(cscene, preview))
            except Exception:
                pass
    if not getattr(cscene, "preview_scrambling_distance", True) and hasattr(cscene, "auto_scrambling_distance"):
        cscene.auto_scrambling_distance = False
        if hasattr(cscene, "scrambling_distance"):
            cscene.scrambling_distance = 1.0
    render = scene.render
    for final, preview in (
        ("simplify_subdivision_render", "simplify_subdivision"),
        ("simplify_child_particles_render", "simplify_child_particles"),
        ("simplify_volumes_render", "simplify_volumes"),
    ):
        if hasattr(render, final) and hasattr(render, preview):
            setattr(render, final, getattr(render, preview))

    visible = set(visible_objects) if visible_objects is not None else None
    in_layer = {obj.name for obj in view_layer.objects}
    for collection in bpy.data.collections:
        collection.hide_render = collection.hide_viewport
    for obj in bpy.data.objects:
        if visible is not None:
            if obj.name in visible:
                obj.hide_render = False
            elif obj.name in in_layer:
                obj.hide_render = True
            else:
                obj.hide_render = obj.hide_viewport
        for modifier in obj.modifiers:
            try:
                modifier.show_render = modifier.show_viewport
                if modifier.type in {'SUBSURF', 'MULTIRES'}:
                    if modifier.type == 'SUBSURF':
                        modifier.render_levels = modifier.levels
                    else:
                        modifier.render_levels = modifier.levels
            except Exception:
                pass
    for settings in bpy.data.particles:
        try:
            settings.rendered_child_count = settings.child_percent
        except Exception:
            pass


def make_view_camera(scene, params):
    import mathutils
    cam_data = bpy.data.cameras.get("JCyclesViewportCamera") or bpy.data.cameras.new("JCyclesViewportCamera")
    cam_obj = bpy.data.objects.get("JCyclesViewportCamera")
    if cam_obj is None:
        cam_obj = bpy.data.objects.new("JCyclesViewportCamera", cam_data)
        scene.collection.objects.link(cam_obj)
    width, height = params["width"], params["height"]
    projection = mathutils.Matrix(params["window_matrix"])
    view = mathutils.Matrix(params["view_matrix"])
    world = view.inverted()
    aspect = height / width
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.sensor_width = 36.0
    if params["is_perspective"]:
        cam_data.type = 'PERSP'
        cam_data.lens = 36.0 * projection[0][0] / 2.0
        cam_data.shift_x = projection[0][2] / 2.0
        cam_data.shift_y = projection[1][2] * aspect / 2.0
        cam_data.clip_start = max(params["clip_start"], 1e-6)
        cam_data.clip_end = max(params["clip_end"], cam_data.clip_start * 2.0)
    else:
        cam_data.type = 'ORTHO'
        cam_data.ortho_scale = 2.0 / projection[0][0]
        cam_data.shift_x = -projection[0][3] / 2.0
        cam_data.shift_y = -projection[1][3] * aspect / 2.0
        # Near and far planes of the viewport projection, possibly behind the view origin.
        near = (projection[2][3] + 1.0) / projection[2][2]
        far = (projection[2][3] - 1.0) / projection[2][2]
        if near > far:
            near, far = far, near
        back = 0.0
        if near < 1e-3:
            back = 1e-3 - near
        forward = world.to_3x3() @ mathutils.Vector((0.0, 0.0, 1.0))
        world = mathutils.Matrix.Translation(forward.normalized() * back) @ world
        cam_data.clip_start = near + back
        cam_data.clip_end = far + back
    cam_obj.matrix_world = world
    cam_data.dof.use_dof = False
    return cam_obj


def read_exr_passes(path, layer_name, passes):
    """Read named passes of a multilayer EXR as float32 arrays (rows bottom to top).

    Blender writes one EXR part per pass, so all parts are searched."""
    import numpy as np
    import OpenImageIO as oiio
    image = oiio.ImageInput.open(path)
    if image is None:
        raise RuntimeError("Cannot open " + path + ": " + oiio.geterror())
    channels = {}
    width = height = 0
    try:
        part = 0
        while image.seek_subimage(part, 0):
            spec = image.spec()
            width, height = spec.width, spec.height
            names = list(spec.channelnames)
            wanted = any(name.startswith(layer_name + "." + p + ".") for p in passes for name in names)
            if wanted:
                pixels = image.read_image(part, 0, 0, spec.nchannels, oiio.FLOAT)
                pixels = np.asarray(pixels, dtype=np.float32).reshape(spec.height, spec.width, spec.nchannels)
                for i, name in enumerate(names):
                    channels[name] = pixels[::-1, :, i]
            part += 1
    finally:
        image.close()
    result = {}
    for pass_name in passes:
        prefix = layer_name + "." + pass_name + "."
        names = [name for name in channels if name.startswith(prefix) and "." not in name[len(prefix):]]
        if not names:
            continue
        order = "RGBAXYZWUVrgba"
        names.sort(key=lambda n: order.find(n[len(prefix):]) if n[len(prefix):] in order else 99)
        stacked = np.stack([channels[name] for name in names], axis=-1)
        result[pass_name] = (np.ascontiguousarray(stacked), [name[len(prefix):] for name in names])
    return result, width, height


def to_rgba(array, channels):
    import numpy as np
    h, w, n = array.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    if n == 1:
        out[:, :, 0] = out[:, :, 1] = out[:, :, 2] = array[:, :, 0]
        out[:, :, 3] = 1.0
        return out
    order = {c: i for i, c in enumerate(channels)}
    for out_index, keys in enumerate((("R", "X", "U"), ("G", "Y", "V"), ("B", "Z", "A" if "U" in order else ""))):
        for key in keys:
            if key and key in order:
                out[:, :, out_index] = array[:, :, order[key]]
                break
    out[:, :, 3] = array[:, :, order["A"]] if ("A" in order and n == 4 and "U" not in order) else 1.0
    return out


def write_denoise_input(path, layer_name, combined, normal, albedo, samples):
    import numpy as np
    import OpenImageIO as oiio
    h, w, _ = combined.shape
    channels = []
    data = []
    # The standalone denoiser also wants motion vectors (only used across frames): zero here.
    motion = np.zeros((h, w, 4), dtype=np.float32)
    for pass_name, array, names in (
        ("Combined", combined, "RGBA"),
        ("Denoising Normal", normal, "XYZ"),
        ("Denoising Albedo", albedo, "RGB"),
        ("Vector", motion, "XYZW"),
    ):
        if array is None:
            continue
        for i, chan in enumerate(names):
            channels.append("{:s}.{:s}.{:s}".format(layer_name, pass_name, chan))
            data.append(array[:, :, i])
    stacked = np.stack(data, axis=-1)[::-1].astype(np.float32)
    spec = oiio.ImageSpec(w, h, len(channels), oiio.FLOAT)
    spec.channelnames = channels
    spec.attribute("compression", "none")
    spec.attribute("cycles.{:s}.samples".format(layer_name), str(samples))
    out = oiio.ImageOutput.create(path)
    if out is None or not out.open(path, spec):
        raise RuntimeError("Cannot write " + path)
    out.write_image(stacked)
    out.close()


def denoise_accumulated(scene, view_layer, combined, normal, albedo, samples, session_dir):
    import _cycles
    in_path = os.path.join(session_dir, "view_denoise_in.exr")
    out_path = os.path.join(session_dir, "view_denoise_out.exr")
    write_denoise_input(in_path, view_layer.name, combined, normal, albedo, samples)
    if os.path.exists(out_path):
        os.remove(out_path)
    _cycles.denoise(bpy.context.preferences.as_pointer(), scene.as_pointer(), view_layer.as_pointer(),
                    input=[in_path], output=[out_path])
    passes, _w, _h = read_exr_passes(out_path, view_layer.name, ["Combined"])
    array, names = passes["Combined"]
    return to_rgba(array, names)


def setup_view_session(command):
    """Prepare the loaded scene for progressive viewport rendering. Done once per snapshot."""
    scene = bpy.data.scenes[command["scene"]]
    view_layer = scene.view_layers[command["view_layer"]]
    prepare_scene_for_render(command)
    render = scene.render
    for layer in scene.view_layers:
        layer.use = layer == view_layer
    render.use_single_layer = False
    render.use_persistent_data = True
    render.use_border = False
    render.use_crop_to_border = True
    render.resolution_percentage = 100
    render.pixel_aspect_x = render.pixel_aspect_y = 1.0
    if hasattr(render, "use_motion_blur") and not command.get("motion_blur_in_viewport", False):
        render.use_motion_blur = False
    apply_viewport_evaluation(scene, view_layer, command.get("visible_objects"))
    cscene = scene.cycles
    denoise = bool(getattr(cscene, "use_preview_denoising", False))
    cscene.use_denoising = False
    view_layer.cycles.use_denoising = False
    set_multilayer_exr(scene)
    display_pass = command.get("render_pass", 'COMBINED')
    if display_pass in VIEW_PASSES:
        (owner, prop), _name = VIEW_PASSES[display_pass]
        target = {"vl": view_layer, "cvl": view_layer.cycles, "scene": cscene}[owner]
        if hasattr(target, prop):
            setattr(target, prop, True)
    if denoise and display_pass == 'COMBINED':
        view_layer.cycles.denoising_store_passes = True
    target_samples = int(getattr(cscene, "samples", 0))
    return {
        "scene": scene,
        "view_layer": view_layer,
        "denoise": denoise and display_pass == 'COMBINED',
        "denoise_start": int(getattr(cscene, "preview_denoising_start_sample", 1)),
        "pass_name": VIEW_PASSES[display_pass][1] if display_pass in VIEW_PASSES else "Combined",
        "target_samples": target_samples if target_samples > 0 else (1 << 24),
        "seed": int(getattr(cscene, "seed", 0)),
    }


def apply_view_camera(session, params):
    scene = session["scene"]
    render = scene.render
    if params["mode"] == 'CAMERA':
        scene.camera = bpy.data.objects[params["camera"]]
        render.resolution_x = max(1, int(params["width"]))
        render.resolution_y = max(1, int(params["height"]))
        if params.get("border"):
            border = params["border"]
            render.use_border = True
            render.border_min_x, render.border_min_y, render.border_max_x, render.border_max_y = border
        else:
            render.use_border = False
    else:
        render.resolution_x = max(1, int(params["width"]))
        render.resolution_y = max(1, int(params["height"]))
        render.use_border = False
        scene.camera = make_view_camera(scene, params)


def cmd_view(command):
    """Progressive viewport render. Runs until another command arrives or all samples are done."""
    import numpy as np
    require_loaded(command)
    session_dir = _args["session_dir"]
    view = _state["view"]
    key = (_state["loaded_token"], command["scene"], command["view_layer"],
           command.get("render_pass"), json.dumps(command.get("visible_objects")))
    if view is None or view["key"] != key:
        view = setup_view_session(command)
        view["key"] = key
        _state["view"] = view
    else:
        apply_device_settings(command.get("device"))
    apply_view_camera(view, command)

    scene = view["scene"]
    cscene = scene.cycles
    view_layer = view["view_layer"]
    use_adaptive = cscene.use_adaptive_sampling
    out_dir = os.path.join(session_dir, "view")
    os.makedirs(out_dir, exist_ok=True)
    reply(command, ok=True)

    done = 0
    accum = {}
    batch = 1
    seq = 0
    start_resolution = int(command.get("start_resolution", 0))
    full_size = (scene.render.resolution_x, scene.render.resolution_y)
    passes = ["Combined"] if view["pass_name"] == "Combined" else [view["pass_name"]]
    if view["denoise"]:
        passes += ["Denoising Normal", "Denoising Albedo"]

    def interrupted():
        return not _commands.empty()

    # A quick low-resolution first image, like Cycles' start resolution in the viewport.
    if start_resolution > 0 and max(full_size) > start_resolution * 1.5:
        factor = start_resolution / max(full_size)
        scene.render.resolution_x = max(1, int(full_size[0] * factor))
        scene.render.resolution_y = max(1, int(full_size[1] * factor))
        cscene.samples = 1
        cscene.sample_offset = 0
        cscene.use_adaptive_sampling = False
        path = os.path.join(out_dir, "low.exr")
        _current_command[0] = command
        try:
            bpy.ops.render.render(scene=scene.name, write_still=False)
        finally:
            _current_command[0] = None
        bpy.data.images["Render Result"].save_render(path, scene=scene)
        data, w, h = read_exr_passes(path, view_layer.name, passes[:1])
        if passes[0] in data:
            seq += 1
            rgba = to_rgba(*data[passes[0]])
            raw = os.path.join(out_dir, "frame_{:d}_{:d}.raw".format(command["view_id"], seq))
            rgba.tofile(raw)
            event(command, "view_frame", path=raw, width=w, height=h, samples=0, view_id=command["view_id"],
                  final=False)
        scene.render.resolution_x, scene.render.resolution_y = full_size
        if interrupted():
            return

    while done < view["target_samples"]:
        n = min(batch, view["target_samples"] - done)
        cscene.samples = n
        cscene.sample_offset = done
        cscene.use_adaptive_sampling = use_adaptive and done == 0 and n >= view["target_samples"]
        path = os.path.join(out_dir, "batch.exr")
        t0 = time.monotonic()
        _current_command[0] = command
        try:
            result = bpy.ops.render.render(scene=scene.name, write_still=False)
        finally:
            _current_command[0] = None
        if 'CANCELLED' in result:
            event(command, "view_error", text="Viewport render failed, see the worker log")
            return
        bpy.data.images["Render Result"].save_render(path, scene=scene)
        elapsed = time.monotonic() - t0
        data, w, h = read_exr_passes(path, view_layer.name, passes)
        for name, (array, channels) in data.items():
            if name in accum:
                accum[name] = (accum[name][0] + array * n, channels)
            else:
                accum[name] = (array * n, channels)
        done += n
        main = accum.get(passes[0])
        if main is None:
            event(command, "view_error", text="Pass {!r} was not rendered".format(passes[0]))
            return
        display = to_rgba(main[0] / done, main[1])
        if view["denoise"] and done >= view["denoise_start"] and "Denoising Normal" in accum:
            try:
                display = denoise_accumulated(
                    scene, view_layer, to_rgba(main[0] / done, main[1]),
                    accum["Denoising Normal"][0] / done, accum["Denoising Albedo"][0] / done,
                    done, session_dir)
            except Exception as ex:
                view["denoise"] = False
                event(command, "view_warning", text="Viewport denoising is unavailable: " + str(ex))
        seq += 1
        raw = os.path.join(out_dir, "frame_{:d}_{:d}.raw".format(command["view_id"], seq))
        np.ascontiguousarray(display, dtype=np.float32).tofile(raw)
        event(command, "view_frame", path=raw, width=w, height=h, samples=done,
              view_id=command["view_id"], final=done >= view["target_samples"])
        if interrupted():
            return
        # Grow the batches while each takes well under a quarter second.
        if elapsed < 0.12:
            batch = min(batch * 2, 64)
        elif elapsed > 0.35 and batch > 1:
            batch = max(1, batch // 2)


# -----------------------------------------------------------------------------
# Other Cycles functions


def cmd_osl_compile(command):
    import _cycles
    ok = _cycles.osl_compile(command["input"], command["output"])
    reply(command, ok=bool(ok))


def cmd_merge(command):
    import _cycles
    ok = _cycles.merge(input=command["input"], output=command["output"])
    reply(command, ok=bool(ok))


def cmd_denoise(command):
    import _cycles
    require_loaded(command)
    scene = bpy.data.scenes[command["scene"]]
    view_layer = scene.view_layers[command["view_layer"]]
    apply_device_settings(command.get("device"))
    ok = _cycles.denoise(bpy.context.preferences.as_pointer(), scene.as_pointer(), view_layer.as_pointer(),
                         input=command["input"], output=command["output"])
    reply(command, ok=bool(ok))


def cmd_bake(command):
    """Bake into float copies of the target images and return their raw pixels."""
    import numpy as np
    require_loaded(command)
    scene = prepare_scene_for_render(command)
    view_layer = scene.view_layers[command["view_layer"]]
    obj = bpy.data.objects[command["object"]]
    targets = []
    replaced = {}
    if command["target"] == 'IMAGE_TEXTURES':
        for slot in obj.material_slots:
            material = slot.material
            if material is None or material.node_tree is None:
                continue
            node = material.node_tree.nodes.active
            if node is None or node.bl_idname != 'ShaderNodeTexImage' or node.image is None:
                continue
            image = node.image
            if image.name not in replaced:
                temp = bpy.data.images.new("JCyclesBake_" + image.name, image.size[0], image.size[1],
                                           alpha=True, float_buffer=True, is_data=True)
                temp.colorspace_settings.is_data = True
                replaced[image.name] = temp
                targets.append(image.name)
            node.image = replaced[image.name]
    bake = scene.render.bake
    # Margins and clearing are applied by the host to its own images.
    bake.margin = 0
    bake.use_clear = True
    # Pass filter, cage, selected-to-active etc. come from the scene's bake settings, like the
    # host's Bake button uses them.
    kwargs = dict(type=command["pass_type"], margin=0, use_clear=True, target=command["target"],
                  save_mode='INTERNAL')
    with bpy.context.temp_override(scene=scene, view_layer=view_layer, object=obj, active_object=obj,
                                   selected_objects=[o for o in view_layer.objects if o.select_get(view_layer=view_layer)]):
        _current_command[0] = command
        try:
            result = bpy.ops.object.bake(**kwargs)
        finally:
            _current_command[0] = None
    if 'FINISHED' not in result:
        raise RuntimeError("Bake failed, see the worker log")
    outputs = []
    for index, name in enumerate(targets):
        temp = replaced[name]
        pixels = np.empty(temp.size[0] * temp.size[1] * 4, dtype=np.float32)
        temp.pixels.foreach_get(pixels)
        path = os.path.join(_args["session_dir"], "bake_{:d}.raw".format(index))
        pixels.tofile(path)
        outputs.append({"image": name, "path": path, "width": temp.size[0], "height": temp.size[1]})
    if command["target"] == 'VERTEX_COLORS':
        mesh = obj.data
        attribute = mesh.color_attributes.active_color
        if attribute is None:
            raise RuntimeError("Object has no active color attribute")
        values = np.empty(len(attribute.data) * 4, dtype=np.float32)
        attribute.data.foreach_get("color", values)
        path = os.path.join(_args["session_dir"], "bake_vertex.raw")
        values.tofile(path)
        outputs.append({"attribute": attribute.name, "path": path, "domain": attribute.domain,
                        "count": len(attribute.data)})
    reply(command, ok=True, outputs=outputs)


def cmd_info(command):
    import _cycles
    reply(command, ok=True, version=list(bpy.app.version), version_string=bpy.app.version_string,
          build_hash=bpy.app.build_hash.decode("utf-8", "replace"), with_osl=bool(_cycles.with_osl))


COMMANDS = {
    "info": cmd_info,
    "devices": cmd_devices,
    "load": cmd_load,
    "render": cmd_render,
    "view": cmd_view,
    "osl_compile": cmd_osl_compile,
    "merge": cmd_merge,
    "denoise": cmd_denoise,
    "bake": cmd_bake,
}


def main():
    _args.update(parse_args())
    if "addon_dir" not in _args or "session_dir" not in _args:
        print("JCycles worker: --addon-dir and --session-dir are required", file=sys.stderr)
        sys.exit(2)
    os.makedirs(_args["session_dir"], exist_ok=True)
    install_handlers()
    threading.Thread(target=stdin_reader, name="JCyclesStdin", daemon=True).start()
    send({"type": "ready", "version": list(bpy.app.version), "pid": os.getpid()})

    while True:
        command = _commands.get()
        name = command.get("cmd")
        if name == "quit":
            break
        handler = COMMANDS.get(name)
        if handler is None:
            send({"type": "reply", "id": command.get("id"), "ok": False, "error": "Unknown command " + repr(name)})
            continue
        try:
            handler(command)
        except Exception as ex:
            traceback.print_exc()
            send({"type": "reply", "id": command.get("id"), "ok": False,
                  "error": "{:s}: {:s}".format(type(ex).__name__, str(ex)),
                  "traceback": traceback.format_exc()})
    send({"type": "bye"})
    sys.stdout.flush()
    # Leave without running the regular Blender shutdown in a background loop.
    os._exit(0)


if __name__ == "__main__":
    main()
