# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Render test scenes for the MetalFX denoiser of Cycles and time them.

  blender -b --factory-startup --python tests/python/cycles_metalfx_scenes.py -- \\
      --output DIRECTORY [--scenes a,b] [--denoiser NONE|OPENIMAGEDENOISE|METALFX] \\
      [--samples N] [--resolution WxH] [--upscale NONE|QUALITY|BALANCED|PERFORMANCE|...] \\
      [--device METAL|CPU|METAL_CPU] [--tag NAME] [--animation N] [--adaptive THRESHOLD] \\
      [--tile-size N] [--time-limit SECONDS]

Every render writes <scene>_<tag>.exr (scene linear, full float) and an entry in
<tag>.json with the render time. Compare directories with cycles_metalfx_compare.py.

--animation N renders N frames of a moving camera and moving object instead of a still,
to exercise motion vectors in final renders.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    return scene


def material(name, base=(0.8, 0.8, 0.8), rough=0.5, metallic=0.0, transmission=0.0,
             emission=None, checker=None, ior=1.45):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        links.new(bsdf.outputs[0], out.inputs[0])
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["IOR"].default_value = ior
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission[0], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission[1]
    if checker is not None:
        tex = nodes.new("ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value = checker[0]
        tex.inputs["Color1"].default_value = (*checker[1], 1.0)
        tex.inputs["Color2"].default_value = (*checker[2], 1.0)
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def add(obj_op, mat, smooth=False, **kwargs):
    obj_op(**kwargs)
    obj = bpy.context.active_object
    obj.data.materials.append(mat)
    if smooth:
        bpy.ops.object.shade_smooth()
    return obj


def camera(scene, location, target, lens=35.0, ortho=None):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens
    if ortho:
        cam_data.type = 'ORTHO'
        cam_data.ortho_scale = ortho
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = location
    empty = bpy.data.objects.new("Target", None)
    empty.location = target
    scene.collection.objects.link(empty)
    con = cam.constraints.new('TRACK_TO')
    con.target = empty
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    scene.camera = cam
    return cam


def world(scene, color=(0.05, 0.07, 0.1), strength=1.0, sky=False):
    w = bpy.data.worlds.new("World")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*color, 1.0)
    bg.inputs[1].default_value = strength
    if sky:
        tex = w.node_tree.nodes.new("ShaderNodeTexSky")
        w.node_tree.links.new(tex.outputs[0], bg.inputs[0])
    scene.world = w


def area_light(scene, location, energy, size, color=(1, 1, 1), rotation=(0, 0, 0)):
    data = bpy.data.lights.new("Area", 'AREA')
    data.energy = energy
    data.size = size
    data.color = color
    obj = bpy.data.objects.new("Area", data)
    obj.location = location
    obj.rotation_euler = rotation
    scene.collection.objects.link(obj)
    return obj


def scene_materials():
    """Diffuse, glossy, metal, glass, textured and emissive surfaces under an area light."""
    scene = reset()
    world(scene, (0.08, 0.1, 0.14), 0.6)
    add(bpy.ops.mesh.primitive_plane_add,
        material("floor", rough=0.6, checker=(14.0, (0.75, 0.72, 0.65), (0.12, 0.13, 0.16))),
        size=14)
    add(bpy.ops.mesh.primitive_plane_add, material("wall", (0.55, 0.2, 0.15), 0.8),
        size=14, location=(0, 4, 3), rotation=(math.radians(90), 0, 0))
    add(bpy.ops.mesh.primitive_uv_sphere_add, material("diffuse", (0.8, 0.25, 0.1), 0.9),
        smooth=True, radius=0.8, location=(-2.6, 0.2, 0.8), segments=64, ring_count=32)
    add(bpy.ops.mesh.primitive_uv_sphere_add, material("gold", (1.0, 0.76, 0.33), 0.25, 1.0),
        smooth=True, radius=0.8, location=(-0.85, 0.6, 0.8), segments=64, ring_count=32)
    add(bpy.ops.mesh.primitive_uv_sphere_add, material("mirror", (0.95, 0.95, 0.95), 0.02, 1.0),
        smooth=True, radius=0.8, location=(0.9, 0.2, 0.8), segments=64, ring_count=32)
    add(bpy.ops.mesh.primitive_uv_sphere_add,
        material("glass", (1.0, 1.0, 1.0), 0.0, 0.0, 1.0),
        smooth=True, radius=0.8, location=(2.65, 0.6, 0.8), segments=64, ring_count=32)
    add(bpy.ops.mesh.primitive_monkey_add, material("plastic", (0.1, 0.35, 0.8), 0.35),
        smooth=True, size=1.1, location=(0.0, -1.7, 0.62), rotation=(math.radians(-25), 0, 0.5))
    add(bpy.ops.mesh.primitive_cube_add,
        material("emitter", (0, 0, 0), 0.5, emission=((1.0, 0.5, 0.15), 12.0)),
        size=0.45, location=(-1.9, -1.6, 0.225))
    add(bpy.ops.mesh.primitive_torus_add, material("rough_metal", (0.7, 0.7, 0.75), 0.5, 1.0),
        smooth=True, location=(2.0, -1.5, 0.3), major_radius=0.6, minor_radius=0.25,
        major_segments=64, minor_segments=32)
    area_light(scene, (1.5, -2.5, 5.0), 900.0, 2.5, rotation=(math.radians(25), 0.2, 0))
    area_light(scene, (-4.0, -1.0, 2.5), 250.0, 1.0, (0.6, 0.75, 1.0),
               rotation=(0, math.radians(-65), 0))
    camera(scene, (0.0, -8.2, 3.2), (0.0, 0.0, 0.6), 38.0)
    return scene


def scene_interior():
    """Indirect-light dominated box with small openings: hard for few samples."""
    scene = reset()
    world(scene, (1.0, 0.95, 0.9), 3.0)
    white = material("white", (0.8, 0.8, 0.8), 0.9)
    red = material("red", (0.7, 0.08, 0.06), 0.9)
    green = material("green", (0.1, 0.55, 0.12), 0.9)
    add(bpy.ops.mesh.primitive_plane_add, material(
        "floor", rough=0.35, checker=(8.0, (0.7, 0.7, 0.7), (0.25, 0.22, 0.2))), size=6)
    add(bpy.ops.mesh.primitive_plane_add, white, size=6, location=(0, 0, 4))
    add(bpy.ops.mesh.primitive_plane_add, white, size=6, location=(0, 3, 2),
        rotation=(math.radians(90), 0, 0))
    add(bpy.ops.mesh.primitive_plane_add, red, size=6, location=(-3, 0, 2),
        rotation=(0, math.radians(90), 0))
    # Wall with a window slit on the right: build from two planes.
    add(bpy.ops.mesh.primitive_plane_add, green, size=1, location=(3, 0, 0.6),
        rotation=(0, math.radians(90), 0), scale=(1.2, 6, 1))
    add(bpy.ops.mesh.primitive_plane_add, green, size=1, location=(3, 0, 3.1),
        rotation=(0, math.radians(90), 0), scale=(1.8, 6, 1))
    add(bpy.ops.mesh.primitive_cube_add, white, size=1.4, location=(-1.1, 0.8, 0.7),
        rotation=(0, 0, 0.4))
    add(bpy.ops.mesh.primitive_uv_sphere_add, material("glossy", (0.9, 0.9, 0.9), 0.15, 1.0),
        smooth=True, radius=0.7, location=(1.0, -0.2, 0.7), segments=64, ring_count=32)
    add(bpy.ops.mesh.primitive_cylinder_add, material("blue", (0.1, 0.2, 0.7), 0.4),
        smooth=True, radius=0.4, depth=1.8, location=(0.2, 1.8, 0.9))
    area_light(scene, (0, 0, 3.95), 60.0, 1.0)
    camera(scene, (0.0, -7.5, 2.0), (0.0, 0.0, 1.7), 30.0)
    return scene


def scene_detail():
    """Thin geometry and high-frequency texture, to judge sharpness and upscaling."""
    scene = reset()
    world(scene, (0.4, 0.5, 0.7), 1.0, sky=False)
    add(bpy.ops.mesh.primitive_plane_add,
        material("floor", rough=0.7, checker=(60.0, (0.9, 0.9, 0.9), (0.05, 0.05, 0.05))),
        size=20)
    thin = material("thin", (0.8, 0.6, 0.1), 0.4)
    for i in range(24):
        angle = i / 24.0 * math.tau
        add(bpy.ops.mesh.primitive_cylinder_add, thin, smooth=True, radius=0.015, depth=2.4,
            location=(math.cos(angle) * 1.5, math.sin(angle) * 1.5, 1.2),
            rotation=(0.25 * math.sin(angle * 3), 0.25 * math.cos(angle * 2), 0))
    add(bpy.ops.mesh.primitive_ico_sphere_add, material("facets", (0.2, 0.6, 0.5), 0.2, 0.6),
        radius=0.9, subdivisions=2, location=(0, 0, 0.9))
    for i in range(7):
        add(bpy.ops.mesh.primitive_cube_add,
            material("c%d" % i, (0.15 + 0.12 * i, 0.3, 0.9 - 0.12 * i), 0.5),
            size=0.35, location=(-3.0 + i, -2.4, 0.175), rotation=(0, 0, 0.3 * i))
    area_light(scene, (2.5, -3.0, 5.0), 1200.0, 0.6, rotation=(math.radians(30), 0.35, 0))
    camera(scene, (4.2, -6.4, 3.0), (0.0, 0.0, 0.8), 40.0)
    return scene


def scene_catcher():
    """Shadow catcher floor under a transparent film: denoised alpha and catcher passes."""
    scene = scene_materials()
    floor = bpy.data.objects["Plane"]
    floor.is_shadow_catcher = True
    wall = bpy.data.objects["Plane.001"]
    bpy.data.objects.remove(wall)
    scene.render.film_transparent = True
    return scene


def scene_ortho():
    """Orthographic camera."""
    scene = scene_detail()
    scene.camera.data.type = 'ORTHO'
    scene.camera.data.ortho_scale = 7.0
    return scene


def scene_panorama():
    """Panoramic camera: no projection matrix, depth is a distance."""
    scene = scene_interior()
    scene.camera.data.type = 'PANO'
    scene.camera.data.panorama_type = 'FISHEYE_EQUISOLID'
    scene.camera.data.fisheye_lens = 12.0
    scene.camera.data.fisheye_fov = math.radians(160.0)
    scene.camera.location = (0.0, -2.5, 1.8)
    return scene


def scene_volume():
    """Scattering volume in front of surfaces, and a volume in the world."""
    scene = scene_materials()
    bpy.ops.mesh.primitive_cube_add(size=1.6, location=(-1.8, -1.2, 0.85))
    cube = bpy.context.active_object
    mat = bpy.data.materials.new("fog")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    vol = nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Density"].default_value = 1.5
    vol.inputs["Color"].default_value = (0.8, 0.9, 1.0, 1.0)
    links.new(vol.outputs[0], out.inputs["Volume"])
    cube.data.materials.append(mat)
    return scene


SCENES = {
    "materials": scene_materials,
    "interior": scene_interior,
    "detail": scene_detail,
    "catcher": scene_catcher,
    "ortho": scene_ortho,
    "panorama": scene_panorama,
    "volume": scene_volume,
}

DEFAULT_SCENES = "materials,interior,detail"

UPSCALE_FACTOR = {
    'NONE': 1.0,
    'QUALITY': 1.0 / 0.66666667,
    'BALANCED': 1.0 / 0.58,
    'PERFORMANCE': 2.0,
    'ULTRA_PERFORMANCE': 3.0,
}


def setup_device(device):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device == 'CPU':
        bpy.context.scene.cycles.device = 'CPU'
        return
    prefs.compute_device_type = 'METAL'
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = dev.type == 'METAL' or (device == 'METAL_CPU' and dev.type == 'CPU')
    bpy.context.scene.cycles.device = 'GPU'


def animate(scene, frames):
    """Orbit the camera and move one object so that every frame differs."""
    cam = scene.camera
    for frame in range(1, frames + 1):
        t = (frame - 1) / max(frames - 1, 1)
        scene.frame_set(frame)
        angle = -0.35 + 0.7 * t
        radius = math.hypot(cam.location.x, cam.location.y)
        base = math.atan2(cam.location.y, cam.location.x) if frame == 1 else animate.base
        if frame == 1:
            animate.base = base
            animate.radius = radius
            animate.z = cam.location.z
        cam.location = (math.cos(animate.base + angle) * animate.radius,
                        math.sin(animate.base + angle) * animate.radius, animate.z)
        cam.keyframe_insert("location", frame=frame)
    scene.frame_start = 1
    scene.frame_end = frames


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", default=DEFAULT_SCENES)
    parser.add_argument("--denoiser", default="NONE")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--resolution", default="960x540")
    parser.add_argument("--upscale", default="NONE")
    parser.add_argument("--device", default="METAL")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--animation", type=int, default=0)
    parser.add_argument("--pixel-jitter", action="store_true",
                        help="Replace the pixel filter by the per-frame jitter of the integrator")
    parser.add_argument("--adaptive", type=float, default=0.0,
                        help="Use adaptive sampling with this noise threshold")
    parser.add_argument("--tile-size", type=int, default=0,
                        help="Render in tiles of this size, which denoises the whole image once")
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--passes", action="store_true",
                        help="Also store denoising data passes in a multilayer EXR")
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("%s_%dspp" % (args.denoiser.lower(), args.samples))
    width, height = (int(v) for v in args.resolution.split("x"))
    results = {}

    for name in args.scenes.split(","):
        scene = SCENES[name]()
        setup_device(args.device)
        cscene = scene.cycles
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.image_settings.exr_codec = 'ZIP'
        cscene.samples = args.samples
        cscene.use_adaptive_sampling = args.adaptive > 0.0
        cscene.adaptive_threshold = args.adaptive
        cscene.use_auto_tile = args.tile_size > 0
        if args.tile_size > 0:
            cscene.tile_size = args.tile_size
        cscene.time_limit = args.time_limit
        cscene.seed = 7
        cscene.use_pixel_jitter = args.pixel_jitter
        cscene.use_denoising = args.denoiser != 'NONE'
        if cscene.use_denoising:
            try:
                cscene.denoiser = args.denoiser
            except TypeError:
                print("SKIP: denoiser %s is not available in this build" % args.denoiser)
                results[name] = {"skipped": True}
                continue
            if args.denoiser == 'OPENIMAGEDENOISE':
                cscene.denoising_use_gpu = True
            if hasattr(cscene, "denoising_upscale_quality"):
                cscene.denoising_upscale_quality = args.upscale
        if args.passes:
            scene.view_layers[0].cycles.denoising_store_passes = True
            scene.view_layers[0].update_render_passes()
            scene.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
            scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'

        entry = {"frames": []}
        if args.animation:
            animate(scene, args.animation)
            frames = range(1, args.animation + 1)
        else:
            frames = [1]
        for frame in frames:
            scene.frame_set(frame)
            suffix = "_%03d" % frame if args.animation else ""
            scene.render.filepath = str(out_dir / ("%s_%s%s.exr" % (name, tag, suffix)))
            start = time.time()
            bpy.ops.render.render(write_still=True)
            entry["frames"].append(time.time() - start)
        entry["time"] = sum(entry["frames"])
        results[name] = entry
        print("METALFX_TEST %s %s: %.2fs" % (name, tag, entry["time"]))

    (out_dir / (tag + ".json")).write_text(json.dumps(
        {"args": vars(args), "results": results}, indent=2))


if __name__ == "__main__":
    main()
