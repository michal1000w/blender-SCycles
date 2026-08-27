import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def configure_cycles(samples, resolution, preserve_displacement_settings):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.preview_samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.use_pixel_displacement = True
    if not preserve_displacement_settings or "PIXEL_DISPLACEMENT_VISUAL_SCALE" in os.environ:
        scene.cycles.pixel_displacement_scale = float(
            os.environ.get("PIXEL_DISPLACEMENT_VISUAL_SCALE", "1.0")
        )
    if not preserve_displacement_settings or "PIXEL_DISPLACEMENT_VISUAL_MAX_DISTANCE" in os.environ:
        scene.cycles.pixel_displacement_max_distance = float(
            os.environ.get("PIXEL_DISPLACEMENT_VISUAL_MAX_DISTANCE", "0.12")
        )
    if not preserve_displacement_settings or "PIXEL_DISPLACEMENT_VISUAL_STEPS" in os.environ:
        scene.cycles.pixel_displacement_steps = int(
            os.environ.get("PIXEL_DISPLACEMENT_VISUAL_STEPS", "32")
        )
    if (
        not preserve_displacement_settings
        or "PIXEL_DISPLACEMENT_VISUAL_MICROMESH_RESOLUTION" in os.environ
    ):
        scene.cycles.pixel_displacement_resolution = int(
            os.environ.get("PIXEL_DISPLACEMENT_VISUAL_MICROMESH_RESOLUTION", "1024")
        )
    scene.cycles.device = "GPU"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.metalrt = os.environ.get("PIXEL_DISPLACEMENT_VISUAL_METALRT", "ON")
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == "METAL"
    except Exception as exc:
        print(f"VISUAL_DEVICE_WARNING {exc}")


def make_height_image(size):
    image = bpy.data.images.new("pixel_displacement_height", size, size, alpha=False, float_buffer=False)
    pixels = [0.0] * (size * size * 4)
    height_mode = os.environ.get("PIXEL_DISPLACEMENT_VISUAL_HEIGHT_MODE", "waves")
    for y in range(size):
        fy = y / max(size - 1, 1)
        for x in range(size):
            fx = x / max(size - 1, 1)
            if height_mode == "gradient":
                h = fx
            else:
                h = (
                    0.50
                    + 0.24 * math.sin(fx * math.tau * 31.0)
                    + 0.16 * math.sin((fx + fy) * math.tau * 59.0)
                    + 0.10 * math.cos(fy * math.tau * 43.0)
                    + 0.05 * math.sin((fx * 13.0 - fy * 17.0) * math.tau)
                )
            h = max(0.0, min(1.0, h))
            offset = (y * size + x) * 4
            pixels[offset + 0] = h
            pixels[offset + 1] = h
            pixels[offset + 2] = h
            pixels[offset + 3] = 1.0
    image.pixels.foreach_set(pixels)
    image.pack()
    return image


def create_grid_mesh(grid_size):
    verts = []
    uvs = []
    for y in range(grid_size + 1):
        v = y / grid_size
        for x in range(grid_size + 1):
            u = x / grid_size
            verts.append(((u - 0.5) * 4.0, (v - 0.5) * 4.0, 0.0))
            uvs.append((u, v))

    faces = []
    for y in range(grid_size):
        for x in range(grid_size):
            v0 = y * (grid_size + 1) + x
            v1 = v0 + 1
            v2 = v0 + (grid_size + 1)
            v3 = v2 + 1
            faces.append((v0, v1, v3))
            faces.append((v0, v3, v2))

    mesh = bpy.data.meshes.new("pixel_displacement_grid_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            uv_layer.data[loop_index].uv = uvs[mesh.loops[loop_index].vertex_index]

    obj = bpy.data.objects.new("pixel_displacement_grid", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def assign_material(obj, image):
    mat = bpy.data.materials.new("pixel_displacement_material")
    mat.use_nodes = True
    if hasattr(mat, "displacement_method"):
        mat.displacement_method = "DISPLACEMENT"
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    output = nodes.get("Material Output")
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = image
    image_node.extension = "REPEAT"
    image_node.interpolation = "Linear"
    displacement = nodes.new("ShaderNodeDisplacement")
    displacement.inputs["Midlevel"].default_value = 0.5
    displacement.inputs["Scale"].default_value = 0.12
    if os.environ.get("PIXEL_DISPLACEMENT_VISUAL_EMISSION", "0") in {"1", "true", "True"}:
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        emission.inputs["Strength"].default_value = 1.0
        mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.55, 0.58, 0.62, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.62
    mat.node_tree.links.new(image_node.outputs["Color"], displacement.inputs["Height"])
    mat.node_tree.links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])
    obj.data.materials.append(mat)


def create_synthetic_scene(grid_size, texture_size):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    image = make_height_image(texture_size)
    obj = create_grid_mesh(grid_size)
    assign_material(obj, image)

    light_data = bpy.data.lights.new("key_area", "AREA")
    light_data.energy = 420.0
    light_data.size = 4.0
    light = bpy.data.objects.new("key_area", light_data)
    light.location = (0.0, -3.0, 4.0)
    bpy.context.collection.objects.link(light)


def cache_bust_displacement_materials():
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for node in list(nodes):
            if node.bl_idname != "ShaderNodeDisplacement":
                continue

            height_input = node.inputs.get("Height")
            if height_input is None or not height_input.is_linked:
                continue

            old_link = height_input.links[0]
            old_socket = old_link.from_socket
            links.remove(old_link)

            light_path = nodes.new("ShaderNodeLightPath")
            zero_mul = nodes.new("ShaderNodeMath")
            zero_mul.operation = "MULTIPLY"
            zero_mul.inputs[1].default_value = 0.0
            add = nodes.new("ShaderNodeMath")
            add.operation = "ADD"

            links.new(light_path.outputs["Is Camera Ray"], zero_mul.inputs[0])
            links.new(old_socket, add.inputs[0])
            links.new(zero_mul.outputs[0], add.inputs[1])
            links.new(add.outputs[0], height_input)


def scene_bounds():
    min_v = Vector((math.inf, math.inf, math.inf))
    max_v = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
            continue
        if obj.hide_render:
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_v.x = min(min_v.x, world.x)
            min_v.y = min(min_v.y, world.y)
            min_v.z = min(min_v.z, world.z)
            max_v.x = max(max_v.x, world.x)
            max_v.y = max(max_v.y, world.y)
            max_v.z = max(max_v.z, world.z)
            found = True
    if not found:
        return Vector((0.0, 0.0, 0.0)), 2.0
    center = (min_v + max_v) * 0.5
    radius = max((max_v - min_v).length * 0.5, 1.0)
    return center, radius


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(view_name):
    center, radius = scene_bounds()
    camera_data = bpy.data.cameras.new(f"Camera_{view_name}")
    camera = bpy.data.objects.new(f"Camera_{view_name}", camera_data)
    bpy.context.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = radius * (2.4 if view_name != "side" else 1.25)

    if view_name == "top":
        camera.location = center + Vector((0.0, 0.0, radius * 3.0))
    elif view_name == "side":
        camera.location = center + Vector((0.0, -radius * 3.0, radius * 0.08))
    else:
        camera.location = center + Vector((radius * 1.8, -radius * 2.2, radius * 1.35))
    look_at(camera, center)
    bpy.context.scene.camera = camera


def load_scene(exact_reference):
    if exact_reference:
        os.environ["CYCLES_PIXEL_DISPLACEMENT_DISABLE_CACHE"] = "1"
    else:
        os.environ.pop("CYCLES_PIXEL_DISPLACEMENT_DISABLE_CACHE", None)

    blend = os.environ.get("PIXEL_DISPLACEMENT_VISUAL_BLEND", "")
    if blend:
        bpy.ops.wm.open_mainfile(filepath=blend)
    else:
        grid = int(os.environ.get("PIXEL_DISPLACEMENT_VISUAL_GRID", "4"))
        texture = int(os.environ.get("PIXEL_DISPLACEMENT_VISUAL_TEXTURE", "512"))
        create_synthetic_scene(grid, texture)

    configure_cycles(
        int(os.environ.get("PIXEL_DISPLACEMENT_VISUAL_SAMPLES", "1")),
        int(os.environ.get("PIXEL_DISPLACEMENT_VISUAL_RESOLUTION", "256")),
        preserve_displacement_settings=bool(blend),
    )
    if exact_reference:
        cache_bust_displacement_materials()


def render_pixels(output_dir, view_name, exact_reference):
    setup_camera(view_name)
    label = "reference" if exact_reference else "cached"
    path = output_dir / f"{view_name}_{label}.png"
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images["Render Result"]
    pixels = list(image.pixels[:])
    return pixels, path


def compare_pixels(a, b):
    if len(a) != len(b):
        return math.inf, math.inf
    total = 0.0
    max_abs = 0.0
    for av, bv in zip(a, b):
        diff = abs(av - bv)
        total += diff * diff
        max_abs = max(max_abs, diff)
    return math.sqrt(total / max(len(a), 1)), max_abs


def main():
    output_dir = Path(os.environ.get("PIXEL_DISPLACEMENT_VISUAL_OUTPUT", "/tmp/pixel_displacement_visual"))
    output_dir.mkdir(parents=True, exist_ok=True)
    views = os.environ.get("PIXEL_DISPLACEMENT_VISUAL_VIEWS", "top,side,angle").split(",")

    failures = 0
    threshold = float(os.environ.get("PIXEL_DISPLACEMENT_VISUAL_RMS_THRESHOLD", "0.015"))
    cache_only = os.environ.get("PIXEL_DISPLACEMENT_VISUAL_CACHE_ONLY", "0") in {
        "1",
        "true",
        "True",
    }
    for view_name in views:
        view_name = view_name.strip()
        if not view_name:
            continue

        load_scene(exact_reference=False)
        cached, cached_path = render_pixels(output_dir, view_name, exact_reference=False)
        if cache_only:
            print(f"VISUAL_CACHED view={view_name} path={cached_path}")
            continue
        load_scene(exact_reference=True)
        reference, reference_path = render_pixels(output_dir, view_name, exact_reference=True)
        rms, max_abs = compare_pixels(cached, reference)
        ok = rms <= threshold
        failures += 0 if ok else 1
        print(
            "VISUAL_COMPARE "
            f"view={view_name} rms={rms:.8f} max={max_abs:.8f} "
            f"cached={cached_path} reference={reference_path} status={'ok' if ok else 'fail'}"
        )

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
