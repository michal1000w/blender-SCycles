import math
import os
import sys
import time

import bpy


def configure_cycles(samples, resolution):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.preview_samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.use_pixel_displacement = os.environ.get(
        "PIXEL_DISPLACEMENT_BENCH_ENABLED", "1"
    ) not in {"0", "false", "False"}
    scene.cycles.pixel_displacement_scale = 1.0
    scene.cycles.pixel_displacement_max_distance = 0.12
    scene.cycles.pixel_displacement_steps = 32
    scene.cycles.pixel_displacement_resolution = int(
        os.environ.get("PIXEL_DISPLACEMENT_BENCH_MICROMESH_RESOLUTION", "1024")
    )
    scene.cycles.device = os.environ.get("PIXEL_DISPLACEMENT_BENCH_DEVICE", "GPU")
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.metalrt = os.environ.get("PIXEL_DISPLACEMENT_BENCH_METALRT", "ON")
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == "METAL"
    except Exception as exc:
        print(f"BENCH_DEVICE_WARNING {exc}")


def make_height_image(size):
    image = bpy.data.images.new("pixel_displacement_height", size, size, alpha=False, float_buffer=False)
    pixels = [0.0] * (size * size * 4)
    for y in range(size):
        fy = y / max(size - 1, 1)
        for x in range(size):
            fx = x / max(size - 1, 1)
            # Several frequencies make texture filtering and cache behavior visible without
            # relying on external assets.
            h = (
                0.50
                + 0.22 * math.sin(fx * math.tau * 47.0)
                + 0.14 * math.sin((fx + fy) * math.tau * 93.0)
                + 0.10 * math.cos(fy * math.tau * 71.0)
                + 0.04 * math.sin((fx * 19.0 - fy * 23.0) * math.tau)
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
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.55, 0.58, 0.62, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.62
    mat.node_tree.links.new(image_node.outputs["Color"], displacement.inputs["Height"])
    mat.node_tree.links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])
    obj.data.materials.append(mat)


def create_scene(grid_size, texture_size, samples, resolution):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    configure_cycles(samples, resolution)
    image = make_height_image(texture_size)
    obj = create_grid_mesh(grid_size)
    assign_material(obj, image)

    light_data = bpy.data.lights.new("key_area", "AREA")
    light_data.energy = 420.0
    light_data.size = 4.0
    light = bpy.data.objects.new("key_area", light_data)
    light.location = (0.0, -3.0, 4.0)
    bpy.context.collection.objects.link(light)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    camera.location = (0.0, -4.0, 3.2)
    camera.rotation_euler = (math.radians(58.0), 0.0, 0.0)
    camera_data.lens = 35.0
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera


def render_once(label):
    start = time.perf_counter()
    bpy.ops.render.render(write_still=False)
    elapsed = time.perf_counter() - start
    print(f"BENCH_RENDER {label} {elapsed:.6f}")
    return elapsed


def main():
    grid_size = int(os.environ.get("PIXEL_DISPLACEMENT_BENCH_GRID", "72"))
    texture_size = int(os.environ.get("PIXEL_DISPLACEMENT_BENCH_TEXTURE", "2048"))
    samples = int(os.environ.get("PIXEL_DISPLACEMENT_BENCH_SAMPLES", "8"))
    resolution = int(os.environ.get("PIXEL_DISPLACEMENT_BENCH_RESOLUTION", "512"))
    create_scene(grid_size, texture_size, samples, resolution)
    blend_path = os.environ.get("PIXEL_DISPLACEMENT_BENCH_BLEND", "/tmp/pixel_displacement_bench.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(
        "BENCH_SCENE "
        f"grid={grid_size} triangles={grid_size * grid_size * 2} "
        f"texture={texture_size} samples={samples} resolution={resolution}"
    )
    first = render_once("first")
    second = render_once("second")
    print(f"BENCH_TOTAL first={first:.6f} second={second:.6f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
