# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Build reflected-path coherence fixtures and independent virtual-source references.

Run in Blender, without rendering:
  blender --background --factory-startup --python this_file.py -- OUTPUT_DIR

The scenes use real Cycles point-light coherence controls, a perfect planar
mirror, an opaque direct-ray mask, and a diffuse detector. They exercise a
reflected path and are expected to remain unsupported by the current direct
source coherence estimator. No image texture encodes the reference pattern.
"""

import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_diffuse_material(name, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (*color, 1.0)
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


def add_horizontal_rectangle(name, center, width, height, material):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (width / 2.0, height / 2.0, 1.0)
    obj.data.materials.append(material)
    return obj


def configure_scene(specular_connections=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 512
    scene.cycles.seed = 19
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.use_bidirectional_path_tracing = True
    scene.cycles.use_guiding = False
    scene.cycles.use_photon_mapping = False
    # The accepted path contains one glossy mirror bounce and one diffuse
    # detector bounce. Cap each class to avoid repeated mirror/detector paths.
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 0
    scene.cycles.transparent_max_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.sample_clamp_direct = 0
    scene.cycles.sample_clamp_indirect = 0
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.unit_settings.system = "NONE"
    scene.unit_settings.scale_length = 1.0
    scene.render.filepath = "//coherent_mirror"

    world = bpy.data.worlds.new("Black world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.world = world

    # Detector occupies x=[0.8,1.2], y=[-0.2,0.2] at z=2 and faces down.
    diffuse = set_diffuse_material("Unit diffuse detector", (1.0, 1.0, 1.0))
    bpy.ops.mesh.primitive_plane_add(
        size=2.0, location=(1.0, 0.0, 2.0), rotation=(math.pi, 0.0, 0.0)
    )
    detector = bpy.context.object
    detector.name = "Diffuse detector — reflected path ROI"
    detector.scale = (0.2, 0.2, 1.0)
    detector.data.materials.append(diffuse)

    # Under z=0 reflection, the detector ROI sees virtual sources at z=-1.
    # Its reflected rays hit the mirror around x=[0.267,0.400], y=[-0.067,0.067].
    mirror = add_horizontal_rectangle(
        "Perfect planar mirror z=0", (0.3333333333, 0.0, 0.0), 0.55, 0.55, None
    )
    mirror_material = bpy.data.materials.new("Unit perfect mirror")
    mirror_material.use_nodes = True
    mirror_nodes = mirror_material.node_tree.nodes
    mirror_nodes.clear()
    glossy = mirror_nodes.new("ShaderNodeBsdfGlossy")
    if specular_connections:
        glossy.distribution = "GGX"
    glossy.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy.inputs["Roughness"].default_value = 0.0
    mirror_output = mirror_nodes.new("ShaderNodeOutputMaterial")
    mirror_material.node_tree.links.new(glossy.outputs["BSDF"], mirror_output.inputs["Surface"])
    mirror.data.materials.clear()
    mirror.data.materials.append(mirror_material)

    # The masks lie just above the emitters. Direct rays to the entire detector
    # cross z=1.05 at x approximately [0.04,0.06]. Reflected rays cross that
    # height after the bounce around x=[0.58,0.82], far outside either mask.
    black = set_diffuse_material("Opaque direct-ray mask", (0.0, 0.0, 0.0))
    mask_centers = (-0.00005, 0.00005)
    masks = []
    for label, x in zip(("A", "B"), mask_centers):
        direct_bundle_x = 0.05 + x * 0.95
        masks.append(add_horizontal_rectangle(
            f"Opaque source-side direct mask {label}",
            (direct_bundle_x, 0.0, 1.05), 0.04, 0.12, black,
        ))

    sources = []
    for label, x in (("A", -50e-6), ("B", 50e-6)):
        bpy.ops.object.light_add(type="POINT", location=(x, 0.0, 1.0))
        source = bpy.context.object
        source.name = f"Coherent point source {label}"
        source.data.energy = 10.0
        source.data.shadow_soft_size = 0.0
        source.data.cycles.coherence_group = 1
        source.data.cycles.coherence_wavelength_nm = 550.0
        source.data.cycles.coherence_length_m = 1.0
        source.data.cycles.coherence_phase = 0.0
        sources.append(source)

    bpy.ops.object.camera_add(location=(1.0, 0.0, 1.2))
    camera = bpy.context.object
    camera.name = "Detector camera"
    camera.rotation_euler = (Vector((1.0, 0.0, 2.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 0.4
    camera.data.clip_start = 0.01
    camera.data.clip_end = 10.0
    scene.camera = camera

    scene["coherence_acceptance_scope"] = "reflected point-source paths through one planar mirror"
    scene["coherence_expected_state"] = "UNSUPPORTED: current estimator handles direct point-source NEE only"
    scene["coherence_direct_masks"] = "opaque masks block direct paths; reflected paths pass beside them"
    return scene, detector, mirror, masks, sources, camera


def camera_world_at_detector(camera, detector_z, resolution, pixel_x, pixel_y):
    """Intersect an orthographic camera pixel ray with the detector plane."""
    matrix = np.asarray([list(row) for row in camera.matrix_world], dtype=np.float64)
    camera_position = matrix[:3, 3]
    right = matrix[:3, 0]
    up = matrix[:3, 1]
    forward = -matrix[:3, 2]
    ortho_width = camera.data.ortho_scale
    aspect = resolution / resolution
    ortho_height = ortho_width / aspect
    local_x = (pixel_x / resolution - 0.5) * ortho_width
    local_y = (pixel_y / resolution - 0.5) * ortho_height
    origin_x = camera_position[0] + right[0] * local_x + up[0] * local_y
    origin_y = camera_position[1] + right[1] * local_x + up[1] * local_y
    origin_z = camera_position[2] + right[2] * local_x + up[2] * local_y
    distance = (detector_z - origin_z) / forward[2]
    return origin_x + forward[0] * distance, origin_y + forward[1] * distance


def make_reference(sources, camera, output_dir):
    resolution = 512
    detector_width = 0.4
    # Fixed in world coordinates before any render; the camera transform below
    # maps every pixel and sample to the actual detector plane.
    roi_bounds = {"x": (0.85, 1.15), "y": (-0.15, 0.15)}
    rows, columns = np.indices((resolution, resolution), dtype=np.float64)
    center_x, center_y = camera_world_at_detector(
        camera, 2.0, resolution, columns + 0.5, rows + 0.5
    )
    roi_mask = ((center_x >= roi_bounds["x"][0]) & (center_x <= roi_bounds["x"][1]) &
                (center_y >= roi_bounds["y"][0]) & (center_y <= roi_bounds["y"][1]))
    if not np.any(roi_mask):
        raise RuntimeError("camera-matrix mapping produced an empty detector ROI")
    pixel_offsets = np.array([0.125, 0.375, 0.625, 0.875], dtype=np.float64)
    source_positions = [np.asarray(tuple(source.location), dtype=np.float64) for source in sources]
    virtual_positions = [position * np.array([1.0, 1.0, -1.0]) for position in source_positions]
    wavelength = 550e-9
    coherence_length = 1.0

    profile_phase0 = np.zeros((resolution, resolution), dtype=np.float64)
    profile_phasepi = np.zeros_like(profile_phase0)
    profile_incoherent = np.zeros_like(profile_phase0)
    for ox in pixel_offsets:
        for oy in pixel_offsets:
            x, y = camera_world_at_detector(
                camera, 2.0, resolution, columns + ox, rows + oy
            )
            receiver = (x, y, np.full_like(x, 2.0))
            radii = []
            radiances = []
            for virtual in virtual_positions:
                dx, dy, dz = x - virtual[0], y - virtual[1], receiver[2] - virtual[2]
                radius = np.sqrt(dx * dx + dy * dy + dz * dz)
                radii.append(radius)
                cosine_detector = dz / radius
                # Blender point energy is radiant flux (W); a unit mirror
                # preserves radiance and a unit Lambertian detector divides
                # irradiance by pi: L_i = P*cos(theta)/(4*pi^2*r^2).
                radiances.append(10.0 * cosine_detector / (4.0 * math.pi**2 * radius**2))

            path_delta = radii[0] - radii[1]
            # Factored independently in double precision to avoid nearly equal
            # range subtraction in the phase reference.
            dx_source = virtual_positions[1][0] - virtual_positions[0][0]
            dy_source = virtual_positions[1][1] - virtual_positions[0][1]
            dz_source = virtual_positions[1][2] - virtual_positions[0][2]
            factored_delta = (
                dx_source * (2 * x - virtual_positions[0][0] - virtual_positions[1][0]) +
                dy_source * (2 * y - virtual_positions[0][1] - virtual_positions[1][1]) +
                dz_source * (2 * receiver[2] - virtual_positions[0][2] - virtual_positions[1][2])
            ) / (radii[0] + radii[1])
            if not np.allclose(path_delta, factored_delta, rtol=0.0, atol=2e-15):
                raise RuntimeError("virtual-source factored range-difference check failed")
            gamma = np.exp(-0.5 * (factored_delta / coherence_length) ** 2)
            incoherent = radiances[0] + radiances[1]
            cross = 2.0 * np.sqrt(radiances[0] * radiances[1]) * gamma
            phase = 2.0 * math.pi * factored_delta / wavelength
            profile_phase0 += (incoherent + cross * np.cos(phase)) / 16.0
            profile_phasepi += (incoherent + cross * np.cos(phase - math.pi)) / 16.0
            profile_incoherent += incoherent / 16.0

    np.savez_compressed(
        output_dir / "virtual_source_reference.npz",
        x_m=center_x,
        y_m=center_y,
        roi_mask=roi_mask,
        phase_0_radiance=profile_phase0,
        phase_pi_radiance=profile_phasepi,
        incoherent_radiance=profile_incoherent,
    )
    assumptions = {
        "status": "INDEPENDENT ANALYTIC REFERENCE; NOT A RENDER",
        "transport_model": "one planar perfect reflection, unfolded to virtual point sources",
        "virtual_sources_m": [position.tolist() for position in virtual_positions],
        "physical_sources_m": [position.tolist() for position in source_positions],
        "source_energy_w": [10.0, 10.0],
        "wavelength_m": wavelength,
        "coherence_length_m": coherence_length,
        "source_phase_cases_radians": {"phase_0": [0.0, 0.0], "phase_pi": [0.0, math.pi]},
        "incoherent_control": "exact zero-length Gaussian coherence, BDPT enabled",
        "radiance_model": "Li=Pi*cos(theta_detector)/(4*pi^2*ri^2); unit mirror reflectance; unit Lambertian detector",
        "pair_model": "L=La+Lb+2*sqrt(La*Lb)*exp(-0.5*(Delta_r/Lc)^2)*cos(2*pi*Delta_r/lambda+phi_a-phi_b)",
        "range_difference": "Delta_r = r_a-r_b, factored double-precision virtual-source identity",
        "pixel_filter": "Cycles BOX filter, width 1.0; independent 4x4 box subpixel quadrature, 16 equal weights per pixel",
        "detector_resolution": [resolution, resolution],
        "detector_width_m": detector_width,
        "image_row_origin": "bottom-left, matching Blender image pixel buffer order",
        "camera_orthographic_scale_m": float(camera.data.ortho_scale),
        "camera_matrix_world": [list(row) for row in camera.matrix_world],
        "roi_pixel_count": int(np.count_nonzero(roi_mask)),
        "roi_world_bounds_m": {"x": list(roi_bounds["x"]), "y": list(roi_bounds["y"])},
        "mask_clearance": {
            "direct_ray_x_at_mask_m": [0.04, 0.06],
            "reflected_ray_x_at_mask_over_roi_m": [0.58, 0.82],
            "mask_x_extents_m": [0.03, 0.07],
            "interpretation": "the predeclared ROI has direct paths blocked and reflected paths clear",
        },
        "assumptions_and_limits": [
            "scalar, monochromatic fields with no polarization transport",
            "coherent phase is accumulated only along the reflected geometric path",
            "no roughness, finite source radius, spectral bandwidth, or detector texture",
            "virtual-source radiance assumes an ideal unit-reflectance specular plane",
            "camera pixel mapping is the orthographic detector ROI; compare only saved ROI arrays",
        ],
        "expected_renderer_state": "UNSUPPORTED: current coherence estimator evaluates direct point-source NEE only; reflected-path phase response is expected to be absent",
        "acceptance_policy": "a finite render is not a pass; require phase-dependent comparison against both saved absolute references",
    }
    (output_dir / "virtual_source_reference.json").write_text(
        json.dumps(assumptions, indent=2) + "\n", encoding="utf-8"
    )


def main():
    if "--" not in sys.argv:
        raise SystemExit("Expected output directory after --")
    args = sys.argv[sys.argv.index("--") + 1:]
    output_dir = Path(args[0]).resolve()
    specular_connections = "--specular-connections" in args[1:]
    output_dir.mkdir(parents=True, exist_ok=False)
    scene, detector, mirror, masks, sources, camera = configure_scene(specular_connections)
    if specular_connections:
        detector.cycles.coherent_interface = 'DETECTOR'
        mirror.cycles.coherent_interface = 'MIRROR'
        scene.cycles.coherent_max_interface_events = 1
    bpy.context.view_layer.update()
    make_reference(sources, camera, output_dir)

    scene_files = {}
    variants = [
        ("phase_0", (0.0, 0.0), 1.0),
        ("phase_pi", (0.0, math.pi), 1.0),
        ("incoherent_bdpt_control", (0.0, 0.0), 0.0),
    ]
    if specular_connections:
        variants.append(("incoherent_connector_control", (0.0, 0.0), 1.0))
    for name, phases, coherence_length in variants:
        scene.cycles.use_coherent_specular_connections = (
            specular_connections and name != "incoherent_bdpt_control")
        for source_index, (source, phase) in enumerate(zip(sources, phases)):
            source.data.cycles.coherence_group = (
                source_index + 1 if name == "incoherent_connector_control" else 1)
            source.data.cycles.coherence_phase = phase
            source.data.cycles.coherence_wavelength_nm = 550.0
            source.data.cycles.coherence_length_m = coherence_length
        scene.render.filepath = f"//{name}"
        scene["coherence_acceptance_variant"] = name
        scene["coherence_expected_state"] = (
            "INCOHERENT CONTROL" if name == "incoherent_bdpt_control" else
            "SPECULAR CONNECTOR DIAGONAL CONTROL" if name == "incoherent_connector_control" else
            "SPECULAR CONNECTOR MIRROR PHASE TEST" if specular_connections else
            "UNSUPPORTED REFLECTED-PATH PHASE TRANSPORT"
        )
        filepath = output_dir / f"{name}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(filepath))
        scene_files[name] = {"path": str(filepath), "sha256": sha256(filepath)}

    manifest = {
        "scope": "two-source reflected-path coherence acceptance fixture",
        "specular_connections": specular_connections,
        "binary": str(Path(bpy.app.binary_path).resolve()),
        "binary_version": bpy.app.version_string,
        "binary_sha256": sha256(Path(bpy.app.binary_path).resolve()),
        "scene_script": str(Path(__file__).resolve()),
        "scene_script_sha256": sha256(Path(__file__).resolve()),
        "scene_variants": scene_files,
        "virtual_source_reference": {
            "json": str(output_dir / "virtual_source_reference.json"),
            "json_sha256": sha256(output_dir / "virtual_source_reference.json"),
            "npz": str(output_dir / "virtual_source_reference.npz"),
            "npz_sha256": sha256(output_dir / "virtual_source_reference.npz"),
        },
        "geometry": {
            "detector_center_m": list(detector.location),
            "detector_size_m": [0.4, 0.4],
            "mirror_center_m": list(mirror.location),
            "mirror_bounds_x_m": [0.0583333333, 0.6083333333],
            "mask_count": len(masks),
            "source_positions_m": [list(source.location) for source in sources],
        },
        "render_policy": "scene files are prepared for optional Metal BDPT runs; this script performs no render",
        "expected_state": (
            "Test mirror connector against unchanged absolute and phase gates"
            if specular_connections else
            "UNSUPPORTED until coherence phase is transported through the mirror bounce"
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scenes": len(scene_files), "reference": "written", "rendered": False}))


if __name__ == "__main__":
    main()
