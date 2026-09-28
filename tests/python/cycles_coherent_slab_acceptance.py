#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Save an editable two-source glass-slab fixture and independent vector references.

Run without rendering:
  blender --background --factory-startup --python cycles_coherent_slab_acceptance.py -- OUTPUT_DIR

Revise reference arrays from a saved fixture without starting Blender:
  python3 cycles_coherent_slab_acceptance.py -- NEW_DIR --reference-only-from OLD_DIR

The reference includes the direct two-transmission branch through a parallel,
lossless slab. Internal-reflection branches are intentionally outside this
bounded gate; this is not a full Fabry-Perot slab solution.
"""

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None
    Vector = None


RESOLUTION = 512
AIR_BEFORE_M = 0.3
SLAB_THICKNESS_M = 0.3
AIR_AFTER_M = 0.4
SLAB_IOR = 1.5
WAVELENGTH_M = 550e-9
COHERENCE_LENGTH_M = 1.0
SOURCE_ENERGY_W = 10.0
SOURCE_OFFSETS_Y_M = (-50e-6, 50e-6)
ROI_BOUNDS_M = {"y": (-0.15, 0.15), "z": (-0.15, 0.15)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def radial_connection(radius, air_before, slab_thickness, air_after, ior):
    """Independent monotone Snell root for a plane-parallel air/glass/air ray.

    The unknown is sin(theta_air). The ray's transverse displacement is the
    sum of air and glass tangents. No renderer manifold/Jacobian code is used.
    """
    radius = np.asarray(radius, dtype=np.float64)
    if np.any(radius < 0) or min(air_before, slab_thickness, air_after) <= 0 or ior < 1.0:
        raise ValueError("This air-to-slab oracle requires positive distances, radius >= 0, and slab IOR >= 1")
    air_total = air_before + air_after
    lo = np.zeros_like(radius)
    hi = np.full_like(radius, 1.0 - 1e-14)
    for _ in range(52):
        u = (lo + hi) * 0.5
        cos_air = np.sqrt(1.0 - u * u)
        sin_glass = u / ior
        cos_glass = np.sqrt(1.0 - sin_glass * sin_glass)
        displacement = air_total * u / cos_air + slab_thickness * sin_glass / cos_glass
        low = displacement < radius
        lo = np.where(low, u, lo)
        hi = np.where(low, hi, u)
    u = (lo + hi) * 0.5
    cos_air = np.sqrt(1.0 - u * u)
    sin_glass = u / ior
    cos_glass = np.sqrt(1.0 - sin_glass * sin_glass)
    displacement = air_total * u / cos_air + slab_thickness * sin_glass / cos_glass
    d_radius_du = air_total / cos_air**3 + slab_thickness / (ior * cos_glass**3)
    effective_normal_distance = air_total + slab_thickness / ior
    spreading = np.where(
        radius > 1e-14,
        u / (np.maximum(radius, 1e-300) * cos_air * d_radius_du),
        1.0 / effective_normal_distance**2,
    )
    optical_length = air_total / cos_air + ior * slab_thickness / cos_glass
    return {
        "sin_air": u,
        "cos_air": cos_air,
        "sin_glass": sin_glass,
        "cos_glass": cos_glass,
        "displacement": displacement,
        "spreading": spreading,
        "optical_length": optical_length,
    }


def flux_transmission(incident_ior, transmitted_ior, incident_cos, transmitted_cos, mode):
    """Lossless scalar s/p power transmission in the tangential-E convention."""
    if mode == "s":
        yi = incident_ior * incident_cos
        yt = transmitted_ior * transmitted_cos
    elif mode == "p":
        yi = incident_ior / incident_cos
        yt = transmitted_ior / transmitted_cos
    else:
        raise ValueError(mode)
    return 4.0 * yi * yt / (yi + yt) ** 2


def path_world_modes(y_displacement, z_displacement, ior=SLAB_IOR):
    """Double-precision transverse transport for three fixed world-axis dipoles.

    Each axis has amplitude sqrt(1/2) after projection onto the source ray.
    The two parallel slab faces share a plane of incidence, so their real
    flux-normalized transmission amplitudes multiply separately in s and p.
    The resulting 3D field is compared directly at the neutral detector.
    """
    radius = np.hypot(y_displacement, z_displacement)
    path = radial_connection(radius, AIR_BEFORE_M, SLAB_THICKNESS_M, AIR_AFTER_M, ior)
    s_y = np.divide(-z_displacement, radius, out=np.ones_like(radius), where=radius > 0)
    s_z = np.divide(y_displacement, radius, out=np.zeros_like(radius), where=radius > 0)
    s = np.stack((np.zeros_like(radius), s_y, s_z), axis=-1)
    direction = np.stack((path["cos_air"],
                          path["sin_air"] * s_z,
                          -path["sin_air"] * s_y), axis=-1)
    p = np.cross(s, direction)
    amplitudes = {}
    for mode in ("s", "p"):
        entry = flux_transmission(1.0, ior, path["cos_air"], path["cos_glass"], mode)
        exit_ = flux_transmission(ior, 1.0, path["cos_glass"], path["cos_air"], mode)
        amplitudes[mode] = np.sqrt(entry * exit_)
    base = SOURCE_ENERGY_W / (4.0 * math.pi**2) * path["spreading"]
    diagonal = 0.5 * base * (amplitudes["s"]**2 + amplitudes["p"]**2)
    return path, {"s": s, "p": p, "amplitude_s": amplitudes["s"],
                  "amplitude_p": amplitudes["p"], "base": base,
                  "diagonal": diagonal}


def world_mode_overlap(left, right):
    """Full-vector overlap, summed over matching independent world dipoles.

    This is half the Frobenius inner product of the two transported transverse
    dyads, expressed through s/p basis dot products. It includes s-to-p terms
    whenever the source rays have different incidence planes.
    """
    ss = np.sum(left["s"] * right["s"], axis=-1)
    sp = np.sum(left["s"] * right["p"], axis=-1)
    ps = np.sum(left["p"] * right["s"], axis=-1)
    pp = np.sum(left["p"] * right["p"], axis=-1)
    return 0.5 * (left["amplitude_s"] * right["amplitude_s"] * ss**2 +
                  left["amplitude_s"] * right["amplitude_p"] * sp**2 +
                  left["amplitude_p"] * right["amplitude_s"] * ps**2 +
                  left["amplitude_p"] * right["amplitude_p"] * pp**2)


def validate_reference_model():
    normal = radial_connection(np.asarray(0.0), AIR_BEFORE_M, SLAB_THICKNESS_M,
                               AIR_AFTER_M, SLAB_IOR)
    expected_opl = AIR_BEFORE_M + AIR_AFTER_M + SLAB_IOR * SLAB_THICKNESS_M
    expected_spreading = 1.0 / (AIR_BEFORE_M + AIR_AFTER_M + SLAB_THICKNESS_M / SLAB_IOR) ** 2
    assert abs(float(normal["optical_length"]) - expected_opl) < 2e-15
    assert abs(float(normal["spreading"]) - expected_spreading) < 2e-15
    expected_interface_power = 4.0 * SLAB_IOR / (1.0 + SLAB_IOR) ** 2
    for mode in ("s", "p"):
        entry = flux_transmission(1.0, SLAB_IOR, 1.0, 1.0, mode)
        exit_ = flux_transmission(SLAB_IOR, 1.0, 1.0, 1.0, mode)
        assert abs(entry - expected_interface_power) < 2e-15
        assert abs(exit_ - expected_interface_power) < 2e-15

    radii = np.asarray([0.0, 0.02, 0.11, 0.3], dtype=np.float64)
    identity = radial_connection(radii, AIR_BEFORE_M, SLAB_THICKNESS_M,
                                 AIR_AFTER_M, 1.0)
    distance = np.sqrt((AIR_BEFORE_M + SLAB_THICKNESS_M + AIR_AFTER_M) ** 2 + radii**2)
    expected_jacobian = (AIR_BEFORE_M + SLAB_THICKNESS_M + AIR_AFTER_M) / distance**3
    assert np.max(np.abs(identity["optical_length"] - distance)) < 3e-15
    assert np.max(np.abs(identity["spreading"] - expected_jacobian)) < 1e-14
    for radius in radii[1:]:
        path = radial_connection(np.asarray(radius), AIR_BEFORE_M, SLAB_THICKNESS_M,
                                 AIR_AFTER_M, SLAB_IOR)
        assert abs(float(path["displacement"]) - radius) < 2e-15
        assert abs(float(path["sin_air"]) - SLAB_IOR * float(path["sin_glass"])) < 2e-15
        h = 1e-5
        upper = radial_connection(np.asarray(radius + h), AIR_BEFORE_M, SLAB_THICKNESS_M,
                                  AIR_AFTER_M, SLAB_IOR)
        lower = radial_connection(np.asarray(radius - h), AIR_BEFORE_M, SLAB_THICKNESS_M,
                                  AIR_AFTER_M, SLAB_IOR)
        # Independent finite-difference derivative of the source solid angle
        # mapping: Omega cone = 2*pi*(1-cos(theta_air)).
        omega_upper = 2.0 * math.pi * (1.0 - float(upper["cos_air"]))
        omega_lower = 2.0 * math.pi * (1.0 - float(lower["cos_air"]))
        numerical_spreading = (omega_upper - omega_lower) / (4.0 * math.pi * radius * h)
        assert abs(numerical_spreading - float(path["spreading"])) < 2e-9
    try:
        radial_connection(np.asarray(0.1), AIR_BEFORE_M, SLAB_THICKNESS_M,
                          AIR_AFTER_M, 0.8)
    except ValueError:
        pass
    else:
        raise AssertionError("IOR below one must not enter this bisection domain")
    # The three globally fixed dipoles remain correlated across directions.
    # As two rays coincide, their full-vector overlap tends to their diagonal
    # power even when s/p gauges flip at normal incidence.
    tiny = 1e-9
    _, near_a = path_world_modes(np.asarray(tiny), np.asarray(0.0))
    _, near_b = path_world_modes(np.asarray(-tiny), np.asarray(0.0))
    near_overlap = float(world_mode_overlap(near_a, near_b))
    near_diagonal_fraction = 0.5 * (float(near_a["amplitude_s"])**2 +
                                    float(near_a["amplitude_p"])**2)
    assert abs(near_overlap - near_diagonal_fraction) < 1e-14

    # Independent cross-plane check: a source ray in xy and one in xz have
    # orthogonal incidence planes. Compare the dyad expression to explicit
    # three-axis vector fields, rather than to a local s/s + p/p shortcut.
    tilt = 0.3
    cos_tilt = math.sqrt(1.0 - tilt**2)
    left = {"s": np.asarray([0.0, 0.0, 1.0]),
            "p": np.asarray([-tilt, cos_tilt, 0.0]),
            "amplitude_s": np.asarray(0.8), "amplitude_p": np.asarray(0.6)}
    right = {"s": np.asarray([0.0, -1.0, 0.0]),
             "p": np.asarray([-tilt, 0.0, cos_tilt]),
             "amplitude_s": np.asarray(0.7), "amplitude_p": np.asarray(0.9)}
    explicit_overlap = 0.0
    for axis in np.eye(3):
        field_a = (left["amplitude_s"] * np.dot(axis, left["s"]) * left["s"] +
                   left["amplitude_p"] * np.dot(axis, left["p"]) * left["p"]) / math.sqrt(2.0)
        field_b = (right["amplitude_s"] * np.dot(axis, right["s"]) * right["s"] +
                   right["amplitude_p"] * np.dot(axis, right["p"]) * right["p"]) / math.sqrt(2.0)
        explicit_overlap += np.dot(field_a, field_b)
    cross_plane_overlap = float(world_mode_overlap(left, right))
    assert abs(cross_plane_overlap - explicit_overlap) < 1e-15
    local_gauge_only = 0.5 * (left["amplitude_s"] * right["amplitude_s"] *
                              np.dot(left["s"], right["s"]) +
                              left["amplitude_p"] * right["amplitude_p"] *
                              np.dot(left["p"], right["p"]))
    assert abs(cross_plane_overlap - local_gauge_only) > 0.1
    return {"normal_optical_length_m": expected_opl,
            "normal_spreading_inverse_m2": expected_spreading,
            "normal_interface_power_each_mode": expected_interface_power,
            "ior_one_radii_m": radii.tolist(),
            "co_directional_world_mode_overlap": near_overlap,
            "co_directional_diagonal_fraction": near_diagonal_fraction,
            "cross_plane_world_mode_overlap": cross_plane_overlap,
            "cross_plane_explicit_axis_sum": explicit_overlap,
            "checks": "normal incidence, IOR=1 free-space identity, Snell residual, finite-difference Jacobian, IOR<1 domain rejection, co-directional world-mode limit, independent cross-plane axis sum"}


def diffuse_material(name, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (*color, 1.0)
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


def glass_material():
    material = bpy.data.materials.new("Ideal lossless glass IOR 1.5")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.distribution = "GGX"
    glass.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.0
    glass.inputs["IOR"].default_value = SLAB_IOR
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(glass.outputs["BSDF"], output.inputs["Surface"])
    return material


def configure_scene():
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
    scene.cycles.max_bounces = 3
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 2
    scene.cycles.transparent_max_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.sample_clamp_direct = 0
    scene.cycles.sample_clamp_indirect = 0
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.render.resolution_x = scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.view_settings.view_transform = "Standard"
    scene.unit_settings.system = "NONE"
    scene.unit_settings.scale_length = 1.0
    scene.cycles.coherent_max_interface_events = 2
    scene.cycles.coherent_polarization_mode = "VECTOR"
    scene.cycles.use_coherent_specular_connections = False

    world = bpy.data.worlds.new("Black world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.world = world

    detector_material = diffuse_material("Unit white Lambertian detector", (1, 1, 1))
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(1.0, 0.0, 0.0),
                                     rotation=(0.0, -math.pi / 2.0, 0.0))
    detector = bpy.context.object
    detector.name = "White detector x=1, facing source"
    detector.scale = (0.2, 0.2, 1.0)
    detector.data.materials.append(detector_material)
    detector.cycles.coherent_interface = "DETECTOR"

    material = glass_material()
    faces = []
    for name, x, rotation in (("Entry glass face x=0.3", AIR_BEFORE_M, -math.pi / 2.0),
                              ("Exit glass face x=0.6", AIR_BEFORE_M + SLAB_THICKNESS_M,
                               math.pi / 2.0)):
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(x, 0.0, 0.0),
                                         rotation=(0.0, rotation, 0.0))
        face = bpy.context.object
        face.name = name
        face.scale = (0.4, 0.4, 1.0)
        face.data.materials.append(material)
        face.cycles.coherent_interface = "GLASS"
        faces.append(face)

    sources = []
    for label, y in zip(("A", "B"), SOURCE_OFFSETS_Y_M):
        bpy.ops.object.light_add(type="POINT", location=(0.0, y, 0.0))
        source = bpy.context.object
        source.name = f"Coherent point source {label}"
        source.data.energy = SOURCE_ENERGY_W
        source.data.shadow_soft_size = 0.0
        source.data.cycles.coherence_group = 1
        source.data.cycles.coherence_wavelength_nm = WAVELENGTH_M * 1e9
        source.data.cycles.coherence_length_m = COHERENCE_LENGTH_M
        source.data.cycles.coherence_phase = 0.0
        sources.append(source)

    bpy.ops.object.camera_add(location=(0.8, 0.0, 0.0))
    camera = bpy.context.object
    camera.name = "Orthographic detector camera, after slab"
    camera.rotation_euler = (Vector((1.0, 0.0, 0.0)) - camera.location).to_track_quat(
        "-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 0.4
    camera.data.clip_start = 0.01
    camera.data.clip_end = 10.0
    scene.camera = camera
    scene["coherence_acceptance_scope"] = "direct two-transmission slab path only"
    scene["coherence_excluded_paths"] = "internal-reflection branches and full Fabry-Perot series"
    scene["coherence_reference"] = "independent Snell radial root; no image texture"
    return scene, detector, faces, sources, camera


def camera_world_at_detector(camera, detector_x, pixel_x, pixel_y):
    matrix = np.asarray([list(row) for row in camera.matrix_world], dtype=np.float64)
    position = matrix[:3, 3]
    right = matrix[:3, 0]
    up = matrix[:3, 1]
    forward = -matrix[:3, 2]
    local_x = (pixel_x / RESOLUTION - 0.5) * camera.data.ortho_scale
    local_y = (pixel_y / RESOLUTION - 0.5) * camera.data.ortho_scale
    origin_x = position[0] + right[0] * local_x + up[0] * local_y
    origin_y = position[1] + right[1] * local_x + up[1] * local_y
    origin_z = position[2] + right[2] * local_x + up[2] * local_y
    distance = (detector_x - origin_x) / forward[0]
    return origin_y + forward[1] * distance, origin_z + forward[2] * distance


def make_reference(sources, camera, output_dir):
    checks = validate_reference_model()
    rows, columns = np.indices((RESOLUTION, RESOLUTION), dtype=np.float64)
    center_y, center_z = camera_world_at_detector(camera, 1.0, columns + 0.5, rows + 0.5)
    roi = ((center_y >= ROI_BOUNDS_M["y"][0]) & (center_y <= ROI_BOUNDS_M["y"][1]) &
           (center_z >= ROI_BOUNDS_M["z"][0]) & (center_z <= ROI_BOUNDS_M["z"][1]))
    if not np.any(roi):
        raise AssertionError("Camera mapping produced an empty detector ROI")
    result = {name: np.zeros((RESOLUTION, RESOLUTION), dtype=np.float64)
              for name in ("phase_0", "phase_pi", "incoherent", "source_a_diagonal",
                           "source_b_diagonal", "pair_cross_phase_0", "world_mode_overlap")}
    pixel_offsets = (0.125, 0.375, 0.625, 0.875)
    source_positions = [np.asarray(tuple(source.location), dtype=np.float64)
                        for source in sources]
    max_snell_residual = 0.0
    for oy in pixel_offsets:
        for ox in pixel_offsets:
            y, z = camera_world_at_detector(camera, 1.0, columns + ox, rows + oy)
            paths = []
            fields = []
            for source in source_positions:
                radius = np.hypot(y - source[1], z - source[2])
                path, field = path_world_modes(y - source[1], z - source[2])
                max_snell_residual = max(max_snell_residual,
                                         float(np.max(np.abs(path["displacement"] - radius))))
                paths.append(path["optical_length"])
                fields.append(field)
            opd = paths[0] - paths[1]
            gamma = np.exp(-0.5 * (opd / COHERENCE_LENGTH_M) ** 2)
            phase = np.remainder(opd, WAVELENGTH_M) * (2.0 * math.pi / WAVELENGTH_M)
            overlap = world_mode_overlap(fields[0], fields[1])
            cross = (2.0 * np.sqrt(fields[0]["base"] * fields[1]["base"]) *
                     overlap * gamma * np.cos(phase))
            baseline = fields[0]["diagonal"] + fields[1]["diagonal"]
            sample = {"phase_0": baseline + cross,
                      "phase_pi": baseline - cross,
                      "incoherent": baseline,
                      "source_a_diagonal": fields[0]["diagonal"],
                      "source_b_diagonal": fields[1]["diagonal"],
                      "pair_cross_phase_0": cross,
                      "world_mode_overlap": overlap}
            for name in result:
                result[name] += sample[name] / 16.0
    if max_snell_residual > 2e-14:
        raise AssertionError(f"Snell radial residual {max_snell_residual}")
    for name, values in result.items():
        if not np.isfinite(values).all() or (name not in ("pair_cross_phase_0",) and
                                             np.min(values) < -1e-12):
            raise AssertionError(f"Invalid {name} reference")
    np.savez_compressed(output_dir / "snell_slab_reference.npz",
                        detector_y_m=center_y, detector_z_m=center_z, roi_mask=roi,
                        phase_0_radiance=result["phase_0"],
                        phase_pi_radiance=result["phase_pi"],
                        incoherent_radiance=result["incoherent"],
                        source_a_diagonal_radiance=result["source_a_diagonal"],
                        source_b_diagonal_radiance=result["source_b_diagonal"],
                        pair_cross_phase_0_radiance=result["pair_cross_phase_0"],
                        world_mode_overlap=result["world_mode_overlap"])
    assumptions = {
        "status": "INDEPENDENT DOUBLE-PRECISION ANALYTIC REFERENCE; NOT A RENDER",
        "path_inventory": "one ordered air-to-glass and glass-to-air transmission path per source; two interface events total",
        "excluded_paths": "all internal-reflection sequences and the full coherent Fabry-Perot series",
        "source_positions_m": [p.tolist() for p in source_positions],
        "source_energy_w": [SOURCE_ENERGY_W] * len(source_positions),
        "slab_face_x_m": [AIR_BEFORE_M, AIR_BEFORE_M + SLAB_THICKNESS_M],
        "slab_ior": SLAB_IOR,
        "wavelength_m": WAVELENGTH_M,
        "coherence_length_m": COHERENCE_LENGTH_M,
        "source_phase_cases_radians": {"phase_0": [0.0, 0.0], "phase_pi": [0.0, math.pi]},
        "radial_root": "rho=(a+b)*tan(theta_air)+t*tan(theta_glass), sin(theta_air)=n*sin(theta_glass); 52 double-precision bisections",
        "optical_path_m": "(a+b)/cos(theta_air)+n*t/cos(theta_glass)",
        "geometric_spreading": "dOmega_source/dA_detector=sin(theta_air)/(rho*d_rho/d_theta_air); normal limit 1/(a+b+t/n)^2",
        "fresnel": "independent tangential-E s/p admittances Ys=n*cos(theta), Yp=n/cos(theta); real flux-normalized slab amplitudes sqrt(T_entry*T_exit)",
        "polarization": "three mutually independent fixed world-axis dipoles X/Y/Z; each emits sqrt(1/2) times its ray-transverse projection so summed source power is direction independent. Corresponding axes across the two sources share the declared phase; distinct axes are incoherent. Each complete path maps its axis modes through its own s/p basis to a full 3D field at the neutral detector; pair overlap includes s/p cross-basis terms",
        "radiance": "P/(4*pi^2)*geometric_spreading times half the sum of slab s/p transmitted powers on a unit white Lambertian detector",
        "pair_model": "baseline=sum_source_diagonals; cross=2*sqrt(base_a*base_b)*world_axis_field_overlap*gamma*cos(2*pi*OPD/lambda+phase_a-phase_b), with world_axis_field_overlap=1/2 Frobenius product of transported transverse dyads",
        "incoherent_control": "same diagonal baseline with no cross term: either zero coherence length with connector off, or positive coherence lengths in distinct source groups with connector on",
        "pixel_filter": "Cycles BOX width 1.0; independent 4x4 equal-weight subpixel quadrature",
        "image_row_origin": "bottom-left Blender pixel order",
        "roi_world_bounds_m": ROI_BOUNDS_M,
        "roi_pixel_count": int(np.count_nonzero(roi)),
        "camera_matrix_world": [list(row) for row in camera.matrix_world],
        "max_snell_displacement_residual_m": max_snell_residual,
        "analytic_checks": checks,
        "acceptance_policy": "compare raw absolute radiance and phase difference in the predeclared ROI; finite pixels alone do not pass",
        "render_policy": "editable .blend scenes only; fixture generation performs no rendering",
    }
    (output_dir / "snell_slab_reference.json").write_text(
        json.dumps(assumptions, indent=2) + "\n", encoding="utf-8")


def main():
    if "--" not in sys.argv:
        raise SystemExit("Expected output directory after --")
    args = sys.argv[sys.argv.index("--") + 1:]
    output_dir = Path(args[0]).resolve()
    if "--reference-only-from" in args:
        source_dir = Path(args[args.index("--reference-only-from") + 1]).resolve()
        previous = json.loads((source_dir / "manifest.json").read_text())
        from types import SimpleNamespace
        old_reference = json.loads((source_dir / "snell_slab_reference.json").read_text())
        camera = SimpleNamespace(matrix_world=np.asarray(old_reference["camera_matrix_world"]),
                                 data=SimpleNamespace(ortho_scale=0.4))
        sources = [SimpleNamespace(location=np.asarray(position))
                   for position in old_reference["source_positions_m"]]
        output_dir.mkdir(parents=True, exist_ok=False)
        make_reference(sources, camera, output_dir)
        reference_manifest = {
            "scope": "revised independent double slab reference; globally projected three-axis source ensemble",
            "original_fixture_manifest": str(source_dir / "manifest.json"),
            "original_fixture_manifest_sha256": sha256(source_dir / "manifest.json"),
            "original_blend_scenes": previous["scene_variants"],
            "original_reference": previous["reference"],
            "updated_oracle_script": str(Path(__file__).resolve()),
            "updated_oracle_script_sha256": sha256(Path(__file__).resolve()),
            "new_reference": {name: {"path": str(output_dir / name),
                                     "sha256": sha256(output_dir / name)}
                              for name in ("snell_slab_reference.npz", "snell_slab_reference.json")},
            "render_policy": "reference-only Python calculation; Blender not started, no render",
        }
        (output_dir / "reference_manifest.json").write_text(
            json.dumps(reference_manifest, indent=2) + "\n", encoding="utf-8")
        print(f"COHERENT_SLAB_WORLD_REFERENCE {output_dir}", flush=True)
        return
    if bpy is None:
        raise SystemExit("Blender Python is required to create editable fixtures")
    enable_connector = "--specular-connections" in args[1:]
    output_dir.mkdir(parents=True, exist_ok=False)
    scene, detector, faces, sources, camera = configure_scene()
    bpy.context.view_layer.update()
    make_reference(sources, camera, output_dir)
    scenes = {}
    variants = (("phase_0", (0.0, 0.0), COHERENCE_LENGTH_M, (1, 1)),
                ("phase_pi", (0.0, math.pi), COHERENCE_LENGTH_M, (1, 1)),
                ("incoherent_bdpt_control", (0.0, 0.0), 0.0, (1, 1)),
                ("incoherent_connector_control", (0.0, 0.0), COHERENCE_LENGTH_M, (1, 2)))
    for name, phases, length, groups in variants:
        scene.cycles.use_coherent_specular_connections = (
            enable_connector and name != "incoherent_bdpt_control")
        for source, phase, group in zip(sources, phases, groups):
            source.data.cycles.coherence_phase = phase
            source.data.cycles.coherence_length_m = length
            source.data.cycles.coherence_group = group
        scene.render.filepath = f"//{name}"
        scene["coherence_acceptance_variant"] = name
        scene["coherence_expected_state"] = (
            "INCOHERENT TWO-TRANSMISSION CONTROL; DISTINCT SOURCE GROUPS" if groups == (1, 2) else
            "INCOHERENT TWO-TRANSMISSION CONTROL; CONNECTOR OFF" if length == 0 else
            "TWO-TRANSMISSION SLAB PHASE ACCEPTANCE; GLASS CONNECTOR REQUIRED"
        )
        path = output_dir / f"{name}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        scenes[name] = {"path": str(path), "sha256": sha256(path)}
    manifest = {
        "scope": "two coherent sources through one parallel lossless glass slab",
        "specular_connections_enabled": enable_connector,
        "binary": str(Path(bpy.app.binary_path).resolve()),
        "binary_sha256": sha256(Path(bpy.app.binary_path).resolve()),
        "scene_script": str(Path(__file__).resolve()),
        "scene_script_sha256": sha256(Path(__file__).resolve()),
        "scene_variants": scenes,
        "reference": {name: {"path": str(output_dir / name),
                             "sha256": sha256(output_dir / name)}
                      for name in ("snell_slab_reference.npz", "snell_slab_reference.json")},
        "geometry": {"detector_center_m": list(detector.location),
                     "detector_size_m": [0.4, 0.4],
                     "slab_face_centers_m": [list(face.location) for face in faces],
                     "slab_aperture_m": [0.8, 0.8],
                     "source_positions_m": [list(source.location) for source in sources]},
        "render_policy": "no render performed; fixed 512 samples, adaptive sampling and denoising off",
        "known_current_renderer_limit": "bounded two-event vector Glass pilot; internal-reflection paths with more than two events and the full Fabry-Perot series are outside this fixture's reference",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    print(f"COHERENT_SLAB_FIXTURE {output_dir}", flush=True)


if __name__ == "__main__":
    main()
