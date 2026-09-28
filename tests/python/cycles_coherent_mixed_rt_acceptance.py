#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Editable two-source T/T plus T/R/T fixture and independent double oracle.

Run under Blender with ``-- OUTPUT_DIR``. This creates .blend scenes and analytic
arrays only; it never renders. The planar Glass/Mirror path inventory is bounded
to three events. Internal slab reflection routes requiring more events are not
part of this oracle.
"""

import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cycles_coherent_slab_acceptance as slab

try:
    import bpy
except ImportError:
    bpy = None


RESOLUTION = 512
SCENE_SCALE = 0.05
AIR_BEFORE = slab.AIR_BEFORE_M * SCENE_SCALE
SLAB_THICKNESS = slab.SLAB_THICKNESS_M * SCENE_SCALE
AIR_AFTER = slab.AIR_AFTER_M * SCENE_SCALE
DETECTOR_X = AIR_BEFORE + SLAB_THICKNESS + AIR_AFTER
SOURCE_ENERGY = slab.SOURCE_ENERGY_W * SCENE_SCALE**2
MIRROR_Y = 0.40 * SCENE_SCALE
MIRROR_X_MIN, MIRROR_X_MAX = 0.31 * SCENE_SCALE, 0.59 * SCENE_SCALE
MIRROR_Z_HALF = 0.30 * SCENE_SCALE
SOURCE_Y = (-25.0e-6, 25.0e-6)
CAMERA_WIDTH = 40.0e-6
ROI_HALF = 18.0e-6
COHERENCE_LENGTH = 1.0
TINY_COHERENCE_LENGTH = 1.0e-9
WAVELENGTH = slab.WAVELENGTH_M
PIXEL_OFFSETS = (0.125, 0.375, 0.625, 0.875)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dot(a, b):
    return np.sum(a * b, axis=-1)


def unit(vector):
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


def reflection_y(vector):
    """Householder reflection I-2*n*n^T, n=(0,1,0)."""
    return vector * np.asarray((1.0, -1.0, 1.0))


def fresnel_transmit_field(fields, direction_in, direction_out, normal, ni, nt):
    """Independent 3D electric-field dyad with flux-normalized s/p factors."""
    s = np.cross(normal, direction_in)
    small = np.linalg.norm(s, axis=-1) < 1.0e-13
    if np.any(small):
        fallback = np.broadcast_to(np.asarray((0.0, 0.0, 1.0)), s.shape)
        s = np.where(small[..., None], np.cross(fallback, direction_in), s)
    s = unit(s)
    p_in = np.cross(s, direction_in)
    p_out = np.cross(s, direction_out)
    ci = np.abs(dot(direction_in, normal))
    ct = np.abs(dot(direction_out, normal))
    ts = np.sqrt(slab.flux_transmission(ni, nt, ci, ct, "s"))
    tp = np.sqrt(slab.flux_transmission(ni, nt, ci, ct, "p"))
    projection_s = np.sum(fields * s[..., None, :], axis=-1)
    projection_p = np.sum(fields * p_in[..., None, :], axis=-1)
    return (ts[..., None, None] * projection_s[..., None] * s[..., None, :] +
            tp[..., None, None] * projection_p[..., None] * p_out[..., None, :])


def analytic_path(source_y, detector_y, detector_z, reflected):
    """Snell root to real or mirror-unfolded receiver, then physical field map."""
    target_y = 2.0 * MIRROR_Y - detector_y if reflected else detector_y
    dy = target_y - source_y
    dz = detector_z
    radius = np.hypot(dy, dz)
    root = slab.radial_connection(radius, AIR_BEFORE, SLAB_THICKNESS,
                                  AIR_AFTER, slab.SLAB_IOR)
    uy = np.divide(dy, radius, out=np.ones_like(radius), where=radius > 0.0)
    uz = np.divide(dz, radius, out=np.zeros_like(radius), where=radius > 0.0)
    air = np.stack((root["cos_air"], root["sin_air"] * uy,
                    root["sin_air"] * uz), axis=-1)
    glass = np.stack((root["cos_glass"], root["sin_glass"] * uy,
                      root["sin_glass"] * uz), axis=-1)
    shape = detector_y.shape
    source = np.broadcast_to(np.asarray((0.0, source_y, 0.0)), shape + (3,))
    entry = source + AIR_BEFORE / air[..., 0, None] * air
    exit_unfolded = entry + SLAB_THICKNESS / glass[..., 0, None] * glass
    valid = np.ones(shape, dtype=bool)
    mirror_x = np.full(shape, np.nan, dtype=np.float64)
    mirror_point = np.full(shape + (3,), np.nan, dtype=np.float64)
    physical_exit = exit_unfolded
    physical_glass_out = glass
    physical_air_out = air
    if reflected:
        # The unfolded straight glass segment must cross the finite mirror
        # between the two slab faces; reflection then maps all later points.
        mirror_t = (MIRROR_Y - entry[..., 1]) / glass[..., 1]
        mirror_point = entry + mirror_t[..., None] * glass
        mirror_x = mirror_point[..., 0]
        valid = ((entry[..., 1] < MIRROR_Y) &
                 (exit_unfolded[..., 1] > MIRROR_Y) &
                 (mirror_x > MIRROR_X_MIN) & (mirror_x < MIRROR_X_MAX) &
                 (np.abs(mirror_point[..., 2]) < MIRROR_Z_HALF))
        physical_exit = exit_unfolded.copy()
        physical_exit[..., 1] = 2.0 * MIRROR_Y - exit_unfolded[..., 1]
        physical_glass_out = reflection_y(glass)
        physical_air_out = reflection_y(air)
    detector = np.stack((np.full(shape, DETECTOR_X), detector_y, detector_z), axis=-1)
    # Independent segment OPL check; unfolding is an isometry. Invalid paths
    # are masked after calculation and are not silently projected to a patch.
    if reflected:
        geometric_opl = (np.linalg.norm(entry - source, axis=-1) +
                         slab.SLAB_IOR * np.linalg.norm(mirror_point - entry, axis=-1) +
                         slab.SLAB_IOR * np.linalg.norm(physical_exit - mirror_point, axis=-1) +
                         np.linalg.norm(detector - physical_exit, axis=-1))
    else:
        geometric_opl = (np.linalg.norm(entry - source, axis=-1) +
                         slab.SLAB_IOR * np.linalg.norm(physical_exit - entry, axis=-1) +
                         np.linalg.norm(detector - physical_exit, axis=-1))
    if np.max(np.abs(geometric_opl[valid] - root["optical_length"][valid])) > 2e-13:
        raise AssertionError("Unfolded and physical segment optical lengths disagree")
    if np.max(np.abs(root["displacement"] - radius)) > 2e-14:
        raise AssertionError("Independent Snell root displacement residual")
    eye = np.eye(3, dtype=np.float64)
    launch = (eye - air[..., :, None] * air[..., None, :]) / math.sqrt(2.0)
    fields = fresnel_transmit_field(launch, air, glass, np.asarray((-1.0, 0.0, 0.0)),
                                    1.0, slab.SLAB_IOR)
    if reflected:
        fields = reflection_y(fields)
    fields = fresnel_transmit_field(fields, physical_glass_out, physical_air_out,
                                    np.asarray((1.0, 0.0, 0.0)), slab.SLAB_IOR, 1.0)
    transverse_error = np.max(np.abs(np.sum(fields * physical_air_out[..., None, :],
                                              axis=-1)[valid]))
    if transverse_error > 2e-13:
        raise AssertionError("Final world-axis fields are not transverse")
    base = SOURCE_ENERGY / (4.0 * math.pi**2) * root["spreading"]
    diagonal = base * np.sum(fields * fields, axis=(-2, -1))
    return {"fields": fields, "base": base, "diagonal": np.where(valid, diagonal, 0.0),
            "opl": root["optical_length"], "valid": valid, "mirror_x": mirror_x,
            "entry_y": entry[..., 1], "exit_unfolded_y": exit_unfolded[..., 1]}


def sample_radiance(paths, phases, groups, coherence_length):
    diagonal = sum(path["diagonal"] for path in paths)
    pair_terms = {}
    total = diagonal.copy()
    for i, j in itertools.combinations(range(len(paths)), 2):
        left, right = paths[i], paths[j]
        if groups[i] != groups[j]:
            pair = np.zeros_like(diagonal)
        else:
            opd = left["opl"] - right["opl"]
            gamma = np.exp(-0.5 * (opd / coherence_length)**2)
            overlap = np.sum(left["fields"] * right["fields"], axis=(-2, -1))
            phase = 2.0 * math.pi * np.remainder(opd, WAVELENGTH) / WAVELENGTH
            pair = 2.0 * np.sqrt(left["base"] * right["base"]) * overlap * gamma * \
                   np.cos(phase + phases[i] - phases[j])
            pair = np.where(left["valid"] & right["valid"], pair, 0.0)
        pair_terms[f"pair_{i}_{j}"] = pair
        total += pair
    return total, diagonal, pair_terms


def validate_mixed_model():
    radius = 2.0 * MIRROR_Y
    root = slab.radial_connection(np.asarray(radius), AIR_BEFORE, SLAB_THICKNESS,
                                  AIR_AFTER, slab.SLAB_IOR)
    h = 1.0e-7
    upper = slab.radial_connection(np.asarray(radius + h), AIR_BEFORE, SLAB_THICKNESS,
                                   AIR_AFTER, slab.SLAB_IOR)
    lower = slab.radial_connection(np.asarray(radius - h), AIR_BEFORE, SLAB_THICKNESS,
                                   AIR_AFTER, slab.SLAB_IOR)
    omega_plus = 2.0 * math.pi * (1.0 - float(upper["cos_air"]))
    omega_minus = 2.0 * math.pi * (1.0 - float(lower["cos_air"]))
    finite_difference = (omega_plus - omega_minus) / (4.0 * math.pi * radius * h)
    relative_error = abs(finite_difference / float(root["spreading"]) - 1.0)
    if relative_error > 1.0e-8:
        raise AssertionError("Mixed unfolded-path spreading failed solid-angle finite difference")
    test_vector = np.asarray((0.31, -0.74, 0.59), dtype=np.float64)
    if np.max(np.abs(reflection_y(reflection_y(test_vector)) - test_vector)) > 1.0e-15:
        raise AssertionError("Mirror Householder map is not an involution")
    y = np.asarray([[0.0]])
    z = np.asarray([[0.0]])
    direct = analytic_path(0.0, y, z, False)
    mixed = analytic_path(0.0, y, z, True)
    overlap = float(np.sum(direct["fields"] * mixed["fields"]))
    normalized_overlap = overlap / math.sqrt(
        float(np.sum(direct["fields"] ** 2) * np.sum(mixed["fields"] ** 2)))
    if not bool(mixed["valid"][0, 0]) or normalized_overlap < 0.15:
        raise AssertionError("Mixed mirror path or vector overlap is not a useful test")
    return {"solid_angle_jacobian_relative_error": relative_error,
            "householder_involution": True,
            "central_direct_mixed_normalized_vector_overlap": normalized_overlap}


def configure_scene():
    scene, detector, faces, sources, camera = slab.configure_scene()
    for obj in bpy.data.objects:
        obj.location *= SCENE_SCALE
        if obj.type == "MESH":
            obj.scale *= SCENE_SCALE
    faces[0].name = f"Entry glass face x={AIR_BEFORE:.3f}m"
    faces[1].name = f"Exit glass face x={AIR_BEFORE + SLAB_THICKNESS:.3f}m"
    detector.name = f"White detector x={DETECTOR_X:.3f}m, facing source"
    scene.cycles.max_bounces = 4
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 2
    scene.cycles.coherent_max_interface_events = 3
    scene.camera.data.ortho_scale = CAMERA_WIDTH
    scene.camera.data.clip_start = 1.0e-5
    for source, y in zip(sources, SOURCE_Y):
        source.location.y = y
        source.data.energy = SOURCE_ENERGY
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.45 * SCENE_SCALE, MIRROR_Y, 0.0),
                                     rotation=(math.pi / 2.0, 0.0, 0.0))
    mirror = bpy.context.object
    mirror.name = "Unit ideal mirror inside glass at y=20mm"
    mirror.scale = ((MIRROR_X_MAX - MIRROR_X_MIN) / 2.0, MIRROR_Z_HALF, 1.0)
    material = bpy.data.materials.new("Unit ideal GGX mirror")
    material.use_nodes = True
    material.node_tree.nodes.clear()
    glossy = material.node_tree.nodes.new("ShaderNodeBsdfGlossy")
    glossy.distribution = "GGX"
    glossy.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy.inputs["Roughness"].default_value = 0.0
    output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(glossy.outputs["BSDF"], output.inputs["Surface"])
    mirror.data.materials.append(material)
    mirror.cycles.coherent_interface = "MIRROR"
    scene["coherence_acceptance_scope"] = "direct T/T and interior T/R/T, three events maximum"
    scene["coherence_excluded_paths"] = "internal slab reflection paths requiring more than three events"
    bpy.context.view_layer.update()
    return scene, detector, faces, sources, camera, mirror


def use_saved_scene_geometry(detector, faces, sources, mirror):
    """Use the float coordinates Blender stores, including phase-critical mirror y."""
    global AIR_BEFORE, SLAB_THICKNESS, AIR_AFTER, DETECTOR_X
    global MIRROR_Y, MIRROR_X_MIN, MIRROR_X_MAX, MIRROR_Z_HALF, SOURCE_ENERGY
    entry_x = float(faces[0].matrix_world.translation.x)
    exit_x = float(faces[1].matrix_world.translation.x)
    DETECTOR_X = float(detector.matrix_world.translation.x)
    AIR_BEFORE = entry_x
    SLAB_THICKNESS = exit_x - entry_x
    AIR_AFTER = DETECTOR_X - exit_x
    vertices = [mirror.matrix_world @ vertex.co for vertex in mirror.data.vertices]
    MIRROR_Y = float(mirror.matrix_world.translation.y)
    MIRROR_X_MIN = min(float(vertex.x) for vertex in vertices)
    MIRROR_X_MAX = max(float(vertex.x) for vertex in vertices)
    MIRROR_Z_HALF = max(abs(float(vertex.z)) for vertex in vertices)
    SOURCE_ENERGY = float(sources[0].data.energy)
    if not (AIR_BEFORE > 0.0 and SLAB_THICKNESS > 0.0 and AIR_AFTER > 0.0):
        raise AssertionError("Saved slab faces and detector are not ordered")


def make_reference(sources, camera, output_dir):
    slab_checks = slab.validate_reference_model()
    mixed_checks = validate_mixed_model()
    rows, columns = np.indices((RESOLUTION, RESOLUTION), dtype=np.float64)
    center_y, center_z = slab.camera_world_at_detector(camera, DETECTOR_X, columns + 0.5,
                                                        rows + 0.5)
    roi = (np.abs(center_y) <= ROI_HALF) & (np.abs(center_z) <= ROI_HALF)
    if not np.any(roi):
        raise AssertionError("Empty mixed-path ROI")
    names = ("phase_0", "phase_pi", "distinct_groups", "diagonal_control",
             "diagonal", "direct_diagonal", "mixed_diagonal", "same_source_cross",
             "direct_mixed_cross",
             "cross_source_cross_phase_0", "mirror_valid_count")
    result = {name: np.zeros((RESOLUTION, RESOLUTION), dtype=np.float64) for name in names}
    mirror_x_limits = [math.inf, -math.inf]
    source_ys = [float(source.location.y) for source in sources]
    for oy in PIXEL_OFFSETS:
        for ox in PIXEL_OFFSETS:
            y, z = slab.camera_world_at_detector(camera, DETECTOR_X, columns + ox, rows + oy)
            paths = [analytic_path(sy, y, z, reflected)
                     for sy in source_ys for reflected in (False, True)]
            for path in (paths[1], paths[3]):
                if not np.all(path["valid"][roi]):
                    raise AssertionError("Mixed T/R/T path misses finite mirror in ROI")
                mirror_x_limits[0] = min(mirror_x_limits[0],
                                         float(np.min(path["mirror_x"][roi])))
                mirror_x_limits[1] = max(mirror_x_limits[1],
                                         float(np.max(path["mirror_x"][roi])))
            zero = (0.0, 0.0, 0.0, 0.0)
            pi = (0.0, 0.0, math.pi, math.pi)
            phase0, diagonal, pairs0 = sample_radiance(paths, zero, (1, 1, 1, 1),
                                                         COHERENCE_LENGTH)
            phasepi, _, pairspi = sample_radiance(paths, pi, (1, 1, 1, 1),
                                                   COHERENCE_LENGTH)
            distinct, _, _ = sample_radiance(paths, zero, (1, 1, 2, 2),
                                              COHERENCE_LENGTH)
            tiny, _, _ = sample_radiance(paths, zero, (1, 1, 2, 2),
                                          TINY_COHERENCE_LENGTH)
            # Pair indices: 0=A direct, 1=A reflected, 2=B direct, 3=B reflected.
            same = pairs0["pair_0_1"] + pairs0["pair_2_3"]
            direct_mixed = sum(pairs0[key] for key in
                               ("pair_0_1", "pair_0_3", "pair_1_2", "pair_2_3"))
            cross_sources = sum(pair for key, pair in pairs0.items()
                                if key not in ("pair_0_1", "pair_2_3"))
            if np.max(np.abs((phase0 + phasepi) - 2.0 * (diagonal + same))) > 2e-12:
                raise AssertionError("Source phase flip changed same-source arm pairs")
            if np.max(np.abs(distinct - diagonal - same)) > 2e-12:
                raise AssertionError("Distinct source groups lost same-source coherence")
            if np.max(np.abs(tiny - diagonal)) > 2e-12:
                raise AssertionError("Tiny-coherence diagonal control has residual pairs")
            sample = {"phase_0": phase0, "phase_pi": phasepi,
                      "distinct_groups": distinct, "diagonal_control": tiny,
                      "diagonal": diagonal,
                      "direct_diagonal": paths[0]["diagonal"] + paths[2]["diagonal"],
                      "mixed_diagonal": paths[1]["diagonal"] + paths[3]["diagonal"],
                      "same_source_cross": same,
                      "direct_mixed_cross": direct_mixed,
                      "cross_source_cross_phase_0": cross_sources,
                      "mirror_valid_count": paths[1]["valid"].astype(float) +
                                             paths[3]["valid"].astype(float)}
            for name in names:
                result[name] += sample[name] / 16.0
    for name in ("phase_0", "phase_pi", "distinct_groups", "diagonal_control", "diagonal"):
        if not np.isfinite(result[name]).all() or np.min(result[name][roi]) < -2e-12:
            raise AssertionError(f"Invalid {name} analytic radiance")
    cross_arm_rms = float(np.sqrt(np.mean(result["direct_mixed_cross"][roi] ** 2)))
    if cross_arm_rms < 0.05 or cross_arm_rms / float(np.mean(result["diagonal"][roi])) < 0.1:
        raise AssertionError("Mixed-arm cross term is too weak for this acceptance fixture")
    np.savez_compressed(output_dir / "mixed_rt_reference.npz",
                        detector_y_m=center_y, detector_z_m=center_z, roi_mask=roi,
                        **{f"{name}_radiance": data for name, data in result.items()})
    report = {
        "scope": "independent double Snell virtual-receiver oracle; no production connector used",
        "path_inventory": "per source: direct entry-T/exit-T and interior entry-T/mirror-R/exit-T",
        "excluded": "internal-reflection routes requiring more than three events; no Fabry-Perot claim",
        "source_model": "three mutually independent world-axis dipoles projected transverse with sqrt(1/2) amplitude each; same axes correlate across paths and same-group sources",
        "polarization": "3D electric-field dyad at each dielectric face with independent s/p flux transmission; ideal mirror Householder I-2nn^T; full world-vector overlap at detector",
        "coherence": "all six unordered pairs of four complete fields evaluated; Gaussian mutual coherence exp(-OPD^2/(2Lc^2)); source phase pi flips only cross-source pairs",
        "path_field_order": ["source A direct T/T", "source A interior T/R/T",
                             "source B direct T/T", "source B interior T/R/T"],
        "coherence_length_m": COHERENCE_LENGTH,
        "diagonal_control_coherence_length_m": TINY_COHERENCE_LENGTH,
        "wavelength_m": WAVELENGTH,
        "mirror_y_m": MIRROR_Y,
        "geometry_input": "evaluated float world coordinates of saved Blender objects",
        "entry_x_m": AIR_BEFORE,
        "exit_x_m": AIR_BEFORE + SLAB_THICKNESS,
        "air_after_m": AIR_AFTER,
        "scene_scale_from_one_metre_slab": SCENE_SCALE,
        "source_energy_w": SOURCE_ENERGY,
        "detector_x_m": DETECTOR_X,
        "mirror_x_bounds_m": [MIRROR_X_MIN, MIRROR_X_MAX],
        "mirror_z_half_extent_m": MIRROR_Z_HALF,
        "solved_mirror_x_range_in_roi_m": mirror_x_limits,
        "minimum_mirror_to_exit_plane_x_gap_m": AIR_BEFORE + SLAB_THICKNESS - mirror_x_limits[1],
        "phase_0_direct_mixed_cross_rms_radiance": cross_arm_rms,
        "phase_0_direct_mixed_cross_rms_fraction_of_mean_diagonal": cross_arm_rms /
            float(np.mean(result["diagonal"][roi])),
        "source_y_m": source_ys,
        "roi_half_extent_yz_m": ROI_HALF,
        "roi_pixels": int(np.count_nonzero(roi)),
        "pixel_filter": "BOX width 1.0; independent 4x4 equal-weight subpixel quadrature",
        "phase_identity": "phase0+phasepi=2*(diagonal+same_source_cross); distinct_groups=diagonal+same_source_cross",
        "slab_reference_checks": slab_checks,
        "mixed_reference_checks": mixed_checks,
        "render_policy": "editable fixture and reference only; no render performed",
    }
    (output_dir / "mixed_rt_reference.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    if bpy is None or "--" not in sys.argv:
        raise SystemExit("Run under Blender with -- OUTPUT_DIR")
    output_dir = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    scene, detector, faces, sources, camera, mirror = configure_scene()
    use_saved_scene_geometry(detector, faces, sources, mirror)
    make_reference(sources, camera, output_dir)
    scenes = {}
    variants = (("phase_0", (0.0, 0.0), (1, 1), COHERENCE_LENGTH),
                ("phase_pi", (0.0, math.pi), (1, 1), COHERENCE_LENGTH),
                ("distinct_groups", (0.0, 0.0), (1, 2), COHERENCE_LENGTH),
                ("diagonal_control", (0.0, 0.0), (1, 2), TINY_COHERENCE_LENGTH))
    for name, phases, groups, length in variants:
        scene.cycles.use_coherent_specular_connections = True
        for source, phase, group in zip(sources, phases, groups):
            source.data.cycles.coherence_phase = phase
            source.data.cycles.coherence_group = group
            source.data.cycles.coherence_length_m = length
        scene.render.filepath = f"//{name}"
        scene["coherence_acceptance_variant"] = name
        scene["coherence_expected_state"] = "MIXED PLANAR T/T PLUS T/R/T ANALYTIC GATE"
        path = output_dir / f"{name}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        scenes[name] = {"path": str(path), "sha256": digest(path)}
    manifest = {
        "scope": "two-source parallel slab plus interior ideal mirror, at most three events",
        "binary": str(Path(bpy.app.binary_path).resolve()),
        "binary_sha256": digest(bpy.app.binary_path),
        "script": str(Path(__file__).resolve()), "script_sha256": digest(__file__),
        "scene_variants": scenes,
        "reference": {name: {"path": str(output_dir / name),
                             "sha256": digest(output_dir / name)}
                      for name in ("mixed_rt_reference.npz", "mixed_rt_reference.json")},
        "geometry": {"detector_center_m": list(detector.location),
                     "slab_face_centers_m": [list(face.location) for face in faces],
                     "mirror_center_m": list(mirror.location),
                     "mirror_scale": list(mirror.scale),
                     "source_positions_m": [list(source.location) for source in sources]},
        "render_policy": "fixture and independent analytic reference only; no render",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"COHERENT_MIXED_RT_FIXTURE {output_dir}", flush=True)


if __name__ == "__main__":
    main()
