# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Shader nodes of this fork that official Blender does not have (or has with fewer inputs).

JCycles runs inside an official Blender, whose C++ code knows nothing about these nodes or the
extra inputs. For each entry the JCycles add-on registers a Python node with exactly the sockets
and settings of this fork's node (read from the built Blender at package time). The render worker
replaces those Python nodes with the real nodes of this fork before rendering.

When a fork change adds a shader node or adds inputs to an upstream node, add it here. The
builder fails when a listed node or socket no longer exists in the build, so renames are noticed.

`kind`:
  "new"       the node type does not exist in official Blender.
  "extended"  official Blender has the node, this fork adds inputs (listed in `extra_inputs`).
`fallback`: node type used inside the Python node so that EEVEE, Material Preview and official
  Cycles still show something sensible. Sockets with the same name are connected to it.
  `None` means the node has no effect outside JCycles.
"""

FORK_NODES = {
    "ShaderNodeBsdfDiffraction": {
        "kind": "new",
        "fallback": "ShaderNodeBsdfMetallic",
    },
    "ShaderNodeVolumeFast": {
        "kind": "new",
        "fallback": "ShaderNodeVolumePrincipled",
    },
    "ShaderNodeVolumeIOR": {
        "kind": "new",
        "fallback": None,
    },
    "ShaderNodeBsdfGlass": {
        "kind": "extended",
        "fallback": "ShaderNodeBsdfGlass",
        "extra_inputs": [
            "Polarizer", "Polarizer Angle",
            "Diffraction Weight", "Diffraction Pitch", "Diffraction Depth",
            "Diffraction Duty Cycle",
        ],
    },
    "ShaderNodeBsdfGlossy": {
        "kind": "extended",
        "fallback": "ShaderNodeBsdfGlossy",
        "extra_inputs": [
            "Diffraction Weight", "Diffraction Pitch", "Diffraction Depth",
            "Diffraction Duty Cycle", "Diffraction Medium IOR",
        ],
    },
    "ShaderNodeBsdfMetallic": {
        "kind": "extended",
        "fallback": "ShaderNodeBsdfMetallic",
        "extra_inputs": [
            "Diffraction Weight", "Diffraction Pitch", "Diffraction Depth",
            "Diffraction Duty Cycle",
        ],
    },
    "ShaderNodeBsdfPrincipled": {
        "kind": "extended",
        "fallback": "ShaderNodeBsdfPrincipled",
        "extra_inputs": [
            "Diffraction Weight", "Diffraction Pitch", "Diffraction Depth",
            "Diffraction Duty Cycle",
        ],
    },
    "ShaderNodeBsdfRefraction": {
        "kind": "extended",
        "fallback": "ShaderNodeBsdfRefraction",
        "extra_inputs": [
            "Diffraction Weight", "Diffraction Pitch", "Diffraction Depth",
            "Diffraction Duty Cycle", "Tangent",
        ],
    },
    "ShaderNodeVolumePrincipled": {
        "kind": "extended",
        "fallback": "ShaderNodeVolumePrincipled",
        "extra_inputs": ["IOR", "Heat Refraction"],
    },
}
