# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Runs inside this repository's Blender build at JCycles package time:

  blender -b --factory-startup --python introspect.py -- <request.json> <output.json>

Writes the compile-time features of the Cycles build and the exact definition (sockets, ranges,
settings, socket availability per setting) of the fork shader nodes listed in the request.
"""

import itertools
import json
import math
import sys

import bpy


def to_json_value(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isinf(value):
            return 3.0e38 if value > 0 else -3.0e38
        return value
    try:
        return [to_json_value(v) for v in value]
    except TypeError:
        return str(value)


def cycles_features():
    import _cycles
    features = {}
    for name in dir(_cycles):
        if name.startswith("_"):
            continue
        value = getattr(_cycles, name)
        if callable(value):
            continue
        if isinstance(value, (bool, int, float, str, tuple, list)):
            features[name] = to_json_value(value)
    try:
        features["build_device_types"] = list(_cycles.get_device_types())
    except Exception as ex:
        features["build_device_types_error"] = str(ex)
    return features


def property_spec(prop, owner):
    spec = {
        "identifier": prop.identifier,
        "name": prop.name,
        "description": prop.description,
        "type": prop.type,
        "subtype": prop.subtype,
    }
    if prop.type == 'ENUM':
        spec["is_enum_flag"] = prop.is_enum_flag
        spec["items"] = [
            [item.identifier, item.name, item.description, item.value] for item in prop.enum_items
        ]
        spec["default"] = prop.default if not prop.is_enum_flag else list(prop.default_flag)
    elif prop.type in {'FLOAT', 'INT'}:
        spec["hard_min"] = to_json_value(prop.hard_min)
        spec["hard_max"] = to_json_value(prop.hard_max)
        spec["soft_min"] = to_json_value(prop.soft_min)
        spec["soft_max"] = to_json_value(prop.soft_max)
        spec["step"] = prop.step
        spec["array_length"] = prop.array_length
        spec["default"] = to_json_value(prop.default_array if prop.array_length else prop.default)
        if prop.type == 'FLOAT':
            spec["precision"] = prop.precision
            spec["unit"] = prop.unit
    elif prop.type == 'BOOLEAN':
        spec["default"] = prop.default
    elif prop.type == 'STRING':
        spec["default"] = prop.default
    elif prop.type == 'POINTER':
        spec["fixed_type"] = prop.fixed_type.identifier
    else:
        return None
    value = getattr(owner, prop.identifier, None)
    if prop.type == 'POINTER':
        value = None
    spec["value"] = to_json_value(value)
    return spec


def socket_spec(socket, panels):
    spec = {
        "identifier": socket.identifier,
        "name": socket.name,
        "bl_idname": socket.bl_idname,
        "type": socket.type,
        "description": getattr(socket, "description", ""),
        "hide_value": socket.hide_value,
        "enabled": socket.enabled,
        "is_multi_input": socket.is_multi_input,
        "panel": panels.get(socket.name) if panels else None,
    }
    if hasattr(socket, "default_value"):
        value = socket.default_value
        spec["default"] = to_json_value(value)
    if hasattr(socket, "subtype"):
        spec["subtype"] = socket.subtype
    return spec


def availability_table(node, enum_props):
    """Which inputs are available for every combination of the node's enum settings."""
    if not enum_props:
        return []
    choices = [[item[0] for item in prop["items"]] for prop in enum_props]
    combinations = list(itertools.product(*choices))
    if len(combinations) > 512:
        return []
    original = {prop["identifier"]: getattr(node, prop["identifier"]) for prop in enum_props}
    table = []
    for combination in combinations:
        for prop, value in zip(enum_props, combination):
            setattr(node, prop["identifier"], value)
        node.id_data.update_tag()
        bpy.context.view_layer.update()
        table.append({
            "settings": {prop["identifier"]: value for prop, value in zip(enum_props, combination)},
            "inputs": [socket.enabled for socket in node.inputs],
            "outputs": [socket.enabled for socket in node.outputs],
        })
    for identifier, value in original.items():
        setattr(node, identifier, value)
    node.id_data.update_tag()
    bpy.context.view_layer.update()
    return table


def node_spec(tree, idname, request):
    node = tree.nodes.new(idname)
    base = {prop.identifier for prop in bpy.types.ShaderNode.bl_rna.properties}
    properties = []
    for prop in node.bl_rna.properties:
        if prop.identifier in base or prop.is_readonly:
            continue
        spec = property_spec(prop, node)
        if spec is not None:
            properties.append(spec)
    panels = request.get("panels", {})
    spec = {
        "idname": idname,
        "label": node.bl_label,
        "description": node.bl_description,
        "width": node.width,
        "kind": request["kind"],
        "fallback": request.get("fallback"),
        "extra_inputs": request.get("extra_inputs", []),
        "panel_order": request.get("panel_order", []),
        "panel_closed": request.get("panel_closed", {}),
        "properties": properties,
        "inputs": [socket_spec(socket, panels) for socket in node.inputs],
        "outputs": [socket_spec(socket, None) for socket in node.outputs],
    }
    enum_props = [p for p in properties if p["type"] == 'ENUM' and not p["is_enum_flag"]]
    spec["availability"] = availability_table(node, enum_props)
    tree.nodes.remove(node)
    return spec


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    request_path, output_path = argv[0], argv[1]
    with open(request_path, encoding="utf-8") as fh:
        request = json.load(fh)

    result = {
        "blender_version": list(bpy.app.version),
        "blender_version_string": bpy.app.version_string,
        "blender_file_version": list(bpy.app.version_file),
        "build_hash": bpy.app.build_hash.decode("utf-8", "replace"),
        "build_branch": bpy.app.build_branch.decode("utf-8", "replace"),
        "build_date": bpy.app.build_date.decode("utf-8", "replace"),
        "build_platform": bpy.app.build_platform.decode("utf-8", "replace"),
        "cycles": cycles_features(),
        "nodes": {},
        "errors": [],
    }

    tree = bpy.data.node_groups.new("JCyclesIntrospect", 'ShaderNodeTree')
    for idname, node_request in request["nodes"].items():
        try:
            result["nodes"][idname] = node_spec(tree, idname, node_request)
        except Exception as ex:
            result["errors"].append("{:s}: {:s}".format(idname, str(ex)))
    bpy.data.node_groups.remove(tree)

    result["all_shader_nodes"] = sorted(
        cls.bl_rna.identifier for cls in bpy.types.ShaderNode.__subclasses__()
        if cls.bl_rna.identifier.startswith("ShaderNode")
    )

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("JCycles introspection written to", output_path)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
