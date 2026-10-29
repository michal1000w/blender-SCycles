# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Python stand-ins for the shader nodes that only exist in the blender-SCycles fork.

Used in two places:
* The JCycles add-on inside an official Blender registers them, so users can add the fork's
  nodes (and the fork's extra inputs of upstream nodes) with every socket and setting. Inside,
  each stand-in holds a small node group with the closest official node, so EEVEE, Material
  Preview and the official Cycles still show something sensible.
* The render worker (the fork's own Blender) registers the same classes to read the saved
  values, then replaces every stand-in with the real fork node (`convert_tree_to_native`).

Everything is generated from `node_specs.json`, which the builder writes by asking the fork build
for the exact node definitions.
"""

import re

import bpy

STANDIN_PREFIX = "JCycles"
GROUP_PREFIX = ".JCycles "

SOCKET_BASES = {
    'VALUE': "NodeSocketFloat",
    'INT': "NodeSocketInt",
    'BOOLEAN': "NodeSocketBool",
    'VECTOR': "NodeSocketVector",
    'RGBA': "NodeSocketColor",
    'STRING': "NodeSocketString",
    'SHADER': "NodeSocketShader",
}


def standin_idname(native_idname):
    return STANDIN_PREFIX + native_idname


def standin_label(spec):
    if spec["kind"] == "extended":
        return spec["label"] + " (JCycles)"
    return spec["label"]


def native_idname_of(standin):
    if standin.startswith(STANDIN_PREFIX + "ShaderNode"):
        return standin[len(STANDIN_PREFIX):]
    return None


def _camel_to_upper_snake(text):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text).upper()


def socket_type_and_subtype(socket_spec):
    base = SOCKET_BASES.get(socket_spec["type"])
    idname = socket_spec["bl_idname"]
    if base is None:
        return idname, None
    subtype = None
    if idname.startswith(base) and len(idname) > len(base):
        subtype = _camel_to_upper_snake(idname[len(base):])
    return base, subtype


# -----------------------------------------------------------------------------
# Class generation


def _prop_kwargs_common(prop):
    return {"name": prop["name"], "description": prop.get("description", "")}


def _make_property(prop, update):
    kind = prop["type"]
    kwargs = _prop_kwargs_common(prop)
    if update is not None:
        kwargs["update"] = update
    if kind == 'ENUM':
        items = [(i[0], i[1], i[2], int(i[3])) for i in prop["items"]]
        if prop.get("is_enum_flag"):
            return bpy.props.EnumProperty(items=items, options={'ENUM_FLAG'},
                                          default=set(prop.get("value") or prop.get("default") or []), **kwargs)
        default = prop.get("value") or prop.get("default") or items[0][0]
        if default not in {i[0] for i in items}:
            default = items[0][0]
        return bpy.props.EnumProperty(items=items, default=default, **kwargs)
    if kind in {'FLOAT', 'INT'}:
        is_float = kind == 'FLOAT'
        default = prop.get("value", prop.get("default"))
        length = prop.get("array_length", 0)
        kwargs.update(min=prop["hard_min"], max=prop["hard_max"],
                      soft_min=prop["soft_min"], soft_max=prop["soft_max"])
        if is_float:
            kwargs["precision"] = prop.get("precision", 3)
            kwargs["step"] = max(1, int(prop.get("step", 3)))
        subtype = prop.get("subtype", 'NONE')
        if length:
            maker = bpy.props.FloatVectorProperty if is_float else bpy.props.IntVectorProperty
            kwargs["size"] = length
            kwargs["default"] = tuple(default) if default is not None else (0,) * length
        else:
            maker = bpy.props.FloatProperty if is_float else bpy.props.IntProperty
            kwargs["default"] = default if default is not None else 0
        try:
            return maker(subtype=subtype, **kwargs)
        except TypeError:
            return maker(**kwargs)
    if kind == 'BOOLEAN':
        return bpy.props.BoolProperty(default=bool(prop.get("value", prop.get("default", False))), **kwargs)
    if kind == 'STRING':
        return bpy.props.StringProperty(default=prop.get("value") or prop.get("default") or "", **kwargs)
    if kind == 'POINTER':
        struct = getattr(bpy.types, prop["fixed_type"], None)
        if struct is None:
            return None
        return bpy.props.PointerProperty(type=struct, **kwargs)
    return None


def _on_setting_changed(self, context):
    try:
        refresh_node(self)
    except Exception as ex:
        print("JCycles: cannot refresh node {!r}: {:s}".format(self.name, str(ex)))


def make_node_classes(specs, with_ui=True):
    classes = []
    for idname, spec in specs["nodes"].items():
        annotations = {}
        for prop in spec["properties"]:
            made = _make_property(prop, _on_setting_changed if with_ui else None)
            if made is not None:
                annotations[prop["identifier"]] = made

        namespace = {
            "__annotations__": annotations,
            "bl_idname": standin_idname(idname),
            "bl_label": standin_label(spec),
            "bl_description": spec.get("description", ""),
            "bl_width_default": spec.get("width", 140.0),
            "jcycles_native_idname": idname,
            "poll": classmethod(lambda cls, ntree: ntree.bl_idname == 'ShaderNodeTree'),
        }
        if with_ui:
            namespace.update({
                "init": _node_init,
                "copy": _node_copy,
                "free": _node_free,
                "draw_buttons": _node_draw_buttons,
                "draw_buttons_ext": _node_draw_buttons,
            })
        cls = type("JCYCLES_NODE_" + idname, (bpy.types.ShaderNodeCustomGroup,), namespace)
        classes.append(cls)
    return classes


# -----------------------------------------------------------------------------
# Host side: node instances


_specs_cache = [None]


def set_specs(specs):
    _specs_cache[0] = specs


def spec_of(node):
    specs = _specs_cache[0]
    if specs is None:
        return None
    return specs["nodes"].get(getattr(node, "jcycles_native_idname", ""))


def _set_interface_value(item, attr, value):
    try:
        setattr(item, attr, value)
    except Exception:
        pass


def build_group(spec):
    """Node group with the sockets of the fork node and the closest official node inside."""
    tree = bpy.data.node_groups.new(GROUP_PREFIX + spec["label"], 'ShaderNodeTree')
    interface = tree.interface
    panels = {}
    closed = spec.get("panel_closed", {})
    for socket in spec["inputs"]:
        parent = None
        if socket.get("panel"):
            parent = panels.get(socket["panel"])
            if parent is None:
                parent = interface.new_panel(socket["panel"], default_closed=bool(closed.get(socket["panel"], True)))
                panels[socket["panel"]] = parent
        base, subtype = socket_type_and_subtype(socket)
        kwargs = {"in_out": 'INPUT', "socket_type": base}
        if parent is not None:
            kwargs["parent"] = parent
        item = interface.new_socket(socket["name"], **kwargs)
        _set_interface_value(item, "description", socket.get("description", ""))
        if subtype:
            _set_interface_value(item, "subtype", subtype)
        if "default" in socket and socket["default"] is not None and hasattr(item, "default_value"):
            _set_interface_value(item, "default_value", socket["default"])
        rng = socket.get("range")
        if rng:
            _set_interface_value(item, "min_value", rng[0])
            _set_interface_value(item, "max_value", rng[1])
        _set_interface_value(item, "hide_value", bool(socket.get("hide_value")))
    for socket in spec["outputs"]:
        base, _subtype = socket_type_and_subtype(socket)
        interface.new_socket(socket["name"], in_out='OUTPUT', socket_type=base)

    group_in = tree.nodes.new('NodeGroupInput')
    group_in.location = (-400, 0)
    group_out = tree.nodes.new('NodeGroupOutput')
    group_out.location = (300, 0)
    fallback_idname = spec.get("fallback")
    if fallback_idname:
        try:
            fallback = tree.nodes.new(fallback_idname)
        except Exception:
            fallback = None
        if fallback is not None:
            fallback.name = "Fallback"
            used = set()
            for out_socket in group_in.outputs:
                if not out_socket.name or out_socket.name in used:
                    continue
                target = fallback.inputs.get(out_socket.name)
                if target is not None and target.enabled and target.type == out_socket.type:
                    tree.links.new(out_socket, target)
                    used.add(out_socket.name)
            if fallback.outputs and group_out.inputs:
                tree.links.new(fallback.outputs[0], group_out.inputs[0])
    return tree


def _node_init(self, context):
    spec = spec_of(self)
    if spec is None:
        return
    self.node_tree = build_group(spec)
    self.width = spec.get("width", self.width)
    refresh_node(self)


def _node_copy(self, node):
    if node.node_tree is not None:
        self.node_tree = node.node_tree.copy()


def _node_free(self):
    tree = self.node_tree
    if tree is not None and tree.users <= 1 and tree.name.startswith(GROUP_PREFIX):
        self.node_tree = None
        try:
            bpy.data.node_groups.remove(tree)
        except Exception:
            pass


def _node_draw_buttons(self, context, layout):
    spec = spec_of(self)
    if spec is None:
        layout.label(text="JCycles node definitions missing", icon='ERROR')
        return
    for prop in spec["properties"]:
        if hasattr(self, prop["identifier"]):
            layout.prop(self, prop["identifier"])


def _availability_row(node, spec):
    table = spec.get("availability") or []
    if not table:
        return None
    for row in table:
        if all(getattr(node, key, None) == value for key, value in row["settings"].items()):
            return row
    return None


def refresh_node(node):
    """Show the sockets the fork node shows for the current settings, mirror the settings into
    the fallback node."""
    spec = spec_of(node)
    if spec is None:
        return
    row = _availability_row(node, spec)
    flags = row["inputs"] if row else [s.get("enabled", True) for s in spec["inputs"]]
    for socket, available in zip(node.inputs, flags):
        if socket.hide != (not available):
            socket.hide = not available
    tree = node.node_tree
    if tree is None:
        return
    fallback = tree.nodes.get("Fallback")
    if fallback is None:
        return
    for prop in spec["properties"]:
        identifier = prop["identifier"]
        if prop["type"] == 'POINTER' or not hasattr(fallback, identifier):
            continue
        try:
            setattr(fallback, identifier, getattr(node, identifier))
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Replacing nodes (both directions), keeping links, values and animation


def _socket_map(src_sockets, dst_sockets):
    """Index in dst for every src socket, matched by name and occurrence of that name."""
    seen = {}
    positions = {}
    for index, socket in enumerate(dst_sockets):
        positions.setdefault(socket.name, []).append(index)
    result = []
    for socket in src_sockets:
        k = seen.get(socket.name, 0)
        seen[socket.name] = k + 1
        candidates = positions.get(socket.name, [])
        result.append(candidates[k] if k < len(candidates) else None)
    return result


def _copy_value(src, dst):
    if not hasattr(src, "default_value") or not hasattr(dst, "default_value"):
        return
    value = src.default_value
    try:
        if hasattr(value, "__len__") and not isinstance(value, str):
            dst_len = len(dst.default_value)
            values = list(value)[:dst_len] + list(dst.default_value)[len(value):]
            dst.default_value = values
        else:
            dst.default_value = value
    except Exception:
        pass


def _iter_fcurves(anim_data):
    if anim_data is None:
        return
    for fcurve in anim_data.drivers:
        yield fcurve
    action = anim_data.action
    if action is None:
        return
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        try:
            for fcurve in legacy:
                yield fcurve
        except Exception:
            pass
    for layer in getattr(action, "layers", ()):
        for strip in layer.strips:
            for channelbag in getattr(strip, "channelbags", ()):
                for fcurve in channelbag.fcurves:
                    yield fcurve


def _remap_animation(tree, node_name, input_map):
    if not any(i is not None and i != j for j, i in enumerate(input_map)):
        return
    prefix = 'nodes["{:s}"].inputs['.format(node_name.replace('"', '\\"'))
    pattern = re.compile(re.escape(prefix) + r"(\d+)\]")
    for fcurve in _iter_fcurves(tree.animation_data):
        path = fcurve.data_path
        match = pattern.search(path)
        if match is None:
            continue
        index = int(match.group(1))
        if index < len(input_map) and input_map[index] is not None:
            fcurve.data_path = path[:match.start(1)] + str(input_map[index]) + path[match.end(1):]


def replace_node(tree, old, new_idname, properties):
    """Replace `old` with a new node of type `new_idname`, keeping name, links and values."""
    new = tree.nodes.new(new_idname)
    for attr in ("location", "width", "label", "parent", "mute", "hide", "use_custom_color", "color"):
        try:
            setattr(new, attr, getattr(old, attr))
        except Exception:
            pass
    try:
        new.location = old.location
    except Exception:
        pass
    for prop in properties:
        identifier = prop["identifier"]
        if hasattr(old, identifier) and hasattr(new, identifier):
            try:
                setattr(new, identifier, getattr(old, identifier))
            except Exception:
                pass
    if hasattr(new, "node_tree") and new.node_tree is not None:
        refresh_node(new)

    input_map = _socket_map(old.inputs, new.inputs)
    output_map = _socket_map(old.outputs, new.outputs)
    for i, socket in enumerate(old.inputs):
        j = input_map[i]
        if j is None:
            continue
        target = new.inputs[j]
        _copy_value(socket, target)
        for link in list(socket.links):
            tree.links.new(link.from_socket, target)
    for i, socket in enumerate(old.outputs):
        j = output_map[i]
        if j is None:
            continue
        source = new.outputs[j]
        for link in list(socket.links):
            tree.links.new(source, link.to_socket)
    name = old.name
    if old.select:
        new.select = True
    if tree.nodes.active == old:
        tree.nodes.active = new
    tree.nodes.remove(old)
    new.name = name
    _remap_animation(tree, name, input_map)
    return new


def convert_tree_to_native(tree, specs):
    """Worker side: replace every JCycles stand-in with the real fork node."""
    converted = 0
    for node in list(tree.nodes):
        native = native_idname_of(node.bl_idname)
        if native is None or native not in specs["nodes"]:
            continue
        replace_node(tree, node, native, specs["nodes"][native]["properties"])
        converted += 1
    return converted
