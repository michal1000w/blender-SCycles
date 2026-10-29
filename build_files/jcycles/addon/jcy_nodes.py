# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Fork shader nodes in the official Blender's node editor."""

import bpy

from . import jcy_nodedefs, jcy_runtime

_classes = []
_patched = {}
_menu_hooks = []


def _specs():
    return jcy_nodedefs._specs_cache[0]


def _is_jcycles(context):
    return getattr(context, "engine", None) == 'JCYCLES'


# -----------------------------------------------------------------------------
# Add menu


def draw_jcycles_menu(self, context):
    layout = self.layout
    specs = _specs()
    if specs is None:
        return
    for kind, title in (("new", None), ("extended", "With fork inputs")):
        items = [(idname, spec) for idname, spec in specs["nodes"].items() if spec["kind"] == kind]
        if not items:
            continue
        if title:
            layout.separator()
            layout.label(text=title)
        for idname, spec in items:
            props = layout.operator("node.add_node", text=jcy_nodedefs.standin_label(spec))
            props.type = jcy_nodedefs.standin_idname(idname)
            props.use_transform = True


class NODE_MT_jcycles_shader_nodes(bpy.types.Menu):
    bl_idname = "NODE_MT_jcycles_shader_nodes"
    bl_label = "JCycles"

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space is not None and getattr(space, "tree_type", "") == 'ShaderNodeTree'

    def draw(self, context):
        draw_jcycles_menu(self, context)
        layout = self.layout
        layout.separator()
        layout.operator("node.jcycles_convert_to_fork", icon='NODETREE')
        layout.operator("node.jcycles_convert_to_official")


def _draw_add_menu_entry(self, context):
    space = context.space_data
    if space is None or getattr(space, "tree_type", "") != 'ShaderNodeTree':
        return
    self.layout.separator()
    self.layout.menu(NODE_MT_jcycles_shader_nodes.bl_idname, icon='NODE')


def _hook_add_menus():
    for name in ("NODE_MT_add", "NODE_MT_shader_node_add_all"):
        menu = getattr(bpy.types, name, None)
        if menu is not None:
            menu.append(_draw_add_menu_entry)
            _menu_hooks.append(menu)
            break
    context_menu = getattr(bpy.types, "NODE_MT_context_menu", None)
    if context_menu is not None:
        context_menu.append(_draw_context_menu)
        _menu_hooks.append(context_menu)


def _unhook_add_menus():
    for menu in _menu_hooks:
        for function in (_draw_add_menu_entry, _draw_context_menu):
            try:
                menu.remove(function)
            except Exception:
                pass
    _menu_hooks.clear()


def _draw_context_menu(self, context):
    space = context.space_data
    if space is None or getattr(space, "tree_type", "") != 'ShaderNodeTree' or not _is_jcycles(context):
        return
    self.layout.separator()
    self.layout.operator("node.jcycles_convert_to_fork")
    self.layout.operator("node.jcycles_convert_to_official")


# -----------------------------------------------------------------------------
# Cycles-only nodes are offered by the official menus only when the engine is 'CYCLES'.


def _patch_shader_menu_polls():
    try:
        from bl_ui import node_add_menu_shader as module
    except Exception:
        return
    original = getattr(module, "cycles_shader_nodes_poll", None)
    if original is None or "cycles" in _patched:
        return

    def cycles_shader_nodes_poll(context):
        return original(context) or getattr(context, "engine", None) == 'JCYCLES'

    module.cycles_shader_nodes_poll = cycles_shader_nodes_poll
    _patched["cycles"] = (module, original)


def _unpatch_shader_menu_polls():
    entry = _patched.pop("cycles", None)
    if entry is not None:
        module, original = entry
        module.cycles_shader_nodes_poll = original


# -----------------------------------------------------------------------------
# Convert operators


def _selected_shader_nodes(context):
    space = context.space_data
    tree = getattr(space, "edit_tree", None)
    if tree is None or tree.bl_idname != 'ShaderNodeTree':
        return None, []
    return tree, [node for node in tree.nodes if node.select]


class NODE_OT_jcycles_convert_to_fork(bpy.types.Operator):
    """Replace the selected official nodes with JCycles nodes that have this fork's extra inputs"""
    bl_idname = "node.jcycles_convert_to_fork"
    bl_label = "Convert to JCycles Nodes"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        tree, nodes = _selected_shader_nodes(context)
        return tree is not None and bool(nodes)

    def execute(self, context):
        specs = _specs()
        tree, nodes = _selected_shader_nodes(context)
        count = 0
        for node in nodes:
            spec = specs["nodes"].get(node.bl_idname)
            if spec is None or spec["kind"] != "extended":
                continue
            jcy_nodedefs.replace_node(tree, node, jcy_nodedefs.standin_idname(node.bl_idname), spec["properties"])
            count += 1
        self.report({'INFO'}, "Converted {:d} node(s)".format(count))
        return {'FINISHED'}


class NODE_OT_jcycles_convert_to_official(bpy.types.Operator):
    """Replace the selected JCycles nodes with the official nodes (the fork's extra inputs are lost)"""
    bl_idname = "node.jcycles_convert_to_official"
    bl_label = "Convert to Official Nodes"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        tree, nodes = _selected_shader_nodes(context)
        return tree is not None and any(jcy_nodedefs.native_idname_of(n.bl_idname) for n in nodes)

    def execute(self, context):
        specs = _specs()
        tree, nodes = _selected_shader_nodes(context)
        count = 0
        for node in nodes:
            native = jcy_nodedefs.native_idname_of(node.bl_idname)
            spec = specs["nodes"].get(native) if native else None
            if spec is None or spec["kind"] != "extended":
                continue
            jcy_nodedefs.replace_node(tree, node, native, spec["properties"])
            count += 1
        self.report({'INFO'}, "Converted {:d} node(s)".format(count))
        return {'FINISHED'}


_static_classes = (
    NODE_MT_jcycles_shader_nodes,
    NODE_OT_jcycles_convert_to_fork,
    NODE_OT_jcycles_convert_to_official,
)


def register():
    try:
        specs = jcy_runtime.node_specs()
    except Exception as ex:
        print("JCycles: node definitions are missing ({:s}); fork nodes are not available".format(str(ex)))
        specs = {"nodes": {}}
    jcy_nodedefs.set_specs(specs)
    _classes[:] = jcy_nodedefs.make_node_classes(specs, with_ui=True)
    for cls in _classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as ex:
            print("JCycles: cannot register node {:s}: {:s}".format(cls.bl_idname, str(ex)))
    for cls in _static_classes:
        bpy.utils.register_class(cls)
    _hook_add_menus()
    _patch_shader_menu_polls()


def unregister():
    _unpatch_shader_menu_polls()
    _unhook_add_menus()
    for cls in reversed(_static_classes):
        bpy.utils.unregister_class(cls)
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    _classes.clear()
