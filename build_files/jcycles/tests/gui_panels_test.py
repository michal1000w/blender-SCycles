# Runs in an official Blender WITH its interface: shows every Properties tab with JCycles active
# so all generated panels draw; errors are printed by the add-on as "JCycles: <Panel>...".
#   blender <scene.blend> --python gui_panels_test.py -- <out_dir>
import os
import sys

import addon_utils
import bpy

out_dir = sys.argv[sys.argv.index("--") + 1]
addon_utils.enable("JCycles", default_set=True)
tabs = ['RENDER', 'OUTPUT', 'VIEW_LAYER', 'SCENE', 'WORLD', 'OBJECT', 'MATERIAL', 'DATA']
state = {"i": 0, "step": 0}


def properties_area():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'PROPERTIES':
                return window, area
    return None, None


def tick():
    window, area = properties_area()
    scene = bpy.context.scene
    scene.render.engine = 'JCYCLES'
    if state["i"] >= len(tabs) * 2:
        bpy.ops.wm.quit_blender()
        return None
    index = state["i"] // 2
    tab = tabs[index]
    objects = {'MATERIAL': "Sphere", 'DATA': "Area", 'OBJECT': "Cube"}
    if state["i"] % 2 == 0:
        name = objects.get(tab)
        if name:
            bpy.context.view_layer.objects.active = bpy.data.objects[name]
        try:
            area.spaces.active.context = tab
        except Exception as ex:
            print("PANELTEST cannot show", tab, ex)
        area.tag_redraw()
    else:
        print("PANELTEST drew", tab, flush=True)
        if tab in {'RENDER', 'MATERIAL'}:
            region = next(r for r in area.regions if r.type == 'WINDOW')
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.screen.screenshot_area(filepath=os.path.join(out_dir, "panel_{:s}.png".format(tab.lower())))
    state["i"] += 1
    return 0.7


bpy.app.timers.register(tick, first_interval=2.0)
