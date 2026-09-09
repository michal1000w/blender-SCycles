#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""A narrow-aperture variant of the shared guiding room, with unchanged materials and light.

Run in Blender using the same arguments as cycles_metal_guiding_scene.py, with
--scene aperture. The baffle reaches 3.85 m into the 4 m room, so illumination must
find a 0.15 m opening. The existing room generator remains unchanged, preserving
its reference-image provenance. This fixture composes its geometry through the
shared factory and retains the actual 'aperture' scene name in saved metrics.
"""

import importlib.util
import pathlib
import sys


def main():
    base_path = pathlib.Path(__file__).with_name('cycles_metal_guiding_scene.py')
    spec = importlib.util.spec_from_file_location('cycles_guiding_room', base_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    args = sys.argv[sys.argv.index('--') + 1:]
    if '--scene' not in args or args[args.index('--scene') + 1] != 'aperture':
        raise ValueError('This fixture requires --scene aperture')
    args[args.index('--scene') + 1] = 'indirect'
    sys.argv = [str(base_path), '--', *args]
    original_box = generator.box
    original_build = generator.build_scene

    def box(name, location, scale, mat):
        if name == 'Hidden-light baffle':
            location = (location[0], location[1], 1.925)
            scale = (scale[0], scale[1], 3.85)
        return original_box(name, location, scale, mat)

    def build_scene(settings):
        settings.scene = 'aperture'
        return original_build(settings)

    generator.box = box
    generator.build_scene = build_scene
    generator.main()


if __name__ == '__main__':
    main()
