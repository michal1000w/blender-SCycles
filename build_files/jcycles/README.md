# JCycles

JCycles puts this repository's Cycles (blender-SCycles) into an **official, unmodified Blender**
as a second render engine called **JCycles**, next to the official Cycles. Pick it in
Render Properties > Render Engine.

## Building

| Platform      | 1. Build Blender     | 2. Make the add-on                       |
|---------------|----------------------|------------------------------------------|
| macOS, Linux  | `./compile.sh`       | `./compileCycles.sh` (add `--zip`)       |
| Windows       | `.\compile.ps1`      | `.\compileCycles.ps1` (add `-Zip`)       |

`compileCycles` writes `build/JCycles/` (and with `--zip` / `-Zip` also
`build/JCycles-<version>-<platform>.zip`). `compileCycles.sh --compile` runs `compile.sh` first.

`compileCycles` refuses to package a build that is older than the sources (it compares the Cycles
add-on files); rebuild, or pass `--allow-stale`.

The package contains the Blender build of the platform it was made on (`runtime/<platform>/`).
To support several platforms in one add-on, build on each and copy the `runtime/<platform>`
folders into one `JCycles` folder.

## Installing

Zip the `JCycles` folder (or use the zip from `--zip`). In an official Blender (5.0 or newer):
Edit > Preferences > Add-ons > (menu) Install from Disk, choose the zip, enable
"JCycles Render Engine". Preferences of the add-on show the bundled build and the render devices
(GPU selection works like in Cycles). It installs as a regular add-on (Add-ons list, folder
`scripts/addons/JCycles`), not as an extension.

Packages made before 2026-10-29 15:30 contained Blender's extension manifest template and were
installed as a broken extension "my_example_extension" (startup error "module loaded with no
associated file"). Remove it in Preferences > Get Extensions (installed, user_default > Uninstall)
or delete `extensions/user_default/my_example_extension` in the Blender user folder, then install
a new package.

## How it works

Official Blender cannot load this fork's Cycles into its own process: it does not export the
internal functions Cycles calls (Windows exports none), its data layout differs between versions,
and it has no code for the fork's shader nodes. So JCycles runs the fork's own Blender build in the
background as a render process:

* **Settings**: the add-on registers the fork's Cycles settings and panels (generated from
  `intern/cycles/blender/addon` at package time, renamed `cycles` -> `jcycles`). The render
  process copies them back to `cycles` before rendering.
* **Fork shader nodes**: Diffraction BSDF, Fast Volume, Volume IOR and the fork's extra inputs of
  Glass, Glossy, Metallic, Principled, Refraction and Principled Volume are Python nodes in the
  node editor (Add > JCycles), with the fork's sockets, ranges, settings and socket visibility.
  The render process replaces them with the real fork nodes. Inside they hold the closest official
  node, so EEVEE and Material Preview show an approximation. Node editor > right-click >
  "Convert to JCycles Nodes" turns official nodes into their fork versions (and back).
  The list lives in `fork_nodes.py`; the builder fails if a listed node or socket disappears.
* **Final renders** (F12, animations, command line `-f`/`-a`): the scene is saved to a temporary
  file (paths made absolute, unsaved painted images included, caches next to the .blend linked),
  the render process opens it, renders all view layers with Cycles and writes a multilayer EXR
  that is loaded into Blender's render result: all passes, AOVs, light groups, Cryptomatte with
  metadata, multi-view. Compositing, stamp and file output stay in the official Blender.
  An animation sends the scene once. With Persistent Data the render process stays alive between
  renders.
* **Viewport (Rendered shading)**: the render process renders progressively what the viewport
  shows (viewport camera or camera view, viewport visibility, modifiers and sampling settings,
  render pass display, viewport denoising). Moving the view re-renders at once; editing the scene
  sends a new snapshot after a short pause.
* **Baking**, **OSL** script nodes and custom OSL cameras (compiled by the fork's OSL),
  **Denoise Animation** and **Merge Images** also run in the render process.

## Tested (macOS arm64, Metal and CPU)

* Official Blender 5.0.1, 5.1.2 and 5.3 alpha as hosts.
* Final render of a scene with fork nodes, 2 view layers, light/data/Cryptomatte passes:
  bit-identical to this repository's Blender on CPU; on Metal within the run-to-run noise of the
  native build (max 1.5e-5).
* OSL script node render, bake (with margin): bit-identical. Animation: scene sent once.
* UI: viewport (free view and camera view), F12, all property tabs draw without errors.
* Render time: the same render plus 1-3 s for starting the render process and saving the scene.

Scripts in `tests/`. Implemented but not tested yet: Windows and Linux (compile.ps1,
compileCycles.ps1, the Linux path of compile.sh), Denoise Animation and Merge Images operators,
stereo/multi-view renders, vertex-color baking.

## Differences from Cycles inside the fork's Blender

* The F12 image appears when the render finishes; until then the window shows progress, samples,
  memory and time.
* Material/world thumbnails and Material Preview use EEVEE. Freestyle strokes are drawn by EEVEE.
* The viewport is emulated with final renders: Geometry Nodes "Is Viewport" is not reproduced,
  and every scene edit resends the whole scene (slow for very large scenes).
* Edits made while an animation renders are not picked up by later frames.
* Bake uses the scene's bake settings (as the Bake button does); bake operator arguments given
  from a script that differ from the scene settings are not passed on.
* A .blend saved by the fork's own Blender with fork nodes loses the fork-only node data when
  opened in an official Blender (official Blender drops data it does not know).
* Official Blender's node search may still hide Cycles-only nodes when JCycles is active; the
  Add menu shows them.
* Settings live in `scene.jcycles` etc.; they are not shared with the official Cycles.

## Files

* `build_jcycles.py`: the packager (called by compileCycles).
* `introspect.py`: runs inside the build to record its Cycles features and node definitions.
* `fork_nodes.py`: the fork's shader nodes JCycles must provide.
* `addon/`: add-on modules (`jcy_*`), the stand-in `_cycles` module, and `jcy_worker/worker_main.py`
  which runs inside the fork's Blender.
* `tests/`: end-to-end tests (host scene, reference render, EXR comparison, viewport, GUI, bake, OSL).
