<!--
Keep this document short & concise,
linking to external resources instead of including content in-line.
See 'release/text/readme.html' for the end user read-me.
-->

Michal1000w's branch information
=======

1. This branch is free and open source.
2. Most hard work is done by me by hand.
3. Some comments in the code, merging with main blender branch, and some stupid bugs are sometimes done by AI as I don't have time or will for that. But all AI committed code is marked with the AI used so you know when something is done by that tool. I've done few first merges myself but I don't like it and probably AI would do it better anyway as I'm not a master of blender's code. So sorry for that
4. Later I'll do a list of new features but for now I don't have time for that

Building
--------

| Platform | Script | GPU backends |
| --- | --- | --- |
| macOS (Apple Silicon) | `./compile.sh` | Metal |
| Windows x64 | `.\compileWindows.ps1` | CUDA, OptiX (HIP/oneAPI with their SDKs) |
| Linux x64/arm64 | `./compileLinux.sh` | CUDA, OptiX (HIP/oneAPI with their SDKs) |

Each script fetches Blender's precompiled libraries on first use, configures the release feature
set, builds, installs to `./install` and checks the installation (`--help` / `Get-Help` for the
options). On Windows and Linux:

* **Requirements**: Visual Studio 2022 with the C++ workload (Windows) or GCC/Clang (Linux),
  CMake, Git LFS, and for GPU rendering the CUDA toolkit (12 or 13 recommended; 11.8 works for
  GPUs up to sm_90) and the OptiX SDK 8.0 or newer. They are found automatically
  (`CUDA_PATH`, `OPTIX_ROOT_DIR`, the default install locations).
* **CUDA architectures**: by default only those of the GPUs in the machine are compiled, which
  keeps the build short. `-CudaArch release` / `--cuda-arch release` builds Blender's full
  release list for a redistributable build.
* **Memory**: a Cycles kernel (a CPU kernel file, a CUDA or an OptiX module) needs up to ~16 GB
  of memory to compile. The build runs one of them at a time per 32 GB of memory, next to the
  regular parallel jobs. `-KernelJobs N` / `--kernel-jobs N` sets the number, and
  `-Jobs N` / `--jobs N` the number of all jobs.
* **Tests**: `-Test` / `--test` renders the demo scenes of every custom feature on the CPU and on
  each CUDA and OptiX device and checks that the GPUs match the CPU
  (`tests/python/jcycles_feature_demo.py`, report in `build/jcycles_feature_demo/report.html`).

CUDA and OptiX keep this fork's features (bidirectional path tracing, vertex merging, photon
mapping, GPU path guiding, pixel-level displacement, diffraction closures, coherent light,
polarization, nested dielectrics and volumes with an index of refraction) in a separate kernel
module that is loaded only for scenes that use them. Ordinary scenes run the regular kernels,
which match standard Cycles in registers and memory use. On OptiX, scenes with pixel-level
displacement render with the CUDA kernels. OSL shading on OptiX uses the standard Cycles kernels
and does not support these features; the CPU supports them with OSL.

Custom (OSL) cameras work on CUDA and OptiX together with all of these features, as on Metal: the
compiled camera shader is translated, compiled at runtime with NVRTC (which ships with the build,
next to `blender.exe` or in `lib/`) and linked into the camera ray kernel. The first render with a
new camera shader takes a few seconds, later ones load it from the kernel cache.
`tests/python/cycles_osl_camera_scenes.py --device CUDA|OPTIX` compares it with the CPU.

Blender
=======

Blender is the free and open source 3D creation suite.
It supports the entirety of the 3D pipeline—modeling, rigging, animation, simulation, rendering, compositing,
motion tracking and video editing.

![Blender screenshot](https://code.blender.org/wp-content/uploads/2018/12/springrg.jpg "Blender screenshot")

Project Pages
-------------

- [Main Website](https://www.blender.org)
- [Reference Manual](https://docs.blender.org/manual/en/latest/index.html)
- [User Community](https://www.blender.org/community/)

Development
-----------

- [Build Instructions](https://developer.blender.org/docs/handbook/building_blender/)
- [Code Review & Bug Tracker](https://projects.blender.org)
- [Developer Forum](https://devtalk.blender.org)
- [Developer Documentation](https://developer.blender.org/docs/)


License
-------

Blender as a whole is licensed under the GNU General Public License, Version 3.
Individual files may have a different but compatible license.

See [blender.org/about/license](https://www.blender.org/about/license) for details.
