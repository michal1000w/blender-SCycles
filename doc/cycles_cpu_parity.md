# Cycles CPU ↔ Metal feature parity

Scope: the features added on top of upstream Blender (`d2d4db030a7`; 73 commits, `68cc50adced` …
`62d4f85972b`) that ran only on the Metal GPU, brought to Cycles CPU with matching results.
Nothing was committed or pushed; all changes are in the working tree.

## 1. Feature inventory

| # | Feature | Metal | CPU before | CPU now |
|---|---|---|---|---|
| 1 | **Bidirectional path tracing**: light-vertex cache with reservoir selection, recursive VCM-style MIS, light-tracing sensor splats, light-tree selection PDFs, every emitter type (area, mesh, point, sphere, spot, sun, world), spectral/dispersive paths, polarization, coherent-path ownership, MNEE sensor manifolds, medium vertices | ✓ | silently fell back to path tracing ("Metal GPU only") | ✓ |
| 2 | **Photon mapping**: progressive caustic map, surface and volume estimators, spectral wavelength kernel, camera oversampling, map update batches, light-group passes, motion time bins, caster-targeted emission | ✓ | disabled | ✓ |
| 3 | **Pixel-level displacement**: per-hit displaced-surface solve, micromesh cache and cache BVH, image evaluator fast paths, displaced normals, SSS/AO/bevel, shadow shared-edge rule, motion blur, exact-sample bake path | ✓ | disabled (mesh displacement used) | ✓ |
| 4 | Path guiding | Metal guiding field | OpenPGL | OpenPGL, now also with BDPT (see §5) |
| 5 | Diffraction BSDFs, Glass polarizer, coherent interference, spectral transmission/dispersion, fast volume, MNEE work | ✓ | ✓ | ✓ (verified unchanged) |

## 2. Why the features were Metal only

* Kernel code was generic but wrapped in `#ifdef __KERNEL_METAL__` (~130 sites) and read its
  buffers through `kernel_integrator_state`, which existed only on GPUs.
* Metal kernels are members of one context class, so they could call functions defined later;
  the CPU kernel needs declarations in dependency order.
* Light-cache passes were scheduled only by `PathTraceWorkGPU`; `Integrator`/`GeometryManager`
  accepted only Metal devices.
* CPU paths have two shadow slots (direct light, AO); BDPT camera connections need a third.
  CPU film writes are not atomic, but light-tracing splats land on arbitrary pixels.
* Pixel displacement hooked only the BVH2 triangle intersector; the CPU uses Embree.

## 3. Implementation

**Kernel (shared with Metal, Metal behaviour unchanged)**
* `kernel/features.h`: `__BDPT__`, `__PHOTON_MAPPING__`, `__PIXEL_DISPLACEMENT__`,
  `__PIXEL_DISPLACEMENT_INTERSECT__/_SHADE__` for Metal and CPU. BDPT/photon/pixel-displacement
  guards use them; Metal guiding and visible-function plumbing stay Metal only.
* `kernel/device/cpu/globals.h`: `KernelTransportStateCPU` mirrors the photon/BDPT members of
  `IntegratorStateGPU`; `kernel_integrator_state` and `pixel_displacement_rays` map to it on CPU.
* Functions that read `kernel_data` without a `KernelGlobals` argument (legal only inside the
  Metal class) now take `kg`; CPU forward declarations for the cross-header cycles
  (bidirectional ↔ volume shading, intersection ↔ displacement solver).
* `VolumeShaderCoefficients`/`volume_shader_sample()` moved to `volume_shader.h`.
* `IntegratorStateCPU::bdpt` shadow slot + `integrator_bdpt_shadow_path_init()`; the megakernel
  runs it and flags rays of intersection kernels for pixel displacement (the Metal
  `pixel_displacement_rays` specialization).
* Deterministic CPU photon map: one slot per emitted path, hash chains linked in path order.
* CPU kernel entry points: photon emit / map build, BDPT light generate / cache order / sensor
  connect (film accumulation of splats serialized per pixel stripe, tracing concurrent).
* `ccl_device_inline_transport`: transport helpers are real calls on CPU (no inlining into the hot
  shading functions); GPU inlining unchanged. GPU-only shading replay stages compile away on CPU.
* Embree **user geometry for pixel-displaced meshes** (`bvh/embree.cpp`, `kernel/device/cpu/bvh.h`)
  with the BVH2 bounds (`Mesh::grow_pixel_displacement_bounds`, shared with the BVH2 builds) and
  callbacks resolving hits exactly like `triangle_intersect()`/`motion_triangle_intersect()`.
  Other geometry stays on Embree's native intersectors.

**Host**
* `PathTraceWorkCPU`: the GPU schedule (map update batches, progressive radius, photon camera
  oversampling, per-update light-path budget scaled by buffer pixels, MIS normalization).
* `Integrator`/`GeometryManager`: CPU, Metal and all-CPU/Metal multi-devices; supported layouts
  include Embree and Metal+Embree.
* OSL services compile the displaced-surface intersector too (OSL `trace()`/`getmessage()`).
* Add-on: panels/descriptions say "CPU and Metal GPU"; other GPU backends get an info box.
* CPU + OpenPGL guiding with BDPT: OpenPGL initializes its per-vertex distribution
  stochastically, so a light subpath cannot re-evaluate the camera sampling density that the
  recursive MIS needs. Vertices inside the BDPT recursion therefore sample unguided (training
  continues; vertices outside the recursion stay guided). Result is unbiased (§5).

## 4. Bugs found and fixed along the way (pre-existing, also on Metal unless noted)

1. **Baking with BDPT enabled produced wrong bakes on Metal** (−7% in the test): light-tracing
   splats projected through the camera into a texel buffer. BDPT is now off while baking.
2. **Switching displacement method in a persistent session/viewport double-displaced meshes**
   (mesh displacement moves vertices in place; pixel displacement then displaced them again).
   Such meshes are re-exported from Blender when pixel-displacement settings change.
3. `photon_state_init()` left `volume_bounds_bounce`, `optical_depth`, … uninitialized; on CPU a
   reused state stopped light paths entering volumes (on Metal the values were inherited).
4. Embree user-geometry hits (also the existing coherent Glass spheres) did not run the query
   filter (self-intersection, shadow recording) in the closest-hit path, and misidentified the
   object in SSS local queries on per-object acceleration structures.

## 5. Validation (all on this machine, Apple M5)

* **Feature suite** (`tests/python/cycles_metal_feature_scenes.py`, 25 scenes) and new
  **parity suite** (`tests/python/cycles_cpu_parity_scenes.py`, 39 scenes: every emitter type,
  light tree on/off, adaptive sampling, tiles, update-per-sample, passes and light groups,
  motion blur, dispersion, volumes, OSL, polarizer, guiding, instanced/low-res/clamped/image
  pixel displacement): CPU vs Metal relative RMSE ≤ 0.01 with equal means for BDPT, photons and
  pixel displacement; BDPT/photon scenes are often identical (same RNG streams).
* Known, explained differences:
  * `displacement_image` +1.2%: GPU texture units filter with fixed-point weights; a smooth
    height map agrees to rrmse 0.001.
  * `bdpt_guiding` −1.5% with default clamping: clamping acts per sample and guiding changes the
    sample distribution. With clamping/Filter Glossy off: PT 0.7255, CPU BDPT 0.7254,
    Metal BDPT 0.7252, CPU guided BDPT 0.7255, Metal guided BDPT 0.7256.
  * `volume` −1.7% (pre-existing): the test cube's bottom face lies in the floor plane; with the
    cube lifted 2 cm CPU and Metal are identical.
  * `hair`, `guiding`: pre-existing run-to-run variation (particle hair) / different guiding
    algorithms.
* **No regressions**: 20 upstream-feature regression scenes and all 276 authored fixtures in
  `build/tests` render bit-identically on CPU vs the original build; Metal output unchanged
  (feature suite identical to the original Metal build).
* Robustness: persistent-data sessions with layout changes, time-limited renders, CPU+Metal
  multi-device, bakes; the compiled `cycles_diffraction_*` test programs pass.

## 6. Benchmarks

See the table in the final summary of the session (256×256, fixed seeds, second render,
median of interleaved runs).
