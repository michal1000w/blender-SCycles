# Rebase onto Blender main, 2026-10-01

The fork (`polaris`, 70 commits on Blender main `d2d4db030a7` of 2026-08-27) was brought onto
Blender main `b3d3c1f6f40` of 2026-10-01: 867 upstream commits, 105 files changed on both sides,
31 with conflicts. This file records what had to change, what was verified and how to repeat the
checks after the next update.

## 1. Branches

| Branch | Content |
|---|---|
| `polaris` | upstream `b3d3c1f6f40` + the 70 fork commits + port and fix commits (linear) |
| `polaris-merge-upstream-20261001` | old `polaris` + one merge commit + the same fix commits; same tree |
| `backup/polaris-pre-rebase-20261001` | `polaris` before the rebase (`ea71a9d174a`) |

The replayed commits keep their authors, dates and messages. Each was replayed preferring the
fork's side of a conflict, so only the tip is a consistent, building state: the commit
"Port to Blender main" carries every conflict resolution. Intermediate commits are history, not
bisectable builds. Nothing was pushed.

## 2. What upstream changed under the fork

| Upstream change | Effect on the fork | Resolution |
|---|---|---|
| Gaussian splats (`PRIMITIVE_GSPLAT`, Embree user geometry, `KERNEL_FEATURE_GSPLATS` = bit 32) | Collided with the fork's Embree user geometry (pixel displacement, coherent Glass spheres) and with `KERNEL_FEATURE_BDPT` | One user-geometry callback dispatches by primitive type; the query's splat callback is kept in the context. Fork feature bits moved to 33–37 |
| Anisotropic Glass (Tangent, Anisotropy, Rotation) | The fork already had a Tangent socket on Glass for gratings | One Tangent socket. Gratings use the unrotated tangent as before, anisotropy the rotated one; the tangent stays linked when either needs it |
| Glass/refraction setups keep `alpha_y`; anisotropic transmission | Fork closures that embed microfacet lobes | Audited: every caller sets both roughness values, embedded carriers are isotropic |
| OSL closures invert the IOR at backfaces themselves | Fork node shaders pre-inverted | Node shaders pass upstream's values plus the fork's parameters; grating closures take the absolute IOR as before |
| Missing image/attribute fallback values (new node inputs, new `svm_image_texture` argument) | Pixel displacement evaluates image programs without an SVM stack | Programs read the constant fallback; the host rejects programs that link it |
| GPU state compression (half floats for ray time, `dD`, guiding and pass weights) | State templates conflicted with the fork's CPU-only guiding fields | Upstream types with the fork's feature guards |
| Concurrent state growth for all GPUs (`ConcurrentStatesParams`) | State size estimate signature; with the smaller compressed state a 16 GB Mac grew to 4× the baseline instead of 2× and became much slower (section 6) | Estimate takes volume stack size and device type; base chip GPUs (fewer than 16 cores) grow to 2× at most |
| Metal: deployment target 13, address helper and macOS 12 constants removed, BLAS build restructured | Device code conflicts | Followed upstream; pixel displacement bounding boxes re-applied on the new BLAS build; removed defines dropped from the precompiled libraries |
| `KernelObject` mesh/volume union | `normal_offset` uses in pixel displacement | `mesh_volume.normal_offset` |
| EEVEE material functions ported from GLSL to BSL | The fork's extra sockets on Glass, Glossy, Metallic, Principled, Refraction; Fast Volume shader | Signatures ported; Fast Volume ported to BSL; `MAX_PARAMETER` 43 |
| Shader node IDs 720–725 taken (lighting nodes) | `SH_NODE_VOLUME_FAST` 720, `SH_NODE_BSDF_DIFFRACTION` 721 | Renumbered to 790, 791. Nodes are stored by idname, existing files load unchanged |
| SVM: Boolean/Integer Math nodes, node reads moved out of feature checks | The fork's shared-node cases for Metal | Upstream layout with shared cases |

## 3. Bugs found and fixed on the way

These existed in the fork before the rebase; the upstream tests and cold caches exposed them.

1. **Light Path depths on Metal.** Separately compiled shading functions lost the difference
   between main and shadow path states, so shadow, shadow volume and dedicated light shaders read
   Ray/Diffuse/Glossy/Transmission/Transparent Depth of unrelated paths. Six upstream tests
   (`light_path_*_depth`, `volume_light_path`) now match the CPU.
2. **Render never starts while kernels prewarm.** The shader cache could purge the pipeline
   variant of an active device (there are more than three generic variants); the device then
   waited forever. Active variants are no longer purged.
3. **Diffraction BSDF crashed EEVEE and material preview.** The node had no GPU function. It now
   previews as a tinted mirror.
4. **Kernel unit tests of the fork** crashed (null kernel globals) and one had a tolerance below
   its table's rounding.

One regression came from upstream itself and is fixed for this class of machine: the state
growth described in section 6.

## 4. Verification

Machine: Apple M5, 16 GB, macOS 27.0. "Before" is the installed build of `ea71a9d174a`.

**Fork suites** (`cycles_metal_feature_scenes.py` 25 scenes, `cycles_cpu_parity_scenes.py`
39 scenes, Metal and CPU, 128 images per build):

* CPU: 21 of 25 feature images bit-identical. The others are `hair` (run-to-run noise, also
  between two runs of the old build), `diffraction` and BDPT scenes with last-bit accumulation
  differences that the old build shows between runs too.
* Metal is never bit-identical between builds. All means agree within 0.03 %. Differences above
  the old build's own run-to-run noise were traced:
  * spectral scenes (gratings, dispersion, tinted spectral glass), CPU and Metal: upstream
    changed the XYZ→Rec.709 constants by 1.6e-4 (`c1d93213041`). With the old constants every
    CPU image, SVM and OSL, is bit-identical to the old build;
  * motion blur and textures on Metal: half-float ray time and `dD` in the GPU state
    (`33b629d65b7`). With full floats these match the old build within noise;
  * subsurface on Metal: half-float albedo, same commit;
  * image pixel displacement on Metal: differs only while the specialized evaluator is still
    compiling (cold cache), the old build does the same.
* CPU versus Metal parity inside one build is unchanged for every scene.

**New tests** in `tests/python`:

* `cycles_fork_node_scenes.py`: 14 scenes for the nodes the fork adds or extends (gratings on
  Glass, Glossy, Refraction, Principled; polarizer; thin film; Fast Volume; Diffraction BSDF),
  SVM and OSL. CPU: bit-identical to the old build with the old colour constants.
* `eevee_fork_node_test.py`: creates each such node, checks sockets (one Tangent, no duplicates)
  and renders it in EEVEE.

**Upstream render regression tests** (`cycles_render_tests.py`, 39 test directories):

| Device | Passed | Failed | Failed in the old build too |
|---|---|---|---|
| CPU | 747 | 35 | all 35 (the old build fails 43) |
| Metal (software BVH) | 715 | the same 35 | — |

The 35 are deviations of the fork from upstream's references, identical before and after except
for the colour constants: spectral noise in tinted and thin glass, and pixel displacement
(on by default) on upstream's displacement scenes. Gaussian splat, point cloud and hair tests
pass on CPU and Metal.

**Unit tests**: `cycles_test` 390 of 390 pass (requires `-DWITH_GTESTS=ON`).

**Benchmarks**: see section 6.

## 5. Things to know

* **Stale macOS shader cache crashes any build.** The crash in
  `-[MTLBinaryEntry initWithData:]` under `addFunctionWithDescriptor` hit the old build as well.
  The cause was the system Metal function cache of Blender
  (`$(getconf DARWIN_USER_CACHE_DIR)org.blenderfoundation.blender/com.apple.metal`), which had
  grown past 1 GB. With a fresh cache the crash is gone. The old cache was moved aside, not
  deleted.
* **Pixel displacement is on by default** and applies to upstream scenes with true displacement:
  grid lines along quad edges, and `vector_displacement_object` loses most of its surface. Same
  in the old build. These scenes are also slow on Metal (up to minutes).
* Upstream features that have not been combined with fork features: Gaussian splats with
  BDPT/photons, anisotropic Glass with gratings or MNEE.
* Metal results differ slightly from the old build wherever the GPU state now stores half
  floats; that is upstream's trade-off.

## 6. Benchmarks

`tools/rebase_validation/benchmark.sh`: old and new build interleaved, one warm-up round and
three measured rounds, median per scene, sum over the suite. Feature suite: second render of the
scene in the same process, Metal 192×192 with 64 samples, CPU 160×160 with 32 samples. Parity
suite: complete render, Metal 160×160 with 64 samples, CPU 128×128 with 32 samples.

| Suite | Scenes | Before | After | Change |
|---|---|---|---|---|
| Feature, Metal | 25 | 105.10 s | 102.84 s | −2.2 % |
| Parity, Metal | 39 | 101.46 s | 103.05 s | +1.6 % |
| Feature, CPU | 25 | 44.43 s | 44.56 s | +0.3 % |
| Parity, CPU | 39 | 35.76 s | 35.80 s | +0.1 % |

No scene changed by more than 8 %; the old build's own spread between rounds is up to 10 %.
Pixel displacement on upstream's displacement test scenes renders at the same speed in both
builds (for example `true_displacement` 77 s and 74 s on Metal).

**State growth.** Straight after the merge Metal was much slower: upstream's new sizing gave
16.7M integrator states on this machine where the old build used 8.4M.

| Metal, second render | 8.4M states | 16.7M states |
|---|---|---|
| 256×256, 64 samples: basic / bdpt / guiding | 0.32 s / 0.86 s / 1.04 s | 0.68 s / 2.83 s / 9.95 s |
| 1280×1280, 96 samples: basic / bdpt / guiding | 5.82 s / 19.9 s / 26.3 s | 5.82 s / 20.0–22.3 s / 29.5–314 s |

More states never helped on the 10-core GPU, so base chip GPUs are limited to 2× the baseline;
GPUs with 16 or more cores keep upstream's 4×. `CYCLES_CONCURRENT_STATES_MAX` overrides it.

## 7. Repeating the checks

```
# Images and timings of one build (Metal and CPU), about 5 minutes
tools/rebase_validation/run_suites.sh <Blender.app> <out>
# Old against new build, or CPU against Metal of one build (same directory twice)
tools/rebase_validation/compare.sh <Blender.app> <out old> <out new>
# Render time, interleaved, warm-up plus three rounds
tools/rebase_validation/benchmark.sh <old Blender.app> <new Blender.app> <out>
tools/rebase_validation/benchmark_report.py <out> --scenes
# Upstream reference tests; fetch tests/files/render from upstream's LFS first
tools/rebase_validation/run_upstream_tests.sh <Blender.app> CPU|METAL|METAL-RT <out>
```

Keep a copy of the old build before installing the new one (`cp -Rc install/Blender.app ...`).
