# Pixel displacement performance investigation

## Delivered result (stage 50)

The user requested finishing the reported JAR 10x improvement with good quality and avoiding
unnecessary additional work. Cached 2048 is explicitly no longer a performance target.
The tested build is `install/Blender-Displacement.app`; the existing `install/Blender.app` and
user .blend are unchanged.

On the supplied JAR scene, Metal RT on Apple M2 Max, 750x500, 16 samples, seed 7, persistent data,
seven sequential candidate renders (first excluded), the final results are:

| Mode | Original warm median | Final warm median | Speedup | Original peak process RSS | Final peak process RSS |
| --- | --- | --- | --- | --- | --- |
| Unlimited | 24.454293 s | 2.266591 s | 10.79x | 1,059,930,112 B | 1,080,197,120 B (+1.91%) |
| 16K | 24.165796 s | 2.238174 s | 10.80x | 1,071,874,048 B | 1,075,888,128 B (+0.37%) |

First candidate renders are 3.277353 / 3.290613 s after shader compilation had been exercised.
These are warm render comparisons; they do not claim zero initial Metal compilation cost.
Both final EXRs differ from their original references by at most 6.42240e-6 over 1,500,000
channels; RMSE is 9.93595e-9 unlimited / 9.59424e-9 at 16K.

The seven-case mesh/UV/generated/missing-UV edit regression also passes at the original-evaluator
level: four cases exact, normals/generated maximum 3.72529e-9, and UV-switch maximum 1.24495e-6
(RMSE 9.35232e-9). The fast path is restricted to static triangles with identical native packed
normals and native float2 coordinate attributes. Other inputs use the original evaluator for
quality; the 10x claim applies to this measured JAR workload, not every possible shader/geometry.
A scene-wide certification flag removes unused fallback code for fully eligible scenes and
resets conservatively on geometry/cache or shader updates.

Native image resolution is unchanged. Metadata remains inside the 16 MiB fallback-cache budget
(128 descriptor bytes for JAR). Resolution clamping is optional, defaults off, and accepts up to
16384 when enabled. Historical trials below include rejected experiments; they are not additional
claims about the delivered implementation. The unused image-bound hierarchy prototype was not
connected to rendering and contributes no claimed speedup.

Evidence: `/tmp/resolution-speed/verified/quality50-large7-{unlimited,16384}.json`, corresponding
EXRs, and `/tmp/resolution-speed/regression-edits/input-cache-original-comparisons50.json`.

## Research

- [TFDM, SIGGRAPH Asia 2021](https://perso.telecom-paristech.fr/boubek/papers/TFDM/): separates
  displacement acceleration from the base mesh so storage need not grow with tessellated geometry.
- [RMIP, SIGGRAPH Asia 2023](https://iliyan.com/publications/RMIP/): inverse mapping and rectangular
  conservative displacement bounds accelerate ray traversal without storing a dense micromesh.
  Its reported 11x example is the paper's result, not a result of this implementation.
- [Ray-Tracing Procedural Displacement Shaders, GI 1998](https://graphicsinterface.org/proceedings/gi1998/gi1998-2/):
  interval/affine bounds can conservatively enclose procedural displacement over a parameter range.
  Finite samples alone cannot prove an arbitrary procedural shader's extrema.
- [GPU Gems 2, displacement distance functions](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-8-pixel-displacement-mapping-distance-functions):
  adaptive empty-space stepping is preferable to uniformly expensive marching, but a volumetric
  distance field would have an unfavorable memory cost for this task.
- [NVIDIA Micro-Mesh](https://developer.nvidia.com/rtx/ray-tracing/micro-mesh): hardware-specific
  compressed microgeometry is not a portable solution for this project's Metal backend.

## Baseline and diagnostic measurements

The source worktree was clean at `dc254206de2` when this investigation started. The original installed
app was cloned to `/tmp/resolution-speed/Baseline.app` before any edits. Do not overwrite that copy.
The .blend file itself has not been saved or modified.

Initial diagnostic configuration: JAR, 150x100 (5% of 3000x2000), 4 samples, fixed seed, no denoising
or adaptive sampling, persistent data, three sequential renders. First renders include compilation
and preparation; they are not included in warm speedup ratios. Measurements below are preliminary.

| Build / mode | Warm render seconds | Peak process RSS bytes | Result |
| --- | --- | --- | --- |
| Original unlimited | 0.8391, 0.8285 | 1,077,231,616 | Reference |
| Original 16K | 0.9093, 0.8536 | 1,056,718,848 | Falls back to direct evaluation at cache budget |
| Original 2K, factory startup | 0.0519, 0.1158 | 1,318,141,952 | More repetitions needed for stable comparison |
| Stage 1: shared patch rows, deferred derivatives, tiny ShaderData | 0.4818, 0.5231 | 1,183,809,536 | About 1.7x; retained |
| Stage 2: compiler-verified 32-float displacement SVM stack | 0.4973, 0.4506 | 1,289,273,344 | Not convincing; removed |
| Stage 3: conservative whole-material height pruning | 0.5727, 0.5215 | 1,276,133,376 | Preserves image, no measured speed benefit; removed |
| Stage 4: memoize existing grazing fallback samples | 0.4637, 0.4366 | 1,187,708,928 | Failed image comparison; removed |

A synthetic 256x256 image-texture baseline completed one warm render in 10.15 seconds, then failed
with a Metal GPU recovery error. This failed run is not a completed benchmark or an optimization
result. Smaller diagnostic renders were used to establish a stable baseline.

Stage 1 vs original unlimited EXR: maximum absolute channel difference `2.3841858e-7`, RMSE
`2.8716853e-9`. Stage 3: maximum `2.3841858e-7`, RMSE `2.8244334e-9`.
These comparisons cover the diagnostic image only, not the full acceptance matrix.

## Larger repeated JAR check (stage 6)

300x200, 8 samples, 7 sequential renders, first render excluded. No GPU tests overlapped.

| Mode | Original warm median | Candidate warm median | Speedup | Original peak RSS | Candidate peak RSS |
| --- | --- | --- | --- | --- | --- |
| unlimited | 2.7365 s | 1.4634 s | 1.87x | 1,057,931,264 B | 1,204,551,680 B |
| 16384 | 2.7939 s | 1.4852 s | 1.88x | 1,061,912,576 B | 1,061,158,912 B |

Both EXRs agree with the original within floating-point roundoff: maximum channel difference
`4.7683716e-7`; unlimited RMSE `6.7585435e-9`, 16K RMSE `6.6537753e-9`.
The unlimited candidate first render included more compilation and peaked approximately 147 MB
above the reference process; the subsequent 16K process RSS was essentially unchanged.
This is approximately 1.9x, **not** the requested 10x. Full-resolution and procedural acceptance
remain open. The visible JAR render uses the image-textured Material.002; procedural materials
stored in the file are not sufficient evidence of procedural rendering performance.

Scene SHA-256: `87585175d268d2efc537bd995166f5e21e808aab382fb61bbdc9b7ae5dc9b98c`.

## Current changes

- Evaluate each vertex once in the existing grazing fallback, keeping one row of positions.
- Evaluate Newton derivatives only when the position residual requires another iteration. Keep the
  existing full normal calculation for final shading.
- Use existing `ShaderDataTinyStorage` for displacement, which does not allocate closures.
- Require specialized Metal displacement kernels when selecting render pipelines. A compilation
  failure must report an error rather than silently selecting generic kernels that omit displacement.

Stage 4 was rejected: maximum EXR difference `0.8674231`, RMSE `0.0130610`. Its speed is not a
valid optimization result. The rejected diff and material-bound sources were archived under
`/tmp/resolution-speed/` before removal.

Stage 5 (direct MetalRT callbacks) was rejected. Its apparent 0.02-second warm render was invalid:
Metal pipeline compilation failed with `XPC_ERROR_CONNECTION_INTERRUPTED`, and the renderer silently
fell back to generic kernels that omit displacement. EXR maximum difference was `30.4543972`, RMSE
`0.6793292`. The traversal experiment was removed; the required-kernel selection fix is retained.

## Reproduction and artifacts

`tools/pixel_displacement_performance.py` provides a reusable loaded-scene benchmark with warm
medians, first-render timing, EXR output, peak process RSS, and build/device/settings metadata.
Use `--factory-startup` before loading the scene to avoid unrelated user add-on callbacks. Run GPU
benchmarks sequentially; do not claim GPU timing from overlapping renders or failed renders.

Intermediate installs and JSON/EXR/log files are under `/tmp/resolution-speed/`, with directories
`baseline`, `baseline-clean`, `stage1-results`, `stage2-results`, and `stage3-results`. Stage 4 has a
separate app installation. The user's installed app remains unchanged by these experiments.

## Remaining work

The 10x target and cached-2K-equivalent speed are not achieved. Measure the retained changes on larger repeated renders, and verify the kernel-failure handling.
A successful acceleration architecture beyond these local optimizations is still needed.
Validate unlimited and high-clamp modes on larger JAR renders and diverse
image/procedural/grazing/smooth/animated/material-edit cases before calling this goal complete.

## Profiling attempt

`xctrace` Metal System Trace was invoked for a bounded 45-second capture. The launched Blender
process remained suspended before startup (empty stdout and no benchmark report), so the trace
contains no valid JAR rendering workload. Do not use it for performance conclusions. The suspended
task-owned Blender process was terminated; the stalled Instruments process was then terminated.
Use Cycles' per-kernel Metal profiling or attach after Blender startup for the next attempt.

## Cached-mode regression and remaining bottleneck

The repeated cached-2048 regression check passed: original median `0.0633312 s`, candidate
`0.0635254 s`, EXR maximum difference `4.7683716e-7`, RMSE `6.8209727e-9`.
Candidate unlimited remains about 23 times slower than cached 2048 on this diagnostic workload.

Cycles built-in Metal profiling completed successfully using `CYCLES_METAL_PROFILING=1`.
For three JAR diagnostic renders, closest intersections consumed 59.01% of measured GPU kernel
time, shadow intersections 39.68%, and surface shading only 1.02%. The log is
`/tmp/resolution-speed/profile-kernels.log`. These instrumented timings are for locating the
bottleneck, not replacing the uninstrumented comparison above.

A bounded next experiment is an exact displacement-only evaluator for compiler-certified simple
node programs, calling the existing image/noise/ramp math at full resolution and retaining the
full SVM for unsupported graphs. Merely reducing SVM stack size was already unsuccessful; any
new evaluator must justify itself with measurements. Conservative native-image min/max traversal
remains a larger architectural alternative. No shortcut may substitute sampled low-resolution
heights for arbitrary shader values without proving its bounds.

The mandatory-kernel selection change built and passed successful render paths. Explicit fault
injection for its failure branch, procedural/smooth/animated/vector displacement coverage, and
larger/full-resolution validation remain to be done. Do not mark the performance goal complete.

## Compact evaluator experiment (stage 9, pending)

The current worktree additionally contains an experimental compact displacement interpreter:

- Shared `displacement_nodes_template.h` defines both the host whitelist and GPU cases, reusing
  existing node implementations. It supports constants, geometry, attributes, conversions, images,
  noise, ramps, mappings, and scalar displacement, including applicable derivative variants.
- The SVM compiler records support for every emitted opcode and the displacement program's peak
  stack allocation. Only fully supported programs needing at most 32 floats are certified.
- `KernelShader` reuses padding for per-shader certification. KernelIntegrator padding records
  whether the scene needs compact, full, or both evaluators, allowing Metal function constants to
  remove unused interpreter branches. Unsupported graphs retain the full SVM.
- No displacement samples, texture resolutions, root-search steps, or output-normal calculations
  are removed by this experiment.

Stage 7's initial JAR probe remained ineligible because it required NODE_GEOMETRY. Its preliminary
GPU run was terminated without measurements; do not count it as an optimization result. A CPU
compiler probe after adding geometry confirmed Material.002 is certified. Stage 8 was an
intermediate build, not a timed GPU result. Stage 9 adds the scene-wide evaluator selector.

Stage 9 source/C++ build and installation completed. The current GPU run is
`/tmp/resolution-speed/verified-compact9-unlimited.log`, process 31934, exec session 74076.
A one-second process sample at `/tmp/resolution-speed/compact9-wait-sample.txt` confirmed two Metal
pipeline compilation requests waiting for their asynchronous callbacks. No render has completed
at the time of this entry. Resume this exact handle/process; do not launch an overlapping GPU run
or assume completion from elapsed time. No stage 9 speed or image-quality result is yet established.

Prepared procedural variant: `/tmp/resolution-speed/jar-procedural.py` assigns the existing
Material.001 to SSand in memory only. Run it before the benchmark script in both baseline and
candidate to validate the noise/ramp graph independently of the image-textured original scene.

Stage 9 was terminated before a render completed. Sampling confirmed the app was waiting on
asynchronous Metal compilation, and its new compiler service was using 99.6% CPU and about
8.3 GiB RSS. Inspection found that compact switch cases omitted the existing SVM_CASE
function-constant checks, retaining unused noise and other node functions. Stage 10 corrects
this by using the same SVM_CASE macro as the original interpreter. This is a new source revision,
not a restart based only on an observation timeout. Stage 9 provides no speed/fidelity result.

## Stage 10 measured results

With unused-node pruning restored, the compact JAR run completed: 300x200, 8 samples,
7 renders, warm median `1.3449005 s`, first render `28.6501 s`, process peak RSS
`1,200,553,984 B`. This is approximately 2.03x the original direct evaluator, but only
about 9% faster than stage 6. EXR maximum difference `4.7683716e-7`, RMSE `6.7769244e-9`.

The procedural variant (150x100, 4 samples, 3 renders) also passed comparison:
original median `1.4174472 s`, compact median `0.6732411 s` (2.11x). Peak process RSS
was `775,110,656 B` original and `784,957,440 B` candidate. Maximum EXR difference
`2.3841858e-7`, RMSE `1.0215099e-8`. Logs and EXRs are in `/tmp/resolution-speed/procedural/`.
These are limited diagnostic results, not full acceptance.

## Revised fallback memoization (stage 11)

Inspection of scene update order confirmed geometry/cache baking precedes integrator upload.
The rejected stage 4 bake applied integrator scale and distance clamp too early. It also left
P11 vertex evaluations uncached. Stage 11 stores raw object-space displacement from the full
shader evaluator and applies the current scene settings in the ray query on the GPU. All fixed
fallback-grid vertices now use memoization. The primary solver still evaluates arbitrary shader
coordinates directly. The 16 MiB budget and eligibility restrictions remain; missing grids,
unsupported materials, motion, and shared meshes fall back to direct evaluation.

Stage 11 C++/offline Metal build passed. Runtime performance, EXR equality, and invalidation are
pending. Source includes the compact evaluator from stage 10. Do not assume stage 11 is correct
until its image comparison passes. The original installed application and .blend remain unchanged.

## Stage 11 verified diagnostic results

Raw fallback memoization now matches the reference:

| Mode / scene | Original warm median | Stage 11 warm median | Speedup | EXR maximum | EXR RMSE |
| --- | --- | --- | --- | --- | --- |
| JAR unlimited, 300x200, 8 samples, 7 renders | 2.73654 s | 0.85385 s | 3.20x | 4.76837e-7 | 6.98943e-9 |
| JAR 16K, same settings | 2.79387 s | 0.83940 s | 3.33x | 4.76837e-7 | 6.88314e-9 |
| Procedural variant, 150x100, 4 samples, 3 renders | 1.41745 s | 0.44221 s | 3.21x | 4.76837e-7 | 1.00235e-8 |

The JAR fallback cache contains 17,792 bytes. Stage 11 process RSS was 1,197,850,624 bytes
in the first unlimited process and 1,054,392,320 bytes in the subsequent 16K process. Procedural
RSS was 763,527,168 bytes. These are process peaks, not a complete system-wide GPU memory audit.
Build activity overlapped part of the procedural diagnostic; repeat final performance tests with
no build activity. Quality comparisons are independent of this timing limitation.

## Conservative magnitude pruning (stage 13, pending)

The current worktree adds compiler-derived scalar displacement magnitude bounds. The previous
stage 3 analysis rejected the geometry Normal link inserted by graph finalization, so the tested
material had no usable bound. The revised analysis uses a symmetric magnitude bound, justified
by object-space displacement's normalization of its normal input; it does not infer a direction.
Integer single-tile linear/nearest images, ramps, constants and supported conversions can provide
bounds. Unknown, HDR, cubic, mixed-UDIM and unsupported graphs retain the user distance bound.

The ray solver preserves its original sampling lattice, retaining bracketing samples around the
bounded interval. Only provably unreachable outer samples are skipped. Transform operator-norm
bounds are precomputed on the CPU and stored in existing KernelObject tail padding (static size
assertion remains 256 bytes). The runtime bound includes forward/inverse transform amplification
and a margin for precision loss in (P + D) - P, so large coordinates and ill-conditioned transforms
fall back to the user bound. KernelShader uses its remaining padding for one magnitude scalar.

Stage 12 was an intermediate host/kernel build without runtime measurements. Stage 13 adds the
transform/roundoff safeguards. Runtime performance and image comparisons remain pending.

## Stages 14–15: measured bounds and edit checks

Stage 13 completed at 0.837294 s warm median, but the material bound remained unknown.
`ImageHandle::num_tiles()` is zero for an ordinary image, not one. Stage 14 fixes that
eligibility check. The actual displacement PNG metadata is USHORT, one channel; the shader
bound is 0.25001. No assumption about floating-point images or PNG filenames is needed.
Stage 14 unlimited/16K medians were 0.766571/0.767203 s with process peaks
1,065,566,208/1,060,110,336 bytes. EXR maxima were 4.76837e-7/9.53674e-7 and RMSEs
6.58826e-9/6.83730e-9 respectively.

Stage 15 also rejects rays outside a conservative bounding box for the entire reachable surface.
It includes the original solver's residual acceptance tolerance. The rejection does not alter
sample positions, root refinement, fallback tessellation, or shader evaluation resolution.
The magnitude bound is computed once per local ray query and reused for sample pruning.

| JAR mode, 300x200 / 8 samples / 7 renders | Original warm median | Stage 15 warm median | Speedup | Stage 15 peak process RSS |
| --- | --- | --- | --- | --- |
| unlimited | 2.736543 s | 0.699438 s | 3.91x | 1,197,588,480 B |
| 16384 | 2.793875 s | 0.689158 s | 4.05x | 1,065,566,208 B |

Unlimited EXR maximum difference is 4.76837e-7, RMSE 6.27789e-9. First renders were
17.0644 s unlimited (including compilation) and 1.72338 s 16K. These remain small diagnostic
renders: the 10x target, cached-2048-equivalent speed and full-resolution acceptance are open.

An in-memory edit sequence at 150x100 / 2 samples tested scene scale, maximum distance and
steps, a nonuniform object transform, material displacement scale, shared mesh creation/removal,
and clamp toggles. The original renderer initially hit a Metal GPU recovery error on the shared
mesh; a sequential retry completed. Candidate/reference differences were:

- Initial and scene-setting edits: exactly equal.
- Nonuniform transform: maximum 0.000245750, RMSE 1.43514e-6.
- Material edit: maximum 1.41561e-7, RMSE 7.07675e-10.
- Shared mesh: maximum 0.000355586, RMSE 2.77943e-6.
- After instance removal and clamp toggles: exactly equal to the original, but **both builds have
  a large output change compared with the equivalent scene before adding the instance**. This is
  an existing issue, not evidence of correct invalidation. Do not count the abnormally fast render
  after instance removal as an optimization gain. The issue remains to be investigated.
- Clamp toggles isolated from instance creation/removal preserve the original result (maximum
  1.41561e-7, RMSE 7.07675e-10 for the final unlimited render).

The larger transform/shared-mesh differences need investigation before broad fidelity acceptance;
these are not described as roundoff-only passes. Reproduction scripts, per-case EXRs and reports
are under `/tmp/resolution-speed/regression-edits/`, with scripts
`/tmp/resolution-speed/regression_edits.py` and `regression_toggle.py`.
Stage 15 procedural measurement is pending at the time of this entry. No installed user app or
source .blend was overwritten.

Stage 15 16K comparison also passed: maximum 4.76837e-7, RMSE 6.37753e-9.
The procedural variant completed with no concurrent build: warm median 0.436916 s versus
1.417447 s original (3.24x), peak process RSS 769,228,800 B versus 775,110,656 B.
Maximum EXR difference 2.38419e-7, RMSE 9.65968e-9. First render 21.1503 s includes compilation.
The cached-2048 regression run is currently pending in
`/tmp/resolution-speed/verified-bounds15-2048.log`, exec session 95592.

Graph inspection confirmed the visible JAR displacement is UV -> identity Mapping -> linear
image Color -> grayscale conversion -> object-space Displacement (midlevel 0, scale 0.25).
The procedural variant uses Noise -> Color Ramp -> Displacement. A next candidate is a certified
straight-line evaluator that removes interpreter dispatch/stack traffic for such graphs while
calling identical node math; it must retain a full-interpreter fallback and be measured before
claiming a gain. No such evaluator beyond the current compact interpreter has been implemented.

The cached-2048 run was still compiling at 3m20s: benchmark PID 42635 was waiting and its
Metal compiler service PID 42636 was active at 99.6% CPU, RSS about 7.4 GB. This is temporary
compiler memory outside the benchmark process's RSS. It has not produced a render or a valid
cached-mode result yet. Cold compilation memory/time must be included in final memory acceptance;
process-only warm-render RSS does not establish that requirement. Resume session 95592, and do
not overlap another GPU benchmark while it remains active.

The stage 15 cached-2048 run completed: warm median 0.0722227 s, first render 269.801 s,
peak benchmark-process RSS 1,219,674,112 B. EXR maximum 4.76837e-7, RMSE 6.59379e-9.
The warm result is about 14% slower than the original 0.0633312 s median; repeat a clean warm
comparison before calling that stable. Cold compilation is plainly expensive and remains open.

## Stage 16: fused image evaluator (implementation, not yet measured)

A symbolic verifier walks every instruction and stack dependency of the compiled displacement
program. It accepts one vector attribute, optional fixed texture mapping, one image, a grayscale
conversion or image alpha, geometry normal, constant midlevel/scale, and object-space displacement.
Unsupported operations/dependencies retain their original evaluator. Original bytecode remains
unchanged; the fused descriptor is appended after its END. Descriptor addresses are relocated with
shader bytecode on each compilation. KernelShader's existing evaluator word encodes either full,
compact, or a fused descriptor address, keeping its record size unchanged.

The GPU fused path calls existing attribute evaluation, transform, texture projection/filtering,
grayscale conversion, and object normal/direction transform math directly. It preserves coordinate
derivatives and image flags. Scene evaluator bits allow Metal to remove the unused interpreters
when all referenced displacement programs are fused. Baking continues to use the original SVM.
Build, eligibility, render fidelity, invalidation and speed verification remain pending.

Stage 16 built and its CPU compiler probe certified Material.002. Its apparent warm median
0.442334 s (clean repeat 0.434718 s, RSS 1,063,059,456 B) is **not a valid speedup result**:
EXR maximum difference 0.253638, RMSE 0.000627010. It failed fidelity. Initial compilation took
317.097 s and the separate Metal compiler service reached about 8.34 GB RSS. Do not count it
against the performance or memory target.

Stage 17 separates the fused numerical operations at the same original node boundaries to
prevent cross-node floating-point contractions, particularly coordinate evaluation and P + D.
This is a hypothesis under test, not a confirmed cause/fix. The symbolic verifier was separated
into `shader_displacement_image.cpp`. Six C++ tests pass for color/alpha/derivative programs,
stack reuse, all truncated prefixes, overwritten vectors, mismatched derivatives, extra output
operations and unknown opcodes. They are registered with the Cycles test suite; this checkout
has WITH_GTESTS off, so the tests were also compiled and run directly against the implementation
and bundled GoogleTest. Runtime validation of stage 17 remains pending.

User steering: also improve caching at unrestricted resolution with minimal or no overhead
relative to the current version. Include preparation, lookups, render memory and cold compiler
cost in acceptance. Do not equate a fixed cache storage budget with a spatial resolution cap;
misses must retain exact evaluation. Investigate reusable shader subexpressions or acceleration
of the existing fixed fallback samples rather than growing a dense micromesh to 16K.

Stage 17 did not fix the fused image discrepancy (maximum 0.253638, RMSE 0.000627010).
Stage 18 restores lcg_state/closure initialization; texture mip selection consults lcg_state even
for image-only graphs. That is a necessary fix, but stage 18 still has the same mismatch.
Neither stage 17's 0.431929 s nor stage 18's 0.418747 s median is a valid optimization result.
The current source makes fused evaluation opt-in with
`CYCLES_PIXEL_DISPLACEMENT_ENABLE_FUSED_IMAGE`; default renders retain the verified evaluator.
The original bytecode and compiler verifier remain available for diagnosis. A per-sample raw
full-versus-fused GPU comparison is the next useful diagnostic; node-boundary speculation has
not explained the actual discrepancy.

## Stage 19: cache row-group bounds (pending)

The existing fallback cache stores float3 displacements in float4 records. Six previously unused
W fields per pair of grid rows now store conservative min/max XYZ bounds for those rows' exact
microtriangles. No device array or cache record grows, and the 16 MiB budget is unchanged.
Bounds are computed from existing baked values and current scene scale/clamp; they include
barycentric tolerance and a host/device arithmetic margin. Nonfinite or uncertain arithmetic
uses unbounded groups. Ray queries preserve original row/triangle order and refill the boundary
row after skipping a group, retaining equal-distance tie handling.

Stage 19 built and installed. Its running JAR unlimited check uses
`CYCLES_PIXEL_DISPLACEMENT_DISABLE_FUSED_IMAGE=1` because that installed revision predates the
source's opt-in default. This tests the new cache bounds with the verified compact evaluator.
Log `/tmp/resolution-speed/verified-cache19-unlimited.log`, exec session 14596. Runtime speed,
image equality and edit invalidation are pending. No valid cached-bounds gain is claimed yet.

Stage 19 completed without a useful gain: JAR median 0.695434 s versus stage 15's 0.699438 s,
within measurement noise; procedural median 0.453524 s versus stage 15's 0.436916 s.
JAR image equality passed (maximum 4.76837e-7, RMSE 6.48538e-9). The extra row-bound work did not
justify retention under the user's minimal-overhead requirement. It was removed; source snapshots
are archived as `/tmp/resolution-speed/cache19-mesh-displace.cpp` and
`cache19-pixel-displacement.h`. No row-bound speedup is claimed.

Stage 20 restores the previous cache implementation, makes the failing fused evaluator opt-in,
and adds a diagnostic comparison of full SVM and fused displacement at identical GPU sample
coordinates. `CYCLES_PIXEL_DISPLACEMENT_ENABLE_FUSED_IMAGE=1` together with
`CYCLES_PIXEL_DISPLACEMENT_COMPARE_FUSED_IMAGE=1` enables it. The diagnostic prints maximum/RMS
world-space position difference and intentionally stops rendering, rather than treating diagnostic
data as cached displacement. Normal operation continues using the verified evaluator and cache.
Build and raw comparison are pending. This is investigation toward the original target, not a
change in acceptance criteria; unrestricted-resolution caching with negligible overhead remains open.

Stage 20's paired GPU diagnostic completed as designed (intentional exit 1 before rendering):
1,112 identical sample coordinates, maximum world-space position difference 2.98023e-8,
RMS 1.2639e-9. Log `/tmp/resolution-speed/raw-compare20.log`. This suggests numerical differences
at the sampled points rather than a large shader-value mismatch there; it does not yet explain
or validate the render discrepancy at arbitrary coordinates. Analyze pixel outliers and expand
paired sampling before accepting/rejecting the fused evaluator. It remains opt-in.

Pixel outlier analysis for the stage 18 image is recorded in
`/tmp/resolution-speed/fused-difference-analysis.json`: median per-pixel max RGB difference
3.94881e-7, 99th percentile 2.19346e-5; 15 of 60,000 pixels exceed 0.001, two exceed 0.01,
and one exceeds 0.1. The largest change is at (47,61). These data are consistent with small
numerical changes amplified by some path/intersection decisions, but that explanation remains
unproven at the outliers. No fidelity acceptance threshold was silently relaxed.

Next bounded implementation option: cache the verified instruction sequence and invoke the
original SVM node functions directly in a fixed order with the original 32-float stack, rather
than returning intermediate values through new functions. Extend the small per-shader descriptor
with original mapping/conversion/geometry/displacement/set-displacement node parameters; preserve
stack dependencies and normal/constant inputs. This would remove dispatch while retaining the
exact shared numerical functions. Measure it before retention. Another larger caching direction
is a conservative hierarchy over native image blocks (not baked lower-resolution displacement),
with exact shader evaluation for candidates; such a hierarchy is not implemented yet.

## Stage 21: cached instruction sequence (pending runtime validation)

The opt-in image descriptor now stores original typed node parameters remapped to a canonical
20-float stack. The GPU executes the original attribute, optional mapping, image, conversion,
geometry, displacement and output functions in fixed order. Source midlevel/scale constants are
preserved exactly; the verifier checks dependencies before remapping. This removes dispatch without
introducing replacement implementations of the numerical operations. Six updated verifier tests
pass, and the full build passed. Raw paired GPU comparison is pending in
`/tmp/resolution-speed/raw-compare21.log`. This path remains opt-in and has no accepted speed result.

Stage 21's paired GPU comparison passed exactly: 1,112 samples, maximum 0, RMS 0.
The diagnostic intentionally exits before rendering. The render benchmark is now running with
`CYCLES_PIXEL_DISPLACEMENT_ENABLE_FUSED_IMAGE=1`, log
`/tmp/resolution-speed/verified-sequence21-unlimited.log`. Exact sampled shader values alone do
not establish complete render fidelity or the performance target; both remain under test.

Stage 21's render still has the same outlier discrepancy: maximum 0.253638, RMSE 0.000627010.
Its 0.558121 s warm median is not accepted as a fidelity-preserving speedup. The exact paired
bake comparison therefore does not establish equality in the rendering path. Stage 22 adds
`CYCLES_PIXEL_DISPLACEMENT_FORCE_FULL_EVALUATOR=1`, retaining the cached descriptor and shader
metadata but forcing original SVM for ray evaluations. Compare this against the baseline to
separate evaluator math from metadata/kernel-specialization effects. This is diagnostic only;
default behavior still leaves the image sequence opt-in.

Stage 22 completed: retaining the descriptor but forcing original SVM restored the reference
image (maximum 4.76837e-7, RMSE 6.62188e-9). Warm median 0.765724 s, process peak RSS
1,190,068,224 bytes. This isolates the discrepancy to the selected evaluator/render execution
path rather than descriptor presence alone. It does not establish a new performance gain.

Stage 23 expands the paired diagnostic from fallback lattice points to 4,096 stratified,
off-grid barycentric coordinates per eligible triangle. All 32,768 GPU samples match exactly
(maximum 0, RMS 0), log `/tmp/resolution-speed/raw-compare23.log`. The diagnostic intentionally
stops before rendering. The first launch rejected repeat=1 before any render; the successful
diagnostic used repeat=3 as required by the benchmark parser. No timing is claimed from either.
The cached sequence remains opt-in because the rendered discrepancy is unresolved. Next useful
isolation is intersection evaluation versus final shading-normal evaluation under the specialized
render kernels; bake-kernel equality alone has now been tested substantially more broadly.

## Stage 24/25: isolate and preserve shading evaluation

`CYCLES_PIXEL_DISPLACEMENT_FORCE_FULL_SHADING=1` forces original SVM only for the final
displaced geometry and tangent/normal calculations. Intersection evaluation remains optimized.
Stage 24, using the cached original-node sequence, passes both JAR image comparisons:
unlimited maximum 4.76837e-7 / RMSE 6.36355e-9; 16K maximum 4.76837e-7 / RMSE 6.73207e-9.
Warm medians are 0.565657 s and 0.587209 s, respectively. This isolates the earlier visible
discrepancy to the optimized evaluator in the shading path for this test.

Stage 25 restores the lean return-value image evaluator using the current verified descriptor,
while retaining original shading through the same diagnostic switch. Both 300x200/8-sample JAR
comparisons pass: unlimited maximum 4.76837e-7 / RMSE 6.35657e-9; 16K maximum 4.76837e-7 /
RMSE 6.30690e-9. Medians are 0.469619 s and 0.483186 s (about 5.8x original in either mode).
Process peak RSS is 1,172,652,032 and 1,073,102,848 bytes. These are process measurements,
not whole-system cold Metal compiler memory. The combination remains opt-in pending broader
validation. Stage 24's sequence implementation is archived in
`/tmp/resolution-speed/sequence24-pixel-displacement.h`.

The benchmark now accepts and records `--seed`. A larger 750x500/16-sample seed-7 baseline
is running, log `/tmp/resolution-speed/verified-baseline-large7.log`, followed sequentially by
the stage 25 candidate. The >10x and cached-2048-speed targets remain unmet.

The larger seed-7 test completed sequentially: baseline median 24.454293 s, candidate
3.699606 s (6.61x). Blender process peak RSS: 1,059,930,112 versus 1,075,183,616 bytes
(1.44% higher). Maximum image-channel difference 6.42240e-6, RMSE 9.70128e-9 over
1,500,000 channels. These data support a broader improvement but do not prove the full
performance or total-memory requirements.

Stage 26 makes original shading automatic for shaders with an image descriptor; the separate
FORCE_FULL_SHADING switch is removed. The image evaluator itself remains opt-in. Non-image
shaders retain their compact/full selection. Seven parser tests pass, including a new
nonidentity texture-mapping test for both derivative modes. Stage 26 built successfully;
runtime validation of the automatic safeguard is next.

Stage 26 automatic shading safeguard passes the small JAR unlimited comparison: maximum
9.53674e-7, RMSE 6.55067e-9; median 0.442928 s, process peak RSS 1,200,783,360 bytes.
The procedural JAR variant also passes: maximum 4.76837e-7, RMSE 1.05506e-8; median
0.446543 s, RSS 768,180,224 bytes. This is comparable to stage 15's 0.436916 s procedural
result, not an additional procedural speedup. Logs are
`/tmp/resolution-speed/verified-automatic26-unlimited.log` and
`/tmp/resolution-speed/automatic26-procedural.log`. No GPU jobs remain from these tests.

Next concrete optimization candidate: specialize ShaderData setup for the verified image
program. It currently transforms Ng and both surface tangents and computes compact position
derivatives on every evaluation. Ordinary image attributes need barycentric derivatives but do
not read those position differentials; the GENERATED fallback with dual coordinates does need
them. Preserve the full setup for general shaders and generated-coordinate fallback, and test
a minimal image setup that skips only provably unread fields. Preserve P/N arithmetic and
original shading. This has not been implemented or measured yet.

## Stages 27–30: lean setup and native image range cache

Stage 27 skips Ng/tangent transforms and compact position derivative computation for verified
image intersection programs whose attribute is not GENERATED. General evaluation, generated
coordinates, raw baking, and original final shading retain full setup. No resolution or sample
lattice changes. Small unlimited JAR median 0.432104 s, maximum difference 4.76837e-7 / RMSE
6.50504e-9. Larger seed-7 median 3.204576 s (7.63x baseline), process RSS 1,075,871,744 bytes;
maximum difference 6.42240e-6 / RMSE 1.00835e-8. Stage 26's header is archived at
`/tmp/resolution-speed/automatic26-pixel-displacement.h`.

Read-only inspection of the supplied displacement image (`low_tide_rocks_disp_4k.png`, 4096²)
found decoded RGB range 0.0380905–0.294441, with sRGB encoding. Stage 28 added lazy native
scalar-integer pixel range collection from the existing loaded device-image host buffer; it
did not tighten this scene's bound because it initially excluded sRGB. Its timing therefore
does not establish a range-cache gain. Stage 29 includes monotonic sRGB decoding and a
1e-6 sampling margin; clip-extension black is always included. Tiled, HDR, multichannel, and
unknown conversion paths retain conservative prior bounds. No extra image allocation is made.
The cached range resets on loading; shader bounds refresh after image updates, before rendering.

Stage 29 logs the actual loaded displacement magnitude bound 0.0736189 versus 0.25001 before
loading. Small 300x200/8-sample results:

| Mode | Median seconds | Process peak RSS | Max image difference | RMSE |
| --- | ---: | ---: | ---: | ---: |
| unlimited | 0.327297 | 1,095,663,616 | 4.76837e-7 | 6.43782e-9 |
| 16384 | 0.315931 | 1,066,860,544 | 4.76837e-7 | 6.45782e-9 |

Independent 750x500/16-sample seed-7 results (three renders each, warm median of last two):

| Mode | Baseline seconds | Candidate seconds | Speedup | Baseline RSS | Candidate RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| unlimited | 24.454293 | 2.345921 | 10.42x | 1,059,930,112 | 1,072,955,392 |
| 16384 | 24.165796 | 2.375576 | 10.17x | 1,071,874,048 | 1,077,116,928 |

Larger image differences: unlimited maximum 6.42240e-6 / RMSE 9.90542e-9; 16K maximum
6.42240e-6 / RMSE 9.55432e-9. These exceed 10x on the specified larger JAR tests, not every
workload. The small tests remain below 10x, procedural performance remains ~3.2x, cached-2048
speed is not reached, and whole-system cold compilation memory is not established.

Stage 30 prevents collecting image statistics when pixel displacement is off and uses an atomic
dirty flag to avoid repeating late shader-bound analysis on unchanged renders. Build passed.
`/tmp/resolution-speed/regression-image-range.py` replaces the original texture with a generated
HDR image, changes its pixel buffer, then restores the original. Baseline/candidate30 comparisons
are exactly equal for all four outputs; replacement changes the image and restoration returns
exactly to the original. However, the in-place buffer edit does NOT change the output in either
build, so it is NOT a valid passed edit-invalidation check. Logs show bound 0.0736189 -> unknown
for HDR -> 0.0736189 on restoration. Results:
`/tmp/resolution-speed/image-range-edits/comparison30.json`. A file-reload fixture that demonstrably
changes rendered pixels is still needed. All these experiments still enable the image evaluator
with `CYCLES_PIXEL_DISPLACEMENT_ENABLE_FUSED_IMAGE=1`; it is not the default yet.

## Stage 31: default image evaluator and sequence invalidation

The image evaluator is now enabled by default for certified programs; the diagnostic opt-out is
`CYCLES_PIXEL_DISPLACEMENT_DISABLE_FUSED_IMAGE=1`. Original shading remains automatic. Stage 31
built and installed under `/tmp/resolution-speed/stage31`; the user's installed app is unchanged.

The first file-reload fixture did not change rendered pixels in either build, despite rewriting
and reloading the temporary PNG, so it does not prove invalidation. The initial sequence fixture
also had an inherited frame offset and requested nonexistent files; those results are invalid.
The corrected sequence fixture explicitly sets offset zero and clears object animation. It loads
real 16-bit grayscale frames, with no image-load errors. Both builds visibly change by the same
amount (maximum channel change 54.8822) and restore exactly. All three baseline/candidate outputs
match exactly. Candidate bounds follow 0.0500103 -> 0.225009 -> 0.0500103. Results:
`/tmp/resolution-speed/image-sequence-fixed/comparison.json`; script
`/tmp/resolution-speed/regression-image-sequence.py`.

Larger cached-2048 reference median is 0.384872 s, RSS 1,312,800,768 bytes. The stage 31 initial
run incurred 311.47 s cold compilation; MTLCompilerService was observed at about 8.35 GB RSS.
Unit-test compilation may overlap the end of that run, so a clean subsequent run was used for
timing: median 0.389928 s, RSS 1,328,758,784 bytes. Cached image comparison maximum 1.90735e-6,
RMSE 8.07618e-9. The large cached reference and unrestricted candidate remain about sixfold apart.

Without diagnostic environment overrides, stage 31 large unlimited (seven renders, seed 7)
has median 2.314369 s (10.57x baseline), RSS 1,083,195,392 bytes. Maximum image difference
6.42240e-6 / RMSE 9.81185e-9. The equivalent seven-render large 16K check is running in
`/tmp/resolution-speed/verified-default31-large7-16384.log`.

## Native bounds hierarchy prototype (not connected to rendering)

`util/image_displacement_bounds.h` builds packed 16-bit minimum/maximum intervals over all native
scalar integer pixels. Leaves cover at least 16x16 pixels; larger images use coarser bounds leaves
to cap the base grid at 256x256. This caps acceleration metadata, not texture or displacement
resolution. Parent nodes conservatively merge children; a rectangle query uses at most four
nodes at a covering level. A square maximum-size hierarchy uses 349,524 bytes of node payload.

Two standalone tests pass, including 800 deterministic rectangle checks against every covered
native pixel, odd/thin dimensions, partial-edge extrema, byte normalization and invalid inputs.
Binary `/tmp/resolution-speed/image-bounds-tests32`. The prototype is registered in CMake but
has not been attached to image loading or GPU intersection queries; no speed gain is claimed.

Next integration must preserve conservative coverage: build from already loaded scalar integer
pixels; impose a total metadata memory budget; upload alongside image data and invalidate on
reload; derive an affine UV rectangle covering the ray's reachable segment plus displacement and
solver-acceptance padding; account for filtering footprints and use global bounds for unsupported
mapping, subdivided attributes, wrapping or uncertain arithmetic. Query bounds only to skip
unreachable portions of the existing solver, preserving its sample lattice. Reuse unused padding
in KernelImageTexture for offset/block-shift if its layout remains unchanged. This integration is
not implemented yet.

Stage 31 large 16K completed (seven renders): median 2.329047 s, RSS 1,077,886,976 bytes,
maximum image difference 6.42240e-6 / RMSE 9.60364e-9. Both larger modes therefore retain
the measured >10x multiplier with the optimization enabled by default.

## Stage 32: compact reference shading

Reference shading now uses the original compact SVM interpreter when all active image programs
meet its opcode/32-float-stack contract. Kernel evaluator bit 16 records any image program that
requires full reference SVM; diagnostic force-full bit 8 still overrides the optimization. This
retains original node operations and source stack layout, unlike the lean image intersection
evaluator. General/ineligible cases keep their existing interpreter selection. The objective here
is reduced cold compiler cost, not a new shading approximation.

Stage 32 built and the small unlimited comparison passes (maximum 4.76837e-7, RMSE 6.44758e-9).
Median 0.330711 s, first render 33.132862 s, process RSS 1,196,834,816 bytes. Cached-mode cold
compilation and warm rendering are under test in
`/tmp/resolution-speed/verified-compactref32-large7-2048.log`; no cold-cost improvement is claimed
yet. Stable stage-31 source backups are `/tmp/resolution-speed/stable31-pixel_displacement_shader.h`
and `/tmp/resolution-speed/stable31-shader.cpp`. The bounds hierarchy remains unconnected to GPU
rendering; its two standalone tests pass.

Stage 32 cached-2048 completed: first render 32.523295 s versus stage 31's 311.470595 s;
warm median 0.384634 s, process RSS 1,306,853,376 bytes. At 20 seconds the active compiler
was observed at 2,091,376 KiB RSS (not a captured peak), versus stage 31's observed ~8.35 GB.
Cached image comparison maximum 1.90735e-6 / RMSE 8.89343e-9. This is a substantial measured
reduction in first-render cost without a warm-render regression; it is not a fair cold-memory
comparison against the original baseline. Larger stage-32 unlimited verification is running in
`/tmp/resolution-speed/verified-compactref32-large7-unlimited.log`.

Stage 32 larger verification completed (750x500, 16 samples, seed 7, seven renders):

| Mode | Warm median | Blender process peak RSS | Maximum image difference | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Unlimited | 2.357957 s | 1,085,030,400 bytes | 6.42240e-6 | 9.72238e-9 |
| 16384 | 2.273452 s | 1,080,721,408 bytes | 6.42240e-6 | 9.73013e-9 |

These are 10.37x and 10.63x faster than their original direct-evaluation references. Process
memory increases are approximately 2.37% and 0.83%, respectively. Process RSS excludes the
separate Metal compiler service and is not a whole-system peak-memory measurement. Cached
2048 remains approximately six times faster than either unrestricted path.

The stage-32 persistent-data settings/transform/material/clamp-toggle fixture also completed.
Initial and settings images match exactly. Transform maximum difference is 0.000245750,
RMSE 1.43514e-6; material, cached-2048 and return-to-unlimited maximum is 1.41561e-7,
RMSE 7.07675e-10. These match the earlier bound-path regression results; the larger transform
difference remains disclosed. This fixture does not test shared-mesh removal or motion.
Results: `/tmp/resolution-speed/regression-edits/stage32-comparisons.json`.

## Stage 33: late displacement-image range requests

Range collection is factored into `ImageSingle::update_displacement_range`. If a loaded image
is first used for displacement after its initial upload, its first range request scans retained
host storage once. Subsequent requests reuse the two-float range, including unsupported-format
results, without image reloads or copies. Existing image-load collection uses the same routine.
Tiled images remain conservative unknowns. The late-use fixture initially uses the height image
as surface color with displacement disconnected and disabled, then restores the surface and
enables displacement without reloading the image. Stage 32 retains the loose 0.25001 magnitude
bound; stage 33 obtains 0.0736189. All three output images (surface only, enabled, repeated)
match exactly between builds. Script: `/tmp/resolution-speed/regression-late-image.py`;
diagnostic logs: `/tmp/resolution-speed/late-image-stage{32,33}.log`. These short regression runs
are not used as a throughput benchmark.

Stage 33 procedural check (150x100, four samples, three renders): warm median 0.404457 s,
first render 19.755774 s, process RSS 785,088,512 bytes. Against the original procedural
reference, maximum image difference is 4.76837e-7, RMSE 1.04722e-8. This is approximately
3.50x faster, still below the 10x target. The seven parser and two hierarchy tests pass;
the Blender build and `git diff --check` pass. The hierarchy remains unused by rendering.

Stage 33 larger unlimited check retains the improvement: median 2.325764 s (10.51x original),
process RSS 1,079,279,616 bytes (1.83% above original), first render 3.323429 s. Maximum image
difference 6.42240e-6, RMSE 9.94429e-9. The corrected low/high/restored image-sequence fixture
also matches the original baseline exactly in all three frames, visibly changes between frames,
and restores exactly. Bounds follow 0.0500103 -> 0.225009 -> 0.0500103. Comparison:
`/tmp/resolution-speed/image-sequence-fixed/comparison33.json`. The user scene's SHA-256 remains
`87585175d268d2efc537bd995166f5e21e808aab382fb61bbdc9b7ae5dc9b98c`.

## Triangle input caching experiments (stages 34–36)

Stage 34 cached three decoded normals per eligible static, single-object triangle inside the
existing 16 MiB fallback-sample budget. Large JAR unlimited median was 2.276856 s; disabling
the normal cache in the same binary gave 2.324572 s. The unchanged JAR image retained maximum
difference 6.42240e-6 / RMSE 9.90989e-9 against the original reference.

Stage 35 additionally cached native triangle coordinate values for certified image programs.
Large unlimited median was 2.145129 s; the same binary with input caching disabled gave
2.447958 s. The initial render incurred 48.655504 s compilation and process peak RSS was
1,201,913,856 bytes; this cold run does not establish warm-run memory overhead. JAR image maximum
difference was 6.42240e-6 / RMSE 9.65570e-9.

**Decoded-normal caching was rejected and removed.** A stronger persistent-data fixture bends
the mesh, enables smooth normals, edits UVs, and switches to another UV attribute. Stages 34
and 35 both differ from stage 33 by maxima 0.00205684, 0.00137973 and 0.00407916 in the three
edited cases (RMSE up to 2.13540e-5). Stage 35 with input caching disabled matches stage 33
exactly in all four cases. This isolates the difference to decoded-normal caching; the
unchanged flat JAR benchmark was insufficient validation. The original packed-normal decoding
and interpolation are restored in stage 36. Script `/tmp/resolution-speed/regression-input-cache.py`;
comparisons `/tmp/resolution-speed/regression-edits/input-cache-comparisons{34,35,35-disabled}.json`.

Stage 36 retains only coordinate caching, pending runtime validation. Four float4 records per
eligible triangle store eligibility plus the three original float2/float3 vertex/corner values;
the maximum payload increase on JAR is 512 bytes. These records count against the existing
total budget. GPU baking certifies supported attribute type/domain and output conversion;
the host records validity in the patch flags, avoiding a validity-buffer read per query.
Interpolation, derivatives, texture mapping, native texture sampling and reference shading
retain their existing operations. Unsupported cases use the existing attribute evaluator;
raw comparison diagnostics bypass the coordinate cache. Cache offsets skip the new records
before accessing the original fallback samples. No texture resolution limit is introduced.

Stage 36 coordinate-only caching matches stage 33 **exactly** in all four initial/normals/
UV-edit/UV-switch images. The fixture is retained as
`tools/pixel_displacement_input_cache_regression.py`. The normal-cache experiment remains
removed; its measured gain is not included in the retained change.

Large JAR verification, seven renders at 750x500/16 samples/seed 7:

| Mode | Warm median | Blender process peak RSS | Maximum image difference | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Unlimited | 2.178916 s | 1,081,950,208 bytes | 6.42240e-6 | 9.74401e-9 |
| 16384 | 2.154731 s | 1,079,410,688 bytes | 6.42240e-6 | 9.48159e-9 |

Speedups against original direct evaluation are 11.22x and 11.22x, with process memory
increases of 2.08% and 0.70%. First renders were 3.296453 s and 3.209602 s, respectively,
after the edit regression had already compiled the new intersection kernels. These are not
cold-compiler memory measurements. Cached-2048 and procedural verification are pending.

Stage 36 initial cached-2048 check completed: median 0.394471 s, first render 42.324885 s,
process RSS 1,308,229,632 bytes. Image maximum difference 2.86102e-6, RMSE 8.30919e-9.
The first procedural check completed: median 0.416152 s, first render 30.779703 s, process
RSS 849,231,872 bytes; image maximum difference 4.76837e-7, RMSE 9.79313e-9.

Two subsequent follow-up jobs labeled `coordcache36-clean-large7` and
`coordcache36-clean-procedural` overlapped. Their timings are excluded. Both were confirmed
terminal before starting isolated replacements. The isolated cached-2048 replacement,
`coordcache36-solo-large7`, has median 0.387117 s (0.58% above the original cached reference),
first render 11.783051 s, and process RSS 1,262,960,640 bytes. The isolated procedural replacement
is pending. Cold compiler costs still vary; these checks do not prove zero initialization cost.

The isolated procedural replacement completed: median 0.414180 s, first render 12.056175 s,
process RSS 820,363,264 bytes. This is 3.42x the original procedural speed, with 5.84% more
process memory in this run. Procedural performance and initialization memory therefore remain
open parts of the full objective. The permanent repository regression fixture also ran and
matched stage 33 exactly in all four cases; report
`/tmp/resolution-speed/regression-edits/input-cache-repo-comparisons36.json`.

Final stage-36 ablation, `CYCLES_PIXEL_DISPLACEMENT_DISABLE_INPUT_CACHE=1`, completed in isolation:
large unlimited median 2.433628 s, process RSS 1,079,246,848 bytes. Enabled coordinate caching
is 10.47% faster in elapsed time than this same-binary control. However, that control is 4.64%
slower than the earlier stage-33 result (2.325764 s); timing variability versus added fallback
code/register cost has not been resolved. Do not claim zero overhead for ineligible cases.

Further useful work: separate the certified-coordinate evaluator from the original fallback
callee to avoid retaining both code paths in its private state; consider constructing coordinate
records directly from retained host attribute buffers to reduce GPU bake/compiler work. For the
certified path, a smaller shader-data setup can be considered while retaining every original
normal/transform operation and the reference shading path. The texture helper reads `lcg_state`
and `runtime_flag` through `kernel/util/image_2d.h`; this is a starting point for an input audit,
not proof that a reduced setup is already safe. No such follow-up change is implemented yet.

`MetalKernelPipeline::should_use_binary_archive` explicitly disables archives for MetalRT
intersection pipelines because binary linked functions are unsupported there. This explains why
a new process can still incur intersection compilation; current measurements do not justify a
claim of zero cold overhead. The supplied .blend SHA-256 remains unchanged and `git diff --check`
passes.

## Stage 37: separate coordinate evaluator variants

The cache-layout check moves to the direct image-evaluation caller, which selects separate
certified-coordinate and original-attribute variants. The certified variant no longer retains
attribute lookup/fallback code. Geometry motion and invalid offsets use the original variant.
Large JAR unlimited median with caching enabled is 2.124522 s, RSS 1,076,969,472 bytes;
disabled is 2.416093 s, RSS 1,208,369,152 bytes (that first run included 40.530161 s compilation).
Image comparisons retain maximum 6.42240e-6, RMSE 9.76158e-9 enabled / 9.98448e-9 disabled.
This improves enabled performance modestly; the fallback-overhead question is not resolved.

## Stage 38: construct coordinate records from native host buffers

Coordinate certification and copying move out of the GPU bake kernel. The host reads the same
compiled attribute ID and attribute-map chain, supports only native float2/float3 vertex/corner
values with vector output, and checks map/index bounds. It copies original float values without
interpolation, mapping, normal decoding, or resampling. Unsupported inputs keep the old evaluator.
Only three float4 records per eligible triangle are stored, inside the existing budget; the
eligibility header is eliminated. GPU baking now evaluates only the original fallback samples.
The original bake kernel is restored, reducing additional GPU code and jobs. The host logs the
number of baked samples and native coordinate records for verification. Runtime checks are pending.

Stage 38 large unlimited median was 2.121291 s, process RSS 1,087,782,912 bytes. The original
four-case edit fixture matched stage 33 exactly, and logs confirmed 8 triangles / 1112 baked
samples / 24 native-coordinate records. However, extending the fixture to generated coordinates
exposed a maximum image difference of 0.175345 (RMSE 0.00131155) versus stage 33. The earlier
GPU-built coordinate cache in stage 37 has exactly the same difference, isolating it from host
copying. Missing UV and repeated missing-UV cases match exactly and allocate no coordinate records.

A temporary stage-39 paired GPU diagnostic compared cached/original generated coordinates at
32768 identical points: maximum value difference 5.96046e-8, derivative differences 2.98023e-8
in both axes. These small arithmetic differences can change this render. The diagnostic's first
render took 304.181 s to initialize; the compiler was observed around 7 GB RSS. Its code was
removed from production source afterward. Diagnostic source snapshots and log are in
`/tmp/resolution-speed/diagnostic39-{bake.h,mesh_displace.cpp}` and
`/tmp/resolution-speed/coordinate-pair39.log`; its intentional exit code 1 is not a benchmark.

The expanded fixture was also checked against the original immutable baseline. Stage 33 already
differs in the nonflat/UV-edit/UV-switch cases by maxima 0.0142778 / 0.0202169 / 0.00298572,
and in generated coordinates by 0.0214014 (RMSE 0.000146959). Thus exact agreement with stage 33
alone does not prove original-baseline fidelity on those newly tested cases. Stage 38's generated
maximum is 0.175345 against the original, too. Reports:
`/tmp/resolution-speed/regression-edits/input-cache-original-comparisons{33,38}.json`.

Stage 40 replaces value caching with attribute-descriptor caching. The host stores only element,
type, and a losslessly split integer offset in one float4 per eligible triangle. The kernel skips
attribute-map traversal but calls the original surface-attribute evaluator to fetch, interpolate,
and convert native values. This eliminates the alternate interpolation path and reduces metadata
to 16 bytes per eligible triangle (128 bytes for JAR). It is under validation; no fidelity or speed
claim is made yet. Earlier scalar/magnitude-bound and lean-intersection differences on nonflat
geometry remain part of the outstanding acceptance work.

Stage 40 completed all seven expanded edit cases and matched stage 33 exactly, including generated
coordinates. The metadata-cache regression is fixed. Large unlimited median is 2.345704 s
(10.43x original), first render 3.396158 s, peak process RSS 1,091,207,168 bytes (2.95% above
original). Against the original large EXR, maximum difference is 6.42240e-6, RMSE 9.82816e-9.
This does not clear the expanded nonflat fidelity failures inherited from stage 33.

Two stage-40 ablations isolate those failures. Forcing the full evaluator and disabling only
image fusion (using the compact original-node interpreter) give the same seven-case comparisons
against the original: initial/UV-edit/missing-UV/repeated-missing exact; normals and generated
maximum 3.72529e-9; UV-switch maximum 1.24495e-6, RMSE 9.35232e-9. The scalar bounds and fallback
cache remain enabled in both. Reports are `input-cache-original-{full40,compact40}.json` under
`/tmp/resolution-speed/regression-edits`. This points to the fused evaluation path rather than
metadata caching as the source of the much larger inherited differences.

Stage 41 tests using the original displacement node for the final height-to-vector operation
inside the fused path. It uses the parser's existing canonical 32-float stack layout, preserving
normal input and object-space operations. Build/runtime validation is pending.

Stage 41 built and completed the seven-case regression. Every comparison value is identical
to stage 40's original-baseline comparison, so restoring only the displacement node does not
resolve the inherited differences. It was reverted; no performance benefit is claimed.
Stage 42 now tests original attribute/mapping/image/conversion node calls in the height path,
using the already certified canonical stack. The lean final vector calculation is restored.
This ablation bypasses descriptor lookup caching while retaining all other acceleration.

Stage 42 also reproduces all stage-40 comparison values exactly; replacing the height path does
not fix the regression. It was reverted. Stage 43 instead removes the compact-interpreter
substitution from the explicitly full final-shading path. This keeps lean image evaluation and
metadata caching for intersections while always calling the original full evaluator for final
shading derivatives. The stage-32 substitution had not been checked on these nonflat fixtures.

Stage 43 also returns exactly the stage-40 comparison values in every expanded case; it was
reverted. Its installed kernel source was checked and contains the intended full-evaluator
branch, so the experiment's source was actually installed. Stages 41–43 are diagnostic snapshots
only. The working source is restored to stage 40's measured descriptor-cache implementation.
The next useful isolation is shader-data setup / intersection-path specialization, since
individual original-node substitutions and final-shading interpreter substitution did not
change the inherited errors, while disabling fusion or forcing the original evaluator did.
No additional cached-2048 performance target was introduced.

Stage 44 restores full shader-data setup for all image evaluations. It reproduces the stage-40
expanded comparison values exactly and was reverted. Inspection found that `ccl_device_noinline`
is empty on Apple Metal, so previous comments about preserving call boundaries were not sufficient
compiler guarantees. Existing Metal-specific noinline patterns exist in image_3d.h and mnee.h.
Stage 45 uses that explicit attribute on `pixel_displacement_eval_image` only, to separate its
position update from the caller's subtraction. Runtime validation is pending.

Stage 45 changes the comparison: generated matches original to 3.72529e-9 and missing UV cases
match exactly, but UV cases worsen (initial max 0.247856, normals 0.096713, UV edit 0.249534,
UV switch 0.412163). It is not accepted. Stage 46 combines the explicit call boundary with full
shader-data initialization to isolate whether the UV-only setup shortcut causes that difference.

Stage 46 is identical to stage 45 in all seven image comparisons, ruling out the omitted
position differentials as the cause of this change. Stage 47 adds explicit Metal noinline
attributes to coordinate, mapping, sampling, height conversion, and vector helpers, retaining
the full setup for this isolation. This tests boundaries that the Apple macro did not enforce.

Stage 47 produces exactly the stage-45/46 comparison values; additional helper boundaries do
not change that result. A same-build run with descriptor caching disabled is testing whether
metadata lookup interacts with this explicit-boundary result. No stage-45+ speed claim is made.

Disabling descriptor caching in stage 47 produces identical comparison values, so metadata
lookup does not explain those errors. Source hashing was checked: Metal hashes expanded source
and specialization constants, and installed experiment headers contain the changes.
Stage 48 tests the entire original-node fixed sequence behind the explicit Metal call boundary,
with full setup and the original full final-shading evaluator. Unlike stage 24's sequence, the
Metal boundary is now actually enforced. This combination is diagnostic until measured.

Stage 48 still returns stage-45's image differences and is rejected. The user then explicitly
requested focusing on the reported JAR 10x result with good quality, without unnecessary extra
work. Stage 49 restores stage 40 and restricts its fast path to cached native float2 coordinates
on static triangles with identical packed corner/vertex normals. Other image inputs retain
full original evaluation, including final shading. This is a geometry/input eligibility rule,
not a resolution cap, material-name exception, or changed texture sampling rate.

Stage 49's seven-case original-baseline comparison passes at the previous full-evaluator level:
initial/UV edit/missing UV/repeated missing exact; normals/generated max 3.72529e-9; UV switch
max 1.24495e-6, RMSE 9.35232e-9. Large unlimited median 2.476588 s, RSS 1,091,993,600 bytes:
9.87x original, slightly below the requested 10x. Stage 50 adds a conservative scene-wide flag:
full fallback is enabled by default and disabled only after every image triangle has valid
certified metadata. Metal can then omit unused fallback code for fully eligible scenes.
Geometry/cache rebuilds and shader uploads reset the conservative flag before recertification.

Stage 50 passes the seven-case quality comparison with the same values as stage 49 and restores
performance to 10.79x unlimited / 10.80x at 16K, with the memory and EXR results summarized above.
No further broad optimization experiments were added after this verification.
