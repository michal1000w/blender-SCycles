# Direct Metal displacement, September 6 follow-up

This investigation targets an **additional 2x** over commit `5f65710aad4`, rather than
counting the earlier stage-50 optimization again. The supplied scene has changed:
SHA-256 `60fc6874361d86ba3aca4ccab799437e05ce29ddd9cc410c58bab759ae8e264d`.
The source `.blend` is never saved by the tests.

## Implementation

- Specialize compiler-certified native float2 triangle attributes, retaining native
  interpolation and the original texture sampler.
- Prepare UVs, the certified uniform normal, and the triangle Gram matrix once per ray.
  Reuse that context for marching and Newton residual evaluation. Other shaders,
  varying normals, missing attributes, shared meshes, and motion retain the existing path.
- Build a small BVH over the **existing** grazing-fallback sample grids. Bounds include
  the original microtriangle edge tolerance and CPU/GPU roundoff. This adds no displacement
  samples and retains the same grazing-fallback geometry. The combined fallback sample
  and BVH allocation is capped at the existing 16 MiB budget; insufficient space uses
  the existing brute-force fallback. The normal full-resolution shader remains the
  intersection evaluator, and the dense micromesh cache is not enabled for unlimited mode.
- Use 128-thread shadow dispatch groups for certified direct scenes on M2 Max/Ultra,
  retaining the existing compiler thread limit/register budget. Other kernels, GPU
  families, and cached-displacement scenes retain the previous tuning.
- Keep render samples, root-search iterations, displacement steps, texture resolution,
  filtering, and final shading settings unchanged.

These changes are independent of framebuffer and texture resolution. Optional clamp modes
still fall back to direct evaluation when their dense cache exceeds its existing budget.
Arithmetic scheduling and hit ordering can change floating-point intersection results;
image comparisons therefore measure actual differences rather than claim bitwise equality.

## Research considered

[RMIP (SIGGRAPH Asia 2023)](https://iliyan.com/publications/RMIP/) combines inverse mapping
with conservative rectangular displacement bounds, avoiding dense tessellation.
[TFDM (SIGGRAPH Asia 2021)](https://perso.telecom-paristech.fr/boubek/papers/TFDM/)
decouples acceleration data from the base domain. These motivate keeping acceleration
metadata small and independent of the native displacement texture resolution.
[Apple's shader performance guidance](https://developer.apple.com/videos/play/tech-talks/111373/)
explains specialization and register/resource pressure. The retained per-ray work reduces
repeated data access and setup. This implementation is not an implementation of RMIP.
[Projective Displacement Mapping (2025)](https://arxiv.org/abs/2502.02011) explores direct
sampling with a top-level hardware BVH, tight prisms, and ray–bilinear patch intersection.
That alternative would change the intersection representation and needs separate fidelity
validation. Here the existing solver and fallback geometry remain in use.

Experiments with resident-only texture sampling, forced inlining, early whole-material BVH
bounds, altered secant convergence, premapped UVs, CPU-scaled fallback samples, finer BVH leaves,
and a reduced Metal compiler thread limit did not improve throughput and were removed.
A deliberately incomplete no-grazing-fallback diagnostic was used only to estimate its cost;
it contributes no performance or quality claim.

## GPU validation

Tests use the supplied scene on Apple M2 Max (38 GPU cores), MetalRT, outside the
sandbox. Each baseline/candidate pair has identical render resolution, samples,
seed 7, displacement steps 16, displacement scale 1, maximum distance 1, native
4096x4096 displacement image, disabled denoising/adaptive sampling, and persistent data.
The table reports warm wall-clock render medians; first renders include compilation
and preparation and are excluded. GPU jobs and builds never overlap.

| Mode | Render pixels | Samples | Baseline | Optimized | Speedup |
| --- | --- | --- | --- | --- | --- |
| Unlimited | 3000x2000 | 16 | 32.5211 s | 13.6275 s | 2.386x |
| 16384 clamp, direct fallback | 3000x2000 | 16 | 32.5880 s | 13.5927 s | 2.397x |
| 3001 clamp, direct fallback | 2593x2053 | 8 | 14.1265 s | 5.9052 s | 2.392x |

The unlimited baseline/candidate peak process RSS is 1,448,558,592 / 1,443,725,312
bytes (-0.33%); the 16384 pair is 1,457,324,032 / 1,447,346,176 bytes (-0.68%).
The arbitrary-resolution pair is 1,407,107,072 / 1,410,383,872 bytes (+0.23%).
Process RSS includes CPU and shared allocations; it is not a dedicated VRAM measurement.
The initial 16384 candidate run drifted from 17.71 to 15.26 s (median 16.68 s) and did
not pass 2x. A fresh process repeated the same unchanged build and settings, producing
stable 13.54–13.64 s warm timings. Both reports are retained for transparency.
These are warm-render improvements, not a claim about kernel compilation latency.

The full unlimited linear-RGB comparison has relative L1 **0.04627%**, mean absolute
error 0.00005354, RMSE 0.00159905, 99th-percentile absolute channel error 0.00067025,
and maximum 1.65199610. 62,665 of 6,000,000 pixels (1.0444%) have a channel difference
above 0.001. Sparse intersection/path changes produce larger outliers; these differences
are not roundoff-only and the result is not bit-identical.

The arbitrary-resolution comparison has relative L1 0.04312%, RMSE 0.0022676,
and maximum 1.876427. Its baseline and candidate previews were exported with AgX.
The 16384 comparison has the same measured differences as unlimited.

The seven-case normal/UV fixture completed on both builds. Candidate-versus-baseline
relative L1 ranges from 0.000011% to 0.12524% at 150x100, two samples. Normal changes,
UV edits, UV-map switching, generated coordinates, and missing UVs all changed the
reference output substantially. Repeating the missing-UV render is identical.

The nine-case setting/texture fixture initially omitted explicit scene update tagging,
so several Python edits did not affect its reference images. Those ineffective checks
are excluded. The corrected fixture explicitly tags the scene and packs the resized
texture in memory, without saving either input file. The corrected nine-case sequence
completed on both builds: initial, scale, distance clamp, steps, nonuniform transform,
2053x2063 texture, 3001 clamp, 2048 cached mode, and restored unlimited mode.
Scale/distance/steps/transform/texture edits all substantially change the reference image.
The 3001 clamp correctly matches unlimited because it uses the same direct fallback.
Switching to 2048 cached mode and back changes the reference output as expected.
Candidate-versus-baseline relative L1 is at most 0.048781% in this sequence, and the
2048 cached result is bit-identical. Every compared image is finite and equal-sized.

The packaged-build debug smoke test reports **8 triangles, 480 BVH nodes, 33,280 bytes**
combined fallback samples and bounds. The new bounds use 15,360 bytes; existing samples
use 17,920 bytes. This is far below the retained 16 MiB combined budget, and unlimited
mode does not construct the dense displacement micromesh. A separate 3001-mode debug
smoke test confirms the same direct-fallback allocation.

`tools/pixel_displacement_performance.py` records device, scene hash, settings, warm
timings, peak RSS, and completion only after the comparison EXR has been saved.
`tools/pixel_displacement_compare.py` compares finite, equal-size linear EXRs.
Raw reports are in `docs/benchmarks/direct-displacement-20260906/`; large EXRs and
additional experimental artifacts reside in `/tmp/direct-speed-20260906/`.

The measured speedups apply to this scene and GPU. The per-ray and fallback-BVH changes
have no power-of-two or 2048-pixel requirement, but speedups depend on shader eligibility
and scene/ray behavior. The new dispatch tuning is restricted to M2 Max/Ultra; other
GPU architectures retain their previous dispatch settings.

## Reproduction and delivered build

The complete CMake-installed candidate is `install/Blender-Direct.app`.
The original baseline is retained at `install/Blender-Displacement.app`.
The baseline executable embeds an older build hash, but its displacement kernel
sources were verified byte-for-byte against checkout `5f65710aad4` before testing.
This is the installed baseline captured before any changes in this investigation.
The candidate executable and runtime kernel header match the complete CMake installation.

From the repository root, run each app sequentially with the same arguments:

```sh
install/Blender-Direct.app/Contents/MacOS/Blender --background --factory-startup \
  /Users/michal/Documents/BlenderProjects/JAR/JAR-test-Cycles.blend \
  --python-exit-code 1 --python tools/pixel_displacement_performance.py -- \
  --mode unlimited --percentage 100 --samples 16 --seed 7 --repeat 5 \
  --label candidate --output /tmp/displacement-reproduction
```

Use `--mode 16384` for the high-clamp test. For the arbitrary-resolution test use
`--mode 3001 --width 2593 --height 2053 --percentage 100 --samples 8 --seed 7 --repeat 3`.
Compare the resulting EXRs with `tools/pixel_displacement_compare.py`.
Run the two regression scripts with their documented baseline/candidate invocation.

Validation completed: Release `blender` target build, full CMake installation, GPU
packaged-build smoke test, matched performance runs, all 16 regression cases on both
builds, finite EXR comparisons, Python bytecode compilation, and `git diff --check`.
The original scene SHA-256 remains unchanged. Source/binary hashes and completion
records are saved beside the benchmark reports.
