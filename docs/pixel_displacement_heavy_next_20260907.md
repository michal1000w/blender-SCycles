# Additional heavy-displacement acceleration

This pass compares against commit `65c65ac597d` and the preserved
`install/Blender-Speed.app`. Gains from the preceding pass are not counted.
The candidate is `install/Blender-Heavy.app`.

Installed-app GPU timing, image comparisons, regressions, and coverage checks passed.

## What changed

The host certifies static normal inputs, a well-conditioned metric, resident linear
image sampling, rigid transforms, and valid grazing fallback grids. Additional
certificates identify scalar images and identity mapping. These certificates are
cleared and rebuilt with displacement updates. Metal can eliminate the general
shader and cached-surface branches when those paths cannot apply.

Eligible varying-normal triangles use conservative native-image bounds in a frame
aligned with their mean normal. A 64-cell hierarchy is the best measured balance
for the optimized solver. General/light cases keep the previous 32-cell world-axis
hierarchy and avoid an unnecessary ray-frame transform. The combined samples and
bounds still have the existing 16 MiB limit; native image resolution is unchanged.

The projected ray coordinates are prepared once per triangle intersection and reused
in the repeated native height evaluations. The normal metric remains available
for evaluating displacement along the interpolated normal. The repeated coordinate
inverse no longer forces inlining. On Apple Metal the
existing `ccl_device_noinline` macro lets the compiler choose; it does not force a
separate machine function. Its equations, convergence threshold, and five-iteration
limit remain unchanged. Prepared native evaluation also supplies Newton's existing
finite-difference stencil. Final shading retains its original evaluator.

Metal specializes displacement step count, global scale, and maximum distance so
it can fold repeated clamps and constant factors. Changing these controls can
therefore require another kernel specialization. On M2 Max/Ultra, certified
closest/shadow kernels use a compiler thread limit of 512 and dispatch groups of
64. Other architectures retain their previous scheduling.

There is no new displacement-height cache, warm-sample approximation, reduced
native resolution, or reduced search/refinement limit. The original coarse sample
positions, 128-point retry, refinement, and grazing fallback are retained.

## Measurement method

Tests run sequentially outside the sandbox on the Apple M2 Max 38-core GPU, with
MetalRT enabled, identical scenes, seed 7, and matching sample counts. Adaptive
sampling and denoising are disabled for comparison. EXRs are compared in linear
RGB. Scene files are loaded but never saved.

Exploratory runs showed substantial timing variation, so final comparisons pass
`--warm-up` to both builds. This is Cycles' existing switch to wait for all Metal
kernel compilation before rendering. Nine renders are measured for each heavy
case. The first includes scene preparation and is reported separately; the warm
median includes all eight subsequent renders. CPU builds and other GPU tests do
not overlap these runs. These are throughput measurements, not cold-start promises.

With the revised affine-coordinate solver at 3000×2000 and 16 samples, compiler
thread limits of 64, 128, and 256 produced warm medians of 8.83, 8.80, and 8.18 s.
A limit of 512 produced 7.26 s with 32-thread dispatch groups and 7.21 s with
64-thread groups. The latter configuration is used in the final app.
The contemporaneous baseline was 15.70 s. These tuning runs do not replace the
final paired results below.

<!-- FINAL RESULTS -->
All tests use 3000×2000 output and 16 samples except the odd-resolution row,
which uses 2593×2053 and 8 samples. Times are warm medians after compilation and
scene preparation, against the preceding optimized build (not stock Cycles).

| Scene / mode | Baseline → candidate | Speedup | Peak process RSS change | Relative RGB L1 |
|---|---:|---:|---:|---:|
| Cycles2-1, unlimited | 15.691 → 7.112 s | 2.206× | +12.3% | 0.423% |
| Cycles2-1, clamp 16384 | 15.710 → 7.153 s | 2.196× | +6.0% | 0.423% |
| Cycles2-1, clamp 4097; 2593×2053 | 6.575 → 2.927 s | 2.246× | +2.3% | 0.387% |
| Cycles2, unlimited | 5.207 → 2.702 s | 1.927× | +1.2% | 0.070% |
| Cycles, unlimited | 3.494 → 3.510 s | 0.995× | -0.7% | 0.000% |

The heaviest scene exceeds 2× in all three tested direct-mode configurations.
The lighter scenes are not claimed to exceed 2×.

<!-- END FINAL RESULTS -->

## Memory and repeated process check

Peak RSS includes the entire process, including kernel compilation and setup; it
is not a dedicated GPU allocation counter. The initial unlimited run peaked 12.3%
above baseline. A subsequent fresh-process pair at the same settings peaked only
0.6% above baseline, with a 2.18× warm speedup (15.775 → 7.235 s; three renders per
build for this supplemental memory check). Both observations are retained. The
main nine-render comparisons above remain the primary speed results. The combined
displacement sample/bounds storage limit remains 16 MiB.

## Correctness checks

The regression matrix covers changes to strength, distance, steps, transforms,
normals, UV edits and selection, generated/missing coordinates, interpolation,
projection, RGBA and alpha heights, and large coordinates. It also tests odd native
image dimensions (2053×2063) before leaving the optimized path, restoration of the
original image, clamped direct mode above 2048, cached mode, and restoration of
direct mode. Full-resolution white-emission coverage checks detect new interior
holes in all three supplied scenes.

All 36 final regression comparisons passed. The maximum relative RGB L1 difference
was 0.432%. All five restoration/repetition checks passed, and all three
full-resolution coverage checks found zero new interior holes. The three native
image-bounds unit tests pass, including conservative extrema and continuous bilinear
footprints. The app build, Python syntax checks, and `git diff --check` passed.

## Rejected approaches

Diagnostics/prototypes were retained under `/tmp/displacement-heavy-next/`, not in
the delivered shader. Removing retry or refinement lost intersections or changed
the image substantially. An adaptive sampling lattice changed relative RGB by
22.3% and was rejected. Larger/finer hierarchies, occupancy masks, analytic inverse
variants, alternate compiler boundaries, and color-conversion approximations did
not provide a worthwhile measured improvement. The uniform-image specialization
also took over five minutes in its first render without improving throughput;
it was removed. The final code keeps the original color conversion and inverse.

## Research and reproducibility

[Projective Displacement Mapping, Hoetzlein (2025)](https://arxiv.org/abs/2502.02011)
examines normal-aligned prisms and direct boundary intersections.
[GPU geometry-image ray tracing](https://www.cs.cmu.edu/~kmcrane/Projects/RayTracingGeometryImages/paper.pdf)
and [pyramidal displacement traversal](https://nahjaeho.github.io/papers/CAG14.pdf)
provide hierarchy-based alternatives to uniform marching. These motivate
conservative spatial rejection, but do not establish this implementation's speedup.

Use `tools/pixel_displacement_performance.py` with the baseline and candidate apps,
matching `--warm-up`, `--mode`, `--percentage 100`, `--samples 16`, `--seed 7`,
and `--repeat 9`.
Use `tools/pixel_displacement_compare.py` for the resulting EXRs. Both scripts
record their actual settings and scene hash.

Artifacts: `/tmp/displacement-heavy-next/grid-tuning/`,
`/tmp/displacement-heavy-next/final-validation/`, and
`/tmp/displacement-heavy-next/regressions/`. The final manifest records executable,
installed-header, and scene hashes. The preserved baseline headers match commit
`65c65ac597d`.

Machine-readable evidence, including all measured render times, process-memory
peaks, image metrics, regression comparisons, coverage results, and hashes, is in
`docs/pixel_displacement_heavy_next_20260907.json`. The supplied scenes were not saved
or modified. Changes are left in the working tree; no commit was created.
