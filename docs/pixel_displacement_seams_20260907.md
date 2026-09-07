# Unlimited displacement seam correction — 2026-09-07

This continues the fast direct evaluator at commit `2f8b8c46919a`. The baseline is
`install/Blender-Next.app`; the complete candidate is `install/Blender-Seams.app`.
The current versions of both supplied .blend files were used. Clamp performance
was excluded from the target at the user's request.

## Intersection correction

The old solver projected ray samples onto the geometric face plane to choose
texture coordinates. With interpolated displacement normals, an interior surface
point can move tangentially across that projection's edge. This caused both a large
seam and scattered geometry misses in the second scene.

The candidate inverts the interpolated normal extrusion field for ray samples.
It intersects the three ruled prism sides analytically, using quadratic edge
solutions to retain narrow valid intervals and reject rays outside the displaced
face. Failed normal-prism searches receive a denser 128-step retry; root refinement
allows up to 20 iterations and terminates on a converged residual. Final hit
validation and shading use the existing displacement evaluation.

Native float2 UV eligibility is now independent of uniform-normal eligibility.
Resident linear images can use prepared UV coordinates and the normal field during
the height search. Image resolution, filtering, displacement scale, samples, seed,
and the saved .blend settings are preserved. Displacement samples the native image
directly; additional storage is small input metadata and per-ray work data.

Whole scenes with uniform, nearly face-parallel displacement directions retain the
compact flat kernel. This classification accounts for 16-bit octahedral normal
quantization. Mixed-normal scenes retain the stronger per-face search, including
steep faces. The specialization flags are cleared and rebuilt when scene inputs
change.

## Research

[Projective Displacement Mapping for Ray Traced Editable Surfaces (Hoetzlein, 2025)](https://arxiv.org/html/2502.02011v1)
identifies prism intersections, watertight boundaries, and thin features as separate
parts of robust direct displacement tracing. Its ray/bilinear-boundary approach
informed the ruled-side clipping here. This implementation retains the existing
normalized displacement field and adds the inverse mapping and analytic edge
intersections around it.

## Validation method

All GPU runs used the Apple M2 Max, 38-core Metal GPU with MetalRT enabled, outside
the sandbox. Jobs ran sequentially. Throughput comparisons used 3000×2000 pixels,
16 samples, seed 7, adaptive sampling and denoising disabled, and three renders per
process. Warm time is the median of the last two renders; the first render includes
kernel compilation and scene preparation. Memory is peak process RSS, not a claim
of directly measured GPU-only allocation.

The coverage fixture replaces only the surface connection with white emission,
preserves displacement, isolates SSand, disables depth of field, and renders one
sample against transparency. It compares alpha against the interior of a clamped
reference, eroded by 48 pixels to exclude silhouette/filtering differences. This
checks actual geometry misses independently of lighting and displaced shading.
The .blend files and their source textures are never saved by the tests.

## Full-resolution results

| Scene | Previous warm time | Candidate warm time | Change | Previous peak RSS | Candidate peak RSS |
|---|---:|---:|---:|---:|---:|
| JAR-test-Cycles.blend | 6.233 s | 6.202 s | -0.49% | 1.356 GiB | 1.372 GiB |
| JAR-test-Cycles2.blend | 51.539 s | 52.491 s | +1.85% | 1.369 GiB | 1.365 GiB |

The original scene is 0.49% faster; the corrected second scene is 1.85% slower than
the previous fast build. Peak process memory changes by +1.13% and -0.26%, respectively.
These measurements preserve the previous throughput to within 2%; they do not claim
an additional 2× gain over Blender-Next.

First-render times, including compilation and preparation, were 26.39 → 34.26 seconds
on the original scene and 70.99 → 79.99 seconds on the second scene.

| Final coverage at 3000×2000 | Interior pixels checked | Missing pixels |
|---|---:|---:|
| JAR-test-Cycles.blend | 5,284,165 | 0 |
| JAR-test-Cycles2.blend | 1,205,393 | 0 |

All compared EXRs are finite. On the original scene, maximum absolute linear RGB
difference is `3.8146973e-6`, relative L1 is `9.2124061e-9`, and no pixel differs by
more than `0.001`. Its output is effectively unchanged.

The second scene changes visibly as missing geometry and incorrect intersections
are repaired. Its relative L1 difference from the faulty unlimited baseline is
`0.661853`; this is a correctness change, not an image-equivalence claim. The repaired
surface was checked against the clamped reference's coverage and overall shape.

[Previous second-scene render](/tmp/displacement-seams-20260907/final-compare-new/baseline.png) ·
[Corrected second-scene render](/tmp/displacement-seams-20260907/final-compare-new/candidate.png) ·
[Clamped reference preview](/tmp/displacement-seams-20260907/reproduction/baseline.png)

## Scene-edit regressions

All **32** Metal GPU cases completed with finite output:

- Nine direct-mode cases on each scene: initial render, scale, maximum distance,
  steps, nonuniform object transform, a 2053×2063 texture, a 3001 clamp, cached mode,
  and restored unlimited mode.
- Seven input cases: initial state, edited normals, edited UVs, another UV layer,
  generated coordinates, missing UVs, and repeated missing UVs.
- Seven sampler/precision cases: initial state, cubic, closest, sphere projection,
  odd-size RGBA, alpha-driven height, and large coordinates.

Restoring unlimited mode matches the earlier direct render to maximum absolute
errors of `4.7684e-7` and `1.1921e-7` on the two scenes. Repeated missing-UV output is
identical. Normal and UV edits produce changed images. The comparison assertions
and per-case records are in
[final-regression-stats.json](/tmp/displacement-seams-20260907/final-regression-stats.json).

## Build and reproduction

The build and complete CMake installation succeeded:

```sh
cmake --build build/macos_arm64_Release --target blender -j 8
cmake --install build/macos_arm64_Release --prefix /tmp/displacement-seams-20260907/final-install
```

The resulting bundle is
[Blender-Seams.app](/Users/michal/Documents/CppApps/blender-SCycles/install/Blender-Seams.app).
Its executable matches the build output; installed shader, kernel data/types, and
Metal entry source match the source tree. The prior app bundles were preserved.
`git diff --check` passes.

Final displacement shader SHA-256:
`b89624c4f668fa9ef68535e3d0de0b9d7a167089def8db85c2d5501ed5358720`.

Input scene SHA-256 values remain unchanged:

- `JAR-test-Cycles.blend`: `04a13941c1179298509633158bae5a812eb6e121a6b3ed4d7475570b10659293`
- `JAR-test-Cycles2.blend`: `d38bab0605c6651e6ad2efa8d00589bb6ff8daddee2424c67d632fa6aa9b9660`

Example unlimited benchmark:

```sh
install/Blender-Seams.app/Contents/MacOS/Blender --background --factory-startup \
  /Users/michal/Documents/BlenderProjects/JAR/JAR-test-Cycles2.blend \
  --python-exit-code 1 --python tools/pixel_displacement_performance.py -- \
  --mode unlimited --percentage 100 --samples 16 --seed 7 --repeat 3 \
  --label verify --output /tmp/displacement-verify
```

The reusable coverage fixture is `tools/pixel_displacement_coverage.py`. Full test
logs, timing JSON, 32-bit EXRs, previews, and the installed-source manifest are under
`/tmp/displacement-seams-20260907/`. The existing earlier optimization report remains
in `docs/pixel_displacement_direct_20260907.md`.
