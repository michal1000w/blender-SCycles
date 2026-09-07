# Second direct-displacement optimization pass (2026-09-07)

This pass treats commit `2d8e59710b4514c64672878942aa7f8704e8e66f` and the preserved
`install/Blender-Direct.app` as its baseline. It does **not** claim the speedup from
[the preceding pass](pixel_displacement_direct_20260906.md) again.

## Implementation

- Separate evaluator-kind matching from input/sampler capability bits. A scene containing
  only image displacement can still specialize to that evaluator after adding a capability.
  This is particularly important on Metal: otherwise unreachable interpreter paths retain
  expensive shader state and substantially reduce intersection throughput.
- Certify the resident, linear, flat-projection image path after displacement inputs are
  prepared. It uses the same native image and GPU sampler, alpha handling, and color
  conversion. Tiled images, UDIMs, other filtering/projections, and uncertified inputs
  retain the general path. Certification is reset during every relevant update.
- Prepare constant triangle UVs, mapping, normals, and the barycentric Gram matrix alongside
  the existing fallback BVH. This adds 64 bytes per triangle, not displacement samples,
  within the existing combined 16 MiB budget. Final shading retains original attribute
  evaluation and node arithmetic.
- Convert ray position to barycentric coordinates once and evaluate the affine coordinates
  in the sampling loop. Anchor them at the clipped interval start to reduce cancellation
  for distant rays. Prepare the uniform displacement direction and scalar clamp once.
- Keep the original evaluator for large coordinates, ill-conditioned transforms or
  triangles, varying normals, and unsupported attributes. The existing dense-cache path
  remains available for its original modes.

Render sample count, native texture resolution, coarse-search lattice, eight safeguarded
secant refinements, Newton checks, and grazing-fallback geometry remain unchanged. Unlimited
mode does not acquire a dense displacement cache. No benchmark-specific environment switch
is required.

## Research and rejected experiments

Apple's [Metal ray-tracing performance guidance](https://developer.apple.com/videos/play/wwdc2022/10105/)
and [GPU optimization discussion](https://developer.apple.com/videos/play/tech-talks/111373/)
motivated reducing live state and specializing out unused paths. The useful result here is
software-BVH direct evaluation on M2 Max; it does not require newer hardware ray-tracing units.

[Projective Displacement Mapping](https://arxiv.org/abs/2502.02011) and
[TFDM](https://perso.telecom-paristech.fr/boubek/papers/TFDM/TFDM_lowres.pdf) describe more
structural intersection/bounding alternatives. They informed investigation, but were not
substituted for the existing surface definition.

Native MetalRT direct evaluation rendered its initial setup but repeatedly caused Apple's
pipeline compiler to abort on specialization. Lower compiler concurrency, thread budgets,
and the resident sampler did not resolve it. All native-direct trial hooks were removed.
Tighter software BVH bounds and reuse of final-shading state did not improve throughput.
An [inverse-quadratic root-finding experiment](https://maths-people.anu.edu.au/~brent/pub/pub005.html)
was slower and changed the image substantially (17.1% relative L1 at 1500x1000); it was removed.
No altered root convergence is part of the delivered candidate.

## Validation

Tests use the supplied JAR scene on the Apple M2 Max (38 GPU cores), outside the sandbox.
Baseline and candidate GPU jobs run sequentially, without overlapping builds. Each timing
below is the median of four renders following the first render; samples, seed 7, scene
settings, native 4096-pixel texture, and output size are identical within each pair.

| Test | Baseline | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| Unlimited, 3000×2000, 16 samples, initial pair | 13.6103 s | 4.7965 s | 2.838× |
| Unlimited, 3000×2000, 16 samples, repeated pair | 10.0594 s | 4.8893 s | **2.057×** |
| 16384 clamp, 3000×2000, 16 samples, initial pair | 10.4512 s | 5.8048 s | 1.800× |
| 16384 clamp, 3000×2000, 16 samples, repeated pair | 11.5759 s | 5.6477 s | 2.050× |
| 3001 clamp, 2593×2053, 8 samples | 4.3840 s | 2.0988 s | 2.089× |

The conservative repeated unlimited result is the principal performance claim. Timings
varied between pairs; the faster initial result is not a universal guarantee. The user
withdrew the clamp speed target while final validation was underway. Its measured results
are retained here, including the initial pair below 2×. These are rendering throughput
measurements, not a claim of 2× faster initial kernel compilation or scene loading.

At 3000×2000, repeated unlimited process peak RSS was 1,444,495,360 bytes for baseline and
1,455,538,176 bytes for candidate (**+0.76%**). The first candidate run, including new kernel
compilation, peaked 9.18% above its paired baseline. RSS is process memory on unified-memory
hardware, not a dedicated-VRAM measurement or whole-system compiler-memory peak.
The code adds 512 bytes of triangle input records for this eight-triangle scene; the native
texture and existing fallback samples retain their sizes.

All compared EXRs are finite. Relative L1 difference from baseline is **0.10649%** at
3000×2000 in both unlimited and 16384 modes, and **0.09900%** at 2593×2053. Full-resolution
p99 absolute RGB difference is 0.0016604; the maximum is 3.4178 in sparse bright samples.
This is not pixel-identical output. AgX previews show the same visible surface/detail.
Odd-sized unlimited and 3001-clamp outputs agree to a maximum absolute difference of
0.0000019073.

All **23** scene-edit/input/sampler regression cases completed on the GPU and produced finite
EXRs. The maximum relative L1 difference in these small fixtures is 0.13742%:

- Nine direct-edit cases: scale, distance, steps, nonuniform transform, odd 2053×2063 texture,
  odd clamp, cached mode, and restored direct mode. Cached mode is pixel-identical.
- Seven input cases: changed normals, UV edits, switched UV layer, generated coordinates,
  missing UVs, and a repeated missing-UV render. Maximum relative L1 is 0.13114%.
- Seven sampler/precision cases in `tools/pixel_displacement_specialization_regression.py`:
  initial, cubic, nearest, spherical projection, colored RGBA, alpha height, and whole-scene
  translation by 100000 units. The translated case is pixel-identical; RGBA and alpha-height
  differences are at most 0.000602% relative L1. Logs confirm that resident specialization
  is disabled for cubic/nearest filtering and enabled again for the eligible RGBA image.

The initial direct-fallback allocation is 33,792 bytes: 17,920 bytes of existing samples and
15,872 bytes of bounds/input records. No dense unlimited-mode cache was introduced.

## Installed result and reproduction

The complete candidate is `install/Blender-Next.app`. It was built with
`cmake --build build/macos_arm64_Release --target blender -j 8`, installed coherently with
`cmake --install ... --prefix /tmp/direct-speed-20260907/final-install`, then copied into
that app bundle. The original `Blender-Direct.app`, `Blender.app`, and
`Blender-Displacement.app` were preserved.

Candidate shader SHA-256:
`2d9df6831dc78183fe0455842e3abaefea49d664c7def402464645cc9fe7a48a`.
Baseline shader SHA-256:
`1e6ec8dffe14e5d14fee4f30b8cd235ea9b1f3fc8b0676fc2dbcb4f7360cd3ee`.
Installed shader, kernel types/data declarations, and Metal entry source match the final
source tree. The build, CMake installation, GPU regressions, and `git diff --check` succeeded.

Example benchmark (run baseline and candidate sequentially):

```sh
install/Blender-Next.app/Contents/MacOS/Blender --background --factory-startup \
  /Users/michal/Documents/BlenderProjects/JAR/JAR-test-Cycles.blend \
  --python-exit-code 1 --python tools/pixel_displacement_performance.py -- \
  --mode unlimited --percentage 100 --samples 16 --seed 7 --repeat 5 \
  --label candidate --output /tmp/displacement-verification
```

These speedups are measured for this scene and M2 Max. Unsupported programs continue to use
the general evaluator; arbitrary image/output dimensions do not depend on a fixed-size
baked displacement grid.

Raw benchmarks, comparisons, and rejected prototypes are under
`/tmp/direct-speed-20260907/`; they are not checked into the repository. The supplied scene
is never saved by the test scripts. Its SHA-256 is
`60fc6874361d86ba3aca4ccab799437e05ce29ddd9cc410c58bab759ae8e264d`.
