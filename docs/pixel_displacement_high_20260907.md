# Direct Metal displacement acceleration — high displacement, 2026-09-07

This pass uses the seam-corrected source at `278039837469` and the preserved
`install/Blender-Seams.app` as its baseline. Earlier optimization gains are not
counted again. The candidate is `install/Blender-Speed.app`.

## Implementation

Static triangles prepare their interpolated normal-field coefficients, native UVs,
and Gram matrix once. The 96-byte record replaces repeated input reconstruction
inside ray queries. Displacement heights still come from the native texture.

For eligible scalar integer images, a conservative min/max pyramid includes every
native pixel. Its dimensions adapt to arbitrary image sizes. The host uses it to
bound continuous bilinear footprints, including repeat/extend/clip boundaries, and
combines those height intervals with bounds on interpolated, normalized normals.
A 32-cell triangular partition supplies a compact hierarchy of surface envelopes.
These are bounding boxes, not a displacement image or an intersection mesh.

The GPU traverses those envelopes using escape links, without a traversal stack.
It rejects unreachable rays and trims the part of the original search lattice
that can contain a root. Both neighboring bracketing samples remain available.
The outer software BVH uses the union of these envelopes and the existing grazing
fallback bounds. Dense-cache mode keeps its previous software-BVH behavior.

Rigid-transform specialization reduces scalar height evaluation to the normal
field's metric. Prepared displacement evaluation also serves final hit validation.
Ill-conditioned metrics retain vector evaluation. Motion, unsupported attributes,
large coordinates, non-rigid bound transforms, and unsupported image formats keep
the applicable general paths. Near-opposed normals and non-finite bound parameters
are excluded from envelope certification.

The existing combined **16 MiB** fallback sample/bounds budget remains the limit.
Optional surface envelopes use the space left after fallback and input records. Only one temporary
image pyramid is retained during construction (at most about 342 KiB); triangles
sharing it reuse the native-pixel scan. No dense height cache is introduced in
unlimited mode. Geometry, UV, image, shader, transform, and integrator updates
rebuild the relevant metadata and specialization flags.

Sample count, seeds, native texture resolution, coarse-search steps, 128-step
normal-prism retry, safeguarded root refinement, and the grazing fallback surface
are retained. Final shading derivatives retain their original evaluator.

## Research

- [TFDM, Thonat et al., SIGGRAPH Asia 2021](https://perso.telecom-paristech.fr/boubek/papers/TFDM/)
  motivates separating displacement acceleration from a fully tessellated surface.
- [RMIP, Thonat et al., SIGGRAPH Asia 2023](https://iliyan.com/publications/RMIP)
  demonstrates the value of tight displacement bounds over rectangular texture
  regions together with inverse mapping. This implementation keeps the current
  solver and builds bounded object-space envelopes before rendering.
- [Projective Displacement Mapping, Hoetzlein, 2025](https://arxiv.org/html/2502.02011v1)
  discusses normal-field projection, prism boundaries, and thin-feature preservation.
  The existing seam-corrected prism definition is retained here.
- [Apple's Metal ray-tracing performance guidance](https://developer.apple.com/videos/play/wwdc2022/10105/)
  informed the emphasis on live state, specialization, and reducing intersection work.

Bounds are built from every native pixel before the first sample. This makes them
safe for later rays even when early rendering samples have not visited a narrow peak.

## Measurement method

All renders run outside the sandbox on the Apple M2 Max, 38-core Metal GPU, with
MetalRT enabled. Direct mode uses the existing software BVH2 path on this hardware.
Baseline and candidate run sequentially, without overlapping builds or GPU jobs.
Each process renders three times; reported warm time is the median of renders two
and three. The first render is recorded separately because it includes shader
compilation and preparation. No speedup claim includes earlier optimization passes
or a reduction in samples, displacement strength, texture resolution, or ray depth.

The supplied `.blend` files and textures are read without saving them. Benchmarks
use seed 7, adaptive sampling and denoising disabled, and identical settings within
each pair. Linear 32-bit EXRs provide image comparisons. RSS is peak **process**
memory on unified-memory hardware, not a separate VRAM or whole-system measurement.

| Scene / mode | Baseline warm seconds | Candidate warm seconds | Speedup | Peak RSS change | Relative RGB L1 |
|---|---:|---:|---:|---:|---:|
| Cycles, unlimited | 6.34 | 4.34 | 1.46× | +0.76% | 0.00000% |
| Cycles2, unlimited | 16.07 | 6.95 | 2.31× | +0.84% | 0.02175% |
| Cycles2-1, unlimited | 52.25 | 21.65 | 2.41× | +0.39% | 0.34064% |
| Cycles2-1, clamp 16384 | 53.28 | 21.80 | 2.44× | +0.07% | 0.34064% |
| Cycles2-1, clamp 4097 | 21.26 | 9.14 | 2.33× | +8.95% | 0.31038% |
| Cycles2-1, clamp 3001 (dense-cache control) | 19.36 | 19.35 | 1.00× | +1.48% | 0.00000% |

The first four rows use 3000×2000 output and 16 samples; the last two use
2593×2053 and 8 samples. The high-displacement unlimited result was independently
repeated: the earlier pair was 52.40 → 21.71 seconds (2.41×). The final installed
build was used for the 4097 pair; the earlier pairs used the same GPU algorithm
before host-side scan reuse and safety cleanup.

**The 2× target is met on both demanding scenes and the tested high-displacement
direct modes, but not on the original lighter scene (1.46×).** These are warm
render gains, not a promise of 2× including first-time shader compilation or on
every scene. The initial high-scene render with shader preparation took 78.23 →
53.13 seconds. The 4097 candidate performed fresh compilation and reached a higher
process RSS peak; repeated unlimited pairs were about +0.4%.

The output is very similar, not bit-identical. High-scene relative RGB L1 is
about 0.34%; sparse individual pixel changes can be larger (maximum absolute
linear-channel error 0.232 unlimited, 0.339 at 4097). Both high-scene previews
were visually inspected. Sample counts and the native height texture are unchanged.


Final installed-app verification after compilation reused the recorded baselines:

| Mode | Baseline warm seconds | Final app warm seconds | Speedup | Peak RSS change |
|---|---:|---:|---:|---:|
| unlimited | 52.25 | 21.50 | 2.43× | +0.05% |
| 4097 | 21.26 | 9.04 | 2.35× | +0.20% |

These additional runs retained the same image errors as the paired measurements.
The 4097 RSS increase dropped from +8.95% during fresh compilation to +0.20%
after compilation. First-render time after compilation was 22.71 seconds unlimited
and 10.10 seconds at 4097. Reports are under `high-final/` and `odd-final/`.

A resolution setting above 2048 does not itself force the current renderer into
direct evaluation: UV-domain size and the existing 8-million-sample cache budget
also matter. The 3001 setting fits the dense cache for the supplied high-displacement
scene. It is retained as a cached-mode control. The odd-sized direct-mode test uses
4097, with a 2593×2053 output image. Mode selection has not been changed.

## Validation

- Release Blender build and complete CMake app installation succeeded.
- Three native-image bounds unit tests passed, including 120,000 bilinear footprint
  samples across repeat, extend, clip, odd dimensions, and invalid inputs.
- 32 paired Metal GPU image cases passed: scale/distance/step changes, nonuniform
  object transforms, 2053×2063 textures, mode transitions, normal changes, UV edits
  and switches, missing UVs, generated coordinates, cubic/closest samplers, sphere
  projection, RGBA/alpha heights, and large coordinates. All pixels were finite;
  maximum relative RGB L1 across the fixtures was 0.3742%.
- Switching back to direct mode reproduced the previous direct result within
  2.39e-7 absolute channel error. Repeated missing-UV renders were identical.
- Full-resolution white-emission coverage comparisons found **zero new interior
  holes** in all three scenes. The original, second, and high scenes had 5,491,186,
  1,305,571, and 1,210,441 tested interior pixels respectively. A 48-pixel erosion
  excludes silhouettes and filtering differences; this is a coverage check, not
  proof that every individual ray intersection is identical.
- `git diff --check` passed. Input scene SHA-256 hashes remained unchanged.

The image fixtures are the existing scripts under `tools/`:
`pixel_displacement_direct_regression.py`, `pixel_displacement_input_cache_regression.py`,
`pixel_displacement_specialization_regression.py`, and `pixel_displacement_coverage.py`.
Full results are in `/tmp/displacement-speed-high/regressions/comparisons.json` and
its `coverage/` directory. The run orchestration and comparison scripts are retained
alongside the raw artifacts.

## Rejected experiments

The following trials were measured and removed: per-ray affine projection reuse;
a per-step gap bitmask; a finer 128-cell envelope grid; a coarser 16-cell grid;
reuse of shader state across the shading stencil; and an analytic normal-inversion
initializer. Their overhead, memory pressure, or image differences outweighed the
benefit. No changed root-search density or cached approximate height surface is
part of the candidate.

## Reproduction and artifacts

```sh
install/Blender-Speed.app/Contents/MacOS/Blender --background --factory-startup \
  /Users/michal/Documents/BlenderProjects/JAR/JAR-test-Cycles2-1.blend \
  --python-exit-code 1 --python tools/pixel_displacement_performance.py -- \
  --mode unlimited --percentage 100 --samples 16 --seed 7 --repeat 3 \
  --label candidate --output /tmp/displacement-verification
```

Use `Blender-Seams.app` for the baseline, keeping all remaining arguments identical.
Use `--mode 16384` or `--mode 4097 --width 2593 --height 2053` for the clamped direct
paths. Raw timings, EXRs, comparisons, and rejected prototypes are under
`/tmp/displacement-speed-high/`. The complete app was installed with CMake and copied
to `install/Blender-Speed.app`; the installed executable matches the build byte for
byte. Installed kernel/type headers match the source. Checksums and unchanged input
scene hashes are recorded in `/tmp/displacement-speed-high/final-manifest.json`.
