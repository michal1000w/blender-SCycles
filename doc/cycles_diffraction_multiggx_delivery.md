# Metal diffraction development delivery: Glossy multiscattering

Current development application: `build/diffraction_delivery_20260927_multiggx_glossy_v3/Blender.app`.
Executable SHA-256: `c84fbd5e480efb0a922f9ca836ee5ddd4c44114cb88a2e78ded47db3ddf75f25`.
All 691 packaged resource hashes were verified against `resource_manifest.json`.
This is a development delivery, not completion of the original full requested scope.

## Implementation in this update

Native reflective Glossy Multi-GGX diffraction now uses a spectral directional
albedo cache built on Metal, with a CPU builder for non-Metal execution and exact-request sharing.
The reciprocal missing-energy lobe accounts for repeated color absorption;
its sampling and evaluation use the same mixture PDF. SVM and OSL receive the
same cache handle. A partial-coverage OSL bug was fixed by selecting the
native `multi_ggx` closure rather than inferring reflectance from the scaled
closure weight. Existing zero-depth native-carrier behavior is retained.
See [the implementation and numerical report](cycles_diffraction_multiggx_glossy.md).

Use constant roughness, anisotropy, pitch, depth, duty and medium IOR for this
new path. Anisotropic Multi-GGX requires a Tangent node that survives graph
optimization (the scene uses UV tangent). Other native Multi-GGX combinations
remain explicitly unsupported. Fast diffraction remains the practical mode;
Realistic remains an experimental Metal/MPS electromagnetic response solver.
The new albedo cache is separate from that much more expensive response cache.

## Validation

- Full application, CPU/OSL and all three Metal variants compiled.
- 27 white-furnace renders (nine each on CPU SVM, CPU OSL and Metal) passed,
  with 1,024 fixed samples, adaptive sampling and denoising disabled.
  Maximum object-region error from unit white was 0.005269.
- Native closure numerical tests: 21,749 accepted sample/evaluation checks,
  zero reported value/PDF discrepancy, reciprocity error at most 1.91e-6,
  independent furnace quadrature error at most 0.002061, and proposal mass
  agreement within 0.00223. Black/gray/white additional-scattering checks passed.
- Ten host validation cases passed, including six intentionally rejected linked
  inputs. Those expected render errors leave Blender's process status at 1;
  this is recorded rather than presented as an ordinary successful process.
- Representative rough Glossy scenes rendered on Metal PT and combined
  BDPT/guiding at 720×400, 512 fixed samples, without denoising or adaptive
  sampling. All pixels were finite. Preparation-inclusive times were 37.48 s
  and 180.09 s respectively; these are not warm timing comparisons.
  No PT/BDPT brightness-equivalence gate was imposed.

The representative outputs are `tests/output/diffraction/multiggx_glossy_presentation_pt_v3b`
and `tests/output/diffraction/multiggx_glossy_presentation_combined_v3`.
Each includes the editable scene, raw EXR, PNG and render report.
The earlier failed OSL partial-coverage render and constant-folded tangent scene
attempt remain preserved in their original output folders.

## Bounded performance result

The matched CD/DVD benchmark used Apple M5 (10-core GPU), Metal PT, 512×369,
512 fixed samples, seed 11, adaptive sampling OFF, denoising OFF, persistent
scene data, one warmup and three measured renders. No other render ran concurrently.
Times include EXR writing. Warm renders: 1.586700, 1.553974, 1.555040 s;
median **1.555040 s**. Cold warmup: **36.297673 s**.

The preceding flat-native build measured a 1.402086 s median on the same fixture;
this run was **10.91% slower**, not an improvement. The image RMSE was
5.20397e-8 and maximum absolute difference 3.81470e-6. This single bounded
comparison at different times does not establish a universal speed ranking.
Files and exact settings: `tests/output/diffraction/benchmark_multiggx_v3`.
The earlier outline build's 1.267193 s result is historical and predates the
native zero-depth correction.

The earlier shared Metal-kernel experiment constructed 22 batched albedo tables
and projected averages in 4.020583 ms (dispatch plus wait). That figure excludes
compilation and is not the end-to-end Blender cache-preparation time. Cold Metal
compilation remains substantial. See [the cache measurements](cycles_diffraction_multiscatter_probe.md).

## Remaining scope

General coherent reflected/refracted multipath between arbitrary objects is
not implemented. Direct scalar point-source interference is available, but the
independent mirror-phase image oracle still fails. Remaining native Multi-GGX
materials and linked cache-dependent shader inputs are unsupported. The table
approximation lacks a general strict passivity bound; Thin Wall is approximate;
arbitrary-profile Realistic convergence and exhaustive pipeline certification
are unproven. These are genuine gaps, not successful tests or completed features.

## Final gallery and export

The final binary completed all ten presentation scenes and all 70 finite-pass
checks. Nine combined Metal BDPT/guiding direct-interference cases passed their
independent reference gates; maximum absolute radiance RMSE was 0.005705561.
The presentation batch took 649.17 s and the nine isolated interference jobs
took 726.00 s including repeated process and cold-preparation costs. These
batch totals are not steady-state render times. The indirect presentation alone
took 326.40 s including preparation and denoising at 2,048 fixed samples.

`tests/output/diffraction/presentation_multiggx_v3/index.html` includes the
current material/interference renders and a separate Multi-GGX comparison.
All 21 editable scene copies were saved and reopened with sample counts,
adaptive/denoising flags, PT/BDPT/guiding settings and separate output paths
verified. All local gallery links were checked. `review.json` records their
hashes and checks. Earlier feature-control images are explicitly historical.

The reproducible native regression runner also passed Ashikhmin, Thin Wall,
albedo construction and Multi-GGX checks on the final libraries; full commands
and outputs are in `build/tests/performance/native_extensions_multiggx_v3/results.json`.

Fast is the recommended practical quality setting in this feature build. It
is not a certified fastest fully correct implementation of the original scope:
no candidate satisfies that entire scope, and the measured CD/DVD fixture was
faster on the narrower preceding flat-native build.
