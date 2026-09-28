# Current validation, 28 September 2026

This record separates accepted results from outstanding integration work. The
full request for arbitrary coherent reflection/refraction and every Cycles
pipeline feature is not complete. Fast is the practical approximate model;
Realistic remains an experimental electromagnetic option, not the default.

## Thin Wall completed Fast model

The cache is built on Metal and accounts for both exterior-air ports. A
reciprocal first-event envelope plus a reciprocal missing-energy return replaces
the previous uncompensated sheet when the cached model is active. A continuous
blend restores the original native shader near zero relief, including dispersed
materials. Cache-dependent linked inputs and resource-limit cases explicitly
retain the documented uncached approximation. See
[model details](cycles_diffraction_thin_sheet_fast.md) and
[cache details](cycles_diffraction_thin_sheet_fast_cache.md).

The immutable v35 package is
`build/diffraction_delivery_20260928_thin_sheet_v35/Blender.app`, executable SHA256
`d28361b782ac95eafcc3eb199b4a49a7e1a0eb2383323a54612c7dec7b95041e`.
Its complete resource manifest is alongside the application.

| Check | Result | Evidence under `build/` |
|---|---|---|
| Cohesive-library actual closure, sample/evaluate, capacity and native limit | 8,836 checks, zero failures | `tests/performance/thin_sheet_completed_cohesive_v35/results.json` |
| CPU cache and symmetry | 891 checks | `tests/performance/thin_sheet_cache_symmetrized_v34/results.json` |
| Actual M5/CPU cache parity | 4/4, finite; Thin maximum directional difference 1.49e-6 | `tests/performance/thin_sheet_gpu_cache_v34/gpu_cache_comparison.json` |
| Metal unit-world furnace | 8/8; maximum RGB mean error 0.2803% | `tests/python/cycles_diffraction_thin_wall_completed_metal_v34/report.json` |
| CPU OSL front/back | 2/2 on v35; maximum mean error 0.2171% | `tests/python/cycles_diffraction_thin_wall_completed_osl_v35/report.json` |
| Metal BDPT with guiding | One rough-sheet case; maximum mean error 0.1196% | `tests/python/cycles_diffraction_thin_wall_completed_bdpt_guiding_v35/report.json` |

Furnaces use 64×64, 256 fixed samples, adaptive sampling and denoising off.
The reference is unit environment radiance, not the brightness of PT. Gates
remain 2% per RGB mean. The eight Metal images use v34; v35 fixes OSL back-face
IOR handling and a MetalRT primitive-type check. The original v34 OSL failure
is retained and is not counted as a pass.

Small-grid cache timings, after one 0.649 s compilation/initialization case,
were 1.67–3.14 ms on the GPU versus 13.45–23.08 ms on the CPU. These are
4-wavelength, 6×8-direction, 256-facet parity cases, not shipping-resolution
or whole-render speed claims. Warm 64×64 furnace times and cold shader compilation
must likewise not be mixed into a general performance ranking.

The final v36 cache run adds one actual shipping-grid case (16 wavelengths,
24 polar and 16 azimuth directions, 512 facet samples). All five cases pass.
The shipping case takes 0.817991 s on the CPU and 0.020525 s on Metal with the
pipeline already initialized, a measured 39.9× cache-construction ratio in
this case. Maximum directional difference is 2.14577e-6 and projected-average
difference is 1.04308e-7. This is not a whole-render speed ratio. Evidence:
`build/tests/performance/thin_sheet_gpu_cache_shipping_v36/gpu_cache_comparison.json`.

## Coherent surface-data passes

The v35 Metal data-pass test passed: all 68 tested data channels agree exactly
with the native control, and enabling them changes Combined by at most
4.7684e-7. It covers depth, position, normals, UV, IDs, Cryptomatte, material
colors, mist, AOVs and stored denoising data. The scene is static; motion data
support does not enable moving coherent interfaces. Light decomposition and
light groups remain rejected. The deliberate Diffuse Direct rejection leaves
Blender exit status 1; the specific error and all positive results are preserved
in `build/tests/python/cycles_coherent_data_passes_v35/report.json` and
`execution_note.json`. Fixed 32 samples, 64×64, adaptive/denoising off.

## Native-sphere coherent reflection

The v36 package is
`build/diffraction_delivery_20260928_coherent_metal_v36/Blender.app`, SHA256
`10bd061aacbe067b36b495cc560aee2b5c85bb1c54842d66a04d0e69248f54a5`.
The missing MetalRT intersection tables in ordinary coherent surface shading
are fixed, including cache identity and ON/OFF table lifetime. Default-off
pipeline behavior is preserved. Earlier black images and diagnostic count
images remain failure/debugging evidence, not accepted physical renders.

All three 256×256, 128-fixed-sample Metal BDPT physical cases pass unchanged
independent double-reference gates: phase-0 RMSE 7.5144e-5, phase-pi RMSE
7.5088e-5, distinct-source-group RMSE 3.3989e-6, phase-difference RMSE
1.5008e-4. No brightness scale is fitted. Evidence:
`build/tests/python/cycles_coherent_sphere_render_v36/results.json`.
Four unsupported-configuration checks also pass in
`build/tests/python/cycles_coherent_sphere_negatives_v36/report.json`.
ON/OFF/ON repeats within 3.73e-9 in the separate routing test.

This is one exterior reflection from an analytic native sphere, with correct
curvature spreading and compensated optical path length. General curved
refraction, multiple curved events and rough coherent transport are not
supplied by this extension. See [scope and derivation](cycles_coherent_curved_sphere.md).

The earlier v32 planar reflection/refraction images and independent-reference
tests remain valid historical evidence. All six also pass on v36 in
`build/tests/python/coherent_planar_regression_v36/results.json`: maximum RMSE
0.002111897 (wrong-face blocker), folded phase-difference RMSE 0.000832976,
and checker black pixels exactly zero. These are not proof of universal
coherent transport.

## Current v36 PT and guiding independent-reference regression

Four cases pass in one persistent Blender process, with the unchanged worker,
128 fixed samples at256×256, adaptive sampling and denoising off, and the same
0.005 mean /0.012 RMSE gates. Sphere PT and PT-guiding RMSE are7.5144e-5;
joined slab PT and PT-guiding RMSE are0.000453176. Every image is finite with
positive minimum radiance. Evidence:
`build/tests/python/coherent_sphere_slab_transport_v36/results.json`.
The original runner bytes are archived as `runner_at_run.py` and match its
recorded SHA. Cold job wall times were61.49/139.75/60.39/146.25s; these include
specialization and are not warm render benchmarks.

The final package at `build/diffraction_delivery_20260928_final_v36/Blender.app`
retains the identical v36 executable/kernel/OSL/Metal hashes. Its documented
resource delta is limited to two UI descriptions and the launcher. This v36 provenance remains historical. The v37 gallery is complete as recorded
below; its final matched benchmark is also complete. Historical v32 reports
are not relabeled current.

## Final v36 appearance and performance audit

All 13 fixed-sample Metal appearance scenes completed in
`build/tests/python/presentation_refresh_v36/`, retaining raw EXRs and editable
scenes. A disk-write failure on the first covered-disc attempt is preserved;
verified lossless archival of old executables allowed an identical-input resume.
The matched 512×369, 512-sample benchmark (adaptive/denoising off, one warmup,
three measured runs) has median 1.651403 s: 3.83% faster observed than v32 and
2.17% slower than matched historical v3. Raw RMSE against v32 is 5.187e-8.
These times include EXR writing and do not establish a universal speed ranking.
See `build/tests/python/cycles_diffraction_benchmark_default_off_v36/comparison.json`.

The independent raw-EXR audit found a real Metallic failure: 119 of 288,000
pixels have luminance below −1e-6, minimum −0.475582. The same locations and
values occur in v32. Thus the earlier blanket interpretation of negative RGB
as harmless gamut excursions was incorrect for this scene. The other twelve
scenes have no luminance below −1e-6; all thirteen are finite. Evidence:
`build/tests/python/presentation_refresh_v36_raw_luminance_audit.json` and
`presentation_refresh_v32_raw_luminance_audit.json`. v36 is not accepted as
a fully correct final material build while this defect remains unresolved.

## Metallic spectral correction — actual v37 Metallic render passed

The preserved v32/v36 Metallic appearance has119 negative-luminance pixels
(minimum linear BT709 Y−0.475582). The source defect is chromatic RGB Fresnel
multiplication after applying a signed wavelength RGB sensor basis. A narrow
Fast correction reconstructs reflective Fresnel into a scalar at the sampled
wavelength before sensor conversion; partial native coverage uses the same
response. Exact zero depth/coverage remains native; closure sizes stay96/144bytes.
The bounded CPU test passes28,665 sample/evaluation comparisons with exact
PDF/value agreement, zero failures and unchanged maximum white furnace error
0.00206083. Explicit native-zero/partial-marker checks and401-wavelength
luminance checks also pass. See
[correction details](cycles_diffraction_metallic_spectral_correction.md) and
`build/tests/performance/metallic_spectral_reflectance_final_v37/results.json`.
The corrected v37 actual Metal image passes its raw NoisyImage audit: zero
pixels with Y<−1e−6, minimum Y+0.000852877, all finite. All5019 negative-RGB
pixels have nonnegative luminance. Evidence:
`build/tests/python/metallic_spectral_correction_v37/raw_luminance_audit.json`.
The original failed images remain preserved. The v37 executable SHA256 is
`46b8fb0639dee1dcea94ae42afd3acd7e794132f6cca10b4735d1ee45caf3a20`; the complete resource manifest is beside the app.
The full13scene v37 gallery is **complete**; only the final matched benchmark
is complete (results below). Coherent
physical transport and68 data-channel results above keep their v36/v35 provenance;
the Metallic correction does not silently certify every material configuration.

## Current v37 appearance refresh

All13 cases complete with executable SHA256
`46b8fb0639dee1dcea94ae42afd3acd7e794132f6cca10b4735d1ee45caf3a20`.
The [gallery](../build/tests/python/presentation_refresh_v37/index.html) and
[actual-image overview](../build/tests/python/presentation_refresh_v37/actual_material_overview.png)
show current images; editable scenes, raw passes and denoised previews remain
separate. Sample counts are fixed:512 for11 cases,2048 forindirect,256 forcoated
furnace, adaptiveOFF. OIDN previews are appearance outputs, not numerical energy
proof. The [raw luminance audit](../build/tests/python/presentation_refresh_v37_raw_luminance_audit.json)
finds every raw image finite and zero Y<−1e−6 across the gallery. The earlier
v36 failed Metallic luminance audit and images remain preserved. This sign audit
does not certify convergence or every original full-pipeline requirement.

The older conductor regression now explicitly checks native zero-depth closure
identity/Fresnel/weights and its sampled native value/PDF rather than casting a
native closure as a grating. All63,276 sampled events pass with zero failures;
flat maximum scaled error9.565e−7, reciprocity9.417e−7. Evidence:
`build/tests/performance/conductor_native_zero_regression_v37/results.json`.
Final v37 matched benchmark is complete; see the bounded comparison below.

## Selected delivery and final matched benchmark

v37 Fast is the selected corrected delivery, with optional coherent connections
off by default. It fixes the known v32/v36 Metallic luminance defect. Realistic
remains experimental, and universal all-shader/arbitrary coherent transport is
not complete. No universal fastest claim is made.

The final M5 run used the identical scene/script hashes, 512×369 resolution,
512 fixed samples, seed 11, adaptive sampling and denoising off, one warmup
and three measured renders. Median including EXR writing: **1.777637 s**.
That is **3.53% slower than v32** (1.717108 s) and **9.98% slower than the
matched historical v3** (1.616313 s). These sequential historical observations
are not a statistically controlled causal overhead estimate or equal-error
ranking. Raw differences are recorded in
`build/tests/python/cycles_diffraction_benchmark_default_off_v37/comparison.json`.
No additional timing sweeps were used to select a favorable result.

Launch `build/diffraction_delivery_20260928_final_v37/Launch Blender.command`.
The complete current appearance suite, raw EXRs and editable scenes are linked
from `build/tests/python/presentation_refresh_v37/index.html`. The report
retains the actual versions of independent coherent and furnace references.
