# Covered grating and shadow-caustics regression

The first aluminum covered-CD presentation (`fast_index_aluminum_covered_v1`)
completed its Metal render but failed visual review: no useful diffraction bands
were visible. The saved EXR, preview and scene remain unchanged. The fixture has
an explicit 1.2 mm dielectric cover, with both grating incident medium and filled
grooves set to 1.58. It embeds the published aluminum optical table.

A paired 384 × 276, 512 fixed-sample diagnostic toggled the lights' shadow-caustics
flags. Both images remained dark (`fast_covered_mnee_before_v1`). Disabling the
object receiver/caster flags restored the spectrum
(`fast_covered_no_object_caustics_v1`). Neither test normalizes image brightness.
The initial within-session light toggles did not explicitly invalidate the
persistent light data, so the second render alone is insufficient evidence of
an effective toggle. The strengthened harness tags light data and reloads each
saved treatment before rendering. The independent object-flag control and
first-render correction comparison are unaffected by this harness limitation.
This comparison isolates an optional sampling feature on the same scene; it does
not use finite-sample PT brightness as a reference for BDPT.

The intersection kernel's receiver-ancestor flag previously included singular
reflections. A subsequent caster transmission could therefore cull the sampled
light connection. MNEE's continuous receiver evaluation cannot replace a delta
reflection or a discrete diffraction order. The correction excludes singular
reflections when establishing a receiver ancestor, preserving those sampled
paths. Diffuse and rough reflective receivers retain their existing handling.
This is a correctness fallback, not a new manifold diffraction sampler.

The benchmarked pre-fix binary and all Cycles sources are preserved in
`tests/output/diffraction/fast_index_tables_benchmarked_snapshot`.
The rebuild and install completed; the corrected binary SHA-256 is
`5dbbe90cae77c726e71b2718d2b25bd5b9df56eb965882e731d95a451f3d9b8a`.
Validation of this correction is recorded separately from the preceding
aluminum-table benchmark.

## Validation

`fast_covered_mnee_after_v1` restores the spectrum with receiver/caster flags
enabled. Against the pre-fix control with those flags disabled, the Metal
MNEE-on EXRs have relative L1 difference 3.84779e-7, maximum pixel difference
3.33787e-6, RMSE 6.64651e-8 and maximum mean RGB difference 2.50566e-8.
Both use the same saved scene, seed 11 and 512 fixed samples at 384 × 276.
This is an unnormalized, pixel-level regression comparison, not a claim that
the approximate material has exact electromagnetic order efficiencies.

The CPU diagnostic is retained under `fast_covered_mnee_after_cpu_v1`, using
256 fixed samples at 256 × 184; its within-session EXRs match exactly, but this
initial toggle check is superseded by the fresh-scene harness described above.
The full-resolution replacement presentation
is generated separately under `fast_index_aluminum_covered_v2`; its camera is
widened to retain the disc label, so it is not a pixel-matched regression image.
It completed at 960 × 691 with 4096 fixed samples, adaptive sampling and denoising
off. The 88.147 s render call includes preparation and is not a warmed benchmark.
Visual review shows the spectrum through the glass and the full disc/label in
frame; some sampling noise remains. Scene, EXR, preview and generator hashes
were verified. This presentation alone does not validate BDPT/guiding; separate
targeted controls are recorded below. Broader pipeline coverage remains open.

## Fresh-scene transport controls

`fast_covered_fresh_suite_v1` completed PT, BDPT, guided PT and CPU pairs with
source/binary identities frozen before and after each job, and checked saved
scene/image hashes. `fast_covered_mnee_bdpt_guided_v2` supplies the combined
BDPT/guiding pair. Each treatment explicitly tags the light data, saves and
reloads its scene, and verifies the requested flags before rendering.
GPU pairs use 512 fixed samples at 384 × 276; CPU uses 256 at 256 × 184.
Adaptive sampling and denoising are disabled. All GPU modes visibly retain the
covered spectrum. Maximum on/off mean-RGB differences are 2.50738e-8 (PT),
5.32212e-8 (BDPT), 1.41066e-5 (guided PT), 2.65588e-6 (BDPT with guiding), and
zero (CPU). Guided differences are recorded as sampling variation, not forced
to match PT. These are targeted transport controls, not complete convergence
or full-pipeline certification.

## Avoiding an unsupported manifold solve

The subsequent build conservatively marks a directly connected smooth
diffraction closure as delta-only. `integrate_surface_mnee` reads this static
shader flag and returns before tracing and solving a manifold, since a
continuous light connection cannot evaluate that closure. It does not inspect
unevaluated runtime closure flags. Mixed, diffuse and arbitrary OSL graphs
remain unclassified and retain their existing handling.

Build/install passed. Binary SHA-256:
`2bb04b4807ff2a32000d7fe87ee1c3e2023f23f0889ac1214547985c98fbc9fe`.
`fast_delta_mnee_skip_device_test.log` passes six table/backface/recompile checks
and verifies that diffuse and mixed replacement graphs clear the delta-only
flag. The Metal PT image against the preserved corrected build has relative L1
4.83476e-7, maximum pixel difference 2.86103e-6 and RMSE 6.95573e-8, without
normalization (`fast_delta_mnee_skip_pt_v1/pixel_comparison.json`).
`fast_delta_mnee_skip_benchmark_v1` compares the preserved corrected build with
the optimized build on the same 384 × 276 covered scene, at 1024 fixed samples.
One warm-up and five measured seeds per build ran sequentially on Metal outside
the sandbox, with matching source snapshots, adaptive sampling and denoising
disabled. Median warmed render time fell from 4.439856 s to 2.626023 s, a
40.853% reduction on this workload. All artifacts and frozen inputs passed
checks. This is scene-specific; treatment order was before then after and it
does not establish a universal speedup. The warm-ups (39.999 s and 3.165 s)
remain separately recorded and are excluded from the medians.

`fast_delta_mnee_skip_transports_v1` completed and artifact-checked the optimized
BDPT, guided PT, combined BDPT/guiding and CPU fresh-scene pairs. The BDPT
before/after pixel comparison has relative L1 2.75843e-7 and mean-RGB error
1.16393e-8; the CPU before/after pixels match exactly. Guided on/off maximum
mean-RGB differences are 5.52427e-6 (PT guiding) and 7.69773e-6 (BDPT guiding).
Both visibly retain the spectrum; these stochastic controls are not forced to
equal PT. The optimized build is selected as the current Fast candidate for
continued full-suite validation. GPU Realistic-cache construction, general
coherent interference and broader shader/pipeline coverage remain unfinished.
