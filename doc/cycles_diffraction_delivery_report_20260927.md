> Current selection: `build/diffraction_delivery_20260927_final_v3/Blender.app`.
> See [the final delivery report](cycles_diffraction_final_delivery_20260927.md) for
> Thin Wall, Ashikhmin and actual direct-source coherence results. Older sections
> below retain their historical binaries and limitations; general coherent
> multipath and Multi-GGX diffraction remain unfinished.

# Delivery report — 27 September 2026

## Selected implementation

Fast local spectral diffraction is the practical default. It avoids physical
response-cache construction and retains wavelength-dependent order directions
and approximate relief efficiencies. Realistic is optional and uses Metal/MPS
for cache construction, but remains experimental. Neither a universally fastest
implementation nor general Maxwell accuracy has been established.

The standalone Apple Silicon package is
`build/diffraction_delivery_20260927_albedo/Blender.app`.
Executable SHA-256:
`5e85be9e60de6a24b645d81040c5ab95b00f0e813fd7c0efed8e5fe7348d2dc8`.
Its adjacent resource manifest verifies the executable, kernel sources, OSL
shaders and all three precompiled Metal libraries. The installed application
has not been replaced.

## Implemented

- Dedicated Diffraction BSDF with Fast/Realistic controls.
- Glass GGX/Beckmann reflection and transmission, including film controls.
- Glossy and Metallic GGX/Beckmann, tangent/anisotropy and partial coverage.
- Principled GGX reflection and transmission, tinted/procedural Fresnel,
  colored transmission and wavelength-dependent transmission dispersion. Tinted/dispersive
  transmission and thin film are now supported together.
- Refraction GGX/Beckmann transmission-only diffraction.
- Shared SVM/CPU OSL support; Metal, PT, BDPT and guiding integration checks.
- Exact zero-relief order pruning; closure-storage checks; a GPU material
  parameter-limit fix required by the added node inputs.

The respective material integration documents give distribution, layer and
approximation limits. Unsupported enabled combinations report errors.

## Tests and scenes

The combined gallery is `tests/output/diffraction/delivery_20260927/index.html`.
It links saved Blender scenes, images and detailed results for CD/DVD discs,
covered discs, indirect diffraction, Glass, Glossy, Metallic, Principled,
Refraction, film, tint, dispersion and partial coverage. Measurement fixtures
include furnaces, angular partitions, order mixtures and inverse/Jacobian tests.
Targeted pipeline scenes cover motion blur, depth of field, volume and AOVs.

The earlier dedicated-node suite contains 72 fixed-sample Metal jobs across PT,
BDPT, guiding and combined BDPT/guiding. Its original binary provenance is retained;
it is not a fresh 72-job validation of this package. Pipeline and material smoke
tests likewise do not establish every possible Cycles combination.

The final dispersion numerical regression passed 562,390 sample/evaluate events,
370,390 reciprocity comparisons and 186,054 independent double-reference flat
comparisons. Maximum double-reference relative error was 4.65396e-6. Initial
native-float discrepancies and their independently diagnosed explanation are
retained in `cycles_diffraction_principled_dispersion.md`.

No image brightness was normalized to PT. Finite-sample PT is not treated as the
ground truth for BDPT. The retrospective in `cycles_diffraction_retrospective.md`
corrects earlier overly strong verdicts and includes an analytic case where
brighter BDPT is closer to the full physical answer. Adaptive sampling and denoising are disabled in the reported
timed rendering tests. Raw linear EXRs are retained, including noisy images.

## Measured performance, with scope

| Measurement | Result | Interpretation |
|---|---:|---|
| Exact flat-profile warm render | 6.732 → 4.739 s | 29.61% faster on that one fixture; first render after change was slower due to preparation |
| Realistic dielectric cache median | 148.52 → 124.64 s | 16.08% improvement in the isolated cache experiment, still slow |
| Same Realistic experiment total | 187.37 → 169.41 s | 9.58% improvement; does not establish arbitrary-profile performance |
| Glass diffraction on/off | approximately +60% | Bounded fixture comparison; diffraction is not generally free |

The dispersion package's per-transport results are recorded in
`tests/output/diffraction/principled_dispersion_delivery_v1/manifest.json`.
Earlier dispersion package results: PT256 **33.26 s**, BDPT128 **101.07 s**, guided128
**93.92 s**, CPU OSL32 **7.49 s**. All four jobs exited successfully, produced
finite linear pixels, and were visually inspected. Ten support-validation
assertions also passed; that deliberate-error test process exits 1.
Different sample counts make these integration timings unsuitable for ranking
PT against BDPT or guiding at equal quality.

## Not completed

General coherent interference between separate objects is not implemented.
Multi-scattering diffraction, thin-wall diffraction,
Glossy Ashikhmin diffraction and exhaustive pipeline validation remain open.
Realistic arbitrary-profile cache convergence/performance is not established;
one conductor-cache attempt timed out after 1,800 seconds. Grazing-angle float
conditioning and native albedo approximations remain documented limitations.

This is a runnable development delivery for the supported material combinations,
not a production-ready completion of the full original request. The research and
remaining transport design are in `cycles_diffraction_remaining_transport_design.md`.

## Earlier physical-film package checks

Metal PT256 33.77 s, BDPT128 104.00 s, guided128 94.94 s, CPU OSL32 8.42 s.
All four standalone jobs passed, produced finite linear pixels and were visually
inspected. Adaptive sampling and denoising are off; visible noise remains.
23,934 sample/evaluate events and the independent quarter-wave AR limit passed.
73,884 Airy-reference comparisons of the shared film kernel passed. Fourteen
node-support assertions passed, including linked film thickness; the deliberate
unsupported-shader test exits 1. See cycles_diffraction_principled_film.md and
`tests/output/diffraction/principled_film_delivery_v1/index.html`.

## Earlier generalized-film package checks

pt 256 spp: 37.30 s, bdpt 128 spp: 117.66 s, guided 128 spp: 104.90 s, osl 32 spp: 8.99 s. All four clean-environment jobs passed with finite
pixels and visually inspected output. Fixed samples, adaptive and denoising OFF.
562,862 sample/evaluate events, 370,862 reciprocity comparisons and 186,328
independent double-precision film/GGX comparisons passed. The film-free dispersion
regression was rebuilt and passed. Fourteen support assertions passed. Native
artistic film tint is bounded to passive reflectance for extreme inputs; this
can differ from unbounded native output. No closure-payload size increase.
See cycles_diffraction_principled_generalized_film.md for model and limits.

## Coherence acceptance references (not renderer support)

Six separate-source Blender scenes and an independent scalar reference suite
are now available in coherence_acceptance_scenes_v1 and coherence_reference_v1
under tests/output/diffraction. Fifteen analytic/complex-quadrature checks pass.
Two fixed-sample Metal controls show no implemented phase response (maximum
pixel difference1.19209e-7); the manifest explicitly records acceptance FAIL.
No reference patterns were inserted into materials. See cycles_coherence_acceptance.md.

## Selected bounded Fast package

Native Fast order support is capped at ±256. Valid large pitches no longer lose
their entire dielectric closure for affected wavelength/interface configurations.
Worst-case omitted passive facet power is bounded by 0.6333%; the largest of
144 tested cases is 0.1192%. This does not bound pixel error. Standard CD/DVD
order ranges are unchanged. See cycles_diffraction_fast_order_bound.md.

The isolated 100 µm fixture improves from 1.327325 to 0.711151 seconds warm
(46.42%); first-render time increases from 8.827047 to 31.830980 seconds. No
universal or cold-start speedup is claimed. 1 mm PT/BDPT/guiding/OSL checks pass,
as do 2,030 high-pitch sample/evaluate events, 1,010 reciprocity comparisons and the
562,862-event generalized-film regression. The selected package also produced a
fresh 1024-sample CD/DVD render in 41.6909 seconds, finite and visually inspected.
The current gallery is tests/output/diffraction/bounded_package_v1/index.html.

## Final denoising integration check

The selected bounded package rendered the CD/DVD scene at 1024 fixed samples on
Metal with OIDN enabled. Adaptive sampling remained OFF. The saved noisy image,
denoised image, albedo, specular albedo, normal, roughness and depth outputs are
finite; the two presentation images were visually inspected. Noise is reduced,
with some smoothing around narrow spectral bands. Render plus denoising took
51.0697 seconds. This is an integration check, not a performance comparison or
physical reference. GPU denoising was requested; the actual OIDN execution device
was not independently verified. Signed/HDR spectral RGB in the raw passes is
preserved; denoising changes pixel values and must not validate physical energy.
Evidence: `tests/output/diffraction/denoising_cd_delivery_v1/report.json`.

A separate source audit found that native diffraction dielectric closures still
fall through to the unit estimate in `bsdf_albedo`; separate generalized/tinted
reflection/transmission weights are not represented there. Consequently dielectric
denoising albedo and glossy/transmission color-pass accuracy are not certified.
The successful CD presentation check does not resolve this material-specific gap.
No unvalidated renderer changes were added to the selected package at delivery.

## Delivery artifact verification

A recursive check of the delivery gallery visited 16 HTML pages and verified
301 local targets with no missing files. All 675 packaged resource-manifest
entries match their recorded SHA-256 hashes. All ten Cycles addon Python files
also match the current source tree. Evidence is preserved in
`tests/output/diffraction/delivery_20260927/link_audit.json` and
`package_audit.json`. These are packaging checks, not additional rendering
correctness claims.

## Confirmed dielectric albedo acceptance failure

`tests/performance/cycles_diffraction_albedo_acceptance_test.cpp` now tests the
exact zero-albedo invariant for disabled reflection and transmission lobes using
the actual kernel `closure_albedo` function. Both return (1, 1, 1), rather than
(0, 0, 0); the executable exits 1. Results are retained in
`tests/output/diffraction/albedo_acceptance_v1.json` and `.log`. This confirms the
previous source-audit limitation with executable evidence. It is not a stochastic
PT-versus-BDPT discrepancy. The packaged renderer is unchanged and the failure
is not waived by the successful CD denoising smoke test.

## Disabled-lobe albedo source correction

The working source now returns zero when all requested GGX/Beckmann dielectric
diffraction lobes are disabled. The two original failures pass, together with
32 distribution/mask/query combinations (`albedo_acceptance_v2.log`). This is
an exact lobe-mask correction; active-lobe tint/Fresnel albedo estimates remain
unresolved. It does not change scattering evaluation or sampling. The earlier
bounded package predates this correction; do not attribute the source fix to
that binary.

The mask correction now has a separate packaged build, selected above. Full
CPU/OSL and all three Metal libraries build successfully. A clean-environment
Metal PT check (128×71, 32 fixed samples, adaptive/denoising OFF) produces finite
pixels and was visually inspected. First/warm times: 33.2802/0.6911 seconds,
integration timing only. Evidence: `albedo_mask_metal_v1/report.json`. Previous
CD/DVD images and broader suites retain their earlier binary provenance; they
were not rerun or relabeled. Active-lobe albedo accuracy remains unresolved.

## Corrected-package transport checks

Three sequential clean-environment checks on the selected albedo-mask package
completed successfully at 32 fixed samples and 128×71 pixels, adaptive sampling
and denoising OFF. Metal BDPT first/warm: 111.0627/3.1465 s; Metal guiding:
92.0286/1.4192 s; CPU OSL: 5.2410/5.1944 s. All pixels are finite; all three
images were visually inspected and remain noisy diagnostic previews. These
checks do not certify active-lobe albedo or establish equal-quality performance.
The manifest preserves the exact binary hash and per-run settings:
`tests/output/diffraction/albedo_mask_transport_v1/manifest.json`. All 675
packaged manifest entries were rechecked with no hash mismatches.

## Active-lobe albedo estimate update (current package)

Dielectric diffraction now estimates active reflection/transmission albedo with
the smooth-interface Fresnel/film response and the stored independent spectral
tints. Generalized Principled tint, pure Refraction, conditional ordinary-closure
normalization and separate matched-index atoms are handled. Straight closures
contribute only to transmission; coated straight atoms use their existing
quadrature mass. The main-lobe estimate adds no order loop and leaves scattering
evaluation, PDFs and closure selection weights unchanged. This supersedes the
earlier unit-estimate defect, but is not exact rough-grating hemispherical albedo:
rough masking, order redistribution and lost pure-refraction orders are omitted.

The actual kernel acceptance test passes analytic normal-incidence Fresnel,
separate tint, matched generalized reflection, pure Refraction, quarter-wave
antireflection film, straight-transmission and all 32 lobe-mask/distribution
combinations. Original failures remain preserved; latest log is
`tests/output/diffraction/albedo_acceptance_v3.log`. CPU/OSL and all three Metal
libraries build. The new package's fixed-32-sample Metal/OIDN dielectric check
exports seven finite outputs in 41.5637 seconds including preparation. The
128×71 denoised preview was inspected and is visibly smoothed. This is not a
performance or physical-reference comparison. Evidence:
`tests/output/diffraction/albedo_denoising_metal_v1/report.json`.

Earlier BDPT/guiding/OSL images and timings retain the albedo-mask package hash;
they have not been relabeled as tests of the active-lobe estimate. The full
original goal, especially cross-object coherence, remains incomplete.

## Current active-albedo package transport verification

The current package (SHA-256 5e85be9e60de6a24b645d81040c5ab95b00f0e813fd7c0efed8e5fe7348d2dc8)
now passes its own sequential BDPT, guiding and OSL integration checks at 32
fixed samples, 128×71, adaptive sampling/denoising OFF. Metal BDPT first/warm
101.6433/3.0076 s; Metal guiding 79.4360/1.3871 s; CPU OSL 5.1506/5.1071 s.
All outputs are finite and visually inspected. These noisy small previews are
not convergence proofs or equal-quality benchmarks. Exact provenance/settings:
`tests/output/diffraction/albedo_transport_v1/manifest.json`. The albedo unit test
also passes with distinct channel values in the two tint spectra
(`albedo_acceptance_v4.log`). No new renderer changes were made during these checks.

## Current-package presentation suite

Eight saved scene families were rendered sequentially with the current active-
albedo package: CD/DVD, Glass, Glossy, Metallic, transmitting Principled
tint/dispersion/film, Refraction, covered CD and indirect illumination. Every
scene uses Metal PT, 512 fixed samples, adaptive OFF, and OIDN ON. All eight
processes exit 0; all seven exported outputs per scene are finite. All eight
denoised previews were inspected. Noisy PNGs and raw EXRs remain alongside
denoised outputs and saved `.blend` files. Gallery:
`tests/output/diffraction/presentation_current_v1/index.html`.

The indirect scene still has pronounced blotchy noise and does NOT meet final
quality acceptance. Other presentation previews show denoising smoothing of fine
spectral structure; they are not converged physical references. The suite does
not prove missing coherent transport or full pipeline support. Render/denoise
elapsed times including preparation are retained only for provenance, not as a
new speed comparison: CD/DVD37.34, Glass8.50, Glossy5.10, Metallic5.74,
Principled10.72, Refraction40.94, covered73.75, indirect50.56 seconds.

## Indirect presentation follow-up

One unchanged indirect scene was rendered on the selected package with Metal
BDPT at 2048 fixed samples, adaptive OFF, OIDN ON. All seven outputs are finite.
It completed in 297.6835 s including preparation and denoising. Visual inspection
shows substantially fewer broad blotches than the earlier PT512 presentation,
but scattered artifacts remain. This is the preferred indirect preview, not a
converged physical reference. Both transport and sample count changed, so no
equal-quality speed ranking follows. No brightness normalization was performed.
Both previews are preserved in the current presentation gallery. Evidence:
`tests/output/diffraction/indirect_bdpt_presentation_v1/report.json`.
