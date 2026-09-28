# Cycles diffraction: Metal development delivery

Latest validated incremental development package: `build/diffraction_delivery_20260928_coherent_exact_reflection_v32/Blender.app`.
This separate Apple Silicon app does not replace installed Blender. Its binary
SHA-256 is `c71a2433f88353dbae8238934c9a8eb7008793899c144c68fa8a614315eedc91`.
A [current 13-scene appearance gallery](../build/tests/python/presentation_refresh_v32/index.html)
and [actual-render overview](../build/tests/python/presentation_refresh_v32/actual_material_overview_v32.png)
now use this exact v32 binary. They retain the existing scene designs and original
fixed sample counts, save raw passes, OIDN previews and editable copies, and provide
appearance evidence rather than exhaustive material/pipeline certification.
The historical v3 gallery and benchmark below retain their original provenance.
See the [Metallic follow-up](cycles_diffraction_multiggx_metallic.md) and
[fully metallic Principled follow-up](cycles_diffraction_multiggx_principled.md)
for their bounded acceptance results. Earlier v12/v19/v21 coherent and
v23–v29 material evidence remains version-labeled. The v29 app remains the
preceding validated material package; v30 retains its wider colored-detector
controls. Current v32 additionally passed the bounded BDPT and PT/guiding
face/slab gates documented below.

In Blender Preferences → System → Cycles Render Devices, select **Metal**
and enable the Apple GPU. The supplied scenes use GPU Compute.

Use **Fast** on the dedicated Diffraction BSDF for practical CD/DVD materials.
Pitch and depth are in nanometers; orient the tangent across the grooves.
**Realistic** is optional and experimental: its Metal/MPS response cache remains
slow, and arbitrary-profile convergence has not been certified.

This is the current feature-development build, not a completed production-ready
implementation of the original full scope. No candidate has passed the entire
requested scope. The optional coherent specular connector covers a declared,
bounded class of ideal planar Mirror and Glass paths; arbitrary coherent
reflected/refracted multipath is unfinished.
The preceding flat-native package remains available as the narrower fallback;
it measured faster than v3 in that earlier matched CD/DVD comparison. The v19
warm run below is a separate measurement, not a cross-package speed ranking.

## What is implemented

| Node | Diffraction support |
|---|---|
| Dedicated Diffraction BSDF | Fast spectral grating; optional experimental Realistic response cache |
| Glossy | GGX, Beckmann, Ashikhmin; newly added reflective Multi-GGX for constant cache-dependent inputs |
| Metallic | GGX/Beckmann conductor/F82 response, film and partial coverage; constant-input Multi-GGX in v5/v6 |
| Glass | GGX/Beckmann reflection/transmission and thin film; two-sided constant-cache-input Multi-GGX with local tint and lossless real film in v23–v26 |
| Principled | GGX reflection/transmission, tint, dispersion, film, supported layers and coverage; approximate Thin Wall; mixed reflective/solid transmission Multi-GGX in v27; bare and coated generalized Fresnel plus dispersion with constant cache inputs in v28/v29 |
| Refraction | GGX/Beckmann transmission |

The v23 Glass Multi-GGX diffraction extension adds a neutral, lossless,
uncoated two-sided multiscatter return to the Glass closure. v24 adds bounded
local RGB tint to the first event and return lobe, including a linked Color
input. The cache-dependent roughness, IOR and grating inputs remain constant.
Tint is one tint per completed return, not per-bounce absorption. v25 adds a
lossless film with real IOR and constant physical thickness/IOR; the spectral
cache resolution follows its shortest Airy period and explicitly rejects the
256-node budget overflow. v26 builds these two-sided caches on the selected
Metal device, with an explicit CPU path for non-Metal devices. Metal cache
failure reports an error instead of silently falling back to CPU. A general
passivity proof remains outside these bounded furnace gates.

Known zero depth and zero coverage preserve the native material branch.
Unsupported enabled combinations report errors. The new Glossy Multi-GGX
cache requires constant roughness, anisotropy, pitch, depth, duty and medium
IOR. Anisotropic Multi-GGX requires an explicit tangent, including constant-folded inputs.
The Metal-built cache is shared by matching material requests and used by both
SVM and OSL. Its reciprocal extra lobe includes repeated color absorption.
See [implementation and tests](cycles_diffraction_multiggx_glossy.md).

Optional [direct scalar point-source interference](cycles_coherent_direct.md)
has emitter group, phase and coherence-length controls. It is not general
coherent wave transport through arbitrary reflected or refracted paths.

The new opt-in **Coherent Specular Connections** mode declares ideal planar
Mirror or Glass interface objects and a passive Lambertian detector. V30 supports
evaluated color on Diffuse or a pure diffuse Principled shader, including exact
black texture regions. Its scalar
mirror mode and vector Glass mode use point-source groups, phase and coherence
length, with a bounded interface-event and candidate budget. The vector source
model is an ensemble of three independent world-axis dipoles with Jones
reflection and transmission. The mode is off by default and rejects unsupported
materials, scene settings, and candidate overflow explicitly. It is a separate
path model; arbitrary BSDF graphs do not acquire an inferred optical phase.
Detector RGB is a local passive spectral-albedo approximation, not an optical
phase or a polarization-sensitive receiver model. Detectors require flat
geometric normals, zero shading terminator offset, full opacity and no other
active closure layers; effective Principled thin film is rejected.

## Results and scenes

The v3 broad-suite gallery is
`tests/output/diffraction/presentation_multiggx_v3/index.html`.
It includes ten material families, nine direct-interference cases, and the new
Glossy Multi-GGX PT/BDPT-guiding comparison. Editable copies live under
`scenes/`, with separate output destinations so new renders preserve evidence.
The gallery links raw EXRs, noisy previews, denoised presentations and settings.
Any earlier-build controls are explicitly labeled historical.

Read [the delivery and performance report](cycles_diffraction_multiggx_delivery.md).
The 27 CPU/OSL/Metal furnace tests passed with maximum region energy error
0.005269. Independent closure tests check reciprocity, energy, sample/evaluate
agreement, proposal mass and color absorption. PT and BDPT brightness were not
forced to match.

The Apple M5 Metal CD/DVD fixture used 512 fixed samples at 512×369 with
adaptive sampling and denoising OFF. Warm median: **1.555040 seconds**; cold
warmup: **36.297673 seconds**. The preceding flat-native build measured
**1.402086 seconds** on the same fixture: the new measurement is 10.91% slower.
These are bounded runs, not a universal speed ranking or a zero-overhead claim.

The planar mirror fixed-reference Metal BDPT gate passed phase 0, phase π, and
connector-on distinct-group incoherent control at 512×512 and 128 fixed samples:
[report](../build/tests/python/cycles_coherent_mirror_render_v12/results.json) and
[fixed-display contact sheet](../build/tests/python/cycles_coherent_mirror_render_v12/display_fixed_ev2_srgb/mirror_v12_contact_sheet_ev2_srgb.png).
For the two-transmission ideal Glass slab, v19 passed the independent Snell/Jones
reference at 512×512 and 512 fixed samples on Metal BDPT: phase-0 RMSE
0.000976234, phase-π RMSE 0.000976215, phase-difference RMSE 0.001952439, and
distinct-group control RMSE 0.000003222. The frozen absolute gates were 0.005
for mean radiance error and 0.012 for image and phase-difference RMSE, without a
fitted scale. See the [full slab report](../build/tests/python/cycles_coherent_slab_acceptance_render_v19/results.json)
and [common-exposure contact sheet](../build/tests/python/cycles_coherent_slab_acceptance_render_v19/display_fixed_ev0_srgb/slab_v19_contact_sheet_ev0_srgb.png).
Targeted Metal PT and PT with guiding phase-0 checks at the same 512 fixed
samples also matched this reference (RMSE 0.000976235 and 0.000976234):
[PT](../build/tests/python/cycles_coherent_slab_pt_v19_phase0/report.json),
[PT with guiding](../build/tests/python/cycles_coherent_slab_pt_guiding_v19_phase0/report.json).

The v21 mixed-path gate adds one interior ideal Mirror to a Glass slab, testing
both direct T/T and three-event T/R/T arms. Its independent reference uses the
exact planar centers saved as Blender floats and includes all six unordered
pairs of the four fields. On Metal BDPT at 512×512 with **128 fixed samples**,
phase 0, phase π, distinct source groups, and tiny-coherence diagonal control
all passed the unchanged absolute gates: mean error ≤0.005, per-image RMSE
≤0.012, and phase-difference RMSE ≤0.012. The measured phase-0/π RMSEs were
0.003577/0.003562; phase-difference RMSE was 0.003818; control RMSEs were
0.003016/0.003003. All predeclared ROI pixels were finite and nonnegative.
See the [four-case report](../build/tests/python/cycles_coherent_mixed_rt_acceptance_render_v21_128/results.json),
[actual-render contact sheet](../build/tests/python/cycles_coherent_mixed_rt_acceptance_render_v21_128/display_fixed_ev-2_srgb/mixed_v21_actual_128spp_ev-2_srgb.png), and
[separately labeled independent-reference sheet](../build/tests/python/cycles_coherent_mixed_rt_acceptance_render_v21_128/display_fixed_ev-2_srgb/mixed_v5_independent_reference_ev-2_srgb.png).
Both images use one fixed −2 EV sRGB transform, with no per-image scaling.
The [row=column seam audit](../build/tests/python/cycles_coherent_mixed_rt_acceptance_render_v21_128/diagonal_seam_audit.json)
found no remaining concentrated detector triangle-edge residual: v21 diagonal
error RMSE 0.003510 versus immediate neighbors 0.003800/0.003703. A targeted
[Metal PT with guiding phase-0 check](../build/tests/python/cycles_coherent_mixed_rt_pt_guiding_v21_phase0_128/report.json)
also matched the same independent reference at 128 fixed samples (RMSE
0.003577). The four mixed BDPT renders took 13.394, 13.296, 13.302, and
13.027 seconds respectively, including EXR save, with a previously warmed
Metal specialization. These are bounded measurements of this fixture.

The earlier v20 mixed [8-sample pilot](../build/tests/python/cycles_coherent_mixed_rt_phase0_pilot_v20_512x8/report.json)
recovered the T/R/T arm but had [RMSE 0.01414 against the corrected reference](../build/tests/python/cycles_coherent_mixed_rt_phase0_pilot_v20_512x8/v5_reference_recheck.json)
and a detector triangle-edge artifact. Its planned 512-sample gate was
[deliberately cancelled](../build/tests/python/cycles_coherent_mixed_rt_acceptance_render_v20/cancellation.json)
after 393 seconds in the first case, with no accepted result. These failure and
cost records are preserved; v21's conservative candidate pruning and shared-edge
handling resolve the bounded test without altering its physical reference.

For the neutral two-sided Glass Multi-GGX extension, the v23 Metal SVM
white-environment furnace used an off-normal camera from the front and back,
64×64 pixels, 256 fixed samples, adaptive sampling and denoising off. The
new two-sided front/back RGB central-ROI means were
(1.00310, 1.00604, 1.00310) and (1.00236, 1.00561, 0.99776). Both passed the
predeclared ±0.03 per-channel unit-energy and front/back-difference gates;
native Multi-GGX front/back controls measured approximately
(1.00019, 1.00019, 1.00019) and (1.00114, 1.00114, 1.00114).
Every furnace EXR was finite and nonnegative. The independent
[Metal furnace report](../build/tests/python/cycles_diffraction_glass_multiggx_furnace_render_v23/report.json)
preserves all six raw EXRs, including single-event controls, and the
[fixed-scale contact sheet](../build/tests/python/cycles_diffraction_glass_multiggx_furnace_contact_v23_final/glass_multiggx_furnace_fixed_ev-1.png)
shows their actual output. CPU OSL front/back renders passed the same gates
with means matching Metal to approximately 10⁻⁷:
[OSL furnace report](../build/tests/python/cycles_diffraction_glass_multiggx_furnace_osl_render_v23/report.json).
CPU SVM and OSL material validation each passed 14 cases:
[SVM](../build/tests/python/cycles_diffraction_glass_multiggx_validation_v22_svm/report.json),
[OSL](../build/tests/python/cycles_diffraction_glass_multiggx_validation_v22_osl/report.json).
The integrated new closure test passed 16,413 checks after rebuilding all
dependent libraries. A v22 preliminary Metal furnace failed because its SVM
branch did not use the new cache; its [failure report](../build/tests/python/cycles_diffraction_glass_multiggx_furnace_render_v22/report.json)
is retained rather than relabeled as a v23 result.

The v24 uncoated chromatic Glass furnace used the same 64×64, 256 fixed-sample
Metal setup with Color (0.35, 0.65, 0.9). The two-sided front/back RGB means
were (0.35192, 0.65388, 0.90256) and (0.35136, 0.65361, 0.89762).
All channels were finite and nonnegative, below the predeclared passive upper
bound of 1.03, visibly chromatic, and within 0.08 front-to-back. The back-side
mean return gain over the single-event control was (0.02920, 0.05434, 0.07278),
passing the predeclared 0.02 mean gain gate. This checks a bounded passive
response and an added return; it does **not** establish that output RGB must
equal the input Color or that tint accumulates per bounce. See the
[Metal tint report](../build/tests/python/cycles_diffraction_glass_multiggx_tint_furnace_render_v24/report.json)
and [fixed-scale contact sheet](../build/tests/python/cycles_diffraction_glass_multiggx_tint_furnace_contact_v24_final/glass_multiggx_furnace_fixed_ev-1.png).
The unchanged neutral six-scene furnace also passed on v24:
[regression report](../build/tests/python/cycles_diffraction_glass_multiggx_furnace_render_v24_neutral_regression/report.json).
CPU SVM and OSL each passed 16 material cases, including spatial Noise Color
linked to Glass Color, with all successful 8×8 EXRs finite:
[SVM validation](../build/tests/python/cycles_diffraction_glass_multiggx_validation_v24_svm_exr/report.json)
and [OSL validation](../build/tests/python/cycles_diffraction_glass_multiggx_validation_v24_osl_exr/report.json).

The v25 lossless-film furnace used a real film IOR of 1.32 and 250 nm physical
thickness, with white Glass at 64×64 and 256 fixed Metal samples. Front/back
two-sided RGB means were (1.00317, 1.00467, 1.00351) and
(1.00238, 1.00510, 0.99665), passing the same ±0.03 unit-energy/front-back
gates. Its back-side return gain over the first-event control was about 0.08
per channel. See the [film furnace report](../build/tests/python/cycles_diffraction_glass_multiggx_film_furnace_render_v25/report.json)
and [fixed-scale contact sheet](../build/tests/python/cycles_diffraction_glass_multiggx_film_furnace_contact_v25/glass_multiggx_furnace_fixed_ev-1.png).
CPU SVM/OSL each passed six film preparation/routing cases with finite EXRs
for successful renders and exact expected rejection messages; Blender returned 1 because the
last render was an expected rejection, so these are report-level passes:
[SVM](../build/tests/python/cycles_diffraction_glass_multiggx_film_validation_v25_svm/report.json),
[OSL](../build/tests/python/cycles_diffraction_glass_multiggx_film_validation_v25_osl/report.json).
The [independent film-cache test](../build/tests/performance/two_sided_film_cache_v25/results.json)
checks dense wavelengths and grazing directions; the integrated coated closure
test passed 16,466 checks.

The v26 CPU-versus-M5 Metal cache test used the identical 512-facet-sample
sequence on both physical sides. At 16 uncoated wavelengths, maximum
directional R/T/deficit difference was 3.58×10⁻⁷; at 40 coated wavelengths it
was 1.47×10⁻⁵. Integral differences were at most 1.91×10⁻⁶, and the shared film
cross fractions were identical. CPU cache construction took 0.388/1.459 s;
Metal warm construction took 0.00939/0.02057 s. The first Metal cache took
0.465 s including 0.445 s of runtime library/pipeline preparation; measured
command commit-and-wait times were 0.00927/0.02002 s for the warm requests.
These are bounded cache timings, not a render-speed claim:
[table comparison](../build/tests/performance/v26_two_sided_gpu_cache_comparison.json)
and [timing log](../build/tests/performance/v26_two_sided_gpu_cache_comparison.log).
The immutable v26 app reran the unchanged fixed-sample Metal
[neutral](../build/tests/python/cycles_diffraction_glass_multiggx_furnace_render_v26/report.json),
[tint](../build/tests/python/cycles_diffraction_glass_multiggx_tint_furnace_render_v26/report.json),
and [film](../build/tests/python/cycles_diffraction_glass_multiggx_film_furnace_render_v26/report.json)
furnaces through its GPU cache path; all gates passed without fitting or
sample escalation. CPU OSL remains on the explicit CPU cache path.

The v27 Principled extension reuses the reflective and two-sided dielectric
returns for mixed metallic and solid transmission weights. The fresh integrated
closure test passed 18,873 checks, including coated and partial-coverage
all-layer allocation within the actual 22-slot shader budget. Its neutral mixed
closure energy was 0.998633; the old Glossy/Metallic sample/evaluate regression
also passed 28,665 comparisons. CPU
[SVM](../build/tests/python/cycles_diffraction_principled_multiggx_validation_v27_svm/report.json)
and [OSL](../build/tests/python/cycles_diffraction_principled_multiggx_validation_v27_osl/report.json)
each passed 21 preparation/routing cases with finite EXRs for every successful
render, including linked Base Color and runtime layer weights. The fixed
64×64/256-sample Metal mixed furnace (metallic 0.3, transmission 0.4) passed the
unchanged unit-environment per-RGB and front/back 0.03 gates; its maximum mean
energy error was 0.003960. See the
[raw furnace report](../build/tests/python/cycles_diffraction_principled_multiggx_furnace_render_v27/report.json).
One targeted [Metal BDPT with guiding case](../build/tests/python/cycles_diffraction_principled_multiggx_furnace_bdpt_guiding_v27/report.json)
also passed the independent unit-radiance gate at 256 fixed samples (maximum
RGB mean error 0.005737); it does not establish all BDPT or guiding combinations.
The [v27 Glass presentation](../build/tests/python/cycles_diffraction_glass_multiggx_presentation_v27/glass.png)
shows neutral, locally tinted, and coated rough Multi-GGX spheres under shared
lighting. The [editable scene](../build/tests/python/cycles_diffraction_glass_multiggx_presentation_v27/glass.blend),
[raw EXR](../build/tests/python/cycles_diffraction_glass_multiggx_presentation_v27/glass.exr),
and [report](../build/tests/python/cycles_diffraction_glass_multiggx_presentation_v27/report.json)
are from the current v27 binary: 540×300, 128 fixed Metal PT samples, adaptive
sampling and denoising off, all pixels finite. The visible indirect noise remains;
this is an appearance/integration scene rather than a convergence proof.
A separate [denoised presentation](../build/tests/python/cycles_diffraction_glass_multiggx_presentation_v27_denoised_512/glass.png)
uses 512 fixed samples with adaptive sampling off. Its
[multilayer EXR](../build/tests/python/cycles_diffraction_glass_multiggx_presentation_v27_denoised_512/glass.exr)
retains Noisy Image and denoising data; all nine stored image/pass subimages
were checked finite. The [editable scene](../build/tests/python/cycles_diffraction_glass_multiggx_presentation_v27_denoised_512/glass.blend)
and [report](../build/tests/python/cycles_diffraction_glass_multiggx_presentation_v27_denoised_512/report.json)
label it as a visual presentation. Denoising softens the small spectral details;
it was not used for any numerical energy or interference gate.

The v27 transmission stage required white Specular Tint, zero dispersion, and
a solid interface; the v28 bare extension below relaxes its tint and dispersion
limits. Cache-dependent roughness, IOR, grating and film inputs remain constant;
linked Color and layer weights are allowed. These bounded tests do
not establish every Principled material or pipeline combination.

The v28 bare-interface Principled extension permits colored or linked Specular
Tint together with native Cauchy transmission dispersion. Constant dispersion,
IOR and grating inputs select two scalar Fresnel endpoint tables; runtime RGB
Fresnel channels interpolate their return. Coated generalized transmission was outside the v28 stage; its v29 result is
reported separately below. The fresh integrated generalized test
passed 172,717 checks (maximum energy deviation 0.006007), and actual Glass
seven-slot / all-layer Principled 22-slot graph allocation passed 18,909 checks:
[focused results](../build/tests/performance/generalized_final_precise_v28/results.json).
CPU [SVM](../build/tests/python/cycles_diffraction_principled_multiggx_validation_v28_svm/report.json)
and [OSL](../build/tests/python/cycles_diffraction_principled_multiggx_validation_v28_osl/report.json)
each passed all 25 routing/validation cases, including 20 finite successful EXRs.

On the immutable v28 app, the bare generalized dispersive unit-world Metal PT
furnace (Specular Tint RGB 0.3/0.7/1, Abbe 20) passed the unchanged 0.03 RGB
mean and front/back gates at 64×64 with 256 fixed samples. Maximum new-model
mean error was 0.002858; one Metal BDPT plus guiding case passed the same
independent unit-energy reference with maximum error 0.001658. Adaptive
sampling and denoising were off:
[PT furnace](../build/tests/python/cycles_diffraction_principled_multiggx_generalized_furnace_render_v28/report.json),
[BDPT with guiding](../build/tests/python/cycles_diffraction_principled_multiggx_generalized_bdpt_guiding_v28/report.json).

The v28 [CPU/M5 cache comparison](../build/tests/performance/v28_two_sided_gpu_cache_comparison_final.json)
passed all five predeclared physical, coated, generalized dispersive, matched,
and lower-index cases with identical sample sequences. Maximum directional
error was 0.000599 and integral error 0.00003034; cross fractions were identical.
The exact matched-index case required a shared division/unit-conversion and
rounding correction: earlier failed reports remain saved. An
[actual M5 helper probe](../build/tests/performance/dielectric_dispersion_metal_v28/results.json)
verified CPU/Metal bit agreement in safe and fast modes, including the original
failing inputs. This avoids silently changing the straight-transmission atom.

The [v28 dispersive presentation](../build/tests/python/cycles_diffraction_principled_dispersion_presentation_v28_denoised_512/principled_dispersion.png)
uses the existing three-sphere lighting fixture with native Cauchy dispersion
(Abbe 10), generalized Fresnel tint, and rough Multi-GGX grating transport.
It was rendered with Metal BDPT at 540×300 and 512 fixed samples, adaptive
sampling off and denoising on; the soft spectral highlights and caustic patches
come from transport rather than a display color effect. Denoising softens fine
details. The [editable scene](../build/tests/python/cycles_diffraction_principled_dispersion_presentation_v28_denoised_512/principled_dispersion.blend),
[raw multilayer EXR](../build/tests/python/cycles_diffraction_principled_dispersion_presentation_v28_denoised_512/principled_dispersion.exr),
and [report](../build/tests/python/cycles_diffraction_principled_dispersion_presentation_v28_denoised_512/report.json)
preserve the noisy image and denoising passes; all nine stored subimages were
checked finite. This is a denoised visual presentation, not a numerical gate.

The v29 Principled extension adds coated generalized Fresnel transmission,
including dispersion and linked Specular Tint. Its sixteen shared scalar F0
knots concentrate around the coated response changes, with a bounded high-IOR
cluster so the nodes remain strictly ordered. Native film thickness ≤0.1 nm is
canonicalized to the bare branch; near matched indices the cache follows the
native physical film and straight-transmission atom. Cache-dependent IOR,
dispersion, roughness, grating, film thickness and film IOR remain constant.
This is an approximate multiscatter return, with one local tint per completed
return rather than a per-bounce absorption model.

The fresh [coated integrated test](../build/tests/performance/coated_generalized_fresh_v29/results.json)
passed 173,737 checks, including sample/evaluate agreement, reciprocity,
off-grid wavelength integrals, capacity and matched-film cases. Maximum full
energy deviation was 0.008402. CPU
[SVM](../build/tests/python/cycles_diffraction_principled_multiggx_validation_v29_svm/report.json)
and [OSL](../build/tests/python/cycles_diffraction_principled_multiggx_validation_v29_osl/report.json)
each passed all 32 preparation/routing cases with 28 finite successful EXRs,
including IOR 3.1/10, cutoff boundaries and linked coated Specular Tint.
These small renders are routing checks, not high-IOR energy certification.

The immutable v29 coated generalized Metal furnace uses white Base Color,
Specular Tint RGB 0.3/0.7/1, Abbe 20, film IOR 1.32 and thickness 250 nm.
At 64×64 with 256 fixed samples, the front/back PT cases passed the unchanged
unit-environment RGB and front/back 0.03 gates (maximum new-model mean error
0.004655). One BDPT plus guiding case passed the same independent energy
reference with maximum error 0.005101; adaptive sampling and denoising were off:
[PT report](../build/tests/python/cycles_diffraction_principled_multiggx_coated_furnace_render_v29/report.json),
[BDPT with guiding](../build/tests/python/cycles_diffraction_principled_multiggx_coated_bdpt_guiding_v29/report.json).
The independently re-read EXRs are shown with one fixed −1 EV transform in the
[v29 furnace contact sheet](../build/tests/python/cycles_diffraction_principled_multiggx_coated_furnace_contact_v29/glass_multiggx_furnace_fixed_ev-1.png).

The [v29 CPU/M5 table comparison](../build/tests/performance/v29_two_sided_gpu_cache_comparison.json)
passed all seven unchanged parity gates, including coated sixteen-knot and
matched-film requests. Their maximum directional errors were 0.001940 and
0.001342; integral errors were 0.0001513 and 0.00002134; shared cross fractions
were identical. For these two bounded 16-wavelength requests, CPU preparation
took 11.408/12.353 s and warm Metal API preparation took 0.2387/0.2567 s.
The [profiling log](../build/tests/performance/v29_two_sided_gpu_cache_comparison.log)
records actual command commit-and-wait time separately. These are cache
measurements, not a general render-speed ranking.

The v19 default-off Metal warm benchmark used the unchanged v3 scene and
benchmark script, 512×369 at 512 fixed samples with adaptive sampling and
denoising off. Its three-render median was **1.343131 seconds**, compared with
**1.616313 seconds** for the earlier v3 run; this is one bounded warm comparison,
not a universal speed claim. The saved scene confirms coherent connections off.
Pixel arrays differed by at most 3.82×10⁻⁶ in absolute radiance. See the
[v19 benchmark report](../build/tests/python/cycles_diffraction_benchmark_default_off_v19/report.json).

The v30 receiver increment passed [nine independent RGB acceptance renders](../build/tests/python/cycles_coherent_detector_color_results_v30_final/report.json):
Diffuse and pure diffuse Principled checker receivers each have phase-zero,
phase-pi and distinct-group controls, plus one constant RGB phase-zero case
for each shader and one Principled checker PT plus guiding case. All use
512×512, 128 fixed samples, adaptive sampling off and denoising off. The
reference multiplies the independent mirror oracle by known RGB 0.25/0.5/0.75
or a world-space checker; a fixed 0.003 m edge band excludes texture boundaries.
The unchanged per-channel gates are mean error ≤0.005 and RMSE ≤0.012, with
the same phase-difference gates. Worst RGB RMSE was 0.00001834 and mean error
0.000000110. Each checker has 64,800 black ROI pixels, all exactly zero; every
tested ROI was finite and nonnegative. This does not validate arbitrary
non-Lambertian receivers or the earlier complete material gallery on v30.
The [six actual checker BDPT images](../build/tests/python/cycles_coherent_detector_contact_v30_final/actual_checker_fixed_ev+3.png)
share one +3 EV sRGB transform, with denoising off. Their
[independent raw-image verification](../build/tests/python/cycles_coherent_detector_contact_v30_final/raw_metric_verification.json)
matches worker metrics within 10⁻⁹; this contact sheet contains renderer output,
without reference images substituted for any panel.

The [focused kernel test](../build/tests/performance/coherent_detector_fresh_v30_final/results.json)
passed 63 checks. The [three actual host rejection checks](../build/tests/python/cycles_coherent_detector_color_acceptance_v30_enabled/detector_negative_validation.json)
passed for effective film, smooth normals and nonzero shading terminator offset;
their process exit 1 reflects Blender's retained expected render-error state.
All fifteen saved positive/negative scenes were reopened and
[checked for mode ON and fixed settings](../build/tests/python/cycles_coherent_detector_color_acceptance_v30_enabled/saved_scene_settings_assertion.json)
before the corrected GPU run. The initial generator omitted mode ON; its
[failed negative report](../build/tests/python/cycles_coherent_detector_color_acceptance_v30_final/detector_negative_validation.json)
and log remain preserved as a fixture failure, not a renderer validation result.

Actual render-operator times include initialization: first Diffuse checker
158.708 s, first Principled checker 151.532 s, constant Diffuse 150.215 s and
PT plus guiding 118.667 s incurred cold Metal specialization. The five other
cases took 1.936–2.965 s using cached kernels. These are observed costs, not a
general speed benchmark; no extra timing or presentation renders were added.

The v32 planar face increment passed [six unchanged independent-reference renderer gates](../build/tests/python/cycles_coherent_face_cluster_results_v32/report.json):
one folded object with two mirror faces at phases zero/pi, a connected mirror ring
with a real hole, an intervening wrong face in the same mirror object, a joined
Glass slab, and one existing colored Diffuse checker regression. All use Metal
BDPT, 128 fixed samples, adaptive sampling off and denoising off. Face cases use
256×256; the historical checker remains 512×512. The folded oracle independently
unfolds every finite visible direct/one/two-reflection family; the slab retains
the independent Snell/Jones reference. No reference scale or numerical gate was fitted.

Connected coplanar triangles now form one optical patch, while different faces
in one object remain different patches. Exact object-qualified primitive sets
validate each segment against the actual finite triangle union, including the
hole and wrong-face occluder. All-reflection planar paths use an exact image-source
solution; mixed reflection/transmission retains the existing numerical solver.
[Thirty host grouping checks](../build/tests/performance/coherent_planar_clusters_v31/results.json),
[18 membership checks](../build/tests/performance/coherent_patch_membership_prebuild_v31/results.json),
and [144 independent strict/fast reflection checks](../build/tests/performance/coherent_corner_dropout_v31b/after_fix_results.json)
cover this bounded increment; the latter also verifies 28 valid corner paths and
28 wrong-order rejections. The six face/checker renders establish current BDPT evidence. Four additional
current PT/plus-guiding face/slab checks are identified separately below; they
do not certify every receiver or pipeline combination.

Folded phase-zero/pi RMSE is 0.00041711/0.00041693, and phase-difference RMSE is
0.00083298. Hole and wrong-face RMSE is 0.00069455/0.00211190; joined slab RMSE is
0.00045318. Checker worst RGB RMSE is 0.00001301, with all 64,800 black ROI pixels
exactly zero. Every ROI is finite and nonnegative. The unchanged gates remain
mean error ≤0.005, RMSE ≤0.012 and the same phase gates; minimum radiance permits
only −10⁻⁶ floating roundoff. Historical v31 separate/joined slab renders also passed,
with phase-zero arrays differing by at most 2.38×10⁻⁷.

The [six actual renderer images](../build/tests/python/cycles_coherent_face_cluster_contact_v32/actual_planar_faces_fixed_exposure.png)
use fixed −2 EV for the face families and fixed +3 EV for the checker, explicitly
labeled and converted to sRGB without per-panel normalization. The
[independent raw-image verification](../build/tests/python/cycles_coherent_face_cluster_contact_v32/raw_metric_verification.json)
matches every worker within 10⁻⁸. References are separate from these image panels.

The folded gate isolates the first Lambertian receiver using diffuse_bounces=0;
[12 actual native path-state checks](../build/tests/performance/coherent_terminal_detector_v31b/results.json)
verify terminal detector evaluation before diffuse continuation ends. Sources,
triangles and reference arrays are unchanged from the earlier fixture. The
[preserved diffuse-one v4 render](../build/tests/python/cycles_coherent_face_cluster_folded_render_v31/results.json)
includes legitimate receiver–wall–receiver illumination omitted by the first-receiver
oracle, so it is unmatched full-transport evidence rather than a proven renderer error.
The [isolated v31b failure](../build/tests/python/cycles_coherent_face_cluster_folded_first_receiver_render_v31b/results.json)
instead exposed real numerical dropout of valid corner reflection paths
(RMSE 0.02054/0.02043), fixed by the exact reflection branch. A 20 µm source baseline
was selected before rendering to resolve the fringes at 256 pixels. Finite Lc=10⁻⁴ m
suppresses between-family cross terms; this validates geometry and within-family
source phase, not a new long-coherence cross-family renderer case.

The initial v32 supervisor exited 120 without an image or worker report, followed
by a confirmed ENOSPC write failure. Its [terminal unmeasured record](../build/tests/python/cycles_coherent_face_cluster_folded_first_receiver_render_v32/results.json)
is preserved; the unchanged binary/reference retry passed after lossless archival.
Historical task-owned executable archives retain verified decompressed SHA-256,
byte counts and restoration commands in adjacent executable_archive.json files.
Scenes and reports remain available; current v32, v30_final and v29 apps remain executable.

Successful folded retry and hole operator times including initialization were
1.675/1.328/1.351 s using cached kernels. Wrong face, slab and checker incurred
132.629/136.624/152.992 s cold specialization. The preceding failed v31b warm phase-pi
case took 3.825 s: this is one bounded fixture observation, not a fastest-package
or universal render-speed claim. The initial disk-failed v32 cold attempt has no
usable render timing. No extra benchmark or quality render was added.

## Current v32 appearance and pipeline refresh

The [four PT checks](../build/tests/python/cycles_coherent_face_cluster_pt_extension_v32/report.json)
passed the same independent references and absolute gates: folded mirror phase zero
and joined Glass slab phase zero, each with PT and PT plus guiding. All use
256×256, 128 fixed samples, adaptive sampling off and denoising off. Folded RMSE
is 0.00041711 and slab RMSE 0.00045318 in both modes; every ROI is finite and
nonnegative. These targets are independent unfolded/Snell/Jones references,
not PT–BDPT brightness equality. Operator times include initialization:
46.323/119.252 s for folded PT/guiding and 41.812/115.167 s for slab PT/guiding.

The [current appearance manifest](../build/tests/python/presentation_refresh_v32/manifest.json)
contains thirteen completed renders with exact v32 SHA-256. Ten retain the v3
scene designs: nine use 512 fixed samples and the indirect BDPT scene uses 2,048;
the two-sided Glass and generalized-dispersion appearances retain 512 samples,
and the coated-generalized white furnace uses 256. Adaptive sampling is off;
OIDN previews and raw passes are saved alongside editable copies. Original
camera/resolution/display transforms are retained and labeled in the
[overview provenance](../build/tests/python/presentation_refresh_v32/overview_manifest.json).
The white furnace panel is a diagnostic, not a beauty render or a new numerical gate.
These images do not establish convergence, general passivity or every native pipeline.

The first gallery process completed ten scenes, then stopped before the next render
because an inherited multilayer output setting restricted the worker's file format.
Its [failed log](../build/tests/performance/v32_presentation_refresh.log) remains.
The minimal test-worker media_type=IMAGE fix resumed only the remaining three;
completed images were not rerendered, and worker hashes/output-policy changes
are recorded in the manifest. Binary, scene materials and render physics did not change.

The indirect raw Noisy Image contains out-of-gamut RGB down to −6.113. Its
[linear BT.709 luminance diagnostic](../build/tests/python/presentation_refresh_v32/indirect/raw_luminance_diagnostic.json)
finds minimum luminance −2.13×10⁻¹² in one pixel and no pixel below −10⁻⁶;
58,770 pixels have at least one negative RGB channel. Combined is the OIDN result;
its positive values are not used as physical proof. Raw data is retained.
Indirect render plus denoise took 427.601 s including initialization, compared
with the historical v3 presentation's 326.402 s; those appearance timings are
not controlled warm speed measurements. No quality-search rerenders were added.

The [single matched default-off CD benchmark](../build/tests/python/cycles_diffraction_benchmark_default_off_v32/report.json)
used the exact historical v3 scene and script hashes, 512×369, 512 fixed samples,
seed 11, PT, adaptive/denoising/guiding off and coherent connections off.
One 47.571 s warmup preceded exactly three measured trials:
1.717108/1.721730/1.660009 s, median **1.717108 s**. The historical v3 median is
1.616313 s: current v32 is **6.24% slower in this matched fixture**.
The [saved-settings and raw-pixel comparison](../build/tests/python/cycles_diffraction_benchmark_default_off_v32/comparison.json)
confirms both hashes/settings and finite pixels, maximum absolute pixel difference
2.86×10⁻⁶ and RMSE 5.16×10⁻⁸. This is one bounded warm observation with EXR-writing
cost, not a fastest-package claim or an isolated causal regression proof.
No extra benchmark repeats were run.

## Limits

General coherent multipath, remaining native Multi-GGX combinations including
Thin Wall and exhaustive native material/pipeline combinations, linked
cache-dependent parameters, arbitrary-profile Realistic convergence and
exhaustive pipeline certification remain unfinished. The mirror, direct
two-transmission slab, three-event mixed path and finite planar face-cluster gates
do not cover curved/nonplanar optical surfaces, arbitrary material topology,
participating media, overlapping/nested dielectrics,
long path families, or the full Fabry–Pérot series of internal slab reflections.
The albedo approximation has no general strict
passivity proof; Thin Wall is approximate, and indirect presentation noise
remains. No images or physical references were substituted for renderer output.

Earlier evidence stays available in [the flat-native report](cycles_diffraction_flat_native_delivery.md),
[the previous consolidated report](cycles_diffraction_final_delivery_20260927.md),
and `tests/output/diffraction/presentation_flat_native_v1/index.html`.
