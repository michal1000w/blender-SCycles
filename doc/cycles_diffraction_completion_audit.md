# Current completion audit — 28 September 2026, v37

The original all-shader, arbitrary coherent-transport and full-pipeline request is **not complete**. This current summary supersedes the historical entries below; successful bounded cases do not certify their excluded configurations. No radiance fitting, replacement of failed references or relaxed acceptance gates is used.

Current corrected package: `build/diffraction_delivery_20260928_final_v37/Blender.app`, executable SHA256 `46b8fb0639dee1dcea94ae42afd3acd7e794132f6cca10b4735d1ee45caf3a20`; exact resources are recorded in `resource_manifest.json`. It adds the narrow Metallic wavelength-domain correction to v36. The actual512sample Metallic raw image passes the luminance audit (zero Y<−1e−6; minimum Y+0.000852877); the v32/v36 negative-luminance images remain failure evidence. See [correction details](cycles_diffraction_metallic_spectral_correction.md). Six planar, sphere phase/control,68 data-channel and four PT/guiding physical reports retain explicit **v36/v35** provenance; they are not relabeled v37. The full13scene v37 appearance refresh is **complete** at `build/tests/python/presentation_refresh_v37/index.html` with an actual-image overview and per-case provenance. All raw images are finite with zero Y<−1e−6 in `build/tests/python/presentation_refresh_v37_raw_luminance_audit.json`. Samples remain fixed:512 for11appearance cases,2048 forindirect and256 forcoated furnace; adaptiveOFF. Raw NoisyImage evidence is separate from OIDN previews. The final matched benchmark is complete: median 1.777637 s, 3.53% slower than v32 and 9.98% slower than matched historical v3; this single fixture does not establish a universal ranking. Historical reports remain preserved.

| Requirement | Current implementation and evidence | Practical limits / remaining gap |
|---|---|---|
| Fast default | Dedicated Diffraction defaults to Fast; scalar orders and bounded tails, native SVM/OSL integration. Native MultiGGX directional-return caches are distinct from the optional electromagnetic cache. | A practical approximation, not a Maxwell solution or universally fastest model. Realistic remains experimental. |
| Solid Glass / mixed Principled MultiGGX | Two-sided reciprocal return, passive tint, film, native layer weights, generalized reflection/transmission bases and exact native Cauchy dispersion. `build/tests/performance/generalized_final_precise_v28/results.json`; `coated_generalized_fresh_v29/results.json`; coated v29 furnace reports. | Completed-return tint is an absorption approximation, not a microscopic random walk. Cache-dependent geometry/optical inputs must be constant; transmission uses the native isotropic carrier. |
| Completed Thin Wall Fast | Reciprocal facet/order envelope and two-air-port cached return; continuous blend to the original native zero-relief endpoint. 8,836 closure/actual graph/native-limit checks; Metal furnace8/8, CPU OSL2/2, one BDPT+guiding furnace pass. `doc/cycles_diffraction_thin_sheet_fast.md`; `doc/cycles_diffraction_thin_sheet_fast_cache.md`. | Component reciprocity does not make the native-compatible transition fully reciprocal. Scalar microscopic phase-screen approximation; supported passive constant inputs and n(lambda)>=1. Linked/resource-limited cases retain the explicitly approximate legacy route. White-world renderer gates do not certify every tint/film combination. |
| Thin cache on actual M5 | Five CPU/Metal cases pass, including shipping16λ×24μ×16φ/512facets; warm shipping construction20.525ms Metal versus817.991ms CPU, max directional difference2.14577e-6. `build/tests/performance/thin_sheet_gpu_cache_shipping_v36/gpu_cache_comparison.json`. | Cache-only timing after initialization; not whole-render performance. Cold shader specializations are separately reported. |
| Planar multipath interference | Deterministic diagonals and same-group complex path pairs, including same-source arms; explicit scalar mirror and three-axis dipole/Jones Glass modes. Exact triangle membership handles joined faces, gaps and blockers. Six v36 BDPT regressions pass: `build/tests/python/coherent_planar_regression_v36/results.json`. | At most64 patches,64 candidates,4 events, with explicit cap errors. Static ideal planar materials and passive Lambertian detector; arbitrary rough/diffuse phase transport is absent. |
| Native-sphere reflection | Exterior analytic mirror sphere, curvature spreading and compensated optical length. Three v36 phase/control cases pass unchanged independent double-reference gates; four unsupported configurations reject. `build/tests/python/cycles_coherent_sphere_render_v36/results.json`; `cycles_coherent_sphere_negatives_v36/report.json`. | One exterior mirror event on a native sphere; no curved refraction, ellipsoids, interior sources or multiple curved events. Earlier black/debug images remain failure evidence. |
| Current PT / guiding / BDPT | Four v36 sphere/joined-slab PT and PT-guiding cases pass unchanged independent references: sphere RMSE7.5144e-5, slab0.000453176; 256²/128fixed, adaptive/denoiseOFF. `build/tests/python/coherent_sphere_slab_transport_v36/results.json`. Six planar and three sphere BDPT cases also pass. | These independent first-receiver references, not PT brightness comparisons, validate only the bounded classes. Guiding does not train on signed coherent contributions. |
| Receiver model | Passive constant/textured RGB Diffuse or exact pure-diffuse Principled; v36 checker RGB RMSE<=1.301e-5 and black exactly0. | Flat receiving triangles, geometric normals, zero shading-terminator offset, no rough diffuse/effective film/extra lobes. RGB response is an explicit approximation, not complex phase extracted from an arbitrary BSDF. |
| Surface-data passes | All68 tested channels match native control exactly; Combined change<=4.7684e-7. Depth, position, normal, UV, IDs, Cryptomatte, colors, mist, AOVs and denoising data. `build/tests/python/cycles_coherent_data_passes_v35/report.json`. | Light decomposition/lightgroups still reject. Stored motion-data support does not permit moving coherent interfaces. Expected negative-render exit1 is preserved separately. |
| Full pipeline | Owned NEE/BDPT prefixes are partitioned; per-candidate bounce controls enforced; marked ideal interfaces retain singular closures. | Coherent mode is Metal-only and excludes motion, volumes, photon mapping, light/shadow linking, shadow catchers, arbitrary transparent blockers and light-pass attribution. All-pipeline scope remains incomplete. |
| All material families / gallery | Earlier dedicated, Glossy/Metallic, Glass/Refraction, Principled, Thin Wall and Ashikhmin evidence retained. Final13scene v37 presentation refresh is complete at `build/tests/python/presentation_refresh_v37/index.html`; all raw images finite, zero negative Y below−1e−6. Fixed sample counts and preview/raw distinctions are recorded in its manifest. | Hair, subsurface, volume, arbitrary layered/custom closures and every distribution combination remain outside proven scope. Appearance is not energy/convergence certification. |
| Performance | Current sparse caches/analytic reflection paths have measured bounded timings. Final matched CD benchmark: 1.777637 s median at 512×369/512 fixed samples, adaptive/denoising off; current comparison report is `build/tests/python/cycles_diffraction_benchmark_default_off_v37/comparison.json`. | Historical v32 default-OFF median1.717108s versus matched v3 1.616313s was6.24% slower observed; preserve it. No universal speed ranking or zero-overhead claim. |

The isolated first-receiver fixtures use Diffuse Bounces0. The older diffuse-one folded scene includes legitimate receiver/mirror interreflection omitted by its oracle; the later Newton corner dropout was a separate genuine implementation bug fixed by exact planar unfolding. Disk/setup failures, the initial OSL backface failure and sphere visibility failures remain preserved. Historical packages that were losslessly archived retain verified gzip binaries and restore JSON; current v36 and corrected v37 are directly available.

## Historical continuation plan — superseded by the v36 summary above

The user authorized another 10 points after the 77% account reading, giving an
87% ceiling for this continuation. v32 remains the immutable validated package
until a replacement passes integration and renderer gates. Work starts with
native analytic sphere coherent reflection and a practical Thin Wall return;
surface-data pass compatibility is a separate pipeline extension. These are
active implementation tasks, not delivered support.

The [joint reciprocal-envelope experiment](cycles_diffraction_thin_wall_reciprocal_envelope_experiment.md)
passed discrete symmetry/energy algebra but exposed a maximum resolved lossless
row excess of 0.000599 with its coarse cache. Its report preserves the failures;
that test-only prototype was not integrated. The practical Fast continuation
must declare its approximation and numerical tolerance rather than imply exact
continuum passivity or silently change the native zero-relief limit.

## Supported choice and smallest meaningful remaining work

The subsequent [actual-closure Thin Wall experiment](cycles_diffraction_thin_wall_return_experiment.md)
rules out a symmetric additive return as a simultaneous energy-and-reciprocity
fix: it leaves the first event's antisymmetric component unchanged. Its resolved
row checks also expose coarse angular quadrature error. This is diagnostic
evidence, not a production correction or a global passivity proof; v32 remains
unchanged.

Use Fast plus the explicit planar coherent connection option for declared ideal
interfaces and passive Lambertian receivers when that model fits the scene.
Its fast all-reflection branch removes the measured corner dropout rather than
changing the image scale. Keep Realistic an optional experimental dedicated-node
choice; no unsupported setting should silently change BSDF distribution or
pretend to be a converged electromagnetic solution.

General smooth reflection/refraction requires stationary-path seeds on actual
curved geometry, native patch/material evaluation, visibility, a correct ray
Jacobian and caustic phase convention, plus an estimator partition for every
retained path. The current face clusters are a useful finite triangulated
geometric model; increasing tessellation and hitting the candidate cap is not
proof of smooth-geometry convergence. Rough-path coherence additionally needs
complex scattering amplitudes and a stated microscopic roughness/source
ensemble. Taking a square root of a radiometric BSDF/throughput is insufficient.

The scheduled bounded v32 PT/guiding gates, 13-case matched presentation and
default-OFF warm benchmark are complete within their stated scope. A next
implementation must choose one missing model explicitly: a
reciprocal Thin Wall missing-energy approximation, or curved ideal-interface
stationary connections with an independent caustic/phase oracle. Full pass
attribution, transparency and motion require additional transport work. They
are not resolved by another appearance render.

## Preserved historical audit (superseded status)

The following chronological entries retain their original binaries, failures
and restrictions. Statements such as “general coherence absent” or “MultiGGX
unsupported” describe those older deliveries and are not the current v32 status.

# Completion audit — selected Fast candidate

This is a requirements audit, not a completion declaration. The original goal
remains active. The selected Fast renderer and its image suite are separate from
the ongoing GPU Realistic-cache integration and validation.

| Requirement | Evidence inspected | Current conclusion |
|---|---|---|
| Fast default and UI quality choice | Diffraction node initialization and RNA quality enum | Implemented for the dedicated node; Fast is approximate |
| Spectral reflection/transmission | Fast interface numerical tests, compiled SVM tests, Metal/OSL furnace records | Targeted checks pass; not full electromagnetic-efficiency equivalence |
| Wavelength-dependent conductor data | Embedded CSV parser, GPU lookup, refresh/recovery tests and aluminum scenes | Implemented and validated in the dedicated node |
| Physically rigorous optional mode | Host electromagnetic solver and response-cache integration | Implemented backend, but arbitrary-profile modal convergence remains unproven |
| GPU construction of Realistic cache | Packaged Metal/MPS backend, selected-device dispatch, standalone adaptive spectral-metal cache tests, and Realistic flat-metal Blender render in `metal_application_dispatch_v1` | Application integration implemented and flat-interface smoke passes; finite-depth application coverage and GPU speed advantage remain unproven |
| All appropriate shader/node integration | Dedicated diffraction SVM/OSL path; older Glossy prototype; Principled/Glass/Metallic node declarations | Incomplete; dedicated-node success does not prove broader node support |
| PT, BDPT and guiding | Covered-scene fresh-state controls, earlier 68-render suite, selected-build 72-render suite | All 72 selected-build renders completed and were audited; targeted controls pass, while indirect presentation convergence remains insufficient |
| Optional general cross-object coherence | Existing architecture research and complex local scattering utilities | No renderer implementation; local diffraction is not general coherent transport |
| Scene representations | Bare CD/DVD, explicit covered CD, transmission display, angular/furnace controls and indirect scene | Multiple fixtures exist; indirect presentation remains noisy and broader pipeline fixtures are incomplete |
| Fixed-sample isolated GPU benchmarks | Aluminum-table benchmark and delta-receiver before/after benchmark | Completed with adaptive sampling disabled; timings belong to their recorded versions and workloads |
| Fastest correct candidate | Delta-receiver bypass, compiled positive/negative graph checks, CPU/PT/BDPT image comparisons | Current Fast candidate selected; not final acceptance of the entire requested feature set |
| Retrospective brightness interpretation | Retrospective document and per-method controls | PT is not treated as physical truth; no image brightness normalization |
| Full-pipeline regression | Existing closure, serialization, refresh and covered controls | Motion, AOVs, volumes, remaining material combinations and complete convergence coverage remain open |

Selected binary for the current full-suite run:
`2bb04b4807ff2a32000d7fe87ee1c3e2023f23f0889ac1214547985c98fbc9fe`.
The completed run is `tests/output/diffraction/fast_selected_full_suite_v2`, with frozen
provenance in `fast_selected_full_suite_v2_provenance.json`. Its 72 jobs comprise
15 numerical fixtures and three appearance fixtures per transport. Appearance
fixtures use the aluminum table; analytic controls keep their reference indices.
Rendering all jobs successfully does not by itself discharge the missing
requirements above or certify Fast as a Maxwell solution.


## Latest Principled reflective integration

The final development binary is
`fded74d7efff270fac6c0d4b5249529c25c61b047d3eae95795e0c6f344dbb5e`.
It extends Principled GGX metal and dielectric base reflection, preserving
layer attenuation, partial coverage, native Fresnel, thin film, tangent and
rotation. Transmitting and multi-scattering diffraction are explicitly rejected.
The build passes; the shared closure CPU regression passes 47,190 accepted
events. `principled_support_validation_v1` passes ordinary-material and two
unsupported-setting checks. `principled_delivery_v2` contains visually inspected
finite-pixel PT/OSL previews with the tangent fix. These are integration checks,
not complete transport convergence proof. Historical results below retain their
own source/binary provenance, including superseded limitations and failures.

## Glass node delivery integration

The Glass node now exposes grating coverage, pitch, depth, duty and tangent.
Both SVM and OSL compile the joint GGX/Beckmann dielectric implementation,
including thin film. Zero grating coverage retains standard glass. Multi-GGX
with diffraction explicitly errors rather than silently changing distribution;
this remains a missing requested combination. CPU/Metal/OSL compilation passes
(`/tmp/diffraction_glass_delivery_build_v2.log`).

New representative saved scene and actual Metal renders:
`glass_delivery_pt_v1` and `glass_delivery_bdpt_v1`, 720x400, 256 samples,
adaptive sampling OFF and denoising OFF. PT took 26.4756 s, BDPT 74.0004 s.
These are integration smoke timings including render preparation, not an
isolated diffraction overhead comparison. Images were visually inspected:
all three materials display grating colors; substantial spectral noise remains.
No PT brightness normalization was performed. Source and binary hashes are in
`glass_delivery_provenance_v1.json`. OSL/guiding runtime checks are separate.

## Delivery budget and finalization constraint (2026-09-27)

The user capped remaining work at 10% of the Codex allowance. Account usage at
acceptance was 33% of the weekly window; use 43% as the ceiling and check usage
at delivery milestones. This is an account percentage, not a token budget.
Prioritize usable material integration, a build, representative renders, and a
concise delivery report. Do not start further open-ended numerical or timing
campaigns. Preserve unresolved feature gaps and do not declare them complete.

The actual Metal surface-mixture test passed on Apple M5:
`tests/output/diffraction/coated_surface_metal_v3.json` (zero failures,
zero same-direction differences, zero same-GPU reevaluation errors). Its added
256 setups exercise actual closure selection and weighted surface MIS, including
coated matched interfaces, changed incident directions, and diffuse mixtures.
It does not prove rendered node integration. Earlier compile failures are
retained in `/tmp/coated_surface_metal_v1.log` and v2. The successful compile
required explicit private address spaces on triangle/motion-triangle array
parameters and removal of a private qualifier from a by-value guiding scalar.

## GPU response and application integration status

The experimental builder keeps material assembly, linear solves, propagation
and boundary matching on Metal. CPU double solvers used in diagnostics are
oracles, not execution fallbacks. The current accuracy candidate uses MPS LU,
inverse-based residual refinement, direct inversion of homogeneous admittance
blocks, optional exact-zero-ky polarization split, and a wider initial layer.
It has passed targeted cutoff/conical/high-order operator comparisons; this
is not certification of arbitrary materials or full electromagnetic convergence.

Physical-power conversion is independently checked against absorbing and
dielectric Fresnel interfaces. Existing double reference profiles still change
R/T totals by up to 0.001301 of incident power between N64 and N128. Earlier
intermittent GPU failures remain unexplained; successful later runs do not
remove them from the record. Standalone build timings still trail CPU on the
measured fixtures. The experimental candidate is now wired into Blender's
Realistic cache manager for active Metal devices, but is not certified as
production-ready. Fast bypasses this builder. The first actual application
render passes the flat-metal Fresnel mean check at 64 fixed samples with
adaptive sampling disabled; this does not validate finite-depth propagation.

Backend cancellation is checked between modal resolutions and GPU command
boundaries; already submitted commands finish before cancellation returns.
Standalone tests prove empty output and same-engine recovery at the tested
injection point. Callback lifetime tests and a full Blender build pass.
The safe-math implementation requires macOS 15 or newer and reports an explicit
error on unsupported systems. No silent CPU fallback is used on Metal failure.

The integrated backend now uses two lazy engine slots for small queries and
exclusive execution for larger queries. In the mixed-order pool regression,
six callers completed N4 through N128 queries with maximum complex-component
error 2.14524491536e-6. Observed queued cancellation, exclusive-solve cancellation,
idle-state restoration and subsequent recovery passed
(`metal_pool_wait_cancellation_v1/run.log`). This does not prove recovery from
every GPU or allocation failure. The separate injected-exception drain regression also passes, including a
finite numerical comparison after recovery: maximum difference from the
pre-failure response is zero in this fixture (`metal_failure_recovery_v2`).

Application persistent-data testing recorded two slot initializations, one cache
build and two cache reuses, with restored-image relative L1 7.71463548e-8
(`metal_pool_application_v1/reuse_verified.json`). The run used only Apple M5,
64 fixed samples, adaptive sampling off and denoising off. This validates reuse
for that fixture, not disk persistence or full-domain relief performance. The
1024-sample full-domain relief attempt reached its predeclared 1800-second
budget and terminated cleanly with child exit code 1 after 1800.378 seconds
(`metal_pool_relief_full_v1/status.json`). The last progress report records
2011 visited nodes, 1002 accepted cells and 92413 reference solves. No EXR was
produced. This is incomplete cache construction, not a numerical rejection or
a completed-render benchmark. Concurrent CPU compilation also excludes that
attempt from isolated performance claims.

Detailed evidence, rejected candidates and limitations are maintained in
`doc/cycles_diffraction_gpu_builder_experiment.md`. Broad adaptive GPU validation,
all appropriate shader integration, optional cross-object coherent rendering,
and the remaining original pipeline/visual requirements are still open.

## Latest lossless backend integration

The Metal backend now applies a bounded host-double polar correction to final
GPU reference matrices for sampled lossless materials only. This fixes the
tested dielectric quadratic-cache rejection without relaxing cache passivity
checks. The integrated backend passes four bounded cache builds and4096
held-out fixed-N4 physical-response comparisons (`metal_integrated_lossless_v1`).
Full Blender compilation and correction rejection tests pass. Actual Blender
rendering with this backend revision, full-domain preparation performance,
and all earlier unmet requirements remain open.

The first full-domain dielectric Blender test of the integrated correction
fails at a previously untested root-cell response with unitarity residual
0.0002933715, above the1e-4 correction bound. No image is produced. Exact
reproduction is in `metal_lossless_application_context_v1`. The bounded
regressions do not discharge this application failure; integration acceptance
remains unresolved.

The conditioned-basis v3 backend subsequently passes the full-domain finite-
depth dielectric PT application furnace (`metal_conditioned_application_v1`):
1024 fixed samples, adaptive/denoising off, maximum RGB-mean error0.000406289
against unit radiance (smoke gate0.005). Cache build148.444s, total195.578s.
The original rejected revision remains recorded; this later pass resolves that
fixture for v3 only. Full-method coverage and all earlier missing features
remain unverified.

The conditioned v3 binary passes the analytic dielectric furnace in all four
transport configurations (`metal_conditioned_transport_v1`),1024 fixed samples
per method with adaptive/denoising off. EXR hashes and finite means verified;
maximum mean error0.000406289. This is a full-domain cache/application smoke
fixture, not proof of complex indirect transport, guiding effectiveness,
modal convergence or cross-object coherence. New tensor early rejection has
CPU buffer-equivalence checks and a passing Blender build; GPU validation is
still pending.

Tensor early rejection now also passes six isolated sequential Metal PT
furnace renders (three per version): median cache preparation148.523→124.639s
(16.08% lower), median total187.367→169.410s (9.58% lower). All render gates
and EXR identities pass. Evidence:`metal_tensor_rejection_benchmark_v1`.
Retained for this workload; no claim of full feature acceptance or adequate
default Realistic startup speed.

### Transmission geometry prerequisite (not shader support)

Added transmitted-facet forward mapping, inverse facet roots, and solid-angle
Jacobian helpers in `bsdf_diffraction_util.h`. These are not connected to Glass,
Metallic, or Principled and do not constitute completed shader support.
The standalone `cycles_diffraction_transmission_geometry_test.cpp` uses 200,000
seeded random inputs: 103,303 propagate, 88,194 regular inverse checks and 64,864
Jacobian comparisons pass. An independent double-precision Snell/tangential-kick
finite difference gives maximum relative Jacobian error 0.000260974; maximum
forward roundtrip error is 0.000407936. 38,440 cases have two facet roots.
Three inverse roots at transmitted cosine approximately 0.0001 cannot be
forward-reproduced in float near the propagation cutoff and are explicitly
counted as unresolved grazing cases. Axial singular neighborhoods are excluded
from the regular inverse-normal check; no broad cutoff accuracy claim is made.
Initial float finite-difference failures were diagnosed with a double oracle,
not treated as physical evidence or silently discarded. These new helpers still
need Metal compilation, cutoff treatment, BSDF integration and render validation.

### Actual Metal transmission geometry check: failed, not accepted

The full Blender build (CPU, OSL, software Metal, MetalRT, and MetalRT motion
libraries) completed successfully. The existing Metal math test now executes
transmission forward mapping, inverse roots, and the Jacobian as well as the
previous reflection/table/grid tests. On Apple M5, 16,384 inputs produced seven
component mismatches at the existing 2e-4 scaled geometry tolerance. Reports:
`tests/output/diffraction/transmission_geometry_metal_v1.json` and
`transmission_geometry_metal_diagnostics_v1.json`. The diagnostic rerun corrects
output stride reporting; both actual results remain failed.

One particularly important fixture, query 11456, gives zero inverse roots on CPU
and two on Metal for nearly identical outgoing directions. Its Jacobian is about
20,680, consistent with a highly ill-conditioned inverse near a fold, but that
observation is not a correctness waiver. Other Jacobian discrepancies include
near-index-matched zero-order transmission (eta=0.999759614, Jacobian ~11.55M).
Do not expose Glass controls or declare this geometry accepted until the
near-singular behavior and its integration consequences are resolved.

The independent CPU test also passed 102,423 source/receiver reverse-direction
checks using eta'=1/eta, delta'=delta/eta and the opposite facet normal. This does
not resolve the GPU discrepancies. A subsequent defensive finite-eta/delta guard
in the transmission Jacobian still needs a fresh full build.

### Zero-order transmission conditioning improvement

Rationalized the zero-order Jacobian denominator using
`(eta-1)*(eta+1)/(eta*cos_i-cos_o)`. This avoids cancellation for nearly equal
indices without altering the mathematical model or tolerance. The independent
CPU test now includes 131 near-index-matched Snell cases through the closest
representable indices around one; maximum relative Jacobian error is
2.8435e-6. The original randomized/reverse checks still pass with the same three
explicitly unresolved grazing roots. Actual M5 rerun report
`transmission_geometry_metal_rationalized_v1.json` has six remaining mismatches,
compared with seven previously: the near-index-matched zero-order mismatch is
resolved. The remaining test still fails and is not an integration acceptance.

`cycles_diffraction_transmission_fold_diagnostic.cpp` reproduces query 11456.
The inverse discriminant of the rounded CPU outgoing direction is negative even
when computed in double (-7.51354e-9), while the original facet's a^2 is positive
(1.81068e-8). Thus simply evaluating the inverse discriminant more accurately
cannot restore that sampled facet: the float direction has crossed the fold.
No clamping or tolerance relaxation was introduced to conceal this. A robust
sampling/evaluation treatment and integration-level bias checks remain necessary.

### Finite-angular-measure test and positive-discriminant correction

Added `cycles_diffraction_transmission_measure_test.cpp`: independent uniform
facet sampling versus uniform outgoing-direction integration of the summed
inverse-root Jacobian, with eight angular bins, nine eta/order-kick cases,
reported standard errors and explicit missing-inverse counts. This is a geometry
measure test, not an efficiency/BSDF/render proof. Each trial samples a facet and
an independent outgoing direction. Acceptance requires bin differences below
6 standard errors + 0.0005 and every bin standard error below 0.005; these gates
were set before the large run. Heavy-tailed estimates are explicitly rejected
when their uncertainty exceeds that limit.

The first 20-million-trial-per-case run failed: eta=2/3, delta=.9 integrated to
34.58391 versus forward probability .8931291, due to a rare artificial density
spike. Preserved in `transmission_measure_20m_v1.log`. The inverse code checked
`length2 > delta*delta`, but then computed `sqrt(1-s*s)` where s could round to
one. Preserve the already-positive discriminant instead:
`sqrt((length2-delta*delta)/length2)`. No density cap or test tolerance change.
The same seeded 180-million-trial suite then passed in
`transmission_measure_20m_discriminant_v1.log`; the affected case integrated to
.89242689 with max bin standard error .00114775. Maximum bin error across all
cases .00227647; maximum bin standard error .00354633. Rare float-direction
missing inverses remain (up to 3421/20M), so this is not zero-bias proof.

CPU random/reverse/double-Jacobian tests still pass, with the same three
explicitly unresolved grazing roots. M5 pointwise rerun
`transmission_geometry_metal_discriminant_v1.json` remains failed with five
component discrepancies (previously six). Do not declare complete GPU geometry
or shader support from the CPU integral result. No render benchmark or installed
binary was changed during this test.

### Actual M5 finite-angular-measure test

Added `--transmission-measure` to the existing provenance-recording Metal test
runner, with `cycles_diffraction_transmission_measure.metal/.mm`. The production
forward/root/Jacobian helpers execute under Metal fast math on Apple M5. Nine
eta/kick configurations each use 20 million independently generated facet and
direction trials; the host accumulates eight angular bins in double precision.
The same predeclared CPU statistical gates apply, including the maximum standard
error gate. No density clamp, normalization, or removed outlier is used.

`tests/output/diffraction/transmission_measure_metal_v1.json` passed all nine
cases (180 million trials). Maximum bin difference .00130063, maximum bin
standard error .00352766. Missing inverse counts remain 0..3245 per 20 million
trials and are reported rather than discarded. Largest recorded integration
weight 21018.0. This is actual GPU geometry-measure evidence, not a rendering
benchmark, BSDF-efficiency validation, proof of zero bias, or a replacement for
the five retained pointwise GPU failures. Installed Fast renderer unchanged.
Next integration work must use the summed preimage density for visible GGX
facets and validate the full transmitting/reflected efficiency model before
claiming Glass/Principled shader coverage.

### Visible-GGX transmission geometry foundation

Added `diffraction_transmission_ggx_order_pdf` and
`diffraction_transmission_ggx_sample_order` to `bsdf_diffraction.h`. These use
Cycles' actual anisotropic GGX visible-normal sampler and D/lambda functions,
sum both valid facet preimages, restrict the transmitted macro-hemisphere, and
retain rejected facets as null events. The functions expressly do not define
order efficiency or a full glass BSDF. Specular roughness and index-matched
zero-order delta events are reserved for separate singular handling. No shader
controls or Glass closure have been exposed using these helpers.

`cycles_diffraction_transmission_ggx_test.cpp` compares independent angular
integration with visible-GGX samples for two incidence cosines, three index
ratios and three order kicks, alpha_x=.2/alpha_y=.55. Two million trials each.
The initial uniform-direction proposal failed its uncertainty gates (eight
failures; sharply concentrated index-matched peaks); retained log
`transmission_ggx_measure_2m_v1.log`. Its initial `missing_inverse=0` field was
unused for that proposal and is not evidence of absence of missing inverses.

The optional uniform-facet proposal, whose unweighted geometry was separately
validated against uniform directions on CPU and GPU, passed all 18 cases:
`transmission_ggx_facet_measure_2m_v1.log`. It reduces variance without capping
weights. It shares inverse geometry with the evaluated density, so it complements
rather than replaces the earlier independent tests. Proposal directions with no
inverse remain explicit null contributions and are counted. Actual GPU GGX
execution, efficiency/reciprocity, complete shader integration, and render
validation still remain.

### Fast dielectric facet efficiency prototype — not accepted

Added `DiffractionDielectricFacet` coefficient helpers alongside the rough
geometry, still disconnected from shader setup. Candidate scalar model:
nonzero reflected Fourier power receives min(F_i,F_o); transmitted power
receives min(1-F_i,1-F_o_reverse), with direction-independent thin-plate phase.
Unused power returns to mirror reflection. This bounds reflected power by F_i
and transmitted power by 1-F_i, and is pairwise reciprocal in exact arithmetic.
It is explicitly not Maxwell and suppresses diffraction-assisted escape beyond
the ordinary Fresnel critical angle. No claim of Realistic accuracy is made.

The zero transmitted order obeys Snell, so its Fresnel factor now uses F_i
without re-evaluating from rounded cos_t. In 20,000 sampled facets, the flat
profile limit error dropped from .000370622 to 2.98023e-8. All passivity checks
pass, minimum residual 3.92199e-5. However the 122,575 reverse-direction power
checks still fail their 2e-6 gate (369 failures, max .000603925 after zero-order
change). Both runs are retained: `dielectric_facet_initial_v1.log` and
`dielectric_facet_snell_zero_v1.log`. This prototype is not accepted or connected
to Glass. Near-critical conditioning must be quantified with independent
high-precision references, and full rough closure reciprocity/energy tested.
The subsequently added finite/range check for order-bound integer conversion
still needs compilation; all newly added coefficient code needs Metal testing.

### Dielectric facet reverse-power conditioning resolved in CPU tests

The candidate now stores both incident/transmitted refractive indices. Reversal
swaps those exact values instead of repeatedly inverting a rounded ratio. The
zero transmitted order evaluates Fresnel from both known cosines; its direction
uses the same scalar Snell construction as ordinary dielectric Fresnel, avoiding
a second reconstruction from a rounded transverse vector. No test gate was
relaxed. The retained earlier failures are superseded only for this tested scope.

`dielectric_facet_index_pair_v1.log`: all 122,575 reverse pairs and 20,000 flat
profiles pass; maximum reverse difference 3.57628e-7, minimum mirror residual
3.92199e-5, flat-profile Fresnel difference 5.04851e-5. An independently written
double-precision Fresnel/Fourier reference agrees within 6.98416e-6 (gate2e-5).
The existing transmission geometry test still passes, including 102,423 reverse
directions and 131 near-matched-index cases; three near-grazing roots remain
explicitly unresolved. This does not yet validate a complete rough dielectric
closure, transport eta factors, Metal coefficient execution or shader UI.

### Combined Fast rough-dielectric evaluator and sampler

Added `bsdf_diffraction_dielectric.h`, registered with kernel CMake and included
by the BSDF dispatcher for compilation, but not hooked to any closure type or
shader socket yet. It combines reciprocal facet coefficients, GGX visible-normal
sampling, both reflection/transmission inverse branches, and summed continuous
solid-angle density. It handles smooth surfaces and index-matched zero-order
straight-through transmission as discrete events. This retains the ordinary
Cycles microfacet transport convention; integrator adjoint handling still needs
integration validation. No Maxwell/Realistic claim applies to this scalar model.

`rough_dielectric_reciprocity_v1.log`: 20,000 rough reciprocal comparisons pass
(max scaled discrepancy3.94804e-6), 19,128 accepted random samples respect
0<=eval<=pdf, 20,000 flat-profile comparisons with conventional analytic GGX
refraction pass (max scaled discrepancy1.03137e-6), plus 1,000 matched-index
flat profiles are exactly unit straight-through delta events.

`rough_dielectric_measure_1m_v1.log` retains the initial failed uncertainty gate
at the sharp matched-index configuration (three failures). Increasing to ten
million trials per configuration, with unchanged gates, passed all six
configurations in `rough_dielectric_measure_10m_v1.log` (60M trials). Four metrics
compare sampler expectations against independent uniform solid-angle integration:
probability, total energy, transmission energy and reflection energy. The known
matched-index delta mass is accounted for separately, not integrated as a
continuous density. Gates: difference <=6SE+.0005, SE<=.005. These are finite
statistical checks, not proof of zero bias or broad physical accuracy.

The full Blender build passed including three Metal variants after the new
header was included. Actual execution of the combined rough model on Metal,
shader/closure integration, material layering/thin-film interactions and render
suite coverage remain. Installed selected Fast application not overwritten.
Source hashes: `rough_dielectric_math_provenance_v1.json`.

### Compact dielectric closure wrapper and world-direction density

Added a reserved dielectric diffraction closure ID and wrappers for world-frame
sampling, evaluation and delta probability. The inline storage is 96 bytes,
exactly the existing CPU ShaderClosure size; it has no extra allocation or
private pointer. No shader hookup or BSDF dispatcher route exists yet.

The initial rotated-frame closure test exposed 41 sample/evaluation differences,
max scaled PDF difference .475056 near folds, because the outgoing world vector
rounds differently when transformed back. The wrapper now evaluates the density
of its actual returned world direction. Direction generation is separate from
evaluation, so this does not duplicate the expensive order sum. The local math
sampler retains its prior behavior. This repairs MIS consistency; it does not
prove finite-precision sampling has zero bias.

`dielectric_closure_world_direction_v1.log`: 25,991 discrete events agree with
queried masses within1.12986e-6; 13,456 continuous events exactly agree with
reevaluation; closure value-copy, output labels, eta and finite-value checks pass.
The smooth and rough matched-index atomic cases are covered with a rotated
normal/tangent frame. The initial failed output is preserved in task tool history.

Integration review identified a remaining requirement: the matched-index rough
model mixes a straight-through atom with continuous diffraction. It must not be
silently admitted to the current continuous-only guiding resampling path, or
have its mass treated as a solid-angle PDF. Closure splitting or explicit mixed
measure guiding/MIS handling must be completed before shader exposure. MNEE,
BDPT reverse atomic mixtures, AOV/roughness classification and shader hookup
remain to be wired and tested. The new ID is deliberately not classified as an
ordinary microfacet (which would trigger invalid structure casts).

### Matched-index closure split and dispatcher integration

Implemented conditional sampling that excludes the known straight-through mass
without rejection or additional random numbers. Closure setup reserves one
straight-through closure and one remaining diffraction closure, weighted by a
and 1-a. This applies at both smooth and rough settings so later glossy blur
cannot create an unsplit atom. Setup refuses insufficient slot capacity before
allocating either component. Shader compilation must still reserve two slots.

`dielectric_split_v1.log` compares the original mixed sampler with the weighted
conditional sampler plus analytic atom for six configurations, 500,000 trials
each. All pass unchanged statistical gates; no atom appears in the continuous
sampler. Cases include a=0 and a=1. `dielectric_split_closure_v1.log` covers
storage, setup weights, insufficient capacity, 20,000 smooth events and 19,056
continuous events; masses and reevaluated densities pass.

Wired sampling/evaluation, delta probability, eta/roughness, blur and labels in
`closure/bsdf.h`; classified the straight closure as singular and the rough
closure as continuous for guiding. Extended the existing grating detection and
delta-query helper so camera/light BDPT paths select atomic reverse-mixture
queries for the new singular closures. They remain outside generic microfacet
classification to avoid invalid casts. These code paths still need GPU renderer
integration validation; classification alone is not proof of guiding correctness.

`dielectric_dispatch_v1.log`: actual CPU BSDF dispatch with real kernel globals
passes1,965 sampled events, including discrete evaluation. Standalone build
links kernel/device/cpu/globals.cpp plus util/profiling.cpp, thread.cpp and
guarded_allocator.cpp, with intern/atomic and bundled TBB include paths.
The full141-step Blender build passed, including all three Metal variants;
log `/tmp/diffraction_split_build.log`. No node socket/SVM/OSL shader hookup yet,
no new rendering claim, installed selected Fast app unchanged.

### Thin-film prerequisite for Glass integration

Glass already exposes coating thickness/IOR, so replacing its Fresnel response
with the uncoated diffraction model would silently discard requested material
behavior. Reviewed the current `fresnel_iridescence_channel` implementation and
Belcour/Barla's primary project source:
https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
Their microfacet/Airy treatment motivates using a spectral film interface response;
it does not establish correctness of our combined grating/coating approximation.

Added `diffraction_thin_film.h`: single lossless film TE/TM characteristic-matrix
reflectance at a sampled wavelength, preserving both external Snell cosines,
with the film-critical limit and a scaled hyperbolic form for evanescent films.
An independent double complex interface-amplitude Airy reference is used in
`cycles_diffraction_thin_film_test.cpp`. 73,884 propagating pairs, including
14,039 internally evanescent films, pass; maximum reference error7.31805e-5,
exact reversal equality in this CPU test. Analytic quarter-wave antireflection,
zero thickness and optically thick evanescent-film limits pass. The initial run
had277 reciprocity failures from asymmetric FMA contraction; canonical summation
of the endpoint transverse momenta fixes it without changing tolerances.
Logs: `thin_film_initial_v1.log`, `thin_film_symmetric_v1.log`.

This helper is not yet connected to the dielectric grating model. A coating can
make matched-index straight-through power depend on facet angle, so the previous
constant-atom split cannot simply be reused. Its integrated atomic mass and
rough-interface sampling need explicit treatment before Glass shader exposure.
The helper is not a corrugated thin-film Maxwell model. Overall scope remains
incomplete; no new Glass socket or unsupported material behavior was introduced.

### Thin-film actual Metal validation

Added `--thin-film` to the provenance-recording Metal harness and a65,536-case
GPU kernel/host test. First actual M5 run failed68 cases, with errors up to.999866
(`thin_film_metal_v1.json`, diagnostic rerun preserved). The failures occurred
for large evanescent optical decay. Evaluating the asymptotic tanh value1 once
the argument reaches10 (where tanh rounds to1 in float) removed the large errors,
leaving one critical-angle error. `thin_film_metal_saturated_v1.json` remains
failed at.000230074 against the unchanged2e-4 gate.

Reformulated q_f^2 as a difference of squares plus FMA, symmetrically averaging
both endpoints. This reduces cancellation near the film critical angle.
The first Metal compile used unavailable `fmaf`; it failed before execution and
is recorded in `thin_film_metal_difference_v1.json`. A Metal-specific fma spelling
fix then passed in `thin_film_metal_difference_v2.json`: max CPU/GPU error
1.00136e-5, max reversal error2.38419e-7, zero failures. The independent CPU double
reference suite still passes (max error9.95359e-5, zero reversal difference),
recorded in `thin_film_difference_cpu_v1.log`. The initial host finite-float2
check compile failure likewise launched no GPU work and was fixed before these
runs. No tolerances were changed.

This establishes only the single lossless film helper, not combined rough
coating/grating correctness, closure integration or a renderer performance gain.

### Combined rough-dielectric Metal validation: harness correction

The first combined test (`rough_dielectric_metal_v1.json`) failed compilation,
not GPU execution: the Python expander generated over 62 million source lines
and incorrectly single-included macro templates, removing KernelData fields.
The runner now follows Cycles `path_source_replace_includes` semantics: pragma
once files are deduplicated, other processed text is cached and reusable, and
macro templates retain repeated inclusion. No synthetic guards are imposed.

The corrected outside-sandbox Apple M5 run is preserved in
`rough_dielectric_metal_v2.json`: 65,536 cases, energy-bin gate passed, maximum
energy-bin difference 1.4323841327268383e-5, one branch mismatch. Maximum
pointwise direction error was 1.2996272744203452e-5; 78 pointwise discrepancies
and maximum scaled PDF error 0.99989125267115342 remain explicitly unresolved.
This validates the declared aggregate local-math gate only. It does not establish
pointwise density agreement, renderer integration, node support, or full-goal
completion. The failed first run remains preserved.

### Same-direction rough dielectric PDF diagnostics

Added CPU evaluation at each GPU-returned continuous direction and saved severe
fixtures by seeded sample index. This distinguishes direction perturbation from
backend evaluation differences. Outside-sandbox M5 results:

- `rough_dielectric_metal_diagnostics_v1.json` (Fast math): 32 same-direction
  scaled PDF differences above 2e-4, maximum 1150.0169677734375 (case 43077:
  CPU evaluation zero, GPU 1150.01697). The aggregate energy gate still passes.
- `rough_dielectric_metal_safe_diagnostics_v1.json` (Safe math): 19 differences,
  maximum 0.0490910369. Both modes have 78 correlated sample pointwise differences
  and one branch mismatch. Safe math is diagnostic only, not selected as a fix.
- Case 60228 changes PDF from CPU-sampled 52.5700188 to GPU-sampled .00571685
  under a direction perturbation ~1.39e-7. CPU evaluation at that GPU direction
  agrees closely, identifying direction-boundary sensitivity separately.

These results do not establish acceptable MIS/BDPT behavior. Need robust
near-fold geometry/density evaluation and actual same-backend sample/eval,
reciprocal, and integration validation before shader exposure. Existing aggregate
`passed` fields are explicitly scoped and must not be read as full acceptance.

### Separate-dispatch Metal sample/evaluation consistency

Added a second Metal compute dispatch that reloads sampled outgoing directions
from device memory and reevaluates their continuous PDF. This prevents compiler
reuse of the first kernel's intermediate sampled expressions. The host gates
finite reevaluation and scaled PDF error <=2e-4, in addition to existing scoped
energy checks. `rough_dielectric_metal_reeval_fast_v1.json` records an authorized
outside-sandbox M5 run of 65,536 input cases: zero reevaluation failures and
exactly zero maximum reevaluation PDF difference for valid continuous samples.
This is useful same-backend local evidence, not a resolution of CPU/GPU boundary
sensitivity, world-frame wrapper behavior, reciprocal transport, or node/render
integration. No production compiler setting was changed.

### Metal reciprocal transport at sampled directions

The separate-dispatch evaluator now also reverses the stored sampled direction.
For transmission it swaps incident/transmitted indices, transforms the local
frame, converts wavelength/pitch and height/wavelength to the reverse medium,
and checks refractive-index-squared BSDF reciprocity after dividing out outgoing
cosines. The threshold 3e-4 matches the existing CPU reciprocity test and was set
before this GPU run. Failures contribute to the test exit status.

`rough_dielectric_metal_reciprocity_fast_v1.json`: M5 Fast math, 65,536 input
cases; zero reciprocity failures for valid continuous samples; maximum scaled
reciprocity error 0.00013448085227507866. Separate-dispatch forward PDF
reevaluation remains exactly equal. Existing CPU/GPU pointwise discrepancies
remain present and recorded. This is local sampled-direction reciprocity
coverage, not full BDPT path-connection, world-frame, or shader integration
acceptance. No renderer performance claim is derived from this numerical test.

### Both refractive-index orientations on Metal

Expanded seeded combined tests from equal/entering-only indices to three
strata: (1,1), (1,1.5), (1.5,1). Preserved first expanded result as
`rough_dielectric_metal_bidirectional_indices_v1.json`; v2 adds observed event
counts. Authorized outside-sandbox M5 Fast math run, 65,536 inputs, passed
existing aggregate energy, separate-dispatch PDF, and reciprocity gates.
Continuous event counts by stratum: 7936 / 18227 / 16667; continuous transmitted
counts: 3880 / 12982 / 4258. Thus leaving-glass transmission is actually exercised,
not merely requested by test parameters. Max scaled reciprocity error
0.00012532190401509655; reevaluation PDF difference remains exactly zero.
Random incidence spans the critical-angle region, but this is not a dedicated
analytic critical-angle sweep. CPU/GPU pointwise discrepancies remain recorded,
with no threshold relaxation. Full integration and other audit gaps remain open.

### Renderer-facing Metal world-frame closure test: failure retained

Added execution of the real `bsdf_diffraction_dielectric_sample`, continuous
`eval`, and singular `delta_mass` wrappers in a rotated shading frame on M5.
It checks event hemisphere, returned eta, finite positive PDF, continuous PDF
reevaluation, and discrete probability agreement against the CPU test's existing
2e-5 mass / 3e-4 PDF gates. This directly exercises the conditional non-straight
closure, not setup allocation or dispatch of the separate straight closure.

`rough_dielectric_metal_world_frame_v1.json` and diagnostic follow-up both FAIL:
13 failures among 8191 valid singular and 52153 valid continuous events. All
13 are singular matched-index events (indices divisible by 24); none reports
bad hemisphere, eta, or finite-value status. Maximum absolute conditional mass
error 0.00051152362721040845. Continuous checks and earlier local checks pass.
Fixtures are saved under `world_case` in diagnostic JSON stdout. Investigate
matched-index conditional residual/mass computation and world-frame rounding;
do not relax thresholds or expose this as validated full shader support.

### Matched-index residual cancellation fixed

Production `diffraction_dielectric_facet_residual` now separates the exact
matched-index straight-through mass before summing other orders. Previously it
summed that near-unit mass with the small other powers and then subtracted from
one. The conditional closure divided the cancellation error by its small
remaining budget. The new mathematically equivalent `(1 - atom) - other` form
matches the conditional sampler, skips redundant zero-order direction/Fresnel
work, and does not clamp probabilities or change test tolerances.

Outside-sandbox M5 `rough_dielectric_metal_stable_residual_v1.json` passes:
world-frame failures 13 -> 0, max discrete probability error
0.00051152362721040845 -> 1.4978421631894889e-6. Valid events: 8191 singular,
52153 continuous. Other declared GPU gates still pass; cross-backend pointwise
boundary discrepancies remain explicitly unresolved.

Recompiled CPU facet, rough dielectric, world-frame closure, and conditional
split tests all pass; see `stable_residual_cpu_regressions_v1.log`. This is a
numerical production fix with CPU and GPU evidence, not completion of missing
node/pipeline integration or rendered feature acceptance.

### Production rebuild after stable residual fix

`ninja -C build/macos_arm64_Release blender` completed with exit 0 after the
matched-index residual change, including regenerated Metal kernel artifacts.
Build log: `/tmp/diffraction_stable_residual_build.log`. Installed selected Fast
application was not replaced. Reviewed actual Glass SVM/OSL entry points and
recorded compatibility requirements in `doc/cycles_diffraction_glass_integration.md`;
node support remains unimplemented rather than silently bypassing existing film,
distribution, tint and caustics controls.

### Distribution implementation prerequisite

Templated local rough dielectric sampling/evaluation on MicrofacetType, default
GGX unchanged. Beckmann uses Cycles' existing VNDF, NDF, and masking functions;
no closure storage increase. This is not wired to renderer closure selection yet.
The existing CPU test now accepts DIFFRACTION_TEST_BECKMANN and checks the matching
analytic distribution. Default GGX regression passes after this refactor.

Beckmann experiment `rough_dielectric_beckmann_v1.log` FAILS 193 sampled energy
bounds, while reciprocity (max 6.2544e-6) and flat-interface equivalence
(max 1.03783e-6) pass. Diagnostics identify existing approximate Beckmann Lambda
slightly negative near its branch boundary, e.g. -6.02167e-5; this makes
value/pdf exceed one. Do not accept Beckmann support until the approximation is
physically bounded and validated. No global microfacet behavior changed.

### Bounded diffraction Beckmann masking

Added diffraction-local `diffraction_dielectric_lambda<m_type>`: GGX unchanged;
Beckmann rational Lambda is bounded below by zero, its physical lower bound,
before constructing G1/G2. Neither BSDF values nor PDFs are clipped, and global
Cycles microfacet behavior is unchanged. CPU Beckmann rough dielectric test now
passes: 193 energy violations -> 0; 20,000 reciprocal pairs, max scaled
reciprocity 6.2544e-6; 19,581 sampled events; 20,000 flat comparisons, max scaled
error 6.15887e-5 against existing unbounded Cycles formula. Existing tolerances
unchanged. Log: `rough_dielectric_beckmann_bounded_v1.log`.

Independent double-precision analytic integral test added at
`tests/performance/cycles_diffraction_beckmann_masking_test.cpp`. 200,001
log-spaced slope cases from 1e-6 to 1e6: unbounded approximation negative in
244 cases; bounded approximation finite and nonnegative throughout; maximum
absolute G1 error 0.00310850024559 versus the erfc/exponential expression,
within the predeclared .005 approximation gate. Maximum scaled Lambda error
0.0051178758534 is reported without calling it exact. Log:
`beckmann_masking_analytic_v1.log`. Beckmann GPU, normalized-measure, closure
selection, and full integration validation remain outstanding.

### Beckmann local math executed on Metal

Runner now accepts `--rough-dielectric --beckmann`, selecting matching template
instantiations in CPU references and actual Metal sample/evaluation kernels.
Reports explicitly distinguish local_distribution=Beckmann from the unchanged
world_distribution=GGX. This is not Beckmann world-closure integration.

Authorized M5 runs, 65,536 inputs, original gates unchanged:
- `rough_dielectric_beckmann_metal_v1.json`, Fast math: FAIL one reciprocity
  fixture (62548): forward 3359.5224609375, reverse 3360.92041015625, scaled
  error .00041594237534622517 > .0003. Energy gate passes and separate-dispatch
  PDF reevaluation is exactly equal. Cross-backend pointwise discrepancies 133.
- `rough_dielectric_beckmann_metal_safe_v1.json`, Safe math: scoped gates pass,
  max scaled reciprocity .00021706966121161171, PDF reevaluation exact, max
  energy-bin difference .000012741995590204169. Cross-backend pointwise
  discrepancies 129 remain outside acceptance scope.

Safe math is a diagnostic candidate only: no renderer compiler mode change,
performance selection, or broad correctness claim. Further geometry precision,
measure validation, and closure/node integration remain required.

### Independent Beckmann directional integration

Extended the existing uniform-solid-angle integration test to select Beckmann
with DIFFRACTION_TEST_BECKMANN. It compares sampled acceptance probability and
energy (total, transmitted, reflected) against directional PDF/BSDF integrals,
adding the known matched-index atom analytically. It does not share the forward
sampler's proposal for the independent integral.

`rough_dielectric_beckmann_measure_1m_v1.log` preserves a failed six-configuration
run: six uncertainty-limit failures (SE > .005), not significant differences
under the unchanged 6-SE + .0005 discrepancy gate. A 10-million/configuration run
was started to reduce uncertainty. These local measure checks do not establish
world-frame or shader/renderer support.

The 10-million/configuration run completed with exit 0, `failures=0` in
`rough_dielectric_beckmann_measure_10m_v1.log` (60 million paired trials total).
All original discrepancy and uncertainty gates pass. No tolerance or estimator
change was made to obtain this result. Approximation bias below those gates is
not ruled out; these results remain scoped to the six tested configurations.

### Beckmann renderer-facing function instantiations

Templated `bsdf_diffraction_dielectric_eval/sample` on MicrofacetType (default
GGX), forwarding the selected distribution into local sampling and evaluation.
No storage field or closure size increase. Existing callers still select GGX;
closure-ID dispatch and shader node selection are not yet implemented.

CPU `beckmann_world_closure_cpu_v1.log`: 96-byte closure, 20,000 singular and
19,562 continuous events, zero failures; max atomic probability error
1.12986e-6, continuous reevaluation exact. Setup allocation tests still pass.

Authorized M5 safe-math `beckmann_world_closure_metal_safe_v1.json` now explicitly
runs both local and world functions with Beckmann: 8191 singular and 55130
continuous world events, zero world failures, max error 2.0269205833756132e-6.
Other scoped gates pass; existing cross-backend pointwise differences persist.
This does not resolve the previously recorded fast-math reciprocity failure,
nor prove full shader dispatch, node support, or rendered integration.

### Beckmann closure dispatch wired

Added a distinct Beckmann diffraction closure ID without enlarging storage.
Templated setup assigns it; the actual BSDF dispatcher selects Beckmann sample
and eval instantiations. Roughness/eta, labels, delta detection/evaluation,
blur, glossy classification, physical-grating detection and guiding continuous
classification include the new ID. It deliberately remains outside generic
microfacet/Glass ranges whose callers cast incompatible storage.

CPU test now selects distribution through setup and invokes actual bsdf_sample
and bsdf_eval/bsdf_eval_delta dispatch with real KernelGlobals. Beckmann run:
1998 events, zero failures (`beckmann_dispatch_cpu_v1.log`). This is a matched-
index rough mixture test, not exhaustive dispatcher coverage or renderer/node
acceptance. Full build launched with log `/tmp/diffraction_beckmann_dispatch_build.log`;
completion must be checked separately. Fast-math reciprocity and node/coating/
multiscattering gaps remain outstanding.

### Dispatcher matrix and full build completed

Full 141-step Blender build after Beckmann dispatch changes exited 0; log
`/tmp/diffraction_beckmann_dispatch_build.log`. Installed selected Fast app was
not replaced.

Expanded actual CPU dispatcher regression over alpha={0,.3} and both incident/
transmitted indices independently in {1,1.5}, covering eight configurations per
distribution. Existing 1e-5 relative sample/evaluation gates unchanged. GGX:
11832 events, zero failures (`ggx_dispatch_matrix_cpu_v1.log`); Beckmann: 11988
events, zero failures (`beckmann_dispatch_matrix_cpu_v1.log`). Both use actual
setup-assigned closure IDs and actual continuous/delta dispatch. At least 500
accepted events required per configuration; observed minimum 922 for GGX and
995 for Beckmann. Incident direction remains fixed; these are not exhaustive
angular, mixture, guiding, or complete path-connection tests.

### Physical parameter conversion for pending Glass node wiring

Added `diffraction_dielectric_parameters` to convert wavelength/pitch/depth in
nanometers and oriented absolute indices into the tested dimensionless Fast
model. Transmission phase uses the scalar relief optical-path difference
2*pi*depth*(n_in-n_out)/wavelength, explicitly angle-independent as an
approximation. Backface reversal swaps indices and reverses phase without
changing binary intensity coefficients. Equal-index relief therefore has zero
physical phase contrast (unlike earlier deliberately nonzero-phase synthetic
matched-index tests). Invalid/overflowing parameters and unsupported order
bounds are rejected before assigning the result.

CPU conversion regression passes 18 wavelength/pitch/depth combinations plus
invalid input checks; `dielectric_physical_parameters_v1.log`. This is shared
parameter preparation only: not yet wired into Blender sockets/SVM/OSL, and
not a Maxwell or coating model.

### Physical input conversion checked against grating equation

Extended the physical parameter test with an independent double-precision
transverse momentum oracle: n_i*wi_parallel+n_o*wo_parallel=m*lambda/pitch,
using n_o=n_i for reflection. Covers 380/550/780 nm, 740/1600 nm pitches,
both indices in {1,1.5}, three oblique incidence directions, reflection and
transmission, orders -5 through 5. No inverse geometry routine is used as the
oracle. Checks propagation/evanescence, hemisphere, unit length, and transverse
components. `dielectric_physical_grating_equation_v1.log`: 723 propagating and
861 evanescent orders, max component error 1.25169754117e-7, zero failures.
This verifies physical direction conversion, not scalar efficiency accuracy,
coatings, multiple scattering, or unimplemented node integration.

### Targeted fast-math precision candidate rejected

Tested explicit componentwise fused multiply-add for transmission inverse
half-vector `eta*wi+wo` to retain cancellation precision. Actual M5 Beckmann
fast-math test still fails the same reciprocity fixture 62548 with unchanged
scaled error .00041594237534622517. The candidate is not selected and its exact
edit was reverted; preserved patch and result:
`beckmann_half_vector_fma_candidate.patch`,
`beckmann_half_vector_fma_metal_v1.json`. Other worktree edits were preserved.
This excludes that isolated arithmetic change as a solution to the known fast-
math failure; safe-math remains only a diagnostic candidate, not a selection.

### Full fast-math reciprocal fixture captured

`beckmann_reciprocity_fixture_metal_v1.json` preserves full material inputs,
randoms and returned direction for case 62548. It is first-order transmission,
ni=1/no=1.5, not reflection. Added a standalone CPU reproducer
`cycles_diffraction_beckmann_reciprocity_fixture.cpp` with per-order root,
power, Jacobian and NDF output (`beckmann_reciprocity_fixture_cpu_v1.log`).
Forward/reverse roots differ in their yz components by about 3.7e-5; the active
forward Jacobian is ~25456.8, amplifying angular roundoff. Both roots are
reported, including the zero-power one. This narrows the next precision work
to reciprocal inverse geometry. A proposed next candidate is symmetric absolute
momentum ni*wi+no*wo instead of ratio-based eta*wi+wo; not yet implemented or
validated. No tolerance change or new correctness claim.

### Symmetric absolute-index geometry candidates not selected

Tested inverse roots from ni*wi+no*wo with absolute grating kick, then extended
that candidate with the absolute-index Jacobian no^2*(-cos_o)/abs(a*b).
Outside-sandbox M5 Fast math still fails fixture 62548: root-only scaled
reciprocity error .00036430439045291401; root+Jacobian .00033604101255223982,
both above unchanged .0003. Baseline was .00041594237534622517.
Results: `beckmann_symmetric_momentum_metal_v1.json` and
`beckmann_symmetric_jacobian_metal_v1.json`. Both scoped tests fail despite
improvement. Exact combined candidate saved in
`beckmann_symmetric_geometry_candidate.patch`; candidate production edits
reverted from pre-experiment snapshots, preserving earlier accepted work.
No performance or correctness selection made.

### Metal safe/fast local sampler timing pilot

Added opt-in CYCLES_DIFFRACTION_TEST_BENCHMARK: nine sequential dispatches after
warmup, actual command-buffer GPU timestamps, same 65,536 seeded inputs and
complete local sampler. Compilation and CPU reference generation excluded;
raw times and median stored in the report. Correctness gates still execute and
still determine exit status. No adaptive sampling involved in this math kernel.

Authorized outside-sandbox sequential M5 pilot:
`beckmann_fast_kernel_timing_v1.json`: median .9981666808016598 ms, scoped
correctness FAIL (known reciprocity case).
`beckmann_safe_kernel_timing_v1.json`: median 1.6277499962598085 ms, scoped gates
PASS. Approximately 63.1% local-kernel overhead. These are single-process-per-
mode pilot measurements, not alternating multi-process statistics or a render
benchmark. No claim of whole-render overhead, universal performance, or final
version selection. The failed fast candidate cannot win on timing alone.

### Precise inverse normalization candidate rejected

Tested Metal precise::sqrt and explicit normalization in transmission inverse
root reconstruction, with otherwise fast math. Actual M5 run
`beckmann_precise_inverse_norm_metal_v1.json` still fails fixture 62548 with
exactly unchanged reciprocity error .00041594237534622517. Candidate reverted;
patch retained as `beckmann_precise_inverse_norm_candidate.patch`.

Timing median .62629164312966168 ms differed substantially from the earlier
~.998 ms fast pilot despite identical failing fixture values. Do not interpret
this unpaired pilot as a speedup; it reinforces the need for alternating,
repeated process measurements before performance selection. Neither candidate
satisfies the correctness gate.

### Reciprocal momentum contraction fix passes initial regressions

Extended the absolute-index inverse roots/Jacobian candidate with explicit
rounding of both products before their sum (`volatile` scalar intermediates).
This prevents fused contraction from choosing a different rounded product when
incident/outgoing endpoints are swapped. Only this small momentum construction
changes; global Metal fast math remains enabled. Candidate retained in worktree
for further validation, not yet final performance selection.

Authorized M5 fast-math runs: Beckmann
`beckmann_uncontracted_momentum_metal_v1.json` passes scoped gates, max scaled
reciprocity 4.975886807567694e-5 (previously failing .00041594237534622517).
GGX `ggx_uncontracted_momentum_metal_v1.json` passes, max scaled reciprocity
3.8564857474719212e-5. Separate-dispatch PDFs remain exact; world-frame checks
pass. Cross-backend pointwise differences remain and are not newly accepted.
CPU local reciprocity/energy/flat-limit tests pass both distributions:
`uncontracted_ggx_cpu_v1.log`, `uncontracted_beckmann_cpu_v1.log`.
Absolute-index Jacobian finite-difference/measure tests, broad build, and paired
performance comparison remain required before declaring this candidate selected.

### Absolute-index Jacobian independent finite differences

Added DIFFRACTION_TEST_ABSOLUTE_INDICES to the existing transmission geometry
test, evaluating the new Jacobian with both absolute indices and kick scaled by
two. This preserves the independent double-precision forward oracle geometry
while exercising the no^2 Jacobian factor. Original finite-difference gates
unchanged. `absolute_jacobian_geometry_v1.log`: 64864 finite-difference checks,
max relative error .000272931, zero failures; 131 near-index-matched checks,
max relative error 3.35958e-6. The three previously documented unresolved
grazing roots remain counted, not accepted as solved. Roots in this specific
test still use the prior ratio API; it validates the new Jacobian, not the new
absolute-momentum inverse root implementation independently.

Recompiled 10-million/configuration independent directional integrals for both
GGX and Beckmann and launched sequentially. Logs:
`absolute_momentum_ggx_measure_10m_v1.log` and
`absolute_momentum_beckmann_measure_10m_v1.log`. Completion pending; do not infer
success from launch or partial output.

The GGX integral run completed with exit 0 and failures=0 (60 million paired
trials, original gates unchanged). The same sequential process has advanced to
Beckmann; its result is still pending.

### Candidate probability integrals and inverse roots verified

Beckmann 10-million/configuration directional integral completed with exit 0,
failures=0. Together with GGX, 120 million paired trials passed original gates;
see absolute_momentum_{ggx,beckmann}_measure_10m_v1.log.

Absolute-index geometry mode now calls the actual new inverse-root path using
symmetric momentum, as well as the new Jacobian. `absolute_inverse_geometry_v1.log`
passes: 103303 propagating inputs, 88194 regular recoveries, 38440 two-root
cases, 64864 Jacobian checks, 131 near-matched checks. The three unresolved
grazing cases remain explicitly counted. No cutoff or tolerance changed.

Started sequential fast/safe/safe/fast process timing runs for the current
candidate; each retains scoped correctness results and nine warmed GPU timings.
Directory `reciprocal_momentum_timing_v1`; final summary pending.

All four timing processes completed and passed their scoped correctness gates.
Fast process medians .9539999883/.9432499937 ms; safe
1.6555416514/1.6547083214 ms. Mean of the two process medians: fast ~.948625 ms,
safe ~1.655125 ms (~42.7% lower execution time with fast math). Results support
retaining the targeted momentum fix for further integration rather than globally
switching to safe math. This is still a local-kernel benchmark, not whole-render
performance or final full-feature version selection. Raw results and summary
preserved in `reciprocal_momentum_timing_v1`.

### Reciprocal momentum fix retained after production build/dispatch checks

Production Blender rebuild completed with exit 0 after the momentum change;
log `/tmp/diffraction_reciprocal_momentum_build.log`. Recompiled actual dispatcher
matrix tests against this revision pass: 11832 GGX and 11988 Beckmann events,
zero failures; logs `reciprocal_momentum_{ggx,beckmann}_dispatch_v1.log`.

Retain targeted momentum rounding and absolute-index geometry for subsequent
integration. The supporting evidence includes unchanged-gate local Metal tests,
CPU reciprocity/flat limits, independent Jacobian checks, directional integrals,
dispatch tests, full build, and alternating kernel timing runs. This is a local
implementation choice, not the requested final whole-renderer version selection.
Cross-backend boundary discrepancies, unresolved grazing cases, Glass coating/
multiscattering/tint/node work, broad rendered validation and coherent-object
support remain open. Installed selected Fast app was not overwritten.

### Per-lobe suppression prerequisite for Glass caustics controls

Added disabled_lobes to diffraction dielectric storage, using packed_float3
for tangent so CPU closure remains 96 bytes. Setup accepts optional reflect/
transmit mask (default enables both), suppresses straight-atom allocation when
transmission is disabled, and rejects both-disabled setup. Evaluation/delta
queries return zero for disabled side; sampling rejects that side rather than
renormalizing its energy into the other lobe. Conditional non-atom budget remains
unchanged. Metal manual test initialization now explicitly zeros the mask.

CPU world-closure regression extended to query each sampled event with its own
side disabled, and with only the opposite side disabled. Disabled side returns
zero; retained continuous value/PDF remain exactly unchanged. Existing 20,000
singular / 19,056 continuous checks pass, storage 96 bytes:
`dielectric_lobe_masks_cpu_v1.log`. Masked setup allocation, sampler rejection,
Metal masks and actual Glass caustics wiring still require validation; this
entry does not claim those integration paths complete.

### Lobe suppression GPU and allocation validation

Extended GGX Metal world-frame test: for every accepted event, disable its
side, require zero eval/PDF and delta mass, then resample with identical randoms
and require LABEL_NONE with zero eval/PDF. `dielectric_lobe_masks_metal_v2.json`
passes on M5 Fast math, zero world failures (8191 singular, 52153 continuous).
The first attempt failed compilation from missing metal::any qualification in
the test only; preserved `dielectric_lobe_masks_metal_v1.json`, fixed qualifier.
No numerical gate changed.

CPU setup checks verify reflection-only matched-index setup succeeds with one
slot and retains weight 1-atom, and both-disabled setup returns false without
allocating or consuming slots. Existing closure regression passes:
`dielectric_lobe_allocation_cpu_v1.log`. These checks do not yet validate actual
Glass caustics socket/control wiring, disabled-lobe guiding behavior, or full
rendering; those remain outstanding.

### Separate reflection/transmission tint setup

Added `bsdf_diffraction_dielectric_setup_tinted`: identical spectral weights
share existing setup; different weights create side-filtered closures and the
transmission-only straight atom. Preflight reserves all required slots (up to
three) before either side allocates. Dark sides omitted through existing Cycles
sample-weight policy; no storage growth. Reflection-only setup no longer sets
SR_BSDF_HAS_TRANSMISSION. Shader compiler reservation must account for three
slots when this helper is wired.

New CPU setup test verifies per-side and atom weights, masks, unchanged state
on insufficient capacity, shared identical-weight allocation, and reflection-only
runtime flag. `dielectric_tint_setup_cpu_v2.log` passes, 96-byte closure. First
compile failed because closure_sample_weight takes a mutable Spectrum reference;
fixed by using local copies for the preflight query. No failed test was accepted.
Mixture/dispatcher sampling, GPU setup and shader-node integration remain pending.

### Weighted tint evaluation through actual dispatcher

Added `cycles_diffraction_dielectric_tint_dispatch_test.cpp`, using real CPU
KernelGlobals and actual bsdf_sample / bsdf_eval / bsdf_eval_delta dispatch.
For smooth/rough and matched/unmatched interfaces, sample directions from the
untinted setup, evaluate the weighted sum of every tinted closure, and compare
componentwise to the untinted weighted sum multiplied by the correct side's
non-gray Spectrum tint. Includes the separate straight atom and smooth orders.

`dielectric_tint_dispatch_cpu_v1.log`: 39561 GGX and 39985 Beckmann events,
zero failures under a predeclared 2e-5 scaled component tolerance. This verifies
weighted BSDF evaluation, not the surface shader's stochastic closure selection,
guiding MIS probabilities, GPU tint allocation, or node integration. Those
remain distinct outstanding checks.

### Actual surface-shader mixture evaluation

Extended the tint dispatcher test to include the actual surface_shader header
and call surface_shader_bsdf_eval_pdfs / surface_shader_bsdf_eval_delta. Compare
returned mixture PDF to an independently accumulated sample-weighted sum of
individual dispatcher PDFs and accumulated BsdfEval to the previous weighted
BSDF reference. Exercises real closure picker and validates index/random-remap
bounds; it does not yet test picker frequencies or use picked directions for
an independent mixture-distribution test.

`dielectric_tint_surface_mixture_cpu_v1.log`: 39515 GGX and 39992 Beckmann
accepted events, zero failures, unchanged 2e-5 scaled gates. Initial standalone
compile lacked ARM NEON declarations required by transitively included image
headers; added platform-guarded arm_neon.h to the test. Production shader code
was not altered to bypass dependencies. Guiding/RIS, actual GPU tint allocation,
node wiring and rendered-scene acceptance remain outstanding.

### Real tinted setup allocation executed on Metal

Added eight deterministic ShaderData allocation cases inside the GPU world-frame
validation kernel (outside the separately timed local-sampler kernel): distinct
tints, identical tints, reflection only, insufficient slots, transmission only,
both dark, equal-index zero-phase pure straight transmission, and unequal
indices. Checks returned allocation status, closure counts/remaining capacity,
and the three weights in the distinct-tint matched-index case.

Authorized M5 Fast math GGX and Beckmann runs both pass all scoped gates:
`dielectric_tint_allocation_metal_v1.json`,
`beckmann_tint_allocation_metal_v1.json`. These exercise actual setup and allocator
on GPU, not CPU-mirrored allocations. Full sampled surface mixture/guiding GPU
validation and shader-node wiring remain outstanding. Cross-backend pointwise
boundary discrepancies are still reported and unresolved.

### Selected tinted closure sampling and MIS reconstruction

The surface-mixture regression now samples the actual closure returned by
surface_shader_bsdf_bssrdf_pick with its remapped random value. Checks the
sampled-closure skip path `_surface_shader_bsdf_eval_mis` against full mixture
queries, weighted BSDF reference, and independently summed PDF. Adds stratified
selection frequencies (predeclared <=2-count error per closure). Existing
2e-5 value/PDF gates unchanged. Replaces the previous nonzero-event floor with
complete attempt accounting because side-filtered proposals legitimately reject.

`dielectric_tint_picked_mixture_cpu_v1.log` passes: GGX 21973 accepted/18027
rejected; Beckmann 22177/17823; exactly 40000 attempts each. Maximum selection
count error .378418. Roughly 45% rejection in these fixtures is an explicit
performance concern for differing tints, not hidden or called optimized.
Shared-tint path does not use the extra side split. This check validates
mixture consistency, not independent integral unbiasedness, GPU surface MIS,
guiding or rendered scene quality.

### Joint tinted proposal prototype reduces rejected attempts

Replaced differing-tint side-split setup with a joint untinted directional
proposal and side-dependent evaluated Spectrum tints. Optional auxiliary storage
holds parameters/tints; main closure uses a union and stays 96 bytes. Shared
weights retain inline storage. Parameter access, roughness/blur, singular eval
and continuous eval/sample paths account for optional data. Full capacity is
preflighted; optional block consumes one closure slot. This introduces pointer
lifetime/copy requirements that must be audited before acceptance.

CPU actual selected-mixture regression passes: GGX 39603 accepted/397 rejected,
Beckmann 39989/11, each 40000 attempts, versus previous ~45% rejection. All
unchanged value/PDF/selection gates pass (`dielectric_joint_tint_mixture_cpu_v1.log`).
Actual M5 GGX allocation cases pass (`dielectric_joint_tint_allocation_metal_v1.json`).
CPU layout/weight test v1 failed an exact float average comparison; v2 uses the
same 1e-6 weight tolerance as other weight checks and passes. Both logs retained.
This is a prototype pending shader-data copy/lifetime audit, GPU tinted sampling,
Beckmann GPU setup, broad build and timing. Do not claim renderer readiness or
whole-render speedup from acceptance counts.

### Joint tinted GPU sampling and lifetime audit progress

Inspected bdpt_reverse_pdf and light-path hit setup in integrator/bidirectional.h:
they create local ShaderData and reevaluate the shader, rather than persisting
raw closure pointers for those paths. This is partial lifetime evidence only,
not proof for every shader-data copying/merging/guiding path.

GPU setup test now samples the actual joint auxiliary-data closure and an inline
untinted reference with identical randoms, checking labels, directions, PDFs
and side-tinted values (32 draws for each matched/unmatched rough interface).
Beckmann M5 Fast math `joint_tint_sampling_metal_v2.json` passes all scoped gates.
First attempt failed compilation from missing metal::fract/length qualifiers in
test code; v1 failure retained, qualifiers corrected. Auxiliary allocation
coverage is included in this run. GGX equivalent run, full lifetime audit,
smooth tinted GPU events, broad build and actual renderer integration remain.

### Smooth and rough joint-tint Metal checks

Expanded the fixed GPU allocation suite from eight cases to sixteen by repeating
all cases with zero roughness. Joint versus untinted sampling comparisons now
exercise both smooth discrete and rough continuous events for matched/unmatched
indices. GGX and Beckmann M5 fast-math runs pass:
`joint_tint_smooth_rough_ggx_metal_v1.json`,
`joint_tint_smooth_rough_beckmann_metal_v1.json`.
The random local-math gates also remain passing; cross-backend pointwise
sensitivity remains reported and unresolved.

Started production rebuild with log `/tmp/diffraction_joint_tint_build.log`;
completion not yet verified. Lifetime review still partial; existing pointer-
free storage comment is now stale for the optional tinted variant and must be
updated after this build without conflating inline and auxiliary storage.
Production rebuild completed with exit 0. Updated storage comments afterward
(comment-only) to state auxiliary ShaderData ownership and joint proposal;
removed stale pointer-free/side-filtered wording. Installed app not replaced.

### Closure filtering integration fix

Lifetime/preparation audit found actual semantic incompatibility: diffraction
is in the broad glossy category, so generic FILTER_CLOSURE_GLOSSY removed its
transmission as well. Added explicit diffraction handling in camera-ray closure
filtering before generic category checks. Reflect/transmit filters set per-lobe
masks; both remove the closure/sample weight. Straight-through closure responds
only to transmission filtering. Optional auxiliary-data flag is preserved.

CPU regression checks sequential filters and straight closure semantics while
retaining existing actual selected-mixture/MIS tests; all pass in
`dielectric_tint_filter_cpu_v1.log` (39603 GGX /39989 Beckmann accepted events).
Also exercises roughness accessor/no-op blur at current alpha; this is not yet
a changing-roughness blur test. GPU filter execution and actual rendered closure
filter controls remain unverified. Audit did not identify relocation in the
inspected preparation loop, which mutates entries in place; broader lifetime
review remains incomplete.

### Auxiliary blur mutation and Metal lobe filtering

CPU tint-mixture test now calls actual bsdf_blur with alpha=.7, verifies the
roughness accessor returns .49, and checks both auxiliary tint spectra remain
unchanged; restores alpha before the existing mixture tests. All pass in
`joint_tint_blur_cpu_v1.log` (39603 GGX/39989 Beckmann accepted events).

Metal fixed setup cases now copy a live auxiliary closure within its owner's
lifetime, filter reflection, check auxiliary flag preservation, filter the
remaining side and require closure removal/zero sample weight. Straight closure
survives reflection filtering and is removed by transmission filtering.
`joint_tint_filter_metal_v1.json` passes on M5 Fast math with Beckmann; other
scoped gates remain passing. These exercise the helper invoked by preparation,
not a rendered camera-ray filter scene or GPU blur dispatcher.

Production build launched, `/tmp/diffraction_tint_filter_build.log`; completion
pending. No installed application replacement.

### Guiding classification audit and CPU transmission query

Confirmed prior production tint/filter build completed exit 0.
Metal guiding glossy-product cast is guarded by microfacet_only across all
continuous closures; diffraction makes that false and uses general proposal
paths. No cast fix was necessary or made there.

CPU surface guiding's fully_opaque test used generic IDs and missed diffraction
transmission. Added an explicit diffraction transmission query respecting the
per-lobe mask and straight closure; wired it into that check. This preserves the
existing CPU guiding restriction on transmitting surfaces (uses ordinary BSDF
sampling there); it does not implement new CPU transmission guiding. The Metal
proposal path remains separate and unaltered by this change.

CPU query/filter/selected-mixture regressions pass in
`dielectric_transmission_query_cpu_v1.log`. No rendered guiding support claim;
actual GPU mixture/guiding scenes, node integration and all earlier gaps remain.
# Coated surface-mixture selection and MIS

`cycles_diffraction_coated_surface_mixture_test.cpp` exercises the actual CPU
surface selector, delta/continuous mixture evaluators and sampled-closure skip
path through `_surface_shader_bsdf_eval_mis`. It compares those results with
explicit weighted per-closure sums and with a separately allocated untinted
coated pair multiplied by the appropriate spectral side color.

Coverage includes GGX/Beckmann, smooth/rough surfaces, zero/nonzero phase,
matched/entering interfaces, and an additional diffuse material. A second set
changes the incident direction after allocation, leaving the original selection
estimates intact. This tests direction-dependent evaluation without confusing
selection weights with physical energy. Stratified selection counts agree with
the stored proposal weights to within one sample. Auxiliary blur and lobe
filter checks are included.

`coated_surface_mixture_cpu_v2.log` passes 640,000 selection attempts: 315,579
accepted/4,421 rejected GGX events and 318,079 accepted/1,921 rejected Beckmann
events, with zero failures. Rejected directions are retained in the accounting;
they are not resampled. The initial unchanged-direction run is preserved as v1.
This validates these CPU mixture paths, not full GPU guiding, BDPT connections,
shader-node integration or rendered scenes. Those requirements remain open.

# Coated companion closure and combined setup

The rough dielectric closure now supports an optional coated auxiliary record
without enlarging its inline record or the existing uncoated tint record.
Coated evaluation returns physical continuous energy and the per-facet
conditioned proposal PDF. For smooth events, delta probability and physical
power are returned separately; the dispatcher no longer assumes they coincide.
Uncoated behavior retains the previous weight/PDF convention.

Combined setup allocates a coated straight atom for matched-index transmission
and its companion lobe, with separate spectral tints and side masks. It validates
parameters and preflights all required storage (four slots for the matched pair,
two for an unmatched companion). Zero thickness delegates to uncoated setup.
Selection estimates retain support and do not become physical closure weights.
The atom and main lobe independently update their material roughness under blur.

Expanded CPU dispatcher tests cover zero/nonzero phase, smooth/rough surfaces,
matched/entering/leaving interfaces and allocation exhaustion. They pass 23,529
GGX and 23,955 Beckmann events in `coated_dispatch{,_beckmann}_cpu_v2.log`.
The uncoated dispatcher regression passes 11,832 events. Metal checks 32 rotated
world-frame combined setups per distribution and validates sample/evaluation
agreement, including distinct delta value and probability; both
`coated_combined_closure_{ggx,beckmann}_metal_v1.json` pass. The Blender build
passes in `/tmp/diffraction_coated_dispatch_build_v1.log`.

Full surface-mixture MIS/selection tests are still needed for this pair.
Shader/node/OSL wiring, multi-scattering, transport renders and quality/timing
selection remain incomplete. The installed selected application is unchanged.

# Dedicated coated straight-through renderer closure

Added `CLOSURE_BSDF_DIFFRACTION_COATED_STRAIGHT_ID` with an auxiliary material
record and explicit quadrature work count. Its stored weight is the material
color, not a frozen atom integral. Sampling and delta evaluation recompute the
physical atom for the current incident direction. The conditional direction
PDF remains a unit delta mass. Roughness blur updates its material parameters;
subsequent evaluation uses the updated distribution. Selection weight is only
an importance estimate, with a small support floor.

Wired sampling, delta/continuous evaluation, labels, roughness/eta queries,
delta classification, blur, transmission queries, camera lobe filters and
physical-grating detection. It remains a singular closure and is excluded from
continuous guiding. Its main record fits ShaderClosure and uses one auxiliary
slot. The setup preflights allocation and validates relevant atom parameters.

`coated_atom_dispatch_cpu_v2.log` passes actual CPU dispatcher tests for GGX
and Beckmann: two incident directions, sample/delta agreement, zero continuous
density, wrong-direction rejection, dynamic blur, filtering and allocation
exhaustion. The first test incorrectly required an exact label and rejected
Cycles' valid background-transparency modifier; that failed log is retained.
The corrected test requires singular transmission and forbids reflection while
allowing the existing modifier. Metal setup/evaluation/filter checks pass in
`coated_atom_closure_{ggx,beckmann}_metal_v1.json`; these are not full Metal
dispatcher or rendered-material tests. The 141-step Blender build passed in
`/tmp/diffraction_coated_atom_dispatch_build_v1.log`.

This closure is not exposed in shader nodes yet. The corresponding coated
continuous renderer closure, combined setup, quality/performance choice and
full guiding/BDPT/render validation remain required. No installed app was
replaced or goal completion claimed.

# Combined local coated mixture reference

Added a local mixed-measure sampler/evaluator to verify the coated atom and
continuous response together before renderer closure integration. It uses the
quadrature atom as the discrete BSDF mass, a bounded 5%-95% atom proposal for
rough matched-index interfaces, and the per-facet-conditioned continuous
proposal. Only proposal probabilities are bounded; physical values are not
clamped or renormalized. Keeping both proposal branches alive prevents a
rounded approximate atom of zero or one from removing real scattering support.
Smooth surfaces use the direct coated facet distribution.

The probability/energy integration test passes all six configurations at ten
million trials each for both GGX and Beckmann (120 million combined trials):
`coated_mixture_{ggx,beckmann}_measure_10m_v1.log`. The probability target adds
the discrete proposal probability, while energy targets add the physical atom
mass; treating these as interchangeable would be incorrect. The per-sample
weight bound is 20 for this proposal because either branch can have probability
0.05. This is a derived bound, not a relaxation of the integral/precision gates.

Metal executes 128 mixed-model cases per distribution, checking continuous
sample/eval agreement, exact rough atom direction/mass/probability and finite
weights. Both `coated_mixture_{ggx,beckmann}_metal_v1.json` pass alongside the
existing continuous-sampler tests. Blender builds successfully in
`/tmp/diffraction_coated_mixture_build_v1.log`. These are local-model checks,
not full coated renderer closures. Guiding requires separate discrete and
continuous closures, so the renderer layout is intentionally still unchanged.
Direction-dependent atom evaluation, storage, blur/filter behavior and the
shader/node/OSL paths remain to be implemented and tested.

# Coated atom quadrature and targeted integration

The outstanding matched-index continuous-probability precision checks pass at
40 million trials each for GGX and Beckmann. Standard errors are 0.00346331
and 0.00413179, below the unchanged 0.005 ceiling. Sampled/integrated
probabilities are 0.907310/0.907950 and 0.986076/0.986474. The targeted test
selects the previously unresolved cosine 0.9, equal-index configuration;
the other configurations remain covered by the ten-million runs. Earlier
precision failures are retained, not overwritten. See
`coated_continuous_{ggx,beckmann}_matched_40m_v1.log`.

Added bounded power-of-two VNDF quadrature for the coated straight atom, with
exact zero-film and smooth limits. It canonicalizes tangential direction signs
to preserve the reversal symmetry of the symmetric anisotropic distributions.
There is no chosen production quality default. The 24-case CPU pilot compares
16/64/256/4096 points with 500,000 sampled reference facets per case, including
thin/thicker films and two incident angles. Maximum absolute errors for GGX
are 0.0147094/0.00771048/0.00200903/0.000215811; Beckmann errors are
0.00600492/0.00163590/0.000588144/0.000130170. Only the highest work bound has
an accuracy acceptance gate in this exploratory test; all reversal, range and
zero-film checks pass. See `coated_atom_quadrature_cpu_v1.log`.

This is an approximate atom integral, not yet a full coated closure. Broader
film/roughness coverage, Metal accuracy/timing, closure storage and mixture
weights, shader hookup and rendered tests remain necessary.

Metal template instantiation and 128 atom range/reversal/zero-film checks pass
for both distributions (`coated_atom_{ggx,beckmann}_metal_v1.json`), alongside
the continuous coated sampling gates. These atom checks are not a CPU/Metal
accuracy comparison or an atom timing benchmark. The Blender build also passed
in `/tmp/diffraction_coated_atom_build_v1.log`; the installed app is unchanged.

# Coated rough continuous proposal

The facet sampler and rough evaluator now accept coating parameters. A shared
facet nonstraight-budget helper keeps sampling and residual evaluation
consistent. The new coated continuous sampler draws the existing VNDF and
conditions order choice separately at each facet; evaluation applies that
facet's budget to the proposal PDF at every inverse root, while retaining the
physical BSDF value. There is no rejection loop or global constant-atom
assumption. Existing uncoated calls retain zero-film defaults.

The Metal harness has a `--coated` local-sampler mode. Both GGX and Beckmann
65,536-case runs pass the strict same-direction PDF, same-GPU reevaluation,
reciprocity and energy-bin gates (`coated_continuous_*_metal_v1.json`). The
uncoated regression also passes. World-frame closure checks remain uncoated;
the report now labels that explicitly. The Blender build passed in
`/tmp/diffraction_coated_continuous_build_v1.log`.

Uniform-solid-angle probability/energy integration is NOT fully verified.
One-million trials per configuration failed precision gates (three GGX, six
Beckmann); ten-million trials per configuration still fail one precision gate
for each distribution. These are preserved in
`coated_continuous_{ggx,beckmann}_measure_{1m,10m}_v1.log`. The remaining
failure is the high-variance matched-index probability integral near a fold;
acceptance thresholds were not relaxed. Further independent integration is
required. The discrete coated atom, rough closure storage/mixture, shader-node
integration, multiscattering and rendered validation are still incomplete.

# Coated dielectric facet coefficients

The Fast dielectric facet power/residual functions now accept an absolute film
IOR and film thickness in vacuum wavelengths. They use the previously tested
lossless Airy helper for reciprocal reflection/transmission budgets. The film
header is registered with the kernel build. Zero thickness retains the ordinary
Fresnel path. Matched exterior indices use an angle-dependent straight-through
budget, with stable residual accounting. This is a scalar relief approximation
with a planar coating response, not a corrugated-film Maxwell solution.

`coated_facet_cpu_v1.log`: 20,000 cases and 116,890 propagating order pairs,
minimum residual 0.000111942, maximum reversal error 4.47e-8 and flat-film
error 2.081e-5, all gates passed. The unchanged uncoated double-reference test
also passes (`coating_uncoated_regression_v1.log`). The Metal rough-dielectric
harness now checks coated residuals and selected forward/reverse order powers
across all 65,536 inputs, separately from its uncoated rough closure checks.
`coated_facet_metal_v1.json` passes, including the strict same-direction PDF
gate for that uncoated closure. The Blender build passed in
`/tmp/diffraction_coated_facet_build_v1.log`.

These new coating arguments are not yet stored or sampled by the rough closure
and are not exposed in shader nodes. Its constant matched-index atom split is
invalid for a coating; routing these coefficients into it unchanged would be
incorrect. Rough coating sampling, multiscattering and node/render integration
remain required. No installed application was replaced or final performance
selection made.

# Compensated transmission folds and axial Jacobian

A conditional axial-norm optimization was tested and reverted. It used the
direct subtraction away from relative cancellation below 0.01 and retained
the robust perpendicular-momentum expression near the axis. GGX, Beckmann and
the inverse-fold fixture test passed unchanged gates. Six sequential isolated
M5 Beckmann processes compared it against the unconditional v3 correction;
each measured nine warmed GPU dispatches. Control process medians were
0.685417, 0.679167 and 0.830208 ms; candidate medians were 0.680458, 0.683250 and
0.678708 ms. The median-of-medians difference is only about -0.72%, with
substantial control spread. This does not establish a useful speedup. Keep the
unconditional v3 expression; the extra branch is not justified by this evidence.
All results, source hashes and the rejected patch remain in
`conditional_axial_timing_v4/` and `conditional_axial_candidate_v4.patch`.
No whole-render performance claim or installed-build change was made.

`cycles_diffraction_inverse_fold_test.cpp` now independently checks the two
worst fixtures at the stored direction. A double-precision angle solver solves
`v.e(theta)=kick`; finite differences of its normals on two outgoing spherical
tangents estimate the inverse solid-angle derivative without calling the
production inverse or Jacobian. It freezes the actual rounded center momentum
and does not round the perturbed reference momentum back to float. This is a
check of the continuous derivative used by the evaluator, not a derivative of
the discontinuous floating-point program. At angular steps 2e-9 and 1e-9 the
maximum relative discrepancy is 1.482e-6 (0.000149%), with the unchanged new
test's declared 0.001 relative gate passing. See
`compensated_inverse_fold_v1.log`. An initial compile required adding the
missing standard initializer-list header. The original forward-facet test and
its larger error remain intact. This closes the targeted inverse-Jacobian
question for these fixtures; it does not prove that all rounded sampling bias
is absent or replace integration/render checks.

Follow-up geometry diagnostics now preserve the two worst finite-difference
fixtures in `compensated_fold_jacobian_fixture_v1.log`. Cases 93627 and 144504
have fold coefficients about 0.001274 and 0.004850. Double arithmetic on the
printed stored vectors shows a 1.14% and -0.97% difference between the magnitude
from the discriminant and that from the original generating facet (analysis
JSON alongside the log). Direction squared-norm errors are only of order
1e-7, but cancellation makes them significant here. The forward finite
difference is centered on the original facet, while the corrected evaluator
recovers geometry from the rounded outgoing direction. This identifies a
conditioning issue in interpreting that comparison; it does not waive the
existing gate or prove that the resulting sampling error is harmless. A
targeted inverse-map differential check remains needed. The test now emits
complete worst-case fixtures without changing its acceptance conditions.

Captured complete input fixtures for CPU/Metal PDF discrepancies in
`rough_density_fixtures_metal_v1.json`. The large discrepancies occur at
transmission folds: float `|ni*wi+no*wo|^2-kick^2` cancellation can remove
positive roots. A reflection-discriminant experiment did not change these
failures and was reverted; its patch and result remain in the output directory.

The retained candidate computes product residuals and compensated sums near
folds (relative discriminant below 0.001). Its Jacobian recovers the fold
coefficient magnitude from that discriminant and uses
`|cross(axis,momentum)|/|cross(axis,h)|` for the other coefficient, avoiding
axial subtraction. Both follow from `momentum=a*h+kick*e`, with `h.e=0`.
The 200,000-case double-reference test passes; naive arithmetic lost the sign
or rounded to zero in 7,751 tested cases. No density clamping or tolerance
relaxation was added.

The final candidate's 65,536-case GGX Metal run has zero same-direction density
differences, versus 33 previously. Sequential Beckmann timing runs likewise
have zero versus 43 previously. Same-direction CPU/Metal comparisons at the
existing 0.0002 scaled threshold are now mandatory failures in the harness.
Different sampled float directions still have large PDF differences near folds;
this is explicitly separate from testing identical directions. One branch
mismatch per distribution also remains in the seeded suite.

Independent probability/energy integration passes for GGX and Beckmann at ten
million trials per configuration, six configurations each:
`compensated_fold_axial_{ggx,beckmann}_measure_10m_v3.log`. The earlier one-million
pilot failed three standard-error precision gates; it is retained. Geometry
finite differences pass their unchanged 1.5% gate, but the maximum Jacobian
relative error increased from about 0.0273% to 1.275%, and three unresolved
grazing roots remain. These are limitations, not evidence of full numerical
closure. The Blender build passed (`/tmp/diffraction_fold_axial_build_v3.log`).

`fold_axial_timing_v3/` contains four sequential M5 Beckmann processes in
candidate/baseline/baseline/candidate order, with nine warmed dispatches each.
Candidate medians: 0.739792 and 0.668333 ms; baseline: 0.658750 and 0.650417 ms.
Mean of medians is 7.56% higher for the candidate, with substantial candidate
spread. This pilot does not establish final rendering overhead or a fastest
version. Source hashes, raw timings and all intermediate results are preserved.
The installed selected Fast application has not been replaced. Shader hookup,
coatings, multiscattering, full transport renders and coherent-object support
remain unfinished.

# Joint tint auxiliary-storage lifetime audit

The mixed-allocation exhaustion regression now also runs inside the real Metal
ShaderData allocator. Sequential M5 runs passed for GGX and Beckmann:
`joint_tint_storage_metal_v1.json` and
`joint_tint_storage_beckmann_metal_v1.json`. Each retains the 65,536-case sampler,
reciprocity and world-frame checks. Both report zero world-frame failures and
zero same-GPU PDF reevaluation failures. This does not resolve the previously
recorded CPU/GPU pointwise density differences (78 GGX, 132 Beckmann), which
remain visible in the reports. No render timing or node-support claim follows
from these allocator checks. The latest transmission-query Blender rebuild
completed successfully in `/tmp/diffraction_transmission_query_build.log`;
the installed selected application was not replaced.

Inspected the surface closure preparation, BDPT reverse evaluation, cached light
vertex reconstruction, and volume phase copying paths. BDPT reconstructs local
ShaderData and reevaluates the surface shader, including in
`bdpt_setup_light_vertex`; it does not persist this closure's auxiliary pointer
in the light vertex. Volume phase copying explicitly selects volume-scatter
closures. Volume closure merging is a volume evaluation operation; its array
compaction does not provide a general surface-closure copying contract.

Extended `cycles_diffraction_dielectric_tint_test.cpp` with two simultaneously
live, differently tinted joint closures, different roughness, a subsequent
inline allocation, exhaustion and a failed allocation. Earlier auxiliary
parameters and colors remain intact. `joint_tint_storage_cpu_v1.log` passes,
with the closure still 96 bytes. This is a focused CPU storage regression and
source audit, not proof of rendered BDPT, OSL or Metal node integration. A copied
ShaderData containing auxiliary pointers still requires explicit relocation;
the implementation must not introduce such copies.

The new Glass guiding render also completed: `glass_delivery_guided_v1`,
56.7392 s with the same fixed 256 samples and resolution. Its PNG was inspected.
The three-image review is `tests/output/diffraction/glass_delivery_gallery_v1.html`.
OSL runtime checking is still separate.

OSL runtime smoke also passed (`glass_delivery_osl_v1`): CPU, 32 fixed samples,
7.9633 s, saved scene/EXR/PNG. The noisy preview was inspected and shows all
three grating materials. This checks closure registration and node parameter
wiring; it does not establish CPU/Metal numerical equivalence.

## Bounded Glass scene cost comparison

`glass_delivery_cost_v1/report.json` records four sequential Metal jobs,
256 fixed samples, adaptive and denoising OFF: enabled 11.4769 s, disabled
2.4215 s, disabled 2.4532 s, enabled 3.9076 s. The last enabled render is
60.32% slower than the disabled mean. Large first-run variation precludes
a universal performance claim; retain all four measurements. All output
pixels are finite. This does not compare equal variance or identical images.
No further parameter/timing search was started.

## Incident Fresnel reuse

The facet sampler and residual mirror sum now reuse incident Fresnel reflectance
across orders of the same facet, including the lossless coating response.
The evaluated powers and probability definitions are unchanged. The focused
Metal GGX coated/surface-mixture regression passes in
`coated_surface_fresnel_reuse_metal_v1.json`: zero failures, zero same-direction
CPU/Metal PDF differences and zero GPU reevaluation failures. Previously recorded
differing-direction conditioning issues remain (156 pointwise differences and
one branch mismatch); they were not hidden or reclassified.

Full Blender build, including Metal kernels and OSL, passes in
`/tmp/diffraction_fresnel_reuse_build.log`. No speedup is claimed from this source
change without measurement. Existing Glass scene timings and images precede
this change and retain their original provenance. The harness now explicitly
records whether the 256-setup/four-trial surface-mixture block was requested.

## Refreshed 72-render suite

`fast_delivery_suite_v4` completed all 72 sequential Metal jobs on build SHA256
`0e557a703cdfa8c7ea8ebed28f6e81122f310974494f6cc40a522b790fa3f469`.
All use 1024 fixed samples, adaptive sampling OFF, denoising OFF. Artifact hashes
and settings were verified by the final analyzer; `fast_delivery_suite_v4_review`
contains the gallery, full numerical comparisons and manual visual audit.
Eight analytic checks pass (maximum mean error 0.000405366); 52 Fast/reference
comparisons document approximation differences; 12 appearance images were
inspected. Maximum angular partition discrepancy is 0.000117127, a single-seed
diagnostic. No PT brightness normalization was applied. All four indirect
images remain noisy and do not meet clean-presentation requirements.

The failed v3 run is retained: the covered-disc generator selected any node with
a Diffraction Pitch input and therefore selected the newly extended Glass node.
The v4 generator explicitly selects the original Glossy node for replacement,
preserving the clear Glass cover.

This completed suite does not certify the missing cross-object coherence,
remaining shader integration, multiscattering, or untested pipeline features.
The later Glass generated-tangent attribute fix is separately validated and
is not silently attributed to this frozen suite binary.

The Glass default-tangent fix now requests generated coordinates before graph
default links are added. Build passes (`/tmp/diffraction_default_tangent_build.log`).
`glass_default_tangent_v1` compares implicit and explicit Geometry Tangent
connections on Metal with identical seeds and 128 fixed samples: mean absolute
pixel difference 3.67182e-8, maximum 6.10352e-5, finite pixels, test passes.
This closes the missing input-attribute declaration; it does not address the
other outstanding material or coherent-transport requirements.

## Metallic conductor-aware diffraction

Metallic GGX/Beckmann now expose grating coverage, pitch, depth and duty in SVM
and OSL. The new closure retains native physical-conductor or F82 Fresnel and
thin-film payloads. It implements reciprocal conservative order powers, full
inverse-root PDF evaluation, smooth delta MIS, roughness/blur, guiding
classification, and partial-coverage mixtures. Multi-GGX is explicitly rejected
when diffraction is enabled; full multiscattering remains unfinished.

Build passes (`/tmp/diffraction_metallic_node_build_v1.log`). The expanded CPU
regression passes 31,462 accepted events including full and half coverage;
flat-profile maximum scaled BSDF error is 6.44633e-6, reciprocity error 2.92063e-6
(`conductor_closure_cpu_v2.log`).

`metallic_delivery_v2` contains actual finite-pixel Metal PT (256 samples), BDPT
(128), guiding (256) and CPU OSL (32) renders, saved scenes and a gallery. All
previews were inspected and retain grating bands, metallic color, and coating
response. These tests do not establish complete energy convergence, overhead,
or all pipeline support. The v1 scene setup failure remains preserved.

The 72-render suite remains tied to its preceding frozen build. These subsequent
Metallic tests are separately versioned; no historical image was relabeled as
being rendered by the new build. Principled integration, general cross-object
coherence, multiscattering and remaining pipeline/accuracy work remain open.

## Packaged bounded delivery (2026-09-27)

`build/diffraction_delivery_20260927/Blender.app` is a separate runnable package,
with the final binary hash stated above. Clean-environment package tests pass:
Metal CD/DVD at 1024 fixed samples, 37.6342 s including preparation, and CPU OSL
Principled at 64 fixed samples, 11.8122 s. Both previews were visually inspected.
Final-binary Principled BDPT and guiding also pass at 256 samples (52.7271 s and
22.4705 s). Adaptive sampling and denoising were disabled in these tests.

`delivery_20260927/index.html` is the combined gallery. Its manifest records
package verification, image provenance and known incomplete scope. App resource
hashes and every gallery target were checked. The account used 36% at delivery,
against the 33% starting point and 43% ceiling. No new parameter sweep was run.
The original goal is incomplete; this package does not implement general
cross-object coherence or the remaining unsupported material/pipeline paths.

## Pipeline parameter-limit regression resolved

Compositor initialization exposed a 41-parameter Principled GPU signature against
a 37-parameter metadata limit. The limit is now 41; full build passes. Updated
package: `build/diffraction_delivery_20260927_pipeline/Blender.app`, SHA256
`dfc78c6c36d3e0873b9d50c77410ac4bc5bd7811c6ea52c0d928a78b14c3b435`.
Twelve fixed-128-sample Metal jobs cover AOV, motion blur, DOF and volume, each
in PT/BDPT/guiding. All return finite, nonempty output; all AOV exports pass the
0.75 scalar expectation. All beauty previews were inspected. DOF blur remains
subtle and volume previews noisy; this is not complete pipeline certification.
The ordinary-material Eevee linking check also passes. Failed fixture/API
attempts remain in v1/v2; final evidence and gallery are pipeline_delivery_v3
and pipeline_delivery_v3_review. Broader material and coherent transport gaps
remain open. Usage is 37%, below the agreed 43% ceiling.

## Glossy distribution correction

The earlier Glossy diffraction component silently used GGX for Beckmann and
other selections. GGX/Beckmann now use the shared reflective closure with unit
facet Fresnel, preserving spectral color, anisotropy, rotation, coverage and
explicit surrounding-medium IOR. Multi-GGX/Ashikhmin diffraction is explicitly
rejected. This is not complete support for those unfinished combinations.

The build passes including all Metal libraries and OSL. Extended anisotropic
CPU tests pass 63,276 accepted events, plus a surrounding-medium grating-equation
check and old/new scalar GGX agreement. Runtime support guards pass. Metal PT,
BDPT, guiding and CPU OSL previews are finite and visually inspected; saved
scenes, hashes and timings are in `glossy_delivery_v1`. The separate package
`build/diffraction_delivery_20260927_glossy/Blender.app` passes a clean-environment
OSL smoke; its executable hash is `902188d468f6eb31aef396a27e57866d344f773090d14e310b72995000c9a265`.
General cross-object coherence, transmitting Principled, multi-scattering and
the remaining original scope remain open. Usage remains 37% against the 43% ceiling.


## 2026-09-27 final packaged candidate and Refraction

Refraction GGX/Beckmann now wired in SVM/OSL and numerically checked. PT512, BDPT128, guided128, CPU OSL32 and packaged clean-environment OSL32 completed, finite and visually inspected. Details: cycles_diffraction_refraction_integration.md. Selected package: build/diffraction_delivery_20260927_final/Blender.app. Fast remains default. General coherence, transmitting Principled and multi-scattering remain absent; full goal incomplete. Usage at handoff38%, below43% budget ceiling.


## Exact final-package native material checks

Four sequential clean-environment Metal PT renders completed at 128 fixed samples, adaptive sampling and denoising OFF. Glass, Glossy, Metallic and reflective Principled all produced finite images and were visually inspected. Glass remains noisy. Evidence: tests/output/diffraction/final_package_materials_v1 (manifest, logs, scenes, EXRs, PNGs and gallery). Binary hash matches the delivered candidate. These targeted checks do not replace broader convergence or missing-feature requirements.


## Refraction passivity check

The expanded CPU regression passes 108 configurations × 2,048 trials (221,184
trials), with 147,984 accepted events. GGX/Beckmann, front/back incidence,
roughness 0/0.2/0.6, depth 0/250/1000 nm and incident cosine 0.05/0.5/1 are
covered. Rejected events contribute zero and remain in the denominator. Maximum
mean throughput and maximum individual throughput are both 1; no negative or
nonfinite throughput occurred. The previous flat, reciprocity, atom and independent
Fourier tests still pass. Log: tests/output/diffraction/refraction_kernel_cpu_v3.log.
This checks passivity in the Cycles closure throughput convention, not a new
Maxwell comparison or exhaustive angular quadrature. The source explains the
G2/G1 masking bound; no render code or acceptance tolerance was changed.


## Remaining transport source/research audit

Confirmed native multiscattering energy_scale uses planar lookup tables and is not applied by the diffraction reflection kernel; simply removing node guards is not a valid implementation. Reviewed the final 2026 Wave Tracing paper author page and public implementation for the remaining cross-object coherence architecture. Findings, primary links and acceptance requirements are recorded in cycles_diffraction_remaining_transport_design.md. No new feature is claimed.


## Updated optimized package

Exact zero-power order pruning is implemented and CPU regressions plus the full Metal build pass. One flat Glass fixture measured a 29.61% faster warm render, with mean absolute pixel difference 4.61141e-8; preparation made the first optimized run slower. Updated package: build/diffraction_delivery_20260927_optimized/Blender.app. Its matching executable, sources, OSL shaders and all three Metal libraries are hash-verified by the reusable packaging script. See cycles_diffraction_flat_profile_optimization.md. No missing feature is claimed as complete.


## Standard-Fresnel Principled transmission implemented

Implemented SVM/OSL joint dielectric grating transmission for GGX Principled under
explicit standard-Fresnel constraints. Numerical sample/evaluate and native flat
limit checks pass (30,852 samples; 15,504 flat comparisons). Full build passes.
Current package and constraints are documented in cycles_diffraction_principled_transmission.md.
The original full-scope goal remains incomplete.

Layered integration completed: Metal PT256 (40.26s) and CPU OSL64 (25.02s), including preparation; both finite and visually inspected. All nine validation assertions pass, including tiny positive weights; expected shader errors retain process exit1. Evidence: principled_layered_delivery_v1 and principled_transmission_validation_v2. Current package: build/diffraction_delivery_20260927_layered/Blender.app, SHA256 13ea1fc0d37a08e08b7941006b0af056763a54a8ad24989ae8791708c57b7bf8.


## Tinted Principled transmission implemented

GGX transmission now supports constant and linked Specular Tint, with native F0-to-white angular response retained. Smooth/rough sample-evaluate, native flat-limit, reciprocity and order-power tests pass. Shared Glass payload growth was corrected by separating generalized storage (ordinary tint64/coating80/generalized96 bytes). Final standalone PT/OSL checks pass. PT/BDPT/guiding/OSL before that storage correction are separately preserved. Package: build/diffraction_delivery_20260927_tinted_v2/Blender.app, SHA256 6e85ee28bf0522c5d5fac651166f346b7eba88e69dbceaeb9e4f656a5a54068a. Details and gallery: cycles_diffraction_principled_tinted_transmission.md and principled_tinted_delivery_v2/index.html. Film/dispersion/thin-wall transmission, multiscattering and cross-object coherence remain open.


## Final bounded dispersion delivery, 27 September 2026

Selected package: build/diffraction_delivery_20260927_dispersion/Blender.app, SHA256 592b7f3588ece72befd4fd1118ea848150092733dfd5f28c597d893789cb8dc0. Full build and matching-resource packaging pass. Numerical regression: 562390 events, 186054 independent double flat comparisons, zero failures. Final clean-environment sequential Metal PT/BDPT/guiding and CPU OSL renders pass; ten support-validation assertions pass (deliberate-error process exits1). See cycles_diffraction_delivery_report_20260927.md and principled_dispersion_delivery_v1/index.html. Full original goal remains incomplete: general cross-object coherence, multiscattering and transmitting film/thin-wall remain open. No further broad benchmark sweep was started. Last account usage40%, below43% ceiling.


## Physical transmitting Principled film extension

Current package build/diffraction_delivery_20260927_film/Blender.app, SHA256 b46643a7be1c36c3ccdc5c9667e816a903ef1134e442813b0bc8b9104c3327d5. Reuses coated Glass kernel for white unlinked Specular Tint and zero unlinked dispersion. Generalized film/thin wall stay explicit errors. Full build, 23934 dispatch events, independent quarter-wave AR limit and 73884 Airy comparisons pass. Four clean-environment PT/BDPT/guiding/OSL renders pass; finite and visually inspected. Fourteen support assertions pass including linked thickness (deliberate-error process exits1). Original coherence/multiscatter/generalized-film/full-pipeline scope remains incomplete. See cycles_diffraction_principled_film.md.


## Generalized transmitting film extension

Current package build/diffraction_delivery_20260927_generalized_film/Blender.app, SHA256 8b383bfaab2e82bb5136b3dcbb985decef983b906516816d2f57217ae40cc0cc. Thin film now combines with Specular Tint and dispersion; near-matched indices use the coated atom when native tint correction is disabled. Extreme tint is clamped for passivity. Payload remains96bytes. Build, 562862 sample/evaluate events, 370862 reciprocity tests, 186328 independent double-reference checks and bare dispersion regression pass. Four standalone Metal PT/BDPT/guiding and CPU OSL renders pass, finite and visually inspected. Fourteen support assertions pass. General coherence, multiscattering, thin-wall and exhaustive pipeline scope remain incomplete.


## Cross-object coherence acceptance references

Added independent scalar two-source reference calculations: 15 checks pass, including complex spectral quadrature and incoherent limits. Six actual separate-source Blender scenes saved. Two Metal PT controls at64 fixed samples, adaptive/denoisingOFF, show unchanged reference-phase response within1.19209e-7; manifest explicitlyFAIL/unimplemented. No feature support or fake texture render claimed. Selected generalized-film package unchanged. See cycles_coherence_acceptance.md. General coherent transport, finite-aperture geometry, source-size and full transport tests remain open.


## Bounded Fast native orders and current presentation render

Selected package build/diffraction_delivery_20260927_bounded/Blender.app, SHA256 9e66f2c9970be3f87e5ecf1247f71244efd3119fd50fb9bed2709b84fb8c67b8. Fixed large valid pitches losing dielectric closures by using symmetric ±256 Fast tail support. Passive omitted facet-power bound0.6333%, measured maximum0.1192% across144 configurations; not a pixel-error bound. 2030 sample/evaluate and1010 reciprocal checks pass; generalized-film562862-event regression passes. Full build and five sequential package cases pass (Metal PT/BDPT/guiding, CPU OSL). Warm100µm fixture1.327325→0.711151s,46.42% faster; first-render preparation slower. Fresh1024-sample CD/DVD image passed and was visually inspected. No broad suite rerun or full original-goal completion claimed.
