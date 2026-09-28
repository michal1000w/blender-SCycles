# Fast reflective/transmissive scalar candidate

This is a numerical experiment in `tests/performance`, not a connected renderer
feature or a completed Fast/Realistic UI. It extends the fast reflection work
without copying the expensive Maxwell cache into the default path. The current
GPU benchmark sources remain unchanged while this experiment is developed.

## Model and intended scope

Use exact grating directions with vacuum wavelength and the two exterior IORs.
Assign direction-independent reciprocal flux budgets R and T, with R+T <= 1.
Normal-incidence interface Fresnel coefficients are a possible approximate
choice; they do not reproduce angular Fresnel reflectance. For an opaque metal,
T=0 and R is the normal-incidence conductor reflectance. For a lossless
dielectric interface, T=1-R. Absorbing finite layers need separate qualification.

Nonzero reflected orders use the existing binary phase-screen power with phase
2 pi h n_i (cos_i+cos_o)/lambda. Transmitted orders use the Fourier power of a
thin binary phase plate with an angle-independent phase delay, for example
2 pi h (n_ridge-n_groove)/lambda. This is a scalar approximation. See the
[binary phase-grating scalar formulation](https://doi.org/10.1364/JOSAA.439269).
The budget allocation and treatment of closed channels below are our model
choices, not claims from that paper.

For duty fraction f and integer order m, the phase-plate power is

- m=0: 1 - 4 sin²(phi/2) f(1-f).
- m!=0: 4 sin²(phi/2) [sin(pi m f)/(pi m)]².

An evanescent order carries no outgoing power. Do not renormalize the surviving
transmitted orders: that produces a direction-dependent normalization which can
break reverse-path consistency. Instead assign the unallocated part of R+T to
the specular reflection port. This is intentionally approximate near cutoffs and
may overestimate reflection. The unsent fraction 1-R-T is absorption.

## Why test this allocation

For nonzero reflected orders, each Fourier power is bounded by
4 [sin(pi m f)/(pi m)]². Summing this bound over all nonzero integers gives
4 f(1-f) <= 1, so their total power cannot exceed R even though the reflection
phase varies with the outgoing angle. The complete constant-phase transmission
Fourier series has total power one; removing evanescent terms cannot increase
its sum, so transmitted power cannot exceed T. Therefore the residual specular
power is nonnegative without clipping or changing individual order efficiencies.

Every nonzero reflection pair has the same order and cos_i+cos_o under reversal.
Every transmission pair has the same order and phase under reversal, with the
exterior IORs exchanged. The remaining specular reflection reverses the incident
tangential momentum on the same side: all sums map order m to -m, and binary
Fourier powers are even in m. Thus its residual is reciprocal too. These are
flux-port statements. A Cycles closure must still use the radiance/importance
eta factors, wavelength flags, and discrete mixture MIS correctly.

## Evidence required before integration

`cycles_diffraction_fast_interface_test.cpp` checks residual positivity over
random profiles, both-side direction and power reversal, Snell's law in the flat
limit, total internal reflection, and inverse-CDF frequencies against separately
evaluated event powers. The candidate rejects unsupported very large order
counts instead of silently truncating them. Passing these tests would establish
mathematical properties of this approximation, not agreement with RCWA.

Next checks must compare errors against the existing RCWA references, implement
and test the production sampling/evaluation and node export paths, render
transmission fixtures on Metal, and measure overhead. The candidate currently
provides no coherent amplitudes, rough transmissive facets, material-dispersion
table support, or production UI. Those cannot be inferred from an intensity-only
power test. In particular, optional cross-object coherence remains separate
unfinished transport work.

## First executed numerical result

`tests/output/diffraction/fast_interface_candidate_v1` records the test output,
exit status and hashes of the candidate, test, shared Fourier utility and test
executable. Compiled with C++20 and `-O2`, the test exited zero with 130,082
reciprocal pairs (54,671 transmitted), maximum power reversal error 2.22633e-6,
maximum direction reversal error 9.58163e-6, minimum residual 7.06568e-9 and
maximum inverse-CDF frequency error 9.24579e-6. No production kernel changed for
this experiment. This is float arithmetic evidence over the stated randomized
domain and explicit flat/TIR cases, not an exhaustive proof or a Metal result.

## Maxwell efficiency comparison

`fast_interface_reference_v1` contains 18 actual reference comparisons: 740 nm
pitch, 150 nm depth, float duty 0.41, ridge IOR 1.5, upper/groove IOR 1, substrate
IOR 1 or 1.5, wavelengths 450/550/650 nm, and incidence 0/30/60 degrees. Oblique
cases use 45-degree azimuth. The comparisons use all orders and unpolarized
powers; nothing is normalized to force agreement. Both fixed N64 and N128
Maxwell solves are retained. Their maximum L1 drift is 4.27506e-6; this is a
two-resolution comparison, not a modal convergence certification.

The candidate's L1 efficiency error ranges from 0.06703 to 0.29863, with median
0.10173 and maximum individual-order error 0.14847. The worst L1 case has lower
IOR 1.5, wavelength 450 nm and 30-degree incidence: approximate total reflection
0.0858 versus about 0.0190 in the N128 reference. Even the normal-incidence air
exterior case at 450 nm has approximate reflection 0.0550 versus 0.0035. Energy
and reciprocity are therefore not evidence of correct reflection/transmission
partitioning. These errors are explicitly retained for judging fast-mode visual
quality and distinguishing it from the realistic option.

## Metal execution

`tests/metal/cycles_diffraction_test.py --fast-interface` expands the same
candidate header and runs an actual Metal fast-math kernel. The host fixture
uses 16,384 randomized profiles, directions and sample variates, including
absorbing budgets and both reflective/transmissive cases. It checks CPU/GPU
directions, powers, probabilities, residuals, sampled branches and GPU reverse
powers. This tests arithmetic and sampling, not rendering speed or integration.

`fast_interface_metal_v1.json` records the Apple M5 result: zero failures, zero
branch mismatches, maximum CPU/GPU component error 1.21891e-5 and maximum GPU
reciprocal-power error 2.84612e-6. No renderer or UI integration is implied by this
result. The new candidate can now proceed to production closure integration and
representative transmission renders, with its measured efficiency bias visible
in the evaluation rather than hidden by the conservation tests.

## Production integration in progress

The mathematical interface has been copied into
`intern/cycles/kernel/closure/bsdf_diffraction_interface.h`. The smooth diffraction
closure now has a fast path using the same singular closure classification and
delta-mixture/reverse-PDF machinery as the physical path. Its parameters use
Cycles' existing extra-closure storage allocation; the shared ShaderClosure
capacity was not increased. The physical path initializes the new pointer to
null and keeps its cache evaluation. The reserved cache handle -2 selects the
fast model; -1 still means preparation failed.

SVM data and the scene compiler are wired to the fast parameters. OSL has a new
`diffraction_fast` builtin; the existing `diffraction_smooth` signature remains
unchanged for external shaders. The node shader selects the appropriate builtin.
The fast model uses normal-incidence interface R/T budgets, with optional mean
normal-incidence Beer attenuation for an absorbing ridge layer. That attenuation
is an additional approximation requiring its own tests. Material-index tables
are currently rejected in fast preparation instead of being silently ignored.

The Blender node source exposes Quality = Fast / Realistic. New nodes initialize
to Fast; stored quality zero means Realistic so older saved physical nodes keep
their model. Physical fixtures explicitly request Realistic and can now select
Fast with `--quality FAST`; their reports and reload assertions record the choice.
`cycles_diffraction_quality_test.py` checks the default, both-mode serialization,
and loading an actual legacy scene without modifying it.

These are source changes, not a completed runtime validation. The full build is
recorded in `fast_interface_integration_build.log`. Its first attempt caught
missing explicit Metal address-space qualifiers on candidate references; those
were added to the production header and the restarted build passed the initial
Metal compilation steps. Full build/install, quality serialization, SVM/OSL,
Metal PT/BDPT/guiding renders and new benchmarks are still required.

The earlier benchmarked binary and Cycles sources were preserved in
`fast_scalar_benchmarked_snapshot` before integration edits. The existing 7.59 s
PT and other published timings describe the earlier Glossy reflection model,
not this newly integrated reflective/transmissive node. They must not be used as
performance measurements of the new Quality = Fast implementation.

## First integrated Blender results

The full build, a follow-up incremental build and installation completed
successfully. The follow-up was necessary because a grazing regression was fixed
while the main build was running. All subsequent renderer tests used the final
binary and a checked kernel-source manifest in `fast_quality_integration_smoke_v1`.

The grazing regression exposed six failures among twelve explicit zero-order
mirror/index-matched transmission cases. Recovering z from 1-x*x-y*y made valid
channels disappear near grazing, or made their reconstructed direction fail the
atomic matching tolerance. Zero-order mirror and index-matched transmission now
use their exact vector expressions. Atomic lookup uses transverse optical
momentum, unit-length checks and the outgoing hemisphere instead of reconstructing
an ill-conditioned cosine. Its momentum tolerance still distinguishes neighboring
orders under the model's order-count limit. Tests reject non-unit inputs and a
nearby direction that is not an atom.

`fast_grazing_before_fix` retains the failures. `fast_grazing_probability_v3.json`
records all twelve final sample/probability checks passing. The production CPU
randomized regression still passes all 130,082 reciprocal pairs. The production
Metal query regression (`fast_interface_atomic_query_metal_v3.json`) passes
16,384 cases, including the explicit grazing cases, with no branch mismatches;
maximum GPU reciprocal-power error is 5.36442e-7.

The Blender quality test passed: new node default Fast, both Fast and Realistic
persist through save/reload, and the two nodes in the actual legacy disc scene
load as Realistic. The legacy file's hash remained unchanged. The new binary hash
is `c5b72cfd29936c09a6cc09145f9ebc78183dc13acdb4d406cd2bf25599fa8cec`.

The integrated Fast dielectric furnace rendered on Metal PT at 1,024 samples.
Mean RGB was (1.000065326, 1.000048573, 0.999594635), maximum error 0.000405365
against unit energy, passing the existing 0.005 smoke tolerance. This conservation
check does not remove the separately measured efficiency bias against Maxwell.

The integrated Fast CD/DVD scene also rendered at 960 by 691 and 1,024 samples
with Metal BDPT and guiding enabled, adaptive sampling and denoising disabled.
The broad CD and narrower DVD spectral bands are visible in the reviewed preview;
band edges still show spectral sampling noise. Scene, EXR and separate preview
hashes were verified. The 104.877-second elapsed render call includes preparation
and is not an isolated warmed benchmark. New-node benchmark comparisons, OSL
rendering, representative transmission images and the remaining full suite are
still required; these successful smoke tests do not establish full completion.

## Backend checks and expanded validation

`fast_quality_backend_checks_v1` completed both requested backend checks. The
Fast dielectric furnace on CPU OSL had maximum unit-energy mean error
0.0004053589. The Realistic flat-metal furnace on Metal had maximum analytic
mean error 0.0003678226. Both passed the existing smoke tolerance; neither is an
accuracy certificate for arbitrary relief geometry.

The frozen integrated Fast suite `fast_quality_full_suite_v1` runs 17 fixtures
under each of PT, BDPT, guided PT and BDPT with guiding. It uses 1,024 fixed
samples, adaptive sampling and denoising off, and separate EXR-derived preview
files. The first 50 completed artifacts were audited in
`fast_quality_full_suite_partial_review_v1`. All six available analytic controls
passed. The angular-region sums agreed with the lossless spectral reference to
about 2.4e-6 for unmixed materials and 1.18e-4 for mirror mixtures. These are
single-seed partition diagnostics, not convergence confidence intervals. The
suite must finish before its final artifact audit is reported.

The CD/DVD PT preview shows broad CD and narrower DVD bands. The BDPT indirect
preview still has substantial spectral noise at 1,024 samples. It is retained
as evidence, not presented as a finished high-quality showcase.

`cycles_diffraction_transmission_display.py` adds a white-slit presentation
fixture with independently calculated first-order centreline wavelength markers.
Its marker positions use the grating equation and the finite camera/source
distances. They do not predict diffraction efficiency. Initial Fast and Realistic
save/reload checks passed; rendering and visual review remain outstanding.

`cycles_diffraction_integrated_benchmark.py` prepares new-node benchmarks with
one warm-up and three measured renders per transport/treatment. Its zero-relief
control preserves the integrated node, graph, optical constants and tangent.
The control changes paths and appearance, so the comparison measures workload
cost rather than equal-image convergence. It must run alone on the GPU. Earlier
Glossy-prototype timings are not substituted for these measurements.

## Closure-storage regression discovered during review

The Fast closure allocates one extra ShaderClosure slot for its scalar parameters,
but DiffractionSmoothBsdfNode initially did not include that slot in its graph
storage estimate. The eight-distinct-grating lossless furnace in
`cycles_diffraction_closure_storage_test.py` reproduced the resulting lost energy:
`fast_closure_storage_before_fix` returned mean RGB
(0.750034992, 0.750021730, 0.749681609), maximum unit-energy error 0.250318391.
It failed the 0.01 tolerance. Distinct pitches prevent graph deduplication; eight
closures need sixteen slots. The original single-node and mirror-mixture suite
does not cover this failure. This failure must be fixed and retested before
selecting the implementation or reporting final timings.

The original integrated Fast suite completed all 68 renders and its full artifact
audit in `fast_quality_full_suite_review_v1`. Eight analytic controls passed;
maximum mean-energy error was 0.000405365867. All eight angular partitions were
present, with maximum absolute partition error 0.000117127027. The 52 reference
comparisons explicitly describe Fast approximation differences. Eight appearance
renders require visual/physical review. The BDPT-guided indirect preview was also
reviewed and remains noisy; its 296.868-second process duration includes
preparation and is not a warmed benchmark. The subsequent storage regression
failure is separate and is not erased by these passes.

The storage estimate now includes one extra slot when `use_fast_model` is true.
Build and installation completed (`fast_closure_storage_build.log`,
`fast_closure_storage_install.log`). The same eight-grating regression then
passed on CPU and Metal: maximum unit-energy errors 0.0004080743 and 0.0004080622,
respectively. Both use 256 fixed samples with adaptive sampling and denoising
off. The pre-fix failed artifacts remain intact. The corrected binary SHA-256
is `4d119ea6afb4f7d379f76374dca9d0d2525ddc6719f09f770066ecc29d7980e9`.

Only `intern/cycles/scene/shader_nodes.h` and the binary differ from the frozen
68-render suite manifest; GPU kernel source is unchanged. The refreshed manifest
is `fast_quality_storage_fixed_provenance.json`. Subsequent benchmarks must use
this corrected build, not the earlier binary.

## Corrected integrated-node benchmark

`fast_quality_integrated_benchmark_v1` completed all eight sequential treatments
outside the sandbox and passed its artifact, source, saved-scene, per-run setting
and seed checks. It uses the corrected binary, Apple M5 10-core GPU, 960×691,
1,024 samples, adaptive sampling and denoising off, persistent data, one warm-up
and three measured renders per treatment.

| Transport | Fast diffraction median (s) | Zero-relief Fast node (s) | Difference |
|---|---:|---:|---:|
| PT | 8.302 | 8.351 | -0.58% |
| BDPT | 18.417 | 18.377 | +0.22% |
| Guided PT | 21.265 | 21.137 | +0.61% |
| BDPT with guiding | 38.300 | 38.446 | -0.38% |

The zero-relief control retains the same spectral node and optical constants;
it is not a non-spectral ordinary-material baseline. The paths differ. These
measure workload cost, not equal-image or equal-noise convergence. Differences
this small in three runs do not establish a statistically significant speed
advantage or penalty. No old Glossy-prototype measurements are substituted.

The corrected build's transmission presentation was reviewed and its EXR,
preview and blend hashes checked. The independently calculated marker equation
residual is 1.11e-16. This establishes marker construction, not radiometric
accuracy. All eight original full-suite appearance previews were also reviewed:
CD/DVD bands are consistent across transports; indirect images remain noisy,
with PT especially sparse. Brighter finite-sample BDPT/guiding images were not
classified as incorrect from their brightness differences.

The corrected eight-grating storage stress test also passed with Metal BDPT and guiding together (`fast_closure_storage_after_fix_bdpt_guided`): maximum unit-energy mean error 0.000408062901 at 256 fixed samples. The render process exited successfully.
