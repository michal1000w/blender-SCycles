# Native diffraction multiscattering: bounded GGX reflection probe

Status: derivation and host-only feasibility measurement, **not enabled renderer support**.

For the unit-reflectance, reflection-only native GGX grating, let
`E(w)` be the hemispherical albedo of the existing single-scatter BSDF at
incident direction `w`. Unlike planar GGX albedo, `E` depends on both incident
azimuth and elevation, `alpha_x/y`, wavelength/pitch, depth/wavelength, and
duty. It can be measured without an inverse diffraction solve: draw a visible
GGX facet and one native diffraction order; a valid outgoing direction
contributes `(1 + Lambda_i)/(1 + Lambda_i + Lambda_o)`, and an invalid one
contributes zero. This is exactly the native `f*cos/pdf` ratio for unit facet
reflectance. The host probe in
`tests/performance/cycles_diffraction_multiscatter_prototype.cpp` implements
this measurement with a fixed low-discrepancy sequence.

With exact `E`, define `A(w)=1-E(w)` and `Ebar=integral(E(w)*mu/pi, hemisphere)`.
The proposed extra lobe, expressed as `f*cos_o`, is

```
L_ms(w_i,w_o) = A(w_i) * A(w_o) * mu_o / (pi * (1-Ebar)).
```

Its hemispherical integral is `A(w_i)`, so the ideal white furnace sums to
one. Dividing by `mu_o` shows reciprocity immediately. The limit `Ebar=1`
has zero extra lobe. For a passive conductor, a conservative spectral factor
can be `Fss^2*Ebar/(1-Fss*(1-Ebar))` for a spectral `Fss` in `[0,1]`;
this is bounded by one and is one for a perfect reflector. The native
single-scatter Fresnel remains in its existing order sum. This factor is a
declared average-Fresnel approximation, not a microscopic multiple-bounce
solution or a conductor measurement. For exact `E`, passivity follows because
the single-scatter spectral albedo cannot exceed the unit-reflectance `E`
and the extra spectral albedo cannot exceed `1-E`.

For sampling, choose the existing visible-facet/order proposal with probability
`1-p` and the deficit proposal `q_ms=A(w_o)*mu_o/(pi*(1-Ebar))` with probability
`p`, for example `p=A(w_i)`. Both sample and evaluate must return the same
mixture density `(1-p)*q_single+p*q_ms`. The existing single branch can retain
null events; its PDF already represents the unconditional facet/order proposal.
The deficit proposal can use cosine-hemisphere rejection with acceptance
`A(w_o)`. If rejection is capped at `K` draws, its accepted PDF is instead
`A(w_o)*mu_o/pi * (1-Ebar^K)/(1-Ebar)` and the remaining mass is null. The
physical BSDF value does not change with this cap. A production implementation
needs the same directional cache in sampling, evaluation, and adjoint paths.

The zero-depth path should keep the original native energy-preserved GGX
carrier exactly. Replacing its compensation with an approximate albedo cache
would regress the required zero-depth limit.

## Feasibility measurement

Apple arm64 CPU, optimized standalone translation unit, `alpha=(.25,.4)`,
`lambda/pitch=.45`, `duty=.42`; each error is against a separate 4,096-sample
native facet/order estimate on an 8×16 direction grid. These are finite-sample
estimates, not strict error bounds.

| Cache | Facet/order samples per node | Depth/lambda | Build time | Max albedo difference | Max estimated white-furnace excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4×8 (128 B) | 128 | 0.4 | 0.8 ms | 2.79% | 2.79% |
| 8×16 (512 B) | 256 | 0.4 | 3.6 ms | 1.20% | 1.20% |
| 8×16 (512 B) | 512 | 0.4 | 8.9 ms | 1.11% | 1.11% |
| 8×16 (512 B) | 256 | 0 | 0.6 ms | 0.21% | 0.21% |

The additional lobe's reciprocity is algebraic, but the cache currently
underestimates `E` for some directions, producing a white-furnace value above
one. The 8×16 construction also costs thousands of native order evaluations
per closure per shading point; these CPU timings do not establish acceptable
GPU cost. No cache reuse across material instances, texture values, or sampled
wavelengths exists. Keep the MultiGGX diffraction guard until a cache build
outside the shading hot path or a validated cheaper directional model is
available. A dielectric needs separate two-sided directional albedos,
reflection/transmission coupling, and the correct refractive transport measure;
the one-sided formula above cannot be applied to it directly.

## Narrow scene-time route for constant inputs

The existing `DiffractionManager` builds and uploads a cache from
`DiffractionSmoothBsdfNode::prepare` before parallel shader compilation. Its
current payload is a physical smooth-grating response, with a different key
and kernel layout; it cannot be reused as the albedo table. The same lifecycle
could host a **separate** GGX albedo buffer and handle: register a constant
material request during scene preparation, deduplicate exact keys, upload one
direction-by-wavelength array, and pass the resulting handle through the
SVM/OSL closure setup to a new compensated conductor closure. Sampling and
evaluation would query the same buffer. The handle must remain stable until
the next successful scene update, as existing diffraction handles do.

The table key must cover distribution, both roughnesses, pitch, depth, duty,
surrounding IOR, spectral domain and spectral refinement tolerance, angular
resolution, quadrature count, and algorithm revision. Fresnel parameters may
be kept outside the geometry table if the declared spectral average-Fresnel
factor above is used; they must be in the key if Fresnel is folded into the
table. A normal or tangent **value** need not be keyed because lookup occurs
in the local grating frame, but a varying tangent still has to define that
frame consistently. The actual sampled wavelength is continuous, so a single
wavelength slice is insufficient. An 8×16 direction grid at 16 wavelength
slices is 8 KiB per unique constant material before metadata; adaptive
wavelength refinement and independent error checks are needed around order
cutoffs. This is an implementation estimate, not evidence that 16 slices pass.

The ordinary Metallic/Glossy/Principled diffraction inputs are SVM node
inputs and may be linked to textures or procedural outputs. Scene preparation
cannot know their per-hit values. Such linked roughness/profile/medium inputs
require a separate validated runtime strategy or an explicit compile error for
the multiscattering choice; substituting the single-scatter closure would
silently change the requested model. Zero depth may dispatch to the native
energy-preserved carrier exactly, since the grating has no nonzero orders.
This route covers only constant-input reflective GGX; it does not complete
Beckmann, dielectric, transmission, coatings, or linked-input support.

## Higher-resolution convergence check

A subsequent host-only sweep uses double accumulation and a 65,536-sample
reference at each of the same 128 validation directions. The depth/lambda 0.4
results were:

| Cache / samples per node | Build time | Max albedo difference | Max furnace excess |
| --- | ---: | ---: | ---: |
| 4×8 / 128 | 0.4 ms | 2.75% | 2.75% |
| 8×16 / 512 | 5.5 ms | 1.00% | 1.00% |
| 16×32 / 1024 | 41.6 ms | 0.93% | 0.35% |
| 32×64 / 2048 | 317.5 ms | 0.42% | 0.38% |

These are single host measurements, not GPU benchmarks. Refinement is not
monotonic in estimated excess and does not establish passivity. The reference
is still a finite deterministic quadrature, not a rigorous bound; only one
roughness/profile/wavelength combination is tested. The zero-depth checks are
also retained in the raw output, but production must preserve its native
zero-depth carrier rather than use this approximate table.

All algebraic reciprocity and capped-proposal mass checks passed. The executable
explicitly distinguishes those checks from reported furnace error: exit zero
is not a physical acceptance gate. Source hash, exact compile command and raw
stdout are preserved in
`build/tests/performance/multiscatter_convergence_v1/results.json`.
The renderer remains unchanged and MultiGGX diffraction remains unsupported.
A production route still needs a validated directional approximation and reuse
outside the shading hot path; simply increasing resolution is insufficient.

## Conditional integration of diffraction orders

The prototype now sums every discrete order on each sampled facet, including
zero-order residual and below-surface null mass. This is the conditional
expectation of the previous facet-plus-order estimator; it changes quadrature,
not the target single-scatter albedo. The original sampled-order implementation
is retained as the 65,536-sample reference instead of comparing only against a
higher-resolution invocation of the new estimator.

For the same depth/lambda 0.4 profile:

| Cache / samples | CPU build | Max albedo difference | Max furnace excess |
| --- | ---: | ---: | ---: |
| 4×8 / 128 | 0.4 ms | 2.40% | 2.40% |
| 8×16 / 512 | 5.3 ms | 0.17% | 0.16% |
| 16×32 / 1024 | 42.3 ms | 0.82% | 0.12% |
| 32×64 / 2048 | 347.9 ms | 0.22% | 0.05% |

The 8×16 excess drops from 1.00% to 0.16% at similar construction cost in these
single CPU runs. Error remains nonmonotonic; no general accuracy threshold,
strict passivity bound, shader integration or GPU performance claim follows.
The algebraic reciprocity and proposal-mass checks still pass. Exact source
hash, compile command and raw output:
`build/tests/performance/multiscatter_order_sum_independent_v1/results.json`.
A preliminary same-estimator reference run is separately retained in
`build/tests/performance/multiscatter_order_sum_v1/results.json`.

## Off-node and grazing correction

The previous 8×16 validation directions coincided with cache nodes. A shifted
10×17 check including mu=0.01 and near-normal incidence exposed **15.06%**
maximum error/excess for the 8×16 summed-order cache. The earlier 0.16% figure
is therefore not representative of its angular domain. Raw evidence:
`build/tests/performance/multiscatter_off_node_v1/results.json`.

The prototype now stores endpoint-inclusive samples uniformly in mu, using
mu=1e-5 to approximate the grazing limit, and analytically integrates its
piecewise-linear interpolation against 2*mu for normalization. Independent
512×128 solid-angle quadrature verifies the capped rejection mass at the
unchanged 1e-4 tolerance. Reciprocity checks also pass. The initial coarse
mass-quadrature failure is retained in `multiscatter_endpoint_v1`; the resolved
check and compile/source provenance are in
`build/tests/performance/multiscatter_endpoint_v2/results.json`.

For depth/lambda 0.4, the corrected 8×16/512 cache has 1.61% maximum albedo
error and 0.63% estimated furnace excess at 5.5 ms CPU construction. The
16×32/1024 cache has 1.17% error and 0.12% excess at 43.1 ms; 32×64/2048 has
0.75% error and 0.05% excess at 341.2 ms. Boundary error and finite-reference
uncertainty remain. No renderer integration or general physical acceptance is
claimed; the old sampled-center figures remain historical narrow measurements.

## Grazing-refined angular coordinates

The prototype now places mu nodes at `(i/(N-1))^2`, brackets queries through
sqrt(mu), and still interpolates linearly in physical mu. Its projected
normalization integrates each actual nonuniform interval exactly; it does not
use a plain node average or interpolate in sqrt(mu). This allocates more
samples to the grazing variation without increasing the table size.

Against the unchanged off-node/boundary sampled-order reference, depth/lambda
0.4 results were: 8×16/512, 1.04% maximum error, 0.45% furnace excess, 5.2 ms
CPU build; 16×32/1024, 0.60% error, 0.36% excess, 40.2 ms; 32×64/2048,
0.13% error, 0.08% excess, 314.8 ms. All reciprocity and independent PDF-mass
checks passed. The largest relief error moved away from the grazing boundary.
Furnace excess does not improve monotonically and is worse than the uniform
mu table for some sizes, so this is not a blanket accuracy/passivity claim.

Evidence: `build/tests/performance/multiscatter_grazing_grid_v1/results.json`.
The renderer is unchanged. Scene cache inspection confirms its existing
buffers hold electromagnetic response matrices, not directional albedos;
production integration needs a distinct shared SVM/OSL representation,
spectral interpolation checks, and a strategy for linked material inputs.

## Varied-profile check

The host probe now accepts `--profiles`. Five additional combinations cover
smooth and rough isotropic GGX, strong anisotropy, smaller/larger wavelength to
pitch ratios, and different duty cycles. Each uses an 8×16 table with 512
summed-order facet samples per node, the same 170 off-node/boundary directions,
and the original sampled-order 65,536-sample reference. Flat and relief cases
are both retained. All algebraic reciprocity and PDF-mass checks pass.

Relief depth/lambda remains 0.4. Maximum albedo errors were 0.81% for
alpha=.05/.05, 1.10% for .8/.8, **4.80%** for .08/.7, 2.01% for .5/.12 with
wavelength/pitch=.2 and duty=.15, and 0.92% for .25/.4 with wavelength/pitch=.85
and duty=.8. The last case's estimated furnace excess was also 0.92%.
These results rule out treating the earlier single-profile error as a general
accuracy bound. Directional resolution/approximation still needs work before
native MultiGGX integration, independently of GPU construction speed.

Exact source hash, compile command and all ten raw results:
`build/tests/performance/multiscatter_profile_sweep_v1/results.json`.

## Native-math Metal construction feasibility

The standalone probe in `tests/metal/cycles_diffraction_multiscatter_albedo.py`
expands the existing Cycles Metal context and calls production GGX VNDF,
masking, diffraction geometry and order-power functions. It does not maintain
a copied replacement BSDF. No production renderer refactor was required.

After host preparation/syntax checks, the root task compiled the actual Metal
library and pipeline, then dispatched on Apple GPU outside the sandbox.
Two profiles (flat and depth/lambda 0.4), 32 grazing-refined directions each,
and 128 facet samples per direction produced **zero failures** against the
matching CPU estimator; maximum absolute CPU/Metal difference was
**6.5565109e-7**, below the predeclared 2e-3 portability tolerance.
The GPU accumulates in float, while the CPU reference accumulates in double.

Exact expanded source, executable, command/source hashes and raw results:
`build/tests/performance/multiscatter_metal_v1`.
This proves a bounded GPU construction path is technically feasible. It does
not establish an end-to-end speedup, convergence, general profile accuracy,
scene-cache integration, or physical acceptance of the missing-energy lobe.
The same-estimator comparison is a portability check, independent of the
separate high-sample physical-albedo checks that still expose approximation
errors. The renderer's MultiGGX guard remains in place.

## Larger GPU tables and SIMD construction

The standalone test now checks six profiles, each with 128 directional nodes
and 512 facet samples per node. These include flat, smooth/rough isotropic,
strongly anisotropic and higher-order-count relief. The original one-thread
per-node kernel passed all 768 CPU/Metal comparisons with maximum absolute
difference 2.44379e-6. Its dispatch-and-wait times ranged 1.179–5.440 ms.
Evidence: `build/tests/performance/multiscatter_metal_profiles_v1`.

The kernel now assigns one SIMD group per node, distributing facet samples
across lanes and reducing with `metal::simd_sum`. The host launches exactly one
SIMD group per threadgroup using the pipeline's execution width. The same
six-profile test passes, maximum CPU/Metal difference **2.98023e-7**.
Dispatch-and-wait times ranged **0.296–1.444 ms**; GPU command timestamps ranged
0.0575–1.1955 ms. This is one dispatch per profile, not a statistical production
benchmark. Pipeline compilation, scene-cache upload, spectral refinement and
shader lookup are excluded. No end-to-end render-speed claim follows.

Evidence, paired timings, expanded shader and source/executable hashes:
`build/tests/performance/multiscatter_metal_simd_v2`.
An initial missing namespace qualification caused a compile failure; its log
is retained, and the corrected source compiled and passed the GPU checks.
Physical approximation errors from the separate high-sample tests remain;
fast CPU/Metal agreement does not validate those approximations. No MultiGGX
renderer path has been enabled by this standalone test.

## Batched GPU material and wavelength tables

The standalone Metal builder now dispatches directional nodes along one grid
axis and profile/wavelength slices along the other. A single SIMD group still
reduces the 512 facet samples for one directional node. Output is profile-major;
all tables share one dispatch and completion wait. The validation batch includes
the six preceding material profiles plus 16 wavelengths from 380 to 780 nm for
a fixed 1600 nm pitch, 150 nm depth, duty 0.41, roughness (0.25, 0.4) profile.
Both wavelength/pitch and depth/wavelength therefore vary consistently.

On Apple M5 outside the sandbox, all 22 x 128 entries passed the existing
same-estimator CPU/Metal 2e-3 gate; maximum absolute difference was 2.98023223877e-07.
One batch took 5.879208 ms GPU command time and 5.993417 ms including dispatch/wait.
These are single-dispatch observations excluding pipeline compilation, not
statistical speed claims or renderer timings. Source snapshots, hashes, raw
output and results are in `build/tests/performance/multiscatter_metal_batch_v1`.

This removes serial dispatch/wait per table from the prototype and supplies a
concrete spectral-batch layout. It does not prove that 16 wavelength slices are
sufficient, validate interpolation across diffraction-order cutoffs, establish
passivity, or integrate the cache with scene preparation and SVM/OSL closures.
The production Multi-GGX guard remains in place.

## Unstored wavelength midpoint check

The batch probe now evaluates 15 unstored wavelength midpoints at all 128
angular nodes (1920 comparisons). It compares linear interpolation of the GPU
tables against a separate 16384-sample native facet-and-order estimator, not
the builder's summed-order quadrature. For the declared 1600 nm pitch, 150 nm
depth, duty 0.41, roughness (0.25, 0.4) material, maximum absolute albedo error
was 0.00181877613 and maximum albedo underestimate was 0.00181877613. The predeclared 0.01
absolute-error gate passed; exit status now fails if that diagnostic gate fails,
as well as on CPU/Metal portability failures.

This supplies evidence for wavelength interpolation of this single profile
at angular nodes. It does not validate arbitrary materials, off-node angular
interpolation, narrow features between tested wavelengths, or strict passivity.
The separate order-sampling reference still has finite quadrature error. No
production Multi-GGX support is enabled. Exact source snapshots, raw output and
results: `build/tests/performance/multiscatter_spectral_midpoint_v1`.

## Combined angular and wavelength interpolation

A subsequent check adds 2550 comparisons at the 15 unstored wavelength
midpoints and a shifted 10 x 17 angular grid. Directions include mu=0.01 and
near-normal incidence, with azimuths shifted from stored nodes and interpolation
wrapping across the azimuth seam. Lookup brackets the quadratic-mu grid but
interpolates in physical mu; wavelength interpolation is linear.

Maximum absolute albedo error was 0.00983476639 and maximum albedo underestimate was
0.00574702024 against the separate 16384-sample native order-sampling reference. The
unchanged 0.01 diagnostic gate passed. Both angular/wavelength and stored-angle
spectral gates now affect process exit status. This covers only the declared
1600 nm pitch, 150 nm depth, duty 0.41, roughness (0.25, 0.4) profile. It does
not establish strict passivity, bound quadrature error, or validate arbitrary
material parameters; cache/closure integration still remains.

Raw output, exact source snapshots, command and hashes are retained in
`build/tests/performance/multiscatter_combined_interpolation_v1`.

## GPU projected-albedo normalization

The Metal batch now computes each table's projected-solid-angle average in a
second compute encoder within the same command buffer. It exactly integrates
the piecewise-linear physical-mu interpolation over the quadratic-mu grid and
periodic azimuth. This normalization is required by the missing-energy lobe
and capped rejection PDF; a plain average of table entries would be incorrect.

All 22 GPU averages were finite and in [0,1]. Maximum error against the
double-precision analytic CPU integral of the same table was 6.29115319839e-08, below
the 1e-6 gate. The existing spectral and combined off-node interpolation gates
also passed unchanged. Total batch plus normalization dispatch/wait was 4.020583 ms
in this single run, excluding pipeline compilation and scene integration.

Exact source snapshots, raw output and command are in
`build/tests/performance/multiscatter_gpu_average_v1`. This validates table
normalization, not the approximate physical model or production closure support.
