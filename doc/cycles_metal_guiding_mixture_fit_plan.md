# Weighted mixture fitting for Metal guiding

Current checkpoint: streaming soft-assignment fitting is now connected to the renderer.
All four point-light Metal controls run at512SPP, but none reaches CPU-guided quality.
The later integration sections supersede the earlier primitive-only status below.
All19 persistent checks and PT/BDPT one-SPP retention checks now pass; broader
quality and transport acceptance remain outstanding.

## Motivation and scope

The physical-source experiment regressed relative to the position-conditioned
model at equal 512 SPP on the retained three-seed material benchmark. A selection
rule did not recover the position baseline. Neither model meets the primary
CPU-guided target. The fixed 16 angular regions currently permit only one fitted
component per region, even when two compact sources share it.

This experiment implements and tests a batch directional fitter before changing
renderer training. It is not yet used by rendering, does not replace parallax or
position conditioning, and has no claimed render quality or speed benefit.

## Implemented primitive

`intern/cycles/kernel/sample/guiding_mixture_fit.h` contains a bounded-capacity
weighted vMF mixture fitter shared by C++ and Metal. Deterministic weighted
farthest-point initialization is independent of histogram region membership.
Expectation steps use log densities and a log-sum-exp normalization. Maximization
uses weighted direction moments and a safeguarded inversion of the exact 3D vMF
mean resultant, with the existing maximum concentration of 16384. Common weight
rescaling avoids squared bright-weight overflow without clipping relative weights.
Repeated iterations rebuild sufficient statistics from the same batch; observations
are not accumulated multiple times. Zero-weight components stay zero, coincident
input can initialize fewer components, and empty publication gives a uniform model.

Publication writes the existing 18-float component format, leaving the existing
sampler's explicit uniform exploration intact. This is a callable publication
primitive, not a connection to the renderer's field scheduler. Observations contain
unit directions and positive finite weights; invalid observations are excluded.
The fitter owns no global allocation and mutates no observations.

The initial GPU test assigned a complete independent fit to one thread. A second
implementation now processes each field with one complete Metal SIMD group,
striding observations across lanes. Reductions combine maximum weights, seed
selection, and sufficient statistics. Equal-score seed ties choose the earliest
observation, preserving the sequential initialization rule. Each lane retains the
same private fit; only lane zero publishes. The caller must supply a complete
SIMD group with identical observation storage and counts for all its lanes.
Spatial/source statistics and renderer dispatch remain required integrations.

## Evidence and tests

The host fixture has two vMF sources (concentration 1500, relative weights 0.35
and 0.65) whose axes lie in one old angular region. Training uses 8192 directions;
a separate 10000-direction sample checks held-out density estimates. Tests require
recovery of both directions, concentrations, and weights, nondecreasing training
likelihood, and greater than one nat per held-out observation improvement over a
single fitted lobe. These are density-fitting checks, not rendered image errors.

Additional checks cover analytic concentration inversion, uniform and coincident
observations, empty publication, and weight-scale invariance at 1e30. Publication
is integrated over a 1024-square equal-area grid and sampled 30000 times using the
actual existing product sampler with a uniform factor. Its density and first
moment must agree with the published mixture.

A dedicated Metal kernel fits two 4096-observation batches for 64 iterations,
including extreme finite weights. Learned axes, concentrations, weights and
published query PDFs are compared with host results. The existing field, immutable
history, parallax, HDR, and BSDF transport tests remain in the same test runner.

The publication test exposed a separate existing numerical defect: a product with
a uniform factor reconstructed the natural parameter and renormalized a narrow
axis, causing approximately 9e-5 relative density error. The product now preserves
the nonuniform lobe exactly and returns the uniform factor as its integral. The
original test tolerance was retained. This shared query correction changes renderer
source; its build is tracked separately from the unintegrated fitter.

## Remaining integration

1. Measure and implement cooperative GPU fitting with the same fixtures and
   parameter/publication agreement checks. A serial thread per field is insufficient
   evidence of speed.
2. Add position and finite-source sufficient statistics under soft component
   responsibilities. Test multiple nearby sources, source covariance, distant
   illumination, unsupported fits, and forward/reverse query consistency.
3. Preserve full camera, shadow/direct, and adjoint training observations within
   explicit memory budgets. Include delta ancestry without treating delta records
   as independent radiance observations. Exhaustion must be explicit.
4. Separate working fit state from the published sampling distribution. Drain
   relevant work before publication and preserve sample equivalence across history
   memory-group boundaries. Do not count batches repeatedly across fitting steps.
5. Re-run persistent-render and cancellation tests, actual PT/BDPT field probes,
   and unchanged multi-seed equal-SPP comparisons against high-SPP CPU-guided
   references before adopting the fit. Measure rendering runtime separately and
   include training/publication costs. Full transport requirements remain intact.

## Primary background

The implementation is independent code based on normalized spherical-mixture
statistics. Related path-guiding representations and fitting approaches are
available in [OpenPGL](https://github.com/OpenPathGuidingLibrary/openpgl), the
[2020 Monte Carlo rendering course notes](https://graphics.cg.uni-saarland.de/papers/RenderingCourse2020_Notes_rev1.pdf),
and [Path Guiding Using Spatio-Directional Mixture Models](https://research.nvidia.com/publication/2021-12_path-guiding-using-spatio-directional-mixture-models).
These sources motivate learned mixtures; they do not establish correctness or
performance of this implementation.

Validation completed: `cycles_guiding_distribution.py --metal --transport`
passes 33 host tests, five transport tests, and all actual Metal checks, including
the new fitter and published PDF checks. Log:
`/tmp/cycles-guiding-mixture-fit-final3.log`. The full Blender build with the
uniform-product correction succeeds. These results do not substitute for renderer
integration or the remaining equal-SPP image benchmarks.


## Cooperative fitting and numerical correction

The first cooperative comparison failed the existing concentration tolerance:
for one component, sequential CPU fitting gave 1484.42 and cooperative Metal gave
1460.15. Independently accumulated double-precision source moments gave 1456.72.
Both float accumulation paths had lost accuracy near unit resultant length; the
sequential sum was worse. Compensated summation now tracks both directional and
mass errors. Clang reassociation is disabled within the iteration so optimization
cannot cancel the compensation. The numerical tolerance was not relaxed.

The host and Metal fixtures now independently compare fitted concentration to
known-source double-precision moments, in addition to parameter and PDF agreement.
The sources are separated by many standard deviations, making known membership
an independent reference for this fixture. This does not substitute for reference
image comparisons in rendering.

The compensated implementation passes the initial 32-field cooperative comparison
and partial batches of 4093 and 17 observations. The strengthened independent
checks are tracked in `/tmp/cycles-guiding-mixture-cooperative-final.log`.
A first GPU command-duration measurement, with identical 32 fields, 4096
observations per field and 64 iterations, was 0.20447 s sequential versus
0.006796 s cooperative (about 30x). This is a two-component fitting microbenchmark,
not a renderer speedup, not a warm multi-run performance acceptance, and not yet
validation of full 16-component occupancy or fitting. Those remain next checks.

The installed renderer is unchanged by this isolated fitter work. The latest
rendering build still needs the documented equal-SPP quality acceptance and full
transport implementation. No current result establishes CPU-guided quality parity.


The strengthened check then caught a smaller cancellation error in the 17-sample
batch: fitted concentration 3160.646 versus the independent 3159.446, exceeding
the new absolute tolerance of one. `/tmp/cycles-guiding-mixture-cooperative-final.log`
is therefore a failed run, not passing evidence. Fitting now also accumulates
compensated squared offsets around the current lobe axis. The unit-direction
identity `Var(D) = 1 - |E(D)|^2` supplies concentration without subtracting a
near-unit resultant from one. For narrow lobes the equivalent rationalized
expression `(1 + sqrt(1 - variance)) / variance` avoids that cancellation; broader
lobes retain the safeguarded inversion. The concentration cap remains unchanged.
The tolerance stays fixed. Validation of this correction is recorded in
`/tmp/cycles-guiding-mixture-cooperative-stable.log`.

The stable-variance run completed successfully: 33 host tests, five transport
tests and all actual Metal checks pass, including the strengthened independent
reference and partial-batch checks. Corrected two-component GPU command durations
were 0.225136 s sequential and 0.00730675 s cooperative for identical work, about
31x in this single microbenchmark. Full 16-component validation, repeated warm
measurements, spatial/source fitting and renderer integration remain outstanding.


## Full-capacity and warm verification

A shared templated Metal test kernel and host harness now exercise 2 and 16
components. The 16-source fixture distributes well-separated narrow lobes over
the sphere; all fitted directions, concentrations and weights must agree with
CPU fits and independent known-source double moments. Bright weights, the 4093
partial batch, and published query PDFs are checked. This supplements the
existing two-peaks-within-one-bin fixture, not a proof of every overlapping-mode
case or rendered scene.

`/tmp/cycles-guiding-mixture-fit-16-warm.log` completes successfully with 33 host
and five transport tests plus all actual Metal checks. Each GPU case runs one
excluded warmup and five measured repetitions; fitting starts from the same
observations each time. `build/metal-guiding-tests/mixture-fit-16/report.json`
retains the full durations and scope. For 8 fields, 4096 observations per field,
16 components and 64 iterations, median GPU time is 3.214582 s sequential and
0.109675 s cooperative, about 29.3x. Sequential min/max is 3.210609/3.223571 s;
cooperative is 0.109450/0.110181 s. These are equal-work fitting microbenchmarks,
not renderer timings. The installed rendering implementation remains unchanged.

The next step is soft position/source sufficient statistics, including source
covariance and effective sample sizes, followed by bounded history integration.
The working fit and published render model must stay separate across memory
flushes. Adding all spatial statistics to every replicated lane may create high
register pressure; compare observation-parallel and component-parallel collection
before choosing the renderer's production dispatch. No final image-quality or
full-transport requirement is waived by the fitter tests.


## Soft position and source collection

`guiding_mixture_statistics.h` adds a 48-byte observation representation and a
component-owned compensated accumulator for the existing 23 position and 13
source statistics. Source tuples retain finite radiance, harmonic distance weight,
and distance weight; they do not collapse multiple endpoints to an average point.
Distances are in the isotropic scene metric, while position regression retains
normalized scene coordinates. A shared maximum radiance weight rescales all
components consistently. Assignment probabilities enter squared weights
quadratically, so splitting one observation between components does not invent
independent observations. Unknown/distant sources retain directional/position
training and contribute no finite-source statistics.

The host fixture assigns the same observations fractionally (0.25/0.75) to two
components and checks total mass, source covariance from distinct endpoints,
anisotropic scene coordinates, and unchanged effective support. Unknown sources
remain unsupported physical fits while retaining radiance training.

The Metal fixture first learns two directional components from 4096 observations.
Then each active component lane collects its own spatial/source moments, with SIMD
reductions normalizing responsibilities over the learned components. No global
floating-point atomics are used. Published statistics must match CPU collection;
fitted source targets must match known geometry, with negligible extent for the
point sources. A subsequent check fits both source and positional models directly
on Metal and compares a position-dependent query to CPU and known source geometry.
This is still isolated fitting work: rendering does not consume this new collector.

The first GPU collection/geometry run passes in
`/tmp/cycles-guiding-mixture-source-metal.log` (34 host tests, five transport tests,
and actual Metal checks). The expanded direct-Metal fit/query validation is tracked
in `/tmp/cycles-guiding-mixture-source-final.log`. No rendered-image improvement is
claimed. Conditional component publication and iteration, full-capacity source
collection, and bounded renderer history integration remain required next steps.

The expanded direct-Metal fit/query run completed successfully: 34 host tests,
five transport tests and all actual Metal checks pass. Source and position fits
are evaluated on the GPU using the collected statistics; their targets and query
directions agree with CPU and known geometry. The installed renderer is unchanged
by this isolated collector implementation.


## Conditional publication and iteration

`guiding_mixture_conditional.h` writes component statistics into the existing
18-float query representation. Supported finite sources use the physical model;
otherwise supported position regression is used, with the previous directional
lobe as fallback. Empty complete publication produces a normalized uniform model.
The source-first policy remains experimental: the earlier fixed-bin physical
model regressed on materials, so this new policy requires image-quality evidence
before adoption. These moment updates are not claimed to maximize a joint
conditional likelihood or to guarantee monotonic optimization.

The shared log-component evaluator uses each observation's position when computing
responsibilities. The Metal test performs four conditional collection/publication
steps in alternating buffers. Each step reads one immutable model, writes the
other model, and verifies that the input remains byte-identical. Component mass
is reduced before publication; the output must sum to one and agree with an
independently orchestrated CPU step. The CPU reference reads the exact same input
model for each step, so this checks each update rather than claiming bitwise
independent trajectory equivalence. The source fixture currently has two active
components; full-capacity source/conditional tests remain outstanding.

Host checks sample a published physical distribution at a different query position,
compare the returned and evaluated PDF, and compare the sampled first moment
with its analytic expectation. Source-statistics input is preserved. All 35 host
tests, five transport tests, and actual Metal checks pass in
`/tmp/cycles-guiding-mixture-conditional-metal.log`, including all four conditional
steps and the existing full-capacity directional fitting checks.

These new fitting/publication primitives are still isolated from renderer history.
Next validate full-capacity conditional collection, then connect bounded complete
observations and separate working/published field buffers to the renderer scheduler.
Unchanged equal-SPP CPU-guided reference comparisons and full BDPT transport remain
required; no rendered quality gain is established by these tests.


## Full-capacity conditional verification and history integration review

The source collection test now uses shared templated host/Metal code for both
2 and 16 active sources. The full-capacity fixture distributes point sources over
the sphere, verifies all 36 statistics per component against CPU collection,
checks physical and positional fits directly on Metal, and performs four
alternating-buffer conditional iterations with normalized outputs and unchanged
inputs. `/tmp/cycles-guiding-mixture-conditional-16.log` passes all 35 host tests,
five transport tests, and actual Metal checks for both capacities. This remains
isolated from rendering; it does not establish material quality parity.

The renderer review identifies a necessary retention change before fitting can be
connected. `guiding_gpu_flush_history` currently consumes complete camera records,
but `guiding_gpu_record_importance` and the direct-observation part of
`guiding_gpu_record_shadow_radiance` record directly into aggregate fields. Those
observations must also be retained for learned component assignments. They must
remain standalone observations, without modifying camera ancestry or shadow heads.
The current history-group reservation covers only non-null camera records and is
insufficient for this additional retention.

Count the full set of event producers before extending that reservation: camera
bounce records, surface/volume adjoint arrivals, NEE shadows, and BDPT connections.
Mixed transparent surfaces may produce evaluable arrivals/direct observations even
when the sampled continuation is transparent, so `max_bounce` alone is not a valid
bound; the transparent limit also matters. Surface stage-zero/cache-retry guards
must continue preventing duplicated observations. Existing shadow-catcher split
reservation is separate and must remain. Implement the retention and derived bound
together, then rerun minimum-memory/single-SPP equivalence, overflow guards,
persistent-render/cancellation, and PT/BDPT probes before adding field fitting.

Do not publish learned query models at a memory-group flush. The sampling batch
must retain an immutable query field regardless of history capacity. Working
statistics/model state needs separate storage until the existing sample-based
publication boundary. All retained camera, direct and adjoint contributions must
participate without silent overflow drops or resampling shortcuts.


## Renderer retention implementation

Completed positive direct-light and adjoint observations now append standalone
records to the same bounded pool as camera histories. They have no parent and do
not update path/shadow heads. Radiance and finite-source W/H/D are retained until
the entire main/shadow group drains, then the existing flush consumes them into
fields. Zero/nonfinite observations remain excluded as before. This prepares full
observations for learned fitting; the renderer still publishes its old field fit.

The group bound reserves `max(max_bounce,1)+1` camera records, plus up to one NEE
observation when direct guiding is enabled and two BDPT observations (adjoint
arrival and connection) per visit. The visit bound includes ordinary bounces,
transparent bounces and two terminal/reserve visits. Arithmetic is 64-bit. The
existing separate shadow-catcher split reservation remains intact. Surface NEE
and BDPT connection each create at most one direct-record shadow per visit;
adjoint arrival is guarded by stage zero after successful shader evaluation.
Volume scattering records one arrival and at most one NEE observation. This is
a conservative reservation, not a new sampling or contribution limit.

The actual Metal history test now interleaves two standalone observations with
each camera path's eight ancestry records, across 100000 paths and four reuse
cycles, including deliberate overflow. It checks complete retention of 200000
standalone records per full group, exact radiance/source moments, no ancestry
changes, preserved shadow prefixes, and untouched memory beyond capacity.
`/tmp/cycles-guiding-retained-unit.log` passes the shared host/transport/Metal suite.
The full Blender build and installation succeed in
`/tmp/cycles-guiding-retained-build.log` and `/tmp/cycles-guiding-retained-install.log`.

Renderer verification is running in `/tmp/cycles-guiding-retained-regression.log`
and `/tmp/cycles-guiding-retained-transparent.log`. The latter uses a new isolated
24-sheet mixed transparent/diffuse scene with only two ordinary bounces allowed,
64 transparent bounces, and compares equal one-SPP PT/BDPT images with 16 and 128
MiB history budgets. Existing benchmark scene scripts and CPU-reference provenance
are unchanged. Pending renderer tests must not be reported as passed.

The retained-observation persistent-render run completed: all 19 checks pass,
including minimum-memory one-SPP equivalence, reset/off/probability controls, UI
modes and CPU/Metal coexistence. Report:
`build/metal-guiding-tests/retained-render-regression/report.json`. The mixed
transparent PT/BDPT stress test is still live and must be checked before claiming
its result. This build retains the extra observations but still uses the existing
field publication model; the learned mixture fitter is not integrated yet.


## Indexed observation grouping and a BDPT reproducibility finding

`guiding_observation_range.h` partitions drained histories by field using count,
prefix and scatter dispatches. It stores one uint index per retained observation,
not a second observation payload. Records and ancestry remain immutable. Capacity
and invalid-field errors are explicit; scatter remains bounded even after an
invalid prefix. The indexed view decodes directions and converts source distances
to scene metric at read time. The directional fitter now accepts either contiguous
arrays or this indexed view through the same implementation.

The GPU fixture checks every one of 4096 records across 16 fields for uniqueness
and correct assignment, verifies unchanged record bytes and untouched capacity
guards, and fits directly through the indexed view. The full Metal suite passes
in `/tmp/cycles-guiding-observation-range-metal.log`. Additional host cases cover
delta records, zero/nonfinite radiance and invalid field identifiers. This is an
integration primitive; its buffers and dispatches are not yet wired into rendering.
Its index allocation must be included in the history budget when that is done.

The mixed-transparent render stress run completed with a failure in the strict
BDPT image comparison. PT passes: maximum difference 2.98e-8, pairwise MSE
3.22e-18 between 16 and 128 MiB at one SPP. BDPT differs (MSE 0.18406, maximum
26.2922). Neither run overflowed the retained pool. This is not a passing result.

A diagnostic repeats BDPT twice at the SAME 128 MiB budget and also fails the
unchanged strict tolerance: pairwise MSE 0.839505, maximum 37.7779. Its raw EXRs
and report remain in `retained-repeat-bdpt`; log
`/tmp/cycles-guiding-retained-repeat-bdpt.log`. The failure therefore cannot be
attributed specifically to changing history budgets. The reservoir's first valid
vertex acquires a slot through an atomic append in `bidirectional.h`; this is a
candidate source of stochastic cache ordering, not yet a proven diagnosis.
Investigate reproducibility before claiming BDPT budget equivalence. Do not loosen
the image tolerance or report either failed run as passed.

The final host grouping checks complete successfully: all 36 host tests pass in
`/tmp/cycles-guiding-observation-range-final-host.log`. The preceding shared
GPU/transport run also passes. BDPT reproducibility remains unresolved; no pending
render process is being treated as completed evidence beyond the logged failures.

## Current integration checkpoint after MIS-aware training audit

The chronological pending items above are not the current status. Cooperative
16-component fitting, soft source/position statistics, immutable conditional
publication, full-history retention and indexed grouping have passed their
standalone host/Metal checks. The BDPT retention/order defect was subsequently
corrected; `area-nee-retention/report.json` verifies identical 4096 canonical
96-byte records and image MSE 1.803423e-16 across 16/128 MiB budgets. That report
predates the latest emitter and training corrections and does not validate them.

Renderer integration still needs budgeted index/working-fit allocations, grouped
observation dispatch, and a working model distinct from the frozen sampling model.
Publication must retain history-group equivalence, avoid counting observations
repeatedly, and survive cancellation and texture retries. The current renderer
still uses fixed angular-region fits.

The latest training audit found that ordinary GPU NEE standalone observations
conflicted with the CPU's MIS-aware training target. Its correction substantially
reduces the point-light regression, but corrected RIS/MIS/ROUGHNESS PT remain
1.57/1.48/1.47 times CPU-guided MSE at equal 512 SPP on the single-seed fixture.
See `cycles_metal_guiding_variance_audit.md`. Preserve that corrected training
semantics when integrating the fitter. Additional light-linking, non-MIS training,
material-convergence and full transport audits remain necessary; neither these
core tests nor the point-light result establish final quality acceptance.

## Renderer scheduling constraint for the next fitting integration

`enqueue_work_tiles` flushes and reuses the immutable history pool whenever all
queues drain; multiple such groups can precede `update_gpu_guiding` publication.
Consequently, fitting each drained group to convergence and replacing the model
would make training depend on the history-memory setting. Simply invoking the
existing batch fitter at each flush is not an acceptable integration.

The bounded-memory integration must collect responsibilities against a frozen
input model throughout the publication interval, merge sufficient statistics
across groups, and update the working model only at the scheduled boundary.
Global per-field scaling must preserve both linear and squared-weight moments
when a later group contains brighter observations. Stable centered directional
variance is needed for narrow lobes; raw near-unit resultant subtraction already
failed the isolated fitter's tests. Spatial refinement must explicitly transfer
or initialize those statistics and sampling models. The full existing histogram
can supply initial support, but must not replace retained observations for soft
assignment. Repeated EM on a single group must not count those observations as
new training samples. This integration is still required and is not implemented
by the scheduler reservation change below.

After the MIS-aware NEE correction, `gpu_guiding_group_size` now reserves a
standalone NEE record only when direct-light training is enabled and MIS-aware
training is disabled, matching the recorder. BDPT adjoint and connection reserves
remain included, as do transparent visits and terminal vertices. This permits
larger complete groups without removing any required history record. Build and
install succeeded in `/tmp/cycles-guiding-nee-reservation-{build,install}.log`;
material and persistent/retention validation is tracked in `nee-training-material`.
No throughput improvement is claimed without a controlled runtime measurement.

## Disjoint batch accumulation

`GuidingMixtureStatistics::merge` now combines disjoint batches whose
responsibilities were evaluated against the same immutable input distribution.
Each batch can have a different maximum-radiance scale. The accumulator increases
its common scale as needed and rescales linear moments by the ratio, and squared
weight moments by its square. Both existing and incoming compensated low parts
are preserved. Invalid scale or nonfinite incoming state is rejected before any
mutation. The method does not run another E-step or change observation counts.

The host test uses group sizes1/17/257, forward/reverse traversal, ordinary and
1e30-scale observations, full-statistic single-pass comparison and independent
double-precision mass/squared-mass checks. It also checks failure atomicity.
The Metal conditional-update test collects4096 observations in257-entry groups
against unchanged input storage, then publishes once. Four successive updates
are compared with single-pass CPU collection for2/16 components. A further
16-component case changes group brightness at1e28 scale; known source targets,
normalized output and immutable input remain checked. The final strengthened
run also compares every sufficient statistic, not only the published model.

The first Metal compile failed because reference parameters lacked explicit
address spaces (`/tmp/cycles-guiding-mixture-batch-merge.log`). The shared method
now uses `ccl_private`. The corrected initial and changing-brightness runs pass
in `/tmp/cycles-guiding-mixture-batch-merge-final.log` and
`/tmp/cycles-guiding-mixture-batch-hdr.log`. Final strengthened verification is
tracked in `/tmp/cycles-guiding-mixture-batch-merge-verified.log`.

This removes a batch-combination dependency; it does not yet wire the fitter into
renderer allocations/dispatch/publication, implement stable centered directional
moments across updates, or prove rendered quality. The installed renderer remains
the MIS-aware NEE/reservation correction with the documented quality failures.

Final strengthened verification completed successfully:37 host tests,8 transport
tests and all actual Metal tests pass in the verified log, including each merged
statistic and the changing-brightness16-component case. No test tolerance was
relaxed. The test binaries are retained in `mixture-batch-merge`; no renderer
quality or timing claim follows from these primitive checks.

## Centered directional statistics for streaming updates

The streaming statistics now include three weighted directional offsets and
one weighted squared offset about a fixed component reference (40 entries total,
plus compensation and the reference direction). All four new entries scale
linearly with radiance, including the squared offset. `directional_fit` recovers
variance as the mean squared offset minus the squared mean offset, avoiding
subtraction of the near-unit resultant from one. Concentration uses the same
stable narrow-lobe inversion as the validated batch fitter; broad lobes use the
safeguarded inversion. Empty data produces a uniform fallback.

Merging rejects different reference directions after accumulation has started;
the initial empty destination adopts its first batch's reference. Every group in
a publication interval must use the same immutable component reference. A changed
reference therefore requires a new accumulation interval, not silently pooling
incompatible centered moments. Nonfinite references are rejected during recording.

The host fixture uses an oblique axis,8192 observations, concentrations0.5/10/1000/
15000, groups1/17/1024 and an independent double-precision moment and inverse-
resultant reference. The existing batch test also checks the four added moments.
The Metal conditional-update fixture sets references from its frozen input model
and uses the newly fitted directional fallback when publishing, while preserving
source/position fitting and input immutability. The first test build failed on a
missing sampler PDF argument (`/tmp/cycles-guiding-mixture-centered.log`); corrected
validation is tracked in `/tmp/cycles-guiding-mixture-centered-final.log`.

This remains a renderer-integration dependency, not a rendered quality result.
The installed renderer and its unresolved PT/BDPT quality failures are unchanged.

The first actual Metal run failed the unchanged full-statistic tolerance at
component0,entry36: GPU-0.104194194 versus CPU-0.104162425. The test had derived
its centered reference separately via a physical-source query on each device.
Centered sums are reference-dependent, so independently rounded query axes need
not yield the same raw offset sum. The fixture now reads the identical stored
component vector as its fixed reference; centered variance does not require that
reference to be renormalized. The original tolerance is retained. Diagnostic log:
`/tmp/cycles-guiding-mixture-centered-diagnostic.log`; corrected run:
`/tmp/cycles-guiding-mixture-centered-fixed-reference.log`. This run is pending
until its tool handle reports completion.

Using identical stored references did not change the discrepancy; that hypothesis
was insufficient (`/tmp/cycles-guiding-mixture-centered-fixed-reference.log`).
The next correction removes redundant normalization in statistic recording:
observations already carry validated unit directions, while CPU and Metal use
different normalization arithmetic. The concentration oracle still computes its
independent double reference from normalized input, and all original tolerances
remain unchanged. Current run: `/tmp/cycles-guiding-mixture-centered-unit-input.log`.
Do not treat this correction as validated until that run completes.

The unit-input correction passes the full suite:38 host tests,8 transport tests
and all actual Metal checks, including every centered statistic, independent
normalized-input double concentration checks, immutable conditional publication
and changing HDR group scales. No tolerance was relaxed. The normalization step
was the relevant difference: changing the fixed reference alone had not helped.
All jobs are terminal. Renderer integration remains the next required step.

## Renderer integration: streaming soft mixture

The renderer now allocates a separate compensated working mixture and observation
counts, partitions each drained history pool by field, and evaluates conditional
responsibilities with one32-lane Apple GPU SIMD group per field. The existing
40-entry statistics merge against the frozen sampling model across every history
group in the publication interval. Linear and squared moments retain their common
radiance scale; source distance is converted by the indexed observation view.
Each record is collected once. Prefix/scatter bounds, failed merges, count overflow
and incompatible SIMD width produce explicit render errors.

The four new kernels are Metal-only and are registered in the device-kernel name
and feature gates. New integrator pointers precede the scalar block, and the Metal
pointer-boundary assertion includes the new last pointer. The initial full build
caught that assertion; the corrected build succeeds. The working mixture and
per-field grouping metadata are included in the field-memory budget; the index
array is included in the history-memory budget. The first64MiB field allocation
uses201 nodes and approximately63.53MiB including working/grouping storage, rather
than allocating these buffers outside the user-visible limit.

Publication now precedes refinement. The histogram supplies initial support and
spatial adaptation; once soft statistics have positive mass, publication replaces
the angular-region mixture with the learned directional/source/position model.
The working buffers are then cleared for the next interval. New children inherit
the just-published parent sampling model and the existing decayed histogram state.
Thus child queries have a valid frozen model while collecting their own data.
Sampling retains the existing explicit uniform exploration. Empty/untrained
fields use their histogram initialization until soft observations are available.

This uses streaming soft assignments with one moment-fit update per publication
interval, not64 repeated
batch-EM passes over a reused memory group. The isolated iterative batch fitter
remains a tested primitive; it is not being described as the renderer's algorithm.
The current cooperative dispatch requires32-lane SIMD groups (Apple GPUs);
a different execution width is an explicit error, not a silent partial fit.
Broader Metal-device execution-width support remains a limitation to resolve.

Build/install logs: `/tmp/cycles-guiding-streaming-build2.log` and
`/tmp/cycles-guiding-streaming-install.log`. The shared38 host,8 transport and
actual Metal primitive suite passes in `/tmp/cycles-guiding-streaming-core.log`.
The first actual renderer test is512SPP point-light PT, with field snapshots in
`streaming-render`; it is pending. Renderer quality, lifecycle and memory-budget
acceptance cannot be inferred from the primitive suite or successful build.

The first renderer attempt crashed at `prepare_gpu_guiding`: the new private
working buffer had been cleared through `device_only_memory::zero_to_device`,
whose Metal implementation expects CPU-accessible shared storage. Reset now uses
`queue_->zero_to_device`, matching the existing private buffers. The crash and
failed log are preserved as `streaming-render/reset-crash.txt` and
`point_pt_ris-reset-failed.log`. Corrected build/install succeed in
`/tmp/cycles-guiding-streaming-reset-{build,install}.log`; the same renderer test
is running again. A strengthened host check also verifies that fresh child
histogram and smooth-mixture PDFs equal the published parent's before any child
observations; all38 host tests pass in `/tmp/cycles-guiding-streaming-child-query.log`.

The reset-fixed rerun then failed runtime Metal compilation because the four
new sampling headers were absent from the installed kernel source tree. They
are now registered in `kernel/CMakeLists.txt`; their installed paths were checked.
The failed log is `point_pt_ris-headers-failed.log`. Build/install succeed in
`/tmp/cycles-guiding-streaming-headers-{build,install}.log`, and the same512SPP
renderer test is running again. These integration failures are retained and do
not count as passing renderer evidence.

The installed-header rerun completed at verified512SPP with a finite image.
Point-light RIS PT MSE is1.300480195102367e-6 against the same CPU-guided8192
reference, versus1.451943485547699e-6 before streaming integration. The roughly
10.4% reduction still leaves error1.4085 times CPU-guided512; parity fails.
This comparison also reflects the budgeted field capacity change259→201 nodes.
The field_128 snapshot and inspection output are retained in `streaming-render`.
MIS,ROUGHNESS and BDPT controls are running at the same512SPP; lifecycle and
memory-budget renderer checks remain required. No speed or final acceptance
claim is made from the single-seed result.

All four equal512SPP point controls completed finite. MSE relative to the common
CPU-guided8192 reference: RIS PT1.300480195102367e-6, MIS PT1.371793549732966e-6,
ROUGHNESS PT1.366381846903805e-6, RIS BDPT5.668094378738207e-6. CPU-guided512
MSE remains9.232944063459344e-7; all four therefore fail parity. MIS/ROUGHNESS
are essentially unchanged from the prior build, and BDPT only slightly improves.
The current `streaming-render/comparison.json` preserves all metrics. Renderer
persistent-session and PT/BDPT retention checks are now running.

Integrated-renderer lifecycle checks completed: all19 persistent cases pass, and
16/128MiB mixed-transparent retention preserves PT (MSE2.176168882585891e-18)
and BDPT (MSE1.7659073909766682e-16), with identical canonical BDPT cache bytes.
Reports are under `streaming-render/persistent` and `streaming-render/retention`.
These one-SPP retention comparisons verify scheduling/contribution retention,
not multi-interval learned-distribution equivalence or final quality. All jobs
are terminal. Material/multi-seed quality, warm performance including fitting,
broader execution-width support and the full transport requirements remain.

## Material follow-up and decayed working prior

The integrated fitter's equal512 material controls completed. Against the same
CPU-guided references, rough-glass PT MSE rose to0.015135555710272761 (1.2984xCPU)
and transmission PT to0.017468926690298838 (1.4902xCPU). BDPT improved to
0.009286127322544377 and0.010674691917314072 (0.7966x and0.9106xCPU), but block
means remain0.6057% and0.5882% above reference. These single-seed results do not
establish correctness or final parity. Raw metrics and object/outside region
breakdowns are retained in `streaming-material`. Most squared error is outside
the projected object rectangles; object-only reasoning is insufficient.

Code inspection also confirms that broad-surface MIS/ROUGHNESS still use the
histogram proposal, while RIS and narrow glossy products use the smooth mixture.
CPU guiding deliberately excludes non-opaque closures; this does not authorize
removing the requested GPU transmission guiding to improve a comparison.

The first streaming integration cleared working statistics at every publication,
unlike the histogram's existing quarter-weight history. The new candidate carries
that decayed prior. Centered statistics are explicitly recentered into the newly
published component frame; lowering their common radiance scale by0.25 implements
weight decay while preserving normalized linear/squared moments and compensation.
Refined children inherit the parent's prior at half its scale, matching the
histogram's half-weight inheritance. Initial untrained models do not add empty
soft-assignment counts. The observation metadata saturates at64, exactly the
eligibility threshold used by both conditional fitters; effective support still
uses all weighted moments. No observation weights or rendered contributions are
removed by that metadata saturation.

`DecayedPriorMatchesExplicitWeightedHistory` rebuilds a12-epoch weighted history
independently and compares all40 statistics plus concentration. The actual Metal
conditional-update test also recenters before publication. All39 host tests,
8 transport tests and Metal checks pass in `/tmp/cycles-guiding-streaming-prior-core.log`.
The prior candidate's renderer build is running; no prior-candidate image result
is claimed yet. The existing material failures remain the current renderer evidence.

The prior candidate builds and installs successfully (`streaming-prior-build.log`
and `streaming-prior-install.log` in `/tmp/cycles-guiding-*`). Its four material
renders are running in `streaming-prior-material`, at the unchanged512SPP settings.
No prior-candidate image has been accepted or substituted for the previous results.
Lifecycle, cancellation/retry and quality validation of this exact build remain.

## Rejected retained-prior candidate and next regularization audit

All four prior-candidate512SPP renders completed. Rough glass PT/BDPT MSE:
0.016626333791032302 /0.009936773376821926. Transmission PT/BDPT:
0.016358818584974035 /0.011250561106145614. Three of four comparisons regress
relative to `streaming-material`; the candidate provides no consistent benefit.
The renderer's prior retention/recentering and child-prior inheritance were
removed. The tested recentering primitive and all failed-candidate artifacts remain.
The no-prior renderer was rebuilt/installed in
`/tmp/cycles-guiding-streaming-prior-revert-{build,install}.log`.
The64-observation eligibility saturation and empty-model early return remain;
the two conditional fitters use that metadata solely as a64-observation gate.

The retention harness now accepts samples, sheet count and bounce limit, retaining
its old defaults and unchanged equality tolerance. A16SPP opaque-room run (zero
sheets,64 bounce limit) is running for PT and BDPT at16/128MiB histories. Its driver
checks actual completed sample counts for both renders, and preserves failures
independently for each integrator. Output: `streaming-learning-retention`.
This deliberately exercises learned sampling across multiple publications.

Primary-source inspection found another concrete CPU/GPU fitting difference.
The bundled OpenPGL config sets weightPrior0.01, meanCosinePriorStrength0.2 and
meanCosinePrior0. The downloaded upstream implementation at commitb4e4b86 uses
MAP mixture-weight and mean-resultant regularization; its incremental update
combines current and previous statistics with sample-count weights. The GPU
publication currently uses unregularized component weights and directional fits.
This can create weakly supported narrow lobes or permanently zero component
weights. Regularization requires a complete audit including physical-source and
position-conditioned queries, not just a fallback concentration adjustment.

Sources checked: bundled `lib/macos_arm64/openpgl/include/openpgl/config.h` and
`/tmp/cycles-guiding-openpgl-distance-research/openpgl/directional/vmm/ParallaxAwareVonMisesFisherWeightedEMFactory.h`
(especially MAP weight and mean-cosine updates). Upstream:
https://github.com/OpenPathGuidingLibrary/openpgl/blob/b4e4b86/openpgl/directional/vmm/ParallaxAwareVonMisesFisherWeightedEMFactory.h
No regularization change or new quality claim is made at this checkpoint.


### Learning repeatability diagnostic after retained-prior rejection

The restored no-prior renderer completed all four 16-SPP, 128-square opaque-room
renders with 64 maximum bounces. The strict 16-versus-128 MiB history-budget image
comparisons failed: PT pairwise MSE 0.28040835, BDPT 0.07921921. These are differences
between two finite render realizations, **not errors against CPU reference**.
The initial BDPT cache bytes matched. Results are preserved under
`build/metal-guiding-tests/streaming-learning-retention`.

To isolate the cause, eight more renders repeated identical settings at each
budget, serially. Every log confirms exactly 16 completed SPP. All four original
strict image comparisons also failed at unchanged budgets:

| Integrator | History MiB | Same-budget pairwise image MSE |
| --- | ---: | ---: |
| PT | 16 | 0.47360310 |
| PT | 128 | 0.21335112 |
| BDPT | 16 | 0.05248467 |
| BDPT | 128 | 0.08724001 |

Thus the cross-budget failure alone cannot establish a budget-specific error.
Initial BDPT caches are byte-identical in these controls. First-publication node
buffers and metadata match exactly, while tree values differ by up to 1.15e-5
and mixture entries by up to 0.0064 (entries have different units). This is
consistent with floating-point accumulation order differences in the initial
histogram fitting; it is not yet a proof of the complete divergence mechanism.
The learned proposal affects subsequent path choices, so first-publication field
differences can propagate into later samples. No equality tolerance was relaxed,
and these failed tests are not reported as passing memory-retention evidence.

Raw fields, EXRs, commands, reports, and the reproducible first-field inspector
are in `build/metal-guiding-tests/streaming-learning-repeat`. The retention harness
now offers `--dump-guiding` to retain separate snapshots for each render, including
intermediate resize/crop renders. Original default one-SPP behavior is unchanged.

Next: distinguish repeatability from statistical estimator correctness with
fixed-observation fitting checks and repeated equal-SPP comparisons to CPU-guided
references. Do not infer a quality gain or loss from these pairwise differences.
The regularization audit above remains pending; no renderer code changed during
this diagnostic. Full quality parity and transport completeness remain unproven.


### Confidence-regularized streaming candidate (implemented; quality pending)

The renderer now publishes a Dirichlet weight prior (0.01 per component) and a
zero-mean directional prior (strength 0.2). These strengths follow the bundled
OpenPGL defaults. This is an adaptation, not a port of OpenPGL's EM schedule:
support is measured with weighted effective sample size rather than raw record
count. Existing full-history collection, publication schedule, and transport
estimator remain in use.

A new sufficient moment stores sum(w^2 * responsibility). Its sum across
components recovers the global squared weight, allowing global effective support
(sum w)^2 / sum(w^2) for the weight prior. It must scale quadratically when merging
HDR batches. The old sum((w * responsibility)^2) remains the local component
support used by conditional fitting. These are different quantities; summing
local effective sample sizes would incorrectly count soft assignments as
independent paths.

The zero-mean prior shrinks the fitted lobe's mean cosine by n/(n+0.2), then
inverts the spherical vMF mean function. The existing safeguarded Newton solver
was moved into the shared spherical-lobe type, preserving the fitter's algorithm.
Fallback and positional concentrations are regularized at publication. Finite
source concentration depends on query position, so its prior is applied after
geometric conditioning at query time. The shared query method supports all three
modes. Zero prior is an exact identity. The prior changes only the proposal;
sampling and PDF evaluation use that same proposal and no radiance is rescaled.

The component layout is now 19 floats (entry 18 stores prior-strength/support for
query-dependent models); sufficient statistics are 41 floats. Memory accounting
uses the new compile-time sizes. The snapshot inspector recognizes the old 5/18
layouts and the current 19-float layout. Empty components receive positive weight
with a uniform fallback, so subsequent responsibilities can revive them.

Validation: 42 host tests, 8 transport tests, and actual Metal tests passed in
`/tmp/cycles-guiding-regularization-core2.log`, including grouped streaming
statistics and four conditional updates against host collection. Independent
host tests verify the prior against double-precision mean cosine, its effect in
all three query modes, and mass conservation/revival. An additional HDR global
support oracle is running in `/tmp/cycles-guiding-regularization-host-final.log`.
Build/install succeeded (`/tmp/cycles-guiding-regularization-{build,install}.log`).
Four equal-512-SPP material renders against the preserved CPU-guided references
are running under `build/metal-guiding-tests/regularization-material`. No image
quality, speed, or full-feature acceptance is claimed yet. Same-build reference
refresh, multiple seeds, full transport, and final benchmarking remain required.

The additional independent HDR global-support oracle passed; the final host-only
run now reports 43 tests passed. Actual-Metal validation remains the completed
42-host/8-transport run above. Renderer code is unchanged since that run.


### Regularization material results (equal 512 SPP)

All four renders completed with logs confirming 512 SPP. Full raw results and
comparison are in `build/metal-guiding-tests/regularization-material`. The table
reports whole-image MSE relative to the CPU-guided 512-SPP control, both evaluated
against the same high-SPP CPU-guided reference.

| Scene | PT ratio | BDPT ratio |
| --- | ---: | ---: |
| Rough glass | 1.356263 | 0.805560 |
| Transmission | 1.413988 | 0.849520 |

Relative to the previous unregularized single runs, rough-glass PT/BDPT errors
increased 4.46%/1.13%; transmission PT/BDPT errors decreased 5.12%/6.71%. These
are mixed single-run results, not statistical proof of a regularization benefit.
BDPT block means differ from reference by +0.431%/+0.905%. Instrumented render
times are higher (13.60/14.07 s PT; 46.06/47.37 s BDPT), but these are not repeated
warm performance benchmarks. No acceptance claim. Point-light 512-SPP controls
are now running under `regularization-point`, with RIS/MIS/ROUGHNESS PT and RIS
BDPT, and snapshots for inspecting the new query layout.


### Regularization point-light results: candidate fails parity

All four point controls completed exactly 512 SPP. `regularization-point/comparison.json`
reports MSE against the CPU-guided 8192-SPP reference:

| Mode | MSE | Ratio to CPU-guided 512-SPP MSE |
| --- | ---: | ---: |
| RIS PT | 1.633025485e-6 | 1.768694 |
| MIS PT | 1.368686753e-6 | 1.482395 |
| ROUGHNESS PT | 1.340656219e-6 | 1.452035 |
| RIS BDPT | 5.581858393e-6 | 6.045589 |

The previous unregularized RIS PT result was 1.300480195e-6. This candidate is
25.6% worse in that single-run comparison; it is not a confidence interval.
Combined with the mixed material results and higher observed render times, the
candidate is not accepted. The 19-float learned-field inspector passed, which
establishes structural validity only. All processes completed. Regularization
remains installed at this checkpoint for reproducibility; no renderer reversion
has yet been made. Preserve the raw candidate data before further changes.
