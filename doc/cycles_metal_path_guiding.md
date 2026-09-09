# Metal path guiding

For the current build instructions, scope, and final validation, start with
[the delivery guide](cycles_metal_guiding_validation.md). The chronological
research notes below include superseded experiments and are not all current status.

## Objective and acceptance criteria

Implement directional path guiding on Apple Metal for the regular and bidirectional
Cycles integrators. Preserve the existing OpenPGL CPU implementation. A checked box,
a successful kernel compilation, or a lower variance image at greater render time is
not sufficient evidence of completion.

Required validation includes normalized sampling densities, sample/evaluation
agreement, independent-seed agreement with high-sample CPU OpenPGL guided references, surface and
volume rendering, bidirectional MIS, reset/cancellation and bounded memory use.
Final comparisons use equal camera sample counts, with elapsed time reported
separately and including training and cache updates. Earlier equal-time results
are historical diagnostics, not the final acceptance criterion. Keep denoising,
clamping, adaptive sampling and exposure controlled in quality comparisons.
Report regressions as well as improvements, and disclose BDPT's extra light paths.

User clarification, 2026-09-08: CPU guided renders are the primary image reference.
Use independent reference seeds to check convergence and preserve small illuminated
features that a finite-sample unguided image may miss. Unguided GPU images remain
useful performance baselines. Earlier GPU-reference results below are preliminary;
repeat their error measurements against the CPU guided reference before conclusions.

Further clarification: CPU guiding is also the requested minimum quality target
at the same sample count. The benchmark runner includes CPU-guided PT at that
count; speed does not excuse a measured equal-sample quality regression.
The user identified the adaptive Metal 128-sample image as visually cleaner than
the CPU 128-sample image in the shared-transform comparison. Keep that visible
detail improvement in the assessment; bright outliers dominating linear MSE must
not erase it, and a favorable image must not hide other material regressions.

User scope clarification, 2026-09-09: the existing BDPT transmission-block
brightness offset is excluded from path-guiding acceptance. It is retained in
historical measurements but is not a reason to delay this guiding deliverable.
Other measured quality regressions must still be reported.

## Research, 2026-09-08

* [OpenPGL](https://github.com/OpenPathGuidingLibrary/openpgl) exposes CPU SIMD
  backends; GPU backends remain future work. Its live C++ objects and path segment
  pointers cannot be passed to Metal kernels.
* [Practical Path Guiding](https://www.jannovak.info/publications/PathGuide/index.html)
  and its [reference implementation](https://github.com/Tom94/practical-path-guiding)
  learn incident radiance using spatial subdivision and directional quadtrees.
  Separate training and sampling distributions permit unbiased adaptive sampling.
* [Wavefront path guiding](https://arxiv.org/abs/2405.06997) addresses persistent
  GPU memory and wavefront scheduling explicitly, storing exitance in a sparse
  voxel octree and constructing directional proposals during rendering.
* [Practical product guiding](https://beltegeuse.github.io/research/publication/2020_PracticalProductGuiding/)
  combines a radiance representation with an approximation of the BSDF. Proposal
  quality and the cost of constructing a product must be measured together.
* [Neural parametric mixtures](https://arxiv.org/abs/2504.04315) model spatial and
  directional correlation using learned mixtures. Integrating and training an
  additional neural runtime on Metal would add dependencies and validation beyond
  the sampling kernel itself.
* [Hyperion's 2025 case study](https://www.yiningkarlli.com/projects/pathguidingcourse2025.html)
  discusses production surface/volume guiding in a wavefront renderer.
* [Public Cycles GPU prototype](https://github.com/phil-martin/pathguiding)
  demonstrates OpenPGL device snapshots and CUDA integration, but explicitly
  reports unresolved low-sample fitting and variance-reduction problems and about
  fourfold rendering overhead. It is useful prior work, not a validated Metal or
  bidirectional implementation to import.

## Repository findings

Baseline: `2d105f8dc29`, Blender 5.3 alpha. The existing Metal bidirectional
integrator generates light paths, caches a reservoir-selected vertex per path,
connects camera paths and splats sensor connections. Its recursive MIS uses both
forward and reverse scattering PDFs. It has partial volume support: light subpaths
stop at their first volume collision, and camera transport falls back to ordinary
PT after an unsupported volume event. Full multiple-scattering BDPT is therefore
an outstanding integrator limitation. Existing regression scenes are in
`tests/python/cycles_bdpt_*`.

At the baseline, CPU guiding uses per-thread OpenPGL distributions and path segment
storage. Device capability and the UI restrict it to CPU. The outer path tracer
updates the CPU field after render batches. Metal already specializes several
guiding switches for one combination of features, but has no guiding backend.

The development machine is an Apple M2 Max, 38 GPU cores. The existing Release
build enables Metal and OpenPGL. Run the installed app under `install/Blender.app`;
the executable in the build tree does not contain its installed resources.
Use an isolated `BLENDER_USER_RESOURCES` directory during tests. GPU execution
requires access outside the filesystem sandbox, which does not enumerate Metal.

## Architecture and implementation sequence

1. Establish executable unguided PT/BDPT baselines and reproducible scene creation.
2. Implement a device-callable equal-solid-angle directional distribution with
   explicit normalization, finite-value handling, and a nonzero exploration
   component. Test the actual shared sampling code on the host before integration.
3. Add a bounded spatial guiding field with independent sampling and accumulation
   storage. Only publish a new sampling distribution after all camera, light, and
   shadow work using the previous distribution has drained. No in-flight query
   may observe a partially trained PDF.
4. Integrate GPU training into wavefront paths. Evaluate stochastic path-vertex
   recording to avoid a full history per active GPU path. The selected vertex
   must be chosen independently of its future contribution. Copy training state
   into shadow paths so late shadows do not refer to recycled main-path slots.
   Record downstream radiance with the actual proposal PDF and throughput.
5. Integrate surface and volume proposals, preserving closure selection, delta
   events, transmission, continuation probabilities and light-sampling MIS.
   Evaluate the same mixture in NEE and continuation; never substitute a BSDF
   PDF for a guided PDF in MIS.
6. Integrate bidirectional guiding. Audit camera continuation, light continuation,
   reverse PDFs, connection endpoints, emitter hits and sensor splats. Keep the
   field generation fixed for each complete cache/camera batch. Radiance and
   adjoint importance are different training targets; all transport-side choices
   must be explicit in sampling and reverse-density evaluation.
7. Add device capability reporting, UI controls, memory/training limits, resets,
   and clear backend-specific control descriptions. CPU settings keep their
   existing behavior.
8. Build and run real Metal renders, including a baseline regression matrix,
   difficult indirect illumination, glossy/transmission and participating media.
   Optimize only after mean agreement and PDF tests pass. Record measured image
   error, render time, seeds, samples, field memory and any failure.

The spatial representation and recording parameters will be finalized against
measured fitting quality and device cost. This document is an implementation plan,
not a claim of completed validation. The implementation is integrated and renders
on Metal; quality, performance and full integrator coverage remain acceptance gates.

## Validation record

### Current backend design

The implementation is independent of OpenPGL and introduces no additional runtime
dependency. Its binary spatial tree and directional quadtrees live in Metal buffers.
The user-selected 16–1024 MiB budget bounds the field allocation; the default is
256 MiB. Each spatial node has fourteen distributions: six primitive-normal sectors
for incident radiance, six for adjoint importance, and separate radiance/importance
volume distributions. Surface sectors use the original geometric normal, undoing
the faceforward flip recorded by `SR_BACKFACING`, so camera and light queries agree
when crossing a transmissive surface. The product factor uses the query's incoming
direction separately.

Each directional tree has five levels and 1024 equal-solid-angle leaves. Regions
with fewer than 32 contribution records collapse to coarser constant densities;
this changes the proposal, not the scattering function. A five-percent uniform
component retains directional support. Spatial leaves split along their longest
axis after 1024 visits, up to depth 20 or the memory limit. Children inherit half
their parent's observations; publication retains one quarter of previous training
history so local records can replace the inherited fit.

Training selects one camera-path vertex independently of future contributions,
favoring early bounces. Main and shadow state store the selected record index and
inverse throughput/PDF weight. Unclamped film contributions train incident radiance;
camera arrivals separately train the adjoint proposal used by light subpaths.
Shadow records are copied by value and survive reuse of their originating main
path slot. Once finite training finishes, the kernels stop initializing and
transporting unused training records.

Publication occurs after 1, 2, 4, 8, 16, 32 and 64 training samples, then every 64,
including the exact final requested sample. All camera, light and shadow work
drains before refinement and publication. Sampling and PDF evaluation therefore
observe the same immutable generation. Persistent-session resets clear training;
each device owns its own field, while CPU devices retain OpenPGL.

Broad surface proposals use a sixteen-cell approximation to the product of learned
radiance/importance and a one- or two-sided cosine factor. Narrow microfacet-only
surfaces additionally use a smooth mixture product, described below. Volume proposals use a
Henyey–Greenstein factor matched to the phase mixture's first moment. Actual BSDF
and phase evaluations are unchanged. Both guided and ordinary samples use the
same mixture PDF, including NEE and bidirectional reverse-density queries. Discrete
events retain their original closure-selection mass. A cosine product is less
accurate for narrow glossy/transmission lobes than full BSDF product guiding;
material-specific variance measurements below are necessary to assess that tradeoff.

The Metal UI exposes the existing enable, surface/volume and training controls,
plus field memory, guiding probabilities and minimum roughness. OpenPGL-specific
debug controls and diagnostic passes remain CPU-only.

For continuous surface scattering, let `p_b` be the original density including
closure-selection mass, `a` the probability of selecting a continuous closure,
`q` the normalized field/product density, and `g` the requested guiding fraction.
The density used by sampling and MIS is `(1-g)*p_b + g*a*q`; its integral is `a`.
The remaining discrete/BSSRDF selection mass is unchanged. A guided direction
selects its transport label from the original per-closure posterior density so
reflection/transmission labels and roughness/eta retain their meaning. In volumes,
the mixture is `(1-g)*p_phase + g*q`. The nonnegative proposal representation can
be approximate without changing the physical BSDF/phase contribution; correctness
requires evaluating the exact density of that approximate sampling procedure.

The field is an adaptive proposal, so training records can be correlated, inherited,
decayed or spatially coarse without directly biasing the estimator. They can still
make variance worse. This distinction is why normalization tests and image-quality
measurements are separate acceptance checks.

### Experiment history

Measurements below are chronological experiments. No general quality or efficiency
improvement has been established; some surface BDPT cases improve while PT regresses.

Initial baseline smoke renders (32 x 32, 32 samples, diffuse area-light scene,
seed 0, no glass) completed on Metal: PT 0.507 s, mean 1.07038; BDPT 0.632 s,
mean 1.07707. These small runs establish execution, not convergence or performance.

The shared 32 x 32 equal-solid-angle directional quadtree passes five host tests.
The same header compiled and ran on the M2 Max for three distributions with one
million samples each: empty/uniform, varying weights, and a concentrated lobe
with 5% uniform exploration. All returned finite unit directions and positive
PDFs, with zero sample/evaluated-PDF mismatches. Histogram chi-square statistics
were 1023.65, 1026.28, and 1111.17 (1023 degrees of freedom). Estimates of the
analytic integral of z-squared over the sphere (4*pi/3 = 4.18879) were 4.18644,
4.19052, and 4.22860. GPU build-plus-sampling times were 0.219, 0.658 and 0.972 ms.
These are isolated sampler measurements, not Cycles rendering speedups.

Reproduce with `python3 tests/python/cycles_guiding_distribution.py --metal`.
These initial sampler tests preceded field integration. Later sections record the
integrated renderer measurements; full acceptance validation remains pending.

### First integrated measurements (not acceptance results)

The first Metal render completed after correcting the GPU launch-state ABI: Metal
relocates every field preceding `sort_partition_divisor` as a device pointer, so
the new guiding scalar controls must follow that boundary. A static assertion now
checks the last-pointer boundary. Detailed logs confirm field growth from 3 to
1753 spatial nodes with a 64 MiB budget and fresh training on a repeated render.

The first 64 x 64, 128-sample, three-seed matrix exposes a quality regression.
For the baffled indirect-light scene, mean squared error is 0.1113 unguided and
0.6914 guided against a 4096-sample PT reference; measured warm render times are
0.841 and 2.197 seconds. Glossy PT also regresses. These results must not be
represented as a path-guiding speedup. Initial data and editable scenes are in
`build/metal-guiding-tests/initial-matrix`.

Two causes under investigation are excessive spatial refinement driven by adjoint
arrival counts and sparse radiance recording from uniform path-depth selection.
The next revision uses early-depth sampling and radiance-driven subdivision, and
geometrically grows update batches to reduce publication overhead.

Unguided BDPT also disagrees with the PT reference (mean 0.5245 versus 0.6697 in
the indirect scene). The existing reservoir connection weight applies inverse
inclusion probability both to the contribution and inside the recursive MIS
denominator, while the competing NEE and sensor weights do not. The correction
keeps reservoir compensation in the contribution and retains the complete-strategy
MIS partition. This requires independent image validation. Relevant derivation:
[Implementing VCM](https://iliyan.com/publications/ImplementingVCM/ImplementingVCM_TechRep2012_rev2.pdf)
and [SmallVCM](https://github.com/SmallVCM/SmallVCM).

### CPU OpenPGL reference comparisons

Primary references now use CPU OpenPGL guiding, averaging independent seeds 991
and 1993 at 8192 samples each. This follows the requested comparison criterion:
unguided images are performance baselines, and missing detail is judged against
the guided CPU reference. Reference seed disagreement is recorded separately;
a finite-sample reference is not assumed noiseless.

The product-proposal revision (before history decay and GPU state allocation
optimization), 64 x 64 pixels, 128 samples, seeds 11/23/47, gives these ratios of
unguided MSE to guided MSE. Values above one favor guiding:

| Scene | Ordinary PT | BDPT |
| --- | ---: | ---: |
| Indirect | 0.318 | 1.164 |
| Glossy | 0.447 | 1.137 |
| Mixed diffuse/delta glass | 0.449 | 1.067 |
| Participating medium | 0.300 | 0.559 |

Raw CPU references and recomparison are in
`build/metal-guiding-tests/cpu-references-64`. All seeds are included. These are
quality measurements at equal sample counts, not speedup claims. They confirm
that the ordinary PT and volume regressions still require work. Longer preliminary
256 x 256 tests also fail to establish an efficiency gain. The BDPT MIS correction
is included in both baseline and guided renders, so its bias correction is not
counted as a guiding gain.

The Cycles time-limit setting overshot a two-second request substantially in this
fork, because one scheduled render chunk can contain thousands of samples.
Those diagnostic results are not equal-wall-time comparisons. The separate
`cycles_metal_guiding_equal_time.py` runner calibrates sample counts using timing
alone on a held-out seed, then measures independent image seeds and reports the
actual wall-time ratio. A result is marked matched only within five percent.

The latest isolated tests pass eight host cases, nine million Metal distribution
samples with zero sample/PDF mismatches, and twelve million spatial-field training
records at a deliberately exhausted 31-node capacity. Full renderer validation
of the latest field-history and allocation changes remains in progress.

Set `CYCLES_METAL_GUIDING_DUMP=/tmp/guide` explicitly to save queue-synchronized
field snapshots. Inspect `/tmp/guide_128` with
`tests/python/cycles_metal_guiding_inspect.py`, optionally passing world-space
`--point x y z` queries. Snapshotting is disabled in normal rendering.

### Field diagnostics and directional regularization

Private Metal buffers require an explicit blit to shared storage for diagnostic
readback, even on Apple Silicon. The synchronized queue readback now handles both
shared and private buffers. The snapshot inspector checks topology and directional
mass conservation. An indirect-scene snapshot at 128 training samples contains
128 spatial leaves, all at depth seven. Cells crossing the baffle mix observations
from its bright and dark sides. The orientation-conditioned revision separates
six dominant-normal sectors per surface transport target, with independent volume
fields (14 total distributions per spatial node). Reverse BDPT density queries
orient that normal toward the swapped incoming direction.

Orientation conditioning alone regresses: the three-seed indirect PT comparison
has MSE 0.4277 versus 0.1072 unguided against the CPU guided reference. The smaller
sets of observations are too sparse for uniformly fine directional histograms.
The subsequent revision tracks contribution counts and collapses undersampled
angular subtrees to constant densities until they contain 32 effective records.
It preserves each collapsed subtree's integrated mass. Counts and sums retain a
quarter of their history per publication, so inherited observations do not dominate
new local fitting indefinitely. This affects the proposal only; image contributions
still use the actual normalized proposal and unchanged physical scattering values.

Ten host tests now cover adaptive mass preservation, retained count leaves,
orientation sectors, independent transport targets, spatial refinement, and
sampling. The Metal stress tests also pass with the expanded storage. Integrated
adaptive-fit rendering is in progress.

All ten persistent-session cases passed before the orientation/adaptive changes:
guiding off/on, finite limits one and five, budget resizing, repeated renders,
surface-only selection, unlimited training, lighting reset, and mixed CPU/Metal.
Detailed logs verify exact finite training limits and reset on each fresh render.
The off/on/off check restores the baseline pixels within 1e-5. These checks do not
yet cover interactive viewport cancellation.

Comparisons also report mean squared log1p RGB error, using the identical transform
for every raw image and CPU reference. It supplements linear MSE and does not
replace it. The shared-transform comparison plot displays errors without per-image
exposure normalization. Original EXR and linear NumPy images remain unchanged.

The adaptive model's three-seed 64 x 64 indirect PT result is still mixed:
linear MSE is 0.1528 guided versus 0.1072 unguided, while log1p MSE improves to
0.00976 from 0.01449. Warm end-to-end render time is 1.318 versus 0.793 seconds.
These results are not an efficiency gain. The output is retained in
`build/metal-guiding-tests/adaptive-matrix`; the reference import records its source
configurations and image hash. The next query refactor retains the distribution
pointer across sampling and PDF evaluation to remove redundant traversal and
closure scans. Larger-image comparisons are pending.

Direct Metal tests now include three sparse/adaptive distributions in addition to
the earlier nine: twelve million directions in total, all with zero sample/PDF
mismatches. The new cases' chi-square statistics are 1023.27, 1034.95 and 1048.16
for 1023 degrees of freedom. Host and device builds agree within float tolerance.

### Larger reference and cancellation results

At 256 x 256 and 1024 samples, with 256 MiB field memory and CPU guided references
(two seeds, 8192 samples each), the adaptive/query-refactored PT revision gives
linear MSE 0.01052 versus 0.01197 unguided, and log1p MSE 0.000906 versus 0.002210.
Its warm end-to-end time is 6.363 versus 4.666 seconds. The CPU reference's estimated
MSE is 0.000292. Both image means agree with that reference. This establishes a
quality improvement at equal samples, but not an efficiency gain under linear
MSE. The subsequent frozen-state traffic change remains to be benchmarked.

A narrower 0.15 m aperture fixture is provided by
`cycles_metal_guiding_aperture_scene.py`, composing the original room factory without
altering its existing reference scenes. The runner hashes both files. Its 256 x 256
CPU guided reference uses 8192 samples per seed and has mean 0.188424, with estimated
reference MSE 0.000111. New PT and BDPT comparisons are in progress.

The cancellation test found an existing background-render responsiveness problem:
the application cancellation callback was not checked inside large GPU work, so a
SIGINT break waited approximately 64 seconds for a scheduled batch to return.
GPU work now polls the application callback every 100 ms in addition to its cheap
viewport cancellation flag. The delayed-signal test, sent after the next batch
entered rendering, cancels in 0.0574 seconds and restarts successfully in the same
session. The restarted publication sequence is exactly 1, 2, 4, 5. The test has a
five-second latency bound and verifies the restarted image is finite and nonempty.
It exercises background cancellation; interactive viewport UI testing is separate.

### Material acceptance findings and reciprocal transport correction

The material suite adds rough Glass BSDF, coated Principled, transmissive Principled
and subsurface fixtures. CPU references use two independent 8192-sample runs at
64 x 64; projected object rectangles provide local error and mean measurements.
The first 512-sample, three-seed Metal material matrix is retained under
`build/metal-guiding-tests/final-material-matrix`. Despite its directory name, it
is **not an accepted final result**: both BDPT variants show a bright transmission
bias. For rough glass, CPU reference mean is 0.665739, ordinary Metal PT is
0.666935, Metal BDPT is 0.740036 and guided Metal BDPT is 0.708727. Large apparent
BDPT MSE gains in this matrix cannot be treated as a successful result.

Inspection found that the existing recursive reverse-density query only swapped
`ShaderData::wi`. Cycles microfacet evaluation requires that incoming direction to
be above its prepared normal; transmission therefore returned zero instead of the
reciprocal density. Reconstructing the shader in the reciprocal orientation also
updates relative IOR and direction-dependent closure weights. This correction is
under device validation. The path-density requirement follows the
[PBRT bidirectional derivation](https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing).

An independent test directly evaluates Cycles' actual GGX glass BSDF in both
orientations. After removing their outgoing cosines, reciprocal/forward values
equal `1 / eta^2`, not `eta^2`. For IOR 1.45, the measured ratio is 0.475624. The
existing light-subpath correction multiplied by 2.1025 instead; it now divides.
The new test checks more than 3000 valid sampled transmission directions, roughness
0.18/0.55 and entering/exiting IORs. Run it with
`python3 tests/python/cycles_guiding_distribution.py --transport`.

Same-budget CPU guided PT comparisons are retained separately in
`build/metal-guiding-tests/cpu-material-budget-512`. At 512 samples, CPU/Metal guided
log1p MSE ratios are 0.528 (rough glass), 1.002 (coated), 0.447 (transmission), and
0.853 (subsurface). Values above one favor Metal. The cosine product needs further
work for narrow glossy lobes. A possible extension is a smooth spherical-mixture
fit to the directional field, permitting analytic products with narrow BSDF
approximations; this is still a design investigation, not implemented functionality.
Relevant prior work is
[Herholz et al., Product Importance Sampling](https://sherholz.github.io/publication/2016-product/).

The first corrected rough-glass BDPT render (seed 11, 512 samples) has mean 0.666377,
linear MSE 0.007580 and log1p MSE 0.001406 against that CPU reference. Warm render
time is 9.974 seconds. Reciprocal shader reconstruction increased cold compilation
substantially: its discarded first render took 862 seconds. Independent-seed
validation is running; this one image is not sufficient to accept the correction.

The first stage of the smooth-lobe extension tested a normalized spherical
Gaussian primitive before renderer integration. Twelve host distribution tests passed;
seven million additional Metal Gaussian samples match analytic moments from the
uniform limit through concentration 16384, including a product distribution.
Existing twelve-million-sample quadtree tests and the exhausted spatial-field test
also pass. Glossy fitting, product mixtures, integration and image gains are still
outstanding work.

### Smooth products for narrow glossy lobes: implementation under validation

The field now publishes a sixteen-component spherical Gaussian companion to each
adaptive histogram. Each component moment-matches one coarse directional region;
its concentration is capped at 512 to respect the histogram's finite resolution.
The field stores an additional eighty floats per distribution, within the existing
memory budget. Publication remains entirely on Metal and sampling generations stay
immutable. The snapshot format now records a separate sampling stride; the inspector
also validates Gaussian weights and axes while retaining compatibility with older
histogram-only snapshots.

For microfacet-only surfaces below roughness 0.5, the query forms smooth reflection
and transmission approximations using closure weights, normals, roughness and IOR.
Thin transmission follows its straight-through lobe. These proposals have analytic
products with the learned spherical mixture; the actual BSDF value is unchanged.
A separate five-percent uniform proposal component keeps full support, including
when the user selects guiding probability one. Mixed diffuse surfaces keep the
earlier adaptive-histogram proposal. Index-matched transmission needs special care:
Cycles samples it as a Dirac event even at nonzero roughness, so those surfaces
retain their original closure sampler and probability masses.

Thirteen host distribution tests pass, including fitted product-mixture normalization
and sample moments. Nine million additional Metal samples cover uniform, narrow and
product Gaussians plus broad and narrow fitted product mixtures. Their sample/PDF
checks and analytic moments pass; host/device field fits also agree. These are sampler
and publication tests, not evidence of a rendering improvement. The new material
render comparison is running under `build/metal-guiding-tests/smooth-material-pt`.

The corrected reciprocal BDPT material matrix is complete for three seeds at 512
samples before the smooth-proposal extension. Rough-glass means are 0.664523
unguided and 0.665292 guided versus CPU reference 0.665739. Guided BDPT linear/log1p
MSE is 0.008440/0.001436, compared with 0.021158/0.002406 for CPU guiding at 512
samples. Unguided BDPT is cleaner still in this case (0.005945/0.000639), reinforcing
the need to improve glossy guiding. The original bright glass bias is no longer
visible in the independent-seed means. Transmissive Principled's corresponding means
are 0.663305/0.663174 versus CPU reference 0.665111; convergence and local errors
still need assessment at higher samples.

The smooth-product PT matrix has now completed at 64×64, 512 samples and three
independent seeds. Rough glass has linear/log1p MSE 0.025932/0.002704 versus
0.021158/0.002406 for CPU guiding; transmissive Principled has 0.034557/0.003969
versus 0.024120/0.002920. The smooth product reduces log1p error by roughly 1.68×
and 1.64× relative to the earlier histogram-only glossy proposal, but still fails
the requested CPU-quality target. Those improvements are not a completion claim.

A separate training-duration experiment retained the same CPU references and
increased GPU training from 128 to 512 samples. Rough-glass log1p MSE increased
from 0.002704 to 0.002833 and time from 1.540 to 1.876 seconds; transmission
increased from 0.003969 to 0.004063 and 1.577 to 1.874 seconds. Keep the 128-sample
setting; longer training did not fix these cases. CPU OpenPGL's iteration schedule
differs from the GPU's sample schedule, and changing the CPU setting from 128 to
512 produced identical 512-sample CPU images in this experiment. Reference reuse
records the reference training setting explicitly.

Equal-wall-time validation at 256×256 is the next acceptance experiment. Timing
calibration selects sample counts using elapsed time only, before evaluating
independent image seeds. Both linear and log1p error, local object regions and
achieved timing differences remain in the report. Installed Metal kernel sources
are hashed separately from the executable because runtime shader edits can change
the renderer without changing its binary hash.

The reciprocal shader is explicitly kept out of line in Metal. Its first cold
32-sample rough-glass BDPT render completed in 315.96 seconds, of which Cycles
reported 1.35 seconds tracing. Earlier cold runs of the inlining-heuristic version
took 834–862 seconds. This is encouraging compilation evidence, not a controlled
steady-state speed comparison: a separate CPU reference render was running during
the new cold run. Warm image and timing validation remains necessary.

The rebuilt smooth/noinline renderer passes all twelve persistent-session checks.
The original BDPT suite stopped at its one-seed hair light-tracing comparison:
0.049245 versus PT 0.049966, a 1.44% difference against its 1% threshold. Eight
additional independent light-tracing seeds average 0.049678 with standard error
0.000306, within one standard error of the PT value. This supports sampling noise
as an explanation, but the original assertion remains recorded as failed; its
threshold has not been relaxed. A diagnostic continuation driver reuses the
completed same-build cases and executes the remaining assertions, preserving a
nonzero exit status for failures.

One possible next glossy improvement is the CPU backend's existing two-candidate
resampled importance sampler (RIS), currently the CPU default. It keeps the
stochastic contribution weight separate from the deterministic density used to
partition NEE MIS weights. A Metal implementation would need the same distinction
throughout both camera and light-path recurrences; treating the reciprocal RIS
weight as a literal marginal PDF would be incorrect. Relevant primary research
includes [Nabata et al., resampling-aware BDPT weights](https://visualcomputing-lab.github.io/pdf/tog2020.pdf)
and [conditional RIS](https://research.nvidia.com/labs/rtr/publication/kettunen2023conditional/).
This is a candidate architecture, not implemented functionality or a claimed gain.

### First measured equal-time CPU comparison

At 256×256 in the indirect room, five held-out seeds (101, 211, 307, 401, 503)
averaged 6.0739 seconds for guided Metal and 6.0499 seconds for CPU guiding, a
0.40% timing difference. Calibration selected 1299 Metal samples and 179 CPU
samples using elapsed time alone. Against the independent two-seed CPU-guided
reference, Metal/CPU linear MSE was 0.008286/0.041116 (4.96× lower on Metal),
and log1p MSE was 0.000720/0.003239 (4.50× lower). Mean radiance was 0.670183
on Metal versus reference 0.670124. This case meets the requested CPU-quality
target at matched time; it does not establish performance for other scenes.

The unguided Metal control did not retain its calibrated timing: measured runs
averaged 4.4607 seconds versus the 6-second target. Its comparison must be
recalibrated before calling it equal-time. The failed timing check is explicit
in `smooth-equal-time-indirect/report.json`. A common-transform CPU/Metal image
and error comparison is saved beside that report.

The standalone two-candidate RIS helper passes three new numerical tests: null
candidates and large finite weights, integration with partial proposal support
and direction-dependent MIS, and a two-vertex bidirectional composition whose
MIS weights depend only on the selected path. Together with the earlier tests,
sixteen sampler/field tests and two actual-BSDF transport tests pass. The helper
is not yet connected to the renderer. Its composition test supports the following
implementation requirement: use the random reciprocal contribution weight for
throughput and guiding observations, but deterministic proposal densities for
all forward/reverse MIS recurrences. Light-path roulette must also use a
deterministic path throughput rather than the latent RIS candidate weights.

The larger equal-sample material baseline confirms the remaining gap: at 256×256
and 512 samples, rough glass has Metal/CPU linear MSE 0.021441/0.011710 and log1p
MSE 0.001888/0.000915. Transmission has 0.023437/0.012034 and 0.002665/0.001222.
Metal takes 3.43/3.70 seconds versus CPU 19.93/22.24 seconds, respectively, but the
same-sample quality target still fails. These five-seed results are retained in
`smooth-material-256`. CPU OpenPGL's current surface eligibility also excludes
transmission: its glass results benefit from guiding elsewhere in the scene.

RIS is now integrated in the working source for continuous surface mixtures on
both camera and light paths. Mixed discrete/continuous surfaces retain their
measure-preserving mixture sampler. The candidate pair contains one ordinary
BSDF sample and one learned product sample; selection targets actual BSDF value
times the learned incident-radiance estimate with a uniform component. Throughput
uses the RIS contribution weight, while NEE and forward/reverse BDPT recurrences
use the deterministic half-BSDF/half-guide density. Light roulette removes the
stochastic sampling factor, matching the existing camera-side convention.

The Metal UI exposes product MIS, product resampling, and roughness-weighted
mixtures through the existing sampling setting. CPU behavior is unchanged.
The rebuild passes after correcting a const-qualification mismatch. An additional
million-sample Metal RIS integration test estimates 9.90593 against the analytic
integral 9.9 and passes its six-standard-error check. Render correctness, quality,
performance, and persistent-session tests of this revision are still pending.

### Resampling render validation and legacy ablation

The installed resampling revision passes all fifteen persistent-session checks,
including each of the three sampling modes. At 256×256 and 512 samples, five
independent seeds produce rough-glass linear/log1p MSE 0.017286/0.001512 and
transmission MSE 0.018671/0.002089. These improve on the previous smooth mixture,
but remain worse than CPU guiding at the same sample count. These runs are
explicitly quality-only: concurrent reference renders and shader compilation
invalidate performance comparisons. Longer GPU training (512 rather than 128)
did not improve the single-seed glass experiment sufficiently to adopt it.

The first 256×256, 512-sample guided BDPT rough-glass image has linear/log1p MSE
0.009471/0.000590 against the CPU-guided reference. This is encouraging but only
one seed, and its elapsed time is not a valid performance measurement. The
three-seed volume PT comparison has guided Metal MSE 0.106626/0.004588 versus
CPU 0.076232/0.002507 and unguided Metal 0.135712/0.010804. The CPU same-budget
mean is itself above the high-sample reference; convergence across sample counts
must be checked before interpreting small mean differences as bias.

The diagnostic continuation of the 74-render legacy BDPT suite records six
failed assertions: hair light tracing, panoramic light tracing, finite-radius
point and spot light tracing, and two sparse volumetric-caustic assertions. A
controlled ablation replaced only the installed bidirectional header with its
original HEAD version in a frozen application copy. All 74 renders reproduced
the same six failures, with nearly identical values in those cases. This isolates
the header changes as not causing that failure set; it is not a full unmodified
checkout test. Original thresholds and nonzero failure status are retained in
`legacy-header-ablation/report.json` and the continuation report.

### Directional-moment refinement under validation

The next revision accumulates actual ray-direction first moments per Gaussian
component, in addition to the adaptive histogram. Four additional floats per
component retain narrow angular features that would otherwise collapse onto a
histogram bin center. Components with fewer than 32 contribution observations
retain their regularized histogram fit. Supported observed fits can reach
concentration 16384; histogram-only fits remain capped at 512. Packed directions
are stored with pending camera/shadow observations, and all publication and
spatial-inheritance operations include the moment data. Surface RIS uses the
smooth incident-radiance model and analytic candidate products; volume guiding
uses an analytic product with a phase first-moment approximation. Actual BSDF
and phase evaluations remain the contribution factors.

Seventeen host sampler/field tests, two actual-BSDF transport tests, twelve
million Metal histogram/product samples, nine million Gaussian samples, one
million resampling samples and twelve million field observations pass. The new
sub-bin observation test verifies recovery of a direction narrower than a bin.
Full-render quality and timing validation of this revision are pending.

The rebuilt directional-moment revision reduces five-seed, 256×256, 512-sample
rough-glass MSE to 0.014480/0.001279 and transmission to 0.015557/0.001711
(linear/log1p). Three-seed volume MSE becomes 0.083620/0.003405, with mean
1.018315 versus CPU-guided reference 1.018023. These improve on the previous
resampling revision but still trail CPU guiding at equal sample count. Timings
are explicitly invalid for comparison because other validation/compilation was
running. A one-seed glass memory experiment increases the field from 256 MiB to
1024 MiB and changes MSE from 0.014372/0.001278 to 0.014115/0.001261; this small
single-seed benefit is insufficient to change the default memory budget.

A new `cycles_metal_guiding_texture_scene.py` creates an actual tiled, mipmapped
`.tx` image and textures the room and glass objects. Glass roughness depends on
view direction, exercising reciprocal shader reconstruction. The generated asset
is retained beside the editable scene. This expands cache-miss coverage beyond
the earlier constant-material tests; two CPU-guided reference seeds are rendered
at 8192 samples. GPU validation remains pending.

The directional-moment build passes all fifteen persistent-session checks. In
three-seed indirect-room tests at 256×256 and 512 samples, PT linear/log1p MSE is
0.014295/0.001222 with guiding versus 0.024485/0.004248 without it. BDPT is
0.009706/0.000466 versus 0.012415/0.000720. The BDPT means (guided 0.668311,
unguided 0.667715) are below CPU reference 0.670124; a high-sample convergence
check is required before interpreting this difference. These are quality-only
runs, not valid timing comparisons.

### Reciprocal shader texture-cache retries

The first tiled-texture BDPT render completes and records real cache misses.
Cold and pre-rendered 1024-sample means are 0.292944 and 0.292834, respectively,
against the first CPU reference seed 0.291833. These alone do not establish bias
or prove cache correctness. Code review identified a concrete late-miss problem:
reciprocal evaluations could request tiles after camera film/shadow writes, and
light generation could expose an incomplete cache without a retry.

The working source now records completed camera surface stages (initial shading,
NEE, vertex connection, scattering). A texture retry reconstructs closures, skips
already committed work, and resolves the reverse shader before changing the ray,
throughput or training anchor. Light generation reports missing emitter, surface,
volume and reverse-shader tiles and rebuilds incomplete reservoirs before sensor
or camera work. Sensor connections retain a completion marker in previously unused
vertex padding, preserving the 96-byte vertex size, so successful queued shadows
are not repeated when another vertex needs a tile. Scenes without tiled images
retain the previous asynchronous generation/sensor queue path. This revision is
under build and render validation; it is not yet an accepted result.

The user also explicitly identified the matched-time comparison image (CPU 6.03 s,
Metal 6.09 s) as clearly better on Metal. Preserve this validated visible quality
improvement as an acceptance case. The five-seed error reductions support that
observation; remaining same-sample regressions on other materials must not obscure
this successful practical-budget comparison.

The cache-retry build passes all fifteen persistent-session checks. Its persistent
textured test records four light-generation and two sensor-connection cache
retries during the first render, and no such retries during the retained-data
second render. The captured means are 0.292689 and 0.292851, versus two-seed CPU
reference 0.291699. Both images are retained. The first capture attempt had a
Python callback variable-shadowing error; it is not used for comparison. The
fixed runner explicitly fails if its first image is absent. The earlier ordinary
warmup did not retain texture data, so it must not be described as a loaded-tile
comparison.

A controlled roulette ablation on the frozen directional-moment build removed
only the light-side survival factors from the MIS recurrences, preserving the
roulette policy and throughput compensation. It did not resolve the indirect
room's high-sample deficit: guided BDPT at 64×64 and 4096 samples had mean
0.666889 versus 0.667017 before the ablation and CPU reference 0.669792. Because
guiding adapts to contributions, subsequent sampled paths need not be identical.
A numerical bidirectional composition test confirms that different roulette
policies can share a deterministic MIS partition that omits survival factors,
provided each contribution retains its survival compensation.

The next transport revision addresses a separate physical inconsistency.
Projected-area Jacobians use geometric normals, whereas the previous recursive
MIS terms used shading normals. Further, an eta-only light-throughput correction
cannot transpose arbitrary direction-dependent shader values. Continuous light
continuations now reuse the full reciprocal BSDF evaluation and convert its
geometric projected-area measure. Dirac events use the analytic shading-normal
and dielectric correction. Camera/light connection endpoints use the same
measure conversion. This follows the distinction in
[PBRT's adjoint shading-normal treatment](https://github.com/mmp/pbrt-v3/blob/master/src/integrators/bdpt.cpp)
between scattering values and geometric area densities. Roulette probabilities
are omitted consistently from both MIS recurrences while retaining survival
compensation in throughput. These are transport corrections, not image scaling.

Eighteen distribution/field/resampling tests and three actual-BSDF transport tests
pass. Transmission tests now include tilted shading normals and check more than
6000 actual GGX samples in both entering/exiting directions against the reciprocal
transport identity. Full-render validation reveals a remaining mismatch: the
64×64 indirect-room guided BDPT render at 4096 samples has mean 0.672081 versus
the CPU-guided reference 0.669792. The flat-normal Lambertian control, with guiding
disabled, also differs: 0.478665 versus the two-seed 8192-sample CPU-guided
reference 0.476864. This isolates the issue from learned guiding and layered
closures; it does not yet identify its cause. Higher-sample CPU and ordinary
Metal PT controls are being rendered. These BDPT results are not accepted as
correctness evidence merely because their noise is lower.

### Final comparison protocol — user clarification

The user subsequently requested **equal sample counts**, not equal render time,
for final CPU/GPU and GPU/GPU comparisons. Final quality acceptance therefore
uses matching SPP and render settings, with elapsed time reported separately.
High-sample CPU-guided images remain the primary reference. The earlier
matched-time image remains a historical result and must not substitute for this
equal-sample acceptance criterion. Retain and report same-sample regressions.

The Lambertian convergence audit confirms a BDPT discrepancy independently of
guiding. Two CPU-guided 65,536-sample runs (seeds 307 and 401) average 0.477203.
Ordinary Metal PT at 16,384 samples (seed 211) averages 0.477197; unguided Metal
BDPT at the same 16,384 samples and seed averages 0.478687, approximately 0.311%
higher. The earlier 4096-sample BDPT seed averaged 0.478665. Raw results and
`transport-convergence.json` are retained under `adjoint-lambert`. These diagnostic
runs overlapped compilation/CPU work and support no timing comparison. A
direct-light-only depth control now explicitly records its modified depth in
the metrics and saves the scene, to separate sensor/NEE transport from longer
light subpaths. Its default configuration remains the original Lambertian room.

The direct-only control also differs: at 16,384 samples and seed 211, CPU guided
has mean 0.0570279, ordinary Metal PT 0.0570242, and Metal BDPT 0.0572571. A
65,536-sample CPU-guided seed 401 gives 0.0570382. Positive differences extend
across the illuminated ceiling, rather than being confined to isolated bright
pixels. This excludes long light continuations and roulette as necessary causes.
A frozen-app diagnostic (`hash-rng-ablation-Blender.app`, with a SHA256 manifest)
scrambles only the LCG floating-point output with `hash_uint`, retaining its
state recurrence. This experiment is pending and is not a production change.

The source support-PDF guard (zero proposal density where the continuous base
BSDF has no support) builds successfully and passes the host distribution and
actual-BSDF transport tests. It has not yet been installed; the above renders
use the earlier installed adjoint revision. Benchmark reports now explicitly
identify equal camera SPP and list observed linear/log-space regressions against
CPU guiding and unguided Metal independently of runtime. Equal SPP does not mean
equal ray count when BDPT adds light subpaths.

### Camera clipping diagnosis

The direct-only discrepancy tracks the near clipping distance. On the unchanged
adjoint build, changing `clip_start` from 0.1 to 0.001 changes the BDPT mean from
0.0572571 to 0.0570394, consistent with the CPU-guided reference 0.0570382.
Inspection identifies a concrete measure mismatch: `camera_sample_perspective`
advances the ray origin to the near plane, but the primary BDPT hit conversion
used that shortened intersection distance while sensor connections use the full
camera-to-surface distance. The corrected perspective/panorama initialization
compensates this distance in the inverse camera PDF before area conversion;
orthographic area density remains distance-independent.

With the default near clip unchanged, the corrected direct-only BDPT render
at 16,384 samples and seed 211 has mean 0.0570372. A new numerical transport test
checks the common BSDF-hit/NEE/sensor MIS partition over four clipping distances.
All eighteen host sampling tests and four transport tests pass. The installed
build includes this correction and the support-PDF guard. Larger-clip and full
indirect controls are still under validation. General perspective DOF/motion
initialization still uses the pre-existing footprint approximation and needs a
separate exact camera-density treatment; do not describe it as covered by this
pinhole validation.

The RNG-output ablation did not materially change the direct-only result
(mean unchanged at float precision; maximum pixel difference 4.77e-7). Its
rendered array differs from the original, and the clone's addon/source location
was verified. It is retained as a diagnostic, not adopted. Optional
`CYCLES_BDPT_DIAGNOSTICS=1` now reports actual emitted/cached/sensor-shadow counts
for the first few batches, with readback overhead restricted to that diagnostic.

The corrected direct-only means remain stable at near clips 0.1 and 1.0:
0.057037203 and 0.057037216. Diagnostics for the full-resolution batch report
16,384 emitted paths, 15,814 cached vertices, and 9740 queued sensor shadows.
`clip-lambert/clip-comparison.json` retains equal-16,384-sample comparisons against
the 65,536-sample CPU-guided reference. Brightness is corrected, but direct-only
BDPT MSE is 6.79e-7 versus CPU-guided 1.90e-7 at the same SPP: this remains a
measured variance regression, not a quality pass. The full indirect 16,384-sample
control is still running. All these diagnostic timings are excluded from speed
claims.

The full indirect Lambertian control completed at 16,384 samples: corrected BDPT
mean 0.477098 versus the two-seed CPU-guided reference mean 0.477203 (previous BDPT
0.478687). This removes the persistent positive offset in this scene; broader
material and guiding acceptance still requires its own measurements.

Perspective camera initialization now reconstructs the actual aperture sample
from the clipped ray, interpolates the camera transform and perspective projection
at the path time, and computes the image/focus-plane solid-angle Jacobian.
This replaces the previous differential-footprint approximation for perspective
DOF and motion. A numerical test compares the analytic density against finite
differences of the sampled direction over aperture offsets, lens shifts, and
focus distances. All eighteen host sampling tests and five transport tests pass.
The installed build includes this revision; a wide-aperture camera with simultaneous
translation, rotation, and focal-length motion is under render validation against
CPU-guided references. The Lambertian runner records all camera controls in its
metrics and can save the exact scene used.

The first combined DOF/motion control completed. At equal 8192 SPP and seed 211,
CPU-guided PT has linear MSE 0.00104744 and log-space MSE 0.000127547; guided
Metal BDPT has 0.000229195 and 0.0000257614, respectively, against the ensemble
of two 32,768-sample CPU-guided references. This is a single-seed improvement
(4.57× linear, 4.95× log), not a completed multi-seed acceptance result. Means
are CPU 0.477036, Metal 0.476400, reference 0.476706. Timings are diagnostic only.

Further source review found that null-transparent events retained the surface
area conversion despite not creating a scattering vertex. Camera paths now
undo that conversion before continuing the same edge; primary rays recompute
their camera measure at the eventual scattering event. Light paths restore their
pre-hit directional measure and preserve the previous ray origin across null
events. A new two-sheet white-transparent control is prepared to check physical
invariance against CPU guiding. Sensor connections also now reject endpoints
outside the camera's near/far clipping interval, matching camera-ray support.
These transparent/visibility corrections are built separately from the above
DOF/motion render and are not yet installed or render-validated.

The null-event measure and sensor-visibility revision is now installed and its
8192-sample GPU control is running with the original default transparent limit
of eight. The corresponding CPU-guided mean is 0.475222, lower than the clear
room because the two sheets can exhaust that finite crossing budget. Therefore
this configuration is not a valid unlimited-transparency invariance test. A
separate explicit limit-64 CPU reference is being generated; the runner records
the configured limit rather than silently changing or relabeling the earlier run.

The subsequent working source also tracks the light prefix's transparent count
in the cached vertex and carries the combined prefix count into connection
shadows. Light tracing stops once its transparent budget is exhausted. The
96-byte vertex size is unchanged: its support word now allocates 8 bits to path
length, 12 to reservoir candidate count, and 12 to transparent count. Widening
the candidate field matters because mixed transparent/continuous surfaces can
produce more candidates than the scattering bounce limit. This count-propagation
revision is under build and is not yet installed or validated on Metal.

The transparent-prefix/support revision is now installed and frozen separately
as `transport-baseline-Blender.app`. A serial validation job renders the two-sheet
scene at limits 64 and 8, both at 8192 samples. The first earlier null-measure-only
GPU run at the default limit had mean 0.475815 versus CPU-guided 0.475222; it must
not be attributed to the later prefix-count fix. The limit-64 CPU reference at
32,768 samples has mean 0.476953 and is retained separately.

### Direct-light training observation experiment

The next working source averages over the independently chosen training anchor
for immediate NEE observations. Instead of discarding the NEE record whenever
that depth was not selected, it records the observation multiplied by that
depth's selection probability. Expected direct training mass therefore matches
the existing single-anchor indirect training mass; no contribution PDF or image
weight changes. Spatial refinement visits retain the original anchor predicate.
The existing shadow record fields hold the observation, so path-state memory
does not grow. A host test exhaustively checks selection frequencies and weighted
training mass, including the finite 32-bit tail. This is under build/test and
has no claimed image-quality benefit until compared at equal SPP to the frozen
transport baseline and CPU guiding.

The direct-observation revision builds and passes nineteen host sampling tests
and five transport tests. `dense-direct-Blender.app` is a controlled clone of the
frozen transport baseline with only `guiding_distribution.h` and `guiding_gpu.h`
replaced; a manifest records before/after hashes. Parallel quality-only jobs
compare both versions on rough glass and transmission at 256×256, 512 SPP, seeds
101/211/307, using the existing CPU-guided references. No speed claims apply to
these overlapping diagnostic jobs.

Inspection also found why a discarded render did not necessarily stabilize older
timings: Metal recognizes `--warm-up` as a request to finish specialization, while
the scene runner uses `--warmup`. The working Metal host source now recognizes
both spellings. This change still requires a build and performance-mode validation;
it does not retroactively validate the earlier asynchronous timing comparisons.

The direct-observation experiment did not provide a meaningful quality improvement
on the three-seed material controls and has been reverted from the working source.
Rough-glass MSE changed from 0.0143451 to 0.0144138; transmission from 0.0155980
to 0.0154563, with slightly worse log-space error in both cases. CPU-guided MSE
at the same 512 SPP is 0.0115913 and 0.0119350. Both frozen applications and their
reports remain available; the experimental clone is not the production backend.

The transparent-prefix revision completed both 8192-sample controls: GPU mean
0.476698 with limit 64 (CPU 32,768-sample reference 0.476953), and 0.474990 with
limit 8 (same-SPP CPU 0.475222). These single-seed values still require broader
statistical checks. A near-clip-20 control, excluding the entire room from the
camera's view, was also rendered to verify sensor support independently.

The excluded-room test returned exactly zero mean and peak. The extra direct
training experiment and its test helper have been removed from the working
source; the warmup-recognition fix remains. A separate frozen-app experiment,
`adaptive-ris-Blender.app`, changes only broad-surface RIS from the smooth mixture
to the already implemented adaptive histogram, preserving smooth glossy products.
Its three-seed, 512-SPP material comparison is running against the same references;
it has not been adopted or credited with a quality improvement.

The restored installed implementation passes all fifteen persistent-session checks
again, including UI visibility, training budgets, scene/light resets, probability
endpoints, all three sampling modes, and CPU coexistence. The host sampling and
transport tests and actual Metal distribution/product/resampling/field tests also
pass (`cycles-guiding-transport-final-unit.log`). Cancellation/restart validation
is being repeated. These functional checks do not erase the remaining equal-SPP
material quality regressions.

Cancellation/restart passes again, with 0.0541-second cancellation latency and
fresh training publications at samples 1/2/4/5 after restart. The adaptive-RIS
material experiment also completed without an improvement: rough-glass MSE
0.0143970 and transmission MSE 0.0156732, both slightly worse than the frozen
smooth-product baseline. It remains isolated and is not adopted.

The installed backend now replaces the single anchor with immutable GPU histories
for all eligible camera-path vertices, following
[the full-history implementation plan](cycles_metal_guiding_full_history_plan.md).
It uses bounded, drained-group lifetimes and a separate training-memory control.
The Release build, 18 host sampling tests, five reciprocal transport tests, and
actual Metal sampling/field tests pass. A new Metal test also passes concurrent
history creation, separately captured shadow prefixes, reuse, and guarded overflow
with 100,000 paths. Real-render integration and equal-SPP quality validation are
in progress; the quality gap is not yet shown to be closed.


### Full-history render validation and remaining representation gap

The initial host integration exposed a counter allocation-order error and a tile
sizing error. Both are fixed. Tile sizes now obey the history group capacity;
a pending tile that cannot fit is an error, not completion. Pre-fix material
results are retained in `history-material/INVALID.json` and excluded from quality
conclusions. The strengthened 19-case persistent render regression passes,
including a one-sample empty-field pixel comparison between 16 and 128 MiB
history budgets. The smaller budget actually drains two groups instead of one.
History and field allocations shrink when their limits are reduced. Cancellation
and restart pass with 0.03745-second latency and fresh publications at 1/2/4/5.

The corrected three-seed, 512-SPP, 256-square material comparison does not close
the CPU gap:

| Material | Metal full-history linear MSE | CPU guided linear MSE | Metal log MSE | CPU log MSE |
| --- | ---: | ---: | ---: | ---: |
| Rough glass | 0.01439930 | 0.01159134 | 0.001256622 | 0.000910410 |
| Transmission | 0.01563793 | 0.01193497 | 0.001701707 | 0.001235353 |

These are valid equal-SPP quality measurements (`history-material-fixed`), with
CPU-guided 2x8192-SPP references. Runtimes from this quality-only run are not
performance comparisons. Full-history training is functional but has not shown
an appreciable material quality improvement over the frozen single-anchor
transport baseline. Both builds remain below the requested CPU quality target.

Further research points to spatial-directional correlation as a candidate
limitation, not a proven diagnosis. Cycles' CPU default is parallax-aware VMM;
the GPU's current regional mixture has no positional correction.
[Robust fitting of parallax-aware mixtures](https://doi.org/10.1145/3386569.3392421)
and [spatio-directional mixture models](https://research.nvidia.com/publication/2021-12_path-guiding-using-spatio-directional-mixture-models)
model this correlation. A separate CPU VMM/parallax-aware VMM diagnostic is
running on the same saved scenes and sample counts. It does not change the
primary reference or acceptance target.


The CPU model diagnostic completed with both developer controls enabled (the
first attempt is marked invalid because Cycles ignored its model selector).
Plain VMM gives rough-glass MSE 0.01361027 and transmission MSE 0.01467480; the
parallax-aware default gives 0.01159134 and 0.01193497. All six default-mode
images match the primary CPU runs exactly. This supports investigating
position-conditioned proposals; it does not relax the CPU reference.

The same audit found that the public Metal sampling-mode UI still used a
sync path guarded by developer preferences. Sampling-mode synchronization now
runs independently of that gate. The strengthened 19-case regression passes;
logs confirm effective modes 0/2/1 for Product MIS/Roughness Weighted/Product RIS,
and MIS/RIS images differ. Earlier non-developer mode-toggle smoke tests did not
actually exercise distinct modes. Default-RIS quality benchmarks remain valid.

Full-history BDPT with tiled textures completed cold and persistent renders,
exercising eight light-generation cache retries. Both images are finite; their
mean values are 0.290740 and 0.290340. Their stochastic difference has not yet
established contribution-equivalence or a CPU-reference quality result. Timing
was not isolated, so this run does not establish a performance gain.

The next step is specified in
[the position-conditioned guiding plan](cycles_metal_guiding_position_plan.md).


### Position-conditioned fitting checkpoint

The position-dependent model and its exact equal-SPP results are documented in
[the position plan](cycles_metal_guiding_position_plan.md). It is implemented and
passes 22 host distribution/field tests, five transport tests, real Metal sampling
and fitting tests, 19 persistent-render cases, and surface/volume BDPT probes.
At 512 SPP, three seeds, it reduces the previous GPU material MSE by about 6%,
but remains 15% worse than CPU guiding on rough glass and 22% worse on
transmission. The final quality requirement remains unmet. The old matched-time
figures do not override these equal-SPP results; final timings still require
isolated warmed measurements. Full BDPT transport limitations remain open.
