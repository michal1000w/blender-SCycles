# BDPT sensor-manifold correctness audit

## Observed discrepancy

The corrected-cache 512-SPP material runs have lower guided BDPT image MSE than
CPU guiding, but both guided and unguided BDPT are about 1.1% brighter in the
tall glass-block region. This discrepancy is separate from the two corrected
cache bugs and is not specific to guiding.

An 8192-SPP guided transmission render, seed 101, completed with its sample
count verified from the log. Against the unchanged average of CPU-guided
8192-SPP seeds 991 and 1993, the block mean is 0.1496246508 versus 0.1478509044:
a 1.1997% difference. The sphere mean differs by 0.2069%. Full-image MSE is
0.0009924692. The raw image and region report are retained in
`build/metal-guiding-tests/cache-order-convergence`. One high-sample seed does
not supply an independent-seed uncertainty estimate, but the discrepancy has
persisted as sample count increased.

## Isolated diagnostic

`cache-order-no-sensor-manifold-diagnostic-Blender.app` is a copy of the tested
application with only the sensor manifold branch disabled in its runtime kernel
source. The source and installed application are unchanged. Original precompiled
libraries are preserved outside the diagnostic application's library lookup path
so the changed source must compile. Binary archives are disabled for the run.
The original and modified header hashes are recorded beside the application.

The diagnostic uses the same transmission scene, seed 101, 512 SPP, 128 training
samples and 256 MiB field budget as the corresponding corrected-cache render.
Its purpose is to isolate the contribution responsible for the brightness
discrepancy. It is not a feature-complete alternative or a proposed production
fix. Timing is not admissible performance evidence.

The guided 512-SPP ablation completed. Its block mean is 0.1486907284,
compared with 0.1494751213 for the same-seed original and 0.1478509044 for the
CPU reference. Its MSE is 0.0112102145 versus 0.0106404551 for the original.
Thus removing the branch reduces this regional offset without eliminating it,
and increases observed image MSE. Retraining also changes the proposal, so this
is not an exact per-path contribution difference. An unguided paired run is
being used to isolate that contribution without training changes.

The unguided paired diagnostic is now complete. Removing sensor-manifold sampling
changes the tall-block mean by only 0.00006112 (original 0.15011763, ablation
0.15005652, CPU reference 0.14785090). Whole-image difference MSE is
3.485643e-9. This rules out that branch as the main source of the observed
brightness offset. The guided ablation's larger change includes the effect of
retraining. The ordinary BDPT transport and weighting audit must therefore
continue; the manifold ablation is not adopted.

## Independent manifold sampling defect

The audit also found that every rough interface requested the same two random
dimensions. The path contribution multiplies per-interface slope-density terms,
but repeated dimensions sampled correlated slopes instead of that product
distribution. `sample/manifold.h` now assigns a separate bounce-sized dimension
block to each interface, retaining the original dimensions for interface zero.
The indexing is immutable, so texture retries reproduce the same samples.

A distribution test uses the real scrambled Sobol sampler: the sum of three
centered uniform coordinates should have second moment 3/12. The old addressing
gives 0.74999970 instead of 0.25 and cross moment 0.08333330 instead of zero.
The corrected addressing passes the original tolerances. Actual Metal execution
with 65536 samples gives second moment 0.251924708 and cross moment 0.000874800.
All 36 guiding and six transport tests pass, as does the complete actual Metal
unit suite. A Metal-reserved parameter name caught by compilation was corrected
before the successful run. The Release build and install succeed; all 19 renderer
regressions on this exact installation pass. This correction is separate
from the unresolved ordinary-BDPT brightness discrepancy.

An additional unguided Metal PT control completed 8192 samples. Its block mean
is 0.14643474, 0.9578% below the CPU-guided reference, while its sphere mean is
0.6057% below. These single-seed unguided differences are not sufficient to
establish bias or replace the guided reference, particularly for difficult
refractive paths. A guided Metal PT control at the same count is running to
separate finite-sample coverage from a shared renderer discrepancy.

That guided PT control completed 8192 samples. Its block mean is 0.14777127,
only 0.0539% below the CPU-guided reference; its sphere mean is 0.1628% below.
The whole-image mean is 0.66491022 versus 0.66492859 reference. This reinforces
the need to use the guided reference instead of treating missing energy in one
unguided realization as ground truth. It does not establish the required
equal-512-SPP PT quality parity. The separate BDPT offset remains unexplained.

The corrected renderer's cold/warm texture check also passes the unchanged pixel
tolerance after three generation retries and one sensor retry; its difference
MSE is 1.835845e-15 (`manifold-rng-texture/comparison.json`).

## Ordinary light-tracing diagnostic

`manifold-rng-lt-only-diagnostic-Blender.app` isolates the ordinary light-tracing
estimator: camera throughput is zero, sensor contribution weights omit MIS, and
sensor-manifold sampling is disabled. Reservoir inverse-inclusion weights and
light emission probabilities remain intact. This is an instrumented diagnostic,
not a replacement implementation. Precompiled original libraries are excluded
from lookup, and all edited source hashes are recorded beside the application.

The 8192-sample transmission run will be compared in the rough-surface object
regions. Directly visible emitters are outside this diagnostic's support and
must not be used to claim whole-image correctness. If ordinary light tracing
matches the reference while combined BDPT does not, the strategy weights become
the next focus; otherwise, the light-side transport needs further isolation.
No result from this diagnostic is a final quality or performance benchmark.

## Weighting audit

The sensor connection code can replace a direct connection by a solved refractive
chain. It then uses the ordinary sensor MIS expression with cached light-side
recurrences. That expression needs to be audited against the added strategy's
density and the complementary weights of camera and vertex-connection paths.
The source of `kernel_path_mnee_sample` also clamps the generalized geometry
factor to two; that inherited choice must be considered separately when proving
the estimator's convergence. Neither observation alone proves the measured
discrepancy's cause.

[Specular Manifold Sampling](https://rgl.epfl.ch/publications/Zeltner2020Specular)
distinguishes unbiased and biased constructions and discusses solution-selection
probabilities. Its explanation of MNEE emphasizes the corresponding MIS treatment
when the deterministic manifold walk fails. The
[authors' implementation](https://github.com/tizian/specular-manifold-sampling)
provides single- and multiple-scattering reference integrators. These are research
references, not code imported into Cycles.

The next implementation must account for overlapping strategies, solver support,
and the actual geometric measure. It must retain the requested transport support
and pass independent-seed CPU-reference convergence checks. An empirical
brightness correction or accepting lower MSE despite a persistent mean error
would not satisfy the objective.

### Ordinary light tracing diagnostic and next PDF audit

The isolated ordinary light-tracing render completed at 8192 SPP (seed 101).
Its sphere mean is 0.225736684 versus the CPU-guided reference 0.250421052;
the block mean is 0.130889315 versus 0.147850904. Raw data and regional
comparison are retained under `build/metal-guiding-tests/manifold-rng-lt-control`.
This diagnostic sets camera throughput to zero, sensor MIS to one and disables
the sensor manifold branch. Its deficit does not establish correctness or prove
that the combined estimator's discrepancy comes exclusively from MIS. It is
not a production alternative or a final quality/performance comparison.

Source inspection identifies a separate PDF inconsistency to test: lamp and
triangle emission-hit weights assume uniform emitter-area NEE densities, while
Cycles surface rectangle sampling uses solid angle and triangle/sphere sampling
is also conditional. The first light-vertex ratio currently corrects only emitter
selection, not the point-on-emitter density. Emission densities must stay distinct
from receiver-dependent NEE densities. A correction must cover emission hits,
sensor connections, vertex connections and first-scatter recurrence together,
including reciprocal receiver normals, transmission, time and volume context.
No partial production correction has been applied. An independent test using
the actual rectangle sampler is the next step before changing these weights.

Final acceptance remains equal SPP across CPU/GPU and GPU/GPU comparisons;
runtime is a separate measurement. High-SPP CPU-guided data remain primary truth.

The new host regression `BidirectionalPDF.RectangularNeeRequiresConditionalAreaDensity`
uses the actual `area_light_rect_sample` on 64 endpoints of a 2-by-2 emitter at
distance one. Its PDF agrees with the independently computed solid angle
`4 atan(1/sqrt(3))`. Independent camera-hit, NEE and light-tracing densities in
common area measure give weights summing to one within 1e-12. Substituting uniform
emission area for conditional NEE in the hit and sensor weights creates a maximum
partition error greater than 0.05. This confirms the density requirement; the
production renderer has not yet been corrected and is not covered by that
partition assertion. All 36 guiding and 7 transport host tests passed in
`/tmp/cycles-guiding-nee-density-unit-final.log`. The initial compile failure
(unqualified `fabs` resolving to a Cycles vector overload) is retained in
`/tmp/cycles-guiding-nee-density-unit.log` and was corrected with `std::fabs`.

### Area-light renderer correction in validation

Area-light camera emission hits now use the actual `LightEval.pdf` converted
to area measure for the NEE competitor, while retaining the emitted direction
density. First light-vertex sensor, VC and recursive-scattering weights apply
the conditional area-light point density relative to uniform emission area,
in addition to tree/flat emitter selection. Evaluation calls the existing
`area_light_eval_from_intersection`, including ellipse and spread behavior.
The emitter endpoint is retained exactly rather than reconstructed from a packed
direction. `KernelBDPTVertex` grows from 96 to 112 bytes; dump records before
`sensor_complete` grow from 84 to 96 bytes. Historical snapshots retain their
original format. Allocation uses the actual type size.

For volume NEE, `shade_volume.h` first selects the emitter using the ray segment
then resamples its position with `light_sample<false>` at the actual scattering
position. The shape-density query therefore uses the scattering position,
while the tree query retains its original segment context.

This change is limited to the area-light defect: triangle and finite spherical
emitter conditional PDFs remain to be corrected. This is an intermediate
implementation, not a claim that all BDPT MIS or transport is complete.
The extended host test checks the same area-light PDF evaluator against the
sampler. All 36 guiding and 7 transport tests and the release build/install pass.
Actual Metal render validation is pending in `/tmp/cycles-guiding-area-nee-render.log`.

The corrected area-light Metal render completed at verified 512 samples and a
16384-vertex cache. For seed 101, transmission block mean changes from 0.149475121
to 0.148837196 (CPU-guided reference 0.147850904), reducing the relative offset
from +1.0986% to +0.6671%. Sphere mean instead changes from 0.251141166 to
0.252514861 (reference 0.250421052). MSE is 0.010748607 versus previous BDPT
0.010640455 and CPU guided at equal 512 SPP 0.011722157. This mixed single-seed
result is not correctness acceptance. The earlier BDPT comparison also predates
the independent MNEE RNG correction, so it is not an isolated paired ablation.
All raw comparisons remain in `build/metal-guiding-tests/area-nee-correction`.
An 8192-SPP convergence render is underway; timing is not assessed here.

The 8192-SPP corrected run completed. Block mean 0.149723257 is still +1.2664%
above the CPU-guided reference; sphere mean 0.250744594 is +0.1292%. MSE is
0.000992604. This does not resolve the material-scene brightness discrepancy.
All 19 general persistent-session cases passed; BDPT-specific cache retention
is a separate running check.

Coverage correction: the material fixture inherits a DISK area light from the
base scene, with a black environment. Its usual NEE point sampling is uniform
in area, so this render does not directly exercise the rectangle-density defect.
The square-sampler host test remains relevant, but an explicit rectangular
render fixture is needed. The shared scene now accepts `--light-shape SQUARE`;
its default stays DISK, preserving all existing CPU reference scenes. Do not
attribute noisy changes in the disk fixture to the rectangle correction.

The BDPT retention check passed with history budgets 16/128 MB and an intermediate
64-pixel resize: all 4096 canonical cache records (96 bytes each) are identical;
image maximum difference is 9.536743e-7 and MSE 1.803423e-16. The explicit square
fixture is now rendering under `build/metal-guiding-tests/rectangle-nee-validation`
with a CPU-guided 8192-SPP reference seed 991 and corrected Metal BDPT 512-SPP
seed 101. Additional reference seed and equal-SPP controls remain required.

### Finite spherical emitter PDF correction

The worktree now evaluates actual point/spot spherical NEE PDFs at the reciprocal
receiver and converts them to emitter area measure for first-light-vertex weights.
Camera lamp-hit weights use the existing evaluated NEE PDF for spheres as well
as area lights. Flat emission-area/direction densities remain separate.
This does not establish full transport support inside spherical emitters or
resolve the disk-scene discrepancy; triangle/motion emission PDFs remain pending.

`SphericalNeeAreaDensityMatchesSampleJacobian` uses the actual point sampler
and intersection PDF evaluator at 64 parameter locations. An independent
finite-difference surface Jacobian checks the solid-angle-to-area conversion.
The first derivative step 0.001 narrowly exceeded 0.5% tolerance on concentric
disk diagonal points; refining the step to 0.0005 passes the unchanged tolerance.
Both logs are preserved as `/tmp/cycles-guiding-sphere-nee-unit{,-refined}.log`.
All 36 guiding and 8 transport host tests pass. Release build is underway;
the installed renderer remains the prior area-light version until validation
inputs finish. Actual Metal sphere validation is still required.

The first square-light CPU reference completed at verified 8192 SPP. An interim
Metal-512 comparison gives MSE 0.010623232, sphere +0.9012% and block +0.7290%
against that single reference. This is not an equal-SPP quality comparison;
the second CPU reference seed 1993 and CPU-512 seed 101 are now running sequentially.
The spherical PDF build is installed. Its actual point-light Metal check first
failed before rendering because the smoke script did not create the `--save-scene`
parent directory. The script now creates it, and the rerun is underway. The initial
log `/tmp/cycles-guiding-sphere-point-metal.log` is retained, with rerun output in
`/tmp/cycles-guiding-sphere-point-metal-final.log`. No sphere rendering acceptance yet.

### Triangle emission and conditional NEE implementation

`triangle_light_emission_pdf` computes fixed-CDF selection mass divided by
shutter-time triangle area. Photon emission PDFs and flux now use this density;
so do the emitted-path alternatives in surface and volume NEE. Triangle
emission-hit weights use `triangle_light_pdf` for the NEE competitor, with its
flat/tree selection convention preserved. The first light-vertex ratio also
evaluates that actual sampler at the retained emitter endpoint and shutter time.
The temporary ShaderData sets the geometry fields used by the PDF evaluator.

The release build passes, but these changes are not installed or render-validated
yet. Existing 44 host tests predate this implementation and must not be cited
as triangle correctness evidence. The smoke scene now provides `--emitter-motion`
for mesh lights, scaling the emitter over the shutter interval. The fixture is
syntax-checked but still needs actual execution and CPU-guided comparison.

Square-light inputs are complete: two CPU-guided 8192-SPP references and CPU/GPU
512-SPP seed 101 controls, with all sample counts verified. MSE is 0.010364285
for CPU guided and 0.010311695 for Metal guided BDPT. The small (~0.5%) Metal
MSE advantage does not establish correctness: sphere/block regional offsets
are +0.9965%/+0.7048% for Metal, versus +0.2415%/-0.0404% for CPU at 512 SPP.
The common-transform `equal512.png` was inspected. All raw inputs and metrics
remain under `rectangle-nee-validation`; no timing claim is made.

Spherical point-light Metal execution completed at verified 512 samples with
finite nonempty output (mean 0.160275589). CPU reference comparison is pending.
The triangle build is now installed; `emitter-nee-validation/run.py` runs moving
mesh and spherical spot cases on Metal and CPU-guided references/equal-SPP
controls for point, moving mesh and spot. These concurrent runs are quality
checks only, and do not yet constitute triangle or sphere correctness acceptance.

CPU controls for point, moving mesh and spot have completed (each 8192-SPP
reference seed 991 and 512-SPP control seed 101). The OIIO analyzer preserves raw
NPY data and verifies sample counts. Point reference mean is 0.160250670; Metal
BDPT-512 mean 0.160275589 differs by +0.01555%. However, its MSE 7.098274e-6
is about 7.7 times CPU-guided-512 MSE 9.232944e-7. This fails the requested
quality target, even though mean brightness is close. An ordinary guided Metal
PT-512 control is running to distinguish guiding from bidirectional sampling.
Moving-mesh and spot GPU validation remain in progress.
