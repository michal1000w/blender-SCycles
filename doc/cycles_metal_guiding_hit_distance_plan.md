# Physical parallax and training robustness investigation

The position-conditioned build is frozen and validated, but the 512-SPP
three-seed material errors remain 15% and 22% above unchanged CPU guiding.
This document records the next investigation, not a completed feature.

## Source audit

OpenPGL upstream was inspected at commit
`b4e4b861bdeaef391cfd4e0a27afa6a60f0d5b53` in a separate research checkout.
Its [path-segment preparation](https://github.com/OpenPathGuidingLibrary/openpgl/blob/b4e4b861bdeaef391cfd4e0a27afa6a60f0d5b53/openpgl/data/PathSegmentDataStorage.h)
floors the training PDF at 0.01 and caps intermediate training throughput at 10.
The bundled dependency is version 0.7.1; its tag at
`5c50cd7a331647a69e28cc8a19ff204039cbe78d` was checked separately and has the
same 0.01 PDF floor and throughput cap. It propagates virtual source distance backwards through smooth/specular events,
including a refraction Jacobian; rough events end that propagation. Its
[parallax-aware fitter](https://github.com/OpenPathGuidingLibrary/openpgl/blob/b4e4b861bdeaef391cfd4e0a27afa6a60f0d5b53/openpgl/directional/vmm/ParallaxAwareVonMisesFisherWeightedEMFactory.h)
reprojects observations to a shared spatial pivot, performs soft mixture
assignment, and estimates component distances using weighted harmonic means.
A nearest-hit-only distance implementation would not reproduce this behavior.
No upstream code has been copied into Cycles for this investigation.

The public [openpgl-cuda fork](https://github.com/thomsontg/openpgl-cuda) was
inspected at its default-branch commit
`979db0cc37d53dc67722d9c59a8045cb8200170d`. That checkout has no CUDA source or
CUDA kernel annotations and is the old v0.5.0 documentation checkpoint. Its name
is not evidence of a usable GPU training implementation. Other branches were not
inspected, so this is a finding about the checked-out branch only.

[Multiple Importance Reweighting for Path Guiding (2025)](https://zhiminfan.work/mi_reweight.html)
is another relevant approach, but changes how estimates from multiple training
iterations are combined. The authors describe consistency and small finite-sample
bias. It is not being used as an unannounced image reweighting or postprocessing
step. The present investigation preserves the existing image estimator.

## Controlled robustness trial

`training-robust-Blender.app` is a clone of `position-baseline-Blender.app`.
Only the inverse PDF used to construct a GPU training observation changes from
`1/pdf` to `1/max(pdf,0.01)`. The actual path contribution, forward/reverse MIS
PDFs, sampling procedure, scene scripts, CPU reference, and SPP remain unchanged.
This is proposal-fitting regularization; it does not clamp the rendered image.
It does not claim to reproduce OpenPGL's complete throughput regularization.

Evaluate rough glass and transmission at 256x256, 512 SPP, training 128,
256 MiB field, seeds 101/211/307. Compare all raw images and both error metrics to
the frozen position build and the unchanged equal-SPP CPU controls. Retain any
regression. Do not adopt or claim a gain until the experiment completes.

## Physical distance implementation requirements

A complete extension needs more than a scalar attached to the last scattering
vertex. Preserve the distance transformations through intermediate delta and
smooth events and the contribution-specific endpoints of NEE, emission, and
bidirectional connections. Environment and distant sources need a controlled
infinite-distance representation. Pure null boundaries must not introduce false
nearby sources.

The current immutable ancestry structure must retain branch safety: camera,
shadow, and shadow-catcher branches can share an ancestor while having different
endpoints. A shared mutable endpoint written by whichever branch finishes first
would be incorrect. Store immutable segment transforms and aggregate radiance-
weighted distance statistics per observation, or retain equivalent per-branch
state. Bound any added segment storage and include it in group-size accounting.
Do not silently drop delta segments or contributions to fit the memory budget.

Reproject/fitting coordinates and queried directions must use a consistent world
metric; the current anisotropic normalization is suitable for linear regression
but cannot be treated as an isotropic physical distance. Sampling and every
forward/reverse density evaluation must apply the same parallax transformation.
Every queried mixture remains normalized and keeps exploration support.

Required tests include source translation/scale, finite and infinite emitters,
reflection/refraction chains, null boundaries, volume scattering, concurrent
shadow ancestry, pool overflow/reuse, and numeric sample/PDF agreement at several
query positions. Repeat equal-SPP CPU/GPU and GPU/GPU comparisons and isolated
warmed timing after correctness. Full BDPT transport limitations remain separate
unfinished work and are not excused by a successful proposal fit.


## Robustness trial outcome

The trial completed. Rough-glass linear MSE changes from 0.01333217 to
0.01342811 (0.7% worse); transmission changes from 0.01460150 to 0.01440882
(1.3% better). Log1p MSE is 0.00115065 and 0.00155693. These mixed, small changes
do not justify adoption and do not meet the CPU target. All six unguided control
images were checked separately; exact-equality outcomes, unchanged equal-SPP CPU
controls, and the decision are retained in `training-robust-material/assessment.json`.
The workspace and installed app retain the position-conditioned implementation.

The unguided controls are not bitwise identical between the two separately
compiled Metal builds. Their pairwise image MSE is at most 2.96e-11, versus
roughly 1e-2 rendering error in this comparison; one rough-glass pixel has an
absolute difference of 0.00239. All per-seed differences are recorded, rather
than reporting these controls as exactly identical. This diagnostic does not
establish which compiler or execution detail caused the small differences.


## Shared math implementation checkpoint

`kernel/sample/guiding_parallax.h` now implements virtual-distance propagation
and a finite-source moment model. Reflection preserves downstream virtual
distance; refraction applies the cosine/eta scale; rough events terminate the
chain. Distant/unknown endpoints and overflow retain an explicit distant value.
Branch transforms are immutable and can be evaluated for different endpoints.

The source model accumulates 13 statistics: radiance weight and its square,
harmonic-distance weight and its square, weighted virtual-source position, and
its symmetric covariance. It fits a target and computes a normalized vMF proposal
axis and concentration at each query position. Both effective sample sizes and
the finite-radiance fraction must support the fit. Coincident, nonfinite,
dominated, or unsupported cases return a broad fallback. Squared-weight sums
scale quadratically under inheritance/decay. This independently implemented
moment model does not claim to reproduce OpenPGL's complete EM algorithm.

Validation: 26 host distribution/field/parallax tests and five transport tests
pass. Actual Metal tests pass for 16,384 concurrent source observations covering
point/area/distant/dominated cases, four query positions, and refraction-distance
composition. The new near-source synthetic test compares directions against
known geometry: the physical model has squared directional error below 1e-8,
while linear regression exceeds 1e-3 on that fixture. This is a math test, not an
image-quality benchmark. One initial Metal test compilation failed because the
fixture used `length` instead of the Cycles `len` helper; the corrected run passes.

These helpers are **not yet connected to render path histories or field
publication**. The installed renderer and all retained 512-SPP image results
remain the position-conditioned build. Do not report physical-parallax image
support or gains on the basis of these unit tests.

## Integration map for the next implementation step

* Retain non-null delta/smooth segment positions and virtual-distance transforms
  in the immutable history. Delta-only nodes have no directional training
  observation, but must remain in the ancestry used to propagate distance.
  Existing group capacity accounts for all non-null bounces; verify it again
  against the final record layout and shadow-catcher branching.
* Traverse each completed contribution's ancestry with its own endpoint and
  distance, accumulating radiance and distance statistics atomically. Do not add
  a mutable endpoint shared by camera/shadow branches. Aggregate a complete
  observation at the existing drained-group flush.
* Surface/light emission needs its actual intersection position. Background and
  distant-light emission need the explicit distant endpoint. Integrated volume
  emission is an extended source; do not fabricate a single collision position.
  Retain its complete directional training and use the supported broad/linear
  proposal where a compact-source fit is not supported.
* Shadow ancestry needs the camera/volume scattering endpoint even when direct
  guiding is disabled. Set this independently of the optional direct-light
  training record. Preserve it before shadow traversal modifies ray state.
  MNEE's final shadow origin is not the camera scattering point, so copying the
  shadow ray origin blindly would be wrong. Audit all creation sites, including
  vertex connections, light linking, and volume direct sampling.
* Direct observations can retain their own finite connection distance or distant
  endpoint. Forward/reverse radiance/importance proposals must query the same
  physical model using one isotropic scene metric. Existing anisotropic
  position normalization needs an explicit metric conversion.
* Extend field allocation, moment inheritance/decay, publication, snapshot
  metadata/inspection, and sample/PDF evaluation together. Then run host/Metal
  integration checks and full renders before rerunning the frozen equal-SPP
  comparisons. No rendering result has yet validated this integration.


## Renderer integration checkpoint

The physical model is now connected to the renderer. Histories use 56-byte
records: non-null delta events retain immutable virtual-distance transfers;
completed camera/shadow contributions accumulate radiance, radiance/distance,
and radiance*distance independently for each ancestor. The latter statistic
preserves source variance across different endpoints instead of collapsing a
whole observation to one invented mean endpoint. Scalar/source moments publish
only after the group drains. Group capacity uses the new record size.

Surface and finite-light emission provide intersection endpoints. Shadow source
endpoints are stored separately from mutable shadow rays, including MNEE's camera
scattering point and dedicated linked-light intersections. Volume direct lighting
and vertex connections carry finite connection distances. Background/distant and
integrated extended-volume emission retain an unknown/distant endpoint; upstream
rough events terminate that virtual-distance chain. Existing directional/linear
models retain support where the finite-source fit is unsupported. Adjoint
training uses reciprocal refraction transfers. Physical source positions and
covariances use an isotropic metric converted explicitly from field coordinates.

The field now includes 13 source statistics per component. Both squared-weight
sums scale quadratically under decay/inheritance. The existing 18-float component
layout stores a physical-source representation under mode 2; all surface/volume
forward/reverse queries use the common component evaluator. Snapshot inspection
handles all three representations and evaluates physical queries in world metric.

The final host build succeeds. Twenty-eight host tests, five transport tests,
actual Metal tests including distinct concurrent source moments, and all 19
persistent-render regression cases pass. A PT probe contains 516 physical source
components. Surface/volume BDPT probes finish with finite images and valid fields;
the volume probe contains 146 physical volume-radiance and 88 physical
volume-importance components. These remain integration checks, not proof of full
BDPT transport coverage or quality parity.

The 256x256, 512-SPP, three-seed material run regresses relative to the frozen
position model: rough-glass MSE is 0.01403240 versus 0.01333217; transmission is
0.01546398 versus 0.01460150. Log1p MSE is 0.00119755 and 0.00162080. CPU guided
controls reproduce 0.01159134 and 0.01193497. All raw results are retained in
`parallax-material`; timings are not performance evidence. This implementation
is not accepted as a quality improvement.

The implementation is frozen as `parallax-baseline-Blender.app`. An isolated
`parallax-select-Blender.app` trial requires the physical model's predicted
concentration at the data pivot to be at least the linear model's concentration,
and its axis to agree with the observed mean within the observed angular spread.
The rule uses training statistics only, not reference pixels. This checks whether
unconditional replacement by an overly broad physical model caused the loss.
The trial is not yet adopted. A richer mixture fit remains a likely next step;
fixed angular bins cannot separate multiple compact sources within one bin.


### Selection trial and finite-radiance correction

The completed selection trial does not recover the position baseline: at the same
512 SPP and three seeds, rough-glass linear MSE is 0.01374895 and transmission
0.01510607 (log1p MSE 0.00119446 and 0.00162865). It remains an isolated,
unadopted experiment. CPU controls for this trial are explicitly reused from the
unchanged physical-model run; the trial itself used `--no-compare-cpu`.

A separate adversarial test exposed a correctness defect in the constant vMF
fallback. A finite radiance sum of 1e20 makes the squared length of its raw first
moment overflow, producing a zero axis and lost PDF mass. Publication now divides
the moment by its mass before computing its length. This leaves the mathematical
fit unchanged and does not clamp radiance or alter rendered contributions.
The host regression first failed on the old code. Its assertions distinguish the
observed component from exploration components: every nonempty axis must be unit,
but only the observed component must point at the observation. A dedicated Metal
test exercises GPU publication with the same extreme finite moment.

This numerical correction is not evidence of a benchmark quality gain. The
physical source model's measured material regression remains unresolved.
Cancellation/restart with the physical build passed with 0.1004 s cancellation
latency and publications at 1, 2, 4, and 5 samples.

### Next architecture experiment: fitting multiple sources

This is a proposed experiment, not implemented functionality. Fixed angular-bin
membership cannot separate two sources in one bin. Test a learned mixture with
soft assignments before changing the production representation again. Start with
a small standalone GPU fitter and controlled one-source, two-source, area-source,
and distant-source fixtures. Require normalized sample/PDF pairs, convergence to
known synthetic distributions, and preservation of source covariance. Compare it
to both existing models on held-out observations, without reference-image input.

Renderer integration would retain complete training observations until a drained
history group can be processed, then bucket them by spatial field. A SIMD group
could evaluate component responsibilities cooperatively and accumulate sufficient
statistics locally. Observation capacity must include camera, direct, and adjoint
records, with explicit exhaustion handling. Delta ancestry records cannot become
standalone observations. Published query distributions must remain frozen across
a sampling batch: memory-group boundaries must not change rendered samples.
Working fitting state therefore needs separate storage and an explicit budget.
Repeated fitting iterations must not count the same observation multiple times.
Component initialization and splitting are necessary to represent multiple sources
within a formerly shared angular bin. A single implementation must serve camera,
volume, and adjoint queries with consistent forward/reverse densities.

Only after the synthetic fitter and capacity tests pass should this replace field
publication. Re-run the 19 persistent-render checks, cancellation, PT/BDPT probes,
and the unchanged equal-SPP material benchmark before considering adoption.

Finite-radiance correction validation completed: 29 host tests, five transport
tests, and the actual Metal suite including `GUIDING_HDR_METAL` pass. Full Blender
build and install succeed. The corrected executable has not yet repeated the
image benchmark or persistent-render suite; those earlier results are explicitly
from the preceding physical build.
