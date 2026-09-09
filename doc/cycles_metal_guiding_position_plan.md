# Position-conditioned Metal guiding

## Evidence and scope

The full-history trainer passes the strengthened 19-case persistent-render suite,
actual Metal history lifetime/overflow tests, and cancellation/restart, but its
three-seed 512-SPP material errors remain about 24% (rough glass) and 31%
(transmission) above CPU guiding. A controlled CPU representation diagnostic
isolates part of that gap. Disabling parallax-aware VMM while preserving the saved
scene and sampling settings increases CPU linear MSE from 0.01159134 to 0.01361027
and from 0.01193497 to 0.01467480 respectively. Restoring parallax-aware VMM
reproduces all six original CPU images exactly.

Sources: [Ruppert et al. 2020](https://doi.org/10.1145/3386569.3392421) and
[Dodik et al. 2021](https://research.nvidia.com/publication/2021-12_path-guiding-using-spatio-directional-mixture-models)
show the importance of spatial-directional correlation. The Apache-licensed
[Vulkan reference](https://github.com/FelixFifi/rtx-pathtracer) evaluates guided
proposals on the GPU but fits its models on the host (`src/PathGuiding.cpp`). It
is not a GPU training backend to drop into Cycles. No code has been copied from
that project. The implementation below is an independently implemented,
regularized conditional-moment model; it is not a claim to reproduce those papers'
complete EM, anisotropic mixture, or model-selection algorithms.

## First: accumulate complete path observations

Before this change, history traversal immediately added each downstream contribution
fragment to the field. Its radiance mass is additive, but fragment counts are not
independent path-observation counts and its moment atomics grow with path depth.
Add one mutable scalar radiance accumulator to each otherwise immutable history
record. Film/shadow hooks atomically accumulate their nonnegative, finite weighted
radiance in each ancestor. At the drained-group boundary, a GPU kernel records
one complete radiance observation per vertex before the pool is reused. Retain
immediate NEE/vertex-connection observations and keep rendered throughput and PDFs
unchanged. This reduces repeated field moment updates and gives the adaptive
count tree more meaningful counts. Validate aggregation, shadow prefixes, pool
reuse, overflow, cancellation, and pixel/sample preservation before relying on it.

## Conditional directional model

Extend the per-component training statistics to include the sample position and
its covariance with outgoing direction. Normalize world positions by the scene's
guiding bounds to avoid sensitivity to scene translation and scale. For weighted
observations (p, d, w), collect W, sum(w p), sum(w d), symmetric sum(w p p^T), and
sum(w d p^T). Directions are unit length, so their total second moment is known.
Fit a ridge-regularized local regression:

A = Cov(d,p) (Cov(p,p) + lambda I)^-1

mu(p_query) = normalize(mean(d) + A (p_query - mean(p)))

Estimate concentration from the residual directional variance. Use positive
regularization, finite/positive-definite checks, sufficient observation support,
and the existing broad fallback when a fit is unsupported. All queried components
remain normalized vMF distributions with nonnegative mixture weights and the
existing exploration component. Sampling, product construction, incident-density
targets, and all forward/reverse proposal evaluations must use the same query
position. This is a change to the proposal only, never a replacement for the
rendered BSDF or a radiance cache used as the image estimator.

Include positions in main-history and immediate-shadow training records, and in
camera/light importance observations. Field subdivision may inherit global
normalized-position moments without changing their coordinate system. Memory
accounting must include every added accumulator and sampling coefficient. The
user's field and training memory limits remain actual allocation bounds.

## Validation and acceptance

* Host and real Metal tests for planar/rank-deficient position data, constant
  fields, translated/scaled scenes, near-field directional changes, nonfinite
  inputs, and bounded concentrations.
* Numeric integration and sample/PDF agreement for conditioned mixtures and
  BSDF/phase products at many query positions.
* Actual rendering with all three exposed sampling modes, surface/volume paths,
  BDPT forward/reverse PDFs, null events, texture retries, cancellation, and
  persistent field/history resizing.
* Equal-SPP three-or-more-seed comparisons against the frozen full-history and
  single-anchor GPU builds and the unchanged default CPU-guided references.
  Preserve scene scripts, raw linear data, detail regions, and log-space errors.
  Keep every regression. Do not credit a CPU model ablation as a weaker reference.
* Serial warmed runtime measurements, including training, field updates, and
  BDPT light work. Quality tests with concurrent jobs do not establish speed.

## Implementation checkpoint

Group-level radiance aggregation is implemented and passes the 19-case actual
Metal persistent-render suite. At 512 SPP, three seeds, its material linear MSE
is 0.01422379 (rough glass) and 0.01545320 (transmission), still above unchanged
CPU guiding at 0.01159134 and 0.01193497. Aggregation alone does not meet the
quality target.

The position-conditioned model is implemented in surface, volume, and shared
forward/reverse BDPT proposal evaluation. Each component collects 23 statistics,
including squared weights for effective sample size. Published components use 18
floats; history records use 40 bytes. Allocation and group sizes account for
these sizes. Inheritance and decay scale squared-weight sums quadratically.
Unsupported fits retain the existing direction-only model and exploration.

The final host build and 22 distribution/field tests plus five transport tests
pass. Actual Metal tests pass for fitting, conditioned sample/PDF agreement,
concurrent history aggregation, pool reuse, and guarded overflow. Scene translation and scales of 0.001, 1, and 1000 preserve the fitted
query directions within the checked tolerance. The new
conditioned mixture tests integrate PDFs at different query positions and compare
sample moments against numerical expectations. All 19 complete-render persistent regression cases pass. A 64-SPP training
probe contains 694 position-conditioned surface components across its spatial
leaves; the snapshot is finite and its mixture/tree masses are normalized.
Image quality and runtime gains have not yet been established. Do not
mark the larger goal complete on the basis of these functional tests.


## Equal-sample rendering outcome

`position-material/report.json` records 256×256, 512 SPP, seeds 101/211/307,
training 128, and a 256 MiB field. CPU guiding was rerendered at the same 512 SPP;
its images reproduce the prior CPU results. The reference remains the unchanged
two-seed, 8192-SPP-per-seed CPU-guided image.

| Scene | Metal unguided MSE | Previous aggregate GPU MSE | Position GPU MSE | CPU guided MSE |
|---|---:|---:|---:|---:|
| Rough glass | 0.02432302 | 0.01422379 | 0.01333217 | 0.01159134 |
| Transmission | 0.02695064 | 0.01545320 | 0.01460150 | 0.01193497 |

The new model reduces linear MSE on each of the six GPU scene/seed pairs.
Mean reductions are 6.3% and 5.5%; paired difference standard errors are
0.0004077 and 0.0003679. Three seeds are preliminary evidence, not a universal
variance-reduction guarantee. Compared with CPU guiding, it still has 15.0% and
22.3% higher linear MSE, and 26.3% higher log1p MSE on both scenes. This misses
the user's acceptance target. All raw images, region errors, provenance, and
512-SPP comparison figures are retained. Timings are not performance evidence.

Surface and volume BDPT probes also complete with finite images. Their learned
fields pass finite/mass checks and contain conditioned radiance and importance
components; the volume probe includes 338 conditioned volume-radiance components
and 456 volume-importance components. These probes establish integration, not
full BDPT convergence or completion of its unsupported transport classes.

The installed app has been frozen as `position-baseline-Blender.app` with a
binary/source manifest before further model work. The remaining gap calls for
better modeling of near-field incident radiance. Physical hit-distance parallax
and richer multimodal fitting are candidates to investigate; neither is claimed
implemented by this checkpoint.
