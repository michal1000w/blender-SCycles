# BDPT MIS exponent audit

Status: logarithmic balance baseline measured; a consistently transformed
power-weighted candidate is installed and under renderer validation. It is not
yet accepted. Matched-work, three-seed point-light measurements
show a predominantly random BDPT error gap that cache refresh does not resolve.
This motivates testing technique weighting; it does not establish that the power
heuristic will improve results or fix the separate material energy discrepancy.

For exponent beta, the weight is `(n_s p_s)^beta / sum_i (n_i p_i)^beta`.
The current two accumulators contain sums of *individual* strategy density
ratios. Squaring the final accumulator introduces cross terms and is wrong.
Each multiplicative density ratio must be raised to beta before combining sums.
Physical throughput and reservoir inverse inclusion factors remain unchanged.

The scalar transformation `H(x) = x^beta` must be applied consistently to:

- Emitter initialization: direct/emission density and cosine/emission density.
- Camera initialization: light-path sample ratio times inverse camera density.
- Geometry conversions: distance squared / receiving cosine, and inverse cosine.
- Continuous-scatter updates: cosine / forward PDF; reverse PDF; and the
  first-light-vertex conditional emitter-selection correction.
- Delta-scatter updates and the inverse geometry conversion at null surfaces.
- Emitter-hit weights for mesh, area, sphere, and infinite emitters.
- NEE weights for surfaces and the currently implemented volume paths, including
  singular emitters and the finite-radius nonphysical light compatibility case.
- Interior connections in `shade_surface.h` and sensor connections in
  `bidirectional.h`, including their different sample-count conventions.

The transformed continuous recurrence is
`dVC' = H(cos / pdfFwd) * (dVC * H(pdfRev) + dVCM * H(selectionRatio))`,
`dVCM' = H(1 / pdfFwd)`. This identity alone does not validate the integrator:
endpoints and every consumer must use accumulators with the same exponent.
Volume distance-density omissions documented separately remain unresolved.

Before integration, test a shared recurrence against an independent enumerator
that computes complete strategy probabilities for short paths, including camera
and light sample counts, asymmetric forward/reverse PDFs, delta exclusions, and
conditional emitter sampling. Verify partition of unity across strategies and
the beta=1 baseline. Use scaled ratios or equivalent stable arithmetic for
extreme PDFs; zero-density and infinite-limit behavior must be explicit.

Then integrate all consumers together and repeat actual Metal lifecycle,
caustic-switch, emitter, material, and point-light tests. Compare equal camera
SPP and equal emitted light work over multiple seeds against the CPU-guided
reference. A successful point-light trial alone cannot accept this change.

Primary reference reviewed: [PBRT 3e, BDPT MIS](https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing).
Its implementation uses balance weights and explains ratio-based accumulation
over alternative strategies; it also identifies power weighting as an available
variant. No source code was imported. The compact recurrence mapping above is
an audit of this fork, not a claim about PBRT's implementation.


## Shared recurrence checkpoint

`BDPTMISRecurrence<Exponent>` now implements the continuous-scatter update in
`bidirectional_transport.h`. Both camera and light continuation call its
exponent-1 variant in source. No endpoint or geometry term has been converted
and power weighting is not active in the renderer. The installed executable is
still the previously validated caustic-control build; the refactor build passed
but was not installed at this checkpoint.

New host tests independently expand products for each strategy term introduced
at depths 1, 2, 8, and 32, with asymmetric PDFs, selection ratios, and zero reverse
support. They compare both exponents and verify exact exponent-1 agreement with
the previous arithmetic. Actual Metal tests compare 1,024 depth-16 paths with
an independent double-precision expansion. These tests cover the recurrence,
not a complete physical path's endpoint densities or all delta conventions.

Validation: 43 guiding host tests, 11 transport tests, and the actual Metal suite
passed in `/tmp/cycles-guiding-mis-recurrence-metal.log`. Full Blender build
passed in `/tmp/cycles-guiding-mis-recurrence-build.log`. No quality claim follows.
Before activating exponent 2, complete the endpoint/geometry mapping above and
handle extreme ratio ranges: the direct float square is only validated over the
finite test range and can overflow. This must not be silently clamped into an
incorrect strategy weight. Preserve the current balance baseline for rendering
comparisons and test complete MIS partition, not just this algebraic expansion.


## Logarithmic recurrence and range checks

`BDPTMISLogRecurrence<1/2>` now provides log-density factors, ratios formed before
division, log products/sums, continuous scatter, and normalized relative weights.
Zero alternatives use negative infinity with explicit product/sum handling.
No density or accumulated ratio is saturated to a finite cap. This representation
is not yet active in the renderer; the source still calls the direct exponent-1
recurrence and the installed executable remains the caustic-control build.

Host tests exercise growth beyond float range followed by reciprocal factors
that return the final ratio to one. Separate normalization checks form complete
synthetic strategy products using mantissa/exponent pairs rather than logs,
with 32 edges, common and asymmetric 1e20/1e-20 factors, unequal 512/2097152
strategy counts, and excluded connections. They verify both exponents and
partition of unity. These are controlled probability models, not an assertion
that all actual emitter/camera endpoint PDFs have been integrated correctly.

Actual Metal tests execute 1,024 depth-16 paths. Half exercise intermediate
extreme growth and recovery; the other half compare finite asymmetric sequences.
The independent host expansion takes powers only after each complete product.
The original direct-recurrence tolerance remains 1e-5 relative; the new log
recurrence is checked at 1e-4 relative, reflecting its float log arithmetic.
Host normalized weights use a 2e-4 absolute bound in the extreme cases. These
bounds are numeric test contracts, not relaxed render acceptance criteria.

43 guiding host tests, 13 transport tests, and the complete actual-Metal suite
pass in `/tmp/cycles-guiding-mis-log-metal.log`. Only formatting followed the run.
No runtime-quality or full-integration claim follows. Next, apply one consistent
representation at all initialization, geometry, delta, endpoint, and connection
sites; explicitly initialize empty log terms and test the resulting complete
strategy partition before rendering a power candidate.

## Typed logarithmic balance integration

The current source and installed executable now use `BDPTMISWeight` with the
logarithmic exponent-1 recurrence. All path/cache reads decode stored terms;
writes encode them. Camera/emitter initialization, continuous/delta scatter,
geometry and transparent-hit inversion, emitter hits, NEE, interior connections,
and sensor connections use the same representation. Empty terms are explicitly
stored as negative infinity. The two-float cache layout is unchanged, but its MIS
field semantics changed. Power weighting remains inactive.

The wrapper deliberately has no conversion to float. It caught mixed linear/log
expressions during full compilation. Metal's enclosing `MetalKernelContext`
requires member operators and explicit typed left operands; reference-returning
compound assignments additionally require `ccl_private`. Initial failed build
logs are retained. The corrected full build and install passed in
`/tmp/cycles-guiding-mis-log-integration-build2.log` and
`/tmp/cycles-guiding-mis-log-integration-install.log`.

43 guiding and 14 transport host tests, plus the actual Metal suite, passed in
`/tmp/cycles-guiding-mis-log-integration-core2.log`. The added typed test checks
storage round trips, geometry/null inversion, endpoint sum normalization, and
recovery after products exceed ordinary float range. These arithmetic checks do
not prove a complete physical path's MIS partition. Upstream scalar PDF
calculations can still overflow before entering the wrapper; this integration
only protects factors and accumulated terms formed inside it.

Three matched-work Metal balance renders are being checked in
`build/metal-guiding-tests/mis-log-balance`: 512 camera SPP and 2,097,152 emitted
light paths per seed, against the prior linear balance runs and primary CPU
8192-SPP guided reference. Render quality and performance remain unaccepted
until those and broader endpoint/material/lifecycle checks finish. The previous
point-light quality gap and material energy discrepancy remain open.

The three-seed point-light check completed successfully with the required work
counts. Mean linear RGB MSE against the CPU-guided reference is 5.61259764e-6,
versus 5.60777852e-6 for previous linear balance (+0.08594%). The ratio to CPU
512-SPP guided MSE remains 5.85423. This establishes neither a quality improvement
nor statistical equivalence; it is a narrow representation regression check.
Per-seed differences and raw renders are preserved in `mis-log-balance`.
Instrumented render times are 13.54, 13.33, and 13.08 seconds, separate from cold
compilation; these are not a controlled warm performance comparison. Source
formatting followed these renders without semantic changes.

The transparent-sheet caustic-switch control passed all four configurations at
its original rtol/atol 1e-6, with identical 1,517,856-byte caches within the new
build. Report: `mis-log-caustic-transparent/report.json`. Each render used 1 SPP.
A separate old-linear versus new-log image comparison does **not** meet that
same strict per-pixel tolerance: max difference 8.58e-6–9.06e-6, MSE
1.76e-13–1.93e-13, relative mean change 1.10e-7–1.17e-7. Those false checks are
preserved in `previous_balance_comparison.json`; they are not relabeled passes.
The four-switch invariant concerns behavior within one representation, while
this additional comparison measures numerical change across representations.

Both material renders completed at 512 SPP, seed 101. Against the existing
CPU-guided two-seed 8192-SPP reference, rough-glass MSE is .00953795802
(.81818 times CPU512, 1.02712 times prior Metal), and transmission MSE is
.01034015035 (.88210 times CPU512, .96866 times prior Metal). Tall-block mean
offsets are +.7112% and +.5826%. These mixed single-seed changes do not establish
an improvement, unbiasedness, or full acceptance. Report and unchanged raw
controls: `build/metal-guiding-tests/mis-log-material/comparison.json`.

No power candidate has been enabled. Next validation must cover complete
physical strategy/endpoint consistency, upstream density ranges, and broader
lifecycle/transport behavior before a beta-2 quality experiment. The current
point-light failure against CPU and the full-volume limitations remain open.

## Complete geometric strategy enumeration and power candidate

A new host test builds folded geometric paths with 3–10 edges, a finite area
emitter, Lambertian surfaces, and a fixed sensor endpoint. Its independent
oracle converts directional probabilities to area, multiplies the complete
probability of each strategy, incorporates conditional emitter selection and
1:1/4096:1 sensor counts, then normalizes. Separately, it evaluates the compact
emitter-hit, NEE, interior, and sensor formulas using stored log terms and the
shared recurrence. Both exponents agree and sum to one within 2e-5. Trials also
include ideal specular events and exclude connections ending at those vertices.
The specular test uses discrete direction probabilities and matched reflection
normals; it does not cover refractive Jacobians, volume distance densities, or
all real emitter/camera implementation branches.

The wrapper is now `BDPTMISWeightT<Exponent>`, with one renderer alias. Its
storage/endpoint arithmetic tests explicitly cover both exponents. A beta-2
candidate is now building for matched-work rendering. This is an experiment,
not accepted behavior. The prior balance wrapper is saved in
`build/metal-guiding-tests/mis-power-candidate/balance_transport.h`; installed
balance images remain preserved. Point renders will verify their installed
header exponent/hash and exact work counts before computing quality metrics.

The geometric oracle now additionally uses normalized cosine/uniform-hemisphere
mixtures with different incident-conditioned weights in the two directions.
This tests asymmetric guided directional PDFs while keeping projected measure
cosines separate from sampling densities. Both exponents and all complete
strategy checks pass in `/tmp/cycles-guiding-mis-geometric-guided-partition2.log`
(43 guiding + 15 transport host tests). The first extension run retained stale
Lambertian PDF arguments at the two scatter calls and failed; that test wiring
was corrected rather than changing its tolerance or oracle. The renderer was
unaffected by the test-only correction. These path models do not independently
validate cache-reservoir inclusion probabilities or unsupported volume paths.

The beta-2 full build/install and actual Metal/core tests passed in
`/tmp/cycles-guiding-mis-power-{build,install,core}.log`. Point quality validation
is running with installed-header provenance in `mis-power-point/candidate.json`.

The power point pilot completed all three seeds at exactly 512 camera SPP and
2,097,152 emitted light paths each. Mean MSE against CPU-guided 8192-SPP is
1.26134981e-6, compared with 5.60777852e-6 for linear balance: a 77.507% reduction.
It remains 1.31565 times the CPU512 guided mean MSE. Per-seed relative image mean
errors are -.00210%, +.00801%, and -.00684%; these do not prove absence of bias.
Measured instrumented render times are 13.55, 13.48, and 16.31 seconds, not a
controlled warm-speed result. Report: `mis-power-point/comparison.json`.
The point gain alone cannot accept power; material/transport checks follow.

The two power material renders completed at 512 SPP, seed 101. Rough-glass MSE
is .01066686929 (CPU512 ratio .91502; log-balance ratio 1.11836). Transmission
MSE is .01137665357 (CPU512 ratio .97053; log-balance ratio 1.10024). Thus power
worsens these single-seed material errors by about 12% and 10% versus balance,
even though both remain below CPU512 for this seed. Tall-block mean offsets
are +.4809% and +.1482%; high-SPP/multiple-seed convergence is still required.
Raw results and ratios are in `mis-power-material`. This tradeoff prevents an
overall quality acceptance from the point improvement alone.

All four transparent-sheet caustic-switch cases pass the original within-build
image and byte-identical-cache invariants for power in
`mis-power-caustic-transparent/report.json`; each used 1 SPP. No old balance
image equality is expected from a changed heuristic. The installed candidate
remains experimental. No live processes remain at this checkpoint. Next work
must resolve the material tradeoff with multiple seeds and regional convergence,
keep the CPU reference as primary truth, and address the remaining PT quality,
full-volume, endpoint, and lifecycle gaps.
