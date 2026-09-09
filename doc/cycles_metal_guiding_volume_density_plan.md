# Full BDPT medium continuation: density contract

Status: implementation requirement, not completed support. This audit accompanies
the equal-SPP guiding experiments; it does not change the active renderer.

## Current implementation evidence

`photon_volume_sample_segment()` in `kernel/integrator/shade_volume.h` reuses
Cycles homogeneous or heterogeneous integration. Its outputs are the weighted
throughput, selected position, receiver object, and event classification. It
does not return the sampling density of the selected event or the reverse
strategy density. `VolumeIntegrateResult` likewise exposes weighted throughput,
not a bidirectional density contract.

`integrator_bdpt_light_generate()` in `kernel/integrator/bidirectional.h` stores the
first medium collision and returns. Its comment explicitly identifies the
missing repeated free-flight strategy densities. This is still a full-feature
gap even though camera paths continue multiple scattering. Removing that return
without extending density accounting would not establish correct BDPT support.

The light-side conversion currently applies squared distance at a medium arrival,
without the surface projected-area cosine. That geometric conversion alone does
not describe how a medium event was selected. The volume NEE weight also reuses
the compact recursive state. Both paths must agree on the density measure when
continuation is extended.

## Required implementation contract

Represent the complete sampling strategy for each traversed segment, including
the no-scatter endpoint probability, collision selection, null events, channel
selection, and any termination/reservoir decisions. Preserve information needed
to evaluate the reverse strategy on the same realized path. Do not reconstruct a
PDF by dividing physical attenuation by a throughput weight: spectral channel
mixtures and randomized selection make that insufficient in general.

The segment contract must cover both homogeneous and heterogeneous integration.
It must survive pure-volume boundaries, mixed surface boundaries, texture-cache
retry, and reservoir selection without changing which event the PDF describes.
After a real collision, evaluate/sample the phase mixture with the appropriate
guiding field, retain forward/reverse phase PDFs, update the recursive BDPT
weights, update path/RNG state, and continue within the configured bounce limits.
Connection shadows need the matching competing strategy probabilities along
their segment; a transmittance estimate alone is not a density certificate.

Use a null-event path-space formulation where required by the heterogeneous
sampler. Keep ratios scaled or in log form to avoid products underflowing along
long paths. Any compact sufficient ratio representation needs a derivation
showing that its factors match explicit path strategy enumeration.

## Validation required before claiming support

- An explicit small-path strategy enumerator, independent of the compact
  recursion, for surface/medium combinations in both directions.
- Homogeneous analytic free-flight and survival cases, including absorption,
  non-gray coefficients, phase anisotropy, and multiple real collisions.
- Heterogeneous null-event realizations with forward/reverse checks and majorant
  segmentation changes; retain the sampled realization when comparing PDFs.
- Actual Metal renders containing two or more light-side medium collisions,
  with counters proving that the intended paths were exercised.
- Equal-SPP PT/BDPT comparisons against high-SPP CPU-guided references over
  multiple seeds, including volume caustics, nested boundaries, and textured
  media. Report light-path work separately; test mean energy and noise.

## Primary references checked

The null-scattering path-integral formulation provides a basis for MIS over
heterogeneous medium paths. PBRT's generalized path-space discussion likewise
motivates retaining explicit path sampling information for bidirectional
integration. These sources guide the contract above; no external implementation
was imported and they do not validate the current Cycles fork.

- Miller, Georgiev, Jarosz (2019), [A null-scattering path integral formulation of
  light transport](https://cs.dartmouth.edu/~wjarosz/publications/miller19null.html).
- [PBRT 4e, The Equation of Transfer, generalized path space](https://www.pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer).

## Throughput-independent event model and connection correction

New model checks in `tests/python/cycles_bdpt_volume_density_model.py` make a
candidate contract concrete. They are analytic/model tests, not renderer support.
For direction-independent coefficients and a conservative shared majorant M,
choose real-event intensity rho = mean(sigma_s), q_real = rho/M, q_null = 1-q_real.
Real-event spectral weight is sigma_s/rho; null-event spectral weight is
(1-sigma_t/M)/q_null. Zero-support cases are explicit. Event selection no longer
depends on incoming throughput, unlike `volume_scatter_probability()` today.

Explicit sums over Poisson null counts agree with spectral exp(-sigma_t*d)
for surviving connection paths and sigma_s*exp(-sigma_t*d) for collision density.
Tests include zero extinction channels, pure absorption, pure scattering,
different majorants, and piecewise null histories reversed through the same
world-space points. All five current model checks pass in
`/tmp/cycles-guiding-volume-density-model2.log`.

There are two distinct connection choices; they cannot be silently interchanged:

1. Use the same null/real roulette on connections, terminating with zero on a
   real event. The sampling density of the shared null realization then matches
   the random-walk segments. This is a correct model but may sacrifice too much
   shadow variance/performance; it has not been selected as the production path.
2. Use deterministic-null ratio tracking on connections. If Q_e is the
   product of random-walk q_null choices on edge e, the probability of the
   strategy connecting edge e has an extra factor 1/Q_e after cancelling common
   path terms. The emitter-hit strategy has no removed edge factor. The model
   explicitly enumerates full probabilities and verifies this correction for
   beta 1 and 2; ignoring it changes normalized weights by more than .1 in its
   counterexample. Thus existing transmittance-only shadows are insufficient.

This is a proposed connection estimator, not a description of the current
shadow implementation; the traversal audit below establishes that distinction.

For ordinary internal medium vertices, direction-independent rho at each fixed
world-space point occurs in every complete strategy and can factor out, along
with common Poisson factors. The model checks that cancellation independently.
This does not cover volume-emitter endpoint sampling, direction-dependent
coefficients, differing proposal epochs, or throughput-dependent roulette.
Storing one forward marginal density from the existing sampler would not prove
these cancellations or provide the corresponding reverse law.

The ratio-tracking option needs a segment log(Q_e) accumulated during both walk
and connection sampling. The connection's selected-strategy MIS cannot be
finalized until that shadow segment's null probabilities are known. Consequently
implementation must extend shadow MIS state and contribution/training finalization,
as well as camera/light continuation and alternative-strategy insertion into the
compact recurrence. It must preserve coherent majorant histories in both
orientations and handle zero-support contributions explicitly. The current
volume reservoir and incoming-throughput-dependent sampling cannot be reused
unchanged. These are implementation obligations, not a claim they are solved.

The null-path PDF distinction is consistent with the primary formulation already
cited above and PBRT's [Volume Scattering Integrators](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Volume_Scattering_Integrators),
which treats the ratio-tracked final shadow segment separately. The particular
rho proposal, cancellation derivation, and model tests here are our candidate
architecture, not code imported from those sources. Conservative majorants and
performance remain explicit validation requirements before renderer integration.

## Shared MIS and event components

`BDPTMISWeightT` now scales by an unpowered log density and normalizes a selected
ratio-tracked connection using its log null-choice probability. The complete
geometric path test was extended with explicit null-choice products, mixed
surface/isotropic-medium vertices, direction-independent real-event intensities,
conditional emitter selection, unequal sensor counts, and specular exclusions.
It constructs full strategy probabilities independently and compares both
exponents. Internal real-event intensities factor out consistently across the
strategies in this model.

The tested compact insertion is: multiply dVCM by 1/Q_e at arrival along a sampled
walk edge, leaving existing dVC alternatives unchanged at that conversion. The
usual scatter recurrence propagates this newly introduced connection strategy.
When the selected strategy itself uses a ratio-tracked connection, multiply the
sum of competing relative terms by Q_selected before adding the selected unit
term. Emitter-hit normalization uses Q_selected=1. Products are represented as
logs, so neither Q nor its inverse needs to be materialized as an ordinary float.
The geometric test passes at its existing2e-5 normalized-weight tolerance. An
actual Metal check covers1,024 log-weight cases, including cancellation of
10,000-sized log factors, with an absolute5e-4 bound for those extreme float logs.
These are numerical contracts, not relaxed image acceptance tolerances.

`bidirectional_volume.h` adds the complete throughput-independent event
probability/weight component for the candidate contract. It rejects invalid
coefficient/majorant bounds and representational loss of event support. Exact
vacuum and pure-scattering endpoints have explicit distributions. Absorption
is represented through spectral weights. This component is registered with the
kernel sources and included in the BDPT compilation context, but it is not yet
used by volume traversal. No continuation or full-volume support is claimed.

Host tests enumerate Poisson null counts and compare spectral transmittance and
collision densities to closed forms. The Metal transport test samples196,608
homogeneous paths over three majorants, checking survival and integrated real
scattering per channel, plus explicit iteration-guard failures. Its stochastic
bound is six measured standard errors plus2e-4; host analytic bounds are5e-7.
The initial Metal test caught a negative null weight at a majorant boundary.
Subtracting sigma_t from the majorant before dividing, disabling reassociation
for the event construction, and validating nonnegative weights resolves that
arithmetic issue; no tolerance was increased. The initial host fixture also used
the wrong spectrum-construction macro, and a later Metal fixture needed an
explicit `metal::any` qualification. Those failures remain in their logs.

Full Blender compilation of the components passed in
`/tmp/cycles-guiding-null-components-build2.log`. Final host/Metal endpoint checks
are recorded separately before installation. Next integration must replace the
throughput-dependent event law on both light and camera paths, accumulate actual
segment log(Q), and defer connection MIS until ratio-tracked shadow traversal
finishes. Emission, heterogeneous majorants, boundaries, retry, phase guiding,
reservoir bookkeeping, and repeated real collisions remain required work.

Final component validation passed in `/tmp/cycles-guiding-null-event-metal4.log`:
43 guiding and18 transport host tests, the full Metal suite,1,024 log-null-MIS
cases, and196,608 homogeneous transport paths. Exact vacuum/pure-scattering
support and guards passed. Across18 spectral survival/scattering means, the
largest absolute deviation was2.04 measured standard errors. Numeric results
are preserved in `build/metal-guiding-tests/null-event-metal4/report.json`.
Installation passed in `/tmp/cycles-guiding-null-components-install.log`, and the
installed event header matches source byte-for-byte. None of this establishes
heterogeneous traversal, repeated light-side volume continuation, or image-quality
acceptance; those integrations remain the next required work.

## Actual traversal audit: connection estimator and bounds

The next source audit found two integration assumptions that require correction
before introducing the new sampler into the renderer:

1. `volume_shadow_null_scattering()` in `shade_volume.h` is not a
   deterministic-null ratio tracker. Its homogeneous branch evaluates analytic
   transmittance. Its heterogeneous branch calls `volume_transmittance()`, which
   uses a randomly selected telescoping correction over regularly spaced shader
   evaluations. `volume_octree_advance_shadow()` also merges intervals according
   to optical thickness. There is no realized Poisson null-point history from
   which the proposed Q can be recovered. The separate ray-marching option has
   no such history either. Multiplying these estimators by a fabricated Q would
   not implement the tested null-path MIS partition.
2. `volume_estimate_extrema()` samples four positions for a heterogeneous shader,
   multiplies the sampled maximum by 1.5, and floors it at .5. This cannot certify
   a conservative bound for arbitrary procedural features. The function receives
   visibility, path flags, and RNG state; `volume_object_get_extrema()` selects
   baked or dynamic estimates based on camera visibility and Light Path nodes.
   Reversing a ray need not give the same proposal. The cancellation proved for
   shared world-space majorants cannot simply be assumed here.

The model suite now contains seven tests. The two additions demonstrate a narrow
procedural feature missed by all four probes, and a full forward/reverse Poisson
density ratio with different majorants. In the latter, the missing exponential
and majorant-product ratio changes MIS weights for both beta 1 and 2. The tests
passed in `/tmp/cycles-guiding-volume-traversal-contract.log`. These are analytic
counterexamples, not shader execution or renderer integration tests.

The current transmittance routine cites
[Misso et al. 2022](https://cs.dartmouth.edu/~wjarosz/publications/misso22unbiased.html),
a debiasing framework distinct from null-path sampling. The later
[progressive null-tracking work](https://cs.dartmouth.edu/~wjarosz/publications/misso23progressive.html)
explicitly discusses underestimating bounds, high variance in weighted tracking,
and a progressive density-clamping approach with finite-sample bias. That last
approach is not an accepted implementation choice for this task: it cannot be
silently substituted while claiming unchanged transport at the tested SPP.

### Concrete integration order

- Establish a proposal field shared by camera, light, and connection walks, with
  a defined response to underestimated bounds. If proposals differ, retain and
  evaluate the full forward/reverse Poisson terms instead of only Q. The
  existing conservative event helper must not silently discard invalid events
  or clamp the physical medium to make its precondition pass.
- Implement one explicit null traversal for the participating BDPT strategies.
  Homogeneous analytic connections require a separately derived marginal-density
  partition or the same explicit null realization as the other strategies.
  Ordinary PT's transmittance implementation need not be replaced to achieve
  that. Direction-dependent coefficients and volume emission also need their
  own complete strategy evaluation; the internal-rho cancellation is conditional.
- Only then add deferred shadow MIS. Sensor connections currently multiply the
  weight into their contribution in `bidirectional.h`; interior surface
  connections do so in `integrate_surface_bidirectional()` in `shade_surface.h`.
  NEE has separate surface and volume weight construction. All participating
  strategies must use the same measure before multiple light collisions are
  enabled. A change confined to sensor splats would leave the partition wrong.
- Initialize any added shadow density state in `integrator_shadow_path_init()`
  in `state_flow.h`, with neutral values for ordinary NEE, AO, and other shadow
  users. Scope allocation to `KERNEL_FEATURE_BDPT` where appropriate.
- Treat each shadow volume segment as a transaction. Currently
  `integrate_transparent_shadow()` keeps throughput local and commits it only
  after `integrate_transparent_volume_shadow()` returns CONTINUE. New density
  accumulators must follow the same commit boundary. A texture-cache retry must
  replay the segment from its prior RNG/density state without accumulating Q
  twice. A surface cache miss after a completed volume segment skips that
  segment on retry. More-hit traversal batches must retain already committed
  terms. Do not reset them when the shadow ray's tmin advances.
- Finalize MIS after the last visibility segment and before both
  `guiding_record_direct_light()` and `film_write_direct_light()` in
  `integrator_shade_shadow()`. Preserve overflow protection: removing an early
  small MIS factor can overflow a previously finite contribution before the
  final multiply. A scaled spectral representation or equivalent safe algebra
  needs explicit numerical tests; an unweighted float throughput is insufficient
  as a general refactor.

No shadow MIS refactor or traversal change was made during this audit. In
particular, there is no new state carrying a dummy Q, and the first light-side
medium-collision return remains. The source/build/render baseline and all prior
equal-SPP quality results remain unchanged.

## Signed event extension and light-path roulette

The event component now supports an underestimated positive proposal rate M
without changing the physical coefficients. The bounded case preserves the
previous q_real = mean(sigma_s)/M law. When max(sigma_t)>M, it selects the real
event with probability

    q_real = sum(sigma_s) / (sum(sigma_s) + sum(abs(M - sigma_t))).

The null probability is 1-q_real; real and null weights remain respectively
sigma_s/(M*q_real) and (M-sigma_t)/(M*q_null). The proposal is positive while
transport may be signed. Scores are scaled before averaging to avoid overflowing
their sum. No coefficient is clipped and the rate is not changed after sampling
a candidate. Invalid physical inputs or loss of representable event support
still return an explicit invalid distribution. The exact pure-scattering branch
also checks sigma_t==sigma_s: sigma_s==M alone is insufficient with an
underestimated bound.

This follows the weighted-event construction discussed by
[Kutz et al. 2017](https://disneyanimation.com/publications/spectral-and-decomposition-tracking-for-rendering-heterogeneous-volumes/),
also cited by the existing Cycles volume integrator. The magnitude-based
throughput-independent proposal here is our component implementation; it does
not reuse the existing integrator's incoming-throughput-dependent roulette.
Internal real-event intensity is now M*q_real, which need not equal
mean(sigma_s) in the underestimated case. Its cancellation still requires the
same world-space proposal and coefficients under both orientations.

An associated renderer change makes BDPT light-path roulette use the maximum
absolute spectral throughput (including the existing unguided factor). Previously
an all-negative path could have zero continuation probability despite nonzero
transport. The camera roulette already uses absolute magnitude. Positive light
paths retain their prior probability and the .95 cap. The per-path vertex
reservoir uses uniform candidate inclusion, so it does not discard candidates
based on their sign. The rest of the signed-contribution path still requires
audit during traversal integration.

Validation in `/tmp/cycles-guiding-signed-volume-metal3.log` passed:

- 43 host guiding tests and 19 host transport tests. Poisson sums using the
  shared event component match analytic survival and collision density at the
  existing 5e-7 tolerance, now including underestimated rates .6 and 1.0. The
  roulette check covers all eight sign patterns and the zero/high-magnitude cases.
- 327,680 actual Metal paths across five rates, checking spectral survival and
  first real scattering. Underestimated cases explicitly exercise negative
  weights; conservative cases still reject them.
- 524,288 actual Metal paths through three piecewise-constant slabs, continuing
  after real collisions, in both slab orders and with bounded/underestimated
  rates. These use a forward delta phase as an analytic transport model, not
  Cycles phase sampling. The independent references for zero, one, and at least
  two collisions are exp(-tau_t), exp(-tau_t)*tau_s, and
  exp(tau_s-tau_t)-exp(-tau_t)*(1+tau_s). Final roulette uses the same new helper
  as the actual light-path kernel. Every mode exercised over 51,000 paths with
  multiple real collisions; signed modes produced over 71,000 negative spectral
  weights after roulette. All 36 means passed the existing six measured standard
  errors plus 2e-4 Monte Carlo bound.

The largest deviation across all 66 means was 4.67 measured standard errors;
raw values and counts are retained in
`build/metal-guiding-tests/signed-volume-events/report.json`. The record also
includes analytic first-scatter/survival variances. For rate .6, blue-channel
survival variance is about 376 times the rate-1.4 case. This is evidence of the
cost of a poor proposal, not a performance improvement. Accurate shared proposal
fields remain necessary. Signed tracking removes the need to clamp physical
extinction; it does not solve reversal, ray-flag-dependent shaders, unbounded
media, emission, shadow density state, or full BDPT continuation.

The full Blender build passed in `/tmp/cycles-guiding-signed-volume-build.log`.
Installation is deliberately pending while the earlier installed baseline runs
the 4096-SPP material comparison. A separate production Metal regression uses
the supported CYCLES_KERNEL_PATH override to compile the current worktree kernel
sources without replacing that baseline. Its completion must be checked in
`/tmp/cycles-guiding-signed-volume-caustic-transparent.log` and the corresponding
report; the component results alone are not a production renderer pass.
