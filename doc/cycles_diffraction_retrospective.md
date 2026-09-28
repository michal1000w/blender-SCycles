# Retrospective review of correctness verdicts

## Review basis

The user clarified that physical correctness, not equality with finite-sample
PT, is the target. This review checked the image-comparison verdicts in
`cycles_diffraction.md`, the scene inventory, and the named reports below.
It does not certify every historical experiment or establish a new ground truth.
Raw images and numerical reports are retained unchanged.

Three different questions must remain separate:

1. Does an estimator sample the transport defined by the configured cutoffs?
2. Have its finite-sample results converged sufficiently to measure that transport?
3. How closely does that truncated transport approximate the physical scene?

A path excluded by a bounce setting can carry physically valid energy. A
renderer admitting it can therefore be brighter and closer to full transport
while still interpreting the setting inconsistently. Conversely, greater
brightness alone does not establish more complete or correct transport.

## Findings

| Experiment / report under `tests/output/diffraction` | Correct interpretation |
| --- | --- |
| `physical_indirect_sphere_transport/verified_comparison.json` | Large PT/BDPT disagreement is observed. The narrative's “failing transport regression” was too strong as an absolute physical verdict and has been corrected. No independent full-scene reference establishes which image is closer to physical transport. |
| `physical_indirect_50mm_clear_transport/verified_comparison.json` | The 11–13% brightness difference is a measured estimator discrepancy. A separate code audit shows missing combined per-type cutoff accounting. That proves a settings-support gap, not that all additional light was physically wrong. |
| `mirror_50mm_comparison.json` and `mirror_50mm_diffuse12_comparison.json` | Increasing the diffuse limit changes the target integral and brings the estimators closer. The higher-limit result must not be rejected merely because it is brighter than the lower-limit PT result. Neither result is an absolute infinite-bounce reference. |
| `mirror_mis_production_comparison.json` and `physical_indirect_mis_candidate/verified_comparison.json` | Finite-seed comparisons after the emission-MIS change do not establish residual physical bias. The justification for the MIS change is the missing alternative-strategy terms, not tuning output brightness toward PT. |
| `mixed_convergence/verified_comparison.json` | The low-sample discrepancy did not persist at 4096 samples against the direct spectral-solver reference. The final record already acknowledges this; no erroneous failure label needs preserving. The same-truncation solver is not an exact Maxwell reference. |
| `mnee_pt_energy_comparison.json` | The initial covered-disc difference remains inconclusive as an absolute physical verdict, even with 65,536-sample PT and four seeds. Rare-path variance and lack of an independent reference remain relevant. |
| `mnee_tight_failed.json` and `mnee_chain_energy_comparison.json` | The tighter-walk experiment's historical filename is retained. Its lower brightness alone is not the reason to reject it: rejected manifold walks with corresponding camera paths still culled leave missing strategy support. Restoring a valid partition is the relevant correction; subsequent PT agreement is supporting evidence only. |
| `prefix_translucent_zero_four_seed_comparison.json` and `prefix_translucent_zero_2048_comparison.json` | The +0.38% residual at 512 samples was not a demonstrated physical error. At 2048 samples the difference is +0.00001235 with SE 0.00003142 (about +0.04% of PT); the earlier discrepancy did not persist. This supports a sampling explanation without establishing absolute correctness. The separately derived reversed-prefix rule concerns consistent configured support. |
| Dielectric full-cache budget failures | These are actual failures to produce a usable cache, independent of PT/BDPT brightness. Their classification is unchanged. |
| Lossless passivity, interface matching and modal-order checks | These use conservation, boundary conditions or solver convergence rather than PT image equality. They remain relevant independent checks, within each recorded numerical scope. |

## Corrections and limits

A subsequent independent enclosure test directly illustrates the user's concern.
With uniform E=0.25 and rho=0.5, infinite-bounce radiance is exactly 0.5. With
diffuse limit four and total limit twelve, the configured five-scatter target
is 0.4921875. The production kernel's seed-11 / 1024-sample means are PT
0.4921905913 and BDPT 0.4959826252, with verified artifacts and provenance.
BDPT is brighter **and closer to the full physical answer** in this example,
although it differs from the configured truncation target. This is a new
reference experiment, not retroactive proof for a different historical scene.
The cutoff correction must be described as consistent setting semantics,
not as a universal improvement in physical accuracy by reducing brightness.

The subsequent higher-depth check uses four independent seeds, 2048 samples,
and both total and diffuse limits twelve. Its analytic truncated target is
0.499969482421875, approaching the infinite-depth value 0.5. The prefix
candidate gives PT 0.4999670123 (standard error 0.0000022350) and BDPT
0.4998447392 (standard error 0.0000091195). The BDPT shortfall of
0.0001247432 is about 0.025% of the analytic target and exceeds the observed
between-seed uncertainty. It remains unresolved; agreement with PT is neither
the reference nor the acceptance criterion. This candidate is not certified
by the earlier agreement tests. See
`tests/output/diffraction/enclosure_prefix_diffuse12_comparison.json` for the
artifact-verified results and source provenance. These runs are correctness
experiments, not isolated performance benchmarks.

The scene inventory and original narrative now distinguish disagreement from
physical error. Both comparison analyzers explicitly identify PT as a diagnostic
estimator, not physical ground truth. No brightness scaling, clipping, tolerance
relaxation or removal of raw results is part of this review.

This review found overconfident labels, but has not independently proved that
an earlier rejected image was physically correct. It would be equally unsupported
to relabel those images as correct solely because BDPT was brighter. Absolute
reference scenes and convergence studies are required to resolve them.

### Aluminum exact-cutoff audit distinction

The eight exact-grazing failures in the modal audit belong to
`diffraction_grating_solve`, which uses a singular physical admittance basis at
that threshold. They do not establish failure of the reference-port solver
used by the cache. The latter succeeds exactly at 400 nm for the aluminum CD
relief at N32/64/128 and agrees with converging one-sided limits. Retain the
original failed cases, but do not classify the complete rendering boundary
path as unsupported from that narrower API result. Evidence:
`tests/output/diffraction/aluminum_cutoff_reference_analysis.json`.
