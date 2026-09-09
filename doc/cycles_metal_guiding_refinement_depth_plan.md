# Spatial refinement depth experiment

Final status: four rounds accepted by the user on 2026-09-09. The research notes
below are historical. Prepared training-duration and RIS-prior follow-ups were
closed without execution when the user requested completion.

The committed delivery checkpoint is `ba5a 19cf 82d`. Its default Metal training
limit freezes after eight publications, each allowing one spatial split along
any lineage. The nominal depth limit of 20 is therefore unreachable under that
schedule. Observed-spread refinement improved material quality, suggesting a
bounded test of additional spatial resolution before changing mixture capacity.
This is a hypothesis, not an established cause of the remaining CPU quality gap.

The candidate runs two ordered refinement dispatches after each single model
publication. A begin-update dispatch between them snapshots the newly allocated
node count. The queues are drained before publication and execute these kernels
in order. Children retain full-support parent proposals; no second fitting update,
histogram decay, camera sample or light path is added. Capacity and depth limits
remain unchanged. Refinement on inherited moments can be less informative than
new child observations, so additional depth is not assumed to improve quality.

Compare 512-SPP rough-glass and transmission PT over seeds 101/211/307 against
the committed observed-spread results and the same CPU-guided controls/reference.
Check actual SPP and all eight publications. Reject regressions unchanged. A
promising result needs BDPT, point lighting, higher-SPP and runtime checks before
adoption. The existing host/Metal field tests exercise successive refinement,
capacity limits and inherited proposal normalization; final integration still
requires actual renderer evidence.

## Two-round results

All six 512-SPP renders passed their sample and publication checks. Mean MSE
ratios to the committed one-round baseline are 0.8416924 (rough glass) and
0.8862411 (transmission); ratios to CPU guided are 1.0665564 and 1.1169627.
Every tested seed improved. Log1p MSE ratios are 0.8073049 and 0.8605742.
This supports testing spatial resolution further, but does not establish CPU
parity or complete acceptance. The two-round source and all images are preserved
in `build/metal-guiding-tests/refinement-depth-candidate`.

A three-round candidate now uses the same sample and publication schedule,
compared against both the two-round candidate and unchanged CPU controls. Its
results must be checked before a default is selected. Neither candidate has yet
passed the required higher-SPP, BDPT, point-light or warmed-runtime comparisons.

## Three-round results

All six exact 512 SPP/publication checks passed. Every seed improved over two rounds.
Mean MSE ratios to two rounds are 0.9197835 and 0.9443889; CPU-guided ratios are
0.9810009 (rough glass) and 1.0548471 (transmission). The remaining gap motivates
one more bounded four-round comparison with the same protocol. All candidates
remain experimental until cross-scene, higher-SPP and runtime checks complete.

## Four-round results and held-out validation

All six exact 512 SPP/publication checks passed. Relative to three rounds, mean
MSE ratios are 1.0056559 (rough glass) and 0.9718095 (transmission). CPU ratios
are 0.9865494 and 1.0251104. This is a mixed marginal change, not another across-
seed improvement. Four rounds is selected provisionally for held-out seeds 401/503
because it reduces the worst measured CPU ratio while preserving approximately
CPU-level rough-glass quality. Tuning on seeds 101/211/307 stops here.

The existing material benchmark runner will generate fresh matching CPU-guided,
Metal PT and Metal BDPT renders at 512 SPP for both held-out seeds. CPU-guided
high-SPP references remain unchanged. No runtime gain is inferred from these
quality-only results. Higher-SPP and point-light checks remain required.

Spatial-node logs support the capacity hypothesis: the original schedule used
511 of 803 nodes after 128 training samples on the rough-glass seed 101 fixture;
two and three rounds each used 803. Their first-sample node counts were7 and 15,
versus3 for the original schedule. Total allocated capacity did not increase.

The held-out 12-render matrix passed all sample/BDPT work checks. CPU MSE ratios
are 0.9814840 rough-glass PT, 1.0731206 transmission PT, 0.7595426 rough-glass
BDPT, and 0.7994382 transmission BDPT. This confirms a persistent transmission
PT gap, despite substantial improvement over the original schedule. The candidate
is not complete or adopted. Eight 4096 SPP GPU checks now compare seeds 101/211
against the committed observed-spread outputs and existing matching CPU controls.

## Separate RIS target observation

During the higher-SPP run, source inspection found another difference worth
isolating later. CPU `calculate_ris_target()` in `integrator/guiding.h` mixes
incident radiance with a constant1/(2pi) BSDF target term. The Metal surface RIS
code uses1/(4pi) when the shader has transmission and1/(2pi) otherwise. A RIS
target need not integrate to one, so this is not evidence of bias or a correctness
bug. It does change relative radiance/prior strength on transmissive surfaces.
Matching the CPU term is a possible separate variance experiment; it must not be
combined silently with the refinement comparison. No target change is active.

Another prepared follow-up revisits training duration on the now fully allocated
four-round field. CPU limit 128 previously corresponded to 128 updates through
camera sample 508; GPU limit 128 freezes at camera sample 128. The earlier rejected
512-camera-sample GPU duration test used the old spatial schedule, so it does not
establish the outcome for the four-round model. A 512-SPP/512-training-limit probe
would preserve final camera work while testing six further model publications.
It remains separate from the RIS prior candidate and is prepared only. No default
or acceptance threshold changes on the strength of this hypothesis.

The completed rough-glass 4096 SPP pair has PT/CPU MSE ratio1.1651910 and BDPT/CPU
ratio0.9465587. PT improves on the committed one-round MSE (0.00172363 versus
0.00191577), but low-sample CPU parity does not persist. Error localization places
about 98% of the excess whole-image PT MSE outside the two object rectangles.
This prioritizes a global training-duration check over treating the transmission
RIS prior as the only explanation. Pixel location is not path isolation: paths
outside those rectangles can still interact with the glass objects. No causal
claim or acceptance change follows from this localization alone.

Final high-SPP, point-light, and held-out results are recorded in
`cycles_metal_guiding_validation.md` and `cycles_metal_guiding_results.json`.
