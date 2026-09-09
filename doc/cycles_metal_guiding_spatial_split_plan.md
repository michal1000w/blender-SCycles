# Guiding spatial refinement candidate

Status: observed-spread refinement restored for final validation.
The original renderer comparison was INVALIDATED by the source-override loader
bug. Only the verified-loader results near the end of this document are current.

The installed Metal loader could select a precompiled library even when
`CYCLES_KERNEL_PATH` named modified source. A strategy-routing diagnostic proved
that the requested source did not execute. Consequently the six spatial candidate
images below cannot establish candidate performance, and the previous rejection
reason is withdrawn. Historical numbers are preserved below only as an audit
trail. The standalone host/Metal component tests compiled their own sources and
remain valid. Re-evaluation requires the corrected loader and proof of execution.
The signed-volume source-override renderer regression likewise did not prove that
its modified kernel executed; it needs a rerun. Normal built/installed benchmark
libraries were regenerated and are unaffected by this specific override issue.

The installed power-MIS baseline remains unchanged while the 4096-SPP material
benchmark runs. Its first rough-glass seed has Metal PT MSE .0020389297 against
CPU guided .0015098151. The prior three-seed 512-SPP comparison also showed a PT
gap. Increasing training duration and changing publication cadence previously
failed to resolve it; those changes are not active.

Source inspection found that spatial refinement uses the longest geometric cell
dimension and its midpoint. That can split perpendicular to a planar set of
observations, leaving their variation unresolved. This is a hypothesis for some
of the quality gap, not evidence that it is the dominant cause.

## Implemented rule

`GuidingField::spatial_split()` aggregates the existing position moments across
directional components and radiance/importance field types. No tracing buffer,
record layout, atomic update, guiding memory budget, or publication schedule was
changed. The rule runs only at drained refinement boundaries.

- Normalize component weights by the largest finite weight before aggregation.
- Exclude invalid statistics and components centered outside the current cell;
  children initially inherit observations from their parent.
- Require effective weighted support of at least 64. Otherwise preserve the
  geometric midpoint fallback. One bright observation must not dictate a split.
- Estimate positional variance, subtract a roundoff allowance, and cap it by
  the variance bound for the current cell interval. Convert normalized variance
  to scene units before choosing the axis.
- Split at the weighted positional mean, constrained to the middle 80 percent
  of the cell. This prevents arbitrarily thin children at bright boundaries.
- Preserve the existing child initialization and full-support parent proposals.
  The split only changes a sampling proposal; it does not rescale, clamp, or
  replace any rendered contribution.

This uses contribution-weighted moments, not an unweighted occupancy histogram.
Inherited data and correlation among observations can affect split quality.
The effective-support threshold is a robustness heuristic, not a statistical
independence certificate. These limitations must be judged with render results.

## Verification and acceptance

The first core run passed 45 host guiding tests, 19 transport tests, and the full
Metal suite (`/tmp/cycles-guiding-spatial-moment-core.log`). New host cases verify
splitting a plane along its observed spread despite a much larger empty cell
dimension, and fallback for a dominant outlier. The Metal field fixture now
records actual normalized positions instead of its former default zero position.
It exercises 12 million records, refinement, inherited proposal normalization,
capacity limits, and radiance/importance queries. A follow-up assertion checks
that the actual Metal root follows the sampled mean rather than the geometric
midpoint; its result is in `/tmp/cycles-guiding-spatial-moment-core2.log` once
that process completes.

Two pilot renders are running via CYCLES_KERNEL_PATH pointing to the worktree:
rough glass and transmission, Metal guided PT, 512 SPP, seed 101, resolution 256,
128 training camera samples, 256 MiB guiding memory. The script verifies actual
SPP, the unchanged publication sequence, and the guiding-field header digest
before/after both renders. Artifacts and the exact commands are in
`build/metal-guiding-tests/spatial-moment-candidate/`.

Compare against the matching installed-baseline Metal PT images and CPU-guided
controls in `mis-power-material-multiseed`, using the existing high-SPP CPU-guided
references. Report linear MSE, regional means, and failure cases. A favorable
single-seed pilot is insufficient for adoption: it needs multiple seeds, point
lighting, BDPT, and higher-SPP checks. A poor pilot should be preserved and the
candidate reconsidered or reverted; do not change reference images or acceptance
tolerances to make it pass. Concurrent verification work makes runtime unsuitable
for warmed performance acceptance.

The exact pre-candidate header is preserved as
`build/metal-guiding-tests/spatial-moment-candidate/baseline_guiding_field.h`.
The signed-volume and roulette work is separate and already passed its production
caustic/transparent regression. None of the spatial work completes the outstanding
full BDPT medium traversal or the overall CPU-guided quality requirement.

## Pilot results and outstanding checks

Both 512-SPP pilot renders completed with the required publication sequence and
unchanged source digest. Against the matching seed-101 Metal baseline, MSE ratios
are .97932563 for rough glass and .98983253 for transmission. Against the matching
CPU-guided controls they are 1.25935247 and 1.36092502. The block region improved
in both scenes, while sphere-region MSE worsened. Full values are in
`spatial-moment-candidate/comparison.json`. This small, mixed single-seed result is
not adoption evidence. Two more seeds (211 and 307), with otherwise identical
settings, are running in `spatial-moment-multiseed`.

The second core run passed the new actual-Metal split assertion: root axis 0,
split -.00413441658, following the sampled mean rather than zero. It then failed
at a later mixture-fit GPU command. The original diagnostic did not print the
Metal command's NSError; no cause has been established. The raw failure is kept
in `/tmp/cycles-guiding-spatial-moment-core2.log`. The harness now prints capacity,
cooperative mode, observation/field counts, run, command status, and NSError on
that failure. Rerun after the render jobs finish; do not silently classify the
failure as contention or report this second run as passed.

The signed-volume production caustic/transparent regression completed
successfully before the spatial changes. Its four image-invariance and exact
cache comparisons passed. The currently installed baseline has still not been
changed. The spatial candidate has not yet had a full Blender build and is not
ready for installation.

## Final decision

The three-seed 512-SPP comparison did not demonstrate a useful improvement.
Mean MSE ratios to the matching Metal PT baseline are .99580512 for rough glass
and 1.00383918 for transmission. Ratios to CPU-guided controls are 1.27185920
and 1.34790566. Individual seeds have substantial variation. The spatial rule
was therefore rejected, and `guiding_field.h` was restored byte-for-byte from
the preserved baseline. Candidate source, host tests, Metal fixture, and harness
are retained under `spatial-moment-candidate/candidate_*` for reproducibility.

All six candidate images actually reside in `spatial-moment-candidate`: a path
replacement in the additional-seed runner did not change its output directory.
The distinct image/command/log names were preserved and their metadata confirms
seeds 101/211/307 and 512 SPP. Its manifest overwrote the pilot manifest with the
additional seeds, so that manifest alone does not describe all six images.
The first multiseed analysis failed because it looked in the intended directory;
the corrected analysis reads the actual artifacts and validates their metadata.
Its result is `spatial-moment-multiseed/comparison.json`; both analysis logs are
preserved. No images were relabeled as another seed or rerendered to repair the
path lookup.

The restored source passed 43 host guiding tests, 19 transport tests, and the
full Metal suite in `/tmp/cycles-guiding-spatial-restored-core.log`, with no
benchmark renders active. This does not establish the cause of the earlier
mixture-command failure, which remains recorded. The improved NSError diagnostic
and the fixture's actual normalized positions are retained; the candidate-specific
split assertions were archived with the candidate and removed from the active
suite. The verified signed-volume/roulette implementation remains independent
of this rejected experiment.

## Verified-loader re-evaluation

Six new renders in `spatial-moment-verified-loader` now pass actual 512-SPP and
all eight publication checks. Three seeds per scene use the original CPU-guided
references and matching CPU/Metal controls. Candidate mean MSE ratios to Metal PT
are .99212347 (rough glass) and .93862348 (transmission). Ratios to CPU guided
are 1.26715694 and 1.26033725. This is promising for transmission but still fails
the CPU quality target; the candidate is not yet adopted. Point lighting, BDPT,
and higher-SPP checks remain necessary.

The first attempted render hit a concurrent disk-full event and failed its
publication-log check despite completing 512 SPP. It is preserved separately in
`disk-pressure-attempt` and excluded. The retry's first image passed, then a free
space guard stopped before the next render. After recovering space, `resume.py`
completed the other five without replacing the valid first image. The analysis
records actual SPP and publication sequences for all six accepted images.

The candidate's six BDPT material and three point-light renders also completed
with equal512SPP and exactly2,097,152 auxiliary light paths each. Material MSE
ratios to the Metal baseline are .89109271 rough glass and .94385806 transmission;
CPU-guided ratios are .817167 and .904999. Point-light mean MSE is1.1456384e-6,
ratio .90826384 to Metal and1.1949595 to CPU. These are quality comparisons, not
warmed runtime acceptance.

Transmission block means are still above reference for all three candidate seeds:
+.95093%, +1.54981%, +1.14742%. Higher-SPP correctness remains essential despite
improved whole-image MSE. Eight equal4096SPP GPU renders (two scenes, two seeds,
PT/BDPT) are running in `spatial-moment-4096`, compared with the existing matching
CPU/Metal4096 controls and original high-SPP CPU-guided references. The production
spatial rule remains the original midpoint until this evaluation is complete.

## Delivery scope update

The user excluded the pre-existing transmission-block offset on 2026-09-09.
The offset is not a guiding acceptance blocker. Observed-spread refinement is
restored in the delivery source with its two host regressions and actual-Metal
root-split assertion; the unfinished mixture-capacity change is removed.
See `cycles_metal_guiding_validation.md` for final checks and quality limitations.
