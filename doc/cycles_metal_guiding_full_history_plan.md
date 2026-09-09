# Full-history Metal guiding training

## Motivation and acceptance

The current GPU trainer retains one independently selected camera-path anchor.
This is a valid importance-sampling training policy, but it discards observations
that CPU OpenPGL collects from other scattering vertices. Equal-512-SPP tests
still favor CPU guiding on rough glass and transmission. Two controlled attempts
to improve the existing representation (more direct-light observations and
histogram-based broad-surface RIS) did not materially improve those cases.

The implementation now collects returned radiance at every eligible camera-path
scattering vertex. The host build and standalone actual-Metal record lifetime tests
pass; render integration and equal-SPP quality validation are in progress. More
training data is not by itself evidence that the quality gap has closed.
Acceptance requires measured equal-SPP improvements against the frozen transport
baseline and CPU-guided references, including the existing detail regions.
Runtime is measured separately, including training and field updates.

## Immutable records

Allocate a GPU buffer of compact records containing:

* Directional field record index (32 bits).
* Packed outgoing direction (32 bits).
* Inverse scattering weight (three floats).
* Parent history index (32 bits).

The intended record size is 24 bytes, verified on host and Metal. A main path
holds a head index; a shadow path captures that index when it branches. Records
remain immutable until all paths in their training group finish. No pointers to
CPU objects or CPU training backend are involved.

After a nonsingular, non-null scatter, append a record using the same radiance
measure as the existing anchor: inverse of post-scatter throughput times the
actual contribution PDF. On film emission, background, or completed shadow
contributions, traverse the captured history and train each ancestor with its
own inverse weight. Immediate NEE records train their current vertex once;
ancestor records train the returned indirect component. All eligible vertices
participate, so direct records no longer use random anchor selection.
Audit explicit BDPT vertex connections too: their completed shadow contribution
can provide an observation at the current camera vertex after removing that
vertex's scattering throughput. Keep its indirect-light semantics distinct from
the user's option to include direct NEE illumination in training.

Preserve pre-clamp training, finite-value checks, spectral conversion, the exact
sampling/evaluation PDFs, and the separation between frozen sampling data and
accumulation. Training history must not alter rendered throughput or MIS weights.

## Bounded lifetime and scheduling

Use an explicit training-record memory budget. Derive a conservative maximum
number of primary paths per group from record capacity and the maximum number
of eligible scattering events, accounting for main-path splits. Do not reserve
history for an entire image/sample batch, which would grow with resolution.

While a training group has live camera or shadow paths, do not replenish it
with new primary paths. Once the authoritative GPU queue counters show that all
of its paths have finished, reset its append counter and schedule the next group.
Accumulate field training across groups and publish only at the existing drained
batch boundary. This avoids a GPU reference-counting/free-list scheme and its
ABA hazards, while allowing shadows to outlive their parent main path safely.

The append-buffer capacity must be sufficient by construction. Overflow is an
error to diagnose and fix, not permission to silently drop training records.
Validate arithmetic at the maximum bounce settings and minimum memory budget.
BDPT light generation trains importance directly and does not need camera history;
sensor shadows must start with an empty history. Account explicitly for those
initial shadows when determining whether a group is drained.

After the training limit, disable history allocation/traversal work and preserve
ordinary wavefront replenishment. Cancellation must release or reset history
before another render can reuse it. Scene changes, persistent sessions, device
changes, and texture retries must preserve the same ownership rules.

## Integration and verification

1. Add shared record layout, host/device pointers, allocator capacity, and bounded
   training-group scheduling. Keep all device pointers in the Metal ABI pointer
   prefix and verify alignment and size.
2. Replace single-anchor state with history-head capture, append, and traversal.
   Cover emission, background, surface/volume NEE, indirect shadow contributions,
   and path splits; retries must not append or train committed work twice.
3. Add necessary memory controls and report actual allocation/group sizes in
   diagnostics. Remove obsolete anchor arrays once the new path is validated.
4. Verify record lifetime with deliberately small budgets and many groups, long
   paths, transparent sheets, volumes, cancellation/restart, and texture retries.
   Check empty histories for BDPT light/sensor work.
5. Validate actual Metal execution, sampling normalization, and independent-seed
   convergence against CPU-guided references. Repeat the current equal-SPP scene
   matrix, retaining any regressions. Measure warm runtime only after compilation
   and specialization have finished and without competing jobs.

Do not mark the overall goal complete until this and the remaining feature,
correctness, quality, and performance requirements have been verified.

## Implementation checkpoint

The 24-byte record layout is shared by host and Metal. Main state now holds only
one history index (4 bytes); shadow state holds that index plus a 20-byte immediate
connection record. The old single-anchor arrays and geometric depth selector are
removed. Surface and volume scatters append records; emission/background/shadow
hooks traverse immutable ancestors. BDPT vertex connections additionally train
the current camera vertex as indirect illumination. Light/sensor paths initialize
with empty camera histories.

A separate 16–1024 MiB training memory control defaults to 128 MiB. The host caps
training groups by buffer capacity divided by the conservative maximum record
count per path and reserves space for shadow-catcher splits. It waits for every
queue to drain before reading the append count, rejecting overflow, and reusing
the buffer. The buffer is freed after finite training and reallocated on restart.
The existing field memory budget and frozen-field publication schedule remain.

Validation so far: Release build; 18 host sampling tests; 5 reciprocal transport
tests; actual Metal sampling/field tests; new Metal history test with 100,000 paths,
eight records per path, separately captured shadow prefixes, four pool reuse
cycles, and deliberate overflow with guard bytes. All pass. Persistent render
checks and equal-SPP material tests are still pending at this checkpoint.

The first render integration pass exposed two host bugs, now fixed: the append
counter needed device allocation before publishing its address, and work-tile
sizes had to be bounded by the training group rather than the entire state array.
The latter could skip a complete oversized tile and falsely finish a batch. The
scheduler now distinguishes exhausted work from a pending tile that does not fit;
the latter fails the render. A one-sample, empty-field comparison between minimum
and large history budgets checks exact pixel/sample preservation. The early
`history-material` renders are marked INVALID and retained for diagnosis; they
are not admissible image-quality measurements.

## BDPT repeatability audit

The mixed-transparent retention test exposed two independent problems outside
the history allocator. Fresh BDPT renders allocated the light cache before the
effective render dimensions were assigned, producing a capacity of one instead
of 16384 in the 128-square diagnostic. Repeating the same render then used the
correct dimensions. Cache allocation now occurs at the start of `render_samples`,
after effective buffer assignment. With that fix, both one-SPP renders emitted
4096 light paths and retained 4096 vertices, with 2732 sensor shadows each.

Their generated vertex multisets were byte-identical, but their order differed:
only 97 of 4096 slots matched. Atomic reservoir allocation therefore changed the
camera connection candidates and the sensor random seeds. The new GPU index
pass compacts the original reservoir slots in light-path order before camera or
sensor consumption. It preserves the chosen vertices and reservoir weights,
adds four bytes per allocated light path, and does not move vertex payloads.
Texture generation retries finish before this pass; sensor completion markers
continue to belong to the original storage slots. The Release build and install
succeeded. The fixed-seed, same-budget BDPT repeat now passes the unchanged
pixel tolerances: maximum absolute difference 9.536743e-7, linear MSE
3.045560e-16. Both canonical snapshots contain 4096 byte-identical vertices in
the same order (`cache-order-repeat`). The 36 host and five transport checks also
pass. Different history budgets and resize validation now pass as well:
`cache-order-retention` compares 16 and 128 MiB at one SPP, with an intermediate
64-square render. PT maximum difference is 2.980232e-8; BDPT is 1.907349e-6,
within the unchanged relative-plus-absolute tolerance. The BDPT cache snapshots
are byte-identical. `cache-order-crop` additionally changes the logical cache
capacity from 16384 to 4096 and back, emitting 4096/1024/4096 light paths;
restored images and cache contents pass the same checks. All 19 persistent
renderer checks pass in `cache-order-regression`. The serial index pass still needs large-budget performance
measurement; its correctness result is not a renderer speed claim.

`CYCLES_BDPT_CACHE_DUMP` enables diagnostic snapshots in consumption order,
excluding the sensor completion marker and struct padding. The retention test's
`--dump-cache` option checks exact snapshot equality as well as the existing
pixel tolerances. Raw failed runs remain in `retained-transparent`,
`retained-repeat-bdpt`, and `cache-size-repeat`. These failures must not be
reclassified as passed history-budget checks.

Earlier fresh-render BDPT measurements require a light-budget provenance audit.
Logs showing a one-vertex cache do not establish behavior at the configured
budget. Final BDPT comparisons must be rerun with the corrected allocation and
actual emitted/cache counts recorded. PT measurements and CPU-guided reference
images are unaffected by this BDPT allocation bug. All final image comparisons
remain equal-SPP comparisons against high-SPP CPU-guided references; compilation
and runtime measurements are separate.

The benchmark runner now verifies the logged completed sample count before
accepting fixed-sample renders, and records actual BDPT light-cache capacities
and initial emitted/cached/sensor-shadow counts. Resume requires this evidence;
older cached results lacking it are rerendered. A protocol check confirms that
incomplete sample counts and missing BDPT diagnostics are rejected. This is a
benchmark-accounting check, separate from image-quality evidence.

The tiled-texture one-SPP comparison (`cache-order-texture`) also passes the
unchanged relative/absolute pixel tolerance after three light-generation retries
and one sensor-connection retry. Cold and warm runs each emitted 4096 paths,
cached 3966 vertices and queued 3023 sensor shadows. Their linear MSE is
1.256158e-15; maximum difference is 3.814697e-6. This exercises successful sensor
completion markers across retries without treating mere finite output as an
equivalence result. The BDPT cancellation/restart test passes with 0.110117-second
cancel latency and fresh training publications at 1, 2, 4 and 5 samples.

## Corrected-cache equal-sample material results

`cache-order-material-bdpt/report.json` contains 18 verified 512-SPP renders:
two materials, three seeds (101/211/307), Metal BDPT with and without guiding,
and CPU-guided PT. The unchanged reference is the average of CPU-guided
8192-SPP seeds 991 and 1993. Resolution is 256 square, training is 128 samples,
and the GPU field budget is 256 MiB. All BDPT logs record the corrected cache
budget from their first sample. Timing comparisons are explicitly disabled.

| Material | Metal BDPT MSE | Guided Metal BDPT MSE | CPU-guided MSE | Guided reduction vs CPU |
| --- | ---: | ---: | ---: | ---: |
| Rough glass | 0.01223443824 | 0.009709964457 | 0.01159134053 | 16.23% |
| Transmission | 0.01317605846 | 0.01043432232 | 0.01193497104 | 12.57% |

Guiding reduces MSE relative to unguided Metal BDPT by 20.63% and 20.81%,
respectively. Log1p MSE is also lower than CPU guiding in both cases. The saved
comparison figures use the predetermined first seed, 101, with one common
display transform and error scale. No image-specific normalization or denoising
is applied.

These observed error reductions do not establish convergence correctness. The
tall-block mean is about 1.1% above the CPU-guided reference in both Metal BDPT
variants: rough-glass guided mean 0.1571419 versus 0.1554614 reference;
transmission guided mean 0.1495086 versus 0.1478509 reference. Unguided BDPT
has essentially the same offsets (0.1571640 and 0.1495172), so this is not
specific to guiding. Raw region statistics and reference uncertainty remain in
the report. An 8192-SPP transmission diagnostic is running to investigate this
transport/convergence discrepancy. PT quality parity, learned-mixture renderer
integration, complete BDPT transport and warmed rendering performance remain
unmet requirements; this checkpoint is not final acceptance.
