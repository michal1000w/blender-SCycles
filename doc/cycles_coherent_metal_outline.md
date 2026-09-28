# Selected Metal development build — combined transport

The additive package restored ordinary coherent-emitter BDPT light paths, but
its combined BDPT/guiding diagnostic produced no image within a declared
600-second preparation limit. That failure remains recorded in
`tests/output/diffraction/coherence_additive_combined_v1/timeout.json`.

The new candidate changes two function attributes, not sampling equations:

- `coherent_direct_light_scale` explicitly uses Metal `noinline`. The portable
  `ccl_device_noinline` macro has no such attribute on Apple Metal, despite the
  original helper's intent to keep its field workspace separate.
- `bsdf_eval` explicitly uses `noinline` in generated Metal libraries whose
  transport mask contains both BDPT and guiding. This avoids repeatedly copying
  the closure dispatch into forward/reverse PDF and guiding mixture queries.
  Other transport-mask combinations retain the previous declaration. The full
  BDPT/photon/guiding mask omits this source define and does not get this second
  attribute; that combination is outside this check.

The coherent-helper attribute applies to all Metal modes. CPU/OSL sampling
code and BSDF equations are unchanged. The bounded validation below supports selecting this development build; it does
not certify arbitrary scenes or the unfinished full feature scope.

Package: `build/diffraction_delivery_20260927_coherence_outline/Blender.app`.
Executable SHA-256:
`356157e7c0c456f3e3075d1b257a72581030e2c6d915827262c1d122fd259a1e`.
CPU/OSL and all three Metal variants built successfully; 689 packaged resource
hashes were verified, including the rebuilt precompiled Metal libraries.

## Combined direct-coherence result

The same phase-zero oracle, 128 fixed samples, adaptive sampling and denoising
OFF, BDPT and guiding both ON, 32 training samples, completed in 142.309 seconds
for the process (138.612 seconds for the preparation-inclusive render).
The previous package timed out without an image after 607 observed seconds.
This is a bounded observed preparation improvement, not a universal compiler
speedup or an equal-quality warm performance comparison.

The unchanged absolute-radiance gate passes: RMSE 9.55896e-5 and mean error
2.53419e-7. The guiding snapshot reports 32 trained samples, 803 spatial nodes,
and 18,762,898 finite sampling floats with 1,327,359 nonzero entries. This
verifies active combined machinery in this scene, not arbitrary-scene guiding
quality or coherent multipath correctness.

Evidence: `tests/output/diffraction/coherence_outline_combined_v1` and the
adjacent `_job.json`. All remaining bounded checks completed successfully.

## Final checks and selection

The outline package is selected for the supported development scope. PT's
phase-zero oracle passes with absolute RMSE 9.17753e-5. The actual spectral
CD/DVD scene completed with BDPT and guiding together at 512 fixed samples,
512 × 369 pixels, adaptive sampling and denoising OFF. Preparation-inclusive
render time was 143.156 seconds. The saved PNG was visually inspected: distinct
spectral sectors are visible, with residual sampling noise. This is a transport
and appearance check, not an independent absolute-physics reference.
The guiding snapshot contains 199 nodes and 64 trained samples; all 4,649,834
sampling floats are finite, with 3,465,507 nonzero entries.

Scene, EXR, PNG and review: `tests/output/diffraction/diffraction_outline_combined_discs_v1`.
Exact commands, binary and script hashes: `tests/output/diffraction/coherence_outline_remaining_checks.json`.
The executed orchestrator is preserved alongside that manifest.

The matched Metal PT benchmark uses the identical scene and script hashes as
all four preceding controls: 512 fixed samples, adaptive sampling and denoising
OFF, persistent data, one warmup and three measured renders including EXR writes.
Warmup: 34.948 seconds. Measured runs: 1.267193, 1.274733, 1.252589 seconds;
median **1.267193 seconds**, the lowest observed median among these controls.
This is **20.41% less elapsed time** than the additive package (1.592092 seconds)
and 16.22% less than the specialization package (1.512459 seconds).
Pixel RMSE versus the additive package is 5.17757e-8; maximum absolute difference
is 3.81470e-6, without normalization. Evidence and all four comparisons are in
`tests/output/diffraction/benchmark_coherence_outline_v1`.
These are small, sequential fixture measurements, not a statistical guarantee
of general speed superiority or an equal-error convergence study.

Earlier 37-render feature and presentation suites retain their original binary
provenance. They were not all rerun on this package; this change alters function
attributes, not their sampling equations.

General reflected/refracted interference and Multi-GGX diffraction remain
unimplemented. The additive package's failing mirror phase references remain
applicable; these function attributes do not add optical phase transport.


## Extended combined-mode check: partial, timeout preserved

A subsequent nine-case run on this exact package, outside the sandbox, used
128 fixed samples and 32 guiding training samples with adaptive sampling and
denoising OFF. Six cases passed their unchanged independent radiance gates:
phase zero, phase pi, zero coherence, partial coherence, changed phase wavelength
and opaque occlusion. The process reached its declared 360-second total limit
before completing the three-source case; unit scaling and the disabled-option
case were not reached. This is not a nine-case pass. Repeated shader preparation
remains a practical cost, despite the earlier single-case and disc successes.
No timeout was silently extended and no acceptance threshold was relaxed.

Partial images and metrics: `tests/output/diffraction/coherence_outline_combined_all_v1`.
Terminal job status and exact commands: `tests/output/diffraction/coherence_outline_combined_all_job.json`.
The executed runner is `tests/output/diffraction/run_coherent_outline_combined_all.py`.


## Remaining combined cases completed separately

The three unfinished cases were then run in separate bounded processes using
exactly the same binary, script, 128 samples, 32 guiding training samples and
unchanged gates. All three processes exited zero. Together with the six earlier
results, all nine direct scalar cases pass. Maximum absolute RMSE is
0.005705560645 (three sources), below the existing 0.006 threshold. This case
has little margin and is not a convergence proof. Off versus zero-coherence
maximum pixel difference is 0.000113368; these guided outputs are not bitwise
identical. Neither image was rescaled or fitted to the other.

The actual nine-image contact sheet was visually inspected; phase shifts,
partial contrast, wavelength spacing, opaque masking and flat disabled controls
are represented. The case named `red` changes the phase wavelength only; the
RGB emission remains white, as specified by the current model.

Exact commands and process results: `tests/output/diffraction/coherence_outline_combined_remaining_jobs.json`.
Aggregate metrics, source-manifest hashes and actual contact sheet:
`tests/output/diffraction/coherence_outline_combined_summary`.
Reproducer: `tests/output/diffraction/summarize_outline_combined.py`.
The earlier all-in-one 360-second timeout remains preserved. Separate successful
runs do not erase that preparation-cost limitation or prove coherent multipath.


## Repeatable feature-suite coverage

`tests/performance/cycles_diffraction_final_feature_suite.py` now includes the
nine combined BDPT/guiding cases as individual 360-second bounded jobs, in
addition to the original 13 jobs. It records a running state before launching,
then an explicit passed/failed/timeout state and worker-script hash. Nonzero
render exits and timeouts cause the suite to exit one. Previous ad hoc runner
snapshots remain unchanged to preserve the provenance of executed experiments.

Mocked orchestration checks covered all 22 successful jobs, fail-fast on a
nonzero worker exit, and fail-fast on timeout. Results:
`tests/output/diffraction/outline_feature_runner_control_checks.json`.
These mocks do not count as GPU evidence; the whole amended 22-job suite has not
been rerun. The real combined-mode results are documented above.

## Full-package regression attempt and per-case limits

`tests/output/diffraction/final_features_outline_v1` records the subsequent
selected-package run: all nine PT reference cases passed, then the grouped
BDPT job passed six cases before its 360-second total limit. The suite exited
one and did not run the later jobs. This is a failed full-suite attempt, not
an all-feature pass. The manifest and partial output are preserved.

The reusable runner now gives all four coherent transport modes individual
case processes (36 jobs), followed by the ten material/control jobs. Each
retains the same 360-second limit. Optional repeated `--job` arguments permit
running unfinished jobs without repeating passed work; manifests explicitly
record all available jobs, scheduled jobs and whether the selection is partial.
This changes process budgeting, not physical gates or sample counts.
Mocked full scheduling (46 jobs), selected scheduling, failure and timeout
checks passed; evidence is `tests/output/diffraction/outline_feature_runner_per_case_checks.json`.
The revised full schedule has not yet completed on the GPU.

## Selected-build feature results completed across bounded jobs

All 22 unfinished jobs in `final_features_outline_remaining_v1` subsequently
completed with zero exits. Together with the prior exact-build results, this
provides **36 direct scalar reference renders** (nine each for PT, BDPT, guided
PT and combined BDPT/guiding) and **ten finite-output material/control renders**
(Thin Wall and Ashikhmin across Metal PT/BDPT/guiding and CPU OSL, plus flat and
uncovered Thin Wall). This is a collection of preserved bounded runs, not a
claim that the earlier timed-out all-in-one process passed.

`tests/output/diffraction/final_features_outline_aggregate_v1` links the original
outputs without copying or changing them. Its source manifest records exact
source locations/hashes, and `review.json` contains per-transport reference
metrics. All ten material images were visually inspected; 128-sample noise is
substantial, so these are transport/appearance tests, not converged beauty
renders. BDPT brightness was not forced to match PT.

Thin Wall flat versus uncovered maximum absolute error is 1.19209e-6, passing
the unchanged 1e-5 limit. Relief versus flat RMSE is 0.232475, confirming the
relief branch changes the image. PT and guided disabled-versus-zero-coherence
controls pass at 1.19209e-7 maximum difference.

The first BDPT control comparison mixed a mid-process zero-coherence image
with a fresh-process disabled image and failed the 1e-5 gate at 0.000113368.
That failure remains in the aggregate `image_controls.json`. A fresh-process
zero-coherence rerender, using the same fixed settings and selected binary,
compared with the fresh disabled image gives maximum difference 1.19209e-7 and
RMSE 7.91339e-9, passing the unchanged gate. Evidence:
`tests/output/diffraction/outline_bdpt_control_fresh_v1/comparison.json`.
This supports a process/scene-history explanation but does not identify the
underlying cache or sampling-state cause. No original image was replaced.

The requested broader production scope remains incomplete: these results do
not implement coherent multipath, MultiGGX diffraction or arbitrary-profile
Realistic-solver validation. Older broader presentation images retain their
original binary provenance.


## Consolidated review artifact

The exact-build review page is
`tests/output/diffraction/final_features_outline_aggregate_v1/index.html`.
It links all 46 feature/control renders and the combined-mode CD/DVD scene,
plus 47 editable scene copies under `editable_scenes`. The exporter changed
only render/compositor destinations, reopened each saved copy and checked its
sample, denoising and transport settings. The page explicitly separates earlier
final_v3 presentation evidence, incomplete features, and retained failures.
All local page targets exist; all 689 frozen selected-package resource hashes
were reverified. The page was assembled from already inspected actual images;
no new rendering or pixel edits were performed for the gallery.

Reproducers: `tests/python/cycles_diffraction_export_verified_scenes.py` and
`tools/build_diffraction_verified_gallery.py`.
