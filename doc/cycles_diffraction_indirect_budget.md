# Indirect diffraction light-path budget pilot

The selected-build 72-render suite completed, but the indirect scene remained
noisy. A separate pilot varied only the requested BDPT light-path count. The
same saved aluminum scene, binary, script, camera sampling, resolution and seeds
were used for all treatments. Adaptive sampling and denoising were disabled.

Configuration: Apple M5 10-core Metal GPU, 480 × 408, 256 camera samples, four
camera samples per light-cache update. Seed 11 is a retained warm-up; seeds 17
and 29 are the measured independent images. Jobs ran sequentially outside the
sandbox. Timings include EXR writing and exclude the first warm-up. They are
render-call measurements, not GPU timestamps.

| Requested light paths | Median measured seconds | Single-seed RGB variance | Variance × seconds |
|---:|---:|---:|---:|
| 16,384 | 6.627542 | 0.181679807 | 1.204090530 |
| 262,144 | 6.455100 | 0.064889257 | 0.418866615 |
| 1,048,576 | 7.659506 | 0.034063136 | 0.260906805 |

The variance is the per-pixel sample variance across the two independent seeds,
then averaged over RGB and the full frame. It measures sampling variation,
not bias or absolute physical correctness. No image brightness normalization
is applied. Spatially correlated splats and only two seeds limit statistical
certainty; this is a pilot, not a general equal-error performance proof.

The one-million setting reduced measured RGB variance by 81.251% relative to
the original budget, for 15.571% more measured time. Its variance-time product
was 78.332% lower. Cycles' informational log confirms an actual cache capacity
of 1,048,576 vertices. The earlier two runs record requested settings but did
not enable that diagnostic log; the host implementation can cap requests to its
concurrent-state capacity. Do not silently equate arbitrary requests with
allocated counts.

Artifacts:
- `indirect_light_budget_pilot_v1`: original and 262,144-path runs, all raw EXRs,
  NumPy pixel arrays, PNGs, saved scenes, provenance and original script snapshot.
- `indirect_light_budget_1m_v1`: one-million run and actual-capacity log audit.
- `indirect_light_budget_three_way_v1.json`: verified image identities, per-seed
  mean standard errors, variance and timing analysis.

The original pilot script was preserved before adding a presentation-only mode.
The new mode skips an extra warm-up and requires exactly one seed; its timing
is explicitly not a benchmark. A 960 × 816, 4096-sample presentation render with
1,048,576 light paths is running under `fast_indirect_presentation_bdpt_1m_v1`.
This setting is selected for that fixture, not promoted to a global BDPT default.
The original feature goal remains incomplete.

## Completed high-sample review

The 960 × 816 BDPT presentation completed at 4096 fixed samples and 1,048,576
logged light-cache vertices. Adaptive sampling and denoising were disabled.
Render-call time including preparation was 344.249 seconds, not a benchmark.
All 1019 frozen source/input/binary hashes and four artifact hashes verified.
The actual image still has substantial chromatic grain and does not pass the
clean final-presentation criterion. See
`tests/output/diffraction/fast_indirect_presentation_bdpt_1m_v1/review.json`.
