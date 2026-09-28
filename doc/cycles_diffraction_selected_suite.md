# Selected-build full-suite audit

The current run is `tests/output/diffraction/fast_selected_full_suite_v2` on
binary `2bb04b4807ff2a32000d7fe87ee1c3e2023f23f0889ac1214547985c98fbc9fe`.
All 72 jobs completed: 15 numerical and three appearance fixtures for each of
PT, BDPT, guided PT and BDPT with guiding. All use 1024 fixed samples, adaptive
sampling and denoising disabled. GPU jobs are sequential and outside the sandbox.
Appearance fixtures embed the aluminum table; numerical controls retain their
independently referenced materials. The runner verifies frozen inputs around
every job. Process durations include preparation and are not benchmarks.

## Numerical audit completed

`fast_selected_full_suite_v2_review` contains the complete gallery, numerical
analysis and `audit.json`. All 72 renders completed with no execution failures.
The final audit verified frozen inputs, planned-job order, saved scene/EXR/preview
identities and fixed-sample/transport settings. Earlier partial reviews remain
preserved as snapshots.

- Eight analytic smoke controls passed; maximum mean RGB error 0.000405365805.
- Fifty-two Fast-model reference comparisons are recorded as approximation
  diagnostics, not electromagnetic-accuracy passes.
- Eight complete reflected/transmitted angular partitions are recorded; maximum
  channel discrepancy from the independent neutral reference is 0.000117127020.
- References match the preserved earlier reference hashes. No PT normalization
  or equality-to-PT criterion is used.

## Presentation review

All twelve appearance renders were inspected. Bare discs retain the broad CD
and narrower DVD spectra across all four modes. Covered discs retain diffraction
through the explicit dielectric layer. Sampling noise remains visible.
All four indirect images are insufficiently converged for a final clean
presentation at this budget. Guided modes reveal clearer geometry and caustic
structure but retain spectral noise. The final combined-mode render took
approximately five minutes including preparation; it is not a warmed benchmark.
All raw images are retained. A light-path-budget experiment is warranted:
16,384 light paths per four-sample batch is sparse relative to 960 × 816 pixels.
The subsequent three-budget pilot supports increasing the light budget: the
one-million-path setting reduced measured two-seed RGB variance by 81% for
about 16% more render-call time at the pilot resolution. This does not establish
the sole cause of all noise or physical accuracy. See
`cycles_diffraction_indirect_budget.md`; a higher-sample presentation render is
being checked separately.

Completion of this suite will not complete the original feature request.
General cross-object coherent transport, GPU construction of the Realistic
cache, broader shader integration and remaining pipeline validation are still
open; see `cycles_diffraction_completion_audit.md`.
