# Fast diffraction status

New Diffraction BSDF nodes default to **Fast**, with a **Realistic** quality option.
Fast evaluates spectral reflection and transmission directly on the GPU. It does
not build a material cache. Realistic retains the electromagnetic relief model
and its CPU-built cache; GPU construction of that cache is not implemented.

Fast is an intensity approximation. It conserves energy and uses grating-law
angles, but is not a full Maxwell solution. In the recorded 18-query comparison,
maximum single-order efficiency error was 0.14847 of incident power. This is a
measured quality tradeoff, not numerical noise. Cross-object coherent interference
is not implemented.

## Validation

- Production CPU reciprocal-pair test: 130,082 pairs, no failures.
- Production Metal test: 16,384 cases, no branch mismatches; maximum reciprocal
  power error 5.36442e-7. Includes grazing and atomic-direction queries.
- New-node default, Fast/Realistic save/reload and actual legacy file loading pass.
- CPU OSL Fast furnace and Metal Realistic flat-metal furnace pass their analytic
  controls.
- Integrated Fast suite: 68 renders completed across PT, BDPT, guided PT and
  BDPT with guiding. All eight analytic controls pass; maximum mean-energy error
  0.000405366. Adaptive sampling and denoising are disabled, with 1,024 samples.
  Full artifacts and gallery: `tests/output/diffraction/fast_quality_full_suite_review_v1`.
- A separate eight-grating mixture exposed a storage-estimation bug: before the
  fix it returned approximately 75% of expected energy. The corrected graph
  estimate passes on CPU and Metal, including combined BDPT and guiding, with maximum mean-energy error 0.000408075.
  Failed artifacts remain under `fast_closure_storage_before_fix`.
- The corrected build renders the white-slit transmission presentation scene at
  960×540 and 4,096 fixed samples. Its first-order colour locations agree visually
  with independently calculated grating-equation markers. This is a position
  check, not an efficiency reference.

The 68-render suite precedes the storage fix. That fix changed only the host node
header and binary; its GPU kernel source was unchanged. The targeted CPU/Metal
regression and transmission scene use that corrected build. Subsequent optical
table support also changes GPU code and has separate validation records. Do not
describe every artifact as coming from one binary.

## Performance

Apple M5 10-core GPU; 960×691; 1,024 fixed samples; adaptive sampling and denoising off. One warm-up and three measured render calls with persistent data per treatment. Jobs ran sequentially outside the sandbox.

The zero-relief control still uses the Fast spectral node. It is not an ordinary non-spectral material baseline. Paths differ, so this does not measure equal-image or equal-noise convergence. Three runs do not establish statistical significance.

| Transport | Fast diffraction (s) | Zero relief (s) | Difference |
|---|---:|---:|---:|
| pt | 8.302 | 8.351 | -0.58% |
| bdpt | 18.417 | 18.377 | +0.22% |
| guided | 21.265 | 21.137 | +0.61% |
| bdpt_guided | 38.300 | 38.446 | -0.38% |

Binary SHA-256: `4d119ea6afb4f7d379f76374dca9d0d2525ddc6719f09f770066ecc29d7980e9`.

All eight benchmark reports, saved scenes, per-run settings, random seeds and input/source identities passed the runner’s artifact checks. Warm-up and individual timings remain in each JSON report.

The newer table-enabled build (`4e955cdb9930c32fc35564a992d89bfe9677b3ee0a247d9f16e1e831774f7a60`)
has a separate completed eight-job aluminum-table benchmark with the same sample
budget and resolution. Median times are 8.450 s (PT), 18.673 s (BDPT), 21.187 s
(guided PT), and 38.518 s (BDPT with guiding). Relative to constant n,k in the
same Fast node, differences range from -1.84% to +0.93%. This indicates no
substantial table-lookup overhead in this scene; it is not an ordinary-material
baseline. Full results: `tests/output/diffraction/fast_index_aluminum_benchmark_v1/report.md`.

## Remaining work

A subsequent covered-CD test exposed incorrect shadow-caustics culling of
singular grating reflections. The correction preserves those sampled paths;
the Metal image agrees with a no-culling control to relative L1 3.84779e-7.
CPU on/off controls match exactly. The corrected binary is
`5dbbe90cae77c726e71b2718d2b25bd5b9df56eb965882e731d95a451f3d9b8a`.
See `doc/cycles_diffraction_covered_regression.md`. The performance measurements
above precede this correction and remain attributed to their recorded binaries.

The subsequent Fast candidate is
`2bb04b4807ff2a32000d7fe87ee1c3e2023f23f0889ac1214547985c98fbc9fe`.
It avoids manifold solving for a surface proven to be a directly connected
smooth diffraction closure. Diffuse and mixed graphs retain their existing path.
On the covered-CD PT benchmark (384 × 276, 1024 fixed samples), five warmed
render calls per build give medians of 4.439856 s before and 2.626023 s after:
a scene-specific 40.853% reduction. PT and BDPT before/after pixel comparisons
pass without normalization, with relative L1 errors 4.83476e-7 and 2.75843e-7.
This does not replace the earlier full-suite results with results from one new
binary or establish a universal rendering speedup.

The full original scope is unfinished: GPU cache construction for Realistic mode,
optional general coherent transport between objects, broader closure integration,
and remaining pipeline coverage. Wavelength-dependent optical-constant tables are now supported; see
`doc/cycles_diffraction_optical_constants.md` for the newer build and validation.
The timing table above describes the preserved preceding build, not the new
table-enabled binary. The indirect-light presentation images remain noisy at 1,024 samples.
The Realistic label does not certify modal convergence for arbitrary profiles.

Detailed derivation and preserved experiments are in
`doc/cycles_diffraction_fast_interface_candidate.md`,
`doc/cycles_diffraction.md` and `doc/cycles_diffraction_retrospective.md`.
