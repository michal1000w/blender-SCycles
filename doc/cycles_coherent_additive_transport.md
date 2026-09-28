# Additive direct-coherence transport correction

This package is superseded by the [selected Metal outline build](cycles_coherent_metal_outline.md),
which preserves this correction and passes the bounded combined-mode checks.
All measurements below retain their original package provenance.

The preceding mirror acceptance run established a real energy deficit: coherent
source means were approximately 0.0101 versus a reflected-source reference near
0.048. The incoherent BDPT control was near 0.0484. The previous implementation
excluded every positive-coherence point source from light-path emission, which
also removed its useful indirect strategies.

The new candidate restores ordinary emitter sampling and all original BDPT MIS
weights. If source i contributes unweighted direct intensity C_i and the direct
group ratio is R = I_coherent / I_incoherent, its NEE contribution becomes
C_i * [w_NEE + R - 1]. Ordinary BDPT estimates the incoherent baseline; the
additional, unweighted-by-MIS NEE estimator supplies only pair cross terms.
Averaging over the original source-selection probabilities gives
I_incoherent + (I_coherent - I_incoherent) for the direct connection. Ordinary
indirect radiometric energy remains present. Zero cross terms leave the baseline
unchanged. Multiplying the correction by w_NEE would underweight interference.
Clamping negative contributions would bias destructive interference.

The renderer retains signed film contributions. Existing shadow termination and
sample-clamp magnitudes use absolute values. Nonzero sample clamping still biases
the signed estimator, so numerical acceptance uses zero clamping. Correction is
limited to ordinary surface NEE, excluding MNEE. In combined BDPT/guiding, the
signed NEE observation is omitted from positive-radiance guide training;
ordinary light-subpath observations continue training. The guide changes the
sampling proposal, not the signed film contribution.

This correction restores missing indirect illumination; it does not transport
coherent phase through a mirror or refraction. The full mirror phase-reference
gates must still fail until general coherent multipath is implemented. A good
mean alone is not accepted as evidence of that feature.

Candidate package: `build/diffraction_delivery_20260927_coherence_additive/Blender.app`,
executable SHA-256 `1d985c9b7b0c2deab6c9c49cea23b0758077281f38cba8cc1dc649a18a3e1d4f`.
CPU/OSL and all three Metal variants built successfully; all 689 packaged
resource hashes were independently verified. This is the selected development package because it restores the omitted ordinary
BDPT light-path strategies. It is not a completed full-scope implementation.

Eleven independent scalar reference/estimator tests pass, including exact
source-draw enumeration against complex-field sums with unequal source PDFs,
unequal MIS weights, destructive interference and an indirect baseline.
All nine actual Metal BDPT direct-coherence cases also pass unchanged gates:
maximum RMSE 0.00570585 (three-source case), maximum mean error 2.10988e-6.
The two phase cases have RMSE below 0.000096. Adaptive sampling and denoising
were OFF; every render used 128 fixed samples. Exact evidence is in
`tests/output/diffraction/coherence_additive_bdpt_v1`.
The unchanged mirror suite completed on Metal with 128 fixed samples per image.
All three full image/phase gates still fail; no thresholds were changed.

| Case | Old mean | Candidate mean | Analytic mean | Candidate RMSE |
|---|---:|---:|---:|---:|
| Phase zero | 0.0101553 | 0.0485017 | 0.0482839 | 0.1415673 |
| Phase pi | 0.0101218 | 0.0483576 | 0.0476758 | 0.1411953 |
| Incoherent control | 0.0483743 | 0.0480451 | 0.0479799 | 0.1364333 |

This is evidence that the large mean-energy deficit is removed. It is not a
converged mirror image or proof of reflected phase transport. Sparse light-path
sampling causes large per-pixel noise. The phase-profile error remains an open
acceptance failure. Raw EXRs, ROI arrays, commands, script snapshots and exact
metrics are in `build/tests/python/cycles_coherent_mirror_render_additive_20260927`.
The PT phase-zero check passes (RMSE 9.17753e-5). The guiding-only check
passes (RMSE 9.17752e-5), with 32 training samples and 128 total fixed samples.
Its diagnostic snapshot contains 803 spatial nodes and 18,762,898 finite
sampling floats, including 179,872 nonzero entries. This verifies active Metal
guiding machinery in this fixture, not convergence or guiding quality in every
scene. PT and guiding preparation-inclusive elapsed times were 43.298 s and
141.647 s respectively; they are not warm performance comparisons.
The matched CD/DVD benchmark used the same scene and script hashes as the
previous package: Metal PT, 512x369, 512 fixed samples, adaptive and denoising
OFF, one warmup and three measured renders including EXR writing.
Warmup was 43.9595 s. Measured runs were 1.49497, 1.59209 and 1.59244 s,
median 1.59209 s. The preceding specialization package measured median
1.51246 s: this candidate is 5.27% slower in this bounded run, with overlapping
ranges. No speed superiority is claimed. The ordinary CD/DVD image differs by
RMSE 5.17200e-8, maximum absolute difference 3.33786e-6, without normalization.
Exact evidence: `tests/output/diffraction/benchmark_coherence_additive_v1`.
The current selection prioritizes restored transport over that measured timing
advantage of the previous, transport-deficient package. No benchmark was
repeated to seek a more favorable number.

The combined BDPT+guiding diagnostic was explicitly stopped after the declared 600-second preparation limit (607 seconds observed immediately before termination). The Metal compiler was active, but no image or guiding snapshot had been produced. This is a recorded timeout, not a physical-reference failure or a passing combined-mode test. Evidence: `tests/output/diffraction/coherence_additive_combined_v1/timeout.json`. It will not be retried in this validation pass.
