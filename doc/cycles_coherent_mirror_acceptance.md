# Reflected-path coherence acceptance fixture

The fixture uses two point sources, a unit-reflectance planar mirror, a white
Lambertian detector and opaque masks that block direct source-to-detector paths.
The camera views a predeclared central detector region. The reference unfolds
each source through the mirror plane and independently evaluates scalar
interference using the complete reflected path lengths. A common mirror phase
cancels between these two paths.

The source script is `tests/python/cycles_coherent_mirror_acceptance.py`.
The three saved scenes and double-precision reference arrays are in
`build/tests/python/cycles_coherent_mirror_acceptance_specialization`.
They cover phase zero, phase pi, and a zero-coherence-length ordinary BDPT
control. Pixel coordinates are derived from the actual orthographic camera
matrix. Both renderer and reference use a box filter; the reference uses 4x4
subpixel quadrature. Bounce limits isolate one diffuse detector and one glossy
mirror bounce. No texture substitutes for the reference pattern.

The render runner is `tests/python/run_cycles_coherent_mirror_acceptance.py`.
It uses Metal BDPT, 128 fixed samples, adaptive sampling OFF and denoising OFF.
The predeclared acceptance limits are absolute mean error 0.005 and RMSE 0.012,
including a separate phase-difference comparison. Finite pixels alone do not
pass. Neither brightness nor exposure is fitted to a PT reference.

This is an acceptance fixture for missing functionality. The direct coherent estimator does not transport phase through the mirror.
The initial specialization package also excluded positive-coherence sources
from radiometric BDPT light subpaths. The subsequent
[additive transport package](cycles_coherent_additive_transport.md) restores
those ordinary strategies and the observed mean illumination, while the phase
gates remain failing. A failure here must not be relabeled as supported general
interference. The measurements below retain their original package provenance.

The first sandboxed attempt stopped before rendering because Metal was not
visible. It is preserved separately from the actual outside-sandbox GPU run.

## Actual outside-sandbox result

All three Blender processes exited successfully and rendered on Apple M5 (10 GPU cores). All three fail the unchanged per-pixel acceptance gates. Results are preserved in `build/tests/python/cycles_coherent_mirror_render_specialization_20260927_outside_sandbox/results.json`.

| Case | Render mean | Reference mean | Absolute RMSE |
|---|---:|---:|---:|
| phase_0 | 0.01015527 | 0.04828394 | 0.05148520 |
| phase_pi | 0.01012181 | 0.04767578 | 0.05105839 |
| incoherent_bdpt_control | 0.04837431 | 0.04797986 | 0.13710054 |

The coherent phase-difference RMSE is 0.06837015, above the predeclared 0.012 limit. The incoherent control has mean error 0.00039445 but substantial sparse-sample noise (RMSE 0.137101); it is not marked a pass. No extra samples or revised thresholds were used to hide that outcome. The coherent cases substantially underestimate average illumination and fail phase response.

`mirror_acceptance.png` shows actual and analytic ROI images side by side with the same fixed linear display range 0–0.105 (values outside that range clip for display only). Arrays and raw EXRs retain full radiance values. This comparison is an acceptance failure, not a completed general-coherence feature.

The exact executed runner and worker are retained beside the results as `executed_runner.py` and `executed_worker.py`. After this run, the reusable runner was changed to preserve HOME and the normal Metal cache, clear only the four Blender/Cycles override variables, use two CPU threads and enforce a 360-second process timeout. Those runner-only changes passed syntax checks and were not used to regenerate the recorded renders.
