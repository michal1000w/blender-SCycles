# Selected development delivery: native flat limit

Selected application: `build/diffraction_delivery_20260927_flat_native/Blender.app`.
Executable SHA-256: `09a18b3ce82e8fd74aec67e175592951780cf7bb1d270617801463bad21b9e27`.
All 689 resource-manifest hashes were independently verified after packaging.
The Release build and `git diff --check` passed.

## Selection

This package includes the preceding outline optimization and corrects a shader
simplification bug: known constant zero diffraction depth now folds diffraction
coverage to zero before distribution validation. Glass, Glossy, Metallic,
Principled and Refraction preserve their ordinary native carrier, including
native multiscattering compensation where available. Linked coverage is removed
only when depth is known to be zero. Unknown linked depth is not assumed flat.
Nonzero Multi-GGX diffraction remains explicitly unsupported.

Prefer this package for correctness of the flat limit. The preceding outline
package is the fastest observed matched CD/DVD control; we do not claim the
selected package is faster. No benchmark was repeated to obtain a favorable result.

## Final checks on this exact binary

`tests/python/cycles_diffraction_flat_native_control.py` renders five node kinds,
three variants each (disabled, flat, flat with linked coverage), on CPU SVM,
CPU OSL and Metal: 45 renders and 30 comparisons. All passed the predeclared
1e-6 maximum-absolute pixel-error gate. CPU matched exactly. Metal's maximum
error was 2.384185791015625e-7. Each render used 16 fixed samples, adaptive
sampling off and denoising off. These are native-limit regression tests, not
validation of nonzero diffraction physics.

Evidence: `tests/output/diffraction/flat_native_cpu_v1/report.json`,
`flat_native_osl_v1/report.json`, and `flat_native_metal_v1/report.json` under the
same output root. Each folder retains EXRs and an executed-script copy.
Existing Glossy (4 cases) and Principled (15 cases) validation suites passed;
nonzero unsupported distributions still produce errors.

The exact-binary Metal PT CD/DVD benchmark used 512 x 369 pixels, 512 fixed
samples, adaptive sampling off, denoising off, persistent data and EXR writing.
The Apple M5 GPU ran outside the sandbox. One warmup took 9.615 s; the three
measured renders took 1.385, 1.402 and 1.415 s, median **1.402 s**.
The preceding outline median was **1.267 s**, so this sequence was 10.65% longer.
A single sequential comparison cannot establish whether this is a causal
regression. Pixel RMSE against the preceding output was **5.081e-8**, maximum
absolute error **2.861e-6**; all pixels were finite.

Evidence and actual images: `tests/output/diffraction/benchmark_flat_native_v1/`:
`report.json`, `comparison.json`, `render.png`, `render.exr`, `pixels.npy`,
`benchmark.blend`. Scene SHA-256:
`533650dc03b53ead432a9b81e5eb616f2022070b71baac165ac694274c335c2e`.
Script SHA-256:
`473177ffdb90a41c1e34cca095d8265e8006e05b94b46e207566b35f277e4b2c`.

## Broader evidence and limits

The [outline report](cycles_coherent_metal_outline.md) records all 36 direct
scalar coherence reference cases across PT, BDPT, guiding and their combination,
ten material/control renders, and the combined spectral disc scene. Its gallery
has 47 editable scene copies with verified saved settings. Those renders retain
the outline binary's provenance; they are not relabeled as renders from this
package. The older presentation gallery likewise retains final_v3 provenance.

Fast remains the default, with no response-cache construction. Realistic uses
Metal/MPS but remains expensive (about 134 s cache construction in the bounded
lossless fixture). The MPS object-reuse experiment did not materially improve
that cost and was reverted. A standalone SIMD Metal multiscattering-albedo
prototype builds the tested small tables in 0.296–1.444 ms including dispatch
and wait, agreeing with its CPU estimator within 2.98e-7. This prototype is not
integrated into the renderer and does not establish physical model accuracy.

This is a development delivery, not completion of the original scope. General
coherent reflected/refracted multipath is not implemented and the mirror
interference acceptance test fails. Nonzero Multi-GGX diffraction, broad
Realistic profile validation and exhaustive pipeline certification remain open.
Very rough Thin Wall relief can lose substantial energy; indirect scenes retain
noise. Original timeouts and failed comparisons remain recorded, including the
mixed-history BDPT control failure and its passing fresh-process follow-up.
No BDPT brightness was normalized to PT.

## Selected-binary combined transport follow-up

All nine direct scalar reference cases also passed on the selected binary with
Metal BDPT and guiding enabled together: phase zero, phase pi, incoherent,
partial coherence, changed phase wavelength, occlusion, three sources, scene
unit scaling and disabled controls. Each used 128 fixed samples with 32 guiding
training samples, adaptive sampling off and denoising off, outside the sandbox.
Maximum absolute RMSE was 0.00570556065518 (gate 0.006); maximum absolute mean error was
1.10779954932e-06 (gate 0.003). No radiance scale was fitted.

Actual images, scenes, EXRs, independent reference arrays, per-process timings
and artifact hashes are in
`tests/output/diffraction/flat_native_combined_acceptance_v1/index.html`.
The phase-zero and phase-pi images were visually inspected and show the expected
fringe shift. First-use process times include substantial preparation; they
are not warm-render benchmarks. All nine processes exited zero, with no timeout.
The standalone 11-test scalar reference suite also passed.

This closes the selected-binary direct-coherence combined-transport gap only.
The broader material gallery still retains the preceding build's provenance.
General coherent multipath remains missing.

## Selected-binary combined spectral discs

The actual CD/DVD fixture was subsequently rendered on the selected package
with Metal BDPT and guiding together: 512 fixed samples, adaptive sampling and
denoising off, Fast quality, outside the sandbox. The saved scene was reopened
and its settings verified before rendering. The process exited zero; all EXR
pixels were finite. Render time was 87.3369 s including preparation, not a warm
benchmark. Actual PNG inspection shows distinct spectral sectors for the 1600 nm
and 740 nm gratings with residual sampling noise. No exposure or brightness
normalization was applied. This is an appearance/transport check, not an
independent efficiency reference.

The final guiding snapshot records 199 nodes and 64 trained samples. All
4,649,834 sampling floats are finite and 3,465,465 are nonzero. Evidence:
`tests/output/diffraction/diffraction_flat_native_combined_discs_v1/`, including
`physical_discs_bdpt_guided.blend`, PNG, EXR, JSON report, `review.json` and
guiding snapshots. The bounded command, binary hash and script hash are recorded
in `tests/output/diffraction/flat_native_combined_discs_job.json`.

## Selected-binary material follow-up

All ten material/control jobs completed on the selected binary: Thin Wall PT,
BDPT, guiding, CPU OSL, flat and uncovered; Ashikhmin PT, BDPT, guiding and CPU
OSL. Each used 128 fixed samples at 480 pixels wide, with adaptive sampling and
denoising disabled. All output pixels were finite. The flat-versus-uncovered
linear RGB maximum error was 9.5367431640625e-7, passing the unchanged 1e-5 gate.
Relief-versus-flat RMSE was 0.2324746974, confirming an observable relief effect,
not physical correctness by itself. PT and BDPT Thin Wall and sphere renders
were visually inspected; substantial noise remains. No brightness normalization
or PT/BDPT equality gate was applied.

Actual images and editable scenes: 
`tests/output/diffraction/flat_native_material_acceptance_v1/index.html`.
The folder retains process manifests, EXRs, image-control verifier and report,
and hashes of all 30 rendered/scene artifacts. This supersedes reliance on the
preceding package for these ten checks; the older broad presentation gallery
retains its original provenance.

## Selected-binary full presentation suite

The ten-scene presentation suite now also uses this exact selected binary:
CD/DVD, Glass, Glossy, Metallic, Principled film, Refraction, covered disc,
indirect illumination, Thin Wall and Ashikhmin distribution comparison. All
ten jobs exited zero with 70 finite image/pass checks. Material previews use
512 fixed samples with OIDN; indirect illumination uses 2048 fixed BDPT samples.
Adaptive sampling is disabled throughout. These are presentation/denoising
integration checks, not performance benchmarks or physical reference solutions.

`tests/output/diffraction/presentation_flat_native_v1/index.html` links every
preview, raw noisy image, linear EXR and scene. It also includes the nine
selected-build combined BDPT/guiding direct-coherence cases. Nineteen editable
scene copies were saved and reopened: sample count, denoising, guiding, BDPT
settings and separate relative output destinations were verified. All gallery
targets exist. `review.json` records checks and hashes of the scene copies.
The actual overview was visually inspected. Indirect noise and denoising blur
remain visible; the preview is not evidence of physical convergence.

This supersedes earlier reliance on final_v3 for the ten presentation families.
Historical images and failures retain their original provenance. Neither this
suite nor the earlier checks completes general coherent multipath, nonzero
Multi-GGX diffraction, or broad Realistic solver certification.
