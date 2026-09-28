# Tinted Principled transmission diffraction

Principled GGX diffraction now retains Generalized Schlick Specular Tint in its
transmission layer. Constant and linked tint inputs are accepted. Base Color
continues to filter transmission independently; half grating coverage retains the
native uncovered material. White Specular Tint retains the existing scalar path.
Thin Film Thickness and Transmission Dispersion Scale must still be zero and
unlinked, and Thin Wall must be off and unlinked. Multiscattering remains unsupported.

## Model and integration

The facet evaluates the native F0-to-white Fresnel interpolation for each spectral
channel, rather than multiplying ordinary reflection by a constant tint. Propagating
nonzero reflected orders use reciprocal pairwise Fresnel budgets. Transmitted orders
use corresponding transmission budgets; the residual mirror carries remaining
power. Smooth delta evaluation and rough all-root evaluation use the same powers.
Sampling uses their spectral mean. SVM and OSL share the implementation.

This remains the Fast scalar relief model. No claim of Maxwell accuracy, general
wave coherence or exact microsurface multiple scattering is added. Native layer
albedo estimates remain approximations for a grating.

A separate generalized payload preserves ordinary Glass storage. Final CPU sizes:
ShaderClosure 96 bytes; ordinary tint 64; coating 80; generalized tint 96. The new
path needs one primary and one auxiliary slot. Two-slot success and one-slot
failure without mutation are regression-tested. The first implementation grew
coating storage to 112 bytes; it was superseded before selecting the package.

## Numerical evidence

The final standalone test passes:
- 233,700 accepted sample/evaluate events, including smooth delta cases.
- 117,210 zero-depth comparisons against native Generalized Schlick; maximum
  scaled error 0.000275289 under the unchanged 0.0003 limit.
- 153,700 reciprocity comparisons; maximum scaled error 1.7714e-5.
- 324 facet configurations spanning front/back incidence, grazing angles, pitch,
  depth, duty cycle and dark/bright spectral tints. Tested individual order powers
  remain in [0,1].
- Separate Refraction regression and its 108 passivity configurations still pass.

Logs: principled_tinted_transmission_cpu_v1 through v6 and
tinted_refraction_regression_v1. These are bounded tests, not exhaustive angular
convergence or a full-pipeline proof. The first Metal build failed on missing
pointer address-space qualifiers; explicit private qualifiers fixed it. The failed
build log is retained as principled_tinted_metal_build_failure_v1.log.

## Renders and package provenance

principled_tinted_delivery_v1 contains Metal PT256, BDPT128, guiding128 and CPU
OSL32 renders before the storage-only correction. All use fixed samples with
adaptive sampling and denoising disabled. Those images keep their original build
provenance. The nine supported/unsupported-input assertions pass; the validation
process returns exit1 due to deliberate unsupported shader errors, not a clean
zero-exit run.

The selected storage-corrected package is
build/diffraction_delivery_20260927_tinted_v2/Blender.app. Its executable, kernel
sources, OSL shaders and three Metal libraries are hash-verified in the adjacent
resource_manifest.json. The earlier _tinted package is superseded. Post-correction
standalone Metal/OSL checks are stored separately in principled_tinted_delivery_v2.
No timing here establishes equal-variance efficiency or a universal fastest build.

Corrected package checks completed: Metal PT256 33.65s and CPU OSL32 7.44s including preparation. Both finite and visually inspected, with development resource overrides removed. Binary SHA256 6e85ee28bf0522c5d5fac651166f346b7eba88e69dbceaeb9e4f656a5a54068a. All ten final input assertions pass, including a procedural Noise Color link to Specular Tint. Deliberate unsupported shader cases retain process exit1.
