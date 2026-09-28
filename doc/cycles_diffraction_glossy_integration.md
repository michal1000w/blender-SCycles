# Glossy diffraction distribution correction

Glossy diffraction now preserves the selected GGX or Beckmann distribution in
SVM and OSL. The earlier prototype always used GGX for its grating component,
even when the uncovered component used Beckmann. The corrected implementation
uses the shared reflective diffraction closure with unit facet Fresnel;
Glossy's existing spectral color basis supplies the closure weight.

Anisotropy, tangent rotation, partial coverage and explicit surrounding-medium
IOR remain active. The wavelength in the medium is lambda_vacuum / n, and both
order spacing and relief phase use that wavelength. This input specifies the
surrounding medium; it is not inferred from the volume stack. Zero diffraction
retains the ordinary Glossy path.

Enabled diffraction with Multi-GGX or Ashikhmin-Shirley now reports an explicit
compiler error. These combinations are still unfinished; silently substituting
GGX was not correct support for them. The native nodes use the Fast scalar
approximation, not the dedicated node's Realistic response cache.

## Validation

- Full Blender build passes, including all three Metal libraries and OSL.
- Expanded shared reflection CPU regression: 63,276 accepted events, zero
  failures, including anisotropic GGX/Beckmann, unit/conductor/F82/Generalized
  Schlick Fresnel, smooth/rough surfaces, and half/full coverage. Maximum scaled
  flat-profile error is 4.07003e-7; reciprocity error is 7.23036e-7. These values
  are from changed anisotropic fixtures, not an accuracy-improvement comparison.
- The unit-Fresnel GGX evaluator also matches the earlier scalar GGX evaluator
  at the tested directions under the unchanged 3e-4 scaled gate.
- A separate smooth-interface check verifies the integer grating equation and
  transverse momentum conservation for surrounding IOR 1.0 and 1.5.
- Runtime checks pass for ordinary default Glossy and enabled Beckmann;
  unsupported Multi-GGX/Ashikhmin diffraction produces the intended error.

Evidence: `glossy_shared_reflection_cpu_v3.log` and
`glossy_support_validation_v1`, under `tests/output/diffraction`.

The representative scene places GGX and Beckmann gratings at equal roughness
side by side; the third material uses rotated anisotropic Beckmann at half
coverage. Metal PT/BDPT/guiding and CPU OSL artifacts are stored separately in
`glossy_delivery_v1`. They use fixed samples with adaptive sampling and denoising
disabled. These integration previews are not convergence or overhead benchmarks.
The earlier GGX-only prototype is not treated as a valid Beckmann reference.

This correction does not add multi-scattering, Refraction-node diffraction,
transmitting Principled diffraction, or general cross-object coherent transport.
It does not establish full electromagnetic accuracy or all pipeline coverage.

All four runtime previews completed with finite pixels and were visually inspected.
The gallery is `tests/output/diffraction/glossy_delivery_v1/index.html`.
The separately packaged application is
`build/diffraction_delivery_20260927_glossy/Blender.app`, SHA256
`902188d468f6eb31aef396a27e57866d344f773090d14e310b72995000c9a265`.
Its clean-environment OSL smoke passes; the bundled shader and kernel hashes
match the build. The package OSL PNG is separately recorded from the inspected development preview.

Diagnostic render times: PT 256 samples 29.78 s, BDPT 128 samples 88.83 s,
guiding 256 samples 75.47 s, CPU OSL 32 samples 4.86 s. Startup and differing
sample counts prevent an efficiency ranking from these integration timings.
