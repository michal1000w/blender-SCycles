# Standard-Fresnel Principled transmission diffraction

Principled GGX now supports the joint reflection/transmission grating for its
transmission layer when Specular Tint is white and unlinked, Thin Film Thickness
and Transmission Dispersion Scale are zero and unlinked, and Thin Wall is off
and unlinked. Other combinations still produce explicit compilation errors.
This is a real integration of the existing Fast scalar dielectric model, not
complete generalized Principled transmission support.

SVM allocates the covered part with separate white reflection and spectrally
converted sqrt(Base Color) transmission weights. The uncovered fraction retains
native Generalized Schlick. Existing coat/sheen layer weights are applied before
this split; other base components retain their original transmission attenuation.
OSL uses the same joint closure, with an optional Principled transmission flag
that retains white reflection independently of the transmission filter. Ordinary
Glass calls default to the previous behavior. Both backends honor caustic visibility.
The compiler reserves ten extra slots for the combined diffraction layers.

The native Fresnel equivalence only holds under the stated input constraints.
The existing shared Glass grating remains approximate; native layer-albedo estimates
are not an exact grating layering solution. Generalized tint, thin film, dispersion,
thin-wall and multiscattering grating support remain open. In particular the new
compiler checks must not be relaxed just to make a scene load.

## Numerical validation

cycles_diffraction_principled_transmission_test.cpp checks the actual closure
against native Generalized Schlick with standard Fresnel and independent white
or colored transmission. Front/back incidence, two roughnesses and zero/nonzero
relief are covered. All 30,852 accepted sample/evaluate comparisons pass. The
15,504 flat-profile comparisons have maximum scaled error 0.000275276 under the
existing 0.0003 limit. Evidence: principled_transmission_cpu_v1.log. These bounded
fixtures do not certify all native controls or full pipeline combinations.

## Runtime fixture

cycles_diffraction_principled_transmission_delivery.py creates three spheres:
smooth GGX transmission grating, rough colored transmission, and rough half grating
coverage. Scenes, EXRs, PNGs and per-run settings are retained in
principled_transmission_delivery_v1. The PT preview is noisy at 256 fixed samples;
no denoising or adaptive sampling is used. This is a node-integration fixture,
not an equal-variance benchmark or a Maxwell comparison.

The full application and OSL/Metal shaders build successfully. The separately
packaged candidate is build/diffraction_delivery_20260927_transmission/Blender.app;
matching binary, source, OSL and Metal resource hashes are in its adjacent manifest.

## Completed render checks

- pt: 256 samples, 30.94 seconds including preparation, finite and visually inspected.
- bdpt: 128 samples, 94.12 seconds including preparation, finite and visually inspected.
- guided: 128 samples, 81.68 seconds including preparation, finite and visually inspected.
- osl: 32 samples, 7.38 seconds including preparation, finite and visually inspected.

The OSL run used the standalone package with development resource environment variables removed. All seven input-validation assertions passed. That validation process returned exit code 1 because of its deliberate unsupported shader errors; its log and detailed report are retained, not treated as a clean zero-exit job.

## Small positive diffraction weights

Principled compiler traits now use the same positive-weight test as SVM/OSL
execution. Previously, the generic closure-weight cutoff could classify a small
positive diffraction weight as absent even though the kernel evaluated it. That
could omit additional closure storage, wavelength flags or unsupported-input
checks. `has_diffraction()` now controls storage, dispersion, tangent retention
and compiler guards consistently. Exactly zero or negative weights remain on the
ordinary path. Regression cases include a 1e-7 weight for both unsupported
multiscattering and supported standard transmission.

A separate layered fixture exercises simultaneous metallic, transmission, coat,
sheen, alpha and anisotropic reflective lobes, including half grating coverage.
It is a visual/runtime integration check, not an exhaustive closure-allocation
proof or exact grating-layer energy validation.

Layered integration completed: Metal PT256 (40.26s) and CPU OSL64 (25.02s), including preparation; both finite and visually inspected. All nine validation assertions pass, including tiny positive weights; expected shader errors retain process exit1. Evidence: principled_layered_delivery_v1 and principled_transmission_validation_v2. Current package: build/diffraction_delivery_20260927_layered/Blender.app, SHA256 13ea1fc0d37a08e08b7941006b0af056763a54a8ad24989ae8791708c57b7bf8.
