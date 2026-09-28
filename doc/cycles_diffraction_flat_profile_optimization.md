# Exact flat-profile order pruning

The shared reflective kernel now enumerates only order zero when relief height
is exactly zero. The dielectric kernel similarly restricts reflected orders at
zero height and transmitted orders at zero transmission phase. Geometric range
validation remains separate: a valid active order bound of zero must not be
mistaken for invalid setup. No epsilon or approximate threshold was introduced.
Nonzero Fourier support follows the existing evaluation path.

The full application and all three generic Metal libraries build successfully.
CPU Refraction (including 108 passivity configurations), the shared reflective
closure (63,276 accepted events), and Glass GGX/Beckmann regression checks pass.
Their prior numerical error maxima are unchanged. Logs use the flat_support name.

A sequential Metal before/after check used the same Glass fixture with all relief
depths set to zero, 512 fixed samples, seed 11, adaptive sampling and denoising OFF.
Each process performed two renders. Before: 7.173 / 6.732 seconds; after:
31.520 / 4.739 seconds. Thus the second, warm render was 29.61% faster. The first
optimized run was slower including preparation. This is a single fixture and
single warm observation per candidate, not a general performance guarantee.
Raw float pixel maximum absolute difference was 6.10352e-5; mean absolute difference
was 4.61141e-8. Both images were finite; the optimized image was visually inspected.
The test does not assert bitwise identity. Linear EXRs and raw float buffers are
retained in flat_profile_before_v1 and flat_profile_after_v1.

The selected updated package is build/diffraction_delivery_20260927_optimized/Blender.app,
SHA-256 784100a8bc3e1b0824e17d75da9217dfe6daad27a25ca88c481096681f256e14.
The previous package is preserved. The reusable tools/package_diffraction_build.py
copies the executable, kernel/util sources, compiled OSL shaders and all three
matching Metal libraries, then verifies every copied file by SHA-256. Its manifest
is adjacent to the application. This optimization does not add multiscattering,
Principled transmission or cross-object coherence; those requirements remain open.

The standalone optimized package also passed a nonzero-relief Refraction Metal PT render at 128 fixed samples in 27.69 seconds including preparation. Pixels were finite and the image visually inspected. Evidence: tests/output/diffraction/optimized_package_refraction_v1.
