# Refraction diffraction delivery

The Refraction BSDF now exposes weight, pitch, depth, duty cycle and tangent for
GGX and Beckmann through SVM and OSL. Partial coverage preserves the ordinary
Refraction contribution. Zero diffraction preserves the native path; zero relief
and matched-index cases have numerical checks. Eevee accepts the socket layout
but does not implement diffraction.

This is a scalar, transmission-only Fast model. Like Blender's native Refraction
building block, it has no Fresnel attenuation. Binary Fourier coefficients allocate
power to propagating transmitted orders. Missing/nonpropagating power is a null
event, not redistributed across surviving orders. This avoids a nonreciprocal
renormalization, but is not a complete lossless optical interface or Maxwell
solution. Use Glass for coupled reflection and transmission. The phase uses the
existing scalar index-contrast approximation.

## Verification

The standalone CPU test passes 45,859 accepted closure samples, 7,665 flat
comparisons, 14,461 reciprocity comparisons and 24,000 matched-index atoms.
Maximum scaled flat error is 0.000283106; reciprocity error is 7.84478e-7.
An independent double-complex Fourier calculation checks propagating probability
mass over 120,000 stratified samples; maximum mass error is 1.34107e-5.
These are bounded fixtures, not an exhaustive precision or energy proof.

The initial failed run is retained as refraction_kernel_cpu_v1.log. Its reverse
reference scaled already-rounded parameters. Version 2 rebuilds both directions
from the same vacuum wavelength and absolute material indices, matching node
construction. No acceptance threshold was relaxed. Existing rough Glass GGX and
Beckmann regression tests also pass (20,000 reciprocal cases each).

The full application and OSL shaders built successfully. The delivery scene has
three labeled single-interface windows: smooth GGX, rough GGX, and half-coverage
rough Beckmann. Neutral white line sources reveal spectral order displacement
and broadening. It is not a full glass slab. Saved scenes, linear EXRs, PNGs and
per-run settings are in tests/output/diffraction/refraction_delivery_v1.
Adaptive sampling and denoising are disabled. Different sample counts make the
integration timings unsuitable for estimator efficiency comparisons.

Reference: https://docs.blender.org/manual/id/5.0/render/shader_nodes/shader/refraction.html

## Completed runtime checks

- pt: 512 fixed samples, 32.06 seconds including preparation; finite and visually inspected.
- bdpt: 128 fixed samples, 89.50 seconds including preparation; finite and visually inspected.
- guided: 128 fixed samples, 66.65 seconds including preparation; finite and visually inspected.
- osl: 32 fixed samples, 3.40 seconds including preparation; finite and visually inspected.

The packaged application also passed a clean-environment CPU OSL Refraction render: 32 fixed samples, 3.31 seconds, finite pixels and visual inspection.


## Refraction passivity check

The expanded CPU regression passes 108 configurations × 2,048 trials (221,184
trials), with 147,984 accepted events. GGX/Beckmann, front/back incidence,
roughness 0/0.2/0.6, depth 0/250/1000 nm and incident cosine 0.05/0.5/1 are
covered. Rejected events contribute zero and remain in the denominator. Maximum
mean throughput and maximum individual throughput are both 1; no negative or
nonfinite throughput occurred. The previous flat, reciprocity, atom and independent
Fourier tests still pass. Log: tests/output/diffraction/refraction_kernel_cpu_v3.log.
This checks passivity in the Cycles closure throughput convention, not a new
Maxwell comparison or exhaustive angular quadrature. The source explains the
G2/G1 masking bound; no render code or acceptance tolerance was changed.
