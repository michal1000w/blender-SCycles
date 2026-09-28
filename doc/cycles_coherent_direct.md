# Experimental direct point-source interference

Cycles now has actual light controls for an optional direct scalar interference
model. This is a limited renderer feature, not general coherent multipath
transport. Local spectral diffraction and this source-interference option are
separate features.

In Light properties, open **Coherent Direct Point Sources (Experimental)**.
Give zero-radius point lights the same positive Group, then set their relative
Phase, Wavelength in nm, and Coherence Length in metres. Group zero disables the
feature. Length zero gives the exact incoherent cross-term limit. Positive
length uses the convention `L_c = 1 / sigma_k` for a Gaussian wavenumber spectrum.
Scene unit scale converts physical lengths to scene coordinates.

The wavelength controls interference phase. It does **not** turn the underlying
RGB light into a monochromatic spectral emitter or change its color. Each RGB
channel is a separate scalar field with the light's radiometric color envelope.
This approximation is useful for controllable interference patterns, but is not
a fully spectral coherent material model.

For a receiver, the ordinary BSDF-weighted irradiance of visible source `i` is
`I_i`. The group intensity is

```
I = sum_i I_i + sum_{i<j} 2 sqrt(I_i I_j)
    exp(-0.5 ((r_i-r_j)/L_c)^2)
    cos(2 pi (r_i-r_j)/lambda + phase_i-phase_j).
```

The scattering amplitude magnitude is `sqrt(I_i)` with zero additional material
phase. This is a declared scalar phase-screen approximation. For ordinary PT, point-source NEE has MIS weight one, so the direct estimator
is multiplied by `I/sum_i I_i`, retaining its selection PDF. For BDPT, only NEE
estimates the extra cross terms: its multiplier is
`w_NEE + (I/sum_i I_i - 1)`, preserving the ordinary radiometric MIS partition.
Opaque visibility is traced for each group member. The stable factored distance
difference avoids cancellation from subtracting two nearly equal ranges. Float
geometry still limits optical phase precision, particularly far from the origin;
moving a tiny source pair to 1000 m can change its represented physical phase
substantially. This limitation is measured, not repaired by phase wrapping.

PT and guiding use the camera-path direct estimator. The selected package retains ordinary BDPT emitter sampling and MIS weights, then adds only
the signed direct-interference correction through NEE. Negative correction
samples are retained. This restores ordinary indirect light-path contributions;
coherent phase is still not propagated through reflected or refracted paths.
See [the estimator and acceptance results](cycles_coherent_additive_transport.md).
The earlier additive package timed out during combined BDPT/guiding preparation.
The subsequent Metal outline optimization resolved the tested case. All nine
direct reference cases now also pass on the selected zero-depth-fix package
with BDPT and guiding together: maximum RMSE 0.0057055606552, maximum absolute
mean error 1.1078e-6. Exact-binary artifacts are in
`tests/output/diffraction/flat_native_combined_acceptance_v1/`. The historical
timeout remains preserved; these direct tests do not validate coherent multipath.

Groups are bounded to 16 matched point lights. Constant emission, equal phase
wavelength and coherence length, matching visibility/light-group settings,
zero radius, and shadow casting are required. Volume and transparent-shadow
shaders, light/shadow linking, photon mapping, and caustic-source flags report
explicit errors. Opaque occlusion is supported. Finite apertures, polarization,
general sensor coherence, and arbitrary coherent multipath remain unimplemented.

`tests/python/cycles_coherent_direct_delivery.py` saves real Cycles scenes and
renders and compares them to a separate float64 scalar Lambertian detector
calculation, with 4x4 box-filter quadrature. There is no fitted brightness scale
or reference texture. The declared gates are absolute RMSE below 0.006 and
absolute mean error below 0.003. Cases cover source phase, incoherence, partial
coherence, wavelength, opaque source masking, three sources, and scene units.
Each manifest records the exact executable and fixed-sample settings.

The initial five-case Metal PT run had maximum RMSE below 0.0001. Initial BDPT
phase-zero RMSE was 0.000091776. These results validate the stated direct model
only. Warm PT cases took roughly 0.63–0.71 seconds at 512x256 and 128 samples;
the first render took 51.67 seconds including preparation. They are not an
equal-quality transport ranking. Final-package results are recorded separately
in the delivery report. Adaptive sampling and denoising are disabled throughout.

Independent tests include eleven float64 reference checks and fifteen compiled
kernel-utility cases. The utility comparisons reached maximum intensity error
0.000176 against the double reference on their normal-coordinate fixtures.
The old metadata-only acceptance scenes and their failed phase response remain
preserved as historical evidence; the new light controls supersede them only
for this direct-source scope.
