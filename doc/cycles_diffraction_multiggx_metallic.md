# Constant-input Metallic MultiGGX diffraction

Development package: `build/diffraction_delivery_20260927_metallic_multiggx_v5/Blender.app`.
Executable SHA256: `99d0c53c3fc6822c05bd0f2339fde0b87ec3f07e1d9758f77a7fd8b2974bc662`.

The Metallic node now prepares the directional grating albedo cache for MultiGGX,
using the same CPU/Metal builder as Glossy. Both physical conductor and F82
closures receive the cache through SVM and OSL. The covered component uses the
grating missing-energy lobe; only the uncovered component retains native planar
GGX energy compensation. Metallic repeated-bounce absorption includes both
Fresnel factors because its closure weight does not already contain base color.

Uncoated average Fresnel uses the native conductor/F82 functions. Coated covered
components use a 16-point angular Gauss-Legendre average of the native film
Fresnel. This is an approximation, not a validated arbitrary-thickness solution.
Cache-dependent roughness, anisotropy, pitch, depth and duty must remain constant.
Anisotropic materials accept live or constant-folded explicit tangents.

## Verified checks

`tests/output/diffraction/metallic_multiggx_{cpu,osl,metal}_v5/report.json`
record 16 passes each: physical conductor and F82, full/half coverage, 0/150 nm
film, anisotropy with a folded constant tangent, flat/disabled controls, and
colored passivity. Each render uses 1024 fixed samples, 32 pixels per dimension,
seed 11, no adaptive sampling and no denoising. Metal ran outside the sandbox.
Maximum unit-reflector mean energy error was 0.51832%; maximum CPU/Metal channel
mean difference was 1.091e-5. Colored cases test passivity, not exact color.

`build/tests/performance/metallic_multiggx_native_v1/results.json` records all
four native suites passing. The extended closure test includes 28,665 sample/eval
comparisons, maximum reciprocity error 1.91e-6 and maximum unit-furnace quadrature
error 0.00206. The standalone fixture does not contain Blender's thin-film LUT;
film verification is limited to the actual Blender render cases above.

## Remaining work

This does not complete Principled, Glass or Refraction MultiGGX integration,
texture-varying cache geometry, or general coherent reflected/refracted transport.
It does not establish arbitrary-profile accuracy, converged film appearance,
or a performance ranking. The general mirror interference reference still fails.
The earlier v3 full gallery and timing results refer to their recorded binary,
not this package. The latest representative Metallic scenes are additional
integration checks, not physical-reference or convergence proofs.

The node validation driver now accepts `--metallic`. Both SVM and OSL pass all
10 cases, including expected rejection of each texture-varying cache parameter.
Reports: `build/tests/performance/metallic_validation_v5/report.json` and
`build/tests/performance/metallic_validation_osl_v5/report.json`. Blender exits
with status 1 after expected render errors; all individual assertions pass.

The PT presentation scene is
`tests/output/diffraction/metallic_multiggx_presentation_pt_v5/metallic.blend`.
It renders three rough metals: single-scatter control, coated conductor MultiGGX,
and coated F82 MultiGGX with half coverage. At 720x400 and 256 fixed samples the
render took 34.789 seconds including preparation. This is not a warm benchmark.
The raw PNG is visibly noisy; subtle diffraction at roughness 0.6 is expected.

The matching BDPT-plus-guiding scene completed with finite pixels at the same
resolution and sample count in 168.335 seconds including preparation:
`tests/output/diffraction/metallic_multiggx_presentation_combined_v5/metallic.blend`.
Both raw previews were visually inspected. No PT/BDPT brightness-equality gate
was applied, and these elapsed times are not an isolated warm-performance ranking.
