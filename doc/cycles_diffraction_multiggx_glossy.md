# Constant Glossy MultiGGX diffraction

The Glossy BSDF now has a bounded multiscattering compensation path for a
reflective, native GGX grating with constant roughness, anisotropy, pitch,
depth, duty cycle, and surrounding-medium IOR. Scene preparation builds one
directional unit-reflector albedo table per exact material request and shares
its handle across SVM and OSL closures. The Metal builder and CPU fallback use
the same native facet, GGX masking, diffraction-order geometry, and order-power
functions. The cache is separate from the smooth physical grating response
cache. Anisotropic materials require a linked or constant-folded Tangent so the
table's local grating frame is defined at each hit. A Tangent node (for example
UV Map) and a linked constant Combine XYZ are both supported in the v4 follow-up.
The v3 delivery incorrectly rejected the constant-folded case.

The default table spans the renderer's 380–780 nm wavelength domain with 16
wavelength slices, 8 quadratic-mu nodes, 16 periodic azimuth nodes, and 512
facet samples per node. It stores the exact projected-solid-angle integral of
its piecewise-linear angular interpolation for each wavelength. Evaluation
adds a reciprocal missing-energy lobe. Sampling mixes the native facet/order
proposal with a cosine proposal, and both branches report the same mixture
PDF. The native proposal can retain below-surface null events. The zero-depth
path keeps the existing native energy-preserved GGX carrier.

Glossy color is passed independently of closure weight. The extra lobe uses
the declared spectral average-Fresnel factor, with one Glossy color factor
already carried by the closure weight. This is an approximation to repeated
facet interactions, not an electromagnetic multibounce solution. The OSL
shader passes color through a zero-default absorption keyword, preserving
white behavior for existing custom OSL uses. It also selects the dedicated
`multi_ggx` closure for the uncovered fraction; this matters when diffraction
coverage is between zero and one.

Linked roughness, anisotropy, pitch, depth, duty, or medium IOR are explicitly
rejected during scene preparation because a scene-time table cannot represent
their per-hit values. This path does not cover dielectric reflection or
transmission, Glass, Refraction, Metallic, Principled, Beckmann, Ashikhmin,
or coatings. Other supported Glossy diffraction distributions keep their
existing behavior. The table approximation has no general strict passivity
proof; the earlier varied-profile probe in
`doc/cycles_diffraction_multiscatter_probe.md` found larger errors outside the
single-profile angular/spectral interpolation checks.

## Bounded numerical checks

The native closure test in
`tests/performance/cycles_diffraction_multiggx_closure_test.cpp` constructs
production CPU tables, installs them into `KernelGlobalsCPU`, and exercises
both sample/evaluate branches at alpha 0.3, 0.6, and 0.9. Across 21,749
accepted events, sample and query PDF/value differences were zero at the
reported precision. Reciprocity error was at most 1.91e-6. Independent
64×128 solid-angle quadrature at two incident elevations per alpha measured
white-furnace energy within 0.002061 of one. Integrated PDF mass matched the
empirical accepted-event fraction within 0.00223; mass below one reflects
native null events. Isolated extra-lobe values for color 0, 0.5, and 1 were
0, 0.0396723, and 0.102398 at the selected test directions. Raw output and
source/executable hashes:
`build/tests/performance/multiggx_closure_v1/results.json`.

The packaged v3 Blender app rendered six nonzero-depth white Glossy profiles
(roughness 0.3/0.6/0.9 × pitch 740/1600 nm), flat and disabled controls, and
half coverage at 32×32, 1,024 fixed samples, with denoising and adaptive
sampling off. CPU, Metal, and CPU OSL each passed all nine cases. The largest
absolute object-region RGB deviation from the white furnace target was
0.005269. CPU/Metal means differed by at most 1.32e-7 across cases; the
CPU/OSL means differed by at most 0.003015. Per-case `.blend`, EXR, PNG,
settings, and hashes are in:

- `tests/output/diffraction/multiggx_glossy_cpu_v3/report.json`
- `tests/output/diffraction/multiggx_glossy_metal_v3/report.json`
- `tests/output/diffraction/multiggx_glossy_osl_v3/report.json`

The packaged v3 host validation passed all ten distribution and linked-input
cases. Its report is
`build/tests/performance/multiggx_glossy_host_validation_v3/report.json`.
Blender's process exited with status 1 because the six intentionally rejected
render attempts set its global render-error status; all expected errors were
caught and matched in the report. The white-furnace renders use one seed and
the tested profiles only. They establish this bounded implementation check,
not a global physical-accuracy or performance bound.

Reproduce the native CPU checks from a configured build with:

```sh
python3 tests/performance/run_cycles_diffraction_native_extensions.py \
  --output-dir build/tests/performance/native_extensions_multiggx_check
```

The runner records compile/link commands, source and executable hashes, and raw
outputs for Ashikhmin, Thin Wall, albedo construction and the Multi-GGX closure.

## Constant-folded tangent follow-up

The v4 host validator accepts `ShaderInput::constant_folded_in`, matching
`SVMCompiler::input_link`; previously it only accepted a live link. No closure
math or GPU kernel changed. The regression suite now includes anisotropic
full-coverage and half-coverage materials with a constant Combine XYZ tangent.
Both cases fail on v3 and pass on v4. All 12 cases pass in both CPU SVM and CPU
OSL; intentional unsupported-input errors still produce process status 1.
Reports: `build/tests/performance/multiggx_folded_tangent_before/report.json`,
`multiggx_folded_tangent_after/report.json`, and
`multiggx_folded_tangent_osl_after/report.json` (the latter two share the same parent).
The v4 package is `build/diffraction_delivery_20260927_folded_tangent_v4/Blender.app`;
this follow-up does not relabel the v3 gallery or its performance measurements.

The previously rejected appearance scene also rendered successfully on Metal
at 360×200, 128 fixed samples, with adaptive sampling and denoising off.
Its scene, raw EXR, PNG and finite-output report are in
`tests/output/diffraction/folded_tangent_metal_v4`.
