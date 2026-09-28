# Tinted and dispersive Principled transmitting film

Principled GGX transmission diffraction now combines Specular Tint, wavelength-
dependent IOR and thin film. Film thickness, film IOR and tint may be linked.
The earlier white-tint/zero-dispersion film restriction is removed. Thin Wall
and multi-scattering diffraction remain explicit unsupported combinations.

## Model and implementation

The film response uses the shared lossless dielectric Airy kernel at the sampled
vacuum wavelength. Per-order endpoint powers retain the reciprocal construction
used by the film-free model. Native Principled's artistic film-tint scaling is
applied to that scalar spectral response. Reflectance is bounded to [0, 1] so
extreme artistic tint cannot create negative transmission or non-passive order
probabilities. This bound can differ from unbounded native output for extreme
tint inputs; exact agreement with those inputs is not claimed.

When physical F0 is at most 1e-5, native Principled disables its film-tint
correction. Those cases use the existing coated Glass kernel. This handles the
dispersive matched-index limit with its direction-dependent straight atom rather
than incorrectly using a constant tinted atom. The existing bounded atom
quadrature approximation is retained.

Two film floats fit into the generalized payload's alignment padding. Its size
remains 96 bytes; ordinary tint and coating payloads remain 64 and 80 bytes.
The physical white-tint/no-dispersion path and film-free paths remain available.
No physical-response cache is constructed by this Fast material path.

## Numerical evidence

`principled_generalized_film_cpu_v2.log` records zero failures:

- 562,862 accepted sample/evaluate events, smooth and rough, front and back.
- 370,862 reciprocal comparisons, maximum relative error 6.69807e-5.
- 186,328 nonsingular flat-limit comparisons against independent double-precision
  complex Airy amplitudes and GGX geometry, maximum relative error 9.20453e-6.
- Smooth flat-limit maximum absolute power error 8.28068e-8.
- 324 facet configurations, including extreme tint: individual powers in [0, 1]
  and total power within 2e-5 of one.
- 3,974 matched-index atom/reflection dispatch events and closure-capacity checks.

The film-free dispersion regression was rebuilt against the new source and
passed all 562,390 events. Its prior native-float disagreements remain recorded;
the independent double-reference accuracy gate still passes.

These are validations of a scalar local relief approximation, not proof of
full-wave corrugated-film accuracy or coherence between separate objects.

## Render fixture

The saved scene combines tinted/dispersive transmission with 420 nm smooth film,
260 nm rough film and 640 nm rough film at half diffraction coverage. The
standalone package runner records Metal PT/BDPT/guiding and CPU OSL separately,
with fixed samples, adaptive sampling OFF, denoising OFF and no brightness
normalization. See `tests/output/diffraction/principled_generalized_film_delivery_v1`.

All four packaged renders passed and were visually inspected: pt 256 spp: 37.30 s, bdpt 128 spp: 117.66 s, guided 128 spp: 104.90 s, osl 32 spp: 8.99 s.
All linear pixels are finite; visible noise remains. Fourteen support-validation
assertions passed (the deliberate unsupported-shader process exits 1). Package
SHA-256: `8b383bfaab2e82bb5136b3dcbb985decef983b906516816d2f57217ae40cc0cc`.
