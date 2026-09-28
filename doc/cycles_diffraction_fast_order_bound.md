# Bounded Fast order support

## Problem and change

The dielectric Fast kernel previously returned an invalid range when the
geometric order bound reached 4096. A pitch within the exposed node range, such
as 1 mm, could therefore lose its entire diffraction closure for affected
wavelength/interface configurations. Reflection could
also perform thousands of order evaluations per sample.

The native Fast relief model now evaluates orders from -256 to +256, further
restricted by actual geometric support. Valid large pitches use this bounded
approximation instead of failing setup. Reflection order conversion is capped
before float-to-int conversion. The zero-relief single-order optimization remains.
The stale Principled tooltip saying transmission must be zero is corrected.

This changes the Fast approximation at high order counts. It is not an exact
optimization and is not presented as one. Typical CD/DVD pitches have fewer
orders and retain the same order support. The dedicated Realistic response-cache
solver is not changed by this native Fast loop bound.

## Error bound and probability treatment

The complex Fourier coefficient of a unit-amplitude binary phase profile obeys
`|c_m|^2 <= 4 / (pi*m)^2` for nonzero m. For passive interface coefficients,
including both signs and both reflection/transmission, the omitted facet power
is bounded by `16 / (pi*pi*M)` using the integral bound on the squared tail.
With M=256, this is **0.006332574**, or **0.6332574% of unit incident facet power**.
Pure transmission has half that bound.

The joint dielectric and reflective models assign residual power to order zero.
Pure Refraction keeps omitted power as null events. No direction-dependent
renormalization is introduced, and the symmetric order cutoff is the same under
path reversal. These choices preserve the implemented model's PDF accounting;
they do not reproduce the omitted physical diffraction directions.

**This is not a pixel-error or image-quality bound.** A detector isolating an
omitted high order can have a large relative error. Absolute amplitudes above
passive interface bounds require the corresponding scaling of the bound. Fast
remains approximate, and this does not certify full-wave corrugated optics.

## Numerical checks

`tests/performance/cycles_diffraction_order_tail_test.cpp` independently integrates
complex binary Fourier coefficients over all omitted propagating flat-facet
orders for 144 configurations: wavelengths 380/550/780 nm, pitches 100 micrometres
and 1 mm, both interface orientations, different depths, duties and incidences.
Maximum omitted joint power is **0.00119230794** (0.119231%); maximum omitted
pure transmission power is **0.00124159069** (0.124159%). Both satisfy the bounds.

2,030 high-pitch sample/evaluate events pass for Glass, Refraction, tinted-film
Principled and Glossy. 1,010 rough reciprocal comparisons pass, with maximum
relative error 1.44306e-5. The 562,862-event generalized-film regression was rebuilt
against the new source and passes. Logs are `order_tail_cpu_v3.log` and
`order_tail_film_regression_v1.log` in `tests/output/diffraction`.

## Bounded runtime fixture

The before/after fixture uses the same 100-micrometre-pitch Glass scene, 128×71
pixels, 32 fixed samples, seed 11, adaptive and denoising OFF. Two renders run
within each isolated process; the second is warm. This comparison measures a
declared bounded approximation, not identical rendering or equal-error speed.
The 1 mm integration cases exercise previously rejected ranges in Metal PT,
BDPT, guiding and CPU OSL. These small images are checks, not final-quality artwork.

The isolated 100 µm-pitch warm render measured 1.327325 → 0.711151 s
(**46.42% faster**). First-render times were 8.827047 → 31.830980 s, so no
cold-start improvement is claimed. The corresponding reports and executable
hashes are retained in `large_pitch_comparison_v1.json`. Only this fixture was
benchmarked; the facet-energy bound is not an equal-image-error certification.

All five standalone cases passed and were visually inspected. Warm 1 mm times:
PT 0.6631 s, BDPT 3.1763 s, guided 1.4322 s, CPU OSL 5.1939 s (32 samples,
128×71). First-render preparation is recorded separately in the manifest. These
are not equal-quality method rankings. A fresh 1024-sample CD/DVD render also
passed on this package. SHA-256: `9e66f2c9970be3f87e5ecf1247f71244efd3119fd50fb9bed2709b84fb8c67b8`.
