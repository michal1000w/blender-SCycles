# Diffraction feature and image suite — 27 September 2026

The currently selected package and its matched performance/check results are in
[the zero-depth-fix delivery report](cycles_diffraction_flat_native_delivery.md).
The feature suite and presentation images below retain their final_v3 provenance.

The application used for this full suite is
`build/diffraction_delivery_20260927_final_v3/Blender.app`, executable SHA-256
`70b855e0c640a9b8ab39455e6d41174ec1e6127f0490352978e2f27d218a4df4`.
It is separate from the installed Blender. The adjacent resource manifest
records the matching CPU executable, Metal libraries, kernel headers, OSL
shaders and Python addon; all 689 recorded resources passed hash verification.
Subsequent CMake dependency-list changes do not alter
this packaged renderer's code or resources.

Choose **Fast** diffraction for normal use. It is the practical approximate
default and does not construct a response cache. **Realistic** remains an
optional experimental Metal/MPS solver; its startup cost and incomplete
arbitrary-profile validation make it unsuitable as the default. This selection
does not imply equal-quality speed superiority on every scene.

This is a runnable, tested development delivery, **not completion of the entire
original production scope**. General coherent reflected/refracted multipath,
multi-scattering GGX diffraction, exhaustive pipeline certification and broad
Realistic-solver validation remain open. The direct coherence option does not
change RGB light emission into a spectral monochromatic source. Very rough Thin
Wall diffraction can darken substantially, and difficult indirect scenes remain
noisy. These limitations are not hidden by normalizing brightness or substituting
reference patterns into materials.

## Implemented behavior

- Dedicated spectral Diffraction BSDF, Fast default and optional GPU-built
  Realistic response cache.
- Native Glass, Glossy, Metallic, Principled and Refraction diffraction in their
  supported distributions using the Fast scalar model. The Realistic response
  cache is an option on the dedicated node. The delivered extensions include Glossy
  Ashikhmin-Shirley and Principled transmitting Thin Wall, with SVM and OSL
  dispatch. Choose GGX for Principled; enabled Multi-GGX diffraction reports an
  explicit error.
- Existing supported combinations include tint, conductor response, thin film,
  dispersion, anisotropy, tangent controls and partial coverage. Flat relief
  uses native carriers where specified; zero coverage retains ordinary branches.
- Optional **Coherent Direct Point Sources (Experimental)** light controls:
  source groups, phase, phase wavelength and Gaussian coherence length. The
  scalar receiver model handles up to 16 matched constant point lights and
  opaque visibility. It supports the camera-path estimator in PT, BDPT and
  guiding; it does not propagate fields through reflected paths.
- Clear host rejection of unsupported coherent configurations, including changes
  to a receiver shader between renders. Coherence length zero restores ordinary
  light-subpath sampling. Peer-source shadow visibility uses the shadow-catcher
  path mask.
- Corrected dielectric denoising albedo masks, separate reflection/transmission
  estimates and straight-atom bookkeeping. This remains a fast smooth-interface
  albedo approximation, not exact rough grating integration.
- Bounded Fast diffraction order evaluation, preserving typical CD/DVD orders;
  the declared passive binary-tail facet-power bound is 0.6333%, not a pixel-error
  guarantee.

## Numerical and support checks

All GPU checks ran on the Apple M5 Metal device outside the sandbox. Performance
and raw acceptance renders use fixed samples, adaptive sampling OFF and
denoising OFF. BDPT brightness is never adjusted to agree with PT.

- Final direct coherence: **27/27** absolute-radiance checks, nine scenes each
  under PT, BDPT and guiding, at 512x256 and 128 fixed samples. Cases cover phase
  zero/pi, incoherence, partial coherence, 633 nm phase wavelength, an opaque
  source mask, three sources, scene-unit scaling and disabled coherence.
  Two-source maximum RMSE is below 0.0001; three-source RMSE is about 0.0057051.
  The predeclared gate is RMSE < 0.006 and absolute mean error < 0.003. The oracle
  is independent float64 scalar Lambertian irradiance with subpixel quadrature;
  it has no fitted brightness scale. Off versus zero-length PT/BDPT control
  differences are at most 1.1921e-7 in all three transports.
- Native Ashikhmin closure: **162,458** accepted continuous samples plus singular
  tests, zero failures; native-flat maximum absolute difference 3.05176e-5,
  reciprocity difference 6.10352e-5, sampled furnace maximum 0.930342.
- Native Thin Wall closure: **36** actual native-carrier comparisons, maximum
  relative error 0.00013405085. A focused nine-point furnace sweep measured
  maximum 0.980753; very rough strong-relief sheets reached only 0.69–0.74.
  This is a measured conservative approximation, not a global passivity proof.
- Thin Wall rendered with finite output on Metal PT/BDPT/guiding and CPU OSL
  at 480x269, 128 fixed samples. Native flat versus uncovered images agree
  within 9.53675e-7 per linear RGB channel; relief changes the image by RMSE
  0.232475. These are image controls and transport smokes, not convergence proof.
- Final raw feature suite: **13/13 processes, 37 rendered images** passed,
  covering the 27 analytic coherence renders and ten material/control renders.
  Thin Wall and Ashikhmin both render on Metal PT/BDPT/guiding and CPU OSL.
  The finite-output shader smokes were also visually inspected; they are not
  treated as independent physical references. Ashikhmin BDPT is slightly
  brighter than PT in this fixture; this was not normalized away or used alone
  to mark either result physically correct or incorrect.
- Final CPU host validation: **9** coherence cases, **4** Glossy cases and
  **15** Principled cases passed. Photon-mapping rejection is not exercised by
  this CPU suite because photon mapping is a Metal-only path.
- Dielectric albedo acceptance passed, including **32** mask/distribution/query
  combinations and active Fresnel/film/straight-atom controls.
- CPU, OSL and all three Metal library variants compile. The new headers are
  registered as build dependencies. `git diff --check` passed.

Durable commands, hashes, stdout and exit codes are recorded in
`build/tests/performance/native_extensions_final_v3/results.json`,
`build/tests/performance/cycles_diffraction_albedo_final_v2/results.json`, and
the `build/tests/python/*final_v3*` reports. Exact render provenance is in
`tests/output/diffraction/final_features_v3/manifest.json` and its per-scene
manifests. Older broad numerical and pipeline evidence retains its original
binary identity in [the historical report](cycles_diffraction_delivery_report_20260927.md).

## Performance interpretation

A bounded matched CD/DVD benchmark used the same original scene and script on
the final and previous albedo packages: Apple M5 (10 GPU cores), Metal PT,
512x369, 512 fixed samples, adaptive sampling OFF, denoising OFF, persistent
scene data, one warmup and three measured renders per package. Timing includes
EXR writing. Final measured times were **1.6163, 1.6142, 1.6184 s**, median
**1.6163 s**; previous median was **1.5056 s**. The final build is **7.35% slower**
in this fixture, so it is not an unqualified speed improvement over the previous
package. Warmups were 45.44 and 36.26 s respectively. Images agree to RMSE
**5.24004e-8**, maximum absolute difference **3.81470e-6**, with no normalization.
The new build is selected for its expanded supported feature set and Fast mode,
not because it won every timing. The previous package lacks the new Thin Wall,
Ashikhmin and direct-coherence support. This one fixture does not establish
performance on every material or transport.

Exact reports, raw pixels and provenance are in
`tests/output/diffraction/benchmark_final_v3` and
`tests/output/diffraction/benchmark_previous_albedo_v3_control`.

On the final tiny direct-source PT fixture, warm renders took approximately
0.66–0.72 s with interference, versus 0.65 s disabled. These are individual
fixture observations, not a repeated or equal-quality speed benchmark. Opaque
source masking took 1.77 s. Initial pipeline preparation took 45.92 s; changing
to three sources caused another 38.60 s including preparation. BDPT and guiding
also incur substantial first-use Metal compilation. This is distinct from the
Realistic diffraction response cache; Fast builds no such cache.

Earlier fixed-sample tuning measured a high-pitch warm render improvement from
1.327 s to 0.711 s, and a flat-profile case from 6.732 s to 4.739 s. Those results
are fixture-specific historical measurements, not new final-package timings.
Enabled diffraction can still cost materially more than an ordinary BSDF; a
previous glass fixture measured about 60% overhead. There is no universal
zero-overhead claim.

The experimental Realistic Metal/MPS cache improved its measured six-run median
from 148.523 s to 124.639 s, but a conductor cache case still timed out at 1800 s.
That failed case is preserved. It is why Realistic is not the selected default.

## Render suite and editable scenes

The final gallery is `tests/output/diffraction/presentation_final_v3/index.html`.
Its **19 editable scene copies** are in `presentation_final_v3/scenes`: ten
material scenes and nine direct-coherence controls. They write new renders to
separate relative folders instead of overwriting acceptance evidence. Original
test scenes and linear EXRs remain available beside their manifests.

All ten material previews were rendered by the selected binary, visually
inspected, and checked across **70 finite pixel outputs** (beauty, noisy image
and denoising auxiliaries). Material previews use OIDN; the raw feature suite
and performance measurements do not. Denoising can smooth fine spectral detail.
The depth statistics were recomputed from unchanged EXRs with float64 sums,
because finite background-depth sentinels can overflow float32 reductions.

These are inclusive presentation times, including preparation, image writing
and denoising—not a transport speed comparison:

| Scene | Transport / fixed samples | Seconds |
|---|---|---:|
| cd_dvd | PT / 512 | 60.52 |
| glass | PT / 512 | 54.01 |
| glossy | PT / 512 | 6.40 |
| metallic | PT / 512 | 6.98 |
| principled_film | PT / 512 | 12.19 |
| refraction | PT / 512 | 50.99 |
| covered_disc | PT / 512 | 91.85 |
| indirect | BDPT / 2048 | 366.88 |
| thin_wall | PT / 512 | 50.64 |
| ashikhmin | PT / 512 | 41.03 |

The indirect BDPT preview still has scattered artifacts and has not met a
convergence/physical-reference quality gate. Its 366.88 s is slower than the
older package's 297.68 s presentation run; those single runs include preparation
and do not isolate a rendering-speed regression. The slower result is retained,
not replaced with the older image. No brightness normalization was applied.

## Model references and boundaries

See [direct coherence](cycles_coherent_direct.md),
[Thin Wall](cycles_diffraction_thin_wall_candidate.md),
[Ashikhmin-Shirley](cycles_diffraction_ashikhmin_integration.md), and
[order truncation](cycles_diffraction_fast_order_bound.md) for the implemented
models and limitations. The [retrospective](cycles_diffraction_retrospective.md)
preserves the distinction between a brighter, more complete BDPT result and a
physically incorrect result; PT is not treated as an absolute ground truth.

The research includes [Wave Tracing: Generalizing the Path Integral to Wave
Optics](https://www.shlomisteinberg.com/2025/08/28/generalizing_the_path_integral/)
and [A Generalized Ray Formulation for Wave-Optics
Rendering](https://shlomisteinberg.com/2023/03/27/rtplt/). Their general wave
transport formulations are not implemented by the direct-source approximation.
No external wave-tracer code was copied.

## Allowance and completion status

The observed account usage moved from the new allowance checkpoint of 43% to
46%, below the requested 53% ceiling. No further benchmark sweep was launched.
The supported build and its scene/image suite are delivered; the original
general-coherence/Multi-GGX/full-certification goal is not marked complete.
