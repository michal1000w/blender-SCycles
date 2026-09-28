# Cycles diffraction delivery — September 27, 2026

The delivered artifact is a runnable Apple Silicon development build. The original
request is **not fully implemented or production-certified**. General coherent
interference between separate objects is absent. This report distinguishes the
usable local diffraction implementation from that unfinished scope.

## Latest Glossy correction

The latest package is `build/diffraction_delivery_20260927_glossy/Blender.app`.
It also fixes the old Glossy path that silently used GGX for Beckmann
diffraction. Native GGX/Beckmann, anisotropy, coverage and medium wavelength
scaling now use the shared reflection closure. CPU, Metal and OSL checks pass;
see the [Glossy report](cycles_diffraction_glossy_integration.md) and
[gallery](../tests/output/diffraction/glossy_delivery_v1/index.html).
Other requested material and coherent-transport gaps remain unfinished.

## Updated pipeline package

Use `build/diffraction_delivery_20260927_pipeline/Blender.app`. It fixes the GPU
material-function parameter limit exposed by compositor initialization. Twelve
additional Metal pipeline fixtures and ordinary-material Eevee rendering pass.
See the [pipeline review](cycles_diffraction_pipeline_validation.md) and
[gallery](../tests/output/diffraction/pipeline_delivery_v3_review/index.html).
The original package and all measurements below retain their recorded provenance.
This update does not complete the remaining original scope.

## Original delivery artifacts

- App: `build/diffraction_delivery_20260927/Blender.app` (about 970 MB).
- [Render gallery and saved scenes](../tests/output/diffraction/delivery_20260927/index.html).
- [Material usage guide](cycles_diffraction_delivery.md).
- [72-render dedicated-node suite](../tests/output/diffraction/fast_delivery_suite_v4_review/index.html).

The app was installed to a separate directory with CMake. The existing installed
application was not replaced. It includes matching libraries, Python scripts,
OSL shaders, Metal libraries and kernel sources. The executable SHA256 is
`fded74d7efff270fac6c0d4b5249529c25c61b047d3eae95795e0c6f344dbb5e`.
The underlying modified Blender tree identifies itself as 5.3.0 Alpha.

## Selected implementation

**Fast is the practical default.** It evaluates wavelength-dependent local
relief diffraction without building a response cache. Reciprocal order powers,
residual mirror power, inverse-root PDFs, spectral sampling and transport/MIS
integration provide the working approximate implementation. It is not a Maxwell
solution and has no negligible-overhead guarantee.

The dedicated Diffraction BSDF also offers **Realistic**, backed by Metal/MPS
response-cache construction. It remains experimental. It is not selected for
ordinary use: full-domain preparation can be expensive and broad modal
convergence is unproven. A UI switch cannot supply cross-object coherence;
that requires additional coherent transport, which has not been implemented.

| Material path | Implemented support | Limits |
|---|---|---|
| Dedicated Diffraction BSDF | Spectral local reflection/transmission; Fast/Realistic; embedded wavelength-dependent conductor tables | Smooth profile; Realistic experimental |
| Glass | GGX/Beckmann reflection and transmission, thin film, partial coverage, tangent, SVM/OSL | No Multi-GGX diffraction |
| Metallic | GGX/Beckmann, native conductor/F82 Fresnel, thin film, coverage and groove orientation, SVM/OSL | No Multi-GGX; no explicit buried-medium control |
| Principled | GGX metallic and base dielectric reflection, partial coverage, film/tint and existing coat attenuation, SVM/OSL | Transmission must be zero; no multi-scattering diffraction |
| Glossy | GGX/Beckmann, anisotropy, rotation, partial coverage, explicit medium IOR, SVM/OSL | No Multi-GGX or Ashikhmin diffraction |
| Other nodes / combinations | Existing behavior retained | Refraction integration and remaining combinations are unfinished |

For the newly extended material nodes, zero Diffraction Weight retains the
ordinary material. Unsupported enabled combinations report errors. Eevee does
not implement these diffraction models. The Realistic option belongs to the
dedicated node; it is not a quality switch on every native material node.

## Validation delivered

- Full Blender build, including CPU, OSL and Metal kernels, passes.
- 72 dedicated-node jobs: 18 scenes across PT, BDPT, guiding and BDPT+guiding;
  all completed at 1024 fixed samples, adaptive sampling and denoising OFF.
  Eight analytic controls pass; maximum RGB-mean error is 0.000405366.
  Angular partition discrepancy is at most 0.000117127 in the recorded seed.
  Fast/reference comparisons retain approximation differences rather than
  treating Fast as electromagnetically exact.
- Shared reflective closure regression: 47,190 accepted events, including full
  and half coverage. Maximum flat-profile scaled BSDF error 6.44633e-6;
  reciprocity error 2.92063e-6. These are targeted tests, not a full energy proof.
- Coated dielectric Metal surface-MIS checks pass with zero same-direction
  CPU/Metal PDF differences and zero same-GPU reevaluation failures. The known
  differing-direction grazing/fold conditioning issues remain documented.
- Separate Glass, Metallic and Principled PT/BDPT/guiding/OSL scene families
  exercise their new inputs. The final Principled tangent correction is tested
  on the delivered binary in all four configurations. Ordinary-material and
  explicit unsupported-setting checks pass.
- The separately packaged application passes a fresh CD/DVD Metal render with
  developer resource overrides removed. Package OSL also passes at 64 fixed samples (11.81 s), with finite
  pixels and a visually inspected preview. Binary, three node OSL shaders,
  the new reflection kernel and gallery links were verified.

The 72-scene suite predates later node integration changes and retains its own
frozen binary provenance. New integration tests are separately identified;
historical renders are not represented as tests of the final executable.
BDPT images were not normalized to PT or rejected simply for being brighter.
See the [retrospective](cycles_diffraction_retrospective.md).

## Measured performance

Apple M5, 10 GPU cores. Fixed samples, adaptive sampling OFF. These figures
have different purposes and are not interchangeable equal-variance benchmarks.

| Measurement | Result | Interpretation |
|---|---:|---|
| Packaged CD/DVD, Metal PT, 960×691, 1024 samples | 37.63 s | One final render including preparation; not warmed steady-state timing |
| Final Principled PT, 720×400, 256 samples | 11.19 s | Integration timing |
| Final Principled BDPT, same resolution/samples | 52.73 s | Integration timing; no quality-equivalent speed claim |
| Final Principled guiding, same resolution/samples | 22.47 s | Integration timing |
| Glass on/off pilot, 256 samples | Last enabled 3.91 s; disabled mean 2.44 s | About 60.3% overhead in this scene; first enabled was 11.48 s, so startup variability matters |
| Realistic dielectric cache optimization | Median 148.52 → 124.64 s | 16.08% less cache time in six isolated runs; total 187.37 → 169.41 s |

The dedicated Fast path avoids Realistic cache startup entirely. The last row
is a workload-specific backend improvement, not evidence that Realistic is
ultra-fast. A conductor cache attempt exceeded 1800 seconds without an image;
its failure record remains preserved. No final fastest-correct claim is made
for the entire original feature request.

## Remaining limitations

General cross-object coherent interference, transmitting/dispersion/thin-wall
Principled diffraction, multi-scattering diffraction, Refraction-node support,
and complete motion/AOV/volume/material pipeline validation remain unfinished.
Realistic conductor/profile convergence is not established. Indirect appearance
renders remain noisy; the gallery labels them accordingly. Native reflective
node albedo/layering approximations and air-side wavelength assumptions also
limit the current model.

The build and scenes are usable for the explicitly supported local-grating
cases, including CD/DVD appearance. They do not justify claiming the requested
complete, ready production renderer. The original goal remains incomplete.
