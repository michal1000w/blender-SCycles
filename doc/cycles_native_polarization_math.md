# Native Glass polarizer transport

The Glass **Polarizer** checkbox enables a linear transmission analyzer. Its angle is in radians in the object's local XY plane; the transformed axis is projected into each ray's transverse plane. Reflection remains the substrate's reflection. Transmission uses the reciprocal Jones sandwich `P_out T P_in`; an IOR-one film therefore applies an idempotent projector, including its back face. Two filters obey Malus' law, and an intermediate filter can transmit between crossed filters.

When all polarizer checkboxes are disabled and unlinked, the polarization kernel feature is absent: no Stokes path buffers or BDPT polarization cache are allocated, Glass keeps its original one-slot Fresnel extra, and native sampling/evaluation follows the original branch. Enabled Glass allocates the two-slot optional Fresnel suffix; its node reserves the additional slot. Metal source/cache identities include the feature so switching it cannot reuse an incompatible specialization.

Native PT and BDPT transport four Stokes components separately for every Spectrum channel. Each component uses the native scalar throughput, multiplied by a normalized polarization state. Camera paths carry adjoint sensitivity in the physical incoming-light frame (opposite the traced camera ray); light subpaths carry forward radiance. Connections contract the adjoint and forward states. Emitters are unpolarized. Proposals, geometric factors and MIS densities retain the native estimator; polarization uses the complete closure mixture evaluated for that same measure. When polarization is enabled, singular Glass and Transparent mixtures use discrete atomic evaluation; exactly matched-index rough Glass transmission is a true null atom. Rough, nonexact IOR values in the native 1e-4 matched-index window retain continuous sampling when polarization is enabled, with a rationalized Snell denominator; continuous lobes cannot be added to a singular sample. Disabled materials retain the original sampling path.

Dielectric single scattering uses complex s/p Fresnel amplitudes and an incidence-plane basis, including total internal reflection. Physical Glass F0 is supported. RGB closure weights and mixtures remain per channel. Native GGX energy-compensation returns, diffraction rough/return lobes, custom tinted Fresnel remapping, thin-film substrate polarization, diffuse, BSSRDF, volume phase and other unsupported closure families use an explicit **depolarizing Fast approximation**, rather than a claim of exact polarized multiple scattering. Their scalar native energy is retained. Enabled analyzers still act on those modeled transmission lobes. Straight grating transmission and plain Transparent preserve polarization. This model describes geometric-optics polarization, not phase coherence between native paths.

MNEE composes the normalized Mueller maps at its solved refractive interfaces in receiver-to-light order. The camera contracts the transpose chain; BDPT sensor manifolds compose it before the cached light-vertex contraction. This preserves the solver's Jacobian and sampling density. Cached light polarization is rotated to the decoded native direction frame. The declared coherent Lambertian detector reradiates unpolarized intensity; camera-side analyzers contract that intensity with the camera sensitivity's intensity component, while source-to-detector branches retain their complex Jones fields.

Rough matched-index thin-film filters require integrated microfacet transmission mass and are explicitly rejected; a linked IOR cannot certify separation from that unsupported case. Smooth films or a constant IOR different from one remain available.

Photon Mapping is rejected with enabled polarizers because its scalar cached photons do not retain polarization. Native polarized mixed-closure diffuse/glossy/transmission light passes currently use the aggregate mixture's radiometric correction; combined radiance is the estimator's output, while detailed per-lobe polarized pass attribution is an approximation. Guiding remains a proposal mechanism and does not replace the polarization state.

Validation must include actual native PT, BDPT and guiding images (coherence disabled), ideal films, multiple crossed/rotated filters, a closed slab, and dielectric glare/Brewster cases before claiming photographic acceptance. Numerical Jones/Mueller tests alone establish the transport algebra, not renderer integration.

## Renderer evidence, 2026-09-28

The v41 native candidate was **rejected**. Metal PT and CPU SVM rendered enabled filters black, whereas CPU OSL passed the ideal-film suite. The preserved [v41 report](../build/tests/python/native_polarizer_render_v41/report.json) records the failure; its images and logs were not replaced. The cause was SVM shader-stage eligibility: adding the global polarization feature to the Glass node's feature set caused its entire closure instruction to be omitted. The corrected compiler excludes that global transport bit from shader-stage eligibility while retaining it in the scene's kernel features.

The fixed, immutable **v42** candidate passes all 26 completed actual Metal PT and BDPT numeric cases in the [v42 report](../build/tests/python/native_polarizer_render_v42/report.json), with the gates declared before GPU testing unchanged. These cases include ideal films, rotated and crossed filters, the three-filter sequence, an idempotent closed slab, and persistent OFF/ON/OFF renders. The [CPU SVM report](../build/tests/python/native_polarizer_svm_v42/report.json) and [CPU OSL report](../build/tests/python/native_polarizer_osl_v42/report.json) each pass 13 cases, with worst RMSE 1.19209e-7. The [configuration guard report](../build/tests/python/native_polarizer_guards_v42/report.json) also passes: unsupported Photon Mapping and rough matched-index thin-film configurations reject explicitly; a constant linked IOR of 1.5 is safely folded and accepted.

At this checkpoint, guiding, photographic glare/Brewster renders, and coherent source-to-detector plus camera-analyzer regressions remain pending. The completed ideal-film gates establish native PT/BDPT integration for those tested paths; they do not establish exact polarized multiple scattering, arbitrary-material polarization, or completion of the remaining general curved coherence work.

Additional bounded shader integration (v42 CPU SVM PT): actual closed normal-incidence
n=1.5 Glass gives 0.519999921 against the independent 0.04 + 0.96/2 = 0.52
reference. First reflection leaves unpolarized World radiance unchanged; after first
transmission the common material projector is idempotent through later interfaces.
The predeclared Bernoulli variance is 0.0096, with 6-sigma image-mean and 2-sigma
pixel-RMS gates at 1024 samples. A closed triangular prism uses single front/back
primitives; all 4096 pixels are tested. The original cube control lost exactly two
samples along its triangulated faces and failed its strict deterministic gate;
that report is preserved, and this test does not claim to repair the legacy seam.

Glass Diffraction Weight 1 with Depth 0 gives exactly 0.5, but the host folds zero
relief back to ordinary Glass, so this alone does not exercise grating metadata.
The positive default relief (150 nm, pitch 1600 nm, duty 0.5) at exact matched IOR 1
has zero optical contrast and exercises the actual straight grating atom plus
polarizer payload. Its CPU means are (0.5000338, 0.5000251, 0.4997978). An independent
200000-point inverse-CDF integration of the existing CIE/D65 sensor tables predicts
(0.50003339, 0.50002478, 0.49979772); the physical target remains 0.5, without
normalization. Gates use the independently computed RGB spectral second moments
and a predeclared 0.002 table/systematic margin. This is identity-payload plumbing,
not a claim about nonzero-index-contrast diffraction polarization.
Evidence: `build/tests/python/native_polarizer_shader_prism_cpu_v42/report.json`;
original failed cube control: `build/tests/python/native_polarizer_shader_cpu_v42/report.json`.
CPU BDPT is explicitly skipped because its native implementation requires Metal.
The same runner prepares PT/BDPT Metal validation; no Metal result is asserted here.

The subsequent actual v42 Metal positive-relief identity test rejected the GPU
joint-path result: PT and BDPT returned approximately 0.58 and 0.577 instead of
0.5. CPU agreement and earlier basic filter results do not supersede that failure.
The reports, generic-only reproduction, and diagnostic source overrides remain
preserved. A temporary diagnostic proved every path hit the film and carried the
metadata; the straight atom's normal-side inference could omit its projector.
The fix classifies straight and coated-straight atoms as transmission by their
native event contract (always `wo=-wi`, `LABEL_TRANSMIT`), rather than inferring
that event again from the stored shading normal. Rough carrier reflection still
uses its ordinary side classification. Actual Metal with a separate fixed-source
override gives ON means (0.500033464, 0.500024769, 0.499797447) and restores the
unfiltered sensor expectation under OFF/ON/OFF. Final packaged v43 validation is
pending; this source-override proof is not substituted for that release check.
