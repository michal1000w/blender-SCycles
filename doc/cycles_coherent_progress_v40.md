# Coherent optics evidence through v40

The v39 mixed sphere/finite-mirror and v40 internal sphere families passed actual Metal rendering against independent numerical fields. The linked gallery shows raw actual images alongside their references with one fixed display exposure per feature. It uses the predeclared 26×26 ROI from 32×32 detector crops. The actual workers used 128 samples, seed 19, adaptive sampling off and denoising off.

- [Actual/reference contact sheet](../tests/output/diffraction/coherent_progress_gallery_v40/coherent_actual_reference_contact.png)
- [Interactive evidence gallery and editable scene links](../tests/output/diffraction/coherent_progress_gallery_v40/index.html)
- [Raw EXR hashes, reference versions and metrics](../tests/output/diffraction/coherent_progress_gallery_v40/gallery_source_manifest.json)

## Validated optical families

All-planar reflection/refraction paths retain their existing exact stationary connector. The new sphere support includes an exterior convex reflection R, complete isolated two-transmission TT branches, and T–R–T / T–R–R–T internal sphere histories. A contiguous sphere block can be bracketed by planar mirror reflections within the four-event budget. Plane reflections are unfolded by isometry; the sphere roots, spreading and Morse phase are inherited, then physical hit points and visibility are reconstructed.

The v39 physical fixtures contain a finite planar mirror and a native analytic sphere. Their independent inventories sum all visible supported direct and alternate paths. The TT fixture includes one direct TT branch plus three mirror-prefix TT branches per source. The v40 native Glass sphere fixtures include direct and exterior Fresnel R, two TRT branches, and an additional TRRT branch when the four-event cap permits it. Their geometry has no isolated fictitious contribution or PT-fitted brightness.

The independent references use the exact stored Blender source positions and power, dielectric flux-normalized Jones factors, three unpolarized world-axis dipole modes, full vector overlap, OPL, Gaussian coherence and Hessian-derived Morse phase. Every source is prospectively declared at 1 mW; the stored Blender value is `0.0010000000474974513 W`. Pixel fields use 2×2 BOX quadrature with dispersed 4×4 checks.

## Actual raw-image results

| Feature | Maximum actual/reference case RMSE | Reference version |
|---|---:|---|
| Mixed sphere R | 0.000202443 | Corrected ideal Mirror Jones basis |
| Mixed sphere TT | 0.0000991195 | Corrected ideal Mirror Jones basis |
| Sphere TRT | 0.000000458 | All-winding Fresnel/Jones/Morse |
| Sphere TRRT | 0.000000458 | All-winding Fresnel/Jones/Morse |

The v39 phase-difference RMSE is `0.000396025` for R and `0.000196815` for TT. All six image cases and both phase checks pass the unchanged absolute radiance gates. [The full reassessment](../build/tests/python/coherent_mixed_sphere_render_v39/corrected_Jones_full_reassessment.json) reloads the same immutable raw EXRs and proves exact agreement with archived worker arrays. It records both independent reference hashes and every raw EXR hash.

The original v39 Python ideal-mirror reference omitted a polarization-basis sign. For the declared Householder convention `H = I − 2nnᵀ`, `Hs = s` and `H p_in = −p_out`, so the ideal outgoing p coefficient is −1. This was corrected algebraically without using rendered-image values. Original references and worker results remain preserved. [Correction provenance](../tests/output/diffraction/coherent_sphere_planar_v39_1mw_jones/reference_correction_provenance.json) records unchanged geometry/power and both reference versions. The maximum independent pixel change was `1.61e−6` for R and `5.14e−5` for TT.

## Internal-path presence checks

The ordinary absolute image gates could miss a weak new internal reflection. Independent omission controls and stricter feature gates were therefore declared before the internal GPU batch. Omitting all TRT/TRRT fields produces a TRT phase-π RMSE of `0.00151981`, which fails the declared `0.0004` mean gate.

A matched cap-four minus cap-three response isolates TRRT while preserving identical geometry, source power, camera and seed. The independent phase-π response mean is `5.893992809e−5`; a missing TRRT family predicts zero and fails the predefined `1e−5` mean and `1.5e−5` response-RMSE limits. Actual Metal response RMSE is `8.29e−10`, with mean error `3.13e−11`. The distinct-group paired response also passes, with RMSE `1.15e−8`. [Feature-presence evidence](../build/tests/python/coherent_sphere_internal_render_v40/feature_presence.json) retains the raw hashes, [pre-GPU gates](../tests/output/diffraction/coherent_sphere_internal_v40_1mw/pre_gpu_feature_gates.json), and independent omission controls.

CPU white-emission diagnostics verified every detector-crop pixel sees the intended detector. Cameras stand 2–3 mm from their planes with a 1 mm near clip. The dark internal phase-π image is therefore a predicted interference result.

## Editable physical views and staged polarizers

[Wide R/TT physical preview scenes](../tests/output/diffraction/coherent_sphere_planar_v39_1mw/publication_preview_manifest.json) are authored at 640×480, 256 samples and denoising on. They contain labels/source markers and separate noncoherent area/background/ground illumination. These ordinary illustrations are separate from detector acceptance references.

Linear-polarizer work is staged in the Glass node: a checkbox and angle in radians select the object-local X/Y pass axis. The independent Jones oracle verifies a single unpolarized filter transmits 50%, parallel filters retain 50%, crossed filters transmit zero, and a 45° middle filter between crossed filters yields 12.5%. At the n=1.5 Brewster angle `56.309932°`, a p-axis analyzer removes ideal central-ray glare.

Editable [coherent Malus/glare geometry](../tests/output/diffraction/coherent_polarizer_pending_v41c/manifest.json) and [native ordinary photographic scenes](../tests/output/diffraction/native_polarizer_pending_v41b/manifest.json) are explicitly marked `pending_shader_api`. The native suite uses white World or visible area emission with coherence off; it includes a closed IOR1 slab to test front/back axis idempotence and a Glass slab over a colored checker floor. The finite-area photograph is qualitative; it is not equated to an ideal central-ray number. These pending scenes require reauthoring with the feature-enabled binary and have not passed polarizer GPU acceptance.

## Mathematical boundaries

Root completeness is established for the isolated analytic sphere inventories and the declared planar-isometry extension. Grazing boundaries, singular folds and axial rings are explicit rejected states. Arbitrary smooth objects, multiple sphere blocks and mixtures involving refracting planar interfaces around the curved block still need independent stationary-connection and visibility validation.

## Native film fixture diagnostics

The clean native Malus fixtures use finite single-triangle films with every tested ray far from an edge. Minimum light and transparent bounce counts are set to eight before GPU testing, removing Russian roulette from the three-filter and two-face slab paths. Legacy CPU unfiltered World1 diagnostics return exactly1 at all4096pixels for these films and the closed slab.

A separate centered quad IOR1 probe retains a preexisting shared-triangle-edge failure: exactly64 diagonal pixels are black at one sample. Factory-created stock Blender5.1.2 and v38b give identical results for one,two and three quads; shifting the quad or using a single triangle gives exactWorld1. The failed quad scenes, actual EXRs and coordinate records remain archived in `native_ior1_seam_stock51_retry` and `native_ior1_seam_v38b`. The seam is outside this polarizer scope; the analytic tests do not claim every native mesh edge is fixed.

The [unfiltered native photographic layout preview](../tests/output/diffraction/native_polarizer_pending_v41b/native_brewster_photo/legacy_unfiltered_layout_preview.png) shows the real Glass slab, colored checker floor and finite area-source reflection. It verifies scene layout only; its legacy shader has no active polarizer.

## Actual coherent light-pass accounting

The [v40 Metal light-pass report](../build/tests/python/coherent_light_pass_render_v40/report.json) passes both primary and secondary cases. Enabling light passes preserves same-seed Combined within 5.961e-8; native Color recomposition differs by at most 9.127e-8. Signed coherent contributions remain represented in the native pass accounting. Raw EXRs and binary, script and scene hashes are retained in the report.
