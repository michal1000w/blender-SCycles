# Previous delivery: coherence specialization

This continuation follows a progress turn: the previous turn delivered the
final_v3 package, numerical checks and a render suite, but did not complete
general coherent transport or Multi-GGX diffraction. Those requirements remain
part of the active goal.

The current selection is documented in [the additive transport report](cycles_coherent_additive_transport.md).
This previous package and its measurements are retained unchanged.

The application tested here is
`build/diffraction_delivery_20260927_coherence_specialization/Blender.app`,
SHA-256 `c0aab5c1d728dc44ff945387059e5a3f6c21333eda54d8d6772e991de7ec5ed8`.
It adds `KERNEL_FEATURE_COHERENT_DIRECT`, set by the scene when any light has a
positive coherence group and length. Disabled lights are conservatively
included because enabled-light classification occurs later. This prevents an
active group from being accidentally compiled out. The feature guards the
joint direct evaluation and coherent light-subpath exclusion. Metal can remove
those paths from specialized ordinary-scene kernels. No field equations,
diffraction closures, source settings, sampling budgets or image normalization
were changed.

CPU, OSL and the three Metal variants built successfully. The exact matched
CD/DVD benchmark used the same scene and script as final_v3: M5 10-core GPU,
Metal PT, 512x369, 512 fixed samples, adaptive sampling and denoising OFF,
persistent scene data, one warmup and three measured renders including EXR
writing.

| Package | Measured seconds | Median | Warmup |
|---|---|---:|---:|
| Earlier albedo | 1.56735, 1.48800, 1.50561 | 1.50561 | 36.26018 |
| final_v3 | 1.61631, 1.61422, 1.61841 | 1.61631 | 45.43768 |
| Specialization | 1.55423, 1.47777, 1.51246 | 1.51246 | 36.71667 |

The selected build median is 6.43% lower than final_v3 and 0.46% above the earlier
albedo package. Its measured range overlaps the earlier package's range; this
does not establish a significant difference between those two builds. It is a
bounded fixture measurement, not a universal speed claim. Its image
differs from final_v3 by RMSE 5.11473e-8, maximum absolute difference 3.81470e-6,
without fitting brightness. Exact evidence is in
`tests/output/diffraction/benchmark_coherence_specialization_v1`.

All nine actual Metal PT direct-coherence oracle checks pass, including
positive/zero-length transitions and disabled groups. The per-scene manifest is
`tests/output/diffraction/coherence_specialization_pt_v1/manifest.json`.
The nine CPU host-validation cases also pass. The Metal BDPT phase-zero check passes with absolute RMSE 9.17759e-5 and mean error 4.64617e-7. The 689 packaged resource hashes were independently rechecked with zero mismatches. The Metal guiding phase-zero check also passes: absolute RMSE 9.17754e-5 and mean error 4.73843e-7. Its 155.634 s includes cold Metal compilation; this is not a warm render-time comparison. The selected-package checks therefore comprise nine PT cases, one BDPT case, one guiding case and nine host-validation cases, in addition to the matched ordinary CD/DVD benchmark. The
previous final_v3 gallery keeps its generating binary identity; its images
must not be relabeled as renders from this package.

This optimization does not implement reflected-path coherence or remove the
limitations of the direct scalar receiver model. The [mirror/virtual-source acceptance fixture](cycles_coherent_mirror_acceptance.md)
now records three actual Metal renders. The coherent cases fail phase and
absolute-radiance checks. The ordinary BDPT control agrees in mean within
0.000395 but fails the per-pixel noise gate. These failures remain open.

The selected package is a tested development delivery for the documented supported
combinations, not a completed implementation of the original full scope. Multi-GGX
diffraction, general coherent multipath and broad Realistic-solver validation remain
open. The preceding gallery and full feature-suite evidence are retained at their
original binary provenance in [the detailed
report](cycles_diffraction_final_delivery_20260927.md).
Package-manifest comparison against final_v3 found changes only in the executable, three Metal libraries, the kernel feature header, two integrator headers and the kernel CMake list. All packaged diffraction closure headers, OSL shaders and addon Python files have identical hashes. The full earlier material suite therefore tests the same closure code, while the new package checks exercise the changed feature dispatch. This is source/resource equivalence evidence, not a claim that the entire gallery was rerendered.
