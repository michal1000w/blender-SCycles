# Curved coherent transport: exterior analytic sphere reflection

The first curved receiver connection uses Cycles' native point-cloud sphere primitive. It solves one ideal Mirror reflection on its actual spherical geometry, rather than fitting a smooth surface to triangles or interpreting a radiometric BSDF as a complex field. Scalar mirror and explicitly selected three-world-dipole vector source models retain their existing meanings. The bounded physical sphere fixture passes on the immutable v36 package; the earlier failed renders and isolated diagnostics are preserved below.

For a source and receiver outside a convex sphere, the reflection normal lies in their common plane with the sphere center. The core brackets the stationary angular root on the intersection of the two visible spherical caps. Thirty bisection steps select the unique front-facing branch; disjoint caps, inside endpoints and grazing zero-measure paths have no represented contribution. Source and receiver distances, radius and center are the exact represented scene inputs.

The local curvature enters the optical Hessian. With unit tangent columns B, segment directions d_s,d_r pointing from their endpoints to the interface, distances l_s,l_r and outward normal n,

```
H = Bᵀ[(I−d_s d_sᵀ)/l_s + (I−d_r d_rᵀ)/l_r]B
    − dot(d_s+d_r,n)/radius * I.
```

Implicit differentiation of the stationary equations with respect to receiver area yields the source-direction differential; its transverse determinant is `dΩ_source/dA_receiver`. On axis this gives `1/(a+b+2ab/radius)^2`, where a and b are the distances from the reflection vertex. This is the curvature spreading used by field transport. It is not a fitted intensity or square root of a sampled BSDF throughput. The exterior convex stationary Hessian is positive, so this branch introduces no relative caustic/Maslov phase. Concave focusing and fold caustics are not supported by assigning them zero phase: they are outside this branch inventory.

The normalized sphere point and both physical segment lengths use compensated float expansions for optical phase. Visibility offsets only the ray origin, leaving these optical segments unchanged. The solved radial normal provides both the visibility offset and the local Jones basis. Primitive membership and native light-path history now include primitive type in addition to object and primitive index, preventing point IDs from aliasing triangle IDs.

Current host limits are explicit: one interface event in scenes declaring spheres, exterior sources, ideal white Mirror material, static exact world spheres, and flat passive Lambertian receiving triangles. Baked native point positions/radii are used verbatim; unbaked instances admit only the conservative exactly representable uniform signed-axis transform subset. Sphere Glass is rejected because native point intersection currently exposes only the front root. Multiple curved interfaces, arbitrary smooth meshes, refraction through sphere interiors, rough scattering field phase and caustic uniformization remain unfinished requirements. Planar reflection/refraction sequences retain their existing solver.

CPU evidence before renderer acceptance:

- `build/tests/performance/coherent_sphere_v33/core_results.json`: 63 checks each in strict and fast math; independent angular root, split OPL, finite-difference Jacobian, axial expression, scales .01/1/100, tiny-angle and near-horizon cases, inside/disjoint-cap rejection.
- `build/tests/performance/coherent_sphere_v33/results.json`: independent reviewer comparison, 108 tilted/translated/scaled cases per mode. Fast maximum OPL error 2.07e−12 m, paired OPD error 8.26e−13 m, spreading relative error 6.69e−6. The independent double reference does not call the production connector.
- `build/tests/performance/coherent_sphere_membership_v33/results.json`: 20 exact membership/history checks, including point/triangle ID collision.
- `build/tests/performance/coherent_sphere_v33/metal_syntax.log`: generic full-kernel Metal syntax check.

The geometric Jacobian framing follows [Specular Manifolds](https://www.iliyan.com/publications/SpecularManifolds/SpecularManifolds_Sig2020_rev3.pdf); curvature differentiation is consistent with the reflected ray differential construction in [PBRT](https://pbr-book.org/3ed-2018/Texture/Sampling_and_Antialiasing). The particular sphere solve and the independent oracle here are derived directly for this bounded transport model.

## Renderer integration diagnosis and v36 fix

The corrected native-sphere fixture (`cycles_coherent_sphere_acceptance_v33_v2`) preserves the independent geometry/reference arrays and explicitly sets the Mirror Color to white. Its actual Geometry Nodes output is one native point of radius .25 m; the companion evaluated mesh has zero vertices and triangles. Host extraction skips that empty interface companion without manufacturing a plane or relaxing the nonempty-detector requirement.

The v33b physical vector suite and the tiny scalar control both rendered entirely black. A native unit-emission sphere diagnostic showed the point primitive was present. CPU reconstruction of the old Metal bounding-box decoder found a separate unconditional triangle-decoding defect, but the aliased detector triangle missed all 18 tested source-to-sphere rays, so that defect was not claimed as the black-render cause.

Diagnostic-only APFS clones of v34 localized the rejection on the actual M5 GPU. The first 16×16/4-sample counter render was exactly RGB (1,1,0) throughout: both sphere candidates were eligible and both solved, with neither visible. The next diagnostic reported failure code 8 throughout: `scene_intersect` returned no hit on segment 0, source to sphere. These are stage encodings, not physical images or acceptance references.

The grounded integration defect was Metal pipeline classification: the coherent evaluator traces rays from ordinary `SHADE_SURFACE`, but that pipeline was not classified as requiring intersection tables. Hardware triangles can conceal the missing custom table; native point bounding boxes require the `__intersection__point` callback. v36 adds a Metal-only feature-aware classification for ordinary surface shading when coherent connections are enabled. All five linking/table/archive/stack/resource call sites use it. The active scene's coherent bit participates in generic as well as specialized cache identity, and tables are released before an ON/OFF pipeline switch. The legacy global classifier and initial feature-OFF cache identity are unchanged.

Evidence retained:

- `build/tests/python/cycles_coherent_sphere_render_v33b_retry/results.json`: failed physical suite, all three cases black.
- `build/tests/performance/coherent_sphere_v33/decoder_reconstruction.json`: authoritative array-layout reconstruction and 18 native CPU ray checks.
- `build/tests/python/cycles_coherent_sphere_stages_diagnostic_v34/report.json`: actual eligible/solver/visibility counters.
- `build/tests/python/cycles_coherent_sphere_visibility_diagnostic_v34/decoded_visibility_failure.json`: actual segment-0 no-hit classification.
- `build/diagnostic_coherent_sphere_stages_v34/`: isolated clone source diffs/hashes; production source was not instrumented.
- `build/tests/performance/coherent_metal_intersection_v35/results.json`: 360 classification checks.
- `build/tests/performance/coherent_metal_intersection_v35/objc_syntax.json`: all three modified ObjC++ translation units pass syntax checking.

## Bounded v36 physical acceptance

The unchanged double reference and predeclared absolute gates pass for all three actual M5 Metal renders: phase 0, phase π, and connector-enabled distinct source groups. At 256×256 and 128 fixed samples, raw radiance RMSE is 7.51441101e-05, 7.50875426e-05, and 3.39887566e-06 respectively; the phase-difference RMSE is .000150078. Adaptive sampling and denoising are OFF. These are BDPT scenes with one permitted glossy interface event and diffuse continuation0, isolating the first flat Lambertian detector encounter. Reference NPZ SHA256 remains `b4524d5d4b2c5d5b34f7933e54f092f72a6995650bf92a7ebdb519103ce62fb2`.

All four saved negative scenes produce their required descriptive host errors: an internal source, a Glass sphere, an unbaked nonuniform instance, and a requested multiple-interface sequence. The runner detects another error or accidental render success as a failure. Current binary SHA256 is `10bd061aacbe067b36b495cc560aee2b5c85bb1c54842d66a04d0e69248f54a5`.

- Physical cases and phase-difference gate: `build/tests/python/cycles_coherent_sphere_render_v36/results.json`.
- Guard execution and per-scene hashes: `build/tests/python/cycles_coherent_sphere_negatives_v36/{execution,report}.json`.
- Actual EXR contact: `build/tests/python/cycles_coherent_sphere_contact_v36/actual_native_sphere_fixed_exposure.png`, all three images at common −1 EV plus sRGB, with no case normalization.
- Independent raw EXR readback: `build/tests/python/cycles_coherent_sphere_contact_v36/raw_metric_verification.json`; means and RMSE match worker reports within 1e−8.

This acceptance covers one exterior reflection on one native analytic convex sphere, two coherent point sources, the globally projected dipole ensemble, and a flat white Lambertian detector. It does not validate curved refraction, interior/concave branches, multiple curved events, caustics, arbitrary smooth meshes, or general BSDF field phases. Passing this fixture is an implemented incremental curved-surface result, not completion of general coherent multipath transport.
