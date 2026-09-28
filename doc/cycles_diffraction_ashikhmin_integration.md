# Glossy Ashikhmin-Shirley diffraction

The Glossy node's Ashikhmin-Shirley distribution now has a native Fast scalar
reflection grating closure in SVM and OSL. Relief depth zero allocates the
ordinary Cycles Ashikhmin-Shirley closure with the original color weight and
setup, preserving its existing sampling/evaluation behavior. Positive depth
uses the native anisotropic half-vector distribution and the grating's existing
facet-order sampler, two-root inverse map, and outgoing solid-angle Jacobian.
The grating tangent defines local X, and wavelength is divided by the explicit
surrounding-medium IOR.

For a facet normal `h`, let `c_i=dot(w_i,h)`, `c_o=dot(w_o,h)`, and `p_h` be the
native Ashikhmin-Shirley half-vector density. The order proposal is
`q_m=p_h P_m J_forward`, summed over every valid inverse root. The physical
`f*cos_o` contribution is

```
q_m * [mu_o/max(mu_i,mu_o)] * [min(c_i,c_o)/c_o].
```

The second factor equals one at order zero, giving native Ashikhmin
evaluation. For nonzero orders, the grating Jacobian obeys
`J_forward=c_o/det`, `J_reverse=c_i/det`; the extra factor therefore makes
`f` reciprocal. It also keeps `f*cos_o/q_m <= 1`, so a unit-reflectance grating
is passive, with below-surface facet/order outcomes retained as null events.
This is a reciprocal Fast scalar approximation, not an electromagnetic
solution for a rough relief surface. It does not add multiple scattering.

`tests/performance/cycles_diffraction_ashikhmin_test.cpp` exercises native
zero-relief evaluation/PDF equality, reciprocal nonzero orders, sample/evaluate
PDF consistency, a unit white furnace, and singular-order sample/delta masses.
On 162,458 accepted continuous samples across two roughnesses, two order
spacings, zero/positive depth, and three incidences, the standalone test had
zero failures. Its largest absolute flat-interface difference was `3.05e-5`,
largest absolute reciprocity difference `6.10e-5`, and largest estimated
white-furnace albedo `0.930342`. These are finite representative checks, not
global bounds or a render validation.

`tests/python/cycles_diffraction_glossy_validation.py` now expects enabled
Ashikhmin diffraction to compile, while Multi-GGX remains an explicit error.
`tests/python/cycles_diffraction_glossy_delivery.py --ashikhmin` adds a fourth
Ashikhmin material to the representative render scene. The final_v3 application completed the fixture on Metal PT, BDPT and guiding
and on CPU OSL at 480x267 and 128 fixed samples, adaptive sampling and denoising
OFF. All outputs are finite and visually inspected; spectral reflection orders
are visible on the Ashikhmin sphere. These are transport/appearance smokes, not
convergence proof. Exact scenes, raw EXRs and reports are in
`tests/output/diffraction/final_features_v3/ashikhmin_*`.
