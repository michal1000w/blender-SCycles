# Principled Thin Wall diffraction candidate

Status: implemented as an experimental approximation in v32. The current
presentation is `build/tests/python/presentation_refresh_v32/thin_wall`; it is
an appearance check, not an energy or reciprocity certification. The numerical
results below describe the earlier focused validation. The known rough-sheet
energy loss remains unresolved; see the current
[completion audit](cycles_diffraction_completion_audit.md).

## Model

The native Thin Glass closure treats a sheet as two interfaces. It computes a
spectral sheet reflection and transmission coefficient, including normal-angle
transmission tint absorption and optional thin film. It models rough reflection
as GGX reflection and rough transmission as GGX reflection of a plane-mirrored
incident ray, with a separate effective transmission roughness.

The Fast grating candidate keeps those sheet coefficients and angular carriers.
On each visible facet it applies a binary phase screen with Fourier power
`|c_m|²`. The reflected phase difference is `4π n_ext d / λ`, with exterior
index `n_ext=1` for the current Thin Wall model. The transmitted phase
difference is `2π(n_sheet−n_ext)d / λ`. This assigns a local effective relief
phase to the sheet; it is not a Maxwell or RCWA solution for a corrugated
two-interface slab. Spectral dispersion changes `n_sheet(λ)` before the phase
and native Fresnel calculation. The grating is intensity only and has no
cross-path optical coherence.

For each facet, positive and negative propagating nonzero orders receive the
same binary Fourier power. Evanescent-order power and truncated-tail power go
to the zero order. This makes the scalar order distribution passive and gives
the two ports reciprocal angular kernels. It does not establish reciprocity of
the full sheet closure: its native sheet coefficients also depend on the
incident macro angle. The zero-order ray is exactly the
native reflected or straight-through sheet direction. The rough carrier uses
the same GGX density and masking function as native Thin Glass. The native
planar GGX energy compensation multiplies each port in the flat limit. For
positive relief, its excess is faded by order-zero Fourier power. This
conservative correction keeps the near-flat limit while reducing energy added
to redirected orders. It remains an approximation because the planar albedo
table does not include order-dependent masking.

At exactly zero relief, the code calls the existing Thin Glass setup. For
positive relief approaching zero, all nonzero order powers vanish as `d²` and
the remaining carrier approaches the native compensated carrier. This requires
the spectral reflection tint to enter once: it is already included in the
native Fresnel coefficient. The same sheet absorption and film coefficients
apply to the zero and nonzero orders. The sheet retains `eta=1` for transport;
the effective sheet IOR affects Fresnel, dispersion and phase, not net bending.

## Validation and limits

The current focused math checks cover independent zero-order GGX expressions,
small-relief convergence for several roughness values, reciprocal angular
evaluation, matching sample/evaluate PDFs, and smooth-facet power accounting.
A standalone CPU test loads Blender's GGX albedo tables and compares the new
closure against actual native Thin Glass reflection and transmission, including
film, at zero and small positive relief. All 36 comparisons passed, with
maximum relative error 0.0001341 across roughness 0.2, 0.6 and 1.0, film 0 and
180 nm, and depth 0, 0.01 and 0.001 nm. A 32-by-64 directional quadrature at
320 nm relief and 1150 nm pitch measured a maximum white-furnace energy of
0.980753 for nine roughness/incidence combinations. At roughness 1, energy
was only 0.69–0.74; the approximation can darken strongly diffracting rough
sheets. These measurements do not prove passivity at all inputs.
The source also parses through CPU kernel, OSL closure and scene translation
units, and the Principled OSL shader compiles. The final_v3 application also completed the 480-pixel-wide, 128-fixed-sample
fixture on Metal PT, BDPT and guiding and on CPU OSL. These raw appearance and
transport smokes have finite outputs; they are not convergence proofs. The
zero-relief and zero-coverage images agree within 9.53675e-7 per linear RGB
channel, while relief changes the image (RMSE 0.232475). Exact reports and
scenes are in `tests/output/diffraction/final_features_v3`. Broader white-furnace
behavior across wavelength, pitch, depth, roughness and angle remains open. The native spectral coefficients are
computed at the incident macro angle, as in existing Thin Glass; this is a
native approximation that need not be reciprocal after changing the macro
incident direction. The reciprocal claim above applies to the scalar grating
angular kernel and facet order distribution.
