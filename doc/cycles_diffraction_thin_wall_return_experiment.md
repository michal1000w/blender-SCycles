# Thin Wall return experiment — v32, no production change

Evidence: `build/tests/performance/thin_wall_return_experiment_v32_actual/results.json`.
Source: `tests/performance/cycles_diffraction_thin_wall_return_experiment_test.cpp`.
The test includes the existing native fixture's actual closure evaluation and
native GGX lookup tables, linked against the fresh v32 libraries. Every setup
uses the existing fixture hero random value0.5 (`1000*sample_wavelength(0.5)`
in nm), full grating coverage, no film or dispersion, and exterior air on both
sides; the tint case keeps reflection tint1 and transmission tint0.8. The packaged
renderer and Thin Wall implementation are unchanged. Three experiment assertions
pass; they establish the obstruction below, not successful new material support.

Both sides of a sheet are exterior air, so the reciprocal kernel is
`K(i,o)=V(i,o)/abs(cos_o)`, where actual Cycles evaluation returns projected
`V=f*abs(cos_o)`. The target per-incident surviving power is the actual native
sheet `R(i)+T(i)` after its Fresnel/internal-return and Beer attenuation, not an
assumed unit budget for absorbing material. A symmetric additive missing-energy
experiment uses `q_i=max(0,target_i-E_i)` and
`K_return(i,j)=q_i*q_j/Q`, with `Q=sum_j q_j abs(cos_j) Δω_j`.
It exactly supplies the missing discrete row power when a row is below target.
However,

```
(K_base(i,j)+K_return(i,j))-(K_base(j,i)+K_return(j,i))
    = K_base(i,j)-K_base(j,i).
```

Therefore a reciprocal added lobe cannot correct a nonreciprocal first event.
The same applies to any symmetric two-port coefficient matrix, not merely the
rank-one experiment. The production first event multiplies each port by
incoming-only native sheet Fresnel/Beer weights and incoming planar energy
compensation faded by zero-order power. Reciprocal angular carrier evaluation
alone does not make those complete weighted kernels reciprocal.

| Actual case (pitch1150nm, depth320nm, duty0.42, IOR1.5) | Exact transmission pair K(i,j), K(j,i) | Selected dense row energy / native surviving target |
|---|---|---|
| Roughness1, lossless | 0.212055945 / 0.924334705 at cos_i0.0625, cos_j−0.3125 | 0.960051912 / 1 |
| Roughness0.6, lossless | 5.75391769 / 15.934886 at cos_i0.0625, cos_j−0.1875 | 0.978436135 / 1 |
| Roughness0.6, normal-incidence transmission tint0.8 | 2.97104263 / 11.1174526 at cos_i0.0625, cos_j−0.1875 | 0.822891302 / 0.839272618 |

The 256-direction discrete matrix (8cosine×16azimuth samples per side)
produces rank-one row-budget residual below1.6e−15 for its own quadrature.
Its complete reciprocity residual remains unchanged: maximum absolute0.712279,
10.180968 and8.146410 in these cases. The roughness0.6 coarse grid overintegrates
narrow redirected lobes (apparent row energies1.25845 and1.03638). Independent
128×256 per-side evaluation of the same worst rows gives the values above;
those coarse overshoots are **not** evidence of physical energy creation.
The few dense rows are likewise not a global continuum passivity proof.

## Decision before any production patch

A small additive return can improve lossless row energy while explicitly
retaining the native incoming-angle approximation. It cannot honestly be called
a globally reciprocal completed Thin Wall model. No production patch is
justified as a simultaneous energy-and-reciprocity fix by this experiment.

A reciprocal completion needs a joint two-direction first-event sheet kernel:
paired interface/film Fresnel and absorption weights, matching order-power and
sampling probabilities, and escape deficits measured from that same kernel.
The tint needs an explicit path-length or declared passive approximation,
rather than treating absorbed sheet power as missing scattering. Directional
energy must be estimated with a resolved grid or unbiased sampling; negative
coarse deficits cannot simply be clamped and described as physics.

There is also a model-contract constraint: the existing rough native Thin Glass
limit itself has these incoming-only weights. A continuously convergent family
of reciprocal kernels cannot converge to a nonreciprocal kernel. Exact native
zero-relief identity, continuity to that native rough limit and strict complete
reciprocity cannot all be claimed simultaneously. A future change must explicitly
choose native compatibility with documented approximation, or a separate
reciprocal sheet model with an independently validated flat limit. This is a
model decision, not something a fitted brightness factor can resolve.
