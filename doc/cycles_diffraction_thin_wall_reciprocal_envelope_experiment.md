# Thin Wall reciprocal-envelope experiment — v32, rejected for production

Prototype: `tests/performance/cycles_diffraction_thin_wall_reciprocal_envelope_test.cpp`.
Evidence: `build/tests/performance/thin_wall_reciprocal_envelope_v32/results.json`.
The [previous additive-return experiment](cycles_diffraction_thin_wall_return_experiment.md)
shows why a symmetric lobe alone cannot remove native first-event antisymmetry.
Production/package v32 is unchanged. No GPU or Blender rebuild was used.
The three existing configurations use the actual Thin Sheet closure, native
GGX tables and weights, with the original native fixture helpers. The model
below is an explicit conservative mathematical Fast heuristic, not a published
microscopic or Maxwell solution.

## Candidate and exact discrete properties

For each port, both sides are exterior air, and the projected native evaluation
is converted to `K_actual(i,o)=V_actual(i,o)/abs(cos_o)`. Reflection and
transmission occupy disjoint hemispheres in these rough cases. Define

```
K_env(i,o) = min(K_actual(i,o), K_actual(o,i))
E_env(i) = integral K_env(i,o) abs(cos_o) dω_o
S(i) = native sheet R(i)+T(i), including its Beer attenuation
q(i) = S(i)-E_env(i)
Q = integral q(o) abs(cos_o) dω_o
K_complete(i,o) = K_env(i,o) + q(i)q(o)/Q
```

This is nonnegative and reciprocal if all q are nonnegative. With accurately
matched integrals it has surviving row power S(i), retaining rather than
recycling native absorption. It is not justified by merely adding a reciprocal
lobe to the old nonreciprocal first event: the first event is explicitly replaced
by its conservative reciprocal envelope.

For 1024directions (16cosine×32azimuth nodes on each side), all3 discrete matrices
have positive deficits and exact pair symmetry. Row-budget and proposal-weighted
expectation residuals are at most4.5e−15; reciprocity residual is zero. These are
finite quadrature/algebra results, not continuum passivity certification.

## Resolved-row failure: stop, no production unlock

A separate fixed128×256-per-side integration checks the smallest-deficit row
in each case. The outgoing return can be normalized exactly as a piecewise
constant q-cell times cosine density; its row integral is the cached q(i).
Consequently the resolved first-event energy plus that cached return exposes
the quadrature mismatch directly:

| Roughness / normal transmission tint | Cached E_env / q | Independently resolved E_env | Resolved completed row / actual surviving budget |
|---|---|---|---|
| 1 / 1 | 0.715697430 / 0.284302570 | 0.715231739 | 0.999534309 / 1 |
| 0.6 / 1 | 0.899291837 / 0.100708163 | 0.899890497 | **1.000598660 / 1** |
| 0.6 / 0.8 | 0.698284242 / 0.086306837 | 0.698715847 | **0.785022684 / 0.784591079** |

The latter two rows exceed their intended surviving-power budgets. The absorbing
case remains below unit incident power but still recycles some intended
absorption. This rejects the current cache-realized candidate as a conservative
completion. No negative deficit was clamped, no threshold changed and no further
adaptive convergence campaign was started. The mathematical construction remains
conditional on resolved and consistently normalized q/Q; this implementation
experiment does not supply that guarantee.

## Concrete unbiased sampling construction

Let `p_native(o|i)` be the actual native closure-mixture proposal, including its
existing sample weights, rejection mass and per-port PDFs. A sampled native
outcome is accepted with `a(i,o)=K_env/K_actual` where the denominator is
positive. Rejected samples contribute zero and are not resampled or normalized
away. A separate return proposal samples `q(o)*abs(cos_o)/Q`, with a cached
cell CDF and cosine-distributed position inside its cell. Choose any interior
mixture probability β(i), for example E_env/S. The actual outcome density is

```
p(o|i) = β p_native(o|i) a(i,o) + (1-β) q(o) abs(cos_o)/Q.
```

Evaluate the full completed projected kernel and divide by this density.
It is unbiased for the declared approximate kernel when support, acceptance
and PDF evaluation agree. The prototype explicitly checks this discrete
expectation using actual native mixture PDFs. Its integral can be below1 because
native rejection remains a legitimate zero-contribution outcome. PDF evaluation
must include acceptance; using the old native PDF alone would be incorrect.

## Requirements before implementation

A real cache must resolve the reciprocal envelope on both sides and wavelengths,
keep q nonnegative without hiding quadrature error, and integrate its interpolant
consistently for Q and the return CDF. First-event evaluations, rejection sampling,
MIS/BDPT reverse PDFs and guiding labels must use the same declared kernel.
This small experiment uses only no-film/no-dispersion IOR1.5, fullcoverage,
pitch1150nm/depth320nm/duty0.42, hero wavelength from sample_wavelength(0.5),
reflection tint1 and gray transmission tint1 or0.8. Smooth atomic orders, film,
spectral/linked tint, partialcoverage and other parameters are not validated.

The native rough flat limit itself is nonreciprocal under incoming-only weights.
Exact native zero-relief identity, continuity to that native limit and strict
reciprocity cannot all be claimed. A future explicit model choice must state
whether it retains native compatibility or adopts a separate reciprocal sheet
flat limit. This experiment does not silently change that contract.

[OpenPBR](https://academysoftwarefoundation.github.io/OpenPBR/#model/thin-walledmode)
describes rough thin-walled transport as approximate and acknowledges
nonreciprocal layer albedo scaling. Native compatibility therefore needs an
explicit approximation contract.
The [JCGT multiple-scattering treatment](https://jcgt.org/published/0008/01/03/paper.pdf)
derives furnace relationships with reciprocal BRDF assumptions; its missing-energy
construction cannot blindly repair a nonreciprocal starting closure.
