Current update: standard-Fresnel transmission is now implemented under explicit
input constraints. See cycles_diffraction_principled_transmission.md. The reflective
integration history and earlier transmission audit below are retained; their blanket
zero-transmission restriction is superseded by that update.

# Principled reflective diffraction

Principled exposes Diffraction Weight, Pitch, Depth and Duty Cycle. The current
implementation covers its metal and dielectric base-reflection layers using
GGX. Transmission Weight must be zero. Multi-scattering and transmitting
Principled combinations report a compiler error when diffraction is enabled;
they remain unfinished requirements, not silently approximated substitutes.
Ordinary materials with zero Diffraction Weight retain their existing behavior.

The shared Fresnel-aware reflection closure now accepts Generalized Schlick in
addition to conductor and F82 payloads. It preserves Specular IOR Level, Specular
Tint and thin-film response. Sheen/coat attenuation is evaluated before creating
the base layers. The original specular albedo estimate attenuates the diffuse/
subsurface layer before the reflective closure is split for partial grating
coverage. OSL computes its layer-albedo return before the same split. Thus
splitting does not accidentally remove only the uncovered fraction of the
layer's attenuation. These native directional albedo estimates remain
approximations for a diffracting surface.

Tangent and Anisotropic Rotation control the grooves even at zero anisotropy.
Graph simplification retains both inputs when diffraction is enabled. The v1
render fixtures predate that correction: they completed in PT/BDPT/guiding/OSL,
but used the fallback tangent. The v2 PT/OSL fixtures validate the corrected
input path and include a quarter-turn grating on the partially covered mixed
material. Neither set establishes convergence or all pipeline coverage.

The expanded CPU closure regression passes 47,190 accepted events, including
Generalized Schlick, conductor and F82, full and half coverage, GGX/Beckmann,
and smooth/rough profiles. Flat-profile maximum scaled BSDF error against
ordinary microfacet reflection is 6.44633e-6; reciprocity error is 2.92063e-6.
The full build passes. Runtime evidence is stored in
`tests/output/diffraction/principled_delivery_v1` and `principled_delivery_v2`.

Still open: diffraction in Principled transmission/dispersion/thin-wall paths,
multiple scattering, explicit buried-medium controls, cross-object coherent
transport, and wider pipeline/numerical validation. Coats retain the existing
Principled layer approximation; this is not a geometric refraction model of a
separate cover object.

The final tangent-corrected Metal BDPT and guiding renders also pass at 256 fixed
samples, with finite pixels and visually inspected previews. Their measured
render times are 52.7271 s and 22.4705 s. Evidence:
`tests/output/diffraction/principled_delivery_final_transport_v1`.

## Transmission audit

The existing Principled transmission closure uses Generalized Schlick Fresnel:
its Specular Tint scales F0, and the angular interpolation approaches white at
grazing incidence. The Glass grating's fixed reflection tint is not equivalent.
Reusing it unchanged would alter even the zero-depth material. A correct
integration must retain the original angle-dependent Fresnel payload in both
order powers and sampling/evaluation, together with wavelength-dependent IOR
from `bsdf_glass_ior`, backface orientation, thin-film response and the separate
transmission filter. Coated matched-index atoms require their own conditional
mass treatment. This audit does not implement transmission; the compiler guard
remains in place. Native planar-limit, reciprocity, sample/evaluate and rendered
transmission/dispersion checks remain acceptance requirements.
