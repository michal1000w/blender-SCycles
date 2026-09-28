# Metallic diffraction integration

The Metallic node exposes grating coverage, pitch/depth in nanometers, and duty
cycle. Its tangent points across grooves; rotation also rotates the grating when
anisotropy is zero. GGX and Beckmann are supported. Nonzero diffraction with
multi-scattering GGX produces a compiler error rather than silently substituting
a different distribution. Zero coverage retains the existing Metallic path.

The closure retains the original Cycles conductor or F82 Fresnel payload,
including thin film. For each facet, nonzero order power is the scalar binary
Fourier power multiplied componentwise by min(F_i,F_o). The residual mirror
receives F_i minus the other propagating powers. Thus the facet remains passive
and reciprocal, retaining absorption; this is a Fast scalar approximation,
not a corrugated-metal Maxwell solver. Sampling normalizes mean order powers
by the incident Fresnel mean, while evaluation retains spectral power. The
continuous PDF sums every inverse root/order. Smooth orders use the renderer's
separate delta-MIS path. Partial coverage keeps an ordinary conductor closure
alongside diffraction with the corresponding mixture weights.

The original Fresnel allocation and a copied microfacet carrier remain owned by
ShaderData auxiliary storage. The conversion reserves two auxiliary closure
slots; partial coverage requires an additional closure. Node closure budgeting
reserves four extra slots. GGX/Beckmann labels, roughness, blur, delta evaluation,
continuous evaluation, guiding classification and grating-aware surface MIS are
wired. Albedo passes retain the original microfacet directional estimate; this
is an approximation, not exact directional grating albedo.

SVM and OSL use the same conversion. OSL conductor/F82 closures accept optional
diffraction keyword parameters; the Metallic OSL shader supplies them. Eevee
accepts the node layout but does not implement diffraction.

Initial validation: 15,731 CPU accepted sample/dispatch events pass, including
colored physical conductor and F82 modes, smooth/rough GGX/Beckmann, and zero/
nonzero depth. Flat-profile maximum scaled BSDF error against the existing
metallic closure is 6.44633e-6; reciprocity maximum scaled error is 2.92063e-6.
This is targeted validation, not a complete energy/convergence proof. The
Blender build including OSL and Metal kernels passes. Material render evidence
is recorded separately under `tests/output/diffraction/metallic_delivery_v2`.
The v1 fixture failure is retained: it tried to set an inactive Base Color
socket on a Physical Conductor node. The corrected fixture sets only inputs
available in the selected Fresnel mode.

Expanded partial-coverage CPU checks pass 31,462 accepted events with the same
error bounds (`conductor_closure_cpu_v2.log`). Metal PT/BDPT/guiding and CPU OSL
node renders all completed with finite pixels; previews were visually reviewed.
`metallic_delivery_v2/index.html` links all scenes and EXRs. Sample counts differ
(256/128/256/32); render timings are not a cross-method performance comparison.
The grating currently uses vacuum wavelength at the interface, matching an
air-side incident-medium assumption. General buried-medium controls remain
part of the unfinished broader material integration.
