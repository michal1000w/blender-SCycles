# Remaining transport requirements: source and research audit

Status: implementation requirements, not delivered support. Inspected 2026-09-27.

## Multiple scattering

Current source evidence:
- `kernel/closure/bsdf_microfacet.h::microfacet_ggx_preserve_energy` reads ordinary
  planar GGX directional/average albedo tables, with separate glass and inverse
  glass tables. It sets `energy_scale` and adjusts the closure weight for
  multibounce Fresnel darkening.
- `bsdf_microfacet_ggx_eval` and its sample wrapper apply `energy_scale`.
- `kernel/closure/bsdf_diffraction_conductor.h` copies the native carrier including
  its Fresnel pointer but evaluates its own order sum without that scale.
- Enabled multiscattering node combinations remain guarded in shader compilation.

Removing the guards would not preserve the native compensated material. Merely
multiplying by the copied scale is also insufficient: the planar albedo tables
have no wavelength/pitch/depth/duty dependence and cannot establish the energy
lost by the grating's order-dependent masking and rejected directions. This is
an inference from the current equations, not a measured counterexample.

A complete implementation needs either a sampled microsurface multiple-bounce
model with diffraction at each interaction, or a declared energy-compensation
approximation whose directional/hemispherical albedos are computed for the grating
model. Required gates include the native zero-depth limit, a white furnace over
roughness/incidence/wavelength, reciprocity/adjoint treatment, and matching
sample/evaluate PDFs. Do not substitute planar compensation and call it validated
multiscattering diffraction.

## General cross-object coherence

Primary research reviewed:

1. Steinberg et al., *A Generalized Ray Formulation For Wave-Optical Light
   Transport*, SIGGRAPH Asia 2024, DOI 10.1145/3687902.
   https://ssteinberg.xyz/2023/03/27/rtplt/
   The authors derive sensor-originated generalized rays as weakly-local queries
   of wave distributions. This is a route to backward wave transport rather than
   summing independently sampled ray intensities.
2. Steinberg and Pharr, *Wave Tracing: Generalizing The Path Integral To Wave
   Optics*, Eurographics 2026, DOI 10.1111/cgf.70322, first published 8 April 2026.
   https://www.shlomisteinberg.com/2025/08/28/generalizing_the_path_integral/
   The work introduces a bilinear path integral and then region-to-region transport
   using elliptical cones. It explicitly addresses interference between paths
   and diffraction by geometry. The 2025 URL/preprint date should not be mistaken
   for the final publication date.
3. Author implementation: https://github.com/ssteinberg/wave_tracer
   The README describes CPU-only execution, early-alpha status and roughly 5–20x
   cost versus classical tracing, depending on the scene. This is the authors'
   estimate, not a Cycles/Metal benchmark or a forecast for this implementation.
   Its stated license is CC BY-NC 4.0; no source was copied into this worktree.

Implementation direction inferred from these sources: retain Fast local grating
shading as the default and introduce an optional generalized-wave transport path.
A single phase float attached to ordinary rays is not sufficient for the requested
partially coherent, separate-object interaction scope.

Current Cycles state has local grating intensities but no transported optical
coherence state. The integrator `state_flow.h` uses the word coherence for GPU
scheduling locality; that is unrelated to optical coherence. No user-visible
coherence toggle should claim a feature until the transport is implemented.

The implementation must represent source spatial/spectral coherence and sensor
response, propagate the wave footprint and phase reliably across object and medium
boundaries, evaluate geometry over that footprint, and accumulate a valid detector
estimate. PT and BDPT must share the same transport measure and reciprocal kernels.
Guiding needs a nonnegative proposal with the correct signed/complex contribution
weight; existing intensity-training assumptions cannot simply be reused unchecked.
These are design requirements inferred from the desired functionality, not claims
that the cited implementations support every Cycles feature or Metal.

Acceptance fixtures should include separate-object double slits (fringe spacing,
visibility and single-slit envelope), two reflectors with controllable path delay,
coherence-length and source-size sweeps, two wavelengths, translated/scaled scenes,
and the incoherent limit. Reference values must come from independent wave
calculations. Changing PT/BDPT brightness to make previews agree is not a gate.

No new renderer support is claimed by this audit. The tested packaged candidate
and its feature restrictions remain unchanged.
