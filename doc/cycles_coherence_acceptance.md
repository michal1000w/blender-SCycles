# Cross-object coherence acceptance references

This document preserves the original metadata-only acceptance experiment and its
failed phase response. A later **direct point-source scalar model** now has
working light controls and separate renderer acceptance scenes; see
[the implemented model and its limits](cycles_coherent_direct.md). General
coherent reflected-path transport remains unimplemented. The historical results
below are unchanged and must not be presented as tests of the new controls.

## Independent reference

`tests/performance/cycles_coherence_reference.py` computes the scalar intensity
of two equal spherical sources separated by 100 micrometres, with a detector
one metre away. It evaluates their complex-field cross term, using a factored
path-length difference to avoid subtracting nearly equal distances. It models
free space only: no finite apertures, material scattering, polarization coupling,
or detector BSDF. Thus it is not a general wave-optical scene solver.

The Gaussian spectrum is defined in wavenumber, with coherence length explicitly
defined here as `1 / sigma_k`; this is not a universal coherence-length convention.
The partial-coherence formula is independently checked by 64-point Gauss-Hermite
integration of complex fields over that spectrum. Random-phase integration and
orthogonal vector components independently recover the incoherent limit.

Fifteen checks pass, covering constructive/destructive interference, phase-shift
complementarity, nonnegative intensity, spectral quadrature, random phase,
physical unit rescaling and the paraxial fringe-position limit. The first exact
spherical phase fringe is at 5.320075291 mm; the paraxial estimate is 5.320000 mm.
Scaling tests scale geometry and wavelength together and compensate inverse-square
amplitude; they do not assert that changing object size at fixed wavelength leaves
the image unchanged. Full translated-scene and finite-aperture tests remain open.

Files in `tests/output/diffraction/coherence_reference_v1` include the JSON
definition and gates, CSV/NPZ profiles and a clearly labeled reference plot.
These curves are not Cycles renders and are never inserted into Blender materials.

## Saved scenes and native controls

`tests/python/cycles_coherence_acceptance_scenes.py` saves six `.blend` files:
in-phase, opposite-phase, incoherent, two partial-coherence lengths, and 633 nm.
Each has two separate point-light objects and a diffuse detector. `reference_*`
properties document the intended physical input; **Cycles does not consume them**.
The native white point lights do not implement the reference monochromatic
spectrum. Until actual coherence controls are bound, these are acceptance
fixtures and unsupported controls, not a passing renderer regression suite.

Two native Metal controls were rendered sequentially at 64 samples, adaptive
sampling OFF and denoising OFF, seed 11. The phase-pair maximum pixel difference
was 1.1920928955078125e-7. The reference center changes from `4/r^2` to zero.
The manifest explicitly records phase-response acceptance as **FAIL / unimplemented**.
No comparison of absolute detector brightness is made: the native diffuse
detector and the scalar reference measure differ.

The first render took 35.0451 s including preparation; the second took 0.4978 s.
This is not a coherence performance benchmark or a speed comparison between
candidate transports. Both images are finite. Binary SHA-256:
`8b383bfaab2e82bb5136b3dcbb985decef983b906516816d2f57217ae40cc0cc`.
The scenes, EXRs, PNGs and manifest are in
`tests/output/diffraction/coherence_acceptance_scenes_v1`.

## Sources and remaining implementation

The two-source fringe limits follow the wave-superposition and interference
treatment in [MIT's interference and diffraction notes](https://ocw.mit.edu/courses/8-02-physics-ii-electricity-and-magnetism-spring-2007/c1ef2e99446d7c4255e32752af8f6d26_ch14_inter_diffr.pdf).
The independent code and plots were written for this work; no source code or
figures were copied. The Gaussian spectrum tests directly integrate the stated
statistical model.

[Wave Tracing: Generalizing The Path Integral To Wave Optics](https://www.shlomisteinberg.com/2025/08/28/generalizing_the_path_integral/)
addresses the broader coherent transport problem using a bilinear path integral.
This reference fixture does not implement that method. General source/sensor
coherence, wave footprints, interaction with geometry/materials, reciprocal
PT/BDPT measures and valid guiding proposals remain required renderer work.
