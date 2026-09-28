# Metallic diffraction wavelength-domain correction

The v32 and v36 Metallic appearance images preserve119 pixels with substantial
negative linear BT709 luminance (minimum −0.475582). This is distinct from
out-of-gamut RGB with nonnegative luminance. The old diffraction conductor kept
chromatic RGB Fresnel coefficients and multiplied them componentwise by the
signed RGB sensor basis of the sampled wavelength. That operation can turn a
positive spectral sample into negative photometric energy.

The correction reconstructs the computed angular reflective Fresnel response
into a bounded scalar spectrum at the sampled wavelength, before applying the
sensor basis. It uses the existing BT709 spectral reconstruction. Physical
conductor optical constants retain their original channelwise interpretation;
there is no arbitrary interpolation of n/k. Coated RGB Fresnel reconstruction
remains an explicit Fast reflectance approximation, not a Maxwell model.

A wavelength marker uses existing MicrofacetBsdf alignment padding (96 bytes
unchanged; conductor extra144 bytes unchanged). Native setups initialize it to
zero. Only active diffraction reflective conductor/F82/generalized layers use
it; partial native coverage shares the same wavelength-domain response.
First-event coefficients, native energy-preservation averages and additional
scattering coefficients are treated consistently. Exact zero depth or coverage
retains native RGB Fresnel with no spectral marker. No rendered RGB/luminance
clamp, absolute-value conversion or image normalization is introduced.

[The bounded CPU closure regression](../build/tests/performance/metallic_spectral_reflectance_final_v37/results.json)
passes compile/link/run with zero failures:28,665 compared samples have exact
sample/evaluation PDF and value agreement; maximum white furnace error remains
0.00206083. Tests explicitly check depth0/coverage0, direct zero-depth setup,
partial native coverage and unchanged storage sizes. Across401 wavelengths,
the old chromatic Fresnel/sensor product reaches luminance−0.0419716; the scalar
correction reaches only−1.40e−8 numerical roundoff. These tests do not replace
the required corrected-binary actual Metallic render and raw luminance audit.

The [actual v37 raw Metal image audit](../build/tests/python/metallic_spectral_correction_v37/raw_luminance_audit.json)
now passes:720×400/512fixed PT, adaptiveOFF, raw NoisyImage all finite,
zero pixels with Y<−1e−6 and minimum linear Rec.709 Y+0.000852877.
All5019 pixels with negative RGB channels have nonnegative luminance; these
are valid out-of-gamut sensor coordinates, not clamped away. The denoised preview
is separate and is not the energy proof. Executable SHA256 is
`46b8fb0639dee1dcea94ae42afd3acd7e794132f6cca10b4735d1ee45caf3a20`. The earlier v32/v36 negative-luminance images remain preserved.
This actual appearance/radiance-sign check does not establish full spectral
accuracy, every material configuration or a Maxwell solution.

The [older conductor regression](../build/tests/performance/conductor_native_zero_regression_v37/results.json)
was updated to verify the zero-depth native closure directly instead of casting
it as a grating. All63,276 sampled events pass (zero failures); native flat
value/PDF comparison maximum scaled error9.565e−7, reciprocity9.417e−7. Type,
weights, Fresnel pointer/value, energy scale, marker and dispersion flag are
explicitly checked at zero depth. Nonzero coverage and grating-order checks
remain in the test.
