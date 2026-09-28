# Two-sided Fast diffraction for Glass MultiGGX

The bounded Glass path combines the existing physical first diffraction event
with an approximate reciprocal return for masked multiple scattering. The
cache stores the missing directional energy on the air and glass sides,
separately for each wavelength. The return redistributes that missing energy
between reflection and transmission under the `n² cos(theta)` etendue measure.
Its side mixing uses the corresponding lossless Fresnel conductance. This is a
Fast model, not an exact simulation of every internal facet sequence.

The cache is independent of Glass Color. SVM and OSL clamp a linked or
constant Color to RGB `[0,1]`, reconstruct the reflection and transmission
spectra, then clamp each spectral tint to `[0,1]` before coverage and material
weight. The same local tint applies to both travel directions at a given
wavelength and once to the completed return. It preserves the reciprocal
two-sided model but is **not** Beer–Lambert absorption or per-bounce tinting.

The lossless real film extension carries film IOR and physical thickness in
the cache key and samples the coated first event from either side. The
cross-side conductance is wavelength-dependent. A zero-thickness film uses
the bare-interface response and shares the same cache independently of the
inactive film IOR. The film's Airy fringes are represented by a linear
wavelength table with at least eight nodes per estimated shortest blue-edge
period, up to 256 nodes and the CPU work budget. That rule bounds work and
limits coarse sampling; it does not certify interpolation error for every
high-finesse film. Linked film, roughness, IOR and grating parameters are
rejected because they would require spatially varying caches. Negative or
complex film absorption is outside this model.

The v23 neutral Metal white-environment furnace passed from both sides after
the SVM branch was wired to the new return closure. At 64×64 and 256 fixed
samples, the central linear RGB means were `[1.0031,1.0060,1.0031]` front and
`[1.0024,1.0056,0.9978]` back; the predeclared per-channel unit-radiance and
front/back tolerances were both `0.03`. Native Glass was near one, while the
single-event back control was about `0.919`. The raw EXRs and report are in
`build/tests/python/cycles_diffraction_glass_multiggx_furnace_render_v23`.

The v24 colored Metal white-environment comparison used a passive chromatic
Color and passed a different, explicitly bounded gate: finite nonnegative
RGB, each channel at most `1.03`, front/back difference at most `0.08`, and
back-side return gain over the single-event control at least `0.02`.
The two-sided means were `[0.3519,0.6539,0.9026]` front and
`[0.3514,0.6536,0.8976]` back. These are not an assertion that rendered RGB
must equal the input Color. SVM and OSL CPU preparation/routing checks each
passed 16 cases, including a spatial Noise Color link and finite EXR pixels.
The raw colored report is in
`build/tests/python/cycles_diffraction_glass_multiggx_tint_furnace_render_v24`.

The v25 250 nm real-film Metal white-environment furnace passed the same
neutral `0.03` gates: `[1.0032,1.0047,1.0035]` front and
`[1.0024,1.0051,0.9967]` back. The single-event back control was about
`0.920`; the return gained about `0.08` per RGB channel. Raw EXRs and report
are in `build/tests/python/cycles_diffraction_glass_multiggx_film_furnace_render_v25`.
An independent double-precision Airy reference for a higher-index film
(`n=2.4`, thickness 300 nm) compared 201 dense wavelengths with the table's
cross conductance; maximum absolute difference was `0.003436`. At one grazing
direction, direct outgoing BSDF quadrature differed from the two oriented
cache deficits by at most `0.000669`. Provenance is in
`build/tests/performance/two_sided_film_cache_v25/results.json`.

These checks cover this planar neutral/colored lossless interface model and
selected directions. They do not establish arbitrary geometry coherence,
general colored absorption, complex films, or a universal strict energy bound
for all table interpolation points.
