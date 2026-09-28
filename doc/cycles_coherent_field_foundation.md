# Coherent scalar field foundation

`intern/cycles/kernel/light/coherent_field.h` defines the complex field and
Fresnel contract used by the bounded paired-path connector. The current
renderer evaluates ideal planar mirror candidates at marked detectors; a
separate explicit vector mode for dielectric candidates is in integration.

A field record holds a complex scalar amplitude and source phase in cycles.
Its amplitude must contain source strength, geometric spreading, the chosen
detector's complex response, and every interface amplitude in one consistent
polarization gauge. For a set of complete paths ending at the *same detector
sample*, the connector adds each path's power and each unordered pair's signed
cross term:

```
2 exp[-(ΔOPL/Lc)^2/2] Re[Aa conj(Ab) exp(i 2π(ΔOPL/λ + φa - φb))].
```

This includes two arms from one source. Zero coherence length removes every
cross term. The detector used in the standalone mirror reference is an ideal
scalar coherent Lambertian detector, with angular amplitude proportional to
`sqrt(cos(theta)/pi)`. A generic Cycles diffuse BSDF does not define such a
complex phase response. Host validation admits only a marked single-closure
Lambertian detector for this estimator. Its response can be passive RGB Color
or a texture, including an exactly pure-diffuse Principled subset. Receiving
meshes may be nonplanar and faceted, but every receiving triangle must be flat
shaded, without custom corner normals, and with zero object shading-terminator
offset. Smooth shading and terminator offsets change native directional response
or shadow visibility and are not represented by this field model. Pure diffuse
Principled requires every other lobe disabled, opaque Alpha, and no effective
film (unlinked finite thickness at most 0.1 nm); main Roughness is unused and
may retain its default. UI Diffuse Bounces may be zero: the first receiver's
complete local field is evaluated before its diffuse continuation terminates.
Allowing subsequent diffuse encounters adds ordinary detector/mirror
interreflection beyond a reference that models only the first receiver.
The coherence
wavelength controls phase; these source and detector envelopes remain RGB,
not monochromatic spectral radiometry. The detector multiplies every diagonal
and pair by its RGB albedo at the same sampled surface point. Linked colors
are bounded to [0,1] before closure allocation and native scattering as well
as coherent evaluation; nonfinite channels become zero. Constants outside this
range are rejected by the host. Black detectors may allocate no closure and
own zero energy. A shared runtime Lambertian predicate controls field evaluation,
direct NEE ownership and eligible light-prefix retention/termination, so filtered
or non-Lambertian endpoints retain ordinary transport.

The interface helper supplies separate scalar s and p channels for a lossless,
nonmagnetic planar dielectric. It uses tangential electric field on both sides:
the real admittances are `n cos(theta)` for s and `n/cos(theta)` for p. Reflection
is `(Yi-Yt)/(Yi+Yt)`, and the propagating transmission amplitude is flux
normalized as `2 sqrt(Yi Yt)/(Yi+Yt)`. Therefore `|r|^2+t^2=1` for each channel.
Under total internal reflection the transmitted propagating amplitude is zero
and the complex reflection phase remains. At the critical angle, grazing
incidence, and matched-index grazing incidence, explicit limiting values avoid
zero-divisions. Invalid IOR or cosine inputs return `valid=false`. The two
polarization channels are separate approximations; a full renderer must track
and rotate polarization bases when a path changes plane of incidence. The
utility now includes two-component Jones field rotation between transverse
frames on a shared propagation segment and application of diagonal s/p
interface factors. The caller must construct consistent incoming and outgoing
frames and supply the p tangential-sign conversion: reflection flips it when
the chosen p basis is `cross(s, direction)`, while ordinary transmission
preserves it. Normal-incidence frame selection and frame transport through
arbitrary geometry belong to the connector. The current ideal mirror test uses
a common unit-reflection phase, which cancels
between its two arms. Absorbing metals, rough interfaces, gratings, and generic
shader mixtures require their own complex amplitude models.

The optional vector path model transports three mutually independent,
equal-power world-axis dipoles projected onto each launch direction. Matching
mode indices remain mutually coherent across sources in the same declared
group; different dipoles do not interfere. The detector compares full electric
vectors in a common local frame. This explicit source ensemble is one possible
unpolarized coherence model, not a universal model of every lamp. Scalar mirror
mode remains the default approximation, and Glass requires selecting vector
mode explicitly.

The pair API accepts *optical path difference*, not two accumulated float
lengths. The connector's planar geometry solver uses split-float length and
pair subtraction, validated on CPU and Metal fast-math for its supported path
sequences. The standalone test shows that subtracting two ordinary `float`
paths of about 3 m loses 0.183 cycles from a quarter-wave difference at 550 nm.
Vector path pairing also reduces the high component of a compensated OPD
modulo the wavelength before adding its low component. At an 18 mm arm
difference this removes up to 0.00141 cycles of lost low-part phase in the
standalone double comparison. Converting exact UI 550 nm to a lone float
wavelength would still introduce about 0.00110 cycles of systematic phase at
that arm difference. The Blender light sync therefore retains a second float
for the wavelength residual, and both components enter a fused quotient
remainder. At a 1 m arm difference, the standalone double test across scene
unit scales `0.01`, `1`, and `100` now measures at most 6.47e-8 cycles error.
Arbitrary nonplanar routes still need a separate precision solution. Scene-unit
geometry and the UI optical wavelengths use the same scene-unit convention;
Blender sync converts physical UI wavelengths and coherence lengths to scene
units before evaluating phase.

`tests/performance/cycles_coherent_field_test.cpp` compares the float utility
to independent double-complex calculations for direct and unfolded mirror
paths, source phase reversal, partial and zero coherence, same-source arms,
lossless transmission/reflection, complex TIR, reciprocity, critical/grazing
limits, non-coplanar Jones frame changes, basis-invariant pair interference,
and scene-unit scales `0.01`, `1`, and `100`. It is a numerical utility
test, separate from rendered multipath acceptance. The renderer now enumerates
bounded source-to-planar-mirror-to-detector paths, validates their exact BVH
segments, and writes complete diagonal plus pair intensity. All-reflection
sequences in one medium use exact planar image-source intersections and their
analytic geometric spreading, avoiding Newton dropout as two reflections
approach a shared corner. Their optical phase still uses compensated physical
segment lengths. Any transmission retains the stationary optical-path solver. It removes the
same source/path class from ordinary NEE and BDPT to avoid double counting;
unrelated indirect paths retain their estimators. The independent 512-pixel,
128-sample phase-0/phase-pi/incoherent mirror gates passed without reference
fitting or tolerance changes. This does not yet establish dielectric rendering
or arbitrary scattering phase.

Run `python3 tests/python/run_cycles_coherent_field_test.py`. It saves the
compile command, source hashes, binary hash, and numerical result under
`build/tests/performance/coherent_field_v1`.
