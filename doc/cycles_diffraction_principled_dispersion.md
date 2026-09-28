# Principled diffraction with transmission dispersion

Principled GGX transmission diffraction now uses the existing Glass/Principled
wavelength-dependent IOR calculation. The material's reference IOR still defines
its tinted F0, while the sampled-wavelength IOR defines refraction geometry and
grating phase. Constant and linked Specular Tint, colored transmission, roughness
and partial diffraction coverage remain supported. SVM and OSL share the setup.
Transmitting thin film, thin wall and multi-scattering diffraction remain unsupported.

When the sampled IOR reaches one, straight transmission is split into its own atom.
The remaining reflective proposal is conditioned on its spectral mean power;
physical values retain the unconditioned power. The regression accepts 1,976 of
2,000 reflection proposals in that case, versus 45 before conditioning.

## Numerical validation

`tests/output/diffraction/principled_dispersion_cpu_v3.log` records:

- 562,390 accepted sample/evaluate events; zero failures.
- 370,390 reciprocal comparisons; maximum relative error 3.38703e-5.
- 186,054 nonsingular flat-limit double-reference comparisons; maximum error 4.65396e-6.
- 324 facet configurations with power in [0, 1], and closure capacity checks.

The initial native-float comparison failed eight cases with maximum discrepancy
0.000310205, slightly above the existing 0.0003 threshold. Those logs are retained.
An independent double-precision calculation found the grating accurate to
5.20502e-7 in those cases, while native float evaluation differed by 0.000310448.
The final test therefore checks accuracy against an independent double reference
at the unchanged tolerance, retaining native disagreements as diagnostics.
This correction concerns numeric reference precision, not PT/BDPT brightness.

The double reference independently evaluates the flat microfacet law. It is not
a Maxwell reference for finite relief. The Fast finite-relief model remains an
approximate scalar local grating model, without general cross-object coherence.

## Reproduction

`tests/performance/cycles_diffraction_principled_dispersion_test.cpp` contains the
numerical test. `cycles_diffraction_flat_double_reference.py` independently checks
the eight logged diagnostics in Python double precision.

`tests/performance/cycles_diffraction_dispersion_package_check.py` runs the packaged
application sequentially with cleared development-resource overrides. Its four
jobs use Metal PT/BDPT/guiding and CPU OSL. Each saves a scene, linear EXR, PNG and
JSON report; adaptive sampling and denoising are disabled. These are integration
and appearance checks, not equal-error transport comparisons.

Final standalone renders passed and were visually inspected: PT256 33.2623 s,
BDPT128 101.0698 s, guided128 93.9249 s, CPU OSL32 7.4927 s. All pixels are
finite; visible noise remains. Ten support-validation cases passed their explicit
assertions. That separate process exits 1 because it intentionally triggers shader
errors; it is not represented as a zero-exit smoke. See
`tests/output/diffraction/principled_dispersion_validation_v1/report.json`.
