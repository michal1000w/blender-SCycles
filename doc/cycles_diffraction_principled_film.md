# Principled transmitting thin-film diffraction

Historical physical-film delivery: the white-tint/zero-dispersion restriction below
is superseded by `cycles_diffraction_principled_generalized_film.md`.


Principled GGX transmission now supports physical dielectric thin film when
Specular Tint is white and unlinked, Transmission Dispersion Scale is zero and
unlinked, and Thin Wall is off and unlinked. Film thickness and IOR may be linked.
Colored Base Color and partial diffraction coverage remain supported. Tinted or
dispersive transmitting film is rejected explicitly; this update does not claim
the full generalized-film problem is solved.

SVM and OSL pass the absolute film IOR and thickness to the same coated
reflection/transmission kernel used by Glass. The native thickness cutoff is
preserved. The existing matched-index atom and conditional rough/order measure
are reused, including their bounded quadrature approximation. No new per-ray
storage or cache construction is introduced. The existing ten reserved extra
Principled slots cover the worst diffraction allocation: two reflective splits
of three slots each plus the four-slot coated transmission/atom pair.

## Validation

- Full CPU/OSL/Metal build passed.
- 23,934 sample/evaluate events passed across smooth/rough, zero/nonzero relief,
  front/back faces, and matched/unmatched indices; insufficient-storage checks
  also passed without partial closure mutation.
- An independent normal-incidence quarter-wave antireflection limit evaluated
  through the Principled setup and delta dispatch gives reflectance 0 and
  transmittance 1.
- The shared film kernel passed 73,884 independent complex Airy-series reference
  comparisons, including 14,039 evanescent-film cases. Maximum absolute error
  was 9.95359443e-5; reversal error was zero in those checks.
- Fourteen node-support assertions (including linked film thickness) passed. The deliberate-error validation process
  exits 1 after testing unsupported combinations; it is not a zero-exit smoke.

Numerical logs are `principled_film_cpu_v2.log` and
`principled_film_airy_reference_v1.log` under `tests/output/diffraction`.
The support report is `principled_film_validation_v2/report.json`.

The saved scene uses colored transmission throughout: 420 nm smooth film,
260 nm rough film, and 640 nm rough film with half diffraction coverage. The bounded
package runner uses fixed samples, adaptive sampling OFF and denoising OFF,
sequential Metal PT/BDPT/guiding followed by CPU OSL. Images and measurements
are under `principled_film_delivery_v1`, with binary provenance in its manifest.

These checks validate the implemented scalar local model and node integration.
They do not establish finite-relief Maxwell accuracy, general cross-object
coherence, complete multi-scattering transport, or exhaustive pipeline support.

The packaged PT256/BDPT128/guided128/CPU OSL32 renders passed in
33.7725/104.0044/94.9382/8.4181 seconds respectively. All were visually
inspected and have finite linear pixels. These different-sample timings do not
rank transport methods at equal quality. Package executable SHA-256: `b46643a7be1c36c3ccdc5c9667e816a903ef1134e442813b0bc8b9104c3327d5`.
