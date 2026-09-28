# Aluminum optical constants for diffraction validation

`Al_Rakic_1995.yml` is an unchanged copy of the refractiveindex.info aluminum
Rakić (1995) table. The upstream dataset is CC0; its license is retained in
`LICENSE_CC0.txt`. Exact revision URLs and SHA-256 hashes are in
`provenance.json`.

Source paper: A. D. Rakić, *Algorithm for the determination of intrinsic
optical constants of metal films: application to aluminum*, Applied Optics
34, 4755–4767 (1995), https://doi.org/10.1364/AO.34.004755 .
Database: https://refractiveindex.info/?shelf=main&book=Al&page=Rakic .

`Al_Rakic_1995_nm.csv` preserves all 206 samples and their n,k values. Only
wavelength units change, from micrometers to nanometers by exact decimal
multiplication by 1000. The range is 0.12399–200000 nm, covering the current
380–780 nm rendering domain. No extra samples, fitted values or smoothing
were introduced. The host grating solver's interpolation policy is separate
from this source data.

These are published reconstructed intrinsic optical constants based on
analysis of aluminum-film optical and electron-energy-loss data. They are
not raw measurements, a guarantee for a particular manufactured film, or a
complete CD/DVD coating stack. Oxide, protective layers, film thickness and
actual groove geometry require separate modeling and validation.

This fixture is now connected to the Diffraction BSDF through its embedded
Conductor n,k text input. CPU SVM, CPU OSL, Metal BDPT/guiding and flat Realistic
furnace checks are recorded in `doc/cycles_diffraction_optical_constants.md`.
The existing synthetic-spectrum tests remain independent regression cases.

The `tests/metal/cycles_diffraction_test.py --scene-sample` harness now loads
this CSV and verifies its hash against the provenance manifest. It uses the
spectrum for both ridge and substrate at CD/DVD pitches, comparing GPU
samples at 520/550/580 nm against explicit fixed-index solver references.
These are numerical GPU tests, not rendered disc-scene validation.
