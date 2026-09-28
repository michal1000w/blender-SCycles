# Wavelength-dependent optical constants

The Diffraction BSDF node now accepts an embedded **Conductor n,k** text datablock.
Load a CSV in Blender's Text Editor, choose it on the node, and use the adjacent
Refresh button after editing the text. The CSV travels inside the saved `.blend`.
The ridge and absorbing substrate use this table; their constant n,k controls are
disabled while a table is assigned. Groove and upper-medium controls remain active.
Clearing the text selection restores the constant-index material.

The header must be `wavelength_nm,n,k`. Samples must be ordered, finite, passive,
and cover 380–780 nm. Conductor extinction k must be positive at every knot.
Blank lines and `#` comments are allowed. Micrometer headers, empty fields,
unresolved float knot spacing and extrapolation are rejected explicitly.

Fast mode interpolates the supplied nonuniform n,k knots at the sampled
wavelength on the GPU, then evaluates its scalar grating model. There is no
response-cache solve. The conductor reflection budget is the normal-incidence
complex-index Fresnel value; Fast diffraction-order strengths remain approximate.
The native Cycles profile API also supports tabulated ridge index/extinction and
lossless groove index for transmitting gratings. The Blender CSV input currently
exposes the conductor case, not arbitrary multi-material tables.

Realistic mode receives the same original double-precision profile samples for
its electromagnetic solver. This does not change its CPU cache-construction
backend or certify modal convergence for arbitrary relief.

Tables use Cycles' existing device lookup buffer. Dynamic registrations compare
contents and own their data until shader-manager cleanup. SVM registers tables
during compilation, after its table cleanup; this avoids invalidating freshly
prepared offsets. Parallel compilation uses the lookup-table mutex. OSL retains
stable registrations across unchanged compiled shaders. Tests exercise refresh,
recompilation, removal of tables, error reporting and recovery.

## Evidence

Blender binary SHA-256 used for the optical-table validation and benchmark:
`4e955cdb9930c32fc35564a992d89bfe9677b3ee0a247d9f16e1e831774f7a60`.
Build, incremental build and install completed. The preceding benchmarked build
and Cycles sources are preserved in `fast_integrated_storage_fixed_snapshot`.
Its earlier timing results must not be attributed to this new binary.
The later covered-grating caustics correction has a separate binary and
regression record in `cycles_diffraction_covered_regression.md`; the timing
table below remains a measurement of this preserved pre-correction build.

- `fast_index_table_numeric_v2.json`: 40,001 interpolation queries, 17 invalid
  inputs rejected, maximum n/k interpolation error 2.76265e-7; all 206 aluminum
  samples parsed. The standalone test initially lacked allocator linkage; its
  corrected build links the actual guarded allocator.
- `fast_index_device_test_v1.log`: six compiled SVM checks for separate ridge and
  groove tables, both incident sides, changed tables and return to constants.
- `fast_index_furnace_cpu_v1`: CPU SVM aluminum furnace error 3.09499e-7 against
  the independent spectral Fresnel reference; refresh and serialization pass.
- `fast_index_furnace_osl_v1`: CPU OSL passes the same checks and error bound.
- `fast_index_furnace_metal_v1`: Metal BDPT with guiding passes; aluminum error
  2.78431e-7. This is total reflected energy, not individual-order accuracy.
- `realistic_index_furnace_metal_v1`: flat Realistic aluminum control passes,
  maximum reference error 0.000257178. This includes cache approximation.
- `fast_index_recovery_v1`: malformed CSV is rejected without an output EXR;
  corrected data refreshes successfully with zero change in mean RGB versus
  the original valid render.

Furnace renders use 1,024 fixed samples, adaptive sampling and denoising off.
The recovery test uses 64 fixed samples. The independent reference interpolates
n,k in double precision, evaluates complex Fresnel reflectance and integrates
against the renderer's published color-matching/D65 tables. Refining its
quadrature from 32 to 64 subdivisions changes RGB by at most 1.21935e-7.
It does not call the packed lookup or diffraction sampler.

## Presentation fixture

`fast_index_aluminum_discs_v1/physical_discs_pt.blend` embeds the original table
and renders a 1600 nm CD beside a 740 nm DVD under two white strip lights.
The Metal PT image uses 4096 fixed samples at 960 × 691, with adaptive sampling
and denoising disabled. The preview is exported from the linear EXR with the
saved display transform; no brightness normalization is applied. Its 51.639 s
render call includes preparation and is not a warmed benchmark result.
The exact generator used is retained as `scene_generator_snapshot.py` beside
the scene. Subsequent generator changes add a separate covered-CD fixture;
they do not change this saved scene or its provenance.

Visual inspection shows broad CD spectra and narrower DVD spectra. This is an
appearance check, not validation of diffraction-order efficiency or recorded
disc pits. The Fast model still has the approximation limits documented in
`cycles_diffraction_fast_interface_candidate.md`.

## Table-lookup benchmark

`fast_index_aluminum_benchmark_v1` completed all eight sequential Metal jobs and
passed the runner's source, binary, scene and per-run setting checks. On the M5
10-core GPU at 960 × 691 and 1024 fixed samples, median warmed render times were:

| Transport | Aluminum table (s) | Constant n,k (s) | Difference |
|---|---:|---:|---:|
| PT | 8.450 | 8.494 | -0.52% |
| BDPT | 18.673 | 18.550 | +0.66% |
| Guided PT | 21.187 | 21.583 | -1.84% |
| BDPT with guiding | 38.518 | 38.162 | +0.93% |

Adaptive sampling and denoising were disabled. Each treatment has one retained
warm-up and three measured seeds. Warm-ups ranged from 8.966 to 103.499 seconds
and are not mixed into these medians. The reports do not isolate compilation
from other preparation. Both treatments use Fast diffraction; this measures
optical-table overhead, not overhead against ordinary non-spectral materials.
The small timing differences do not establish statistical significance or
equivalent-noise performance.

## Aluminum source

The fixture retains all 206 published Rakić (1995) samples from the CC0
[refractiveindex.info database at the pinned revision](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/37773dce8128ccabcd9d5108d9958d6041244661/database/data/main/Al/nk/Rakic.yml).
Only wavelength units were converted to nanometers. Original data, license,
conversion details and hashes are in `tests/scenes/diffraction/optical_constants`.
The [source paper](https://doi.org/10.1364/AO.34.004755) reconstructs intrinsic
constants from analyzed aluminum-film data; these are not raw measurements or a
complete manufactured CD/DVD coating model. Protective layers, oxide, pits and
coherent transport require separate modeling.
