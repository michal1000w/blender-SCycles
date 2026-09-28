# GPU response-builder investigation, 2026-09-27

Production remains unchanged. Fast has no material-response cache; Realistic
still constructs its response on the host. The previous turn completed the
high-sample image review, so this work proceeds with no live rendering job.

Inspection of `intern/cycles/scene/diffraction.cpp` confirms the finite-depth
path constructs factorized P,Q matrices, diagonalizes P Q, propagates decaying
modes, and solves the exterior boundary system. Zero-depth and exact zero-ky
special cases are already implemented. Replacing the cache upload alone would
not move this expensive construction onto the GPU.

Research sources inspected:

- [Yang, Song and Cai, August 2026 preprint, v2](https://arxiv.org/abs/2608.06185):
  a boundary-field cascade avoids internal eigendecomposition. Its GPU results
  motivate a candidate; they do not establish performance or precision on M5.
- [RigorousCoupledWaveAnalysis.jl](https://github.com/jonschlipf/RigorousCoupledWaveAnalysis.jl):
  documents both a hybrid CUDA path with CPU eigen fallback and an eigen-free
  Taylor alternative, citing Xu and Charlton (IEEE Access, 2024). A hybrid
  implementation must not be described as entirely GPU cache construction.

No external implementation was copied. The new standalone NumPy experiment
constructs the same isotropic Fourier-factorized matrices as production. It
initializes a small-step transfer operator using a 12-term exponential series,
converts it into scattering form, then doubles with shared-port elimination.
The independent reference uses decaying modal amplitudes anchored at opposing
boundaries. Comparison includes the complete complex operator, in an artificial
E +/- H basis. These are not physical flux-normalized ports and this is not a
Maxwell convergence test, a cache, or a GPU implementation.

There are 48 cases: half-order counts 2,4,8,16; dielectric 1.5 and absorbing
0.9+6i ridges; ky=0 and 0.27; depths 0,150,1200 nm. Pitch is 740 nm, wavelength
580 nm, duty 0.41, and base momentum -0.2286. Gates fixed before execution:
maximum complex component error below 1e-9 in double and 3e-4 in float.

The first implementation passed double precision (maximum 5.73438e-10), but
failed float (0.1025304). Its source snapshot and report are preserved as
`tests/output/diffraction/boundary_cascade_algebra_v1.py` and `.json`.

A second candidate stores transmission-minus-identity increments to reduce
small-step cancellation. It failed both gates: maximum double error
1.73231e-7 and float error 8943.92845. Its current source is
`tests/performance/cycles_diffraction_boundary_cascade.py`; report is
`tests/output/diffraction/boundary_cascade_algebra_v2.json`. Each report records
its source hash. Both failures remain failures; no thresholds were relaxed.
The cause of the second failure is not yet established. Check recurrence
algebra against the first version at every doubling before attributing it to
conditioning or float arithmetic.

Next required work: isolate first divergent doubling, compare an exterior-
admittance port basis against the artificial basis, and validate complete
physical Jones operators against the production solver. Only a candidate that
passes those checks should proceed to Metal construction and end-to-end cache
integration. Large Fourier counts, cutoffs, deep absorbing relief, substrate
incidence, interpolation validation, and actual GPU timings remain unproven.

## Air-admittance basis and doubling trace

Subsequent experiments preserve v2's source as
`tests/output/diffraction/boundary_cascade_algebra_v2.py`. Version 3 changes
both the port basis to the homogeneous air admittance and the increment
strategy: transmission increments are used only while small, then converted
to ordinary transmission matrices. It passes the double gate with maximum
1.74166e-11, but eight of 48 float cases fail; maximum float error is
0.00541109. Because the basis changed, raw errors are not directly comparable
to the earlier artificial-basis errors. The v2 failure's precise cause is not
established by changing two things together.

Version 4 uses a 20-term series and initial norm at most 1 instead of a 12-term
series and norm at most 0.125, reducing the number of doublings. It still fails
float: maximum 0.00537865. Version 5 adds an independent modal reference with
float-rounded P,Q inputs computed in double precision. The largest input-
quantization error over all cases is only 3.82026e-6. This is not enough to
explain the observed cascade error.

Version 6 adds a per-doubling trace for the worst float case: half-orders 16,
metal 0.9+6i, ky=0, depth 1200 nm. Each partial thickness is compared with a
fresh modal solution. Initial error is 5.77489e-8. At doubling 10 it is
0.000248573; at doubling 11 it is 0.00118092, with feedback-system condition
number about 562.5. Final error is 0.00537865. That case's input-quantization
error is 2.32218e-6. The trace supports accumulated propagation/solve error;
it does not prove which floating-point operation dominates it.

Reports v3 through v6 and source snapshots v3 through v5 are preserved under
`tests/output/diffraction/`; current source corresponds to v6 and its hash is
recorded in that report. All retain the original gates and failing status.
No Metal speedup or production correctness is claimed. Next investigate a
better-conditioned reference admittance and compensated matrix arithmetic,
then convert to physical, flux-normalized exterior ports before assessing
actual diffraction powers. Full physical and large-order validation remain
necessary even if this algebraic comparison eventually passes.

## Internal-basis comparison with common exterior ports

`tests/performance/cycles_diffraction_boundary_basis.py` compares four internal
admittances: air, index 1+0.2i, index 1+1i, and the square root of the mean
permittivity. Every result is converted back to the same air exterior ports by
solving the incoming/outgoing basis relation. Thus these errors can be compared
across internal bases. Candidate basis transforms also run in candidate precision.
The NumPy matrices constructing material P,Q remain double before quantization;
this is not yet an entirely single-precision material-construction test.

Version 1 compares all complex components. Version 2 also constructs propagating
TE/TM-equivalent transverse frames normalized to unit flux and compares Jones
amplitudes and power columns. These fixtures are finite layers between air on both
sides, not the production metal-substrate geometry. Modal truncation is shared
with the reference, so none of this establishes Fourier convergence.

All four bases pass the unchanged double component gate. Float results:

| Internal basis | Failed full-operator cases / 48 | Maximum component error | Maximum Jones error | Maximum power-column L1 |
| --- | ---: | ---: | ---: | ---: |
| Air | 7 | 0.0054340 | 0.0012882 | 0.0020117 |
| 1+0.2i | 6 | 0.0040938 | 0.0022092 | 0.0040985 |
| 1+1i | 6 | 0.0010060 | 0.0009128 | 0.0016387 |
| Mean permittivity | 4 | 0.0026345 | 0.0004466 | 0.0007904 |

The errors therefore affect propagating light as well as evanescent ports.
Changing basis alone does not pass the full-operator gate, and minimizing that
error does not necessarily minimize physical power error. No candidate is
promoted. The physical columns are additional diagnostics, not replacement gates.
Double-precision lossless maximum-power-gain deviations are below 4.1e-13;
float maximum gains across these cases range from 1.000214 to 1.000469 by basis.
These diagnostics do not justify renormalizing the response or hiding its error.

Reports are `tests/output/diffraction/boundary_basis_v1.json` and `_v2.json`;
the first script is preserved as `boundary_basis_v1.py`. Reports identify both
the driver and cascade helper by SHA256. Both runs are terminal failures with
exit code 1, as required by the unchanged full-operator gate. No GPU job was run.

## Arithmetic attribution and compensated-product prototype

`cycles_diffraction_boundary_precision.py` instruments the existing diagnostic
through an AST transform of matrix products and solve calls. Four CPU treatments
retain complex64 matrix storage: unchanged arithmetic, products accumulated in
double then rounded, solves with two double-residual refinement steps, and both.
This is attribution only; it does not assume native Metal double arithmetic.
`boundary_precision_v1/summary.json` records all cases and both script hashes.
Its original instrumentation driver is preserved as `driver.py` in that directory.

For the mean-permittivity basis, more accurate product accumulation reduces
full-operator float failures from four to one, with maximum error 0.000381055
and maximum physical power-column L1 0.000248035. It still fails the unchanged
component gate. Solve refinement changes none of these reported results;
combining it with product accumulation reproduces the product-only result.
The air basis still has three failures with accurate products. This identifies
matrix product accumulation as significant, without proving it is the only error.

A new float-only CPU prototype, `cycles_diffraction_compensated_product.py`, uses
binary32 split products and compensated sums, then returns complex64 matrices.
It uses no double arithmetic inside the product function. Three deterministic
random matrix checks (sizes 3,17,66) exactly match double-accumulated products
rounded to complex64. These are smoke checks, not broad error bounds. Inputs
near overflow and subnormal behavior remain unvalidated. A Metal FMA variant
would need separate implementation and tests; no GPU speed claim follows.
The full four-basis numerical comparison using this prototype was launched
separately; its terminal result must be checked before drawing conclusions.

The full compensated CPU run is terminal. Its summary exactly matches the
higher-precision product diagnostic, including the one remaining failed case
for the mean-permittivity basis. The instrumentation runner exits zero after
successfully collecting treatments, not because all numerical gates passed;
read the per-treatment failures. All four recorded source hashes verified.

## First actual Metal compensated-product validation

New standalone sources:
`tests/metal/cycles_diffraction_compensated_product.metal` and `.mm`.
The shader uses explicit float FMA product residuals and compensated summation.
It is compiled with safe math: algebraic reassociation would invalidate the
error-free transforms. No double arithmetic occurs on GPU. Ordinary products
are tested under the same math mode as a diagnostic comparison.

The isolated outside-sandbox M5 run completed successfully. Four batches of
sizes 3,17,66,132 include random, reciprocal-scaled, and nearly cancelling
inputs. CPU complex-double accumulation rounded to float is the reference.
The fixed product-kernel tolerance is 2e-6 + 1e-6 times reference magnitude;
this is separate from, and does not replace, the Maxwell-operator gate.
All five dispatches per size are checked; the first two are timing warmups.
Compensated products have zero failed entries. Maximum complex error at size
132 is 9.09495e-13. Ordinary products have 25 failed comparisons there across
five dispatches (five entries repeated), maximum error 1.04904e-5.

Raw report: `tests/output/diffraction/metal_compensated_product_v1.json`.
Source, binary and output hashes recorded after the run in the accompanying
`_audit.json`; these are not falsely described as a pre-launch freeze.
Recorded GPU timings are variable: the three size-132 compensated dispatches
are approximately 1.031,1.045,1.004 ms for four matrices. Ordinary dispatches
range 0.325–3.464 ms. No speed ratio is inferred. This is a simple one-output-
element-per-thread kernel, not a tuned tiled multiplication or cache benchmark.

This validates a useful GPU arithmetic building block, not the Realistic cache
builder. Residual propagation error, complete GPU matrix solves, material matrix
construction, physical substrate interfaces, adaptive cache generation and
end-to-end correctness/performance remain unfinished. Safe math isolation must
be preserved if this kernel is integrated alongside Cycles fast-math kernels.

## Fused-update, initialization and held-out diagnostics

Two further CPU attribution runs keep accurate product accumulation and the
original gates. `boundary_precision_fused_v1` avoids intermediate product
rounding in directly adjacent matrix product/add or subtract expressions.
It does not fuse arbitrary expression trees. Mean-permittivity maximum error
is 0.000376761 with two failed cases; this does not fix the problem.
`boundary_precision_initial_double_v1` initializes the thin layer in double,
then stores its scattering matrices in float before the doubling loop.
Mean-permittivity maximum error is 0.000372872 with one failed case. Neither
candidate is promoted. Both use CPU doubles only as attribution diagnostics.
The pre-change fused driver is preserved inside its result directory.

A fresh 128-case validation (`cycles_diffraction_boundary_heldout.py`, seed
928417) uses a fixed mean-permittivity basis and double-accumulated products
rounded to float, the arithmetic already reproduced by the compensated CPU
prototype on the original cases. It varies pitch 600–1800 nm, wavelength
380–780 nm, depth 20–1200 nm, duty 0.2–0.8, both signs of ky, Fourier half-orders
2/4/8/16, and dielectric/absorbing ridge indices. There is no oracle choice of
basis or tolerance change. Like preceding diagnostics, the finite layer has
air exterior media and a shared finite Fourier truncation with its reference.

The held-out run is terminal with exit 1. Double has zero failures, maximum
component error 4.68847e-12. Float has one failure, maximum component error
0.000407784 and maximum physical power-column L1 0.000416258. Raw cases and all
four source identities are in `tests/output/diffraction/boundary_heldout_v1.json`.
This expands evidence beyond the original 48 fixtures but still does not pass
the complete gate, establish modal convergence, or validate a full GPU solver.

## Larger initial interval: first passing diagnostic candidate

A single global change uses initial generator norm at most 8 (previously 1)
and a 40-term exponential series (previously 20), reducing the doubling count
by three where applicable. No per-case reference-driven choices or gate
changes are made. Accurate product accumulation remains enabled. This tests
whether reducing accumulated float storage/propagation error outweighs the
longer series; it does not replace the scattering cascade with an unstable
full-thickness transfer matrix.

`boundary_precision_wide_step_v1` passes the original 48 cases in both
precisions and all four bases. Maximum float component errors by basis:
air 0.000189651, weak absorption 0.000120227, absorbing 0.0000885441,
mean permittivity 0.0000642782. This uses CPU double-accumulated products rounded
to float, not a complete GPU solver. Original driver snapshot is retained.

`boundary_heldout_wide_step_v1.json` passes all 128 held-out cases in the fixed
mean-permittivity basis, with unchanged gates. Maximum double component error
is 9.34008e-13; maximum float component error is 3.60927e-5; maximum float
physical power-column L1 is 2.82389e-5. These are same-truncation comparisons,
not evidence of converged Maxwell accuracy.

Two follow-ups were launched: the original 48-case/four-basis test using actual
float-only compensated products (`boundary_precision_wide_compensated_v1`),
and 1024 new held-out cases with seed 761924 using accurate-product diagnostics
(`boundary_heldout_wide_step_1024_v1.json`). Check terminal results before
claiming either passes. They may overlap because neither is a performance
measurement. No GPU job is active during these CPU numerical checks.

The 1024-case diagnostic is now terminal, exit 0, with all source hashes
verified. Both gates pass. Maximum double component error is 1.06872e-11;
maximum float component error is 0.000249700 and maximum float physical
power-column L1 is 0.000154827. The float result is below, but fairly close to,
the unchanged 0.0003 component gate. Do not infer reliability at higher Fourier
counts, near exact cutoffs, different exterior substrates, or broader optical
constants from this coverage. The float-only compensated follow-up remains
separate and must still be checked for completion.

## Compensated candidate completion, higher orders and physical substrates

All numerical jobs in this section are terminal. The original 48-case,
four-basis run with actual float-only compensated matrix products completed
successfully (`boundary_precision_wide_compensated_v1`). Both precision gates
pass in every basis. Mean-permittivity maximum float component error is
6.42782e-5, matching the corresponding double-accumulation diagnostic.
Source identities were verified after completion. This describes the matrix
products: material P,Q construction and some analysis still use host doubles,
and linear solves still use NumPy. It is not an all-float or all-GPU builder.

A new independent seed (472816) with 24 cases at half-orders 32 and 64 also
passes the accurate-product diagnostic. Maximum double component error is
1.23518e-10 and maximum float error is 0.000127178; maximum float power-column
L1 is 0.000168760. Report: `boundary_heldout_wide_step_high_orders_v1.json`.
The previous held-out driver is preserved alongside the 1024-case report;
the updated driver accepts explicit Fourier counts instead of changing defaults.

`cycles_diffraction_boundary_substrate.py` adds differing exterior admittances.
Its separate reference constructs production-style decaying modal amplitudes
and solves both boundary equations. The candidate converts from its internal
mean-permittivity basis to those exterior ports. It compares complete complex
operators and separately computes unit-flux reflected Jones operators for
propagating upper-medium channels; it does not assign flux normalization to
fictitious incoming waves in an absorbing substrate.

The 48 fixtures use metal index 0.9+6i, a 150 nm binary relief with duty 0.41,
metal substrate, upper/groove index 1 or 1.58, pitches 740/1600 nm, wavelengths
450/580/700 nm, ky 0/0.27, and half-orders 16/32. This covers bare and
plastic-filled metal relief, but is not a macroscopic covered-disc render.
Both gates pass: maximum double component error 2.16158e-12, float error
3.00909e-5, and float reflected power-column L1 3.26280e-5.
Report: `tests/output/diffraction/boundary_substrate_v1.json`. All source hashes
and case counts verified for this run and the higher-order run.

These results justify developing the complete GPU numerical pipeline next.
They do not establish modal convergence, cutoff robustness, arbitrary material
coverage, CPU/GPU end-to-end equivalence, or actual cache-build performance.
Only the standalone compensated product has executed on Metal so far. The
production renderer and the selected Fast build remain unchanged.

## First actual Metal pivoted complex solve

New files `tests/metal/cycles_diffraction_complex_solve.metal` and `.mm`
implement a batched complex Gauss-Jordan solve with row pivoting. Each matrix
and all RHS columns are owned by one threadgroup; global-memory workspace is
synchronized after row swaps, normalization and elimination. Explicit complex
FMA updates and scaled complex division run under safe math. Singular and
nonfinite outputs receive failure status; there is no silent regularization.

The outside-sandbox M5 test is terminal, exit 0. Four sizes (3,17,66,132), each
with five matrices and n RHS columns, include diagonally dominant complex
systems, uniform scales 1e-8 and 1e8, a permutation requiring pivoting, and an
exactly rank-deficient system. All five dispatches are validated per size.
Every singular matrix is rejected. Valid systems have maximum solution error
2.36173e-6 against known generating solutions; maximum relative residual
computed in host double from the actual rounded A,B inputs is 4.96948e-7.
Fixed gates are solution error 2e-5 and relative residual 2e-6.

Report: `tests/output/diffraction/metal_complex_solve_v1.json`; identities are
recorded after execution in `_audit.json`. At size 132, three measured GPU
batches after two warmups took 7.619,5.977,6.099 ms. This is an untuned
standalone solver and no speedup or cache-build performance is inferred.
It must next be tested against matrices extracted from the actual propagation
steps, including ill-conditioning and multiple RHS shapes. Complete GPU
propagation, material construction, quality/error handling, cache publication
and Blender integration remain unfinished. The production renderer is unchanged.

## GPU solves of captured diffraction systems

`cycles_diffraction_solve_fixture.py` now exports actual complex64 A,B matrices
from the larger-step propagation candidate. It records the initial admittance
solve, thin-layer boundary solve, and subsequent feedback solves. Fixtures are
a dielectric layer (N8, 600 nm depth, ky=0.27), deep metal (N16, 1200 nm depth,
ky=0), and plastic-filled metal (N32, 150 nm depth, ky=0.27), all at pitch 740 nm
and wavelength 580 nm. There are 29 systems, maximum measured condition number
102.272. Reference solutions are recomputed in CPU double from the exact
exported float inputs, then rounded to float. Generator and dependencies are
identified in `metal_solve_propagation_fixture_v1/manifest.json`.

New `tests/metal/cycles_diffraction_complex_solve_fixture.mm` reads that binary
fixture and runs the existing Metal solve kernel, checking all five dispatches
per system. It preserves the existing solution-error 2e-5 and relative-residual
2e-6 gates. The outside-sandbox M5 run is terminal, exit 0, with zero failures.
Maximum solution error is 1.07949e-5 and relative residual is 1.30478e-6. All
29 record dimensions and fixture/source hashes verified. Raw report is
`tests/output/diffraction/metal_solve_propagation_v1.json`, with an accompanying
post-run source/binary/fixture/output identity audit.

This is stronger than the initial synthetic test but still tests each solve
independently: GPU output is not yet fed into subsequent propagation operations.
The next required validation is the connected GPU multiplication/solve pipeline
against complete complex scattering operators. Host fixture construction, full
cache generation, speed, and Blender integration are not claimed complete.

## Connected Metal propagation

New binary RPC harness (`cycles_diffraction_matrix_rpc.mm` and `.metal`) exposes
rectangular compensated products and pivoted solves to the numerical driver.
`cycles_diffraction_metal_propagation.py` routes every candidate complex64
matrix product and solve through that Metal process, with no CPU arithmetic
fallback for these operations. Each GPU result feeds the following step.
Material matrices, elementwise operations and orchestration remain on CPU;
this is a connected GPU linear-algebra diagnostic, not full GPU cache building.
Sources and executable are hashed before launch and rechecked afterwards;
complete actual/reference operators are retained as NPZ artifacts.

The M5 run `metal_connected_propagation_v1` completed with all six cases passing:
N8/N16 dielectric, deep metal, and covered metal, with differing substrates.
It executes 582 GPU products and 69 GPU solves. Maximum component error is
0.000107831, below the unchanged 0.0003 gate. The original driver is preserved
inside that output directory before adding configurable Fourier counts.

`metal_connected_propagation_high_orders_v1` executes 702 GPU products and 84
GPU solves for N32/N64 versions of the same fixtures. It is terminal with exit
1: all N32 cases pass, but the N64 deep-metal case fails at 0.00254888 component
error and 0.00243906 reflected power-column L1. N64 covered metal passes narrowly
at 0.000287383 component error. All source and artifact identities verified.
No tolerance was changed and this failure prevents promoting the solver.

`cycles_diffraction_metal_propagation_compare.py` re-evaluates the identical
profiles using the CPU accurate-product float algorithm and compares against
the saved GPU operators and common modal reference. N64 deep-metal CPU error is
0.000293913, versus GPU error 0.00254888; GPU/CPU difference is 0.00235381.
For N64 covered metal the CPU error is 1.18737e-5 versus GPU 0.000287383.
This points to a remaining GPU arithmetic/solve contribution, while the CPU
algorithm itself is near the gate on the deep-metal case. The comparison alone
does not identify a specific kernel defect. Raw results are in
`metal_connected_propagation_high_orders_v1_cpu_comparison.json`.

Next investigate GPU residual refinement and per-stage solve accuracy, retaining
the original operators and gates. Standalone solve success on N<=132 systems
did not establish end-to-end accuracy for the larger N64 systems (258 modes).
No cache-build or render-speed measurement is inferred from RPC wall time.

## Connected GPU residual refinement

The failed high-order version's shader, host RPC executable/source and driver
were preserved in `metal_connected_propagation_high_orders_v1/source_snapshot`.
The new RPC operation computes B-A X with compensated products on Metal, starting
the compensated sum from B before the final float rounding. This avoids losing
the small residual by subtracting an already-rounded product. Two correction
solves and float solution updates are applied per candidate solve. Products,
residuals and correction solves remain on GPU; host array updates remain CPU.
There is no hidden CPU linear solve fallback.

`metal_connected_propagation_refined_v1` is terminal, exit 0. All six N32/N64
connected cases pass the unchanged 0.0003 component gate. Source and NPZ hashes
verified after completion. The N64 deep-metal component error drops from
0.00254888 to 0.000293913, narrowly passing. Its reflected power-column L1 is
0.000333830; this diagnostic is not mislabeled as a 0.0003 power-error pass.
N64 covered-metal component error drops to 1.18737e-5.

The CPU comparison report is
`metal_connected_propagation_refined_v1_cpu_comparison.json`. The largest
GPU/CPU component discrepancy across these six cases is 1.97056e-6; most are
much smaller. This supports that refinement removes the dominant additional
GPU solve error for these fixtures. It does not remove the underlying float
propagation error, prove performance, or establish general cutoff/higher-order
robustness. Cache integration and all originally identified remaining scope
remain open. The production Fast renderer has not been changed by this work.

## Constitutive linear algebra on Metal

The connected driver now has `--gpu-constitutive`. In that mode the material
permittivity inverse, inverse-profile factorization solve, and constitutive
matrix products execute through the GPU solver/product kernels. Fourier
coefficients, index arithmetic, diagonal/block assembly and elementwise array
operations remain host-side. The independent modal reference remains CPU double.
The earlier refined driver is preserved in its result directory.

`metal_connected_constitutive_v1` is terminal, exit 0, with all six N32/N64
cases passing the same component gate. The N64 deep-metal error is 0.000293080;
its reflected power-column L1 is 0.000332121. The latter is a separate reported
diagnostic, not a claimed 0.0003 power-error pass. N64 covered-metal component
error is 1.24310e-5. This extends which dense operations run on GPU but remains
a host-orchestrated test and not a production cache implementation.

A subsequent frozen run, `metal_connected_constitutive_random_v1`, uses twelve
new material/index/depth/ky profiles (seed 841291) at both N32 and N64, with
GPU constitutive preparation and two residual refinements. Pitch/wavelength
remain 740/580 nm and duty 0.41, so these are not full-domain coverage tests.
This run must be checked for terminal completion before claiming its results.
The constitutive v1 driver is preserved before adding random-case controls.

The randomized constitutive run is now terminal, exit 0. All 24 cases pass,
with maximum component error 0.000156048 and maximum reflected power-column
L1 0.000182897. It executes 2816 GPU products, 1119 GPU solves and 746 GPU
residual evaluations. All frozen source/executable identities and 24 saved
operator artifact hashes verified. The earlier six-case result is also
preserved. These results support the connected numerical implementation in
the tested region; broad wavelength/pitch/cutoff coverage, device-resident
execution, adaptive response-cache publication, Blender integration and actual
build-time benchmarking remain outstanding.

## Connected spectral/material parameter coverage

The driver now supports spectral randomization in addition to material/depth/
incidence randomization. Pitch, wavelength, duty, and base transverse momentum
are recorded per case and used consistently in constitutive construction,
exterior admittance, modal reference, and propagation. The old randomized
driver was preserved in its result directory before this extension.

`metal_connected_spectral_random_v1` is terminal, exit 0. Sixteen fresh profiles
(seed 917835; spectral generator seed+1) are evaluated at N32 and N64: 32 cases,
all passing. Sampling ranges are pitch 600–1800 nm, wavelength 380–780 nm, duty
0.2–0.8, base momentum -0.35–0.35, ky -0.6–0.6, upper/groove index 1–1.6,
depth 20–1200 nm, and dielectric/absorbing ridge/substrate indices. These are
sampled ranges, not guaranteed extrema, cutoff, or continuous-domain coverage.
Maximum complex component error is 8.31126e-5 and maximum reflected power-column
L1 is 7.71525e-5. It executes 3464 GPU products and 1383 GPU solves. All source,
executable, and saved operator hashes verified after completion.

The CPU comparison helper now reads per-case spectral/profile parameters and
explicitly labels its double-material-construction scope. Its original source
was recovered byte-for-byte, checked against the previous report hash, and
preserved as `tests/output/diffraction/metal_propagation_compare_v1.py`.
No old comparison results were rewritten.

Production integration inspection confirms Realistic preparation currently
calls `DiffractionManager::get_or_build` from shader preparation, invoking the
host cache builders before device upload. A GPU backend must enter response
construction there, preserve cancellation and complete-cache publication, and
supply the reference-port convention required by existing cache interpolation.
Merely accelerating upload or attaching the diagnostic RPC would not complete
that integration. Resident execution, backend integration and build-time
benchmarks remain outstanding; no production renderer change is claimed here.

## Production reference-port convention at cutoffs

`cycles_diffraction_reference_cutoff_fixture.cpp` links the current production
`intern/cycles/scene/diffraction.cpp` and calls
`diffraction_grating_solve_reference` directly. It exports 24 complex operators
at N16/N32 with retained half-orders 3, metal or dielectric ridge/substrate,
pitch 1000 nm, wavelength 580 nm, 150 nm depth and duty 0.41. At ky=0, bases
0.42 and 0.34 place an order at the upper-medium index-1 or dielectric
substrate index-1.5 cutoff; each is sampled at offset -1e-6,0,+1e-6. Some
metal cases have no physical substrate cutoff but exercise the same coordinates.
Production reference operators and source/binary hashes are retained in
`reference_cutoff_production_v1.json` and its provenance file.

`cycles_diffraction_metal_reference_ports.py` now implements the actual cache
reference-port admittance convention: replace retained exterior channels by
the fixed reference admittance, retain physical admittance outside them, and
do not create lower reference ports for an absorbing substrate. Avoid computing
singular physical admittance in channels replaced by the reference convention.
It converts the connected GPU operator to the production interleaved retained
port order before comparing complete complex amplitudes. This is not a
flux-normalized physical power test at exactly grazing incidence.

The outside-sandbox M5 run `metal_reference_cutoff_v1` is terminal, exit 0;
all 24 cases pass the unchanged 0.0003 component gate. Sources, fixture identity
and saved operators verified after completion. This directly checks agreement
with the production C++ reference-port solver, rather than only the NumPy modal
formulation. It does not validate interpolated cache cells, modal convergence,
or renderer behavior at cutoffs. The dielectric reference blocks contain 28
polarization channels, exceeding the existing 20-channel production evaluator
capacity; this test does not change or bypass that renderer limitation.

## Device-resident matrix-operation chain

`tests/metal/cycles_diffraction_resident_matrix.h` adds a C++ Metal matrix
engine with resident multiply, compensated residual, pivoted solve, refined
solve, and addition. The accompanying shader supplies device-side workspace
packing, solution extraction and addition. GPU outputs feed later operations
without round trips through Python or host matrix memory. Commands are queued
on one Metal queue; every command status and solve status is checked before
explicit result download. There is no CPU solve/product fallback. This remains
experimental test code, not a production backend wired into Cycles.

`cycles_diffraction_resident_matrix.mm` validates that chain on the same 29
captured propagation systems. Each case uploads A and B, executes a solve and
two residual/correction iterations, then computes A X on GPU. Only X and A X
are downloaded for final validation. It asserts that no payload download
occurs while building the operation chain. Status words are checked at final
synchronization and are not counted as matrix payloads.

The isolated outside-sandbox M5 run is terminal, exit 0. All 29 systems pass
unchanged solution, residual and product checks. There are 58 input uploads,
58 final validation downloads and zero intermediate payload downloads.
Maximum solution error versus the rounded CPU-double reference is
1.81899e-12; maximum relative residual is 8.38448e-8. Raw results and post-run
source/binary/fixture identity audit: `metal_resident_matrix_v1.json` and
`metal_resident_matrix_v1_audit.json` in `tests/output/diffraction`.

This removes intermediate host transfers for the tested solve/refinement/product
chain. The complete propagation algorithm still needs resident block assembly,
series accumulation and scattering-composition operations. No production cache
speedup, end-to-end resident propagation, or Blender integration is claimed yet.

## Complete resident layer-propagation loop

The resident matrix engine now supplies GPU identity, complex scaling,
horizontal join, slicing, block assembly and 1-norm operations. Earlier engine
sources were preserved in `metal_resident_matrix_v1_sources` before extending
them. `cycles_diffraction_resident_propagation.h` implements the full 40-term
small-step series, initial boundary solve, increment-preserving small-interval
updates, and repeated scattering combination with refined GPU solves. All
intermediate matrices stay resident. Only scalar norms needed for step scaling
and the increment/ordinary representation decision are read by the host.

The fixture generator uploads transformed P,Q matrices prepared on the host;
this test does not yet include resident material construction or exterior-port
matching. The reference is a double modal solve of those exact float P,Q
matrices and float thickness, then rounded to float for comparison. It is an
algebraic same-truncation test, not a converged physical cache validation.

`metal_resident_propagation_v1` completed outside the sandbox on M5, exit 0.
All 12 dielectric/deep-metal/covered-metal fixtures at N8,N16,N32,N64 pass the
unchanged 0.0003 component gate. Maximum complex error is 0.000104005. There
are 24 input uploads, 12 final operator downloads, zero intermediate matrix
payload downloads, and 36 scalar norm readbacks. Fixture source identities
verified; engine, shader, executable and result identities are recorded after
the run in `metal_resident_propagation_v1_audit.json`.

This removes matrix round trips from the complete layer-propagation loop.
It does not yet remove host material preparation, supply production cache port
operators, build adaptive cache cells, or connect the backend to Blender.
No speedup is inferred from the diagnostic run; timing needs a complete,
representative build workload and an explicit comparable CPU baseline.

## Standalone resident material-to-reference response

`cycles_diffraction_resident_response.h` now connects GPU material matrix
construction, refined constitutive solves, resident layer propagation, exterior
reference-port matching and retained-port extraction. The material shader
constructs Fourier coefficients, complex permittivity/profile matrices,
transverse momentum and exterior/internal admittances from a compact profile.
No host matrix preparation or intermediate matrix payload transfer is required
by this path. Scalar norms still synchronize host control. The prior resident
engine sources were preserved in `metal_resident_propagation_v1_sources`.

The companion Objective-C++ harness passes the production C++ cutoff fixtures
to this response builder. `metal_resident_response_v1` completed on M5 outside
the sandbox with exit 0. All 24 cases pass the unchanged 0.0003 component gate.
There are zero matrix input uploads, 24 final operator downloads, and no
intermediate matrix payload downloads. Uniform profile parameters are supplied
to the GPU and are not mislabeled as no data transfer. The report and post-run
source/binary/fixture identity audit are retained under `tests/output/diffraction`.

This is a complete standalone GPU response computation for the tested profile
family, not a production adaptive cache builder. Input validation, zero-depth
special handling, broader robustness/convergence, cancellation, resident memory
reuse, adaptive cache integration, and actual CPU/GPU build-time comparison
remain required. Existing renderer evaluator capacity and general coherent
transport limitations are unchanged. Do not claim full Cycles integration or
speedup from these numerical tests.

## Flat-interface path and input validation

The standalone resident response now handles exactly zero relief depth by
solving the exterior continuity equation directly. It skips constitutive
inversion, the internal admittance basis, series initialization and propagation.
This follows the physical fact that a zero-thickness layer has no response;
no epsilon threshold approximates a small nonzero depth as flat.

Profile validation now rejects nonfinite parameters, unsupported passive-index
representations, invalid dimensions/ranges, and retained windows that omit
propagating exterior orders. Fourteen invalid profiles exercise these checks.
The previous response header/harness are preserved in
`metal_resident_response_v1_sources`.

Production `diffraction_grating_solve_reference` generated 24 additional
zero-depth fixtures, including dielectric and metal substrates at and around
the same cutoffs. The outside-sandbox M5 run `metal_resident_flat_v1` passes
all 24: maximum component error 6.80030e-8, zero propagation doublings and zero
scalar norm readbacks. All 14 invalid profiles are rejected. The existing
24 finite-depth cases were then rerun as `metal_resident_response_v2`; all
still pass, maximum error unchanged at 1.67438e-5. Both jobs are terminal,
exit 0. Source/binary/reference/result identities are in
`metal_resident_flat_and_regression_v1_audit.json`, recorded after execution.

This is numerical validation of the new exact flat branch, not a measured
speed ratio. The standalone implementation still needs broader material and
cutoff robustness, cancellation/resource controls, cache-builder integration
and representative build-time benchmarks before production selection.

## First isolated CPU versus resident-Metal response benchmark

The standalone candidate is numerically valid on these fixtures but **not fast**.
`cycles_diffraction_response_benchmark.cpp` calls the production CPU reference
solver. `cycles_diffraction_resident_response_benchmark.mm` measures the resident
GPU response, including allocations, final readback and per-response autorelease
cleanup, with shader compilation excluded. Each of the same 24 zero-ky cutoff
fixtures has one warmup and three measured runs. CPU completes before GPU starts;
no task benchmark jobs overlap. GPU outputs are validated on every iteration,
outside the timed region. This is response preparation, not rendering, so adaptive
sampling is not involved; previous render benchmarks remain fixed-sample/adaptive-off.

Results in `tests/output/diffraction/resident_response_benchmark_v1`:

| Half-orders | Material | Median of case CPU medians (ms) | Median of case GPU medians (ms) | Median per-case GPU/CPU ratio |
| ---: | --- | ---: | ---: | ---: |
| 16 | dielectric | 1.773 | 39.615 | 22.65 |
| 16 | metal | 1.173 | 59.657 | 50.26 |
| 32 | dielectric | 6.115 | 290.479 | 47.10 |
| 32 | metal | 6.348 | 402.108 | 63.32 |

Both runs are terminal, exit 0; all numerical gates pass. Frozen source and
executable identities verified after both runs. `cpu.json` and `metal.json`
retain warmups and all measured times; `analysis.json` retains per-case values.
The CPU solver exploits its existing exact zero-ky polarization decoupling;
the GPU candidate currently does not. These results must not be generalized to
conical incidence or larger batches, but clearly reject promoting this candidate
as a speed improvement on the tested workload.

Resident storage alone is insufficient. The current kernels solve each matrix
with one threadgroup, issue many separate commands, allocate many temporaries,
and form full operators before retaining cache channels. Profiling must separate
those costs before selecting an optimization; they are observations of the
implementation, not measured attribution percentages. Batched/tiled linear
algebra, reduced RHS work and resource reuse are possible next approaches, not
claimed speedups. The selected Fast default avoids this builder entirely.

## Operation profile and rejected access-layout candidate

Optional instrumentation in the resident matrix engine labels each command
and accumulates its GPU start/end span, plus matrix allocation counts/bytes.
The original uninstrumented header is preserved in the first benchmark's
`source_snapshot`. `cycles_diffraction_resident_response_profile.mm` profiles
one representative cutoff fixture per N/material group, with one warmup and
three measured runs. Frozen sources/executable/fixtures are recorded in
`resident_response_profile_v1/provenance.json`.

The instrumented M5 run completes with numerical checks passing. Complex solves
account for 78.1%,80.7%,83.9%,86.4% of summed GPU command spans in the N16
lossless/metal and N32 lossless/metal cases respectively. These are command-span
shares, not hardware-counter shader utilization. There are 320–476 matrix
allocations and 27–138 MB of cumulative matrix allocation traffic per response;
these values are not peak resident memory. Products are the next largest
command category, about 9–10%. This points to the solver as the first optimization
target but does not measure separate causal costs of occupancy versus memory.

An isolated shader, `cycles_diffraction_matrix_rpc_coalesced.metal`, assigns
neighboring lanes to neighboring columns during elimination. Factors remain
unchanged until every lane finishes reading them, then the pivot column is
zeroed after a barrier. The candidate passes the 29 captured-system solver
checks and all 24 response benchmark cases, with unchanged numerical gates.
The benchmark uses the identical previously compiled host executable, fixtures,
one warmup and three measured runs, substituting only this shader. All recorded
source/executable identities verified. Output:
`resident_response_benchmark_coalesced_v1`.

It does not improve measured performance. Median per-case candidate/baseline
GPU ratios are 1.123,1.097,1.136,1.094 for the same four groups. Group median
case times are 44.506,65.943,328.544,439.577 ms. There is no speedup, and the
candidate is not promoted. The raw timings are retained; a small ordering/thermal
contribution is possible, but nothing here supports selecting this change.
A different solve algorithm or optimized GPU linear-algebra backend is needed
before claiming fast cache construction. Fast default rendering is unchanged.


## Apple MPS LU experiment (resident_response_mps_v1)

Inspected installed Apple SDK declarations for MPSMatrixDecompositionLU and
MPSMatrixSolveLU. Added a compile-time experimental DIFFRACTION_MPS_SOLVE
backend to the standalone resident matrix engine. It embeds complex systems
as real block matrices on GPU, performs partial-pivot LU and solve through MPS,
and converts the solution back on GPU. Existing compensated residual refinement
remains enabled. No intermediate matrix payload readback or CPU solve fallback.
Default experimental build and production renderer are unchanged.

Sequential external-sandbox M5 test: all 24 production reference cutoff cases
passed (maximum component error 1.67438321e-5, unchanged 3e-4 gate); 14 invalid
profiles rejected. All 29 captured propagation systems also passed original
solution/residual/product gates; maximum refined solution error 1.8189894e-12.
Source and executable identities were recorded before the response run and
verified afterward. Captured-system run is additional correctness evidence,
not a timed benchmark. Singular/nonfinite MPS behavior needs dedicated coverage.

One warmup and three measured complete response builds per case; allocations,
conversion, final readback and pool cleanup included, shader compilation excluded:

| Group | MPS median ms | MPS / prior GPU | MPS / CPU |
| --- | ---: | ---: | ---: |
| N16 dielectric | 39.295 | 0.993 | 22.78 |
| N16 metal | 58.166 | 0.973 | 49.28 |
| N32 dielectric | 126.272 | 0.435 | 20.68 |
| N32 metal | 161.524 | 0.402 | 25.49 |

Ratios are medians of paired case ratios. Prior CPU and GPU timing records are
from the preceding experiments, not interleaved contemporaneous runs. This is
substantial improvement for larger tested matrices, but still far slower than
CPU on these zero-ky fixtures (CPU exploits polarization decoupling). It is not
production-ready GPU cache acceleration. Do not promote on these results.
Next opportunities: reuse LU factors across residual corrections, exploit exact
polarization blocks, reduce command/allocation overhead and batch independent
responses. General conical/deep/high-order robustness remains unproven for MPS.
Fast remains the selected cache-free default. Full diffraction/coherence goal
remains active. Evidence: tests/output/diffraction/resident_response_mps_v1/.

## Reuse MPS factorization during refinement (resident_response_mps_reuse_v1)

Separated resident LU factorization from application to right-hand sides. A
refined solve now factors its unchanged matrix once and reuses that factor and
pivot vector for the original solve and both compensated residual corrections.
The previous implementation unnecessarily factored the matrix three times.
The factor's lifetime covers all submissions; command buffers retain GPU
resources. No CPU solve or intermediate matrix payload readback was introduced.

All 24 response fixtures pass the unchanged 3e-4 component gate, maximum error
1.67438321e-5. All 29 captured propagation systems pass original solution,
residual and product gates. Pre-run identities verified after both terminal runs.

| Group | Reused LU median ms | Ratio to previous MPS | Ratio to CPU |
| --- | ---: | ---: | ---: |
| N16 dielectric | 26.415 | 0.678 | 15.13 |
| N16 metal | 35.790 | 0.622 | 30.32 |
| N32 dielectric | 91.012 | 0.723 | 14.85 |
| N32 metal | 113.869 | 0.705 | 17.85 |

Same one-warmup/three-measurement harness and 24 zero-ky fixtures as the previous
experiment. Ratios are medians of paired case ratios; previous reference runs
were not interleaved. This removes redundant work and improves observed times
by 28–38%, but does not establish GPU advantage or production readiness. CPU
still exploits exact polarization decomposition absent in this GPU candidate.
No production renderer change or new render-time claim. Evidence and source
snapshot: tests/output/diffraction/resident_response_mps_reuse_v1/.

## MPS invalid-result publication guard (mps_failure_guard_v1)

Added a dedicated singular/nonfinite solve test with sentinel output memory.
The unguarded MPS backend rejected an exactly singular matrix, but accepted
NaN/Inf coefficients and NaN/Inf right-hand sides, publishing invalid output
(four failures preserved in mps_failure_before_v1.log). MPS decomposition
success alone is insufficient to validate these inputs.

Added GPU finite-value reductions on the coefficient matrix, each right-hand
side and each solved output. Atomic status flags are checked at synchronization
before download copies any payload. These checks do not read matrices back to
CPU. The guarded test rejects all five cases, leaves sentinel memory untouched
and records zero downloads. The engine remains fail-closed after an error;
recovery/reuse of a failed engine is not supported by this experiment.

All 24 production reference responses still pass, maximum error
1.67438321e-5. The unchanged one-warmup/three-measurement timing harness reports
26.310, 36.470, 93.158 and 112.312 ms group medians respectively for N16
dielectric/metal and N32 dielectric/metal. These are standalone build timings,
not rendering benchmarks. Source identities recorded before this regression
were verified afterward; archived source and results are in
 tests/output/diffraction/mps_failure_guard_v1/.

This closes a concrete robustness hole in the experimental backend; it does
not establish production integration or GPU speed advantage. Fast default and
production binaries remain unchanged. Full goal remains active.

## Conical and distinct-substrate response regression (resident_conical_response_v1)

Added a production-double reference fixture generator covering 24 cases at
N16/N32: dielectric ridge index 1.7 on substrate 1.3, metal 0.9+6i from air,
and the same metal from incident/groove index 1.5. Four deterministic profiles
per family vary wavelength 420–690 nm, pitch 900–1290 nm, depth 50–1100 nm,
duty 0.23–0.71, kx -0.31–0.26 and nonzero ky -0.45–0.36. Parameters covary;
this is a regression grid, not exhaustive/random coverage. Higher-index
incidence is an interface test, not propagation through a finite coating.
Retained order window is six, chosen to contain exterior propagating orders.

Extended standalone response reader to honor optional explicit groove and
substrate indices (legacy fixtures retain previous defaults). All 24 MPS
resident responses pass the unchanged 3e-4 maximum complex-component gate;
maximum error 5.40064364e-5. Production reference maximum boundary residual
3.21814045e-15. All source/executable/fixture hashes match pre-GPU-run records.
No intermediate matrix payload readback, no CPU solve fallback. The CPU fixture
generation completed before the isolated external-sandbox GPU run.

This compares two implementations at the same finite order truncation; it
is not evidence of order convergence or integrated rendered-image accuracy.
The 52-channel dielectric reference matrices also exceed the current production
cache evaluator capacity; the standalone regression does not resolve that
integration limitation. No production binary was changed. Full goal stays active.
Evidence and source snapshots: tests/output/diffraction/resident_conical_response_v1/.

## Higher-order conical regression and truncation changes

Extended the production reference generator with validated order-count arguments
and generated the same twelve conical profiles at N64 and N128. All 24 GPU
responses pass the unchanged 3e-4 complex-component gate. Added a reusable
reference-order analysis script that checks identical ports and profile identity
before comparing successive retained operators. Evidence is under
 tests/output/diffraction/resident_conical_orders_v1/; its identity audit was
captured after execution and is explicitly not prelaunch provenance.

Maximum operator changes across the twelve profiles are 0.0232715 for N16 to
N32, 0.00718635 for N32 to N64, and 0.00281531 for N64 to N128. Corresponding
minimum changes are 0.000539379, 0.000135323 and 0.0000383205. These include
reference-port evanescent components and cannot be interpreted directly as
rendered power errors. Decreasing changes support further investigation, but
neither CPU/GPU agreement nor a declining sequence certifies convergence.
Do not label all these responses physically converged. Further physical-power
and modal-convergence checks remain necessary. Production integration and full
original scope remain unfinished; Fast is still selected.

## Stable complex-root candidate (resident_stable_root_v1, not promoted)

Investigated cancellation in material_root's sqrt((radius +/- real)/2).
Created an isolated shader variant that computes the larger root component
first and derives the smaller from abs(imaginary)/(2*larger), preserving the
outgoing-root sign convention. Production and baseline experimental shader
remain unchanged. The worst earlier near-gate response was N64 bare metal,
750 nm depth, 600 nm wavelength (error 0.000296463).

First full N64/N128 candidate sequence terminated with the finite-value guard
("Metal solve failed") at case 13 after completing case 12. Its exit was 2,
and the partial stderr evidence is preserved; no valid complete report exists
for that run. The isolated case 13 then passed with error 3.10686271e-6.
Thus a deterministic profile-specific failure is not established. A repeat of
the exact full sequence was launched (session 78655); inspect its terminal
result before drawing conclusions. Do not promote this candidate while the
intermittent backend failure remains unexplained. Pre-run source identities
match the subsequent audit. This result exposes a robustness question rather
than proving an improvement from the root formula.

The repeat full sequence also terminated with exit 2 (finite-value guard).
Session 78655 is terminal; do not restart it. The isolated pass does not clear
these two sequence failures. Candidate remains unpromoted pending localization.

## Matrix-role diagnostic follow-up (resident_mps_diagnostic_v1)

Finite-value status buffers now record coefficient/right-hand-side/solution,
matrix dimensions and submission position. Failure messages expose this label,
allowing the first detected invalid matrix to be distinguished from a generic
solve failure. This adds diagnostics, not a numerical fix.

The instrumented full 24-case N64/N128 stable-root sequence completed, maximum
component error 0.000297054217, below but close to the unchanged 0.0003 gate.
A bounded repeat of the twelve N64 profiles three times (36 responses) also
completed with zero gate failures. Both GPU sessions (65744 and 13103) are
terminal. Post-run artifact hashes and header snapshot are recorded explicitly
as post-run evidence. No failed matrix was captured in these runs; the two
prior failures remain unexplained and must not be dismissed. The stable-root
candidate gives no useful improvement of the worst-case error and is not
promoted. Do not infer reliability from these successful repetitions alone.

## Exact zero-ky polarization split (resident_polarization_split_v1)

Added DIFFRACTION_POL_SPLIT as an experimental compile-time candidate. At
exactly ky=0 the transformed internal generators and boundary admittances are
polarization-block diagonal. Propagation and exterior matching now can run on
two half-size systems, then reconstruct the original side/polarization order.
Conical incidence retains the full coupled response. The initial constitutive
and internal-admittance solves are not split yet. The default compilation path
remains unsplit; no production binary or Fast shader changes.

All 24 cutoff cases pass the same gate, maximum error 1.67438321e-5. The
additional 24 conical cases also pass. One warmup and three measured complete
response builds per cutoff case, same baseline shader (not stable-root variant):

| Group | Split median ms | Ratio to guarded unsplit | Ratio to CPU |
| --- | ---: | ---: | ---: |
| N16 dielectric | 30.076 | 1.123 | 17.08 |
| N16 metal | 35.814 | 0.986 | 30.25 |
| N32 dielectric | 79.458 | 0.848 | 12.95 |
| N32 metal | 90.514 | 0.807 | 14.10 |

Ratios are medians of paired case ratios against prior, non-interleaved timing
records. This improves larger tested responses by 15–19%, while smaller
responses are dispatch/overhead-sensitive and N16 dielectric regresses.
Do not enable indiscriminately or claim GPU advantage. Pre-run benchmark
identities match post-run audit. Conical regression is a subsequent untimed
correctness run. Evidence: tests/output/diffraction/resident_polarization_split_v1/.

## Command batching candidate (resident_command_batch_v1, not promoted)

Added compile-time DIFFRACTION_COMMAND_BATCH to hold sequential compute/MPS
encoders in one command buffer until a required scalar synchronization or final
download. All solve statuses are still checked before payload publication.
Default is unchanged (individual submitted commands). Profiling in this mode
labels aggregate command spans matrix_batch, not individual operation costs.

The 24 cutoff benchmarks pass unchanged error gates (maximum 1.67438321e-5).
Contrary to the overhead hypothesis, measured batching is slower. Candidate
medians are 30.025/41.616/97.953/119.734 ms for N16 dielectric/metal and N32
dielectric/metal. A freshly compiled same-source unbatched control, run directly
afterward, also passes and measures 26.427/36.968/94.557/110.507 ms. Median paired
case candidate/control ratios are 1.127/1.127/1.047/1.080. These are sequential
whole-suite runs, not randomized interleaving; no speed benefit is supported.
Do not promote batching. Timing includes allocations/final readback, uses one
warmup and three measured builds, and excludes compilation. This is not a
render benchmark. Candidate prelaunch identities verified; control executable
identity recorded post-run. Full original goal remains active.

## Current MPS profile and inverse-refinement candidate

Reprofiled the split MPS builder (resident_mps_profile_v1). Across four fixture
groups, MPS LU application takes 46.8–52.6% of summed GPU command spans and
factorization 26.1–37.8%; products take 5.8–10.2%. These are instrumented command
spans, not isolated shader hardware counters. All profiled cases passed.

Added experimental DIFFRACTION_INVERSE_REFINEMENT: factor A once, solve against
identity to construct a float GPU inverse, then use compensated GPU products
for the initial solution and both residual corrections. No CPU solve fallback.
This can be less robust for ill-conditioned systems, so unchanged numerical
checks remain mandatory and the option is not production-selected.

All 24 cutoff benchmark cases pass (maximum error 1.67438321e-5), as do 24
conical/substrate regression cases (maximum 5.39722535e-5). With polarization
split also enabled, one-warmup/three-measurement response-build medians are
21.722/24.525/49.074/58.979 ms for N16 dielectric/metal and N32 dielectric/metal.
Paired-case ratios to earlier split timings are 0.712/0.684/0.620/0.644.
These non-interleaved runs support a promising 29–38% reduction, not a final
speed claim or superiority to CPU. Shader compilation excluded; allocations,
cleanup and final readback included. No rendering benchmark was performed.

High-order, singular/nonfinite, and captured ill-conditioned system checks
remain required for this new candidate. Existing MPS intermittent failures are
not resolved by this result. Evidence and explicitly post-run identities:
 tests/output/diffraction/resident_inverse_refinement_v1/.
Full goal remains active; Fast remains the selected production default.

## Inverse-refinement robustness follow-up (resident_inverse_validation_v1)

Added explicit GPU finite checks for the original right-hand side and final
refined solution in DIFFRACTION_INVERSE_REFINEMENT. The MPS application checks
otherwise only covered the identity right-hand side and inverse matrix, not
the new product-based correction path. All five singular/NaN/Inf tests reject
without writing sentinel output or recording a download.

All 29 captured propagation systems pass unchanged solution, residual and
product gates. Maximum solution error is 1.45519152e-11; maximum relative
residual 8.38448056e-8. All 24 conical N64/N128 response fixtures also pass the
unchanged 3e-4 complex-component gate. See analysis.json for maximum response
error. Higher-order run identities were captured before launch and verified
after terminal completion; source snapshots retained. Runs sequential on GPU
outside sandbox. No new speed measurement after adding the two finite guards,
so prior timing belongs to the prior executable and is not silently reassigned.
These results strengthen the candidate but do not clear earlier intermittent
MPS failures, establish modal convergence, or complete production integration.
Fast stays selected; full original goal remains active.

## Direct internal admittance inversion

Added experimental DIFFRACTION_ANALYTIC_ADMITTANCE to replace three dense
internal-basis solves with GPU products using an explicitly inverted homogeneous
admittance. The first candidate used the mathematical identity Y^-1=-Y/epsilon.
It failed two dielectric near-cutoff fixtures (maximum error 0.000383067,
unchanged gate 0.0003); preserve resident_analytic_admittance_v1 as a failure.

Revised kind-6 material kernel inverts the actually rounded float 2x2 blocks.
It computes det(Y)=-a*a-b*c using compensated FMA products, then -Y/det(Y).
This preserves the inverse of the basis used by the numerical calculation,
including near-cutoff roundoff. It uses no general LU solve for these blocks.
All 24 cutoff responses now pass (max 1.70518483e-5), as do 24 conical responses
(max 5.47855431e-5). Compile-time option remains off by default; production
renderer unchanged. Combined experimental shader for the failed ideal formula
is retained separately from the corrected variant.

One warmup/three measured cutoff builds give group medians 15.703/20.647/
39.139/47.890 ms (N16 dielectric/metal, N32 dielectric/metal), including
allocations, cleanup, final readback; compilation excluded. Previous inverse
candidate timings were 21.722/24.525/49.074/58.979 ms but lacked the subsequently
added refinement finite guards, so this is not a controlled single-change
speed comparison. Higher-order validation is still required. Post-run source
snapshots and identities: tests/output/diffraction/resident_block_inverse_v1/.
No modal-convergence, production-readiness, or full-goal completion claim.

## Direct-block high-order validation and harness cleanup

All 24 N64/N128 conical responses for direct rounded-block admittance inversion
pass the unchanged component gate. Maximum error is 0.000297743755, still close
to 0.0003, so this is not evidence of comfortable accuracy margin. Prelaunch
artifact identities verified in resident_block_inverse_high_v1.

The correctness harness previously kept its outer autorelease pool open for the
entire fixture suite. Added a per-response pool, drained only after download has
checked GPU completion. This bounds temporary-object lifetime between responses;
it does not prove a memory-related cause of earlier intermittent failures. The
benchmark harness already had per-iteration pools and is unaffected.

A subsequent full 24-case pooled run passed with identical per-case error and
step metrics. Peak process RSS from macOS getrusage was 54,018,048 bytes; this
is process RSS, not total GPU allocation or unified-memory residency. No earlier
RSS control was collected, so no quantified memory-reduction claim. Identity
audit is post-run; harness received whitespace/comment-only formatting after
compilation, explicitly noted in the audit. Evidence:
 tests/output/diffraction/resident_block_pooled_v1/.

No new renderer or visual-suite results. GPU backend integration, modal
convergence, intermittent-failure resolution and full original scope remain
unfinished. Fast remains selected; goal stays active.

## Fourier phase and input-quantization diagnostics

Isolated sinpi(order*duty) in place of sin(pi*order*duty) on the previously
worst N64 case. The candidate worsened error to 0.000305065423 and failed the
unchanged 0.0003 gate. It is rejected and retained in resident_sinpi_v1; baseline
shader remains unchanged. No broad success or speed claim.

Extended conical reference generator with --float-inputs to evaluate the double
solver at the exact float-representable input values received by Metal. This is
a diagnostic option, not a replacement for original reference acceptance. For
the worst case, rounding input parameters changes the double reference operator
by at most 6.31730663e-8. GPU error against that matched-input reference is still
0.000297681190, nearly unchanged from 0.000297743755 against original inputs.
Thus input quantization does not explain the near-gate discrepancy in this case;
the internal float computation remains the relevant target. Evidence under
resident_float_input_v1 includes all twelve N64 rounded-input references and
operator changes; only the worst case was dispatched to GPU in this diagnostic.
Original gates and original-reference results are retained. No production
selection or completion claim; full goal remains active.

## Compensated phase and direct material precision diagnostic

Tested a Fourier coefficient candidate retaining order*duty's multiplication
residual with FMA, evaluating sinpi of high/low parts, and correcting division
by split pi. On the exact-float-input worst N64 profile, response error worsens
to 0.000307615947 (fails unchanged 0.0003 gate). It remains isolated/rejected in
resident_compensated_phase_v1; no baseline shader changes.

Added a standalone material precision harness evaluating GPU permittivity and
inverse-profile Fourier matrices against independently computed double analytic
coefficients at identical float input values. Baseline relative Frobenius errors
are 2.97059986e-7 and 3.27343671e-7; candidate errors are 6.06156429e-8 and
1.23184845e-8. Maximum complex-entry errors decrease from 1.90661029e-6 to
1.45460447e-6 for permittivity and 5.16851275e-8 to 3.59716288e-9 for the
inverse profile. Thus coefficient accuracy improves while the complete response
worsens: coefficient rounding alone cannot explain or remedy the response error.
This also cautions against assuming the baseline's near-gate pass is a robust
accuracy margin. No gate relaxation or selective promotion. Both GPU diagnostics
ran sequentially outside sandbox; post-run identities and harness snapshot saved.
Further localization should examine constitutive inversion/propagation rather
than continuing blind Fourier-formula changes. Full goal remains active.

## Constitutive inverse localization (resident_constitutive_precision_v1)

Extended the direct material diagnostic to compute GPU refined inverses, then
measure A*X-I in independent CPU double arithmetic. Added an Eigen double LU
oracle on the *same downloaded float matrix* and its 1-norm condition number.
The CPU oracle is diagnostic only, never a builder fallback. Original response
reference and accuracy gates remain unchanged.

For the worst N64 profile, baseline permittivity/inverse-profile solve relative
forward errors are 2.40325e-8 / 2.59313e-8; corresponding 1-norm condition numbers
are 1125.623 / 915.837. Compensated-phase candidate values are 2.42940e-8 /
2.45604e-8 with conditions 1125.623 / 915.831. Relative Frobenius residuals
are about 4.6–4.9e-7. These observations do not establish exact physical input
accuracy: they validate inversion of the rounded GPU matrices, whose coefficient
errors may be amplified by conditioning. They argue against spending more solve
refinement steps as the primary remedy. Further localization must compare
constitutive effects and propagation, not infer full response accuracy from a
small inverse residual. Both pairs of GPU runs were sequential and terminal;
post-run hashes, source snapshot, raw metrics retained. Full goal stays active.

## Exact-material inverse diagnostic (resident_constitutive_exact_v1)

Extended the precision harness with independently assembled double analytic
material matrices at the exact float-valued profile inputs, and double LU
inverses of those matrices. This separates solve error on rounded GPU inputs
from coefficient-induced constitutive error. Diagnostic CPU reference only;
no change to the GPU builder or its acceptance gates.

Baseline relative inverse errors against exact material are 6.36240e-6
(permittivity) and 8.18988e-6 (inverse-profile). The rejected compensated-phase
candidate improves these to 1.70351e-6 and 5.86869e-7. Double inversion of the
rounded matrices gives nearly identical coefficient-induced effects, confirming
that GPU inversion itself contributes little to this measured difference.
Despite this improvement, the candidate's full response error previously
increased and failed the gate. Thus neither Fourier coefficients nor their
constitutive inversion alone explain the complete-response discrepancy;
propagation/boundary matching remain to localize. Do not infer error cancellation
or a precise cause without further stage comparisons. Source snapshot and
explicitly post-run identities saved; both GPU runs terminal, sequential.
No production selection or completion claim; full original goal stays active.

## Captured stage localization (resident_stage_capture_v1)

Added an optional diagnostic observer to the standalone response helper,
unused by default. A dedicated harness captures transformed P/Q, upper/lower
basis transforms, propagation, full exterior response and retained response.
It deliberately downloads/synchronizes stages, so it is not a resident-path
performance test and not a CPU fallback. The captured retained result reproduces
the earlier worst-case component error 0.000297743755.

An independent double modal propagation reference consumes the exact captured
float P/Q and host-float thickness. Double boundary matching consumes the same
captured transforms. Results for this one N64 metal profile:
- Internal propagation max component discrepancy: 0.000151252427.
- Full boundary arithmetic discrepancy at fixed GPU propagation: 3.12467e-7.
- Retained boundary arithmetic discrepancy: 1.04733e-7.
- Retained response effect of GPU vs double propagation: 0.000307604983.
- Double propagation and boundaries on captured inputs vs original production
  reference: 9.97763e-6.

This localizes the dominant discrepancy to propagation for this profile and
shows partial cancellation in the total GPU error. It does not prove the same
for every profile or establish modal convergence. The reference is an independent
modal solve at the same finite truncation. Raw binary captures, stage-analysis
script, source snapshots and post-run identities preserved. Next work should
address float propagation error, not boundary matching or extra constitutive
solve refinement. Production unaffected; full goal remains active.

## Propagation doubling localization and wider initial step

Added optional per-doubling observer (inactive by default). Captured the worst
N64 profile's initial layer and 13 doublings. Independent double modal references
use the exact captured transformed inputs and each intermediate thickness.
Maximum discrepancy starts at 6.73709e-8, reaches 1.29060e-5 at step 10,
0.000114665 at step 12 and 0.000151252 at step 13. The thin-layer initialization
is close; error grows during composition. Diagnostic readbacks intentionally
synchronize and are not production timing. Evidence: resident_step_capture_v1.

Added experimental compile-time initial-norm/series-length parameters, preserving
default norm 8 and 40 terms. Candidate norm 16 with 64 terms removes one doubling
for the worst case while extending the exponential series. Its complete-response
error decreases from 0.000297744 to 0.000211164. The subsequent 24-case N64/N128
sweep passes the unchanged 0.0003 gate; individual outcomes and improved-case
count are in resident_wider_step_v1/analysis.json. This is not a universal
accuracy proof or a performance result; low-order/cutoff coverage and timing
remain needed before selection. Captured source identities explicitly date
after dispatch and are verified after completion. Default and production
renderer unchanged. Full original goal remains active.

## Wider-step cutoff and timing regression (resident_wider_benchmark_v1)

Compared norm16/64 terms against norm8/40 terms, with the same current sources,
MPS inverse refinement, direct rounded admittance inverse and zero-ky
polarization split. Each cutoff profile uses one warmup and three measured
builds, fixed workload, compilation excluded and allocations/final readback
included. Candidate and control were run sequentially, not randomized.
All 24 cutoff cases pass in both variants; wide maximum error 1.97010673e-5.

| Group | Wide ms | Control ms | Paired median ratio |
| --- | ---: | ---: | ---: |
| N16 dielectric | 16.832 | 15.605 | 1.067 |
| N16 metal | 21.490 | 20.043 | 1.073 |
| N32 dielectric | 39.913 | 37.613 | 1.054 |
| N32 metal | 47.736 | 45.104 | 1.060 |

The wider step costs about 5–7% on these response-build fixtures while improving
21/24 high-order cases in the preceding sweep. A subsequent 24-case lower-order
conical regression passes as well. This supports retaining wider-step as an
accuracy candidate, not a speed winner. CPU remains faster on the cutoff
fixtures; no production GPU cache integration or render-time claim. Post-run
identities, source snapshot and raw records preserved. Production/Fast default
unchanged, full goal active.

## Physical reflected-power diagnostic

Added reference-to-physical admittance conversion for absorbing-substrate
reflection blocks, including evanescent reference channels during conversion
and flux normalization on propagating channels. Independent flat-interface
Fresnel checks at three incidence angles agree within 3.33e-16. Exact grazing
requires limiting treatment and is explicitly rejected by this diagnostic.
Lossless substrates/transmission are not covered yet.

For the captured worst N64 metal profile, propagating incident orders are
-2,-1,0,1. Baseline maximum per-order unpolarized reflected-power error is
4.69939e-5 of incident power; maximum total reflected-power error is 6.47268e-5.
Wider-step values are respectively 5.75984e-5 (worse per-order maximum) and
1.74454e-5 (better total reflection). Do not claim all physical metrics improve.
These correspond to at most 0.00647 and 0.00174 percentage points of incident
power for total reflection, on this one profile only.

Flux-normalized reflection-operator squared spectral norms are 0.968218767
(baseline), 0.968221442 (wide), and 0.968220255 (reference), consistent with
passivity here. This coherent input-port norm is a diagnostic mathematical
bound, NOT implementation of requested cross-object coherent rendering.
Original component gates remain unchanged. Same finite truncation comparisons
are not modal-convergence evidence. Raw captures/metrics/scripts/post-run
identities retained in resident_stage_capture_v1 and resident_wide_power_v1.
Full original goal remains active; no production selection changed.

## Physical power across modal truncations

Extended flux conversion to per-port lossless exterior indices, supporting
both reflection and transmission. Added independent zero-thickness dielectric
Fresnel and flux-unitarity checks at three incidence angles in addition to
absorbing Fresnel checks; maximum discrepancy 4.44089e-16. Corrected temporary
ratio-matrix allocation to always use complex dtype (a float test input had
issued a ComplexWarning; final run has no warning).

Analyzed the twelve conical production-double profiles at N16/N32/N64/N128
(48 operators). Lossless-case max flux-unitarity defect is 2.87748e-12. Maximum
squared spectral norm across all cases is 1.000000000009686, consistent with
passivity to numerical precision. These are reference checks, not GPU results.

| Truncation pair | Max per-order power change | Max R or T total change |
| --- | ---: | ---: |
| N16 to N32 | 0.00859454 | 0.00973552 |
| N32 to N64 | 0.00347734 | 0.00389388 |
| N64 to N128 | 0.000731924 | 0.00130115 |

All values are fractions of incident power. The remaining N64-to-N128 change
is up to 0.1301 percentage points in R/T totals, so finite-order accuracy cannot
be inferred from CPU/GPU operator agreement or energy conservation. No automatic
convergence threshold or altered acceptance gate introduced. This strengthens
the physical diagnostic pipeline while leaving required convergence/integration
work unfinished. Scripts, input hashes and post-run audit retained under
reference_power_convergence_v1. Full goal remains active.

## GPU physical-power suite (resident_power_suite_v1)

Extended the standalone correctness harness with an optional output directory
for final response matrices. Normal execution remains resident until final
readback; this option saves that already-downloaded result and does not perform
intermediate readbacks. Compiled the wider-step candidate with MPS inverse
refinement, exact-zero-ky split and direct rounded admittance inversion.

All 24 N16/N32 conical dielectric/bare-metal/higher-index-metal cases pass the
existing component gate. Physical-port analysis on final GPU operators vs
production-double references reports maximum per-order power error 1.05264e-5
and maximum R/T total error 4.69863e-6 of incident power (0.000470 percentage
points). Maximum lossless flux-unitarity defect is 1.81977e-6. Maximum squared
spectral norm is 1.0000061758: a small numerical passivity excess remains and
was not clamped or silently normalized away. This is measured error, not exact
passivity. Analytic Fresnel conversion checks remain within 4.44089e-16.

New script records every per-case physical metric and capture hashes, not just
aggregate maxima. Comparisons use identical finite truncation and do not prove
modal convergence, image quality or full backend integration. No new physical
acceptance threshold was invented to relabel these metrics as a broad pass.
Post-run identities and source snapshots preserved. Fast production selection
unchanged; full original goal active.

## High-order GPU physical-power suite (resident_power_high_v1)

Captured the wider-step candidate's final operators for all 24 N64/N128
conical profiles. All original matrix gates pass. Same-truncation physical
power comparisons give maximum per-order error 5.75984e-5 and maximum R/T
total error 2.80659e-5 of incident power (0.002807 percentage points).
Lossless max flux-unitarity defect is 1.83744e-5; max squared spectral norm is
1.0000656514. This numerical passivity excess is retained, not normalized away.
No new physical acceptance threshold was imposed, and these metrics are not a
proof of exact conservation or modal convergence.

Prelaunch executable/shader/fixture/analysis identities verified after terminal
GPU completion. Final matrix captures and per-case physical metrics preserved.
The completion audit now explicitly distinguishes implemented standalone GPU
response construction from still-missing Blender adaptive cache integration.
No production render or shader selection changed; full original goal active.

## Reference backend integration seam (reference_backend_integration_v1)

Added optional DiffractionReferenceSolver callbacks to cache and modal options.
Empty callback preserves the existing CPU default. Both fixed-resolution cache
queries and every modal-convergence iteration can delegate reference generation.
Fixed-resolution custom results undergo existing physical reference matching
before publication; failed/malformed results leave output empty. Modal queries
retain existing physical matching, adjacent/spanning convergence tests and
progress cancellation. Custom backends must handle concurrent validation calls
and fail explicitly rather than silently substitute a CPU solver.

DiffractionManager requires a nonempty stable implementation/version key for a
custom callback, reserves empty identity for CPU, and includes that key in cache
identity. This prevents cross-backend cache reuse. Caller-provided identity must
change when numerical backend behavior changes; std::function identity is not
used or inferred. No Metal backend is registered with Blender yet.

Standalone linked-source tests pass: CPU delegation equals original output;
modal queries invoke all three required resolutions; backend failures and
malformed success reject without result publication; cancellation stops before
backend invocation. Existing intensity and tensor cache modal tests pass (105
and 1707 reference solves respectively). Extended manager tests verify missing
identity rejection, distinct custom cache identity, and no publication on backend
failure. Both test executables terminate successfully. Initial test compile had
missing options arguments, corrected before execution; no renderer workaround.

These are real host integration changes but not completed GPU integration.
Full Blender rebuild and actual Metal adapter/cache validation remain outstanding.
Raw logs, source snapshots and post-run hashes preserved. Fast default unchanged;
full original goal remains active.

## First adaptive cache through Metal callback (metal_adaptive_cache_v1)

Added a standalone Objective-C++ adapter harness connecting resident Metal
reference construction to the actual production diffraction_grating_build_cache
callback. Calls serialize access to the single Metal engine (validation_workers=2
is supported by locking, not claimed GPU parallelism). Only completed finite
matrices are published; failures return explicitly with no CPU solve fallback.
Unknown reference-only diagnostic metadata uses NaN, not fabricated zero values;
production physical matching computes its own physical diagnostics. The harness
supports constant indices only and explicitly rejects tabulated profiles.

A nonzero-depth dielectric profile (pitch740nm/depth30nm/duty0.41/index1.5),
N4/retained2, domain kx/ky +/-1/1024 and wavelength600–601nm constructs one
accepted cell with 35 GPU reference calls. Packed coefficient max difference
from a separate CPU-built cache is 8.4293697e-8 (test gate3e-4). Cancellation
returns failure without publishing cells. Initial bounds +/-0.001 were rejected
by the existing float-representability requirement; original failure log kept,
fixture corrected to exact binary bounds and rerun successfully.

This proves an end-to-end host adaptive cache can use the Metal response
backend. It is a deliberately narrow fixed-N fixture, not evidence of full-domain
cache convergence, high-order adaptive success, GPU speed advantage, spectral
adapter support, tensor-backend integration or Blender UI/device integration.
No production Metal backend is registered yet. CPU cache is an independent test
control, not a fallback for the GPU run. Source snapshots/post-run identities
and logs preserved. Full original goal remains active.

## Spectral-table adaptive Metal adapter (metal_spectral_cache_v1)

Exposed diffraction_grating_sample_indices using the existing layer/substrate
validation and interpolation functions, and routed the CPU reference solver
through it. The standalone Metal adapter now resolves ridge, groove and absorbing
substrate tables through the same helper before uploading the compact float
profile. Scalar index interpolation is on CPU; material matrices and wave solves
remain GPU. No duplicate interpolation formula, extrapolation or CPU matrix-solve
fallback. All index outputs clear on sampling failure.

A dielectric spectral fixture with an interior wavelength knot at600.5nm,
domain600–601nm, builds two accepted adaptive cells from66 Metal reference
calls. Packed coefficient max difference from independently built CPU cache is
1.22878123e-7; cancellation still publishes no partial cells. This tests actual
spectral-knot splitting, not just single-wavelength solver agreement.

Dedicated shared-resolver tests pass interpolation of complex indices, exact
endpoints, out-of-range rejection, clearing outputs on failure, invalid absorbing
substrate topology, and duplicate-wavelength rejection. Reference backend
convergence/delegation/cancellation regression also passes after the CPU refactor.
Absorbing-table sampling is unit-tested but an absorbing spectral adaptive GPU
cache has not yet been exercised. Full Blender rebuild/integration remain open.
Source snapshots/post-run identities preserved; full goal active.

## Absorbing spectral adaptive cache and modal rejection

Added metal/metal-modal modes to the standalone adapter test. Ridge and absorbing
substrate both use a three-knot complex-index table with an interior600.5nm
knot; groove also varies spectrally. Fixed N4 construction accepts two cells
with66 GPU reference solves; packed coefficient max difference from CPU cache
is8.10710818e-7. Cancellation still rejects without publishing partial cells.

Enabling the existing0.001 modal power criterion with N4/8/16 correctly fails
at the first query with a Fourier-resolution-budget error. N8 adjacent power
difference is0.03393535; N16 adjacent is0.02585865 and spanning is0.04195865.
These are the production modal criterion's per-incident-column sums of absolute
order-power changes, not a single total-R discrepancy. No tolerance relaxation,
CPU fallback or forced success. This demonstrates actual GPU modal orchestration
and preserves the important distinction between fixed-order CPU/GPU agreement
and physical convergence. It is not a converged metal-cache success.

Both GPU sessions terminal; fixed-order exit0 and modal exit2 retained under
metal_absorbing_spectral_cache_v1 with raw logs, source snapshot and post-run
identities. Full Blender integration, broad convergence and original scope
remain unfinished; full goal active.

## Full Blender compile and Metal application smoke regression

Configured Release target rebuilt successfully (31 compile/link actions),
including changed host solver, cache manager, tensor builder and Cycles scene
consumers. New validation executable SHA256:
b12f579c8b46e61f65fe380606a8a582393e497843e5a4fac396fffd5d45cba4.
Direct launch of the bare build bundle failed at dyld because Resources/lib was
absent, not because of renderer execution. Preserved that log; made an isolated
validation app with the new executable and symlinks to existing installed
Resources/Info.plist. Did not install over the selected Fast binary.

The saved Fast PT CD/DVD scene rendered on Metal with1024 fixed samples,
adaptive OFF, denoise OFF. Initial preferences listed stale M2 and M5 entries;
a second isolated run explicitly enabled only Apple M5. That run's image relative
L1 difference from the selected-build EXR is2.48835884e-7, max absolute channel
difference2.83718109e-5. Actual PNG was inspected: diffraction fans and disc
geometry match, with visible sample grain. This is a smoke regression, not a
new performance benchmark, Realistic GPU-cache render, or clean presentation
acceptance. Installed binary remains selected SHA2562bb04b48...fbc9fe.

Evidence, logs, EXR/PNG and identity audit:
 tests/output/diffraction/reference_backend_blender_build_v1/.
Full Blender build verification is now complete for these host changes; actual
production Metal cache registration, broad convergence and original feature
scope remain unfinished. Full goal stays active.

## Rebuilt Blender BDPT and guided-PT application smoke checks

Ran the saved CD/DVD BDPT and guided-PT fixtures sequentially with validation
binary b12f579c..., only Apple M5 enabled,1024 fixed samples, adaptive OFF and
denoise OFF. Both application runs terminate normally. Comparisons are against
the same estimator's selected-build EXR, never normalized to PT brightness.
BDPT relative L1 difference3.98336e-11/max absolute9.53674e-7; guided PT
relative L1 difference1.73275e-6/max absolute7.78590e-5. Actual PNGs inspected:
expected disc geometry and diffraction fans, visible sampling grain, no obvious
new visual artifact. Guiding is not assumed bitwise deterministic.

Evidence: reference_backend_blender_build_v1/m5_bdpt_smoke and
m5_guided_smoke, raw logs/scripts and transport_smoke_audit.json. These are
host-change smoke regressions of the existing Fast path, not benchmarks,
full-pipeline certification, Realistic GPU renders or new physical ground truth.
Installed selected executable remains unchanged. Full goal stays active.

## Modal resolution target for spectral metal query

Probed the actual rejected cache query through N4/8/16/32/64/128 with both
Metal and CPU, retaining observations rather than constructing a full cache.
Both reject N128: adjacent power differences0.000481495 (GPU) and0.000481926
(CPU) meet0.001, but spanning differences0.001316269/0.001315872 do not.
The diagnostic process exits0 to indicate complete collection; report.accepted
is false for both and no converged result is published. Do not call this a pass.

An additional CPU-only N256 probe accepts the unchanged production criterion:
adjacent0.000221855432, spanning0.000703781247, with prior adjacent comparison
also passing. This is empirical convergence of this one query, not a uniform
material/domain guarantee. It establishes a concrete gap: current Metal adapter
supports at most N128, below the required resolution here. Extending GPU size
support needs memory/resource and numerical validation, not merely raising a
constant. No GPU fallback or relaxed convergence tolerance used. Source/log/hash
snapshots in metal_modal_probe128_v1. Full original goal remains active.

## Experimental N256 Metal modal query (metal_modal_probe256_v1)

Added opt-in DIFFRACTION_LARGE_ORDERS raising the resident/adapter limit to256;
default remains128. Matrix allocation checks Metal maxBufferLength. Large mode
also limits sampled allocated device memory to min(4GiB,75% of recommended
working set) and drains/validates pending command buffers every16 submissions.
This bounds retained in-flight resources during the exponential series; it does
not claim zero synchronization overhead. Expanded matrix row cap to8192 for the
real embedding of larger complex systems. No production backend registration.

The actual spectral-metal cache query now passes the unchanged0.001 empirical
modal criterion on GPU atN256: adjacent power difference0.000221181010,
spanning0.000701727243. CPU accepts atN256 with0.000221855432/0.000703781247.
Both retain the prior adjacent-pass requirement. Sampled Metal allocated-memory
peak is902,053,888 bytes. This counter samples device.currentAllocatedSize at
our allocations, not an exact hardware peak; its repeated value in the CPU row
is the same engine counter, not CPU memory usage. No speed benchmark performed.

The diagnostic process and both modal solves complete successfully. Source and
executable identities captured while the run was active match after completion.
This is one converged query, not an accepted whole spectral material cache or
proof of N256 robustness across profiles. Larger-order numerical/operator and
resource-bound tests remain necessary. Full original goal stays active.

## Accepted N256 GPU/CPU operator comparison

Evidence: `metal_modal256_comparison_v1`. Repeated the same modal query and
compared the accepted operators directly, rather than inferring agreement from
separate convergence passes. Both accept N256. Maximum complex reference
component difference is 6.10814548e-6, passing the unchanged 3e-4 gate.
Production reference matching and power conversion give maximum individual
outgoing-order power error 1.50823487e-6 (0.000150823 percentage points of
incident power); maximum column sum of absolute power differences is
2.13611633e-6. These are measured differences, not new acceptance tolerances.

Sampled allocated GPU memory remains 902,053,888 bytes. The diagnostic now
reports this counter only for the GPU row. The process exited successfully;
both explicit acceptance flags and the component gate were inspected. Source,
object and executable hashes were captured after completion, not before launch.
This is still one spectral-metal query, not broad N256 validation, a complete
material cache, a speed benchmark, or production Metal backend integration.

## Explicit GPU selection prerequisite

The resident engine now accepts an explicit MTLDevice. A nil selection fails
without silently choosing another GPU, and command-queue allocation is checked.
The convenience constructor remains for standalone experiments. Integration
must use Blender's selected Metal device, not the convenience constructor.

The spectral-metal adaptive-cache adapter exercised the explicit constructor
outside the sandbox on Apple M5 (registry ID 4294968422). Missing-device
rejection passed; two cache cells required 66 reference calls, with packed
coefficient maximum difference 8.10710818e-7 from the CPU control. Cancellation
before construction still rejected without publishing cells. Evidence and
post-run identities: `metal_selected_device_v1`. This is not a multi-GPU test.

Lifecycle inspection also identified remaining cancellation latency: cache
progress is checked per adaptive node, while its modal-reference progress
callback currently only counts solves. Thus the existing cancellation test
does not prove prompt cancellation during a large modal solve. Device selection
is now expressible, but production registration and responsive cancellation
remain incomplete. The full goal stays active.

## Cancellation between modal reference solves

Added a separate thread-safe cancellation query to cache options and checked it
before reference generation and between modal resolutions. This avoids calling
the adaptive-node progress reporter concurrently from validation workers.
Shader preparation connects it to a locked Progress query that does not invoke
the application's cancellation callback on worker threads. The manager rejects
cancelled requests before lookup and after construction, before registration.
Stored material identities now discard solver and cancellation callbacks as
well as the progress callback, so they do not retain per-build captured state.

The reference-backend regression cancels after the first successful solve and
verifies exactly one solve, an explicit cancellation error, and empty output.
It also checks fixed-resolution pre-cancellation. The compiled regression
passes; evidence: `reference_cancellation_v1/test.log`. Full Blender compilation
completed successfully (68 build actions); build log and post-build source and
executable identities are saved alongside the regression. The installed selected
renderer was not replaced, and application rendering has not yet been repeated
for these cancellation changes.
An already executing CPU/GPU solve is not preempted by this change. GPU command
batch cancellation and full production registration remain outstanding.

## Metal cancellation and callback lifetime regressions

The explicit-device Metal adapter now tests cancellation after one completed
reference solve through the production cache-reference entry point. On Apple
M5 outside the sandbox, the test stopped before a second modal solve, reported
cancellation, and published an empty operator. A subsequent fixed-order query
on the same engine succeeded. The adaptive spectral-metal comparison also
remained at two cells, 66 reference calls, and maximum packed coefficient
difference 8.10710818e-7. These checks do not preempt an in-flight GPU command.

Extended the manager regression with weak-pointer lifetime checks: successful
registration releases captured solver/cancellation state once the caller drops
its options. Cancellation rejects an existing cached request, preserves cache
count, and removing cancellation allows reuse of the same handle. Both intensity
and tensor modal caches pass (105 and 1707 reference solves respectively).
All processes exited zero; logs, test-source snapshots, and post-run hashes are
in `metal_cancellation_v1`. Production registration and broader feature scope
remain incomplete; no speed or rendering-quality conclusion follows from these
lifecycle tests.

## Cancellation within the experimental Metal solve

The resident engine now accepts a cancellation query at command creation and
synchronization boundaries. On cancellation it discards any uncommitted batch,
waits for already committed commands, clears pending status resources, and
raises an explicit cancellation error. A Metal command failure during draining
is reported as a failure rather than hidden as successful cancellation. This
does not interrupt an individual executing Metal command or MPS factorization.

An Apple M5 run outside the sandbox injected cancellation at checkpoint 10
inside reference construction. The adapter returned an empty operator without
payload download; a subsequent query using the same engine succeeded. Existing
adaptive-cache comparison and between-resolution cancellation checks also pass.
Evidence: `metal_command_cancellation_v1`, with raw log, source snapshots and
post-run hashes. Only one internal injection point was tested; no cancellation
latency bound or production UI integration is claimed. The full goal stays active.

## Shared-ownership Metal reference backend extraction

Moved the resident matrix, propagation and response implementation into
`intern/cycles/device/metal/diffraction/`; former test headers now forward to
those files, avoiding divergent copies. Added a reusable serialized reference
backend and a callback factory with shared ownership. Device and shader source
remain explicit constructor inputs. Matrix output must be square and agree
with the retained physical port layout. The production Metal device does not
yet instantiate this backend; CMake lists the headers but does not register it.

The existing adaptive spectral-metal, command cancellation and recovery tests
now exercise the extracted implementation. An added lifetime test releases the
creator's shared pointer, performs a successful query through the retained
callback, then verifies destruction after releasing that callback. All checks
pass on Apple M5 outside the sandbox; adaptive coefficient difference remains
8.10710818e-7. Logs, source snapshots and post-run hashes are preserved in
`metal_owned_backend_v1`. Production shader-source packaging, device dispatch,
settings and performance selection remain outstanding.

## CMake-packaged Metal shader source

Added canonical matrix-operation, material and MPS-conversion shader sources to
the Metal backend directory. CMake concatenates them into a generated C++ raw
string header; source changes trigger reconfiguration. This removes runtime
dependence on test-output shader files for the packaged adapter. The initial
sources match the previously validated shader combination. Historical test
shader files remain available for prior experiments.

Blender configuration and build succeeded. The standalone adapter compiled with
the actual CMake-generated header and ran with no shader-path arguments on Apple
M5 outside the sandbox. Spectral-metal cache comparison, both cancellation
checks, recovery and shared-owner lifetime tests all pass, retaining the
8.10710818e-7 maximum packed coefficient difference. Evidence and post-run hashes:
`metal_packaged_source_v1`. Blender itself still does not instantiate the backend;
the successful application build alone does not establish runtime integration.

## Compiled Metal device factory

Added a virtual device configuration entry point with explicit unsupported-device
failure, and a Metal implementation using the existing selected `mtlDevice`.
The implementation constructs the shared backend from packaged shader source,
passes the per-build cancellation query, and installs the reference callback
with a stable implementation identity. It uses the validated MPS, polarization
split, inverse-refinement, analytic-admittance and 16/64 propagation settings,
plus bounded N256 support. No silent CPU fallback is introduced.

The Objective-C++ translation unit uses ARC and links MetalPerformanceShaders.
Full Blender compilation and linking succeeded. A macOS 15 availability guard
now protects the safe-math property; older systems receive an explicit error.
The final build has no warnings originating in the diffraction source, though
SDK header warnings remain in the log. Evidence and post-build executable/source
identities: `metal_device_factory_v1`. Shader preparation does not call this
factory yet, so application runtime dispatch and rendering remain unverified.

## First application render using the Metal cache backend

Shader preparation now selects the first active Metal rendering subdevice for
Realistic cache construction, preserving an explicitly supplied solver. Fast
returns before this path. Metal initialization failure reports an error without
CPU fallback; non-Metal rendering retains the existing CPU reference builder.

The rebuilt validation application rendered the saved Realistic flat-metal
fixture on Apple M5 outside the sandbox, 64 fixed samples, adaptive sampling
and denoising disabled. The log explicitly names `metal-mps-wide64-inverse-v1`
and Apple M5, then reports one built cache cell, 6400 matrix bytes and sampled
power error 0.000205431. Preparation took 0.785813 seconds in this cold smoke
run; this includes initialization and is not a comparative benchmark.

The saved EXR passed the existing analytic flat-interface check: largest mean
RGB error 0.0003678623 against neutral Fresnel reflectance 0.9091138623
(unchanged smoke tolerance 0.005). Evidence, saved blend/EXR, report, script,
build log and prelaunch binary/script identities: `metal_application_dispatch_v1`.
This validates application dispatch for a zero-depth interface, not finite-depth
propagation, modal convergence, all transports, performance advantage or the full
requested feature set. The installed selected Fast application is unchanged.

## Finite-depth application validation in progress

Launched the same validation binary with the saved Realistic relief-metal
fixture (150 nm depth, 740 nm pitch), Apple M5 only, 64 fixed samples and
adaptive sampling off. The log confirms the Metal cache backend. At the first
process inspection it was still active after 67 seconds (34.59 CPU seconds,
approximately 255 MiB RSS); no render had been produced. This is not evidence
of a hang, success, or speed advantage. Keep monitoring the existing exec
session 57621 rather than restarting from a missing image. Logs and during-run
identities are in `metal_application_dispatch_v1/relief_run.log` and
`relief_active_audit.json`. The run must be resolved before another GPU job.

A one-second process stack sample subsequently confirmed recursive adaptive
cell construction, actual resident Metal propagation and MPS work, plus waiting
validation workers serialized by the backend mutex. This is qualitative evidence
of active work, not an unbiased timing breakdown. The sample is preserved as
`relief_stack_sample.txt`; the sampling process terminated normally. The render
session remains live. Normal logs lacked periodic progress because those counts
used LOG_DEBUG. The source now emits the existing five-second periodic counts
at INFO level and updates render substatus with accepted cells, solve count and
elapsed seconds. This logging edit has not yet been rebuilt and does not affect
the live validation binary. It does not change solver settings or tolerances.

## Deferred backend initialization on cache reuse

Device configuration previously compiled the Metal library before the manager
could discover an existing material cache. Initialization is now deferred until
the first reference query, guarded by a shared state mutex. The state owns the
explicit selected device and cancellation query; cache hits invoke no solver
and therefore allocate no backend. Initialization and solve failures still
return an explicit error with empty output. Numerical implementation identity
is unchanged because the solver and tolerances are unchanged.

The complete Blender build passes, including the INFO/substatus progress edit.
Evidence and post-build identities: `metal_lazy_initialization_v1`. Runtime
validation of lazy initialization remains pending behind the live relief job
(session 57621), which continues using its frozen earlier binary. The CPU build
overlapped that diagnostic run; do not report its elapsed time as an isolated
performance benchmark. No additional GPU job was launched.

Prepared `tests/python/cycles_diffraction_cache_reuse.py` for the next runtime
check. It uses one explicitly selected Metal device, fixed sampling, adaptive
and denoising off, and persistent scene data. It renders original/edited/restored
color states of one Realistic diffraction material, checks finite pixels and
image restoration, and records EXRs plus numerical results. Its report explicitly
requires separate log verification of cache reuse; image agreement is not proof
of reuse or deferred initialization. Syntax compilation passes, but it has not
run. The relief session 57621 was polled again and remains active; no competing
GPU render was started.

The lazy factory now records initialization failures for the current build
request, so waiting validation workers return the same error without repeating
shader compilation. A fresh request gets fresh state and may retry. This only
memoizes initialization failure, not a numerical solve failure. The full build
passes; failure injection remains untested at runtime. Build log and post-build
identities are in `metal_lazy_initialization_v1/failure_build*`. The long-running
relief process still uses its earlier frozen binary and remains active.

## Relief application diagnostic intentionally cancelled

The original relief application job was gracefully cancelled with SIGINT after
more than 26 minutes of cache construction. The last process sample recorded
1552 seconds elapsed, 789.30 CPU seconds, approximately 336 MiB RSS and ongoing
activity. Session 57621 then returned terminal exit code 1. No image was produced.
This was an operational decision to redirect effort toward reducing construction
cost, not a predeclared timeout gate, numerical rejection, hang diagnosis or
isolated performance benchmark (CPU builds and stack samples overlapped).

Preserved log, saved blend, reference selection and `relief_termination.json`
under `metal_application_dispatch_v1` document the incomplete run. Full-domain
finite-depth application correctness remains unproven. This job is now terminal;
the next GPU test can run sequentially without overlap.

## Application cache reuse and queue-limit timing

The lazy-initialization application regression passes on Apple M5: one actual
initialization, one cache build, and two logged cache hits across original,
half-color, and restored-color renders. Relative L1 edited change is
0.5000000035; restored error is 7.58403e-8. All three use 64 fixed samples,
adaptive/denoising off. Evidence: `metal_cache_reuse_v1/verified_report.json`,
raw log, EXRs and prelaunch identities. These are regression runs, not timings
to advertise as a speedup.

Measured the production-style LARGE_ORDERS configuration against the same-source
small-order control sequentially on 24 warmed N16/N32 fixtures (one warmup,
three measured runs each). Median paired bounded/control ratio is 1.34052.
Six-case group medians in milliseconds: bounded 23.028/30.768/47.477/57.683;
control 15.716/19.658/36.783/44.843. Both pass all reference component gates,
maximum error 1.97010673e-5. The compared macro also enables working-set checks;
this experiment does not independently separate those checks from periodic
queue draining. Compilation is excluded. This is not a full-cache benchmark or
proof that the queue limit explains the long relief build. Evidence and post-run
identities: `metal_queue_limit_benchmark_v1`. The invalid-order test now uses
the configured maximum plus one, so N256-enabled builds test the correct limit.

## Queue limit isolated comparison

Added an overridable positive pending-command limit, retaining the production
default of16. Compared a256-command candidate with a fresh16-command control;
both use LARGE_ORDERS and identical memory guards and solver settings. GPU runs
were sequential outside the sandbox, candidate first, one warmup and three
measured samples for each of24 N16/N32 reference fixtures. Median paired
candidate/control ratio is0.742779 (about25.7% lower response time).
Six-case group medians: candidate15.789/20.036/36.582/45.253ms;
control22.784/30.769/47.353/58.644ms. Both pass all accuracy gates, maximum
component error1.97010673e-5. Shader compilation is excluded.

Evidence, raw outputs and post-run identities: `metal_queue_256_v1`. This
isolates queue draining more directly than the prior LARGE_ORDERS comparison.
The candidate is not promoted for larger matrices; high-order memory/resource
validation remains necessary. No full-cache speedup or final rendering-method
selection is claimed. Production default remains16.

## N256 validation of the256-command candidate

The spectral-metal modal query accepts N256 with the256-command limit. Both
GPU and CPU accept the unchanged convergence criteria; maximum complex operator
error remains6.10814548e-6, maximum individual-order power error1.50823487e-6,
and maximum column power L1 error2.13611633e-6. The component gate passes.
Sampled Metal allocation peak rises to2,442,215,424 bytes, versus902,053,888
for the16-command version. This is sampled allocated memory, not exact hardware
peak, and this run is not a speed benchmark.

Evidence: `metal_queue256_high_v1`; process exit0, explicit acceptance flags
inspected, during-run identities match after completion. This result supports
testing a size-dependent queue policy rather than globally replacing16 with256.
Production settings remain unchanged; broader high-order and application
validation remain incomplete.

## Size-dependent queue selected for the integrated backend

The new policy uses256 pending commands for half-order counts at most32 and16
above32. Memory guards, solver math and reference tolerances remain unchanged.
The N4-through-N256 modal sequence passes with maximum complex error6.10814548e-6
and sampled peak allocation902,053,888 bytes, restoring the tighter queue's
resource use. The24 small-response accuracy fixtures pass; six-case timing
medians15.605/19.661/36.853/44.740ms retain the faster queue's measured range.
This last run was not paired with another fresh control and does not add a new
full-cache speedup claim.

Adaptive spectral-metal cache comparison, cancellation before and within a
solve, same-engine recovery and shared-owner lifetime tests also pass. All GPU
runs were sequential outside the sandbox. The integrated Metal factory now
enables this policy and the full Blender build succeeds. Evidence and post-run
identities: `metal_adaptive_queue_v1`. Application runtime regression and another
full-domain finite-depth test are still needed; the earlier26-minute incomplete
run remains preserved. This selection concerns queue scheduling only, not final
acceptance of the full rendering feature set.

## Adaptive-queue application regression and bounded relief diagnostic

The new application build passes persistent-data color update/restoration:
one initialization, one build, two verified cache hits; restored relative L1
7.73317836e-8 and edited change0.5000000031. Apple M5 only,64 fixed samples,
adaptive/denoising off. Evidence: `metal_adaptive_application_v1/reuse_verified.json`.

A separate relief run had a60-second diagnostic budget declared before launch.
The last periodic report at55.4056 seconds shows46 visited nodes,19 accepted
cells and2123 reference solves. The process was gracefully cancelled at its
budget and returned1; wrapper elapsed61.372 seconds. No image or correctness
acceptance follows. Saved command/budget, scene, log and terminal status are in
the same evidence directory (`relief_plan.json`, `relief.log`, `relief_status.json`).
This resolves the open-ended observation problem and provides a concrete
construction-rate baseline. Bounded concurrency of independent small reference
queries remains a candidate; current validation workers serialize on one engine.

## Two-engine small-response throughput experiment

Added a standalone benchmark comparing one and two independent resident engines
on the same selected Apple M5. Each configuration receives one whole-suite
warmup, followed by three alternating measured suites of24 N16/N32 fixtures.
Each output is checked against its CPU reference; all gates pass with maximum
component error1.97010673e-5. Median whole-suite time falls from705.462ms to
532.969ms (ratio0.75549, about24.5% lower). Timing includes thread startup,
allocation, readback and checking, and excludes engine/library initialization.

This is controlled concurrency inside one isolated GPU benchmark job, not
overlapping independent render jobs. Evidence and post-run identities:
`metal_concurrent_benchmark_v1`. No production pool is enabled yet. The result
does not prove a full-cache speedup, large-order resource safety, or arbitrary
concurrent-material correctness. A bounded pool and large-solve exclusion policy
remain necessary before application integration.

## Bounded two-slot reference pool prototype

Implemented `reference_pool.h`: two independently owned lazy engines for small
queries, exclusive access for half-order counts above32, and priority for waiting
large queries. Waiting callers poll cancellation every25ms. Slot release is
scoped across normal and exceptional exits; each slot remembers initialization
failure. The implementation is not enabled by the production factory yet.

The small-response benchmark submits the same24 fixtures through this actual
pool with one versus six caller threads. One warmup per configuration precedes
three alternating measured suites. Every output passes the3e-4 component gate;
maximum error1.97010673e-5. Median suite times706.823ms and516.537ms give ratio
0.730787 (about26.9% lower). This is not a full-cache benchmark. Evidence and
post-run hashes: `metal_pool_v1`. Mixed large/small exclusion, cancellation while
waiting, failure recovery and application integration remain to be validated.

## Mixed-size pool and rejection recovery check

Six callers submitted eight queries spanning N4/N16/N32/N64/N128 through the
actual two-slot pool on Apple M5. All completed operators agree with separately
computed CPU references within2.14524491536e-6 maximum complex-component error
(unchanged3e-4 gate). A subsequent invalid N257 exclusive query is rejected
with empty output. Pre-cancellation also returns an explicit cancellation error
and empty output; clearing cancellation permits a successful query on the same
pool. Evidence and post-run identities: `metal_pool_mixed_v1`.

These checks exercise mixed-size scheduling and release after rejected work,
but do not directly establish cancellation of an already-waiting request,
starvation bounds, or complete concurrent memory coverage. No pool production
integration or full-cache speedup is claimed yet.

## Queued cancellation and pool factory connection

Added a locked scheduling snapshot and a targeted regression that observes an
exclusive N256 request active with one small request waiting, then requests
cancellation. Both return cancellation errors and empty output; active/waiting
counts and exclusivity clear, and a subsequent query succeeds. The earlier
mixed-size reference and rejection checks also pass. This proves cancellation
after observed queueing, not a bound on interrupting an individual GPU command.

The Metal factory now creates this lazy two-slot pool; each actual engine
initialization emits a distinct pool-slot log entry. Cache hits still create no
engines. The full Blender build succeeds, but pooled application rendering and
cache-reuse logs have not yet been tested. Evidence: `metal_pool_wait_cancellation_v1`.
Post-build identities are explicitly captured after adding the initialization
notification to the tested header. No full-cache speedup is inferred yet.

## Pooled application regression and relief progress

The integrated pool passes Blender's persistent-data regression: two slot
initializations, one built cache, and two cache hits with no later slot
initialization. Restored relative L1 is7.71463548e-8; half-color change is
0.5000000034. Apple M5 only,64 fixed samples, adaptive/denoising off. Evidence:
`metal_pool_application_v1/reuse_verified.json`, EXRs and log.

The predeclared60-second relief diagnostic also cancels cleanly. At its last
report (57.0287s), it reached62 visited nodes,26 accepted cells and2905 solves.
The previous one-engine diagnostic reported46 nodes,19 cells and2123 solves
at55.4056s. Logs support greater throughput but report at different points;
neither completed its full cache or produced an accepted image, and these are
not isolated repeated full-render benchmarks. Raw observations are preserved in
`metal_pool_application_v1/progress_comparison.json`. Wrapper terminal status:
exit1 after the declared budget,60.7973s elapsed. No GPU job remains active.

## Full pooled relief attempt and failure cleanup review

Started a full-domain pooled relief render with1024 fixed samples, adaptive off,
and a predeclared1800-second limit. Frozen application binary/script identities,
command and PID are in `metal_pool_relief_full_v1`. Session72174 remains active;
at73.4256 seconds the log reports79 nodes,36 accepted cells and3732 solves.
No image or correctness conclusion exists yet. Poll that handle before starting
another GPU job. The command wrapper waits in at-most30-second intervals and
will record terminal status after completion or graceful budget cancellation.

Review found that non-cancellation exceptions could return an engine slot while
retaining submitted commands. Added common discard/drain cleanup to the engine,
used both for cancellation and reference-backend exceptions. It waits for
submitted commands before slot release, clears discarded status resources, and
preserves command failures in the error. Full Blender compilation passes;
an injected-exception recovery regression is prepared in the adapter but has
not run while the full relief job occupies the GPU. Build evidence:
`metal_failure_drain_v1`. The live relief binary is unchanged. The overlapping
CPU build means this run is validation, not an isolated performance benchmark.

## Full pooled attempt terminal result and numerical recovery

Session 72174 is terminal. The predeclared 1800-second budget was reached;
child exit code 1, elapsed 1800.378 seconds. Last progress: 2011 nodes,
1002 accepted cells, 92413 solves at 1794.21 seconds. No EXR was produced.
This is an incomplete attempt, not a numerical failure or isolated benchmark.
Evidence: `metal_pool_relief_full_v1/status.json` and `run.log`.

After that job terminated, `metal_failure_recovery_v2` ran on Apple M5 outside
the sandbox and exited 0. The two-cell spectral-metal fixture differs from
CPU packed coefficients by at most 8.10710818e-7. Both checkpoint cancellation
and injected-exception recovery match the pre-failure response with zero
measured component difference (gate 3e-4). Callback ownership also passes.
The strengthened test supersedes the compiled-only v1 test; this is targeted
injection coverage, not proof of recovery from every driver/allocation fault.
No GPU job remains active after these terminal results.

## Cache work diagnostic

Added progress counters for reference hits, peak stored reference-matrix bytes,
accepted domain fraction and depth; full Blender compilation passed. A fresh
60-second Metal application diagnostic terminated at its declared budget
(exit 1, 60.3203 seconds). At 57.1762 seconds: 2905 solves, 5114 hits,
4,648,000 stored matrix bytes, 26 accepted cells and accepted domain fraction
0.0163574. The reference matrix cache is below its 64 MiB limit, so early
eviction cannot explain this preparation cost. The hit fraction is about
63.8%; refinement remains expensive despite reuse. Domain fraction measures
parameter-space volume, not elapsed-work percentage or a completion forecast.
Evidence: `metal_cache_diagnostics_v1` with prelaunch hashes, build log, raw
progress, terminal status and parsed observations. No GPU job remains active.

## Staged quadratic validation candidate

Quadratic cache validation now evaluates its quarter-point lattice first. If
it already exceeds either enabled tolerance, refinement is required and the
second Gauss lattice is skipped. All 27 first-lattice samples remain available
for curvature; every accepted quadratic cell still passes all 54 samples.
No tolerance, domain or accepted-cell interpolation order was changed.

Four bounded conductor/dielectric fixtures, each with power-only or additional
complex validation, produce byte-identical packed caches against the pre-edit
source compiled with matched settings. Solve counts change 595→465, 783→601,
79→79 and 383→305. Full Blender compilation passes. These fixed-N4 cases
validate the scheduling change, not physical modal convergence or broad GPU
performance. GPU application timing remains pending.

Evidence: `staged_validation_v1`. The initial comparison used a previously
compiled host object and failed byte equality with matrix differences at most
6.47e-15; topology was identical. That failure is retained in `comparison.json`.
Recompiling the saved pre-edit source with the exact candidate compiler flags
produced four byte-equality passes in `matched_build_comparison.json`.
Post-test provenance is labeled accordingly; short CPU timings are not a
repeated or isolated GPU benchmark.

## Staged application and complete-cache GPU checks

`metal_staged_validation_v1` reached 78 nodes/35 accepted cells at55.133s,
versus62/26 at57.1762s for the earlier diagnostic. Both reached their60s
budget without a render; these unequal checkpoints are not a repeated
completed-render timing comparison.

A subsequent four-fixture complete-cache GPU control (`metal_staged_complete_v1`)
failed: conductor cases completed with the CPU topology, but dielectric case2
used71 nodes/36 cells/no quadratic cells versus CPU1 node/1 quadratic cell;
dielectric case3 exceeded1023 nodes. Control exit1 is preserved. Candidate
comparison is now running; this newly exposed discrepancy is unresolved and
blocks accepting the staged candidate as fully validated.

## Dielectric quadratic rejection isolated

Both complete-cache GPU versions terminated exit1: dielectric case3 exceeds
1023 nodes, and case2 uses71 nodes/36 non-quadratic cells in both. Conductor
solve counts fall595→465 and783→601 with unchanged cell topology. Candidate
timings are diagnostic only; CPU compilation overlapped the later work.

An instrumented, unmodified-threshold run of case2 confirms quadratic-control
passivity rejection: the first12 recorded minimum dissipations range from
−4.3003e-7 to−1.3844e-7, below the−1e-9 threshold. It completes using linear
cells (exit0), reproducing71 nodes. This identifies why quadratic fits are
rejected, not proof that every dielectric discrepancy has one cause. No
passivity gate was relaxed. Evidence and instrumentation sources are retained
in `metal_dielectric_passivity_v1`; all GPU jobs are terminal. A structure-
preserving lossless formulation remains to be investigated.

## Lossless structure check

The27 root fitting queries for the dielectric case (N4, retained2, pitch740nm,
depth150nm, duty0.41, ridge/substrate1.5, groove/incident1) show maximum
CPU/GPU complex-component difference3.48591298e-6. Nevertheless the maximum
Frobenius unitarity residual ||S* S−I|| is1.11839049e-5 on GPU versus
5.72124907e-15 on CPU. No matrix projection or tolerance adjustment was used.
Evidence: `metal_lossless_structure_v1`, exit0,27 samples. This fixed-order
check measures arithmetic/structure error, not convergence to physical truth.
The existing tensor cache rejects unitarity residual above1e-8, so switching
to that cache alone cannot resolve this discrepancy.

Research direction: Dieci, Russell and Van Vleck, “The Cayley Transform in the
Numerical Solution of Unitary Differential Systems” (1997), institutional
record https://ricerca.uniba.it/handle/11586/40766, describes integration in
skew-Hermitian coordinates to preserve unitary structure. Applicability to
the current mixed propagating/evanescent reference-port formulation still
needs derivation and validation; no claim of a completed fix is made.

## Experimental bounded polar correction (not production)

Tested two host-double Newton–Schulz steps X←X(3I−X*X)/2 on final GPU
responses for constant-index lossless fixtures only. Initial Frobenius
unitarity residual must be≤1e-4; correction per component must be≤1e-4;
post-correction residual must be≤1e-12. Larger errors fail instead of being
normalized. This explicitly modifies the numerical response; it is not an
unmodified GPU result and is not enabled in production.

For27 fitting queries, maximum correction3.51033064e-6, CPU component error
3.48591298e-6→4.96137496e-7, unitarity residual1.11839049e-5→1.60027707e-15.
Independent CPU comparison gate3e-4 passes. Evidence:`metal_lossless_polar_v1`.
Method source: Nakatsukasa/Higham, backward stability of polar iterations,
https://eprints.maths.manchester.ac.uk/1717/1/paper.pdf (Newton–Schulz equation6.5).

Wrapping the four bounded cache fixtures with this experimental correction
leaves conductor processing unchanged and allows both dielectric caches to
finish under unchanged passivity/interpolation gates:1 node/1 quadratic cell
and7 nodes/4 quadratic cells, matching CPU tree/layout/port buffers. The old
dielectric cases required71 nodes and exceeded1023 nodes respectively. All
four complete, exit0. Packed matrices and chart rotation data differ slightly
from CPU: maximum scalar coefficient difference2.365e-5 overall,2.116e-6 for
dielectric cases. The first byte-equality assertion failed; per-buffer
comparison records the actual differences instead. These are coordinate
coefficients, not a held-out physical-power validation. Evidence:
`metal_staged_polar_v1`. Higher-order, cutoff, reciprocal/physical-power,
held-out cache, rendering and performance validation remain outstanding.
All GPU jobs are terminal.

## Held-out physical response checks for experimental polar cache

Added explicitly bounded uniform sampling to the packed audit helper; its
existing full-domain/cutoff sampling remains the default. The bounded fixture
harness uses1024 independent points per case, seed617923, fixed-N4 CPU
physical scattering as reference, with the unchanged held-out threshold0.002.
All4096 queries are physical, with zero query failures. Worst power-column
errors for the four Metal cases:0.0011161821,0.0002649652,0.0006634690,
0.0001441490. This is not full-domain cutoff coverage or modal convergence.

The first conductor case has32 samples above construction tolerance0.001,
although it passes the pre-existing0.002 held-out gate; that limitation is
reported rather than hidden. Matched CPU-built cache audit is preserved in
`metal_staged_polar_heldout_v1/cpu_comparison.json`. GPU and final CPU runs
exit0; an initial premature CPU launch returned127 while compilation was
still pending and is retained in `cpu.log`. No production polar correction
has been enabled. All GPU jobs are terminal.

## Higher-order and cutoff coverage for experimental polar correction

`metal_lossless_polar_orders_v1`:108 queries,27 each atN4/16/32/64, all
predeclared correction and independent CPU-component gates pass (exit0).
Maximum corrected CPU error remains below4.962e-7; maximum correction below
3.529e-6. No production response policy was changed.

`metal_lossless_polar_cutoffs_v1`:96 queries across the same orders, both
ky signs, air/substrate first-order thresholds with±1e-3,±1e-5,±1e-7
offsets. All gates pass, exit0. Maximum raw CPU error3.11267814e-6, corrected
CPU error9.63931010e-7, correction3.06762435e-6 and corrected unitarity
residual1.93300494e-15. Prelaunch identities/gates are preserved. These
are fixed-order arithmetic/structure checks for one lossless profile, not
physical modal convergence, broad-profile certification or render validation.
All GPU jobs are terminal.

## Six-profile lossless correction coverage

`metal_lossless_polar_profiles_v1` completes324 fixed-order comparisons
(6 profiles × N16/N64 ×27 wavelengths/directions), exit0. Profiles vary
pitch500–1600nm, depth0–500nm, duty0.1–0.9, incident index1/1.33, ridge
index1–2.4, groove index1–1.5 and substrate index1–1.5. Retained windows
cover propagating channels throughout each tested wavelength range.
All predeclared bounds pass: maximum raw unitarity residual3.58382823e-5,
correction6.80317600e-6, corrected unitarity3.25984044e-15, and independent
CPU component error3.79375332e-6 (gate3e-4). The CPU reference itself has
unitarity residual up to8.30968683e-11. Prelaunch hashes are in `plan.json`.
These tests cover arithmetic correction, not modal convergence or production
rendering. Combined direct-response coverage is528 queries; the correction
is still test-only, and absorbing responses are not corrected. All GPU jobs
are terminal.

## Bounded lossless correction integrated

Metal reference_backend now invokes the bounded two-step host-double polar
correction only when sampled ridge, groove and substrate imaginary indices
are all exactly zero. GPU response assembly remains unchanged. Absorbing
responses bypass correction. Initial defect1e-4, correction1e-4 and final
unitarity1e-12 gates are unchanged from the experiments; cache passivity and
interpolation acceptance are unchanged. Cache backend identity is now
`metal-mps-wide64-inverse-lossless-polar-v2` to prevent stale reuse.

Full Blender build passes. A production-helper unit regression verifies
small-defect correction and rejection of large defects, NaN and invalid
dimensions without modifying rejected data. The actual integrated backend,
without the experimental wrapper, completes all four bounded caches and
passes4096 held-out fixed-N4 physical-response comparisons, exit0. Evidence:
`metal_integrated_lossless_v1` with prelaunch identities and build log.
This is not yet an application-render pass, arbitrary-material convergence
proof, or a completed full-domain cache benchmark. All GPU jobs are terminal.

## Full-domain application rejects a previously untested response

`metal_lossless_application_v1` fails after3.315s (child exit1, no budget
limit reached). No render is produced. The integrated correction rejects a
response at the root cell rather than widening its1e-4 bound. Added precise
residual/query context to the error; full Blender rebuild passes. A fresh
reproduction (`metal_lossless_application_context_v1`) also exits1 and records:
residual0.0002933714755060846, wavelength446.66666666666669nm,
kx−0.30180180180180183, ky−0.83333333333333337, N16, retained2.
Profile:740nm pitch,150nm depth,float0.41 duty, ridge1.5, groove/incident/
substrate1, no absorption. Thus prior528 response checks and bounded cache
passes do not establish full-domain compatibility. This rejection must be
resolved before accepting the integrated revision. Both attempts, binaries
and logs are preserved; all GPU jobs are terminal. No acceptance gate was
relaxed and the selected installed Fast binary remains unchanged.

## Full-domain rejection isolated: scaling does not resolve it

Raw reproduction (`metal_lossless_failure_v1`) atN16/32/64 confirms GPU
unitarity residual2.9337e-4/4.1043e-4/3.6638e-4, while CPU residual is
2.42e-13–3.21e-12. Rounded GPU inputs change CPU response by only6.82e-8;
raw GPU component errors are3.565e-5–7.441e-5. Unconstrained diagnostic polar
results are recorded separately and were not used to bypass the production
1e-4 defect bound.

Initial propagation norms4 and32 were tested with64 Taylor terms. Neither
passes the unchanged raw-unitarity bound at this query; norm4 worsens N64
to0.0010375. Evidence:`metal_lossless_failure_scaling_v1`, both processes
exit0 as diagnostics, not numerical acceptance. Production scaling unchanged.
Source inspection identifies a plausible conditioning issue: the internal
average-permittivity admittance is near its own cutoff for one order at
this query. A changed internal reference basis is the next experiment;
physical material indices must remain unchanged. All GPU jobs are terminal.

## Conditioned internal reference basis

Experiment `metal_internal_basis_v1` adds imaginary0.25 to the internal
reference permittivity only when it is real (material shader kinds5/6).
Physical constitutive matrices and exterior admittances are unchanged.
At the isolated failure, raw-unitarity residualN16 falls2.9337e-4→2.5109e-6
and raw CPU component error3.5650e-5→2.6562e-7; N32/N64 improve similarly.
This raw comparison bypasses the polar correction and does not alter the
independent CPU model.

The same alternate shader with integrated correction passes324 six-profile
queries and96 cutoff queries (both exit0). Maximum integrated CPU errors
are1.8772e-6 and5.1572e-7. These latter outputs include the correction; they
are not raw-GPU residual claims. Production shader now contains this basis
conditioning, with backend identity `metal-mps-wide64-conditioned-lossless-v3`.
Full Blender rebuild is running; the previously failing full-domain
application test has not yet been repeated. All GPU jobs are terminal.

## Full-domain dielectric application render passes

`metal_conditioned_application_v1` completes with child exit0 in195.578s,
within its predeclared300s limit. Apple M5 only, PT,1024 fixed samples,
adaptive sampling/denoising off. The full-domain tensor cache builds in
148.444s:13 cells,4,470,640 matrix bytes, sampled power error0.000202208.
The lossless shader selects the Hermitian tensor cache; earlier standalone
four-fixture tests used quadratic cells, so this is separate pipeline evidence.

Actual EXR mean RGB=[1.0000656303,1.0000482961,0.9995937114], compared
with analytic unit-radiance furnace reference1. Maximum mean error0.000406289
passes the unchanged0.005 smoke gate. Serialization, finite pixels and EXR
identity are recorded in the scene report. Standard-view PNG preview was
generated from this EXR and inspected. The near-white result is expected for
this energy-conservation fixture; it is not a CD appearance demonstration.
This single completed run is not a repeated performance benchmark or proof
of modal convergence, BDPT/guiding/coherent transport or all materials.
All GPU jobs are terminal.

## Four-transport analytic application suite started

`metal_conditioned_transport_v1` runs the saved successful full-domain
dielectric fixture through PT, BDPT, guided PT and guided BDPT sequentially
in one Blender process,1024 fixed samples each, adaptive/denoising off.
The independent reference is unit radiance for each method; there is no
normalization against PT. Per-method EXRs, PNGs, saved scenes and incremental
reports are produced by `cycles_diffraction_transport_furnace.py`.
Persistent data is enabled, but cache reuse must be verified in the log.
Prelaunch identities and a1200s suite limit are recorded. Session32321 is
active, currently building the first cache; poll it before another GPU job.
No completed suite result exists yet.

The four-transport session32321 remains active. PT has completed and passes
with maximum RGB-mean error0.0004062889; BDPT is rendering. No acceptance
claim is made for the remaining methods.

Separately, tensor validation now skips further probes after its cell has
already exceeded the existing power/complex rejection threshold. Refinement
curvature derives from the fitted tensor, so this does not remove inputs to
the split heuristic. Accepted cells still run every probe. Matched baseline
and candidate CPU objects compile. Full-domain CPU buffer comparison is
running in session30594 (`tensor_early_rejection_v1`), not yet passed.
The live Blender binary predates this change. Overlapping CPU work further
excludes the four-transport run from isolated timing claims.

## Four-transport furnace suite complete; tensor rejection optimization

Session32321 is terminal: suite exit0,419.944s, no budget limit reached.
PT/BDPT/guidedPT/guidedBDPT all render1024 fixed samples on Apple M5 with
adaptive/denoising off. Each independently passes unit-radiance reference
with maximum RGB-mean error about0.000406289 (gate0.005). All four EXR hashes
match reports; PNGs were inspected. Log shows1 cache build and1 engine
initialization,0 explicit reuse events: the persistent session retained its
cache, rather than logging three cache lookups. Actual images are near-white,
as expected for the furnace; this does not exercise difficult indirect
transport or prove guiding quality, modal convergence or coherent transport.
Evidence:`metal_conditioned_transport_v1`. CPU comparison/build work overlapped,
so per-method times are diagnostic only, not isolated performance rankings.

Tensor early rejection now has two full-domain matched CPU comparisons with
byte-identical buffers. Power-only:8238→6895 solves,25 nodes/13 cells. With
complex tolerance0.001:12677→10604 solves,39 nodes/20 cells. All accepted-cell
validation remains complete; only already-rejected cells skip later probes.
Full Blender build passes. Evidence:`tensor_early_rejection_v1` and
`tensor_early_rejection_complex_v1`. GPU/render validation of that optimization
is still pending; the completed four-method binary predates it. All GPU and
CPU comparison jobs are terminal.

## Repeated Metal benchmark in progress

`metal_tensor_rejection_benchmark_v1` freezes control/candidate binaries and
scene-script identities before six independent Blender runs in order
control,candidate,candidate,control,control,candidate. Each uses Apple M5,
1024 fixed samples, seed11, adaptive/denoising off, full-domain lossless
furnace, and a300s limit. Each must pass the original analytic smoke gate.
Median logged cache-build time is the primary metric; total process times
are separate. No other GPU jobs or CPU builds are scheduled concurrently.
A supervisor syntax error prevented its initial launch (no GPU work started);
it was corrected and syntax-checked before the actual launch. Session47820
is active, in the first control cache build; poll this handle before another
GPU job. No speedup or final selection is claimed until all runs complete.

## Repeated Metal benchmark complete: tensor optimization retained

Session47820 is terminal. All6 independent PT processes pass analytic
furnace validation at1024 fixed samples, seed11, adaptive/denoising off.
EXR identities verified. Control cache seconds148.865/148.523/148.092;
candidate125.051/124.639/124.480. Median preparation148.523→124.639s,
16.08% lower. Median total process time187.367→169.410s,9.58% lower.
No competing GPU jobs or CPU builds were launched during these measurements.
Maximum RGB-mean error across runs0.0004062895314, below unchanged0.005.
Evidence:`metal_tensor_rejection_benchmark_v1/verified_summary.json` and
`report.md`, with raw logs, saved scenes and EXRs retained.

Retain tensor early rejection for this validated workload. It is not final
acceptance of all diffraction requirements; the installed selected Fast
binary is unchanged. Cache construction is still over2 minutes and is not
the default path. All benchmark processes are terminal.
