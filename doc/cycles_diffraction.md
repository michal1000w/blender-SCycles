# Spectral diffraction and coherent transport development

## Scope and status

User request (2026-09-25): spectral grating materials throughout applicable Cycles
nodes, Metal execution, PT/BDPT/guiding support, numerical and visual validation,
and measured performance. Subsequently expanded to also include coherent
interference between separate objects. Both are required; neither is complete.

Current code includes a dedicated diffraction node with a cache-free approximate
Fast default and optional Realistic response caches. The latter now has a Metal
builder with bounded host-double correction for lossless reference responses
and a conditioned internal basis. A full-domain dielectric furnace passes in
PT, BDPT and both guiding configurations at fixed samples; this does not prove
arbitrary-profile convergence or complex indirect transport. The selected Fast
binary and its 72-render suite remain separate from the newer Realistic backend
under validation. All appropriate shader integration is still incomplete.
See `cycles_diffraction_completion_audit.md` for current requirements and
`cycles_diffraction_gpu_builder_experiment.md` for accepted and failed experiments.
The chronological prototype results below are retained as historical evidence,
not declarations that the latest implementation or original scope is complete.
Coherent transport is explicitly requested as an opt-in mode, disabled by default;
ordinary grating rendering must not allocate or process coherent transport state.

## Research

* [Stam, Diffraction Shaders (1999)](https://www.josstam.com/publications):
  scalar surface Fourier optics is the starting point for local grating scattering.
  The [GPU Gems simplification](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-8-simulating-diffraction)
  demonstrates CD geometry, but its additive rainbow-color approximation is not
  suitable as an energy-consistent Cycles closure.
* [Cuypers et al., Ray-Based Reflectance Model for Diffraction (2011)](https://arxiv.org/abs/1101.5490):
  Wigner scattering can encode interference using signed contributions. Its
  paraxial and polarization assumptions must be distinguished from general
  electromagnetic transport.
* [Steinberg et al., Towards Practical Physical-Optics Rendering (2022)](https://sites.cs.ucsb.edu/~lingqi/publications/202203_practical_plt_paper_lowres.pdf):
  section 4.1.3 derives sinusoidal-grating Fourier orders and discusses sampling
  overlapping diffraction lobes. Its coherence-aware transport is more extensive
  than a local spectral BSDF.
* [A Free-Space Diffraction BSDF (2024)](https://research.nvidia.com/labs/rtr/publication/steinberg2024diffraction/):
  triangle-edge Fraunhofer diffraction addresses geometric apertures and edges,
  a different problem from a periodic surface profile.
* [Yang et al., A wave-optics BSDF for correlated scatterers (2025)](https://cs.dartmouth.edu/~wjarosz/publications/yang25wave.html):
  separates particle diffraction from spatial correlation; relevant to future
  non-periodic microstructure, not a direct grating implementation.
* [Steinberg and Pharr, Wave Tracing (Computer Graphics Forum / Eurographics, April 2026)](https://research.nvidia.com/labs/rtr/publication/steinberg2026pathintegral/):
  bilinear and weakly-local transport provide the relevant framework for the
  requested inter-object interference. A conventional intensity path state alone
  cannot represent the cross terms.
  The [published supplement, S1.1.1](https://research.nvidia.com/labs/rtr/publication/steinberg2026pathintegral/wave_tracing_supplemental.pdf)
  states the time-averaging and wave-ensemble assumptions under which different
  wavelengths contribute incoherently. A cross-object coherent mode must
  preserve interference within a spectral sample without summing unrelated
  wavelength amplitudes. This is a transport requirement, not a completed feature.
* [Reference wave_tracer repository](https://github.com/ssteinberg/wave_tracer):
  the authors describe a CPU-only early alpha with approximately 5–20 times the
  cost of ray tracing, depending on scene. This is not a performance prediction
  for this fork. Its CC BY-NC 4.0 code is not being copied into Blender.
* [PLTFalcor](https://github.com/ssteinberg/PLTFalcor): spectral/coherence-aware
  material and transport reference, also noncommercial licensed; not copied.
* [Maxon grating OSL example](https://github.com/Maxon-Computer/Redshift-OSL-Shaders/blob/main/DiffractionGrating.osl):
  explicitly an RGB shifted-microfacet approximation, not the requested spectral
  implementation.
* [Newport grating efficiency](https://www.newport.com/n/efficiency-characteristics)
  and [Plymouth Grating Laboratory](https://www.plymouthgrating.com/guidance/technical-notes/fundamentals/diffraction-efficiency/):
  order direction and order efficiency are different quantities. Scalar Fourier
  amplitudes do not replace rigorous vector efficiencies near wavelength-sized
  features. RCWA reference comparisons are desirable for CD/DVD groove profiles.

## Existing fork integration points

* `ShaderData::rand_wavelength`, `sample_wavelength`, and
  `update_path_throughput_for_dispersion` supply the existing hero wavelength.
* Shader runtime dispersion flags activate spectral throughput before direct
  lighting. Both light and camera subpaths must obey the same contract.
* BDPT currently connects spectral paths with a wavelength reconstruction kernel;
  validate its behavior with narrow diffraction orders rather than assuming the
  glass tests imply grating correctness.
* SVM node structures, scene nodes, Blender declarations, OSL shaders/closures,
  closure classification, albedo/passes, reverse PDFs and guiding all need audit.
* A disabled grating must retain the existing closure and avoid additional
  wavelength sampling or per-path coherent state costs.

## Validation gates

1. Kinematics: vector grating equation, conical incidence, transmission IOR,
   reversed paths, grazing/evanescent orders.
2. Profiles: independent numerical Fourier integration, Parseval, flat limit,
   depth, duty cycle, and wavelength response.
3. Closure: sample/eval agreement, PDF mass including null events, reciprocity
   or the correct adjoint relation, white-furnace energy, roughness limits.
4. Transport: CPU/Metal and SVM/OSL agreement, PT/BDPT/guiding convergence,
   caustics, lighting types, textured frames, bump, motion, AOVs and mixtures.
5. Coherence: two-beam phase sweep; double-slit fringe spacing/envelope;
   independent sources; finite bandwidth visibility; multiple objects; distance
   and scale changes. Averaging noisy fields and then squaring them introduces
   estimator bias and is not an acceptable substitute for these checks.
6. Visual fixtures: CD and DVD with distinct pitch and explicit protective layer,
   transmission grating, and geometric aperture interference scenes.
7. Performance: discarded warmup, repeated independent measured runs, rendered
   sample counts, equal-error comparisons, executable and kernel provenance,
   separate disabled/enabled costs. Preserve unsuccessful results as well.

## Initial machine baseline

Apple M5, 10 GPU cores, 16 GB unified memory; Release build, Metal and OSL enabled,
CPU OpenPGL and custom Metal guiding present. Initial unchanged glossy PT scene:
128×128, 64 samples, seed 0, 0.6004 s measured after 34.7476 s discarded warmup.
This is one baseline sample, not a statistically established performance result.
Raw outputs are in `tests/output/diffraction/baseline/`.

The installed build changed during development from `32ff8076d38d` to
`10a55bc29ef5` (existing guiding changes outside this work). The first PT and
guided PT runs used the former, while the first BDPT run used the latter.
Do not use these mixed-build observations as a comparative benchmark. Before
measuring changes, capture a matched baseline with executable and kernel hashes.

## Mathematical foundation checks

The eleven `KernelDiffraction` tests currently pass. They test the vector grating
equation, evanescent-order rejection, reflection and transmission reversal,
binary complex Fourier coefficients against independent double-precision
quadrature, Parseval's identity, groove translation phase, and the flat limit.
The coherent cancellation check is an algebraic unit test, not a claim of
inter-object coherent rendering.

The rough-facet reflection geometry additionally tests 50,000 deterministic
random configurations, both inverse branches, the reverse Jacobian identity,
and an independent finite-difference solid-angle Jacobian. Initial failures
identified an unstable projection when a root nearly coincides with the grating
axis. The inverse now retains the facet grating axis explicitly. The normal
round-trip bound accounts for the inverse's singular azimuth at axial symmetry;
every accepted root also meets a separate 3e-6 vector momentum residual bound.
The GPU harness now also exercises facet directions, both inverse roots and
forward/reverse solid-angle Jacobians. This exposed grazing cancellation in the
zero-order direction reconstruction; using the exact mirror formula fixed it.
A dedicated grazing CPU regression covers incident cosines down to 1e-5.

### Rough-facet geometry derivation

Let `h` be a facet normal, `T` the unit macro grating axis, and
`e(h) = normalize(T - dot(T,h) h)` its projection onto that facet. For reflection,
the grating equation is `wi + wo = a h + delta e`, with
`a = dot(wi,h) + dot(wo,h)` and `delta = m lambda / pitch`.
This equation can have two positive-facing solutions for `h`; evaluation must
sum both preimages, while normal sampling naturally chooses one.

Projecting its differential onto the facet gives two eigenvalues: `a` along `e`,
and `b = a - delta dot(T,h) / length(cross(T,h))` across `e`. Therefore the
solid-angle change of variables is `J_forward = abs(dot(wo,h)/(a*b))`.
The reverse map replaces the numerator by `dot(wi,h)`, establishing
`dot(wi,h) J_forward = dot(wo,h) J_reverse`. At order zero this reduces to the
ordinary microfacet reflection Jacobian. The implementation checks the
direction equation independently of this derivative.

The geometry alone does not define a BSDF: it also needs per-order power,
masking, wavelength throughput, normal-distribution sampling, and mixture PDFs.
In particular, Parseval for scalar Fourier coefficients alone does
not prove physical energy conservation after selecting propagating orders and
applying obliquity or Fresnel terms. Do not normalize direction-dependent
efficiencies independently in each direction without checking reciprocity.

### Reflection scattering prototype

`kernel/closure/bsdf_diffraction.h` now implements a local GGX-facet reflection
sampler and evaluator. It sums all propagating diffraction orders and both
normal preimages, and reports the complete solid-angle mixture density. Smooth
facets use discrete order probabilities; their directional evaluation is zero.
It uses the existing anisotropic GGX visible-normal sampler and Smith masking.

Its provisional power model retains binary phase-screen nonzero powers at the
actual angle pair and puts the residual in order zero. This is explicitly a
flux-normalized scalar approximation, **not** an electromagnetic boundary
solution. Nonzero powers are bounded by `4 sin(pi*m*duty)^2/(pi*m)^2`, whose
two-sided sum is `4*duty*(1-duty)<=1`. The specular completion is passive and
reciprocal. It avoids independently renormalizing forward and reverse lobes.
Accuracy against physical groove efficiencies is still unproven and must be
tested before choosing this as the final material model.

Five `KernelDiffractionBSDF` tests pass: 10,000 randomized facet power checks,
20,000 rough reciprocity checks, the flat-profile GGX limit, smooth sampling,
and independent uniform-solid-angle integration versus sampling at incident
cosines 0.3 and 0.9 (500,000 samples per estimator at each angle). The latter
checks probability mass including null events, furnace energy, and two
directional moments using estimated integration uncertainty. These validate the
mathematical model, not its agreement with Maxwell's equations or GPU rendering.

Further efficiency research:
[Madsen (2021)](https://opg.optica.org/ao/abstract.cfm?uri=ao-60-9-2695)
compares nonparaxial scalar estimates against vector simulations and describes
corrections near order cutoffs. Only the abstract was accessible; its reported
accuracy must not be attributed to the present implementation. The
[KostaCLOUD manual](https://docs.kostacloud.com/kostacloud/advanced-optics-ray-trace/gratings-kinoforms-and-phase-masks)
also distinguishes a scalar efficiency option from direction-only grating tracing.

`tests/metal/cycles_diffraction_test.py` compiles the actual kernel utility header
with Metal fast math and runs 16,384 deterministic cases on the GPU. On Apple M5,
the original direction/Fourier cases pass a 1e-5 absolute tolerance (maximum
error 4.70877e-6). Rough geometry passes a 2e-4 scaled-error bound, with a
maximum 7.84741e-5; validity and inverse-root counts match exactly. It records
source digests and device identity in its JSON report. Run outside the sandbox:

```
python3 tests/metal/cycles_diffraction_test.py --output tests/output/diffraction/metal_math.json
```

The harness uses runtime Metal compilation, matching Cycles. Apple Metal
Toolchain 27A266a was subsequently installed because the full Blender build
also precompiles Metal shaders. This numerical harness does not validate a shader, a rendered image,
transport integration, or rendering speed.


## Independent electromagnetic efficiency check

`tests/python/cycles_diffraction_reference.py` implements a separate TE Fourier
modal boundary solve for one binary lamellar layer at nonconical incidence.
It derives the eigenproblem from the TE Helmholtz equation and matches electric
field and its normal derivative at both interfaces. It does not import the
closure's Fourier power formula into the reference solve. Its scope is TE only;
it is not yet a validated general vector/conical reference.

Initial reference checks: zero-depth planar Fresnel reflection error below
6.7e-16 for dielectric and lossy substrates, and three nontrivial lossless
boundary/energy checks below 1e-10. Truncations 16, 32, 48, 64 establish numerical
convergence for six synthetic conductor cases (n=0.9+6i, not measured aluminum).
The last truncation changes summed reflected order powers by 1.8e-5 to 1.2e-4.
Pitch is 740/1600 nm, depth 150 nm, duty 0.5, wavelengths 450/550/650 nm.

The current scalar model, scaled by the conductor's planar reflectance, differs
by **0.045 to 0.460 in summed absolute order-efficiency error**. Notably, the
vector boundary problem produces even-order reflection where the binary phase
screen's even orders vanish. This is a material physical-fidelity failure, not a
sampling bug, and rules out claiming the present model as a validated realistic
metal-relief solution. More accurate efficiencies and polarization/conical
reference coverage remain required. Raw results: `tests/output/diffraction/te_reference.json`.


The independent TE reference was cross-checked against
[NIST SCATMECH's RCW solver](https://pages.nist.gov/SCATMECH/docs/rcwmodel.htm)
through pySCATMECH 0.1.9 in a temporary external virtual environment, with 64
Fourier orders on each side. Maximum reflected TE order-efficiency discrepancy
across all six cases was 1.03156e-7. The report also records NIST's TM efficiencies,
which are not represented by the scalar closure. The NIST package is used only
as an external validation dependency, not linked into Blender.

```
/tmp/cycles-grating-reference/bin/python tests/python/cycles_diffraction_reference.py \
  --nist --output tests/output/diffraction/te_reference_nist.json
```

The original installed app was preserved at `/tmp/cycles-diffraction-baseline.app`
before installing the new build, for subsequent matched regression/performance
checks. CPU kernel and scene-node compilation, OSL shader compilation, Eevee
shader validation, and full linking passed. This does not establish runtime
correctness for those paths.


## Initial integrated renders

`tests/python/cycles_diffraction_scene.py` builds CD/DVD/plane fixtures, records
linear EXR/NumPy buffers, display PNGs and provenance, and optionally saves the
Blender scene. At 128x128 and 512 samples, the CD fixture's CPU/SVM and CPU/OSL
buffers matched exactly. Metal PT relative L2 difference from CPU was 4.28e-6.
Combined Metal BDPT/guiding completed with relative L2 difference 0.00150 from
PT. This predominantly directly lit scene is only an integration smoke check;
it does not validate spectral BDPT connections, caustics or guiding convergence.
The image was visually inspected. It shows two spectral reflection bands on a
radial annulus; this does not resolve the physical efficiency failure above.

The first 128x128, 64-sample Metal run took 0.354 s after a 15.57 s warmup. This
single end-to-end observation is not a performance claim. Repeated baseline,
disabled and enabled measurements are being collected separately, without
concurrent CPU/GPU workloads.

### First repeated performance check

Five warmed Metal PT runs per variant, fixed seed, 256x256, 1024 samples:

| Variant | Median end-to-end seconds | Range |
| --- | ---: | ---: |
| Preserved original build | 0.7816 | 0.7678–0.8238 |
| New build, grating disabled | 0.7666 | 0.7578–0.8205 |
| New build, grating enabled | 0.9744 | 0.9394–0.9998 |

These include EXR writing, use consecutive batches rather than randomized
interleaving, and cover one directly lit fixture only. Disabled image relative
L2 error vs the original is 8.34e-8. The disabled median difference is within
observed timing variation; do not claim a speedup. Enabled median overhead is
27.1% vs disabled, before physical-efficiency improvements or coherent transport.
This is not an equal-error benchmark. All source/binary hashes and raw durations
are in `tests/output/diffraction/pilot_benchmark.json`.

### Incident-medium contract

Glossy now exposes `Diffraction Medium IOR` (default 1). SVM and OSL convert the
sampled vacuum wavelength to lambda/n before evaluating both order direction and
relief phase. The value is explicitly the medium above the grating, not the
substrate index; the renderer does not infer it from the volume stack. The
`--cover` fixture supplies 1.58, matching its glass layer.

A 512-sample image similarity test compares n=1.58/pitch=1600/depth=150 to
n=1/pitch=2528/depth=237. Metal vs the equivalent CPU image has relative L2 error
3.14e-6; OSL vs the equivalent CPU image has error 3.78e-7. This validates optical
length scaling through the node integration, not physical groove efficiencies.
Full layered-material and dispersive-medium integration remains required.

## Vector electromagnetic reference

`tests/python/cycles_diffraction_vector_reference.py` solves transverse Maxwell
fields with both polarizations and conical incidence. Permittivity uses the
inverse Fourier factorization for the electric field normal to groove walls.
E and H are matched across a single lamellar layer; incident s and p states are
solved together, retaining the complex transverse fields for later phase and
polarization checks. Production sampling and shader code do not yet use this
solver.

Sixteen cases (dielectric/absorbing metal, flat/150 nm relief, normal/oblique/
conical incidence) agree with NIST SCATMECH at the same 24-order truncation to a
maximum order-efficiency error of 9.35e-8. Lossless energy and interface residuals
also pass 1e-10 bounds. Equal-truncation agreement verifies the implementation,
not convergence to the infinite-mode solution.

A separate 16/24/32/48/64-order study shows slower metal TM convergence: the
summed change in both polarizations from 48 to 64 is 0.00094–0.00278 for three
550 nm cases. Single observed CPU solves range from about 3 ms (16 orders) to
77 ms (64 orders); these are development estimates, not production benchmarks.
Per-hit solves cannot meet the requested speed. A cached electromagnetic response
needs error-controlled angular/spectral interpolation, correct order cutoffs,
reciprocity, passivity and complex polarization conventions before replacing the
scalar prototype. Textured material parameters must not silently fall back to
an inaccurate model.

Exact grazing external orders and zero-eigenvalue layer modes need limiting
solutions; these are not covered by the current vector reference. Reflection
and transmission response, including both interfaces' different indices, must
be validated before a production implementation is considered complete.

Raw records: `tests/output/diffraction/vector_reference_nist.json`,
`vector_reference_convergence.json`, and `medium_similarity.json`.
Transmission and cross-object coherent transport still have no renderer
implementation. Coherent transport remains an opt-in requirement, not an
implemented switch.

## Covered fixture and MNEE scale investigation

The first cover fixture incorrectly placed its lower face 1 micrometer above
the reflector, creating an air gap inconsistent with the supplied medium IOR.
It is now 1 micrometer below the opaque reflector; the upper interface is 1.2 mm
above it. Earlier `pilot_cd_cover*` images are retained but are not physical
validation of an embedded grating. Corrected images use the `embedded_*` prefix.

The corrected fixture still lost nearly all direct light with MNEE. A 100x
macroscopic scale test (source power scaled by 10000 to preserve radiance,
grating wavelength/pitch/depth unchanged) recovered the illumination. Inspection
identified two existing scale-dependent surface-MNEE approximations:

* The Newton walk rejected steps below 0.1 mm, despite needing much smaller
  corrections through a 1.2 mm cover. Surface calls now disable that absolute
  cutoff, as volume calls already did; iteration and line-search limits remain.
* The generalized geometry factor was capped at 2 despite having inverse-area
  units. The cap therefore changed radiance with scene scale. It is removed;
  singular systems are still rejected by the transfer-matrix solve.

At 128x128/512 samples, after both changes, scale 1 vs 100 mean radiance differs
by 0.0205% (image relative L2 0.00312). Metal vs CPU relative L2 is 0.000127 and
mean difference 0.00054%. These are regression observations, not complete MNEE
validation. The default physical-size scene is no longer dark. Independent
ordinary-PT energy checks are being collected. The unclamped method can reveal
high-variance caustics previously suppressed by the biased cap; broad performance
and regression coverage remain required before accepting this as final.

Saved records: `mnee_scale_comparison.json`, `embedded_cd_mnee_unclamped_*`.

## Coherent transport architecture investigation

The [Wave Tracing preprint, sections 4–5](https://arxiv.org/html/2508.17386v1)
distinguishes a path-pair formulation from region transport using Gaussian
beams. The former has globally cancelling contributions that ordinary local
sampling and guiding cannot efficiently predict. The latter gathers geometry
within a beam footprint and evaluates local interference, supporting nonnegative
sample contributions. Its proposed sampling strategy includes the interference
terms, not merely independent intensities. This is a reason to investigate
region transport for the requested opt-in mode, rather than adding a phase
number to the current intensity path state.

Repository-specific implementation requirements still to solve:

* A separate enabled-only beam state and conservative multi-triangle region
  query, including triangles belonging to different objects. Ordinary MetalRT
  closest-ray queries alone do not provide this region query.
* Explicit source coherence and polarization, detector integration, finite
  bandwidth, and precision sufficient for optical phases in meter-scale scenes.
* Complex scattering response with a common phase origin and polarization basis.
  The host Maxwell solver now validates complex reciprocal scattering blocks
  in a common Cartesian polarization gauge; coherent transport and its region
  scattering operators still need implementation.
* Defined treatment of rough surfaces and volumes; taking square roots of their
  intensity BSDFs would not supply a valid coherent scattering operator.
* Separate tests for cross-object interference and its incoherent limit. No
  algebraic Fourier unit test or colorful local material render substitutes for
  those tests.

No coherent renderer option is exposed yet, because no working coherent
transport implementation backs it.

Four independent ordinary-PT renders at 65,536 samples per seed were compared
with four MNEE renders at 4,096 samples per seed. MNEE mean radiance is 2.91%
lower. Per-channel differences are -1.86, -1.51, and -4.08 estimated combined
standard errors; only four seeds are available, so these are not Gaussian
significance claims. Image RMSE is 1.055 times the estimated sampling-noise RMS.
The blue-channel discrepancy is unresolved and prevents accepting the MNEE
changes as fully validated. A tighter surface constraint threshold (1e-5,
matching the existing volume endpoint setting) is being tested. Full raw data:
`tests/output/diffraction/mnee_pt_energy_comparison.json`.

The tighter-constraint experiment reduced mean radiance by 19.4% relative to
PT and was reverted; failing more walks while the corresponding camera paths
remain culled cannot improve energy correctness. Results are retained in
`mnee_tight_failed.json`. Further inspection found that receiver ancestry was
never reset after an intervening reflection or other scattering event. Culling
now follows a reflective receiver plus a contiguous refractive-caster chain;
internal reflections terminate the chain. Transparent continuations preserve
its state. This partition correction is under independent energy testing; it
does not establish complete MNEE unbiasedness or solve all failed-walk cases.

Additional material limitations of the current prototype: its grating fraction
uses a single-scattering GGX facet model even when the original Glossy fraction
uses another distribution. Distribution-specific scattering and multiscattering
energy compensation must be addressed when integrating the final material
response. The high-order pitch/index range also needs bounded-cost sampling
without silently suppressing valid orders. These are outstanding requirements,
not supported production behavior.

After correcting the transmission-chain partition, four 4,096-sample MNEE runs
agree with the four 65,536-sample PT references to 0.0643% in mean radiance.
Channel differences are -0.55, -0.26, and +0.78 estimated combined standard
errors. Restoring the previously discarded paths increases variance, especially
in blue. This supports the correction for the tested covered-disc fixture,
without proving general MNEE convergence or unbiasedness. Record:
`tests/output/diffraction/mnee_chain_energy_comparison.json`.

## Host electromagnetic response generation

`intern/cycles/scene/diffraction.cpp` now implements vector Fourier-modal
Maxwell response generation in C++ using Eigen. This is built into the Cycles
scene library, but it is **not connected to the rendering closure yet**. The
visible grating material therefore still uses the provisional scalar model
described above. There is still no cross-object coherent renderer.

The host solver uses inverse Fourier factorization normal to the groove walls,
stable evanescent propagation, and interface matching. It retains complex
flux-normalized reflection and (for a lossless substrate) transmission. For
absorbing substrates, the single-incidence response distinguishes flux entering
the substrate from far-field transmission. Phase origins are the upper and lower
interfaces, respectively; nanometer lengths and passive complex indices are
explicit inputs.

One modal decomposition can solve a complete Bloch scattering block containing
all propagating incident/outgoing diffraction channels. Its smooth Cartesian
polarization basis avoids the normal-incidence singularity of s/p coordinates.
The full matrix passes reciprocity under port reversal and transposition;
its largest eigenvalue of S†S checks passivity for arbitrary coherent channel
superpositions, rather than only one incident polarization at a time. Direct
normalized wavevector coordinates allow substrate-only propagating channels.
A Fourier window that omits any propagating channel is rejected explicitly.
Exact external grazing and degenerate layer modes still require limiting
solutions; the solver reports these cases as errors, not fallback materials.

The compact ordinary-rendering representation stores one unpolarized intensity
per port pair (half the squared Frobenius norm of the Jones submatrix). Positive
interpolation in a common port indexing preserves the column power bound and,
with symmetric sampling, reciprocity. Tests check interpolation accuracy away
from a propagation cutoff and passivity across one. The latter does **not**
establish cutoff accuracy; an error-controlled cache and kernel lookup remain
outstanding. Coherent transport will need the complex representation separately.

Seventeen host tests pass, covering complex Fresnel limits, energy, arbitrary
polarization passivity, incident-medium similarity, four independent NIST
conical reference cases, complete-channel reciprocity, interpolation, direct
Bloch coordinates, substrate-only channels, and invalid/truncated input.
The independent NIST values use a synthetic metal index, not measured aluminum.
Record: `tests/output/diffraction/host_solver_tests.xml`.

`tests/performance/cycles_diffraction_solver.py` measures host precomputation
with five alternating-order repeats. The full block includes lower-side
incidence for dielectrics; the separate comparison covers upper incidence only.
Both use the same modal count but different finite Fourier windows, so the
ratio is not an equal-error performance comparison. This is not a GPU rendering
benchmark and does not update the earlier prototype render timing results.

## Packed spectral response cache: implementation and failed accuracy audit

The host solver now produces compact float intensity blocks read by
`kernel/util/diffraction_table.h`. Upper and lower propagating orders occupy
contiguous ranges, permitting constant-time port lookup. Packing checks shape,
ordering, finite nonnegative coefficients, and column passivity. It does not
renormalize physical powers. Empty far-field channel sets are valid empty blocks.

`diffraction_grating_build_grid` generates a wavelength/Bloch/ky grid with
cancellation, explicit modal-error propagation, and a 64 MiB packed-data limit.
`kernel/util/diffraction_grid.h` performs eight-node interpolation, shifts port
indices at the periodic Bloch seam, and projects onto physically propagating
query channels. It is not connected to the material closure yet. A successful
build does not establish acceptable interpolation error.

Host tests cover packed Maxwell powers, Fresnel interpolation, spectral bounds,
periodic seams, all-order reciprocity and passivity, and cancellation. The
Apple M5 GPU passed 16,384 packed-port addressing cases exactly and 16,384
spectral cache queries within 7.75e-7 of CPU. These GPU checks validate lookup
arithmetic against CPU, not approximation accuracy against Maxwell. The cached
fixture uses 8x8x3 samples and N=8 solely to exercise the reader. Record:
`tests/output/diffraction/metal_math_grid.json`. The subsequent grazing-extension
GPU recheck also passed all cases, with maximum cache error 7.45e-7:
`tests/output/diffraction/metal_math_grid_grazing.json`. The full install build
passed in `build_host_grid_grazing.log`.

The independent cache audit is deliberately more demanding. It samples 256
incident directions uniformly in solid angle and wavelengths uniformly over
380–780 nm for synthetic-metal profiles with pitches 740 and 1600 nm. It compares
cached reflected-order powers against direct N=16 block solves, and separately
compares N=16 against N=32. Error is the sum of absolute order-power differences
per incident direction, not image error or relative percentage error.

The 8x16x9 zero-padded grids FAILED: mean L1 errors were 0.1833 and 0.1859,
with maxima 0.9805 and 0.9991. Refining angles to 16x32 while keeping nine
wavelengths only reduced the means to 0.1471 and 0.1640. Missing incoming channels
near propagation cutoffs are a major problem: zero padding does not reproduce
the reflecting grazing limit. This evidence prevents using those grids as the
final material response.

An explicit optional numerical extension pads missing channels with unit
specular reflection. In the common port space it preserves contraction and
reciprocity. Separate direct-solve tests approach this limit for the tested
metal profile at three wavelengths and three azimuths. This extension is not a
general boundary solution: a perfectly matched transmitting interface requires
separate handling, and actual interpolation error still needs validation.
At 8x16x9 it reduces mean L1 error to 0.0994/0.1143 and maximum error to
0.5493/0.5797, which is still unacceptable. A 16x32x33 grid reduces the mean errors to 0.0482/0.0586, but maxima
remain 0.4245/0.4804. These grids also FAIL the accuracy requirement. Building
each took about 30 seconds and stored 0.65/1.83 MiB; these are single-run
preparation measurements, not controlled render benchmarks. Further work needs
cutoff-aware sampling and modal convergence checks.

The N=16 versus N=32 comparison itself has mean L1 errors 0.00452/0.01006 and
maxima 0.02594/0.06840. Modal convergence therefore also remains a requirement;
merely increasing cache resolution cannot eliminate this discrepancy.

Accuracy records: `cache_accuracy_8.json`, `cache_accuracy_16.json`,
`cache_accuracy_8_grazing.json`, and `cache_accuracy_16_33_grazing.json` in
`tests/output/diffraction/`.
Host precomputation timing at N=64 was 74–82 ms per complete block in the
five-repeat benchmark, approximately 2.6–5.9 times faster than solving the tested
upper incident directions separately across all measured modal counts. These
are host solve times, not GPU rendering overhead.

## Follow-up research after the cutoff accuracy failure

[Dyakov et al., JETP Letters (2025)](https://link.springer.com/article/10.1134/S0021364025609340)
interpolate nonresonant substructures and recombine their scattering matrices.
Their method still assumes separation from Rayleigh anomalies; the reported
error increases near those thresholds. This does not provide a drop-in solution
to the cache failure above.

[Gromyko et al.](https://arxiv.org/html/2112.05014v2) formulate resonant
approximations in normal wavevector components and discuss multiple nearby
diffraction thresholds. This supports investigating coordinates/operators that
retain the square-root cutoff dependence explicitly. It does not establish
passivity or sufficient accuracy for a new Cycles interpolation scheme.

A candidate to investigate next is a reduced surface admittance, with the
external plane-wave admittance evaluated analytically at the query. Using the
existing modal matrices W, V, propagation X, and lower-medium admittance L,
interface elimination suggests:

* R = (V + L W)^(-1) (V - L W)
* Y = V (I - X R X) [W (I + X R X)]^(-1)
* (Y + U) E_total = 2 U E_incident, where U is the external upper admittance.

These equations are a proposed derivation from the implemented boundary system,
not a validated new solver. Eliminate only high evanescent channels through a
Schur complement of Y + U; retain every channel that can approach a cutoff in
the table domain. A rotation of tangential H to the normal-flux convention is
needed before checking positive Hermitian parts. Potential benefits are smoother
cached operators and exact external cutoff terms. Outstanding risks include
admittance poles, numerical conditioning, truncation error, preservation of
reciprocity/passivity under reduction and interpolation, and GPU matrix-solve
cost. No performance or correctness claim is made for this candidate.

## Fixed-reference-port formulation

A first implementation now avoids caching external cutoff singularities by
replacing the retained external channels with constant-admittance reference
ports. It reuses the vector Maxwell boundary solve. Omitted harmonics remain
terminated in the actual external medium and must all be evanescent. A lossless
substrate has reference ports on both sides; an absorbing substrate is terminated
inside the operator. The retained Ex/Ey reference channels have a unit flux
metric, including channels that would be evanescent in the physical exterior.
They are mathematical ports, not new physical light paths.

The resulting complex S0 operator is contractive, and unitary for the tested
lossless cases. It can be interpolated without changing channel count at a
physical diffraction cutoff. Matching to the actual exterior uses bounded
analytic coefficients. In each order's tangential TE/TM basis, with normal
wavevector q and real external index n:

* R_TE = (1-q)/(1+q), R_TM = (q-n²)/(q+n²).
* For propagating physical ports,
  T_TE = 2 sqrt(q)/(1+q), T_TM = 2 n sqrt(q)/(q+n²).
* S_physical = -R + T S0 (I-R S0)^(-1) T, projected onto propagating ports.

The implementation evaluates R also for evanescent and exactly grazing retained
channels, while T connects only propagating physical channels. This avoids
forming the divergent TM admittance n²/q at q=0. It also avoids explicit surface
admittance poles at this stage. Degenerate modes inside the patterned layer
remain an unresolved limiting case.

Twenty-one host tests pass. New checks show complex agreement with the original
full Maxwell solve for flat and relief dielectric/metal profiles at three
wavelengths, reference-port reciprocity and passivity, exact external grazing,
and interpolation through a cutoff. The exact-cutoff result is also compared
with nearby direct-solve limits. These tests do not establish a general
interpolation accuracy bound or GPU performance.

The first 8x16x9 reference-operator cache audit improves mean reflected-power L1
error to 0.0271/0.0325 for 740/1600 nm pitch, with maxima 0.1500/0.1481. This is
substantially better than the intensity cache but still insufficient. The
reported matrix float-equivalent sizes are 0.88/2.85 MiB; the audit currently
stores complex doubles on the host and has no GPU operator cache implementation.
Its Bloch grid includes both endpoints with a fixed retained order window,
avoiding interpolation of incompatible artificial-port sets at a wrapped seam.
Record: `tests/output/diffraction/cache_accuracy_reference_8.json`.

Important remaining work for this formulation: broader accuracy/convergence
measurements, reference-data validation, exact internally degenerate limits,
lossless conservation under interpolation (a convex average of unitary matrices
is generally contractive, not unitary), an efficient GPU matching solve, and
integration with actual materials and transport. No final rendering performance
claim applies to this host-only formulation.

The denser 16x32x33 reference cache reduces mean L1 power error to
0.00459/0.00570 and maximum error to 0.03635/0.04363. Those interpolation errors
are now comparable to the separately measured N=16/N=32 discrepancy in some
regions, but neither is an adequate final accuracy guarantee. Float-equivalent
matrix storage is 12.89/41.77 MiB; current host complex-double storage is twice
that. The older audit field `packed_bytes` describes this float-equivalent
estimate for reference caches, not an uploaded GPU buffer. The audit now labels
host storage and float estimates separately. Record:
`tests/output/diffraction/cache_accuracy_reference_16_33.json`.

## Lossless-preserving interpolation charts

`diffraction_grating_reference_to_chart` and its inverse implement a rotated
Cayley transform Y = (I-c S0)(I+c S0)^(-1), where c is a unit complex phase.
The Hermitian part of Y is nonnegative for passive S0 and zero for lossless S0.
Interpolating Y with nonnegative weights in the same chart therefore preserves
passivity and lossless conservation. The inverse is
S0 = conjugate(c) (I-Y)(I+Y)^(-1). This avoids the artificial absorption caused
by averaging different unitary scattering matrices directly.

Focused tests check the round trip and lossless conservation through an exact
external cutoff. A transparent reference operator demonstrates a chart pole:
phase zero is rejected, while a quarter-turn chart succeeds. A production cache
still needs automatic, well-conditioned chart selection, compatible chart
choices for reciprocal query pairs, and a broad approximation audit. There is
no global fixed-phase guarantee for every material and parameter range.

For later GPU optimization, combining the chart inverse with the external
matching equations suggests one solve rather than two:

* M = (I+Y) - conjugate(c) R (I-Y).
* M X = T for the selected incident physical channels.
* S_physical = -R + T conjugate(c) (I-Y) X.

The combined GPU form was subsequently implemented and validated as described below. For a
propagating output port, an alternative evaluation of the reference outgoing
field is b = (R+c I)^(-1) (2X-T); phases away from 0 and pi avoid a small
denominator there. This is a proposed optimization that requires numerical
verification, particularly near grazing incidence and resonances.

The full install build with the reference-port and chart APIs passed:
`tests/output/diffraction/build_reference_charts.log`. At that build, the new operators were host-only. The earlier Metal intensity-grid
tests do not validate reference-port matching; the separate tests below now do.

The host suite now contains 25 passing tests, including chart reciprocity and
passivity for both dielectric and absorbing-metal operators. Current record:
`tests/output/diffraction/host_solver_tests.xml`.

## Float conditioning and Metal reference matching

The shared `kernel/util/diffraction_reference.h` now implements the combined
Cayley/exterior solve with complex float arithmetic and partial pivoting. It
returns both Jones input columns without clipping or probability renormalization.
The fixture constructs exterior coefficients in double precision on the host;
this isolates matrix quantization and GPU matching, not GPU coefficient
construction or interpolation error.

A fixed quarter-turn Cayley chart failed the original float tolerance on one
lossless dielectric case: chart Frobenius norm 9615.98, matrix reciprocal
condition estimate 1.55e-5, and maximum float amplitude error 2.10e-4.
The failure is retained in `tests/output/diffraction/host_solver_fixed_phase_failed.log`
and the matching diagnostic files. No test tolerance was relaxed.
`diffraction_grating_choose_chart` now evaluates fixed rotations and gaps between
reference eigenvalue pole angles, selecting the smallest worst chart norm for a
group of operators. This is a conditioning heuristic; it does not guarantee
accuracy at every physical resonance. Reciprocal cache cells must share chart
selection when interpolation is introduced.

The actual M5 fast-math test passed all 192 fixtures (64 each for DVD-sized metal,
CD-sized metal, and dielectric reflection/transmission), including near-grazing
and exact external-cutoff cases. Maximum complex amplitude errors were
4.25e-7, 3.74e-7, and 4.63e-7 respectively. The test also checks power for arbitrary
incident polarization through the maximum eigenvalue of the two-column Gram
matrix. Record: `tests/output/diffraction/metal_reference_matching_warm.json`.

For 262144 hot-data evaluations, three measured GPU times after two full-batch
warmups were 5.37–6.51 ms (10 channels), 53.33–54.01 ms (18 channels), and
74.86–75.50 ms (20 channels). These times exclude cache lookup/interpolation,
exterior-coefficient construction, ray traversal, and renderer integration.
They demonstrate that a full per-query LU solve is costly, not a final rendering
overhead measurement. The harness now additionally compares benchmark output
power checksums with each independent double-precision fixture, so the timed
entry point is checked separately from the full-Jones test entry point.

## Reduced reference feedback

`diffraction_grating_prepare_hybrid` terminates distant evanescent orders and
converts noncritical propagating orders to physical Cartesian flux ports. A
caller-supplied subset remains at unit-admittance reference ports. Preparation
is a passive network composition, preserving lossless unitarity; it does not
complete missing energy with a specular term. `diffraction_grating_match_hybrid`
then solves only the reference-channel feedback system. For active channel set A,
B=S T and F=S[:,A] R[A,A], the solve is
`(I-F[A,:]) Z = B[A,:]`, followed by `S_out=-R+T^t (B+F Z)`.
Identity-coupled physical ports do not enter the LU system.

The shared CPU/Metal template uses private storage proportional to the square
of the feedback-channel count, rather than the full operator size. Zero feedback
requires no LU solve. Full-Jones output still includes every represented physical
order. This code is not yet connected to the shader cache or material closure.

The 31-test host suite passes. New tests compare full and reduced matching within
1e-10 for metal and dielectric operators, including exact cutoffs and several
reference subsets, verify passivity/lossless conservation, check empty physical
channel sets, and reject unrepresented propagating orders or physical ports
crossing a cutoff. Float tests cover zero, two, and four feedback channels against
double Maxwell coefficients. Record: `tests/output/diffraction/host_solver_tests.xml`.

Crucial remaining work: select reference subsets from conservative bounds over
an entire cache cell, refine difficult cells, and audit interpolation. Agreement
at a prepared query does not prove agreement elsewhere in a cell. For ordinary
intensity transport, matching each corner to the query's critical channels and
mixing resulting powers is a candidate that preserves lossless energy; it has
not yet been implemented or validated. Coherent transport requires complex
operators and a separate interference estimator. Neither cross-object coherent
rendering nor a user-facing coherence option is implemented yet.

The reduced Metal tests now pass all 576 additional fixtures (three profiles,
64 queries, and feedback counts 0/2/4). Maximum complex amplitude error across
these tests is 2.67e-7. The benchmark checksum checks also pass. For the CD metal
case with four feedback channels, the three GPU times are 1.15–1.50 ms per
262144 evaluations, versus 53.73–55.26 ms for the full 18-channel solve in the
same run. Several shorter kernels have substantial timing outliers; do not
interpret this as a precise or scene-level speedup. Fixtures pad removed ports
with zero rows/columns to retain fixed strides; a production cache can compact
them. Record: `tests/output/diffraction/metal_hybrid_matching.json`.
The full install build passed after these kernel and host changes:
`tests/output/diffraction/build_hybrid_matching.log`.

The host suite now has 32 passing tests. The additional interpolation test mixes
matched corner powers across an exact external cutoff and checks lossless column
sums and reciprocity in a symmetric lossless fixture. This verifies invariants,
not approximation accuracy.

`--hybrid-cache` in the host audit now bounds each order's squared longitudinal
wave number over a Bloch/ky/wavelength cell, retaining any order whose interval
crosses zero. It prepares each corner's hybrid operator and mixes its matched
query powers. This audit prepares hybrids on demand from the full reference
grid; reported grid bytes/build time describe that underlying grid, not a
production hybrid-cache allocation or preprocessing time.

The first approximation results are insufficient for production:

| Grid | DVD mean/max L1 | CD mean/max L1 | DVD/CD mean feedback channels |
| --- | --- | --- | --- |
| 8x16x9 | 0.02926 / 0.22066 | 0.04714 / 0.29036 | 1.91 / 3.01 |
| 16x32x33 | 0.01089 / 0.10717 | 0.01707 / 0.17542 | 1.08 / 1.51 |

The denser grid requires at most 4/6 feedback channels in these 256 sampled
DVD/CD queries. Low average channel count is promising for cost, but it does not
justify accepting the interpolation errors. Records:
`tests/output/diffraction/cache_accuracy_hybrid_8.json` and
`tests/output/diffraction/cache_accuracy_hybrid_16_33.json`.
An optional audit `--cutoff-margin` retains nearby orders as well, to examine
whether rapidly varying exterior coupling outside a crossing cell causes these
errors. It is an experiment parameter, not a validated production threshold.

With a squared-wave-number cutoff margin of 0.25 on the 16x32x33 grid, mean L1
errors improve to 0.00528/0.00912 and maxima to 0.08047/0.09133 (DVD/CD).
Mean feedback counts rise to 2.89/5.15, with maxima 6/14. Thus nearby exterior
orders do explain much of the error, but this uniform margin is neither a final
accuracy solution nor a guarantee of a four-channel solve. Further work should
refine cells using an explicit error criterion and measure the resulting
storage and feedback-count distribution. The GPU benchmark above covers only
0/2/4 feedback channels, not this entire distribution. Record:
`tests/output/diffraction/cache_accuracy_hybrid_margin_16_33.json`.

## Prepared cells and adaptive accuracy audit

The host API now has `diffraction_grating_prepare_cell` and
`diffraction_grating_cell_power`. A cell bounds Bloch momentum, tangential
momentum and wavelength. Preparation conservatively bounds external cutoff
locations, including a roundoff guard, and constructs eight hybrid corner
operators. Evaluation matches those corners to the query and mixes powers.
The cell API rejects queries outside its bounds. A new test compares exact
corner values with direct Maxwell solves and checks intermediate lossless
energy through a cutoff. The install build passed in
`tests/output/diffraction/build_diffraction_cells.log`.

`tests/performance/cycles_diffraction_adaptive.cpp` audits lazy refinement along
held-out query paths. Cell acceptance uses fixed validation locations independent
of the query. Failed cells subdivide; reaching the depth limit is recorded as
unresolved, not silently accepted. Storage and timing describe only visited
cells, not a complete production cache. No shader or GPU cache integration is
implied by these results.

The initial seven-location criterion at tolerance 0.002 missed a DVD held-out
query (error 0.00479) and left one CD query unresolved at depth six. Record:
`tests/output/diffraction/cache_accuracy_adaptive_64.json`.
The revised criterion uses 27 tensor-product interior locations and an acceptance
threshold of 0.001, retaining a held-out target of 0.002. A fresh seed (571291),
256 directions per profile, and maximum depth eight produced:

| Profile | Mean/max held-out maximum-column L1 | Unresolved | Mean/max feedback channels |
| --- | --- | --- | --- |
| DVD | 0.000339 / 0.000947 | 0 | 2.36 / 6 |
| CD | 0.000348 / 0.000916 | 0 | 4.29 / 12 |

This metric checks the worst incoming column of the complete unpolarized power
matrix at each query, unlike earlier audits of only the sampled incoming column.
It is empirical evidence on the sampled directions, not a global error bound.
The N16/N32 maximum-column discrepancies remain much larger: mean 0.00581/0.01349
and maximum 0.02404/0.03853 for DVD/CD. Modal convergence, a complete cache build,
GPU packing/lookup, and production integration are still required. Record:
`tests/output/diffraction/cache_accuracy_adaptive_256.json`.

A higher-mode adaptive audit (32 held-out directions, N32 cells versus N64 direct
solves) again meets the sampled interpolation target: maxima 0.000640/0.000885
for DVD/CD. Modal discrepancies remain larger, with maximum-column L1 maxima
0.00860/0.01113. The separate N64/N128 direct audit reduces those maxima to
0.00309/0.00423; maximum complex-amplitude differences are 0.00274/0.00559.
These finite-truncation comparisons are convergence evidence, not exact answers.
Records: `cache_accuracy_adaptive_N32_32.json` and `modal_convergence_64_128.json`
in `tests/output/diffraction/`.

The host suite has 34 passing tests, including the prepared-cell checks and
complex mirror symmetry for both physical and artificial ports. The binary
lamellar ridge is centered at x=0, which fixes its order-phase origin. Reflection
in x reverses diffraction-order indices and the x polarization component;
reflection in y reverses the y polarization component. Both transforms were
checked on complex amplitudes, not only intensities. This supports future cache
domain reduction without discarding the phase information needed by coherence.

## Representative Blender scene suite

At the user's request, `tests/python/cycles_diffraction_suite.py` now generates
nine dedicated visual/diagnostic cases rather than relying on one disc scene:
pitch, roughness, depth/duty, tangent orientation, weight, unit-white furnace,
CD/DVD studio, embedded cover, and an indirect-lighting room. Editable scenes,
linear EXR/NumPy results, PNG previews and provenance JSON are in
`tests/output/diffraction/visual_suite/`. These use the provisional shader.
The full coverage plan and explicit missing-feature scenes are documented in
`tests/scenes/diffraction/README.md`.

Visual review exposed unwanted swatch-to-swatch reflections in the first charts.
The revised parameter charts isolate their swatches, use identical distant
strip illumination, and disable their influence on other ray types. The studio
and indirect room retain ordinary interreflection. The studio discs have an
ungrooved hub/rim and a 24–58 mm radial grating region. The cover encloses its
reflector and uses the matching internal index. All nine revised PT scenes were
rendered on M5 Metal at 2048 samples without denoising or radiance clamps.

`tests/python/cycles_diffraction_measure.py` records raw furnace and zero-depth
control statistics. In the interior furnace region, the flat disabled control
is exactly RGB=(1,1,1); the flat diffraction-on mean is approximately
(1.000055,1.000050,0.999590). These are one-render observations. Spatial variation
is not an independent-seed Monte Carlo uncertainty estimate, and this is not a
global energy proof. The indirect room remains noisy at this sample count and
is retained specifically to test difficult transport and convergence.

## GPU exterior boundary construction

`kernel/util/diffraction_boundary.h` constructs exterior reference-port
coefficients directly from a float incident direction, refractive indices,
wavelength, pitch and relative diffraction order. A two-float expansion retains
the incident normal component and cancellation residuals near grazing angles
and order cutoffs. It does not provide general IEEE double precision.

The first Metal compilation rejected `fmaf` and ignored `float_control`.
The Metal implementation now uses `metal::fma` and function-scoped Clang FP
reassociation/contraction controls. This avoids changing math settings in the
rest of the renderer. The rejected compile report is retained as
`tests/output/diffraction/metal_boundary_compile_failure.json`.

`tests/metal/cycles_diffraction_test.py --boundary` runs the production header
under Metal fast math against a double reference computed from the same
quantized inputs. All 1035 random and targeted cases passed on M5, including
adjacent float wavelengths around a cutoff and grazing incident directions.
Maximum coefficient error was 2.53e-7; maximum R-squared plus T-squared error
was 3.59e-7. The test checks all outputs of each of five 262144-query dispatches.
The three post-warmup timings were 2.658, 0.249 and 2.751 ms. This variation
precludes a stable throughput claim; these are hot-input microbenchmarks with
output writes, excluding matching, cache lookup and rendering.
The source hashes and measurements are in `metal_boundary.json`.

Boundary construction is not yet connected to physical-cache shading. These
tests therefore do not validate a complete Maxwell renderer or coherent
transport. The host suite now contains 35 tests including boundary coefficients.

### Combined GPU construction and matching

The reference and hybrid matchers now deduce the boundary pointer address space,
allowing coefficients constructed in Metal thread memory to feed the solve
directly. `--reference --construct-boundaries` exercises this path against double
RCWA solves evaluated at the same quantized direction/wavelength inputs. The
test covers metal CD/DVD profiles and a lossless dielectric with incidence from
both sides, full reference matching and 0/2/4 feedback-channel hybrid matching.
It includes grazing and exact-cutoff cases. It does not interpolate cache cells.

All 768 cases passed on M5: maximum complex coefficient error 5.79e-7 and maximum
column-power error 8.95e-7. Benchmark dispatch checksums also passed. With four
feedback channels the CD fixture took 1.70–2.57 ms per 262144 evaluations,
including boundary construction, versus 56.08–56.73 ms for the full 18-channel
reference solve. These hot-data measurements omit cache lookup, traversal and
renderer costs and contain timing variation. They are not rendering overhead
measurements. Results: `metal_constructed_matching.json`.

The existing device-buffer coefficient path also passed all 768 cases after the
pointer change (`metal_matching_pointer_regression.json`), and all 35 host tests
passed. This verifies the shared matching change without asserting that the
physical cache is connected to the renderer.

### Prepared cell upload and intensity evaluation

`diffraction_grating_pack_cell` validates common corner topology, ordered port
identities, feedback counts and finite float representability before packing
eight complex matrices for GPU upload. Failure leaves no partially usable output.
The port list and bounds remain explicit cache-index metadata.

`diffraction_hybrid_cell_power` matches each corner to the same query boundary
and then blends unpolarized powers with nonnegative trilinear weights. It rejects
invalid coordinates and insufficient scratch capacity. Complex matrices are
not interpolated by this ordinary-intensity routine; it is not a coherence
estimator. The kernel helper still needs a dedicated Metal cell test and cache
lookup/shader integration.

The 36-test host suite passes. The added test compares packed float evaluation
against double cell evaluation for every incoming physical port in metal and
lossless dielectric fixtures, at endpoints, interiors and either side of an
order cutoff. It also checks malformed topology, feedback-count mismatches and
nonfinite matrix rejection. This verifies packing and evaluation separately
from the unresolved global cache accuracy and build-cost requirements.

### Metal prepared-cell validation

The Blender target build passed after adding the packed-cell evaluator.
`tests/metal/cycles_diffraction_test.py --cell` now tests uploaded eight-corner
matrices with GPU boundary construction and intensity blending. It exercises
87 queries across metal DVD/CD and lossless dielectric cells, including all
incoming physical ports at the selected endpoints, interior points and cutoffs.
These fixtures use four feedback channels; larger feedback sets remain to be
benchmarked for cell evaluation.

The first comparison exposed a reference-input mismatch at a CD cutoff:
rounding a ray to float can open an order that is closed for the original
double ray. The failed report is retained in
`metal_cell_unquantized_reference_failure.json` (maximum discrepancy 1.73e-4).
The corrected reference matches each double corner at the exact quantized ray
provided to Metal and blends with the same supplied weights. It does not use
the GPU result to construct its expected output or relax the 3e-5 tolerance.
All cases then passed, with maximum power error 2.38e-7 and maximum dielectric
energy error 3.73e-7. Every output in five 262144-query dispatches was checked.

Post-warmup times were DVD 4.72–5.33 ms, CD 9.33–9.95 ms and dielectric
6.18–6.20 ms per 262144 evaluations. These include boundary construction and
eight-corner matching, with endpoint queries skipping zero-weight corners.
They use hot cells and exclude cache search, global memory behavior of a full
cache, interpolation error against direct RCWA, and rendering. Results are in
`tests/output/diffraction/metal_cell.json`. Cache-index construction and physical
shader integration remain unfinished.

### Complete cache construction and binary lookup

The host now builds a complete binary partition using cutoff-aware cells and 27
interior validation points per candidate. Accepted matrices are packed for GPU
upload. The baseline split policy bisects the largest normalized axis extent.
Node, depth and matrix-memory limits produce an explicit failure with no partial
cache returned. Statistics retain the visited/accepted counts and failing bounds.
The shared kernel lookup assigns split planes to the right child and rejects
out-of-domain and malformed-index queries instead of extrapolating.

All 38 host tests pass, including a complete small-domain build, lookup endpoints,
split ownership, invalid indices and resource exhaustion. Validation uses a
direct reference-port solve and physical matching, since the older direct
physical-port solver rejects exact exterior grazing. This changes the formulation
at a known singular boundary, not the interpolation tolerance. The criterion is
still empirical and at a fixed modal truncation, not a uniform physical-error
guarantee. The full cache index has not yet been exercised on Metal.

The full-domain audit covers Bloch [-0.5,0.5], ky [-1,1], wavelengths 380–780 nm,
N16, tolerance 0.001, cutoff margin 0.1, 256-node budget and depth limit 24.
Neither material completed. DVD stopped after 228 visited nodes/104 accepted
cells at error 0.00100342; CD stopped after 26 nodes/one accepted cell at
0.00112489. Both failures were near grazing incidence, and both returned zero
cache cells. Times were 15.35/1.72 seconds. These are incomplete-build diagnostics,
not full-cache memory or timing estimates. Exact failing bounds and provenance:
`tests/output/diffraction/full_cache_reference_256.json`.

This baseline is not ready for shader integration. Refinement efficiency,
full-domain completion, independent held-out accuracy, modal convergence and
practical cache memory/build costs still require work.

### Response-based refinement

The split policy now compares squared complex-operator differences along the
four parallel edges for each axis and bisects the axis with greatest variation.
The hybrid topology/basis is common to the corners. This heuristic reuses the
already computed matrices; it does not replace or relax the 27-point direct
validation. Normalized geometric extent breaks ties, and `--geometric-splits`
retains the previous policy for comparison.

With the same 256-node budget, depth 24 and tolerance 0.001, both profiles now
reach the node budget instead of the prior grazing depth failure. Each accepted
121 leaves before stopping; neither returned a partial cache. Times were
16.99/17.10 seconds for DVD/CD. This is progress in allocation, not evidence that
the complete cache fits practical limits. Results:
`tests/output/diffraction/full_cache_adaptive_256.json`. All 38 host tests pass.
A larger 4096-node/depth-36 audit has been started to measure the next limit.

The Metal harness now also checks the shared binary lookup with an uploaded
seven-node index. All 1149 random, split-plane, domain-edge and nonfinite queries
returned the expected leaf under Metal fast math. The 87 cell-evaluation queries
passed again with unchanged errors. This validates lookup separately from cell
evaluation, not a complete renderer cache traversal. Record:
`tests/output/diffraction/metal_cell_and_lookup.json`. Its microbenchmark timings
were collected while the larger host cache audit was active and should not be
used as an isolated performance comparison.

### Reusing identical host reference solves

Cache construction now keeps a bounded per-build LRU of reference operators,
keyed by exact wavelength/kx/ky values. The profile and Fourier truncation are
fixed for that build. Neighbouring cells can reuse their shared corners and
validation samples without rounding keys or approximating another response.
The default matrix-data budget is 64 MiB; port metadata and container overhead
are additional memory. `--reference-cache-mib 0` disables reuse for comparison.

All 39 host tests pass. The new test forces subdivision and compares reuse-on
and reuse-off construction: node indices, bounds, active ports and every packed
float matrix component match exactly. It also checks the matrix-data bound and
that actual solve count falls. A 64 KiB trial was too small to retain the parent
samples until child evaluation and yielded no hits; the test uses 512 KiB to
exercise useful reuse and bounded storage. This optimization changes build work,
not the interpolation criterion or rendering model.

In the 256-node full-domain audits, reuse reduced each profile from 8960
requested solves to 4816 actual solves plus 4144 hits (46.25% avoided). Accepted
counts, validation errors and final bounds exactly match the earlier audit.
Reference matrix storage peaked at 7,705,600 bytes for DVD and 24,966,144 bytes
for CD. Elapsed times were 9.38/9.25 seconds, with another host audit running;
solve-count reduction is the reliable comparison here. Neither limited build
completed. Record: `tests/output/diffraction/full_cache_reuse_256.json`.

Construction now exposes a cancellable progress callback and reports the sum
of accepted leaf volumes relative to the requested parameter domain. That
fraction includes regions without physical ports; it is not a fraction of
camera rays or light transport. Cancellation returns no partial cache. The
39-test suite passes after checks for cancellation before solving and complete
domain coverage in both single-leaf and subdivided builds.

A 4096-node audit with reuse and coverage reporting is running alongside the
original no-reuse audit. Compare solve counts and subdivision outcomes; their
overlapping wall times are not isolated performance measurements.

The original 4096-node audit finished at the node budget for both profiles:
DVD accepted 2040 leaves (1,000,704 matrix bytes) in 276.07 s; CD accepted 2042
(5,011,200 bytes) in 270.67 s. Both returned zero cache cells because coverage
was incomplete. These partial-storage totals are not full-cache memory
estimates. The errors of accepted cells stayed below 0.001. Record:
`tests/output/diffraction/full_cache_adaptive_4096.json`. Early coverage output
from the reuse-enabled build confirms that large leaf counts can cover a small
parameter volume; the representation/refinement cost remains unresolved.

### Hybrid chart-cell alternative

The reuse-enabled 4096-node audit completed at the node budget, with the same
subdivision outcomes as before. DVD covered 0.22536% and CD 0.36888% of parameter
volume. It performed 75638/75906 actual solves with 67722/67454 hits and bounded
reference-matrix storage below 64 MiB. Times were 146.29/144.34 s, overlapping
other work. This confirms that exact reuse helps build work but does not resolve
the intensity-cell coverage problem. Record: `full_cache_reuse_4096.json`.

`cycles_diffraction_cell_chart.cpp` compares common Cayley interpolation of the
hybrid corner operators against intensity blending, with 27 lattice and 256
random probes per cell. On the earlier DVD grazing cell, maximum column-L1
error drops from 0.00103023 to 0.0000119892; on the CD grazing cell, from
0.00112489 to 0.000131136. The lossless cutoff cell improves from 0.00604024 to
0.00163557, and the CD cutoff cell from 0.0435685 to 0.0203229. The latter two
still fail the 0.001 target. The chart is not a universal accuracy fix.
Record: `tests/output/diffraction/hybrid_cell_chart_audit.json`.

The host now exposes `DiffractionGratingChartCell`, preparation of a common
chart and physical complex matching at a query. All 40 tests pass, including
complex corner agreement, all-polarization passivity/lossless energy and
malformed chart rejection. This representation requires a dense query solve;
its potential reduction in cell count must be weighed against GPU cost. It is
not yet selected by the cache builder, packed for GPU cell evaluation, or
connected to shaders. Reciprocal cells must share chart choices (or a common
symmetry-mapped cell); phase-preserving interpolation alone is not a global
coherent-transport implementation.

### Validated chart selection in cache construction

The builder now tests the intensity representation first and, if it misses the
tolerance, prepares a common chart and validates that representation at the same
27 points. It subdivides when neither passes. `--intensity-only` retains the old
construction mode. Packed cells carry an explicit representation flag and chart
rotation; chart payloads must not be interpreted as scattering matrices.

All 41 host tests pass, including the previously failing DVD grazing cell:
with depth fixed to zero, intensity-only construction fails while chart selection
passes without changing tolerance. Packing preserves the selected chart data
and rejects inconsistent rotations. GPU chart-cell evaluation is still pending.

The 256-node audit with chart selection accepted 45 chart cells for DVD and 90
for CD. Domain coverage reached 0.19712%/0.15259%, versus approximately
0.0443%/0.0381% in the intensity-only audit. Neither build completed; early
depth-first coverage is not an estimate of total cache size. Elapsed times were
9.44/9.21 seconds. Record: `tests/output/diffraction/full_cache_chart_256.json`.
The larger full-domain feasibility, GPU cost and reciprocal chart selection
requirements remain open.

### Metal chart-cell interpolation and matching

The kernel now interpolates eight chart matrices in thread memory and performs
one dense physical matching solve. The shared reference matcher accepts either
device or thread chart storage. `--cell --chart-cell` exercises 6/12/18-channel
packed charts with GPU-constructed boundaries. All 87 queries passed against
double matching at identical quantized inputs; maximum power error was 4.77e-7
and dielectric energy error 5.70e-7. The 1149 lookup queries also passed. This
cell harness checks powers, not complex phase accuracy of interpolated GPU
outputs. It does not validate global coherence or rendering integration.

For 262144 hot-cell evaluations, the chart timings were 1.32–2.38 ms (6 channels),
18.67–19.44 ms (12) and 67.45–73.86 ms (18). The intensity path regression passed
with times 4.47–5.24, 6.19–7.07 and 8.93–9.71 ms respectively. The regression
overlapped host-test compilation, so these are microbenchmark observations, not
an isolated rendering-overhead comparison. Large dense chart solves are costly;
retaining the reduced intensity path where it meets accuracy remains useful.
Records: `metal_chart_cell.json` and `metal_intensity_cell_regression.json`.
All 41 host tests passed after the shared matcher change.

### GPU chart phase comparison

The Metal cell harness now records the complex Jones coefficients for each
unique chart query, in addition to checking powers over the full dispatch.
All 2520 complex coefficients per run passed in five dispatches, with maximum
absolute complex error 5.46e-7 against double matching at the same quantized
ray and corner weights. This checks local interpolation/matching phase, not
cross-object propagation phase or a coherent transport estimator. Record:
`tests/output/diffraction/metal_chart_cell_phase.json`.

The Blender target rebuilt successfully with the current cache/chart code
(`build_chart_cache.log`). A 4096-node chart-enabled domain audit is running;
neither the build nor these local tests establish full renderer integration.

### Shared mirror domain and reciprocal reconstruction

Cache construction can now store only the negative Bloch/conical quadrant for
the centred scalar lamellar profile. It requires origin-symmetric requested
bounds and records the symmetry mode. Lookup folds the query; an x reflection
reverses port orders, while a single-axis reflection changes the sign of both
cross-polarized Jones entries. Intensity values are unaffected by that sign.
This gives opposite directions the same interpolated operator/chart rather
than independently choosing approximations. Shader consumption of this mapping
is still pending; the mode is explicitly enabled with `--mirror-symmetry`.

All 42 host tests pass. The new test covers metal and lossless dielectric
reflection/transmission, compares complex reconstructed corner responses with
independent full-direction solves, checks reciprocal symmetry at interpolated
queries, and rejects incompatible bounds. The new folded lookup path still
requires Metal validation; earlier GPU lookup tests cover the unfurled path.

In the 256-node symmetry audit, represented domain fractions were 0.78869%
(DVD) and 0.61035% (CD), approximately four times the corresponding unfurled
coverage. Neither build completed. Record: `full_cache_mirror_256.json`.
The previous 4096-node chart audit also ended at its node budget, with coverage
0.23394%/1.10903%, 81/696 chart leaves and approximately 146 seconds per profile
(`full_cache_chart_4096.json`). These partial-domain results do not prove full
cache feasibility or rendering performance.

### Parallel validation during cache construction

Independent validation points can now run on a bounded number of workers. A
mutex protects exact-solve reuse and its statistics; matrix work runs outside
the lock. Per-sample errors/results are private, and reduction occurs in fixed
sample order. The default remains one worker; `--workers 6` enables the tested
parallel configuration. This parallelism is in host precomputation only.

All 43 host tests pass. Serial and four-worker construction produce identical
nodes, metadata, packed matrix components, accepted errors and coverage in both
intensity and chart fixtures. In the six-worker 256-node mirror audit, times
were 2.50/2.51 seconds for DVD/CD versus 9.27/9.19 seconds in the earlier serial
run. Acceptance statistics and solve/hit counts are identical. These are small
build audits, not full-cache or rendering performance claims. Record:
`tests/output/diffraction/full_cache_parallel_256.json`.

The ThreadSanitizer build completed and five repetitions of
`ParallelCacheValidationMatchesSerialExactly` passed without sanitizer reports
(`tsan_compile.log`, `tsan_parallel_cache.log`). This exercises the tested
parallel builder fixtures; it is not a whole-renderer race audit.

### Metal folded lookup validation

The GPU cell harness now checks both ordinary and mirror-folded lookup, with
1149 queries each, including boundaries and nonfinite inputs. It also checks
the returned order reversal and cross-polarization sign against independent
host expectations. Both paths passed on the Apple M5. The accompanying chart
matching regression passed all 87 queries and 2520 complex coefficients per
dispatch; maximum complex error was 5.46e-7. Record:
`tests/output/diffraction/metal_mirror_lookup.json`.

This validates the lookup mapping and local matching separately, not a complete
uploaded cache or integrated material. A 65535-node, six-worker mirror-domain
audit is still running; no complete cache has yet been established.

The Blender target rebuilt successfully with parallel validation and mirror
lookup changes, including the three configured Metal library variants. Record:
`tests/output/diffraction/build_parallel_mirror_cache.log`. The installed app
and existing scene renders have not been updated by this build.

### Interior-curvature split experiment

An opt-in `--curvature-splits` audit mode uses second differences of the
artificial-port scattering operator along the existing 27-point validation
lattice. This avoids additional modal solves and keeps the physical-power
acceptance criterion unchanged. The fixed reference topology avoids confusing
exterior port appearance/disappearance with matrix indexing changes. All 43
host tests pass, including exact serial/parallel results in this mode.

The matched 256-node mirror audit (tolerance 0.001, cutoff margin 0.1, six
workers) accepted 0.58517%/0.50735% of the DVD/CD domains in 3.58/3.67 seconds.
Both builds exhausted their node budget. These fractions are below the prior
corner-variation results of 0.78869%/0.61035%; this experiment does not establish
an improvement and is not the default. Timings overlapped the ongoing large
cache build and are not controlled speed comparisons. Record:
`full_cache_curvature_matched_256.json`. The earlier
`full_cache_curvature_256.json` used different default tolerance/margin and is
not a matched comparison. Full-domain feasibility remains unresolved.

### Additional RCWA ray-tracing implementation comparison

The Ansys surface-relief RCWA documentation describes lazy direction-cosine
grids, interpolating electric fields as well as efficiencies, with separate
grids for changed wavelength/profile parameters. It also documents interpolation
failure near grazing incidence and singular cases at exact diffraction cutoffs.
These limitations must not be imported as silent ray rejection in Cycles.
The source therefore supports the general precomputation architecture, but does
not establish that our full spectral, textured-parameter domain is affordable
or correct. Source (consulted 2026-09-25):
https://optics.ansys.com/hc/en-us/articles/42661666095891-Simulating-diffraction-efficiency-of-surface-relief-grating-using-the-RCWA-method

### Large-domain audit: first profile exhausted its node budget

The ongoing 65535-node mirror-domain audit reported the DVD profile terminal:
`complete=0`, `visited=65535`, `Grating cache exceeded node budget`. Its final
periodic progress sample, at 65280 nodes, covered 0.116333 of the requested
domain. This is a volume fraction, not an incident-ray probability or a bound
on omitted energy. The combined audit is continuing with CD, so its final JSON
and exact terminal statistics are not yet available. No usable partial cache
was published. The current strategy has not demonstrated practical complete
coverage; simply integrating this incomplete data would not satisfy the task.

### All-reference versus reduced-channel representation audit

With cutoff margin 100 (all retained channels artificial), the 256-node mirror
audit accepted domain fractions 0.0184326/0.0177002 for DVD/CD, versus
0.0078869/0.0061035 with margin 0.1. Both builds still exhausted their budgets.
Packed matrices required 793600/2509056 bytes rather than 67840/244224 bytes.
This is increased partial coverage at substantially increased memory, not a
complete-cache solution. Record: `full_cache_all_reference_256.json`.

The cell-chart audit now accepts `--cutoff-margin` explicitly, allowing the
same four cells and 283 probes per cell to be compared without changing
geometry or random samples. Maximum physical-power column errors for chart
interpolation, reduced versus all-reference, were:

| Cell | Margin 0.1 | Margin 100 |
| --- | ---: | ---: |
| DVD grazing | 0.0000119892 | 0.000114862 |
| CD grazing | 0.000131136 | 0.0000953344 |
| Dielectric cutoff | 0.00163557 | 0.00237945 |
| CD cutoff | 0.0203229 | 0.0129407 |

Keeping every channel artificial is therefore not uniformly more accurate;
neither representation meets 0.001 in both cutoff cells. Each independently
chooses its chart, so this comparison measures the complete representation
choice, not solely normalization. Full-reference matrices here have 22/44
channels and have not been benchmarked in the GPU cell harness. Records:
`cell_chart_reduced_matched.json`, `cell_chart_all_reference.json`.

### Complex-response interpolation error against direct Maxwell solves

The cell audit now compares interpolated Jones coefficients to direct solves
at identical queries, with the same modal count and phase origin. No global
phase alignment is applied. It also reports the Frobenius error, a conservative
bound on output-field error for unit-norm incident channel superpositions.
Across the same 283 probes per cell at margin 0.1:

| Cell | Maximum coefficient error | Maximum Frobenius error |
| --- | ---: | ---: |
| DVD grazing | 0.0000602569 | 0.0000760032 |
| CD grazing | 0.000203721 | 0.000431222 |
| Dielectric cutoff | 0.0158315 | 0.0259467 |
| CD cutoff | 0.0383568 | 0.0653290 |

These are interpolation errors relative to the same finite modal solve, not
modal convergence bounds. In particular, the dielectric cell's much smaller
power error does not establish complex-response accuracy. The current cache
builder's unpolarized-power criterion is insufficient for coherent use; that
mode requires complex-response acceptance and must not consume intensity-only
cells. Record: `tests/output/diffraction/cell_chart_complex_error.json`.

### Complex-response cache acceptance

The builder now accepts a separate `complex_tolerance` (zero disables it).
When enabled it requires chart cells and checks the Frobenius norm of the
physical scattering-matrix error at each validation sample, without phase
alignment. Both power and complex criteria must pass. Completed output records
`complex_validated`; failures clear that flag along with all cache data.
This flag denotes the sampled criterion only, not a global error bound or a
coherent transport implementation. Statistics expose accepted and last complex
errors. The audit driver exposes `--complex-tolerance`.

All 44 host tests passed. The new regression demonstrates rejection under a
complex tolerance despite passing the power tolerance, then successful
four-worker refinement of the same region, with every output cell retaining
complex response. Incompatible intensity-only settings fail explicitly.

The 256-node mirror audit with both tolerances 0.001 accepted 121/122 chart
cells for DVD/CD; maximum accepted complex errors were 0.000995554/0.000995598.
Domain fractions were 0.00354385/0.00217819. Both exhausted the node budget and
returned no cache (`full_cache_complex_256.json`). This validates the exercised
acceptance behavior, not full-domain feasibility.

### Stored-cache lookup through float matching

All 45 host tests pass. The added end-to-end stored-data regression builds a
complete small mirrored cache around the 740 nm cutoff with complex tolerance
0.001, then evaluates 37 held-out queries. It uses the actual binary tree,
packed float chart matrices, interpolation coordinates, shared float matcher,
and order/sign unfolding. Maximum Frobenius complex error against direct
same-modal-count Maxwell solves was approximately 0.000526 (the predetermined
held-out threshold is 0.002). The test includes physical order appearance and
disappearance across wavelength/direction changes.

This runs the shared kernel math on CPU, with boundary coefficients computed
in double and converted to float. Metal boundary construction has separate
tests, but their combination with an uploaded full cache remains untested.
It does not establish full-domain coverage, modal convergence, or rendering
performance. Record: `host_solver_tests.xml`, property
`maximum_stored_cache_complex_error`.

### Combined stored-cache evaluation on Metal

The Metal harness now uploads the complete four-cell fixture and evaluates
lookup, interpolation coordinates, boundary construction, chart matching, order
unfolding and cross-polarization signs in a single dispatch. All 73 physical
incident-channel queries at 37 held-out directions/wavelengths passed on the
Apple M5 outside the sandbox. Maximum complex-column error against direct
same-modal-count solves was 0.000489743 (threshold 0.002). The host reference
uses the query coordinates; the GPU constructs and normalizes its float ray,
so this error includes that conversion as well as interpolation/packing.
The single-cell regressions and both 1149-query lookup tests also passed.
Record: `tests/output/diffraction/metal_stored_cache.json`.

This fixture has three retained upper-medium ports and six channels; it does
not validate arbitrary layouts, a complete material domain or renderer
integration. Its tiny dispatch is a correctness check, not a performance claim.

### Completed large-domain construction audit

`full_cache_mirror_parallel_65535.json` is now complete. Both profiles exhausted
65535 nodes and returned zero cells. DVD accepted 32762 leaves (4983 chart),
covered 11.6470% of domain volume and used 26730752 packed matrix bytes during
construction, taking 629.41 seconds. CD accepted 32762 leaves (5714 chart),
covered 14.5403% and used 169083648 bytes, taking 657.14 seconds. Other test work
overlapped this audit, so these are observed wall times rather than isolated
performance measurements. Complete-domain feasibility is contradicted at the
tested budget; the small GPU success does not resolve that construction cost.

### Chart rotation selected by interpolation error: diagnostic

The cell audit now compares the conditioning-selected chart with twelve evenly
spaced phase rotations. It selects using only the 27 interior lattice samples;
256 separate random queries measure held-out error. A candidate failing during
held-out evaluation cannot cause selection of a different chart: the selected
candidate must itself remain valid. This is an audit, not a builder policy.

The initial run found unchanged DVD-grazing error, a small CD-grazing reduction
(0.000131136 to 0.000129420), a dielectric-cutoff regression (0.00163557 to
0.00179979), and a CD-cutoff improvement (0.0203229 to 0.00949953). All thirteen
candidates evaluated successfully. The latter remains well above 0.001, so
this does not solve the refinement problem. The corrected held-out failure
handling was rerun successfully with identical results in
`cell_chart_phase_selection_verified.json`; the initial record is
`cell_chart_phase_selection.json`.

The Blender target rebuilt successfully with the complex-response cache
acceptance code (`build_complex_cache.log`). This build did not update the
installed app or rerender the provisional scenes.

### Tensor-quadratic Bernstein chart experiment

The new `--quadratic-cell` audit fits a common chart to a 3x3x3 nodal lattice,
converts those values to quadratic Bernstein control matrices, then matches the
interpolated operator to exact external channels. Bernstein weights are
nonnegative and sum to one inside the cell. Positive-semidefinite Hermitian
parts of all control matrices therefore suffice for passivity throughout the
cell; the audit checks their eigenvalues rather than assuming a higher-order
interpolant is passive. No clamping or energy rescaling is applied.

On the same four cells and 283 queries as the linear-chart audit:

| Cell | Maximum power error | Maximum complex Frobenius error | Minimum control dissipation |
| --- | ---: | ---: | ---: |
| DVD grazing | 2.41684e-7 | 9.95971e-7 | 0.0224206 |
| CD grazing | 1.27707e-6 | 4.53044e-6 | 0.00516650 |
| Dielectric cutoff | 4.56585e-5 | 0.000486948 | -1.94e-13 |
| CD cutoff | 0.000476401 | 0.00234852 | 0.0247601 |

The dielectric negative eigenvalue is within the explicit 1e-9 numerical
roundoff tolerance; its maximum physical polarization gain was 1. All metal
controls had positive dissipation. Power errors improved by approximately
36–103 times over the baseline linear charts. The CD cutoff still fails a
0.001 complex tolerance. This representation uses 27 matrices versus 8 per
cell and has not yet been integrated into cache construction or Metal.
Records: `quadratic_cell_audit.json`, `quadratic_cell_verified.json`. The latter
verified reconstruction at all 27 nodes, with maximum relative matrix-component
error 5.33e-16 and unchanged response results.

### Host quadratic cells and adaptive builder integration

The host implementation now prepares degree-two chart cells, validates all 27
control matrices for passivity, evaluates them with nonnegative tensor Bernstein
weights, and packs their degree with the matrix payload. Linear cells retain
their existing eight-control layout. `--quadratic-cache` enables a builder
attempt after lower-cost representations fail; an inadmissible fit is discarded
and the region subdivided. Modal solves reuse the existing per-build cache.

All 47 host tests pass. New tests cover metallic and lossless profiles, exact
reconstruction at all 27 fit nodes, independent complex-response queries,
all-polarization lossless energy, packing layout and malformed control counts.
A builder test verifies a dielectric cutoff region fails at depth zero with
linear cells but passes as one quadratic cell for both power and complex
tolerances of 0.001. Quadratic construction remains opt-in pending domain
audits and Metal support; current GPU consumers still handle degree one only.

The 256-node quadratic audit covered 4.0039%/6.8207% of the DVD/CD domains,
versus 0.78869%/0.61035% with linear charts at the same node budget. It accepted
121/124 quadratic cells and used 352160/1966464 matrix bytes. Both remained
incomplete. Observed times were 4.36/4.92 seconds; domain fractions are not ray
coverage or uniform progress estimates. Record: `full_cache_quadratic_256.json`.
A 4096-node audit is running to test whether this improvement scales.

### Quadratic interpolation on Metal and larger construction audit

The shared chart matcher now specializes by degree, supporting both eight
linear and 27 quadratic Bernstein controls. The degree-one arithmetic is
preserved. The Metal harness selects the matching specialization and checks
quadratic results against double interpolation at identical weights/rays.
All 87 queries and 2520 complex coefficients per dispatch passed; maximum
complex error was 5.10e-7 and lossless energy error 4.41e-7. Degree-one GPU
regression and all 47 host tests also passed. Records: `metal_quadratic_cell.json`,
`metal_linear_after_quadratic.json`.

The stored-cache fixture now uploads matrix offsets and polynomial degrees.
With quadratic fitting enabled, its four linear cells became one quadratic
cell. All 73 incident-channel queries passed end-to-end on the M5; maximum
complex-column error against direct solves was 6.89e-6. Record:
`metal_quadratic_stored_cache.json`. This remains a small fixed-layout fixture,
not general scene/device cache integration.

The completed 4096-node quadratic audit covered 50.6645%/51.6602% of DVD/CD
domain volume, with 7893408/45739296 packed matrix bytes and 1938/2044 quadratic
leaves. Both exhausted their node budget and returned no cache. A 16384-node
audit is running with the same tolerance 0.001 and memory limit 256 MiB.
Record: `full_cache_quadratic_4096.json`. Larger-chart GPU matching remains
costly (about 69–71 ms for 262144 18-channel hot-cell evaluations in the
quadratic run); these measurements exclude rendering and full-cache access.

### Complete quadratic domain builds

Both cases in `full_cache_quadratic_16384.json` completed at the unchanged
0.001 sampled power-error criterion. DVD used 6661 nodes, 3331 cells and
17674080 packed matrix bytes; CD used 7091 nodes, 3546 cells and 91766304 bytes.
Observed construction times were 104.18/135.49 seconds, including overlap with
other tests. Each returned its complete cache and reached domain fraction 1.
This establishes construction feasibility for these fixed N=16 profiles, not
held-out accuracy, modal convergence, textured-parameter coverage or shader
integration. The audit currently discards its in-memory cache after reporting;
persistence and independent whole-domain evaluation remain necessary.

### Reduced-scratch chart matching experiment

An opt-in matcher overwrites the interpolated chart during elimination. For
`c=conj(rotation)` and `A=(I+Y)-c R (I-Y)`, solve `A x=T`, then recover the
outgoing reference field through `(I+c R) z=c (2x-T)`. Per-port TE/TM bases
diagonalize R and T, avoiding a second dense matrix. Where a propagating
boundary denominator has magnitude below 0.25, the original formulation is
used before mutation to avoid cancellation amplification near grazing.

All 48 host tests pass, including nearly real chart rotations and longitudinal
wave numbers down to 1e-12 against an analytic scalar scattering response.
The guarded Metal path passed the existing 87 queries, with maximum complex
error 6.96e-7. Observed 18-channel times were 59–61 ms per 262144 hot-cell
evaluations, versus roughly 69–71 ms in earlier quadratic runs. These were not
isolated paired benchmarks. The path remains opt-in (`--inplace-chart`) and
requires broader GPU conditioning tests before adoption. Records:
`metal_inplace_chart.json` (unguarded experiment),
`metal_inplace_chart_guarded.json` (guarded implementation).

### Whole-domain held-out packed-data audit

The complete-cache driver now optionally validates the returned packed data
using independent samples (`--queries`, `--seed`). Half the probes are uniform
in the requested domain; half target Rayleigh thresholds with randomized
signed offsets. Empty physical responses are counted separately. The audit
uses the actual tree, float matrices, chart degrees and shared float matching
routines, and compares every physical incident/output order against direct
same-modal-count Maxwell solves. Boundary coefficients are constructed in
double then quantized, so GPU boundary arithmetic is not exercised here.

Reports include failures, maximum and 95th-percentile power-column error,
the number exceeding construction tolerance, and the worst query. The held-out
report threshold is explicitly twice the construction tolerance, consistent
with earlier independent-query audits; it is not a change to construction
acceptance. No failed query is dropped from the failure count. A 2048-query
run per material is in progress (`full_cache_held_out_2048.json`). The initial
compile failure due to a missing `<iomanip>` include was corrected and retained
in `full_cache_held_out_compile_failure.log`.

### Held-out result and audit artifact export

`full_cache_held_out_2048.json` completed. DVD evaluated 1909 physical queries
and 139 empty queries with no operational failures, but failed the declared
0.002 held-out threshold: maximum error 0.00249570, p95 0.000632111, with 12
queries above construction tolerance 0.001. Worst folded query was
(-0.0160648692, -0.6939837933, 644.872498 nm). CD evaluated 1985 physical/63
empty queries, no operational failures, maximum 0.00168655 and p95 0.000559671;
14 exceeded construction tolerance, but the 0.002 held-out criterion passed.
The DVD failure remains unresolved; successful construction was not sufficient.

The audit driver can now retain complete packed caches with `--cache-directory`.
Its versioned JSON/binary audit format has explicit profile, bounds, flags,
16-byte-aligned typed sections and per-cell matrix/port/active offsets plus
polynomial degree. It is not a Blender scene or production disk-cache format.
Existing artifacts are never overwritten. Reports record SHA-256 hashes and
associate exported artifacts with their held-out results, including failures.
The small export smoke test passed section size/alignment, finite-value and
offset checks, and rejected a second export to the same location. Evidence:
`export_smoke_v1/verification.json`, `export_smoke.log`.

A repeat build is running to retain the same full caches for failure diagnosis
and later GPU upload (`full_cache_exported_audit.json`,
`full_cache_artifacts_v1/`). Export does not approve a cache for rendering.

### DVD failure diagnosis and near-face acceptance probes

Both audit caches were exported and the same held-out results reproduced.
The first report omitted its export-status fields; this reporting bug is fixed
for future runs. The original report is preserved, with binary/manifest hashes
and validation results recorded separately in
`full_cache_artifacts_v1/artifact_manifest.json`.

The new cache inspector locates cells without rebuilding. DVD's worst query
lies in quadratic cell 1597, bounds (-0.125,-0.71875,630) to (0,-0.6875,680),
at local coordinates approximately (0.87148,0.79252,0.29745). A standalone cell
probe reproduced power error 0.00249565809 in double versus packed float
0.00249569795, identifying interpolation rather than quantization as the cause.
The regenerated packed matrix payload exactly matched the exported cell's
FNV-1a identity hash `50ea1c8f38aa3d39`. Records: `dvd_worst_cell.json`,
`dvd_worst_cell_probe.json`. The probe is currently limited to the fixed metal
audit profiles and explicitly rejects other profiles.

Quadratic acceptance now evaluates a second tensor lattice at three-point
Gauss-Legendre abscissae, approximately 0.112702/0.5/0.887298, in addition to
the original quarter-point lattice. This increases checks to 54 per quadratic
cell and samples nearer its faces; linear/intensity checks remain unchanged.
All 49 host tests pass. The regression rejects the formerly accepted region
at depth zero and completes after refinement without changing tolerance.
This addresses the observed sampling gap but does not establish a uniform
bound. A new full-domain run with 4096 held-out queries is in progress:
`full_cache_near_face_audit.json`, `full_cache_near_face_artifacts/`.

### Completed near-face whole-domain audit

Both caches completed and exported. DVD used 7037 nodes, 3519 cells and
18066688 matrix bytes; CD used 7561 nodes, 3781 cells and 96033600 bytes.
Observed construction times were 189.49/225.59 seconds. Of 4096 queries per
material, DVD had 3814 physical/282 empty responses and CD 3961 physical/135
empty responses. Both had zero operational failures and passed the unchanged
0.002 held-out threshold. Maximum power-column errors were
0.00108473006/0.00102866883 and p95 errors 0.00056813749/0.00050499317. Four DVD
and two CD queries exceeded construction tolerance 0.001; this remains an
empirical criterion rather than a uniform bound. Export hashes are included
in `full_cache_near_face_audit.json`. Modal convergence is not established by
comparison against the same N=16 solver.

Work started on an exported full-cache Metal harness supporting variable cell
sizes and degrees. The outside-sandbox command that would create its host
driver and launch it did not execute: automatic approval review returned an
authentication-service failure (401), not an unsafe-action determination.
Only its `.metal` source was written before that command. No full exported-cache
GPU result is claimed, and no alternate route around the approval check was
used. CPU and compile-only work can continue independently.

The Blender target rebuilt successfully with quadratic cells, near-face
validation and the shared matching changes (`build_quadratic_near_face.log`).
The installed app and provisional renders were not replaced by this build.

### Modal error and stricter packed-power audit (2026-09-26)

The same 64 deterministic samples (seed 617923) were compared at multiple
modal truncations for the fixed metal profiles used by the full caches.
`modal_16_64_cache_profiles.json` reports maximum power-column L1 differences
0.0198621 (740 nm pitch) and 0.0580441 (1600 nm). Increasing the lower
truncation to 32 reduces these to 0.0101815 and 0.0104704 in
`modal_32_64_cache_profiles.json`. These are absolute power-column differences,
not relative image errors. The higher-order solve is another approximation,
not ground truth. Each report now records its worst power query. Cache
interpolation accuracy against N=16 does not remove this modal error.

The packed cache audit previously compared only outputs present in the
reference physical-port list. It now compares every stored output, treating
ports absent from the reference as zero-power outputs, and rejects nonfinite
or negative power, missing reference ports and duplicate physical mappings.
Seven synthetic regression checks pass in `packed_audit_regression.log`;
source: `tests/performance/cycles_diffraction_packed_audit_test.cpp`.
The earlier full-cache audit results predate this stricter comparison and
have not been relabeled as reruns.

A convergence-acceleration candidate is adaptive spatial resolution:
[Messina et al., arXiv:1612.05516](https://arxiv.org/html/1612.05516v1),
section III, derives a coordinate stretch concentrating resolution at
permittivity interfaces, transformed Maxwell operators and boundary matching
for conical incidence. Its reported application is metallic-grating heat
transfer. Applying it here requires transforming exterior bases consistently
and validating amplitudes, passivity and convergence; its reported improvement
is not a measured speedup of our solver. No ASR implementation is claimed yet.

The same-sample N=64 versus N=128 run completed in
`modal_64_128_cache_profiles.json`. Maximum power-column differences were
0.00232484 (DVD) and 0.00464277 (CD); maximum individual complex-amplitude
differences were 0.00240208 and 0.00403919. These still do not establish a
converged reference.

### Exported full-cache Metal harness

`tests/metal/cycles_diffraction_cache.mm` now loads audit exports, checks buffer
sizes/layouts, prepares exterior boundary coefficients and compares GPU powers
and leaf IDs with the shared CPU float matcher. The driver exposes
`--cache-directory`, `--cache-queries` and a strictly CPU `--host-only` mode.
The latter returns before creating a Metal device. CPU preparation loaded
3519 DVD/3781 CD cells, with 1116/2405 physical incident queries from 512
sample positions (`full_cache_host_load_740.json`, `full_cache_host_load_1600.json`).
Host compilation succeeded. This harness checks upload, lookup and evaluation;
it uses precomputed exterior coefficients and does not test modal convergence,
GPU boundary construction, transmitted incident ports or actual rendering.
The outside-sandbox approval path succeeded on the subsequent request; the
DVD GPU run was launched, superseding the earlier authentication-service block.

Both outside-sandbox GPU runs passed on Apple M5:
`metal_full_cache_740.json` (1116 incident queries) and
`metal_full_cache_1600.json` (2405), zero failures, maximum absolute power
error 4.76837158e-7 in each. Every query also checked its leaf ID. Observed
single-dispatch GPU times were 0.4571 ms / 5.0907 ms. These include neither
reference preparation nor cache construction; they are not repeated timing
benchmarks or render overhead measurements. No physical material integration
is implied. The driver subsequently added the CPU audit-helper hash to future
reports; these first two reports predate that metadata addition.

### Full-cache evaluation with Metal boundary construction

The exported-cache harness now supports `--construct-boundaries`. In this
mode it uploads float ray/profile inputs instead of precomputed boundary
coefficients; the Metal kernel uses `diffraction_grating_boundary` for every
cell port before matching. CPU reference coefficients are calculated in
double from the same quantized ray. Chart and intensity dispatch now accept
either device or thread boundary pointers. The existing precomputed mode is
retained. Lookup coordinates remain the sampled cache coordinates: this test
does not yet derive cache coordinates from a production ShaderData ray.

Host compilation and CPU preparation passed. The outside-sandbox DVD run
`metal_full_cache_boundary_740.json` passed 1116 physical incident queries
with zero failures, maximum absolute power difference 8.34465027e-7 and a
single-dispatch time of 0.4827 ms. These results include coefficient
construction and matching, but do not establish render overhead, modal
convergence, transmitted incident support or coherent transport.

The corresponding CD run `metal_full_cache_boundary_1600.json` also passed:
2405 physical incident queries, zero failures, maximum absolute power
difference 7.15255737e-7, single-dispatch time 6.4040 ms. Neither single-run
timing should be used as a controlled performance comparison with previous
runs. Both reports include source and exported-cache hashes.

The precomputed-boundary mode was rerun after the pointer-dispatch changes:
`metal_full_cache_precomputed_regression.json` passed all 1116 DVD incident
queries, maximum power difference 4.76837158e-7, zero failures. This preserves
the original upload/matching check alongside the new construction mode.

### Ray-to-cache coordinate mapping

Added shared kernel helper `diffraction_coordinates.h` to reduce a signed
incident wave-vector direction into a canonical Bloch coordinate and integer
incoming order. It normalizes input directions, uses extended-float arithmetic
for reduction, canonicalizes float-rounded half-order seams, and applies the
existing mirror transform to both coordinates and incoming order. Invalid
inputs and orders outside safe int32 representation fail explicitly.

`cycles_diffraction_coordinates_test.cpp` independently reconstructs momentum
against double arithmetic for 40024 cases (including both mirror modes,
non-unit directions, exact/adjacent half-order seams and large order indices).
All checks passed, maximum absolute momentum error 2.74622e-8, plus three
invalid-input rejection checks (`coordinates_host.json`). This helper is not
yet called by a Blender shader or the full-cache evaluator. A dedicated Metal
harness checks the shared mapping under fast math using boundary-fixture rays
and additional seam inputs; it is exposed as `--coordinates`.

The outside-sandbox coordinate test passed on Apple M5 under fast math:
1047 fixture inputs, 262144 evaluations per run across five runs, zero
failures and zero observed CPU/GPU coordinate differences
(`metal_coordinates.json`). The initial report inherited coefficient/q2
field names from the boundary harness; the source now names the error
`max_coordinate_error` and removes the irrelevant q2 field. No performance
claim is based on its variable single-dispatch timings.

### Ray-derived full-cache lookup

The full-cache harness now has `--ray-lookup`, which implies GPU boundary
construction. It samples incident directions and wavelengths, derives the
canonical/folded coordinates and incoming order, looks up the cell, and
constructs each exterior boundary from the ray and loaded port metadata.
The GPU receives raw directions and wavelengths for lookup rather than
CPU-selected cache coordinates or incident order indices. CPU reference
preparation independently looks up the same shared-coordinate result and
matches with double-derived boundary coefficients. Every physical ray must
produce exactly one input; missing inputs fail the run.

Host preparation passed for 4096 DVD rays (`ray_cache_host_740.json`). This
connects the coordinate helper to the exported-cache evaluation harness,
not yet to ShaderData or a Blender material. It compares unpolarized powers;
restoring Jones matrices to an unfolded polarization frame remains outside
this harness's scope.

Both ray-derived GPU runs passed outside the sandbox on Apple M5:
`metal_ray_cache_740.json` and `metal_ray_cache_1600.json`, each 4096 rays,
zero failures and maximum absolute power difference 7.74860382e-7. Leaf IDs
also matched. Observed single-dispatch times were 0.5505 ms and 9.6954 ms;
these are not controlled repeated benchmarks or render-overhead estimates.
The much larger CD evaluation cost still needs investigation. Both tests use
fixed-profile, finite-N exported caches and cannot prove modal convergence.

### Repeated full-cache timing and guarded in-place comparison

The full-cache harness now validates seven consecutive dispatches and records
five GPU timings after two warmups. It explicitly labels the workload as a
hot repeated ray batch, excluding allocation, compilation, reference preparation
and rendering. `--inplace-chart` is now available for full-cache tests and
selects the existing guarded in-place chart solver on the GPU; the CPU
reference continues to use the original matcher.

The CD baseline run `metal_ray_cache_cd_repeated.json` passed all seven
4096-ray dispatches, maximum power difference 7.74860382e-7. Recorded GPU
times were 2.4671, 2.2831, 2.5751, 2.3835 and 2.2594 ms (median 2.3835 ms).
The earlier 9.6954 ms single dispatch is therefore not representative of this
hot repeated workload. No rendering performance inference is made.

The in-place run `metal_ray_cache_cd_inplace_repeated.json` also passed all
seven dispatches, maximum power difference 8.34465027e-7. Its five GPU timings
were 2.8186, 2.3370, 2.1565, 2.4012 and 1.9671 ms (median 2.3370 ms).
The ranges overlap substantially; this sequential comparison does not
establish a repeatable speedup. The original matcher remains the default,
and no numerical tolerance or fallback condition was relaxed.

### Shared kernel dispatch

Moved variable-size chart/intensity matching dispatch from the standalone
Metal harness into `kernel/util/diffraction_evaluate.h`. Both the CPU packed
cache audit and full-cache Metal harness now call this shared implementation.
The caller declares channel capacity; oversized cells fail rather than silently
truncating physical ports. Invalid incident indices, feedback counts and chart
degrees are rejected. Original and guarded in-place chart implementations remain
selectable; the original remains default. The helper produces Jones data for
chart cells and powers for incoherent intensity cells, without conflating them.

All 12 packed audit/dispatch regression checks passed
(`packed_audit_dispatch_regression.log`). The shared Metal path passed seven
4096-ray CD runs outside the sandbox, zero failures and maximum power difference
7.74860382e-7 (`metal_shared_dispatch_cd.json`). Blender and its Metal libraries
rebuilt successfully (`build_shared_dispatch.log`). This is reusable kernel
infrastructure; no physical shader, scene cache manager or coherent integrator
connection is claimed. The installed app was not updated.

### Reflected and transmitted order directions

Added `kernel/util/diffraction_direction.h`, constructing a propagating order
from signed tangential incident momentum, real exterior indices, wavelength,
pitch and relative order. Reflection uses positive local z and transmission
negative local z. Evanescent and exactly zero-normal-flux orders return false;
callers must distinguish this from absorption when enumerating physical ports.
The helper uses the existing compensated boundary calculation and preserves
the longitudinal component near grazing. It is not yet wired to a BSDF sampler.

The independent CPU test covers 40000 reflection/transmission cases:
13256 propagating and 26744 closed, maximum momentum/unit-length error
1.78814e-7, plus invalid and exact-zero-flux rejection checks. The first test
report (`direction_host.json`) failed because its reference multiplied two
floats before promotion to double. Correcting that reference, without changing
the helper or tolerance, produced `direction_host_verified.json` with zero
failures. Both reports are retained. A `--direction` Metal mode compares the
helper on 1047 fixture inputs, including exterior cutoffs and seam examples.

The outside-sandbox Metal direction test passed on Apple M5: 1047 inputs,
262144 evaluations per run over five runs, zero failures, maximum CPU/GPU
direction difference 1.19209290e-7 (`metal_direction.json`). The initial report
retained an inherited coordinate-test scope string; the source's scope text
has been corrected for subsequent reports. This tests directions and open/closed
classification, not diffraction efficiencies or end-to-end scattering sampling.

### Shared compensated momentum calculation

Extracted `diffraction_grating_momentum` from boundary construction. It returns
compensated tangential momentum and the signed longitudinal square without
Fresnel work. Boundary matching consumes this result, while direction
construction now avoids the former boundary-coefficient calculation and second
normalization. No classification tolerance or numerical fallback was changed.
The 40000-case CPU direction test retains its 1.78814e-7 maximum error and
zero failures (`direction_shared_momentum_host.json`).

Outside-sandbox Metal boundary regression passed all 1035 fixture cases across
five runs (`metal_boundary_shared_momentum.json`): maximum coefficient error
2.52881091e-7 and energy error 3.59338817e-7. This establishes correctness for
the tested inputs, not a measured end-to-end speedup.

The streamlined direction helper also passed the Metal regression:
1047 cases over five runs, zero failures and maximum CPU/GPU direction
error 1.19209290e-7 (`metal_direction_shared_momentum.json`). Highly variable
microbenchmark timings are retained without a speedup claim. The helper still
requires connection to BSDF order sampling and the physical material pipeline.

### Discrete sampling of diffraction powers

Added `kernel/util/diffraction_sample.h` for incoherent order selection.
It samples port m with probability P_m/sum(P) and returns throughput sum(P),
so normalization of sampling probabilities does not remove absorption. It
rejects invalid/all-zero columns, never selects zero-power ports, and does not
clamp an over-unity input column to conceal gain. Radiance/importance index
conversion, directional Jacobians and polarization transport are explicitly
outside this helper and remain BSDF responsibilities.

The CPU regression uses 65536 stratified samples of a column summing to 0.5;
counts were [0,16384,32768,0,16384], and each estimated outgoing power matched
its reference exactly. Endpoint, zero, negative, nonfinite and unclamped-gain
checks passed (`order_sample_host.json`). The full-cache harness now exposes
`--sample-orders`, deriving lookup from rays and validating sampled port,
probability and throughput against CPU-computed columns on every dispatch.
This is not yet a connected Blender BSDF or a coherent-amplitude sampler.

The outside-sandbox Apple M5 sampling run passed seven 4096-ray CD dispatches
with zero failures (`metal_sample_orders_cd.json`). Validation checked all
power outputs, leaf IDs, sampled ports, probability and throughput (2e-5
absolute tolerance for the latter two). Maximum power difference remained
7.74860382e-7. These finite sampled checks do not establish general unbiasedness
of floating-point sampling or render performance; actual BSDF integration,
transport-mode factors and coherent transport are still unfinished.

### Scene-layer device buffer preparation

Inspection confirms the existing Glossy SVM/OSL paths still select the
provisional phase-screen closure. The physical cache has no Scene-owned
device manager or closure handle yet. As a prerequisite to that connection,
flattening moved from the audit exporter into
`diffraction_grating_device_buffers` in the scene layer. The contiguous layout
matches the arrays already exercised by the full-cache Metal tests. The audit
exporter now calls this shared preparation function.

Preparation validates tree structure/reachability and leaf references, int32
sizes/offsets, matrix shape and finiteness, port ordering, active-port indices,
and representable nonempty cell bounds. On failure it returns no partial output.
This validates upload representation, not electromagnetic accuracy or passivity.
A new scene regression checks offsets across intensity/quadratic cells and
rejects malformed trees, matrix lengths, active sets and bounds. This does not
yet allocate a device buffer or change shader behavior.

All 50 host solver tests passed (`device_buffers_tests.log` / XML), including
the new packing regression. Blender and the Metal libraries rebuilt successfully
(`build_device_buffers.log`). The installed application was not replaced.
Scene-owned allocation/lifetime and a physical closure handle remain pending.

### Scene-owned diffraction manager

Added `DiffractionManager`, owned by Scene, with transactional registration of
multiple prepared caches, stable index handles within each registered set,
contiguous typed device arrays, per-cache base descriptors/domains, scene-update
upload and scene-cleanup release. Cell-local offsets remain local; descriptor
bases identify each cache's ranges. Replacing with an empty set frees storage on
the next update. Failed registration preserves the prior set. No cache data is
allocated by default. Callers follow the existing scene-update lock convention.

The manager is included in Scene's update predicate and device update/free
paths. Device arrays are declared for kernel backends. Registration and upload
implementations are separate so pure host registration can be tested without
creating a device. All 51 host tests passed (`manager_host_tests.log`/XML).
Both the initial and final reorganized builds passed
(`build_diffraction_manager.log`, `build_diffraction_manager_split.log`).
Shader compilation does not yet register a physical cache or provide a closure
handle, so no physical grating rendering or nonempty manager upload is claimed.

Runtime validation: launching the bare build app first failed before startup
because its bundle lacks liboslexec.dylib (`manager_metal_smoke.log`). Running
the new executable with the installed runtime libraries/resources rendered the
128x64, 4-sample provisional pitch scene successfully
(`manager_metal_smoke_runtime.log`, `/tmp/diffraction_manager_smoke.png`).
That run may use installed precompiled kernels, so a separate source-override
run was launched with CYCLES_KERNEL_PATH pointing to this worktree and only
Apple M5 enabled. The installed executable/resources were not modified.

The current-source M5 smoke render completed successfully
(`manager_metal_smoke_source.log`, `/tmp/diffraction_manager_source_smoke.png`),
with CPU and the stale M2 preference entry disabled. The image was inspected;
it is a very low-sample provisional pitch chart, not a physical-quality result.
This verifies the existing renderer can run with the expanded kernel layout
and empty diffraction manager. Nonempty manager upload and closure consumption
still require dedicated validation.

### Upload failure handling

Inspection found that `DiffractionManager::device_update` cleared its dirty
flag regardless of device-copy errors. It now stops on the first device error
and retains the pending host set so recovery retries all arrays. It also checks
host allocation before copying. Scene's existing device-error checks prevent
rendering after a partial failed upload. This is retry behavior, not an atomic
GPU transaction. Host copies now use typed assignment rather than memcpy to
avoid non-trivial-vector compiler warnings.

The first build passed but emitted two vector-copy warnings
(`build_manager_upload_errors.log`); the typed-copy follow-up is being rebuilt.
Runtime allocation-failure injection and nonempty manager uploads remain
unverified. No successful recovery test is claimed from code inspection alone.
The typed-copy follow-up rebuilt successfully without warnings in its log
(`build_manager_upload_errors_clean.log`).

### Integer-order precision

The momentum helper previously converted the integer order directly to float,
losing unit increments above 2^24. It now splits the integer into exactly
representable high/low parts before compensated multiplication. The split
avoids converting rounded INT_MAX back to int. Two exact cancellation tests
at orders +/-33554431 now recover outgoing x = -/+2^-25 rather than zero.
The 40000-case CPU direction regression also passes
(`direction_large_order_host.json`). This extends helper precision; it does
not increase the channel capacity of prepared caches or validate enormous
physical gratings. The existing Metal boundary fixture remains a regression
for its ordinary order range, not a large-integer GPU test.
The first integer split was not normalized, violating the compensated
multiply's expansion assumption; Metal grazing tests caught this
(`metal_boundary_order_split.json`, retained). Normalizing with TwoSum fixed
the regression without changing tolerances. The CPU cancellation tests pass
(`direction_large_order_normalized.json`) and all 1035 Metal boundary cases
pass (`metal_boundary_order_split_normalized.json`), maximum coefficient
error 2.52881091e-7.

### Integer-valued GPU order fixture

The Metal direction fixture now uploads orders in an int32 buffer, avoiding
loss during float fixture encoding. Four new cancellation cases use orders
+/-33554431 and +/-2147483647. Their outgoing x component is checked exactly
against a double-derived analytic reference, so the ordinary 3e-5 vector
comparison cannot hide loss of a unit order. All 1051 cases passed over five
outside-sandbox Apple M5 runs (`metal_direction_integer_orders.json`), zero
failures and maximum ordinary-case direction error 1.19209290e-7. This closes
the GPU precision-test gap for those four cases, not general cache capacity
or physical modal convergence.

### Device-memory integration test target

Added the off-by-default WITH_CYCLES_DIFFRACTION_DEVICE_TEST target, separate
from general GTests. It constructs a real Cycles Device/DeviceScene, registers
two nonempty synthetic caches, uploads and reads back matrices/descriptors,
then exercises release/re-upload and replacement with an empty set. It accepts
--metal for the GPU device and defaults to CPU. These synthetic matrices test
memory layout/lifetime, not a physical material response. The initial target
build failed because the test subdirectory was gated on WITH_GTESTS; the new
option now enables that subdirectory independently. Runtime checks remain
pending until the target links and executes successfully.

The target linked successfully (`build_device_test_enabled.log`; linker noted
existing duplicate static libraries). Both real-device runs passed:
`device_memory_cpu.log` and outside-sandbox `device_memory_metal.log` on Apple
M5. Each verified two-cache matrix/descriptor readback, release/re-upload and
clear through DiffractionManager/DeviceScene. This validates nonempty memory
operations for the synthetic fixture, not shader consumption, physical results,
or allocation-failure recovery.

### Replacement and device-error recovery test

Extended the real-device test to replace values without changing buffer sizes,
then register another change while intentionally setting a sticky Device error.
The manager must remain dirty and leave old host data unchanged. After releasing
the old storage, a fresh Device/DeviceScene receives the pending data, verified
by readback, and the cache set is cleared. CPU and outside-sandbox Metal runs
passed (`device_recovery_cpu.log`, `device_recovery_metal.log`). Logged errors
are deliberately injected by the test. This covers pre-existing device-error
recovery with device recreation, not allocation failure midway through upload.

### Multi-cache cell view

Added `kernel/util/diffraction_cache_view.h` to resolve a registered cache
handle into a prepared cell's global matrix/port/active offsets, interpolation
coordinates, degree, rotation and mirror transforms. It uses the manager's
per-cache descriptors/domains and existing tree lookup. Inputs are validated
manager arrays; invalid handles and absent/out-of-domain cells fail without
extrapolation. This is a shared kernel helper, not yet a closure call site.

The real-device integration executable now checks second-cache offsets and
center coordinates, plus invalid handles and an out-of-domain wavelength.
Its CPU run passed (`cache_view_cpu.log`, `build_cache_view_test.log`). The
new view helper ran on the host; this is not evidence of GPU execution of it.
Existing upload/replacement/recovery checks also continued to pass.

### Metal multi-cache view test

Added a dedicated `--cache-view` Metal test using two distinct cache base
layouts and independent literal expected results. Seven cases cover global
offsets for the second cache, folded coordinates, reverse-order and cross-
polarization signs, wavelength endpoints, invalid handles and out-of-domain
lookup. The outside-sandbox Apple M5 run passed all 84 compared components
exactly (`metal_multi_cache_view.json`). This executes the shared cell-view
helper on GPU; it does not yet combine real manager allocations with shader
consumption or validate physical scattering.

### KernelGlobals scene-cache access

Added `diffraction_scene_cell_view` using the normal kernel_data_array memory
abstraction. KernelTables now uses its former padding integer for
num_diffraction_caches, preserving structure size. The manager publishes this
count after successful upload and resets it when freeing storage. No closure
currently calls the wrapper.

The integration test constructs ThreadKernelGlobalsCPU over the scene arrays,
checks second-cache resolution/invalid handles through the wrapper, and verifies
count reset on free/clear. It passed (`kernel_scene_view_cpu.log`). Including
kernel globals exposed an ambiguous Device name from OpenPGL; the first build
failure is retained, and explicit ccl::Device qualification fixed it
(`build_kernel_scene_view_fixed.log`). GPU execution of this wrapper remains
unverified, though the lower-level view was previously tested on Metal.
Blender rebuilt successfully with the cache-count publication change
(`build_blender_cache_count.log`). The installed app was not replaced.

### Matching through scene-owned arrays

Added KernelGlobals entry points for complex chart matching and incoherent
power evaluation from a resolved scene cell view. The complex entry explicitly
rejects intensity-interpolated cells. Results remain in the folded basis;
transport-mode scaling and sampling normalization are not applied here.

The CPU integration test now registers one chart cell and one intensity cell.
Intensity power matches 0.15625 exactly and coherent access is rejected. For a
constant rank-one chart with entries 0.25-0.125i, its independent analytic Jones
reference has diagonal 24/37+4i/37 and off-diagonal -13/37+4i/37; expected
unpolarized power is 777/1369. These pass within 2e-6 through scene-owned
KernelGlobals arrays (`scene_matching_analytic_cpu.log`). Existing upload and
recovery checks also pass. This is CPU kernel evaluation, not a physical BSDF
connection or GPU execution of the new wrappers.

### Real kernel compilation of scene helpers

The diffraction closure header now includes the scene-evaluation helpers so
they are parsed within the actual kernel compilation environment, including
MetalKernelContext rather than only standalone functions. Blender rebuilt
successfully with CPU kernel.cpp, OSL closures.cpp and software/MetalRT/motion
Metal libraries (`build_scene_helpers_in_kernel.log`). This verifies compile
compatibility; no call from the provisional closure or GPU execution of the
scene wrappers is claimed. The physical material integration remains pending.

### Targeted high-order convergence

Added `--modal-query BLOCH KY WAVELENGTH` to reproduce a specified incidence
instead of drawing a random batch. The C++ driver validates finite/domain
inputs and evaluates one query per fixed profile; source hashes and exact
arguments remain recorded. Query (-0.054513542712, 0.956627899111,
502.654393085 nm) reproduces the earlier worst CD incidence. Comparing
N=128 with N=256 yielded power-column L1 differences 0.00162159 for 740 nm
pitch and 0.00161379 for 1600 nm pitch, with maximum individual complex
amplitude differences 0.00209996/0.00160640
(`modal_target_128_256.json`). The same Bloch coordinate is applied to both
profiles; it is not the same physical angle for different pitches.

Each two-order comparison took about 5.2 seconds at this single query.
N=256 is not a ground-truth claim. These measurements reinforce that modal
error must be resolved independently of the N=16 cache interpolation tests;
raising modal order alone remains expensive. No physical closure connection
is justified by the cache's same-order agreement alone.

### Adaptive-coordinate convergence investigation

Reviewed Messina et al., arXiv:1612.05516v1, sections III.1–III.3:
https://arxiv.org/html/1612.05516v1 . This is a candidate method, not an
implemented solver or a demonstrated acceleration for the visible-light
profiles here. The published heat-transfer convergence results do not establish
our material accuracy.

For x=F(u), f=dF/du, the covariant tangential electric components are
E_u=f E_x and E_y unchanged. The exterior homogeneous regions also require
consistent transformed admittances. Equations (33)–(34) map transformed Fourier
coefficients back to Cartesian coefficients: T_x integrates
exp(i[k_m u-k_n F(u)])/D, while T_y includes an additional factor f(u).
The phase includes the Bloch offset, not just integer Fourier harmonics.
Consequently, replacing the layer's permittivity matrices alone while keeping
Cartesian exterior matching unchanged is not a consistent implementation.
Before adopting ASR, verify the identity-map limit, homogeneous-interface
Fresnel response, centered-profile phase/reciprocity, flux conservation, and
convergence against the existing high-order solver. Artificial reference ports
must remain defined in the physical Cartesian basis used by the cache.

The modal audit's maximum reference order was raised from 256 to 512 to match
the existing solver's accepted range. This changes only the audit driver; no
production solver truncation or cache defaults were changed.

The targeted N=256→512 run completed (`modal_target_256_512.json`). At the
same difficult query recorded above, maximum power-column L1 differences are
0.000539092241 (740 nm) and 0.000698938472 (1600 nm); maximum complex-entry
differences are 0.000506549651 and 0.000502542517. The paired solves took
45.65 and 46.49 seconds respectively on the host. This is improved successive
truncation agreement at one query, not a bound on error to the exact solution
or the whole domain. N=512 is still not designated ground truth. The cost
supports investigating faster convergence before rebuilding production-scale
caches; these timings are not GPU render overhead.

### Exact zero-depth interface limit

Zero-depth profiles now bypass the internal constitutive eigensystem. With
unit propagation and any nonsingular internal field basis, the existing
boundary equations reduce to continuity of tangential electric and magnetic
fields at the exterior interface. Using identity internal bases avoids both
the unnecessary eigensolve and failures caused by zero-q modes of a material
that occupies no thickness. Positive-depth profiles retain the modal solve.

Added `ZeroDepthIgnoresDegenerateInternalModes`: a 1000 nm-period absent layer
with index 0.55 at 550 nm has internal cutoff orders, but the air/glass
interface must still yield R=0.04 and T=0.96 with no nonzero diffraction orders.
All 52 host tests pass, including existing complex Fresnel, scattering-block,
reference-port and cache checks (`zero_depth_host_tests.log` and XML). Blender
rebuilt successfully (`build_zero_depth.log`). This is a host material-solver
fix, not physical shader integration or a GPU performance claim.

### Scene-owned physical order sampler

Added `diffraction_scene_sample` to connect ray coordinates, scene cache lookup,
physical boundary construction, unpolarized power evaluation, discrete order
selection and outgoing direction reconstruction. Its frame has x across the
grooves, y along them and +z into the upper exterior. Incident directions carry
propagation momentum. Both incident sides are supported by the interface;
mirrored cache orders are unfolded before direction construction. Relative
order arithmetic explicitly checks signed overflow. Flux throughput is not
clamped or normalized to unity. It does not apply radiance/importance index
conversion or a rough-surface Jacobian, which remain BSDF responsibilities.
The caller must supply the cache's matching pitch and real exterior indices.

The device integration executable now binds the port array into KernelGlobals
and tests the complete sampler against the analytic one-port chart used by the
existing evaluator test: reflected direction +z, relative order zero, discrete
probability one and throughput 777/1369. Missing opposite-side incidence and
out-of-domain wavelength reject. The CPU execution passed
(`scene_sample_cpu.log`). Blender and the integration executable rebuilt,
including the CPU/OSL and three Metal kernel variants
(`build_scene_sample.log`). Template parsing in Metal is not GPU execution of
this sampler. This analytic check does not cover nonzero orders, mirrored or
transmitted sampling; these require dedicated subsequent checks. No shader
calls this entry point yet and no physical render performance is claimed.

### Combined sampler routing coverage

The device integration executable now constructs a passive synthetic two-port
operator transferring half the incoming flux to the other side and a different
order. Forty-eight CPU sampler checks cover orders +/-1, both incident sides,
unequal exterior indices (1 and 1.5), all x/y sign combinations, mirrored and
unmirrored cache registration, and random inputs 0, 0.5 and the float below 1.
Expected output momenta are specified independently: top kx=+/-0.1 connects to
bottom kx=+/-0.6 for wavelength/pitch=0.5, with ky=+/-0.2 conserved. The tests
check signed output hemisphere, direction components, relative order, unit
selection probability and throughput 0.5. All passed (`scene_routing_cpu.log`),
as did existing registration/recovery checks. The executable rebuilt
(`build_scene_routing.log`). These are synthetic routing tests, not RCWA
accuracy, arbitrary power-distribution convergence, or GPU sampler execution.

### Solver-cache Fresnel sampling reference

Added six physical zero-depth air/glass cache fixtures (incidence angles 0,
0.4 and 0.9 radians from both sides, azimuth 0.3), built using the production
solver/cache builder. Cache bounds are small float-representable neighborhoods
of each query; acceptance tolerance is 1e-5 with N=4. These fixtures isolate
the flat-interface limit, not relief convergence. The combined sampler is
checked at 1024 stratified random inputs per fixture against independent
complex Fresnel equations and Snell momentum conservation. Checks include
per-event probability, unit total flux throughput, output direction, zero
relative order, and reflection frequency within one stratification bin plus
the numeric tolerance. The glass-to-air 0.9-radian case is total internal
reflection.

All 6144 physical-reference samples and 48 synthetic routing samples passed
(`scene_fresnel_cpu.log`). The executable rebuilt (`build_scene_fresnel.log`).
An initial attempt correctly rejected decimal-double cache bounds that were
not exactly float-representable; the fixture now explicitly constructs float
bounds. Running the same executable outside the sandbox with `--metal` passed
cache upload/recovery on Apple M5 (`scene_fresnel_metal_storage.log`). The
sample calculations in that executable still run on CPU through KernelGlobals;
Metal storage validation is not GPU execution of the combined sampler.

### Combined sampler executed on Metal

Extracted the array-backed implementation into `diffraction_scene_data.h`.
`diffraction_scene.h` retains the existing API and delegates using pointers
from KernelGlobals. The same implementation can now be instantiated in the
standalone Metal harness without duplicating sampling logic or substituting
kernel math. Array lifetime/validation requirements are unchanged.

Added `--scene-sample` to the Metal harness. It uploads four cache descriptors
and the passive two-port routing fixtures, then executes 48 combined samples
on GPU: +/-1 orders, unequal indices, both incident sides, x/y signs, mirror
folding and CDF endpoints. All 384 output components agree with independently
specified momentum/probability/throughput expectations within 2e-6; zero
failures (`metal_scene_sample.json`, source hashes included). This run used
Metal outside the sandbox. It is actual execution of the shared sampling
implementation, but not a shader render or a test of Cycles argument-buffer
binding. GPU physical Fresnel/relief cache checks remain separate future work.

The full Blender and integration target rebuilt (`build_scene_data.log`), and
the CPU integration test retained all 6144 physical Fresnel and 48 routing
passes after extraction (`scene_data_cpu.log`). No production shader invokes
the physical sampler yet. No rendering-overhead claim follows from 48 samples.

### Physical Fresnel caches sampled on GPU

Extended `--scene-sample` with six production-solver/cache-generated flat
interfaces at 0, 0.4 and 0.9 radians from both air and glass. The host flattens
each validated cache with `diffraction_grating_device_buffers`, packs its own
base offsets into the combined GPU arrays, and supplies pitch per ray. The
kernel invokes the shared combined sampler with four-channel capacity.

Actual Metal execution outside the sandbox passed 6192 cases: 6144 physical
Fresnel samples plus the existing 48 synthetic nonzero-order/mirror samples.
All 49,536 output components agree with independently calculated Fresnel
probabilities, Snell directions and unit flat-interface flux throughput within
2e-6 (`metal_scene_fresnel.json`, source hashes recorded). Stratified inputs
also check event selection against the analytical CDF, including total
internal reflection from glass. This is a flat-interface test, not validation
of finite-depth relief physics, a shader render, or rendering performance.

### Finite-depth dielectric relief sampled on GPU

Extended the combined Metal sampler test with six dielectric relief fixtures:
740 nm pitch, 150 nm depth, duty 0.41, ridge/substrate index 1.5 and groove/top
index 1, wavelength 550 nm, azimuth 0.3, incidence angles 0/0.4/0.9 radians
from both sides. Each small-domain cache uses N=16, retained half-orders 3 and
5e-7 interpolation acceptance. Reference probabilities come from a direct
N=16 Bloch scattering solve at the query, independently of cached matching;
expected outgoing directions follow momentum conservation. The Metal sampler
now instantiates its full 20-channel dispatch capacity.

The run outside the sandbox passed all 12,336 samples (98,688 output
components at tolerance 2e-6): 6144 finite-depth relief samples, 6144 analytic
Fresnel samples and 48 synthetic routing samples. See
`metal_scene_relief.json` for source hashes and execution result. Per-sample
checks cover selected order and side, direction, probability and total flux
throughput. This establishes same-truncation solver/cache/GPU consistency,
not convergence to exact Maxwell response; the independent modal audit still
applies. No physical shader render or render-speed claim is made.

### Incremental material cache registration

Added `DiffractionManager::add_cache`, returning a stable handle while
preserving previous registrations. It validates the new flattened cache,
domain and combined int32 limits before appending any content. Validation
failure returns -1 without consuming a handle. `set_caches` now builds a
replacement manager using the same registration path and commits it only
when every cache validates. Callers still need the scene update lock; this
is not concurrent registration or persistent material deduplication.

The registration test verifies sequential handles, invalid-add rejection,
continued allocation without a skipped handle, and transactional replacement
(`cache_registration_host.log`/XML). The integration executable now registers
its two nonempty caches incrementally and checks their global offsets,
analytic evaluations, upload and recovery; it passes
(`cache_registration_cpu.log`). Blender and the executable rebuild
(`build_cache_registration.log`). This prepares the host registration API;
shader compilation still does not create/register a physical material cache.

### Combined-sampler GPU timing baseline

The Metal combined-sampler harness now runs seven dispatches, validates every
run, and records five GPU timestamps after two warmups. Threadgroup size is
min(64, pipeline capacity). It rejects missing/nonpositive timing data and
reports the cache footprint and device. The M5 outside-sandbox run
`metal_scene_sampler_timing.json` measured 0.300250, 0.304542, 0.301083,
0.324625 and 0.300625 ms for 12,336 samples each (median 0.301083 ms).
All 690,816 checked components across the seven runs passed; maximum absolute
component difference was 1.19209e-7. Cache buffers occupy 130,516 bytes.

This is a hot fixed-query benchmark with neighboring rays grouped by fixture,
small caches and repeated directions. It excludes solver/cache construction,
upload, kernel compilation, BVH traversal, shader evaluation and image
accumulation. It cannot establish full-scene overhead or performance with
large caches and divergent materials. No speedup claim is made against the
old single-dispatch test, which used a different threadgroup size.

### Absorbing CD/DVD-pitch sampler fixtures

The combined Metal test now includes metal ridge/substrate profiles with
740 and 1600 nm pitch, 150 nm depth, duty 0.41, fixed complex index 0.9+6i,
and 550 nm illumination at 0/0.4/0.9 radians from air. These are generic
absorbing reference profiles, not measured aluminum dispersion or final disc
materials. No bottom incoming far-field channel is assigned to the absorbing
substrate. Reference columns are direct N=16 solves; each must have total
reflected power strictly between zero and one. GPU output throughput must
match that total rather than being normalized back to one.

Outside-sandbox M5 execution passed all 18,480 cases in each of seven runs:
1,034,880 component comparisons, maximum difference 1.78814e-7
(`metal_scene_absorption.json`). Five measured GPU batch times were 0.506500,
0.760458, 0.996500, 0.516542 and 0.573708 ms (median 0.573708 ms), with
177,584 cache bytes. Timing variability and the changed fixture mix prevent
a speedup/slowdown conclusion versus the earlier dielectric-only run. The
same hot fixed-query and non-rendering limitations apply. Modal convergence,
wavelength-dependent material constants, shader integration and physical disc
renders remain outstanding.

### Wavelength-dependent layer optical constants

`DiffractionGratingProfile` now optionally stores ridge and groove optical
constant spectra as wavelength/n+ik samples. The modal solve evaluates each
nonempty table by piecewise-linear interpolation in n and k; empty tables
retain the existing constant-index behavior. Samples must have strictly
increasing positive finite wavelengths and finite passive nonzero indices.
Queries outside table coverage fail instead of extrapolating or clamping.
The cache builder's wavelength-dependent modal calls therefore use these
layer values; exterior indices remain constant in this implementation.

`TabulatedLayerOpticalConstants` compares endpoints and an interior wavelength
against separate fixed-index scattering solves, including complex amplitudes
and passivity, and checks duplicate wavelengths, gain and out-of-range queries.
All 53 host tests pass (`layer_dispersion_host.log`/XML), and Blender rebuilds
(`build_layer_dispersion.log`). The test tables are synthetic, not measured
metal data. No substrate/exterior dispersion, data import, shader sockets or
GPU/render validation of dispersive caches is claimed yet.

### Spectrum coverage checked before cache pruning

Cache construction now validates both layer spectra at the domain's wavelength
endpoints before topology pruning or modal work. This catches invalid tables
or insufficient coverage even in domains with no propagating ports, where
there may be no solver call to perform validation. Rejection leaves the output
cache empty and performs zero reference solves.

`LayerSpectrumCacheCoverage` verifies successful construction with a covered
synthetic dispersive ridge, rejection beyond coverage, and rejection of a
malformed table in an all-evanescent domain. This and the layer interpolation
test pass (`spectrum_coverage_host.log`); Blender rebuilds
(`build_spectrum_coverage.log`). This is coverage validation, not a quantitative
GPU interpolation test of dispersive material data.

### Preserve optical-constant knots in cache topology

The adaptive builder now forces wavelength splits at all internal ridge/groove
spectrum knots before preparing/interpolating cells. Sorted unique split
positions are selected around their median to avoid an unnecessarily linear
tree. For a double-precision knot not exactly representable as float, both
adjacent float wavelengths are boundaries; the original spectrum is unchanged.
The tiny interval between them has no additional representable GPU wavelength.
These splits obey the normal depth/node budgets and fail explicitly when the
budget cannot resolve them. This prevents finite interior validation lattices
from completely overlooking a narrow tabulated material feature.

A regression uses a 0.25 nm-wide feature around 545.125 nm in a 540–560 nm
domain. It verifies no accepted cell straddles the representable knots, checks
both adjacent float boundaries for a knot at 545.13 nm, and checks empty-output
failure when the depth budget is insufficient. All 55 host tests pass
(`spectral_knots_host.log`/XML), and Blender rebuilds
(`build_spectral_knots.log`). Forced splitting preserves the tabulated feature;
it is not a uniform bound on Maxwell or interpolation error between samples.

### Dispersive ridge spectra exercised on Metal

The combined sampler now accepts wavelength per query. Nine additional
fixtures use a synthetic ridge n+ik table at 500/550/600 nm, sampled at
520/550/580 nm and incidence 0/0.4/0.9 radians. Their substrate remains the
constant absorbing 0.9+6i medium. Reference scattering uses independently
constructed fixed-index profiles at each wavelength, with spectra removed;
this avoids comparing the table lookup against itself. The 550 nm fixtures
also exercise lookup at a forced cache split on a material knot.

M5 execution outside the sandbox passed 27,696 samples in each of seven runs
(1,550,976 component checks); maximum difference 5.36442e-7
(`metal_scene_layer_dispersion.json`). GPU times were 1.87042, 2.46908,
0.766042, 0.677333 and 0.679583 ms; median 0.766042 ms. The spread is retained,
not filtered. This remains a hot small-cache benchmark, now 203,244 bytes,
and does not establish renderer overhead. Data are synthetic; measured metal
spectra, exterior/substrate dispersion and shader integration are not supplied
by this test.

### Absorbing-substrate optical-constant tables

Profiles now also accept `absorbing_substrate_spectrum`. The constant substrate
and every table entry must have positive extinction coefficient; tables that
change to a lossless half-space are rejected. This preserves the existing
absence of substrate far-field reference ports. The modal boundary admittance
uses the sampled complex index. Reference matching validates coverage, and
cache-domain validation/mandatory knot splits include the substrate table.
Lossless exterior dispersion is still unsupported and requires wavelength-
dependent port-bound handling plus matching GPU exterior indices.

The regression compares complex scattering matrices at 500/550/600 nm with
independently constructed constant-substrate profiles, checks passivity and
absence of substrate output ports, and rejects uncovered wavelengths and a
table reaching zero extinction. All 56 host tests pass
(`substrate_dispersion_host.log`/XML); Blender rebuilds
(`build_substrate_dispersion.log`). Tables remain synthetic. GPU validation
and material-data integration of this new substrate path are still pending.

### Substrate and shared-metal dispersion on GPU

Added eighteen combined-sampler fixtures: substrate-only dispersion and a
ridge/substrate pair sharing the same synthetic n+ik table, each at
520/550/580 nm and incidence 0/0.4/0.9 radians. Reference profiles clear all
tables and explicitly set the independently interpolated constants. These
cases exercise sampled substrate admittance, cache construction, packing,
Metal matching and absorption-preserving order sampling.

The outside-sandbox M5 run passed 46,128 cases in each of seven runs
(2,583,168 checked components, maximum difference 5.36442e-7), recorded in
`metal_scene_substrate_dispersion.json`. Timings were 0.319708, 0.308417,
0.440750, 0.316125 and 0.290750 ms; median 0.316125 ms, cache bytes 256,988.
No cross-run speedup is inferred: fixture mix, generated kernel and device
conditions differ, and these hot batches remain unlike full-scene rendering.
The spectra are synthetic, modal order remains N=16, and physical material
shader integration is still absent.

### Published aluminum optical-constant fixture

Retrieved the Rakić 1995 aluminum table from the CC0 refractiveindex.info
database, pinned to revision 37773dce8128ccabcd9d5108d9958d6041244661. Retained
the raw YAML, upstream license, exact source URLs and SHA-256 hashes under
`tests/scenes/diffraction/optical_constants`. Converted all 206 wavelengths
from micrometers to nanometers using decimal arithmetic, with n,k unchanged,
into `Al_Rakic_1995_nm.csv`; its hash and transformation are recorded too.
Validated increasing positive wavelengths, nonnegative n and positive k.
Coverage is 0.12399–200000 nm, encompassing the visible render domain.

The source describes reconstructed intrinsic optical constants based on
analyzed evaporated-aluminum film data (DOI 10.1364/AO.34.004755), not raw
measurements or a complete disc stack. This supplies a reproducible published
material reference; no renderer connection or aluminum render validation is
claimed at this step.

### Published aluminum data through the Metal sampler

The combined harness now loads all 206 CSV samples, validates ordering and
passive finite values, and checks the CSV hash against its pinned provenance.
Eighteen new aluminum fixtures use both ridge and substrate dispersion at
740/1600 nm pitch, 520/550/580 nm wavelength and 0/0.4/0.9-radian incidence.
Fixed-index references interpolate the imported table separately and clear
all spectra before direct scattering solves. Dataset/provenance hashes are
included in the GPU report.

The M5 outside-sandbox run passed all 64,560 samples in each of seven runs,
3,615,360 component checks, maximum difference 5.36442e-7
(`metal_scene_aluminum.json`). Five GPU batch times were 0.616167, 0.582000,
1.222120, 0.549625 and 0.568500 ms (median 0.582000 ms); cache storage is
501,360 bytes. This remains same-N=16 pipeline validation on small hot caches,
not proof of modal convergence or final aluminum-disc render accuracy.

### Explicit sampler IOR ratio for closure transport

`DiffractionSceneSample` now reports eta as outgoing/incident refractive index
for transmission and one for reflection. This matches the existing Cycles
closure convention (`bsdf_microfacet_sample`); BDPT's delta transpose uses
the supplied eta in its dielectric measure conversion. The sampler still
returns flux throughput unchanged and does not apply a second eta factor.
This is transport metadata for the future closure, not a completed BDPT
integration or proof of rough-grating reciprocity.

CPU routing/Fresnel integration checks now verify eta from both sides
(`sampler_eta_cpu.log`). The Metal test replaces its successful-result marker
with eta and checks it against the independent expected ratio for every
sample; invalid results remain all-zero and fail comparison. All 64,560
cases passed in seven GPU runs (`metal_scene_eta.json`). Blender and the
integration executable rebuilt (`build_sampler_eta.log`).

### Explicit discrete order-probability evaluation

Added `diffraction_data_order_probability` and the KernelGlobals wrapper
`diffraction_scene_order_probability`. Sampling and evaluation share one
physical power-column preparation routine. The evaluator returns unnormalized
order power and normalized discrete selection mass; absent output orders have
zero mass, whereas invalid cache/input data fail. These are discrete masses,
not solid-angle PDFs: a rough closure still needs its geometric Jacobian.

The Metal harness now compares each selected order's queried power/probability
against sampled probability and throughput, and separately queries an absent
order. All 64,560 cases passed over seven runs
(`metal_order_probability.json`), retaining independent reference checks on
the samples. This kernel now performs sampling plus two evaluations; its
1.80917 ms median cannot be compared as sampler-only cost with earlier runs.
CPU integration checks pass after the shared-code extraction
(`order_probability_cpu.log`), and Blender rebuilt
(`build_order_probability.log`). Reverse-incidence probability queries and
end-to-end BDPT use remain to be connected and validated.

### Reverse probability query implementation (GPU validation pending)

Added array-backed and KernelGlobals reverse-probability entry points. They
reverse the sampled outgoing propagation direction, use the outgoing side as
the reverse incident side, preserve the signed reciprocal-lattice order, and
recompute the reverse column normalization rather than reusing the forward
probability. CPU checks on reciprocal synthetic mirrored caches pass
(`reverse_probability_cpu.log`); Blender rebuilt
(`build_reverse_probability.log`). This is not full BDPT integration.

The first Metal test extension exited with status 2 and no diagnostic
(`metal_reverse_probability.json`). Added explicit pipeline-creation error
reporting to the host harness. A diagnostic rerun is in progress; no GPU
reverse-query pass is claimed. Prior forward-evaluation reports remain scoped
to their recorded source hashes.

The diagnostic Metal run completed with pipeline-creation failure:
`XPC_ERROR_CONNECTION_INTERRUPTED` after compiler-service retries
(`metal_reverse_probability_diagnostic.json`). The reverse fixtures use two
ports (four polarization channels), so their specialization was changed from
20 to 4 channels; forward physical fixtures retain 20-channel dispatch. That
run passed all existing 64,560 cases plus the 48 synthetic reverse queries
in each of seven runs (`metal_reverse_probability_bounded.json`). These
reverse tests cover both incident sides, +/-1 orders, mirrors and paired
unmirrored domains. They are not twenty-channel reverse GPU validation.

The successful combined test median was 3.97192 ms, with considerable spread
(5.44725 down to 3.00063 ms). It now includes additional evaluation work and
changed generated code, so it is not comparable as sampler-only performance.
The failed full-capacity combined compilation remains an integration concern;
no claim of general reverse-query GPU scalability is made.

### Avoid forced expansion of shared power-column dispatch

Changed `diffraction_data_power_column` from forced-inline to ordinary
`ccl_device`, allowing compiler outlining. Sampling, direct probability and
reverse probability retain the same implementation and capacities. Restored
the reverse fixture call to `<20>` rather than the four-channel workaround.
The previously failing combined Metal pipeline now compiles and passes all
64,560 cases in seven runs (`metal_reverse_probability_outline.json`), with
maximum component error 5.36442e-7. This resolves the observed compilation
failure for this source/configuration; it is not a guarantee against future
Metal compiler-service failures. Reverse numerical fixtures still have two
ports even though their dispatch capacity is twenty channels.

CPU integration checks pass (`probability_outline_cpu.log`), and Blender plus
the integration target rebuilt (`build_probability_outline.log`). Combined
GPU times were 1.86258, 2.54288, 2.11000, 2.18500 and 1.83671 ms, median
2.11000 ms, including sampling and explicit probability checks. No renderer
speedup is inferred from these hot-batch measurements.

### Physical aluminum reverse-column test

Added a CPU scene-owned cache test using the published Rakić table entry at
516.6 nm (n=0.87340, k=6.2418) for both ridge and substrate, pitch 740 nm,
depth 150 nm and duty 0.41. Separate small forward/reverse Bloch-domain caches
are built at N=16 with interpolation tolerance 1e-6. For 1024 stratified
samples, the actual reverse-query API must return power matching sampled
forward power within 2e-5 and a valid normalized probability. The test also
requires nonzero diffraction orders and a measurable difference between
forward/reverse normalized probabilities, preventing a trivial forward-PDF
reuse implementation from passing.

All checks pass (`physical_reverse_cpu.log`): 931 samples select nonzero
orders and 931 have normalized probability differences above 1e-4. The
integration executable rebuilt (`build_physical_reverse.log`), and prior
routing/Fresnel/storage tests also pass. This is CPU physical reverse-query
validation at one incidence and wavelength, not GPU physical reverse testing,
modal-convergence evidence, or end-to-end BDPT validation.

### Physical aluminum reverse probabilities on Metal

Extended the Metal test with a separately built reverse-Bloch cache for the
published-aluminum 740 nm-pitch fixture at 550 nm and 0.9-radian incidence.
For each of its 1024 stratified paths, the host independently solves the
reverse Maxwell scattering block, identifies the reversed incoming/output
ports, and computes both order power and its reverse-column normalization.
Those references are uploaded separately. The GPU reverse-query result must
match both within 2e-6; matching forward power alone cannot pass this check.

The full outside-sandbox M5 run passes (`metal_physical_reverse.json`):
64,560 samples in seven runs, including the 1024 physical reverse comparisons
per run and existing synthetic reverse cases. Maximum reported sample-output
difference is 5.36442e-7; reverse checks are pass/fail at 2e-6 and are not
included in that maximum. Median combined test time is 1.98288 ms. This is
physical reverse-probability validation at one material/incidence/wavelength
and the same N=16 truncation, not full BDPT or solver-convergence validation.

### Physical smooth-grating closure core

Added `DiffractionSmoothBsdf` and its own closure ID, distinct from the
provisional scalar rough-reflection model. Its sampling path transforms the
incoming propagation vector into the fixed cache frame, invokes the physical
sampler, transforms the selected order back, checks geometric/shading
hemispheres, and returns a singular reflection/transmission event with the
sampled IOR ratio. It uses Cycles' smooth-microfacet 1e6 Dirac MIS convention;
ordinary-direction evaluation is zero. Classification, sample dispatch,
roughness/eta queries and labels are wired in `bsdf.h`.

The initial build exposed missing explicit Metal pointer address spaces; these
were corrected. The final complete Blender and integration executable build
passes (`build_smooth_closure_fixed.log`). A direct CPU closure test checks
analytic reflected direction, event label, zero roughness, eta and eval/pdf
against the one-port physical-cache reference (`smooth_closure_cpu.log`).
Existing numerical and storage integration checks still pass.

This is the smooth closure core, not a user-ready shader. Allocation/node
setup and shader cache registration are not wired, no scene can select it yet,
rough-grating evaluation is absent, and the new closure has not been executed
on GPU or in an end-to-end render. Prior standalone GPU sampler validation
must not be presented as closure/render validation.

### Smooth closure allocation and wavelength setup

Added `bsdf_diffraction_smooth_setup`. It requires shader wavelength metadata
and an existing registered cache, takes the path's sampled vacuum wavelength,
checks wavelength coverage, validates indices/pitch and normalizes the normal
and projected grating tangent. It allocates through `bsdf_alloc`, applies the
existing spectral color-weight conversion, initializes the physical closure,
and marks dispersion without ordinary-direction evaluation. It returns false
for unsupported/invalid setup instead of substituting an RGB wavelength.
Shader-node compilation still must supply the cache and wavelength flag.

The CPU integration test now allocates its analytic closure through this
function and samples that allocated closure. It checks normalization,
wavelength, flags, allocation counters, missing wavelength metadata, invalid
handle, degenerate tangent and closure exhaustion. It passes
(`smooth_setup_cpu.log`). Blender and the integration target rebuild
(`build_smooth_setup.log`, `build_smooth_setup_test.log`). This is not yet
SVM/OSL node wiring or a user-renderable physical material.

### Smooth closure frame validation

Checked surface closure allocation: `bsdf_alloc` allocates separate entries;
there is no surface closure merging pass in the current kernel (the volume
closure merge is unrelated). No grating merge predicate is needed there.

Extended the CPU integration executable with 144 actual smooth-closure samples
across three orthonormal world-space frames, including a general rotation,
both incident sides, unequal indices, mirrored/unmirrored cache routing and
CDF endpoints. Analytic transmitted order directions, throughput, discrete
sampling weight, eta and singular labels are checked. Each case also reverses
the geometric normal and verifies rejection with zero evaluation and PDF.
These cases specifically test the closure's fixed-cache-frame conversion,
which the prior local-coordinate sampler tests did not exercise.

The target rebuilt and the complete integration executable passed
(`build_smooth_frame_test.log`, `smooth_frame_cpu.log`), including the existing
Fresnel, reciprocal probability and storage lifecycle checks. This run used
CPU kernel evaluation; the device name in the storage test summary does not
mean these closure samples executed on Metal. GPU closure execution and
shader-node/cache compilation wiring remain outstanding.

### Physical smooth SVM evaluation

Added `SVMNodeDiffractionSmoothBsdfData` and its `NODE_CLOSURE_BSDF` evaluation
case. The payload stores the registered cache handle, pitch and exterior
indices as constants matching the cache, plus stack offsets for normal and
groove tangent. Evaluation applies the existing closure/mix weight and derives
incident substrate side from `SR_BACKFACING`. The skip path consumes the same
payload for zero mixing and non-surface evaluation, preserving bytecode
alignment. Missing frame links use the usual shading normal and `dPdu`.

The integration executable now runs the actual SVM closure evaluator with a
serialized payload and a registered analytic cache. Linked normal/tangent,
quarter mixing, backface propagation and zero-weight skipping pass. The full
existing CPU integration suite also passes (`smooth_svm_cpu.log`). Blender
and the integration executable rebuild successfully (`build_smooth_svm.log`,
`build_smooth_svm_test.log`). This is evaluator support: shader-node compilation
does not yet emit this payload or build/register its caches, and it is not
an end-to-end material render or GPU execution test.

### Material cache request registration

`DiffractionManager::get_or_build` now builds and registers physical material
requests and reuses exact successful requests. Identity includes all profile
constants and optical-constant samples, the domain, solver/retained orders,
interpolation tolerances, construction choices and resource limits. It does
not retain progress callbacks. A callback may cancel even a reuse request;
failed/cancelled requests publish no handle. Successful `set_caches` clears
request identities; failed replacement leaves them intact. This operation is
explicitly serialized under scene update, before parallel shader compilation;
it is not a thread-safe compiler callback or a persistent on-disk cache.

The new host test verifies reuse, callback cancellation of reuse, separation
of substrate and tolerance changes, invalid material failure, and identity
lifetime across failed replacement and successful clear. Its first execution
correctly rejected non-float-representable test domain literals; the fixture
was corrected to float literals without relaxing production validation. All
57 host tests pass (`material_registry_host.log`), Blender builds
(`build_material_registry.log`) and CPU integration passes
(`material_registry_cpu.log`). Shader graph preparation still needs to call
this registration path and emit the physical SVM payload; user material and
render integration remain incomplete.

### Physical smooth OSL closure bridge

Registered `diffraction_smooth` in the OSL closure template and declared it in
`stdcycles.h`, with the same cache handle, pitch, exterior indices and frame
as the physical SVM payload. Its setup calls the shared spectral allocator
and derives incident side from runtime backfacing. For OSL layer attenuation
it queries total surviving column flux before geometric-normal clipping;
a randomly selected clipped order therefore cannot make the whole layer
appear transparent. Failed setup leaves layer albedo zero.

CPU integration validates missing wavelength metadata, allocation and an
analytic absorbing operator's quarter-weight layer albedo
(`smooth_osl_cpu.log`). The OSL declaration compiles with the bundled oslc
(`smooth_osl_declaration.log`; runtime library and absolute include paths were
required). Blender and the integration target rebuild, including generic
Metal libraries (`build_smooth_osl.log`, `build_smooth_osl_test.log`). This
checks closure setup and declaration, not execution of an OSL shader group or
a physical material render. Cache preparation must precede the parallel
shader compilation tasks; graph preparation and node emission are still not
connected. No cross-object coherent transport has been implemented.

### Internal physical shader graph node

Added `DiffractionSmoothBsdfNode`, an internal Cycles graph node with color,
normal and tangent inputs, a host physical profile/cache request and a private
prepared handle. Shader preprocessing prepares its cache before the parallel
SVM/OSL compilation stage; preparation failure is reported through `Progress`
and cancels the update. Both compilers reject an unprepared node. The SVM
compiler emits the physical payload; the OSL compiler supplies the matching
constants to `node_diffraction_smooth_bsdf.osl`. The node advertises wavelength
dependence through the existing shader metadata mechanism.

The CPU integration test constructs a real graph, connects the physical BSDF
to its surface output, runs shader preprocessing and SVM compilation, and
checks cache registration plus dispersion metadata (`physical_node_cpu.log`).
Its deliberately small angular/wavelength domain isolates compilation; it is
not a usable full-scene cache or a rendered validation. The OSL node source
compiles with oslc (`physical_node_oslc.log`). Blender and the integration
target rebuild (`build_physical_node_fixed.log`; the first build caught an
OSL compiler Progress pointer access and was corrected).

This node has no Blender UI/RNA/synchronization mapping yet. Host callers
currently configure the profile and numerical request directly and must tag
shader changes. Full-domain cache generation/convergence, user-facing node
mapping, shader-group execution and end-to-end physical render validation
remain required. The existing provisional Glossy controls still use their
original implementation, and no previous render is reclassified as physical.

### Compiled physical graph execution

Extended the real graph integration test past compilation: upload the prepared
cache, bind the compiler's bytecode to kernel globals and execute the complete
`svm_eval_nodes` interpreter. The default geometry-linked normal/tangent inputs
are evaluated and the physical closure is allocated. For the zero-depth
index-1 to index-1.5 fixture, 1000 stratified samples give exactly 40 reflection
and 960 transmission samples; each sample checks the analytic Fresnel mass,
unit surviving throughput, normal-incidence direction and transmitted eta.
The test numerically inverts the path wavelength CDF to stay inside the small
550 nm fixture cache. Wavelength metadata is supplied to ShaderData explicitly,
so full renderer metadata upload is not covered by this test.

The complete CPU integration executable passes (`graph_execute_cpu.log`), and
the target rebuilds (`build_graph_execute.log`). This is actual compiled graph
execution, but not image rendering, GPU execution or an angular/spectral-domain
accuracy claim. The test's `OBJECT_NONE` geometry exercises the derivative
tangent fallback; generated-coordinate tangents on real geometry still need
representative scene validation.

### Evaluator capacity validation at material preparation

The default kernel evaluator supports 20 polarization channels (10 ports).
Previously an internal node could register a larger host cache successfully
and then receive sampling failures at render time. Material preparation now
passes the shared `DIFFRACTION_MAX_CHANNELS` limit to registration. Both new
builds and reuse requests are checked against the maximum cell port count;
failures report required/supported channel counts and never give the shader a
handle. No ports or orders are discarded. Host reference callers can still
request unrestricted registration explicitly through the default zero limit.

The capacity test verifies a four-channel cache is rejected by a two-channel
request before mutation, accepted at four, and cannot bypass the smaller
capacity through reuse. All 58 host tests pass (`capacity_host.log`), the CPU
graph/closure integration suite passes (`capacity_cpu.log`), and Blender builds
including Metal kernels (`build_capacity_guard.log`). This makes the existing
capacity limit explicit; it does not solve larger-channel rendering or prove
full CD/DVD domain support. A larger-capacity or streamed evaluator remains
necessary for responses exceeding this limit.

### Shader metadata integration and current Metal sampler rerun

The graph integration test now uses `ShaderManager::device_update_post`, the
manager's uploaded global bytecode (including its jump table) and emitted
`KernelShader` flags. ShaderData reads the actual shader-index flag entry,
rather than manually asserting wavelength dependence. The compiled graph still
passes all 1000 analytic Fresnel samples (`graph_metadata_cpu.log`), alongside
the rest of the CPU integration suite. This covers manager metadata generation
and upload plus CPU interpreter execution, not a ray-intersection/render loop.

Reran the shared scene sampling harness outside the sandbox on Apple M5 with
the current source (`metal_current_scene_sampler.json`). Seven dispatches of
64,560 cases compare 3,615,360 output components with zero failures and maximum
component error 5.36442e-7. After two excluded warmups, measured dispatches are
4.07133, 3.75371, 1.88787, 1.88312 and 1.92363 ms (median 1.92363 ms).
The dispatch includes forward sampling, order-probability checks and selected
reverse checks, on repeatedly accessed small caches. It excludes host cache
construction, upload and rendering, and does not execute the compiled shader
graph on GPU. These are same-order solver/cache consistency fixtures, not a
new modal-convergence or rendered-performance result.

### Absorbing substrate node parameter correction

Found and corrected a real graph integration failure: the node passed the real
part of an absorbing substrate index as the kernel's lower exterior index.
For a valid purely imaginary index this was zero and caused closure setup to
reject the material. Absorbing substrates have no lower far-field ports, so
that kernel parameter is unused; both SVM and OSL now pass a positive incident
index for the absent boundary. The cache still contains the full complex
substrate admittance and absorption. Lossless substrates retain their actual
lower index and transmission behavior. No fictitious substrate port is added.

The compiled graph/metadata/interpreter fixture now covers three flat
interfaces: n=1.5, n=0.9+6i and n=0+6i. Each runs 1000 closure samples against
analytic normal-incidence Fresnel power, normalized sampling probability,
surviving throughput, direction and eta. All three and the remaining CPU
integration suite pass (`absorbing_node_cpu.log`). Blender and the test target
rebuild (`build_absorbing_node.log`). These flat fixtures validate the parameter
mapping and absorption path, not finite-depth conductor convergence, GPU graph
execution or disc renders.

### Finite-depth compiled graph order validation

Added a finite-depth case to the full shader-manager/bytecode/interpreter test:
740 nm pitch, 150 nm depth, 0.41 duty, constant n=0.9+6i ridge/substrate, air
grooves, normal incidence at the path wavelength nearest 550 nm. Preparation
uses 16 modal half-orders and a small domain at 1e-6 cache tolerance. A direct
solver query at the actual sampled wavelength supplies each reflected order's
unpolarized power; the test verifies every sampled world direction, discrete
probability, surviving throughput and eta, plus the stratified histogram.

All 4096 samples pass: counts for orders -1,0,+1 are 1859,378,1859 and total
reflected power is 0.884099 (`relief_graph_cpu.log`). Both nonzero orders are
required to receive samples. The complete integration suite passes and the
test target rebuilds (`build_relief_graph.log`). This demonstrates that a
finite-depth physical profile survives graph preparation and execution, but
uses the same N=16 solver as the cache reference. It is not an independent
Maxwell accuracy or modal-convergence claim, GPU graph execution, or an image
render. The optical constants in this fixture are generic test values, not a
claim about a measured disc coating.

### First real CPU and Metal physical-material render fixture

Added `tests/performance/cycles_diffraction_render_test.h`, invoked through the
integration executable with `--render` or `--render --metal`. It constructs a
real Cycles Session containing a flat triangle-mesh reflector, internal
physical graph node, orthographic camera, uniform white background and Combined
output pass. The 16x16 image uses 256 samples per pixel, adaptive sampling off,
a zero-depth n=0.9+6i substrate and a full 380–780 nm cache over the small
normal-incidence angular domain. A real OutputDriver reads the rendered linear
pixels; nonfinite/missing output fails. This is not a replacement shader or a
host-computed image.

CPU rendering passes (`physical_render_cpu.log`): mean RGB 0.909005, analytic
normal-incidence Fresnel reflectance 0.909114. Metal rendering outside the
sandbox on Apple M5 GPU also passes (`physical_render_metal.log`) with mean
0.909005. The CPU device description says Apple M5; the GPU run explicitly
selects DEVICE_MASK_METAL and reports Apple M5 (GPU - 10 cores). Both use the
source kernel path. The integration target builds (`build_render_fixture.log`)
and the existing numerical/graph suite still passes
(`render_fixture_regression.log`).

This is the first actual physical-node renderer smoke fixture, including
scene updates, shader metadata/bytecode, intersection, spectral closure sampling
and film output on Metal. Its 0.05 absolute smoke threshold is deliberately not
a statistical accuracy claim; agreement at the printed mean is evidence for
this one fixture only. It does not validate finite-depth diffraction images,
full angular coverage, BDPT/guiding, OSL rendering, performance overhead, or
cross-object coherence. It is a programmatically constructed Cycles scene,
not a saved Blender `.blend` scene or Blender UI integration. Representative
physical Blender scenes and multi-seed quantitative render tests remain due.

### Broadband finite-depth CPU and Metal render validation

The real Session fixture now supports `--render-relief` (optionally followed
by `--metal`): 740 nm pitch, 150 nm depth, 0.41 duty, constant n=0.9+6i ridge
and substrate, full 380–780 nm wavelength range and a small normal-incidence
angular domain. Cache interpolation tolerance is 1e-4 with N=16. It renders
256 spp at 16x16 under the same uniform white environment.

Replaced the scalar reference check with an RGB spectral quadrature: 1024
midpoints in the path wavelength CDF, direct Maxwell responses at each sampled
wavelength, D65/CIE weights divided by the wavelength density and the scene's
working-space XYZ transform. It does not read the interpolation cache.
The flat fixture uses analytic Fresnel power in the same spectral quadrature.

CPU and actual Metal GPU renders both pass, with mean RGB
(0.880499, 0.888333, 0.822580), versus direct spectral reference
(0.880586, 0.888312, 0.822620). Logs are `relief_render_cpu.log` and
`relief_render_metal.log`; Metal ran outside the sandbox with the source kernel
path. The flat spectral-reference rerun also passes
(`flat_render_spectral_reference.log`). Build: `build_relief_render.log`.

The direct reference is at the SAME modal order N=16 as the cache. This checks
spectral transport, cache interpolation and rendering consistency, not modal
convergence or an independent electromagnetic reference. The broad 0.03 RGB
smoke threshold is not a confidence interval. Multi-seed tests, retained raw
images, angularly resolved sources/screens, representative Blender scenes,
transport variants and performance measurements remain required.

### Retained physical render artifacts and reproducible inputs

The render fixture now accepts `--seed`, `--samples` and `--output PREFIX`
(with optional `--metal` in any position after the render mode). Integer
arguments are validated. Output saves unclamped RGB float PFM pixels and JSON
with seed/sample count, device type, dimensions, profile, measured/reference
RGB, reference method/order, quadrature count and transport. These are actual
OutputDriver pixels. No denoising or tone mapping is applied to the PFM.

Reran the finite-depth fixture at seed 11, 256 spp on CPU and Metal outside the
sandbox. Artifacts are in `tests/output/diffraction/physical_images/`:
`cpu_seed11.{pfm,json}` and `metal_seed11.{pfm,json}`. PFM shape, byte count,
finite values and recomputed channel means agree with the reports. Maximum
per-pixel RGB CPU/Metal difference is 1.0728836e-6; maximum channel mean error
against the same-order spectral reference is about 7.27e-5. `comparison.json`
records these checks and hashes of artifacts, binary and fixture source.
Build: `build_render_artifacts.log`; runs: `render_artifacts_cpu.log`,
`render_artifacts_metal.log`.

This is one matching-seed pair, not an independent-seed confidence estimate.
The 16x16 uniform-environment patch is a numerical image fixture; it does not
replace the requested explanatory grating/disc/interference Blender scenes.
No new `.blend` is claimed, and cross-object coherence remains unimplemented.

### Transport-selectable physical render fixture

Added `--transport pt|bdpt|guided|bdpt_guided` to the real render fixture.
The requested settings are applied to the integrator, recorded in the JSON,
and checked against the final KernelIntegrator flags. A device fallback that
disables a requested mode fails the test rather than being reported as a pass.

The finite-depth seed-11 Metal fixture passes with both BDPT and guiding
enabled (`render_transport_verified_metal.log`, artifacts
`physical_images/metal_bdpt_guided_verified_seed11.{pfm,json}`). Both actual
device flags were checked. It retains mean RGB approximately
(0.880513,0.888364,0.822597) against reference
(0.880586,0.888312,0.822620). The first run without the explicit final flag
check is separately retained as `render_transport_metal.log`; the verified
rerun used the rebuilt target (`build_transport_flags.log`).

This scene has a smooth singular grating and a directly visible environment.
It validates basic compatibility when the modes are enabled, not the
correctness of nontrivial BDPT connection weights, reverse discrete masses,
caustic connections, learning a guiding distribution or indirect transport.
The standalone bdpt/guided variants and representative indirect feature
scenes still need validation. The goal remains incomplete.

### Bidirectional singular-event audit

Inspected both camera and light path handling after the direct compatibility
render. `shade_surface.h` sets the reverse density equal to the forward density
for singular events. The light path in `bidirectional.h` likewise initializes
reverse_pdf from mis_pdf and bypasses reciprocal shader evaluation for singular
events. Both singular MIS recurrence branches reset d_vcm and multiply d_vc by
cos_out, without consuming either forward or reverse density. Consequently,
the existing independently validated diffraction reverse-order probability
helper is not integrated into these recurrences. The direct-environment test
cannot establish correctness of those missing path-strategy ratios.

A reciprocal power operator need not have equal normalized order probabilities:
for a symmetric two-port power matrix [[0.1,0.4],[0.4,0.2]], the cross-port
probabilities are 0.4/0.5 and 0.4/0.6. Their ratio is 5/6, despite equal physical
cross-port power. This is a probability example, not yet a derivation of the
complete geometric MIS correction. It identifies the normalization issue that
an indirect path-space fixture must distinguish from ordinary reciprocal
specular reflection/transmission.

Primary reference checked: PBRT 3e, Bidirectional Path Tracing,
https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing
Its implementation marks delta vertices and sets continuous forward/reverse
PDFs to zero. A discrete diffraction probability cannot be substituted into
the ordinary solid-angle evaluator. Required next work is a derivation and
validation of discrete-order strategy ratios with the grating direction-map
Jacobian, closure selection and adjoint transport. No production MIS formula
was changed based only on this audit, and no indirect BDPT correctness claim
is made.

### Fixed-order solid-angle Jacobian

Added `diffraction_grating_solid_angle_jacobian` to the shared direction
header. For one fixed order, transverse momentum gives
`o_t = (n_i/n_o) i_t + constant`. The transverse area determinant is
`(n_i/n_o)^2`; converting each projected area to solid angle using
`domega = dtx dty / abs(cos(theta))` gives
`J = (n_i/n_o)^2 abs(cos_i/cos_o)`. This applies to either outgoing hemisphere.
It is a direction-map derivative, not the discrete order-selection mass or
a continuous BSDF PDF. Zero-normal-flux endpoints are not invertible finite
maps and the helper rejects them.

New host tests perturb the incident sphere along an orthonormal tangent frame
and measure the cross product of the production direction mapper's output
derivatives. More than 100 reflection/transmission/index/order cases pass,
including more than 50 shifted-order cases. Forward and reverse Jacobians
multiply to one within 3e-6; finite-difference relative tolerance is 0.0015
away from grazing (outgoing cosine at least 0.25). Invalid index and exact
grazing endpoints are also checked. All 59 host tests pass
(`order_jacobian_host.log`), and Blender/Metal kernels plus the integration
executable rebuild (`build_order_jacobian.log`).

This establishes the local measure conversion needed by the bidirectional
work; it has not yet been wired into the singular MIS recurrence. Mixture
selection, full path-strategy ratios and indirect render verification remain
required. No claim of corrected BDPT grating MIS follows from this test alone.

### Direction-based discrete probability and closure frame bridge

Added `diffraction_data_direction_probability` plus scene and smooth-closure
wrappers. The query prepares the physical power column, reconstructs each
propagating order direction and identifies the already-sampled outgoing ray.
It returns raw power and normalized discrete mass, not a continuous density.
A 4e-6 direction-distance tolerance accommodates floating-point frame round
trips; multiple matching orders fail explicitly rather than being combined.
A nonmatching direction returns zero mass. Exact arbitrary-direction BSDF
evaluation remains zero, as required for this singular closure.

CPU tests cover all existing mirrored/unmirrored, unequal-index, two-sided
routing cases, reject off-order directions, and agree with order-index reverse
queries for the physical absorbing grating. The closure wrapper passes the 144
rotated world-frame cases. Full CPU integration passes
(`direction_probability_cpu.log`, `delta_probability_cpu.log`), and Blender
including Metal kernels rebuilds (`build_delta_probability.log`).

The Metal scene harness now additionally recovers every sampled direction's
mass and power. It passes seven runs of 64,560 cases outside the sandbox on M5,
zero failures, maximum sample-component error 5.36442e-7
(`metal_direction_probability.json`). Median dispatch is 3.29887 ms and includes
the added probability queries plus earlier forward/reverse validation; this is
not sampling-only time or render overhead. The standalone GPU test exercises
the shared data helper; the world-space closure wrapper is CPU-tested and
Metal-compiled, not separately executed in that harness.

These helpers bridge the loss of the sampled order index at the general BSDF
interface. Full reverse-shader mixture selection and the singular BDPT MIS
recurrence are not yet wired or verified by an indirect render. The matching
tolerance also needs further adversarial near-degeneracy validation before
being treated as a general-purpose delta-evaluation interface.

### Grazing reflection/transmission direction disambiguation

An adversarial check exposed a failure in the new direction query: a reflected
and transmitted order at equal exterior indices can have directions closer
than the 4e-6 matching tolerance near grazing, despite being distinct sides.
The query previously rejected the two matches as ambiguous. Added a synthetic
passive two-port fixture with 0.25 reflection and 0.25 transmission power,
normal components 1e-5, 1e-6 and 1e-7, incidence from both sides and queries to
both outgoing sides. The test demonstrably fails before the fix
(`grazing_direction_before.log`).

Direction matching now first distinguishes the sign of the outgoing normal
component. All ports still contribute to total power normalization, so each
side retains its correct probability 0.5. Exact zero-normal-flux directions
are rejected. Same-side ambiguous matches remain explicit failures. The new
fixture and all 60 host tests pass (`grazing_direction_host.log`), CPU graph
integration passes (`grazing_direction_cpu.log`), and Blender/Metal kernels
rebuild (`build_grazing_direction.log`). The new adversarial fixture is
host-executed; this turn does not claim a dedicated GPU grazing test or the
pending complete bidirectional MIS integration.

### Mixed discrete/continuous shader render fixture

Inspection confirmed that surface sampling selects closures by sample_weight
and combines sampled/evaluated closure contributions in the surface MIS path.
Added `--render-mixed`: a real MixClosure graph with weight 0.5 on the physical
finite-depth grating and 0.5 on a Lambertian diffuse BSDF of albedo 0.25. The
uniform-environment reference is integrated spectrally as
`0.5 * R_grating(lambda) + 0.5 * 0.25`. The output report records mix weight
and diffuse albedo, and retains the existing device transport-flag checks.

The CPU PT seed-11/256-spp fixture passes with RGB
(0.565417,0.569039,0.537119) versus reference
(0.565301,0.569162,0.536261). Metal outside the sandbox with BDPT and guiding
both enabled passes the smoke threshold at
(0.566053,0.569358,0.537845). Logs: `mixed_render_cpu.log`,
`mixed_render_metal.log`; raw images/reports:
`physical_images/cpu_mixed_seed11.{pfm,json}` and
`physical_images/metal_mixed_bdpt_guided_seed11.{pfm,json}`.
The target rebuilds (`build_mixed_render.log`).

The largest observed reference discrepancy is about 0.00158 on the combined
Metal run. These are different estimator configurations at one seed, not a
confidence test or evidence of zero bias. This fixture checks the interaction
of a discrete grating closure with a continuous diffuse closure and ordinary
Mix-node weighting. It still has no nontrivial indirect connection; reverse
closure selection, discrete singular path-strategy weighting and multi-seed
convergence remain outstanding. No production BDPT MIS formula changed in
this step.
## Mixed-material Metal convergence, 2026-09-26

The independent-seed batch in `tests/output/diffraction/mixed_convergence`
completed all 16 actual Metal renders: PT and BDPT with guiding, each at
256 and 4096 samples, with seeds 11, 29, 47 and 83. The manifest retains
raw PFM/report hashes and per-render means. At 4096 samples the RGB mean
errors against the direct N=16 spectral solver reference were:

| Transport | RGB mean error | RGB standard error across four seeds |
| --- | --- | --- |
| PT | (-2.84e-5, 7.97e-7, 3.38e-5) | (3.56e-5, 1.74e-5, 7.33e-5) |
| BDPT + guiding | (3.38e-5, 3.56e-5, -5.43e-5) | (7.75e-5, 5.83e-5, 1.06e-4) |

Every high-sample channel residual is below its between-seed standard error;
the larger low-sample discrepancy did not persist. This is evidence for this
uniform-environment mixed-closure fixture, not proof of zero bias, converged
Maxwell accuracy, nontrivial bidirectional connections, or learned guiding
effectiveness. Reference quadrature and modal truncation errors are not
included in these standard errors. These runs are not performance benchmarks.

## Lossless transmission render fixture, 2026-09-26

`cycles_diffraction_device_test --render-transmission` adds a real Session
render of a lossless dielectric lamellar relief: pitch 740 nm, depth 150 nm,
duty 0.41, ridge index 1.5, with air grooves and air on both sides. The
uniform white environment is preserved by combined reflection and
transmission. The reference integrates unit spectral throughput through the
renderer wavelength distribution and working-space transform; it does not
query RCWA or the interpolated cache. Equal exterior indices are deliberate:
this does not test the radiance measure across unequal media.

The first actual M5 Metal PT run (seed 11, 1024 samples) passed:
mean RGB `(1.00002146, 1.00000787, 0.999563098)` versus reference
`(1.0000695, 1.00004888, 0.999606848)`. Raw float PFM and JSON are saved as
`tests/output/diffraction/physical_images/metal_transmission_seed11.*`.
The test currently uses the same broad 0.03 smoke tolerance as the other
render fixtures. A single passing mean cannot establish separate reflection
and transmission efficiencies, outgoing angles, or convergence. This is an
internal renderer fixture, not a Blender scene or a complete transmission
validation suite.

The same seed/sample furnace subsequently passed on CPU PT and Metal BDPT
with guiding. Maximum raw RGB pixel differences were 1.431e-6 for both
CPU versus Metal PT and Metal PT versus BDPT with guiding. All three mean
errors were below 4.83e-5. `physical_images/transmission_furnace_comparison.json`
records raw image/report hashes and comparisons. This remains a single-seed
uniform-environment test.

The fixture also now accepts `--illumination reflection|transmission|both`.
The first two use a shader-graph hemisphere mask on background strength,
with the unlit hemisphere black. Their reference sums only reflected or
transmitted flux from the direct N=16 solver. They can distinguish incorrect
partitioning between reflection and transmission that a total-energy furnace
would miss, but cannot distinguish orders within either hemisphere. These
references provide cache/pipeline consistency, not independent electromagnetic
accuracy. Actual half-environment Metal validation is tracked separately from
the completed both-hemisphere furnace runs above.

The first reflection-only Metal run (seed 11, 4096 rendering samples,
1024 reference quadrature samples) produced RGB
`(0.0609528683, 0.0319558047, 0.00185930426)` versus direct reference
`(0.0605737865, 0.0317641944, 0.00205143285)`. This passes the broad smoke
tolerance, but the blue residual is about 9.4% of its small reference signal.
It is not accepted as a converged validation. The fixture now supports
`--reference-samples` independently of rendering samples, so numerical
quadrature convergence can be checked before interpreting that discrepancy
as cache error, transport bias, or Monte Carlo noise. The updated executable
built successfully and rejected zero, negative and nonnumeric reference
sample counts before rendering.

The 4096-reference-sample run then completed with reference RGB
`(0.0605737753, 0.0317641422, 0.00205146684)`. Refining from 1024 samples
changed the reference by `(-1.12e-8, -5.22e-8, 3.399e-8)`, far below the
observed residual. Render RGB stayed unchanged to 5.8e-10 in the recorded
means. Thus this quadrature refinement does not explain the discrepancy;
it does not prove modal convergence or exclude transport/cache error.
`physical_images/dielectric_reference_quadrature_comparison.json` records the
comparison and report hashes. Independent seeds 29, 47 and 83 at 4096
rendering samples are being evaluated separately from seed 11 to assess
between-seed variability.

All four reflection seeds (11, 29, 47, 83) have now completed at 4096
rendering samples. Recomputed from raw PFM pixels, their mean RGB is
`(0.06039807165, 0.03184056165, 0.001965699863)`, with between-seed standard
errors `(0.0002298516, 0.00006970817, 0.00005948257)`. Mean errors against
the direct reference are `(-0.0001757112, 0.00007636725, -0.00008573299)`.
These are within 1.5 estimated standard errors; four seeds do not prove
absence of bias or provide a high-precision validation of the weak blue signal.

The added deterministic cache diagnostic evaluates the actual scene buffers
on the host and sums all selected-hemisphere port powers at each wavelength,
without random order selection. Integrated cache RGB was
`(0.060563419, 0.0317617543, 0.00205434952)`, differing from the direct
reference by `(-1.03638e-5, -2.44010e-6, 2.91667e-6)`. Thus cache interpolation
does not explain the much larger single-seed blue residual in this fixture.
The current evidence is consistent with sampling variability; no production
sampling correction was made on this basis. This diagnostic is now included
as `integrated_cache_rgb` in render reports. It does not replace independent
electromagnetic validation, angular tests, or nontrivial transport tests.

The diagnostic was linked into a separate executable while the seed batch
ran; the batch executable's hash was checked unchanged. Both jobs completed
successfully before updating the main test executable. Raw-image statistics
and hashes are retained in
`physical_images/dielectric_reflection_four_seed_comparison.json`; diagnostic
binary/source hashes are in `physical_images/cache_reference_provenance.json`.

## Angular acceptance fixture and tangent correction

The transmission fixture now supports central, middle and outer world-X
angular regions, selected using absolute direction-component boundaries 0.25
and 0.9. Its reference independently selects orders by `abs(m * wavelength /
pitch)` at normal incidence in air. The 0.9 boundary cuts the first-order
spectrum at approximately 666 nm for the 740 nm pitch.

The first angular run exposed an error in the fixture's tangent specification.
Setting the tangent default does not create a link: graph finalization supplies
the geometry tangent for `LINK_TANGENT` sockets. The central-region Metal
render consequently failed with mean RGB `(0.899294019, 0.908777237,
0.89520669)` versus `(0.858921587, 0.849499345, 0.794503152)` reference.
The batch stopped on that failure; it did not render the other two regions.
These failed artifacts remain in `tests/output/diffraction/angular_regions`.

The fixture now connects an explicit Combine XYZ vector `(1,0,0)` to the
material Tangent input. This fixes the specified coordinate frame rather than
loosening tolerances or changing reference physics. Corrected angular runs
must be stored separately and verified before claiming validation. Hemisphere
and furnace results do not establish this coordinate-frame property.

Two additional compiled-graph integration cases now link non-unit tangents
at 45 and 90 degrees through Combine XYZ. They check the normalized tangent
after shader-manager finalization and full SVM evaluation, then check 4096
sampled world-space diffraction directions and order probabilities per case.
Both passed on CPU, alongside the existing unlinked/default-tangent case;
each obtained order counts `(1859, 378, 1859)` and total reflected power
0.884099. The complete CPU integration executable also passed its existing
Fresnel, routing, reciprocal-query, OSL-setup and storage-lifecycle checks.
Output is retained in `tests/output/diffraction/linked_tangent_graph_cpu.log`.
The executable was linked separately to preserve the running angular Metal
batch's binary hash. These tests verify graph plumbing and frame conversion,
not yet transformed Blender instances or textured tangent mappings.

The corrected angular Metal runs all completed at 4096 samples. Maximum
absolute mean RGB errors against the direct spectral references were
0.0008543 (central), 0.0012424 (middle), and 0.000004130 (outer). The outer
reference has negative green/blue working-space components; raw PFM values
were retained without clamping. Results and image hashes are in
`tests/output/diffraction/angular_regions_linked/verified_comparison.json`.
The outer run was interrupted before writing an image and was rerun only
after confirming its process was absent; completed central/middle outputs
were retained. These remain single-seed angular acceptance tests, not a
convergence result or a finite-aperture wave-optics screen.

## Blender-facing physical node integration (in progress)

`ShaderNodeBsdfDiffraction` now has DNA storage, RNA properties, node
declaration/registration, a Cycles add-menu entry, and Blender-to-Cycles graph
mapping to the physical smooth closure. Uniform profile parameters are pitch,
depth, duty cycle, upper/groove indices, and complex ridge/substrate indices.
Color, normal and across-groove tangent are shader inputs. The node is offered
for Cycles, and does not substitute an unrelated Eevee material model.
The full Blender build and actual node runtime checks are still pending.

`tests/python/cycles_physical_diffraction_scene.py` prepares real planar
Blender-node fixtures for flat conductor, relief conductor, and relief
dielectric materials. It saves/reopens `.blend` data and verifies profile
properties and the tangent link, with optional EXR rendering. This script
has passed Python syntax checking only so far; no completed physical `.blend`
scene or Blender render is claimed yet. Defaults use constant optical indices,
not the published aluminum table, and no coherent-transport option is exposed.

The full Blender build succeeded. An initial runtime check found a missing
entry in this branch's explicit shader registration list; adding the
declaration and call in `node_shader_register.hh/.cc` corrected it, followed
by a successful rebuild. Flat-conductor, relief-conductor and relief-dielectric
`.blend` files have now been created and reopened successfully. All nine
profile properties and the explicit tangent link survived serialization.
Artifacts are under `tests/output/diffraction/blender_physical`; the reports
remain `saved_not_rendered` until rendering and numerical checks complete.
The flat-conductor Metal Blender render is in progress. These scene files
use the physical node, unlike the older provisional Glossy suite.

The flat-conductor Blender render then completed with Metal selected and
CPU rendering disabled in device preferences. At 128 by 128 pixels and 1024
samples, reloaded raw EXR mean RGB was `(0.9091735979, 0.9091583210,
0.9087455517)` versus neutral analytic Fresnel reflectance `0.9091138623`.
Maximum mean error was 0.00036831, below the explicit 0.005 smoke tolerance.
The report records `analytic_smoke_pass` and hashes the `.blend` and EXR.
This is a zero-depth pipeline control, not finite-depth diffraction validation.
The finite-depth conductor Blender render is being tested next with Cycles
device diagnostics enabled.

The finite-depth Blender render failed before producing an image: full-domain
cache preparation reached its maximum subdivision depth without meeting
the 0.001 interpolation tolerance. This is a real limitation of the current
Blender-facing integration; the small normal-incidence internal fixtures do
not establish full-domain readiness. The failing log is
`tests/output/diffraction/blender_physical_render_relief_metal.log`.
No tolerance was loosened and no partial cache was published. Investigation
now uses the full-cache probe to retain the failing cell bounds, error and
subdivision statistics for both DVD and CD pitches before changing cache
construction or its budgets.

The current full-cache probe reproduced the DVD failure after 851 visited
nodes: error 0.0010040141 at depth 24 in the cell bounded by
`[-0.0625, -0.99658203125, 380]` and `[0, -0.99609375, 381.5625]`.
This is near grazing incidence. Reviewing the completed earlier whole-domain
audits showed that the new Blender adapter had not enabled their quadratic
cells, mirror symmetry or depth-36 configuration. That tested configuration
is now wired into the adapter, with validation workers capped at six and
respecting the task scheduler's thread limit. The power tolerance remains
0.001 and the domain is unchanged. The rebuilt Blender is rerendering the
relief scene into `blender_physical/relief_metal_quadratic`; this is a new
validation run, not a relabeling of the failed linear-cache result.

That quadratic-cache Blender run completed on the Apple M5 Metal device.
The raw mean RGB is `(0.8805794602, 0.8883166067, 0.8226021715)`;
the maximum difference from the internal N=16 direct-solver reference is
0.000017684. `pipeline_comparison.json` records the report hash and device
log evidence. Rendering excluding synchronization took 1.05254 seconds, while
total observed time was 328.089 seconds. This was a single diagnostic run
with an overlapping CPU probe, not an isolated performance benchmark.
Preparation dominates this small image and still needs improvement.

The adapter now derives a conservative external reference-order window from
the complete wavelength/Bloch domain, exterior indices and cutoff margin.
This leaves the internal RCWA modal count at N=16 and retains every potentially
propagating or near-cutoff exterior channel. For the conductor DVD/CD profiles
the symmetric reference bounds are 2/4 instead of 16; a lossless index-1.5
substrate needs 3/6. Omitted exterior channels keep the existing exact outgoing
boundary condition. Invalid domains or insufficient internal modal count
fail explicitly. New tests check the first excluded order across the domain
and compare every physical operator entry against the full reference window
at 72 material/angle/wavelength combinations, within 1e-9. All 62 host tests
passed. The rebuilt Blender is testing this optimization in a separate output
directory, `blender_physical/relief_metal_bounded_reference`.

Three saved dielectric transmission angular controls now complement the
internal render tests: central, middle and outer acceptance, under
`blender_physical/angular_controls`. Save/reload checks passed, and each file
contains explanatory measurement text. These files are not yet rendered.
The scene generator excludes masked illumination from the unmasked analytic
furnace check rather than comparing a different physical experiment to unity.

The bounded-reference Blender Metal run completed successfully. The actual
cache contains 3519 cells and 18,066,688 matrix bytes; its sampled accepted-cell
power error is 0.000998666. Preparation took 205.219 seconds and rendering
excluding synchronization took 1.00776 seconds (206.681 seconds total observed).
Compared with the previous retained-16 reference-window image, the maximum
raw pixel-component difference is 3.57628e-7 and maximum RGB mean difference
is 8.73115e-11. The comparison hashes both EXRs. The unchanged internal N=16
solver and full angular/wavelength domain are preserved. Concurrent CPU work
means the observed reduction from 328 seconds is not an isolated benchmark.
The current device/scene integration executable also passed on CPU, including
its intentional error-recovery test; that command is not a GPU sampling test.

`tests/python/cycles_physical_diffraction_discs.py` now creates a separate
physical-node presentation fixture with CD/DVD pitches, 120 mm disc geometry,
radial object-to-world tangent links and white strip illumination. It reuses
the existing studio geometry while replacing all assigned provisional grating
closures and removing the unused provisional template. Save/reload passed.
The profile is smooth 150 nm lamellar relief, duty 0.41, constant conductor
index 0.9+6i: not recorded pits, measured aluminum, a covered disc, or coherent
transport. A 640-pixel, 256-sample actual Metal pilot is running in
`tests/output/diffraction/physical_discs_metal_pilot`; visual/numerical review
remains pending. The generator preserves linear EXR, display PNG, metadata,
and source/binary/image hashes, and does not label a completed render validated.

### Adaptive-coordinate numerical experiments (not integrated)

Standalone generated solver copies in `tests/output/diffraction/asr_experiment`
exercise the covariant Maxwell matrices from Messina et al., arXiv:1612.05516,
with a centered three-interval coordinate map, minimum slope 0.001 and 2048
midpoint quadrature samples. The production solver and running disc render
are unchanged. `provenance.json` hashes the source copies and production source.

Projecting both electric and magnetic fields with the Cartesian transform
gave promising conductor comparisons but failed lossless passivity. Across
16 seeded queries per pitch, ASR16 versus ordinary N128 had maximum power
column L1 differences 0.000630 (740 nm) and 0.01583 (1600 nm). The corresponding
lossless dielectric test against ordinary N64 reached coherent-channel power
gain 1.09269 for the CD pitch. This is a rejected finite-basis boundary scheme,
not evidence of a usable acceleration.

A second experiment uses a flux-dual magnetic projection: for electric
blocks Tx and Ty, magnetic blocks are inverse(Ty-adjoint) and
inverse(Tx-adjoint). This preserves the discrete tangential Poynting pairing.
The tested dielectric maximum gain fell to 1.00000000004, but the conductor
ASR32 versus ordinary N128 maximum power-column differences were 0.01333 and
0.03096. Passing conservation alone therefore does not justify integration.
Successive adaptive-order comparisons are being measured separately. Ordinary
N128 is itself not an exact reference, and neither experiment has passed the
required convergence, identity-map, reciprocity or whole-domain accuracy gates.

The flux-dual ASR32-to-ASR64 run then failed the existing interface-system
residual check before producing complete statistics. Its exit status and
error are retained in `dual_metal32_asr64_status.json`; no residual tolerance
was relaxed. The dual variant therefore also remains rejected for production.

### Physical disc pilot inspected

The physical-node 640-pixel, 256-sample Metal disc pilot completed and its PNG
was visually inspected. Radial spectral bands are visible. The single strip
source reveals mostly violet on the DVD, so it does not adequately present
the full wavelength response. Both discs use the same light. The image is
retained as a diagnostic, not a validated manufactured-disc reference.
The actual Metal trace records cache preparation of 245.765 seconds (CD) and
209.296 seconds (DVD), render time excluding synchronization 8.8741 seconds,
and total render time 482.171 seconds. This includes overlapping CPU work and
is not an isolated performance benchmark.

The generator now has `--lighting pilot|broad`. Broad adds an otherwise
identical white strip at the opposite Y azimuth, improving angular coverage
without changing material efficiency or adding artificial spectral colors.
The original lighting remains reproducible with `--lighting pilot`. A separate
960-pixel, 1024-sample Metal render is running in `physical_discs_metal_broad`;
its appearance and numerical validation remain pending.

### Weak boundary projection and exterior-basis refinement

The standalone ASR boundary experiment was reformulated without explicitly
inverting the electric projection blocks. With T=diag(Tx,Ty), let
U=diag(Ty-adjoint,Tx-adjoint). The boundary equation uses U Y T on the
internal electric field and U on the Cartesian incident magnetic field.
The identity J U = T-adjoint J preserves the discrete Poynting pairing,
where J has off-diagonal blocks I and -I. This is algebraically equivalent
to the earlier flux-dual scheme when square transforms are invertible, but
avoids those inverses and also permits a rectangular exterior basis.

The equal-size formulation completed ASR32-to-ASR64 without the earlier
residual failure. Its maximum power-column L1 differences over the same 16
seeded queries were still 0.00818/0.01941 (DVD/CD). Separately increasing the
Cartesian exterior half-order window to twice the internal one reduced the
ASR32-versus-ordinary128 maxima to 0.00392/0.01060. Four times the exterior
window reduced them further to 0.001624/0.003057. Ordinary128 is not an exact
reference, so this trend is evidence for further investigation, not a final
error bound. The doubled-exterior dielectric ASR16-versus-ordinary64 run
retained maximum coherent gain within 4e-11 of unity in these samples.

All variants remain standalone source copies in `asr_experiment`; none
changes the production renderer. A recurrence-based evaluation of the same
projection quadrature is being compared with the direct exponential sums
to reduce experimental preparation cost without changing the equations.

The direct-exponential and recurrence variants completed the same 16-query
ASR32/ordinary128 comparison. Their reported power/complex error statistics
agree within 2.1e-14. This is a comparison of summary statistics, not a saved
per-entry matrix comparison or isolated timing result. A targeted comparison
against ordinary N512 is running at the worst sampled CD query to assess
reference truncation separately.

The earlier bounded linear full-domain probe has also finished. DVD failed
at depth 24 as recorded above; CD exhausted 65,535 nodes after covering only
0.0436311 of the domain (1,212,514 solves, 2611.22 seconds with concurrent
work). Neither partial cache was returned or exported. This further confirms
that the original linear adapter settings were unsuitable; the completed
physical Blender renders use the previously audited quadratic configuration.

The targeted ASR32/fourfold-exterior comparison against ordinary N512 completed
at (Bloch 0.138289589824, ky -0.439921165636, wavelength 413.547736616 nm).
Power-column L1 differences were 0.00171887 (DVD pitch) and 0.00340562 (CD
pitch), with complex-entry maxima 0.00193373/0.00271658. It took about 49/52
seconds per profile including the high-order solve. This does not meet the
desired accuracy and is not a validated production replacement.

The broad-lighting physical disc render completed on the actual Apple M5
Metal device at 960 pixels and 1024 samples. Its displayed PNG was inspected:
the added white strip reveals extra red/green DVD bands while retaining the
original source's violet band. This improves the explanatory scene; it does
not resolve physical roughness, actual pit geometry, or modal accuracy.
Preparation took 247.842 seconds (CD) and 210.073 seconds (DVD). Rendering
excluding synchronization took 77.4389 seconds; total observed time was
550.18 seconds. Concurrent numerical work means this is not an isolated
benchmark. The saved physical dielectric furnace is now rendering separately
in `blender_physical/dielectric_metal_furnace` against the analytic unit-radiance
reference.

### Saved physical indirect-light transport fixture

The presentation generator now also supports `--case indirect`. It reuses
the achromatic room, restricted white source, occluder and sphere, replaces
the assigned provisional grating with the physical smooth node, and creates
an explicit +X tangent link. Metadata removes the old roughness claim. PT,
BDPT, guided PT and guided BDPT scene variants all passed save/reload and
transport-setting checks in `physical_indirect_saved`. No render-convergence
claim follows from those checks.

`tests/python/cycles_physical_diffraction_transport.py` runs a saved physical
scene across matched transport variants and independent seeds within one
Blender process. It enables persistent scene data to permit material-cache
reuse, which must still be verified in the actual logs. It saves per-run
configurations, unclamped EXR, bottom-left-origin float NumPy RGB, display PNG,
hashes and observed times. The manifest is updated after each completed run,
records failures, verifies the Blender binary and Cycles source fingerprint
at batch boundaries, and leaves convergence analysis pending. Settings-only
validation passed for all four modes and seeds 11/29/47/83.

The first actual Metal-configured batch is running in
`physical_indirect_transport_metal`: four modes, four seeds, 512 samples,
320 by 272 pixels. It overlaps the physical dielectric control's cache
preparation, so it is a correctness experiment, not an isolated performance
benchmark. Receiver-image comparison and uncertainty analysis remain pending.

The batch-analysis tool `cycles_diffraction_transport_analysis.py` verifies
completed manifests, artifact hashes, array shapes and raw means before
calculating matched-seed differences and standard errors. It stores pixel
difference/standard-error maps, but does not treat pixels as independent
observations or automatically label agreement a convergence proof. Checks
passed for a known four-seed standard error, invariance to duplicated pixels,
and rejection of the still-running incomplete batch.

The first PT/BDPT/guided seed-11 images were inspected and are nearly black
with sparse bright samples. PT and BDPT raw pixels are bit-identical. This
does not validate bidirectional transport; it may indicate missing light-path
contributions or a batch/update problem and needs isolation with controls.
Saved-scene inspection confirms reflective/refractive caustics enabled, BDPT
enabled, eight light bounces, GPU selected, source maximum bounces 1024, and
ordinary camera/diffuse/glossy/shadow visibility for all room objects. Logs
show BDPT buffers and a light-cache allocation, which is not evidence of
nonzero generated vertices or successful connections. The first-seed hashes
and differences are retained in `first_seed_diagnostic.json`; the full batch
continues unchanged so the independent-seed behavior can be inspected.

### Indirect emitter diagnostic resolved

The 16-render batch completed, passed binary/source and artifact verification,
and produced paired-seed statistics. All four PT/BDPT pairs were bit-identical.
The log has one material-cache build and 16 completed renders, confirming
persistent reuse. Guided variants differed; four low-sample, heavy-tailed
caustic observations do not establish their convergence.

The scene's finite-radius spotlight had `use_soft_falloff=True`. In this fork
that maps to a receiver-facing disk approximation (`is_sphere=false`), and
`photon_sample_emitter` explicitly excludes that emitter from forward light
generation because it has no fixed reciprocal emitting surface. BDPT retains
the camera-path fallback for it. The saved-file property was checked directly.
Thus the identical output is explained by the emitter choice, not evidence
that physical grating light-path transport works or fails.

A separate diffuse-room control reproduced near-identical PT/BDPT results
with that emitter. Changing only to a physical sphere emitter made PT/BDPT
distinct: at 160 by 136 pixels and 64 samples, mean neutral RGB was about
0.092096/0.094039 and maximum pixel difference 0.06910. This single-seed
control isolates the fallback; it is not a convergence or unbiasedness proof.
Both controls and their saved scenes/raw images remain available.

The physical indirect generator now defaults to `--emitter sphere`, explicitly
sets and verifies the saved light property, and records the emitter model.
The compatibility setting remains reproducible. All four corrected variants
passed save/reload in `physical_indirect_sphere`; material custom metadata now
also describes the actual physical profile. The new four-mode/four-seed Metal
batch is running in `physical_indirect_sphere_transport`. Its manifest now
records emitter settings, caustic controls, and BDPT path/update counts.

### Receiver-specific measurements

`cycles_diffraction_receiver_masks.py` casts center camera rays through the
saved scene to identify primary visible objects. It generates hashed masks
for each room receiver, the tile, occluder and sphere, with a configurable
exclusion border (three pixels by default). The masks are static geometric
selections, not exact stochastic-filter, motion-blur or DOF reconstructions.
The first label image was visually inspected against the room geometry;
non-overlap checks passed. The three-pixel masks for the corrected sphere
fixture are in `physical_indirect_sphere_masks_border3`.

The batch analyzer accepts `--masks`, verifies the input-scene hash, mask
hash, dimensions, origin and pixel counts, and computes paired-seed receiver
differences separately from whole-image differences. This prevents unaffected
parts of the frame from concealing receiver-specific discrepancies. The live
corrected batch remains incomplete, so no partial convergence statistics have
been published for it.


### Completed indirect batch and isolated MIS diagnostic

The corrected physical sphere-emitter batch completed all 16 Metal renders;
artifact provenance and three-pixel receiver masks passed analyzer verification.
BDPT minus PT mean RGB is (0.08023189, 0.08044470, 0.08184576), paired-seed SE
(0.00187383, 0.00027010, 0.00226214). This establishes estimator disagreement,
not an absolute physical error or a requirement to match finite-sample PT.
A built-in smooth mirror in the same room also reproduces higher BDPT
brightness: single-seed 64-sample PT mean 0.06228346 versus BDPT 0.12801158.
The isolated `/tmp/cycles-diffraction-mis-candidate` removes the singular-event
MIS_SKIP early return in the three emission MIS functions, retaining the
primary-ray guard. Its Metal render completed with unchanged PT and BDPT mean
0.06572838. This supports the suspected loss of prior connectible-vertex MIS
alternatives, but is not a convergence pass or a production change. Candidate
source hashes and outputs are preserved in `mis_candidate.json` and
`indirect_mirror_mis_candidate`.

The full-domain dielectric Blender furnace failed before rendering after
2055.540 seconds of cache preparation and exhaustion of the 65,535-node budget.
Tolerance remains 0.001; no partial cache was used. This remains an unresolved
cache scalability failure, separately from the indirect transport discrepancy.


### Multi-order delta MIS coverage audit

The isolated correction is now undergoing the same four-seed, four-mode
512-sample physical-room batch in `physical_indirect_mis_candidate`, using the
copied kernel source tree; the production kernel and binary are unchanged.
The control generator also accepts explicit seed, sample count and resolution,
so the mirror diagnostic can be repeated with independent renders.

Source inspection found a separate coverage gap: the full-product oracle in
`kernel_bidirectional_pdf_test.cpp` assigns unit forward and reverse probability
to every delta event and constructs mirror-symmetric normals there. It does
not validate a grating with multiple discrete orders, nonunit order selection
probabilities or unequal incident/outgoing cosines. Both camera and light
singular recurrences currently clear d_vcm and multiply d_vc by the outgoing
cosine without consulting order probabilities. The physical closure's known
order probability query and direction-map Jacobian are available but not wired
into this recurrence. This audit is not by itself proof of the correct replacement:
delta constraints, their measure conversion and sampling probabilities must be
derived together before any additional production change.

Relevant primary references are Veach's thesis chapter 10
(https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf) and PBRT's
bidirectional path tracing treatment
(https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing).
In particular, skipping deterministic connections at delta vertices does not
justify discarding all competing strategies at earlier connectible vertices.


The executable `tests/performance/cycles_diffraction_delta_mis_audit.py`
enumerates complete sampling-strategy probabilities for a diffuse/delta/diffuse
chain in a discrete unit-geometry model. Across balance and power heuristics,
unequal forward/reverse branch masses disagree with the unit-delta recurrence;
retaining the reverse/forward ratio matches the enumerated endpoint weight.
The equal-mass control also agrees. Results are retained in
`tests/output/diffraction/delta_mis_discrete_audit.json`. This is a bounded
mathematical diagnostic, not a production regression or proof for a geometric
grating: constraint Jacobians, wavelength and polarization remain outside it.


The discrete audit now checks all admissible split strategies in 1,024
reproducible random chains of 3–10 edges, with balance/power weighting and
multiple adjacent delta events. The branch-ratio recurrence agrees with
independently enumerated full-product weights to maximum absolute error
3.33e-16; every partition sums to one. Production kernels and geometric
constraint measures are still explicitly outside this diagnostic's scope.


The transport analyzer now accepts `--baseline` for verified candidate/baseline
comparisons. Both complete batches undergo artifact verification; input scene,
binary, runner script, seeds, samples, emitter and transport controls must match.
It records changed kernel fingerprints, paired-seed changes per mode and raw
array identity by seed. Baseline self-comparison produced exactly zero change
and identical arrays; an unfinished candidate batch was correctly rejected.


### Completed isolated MIS candidate batch

All 16 physical-room Metal renders completed, and the baseline/mask/artifact
analyzer passed. Candidate BDPT minus PT mean RGB is
(0.00573846, 0.00742331, 0.00603947), with paired-seed SE
(0.00771710, 0.00669989, 0.00942128). The original excess was approximately
0.080 per channel. All four unguided PT arrays are byte-identical to baseline.
Guided arrays are not identical across these runs; their changes must not be
attributed solely to the emission fix without investigating guiding determinism.
Candidate guided minus PT green difference is -0.00698299 with SE 0.00138274.
Receiver-specific residuals also remain: ceiling green -0.0686452 (SE 0.0161058)
and occluder red +0.142689 (SE 0.0415164). These four-seed observations support
removing the emitter MIS_SKIP guard but do not establish full unbiasedness.
See `physical_indirect_mis_candidate/verified_comparison.json`.

The first dielectric curvature probe was cancelled (exit 130) after detecting
an unmatched cutoff margin. The replacement `dielectric_curvature_matched_probe`
uses Blender's cutoff 0.1 and 64 MiB reference cache, tolerance 0.001, N=16,
quadratic/mirror cells, two validation workers and a bounded 8,191-node budget.
Its source/binary hashes and exact settings are in the adjacent provenance JSON.
This is a splitting diagnostic, not a full-domain readiness test or benchmark.


The tested emitter MIS correction was promoted verbatim to
`kernel/integrator/bidirectional.h` after the completed physical-room batch.
It removes the MIS_SKIP early return in infinite, surface and lamp emission
weights while retaining primary-ray handling. Blender and Metal kernels rebuilt
successfully; the CPU device/integration test exited zero. Its intentional
error-recovery diagnostic is expected. A four-seed, 512-sample, 320-pixel mirror
control batch on the rebuilt binary is running under `mirror_mis_production_*`.
This production edit fixes the observed lost-alternative bug; it does not claim
to resolve multi-order delta recurrence, guiding or receiver residuals.


### Larger-source convergence fixture and matched cache probes

The rebuilt-binary mirror control completed all four seeds at 512 spp and
320x272. Artifact hashes and recorded source/binary hashes verified. PT mean
is 0.06066658 and BDPT 0.06882982; paired difference 0.00816324 with SE
0.00442869 in every channel. This remains insufficient evidence of convergence.

The separate 50 mm physical sphere-source grating room passed save/reload and
has matching three-pixel receiver masks. Its four-seed PT/BDPT/guided/BDPT-guided
Metal batch is running at 1024 spp in `physical_indirect_50mm_transport`.
This complements the retained 5 mm stress fixture; changing source size is not
an estimator comparison against the original scene.

A response-split baseline now runs with exactly the bounded curvature probe's
material, cutoff, tolerances, node/memory budgets and worker count, differing
only in `curvature_adaptive_splits=false`. Outputs and provenance are under
`dielectric_response_matched_probe`. All concurrent timings are observational,
not isolated performance benchmarks. Production kernel sources remain frozen
for the live Metal batch's provenance verification.


### Guiding reset source audit

`PathTrace::init_render_buffers` calls `zero_render_buffers` when initialization
is scheduled. The GPU implementation marks `guiding_reset_pending_`, and
`prepare_gpu_guiding` clears the node, accumulation, sampling and fitting
buffers plus trained-sample count on that flag. History counters are reset
per batch separately. This is source evidence against simple persistent-field
carry-over, not a runtime demonstration of reset correctness or determinism.
The changed guided images and residual means remain unresolved; they must not
be treated as an effect attributable solely to the emitter MIS edit.


The initial 50 mm fixture was rejected: its original z=0.48 source center
put part of the physical sphere above the z=0.5 ceiling. Its batch was
cancelled, and no result is accepted. The corrected generator lowers the
source to z=0.44 (10 mm ceiling clearance), re-aims at the tile, records its
position and verifies clearance after reload. The replacement scene is
`physical_indirect_sphere_50mm_clear`; its fresh Metal batch and hashed masks
are `physical_indirect_50mm_clear_transport` and
`physical_indirect_50mm_clear_masks`. Save/reload and mask generation passed.


The matched curvature probe completed its 740 nm case unsuccessfully:
8,191 nodes, 386,349 reference solves, accepted-domain fraction 0.00946981,
no exported/returned cache. Its last accepted cell was at depth 30; stopping
was the node budget, not the maximum-depth criterion. The 480.66-second time
includes concurrent workloads. The 1600 nm case and matched response baseline
are still running. The completed case is preserved separately as
`dielectric_curvature_740_completed_case.json`; it does not justify switching
Blender's default heuristic.


### Resonance-aware cache research follow-up

Primary references identified for the dielectric cache scaling problem:
Nakatsukasa, Sete and Trefethen, *The AAA algorithm for rational approximation*
(https://arxiv.org/abs/1612.00337); Betz et al., *Efficient rational approximation
of optical response functions with the AAA algorithm*
(https://arxiv.org/abs/2403.19404); Fischbach et al., *Pole-Expansion of the
T-Matrix Based on a Matrix-Valued AAA-Algorithm*
(https://arxiv.org/html/2602.18414v1).
The latter jointly approximates matrix entries using shared poles, avoiding
separate element-wise pole sets and redundant storage. It is a candidate for
an offline experiment on fixed-topology grating reference operators, not an
implemented cache replacement. Its frequency-domain demonstrations do not
establish accuracy over this cache's two angular coordinates, exterior cutoffs,
or guarantee that an interpolated grating preserves passivity. Required gates
remain direct held-out physical-power and complex-response checks, reciprocity,
conservation, cutoff robustness and eventual GPU evaluation cost. No tolerance
or acceptance rule changes follow from these papers alone.


### First actual-reference rational interpolation experiment

`cycles_diffraction_rational_samples.cpp` exports 161 N16 dielectric reference
operators over 380–400 nm at Bloch=-0.0517578125, ky=-0.946533203125,
retained half-orders=2, pitch=740 nm, depth=150 nm and duty=.41. Alternating
samples separate fitting and held-out sets. These are artificial reference
operators, not physical BSDF matrices or an independent modal reference.
`cycles_diffraction_rational_fit.py` implements a shared-denominator barycentric
AAA fit with a stacked complex Loewner SVD, checks an analytic shared-pole
control, and rejects ill-conditioned evaluations. Six support points fit the
81 training matrices with max Frobenius error 3.3013e-10 and the 80 held-out
matrices with error 3.3042e-10. Data and report are `rational_dielectric_samples.json`
and `rational_dielectric_fit.json`. This single wavelength slice may be easier
than the unresolved angular variation; no production integration, physical-power
validation, passivity guarantee or 3D performance claim follows from it.


The rational sampler now independently sweeps wavelength 380–400 nm, Bloch
-0.1–0, and ky -0.98–-0.90 around the difficult region (fixed wavelength 384.8 nm
for angular slices). Each uses 81 fit and 80 held-out matrices. Six supports
suffice in all cases: held-out Frobenius errors are 3.304e-10, 1.180e-9 and
8.741e-9 respectively. Degree-five Chebyshev least-squares fits using all 81
training samples yield 1.646e-8, 1.227e-9 and 1.744e-7. This is a representation
comparison, not equal solver-sample acquisition cost. The smooth reference
operators suggest investigating amplification during exterior matching; direct
physical-power checks are required before interpreting these small errors.
Reports are `rational_dielectric_{wavelength,bloch,ky}_fit.json`.


`cycles_diffraction_rational_power.cpp` now re-solves every held-out query,
replaces its reference matrix with the fitted prediction, performs actual
exterior matching, verifies port identities and measures physical unpolarized
power-column L1 error and lossless column sums. For 80 queries on each slice,
maximum power errors are 9.208e-10 (wavelength), 2.544e-8 (Bloch) and
5.519e-8 (ky); maximum conservation residuals are 9.183e-10, 2.544e-8 and
5.302e-8. Direct N16 conservation residuals are at most 1.14e-13. These are
local same-solver accuracy checks, not exact passivity enforcement, arbitrary
polarization checks, independent modal accuracy or full-domain coverage.
Predictions and reports are retained as `rational_dielectric_*_predictions.txt`
and `rational_dielectric_*_power.json`.


The offline rational evaluator now supports `--precision float32`, quantizing
nodes, complex weights, matrix samples and evaluation coordinates before
barycentric arithmetic. Re-running the three held-out slices yields physical
power-column errors 7.207e-6 (wavelength), 1.351e-5 (Bloch), 1.143e-5 (ky).
Conservation residuals are of the same magnitude. This is host NumPy complex64
evaluation followed by double-precision exterior matching, not an actual Metal
kernel or full single-precision pipeline test. Its errors remain below the
current 0.001 interpolation tolerance locally, without proving exact passivity.
Reports and predictions have `_float32_` in their filenames.


### Coherent-channel gain check of rational predictions

The held-out matcher now reports minimum/maximum eigenvalues of S* S using
the physical flux-normalized scattering matrix, bounding power for arbitrary
coherent incident channel combinations. This is a stronger check than separate
unpolarized columns, but is not cross-object coherent rendering.
Float32 fits show maximum gains 1.00002637 (wavelength), 1.00003927 (Bloch),
1.00004131 (ky); minimum gains are 0.99998181, 0.99994993, 0.99999005.
Float64 fits also have small nonunitarity (worst deviation about 2.21e-7),
whereas the direct reference is unitary within 9.6e-12. Thus the unconstrained
rational fit is not strictly passive even locally. No normalization/clipping
has been applied to conceal this. A structure-preserving representation and
full-domain validation are prerequisites for production/coherent use.
Detailed reports are `rational_dielectric_*_gain.json`.


### Lossless structure-preserving rational experiment

`cycles_diffraction_rational_unitary.py` checks input unitarity, selects one
well-conditioned phase-rotated Cayley chart using training samples only, fits
Hermitian Cayley matrices with real shared rational weights, and reconstructs
unitary reference operators algebraically. It records/rejects non-Hermitian
roundoff correction above 1e-8; observed projection norms are at most 5e-12.
No physical power clipping is used. Wavelength/Bloch/ky slices use 6/7/8
supports. Held-out physical-power column errors are 2.745e-9, 6.117e-11 and
1.183e-8. Matched coherent power gains differ from one by at most 7.1e-14.
This double-precision lossless-only result is local, not a full-domain cache,
absorbing-material solution or Metal implementation. Reports and predictions
are `rational_unitary_*`. Single-precision reconstruction and 3D extension
remain necessary before considering production changes.


Single-precision lossless Cayley evaluation/reconstruction was tested on the
same held-out slices. Maximum physical-power column errors are 3.662e-6,
3.888e-6 and 2.794e-5 for wavelength/Bloch/ky. Maximum coherent gains are
1.000000193, 1.000000055 and 1.000000295; minimum gains are 0.999986015,
0.999985485 and 0.999991896. Thus finite-precision effects remain, though
positive gain is much smaller than the unconstrained fit. This uses NumPy
complex64 reconstruction followed by host double-precision exterior matching,
not actual Metal evaluation. Reports are `rational_unitary_*_float32_gain.json`.

The bounded curvature probe completed both pitches unsuccessfully (740 nm:
node budget; 1600 nm: matrix-memory budget):
accepted-domain fractions 0.00946981 (740 nm) and 0.02970123 (1600 nm), no
usable cache. Source/binary hashes were rechecked and its provenance status
updated. Successful process exit here means the diagnostic ran, not cache
construction success. The matched response probe remains in progress.


### Spectral BDPT connection audit

Source inspection found that `bdpt_light_vertex_spectral_weight` uses
`photon_spectral_kernel` when both the cached light subpath and camera subpath
are spectral. The helper is a finite-bandwidth Epanechnikov wavelength
reconstruction (bandwidth 0.02 in the wavelength units used by the sampler),
not exact same-wavelength subpath conditioning. This existing approximation
must be explicitly tested for narrow diffraction spectra; increasing render
samples alone does not eliminate a fixed reconstruction bandwidth. No source
change was made during the provenance-frozen Metal batch. This issue is
separate from the corrected emitter MIS guard and the multi-order delta mass
recurrence. The 50 mm seed-11 PT/BDPT previews were visually reviewed: indirect
patterns are now visible in PT, but both remain noisy and are not references.


`cycles_diffraction_spectral_connection_audit.py` integrates the existing
20 nm half-width Epanechnikov reconstruction after cancellation of the sampled
wavelength density with its inverse-density factor. It verifies quadrature
refinement and constant-response controls. At 380/780 nm, truncated support
retains only 0.5 of a constant response. At 550 nm, unit-peak Gaussians with
sigma 1/5/20 nm reconstruct as 0.0937636/0.440622/0.909796. This is an analytic
kernel diagnostic, not a claim that a particular rendered grating has those
errors. It establishes a fixed-bandwidth spectral approximation needing
replacement or explicit convergence treatment for exact spectral BDPT.
`integrator_bdpt_light_generate` samples wavelength from a light-path-index
seed and iteration; the camera uses its pixel seed and sample, confirming
independent subpath wavelengths. Report: `spectral_connection_audit.json`.


The exact-wavelength design audit found two additional integration constraints.
`enqueue_bdpt_light_paths` uses one cache per sample when update_samples=1,
but one cache per batch otherwise; the camera chooses cache by sample offset
only in the former case. Shared-wavelength conditioning must therefore respect
actual cache grouping, including partial scheduler batches, instead of assuming
one light cache per camera sample. Cached vertices currently reconstruct the
wavelength random variable from a 15-bit field in `time_wavelength`; merely
sharing the initial RNG value would still leave a quantized reconstruction.
A replacement needs consistent wavelength identity, PDF accounting and cache
lifetime on both subpaths and sensor connections. No partial wavelength patch
has been applied while the current render batch is running.


Both matched split probes are now terminal and source/binary provenance has
been rechecked. Detailed comparison: `dielectric_split_comparison.json`.
740 nm exhausts 8,191 nodes with accepted fractions 0.00946981 (curvature)
and 0.00947506 (response). For 1600 nm, curvature hits the 256 MiB matrix
budget at 8,038 nodes and fraction 0.02970123; response reaches 8,191 nodes
and fraction 0.03904819, using about 237 MB of matrices. These fractions depend
on depth-first traversal and are not random-query success probabilities.
No partial cache was returned. Neither heuristic resolves scalability, and
concurrent elapsed times are not comparative performance benchmarks.


### Completed corrected larger-source transport batch

All 16 renders completed and source/binary/artifact/mask verification passed.
PT mean RGB is (0.07614695, 0.07569531, 0.07579710); BDPT is
(0.08515186, 0.08589266, 0.08392061). Paired BDPT minus PT difference is
(0.00900491, 0.01019735, 0.00812351), SE (0.00033339, 0.00056565, 0.00134033).
This exposes a clear residual brightness discrepancy. Guided PT minus PT is
(-0.00010210, -0.00007777, 0.00066178), SE (0.00054896, 0.00021649, 0.00101080).
BDPT-guided remains high by (0.00673372, 0.00679513, 0.00772930).
The original 5 mm source is retained; this distinct fixture provides lower
variance, not a replacement of its physical problem. A four-seed 1024-sample
built-in mirror comparison on this exact 50 mm source room is now running
under `mirror_50mm_seed*` to separate generic BDPT residuals from diffraction.


### Per-type bounce-limit hypothesis

Saved-scene inspection confirms total/glossy/transmission limits 12, diffuse
limit 4 and volume limit 0. The inspected camera-to-light connection support
check combines total path lengths but does not combine diffuse event counts.
A 12-diffuse-bounce mirror control batch was launched separately to isolate
this potential support mismatch; it does not replace or relax the required
4-bounce regression. The control generator now records all per-type limits
and accepts an explicit diffuse limit. Log: `indirect_bounce_settings_audit.log`.


The original 50 mm mirror batch is complete and its eight artifact sets verified:
PT mean 0.06237456, BDPT 0.07008105, paired difference 0.00770648 with SE
0.00026525. This establishes the residual in a built-in nonspectral mirror,
independent of the new diffraction closure. `KernelBDPTVertex` stores total
length and transparent prefix count but no diffuse/glossy/transmission/volume
counts. Camera/light concatenation therefore cannot currently enforce combined
per-type support from that record alone. `path_state_next` updates these counts
and sets termination flags, requiring an audit of light-side flag handling too.
The 12-diffuse-bounce controls remain a diagnostic, not the proposed fix.


Follow-up source audit: `PATH_RAY_TERMINATE` is a composite that includes
`TERMINATE_AFTER_TRANSPARENT`, and light generation does check it after shader
evaluation. Thus the source does not simply ignore all per-type termination;
the missing combined-prefix counts remain the concrete gap. Kernel bounce
limits are user limits plus one (`scene/integrator.cpp`), so endpoint accounting
must respect that convention. Mixed shaders also require per-lobe support at
both connection endpoints rather than classifying the entire material as one
bounce type; otherwise a mixed diffuse/glossy node would be incorrectly rejected
or admitted. Early 12-diffuse-limit mirror outputs are closer, but the full
independent-seed batch is still pending and no partial pass is claimed.


The twelve-diffuse-limit mirror batch completed and verified: PT mean
0.07172840, BDPT 0.07130138; difference -0.00042701, SE 0.00035592. This
strongly supports combined per-type support as the remaining generic issue.
An isolated `/tmp/cycles-bdpt-bounce-candidate` carries packed prefix counts
and tests endpoint-lobe pairs for surface connections plus straight sensor
connections. It is a diagnostic, not production: manifold and volume support
are not yet covered. Original four-diffuse-limit seed-11 Metal validation is
running in `mirror_50mm_bounce_candidate_seed11`. Candidate hashes are retained.
# Combined bounce-count experiment: four-seed result

The four-seed analytic enclosure batch completed with verified artifacts and
stable provenance. Expected finite-depth radiance is 0.4921875. Candidate PT
mean is 0.492179666912 (error −0.00000783309, SE 0.00000343576); BDPT mean is
0.492109542885 (error −0.00007795711, SE 0.00001954492). The small BDPT residual
is unresolved, not automatically accepted or attributed to a specific cause.
Production seed 11 gives essentially the same result as the candidate here:
equal total and diffuse limits mean total-length checks already bound this
all-diffuse scene. This is an absolute sanity check, not evidence that the new
per-type rule was exercised. A separate analytic run now holds diffuse limit
four while raising total limit to twelve in both production and candidate.
The expected five-scatter value remains 0.4921875; full physical infinite-depth
radiance remains 0.5. The generator/analyzer explicitly record both limits.

An absolute shared-integrator reference now exists in
`cycles_diffraction_enclosure.py`: a closed cube with uniform emitted radiance
E=0.25 and reflectance rho=0.5, camera inside, black exterior, no other lights.
Every reflected ray meets the same wall response, so L_N=E*sum(rho^k, k=0..N)
and L_infinity=0.5. At user bounce limit four (N=5 scattering events), the
analytic finite-depth value is 0.4921875. Geometry/material reload checks confirm
a closed manifold, an interior camera and the recorded optical parameters.
This scene deliberately has no grating: it isolates shared transport without
treating PT as the physical reference.

The prefix candidate's first Metal seed at 512 samples measures PT
0.492175947209 and BDPT 0.492071474731 against that analytic target. Artifacts
and stable source/binary provenance verified. Three more candidate seeds and
a production-kernel comparison are running. The new enclosure analyzer reports
each estimator's error against the analytic value and independent-seed standard
error; it does not declare a pass solely from PT/BDPT agreement.

Retrospective clarification requested by the user: PT is not physical ground
truth. The review in `cycles_diffraction_retrospective.md` corrects earlier
overstrong image-comparison wording and distinguishes configured cutoff support
from full physical transport. Raw measurements remain unchanged.

Both pending batches have completed and passed artifact/provenance analysis.
At 2048 samples, the reversed-prefix translucent control has BDPT minus PT
+0.00001234747 with paired-seed SE 0.00003141965, versus +0.00012147435 with
SE 0.00004803358 at 512 samples. The earlier residual does not persist at this
precision; neither result alone establishes an absolute physical answer.

For the older terminal-only candidate's 16-render physical room batch, BDPT
minus PT RGB is (−0.00019927, +0.00103029, −0.00186877), SE
(0.00042686, 0.00052381, 0.00075286). BDPT+guiding minus PT is
(+0.00002890, +0.00037012, −0.00022018), SE
(0.00093840, 0.00020240, 0.00035351). All receiver masks and baseline artifacts
were verified. PT mean changes from the prior source are approximately 1e-11;
BDPT brightness changes substantially. These are estimator observations, not
evidence that matching PT is the correct physical target. Guiding images differ
between runs; training-state independence still needs runtime verification.
Runs overlapped other diagnostics, so elapsed times are not isolated benchmarks.

The reversed-prefix candidate's four-seed, 512-sample transmission-limit-zero
batch is complete. With required matching provenance and verified artifacts,
BDPT minus PT is +0.000121474348 with paired-seed standard error 0.0000480335835.
This is approximately 0.38% of PT and 2.53 standard errors from zero; it is not
being declared a pass. A fresh four-seed run at 2048 samples is now testing
whether the remaining discrepancy persists. The seed-11 result changed from
PT 0.031945190116 / label-only BDPT 0.025385486706 to prefix-candidate PT
0.031945190108 / BDPT 0.032206331605. No tolerance has been relaxed and no
production change has been made.

The label-only raised-translucent control at transmission limit zero completed
with verified artifacts/provenance: seed 11, 512 samples, PT mean 0.031945190116,
BDPT mean 0.025385486706 (about 20.5% lower than PT). This is a single-seed
discrepancy consistent with the separately identified missing reversed
light-prefix strategies, not an absolute physical-error measurement. The prefix
candidate's identical-settings comparison remains running; no fix is claimed
from this failure alone.

Coherent-transport research follow-up (not implemented):
[Steinberg and Pharr, *Wave Tracing: Generalizing The Path Integral To Wave
Optics*](https://arxiv.org/html/2508.17386v1) formulates interference with a
bilinear path integral and explains why independently sampled phase-carrying
paths make local importance sampling difficult. Its alternative transports
wave information between finite regions using elliptical cones and develops
a framework that admits bidirectional algorithms. The associated
[wave_tracer repository](https://github.com/ssteinberg/wave_tracer) is a primary
implementation reference, not a dependency currently integrated here.
Design implication for this project: the requested optional cross-object
coherent mode needs source/sensor coherence and wavefront extent, or an explicit
paired-path estimator. Adding optical path length to ordinary RGB throughput
alone would not establish this support. The current spectral grating BSDF and
the bounce-limit diagnostics do not implement that separate transport mode.

The label candidate has now compiled and completed Metal PT/BDPT smoke renders
for mixed reflection and raised diffuse transmission, each 64 samples at
160 × 136. Artifact hashes, finite arrays, means and stable before/after
provenance passed. Mixed means: PT 0.0792667421, BDPT 0.0772595455. Translucent
means: PT 0.0597097421, BDPT 0.0605982870. These single-seed previews are noisy
and establish neither convergence nor benchmark performance.

A separate `/tmp/cycles-bdpt-prefix-candidate` changes light-path continuation
to check each prefix in reversed (camera) order. The optional scattering-limit
argument to `path_state_next` defaults to the existing behavior; only the light
generator's non-null continuation bypasses that directional per-type stop and
applies the explicit reversed-prefix rule. Total and transparent limits retain
their existing behavior. The discrete 88,560-sequence audit confirms that all
reversed-prefix checks together exactly match sequential camera support.
This is still not a full MIS or manifold/volume proof. The prefix candidate is
being compared to the label candidate on the raised translucent control with
transmission limit zero, seed 11, 512 samples, 320 × 272. Production is unchanged.

Future control renders now record input/binary/script/full kernel-tree hashes
before and after rendering, require fresh output directories, and fail if any
of those inputs changes. The analyzer supports `--require-provenance` and
checks consistency across seeds; historical controls without those records
remain explicitly distinguishable and are rejected when the flag is requested.
The prior four-seed numerical result was reproduced after this analyzer change.

The label candidate's mixed-material Metal smoke render has been launched in
`label_mixed_compile_smoke`, overlapping the physical terminal-candidate batch.
Neither timing is an isolated benchmark. During review, the existing MNEE
routine was found to impose transmission, diffuse and total limits internally;
simply adding manifold counts at the final connection would not establish
consistent strategy support. This needs explicit reconciliation and dedicated
covered-material/volume controls before production promotion.

The label candidate now keeps camera counters full-width; only light-prefix
counts remain packed, since the integrator clamps light-path depth to 64.
An extracted actual C++ predicate passed host assertions at 255/256 and 1024,
combined camera/light prefixes, and terminal-order cases. This does not compile
the full kernel. The stored candidate patch/provenance was refreshed.
The control generator exposes separate diffuse/glossy/transmission/volume
limits for upcoming boundary tests.

The saved translucent control was improved before rendering: its former
floor-flush tile hid the receiving surface. The replacement tile is at z=0.22 m,
tilted 60 degrees, with the spotlight aimed at its center. Its lowest edge has
over 0.07 m floor clearance. PT/BDPT versions in
`translucent_transport_raised_saved` passed reload assertions for geometry,
node type and transport settings. These still need rendered visual validation.

Mixed-closure follow-up: `CLOSURE_IS_BSDF_DIFFUSE` includes translucent closures,
but `bsdf_label` assigns them `LABEL_TRANSMIT | LABEL_DIFFUSE`, and path state
increments the transmission counter. This confirms that the terminal
candidate's light-pass partition is insufficient for general closure support.
A separate `/tmp/cycles-bdpt-label-candidate` accumulates three event-specific
`BsdfEval` partitions inside the existing BSDF evaluation loop, using scattering
labels and retaining the original pass components within each partition.
Camera/light endpoint products and straight sensor filtering consume those
partitions without reevaluating the BSDF. This candidate is not compiled or
rendered yet. It still excludes manifold/volume support, retains the diagnostic
packed-count limits, and needs MIS-support and register/performance validation.
Its patch and hashes are saved as `label_candidate.patch` and
`label_candidate_provenance.json`; production and the running physical batch's
terminal candidate are unchanged.

The control generator now creates equal diffuse/rough-glossy mixtures and
translucent controls, plus an explicit `--save-only` mode that does not require
GPU access. Four scenes were saved and reloaded with node/transport assertions
in `mixed_translucent_saved_verification.json`. These are unrendered controls,
not validation of mixed diffraction materials or transmission correctness.

The terminal-event candidate's four-seed mirror batch is now complete and
artifact-verified: BDPT minus PT is −0.000375380602 with paired-seed standard
error 0.000291112289. This is about 1.29 standard errors from zero; it removes
the clear discrepancy of the earlier control without proving convergence or
general correctness. Source hashes were reverified. Visual inspection of both
seed-11 PNGs shows corresponding ceiling/receiver illumination, but PT has
sparse caustic samples and both remain noisy. The same frozen source is now
running the physical grating room at 1024 samples, four seeds and four
transport modes in `physical_indirect_terminal_candidate`. No production
source change or performance claim follows from these diagnostic results.

Terminal-event candidate follow-up: both initial Metal tests completed with
unchanged candidate source hashes. At diffuse limit zero, PT mean is
0.000684334312 and BDPT 0.000684335592; mean difference is 1.2804e-9 and
pixel RMS difference 3.2806e-7. At diffuse limit four, seed 11 gives PT
0.062417466953 and BDPT 0.061685137981, a difference of −0.000732328972.
The original four-limit discrepancy was +0.008210733520 for this seed.
Three additional seeds (29, 47, 83) are running for the terminal candidate;
the first-seed result alone does not establish convergence. All saved
blend/EXR/NumPy/PNG hashes, raw means and finiteness were checked. A patch
against the production headers is retained as `terminal_candidate.patch`,
with patch and base-source hashes in its provenance JSON, so the diagnostic
does not depend solely on a temporary source directory surviving.

Follow-up: per-type limits also constrain event order. After an event reaches
the stored limit (user limit plus one), Cycles permits emitter evaluation but
does not allocate further scattering closures. Consequently, for a complete
camera-to-emitter sequence, every nonterminal event type must have a total
strictly below its stored limit. Only the type of the final scattering event
may equal its limit. Counts without that terminal type accept extra paths.
`cycles_diffraction_bounce_support_audit.py` exhaustively checks 88,560 sequences
and limit combinations: the terminal-aware rule matches sequential stopping;
the count-only rule has 8,298 false positives. In 1,980 cases reversal changes
support, so reciprocal MIS strategy availability also needs explicit handling.
This is a discrete support audit, not a renderer correctness proof.

The Metal zero-diffuse-continuation mirror control at seed 11 / 1024 samples
has PT mean 0.000684334316, production BDPT 0.034806664627, and count-only
candidate BDPT 0.016254388349. Saved artifact hashes and NumPy means were
verified. A separate `/tmp/cycles-bdpt-terminal-candidate` now records the first
light-side surface scattering type (the final camera-side event) and applies
the terminal-aware connection filter. Its diffuse-limit-zero and original
diffuse-limit-four tests have been launched. It remains an isolated surface
diagnostic, with no production promotion or volume/manifold/MIS completeness
claim. Provenance is in `terminal_candidate_provenance.json`.

The isolated `/tmp/cycles-bdpt-bounce-candidate` completed Metal renders for
seeds 11, 29, 47 and 83 at 1024 samples, 320 × 272, using the corrected
50 mm sphere-emitter mirror room and its original diffuse limit of four.
The candidate is not applied to production. Source hashes were reverified
after the batch. `bounce_candidate_four_seed_comparison.json` verifies all
saved blend/EXR/NumPy/PNG hashes, dimensions, finiteness, settings and raw means.
The mean BDPT minus PT difference is 0.00351297767 with paired-seed standard
error 0.00026177405 (neutral RGB). This remains a clear discrepancy, despite
improving on the original difference 0.00770648093. The candidate therefore
does not establish transport correctness and must not be promoted as a fix.

The new `cycles_diffraction_control_analysis.py` reproduces the previously
measured diffuse-limit-12 control difference −0.00042701420 ± 0.00035592332
(standard error), and rejects single-seed and duplicate-seed comparisons.
It reports observations without declaring a convergence pass. Its artifact
validation does not establish binary/source provenance by itself.

Endpoint filtering currently uses light-pass diffuse/glossy partitions, which
are not a general classification of path-state bounce counters: transmitted
diffuse closures and special closures require explicit scattering labels.
The diagnostic also omits manifold and volume endpoint accounting and does
not repair reciprocal strategy support in MIS. These are remaining algorithm
requirements, not permissible exclusions from the requested final feature.

### Joint-coordinate lossless rational interpolation experiment

The prior one-dimensional rational probes were extended with
`tests/performance/cycles_diffraction_tensor_samples.cpp` and
`cycles_diffraction_tensor_fit.py`. The new fixture samples a 7-by-7-by-7 grid
in normalized Bloch coordinate [-0.1, 0], ky [-0.98, -0.90], and wavelength
[380, 400] nm, plus 128 independently positioned deterministic validation
points. All use the 740 nm pitch / 150 nm depth / 0.41 duty, lossless ridge
IOR 1.5, air-exterior N16 reference solver with two retained half orders.
This is a local three-dimensional domain, not the full material domain.

The experiment uses a tensor product of cubic Floater-Hormann rational bases
on a Hermitian Cayley chart. The chart phase is selected using training data
only. The interpolation weights are real; their polynomial reproduction
through degree three and endpoint evaluation were checked to 2e-14. The
method follows Floater and Hormann's 2007 paper:
https://www.inf.usi.ch/hormann/papers/Floater.2007.BRI.pdf
Neither passivity nor small artificial-operator error substitutes for checking
physical output powers after exterior matching.

Artifacts `tensor_dielectric_fit{64,32}.json` and
`tensor_dielectric_power{64,32}.json` under `tests/output/diffraction` show:

- Double precision: maximum operator Frobenius error 0.000187586; physical
  power-column L1 error 0.001591903; maximum coherent power gain
  1.0000000000000577 and column conservation residual 1.90e-14.
- Single precision: maximum operator error 0.000187597; physical power-column
  L1 error 0.001592631; coherent gain range [0.9999844733, 1.0000002701]
  and column conservation residual 4.14e-6.
- The largest tensor coefficient absolute sum was 13.3354. These coefficients
  are not convex interpolation weights. Hermitian structure, rather than
  convexity, explains the double-precision conservation result.

The physical-power validator's legacy scope string says "slice"; these input
queries vary all three coordinates. Its matching and power calculations are
unchanged. Source, binaries and output hashes are recorded post-run in
`tensor_experiment_provenance.json`. No source was changed during execution.
These are CPU numerical experiments, not Metal performance measurements.

This supplies joint-coordinate evidence missing from the prior slice tests.
It does not resolve the production cache failure: 343 matrix supports for one
small domain are still substantial, physical errors are larger than operator
errors, float32 loses exact passivity, and absorbing materials, full-domain
coverage, modal convergence, device cost and adaptive chart selection remain
unverified. No cache defaults or acceptance tolerances were changed.

### Compressed joint-coordinate model and fresh validation

The tensor experiment now optionally uses a real SVD of an isometric Hermitian
packing (diagonal reals and sqrt(2)-scaled real/imaginary upper-triangle entries).
This preserves Hermitian structure when unpacked and measures compression error
in the operator Frobenius norm. Rank selection uses training residuals only.
Roundtrip, norm preservation, and exact float32 conjugate symmetry were checked.

A training residual bound of 1e-7 selected rank 62. Storage fell from 137,200
real scalars for a packed Hermitian grid to 46,466 (66.1% reduction, excluding
small common metadata). Cubic interpolation's physical error remained about
0.001592. This reduces storage, but reconstruction and a dense Cayley solve
remain; no GPU speedup is inferred from the size reduction.

Degree-five interpolation improved the existing validation set. To avoid
accepting that choice on the same set used to observe the improvement, a new
seed 671239 generated 128 fresh joint-coordinate queries. The 343 fitting
samples were verified identical and the two validation sets disjoint. With
float32 degree-five interpolation and rank-62 compression, the fresh set gave:

- Maximum operator Frobenius error: 0.0000316083.
- Maximum physical power-column L1 error: 0.000243850.
- Coherent power gain range: [0.9999749553, 1.0000007235].
- Maximum lossless column-sum residual: 0.00000651866.
- Maximum coefficient absolute sum: 72.7435, demonstrating that interpolation
  amplification near the boundary remains relevant.

These observations are local same-modal-truncation accuracy evidence, not
full-domain, rigorous worst-case, modal-convergence, or Metal validation. The
float32 calculation still has nonzero conservation error. No production cache
or acceptance threshold changed. The exported model for a subsequent device
experiment is `tests/output/diffraction/tensor_dielectric_fresh_degree5.npz`;
reports share that prefix. `tensor_compression_provenance.json` records source,
binary and artifact hashes plus checks of the sample split. The earlier
experiment's provenance remains untouched.

### Metal compressed-tensor experiment

`tests/metal/cycles_diffraction_tensor.metal` implements two stages for the
exported rank-62 model: interpolate real coefficients, then reconstruct packed
Hermitian entries. `cycles_diffraction_tensor.mm` runs the stages on Metal,
checks finite outputs against supplied CPU values and exports every result.
This fixed-size experimental evaluator is not connected to production shading.

On Apple M5 outside the sandbox, 128 fresh queries completed with maximum
packed-entry CPU/GPU difference 1.43051e-6. Mean GPU time over ten measured
submissions after one warmup was 0.000422746 seconds. This tiny batch includes
two compute dispatches; it is not a shader throughput or rendering-overhead
benchmark. Metal safe math was used. No full BSDF speedup is established.

Passing the actual GPU packed output through CPU float32 Cayley reconstruction
and the existing double-precision exterior-matching validator gave maximum
physical power-column L1 error 0.000243115, column-sum residual 3.32714e-6,
and coherent gain range [0.9999872975, 1.0000003219]. Thus the tested Metal
stages retain the local interpolation accuracy. The Cayley solve and exterior
matching were NOT evaluated on GPU by this experiment. Full-domain coverage,
absorbing profiles, complete device evaluation and production integration
remain outstanding. Data and reports use the `tensor_metal_` prefix under
`tests/output/diffraction`; post-run hashes are in
`tensor_metal_provenance.json`.

### Metal Cayley reconstruction extension

The compressed tensor Metal experiment now includes partial-pivot complex
elimination for the full 20-by-20 Cayley reconstruction. This is a diagnostic
implementation with one right-hand side per lane, not a production choice:
it repeats factorization per column and computes columns unused by a single
incident ray. The host accepts an optional seventh argument to enable it.

On Apple M5 outside the sandbox, the same 128 fresh queries gave maximum real
or imaginary component difference 1.08033e-6 against CPU reconstruction and
maximum operator unitarity residual 1.88064e-6. The report's legacy
`maximum_packed_error` field measures component error in this mode. Mean GPU
batch time across ten submissions after warmup was 1.56004 ms, including
interpolation, packed reconstruction and the full Cayley solve, but excluding
exterior matching, BSDF sampling and rendering.

CPU exterior matching of actual GPU amplitudes measured physical power-column
L1 error 0.000243292, lossless column-sum residual 1.11611e-5 and coherent gain
range [0.9999601451, 1.0000437015]. The local interpolation accuracy survives,
but finite-precision conservation is worse than CPU Cayley reconstruction.
No projection, renormalization or clipping was applied to hide that difference.

The existing `diffraction_reference_match_inplace` already combines chart
reconstruction with exterior matching and solves two incident polarization
columns. It is the relevant integration path to evaluate next, rather than
promoting this full-matrix diagnostic implementation. The compressed model's
Hermitian H corresponds to the existing chart Y=iH, subject to checking the
rotation convention and boundary data. Output and post-run provenance use
`tensor_cayley_` prefixes under `tests/output/diffraction`.

### Combined matching convention check

`tests/performance/cycles_diffraction_tensor_match.cpp` now evaluates the
actual Metal-interpolated Hermitian chart with the production
`diffraction_reference_match_inplace<20>` on CPU, solving both incident
polarizations for every propagating input port. It compares unpolarized
physical-order powers to the direct N16 solver, not to another image estimator.
For 128 fresh queries / 256 physical input ports, maximum power-column L1
error was 0.000243588 and conservation residual 8.21892e-6.

This check exposed and corrected an interface-convention mismatch before
integration: the experiment stores `phase` in
S=phase*(I-iH)/(I+iH), while Cycles stores `rotation` in
Y=(I-rotation*S)/(I+rotation*S). Thus Y=iH and
rotation=conj(phase). Passing phase directly gave a very large 0.847024 power
error despite a small 4.90464e-7 conservation residual. That failed result is
retained in `tensor_combined_match_wrong_phase.json`; it illustrates why
passivity/conservation alone does not establish physical correctness.
The correction is in the experiment adapter, not in the production matcher.

The corrected report and source/binary/input hashes use the
`tensor_combined_match_` prefix. This verifies compatibility of the compressed
model's mathematical representation with the existing two-polarization solver.
It does not yet measure that combined solve on Metal or integrate compressed
caches into scene management, device buffers or shaders.

### Production combined matcher on Metal with compressed-chart inputs

`tests/metal/cycles_diffraction_tensor_match.{metal,mm}` invokes the existing
`diffraction_reference_match_inplace<20>` on Metal using actual previously
GPU-interpolated Hermitian charts, the corrected conjugate rotation, and
precomputed exterior boundaries. The fixture exporter includes every physical
input port, giving 256 incident cases across 128 fresh queries. Shader source
includes are expanded exactly from current kernel headers; this is not a
reimplementation of the matching equations.

The Apple M5 run outside the sandbox completed all cases. Maximum GPU/CPU
Jones component difference was 3.25441e-5. Ten measured submissions after one
warmup averaged 0.934358 ms for the matching stage. Boundary construction,
tensor interpolation, shader scheduling and rendering are excluded from this
timing. It cannot establish production rendering overhead.

The validator then read the actual GPU Jones values and compared their physical
order powers directly with the N16 modal solver. Maximum power-column L1 error
was 0.000243331 and maximum lossless column-sum residual 1.72171e-5. No output
was clipped or renormalized. This confirms local same-truncation accuracy
through the existing device matcher, while retaining the measured conservation
error. Artifacts use `tensor_match_gpu_` and the post-run hash record is
`tensor_match_provenance.json`.

This completes a staged GPU numerical check of the compressed local model,
not production integration. The stages still exchange fixture files, the
host cache builder cannot generate adaptive compressed full-domain caches,
and the material/shader pipeline does not select this representation.

### Full-domain single-table rejection

The sample exporter now supports a `full` domain argument after the seed.
A 7-by-7-by-7 degree-five compressed table was built over Bloch [-0.5,0.5],
ky [-1,1], wavelength [380,780] nm with 128 fresh validation queries (seed
817239), using the same lossless 740 nm profile and N16 reference solver.
This tests the proposed representation beyond the favorable local region.

The full-domain table is not acceptable. Maximum physical power-column L1
error was 1.169387, median operator Frobenius error 2.21603, maximum operator
error 4.20514. Training-chart condition was 126.659; rank 110 required 82,130
real scalars versus 137,200 uncompressed Hermitian scalars. The small training
compression residual (3.45e-13) does not measure interpolation between nodes.
Conservation residual stayed small (1.62e-6), again demonstrating that energy
conservation cannot establish diffraction accuracy.

The power validator's minimum gain zero and reference gain error one include
queries with no propagating exterior ports; those fields are not evidence of
absorption or physical nonpassivity in this experiment. The large nonempty
power-column error independently rejects this table. Its legacy "slice" scope
string is narrower than the actual recorded full-domain sample bounds.

Results use `tensor_full_` prefixes; no production defaults or tolerances were
changed. A usable tensor cache needs adaptive subdivision with physical-power
validation per region, boundary/cutoff coverage and memory/build-time limits.
The existing local Metal tests do not discharge these requirements.

### Bounded adaptive tensor builder experiment

`cycles_diffraction_tensor_adaptive.py` now performs breadth-first adaptive
subdivision of the full 740 nm lossless domain. Each candidate receives 343
reference fitting points and 128 distinct validation queries. A physical-power
error <=0.001 is required for acceptance. Raw reference-operator second
differences choose the split axis; they do not relax the acceptance test.
The builder enforces node, depth, float-coordinate and memory limits, records
source/binary stability, and never marks an unresolved tree complete. This is
an offline experiment, not a production cache format or worst-case proof.

The first bounded run examined 31 regions in 61.105 seconds and accepted none.
It stopped at the requested node budget with 32 pending leaves, zero accepted
domain coverage and zero accepted model bytes. The tree's disjoint recursive
partition was independently checked. The complete raw run and manifest are
in `tests/output/diffraction/tensor_adaptive_31`. This rules out a very small
adaptive tree at this accuracy; it does not establish how many regions suffice.
Training chart condition and physical interpolation error remain substantial
in the initial subdivisions. No partial tree was made available to rendering.

The sample exporter accepts explicit cell bounds for repeatable subdivision.
The physical-power validator now separates empty queries from propagating
queries when reporting coherent gain. Re-evaluating the earlier full-domain
predictions found 3 empty and 125 physical queries; minimum physical gain was
0.999992236, not zero. The 1.169387 power error was unchanged. This reporting
fix does not hide or alter any nonempty-channel power error; the earlier report
is retained alongside `tensor_full_power_nonempty.json`.

### Matrix-anchored charts resolve artificial scalar-chart poles

The 127-region scalar-chart run completed in 252.194 seconds with only 21.875%
accepted domain coverage. A controlled diagnostic interpolated the reference
scattering matrices directly on the same grids (diagnostic only: that generally
violates passivity). In node 30, scalar-chart operator error was 2.53382, while
direct interpolation error was 7.52484e-5. The scalar-chart condition at validation
points reached 4295, versus 248 at training points. This isolates a major
coordinate problem, rather than demanding arbitrarily fine grids for a
comparably smooth physical response. Reports: `tensor_chart_diagnostic`.

`cycles_diffraction_tensor_anchor.py` instead chooses the central TRAINING
scattering matrix U as a matrix anchor and fits the Hermitian Cayley chart of
U-adjoint*S. It reconstructs S=U*(I-iH)*(I+iH)^-1. No output-power normalization
or clipping is used. The anchor comes from the modal solver, retaining its
measured numerical residual rather than projecting it to conceal error.
Node 30's physical-power error fell to 6.14815e-6; other selected previously
failing regions also passed 0.001. The full-domain cell still requires splitting.

The anchored adaptive run examined 27 nodes, accepted 14 leaves spanning the
full domain, and stored 4,140,184 bytes including anchors and metadata, in
49.406 seconds. Tree partition and exact array-storage counts were independently
verified. This timing overlapped another CPU experiment and is not an isolated
performance benchmark. Files: `tensor_anchor_adaptive_31`.

Crucially, fresh validation of these frozen models (no refitting) used 1,792
new queries and FAILED the 0.001 criterion: node 13 reached 0.001470449.
Maximum conservation residual was 3.32308e-6. The builder's complete flag means
its initial sampled checks covered the tree, not that the cache is certified.
This cache must be refined before integration; thresholds remain unchanged.
Files: `tensor_anchor_fresh_validation`.

A new unused kernel primitive, `diffraction_reference_match_anchor`, implements
combined physical matching for matrix anchors with two incident polarizations:
V=U*(I-Y), A=(I+Y)-R*V, A*x=T, outgoing=T*V*x-R. It shares one factorization and
does not reconstruct the full scattering matrix first. Independent Eigen dense
algebra tests with 2, 4 and 20 channels, noncommuting random unitary anchors,
identity anchors, and grazing boundaries passed with maximum complex error
3.10875e-7. The existing scalar primitive and shader execution remain unchanged.
The standalone test is `cycles_diffraction_anchor_match_test.cpp`; result is
`anchor_match_algebra.json`. The new primitive still needs actual compressed
profile and Metal tests, plus cache/device/shader integration.

### Anchored refinement margin and Metal matcher validation

The anchored builder now accepts a construction safety factor. A factor of four
keeps the external validation criterion at 0.001 but subdivides unless the
construction probes are below 0.00025. This is a stricter construction target,
not a relaxation of the failed independent check. The resulting run examined
41 nodes and accepted 21 leaves spanning the full domain, with 5,728,812 bytes
of model arrays and metadata. Domain partition and storage were independently
verified. Construction took 70.245 seconds while other work was running, so it
is not an isolated build-time benchmark. Files: `tensor_anchor_margin4`.

Frozen-model validation with a new seed base 67193 used 2,688 additional queries
and passed the unchanged 0.001 criterion: maximum physical power-column L1
error was 0.000396898 and maximum conservation residual 6.23095e-6. No models
were refitted during validation. This is still one N16 lossless profile with
random probes; cutoff-adjacent systematic validation, modal convergence,
complex-amplitude accuracy and other material profiles remain required.
Files: `tensor_anchor_margin4_fresh`.

The matrix-anchor matching primitive was also tested on Apple M5 Metal outside
the sandbox, using the earlier failed node 13 deliberately to retain a known
nontrivial error case. All 822 physical incident ports completed. Maximum
CPU/GPU Jones-component difference was 5.36442e-7. The actual GPU amplitudes
compared with the direct N16 solver gave power error 0.001470123, reproducing
the known cell interpolation failure rather than concealing it. Conservation
residual was 1.89350e-6. Mean matching-stage GPU time was 2.0329 ms across ten
submissions after warmup; interpolation, boundary construction and rendering
are excluded. Files and verified source hashes: `anchor_metal_node13`.

The reusable fixture preparer is `tests/metal/cycles_diffraction_anchor_prepare.py`;
it validates model/sample bounds and expands current production headers with
dependency hashes. The CPU exporter/physical validator is
`cycles_diffraction_anchor_fixture.cpp`. The new matcher is not yet selected
by renderer shaders or scene cache descriptors, so these results do not claim
production integration or rendering overhead.

### Cutoff/face probes and stable barycentric weights

`cycles_diffraction_tensor_cutoffs.py` adds deterministic Rayleigh-cutoff probes
at nine wavelengths and nine Bloch coordinates for orders -2 through 2, with
both signs of ky, adjacent float32 coordinates, and offsets of 1e-5 and 1e-3.
It also probes all adaptive-cell faces at a tensor lattice and adjacent floats.
Coordinates are rounded to float32 BEFORE both cache and direct-reference
queries, isolating interpolation error from coordinate quantization.

The first run caught a genuine evaluator failure: w/(x-node) overflowed for
float32 coordinates one representable step from a knot, giving nonfinite
normalized weights. The failed run remains in `tensor_anchor_cutoffs`.
The scalar and anchored Python evaluators and experimental Metal evaluator now
scale every term by the nearest-node distance before summing. This common
factor cancels in the barycentric ratio and bounds individual distance ratios;
it changes no interpolation model or acceptance criterion.

`cycles_diffraction_tensor_basis_test.py` checks both evaluators at every knot
and adjacent floats in float32/float64, partition of unity, linear reproduction
and polynomial reproduction through degree five. Tests passed. A Metal run
outside the sandbox evaluated 57 in-domain adjacent-knot queries: all finite,
maximum packed-entry CPU/GPU difference 1.43051e-6. That small-batch timing is
not a rendering benchmark.

With the stable evaluator, the frozen 21-cell cache passed all 4,792 systematic
queries (4,102 with propagating channels). Maximum physical-power column L1
error was 0.000690375, below the unchanged 0.001 criterion; maximum column-sum
conservation residual was 1.89271e-6. This includes cutoff and cell-face
neighborhoods but remains a finite same-N16 test, not a rigorous error bound or
modal convergence proof. Files: `tensor_anchor_cutoffs_scaled`,
`tensor_scaled_knots_*`, and `tensor_scaled_basis_provenance.json`.

### Reusable variable-rank tensor kernel

`intern/cycles/kernel/util/diffraction_tensor.h` now supplies a reusable
seven-node barycentric basis and compressed Hermitian-chart evaluator. It
supports variable rank from zero through N squared, rejects short model buffers,
invalid ranks and out-of-domain/nonfinite coordinates, reconstructs exact
skew-Hermitian entry symmetry, and uses the stable nearest-distance basis.
The layout is nodes, weights, tensor coefficients, real matrix directions and
mean; the anchor is separate. No probabilities are normalized or clipped.
The function is not yet connected to scene descriptors or shader evaluation.

CPU tests cover the 57 adjacent-knot queries, constant rank-zero charts, and
128 queries each for ranks 80 and 109, the minimum/maximum ranks in the refined
21-cell cache. Invalid-input rejection is exercised. Maximum component errors
against the independent Python evaluator were 1.86265e-7 (rank 80) and
2.98023e-7 (rank 109), with exact reconstructed conjugate symmetry.

The same production header compiled and ran on Apple M5 Metal outside the
sandbox. For 128 queries, maximum component errors were 1.86265e-9 (constant),
1.86265e-7 (rank 80), and 2.98023e-7 (rank 109). Mean GPU times over ten measured
submissions after warmup were respectively 0.500512, 1.73518 and 1.94069 ms.
These small-batch numbers cover interpolation/reconstruction only and do not
establish rendering throughput or overhead. Larger-batch profiling and scratch
storage/occupancy work remain before any speed claim.

Tests are `cycles_diffraction_tensor_kernel_test.cpp` and
`tests/metal/cycles_diffraction_tensor_kernel.{metal,mm}`. Artifacts use
`tensor_kernel_` prefixes; expanded-header dependency hashes were verified
unchanged and are recorded with outputs in `tensor_kernel_provenance.json`.

### Larger-batch interpolation throughput and rejected scratch specialization

An isolated Apple M5 Metal run repeated the 128 validated inputs 64 times,
producing batches of 8,192 queries. This measures reuse of one material model,
not diverse-material rendering. Ten measured submissions after warmup averaged
1.71103 ms for the constant chart, 4.14199 ms at rank 80, and 6.05729 ms at
rank 109. Numerical component errors were unchanged. These timings exclude
matching, boundary construction, path scheduling and rendering.

A candidate compile-time capacity of 128 reduced the coefficient scratch array
from 400 floats to 128. It gave 4.63095 ms (rank 80) and 6.22412 ms (rank 109),
with bit-identical output to the full-capacity kernel. This provides no measured
speedup; the candidate API/source change was reverted. Expanded candidate shader
and reports remain for inspection. No compression direction was dropped.
Artifacts use `tensor_kernel_rank*_8192_*` and `*_capacity128_*`; the separate
throughput provenance record notes that the candidate is not retained.

### Tensor cell registration and scene-data dispatch

The packed-cell format now accepts representation tag 3 for matrix-anchored
tensor cells. Existing tags 0/1/2 retain their meaning. New cells carry rank in
the second layout record's z component and real model length in w. Complex
anchors and paired real model data share the existing float2 device buffer;
`DiffractionTensorFloatView` reads the real payload without pointer type-punning.
Registration checks lengths with int32 overflow guards, full reference-port
coverage, ordered interpolation nodes, nonzero weights, finite data and anchor
unitarity within float precision. Legacy cells must retain tensor_rank=-1.

Scene-data Jones evaluation dispatches tag 3 through tensor reconstruction and
the matrix-anchor matcher. The existing power evaluation and sampling code then
use those Jones values, preserving the same probability/transport conventions.
The material cache builder still does not generate this format automatically;
this change is the registration/evaluation connection, not completed UI or node
integration of the new cache representation.

`cycles_diffraction_tensor_scene_test.cpp` passed registration, lookup, physical
boundary matching and order sampling at three wavelengths for an analytic
constant lossless tensor cell. It rejected six malformed payloads without
publishing partial buffers. The packed adapter was also checked bit-for-bit
against unpacked real model evaluation for 128 queries each at ranks 0, 80 and
109. Reports: `tensor_scene_test.json` and `tensor_kernel_rank*_packed_cpu.json`.

An existing Metal scene-sampling regression was launched after this change:
`metal_scene_tensor_dispatch_regression.log`. At this note's writing its process
is still live and no completed report has been produced; a Metal regression
pass is NOT claimed here. Keep its input sources frozen until completion.

### C++ tensor-cell preparation

`scene/diffraction_tensor.{h,cpp}` now implements host-side preparation of one
lossless matrix-anchored tensor cell and is listed in the scene library CMake
sources. It performs 343 reference solves, checks fixed port topology and
lossless unitarity, anchors at the central training operator, solves the
Hermitian chart with residual checks, compresses its real isometric packing by
SVD, selects rank from the maximum training residual, and packs the device
layout. Resource overflow, nonfinite results, coordinate poles and unsupported
nonunitary references fail without publishing a cell. Cancellation is checked
between reference solves, before compression and before publication.

This is a candidate-cell constructor, not a replacement for adaptive physical
validation. It does not itself mark a cache usable or select the representation
for shader nodes. No Python interpreter or external model file is needed to
prepare its cell.

A C++ build and test produced the [0,0.5,580] to [0.5,1,780] test cell at rank 87,
65,055 real model floats and 32,928 total float2 pairs including its anchor.
Independent held-out physical validation gave maximum power error 6.09535e-6
and conservation residual 9.80102e-7. The cancellation-enabled build reproduced
that packed model bit-for-bit and rejected cancellation without partial output.
The initial single-cell time was 0.688 seconds, measured during other work, so
it is not an isolated build benchmark. Artifacts use `tensor_host_` prefixes.

The live full scene Metal regression remains in compilation at this note.
An isolated expanded-source candidate relaxes forced inlining for tensor
interpolation, matrix-anchor matching and channel dispatch; original inputs
remain frozen. Both tests must finish before promoting compiler changes or
claiming a Metal scene regression pass. Candidate files use
`metal_tensor_outline_` prefixes.

### C++ packed tensor evaluation for adaptive validation

`diffraction_grating_tensor_reference` now evaluates a packed cell on the host
without Python. It checks dimensions, payload length and query bounds, uses
float arithmetic for the same tensor interpolation/Hermitian unpacking, and
reconstructs the artificial scattering operator with the matrix anchor. It
reports coherent power bounds from the reconstructed operator. Its residual
field is explicitly the Cayley linear-solve residual, not a claim of Maxwell
modal convergence. Physical exterior matching remains a separate step.

The C++ cell test now uses 128 fresh float-representable queries (seed 7542),
compares against direct N16 physical solutions, and rechecks cancellation.
It passed with maximum physical power-column L1 error 1.42877e-5. Inputs differ
from the earlier Python validation, so the different maximum is not itself an
estimator discrepancy. Files: `tensor_host_eval_bounds.*` and
`tensor_host_evaluation_provenance.json`.

Both pending Metal scene compilations were verified to have active compiler
processes, rather than being assumed finished from silent logs. The original
forced-inline run had already exceeded fourteen minutes; the relaxed-inline
candidate also remained active. This compile-time cost is unresolved and must
be addressed before declaring the scene integration ready. No shader-source
changes were made to the live original regression inputs.

### Native tensor cache and explicit Metal outlining

The native adaptive builder completed the 740 nm pitch, 150 nm depth, 0.41
fill, lossless index-1.5 profile over the complete configured domain. The run
produced 89 tree nodes, 45 cells and 10,730,528 matrix bytes in 124.947 seconds,
using 61,870 reference solves. Maximum accepted construction error was
0.000243393. This is construction validation, not fresh-query validation or
modal convergence certification. Production buffer export is retained in
`tests/output/diffraction/tensor_host_cache.bin`.

The expanded Metal candidate explicitly outlining tensor chart interpolation,
anchored matching and fixed-channel dispatch passed 3,615,360 compared
components with zero failures and maximum component error 5.36442e-7.
Seven dispatches (two warmups) gave median GPU time 4.21413 ms for 64,560
queries. End-to-end candidate invocation took 82.8235504 seconds. The fixtures
exercise pre-existing cell representations; this establishes regression and
compilation behavior, not physical tensor-cache device validation or rendering
overhead. Forced-inline and compiler-choice candidates were deliberately
terminated after 22:30 and 14:13 elapsed, respectively, with exact client
identity checks. They are interrupted experiments, not numerical failures.
The three explicit Metal-only noinline qualifiers were then promoted to
source; other backends retain their previous inline qualifiers. A separate
source-based regression is recorded as `metal_scene_tensor_source_noinline`.

The source-based Metal regression subsequently completed successfully: zero
failures, maximum component error 5.36442e-7, median fixed-query dispatch
3.18467 ms. The native exported tensor cache also passed a separate 4,096-ray
CPU test through production scene lookup, interpolation, exterior matching
and power evaluation. Seed 913771 covers both incident sides and random visible
wavelengths. Against fresh N16 Maxwell solves, maximum power-column L1 error
was 0.000195038 and maximum energy-conservation error was 0.0000136346.
The frozen cache was not refitted. The driver, result and source/input hashes
are `cycles_diffraction_tensor_cache_validate.cpp`,
`tensor_cache_fresh_scene_validation.json` and
`tensor_cache_fresh_scene_provenance.json`, respectively. This remains a
same-modal-resolution CPU integration test, not tensor-cache GPU or final
physical convergence certification.

### Full tensor-cache Metal scene test: failure retained

`metal_tensor_full_scene.json` loads the native 45-cell cache and runs 4,096
random incident rays from both sides through the production scene sampler,
order-probability and direction-probability functions. Against CPU execution
of the same cache, 252 component comparisons across seven runs exceeded the
unchanged 2e-6 tolerance; maximum difference was 1.66893e-5. The test failed.
Median GPU dispatch was 146.804 ms. This dispatch includes repeated probability
validation per ray and is not a single-BSDF or full-render benchmark. The
cache is not yet selected automatically by the material manager.

A follow-up uses Metal safe math and logs individual discrepant components.
Its result is `metal_tensor_full_scene_safe.json`; the initial fast-math failure
is retained separately. This experiment does not relax numerical criteria or
establish which estimator is physically correct merely by comparing CPU/GPU.

The safe-math follow-up also failed: 140 comparisons, maximum difference
2.03848e-5, median dispatch 150.021 ms. Its first-run diagnostics identify 18
throughput and two probability discrepancies, with no direction discrepancies.
CPU is not an absolute reference: one CPU throughput is about 1.00002 while
Metal gives approximately 0.999998 for this lossless model.

A fused-accumulation candidate in anchored matrix products is under evaluation.
Its CPU fresh-ray maximum power-column error is 0.000194918 and conservation
error 1.86414e-5, the latter worse than the previous 1.36346e-5. It is not an
accepted numerical improvement. The first Metal compile rejected plain fmaf;
the portable candidate uses metal::fma under Metal and retains that compile
failure in `metal_tensor_full_scene_fma.json`. The numerical GPU follow-up is
`metal_tensor_full_scene_fma_portable.json`. No thresholds or power values were
normalized to pass these experiments.

### Isolating the full-cache matching discrepancy

The fused-accumulation GPU candidate still failed (147 component comparisons,
maximum error 1.60933e-5, median 146.802 ms). It was removed from production;
its header is retained as `diffraction_reference_fma_candidate.h` with the
experiment artifacts. The previously validated source outlining remains.

`cycles_diffraction_tensor_condition.cpp` reproduces query 397 and compares
production matching with independent Eigen double-precision algebra. Results:
float power 1.00001836501; double matching of the same float inputs
1.0000149477; replacing the stored anchor with its double polar factor gives
1.00001489453. Thus the initial suspicion that anchor quantization dominates
was not supported. Reconstructing exterior boundary coefficients in double
precision, retaining the same interpolated chart and polar anchor, gives power
1 to the printed 12-digit precision. Matching condition number is 85.7764;
anchor unitarity residual is 1.95e-7. This isolates boundary coefficient precision
as the dominant error for this query, not a universal diagnosis of all queries.
The diagnostic does not modify the cache or normalize production throughput.
Results are `tensor_matching_condition.json`, `tensor_matching_polar_condition.json`
and `tensor_matching_boundary_condition.json`.

### Direct-admittance matching candidate

A test-only matcher now forms boundary equations directly from normal momentum
q and index n. With B=I+Y and V=U(I-Y), its TE rows are (B-V)+q(B+V),
and TM rows are q(B-V)+n^2(B+V). The TM row scaling avoids dividing by q at
Rayleigh thresholds. Incident right-hand sides are 2 sqrt(q) and 2 n sqrt(q),
respectively, in the transverse TE/TM basis. Output extraction uses the
corresponding B+V and B-V combinations; no power normalization is used.

For query 397, float power is 0.999999270714, versus 1.00001836501 from the
original matching path. The 4,096 fresh-ray CPU test has maximum power-column
L1 error 0.000195336 versus N16 Maxwell and maximum conservation error
8.73208e-6 (previous original path: 1.36346e-5). Independent random-unitary
algebra tests for 2, 4 and 20 channels, unequal exterior indices and grazing
normal momentum pass with maximum complex error 3.83933e-7. These results
support further evaluation but do not yet meet the strict GPU comparison gate.
The candidate remains test-only in `diffraction_admittance_candidate.h`;
production matching has not been replaced. Artifacts: `tensor_admittance_condition.json`,
`tensor_admittance_fresh_validation.json`, `admittance_algebra.json`.

The direct-admittance matcher was executed on Metal with 4,096 frozen CPU
chart/anchor/exterior fixtures. Mean matching-only GPU time was 6.43285 ms
(ten measured runs after one warmup). The strict CPU/GPU component check failed:
seven components exceed 2e-6, maximum 3.27826e-6. GPU maximum conservation error
is 7.31157e-6; CPU is 8.77969e-6 when recomputed from saved Jones components in
double precision. Maximum CPU/GPU power-column L1 difference is 5.16371e-6.
These matching-only times are not full-cache sampler or rendering times.

Independent algebra tests now include mixed propagating/evanescent ports and
exact Rayleigh cutoffs, with unequal indices and random unitary anchors for
2/4/20 channels. They pass with maximum complex error 1.00352e-6.
`cycles_diffraction_admittance_test.py` records source/input hashes, the failed
strict criterion and GPU output. The saved GPU Jones columns are also passed
to the fresh Maxwell validator independently of the CPU/GPU comparison.
The GPU-output Maxwell check completed successfully: 4,096 queries, maximum
power-column L1 error 0.000195812 and maximum float-accumulated conservation
error 7.33137e-6 (`metal_admittance_maxwell_validation.json`). This uses the
same N16 modal resolution and does not establish modal convergence. The
separate strict CPU/GPU component failure remains recorded and unresolved.

### Scene integration of direct-admittance matching

The candidate is now implemented in `kernel/util/diffraction_admittance.h` and
wired into the degree-3 tensor power-column path. Other cache representations
retain their existing matcher. The legacy Jones API accepting precomputed R/T
coefficients also remains separate; ray-based tensor sampling now supplies
normal momentum and exterior index directly. Metal outlines the fixed-size
matcher to bound compilation cost. The tensor and admittance headers are now
listed in kernel CMake sources.

CPU scene registration/lookup/matching/sampling tests pass, including six
malformed-payload rejections. The integrated 4,096-ray fresh Maxwell test gives
maximum power-column L1 error 0.000195336 and conservation error 8.73208e-6,
matching the standalone candidate. Full Metal sampling is tested separately
in `metal_tensor_scene_admittance.json`; the cache manager does not yet select
this representation automatically for Blender materials.

The integrated Metal sampler completed with one throughput component exceeding
the strict 2e-6 criterion (2.02656e-6), repeated across seven dispatches. It
remains a failed strict comparison, although greatly reduced from the original
matching path. Median dispatch was 141.956 ms, including the test's repeated
probability validations; no rendering speed claim follows.

Material preparation now selects tensor cells for lossless physical grating
nodes, with the requested representation included in manager cache identity.
Absorbing profiles retain the previous builder. The normal Blender build
caught a missing Metal output-pointer address-space qualifier; after fixing
it, both Blender and the diffraction device-test target build successfully.
The mirrored full-domain dielectric cache completed in 35.9713 seconds with
25 tree nodes, 13 cells and 3,023,256 matrix bytes. Fresh 4,096-ray validation
through mirrored scene lookup gives maximum power-column L1 error 0.000266435
and conservation error 2.86102e-6. Its result is
`tensor_mirror_fresh_validation.json`.

A real 128x128, 1024-sample PT Metal dielectric furnace render has been started
with the rebuilt Blender binary. Its saved scene and render reports live in
`tensor_dielectric_metal_furnace`; source/binary hashes are recorded in
`tensor_dielectric_material_provenance.json`. The analytic radiance target is
one. Rendering completion and validation must be checked before calling this
an image test pass.

### First rebuilt Blender dielectric furnace result

The 128x128, 1024-sample physical-node PT Metal furnace completed successfully.
Raw mean RGB is (1.00006559322, 1.00004832009, 0.999593706973), versus analytic
unit radiance. Maximum mean error is 0.000406293, within the pre-existing 0.005
smoke criterion. The linear EXR is preserved and a Standard-view PNG was
visually inspected: uniformly white, as expected for this radiometric control.
This is not an angular-efficiency or coherence validation. The log's 78.740 s
save timestamp includes preparation/compilation and is not an isolated render
benchmark.

The BDPT/guided/BDPT-guided furnace controls and six PT angular-region controls
are dispatched sequentially by `cycles_diffraction_blender_tensor_suite.py`.
It verifies frozen source/binary hashes before each job, saves each .blend,
records results and stops on any renderer failure. Angular regions require
independent radiometric comparison; successful image creation alone is not
an accuracy pass. Results are under `tensor_blender_controls`.

A separate normal-incidence modal check uses nine wavelengths from 380 to
780 nm for the exact float-stored dielectric profile. Maximum combined R/T
order-power L1 differences are: 16->32 half-orders 0.0001574511; 32->64
0.0000369029; 64->96 0.00000572257; 16->96 0.0002000765. This supports the
normal-incidence controls at these wavelengths, not all wavelengths or oblique
incidence. Raw outputs and comparison are `dielectric_normal_modal.json` and
`dielectric_normal_modal_comparison.json`.

The BDPT Blender furnace control completed with maximum mean error
0.0004062933 against analytic unit radiance. Its near equality to PT is not
used as the reference, and this simple scene does not test difficult BDPT
connections or mixed delta strategies.

The modal study now includes nine wavelengths, polar angles 0.5/1.1/1.55
radians and azimuths 0/0.6/1.2 radians for the same 740/150 nm dielectric.
Maximum order-power L1 differences are 16->32: 0.0003936318;
32->64: 0.0000939433; 64->96: 0.0000191971; 16->96: 0.0005067722.
The worst sampled case for all comparisons is 680 nm, angle 1.1, azimuth 1.2.
These sampled convergence results do not cover other profiles or prove a
uniform angular/wavelength bound. Raw and summarized outputs are
`dielectric_oblique_modal.json` and `dielectric_oblique_modal_comparison.json`.

`cycles_diffraction_angular_reference.cpp` independently integrates direct
Maxwell order efficiencies into six angular-region RGB references using the
renderer CIE/D65 tables and linear BT.709 conversion. It splits quadrature at
CIE knots and the 666 nm angular boundary, and compares 1/2/4/8 subdivisions
per interval with 16/64 half-orders. Its output must be checked for both
quadrature and modal differences before using it to assess the saved renders.

The direct spectral angular references completed. Maximum RGB change for
quadrature subdivisions 4->8 is 7.27923e-6; maximum 16->64 modal change at
subdivision eight is 5.74831e-6. The six region sums agree with the integrated
neutral RGB to less than 3e-14. Neutral is
(1.00006770093, 1.00004911832, 0.999592659622), closely matching the rendered
furnace means: the small blue offset from ideal unit RGB is predominantly the
CIE/D65 table conversion, not diffraction energy loss. No render or reference
values were normalized. The raw signed RGB values for the outer spectral bands
include negative out-of-gamut components and are preserved.

Reference results, convergence summary and source/input hashes are
`dielectric_angular_reference.json`, `dielectric_angular_reference_summary.json`
and `dielectric_angular_reference_provenance.json`. The comparison script
`cycles_diffraction_blender_angular_analysis.py` verifies scene/EXR hashes and
material parameters, reports signed RGB errors and separate quadrature/modal
changes, and explicitly leaves single-seed comparisons requiring convergence.

The first saved angular render, PT central reflection, completed and passed
artifact hash verification. Mean RGB is (0.02175286894, 0.01097309337,
0.00030390103), versus the 64-mode spectral reference
(0.02185965922, 0.01096449573, 0.00028697256). Maximum absolute component
difference is 0.0001067903. This is recorded as a single-seed comparison
requiring convergence, not an accuracy pass. The current snapshot is
`tensor_angular_comparison.json`; remaining angular scenes are still pending.

`cycles_diffraction_tensor_benchmark.py` prepares repeat-render measurements
using persistent scene data: one recorded warmup followed by at least three
measured renders with distinct seeds. It saves the exact benchmark .blend and
records device and input/output hashes. The ideal mirror baseline is explicitly
a one-bounce workload baseline, not a physically equivalent grating. Timings
are host-observed render calls, not GPU-only timings. No benchmark has yet
been run; it must wait until the angular GPU batch completes to avoid contention.

The PT middle-reflection angular scene also completed and passed artifact
verification. RGB errors against the 64-mode reference are
(-1.53912e-5, -1.51880e-5, 1.15166e-5). Standard-view previews of the completed
central and middle controls were saved; the middle control was visually
inspected as the expected uniform, warm low-radiance field with sampling noise.
Raw EXRs remain authoritative and signed RGB is preserved in comparisons.

`cycles_diffraction_saved_scene_convergence.py` now provides independent-seed
rerenders of saved .blend controls. It saves every seed's scene and EXR,
verifies finite pixels, and reports between-seed standard errors rather than
pixel-derived uncertainty. It requires at least three distinct seeds and does
not normalize or clip measured RGB. This follow-up has not yet been run; the
initial sequential angular-render batch remains active.

All three initial reflection-region Blender images have now been compared
against the independent spectral reference with artifact verification. The
outer-region maximum absolute RGB difference is 3.15866e-6. Transmission-region
renders remain in the live batch. These remain single-seed comparisons; the
prepared convergence rerenders have not yet run.

The guiding-enabled furnace is an integration smoke check, not evidence of
useful guiding of the singular grating event. This fork advertises Metal
guiding support (`device/metal/device.mm`), but mixed diffuse/glossy transport
scenes and guided sampling statistics remain necessary. Likewise, no result
here implements the separately requested general cross-object coherent mode.

All six initial angular-region scenes have completed and their .blend/EXR
hashes are verified by `tensor_angular_comparison.json`. Maximum RGB component
errors are: reflection central 1.06790e-4, middle 1.53912e-5, outer 3.15866e-6;
transmission central 3.93480e-4, middle 2.72723e-4, outer 2.47785e-6. These are
single-seed diagnostic comparisons, not convergence certification. Three
independent seeds (29/47/83) at 4096 samples have been launched for the saved
central-reflection scene in `tensor_reflection_central_seeds`.

A full mirrored dielectric CD-pitch (1600 nm) cache build is running separately
with all nine upper and nine lower ports, requiring 36 polarization channels.
No orders are dropped to fit the current 20-channel renderer capacity. The
independent admittance algebra test extended to 36 channels passes with maximum
complex error 1.08448e-6. This is CPU algebra coverage, not yet Metal capacity
or full CD material support. The cache run is `tensor_cd_cache` and the algebra
result is `admittance_cd_algebra.json`.
The CD cache run subsequently terminated without publishing a cache: after
54.1984 seconds and 33 nodes (15.6738% diagnostic accepted coverage), preparation
reported `Tensor chart solve exceeds roundoff tolerance`. `complete` is false
and output cells are zero. This failure is retained; the tolerance was not
relaxed and partial coverage is not treated as supported CD dielectric shading.

The CD failure occurs during the anchored-chart conversion after the direct
reference operators passed the unchanged lossless-unitarity check. An isolated
candidate now classifies preparation failures explicitly as fatal or requiring
refinement. Both an ill-conditioned chart coordinate pole and failure of the
chart solve/Hermitian roundoff criterion request subdivision; neither permits
acceptance of that candidate cell. All child candidates still face the original
roundoff, compression and independent physical validation criteria. Resource,
profile and cancellation failures remain fatal, and no partial cache is
published. This is a candidate, not yet promoted to production.

The candidate is running the same full CD domain under the same node/depth
limits in `tensor_cd_refinement_cache`. Its frozen source snapshot and hashes
are in `cd_refinement_candidate_sources`. The active Blender independent-seed
render continues using the previously built binary and kernel source.

The rebuilt CPU diffraction device integration executable completed with exit
zero. It covers physical shader graph compilation/SVM execution, linked
orientation, mixing/backfaces, order routing with unequal indices and mirrored
queries, OSL closure setup, and cache upload/replacement/release/error recovery.
The intentional device-error injection is expected test coverage, not a renderer
failure. Full output is `tensor_material_cpu_device_regression.log`. This
regression does not establish GPU performance or general coherent transport.

The isolated CD refinement candidate has progressed past the original node-33
chart-conversion failure; full domain completion is still pending. The first
4096-sample independent-seed central-reflection EXR is saved; remaining seeds
are still running, so no between-seed convergence result is claimed yet.

The isolated CD refinement candidate exhausted its unchanged 127-node test
budget after 198.713 seconds, 78,856 reference solves and 16.4062% diagnostic
coverage. It published zero cells and no cache. Accepted partial-cell maximum
construction error was 0.000246403; this is not a whole-domain pass. The result
is `tensor_cd_refinement_cache.json`. A DVD mirrored-cache regression of the
same candidate is now running before any production promotion.

The independent dielectric CD modal study sampled nine wavelengths, four polar
angles (0/0.5/1.1/1.55 radians) and three azimuths (0/0.6/1.2 radians).
Maximum order-power L1 changes: 16->32 0.00161139; 32->64 0.000327215;
64->96 0.0000483934; 16->96 0.00198700. The worst sampled point is 380 nm,
angle 1.55, azimuth 1.2. Thus same-N16 cache validation alone is insufficient
for a 0.001 total physical error claim for this profile. Raw results and
comparisons are `dielectric_cd_modal.json` and `dielectric_cd_modal_comparison.json`.

The three 4096-sample independent-seed central-reflection renders completed.
After verifying every .blend and EXR hash, mean RGB errors against the N64
spectral reference are (2.28207e-5, 1.19289e-5, 1.32503e-5). Between-seed
standard errors are (5.46100e-6, 4.07660e-6, 7.04362e-6). The red difference
is about 4.18 observed standard errors; with only three seeds and numerical
reference/cache errors, this is a diagnostic discrepancy, not a convergence
pass. No values were brightness-normalized. Details are in
`tensor_reflection_central_seed_comparison.json`.

The isolated refinement candidate's DVD mirrored-cache regression completed
with 25 nodes, 13 cells and 3,023,256 matrix bytes in 32.8967 seconds. Its
exported cache is byte-identical to the previously validated DVD cache. This
supports the refinement classification change but does not fix CD resource
requirements; production promotion is still pending.

With preceding CPU/GPU experiments terminal, the isolated diffraction rendering
benchmark has started: one recorded warmup, three measured 1024-sample renders,
persistent scene data and the existing 128x128 furnace geometry. Output is
`tensor_benchmark_diffraction.json`. The mirror baseline must run separately
before drawing a relative timing conclusion.

Prepared two follow-ups while the isolated benchmark remains active:
`cycles_diffraction_angular_cache_reference.cpp` integrates the exported
mirrored cache through the production scene-power path using the same angular
bins and spectral quadrature as the direct Maxwell reference. It has not yet
been compiled/run; it is intended to separate cache interpolation bias from
render sampling error. The exported host cache's duty cycle is 0.41 in double,
whereas Blender stores float(0.41); this difference must be retained in the
interpretation or removed by building an exactly matching cache.

An isolated interpolation candidate skips exactly zero barycentric weights
instead of loading all 343 terms at grid-node coordinates. It does not change
nonzero weights, normalization or the cache. Candidate header snapshot is
`diffraction_tensor_nodal_candidate.h`; no speed or accuracy claim is made
before tests, and the active renderer source remains unchanged.

### Explicit fixed-sample benchmark audit and nodal candidate check

The repeat-render benchmark now writes schema 2 with effective sample count,
adaptive-sampling and denoising flags, and resolution percentage, both at report
level and for each run. It rejects changes before/after rendering. The analyzer
requires these fields and rejects enabled adaptive sampling or denoising. Older
reports and their archived comparison are retained unchanged; their runner
explicitly disabled both settings, but did not serialize those flags. Both edited
Python scripts pass syntax compilation. No new rendering performance claim is
made by this metadata change.

The isolated zero-basis-term interpolation candidate completed the normal-incidence
Metal test: 4096 queries, seven runs, 14 component failures (two distinct queries
repeated seven times), maximum CPU/Metal discrepancy 4.41074e-6 against the unchanged
2e-6 limit. Measured dispatch median was 31.9175 ms; this includes repeated sampling
checks and is not a render benchmark. The candidate remains unpromoted. The matching
unmodified normal-incidence baseline was launched separately, with no concurrent
GPU timing job, to distinguish existing discrepancies from candidate regressions.

The unchanged normal-incidence Metal baseline completed with exactly the same two
query/component discrepancies (14 repeated failures, maximum 4.41074e-6). Its
median dispatch was 30.2108 ms, compared with 31.9175 ms for the candidate. This
one comparison provides no evidence of speed improvement, so the candidate is
not promoted. CPU comparisons at ranks 0, 80 and 109 each covered 4439 queries
and found zero differing complex components. Full evidence is retained in
`tensor_nodal_decision.json`. A separate dense normal-incidence spectral scan
against the direct Maxwell solver was launched to assess absolute power and
conservation errors rather than treating CPU/Metal agreement as physical truth.

The dense normal-incidence scan completed successfully: 2048 wavelength
midpoints on each incident side, 4096 evaluations total. Maximum order-power
column L1 difference versus the direct N16 Maxwell solver was 0.000199736;
maximum lossless conservation error was 0.0000364333. Both maxima occurred
near 750.215 nm, consistent with the region of the strict CPU/Metal discrepancy.
These pass the existing 0.001 power and 0.0001 conservation gates, but do not
resolve the separate 2e-6 CPU/Metal component gate or establish modal convergence.
Result: `tensor_normal_spectral_validation.json`. No renderer source was changed
by these experiments.

### Single-sample Metal profiling and dispatcher experiment

The Metal test runner supports an explicit tensor-sample-only mode. It still
compares every sampled direction, order, eta, probability, and throughput with
the production CPU evaluator at the unchanged 2e-6 threshold; it omits additional
order/direction probability API calls from the timed kernel. The default full
API test remains unchanged. `metal_tensor_single_sample_random.json` records
4096 mixed-angle queries, two warmups and five measured dispatches: median
22.3271 ms, one distinct throughput discrepancy 2.02656e-6 repeated seven times.
This is a sampler microbenchmark, not a Blender render benchmark.

An isolated evaluator candidate moves the per-channel local chart array into a
selected fixed-size outlined function and leaves an array-free inline dispatcher.
The hypothesis is that removing nested outlined dispatch frames reduces temporary
GPU storage and call overhead. Physics, interpolation, solve and channel capacities
are unchanged. The candidate is retained as
`diffraction_evaluate_dispatch_candidate.h`; it is not production code.
Its single-sample Metal comparison is running separately from the baseline.

The dispatcher candidate completed: median 22.5004 ms versus baseline
22.3271 ms, with the identical query-1531 throughput discrepancy of 2.02656e-6.
No speed benefit was demonstrated; the candidate is not promoted. Decision and
report hashes are retained in `tensor_dispatch_decision.json`. Both timing jobs
are terminal. No production renderer source was changed.

### Isolated fused elimination candidate

A candidate changes only Gaussian-elimination and back-substitution complex
subtract-product updates to explicit fused multiply-add operations. It leaves
chart reconstruction, boundary coefficients, pivot selection and division
unchanged. This differs from the earlier broader FMA experiment. It is tested
against the unchanged CPU sampler, with the same 2e-6 component gate and
single-sample timing workload. The first Metal compilation failed because
`fma` requires the `metal::` namespace in this expanded source; that failure is
retained in `metal_tensor_single_sample_solve_fma.json`. The corrected isolated
header is `diffraction_reference_solve_fma_metal_candidate.h`, and its GPU
comparison is running. Nothing has been promoted to renderer source.

The qualified fused-elimination candidate completed with median 22.3446 ms
(no demonstrated speed benefit versus 22.3271 ms baseline). It has two distinct
CPU/Metal discrepancies, maximum 2.68221e-6, compared with one in the baseline.
This is not yet a judgment of physical accuracy: the lossless target is unit
flux, not float-CPU agreement. The tensor scene test now reports maximum CPU
and GPU throughput deviation from unity independently. A fresh baseline with
this instrumentation is running before the corresponding candidate measurement.

Absolute energy audit completed for the same 4096 mixed-angle queries.
Production GPU maximum unit-flux error: 2.26498e-6; fused-solve candidate:
2.08616e-6; unchanged CPU: 2.44379e-6. Thus the candidate has slightly lower
worst conservation error despite more strict CPU/Metal component discrepancies.
No claim of worse physical accuracy is made based on those discrepancies.
Fresh medians were 22.0851 ms baseline and 22.4994 ms candidate. The candidate
is not promoted: no speed benefit was shown and conservation alone cannot
certify individual diffraction-order powers. `tensor_solve_fma_decision.json`
retains the decision and source report hashes. All jobs are terminal.

### CD refinement trace

A separate copy of the unpromoted refinement candidate now logs each prepared
cell's bounds, depth, rank, preparation failure, validation errors, and split
axis/curvature. The physical model, tolerance, N16 modal truncation and 127-node
budget are unchanged. Sources are preserved under `cd_trace_candidate_sources`
with `cd_trace_provenance.json`. This is a diagnosis of cache construction, not
a claim that N16 suffices for converged CD physics.

The trace reproduces the roundoff failure at node 32, depth 11, bounds
[-0.5,-0.59375,380] to [-0.4921875,-0.5625,780]. It reveals substantial angular
refinement before spectral refinement. The run is still active. The new
`cycles_diffraction_tensor_trace_analysis.py` summarizes completed traces without
changing acceptance criteria or publishing incomplete caches.

The completed trace reproduces the prior diagnostic totals exactly: 127 nodes,
78,856 reference solves, diagnostic coverage 0.164062, 13,039,008 bytes, and no
published cache. There were two preparation roundoff failures. Split counts
were 11 in Bloch coordinate, 7 in transverse momentum, and 49 in wavelength;
maximum visited depth was 17. Thus the early angular concentration must not
be mistaken for the overall split distribution. Analysis is retained in
`tensor_cd_trace_analysis.json`.

A separate candidate replaces second-difference split scores with squared
sixth differences over the seven nodes. This targets the highest resolved
polynomial variation instead of smooth low-order curvature. It is a heuristic,
not an error estimator or acceptance criterion. All direct physical probes,
accuracy gates, retained orders, N16 modal resolution, and resource limits are
unchanged. Candidate sources and hashes are preserved in
`cd_sixth_candidate_sources` and `cd_sixth_provenance.json`. Its CD run is active.

Interpolation reference revisited: Floater and Hormann (2007),
https://www.inf.usi.ch/hormann/papers/Floater.2007.BRI.pdf .
The stored seven-node weights are proportional to the alternating binomial
coefficients of degree six. The sixth difference is therefore a resolved
high-order coefficient indicator, not an unresolved term; the candidate
comments explicitly avoid claiming an error bound.

### Tabulated-aluminum CD modal audit

Added `cycles_diffraction_aluminum_cd_modal.cpp`, a direct Maxwell convergence
audit for a 1600 nm pitch, 150 nm depth, float(.41) duty lamellar aluminum relief
on an aluminum half-space in air. It loads the full retained Rakic-1995 optical
constant table for both ridge and substrate. The table hash was verified against
the source manifest; source/input hashes are recorded in
`aluminum_cd_modal_provenance.json`. This is not a complete commercial CD model
(no cover layer, pits or oxide), and no claim of raw measured optical constants
is made.

The audit compares half-orders 16/32/64/96 at wavelengths 400/550/700 nm,
incidence 0/0.7/1.3 radians, and azimuth 0/0.6 radians. It records every
unpolarized reflected-order power, substrate-entering flux, layer absorption
and boundary residual. Substrate-entering flux must not be described as
far-field transmission in this absorbing material. The audit is running.
This CPU accuracy job overlaps the CD refinement diagnostic; their wall times
must not be used as isolated performance benchmarks. No GPU render benchmark
is running concurrently.

The sixth-difference experiment is terminal and rejected: it exhausted 127
nodes at coverage 0.0546875 (versus 0.164062 for the second-difference baseline),
with 79,360 reference solves and 12,580,096 diagnostic bytes. No partial cache
was published. The analyzer completed and retained the full trace in
`tensor_cd_sixth_analysis.json`. Construction times are not isolated benchmarks.

The first aluminum modal audit stopped at the exact 400 nm normal-incidence
Rayleigh threshold: the direct physical solver explicitly requires a limiting
solution there. The original source/report/error are retained. The updated
audit records individual failed cases and continues the remaining requested
queries, returning failure if any case failed. Wavelengths were not perturbed
and unsupported cases are not omitted. The all-cases run is active.

The aluminum all-cases audit is terminal: 64 successful solves and eight
explicit exact-grazing failures (400 nm, normal incidence, both azimuth entries,
all four modal resolutions). The worst modal discrepancy occurs at 400 nm,
incidence 1.3 radians, azimuth zero. Reflected-order L1 changes are 0.05316965
for N16→32, 0.00802977 for N32→64, and 0.00101414 for N64→96; N16→96 is
0.06144012. Including substrate-entering flux, the corresponding final total
is 0.09783873. N16 is clearly inadequate for this aluminum relief, and N96
is not yet proven converged to a 0.001 target. The near-zero reported energy
balance error is only bookkeeping: this solver defines layer absorption as
the residual, so it does not independently validate accuracy. Results are
retained in `aluminum_cd_modal_analysis.json`. All current jobs are terminal.

### Higher-resolution aluminum modal check

At the previously worst 400 nm / 1.3 rad / zero-azimuth case, the extended
N64/96/128/192 audit completed successfully. Reflection-order L1 differences
were 0.00101414 (64→96), 0.000423840 (96→128), and 0.000442917 (128→192).
The combined 96→192 reflection difference is 0.000802801. These finite
resolution comparisons do not establish an absolute error bound. A further
N256 solve is running; no default solver resolution was changed.

Added `cycles_diffraction_modal_analysis.py` to retain all pairwise modal
comparisons and unsupported cases, preserve signed powers, reject nonfinite
values and conflicting repeated results, and hash input reports. The combined
audit retains eight grazing failures alongside 105 modal comparisons.
Absorbing-substrate flux remains separate from far-field reflection.

The N256 solve is terminal and successful. At this case, N192→256 changes
reflected-order powers by L1 0.000307060; N96→256 is 0.00109476, exceeding
the 0.001 target despite the smaller earlier 96→192 difference. This is
evidence against certifying convergence from a single adjacent comparison.
N16→256 reflection L1 is 0.06215062. Full comparisons and retained failures
are in `aluminum_cd_modal_through_256.json`. This remains one angular/spectral
point, not a full-material convergence guarantee. All jobs are terminal.

### Exact zero-ky modal decomposition candidate

An isolated host-solver candidate exploits the exact block diagonal form of
P*Q when ky equals zero. It solves the two count-by-count polarization blocks
separately and embeds their full eigenvectors in the existing mode basis.
All Fourier orders remain present; nonzero ky uses the original coupled solve.
No angular epsilon or physical approximation is introduced.

Across the 72-case tabulated-aluminum audit, 64 supported cases match the
original solver within 1.6598e-13 per reflected/substrate-flux order, against
a 1e-9 comparison gate. Eight exact-grazing failures remain identical. The
N128 worst-case CPU preparation benchmark runs one warmup and four measured
solves per implementation, sequentially without other task CPU/GPU jobs.
This measures host Maxwell preparation, not GPU rendering; adaptive rendering
sampling is not involved. The candidate remains unpromoted pending timing
and dielectric regression evidence.

The zero-ky decomposition is now promoted to `scene/diffraction.cpp`,
byte-identical to the validated candidate. Additional evidence: 432 dielectric
CD cases differ by at most 3.1003e-14 per order; 37,632 real/imaginary Jones
components across dielectric/metal, flat/relief, planar/conical configurations
differ by at most 4.7574e-14. All use the predeclared 1e-9 comparison gate.
The CPU N128 preparation benchmark measured medians 0.4279398125 s original
and 0.283190667 s decoupled, a 1.5111× speedup for that planar-incidence case.
This is not a GPU rendering speedup and does not fix modal convergence or
exact-grazing support. Full Blender and the CPU device integration target are
being rebuilt; prior render/binary provenance remains archived unchanged.

The promoted zero-ky change passed the full Blender/device-test rebuild and
CPU device integration executable (exit 0), including expected intentional
device-error recovery. Logs: `build_modal_decoupled.log` and
`modal_decoupled_device_regression.log`. A fixed-1024-sample Metal PT lossless
furnace render is now active with the rebuilt Blender, adaptive sampling and
denoising disabled by the scene runner. New binary/source/script hashes are
retained in `modal_decoupled_blender_provenance.json`.

The rebuilt Metal PT furnace completed successfully at 1024 fixed samples.
Verified saved blend/EXR hashes and analytic-smoke status. Raw RGB is
[1.00006559378, 1.00004831992, 0.99959370719], maximum unit-neutral deviation
0.000406293, matching the previous result within sampling/roundoff precision.
The Standard-view PNG was inspected and is a uniform white furnace, as
expected. This is an integration check, not a rendering performance result.

### Exact aluminum Rayleigh cutoff reference-path check

The reference-port solve and physical boundary matcher succeed at the exact
400 nm normal-incidence cutoff for the tabulated-aluminum CD relief at
N32/64/128. At N128, R totals 0.91714188356 and the propagating order set is
-3 through 3; threshold ±4 carry zero far-field flux. One-sided power-column
L1 differences from the exact cutoff decrease consistently as wavelength
offsets shrink from 0.01 to 0.000001 nm. At the smallest offsets the differences
are 3.34352e-5 below and 6.10552e-6 above the threshold.

This distinguishes the earlier direct-response API limitation from the
reference-port path used to build caches. It does not solve the remaining
CD cache resource limits or certify modal convergence. A cross-check against
the direct physical API at all nonexact wavelengths is running with a 1e-8
L1 gate; the exact direct-API failures remain explicit in the output.

The cutoff cross-check is terminal and passed: 18 nonexact cases agree
with the direct physical solver to maximum reflected-order L1 1.714212652568383e-15.
All three exact reference solves succeeded while preserving their explicit
direct-API failure records. All current jobs are terminal.

### Empirical modal-convergence host API

Added `scene/diffraction_convergence.{h,cpp}` and registered it in the scene
library. The routine compares complete physical scattering operators after
reference-port matching at increasing Fourier resolutions. It checks maximum
unpolarized power-column L1 differences and optional complex Frobenius
differences. Success requires two consecutive adjacent comparisons plus the
comparison spanning those three resolutions to pass. It returns the highest
resolution reference operator with diagnostic observations; cancellation,
invalid input, numerical errors, and exhausted budgets publish no operator.
This is an empirical stopping criterion, not a proven truncation bound.

The standalone regression passed analytic flat-interface Fresnel powers,
complex-amplitude agreement, insufficient-budget rejection, cancellation,
nonfinite-option rejection, dielectric-relief convergence, and under-resolved
metal rejection. The dielectric query selected N64 with spanning power
difference 1.83706e-5. The metal query at its N64 cap had adjacent difference
0.00503504 and spanning difference 0.027845, correctly failing the 0.001 gate.
Evidence: `modal_convergence_test.json` and source hashes in
`modal_convergence_test_provenance.json`.

Automatic material/cache construction does not yet call this API; its fixed
modal-resolution accuracy limitation remains open. The full Blender and CPU
device integration target are being rebuilt with the new library source.

The Blender and device-test rebuild with the convergence API completed
successfully (`build_modal_convergence.log`). The API remains a validated host
primitive pending cache-builder integration; no rendering-performance or
full-material accuracy claim follows from this addition.

### Cache-builder modal convergence integration

Both ordinary and tensor cache builders now use a shared convergence-aware
reference solve for fitting and validation. Cache options expose empirical
modal power/complex tolerances and a maximum Fourier resolution; zero power
tolerance preserves fixed-resolution behavior. Manager identity includes all
three new settings. Modal solve counts account for internal refinement solves
in successful queries. The convergence implementation moved into
`scene/diffraction.cpp` (header remains separate), preserving existing
standalone solver tools that link that translation unit directly.

The initial broad flat-interface cache fixture exhausted its 31-node
interpolation budget (`cache_modal_test.log`). This was retained. A focused
600–610 nm, ±0.015625 angular-coordinate fixture isolates modal integration:
both builders construct one validated cell, with 105 actual modal solves
for the ordinary builder and 1788 for the tensor builder. Both reject a cap
allowing only two resolutions and publish no cells. This narrow integration
test is not evidence of full-domain cache coverage. Manager identity/failed
publication regression and the full Blender rebuild are running.

Automatic physical material nodes still leave modal refinement disabled.
Full-profile cost and acceptance must be validated before enabling it there;
the previously identified fixed-resolution material accuracy issue remains
open. The current implementation adds actual cache-builder support, not a
claim that all material construction is now converged.

Cache-manager regression passed: exact requests reuse handles, changed modal
tolerances get distinct handles, and an insufficient modal budget leaves
existing registrations unchanged. Full Blender/device rebuild and CPU device
integration passed (exit 0). Logs: `cache_modal_manager_test.log`,
`build_cache_modal.log`, `cache_modal_device_regression.log`. Sources are
hashed in `cache_modal_provenance.json`. All jobs are terminal.

### Full-domain DVD cache with empirical modal refinement

A new full-domain run uses pitch 740 nm, depth 150 nm, exact float(.41) duty,
lossless index 1.5 ridges in air, the mirrored full angular domain, and
380–780 nm wavelengths. Every fitting and validation reference query uses
modal power tolerance 0.00025, starting N16 and capped at N256. The original
interpolation criterion (0.001 with fourfold construction margin), 127-node
budget, depth 18, and memory cap remain unchanged. This separates modal and
interpolation agreement criteria without claiming either is a rigorous bound.

The driver is `cycles_diffraction_tensor_modal_cache.cpp`; source and executable
hashes are retained in `tensor_modal_cache_provenance.json`. The run is active
and has progressed through at least 516 actual modal solves in its first cell.
No other heavy task CPU/GPU job is running, and solver sources remain frozen.
This profile uses Blender's float duty rather than the old double-.41 host
fixture, so a later comparison must disclose that small input difference.

Prepared the native cache validator for the modal-refined cache: it now accepts
an explicit direct-reference Fourier resolution (up to N512), sample count,
independent RNG seed, and Blender-float duty selection. It preserves the old
defaults for archived fixtures and the existing 0.001 power-column / 0.0001
conservation gates. Reports include the actual resolution, seed, query count
and duty at full double precision. A syntax-only compile passed. Validation
of the new cache is pending its successful complete publication; no run is
claimed yet.

A read-only transport review reconfirmed a separate outstanding integration
issue: `bsdf_diffraction_smooth_delta_probability` currently has no integrator
call sites, and both camera/light singular-event MIS branches retain the
existing cosine-only update. Unequal forward/reverse grating order masses
still require explicit path-strategy treatment; this is not resolved by the
modal-accuracy work or the furnace render.

The first full-domain modal-refined DVD build is terminal: it failed the
N256 modal-resolution budget in the second cell after 165.259 s, with no
published cells or matrix bytes. The report counted 2855 reference solves;
inspection found that a failed tensor fitting query had not propagated its
last local solve count to outer diagnostics. This reporting issue is now
corrected by forwarding progress after each reference attempt.

Cache modal failures now include full-precision wavelength, kx, ky and the
adjacent/spanning power and complex differences at every attempted completed
resolution. A new diagnostic run repeats the same profile and unchanged
acceptance criteria to locate the failing query. Its outputs use the
`tensor_modal_diagnostic_cache` prefix; the first failed run is retained.
No accuracy threshold or resource cap was relaxed, and automatic material
settings remain unchanged.

Prepared `cycles_diffraction_modal_query.cpp` to reproduce a single full-precision
wavelength/kx/ky query for the Blender-float DVD relief at N16/32/64/128/256/512.
It records full physical Jones operators, propagating-port identities, boundary
residuals and power-gain ranges, retaining per-resolution failures. The companion
`cycles_diffraction_modal_query_analysis.py` computes complete power-column and
complex Frobenius differences between all completed resolutions. Syntax checks
passed; numerical execution awaits the active full-domain diagnostic's failing
query. Diagnostic source/executable hashes are recorded separately.

The diagnostic DVD build completed with the same failure at wavelength
580 nm, kx=-0.2286036036036036, ky=-0.16666666666666663. Corrected solve
count is 2860 (five attempts were missing from the earlier report). The
query's adjacent power differences decrease from 0.00457645 at N32 to
0.00117488 at N64, 0.000331804 at N128, and 0.0000867767 at N256. The
N64→256 spanning difference 0.000418581 exceeds the 0.00025 criterion.

The complete single-query N512 audit succeeded. N256→512 power-column L1
is 0.0000210893 and N128→512 is 0.000107866; these meet the unchanged
consecutive/spanning criterion. Raw complex operators and pairwise
comparisons are retained in `modal_failed_query.json` and its analysis.
This supports N512 for this query, not a full-domain convergence claim.

Added a per-build bounded exact-query LRU to the tensor cache builder, using
the existing reference-matrix byte budget. It reuses reference operators
without coordinate rounding, interpolation, or changed physical tolerances.
Standalone cell preparation retains its original independent behavior.
The modal-refined flat fixture produced byte-identical device buffers with
cache disabled, a 64 MiB limit, and a one-matrix 6400-byte limit. Successful
modal solves dropped from 1788 to 1707 with 27 hits; peak matrix memory was
3,641,600 bytes. The one-matrix case respected its cap and retained identical
output. A full relief-domain, multi-cell equivalence/eviction test is active.

The full fixed-N16 DVD relief memoization test passed across 13 cells: exported
device buffers are byte-identical with zero reuse budget, 64 MiB, and a
one-matrix 6400-byte limit. Actual reference solves dropped from 17,102 to
8,238 (8,864 exact hits) with peak reference-matrix memory 52,723,200 bytes.
The one-matrix eviction run stayed within its cap and retained identical
output. This proves solve-count reduction and numerical equivalence for this
fixture, not a measured GPU rendering speedup. Full Blender/device rebuild
and CPU device integration passed.

The full-domain modal-refined DVD run is now relaunched with maximum N512,
supported by the isolated failing-query result and the API's existing limit.
Modal/interpolation tolerances, spatial/spectral domain, depth/node limits
and output-memory limits remain unchanged. Exact-query reuse is enabled.
Output prefix: `tensor_modal512_cache`. Solver sources and executable remain
frozen during the run; no other heavy CPU/GPU jobs are active.

During the active N512 full-domain run, all recorded solver/driver source
hashes were reverified and the live executable hash added to its provenance.
The run has progressed through at least 1032 modal solves; no completed cache
or validation result is claimed.

Prepared an isolated next optimization in
`diffraction_real_modal_candidate.cpp`: use Eigen's real nonsymmetric
eigensolver only when every imaginary component of the modal matrix is
exactly zero; retain the complex eigensolver otherwise. This also preserves
the previously validated zero-ky block decomposition. No epsilon or physical
approximation is introduced by the branch criterion. The candidate is not
yet compiled, numerically tested, benchmarked, or promoted. Those checks are
deferred until the active preparation run ends to avoid timing contention.

Prepared `cycles_diffraction_dielectric_modal_benchmark.cpp` for that candidate:
the Blender-float DVD profile at 580 nm, kx=-0.2286036036036036, both zero ky
and ky=-0.16666666666666663, and N64/128/256. Each case has one warmup and
four measured repetitions and records complete physical complex operators,
port identities and power-gain extrema alongside CPU preparation time.
Syntax checking passed; no numerical or performance result is claimed yet.
The active full-domain run was re-polled successfully at 5590 solves.

Render benchmark sampling guards also explicitly set and record zero render
time limit and `use_layer_samples='IGNORE'`, in addition to disabling adaptive
sampling and denoising. The analysis rejects reports missing those effective
settings or enabling either alternative stopping/override mechanism. Python
syntax checks passed; existing archived timings are not rewritten.

The N512 build subsequently reached 7201 modal solves and 0.125 accepted
domain coverage, with 526840 matrix bytes in its in-progress report. This is
not a complete published cache. The dielectric optimization's comparison
tool, `cycles_diffraction_dielectric_modal_analysis.py`, now requires all
30 cases, identical port topology, finite operators/times and correct warmup
classification. It checks complete complex components and power-column L1
at 1e-9 and reports per-case median preparation times separately. Only syntax
checking has run; the candidate's numerical result remains pending.

After the full-domain run reached 10280 solves, candidate compilation and
numerical checks were allowed to overlap it to continue accuracy work. This
supersedes the earlier isolation condition: `tensor_modal512_cache` elapsed
time is now diagnostic, not an isolated preparation benchmark. Its frozen
solver and executable are unchanged. Both real and baseline dielectric
benchmark binaries compiled successfully; the real candidate completed all
30 cases, while the baseline comparison is pending. The analysis now provides
`--numeric-only` to suppress timing comparisons for these contended runs.
Source/binary hashes and these conditions are in
`real_modal_candidate_provenance.json`. No candidate promotion or speedup
claim follows from this execution.

The 30-case real/complex dielectric comparison completed and passed the
unchanged 1e-9 gates: maximum complex-component difference 9.7657181e-10,
maximum power-column L1 7.4106983e-10. Timings are suppressed in
`dielectric_real_numeric_comparison.json`. The complex error is close to the
gate; an additional candidate solve of the difficult 580 nm conical query
through N512 is running before promotion can be considered.

The N512 candidate comparison completed and failed the unchanged 1e-9
equivalence gate: complex-component difference 2.6624798e-9, power-column L1
2.0122433e-9. The candidate remains isolated and unpromoted. N512 physical
boundary residuals are 8.20e-16 (baseline) and 9.77e-16 (candidate); maximum
absolute unit-power-gain deviations are approximately 1.96e-10 and 3.38e-10.
These diagnostics do not establish an absolute complex-amplitude reference or
prove either eigensolver wrong. The failed comparison is retained in
`modal_failed_query_real_comparison.json`; no tolerance was changed. The
single-query analyzer now supports same-query, same-resolution cross-solver
comparisons through `--compare-to`, including source-report hashes.

An isolated independent complex eigensolver experiment uses the installed
Apple Accelerate LAPACK `zgeev_` interface (new LAPACK API), preserving the
same modal matrices and zero-ky decomposition. It compiled and completed the
580 nm conical query through N512. Against the original Eigen complex solver,
N512 differences are 7.2763080e-9 in complex components and 5.4473487e-9 in
power-column L1, failing the same 1e-9 equivalence gate. N256 differences are
4.8197938e-10 and 3.6905425e-10. This is evidence of high-resolution solver
sensitivity, not proof that the original or alternative is the absolute
reference. Neither alternative is promoted; no gate was relaxed. The
candidate, source/binary provenance and raw/comparison results use the
`lapack_modal_candidate` and `modal_failed_query_lapack` prefixes. Execution
overlapped the full-domain build and provides no isolated timing result.

The full-domain N512 build is terminal and failed modal convergence after
23506 actual reference solves, 7975 exact-query reuse hits and 75% accepted
coverage. It published zero cells. The 2766872 reported bytes describe partial
construction only; elapsed 2037.002 s is diagnostic because candidate jobs
overlapped. Peak reference-matrix cache memory was 47801600 bytes.

The failing query is wavelength 775.7257080078125 nm,
kx=-0.071163542268691718, ky=-0.27090960741043091. N128→256 power-column
L1 was 0.000305155475; N256→512 was 0.0000739359470; N128→512 was
0.000379091422. Thus the unchanged 0.00025 consecutive/spanning criterion
was not met. Full exported-cache validation cannot start without publication.
An isolated diagnostic extends only the direct solver's modal budget to N768
and repeats this query at N256/512/768. Production limits and tolerances remain
unchanged; no successful higher-resolution result is claimed yet.

The isolated N768 diagnostic completed successfully. N256→512 power-column
L1 is 7.3935947e-5, N512→768 is 1.3881390e-5, and the spanning N256→768
difference is 8.7817337e-5. All meet the unchanged 0.00025 criterion for this
query. This is empirical convergence evidence, not an absolute error bound.
Raw operators and analysis are retained as `modal_failed_query775_768*`.

A full-domain N768 cache experiment now uses isolated solver/driver snapshots
that raise only their maximum modal budget. It preserves the original Eigen
complex solver, all accuracy criteria, reference reuse budget, node/depth
limits, profile and domain. Production source and default limits remain
unchanged. Output prefix: `tensor_modal768_cache`; full-domain success and
exported-cache validation remain pending.

The BDPT log recurrence now has explicit delta-event treatment: exclude new
connections at the singular vertex while multiplying existing strategy ratios
by cosine and reverse/forward discrete mass. Both camera and light singular
branches call this shared recurrence. The independent explicit-path-product
tests now draw unequal forward/reverse delta masses instead of assigning both
one; all 22 `BidirectionalPDF` tests passed in a standalone GTest build.
This validates recurrence algebra, not completed grating integration: singular
callers still initialize reverse PDF from the forward PDF. Reverse shading,
closure-mixture mass evaluation and renderer checks remain required.
The initial standalone compile lacked the bundled TBB include path; rebuilding
with that path succeeded. Logs: `bdpt_delta_probability_build.log` and
`bdpt_delta_probability_test.log`.

Full Blender/device rebuild, including Metal kernel generation, and the CPU
device regression passed after recurrence integration. A fixed-1024-sample
Metal BDPT dielectric furnace render is running outside the sandbox, with
adaptive sampling and denoising disabled. It is an integration/analytic
radiance check, not a benchmark or proof of unequal diffraction mass handling.
The N768 cache build remains active; its elapsed time is diagnostic because
compilation and this regression overlap it.

The Metal BDPT furnace completed and passed: mean RGB
[1.0000655932853988, 1.0000483201984025, 0.9995937067287741], maximum
unit-radiance channel error 0.000406293271. The saved EXR and Standard-view
PNG were inspected; the furnace is uniformly white. The GTest suite was also
rerun with both original unit-mass delta events and asymmetric branch masses
retained; all 22 tests passed. This remains a recurrence/integration check,
not evidence that reverse grating closure-mixture probabilities are wired.

Added an explicit `--audit-delta-mixture` device-test diagnostic. It constructs
two identical deterministic physical grating closures and checks the actual
`_surface_shader_bsdf_eval_mis` result for their shared outgoing direction.
The current implementation reports mass 0.5 instead of the analytic marginal
mass 1: ordinary `bsdf_eval` contributes zero for the second delta closure.
This is a reproduced probability-accounting defect, not a PT/BDPT brightness
comparison. The audit deliberately fails until the mixture implementation is
corrected; default device regression remains separate.

An initial broader sampler fixture crashed before reporting a probability;
the debugger stalled at launch and was terminated. Narrowing to the actual
mixture-PDF function produced the reproducible numeric failure above. The
original crash log is retained and is not counted as physical evidence.
Logs: `delta_mixture_audit_before.log`, `delta_mixture_audit_debug.log`, and
`delta_mixture_audit_pdf_before.log`.

The coincident physical-grating mixture defect is corrected: sampled grating
events explicitly evaluate other physical grating atoms, accumulating their
power and probability mass. Ordinary BSDF evaluation still returns zero for
these delta closures. The atom evaluation applies the sampler's geometric
hemisphere and reflection shading corrections. The regression now passes both
the mixture-PDF function and the full closure sampler across the existing
rotated/both-side routing fixtures, checking marginal mass and summed power.
Full Blender/device rebuild including Metal kernels passed. This does not yet
handle coincidence between a grating and a different delta-closure family,
nor supply reverse-shader masses to BDPT; those remain open.

The audit crashes were traced to omitted object data in its synthetic kernel
globals. Supplying a zero-initialized fixture object allowed both the previous
direct audit and the full sampler to run. The final diagnostic passed with
log `delta_mixture_audit_after.log`; earlier failures are retained.

Explicit sampled-atom evaluation now covers sharp GGX/Beckmann reflection,
refraction and glass using their existing Fresnel coefficients and deterministic
directions. Grating-selected mixture evaluation can include these atoms;
zero-roughness microfacet-selected evaluation can include physical gratings.
The evaluator rejects rough microfacets and nonmatching directions, and does
not alter ordinary continuous BSDF evaluation. All 68 sampler/atom comparisons
passed over both IOR directions, several angles and branch endpoints, along
with the physical grating-mixture audit. Full Blender/device/Metal rebuild
passed after correcting an explicit private pointer qualifier required by
Metal. Logs: `build_microfacet_delta.log` (initial compiler failure),
`build_microfacet_delta_retry.log`, `microfacet_delta_device_test.log`.
Rendered grating/microfacet overlap, other delta families, near-singular
threshold consistency and reverse-shader BDPT mass integration remain open.

Microfacet-selected/grating mixture classification now uses the sampler's
actual `roughness_is_almost_specular` predicate through a shared closure-type
check, rather than requiring exactly zero roughness. The test includes alpha
0 and 1e-5: all 136 atom/sample comparisons and the mixture audit passed;
full Blender/Metal rebuild passed. Rough near-unit-IOR transmission and other
delta families still require separate treatment.

The physical scene generator now supports `--mix-mirror`, saving an equal
physical-grating/unit-mirror mixture with an explanatory scene text and
serialization checks. Its zero reflection orders overlap while nonzero
grating orders remain distinct. A fixed-1024-sample Metal BDPT unit-environment
render is running under `mirror_grating_overlap_metal_bdpt`; its analytic
expected radiance is one. Adaptive sampling and denoising are disabled.
This is an overlap integration fixture, not a performance benchmark or a
complete reverse-probability validation.

The grating/mirror Metal BDPT render completed: mean RGB
[1.000104307928268, 0.9999928738252493, 0.9994355647213524], maximum
unit-radiance channel error 0.000564435279, passing the 0.005 smoke gate.
The Standard-view PNG was inspected and is uniformly white. Scene, EXR,
JSON, PNG and source/binary provenance are retained under the overlap prefix.
This checks the mixed-material furnace; a caustic/strategy-density test is
still required for reverse BDPT probabilities.

Sharp thin-glass transmission is now included in sampled-atom evaluation:
its deterministic direction is -wi with unit conditional mass, matching the
existing sampler. Rough thin glass is excluded. All 160 atom/sample comparisons
and the physical mixture audit passed, followed by a successful full
Blender/Metal build. An initial duplicate case introduced in the wrong switch
was caught by compilation and corrected; both build logs are retained under
`build_thin_glass_delta*`. The test log is `thin_glass_delta_device_test.log`.

Added a discrete mixture evaluator including all closure-selection weights
in its denominator and only supported atomic contributions in its numerator.
The device audit verifies that a diffuse closure dilutes selection probability
without adding atomic power. Sampled grating mixtures likewise exclude
continuous densities from their atomic numerator.

For supported singular events in shaders containing a physical grating, both
camera and light BDPT paths now calculate forward mixture mass and obtain
reverse mass by reevaluating the shader from the outgoing direction at the
same wavelength. Reverse texture-cache misses replay before state changes.
These masses feed the tested singular MIS recurrence. Full Blender/Metal
rebuild and device audit passed; a fixed-sample Metal BDPT overlap render is
running under `reverse_mass_overlap_metal_bdpt`.

Other delta families, rough near-unit-IOR transmission, mixed-normal adjoint
contributions and caustic/path-strategy convergence tests remain outstanding.
The previous furnace result does not establish their correctness.

Angular-reference analysis now accepts the supported transport modes and
`--mix-mirror` scene reports. Mixed reference RGB is half the direct Maxwell
grating result plus half the spectral neutral reference only for central
reflection; the mirror contributes zero in other angular regions. Signed
RGB and separate quadrature/modal differences are retained. The modified
analysis successfully processed the existing six unmixed angular controls;
new mixed angular renders remain pending. Output:
`angular_analysis_mixture_extension_regression.json`.

The reverse-mass Metal BDPT overlap render completed and passed its analytic
check: mean RGB [1.0001271340661333, 1.000013128581486,
0.9994525704205444], maximum unit-radiance channel error 0.000547429579.
This is the fixed-sample integration fixture, not a performance measurement
or proof of caustic/path-strategy convergence.

Removed redundant forward atomic-mixture evaluation in both BDPT directions:
the closure sampler already returns that mass, and supported delta closures
bypass continuous guiding proposals. The audit now compares sampled and
explicitly evaluated forward mass even after adding a diffuse selection
component. Full rebuild and device audit passed. This removes an extra
lookup per supported bounce but has no measured render-speed claim yet.

The angular suite runner now accepts explicit output/provenance paths,
transport, sample count, angular-only and mirror-mixture options, retaining
artifact overwrite protection and per-run source hashes. Six 1024-fixed-sample
Metal BDPT mixed angular controls are running under `mixed_angular_metal_bdpt`.
Their raw signed RGB will be compared with direct spectral Maxwell references;
single-seed differences remain diagnostics rather than convergence proof.

The first mixed angular render failed before producing an image: Metal's
`PSO_SPECIALIZED_LIGHT_CACHE` pipeline for `integrator_bdpt_light_generate`
exceeded available stack space. The suite stopped with a retained failure
manifest; it is not a radiometric failure. Generic Metal compilation had
passed, demonstrating why runtime specialization must also be exercised.
An explicit Metal noinline boundary around atomic BSDF evaluation is now
being built to keep the grating matrix workspace out of each mixture caller.
The specialized scene must pass before this can be considered resolved.

The explicit noinline build and CPU atomic-mixture audit passed, but the
runtime retry in `mixed_angular_noinline_retry.log` failed with the same
specialized light-kernel stack-space error. This attempted fix is insufficient;
no rendered image or performance measurement was produced by that retry.

Benchmark policy: adaptive sampling is disabled, with fixed scene samples,
denoising disabled, no render time limit, and layer sample overrides ignored.
The repeat-render runner records and verifies effective sampling settings
before and after every render; analysis rejects reports lacking that evidence.
Concurrent cache preparation and correctness checks are not isolated timing
measurements. Performance comparisons must be run without competing workloads.

The next stack-space change separates inline channel-count dispatch from
outlined fixed-size tensor evaluation. Previously, outlined recursive dispatch
could retain matrix scratch for each smaller channel count on the call chain.
Each selected evaluator now owns only its own chart workspace; the mathematical
operations and channel capacity are unchanged. Full Blender/device build and
the device integration audit passed. The previously failing mixed central
reflection scene then completed on Apple M5 Metal with 1024 fixed samples.
Its raw RGB is compared without rescaling against the spectral reference in
`mixed_angular_leaf_dispatch_reference_comparison.json`; this single seed is
not a convergence acceptance test or an isolated performance measurement.

At the user's request for a final candidate suite, 417 source/binary hashes are
frozen in `final_candidate_leaf_dispatch_provenance.json`. The sequential
runner schedules 68 physical-node controls and appearance renders across PT,
BDPT, guided PT and guided BDPT. It stops on failure and retains every log.
Preview export verifies fixed sampling and preserves the fixture display
transform. Source changes invalidate the suite. The existing Eigen solver is
retained: experimental alternate eigensolvers have not passed the accuracy
gate. This is a candidate selection for testing, not a claim of fastest
validated performance or completed physical support.

`scene_archive_20260926/index.html` catalogs 255 historical saved scenes with
their available artifacts and recorded statuses. It explicitly distinguishes
archived experiments from current-version certification. The final suite is
separate; cross-object coherence, full modal convergence and the previously
listed pipeline gaps remain unresolved.

The frozen candidate's new dispatch passed 4096 random mirrored-domain cache
queries against direct N16 Maxwell solves (seed 482903, float duty cycle).
Maximum power-column L1 error was 0.0008385890092 against the unchanged 0.001
gate; maximum conservation error was 0.000007092953 against 0.0001. This checks
dispatch/interpolation at the same modal truncation, not absolute Maxwell
convergence. Report and provenance use the `leaf_dispatch_random_validation`
prefix. The current-suite analysis script also verifies artifact hashes and
effective sampling settings, computes analytic control errors, and retains
signed angular-reference, quadrature and modal differences without using PT
as ground truth. Single-seed angular comparisons remain explicitly unaccepted
until convergence evidence is available.

The first complete six-region PT partition in the frozen final suite sums to
RGB (1.0000655962242961, 1.000048320330259, 0.9995937203608838).
Compared with the neutral RGB integral from the N64/subdivision-8 spectral
reference, its largest component discrepancy is 0.000002104707. This is a
lossless angular-partition check, not proof of individual-order convergence:
order-efficiency errors can cancel in the sum. The analysis retains that
distinction under `angular_partitions` in `final_candidate_review_02`.

An independent direct spectral furnace integration now audits the exact saved
generic-metal profile (including float-rounded n and duty). The new driver
`cycles_diffraction_metal_furnace_reference.cpp` integrates normal-incidence
reflected power only; absorbed substrate flux is not treated as transmitted
radiance. At identical subdivision-4 quadrature, the completed N16 and N64
stages differ by up to 0.003926700919 in linear RGB. N128, a separate quadrature
refinement, and N256 remain running. This already demonstrates why agreement
between the N16 renderer and an N16 direct reference is only a pipeline check,
not sufficient evidence of modal convergence. No final-candidate physical
accuracy acceptance is inferred from that agreement.

The N128/subdivision-4 metal furnace stage also completed. Its largest RGB
change from N64 is 0.0002389151273, smaller than N16-to-N64 but still requiring
the planned quadrature and N256 checks. The frozen leaf-dispatch executable,
complete Cycles source tree and diffraction fixture scripts are preserved in
`candidate_leaf_dispatch_snapshot` (1029 hashed files) for later comparisons.
The copied executable passes `--version` with the configured runtime library
and resource paths. This snapshot preserves the measured candidate; it does
not certify its unresolved modal accuracy or broader feature completeness.

The frozen-suite PT CD/DVD presentation render completed at 960 x 691 and
1024 fixed samples. Inspection shows the two pitches producing distinct radial
spectral bands under the two white strips. The material is still the generic
lamellar conductor, without pits or cover; this is appearance evidence, not a
converged realistic-disc certification. Its recorded 898.81 seconds includes
preparation and overlaps CPU reference jobs, so it is not an isolated render
benchmark. The suite proceeded to the PT indirect-light scene.

The artifact audit found that the suite's EXR-preview exporter superseded the
disc fixture's original PNG at the same path. Its original PNG hash therefore
does not match the current preview. The original linear EXR hash does match
both records. Analysis now explicitly records the original PNG as unavailable,
retains both PNG hashes, and verifies the displayed PNG against the separate
preview report and unchanged source EXR. It does not waive blend/EXR checks or
claim the regenerated preview is the original fixture PNG. Future preview
export should use a distinct filename; the active suite's frozen script is
left intact. `final_candidate_review_pt_discs` records this provenance issue.

The generic-metal furnace reference study is complete and source hashes match
its recorded provenance. The PT render differs from direct N16 by at most
0.000007832675, but from N256/subdivision-4 by up to 0.004303542457. N128-to-N256
changes integrated RGB by up to 0.000163066943; subdivision 4-to-8 at N128
changes it by only 0.000003530481. Thus the dominant observed discrepancy is
modal truncation, not the wavelength quadrature tested here. This rejects
same-N16 agreement as evidence of a converged material response. N256 itself
is a finite-resolution reference, not an exact solution or a full-domain bound.
`final_candidate_review_metal_convergence` retains all signed differences.
The current fixed-N16 production cache is not qualified as the final accurate
version on this evidence; its completed renders remain pipeline/appearance
tests while adaptive modal preparation is being validated separately.

The selected-wavelength metal convergence audit completed with all seven
normal-incidence queries (425..725 nm in 50 nm steps) exhausting N512 at the
unchanged 0.00025 complete-power-column criterion. It retains every adjacent
and spanning difference and publishes no unconverged operator. An isolated
N768 audit uses the already preserved higher-budget solver candidate; no
production limit has changed. This is seven queries, not full-domain proof.

All 17 PT jobs in the frozen suite have now rendered. Inspection of the
indirect-light scene shows severe spectral sampling noise at 1024 samples;
finite pixels and a completed render do not constitute a presentation-quality
pass. The suite has proceeded to BDPT. Its radiance will be assessed against
physical/convergence evidence, without normalization to the noisy PT image.

A one-second stack sample of the active BDPT relief-metal test identified
`DiffractionSmoothBsdfNode::prepare` / `diffraction_grating_build_cache` on the
busy render thread, with recursive cache fitting below it. Metal compiler
services were idle. The retained `bdpt_metal_preparation_stack_sample.txt`
therefore identifies material preparation as the work active at that instant;
it is not a full-duration profile or a GPU render-time measurement. Repeated
process-level test durations must not be interpreted as integrator timings.

The frozen candidate's three BDPT furnace jobs completed. Flat-metal Fresnel
and dielectric energy pass the same analytic smoke thresholds as their PT
controls (maximum RGB errors 0.000367822725 and 0.000406293304 respectively).
The relief-metal RGB remains about 0.0043 below the N256 reference, consistent
with the identified N16 modal error; agreement with PT does not turn that into
a physical accuracy pass. The suite has proceeded to BDPT angular controls.
`final_candidate_review_bdpt_furnaces` retains the checked artifacts and signed
reference comparisons.

All seven selected normal-incidence metal queries pass the isolated N768
convergence audit at unchanged power tolerance 0.00025. Maximum adjacent and
spanning differences are 0.000038202381 and 0.000171480758; source hashes were
verified. `metal_modal768_audit_summary.json` records the full result. This
supports the increased budget at these queries, not full-domain accuracy or
a rigorous truncation bound. Two additional conical queries at wavelengths
450/650 nm and kx=0.2, ky=0.3 are now running with the same criterion. Production
limits and the running Blender suite remain unchanged.

Both selected conical metal queries completed at N768. At kx=0.2, ky=0.3,
the 450 nm adjacent/spanning differences are 0.000033541961 / 0.000161063134;
the 650 nm differences are 0.000052789363 / 0.000227585691. Both satisfy the
unchanged 0.00025 criterion and source hashes match the frozen audit sources.
`metal_modal768_conical_summary.json` records the result. These two queries
extend the focused normal-incidence audit; they do not establish a full-domain
cache error bound or complete production readiness.

The six plain-grating BDPT angular controls completed in the frozen suite.
Their summed RGB differs from the lossless neutral spectral integral by at
most 0.000002104520; the largest individual-region difference from the N64
reference is 0.000393480439. These remain single-seed diagnostics. An
independent-seed PT central-reflection check is now running on the preserved
candidate executable and runtime sources at 4096 samples, seeds 29/47/83/113,
with adaptive sampling, denoising, time limits and layer overrides disabled.
It overlaps the correctness suite and therefore provides no isolated timing
evidence. Production source hashes remain unchanged.

The attempted concurrent independent-seed run was intentionally terminated
before any seed EXR completed: the 16 GB host had approximately 12.9 GB wired
and only 115 MB free, with both Blender processes largely idle. Only the newly
started seed process was sent SIGTERM; exit code 143 and its artifacts are
retained under `final_candidate_pt_reflection_central_seeds/interruption.json`.
This is a resource-pressure interruption, not a physical rendering failure.
After it exited, free memory rose to about 4.4 GB and the original suite
advanced. Future seed checks will run sequentially in a new directory. The
interrupted attempt supplies no sampling statistics or benchmark evidence.

All 15 frozen-suite BDPT measurement controls are complete. The equal
grating/mirror six-region partition differs from the lossless neutral integral
by at most 0.000117118809 RGB. `final_candidate_review_bdpt_controls` verifies
the current 32 completed jobs and retains the individual signed comparisons.
The partition remains a single-seed diagnostic; it does not resolve the metal
modal error or validate general caustic transport. The BDPT disc render is next.

## Selected metal complex-amplitude diagnostic

The standalone metal modal audit now records adjacent and spanning complex-matrix
Frobenius differences alongside its existing maximum power-column L1 differences.
An optional argument limits its diagnostic resolution budget (64 through 512);
the default remains 512 and the power criterion remains 0.00025.

`tests/output/diffraction/metal_phase_diagnostic_64/` retains source and executable
hashes, the seven normal-incidence queries at 425–725 nm, and exit status 1. All
seven failed power convergence at N64, as expected from earlier modal evidence.
N32-to-N64 complex differences range from 0.00233196 to 0.00531126; N16-to-N64
differences range from 0.00914014 to 0.02178251. These are matrix norms, not phase
angles or relative percentages. Complex differences are diagnostic only here; no
complex convergence gate was enabled. No coherent renderer or full-domain phase
accuracy is demonstrated by this run. The frozen Metal suite sources are unchanged.

## Fresh cache validation with modal-refined references

`cycles_diffraction_tensor_cache_validate.cpp` now accepts `--modal-reference`.
Each fresh sample must satisfy the existing adjacent, preceding-adjacent and
spanning power criterion of 0.00025 before its reference is used. Failure exits
without a success report. `--reference-half-orders` sets the budget; accepting
768 in the CLI does not change the linked production solver's 512 limit. An
isolated higher-budget solver is needed to use that value. The ordinary fixed
reference path remains explicitly identified in JSON.

Functional evidence is retained in `modal_reference_validator_check/`: 16 fresh
queries, seed 937117, on the prior N16 mirrored dielectric DVD cache. Fixed N16
references yield maximum L1 error 0.000105594 and exit 0. A modal budget of 128
fails reference convergence at query 13 (exit 4). With budget 512 every reference
converges, using at most N256 and 50 solves in total, but the old cache fails its
unchanged 0.001 L1 gate: maximum error 0.001231738 at 549.167114 nm (exit 1).
Conservation error is 6.23986e-7 in both completed comparisons. This directly
separates same-resolution interpolation agreement from accuracy against refined
references; it is not a full-domain proof or coherent-amplitude validation. No
production source, frozen suite input or threshold was changed for this audit.

## Revised user priority: fast default and optional realistic mode

The user explicitly revised the accuracy/performance tradeoff: default diffraction
should be fast and visually convincing rather than maximally physically accurate,
with an optional realistic UI setting. They also requested GPU cache construction.
The prior 0.001 reference criterion remains useful for the realistic candidate, but
it is no longer a user-mandated acceptance threshold for the fast approximation.
Approximation error must still be measured and reported, not hidden. Spectral
behaviour, PT/BDPT/guiding compatibility, optional inter-object coherence, fixed
sample isolated benchmarks and representative images remain in scope.

Existing scalar phase-screen reflection code in `bsdf_diffraction.h` provides a
possible fast building block and already evaluates on the rendering device. It
is explicitly not a full relief-boundary Maxwell solution, and reflection alone
does not satisfy the transmission scope. No fast/realistic selector is exposed
yet. UI changes must be connected to working implementations rather than merely
renaming the existing fixed-N16 cache. GPU construction also requires a measured
implementation; no speedup is claimed from the requested change alone.

### Fast scalar disc candidate: first isolated timing

`tests/output/diffraction/fast_disc_isolated_pt_v1/report.json` records a Metal
PT benchmark of the saved scalar-reflection CD/DVD scene at 960 by 691 pixels,
1,024 fixed samples, adaptive sampling off, denoising off, no time limit and
layer sample overrides disabled. On the Apple M5 10-core GPU, the warm-up took
16.9826 seconds; the three subsequent render calls took 9.5818, 9.5652 and
9.5524 seconds (median 9.5652). Persistent data was enabled. These are elapsed
render-call times, not GPU timestamps. The identified competing CPU reference
experiment was suspended throughout and resumed afterwards. These measurements
are not a comparison against an isolated realistic-mode baseline.

The PT, BDPT and guided PT disc renders completed and their saved scene, EXR
and distinct preview hashes were checked in `fast_disc_review_v1/review.json`.
The material uses the existing GPU scalar phase-screen model with constant
normal-incidence metal reflectance, and requires no Maxwell cache. It retains
the broad CD and narrower DVD spectral bands, with visible sampling noise at
band edges. This is a reflection appearance candidate, not a qualified general
dielectric or coherent model.

The combined BDPT/guiding render failed Metal compilation with "Compute pipeline
exceeds available stack space" in `integrator_bdpt_light_generate`. This failure
also occurs with the fast material and remains a release blocker. An isolated
compiler experiment outlines the physical closure sample and delta-probability
entry points so their solver scratch is not copied into each BSDF dispatcher.
It does not change the scattering model or disable guiding. With that change,
the fast BDPT/guiding scene completed both an eight-sample compilation diagnostic
and a full 1,024-sample render (`fast_disc_stack_outline_v1` and
`fast_disc_stack_outline_1024`). The full render's scene, EXR and preview hashes
were verified and its preview inspected. Its 37.7544-second elapsed time includes
preparation and overlaps the CPU reference build; it is not an isolated benchmark.
The physical-cache regression is tracked separately in
`physical_stack_outline_seeds_v1`; the fast scene cannot establish its result.

The repeated-render benchmark script now accepts `--transport` for all four
transport configurations and records both BDPT and guiding switches in every
run's checked sampling settings. Its default remains PT for existing callers.

`--material undiffracted` provides a scalar-model workload control by setting
only unlinked, positive Glossy diffraction weights to zero. It preserves the
remaining node graph and records the affected materials, nodes and original
weights. Linked weights are rejected rather than silently overriding their
graphs. This control changes light paths and does not establish equal-image or
equal-noise performance; isolated measurements of it are still pending.

The physical-cache regression produced a completed seed-29 render with BDPT and
guiding enabled. Its mean RGB is (0.021850284, 0.011022508, 0.000294164), compared
with the N64/subdivision-8 central-reflection reference
(0.021859659, 0.010964496, 0.000286973). The maximum component difference is
0.000058013. This single-seed comparison is not a statistical convergence result;
the remaining independent renders are tracked by their run manifest.

New suite preview exports use a distinct `.preview.png` and refuse to overwrite
an existing preview or provenance report. The analyzer verifies both original
fixture artifacts and the separately named preview. Its legacy path retains the
earlier suite's explicitly reported overwritten-PNG provenance defect. Export
of the completed seed-29 EXR succeeded and both its unchanged source EXR hash and
new preview hash were verified. Reanalysis of the previous frozen suite still
reports its original stopped-on-failure outcome (54 successful, one failed).

### Completed physical-cache stack regression and failed refined cache build

The outlined physical closure completed all three BDPT/guiding central-reflection
renders at 1,024 fixed samples (seeds 29, 47 and 83). The independently checked
aggregate in `physical_stack_outline_seeds_v1/comparison.json` has mean RGB
(0.021832016, 0.010994827, 0.000288004), with between-seed standard errors
(0.000012501, 0.000022222, 0.000005514). Against N64/subdivision-8, the errors
are (-0.000027643, 0.000030332, 0.000001032), or (-2.21, 1.36, 0.19) estimated
standard errors. Three seeds provide limited uncertainty evidence; this is a
completed stack/transport regression, not full physical certification. No image
brightness normalization was used.

The N768 refined CPU tensor-cache experiment is terminal and failed:
`tensor_modal768_completion.json` records exit 1 and the maximum-depth tolerance
failure. It used 48,626 reference solves, reported 92.2871% construction coverage,
and took 14,212.7 elapsed seconds (including a temporary benchmark suspension).
No `.bin` cache was published. Its partial coverage must not be rendered as a
complete model or described as a successful cache build. Repeating this exact
experiment unchanged is not justified; its interpolation and domain subdivision
strategy needs investigation before another high-cost attempt.

### Isolated matched fast-mode PT benchmark after the stack fix

`fast_disc_matched_bench_v1/comparison.json` verifies the scene, script, binary
and 433 kernel-source hashes for the outlined-kernel candidate. At 960 by 691,
1,024 samples, adaptive sampling off and denoising off on the M5 10-core GPU,
the three warmed persistent-data render times were 7.5629, 7.6096 and 7.5935
seconds with diffraction, versus 7.4891, 7.4902 and 7.4813 seconds with only the
two diffraction weights set to zero. The median ratio is 1.01394 (1.394% measured
overhead). First-run times, excluded from these medians, were 25.3590 and 24.1753
seconds. No other task render or reference-build process was active.

This is a scene-specific workload comparison, not an equal-image/equal-noise
comparison or a statistical significance claim. The materials retain constant
normal-incidence metal reflectance and scalar reflection; these timings do not
apply to the physical cache, transmission or cross-object coherence. The earlier
9.5652-second PT measurement used the kernel before the outlining fix and remains
a separate historical result, not an interchangeable repeat of this candidate.

The remaining matched transport benchmarks completed and were checked by
`cycles_diffraction_fast_benchmark_analysis.py`. The consolidated artifact is
`fast_disc_all_transport_benchmark_review_v1/comparison.json`:

| Transport | Fast diffraction median | Disabled median | Measured difference |
|---|---:|---:|---:|
| PT | 7.594 s | 7.489 s | +1.39% |
| BDPT | 16.499 s | 16.295 s | +1.25% |
| Guided PT | 16.984 s | 17.595 s | -3.47% |
| BDPT with guiding | 36.431 s | 34.466 s | +5.70% |

All used the same input scene, executable, script and kernel manifest, with
fixed 1,024 samples, adaptive sampling and denoising off, and one excluded warm-up
followed by three measured runs. The combined-mode disabled runs ranged from
33.587 to 36.882 seconds, so the median difference should not be overstated.
Guided PT's negative difference is also only a workload observation: removing
diffraction changes paths and their wavelength requirements. These measurements
do not establish equal-noise efficiency or negligible overhead in every scene.

A separate fast reflective/transmissive scalar candidate now has executed
numerical reciprocity, passivity, flat-interface, total-internal-reflection and
sampling checks. See `cycles_diffraction_fast_interface_candidate.md`. It remains
outside the production renderer pending reference comparisons and integration;
the benchmark table above measures the existing reflection model only.
