# Diffraction scene coverage

## Dedicated caustic screen fixture

`tests/python/cycles_diffraction_caustic_screen.py` creates a 1 × 25 mm physical
grating strip and a separate diffuse receiver at x = 40 mm. A restricted finite
area source illuminates the strip while its direct cone misses the receiver.
At normal incidence the central +1-order ray reaches height
`z = 0.04 * sqrt((740 / wavelength_nm)^2 - 1)`; finite emitter size and strip
width broaden this geometric prediction. It is not an absolute radiance oracle.

All four PT/BDPT/guiding variants have passed save/reload and geometry checks in
`tests/output/diffraction/caustic_screen_saved_v2`. They have **not rendered** yet
and are additional fixtures beyond the running 68-job suite. The material is
a generic smooth lamellar conductor. This tests inter-object radiometric
caustics, not coherent interference. Raw linear EXRs and separately named
`.preview.png` files will be retained when rendered; brightness will not be
normalized to PT.

Physical correctness is the target, not equality with finite-sample PT.
See `doc/cycles_diffraction_retrospective.md` for the review of earlier
brightness-based verdicts. Configured bounce-limit support, sampling convergence
and approximation to full physical transport are separate validation questions.

## Current scene inventory (2026-09-26)

Paths below are relative to `tests/output/diffraction`. A saved scene, a
completed render and a correctness pass are distinct milestones.

| Saved scene / family | Purpose and presentation | Current evidence |
| --- | --- | --- |
| `physical_discs_metal_broad/physical_discs_pt.blend` | Side-by-side 120 mm CD/DVD gratings, radial grooves, two white strips revealing different spectral orders | Metal 960-pixel, 1024-sample render; generic conductor, no pits or cover; convergence unresolved |
| `blender_physical/flat_metal/flat_metal_both.blend` | Uniform planar zero-depth control for Fresnel reflectance | Metal measurement passed within reported sampling error |
| `blender_physical/relief_metal_bounded_reference/relief_metal_both.blend` | Finite-depth conductor under controlled illumination, separating efficiency from studio appearance | Metal/reference pipeline comparison completed; same modal truncation, not independent Maxwell convergence |
| `physical_indirect_sphere_50mm_clear/physical_indirect_pt.blend` | Grating tile, neutral room, occluder and sphere; finite physical source with verified ceiling clearance | Four modes × four seeds at 1024 samples in `physical_indirect_50mm_clear_transport`; BDPT remains approximately 11–13% brighter than PT |
| `indirect_diffuse_sphere_control/diffuse_pt.blend` and `mirror_50mm_seed11/mirror_pt.blend` | Same transport geometry with ordinary diffuse/mirror materials to isolate integrator errors | Paired PT/BDPT renders; mirror bounce-limit regression identified |
| `mixed_transport_saved/mixed_{pt,bdpt}.blend` | Equal diffuse / rough glossy (roughness 0.2) tile in the neutral indirect room; separate bounce budgets within one node graph | Saved and reloaded; node links and transport settings verified, not rendered |
| `enclosure_prefix_diffuse4_seed11/diffuse_{pt,bdpt}.blend` | Closed, uniformly emitting and reflecting enclosure; absolute radiance from a geometric series | First Metal render completed against analytic 0.4921875; independent-seed checks running. No diffraction enabled; shared transport control |
| `translucent_transport_raised_saved/translucent_{pt,bdpt}.blend` | Raised, tilted diffuse-transmitting tile aimed toward visible floor/rear receivers; distinguishes transmission bounce counting from diffuse render-pass classification | Saved and reloaded; node type, tile clearance/tilt and transport settings verified, not rendered |
| `blender_physical/dielectric_metal_furnace/relief_dielectric_both.blend` | Lossless reflected + transmitted energy measurement | Saved; cache budget failure prevents a completed Blender render |
| `blender_physical/angular_controls/*.blend` | Separate central, middle and outer transmitted-order acceptance | Three scenes saved/reloaded, not yet rendered |
| `visual_suite/*.blend` | Pitch, depth/duty, tangent orientation, roughness, weight, furnace, cover, studio and indirect charts | Rendered earlier provisional Glossy implementation; not validation of the physical node |

The original `physical_indirect_sphere_50mm` fixture is rejected because its
source intersects the ceiling. Use the `_clear` fixture above.
The original `translucent_transport_saved` tile was flush with the floor and
hid the transmitted receiver. Use the raised replacement above for visual tests.

Still required for the physical implementation: the parameter charts, roughness
and mixed-node cases, measured dispersive materials, covered discs/MNEE,
transmission caustics, volumes, motion blur, depth of field, AOVs and OSL
equivalence. General cross-object coherent transport is not implemented and
has no rendered validation scenes yet. Its required dedicated fixtures are a
two-path interferometer with a phase sweep, separate-object double-slit
diffraction with analytic fringe spacing, and finite bandwidth/coherence
controls that suppress fringes. Those must test optical path difference and
visibility numerically as well as show the fringes clearly.

Every completed feature needs an isolating measurement scene and, where useful,
a representative appearance scene. Comparisons must retain linear images,
settings, provenance, independent-seed uncertainty and explicit pass/fail
criteria; presentation lighting must not substitute for a physical reference.

## Physical Blender node controls

The new `ShaderNodeBsdfDiffraction` can now be created in Blender and saved
in `.blend` files. Three planar controls have passed save/reload checks:
flat conductor, finite-depth conductor relief, and lossless dielectric relief.
They are stored under `tests/output/diffraction/blender_physical`. Their JSON
reports distinguish `saved_not_rendered` from completed render checks; saving
a scene is not evidence that its rendering is correct. The flat conductor
has passed a Metal render against normal-incidence Fresnel reflectance
(maximum RGB mean error 0.000369). The finite-depth conductor has also rendered
on the actual Apple M5 Metal device; its RGB mean differs from the internal
direct-solver fixture by at most 0.0000177. That comparison uses the same modal
truncation and is a pipeline check, not independent Maxwell convergence.

Three additional saved/reloaded dielectric transmission controls live in
`blender_physical/angular_controls`: central, middle, and outer angular
acceptance. Generate these with `--case relief_dielectric --illumination
transmission --angular-region central|middle|outer`. They use the same masks
as the internal angular tests described below. These Blender files have not
yet been rendered; the internal test images must not be attributed to them.
Each file contains a `READ ME - measurement setup` text block explaining the
measurement. The uniform image is intentional: its raw mean measures power
in the selected orders without presentation lighting changing the answer.

Generate them with `tests/python/cycles_physical_diffraction_scene.py` and the
current Blender build. It uses an explicit linked +X tangent, a normal-incidence
orthographic camera, and unit environment illumination of both or one side.
Conductor indices are constant `0.9 + 6i`, explicitly a generic material rather
than measured aluminum. The dielectric has air on both sides and ridge IOR
1.5. These are measurement controls, not the finished CD/DVD presentation suite.

## Physical disc presentation fixture

`tests/python/cycles_physical_diffraction_discs.py` saves a separate physical
smooth-node studio scene with 120 mm discs, CD/DVD pitches (1600/740 nm),
radial tangent links and a white strip source. Save/reload and the first
640-pixel, 256-sample Metal pilot completed in `physical_discs_metal_pilot`.
Visual inspection shows radial color bands, but the source angle reveals
mostly violet on the DVD; lighting coverage and convergence need improvement.
This is not yet a validated realistic-disc reference. Run with `--output DIR
--device GPU --samples 1024 --resolution 960 --render` using the current
Blender build. Omit `--render` to save/reload only.

This is a lamellar grating with depth 150 nm, duty 0.41 and constant generic
conductor index 0.9+6i. It is not a recorded-pit model or measured aluminum,
and has no plastic cover or cross-object coherence. It tests disc-scale
mapping and appearance; planar controls remain the absolute-efficiency tests.

The `--lighting broad` preset adds a second identical white strip at the
opposite Y azimuth. Its 960-pixel, 1024-sample Metal render is complete in
`physical_discs_metal_broad`; visual inspection confirms additional red/green
DVD bands and clearer angular coverage. The original single-strip setup is
available with `--lighting pilot`. These are presentation diagnostics with
unresolved modal-convergence error, not final realistic-material references.

## Physical indirect transport fixtures

The physical presentation generator also supports `--case indirect`. It
replaces the room fixture's provisional material with the smooth physical
node and an explicitly linked +X tangent. The restricted white spotlight,
neutral receivers, occluder and sphere test light reaching other surfaces.
All four transport variants have passed save/reload checks. The corrected
physical sphere-emitter files are in `physical_indirect_sphere`; these saved
checks are not render convergence. `--emitter sphere` is now the default.
`--emitter compatibility` reproduces the original camera-facing finite-radius
light, which deliberately uses the camera-path fallback in this fork's BDPT.

`tests/python/cycles_physical_diffraction_transport.py` runs a saved physical
scene with matched independent seeds and `pt`, `bdpt`, `guided`, and
`bdpt_guided`. It keeps persistent scene data, saves each configuration and
linear EXR/NumPy/display PNG output, and verifies binary/source provenance.
Its manifest distinguishes settings validation from completed renders and
leaves convergence analysis explicit. Pixel arrays have bottom-left origin.
The first four-seed, 512-sample, 320-pixel Metal comparison completed in
`physical_indirect_transport_metal`, but its compatibility emitter did not
exercise forward BDPT emission. PT/BDPT pixels were identical across all four
seeds, consistent with that fallback. A non-diffraction diffuse control with a
physical sphere emitter produced distinct PT/BDPT output. The corrected grating
comparison completed in `physical_indirect_sphere_transport`. Its verified
four-seed results expose a substantial PT/BDPT brightness discrepancy. This
alone does not identify which estimator is physically correct. The separate
source audit found inconsistent configured bounce-limit support; paths beyond
those cutoffs may still represent physically valid light transport. These runs overlapped
other work and are not isolated performance benchmarks. Use a fresh output directory for each batch.

Example (arguments after Blender's `--`):

```sh
--input tests/output/diffraction/physical_indirect_sphere/physical_indirect_pt.blend \
--output tests/output/diffraction/my_indirect_comparison \
--transports pt bdpt guided bdpt_guided --seeds 11 29 47 83 \
--samples 512 --resolution 320
```

## Provisional Glossy scenes

The scene suite is part of the ongoing diffraction implementation. **Current
renders exercise the provisional Glossy closure, not the new Maxwell cache or
cross-object coherent transport.** A visually attractive image is not evidence
that the electromagnetic response is correct.

Generate the suite with this branch's Blender build:

```sh
install/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python-exit-code 1 --threads 6 --python tests/python/cycles_diffraction_suite.py -- \
  --case all --samples 512 --resolution 960 \
  --output tests/output/diffraction/visual_suite
```

Each case saves an editable `.blend`, unclamped 32-bit linear EXR, linear NumPy
RGB array, AgX PNG, and JSON with parameters, binary/source hashes, device and
render time. Source snapshots are retained beside the results. Denoising,
adaptive sampling, direct/indirect clamping and glossy filtering are disabled.
Annotations and the parameter-chart backing are camera-only. Chart swatches
are excluded from secondary/shadow rays so they cannot reflect in one another
or occlude the common source. Parameter charts isolate direct response with identical illumination at every swatch; indirect
transport has its own scene with the full bounce limit.

## Dedicated scenes in the generator

| Case | What the scene isolates | What to inspect |
| --- | --- | --- |
| `pitch` | 400/740/1000/1600/3200 nm pitch, all else fixed | Order separation and changing wavelength support across identical spherical normal fields |
| `roughness` | 0/0.025/0.075/0.15/0.3 facet roughness | Band broadening, singular endpoint and color contrast |
| `depth_duty` | Depth 0/75/150/300 nm crossed with duty 0.25/0.5/0.75 | Zero-depth control and redistribution of reflected orders |
| `orientation` | Linked tangent node at 0/45/90/135 degrees | Rotation of the angular response and node evaluation |
| `weight` | Weight 0/0.25/0.5/0.75/1 | Disabled endpoint and continuous mixing |
| `furnace` | Uniform unit white radiance; weight and roughness endpoints | Energy-gain regressions, using raw data and convergence rather than isolated bright pixels |
| `studio` | 120 mm CD and DVD, radial object-space tangent fields | Representative disc appearance under a white strip source |
| `covered` | Reflector embedded in a 1.2 mm IOR-1.58 cover | Refraction, internal incident index and shadow-caustic/MNEE integration |
| `indirect` | Restricted white illumination of a grating tile in an achromatic room | Spectral light reaching diffuse walls, occlusion and multiple bounces |

Use `--transport pt`, `bdpt`, `guided`, or `bdpt_guided` on the **same scene**
for estimator comparisons. The indirect room is the main transport fixture;
agreement on a directly lit disc alone is insufficient. Saved report metadata
states the transport variant. Final comparisons need multiple seeds and
converged references, not a single noisy preview.

Use `--seed 29` (default 11) to select an independent render seed. Save each
seed to a separate `--output` directory to preserve every image and report;
the report records the actual Cycles seed. Compare the same set of seeds and
sample counts across transport variants, and estimate uncertainty across
independent renders rather than treating neighboring pixels as independent.

Existing earlier fixtures are generated by `tests/python/cycles_diffraction_scene.py`:
CD/DVD discs, a planar grating option, the correctly embedded cover, and world
scale/device/OSL/transport variants. The old `pilot_cd_cover*` files predate the
corrected embedding and are not physical cover references; use `embedded_*`.

## Required additions as the implementation becomes renderable

The physical smooth-grating implementation also has actual Cycles Session
render fixtures in `tests/performance/cycles_diffraction_render_test.h`:
flat conductor, finite-depth reflective grating, and a 50/50 mix of that
grating with a diffuse closure, under a uniform white environment. These
exercise the internal shader graph node on CPU and Metal. Raw float PFM
images and JSON measurements are in `tests/output/diffraction/physical_images`.
They are numerical render fixtures, **not saved Blender scenes**, and do not
replace the explanatory scenes below. Their references use the same modal
solver as the cache, evaluated directly without cache interpolation; they do
not establish independent Maxwell accuracy or full angular-domain coverage.

`tests/performance/cycles_diffraction_render_convergence.py` runs independent
seeds and sample counts for the mixed fixture with PT and BDPT plus guiding.
Its manifest preserves per-render means, raw-image hashes and uncertainty
across seeds. This uniform-environment fixture does not establish correctness
of nontrivial bidirectional connections or the effectiveness of trained guiding.

The `--render-transmission` fixture uses a 740 nm pitch, 150 nm depth,
0.41 duty-cycle dielectric relief (ridge index 1.5, air grooves and air on
both sides). Its uniform-environment reference is analytic conservation of
radiance in equal exterior media, rather than the cached solver's prediction.
It exercises combined reflection and transmission, but cannot validate their
individual efficiencies or outgoing angles: an angular-screen scene remains
required. This fixture is also an internal Session scene, not a `.blend` file.

Use `--illumination reflection` or `--illumination transmission` with that
fixture to light only the corresponding environment hemisphere. These modes
compare the separate reflected/transmitted spectral energy against the direct
N=16 solver, without cache interpolation. The default `--illumination both`
retains the analytic conservation reference. Hemisphere tests distinguish
energy sent to the wrong side, but still integrate over all outgoing orders
and therefore do not establish individual order efficiencies or angles.

`--reference-samples N` controls the midpoint wavelength quadrature used for
the render fixture's reference (default 1024), independently of rendering
`--samples N`. Increase both separately when investigating a residual; a
fixed low-resolution spectral quadrature is not an exact reference for
narrow spectral features. The report records the actual `quadrature_count`.

For independent-seed hemisphere comparisons on Metal, use the convergence
runner with `--fixture transmission --illumination reflection --transports pt`
(or select the transmission hemisphere and other transport modes). For example:

```sh
python3 tests/performance/cycles_diffraction_render_convergence.py \
  --output tests/output/diffraction/dielectric_reflection_convergence \
  --fixture transmission --illumination reflection --transports pt \
  --seeds 29 47 83 --samples 4096 --reference-samples 1024
```

Choose a separate output directory for each experiment. The manifest records
the fixture, illumination and reference sample count and verifies these against
each render's report before including it in the statistics.

The internal transmission fixture also supports `--angular-region
all|central|middle|outer`. These regions mask the environment using the
absolute world-X component of its direction: central is below 0.25, middle
is between 0.25 and 0.9, and outer is above 0.9. The grating tangent is
explicitly world +X. At normal incidence in air the independent reference
uses `abs(order * wavelength / pitch)` to select the same angular region,
then integrates its direct-solver power over wavelength. With the fixture's
740 nm pitch, the 0.9 boundary cuts the first-order spectrum at approximately
666 nm; zero-order light lies in the central region. These are angular
acceptance tests, not a finite-aperture diffraction screen or a saved `.blend`
scene. Their sharp spectral cutoffs also require quadrature refinement.

These are **coverage requirements, not completed scenes**. No empty `.blend`
files stand in for unsupported functionality.

| Feature | Dedicated validation scene required |
| --- | --- |
| Absolute Maxwell reflection efficiency | Planar grating, calibrated angular measurements and independent RCWA reference; measure every propagating order |
| Transmission and unequal surrounding media | Transmission grating with an angular screen, plus the same grating immersed in a known medium |
| Spectral wavelength dependence | Actual narrowband spectral sources and a broadband sweep; RGB red/green/blue alone are not monochromatic references |
| Polarization | Matched s/p and rotated/elliptical input states with Jones-matrix reference measurements |
| Other applicable shaders and node combinations | Matched Glossy/Glass/Refraction/Principled and Mix/Add fixtures as their implementations are integrated |
| Mapping and geometry | UV/object/world tangent mappings, transformed instances, normal maps, displacement and equivalent world scales |
| Cross-object coherence | Separate-object two-aperture interference and an interferometer with controlled optical path difference |
| Coherence controls | Source bandwidth, source size, coherence length and phase sweeps, including the disabled option |
| Near-field diffraction | Finite aperture with screens at several distances, compared with a wave-optics reference |
| Full pipeline | Same feature scenes with PT/BDPT/guiding, CPU/Metal and CPU OSL where supported; motion/DOF/transparency/AOV and feature-off regressions as relevant |

Each feature needs both an explanatory view that makes the effect easy to see
and a measurement setup with a stated expected result. All future physical
material and coherence scenes must be re-rendered after integration; these
provisional images must not be relabeled as final results.

## Current results

The revised nine-case PT suite has been rendered on the M5 at 2048 samples.
The main presentation is `tests/output/diffraction/visual_suite/studio.png`;
`depth_duty.png` and `pitch.png` provide controlled parameter comparisons.
`covered.png` shows the embedded-cover fixture under useful illumination.
The indirect room is intentionally retained as a difficult convergence test;
its 2048-sample preview is not a converged reference.

Run raw control measurements with a Python environment containing NumPy:

```sh
python3 tests/python/cycles_diffraction_measure.py --directory tests/output/diffraction/visual_suite
```

The measurement script handles Blender's AUTO orthographic fit, including the
portrait depth/duty chart. It records regional statistics without interpreting
spatial variation as independent-seed uncertainty. Final energy and transport
claims require convergence and multiple-seed comparisons.

## Latest failure diagnostics (2026-09-26)

The full-domain dielectric furnace saved successfully but failed during cache
preparation after exhausting 65,535 nodes at tolerance 0.001. It produced no
render and no partial cache was published. See
`blender_physical/dielectric_metal_furnace/failure.json`.

The physical sphere-emitter room completed all four transport modes with
four independent seeds, 512 samples and 320-pixel width. Hashed artifact and
receiver-mask verification passed. BDPT minus PT whole-image mean RGB was
(0.08023, 0.08044, 0.08185), with paired-seed standard errors
(0.00187, 0.00027, 0.00226). This is a serious unresolved discrepancy.

Dedicated built-in diffuse and smooth-mirror versions of the same room isolate
transport from diffraction. The 64-sample mirror control reproduced the excess:
PT mean 0.06228, BDPT mean 0.12801. An isolated Metal kernel candidate retaining
prior-vertex MIS alternatives after a singular event reduced BDPT to 0.06573,
with unchanged PT. One seed at 64 samples is diagnostic evidence only, not a
statistical pass; this candidate is not yet promoted to production. Outputs are
in `indirect_mirror_sphere_control` and `indirect_mirror_mis_candidate`.

Cross-object coherence scenes remain required, not implemented or validated.
Their eventual suite must include separate-object two-aperture interference,
an optical-path sweep interferometer, finite-bandwidth/source-size visibility
sweeps, and the coherence-disabled control, with analytic/wave-optics references.


A separate 50 mm radius physical sphere-source room has been saved and
reload-checked in `physical_indirect_sphere_50mm`. Generate it with
`--case indirect --indirect-source-radius-mm 50`; the default remains 5 mm.
It is not yet rendered. This variant is intended to improve visibility and
convergence of indirect bands; the original small-source regression is retained.
The source size changes the physical problem, so its numerical results must
not be compared directly against the 5 mm fixture as estimator agreement.


The initial 50 mm fixture was rejected: its original z=0.48 source center
put part of the physical sphere above the z=0.5 ceiling. Its batch was
cancelled, and no result is accepted. The corrected generator lowers the
source to z=0.44 (10 mm ceiling clearance), re-aims at the tile, records its
position and verifies clearance after reload. The replacement scene is
`physical_indirect_sphere_50mm_clear`; its fresh Metal batch and hashed masks
are `physical_indirect_50mm_clear_transport` and
`physical_indirect_50mm_clear_masks`. Save/reload and mask generation passed.

The rebuilt lossless tensor material now has a completed PT Metal furnace
control at `tests/output/diffraction/tensor_dielectric_metal_furnace/relief_dielectric_both.blend`.
Its EXR mean differs from analytic unit radiance by at most 0.0004063 at
1024 samples; the saved PNG is visually uniform white. This establishes the
furnace smoke check only. Additional transport and angular-region controls
are tracked in `tests/output/diffraction/tensor_blender_controls/manifest.json`;
consult each status before treating it as rendered or validated.
# Grating/mirror overlap control

`tests/python/cycles_physical_diffraction_scene.py --mix-mirror` creates an
equal physical-grating/unit-mirror mixture. The zero reflected grating order
overlaps the mirror direction; higher orders remain distinct. The saved scene
contains measurement instructions. With relief dielectric and illumination
from both hemispheres, the analytic outgoing radiance is one.

The fixed-1024-sample Metal BDPT run in
`tests/output/diffraction/mirror_grating_overlap_metal_bdpt` passed its analytic
smoke check (maximum mean-channel error 0.000564435279); its PNG was visually
inspected. Adaptive sampling and denoising were disabled. This is a uniform
radiometric control, not a caustic test or a performance benchmark.
