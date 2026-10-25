# Diffraction BSDF: Grating model (port of Al-Toto/blender `diffraction-bsdf`)

Status: done (2026-10-25), validated on an M5 Mac. Uncommitted work in the working tree.

## Sources

* Branch `diffraction-bsdf` of https://projects.blender.org/Al-Toto/blender, commits
  `7eb9ce789f7`, `8681fd48d91`, `ca7b0ef6a83` (fetched locally as `refs/altoto/diffraction-bsdf`).
  Model: Sun et al., "Rendering Iridescent Colors of Optical Disks" (EGWR 2000).
* Thread https://blenderartists.org/t/wip-cycles-diffraction-grating-shader/1652231 (CD with
  rectangular / sinusoidal 1D grooves, 2D sinusoidal "screen", CD hero shot, cloth, sign).

## What each implementation has

| | This fork (before) | Al-Toto branch |
|---|---|---|
| Standalone node | `ShaderNodeBsdfDiffraction`, Fast (smooth scalar, reflection + transmission) and Realistic (RCWA cache) | `ShaderNodeBsdfDiffraction`, scalar thin-phase grating, reflection only |
| Grating | 1D lamellar only, grooves across the tangent | 1D or 2D (oblique) lattice, rotation per axis |
| Profile | rectangular | rectangular or sinusoidal (Bessel, Miller recurrence) |
| Roughness | none on the standalone node (delta orders) | GGX lobe per order |
| Parameters | node properties in nm (not texturable) | sockets (texturable), pitch in um, depth in nm |
| Default tangent | generated-space circular tangent | UV map tangent |
| Diffraction on Glossy/Glass/Principled/Metallic/Refraction | yes (facet gratings, multiscatter tables) | no |
| Physical options | substrate/ridge IOR, n,k tables, EM solve | colour tint only |
| BDPT / photons / VCM / guiding / Metal | yes | path tracing only (upstream Cycles) |

Notes on the branch's closure:

1. Its pdf normalization skips an order whose half vector is invalid for the evaluated direction,
   unlike its order pick. Checked: for two upper hemisphere directions that case cannot occur, so
   pdf and sampling agree (unit test `SamplingMatchesBranch`). The port normalizes by all
   propagating orders, which is the same value without relying on that.
2. The order pick keeps a `float[17 * 17]` array on the stack (1.1 KB per thread, bad on GPUs),
   and the closure needs a second closure slot for the Y axis parameters.
3. Its efficiencies depend on the incidence angle only, so the BSDF is not reciprocal. The fork's
   BDPT evaluates light-path vertices with the reciprocal shader (adjoint), so this is kept as is
   to render exactly like the branch.

## Design of the hybrid

* One node, one more model. `NodeShaderDiffraction::quality` gains `2 = GRATING`, the default for
  new nodes. Fast (1) and Realistic (0) are unchanged, so existing files render as before.
  The Grating model uses the branch's sockets (`Roughness`, `Pitch`, `Groove Depth`,
  `Duty Cycle`, `Rotation`, plus `Color`, `Normal`, `Tangent`) and its enums in `custom1`
  (profile) and `custom2` (grating type). Physical properties are shown only for Fast/Realistic,
  grating sockets only for Grating.
* Files written by the branch (node without storage) load as Grating: storage is created when the
  file is read (`node_blend_read_data_storage`), otherwise Blender would turn the node into an
  undefined node.
* Kernel: closure `CLOSURE_BSDF_DIFFRACTION_GRATING_ID` takes the enum slot of the unused
  `CLOSURE_BSDF_DIFFRACTION_ID` (its setup was never called), so no closure is renumbered.
  `kernel/closure/bsdf_diffraction_grating.h`: the branch's math (efficiencies, reciprocal
  lattice, virtual incident direction, GGX lobes), lattice resolved at setup into world space
  momenta so the closure fits one slot, no weight array, Miller recurrence that rescales.
* SVM node data `SVMNodeDiffractionGratingBsdfData`, OSL closure `diffraction_grating` and shader
  `node_diffraction_grating_bsdf.osl`, Cycles node `DiffractionGratingBsdfNode`, socket flag
  `LINK_TANGENT_UV` for the UV-map default tangent.
* Integrators: generic glossy closure with hero wavelength (`SR_BSDF_HAS_DISPERSION`), so path
  tracing, BDPT, photons, VCM, guiding (CPU and Metal), MNEE and shadow catchers need no special
  code. Scene setting "Material Diffraction" off replaces it by a Glossy GGX with the same
  roughness and colour.
* EEVEE previews the Grating model as glossy with its roughness.

## Validation plan

* Unit tests (gtest, `kernel_diffraction_grating_test.cpp`): eval matches the branch's eval;
  pdf integrates to the propagating fraction; sampled `eval/pdf` matches eval at the sample;
  sampling histogram matches pdf; passivity (albedo <= 1); 1D equals 2D with zero Y depth.
* Reference build of the branch (`build/altoto_ref`, the 3 commits on upstream `205bd3d2d29`).
  Same scenes (`tests/python/cycles_diffraction_grating_scenes.py`) rendered by both builds and
  compared numerically and visually.
* Matrix: CPU, Metal x path tracing, BDPT, photon mapping, VCM x guiding off/on, plus OSL (CPU).
* Regression: fork suites (`cycles_fork_node_scenes.py`, `cycles_metal_feature_scenes.py`,
  diffraction delivery checks for Fast/Realistic and the BSDF panels) before/after.
* Benchmarks before/after: non-diffraction scenes (no slowdown allowed), Fast model scenes, and
  the Grating model against the branch.

## Tools

* `tests/python/cycles_diffraction_grating_scenes.py`: demo scenes (`cd_rectangular`,
  `cd_sinusoidal`, `grid_2d`, `cd_hero`, `cloth`, `sign`, `sphere_uv`) and validation scenes
  (`furnace_1d`, `furnace_2d`, `consistency`), any device / integrator / guiding / OSL. Runs in
  the reference build too.
* `tests/python/cycles_diffraction_grating_compare.py`: noise-tolerant comparison of two render
  directories (channel mean ratio and block-averaged error).
* `tests/python/cycles_diffraction_grating_open_files.py`: opens files of another build, checks
  that every Diffraction BSDF loads as the Grating model with the same settings, renders them.
* `tools/diffraction_grating/run_matrix.sh <app> <out> [scenes] [res] [spp]`: all configurations
  and comparisons against CPU path tracing.
* `intern/cycles/test/kernel_diffraction_grating_test.cpp`: unit tests (needs `WITH_GTESTS`).
* Builds: `build/diffraction_hybrid/Blender_before.app` (fork before the port),
  `build/altoto_ref/install/Blender.app` (the branch's 3 commits on upstream `205bd3d2d29`,
  worktree `build/altoto_ref/src`).

## Results (2026-10-25, Apple M5, macOS 27)

Correctness

* Unit tests: 7/7 grating tests pass (values and pdf equal the branch's closure for 4800 random
  direction pairs over 12 configurations, sampling histogram matches the pdf, sampled and
  integrated albedo agree, albedo <= 1, efficiencies equal J_m(A)^2 and sum to one, 1D == flat 2D,
  bad inputs stay finite). Whole `cycles_test`: 410/410 pass.
* Same .blend files rendered by the branch and by this fork (CPU, same seed): channel means within
  0.1-0.4%, block error 0.0001-0.005. Files saved by the branch load as the Grating model with all
  settings (`cycles_diffraction_grating_open_files.py`).
* Matrix (`build/diffraction_hybrid/matrix_all`): CPU and Metal x path tracing, BDPT, photon
  mapping, VCM x guiding off/on, plus OSL, all 10 scenes, compared with CPU path tracing. All
  pass except guiding on the two CD scenes, which is noise: those scenes differ by up to 7%
  between two seeds without guiding (a 3 mm light seen through sharp orders), while guiding on/off
  with the same seed differ by <= 2%. White furnace: the grating sphere is 0.798 (1D) / 0.788 (2D)
  in every configuration, nothing above the white world.
* Regression (`suites_before` / `suites_after`, feature, CPU parity, node scenes on CPU and Metal,
  EEVEE nodes): bit-identical on CPU except the known non-deterministic `hair` and photon/BDPT
  float order noise (relRMSE <= 0.001); Metal differences equal the run-to-run differences of the
  old build itself (checked for bdpt_volume, displacement_subsurface, guiding). The Fast model
  renders bit-identically; only the default color of new Diffraction nodes changed (0.9).
* Scene setting Material Diffraction off turns the Grating model into a Glossy BSDF.

Speed (second render of a process, median of 3 rounds)

* Scenes without a grating, Metal, 512 px 256 spp, 8 interleaved rounds: 1.003 +- 0.009 of the
  build before the port. Keeping the order arrays out of the BSDF dispatch frame on Metal was
  needed: inlined, the same test showed about +4%. Fork suites (64 scenes, CPU and Metal):
  -0.1% to +0.9%, noise.
* Grating model against the branch's build, 512 px 256 spp: CPU 0.95x geometric mean (grid_2d
  0.82x, CD 0.96-0.99x). Metal 1.27x, but not because of the grating: with the grating replaced
  by a Glossy BSDF the fork's Metal kernels are already 2.5x (CD) and 1.5x (grid) slower than
  upstream in these scenes, before and after the port. The grating itself costs +8% (CD) and
  +21% (grid) over Glossy in the fork, against +0% and +50% in the branch.
* Memory: one ShaderClosure per grating (the branch: two), no extra kernel data.

Known issues found on the way, not caused by the port

* BDPT lost up to 20% of the energy at small camera clip starts, and ignored the spread of area
  lights (up to 27x too bright). Both fixed on 2026-10-25, see `doc/cycles_bdpt_light_fixes.md`.
* The fork's Metal kernels are slower than upstream in simple scenes (see above). Not fixed.

## Grating is the default model, also in old files (2026-10-25)

New nodes use the Grating model. Nodes of files saved before (Fast or Realistic) are converted to
it when the file is read: `version_diffraction_bsdf_grating_default()` in
`blenloader/intern/versioning_503.cc`, unconditional at the end of `blo_do_versions_503()`.

* Carried over: 1D rectangular grating, pitch (nm to um), depth, duty cycle, color, links (the
  Tangent link keeps its meaning: grooves across it at zero rotation), roughness 0 (the physical
  models were smooth). The color is scaled by the normal-incidence reflectance of the substrate
  (about 0.04 for IOR 1.5, about 0.9 for the metal of the tests) so the brightness stays close.
  Lost: transmission, IORs, n,k tables (the properties stay in the storage, unused).
* Marker: `NodeShaderDiffraction::flag` bit `SHD_DIFFRACTION_FLAG_GRATING_DEFAULT`, set on new
  nodes, on converted nodes and on nodes of the reference branch. A node set to Fast or Realistic
  in this build keeps that model when saved and opened again. The file version is not bumped, so
  the fork keeps upstream's file version numbers.
* Test: `tests/python/cycles_diffraction_quality_test.py --legacy-scene <file>` (new nodes
  Grating, explicit models survive a round trip, legacy nodes converted with their values); a
  legacy fixture made with the build before the port is
  `build/diffraction_hybrid/legacy/legacy_physical.blend`.
