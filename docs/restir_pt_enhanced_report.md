# Cycles Metal ReSTIR PT Enhanced: Implementation and Validation Report

## Scope

This document tracks the opt-in ReSTIR PT Enhanced implementation for Cycles on Metal and
MetalRT. It is separate from the existing 48-byte direct-light RIS implementation documented in
`restir_direct_lighting_report.md`.

The implementation is local to this working tree. No remote repository, branch, or pull request
was modified.

## Algorithm sources

The design follows the 2026 ReSTIR PT Enhanced paper and supplemental document, with the public
RTXDI implementation used only as behavioral reference because its runtime shader sources carry a
proprietary NVIDIA license.

- <https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/>
- <https://github.com/NVIDIA-RTX/RTXDI>

Required components identified from the paper are:

1. a unified direct/global-illumination path reservoir;
2. complete primary-sample-space random replay and hybrid reconnection;
3. dual ray-footprint and previous-lobe roughness tests (`c = 0.02`, `alpha = 0.2` defaults);
4. pairwise domain-coverage MIS and reciprocal spatial neighbor pairing;
5. temporal confidence limiting and a 17 x 17 seed-duplication map;
6. vector-valued shading weights for color-noise reduction;
7. temporal reprojection, including a second disocclusion motion candidate;
8. forced NEE/light reconnection and Russian roulette only during initial sampling;
9. compact persistent reservoirs and replay compaction.

## Implemented architecture

- ReSTIR DI and ReSTIR PT are disjoint. Primary NEE uses the existing multi-candidate ReSTIR DI
  proposal, directly visible emission/background keep exact Cycles film writes, and only indirect
  paths of length three or greater enter the path-tree reservoir. This matches the paper's
  assumption that a separate DI method exists and prevents short paths without a reconnection
  vertex from accumulating invalid temporal confidence.
- Unsupported work falls back per path. Shadow-catcher paths and BDPT-supported strategies retain
  their existing estimators; their presence does not disable ReSTIR PT for another pixel.
- Sharp surfaces retain normal Cycles BSDF/light MIS while eligible rough surfaces can use the
  existing multi-candidate light RIS as the NEE proposal inside the unified path reservoir.
- Initial, previous, and current path reservoirs are separate. Reuse reads an immutable source and
  writes a cleared destination, preventing in-pass feedback.
- Replay uses the normal Cycles wavefront kernels and a scratch film buffer. Camera sample
  accounting and convergence buffers are not incremented by auxiliary replay paths.
- Replay pairs without a compatible source are rejected in camera initialization, before a ray is
  generated or enters the divergent wavefront queues. This keeps the dense camera-init launch but
  compacts all expensive downstream random-replay work.
- Background rays now use the same source-availability gate as primary surface rays. Indirect
  environment hits enter the path reservoir, while directly visible environment remains exact;
  this prevents an auxiliary camera miss from leaking an extra world sample into the main film.
- Temporal replay restores camera filter, lens, time, and post-primary dimensions. Spatial replay
  retains the target pixel's camera ray.
- Reservoirs carry a rolling object/primitive/shader path fingerprint. A replayed candidate is
  rejected locally if its path length, sampling technique, or topology differs.
- RGB vector numerators are accumulated independently from scalar reservoir selection.
- Hybrid replay stores the first rough/large-footprint reconnection vertex, replays the prefix,
  evaluates the target BSDF toward that vertex, applies a crossed-geometry PSS Jacobian, and sends
  the connection through Cycles' ordinary visibility/transparent-volume shadow path.
- If no earlier vertex qualifies, indirect NEE paths ending on triangle emitters cache their
  unweighted prefix/BSDF factor and area endpoint for forced light reconnection. The shifted
  target evaluates `triangle_light_pdf` and the light-tree selection PDF, then applies the exact
  source/target density ratio. Analytic delta and distant lights retain per-candidate exact
  fallback because they require a different discrete/directional mapping from the area-measure
  Jacobian.
- Random surface reconnection is restricted to diffuse reflection vertices, for which the omitted
  outgoing directional density ratio is exactly one. Glossy, transmission, singular, hair,
  subsurface, and volume paths remain fully rendered but reject an unsupported spatial shift
  locally; they never disable ReSTIR PT for the image.
- Spatial neighbor maps are deterministic involutions, so A selecting B implies B selects A.
  Spatial shifts are staged in the initial-reservoir allocation; after all rays drain, each pair's
  A-to-B and B-to-A evaluations are read together for balance-heuristic pairwise MIS.
- A 17 x 17 duplication pass compares selected random seeds and feeds the paper's nonlinear
  temporal confidence reduction (`lerp(20, 1, D^0.1)`). Temporal age is also treated as a
  correlation signal so an isolated firefly cannot retain maximum confidence indefinitely.
- Primary surfaces store previous/current motion vectors. Temporal reuse first tries pre-motion
  reprojection and tries the post-motion candidate if the first surface fails compatibility.
- Russian roulette remains active only during initial sampling. Replay returns continuation
  probability one, avoiding a second stochastic termination of an already selected source path.
- When photon mapping is enabled it owns indirect transport and ReSTIR PT does not stream the same
  GI contributions; eligible direct surfaces can still use ReSTIR DI.
- Software Metal, MetalRT, and MetalRT-motion libraries share the estimator and compile from the
  same kernel source.
- Reconnection capability is inferred from the cached positive densities, and the transient
  reciprocal-stage validity reuses `replay_accounted`. Removing the redundant aligned tail shrank
  each persistent Cycles reservoir from 144 to 128 bytes without changing rendered output.

## Safety invariant and current spatial status

Spatial random replay without reconnection/domain MIS was experimentally shown to be invalid: a
16-sample 16 x 16 test reached a maximum channel value of `1.392535e14`. Consequently, a spatial
candidate is currently admitted only if its reservoir contains a positive reconnection PDF and
Jacobian. Otherwise that pixel keeps its canonical fresh reservoir. This is a per-candidate
fallback, never a whole-image disable.

Hybrid reconnection and staged reciprocal pairwise MIS are now implemented. The guard remains as a
normal per-candidate domain check: only reservoirs with a valid stored reconnection PDF/Jacobian
can enter a spatial pass. Singular/transparent lobes, incompatible surfaces, non-reciprocal border
links, invalid PDFs/Jacobians, and occluded connections retain the canonical reservoir locally.

One reciprocal spatial neighbor is the opt-in feature's default; users can set it to zero for a
temporal-only mode or raise it for experimentation. Two distinct
energy defects were fixed: background replays bypassed the primary source gate, and queued
FINALIZE/DUPLICATION kernels could observe the next sample's rewritten phase/pointers. A completion
boundary now protects the shared integrator constant. A no-history copy/materialization pass is
bit-identical to spatial-off at 1, 2, and 16 samples. In an active triangle-emitter GI test, one
paired pass preserved mean energy within `0.002%` and improved RMSE by about `1.1%`; this is
promising but not yet broad enough to change the production default.

## Intermediate Metal measurements

All values below are deterministic diagnostic scenes without denoising. They are not final quality
claims.

| Configuration | Result |
| --- | --- |
| Fresh unified reservoir, 32 x 32, 16 spp | finite; mean `0.065079` vs reference `0.064771`; RMSE ratio `1.0078` |
| Temporal confidence 1, 16 x 16, 16 spp | finite; RMSE ratio `0.9957`; time ratio `1.469` |
| Unguarded temporal confidence 20 | rejected: persistent firefly, max `7.0618`, RMSE ratio `13.21` |
| One reciprocal spatial scheduling pass, confidence 1 | finite; RMSE ratio `1.0001`; time ratio `1.061` |
| Unguarded three-pass spatial reuse | rejected: maximum `1.392535e14` |
| DI/PT separated, temporal confidence 20, spatial 0 | mean `0.064985` vs reference `0.064916`; RMSE ratio `0.7979`; time ratio `1.696` |
| DI/PT separated, pre-pairwise spatial 3 | finite; RMSE ratio `0.8934`; time ratio `1.878` |
| Staged pairwise spatial before phase-lifetime fix | finite; max `0.2332`; RMSE ratio `0.7522`; mean `0.05833` vs reference `0.06492` (rejected) |
| Spatial copy-only after background/finalizer fixes | bit-identical to spatial-off; 16-spp mean `0.06498366`, max `0.28535196` |
| Exact triangle-NEE pair, mesh GI, 32 spp | mean `0.5431778` vs control `0.5431666`; RMSE `0.0171262` vs `0.0173115` (`0.9893x`) |
| Analytic point lights after endpoint restriction, 64 x 64 | finite; no Metal command-buffer fault; unsupported endpoint mapping falls back per candidate |
| 128-byte reservoir + early replay compaction | mesh-GI spatial image bit-identical to the 144-byte implementation |
| Continuous moving 3-frame scene, 8 spp, temporal 20 | RMSE ratio `0.4395`; residual-flicker ratio `0.4093`; max `0.4533` vs baseline `1.2197`; time ratio `1.637` |
| Interactive rendered viewport, same-session scene update | Metal completed both 16-sample states; `10.24 s` initial and `10.00 s` motion-settle windows |

The very first Metal run after a kernel-layout change includes several minutes of Apple Metal
specialization. Reported render-time ratios use subsequent cached runs; specialization time is not
treated as per-render cost.

The animation result above is one `bpy.ops.render.render(animation=True)` call per baseline or
enhanced sequence. Reservoir history therefore survives ordinary frame transitions in the same
render session. A separate GUI test switched a real Cycles viewport to rendered shading, waited
for convergence, changed the scene to animation frame two without restarting Blender, and reached
`Rendering Done` again. Blender's startup splash remained a macOS compositor overlay in captured
screenshots, but it did not interrupt the viewport render or its persistent session.

## Known limits and follow-up work

The complete Cycles integration is implemented and guarded by exact local fallback, but it does
not claim that every path class uses every reuse strategy from the research prototype. The
following extensions would broaden reuse coverage or reduce cost without being correctness gates:

- validate pairwise spatial variance on a larger production-scene corpus; the focused feature
  matrix is finite and exact fallback is local, but the measured mesh-GI gain is only about 1.1%;
- extend non-diffuse random reconnection with its exact shifted outgoing-density ratio; those
  surfaces currently use correct per-candidate fallback;
- continue reservoir compression beyond the current 128-byte record. Three reservoir layers and
  two 48-byte surface layers total 480 bytes/pixel (about 3.71 GiB at 4K) before scratch film and
  ordinary Cycles state, so high-resolution memory use is the largest remaining engineering cost;
- replace post-motion fallback with the paper's explicit disocclusion motion construction; early
  camera-stage replay compaction is implemented, but a packed pair-index launch could remove the
  remaining dense initialization work;
- extend forced NEE reconnection from triangle emitters to explicit, mathematically correct
  mappings for analytic finite and distant lights;
- make vector-weighted combined and every specialized Cycles light pass agree exactly;
- add a larger independent production-scene corpus and long animation sequences; the current
  deterministic matrix is a regression suite, not proof over every possible Blender file.

## Regression status

The focused suite passed point, mesh, world, sharp-glass fallback, hair, BSSRDF, volume, MetalRT
motion, spatial opt-in, BDPT interoperability, and shadow-catcher cases with finite output. CPU
fallback was bit-identical. A broad run also passed light linking, panorama, crop/adaptive, and
photon-only cases. A matched 16 x 16 photon control measured mean `1.81804`; photon plus ReSTIR PT
measured `1.83254` (ratio `1.0080`), passing the disjoint-estimator energy bound. Two earlier
attempts were aborted by macOS with `MTLCommandBufferErrorDomain ... Impacting Interactivity`
during pipeline activation; a later isolated run completed successfully.

After the final 128-byte ABI and triangle-PDF changes, the MetalRT-motion specialization took
`334.50 s`; its actual four-sample render is otherwise sub-second once cached. This supports
separating Apple pipeline specialization from steady-state render cost.

The final post-change focused regression completed all 17 cases in one run, including matched photon energy,
MetalRT motion, point/mesh/world emitters, glass, hair, BSSRDF, volume, spatial opt-in finiteness,
BDPT interoperability, and ReSTIR PT shadow-catcher fallback. The final photon ratio was
`1.81301191 / 1.81804305 = 0.99723`; BDPT and shadow-catcher fallback images matched their controls.
