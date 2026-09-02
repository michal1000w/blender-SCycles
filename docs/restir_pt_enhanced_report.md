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

- ReSTIR PT owns indirect paths and automatically layers fresh ReSTIR DI at every eligible primary
  surface, including very large heterogeneous light trees. The RIS target and selected NEE
  contribution include Cycles' ordinary light/BSDF balance heuristic, while forward BSDF light
  hits retain the complementary MIS weight. The explicit ReSTIR DI toggle remains authoritative.
  Directly visible emission/background keep exact Cycles film writes, and only indirect paths of
  length three or greater enter the path-tree reservoir.
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
- Temporal replay restores camera filter, lens, time, and post-primary dimensions. It consumes
  only history explicitly preserved across a compatible render reset, once at the start of the
  new render. Samples already accumulated in the same still image are not replayed and counted a
  second time. Spatial replay retains the target pixel's camera ray.
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
- Random surface reconnection accepts all finite diffuse/glossy reflection and transmission
  closures. It uses Cycles' black-box marginal directional PDFs and applies the paper's roughness
  threshold only at the previous vertex; the current vertex is controlled by the inverse-footprint
  term. Singular and transparent events, or invalid shifted PDFs, reject only that spatial shift.
- Spatial neighbor maps are deterministic involutions, so A selecting B implies B selects A.
  Spatial shifts are staged in the initial-reservoir allocation; after all rays drain, each pair's
  A-to-B and B-to-A evaluations are read together for balance-heuristic pairwise MIS.
- A temporal selection is not fed through a second spatial Jacobian. Age-zero canonical pixels
  can still pair spatially, while pixels that selected temporal history keep that estimate locally;
  this prevents compound inverse-density outliers without disabling reuse for the image.
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

Spatial reuse defaults to zero because the production city scene showed no time-to-quality gain;
users can opt into reciprocal pairs for mesh-light GI workloads. Two distinct
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
| Continuous moving 3-frame scene after same-frame replay fix | image and flicker ratios `1.000000x`; time ratio `1.415` with generic Metal; the offline animation driver did not preserve GPU history between frame sessions |
| Final 48-light moving 3-frame scene, 32 x 32, 4 spp | RMSE ratio `0.9208`; residual-flicker ratio `0.9521`; time ratio `1.1038` with generic Metal |
| Interactive rendered viewport, same-session scene update | Metal completed both 16-sample states; `10.24 s` initial and `10.00 s` motion-settle windows |

The very first Metal run after a kernel-layout change includes several minutes of Apple Metal
specialization. Reported render-time ratios use subsequent cached runs; specialization time is not
treated as per-render cost.

The animation result above is one `bpy.ops.render.render(animation=True)` call per baseline or
enhanced sequence. The earlier large apparent gain came from replaying samples already present in
the same progressive film; a production city scene proved that this increases correlation and can
produce severe outliers, so it was removed. The offline animation driver currently creates a new
GPU history lifetime per frame and therefore receives exact spatial/fresh behavior, not temporal
reuse. A separate GUI test switched a real Cycles viewport to rendered shading, changed the scene
without restarting Blender, and completed the second render; compatible viewport resets can retain
history.

## Untitled.blend production-scene regression

The 1.9 MB scene links a large city asset, uses four analytic point-light groups, and had both
ReSTIR toggles enabled with temporal confidence 20 and one spatial neighbor. Early revisions were
slower and biased dark, and correlated same-film temporal-to-spatial replay could produce severe
outliers. The apparent `275.10` firefly in the fresh PT pass was diagnosed later as a real highlight:
the 64-spp reference value at that pixel is `330.451`. The actual error was the baseline estimate
of `1.518`; the ReSTIR estimate was closer to the reference.

After separating DI/PT, rejecting aged temporal reservoirs from the next spatial domain, and
restricting temporal reuse to external render history, the saved configuration produced:

| Resolution / samples | Standard Cycles | Fixed ReSTIR PT | Result |
| --- | --- | --- | --- |
| 480 x 270 / 8 | `18.640 s`, RMSE `0.797195`, mean `0.292918`, max `45.0813` | `19.145 s`, RMSE `0.797292`, mean `0.292976`, max `45.0813` | `1.027x` time; matched quality and peak |
| 960 x 540 / 8 | `18.714 s`, mean `0.293977` | `19.893 s`, mean `0.294006` | `1.063x` time; mean differs by `0.0102%`, inter-image RMSE `0.00878` |
| 480 x 270 / 32 | `18.890 s`, RMSE `1.142139`, mean `0.299244`, max `421.0963` | `22.087 s`, RMSE `1.142147`, mean `0.299276`, max `421.0963` | `1.169x` time; statistically identical convergence and peak |

The source `.blend` was never saved or modified. Nine missing texture files and 411 missing linked
data-blocks were reported identically for baseline and enhanced runs, so they do not explain the
before/after estimator difference.

## Final time-to-noise optimization

The final optimization pass compared equal-sample, no-denoiser images against independent
high-sample references. It made three production changes:

1. When PT temporal/spatial reuse is inactive, initial reservoirs now stream the whole sample
   batch instead of forcing one-sample reset/finalize cycles.
2. Textured and procedural emitter candidates evaluate their actual endpoint emission in the DI
   target. Visibility remains deferred until after selection; a texture-cache miss falls back
   locally.
3. ReSTIR NEE now participates in Cycles' existing two-technique MIS instead of replacing it.
   This both preserves BSDF-hit illumination and damps low-light-PDF NEE events with the balance
   heuristic. ReSTIR remains active on Untitled's 88 device lights and 16,302 tree emitters.

Matched Metal results:

| Scene / configuration | Standard Cycles | Enhanced | Interpretation |
| --- | --- | --- | --- |
| 96 point lights, 128 x 128, 16 spp | RMSE `0.292216`, `0.5368 s` | RMSE `0.262877`, `0.6016 s` | `10.0%` lower RMSE; about `10%` faster to equal error under inverse-square-root convergence |
| Untitled, seed 17, 480 x 270, 8 spp | RMSE `1.104828`, MAE `0.124697`, max absolute error `328.933`, `18.4386 s` | RMSE `0.762650`, MAE `0.122636`, max absolute error `241.355`, `19.1285 s` | `31.0%` lower RMSE, `26.6%` lower worst error; about `2.02x` faster to equal RMSE |
| Untitled, seed 23, 8 spp | RMSE `1.109543`, max absolute error `330.401`, `18.0472 s` | RMSE `1.054713`, max absolute error `270.043`, `19.1267 s` | `4.9%` lower RMSE and `18.3%` lower worst error; time-to-RMSE approximately break-even |
| Untitled, seed 47, 8 spp | RMSE `4.425112`, max absolute error `1868.405`, `18.0051 s` | RMSE `1.325496`, max absolute error `274.416`, `19.2310 s` | `70.0%` lower RMSE and `85.3%` lower worst error |
| 48 point lights, Metal, 64 x 64, 16 spp | RMSE `0.116690`, `0.5229 s` | RMSE `0.098219`, `0.5517 s` | `15.8%` lower RMSE; about `1.34x` faster to equal error |
| 48 point lights, MetalRT, 32 x 32, 8 spp | RMSE `0.229556`, `0.5878 s` | RMSE `0.148613`, `0.6887 s` | `35.3%` lower RMSE; about `2.04x` faster to equal error |
| Moving 48-light scene, Metal, 3 frames, 32 x 32, 4 spp | RMSE `0.403822`, residual flicker `0.603219`, `1.5225 s` | RMSE `0.371824`, residual flicker `0.574344`, `1.6805 s` | `7.9%` lower RMSE and `4.8%` lower residual flicker for `10.4%` overhead |

The earlier “275 firefly” diagnosis was wrong: the comparison tool reported maximum pixel value,
not maximum error. That pixel's 64-spp reference is `330.451`; baseline produced only `1.518`, while
ReSTIR produced `275.107` and was substantially closer. The comparison tool now reports maximum
absolute error, its location, the rendered value, and the corresponding reference value. Across
three 8-spp seeds, average RMSE fell from about `2.213` to `1.048` (`52.7%`) for about `6%` cached
overhead. Same-film temporal reuse was separately tested and rejected: at 16 spp it raised RMSE
from `0.11266` to `0.1818` and
cost `2.24x`, confirming that correlated progressive samples must not be replayed as independent
film samples.

## Known limits and follow-up work

The complete Cycles integration is implemented and guarded by exact local fallback, but it does
not claim that every path class uses every reuse strategy from the research prototype. The
following extensions would broaden reuse coverage or reduce cost without being correctness gates:

- validate pairwise spatial variance on a larger production-scene corpus; the focused feature
  matrix is finite and exact fallback is local, but the measured mesh-GI gain is only about 1.1%;
- validate the black-box glossy/transmission reconnection PDFs over a larger material corpus; the
  feature is implemented and locally guarded, but the current focused test was quality-neutral and
  roughly doubled runtime when a spatial neighbor was forced;
- continue reservoir compression beyond the current 128-byte record. Three reservoir layers and
  two 48-byte surface layers total 480 bytes/pixel (about 3.71 GiB at 4K) before scratch film and
  ordinary Cycles state, so high-resolution memory use is the largest remaining engineering cost;
- replace post-motion fallback with the paper's explicit disocclusion motion construction; early
  camera-stage replay compaction is implemented, but a packed pair-index launch could remove the
  remaining dense initialization work;
- extend forced NEE reconnection from triangle emitters to explicit, mathematically correct
  mappings for analytic finite and distant lights;
- make vector-weighted combined and every specialized Cycles light pass agree exactly;
- persist GPU temporal reservoirs across the offline animation driver's per-frame PathTrace
  lifetime; same-frame reuse is intentionally disabled because it duplicates film samples;
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

After a kernel-layout change, the MetalRT-motion specialization took `484.50 s`; its actual
four-sample render is otherwise sub-second once cached. This supports
separating Apple pipeline specialization from steady-state render cost.

The final post-production-scene-fix regression completed all 17 cases in one run, including photon
energy, MetalRT motion, point/mesh/world emitters, glass, hair, BSSRDF, volume, spatial opt-in
finiteness, BDPT interoperability, and ReSTIR PT shadow-catcher fallback. Photon plus PT measured
`1.81295994` mean versus `1.81804305` for its control (ratio `0.9972`); BDPT and shadow-catcher
fallback images matched their controls. The same 17-case suite passed after the final MIS changes.
