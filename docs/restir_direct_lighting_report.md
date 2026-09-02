# Metal Reservoir Direct Lighting: Design and Validation Report

## Outcome

This branch adds an opt-in reservoir direct-light estimator to Cycles on Metal. It is not a
replacement for every transport algorithm. It is the many-light layer in a hybrid renderer:

- fresh reservoir importance sampling (RIS) handles direct lighting on eligible surface vertices;
- regular Cycles path tracing and MIS handle sharp glossy paths, volumes, transparent paths,
  forward-only emitters, and every non-Metal device;
- the existing photon mapper remains available for difficult surface and volume caustics;
- the existing bidirectional path tracer (BDPT) retains its recursive, technique-aware MIS on
  supported vertices, while paths that locally leave its strategy set may use reservoir lighting;
- MNEE retains ownership of its manifold caustic connections.

Estimator choice is made per path vertex. A shadow catcher, unsupported closure, or sharp material
does not disable reservoir lighting elsewhere in the image.

The production defaults use eight fresh candidates with no temporal or spatial reuse. This was the
best tested quality/reliability tradeoff for offline rendering and animation: samples remain
independent and progressive scheduling remains efficient. Bounded screen-space reuse is available
as an explicit low-sample-count viewport option.

## Research decision

The implementation follows the conclusion common to the relevant primary literature: generalized
reservoir resampling is a framework, not one universally best path-space proposal.

- **ReSTIR DI** established spatiotemporal reservoir reuse for direct lighting with many dynamic
  lights.
- **Generalized RIS / ReSTIR PT** extended reservoir operations to general path spaces and made the
  target/proposal/Jacobian requirements explicit.
- **ReSTIR PT Enhanced** improves difficult indirect transport with richer path proposals and
  reconnection, but does not make a compact DI reservoir a universal replacement for BSDF, BDPT,
  MNEE, or photon techniques.
- **Compatibility-guided sampling** and **reservoir splatting** motivate conservative geometric
  compatibility tests and immutable source reservoirs.
- **ReSTIR BDPT** requires technique-aware extended-path-space weighting; it is not correct to mix a
  screen-space DI reservoir into BDPT's strategy set with ordinary balance weights.

Primary references:

- <https://research.nvidia.com/index.php/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct>
- <https://research.nvidia.com/labs/rtr/publication/lin2022generalized/>
- <https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/>
- <https://research.nvidia.com/labs/rtr/publication/junkins2026compatibility/>
- <https://research.nvidia.com/labs/rtr/publication/liu2025splatting/>
- <https://research.nvidia.com/labs/rtr/publication/hedstrom2025restir/>
- <https://research.nvidia.com/labs/rtr/publication/zeng2025restirpg/>
- <https://research.nvidia.com/labs/rtr/publication/sawhney2024decorrelating/>

## Estimator

At an eligible surface vertex, Cycles draws `M` ordinary light-tree or flat-distribution samples.
For candidate `i`, a positive scalar target `p_hat_i` is evaluated from the unoccluded BSDF,
constant light emission when available, and geometry term. Its reservoir weight is:

```text
w_i = p_hat_i / p_i
```

Streaming weighted reservoir sampling chooses one candidate. The final multiplier is:

```text
W = sum(w_i) / (M * p_hat_selected)
```

The selected light is then reconstructed and its full shader, vector contribution, shadow ray,
light linking, passes, and visibility are evaluated through the existing Cycles direct-light path.
Visibility is never reused. With fresh candidates this is an unbiased RIS estimate, including when
the selection target is only a positive proxy for a non-constant emission shader.

The selected sample is stored canonically as emitter ID plus two intra-emitter random dimensions.
This keeps a persistent reservoir at 48 bytes and allows the light's position and selection PDF to
be reconstructed at the current shading point and shutter time.

Direct emitter hits after a non-delta RIS vertex are omitted because the direct term is represented
by the pure light-sampling estimator and a usable output PDF for the resampled proposal is not
available to ordinary BSDF/light MIS. Forward-only emitters are detected and retained. Singular
paths retain their normal emission contribution.

## Local selection and feature behavior

| Scene/path feature | Behavior |
| --- | --- |
| Diffuse and rough surfaces | Fresh RIS at every eligible bounce |
| Sharp or moderately sharp glossy surface | Classic BSDF/light MIS below the configured roughness threshold |
| Mixed closures | Selected per vertex from Cycles' evaluated average closure roughness |
| Hair and BSSRDF | Evaluated when supported; otherwise existing closure/path fallback remains local |
| Transparent, holdout, ray portal | Existing Cycles path behavior |
| Volume scattering | Existing volume NEE; a volume event clears the preceding surface RIS marker |
| MNEE | Existing manifold estimator at the MNEE vertex |
| Photon mapping | Runs alongside RIS; combined mode uses one camera path per requested sample to preserve film normalization |
| BDPT | Recursive BDPT estimator on supported vertices; local BDPT fallback paths may use RIS |
| Shadow catcher | Only shadow-catcher paths bypass RIS; other paths in the same image remain eligible |
| Light and shadow linking | Existing sampling, receiver filtering, and shadow initialization paths |
| Mesh, point, spot, area, sun, world lights | Existing light-tree/distribution sampling and exact final shader evaluation |
| Motion blur and moving lights | Canonical samples reconstruct at current shutter time; production default carries no cross-frame state |
| Panorama, orthographic, border, adaptive sampling | Uses render-buffer indices and normal Cycles scheduling |
| CPU/CUDA/HIP/oneAPI/OptiX | Feature is ignored; renderer is bit-identical to its existing estimator |
| Metal and MetalRT | Shared shading estimator; only the intersection backend differs |

## Settings

The controls are under **Render Properties > Light Paths > Reservoir Direct Lighting**.

- **Light Candidates**: 8 by default; range 1-16. Eight is the tested offline balance.
- **History Length**: 0 by default. Positive values reuse the previous progressive sample at a
  compatible primary hit.
- **Spatial Neighbors**: 0 by default. Up to four compatible reservoirs can be selected from the
  immutable previous layer.
- **Spatial Radius**, **Normal Threshold**, **Position Threshold**: screen-space candidate search and
  geometric rejection controls for optional reuse.
- **Minimum Roughness**: 0.25 by default. Lower values expand RIS to glossier surfaces; the default
  preserves BSDF candidates where they are important.

History buffers are allocated only when history or neighbors are enabled. With production defaults,
the renderer retains normal multi-sample batching and pays no persistent reservoir-memory cost.

## Implementation map

- `intern/cycles/kernel/integrator/restir.h`: reservoir streaming, canonical reconstruction,
  targets, compatibility, temporal/spatial merge, and final weight.
- `intern/cycles/kernel/integrator/shade_surface.h`: per-vertex estimator selection and exact final
  direct-light evaluation.
- `intern/cycles/kernel/integrator/shade_light.h` and `shade_background.h`: forward-emission
  ownership for the pure RIS estimator and forward-only fallback.
- `intern/cycles/kernel/integrator/path_state.h`: volume-transition correctness.
- `intern/cycles/integrator/path_trace_work_gpu.*`: optional ping-pong allocation, reset, scheduling,
  cancellation safety, and viewport/offline shared state.
- `intern/cycles/scene/integrator.*`, `intern/cycles/blender/sync.cpp`, and the Cycles add-on UI:
  scene settings, device gating, synchronization, and controls.
- `tests/python/cycles_restir_metal_benchmark.py`: independent-reference quality, timing, and moving
  animation residual-flicker benchmark.
- `tests/python/cycles_restir_metal_regression.py`: isolated feature/fallback matrix on Metal and
  MetalRT.

## Validation

All compile-time Metal variants are built: software intersection, MetalRT, and MetalRT motion.
CPU compilation also includes the shared kernel headers to catch layout and conditional-compilation
errors.

A focused seven-suite CPU CTest run passed `cycles_light_cpu`, `cycles_shadow_catcher_cpu`,
`cycles_sss_cpu`, `cycles_volume_cpu`, and `cycles_updates_cpu`. The current `bidir` branch still
reports image-reference mismatches in four `cycles_integrator_cpu` images and one
`cycles_light_linking_cpu` image. ReSTIR is device-disabled on CPU, its shared arithmetic retains
the original operation order, and the dedicated CPU fallback pair is bit-identical with the option
off and on; the reference mismatches are therefore reported separately rather than counted as
ReSTIR feature regressions.

The deterministic 48-point-light benchmark uses an independent 256-spp reference, 16-spp baseline
and reservoir renders, no denoising, no adaptive sampling, and a box filter. On the tested Apple
Metal GPU at 256 x 256, the post-correctness-fix result was:

| Metric | Cycles baseline | Reservoir DI | Ratio |
| --- | ---: | ---: | ---: |
| RMSE to independent reference | 0.108549 | 0.090528 | 0.8340 |
| Median render time, 5 warm runs | 0.579 s | 0.627 s | 1.0831 |
| Mean image value | 0.347683 | 0.347509 | — |
| Independent reference mean | 0.348012 | 0.348012 | — |

This is a 16.6% RMSE reduction for 8.3% more median render time. The squared-error ratio is about 0.696,
equivalent to approximately 1.44 times as many baseline samples if variance scales ideally. Actual
end-to-end crossover depends on fixed scene/device overhead and must be measured on production
scenes rather than inferred solely from this small benchmark.

The corresponding MetalRT test at 128 x 128, 16 spp, and an independent 128-spp reference measured
RMSE 0.122787 versus 0.103748 (ratio 0.8449) and five-run median time 0.622 versus 0.628 seconds
(ratio 1.0094). This confirms that the shared shading estimator produces the same quality trend
with the hardware ray-tracing intersection backend.

The feature regression covers CPU fallback, light-tree and flat light distributions, point/mesh/
world emitters, sharp glass, hair, BSSRDF, volume transitions, light linking, panorama, cropped
adaptive rendering, photon coexistence, optional history/spatial reuse, MetalRT motion and depth of
field, BDPT coexistence, and shadow catchers. Each render is checked for finite non-empty output;
fallback pairs additionally compare image statistics.

The final isolated matrix completed 21 cases. Photon-only and photon-plus-reservoir runs in the same
one-camera-sample domain differed in mean by 0.35% (0.887723 versus 0.884624). BDPT with the ReSTIR
option enabled matched BDPT alone within the 1e-6 relative tolerance, demonstrating that local
estimator selection does not perturb BDPT-supported vertices.

A separate four-frame test rotated all 48 lights and a foreground object. At 128 x 128, 8 spp, and
an independent 128-spp reference per frame:

| Animation metric | Cycles baseline | Reservoir DI | Ratio |
| --- | ---: | ---: | ---: |
| Aggregate frame RMSE | 0.211581 | 0.185822 | 0.8783 |
| Reference-relative residual flicker | 0.277404 | 0.243503 | 0.8778 |
| Sum of per-frame 3-run median times | 2.145 s | 2.256 s | 1.0515 |

Thus the tested animation reduced both spatial error and frame-to-frame noise by about 12.2% for
5.1% additional median time, without carrying reservoirs between frames.

## Reliability boundaries

- Screen-space history and spatial reuse are optional, bounded, and not the offline default.
  Correlated reuse can improve very-low-spp interaction but may increase finite-sample error or
  ghosting in difficult motion/disocclusion cases. Render and frame resets clear both layers.
- No visibility value is cached. Every selected sample traces a current shadow ray, favoring
  reliability over the more aggressive visibility-reuse variants used by some real-time renderers.
- This is direct-light RIS, not a claim that a compact screen reservoir implements full ReSTIR PT
  Enhanced. Indirect illumination remains the responsibility of Cycles PT, BDPT, photon mapping,
  MNEE, and their existing feature fallbacks.
- The feature is opt-in and Metal-only. Unrelated render paths and other devices retain their old
  behavior.
