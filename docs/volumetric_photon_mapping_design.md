# Volumetric Photon Mapping Design

## Scope

Extend the existing Metal progressive caustic photon mapper from surface receivers to participating
media without replacing Cycles' established volume integrator. The first implementation targets
the difficult `L S+ V E` transport class: light paths pass through one or more sufficiently sharp
surface reflection/transmission events and then scatter in a volume visible to the camera.

Ordinary path tracing remains responsible for direct volume lighting, diffuse-before-specular
transport, non-caustic multiple scattering, unsupported closures, and every non-Metal device. The
feature remains opt-in through the existing Photon Mapping switch.

## Reference implementation assessment

JustARenderer (JAR) implements volumetric photon mapping as follows:

1. Aim photon launches at a list of caustic target bounds.
2. Sample exponential free-flight distances in a single homogeneous medium (or a global box).
3. Store a point photon at every accepted scattering event.
4. Ray-march camera segments with a fixed step count and gather photons with a spherical cone
   kernel.

This is a useful prototype and demonstrates the required data flow. It is not directly portable to
Cycles because its medium coefficients, bounds, step limits, gain, and spectral response are
renderer-global rather than shader-derived. The fixed-step camera integration also duplicates and
would disagree with Cycles' null-scattering/ray-marching implementations.

## Research findings

- Traditional volumetric photon mapping uses point photons and a 3D density estimate. It is simple
  and robust but needs more photons than beam estimators.
- The beam radiance estimate removes repeated camera-ray point queries. Photon beams additionally
  reuse complete light-path segments and generally reduce variance substantially.
- Progressive photon beams give bounded-memory consistency by rebuilding independent maps while
  shrinking the beam radius with the dimensionality-specific schedule.
- Unified Points, Beams, and Paths combines point/beam merging with path sampling through MIS,
  improving robustness across transport classes at considerable implementation and state cost.
- Photon planes and higher-dimensional samples can reduce variance further, but require new
  intersection estimators and acceleration structures; they are not a small extension of the
  current spatial hash.
- Volume path guiding improves free-flight distance, phase direction, roulette, and splitting
  decisions from a learned adjoint estimate. Cycles already has volume scattering-probability and
  directional guiding. Photon mapping should complement these systems for specularly focused
  caustics rather than introduce a second general volume guide.

Primary sources:

- Jensen and Christensen, [*Efficient Simulation of Light Transport in Scenes with Participating
  Media using Photon Maps*](https://graphics.ucsd.edu/~henrik/papers/sig98.html) (SIGGRAPH
  1998).
- Jarosz et al., [*The Beam Radiance Estimate for Volumetric Photon
  Mapping*](https://cs.dartmouth.edu/~wjarosz/publications/jarosz08beam.html) (EGSR 2008).
- Jarosz et al., [*A Comprehensive Theory of Volumetric Radiance Estimation using Photon Points and
  Beams*](https://graphics.ucsd.edu/~henrik/papers/volumetric_radiance_using_photon_points_and_beams.pdf)
  (TOG 2011).
- Jarosz et al., [*Progressive Photon
  Beams*](https://cs.dartmouth.edu/~wjarosz/publications/jarosz11progressive.html) (TOG 2011).
- Krivanek et al., [*Unifying Points, Beams, and Paths in Volumetric Light Transport
  Simulation*](https://cgg.mff.cuni.cz/~jaroslav/papers/2014-upbp/) (TOG 2014).
- Bitterli and Jarosz, [*Beyond Points and Beams: Higher-Dimensional Photon Samples for Volumetric
  Light Transport*](https://benedikt-bitterli.me/photon-planes/photon-planes.pdf) (TOG 2017).
- Novak et al., [*Monte Carlo Methods for Volumetric Light Transport
  Simulation*](https://cs.dartmouth.edu/~wjarosz/publications/novak18monte.html) (CGF 2018).
- Herholz et al., *Volume Path Guiding Based on Zero-Variance Random Walk Theory* (TOG 2019).
- Kettunen et al., [*Lightweight Photon
  Mapping*](https://cgg.mff.cuni.cz/~jaroslav/papers/2018-lwpm/index.htm) (TOG 2018).
- Lin et al., [*Fast Volume Rendering with Spatiotemporal Reservoir
  Resampling*](https://research.nvidia.com/publication/2021-11_fast-volume-rendering-spatiotemporal-reservoir-resampling)
  (SIGGRAPH Asia 2021).
- Kutz et al., [*Spectral and Decomposition Tracking for Rendering Heterogeneous
  Volumes*](https://disneyanimation.com/publications/spectral-and-decomposition-tracking-for-rendering-heterogeneous-volumes/)
  (TOG 2017).

## Selected estimator

Use scattering-collision photons with a normalized 3D Epanechnikov kernel:

`K(x) = 15 / (8 pi r^3) * (1 - ||x||^2 / r^2)`, for `||x|| < r`.

Photon free-flight sampling must call the same homogeneous weighted sampling or heterogeneous null
tracking used by camera paths. This preserves shader coefficients, overlapping volume behavior,
NanoVDB octree majorants, spectral weights, motion time, and cache handling.

Scattering-collision density estimates the local volume source term. Camera paths already include
the receiver scattering coefficient in their sampled-throughput weight, so gathering divides the
collision density by the receiver's `sigma_s` before multiplying by that camera weight. This avoids
the extra scattering-coefficient factor present in a naive collision-point port.

The current map record stays 48 bytes. A volume receiver is tagged in the high bit of the signed
receiver object field; surface matching therefore rejects it automatically. Position, incident
direction, flux, emitter, time, and spectral metadata retain their existing representation.

## Transport partition and stability

- Store only the first volume collision after a supported sharp surface chain, then terminate that
  photon path. This matches the current surface map's first broad receiver rule and prevents one
  emitted path from consuming unbounded map capacity.
- Gather only tagged volume photons from the same receiver object and shutter-time bin.
- After a camera volume scattering event is represented by the map, mark it as a photon receiver.
  Existing closure filtering then removes subsequent sharp camera-path transport, keeping photon
  and ordinary estimators disjoint.
- Use two-pass neighbor selection: first count all valid photons, then evaluate a randomized
  systematic subset. This retains energy in dense buckets while bounding expensive phase
  evaluation.
- Reject non-finite powers, invalid links, zero scattering coefficients, and out-of-range records.
- Use a larger volume radius derived from the surface radius because a 3D estimator needs more
  neighbors; expose a conservative scale rather than an unphysical gain.
- Launch delta point-light photons with an unbiased mixture: 90% toward the aggregate sharp-caster
  bounds and 10% over the full sphere. The complete mixture PDF preserves energy and support while
  spending substantially more paths on useful caustic transport.
- Keep independent map refreshes so finite-map noise averages over camera samples. Radius shrinkage
  remains user-controlled and is clamped to the convergence-safe range already exposed by Cycles.

## Integration points

1. Generalize volume-stack initialization so photon launches can use glossy visibility and correctly
   detect background or enclosing volumes.
2. In `integrator_photon_emit`, limit each surface intersection to a volume segment, sample that
   segment with native volume tracking, and store/terminate at a qualified collision.
3. Pass through pure volume-boundary meshes with the same transparent boundary transition as
   camera paths, then update the photon volume stack. This is required for emitters outside the
   medium and does not consume photon transport bounce depth.
4. Update the photon volume stack after transmitted surface events.
5. Add a volume matching/gather routine beside the current surface gather.
6. Invoke volume gather at the selected indirect volume collision before phase continuation and
   write the contribution through the standard volume indirect/combined pass paths.
7. Mark the camera volume event as a photon-map receiver before its phase bounce.
8. Reuse the existing host-computed sharp-caster bounds to importance-sample point-light emission.

## Validation matrix

- Kernel/unit: packing/tagging, normalized kernel values, time/object filtering, dense-neighbor
  selection, and finite-value guards where host-testable.
- Compile: Metal kernel compilation plus normal C++ build; feature-off compilation must not add
  work to other backends.
- Render A/B: homogeneous fog behind glass, heterogeneous VDB behind glass, anisotropic phase,
  colored absorption, multiple overlapping volumes, world volume, motion blur, emitters both
  inside and outside volume bounds, and emissive mesh/point/spot/sun sources.
- Convergence: compare photon mapping off/on against a high-sample path-traced reference; track
  mean luminance, RMSE, and temporal/sample variance at several photon counts and radii.
- Regression: photon mapping off must be image-identical; surface photon caustics must remain
  unchanged; CPU and unsupported GPU devices must retain prior behavior.

## Low-sample convergence study

Profiling `prism-volume.blend` showed that low-sample error is dominated by camera-path volume
free-flight sampling, not by the number of gathered neighbors. The photon map is expensive enough
that rebuilding it more often can consume most of the render time without adding camera samples.
This suggested amortizing an independent map across a small number of complete camera paths.

The selected implementation adds **Camera Samples**, clamped to 1--4 and defaulting to 2. For fixed
sampling, Cycles traces that many complete camera paths per requested render sample and increases
the map reuse interval by the same factor. Consequently, Camera Samples 2 doubles independent
camera free-flight samples while keeping the approximate number of emitted photon paths constant.
Film normalization uses the actual per-pixel sample count, so this remains the original raw
estimator rather than a denoiser, clamp, partial-path split, or post-process. Camera Samples 1 uses
the legacy sample indices and map schedule.

Adaptive sampling convergence tests operate in the render scheduler's requested-sample domain.
Until that scheduler can account for internal photon-camera oversampling, adaptive renders safely
fall back to one camera path per requested sample. This also leaves CPU and other unsupported
devices unchanged. The UI describes the optimization as fixed-sampling only.

### Controlled Metal result

The benchmark used the supplied scene at 256 by 256 pixels, 32 requested samples, 4,194,304 photons
per map, gather limit 256, radius 0.04, volume-radius scale 2.26, no radius decay, no denoising, and
no adaptive sampling. Four seeds were compared as raw linear EXRs against a 512-sample photon-map
reference from the same scene and settings.

| Configuration | Camera paths | Photon maps | Mean RMSE | Current render time |
| --- | ---: | ---: | ---: | ---: |
| Legacy, Camera Samples 1 | 32 | 4 | 0.131890 | 5.56 s (representative rerun) |
| Camera Samples 2 | 64 | 2 | 0.093615 | 5.584 s average |

The new schedule reduced mean RMSE by **29.0%** at effectively equal measured time. Its four-seed
mean radiance was 0.132345, compared with 0.132282 for the 512-sample reference. This small 0.05%
difference is far below the seed-to-seed noise and does not indicate an energy shift. Metal photon
insertion is not bit-deterministic because concurrent records race for finite hash capacity;
repeat legacy renders differed by RMSE 0.00729, so compatibility is assessed by schedule, radiance,
and convergence statistics rather than bit equality.

### Rejected alternatives

- Branching only the volume free-flight estimator changed mean radiance by roughly 7--10% because
  its state/weight partition did not match the complete Cycles path estimator.
- A quasi-Monte Carlo photon-bounce sampler preserved the mean but did not improve RMSE and was
  slower in this scene.
- A stratified line-integral gather shifted energy by about 23%; it is not a valid point-estimator
  substitution without a proper beam formulation and transmittance measure.
- Smaller, more frequently refreshed maps made camera sampling cheaper, but produced a persistent
  photon-count-dependent low bias of about 3.4% even at high camera samples.
- Raising the gather cap from 256 to 1024 produced negligible improvement, confirming that camera
  free-flight noise was the useful target for the current backend.

These failures are why the production change reuses maps only across complete camera paths. More
aggressive reuse or splitting needs a derived multiple-proposal/MIS estimator, not an empirical
gain.

## Deferred SOTA upgrade

The next backend should store photon path segments and implement point-beam gathering along the
camera segment, followed by progressive photon beams. It should be a separate estimator mode with a
segment BVH/grid, heterogeneous transmittance estimator, dimensionality-correct progressive radius,
and MIS against collision points/path tracing. Photon planes are promising after that foundation,
but are not justified before beam validation.

The collision-point implementation is deliberately the production-safe first stage, not a claim
that points are the lowest-variance volumetric estimator. Camera-side free-flight sampling remains
the dominant noise source around very sharp volume caustics at low samples; point-beam integration
is the researched next step for removing that variance rather than hiding it with a gain or clamp.
Reservoir reuse is also promising once candidates can be represented with cheap proposal weights
and an unbiased final volume evaluation. Lightweight Photon Mapping's emission guiding could reduce
wasted launches further, but should be trained from useful camera/light connections and retain a
support-preserving mixture, as the current target-bound point-light launcher does.
