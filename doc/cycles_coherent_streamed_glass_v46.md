# v46: CPU coherent transport, streamed closed-convex Glass, and regressions fixed

This increment continues [the handoff](cycles_diffraction_handoff.md). It adds CPU support for coherent specular connections, implements the pending "real closed-Glass refraction" dependency for the streamed facet mode, and fixes three pre-existing defects found while validating on CPU. All CPU and Metal acceptance sets pass (see "Evidence").

## What is new

**CPU backend.** Coherent specular connections (bounded and streamed modes) now run on CPU and on Metal; the host rejects other GPU backends (`scene.cpp: coherent_specular_device_supported`). CPU renders PT and CPU path guiding; BDPT and GPU guiding stay Metal features (the integrator falls back to PT on CPU, as before). CPU OSL shading also passes.

**Streamed closed-convex Glass** (`coherent_transport_mode = FACET_SINGLE_REFLECTION`, UI name now "Streamed Facets"). Objects declared *Ideal Glass* may be flat-shaded triangle meshes forming one closed, edge-manifold, outward-oriented, convex volume. With *Max Interface Events* 1 the inventory adds exterior Fresnel reflection on every Glass facet; with 2 it adds every ordered pair of reflections on mirrors and exterior Glass sides, and entry/exit refraction (TT) through every ordered pair of distinct facets of one Glass object. Sources and detector are in exterior air. The Glass polarizer (transmission analyzer) is honoured on TT.

* TT uses the existing Fermat/Newton planar solver (`coherent_geometry_connect`); for two planes the optical length is convex, so the stationary chord is unique. Flux-normalized Fresnel s/p Jones factors, implicit-differentiation spreading and compensated OPL are the same components as bounded planar Glass.
* Facet orientation follows native Cycles: outward = object-space winding, reversed under negative object scale (`SD_OBJECT_NEGATIVE_SCALE`), exactly like `triangle_normal()`.
* BDPT ownership: `coherent_history_stream_after_interface` owns mirror R, exterior Glass R and one complete entry/exit pair through one object. A prefix inside Glass is never owned; internal reflection, rough/unmarked events or exits through another object return the path to native transport.
* Host validation (`scene/coherent_convex_mesh.h`): at least four faces, every edge shared by exactly two oppositely oriented faces, one connected component, positive (outward) volume, convexity against the float-representation bound. Sources must be strictly outside every Glass hull. Glass bounds may not overlap any other declared Mirror/Glass/Detector object (no nesting). Vector polarization mode is required. Cost: O(F) for one event, O(F²) for two (TT pairs run a small Newton solve).

## Defects fixed

1. **CPU could not see exits from declared Glass point spheres.** Embree 4.4.1 `RTC_GEOMETRY_TYPE_SPHERE_POINT` never reports the exit root of a ray that starts inside the sphere (probe: inside-origin rays return no hit). The GPU intersector was already two-sided for declared coherent Glass points. Static coherent Glass point clouds are now Embree user geometry whose kernel callbacks use Cycles' own analytic test (front root, and the exit root only when coherent connections are enabled) and invoke the normal query filters (`kernel/device/cpu/bvh.h`, `bvh/embree.cpp`). The BVH is rebuilt when the declaration changes. Without this, CPU TT/TRT/TRRT sphere routes were silently missing (error = the omitted-internal-path control, 1.52e-3).
2. **v39 mixed mirror + sphere TT fixture had been broken since v40** (Metal too, confirmed with the immutable v40 and v45 packages): v40's reservation of three root branches per sphere TT raised the candidate count to 90 > 64. The host limit is now 256 candidates; the kernel still stores at most 64 *connected* routes per detector point and rejects the render with an explicit error instead of dropping routes if exceeded.
3. **CPU silently dropped coherent solver failures** (unresolved sphere caustic etc.). CPU now reports them as a render error through a per-thread flag, like Metal (`COHERENT_SPECULAR_ERROR_MESSAGE`).

## Evidence

All renders: 128 samples, seed 19, BOX filter, adaptive and denoising off, fixed declared 1 mW sources, independent references and gates declared before rendering. Result folders are under `build/tests/performance/{cpu,metal}_coherent_v46/`.

| Set | Reference / gates | CPU (PT + guiding) | Metal (PT, BDPT, guiding) |
| --- | --- | --- | --- |
| v46 closed slab (pending since v45) | radial oracle, 1e-5 / 2e-5 | 6/6, mean error 8.6e-10 | 9/9 |
| New streamed Glass: prism TT (phase 0/π, distinct, partial coherence, negative scale, linked instance), rotated cube + mirror (exterior R, mirror R, mirror→Glass RR) | new independent oracle, corrected references | 18/18 | 27/27 |
| v45 two reflections | unchanged v45 | 6/6, bit-identical to pre-Glass build | 9/9 |
| v44 one reflection | unchanged v44 | 18/18, bit-identical | 27/27 |
| v42 coherent polarizer | unchanged v42 | 32/32 | (v43 Metal evidence retained) |
| v39 R/TT + v40 TRT/TRRT spheres | v39 corrected-Jones / v40 | 24/24 | 36/36 (`metal_coherent_v46/bounded`) |
| Host guards (open, inward, concave, disconnected, source inside, overlap, scalar mode, smooth shading; valid variants render) | expected messages | 11/11 | n/a (host) |
| Standalone: convex validator + streamed history + budget split | — | 41/41 strict and fast-math | — |
| Existing standalone coherent unit tests | — | all pass | path-field Metal compile pass |
| CPU OSL spot checks (prism, cube, v45 corner) | same gates | 5/5 | — |

**Independent Glass oracle** (`tests/python/cycles_coherent_streamed_glass_reference.py`): 3D complex E-vectors, Householder mirrors, Born–Wolf Fresnel with flux normalization, SciPy Fermat minimization for TT, finite-difference spreading, brute-force double visibility. It agrees with the separately derived v46 slab radial oracle to 5e-11. Gates (`cycles_coherent_streamed_glass_plan.py`) come from a 128-spp Monte Carlo noise model (including the stochastic Gaussian-coherence variance for the partial-coherence variant) plus 4×4-vs-2×2 quadrature error; every omission control (drop TT, drop exterior Glass R, drop two-event reflections, IOR→1) exceeds the gates by 10–600×.

**Oracle correction (preserved).** The first declared references (`build/tests/performance/coherent_streamed_glass_v46`) contained oracle defects, noticed because the negative-scale reference differed from phase 0 although the geometry is identical: the spreading stencil dropped paths within ~1e-7 m of internal triangle edges (13 pixels in two variants), warm starts were shared across sources, and exact shared-edge hits were double counted. The corrections were verified only against geometry identities and the slab oracle, never against render values; gate formulas are unchanged. Originals, CPU results against them (18/18 pass) and `coherent_streamed_glass_v46_oracle_fix/reference_correction_provenance.json` are kept.

## v47 follow-up

* **Viewport fix.** Viewport renders failed with "…unsupported: adaptive_aux_buffer": internal bookkeeping passes (adaptive aux buffer, render time, guiding debug, denoising history, volume majorants) are now accepted. They carry no light decomposition.
* **Longer chains and internal reflections.** Streamed Facets accept Max Interface Events 1–4. Routes with 3–4 events are enumerated depth-first (`coherent_facet_stream_long_routes`): mirror and exterior Glass reflections in air, entry into a Glass volume, any number of internal reflections (Fresnel or total internal reflection, complex r), exit, and continuation in air. Leg half-space conditions prune whole subtrees. Cost O(F^k). Routes of ≤2 events use the unchanged v46 code, bit-identical.
* **Concave and non-nested overlapping-bounds Glass.** Convexity is no longer required: every leg is BVH-tested, so a chord that leaves the volume is blocked; exit and re-entry are separate events. Bounding-box rules were replaced by exact tests: no surface contact between declared objects, and no vertex of one inside another Glass volume (generalized winding number). Sources must be strictly outside (winding ≈ 0, not on the surface). Nested Glass remains rejected, because native Cycles has no nested-dielectric priority either.
* **Volumes.** A declared streamed Glass may contain a constant, non-emissive Absorption / Scatter / Principled Volume medium (Add or constant Mix). The coherent ballistic field is attenuated by exp(−σₜd/2) per RGB channel on interior legs. Medium-scattered light stays native and incoherent: BDPT already invalidates ownership at a medium collision.
* **Photon mapping.** Photons now carry the same coherent history as BDPT light prefixes, and a caustic photon that follows an owned history is not stored at a declared detector. With oversampling set to 1, the coherent photon-mapped render matches the reference exactly. This also exposed a **pre-existing photon-mapping bug**: with the default camera oversampling (2), final renders and viewport were 2× too bright, because the film was normalized by scheduled samples instead of accumulated samples. It is now fixed in `PathTrace::get_num_accumulated_samples`.
* **Motion blur.** Streamed facets use `triangle_world_space_vertices` at the sample time for moving objects (static objects keep the direct fetch). Rigid object motion is accepted in streamed mode; Glass validity, source containment and contact tests run at every motion step; deforming Glass is rejected.
* **Broadband coherence.** The shared per-sample Gaussian phase exp(iZL/Lc) is a sampled wavenumber offset Δk = Z/Lc. For non-dispersive media it is therefore exact Gaussian-spectrum broadband transport, validated by the partial-coherence fixture against Gauss–Hermite spectral integration. **Dispersion** (IOR varying with wavelength inside the coherent sum) is not implemented.
* **Not implemented:** diffraction at mesh edges (requires a UTD/physical-optics edge model), smooth-shaded or rough curved surfaces in the streamed model (analytic spheres remain in the bounded model), and dispersion.

### v47 evidence (all pass; gates declared before rendering, hashes in each plan folder)

| Set | CPU (PT + guiding) | Metal (PT, BDPT, guiding) |
| --- | --- | --- |
| 3–4 events: concave L (TT, exit→exterior R, T-R-R-T incl. corner reflections), slab T-R-T/T-R-R-T, prism total internal reflection, 3-mirror corner | 24/24 | 36/36 |
| Constant absorbing / scattering medium inside Glass (scattering: PT/guiding only, BDPT adds real scattered light) | 12/12 | 15/15 |
| Rigid motion blur (moving Glass prism, time-averaged reference; ignore-motion control detectable) | 4/4 | 6/6 |
| Photon mapping (PT): streamed Glass, v45 mirrors, v39/v40 spheres | — | 9/9, 3/3, 12/12 |
| Regressions: v46 streamed Glass, closed slab, v45, v44, bounded spheres | earlier runs; ≤2-event output bit-identical | 27/27, 9/9, 9/9, 27/27, 36/36 |

Bugs found and fixed during v47: 4-event chains with internal reflections near a corner failed the full-coordinate Newton solve (near-singular Hessian). Mixed reflection/transmission chains now unfold the reflections as isometries and solve Fermat only over the transmission planes (`coherent_geometry_connect_unfolded`). Separately, volume and rigid-motion scenes were rejected by stale host guards; both guards are fixed. Photon runs set photon bounces to the scene's max bounces. With the default of 8, photons also (correctly) add caustics of longer paths that the ≤2-event references exclude. Oracle bugs fixed before any comparison: TIR complex conjugation, internal-reflection opposite medium, and IOR-independent medium tracking.

## Remaining scope (v46 statement, superseded by the v47 list above)

Supported in v46: native polarizer; bounded planar/sphere histories; streamed mirror facets through two events; streamed closed convex Glass with exterior R, mixed RR and one entry/exit pair.

## Reproduction

```sh
./compile.sh -j4 --no-fetch-libraries
python3 tests/performance/run_cycles_coherent_streamed_glass_host_test.py
install/Blender.app/Contents/MacOS/Blender --background --factory-startup --python tests/python/cycles_coherent_streamed_glass_scene.py -- OUT
python3 tests/python/cycles_coherent_streamed_glass_plan.py OUT PLAN_DIR
python3 tests/python/cycles_coherent_cpu_plan.py [--metal] --plan PLAN_DIR/plan.json --output-dir RUN --out-plan RUN/plan.json
install/Blender.app/Contents/MacOS/Blender --background --factory-startup --python-exit-code 73 --python tests/python/cycles_coherent_mirror_batch_worker.py -- RUN/plan.json
python3 tests/python/cycles_coherent_polarizer_check.py --plan RUN/plan.json --report RUN/gates.json
```
