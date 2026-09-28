# Partitioning deterministic coherent paths from ordinary PT and BDPT

The renderer evaluates candidate diagonal powers and same-group cross terms
deterministically at each marked detector hit. PT does not reliably sample
point-source paths through an ideal mirror, and the BDPT baseline is sparse;
both ordinary estimates of that path class must therefore be removed.

`kernel/light/coherent_history.h` defines a tested 8-byte optional sidecar for
one stored BDPT light vertex. Its two words encode up to four ordered patch IDs
(six bits each), four events and incident sides (two bits each), count, and
validity. Opposite-side paths through the same interface cannot share a
signature. The existing
vertex's `emitter_object` identifies the source. A separate device vector,
allocated only when the feature is enabled, can be indexed alongside physical
BDPT vertex slots; ordinary `KernelBDPTVertex` remains unchanged. The renderer
now records this sidecar during BDPT light-path generation.
`coherent_history_matches_source` also compares the vertex's emitter object
with the candidate light's object, so equal interface chains from unrelated
sources cannot be suppressed.

The light generator begins with a valid empty history for a participating
point source. After an ideal patch reflection/transmission sample,
append that patch ID, event and incident side. Any other scattering, medium event, unsupported
interface, or depth overflow invalidates the history for all later vertices.
The helper `coherent_history_after_scatter` enforces this classification from
the actual sampled closure label: only singular reflection on a marked mirror,
or singular reflection/transmission on a marked glass patch, can extend a
history. The caller must map the hit object to its patch ID and determine the
incident side from the oriented patch normal. This classification is tested
with ideal, rough, unmarked, and diffuse events.
At the first marked detector reached with an owned source/interface history,
light tracing terminates **before** reservoir storage, sensor splatting or
diffuse scattering. The camera estimator owns any subsequent suffix through
this detector, even if the camera-side connection vertex is unmarked. A light
path reaching an unmarked scatterer first has its history invalidated and
remains in ordinary BDPT.

The deterministic contribution owns every candidate sequence ending at a
marked detector, including the empty direct sequence. To prevent double
counting, suppress only matching ordinary strategies:

The older direct point-source group correction is disabled globally while
specular-connection mode is enabled. It is a separate signed estimator whose
path partition is not the marked-detector model; allowing it on a Glass
interface can feed an overlapping correction through the camera continuation.
Unmarked surfaces retain ordinary incoherent NEE. This choice is explicit:
only marked Lambertian detectors receive coherent fields in this mode.

The renderer keeps declared mirror and Glass patch closures ideal even when
Cycles' Filter Glossy control is nonzero. That control normally blurs a delta
closure after a low-probability camera bounce, including a diffuse detector
bounce. Such blurring would create ordinary broad-lobe source paths absent
from the deterministic ideal-interface candidate set. The override applies
only to participating coherent patch objects while this mode is enabled;
other materials retain their normal Filter Glossy behavior.

1. At detector NEE, skip selected coherent point lights for which the direct
   candidate is owned by the deterministic evaluator.
2. At every BDPT vertex connection, skip only a light-side detector prefix
   whose emitter and valid ordered history match a deterministic candidate.
   The camera-side vertex need not be marked. Preserve unrelated indirect
   light paths.
3. In BDPT sensor splats, skip the same owned light-side detector prefix.
   Terminating it during light tracing is the primary guard; the connection
   and splat guards also protect cached paths.

The camera path writes the deterministic complete intensity (all candidate
diagonals plus unordered pair terms) once at each detector hit, multiplied by
its ordinary camera throughput. Coherence groups govern cross terms; each
diagonal is included once regardless of group. Signed cross terms remain
unclamped. Candidate evaluation also checks the current camera-path bounce,
glossy and transmission counts together with the candidate's interface count
against the global, BDPT and light bounce budgets. This avoids adding a
source/interface path after the eye suffix has exhausted its continuation
budget. Opaque visibility, host candidate support, material model, film
passes and source/sensor coherence still need the renderer checks described in
the field foundation document.

The sidecar test runs with
`python3 tests/python/run_cycles_coherent_history_test.py`. It checks direct
history, maximum-length ordered reflection/transmission sequence, patch/event/side
mismatch, invalidation and overflow, direct-source selection, and light-side
ownership independent of camera-side detector status. Rendering against an
independent phase and incoherent reference is still required to validate the
complete PT/BDPT estimator partition.

## Glass slab regression, September 2026

The two-face air/glass/air fixture is a bounded test of the direct two-transmission
class at a marked white Lambertian detector. Its independent double-precision
reference solves Snell's radial equation and uses the globally projected
three-axis source ensemble. It excludes internal-reflection sequences and the
Fabry-Perot series; passing it does not establish arbitrary curved interfaces,
volumes, or all multipath sequences.

The first Metal slab render showed an exact half-image discontinuity. Both
transmission candidates solved geometrically in both halves, but BVH visibility
rejected the interface-to-interface segment where its ray began on one triangle
and immediately re-hit the adjacent triangle of the same planar face. The
connector now offsets only that outgoing visibility ray in the direction of its
next endpoint. Stationary points, optical length, and Jones transport retain
their unshifted values. The v18 diagnostic removed the seam: top/bottom ROI
means were 0.849883/0.846534, and its column-averaged fringe profile correlated
0.991640 with the analytic phase-zero profile. Its absolute mean still exceeded
the reference by 0.286977.

The saved fixture had Cycles Filter Glossy at its default 1.0. After the
detector's diffuse bounce, the normal filter broadened the marked roughness-zero
GGX Glass closures, admitting ordinary broad-lobe routes outside the declared
ideal class. The optional coherent mode now preserves ideal closures on its
validated marked patch objects while leaving unmarked materials' filter behavior
unchanged. At 512 square pixels and eight samples, both PT and BDPT v19
diagnostics then measured ROI mean 0.561202 versus analytic 0.561231, RMSE
0.014292, and top/bottom means 0.561175/0.561230. These are diagnostic checks;
the separate full-sample phase and incoherent controls determine acceptance.
At one fixed 0 EV sRGB display transform, PT, BDPT and reference show the same
unbroken fringes; the column-averaged v19 phase-zero profile correlates
0.999998 with the reference. The saved fixture also has a connector-on
incoherent control with source groups 1 and 2 and positive coherence lengths,
alongside its older connector-off, zero-coherence control. Both use the same
independent sum-of-diagonals reference.
