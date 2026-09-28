# Analytic sphere: completed two-transmission paths

This is a numerically tested module with bounded Metal renderer acceptance. The
module is `kernel/light/coherent_sphere_transmit_geometry.h`. It describes only
outside-source → entry T → exit T → outside-receiver paths through one homogeneous,
lossless sphere. Internal reflection paths belong to a separate inventory.

## Renderer acceptance status — 2026-09-28

The v38b Metal renderer passes all three corrected 64×64, 128 fixed-sample
BDPT field checks against the unchanged independent reference. Adaptive sampling
and denoising are disabled. Phase-zero RMSE is 0.000272314; phase-pi RMSE
0.000259748; distinct-groups RMSE 0.0000398234. Results and actual EXRs:
`build/tests/python/coherent_sphere_transmission_render_v38e/`.

The earlier v38d black images were a **fixture camera clipping error**, not
proof of a transport failure: a newly created camera inherited a 0.1 m near
clip while the detector was only 0.1 m away. Float rounding clipped it away.
The corrected fixtures set near clip to 0.001 m and retain byte-identical
reference arrays and optical setup. Both failures and corrections are preserved.
Camera construction scripts now set explicit clipping distances.

The temporary host rejection was removed after all four transport regressions
passed: sphere transmission PT and PT/guiding, exterior mirror sphere BDPT, and
joined planar slab BDPT. Evidence: `build/tests/python/coherent_sphere_transmission_transport_v38e/results.json`.
The labels sphere_pt and slab_joined_pt in that report identify reused cases;
their recorded transport is BDPT. The established material-suite delivery remains
v37; v38b is the curved-transmission acceptance candidate. Numerical validation
includes 374 strict and 374 optimized checks, 139 point-policy checks, and 36
history checks. This supports one ideal sphere's isolated entry/exit branches,
not arbitrary curved geometry, internal reflections, or caustic singularities.

`coherent_sphere_transmit_inventory` returns up to three paths ordered by signed
angular momentum, two solved local interface frames per path, the Hessian Morse
index, and the relative Maslov phase in cycles. Arguments use one consistent
length unit. The split optical lengths must be converted to physical metres with
compensated arithmetic before optical phase evaluation. The supplied receiver
normal defines area orientation and is normalized internally; material sidedness
and visibility remain the caller's responsibility.

## Complete isolated root inventory

Let `a,b>R` be endpoint distances from the centre, `theta∈[0,pi]` their angle,
`A=R/a`, `B=R/b`, and `t=h/R∈(-1,1)` the signed impact parameter. For `n>1`:

```
F(t) = pi + 2 asin(t) - asin(A t) - asin(B t) - 2 asin(t/n) - theta.
```

Multiplying its derivative by `sqrt(1-t²)` gives

```
D(t) = 2 - A sqrt((1-t²)/(1-A²t²))
         - B sqrt((1-t²)/(1-B²t²))
         - (2/n) sqrt((1-t²)/(1-t²/n²)).
```

Each square-root ratio decreases strictly with positive `t`, so `D` increases
strictly there. It is even. Thus `F` has at most two derivative zeros and at most
three isolated roots. Bisection on the resulting monotone intervals enumerates
all roots. The `n=1` identity is handled separately as one decreasing interval.
The reference uses the same physical conservation law in independent double
arithmetic, not a call to production geometry.

## Curvature and phase

The four-dimensional constrained Hessian contains free-segment transverse
projectors and both spherical curvature terms. The module solves its full
receiver differential to obtain `dΩ_source/dA_receiver`. It does not replace the
sphere with a tangent plane. The Hessian can be indefinite. A Jacobi inertia
calculation counts negative eigenvalues; with the declared `exp(+ik OPL)`
convention the relative phase is `-MorseIndex/4` cycles, anchored to the positive
Hessian/free-space branch.

World points are reconstructed with compensated normalization and affine
arithmetic. Segment lengths and index-weighted sums retain their low parts.
Rounded hit coordinates are for BVH and field frames; optical phase uses the
compensated physical segments.

## Explicit unresolved inventories

The return status distinguishes empty inventory, invalid input, grazing endpoint,
caustic/ill-conditioned Hessian, and nonisolated axial rings. Numerical fold
proximity is conservatively classified using an angular equation tolerance
`8e-7`; eigenvalues within `1e-6` of the largest magnitude are unresolved.
Near-axis multiple-root inventories use a `1e-6` angular guard. These guards can
exclude nearby finite contributions and therefore require a renderer-level
error/unsupported policy; they must never silently remove one branch and render
the rest. `count` remains zero unless the entire inventory succeeds. There is no
energy or positivity clamp. This geometrical-optics module supplies no uniform
caustic diffraction approximation.

## Measured focused checks

`python3 tests/performance/run_cycles_coherent_sphere_transmit_test.py` checks the
actual header in `-O2` strict and `-O2 -ffast-math` modes against independent double
roots, Snell residuals, Hessian finite differences, receiver-direction finite
differences, and pair optical-length differences. It covers identity IOR,
normal incidence, one/three roots, all Morse indices, rotated and translated
coordinates, scene scales `.01/1/100`, and explicit failure statuses.

The recorded run has 374 checks per mode, zero failures. Maximum spreading
relative error is `3.38e-5`; normalized split optical-length error is
`3.66e-12` per sphere radius. For the proposed metre-scale two-source scene,
pair OPD error is `5.00e-14 m`. Independent Hessian and spreading finite-difference
errors are `4.75e-6` and `9.41e-6` respectively. Source hashes, commands, and raw
outputs are in `build/tests/performance/coherent_sphere_tt/results.json`.
Metal fast-math syntax compiles; actual GPU numerical execution is still pending.

Production integration additionally needs the opt-in native point back-root and
self-intersection policy, repeated entry/exit membership on the same primitive,
three branch records per source/candidate, Jones transport with branch phase,
and an explicit unresolved-caustic policy. No universal curved-surface or
Fabry–Perot completeness is claimed here.

## Mixed separate planar mirrors and one sphere block

`kernel/light/coherent_unfold_geometry.h` now embeds either one exterior sphere
reflection or one contiguous sphere entry/exit in up to four interface events.
Planar ideal reflections may precede and follow the sphere on separate objects.
A planar Glass reflection in air also follows this geometric construction; its
Jones coefficient remains the Glass material coefficient. Prefix/source images
and suffix/receiver images unfold independently, leaving the physical sphere
unchanged. The receiver normal is transformed by the suffix isometry. The sphere
inventory retains its one reflection root or up to three transmission roots;
spreading is unchanged by the orthogonal endpoint maps, and TT Morse phase is
inherited. Physical planar points are reconstructed in reverse image order and
checked against finite patch bounds, side conditions, and native BVH membership.
The field evaluator transports Jones frames in physical event order.

Virtual endpoints are evaluated as compensated float pairs. The sphere solve
uses their high components and its compensated optical length receives the
endpoint first variation `-launch_direction dot delta_source + arrival_direction
dot delta_receiver`. This removes first-order image rounding from phase, leaving
second-order stationary error. Reflection divides by compensated normal squared,
so a float normal's imperfect unit length does not alter the plane isometry.
The physical segment lengths remain available for visibility and material checks.

This is an explicitly bounded coherence approximation, not an exhaustive curved
transport solver. Multiple separated sphere blocks and planar transmissions mixed with a sphere
are outside the coherent inventory.
Those histories retain their ordinary native transport writer and therefore do
not interfere with this finite coherent inventory. The opt-in UI and tooltip
state this limitation. Existing limits of 64 candidates and 256 searched states
still apply; no truncation or normalization to native path-tracing energy is used.

## Sphere internal-reflection orders

`coherent_sphere_internal_geometry.h` enumerates `T-R-T` and `T-R-R-T` for a
lossless sphere with exterior endpoints. With `p=m+1` interior chords, signed
angular momentum `t` obeys

```
Delta(t) = p*pi + 2*asin(t) - asin(A*t) - asin(B*t) - 2*p*asin(t/n)
Delta(t) = theta + 2*pi*w
```

For `0<theta<pi`, `A,B<1`, `n>1`, `0<=Delta<=2*p*pi`; therefore exactly the
possible windings `w=0..p-1` are searched. The derivative multiplied by
`sqrt(1-t*t)` is even and strictly increases with `abs(t)`. There are at most
two critical points, hence at most three isolated roots per winding: reserve
six branches for one internal reflection and nine for two. No root sampling,
seed selection or truncation supplies this inventory. Tangencies, folds and
axial rings fail the whole inventory explicitly. Signed roots are sorted.

Each successive sphere normal advances by `pi-2*asin(t/n)`. The full six/eight
coordinate constrained optical-length Hessian includes every sphere curvature
term and neighboring-segment coupling. Its inertia supplies the Morse phase;
its receiver derivative supplies spreading. Compensated normalized sphere
points supply OPL. Internal Fresnel reflections use incident glass IOR and
opposite air IOR; Jones transport follows the physical `T-R^m-T` order. The same
unfolding wrapper permits planar mirror events around the block if the four
event limit leaves room. Native membership records the repeated sphere primitive
and interior incident side; ordinary unsupported histories retain their writer.

Focused CPU strict and fast tests currently each pass 171 checks, including
physical Snell/reflection residuals, endpoint finite-difference Jacobians and an
independent double all-winding inventory. Maximum compensated OPL disagreement
is `7.2e-13 m`; maximum spreading relative disagreement is `1.3e-5`. Actual Metal
numerical and renderer ownership validation must also pass before acceptance.

Production branch candidates use selected-branch evaluation: scalar root isolation
still covers the complete inventory, but each candidate solves only its requested
Hessian, spreading and frame. Exhaustive mode remains the independent-test entry
point. Every reserved branch is evaluated by the renderer, so an unresolved
branch still triggers the inventory error instead of disappearing. This avoids
computing all six/nine expensive Hessians six/nine times at every detector point.
