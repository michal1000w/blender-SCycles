# Refraction integration audit after v45

The full coherent-transport goal remains incomplete. This audit identifies the next implementation dependency; it is not a delivered refraction feature or a replacement definition of success.

Current source evidence:

- `intern/cycles/scene/scene.cpp:339` rejects Glass in streamed facet mode. The streamed descriptors and direct-source inventory therefore do not currently enumerate refractive paths.
- `intern/cycles/kernel/light/coherent_history.h:95` implements mirror-only streamed ownership for one or two events. Removing the host guard alone would leave native refractive energy ownership inconsistent with the coherent replacement.
- The bounded enumerator in `intern/cycles/scene/scene.cpp:680` tracks medium transitions using scalar refractive indices. Equal indices do not establish connected-volume identity, especially for nested or overlapping separate objects.
- Existing planar Snell/Jones solvers are reusable numerical components. Another disconnected solver would not address those integration gaps.

The next defensible integrated step requires connected-volume identity, consistent oriented boundary validation, ordered entry/exit facet enumeration, per-interface medium transitions, and matching native typed transmission-history ownership. Closed convex Glass meshes with exterior endpoints provide an initial independently verifiable domain; that domain must not be described as unrestricted nested-volume or arbitrary-chain support. Tests must independently check Snell angles, the stationary-path Jacobian/spreading, optical phase, visibility, native double-counting controls, and PT/BDPT/guiding behavior before expanding topology and chain depth.

No renderer changes were made in this audit; the immutable v45 package remains the accepted delivery. Broader refraction work requires implementation and actual GPU validation, not merely removing guards or relaxing existing gates.

The preserved two-reflection fixture error also motivated a test-runner preflight. Future two-reflection plans require at least two effective glossy bounces, checked before device discovery or rendering. The archived one-bounce scene was verified to fail early, with no EXR produced; all nine regenerated future jobs carry the explicit requirement. Evidence: `build/tests/performance/coherent_bounce_preflight_v45/verification.json`. Historical scene, reference, acceptance and timing artifacts were not rewritten.
