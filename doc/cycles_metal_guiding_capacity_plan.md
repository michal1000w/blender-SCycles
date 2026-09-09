# Mixture capacity experiment

Status: not adopted. Production retains the original 16-component implementation.
The capacity template and its dedicated test are archived locally under
`build/metal-guiding-tests/mixture-capacity-unadopted`. The coherent header-overlay
option remains available to the standalone test runner. Both 16- and 32-component
host suites passed; 32-component renderer and Metal validation were not completed.
The experiment is excluded from the deliverable, rather than presented as a feature.

Historical preparation notes follow.
The current spatial candidate improves several equal512SPP controls but still
trails CPU guiding in PT and point lighting. Its higher-SPP checks are running.
Capacity is a separate hypothesis, not an established explanation of that gap.

The smooth model is now `GuidingGaussianMixtureT<Components>`, with supported
capacities16/32 and the existing16-component alias retained. Components divide
contiguous Morton-order histogram leaves. Sixteen components use individual
depth-two nodes;32 use pairs of depth-three nodes. Both mass and observation
counts must use that mapping. Merely changing the old constant would incorrectly
read `tree[5+i]` for the upper components. The new sparse32-component host test
checks first/last-component mass and full-sphere PDF normalization.

Host and GPU allocations depend on this capacity. Never run a32-component kernel
source override against a16-component Blender host. The standalone runner now
accepts a header overlay consistently for both its host binaries and Metal
compilation. The isolated32 variant changes only its alias. A renderer experiment
requires a matching full Blender build and complete memory-accounting checks.

Initial32 host failures exposed fixture assumptions: all4096 observations were
assumed to occupy one component, and a prior normalization denominator embedded16.
The fixtures now check the total observation mass across components, dominant
component effective support, and the known quarter-weight publication retention;
the prior denominator derives from the actual component count. Tolerances are
unchanged. The first generalization omitted the quarter-weight retention, failed,
and was corrected explicitly. All failed logs are retained.

The default16 Metal build first exposed a fixture include-order regression from
formatting: resource-context headers must remain ordered. The order is restored
and protected with clang-format directives. Its subsequent run compiled and
passed the early sampling/field checks but failed the long sequential16-component
fit dispatch with Metal error `Impacting Interactivity
(0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)`. This is not a
passing run, and concurrency is not a proven cause. Repeat serially after the
4096 benchmark; if needed, split the test's iteration work into bounded dispatches
while preserving all iterations and numerical checks. Do not weaken acceptance.

Actual32-component Metal checks, full renderer32 build, and equal-SPP image/noise,
regional-energy and warmed performance comparisons remain required before any
capacity change can be adopted. None of this completes the outstanding full
BDPT medium traversal or broad feature-support requirements.
