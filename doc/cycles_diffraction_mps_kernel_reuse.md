# MPS solver-object reuse candidate

The tested candidate Realistic cache backend reused MPS LU factorization and solve kernel
objects for repeated matrix dimensions within one serialized matrix engine.
Each dictionary is bounded to 64 shapes. Matrices, pivots, completion status
buffers and finite checks remain per operation; numerical algorithms, float
precision, refinement counts and readback validation are unchanged. This does
not reuse matrix factorizations between different input matrices.

The objective is to remove repeated CPU-side MPS object construction while
building a GPU response cache. It does not address the dominant modal solve or
adaptive refinement cost and has no measured end-to-end speedup yet. Fast
mode already skips this response cache and receives no claimed benefit.

Apple M5 GPU tests ran outside the sandbox:

- 29 captured complex-system refined-solve/product cases passed against their
  saved independent reference solutions. Maximum relative residual was
  8.38448056e-8; there were no intermediate payload downloads.
- Five singular/nonfinite rejection cases passed, with untouched output and
  zero invalid-result downloads.
- The full Blender build succeeded after the change. `git diff --check` passed.

Raw results, fixture/source/executable hashes and compile flags are preserved
in `build/tests/performance/mps_kernel_reuse_v1`. Build log:
`/tmp/cycles_mps_kernel_reuse_build.log`.

This source/build candidate has **not replaced** the selected frozen outline
package. A full Realistic-cache result and timing comparison is still required
before selecting it as a performance improvement. These primitive checks do
not establish arbitrary-profile convergence or complete rendering correctness.


## Paired cache comparison and decision

The selected outline package and frozen candidate package each rendered the
same Realistic lossless furnace fixture on Apple M5 outside the sandbox:
1024 fixed samples, seed 11, adaptive sampling and denoising OFF, sequential
processes with a 360-second limit each. Both passed the existing analytic gate;
maximum RGB-mean error was approximately 0.0004062885 against a 0.005 limit.

Logged cache construction was 134.113 seconds for the control and 133.537 for
the candidate, only 0.4295% less in one pair. Total process times were 202.596
and 180.909 seconds, but include unrelated shader preparation and rendering;
they must not be substituted for the cache metric. Raw-image RMSE is 5.6333e-8
and maximum absolute pixel difference 5.96046e-7, without normalization.

There is insufficient evidence of a useful end-to-end cache improvement.
**Keep the outline package selected.** The object-reuse change was removed
from active source, and the Blender rebuild after removal passed. The exact
candidate header is retained as
`build/tests/performance/mps_kernel_reuse_v1/matrix_candidate.h`; the frozen
candidate application remains at
`build/diffraction_delivery_20260927_mps_kernel_reuse/Blender.app`, executable
SHA-256 `a3000451c7979713d1332c305e97e0ba6432d706b94c0852931e70828b873f98`.
It is an experimental comparison artifact, not the selected delivery.

Commands, hashes, logs, real scenes/EXRs, analytic reports and the comparison
are in `tests/output/diffraction/mps_kernel_reuse_cache_pair_v1`.
No extra repetitions were run to search for a favorable timing result.
