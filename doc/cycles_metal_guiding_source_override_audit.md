# Metal source-override validation audit

The strategy light-group diagnostic exposed a loader defect: `make_source()`
respected `CYCLES_KERNEL_PATH`, but `compile_and_load()` could load an installed
precompiled generic library instead. Source hashes and successful render completion
therefore did not establish that modified source ran.

The diagnostic routes ordinary sensor, manifold sensor, and vertex connections
to separate groups, with all emitters assigned a fourth group. Before the fix,
all first three groups were exactly zero, and the fourth contained all light.
The EXR and scene are preserved under
`build/metal-guiding-tests/strategy-pass-diagnostic/ignored-override-evidence/`.

The fix excludes precompiled libraries whenever the environment variable is set,
including an empty value, matching the source-path utility's detection. It also
adds a define to override source so its pipeline cache cannot reuse entries
populated by the old loader. Actual source remains part of the pipeline digest.
Normal installed builds still use their precompiled libraries.

Affected evidence: the spatial-moment candidate's six source-override renders
cannot establish its quality effect. Its previous performance-based rejection
is withdrawn; source remains reverted pending valid re-evaluation. The signed
volume source-override renderer regression likewise cannot validate execution of
that modified kernel. Standalone Metal tests compile explicit source and remain
valid. Normal installed 512/4096-SPP material benchmarks regenerated precompiled
libraries and are unaffected by this particular defect.

# Manifold BSDF consistency

Two new host regressions failed against the original helpers and pass after fixes:
Beckmann glass now shares the Beckmann refraction normal distribution and masking
function; GGX manifold transmission now applies the same energy compensation as
the ordinary camera BSDF. The shared functions moved into
`kernel/closure/bsdf_microfacet_manifold.h`, included by MNEE and transport tests.

The distribution test uses 10,000 stratified samples for each of four closure
variants. The contribution test compares the helper with the actual camera BSDF
converted into the sampled half-vector measure, for multiple incidence angles
and energy scales. All 43 guiding and 21 transport tests pass in
`/tmp/cycles-guiding-manifold-bsdf-after.log`; both added tests failed before the
fix in `/tmp/cycles-guiding-manifold-bsdf-before2.log`.

Full Blender build and installation succeeded, including regeneration of all
three precompiled Metal libraries. Direct Metal numerical checks of these new
helpers and renderer correctness/quality comparisons remain necessary. These
fixes do not establish correctness of manifold strategy MIS or complete BDPT
volume transport, and do not satisfy the overall CPU-guided quality requirement.

The full standalone suite subsequently passed on the actual Metal device in
`/tmp/cycles-guiding-manifold-loader-core.log` (43 host guiding, 21 host transport,
and all Metal checks). The two new BSDF numerical cases still run only on CPU;
compiling them into the renderer is not a GPU numerical comparison.

## Completed execution check

The corrected-loader diagnostic completed successfully (512 camera SPP,
2,097,152 auxiliary light paths, contiguous sample coverage). All four strategy
passes are nonzero and their sum matches Combined within the original floating
point tolerance. This proves that the requested routing source executed. The
frozen diagnostic source intentionally predates the new BSDF fixes.

`strategy-pass-diagnostic/fixed-regions-report.json` uses the exact original
benchmark object regions, unlike the initial render script's larger untrimmed
bounding rectangles (retained in `report.json`). The sphere mean is .25062309
versus CPU reference .25042105; the block is .14905476 versus .14785090. The
block's manifold sensor contribution is .00000268073, only .001813% of its
reference mean, whereas the combined offset is about .814%. This one sample
set cannot establish persistent bias, but its manifold contribution cannot
account for the observed offset. Ordinary VC/NEE strategy densities remain
under investigation; removing the manifold branch is not a justified fix.

## Direct Metal BSDF validation

The actual renderer Metal resource context now wraps the shared BSDF headers in
the standalone fixture. No BSDF implementation is substituted. It checks all
four GGX/Beckmann glass/refraction variants, two energy scales and three incident
angles, and 10,000 stratified normal samples per case. Resource buffer size is
queried from Metal's actual parameter type; dielectric Fresnel needs no lookup
resources. Maximum relative contribution error against the camera BSDF converted
to half-vector measure is 2.20537186e-6. All 43 host guiding, 21 host transport,
and Metal tests pass in `/tmp/cycles-guiding-manifold-bsdf-metal.log`.

## Failed generic pipeline wait

The installed caustic-control rerun waited indefinitely in `get_best_pipeline`
with all compiler workers idle (stack sample retained in
`/tmp/cycles-guiding-caustic-control-sample.txt`). The source search ignored
published failed generic entries, although publication only happens after
compilation completes. It now returns failure when the matching generic entry
failed and no usable loaded replacement exists. Final pipeline compiler errors
also reach the Cycles log. The original control was terminated after diagnosis,
with exit143; it was not counted as passed. Its underlying failure remains under
investigation via a fresh control render with Metal diagnostics.

Full build/install succeeded. `cycles_metal_pipeline_failure.py` creates an
isolated source copy with the generic guiding_begin_update entry renamed. The
actual Metal test passed: Blender exited nonzero with the expected missing-kernel
error, rather than hanging or silently using installed source. Evidence is in
`build/metal-guiding-tests/failed-generic-pipeline/report.json`.

Disk exhaustion during these tests was mitigated by deleting rebuildable
standalone test binaries/module caches and losslessly archiving inactive frozen
executables. Every compressed executable was SHA-256 verified before removing
its uncompressed copy. `frozen-executable-archives/manifest.json` and `restore.py`
preserve original paths and modes. Sources, scenes, images and logs were retained.
Identical vendored dependency files in frozen apps now share hardlinks; treat
those dependencies as immutable. The installed and build executables were not
archived. Free space recovered to about2GiB; continued monitoring is necessary.
