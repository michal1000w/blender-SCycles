# Metal path guiding: build, use, and validation

This branch adds a Metal-native path-guiding backend for Cycles PT and the fork's
BDPT integrator. CPU OpenPGL remains available. The implementation includes bounded
GPU training histories, separate radiance/importance fields, spatial refinement,
position-conditioned directional mixtures, product sampling, and guiding-aware PDFs.
The field is published only after outstanding path and shadow work drains.

The renderer retains 16 mixture components. The unfinished 32-component experiment
is not included. Spatial refinement uses the observed positional spread with a
geometric fallback for insufficient support. This changes proposals, not image
contributions. The source-override loader and failed Metal pipeline handling are
also corrected, with executable regressions.

## Use

Build the existing configured Release tree and install its runtime resources:

```sh
cmake --build build/macos_arm 64_Release --target blender -j 4
cmake --install build/macos_arm 64_Release
```

Launch `install/Blender.app`. In Preferences, select Metal and your Apple GPU.
Select Cycles, GPU Compute, then Render Properties → Sampling → Path Guiding.
The panel exposes field memory, training-history memory, training camera samples,
and Product MIS / Product Resampling / Roughness Weighted sampling. The existing
bidirectional integrator selector enables BDPT. CPU training limits count field
updates; Metal training limits count camera samples. Equal numerical training
limits therefore do not imply equal training work.

A fresh checkout requires Blender's normal macOS build prerequisites and bundled
libraries, with both `WITH_CYCLES_DEVICE_METAL` and `WITH_CYCLES_PATH_GUIDING` enabled.
Build products, rendered images, and local benchmark caches are not source assets.
The Python scene factories reproduce the test scenes; `--save-scene` saves `.blend`
files alongside render outputs.

## Reproduce checks

From the repository root, on a machine with a working Metal device:

```sh
python3 tests/python/cycles_guiding_distribution.py --metal --transport
BLENDER_USER_RESOURCES=/tmp/cycles-guiding-user \
  install/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python-exit-code 1 --debug-cycles --log-level debug \
  --python tests/python/cycles_metal_guiding_regression.py -- \
  --output build/metal-guiding-tests/reproduce-regression
```

The standalone suite executes the shared sampling and fitting code on both the
host and actual Metal hardware. Missing hardware or a failed GPU command is a
failure, not a passing skip. The renderer regression exercises UI capabilities,
training limits, memory changes, persistent sessions, sampling modes, and CPU
coexistence.

Run equal-sample material comparisons, generating independent CPU-guided references:

```sh
python3 tests/python/cycles_metal_guiding_benchmark.py \
  --blender install/Blender.app/Contents/MacOS/Blender \
  --scene-script tests/python/cycles_metal_guiding_material_scene.py \
  --scenes rough_glass transmission --integrators pt bdpt \
  --samples 512 --seeds 101 211 307 --reference-samples 8192 \
  --reference-seeds 991 1993 --resolution 256 --memory-mb 256 \
  --output build/metal-guiding-tests/reproduce-materials
```

This runner compares CPU guided, Metal guided, and Metal unguided at equal camera
samples. It retains discarded warmups and records timing separately. It also
records BDPT auxiliary light work; equal camera SPP is not equal total rays.
Do not use the historical equal-time plots as final equal-sample evidence.

## Scope and known limitations

The user explicitly excluded the pre-existing BDPT transmission-block brightness
offset from path-guiding acceptance. It has not been hidden by normalizing images.
The fork's pre-existing incomplete BDPT medium traversal also remains: light paths
stop at the first volume collision. Component tests for signed volume event weights
do not establish a full multiple-scattering BDPT implementation.

The implementation does not establish universal CPU quality parity. The user
accepted the measured results on 2026-09-09; remaining differences are reported
below. Broad scene coverage and a universal speed gain are not inferred from
successful compilation or these limited fixtures.

## Accepted results — 2026-09-09

The user explicitly accepted the current best results as good enough and requested
completion. Four drained refinement rounds per publication are the final choice.
The remaining CPU error differences below are accepted limitations, not claims of
CPU parity. Further training-duration and RIS-prior experiments were prepared but
not run; they are not part of this implementation.

The refinement uses the existing memory budget more fully before training freezes.
It changes neither camera sample counts nor model-publication cadence. No image
normalization, denoising, clamping, or reference replacement was used to improve
the comparisons.

At **512 camera SPP**, on held-out seeds 401/503, against fresh CPU-guided controls:

| Scene | Metal PT / CPU MSE | Metal BDPT / CPU MSE |
| --- | ---: | ---: |
| Rough glass | 0.9815 | 0.7595 |
| Transmission | 1.0731 | 0.7994 |

At **4096 camera SPP**, seeds 101/211, against matching CPU-guided controls:

| Scene | Metal PT / CPU MSE | Metal BDPT / CPU MSE |
| --- | ---: | ---: |
| Rough glass | 1.1652 | 0.9466 |
| Transmission | 1.2193 | 1.0053 |

Lower is better. Relative to the previous committed one-round refinement, the
4096-SPP PT MSE fell 10.0% and 11.4%; BDPT MSE fell0.9% and1.0%. The point-light
BDPT test at 512 SPP over three seeds improved4.4% over that same Metal baseline,
while remaining 14.2% above CPU-guided MSE. These are measured mean errors over
specific fixtures, not universal quality guarantees.

The material reference is the unchanged mean of two independent 8192-SPP
CPU-guided renders. The point-light fixture uses its original 8192-SPP CPU-guided
reference. Each BDPT image traced2,097,152 auxiliary light paths at 512 SPP or
16,777,216 at 4096 SPP, verified against contiguous camera-sample coverage. Equal
camera SPP therefore does not imply equal total tracing work. These quality runs
do not establish a universal end-to-end speedup; cold pipeline compilation and
warmed rendering must be distinguished.

Machine-readable results are in `cycles_metal_guiding_results.json`. Local raw
images, commands, source provenance and work-count logs are preserved under:

- `build/metal-guiding-tests/refinement-four-heldout`
- `build/metal-guiding-tests/refinement-four-4096`
- `build/metal-guiding-tests/refinement-four-point`

## Verification

The Release build and installation succeeded. Packaged guiding-field, mixture and
manifold BSDF headers were compared byte-for-byte with the source. The shared
kernel code passed 45 host guiding tests,21 host transport tests, and the complete
actual-Metal suite. The final refinement change only adds ordered host dispatches;
the tested kernel code remains unchanged. The manifold BSDF check covers 24 cases
with 10,000 samples per case and maximum relative error2.2054e-6.

All final 512/4096 SPP benchmark renders passed actual sample and auxiliary-work
checks. The installed BDPT volume smoke passed 32 SPP with 131,072 auxiliary light
paths and the expected training publications. Its saved scene is
`build/metal-guiding-tests/final-delivery/bdpt_volume.blend`. This exercises guiding
within the fork's existing volume transport, not full light-side medium traversal.

The earlier archive-restore timestamp problem was corrected by rebuilding the
inputs, regenerating Metal libraries, and verifying installed header bytes. The
old sequential-fit watchdog failure remains in the audit; the final component
suite passed the original workload and tolerances. Rejected and invalidated
experiments are retained in the research notes, not counted as passing evidence.

The accepted four-round implementation passed all **19 final integration cases**,
including UI modes, memory changes, persistent resets and CPU coexistence. BDPT
cancellation and same-session restart passed with 0.0997-second cancel latency.
An optional shader compilation interrupted by application shutdown no longer reads
an unfinished error result, retries an archive, or reports a false compilation
failure. The rebuilt package passed the integration suite without that shutdown
error. Actual compile failures while the application is running retain their
error reporting and generic-pipeline failure handling.

Final logs: `/tmp/cycles-guiding-accepted-final-build2.log`,
`/tmp/cycles-guiding-accepted-final-install.log`,
`/tmp/cycles-guiding-accepted-final-regression.log`, and
`/tmp/cycles-guiding-accepted-four-cancel.log`.
