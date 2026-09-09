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
cmake --build build/macos_arm64_Release --target blender -j 4
cmake --install build/macos_arm64_Release
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

The tested implementation does **not** establish the requested universal minimum
quality of CPU guiding. Equal-SPP regressions are reported below; improved BDPT
results do not excuse worse PT results. Broad production readiness and a speed gain
must not be inferred from successful compilation or these limited scenes.

## Results

Final installed-build checks and equal-SPP numerical results are recorded below.
The earlier research and experiment history is in `cycles_metal_path_guiding.md`
and the linked audit documents. Rejected and invalidated runs remain identified
there; they are not included as passing final evidence.

At **512 camera SPP**, three independent seeds, using the original two-seed,
8192-SPP CPU-guided references:

| Scene / integrator | MSE ratio to previous Metal guided | MSE ratio to CPU guided at 512 SPP |
| --- | ---: | ---: |
| Rough glass / PT | 0.9921 | 1.2672 |
| Transmission / PT | 0.9386 | 1.2603 |
| Rough glass / BDPT | 0.8911 | 0.8172 |
| Transmission / BDPT | 0.9439 | 0.9050 |
| Spherical point light / BDPT | 0.9083 | 1.1950 |

Lower is better. The material BDPT images improve on both controls, but ordinary
PT and point-light BDPT still miss the equal-SPP CPU quality target. Each BDPT
render also traced exactly 2,097,152 auxiliary light paths, checked against
contiguous camera-sample coverage. Training publications were checked separately.
These particular runs do not establish warmed runtime gains.

Local raw evidence: `build/metal-guiding-tests/spatial-moment-verified-loader`,
`spatial-moment-bdpt`, and `spatial-moment-point`, each with `comparison.json`,
image files, metadata and logs. The CPU references were not exposure-adjusted,
clamped or replaced to improve the ratios.

At **4096 camera SPP**, two independent seeds, with the same CPU-guided references:

| Scene / integrator | MSE ratio to previous Metal guided | MSE ratio to CPU guided at 4096 SPP |
| --- | ---: | ---: |
| rough glass / PT | 0.9044 | 1.2951 |
| rough glass / BDPT | 0.9307 | 0.9556 |
| transmission / PT | 0.9536 | 1.3758 |
| transmission / BDPT | 0.9340 | 1.0159 |

All eight GPU renders passed actual 4096-SPP and publication checks. Each BDPT
render traced exactly 16,777,216 auxiliary light paths with contiguous sample
coverage. Raw evidence is in `build/metal-guiding-tests/spatial-moment-4096`.
These measurements support adopting observed-spread refinement over the previous
Metal implementation. They do not meet the CPU-guided quality target in every case.

## Final verification record

- Release build and installation succeeded. Installed guiding-field, mixture and
  manifold BSDF headers were compared byte-for-byte with the delivery source.
- 45 host guiding tests and 21 host transport tests passed.
- The complete actual-Metal suite passed, including spatial root selection,
  normalization, fitting, history retention, conditional queries, and transport
  component checks. The 24-case manifold BSDF check has maximum relative error
  2.2054e-6 across 10,000 samples per case.
- Python syntax and staged whitespace checks passed.

The first incremental packaging attempt retained an older spatial header because
an archive restore preserved its timestamp. It is excluded as final-package
validation. The corrected build refreshed the inputs, regenerated the three Metal
libraries, and verified the installed header bytes. The earlier sequential-fit
Metal watchdog failure is retained in the capacity audit; the final full suite
passed all original iterations and tolerances without reducing work.

The verified installed build passed all **19 persistent-session renderer cases**
(`/tmp/cycles-guiding-final-regression-verified.log`). This includes restoring the
unguided image after disabling guiding, zero probability, all three public sampling
modes, memory resizing, unlimited training, lighting reset, and mixed CPU/Metal use.

The final installed BDPT volume smoke also passed: 32 actual camera SPP,
131,072 auxiliary light paths, contiguous sample coverage, publications at
1/2/4/8/16 camera samples, and finite nonempty output. Its saved scene is
`build/metal-guiding-tests/final-delivery/bdpt_volume.blend`. This checks guiding
within the existing volume transport; it does not remove the medium-traversal
limitation described above.
