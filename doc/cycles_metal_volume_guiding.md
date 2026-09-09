# Metal volumetric path guiding

Completed and verified under the user's requested scope. The remaining BDPT
energy discrepancy is explicitly deferred; the guiding work is finished.

## Delivered behavior

Metal PT and Metal BDPT camera paths use the GPU volume guiding field to sample
a mixture of the actual phase function and a learned radiance/phase proposal.
The physical phase evaluation and the matching proposal PDF determine path
throughput and MIS. Surface and volume guiding can be enabled independently in
Render Properties → Sampling → Path Guiding. Metal exposes training camera
samples, field/training memory, sampling mode, and separate guiding probabilities.

This change also reconstructs the medium stack for cached surface-to-camera
connections and preserves three-strategy emitter MIS after the first camera
volume event, including lamps, meshes, world light, and transparent continuation.
The preparatory connection/volume-write refactor was removed after the user
excluded further transport work. Its tested snapshots and investigation notes
remain in the local build artifacts; it is not part of the delivered source.

## Explicitly deferred transport limitation

The BDPT light walk stops at its first volume collision. Camera paths continue
through volumes, but complete bidirectional multi-volume traversal and every
surface/volume connection strategy are not implemented. Some longer volumetric
BDPT paths consequently lose energy. The user explicitly asked to leave that
issue for now. This release does **not** claim to fix it or to establish full
radiometric equivalence of volumetric BDPT. Ordinary Metal PT does not use that
BDPT fallback.

No reference exposure, pixel values, sample limits, or transport weights were
adjusted to disguise the remaining discrepancy. Full reference-error results,
including the backward-scattering regression, remain in the benchmark reports.

## Validation protocol

Contenders use 256 camera samples, 64×64 pixels, seeds 101, 211, and 307.
Each is compared directly with the same scene's reference: the mean of two
8,192-sample CPU OpenPGL-guided images, seeds 991 and 1993. Denoising, adaptive
termination, clamping, and time limits are disabled. BDPT additionally emits
1,048,576 light paths per 256-sample render, verified from actual batch logs.
Runtime is recorded, but comparisons and acceptance use equal camera samples.

Reports retain raw linear EXR/NumPy images, linear RGB MSE, log1p MSE, reference
uncertainty, settings, and binary/kernel provenance. Across-seed variance is
reported separately as a noise measure; it does not establish freedom from bias.
The scope covers thin, dense, heterogeneous, and backward-scattering fog, plus
absorption, mesh/null-mesh/world emitters, and primary-event diagnostics.

The installed-build matrices are in `build/metal-guiding-tests/volume-final-thin`
and `volume-final-profiles`. UI/session tests exercise both PT and BDPT with the
volume scene. Host verification passed 45 guiding and 22 transport tests.

### Equal-sample results

Mean linear RGB MSE to the high-sample reference, three independent contender
seeds at 256 camera samples:

| Medium | CPU guided | Metal PT guided | Metal BDPT guided |
| --- | ---: | ---: | ---: |
| Thin forward | 0.256392 | 0.182503 | 0.107674 |
| Dense forward | 0.404742 | 0.259052 | 0.212901 |
| Heterogeneous | 0.320959 | 0.269860 | 0.177660 |
| Backward | 0.297521 | 0.187460 | 0.335817 |

The final backward BDPT MSE remains worse than CPU, consistent with the energy
issue the user explicitly deferred. It is not reported as a correctness pass.

Measured across-seed noise variance is lower than CPU guiding by **11–41% for
Metal PT** and **29–55% for Metal BDPT** across these four fixtures. Compared
with the same unguided Metal integrator, guiding reduces this variance by
31–59% for PT and 4–64% for BDPT. These are observed results for this matrix,
not universal guarantees. No images were normalized or altered for these
calculations. `build/metal-guiding-tests/volume-final-summary.json` records
MSE independently recomputed from raw images, verified sample/light counts,
and each variance and mean-signal difference.

Both volume UI/session suites passed 19 checks each, including training and
memory changes, disabling guiding, zero/one sampling probability, public
sampling modes, lighting resets, and CPU/Metal coexistence. Both demo files
were reopened and rendered at 256×256 and 256 samples with finite output;
their saved 512×512/1,024-sample settings and file checksums were verified.

The final review additionally extends the existing surface retry stages to
primary-volume emitter continuations. This prevents a shader-cache retry from
repeating the new MIS measure conversion or film writes. A separate installed
control passed, with mean absolute RGB difference 4.05e-8 and maximum difference
1.53e-5 from the pre-guard unguided image. The 19-check BDPT session suite also
passed again with this final guard installed. These checks verify ordinary
render equivalence and session behavior, not reproduction of a specific late
texture-cache miss.

## Reproduce

Build and install with:

```sh
cmake --build build/macos_arm64_Release --target install --parallel 8
python3 tests/python/cycles_guiding_distribution.py --transport
```

Use the repository Blender Python (NumPy required) to run
`tests/python/cycles_metal_guiding_benchmark.py`. Select the volume fixture with
`--scene-script tests/python/cycles_metal_guiding_volume_scene.py`, and choose
`--scenes volume_dense volume_heterogeneous volume_backward --integrators pt bdpt
--samples 256 --resolution 64 --seeds 101 211 307 --quality-only`. Supply
`--blender`, `--output`, and optionally a matching `--reference-directory`.
The benchmark checks reference settings and fixture provenance before reuse.

Create both editable demos, reopen them, and verify them on Metal with:

```sh
BLENDER_USER_RESOURCES=/tmp/cycles-volume-demo-user \
  install/Blender.app/Contents/MacOS/Blender -b --factory-startup \
  --debug-cycles --python-exit-code 1 \
  --python tests/python/cycles_metal_guiding_volume_demo.py -- \
  --output build/metal-guiding-tests/volume-final-demo --verify
```

Demos save at 512×512, 1,024 samples, with AgX and guiding enabled. Verification
uses 256×256 and 256 samples without changing those saved settings. PT and BDPT
are separate files so the user can open either directly. No GUI instance is
started by the generator.

Delivered files (both reopened successfully in the final installed build):

- `/Users/michal/Documents/BlenderProjects/JAR/metal_volume_guiding_PT.blend`
- `/Users/michal/Documents/BlenderProjects/JAR/metal_volume_guiding_BDPT.blend`

Use the repository build at `install/Blender.app` to open these files; stock
Blender does not contain this Metal guiding implementation. The scene assets
are procedural and require no external textures.

## Research basis

- [OpenPGL](https://github.com/openpathguidinglibrary/openpgl).
- [Path Guiding in Production](https://tom94.net/data/courses/vorba19guiding/vorba19guiding.pdf).
- [PBRT volume scattering integrators](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Volume_Scattering_Integrators).

The GPU guiding backend already existed in the starting checkout. This work
validates its volume behavior, improves the primary-volume BDPT partition, and
adds reproducible volume fixtures, tests, and demos; it does not claim that the
CPU OpenPGL implementation was ported wholesale to Metal.
