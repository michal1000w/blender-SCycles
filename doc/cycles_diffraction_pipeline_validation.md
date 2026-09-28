# Diffraction pipeline integration

This bounded suite exercises four independent features on the packaged Apple
Metal build: a shader AOV, object motion blur, depth of field, and a participating
volume. Each runs with PT, BDPT and guiding at 128 fixed samples, 720×400,
adaptive sampling OFF, denoising OFF, and no brightness normalization.

## Regression discovered and fixed

The new Principled node signature has 41 GPU material-function parameters,
including internal arguments and its output. Blender's parser allowed only 37.
Initializing the compositor exposed the `Too many parameters in function`
error, despite earlier Cycles-only renders completing. The capacity in
`source/blender/gpu/intern/gpu_material_library.hh` is now 41. It is CPU-side
shader metadata storage; the fix adds no per-ray diffraction work.

Full build passes in `/tmp/diffraction_gpu_parameter_build.log`. A separate
updated package is `build/diffraction_delivery_20260927_pipeline/Blender.app`.
Its executable SHA256 is
`dfc78c6c36d3e0873b9d50c77410ac4bc5bd7811c6ea52c0d928a78b14c3b435`.
The preceding package remains preserved with its original provenance; use this
updated package for compositor/material-graph work.

The first fixture attempt (`pipeline_delivery_v1`) failed before rendering
because the Blender 5.3 File Output node defaults to multilayer media. The second
attempt (`pipeline_delivery_v2`) exposed the parameter error and had no exported
AOV because the fixture linked to an empty file-output extension socket without
creating an item. Both fixture API issues were corrected explicitly. They are
separate from the material parser regression; neither failure is hidden.

## What the fixtures cover

- **AOV:** Principled grating materials write a value of 0.75; the compositor
  exports the pass to a linear EXR. PT, BDPT and guiding each return the expected
  maximum 0.75 and minimum zero, with finite pixels and no parser error.
- **Motion:** one of three Principled spheres translates over the shutter;
  the other two remain static. The saved animation and one-frame shutter are
  part of the fixture, rather than a post-processing blur.
- **Depth of field:** three Principled gratings occupy different depths, with
  the middle object defining the focal plane of a perspective camera at f/0.8.
  Blur is subtle at this meter-scale setup; these images are weaker visual
  evidence than the motion fixture and do not validate a numerical blur radius.
- **Volume:** Glass gratings are enclosed in a finite homogeneous scattering
  volume (density 0.045, anisotropy 0.3, four volume bounces).

Finite/nonempty output and visual inspection are integration evidence. Only
the AOV pass has a deterministic scalar expectation here. These fixtures are
not analytic transport solutions, convergence proofs, or benchmarks at equal
variance. They do not certify all motion, volume, AOV or camera combinations,
and do not implement the remaining coherent-transport or material-model gaps.

## Reproduce

The suite runner executes one Blender process at a time and stops on failure
or a per-job timeout (default 300 seconds), retaining logs and reports. Use a
fresh output directory:

```sh
python3 tests/performance/cycles_diffraction_pipeline_suite.py \
  --blender build/diffraction_delivery_20260927_pipeline/Blender.app/Contents/MacOS/Blender \
  --output tests/output/diffraction/pipeline_rerun
python3 tests/performance/cycles_diffraction_pipeline_report.py \
  --suite tests/output/diffraction/pipeline_rerun \
  --output tests/output/diffraction/pipeline_rerun_review
```

The saved Principled and Glass scenes from the earlier delivery are required
by default. Override `--principled-scene` and `--glass-scene` to use other copies.
Scene construction lives in `tests/python/cycles_diffraction_pipeline_delivery.py`.

## Completed result

All twelve jobs in `pipeline_delivery_v3` completed with finite, nonempty output.
The three AOV gates pass at unchanged tolerance, and all beauty previews were
visually inspected. Gallery and artifact checks are in
`tests/output/diffraction/pipeline_delivery_v3_review`. The ordinary Principled
Eevee regression also passes (`pipeline_eevee_regression_v1`), with no GPU
parameter parsing error. This checks material linking, not Eevee diffraction.

Diagnostic render times including preparation (seconds):

| Feature | PT | BDPT | Guiding |
|---|---:|---:|---:|
| aov | 10.18 | 92.88 | 74.76 |
| motion | 35.61 | 94.14 | 77.10 |
| dof | 28.53 | 88.08 | 70.28 |
| volume | 46.59 | 118.83 | 105.85 |

These are single integration runs with startup variation, not efficiency comparisons.
Usage at this milestone is 37% of the weekly allowance, against the agreed 43% ceiling.
