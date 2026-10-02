# MetalFX denoising and upscaling for Cycles on Metal

Counterpart of the NVIDIA DLSS Ray Reconstruction denoiser (`integrator/denoiser_dlss.cpp`) for
Apple GPUs, built on `MTLFXTemporalDenoisedScaler` (MetalFX, macOS 26 and newer). It is built
against the macOS 27 SDK and was run on macOS 27; the denoiser API itself gained nothing in 27
that Cycles has data for, so no 27-only call is made.

## Status (2026-10-02)

Working in the viewport (denoising, upscaling up to 3x, camera and object motion) and in final
renders (denoising). Tested on an Apple M5 with macOS 27; nothing is committed.

## How to use

* Viewport: Render properties > Sampling > Viewport > Denoise, denoiser "MetalFX", then pick an
  upscale mode: None, Quality (1.5x), Balanced (1.72x), Performance (2x), Ultra Performance (3x).
* Final render: Render properties > Sampling > Render > Denoise, denoiser "MetalFX".
* Python: `scene.cycles.preview_denoiser = 'METALFX'`,
  `scene.cycles.preview_denoising_upscale_quality = 'PERFORMANCE'`,
  `scene.cycles.denoiser = 'METALFX'`.
* On a device without MetalFX (CPU rendering without a Metal GPU for denoising, macOS before 26)
  OpenImageDenoise is used instead and the panel says so.
* Diagnostics: `CYCLES_METALFX_SHOW=color|depth|motion|albedo|normal|roughness` shows that input
  as MetalFX gets it, `CYCLES_METALFX_STATS=1` prints frame timings to the terminal.

## What was built

| File | Role |
| --- | --- |
| `device/metal/metalfx.{h,mm}` | The scaler, its textures and two compute kernels (MSL compiled at run time) that convert between the render buffer and the textures. All Objective-C is here. |
| `integrator/denoiser_metalfx.{h,cpp}` | `MetalFXDenoiser`, a `DenoiserGPU`. One scaler per denoised pass (combined, shadow catcher, shadow catcher matte), each with its own history. |
| `integrator/denoiser.h` | `Denoiser::FrameInfo`: camera of the frame and whether the buffer accumulates. |
| `integrator/render_scheduler.*` | "Progressive" denoiser mode: denoise after every batch of samples. |
| `scene/film.cpp`, `scene/integrator.cpp` | Feature options and jitter while MetalFX is active. |
| `blender/sync.cpp`, `blender/python.cpp`, `blender/addon/*.py` | Settings, device capability, UI. |
| `kernel/data_template.h` | Fix of the layout of `pixel_jitter` (see below). |

### How frames are fed

MetalFX accumulates over a sequence of noisy frames, but only over a short history. Measured,
not assumed:

* **Viewport**. A change of the scene restarts the render, as always, and the first frame goes
  to MetalFX with motion vectors, so the image stays sharp while navigating (verified against
  zeroed and flipped motion vectors, which smear). While nothing changes the render buffer keeps
  accumulating like a normal viewport render and MetalFX gets what was added since the last
  frame, with a new sub-pixel jitter per frame. Compared with restarting every frame as the DLSS
  integration does, this is 0.9 to 1.8 dB better after the same time, keeps improving, honours
  the viewport sample count and stops when it is reached.
* **Final render**. The samples are split into 8 batches (best of 2 to 64 at 16, 64, 256 and
  1024 samples) and MetalFX runs after each. No jitter: the pixel filter stays, so every other
  pass, including "Noisy Image", is the same as in a render without MetalFX.
* The features (albedo, normal, roughness, depth) are the average of all samples so far, only
  the color is per frame.
* Tiled renders are denoised once at the end from a single frame, which is weaker.

### Inputs, as measured

| Choice | Effect in the test scenes |
| --- | --- |
| Features follow sharp reflections and refractions, rough ones use the surface itself | mirrors and glass stay sharp instead of blurred; +2 to +5 dB |
| No specular albedo | +1 to +3 dB, and no energy loss on metals (mean 0.97 instead of 0.90) |
| Roughness scaled by 0.5 | up to +5.5 dB in diffuse interiors: MetalFX barely denoises at roughness 1 |
| Automatic exposure | +1.4 to +2 dB at few samples, and independent of the scene's light levels |
| World space normals | slightly better than view space, as Apple documents |
| Jitter sign (x, -y) | the only one of four that is sharp when upscaling |
| Neutral albedo for the shadow catcher pass | otherwise the catcher's texture shows in its shadow |

Depth and the camera matrices made no measurable difference in still scenes.

### Bugs found on the way

* `KernelIntegrator::pixel_jitter` was a `float2` at an offset that is not a multiple of 8 after
  the fork added integrator members. The Metal shading language aligns it to 8, so the GPU read
  the jitter 4 bytes off: X got the value of Y and Y was garbage. The "pixel jitter" option was
  broken on Metal. Fixed with a padding member.
* `pixel_jitter` was a specialization constant of the Metal kernels although it changes every
  frame. Marked as not specialized.
* The kernel subtracts the jitter from the integer raster position, the corner of the pixel,
  while the pixel filter is centred on the middle: jittered frames are half a pixel off. For
  MetalFX the jitter is moved to the middle in `Integrator::device_update`. Left as it is for
  DLSS and the plain option.
* At the largest upscale factor the render size is rounded down and the ratio ends up above
  what MetalFX accepts (2096 / 698 > 3). MetalFX then scales to a few pixels less and the last
  rows and columns are extended.

## Results

Final renders, 960x540, PSNR against a 4096 sample reference (higher is better) and render
time, `tests/python/cycles_metalfx_scenes.py`:

| Scene | Samples | Noisy | OpenImageDenoise | MetalFX | Time noisy / OIDN / MetalFX |
| --- | --- | --- | --- | --- | --- |
| materials | 4 | 20.9 | 36.9 | 29.5 | 0.60 / 1.00 / 0.80 s |
| materials | 16 | 27.0 | 41.0 | 32.1 | 0.91 / 1.31 / 1.39 s |
| materials | 64 | 33.0 | 43.6 | 36.8 | 2.15 / 2.58 / 2.68 s |
| materials | 256 | | 46.4 | 38.9 | - / 7.82 / 7.88 s |
| interior | 4 | 19.1 | 43.1 | 37.0 | 0.33 / 0.72 / 0.48 s |
| interior | 16 | 27.3 | 46.6 | 40.0 | 0.67 / 1.04 / 1.07 s |
| interior | 64 | 34.4 | 49.1 | 43.1 | 2.15 / 2.55 / 2.64 s |
| interior | 256 | | 50.9 | 44.2 | - / 8.93 / 9.02 s |
| detail | 4 | 24.4 | 33.6 | 27.3 | 0.26 / 0.63 / 0.41 s |
| detail | 16 | 31.9 | 38.6 | 32.2 | 0.45 / 0.82 / 0.78 s |
| detail | 64 | 39.1 | 43.3 | 36.9 | 1.25 / 1.65 / 1.70 s |
| detail | 256 | | 48.3 | 38.2 | - / 5.17 / 5.36 s |

For stills OpenImageDenoise is clearly the more accurate denoiser, by 5 to 10 dB. MetalFX is a
real-time denoiser: its place is the viewport.

Viewport, 2096x1388, Apple M5 (10 GPU cores), `tests/python/cycles_metalfx_viewport.py`:

| Mode | Rendered at | Time per frame | MetalFX itself | PSNR after 6 s | after 20 s |
| --- | --- | --- | --- | --- | --- |
| no denoiser | 2096x1388 | 180 ms per sample | | 28.0 | |
| OpenImageDenoise | 2096x1388 | | | 39.7 | 42.2 |
| MetalFX, no upscaling | 2096x1388 | 300 ms | 26 ms | 33.0 | 36.5 |
| MetalFX Quality | 1397x925 | 145 ms | 16 ms | 33.7 | |
| MetalFX Balanced | 1215x805 | 118 ms | 14 ms | 32.7 | |
| MetalFX Performance | 1048x694 | 105 ms | 12 ms | 32.9 | 36.1 |
| MetalFX Ultra Performance | 698x462 | 75 ms | 8 ms | 28.9 | 30.1 |

The frame time is path tracing; MetalFX is about a tenth of it. Performance mode gives the image
of the full resolution mode at a third of the frame time.

Before and after, renders that do not use MetalFX: same render times (2.15 / 2.15 / 1.25 s for
64 samples before and after), images equal to within the run-to-run difference of Metal
(over 110 dB), bit-identical on the CPU.

Also run without errors: orthographic and panoramic cameras, volumes, shadow catcher with
transparent film, adaptive sampling, a time limit, 1 sample, 40x24 and 3840x2160 pixels, tiles,
rendering on the CPU with denoising on the GPU, GPU and CPU together, an animation (no ghost of
the previous frame), and in the viewport switching between denoisers and upscale modes in one
session, editing a material, moving an object, orbiting the camera.

## Not done

* Upscaling in final renders. The render buffer holds every pass at the rendered resolution and
  only the denoised pass at the output resolution, so the other passes of the file would be
  wrong. It needs the output driver to resample them.
* MetalFX has an optional specular hit distance input; Cycles has no such pass.
* Residual grain in glass and in volumes while moving: the features behind a refraction are
  those of one random path per sample.
* Not run: the upstream render test suite, a second Metal GPU, a real viewport resize by
  dragging (changing the upscale mode, which also recreates the scaler, is tested).

## Implementation plan

The plan as written before the implementation. Where measurements led elsewhere (specular
albedo, features of the first surface, restarting the viewport every frame) the sections above
describe what was built.

### What MetalFX needs (research)

Sources: the macOS 27 SDK headers (`MetalFX.framework/Headers/MTLFXTemporalDenoisedScaler.h`),
WWDC25 session 211 "Go further with Metal 4 games", a probe program on the Apple M5.

| Input | MetalFX expectation | Cycles source | Conversion |
| --- | --- | --- | --- |
| color | linear HDR, noisy, jittered frame | combined pass | divide by samples |
| depth | clip-space depth of `viewToClipMatrix`, reversed Z by default | `PASS_DENOISING_DEPTH` (camera Z) | project with the matrix handed to MetalFX |
| motion | pixels, pointing to the previous position, origin top-left | `PASS_MOTION` xy (previous - current, origin bottom-left) | flip Y |
| normal | world space, signed format | `PASS_DENOISING_NORMAL` (camera space) | rotate with camera-to-world |
| diffuse albedo | diffuse base colour, dark for metals | `PASS_DENOISING_ALBEDO` | none |
| specular albedo | noise-free specular reflectance with Fresnel | `PASS_DENOISING_SPECULAR_ALBEDO` | none |
| roughness | linear (perceptual) roughness | `PASS_DENOISING_ROUGHNESS` | none |
| jitter | sub-pixel offset of the frame | `KernelIntegrator::pixel_jitter` | sign found by measurement |
| matrices | world-to-view, view-to-clip | `KernelCamera` | built per frame |

The device probe: `supportsDevice` is true on the M5, scale factors 1.0 to 3.0 are accepted, 4.0
is refused, inputs as small as 8 pixels are accepted.

### Design

1. **Device layer** (`device/metal/metalfx.{h,mm}`): everything Objective-C. Owns the scaler, the
   input and output textures and two small compute pipelines compiled from MSL source at run time:
   one unpacks the interleaved render buffer into the textures MetalFX wants, one writes the
   output texture back into the denoised pass. No change to the Cycles kernels or the metallibs.
2. **Denoiser** (`integrator/denoiser_metalfx.{h,cpp}`): a `DenoiserGPU` like the DLSS one, plain
   C++, calls the device layer.
3. **Shared plumbing**: `DENOISER_METALFX` type, `Denoiser::set_frame_info()` to hand the camera
   of the frame to the denoiser, the scheduler's "interactive denoiser" mode extended to MetalFX,
   primary-hit denoising features (no "follow reflections") while MetalFX is active.
4. **Viewport**: same model as DLSS. Every frame is an independent jittered render, MetalFX
   accumulates over time, with optional upscaling (Quality 1.5x, Balanced 1.72x, Performance 2x,
   Ultra Performance 3x).
5. **Final render**: the samples of a render are split into sub-frames. The render buffer keeps
   accumulating as usual, so every other pass is identical to a render without MetalFX; the
   device layer feeds MetalFX the difference between consecutive states of the buffer.
6. **UI**: MetalFX entry in the viewport and render denoiser menus when a Metal device supports
   it, upscale mode for the viewport.

### Verification

* Baseline before any change: render times and images without denoising and with
  OpenImageDenoise, a 4096 sample reference per scene (`tests/python/cycles_metalfx_scenes.py`).
* After: same runs must be unchanged in time and image; MetalFX results are compared against the
  reference (PSNR, relative MSE) next to OpenImageDenoise at equal sample counts.
* Input conventions that the documentation leaves open (jitter sign, depth, motion direction) are
  settled by measurement, not by assumption.
* Viewport: a scripted Blender session with a rendered viewport, still and moving camera,
  every upscale mode, viewport resize, switching denoisers.
