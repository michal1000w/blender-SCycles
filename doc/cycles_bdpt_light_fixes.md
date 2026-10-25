# BDPT: clip start precision and area light spread (2026-10-25)

Two bugs of the fork's bidirectional path tracer, found while validating the Diffraction BSDF's
Grating model. Both existed before (`build/diffraction_hybrid/Blender_before.app`), on CPU and
Metal, with and without path guiding. Uncommitted in the working tree.

## 1. Light subpath to camera connections dropped at small clip starts

`bdpt_sample_camera_endpoint()` and `bdpt_perspective_ray_to_raster()` solved for the raster
position on the image plane of `rastertocamera`, which lies at the near clipping distance, and
rejected the connection when the 2x2 determinant was below `1e-20`. The determinant is about the
fourth power of a pixel's size on that plane: with a 1 mm clip start and 96 to 192 px, every
light-to-camera connection failed while the MIS weights still gave that strategy its share. BDPT
lost up to 20% of the energy, depending on clip start, resolution and how much weight light
tracing gets (more with area lights than with point lights).

Fix: `bdpt_image_plane_to_raster()` solves on the plane at unit distance and only rejects
(nearly) parallel pixel axes, a scale-free test.

## 2. Area light spread ignored by light subpaths

`photon_sample_emitter()` (light subpaths of BDPT, VCM and photon mapping) passed the emission
direction to `area_light_spread_attenuation()`, which takes the direction towards the light. For
directions away from the light the tangent is negative, so the attenuation never reached zero:
light was emitted over the whole hemisphere with the cone's normalization, 5.5x too bright on a
floor under a 60 degree spread light, 27x in a caustic.

Fix (default 180 degree lights are unchanged):
* The attenuation gets the direction towards the light.
* Lights with 0 < spread < 180 degrees launch light subpaths cosine-weighted inside the spread
  cone (`area_light_sample_emission_direction()`, `area_light_emission_direction_pdf()` in
  `light/area.h`), instead of wasting most of them outside of it. The BDPT MIS weights that
  assumed the hemisphere density (`bdpt_emission_mis_weight_lamp()`, `bdpt_nee_mis_weight()`,
  `bdpt_volume_nee_mis_weight()`) use that density through
  `bdpt_area_emission_direction_ratio()`.
* A spread of 0 (collimated beam) cannot be sampled by light subpaths: such lights get no light
  subpaths and the regular two-strategy weights, like the fork's disk-impostor point lights.

## Tests

`tests/python/cycles_bdpt_light_scenes.py`: one light of each kind (point, spot, square, disk,
rectangle, large, spread 60/10/0 degrees, sun, mesh, world, offset and frame-edge variants) above
a diffuse floor, and caustics through a glass sphere (area, spread, mesh, rough glass, IOR 2),
rendered with path tracing and BDPT / VCM / photon mapping; compares image means.
`--clip-start`, `--guiding`, `--integrators`, `--no-light-tree`, `--no-polarization`.

Results after the fix: every scene within 0.1% of path tracing on CPU and Metal, guiding on and
off, clip start 0.001 and 0.1; caustic scenes within 0.4% for BDPT, VCM and photon mapping.

Note for such comparisons: Filter Glossy (`blur_glossy`, default 1) blurs glass after a diffuse
bounce of camera paths only. It is biased by design and makes path tracing disagree with light
subpath strategies on caustics (6% in the glass sphere scene); the test sets it to 0.
