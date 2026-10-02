/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_METAL

#  include "device/memory.h"

#  include "util/string.h"
#  include "util/unique_ptr.h"

CCL_NAMESPACE_BEGIN

class Device;
class DeviceInfo;

/* One call of the MetalFX temporal denoiser: where the inputs are in the render buffer, where the
 * result goes and the state of the camera for the frame. Pass offsets are in floats from the
 * start of a pixel, -1 for a pass that is not available. */
struct MetalFXFrame {
  device_ptr render_buffer = 0;

  /* Layout of the rendered pixels, at the resolution that was path traced. */
  int width = 0;
  int height = 0;
  int full_x = 0;
  int full_y = 0;
  int offset = 0;
  int stride = 0;
  int pass_stride = 0;

  /* Noisy color, already divided by the number of samples. */
  int pass_color = -1;
  /* Feature passes, summed over the samples. */
  int pass_depth = -1;
  int pass_albedo = -1;
  int pass_normal = -1;
  int pass_roughness = -1;
  int pass_motion = -1;
  int pass_motion_weight = -1;
  /* Per pixel number of samples, as stored by adaptive sampling. */
  int pass_sample_count = -1;
  /* Number of samples in the buffer when there is no sample count pass. */
  int num_samples = 1;

  /* Layout of the denoised pixels, at the output resolution. */
  int out_width = 0;
  int out_height = 0;
  int out_full_x = 0;
  int out_full_y = 0;
  int out_offset = 0;
  int out_stride = 0;
  int pass_noisy = -1;
  int pass_denoised = -1;
  int num_components = 3;
  bool use_compositing = false;
  /* The pass is modulated by the albedo of the surfaces. Not the case for the shadow catcher
   * pass, which is a ratio of light with and without the shadows. */
  bool use_albedo = true;
  float upscale_factor = 1.0f;

  /* Sub-pixel offset of the frame, in rendered pixels, as used by the camera. */
  float jitter_x = 0.0f;
  float jitter_y = 0.0f;

  /* The render buffer keeps accumulating samples between calls, until the render restarts.
   * Otherwise every call is an independent render. */
  bool accumulate = false;
  /* A restarted render has nothing to do with the previous one, as opposed to a viewport where
   * the motion vectors relate it to what was shown before. */
  bool reset_history_on_restart = false;
  /* Forget everything from earlier calls. */
  bool reset_history = false;

  /* Camera of the frame, row-major 4x4 matrices with column vectors. View space is the camera
   * space of Cycles, clip space is the one of Metal with reversed depth. */
  float world_to_view[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  float view_to_clip[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  float view_to_world[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  /* The depth pass holds a distance from the camera instead of a view space Z. */
  bool depth_is_distance = false;
};

/* MetalFX temporal denoiser and upscaler (MTLFXTemporalDenoisedScaler), with the conversion
 * between the interleaved render buffer of Cycles and the textures MetalFX works on. */
class MetalFXDenoiserContext {
 public:
  /* Whether the Metal device with this index can run the denoiser. */
  static bool is_device_supported(const DeviceInfo &info);
  /* Range of output/input scale factors the device accepts. */
  static void supported_scale_range(const DeviceInfo &info, float &min_scale, float &max_scale);

  explicit MetalFXDenoiserContext(Device *device);
  ~MetalFXDenoiserContext();

  /* Create or recreate the scaler for these sizes. Returns false without an error when the
   * sizes are not supported, in which case there is nothing to denoise with. */
  bool ensure(int width, int height, int out_width, int out_height, string &error);

  /* Denoise one frame. The device queue must have been synchronized. */
  bool denoise(const MetalFXFrame &frame, string &error);

  /* Release the scaler and its textures. */
  void free();

 private:
  struct Impl;
  unique_ptr<Impl> impl_;
};

CCL_NAMESPACE_END

#endif /* WITH_METAL */
