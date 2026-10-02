/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_METAL

#  include "integrator/denoiser_gpu.h"

#  include "util/map.h"

CCL_NAMESPACE_BEGIN

class MetalFXDenoiserContext;

/* Implementation of denoising API which uses the MetalFX temporal denoiser and upscaler of Apple
 * GPUs. Like DLSS it works on a sequence of frames: in the viewport every frame is rendered
 * independently with a sub-pixel jitter, in final renders the frames are the samples that were
 * added to the render buffer since the previous call. */
class MetalFXDenoiser : public DenoiserGPU {
 public:
  MetalFXDenoiser(Device *denoiser_device, const DenoiseParams &params);
  ~MetalFXDenoiser() override;

  static bool is_device_supported(const DeviceInfo &device);

  /* Largest upscale factor the device supports that does not exceed the given one. */
  static float clamp_upscale_factor(const DeviceInfo &device, float upscale_factor);

 private:
  bool denoise_create_if_needed(DenoiseContext &context) override;
  bool denoise_configure_if_needed(DenoiseContext &context) override;

  /* MetalFX reads the passes straight from the render buffer in #denoise_run and writes the
   * denoised pass there, the kernels that prepare buffers for other denoisers are not used. */
  bool denoise_filter_color_preprocess(const DenoiseContext &context,
                                       const DenoisePass &pass) override;
  bool denoise_filter_color_postprocess(const DenoiseContext &context,
                                        const DenoisePass &pass) override;
  bool denoise_filter_guiding_preprocess(DenoiseContext &context) override;
  bool denoise_filter_guiding_set_fake_albedo(DenoiseContext &context) override;

  bool denoise_run(const DenoiseContext &context, const DenoisePass &pass) override;

  MetalFXDenoiserContext *ensure_pass_context(const DenoiseContext &context, PassType type);

  /* Timing of the frames at the current resolution, reported in the log. */
  struct Stats {
    int num_frames = 0;
    double denoise_time = 0.0;
    double interval_time = 0.0;
    double last_end_time = 0.0;
    int width = 0;
    int height = 0;
    int out_width = 0;
    int out_height = 0;
  };
  Stats stats_;
  void log_stats();

  /* Every denoised pass has its own temporal history. */
  map<PassType, unique_ptr<MetalFXDenoiserContext>> pass_contexts_;
};

CCL_NAMESPACE_END

#endif
