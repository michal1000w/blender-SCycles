/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_METAL

#  include "integrator/denoiser_metalfx.h"

#  include "device/device.h"
#  include "device/metal/metalfx.h"
#  include "device/queue.h"

#  include "session/buffers.h"

#  include "util/log.h"
#  include "util/time.h"

CCL_NAMESPACE_BEGIN

namespace {

void matrix_multiply(const float a[16], const float b[16], float r[16])
{
  for (int row = 0; row < 4; row++) {
    for (int column = 0; column < 4; column++) {
      float sum = 0.0f;
      for (int k = 0; k < 4; k++) {
        sum += a[row * 4 + k] * b[k * 4 + column];
      }
      r[row * 4 + column] = sum;
    }
  }
}

void matrix_from_transform(const Transform &tfm, float r[16])
{
  const float4 rows[4] = {tfm.x, tfm.y, tfm.z, make_float4(0.0f, 0.0f, 0.0f, 1.0f)};
  for (int row = 0; row < 4; row++) {
    r[row * 4 + 0] = rows[row].x;
    r[row * 4 + 1] = rows[row].y;
    r[row * 4 + 2] = rows[row].z;
    r[row * 4 + 3] = rows[row].w;
  }
}

void matrix_from_projection(const ProjectionTransform &projection, float r[16])
{
  const float4 rows[4] = {projection.x, projection.y, projection.z, projection.w};
  for (int row = 0; row < 4; row++) {
    r[row * 4 + 0] = rows[row].x;
    r[row * 4 + 1] = rows[row].y;
    r[row * 4 + 2] = rows[row].z;
    r[row * 4 + 3] = rows[row].w;
  }
}

/* Fill in the camera matrices of the frame.
 *
 * View space is the camera space of Cycles. Clip space is the one of Metal: X to the right, Y up,
 * both -1..1 after the perspective division, with the depth reversed so that it is 1 at the near
 * and 0 at the far clipping plane. */
void frame_set_camera(MetalFXFrame &frame, const Denoiser::FrameInfo &info)
{
  if (!info.has_camera) {
    return;
  }

  matrix_from_transform(info.world_to_camera, frame.world_to_view);
  matrix_from_transform(info.camera_to_world, frame.view_to_world);
  frame.depth_is_distance = info.depth_is_distance;

  if (!info.depth_is_distance) {
    /* Perspective and orthographic cameras: the NDC of Cycles is 0..1 in X, Y and depth, with
     * the depth 0 at the near clipping plane. */
    const float ndc_to_clip[16] = {
        2.0f, 0.0f, 0.0f,  -1.0f, 0.0f, 2.0f, 0.0f, -1.0f,
        0.0f, 0.0f, -1.0f, 1.0f,  0.0f, 0.0f, 0.0f, 1.0f,
    };
    float world_to_ndc[16], view_to_ndc[16];
    matrix_from_projection(info.world_to_ndc, world_to_ndc);
    matrix_multiply(world_to_ndc, frame.view_to_world, view_to_ndc);
    matrix_multiply(ndc_to_clip, view_to_ndc, frame.view_to_clip);
  }
  else {
    /* Panoramic and custom cameras have no projection matrix. Describe a perspective camera
     * with a wide field of view, so that the depth has a consistent meaning. */
    const float n = (info.nearclip > 0.0f) ? info.nearclip : 0.01f;
    const float f = (info.farclip > n && info.farclip < 1e30f) ? info.farclip : n * 1e6f;
    const float clip[16] = {
        1.0f, 0.0f, 0.0f,         0.0f,            0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -n / (f - n), f * n / (f - n), 0.0f, 0.0f, 1.0f, 0.0f,
    };
    memcpy(frame.view_to_clip, clip, sizeof(clip));
  }

  for (float &value : frame.view_to_clip) {
    if (!isfinite(value)) {
      value = 0.0f;
    }
  }
}

}  // namespace

MetalFXDenoiser::MetalFXDenoiser(Device *denoiser_device, const DenoiseParams &params)
    : DenoiserGPU(denoiser_device, params)
{
}

void MetalFXDenoiser::log_stats()
{
  if (stats_.num_frames == 0) {
    return;
  }
  const string message = string_printf(
      "MetalFX: %d frames %dx%d to %dx%d, %.2f ms per frame for denoising, %.2f ms between frames",
      stats_.num_frames,
      stats_.width,
      stats_.height,
      stats_.out_width,
      stats_.out_height,
      1000.0 * stats_.denoise_time / stats_.num_frames,
      1000.0 * stats_.interval_time / max(stats_.num_frames - 1, 1));
  LOG_INFO << message;
  /* For benchmarks, without the rest of the log. */
  if (getenv("CYCLES_METALFX_STATS")) {
    fprintf(stderr, "%s\n", message.c_str());
  }
  stats_ = Stats();
}

MetalFXDenoiser::~MetalFXDenoiser()
{
  log_stats();

  /* Wait for anything that still uses the textures. */
  denoiser_queue_->synchronize();
  pass_contexts_.clear();
}

bool MetalFXDenoiser::is_device_supported(const DeviceInfo &device)
{
  /* When rendering with several devices the denoiser runs on the one that supports it. */
  if (!device.multi_devices.empty()) {
    for (const DeviceInfo &sub_device : device.multi_devices) {
      if (is_device_supported(sub_device)) {
        return true;
      }
    }
    return false;
  }

  if (device.type != DEVICE_METAL) {
    return false;
  }

  /* Called often, and the answer for a device does not change. */
  static thread_mutex cache_mutex;
  static map<int, bool> cache;
  const thread_scoped_lock lock(cache_mutex);
  if (const auto it = cache.find(device.num); it != cache.end()) {
    return it->second;
  }
  const bool supported = MetalFXDenoiserContext::is_device_supported(device);
  cache.emplace(device.num, supported);
  return supported;
}

float MetalFXDenoiser::clamp_upscale_factor(const DeviceInfo &device, const float upscale_factor)
{
  float min_scale, max_scale;
  MetalFXDenoiserContext::supported_scale_range(device, min_scale, max_scale);
  return clamp(upscale_factor, 1.0f, max(max_scale, 1.0f));
}

MetalFXDenoiserContext *MetalFXDenoiser::ensure_pass_context(const DenoiseContext &context,
                                                             const PassType type)
{
  unique_ptr<MetalFXDenoiserContext> &pass_context = pass_contexts_[type];
  if (!pass_context) {
    pass_context = make_unique<MetalFXDenoiserContext>(denoiser_device_);
  }

  string error;
  if (!pass_context->ensure(context.buffer_params.width,
                            context.buffer_params.height,
                            context.denoised_buffer_params.width,
                            context.denoised_buffer_params.height,
                            error))
  {
    if (!error.empty()) {
      set_error(error);
    }
    return nullptr;
  }
  return pass_context.get();
}

bool MetalFXDenoiser::denoise_create_if_needed(DenoiseContext &context)
{
  /* Recreating the denoiser releases textures that a command buffer may still use. */
  denoiser_queue_->synchronize();

  if (stats_.num_frames > 0 && (stats_.width != context.buffer_params.width ||
                                stats_.height != context.buffer_params.height ||
                                stats_.out_width != context.denoised_buffer_params.width ||
                                stats_.out_height != context.denoised_buffer_params.height))
  {
    log_stats();
  }

  /* Passes that are no longer rendered do not need their denoiser. */
  for (auto it = pass_contexts_.begin(); it != pass_contexts_.end();) {
    if (context.buffer_params.get_pass_offset(it->first, PassMode::NOISY) == PASS_UNUSED) {
      it = pass_contexts_.erase(it);
    }
    else {
      ++it;
    }
  }

  /* Without a denoiser for these sizes there is no denoised result, which is not an error:
   * the noisy render is shown instead. */
  return ensure_pass_context(context, PASS_COMBINED) != nullptr;
}

bool MetalFXDenoiser::denoise_configure_if_needed(DenoiseContext & /*context*/)
{
  return true;
}

bool MetalFXDenoiser::denoise_filter_color_preprocess(const DenoiseContext & /*context*/,
                                                      const DenoisePass & /*pass*/)
{
  return true;
}

bool MetalFXDenoiser::denoise_filter_color_postprocess(const DenoiseContext & /*context*/,
                                                       const DenoisePass & /*pass*/)
{
  return true;
}

bool MetalFXDenoiser::denoise_filter_guiding_preprocess(DenoiseContext & /*context*/)
{
  return true;
}

bool MetalFXDenoiser::denoise_filter_guiding_set_fake_albedo(DenoiseContext & /*context*/)
{
  return true;
}

bool MetalFXDenoiser::denoise_run(const DenoiseContext &context, const DenoisePass &pass)
{
  /* The noisy pass has been copied to the denoised pass by kernels that are still queued. */
  if (!denoiser_queue_->synchronize()) {
    return false;
  }

  MetalFXDenoiserContext *pass_context = ensure_pass_context(context, pass.type);
  if (!pass_context) {
    return false;
  }

  const BufferParams &buffer_params = context.buffer_params;
  const BufferParams &denoised_buffer_params = context.denoised_buffer_params;

  MetalFXFrame frame;
  frame.render_buffer = context.render_buffers->buffer.device_pointer;

  frame.width = buffer_params.width;
  frame.height = buffer_params.height;
  frame.full_x = buffer_params.full_x;
  frame.full_y = buffer_params.full_y;
  frame.offset = buffer_params.offset;
  frame.stride = buffer_params.stride;
  frame.pass_stride = buffer_params.pass_stride;

  frame.pass_color = pass.denoised_offset;
  frame.pass_depth = buffer_params.get_pass_offset(PASS_DENOISING_DEPTH);
  frame.pass_albedo = buffer_params.get_pass_offset(PASS_DENOISING_ALBEDO);
  frame.pass_normal = buffer_params.get_pass_offset(PASS_DENOISING_NORMAL);
  frame.pass_roughness = buffer_params.get_pass_offset(PASS_DENOISING_ROUGHNESS);
  frame.pass_sample_count = context.pass_sample_count;
  frame.num_samples = context.num_samples;

  frame.accumulate = frame_info_.accumulate;
  frame.reset_history_on_restart = !frame_info_.interactive;
  if (frame_info_.interactive && (context.denoise_params.passes & DENOISER_PASS_MOTION)) {
    frame.pass_motion = buffer_params.get_pass_offset(PASS_MOTION);
    frame.pass_motion_weight = buffer_params.get_pass_offset(PASS_MOTION_WEIGHT);
  }

  frame.out_width = denoised_buffer_params.width;
  frame.out_height = denoised_buffer_params.height;
  frame.out_full_x = denoised_buffer_params.full_x;
  frame.out_full_y = denoised_buffer_params.full_y;
  frame.out_offset = denoised_buffer_params.offset;
  frame.out_stride = denoised_buffer_params.stride;
  frame.pass_noisy = pass.noisy_offset;
  frame.pass_denoised = pass.denoised_offset;
  frame.num_components = pass.num_components;
  frame.use_compositing = Pass::get_info(pass.type).use_compositing;
  frame.use_albedo = pass.use_denoising_albedo;
  frame.upscale_factor = params_.upscale_factor;

  /* No jitter is stored as FLT_MAX: the pixel filter is sampled instead. */
  if (context.pixel_jitter.x != FLT_MAX && isfinite(context.pixel_jitter.x) &&
      isfinite(context.pixel_jitter.y))
  {
    /* The integrator moved the jitter to the middle of the pixel for MetalFX, which wants the
     * offset from there. */
    frame.jitter_x = context.pixel_jitter.x + 0.5f;
    frame.jitter_y = context.pixel_jitter.y + 0.5f;
  }

  frame_set_camera(frame, frame_info_);

  const double start_time = time_dt();

  string error;
  if (!pass_context->denoise(frame, error)) {
    if (!error.empty()) {
      set_error(error);
    }
    return false;
  }

  if (pass.type == PASS_COMBINED) {
    const double end_time = time_dt();
    if (stats_.num_frames > 0) {
      stats_.interval_time += end_time - stats_.last_end_time;
    }
    stats_.last_end_time = end_time;
    stats_.denoise_time += end_time - start_time;
    stats_.num_frames++;
    stats_.width = frame.width;
    stats_.height = frame.height;
    stats_.out_width = frame.out_width;
    stats_.out_height = frame.out_height;
  }

  return true;
}

CCL_NAMESPACE_END

#endif
