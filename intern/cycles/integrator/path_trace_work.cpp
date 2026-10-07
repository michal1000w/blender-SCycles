/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "device/device.h"

#include "integrator/path_trace_display.h"
#include "integrator/path_trace_work.h"
#include "integrator/path_trace_work_cpu.h"
#include "integrator/path_trace_work_gpu.h"

#include "scene/film.h"
#include "scene/scene.h"

#include "session/buffers.h"

#include "kernel/types.h"

CCL_NAMESPACE_BEGIN

bool path_trace_use_bidirectional(const DeviceScene *device_scene)
{
  if (!device_scene->data.integrator.use_bidirectional_path_tracing) {
    return false;
  }

  if (device_scene->data.integrator.use_vertex_merging) {
    /* Merging does not invert the camera, see bdpt_recursion_supported(). */
    return true;
  }

  const KernelCamera &camera = device_scene->data.cam;
  const CameraType camera_type = CameraType(camera.type);
  if (camera.interocular_offset != 0.0f || camera_type == CAMERA_CUSTOM) {
    return false;
  }
  return camera_type == CAMERA_PERSPECTIVE ||
         ((camera_type == CAMERA_PANORAMA || camera_type == CAMERA_ORTHOGRAPHIC) &&
          camera.aperturesize == 0.0f && camera.num_motion_steps == 0);
}

uint vertex_merging_path_slots(const KernelIntegrator &integrator)
{
  /* A subpath that only keeps the vertices of caustics stops at the first surface that is not
   * sharp, and rarely has more than one. With more vertices to keep than slots a subpath keeps
   * a random subset that stands for all, see vcm_finish_light_path() in the kernel. */
  const uint bounces = uint(integrator.bdpt_max_bounces);
  return integrator.vcm_caustics_only ? 2u : min(bounces + 4u, 8u);
}

uint bidirectional_light_paths(const KernelIntegrator &integrator,
                               const int width,
                               const int height,
                               const int update_samples)
{
  const uint64_t pixels = uint64_t(max(width, 1)) * uint64_t(max(height, 1));
  if (integrator.use_vertex_merging && !integrator.bdpt_use_connections &&
      integrator.vcm_light_path_ratio > 0.0f)
  {
    /* The light subpaths only provide vertices to merge with, and a camera path merges within
     * a few pixels: keep their number in proportion to the camera paths of one update. */
    const double paths = double(integrator.vcm_light_path_ratio) * double(pixels) *
                         double(max(update_samples, 1));
    return uint(std::min(std::max(paths, 1024.0), 4.0 * 1024.0 * 1024.0));
  }
  /* Keep the light-subpath density constant as image resolution changes. The setting is the
   * budget at the scene's full render resolution; previews and cropped buffers receive the
   * proportional share. */
  const uint64_t scaled_light_paths = uint64_t(integrator.bdpt_light_paths) * pixels;
  const uint64_t reference_pixels = uint64_t(max(integrator.bdpt_reference_pixels, 1));
  const uint64_t scaled_count = (scaled_light_paths + reference_pixels - 1u) / reference_pixels;
  return uint(std::clamp(scaled_count, uint64_t(1), uint64_t(integrator.bdpt_light_paths)));
}

VertexMergingRadius vertex_merging_radius(const KernelIntegrator &integrator,
                                          const KernelCamera &camera,
                                          const int iteration,
                                          const uint light_paths)
{
  /* Progressive radius reduction of Georgiev et al.: the variance of merging stays bounded while
   * its blur vanishes. */
  const float shrink = powf(float(iteration + 1), 0.5f * (integrator.vcm_radius_alpha - 1.0f));
  const float pixels = max(integrator.vcm_radius_pixels, 1.0e-3f) * shrink;
  const float scene_radius = max(integrator.photon_scene.w, 1.0e-6f);

  /* The footprint of a pixel grows with the distance from a perspective or panoramic camera,
   * and is the same everywhere for an orthographic one. */
  VertexMergingRadius result;
  /* Twice the radius at the distance of the middle of the scene. */
  float largest;
  if (camera.type == CAMERA_ORTHOGRAPHIC) {
    result.radius_base = pixels * 0.5f * (len(make_float3(camera.dx)) + len(make_float3(camera.dy)));
    result.radius_slope = 0.0f;
    largest = 2.0f * result.radius_base;
  }
  else {
    float pixel_angle;
    if (camera.type == CAMERA_PERSPECTIVE) {
      /* The raster derivatives are lengths on the plane that the raster unprojects to. */
      const ProjectionTransform raster_to_camera = camera.rastertocamera;
      const float3 center = transform_perspective(
          &raster_to_camera, make_float3(0.5f * camera.width, 0.5f * camera.height, 0.0f));
      pixel_angle = 0.5f * (len(make_float3(camera.dx)) + len(make_float3(camera.dy))) /
                    max(len(center), 1.0e-8f);
    }
    else {
      const float fov = (camera.panorama_type == PANORAMA_FISHEYE_EQUIDISTANT ||
                         camera.panorama_type == PANORAMA_FISHEYE_EQUISOLID) ?
                            camera.fisheye_fov :
                            M_2PI_F;
      pixel_angle = fov / max(camera.width, 1.0f);
    }
    result.radius_base = 0.0f;
    result.radius_slope = pixels * max(pixel_angle, 1.0e-8f);
    const float3 camera_P = make_float3(
        camera.cameratoworld.x.w, camera.cameratoworld.y.w, camera.cameratoworld.z.w);
    largest = 2.0f * result.radius_slope *
              max(len(camera_P - make_float3(integrator.photon_scene)), 0.25f * scene_radius);
  }
  /* The cells of the grid are as wide as the largest radius, and a camera vertex walks through
   * all vertices of the cells that its merge disk touches: keep them close to the disks of most
   * camera vertices, and bound them in the scene for a camera that is far away. Paths with a
   * larger footprint merge within this radius. */
  const float bound = (integrator.vcm_radius > 0.0f ? integrator.vcm_radius :
                                                      0.01f * scene_radius) *
                      shrink;
  result.radius = max(min(largest, bound), 1.0e-6f);
  result.light_paths = light_paths;
  /* A camera vertex only merges with the subpaths of its time bin, and counts each as many. */
  result.eta_scale = M_PI_F * float(light_paths) / float(max(integrator.photon_time_bins, 1));
  return result;
}

unique_ptr<PathTraceWork> PathTraceWork::create(Device *device,
                                                Film *film,
                                                DeviceScene *device_scene,
                                                const bool *cancel_requested_flag)
{
  if (device->info.type == DEVICE_CPU) {
    return make_unique<PathTraceWorkCPU>(device, film, device_scene, cancel_requested_flag);
  }
  if (device->info.type == DEVICE_DUMMY) {
    /* Dummy devices can't perform any work. */
    return nullptr;
  }

  return make_unique<PathTraceWorkGPU>(device, film, device_scene, cancel_requested_flag);
}

PathTraceWork::PathTraceWork(Device *device,
                             Film *film,
                             DeviceScene *device_scene,
                             const bool *cancel_requested_flag)
    : device_(device),
      film_(film),
      device_scene_(device_scene),
      buffers_(make_unique<RenderBuffers>(device)),
      effective_buffer_params_(buffers_->params),
      effective_denoised_buffer_params_(buffers_->params),
      cancel_requested_flag_(cancel_requested_flag)
{
}

PathTraceWork::~PathTraceWork() = default;

RenderBuffers *PathTraceWork::get_render_buffers()
{
  return buffers_.get();
}

void PathTraceWork::set_effective_buffer_params(
    const BufferParams &effective_big_tile_params,
    const BufferParams &effective_buffer_params,
    const BufferParams &effective_denoised_big_tile_params,
    const BufferParams &effective_denoised_buffer_params)
{
  effective_big_tile_params_ = effective_big_tile_params;
  effective_buffer_params_ = effective_buffer_params;
  effective_denoised_big_tile_params_ = effective_denoised_big_tile_params;
  effective_denoised_buffer_params_ = effective_denoised_buffer_params;
}

bool PathTraceWork::has_multiple_works() const
{
  /* Assume if there are multiple works working on the same big tile none of the works gets the
   * entire big tile to work on. */
  return !(effective_big_tile_params_.width == effective_buffer_params_.width &&
           effective_big_tile_params_.height == effective_buffer_params_.height &&
           effective_big_tile_params_.full_x == effective_buffer_params_.full_x &&
           effective_big_tile_params_.full_y == effective_buffer_params_.full_y);
}

void PathTraceWork::copy_to_render_buffers(RenderBuffers *render_buffers)
{
  copy_render_buffers_from_device();

  const int64_t width = effective_buffer_params_.width;
  const int64_t height = effective_buffer_params_.height;
  const int64_t pass_stride = effective_buffer_params_.pass_stride;
  const int64_t row_stride = width * pass_stride;
  const int64_t data_size = row_stride * height * sizeof(float);

  const int64_t offset_y = effective_buffer_params_.full_y - effective_big_tile_params_.full_y;
  const int64_t offset_in_floats = offset_y * row_stride;

  const float *src = buffers_->buffer.data();
  float *dst = render_buffers->buffer.data() + offset_in_floats;

  memcpy(dst, src, data_size);
}

void PathTraceWork::copy_from_render_buffers(const RenderBuffers *render_buffers)
{
  const int64_t width = effective_buffer_params_.width;
  const int64_t height = effective_buffer_params_.height;
  const int64_t pass_stride = effective_buffer_params_.pass_stride;
  const int64_t row_stride = width * pass_stride;
  const int64_t data_size = row_stride * height * sizeof(float);

  const int64_t offset_y = effective_buffer_params_.full_y - effective_big_tile_params_.full_y;
  const int64_t offset_in_floats = offset_y * row_stride;

  const float *src = render_buffers->buffer.data() + offset_in_floats;
  float *dst = buffers_->buffer.data();

  memcpy(dst, src, data_size);

  copy_render_buffers_to_device();
}

void PathTraceWork::copy_from_denoised_render_buffers(const RenderBuffers *render_buffers)
{
  const int64_t width = effective_denoised_buffer_params_.width;
  const int64_t offset_y = effective_denoised_buffer_params_.full_y -
                           effective_denoised_big_tile_params_.full_y;
  const int64_t offset = offset_y * width;

  render_buffers_host_copy_denoised(buffers_.get(),
                                    effective_denoised_buffer_params_,
                                    render_buffers,
                                    effective_denoised_buffer_params_,
                                    offset);

  copy_render_buffers_to_device();
}

bool PathTraceWork::get_render_tile_pixels(const PassAccessor &pass_accessor,
                                           const PassAccessor::Destination &destination)
{
  const int offset_y = (effective_buffer_params_.full_y + effective_buffer_params_.window_y) -
                       (effective_big_tile_params_.full_y + effective_big_tile_params_.window_y);
  const int width = effective_buffer_params_.width;

  PassAccessor::Destination slice_destination = destination;
  slice_destination.offset += offset_y * width;

  return pass_accessor.get_render_tile_pixels(buffers_.get(), slice_destination);
}

bool PathTraceWork::set_render_tile_pixels(PassAccessor &pass_accessor,
                                           const PassAccessor::Source &source)
{
  const int offset_y = effective_buffer_params_.full_y - effective_big_tile_params_.full_y;
  const int width = effective_buffer_params_.width;

  PassAccessor::Source slice_source = source;
  slice_source.offset += offset_y * width;

  return pass_accessor.set_render_tile_pixels(buffers_.get(), slice_source);
}

PassAccessor::PassAccessInfo PathTraceWork::get_display_pass_access_info(PassMode pass_mode) const
{
  const KernelFilm &kfilm = device_scene_->data.film;
  const KernelBackground &kbackground = device_scene_->data.background;

  const BufferParams &params = buffers_->params;

  const BufferPass *display_pass = params.get_actual_display_pass(film_->get_display_pass());
  if (display_pass == nullptr) {
    /* Happens when interactive session changes display pass but render
     * buffer does not contain it yet. */
    return PassAccessor::PassAccessInfo();
  }

  PassAccessor::PassAccessInfo pass_access_info;
  pass_access_info.type = display_pass->type;
  pass_access_info.offset = PASS_UNUSED;

  if (pass_mode == PassMode::DENOISED) {
    pass_access_info.mode = PassMode::DENOISED;
    pass_access_info.offset = params.get_pass_offset(pass_access_info.type, PassMode::DENOISED);
  }

  if (pass_access_info.offset == PASS_UNUSED) {
    pass_access_info.mode = PassMode::NOISY;
    pass_access_info.offset = params.get_pass_offset(pass_access_info.type);
  }

  pass_access_info.use_approximate_shadow_catcher = kfilm.use_approximate_shadow_catcher;
  pass_access_info.use_approximate_shadow_catcher_background =
      kfilm.use_approximate_shadow_catcher && !kbackground.transparent;

  if (pass_access_info.mode == PassMode::DENOISED &&
      (effective_denoised_buffer_params_.width != effective_buffer_params_.width ||
       effective_denoised_buffer_params_.height != effective_buffer_params_.height))
  {
    /* Avoid using sample count to filter pass after upscaling, since it is stored at a different
     * resolution. The denoiser should have applied scaling again in this case. */
    pass_access_info.use_sample_count = false;
    pass_access_info.use_approximate_shadow_catcher_background = false;
  }

  pass_access_info.show_active_pixels = film_->get_show_active_pixels();

  return pass_access_info;
}

PassAccessor::Destination PathTraceWork::get_display_destination_template(
    const PathTraceDisplay *display, const PassMode mode) const
{
  PassAccessor::Destination destination(film_->get_display_pass(), mode);

  const BufferParams &effective_big_tile_params = (mode == PassMode::DENOISED) ?
                                                      effective_denoised_big_tile_params_ :
                                                      effective_big_tile_params_;
  const BufferParams &effective_buffer_params = (mode == PassMode::DENOISED) ?
                                                    effective_denoised_buffer_params_ :
                                                    effective_buffer_params_;

  const int2 display_texture_size = display->get_texture_size();
  const int texture_x = effective_buffer_params.full_x - effective_big_tile_params.full_x +
                        effective_buffer_params.window_x - effective_big_tile_params.window_x;
  const int texture_y = effective_buffer_params.full_y - effective_big_tile_params.full_y +
                        effective_buffer_params.window_y - effective_big_tile_params.window_y;

  destination.offset = texture_y * display_texture_size.x + texture_x;
  destination.stride = display_texture_size.x;

  return destination;
}

CCL_NAMESPACE_END
