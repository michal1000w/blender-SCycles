/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/path_trace_work_gpu.h"

#include "device/device.h"

#include "integrator/pass_accessor_gpu.h"
#include "integrator/path_trace_display.h"

#include "scene/scene.h"
#include "session/buffers.h"

#include "util/log.h"
#include "util/string.h"

#include "kernel/device/gpu/block_sizes.h"
#include "kernel/types.h"

CCL_NAMESPACE_BEGIN

static bool use_bidirectional_path_tracing(const DeviceScene *device_scene)
{
  if (!device_scene->data.integrator.use_bidirectional_path_tracing) {
    return false;
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

static size_t estimate_single_state_size(const uint64_t kernel_features)
{
  size_t state_size = 0;

#define KERNEL_STRUCT_BEGIN(name) \
  for (int array_index = 0;; array_index++) {

#ifdef __INTEGRATOR_GPU_PACKED_STATE__
#  define KERNEL_STRUCT_MEMBER(parent_struct, type, name, feature) \
    state_size += (KernelFeatureRequest(feature).test(kernel_features)) ? sizeof(type) : 0;
#  define KERNEL_STRUCT_MEMBER_PACKED(parent_struct, type, name, feature)
#  define KERNEL_STRUCT_BEGIN_PACKED(parent_struct, feature) \
    KERNEL_STRUCT_BEGIN(parent_struct) \
    KERNEL_STRUCT_MEMBER(parent_struct, packed_##parent_struct, packed, feature)
#else
#  define KERNEL_STRUCT_MEMBER(parent_struct, type, name, feature) \
    state_size += (KernelFeatureRequest(feature).test(kernel_features)) ? sizeof(type) : 0;
#  define KERNEL_STRUCT_MEMBER_PACKED KERNEL_STRUCT_MEMBER
#  define KERNEL_STRUCT_BEGIN_PACKED(parent_struct, feature) KERNEL_STRUCT_BEGIN(parent_struct)
#endif

#define KERNEL_STRUCT_ARRAY_MEMBER(parent_struct, type, name, feature) \
  state_size += (KernelFeatureRequest(feature).test(kernel_features)) ? sizeof(type) : 0;
#define KERNEL_STRUCT_END(name) \
  (void)array_index; \
  break; \
  }
#define KERNEL_STRUCT_END_ARRAY(name, cpu_array_size, gpu_array_size) \
  if (array_index >= gpu_array_size - 1) { \
    break; \
  } \
  }
/* TODO(sergey): Look into better estimation for fields which depend on scene features. Maybe
 * maximum state calculation should happen as `alloc_work_memory()`, so that we can react to an
 * updated scene state here.
 * For until then use common value. Currently this size is only used for logging, but is weak to
 * rely on this. */
#define KERNEL_STRUCT_VOLUME_STACK_SIZE 4

#include "kernel/integrator/state_template.h"

#include "kernel/integrator/shadow_state_template.h"

#undef KERNEL_STRUCT_BEGIN
#undef KERNEL_STRUCT_BEGIN_PACKED
#undef KERNEL_STRUCT_MEMBER
#undef KERNEL_STRUCT_MEMBER_PACKED
#undef KERNEL_STRUCT_ARRAY_MEMBER
#undef KERNEL_STRUCT_END
#undef KERNEL_STRUCT_END_ARRAY
#undef KERNEL_STRUCT_VOLUME_STACK_SIZE

  return state_size;
}

PathTraceWorkGPU::PathTraceWorkGPU(Device *device,
                                   Film *film,
                                   DeviceScene *device_scene,
                                   const bool *cancel_requested_flag)
    : PathTraceWork(device, film, device_scene, cancel_requested_flag),
      queue_(device->gpu_queue_create()),
      integrator_state_soa_kernel_features_(0),
      integrator_queue_counter_(device, "integrator_queue_counter", MEM_READ_WRITE),
      integrator_shader_sort_counter_(device, "integrator_shader_sort_counter", MEM_READ_WRITE),
      integrator_shader_raytrace_sort_counter_(
          device, "integrator_shader_raytrace_sort_counter", MEM_READ_WRITE),
      integrator_shader_sort_prefix_sum_(
          device, "integrator_shader_sort_prefix_sum", MEM_READ_WRITE),
      integrator_shader_sort_partition_key_offsets_(
          device, "integrator_shader_sort_partition_key_offsets", MEM_READ_WRITE),
      integrator_next_main_path_index_(device, "integrator_next_main_path_index", MEM_READ_WRITE),
      integrator_next_shadow_path_index_(
          device, "integrator_next_shadow_path_index", MEM_READ_WRITE),
      queued_paths_(device, "queued_paths", MEM_READ_WRITE),
      num_queued_paths_(device, "num_queued_paths", MEM_READ_WRITE),
      work_tiles_(device, "work_tiles", MEM_READ_WRITE),
      photons_(device, "photon_map"),
      photon_hash_(device, "photon_hash"),
      photon_stored_(device, "photon_stored", MEM_READ_WRITE),
      restir_reservoirs_a_(device, "restir_di_reservoirs_a"),
      restir_reservoirs_b_(device, "restir_di_reservoirs_b"),
      restir_pt_initial_(device, "restir_pt_initial"),
      restir_pt_reservoirs_a_(device, "restir_pt_reservoirs_a"),
      restir_pt_reservoirs_b_(device, "restir_pt_reservoirs_b"),
      restir_pt_surfaces_a_(device, "restir_pt_surfaces_a"),
      restir_pt_surfaces_b_(device, "restir_pt_surfaces_b"),
      restir_pt_duplication_(device, "restir_pt_duplication"),
      restir_pt_scratch_buffer_(device, "restir_pt_scratch_buffer"),
      bdpt_vertices_(device, "bdpt_light_vertices"),
      bdpt_vertex_count_(device, "bdpt_light_vertex_count", MEM_READ_WRITE),
      display_rgba_half_(device, "display buffer half", MEM_READ_WRITE),
      max_num_paths_(0),
      min_num_active_main_paths_(0),
      max_active_main_path_index_(0)
{
  memset(&integrator_state_gpu_, 0, sizeof(integrator_state_gpu_));
}

void PathTraceWorkGPU::alloc_integrator_soa()
{
  /* IntegrateState allocated as structure of arrays. */

  /* Check if we already allocated memory for the required features.
   * Note that both disabling and enabling features may require memory
   * allocations, so we check for equality. */
  const int requested_volume_stack_size = device_scene_->data.volume_stack_size;
  const uint64_t kernel_features = device_scene_->data.kernel_features;
  if (integrator_state_soa_kernel_features_ == kernel_features &&
      integrator_state_soa_volume_stack_size_ >= requested_volume_stack_size)
  {
    return;
  }
  integrator_state_soa_kernel_features_ = kernel_features;
  integrator_state_soa_volume_stack_size_ = max(integrator_state_soa_volume_stack_size_,
                                                requested_volume_stack_size);

  /* Determine the number of path states. Deferring this for as long as possible allows the
   * back-end to make better decisions about memory availability. */
  if (max_num_paths_ == 0) {
    const size_t single_state_size = estimate_single_state_size(kernel_features);

    max_num_paths_ = queue_->num_concurrent_states(single_state_size);
    min_num_active_main_paths_ = queue_->num_concurrent_busy_states(single_state_size);

    /* Limit number of active paths to the half of the overall state. This is due to the logic in
     * the path compaction which relies on the fact that regeneration does not happen sooner than
     * half of the states are available again. */
    min_num_active_main_paths_ = min(min_num_active_main_paths_, max_num_paths_ / 2);
  }

  /* Allocate a device only memory buffer before for each struct member, and then
   * write the pointers into a struct that resides in constant memory.
   *
   * TODO: store float3 in separate XYZ arrays. */
#define KERNEL_STRUCT_BEGIN(name) \
  for (int array_index = 0;; array_index++) {
#define KERNEL_STRUCT_MEMBER(parent_struct, type, name, feature) \
  if ((KernelFeatureRequest(feature).test(kernel_features)) && \
      (integrator_state_gpu_.parent_struct.name == nullptr)) \
  { \
    string name_str = string_printf("%sintegrator_state_" #parent_struct "_" #name, \
                                    shadow ? "shadow_" : ""); \
    auto array = make_unique<device_only_memory<type>>(device_, name_str.c_str()); \
    array->alloc_to_device(max_num_paths_); \
    memcpy(&integrator_state_gpu_.parent_struct.name, \
           &array->device_pointer, \
           sizeof(array->device_pointer)); \
    integrator_state_soa_.emplace_back(std::move(array)); \
  }
#ifdef __INTEGRATOR_GPU_PACKED_STATE__
#  define KERNEL_STRUCT_MEMBER_PACKED(parent_struct, type, name, feature) \
    if ((KernelFeatureRequest(feature).test(kernel_features))) { \
      string name_str = string_printf("%sintegrator_state_" #parent_struct "_" #name, \
                                      shadow ? "shadow_" : ""); \
      LOG_TRACE << "Skipping " << name_str \
                << " -- data is packed inside integrator_state_" #parent_struct "_packed"; \
    }
#  define KERNEL_STRUCT_BEGIN_PACKED(parent_struct, feature) \
    KERNEL_STRUCT_BEGIN(parent_struct) \
    KERNEL_STRUCT_MEMBER(parent_struct, packed_##parent_struct, packed, feature)
#else
#  define KERNEL_STRUCT_MEMBER_PACKED KERNEL_STRUCT_MEMBER
#  define KERNEL_STRUCT_BEGIN_PACKED(parent_struct, feature) KERNEL_STRUCT_BEGIN(parent_struct)
#endif

#define KERNEL_STRUCT_ARRAY_MEMBER(parent_struct, type, name, feature) \
  if ((KernelFeatureRequest(feature).test(kernel_features)) && \
      (integrator_state_gpu_.parent_struct[array_index].name == nullptr)) \
  { \
    string name_str = string_printf( \
        "%sintegrator_state_" #name "_%d", shadow ? "shadow_" : "", array_index); \
    auto array = make_unique<device_only_memory<type>>(device_, name_str.c_str()); \
    array->alloc_to_device(max_num_paths_); \
    memcpy(&integrator_state_gpu_.parent_struct[array_index].name, \
           &array->device_pointer, \
           sizeof(array->device_pointer)); \
    integrator_state_soa_.emplace_back(std::move(array)); \
  }
#define KERNEL_STRUCT_END(name) \
  (void)array_index; \
  break; \
  }
#define KERNEL_STRUCT_END_ARRAY(name, cpu_array_size, gpu_array_size) \
  if (array_index >= gpu_array_size - 1) { \
    break; \
  } \
  }
#define KERNEL_STRUCT_VOLUME_STACK_SIZE (integrator_state_soa_volume_stack_size_)

  bool shadow = false;
#include "kernel/integrator/state_template.h"

  shadow = true;
#include "kernel/integrator/shadow_state_template.h"

#undef KERNEL_STRUCT_BEGIN
#undef KERNEL_STRUCT_BEGIN_PACKED
#undef KERNEL_STRUCT_MEMBER
#undef KERNEL_STRUCT_MEMBER_PACKED
#undef KERNEL_STRUCT_ARRAY_MEMBER
#undef KERNEL_STRUCT_END
#undef KERNEL_STRUCT_END_ARRAY
#undef KERNEL_STRUCT_VOLUME_STACK_SIZE

  if (LOG_IS_ON(LOG_LEVEL_TRACE)) {
    size_t total_soa_size = 0;
    for (auto &&soa_memory : integrator_state_soa_) {
      total_soa_size += soa_memory->memory_size();
    }

    LOG_TRACE << "GPU SoA state size: " << string_human_readable_size(total_soa_size);
  }
}

void PathTraceWorkGPU::alloc_integrator_queue()
{
  if (integrator_queue_counter_.size() == 0) {
    integrator_queue_counter_.alloc(1);
    integrator_queue_counter_.zero_to_device();
    integrator_queue_counter_.copy_from_device();
    integrator_state_gpu_.queue_counter = (IntegratorQueueCounter *)
                                              integrator_queue_counter_.device_pointer;
  }

  /* Allocate data for active path index arrays. */
  if (num_queued_paths_.size() == 0) {
    num_queued_paths_.alloc(1);
    num_queued_paths_.zero_to_device();
  }

  if (queued_paths_.size() == 0) {
    queued_paths_.alloc(max_num_paths_);
    /* TODO: this could be skip if we had a function to just allocate on device. */
    queued_paths_.zero_to_device();
  }
}

void PathTraceWorkGPU::alloc_integrator_sorting()
{
  num_sort_partitions_ = queue_->num_sort_partitions(max_num_paths_,
                                                     device_scene_->data.max_shaders);

  integrator_state_gpu_.sort_partition_divisor = (int)divide_up(max_num_paths_,
                                                                num_sort_partitions_);

  if (num_sort_partitions_ > 1 && queue_->supports_local_atomic_sort()) {
    /* Allocate array for partitioned shader sorting using local atomics. */
    const int num_offsets = (device_scene_->data.max_shaders + 1) * num_sort_partitions_;
    if (integrator_shader_sort_partition_key_offsets_.size() < num_offsets) {
      integrator_shader_sort_partition_key_offsets_.alloc(num_offsets);
      integrator_shader_sort_partition_key_offsets_.zero_to_device();
    }
    integrator_state_gpu_.sort_partition_key_offsets =
        (int *)integrator_shader_sort_partition_key_offsets_.device_pointer;
  }
  else {
    /* Allocate arrays for shader sorting. */
    const int sort_buckets = device_scene_->data.max_shaders * num_sort_partitions_;
    if (integrator_shader_sort_counter_.size() < sort_buckets) {
      integrator_shader_sort_counter_.alloc(sort_buckets);
      integrator_shader_sort_counter_.zero_to_device();
      integrator_state_gpu_.sort_key_counter[DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE] =
          (int *)integrator_shader_sort_counter_.device_pointer;

      integrator_shader_sort_prefix_sum_.alloc(sort_buckets);
      integrator_shader_sort_prefix_sum_.zero_to_device();
    }

    if (device_scene_->data.kernel_features & KERNEL_FEATURE_NODE_RAYTRACE) {
      if (integrator_shader_raytrace_sort_counter_.size() < sort_buckets) {
        integrator_shader_raytrace_sort_counter_.alloc(sort_buckets);
        integrator_shader_raytrace_sort_counter_.zero_to_device();
        integrator_state_gpu_.sort_key_counter[DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE] =
            (int *)integrator_shader_raytrace_sort_counter_.device_pointer;
      }
    }
  }
}

void PathTraceWorkGPU::alloc_integrator_path_split()
{
  if (integrator_next_shadow_path_index_.size() == 0) {
    integrator_next_shadow_path_index_.alloc(1);
    integrator_next_shadow_path_index_.zero_to_device();

    integrator_state_gpu_.next_shadow_path_index =
        (int *)integrator_next_shadow_path_index_.device_pointer;
  }

  if (integrator_next_main_path_index_.size() == 0) {
    integrator_next_main_path_index_.alloc(1);
    integrator_next_shadow_path_index_.data()[0] = 0;
    integrator_next_main_path_index_.zero_to_device();

    integrator_state_gpu_.next_main_path_index =
        (int *)integrator_next_main_path_index_.device_pointer;
  }
}

void PathTraceWorkGPU::alloc_work_memory()
{
  alloc_integrator_soa();
  alloc_integrator_queue();
  alloc_integrator_sorting();
  alloc_integrator_path_split();
  alloc_photon_mapping();
  alloc_restir();
  alloc_restir_pt();
  alloc_bidirectional_path_tracing();
}

void PathTraceWorkGPU::alloc_restir_pt()
{
  if (!device_scene_->data.integrator.use_restir_pt ||
      effective_buffer_params_.pass_stride <= 0)
  {
    restir_pt_initial_.free();
    restir_pt_reservoirs_a_.free();
    restir_pt_reservoirs_b_.free();
    restir_pt_surfaces_a_.free();
    restir_pt_surfaces_b_.free();
    restir_pt_duplication_.free();
    restir_pt_scratch_buffer_.free();
    integrator_state_gpu_.restir_pt_initial = nullptr;
    integrator_state_gpu_.restir_pt_previous = nullptr;
    integrator_state_gpu_.restir_pt_current = nullptr;
    integrator_state_gpu_.restir_pt_previous_surfaces = nullptr;
    integrator_state_gpu_.restir_pt_source_surfaces = nullptr;
    integrator_state_gpu_.restir_pt_current_surfaces = nullptr;
    integrator_state_gpu_.restir_pt_duplication = nullptr;
    integrator_state_gpu_.restir_pt_reservoir_capacity = 0;
    restir_pt_history_valid_ = false;
    restir_pt_external_history_available_ = false;
    return;
  }

  const size_t capacity = buffers_->buffer.size() / size_t(effective_buffer_params_.pass_stride);
  if (capacity == 0) {
    return;
  }

  const bool resized = restir_pt_initial_.data_size != capacity ||
                       restir_pt_reservoirs_a_.data_size != capacity ||
                       restir_pt_reservoirs_b_.data_size != capacity ||
                       restir_pt_surfaces_a_.data_size != capacity ||
                       restir_pt_surfaces_b_.data_size != capacity ||
                       restir_pt_duplication_.data_size != capacity ||
                       restir_pt_scratch_buffer_.data_size != buffers_->buffer.data_size;
  if (resized) {
    restir_pt_initial_.alloc_to_device(capacity, false);
    restir_pt_reservoirs_a_.alloc_to_device(capacity, false);
    restir_pt_reservoirs_b_.alloc_to_device(capacity, false);
    restir_pt_surfaces_a_.alloc_to_device(capacity, false);
    restir_pt_surfaces_b_.alloc_to_device(capacity, false);
    restir_pt_duplication_.alloc_to_device(capacity, false);
    restir_pt_scratch_buffer_.alloc_to_device(buffers_->buffer.data_size, false);
    queue_->zero_to_device(restir_pt_initial_);
    queue_->zero_to_device(restir_pt_reservoirs_a_);
    queue_->zero_to_device(restir_pt_reservoirs_b_);
    queue_->zero_to_device(restir_pt_surfaces_a_);
    queue_->zero_to_device(restir_pt_surfaces_b_);
    queue_->zero_to_device(restir_pt_duplication_);
    queue_->zero_to_device(restir_pt_scratch_buffer_);
    restir_pt_previous_is_a_ = true;
    restir_pt_current_is_a_ = false;
    restir_pt_surface_previous_is_a_ = true;
    restir_pt_surface_current_is_a_ = false;
    restir_pt_history_valid_ = false;
    restir_pt_external_history_available_ = false;
  }

  integrator_state_gpu_.restir_pt_reservoir_capacity = uint(capacity);
  integrator_state_gpu_.restir_pt_initial =
      (KernelReSTIRPTReservoir *)restir_pt_initial_.device_pointer;
  integrator_state_gpu_.restir_pt_previous =
      (KernelReSTIRPTReservoir *)(restir_pt_previous_is_a_ ?
                                      restir_pt_reservoirs_a_.device_pointer :
                                      restir_pt_reservoirs_b_.device_pointer);
  integrator_state_gpu_.restir_pt_current =
      (KernelReSTIRPTReservoir *)(restir_pt_previous_is_a_ ?
                                      restir_pt_reservoirs_b_.device_pointer :
                                      restir_pt_reservoirs_a_.device_pointer);
  integrator_state_gpu_.restir_pt_previous_surfaces =
      (KernelReSTIRPTSurface *)(restir_pt_surface_previous_is_a_ ?
                                    restir_pt_surfaces_a_.device_pointer :
                                    restir_pt_surfaces_b_.device_pointer);
  integrator_state_gpu_.restir_pt_current_surfaces =
      (KernelReSTIRPTSurface *)(restir_pt_surface_previous_is_a_ ?
                                    restir_pt_surfaces_b_.device_pointer :
                                    restir_pt_surfaces_a_.device_pointer);
  integrator_state_gpu_.restir_pt_source_surfaces =
      integrator_state_gpu_.restir_pt_previous_surfaces;
  integrator_state_gpu_.restir_pt_duplication =
      (float *)restir_pt_duplication_.device_pointer;
}

void PathTraceWorkGPU::alloc_restir()
{
  const bool use_history = device_scene_->data.integrator.restir_history_length > 0 ||
                           device_scene_->data.integrator.restir_spatial_neighbors > 0;
  if (!device_scene_->data.integrator.use_restir || !use_history ||
      effective_buffer_params_.pass_stride <= 0)
  {
    restir_reservoirs_a_.free();
    restir_reservoirs_b_.free();
    integrator_state_gpu_.restir_previous = nullptr;
    integrator_state_gpu_.restir_current = nullptr;
    integrator_state_gpu_.restir_reservoir_capacity = 0;
    return;
  }

  const size_t capacity = buffers_->buffer.size() / size_t(effective_buffer_params_.pass_stride);
  if (capacity == 0) {
    return;
  }

  const bool resized = restir_reservoirs_a_.data_size != capacity ||
                       restir_reservoirs_b_.data_size != capacity;
  if (resized) {
    restir_reservoirs_a_.alloc_to_device(capacity, false);
    restir_reservoirs_b_.alloc_to_device(capacity, false);
    queue_->zero_to_device(restir_reservoirs_a_);
    queue_->zero_to_device(restir_reservoirs_b_);
    restir_previous_is_a_ = true;
  }

  integrator_state_gpu_.restir_reservoir_capacity = uint(capacity);
  integrator_state_gpu_.restir_previous =
      (KernelReSTIRDIReservoir *)(restir_previous_is_a_ ? restir_reservoirs_a_.device_pointer :
                                                          restir_reservoirs_b_.device_pointer);
  integrator_state_gpu_.restir_current =
      (KernelReSTIRDIReservoir *)(restir_previous_is_a_ ? restir_reservoirs_b_.device_pointer :
                                                          restir_reservoirs_a_.device_pointer);
}

void PathTraceWorkGPU::prepare_restir_sample()
{
  if (!device_scene_->data.integrator.use_restir ||
      integrator_state_gpu_.restir_reservoir_capacity == 0)
  {
    return;
  }

  device_only_memory<KernelReSTIRDIReservoir> &current = restir_previous_is_a_ ?
                                                             restir_reservoirs_b_ :
                                                             restir_reservoirs_a_;
  queue_->zero_to_device(current);
  integrator_state_gpu_.restir_previous =
      (KernelReSTIRDIReservoir *)(restir_previous_is_a_ ? restir_reservoirs_a_.device_pointer :
                                                          restir_reservoirs_b_.device_pointer);
  integrator_state_gpu_.restir_current = (KernelReSTIRDIReservoir *)current.device_pointer;
  integrator_state_gpu_.restir_buffer_full_x = effective_buffer_params_.full_x;
  integrator_state_gpu_.restir_buffer_full_y = effective_buffer_params_.full_y;
  integrator_state_gpu_.restir_buffer_width = effective_buffer_params_.width;
  integrator_state_gpu_.restir_buffer_height = effective_buffer_params_.height;
  integrator_state_gpu_.restir_buffer_offset = effective_buffer_params_.offset;
  integrator_state_gpu_.restir_buffer_stride = effective_buffer_params_.stride;
  device_->const_copy_to(
      "integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
}

void PathTraceWorkGPU::finish_restir_sample(const bool completed)
{
  if (completed && device_scene_->data.integrator.use_restir &&
      integrator_state_gpu_.restir_reservoir_capacity != 0)
  {
    restir_previous_is_a_ = !restir_previous_is_a_;
  }
}

void PathTraceWorkGPU::prepare_restir_pt_sample()
{
  if (!device_scene_->data.integrator.use_restir_pt ||
      integrator_state_gpu_.restir_pt_reservoir_capacity == 0)
  {
    return;
  }

  queue_->zero_to_device(restir_pt_initial_);
  device_only_memory<KernelReSTIRPTSurface> &current_surfaces =
      restir_pt_surface_previous_is_a_ ? restir_pt_surfaces_b_ : restir_pt_surfaces_a_;
  restir_pt_surface_current_is_a_ = !restir_pt_surface_previous_is_a_;
  queue_->zero_to_device(current_surfaces);
  device_only_memory<KernelReSTIRPTReservoir> &current = restir_pt_previous_is_a_ ?
                                                             restir_pt_reservoirs_b_ :
                                                             restir_pt_reservoirs_a_;
  restir_pt_current_is_a_ = !restir_pt_previous_is_a_;
  queue_->zero_to_device(current);
  integrator_state_gpu_.restir_pt_initial =
      (KernelReSTIRPTReservoir *)restir_pt_initial_.device_pointer;
  integrator_state_gpu_.restir_pt_previous =
      (KernelReSTIRPTReservoir *)(restir_pt_previous_is_a_ ?
                                      restir_pt_reservoirs_a_.device_pointer :
                                      restir_pt_reservoirs_b_.device_pointer);
  integrator_state_gpu_.restir_pt_source = integrator_state_gpu_.restir_pt_previous;
  integrator_state_gpu_.restir_pt_current =
      (KernelReSTIRPTReservoir *)current.device_pointer;
  integrator_state_gpu_.restir_pt_previous_surfaces =
      (KernelReSTIRPTSurface *)(restir_pt_surface_previous_is_a_ ?
                                    restir_pt_surfaces_a_.device_pointer :
                                    restir_pt_surfaces_b_.device_pointer);
  integrator_state_gpu_.restir_pt_source_surfaces =
      integrator_state_gpu_.restir_pt_previous_surfaces;
  integrator_state_gpu_.restir_pt_current_surfaces =
      (KernelReSTIRPTSurface *)current_surfaces.device_pointer;
  integrator_state_gpu_.restir_pt_phase = 0u;
  integrator_state_gpu_.restir_buffer_full_x = effective_buffer_params_.full_x;
  integrator_state_gpu_.restir_buffer_full_y = effective_buffer_params_.full_y;
  integrator_state_gpu_.restir_buffer_width = effective_buffer_params_.width;
  integrator_state_gpu_.restir_buffer_height = effective_buffer_params_.height;
  integrator_state_gpu_.restir_buffer_offset = effective_buffer_params_.offset;
  integrator_state_gpu_.restir_buffer_stride = effective_buffer_params_.stride;
  device_->const_copy_to(
      "integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
}

void PathTraceWorkGPU::enqueue_restir_pt_finalize()
{
  if (!device_scene_->data.integrator.use_restir_pt ||
      integrator_state_gpu_.restir_pt_reservoir_capacity == 0)
  {
    return;
  }
  const int num_pixels = int(integrator_state_gpu_.restir_pt_reservoir_capacity);
  const DeviceKernelArguments args(&num_pixels, &buffers_->buffer.device_pointer);
  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_RESTIR_PT_FINALIZE, num_pixels, args);
}

void PathTraceWorkGPU::enqueue_restir_pt_begin_reuse()
{
  if (!device_scene_->data.integrator.use_restir_pt ||
      integrator_state_gpu_.restir_pt_reservoir_capacity == 0)
  {
    return;
  }
  const int num_pixels = int(integrator_state_gpu_.restir_pt_reservoir_capacity);
  const DeviceKernelArguments args(&num_pixels);
  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_RESTIR_PT_BEGIN_REUSE, num_pixels, args);
}

void PathTraceWorkGPU::enqueue_restir_pt_end_reuse()
{
  if (!device_scene_->data.integrator.use_restir_pt ||
      integrator_state_gpu_.restir_pt_reservoir_capacity == 0)
  {
    return;
  }
  const int num_pixels = int(integrator_state_gpu_.restir_pt_reservoir_capacity);
  const DeviceKernelArguments args(&num_pixels);
  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_RESTIR_PT_END_REUSE, num_pixels, args);
}

void PathTraceWorkGPU::enqueue_restir_pt_normalize()
{
  if (!device_scene_->data.integrator.use_restir_pt ||
      integrator_state_gpu_.restir_pt_reservoir_capacity == 0)
  {
    return;
  }
  const int num_pixels = int(integrator_state_gpu_.restir_pt_reservoir_capacity);
  const DeviceKernelArguments args(&num_pixels);
  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_RESTIR_PT_NORMALIZE, num_pixels, args);
}

void PathTraceWorkGPU::enqueue_restir_pt_duplication()
{
  if (!device_scene_->data.integrator.use_restir_pt ||
      !device_scene_->data.integrator.restir_pt_decorrelate ||
      integrator_state_gpu_.restir_pt_reservoir_capacity == 0)
  {
    return;
  }
  const int num_pixels = int(integrator_state_gpu_.restir_pt_reservoir_capacity);
  const DeviceKernelArguments args(&num_pixels);
  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_RESTIR_PT_DUPLICATION, num_pixels, args);
}

void PathTraceWorkGPU::finish_restir_pt_sample(const bool completed)
{
  if (completed && device_scene_->data.integrator.use_restir_pt &&
      integrator_state_gpu_.restir_pt_reservoir_capacity != 0)
  {
    restir_pt_previous_is_a_ = restir_pt_current_is_a_;
    restir_pt_surface_previous_is_a_ = restir_pt_surface_current_is_a_;
    restir_pt_history_valid_ = true;
  }
}

void PathTraceWorkGPU::alloc_bidirectional_path_tracing()
{
  if (!use_bidirectional_path_tracing(device_scene_)) {
    bdpt_vertices_.free();
    bdpt_vertex_count_.free();
    integrator_state_gpu_.bdpt_vertices = nullptr;
    integrator_state_gpu_.bdpt_vertex_count = nullptr;
    integrator_state_gpu_.bdpt_vertex_capacity = 0;
    integrator_state_gpu_.bdpt_light_path_count = 0;
    integrator_state_gpu_.bdpt_light_path_sample_ratio = 0.0f;
    return;
  }

  /* Keep the light-subpath density constant as image resolution changes. The setting remains the
   * total budget at the scene's full render resolution, while previews and cropped buffers receive
   * the proportional share. */
  const uint64_t scaled_light_paths = uint64_t(device_scene_->data.integrator.bdpt_light_paths) *
                                      uint64_t(max(effective_buffer_params_.width, 1)) *
                                      uint64_t(max(effective_buffer_params_.height, 1));
  const uint64_t reference_pixels = uint64_t(
      max(device_scene_->data.integrator.bdpt_reference_pixels, 1));
  const uint64_t scaled_count = (scaled_light_paths + reference_pixels - 1u) / reference_pixels;
  const uint light_paths = uint(min(scaled_count, uint64_t(max_num_paths_)));
  /* Each emitted light path reservoir-selects one potential surface bounce. The connection
   * estimator carries the selection support explicitly, keeping memory linear in path count. */
  const uint capacity = light_paths;

  bdpt_vertices_.alloc_to_device(capacity, false);
  if (bdpt_vertex_count_.size() != 1) {
    bdpt_vertex_count_.free();
    bdpt_vertex_count_.alloc(1);
    bdpt_vertex_count_.zero_to_device();
  }

  integrator_state_gpu_.bdpt_vertices = (KernelBDPTVertex *)bdpt_vertices_.device_pointer;
  integrator_state_gpu_.bdpt_vertex_count = (uint *)bdpt_vertex_count_.device_pointer;
  integrator_state_gpu_.bdpt_vertex_capacity = capacity;
  integrator_state_gpu_.bdpt_light_path_count = light_paths;
  integrator_state_gpu_.bdpt_light_path_sample_ratio = float(light_paths);
}

void PathTraceWorkGPU::alloc_photon_mapping()
{
  if (!device_scene_->data.integrator.use_photon_mapping) {
    photons_.free();
    photon_hash_.free();
    photon_stored_.free();
    integrator_state_gpu_.photons = nullptr;
    integrator_state_gpu_.photon_hash = nullptr;
    integrator_state_gpu_.photon_stored = nullptr;
    integrator_state_gpu_.photon_hash_size = 0;
    integrator_state_gpu_.photon_capacity = 0;
    return;
  }

  const uint capacity = uint(min(device_scene_->data.integrator.photon_count, max_num_paths_));
  const uint hash_size = max(2048u, next_power_of_two(2u * capacity));

  photons_.alloc_to_device(capacity, false);
  photon_hash_.alloc_to_device(hash_size, false);
  if (photon_stored_.size() == 0) {
    photon_stored_.alloc(1);
    photon_stored_.zero_to_device();
  }

  integrator_state_gpu_.photons = (KernelPhoton *)photons_.device_pointer;
  integrator_state_gpu_.photon_hash = (uint *)photon_hash_.device_pointer;
  integrator_state_gpu_.photon_stored = (uint *)photon_stored_.device_pointer;
  integrator_state_gpu_.photon_hash_size = hash_size;
  integrator_state_gpu_.photon_capacity = capacity;
  integrator_state_gpu_.photon_iteration = 0;
  integrator_state_gpu_.photon_radius = device_scene_->data.integrator.photon_radius;
  integrator_state_gpu_.photon_volume_radius =
      device_scene_->data.integrator.photon_radius *
      device_scene_->data.integrator.photon_volume_radius_scale;
}

void PathTraceWorkGPU::init_execution()
{
  queue_->init_execution();

  /* Copy to device side struct in constant memory. */
  device_->const_copy_to(
      "integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
}

bool PathTraceWorkGPU::update_queue_counter_and_cache()
{
  /* Copy stats from the device. */
  queue_->copy_from_device(integrator_queue_counter_);

  if (!queue_->synchronize()) {
    return false;
  }

  /* Update image cache if needed. */
  /* TODO: If the number of kernels with cache misses is small compared to the total
   * number of queued kernels, we could try asynchronously updating the image cache
   * while continuing to work on the majority of states? */
  IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();
  if (queue_counter->cache_miss) {
    LOG_DEBUG << "Image cache miss in GPU kernel, updating to load requested tiles";
    device_->image_load_requested_gpu(*queue_);
    queue_counter->cache_miss = 0;
    queue_->copy_to_device(integrator_queue_counter_);
  }

  return true;
}

void PathTraceWorkGPU::render_samples(RenderStatistics &statistics,
                                      const int start_sample,
                                      const int samples_num,
                                      const int sample_offset,
                                      const bool adaptive_sampling)
{
  /* Limit number of states for the tile and rely on a greedy scheduling of tiles. This allows to
   * add more work (because tiles are smaller, so there is higher chance that more paths will
   * become busy after adding new tiles). This is especially important for the shadow catcher which
   * schedules work in halves of available number of paths. */
  work_tile_scheduler_.set_max_num_path_states(max_num_paths_ / 8);
  work_tile_scheduler_.set_accelerated_rt(
      (device_->get_bvh_layout_mask(device_scene_->data.kernel_features) & BVH_LAYOUT_OPTIX) != 0);

  int num_iterations = 0;
  uint64_t num_busy_accum = 0;
  /* Adaptive sampling schedules convergence checks in the requested sample index domain. Keep
   * that contract intact until the render scheduler can account for photon camera oversampling. */
  const bool use_restir_pt = device_scene_->data.integrator.use_restir_pt;
  /* Consume preserved history at most once. Samples generated later in this call, or in a later
   * scheduler chunk of the same progressive render, are already represented in the film and must
   * not be replayed as if they were independent temporal observations. */
  const bool temporal_history_from_previous_render = restir_pt_external_history_available_;
  restir_pt_external_history_available_ = false;
  const bool use_restir_history = device_scene_->data.integrator.use_restir &&
                                  (device_scene_->data.integrator.restir_history_length > 0 ||
                                   device_scene_->data.integrator.restir_spatial_neighbors > 0);
  const int camera_samples = device_scene_->data.integrator.use_photon_mapping &&
                                     !device_scene_->data.integrator.use_restir &&
                                     !adaptive_sampling ?
                                 device_scene_->data.integrator.photon_camera_samples :
                                 1;
  /* Camera oversampling is deliberately amortized over fewer photon maps. The expensive emitted
   * path budget stays approximately constant while the mapped volume term receives more complete
   * free-flight samples. */
  const bool use_light_cache = device_scene_->data.integrator.use_photon_mapping ||
                               use_restir_pt ||
                               use_restir_history || use_bidirectional_path_tracing(device_scene_);
  /* PT's initial path-tree reservoir can stream an arbitrary sample batch: its vector numerator
   * is the exact sum of all accepted contributions. One-spp batches are only required when a
   * temporal/spatial pass will replay that reservoir. Avoiding needless reset/finalize cycles is
   * important for offline renders where reuse is disabled and primary ReSTIR DI provides the
   * many-light variance reduction. */
  const bool use_restir_pt_reuse = use_restir_pt &&
                                   (temporal_history_from_previous_render ||
                                    device_scene_->data.integrator.restir_pt_spatial_neighbors >
                                        0);
  const int update_samples = (use_restir_history || use_restir_pt_reuse) ?
                                 1 :
                             device_scene_->data.integrator.use_photon_mapping ?
                                 device_scene_->data.integrator.photon_map_update_samples :
                             use_bidirectional_path_tracing(device_scene_) ?
                                 device_scene_->data.integrator.bdpt_update_samples :
                                 samples_num;
  const int map_update_samples = use_light_cache ? update_samples * camera_samples : samples_num;

  /* A finite photon map is one Monte Carlo realization. Reusing it for an arbitrarily large
   * camera batch leaves its density-estimation noise frozen in the image, regardless of the
   * displayed sample count. Bound the reuse interval so offline renders and long-running viewport
   * renders average independent maps and advance the progressive radius on the requested render
   * sample index. */
  for (int samples_done = 0; samples_done < samples_num;) {
    const int batch_samples = min(map_update_samples, samples_num - samples_done);
    const int batch_start_sample = start_sample + samples_done;
    work_tile_scheduler_.reset(effective_buffer_params_,
                               batch_start_sample * camera_samples,
                               batch_samples * camera_samples,
                               sample_offset * camera_samples,
                               device_scene_->data.integrator.scrambling_distance);

    enqueue_reset();
    prepare_restir_sample();
    prepare_restir_pt_sample();
    enqueue_photon_mapping(batch_start_sample);
    enqueue_bidirectional_light_paths(batch_start_sample, batch_samples);

    auto drain_paths = [&]() {
      bool complete = false;
      /* TODO: set a hard limit in case of undetected kernel failures? */
      while (true) {
        /* Enqueue work from the scheduler, on start or when there are not enough
         * paths to keep the device occupied. */
        /* enqueue_work_tiles() may return early while a non-intersection kernel is queued. Keep
         * the completion flag deterministic in that case. */
        bool finished = false;
        if (enqueue_work_tiles(finished)) {
          if (!update_queue_counter_and_cache()) {
            complete = false;
            break;
          }
        }

        if (is_cancel_requested()) {
          complete = false;
          break;
        }
        if (finished) {
          complete = true;
          break;
        }

        if (enqueue_path_iteration()) {
          if (!update_queue_counter_and_cache()) {
            complete = false;
            break;
          }
        }
        if (is_cancel_requested()) {
          complete = false;
          break;
        }

        num_busy_accum += num_active_main_paths_paths();
        ++num_iterations;
      }
      return complete;
    };

    bool batch_complete = drain_paths();

    const bool temporal_reuse = use_restir_pt && temporal_history_from_previous_render &&
                                samples_done == 0 &&
                                device_scene_->data.integrator.restir_pt_temporal_history > 0;
    if (batch_complete && temporal_reuse) {
      integrator_state_gpu_.restir_pt_phase = 1u;
      integrator_state_gpu_.restir_pt_source = integrator_state_gpu_.restir_pt_previous;
      integrator_state_gpu_.restir_pt_source_surfaces =
          integrator_state_gpu_.restir_pt_previous_surfaces;
      device_->const_copy_to(
          "integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
      enqueue_restir_pt_begin_reuse();
      queue_->synchronize();
      queue_->zero_to_device(restir_pt_scratch_buffer_);

      work_tile_scheduler_.reset(effective_buffer_params_,
                                 batch_start_sample * camera_samples,
                                 batch_samples * camera_samples,
                                 sample_offset * camera_samples,
                                 device_scene_->data.integrator.scrambling_distance);
      enqueue_reset();
      batch_complete = drain_paths();
      if (batch_complete) {
        enqueue_restir_pt_end_reuse();
      }
    }

    const int spatial_reuse_passes = use_restir_pt ?
                                         device_scene_->data.integrator.restir_pt_spatial_neighbors :
                                         0;
    if (batch_complete && spatial_reuse_passes > 0) {
      if (!temporal_reuse) {
        /* Materialize the canonical initial reservoir without tracing a redundant identity shift. */
        integrator_state_gpu_.restir_pt_phase = 1u;
        device_->const_copy_to(
            "integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
        enqueue_restir_pt_begin_reuse();
        /* BEGIN_REUSE reads the phase to select initial versus source storage. Do not mutate the
         * shared integrator constant for phase 2 until this materialization has executed. */
        queue_->synchronize();
      }

      for (int spatial_iteration = 0;
           spatial_iteration < spatial_reuse_passes && batch_complete;
           ++spatial_iteration)
      {
        /* The completed layer is in raw GRIS form. Normalize it before it becomes an input to the
         * next paired spatial pass, then ping-pong into the other reservoir layer. */
        enqueue_restir_pt_normalize();
        /* NORMALIZE also branches on the current pass type. The following constant update changes
         * that phase, so make the boundary explicit instead of relying on constant-copy timing. */
        queue_->synchronize();
        integrator_state_gpu_.restir_pt_source = integrator_state_gpu_.restir_pt_current;
        device_only_memory<KernelReSTIRPTReservoir> &destination = restir_pt_current_is_a_ ?
                                                                        restir_pt_reservoirs_b_ :
                                                                        restir_pt_reservoirs_a_;
        queue_->zero_to_device(destination);
        restir_pt_current_is_a_ = !restir_pt_current_is_a_;
        integrator_state_gpu_.restir_pt_current =
            (KernelReSTIRPTReservoir *)destination.device_pointer;
        integrator_state_gpu_.restir_pt_source_surfaces =
            integrator_state_gpu_.restir_pt_current_surfaces;
        integrator_state_gpu_.restir_pt_phase = uint(spatial_iteration + 2);
        integrator_state_gpu_.restir_pt_spatial_iteration = uint(spatial_iteration);
        device_->const_copy_to(
            "integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
        enqueue_restir_pt_begin_reuse();
        /* Reuse the initial-reservoir allocation as a per-pass pairwise shift staging buffer. */
        queue_->zero_to_device(restir_pt_initial_);
        queue_->zero_to_device(restir_pt_scratch_buffer_);

        work_tile_scheduler_.reset(effective_buffer_params_,
                                   batch_start_sample * camera_samples,
                                   batch_samples * camera_samples,
                                   sample_offset * camera_samples,
                                   device_scene_->data.integrator.scrambling_distance);
        enqueue_reset();
        batch_complete = drain_paths();
        if (batch_complete) {
          enqueue_restir_pt_end_reuse();
        }
      }
    }

    if (!batch_complete) {
      finish_restir_sample(false);
      finish_restir_pt_sample(false);
      break;
    }

    enqueue_restir_pt_finalize();
    enqueue_restir_pt_duplication();
    if (use_restir_pt) {
      /* ReSTIR PT changes phase and reservoir pointers through the shared integrator constant.
       * The next one-sample batch rewrites that same constant in prepare_restir_pt_sample(). Keep
       * the finalized layer alive and its phase stable until FINALIZE and DUPLICATION have
       * consumed it; otherwise sufficiently fast batches can finalize against the next batch's
       * phase/pointers and silently lose energy. */
      queue_->synchronize();
    }
    finish_restir_sample(true);
    finish_restir_pt_sample(true);
    samples_done += batch_samples;
  }

  if (num_iterations) {
    statistics.occupancy = float(num_busy_accum) / num_iterations / max_num_paths_;
  }
  else {
    statistics.occupancy = 0.0f;
  }

  if (device_scene_->data.integrator.use_photon_mapping && LOG_IS_ON(LOG_LEVEL_DEBUG)) {
    queue_->copy_from_device(photon_stored_);
    if (queue_->synchronize()) {
      LOG_DEBUG << "Photon map stored "
                << min(photon_stored_.data()[0], integrator_state_gpu_.photon_capacity) << " / "
                << integrator_state_gpu_.photon_capacity << " photons at radius "
                << integrator_state_gpu_.photon_radius;
    }
  }
}

DeviceKernel PathTraceWorkGPU::get_most_queued_kernel() const
{
  const IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();

  int max_num_queued = 0;
  DeviceKernel kernel = DEVICE_KERNEL_NUM;

  for (int i = 0; i < DEVICE_GPU_KERNEL_INTEGRATOR_NUM; i++) {
    /* SHADOW_PATH_MNEE_PENDING is a sentinel marker on shadow slots holding an MNEE precompute
     * payload; there is no kernel to dispatch for it. The slot transitions to a real shadow
     * kernel (or terminates) when integrator_shade_surface runs on the main path. */
    if (i == DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING) {
      continue;
    }
    if (queue_counter->num_queued[i] > max_num_queued) {
      kernel = (DeviceKernel)i;
      max_num_queued = queue_counter->num_queued[i];
    }
  }

  return kernel;
}

void PathTraceWorkGPU::enqueue_reset()
{
  const DeviceKernelArguments args(&max_num_paths_);

  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_RESET, max_num_paths_, args);
  queue_->zero_to_device(integrator_queue_counter_);
  if (integrator_shader_sort_counter_.size() != 0) {
    queue_->zero_to_device(integrator_shader_sort_counter_);
  }
  if (device_scene_->data.kernel_features & KERNEL_FEATURE_NODE_RAYTRACE &&
      integrator_shader_raytrace_sort_counter_.size() != 0)
  {
    queue_->zero_to_device(integrator_shader_raytrace_sort_counter_);
  }

  /* Tiles enqueue need to know number of active paths, which is based on this counter. Zero the
   * counter on the host side because `zero_to_device()` is not doing it. */
  if (integrator_queue_counter_.host_pointer) {
    memset(integrator_queue_counter_.data(), 0, integrator_queue_counter_.memory_size());
  }

  /* All states have just been invalidated, so allocation of split main and shadow paths must
   * restart at the beginning of their arrays as well. This normally only mattered once per
   * render, but photon and BDPT cache refreshes deliberately start multiple independent batches.
   * Leaving either bump allocator at its previous high-water mark eventually makes valid shadow
   * branches silently run out of state slots. */
  integrator_next_main_path_index_.data()[0] = 0;
  integrator_next_shadow_path_index_.data()[0] = 0;
  queue_->copy_to_device(integrator_next_main_path_index_);
  queue_->copy_to_device(integrator_next_shadow_path_index_);
  max_active_main_path_index_ = 0;
}

void PathTraceWorkGPU::enqueue_photon_mapping(const int start_sample)
{
  if (!device_scene_->data.integrator.use_photon_mapping ||
      integrator_state_gpu_.photon_capacity == 0)
  {
    return;
  }

  queue_->zero_to_device(photon_hash_);
  queue_->zero_to_device(photon_stored_);

  const int iteration = max(start_sample, 0);
  integrator_state_gpu_.photon_iteration = iteration;
  integrator_state_gpu_.photon_radius = max(
      device_scene_->data.integrator.photon_radius *
          powf(float(iteration + 1), -device_scene_->data.integrator.photon_radius_decay),
      1.0e-6f);
  /* A three-dimensional point estimate needs N*r^3 to grow. Keep its shrink exponent safely
   * below 1/3 even when the surface estimator is configured to converge more aggressively. */
  const float volume_decay = min(device_scene_->data.integrator.photon_radius_decay, 0.3f);
  integrator_state_gpu_.photon_volume_radius = max(
      device_scene_->data.integrator.photon_radius *
          device_scene_->data.integrator.photon_volume_radius_scale *
          powf(float(iteration + 1), -volume_decay),
      1.0e-6f);
  device_->const_copy_to(
      "integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));

  const int num_photons = int(integrator_state_gpu_.photon_capacity);
  const DeviceKernelArguments args(&num_photons, &iteration);
  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_PHOTON_EMIT, num_photons, args);
}

void PathTraceWorkGPU::enqueue_bidirectional_light_paths(const int start_sample,
                                                         const int batch_samples)
{
  if (!use_bidirectional_path_tracing(device_scene_) ||
      integrator_state_gpu_.bdpt_light_path_count == 0)
  {
    return;
  }

  queue_->zero_to_device(bdpt_vertex_count_);
  const int iteration = max(start_sample, 0);
  /* Render schedulers divide the same sample range into different work-call sizes. Emit the
   * proportional share of the per-update budget so light paths per camera sample stay constant. */
  const uint update_samples = uint(max(device_scene_->data.integrator.bdpt_update_samples, 1));
  const uint batch_light_paths = max(
      1u,
      uint((uint64_t(integrator_state_gpu_.bdpt_vertex_capacity) * uint64_t(batch_samples) +
            update_samples - 1u) /
           update_samples));
  const int num_light_paths = int(
      min(integrator_state_gpu_.bdpt_vertex_capacity, batch_light_paths));
  integrator_state_gpu_.bdpt_light_path_count = uint(num_light_paths);
  integrator_state_gpu_.bdpt_light_path_sample_ratio = float(num_light_paths) /
                                                       float(max(batch_samples, 1));
  integrator_state_gpu_.bdpt_buffer_full_x = effective_buffer_params_.full_x;
  integrator_state_gpu_.bdpt_buffer_full_y = effective_buffer_params_.full_y;
  integrator_state_gpu_.bdpt_buffer_width = effective_buffer_params_.width;
  integrator_state_gpu_.bdpt_buffer_height = effective_buffer_params_.height;
  integrator_state_gpu_.bdpt_buffer_offset = effective_buffer_params_.offset;
  integrator_state_gpu_.bdpt_buffer_stride = effective_buffer_params_.stride;
  device_->const_copy_to(
      "integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
  const DeviceKernelArguments generate_args(&num_light_paths, &iteration, &batch_samples);
  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_BDPT_LIGHT_GENERATE, num_light_paths, generate_args);
  const DeviceKernelArguments sensor_args(
      &num_light_paths, &iteration, &batch_samples, &buffers_->buffer.device_pointer);
  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_BDPT_SENSOR_CONNECT, num_light_paths, sensor_args);

  /* Light tracing can enqueue at most one sensor shadow per light path. Synchronize this single
   * counter before camera work starts so later shadow compaction cannot overwrite those paths. */
  queue_->copy_from_device(integrator_next_shadow_path_index_);
  queue_->synchronize();
}

bool PathTraceWorkGPU::enqueue_path_iteration()
{
  /* Find kernel to execute, with max number of queued paths. */
  const IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();

  int num_active_paths = 0;
  for (int i = 0; i < DEVICE_GPU_KERNEL_INTEGRATOR_NUM; i++) {
    num_active_paths += queue_counter->num_queued[i];
  }

  if (num_active_paths == 0) {
    return false;
  }

  /* Find kernel to execute, with max number of queued paths. */
  const DeviceKernel kernel = get_most_queued_kernel();
  if (kernel == DEVICE_KERNEL_NUM) {
    return false;
  }

  /* For kernels that add shadow paths, check if there is enough space available.
   * If not, schedule shadow kernels first to clear out the shadow paths. */
  int num_paths_limit = INT_MAX;

  if (kernel_creates_shadow_paths(kernel)) {
    compact_shadow_paths();

    const int available_shadow_paths = max_num_paths_ -
                                       integrator_next_shadow_path_index_.data()[0];
    if (available_shadow_paths < queue_counter->num_queued[kernel]) {
      if (queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE]) {
        enqueue_path_iteration(DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE);
        return true;
      }
      if (queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW]) {
        enqueue_path_iteration(DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW);
        return true;
      }
      if (queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW]) {
        enqueue_path_iteration(DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW);
        return true;
      }
    }
    else if (kernel_creates_ao_paths(kernel) ||
             (use_bidirectional_path_tracing(device_scene_) &&
              (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ||
               kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE)))
    {
      /* Surface shading can branch to direct light, AO, and a BDPT vertex connection. */
      int shadow_paths_per_state = 1;
      if (kernel_creates_ao_paths(kernel)) {
        shadow_paths_per_state++;
      }
      if (use_bidirectional_path_tracing(device_scene_)) {
        shadow_paths_per_state++;
      }
      num_paths_limit = available_shadow_paths / shadow_paths_per_state;
    }
  }

  /* Schedule kernel with maximum number of queued items. */
  enqueue_path_iteration(kernel, num_paths_limit);

  /* Update next shadow path index for kernels that can add shadow paths. */
  if (kernel_creates_shadow_paths(kernel)) {
    queue_->copy_from_device(integrator_next_shadow_path_index_);
  }

  return true;
}

void PathTraceWorkGPU::enqueue_path_iteration(DeviceKernel kernel, const int num_paths_limit)
{
  device_ptr d_path_index = 0;
  device_ptr d_render_buffer =
      (integrator_state_gpu_.restir_pt_phase != 0u &&
       restir_pt_scratch_buffer_.device_pointer) ?
          restir_pt_scratch_buffer_.device_pointer :
          buffers_->buffer.device_pointer;

  /* Create array of path indices for which this kernel is queued to be executed. */
  int work_size = kernel_max_active_main_path_index(kernel);

  IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();
  const int num_queued = queue_counter->num_queued[kernel];

  if (kernel_uses_sorting(kernel)) {
    /* Compute array of active paths, sorted by shader. */
    work_size = num_queued;
    d_path_index = queued_paths_.device_pointer;

    compute_sorted_queued_paths(kernel, num_paths_limit);
  }
  else if (num_queued < work_size) {
    work_size = num_queued;
    d_path_index = queued_paths_.device_pointer;

    if (kernel_is_shadow_path(kernel)) {
      /* Compute array of active shadow paths for specific kernel. */
      compute_queued_paths(DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY, kernel);
    }
    else {
      /* Compute array of active paths for specific kernel. */
      compute_queued_paths(DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY, kernel);
    }
  }

  work_size = min(work_size, num_paths_limit);

  DCHECK_LE(work_size, max_num_paths_);

  switch (kernel) {
    case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA: {
      const int num_path_indices = work_size;
      device_ptr null_tiles = 0;
      int zero = 0;
      const DeviceKernelArguments args(&null_tiles,
                                       &zero,
                                       &d_render_buffer,
                                       &zero,
                                       &d_path_index,
                                       &num_path_indices);

      queue_->enqueue(kernel, work_size, args);
      break;
    }
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST: {
      /* Closest ray intersection kernels with integrator state and render buffer. */
      const DeviceKernelArguments args(&d_path_index, &d_render_buffer, &work_size);

      queue_->enqueue(kernel, work_size, args);
      break;
    }

    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_DEDICATED_LIGHT:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE: {
      /* Ray intersection kernels with integrator state. */
      const DeviceKernelArguments args(&d_path_index, &work_size);

      queue_->enqueue(kernel, work_size, args);
      break;
    }
    case DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_FORWARD:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME_RAY_MARCHING:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT: {
      /* Shading kernels with integrator state and render buffer. */
      const DeviceKernelArguments args(&d_path_index, &d_render_buffer, &work_size);

      queue_->enqueue(kernel, work_size, args);
      break;
    }
    default:
      LOG_FATAL << "Unhandled kernel " << device_kernel_as_string(kernel)
                << " used for path iteration, should never happen.";
      break;
  }
}

void PathTraceWorkGPU::compute_sorted_queued_paths(DeviceKernel queued_kernel,
                                                   const int num_paths_limit)
{
  int d_queued_kernel = queued_kernel;

  /* Launch kernel to fill the active paths arrays. */
  if (num_sort_partitions_ > 1 && queue_->supports_local_atomic_sort()) {
    const int work_size = kernel_max_active_main_path_index(queued_kernel);
    device_ptr d_queued_paths = queued_paths_.device_pointer;

    int partition_size = (int)integrator_state_gpu_.sort_partition_divisor;

    const DeviceKernelArguments args(
        &work_size, &partition_size, &num_paths_limit, &d_queued_paths, &d_queued_kernel);

    queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_SORT_BUCKET_PASS,
                    GPU_PARALLEL_SORT_BLOCK_SIZE * num_sort_partitions_,
                    args);
    queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_SORT_WRITE_PASS,
                    GPU_PARALLEL_SORT_BLOCK_SIZE * num_sort_partitions_,
                    args);
    return;
  }

  device_ptr d_counter = (device_ptr)integrator_state_gpu_.sort_key_counter[d_queued_kernel];
  device_ptr d_prefix_sum = integrator_shader_sort_prefix_sum_.device_pointer;
  assert(d_counter != 0 && d_prefix_sum != 0);

  /* Compute prefix sum of number of active paths with each shader. */
  {
    const int work_size = 1;
    int sort_buckets = device_scene_->data.max_shaders * num_sort_partitions_;

    const DeviceKernelArguments args(&d_counter, &d_prefix_sum, &sort_buckets);

    queue_->enqueue(DEVICE_KERNEL_PREFIX_SUM, work_size, args);
  }

  queue_->zero_to_device(num_queued_paths_);

  /* Launch kernel to fill the active paths arrays. */
  {
    /* TODO: this could be smaller for terminated paths based on amount of work we want
     * to schedule, and also based on num_paths_limit.
     *
     * Also, when the number paths is limited it may be better to prefer paths from the
     * end of the array since compaction would need to do less work. */
    const int work_size = kernel_max_active_main_path_index(queued_kernel);

    device_ptr d_queued_paths = queued_paths_.device_pointer;
    device_ptr d_num_queued_paths = num_queued_paths_.device_pointer;

    const DeviceKernelArguments args(&work_size,
                                     &num_paths_limit,
                                     &d_queued_paths,
                                     &d_num_queued_paths,
                                     &d_counter,
                                     &d_prefix_sum,
                                     &d_queued_kernel);

    queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY, work_size, args);
  }
}

void PathTraceWorkGPU::compute_queued_paths(DeviceKernel kernel, DeviceKernel queued_kernel)
{
  int d_queued_kernel = queued_kernel;

  /* Launch kernel to fill the active paths arrays. */
  const int work_size = kernel_max_active_main_path_index(queued_kernel);
  device_ptr d_queued_paths = queued_paths_.device_pointer;
  device_ptr d_num_queued_paths = num_queued_paths_.device_pointer;

  const DeviceKernelArguments args(
      &work_size, &d_queued_paths, &d_num_queued_paths, &d_queued_kernel);

  queue_->zero_to_device(num_queued_paths_);
  queue_->enqueue(kernel, work_size, args);
}

void PathTraceWorkGPU::compact_main_paths(const int num_active_paths)
{
  /* Early out if there is nothing that needs to be compacted. */
  if (num_active_paths == 0) {
    max_active_main_path_index_ = 0;
    return;
  }

  const int min_compact_paths = 32;
  if (max_active_main_path_index_ == num_active_paths ||
      max_active_main_path_index_ < min_compact_paths)
  {
    return;
  }

  /* Compact. */
  compact_paths(num_active_paths,
                max_active_main_path_index_,
                DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY,
                DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY,
                DEVICE_KERNEL_INTEGRATOR_COMPACT_STATES);

  /* Adjust max active path index now we know which part of the array is actually used. */
  max_active_main_path_index_ = num_active_paths;
}

void PathTraceWorkGPU::compact_shadow_paths()
{
  IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();
  const int num_active_paths =
      queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE] +
      queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW] +
      queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW] +
      queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING];

  /* Early out if there is nothing that needs to be compacted. */
  if (num_active_paths == 0) {
    if (integrator_next_shadow_path_index_.data()[0] != 0) {
      integrator_next_shadow_path_index_.data()[0] = 0;
      queue_->copy_to_device(integrator_next_shadow_path_index_);
    }
    return;
  }

  /* Compact if we can reduce the space used by half. Not always since
   * compaction has a cost. */
  const float max_overhead_factor = 2.0f;
  const int min_compact_paths = 32;
  const int num_total_paths = integrator_next_shadow_path_index_.data()[0];
  if (num_total_paths < num_active_paths * max_overhead_factor ||
      num_total_paths < min_compact_paths)
  {
    return;
  }

  /* Compact. */
  compact_paths(num_active_paths,
                num_total_paths,
                DEVICE_KERNEL_INTEGRATOR_TERMINATED_SHADOW_PATHS_ARRAY,
                DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_PATHS_ARRAY,
                DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES);

  /* Adjust max active path index now we know which part of the array is actually used. */
  integrator_next_shadow_path_index_.data()[0] = num_active_paths;
  queue_->copy_to_device(integrator_next_shadow_path_index_);
}

void PathTraceWorkGPU::compact_paths(const int num_active_paths,
                                     const int max_active_path_index,
                                     DeviceKernel terminated_paths_kernel,
                                     DeviceKernel compact_paths_kernel,
                                     DeviceKernel compact_kernel)
{
  /* Compact fragmented path states into the start of the array, moving any paths
   * with index higher than the number of active paths into the gaps. */
  device_ptr d_compact_paths = queued_paths_.device_pointer;
  device_ptr d_num_queued_paths = num_queued_paths_.device_pointer;

  /* Create array with terminated paths that we can write to. */
  {
    /* TODO: can the work size be reduced here? */
    int offset = num_active_paths;
    const int work_size = num_active_paths;

    const DeviceKernelArguments args(&work_size, &d_compact_paths, &d_num_queued_paths, &offset);

    queue_->zero_to_device(num_queued_paths_);
    queue_->enqueue(terminated_paths_kernel, work_size, args);
  }

  /* Create array of paths that we need to compact, where the path index is bigger
   * than the number of active paths. */
  {
    const int work_size = max_active_path_index;

    const DeviceKernelArguments args(
        &work_size, &d_compact_paths, &d_num_queued_paths, &num_active_paths);

    queue_->zero_to_device(num_queued_paths_);
    queue_->enqueue(compact_paths_kernel, work_size, args);
  }

  queue_->copy_from_device(num_queued_paths_);
  queue_->synchronize();

  const int num_compact_paths = num_queued_paths_.data()[0];

  /* Move paths into gaps. */
  if (num_compact_paths > 0) {
    int work_size = num_compact_paths;
    int active_states_offset = 0;
    int terminated_states_offset = num_active_paths;

    const DeviceKernelArguments args(
        &d_compact_paths, &active_states_offset, &terminated_states_offset, &work_size);

    queue_->enqueue(compact_kernel, work_size, args);
  }
}

bool PathTraceWorkGPU::enqueue_work_tiles(bool &finished)
{
  /* If there are existing paths wait them to go to intersect closest kernel, which will align the
   * wavefront of the existing and newly added paths. */
  /* TODO: Check whether counting new intersection kernels here will have positive affect on the
   * performance. */
  const DeviceKernel kernel = get_most_queued_kernel();
  if (kernel != DEVICE_KERNEL_NUM && kernel != DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST) {
    return false;
  }

  const int num_active_paths = num_active_main_paths_paths();

  /* Don't schedule more work if canceling. */
  if (is_cancel_requested()) {
    if (num_active_paths == 0) {
      finished = true;
    }
    return false;
  }

  finished = false;

  vector<KernelWorkTile> work_tiles;

  int max_num_camera_paths = max_num_paths_;
  int num_predicted_splits = 0;

  if (has_shadow_catcher()) {
    /* When there are shadow catchers in the scene bounce from them will split the state. So we
     * make sure there is enough space in the path states array to fit split states.
     *
     * Basically, when adding N new paths we ensure that there is 2*N available path states, so
     * that all the new paths can be split.
     *
     * Note that it is possible that some of the current states can still split, so need to make
     * sure there is enough space for them as well. */

    /* Number of currently in-flight states which can still split. */
    const int num_scheduled_possible_split = shadow_catcher_count_possible_splits();

    const int num_available_paths = max_num_paths_ - num_active_paths;
    const int num_new_paths = num_available_paths / 2;
    max_num_camera_paths = max(num_active_paths,
                               num_active_paths + num_new_paths - num_scheduled_possible_split);
    num_predicted_splits += num_scheduled_possible_split + num_new_paths;
  }

  /* Schedule when we're out of paths or there are too few paths to keep the
   * device occupied. */
  int num_paths = num_active_paths;
  if (num_paths == 0 || num_paths < min_num_active_main_paths_) {
    /* Get work tiles until the maximum number of path is reached. */
    while (num_paths < max_num_camera_paths) {
      KernelWorkTile work_tile;
      if (work_tile_scheduler_.get_work(&work_tile, max_num_camera_paths - num_paths)) {
        work_tiles.push_back(work_tile);
        num_paths += work_tile.w * work_tile.h * work_tile.num_samples;
      }
      else {
        break;
      }
    }

    /* If we couldn't get any more tiles, we're done. */
    if (work_tiles.empty() && num_paths == 0) {
      finished = true;
      return false;
    }
  }

  /* Initialize paths from work tiles. */
  if (work_tiles.empty()) {
    return false;
  }

  /* Compact state array when number of paths becomes small relative to the
   * known maximum path index, which makes computing active index arrays slow. */
  compact_main_paths(num_active_paths);

  if (has_shadow_catcher()) {
    integrator_next_main_path_index_.data()[0] = num_paths;
    queue_->copy_to_device(integrator_next_main_path_index_);
  }

  enqueue_work_tiles((device_scene_->data.bake.use) ? DEVICE_KERNEL_INTEGRATOR_INIT_FROM_BAKE :
                                                      DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA,
                     work_tiles.data(),
                     work_tiles.size(),
                     num_active_paths,
                     num_predicted_splits);

  return true;
}

void PathTraceWorkGPU::enqueue_work_tiles(DeviceKernel kernel,
                                          const KernelWorkTile work_tiles[],
                                          const int num_work_tiles,
                                          const int num_active_paths,
                                          const int num_predicted_splits)
{
  /* Copy work tiles to device. */
  if (work_tiles_.size() < num_work_tiles) {
    work_tiles_.alloc(num_work_tiles);
  }

  int path_index_offset = num_active_paths;
  int max_tile_work_size = 0;
  for (int i = 0; i < num_work_tiles; i++) {
    KernelWorkTile &work_tile = work_tiles_.data()[i];
    work_tile = work_tiles[i];

    const int tile_work_size = work_tile.w * work_tile.h * work_tile.num_samples;

    work_tile.path_index_offset = path_index_offset;
    work_tile.work_size = tile_work_size;

    path_index_offset += tile_work_size;

    max_tile_work_size = max(max_tile_work_size, tile_work_size);
  }

  queue_->copy_to_device(work_tiles_);

  const device_ptr d_work_tiles = work_tiles_.device_pointer;
  device_ptr d_render_buffer =
      (integrator_state_gpu_.restir_pt_phase != 0u &&
       restir_pt_scratch_buffer_.device_pointer) ?
          restir_pt_scratch_buffer_.device_pointer :
          buffers_->buffer.device_pointer;

  /* Launch kernel. */
  device_ptr null_ptr = 0;
  int zero = 0;
  const DeviceKernelArguments args(
      &d_work_tiles, &num_work_tiles, &d_render_buffer, &max_tile_work_size, &null_ptr, &zero);

  queue_->enqueue(kernel, max_tile_work_size * num_work_tiles, args);

  max_active_main_path_index_ = path_index_offset + num_predicted_splits;
}

int PathTraceWorkGPU::num_active_main_paths_paths()
{
  IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();

  int num_paths = 0;
  for (int i = 0; i < DEVICE_GPU_KERNEL_INTEGRATOR_NUM; i++) {
    DCHECK_GE(queue_counter->num_queued[i], 0)
        << "Invalid number of queued states for kernel "
        << device_kernel_as_string(static_cast<DeviceKernel>(i));

    if (!kernel_is_shadow_path((DeviceKernel)i)) {
      num_paths += queue_counter->num_queued[i];
    }
  }

  return num_paths;
}

bool PathTraceWorkGPU::should_use_graphics_interop(PathTraceDisplay *display)
{
  /* There are few aspects with the graphics interop when using multiple devices caused by the fact
   * that the PathTraceDisplay has a single texture:
   *
   *   CUDA will return `CUDA_ERROR_NOT_SUPPORTED` from `cuGraphicsGLRegisterBuffer()` when
   *   attempting to register OpenGL PBO which has been mapped. Which makes sense, because
   *   otherwise one would run into a conflict of where the source of truth is. */
  if (has_multiple_works()) {
    return false;
  }

  if (!interop_use_checked_) {
    Device *device = queue_->device;
    interop_use_ = device->should_use_graphics_interop(display->graphics_interop_get_device(),
                                                       true);

    if (interop_use_) {
      LOG_INFO << "Using graphics interop GPU display update.";
    }
    else {
      LOG_INFO << "Using naive GPU display update.";
    }

    interop_use_checked_ = true;
  }

  return interop_use_;
}

void PathTraceWorkGPU::copy_to_display(PathTraceDisplay *display,
                                       PassMode pass_mode,
                                       const int num_samples)
{
  if (device_->have_error()) {
    /* Don't attempt to update GPU display if the device has errors: the error state will make
     * wrong decisions to happen about interop, causing more chained bugs. */
    return;
  }

  if (!buffers_->buffer.device_pointer) {
    LOG_WARNING << "Request for GPU display update without allocated render buffers.";
    return;
  }

  if (should_use_graphics_interop(display)) {
    if (copy_to_display_interop(display, pass_mode, num_samples)) {
      return;
    }

    /* If error happens when trying to use graphics interop fallback to the native implementation
     * and don't attempt to use interop for the further updates. */
    interop_use_ = false;
  }

  copy_to_display_naive(display, pass_mode, num_samples);
}

void PathTraceWorkGPU::copy_to_display_naive(PathTraceDisplay *display,
                                             PassMode pass_mode,
                                             const int num_samples)
{
  const BufferParams &effective_big_tile_params = (pass_mode == PassMode::DENOISED) ?
                                                      effective_denoised_big_tile_params_ :
                                                      effective_big_tile_params_;
  const BufferParams &effective_buffer_params = (pass_mode == PassMode::DENOISED) ?
                                                    effective_denoised_buffer_params_ :
                                                    effective_buffer_params_;

  const int full_x = effective_buffer_params.full_x;
  const int full_y = effective_buffer_params.full_y;
  const int width = effective_buffer_params.window_width;
  const int height = effective_buffer_params.window_height;
  const int final_width = buffers_->params.window_width;
  const int final_height = buffers_->params.window_height;

  const int texture_x = full_x - effective_big_tile_params.full_x +
                        effective_buffer_params.window_x - effective_big_tile_params.window_x;
  const int texture_y = full_y - effective_big_tile_params.full_y +
                        effective_buffer_params.window_y - effective_big_tile_params.window_y;

  /* Re-allocate display memory if needed, and make sure the device pointer is allocated.
   *
   * NOTE: allocation happens to the final resolution so that no re-allocation happens on every
   * change of the resolution divider. However, if the display becomes smaller, shrink the
   * allocated memory as well. */
  if (display_rgba_half_.data_width != final_width ||
      display_rgba_half_.data_height != final_height)
  {
    display_rgba_half_.alloc(final_width, final_height);
    /* TODO(sergey): There should be a way to make sure device-side memory is allocated without
     * transferring zeroes to the device. */
    queue_->zero_to_device(display_rgba_half_);
  }

  PassAccessor::Destination destination(film_->get_display_pass(), pass_mode);
  destination.d_pixels_half_rgba = display_rgba_half_.device_pointer;

  get_render_tile_film_pixels(destination, pass_mode, num_samples);

  queue_->copy_from_device(display_rgba_half_);
  queue_->synchronize();

  display->copy_pixels_to_texture(display_rgba_half_.data(), texture_x, texture_y, width, height);
}

bool PathTraceWorkGPU::copy_to_display_interop(PathTraceDisplay *display,
                                               PassMode pass_mode,
                                               const int num_samples)
{
  if (!device_graphics_interop_) {
    device_graphics_interop_ = queue_->graphics_interop_create();
  }

  GraphicsInteropBuffer &interop_buffer = display->graphics_interop_get_buffer();
  device_graphics_interop_->set_buffer(interop_buffer);

  const device_ptr d_rgba_half = device_graphics_interop_->map();
  if (!d_rgba_half) {
    return false;
  }

  PassAccessor::Destination destination = get_display_destination_template(display, pass_mode);
  destination.d_pixels_half_rgba = d_rgba_half;

  get_render_tile_film_pixels(destination, pass_mode, num_samples);

  device_graphics_interop_->unmap();

  return true;
}

void PathTraceWorkGPU::destroy_gpu_resources(PathTraceDisplay *display)
{
  if (!device_graphics_interop_) {
    return;
  }
  display->graphics_interop_activate();
  device_graphics_interop_ = nullptr;
  display->graphics_interop_deactivate();
}

void PathTraceWorkGPU::get_render_tile_film_pixels(const PassAccessor::Destination &destination,
                                                   PassMode pass_mode,
                                                   const int num_samples)
{
  const KernelFilm &kfilm = device_scene_->data.film;

  const PassAccessor::PassAccessInfo pass_access_info = get_display_pass_access_info(pass_mode);
  if (pass_access_info.type == PASS_NONE) {
    return;
  }

  const BufferParams &effective_buffer_params = (pass_mode == PassMode::DENOISED) ?
                                                    effective_denoised_buffer_params_ :
                                                    effective_buffer_params_;

  const PassAccessorGPU pass_accessor(queue_.get(), pass_access_info, kfilm.exposure, num_samples);

  pass_accessor.get_render_tile_pixels(buffers_.get(), effective_buffer_params, destination);
}

int PathTraceWorkGPU::adaptive_sampling_converge_filter_count_active(const float threshold,
                                                                     bool reset)
{
  const int num_active_pixels = adaptive_sampling_convergence_check_count_active(threshold, reset);

  if (num_active_pixels) {
    enqueue_adaptive_sampling_filter_x();
    enqueue_adaptive_sampling_filter_y();
    queue_->synchronize();
  }

  return num_active_pixels;
}

int PathTraceWorkGPU::adaptive_sampling_convergence_check_count_active(const float threshold,
                                                                       bool reset)
{
  device_vector<uint> num_active_pixels(device_, "num_active_pixels", MEM_READ_WRITE);
  num_active_pixels.alloc(1);

  queue_->zero_to_device(num_active_pixels);

  const int work_size = effective_buffer_params_.width * effective_buffer_params_.height;
  if (!work_size) {
    return 0;
  }

  const int reset_int = reset; /* No bool kernel arguments. */

  const DeviceKernelArguments args(&buffers_->buffer.device_pointer,
                                   &effective_buffer_params_.full_x,
                                   &effective_buffer_params_.full_y,
                                   &effective_buffer_params_.width,
                                   &effective_buffer_params_.height,
                                   &threshold,
                                   &reset_int,
                                   &effective_buffer_params_.offset,
                                   &effective_buffer_params_.stride,
                                   &num_active_pixels.device_pointer);

  queue_->enqueue(DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK, work_size, args);

  queue_->copy_from_device(num_active_pixels);
  queue_->synchronize();

  return num_active_pixels.data()[0];
}

void PathTraceWorkGPU::enqueue_adaptive_sampling_filter_x()
{
  const int work_size = effective_buffer_params_.height;
  DCHECK_GT(work_size, 0);

  const DeviceKernelArguments args(&buffers_->buffer.device_pointer,
                                   &effective_buffer_params_.full_x,
                                   &effective_buffer_params_.full_y,
                                   &effective_buffer_params_.width,
                                   &effective_buffer_params_.height,
                                   &effective_buffer_params_.offset,
                                   &effective_buffer_params_.stride);

  queue_->enqueue(DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X, work_size, args);
}

void PathTraceWorkGPU::enqueue_adaptive_sampling_filter_y()
{
  const int work_size = effective_buffer_params_.width;
  DCHECK_GT(work_size, 0);

  const DeviceKernelArguments args(&buffers_->buffer.device_pointer,
                                   &effective_buffer_params_.full_x,
                                   &effective_buffer_params_.full_y,
                                   &effective_buffer_params_.width,
                                   &effective_buffer_params_.height,
                                   &effective_buffer_params_.offset,
                                   &effective_buffer_params_.stride);

  queue_->enqueue(DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y, work_size, args);
}

void PathTraceWorkGPU::cryptomatte_postproces()
{
  const int work_size = effective_buffer_params_.width * effective_buffer_params_.height;
  if (!work_size) {
    return;
  }

  const DeviceKernelArguments args(&buffers_->buffer.device_pointer,
                                   &work_size,
                                   &effective_buffer_params_.offset,
                                   &effective_buffer_params_.stride);

  queue_->enqueue(DEVICE_KERNEL_CRYPTOMATTE_POSTPROCESS, work_size, args);
}

void PathTraceWorkGPU::denoise_volume_guiding_buffers()
{
  if (effective_buffer_params_.width == 0 || effective_buffer_params_.height == 0) {
    return;
  }

  const DeviceKernelArguments args(&buffers_->buffer.device_pointer,
                                   &effective_buffer_params_.full_x,
                                   &effective_buffer_params_.full_y,
                                   &effective_buffer_params_.width,
                                   &effective_buffer_params_.height,
                                   &effective_buffer_params_.offset,
                                   &effective_buffer_params_.stride);

  {
    const int work_size = effective_buffer_params_.width * effective_buffer_params_.height;
    DCHECK_GT(work_size, 0);
    queue_->enqueue(DEVICE_KERNEL_VOLUME_GUIDING_FILTER_X, work_size, args);
  }

  {
    const int work_size = effective_buffer_params_.width;
    DCHECK_GT(work_size, 0);
    queue_->enqueue(DEVICE_KERNEL_VOLUME_GUIDING_FILTER_Y, work_size, args);
  }
}

bool PathTraceWorkGPU::copy_render_buffers_from_device()
{
  /* May not exist if cancelled before rendering started. */
  if (!buffers_->buffer.device_pointer) {
    return false;
  }

  queue_->copy_from_device(buffers_->buffer);

  /* Synchronize so that the CPU-side buffer is available at the exit of this function. */
  return queue_->synchronize();
}

bool PathTraceWorkGPU::copy_render_buffers_to_device()
{
  queue_->copy_to_device(buffers_->buffer);

  /* NOTE: The direct device access to the buffers only happens within this path trace work. The
   * rest of communication happens via API calls which involves `copy_render_buffers_from_device()`
   * which will perform synchronization as needed. */

  return true;
}

bool PathTraceWorkGPU::zero_render_buffers(const bool preserve_reuse_history)
{
  queue_->zero_to_device(buffers_->buffer);

  if (restir_reservoirs_a_.data_size != 0) {
    if (!preserve_reuse_history) {
      queue_->zero_to_device(restir_reservoirs_a_);
      queue_->zero_to_device(restir_reservoirs_b_);
      restir_previous_is_a_ = true;
    }
  }
  if (restir_pt_initial_.data_size != 0) {
    restir_pt_external_history_available_ = preserve_reuse_history && restir_pt_history_valid_;
    queue_->zero_to_device(restir_pt_initial_);
    queue_->zero_to_device(restir_pt_duplication_);
    queue_->zero_to_device(restir_pt_scratch_buffer_);
    if (!preserve_reuse_history) {
      queue_->zero_to_device(restir_pt_reservoirs_a_);
      queue_->zero_to_device(restir_pt_reservoirs_b_);
      queue_->zero_to_device(restir_pt_surfaces_a_);
      queue_->zero_to_device(restir_pt_surfaces_b_);
      restir_pt_previous_is_a_ = true;
      restir_pt_current_is_a_ = false;
      restir_pt_surface_previous_is_a_ = true;
      restir_pt_surface_current_is_a_ = false;
      restir_pt_history_valid_ = false;
      restir_pt_external_history_available_ = false;
    }
  }

  return true;
}

bool PathTraceWorkGPU::has_shadow_catcher() const
{
  return device_scene_->data.integrator.has_shadow_catcher;
}

int PathTraceWorkGPU::shadow_catcher_count_possible_splits()
{
  if (max_active_main_path_index_ == 0) {
    return 0;
  }

  if (!has_shadow_catcher()) {
    return 0;
  }

  queue_->zero_to_device(num_queued_paths_);

  const int work_size = max_active_main_path_index_;
  device_ptr d_num_queued_paths = num_queued_paths_.device_pointer;

  const DeviceKernelArguments args(&work_size, &d_num_queued_paths);

  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_SHADOW_CATCHER_COUNT_POSSIBLE_SPLITS, work_size, args);
  queue_->copy_from_device(num_queued_paths_);
  queue_->synchronize();

  return num_queued_paths_.data()[0];
}

bool PathTraceWorkGPU::kernel_uses_sorting(DeviceKernel kernel)
{
  return (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE);
}

bool PathTraceWorkGPU::kernel_creates_shadow_paths(DeviceKernel kernel)
{
  return (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE ||
          kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME_RAY_MARCHING ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT);
}

bool PathTraceWorkGPU::kernel_creates_ao_paths(DeviceKernel kernel)
{
  return (device_scene_->data.kernel_features & KERNEL_FEATURE_AO) &&
         (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE);
}

bool PathTraceWorkGPU::kernel_is_shadow_path(DeviceKernel kernel)
{
  return (kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW ||
          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE);
}

int PathTraceWorkGPU::kernel_max_active_main_path_index(DeviceKernel kernel)
{
  return (kernel_is_shadow_path(kernel)) ? integrator_next_shadow_path_index_.data()[0] :
                                           max_active_main_path_index_;
}

CCL_NAMESPACE_END
