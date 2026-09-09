/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/path_trace_work_gpu.h"

#include <cstdlib>
#include <fstream>

#include "device/device.h"

#include "integrator/pass_accessor_gpu.h"
#include "integrator/path_trace_display.h"

#include "scene/scene.h"
#include "session/buffers.h"

#include "util/log.h"
#include "util/string.h"
#include "util/time.h"

#include "kernel/device/gpu/block_sizes.h"
#include "kernel/sample/guiding_field.h"
#include "kernel/sample/guiding_mixture_statistics.h"
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

static size_t estimate_single_state_size(const uint64_t kernel_features,
                                         const DeviceType device_type)
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
#define KERNEL_STRUCT_CPU_GUIDING_FEATURE 0
#define KERNEL_STRUCT_GPU_GUIDING_FEATURE \
  (device_type == DEVICE_METAL ? KERNEL_FEATURE_PATH_GUIDING : 0)

#include "kernel/integrator/state_template.h"

#include "kernel/integrator/shadow_state_template.h"
#undef KERNEL_STRUCT_CPU_GUIDING_FEATURE
#undef KERNEL_STRUCT_GPU_GUIDING_FEATURE

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
      bdpt_vertices_(device, "bdpt_light_vertices"),
      bdpt_vertex_indices_(device, "bdpt_vertex_indices"),
      bdpt_vertex_count_(device, "bdpt_light_vertex_count", MEM_READ_WRITE),
      guiding_nodes_(device, "guiding_spatial_nodes"),
      guiding_accumulation_(device, "guiding_accumulation"),
      guiding_sampling_(device, "guiding_sampling"),
      guiding_counts_(device, "guiding_counts", MEM_READ_WRITE),
      guiding_history_(device, "guiding_history"),
      guiding_history_count_(device, "guiding_history_count", MEM_READ_WRITE),
      guiding_partition_(device, "guiding_partition", MEM_READ_WRITE),
      guiding_indices_(device, "guiding_indices"),
      guiding_fit_(device, "guiding_working_fit"),
      guiding_fit_counts_(device, "guiding_fit_counts", MEM_READ_WRITE),
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
    const size_t single_state_size = estimate_single_state_size(kernel_features,
                                                                device_->info.type);

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
  /* OpenPGL path-segment pointers and CPU sampling state have no GPU consumer. Keep their
   * pointer slots in the shared ABI, but do not allocate or copy unused arrays for every path. */
#define KERNEL_STRUCT_CPU_GUIDING_FEATURE 0
#define KERNEL_STRUCT_GPU_GUIDING_FEATURE \
  (device_->info.type == DEVICE_METAL ? KERNEL_FEATURE_PATH_GUIDING : 0)

  bool shadow = false;
#include "kernel/integrator/state_template.h"

  shadow = true;
#include "kernel/integrator/shadow_state_template.h"
#undef KERNEL_STRUCT_CPU_GUIDING_FEATURE
#undef KERNEL_STRUCT_GPU_GUIDING_FEATURE

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
  alloc_gpu_guiding();
}

void PathTraceWorkGPU::alloc_gpu_guiding()
{
  LOG_DEBUG << "Metal guiding allocation: device=" << device_->info.type
            << " enabled=" << device_scene_->data.integrator.use_guiding
            << " directional_sampling="
            << device_scene_->data.integrator.guiding_directional_sampling_type;
  if (device_->info.type != DEVICE_METAL || !device_scene_->data.integrator.use_guiding) {
    guiding_nodes_.free();
    guiding_accumulation_.free();
    guiding_sampling_.free();
    guiding_counts_.free();
    guiding_history_.free();
    guiding_history_count_.free();
    guiding_partition_.free();
    guiding_indices_.free();
    guiding_fit_.free();
    guiding_fit_counts_.free();
    integrator_state_gpu_.guiding_partition = nullptr;
    integrator_state_gpu_.guiding_indices = nullptr;
    integrator_state_gpu_.guiding_fit = nullptr;
    integrator_state_gpu_.guiding_fit_counts = nullptr;
    integrator_state_gpu_.guiding_history = nullptr;
    integrator_state_gpu_.guiding_history_count = nullptr;
    integrator_state_gpu_.guiding_history_capacity = 0;
    guiding_history_group_active_ = false;
    integrator_state_gpu_.guiding_nodes = nullptr;
    integrator_state_gpu_.guiding_accumulation = nullptr;
    integrator_state_gpu_.guiding_sampling = nullptr;
    integrator_state_gpu_.guiding_counts = nullptr;
    integrator_state_gpu_.guiding_capacity = 0;
    integrator_state_gpu_.guiding_training = 0;
    guiding_reset_pending_ = true;
    return;
  }
  const size_t bytes_per_node = sizeof(GuidingSpatialNode) + GUIDING_FIELD_TYPES *
                                                                 (GuidingField::accumulation_size +
                                                                  GuidingField::sampling_size +
                                                                  GuidingMixtureStatistics::working_size) *
                                                                 sizeof(float) +
                                GUIDING_FIELD_TYPES * 4 * sizeof(uint);
  const size_t budget = size_t(device_scene_->data.integrator.guiding_gpu_memory_mb) * 1024 * 1024;
  /* Root plus pairs of children. Reserve counts within the user-visible memory budget. */
  const size_t capacity = max(size_t(3),
                              (((budget - 3 * sizeof(uint)) / bytes_per_node) - 1) | size_t(1));
  if (integrator_state_gpu_.guiding_capacity != capacity) {
    guiding_nodes_.alloc_to_device(capacity);
    guiding_accumulation_.alloc_to_device(capacity * GUIDING_FIELD_TYPES *
                                          GuidingField::accumulation_size);
    guiding_sampling_.alloc_to_device(capacity * GUIDING_FIELD_TYPES *
                                      GuidingField::sampling_size);
    guiding_fit_.alloc_to_device(capacity * GUIDING_FIELD_TYPES *
                                 GuidingMixtureStatistics::working_size);
    guiding_fit_counts_.alloc(capacity * GUIDING_FIELD_TYPES);
    guiding_fit_counts_.zero_to_device();
    guiding_partition_.alloc(3 * capacity * GUIDING_FIELD_TYPES + 1);
    guiding_partition_.zero_to_device();
    guiding_counts_.alloc(2);
    guiding_counts_.zero_to_device();
    guiding_reset_pending_ = true;
  }
  integrator_state_gpu_.guiding_nodes = (GuidingSpatialNode *)guiding_nodes_.device_pointer;
  integrator_state_gpu_.guiding_accumulation = (float *)guiding_accumulation_.device_pointer;
  integrator_state_gpu_.guiding_sampling = (float *)guiding_sampling_.device_pointer;
  integrator_state_gpu_.guiding_counts = (uint *)guiding_counts_.device_pointer;
  integrator_state_gpu_.guiding_capacity = uint(capacity);
  integrator_state_gpu_.guiding_partition = (uint *)guiding_partition_.device_pointer;
  integrator_state_gpu_.guiding_fit = (float *)guiding_fit_.device_pointer;
  integrator_state_gpu_.guiding_fit_counts = (uint *)guiding_fit_counts_.device_pointer;
}

void PathTraceWorkGPU::prepare_gpu_guiding()
{
  if (integrator_state_gpu_.guiding_capacity == 0) {
    return;
  }
  if (guiding_reset_pending_) {
    queue_->zero_to_device(guiding_nodes_);
    queue_->zero_to_device(guiding_accumulation_);
    queue_->zero_to_device(guiding_sampling_);
    queue_->zero_to_device(guiding_fit_);
    queue_->zero_to_device(guiding_fit_counts_);
    guiding_counts_.data()[0] = 1;
    guiding_counts_.data()[1] = 1;
    queue_->copy_to_device(guiding_counts_);
    guiding_trained_samples_ = 0;
    guiding_reset_pending_ = false;
  }
  const int training_samples = device_scene_->data.integrator.guiding_training_samples;
  integrator_state_gpu_.guiding_training = training_samples == 0 ||
                                           guiding_trained_samples_ < training_samples;
  /* Complete batches have drained before this call. Histories are only needed while training;
   * allocate again here if a persistent session restarts after releasing its training buffer. */
  if (integrator_state_gpu_.guiding_training) {
    const size_t budget = size_t(device_scene_->data.integrator.guiding_gpu_history_memory_mb) *
                          1024 * 1024;
    const size_t capacity = (budget - sizeof(uint)) / (sizeof(GuidingHistoryRecord) + sizeof(uint));
    if (guiding_history_.memory_size() != capacity * sizeof(GuidingHistoryRecord)) {
      guiding_history_.alloc_to_device(capacity);
      guiding_indices_.alloc_to_device(capacity);
      guiding_history_count_.alloc(1);
      /* alloc() creates host storage only. Allocate the device counter before publishing
       * its address in integrator_state; enqueue_reset() clears it again for each batch. */
      guiding_history_count_.zero_to_device();
    }
    integrator_state_gpu_.guiding_history = (GuidingHistoryRecord *)
                                                guiding_history_.device_pointer;
    integrator_state_gpu_.guiding_history_count = (uint *)guiding_history_count_.device_pointer;
    integrator_state_gpu_.guiding_history_capacity = uint(capacity);
    integrator_state_gpu_.guiding_indices = (uint *)guiding_indices_.device_pointer;
  }
  else {
    guiding_indices_.free();
    integrator_state_gpu_.guiding_indices = nullptr;
    guiding_history_.free();
    guiding_history_count_.free();
    integrator_state_gpu_.guiding_history = nullptr;
    integrator_state_gpu_.guiding_history_count = nullptr;
    integrator_state_gpu_.guiding_history_capacity = 0;
  }
  guiding_history_group_active_ = false;
  device_->const_copy_to(
      "integrator_state", &integrator_state_gpu_, sizeof(integrator_state_gpu_));
}

int PathTraceWorkGPU::gpu_guiding_group_size() const
{
  if (!integrator_state_gpu_.guiding_training) {
    return max_num_paths_;
  }
  const auto &integrator = device_scene_->data.integrator;
  const uint64_t camera_records = uint64_t(max(integrator.max_bounce, 1)) + 1;
  /* Non-MIS training can record standalone NEE, and BDPT can additionally produce
   * an adjoint arrival and a connection shadow. Mixed transparent surfaces can
   * produce these observations without consuming an ordinary bounce. Terminal
   * visits are included; shadow-catcher splitting is reserved by the scheduler. */
  const uint64_t visits = uint64_t(max(integrator.max_bounce, 1)) +
                          uint64_t(max(integrator.transparent_max_bounce, 0)) + 2;
  const bool standalone_nee = integrator.use_guiding_direct_light &&
                              !integrator.use_guiding_mis_weights;
  const uint64_t observations_per_visit = (standalone_nee ? 1 : 0) +
                                           (integrator.use_bidirectional_path_tracing ? 2 : 0);
  const uint64_t records_per_path = camera_records + observations_per_visit * visits;
  return min(max_num_paths_,
             int(uint64_t(integrator_state_gpu_.guiding_history_capacity) / records_per_path));
}

void PathTraceWorkGPU::update_gpu_guiding(const int batch_samples)
{
  if (!integrator_state_gpu_.guiding_training) {
    return;
  }
  guiding_trained_samples_ += batch_samples;
  const int training_limit = device_scene_->data.integrator.guiding_training_samples;
  /* Grow training batches geometrically, then publish every 64 samples. Small BDPT cache
   * refreshes may occur within a training batch; they continue using the same immutable field. */
  const bool publish = (guiding_trained_samples_ <= 64 &&
                        is_power_of_two(guiding_trained_samples_)) ||
                       guiding_trained_samples_ % 64 == 0 ||
                       (training_limit > 0 && guiding_trained_samples_ >= training_limit);
  if (!publish) {
    return;
  }
  const int one = 1;
  const int capacity = int(integrator_state_gpu_.guiding_capacity);
  const int distributions = capacity * GUIDING_FIELD_TYPES;
  queue_->enqueue(DEVICE_KERNEL_GUIDING_BEGIN_UPDATE, one, DeviceKernelArguments(&one));
  queue_->enqueue(
      DEVICE_KERNEL_GUIDING_PUBLISH, distributions, DeviceKernelArguments(&distributions));
  queue_->enqueue(DEVICE_KERNEL_GUIDING_REFINE, capacity, DeviceKernelArguments(&capacity));
  queue_->zero_to_device(guiding_fit_);
  queue_->zero_to_device(guiding_fit_counts_);
  queue_->copy_from_device(guiding_partition_);
  queue_->synchronize();
  if (guiding_partition_.data()[guiding_partition_.size() - 1] != 0) {
    device_->set_error("Metal guiding mixture publication failed");
    return;
  }
  LOG_DEBUG << "Metal guiding publish: trained_samples=" << guiding_trained_samples_;
}

void PathTraceWorkGPU::alloc_bidirectional_path_tracing()
{
  if (!use_bidirectional_path_tracing(device_scene_)) {
    bdpt_vertices_.free();
    bdpt_vertex_indices_.free();
    bdpt_vertex_count_.free();
    integrator_state_gpu_.bdpt_vertices = nullptr;
    integrator_state_gpu_.bdpt_vertex_indices = nullptr;
    integrator_state_gpu_.bdpt_vertex_count = nullptr;
    integrator_state_gpu_.bdpt_vertex_capacity = 0;
    integrator_state_gpu_.bdpt_light_path_count = 0;
    integrator_state_gpu_.bdpt_light_path_sample_ratio = 0.0f;
    return;
  }

  /* Keep the light-subpath density constant as image resolution changes. The setting remains the
   * total budget at the scene's full render resolution, while previews and cropped buffers receive
   * the proportional share. */
  const uint64_t scaled_light_paths =
      uint64_t(device_scene_->data.integrator.bdpt_light_paths) *
      uint64_t(max(effective_buffer_params_.width, 1)) *
      uint64_t(max(effective_buffer_params_.height, 1));
  const uint64_t reference_pixels = uint64_t(
      max(device_scene_->data.integrator.bdpt_reference_pixels, 1));
  const uint64_t scaled_count = (scaled_light_paths + reference_pixels - 1u) / reference_pixels;
  const uint light_paths = uint(min(scaled_count, uint64_t(max_num_paths_)));
  /* Each emitted light path reservoir-selects one potential surface bounce. The connection
   * estimator carries the selection support explicitly, keeping memory linear in path count. */
  const uint capacity = light_paths;

  LOG_INFO << "BDPT light cache: " << capacity << " vertices, light tree "
           << (device_scene_->data.integrator.use_light_tree ? "enabled" : "disabled");
  bdpt_vertices_.alloc_to_device(capacity, false);
  bdpt_vertex_indices_.alloc_to_device(capacity, false);
  if (bdpt_vertex_count_.size() != 1) {
    bdpt_vertex_count_.free();
    bdpt_vertex_count_.alloc(1);
    bdpt_vertex_count_.zero_to_device();
  }

  integrator_state_gpu_.bdpt_vertices = (KernelBDPTVertex *)bdpt_vertices_.device_pointer;
  integrator_state_gpu_.bdpt_vertex_indices = (uint *)bdpt_vertex_indices_.device_pointer;
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
  /* Effective tile dimensions are assigned after alloc_work_memory(). Allocating
   * this cache earlier used the previous render's dimensions (or zero on the
   * first render), changing emitted light paths despite identical settings. */
  alloc_bidirectional_path_tracing();
  double next_cancel_poll = 0.0;
  const auto cancelled = [&]() {
    if (is_cancel_requested()) {
      return true;
    }
    const double now = time_dt();
    if (cancel_callback_ && now >= next_cancel_poll) {
      next_cancel_poll = now + 0.1;
      return cancel_callback_();
    }
    return false;
  };
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
  const int camera_samples = device_scene_->data.integrator.use_photon_mapping &&
                                     !adaptive_sampling ?
                                 device_scene_->data.integrator.photon_camera_samples :
                                 1;
  /* Camera oversampling is deliberately amortized over fewer photon maps. The expensive emitted
   * path budget stays approximately constant while the mapped volume term receives more complete
   * free-flight samples. */
  const bool use_light_cache = device_scene_->data.integrator.use_photon_mapping ||
                               use_bidirectional_path_tracing(device_scene_);
  const int update_samples = device_scene_->data.integrator.use_photon_mapping ?
                                 device_scene_->data.integrator.photon_map_update_samples :
                                 device_scene_->data.integrator.bdpt_update_samples;
  const int map_update_samples = use_light_cache ? update_samples * camera_samples : samples_num;

  /* A finite photon map is one Monte Carlo realization. Reusing it for an arbitrarily large
   * camera batch leaves its density-estimation noise frozen in the image, regardless of the
   * displayed sample count. Bound the reuse interval so offline renders and long-running viewport
   * renders average independent maps and advance the progressive radius on the requested render
   * sample index. */
  for (int samples_done = 0; samples_done < samples_num;) {
    prepare_gpu_guiding();
    int guiding_batch_samples = samples_num;
    if (integrator_state_gpu_.guiding_training) {
      const int next_update = guiding_trained_samples_ < 64 ?
                                  next_power_of_two(guiding_trained_samples_) :
                                  (guiding_trained_samples_ / 64 + 1) * 64;
      guiding_batch_samples = next_update - guiding_trained_samples_;
      const int training_limit = device_scene_->data.integrator.guiding_training_samples;
      if (training_limit > 0) {
        guiding_batch_samples = min(guiding_batch_samples,
                                    training_limit - guiding_trained_samples_);
      }
    }
    const int batch_samples = min(min(map_update_samples, guiding_batch_samples),
                                  samples_num - samples_done);
    const int batch_start_sample = start_sample + samples_done;
    /* A work tile must fit a fresh group, not only the global state array. Otherwise
     * get_work() can report no fitting tile and an entire training batch would be skipped. */
    const int group_size = gpu_guiding_group_size() / (has_shadow_catcher() ? 2 : 1);
    work_tile_scheduler_.set_max_num_path_states(min(max_num_paths_ / 8, max(group_size, 1)));
    work_tile_scheduler_.reset(effective_buffer_params_,
                               batch_start_sample * camera_samples,
                               batch_samples * camera_samples,
                               sample_offset * camera_samples,
                               device_scene_->data.integrator.scrambling_distance);

    enqueue_reset();
    enqueue_photon_mapping(batch_start_sample);
    enqueue_bidirectional_light_paths(batch_start_sample, batch_samples);

    bool batch_complete = false;
    /* TODO: set a hard limit in case of undetected kernel failures? */
    while (true) {
      /* Enqueue work from the scheduler, on start or when there are not enough
       * paths to keep the device occupied. */
      /* enqueue_work_tiles() may return early while a non-intersection kernel is queued. Keep the
       * completion flag deterministic in that case; an indeterminate/stale true value would skip
       * the queued work and prematurely finish a refreshed light-cache batch. */
      bool finished = false;
      if (enqueue_work_tiles(finished)) {
        if (!update_queue_counter_and_cache()) {
          batch_complete = false;
          break; /* Stop on error. */
        }
      }

      if (device_->have_error() || cancelled()) {
        batch_complete = false;
        break;
      }

      /* Stop if no more work remaining. */
      if (finished) {
        batch_complete = true;
        break;
      }

      /* Enqueue one of the path iteration kernels. */
      if (enqueue_path_iteration()) {
        if (!update_queue_counter_and_cache()) {
          batch_complete = false;
          break; /* Stop on error. */
        }
      }

      if (device_->have_error() || cancelled()) {
        batch_complete = false;
        break;
      }

      num_busy_accum += num_active_main_paths_paths();
      ++num_iterations;
    }

    if (!batch_complete) {
      break;
    }
    update_gpu_guiding(batch_samples);
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
  if (integrator_state_gpu_.guiding_capacity > 0 && LOG_IS_ON(LOG_LEVEL_DEBUG)) {
    queue_->copy_from_device(guiding_counts_);
    if (queue_->synchronize()) {
      LOG_DEBUG << "Metal guiding: trained_samples=" << guiding_trained_samples_
                << " spatial_nodes=" << guiding_counts_.data()[0]
                << " capacity=" << integrator_state_gpu_.guiding_capacity;
    }
  }
  /* Explicit diagnostic snapshots let sampling tests inspect the learned field, including
   * empty regions and inherited observations. Distribution data is only read back on request. */
  const char *guiding_dump = std::getenv("CYCLES_METAL_GUIDING_DUMP");
  if (integrator_state_gpu_.guiding_capacity > 0 && guiding_dump && guiding_dump[0]) {
    queue_->copy_from_device(guiding_counts_);
    if (queue_->synchronize()) {
      const uint count = guiding_counts_.data()[0];
      const string prefix = string(guiding_dump) + "_" + std::to_string(guiding_trained_samples_);
      vector<uint8_t> storage;
      const auto write_snapshot = [&](device_memory &memory, const char *suffix, size_t bytes) {
        const void *data = queue_->copy_from_device_synchronized(memory, storage);
        std::ofstream file(prefix + suffix, std::ios::binary);
        if (data && file) {
          file.write(static_cast<const char *>(data), bytes);
        }
        if (!data || !file) {
          LOG_ERROR << "Unable to write Metal guiding snapshot " << prefix << suffix;
        }
      };
      write_snapshot(guiding_nodes_, ".nodes", size_t(count) * sizeof(GuidingSpatialNode));
      write_snapshot(guiding_sampling_,
                     ".sampling",
                     size_t(count) * GUIDING_FIELD_TYPES * GuidingField::sampling_size *
                         sizeof(float));
      const KernelIntegrator &integrator = device_scene_->data.integrator;
      std::ofstream metadata(prefix + ".json");
      metadata << "{\"nodes\":" << count << ",\"trained_samples\":" << guiding_trained_samples_
               << ",\"bounds_min\":[" << integrator.guiding_bounds_min.x << ","
               << integrator.guiding_bounds_min.y << "," << integrator.guiding_bounds_min.z
               << "],\"bounds_max\":[" << integrator.guiding_bounds_max.x << ","
               << integrator.guiding_bounds_max.y << "," << integrator.guiding_bounds_max.z
               << "],\"fields\":" << GUIDING_FIELD_TYPES << ",\"bins\":" << GuidingField::bins
               << ",\"tree_size\":" << GuidingField::tree_size
               << ",\"sampling_size\":" << GuidingField::sampling_size << "}\n";
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
  guiding_history_group_active_ = false;
  if (integrator_state_gpu_.guiding_training) {
    queue_->zero_to_device(guiding_history_count_);
  }
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
  const bool has_tiled_images = device_scene_->image_texture_tile_access_state.size() != 0;
  const auto resolve_cache_misses = [&](const char *stage) {
    queue_->copy_from_device(integrator_queue_counter_);
    if (!queue_->synchronize()) {
      return false;
    }
    if (!integrator_queue_counter_.data()->cache_miss) {
      return false;
    }
    LOG_DEBUG << "BDPT image cache miss during " << stage << ", loading requested tiles";
    device_->image_load_requested_gpu(*queue_);
    integrator_queue_counter_.data()->cache_miss = 0;
    queue_->copy_to_device(integrator_queue_counter_);
    return true;
  };
  do {
    if (is_cancel_requested() || (cancel_callback_ && cancel_callback_())) {
      return;
    }
    /* Generation has no film/shadow side effects. Discard incomplete reservoirs and replay
     * the same light samples after loading requested tiles, before exposing the cache. */
    queue_->zero_to_device(bdpt_vertex_count_);
    queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_BDPT_LIGHT_GENERATE, num_light_paths, generate_args);
  } while (has_tiled_images && resolve_cache_misses("light generation"));
  if (device_->have_error()) {
    return;
  }
  const int one = 1;
  queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_BDPT_CACHE_ORDER, one,
                   DeviceKernelArguments(&num_light_paths));
  /* Optional diagnostic snapshot before sensor work. Omit the completion marker
   * and tail padding so record comparisons contain only generated vertex data. */
  if (const char *prefix = std::getenv("CYCLES_BDPT_CACHE_DUMP")) {
    queue_->copy_from_device(bdpt_vertex_count_);
    if (queue_->synchronize()) {
      const uint count = min(bdpt_vertex_count_.data()[0],
                             integrator_state_gpu_.bdpt_vertex_capacity);
      vector<uint8_t> staging;
      const auto *vertices = static_cast<const KernelBDPTVertex *>(
          queue_->copy_from_device_synchronized(bdpt_vertices_, staging));
      vector<uint8_t> index_staging;
      const auto *order = static_cast<const uint *>(
          queue_->copy_from_device_synchronized(bdpt_vertex_indices_, index_staging));
      const string path = string(prefix) + "_" + std::to_string(iteration) + ".vertices";
      std::ofstream file(path, std::ios::binary);
      constexpr size_t bytes = offsetof(KernelBDPTVertex, sensor_complete);
      if (vertices && order && file) {
        for (uint i = 0; i < count; ++i) {
          if (order[i] >= integrator_state_gpu_.bdpt_vertex_capacity) {
            file.setstate(std::ios::failbit);
            break;
          }
          file.write(reinterpret_cast<const char *>(vertices + order[i]), bytes);
        }
      }
      if (!vertices || !order || !file) {
        LOG_ERROR << "Unable to write BDPT cache snapshot " << path;
      }
      else {
        LOG_INFO << "BDPT cache snapshot: path=" << path << " vertices=" << count
                 << " record_bytes=" << bytes;
      }
    }
  }
  const DeviceKernelArguments sensor_args(
      &num_light_paths, &iteration, &batch_samples, &buffers_->buffer.device_pointer);
  do {
    if (is_cancel_requested() || (cancel_callback_ && cancel_callback_())) {
      return;
    }
    /* Successful vertices retain a completion marker. Only unfinished connections replay,
     * preserving already queued sensor shadows when a different vertex requests a tile. */
    queue_->enqueue(DEVICE_KERNEL_INTEGRATOR_BDPT_SENSOR_CONNECT, num_light_paths, sensor_args);
  } while (has_tiled_images && resolve_cache_misses("sensor connection"));
  if (device_->have_error()) {
    return;
  }

  /* Light tracing can enqueue at most one sensor shadow per light path. Synchronize this single
   * counter before camera work starts so later shadow compaction cannot overwrite those paths. */
  queue_->copy_from_device(integrator_next_shadow_path_index_);
  const bool diagnose_bdpt = getenv("CYCLES_BDPT_DIAGNOSTICS") != nullptr;
  if (diagnose_bdpt) {
    queue_->copy_from_device(bdpt_vertex_count_);
  }
  queue_->synchronize();
  if (diagnose_bdpt) {
    LOG_INFO << "BDPT diagnostics: sample=" << iteration << " emitted=" << num_light_paths
             << " camera_samples=" << batch_samples
             << " cached=" << bdpt_vertex_count_.data()[0]
             << " sensor_shadows=" << integrator_next_shadow_path_index_.data()[0];
  }
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
                                       &buffers_->buffer.device_pointer,
                                       &zero,
                                       &d_path_index,
                                       &num_path_indices);

      queue_->enqueue(kernel, work_size, args);
      break;
    }
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST: {
      /* Closest ray intersection kernels with integrator state and render buffer. */
      const DeviceKernelArguments args(
          &d_path_index, &buffers_->buffer.device_pointer, &work_size);

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
      const DeviceKernelArguments args(
          &d_path_index, &buffers_->buffer.device_pointer, &work_size);

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
  if (integrator_state_gpu_.guiding_training) {
    /* Immutable histories can be referenced by shadows after their main state terminates.
     * Reclaim only after ALL queues drain; do not replenish a partially completed group. */
    if (num_active_paths != 0 || kernel != DEVICE_KERNEL_NUM) {
      return false;
    }
    if (guiding_history_group_active_) {
      queue_->copy_from_device(guiding_history_count_);
      if (!queue_->synchronize()) {
        return false;
      }
      if (guiding_history_count_.data()[0] > integrator_state_gpu_.guiding_history_capacity) {
        device_->set_error("Metal guiding training history overflow: render rejected");
        return false;
      }
      const int records = int(guiding_history_count_.data()[0]);
      if (records > 0) {
        queue_->enqueue(
            DEVICE_KERNEL_GUIDING_FLUSH_HISTORY, records, DeviceKernelArguments(&records));
        queue_->zero_to_device(guiding_partition_);
        queue_->enqueue(DEVICE_KERNEL_GUIDING_PARTITION_COUNT, records, DeviceKernelArguments(&records));
        const int one = 1;
        queue_->enqueue(DEVICE_KERNEL_GUIDING_PARTITION_PREFIX, one, DeviceKernelArguments(&one));
        queue_->enqueue(DEVICE_KERNEL_GUIDING_PARTITION_SCATTER, records, DeviceKernelArguments(&records));
        const int fit_threads = int(integrator_state_gpu_.guiding_capacity) * GUIDING_FIELD_TYPES * 32;
        queue_->enqueue(DEVICE_KERNEL_GUIDING_FIT, fit_threads, DeviceKernelArguments(&fit_threads));
        queue_->copy_from_device(guiding_partition_);
        if (!queue_->synchronize()) {
          return false;
        }
        if (guiding_partition_.data()[guiding_partition_.size() - 1] != 0) {
          device_->set_error("Metal guiding observation grouping or fitting failed");
          return false;
        }
      }
      LOG_DEBUG << "Metal guiding history: records=" << guiding_history_count_.data()[0]
                << " capacity=" << integrator_state_gpu_.guiding_history_capacity;
      guiding_history_group_active_ = false;
    }
    queue_->zero_to_device(guiding_history_count_);
    /* The tile scheduler uses the same bound, including the split reservation below. */
    max_num_camera_paths = gpu_guiding_group_size();
    if (max_num_camera_paths < (has_shadow_catcher() ? 2 : 1)) {
      device_->set_error("Metal guiding training memory cannot fit a complete path group");
      return false;
    }
  }
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

    const int num_available_paths = max_num_camera_paths - num_active_paths;
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
      if (work_tile_scheduler_.has_work()) {
        device_->set_error("Pending render tile does not fit the GPU path group: render rejected");
        return false;
      }
      finished = true;
      return false;
    }
  }

  /* Initialize paths from work tiles. */
  if (work_tiles.empty()) {
    return false;
  }

  guiding_history_group_active_ = integrator_state_gpu_.guiding_training != 0;

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
  device_ptr d_render_buffer = buffers_->buffer.device_pointer;

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

bool PathTraceWorkGPU::zero_render_buffers()
{
  guiding_reset_pending_ = true;
  queue_->zero_to_device(buffers_->buffer);

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
