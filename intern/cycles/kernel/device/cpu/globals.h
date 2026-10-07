/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Constant Globals */

#pragma once

#include "kernel/types.h"
#include "kernel/util/profiler.h"

#ifdef __OSL__
#  include "kernel/osl/globals.h"
#endif

#include "util/guiding.h"      // IWYU pragma: keep
#include "util/types_image.h"  // IWYU pragma: keep
#include "util/unique_ptr.h"

CCL_NAMESPACE_BEGIN

struct OSLGlobals;
struct IntegratorQueueCounter;
struct KernelPhoton;
struct KernelBDPTVertex;
struct KernelVCMVertex;
struct KernelPolarizationState;
struct CoherentPathHistory;

/* Working memory of the light-cache transport passes (photon mapping and bidirectional path
 * tracing), owned by PathTraceWorkCPU for the duration of a render batch. The members mirror the
 * corresponding fields of IntegratorStateGPU, so that kernel code accesses both through
 * `kernel_integrator_state`. Pointers are null and capacities zero while a feature is disabled.
 * Guiding members exist only for the Metal guiding field, which the CPU does not use: it guides
 * with OpenPGL instead. */
struct KernelTransportStateCPU {
  IntegratorQueueCounter *queue_counter = nullptr;

  KernelPhoton *photons = nullptr;
  uint *photon_hash = nullptr;
  uint *photon_stored = nullptr;
  /* CPU only: one flag per emitted photon path. A photon path stores at most one photon, so the
   * record of path `i` lives in slot `i` and the host links hash chains in path order. This keeps
   * the photon map, and therefore the render, independent of thread scheduling. */
  uint8_t *photon_valid = nullptr;
  KernelVCMVertex *vcm_vertices = nullptr;

  KernelBDPTVertex *bdpt_vertices = nullptr;
  CoherentPathHistory *bdpt_coherent_history = nullptr;
  KernelPolarizationState *bdpt_polarization = nullptr;
  uint *bdpt_vertex_indices = nullptr;
  uint *bdpt_vertex_count = nullptr;

  uint guiding_capacity = 0;
  uint guiding_training = 0;

  uint photon_hash_size = 0;
  uint photon_capacity = 0;
  uint photon_iteration = 0;
  float photon_radius = 0.0f;
  float photon_volume_radius = 0.0f;

  uint bdpt_vertex_capacity = 0;
  uint bdpt_light_path_count = 0;
  uint bdpt_cache_capacity = 0;
  uint bdpt_cache_count = 0;
  uint bdpt_cache_start_sample = 0;
  float bdpt_light_path_sample_ratio = 0.0f;
  int bdpt_buffer_full_x = 0;
  int bdpt_buffer_full_y = 0;
  int bdpt_buffer_width = 0;
  int bdpt_buffer_height = 0;
  int bdpt_buffer_offset = 0;
  int bdpt_buffer_stride = 0;

  uint vcm_path_slots = 0;
  uint vcm_light_path_count = 0;
  uint vcm_cache_slots = 0;
  float vcm_eta_scale = 0.0f;
  float vcm_radius_base = 0.0f;
  float vcm_radius_slope = 0.0f;
  uint vcm_groups = 1;
};

/* On the CPU, we pass along the struct KernelGlobals to nearly everywhere in
 * the kernel, to access constant data. These are all stored as flat arrays.
 * these are really just standard arrays. We can't use actually globals because
 * multiple renders may be running inside the same process. */

/* Array for kernel data, with size to be able to assert on invalid data access. */
template<typename T> struct kernel_array {
  const ccl_always_inline T &fetch(const int index) const
  {
    kernel_assert(index >= 0 && index < width);
    return data[index];
  }

  ccl_always_inline void write(const int index, const T &value) const
  {
    data[index] = value;
  }

  T *data = nullptr;
  int width = 0;
};

/* Constant globals shared between all threads. */
struct KernelGlobalsCPU {
#define KERNEL_DATA_ARRAY(type, name) kernel_array<const type> name;
#define KERNEL_DATA_ARRAY_WRITABLE(type, name) kernel_array<type> name;
#include "kernel/data_arrays.h"

  KernelData data = {};

  KernelImageLoadRequestedCPU image_load_requested_cpu;

  ProfilingState profiler;
};

/* Per-thread global state.
 *
 * To avoid pointer indirection, the constant globals are copied to each thread.
 *
 * This may not be ideal for cache pressure. Alternative would be to pass an
 * additional thread index to every function, and potentially to make the shared
 * part an actual global variable. That would match the GPU more closely, but
 * also require mutex locks for multiple Cycles instances. */
struct ThreadKernelGlobalsCPU : public KernelGlobalsCPU {
  ThreadKernelGlobalsCPU(const KernelGlobalsCPU &kernel_globals,
                         OSLGlobals *osl_globals_memory,
                         Profiler &cpu_profiler,
                         const int thread_index);

  ThreadKernelGlobalsCPU(ThreadKernelGlobalsCPU &other) = delete;
  ThreadKernelGlobalsCPU(ThreadKernelGlobalsCPU &&other) noexcept = default;
  ThreadKernelGlobalsCPU &operator=(const ThreadKernelGlobalsCPU &other) = delete;
  ThreadKernelGlobalsCPU &operator=(ThreadKernelGlobalsCPU &&other) = delete;

  void start_profiling();
  void stop_profiling();

#ifdef __OSL__
  OSLThreadData osl;
#endif

  /* Set by the coherent connector when it must reject a render rather than
   * silently drop a supported path (unresolved caustic, stored-path overflow).
   * Per thread, so no atomics are needed; read after each sample batch. */
  mutable uint coherent_error = 0;

  /* Light-cache transport memory of the render work that currently uses these globals. Kernel
   * code only writes through the contained pointers. Never null. */
  const KernelTransportStateCPU *transport_state;

  /* Photon path being emitted by this thread, the slot of its photon record. */
  mutable uint photon_emit_index = 0;

  /* Matches the Metal `pixel_displacement_rays` function constant: only rays of the
   * intersection kernels intersect pixel displaced surfaces. Rays traced while shading
   * (ambient occlusion, bevel, light-cache transport connections) use the base triangles. */
  mutable bool pixel_displacement_intersect_rays = false;

#if defined(__PATH_GUIDING__)
  /* Pointers to shared global data structures. */
  openpgl::cpp::SampleStorage *opgl_sample_data_storage = nullptr;
  openpgl::cpp::Field *opgl_guiding_field = nullptr;

  /* Local data structures owned by the thread. */
  unique_ptr<openpgl::cpp::PathSegmentStorage> opgl_path_segment_storage;
  unique_ptr<openpgl::cpp::SurfaceSamplingDistribution> opgl_surface_sampling_distribution;
  unique_ptr<openpgl::cpp::VolumeSamplingDistribution> opgl_volume_sampling_distribution;
#endif

 protected:
  Profiler &cpu_profiler_;
};

using KernelGlobals = const ThreadKernelGlobalsCPU *;

/* Abstraction macros */
#define kernel_data_fetch(name, index) (kg->name.fetch(index))
#define kernel_data_write(name, index, value) (kg->name.write(index, value))
#define kernel_data_array(name) (kg->name.data)
#define kernel_data (kg->data)
#define kernel_integrator_state (*kg->transport_state)
#define pixel_displacement_rays (kg->pixel_displacement_intersect_rays)
#if defined(WITH_PATH_GUIDING)
#  define guiding_guiding_field kg->opgl_guiding_field
#  define guiding_ssd kg->opgl_surface_sampling_distribution
#  define guiding_vsd kg->opgl_volume_sampling_distribution
#endif

CCL_NAMESPACE_END
