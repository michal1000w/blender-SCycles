/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/device/cpu/globals.h"
#include "kernel/integrator/state.h"

#include "device/queue.h"

#include "integrator/path_trace_work.h"

#include "util/array.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

struct KernelWorkTile;
struct ThreadKernelGlobalsCPU;
struct IntegratorStateCPU;

class CPUKernels;

/* Implementation of PathTraceWork which schedules work on to queues pixel-by-pixel,
 * for CPU devices.
 *
 * NOTE: For the CPU rendering there are assumptions about TBB arena size and number of concurrent
 * queues on the render device which makes this work be only usable on CPU. */
class PathTraceWorkCPU : public PathTraceWork {
 public:
  PathTraceWorkCPU(Device *device,
                   Film *film,
                   DeviceScene *device_scene,
                   const bool *cancel_requested_flag);

  void init_execution() override;
  void deinit_execution() override;

  void render_samples(RenderStatistics &statistics,
                      const int start_sample,
                      const int samples_num,
                      const int sample_offset,
                      const bool adaptive_sampling) override;

  void copy_to_display(PathTraceDisplay *display,
                       PassMode pass_mode,
                       const int num_samples) override;
  void destroy_gpu_resources(PathTraceDisplay *display) override;

  bool copy_render_buffers_from_device() override;
  bool copy_render_buffers_to_device() override;
  bool zero_render_buffers() override;

  int adaptive_sampling_converge_filter_count_active(const float threshold, bool reset) override;
  void cryptomatte_postproces() override;
  void denoise_volume_guiding_buffers() override;

#if defined(WITH_PATH_GUIDING)
  /* Initializes the per-thread guiding kernel data. The function sets the pointers to the
   * global guiding field and the sample data storage as well es initializes the per-thread
   * guided sampling distributions (e.g., SurfaceSamplingDistribution and
   * VolumeSamplingDistribution). */
  void guiding_init_kernel_globals(void *guiding_field,
                                   void *sample_data_storage,
                                   const bool train) override;

  /* Pushes the collected training data/samples of a path to the global sample storage.
   * This function is called at the end of a random walk/path generation. */
  void guiding_push_sample_data_to_global_storage(ThreadKernelGlobalsCPU *kg,
                                                  IntegratorStateCPU *state,
                                                  ccl_global float *ccl_restrict render_buffer);
#endif

 protected:
  /* Render camera samples of all pixels of the effective buffer. */
  void render_camera_samples(const int start_sample,
                             const int samples_num,
                             const int sample_offset);

  /* Core path tracing routine. Renders given work time on the given queue. */
  void render_samples_full_pipeline(ThreadKernelGlobalsCPU *kernel_globals,
                                    const KernelWorkTile &work_tile,
                                    const int samples_num);

  /* Light-cache transport, scheduled like PathTraceWorkGPU: every batch of camera samples first
   * builds its photon map or light-vertex cache. */
  void render_samples_light_cache(const int start_sample,
                                  const int samples_num,
                                  const int sample_offset,
                                  const bool adaptive_sampling);
  void alloc_photon_mapping();
  void alloc_bidirectional_path_tracing();
  /* Photon map of the light subpath vertices for vertex merging, of `light_paths` subpaths. */
  void alloc_vertex_merging(const uint light_paths);
  /* Emit and link an independent photon map for the given render sample. */
  void update_photon_map(const int start_sample);
  /* Generate light subpaths for a batch of camera samples, and splat their camera connections. */
  void update_bidirectional_light_cache(const int start_sample, const int batch_samples);
  /* Run `func(kernel_globals, state, index)` for all indices, in parallel. */
  template<typename Func> void parallel_for_light_paths(const int num, const Func &func);

  /* CPU kernels. */
  const CPUKernels &kernels_;

  /* Pointer to device-owned kernel globals which is suitable for concurrent access from multiple
   * threads. This allows dynamic updates to image_info when textures are loaded on demand. */
  vector<ThreadKernelGlobalsCPU> *kernel_thread_globals_ = nullptr;

  /* Light-cache transport memory, referenced by the kernel thread globals while rendering. */
  KernelTransportStateCPU transport_state_;
  IntegratorQueueCounter transport_queue_counter_ = {};
  array<KernelPhoton> photons_;
  array<uint> photon_hash_;
  array<uint8_t> photon_valid_;
  /* Vertex merging keeps its light subpath vertices in the photon map, with these MIS terms. */
  array<KernelVCMVertex> vcm_vertices_;
  uint vcm_path_capacity_ = 0;
  uint photon_stored_ = 0;
  array<KernelBDPTVertex> bdpt_vertices_;
  array<CoherentPathHistory> bdpt_coherent_history_;
  array<KernelPolarizationState> bdpt_polarization_;
  array<uint> bdpt_vertex_indices_;
  array<uint> bdpt_vertex_count_;
  /* Serialize film accumulation of light-tracing splats, which may land on any pixel. */
  array<uint> film_locks_;
};

CCL_NAMESPACE_END
