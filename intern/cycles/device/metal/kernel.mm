/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_METAL

#  include <algorithm>
#  include <atomic>
#  include <chrono>
#  include <deque>
#  include <map>
#  include <set>
#  include <thread>
#  include <vector>

#  include "device/metal/device_impl.h"
#  include "device/metal/kernel.h"

#  include "kernel/device/metal/function_constants.h"

#  include "util/debug.h"
#  include "util/log.h"
#  include "util/md5.h"
#  include "util/path.h"
#  include "util/tbb.h"
#  include "util/time.h"
#  include "util/unique_ptr.h"

CCL_NAMESPACE_BEGIN

const char *kernel_type_as_string(MetalPipelineType pso_type)
{
  switch (pso_type) {
    case PSO_GENERIC:
      return "PSO_GENERIC";
    case PSO_SPECIALIZED_INTERSECT:
      return "PSO_SPECIALIZED_INTERSECT";
    case PSO_SPECIALIZED_SHADE:
      return "PSO_SPECIALIZED_SHADE";
    case PSO_SPECIALIZED_LIGHT_CACHE:
      return "PSO_SPECIALIZED_LIGHT_CACHE";
    case PSO_SPECIALIZED_EVAL:
      return "PSO_SPECIALIZED_EVAL";
    default:
      assert(0);
  }
  return "";
}

static bool is_light_cache_kernel(const DeviceKernel kernel)
{
  return kernel == DEVICE_KERNEL_INTEGRATOR_PHOTON_EMIT ||
         kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_LIGHT_GENERATE ||
         kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_CACHE_ORDER ||
         kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_SENSOR_CONNECT;
}

static bool is_shader_eval_kernel(const DeviceKernel kernel)
{
  return kernel >= DEVICE_KERNEL_SHADER_EVAL_DISPLACE &&
         kernel <= DEVICE_KERNEL_SHADER_EVAL_VOLUME_DENSITY;
}

struct ShaderCache {
  ShaderCache(id<MTLDevice> _mtlDevice) : mtlDevice(_mtlDevice)
  {
    /* Initialize occupancy tuning LUT. */

    // TODO: Look into tuning for DEVICE_KERNEL_INTEGRATOR_INTERSECT_DEDICATED_LIGHT and
    // DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT, DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_*.

    switch (MetalInfo::get_apple_gpu_architecture(mtlDevice)) {
      default:
      case APPLE_M3:
        /* Peak occupancy is achieved through Dynamic Caching on M3 GPUs. */
        for (size_t i = 0; i < DEVICE_KERNEL_NUM; i++) {
          occupancy_tuning[i] = {64, 64};
        }
        break;
      case APPLE_M2_BIG:
        /* The light-path megakernel retains shader and reciprocal-transport state.
         * A small group gives the compiler more registers per active thread. */
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_BDPT_LIGHT_GENERATE] = {64, 64};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES] = {384, 128};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA] = {640, 128};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST] = {1024, 64};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW] = {704, 704};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE] = {640, 32};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY] = {896, 768};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND] = {512, 128};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW] = {32, 32};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE] = {768, 576};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY] = {896, 768};
        break;
      case APPLE_M2:
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES] = {32, 32};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA] = {832, 32};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST] = {64, 64};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW] = {64, 64};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE] = {704, 32};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY] = {1024, 256};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND] = {64, 32};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW] = {256, 256};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE] = {448, 384};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY] = {1024, 1024};
        break;
      case APPLE_M1:
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES] = {256, 128};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA] = {768, 32};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST] = {512, 128};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW] = {384, 128};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE] = {512, 64};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY] = {512, 256};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND] = {512, 128};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW] = {384, 32};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE] = {576, 384};
        occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY] = {832, 832};
        break;
    }

    occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SORT_BUCKET_PASS] = {1024, 1024};
    occupancy_tuning[DEVICE_KERNEL_INTEGRATOR_SORT_WRITE_PASS] = {1024, 1024};
  }
  ~ShaderCache();

  /* Get the fastest available pipeline for the specified kernel. */
  std::shared_ptr<const MetalKernelPipeline> get_best_pipeline(DeviceKernel kernel,
                                                             const MetalDevice *device);

  /* Non-blocking request for a kernel, optionally specialized to the scene being rendered by
   * device. */
  void load_kernel(DeviceKernel kernel,
                   MetalDevice *device,
                   MetalPipelineType pso_type,
                   bool prewarm = false);

  /* With `prewarm`, also accept kernels that the current scene does not use yet. */
  bool should_load_kernel(DeviceKernel device_kernel,
                          const MetalDevice *device,
                          MetalPipelineType pso_type,
                          bool prewarm = false);

  void wait_for_all();

  friend ShaderCache *get_shader_cache(id<MTLDevice> mtlDevice);

  void compile_thread_func();
  /* Links pipelines compiled before the shading functions were ready. */
  void link_thread_func();
  /* Make a finished pipeline available to get_best_pipeline(). */
  void publish(unique_ptr<MetalKernelPipeline> pipeline);

  using PipelineCollection = std::vector<std::shared_ptr<MetalKernelPipeline>>;

  struct OccupancyTuningParameters {
    int threads_per_threadgroup = 0;
    int num_threads_per_block = 0;
  } occupancy_tuning[DEVICE_KERNEL_NUM];

  std::mutex cache_mutex;

  PipelineCollection pipelines[DEVICE_KERNEL_NUM];
  id<MTLDevice> mtlDevice;

  static bool running;
  std::condition_variable cond_var;
  std::deque<unique_ptr<MetalKernelPipeline>> request_queue;
  /* Prewarm requests only start when no request of a scene is pending. */
  std::deque<unique_ptr<MetalKernelPipeline>> prewarm_queue;
  int incomplete_scene_requests = 0;
  /* Prewarming leaves one compilation thread free for requests of a scene. */
  int active_prewarm_compilations = 0;
  std::vector<std::thread> compile_threads;
  std::atomic_int incomplete_requests = 0;
  std::atomic_int incomplete_specialization_requests = 0;

  /* Separately compiled shading functions, by library checksum. */
  std::map<string, std::shared_ptr<MetalVisibleFunctions>> visible_functions;
  std::vector<std::thread> visible_function_threads;
  /* Specialized pixel displacement functions, by generic functions and evaluator set. */
  std::map<std::pair<const MetalVisibleFunctions *, int>,
           std::shared_ptr<MetalDisplacementFunctions>>
      displacement_functions;
  /* Requests that are queued or compiling, to avoid compiling the same pipeline twice. */
  std::set<std::pair<DeviceKernel, string>> in_flight;
  std::deque<unique_ptr<MetalKernelPipeline>> link_queue;
  std::condition_variable link_cond;
  std::thread link_thread;
};

bool ShaderCache::running = true;

const int MAX_POSSIBLE_GPUS_ON_SYSTEM = 8;
using DeviceShaderCache = std::pair<id<MTLDevice>, unique_ptr<ShaderCache>>;
int g_shaderCacheCount = 0;
DeviceShaderCache g_shaderCache[MAX_POSSIBLE_GPUS_ON_SYSTEM];

/* Next UID for associating a MetalDispatchPipeline with an originating MetalKernelPipeline. */
static std::atomic_int g_next_pipeline_id = 0;

ShaderCache *get_shader_cache(id<MTLDevice> mtlDevice)
{
  for (int i = 0; i < g_shaderCacheCount; i++) {
    if (g_shaderCache[i].first == mtlDevice) {
      return g_shaderCache[i].second.get();
    }
  }

  static thread_mutex g_shaderCacheCountMutex;
  g_shaderCacheCountMutex.lock();
  int index = g_shaderCacheCount++;
  g_shaderCacheCountMutex.unlock();

  assert(index < MAX_POSSIBLE_GPUS_ON_SYSTEM);
  g_shaderCache[index].first = mtlDevice;
  g_shaderCache[index].second = make_unique<ShaderCache>(mtlDevice);
  return g_shaderCache[index].second.get();
}

ShaderCache::~ShaderCache()
{
  running = false;
  cond_var.notify_all();
  link_cond.notify_all();

  metal_printf("Waiting for ShaderCache threads... (incomplete_requests = %d)",
               int(incomplete_requests));
  for (auto &thread : compile_threads) {
    thread.join();
  }
  if (link_thread.joinable()) {
    link_thread.join();
  }
  for (auto &thread : visible_function_threads) {
    thread.join();
  }
  metal_printf("ShaderCache shut down.");
}

void ShaderCache::wait_for_all()
{
  while (incomplete_requests > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void ShaderCache::publish(unique_ptr<MetalKernelPipeline> pipeline)
{
  const DeviceKernel device_kernel = pipeline->device_kernel;
  const MetalPipelineType pso_type = pipeline->pso_type;
  /* Collected before taking cache_mutex: devices request kernels while holding their own
   * registry mutex. */
  const vector<string> active_md5 = MetalDevice::active_kernels_md5(pso_type);
  {
    thread_scoped_lock lock(cache_mutex);
    if (pso_type == PSO_GENERIC) {
      in_flight.erase({device_kernel, pipeline->kernels_md5});
    }
    if (!pipeline->prewarm) {
      incomplete_scene_requests--;
    }
    auto &collection = pipelines[device_kernel];

    /* Cache up to 3 kernel variants with the same pso_type in memory, purging oldest first.
     * Never purge the variant of an active device: it has already requested its kernels and
     * waits for exactly this pipeline, see get_best_pipeline() and MetalDevice::is_ready().
     * Other variants are published at any time by prewarming and by other devices. */
    int max_entries_of_same_pso_type = 3;
    for (int i = (int)collection.size() - 1; i >= 0; i--) {
      if (collection[i]->pso_type == pso_type) {
        max_entries_of_same_pso_type -= 1;
        if (max_entries_of_same_pso_type <= 0 &&
            std::find(active_md5.begin(), active_md5.end(), collection[i]->kernels_md5) ==
                active_md5.end())
        {
          metal_printf("Purging oldest %s:%s kernel from ShaderCache",
                       kernel_type_as_string(pso_type),
                       device_kernel_as_string(device_kernel));
          collection.erase(collection.begin() + i);
          break;
        }
      }
    }
    collection.push_back(std::move(pipeline));
  }
  incomplete_requests--;
  if (pso_type != PSO_GENERIC) {
    incomplete_specialization_requests--;
  }
  /* Waiting prewarm requests may start now. */
  cond_var.notify_all();
}

void ShaderCache::compile_thread_func()
{
  while (running) {

    /* wait for / acquire next request */
    unique_ptr<MetalKernelPipeline> pipeline;
    {
      thread_scoped_lock lock(cache_mutex);
      const int max_prewarm = max(int(compile_threads.size()) - 1, 1);
      const auto has_work = [&] {
        return !request_queue.empty() ||
               (!prewarm_queue.empty() && incomplete_scene_requests == 0 &&
                active_prewarm_compilations < max_prewarm);
      };
      cond_var.wait(lock, [&] { return !running || has_work(); });
      if (!running || !has_work()) {
        continue;
      }

      auto &queue = request_queue.empty() ? prewarm_queue : request_queue;
      pipeline = std::move(queue.front());
      queue.pop_front();
      if (pipeline->prewarm) {
        active_prewarm_compilations++;
      }
    }

    /* Service the request. */
    DeviceKernel device_kernel = pipeline->device_kernel;
    MetalPipelineType pso_type = pipeline->pso_type;

    if (pso_type != PSO_GENERIC &&
        MetalDevice::is_device_cancelled(pipeline->originating_device_id))
    {
      /* The originating MetalDevice is no longer active, so this scene specialization is
       * obsolete. Generic pipelines do not depend on the device or scene and are always
       * finished: a later device reuses them, and may already rely on this request. A cancelled
       * request is not published, so it does not look like a failed compilation. */
      metal_printf("Cancelling compilation of %s (%s)",
                   device_kernel_as_string(device_kernel),
                   kernel_type_as_string(pso_type));
      [pipeline->mtlLibrary release];
      pipeline->mtlLibrary = nil;
      {
        thread_scoped_lock lock(cache_mutex);
        incomplete_scene_requests--;
      }
      incomplete_requests--;
      incomplete_specialization_requests--;
      cond_var.notify_all();
      continue;
    }
    else {
      /* Drain temporary compiler objects after each job. This worker lives for the entire
       * session; archives and descriptors must not accumulate across scene specializations. */
      @autoreleasepool {
        pipeline->compile();
        /* Completed PSOs do not need the source library or their creation function. Keep only
         * the intersection functions used when constructing dispatch tables. */
        [pipeline->function release];
        pipeline->function = nil;
        [pipeline->mtlLibrary release];
        pipeline->mtlLibrary = nil;
      }
      if (pipeline->prewarm) {
        thread_scoped_lock lock(cache_mutex);
        active_prewarm_compilations--;
        cond_var.notify_all();
      }

      if (pipeline->awaiting_link) {
        /* Keep compiling other kernels while the shading functions finish. */
        {
          thread_scoped_lock lock(cache_mutex);
          link_queue.push_back(std::move(pipeline));
        }
        link_cond.notify_one();
        continue;
      }
    }
    publish(std::move(pipeline));
  }
}

void ShaderCache::link_thread_func()
{
  while (running) {
    unique_ptr<MetalKernelPipeline> pipeline;
    {
      thread_scoped_lock lock(cache_mutex);
      link_cond.wait(lock, [&] { return !running || !link_queue.empty(); });
      if (!running || link_queue.empty()) {
        continue;
      }
      pipeline = std::move(link_queue.front());
      link_queue.pop_front();
    }
    @autoreleasepool {
      pipeline->link_visible_functions();
    }
    publish(std::move(pipeline));
  }
}

/* Whether the current scene can enqueue the kernel. */
static bool should_load_kernel_for_scene(const DeviceKernel device_kernel,
                                         const MetalDevice *device,
                                         const MetalPipelineType pso_type)
{
  if (device_kernel == DEVICE_KERNEL_INTEGRATOR_INIT_FROM_BAKE &&
      !(device->kernel_features & KERNEL_FEATURE_BAKING))
  {
    return false;
  }
  if (device_kernel == DEVICE_KERNEL_SHADER_EVAL_CURVE_SHADOW_TRANSPARENCY &&
      !(device->kernel_features & KERNEL_FEATURE_HAIR))
  {
    return false;
  }
  if ((device_kernel == DEVICE_KERNEL_SHADER_EVAL_VOLUME_DENSITY ||
       device_kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK ||
       device_kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME ||
       device_kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME_RAY_MARCHING) &&
      !(device->kernel_features & KERNEL_FEATURE_VOLUME))
  {
    return false;
  }
  if (device_kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE &&
      !(device->kernel_features & KERNEL_FEATURE_SUBSURFACE))
  {
    return false;
  }
  if ((device_kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_DEDICATED_LIGHT ||
       device_kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT) &&
      !(device->kernel_features & KERNEL_FEATURE_SHADOW_LINKING))
  {
    return false;
  }
  if (device_kernel == DEVICE_KERNEL_INTEGRATOR_PHOTON_EMIT &&
      !(device->scene_kernel_features & KERNEL_FEATURE_PHOTON_MAPPING))
  {
    return false;
  }
  if ((device_kernel == DEVICE_KERNEL_GUIDING_BEGIN_UPDATE ||
       device_kernel == DEVICE_KERNEL_GUIDING_REFINE ||
       device_kernel == DEVICE_KERNEL_GUIDING_PUBLISH ||
       device_kernel == DEVICE_KERNEL_GUIDING_FLUSH_HISTORY ||
       device_kernel == DEVICE_KERNEL_GUIDING_PARTITION_COUNT ||
       device_kernel == DEVICE_KERNEL_GUIDING_PARTITION_PREFIX ||
       device_kernel == DEVICE_KERNEL_GUIDING_PARTITION_SCATTER ||
       device_kernel == DEVICE_KERNEL_GUIDING_FIT ||
       device_kernel == DEVICE_KERNEL_GUIDING_FIT_REDUCE) &&
      !(device->scene_kernel_features & KERNEL_FEATURE_PATH_GUIDING))
  {
    return false;
  }
  if ((device_kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_LIGHT_GENERATE ||
       device_kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_CACHE_ORDER ||
       device_kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_SENSOR_CONNECT) &&
      (!(device->scene_kernel_features & KERNEL_FEATURE_BDPT) ||
       (device->scene_kernel_features & KERNEL_FEATURE_SHADOW_CATCHER)))
  {
    return false;
  }
  if (device_kernel == DEVICE_KERNEL_INTEGRATOR_SHADOW_CATCHER_COUNT_POSSIBLE_SPLITS &&
      !(device->kernel_features & KERNEL_FEATURE_SHADOW_CATCHER))
  {
    return false;
  }

  if (device_kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE) {
    if ((device->kernel_features & KERNEL_FEATURE_NODE_RAYTRACE) == 0) {
      /* Skip shade_surface_raytrace kernel if the scene doesn't require it. */
      return false;
    }
  }

  if (device_kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE) {
    if ((device->kernel_features & KERNEL_FEATURE_MNEE) == 0) {
      /* Skip the MNEE kernel if the scene doesn't require it. */
      return false;
    }
    /* BDPT needs the manifold solver inside its light tracer even when regular camera-path
     * shadow caustics are disabled. Only the latter can enqueue this standalone kernel. */
    if (pso_type != PSO_GENERIC && !device->launch_params->data.integrator.use_caustics) {
      return false;
    }
  }

  return true;
}

bool ShaderCache::should_load_kernel(DeviceKernel device_kernel,
                                     const MetalDevice *device,
                                     MetalPipelineType pso_type,
                                     const bool prewarm)
{
  if (!running) {
    return false;
  }

  if (!device_kernel_has_gpu_function(device_kernel, true)) {
    /* Skip megakernel and other markers without a GPU function. */
    return false;
  }

  if (pso_type == PSO_GENERIC && device->requires_scene_specialization() &&
      (is_light_cache_kernel(device_kernel) || is_shader_eval_kernel(device_kernel) ||
       (device_kernel >= DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND &&
        device_kernel <= DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT)))
  {
    return false;
  }

  /* Small-memory devices require the specialized integrator set. Displacement also requires
   * it on larger devices. Do not compile an unused generic set before the required pipelines. */
  if (pso_type == PSO_GENERIC &&
      (device->requires_scene_specialization() ||
       device->pixel_displacement_requires_specialization()) &&
      device_kernel >= DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST &&
      device_kernel <= DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT)
  {
    return false;
  }

  /* Avoid materializing pipelines that the current scene cannot enqueue. If a scene edit enables
   * one of these feature bits, load_kernels() is called again and requests the newly required
   * pipeline. Besides reducing cold-start work, this is particularly important for the mutually
   * exclusive photon and BDPT kernels, whose large shading call graphs are expensive on Metal.
   * Prewarming compiles them anyway, behind the pipelines the scene needs. */
  if (!prewarm && !should_load_kernel_for_scene(device_kernel, device, pso_type)) {
    return false;
  }

  if (pso_type == PSO_SPECIALIZED_EVAL) {
    if (!device->requires_scene_specialization() || !is_shader_eval_kernel(device_kernel)) {
      return false;
    }
  }
  else if (pso_type == PSO_SPECIALIZED_LIGHT_CACHE) {
    if (!device->requires_scene_specialization() || !is_light_cache_kernel(device_kernel)) {
      return false;
    }
  }
  else if (pso_type != PSO_GENERIC) {
    /* Only specialize kernels where it can make an impact. */
    if (device_kernel < DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST ||
        device_kernel > DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL)
    {
      return false;
    }

    /* Only specialize shading / intersection kernels as requested. */
    bool is_shade_kernel = (device_kernel >= DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND);
    bool is_shade_pso = (pso_type == PSO_SPECIALIZED_SHADE);
    if (is_shade_pso != is_shade_kernel) {
      return false;
    }
  }

  {
    /* check whether the kernel has already been requested / cached */
    thread_scoped_lock lock(cache_mutex);
    for (auto &pipeline : pipelines[device_kernel]) {
      if (pipeline->kernels_md5 == device->kernels_md5[pso_type]) {
        return false;
      }
    }
  }

  return true;
}

void ShaderCache::load_kernel(DeviceKernel device_kernel,
                              MetalDevice *device,
                              MetalPipelineType pso_type,
                              const bool prewarm)
{
  {
    /* create compiler threads on first run */
    thread_scoped_lock lock(cache_mutex);
    if (compile_threads.empty()) {
      /* Limit to 2 MTLCompiler instances by default. In macOS >= 13.3 we can query the upper
       * limit. */
      int max_mtlcompiler_threads = 2;

#  if defined(MAC_OS_VERSION_13_3)
      if (@available(macOS 13.3, *)) {
        /* Subtract one to avoid contention with the real-time GPU module. */
        max_mtlcompiler_threads = max(2,
                                      int([mtlDevice maximumConcurrentCompilationTaskCount]) - 1);
      }
#  endif

      if (device->requires_scene_specialization()) {
        max_mtlcompiler_threads = 1;
      }
      else if (MetalInfo::use_low_memory_compilation()) {
        /* Kernels no longer contain the shading code; the shading functions compile on their
         * own threads concurrently. Bound the total number of compiler jobs by memory. */
        max_mtlcompiler_threads = min(max_mtlcompiler_threads, 3);
      }

      metal_printf("Spawning %d Cycles kernel compilation threads", max_mtlcompiler_threads);
      for (int i = 0; i < max_mtlcompiler_threads; i++) {
        compile_threads.emplace_back([this] { this->compile_thread_func(); });
      }
      link_thread = std::thread([this] { this->link_thread_func(); });
    }
  }

  if (!should_load_kernel(device_kernel, device, pso_type, prewarm)) {
    return;
  }
  if (pso_type == PSO_GENERIC) {
    /* Generic requests are shared by all devices, see compile_thread_func(). */
    thread_scoped_lock lock(cache_mutex);
    if (!in_flight.insert({device_kernel, device->kernels_md5[pso_type]}).second) {
      if (!prewarm) {
        /* The scene now needs a pipeline that is only queued for prewarming: compile it next. */
        for (auto it = prewarm_queue.begin(); it != prewarm_queue.end(); ++it) {
          if ((*it)->device_kernel == device_kernel &&
              (*it)->kernels_md5 == device->kernels_md5[pso_type])
          {
            (*it)->prewarm = false;
            incomplete_scene_requests++;
            request_queue.push_front(std::move(*it));
            prewarm_queue.erase(it);
            cond_var.notify_one();
            break;
          }
        }
      }
      return;
    }
  }

  incomplete_requests++;
  if (pso_type != PSO_GENERIC) {
    incomplete_specialization_requests++;
  }

  unique_ptr<MetalKernelPipeline> pipeline = make_unique<MetalKernelPipeline>();

  /* Keep track of the originating device's ID so that we can cancel requests if the device ceases
   * to be active. */
  pipeline->pipeline_id = g_next_pipeline_id.fetch_add(1);
  pipeline->originating_device_id = device->device_id;
  pipeline->kernel_data_ = device->launch_params->data;
  pipeline->pso_type = pso_type;
  pipeline->mtlDevice = mtlDevice;
  pipeline->kernels_md5 = device->kernels_md5[pso_type];
  /* A scene edit may replace the device's library while this request is queued. */
  pipeline->mtlLibrary = [device->mtlLibrary[pso_type] retain];
  pipeline->complete_generic = pso_type == PSO_GENERIC && device->use_visible_shading;
  /* The software BVH library of the complete generic set calls its traversal and the pixel
   * displacement ray solver as separate functions. */
  pipeline->software_bvh_library = pipeline->complete_generic &&
                                         !device->use_metalrt_for_current_scene();
  if (pipeline->complete_generic &&
      metal_kernel_uses_shading_functions(device_kernel, pipeline->software_bvh_library))
  {
    pipeline->visible_functions = device->visible_functions;
  }
  pipeline->device_kernel = device_kernel;
  pipeline->threads_per_threadgroup = device->max_threads_per_threadgroup;

  if (occupancy_tuning[device_kernel].threads_per_threadgroup) {
    pipeline->threads_per_threadgroup = occupancy_tuning[device_kernel].threads_per_threadgroup;
    pipeline->num_threads_per_block = occupancy_tuning[device_kernel].num_threads_per_block;
  }

  /* General direct displacement keeps shadow rays active for very different numbers of samples.
   * Smaller dispatch groups reduce that scheduling tail on M2 Max/Ultra. Preserve the
   * architecture's compiler register budget; reducing it hurts this evaluator's throughput. */
  if (MetalInfo::get_apple_gpu_architecture(mtlDevice) == APPLE_M2_BIG &&
      pso_type == PSO_SPECIALIZED_INTERSECT && device->scene_use_pixel_displacement &&
      device->scene_pixel_displacement_scale != 0.0f &&
      device->scene_pixel_displacement_max_distance > 0.0f &&
      !(pipeline->kernel_data_.integrator.pixel_displacement_evaluator_set &
        PIXEL_DISPLACEMENT_UNCERTIFIED_INPUTS) &&
      device_kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW)
  {
    pipeline->num_threads_per_block = 128;
  }

  /* The certified normal-image solver omits the general evaluator and its large
   * temporary state. On M2 Max/Ultra, a reduced compiler thread limit
   * and small dispatch groups improve occupancy and reduce long-ray scheduling tails. */
  if (MetalInfo::get_apple_gpu_architecture(mtlDevice) == APPLE_M2_BIG &&
      pso_type == PSO_SPECIALIZED_INTERSECT &&
      (pipeline->kernel_data_.integrator.pixel_displacement_evaluator_set &
       PIXEL_DISPLACEMENT_NORMAL_IMAGE_INPUTS) &&
      (device_kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST ||
       device_kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW))
  {
    pipeline->threads_per_threadgroup = 512;
    pipeline->num_threads_per_block = 64;
  }

  /* metalrt options */
  pipeline->use_metalrt = device->use_metalrt_for_current_scene();
  /* Geometry features accumulate on the device, but coherent surface tracing
   * follows the active scene. Its intersection-table choice is in the cache key. */
  pipeline->kernel_features =
      (device->kernel_features & ~(KERNEL_FEATURE_COHERENT_SPECULAR | KERNEL_FEATURE_POLARIZATION)) |
      (device->scene_kernel_features & (KERNEL_FEATURE_COHERENT_SPECULAR | KERNEL_FEATURE_POLARIZATION));

  pipeline->prewarm = prewarm;
  {
    thread_scoped_lock lock(cache_mutex);
    if (prewarm) {
      prewarm_queue.push_back(std::move(pipeline));
    }
    else {
      incomplete_scene_requests++;
      request_queue.push_back(std::move(pipeline));
    }
  }
  cond_var.notify_one();
}

std::shared_ptr<const MetalKernelPipeline> ShaderCache::get_best_pipeline(
    DeviceKernel kernel, const MetalDevice *device)
{
  /* Generic kernels omit pixel displacement. These specializations are required for
   * correctness, so pending or failed compilation must never select the generic fallback. */
  const bool requires_displacement = device->pixel_displacement_requires_specialization() &&
                                     kernel >= DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST &&
                                     kernel <= DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT;
  const bool requires_scene_integrator = device->requires_scene_specialization() &&
                                        kernel >= DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST &&
                                        kernel <= DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT;
  const bool requires_light_cache = device->requires_scene_specialization() &&
                                    is_light_cache_kernel(kernel);
  const bool requires_shader_eval = device->requires_scene_specialization() &&
                                    is_shader_eval_kernel(kernel);
  const MetalPipelineType required_type = requires_shader_eval ? PSO_SPECIALIZED_EVAL :
                                          requires_light_cache ? PSO_SPECIALIZED_LIGHT_CACHE :
                                          kernel < DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND ?
                                                                 PSO_SPECIALIZED_INTERSECT :
                                                                 PSO_SPECIALIZED_SHADE;
  while (running && !device->has_error) {
    /* Search all loaded pipelines with matching kernels_md5 checksums. */
    std::shared_ptr<MetalKernelPipeline> best_match;
    bool generic_failed = false;
    {
      thread_scoped_lock lock(cache_mutex);
      for (auto &candidate : pipelines[kernel]) {
        /* Only completed requests enter this collection. A matching failed generic
         * entry cannot become ready by waiting; optional specializations may still
         * provide a usable replacement, so finish the search before reporting it. */
        generic_failed |= candidate->pso_type == PSO_GENERIC && !candidate->loaded &&
                          candidate->kernels_md5 == device->kernels_md5[PSO_GENERIC];
        if (requires_displacement || requires_scene_integrator || requires_light_cache ||
            requires_shader_eval)
        {
          if (candidate->pso_type != required_type ||
              candidate->kernels_md5 != device->kernels_md5[required_type])
          {
            continue;
          }
          /* Entries are published only after compile() returns. An unloaded matching
           * entry therefore represents a failed compilation, not an in-flight request. */
          if (!candidate->loaded) {
            return nullptr;
          }
        }
        if (candidate->loaded &&
            candidate->kernels_md5 == device->kernels_md5[candidate->pso_type])
        {
          /* Replace existing match if candidate is more specialized. */
          if (!best_match || candidate->pso_type > best_match->pso_type) {
            best_match = candidate;
          }
        }
      }
    }

    if (best_match) {
      if (best_match->usage_count == 0 && best_match->pso_type != PSO_GENERIC) {
        metal_printf("Swapping in %s version of %s",
                     kernel_type_as_string(best_match->pso_type),
                     device_kernel_as_string(kernel));
      }
      best_match->usage_count += 1;
      return best_match;
    }

    if (generic_failed) {
      return nullptr;
    }

    /* Spin until a matching kernel is loaded, or we're shutting down. */
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return nullptr;
}

bool MetalKernelPipeline::should_use_binary_archive() const
{
  /* Issues with binary archives in older macOS versions. */
  if (@available(macOS 15.4, *)) {
    if (auto *str = getenv("CYCLES_METAL_DISABLE_BINARY_ARCHIVES")) {
      if (atoi(str) != 0) {
        /* Don't archive if we have opted out by env var. */
        return false;
      }
    }

    if (use_metalrt && metal_kernel_has_intersection(device_kernel, kernel_features)) {
      /* Binary linked functions aren't supported in binary archives. */
      return false;
    }

    if (pso_type == PSO_GENERIC) {
      /* Archive the generic kernels. */
      return true;
    }

    if (pso_type == PSO_SPECIALIZED_INTERSECT || pso_type == PSO_SPECIALIZED_LIGHT_CACHE ||
        (device_kernel >= DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND &&
         device_kernel <= DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW) ||
        (device_kernel >= DEVICE_KERNEL_SHADER_EVAL_DISPLACE &&
         device_kernel <= DEVICE_KERNEL_SHADER_EVAL_VOLUME_DENSITY))
    {
      /* Archive all specialized intersection and shade kernels. Intersection kernels used to be
       * left to Metal's opaque system shader cache on the assumption that they were cheap. Pixel
       * displacement and bidirectional/photon tracing make that assumption invalid, and losing
       * the system cache otherwise forces full recompilation. */
      return true;
    }

    /* The remaining kernels are all fast to compile. They may get cached by the system shader
     * cache, but will be quick to regenerate if not. */
  }
  return false;
}

/* Kernels whose rays intersect pixel displaced surfaces. Rays traced while shading use the base
 * triangles, see pixel_displacement_intersects(). */
static bool metal_kernel_pixel_displacement_rays(const DeviceKernel kernel)
{
  return kernel >= DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST &&
         kernel <= DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE;
}

/* `displacement_evaluator_set` specializes the pixel displacement functions when not negative,
 * see MetalDisplacementFunctions. */
static MTLFunctionConstantValues *GetConstantValues(const KernelData *data = nullptr,
                                                    const MetalPipelineType pso_type = PSO_GENERIC,
                                                    const bool pixel_displacement_rays = false,
                                                    const int displacement_evaluator_set = -1,
                                                    const int displacement_bvh_features = 0)
{
  MTLFunctionConstantValues *constant_values = [MTLFunctionConstantValues new];
  [constant_values setConstantValue:&pixel_displacement_rays
                               type:MTLDataTypeBool
                            atIndex:Kernel_PixelDisplacementRays];
  const bool displacement_specialized = displacement_evaluator_set >= 0;
  const int evaluator_set = max(displacement_evaluator_set, 0);
  [constant_values setConstantValue:&displacement_specialized
                               type:MTLDataTypeBool
                            atIndex:Kernel_PixelDisplacementSpecialized];
  [constant_values setConstantValue:&evaluator_set
                               type:MTLDataTypeInt
                            atIndex:Kernel_PixelDisplacementEvaluatorSet];
  [constant_values setConstantValue:&displacement_bvh_features
                               type:MTLDataTypeInt
                            atIndex:Kernel_PixelDisplacementBVHFeatures];

  MTLDataType MTLDataType_int = MTLDataTypeInt;
  MTLDataType MTLDataType_float = MTLDataTypeFloat;
  MTLDataType MTLDataType_float2 = MTLDataTypeFloat2;
  MTLDataType MTLDataType_float4 = MTLDataTypeFloat4;
  KernelData zero_data = {0};
  if (!data) {
    data = &zero_data;
  }
  KernelData specialization_data = *data;
  if (pso_type == PSO_SPECIALIZED_INTERSECT) {
    specialization_data.integrator.use_bidirectional_path_tracing = 0;
    specialization_data.integrator.use_photon_mapping = 0;
    data = &specialization_data;
  }
  [constant_values setConstantValue:&zero_data type:MTLDataType_int atIndex:Kernel_DummyConstant];

  bool next_member_is_specialized = true;

#  define KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE next_member_is_specialized = false;

#  define KERNEL_STRUCT_MEMBER(parent, _type, name) \
    [constant_values setConstantValue:next_member_is_specialized ? (void *)&data->parent.name : \
                                                                   (void *)&zero_data \
                                 type:MTLDataType_##_type \
                              atIndex:KernelData_##parent##_##name]; \
    next_member_is_specialized = true;

#  include "kernel/data_template.h"

  [constant_values setConstantValue:&data->kernel_features
                               type:MTLDataTypeULong
                            atIndex:KernelData_kernel_features];

  return [constant_values autorelease];
}

/* -------------------------------------------------------------------- */
/** \name Separately compiled shading functions
 * \{ */

static bool metal_binary_archives_enabled()
{
  /* Same policy as pipeline archives: issues with binary archives in older macOS versions. */
  if (@available(macOS 15.4, *)) {
    if (const char *str = getenv("CYCLES_METAL_DISABLE_BINARY_ARCHIVES")) {
      return atoi(str) == 0;
    }
    return true;
  }
  return false;
}

MetalVisibleFunctions::MetalVisibleFunctions(id<MTLDevice> device,
                                             id<MTLLibrary> library,
                                             const string &library_md5)
    : device_(device), library_([library retain]), library_md5_(library_md5)
{
  /* Table entries are numbered consecutively; the library defines how many exist. */
  const auto add_numbered = [&](const MetalVisibleFunctionTable table, const char *prefix) {
    for (int i = 0;; i++) {
      const string name = string_printf("%s%d", prefix, i);
      id<MTLFunction> function = [library_ newFunctionWithName:@(name.c_str())];
      if (!function) {
        break;
      }
      [function release];
      names_[table].push_back(name);
    }
  };
  /* A function can be compiled out together with its feature; its table then stays empty and
   * no kernel calls it. */
  const auto add_single = [&](const MetalVisibleFunctionTable table, const char *name) {
    id<MTLFunction> function = [library_ newFunctionWithName:@(name)];
    if (function) {
      [function release];
      names_[table].push_back(name);
    }
  };
  add_numbered(METAL_VFT_SVM, "cycles_metal_svm_");
  add_single(METAL_VFT_SVM_NODE, "cycles_metal_svm_node");
  add_single(METAL_VFT_SVM_CLOSURE, "cycles_metal_svm_closure");
  add_single(METAL_VFT_BSDF_EVAL, "cycles_metal_bsdf_eval");
  add_single(METAL_VFT_BSDF_EVAL_DELTA, "cycles_metal_bsdf_eval_delta");
  add_single(METAL_VFT_BSDF_SAMPLE, "cycles_metal_bsdf_sample");
  add_numbered(METAL_VFT_SURFACE, "cycles_metal_surface_");
  add_single(METAL_VFT_POLARIZATION, "cycles_metal_polarization_surface_transport");
  add_single(METAL_VFT_DIFFRACTION, "cycles_metal_diffraction_power_column");
  add_single(METAL_VFT_MNEE, "cycles_metal_mnee_sample");
  add_single(METAL_VFT_PIXEL_DISPLACEMENT_EVAL, "cycles_metal_pixel_displacement_eval");
  add_single(METAL_VFT_PIXEL_DISPLACEMENT_INTERSECT, "cycles_metal_pixel_displacement_intersect");
  add_single(METAL_VFT_SCENE_INTERSECT, "cycles_metal_scene_intersect");
}

MetalVisibleFunctions::~MetalVisibleFunctions()
{
  for (NSArray *functions : table_functions) {
    [functions release];
  }
  [binary_functions release];
  [library_ release];
}

id<MTLFunction> MetalVisibleFunctions::compile_function(const string &name,
                                                        string &error,
                                                        const int evaluator_set,
                                                        const int bvh_features)
{
  /* MetalDevice only uses visible shading on macOS 13 and newer. */
  if (@available(macOS 13.0, *)) {
    return compile_function_macos13(name, error, evaluator_set, bvh_features);
  }
  error = "unsupported macOS version";
  return nil;
}

id<MTLFunction> MetalVisibleFunctions::compile_specialized(const string &name,
                                                           const int evaluator_set,
                                                           const int bvh_features,
                                                           string &error)
{
  return compile_function(name, error, evaluator_set, bvh_features);
}

id<MTLFunction> MetalVisibleFunctions::compile_function_macos13(const string &name,
                                                                string &error,
                                                                const int evaluator_set,
                                                                const int bvh_features)
{
  MTLFunctionDescriptor *desc = [MTLFunctionDescriptor functionDescriptor];
  desc.name = @(name.c_str());
  desc.options = MTLFunctionOptionCompileToBinary;
  desc.constantValues = GetConstantValues(
      nullptr, PSO_GENERIC, false, evaluator_set, bvh_features);

  /* A binary archive per function keeps the GPU binary across sessions, independent of the
   * system shader cache. The library checksum identifies the build and kernel configuration. */
  id<MTLBinaryArchive> archive = nil;
  string archive_path;
  bool loading_archive = false;
  if (metal_binary_archives_enabled()) {
    MD5Hash md5;
    md5.append(library_md5_);
    md5.append(name);
    if (evaluator_set >= 0) {
      md5.append(string_printf("evaluator_set=%d bvh_features=%d", evaluator_set, bvh_features));
    }
    md5.append([[[NSProcessInfo processInfo] operatingSystemVersionString] UTF8String]);
    string device_name = [device_.name UTF8String];
    for (char &c : device_name) {
      if (!isalnum(c)) {
        c = '_';
      }
    }
    /* Specialized functions have their own directory, so their variants never evict the
     * generic functions. */
    const string directory = (evaluator_set >= 0) ? name + "_specialized" : name;
    archive_path = path_cache_get(path_join(
        "kernels", path_join(device_name, path_join(directory, md5.get_hex() + ".bin"))));
    path_create_directories(archive_path);
    loading_archive = path_cache_kernel_exists_and_mark_used(archive_path);

    MTLBinaryArchiveDescriptor *archive_desc = [[MTLBinaryArchiveDescriptor alloc] init];
    if (loading_archive) {
      archive_desc.url = [NSURL fileURLWithPath:@(archive_path.c_str())];
    }
    NSError *archive_error = nil;
    archive = [device_ newBinaryArchiveWithDescriptor:archive_desc error:&archive_error];
    [archive_desc release];
    [archive autorelease];
  }

  NSError *compile_error = nil;
  id<MTLFunction> function = nil;
  if (archive && loading_archive) {
    desc.binaryArchives = @[ archive ];
    if (@available(macOS 15.0, *)) {
      /* Otherwise a stale archive compiles the function again, and is rewritten below only
       * when it fails to load. */
      desc.options = MTLFunctionOptionCompileToBinary | MTLFunctionOptionFailOnBinaryArchiveMiss;
    }
    function = [library_ newFunctionWithDescriptor:desc error:&compile_error];
    desc.binaryArchives = nil;
    desc.options = MTLFunctionOptionCompileToBinary;
    if (!function) {
      /* Stale or incomplete archive: rebuild it. */
      path_remove(archive_path);
      loading_archive = false;
      NSError *archive_error = nil;
      MTLBinaryArchiveDescriptor *archive_desc = [[MTLBinaryArchiveDescriptor alloc] init];
      archive = [[device_ newBinaryArchiveWithDescriptor:archive_desc error:&archive_error]
          autorelease];
      [archive_desc release];
    }
  }

  if (!function) {
    compile_error = nil;
    function = [library_ newFunctionWithDescriptor:desc error:&compile_error];
    if (!function) {
      error = compile_error ? [[compile_error localizedDescription] UTF8String] : "unknown error";
      return nil;
    }
    if (archive && ShaderCache::running) {
      NSError *archive_error = nil;
      if ([archive addFunctionWithDescriptor:desc library:library_ error:&archive_error] &&
          [archive serializeToURL:[NSURL fileURLWithPath:@(archive_path.c_str())]
                            error:&archive_error])
      {
        /* Keep a few variants: software/MetalRT libraries and the previous build, or the
         * evaluator sets of recent scenes. */
        path_cache_kernel_mark_added_and_clear_old(archive_path, (evaluator_set >= 0) ? 16 : 4);
      }
      else {
        metal_printf("Failed to archive visible function %s: %s",
                     name.c_str(),
                     archive_error ? [[archive_error localizedDescription] UTF8String] : "nil");
      }
    }
  }
  function.label = @(name.c_str());
  return function;
}

void MetalVisibleFunctions::compile(const int num_threads)
{
  const double start_time = time_dt();

  /* Longest functions first, so short ones fill the remaining compiler threads at the end.
   * The order only affects scheduling; it lists the measured slowest functions. */
  const char *slowest_first[] = {"cycles_metal_bsdf_sample",
                                 "cycles_metal_surface_1",
                                 "cycles_metal_diffraction_power_column",
                                 "cycles_metal_surface_2",
                                 "cycles_metal_surface_0",
                                 "cycles_metal_svm_node",
                                 "cycles_metal_pixel_displacement_eval",
                                 "cycles_metal_svm_1",
                                 "cycles_metal_surface_4",
                                 "cycles_metal_mnee_sample",
                                 "cycles_metal_bsdf_eval",
                                 "cycles_metal_scene_intersect",
                                 "cycles_metal_pixel_displacement_intersect",
                                 "cycles_metal_svm_closure",
                                 "cycles_metal_surface_6"};
  const auto priority = [&](const Job &job) {
    for (int p = 0; p < int(std::size(slowest_first)); p++) {
      if (names_[job.table][job.index] == slowest_first[p]) {
        return p;
      }
    }
    return int(std::size(slowest_first));
  };

  vector<std::thread> threads;
  {
    thread_scoped_lock lock(mutex_);
    for (int table = 0; table < METAL_VFT_NUM; table++) {
      for (int i = 0; i < int(names_[table].size()); i++) {
        jobs_.push_back({MetalVisibleFunctionTable(table), i});
      }
    }
    std::stable_sort(jobs_.begin(), jobs_.end(), [&](const Job &a, const Job &b) {
      return priority(a) < priority(b);
    });
    results_.resize(METAL_VFT_NUM);
    for (int table = 0; table < METAL_VFT_NUM; table++) {
      results_[table].resize(names_[table].size(), nil);
    }
    started_ = true;
    num_threads_ = max(max(num_threads, requested_threads_), 1);
    active_workers_ = num_threads_;
    for (int i = 1; i < num_threads_; i++) {
      threads.emplace_back([this]() { run_worker(); });
    }
  }
  run_worker();
  for (std::thread &thread : threads) {
    thread.join();
  }

  int final_num_threads;
  {
    /* Threads added by raise_threads() are joined by their owner, but finish the same jobs. */
    thread_scoped_lock lock(mutex_);
    cond_.wait(lock, [&] { return active_workers_ == 0; });
    final_num_threads = num_threads_;
  }

  NSMutableArray *all_functions = [[NSMutableArray alloc] init];
  bool complete = !failed_ && ShaderCache::running;
  for (int table = 0; table < METAL_VFT_NUM; table++) {
    NSMutableArray *functions = [[NSMutableArray alloc] init];
    for (id<MTLFunction> function : results_[table]) {
      if (function) {
        [functions addObject:function];
        [all_functions addObject:function];
        [function release];
      }
      else {
        complete = false;
      }
    }
    table_functions[table] = functions;
  }
  binary_functions = all_functions;

  if (!complete && !first_error_.empty()) {
    LOG_ERROR << "Metal visible function compilation failed: " << first_error_;
  }
  metal_printf("Visible functions %s in %.1f seconds (%d functions, %d threads)",
               complete ? "compiled" : "FAILED",
               time_dt() - start_time,
               int(binary_functions.count),
               final_num_threads);

  thread_scoped_lock lock(mutex_);
  success_ = complete;
  finished_ = true;
  cond_.notify_all();
}

void MetalVisibleFunctions::run_worker()
{
  while (ShaderCache::running && !failed_) {
    const int job_index = next_job_.fetch_add(1);
    if (job_index >= int(jobs_.size())) {
      break;
    }
    const Job &job = jobs_[job_index];
    const string &name = names_[job.table][job.index];
    @autoreleasepool {
      const double function_start = time_dt();
      string error;
      id<MTLFunction> function = compile_function(name, error);
      if (!function) {
        thread_scoped_lock lock(mutex_);
        if (first_error_.empty()) {
          first_error_ = name + ": " + error;
        }
        failed_ = true;
        break;
      }
      results_[job.table][job.index] = function;
      metal_printf("%16s | %-55s | %7.2fs",
                   "VISIBLE_FUNCTION",
                   name.c_str(),
                   time_dt() - function_start);
    }
  }
  thread_scoped_lock lock(mutex_);
  active_workers_--;
  cond_.notify_all();
}

void MetalVisibleFunctions::raise_threads(const int num_threads,
                                          std::vector<std::thread> &threads)
{
  thread_scoped_lock lock(mutex_);
  if (!started_) {
    /* compile() starts with at least this many threads. */
    requested_threads_ = max(requested_threads_, num_threads);
    return;
  }
  if (finished_ || active_workers_ == 0 || next_job_ >= int(jobs_.size())) {
    return;
  }
  for (; num_threads_ < num_threads; num_threads_++) {
    active_workers_++;
    threads.emplace_back([this]() { run_worker(); });
  }
}

bool MetalVisibleFunctions::wait()
{
  thread_scoped_lock lock(mutex_);
  cond_.wait(lock, [&] { return finished_; });
  return success_;
}

/** \} */

void MetalDispatchPipeline::free_intersection_function_tables()
{
  for (int table = 0; table < METALRT_TABLE_NUM; table++) {
    if (intersection_func_table[table]) {
      /* Add the table to the delayed free list of the device that created it. */
      metal_device->metal_mem_free(intersection_func_table[table]);
      intersection_func_table[table] = nil;
    }
  }
}

MetalKernelPipeline::~MetalKernelPipeline()
{
  [function release];
  [mtlLibrary release];
  [pipeline release];
  for (NSArray *functions : table_functions) {
    [functions release];
  }
  for (auto &it : displacement_pipelines_) {
    [it.second release];
  }
}

id<MTLComputePipelineState> MetalKernelPipeline::displacement_pipeline(
    const MetalDisplacementFunctions &functions) const
{
  thread_scoped_lock lock(displacement_mutex_);
  auto it = displacement_pipelines_.find(functions.key());
  if (it != displacement_pipelines_.end()) {
    return it->second;
  }
  id<MTLComputePipelineState> result = nil;
  if (pipeline && functions.eval) {
    /* Linking finished binaries does not recompile the pipeline. */
    NSMutableArray *added = [NSMutableArray arrayWithObject:functions.eval];
    if (functions.intersect) {
      [added addObject:functions.intersect];
    }
    if (functions.scene_intersect) {
      [added addObject:functions.scene_intersect];
    }
    NSError *error = nil;
    result = [pipeline newComputePipelineStateWithAdditionalBinaryFunctions:added error:&error];
    if (!result) {
      LOG_WARNING << "Failed to link specialized pixel displacement functions into "
                  << device_kernel_as_string(device_kernel) << ": "
                  << (error ? [[error localizedDescription] UTF8String] : "nil");
    }
  }
  /* Also remember failures, the generic functions remain in use. */
  displacement_pipelines_[functions.key()] = result;
  return result;
}

MetalDisplacementFunctions::MetalDisplacementFunctions(
    std::shared_ptr<MetalVisibleFunctions> generic, const int evaluator_set, const int bvh_features)
    : generic(std::move(generic)), evaluator_set(evaluator_set), bvh_features(bvh_features)
{
}

MetalDisplacementFunctions::~MetalDisplacementFunctions()
{
  [eval release];
  [intersect release];
  [scene_intersect release];
}

static std::atomic<int64_t> g_displacement_request_serial = 0;

void MetalDisplacementFunctions::compile()
{
  /* Leave the compiler to the functions a scene waits for. */
  if (!generic->wait() || !ShaderCache::running) {
    queued = false;
    return;
  }
  static thread_mutex compile_mutex;
  thread_scoped_lock compile_lock(compile_mutex);
  if (request_serial != g_displacement_request_serial || !ShaderCache::running) {
    /* A scene edit requested another specialization meanwhile. Requesting this one again
     * queues it again. */
    queued = false;
    return;
  }
  const double start_time = time_dt();
  /* The ray solver and traversal functions exist only in the software BVH library. */
  const bool software_bvh =
      generic->table_functions[METAL_VFT_PIXEL_DISPLACEMENT_INTERSECT].count != 0 &&
      generic->table_functions[METAL_VFT_SCENE_INTERSECT].count != 0;
  const char *names[3] = {"cycles_metal_pixel_displacement_eval",
                          "cycles_metal_pixel_displacement_intersect",
                          "cycles_metal_scene_intersect"};
  const int num_functions = software_bvh ? 3 : 1;
  id<MTLFunction> functions[3] = {nil, nil, nil};
  string errors[3];
  vector<std::thread> threads;
  for (int i = 1; i < num_functions; i++) {
    threads.emplace_back([&, i]() {
      @autoreleasepool {
        functions[i] = generic->compile_specialized(
            names[i], evaluator_set, bvh_features, errors[i]);
      }
    });
  }
  @autoreleasepool {
    functions[0] = generic->compile_specialized(names[0], evaluator_set, bvh_features, errors[0]);
  }
  for (std::thread &thread : threads) {
    thread.join();
  }
  for (int i = 0; i < num_functions; i++) {
    if (!functions[i]) {
      if (ShaderCache::running) {
        LOG_WARNING << "Failed to specialize " << names[i] << ": " << errors[i];
        failed = true;
      }
      for (id<MTLFunction> function : functions) {
        [function release];
      }
      queued = false;
      return;
    }
  }
  eval = functions[0];
  intersect = functions[1];
  scene_intersect = functions[2];
  metal_printf(
      "Pixel displacement functions specialized for evaluator set %d, BVH features %d in %.1f "
      "seconds",
      evaluator_set,
      bvh_features,
      time_dt() - start_time);
  ready_ = true;
}

void MetalDispatchPipeline::free_visible_function_tables()
{
  for (int table = 0; table < METAL_VFT_NUM; table++) {
    if (visible_func_table[table]) {
      metal_device->metal_mem_free(visible_func_table[table]);
      visible_func_table[table] = nil;
    }
  }
  use_visible_shading = false;
}

MetalDispatchPipeline::~MetalDispatchPipeline()
{
  free_intersection_function_tables();
  free_visible_function_tables();
  [pipeline release];
}

bool MetalDispatchPipeline::update(MetalDevice *metal_device, DeviceKernel kernel)
{
  this->metal_device = metal_device;
  /* Keep the cache entry alive while constructing dispatch tables, even if another worker
   * evicts it. The dispatch instance then owns its own PSO reference. Command buffers retain
   * resources already encoded for GPU execution. */
  const auto best_pipeline = MetalDeviceKernels::get_best_pipeline(metal_device, kernel);
  if (!best_pipeline) {
    if (!metal_device->have_error()) {
      metal_device->set_error(string_printf(
          "Failed to load required Metal kernel %s. See the Cycles log for the compiler error.",
          device_kernel_as_string(kernel)));
    }
    return false;
  }

  /* Pixel displacement functions specialized for the scene, once they are compiled and linked.
   * Until then the generic functions are used. */
  std::shared_ptr<MetalDisplacementFunctions> displacement;
  id<MTLComputePipelineState> displacement_pso = nil;
  /* Only for rendering: preprocessing kernels run with the evaluator set of an unfinished scene
   * update, which would specialize the functions for flags the render never uses. */
  if (best_pipeline->visible_functions && kernel >= DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA &&
      kernel <= DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL)
  {
    displacement = metal_device->specialized_displacement_functions(
        best_pipeline->visible_functions);
    if (displacement) {
      displacement_pso = best_pipeline->displacement_pipeline(*displacement);
      if (!displacement_pso) {
        displacement = nullptr;
      }
    }
  }
  const int new_displacement_key = displacement ? displacement->key() : -1;

  if (pipeline_id == best_pipeline->pipeline_id && displacement_key == new_displacement_key) {
    /* The best pipeline is already active - nothing to do. */
    return true;
  }
  pipeline_id = best_pipeline->pipeline_id;
  displacement_key = new_displacement_key;
  [pipeline release];
  pipeline = [(displacement_pso ? displacement_pso : best_pipeline->pipeline) retain];
  pso_type = best_pipeline->pso_type;
  num_threads_per_block = best_pipeline->num_threads_per_block;
  use_metalrt = best_pipeline->use_metalrt;

  /* Release tables from the previous pipeline before an ON/OFF scene switch.
   * A coherent surface pipeline has tables; an ordinary surface pipeline does not. */
  free_intersection_function_tables();
  free_visible_function_tables();

  /* Function handles are specific to the pipeline that linked the shading functions. */
  if (best_pipeline->visible_functions) {
    use_visible_shading = true;
    for (int table = 0; table < METAL_VFT_NUM; table++) {
      @autoreleasepool {
        NSArray<id<MTLFunction>> *functions = best_pipeline->visible_functions->table_functions[table];
        if (displacement && table == METAL_VFT_PIXEL_DISPLACEMENT_EVAL) {
          functions = @[ displacement->eval ];
        }
        else if (displacement && displacement->intersect &&
                 table == METAL_VFT_PIXEL_DISPLACEMENT_INTERSECT)
        {
          functions = @[ displacement->intersect ];
        }
        else if (displacement && displacement->scene_intersect &&
                 table == METAL_VFT_SCENE_INTERSECT)
        {
          functions = @[ displacement->scene_intersect ];
        }
        MTLVisibleFunctionTableDescriptor *vft_desc =
            [[[MTLVisibleFunctionTableDescriptor alloc] init] autorelease];
        vft_desc.functionCount = max(int(functions.count), 1);
        visible_func_table[table] = [pipeline newVisibleFunctionTableWithDescriptor:vft_desc];
        if (!visible_func_table[table]) {
          metal_device->set_error(string_printf("Failed to create Metal function table for %s",
                                                device_kernel_as_string(kernel)));
          return false;
        }
        for (int i = 0; i < int(functions.count); i++) {
          id<MTLFunctionHandle> handle = [pipeline functionHandleWithFunction:functions[i]];
          if (!handle) {
            metal_device->set_error(string_printf("Missing Metal function handle for %s",
                                                  device_kernel_as_string(kernel)));
            return false;
          }
          [visible_func_table[table] setFunction:handle atIndex:i];
        }
        metal_device->metal_mem_alloc(visible_func_table[table]);
      }
    }
  }

  /* Create the MTLIntersectionFunctionTables if needed. */
  if (best_pipeline->use_metalrt &&
      metal_kernel_has_intersection(best_pipeline->device_kernel, best_pipeline->kernel_features))
  {

    for (int table = 0; table < METALRT_TABLE_NUM; table++) {
      @autoreleasepool {
        MTLIntersectionFunctionTableDescriptor *ift_desc =
            [[[MTLIntersectionFunctionTableDescriptor alloc] init] autorelease];
        ift_desc.functionCount = best_pipeline->table_functions[table].count;
        intersection_func_table[table] = [this->pipeline
            newIntersectionFunctionTableWithDescriptor:ift_desc];

        /* Finally write the function handles into this pipeline's table */
        int size = int([best_pipeline->table_functions[table] count]);
        for (int i = 0; i < size; i++) {
          id table_function = best_pipeline->table_functions[table][i];
          if (table_function == [NSNull null]) {
            continue;
          }
          id<MTLFunctionHandle> handle = [pipeline
              functionHandleWithFunction:table_function];
          [intersection_func_table[table] setFunction:handle atIndex:i];
        }

        /* Bind launch_params into the intersection function table once, when the table is
         * (re)created. launch_params_buffer is allocated once and never moves, and the binding
         * persists on the table, so there's no need to rebind it on every dispatch. */
        [intersection_func_table[table] setBuffer:metal_device->launch_params_buffer
                                           offset:0
                                          atIndex:1];

        metal_device->metal_mem_alloc(intersection_func_table[table]);
      }
    }
  }

  return true;
}

id<MTLFunction> MetalKernelPipeline::make_intersection_function(const char *function_name)
{
  MTLFunctionDescriptor *desc = [MTLIntersectionFunctionDescriptor functionDescriptor];
  desc.name = @(function_name);

  const bool pixel_displacement_rays = metal_kernel_pixel_displacement_rays(device_kernel);
  if (pso_type != PSO_GENERIC) {
    desc.constantValues = GetConstantValues(&kernel_data_, pso_type, pixel_displacement_rays);
  }
  else {
    desc.constantValues = GetConstantValues(nullptr, PSO_GENERIC, pixel_displacement_rays);
  }

  NSError *error = nullptr;
  id<MTLFunction> rt_intersection_function = [mtlLibrary newFunctionWithDescriptor:desc
                                                                             error:&error];

  if (rt_intersection_function == nil) {
    NSString *err = [error localizedDescription];
    string errors = [err UTF8String];

    error_str = string_printf(
        "Error getting intersection function \"%s\": %s", function_name, errors.c_str());
  }
  else {
    rt_intersection_function.label = [@(function_name) copy];
  }
  return rt_intersection_function;
}

void MetalKernelPipeline::compile()
{
  /* Monolithic pipelines can require gigabytes each; compile one at a time on low-memory
   * devices. Pipelines linking the separately compiled shading functions are small. */
  thread_scoped_lock compilation_lock(metal_compilation_mutex(), std::defer_lock);
  if (MetalInfo::use_low_memory_compilation() && !complete_generic) {
    compilation_lock.lock();
  }

  const std::string function_name = std::string("cycles_metal_") +
                                    device_kernel_as_string(device_kernel);

  NSError *error = nullptr;

  MTLFunctionDescriptor *func_desc = [MTLIntersectionFunctionDescriptor functionDescriptor];
  func_desc.name = @(function_name.c_str());

  const bool pixel_displacement_rays = metal_kernel_pixel_displacement_rays(device_kernel);
  if (pso_type != PSO_GENERIC) {
    func_desc.constantValues = GetConstantValues(&kernel_data_, pso_type, pixel_displacement_rays);
  }
  else {
    func_desc.constantValues = GetConstantValues(nullptr, PSO_GENERIC, pixel_displacement_rays);
  }

  function = [mtlLibrary newFunctionWithDescriptor:func_desc error:&error];

  if (function == nil) {
    NSString *err = [error localizedDescription];
    string errors = [err UTF8String];
    metal_printf("Error getting function \"%s\": %s", function_name.c_str(), errors.c_str());
    return;
  }

  function.label = @(function_name.c_str());

  NSArray *linked_functions = nil;

  if (use_metalrt && metal_kernel_has_intersection(device_kernel, kernel_features)) {

    NSMutableSet *unique_functions = [[[NSMutableSet alloc] init] autorelease];
    bool required_intersection_function_missing = false;

    auto add_intersection_functions = [&](int table_index,
                                          const char *tri_fn,
                                          const char *curve_fn = nullptr,
                                          const char *point_fn = nullptr,
                                          const char *pixel_displacement_fn = nullptr) {
      const char *function_names[] = {tri_fn, curve_fn, point_fn, pixel_displacement_fn};
      int function_count = 4;
      while (function_count > 0 && function_names[function_count - 1] == nullptr) {
        function_count--;
      }

      NSMutableArray *functions = [NSMutableArray arrayWithCapacity:function_count];
      for (int i = 0; i < function_count; i++) {
        if (function_names[i]) {
          id<MTLFunction> intersection_function = make_intersection_function(function_names[i]);
          if (intersection_function) {
            [functions addObject:intersection_function];
            [unique_functions addObject:intersection_function];
            [intersection_function release];
          }
          else {
            /* Adaptive compilation can remove an intersection function for a geometry feature
             * absent from the scene. Preserve its table offset without inserting nil into the
             * NSMutableArray (which raises an Objective-C exception). */
            [functions addObject:[NSNull null]];
            const bool optional_scene_function =
                (i == 1 && !(kernel_features & KERNEL_FEATURE_HAIR)) ||
                (i == 2 && !(kernel_features & KERNEL_FEATURE_POINTCLOUD));
            required_intersection_function_missing |= !optional_scene_function;
          }
        }
        else {
          /* Keep the slot empty so geometry intersectionFunctionTableOffset values continue to
           * address the same function kind. NSArray's variadic constructor cannot represent this:
           * its first nil argument terminates the array and used to silently discard slot 3. */
          [functions addObject:[NSNull null]];
        }
      }
      /* The pipeline cache uses these tables after the worker's autorelease pool drains. */
      table_functions[table_index] = [functions retain];
    };

    add_intersection_functions(METALRT_TABLE_DEFAULT,
                               "__intersection__tri",
                               "__intersection__curve",
                               "__intersection__point",
                               "__intersection__pixel_displacement");

    const bool is_light_cache_kernel =
        device_kernel == DEVICE_KERNEL_INTEGRATOR_PHOTON_EMIT ||
        device_kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_LIGHT_GENERATE ||
        device_kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_SENSOR_CONNECT;
    if (is_light_cache_kernel) {
      /* These self-contained light-cache kernels use ordinary scene intersections plus volume
       * stack initialization. Linking the shadow, local and single-hit tables as well makes Metal
       * optimize several large, unreachable intersection call graphs into each pipeline. */
      add_intersection_functions(METALRT_TABLE_VOLUME, "__intersection__volume_tri");
    }
    else {
      add_intersection_functions(METALRT_TABLE_SHADOW,
                                 "__intersection__tri_shadow",
                                 "__intersection__curve_shadow",
                                 "__intersection__point_shadow",
                                 "__intersection__pixel_displacement_shadow");
      add_intersection_functions(METALRT_TABLE_SHADOW_ALL,
                                 "__intersection__tri_shadow_all",
                                 "__intersection__curve_shadow_all",
                                 "__intersection__point_shadow_all",
                                 "__intersection__pixel_displacement_shadow_all");
      add_intersection_functions(METALRT_TABLE_VOLUME, "__intersection__volume_tri");
      add_intersection_functions(METALRT_TABLE_LOCAL,
                                 "__intersection__local_tri",
                                 nullptr,
                                 nullptr,
                                 "__intersection__local_pixel_displacement");
      add_intersection_functions(METALRT_TABLE_LOCAL_MBLUR, "__intersection__local_tri_mblur");
      add_intersection_functions(METALRT_TABLE_LOCAL_SINGLE_HIT,
                                 "__intersection__local_tri_single_hit",
                                 nullptr,
                                 nullptr,
                                 "__intersection__local_pixel_displacement_single_hit");
      add_intersection_functions(METALRT_TABLE_LOCAL_SINGLE_HIT_MBLUR,
                                 "__intersection__local_tri_single_hit_mblur");
    }

    if (required_intersection_function_missing) {
      metal_printf("Required MetalRT intersection function is missing: %s", error_str.c_str());
      return;
    }

    for (const int table : {METALRT_TABLE_LOCAL, METALRT_TABLE_LOCAL_SINGLE_HIT}) {
      if (table_functions[table] &&
          (table_functions[table].count <= 3 || table_functions[table][3] == [NSNull null]))
      {
        metal_printf("MetalRT pixel-displacement local intersection table is incomplete");
        return;
      }
    }

    linked_functions = [[NSArray arrayWithArray:[unique_functions allObjects]]
        sortedArrayUsingComparator:^NSComparisonResult(id<MTLFunction> f1, id<MTLFunction> f2) {
          return [f1.label compare:f2.label];
        }];
    unique_functions = nil;
  }

  MTLComputePipelineDescriptor *computePipelineStateDescriptor =
      [[[MTLComputePipelineDescriptor alloc] init] autorelease];

  computePipelineStateDescriptor.buffers[0].mutability = MTLMutabilityImmutable;
  computePipelineStateDescriptor.buffers[1].mutability = MTLMutabilityImmutable;
  computePipelineStateDescriptor.buffers[2].mutability = MTLMutabilityImmutable;

  computePipelineStateDescriptor.maxTotalThreadsPerThreadgroup = threads_per_threadgroup;
  computePipelineStateDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;

  computePipelineStateDescriptor.computeFunction = function;

  /* Attach the additional functions to an MTLLinkedFunctions object */
  if (linked_functions) {
    computePipelineStateDescriptor.linkedFunctions = [[[MTLLinkedFunctions alloc] init]
        autorelease];
    computePipelineStateDescriptor.linkedFunctions.functions = linked_functions;
  }
  computePipelineStateDescriptor.maxCallStackDepth = 1;
  if (use_metalrt && metal_kernel_has_intersection(device_kernel, kernel_features)) {
    computePipelineStateDescriptor.maxCallStackDepth = 2;
  }
  if (visible_functions) {
    /* The shading functions compile concurrently with this pipeline and are added afterwards.
     * They call each other: the shader interpreter calls the shared node and closure functions
     * or traces rays through intersection functions, and surface kernels call the interpreter
     * from their separately compiled stages. The GPU reserves stack memory for every level, so
     * request exactly the depth that each kernel can reach. */
    computePipelineStateDescriptor.supportAddingBinaryFunctions = YES;
    computePipelineStateDescriptor.maxCallStackDepth = max(
        int(computePipelineStateDescriptor.maxCallStackDepth),
        metal_kernel_shading_call_depth(device_kernel, software_bvh_library));
  }

  MTLPipelineOption pipelineOptions = MTLPipelineOptionNone;

  bool use_binary_archive = should_use_binary_archive();
  bool loading_existing_archive = false;
  bool creating_new_archive = false;

  id<MTLBinaryArchive> archive = nil;
  string metalbin_path;
  string metalbin_name;
  if (use_binary_archive) {
    NSProcessInfo *processInfo = [NSProcessInfo processInfo];
    string osVersion = [[processInfo operatingSystemVersionString] UTF8String];
    MD5Hash local_md5;
    local_md5.append(kernels_md5);
    local_md5.append(osVersion);
    local_md5.append((uint8_t *)&this->threads_per_threadgroup,
                     sizeof(this->threads_per_threadgroup));

    /* Replace non-alphanumerical characters with underscores. */
    string device_name = [mtlDevice.name UTF8String];
    for (char &c : device_name) {
      if ((c < '0' || c > '9') && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z')) {
        c = '_';
      }
    }

    metalbin_name = device_name;
    metalbin_name = path_join(metalbin_name, device_kernel_as_string(device_kernel));
    metalbin_name = path_join(metalbin_name, kernel_type_as_string(pso_type));
    metalbin_name = path_join(metalbin_name, local_md5.get_hex() + ".bin");

    metalbin_path = path_cache_get(path_join("kernels", metalbin_name));
    path_create_directories(metalbin_path);

    /* Check if shader binary exists on disk, and if so, update the file timestamp for LRU purging
     * to work as intended. */
    loading_existing_archive = path_cache_kernel_exists_and_mark_used(metalbin_path);
    creating_new_archive = !loading_existing_archive;

    MTLBinaryArchiveDescriptor *archiveDesc = [[MTLBinaryArchiveDescriptor alloc] init];
    if (loading_existing_archive) {
      archiveDesc.url = [NSURL fileURLWithPath:@(metalbin_path.c_str())];
    }
    NSError *error = nil;
    archive = [mtlDevice newBinaryArchiveWithDescriptor:archiveDesc error:&error];
    [archive autorelease];
    if (!archive) {
      const char *err = error ? [[error localizedDescription] UTF8String] : nullptr;
      metal_printf("newBinaryArchiveWithDescriptor failed: %s", err ? err : "nil");
    }
    [archiveDesc release];

    if (loading_existing_archive) {
      pipelineOptions = MTLPipelineOptionFailOnBinaryArchiveMiss;
      computePipelineStateDescriptor.binaryArchives = [NSArray arrayWithObjects:archive, nil];
    }
  }

  bool recreate_archive = false;
  bool archive_complete = true;
  string compilation_error;

  /* Lambda to do the actual pipeline compilation. */
  auto do_compilation = [&]() {
    __block bool compilation_finished = false;
    __block string error_str;

    if (loading_existing_archive || !DebugFlags().metal.use_async_pso_creation) {
      /* Use the blocking variant of newComputePipelineStateWithDescriptor if an archive exists on
       * disk. It should load almost instantaneously, and will fail gracefully when loading a
       * corrupt archive (unlike the async variant). */
      NSError *error = nil;
      pipeline = [mtlDevice newComputePipelineStateWithDescriptor:computePipelineStateDescriptor
                                                          options:pipelineOptions
                                                       reflection:nullptr
                                                            error:&error];
      const char *err = error ? [[error localizedDescription] UTF8String] : nullptr;
      error_str = err ? err : "nil";
    }
    else {
      /* Use the async variant of newComputePipelineStateWithDescriptor if no archive exists on
       * disk. This allows us to respond to app shutdown. */
      [mtlDevice
          newComputePipelineStateWithDescriptor:computePipelineStateDescriptor
                                        options:pipelineOptions
                              completionHandler:^(id<MTLComputePipelineState> computePipelineState,
                                                  MTLComputePipelineReflection * /*reflection*/,
                                                  NSError *error) {
                                pipeline = computePipelineState;

                                /* Retain the pipeline so we can use it safely past the completion
                                 * handler. */
                                if (pipeline) {
                                  [pipeline retain];
                                }
                                const char *err = error ?
                                                      [[error localizedDescription] UTF8String] :
                                                      nullptr;
                                error_str = err ? err : "nil";

                                compilation_finished = true;
                              }];

      /* Immediately wait for either the compilation to finish or for app shutdown. */
      while (ShaderCache::running && !compilation_finished) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      /* Shutdown can stop this wait before the callback provides a result. Do not
       * read its unfinished error string or report cancellation as compilation failure. */
      if (!ShaderCache::running) {
        return;
      }
    }

    compilation_error = error_str;
    if (creating_new_archive && pipeline) {
      /* Add pipeline into the new archive. */
      NSError *error;
      if (![archive addComputePipelineFunctionsWithDescriptor:computePipelineStateDescriptor
                                                        error:&error])
      {
        /* Do not write an archive that would miss on every later load. */
        archive_complete = false;
        NSString *errStr = [error localizedDescription];
        metal_printf("Failed to add PSO to archive:\n%s", errStr ? [errStr UTF8String] : "nil");
      }
    }

    if (!pipeline) {
      metal_printf(
          "newComputePipelineStateWithDescriptor failed for \"%s\"%s. "
          "Error:\n%s\n",
          device_kernel_as_string(device_kernel),
          (archive && !recreate_archive) ? " Archive may be incomplete or corrupt - attempting "
                                           "recreation.." :
                                           "",
          compilation_error.c_str());
    }
  };

  double starttime = time_dt();

  do_compilation();
  if (!ShaderCache::running) {
    return;
  }

  /* An archive might have a corrupt entry and fail to materialize the pipeline. This shouldn't
   * happen, but if it does we recreate it. */
  if (pipeline == nil && archive) {
    recreate_archive = true;
    pipelineOptions = MTLPipelineOptionNone;
    path_remove(metalbin_path);

    do_compilation();
    if (!ShaderCache::running) {
      return;
    }
  }

  double duration = time_dt() - starttime;

  if (pipeline == nil) {
    LOG_ERROR << "Metal pipeline compilation failed for " << device_kernel_as_string(device_kernel)
              << " (" << kernel_type_as_string(pso_type) << "): " << compilation_error;
    metal_printf("%16s | %2d | %-55s | %7.2fs | FAILED!",
                 kernel_type_as_string(pso_type),
                 device_kernel,
                 device_kernel_as_string(device_kernel),
                 duration);
    return;
  }

  /* The shading functions are linked once they finished compiling, see
   * link_visible_functions(). */
  awaiting_link = visible_functions != nullptr;

  if (!num_threads_per_block && !awaiting_link) {
    num_threads_per_block = round_down(pipeline.maxTotalThreadsPerThreadgroup,
                                       pipeline.threadExecutionWidth);
    num_threads_per_block = std::max(num_threads_per_block, (int)pipeline.threadExecutionWidth);
  }

  if (ShaderCache::running) {
    if ((creating_new_archive || recreate_archive) && archive_complete) {
      if (![archive serializeToURL:[NSURL fileURLWithPath:@(metalbin_path.c_str())] error:&error])
      {
        metal_printf("Failed to save binary archive to %s, error:\n%s",
                     metalbin_path.c_str(),
                     [[error localizedDescription] UTF8String]);
      }
      else {
        /* Scene specialization produces a content-addressed archive variant. Keep enough variants
         * for normal production scene switching instead of cycling through the old five-entry
         * window and repeatedly recompiling unchanged kernels. */
        path_cache_kernel_mark_added_and_clear_old(metalbin_path, 32);
      }
    }
  }

  this->loaded = !awaiting_link;
  computePipelineStateDescriptor = nil;

  if (!use_binary_archive) {
    metal_printf("%16s | %2d | %-55s | %7.2fs",
                 kernel_type_as_string(pso_type),
                 int(device_kernel),
                 device_kernel_as_string(device_kernel),
                 duration);
  }
  else {
    metal_printf("%16s | %2d | %-55s | %7.2fs | %s: %s",
                 kernel_type_as_string(pso_type),
                 device_kernel,
                 device_kernel_as_string(device_kernel),
                 duration,
                 creating_new_archive ? " new" : "load",
                 metalbin_name.c_str());
  }
}

static int visible_function_threads(id<MTLDevice> mtlDevice)
{
  /* Compilation is throughput bound. Six threads peak below 5 GB of compiler memory, but keep
   * machines with less than 16 GB at four. */
  const int max_threads = ([NSProcessInfo processInfo].physicalMemory >= (16ull << 30)) ? 6 : 4;
  int num_threads = 4;
  if (@available(macOS 13.3, *)) {
    num_threads = std::clamp(
        int([mtlDevice maximumConcurrentCompilationTaskCount]) - 2, 2, max_threads);
  }
  if (const char *str = getenv("CYCLES_METAL_VISIBLE_FUNCTION_THREADS")) {
    num_threads = max(atoi(str), 1);
  }
  return num_threads;
}

std::shared_ptr<MetalVisibleFunctions> MetalDeviceKernels::request_visible_functions(
    id<MTLDevice> mtlDevice, id<MTLLibrary> library, const string &library_md5)
{
  const int num_threads = visible_function_threads(mtlDevice);
  ShaderCache *shader_cache = get_shader_cache(mtlDevice);
  std::shared_ptr<MetalVisibleFunctions> functions;
  {
    thread_scoped_lock lock(shader_cache->cache_mutex);
    auto &entry = shader_cache->visible_functions[library_md5];
    if (entry) {
      /* The functions may still compile in the background, see prewarm_visible_functions(). */
      entry->raise_threads(num_threads, shader_cache->visible_function_threads);
      return entry;
    }
    entry = std::make_shared<MetalVisibleFunctions>(mtlDevice, library, library_md5);
    functions = entry;
  }

  /* Compile concurrently with the kernel pipelines, which wait for these functions only when
   * linking them at the end of their own compilation. */
  {
    thread_scoped_lock lock(shader_cache->cache_mutex);
    shader_cache->visible_function_threads.emplace_back(
        [functions, num_threads]() { functions->compile(num_threads); });
  }
  return functions;
}

std::shared_ptr<MetalDisplacementFunctions> MetalDeviceKernels::request_displacement_functions(
    id<MTLDevice> mtlDevice,
    const std::shared_ptr<MetalVisibleFunctions> &generic,
    const int evaluator_set,
    const int bvh_features)
{
  ShaderCache *shader_cache = get_shader_cache(mtlDevice);
  thread_scoped_lock lock(shader_cache->cache_mutex);
  auto &entry =
      shader_cache->displacement_functions[{generic.get(), evaluator_set | (bvh_features << 24)}];
  if (!entry) {
    entry = std::make_shared<MetalDisplacementFunctions>(generic, evaluator_set, bvh_features);
  }
  if (!entry->ready() && !entry->failed && !entry->queued.exchange(true)) {
    entry->request_serial = ++g_displacement_request_serial;
    shader_cache->visible_function_threads.emplace_back(
        [functions = entry]() { functions->compile(); });
  }
  return entry;
}

void MetalDeviceKernels::prewarm_visible_functions(id<MTLDevice> mtlDevice,
                                                   const string &library_path,
                                                   const string &library_md5)
{
  ShaderCache *shader_cache = get_shader_cache(mtlDevice);
  thread_scoped_lock lock(shader_cache->cache_mutex);
  if (shader_cache->visible_functions.count(library_md5)) {
    return;
  }
  /* One thread, joined when the cache shuts down, waits and compiles in place. */
  shader_cache->visible_function_threads.emplace_back(
      [shader_cache, mtlDevice, library_path, library_md5]() {
        /* Start after the kernels of the current scene and their shading functions. */
        while (ShaderCache::running) {
          {
            thread_scoped_lock lock(shader_cache->cache_mutex);
            if (shader_cache->incomplete_scene_requests == 0 &&
                shader_cache->request_queue.empty())
            {
              break;
            }
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (!ShaderCache::running) {
          return;
        }
        @autoreleasepool {
          NSError *error = nil;
          id<MTLLibrary> library = [mtlDevice
              newLibraryWithURL:[NSURL fileURLWithPath:@(library_path.c_str())]
                          error:&error];
          if (!library) {
            return;
          }
          std::shared_ptr<MetalVisibleFunctions> functions;
          {
            thread_scoped_lock lock(shader_cache->cache_mutex);
            auto &entry = shader_cache->visible_functions[library_md5];
            if (!entry) {
              entry = std::make_shared<MetalVisibleFunctions>(mtlDevice, library, library_md5);
              functions = entry;
            }
          }
          [library release];
          if (functions) {
            metal_printf("Compiling shading functions of another library in the background");
            functions->compile(min(visible_function_threads(mtlDevice), 2));
          }
        }
      });
}

void MetalKernelPipeline::link_visible_functions()
{
  awaiting_link = false;
  /* Adding the finished binaries links them without recompiling the pipeline. */
  const double wait_start = time_dt();
  const bool functions_ready = visible_functions->wait();
  const double wait_duration = time_dt() - wait_start;
  id<MTLComputePipelineState> linked_pipeline = nil;
  NSError *link_error = nil;
  if (functions_ready && pipeline) {
    linked_pipeline = [pipeline
        newComputePipelineStateWithAdditionalBinaryFunctions:visible_functions->binary_functions
                                                       error:&link_error];
  }
  [pipeline release];
  pipeline = linked_pipeline;
  if (pipeline == nil) {
    if (ShaderCache::running) {
      LOG_ERROR << "Metal pipeline linking failed for " << device_kernel_as_string(device_kernel)
                << ": "
                << (link_error ? [[link_error localizedDescription] UTF8String] :
                                 "shading functions failed to compile");
    }
    return;
  }

  if (!num_threads_per_block) {
    num_threads_per_block = round_down(pipeline.maxTotalThreadsPerThreadgroup,
                                       pipeline.threadExecutionWidth);
    num_threads_per_block = std::max(num_threads_per_block, (int)pipeline.threadExecutionWidth);
  }
  loaded = true;
  metal_printf("%16s | %2d | %-55s | linked after waiting %.1fs",
               kernel_type_as_string(pso_type),
               int(device_kernel),
               device_kernel_as_string(device_kernel),
               wait_duration);
}

id<MTLLibrary> MetalDeviceKernels::find_generic_library(id<MTLDevice> mtlDevice,
                                                        const string &library_md5)
{
  ShaderCache *shader_cache = get_shader_cache(mtlDevice);
  thread_scoped_lock lock(shader_cache->cache_mutex);
  auto it = shader_cache->visible_functions.find(library_md5);
  return it != shader_cache->visible_functions.end() ? [it->second->library() retain] : nil;
}

bool MetalDeviceKernels::load(MetalDevice *device, MetalPipelineType pso_type)
{
  auto *shader_cache = get_shader_cache(device->mtlDevice);
  /* Request the most expensive kernels first, so they compile concurrently with each other and
   * with the shading functions instead of being left for last. */
  const DeviceKernel expensive_kernels[] = {DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE,
                                            DEVICE_KERNEL_INTEGRATOR_BDPT_LIGHT_GENERATE,
                                            DEVICE_KERNEL_INTEGRATOR_BDPT_SENSOR_CONNECT,
                                            DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE,
                                            DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE,
                                            DEVICE_KERNEL_INTEGRATOR_PHOTON_EMIT,
                                            DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME,
                                            DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME_RAY_MARCHING,
                                            DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST,
                                            DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW};
  for (const DeviceKernel kernel : expensive_kernels) {
    shader_cache->load_kernel(kernel, device, pso_type);
  }
  for (int i = 0; i < DEVICE_KERNEL_NUM; i++) {
    if (std::find(std::begin(expensive_kernels), std::end(expensive_kernels), DeviceKernel(i)) ==
        std::end(expensive_kernels))
    {
      shader_cache->load_kernel((DeviceKernel)i, device, pso_type);
    }
  }

  /* The complete generic library does not depend on the scene. Compile the kernels of features
   * the scene does not use yet behind the required ones, so enabling BDPT, photon mapping,
   * guiding, volumes and other features later needs no compilation. The caches keep them for
   * later sessions, so this runs once per Blender build. Rendering does not wait for them. */
  const char *prewarm_env = getenv("CYCLES_METAL_PREWARM_KERNELS");
  if (pso_type == PSO_GENERIC && device->use_visible_shading &&
      !(prewarm_env && atoi(prewarm_env) == 0))
  {
    for (const DeviceKernel kernel : expensive_kernels) {
      shader_cache->load_kernel(kernel, device, pso_type, true);
    }
    for (int i = 0; i < DEVICE_KERNEL_NUM; i++) {
      shader_cache->load_kernel((DeviceKernel)i, device, pso_type, true);
    }
    /* Pixel displacement without a MetalRT compatible cache needs the software BVH library. */
    device->prewarm_software_library();
  }
  return true;
}

void MetalDeviceKernels::wait_for_all()
{
  for (int i = 0; i < g_shaderCacheCount; i++) {
    g_shaderCache[i].second->wait_for_all();
  }
}

int MetalDeviceKernels::num_incomplete_specialization_requests()
{
  /* Return true if any ShaderCaches have ongoing specialization requests (typically there will be
   * only 1). */
  int total = 0;
  for (int i = 0; i < g_shaderCacheCount; i++) {
    total += g_shaderCache[i].second->incomplete_specialization_requests;
  }
  return total;
}

int MetalDeviceKernels::get_loaded_kernel_count(const MetalDevice *device,
                                                MetalPipelineType pso_type)
{
  auto *shader_cache = get_shader_cache(device->mtlDevice);
  int loaded_count = DEVICE_KERNEL_NUM;
  for (int i = 0; i < DEVICE_KERNEL_NUM; i++) {
    if (shader_cache->should_load_kernel((DeviceKernel)i, device, pso_type)) {
      loaded_count -= 1;
    }
  }
  return loaded_count;
}

bool MetalDeviceKernels::should_load_kernels(const MetalDevice *device, MetalPipelineType pso_type)
{
  return get_loaded_kernel_count(device, pso_type) != DEVICE_KERNEL_NUM;
}

std::shared_ptr<const MetalKernelPipeline> MetalDeviceKernels::get_best_pipeline(
    const MetalDevice *device, DeviceKernel kernel)
{
  return get_shader_cache(device->mtlDevice)->get_best_pipeline(kernel, device);
}

bool MetalDeviceKernels::is_benchmark_warmup()
{
  NSArray *args = [[NSProcessInfo processInfo] arguments];
  for (int i = 0; i < args.count; i++) {
    if (const char *arg = [[args objectAtIndex:i] cStringUsingEncoding:NSASCIIStringEncoding]) {
      /* Also recognize the spelling used by the Cycles scene benchmark runner.
       * Its discarded warmup must finish specialization before measured renders. */
      if (!strcmp(arg, "--warm-up") || !strcmp(arg, "--warmup")) {
        return true;
      }
    }
  }
  return false;
}

void MetalDeviceKernels::static_deinitialize()
{
  for (int i = 0; i < g_shaderCacheCount; i++) {
    g_shaderCache[i] = DeviceShaderCache();
  }
}

CCL_NAMESPACE_END

#endif /* WITH_METAL */
