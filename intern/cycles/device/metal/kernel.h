/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_METAL

#  include "device/kernel.h"
#  include "kernel/features.h"

#  include "util/string.h"
#  include "util/thread.h"
#  include "util/vector.h"

#  include <Metal/Metal.h>
#  include <condition_variable>
#  include <memory>
#  include <map>
#  include <mutex>
#  include <thread>

CCL_NAMESPACE_BEGIN

class MetalDevice;

/* Coherent connections trace visibility rays inside ordinary surface shading.
 * Native point and curve primitives require their MetalRT intersection tables
 * there as well. Leave pipelines without coherent connections unchanged. */
static inline bool metal_kernel_has_intersection(const DeviceKernel kernel,
                                                 const uint64_t kernel_features)
{
  return device_kernel_has_intersection(kernel) ||
         (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE &&
          (kernel_features & KERNEL_FEATURE_COHERENT_SPECULAR));
}

/* Maximum nesting of shading function calls, including MetalRT intersection functions. The
 * GPU reserves call stack memory for every level. `software_bvh` selects the software BVH
 * library: its traversal is a separate function, which calls the displacement ray solver, two
 * levels deeper. */
static inline int metal_kernel_shading_call_depth(const DeviceKernel kernel,
                                                  const bool software_bvh)
{
  int depth = 2;
  /* Surface stages call the shader interpreter. Its raytrace nodes can evaluate the
   * displacement of local hits, which in turn calls the shared node functions. */
  if (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ||
      kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE)
  {
    depth = 4;
  }
  /* The manifold solver and the light-cache kernels evaluate the shader interpreter, which calls
   * the shared node functions or the displacement evaluator. */
  else if (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME ||
           kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME_RAY_MARCHING ||
           kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE ||
           kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_SENSOR_CONNECT ||
           kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_LIGHT_GENERATE ||
           kernel == DEVICE_KERNEL_INTEGRATOR_PHOTON_EMIT)
  {
    depth = 3;
  }
  return depth + (software_bvh ? 2 : 0);
}

/* Kernels that can evaluate shaders or closures and therefore call the separately compiled
 * shading functions. Only these link the functions: a pipeline able to make such calls gets a
 * call stack reservation for every thread, which the large thread groups of the sorting and
 * compaction kernels cannot afford. */
static inline bool metal_kernel_uses_shading_functions(const DeviceKernel kernel,
                                                       const bool software_bvh)
{
  /* Verified against the kernel call graph by tests/python/cycles_metal_visible_call_depth.py,
   * together with metal_kernel_shading_call_depth(). */
  return (kernel >= DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND &&
          kernel <= DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT) ||
         kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE ||
         kernel == DEVICE_KERNEL_INTEGRATOR_PHOTON_EMIT ||
         kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_LIGHT_GENERATE ||
         kernel == DEVICE_KERNEL_INTEGRATOR_BDPT_SENSOR_CONNECT ||
         (kernel >= DEVICE_KERNEL_SHADER_EVAL_DISPLACE &&
          kernel <= DEVICE_KERNEL_SHADER_EVAL_VOLUME_DENSITY) ||
         /* Local intersections evaluate the displaced normal of their hits. */
         kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE ||
         kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK ||
         /* The software BVH traversal is a separate function, which also evaluates displacement
          * while intersecting. */
         (software_bvh && kernel >= DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST &&
          kernel <= DEVICE_KERNEL_INTEGRATOR_INTERSECT_DEDICATED_LIGHT);
}

enum {
  METALRT_TABLE_DEFAULT,
  METALRT_TABLE_SHADOW,
  METALRT_TABLE_SHADOW_ALL,
  METALRT_TABLE_VOLUME,
  METALRT_TABLE_LOCAL,
  METALRT_TABLE_LOCAL_MBLUR,
  METALRT_TABLE_LOCAL_SINGLE_HIT,
  METALRT_TABLE_LOCAL_SINGLE_HIT_MBLUR,
  METALRT_TABLE_NUM
};

/* Pipeline State Object types */
enum MetalPipelineType {
  /* A kernel that can be used with all scenes, supporting all features.
   * It is slow to compile, but only needs to be compiled once and is then
   * cached for future render sessions. This allows a render to get underway
   * on the GPU quickly.
   */
  PSO_GENERIC,

  /* A intersection kernel that is very quick to specialize and results in faster intersection
   * kernel performance. It uses Metal function constants to replace several KernelData variables
   * with fixed constants.
   */
  PSO_SPECIALIZED_INTERSECT,

  /* A shading kernel that is slow to specialize, but results in faster shading kernel performance
   * rendered. It uses Metal function constants to replace several KernelData variables with fixed
   * constants and short-circuit all unused SVM node case handlers.
   */
  PSO_SPECIALIZED_SHADE,

  /* Scene-specialized light-cache kernels keep cold compilation bounded on small-memory GPUs. */
  PSO_SPECIALIZED_LIGHT_CACHE,

  /* Specialize only node usage; preparation runs before final KernelData is uploaded. */
  PSO_SPECIALIZED_EVAL,

  PSO_NUM
};

#  define METALRT_FEATURE_MASK \
    (KERNEL_FEATURE_HAIR | KERNEL_FEATURE_HAIR_THICK | KERNEL_FEATURE_POINTCLOUD | \
     KERNEL_FEATURE_COHERENT_SPECULAR | KERNEL_FEATURE_POLARIZATION)

#  define METAL_TRANSPORT_FEATURE_MASK \
    (KERNEL_FEATURE_BDPT | KERNEL_FEATURE_PHOTON_MAPPING | KERNEL_FEATURE_PATH_GUIDING | KERNEL_FEATURE_POLARIZATION)

const char *kernel_type_as_string(MetalPipelineType pso_type);

/* Tables of separately compiled shading functions, in their MetalAncillaries order. The kernel
 * defines the entries of each table, see `kernel/device/metal/kernel.metal`. */
enum MetalVisibleFunctionTable {
  METAL_VFT_SVM,
  METAL_VFT_SVM_NODE,
  METAL_VFT_SVM_CLOSURE,
  METAL_VFT_BSDF_EVAL,
  METAL_VFT_BSDF_EVAL_DELTA,
  METAL_VFT_BSDF_SAMPLE,
  METAL_VFT_SURFACE,
  METAL_VFT_POLARIZATION,
  METAL_VFT_DIFFRACTION,
  METAL_VFT_MNEE,
  METAL_VFT_PIXEL_DISPLACEMENT_EVAL,
  METAL_VFT_PIXEL_DISPLACEMENT_INTERSECT,
  METAL_VFT_SCENE_INTERSECT,
  METAL_VFT_NUM
};

/* GPU binaries of the separately compiled shading functions of one generic library. Compiling
 * the shader interpreter and closures once, instead of optimizing them again inside every
 * kernel, keeps cold compilation short and independent of the scene. The functions compile
 * concurrently with the kernel pipelines, which link them when both are ready. */
class MetalVisibleFunctions {
 public:
  MetalVisibleFunctions(id<MTLDevice> device, id<MTLLibrary> library, const string &library_md5);
  ~MetalVisibleFunctions();

  /* Compile every function on `num_threads` threads. Blocks until done. */
  void compile(int num_threads);
  /* Raise the number of compiler threads of a compilation in progress to `num_threads`, for
   * functions compiled in the background that a scene now waits for. Started threads are
   * appended to `threads`. */
  void raise_threads(int num_threads, std::vector<std::thread> &threads);
  /* Wait for compile() to finish. Returns false if any function failed to compile. */
  bool wait();

  /* Functions of each table, ordered by table index. */
  NSArray<id<MTLFunction>> *table_functions[METAL_VFT_NUM] = {nil};
  /* All functions, for linking into pipelines. */
  NSArray<id<MTLFunction>> *binary_functions = nil;

  id<MTLLibrary> library() const
  {
    return library_;
  }

  /* Compile one of the functions with pixel displacement specialized for a scene, see
   * MetalDisplacementFunctions. */
  id<MTLFunction> compile_specialized(const string &name,
                                      int evaluator_set,
                                      int bvh_features,
                                      string &error);

 private:
  /* A negative `evaluator_set` compiles the generic function. */
  id<MTLFunction> compile_function(const string &name,
                                   string &error,
                                   int evaluator_set = -1,
                                   int bvh_features = 0);
  API_AVAILABLE(macos(13.0))
  id<MTLFunction> compile_function_macos13(const string &name,
                                           string &error,
                                           int evaluator_set,
                                           int bvh_features);

  /* Compile jobs until none are left. */
  void run_worker();

  id<MTLDevice> device_ = nil;
  id<MTLLibrary> library_ = nil;
  string library_md5_;
  vector<string> names_[METAL_VFT_NUM];

  /* Jobs of compile(), in scheduling order, and their results by table and index. */
  struct Job {
    MetalVisibleFunctionTable table;
    int index;
  };
  vector<Job> jobs_;
  vector<vector<id<MTLFunction>>> results_;
  std::atomic_int next_job_ = 0;
  std::atomic_bool failed_ = false;
  string first_error_;

  std::mutex mutex_;
  std::condition_variable cond_;
  /* Worker threads, guarded by `mutex_`. */
  int num_threads_ = 0;
  int requested_threads_ = 0;
  int active_workers_ = 0;
  bool started_ = false;
  bool finished_ = false;
  bool success_ = false;
};

/* Pixel displacement functions of a generic library, specialized for the evaluator set and BVH
 * traversal variant of a scene. The generic functions evaluate every kind of displacement
 * shader through separate function calls; specialized, the evaluator and ray solver compile
 * into the traversal, which makes displaced rendering about twice as fast. They compile in the
 * background, the generic functions are used until they are ready. */
class MetalDisplacementFunctions {
 public:
  MetalDisplacementFunctions(std::shared_ptr<MetalVisibleFunctions> generic,
                             int evaluator_set,
                             int bvh_features);
  ~MetalDisplacementFunctions();

  /* Compile the functions once the generic functions are ready. Blocks until done. Only one
   * specialization compiles at a time, and one superseded by a newer request is skipped. */
  void compile();
  bool ready() const
  {
    return ready_;
  }

  /* Order of the latest request for these functions among all requests. */
  std::atomic<int64_t> request_serial = 0;
  /* A compile() call is queued or running. */
  std::atomic_bool queued = false;
  /* Compilation failed, do not retry. */
  std::atomic_bool failed = false;

  const std::shared_ptr<MetalVisibleFunctions> generic;
  const int evaluator_set;
  /* Bit 0 for object motion, bit 1 for curves. */
  const int bvh_features;
  /* Identifies the specialization. */
  int key() const
  {
    return evaluator_set | (bvh_features << 24);
  }

  id<MTLFunction> eval = nil;
  /* Software BVH library only. */
  id<MTLFunction> intersect = nil;
  id<MTLFunction> scene_intersect = nil;

 private:
  std::atomic_bool ready_ = false;
};

/* A pipeline object that can be shared between multiple instances of MetalDeviceQueue. */
class MetalKernelPipeline {
 public:
  ~MetalKernelPipeline();
  void compile();

  int pipeline_id;
  int originating_device_id;

  id<MTLLibrary> mtlLibrary = nil;
  MetalPipelineType pso_type;
  string kernels_md5;
  size_t usage_count = 0;

  KernelData kernel_data_;
  bool use_metalrt;
  uint64_t kernel_features = 0;

  int threads_per_threadgroup;

  DeviceKernel device_kernel;
  bool loaded = false;
  id<MTLDevice> mtlDevice = nil;
  id<MTLFunction> function = nil;
  id<MTLComputePipelineState> pipeline = nil;
  int num_threads_per_block = 0;

  /* Separately compiled shading functions linked into this pipeline, if its library uses them. */
  std::shared_ptr<MetalVisibleFunctions> visible_functions;
  /* compile() finished, the shading functions still have to be linked. */
  bool awaiting_link = false;
  /* Requested ahead of time for a feature the scene does not use yet. */
  bool prewarm = false;
  /* Compiled from the complete generic library with separately compiled shading functions.
   * Such pipelines are small and need no low-memory serialization. */
  bool complete_generic = false;
  /* From the software BVH variant of the complete generic library. */
  bool software_bvh_library = false;
  void link_visible_functions();

  /* The linked pipeline with specialized displacement functions added, or nil if adding them
   * failed. Created on first use for each evaluator set. */
  id<MTLComputePipelineState> displacement_pipeline(
      const MetalDisplacementFunctions &functions) const;

  bool should_use_binary_archive() const;
  id<MTLFunction> make_intersection_function(const char *function_name);

  string error_str;

  NSArray *table_functions[METALRT_TABLE_NUM] = {nil};

 private:
  mutable thread_mutex displacement_mutex_;
  mutable std::map<int, id<MTLComputePipelineState>> displacement_pipelines_;
};

/* An actively instanced pipeline that can only be used by a single instance of MetalDeviceQueue.
 */
class MetalDispatchPipeline {
 public:
  ~MetalDispatchPipeline();

  bool update(MetalDevice *metal_device, DeviceKernel kernel);
  void free_intersection_function_tables();
  void free_visible_function_tables();

 private:
  friend class MetalDeviceQueue;
  friend struct ShaderCache;

  int pipeline_id = -1;
  /* Key of the specialized displacement functions in use, or -1 for the generic. */
  int displacement_key = -1;

  MetalDevice *metal_device = nullptr;
  MetalPipelineType pso_type;
  id<MTLComputePipelineState> pipeline = nil;
  int num_threads_per_block = 0;
  bool use_metalrt = false;

  API_AVAILABLE(macos(11.0))
  id<MTLIntersectionFunctionTable> intersection_func_table[METALRT_TABLE_NUM] = {nil};

  bool use_visible_shading = false;
  API_AVAILABLE(macos(11.0))
  id<MTLVisibleFunctionTable> visible_func_table[METAL_VFT_NUM] = {nil};
};

/* Cache of Metal kernels for each DeviceKernel. */
namespace MetalDeviceKernels {

int num_incomplete_specialization_requests();
int get_loaded_kernel_count(const MetalDevice *device, MetalPipelineType pso_type);
bool should_load_kernels(const MetalDevice *device, MetalPipelineType pso_type);
bool load(MetalDevice *device, MetalPipelineType pso_type);
/* Separately compiled shading functions of a generic library, shared by all devices using the
 * same library. The first request starts their compilation in the background. */
std::shared_ptr<MetalVisibleFunctions> request_visible_functions(id<MTLDevice> mtlDevice,
                                                                 id<MTLLibrary> library,
                                                                 const string &library_md5);
/* Pixel displacement functions specialized for `evaluator_set`, compiling in the background
 * when first requested. */
std::shared_ptr<MetalDisplacementFunctions> request_displacement_functions(
    id<MTLDevice> mtlDevice,
    const std::shared_ptr<MetalVisibleFunctions> &generic,
    int evaluator_set,
    int bvh_features);
/* Compile the shading functions of another generic library on few threads, after the kernels
 * that scenes wait for. request_visible_functions() for the same library then adds threads to
 * a compilation in progress, or returns the finished functions. */
void prewarm_visible_functions(id<MTLDevice> mtlDevice,
                               const string &library_path,
                               const string &library_md5);
/* A generic library with the given source checksum loaded earlier, retained, or nil. */
id<MTLLibrary> find_generic_library(id<MTLDevice> mtlDevice, const string &library_md5);
std::shared_ptr<const MetalKernelPipeline> get_best_pipeline(const MetalDevice *device,
                                                           DeviceKernel kernel);
void wait_for_all();
bool is_benchmark_warmup();

/* Deinitialize all static variables, so that no code would run on application exit. */
void static_deinitialize();

} /* namespace MetalDeviceKernels */

CCL_NAMESPACE_END

#endif /* WITH_METAL */
