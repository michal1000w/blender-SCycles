/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_CUDA

#  include "device/cuda/osl_camera.h"
#  include "device/cuda/device_impl.h"
#  include "device/cuda/osl_camera_compat.h"

#  include "kernel/device/cuda/globals.h"

#  include "util/log.h"
#  include "util/md5.h"
#  include "util/path.h"
#  include "util/string.h"
#  include "util/thread.h"
#  include "util/time.h"

CCL_NAMESPACE_BEGIN

/* The camera ray kernel of `kernel/device/gpu/kernel.h`, with the custom camera code of Metal
 * enabled. `metal_ancillaries->vft_osl_camera[0](...)` in `camera_sample_custom()` calls the
 * translated function, which is linked from its own compilation unit. */
static const char *camera_kernel_source = R"CUDA(
#define __KERNEL_METAL_OSL_CAMERA__
#include "kernel/device/cuda/compat.h"
#include "kernel/device/cuda/config.h"
#include "kernel/device/cuda/globals.h"
#include "kernel/device/gpu/image.h"

extern "C" __device__ void cycles_metal_osl_camera(const void *launch_params,
                                                   const unsigned int *params,
                                                   const float *input,
                                                   float *output);
struct CyclesOSLCameraCall {
  __device__ void operator()(const void *launch_params,
                             const unsigned int *params,
                             const float *input,
                             float *output) const
  {
    cycles_metal_osl_camera(launch_params, params, input, output);
  }
};
struct CyclesOSLCameraAncillaries {
  CyclesOSLCameraCall vft_osl_camera[1];
};
static __device__ const CyclesOSLCameraAncillaries cycles_osl_camera_ancillaries = {};
#define metal_ancillaries (&cycles_osl_camera_ancillaries)
#define launch_params_metal kernel_params

#include "kernel/tables.h"
#include "kernel/device/gpu/work_stealing.h"
#include "kernel/integrator/state.h"
#include "kernel/integrator/state_flow.h"
#include "kernel/integrator/state_util.h"
#include "kernel/util/image_3d.h"
#include "kernel/integrator/init_from_camera.h"

/* Same as in `kernel/device/gpu/kernel.h`. */
ccl_gpu_kernel(GPU_KERNEL_BLOCK_NUM_THREADS, GPU_KERNEL_MAX_REGISTERS)
    ccl_gpu_kernel_signature(integrator_init_from_camera,
                             ccl_global KernelWorkTile *tiles,
                             const int num_tiles,
                             ccl_global float *render_buffer,
                             const int max_tile_work_size,
                             const ccl_global int *path_index_array,
                             const int num_active_paths)
{
  const int work_index = ccl_gpu_global_id_x();

  if (tiles) {
    if (work_index >= max_tile_work_size * num_tiles) {
      return;
    }

    const int tile_index = work_index / max_tile_work_size;
    const int tile_work_index = work_index - tile_index * max_tile_work_size;

    const ccl_global KernelWorkTile *tile = &tiles[tile_index];

    if (tile_work_index >= tile->work_size) {
      return;
    }

    const int state = tile->path_index_offset + tile_work_index;

    uint x, y, sample;
    ccl_gpu_kernel_call(get_work_pixel(tile, tile_work_index, &x, &y, &sample));

    ccl_gpu_kernel_call(
        integrator_init_from_camera(nullptr, state, tile, render_buffer, x, y, sample));
  }
  else {
    if (work_index >= num_active_paths) {
      return;
    }

    const int state = (path_index_array) ? path_index_array[work_index] : work_index;

    ccl_gpu_kernel_call(
        integrator_init_from_camera(nullptr, state, nullptr, render_buffer, 0, 0, 0));
  }
}
ccl_gpu_kernel_postfix
)CUDA";

/* Headers that NVRTC cannot open, with replacements. */
static const char *system_headers[] = {"cstdlib",
                                       "cstdint",
                                       "cstdio",
                                       "cmath",
                                       "cstring",
                                       "cfloat",
                                       "climits",
                                       "stdint.h",
                                       "math.h",
                                       "float.h",
                                       "limits.h"};

/* Offsets of kernel data read by translated camera shaders, in floats from the start of the
 * kernel parameters, which the function receives as `launch_params`. As for Metal, see
 * `camera_function_kernel_offsets()` in `device/metal/kernel.mm`. */
static string camera_kernel_offsets()
{
  static const KernelParamsCUDA params = {};
  const KernelData &data = params.data;
  const auto define = [&](const char *name, const void *member) {
    const size_t offset = (const char *)member - (const char *)&params;
    assert(offset % sizeof(float) == 0);
    return string_printf("#define OSL_KD_%s %d\n", name, int(offset / sizeof(float)));
  };
  string defines;
  defines += define("CAM_SENSORWIDTH", &data.cam.sensorwidth);
  defines += define("CAM_SENSORHEIGHT", &data.cam.sensorheight);
  defines += define("CAM_WIDTH", &data.cam.width);
  defines += define("CAM_HEIGHT", &data.cam.height);
  defines += define("CAM_INV_APERTURE_RATIO", &data.cam.inv_aperture_ratio);
  defines += define("CAM_APERTURESIZE", &data.cam.aperturesize);
  defines += define("CAM_FOCALDISTANCE", &data.cam.focaldistance);
  defines += define("CAM_CAMERATOWORLD", &data.cam.cameratoworld);
  defines += define("CAM_WORLDTOCAMERA", &data.cam.worldtocamera);
  defines += define("CAM_SCREENTOWORLD", &data.cam.screentoworld);
  defines += define("CAM_RASTERTOWORLD", &data.cam.rastertoworld);
  defines += define("CAM_NDCTOWORLD", &data.cam.ndctoworld);
  defines += define("CAM_WORLDTOSCREEN", &data.cam.worldtoscreen);
  defines += define("CAM_WORLDTORASTER", &data.cam.worldtoraster);
  defines += define("CAM_WORLDTONDC", &data.cam.worldtondc);
  defines += define("SCENE_TIME", &data.scene_time.time);
  defines += define("SCENE_FRAME", &data.scene_time.frame);
  return defines;
}

static bool is_identifier_char(const char c)
{
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

/* The translated Metal source as CUDA, for the compatibility layer: no Metal standard library,
 * swizzles are functions, and the entry point has C linkage for the link with the kernel. */
static string camera_source_as_cuda(const string &msl)
{
  string source = msl;
  string_replace(source, "#include <metal_stdlib>\n", "");
  string_replace(source, "using namespace metal;\n", "");
  string_replace(source,
                 "[[visible]] void cycles_metal_osl_camera(",
                 "extern \"C\" void cycles_metal_osl_camera(");

  /* `.xyz` and `.xy` are the only swizzles that translated shaders and the prelude use, and only
   * to read. Generated names start with `s_`, so they cannot be members named like swizzles. */
  string result;
  result.reserve(source.size() + source.size() / 64);
  size_t i = 0;
  while (i < source.size()) {
    size_t swizzle_len = 0;
    if (source[i] == '.') {
      for (const size_t len : {size_t(4), size_t(3)}) {
        if (source.compare(i, len, string(".xyz").substr(0, len)) == 0 &&
            (i + len >= source.size() || !is_identifier_char(source[i + len])))
        {
          swizzle_len = len;
          break;
        }
      }
    }
    if (swizzle_len) {
      result.append(source, i, swizzle_len);
      result += "()";
      i += swizzle_len;
    }
    else {
      result += source[i++];
    }
  }
  return result;
}

static string join_options(const vector<string> &options)
{
  string joined;
  for (const string &option : options) {
    joined += option + " ";
  }
  return joined;
}

#  ifdef WITH_CUDA_DYNLOAD
/* NVRTC is loaded with the CUDA driver through cuew. */
static string nvrtc_error(const nvrtcResult result)
{
  const char *message = nvrtcGetErrorString ? nvrtcGetErrorString(result) : nullptr;
  return message ? message : string_printf("NVRTC error %d", int(result));
}

/* Compile `source` to PTX. On failure, `error` holds the first errors of the log. */
static bool nvrtc_compile(const string &source,
                          const char *name,
                          const vector<string> &options,
                          const vector<std::pair<string, const char *>> &headers,
                          string &ptx,
                          string &error)
{
  vector<const char *> header_names;
  vector<const char *> header_sources;
  for (const auto &header : headers) {
    header_names.push_back(header.first.c_str());
    header_sources.push_back(header.second);
  }
  nvrtcProgram program = nullptr;
  nvrtcResult result = nvrtcCreateProgram(&program,
                                          source.c_str(),
                                          name,
                                          int(headers.size()),
                                          header_sources.data(),
                                          header_names.data());
  if (result != NVRTC_SUCCESS) {
    error = "NVRTC could not create a program: " + nvrtc_error(result);
    return false;
  }

  vector<const char *> option_pointers;
  for (const string &option : options) {
    option_pointers.push_back(option.c_str());
  }
  result = nvrtcCompileProgram(program, int(option_pointers.size()), option_pointers.data());

  size_t log_size = 0;
  nvrtcGetProgramLogSize(program, &log_size);
  string log(log_size, '\0');
  if (log_size > 1) {
    nvrtcGetProgramLog(program, &log[0]);
    log.resize(log_size - 1);
  }

  if (result == NVRTC_SUCCESS) {
    size_t ptx_size = 0;
    nvrtcGetPTXSize(program, &ptx_size);
    ptx.assign(ptx_size, '\0');
    nvrtcGetPTX(program, &ptx[0]);
    /* Without the terminating null. */
    if (!ptx.empty() && ptx.back() == '\0') {
      ptx.pop_back();
    }
  }
  else {
    /* The errors, without the warnings. */
    string errors;
    vector<string> lines;
    string_split(lines, log, "\n", false);
    int count = 0;
    for (const string &line : lines) {
      if (line.find("error") != string::npos && count++ < 8) {
        errors += line + "\n";
      }
    }
    error = string_printf("NVRTC failed to compile %s (%s)", name, nvrtc_error(result).c_str());
    if (!errors.empty()) {
      error += ":\n" + errors;
    }
    LOG_WARNING << error;
  }
  nvrtcDestroyProgram(&program);
  return result == NVRTC_SUCCESS;
}

/* Register limit of the kernels of an architecture, see `kernel/device/cuda/config.h`. */
static int camera_kernel_max_registers(const int arch)
{
  return arch <= 69 ? 64 : 168;
}

/* Kernel sources do not change while Blender runs. */
static string kernel_sources_md5()
{
  static thread_mutex mutex;
  static string md5;
  thread_scoped_lock lock(mutex);
  if (md5.empty()) {
    md5 = path_files_md5_hash(path_get("source"));
  }
  return md5;
}

bool cuda_osl_camera_build(CUDADevice *device,
                           const string &camera_source,
                           const bool transport,
                           vector<uint8_t> &cubin,
                           string &error)
{
  /* One build at a time, they share the cache. */
  static thread_mutex build_mutex;
  thread_scoped_lock build_lock(build_mutex);

  if (cuewInit(CUEW_INIT_NVRTC) != CUEW_SUCCESS || nvrtcCreateProgram == nullptr) {
    error =
        "the NVRTC runtime compiler of the CUDA toolkit was not found (nvrtc64_*.dll or "
        "libnvrtc.so). It ships with Blender builds that have CUDA, or install the CUDA toolkit";
    return false;
  }
  int nvrtc_major = 0, nvrtc_minor = 0;
  nvrtcVersion(&nvrtc_major, &nvrtc_minor);
  int driver_version = 0;
  cuDriverGetVersion(&driver_version);

  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device->cuDevice);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device->cuDevice);
  /* Kernels support architectures up to 12.9, see `kernel/device/cuda/config.h`. Newer GPUs run
   * PTX of an older architecture. NVRTC may also not know the GPU, then try older ones. */
  vector<int> archs = {min(major * 10 + minor, 120), 90, 86, 75};

  const string source_path = path_get("source");
  const string source_md5 = kernel_sources_md5();
  const string camera_cuda = string(osl_camera_cuda_compat_1) + osl_camera_cuda_compat_2 +
                             camera_kernel_offsets() + camera_source_as_cuda(camera_source);

  vector<std::pair<string, const char *>> kernel_headers;
  for (const char *name : system_headers) {
    kernel_headers.emplace_back(name, osl_camera_cuda_system_stub);
  }
  kernel_headers.emplace_back("util/types_rgbe.h", osl_camera_cuda_types_rgbe);

  const double start_time = time_dt();
  for (size_t arch_index = 0; arch_index < archs.size(); arch_index++) {
    const int arch = archs[arch_index];
    if (arch_index > 0 && arch >= archs[0]) {
      continue;
    }
    const string arch_option = string_printf("-arch=compute_%d", arch);
    const string registers_option = string_printf("-maxrregcount=%d",
                                                  camera_kernel_max_registers(arch));
    /* The options of the precompiled kernels, see `kernel/device/cuda/CMakeLists.txt`. */
    vector<string> kernel_options = {arch_option,
                                     "-rdc=true",
                                     "-default-device",
                                     "-std=c++17",
                                     "--use_fast_math",
                                     "-w",
                                     registers_option,
                                     "-DCCL_NAMESPACE_BEGIN=",
                                     "-DCCL_NAMESPACE_END=",
                                     "-DNVCC",
                                     "-I" + source_path};
#  ifdef WITH_NANOVDB
    kernel_options.push_back("-DWITH_NANOVDB");
#  endif
    if (transport) {
      kernel_options.push_back("-D__KERNEL_GPU_TRANSPORT__");
    }
    /* Camera shaders keep IEEE division and square roots, like the OSL runtime of other
     * devices, since shaders test for infinities and NaN. */
    const vector<string> camera_options = {
        arch_option, "-rdc=true", "-default-device", "-std=c++17", "-w", registers_option};

    MD5Hash kernel_md5;
    kernel_md5.append(camera_kernel_source);
    kernel_md5.append(osl_camera_cuda_system_stub);
    kernel_md5.append(osl_camera_cuda_types_rgbe);
    kernel_md5.append(source_md5);
    kernel_md5.append(join_options(kernel_options));
    kernel_md5.append(string_printf("nvrtc %d.%d", nvrtc_major, nvrtc_minor));
    const string kernel_key = kernel_md5.get_hex();

    MD5Hash cubin_md5;
    cubin_md5.append(kernel_key);
    cubin_md5.append(camera_cuda);
    cubin_md5.append(join_options(camera_options));
    cubin_md5.append(string_printf("driver %d", driver_version));
    const string cubin_path = path_cache_get(
        path_join("kernels", "cycles_osl_camera_" + cubin_md5.get_hex() + ".cubin"));
    if (path_read_binary(cubin_path, cubin) && !cubin.empty()) {
      LOG_INFO << "Loaded custom camera kernel from cache " << cubin_path;
      return true;
    }

    /* The kernel does not depend on the camera shader. */
    const string kernel_ptx_path = path_cache_get(
        path_join("kernels", "cycles_osl_camera_kernel_" + kernel_key + ".ptx"));
    string kernel_ptx;
    if (!path_read_text(kernel_ptx_path, kernel_ptx) || kernel_ptx.empty()) {
      const double kernel_start = time_dt();
      string kernel_error;
      if (!nvrtc_compile(camera_kernel_source,
                         "osl_camera_kernel.cu",
                         kernel_options,
                         kernel_headers,
                         kernel_ptx,
                         kernel_error))
      {
        /* An architecture that this NVRTC does not know: try an older one. */
        if (kernel_error.find("INVALID_OPTION") != string::npos ||
            kernel_error.find("invalid option") != string::npos)
        {
          continue;
        }
        error = kernel_error;
        return false;
      }
      LOG_INFO << "Compiled custom camera ray kernel for compute_" << arch << " in "
               << time_dt() - kernel_start << " seconds";
      path_create_directories(kernel_ptx_path);
      path_write_text(kernel_ptx_path, kernel_ptx);
    }

    string camera_ptx;
    if (!nvrtc_compile(
            camera_cuda, "osl_camera.cu", camera_options, {}, camera_ptx, error))
    {
      return false;
    }

    /* Link both with the driver. Separately compiled functions need the register limit of the
     * kernel that calls them. */
    const CUDAContextScope scope(device);
    char log_buffer[8192] = "";
    CUjit_option link_options[] = {CU_JIT_ERROR_LOG_BUFFER,
                                   CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                                   CU_JIT_MAX_REGISTERS};
    void *link_values[] = {log_buffer,
                           (void *)(size_t)sizeof(log_buffer),
                           (void *)(size_t)camera_kernel_max_registers(arch)};
    CUlinkState link = nullptr;
    CUresult result = cuLinkCreate(3, link_options, link_values, &link);
    if (result == CUDA_SUCCESS) {
      result = cuLinkAddData(link,
                             CU_JIT_INPUT_PTX,
                             (void *)kernel_ptx.c_str(),
                             kernel_ptx.size() + 1,
                             "osl_camera_kernel.ptx",
                             0,
                             nullptr,
                             nullptr);
    }
    if (result == CUDA_SUCCESS) {
      result = cuLinkAddData(link,
                             CU_JIT_INPUT_PTX,
                             (void *)camera_ptx.c_str(),
                             camera_ptx.size() + 1,
                             "osl_camera.ptx",
                             0,
                             nullptr,
                             nullptr);
    }
    void *linked = nullptr;
    size_t linked_size = 0;
    if (result == CUDA_SUCCESS) {
      result = cuLinkComplete(link, &linked, &linked_size);
    }
    if (result == CUDA_SUCCESS) {
      cubin.assign((const uint8_t *)linked, (const uint8_t *)linked + linked_size);
    }
    if (link) {
      cuLinkDestroy(link);
    }
    if (result != CUDA_SUCCESS) {
      error = string_printf("linking the custom camera kernel failed (%s): %s",
                            cuewErrorString(result),
                            log_buffer);
      return false;
    }

    LOG_INFO << "Built custom camera kernel for compute_" << arch << " in "
             << time_dt() - start_time << " seconds";
    path_create_directories(cubin_path);
    path_write_binary(cubin_path, cubin);
    return true;
  }

  error = "NVRTC does not support this GPU";
  return false;
}
#  else
bool cuda_osl_camera_build(CUDADevice * /*device*/,
                           const string & /*camera_source*/,
                           const bool /*transport*/,
                           vector<uint8_t> & /*cubin*/,
                           string &error)
{
  error = "custom cameras need a build that loads CUDA dynamically (WITH_CUDA_DYNLOAD)";
  return false;
}
#  endif

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
