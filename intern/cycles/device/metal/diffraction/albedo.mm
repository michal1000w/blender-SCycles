/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */
#ifdef WITH_METAL
#include "device/metal/device_impl.h"
#include "scene/diffraction_albedo.h"
#include "kernel/util/dielectric_dispersion.h"
#include "kernel/util/dielectric_f0_cache.h"
#include "util/path.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

CCL_NAMESPACE_BEGIN

/* Owned by one device; scene preparation serializes construction and use.
 * ARC releases all Metal objects when the device's shared owner is destroyed. */
class MetalDiffractionAlbedoBuilder {
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLComputePipelineState> build_, average_, two_sided_, thin_sheet_;
  id<MTLBuffer> context_;
  double compilation_seconds_ = 0.0;
  double two_sided_dispatch_wait_seconds_ = 0.0;

  static void completed(id<MTLCommandBuffer> command)
  {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      throw std::runtime_error(command.error ? command.error.localizedDescription.UTF8String :
                                             "Metal albedo command failed");
    }
  }

 public:
  explicit MetalDiffractionAlbedoBuilder(id<MTLDevice> device) : device_(device)
  {
    const auto start = std::chrono::steady_clock::now();
    if (!device_) throw std::runtime_error("Metal albedo device unavailable");
    queue_ = [device_ newCommandQueue];
    if (!queue_) throw std::runtime_error("Metal albedo queue allocation failed");
    const string source = path_source_replace_includes(
        "#include \"kernel/device/metal/diffraction_albedo.metal\"\n", path_get("source"));
    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
    else options.fastMathEnabled = NO;
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(source.c_str())
                                                   options:options error:&error];
    if (!library) throw std::runtime_error(error ? error.localizedDescription.UTF8String : "Metal albedo library compilation failed");
    auto pipeline = [&](NSString *name) {
      NSError *pipeline_error = nil;
      id<MTLComputePipelineState> result = [device_ newComputePipelineStateWithFunction:
          [library newFunctionWithName:name] error:&pipeline_error];
      if (!result) throw std::runtime_error(pipeline_error ? pipeline_error.localizedDescription.UTF8String : "Metal albedo pipeline compilation failed");
      return result;
    };
    build_ = pipeline(@"diffraction_multiscatter_albedo");
    average_ = pipeline(@"diffraction_multiscatter_average");
    two_sided_ = pipeline(@"diffraction_two_sided_albedo");
    thin_sheet_ = pipeline(@"diffraction_thin_sheet_albedo");
    id<MTLComputePipelineState> size_pipeline = pipeline(@"diffraction_multiscatter_context_size");
    id<MTLBuffer> size = [device_ newBufferWithLength:sizeof(unsigned)
                                               options:MTLResourceStorageModeShared];
    if (!size) throw std::runtime_error("Metal albedo context allocation failed");
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    if (!command) throw std::runtime_error("Metal albedo command allocation failed");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) throw std::runtime_error("Metal albedo encoder allocation failed");
    [encoder setComputePipelineState:size_pipeline];
    [encoder setBuffer:size offset:0 atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    completed(command);
    const unsigned bytes = *(const unsigned *)size.contents;
    if (!bytes || bytes > 16 * 1024 * 1024) throw std::runtime_error("Invalid Metal albedo context size");
    context_ = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!context_) throw std::runtime_error("Metal albedo context allocation failed");
    memset(context_.contents, 0, bytes);
    compilation_seconds_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
  }

  double compilation_seconds() const { return compilation_seconds_; }
  double two_sided_dispatch_wait_seconds() const { return two_sided_dispatch_wait_seconds_; }

  void build(const DiffractionAlbedoRequest &request, DiffractionAlbedoTable &table)
  {
    const unsigned mu_count = request.mu_count, phi_count = request.phi_count;
    const unsigned count = mu_count * phi_count, samples = request.facet_samples;
    std::vector<float4> directions;
    directions.reserve(count);
    for (unsigned m = 0; m < mu_count; ++m) {
      const float x = float(m) / (mu_count - 1);
      const float mu = std::max(1e-5f, x * x), radius = std::sqrt(1 - mu * mu);
      for (unsigned j = 0; j < phi_count; ++j) {
        const float phi = M_2PI_F * (j + .5f) / phi_count;
        directions.push_back(make_float4(radius * std::cos(phi), radius * std::sin(phi), mu, 0));
      }
    }
    std::vector<float> profiles;
    const size_t profile_count=request.wavelength_count;
    for (int i = 0; i < request.wavelength_count; ++i) {
      const float wavelength = request.wavelength_min_nm +
          (request.wavelength_max_nm - request.wavelength_min_nm) * i / (request.wavelength_count - 1);
      if(request.thin_sheet) {
        const float n=dielectric_ior_at_wavelength(request.inside_ior,request.inv_abbe,
                                                   dielectric_wavelength_um(wavelength));
        const float alpha_t=std::clamp(request.alpha_x*std::sqrt(
            3.4f*(n-1.0f)*(n-.5f)*(n-.5f)/(n*n*n)),0.0f,1.0f);
        const float p[]={request.alpha_x,alpha_t,wavelength/request.pitch_nm,
                         2.0f*M_2PI_F*request.depth_nm/wavelength,
                         M_2PI_F*(n-1.0f)*request.depth_nm/wavelength,request.duty,n,
                         request.reflection_tint,
                         diffraction_albedo_thin_sheet_transmission_tint(request,wavelength),
                         request.film_ior,request.film_thickness_nm/wavelength};
        profiles.insert(profiles.end(),p,p+11);
      }
      else {
        const float p[]={request.alpha_x,request.alpha_y,
                         wavelength/(request.medium_ior*request.pitch_nm),
                         request.depth_nm*request.medium_ior/wavelength,request.duty};
        profiles.insert(profiles.end(),p,p+5);
      }
    }
    auto buffer = [&](const void *data, size_t bytes) {
      id<MTLBuffer> result = data ? [device_ newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared] :
                                  [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      if (!result) throw std::runtime_error("Metal albedo table allocation failed");
      return result;
    };
    id<MTLBuffer> input = buffer(directions.data(), directions.size() * sizeof(float4));
    id<MTLBuffer> output = buffer(nullptr, profile_count * count * sizeof(float));
    id<MTLBuffer> parameters = buffer(profiles.data(), profiles.size() * sizeof(float));
    id<MTLBuffer> averages = buffer(nullptr, profile_count * sizeof(float));
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    if (!command) throw std::runtime_error("Metal albedo command allocation failed");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) throw std::runtime_error("Metal albedo encoder allocation failed");
    const id<MTLComputePipelineState> builder=request.thin_sheet ? thin_sheet_ : build_;
    [encoder setComputePipelineState:builder];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:context_ offset:0 atIndex:2];
    [encoder setBuffer:parameters offset:0 atIndex:3];
    [encoder setBytes:&samples length:sizeof(samples) atIndex:4];
    [encoder setBytes:&count length:sizeof(count) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(count, profile_count, 1)
        threadsPerThreadgroup:MTLSizeMake(builder.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    if (!encoder) throw std::runtime_error("Metal albedo average encoder allocation failed");
    const unsigned shape[2] = {mu_count, phi_count};
    [encoder setComputePipelineState:average_];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBuffer:averages offset:0 atIndex:1];
    [encoder setBytes:shape length:sizeof(shape) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(profile_count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(average_.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];
    completed(command);
    table.request = request;
    const float *values = (const float *)output.contents;
    table.values.assign(values, values + profile_count * count);
    values = (const float *)averages.contents;
    table.averages.assign(values, values + profile_count);
    diffraction_albedo_symmetrize_thin_sheet(table);
  }

  void build(const DiffractionTwoSidedAlbedoRequest &request,
             DiffractionTwoSidedAlbedoTable &table)
  {
    const unsigned mu_count = request.mu_count, phi_count = request.phi_count;
    const unsigned direction_count = mu_count * phi_count;
    const unsigned samples = request.facet_samples;
    std::vector<float4> directions;
    directions.reserve(direction_count);
    for (unsigned m = 0; m < mu_count; ++m) {
      const float x = float(m) / float(mu_count - 1);
      const float mu = std::max(1e-5f, x * x);
      const float radial = std::sqrt(std::max(0.0f, 1.0f - mu * mu));
      for (unsigned j = 0; j < phi_count; ++j) {
        const float phi = M_2PI_F * (float(j) + 0.5f) / float(phi_count);
        directions.push_back(make_float4(radial * std::cos(phi), radial * std::sin(phi), mu, 0));
      }
    }
    const int basis_count = request.generalized_f0_count == 0 ? 1 : request.generalized_f0_count;
    std::vector<std::array<float, 11>> profiles;
    profiles.reserve(basis_count * 2 * request.wavelength_count);
    for (int basis = 0; basis < basis_count; ++basis) {
      const float f0 = request.generalized_f0_count == 0 ?
                           -1.0f : dielectric_f0_cache_node(request.inside_ior, basis_count, basis);
      for (unsigned side = 0; side < 2; ++side) {
        for (int w = 0; w < request.wavelength_count; ++w) {
          const float wavelength = 380.0f + 400.0f * float(w) /
                                                float(request.wavelength_count - 1);
          const float inside_ior = dielectric_ior_at_wavelength(
              request.inside_ior, request.inv_abbe, dielectric_wavelength_um(wavelength));
          const float ni = side == 0 ? 1.0f : inside_ior;
          const float nt = side == 0 ? inside_ior : 1.0f;
          profiles.push_back({request.alpha_x, request.alpha_y, request.pitch_nm,
                              request.depth_nm, request.duty, ni, nt, wavelength,
                              request.film_ior, request.film_thickness_nm, f0});
        }
      }
    }
    static_assert(sizeof(profiles[0]) == 11 * sizeof(float));
    auto buffer = [&](const void *data, const size_t bytes) {
      id<MTLBuffer> result = data ? [device_ newBufferWithBytes:data length:bytes
                                 options:MTLResourceStorageModeShared] :
                                   [device_ newBufferWithLength:bytes
                                 options:MTLResourceStorageModeShared];
      if (!result) throw std::runtime_error("Metal two-sided albedo buffer allocation failed");
      return result;
    };
    id<MTLBuffer> input = buffer(directions.data(), directions.size() * sizeof(float4));
    id<MTLBuffer> output = buffer(nullptr, profiles.size() * direction_count * sizeof(float2));
    id<MTLBuffer> parameters = buffer(profiles.data(), profiles.size() * sizeof(profiles[0]));
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    if (!command) throw std::runtime_error("Metal two-sided albedo command allocation failed");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) throw std::runtime_error("Metal two-sided albedo encoder allocation failed");
    [encoder setComputePipelineState:two_sided_];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:context_ offset:0 atIndex:2];
    [encoder setBuffer:parameters offset:0 atIndex:3];
    [encoder setBytes:&samples length:sizeof(samples) atIndex:4];
    [encoder setBytes:&direction_count length:sizeof(direction_count) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(direction_count, profiles.size(), 1)
            threadsPerThreadgroup:MTLSizeMake(two_sided_.threadExecutionWidth, 1, 1)];
    [encoder endEncoding];
    const auto dispatch_start = std::chrono::steady_clock::now();
    completed(command);
    two_sided_dispatch_wait_seconds_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - dispatch_start).count();
    table.request = request;
    const float2 *escaped = (const float2 *)output.contents;
    const size_t count = profiles.size() * direction_count;
    table.reflectance.resize(count);
    table.transmittance.resize(count);
    for (size_t i = 0; i < count; ++i) {
      table.reflectance[i] = escaped[i].x;
      table.transmittance[i] = escaped[i].y;
    }
  }
};

bool MetalDevice::build_diffraction_albedo(const DiffractionAlbedoRequest &request,
                                          DiffractionAlbedoTable &table,
                                          string &error,
                                          const std::function<bool()> &cancelled)
{
  if (!diffraction_albedo_validate_request(request, error)) return false;
  auto is_cancelled = [&]() {
    if (cancelled && cancelled()) {
      error = "Diffraction albedo construction cancelled";
      return true;
    }
    return false;
  };
  if (is_cancelled()) return false;
  @autoreleasepool {
    try {
      if (!diffraction_albedo_builder) {
        auto builder = std::make_shared<MetalDiffractionAlbedoBuilder>(mtlDevice);
        if (is_cancelled()) return false;
        diffraction_albedo_builder = std::move(builder);
      }
      DiffractionAlbedoTable candidate;
      diffraction_albedo_builder->build(request, candidate);
      if (is_cancelled() || !diffraction_albedo_validate_table(candidate, error)) return false;
      table = std::move(candidate);
      error.clear();
      return true;
    }
    catch (const std::exception &exception) {
      error = exception.what();
      return false;
    }
  }
}

bool MetalDevice::build_diffraction_two_sided_albedo(
    const DiffractionTwoSidedAlbedoRequest &request,
    DiffractionTwoSidedAlbedoTable &table,
    string &error,
    const std::function<bool()> &cancelled)
{
  if (!diffraction_two_sided_albedo_validate_request(request, error)) return false;
  auto is_cancelled = [&]() {
    if (cancelled && cancelled()) {
      error = "Two-sided diffraction albedo construction cancelled";
      return true;
    }
    return false;
  };
  if (is_cancelled()) return false;
  @autoreleasepool {
    try {
      bool compiled_now = false;
      if (!diffraction_albedo_builder) {
        auto builder = std::make_shared<MetalDiffractionAlbedoBuilder>(mtlDevice);
        if (is_cancelled()) return false;
        diffraction_albedo_builder = std::move(builder);
        compiled_now = true;
      }
      DiffractionTwoSidedAlbedoTable candidate;
      diffraction_albedo_builder->build(request, candidate);
      if (is_cancelled() ||
          !diffraction_two_sided_albedo_complete_from_directional(candidate, error))
      {
        return false;
      }
      if (std::getenv("CYCLES_DIFFRACTION_CACHE_PROFILE")) {
        std::fprintf(stderr,
                     "DIFFRACTION_TWO_SIDED_GPU_TIMING compile_s=%.9f dispatch_wait_s=%.9f "
                     "cold_compile=%d wavelengths=%d\n",
                     compiled_now ? diffraction_albedo_builder->compilation_seconds() : 0.0,
                     diffraction_albedo_builder->two_sided_dispatch_wait_seconds(),
                     int(compiled_now), request.wavelength_count);
      }
      table = std::move(candidate);
      error.clear();
      return true;
    }
    catch (const std::exception &exception) {
      error = exception.what();
      return false;
    }
  }
}
CCL_NAMESPACE_END
#endif
