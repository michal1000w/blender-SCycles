/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_METAL

#  include "device/metal/metalfx.h"

#  include "device/metal/device_impl.h"
#  include "device/metal/util.h"

#  include "util/log.h"

#  include <Metal/Metal.h>
#  include <MetalFX/MetalFX.h>
#  include <simd/simd.h>

CCL_NAMESPACE_BEGIN

/* The denoiser was added to MetalFX in macOS 26. */
#  define METALFX_AVAILABLE @available(macOS 26.0, *)

namespace {

/* Pixel formats of the textures handed to MetalFX. Color is half float as in the applications
 * MetalFX is made for, values are clamped to its range. Normals need a signed format. */
const MTLPixelFormat FORMAT_COLOR = MTLPixelFormatRGBA16Float;
const MTLPixelFormat FORMAT_DEPTH = MTLPixelFormatR32Float;
const MTLPixelFormat FORMAT_MOTION = MTLPixelFormatRG16Float;
const MTLPixelFormat FORMAT_ALBEDO = MTLPixelFormatRGBA16Float;
const MTLPixelFormat FORMAT_NORMAL = MTLPixelFormatRGBA16Float;
const MTLPixelFormat FORMAT_ROUGHNESS = MTLPixelFormatR16Float;
const MTLPixelFormat FORMAT_OUTPUT = MTLPixelFormatRGBA16Float;

/* Number of floats kept per pixel to remember the color sum and sample count of the render
 * buffer at the previous call, see #MetalFXFrame::accumulate. */
const int PREVIOUS_STRIDE = 4;

/* Parameters of the conversion kernels. Only 4 byte members, so that the layout is the same in
 * C++ and in the Metal shading language. */
struct ConvertParams {
  int width;
  int height;
  int full_x;
  int full_y;
  int offset;
  int stride;
  int pass_stride;

  int pass_color;
  int pass_depth;
  int pass_albedo;
  int pass_normal;
  int pass_roughness;
  int pass_motion;
  int pass_motion_weight;
  int pass_sample_count;
  int num_samples;

  int out_width;
  int out_height;
  int out_full_x;
  int out_full_y;
  int out_offset;
  int out_stride;
  int pass_noisy;
  int pass_denoised;
  int num_components;
  int use_compositing;
  float upscale_factor;

  /* Use the previous state of the render buffer, and whether that state is valid. */
  int accumulate;
  int previous_valid;
  /* Motion vectors are valid for this frame. */
  int use_motion;
  /* The albedo pass applies to the pass that is denoised. */
  int use_albedo;
  /* Size of the output texture of MetalFX, which can be a few pixels smaller than the output. */
  int scaler_out_width;
  int scaler_out_height;

  /* Clip space depth is (depth_a * z + depth_b) / (depth_c * z + depth_d). */
  float depth_a;
  float depth_b;
  float depth_c;
  float depth_d;

  /* Rotation from the camera space of Cycles to world space, rows. */
  float rotation[9];

  /* Diagnostics: show this input instead of the result, see #show_input. */
  int show_input;
};

const char *convert_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct ConvertParams {
  int width;
  int height;
  int full_x;
  int full_y;
  int offset;
  int stride;
  int pass_stride;

  int pass_color;
  int pass_depth;
  int pass_albedo;
  int pass_normal;
  int pass_roughness;
  int pass_motion;
  int pass_motion_weight;
  int pass_sample_count;
  int num_samples;

  int out_width;
  int out_height;
  int out_full_x;
  int out_full_y;
  int out_offset;
  int out_stride;
  int pass_noisy;
  int pass_denoised;
  int num_components;
  int use_compositing;
  float upscale_factor;

  int accumulate;
  int previous_valid;
  int use_motion;
  int use_albedo;
  int scaler_out_width;
  int scaler_out_height;

  float depth_a;
  float depth_b;
  float depth_c;
  float depth_d;

  float rotation[9];

  /* Diagnostics: show this input instead of the result, see #show_input. */
  int show_input;
};

#define PREVIOUS_STRIDE 4
#define HALF_MAX 65504.0f
#define ROUGHNESS_SCALE 0.5f

#define SHOW_COLOR 1
#define SHOW_DEPTH 2
#define SHOW_MOTION 3
#define SHOW_ALBEDO 4
#define SHOW_NORMAL 5
#define SHOW_ROUGHNESS 6

static inline float finite_or_zero(float value)
{
  return (isnan(value) || isinf(value)) ? 0.0f : value;
}

static inline float3 finite_or_zero(float3 value)
{
  return float3(finite_or_zero(value.x), finite_or_zero(value.y), finite_or_zero(value.z));
}

/* Render buffer to the input textures of MetalFX. Texture rows run top to bottom, the render
 * buffer bottom to top. */
kernel void metalfx_pack(texture2d<float, access::write> color_texture [[texture(0)]],
                         texture2d<float, access::write> depth_texture [[texture(1)]],
                         texture2d<float, access::write> motion_texture [[texture(2)]],
                         texture2d<float, access::write> albedo_texture [[texture(3)]],
                         texture2d<float, access::write> specular_albedo_texture [[texture(4)]],
                         texture2d<float, access::write> normal_texture [[texture(5)]],
                         texture2d<float, access::write> roughness_texture [[texture(6)]],
                         const device float *render_buffer [[buffer(0)]],
                         device float *previous_buffer [[buffer(1)]],
                         constant ConvertParams &params [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(params.width) || gid.y >= uint(params.height)) {
    return;
  }

  const int x = int(gid.x);
  const int y = params.height - 1 - int(gid.y);

  const ulong pixel_index = ulong(params.offset) + ulong(x + params.full_x) +
                            ulong(y + params.full_y) * ulong(params.stride);
  const device float *buffer = render_buffer + pixel_index * ulong(params.pass_stride);

  float num_samples = float(params.num_samples);
  if (params.pass_sample_count >= 0) {
    num_samples = float(as_type<uint>(buffer[params.pass_sample_count]));
  }

  /* Sums over all samples in the buffer. */
  float3 color = float3(0.0f);
  if (params.pass_color >= 0) {
    color = float3(buffer[params.pass_color + 0],
                   buffer[params.pass_color + 1],
                   buffer[params.pass_color + 2]) *
            num_samples;
  }
  float depth = (params.pass_depth >= 0) ? buffer[params.pass_depth] : 0.0f;
  float3 albedo = float3(0.0f);
  if (params.pass_albedo >= 0) {
    albedo = float3(buffer[params.pass_albedo + 0],
                    buffer[params.pass_albedo + 1],
                    buffer[params.pass_albedo + 2]);
  }
  float3 normal = float3(0.0f);
  if (params.pass_normal >= 0) {
    normal = float3(buffer[params.pass_normal + 0],
                    buffer[params.pass_normal + 1],
                    buffer[params.pass_normal + 2]);
  }
  float roughness = (params.pass_roughness >= 0) ? buffer[params.pass_roughness] : 0.0f;

  color = finite_or_zero(color);
  depth = finite_or_zero(depth);
  albedo = finite_or_zero(albedo);
  normal = finite_or_zero(normal);
  roughness = finite_or_zero(roughness);

  /* Number of samples the color of the frame for MetalFX consists of. */
  float frame_samples = num_samples;

  if (params.accumulate) {
    /* The color is what was added to the buffer since the previous call: an independent noisy
     * frame, as MetalFX expects. The features are kept as the average of all samples, they are
     * the same for every frame and the less noise they have the better. */
    device float *previous = previous_buffer +
                             (ulong(x) + ulong(y) * ulong(params.width)) * PREVIOUS_STRIDE;

    float3 previous_color = float3(0.0f);
    float previous_samples = 0.0f;
    if (params.previous_valid) {
      previous_color = float3(previous[0], previous[1], previous[2]);
      previous_samples = previous[3];
    }

    frame_samples = num_samples - previous_samples;
    if (frame_samples <= 0.0f && params.previous_valid) {
      /* Nothing was added to this pixel, as with adaptive sampling once it has converged. The
       * textures still hold the previous frame, which is the best estimate there is. */
      return;
    }

    previous[0] = color.x;
    previous[1] = color.y;
    previous[2] = color.z;
    previous[3] = num_samples;

    color -= previous_color;
  }

  const float color_scale = (frame_samples > 0.0f) ? 1.0f / frame_samples : 0.0f;
  const float scale = (num_samples > 0.0f) ? 1.0f / num_samples : 0.0f;

  color = clamp(color * color_scale, 0.0f, HALF_MAX);
  color_texture.write(float4(color, 1.0f), gid);

  /* Depth: the pass holds the distance along the view axis, the far clipping distance for the
   * background. Behind sharp reflections and refractions it continues to what is seen in them,
   * like the other features. Convert to the reversed clip space depth of the projection. */
  float view_z = depth * scale;
  float clip_depth = 0.0f;
  if (view_z > 0.0f) {
    const float clip_w = params.depth_c * view_z + params.depth_d;
    if (fabs(clip_w) > 1e-20f) {
      clip_depth = clamp((params.depth_a * view_z + params.depth_b) / clip_w, 0.0f, 1.0f);
    }
  }
  depth_texture.write(float4(clip_depth, 0.0f, 0.0f, 0.0f), gid);

  /* Motion: the pass holds the offset in pixels to the position in the previous frame. */
  float2 motion = float2(0.0f);
  if (params.pass_motion >= 0 && params.use_motion) {
    float motion_weight = num_samples;
    if (params.pass_motion_weight >= 0) {
      motion_weight = buffer[params.pass_motion_weight];
    }
    if (motion_weight > 0.0f) {
      motion = float2(buffer[params.pass_motion + 0], buffer[params.pass_motion + 1]) /
               motion_weight;
      motion = float2(finite_or_zero(motion.x), -finite_or_zero(motion.y));
      motion = clamp(motion, -HALF_MAX, HALF_MAX);
    }
  }
  motion_texture.write(float4(motion, 0.0f, 0.0f), gid);

  /* Albedo above one happens for emission, bring it into range. */
  albedo = max(albedo * scale, 0.0f);
  albedo = clamp(albedo / (1.0f + albedo), 0.0f, 1.0f);
  if (!params.use_albedo) {
    /* Otherwise the textures of the surfaces show up in the result. */
    albedo = float3(0.5f);
  }
  albedo_texture.write(float4(albedo, 1.0f), gid);

  /* The albedo pass of Cycles is the albedo of everything that is not a sharp reflection or
   * refraction, and continues behind those. MetalFX keeps the detail seen in mirrors and glass
   * with that as its diffuse albedo and no specular albedo. With the specular albedo of the
   * first surface it blurs them, and is less accurate everywhere else. */
  specular_albedo_texture.write(float4(0.0f, 0.0f, 0.0f, 1.0f), gid);

  /* Normal: camera space in the pass, world space for MetalFX. */
  const float3 normal_world = float3(dot(float3(params.rotation[0], params.rotation[1], params.rotation[2]), normal),
                                     dot(float3(params.rotation[3], params.rotation[4], params.rotation[5]), normal),
                                     dot(float3(params.rotation[6], params.rotation[7], params.rotation[8]), normal));
  const float normal_length = length(normal_world);
  const float3 normal_out = (normal_length > 1e-8f) ? normal_world / normal_length : float3(0.0f);
  normal_texture.write(float4(normal_out, 0.0f), gid);

  /* Roughness: the pass averages over all closures, where diffuse ones count as fully rough.
   * MetalFX denoises much less at a roughness near one, which the materials it was made for
   * rarely have, and best around the middle of the range. */
  roughness = clamp(roughness * scale, 0.0f, 1.0f) * ROUGHNESS_SCALE;
  roughness_texture.write(float4(roughness, 0.0f, 0.0f, 0.0f), gid);
}

/* Output texture of MetalFX to the denoised pass of the render buffer. */
kernel void metalfx_unpack(texture2d<float, access::read> output_texture [[texture(0)]],
                           texture2d<float, access::read> input_texture [[texture(1)]],
                           device float *render_buffer [[buffer(0)]],
                           constant ConvertParams &params [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(params.out_width) || gid.y >= uint(params.out_height)) {
    return;
  }

  const int x = int(gid.x);
  const int y = params.out_height - 1 - int(gid.y);

  /* The rendered pixel this output pixel came from. */
  const int render_x = min(int(float(x) / params.upscale_factor), params.width - 1);
  const int render_y = min(int(float(y) / params.upscale_factor), params.height - 1);
  const ulong render_pixel_index = ulong(params.offset) + ulong(render_x + params.full_x) +
                                   ulong(render_y + params.full_y) * ulong(params.stride);
  const device float *buffer = render_buffer + render_pixel_index * ulong(params.pass_stride);

  float pixel_scale = float(params.num_samples);
  if (params.pass_sample_count >= 0) {
    pixel_scale = float(as_type<uint>(buffer[params.pass_sample_count]));
  }

  /* Alpha comes from the noisy pass. Read before writing: without upscaling this is the same
   * pixel as the one written below. */
  float alpha = 0.0f;
  if (params.num_components != 3 && !params.use_compositing) {
    alpha = buffer[params.pass_noisy + 3];
    if (params.pass_sample_count >= 0 && params.upscale_factor != 1.0f) {
      alpha /= max(pixel_scale, 1.0f);
    }
  }

  const ulong denoised_pixel_index = ulong(params.out_offset) + ulong(x + params.out_full_x) +
                                     ulong(y + params.out_full_y) * ulong(params.out_stride);
  device float *denoised = render_buffer + denoised_pixel_index * ulong(params.pass_stride) +
                           params.pass_denoised;

  const uint2 output_coord = min(
      gid, uint2(uint(params.scaler_out_width - 1), uint(params.scaler_out_height - 1)));
  float3 color = finite_or_zero(output_texture.read(output_coord).xyz);
  color = max(color, 0.0f);

  if (params.show_input) {
    const float4 value = input_texture.read(
        uint2(uint(render_x), uint(params.height - 1 - render_y)));
    if (params.show_input == SHOW_MOTION) {
      color = float3(0.5f) + float3(value.xy, 0.0f) * 0.05f;
    }
    else if (params.show_input == SHOW_NORMAL) {
      color = value.xyz * 0.5f + 0.5f;
    }
    else if (params.show_input == SHOW_DEPTH || params.show_input == SHOW_ROUGHNESS) {
      color = float3(value.x);
    }
    else {
      color = value.xyz;
    }
  }

  /* The passes are stored multiplied by the number of samples. With upscaling and a per pixel
   * sample count there is no count at the output resolution, and the display does not use it. */
  if (params.pass_sample_count < 0 || params.upscale_factor == 1.0f) {
    color *= pixel_scale;
  }

  denoised[0] = color.x;
  denoised[1] = color.y;
  denoised[2] = color.z;
  if (params.num_components != 3) {
    denoised[3] = alpha;
  }
}
)MSL";

/* CYCLES_METALFX_SHOW=color|depth|motion|albedo|normal|roughness replaces the denoised result by
 * that input, as MetalFX gets it. For checking what the denoiser works with. */
int show_input()
{
  static const int show = []() {
    const char *env = getenv("CYCLES_METALFX_SHOW");
    if (!env) {
      return 0;
    }
    const char *names[] = {"", "color", "depth", "motion", "albedo", "normal", "roughness"};
    for (int i = 1; i < 7; i++) {
      if (strcmp(env, names[i]) == 0) {
        return i;
      }
    }
    return 0;
  }();
  return show;
}

void matrix_to_simd(const float m[16], simd_float4x4 &r)
{
  for (int row = 0; row < 4; row++) {
    for (int column = 0; column < 4; column++) {
      r.columns[column][row] = m[row * 4 + column];
    }
  }
}

id<MTLDevice> device_from_info(const DeviceInfo &info)
{
  const vector<id<MTLDevice>> &devices = MetalInfo::get_usable_devices();
  if (info.type != DEVICE_METAL || info.num < 0 || size_t(info.num) >= devices.size()) {
    return nil;
  }
  return devices[info.num];
}

}  // namespace

struct MetalFXDenoiserContext::Impl {
  MetalDevice *metal_device = nullptr;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;

  id<MTLComputePipelineState> pack_pipeline = nil;
  id<MTLComputePipelineState> unpack_pipeline = nil;

  /* id<MTLFXTemporalDenoisedScaler>, typed where it is used because of availability. */
  id scaler = nil;

  id<MTLTexture> color = nil;
  id<MTLTexture> depth = nil;
  id<MTLTexture> motion = nil;
  id<MTLTexture> albedo = nil;
  id<MTLTexture> specular_albedo = nil;
  id<MTLTexture> normal = nil;
  id<MTLTexture> roughness = nil;
  id<MTLTexture> output = nil;

  /* State of the render buffer at the previous call, for accumulating renders. */
  id<MTLBuffer> previous = nil;
  bool previous_valid = false;
  int previous_num_samples = 0;

  int width = 0;
  int height = 0;
  int out_width = 0;
  int out_height = 0;
  /* Size MetalFX scales to. The render resolution is the output resolution divided by the
   * upscale factor and rounded down, which at the largest factor the device supports ends up
   * just above that factor: MetalFX then leaves out the last pixels, which are extended from
   * their neighbors. */
  int scaler_out_width = 0;
  int scaler_out_height = 0;

  bool need_history_reset = true;

  bool compile(string &error);
  id<MTLTexture> create_texture(MTLPixelFormat format,
                                int texture_width,
                                int texture_height,
                                MTLTextureUsage usage,
                                MTLStorageMode storage,
                                NSString *label);
  void dispatch(id<MTLCommandBuffer> command_buffer,
                id<MTLComputePipelineState> pipeline,
                const ConvertParams &params,
                id<MTLBuffer> render_buffer,
                int grid_width,
                int grid_height,
                bool pack);
  void free_scaler();
};

bool MetalFXDenoiserContext::Impl::compile(string &error)
{
  if (pack_pipeline && unpack_pipeline) {
    return true;
  }

  NSError *ns_error = nil;
  MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
  id<MTLLibrary> library = [device newLibraryWithSource:@(convert_source)
                                                options:options
                                                  error:&ns_error];
  if (!library) {
    error = string("MetalFX: failed to compile the conversion kernels: ") +
            (ns_error ? [[ns_error localizedDescription] UTF8String] : "unknown error");
    return false;
  }

  id<MTLFunction> pack_function = [library newFunctionWithName:@"metalfx_pack"];
  id<MTLFunction> unpack_function = [library newFunctionWithName:@"metalfx_unpack"];
  if (pack_function) {
    pack_pipeline = [device newComputePipelineStateWithFunction:pack_function error:&ns_error];
  }
  if (unpack_function) {
    unpack_pipeline = [device newComputePipelineStateWithFunction:unpack_function
                                                             error:&ns_error];
  }
  if (!pack_pipeline || !unpack_pipeline) {
    error = string("MetalFX: failed to create the conversion pipelines: ") +
            (ns_error ? [[ns_error localizedDescription] UTF8String] : "unknown error");
    pack_pipeline = nil;
    unpack_pipeline = nil;
    return false;
  }
  return true;
}

id<MTLTexture> MetalFXDenoiserContext::Impl::create_texture(MTLPixelFormat format,
                                                            const int texture_width,
                                                            const int texture_height,
                                                            MTLTextureUsage usage,
                                                            MTLStorageMode storage,
                                                            NSString *label)
{
  MTLTextureDescriptor *descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:texture_width
                                                        height:texture_height
                                                     mipmapped:NO];
  descriptor.usage = usage;
  descriptor.storageMode = storage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  texture.label = label;
  if (texture) {
    metal_device->stats.mem_alloc(texture.allocatedSize);
  }
  return texture;
}

void MetalFXDenoiserContext::Impl::free_scaler()
{
  for (id<MTLTexture> texture :
       {color, depth, motion, albedo, specular_albedo, normal, roughness, output})
  {
    if (texture) {
      metal_device->stats.mem_free(texture.allocatedSize);
    }
  }
  if (previous) {
    metal_device->stats.mem_free(previous.allocatedSize);
  }

  scaler = nil;
  color = nil;
  depth = nil;
  motion = nil;
  albedo = nil;
  specular_albedo = nil;
  normal = nil;
  roughness = nil;
  output = nil;
  previous = nil;
  previous_valid = false;
  previous_num_samples = 0;
  width = 0;
  height = 0;
  out_width = 0;
  out_height = 0;
  scaler_out_width = 0;
  scaler_out_height = 0;
  need_history_reset = true;
}

void MetalFXDenoiserContext::Impl::dispatch(id<MTLCommandBuffer> command_buffer,
                                            id<MTLComputePipelineState> pipeline,
                                            const ConvertParams &params,
                                            id<MTLBuffer> render_buffer,
                                            const int grid_width,
                                            const int grid_height,
                                            const bool pack)
{
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  encoder.label = pack ? @"Cycles MetalFX pack" : @"Cycles MetalFX unpack";
  [encoder setComputePipelineState:pipeline];
  if (pack) {
    [encoder setTexture:color atIndex:0];
    [encoder setTexture:depth atIndex:1];
    [encoder setTexture:motion atIndex:2];
    [encoder setTexture:albedo atIndex:3];
    [encoder setTexture:specular_albedo atIndex:4];
    [encoder setTexture:normal atIndex:5];
    [encoder setTexture:roughness atIndex:6];
    [encoder setBuffer:previous offset:0 atIndex:1];
  }
  else {
    id<MTLTexture> inputs[] = {color, color, depth, motion, albedo, normal, roughness};
    [encoder setTexture:output atIndex:0];
    [encoder setTexture:inputs[show_input()] atIndex:1];
  }
  [encoder setBuffer:render_buffer offset:0 atIndex:0];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];

  const NSUInteger group_width = pipeline.threadExecutionWidth;
  const NSUInteger group_height = max(pipeline.maxTotalThreadsPerThreadgroup / group_width,
                                      NSUInteger(1));
  [encoder dispatchThreads:MTLSizeMake(grid_width, grid_height, 1)
      threadsPerThreadgroup:MTLSizeMake(group_width, group_height, 1)];
  [encoder endEncoding];
}

bool MetalFXDenoiserContext::is_device_supported(const DeviceInfo &info)
{
  @autoreleasepool {
    if (METALFX_AVAILABLE) {
      id<MTLDevice> device = device_from_info(info);
      return device && [MTLFXTemporalDenoisedScalerDescriptor supportsDevice:device];
    }
    return false;
  }
}

void MetalFXDenoiserContext::supported_scale_range(const DeviceInfo &info,
                                                   float &min_scale,
                                                   float &max_scale)
{
  min_scale = 1.0f;
  max_scale = 1.0f;
  @autoreleasepool {
    if (METALFX_AVAILABLE) {
      id<MTLDevice> device = device_from_info(info);
      if (device) {
        min_scale = [MTLFXTemporalDenoisedScalerDescriptor
            supportedInputContentMinScaleForDevice:device];
        max_scale = [MTLFXTemporalDenoisedScalerDescriptor
            supportedInputContentMaxScaleForDevice:device];
      }
    }
  }
}

MetalFXDenoiserContext::MetalFXDenoiserContext(Device *device) : impl_(make_unique<Impl>())
{
  impl_->metal_device = static_cast<MetalDevice *>(device);
  impl_->device = impl_->metal_device->mtlDevice;
  impl_->queue = impl_->metal_device->mtlComputeCommandQueue;
}

MetalFXDenoiserContext::~MetalFXDenoiserContext()
{
  free();
}

void MetalFXDenoiserContext::free()
{
  @autoreleasepool {
    impl_->free_scaler();
  }
}

bool MetalFXDenoiserContext::ensure(const int width,
                                    const int height,
                                    const int out_width,
                                    const int out_height,
                                    string &error)
{
  Impl &impl = *impl_;

  if (impl.scaler && impl.width == width && impl.height == height &&
      impl.out_width == out_width && impl.out_height == out_height)
  {
    return true;
  }

  @autoreleasepool {
    impl.free_scaler();

    if (width <= 0 || height <= 0 || out_width < width || out_height < height) {
      return false;
    }

    if (METALFX_AVAILABLE) {
      if (![MTLFXTemporalDenoisedScalerDescriptor supportsDevice:impl.device]) {
        error = "MetalFX denoising is not supported by this GPU";
        return false;
      }

      /* Sizes outside of what the device supports are not an error: there is no denoising for
       * them, as for viewports that are too small. */
      const float max_scale = max(
          [MTLFXTemporalDenoisedScalerDescriptor supportedInputContentMaxScaleForDevice:impl.device],
          1.0f);
      const int scaler_out_width = min(out_width, int(floorf(float(width) * max_scale)));
      const int scaler_out_height = min(out_height, int(floorf(float(height) * max_scale)));
      /* A few pixels are extended, more than that is a scale that is not supported. */
      if (out_width - scaler_out_width > 4 || out_height - scaler_out_height > 4) {
        LOG_WARNING << "MetalFX: scale from " << width << "x" << height << " to " << out_width
                    << "x" << out_height << " is not supported";
        return false;
      }

      if (!impl.compile(error)) {
        return false;
      }

      MTLFXTemporalDenoisedScalerDescriptor *descriptor =
          [[MTLFXTemporalDenoisedScalerDescriptor alloc] init];
      descriptor.colorTextureFormat = FORMAT_COLOR;
      descriptor.depthTextureFormat = FORMAT_DEPTH;
      descriptor.motionTextureFormat = FORMAT_MOTION;
      descriptor.diffuseAlbedoTextureFormat = FORMAT_ALBEDO;
      descriptor.specularAlbedoTextureFormat = FORMAT_ALBEDO;
      descriptor.normalTextureFormat = FORMAT_NORMAL;
      descriptor.roughnessTextureFormat = FORMAT_ROUGHNESS;
      descriptor.outputTextureFormat = FORMAT_OUTPUT;
      descriptor.inputWidth = width;
      descriptor.inputHeight = height;
      descriptor.outputWidth = scaler_out_width;
      descriptor.outputHeight = scaler_out_height;
      /* Renders have any range of radiance. MetalFX works on the color multiplied by an
       * exposure, and is clearly more accurate when it picks that itself than without one. */
      descriptor.autoExposureEnabled = YES;
      /* The result is the same either way, but without this the first frames are slow. */
      descriptor.requiresSynchronousInitialization = YES;

      id<MTLFXTemporalDenoisedScaler> scaler = nil;
      @try {
        scaler = [descriptor newTemporalDenoisedScalerWithDevice:impl.device];
      }
      @catch (NSException *exception) {
        LOG_WARNING << "MetalFX: " << [[exception reason] UTF8String];
        scaler = nil;
      }
      if (!scaler) {
        LOG_WARNING << "MetalFX: no denoiser for " << width << "x" << height << " to "
                    << out_width << "x" << out_height;
        return false;
      }

      const MTLTextureUsage write = MTLTextureUsageShaderWrite;
      impl.color = impl.create_texture(FORMAT_COLOR,
                                       width,
                                       height,
                                       scaler.colorTextureUsage | write,
                                       MTLStorageModePrivate,
                                       @"Cycles MetalFX color");
      impl.depth = impl.create_texture(FORMAT_DEPTH,
                                       width,
                                       height,
                                       scaler.depthTextureUsage | write,
                                       MTLStorageModePrivate,
                                       @"Cycles MetalFX depth");
      impl.motion = impl.create_texture(FORMAT_MOTION,
                                        width,
                                        height,
                                        scaler.motionTextureUsage | write,
                                        MTLStorageModePrivate,
                                        @"Cycles MetalFX motion");
      impl.albedo = impl.create_texture(FORMAT_ALBEDO,
                                        width,
                                        height,
                                        scaler.diffuseAlbedoTextureUsage | write,
                                        MTLStorageModePrivate,
                                        @"Cycles MetalFX diffuse albedo");
      impl.specular_albedo = impl.create_texture(FORMAT_ALBEDO,
                                                 width,
                                                 height,
                                                 scaler.specularAlbedoTextureUsage | write,
                                                 MTLStorageModePrivate,
                                                 @"Cycles MetalFX specular albedo");
      impl.normal = impl.create_texture(FORMAT_NORMAL,
                                        width,
                                        height,
                                        scaler.normalTextureUsage | write,
                                        MTLStorageModePrivate,
                                        @"Cycles MetalFX normal");
      impl.roughness = impl.create_texture(FORMAT_ROUGHNESS,
                                           width,
                                           height,
                                           scaler.roughnessTextureUsage | write,
                                           MTLStorageModePrivate,
                                           @"Cycles MetalFX roughness");
      impl.output = impl.create_texture(FORMAT_OUTPUT,
                                        scaler_out_width,
                                        scaler_out_height,
                                        scaler.outputTextureUsage | MTLTextureUsageShaderRead,
                                        MTLStorageModePrivate,
                                        @"Cycles MetalFX output");

      if (!impl.color || !impl.depth || !impl.motion || !impl.albedo || !impl.specular_albedo ||
          !impl.normal || !impl.roughness || !impl.output)
      {
        impl.free_scaler();
        error = "MetalFX: out of memory allocating textures";
        return false;
      }

      impl.scaler = scaler;
      impl.width = width;
      impl.height = height;
      impl.out_width = out_width;
      impl.out_height = out_height;
      impl.scaler_out_width = scaler_out_width;
      impl.scaler_out_height = scaler_out_height;
      impl.need_history_reset = true;

      LOG_INFO << "MetalFX: created denoiser " << width << "x" << height << " to " << out_width
               << "x" << out_height;
      return true;
    }
  }

  error = "MetalFX denoising requires macOS 26 or newer";
  return false;
}

bool MetalFXDenoiserContext::denoise(const MetalFXFrame &frame, string &error)
{
  Impl &impl = *impl_;

  if (!impl.scaler || frame.width != impl.width || frame.height != impl.height ||
      frame.out_width != impl.out_width || frame.out_height != impl.out_height)
  {
    error = "MetalFX: denoiser is not configured for this frame";
    return false;
  }

  @autoreleasepool {
    if (METALFX_AVAILABLE) {
      id<MTLFXTemporalDenoisedScaler> scaler = impl.scaler;

      id<MTLBuffer> render_buffer = (__bridge id<MTLBuffer>)impl.metal_device->get_native_buffer(
          frame.render_buffer);
      if (!render_buffer) {
        error = "MetalFX: render buffer is not on the Metal device";
        return false;
      }

      bool reset_history = frame.reset_history || impl.need_history_reset;

      if (frame.accumulate) {
        if (!impl.previous) {
          impl.previous = [impl.device
              newBufferWithLength:size_t(impl.width) * impl.height * PREVIOUS_STRIDE *
                                  sizeof(float)
                          options:MTLResourceStorageModePrivate];
          if (!impl.previous) {
            error = "MetalFX: out of memory allocating the accumulation buffer";
            return false;
          }
          impl.metal_device->stats.mem_alloc(impl.previous.allocatedSize);
          impl.previous_valid = false;
        }
        /* A sample count that did not grow means that a new render has started. */
        if (frame.num_samples <= impl.previous_num_samples) {
          impl.previous_valid = false;
        }
        if (!impl.previous_valid && frame.reset_history_on_restart) {
          reset_history = true;
        }
      }
      else {
        impl.previous_valid = false;
      }

      ConvertParams params = {};
      params.width = frame.width;
      params.height = frame.height;
      params.full_x = frame.full_x;
      params.full_y = frame.full_y;
      params.offset = frame.offset;
      params.stride = frame.stride;
      params.pass_stride = frame.pass_stride;
      params.pass_color = frame.pass_color;
      params.pass_depth = frame.pass_depth;
      params.pass_albedo = frame.pass_albedo;
      params.pass_normal = frame.pass_normal;
      params.pass_roughness = frame.pass_roughness;
      params.pass_motion = frame.pass_motion;
      params.pass_motion_weight = frame.pass_motion_weight;
      params.pass_sample_count = frame.pass_sample_count;
      params.num_samples = frame.num_samples;
      params.out_width = frame.out_width;
      params.out_height = frame.out_height;
      params.out_full_x = frame.out_full_x;
      params.out_full_y = frame.out_full_y;
      params.out_offset = frame.out_offset;
      params.out_stride = frame.out_stride;
      params.pass_noisy = frame.pass_noisy;
      params.pass_denoised = frame.pass_denoised;
      params.num_components = frame.num_components;
      params.use_compositing = frame.use_compositing;
      params.upscale_factor = frame.upscale_factor;
      params.accumulate = frame.accumulate;
      params.previous_valid = impl.previous_valid;
      /* While a render accumulates nothing moves. Its first frame is related to what was
       * denoised before by the motion pass, when there is one. */
      params.use_motion = !frame.accumulate || !impl.previous_valid;
      params.use_albedo = frame.use_albedo;
      params.scaler_out_width = impl.scaler_out_width;
      params.scaler_out_height = impl.scaler_out_height;
      /* Clip Z and W as functions of the view Z, on the view axis. */
      params.depth_a = frame.view_to_clip[10];
      params.depth_b = frame.view_to_clip[11];
      params.depth_c = frame.view_to_clip[14];
      params.depth_d = frame.view_to_clip[15];
      for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
          params.rotation[row * 3 + column] = frame.view_to_world[row * 4 + column];
        }
      }
      params.show_input = show_input();

      id<MTLCommandBuffer> command_buffer = [impl.queue commandBuffer];
      command_buffer.label = @"Cycles MetalFX";

      impl.dispatch(command_buffer,
                    impl.pack_pipeline,
                    params,
                    render_buffer,
                    impl.width,
                    impl.height,
                    true);

      scaler.colorTexture = impl.color;
      scaler.depthTexture = impl.depth;
      scaler.motionTexture = impl.motion;
      scaler.diffuseAlbedoTexture = impl.albedo;
      scaler.specularAlbedoTexture = impl.specular_albedo;
      scaler.normalTexture = impl.normal;
      scaler.roughnessTexture = impl.roughness;
      scaler.outputTexture = impl.output;
      /* The camera shifts the raster position by minus the jitter, with Y up: the image moves
       * by the jitter in X and by minus the jitter in Y of the texture, which is what MetalFX
       * takes. Confirmed by measurement, any other sign blurs upscaled results. */
      scaler.jitterOffsetX = frame.jitter_x;
      scaler.jitterOffsetY = -frame.jitter_y;
      /* Motion is in pixels of the rendered frame. */
      scaler.motionVectorScaleX = 1.0f;
      scaler.motionVectorScaleY = 1.0f;
      scaler.depthReversed = YES;
      scaler.preExposure = 1.0f;
      scaler.shouldResetHistory = reset_history;

      simd_float4x4 world_to_view, view_to_clip;
      matrix_to_simd(frame.world_to_view, world_to_view);
      matrix_to_simd(frame.view_to_clip, view_to_clip);
      scaler.worldToViewMatrix = world_to_view;
      scaler.viewToClipMatrix = view_to_clip;

      [scaler encodeToCommandBuffer:command_buffer];

      impl.dispatch(command_buffer,
                    impl.unpack_pipeline,
                    params,
                    render_buffer,
                    impl.out_width,
                    impl.out_height,
                    false);

      [command_buffer commit];
      [command_buffer waitUntilCompleted];

      if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        NSError *ns_error = command_buffer.error;
        error = string("MetalFX: denoising failed: ") +
                (ns_error ? [[ns_error localizedDescription] UTF8String] : "unknown error");
        impl.need_history_reset = true;
        impl.previous_valid = false;
        return false;
      }

      impl.need_history_reset = false;
      if (frame.accumulate) {
        impl.previous_valid = true;
        impl.previous_num_samples = frame.num_samples;
      }
      else {
        impl.previous_num_samples = 0;
      }
      return true;
    }
  }

  error = "MetalFX denoising requires macOS 26 or newer";
  return false;
}

CCL_NAMESPACE_END

#endif /* WITH_METAL */
