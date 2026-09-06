/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/shader_displacement.h"

#include <cmath>
#include <unordered_map>

#include "scene/integrator.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"
#include "util/colorspace.h"
#include "util/log.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

namespace {

struct Range {
  float3 lo = zero_float3();
  float3 hi = zero_float3();
  bool valid = false;
};

class DisplacementBounds {
 public:
  DisplacementBounds(Scene *scene, Progress &progress) : scene_(scene), progress_(progress) {}

  Range input(ShaderInput *input, const int depth = 0)
  {
    if (input->link) {
      return output(input->link, depth + 1);
    }
    if (input->type() == SocketType::FLOAT) {
      const float value = input->parent->get_float(input->socket_type);
      return constant(make_float3(value));
    }
    if (SocketType::is_float3(input->type())) {
      return constant(input->parent->get_float3(input->socket_type));
    }
    return {};
  }

  static Range constant(const float3 value)
  {
    return {value, value, isfinite_safe(value)};
  }

 private:
  Range output(ShaderOutput *output, const int depth)
  {
    if (depth > 64) {
      return {};
    }
    if (const auto found = memo_.find(output); found != memo_.end()) {
      return found->second;
    }
    const Range range = evaluate(output, depth);
    memo_[output] = range;
    return range;
  }

  Range gray(Range range)
  {
    if (!range.valid) {
      return {};
    }
    /* Do not assume positive luminance weights: wide-gamut working spaces can have negative
     * components. Interval projection uses the actual scene color transform. */
    const float3 weights = make_float3(
        scene_->shader_manager->linear_rgb_to_gray(make_float3(1.0f, 0.0f, 0.0f)),
        scene_->shader_manager->linear_rgb_to_gray(make_float3(0.0f, 1.0f, 0.0f)),
        scene_->shader_manager->linear_rgb_to_gray(make_float3(0.0f, 0.0f, 1.0f)));
    const float3 a = range.lo * weights;
    const float3 b = range.hi * weights;
    return {make_float3(reduce_add(min(a, b))), make_float3(reduce_add(max(a, b))), true};
  }

  Range evaluate(ShaderOutput *output, const int depth)
  {
    ShaderNode *node = output->parent;
    if (node->type == RGBRampNode::get_node_type()) {
      const RGBRampNode *ramp = static_cast<RGBRampNode *>(node);
      Range range;
      /* The runtime ramp clamps its input and linearly interpolates the same table. Its extrema
       * are bounded by the entries even if the input is an arbitrary procedural shader. */
      if (output->name() == ustring("Alpha")) {
        for (const float value : ramp->get_ramp_alpha()) {
          if (!std::isfinite(value)) {
            return {};
          }
          const float3 v = make_float3(value);
          range = range.valid ? Range{min(range.lo, v), max(range.hi, v), true} : constant(v);
        }
      }
      else {
        for (const packed_float3 value : ramp->get_ramp()) {
          const float3 v = float3(value);
          if (!isfinite_safe(v)) {
            return {};
          }
          range = range.valid ? Range{min(range.lo, v), max(range.hi, v), true} : constant(v);
        }
      }
      return range;
    }
    if (node->type == ImageTextureNode::get_node_type()) {
      ImageTextureNode *image = static_cast<ImageTextureNode *>(node);
      if (image->get_interpolation() != INTERPOLATION_LINEAR &&
          image->get_interpolation() != INTERPOLATION_CLOSEST)
      {
        return {};
      }
      /* Metadata describes one tile; mixed-format UDIM sets need a bound for every tile. */
      if (image->handle.num_tiles() != 0) {
        return {};
      }
      const ImageMetaData metadata = image->handle.metadata(progress_);
      LOG_DEBUG << "Displacement image bound metadata: " << image->name << " type "
                << int(metadata.type) << " channels " << metadata.channels;
      const bool integer_image = metadata.type == IMAGE_DATA_TYPE_BYTE ||
                                 metadata.type == IMAGE_DATA_TYPE_BYTE4 ||
                                 metadata.type == IMAGE_DATA_TYPE_USHORT ||
                                 metadata.type == IMAGE_DATA_TYPE_USHORT4;
      if (!integer_image || metadata.width <= 0 || metadata.height <= 0) {
        return {};
      }
      const bool unassociate = !image->output("Alpha")->links.empty() &&
                               !(ColorSpaceManager::colorspace_is_data(image->get_colorspace()) ||
                                 image->get_alpha_type() == IMAGE_ALPHA_CHANNEL_PACKED ||
                                 image->get_alpha_type() == IMAGE_ALPHA_IGNORE);
      if (unassociate && (metadata.channels == 2 || metadata.channels == 4) &&
          output->name() != ustring("Alpha"))
      {
        return {};
      }
      if (scene_->integrator->get_use_pixel_displacement() && output->name() != ustring("Alpha")) {
        const float2 pixels = image->handle.displacement_range();
        if (pixels.x <= pixels.y) {
          return {make_float3(pixels.x), make_float3(pixels.y), true};
        }
      }
      return {zero_float3(), one_float3(), true};
    }
    if (node->type == ValueNode::get_node_type()) {
      return constant(make_float3(static_cast<ValueNode *>(node)->get_value()));
    }
    if (node->type == ColorNode::get_node_type()) {
      return constant(static_cast<ColorNode *>(node)->get_value());
    }
    if (node->special_type == SHADER_SPECIAL_TYPE_AUTOCONVERT ||
        node->shader_node_type() == NODE_CONVERT)
    {
      ConvertNode *convert = static_cast<ConvertNode *>(node);
      const Range range = input(node->inputs[0], depth);
      switch (convert->convert_type()) {
        case NODE_CONVERT_CF:
          return gray(range);
        case NODE_CONVERT_FV:
          return range;
        default:
          return {};
      }
    }
    if (node->type == RGBToBWNode::get_node_type()) {
      return gray(input(node->input("Color"), depth));
    }
    /* Unknown graphs keep the user-specified conservative bound. In particular, do not infer
     * procedural extrema from a finite collection of samples. */
    return {};
  }

  Scene *scene_;
  Progress &progress_;
  std::unordered_map<ShaderOutput *, Range> memo_;
};

}  // namespace

float2 shader_displacement_bounds(Shader *shader, Scene *scene, Progress &progress)
{
  const float2 unknown = make_float2(1.0f, -1.0f);
  if (!shader->has_displacement || !shader->graph) {
    return unknown;
  }
  ShaderInput *input = shader->graph->output()->input("Displacement");
  if (!input->link || input->link->parent->type != DisplacementNode::get_node_type()) {
    return unknown;
  }
  DisplacementNode *node = static_cast<DisplacementNode *>(input->link->parent);
  /* Object-space displacement normalizes its input normal before scaling. A magnitude
   * bound therefore remains valid for any normal input, including the geometry node
   * inserted by graph finalization. The renderer uses the symmetric enclosing interval. */
  if (node->get_space() != NODE_NORMAL_MAP_OBJECT) {
    return unknown;
  }
  DisplacementBounds evaluator(scene, progress);
  const Range height = evaluator.input(node->input("Height"));
  const Range midlevel = evaluator.input(node->input("Midlevel"));
  const Range scale = evaluator.input(node->input("Scale"));
  if (!height.valid || !midlevel.valid || !scale.valid) {
    return unknown;
  }
  const float lo = height.lo.x - midlevel.hi.x;
  const float hi = height.hi.x - midlevel.lo.x;
  const float a = lo * scale.lo.x;
  const float b = lo * scale.hi.x;
  const float c = hi * scale.lo.x;
  const float d = hi * scale.hi.x;
  const float min_value = min(min(a, b), min(c, d));
  const float max_value = max(max(a, b), max(c, d));
  if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
    return unknown;
  }
  const float safety = 1.0e-5f * max(1.0f, max(fabsf(min_value), fabsf(max_value)));
  return make_float2(min_value - safety, max_value + safety);
}

CCL_NAMESPACE_END
