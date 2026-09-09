/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/shader_displacement.h"

#include <cmath>
#include <cstring>

#include "kernel/svm/node_types.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

namespace {

/* Symbolic stack verification allows normal compiler stack reuse, but rejects any dependency
 * the fused path does not implement. Each vector lane is tracked independently. */
class DisplacementImageProgram {
  enum Kind { UNKNOWN, CONSTANT, COORDINATE, MAPPED, IMAGE, ALPHA, HEIGHT, NORMAL, DISPLACEMENT };
  struct Value {
    Kind kind = UNKNOWN;
    int lane = 0;
    float value = 0.0f;
  };
  Value stack_[SVM_STACK_SIZE];
  const array<int> &program_;
  size_t offset_ = 0;
  bool attribute_ = false, image_ = false, displacement_ = false, set_ = false;

  template<typename T> bool read(T &node)
  {
    constexpr size_t words = sizeof(T) / sizeof(int);
    if (offset_ + words > program_.size()) {
      return false;
    }
    memcpy(&node, program_.data() + offset_, sizeof(T));
    offset_ += words;
    return true;
  }

  bool matches(const int offset, const Kind kind, const int width = 1) const
  {
    if (offset < 0 || offset + width > SVM_STACK_SIZE) {
      return false;
    }
    for (int i = 0; i < width; i++) {
      if (stack_[offset + i].kind != kind || stack_[offset + i].lane != i) {
        return false;
      }
    }
    return true;
  }

  bool write(const int offset, const Kind kind, const int width = 1)
  {
    if (offset < 0 || offset + width > SVM_STACK_SIZE) {
      return false;
    }
    for (int i = 0; i < width; i++) {
      stack_[offset + i] = {kind, i};
    }
    return true;
  }

  Value input(const SVMInputFloat input) const
  {
    if ((input.bits >> 8) == (SVM_INPUT_STACK_OFFSET_MASK >> 8)) {
      const uint offset = input.bits & 0xffu;
      return offset < SVM_STACK_SIZE ? stack_[offset] : Value{};
    }
    return {CONSTANT, 0, __uint_as_float(input.bits)};
  }

 public:
  explicit DisplacementImageProgram(const array<int> &program) : program_(program) {}

  bool compile(SVMDisplacementImage &result)
  {
    result = {};
    while (offset_ < program_.size()) {
      const ShaderNodeType type = ShaderNodeType(program_[offset_++]);
      switch (type) {
        case NODE_END:
          if (!(offset_ == program_.size() && attribute_ && image_ && displacement_ && set_)) {
            return false;
          }
          /* A canonical stack lets the cached sequence call the original node functions
           * independently of the source program's allocation order and dead stack slots. */
          result.attribute.out_offset = 0;
          result.mapping.vec_offset = 0;
          result.mapping.out_offset = 0;
          result.image.co = 0;
          result.image.out_offset = 9;
          result.image.alpha_offset = 12;
          result.convert = {NODE_CONVERT_CF, 9, 13, {0, 0}};
          result.geometry = {NODE_GEOM_N, NODE_BUMP_OFFSET_CENTER, 0, 14, 1.0f};
          result.displacement.height = {
              SVM_INPUT_STACK_OFFSET_MASK | uint(result.height_is_alpha ? 12 : 13)};
          if (result.displacement.normal_offset != SVM_STACK_INVALID) {
            result.displacement.normal_offset = 14;
          }
          result.displacement.out_offset = 17;
          result.output = {17, {0, 0, 0}};
          return true;
        case NODE_ATTR:
        case NODE_ATTR_DERIVATIVE: {
          SVMNodeAttr node;
          if (attribute_ || !read(node) || node.output_type != NODE_ATTR_OUTPUT_FLOAT3 ||
              node.bump_offset != NODE_BUMP_OFFSET_CENTER ||
              (type == NODE_ATTR_DERIVATIVE && !node.store_derivatives))
          {
            return false;
          }
          attribute_ = true;
          result.attribute = node;
          result.use_derivatives = type == NODE_ATTR_DERIVATIVE;
          if (!write(node.out_offset, COORDINATE, result.use_derivatives ? 9 : 3)) {
            return false;
          }
          break;
        }
        case NODE_TEXTURE_MAPPING:
        case NODE_TEXTURE_MAPPING_DERIVATIVE: {
          SVMNodeTextureMapping node;
          const bool derivatives = type == NODE_TEXTURE_MAPPING_DERIVATIVE;
          const int width = derivatives ? 9 : 3;
          if (!read(node) || image_ || result.use_mapping ||
              derivatives != bool(result.use_derivatives) ||
              !matches(node.vec_offset, COORDINATE, width) ||
              !write(node.out_offset, MAPPED, width))
          {
            return false;
          }
          result.mapping = node;
          result.use_mapping = true;
          break;
        }
        case NODE_TEX_IMAGE:
        case NODE_TEX_IMAGE_DERIVATIVE: {
          SVMNodeTexImage node;
          const bool derivatives = type == NODE_TEX_IMAGE_DERIVATIVE;
          if (image_ || !read(node) || derivatives != bool(result.use_derivatives) ||
              !matches(node.co, result.use_mapping ? MAPPED : COORDINATE, derivatives ? 9 : 3))
          {
            return false;
          }
          image_ = true;
          result.image = node;
          if (node.out_offset != SVM_STACK_INVALID && !write(node.out_offset, IMAGE, 3)) {
            return false;
          }
          if (node.alpha_offset != SVM_STACK_INVALID && !write(node.alpha_offset, ALPHA)) {
            return false;
          }
          break;
        }
        case NODE_CONVERT: {
          SVMNodeConvert node;
          if (!read(node) || node.convert_type != NODE_CONVERT_CF ||
              !matches(node.from_offset, IMAGE, 3) || !write(node.to_offset, HEIGHT))
          {
            return false;
          }
          break;
        }
        case NODE_GEOMETRY: {
          SVMNodeGeometry node;
          if (!read(node) || node.geom_type != NODE_GEOM_N ||
              node.bump_offset != NODE_BUMP_OFFSET_CENTER || !write(node.out_offset, NORMAL, 3))
          {
            return false;
          }
          break;
        }
        case NODE_VALUE_F: {
          SVMNodeValueF node;
          if (!read(node) || !write(node.out_offset, CONSTANT) || !std::isfinite(node.value)) {
            return false;
          }
          stack_[node.out_offset].value = node.value;
          break;
        }
        case NODE_DISPLACEMENT: {
          SVMNodeDisplacement node;
          if (displacement_ || !read(node) || node.space != NODE_NORMAL_MAP_OBJECT) {
            return false;
          }
          const Value height = input(node.height);
          const Value midlevel = input(node.midlevel);
          const Value scale = input(node.scale);
          if ((height.kind != HEIGHT && height.kind != ALPHA) || height.lane != 0 ||
              midlevel.kind != CONSTANT || scale.kind != CONSTANT ||
              !std::isfinite(midlevel.value) || !std::isfinite(scale.value) ||
              (node.normal_offset != SVM_STACK_INVALID &&
               !matches(node.normal_offset, NORMAL, 3)) ||
              !write(node.out_offset, DISPLACEMENT, 3))
          {
            return false;
          }
          displacement_ = true;
          result.height_is_alpha = height.kind == ALPHA;
          result.displacement = node;
          result.displacement.midlevel = {__float_as_uint(midlevel.value)};
          result.displacement.scale = {__float_as_uint(scale.value)};
          break;
        }
        case NODE_SET_DISPLACEMENT: {
          SVMNodeSetDisplacement node;
          if (set_ || !read(node) || !matches(node.fac_offset, DISPLACEMENT, 3)) {
            return false;
          }
          set_ = true;
          /* Any later operations could read the displaced position. The fused evaluator must
           * end at this point, just like a normal scalar displacement output. */
          if (offset_ + 1 != program_.size() || program_[offset_] != NODE_END) {
            return false;
          }
          break;
        }
        default:
          return false;
      }
    }
    return false;
  }
};

}  // namespace

bool shader_displacement_image(const array<int> &program, SVMDisplacementImage &result)
{
  return DisplacementImageProgram(program).compile(result);
}

CCL_NAMESPACE_END
