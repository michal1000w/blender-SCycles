/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>

#include "kernel/svm/node_types.h"
#include "scene/shader_displacement.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN
namespace {

template<typename T> void emit(array<int> &program, ShaderNodeType type, const T &node)
{
  program.push_back_slow(type);
  const size_t offset = program.size();
  program.resize(offset + sizeof(T) / sizeof(int));
  memcpy(program.data() + offset, &node, sizeof(T));
}

SVMInputFloat stack_input(const uint offset)
{
  return {SVM_INPUT_STACK_OFFSET_MASK | offset};
}

array<int> image_program(bool derivatives = false, bool alpha = false, bool mapping = false)
{
  array<int> p;
  emit(p,
       derivatives ? NODE_ATTR_DERIVATIVE : NODE_ATTR,
       SVMNodeAttr{.attr = 123,
                   .out_offset = 0,
                   .output_type = NODE_ATTR_OUTPUT_FLOAT3,
                   .bump_offset = NODE_BUMP_OFFSET_CENTER,
                   .store_derivatives = uint8_t(derivatives),
                   .bump_filter_width = 1.0f});
  if (mapping) {
    emit(p,
         derivatives ? NODE_TEXTURE_MAPPING_DERIVATIVE : NODE_TEXTURE_MAPPING,
         SVMNodeTextureMapping{.vec_offset = 0,
                               .out_offset = 0,
                               .tfm = Transform{make_float4(2.0f, 0.0f, 0.0f, -0.3f),
                                                make_float4(0.0f, -3.0f, 0.0f, 0.7f),
                                                make_float4(0.0f, 0.0f, 4.0f, 1.1f)}});
  }
  emit(p,
       derivatives ? NODE_TEX_IMAGE_DERIVATIVE : NODE_TEX_IMAGE,
       SVMNodeTexImage{.id = 45,
                       .projection = NODE_IMAGE_PROJ_FLAT,
                       .flags = NODE_IMAGE_COMPRESS_AS_SRGB,
                       .co = 0,
                       .out_offset = 9,
                       .alpha_offset = 12});
  if (!alpha) {
    emit(p,
         NODE_CONVERT,
         SVMNodeConvert{.convert_type = NODE_CONVERT_CF, .from_offset = 9, .to_offset = 12});
  }
  /* Reuse the old coordinate stack space for the normal and output. */
  emit(p,
       NODE_GEOMETRY,
       SVMNodeGeometry{.geom_type = NODE_GEOM_N,
                       .bump_offset = NODE_BUMP_OFFSET_CENTER,
                       .store_derivatives = 0,
                       .out_offset = 0,
                       .bump_filter_width = 1.0f});
  emit(p,
       NODE_DISPLACEMENT,
       SVMNodeDisplacement{.space = NODE_NORMAL_MAP_OBJECT,
                           .height = stack_input(12),
                           .midlevel = {__float_as_uint(0.2f)},
                           .scale = {__float_as_uint(-0.5f)},
                           .normal_offset = 0,
                           .out_offset = 3});
  emit(p, NODE_SET_DISPLACEMENT, SVMNodeSetDisplacement{.fac_offset = 3});
  p.push_back_slow(NODE_END);
  return p;
}

TEST(scene_displacement_image, AcceptsColorAlphaDerivativesAndStackReuse)
{
  for (const bool derivatives : {false, true}) {
    for (const bool alpha : {false, true}) {
      const array<int> p = image_program(derivatives, alpha);
      SVMDisplacementImage result;
      ASSERT_TRUE(shader_displacement_image(p, result));
      EXPECT_EQ(result.attribute.attr, 123);
      EXPECT_EQ(result.image.id, 45);
      EXPECT_EQ(result.use_derivatives, derivatives);
      EXPECT_EQ(result.height_is_alpha, alpha);
      EXPECT_EQ(__uint_as_float(result.displacement.midlevel.bits), 0.2f);
      EXPECT_EQ(__uint_as_float(result.displacement.scale.bits), -0.5f);
    }
  }
}

TEST(scene_displacement_image, RejectsEveryTruncatedProgram)
{
  const array<int> valid = image_program();
  for (size_t length = 0; length < valid.size(); length++) {
    array<int> p(length);
    std::copy_n(valid.data(), length, p.data());
    SVMDisplacementImage result;
    EXPECT_FALSE(shader_displacement_image(p, result)) << length;
  }
}

TEST(scene_displacement_image, PreservesNonidentityTextureMapping)
{
  for (const bool derivatives : {false, true}) {
    const array<int> p = image_program(derivatives, false, true);
    SVMDisplacementImage result;
    ASSERT_TRUE(shader_displacement_image(p, result));
    EXPECT_TRUE(result.use_mapping);
    EXPECT_EQ(result.mapping.tfm.x.x, 2.0f);
    EXPECT_EQ(result.mapping.tfm.x.w, -0.3f);
    EXPECT_EQ(result.mapping.tfm.y.y, -3.0f);
    EXPECT_EQ(result.mapping.tfm.y.w, 0.7f);
    EXPECT_EQ(result.mapping.tfm.z.z, 4.0f);
    EXPECT_EQ(result.mapping.tfm.z.w, 1.1f);
  }
}

TEST(scene_displacement_image, RejectsExtraOperationsAfterOutput)
{
  array<int> p = image_program();
  p.resize(p.size() - 1);
  emit(p, NODE_VALUE_F, SVMNodeValueF{.value = 1.0f, .out_offset = 0});
  p.push_back_slow(NODE_END);
  SVMDisplacementImage result;
  EXPECT_FALSE(shader_displacement_image(p, result));
}

TEST(scene_displacement_image, RejectsPartialVectorOverwrite)
{
  array<int> p = image_program();
  const size_t normal_offset = 1 + sizeof(SVMNodeAttr) / sizeof(int) + 1 +
                               sizeof(SVMNodeTexImage) / sizeof(int) + 1 +
                               sizeof(SVMNodeConvert) / sizeof(int);
  for (size_t i = 0; i < p.size(); i++) {
    if (i == normal_offset) {
      /* The grayscale height is at 12; destroy one of the future normal lanes after it is
       * written by replacing the displacement's normal input with a partly unknown vector. */
      SVMNodeGeometry node;
      memcpy(&node, p.data() + i + 1, sizeof(node));
      node.out_offset = 1;
      memcpy(p.data() + i + 1, &node, sizeof(node));
      break;
    }
  }
  SVMDisplacementImage result;
  EXPECT_FALSE(shader_displacement_image(p, result));
}

TEST(scene_displacement_image, RejectsMismatchedCoordinateDerivatives)
{
  array<int> p = image_program(true);
  p[1 + sizeof(SVMNodeAttr) / sizeof(int)] = NODE_TEX_IMAGE;
  SVMDisplacementImage result;
  EXPECT_FALSE(shader_displacement_image(p, result));
}

TEST(scene_displacement_image, RejectsUnknownOpcodeAndTrailingData)
{
  array<int> p = image_program();
  p[0] = NODE_TEX_NOISE;
  SVMDisplacementImage result;
  EXPECT_FALSE(shader_displacement_image(p, result));
  p = image_program();
  p.push_back_slow(NODE_END);
  EXPECT_FALSE(shader_displacement_image(p, result));
}

}  // namespace
CCL_NAMESPACE_END
