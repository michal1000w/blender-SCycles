/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Translation of OSL camera shader bytecode for Metal. This only tests the translator, which
 * needs no GPU. `tests/python/cycles_osl_camera_scenes.py` compares renders of translated
 * shaders with the OSL runtime of the CPU device. */

#include <gtest/gtest.h>

#include "device/metal/osl_camera_translate.h"

CCL_NAMESPACE_BEGIN

/* Bytecode of this shader, as written by the OSL compiler, without the hints the translator
 * does not use:
 *
 *   float pick(float x, float table[3])
 *   {
 *     for (int i = 0; i < 3; i++) {
 *       if (table[i] > x) {
 *         return table[i];
 *       }
 *     }
 *     return 0.0;
 *   }
 *   shader unit(float scale = 2.0, int mode = 1, vector tint = vector(1.0, 0.5, 0.25),
 *               string space = "camera", float unset = 3.0,
 *               output point position = point(0.0),
 *               output vector direction = vector(0.0, 0.0, 1.0),
 *               output color throughput = color(1.0))
 *   {
 *     float table[3] = {0.25, 0.5, 0.75};
 *     float jitter = hashnoise(P);
 *     float value = pick(P[0] * scale, table) + unset;
 *     if (mode == 1 && space == "camera") {
 *       value += jitter;
 *     }
 *     direction = normalize(vector(P[0], P[1], value));
 *     throughput = tint * jitter;
 *   }
 */
static const char *unit_bytecode = R"OSO(OpenShadingLanguage 1.00
shader unit
param	float	scale	2
param	int	mode	1
param	vector	tint	1 0.5 0.25
param	string	space	"camera"
param	float	unset	3
oparam	point	position	0 0 0
oparam	vector	direction	0 0 1
oparam	color	throughput	1 1 1
global	point	P
local	int	___377_i
local	float[3]	table
local	float	jitter
local	float	value
const	float[3]	$const1	0.25 0.5 0.75
temp	float	$tmp1
const	int	$const2	0
temp	float	$tmp2
temp	float	$tmp3
const	string	$const3	"pick"
const	int	$const4	3
temp	int	$tmp4
temp	int	$tmp5
temp	float	$tmp6
temp	int	$tmp7
const	int	$const5	1
temp	int	$tmp8
const	float	$const6	0
temp	int	$tmp9
temp	int	$tmp10
const	string	$const7	"camera"
temp	int	$tmp11
temp	int	$tmp12
temp	vector	$tmp13
temp	float	$tmp14
temp	float	$tmp15
code ___main___
	assign		table $const1 	%argrw{"wr"}
	hashnoise	jitter P 	%argrw{"wr"}
	compref		$tmp2 P $const2 	%argrw{"wrr"}
	mul		$tmp3 $tmp2 scale 	%argrw{"wrr"}
	functioncall	$const3 17 	%argrw{"r"}
	for		$tmp5 7 9 14 16 	%argrw{"r"}
	assign		___377_i $const2 	%argrw{"wr"}
	lt		$tmp4 ___377_i $const4 	%argrw{"wrr"}
	neq		$tmp5 $tmp4 $const2 	%argrw{"wrr"}
	aref		$tmp6 table ___377_i 	%argrw{"wrr"}
	gt		$tmp7 $tmp6 $tmp3 	%argrw{"wrr"}
	if		$tmp7 14 14 	%argrw{"r"}
	aref		$tmp1 table ___377_i 	%argrw{"wrr"}
	return
	assign		$tmp8 ___377_i 	%argrw{"wr"}
	add		___377_i ___377_i $const5 	%argrw{"wrr"}
	assign		$tmp1 $const6 	%argrw{"wr"}
	add		value $tmp1 unset 	%argrw{"wrr"}
	eq		$tmp9 mode $const5 	%argrw{"wrr"}
	neq		$tmp10 $tmp9 $const2 	%argrw{"wrr"}
	if		$tmp10 24 24 	%argrw{"r"}
	eq		$tmp11 space $const7 	%argrw{"wrr"}
	neq		$tmp12 $tmp11 $const2 	%argrw{"wrr"}
	assign		$tmp10 $tmp12 	%argrw{"wr"}
	if		$tmp10 26 26 	%argrw{"r"}
	add		value value jitter 	%argrw{"wrr"}
	compref		$tmp14 P $const2 	%argrw{"wrr"}
	compref		$tmp15 P $const5 	%argrw{"wrr"}
	vector		$tmp13 $tmp14 $tmp15 value 	%argrw{"wrrr"}
	normalize	direction $tmp13 	%argrw{"wr"}
	mul		throughput tint jitter 	%argrw{"wrr"}
	end
)OSO";

/* A shader that samples a texture. */
static const char *texture_bytecode = R"OSO(OpenShadingLanguage 1.00
shader bad
oparam	point	position	0 0 0
oparam	vector	direction	0 0 1
oparam	color	throughput	1 1 1
global	point	P
const	string	$const1	"image.png"
const	int	$const2	0
temp	float	$tmp1
const	int	$const3	1
temp	float	$tmp2
code ___main___
	compref		$tmp1 P $const2 	%argrw{"wrr"}
	compref		$tmp2 P $const3 	%argrw{"wrr"}
	texture		throughput $const1 $tmp1 $tmp2 	%argrw{"wrrr"}
	end
)OSO";

/* Arrays:
 *
 *   color palette[3] = {color(1.0, 0.0, 0.0), color(0.0, 1.0, 0.0), color(0.0, 0.0, 1.0)};
 *   float weights[3];
 *   weights[0] = 0.5;
 *   weights[1] = P[0];
 *   weights[2] = 2.0;
 *   throughput = palette[index] * weights[index];
 */
static const char *table_bytecode = R"OSO(OpenShadingLanguage 1.00
shader table
param	int	index	1
oparam	point	position	0 0 0
oparam	vector	direction	0 0 1
oparam	color	throughput	1 1 1
global	point	P
local	color[3]	palette
local	float[3]	weights
const	color	$const1	1 0 0
const	int	$const2	0
const	color	$const3	0 1 0
const	int	$const4	1
const	color	$const5	0 0 1
const	int	$const6	2
const	float	$const7	0.5
temp	float	$tmp4
const	float	$const8	2
temp	color	$tmp5
temp	float	$tmp6
code ___main___
	aassign		palette $const2 $const1 	%argrw{"wrr"}
	aassign		palette $const4 $const3 	%argrw{"wrr"}
	aassign		palette $const6 $const5 	%argrw{"wrr"}
	aassign		weights $const2 $const7 	%argrw{"wrr"}
	compref		$tmp4 P $const2 	%argrw{"wrr"}
	aassign		weights $const4 $tmp4 	%argrw{"wrr"}
	aassign		weights $const6 $const8 	%argrw{"wrr"}
	aref		$tmp5 palette index 	%argrw{"wrr"}
	aref		$tmp6 weights index 	%argrw{"wrr"}
	mul		throughput $tmp5 $tmp6 	%argrw{"wrr"}
	end
)OSO";

static OSLCameraTranslateOptions unit_options()
{
  OSLCameraTranslateOptions options;
  options.params["scale"] = {OSLCameraTranslateParam::FLOAT, 1, ""};
  options.params["mode"] = {OSLCameraTranslateParam::INT, 1, ""};
  options.params["tint"] = {OSLCameraTranslateParam::FLOAT, 3, ""};
  options.params["space"] = {OSLCameraTranslateParam::STRING, 0, "camera"};
  return options;
}

static bool contains(const std::string &text, const char *substring)
{
  return text.find(substring) != std::string::npos;
}

TEST(metal_osl_camera, translate)
{
  OSLCameraTranslateResult result;
  std::string error;
  ASSERT_TRUE(osl_camera_translate_msl(unit_bytecode, unit_options(), result, error)) << error;

  /* The function called by the kernel. */
  EXPECT_TRUE(contains(result.source, "[[visible]] void cycles_metal_osl_camera("));
  /* The position is never written: no derivatives. The direction depends on the sensor
   * position and the throughput only on a hash of it. */
  EXPECT_TRUE(contains(result.source, "float3 s_position;"));
  EXPECT_TRUE(contains(result.source, "DualV s_direction;"));
  EXPECT_TRUE(contains(result.source, "float3 s_throughput;"));
  EXPECT_TRUE(contains(result.source, "float s_jitter"));
  /* The local table is only a copy of a constant: it reads the constant. */
  EXPECT_TRUE(contains(result.source, "constant float s_const1[3]"));
  EXPECT_FALSE(contains(result.source, "s_table["));
  /* A return from inside the loop of the inlined function leaves the loop, then the function. */
  EXPECT_TRUE(contains(result.source, "returned_"));
}

TEST(metal_osl_camera, parameters)
{
  OSLCameraTranslateResult result;
  std::string error;
  ASSERT_TRUE(osl_camera_translate_msl(unit_bytecode, unit_options(), result, error)) << error;

  /* Numbers with a value are read at render time, in the order of the shader. */
  ASSERT_EQ(result.slots.size(), 3);
  EXPECT_EQ(result.num_words, 5);
  EXPECT_EQ(result.slots[0].name, "scale");
  EXPECT_EQ(result.slots[0].offset, 0);
  EXPECT_FALSE(result.slots[0].is_int);
  EXPECT_EQ(result.slots[1].name, "mode");
  EXPECT_EQ(result.slots[1].offset, 1);
  EXPECT_TRUE(result.slots[1].is_int);
  EXPECT_EQ(result.slots[2].name, "tint");
  EXPECT_EQ(result.slots[2].offset, 2);
  EXPECT_EQ(result.slots[2].size, 3);
  EXPECT_TRUE(contains(result.source, "as_type<int>(prm[1])"));
  /* A parameter without a value keeps the default of the shader. */
  EXPECT_TRUE(contains(result.source, "float s_unset = 3.0f;"));

  /* Without values all parameters are constants. */
  OSLCameraTranslateResult defaults;
  ASSERT_TRUE(osl_camera_translate_msl(unit_bytecode, OSLCameraTranslateOptions(), defaults, error))
      << error;
  EXPECT_EQ(defaults.slots.size(), 0);
  EXPECT_EQ(defaults.num_words, 0);
  EXPECT_TRUE(contains(defaults.source, "float s_scale = 2.0f;"));
}

TEST(metal_osl_camera, arrays)
{
  OSLCameraTranslateResult result;
  std::string error;
  ASSERT_TRUE(
      osl_camera_translate_msl(table_bytecode, OSLCameraTranslateOptions(), result, error))
      << error;

  /* An array that is only filled with constants is a constant, not assigned for every ray. */
  EXPECT_TRUE(contains(result.source, "constant float3 s_palette[3]"));
  EXPECT_FALSE(contains(result.source, "s_palette[osl_idx(0, 3)] ="));
  /* An array with a computed element is a local variable. */
  EXPECT_TRUE(contains(result.source, "float s_weights[3];"));
  EXPECT_TRUE(contains(result.source, "s_weights[osl_idx(1, 3)] ="));
}

TEST(metal_osl_camera, strings)
{
  /* Strings are constants of the translation: comparing them yields a constant. */
  OSLCameraTranslateOptions options = unit_options();
  OSLCameraTranslateResult camera;
  std::string error;
  ASSERT_TRUE(osl_camera_translate_msl(unit_bytecode, options, camera, error)) << error;
  EXPECT_TRUE(contains(camera.source, "s_tmp11 = 1;"));

  options.params["space"].string_value = "world";
  OSLCameraTranslateResult world;
  ASSERT_TRUE(osl_camera_translate_msl(unit_bytecode, options, world, error)) << error;
  EXPECT_TRUE(contains(world.source, "s_tmp11 = 0;"));
}

TEST(metal_osl_camera, unsupported)
{
  OSLCameraTranslateResult result;
  std::string error;
  EXPECT_FALSE(
      osl_camera_translate_msl(texture_bytecode, OSLCameraTranslateOptions(), result, error));
  EXPECT_TRUE(contains(error, "texture")) << error;

  error.clear();
  EXPECT_FALSE(osl_camera_translate_msl("", OSLCameraTranslateOptions(), result, error));
  EXPECT_FALSE(error.empty());

  error.clear();
  EXPECT_FALSE(osl_camera_translate_msl(
      "OpenShadingLanguage 1.00\nshader empty\n", OSLCameraTranslateOptions(), result, error));
  EXPECT_FALSE(error.empty());

  /* An instruction that uses a symbol which was never declared. */
  error.clear();
  EXPECT_FALSE(osl_camera_translate_msl("OpenShadingLanguage 1.00\nshader broken\n"
                                        "code ___main___\n\tassign\t\ta b\n",
                                        OSLCameraTranslateOptions(),
                                        result,
                                        error));
  EXPECT_TRUE(contains(error, "unknown symbol")) << error;
}

CCL_NAMESPACE_END
