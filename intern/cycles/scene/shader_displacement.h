/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/array.h"
#include "util/types.h"

CCL_NAMESPACE_BEGIN

class Shader;
class Scene;
class Progress;
struct SVMDisplacementImage;

/* Conservative object-space scalar displacement range, before the integrator multiplier.
 * An inverted interval means that the graph cannot currently be bounded. */
float2 shader_displacement_bounds(Shader *shader, Scene *scene, Progress &progress);

/* Verify the complete program and its stack dependencies before selecting the fused evaluator. */
bool shader_displacement_image(const array<int> &program, SVMDisplacementImage &result);

CCL_NAMESPACE_END
