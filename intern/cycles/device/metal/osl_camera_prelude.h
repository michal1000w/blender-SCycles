/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Metal implementation of the OSL operations used by translated camera shaders, see
 * `osl_camera_translate.cpp`. The semantics follow the OSL language specification: division by
 * zero yields zero, square roots and logarithms of invalid values are well defined, and so on.
 *
 * Values that carry screen space derivatives are `OslDual`: the translated shader computes the
 * ray differentials by automatic differentiation, as the OSL runtime does on other devices.
 *
 * The hash functions are the ones of OpenImageIO (`hash.h`) and Open Shading Language
 * (`oslnoise.h`), both BSD-3-Clause, so `hashnoise()` and `cellnoise()` return the same values
 * as on other devices. */

CCL_NAMESPACE_BEGIN

static const char *osl_camera_msl_prelude_1 = R"MSL(
#include <metal_stdlib>
using namespace metal;

#define OSL_FLT_MAX 3.402823466e+38f
#define OSL_FLT_MIN 1.175494351e-38f
#define OSL_INF as_type<float>(0x7f800000u)
#define OSL_NAN as_type<float>(0x7fc00000u)
/* Loops end after this many iterations, so that a shader that never terminates does not hang
 * the GPU. */
#define OSL_LOOP_LIMIT 1000000

template<typename T> struct OslDual {
  T v;
  T dx;
  T dy;
};
typedef OslDual<float> DualF;
typedef OslDual<float3> DualV;

inline float osl_if(bool c, float a, float b)
{
  return c ? a : b;
}
inline float3 osl_if(bool3 c, float3 a, float3 b)
{
  return select(b, a, c);
}

inline DualF osl_dual(float a)
{
  return DualF{a, 0.0f, 0.0f};
}
inline DualV osl_dual(float3 a)
{
  return DualV{a, float3(0.0f), float3(0.0f)};
}
inline DualV osl_dual3(DualF a)
{
  return DualV{float3(a.v), float3(a.dx), float3(a.dy)};
}
template<typename T> inline OslDual<T> osl_chain(OslDual<T> a, T f, T df)
{
  return OslDual<T>{f, df * a.dx, df * a.dy};
}

/* Arithmetic. */

template<typename T> inline OslDual<T> osl_add(OslDual<T> a, OslDual<T> b)
{
  return OslDual<T>{a.v + b.v, a.dx + b.dx, a.dy + b.dy};
}
template<typename T> inline OslDual<T> osl_sub(OslDual<T> a, OslDual<T> b)
{
  return OslDual<T>{a.v - b.v, a.dx - b.dx, a.dy - b.dy};
}
template<typename T> inline OslDual<T> osl_neg(OslDual<T> a)
{
  return OslDual<T>{-a.v, -a.dx, -a.dy};
}
template<typename T> inline OslDual<T> osl_mul(OslDual<T> a, OslDual<T> b)
{
  return OslDual<T>{a.v * b.v, a.v * b.dx + a.dx * b.v, a.v * b.dy + a.dy * b.v};
}

inline int osl_div(int a, int b)
{
  return (b == 0) ? 0 : a / b;
}
inline int osl_mod(int a, int b)
{
  return (b == 0) ? 0 : a % b;
}
inline float osl_div(float a, float b)
{
  return (b == 0.0f) ? 0.0f : a / b;
}
inline float3 osl_div(float3 a, float3 b)
{
  return float3(osl_div(a.x, b.x), osl_div(a.y, b.y), osl_div(a.z, b.z));
}
template<typename T> inline OslDual<T> osl_div(OslDual<T> a, OslDual<T> b)
{
  const T binv = osl_div(T(1.0f), b.v);
  const T q = a.v * binv;
  return OslDual<T>{q, binv * (a.dx - q * b.dx), binv * (a.dy - q * b.dy)};
}

inline int osl_ftoi(float a)
{
  return int(a);
}
inline int osl_idx(int i, int n)
{
  return clamp(i, 0, n - 1);
}

/* Functions of one argument. `D` is the derivative, with `x` the argument and `f` the value. */

#define OSL_UNARY_DUAL(name, D) \
  template<typename T> inline OslDual<T> name(OslDual<T> a) \
  { \
    const T x = a.v; \
    const T f = name(x); \
    return osl_chain(a, f, T(D)); \
  }

template<typename T> inline T osl_sin(T x)
{
  return sin(x);
}
OSL_UNARY_DUAL(osl_sin, cos(x))

template<typename T> inline T osl_cos(T x)
{
  return cos(x);
}
OSL_UNARY_DUAL(osl_cos, -sin(x))

template<typename T> inline T osl_tan(T x)
{
  return tan(x);
}
OSL_UNARY_DUAL(osl_tan, T(1.0f) / (cos(x) * cos(x)))

template<typename T> inline T osl_asin(T x)
{
  return asin(clamp(x, T(-1.0f), T(1.0f)));
}
OSL_UNARY_DUAL(osl_asin,
               osl_if(abs(x) < T(1.0f), T(1.0f) / sqrt(max(T(1.0f) - x * x, T(1e-30f))), T(0.0f)))

template<typename T> inline T osl_acos(T x)
{
  return acos(clamp(x, T(-1.0f), T(1.0f)));
}
OSL_UNARY_DUAL(osl_acos,
               osl_if(abs(x) < T(1.0f), T(-1.0f) / sqrt(max(T(1.0f) - x * x, T(1e-30f))), T(0.0f)))

template<typename T> inline T osl_atan(T x)
{
  return atan(x);
}
OSL_UNARY_DUAL(osl_atan, T(1.0f) / (T(1.0f) + x * x))

template<typename T> inline T osl_sinh(T x)
{
  return sinh(x);
}
OSL_UNARY_DUAL(osl_sinh, cosh(x))

template<typename T> inline T osl_cosh(T x)
{
  return cosh(x);
}
OSL_UNARY_DUAL(osl_cosh, sinh(x))

template<typename T> inline T osl_tanh(T x)
{
  return tanh(x);
}
OSL_UNARY_DUAL(osl_tanh, T(1.0f) / (cosh(x) * cosh(x)))

template<typename T> inline T osl_exp(T x)
{
  return exp(x);
}
OSL_UNARY_DUAL(osl_exp, f)

template<typename T> inline T osl_exp2(T x)
{
  return exp2(x);
}
OSL_UNARY_DUAL(osl_exp2, f * T(0.69314718056f))

template<typename T> inline T osl_expm1(T x)
{
  /* Series for small arguments, where exp(x) - 1 cancels. */
  return osl_if(abs(x) < T(1e-3f),
                x * (T(1.0f) + x * (T(0.5f) + x * T(0.16666667f))),
                exp(x) - T(1.0f));
}
OSL_UNARY_DUAL(osl_expm1, exp(x))

template<typename T> inline T osl_log(T x)
{
  return osl_if(x > T(0.0f), log(max(x, T(OSL_FLT_MIN))), T(-OSL_FLT_MAX));
}
OSL_UNARY_DUAL(osl_log, osl_if(x < T(OSL_FLT_MIN), T(0.0f), T(1.0f) / x))

template<typename T> inline T osl_log2(T x)
{
  return osl_if(x > T(0.0f), log2(max(x, T(OSL_FLT_MIN))), T(-OSL_FLT_MAX));
}
OSL_UNARY_DUAL(osl_log2, osl_if(x < T(OSL_FLT_MIN), T(0.0f), T(1.44269504089f) / x))

template<typename T> inline T osl_log10(T x)
{
  return osl_if(x > T(0.0f), log10(max(x, T(OSL_FLT_MIN))), T(-OSL_FLT_MAX));
}
OSL_UNARY_DUAL(osl_log10, osl_if(x < T(OSL_FLT_MIN), T(0.0f), T(0.43429448190f) / x))

template<typename T> inline T osl_logb(T x)
{
  return osl_if(x != T(0.0f), floor(log2(max(abs(x), T(OSL_FLT_MIN)))), T(-OSL_FLT_MAX));
}

template<typename T> inline T osl_sqrt(T x)
{
  return osl_if(x > T(0.0f), sqrt(max(x, T(0.0f))), T(0.0f));
}
OSL_UNARY_DUAL(osl_sqrt, osl_if(x > T(0.0f), T(0.5f) / max(f, T(OSL_FLT_MIN)), T(0.0f)))

template<typename T> inline T osl_inversesqrt(T x)
{
  return osl_if(x > T(0.0f), T(1.0f) / sqrt(max(x, T(OSL_FLT_MIN))), T(0.0f));
}
OSL_UNARY_DUAL(osl_inversesqrt,
               osl_if(x > T(0.0f), T(-0.5f) * f / max(x, T(OSL_FLT_MIN)), T(0.0f)))

template<typename T> inline T osl_cbrt(T x)
{
  return sign(x) * pow(abs(x), T(1.0f / 3.0f));
}
OSL_UNARY_DUAL(osl_cbrt,
               osl_if(x != T(0.0f), T(1.0f) / max(T(3.0f) * f * f, T(OSL_FLT_MIN)), T(0.0f)))

template<typename T> inline T osl_abs(T x)
{
  return abs(x);
}
OSL_UNARY_DUAL(osl_abs, osl_if(x >= T(0.0f), T(1.0f), T(-1.0f)))

template<typename T> inline T osl_erf(T x)
{
  /* Abramowitz and Stegun 7.1.28. */
  const T a = abs(x);
  const T b = T(1.0f) - (T(1.0f) - a);
  const T r = T(1.0f) +
              b * (T(0.0705230784f) +
                   b * (T(0.0422820123f) +
                        b * (T(0.0092705272f) +
                             b * (T(0.0001520143f) +
                                  b * (T(0.0002765672f) + b * T(0.0000430638f))))));
  const T s = r * r;
  const T t = s * s;
  const T u = t * t;
  const T v = u * u;
  return sign(x) * (T(1.0f) - T(1.0f) / v);
}
OSL_UNARY_DUAL(osl_erf, T(1.12837916709f) * exp(-x * x))

template<typename T> inline T osl_erfc(T x)
{
  return T(1.0f) - osl_erf(x);
}
OSL_UNARY_DUAL(osl_erfc, T(-1.12837916709f) * exp(-x * x))

template<typename T> inline T osl_floor(T x)
{
  return floor(x);
}
template<typename T> inline T osl_ceil(T x)
{
  return ceil(x);
}
template<typename T> inline T osl_round(T x)
{
  return round(x);
}
template<typename T> inline T osl_trunc(T x)
{
  return trunc(x);
}
template<typename T> inline T osl_sign(T x)
{
  return sign(x);
}

inline int osl_isnan(float x)
{
  return int((as_type<uint>(x) & 0x7fffffffu) > 0x7f800000u);
}
inline int osl_isinf(float x)
{
  return int((as_type<uint>(x) & 0x7fffffffu) == 0x7f800000u);
}
inline int osl_isfinite(float x)
{
  return int((as_type<uint>(x) & 0x7f800000u) != 0x7f800000u);
}

/* Functions of several arguments. */

inline float osl_pow(float x, float y)
{
  if (y == 0.0f) {
    return 1.0f;
  }
  if (x == 0.0f) {
    return 0.0f;
  }
  if (x < 0.0f) {
    /* Only integer powers of negative numbers are real. */
    const float yi = floor(y);
    if (yi != y) {
      return 0.0f;
    }
    const float r = pow(-x, y);
    return (fmod(abs(yi), 2.0f) == 1.0f) ? -r : r;
  }
  return pow(x, y);
}
inline float3 osl_pow(float3 x, float3 y)
{
  return float3(osl_pow(x.x, y.x), osl_pow(x.y, y.y), osl_pow(x.z, y.z));
}
template<typename T> inline OslDual<T> osl_pow(OslDual<T> u, OslDual<T> v)
{
  const T powuvm1 = osl_pow(u.v, v.v - T(1.0f));
  const T powuv = powuvm1 * u.v;
  const T logu = osl_if(u.v > T(0.0f), osl_log(u.v), T(0.0f));
  return OslDual<T>{powuv,
                    v.v * powuvm1 * u.dx + logu * powuv * v.dx,
                    v.v * powuvm1 * u.dy + logu * powuv * v.dy};
}

inline float osl_atan2(float y, float x)
{
  return (x == 0.0f && y == 0.0f) ? 0.0f : atan2(y, x);
}
inline float3 osl_atan2(float3 y, float3 x)
{
  return float3(osl_atan2(y.x, x.x), osl_atan2(y.y, x.y), osl_atan2(y.z, x.z));
}
template<typename T> inline OslDual<T> osl_atan2(OslDual<T> y, OslDual<T> x)
{
  /* The derivative has the sign convention of the OSL runtime (`dual.h`), which is the
   * opposite of the mathematical one, so that ray differentials match other devices. */
  const T denom = x.v * x.v + y.v * y.v;
  const T inv = osl_div(T(1.0f), denom);
  return OslDual<T>{osl_atan2(y.v, x.v),
                    (y.v * x.dx - x.v * y.dx) * inv,
                    (y.v * x.dy - x.v * y.dy) * inv};
}

inline float osl_fmod(float a, float b)
{
  return (b != 0.0f) ? fmod(a, b) : 0.0f;
}
inline float3 osl_fmod(float3 a, float3 b)
{
  return float3(osl_fmod(a.x, b.x), osl_fmod(a.y, b.y), osl_fmod(a.z, b.z));
}
template<typename T> inline OslDual<T> osl_fmod(OslDual<T> a, OslDual<T> b)
{
  return OslDual<T>{osl_fmod(a.v, b.v), a.dx, a.dy};
}

template<typename T> inline T osl_min(T a, T b)
{
  return min(a, b);
}
template<typename T> inline T osl_max(T a, T b)
{
  return max(a, b);
}
/* Equal values select the second argument, as in the OSL runtime. */
template<typename T> inline OslDual<T> osl_min(OslDual<T> a, OslDual<T> b)
{
  return OslDual<T>{osl_if(a.v < b.v, a.v, b.v),
                    osl_if(a.v < b.v, a.dx, b.dx),
                    osl_if(a.v < b.v, a.dy, b.dy)};
}
template<typename T> inline OslDual<T> osl_max(OslDual<T> a, OslDual<T> b)
{
  return OslDual<T>{osl_if(a.v > b.v, a.v, b.v),
                    osl_if(a.v > b.v, a.dx, b.dx),
                    osl_if(a.v > b.v, a.dy, b.dy)};
}

template<typename T> inline T osl_mix(T a, T b, T t)
{
  return a * (T(1.0f) - t) + b * t;
}
template<typename T> inline OslDual<T> osl_mix(OslDual<T> a, OslDual<T> b, OslDual<T> t)
{
  return osl_add(osl_mul(a, osl_sub(osl_dual(T(1.0f)), t)), osl_mul(b, t));
}

template<typename T> inline T osl_step(T edge, T x)
{
  return osl_if(x < edge, T(0.0f), T(1.0f));
}

template<typename T> inline T osl_select(T a, T b, T c)
{
  return osl_if(c != T(0.0f), b, a);
}
template<typename T> inline OslDual<T> osl_select(OslDual<T> a, OslDual<T> b, T c)
{
  return OslDual<T>{osl_if(c != T(0.0f), b.v, a.v),
                    osl_if(c != T(0.0f), b.dx, a.dx),
                    osl_if(c != T(0.0f), b.dy, a.dy)};
}
inline int osl_select(int a, int b, int c)
{
  return (c != 0) ? b : a;
}

inline float osl_smoothstep(float e0, float e1, float x)
{
  if (x < e0) {
    return 0.0f;
  }
  if (x >= e1) {
    return 1.0f;
  }
  const float t = (x - e0) / (e1 - e0);
  return (3.0f - 2.0f * t) * (t * t);
}
inline DualF osl_smoothstep(DualF e0, DualF e1, DualF x)
{
  if (x.v < e0.v) {
    return osl_dual(0.0f);
  }
  if (x.v >= e1.v) {
    return osl_dual(1.0f);
  }
  const DualF t = osl_div(osl_sub(x, e0), osl_sub(e1, e0));
  return osl_mul(osl_sub(osl_dual(3.0f), osl_mul(osl_dual(2.0f), t)), osl_mul(t, t));
}

/* Derivative access. */

template<typename T> inline T osl_filterwidth(OslDual<T> a)
{
  return sqrt(a.dx * a.dx + a.dy * a.dy);
}

/* Vectors. */

inline float osl_dot(float3 a, float3 b)
{
  return dot(a, b);
}
inline DualF osl_dot(DualV a, DualV b)
{
  return DualF{dot(a.v, b.v), dot(a.v, b.dx) + dot(a.dx, b.v), dot(a.v, b.dy) + dot(a.dy, b.v)};
}
inline float3 osl_cross(float3 a, float3 b)
{
  return cross(a, b);
}
inline DualV osl_cross(DualV a, DualV b)
{
  return DualV{cross(a.v, b.v),
               cross(a.v, b.dx) + cross(a.dx, b.v),
               cross(a.v, b.dy) + cross(a.dy, b.v)};
}
inline float osl_length(float3 a)
{
  return osl_sqrt(dot(a, a));
}
inline DualF osl_length(DualV a)
{
  return osl_sqrt(osl_dot(a, a));
}
inline float osl_distance(float3 a, float3 b)
{
  return osl_length(a - b);
}
inline DualF osl_distance(DualV a, DualV b)
{
  return osl_length(osl_sub(a, b));
}
inline float3 osl_normalize(float3 a)
{
  const float len = osl_length(a);
  return (len == 0.0f) ? float3(0.0f) : a / len;
}
inline DualV osl_normalize(DualV a)
{
  const DualF len = osl_length(a);
  if (len.v == 0.0f) {
    return osl_dual(float3(0.0f));
  }
  return osl_mul(a, osl_dual3(osl_div(osl_dual(1.0f), len)));
}
inline float osl_comp(float3 a, int i)
{
  return a[osl_idx(i, 3)];
}
inline DualF osl_comp(DualV a, int i)
{
  const int c = osl_idx(i, 3);
  return DualF{a.v[c], a.dx[c], a.dy[c]};
}
inline float3 osl_setcomp(float3 a, int i, float x)
{
  a[osl_idx(i, 3)] = x;
  return a;
}
inline DualV osl_setcomp(DualV a, int i, DualF x)
{
  const int c = osl_idx(i, 3);
  a.v[c] = x.v;
  a.dx[c] = x.dx;
  a.dy[c] = x.dy;
  return a;
}
inline float3 osl_vec(float x, float y, float z)
{
  return float3(x, y, z);
}
inline DualV osl_vec(DualF x, DualF y, DualF z)
{
  return DualV{float3(x.v, y.v, z.v), float3(x.dx, y.dx, z.dx), float3(x.dy, y.dy, z.dy)};
}
)MSL";

static const char *osl_camera_msl_prelude_2 = R"MSL(
/* Matrices, in the row vector convention of OSL: element [row][column], points transform as
 * `p * M` and the translation is in the last row. */

struct OslMat {
  float4 r[4];
};

inline OslMat osl_mat(float m00, float m01, float m02, float m03,
                      float m10, float m11, float m12, float m13,
                      float m20, float m21, float m22, float m23,
                      float m30, float m31, float m32, float m33)
{
  OslMat m;
  m.r[0] = float4(m00, m01, m02, m03);
  m.r[1] = float4(m10, m11, m12, m13);
  m.r[2] = float4(m20, m21, m22, m23);
  m.r[3] = float4(m30, m31, m32, m33);
  return m;
}
inline OslMat osl_mat_diag(float f)
{
  return osl_mat(f, 0.0f, 0.0f, 0.0f, 0.0f, f, 0.0f, 0.0f, 0.0f, 0.0f, f, 0.0f, 0.0f, 0.0f, 0.0f, f);
}
inline OslMat osl_mat_mul(OslMat a, OslMat b)
{
  OslMat m;
  for (int i = 0; i < 4; i++) {
    m.r[i] = a.r[i].x * b.r[0] + a.r[i].y * b.r[1] + a.r[i].z * b.r[2] + a.r[i].w * b.r[3];
  }
  return m;
}
inline OslMat osl_mat_scale(OslMat a, float f)
{
  OslMat m;
  for (int i = 0; i < 4; i++) {
    m.r[i] = a.r[i] * f;
  }
  return m;
}
inline OslMat osl_mat_neg(OslMat a)
{
  return osl_mat_scale(a, -1.0f);
}
inline OslMat osl_mat_transpose(OslMat a)
{
  return osl_mat(a.r[0].x, a.r[1].x, a.r[2].x, a.r[3].x,
                 a.r[0].y, a.r[1].y, a.r[2].y, a.r[3].y,
                 a.r[0].z, a.r[1].z, a.r[2].z, a.r[3].z,
                 a.r[0].w, a.r[1].w, a.r[2].w, a.r[3].w);
}
inline int osl_mat_eq(OslMat a, OslMat b)
{
  return int(all(a.r[0] == b.r[0]) && all(a.r[1] == b.r[1]) && all(a.r[2] == b.r[2]) &&
             all(a.r[3] == b.r[3]));
}
inline float osl_mat_comp(OslMat a, int row, int col)
{
  return a.r[osl_idx(row, 4)][osl_idx(col, 4)];
}
inline OslMat osl_mat_setcomp(OslMat a, int row, int col, float x)
{
  a.r[osl_idx(row, 4)][osl_idx(col, 4)] = x;
  return a;
}
inline float osl_mat_determinant(OslMat a)
{
  const float4x4 m = float4x4(a.r[0], a.r[1], a.r[2], a.r[3]);
  return determinant(m);
}
/* Inverse by cofactors. Singular matrices invert to the identity, as in OSL. */
inline OslMat osl_mat_inverse(OslMat a)
{
  const float m00 = a.r[0].x, m01 = a.r[0].y, m02 = a.r[0].z, m03 = a.r[0].w;
  const float m10 = a.r[1].x, m11 = a.r[1].y, m12 = a.r[1].z, m13 = a.r[1].w;
  const float m20 = a.r[2].x, m21 = a.r[2].y, m22 = a.r[2].z, m23 = a.r[2].w;
  const float m30 = a.r[3].x, m31 = a.r[3].y, m32 = a.r[3].z, m33 = a.r[3].w;

  const float s0 = m00 * m11 - m10 * m01;
  const float s1 = m00 * m12 - m10 * m02;
  const float s2 = m00 * m13 - m10 * m03;
  const float s3 = m01 * m12 - m11 * m02;
  const float s4 = m01 * m13 - m11 * m03;
  const float s5 = m02 * m13 - m12 * m03;
  const float c5 = m22 * m33 - m32 * m23;
  const float c4 = m21 * m33 - m31 * m23;
  const float c3 = m21 * m32 - m31 * m22;
  const float c2 = m20 * m33 - m30 * m23;
  const float c1 = m20 * m32 - m30 * m22;
  const float c0 = m20 * m31 - m30 * m21;

  const float det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
  if (det == 0.0f) {
    return osl_mat_diag(1.0f);
  }
  const float inv = 1.0f / det;
  return osl_mat((m11 * c5 - m12 * c4 + m13 * c3) * inv,
                 (-m01 * c5 + m02 * c4 - m03 * c3) * inv,
                 (m31 * s5 - m32 * s4 + m33 * s3) * inv,
                 (-m21 * s5 + m22 * s4 - m23 * s3) * inv,
                 (-m10 * c5 + m12 * c2 - m13 * c1) * inv,
                 (m00 * c5 - m02 * c2 + m03 * c1) * inv,
                 (-m30 * s5 + m32 * s2 - m33 * s1) * inv,
                 (m20 * s5 - m22 * s2 + m23 * s1) * inv,
                 (m10 * c4 - m11 * c2 + m13 * c0) * inv,
                 (-m00 * c4 + m01 * c2 - m03 * c0) * inv,
                 (m30 * s4 - m31 * s2 + m33 * s0) * inv,
                 (-m20 * s4 + m21 * s2 - m23 * s0) * inv,
                 (-m10 * c3 + m11 * c1 - m12 * c0) * inv,
                 (m00 * c3 - m01 * c1 + m02 * c0) * inv,
                 (-m30 * s3 + m31 * s1 - m32 * s0) * inv,
                 (m20 * s3 - m21 * s1 + m22 * s0) * inv);
}

inline float3 osl_transform_vector(OslMat m, float3 a)
{
  return a.x * m.r[0].xyz + a.y * m.r[1].xyz + a.z * m.r[2].xyz;
}
inline DualV osl_transform_vector(OslMat m, DualV a)
{
  return DualV{osl_transform_vector(m, a.v),
               osl_transform_vector(m, a.dx),
               osl_transform_vector(m, a.dy)};
}
inline float3 osl_transform_point(OslMat m, float3 a)
{
  const float3 p = osl_transform_vector(m, a) + m.r[3].xyz;
  const float w = a.x * m.r[0].w + a.y * m.r[1].w + a.z * m.r[2].w + m.r[3].w;
  return (w != 0.0f) ? p / w : float3(0.0f);
}
inline DualV osl_transform_point(OslMat m, DualV a)
{
  const float3 wcol = float3(m.r[0].w, m.r[1].w, m.r[2].w);
  const float3 p = osl_transform_vector(m, a.v) + m.r[3].xyz;
  const float w = dot(a.v, wcol) + m.r[3].w;
  if (w == 0.0f) {
    return osl_dual(float3(0.0f));
  }
  const float inv = 1.0f / w;
  const float3 q = p * inv;
  return DualV{q,
               (osl_transform_vector(m, a.dx) - q * dot(a.dx, wcol)) * inv,
               (osl_transform_vector(m, a.dy) - q * dot(a.dy, wcol)) * inv};
}
inline float3 osl_transform_normal(OslMat m, float3 a)
{
  return osl_transform_vector(osl_mat_transpose(osl_mat_inverse(m)), a);
}
inline DualV osl_transform_normal(OslMat m, DualV a)
{
  return osl_transform_vector(osl_mat_transpose(osl_mat_inverse(m)), a);
}

/* Kernel data, addressed by offsets in floats from the start of the kernel parameters. */

inline float osl_kd(constant void *lp, int offset)
{
  return ((constant float *)lp)[offset];
}
/* A 3x4 `Transform`, which has the translation in the last column. */
inline OslMat osl_kd_transform(constant void *lp, int offset)
{
  constant float *t = ((constant float *)lp) + offset;
  return osl_mat(t[0], t[4], t[8], 0.0f,
                 t[1], t[5], t[9], 0.0f,
                 t[2], t[6], t[10], 0.0f,
                 t[3], t[7], t[11], 1.0f);
}
/* A 4x4 `ProjectionTransform`. */
inline OslMat osl_kd_projection(constant void *lp, int offset)
{
  constant float *t = ((constant float *)lp) + offset;
  return osl_mat(t[0], t[4], t[8], t[12],
                 t[1], t[5], t[9], t[13],
                 t[2], t[6], t[10], t[14],
                 t[3], t[7], t[11], t[15]);
}

/* Hash and cell noise. */

inline uint osl_rotl32(uint x, int k)
{
  return (x << k) | (x >> (32 - k));
}
inline uint osl_bjfinal(uint a, uint b, uint c)
{
  c ^= b; c -= osl_rotl32(b, 14);
  a ^= c; a -= osl_rotl32(c, 11);
  b ^= a; b -= osl_rotl32(a, 25);
  c ^= b; c -= osl_rotl32(b, 16);
  a ^= c; a -= osl_rotl32(c, 4);
  b ^= a; b -= osl_rotl32(a, 14);
  c ^= b; c -= osl_rotl32(b, 24);
  return c;
}
inline uint osl_inthash(uint k0)
{
  const uint start = 0xdeadbeefu + (1u << 2) + 13u;
  return osl_bjfinal(start + k0, start, start);
}
inline uint osl_inthash(uint k0, uint k1)
{
  const uint start = 0xdeadbeefu + (2u << 2) + 13u;
  return osl_bjfinal(start + k0, start + k1, start);
}
inline uint osl_inthash(uint k0, uint k1, uint k2)
{
  const uint start = 0xdeadbeefu + (3u << 2) + 13u;
  return osl_bjfinal(start + k0, start + k1, start + k2);
}
inline uint osl_inthash_mix(uint start, uint k0, uint k1, uint k2, uint k3, uint k4)
{
  uint a = start + k0;
  uint b = start + k1;
  uint c = start + k2;
  a -= c; a ^= osl_rotl32(c, 4); c += b;
  b -= a; b ^= osl_rotl32(a, 6); a += c;
  c -= b; c ^= osl_rotl32(b, 8); b += a;
  a -= c; a ^= osl_rotl32(c, 16); c += b;
  b -= a; b ^= osl_rotl32(a, 19); a += c;
  c -= b; c ^= osl_rotl32(b, 4); b += a;
  b += k4;
  a += k3;
  return osl_bjfinal(a, b, c);
}
inline uint osl_inthash(uint k0, uint k1, uint k2, uint k3)
{
  return osl_inthash_mix(0xdeadbeefu + (4u << 2) + 13u, k0, k1, k2, k3, 0u);
}
inline uint osl_inthash(uint k0, uint k1, uint k2, uint k3, uint k4)
{
  return osl_inthash_mix(0xdeadbeefu + (5u << 2) + 13u, k0, k1, k2, k3, k4);
}
inline float osl_bits_to_01(uint bits)
{
  return float(bits) * 2.3283064370807974e-10f;
}
inline uint osl_hash_key(float x)
{
  return as_type<uint>(x);
}
inline uint osl_cell_key(float x)
{
  return uint(int(floor(x)));
}

#define OSL_HASH_NOISE(name, key) \
  inline float name##_f1(float x) \
  { \
    return osl_bits_to_01(osl_inthash(key(x))); \
  } \
  inline float name##_f2(float x, float y) \
  { \
    return osl_bits_to_01(osl_inthash(key(x), key(y))); \
  } \
  inline float name##_f3(float3 p) \
  { \
    return osl_bits_to_01(osl_inthash(key(p.x), key(p.y), key(p.z))); \
  } \
  inline float name##_f4(float3 p, float t) \
  { \
    return osl_bits_to_01(osl_inthash(key(p.x), key(p.y), key(p.z), key(t))); \
  } \
  inline float3 name##_v1(float x) \
  { \
    return float3(osl_bits_to_01(osl_inthash(key(x), 0u)), \
                  osl_bits_to_01(osl_inthash(key(x), 1u)), \
                  osl_bits_to_01(osl_inthash(key(x), 2u))); \
  } \
  inline float3 name##_v2(float x, float y) \
  { \
    return float3(osl_bits_to_01(osl_inthash(key(x), key(y), 0u)), \
                  osl_bits_to_01(osl_inthash(key(x), key(y), 1u)), \
                  osl_bits_to_01(osl_inthash(key(x), key(y), 2u))); \
  } \
  inline float3 name##_v3(float3 p) \
  { \
    return float3(osl_bits_to_01(osl_inthash(key(p.x), key(p.y), key(p.z), 0u)), \
                  osl_bits_to_01(osl_inthash(key(p.x), key(p.y), key(p.z), 1u)), \
                  osl_bits_to_01(osl_inthash(key(p.x), key(p.y), key(p.z), 2u))); \
  } \
  inline float3 name##_v4(float3 p, float t) \
  { \
    return float3(osl_bits_to_01(osl_inthash(key(p.x), key(p.y), key(p.z), key(t), 0u)), \
                  osl_bits_to_01(osl_inthash(key(p.x), key(p.y), key(p.z), key(t), 1u)), \
                  osl_bits_to_01(osl_inthash(key(p.x), key(p.y), key(p.z), key(t), 2u))); \
  }

OSL_HASH_NOISE(osl_hashnoise, osl_hash_key)
OSL_HASH_NOISE(osl_cellnoise, osl_cell_key)

/* The integer hash() function. */
inline int osl_hash_i1(int x)
{
  return int(osl_inthash(uint(x)));
}
inline int osl_hash_f1(float x)
{
  return int(osl_inthash(osl_hash_key(x)));
}
inline int osl_hash_f2(float x, float y)
{
  return int(osl_inthash(osl_hash_key(x), osl_hash_key(y)));
}
inline int osl_hash_f3(float3 p)
{
  return int(osl_inthash(osl_hash_key(p.x), osl_hash_key(p.y), osl_hash_key(p.z)));
}
inline int osl_hash_f4(float3 p, float t)
{
  return int(
      osl_inthash(osl_hash_key(p.x), osl_hash_key(p.y), osl_hash_key(p.z), osl_hash_key(t)));
}

/* Colors. */

inline float3 osl_hsv_to_rgb(float3 hsv)
{
  const float h = hsv.x, s = hsv.y, v = hsv.z;
  if (s < 0.0001f) {
    return float3(v);
  }
  const float hh = 6.0f * (h - floor(h));
  const int hi = int(hh);
  const float f = hh - float(hi);
  const float p = v * (1.0f - s);
  const float q = v * (1.0f - s * f);
  const float t = v * (1.0f - s * (1.0f - f));
  switch (hi) {
    case 0:
      return float3(v, t, p);
    case 1:
      return float3(q, v, p);
    case 2:
      return float3(p, v, t);
    case 3:
      return float3(p, q, v);
    case 4:
      return float3(t, p, v);
    default:
      return float3(v, p, q);
  }
}
inline float3 osl_rgb_to_hsv(float3 rgb)
{
  const float r = rgb.x, g = rgb.y, b = rgb.z;
  const float mincomp = min(r, min(g, b));
  const float maxcomp = max(r, max(g, b));
  const float delta = maxcomp - mincomp;
  float h = 0.0f, s = 0.0f;
  const float v = maxcomp;
  if (maxcomp > 0.0f) {
    s = delta / maxcomp;
  }
  if (s > 0.0f) {
    if (r >= maxcomp) {
      h = (g - b) / delta;
    }
    else if (g >= maxcomp) {
      h = 2.0f + (b - r) / delta;
    }
    else {
      h = 4.0f + (r - g) / delta;
    }
    h *= (1.0f / 6.0f);
    if (h < 0.0f) {
      h += 1.0f;
    }
  }
  return float3(h, s, v);
}
inline float3 osl_hsl_to_rgb(float3 hsl)
{
  const float h = hsl.x, s = hsl.y, l = hsl.z;
  const float v = (l <= 0.5f) ? (l * (1.0f + s)) : (l * (1.0f - s) + s);
  if (v <= 0.0f) {
    return float3(0.0f);
  }
  const float m = 2.0f * l - v;
  return osl_hsv_to_rgb(float3(h, (v - m) / v, v));
}
inline float3 osl_rgb_to_hsl(float3 rgb)
{
  const float mincomp = min(rgb.x, min(rgb.y, rgb.z));
  const float3 hsv = osl_rgb_to_hsv(rgb);
  const float h = hsv.x, v = hsv.z;
  const float l = 0.5f * (mincomp + v);
  float s;
  if (l <= 0.0f) {
    return float3(0.0f);
  }
  if (l <= 0.5f) {
    s = (v - mincomp) / (v + mincomp);
  }
  else {
    s = (v - mincomp) / (2.0f - (v + mincomp));
  }
  return float3(h, s, l);
}
inline float3 osl_yiq_to_rgb(float3 c)
{
  return float3(c.x + 0.9557f * c.y + 0.6199f * c.z,
                c.x - 0.2716f * c.y - 0.6469f * c.z,
                c.x - 1.1082f * c.y + 1.7051f * c.z);
}
inline float3 osl_rgb_to_yiq(float3 c)
{
  return float3(0.299f * c.x + 0.587f * c.y + 0.114f * c.z,
                0.596f * c.x - 0.275f * c.y - 0.321f * c.z,
                0.212f * c.x - 0.523f * c.y + 0.311f * c.z);
}
inline float3 osl_xyy_to_xyz(float3 c)
{
  const float n = c.z / ((c.y == 0.0f) ? 1.0e-37f : c.y);
  return float3(c.x * n, c.z, (1.0f - c.x - c.y) * n);
}
inline float3 osl_xyz_to_xyy(float3 c)
{
  const float n = c.x + c.y + c.z;
  const float n_inv = (n == 0.0f) ? 0.0f : 1.0f / n;
  return float3(c.x * n_inv, c.y * n_inv, c.y);
}
inline float osl_srgb_to_linear(float x)
{
  return (x <= 0.04045f) ? (x * (1.0f / 12.92f)) : pow((x + 0.055f) * (1.0f / 1.055f), 2.4f);
}
inline float osl_linear_to_srgb(float x)
{
  return (x <= 0.0031308f) ? (12.92f * x) : (1.055f * pow(x, 1.0f / 2.4f) - 0.055f);
}
inline float3 osl_srgb_to_linear(float3 c)
{
  return float3(osl_srgb_to_linear(c.x), osl_srgb_to_linear(c.y), osl_srgb_to_linear(c.z));
}
inline float3 osl_linear_to_srgb(float3 c)
{
  return float3(osl_linear_to_srgb(c.x), osl_linear_to_srgb(c.y), osl_linear_to_srgb(c.z));
}
/* The rows of the matrix are given as vectors. */
inline float3 osl_color_matrix(float3 c, float3 r, float3 g, float3 b)
{
  return float3(dot(c, r), dot(c, g), dot(c, b));
}
)MSL";

/* Perlin noise, with derivatives when the type is dual. */
static const char *osl_camera_msl_prelude_3 = R"MSL(
template<typename T> inline OslDual<T> operator+(OslDual<T> a, OslDual<T> b)
{
  return osl_add(a, b);
}
template<typename T> inline OslDual<T> operator-(OslDual<T> a, OslDual<T> b)
{
  return osl_sub(a, b);
}
template<typename T> inline OslDual<T> operator*(OslDual<T> a, OslDual<T> b)
{
  return osl_mul(a, b);
}
template<typename T> inline OslDual<T> operator-(OslDual<T> a)
{
  return osl_neg(a);
}
inline DualF operator*(float a, DualF b)
{
  return DualF{a * b.v, a * b.dx, a * b.dy};
}
inline DualF operator*(DualF a, float b)
{
  return DualF{a.v * b, a.dx * b, a.dy * b};
}
inline DualF operator+(DualF a, float b)
{
  return DualF{a.v + b, a.dx, a.dy};
}
inline DualF operator-(DualF a, float b)
{
  return DualF{a.v - b, a.dx, a.dy};
}
inline DualF operator-(float a, DualF b)
{
  return DualF{a - b.v, -b.dx, -b.dy};
}

inline float osl_val(float x)
{
  return x;
}
inline float osl_val(DualF x)
{
  return x.v;
}
template<typename T> inline T osl_noise_select(bool c, T a, T b)
{
  if (c) {
    return a;
  }
  return b;
}
template<typename T> inline T osl_negate_if(T a, bool c)
{
  if (c) {
    return -a;
  }
  return a;
}
template<typename T> inline T osl_fade(T t)
{
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
template<typename T> inline T osl_lerp(T a, T b, T u)
{
  return a * (1.0f - u) + b * u;
}
template<typename T> inline T osl_bilerp(T v0, T v1, T v2, T v3, T s, T t)
{
  const T s1 = 1.0f - s;
  return (1.0f - t) * (v0 * s1 + v1 * s) + t * (v2 * s1 + v3 * s);
}
template<typename T>
inline T osl_trilerp(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T s, T t, T r)
{
  const T s1 = 1.0f - s;
  const T t1 = 1.0f - t;
  const T r1 = 1.0f - r;
  return r1 * (t1 * (v0 * s1 + v1 * s) + t * (v2 * s1 + v3 * s)) +
         r * (t1 * (v4 * s1 + v5 * s) + t * (v6 * s1 + v7 * s));
}

/* Gradients: the dot product with a pseudo-random vector chosen by the hash. */
template<typename T> inline T osl_grad(int hash, T x)
{
  const int h = hash & 15;
  float g = float(1 + (h & 7));
  if (h & 8) {
    g = -g;
  }
  return g * x;
}
template<typename T> inline T osl_grad(int hash, T x, T y)
{
  const int h = hash & 7;
  const T u = osl_noise_select(h < 4, x, y);
  const T v = 2.0f * osl_noise_select(h < 4, y, x);
  return osl_negate_if(u, (h & 1) != 0) + osl_negate_if(v, (h & 2) != 0);
}
template<typename T> inline T osl_grad(int hash, T x, T y, T z)
{
  const int h = hash & 15;
  const T u = osl_noise_select(h < 8, x, y);
  const T v = osl_noise_select(h < 4, y, osl_noise_select(h == 12 || h == 14, x, z));
  return osl_negate_if(u, (h & 1) != 0) + osl_negate_if(v, (h & 2) != 0);
}
template<typename T> inline T osl_grad(int hash, T x, T y, T z, T w)
{
  const int h = hash & 31;
  const T u = osl_noise_select(h < 24, x, y);
  const T v = osl_noise_select(h < 16, y, z);
  const T s = osl_noise_select(h < 8, z, w);
  return osl_negate_if(u, (h & 1) != 0) + osl_negate_if(v, (h & 2) != 0) +
         osl_negate_if(s, (h & 4) != 0);
}

/* `shift` selects the hash bits of one component of vector valued noise. */
inline int osl_noise_hash(uint h, int shift)
{
  return int((h >> shift) & 0xFFu);
}
#define OSL_PH1(i) osl_noise_hash(osl_inthash(uint(X + i)), shift)
#define OSL_PH2(i, j) osl_noise_hash(osl_inthash(uint(X + i), uint(Y + j)), shift)
#define OSL_PH3(i, j, k) osl_noise_hash(osl_inthash(uint(X + i), uint(Y + j), uint(Z + k)), shift)
#define OSL_PH4(i, j, k, l) \
  osl_noise_hash(osl_inthash(uint(X + i), uint(Y + j), uint(Z + k), uint(W + l)), shift)

template<typename T> inline T osl_perlin(T x, int shift)
{
  const int X = int(floor(osl_val(x)));
  const T fx = x - float(X);
  const T u = osl_fade(fx);
  return 0.25f * osl_lerp(osl_grad(OSL_PH1(0), fx), osl_grad(OSL_PH1(1), fx - 1.0f), u);
}
template<typename T> inline T osl_perlin(T x, T y, int shift)
{
  const int X = int(floor(osl_val(x)));
  const int Y = int(floor(osl_val(y)));
  const T fx = x - float(X);
  const T fy = y - float(Y);
  const T u = osl_fade(fx);
  const T v = osl_fade(fy);
  return 0.6616f * osl_bilerp(osl_grad(OSL_PH2(0, 0), fx, fy),
                              osl_grad(OSL_PH2(1, 0), fx - 1.0f, fy),
                              osl_grad(OSL_PH2(0, 1), fx, fy - 1.0f),
                              osl_grad(OSL_PH2(1, 1), fx - 1.0f, fy - 1.0f),
                              u,
                              v);
}
template<typename T> inline T osl_perlin(T x, T y, T z, int shift)
{
  const int X = int(floor(osl_val(x)));
  const int Y = int(floor(osl_val(y)));
  const int Z = int(floor(osl_val(z)));
  const T fx = x - float(X);
  const T fy = y - float(Y);
  const T fz = z - float(Z);
  const T u = osl_fade(fx);
  const T v = osl_fade(fy);
  const T w = osl_fade(fz);
  return 0.9820f * osl_trilerp(osl_grad(OSL_PH3(0, 0, 0), fx, fy, fz),
                               osl_grad(OSL_PH3(1, 0, 0), fx - 1.0f, fy, fz),
                               osl_grad(OSL_PH3(0, 1, 0), fx, fy - 1.0f, fz),
                               osl_grad(OSL_PH3(1, 1, 0), fx - 1.0f, fy - 1.0f, fz),
                               osl_grad(OSL_PH3(0, 0, 1), fx, fy, fz - 1.0f),
                               osl_grad(OSL_PH3(1, 0, 1), fx - 1.0f, fy, fz - 1.0f),
                               osl_grad(OSL_PH3(0, 1, 1), fx, fy - 1.0f, fz - 1.0f),
                               osl_grad(OSL_PH3(1, 1, 1), fx - 1.0f, fy - 1.0f, fz - 1.0f),
                               u,
                               v,
                               w);
}
template<typename T> inline T osl_perlin(T x, T y, T z, T t, int shift)
{
  const int X = int(floor(osl_val(x)));
  const int Y = int(floor(osl_val(y)));
  const int Z = int(floor(osl_val(z)));
  const int W = int(floor(osl_val(t)));
  const T fx = x - float(X);
  const T fy = y - float(Y);
  const T fz = z - float(Z);
  const T fw = t - float(W);
  const T u = osl_fade(fx);
  const T v = osl_fade(fy);
  const T w = osl_fade(fz);
  const T s = osl_fade(fw);
  const T a = osl_trilerp(osl_grad(OSL_PH4(0, 0, 0, 0), fx, fy, fz, fw),
                          osl_grad(OSL_PH4(1, 0, 0, 0), fx - 1.0f, fy, fz, fw),
                          osl_grad(OSL_PH4(0, 1, 0, 0), fx, fy - 1.0f, fz, fw),
                          osl_grad(OSL_PH4(1, 1, 0, 0), fx - 1.0f, fy - 1.0f, fz, fw),
                          osl_grad(OSL_PH4(0, 0, 1, 0), fx, fy, fz - 1.0f, fw),
                          osl_grad(OSL_PH4(1, 0, 1, 0), fx - 1.0f, fy, fz - 1.0f, fw),
                          osl_grad(OSL_PH4(0, 1, 1, 0), fx, fy - 1.0f, fz - 1.0f, fw),
                          osl_grad(OSL_PH4(1, 1, 1, 0), fx - 1.0f, fy - 1.0f, fz - 1.0f, fw),
                          u,
                          v,
                          w);
  const T b = osl_trilerp(
      osl_grad(OSL_PH4(0, 0, 0, 1), fx, fy, fz, fw - 1.0f),
      osl_grad(OSL_PH4(1, 0, 0, 1), fx - 1.0f, fy, fz, fw - 1.0f),
      osl_grad(OSL_PH4(0, 1, 0, 1), fx, fy - 1.0f, fz, fw - 1.0f),
      osl_grad(OSL_PH4(1, 1, 0, 1), fx - 1.0f, fy - 1.0f, fz, fw - 1.0f),
      osl_grad(OSL_PH4(0, 0, 1, 1), fx, fy, fz - 1.0f, fw - 1.0f),
      osl_grad(OSL_PH4(1, 0, 1, 1), fx - 1.0f, fy, fz - 1.0f, fw - 1.0f),
      osl_grad(OSL_PH4(0, 1, 1, 1), fx, fy - 1.0f, fz - 1.0f, fw - 1.0f),
      osl_grad(OSL_PH4(1, 1, 1, 1), fx - 1.0f, fy - 1.0f, fz - 1.0f, fw - 1.0f),
      u,
      v,
      w);
  return 0.8344f * osl_lerp(a, b, s);
}

inline float osl_noise_unsigned(float a)
{
  return 0.5f * (a + 1.0f);
}
inline DualF osl_noise_unsigned(DualF a)
{
  return DualF{0.5f * (a.v + 1.0f), 0.5f * a.dx, 0.5f * a.dy};
}
inline float3 osl_noise_unsigned(float3 a)
{
  return 0.5f * (a + float3(1.0f));
}
inline DualV osl_noise_unsigned(DualV a)
{
  return DualV{0.5f * (a.v + float3(1.0f)), 0.5f * a.dx, 0.5f * a.dy};
}

/* Signed and unsigned noise of 1 to 4 dimensions, with float or vector result. `F` and `V` are
 * the float and vector types, with or without derivatives. */
#define OSL_PERLIN_NOISE(F, V) \
  inline F osl_snoise_f1(F x) \
  { \
    return osl_perlin(x, 0); \
  } \
  inline F osl_snoise_f2(F x, F y) \
  { \
    return osl_perlin(x, y, 0); \
  } \
  inline F osl_snoise_f3(V p) \
  { \
    return osl_perlin(osl_comp(p, 0), osl_comp(p, 1), osl_comp(p, 2), 0); \
  } \
  inline F osl_snoise_f4(V p, F t) \
  { \
    return osl_perlin(osl_comp(p, 0), osl_comp(p, 1), osl_comp(p, 2), t, 0); \
  } \
  inline V osl_snoise_v1(F x) \
  { \
    return osl_vec(osl_perlin(x, 0), osl_perlin(x, 8), osl_perlin(x, 16)); \
  } \
  inline V osl_snoise_v2(F x, F y) \
  { \
    return osl_vec(osl_perlin(x, y, 0), osl_perlin(x, y, 8), osl_perlin(x, y, 16)); \
  } \
  inline V osl_snoise_v3(V p) \
  { \
    const F x = osl_comp(p, 0), y = osl_comp(p, 1), z = osl_comp(p, 2); \
    return osl_vec(osl_perlin(x, y, z, 0), osl_perlin(x, y, z, 8), osl_perlin(x, y, z, 16)); \
  } \
  inline V osl_snoise_v4(V p, F t) \
  { \
    const F x = osl_comp(p, 0), y = osl_comp(p, 1), z = osl_comp(p, 2); \
    return osl_vec( \
        osl_perlin(x, y, z, t, 0), osl_perlin(x, y, z, t, 8), osl_perlin(x, y, z, t, 16)); \
  } \
  inline F osl_noise_f1(F x) \
  { \
    return osl_noise_unsigned(osl_snoise_f1(x)); \
  } \
  inline F osl_noise_f2(F x, F y) \
  { \
    return osl_noise_unsigned(osl_snoise_f2(x, y)); \
  } \
  inline F osl_noise_f3(V p) \
  { \
    return osl_noise_unsigned(osl_snoise_f3(p)); \
  } \
  inline F osl_noise_f4(V p, F t) \
  { \
    return osl_noise_unsigned(osl_snoise_f4(p, t)); \
  } \
  inline V osl_noise_v1(F x) \
  { \
    return osl_noise_unsigned(osl_snoise_v1(x)); \
  } \
  inline V osl_noise_v2(F x, F y) \
  { \
    return osl_noise_unsigned(osl_snoise_v2(x, y)); \
  } \
  inline V osl_noise_v3(V p) \
  { \
    return osl_noise_unsigned(osl_snoise_v3(p)); \
  } \
  inline V osl_noise_v4(V p, F t) \
  { \
    return osl_noise_unsigned(osl_snoise_v4(p, t)); \
  }

OSL_PERLIN_NOISE(float, float3)
OSL_PERLIN_NOISE(DualF, DualV)
)MSL";

CCL_NAMESPACE_END
