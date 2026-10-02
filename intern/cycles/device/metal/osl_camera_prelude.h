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

/* Numbers that do not fit saturate and NaN is zero, as on the processors that Metal runs on. */
inline int osl_ftoi(float a)
{
  if (a >= 2147483648.0f) {
    return 2147483647;
  }
  if (a <= -2147483648.0f) {
    return -2147483647 - 1;
  }
  return (a != a) ? 0 : int(a);
}
inline int osl_idx(int i, int n)
{
  return clamp(i, 0, n - 1);
}
/* Index into the table of all parts of a string of length `slen`, see substr(). */
inline int osl_substr_index(int start, int len, int slen)
{
  int b = start;
  if (b < 0) {
    b += slen;
  }
  return clamp(b, 0, slen) * (slen + 1) + clamp(len, 0, slen);
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

/* The upper limit is applied first, as in the OSL runtime. */
template<typename T> inline T osl_clamp(T x, T low, T high)
{
  return osl_max(osl_min(x, high), low);
}
inline int osl_clamp(int x, int low, int high)
{
  return max(min(x, high), low);
}
template<typename T> inline T osl_degrees(T x)
{
  return x * T(57.29577951308232f);
}
template<typename T> inline OslDual<T> osl_degrees(OslDual<T> x)
{
  return OslDual<T>{osl_degrees(x.v), osl_degrees(x.dx), osl_degrees(x.dy)};
}
template<typename T> inline T osl_radians(T x)
{
  return x * T(0.017453292519943295f);
}
template<typename T> inline OslDual<T> osl_radians(OslDual<T> x)
{
  return OslDual<T>{osl_radians(x.v), osl_radians(x.dx), osl_radians(x.dy)};
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

/* Position in a period that is a whole number of at least one. */
inline float osl_wrap(float s, float period)
{
  period = floor(period);
  if (period < 1.0f) {
    period = 1.0f;
  }
  return s - period * floor(s / period);
}
inline float3 osl_wrap(float3 s, float3 period)
{
  return float3(osl_wrap(s.x, period.x), osl_wrap(s.y, period.y), osl_wrap(s.z, period.z));
}

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

/* Operators and helpers of the noise functions, which have derivatives when the type is dual. */
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

)MSL";

/* Perlin noise, only for shaders that use it. */
static const char *osl_camera_msl_perlin = R"MSL(
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

/* Periodic noise. */
static const char *osl_camera_msl_periodic = R"MSL(
/* Always in [0, b), also for negative numbers. */
inline int osl_imod(int a, int b)
{
  const int remainder = a % b;
  return (remainder < 0) ? remainder + b : remainder;
}
inline int osl_period(float p)
{
  const int i = int(floor(p));
  return (i < 1) ? 1 : i;
}

#define OSL_PPH1(i) osl_noise_hash(osl_inthash(uint(osl_imod(X + i, px))), shift)
#define OSL_PPH2(i, j) \
  osl_noise_hash(osl_inthash(uint(osl_imod(X + i, px)), uint(osl_imod(Y + j, py))), shift)
#define OSL_PPH3(i, j, k) \
  osl_noise_hash(osl_inthash(uint(osl_imod(X + i, px)), \
                             uint(osl_imod(Y + j, py)), \
                             uint(osl_imod(Z + k, pz))), \
                 shift)
#define OSL_PPH4(i, j, k, l) \
  osl_noise_hash(osl_inthash(uint(osl_imod(X + i, px)), \
                             uint(osl_imod(Y + j, py)), \
                             uint(osl_imod(Z + k, pz)), \
                             uint(osl_imod(W + l, pw))), \
                 shift)

template<typename T> inline T osl_pperlin(T x, int px, int shift)
{
  const int X = int(floor(osl_val(x)));
  const T fx = x - float(X);
  const T u = osl_fade(fx);
  return 0.25f * osl_lerp(osl_grad(OSL_PPH1(0), fx), osl_grad(OSL_PPH1(1), fx - 1.0f), u);
}
template<typename T> inline T osl_pperlin(T x, T y, int px, int py, int shift)
{
  const int X = int(floor(osl_val(x)));
  const int Y = int(floor(osl_val(y)));
  const T fx = x - float(X);
  const T fy = y - float(Y);
  const T u = osl_fade(fx);
  const T v = osl_fade(fy);
  return 0.6616f * osl_bilerp(osl_grad(OSL_PPH2(0, 0), fx, fy),
                              osl_grad(OSL_PPH2(1, 0), fx - 1.0f, fy),
                              osl_grad(OSL_PPH2(0, 1), fx, fy - 1.0f),
                              osl_grad(OSL_PPH2(1, 1), fx - 1.0f, fy - 1.0f),
                              u,
                              v);
}
template<typename T> inline T osl_pperlin(T x, T y, T z, int px, int py, int pz, int shift)
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
  return 0.9820f * osl_trilerp(osl_grad(OSL_PPH3(0, 0, 0), fx, fy, fz),
                               osl_grad(OSL_PPH3(1, 0, 0), fx - 1.0f, fy, fz),
                               osl_grad(OSL_PPH3(0, 1, 0), fx, fy - 1.0f, fz),
                               osl_grad(OSL_PPH3(1, 1, 0), fx - 1.0f, fy - 1.0f, fz),
                               osl_grad(OSL_PPH3(0, 0, 1), fx, fy, fz - 1.0f),
                               osl_grad(OSL_PPH3(1, 0, 1), fx - 1.0f, fy, fz - 1.0f),
                               osl_grad(OSL_PPH3(0, 1, 1), fx, fy - 1.0f, fz - 1.0f),
                               osl_grad(OSL_PPH3(1, 1, 1), fx - 1.0f, fy - 1.0f, fz - 1.0f),
                               u,
                               v,
                               w);
}
template<typename T>
inline T osl_pperlin(T x, T y, T z, T t, int px, int py, int pz, int pw, int shift)
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
  const T a = osl_trilerp(osl_grad(OSL_PPH4(0, 0, 0, 0), fx, fy, fz, fw),
                          osl_grad(OSL_PPH4(1, 0, 0, 0), fx - 1.0f, fy, fz, fw),
                          osl_grad(OSL_PPH4(0, 1, 0, 0), fx, fy - 1.0f, fz, fw),
                          osl_grad(OSL_PPH4(1, 1, 0, 0), fx - 1.0f, fy - 1.0f, fz, fw),
                          osl_grad(OSL_PPH4(0, 0, 1, 0), fx, fy, fz - 1.0f, fw),
                          osl_grad(OSL_PPH4(1, 0, 1, 0), fx - 1.0f, fy, fz - 1.0f, fw),
                          osl_grad(OSL_PPH4(0, 1, 1, 0), fx, fy - 1.0f, fz - 1.0f, fw),
                          osl_grad(OSL_PPH4(1, 1, 1, 0), fx - 1.0f, fy - 1.0f, fz - 1.0f, fw),
                          u,
                          v,
                          w);
  const T b = osl_trilerp(
      osl_grad(OSL_PPH4(0, 0, 0, 1), fx, fy, fz, fw - 1.0f),
      osl_grad(OSL_PPH4(1, 0, 0, 1), fx - 1.0f, fy, fz, fw - 1.0f),
      osl_grad(OSL_PPH4(0, 1, 0, 1), fx, fy - 1.0f, fz, fw - 1.0f),
      osl_grad(OSL_PPH4(1, 1, 0, 1), fx - 1.0f, fy - 1.0f, fz, fw - 1.0f),
      osl_grad(OSL_PPH4(0, 0, 1, 1), fx, fy, fz - 1.0f, fw - 1.0f),
      osl_grad(OSL_PPH4(1, 0, 1, 1), fx - 1.0f, fy, fz - 1.0f, fw - 1.0f),
      osl_grad(OSL_PPH4(0, 1, 1, 1), fx, fy - 1.0f, fz - 1.0f, fw - 1.0f),
      osl_grad(OSL_PPH4(1, 1, 1, 1), fx - 1.0f, fy - 1.0f, fz - 1.0f, fw - 1.0f),
      u,
      v,
      w);
  return 0.8344f * osl_lerp(a, b, s);
}

/* Signed and unsigned periodic Perlin noise. Periods have no derivatives. */
#define OSL_PERIODIC_PERLIN_NOISE(F, V) \
  inline F osl_psnoise_f1(F x, float px) \
  { \
    return osl_pperlin(x, osl_period(px), 0); \
  } \
  inline F osl_psnoise_f2(F x, F y, float px, float py) \
  { \
    return osl_pperlin(x, y, osl_period(px), osl_period(py), 0); \
  } \
  inline F osl_psnoise_f3(V p, float3 pp) \
  { \
    return osl_pperlin(osl_comp(p, 0), osl_comp(p, 1), osl_comp(p, 2), osl_period(pp.x), \
                       osl_period(pp.y), osl_period(pp.z), 0); \
  } \
  inline F osl_psnoise_f4(V p, F t, float3 pp, float pt) \
  { \
    return osl_pperlin(osl_comp(p, 0), osl_comp(p, 1), osl_comp(p, 2), t, osl_period(pp.x), \
                       osl_period(pp.y), osl_period(pp.z), osl_period(pt), 0); \
  } \
  inline V osl_psnoise_v1(F x, float px) \
  { \
    const int ix = osl_period(px); \
    return osl_vec(osl_pperlin(x, ix, 0), osl_pperlin(x, ix, 8), osl_pperlin(x, ix, 16)); \
  } \
  inline V osl_psnoise_v2(F x, F y, float px, float py) \
  { \
    const int ix = osl_period(px), iy = osl_period(py); \
    return osl_vec(osl_pperlin(x, y, ix, iy, 0), osl_pperlin(x, y, ix, iy, 8), \
                   osl_pperlin(x, y, ix, iy, 16)); \
  } \
  inline V osl_psnoise_v3(V p, float3 pp) \
  { \
    const F x = osl_comp(p, 0), y = osl_comp(p, 1), z = osl_comp(p, 2); \
    const int ix = osl_period(pp.x), iy = osl_period(pp.y), iz = osl_period(pp.z); \
    return osl_vec(osl_pperlin(x, y, z, ix, iy, iz, 0), osl_pperlin(x, y, z, ix, iy, iz, 8), \
                   osl_pperlin(x, y, z, ix, iy, iz, 16)); \
  } \
  inline V osl_psnoise_v4(V p, F t, float3 pp, float pt) \
  { \
    const F x = osl_comp(p, 0), y = osl_comp(p, 1), z = osl_comp(p, 2); \
    const int ix = osl_period(pp.x), iy = osl_period(pp.y), iz = osl_period(pp.z); \
    const int it = osl_period(pt); \
    return osl_vec(osl_pperlin(x, y, z, t, ix, iy, iz, it, 0), \
                   osl_pperlin(x, y, z, t, ix, iy, iz, it, 8), \
                   osl_pperlin(x, y, z, t, ix, iy, iz, it, 16)); \
  } \
  inline F osl_pnoise_f1(F x, float px) \
  { \
    return osl_noise_unsigned(osl_psnoise_f1(x, px)); \
  } \
  inline F osl_pnoise_f2(F x, F y, float px, float py) \
  { \
    return osl_noise_unsigned(osl_psnoise_f2(x, y, px, py)); \
  } \
  inline F osl_pnoise_f3(V p, float3 pp) \
  { \
    return osl_noise_unsigned(osl_psnoise_f3(p, pp)); \
  } \
  inline F osl_pnoise_f4(V p, F t, float3 pp, float pt) \
  { \
    return osl_noise_unsigned(osl_psnoise_f4(p, t, pp, pt)); \
  } \
  inline V osl_pnoise_v1(F x, float px) \
  { \
    return osl_noise_unsigned(osl_psnoise_v1(x, px)); \
  } \
  inline V osl_pnoise_v2(F x, F y, float px, float py) \
  { \
    return osl_noise_unsigned(osl_psnoise_v2(x, y, px, py)); \
  } \
  inline V osl_pnoise_v3(V p, float3 pp) \
  { \
    return osl_noise_unsigned(osl_psnoise_v3(p, pp)); \
  } \
  inline V osl_pnoise_v4(V p, F t, float3 pp, float pt) \
  { \
    return osl_noise_unsigned(osl_psnoise_v4(p, t, pp, pt)); \
  }

OSL_PERIODIC_PERLIN_NOISE(float, float3)
OSL_PERIODIC_PERLIN_NOISE(DualF, DualV)

/* Periodic cell and hash noise wrap the position. */
#define OSL_PERIODIC_HASH_NOISE(name, base) \
  inline float name##_f1(float x, float px) \
  { \
    return base##_f1(osl_wrap(x, px)); \
  } \
  inline float name##_f2(float x, float y, float px, float py) \
  { \
    return base##_f2(osl_wrap(x, px), osl_wrap(y, py)); \
  } \
  inline float name##_f3(float3 p, float3 pp) \
  { \
    return base##_f3(osl_wrap(p, pp)); \
  } \
  inline float name##_f4(float3 p, float t, float3 pp, float pt) \
  { \
    return base##_f4(osl_wrap(p, pp), osl_wrap(t, pt)); \
  } \
  inline float3 name##_v1(float x, float px) \
  { \
    return base##_v1(osl_wrap(x, px)); \
  } \
  inline float3 name##_v2(float x, float y, float px, float py) \
  { \
    return base##_v2(osl_wrap(x, px), osl_wrap(y, py)); \
  } \
  inline float3 name##_v3(float3 p, float3 pp) \
  { \
    return base##_v3(osl_wrap(p, pp)); \
  } \
  inline float3 name##_v4(float3 p, float t, float3 pp, float pt) \
  { \
    return base##_v4(osl_wrap(p, pp), osl_wrap(t, pt)); \
  }

OSL_PERIODIC_HASH_NOISE(osl_pcellnoise, osl_cellnoise)
OSL_PERIODIC_HASH_NOISE(osl_phashnoise, osl_hashnoise)
)MSL";

/* Simplex noise, after `sfm_simplex.h` of Open Shading Language (BSD-3-Clause), which is
 * based on the public domain code of Stefan Gustavson. */
static const char *osl_camera_msl_simplex = R"MSL(
constant float2 osl_simplex_grad2[8] = {
    float2(-1.0f, -1.0f), float2(1.0f, 0.0f), float2(-1.0f, 0.0f), float2(1.0f, 1.0f),
    float2(-1.0f, 1.0f), float2(0.0f, -1.0f), float2(0.0f, 1.0f), float2(1.0f, -1.0f)};
constant float3 osl_simplex_grad3[16] = {
    float3(1.0f, 0.0f, 1.0f), float3(0.0f, 1.0f, 1.0f), float3(-1.0f, 0.0f, 1.0f),
    float3(0.0f, -1.0f, 1.0f), float3(1.0f, 0.0f, -1.0f), float3(0.0f, 1.0f, -1.0f),
    float3(-1.0f, 0.0f, -1.0f), float3(0.0f, -1.0f, -1.0f), float3(1.0f, -1.0f, 0.0f),
    float3(1.0f, 1.0f, 0.0f), float3(-1.0f, 1.0f, 0.0f), float3(-1.0f, -1.0f, 0.0f),
    float3(1.0f, 0.0f, 1.0f), float3(-1.0f, 0.0f, 1.0f), float3(0.0f, 1.0f, -1.0f),
    float3(0.0f, -1.0f, -1.0f)};
constant float4 osl_simplex_grad4[32] = {
    float4(0.0f, 1.0f, 1.0f, 1.0f), float4(0.0f, 1.0f, 1.0f, -1.0f),
    float4(0.0f, 1.0f, -1.0f, 1.0f), float4(0.0f, 1.0f, -1.0f, -1.0f),
    float4(0.0f, -1.0f, 1.0f, 1.0f), float4(0.0f, -1.0f, 1.0f, -1.0f),
    float4(0.0f, -1.0f, -1.0f, 1.0f), float4(0.0f, -1.0f, -1.0f, -1.0f),
    float4(1.0f, 0.0f, 1.0f, 1.0f), float4(1.0f, 0.0f, 1.0f, -1.0f),
    float4(1.0f, 0.0f, -1.0f, 1.0f), float4(1.0f, 0.0f, -1.0f, -1.0f),
    float4(-1.0f, 0.0f, 1.0f, 1.0f), float4(-1.0f, 0.0f, 1.0f, -1.0f),
    float4(-1.0f, 0.0f, -1.0f, 1.0f), float4(-1.0f, 0.0f, -1.0f, -1.0f),
    float4(1.0f, 1.0f, 0.0f, 1.0f), float4(1.0f, 1.0f, 0.0f, -1.0f),
    float4(1.0f, -1.0f, 0.0f, 1.0f), float4(1.0f, -1.0f, 0.0f, -1.0f),
    float4(-1.0f, 1.0f, 0.0f, 1.0f), float4(-1.0f, 1.0f, 0.0f, -1.0f),
    float4(-1.0f, -1.0f, 0.0f, 1.0f), float4(-1.0f, -1.0f, 0.0f, -1.0f),
    float4(1.0f, 1.0f, 1.0f, 0.0f), float4(1.0f, 1.0f, -1.0f, 0.0f),
    float4(1.0f, -1.0f, 1.0f, 0.0f), float4(1.0f, -1.0f, -1.0f, 0.0f),
    float4(-1.0f, 1.0f, 1.0f, 0.0f), float4(-1.0f, 1.0f, -1.0f, 0.0f),
    float4(-1.0f, -1.0f, 1.0f, 0.0f), float4(-1.0f, -1.0f, -1.0f, 0.0f)};
/* The order in which the corners of a 4D simplex are traversed. */
constant uchar4 osl_simplex_lut[64] = {
    uchar4(0, 1, 2, 3), uchar4(0, 1, 3, 2), uchar4(0, 0, 0, 0), uchar4(0, 2, 3, 1),
    uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(1, 2, 3, 0),
    uchar4(0, 2, 1, 3), uchar4(0, 0, 0, 0), uchar4(0, 3, 1, 2), uchar4(0, 3, 2, 1),
    uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(1, 3, 2, 0),
    uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0),
    uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0),
    uchar4(1, 2, 0, 3), uchar4(0, 0, 0, 0), uchar4(1, 3, 0, 2), uchar4(0, 0, 0, 0),
    uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(2, 3, 0, 1), uchar4(2, 3, 1, 0),
    uchar4(1, 0, 2, 3), uchar4(1, 0, 3, 2), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0),
    uchar4(0, 0, 0, 0), uchar4(2, 0, 3, 1), uchar4(0, 0, 0, 0), uchar4(2, 1, 3, 0),
    uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0),
    uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0),
    uchar4(2, 0, 1, 3), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0),
    uchar4(3, 0, 1, 2), uchar4(3, 0, 2, 1), uchar4(0, 0, 0, 0), uchar4(3, 1, 2, 0),
    uchar4(2, 1, 0, 3), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0), uchar4(0, 0, 0, 0),
    uchar4(3, 1, 0, 2), uchar4(0, 0, 0, 0), uchar4(3, 2, 0, 1), uchar4(3, 2, 1, 0)};

inline uint osl_scramble(uint v0, uint v1, uint v2)
{
  return osl_bjfinal(v0, v1, v2 ^ 0xdeadbeefu);
}
inline float osl_simplex_grad(int i, int seed)
{
  const uint h = osl_scramble(uint(i), uint(seed), 0u);
  const float g = 1.0f + float(h & 7u);
  return (h & 8u) ? -g : g;
}
inline float2 osl_simplex_grad(int i, int j, int seed)
{
  return osl_simplex_grad2[osl_scramble(uint(i), uint(j), uint(seed)) & 7u];
}
inline float3 osl_simplex_grad(int i, int j, int k, int seed)
{
  return osl_simplex_grad3[
      osl_scramble(uint(i), uint(j), osl_scramble(uint(k), uint(seed), 0u)) & 15u];
}
inline float4 osl_simplex_grad(int i, int j, int k, int l, int seed)
{
  return osl_simplex_grad4[
      osl_scramble(uint(i), uint(j), osl_scramble(uint(k), uint(l), uint(seed))) & 31u];
}

/* The noise and its derivative by each coordinate. */
struct OslSimplex {
  float n;
  float4 d;
};

inline OslSimplex osl_simplex(float x, int seed)
{
  const int i0 = int(floor(x));
  const int i1 = i0 + 1;
  const float x0 = x - float(i0);
  const float x1 = x0 - 1.0f;
  const float x20 = x0 * x0;
  const float t0 = 1.0f - x20;
  const float t20 = t0 * t0;
  const float t40 = t20 * t20;
  const float gx0 = osl_simplex_grad(i0, seed);
  const float n0 = t40 * gx0 * x0;
  const float x21 = x1 * x1;
  const float t1 = 1.0f - x21;
  const float t21 = t1 * t1;
  const float t41 = t21 * t21;
  const float gx1 = osl_simplex_grad(i1, seed);
  const float n1 = t41 * gx1 * x1;
  const float scale = 0.36f;
  float dx = t20 * t0 * gx0 * x20;
  dx += t21 * t1 * gx1 * x21;
  dx *= -8.0f;
  dx += t40 * gx0 + t41 * gx1;
  dx *= scale;
  return OslSimplex{scale * (n0 + n1), float4(dx, 0.0f, 0.0f, 0.0f)};
}

inline OslSimplex osl_simplex(float x, float y, int seed)
{
  const float F2 = 0.366025403f;
  const float G2 = 0.211324865f;
  const float s = (x + y) * F2;
  const float xs = x + s;
  const float ys = y + s;
  const int i = int(floor(xs));
  const int j = int(floor(ys));
  const float t = float(i + j) * G2;
  const float X0 = float(i) - t;
  const float Y0 = float(j) - t;
  const float x0 = x - X0;
  const float y0 = y - Y0;
  const int i1 = int(x0 > y0);
  const int j1 = 1 - i1;
  const float x1 = x0 - float(i1) + G2;
  const float y1 = y0 - float(j1) + G2;
  const float x2 = x0 - 1.0f + 2.0f * G2;
  const float y2 = y0 - 1.0f + 2.0f * G2;

  const float t0 = 0.5f - x0 * x0 - y0 * y0;
  const float t20 = t0 * t0;
  const float t40 = t20 * t20;
  float2 g0 = osl_simplex_grad(i, j, seed);
  float n0 = t40 * (g0.x * x0 + g0.y * y0);
  if (t0 < 0.0f) {
    n0 = 0.0f;
    g0 = float2(0.0f);
  }
  const float t1 = 0.5f - x1 * x1 - y1 * y1;
  const float t21 = t1 * t1;
  const float t41 = t21 * t21;
  float2 g1 = osl_simplex_grad(i + i1, j + j1, seed);
  float n1 = t41 * (g1.x * x1 + g1.y * y1);
  if (t1 < 0.0f) {
    n1 = 0.0f;
    g1 = float2(0.0f);
  }
  const float t2 = 0.5f - x2 * x2 - y2 * y2;
  const float t22 = t2 * t2;
  const float t42 = t22 * t22;
  float2 g2 = osl_simplex_grad(i + 1, j + 1, seed);
  float n2 = t42 * (g2.x * x2 + g2.y * y2);
  if (t2 < 0.0f) {
    n2 = 0.0f;
    g2 = float2(0.0f);
  }
  const float scale = 64.0f;
  const float temp0 = t20 * t0 * (g0.x * x0 + g0.y * y0);
  const float temp1 = t21 * t1 * (g1.x * x1 + g1.y * y1);
  const float temp2 = t22 * t2 * (g2.x * x2 + g2.y * y2);
  float2 d = float2(temp0 * x0, temp0 * y0) + float2(temp1 * x1, temp1 * y1) +
             float2(temp2 * x2, temp2 * y2);
  d *= -8.0f;
  d += t40 * g0 + t41 * g1 + t42 * g2;
  d *= scale;
  return OslSimplex{scale * (n0 + n1 + n2), float4(d.x, d.y, 0.0f, 0.0f)};
}

inline OslSimplex osl_simplex(float x, float y, float z, int seed)
{
  const float F3 = 0.333333333f;
  const float G3 = 0.166666667f;
  const float s = (x + y + z) * F3;
  const float xs = x + s;
  const float ys = y + s;
  const float zs = z + s;
  const int i = int(floor(xs));
  const int j = int(floor(ys));
  const int k = int(floor(zs));
  const float t = float(i + j + k) * G3;
  const float X0 = float(i) - t;
  const float Y0 = float(j) - t;
  const float Z0 = float(k) - t;
  const float3 p0 = float3(x - X0, y - Y0, z - Z0);

  const int bg0 = int(p0.x >= p0.y);
  const int bg1 = int(p0.y >= p0.z);
  const int bg2 = int(p0.x >= p0.z);
  const int nbg0 = 1 - bg0;
  const int nbg1 = 1 - bg1;
  const int nbg2 = 1 - bg2;
  const int i1 = bg0 & (bg1 | bg2);
  const int j1 = nbg0 & bg1;
  const int k1 = nbg1 & ((bg0 & nbg2) | nbg0);
  const int i2 = bg0 | (bg1 & bg2);
  const int j2 = bg1 | nbg0;
  const int k2 = (bg0 & nbg1) | (nbg0 & (nbg1 | nbg2));

  const float3 p1 = p0 - float3(float(i1), float(j1), float(k1)) + G3;
  const float3 p2 = p0 - float3(float(i2), float(j2), float(k2)) + 2.0f * G3;
  const float3 p3 = p0 - 1.0f + 3.0f * G3;

  const float t0 = 0.5f - p0.x * p0.x - p0.y * p0.y - p0.z * p0.z;
  const float t20 = t0 * t0;
  const float t40 = t20 * t20;
  float3 g0 = osl_simplex_grad(i, j, k, seed);
  float n0 = t40 * (g0.x * p0.x + g0.y * p0.y + g0.z * p0.z);
  if (t0 < 0.0f) {
    n0 = 0.0f;
    g0 = float3(0.0f);
  }
  const float t1 = 0.5f - p1.x * p1.x - p1.y * p1.y - p1.z * p1.z;
  const float t21 = t1 * t1;
  const float t41 = t21 * t21;
  float3 g1 = osl_simplex_grad(i + i1, j + j1, k + k1, seed);
  float n1 = t41 * (g1.x * p1.x + g1.y * p1.y + g1.z * p1.z);
  if (t1 < 0.0f) {
    n1 = 0.0f;
    g1 = float3(0.0f);
  }
  const float t2 = 0.5f - p2.x * p2.x - p2.y * p2.y - p2.z * p2.z;
  const float t22 = t2 * t2;
  const float t42 = t22 * t22;
  float3 g2 = osl_simplex_grad(i + i2, j + j2, k + k2, seed);
  float n2 = t42 * (g2.x * p2.x + g2.y * p2.y + g2.z * p2.z);
  if (t2 < 0.0f) {
    n2 = 0.0f;
    g2 = float3(0.0f);
  }
  const float t3 = 0.5f - p3.x * p3.x - p3.y * p3.y - p3.z * p3.z;
  const float t23 = t3 * t3;
  const float t43 = t23 * t23;
  float3 g3 = osl_simplex_grad(i + 1, j + 1, k + 1, seed);
  float n3 = t43 * (g3.x * p3.x + g3.y * p3.y + g3.z * p3.z);
  if (t3 < 0.0f) {
    n3 = 0.0f;
    g3 = float3(0.0f);
  }
  const float scale = 68.0f;
  const float temp0 = t20 * t0 * (g0.x * p0.x + g0.y * p0.y + g0.z * p0.z);
  const float temp1 = t21 * t1 * (g1.x * p1.x + g1.y * p1.y + g1.z * p1.z);
  const float temp2 = t22 * t2 * (g2.x * p2.x + g2.y * p2.y + g2.z * p2.z);
  const float temp3 = t23 * t3 * (g3.x * p3.x + g3.y * p3.y + g3.z * p3.z);
  float3 d = temp0 * p0 + temp1 * p1 + temp2 * p2 + temp3 * p3;
  d *= -8.0f;
  d += t40 * g0 + t41 * g1 + t42 * g2 + t43 * g3;
  d *= scale;
  return OslSimplex{scale * (n0 + n1 + n2 + n3), float4(d.x, d.y, d.z, 0.0f)};
}

inline OslSimplex osl_simplex(float x, float y, float z, float w, int seed)
{
  const float F4 = 0.309016994f;
  const float G4 = 0.138196601f;
  const float s = (x + y + z + w) * F4;
  const float xs = x + s;
  const float ys = y + s;
  const float zs = z + s;
  const float ws = w + s;
  const int i = int(floor(xs));
  const int j = int(floor(ys));
  const int k = int(floor(zs));
  const int l = int(floor(ws));
  const float t = float(i + j + k + l) * G4;
  const float4 p0 = float4(x - (float(i) - t), y - (float(j) - t), z - (float(k) - t),
                           w - (float(l) - t));

  const int c1 = (p0.x > p0.y) ? 32 : 0;
  const int c2 = (p0.x > p0.z) ? 16 : 0;
  const int c3 = (p0.y > p0.z) ? 8 : 0;
  const int c4 = (p0.x > p0.w) ? 4 : 0;
  const int c5 = (p0.y > p0.w) ? 2 : 0;
  const int c6 = (p0.z > p0.w) ? 1 : 0;
  const uchar4 order = osl_simplex_lut[c1 | c2 | c3 | c4 | c5 | c6];
  const int4 o1 = int4(order >= uchar4(3));
  const int4 o2 = int4(order >= uchar4(2));
  const int4 o3 = int4(order >= uchar4(1));

  const float4 p1 = p0 - float4(o1) + G4;
  const float4 p2 = p0 - float4(o2) + 2.0f * G4;
  const float4 p3 = p0 - float4(o3) + 3.0f * G4;
  const float4 p4 = p0 - 1.0f + 4.0f * G4;

  const float t0 = 0.5f - p0.x * p0.x - p0.y * p0.y - p0.z * p0.z - p0.w * p0.w;
  const float t20 = t0 * t0;
  const float t40 = t20 * t20;
  float4 g0 = osl_simplex_grad(i, j, k, l, seed);
  float n0 = t40 * (g0.x * p0.x + g0.y * p0.y + g0.z * p0.z + g0.w * p0.w);
  if (t0 < 0.0f) {
    n0 = 0.0f;
    g0 = float4(0.0f);
  }
  const float t1 = 0.5f - p1.x * p1.x - p1.y * p1.y - p1.z * p1.z - p1.w * p1.w;
  const float t21 = t1 * t1;
  const float t41 = t21 * t21;
  float4 g1 = osl_simplex_grad(i + o1.x, j + o1.y, k + o1.z, l + o1.w, seed);
  float n1 = t41 * (g1.x * p1.x + g1.y * p1.y + g1.z * p1.z + g1.w * p1.w);
  if (t1 < 0.0f) {
    n1 = 0.0f;
    g1 = float4(0.0f);
  }
  const float t2 = 0.5f - p2.x * p2.x - p2.y * p2.y - p2.z * p2.z - p2.w * p2.w;
  const float t22 = t2 * t2;
  const float t42 = t22 * t22;
  float4 g2 = osl_simplex_grad(i + o2.x, j + o2.y, k + o2.z, l + o2.w, seed);
  float n2 = t42 * (g2.x * p2.x + g2.y * p2.y + g2.z * p2.z + g2.w * p2.w);
  if (t2 < 0.0f) {
    n2 = 0.0f;
    g2 = float4(0.0f);
  }
  const float t3 = 0.5f - p3.x * p3.x - p3.y * p3.y - p3.z * p3.z - p3.w * p3.w;
  const float t23 = t3 * t3;
  const float t43 = t23 * t23;
  float4 g3 = osl_simplex_grad(i + o3.x, j + o3.y, k + o3.z, l + o3.w, seed);
  float n3 = t43 * (g3.x * p3.x + g3.y * p3.y + g3.z * p3.z + g3.w * p3.w);
  if (t3 < 0.0f) {
    n3 = 0.0f;
    g3 = float4(0.0f);
  }
  const float t4 = 0.5f - p4.x * p4.x - p4.y * p4.y - p4.z * p4.z - p4.w * p4.w;
  const float t24 = t4 * t4;
  const float t44 = t24 * t24;
  float4 g4 = osl_simplex_grad(i + 1, j + 1, k + 1, l + 1, seed);
  float n4 = t44 * (g4.x * p4.x + g4.y * p4.y + g4.z * p4.z + g4.w * p4.w);
  if (t4 < 0.0f) {
    n4 = 0.0f;
    g4 = float4(0.0f);
  }
  const float scale = 54.0f;
  const float temp0 = t20 * t0 * (g0.x * p0.x + g0.y * p0.y + g0.z * p0.z + g0.w * p0.w);
  const float temp1 = t21 * t1 * (g1.x * p1.x + g1.y * p1.y + g1.z * p1.z + g1.w * p1.w);
  const float temp2 = t22 * t2 * (g2.x * p2.x + g2.y * p2.y + g2.z * p2.z + g2.w * p2.w);
  const float temp3 = t23 * t3 * (g3.x * p3.x + g3.y * p3.y + g3.z * p3.z + g3.w * p3.w);
  const float temp4 = t24 * t4 * (g4.x * p4.x + g4.y * p4.y + g4.z * p4.z + g4.w * p4.w);
  float4 d = temp0 * p0 + temp1 * p1 + temp2 * p2 + temp3 * p3 + temp4 * p4;
  d *= -8.0f;
  d += t40 * g0 + t41 * g1 + t42 * g2 + t43 * g3 + t44 * g4;
  d *= scale;
  return OslSimplex{scale * (n0 + n1 + n2 + n3 + n4), d};
}

/* With derivatives by the chain rule. */
inline float osl_sx(float x, int seed)
{
  return osl_simplex(x, seed).n;
}
inline DualF osl_sx(DualF x, int seed)
{
  const OslSimplex r = osl_simplex(x.v, seed);
  return DualF{r.n, r.d.x * x.dx, r.d.x * x.dy};
}
inline float osl_sx(float x, float y, int seed)
{
  return osl_simplex(x, y, seed).n;
}
inline DualF osl_sx(DualF x, DualF y, int seed)
{
  const OslSimplex r = osl_simplex(x.v, y.v, seed);
  return DualF{r.n, r.d.x * x.dx + r.d.y * y.dx, r.d.x * x.dy + r.d.y * y.dy};
}
inline float osl_sx(float3 p, int seed)
{
  return osl_simplex(p.x, p.y, p.z, seed).n;
}
inline DualF osl_sx(DualV p, int seed)
{
  const OslSimplex r = osl_simplex(p.v.x, p.v.y, p.v.z, seed);
  return DualF{r.n, dot(r.d.xyz, p.dx), dot(r.d.xyz, p.dy)};
}
inline float osl_sx(float3 p, float t, int seed)
{
  return osl_simplex(p.x, p.y, p.z, t, seed).n;
}
inline DualF osl_sx(DualV p, DualF t, int seed)
{
  const OslSimplex r = osl_simplex(p.v.x, p.v.y, p.v.z, t.v, seed);
  return DualF{r.n, dot(r.d.xyz, p.dx) + r.d.w * t.dx, dot(r.d.xyz, p.dy) + r.d.w * t.dy};
}

#define OSL_SIMPLEX_NOISE(F, V) \
  inline F osl_simplexnoise_f1(F x) \
  { \
    return osl_sx(x, 0); \
  } \
  inline F osl_simplexnoise_f2(F x, F y) \
  { \
    return osl_sx(x, y, 0); \
  } \
  inline F osl_simplexnoise_f3(V p) \
  { \
    return osl_sx(p, 0); \
  } \
  inline F osl_simplexnoise_f4(V p, F t) \
  { \
    return osl_sx(p, t, 0); \
  } \
  inline V osl_simplexnoise_v1(F x) \
  { \
    return osl_vec(osl_sx(x, 0), osl_sx(x, 1), osl_sx(x, 2)); \
  } \
  inline V osl_simplexnoise_v2(F x, F y) \
  { \
    return osl_vec(osl_sx(x, y, 0), osl_sx(x, y, 1), osl_sx(x, y, 2)); \
  } \
  inline V osl_simplexnoise_v3(V p) \
  { \
    return osl_vec(osl_sx(p, 0), osl_sx(p, 1), osl_sx(p, 2)); \
  } \
  inline V osl_simplexnoise_v4(V p, F t) \
  { \
    return osl_vec(osl_sx(p, t, 0), osl_sx(p, t, 1), osl_sx(p, t, 2)); \
  } \
  inline F osl_usimplexnoise_f1(F x) \
  { \
    return osl_noise_unsigned(osl_simplexnoise_f1(x)); \
  } \
  inline F osl_usimplexnoise_f2(F x, F y) \
  { \
    return osl_noise_unsigned(osl_simplexnoise_f2(x, y)); \
  } \
  inline F osl_usimplexnoise_f3(V p) \
  { \
    return osl_noise_unsigned(osl_simplexnoise_f3(p)); \
  } \
  inline F osl_usimplexnoise_f4(V p, F t) \
  { \
    return osl_noise_unsigned(osl_simplexnoise_f4(p, t)); \
  } \
  inline V osl_usimplexnoise_v1(F x) \
  { \
    return osl_noise_unsigned(osl_simplexnoise_v1(x)); \
  } \
  inline V osl_usimplexnoise_v2(F x, F y) \
  { \
    return osl_noise_unsigned(osl_simplexnoise_v2(x, y)); \
  } \
  inline V osl_usimplexnoise_v3(V p) \
  { \
    return osl_noise_unsigned(osl_simplexnoise_v3(p)); \
  } \
  inline V osl_usimplexnoise_v4(V p, F t) \
  { \
    return osl_noise_unsigned(osl_simplexnoise_v4(p, t)); \
  }

OSL_SIMPLEX_NOISE(float, float3)
OSL_SIMPLEX_NOISE(DualF, DualV)
)MSL";

/* Splines. */
static const char *osl_camera_msl_spline = R"MSL(
/* Bases in the order catmull-rom, bezier, bspline, hermite, linear, constant. */
constant float osl_spline_basis[6][16] = {
    {-0.5f, 1.5f, -1.5f, 0.5f, 1.0f, -2.5f, 2.0f, -0.5f, -0.5f, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f,
     0.0f, 0.0f},
    {-1.0f, 3.0f, -3.0f, 1.0f, 3.0f, -6.0f, 3.0f, 0.0f, -3.0f, 3.0f, 0.0f, 0.0f, 1.0f, 0.0f,
     0.0f, 0.0f},
    {-1.0f / 6.0f, 3.0f / 6.0f, -3.0f / 6.0f, 1.0f / 6.0f, 3.0f / 6.0f, -6.0f / 6.0f,
     3.0f / 6.0f, 0.0f, -3.0f / 6.0f, 0.0f, 3.0f / 6.0f, 0.0f, 1.0f / 6.0f, 4.0f / 6.0f,
     1.0f / 6.0f, 0.0f},
    {2.0f, 1.0f, -2.0f, 1.0f, -3.0f, -2.0f, 3.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
     0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
     0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
     0.0f}};
constant int osl_spline_step[6] = {1, 3, 1, 2, 1, 1};

inline int osl_spline_segments(int basis, int count)
{
  return ((count - 4) / osl_spline_step[basis]) + 1;
}
/* The segment that contains `x`. */
inline int osl_spline_segment(float x, int nsegs)
{
  const int segnum = int(clamp(x, 0.0f, 1.0f) * float(nsegs));
  return clamp(segnum, 0, max(nsegs - 1, 0));
}
/* Position inside the segment. */
inline float osl_spline_param(float x, int nsegs, int segnum)
{
  return clamp(x, 0.0f, 1.0f) * float(nsegs) - float(segnum);
}
inline DualF osl_spline_param(DualF x, int nsegs, int segnum)
{
  const bool inside = (x.v >= 0.0f && x.v <= 1.0f);
  const float scale = inside ? float(nsegs) : 0.0f;
  return DualF{clamp(x.v, 0.0f, 1.0f) * float(nsegs) - float(segnum), x.dx * scale,
               x.dy * scale};
}

inline float osl_spline_mul(float a, float x)
{
  return a * x;
}
inline float3 osl_spline_mul(float3 a, float x)
{
  return a * x;
}
inline DualF osl_spline_mul(DualF a, DualF x)
{
  return osl_mul(a, x);
}
inline DualV osl_spline_mul(DualV a, DualF x)
{
  return osl_mul(a, osl_dual3(x));
}
inline DualF osl_spline_scale(float c, DualF a)
{
  return DualF{c * a.v, c * a.dx, c * a.dy};
}
inline DualV osl_spline_scale(float c, DualV a)
{
  return DualV{c * a.v, c * a.dx, c * a.dy};
}
inline float osl_spline_scale(float c, float a)
{
  return c * a;
}
inline float3 osl_spline_scale(float c, float3 a)
{
  return c * a;
}
inline float osl_spline_add(float a, float b)
{
  return a + b;
}
inline float3 osl_spline_add(float3 a, float3 b)
{
  return a + b;
}
template<typename T> inline OslDual<T> osl_spline_add(OslDual<T> a, OslDual<T> b)
{
  return osl_add(a, b);
}

/* Evaluate one segment from its four knots. */
template<typename T, typename X> inline T osl_spline_eval(int basis, X x, T p0, T p1, T p2, T p3)
{
  T tk[4];
  for (int k = 0; k < 4; k++) {
    tk[k] = osl_spline_add(osl_spline_add(osl_spline_scale(osl_spline_basis[basis][k * 4], p0),
                                          osl_spline_scale(osl_spline_basis[basis][k * 4 + 1],
                                                           p1)),
                           osl_spline_add(osl_spline_scale(osl_spline_basis[basis][k * 4 + 2],
                                                           p2),
                                          osl_spline_scale(osl_spline_basis[basis][k * 4 + 3],
                                                           p3)));
  }
  T result = osl_spline_add(osl_spline_mul(tk[0], x), tk[1]);
  result = osl_spline_add(osl_spline_mul(result, x), tk[2]);
  return osl_spline_add(osl_spline_mul(result, x), tk[3]);
}

inline float osl_spline_value(int basis, float x, thread const float *knots, int count, int len)
{
  const int nsegs = osl_spline_segments(basis, count);
  const int segnum = osl_spline_segment(x, nsegs);
  if (basis == 5) {
    return knots[osl_idx(segnum + 1, len)];
  }
  const int s = segnum * osl_spline_step[basis];
  return osl_spline_eval(basis,
                         osl_spline_param(x, nsegs, segnum),
                         knots[osl_idx(s, len)],
                         knots[osl_idx(s + 1, len)],
                         knots[osl_idx(s + 2, len)],
                         knots[osl_idx(s + 3, len)]);
}

/* Find `x` with the spline value `y` in the range, by regula falsi and bisection. */
inline float osl_spline_invert(int basis,
                               float y,
                               float xmin,
                               float xmax,
                               thread const float *knots,
                               int count,
                               int len,
                               thread bool &bracketed)
{
  const int maxiters = 32;
  const float eps = 1.0e-6f;
  float v0 = osl_spline_value(basis, xmin, knots, count, len);
  float v1 = osl_spline_value(basis, xmax, knots, count, len);
  float x = xmin;
  float v = v0;
  const bool increasing = (v0 < v1);
  const float vmin = increasing ? v0 : v1;
  const float vmax = increasing ? v1 : v0;
  bracketed = (y >= vmin && y <= vmax);
  if (!bracketed) {
    return ((y < vmin) == increasing) ? xmin : xmax;
  }
  if (abs(v0 - v1) < eps) {
    return xmin;
  }
  const int rfiters = (3 * maxiters) / 4;
  for (int iters = 0; iters < maxiters; iters++) {
    float t;
    if (iters < rfiters) {
      t = (y - v0) / (v1 - v0);
      if (t <= 0.0f || t >= 1.0f) {
        t = 0.5f;
      }
    }
    else {
      t = 0.5f;
    }
    x = xmin * (1.0f - t) + xmax * t;
    v = osl_spline_value(basis, x, knots, count, len);
    if ((v < y) == increasing) {
      xmin = x;
      v0 = v;
    }
    else {
      xmax = x;
      v1 = v;
    }
    if (abs(xmax - xmin) < eps || abs(v - y) < eps) {
      return x;
    }
  }
  return x;
}

inline float osl_splineinverse(int basis, float y, thread const float *knots, int count, int len)
{
  const int step = osl_spline_step[basis];
  const int lowindex = (step == 1) ? 1 : 0;
  const int highindex = (step == 1) ? count - 2 : count - 1;
  const float low = knots[osl_idx(lowindex, len)];
  const float high = knots[osl_idx(highindex, len)];
  const bool increasing = knots[osl_idx(1, len)] < knots[osl_idx(count - 2, len)];
  if (increasing) {
    if (y <= low) {
      return 0.0f;
    }
    if (y >= high) {
      return 1.0f;
    }
  }
  else {
    if (y >= low) {
      return 0.0f;
    }
    if (y <= high) {
      return 1.0f;
    }
  }
  const int nsegs = osl_spline_segments(basis, count);
  const float nseginv = 1.0f / float(nsegs);
  /* Without a segment that contains the value, the result is the end of the last segment
   * that is nearest to it. */
  float r = 0.0f;
  for (int s = 0; s < min(nsegs, 4096); s++) {
    bool bracketed = false;
    r = osl_spline_invert(
        basis, y, float(s) * nseginv, float(s + 1) * nseginv, knots, count, len, bracketed);
    if (bracketed) {
      break;
    }
  }
  return r;
}
)MSL";

/* Color space conversions with derivatives. */
static const char *osl_camera_msl_color_derivs = R"MSL(
inline DualF osl_dual_if(bool c, DualF a, DualF b)
{
  if (c) {
    return a;
  }
  return b;
}
inline DualV osl_hsv_to_rgb(DualV hsv)
{
  const DualF h = osl_comp(hsv, 0), s = osl_comp(hsv, 1), v = osl_comp(hsv, 2);
  if (s.v < 0.0001f) {
    return osl_vec(v, v, v);
  }
  const DualF hh = 6.0f * (h - floor(h.v));
  const int hi = int(hh.v);
  const DualF f = hh - float(hi);
  const DualF p = v * (1.0f - s);
  const DualF q = v * (1.0f - s * f);
  const DualF t = v * (1.0f - s * (1.0f - f));
  switch (hi) {
    case 0:
      return osl_vec(v, t, p);
    case 1:
      return osl_vec(q, v, p);
    case 2:
      return osl_vec(p, v, t);
    case 3:
      return osl_vec(p, q, v);
    case 4:
      return osl_vec(t, p, v);
    default:
      return osl_vec(v, p, q);
  }
}
inline DualV osl_rgb_to_hsv(DualV rgb)
{
  const DualF r = osl_comp(rgb, 0), g = osl_comp(rgb, 1), b = osl_comp(rgb, 2);
  const DualF mincomp = osl_min(r, osl_min(g, b));
  const DualF maxcomp = osl_max(r, osl_max(g, b));
  const DualF delta = maxcomp - mincomp;
  DualF h = osl_dual(0.0f), s = osl_dual(0.0f);
  if (maxcomp.v > 0.0f) {
    s = osl_div(delta, maxcomp);
  }
  if (s.v > 0.0f) {
    if (r.v >= maxcomp.v) {
      h = osl_div(g - b, delta);
    }
    else if (g.v >= maxcomp.v) {
      h = osl_div(b - r, delta) + 2.0f;
    }
    else {
      h = osl_div(r - g, delta) + 4.0f;
    }
    h = h * (1.0f / 6.0f);
    if (h.v < 0.0f) {
      h = h + 1.0f;
    }
  }
  return osl_vec(h, s, maxcomp);
}
inline DualV osl_hsl_to_rgb(DualV hsl)
{
  const DualF h = osl_comp(hsl, 0), s = osl_comp(hsl, 1), l = osl_comp(hsl, 2);
  const DualF v = osl_dual_if(l.v <= 0.5f, l * (s + 1.0f), l * (1.0f - s) + s);
  if (v.v <= 0.0f) {
    return osl_dual(float3(0.0f));
  }
  const DualF m = 2.0f * l - v;
  return osl_hsv_to_rgb(osl_vec(h, osl_div(v - m, v), v));
}
inline DualV osl_rgb_to_hsl(DualV rgb)
{
  const DualF mincomp = osl_min(osl_comp(rgb, 0), osl_min(osl_comp(rgb, 1), osl_comp(rgb, 2)));
  const DualV hsv = osl_rgb_to_hsv(rgb);
  const DualF h = osl_comp(hsv, 0), v = osl_comp(hsv, 2);
  const DualF l = 0.5f * (mincomp + v);
  if (l.v <= 0.0f) {
    return osl_dual(float3(0.0f));
  }
  DualF s;
  if (l.v <= 0.5f) {
    s = osl_div(v - mincomp, v + mincomp);
  }
  else {
    s = osl_div(v - mincomp, 2.0f - (v + mincomp));
  }
  return osl_vec(h, s, l);
}
inline DualV osl_color_matrix(DualV c, float3 r, float3 g, float3 b)
{
  return DualV{osl_color_matrix(c.v, r, g, b), osl_color_matrix(c.dx, r, g, b),
               osl_color_matrix(c.dy, r, g, b)};
}
inline DualV osl_yiq_to_rgb(DualV c)
{
  return DualV{osl_yiq_to_rgb(c.v), osl_yiq_to_rgb(c.dx), osl_yiq_to_rgb(c.dy)};
}
inline DualV osl_rgb_to_yiq(DualV c)
{
  return DualV{osl_rgb_to_yiq(c.v), osl_rgb_to_yiq(c.dx), osl_rgb_to_yiq(c.dy)};
}
inline DualV osl_xyy_to_xyz(DualV c)
{
  const DualF x = osl_comp(c, 0), y = osl_comp(c, 1), Y = osl_comp(c, 2);
  const DualF n = osl_div(Y, osl_dual_if(y.v == 0.0f, osl_dual(1.0e-37f), y));
  return osl_vec(x * n, Y, (1.0f - x - y) * n);
}
inline DualV osl_xyz_to_xyy(DualV c)
{
  const DualF X = osl_comp(c, 0), Y = osl_comp(c, 1), Z = osl_comp(c, 2);
  const DualF n = X + Y + Z;
  const DualF n_inv = osl_dual_if(n.v == 0.0f, osl_dual(0.0f), osl_div(osl_dual(1.0f), n));
  return osl_vec(X * n_inv, Y * n_inv, Y);
}
inline DualF osl_srgb_to_linear(DualF x)
{
  if (x.v <= 0.04045f) {
    return x * (1.0f / 12.92f);
  }
  return osl_pow((x + 0.055f) * (1.0f / 1.055f), osl_dual(2.4f));
}
inline DualF osl_linear_to_srgb(DualF x)
{
  if (x.v <= 0.0031308f) {
    return 12.92f * x;
  }
  return 1.055f * osl_pow(x, osl_dual(1.0f / 2.4f)) - 0.055f;
}
inline DualV osl_srgb_to_linear(DualV c)
{
  return osl_vec(osl_srgb_to_linear(osl_comp(c, 0)), osl_srgb_to_linear(osl_comp(c, 1)),
                 osl_srgb_to_linear(osl_comp(c, 2)));
}
inline DualV osl_linear_to_srgb(DualV c)
{
  return osl_vec(osl_linear_to_srgb(osl_comp(c, 0)), osl_linear_to_srgb(osl_comp(c, 1)),
                 osl_linear_to_srgb(osl_comp(c, 2)));
}
)MSL";

/* Gabor noise, with the algorithm and constants of Open Shading Language (BSD-3-Clause):
 * sparse convolution of Gabor kernels at random impulses, filtered with the derivatives of the
 * position. The random numbers, the approximations of exp2() and sincos() and the fallbacks for
 * singular matrices are the ones of the OSL runtime, so that the noise has the same values. */
static const char *osl_camera_msl_gabor = R"MSL(
struct OslGaborParams {
  int anisotropic;
  float3 direction;
  float bandwidth;
  float impulses;
  int do_filter;
};
inline OslGaborParams osl_gabor_params(
    int anisotropic, float3 direction, float bandwidth, float impulses, int do_filter)
{
  return OslGaborParams{anisotropic, direction, bandwidth, impulses, do_filter};
}

struct OslGabor {
  float3 omega;
  int anisotropic;
  bool do_filter;
  float a;
  float weight;
  float3 N;
  /* 2x2 matrices are stored as (m00, m01, m10, m11). */
  float4 filter;
  /* The tangent, bitangent and normal that make up the tangent space. */
  float3 t;
  float3 b;
  float3 n;
  bool periodic;
  float3 period;
  float lambda;
  float sqrt_lambda_inv;
  float radius;
  float radius2;
  float radius3;
  float radius_inv;
};

inline float osl_fast_exp2(float xval)
{
  float x = clamp(xval, -126.0f, 126.0f);
  const int m = int(x);
  x -= float(m);
  x = 1.0f - (1.0f - x);
  float r = 1.33336498402e-3f;
  r = fma(x, r, 9.810352697968e-3f);
  r = fma(x, r, 5.551834031939e-2f);
  r = fma(x, r, 0.2401793301105f);
  r = fma(x, r, 0.693144857883f);
  r = fma(x, r, 1.0f);
  return as_type<float>(as_type<uint>(r) + (uint(m) << 23));
}
/* Returns the cosine and the sine. */
inline float2 osl_fast_cossin(float x)
{
  const int q = int(x * 0.318309886f + ((x < 0.0f) ? -0.5f : 0.5f));
  const float qf = float(q);
  x = fma(qf, -0.78515625f * 4.0f, x);
  x = fma(qf, -0.00024187564849853515625f * 4.0f, x);
  x = fma(qf, -3.7747668102383613586e-08f * 4.0f, x);
  x = fma(qf, -1.2816720341285448015e-12f * 4.0f, x);
  x = 1.5707963267948966f - (1.5707963267948966f - x);
  const float s = x * x;
  if ((q & 1) != 0) {
    x = -x;
  }
  float su = 2.6083159809786593541503e-06f;
  su = fma(su, s, -0.0001981069071916863322258f);
  su = fma(su, s, 0.00833307858556509017944336f);
  su = fma(su, s, -0.166666597127914428710938f);
  su = fma(s, su * x, x);
  float cu = -2.71811842367242206819355e-07f;
  cu = fma(cu, s, 2.47990446951007470488548e-05f);
  cu = fma(cu, s, -0.00138888787478208541870117f);
  cu = fma(cu, s, 0.0416666641831398010253906f);
  cu = fma(cu, s, -0.5f);
  cu = fma(cu, s, 1.0f);
  if ((q & 1) != 0) {
    cu = -cu;
  }
  return float2(clamp(cu, -1.0f, 1.0f), clamp(su, -1.0f, 1.0f));
}

/* Random numbers in [0, 1), with a linear congruential generator. */
inline float osl_gabor_rng(thread uint &seed)
{
  seed *= 3039177861u;
  return float(seed) * 2.3283064365386963e-10f;
}

inline float osl_m22_det(float4 m)
{
  return m.x * m.w - m.z * m.y;
}
/* A matrix that cannot be inverted has the identity as its inverse. */
inline float4 osl_m22_inverse(float4 m)
{
  const float4 s = float4(m.w, -m.y, -m.z, m.x);
  const float r = osl_m22_det(m);
  if (abs(r) >= 1.0f) {
    return s / r;
  }
  const float mr = abs(r) * 8.507059173e37f;
  if (all(float4(mr) > abs(s))) {
    return s / r;
  }
  return float4(1.0f, 0.0f, 0.0f, 1.0f);
}

inline DualF osl_gabor_dot(float3 a, DualV x)
{
  return DualF{dot(a, x.v), dot(a, x.dx), dot(a, x.dy)};
}

/* A harmonic in a Gaussian envelope. */
inline DualF osl_gabor_kernel(DualF weight, float3 omega, DualF phi, float bandwidth, DualV x)
{
  const DualF g = osl_exp((-3.14159265358979f * (bandwidth * bandwidth)) * osl_dot(x, x));
  const DualF h = osl_cos(6.28318530717959f * osl_gabor_dot(omega, x) + phi);
  return weight * g * h;
}
inline DualF osl_gabor_kernel(
    DualF weight, float2 omega, DualF phi, float bandwidth, DualF x, DualF y)
{
  const DualF g = osl_exp((-3.14159265358979f * (bandwidth * bandwidth)) * (x * x + y * y));
  const DualF h = osl_cos(6.28318530717959f * (omega.x * x + omega.y * y) + phi);
  return weight * g * h;
}

inline DualF osl_gabor_filtered_kernel(thread const OslGabor &gp, float3 omega, float phi, DualV x)
{
  const float two_pi = 6.28318530717959f;
  /* The orientation of the impulse in tangent space. */
  const float3 omega_t = float3(dot(omega, gp.t), dot(omega, gp.b), dot(omega, gp.n));
  /* Slice the kernel with the tangent plane. */
  const DualF d = -osl_gabor_dot(gp.N, x);
  const DualF w_s = gp.weight * osl_exp((-3.14159265358979f * (gp.a * gp.a)) * (d * d));
  const float2 mu_g = omega_t.xy;
  const DualF phi_s = phi - (two_pi * omega_t.z) * d;

  /* Filter the sliced kernel: the product of two Gaussians in the frequency domain. */
  const float sigma_g = gp.a * gp.a / two_pi;
  const float c_f = 1.0f / (two_pi * sqrt(osl_m22_det(gp.filter)));
  const float4 sigma_f = 0.0253302959105844f * osl_m22_inverse(gp.filter);
  const float4 sigma_sum = float4(sigma_g, 0.0f, 0.0f, sigma_g) + sigma_f;
  const float4 sum_inv = osl_m22_inverse(sigma_sum);
  const float quadratic = mu_g.y * (mu_g.x * sum_inv.y + mu_g.y * sum_inv.w) +
                          (mu_g.x * sum_inv.x + mu_g.y * sum_inv.z) * mu_g.x;
  const DualF w_f = (c_f * (1.0f / (two_pi * sqrt(osl_m22_det(sigma_sum)))) *
                     exp(-0.5f * quadratic)) *
                    w_s;
  const float4 sigma_g_inv = osl_m22_inverse(float4(sigma_g, 0.0f, 0.0f, sigma_g));
  const float4 sigma_gf = osl_m22_inverse(osl_m22_inverse(sigma_f) + sigma_g_inv);
  /* The product with the inverse, which is a multiple of the identity. */
  const float4 product = sigma_gf * sigma_g_inv.x;
  const float2 mu_gf = float2(mu_g.x * product.x + mu_g.y * product.z,
                              mu_g.x * product.y + mu_g.y * product.w);
  const float a_f = sqrt(two_pi * sqrt(osl_m22_det(sigma_gf)));

  /* Evaluate the filtered kernel in the tangent plane. */
  const DualF gk = osl_gabor_kernel(
      w_f, mu_gf, phi_s, a_f, osl_gabor_dot(gp.t, x), osl_gabor_dot(gp.b, x));
  if (osl_isfinite(gk.v) == 0) {
    /* The filter failed numerically. */
    return osl_gabor_kernel(osl_dual(gp.weight), omega, osl_dual(phi), gp.a, x);
  }
  return gk;
}

/* The sum of all impulses in one cell. `x_c` is the position relative to the cell. */
inline DualF osl_gabor_cell(thread const OslGabor &gp, float3 cell, DualV x_c, int seed_offset)
{
  if (gp.periodic) {
    cell = osl_wrap(cell, gp.period);
  }
  uint seed = osl_inthash(uint(int(floor(cell.x))),
                          uint(int(floor(cell.y))),
                          uint(int(floor(cell.z))),
                          uint(seed_offset));
  if (seed == 0u) {
    seed = 1u;
  }
  /* The number of impulses has a Poisson distribution. */
  const float g = exp(-(gp.lambda * gp.radius3));
  int num_impulses = 0;
  float t = osl_gabor_rng(seed);
  while (t > g && num_impulses < 1024) {
    num_impulses++;
    t *= osl_gabor_rng(seed);
  }

  DualF sum = osl_dual(0.0f);
  for (int i = 0; i < num_impulses; i++) {
    const float z_rng = osl_gabor_rng(seed);
    const float y_rng = osl_gabor_rng(seed);
    const float x_rng = osl_gabor_rng(seed);
    const float3 x_i = float3(x_rng, y_rng, z_rng);
    const DualV x_k = DualV{gp.radius * (x_c.v - x_i), gp.radius * x_c.dx, gp.radius * x_c.dy};

    /* Orientation and phase of the impulse. */
    float3 omega;
    if (gp.anisotropic == 1) {
      omega = gp.omega;
    }
    else if (gp.anisotropic == 0) {
      const float2 cs = osl_fast_cossin(6.28318530717959f * osl_gabor_rng(seed));
      const float u = osl_gabor_rng(seed);
      const float cos_p = u - (1.0f - u);
      const float sin_p = sqrt(max(0.0f, 1.0f - cos_p * cos_p));
      omega = osl_normalize(float3(cs.x * sin_p, cs.y * sin_p, cos_p));
    }
    else {
      const float2 cs = osl_fast_cossin(6.28318530717959f * osl_gabor_rng(seed));
      omega = osl_length(gp.omega) * float3(cs.x, cs.y, 0.0f);
    }
    const float phi = 6.28318530717959f * osl_gabor_rng(seed);

    if (dot(x_k.v, x_k.v) < gp.radius2) {
      if (gp.do_filter) {
        sum = sum + osl_gabor_filtered_kernel(gp, omega, phi, x_k);
      }
      else {
        sum = sum + osl_gabor_kernel(osl_dual(gp.weight), omega, osl_dual(phi), gp.a, x_k);
      }
    }
  }
  return sum;
}

inline DualF osl_gabor_evaluate(thread const OslGabor &gp, DualV p, int seed)
{
  const DualV x_g = DualV{p.v * gp.radius_inv, p.dx * gp.radius_inv, p.dy * gp.radius_inv};
  const float3 cell = floor(x_g.v);
  DualF sum = osl_dual(0.0f);
  for (int k = -1; k <= 1; k++) {
    for (int j = -1; j <= 1; j++) {
      for (int i = -1; i <= 1; i++) {
        const float3 c = float3(float(i), float(j), float(k));
        sum = sum + osl_gabor_cell(gp, cell + c, DualV{x_g.v - cell - c, x_g.dx, x_g.dy}, seed);
      }
    }
  }
  return sum * gp.sqrt_lambda_inv;
}

inline OslGabor osl_gabor_setup(OslGaborParams opt, DualV p, bool periodic, float3 period)
{
  OslGabor gp;
  gp.omega = opt.direction;
  gp.anisotropic = opt.anisotropic;
  gp.do_filter = (opt.do_filter != 0);
  gp.weight = 1.0f;
  gp.periodic = periodic;
  gp.period = period;
  gp.N = float3(0.0f);
  gp.filter = float4(1.0f, 0.0f, 0.0f, 1.0f);
  gp.t = float3(1.0f, 0.0f, 0.0f);
  gp.b = float3(0.0f, 1.0f, 0.0f);
  gp.n = float3(0.0f, 0.0f, 1.0f);

  const float bandwidth = clamp(opt.bandwidth, 0.01f, 100.0f);
  const float two_to_bandwidth = osl_fast_exp2(bandwidth);
  gp.a = 2.0f * ((two_to_bandwidth - 1.0f) / (two_to_bandwidth + 1.0f)) * 2.12893403886245f;
  /* Impulses further away than this have a negligible envelope. */
  gp.radius = 1.11590123f / gp.a;
  gp.radius2 = gp.radius * gp.radius;
  gp.radius3 = gp.radius2 * gp.radius;
  gp.radius_inv = 1.0f / gp.radius;
  gp.lambda = clamp(opt.impulses, 1.0f, 32.0f) / (gp.radius3 * 4.18877649f);
  gp.sqrt_lambda_inv = 1.0f / sqrt(gp.lambda);

  if (gp.do_filter) {
    /* The filter in the tangent space of the surface that the position moves on. */
    float3 n = cross(p.dx, p.dy);
    if (dot(n, n) < 1.0e-6f) {
      gp.do_filter = false;
    }
    else {
      n = osl_normalize(n);
      float3 t = (abs(n.x) < 0.9f) ? float3(0.0f, n.z, -n.y) : float3(-n.z, 0.0f, n.x);
      t = osl_normalize(t);
      const float3 b = cross(n, t);
      const float m00 = p.dx.x * t.x + p.dy.x * t.y;
      const float m01 = p.dx.x * b.x + p.dy.x * b.y;
      const float m10 = p.dx.y * t.x + p.dy.y * t.y;
      const float m11 = p.dx.y * b.x + p.dy.y * b.y;
      const float off = 0.25f * (m00 * m01 + m10 * m11);
      gp.filter = float4(0.25f * (m00 * m00 + m10 * m10), off, off,
                         0.25f * (m01 * m01 + m11 * m11));
      gp.N = n;
      gp.t = t;
      gp.b = b;
      gp.n = n;
      if (osl_m22_det(gp.filter) < 1.0e-18f) {
        gp.do_filter = false;
      }
    }
  }
  return gp;
}

/* Scale to make the noise fit in [-1, 1]. */
inline float osl_gabor_scale(float a)
{
  const float variance = 1.0f / (5.65685415f * (a * a * a));
  return 0.5f * (1.0f / (3.0f * sqrt(variance)));
}

inline DualF osl_gabor_noise(DualV p, OslGaborParams opt, bool periodic, float3 period)
{
  const OslGabor gp = osl_gabor_setup(opt, p, periodic, period);
  return osl_gabor_evaluate(gp, p, 0) * osl_gabor_scale(gp.a);
}
inline DualV osl_gabor_noise3(DualV p, OslGaborParams opt, bool periodic, float3 period)
{
  const OslGabor gp = osl_gabor_setup(opt, p, periodic, period);
  const float scale = osl_gabor_scale(gp.a);
  return osl_vec(osl_gabor_evaluate(gp, p, 0) * scale,
                 osl_gabor_evaluate(gp, p, 1) * scale,
                 osl_gabor_evaluate(gp, p, 2) * scale);
}

inline DualV osl_gabor_p(DualF x)
{
  return osl_vec(x, osl_dual(0.0f), osl_dual(0.0f));
}
inline DualV osl_gabor_p(DualF x, DualF y)
{
  return osl_vec(x, y, osl_dual(0.0f));
}

/* Noise of 1 to 4 dimensions. The fourth dimension is not used. */
inline DualF osl_gabor_f1(DualF x, OslGaborParams opt)
{
  return osl_gabor_noise(osl_gabor_p(x), opt, false, float3(0.0f));
}
inline DualF osl_gabor_f2(DualF x, DualF y, OslGaborParams opt)
{
  return osl_gabor_noise(osl_gabor_p(x, y), opt, false, float3(0.0f));
}
inline DualF osl_gabor_f3(DualV p, OslGaborParams opt)
{
  return osl_gabor_noise(p, opt, false, float3(0.0f));
}
inline DualF osl_gabor_f4(DualV p, DualF t, OslGaborParams opt)
{
  return osl_gabor_noise(p, opt, false, float3(0.0f));
}
inline DualV osl_gabor_v1(DualF x, OslGaborParams opt)
{
  return osl_gabor_noise3(osl_gabor_p(x), opt, false, float3(0.0f));
}
inline DualV osl_gabor_v2(DualF x, DualF y, OslGaborParams opt)
{
  return osl_gabor_noise3(osl_gabor_p(x, y), opt, false, float3(0.0f));
}
inline DualV osl_gabor_v3(DualV p, OslGaborParams opt)
{
  return osl_gabor_noise3(p, opt, false, float3(0.0f));
}
inline DualV osl_gabor_v4(DualV p, DualF t, OslGaborParams opt)
{
  return osl_gabor_noise3(p, opt, false, float3(0.0f));
}
inline DualF osl_pgabor_f1(DualF x, float px, OslGaborParams opt)
{
  return osl_gabor_noise(osl_gabor_p(x), opt, true, float3(px, 0.0f, 0.0f));
}
inline DualF osl_pgabor_f2(DualF x, DualF y, float px, float py, OslGaborParams opt)
{
  return osl_gabor_noise(osl_gabor_p(x, y), opt, true, float3(px, py, 0.0f));
}
inline DualF osl_pgabor_f3(DualV p, float3 pp, OslGaborParams opt)
{
  return osl_gabor_noise(p, opt, true, pp);
}
inline DualF osl_pgabor_f4(DualV p, DualF t, float3 pp, float pt, OslGaborParams opt)
{
  return osl_gabor_noise(p, opt, true, pp);
}
inline DualV osl_pgabor_v1(DualF x, float px, OslGaborParams opt)
{
  return osl_gabor_noise3(osl_gabor_p(x), opt, true, float3(px, 0.0f, 0.0f));
}
inline DualV osl_pgabor_v2(DualF x, DualF y, float px, float py, OslGaborParams opt)
{
  return osl_gabor_noise3(osl_gabor_p(x, y), opt, true, float3(px, py, 0.0f));
}
inline DualV osl_pgabor_v3(DualV p, float3 pp, OslGaborParams opt)
{
  return osl_gabor_noise3(p, opt, true, pp);
}
inline DualV osl_pgabor_v4(DualV p, DualF t, float3 pp, float pt, OslGaborParams opt)
{
  return osl_gabor_noise3(p, opt, true, pp);
}
)MSL";

CCL_NAMESPACE_END
