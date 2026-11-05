/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Sources for the runtime compilation of custom camera shaders with NVRTC, see `osl_camera.h`.
 * Each literal stays below the 16380 bytes that MSVC allows for one string literal. */

CCL_NAMESPACE_BEGIN

/* The subset of the Metal shading language that translated camera shaders use, for CUDA. The
 * translation is the one of Metal, see `device/metal/osl_camera_translate.cpp`. */
static const char *osl_camera_cuda_compat_1 = R"CUDA(
/* The subset of the Metal shading language that translated camera shaders use, for CUDA.
 *
 * Compiled with NVRTC and `-default-device`, so functions and namespace scope variables without
 * an execution space are device entities. Vector types are renamed so that they do not collide
 * with the CUDA built-in types. Swizzles are functions: the source uses `.xyz()` for `.xyz`. */

#define constant const
#define thread
#define device

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

#define float2 msl_float2
#define float3 msl_float3
#define float4 msl_float4
#define int2 msl_int2
#define int3 msl_int3
#define int4 msl_int4
#define uint2 msl_uint2
#define uint3 msl_uint3
#define uint4 msl_uint4
#define bool2 msl_bool2
#define bool3 msl_bool3
#define bool4 msl_bool4
#define uchar4 msl_uchar4
#define float3x3 msl_float3x3
#define float4x4 msl_float4x4

/* Scalars. */

__forceinline__ float msl_min(float a, float b)
{
  return fminf(a, b);
}
__forceinline__ float msl_max(float a, float b)
{
  return fmaxf(a, b);
}
__forceinline__ int msl_min(int a, int b)
{
  return a < b ? a : b;
}
__forceinline__ int msl_max(int a, int b)
{
  return a > b ? a : b;
}
__forceinline__ uint msl_min(uint a, uint b)
{
  return a < b ? a : b;
}
__forceinline__ uint msl_max(uint a, uint b)
{
  return a > b ? a : b;
}
#define min msl_min
#define max msl_max

__forceinline__ float clamp(float x, float lo, float hi)
{
  return fminf(fmaxf(x, lo), hi);
}
__forceinline__ int clamp(int x, int lo, int hi)
{
  return x < lo ? lo : (x > hi ? hi : x);
}
__forceinline__ uint clamp(uint x, uint lo, uint hi)
{
  return x < lo ? lo : (x > hi ? hi : x);
}
__forceinline__ float mix(float a, float b, float t)
{
  return a + (b - a) * t;
}
__forceinline__ float sign(float x)
{
  return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
}
__forceinline__ float fract(float x)
{
  return fminf(x - floorf(x), 0x1.fffffep-1f);
}
__forceinline__ float step(float edge, float x)
{
  return x < edge ? 0.0f : 1.0f;
}
__forceinline__ float rsqrt(float x)
{
  return rsqrtf(x);
}
__forceinline__ float saturate(float x)
{
  return fminf(fmaxf(x, 0.0f), 1.0f);
}
__forceinline__ float select(float a, float b, bool c)
{
  return c ? b : a;
}
__forceinline__ int select(int a, int b, bool c)
{
  return c ? b : a;
}
__forceinline__ uint select(uint a, uint b, bool c)
{
  return c ? b : a;
}

template<typename T, typename U> __forceinline__ T as_type(U u)
{
  static_assert(sizeof(T) == sizeof(U), "as_type requires types of equal size");
  union {
    U u;
    T t;
  } v;
  v.u = u;
  return v.t;
}

/* Vectors. */

struct bool2 {
  typedef int msl_vector;
  bool x, y;
  bool2() = default;
  constexpr bool2(bool s) : x(s), y(s) {}
  constexpr bool2(bool a, bool b) : x(a), y(b) {}
};
struct bool3 {
  typedef int msl_vector;
  bool x, y, z;
  bool3() = default;
  constexpr bool3(bool s) : x(s), y(s), z(s) {}
  constexpr bool3(bool a, bool b, bool c) : x(a), y(b), z(c) {}
};
struct bool4 {
  typedef int msl_vector;
  bool x, y, z, w;
  bool4() = default;
  constexpr bool4(bool s) : x(s), y(s), z(s), w(s) {}
  constexpr bool4(bool a, bool b, bool c, bool d) : x(a), y(b), z(c), w(d) {}
};
__forceinline__ bool all(bool2 v)
{
  return v.x && v.y;
}
__forceinline__ bool all(bool3 v)
{
  return v.x && v.y && v.z;
}
__forceinline__ bool all(bool4 v)
{
  return v.x && v.y && v.z && v.w;
}
__forceinline__ bool any(bool2 v)
{
  return v.x || v.y;
}
__forceinline__ bool any(bool3 v)
{
  return v.x || v.y || v.z;
}
__forceinline__ bool any(bool4 v)
{
  return v.x || v.y || v.z || v.w;
}
__forceinline__ bool3 operator!(bool3 v)
{
  return bool3(!v.x, !v.y, !v.z);
}
__forceinline__ bool3 operator&&(bool3 a, bool3 b)
{
  return bool3(a.x && b.x, a.y && b.y, a.z && b.z);
}
__forceinline__ bool3 operator||(bool3 a, bool3 b)
{
  return bool3(a.x || b.x, a.y || b.y, a.z || b.z);
}

/* Two, three and four component vectors of a scalar type T, with the Metal constructors,
 * component-wise arithmetic and comparisons. */
#define MSL_VEC2(V, T) \
  struct V { \
    typedef int msl_vector; \
    T x, y; \
    V() = default; \
    constexpr V(T s) : x(s), y(s) {} \
    constexpr V(T a, T b) : x(a), y(b) {} \
    template<typename W, typename = typename W::msl_vector> constexpr explicit V(W v) : x(T(v.x)), y(T(v.y)) {} \
    T &operator[](int i) \
    { \
      return (&x)[i]; \
    } \
    T operator[](int i) const \
    { \
      return (&x)[i]; \
    } \
    V xy() const \
    { \
      return *this; \
    } \
  };
#define MSL_VEC3(V, V2, T) \
  struct V { \
    typedef int msl_vector; \
    T x, y, z; \
    V() = default; \
    constexpr V(T s) : x(s), y(s), z(s) {} \
    constexpr V(T a, T b, T c) : x(a), y(b), z(c) {} \
    constexpr V(V2 a, T c) : x(a.x), y(a.y), z(c) {} \
    constexpr V(T a, V2 b) : x(a), y(b.x), z(b.y) {} \
    template<typename W, typename = typename W::msl_vector> constexpr explicit V(W v) : x(T(v.x)), y(T(v.y)), z(T(v.z)) {} \
    T &operator[](int i) \
    { \
      return (&x)[i]; \
    } \
    T operator[](int i) const \
    { \
      return (&x)[i]; \
    } \
    V2 xy() const \
    { \
      return V2(x, y); \
    } \
    V xyz() const \
    { \
      return *this; \
    } \
  };
#define MSL_VEC4(V, V3, V2, T) \
  struct V { \
    typedef int msl_vector; \
    T x, y, z, w; \
    V() = default; \
    constexpr V(T s) : x(s), y(s), z(s), w(s) {} \
    constexpr V(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {} \
    constexpr V(V3 a, T d) : x(a.x), y(a.y), z(a.z), w(d) {} \
    constexpr V(T a, V3 b) : x(a), y(b.x), z(b.y), w(b.z) {} \
    constexpr V(V2 a, V2 b) : x(a.x), y(a.y), z(b.x), w(b.y) {} \
    constexpr V(V2 a, T c, T d) : x(a.x), y(a.y), z(c), w(d) {} \
    template<typename W, typename = typename W::msl_vector> \
    constexpr explicit V(W v) : x(T(v.x)), y(T(v.y)), z(T(v.z)), w(T(v.w)) \
    { \
    } \
    T &operator[](int i) \
    { \
      return (&x)[i]; \
    } \
    T operator[](int i) const \
    { \
      return (&x)[i]; \
    } \
    V2 xy() const \
    { \
      return V2(x, y); \
    } \
    V3 xyz() const \
    { \
      return V3(x, y, z); \
    } \
  };

MSL_VEC2(float2, float)
MSL_VEC3(float3, float2, float)
MSL_VEC4(float4, float3, float2, float)
MSL_VEC2(int2, int)
MSL_VEC3(int3, int2, int)
MSL_VEC4(int4, int3, int2, int)
MSL_VEC2(uint2, uint)
MSL_VEC3(uint3, uint2, uint)
MSL_VEC4(uint4, uint3, uint2, uint)
MSL_VEC2(msl_uchar2, uchar)
MSL_VEC3(msl_uchar3, msl_uchar2, uchar)
MSL_VEC4(uchar4, msl_uchar3, msl_uchar2, uchar)

/* Component-wise operations, for each vector type. N is the number of components. */
#define MSL_COMPONENTS2(e) e(x) e(y)
#define MSL_COMPONENTS3(e) e(x) e(y) e(z)
#define MSL_COMPONENTS4(e) e(x) e(y) e(z) e(w)

#define MSL_BINARY_OP(V, T, B, N, op) \
  __forceinline__ V operator op(V a, V b) \
  { \
    V r; \
    MSL_COMPONENTS##N(MSL_BINARY_COMPONENT_##N##_##op) return r; \
  }

/* Generic component helpers through lambdas would not be constexpr friendly; spell them out. */
#define MSL_APPLY2(V, f, a) V(f((a).x), f((a).y))
#define MSL_APPLY3(V, f, a) V(f((a).x), f((a).y), f((a).z))
#define MSL_APPLY4(V, f, a) V(f((a).x), f((a).y), f((a).z), f((a).w))
#define MSL_APPLY2_2(V, f, a, b) V(f((a).x, (b).x), f((a).y, (b).y))
#define MSL_APPLY3_2(V, f, a, b) V(f((a).x, (b).x), f((a).y, (b).y), f((a).z, (b).z))
#define MSL_APPLY4_2(V, f, a, b) \
  V(f((a).x, (b).x), f((a).y, (b).y), f((a).z, (b).z), f((a).w, (b).w))
#define MSL_APPLY2_3(V, f, a, b, c) V(f((a).x, (b).x, (c).x), f((a).y, (b).y, (c).y))
#define MSL_APPLY3_3(V, f, a, b, c) \
  V(f((a).x, (b).x, (c).x), f((a).y, (b).y, (c).y), f((a).z, (b).z, (c).z))
#define MSL_APPLY4_3(V, f, a, b, c) \
  V(f((a).x, (b).x, (c).x), f((a).y, (b).y, (c).y), f((a).z, (b).z, (c).z), f((a).w, (b).w, (c).w))

#define MSL_OP_ADD(a, b) ((a) + (b))
#define MSL_OP_SUB(a, b) ((a) - (b))
#define MSL_OP_MUL(a, b) ((a) * (b))
#define MSL_OP_DIV(a, b) ((a) / (b))
#define MSL_OP_MOD(a, b) ((a) % (b))
#define MSL_OP_AND(a, b) ((a) & (b))
#define MSL_OP_OR(a, b) ((a) | (b))
#define MSL_OP_XOR(a, b) ((a) ^ (b))
#define MSL_OP_SHL(a, b) ((a) << (b))
#define MSL_OP_SHR(a, b) ((a) >> (b))
#define MSL_OP_LT(a, b) ((a) < (b))
#define MSL_OP_LE(a, b) ((a) <= (b))
#define MSL_OP_GT(a, b) ((a) > (b))
#define MSL_OP_GE(a, b) ((a) >= (b))
#define MSL_OP_EQ(a, b) ((a) == (b))
#define MSL_OP_NE(a, b) ((a) != (b))
#define MSL_OP_NEG(a) (-(a))

#define MSL_ARITH_OP(V, T, N, op, F) \
  __forceinline__ V operator op(V a, V b) \
  { \
    return MSL_APPLY##N##_2(V, F, a, b); \
  } \
  __forceinline__ V operator op(V a, T b) \
  { \
    return MSL_APPLY##N##_2(V, F, a, V(b)); \
  } \
  __forceinline__ V operator op(T a, V b) \
  { \
    return MSL_APPLY##N##_2(V, F, V(a), b); \
  } \
  __forceinline__ V &operator op##=(V & a, V b) \
  { \
    a = a op b; \
    return a; \
  } \
  __forceinline__ V &operator op##=(V & a, T b) \
  { \
    a = a op V(b); \
    return a; \
  }

#define MSL_COMPARE_OP(V, T, B, N, op, F) \
  __forceinline__ B operator op(V a, V b) \
  { \
    return MSL_APPLY##N##_2(B, F, a, b); \
  } \
  __forceinline__ B operator op(V a, T b) \
  { \
    return MSL_APPLY##N##_2(B, F, a, V(b)); \
  } \
  __forceinline__ B operator op(T a, V b) \
  { \
    return MSL_APPLY##N##_2(B, F, V(a), b); \
  }

#define MSL_VECTOR_OPS(V, T, B, N) \
  MSL_ARITH_OP(V, T, N, +, MSL_OP_ADD) \
  MSL_ARITH_OP(V, T, N, -, MSL_OP_SUB) \
  MSL_ARITH_OP(V, T, N, *, MSL_OP_MUL) \
  MSL_ARITH_OP(V, T, N, /, MSL_OP_DIV) \
  MSL_COMPARE_OP(V, T, B, N, <, MSL_OP_LT) \
  MSL_COMPARE_OP(V, T, B, N, <=, MSL_OP_LE) \
  MSL_COMPARE_OP(V, T, B, N, >, MSL_OP_GT) \
  MSL_COMPARE_OP(V, T, B, N, >=, MSL_OP_GE) \
  MSL_COMPARE_OP(V, T, B, N, ==, MSL_OP_EQ) \
  MSL_COMPARE_OP(V, T, B, N, !=, MSL_OP_NE) \
  __forceinline__ V operator-(V a) \
  { \
    return MSL_APPLY##N(V, MSL_OP_NEG, a); \
  } \
  __forceinline__ V operator+(V a) \
  { \
    return a; \
  } \
  __forceinline__ V min(V a, V b) \
  { \
    return MSL_APPLY##N##_2(V, msl_min, a, b); \
  } \
  __forceinline__ V min(V a, T b) \
  { \
    return MSL_APPLY##N##_2(V, msl_min, a, V(b)); \
  } \
  __forceinline__ V max(V a, V b) \
  { \
    return MSL_APPLY##N##_2(V, msl_max, a, b); \
  } \
  __forceinline__ V max(V a, T b) \
  { \
    return MSL_APPLY##N##_2(V, msl_max, a, V(b)); \
  } \
  __forceinline__ V clamp(V x, V lo, V hi) \
  { \
    return MSL_APPLY##N##_3(V, clamp, x, lo, hi); \
  } \
  __forceinline__ V clamp(V x, T lo, T hi) \
  { \
    return MSL_APPLY##N##_3(V, clamp, x, V(lo), V(hi)); \
  } \
  __forceinline__ V select(V a, V b, B c) \
  { \
    return MSL_APPLY##N##_3(V, select, a, b, c); \
  }

#define MSL_INT_VECTOR_OPS(V, T, N) \
  MSL_ARITH_OP(V, T, N, %, MSL_OP_MOD) \
  MSL_ARITH_OP(V, T, N, &, MSL_OP_AND) \
  MSL_ARITH_OP(V, T, N, |, MSL_OP_OR) \
  MSL_ARITH_OP(V, T, N, ^, MSL_OP_XOR) \
  MSL_ARITH_OP(V, T, N, <<, MSL_OP_SHL) \
  MSL_ARITH_OP(V, T, N, >>, MSL_OP_SHR)

MSL_VECTOR_OPS(float2, float, bool2, 2)
MSL_VECTOR_OPS(float3, float, bool3, 3)
MSL_VECTOR_OPS(float4, float, bool4, 4)
MSL_VECTOR_OPS(int2, int, bool2, 2)
MSL_VECTOR_OPS(int3, int, bool3, 3)
MSL_VECTOR_OPS(int4, int, bool4, 4)
MSL_VECTOR_OPS(uint2, uint, bool2, 2)
MSL_VECTOR_OPS(uint3, uint, bool3, 3)
MSL_VECTOR_OPS(uint4, uint, bool4, 4)
MSL_COMPARE_OP(uchar4, uchar, bool4, 4, <, MSL_OP_LT)
MSL_COMPARE_OP(uchar4, uchar, bool4, 4, <=, MSL_OP_LE)
MSL_COMPARE_OP(uchar4, uchar, bool4, 4, >, MSL_OP_GT)
MSL_COMPARE_OP(uchar4, uchar, bool4, 4, >=, MSL_OP_GE)
MSL_COMPARE_OP(uchar4, uchar, bool4, 4, ==, MSL_OP_EQ)
MSL_INT_VECTOR_OPS(int2, int, 2)
MSL_INT_VECTOR_OPS(int3, int, 3)
MSL_INT_VECTOR_OPS(int4, int, 4)
MSL_INT_VECTOR_OPS(uint2, uint, 2)
MSL_INT_VECTOR_OPS(uint3, uint, 3)
MSL_INT_VECTOR_OPS(uint4, uint, 4)

)CUDA";

static const char *osl_camera_cuda_compat_2 = R"CUDA(
/* Component-wise math of float vectors. */
#define MSL_FLOAT_FUNC1(f, sf) \
  __forceinline__ float2 f(float2 a) \
  { \
    return MSL_APPLY2(float2, sf, a); \
  } \
  __forceinline__ float3 f(float3 a) \
  { \
    return MSL_APPLY3(float3, sf, a); \
  } \
  __forceinline__ float4 f(float4 a) \
  { \
    return MSL_APPLY4(float4, sf, a); \
  }
#define MSL_FLOAT_FUNC2(f, sf) \
  __forceinline__ float2 f(float2 a, float2 b) \
  { \
    return MSL_APPLY2_2(float2, sf, a, b); \
  } \
  __forceinline__ float3 f(float3 a, float3 b) \
  { \
    return MSL_APPLY3_2(float3, sf, a, b); \
  } \
  __forceinline__ float4 f(float4 a, float4 b) \
  { \
    return MSL_APPLY4_2(float4, sf, a, b); \
  }

MSL_FLOAT_FUNC1(floor, floorf)
MSL_FLOAT_FUNC1(ceil, ceilf)
MSL_FLOAT_FUNC1(trunc, truncf)
MSL_FLOAT_FUNC1(round, roundf)
MSL_FLOAT_FUNC1(rint, rintf)
MSL_FLOAT_FUNC1(abs, fabsf)
MSL_FLOAT_FUNC1(fabs, fabsf)
MSL_FLOAT_FUNC1(sqrt, sqrtf)
MSL_FLOAT_FUNC1(rsqrt, rsqrtf)
MSL_FLOAT_FUNC1(exp, expf)
MSL_FLOAT_FUNC1(exp2, exp2f)
MSL_FLOAT_FUNC1(log, logf)
MSL_FLOAT_FUNC1(log2, log2f)
MSL_FLOAT_FUNC1(log10, log10f)
MSL_FLOAT_FUNC1(sin, sinf)
MSL_FLOAT_FUNC1(cos, cosf)
MSL_FLOAT_FUNC1(tan, tanf)
MSL_FLOAT_FUNC1(asin, asinf)
MSL_FLOAT_FUNC1(acos, acosf)
MSL_FLOAT_FUNC1(atan, atanf)
MSL_FLOAT_FUNC1(sinh, sinhf)
MSL_FLOAT_FUNC1(cosh, coshf)
MSL_FLOAT_FUNC1(tanh, tanhf)
MSL_FLOAT_FUNC1(sign, sign)
MSL_FLOAT_FUNC1(fract, fract)
MSL_FLOAT_FUNC1(saturate, saturate)
MSL_FLOAT_FUNC2(pow, powf)
MSL_FLOAT_FUNC2(fmod, fmodf)
MSL_FLOAT_FUNC2(atan2, atan2f)
MSL_FLOAT_FUNC2(step, step)
MSL_FLOAT_FUNC2(copysign, copysignf)

#define MSL_INT_ABS(V, N) \
  __forceinline__ V abs(V a) \
  { \
    return MSL_APPLY##N(V, ::abs, a); \
  }
MSL_INT_ABS(int2, 2)
MSL_INT_ABS(int3, 3)
MSL_INT_ABS(int4, 4)

__forceinline__ bool3 isnan(float3 a)
{
  return bool3(isnan(a.x), isnan(a.y), isnan(a.z));
}
__forceinline__ bool3 isinf(float3 a)
{
  return bool3(isinf(a.x), isinf(a.y), isinf(a.z));
}
__forceinline__ bool3 isfinite(float3 a)
{
  return bool3(isfinite(a.x), isfinite(a.y), isfinite(a.z));
}

__forceinline__ float2 fma(float2 a, float2 b, float2 c)
{
  return MSL_APPLY2_3(float2, fmaf, a, b, c);
}
__forceinline__ float3 fma(float3 a, float3 b, float3 c)
{
  return MSL_APPLY3_3(float3, fmaf, a, b, c);
}
__forceinline__ float4 fma(float4 a, float4 b, float4 c)
{
  return MSL_APPLY4_3(float4, fmaf, a, b, c);
}
__forceinline__ float2 mix(float2 a, float2 b, float t)
{
  return a + (b - a) * t;
}
__forceinline__ float3 mix(float3 a, float3 b, float t)
{
  return a + (b - a) * t;
}
__forceinline__ float4 mix(float4 a, float4 b, float t)
{
  return a + (b - a) * t;
}
__forceinline__ float2 mix(float2 a, float2 b, float2 t)
{
  return a + (b - a) * t;
}
__forceinline__ float3 mix(float3 a, float3 b, float3 t)
{
  return a + (b - a) * t;
}
__forceinline__ float4 mix(float4 a, float4 b, float4 t)
{
  return a + (b - a) * t;
}

__forceinline__ float dot(float2 a, float2 b)
{
  return a.x * b.x + a.y * b.y;
}
__forceinline__ float dot(float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__forceinline__ float dot(float4 a, float4 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__forceinline__ float3 cross(float3 a, float3 b)
{
  return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__forceinline__ float length(float2 a)
{
  return sqrtf(dot(a, a));
}
__forceinline__ float length(float3 a)
{
  return sqrtf(dot(a, a));
}
__forceinline__ float length(float4 a)
{
  return sqrtf(dot(a, a));
}
__forceinline__ float length_squared(float3 a)
{
  return dot(a, a);
}
__forceinline__ float distance(float3 a, float3 b)
{
  return length(a - b);
}
__forceinline__ float3 normalize(float3 a)
{
  return a * rsqrtf(dot(a, a));
}
__forceinline__ float2 normalize(float2 a)
{
  return a * rsqrtf(dot(a, a));
}
__forceinline__ float3 reflect(float3 i, float3 n)
{
  return i - 2.0f * dot(n, i) * n;
}

/* Matrices, stored as columns like Metal. */
struct float3x3 {
  float3 c[3];
  float3x3() = default;
  constexpr float3x3(float3 a, float3 b, float3 d) : c{a, b, d} {}
  float3 &operator[](int i)
  {
    return c[i];
  }
  float3 operator[](int i) const
  {
    return c[i];
  }
};
struct float4x4 {
  float4 c[4];
  float4x4() = default;
  constexpr float4x4(float s) : c{float4(s, 0, 0, 0), float4(0, s, 0, 0), float4(0, 0, s, 0), float4(0, 0, 0, s)} {}
  constexpr float4x4(float4 a, float4 b, float4 d, float4 e) : c{a, b, d, e} {}
  float4 &operator[](int i)
  {
    return c[i];
  }
  float4 operator[](int i) const
  {
    return c[i];
  }
};
__forceinline__ float4 operator*(float4x4 m, float4 v)
{
  return m.c[0] * v.x + m.c[1] * v.y + m.c[2] * v.z + m.c[3] * v.w;
}
__forceinline__ float4 operator*(float4 v, float4x4 m)
{
  return float4(dot(v, m.c[0]), dot(v, m.c[1]), dot(v, m.c[2]), dot(v, m.c[3]));
}
__forceinline__ float4x4 operator*(float4x4 a, float4x4 b)
{
  return float4x4(a * b.c[0], a * b.c[1], a * b.c[2], a * b.c[3]);
}
__forceinline__ float4x4 operator*(float4x4 a, float s)
{
  return float4x4(a.c[0] * s, a.c[1] * s, a.c[2] * s, a.c[3] * s);
}
__forceinline__ float4x4 operator+(float4x4 a, float4x4 b)
{
  return float4x4(a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2], a.c[3] + b.c[3]);
}
__forceinline__ float4x4 operator-(float4x4 a, float4x4 b)
{
  return float4x4(a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2], a.c[3] - b.c[3]);
}
__forceinline__ float3 operator*(float3x3 m, float3 v)
{
  return m.c[0] * v.x + m.c[1] * v.y + m.c[2] * v.z;
}
__forceinline__ float4x4 transpose(float4x4 m)
{
  return float4x4(float4(m.c[0].x, m.c[1].x, m.c[2].x, m.c[3].x),
                  float4(m.c[0].y, m.c[1].y, m.c[2].y, m.c[3].y),
                  float4(m.c[0].z, m.c[1].z, m.c[2].z, m.c[3].z),
                  float4(m.c[0].w, m.c[1].w, m.c[2].w, m.c[3].w));
}
__forceinline__ float determinant(float4x4 m)
{
  const float a = m.c[0].x, b = m.c[1].x, c = m.c[2].x, d = m.c[3].x;
  const float e = m.c[0].y, f = m.c[1].y, g = m.c[2].y, h = m.c[3].y;
  const float i = m.c[0].z, j = m.c[1].z, k = m.c[2].z, l = m.c[3].z;
  const float n = m.c[0].w, o = m.c[1].w, p = m.c[2].w, q = m.c[3].w;
  const float kq = k * q - l * p, jq = j * q - l * o, jp = j * p - k * o;
  const float iq = i * q - l * n, ip = i * p - k * n, io = i * o - j * n;
  return a * (f * kq - g * jq + h * jp) - b * (e * kq - g * iq + h * ip) +
         c * (e * jq - f * iq + h * io) - d * (e * jp - f * ip + g * io);
}
)CUDA";

/* NVRTC has no C++ standard library. Kernel headers include some of its headers, from which they
 * only use these definitions. */
static const char *osl_camera_cuda_system_stub = R"CUDA(
/* NVRTC has no C++ standard library. The kernel only needs these definitions from it. */
#pragma once
#ifndef CYCLES_NVRTC_SYSTEM_STUB
#  define CYCLES_NVRTC_SYSTEM_STUB
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef unsigned long long size_t;
#  define FLT_MIN 1.175494350822287507969e-38f
#  define FLT_MAX 340282346638528859811704183484516925440.0f
#  define FLT_EPSILON 1.192092896e-07F
#  define INT_MAX 2147483647
#  define INT_MIN (-2147483647 - 1)
#  define UINT_MAX 0xffffffffu
#  define INFINITY __int_as_float(0x7f800000)
#  define NAN __int_as_float(0x7fffffff)
#endif
)CUDA";

/* `util/types_rgbe.h` for NVRTC, which does not accept the anonymous struct of RGBE. */
static const char *osl_camera_cuda_types_rgbe = R"CUDA(
/* util/types_rgbe.h for runtime compilation with NVRTC, which does not accept the anonymous
 * struct of RGBE. The bytes r, g, b, e are packed into the bits of `f` in memory order, so values
 * are identical to the regular header. */
#pragma once

#include "util/math_fast.h"
#include "util/math_float3.h"
#include "util/types_base.h"

CCL_NAMESPACE_BEGIN

struct RGBE {
  float f;

  RGBE() = default;

  ccl_device_inline_method RGBE(const float f_) : f(f_) {}
};

#define RGBE_EXP_BIAS 15
#define RGBE_MANTISSA_BITS 8
#define RGBE_EXPONENT_BITS 5
#define RGBE_MAX 65280.0f

ccl_device RGBE rgb_to_rgbe(float3 rgb)
{
  const float max_v = min(reduce_max(fabs(rgb)), RGBE_MAX);
  if (max_v < ldexpf(0.5f, -RGBE_EXP_BIAS - RGBE_MANTISSA_BITS)) {
    return RGBE(0.0f);
  }

  int e = max(-RGBE_EXP_BIAS - 1, floor_log2f(max_v)) + 1;
  float v = ldexpf(1.0f, RGBE_MANTISSA_BITS - e);

  if (int(roundf(max_v * v)) == power_of_2(RGBE_MANTISSA_BITS)) {
    e += 1;
    v *= 0.5f;
  }

  const uint sign_bits = ((__float_as_uint(rgb.x) >> 31) << 7) |
                         ((__float_as_uint(rgb.y) >> 31) << 6) |
                         ((__float_as_uint(rgb.z) >> 31) << 5);

  rgb = min(round(fabs(rgb) * v), make_float3(255.0f));
  const uint r = uint(uint8_t(rgb.x));
  const uint g = uint(uint8_t(rgb.y));
  const uint b = uint(uint8_t(rgb.z));
  const uint ex = uint(uint8_t(((e + RGBE_EXP_BIAS) & 0x1Fu) | sign_bits));
  return RGBE(__uint_as_float(r | (g << 8) | (b << 16) | (ex << 24)));
}

ccl_device_inline float3 rgbe_to_rgb(const RGBE rgbe)
{
  if (rgbe.f == 0.0f) {
    return zero_float3();
  }

  const uint bits = __float_as_uint(rgbe.f);
  const uint r = bits & 0xFFu, g = (bits >> 8) & 0xFFu, b = (bits >> 16) & 0xFFu;
  const uint ex = bits >> 24;
  const int e = ex & 0x1Fu;
  const float f = ldexpf(1.0f, e - (int)(RGBE_EXP_BIAS + RGBE_MANTISSA_BITS));
  float3 result = make_float3(float(r), float(g), float(b)) * f;

  result.x = or_mask(result.x, (ex & 0x80u) << 24);
  result.y = or_mask(result.y, (ex & 0x40u) << 25);
  result.z = or_mask(result.z, (ex & 0x20u) << 26);
  return result;
}

CCL_NAMESPACE_END
)CUDA";

CCL_NAMESPACE_END
