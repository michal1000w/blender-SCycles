/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline float2 diffraction_complex_mul(const float2 a, const float2 b)
{
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

ccl_device_inline float2 diffraction_complex_div(const float2 a, const float2 b)
{
  /* Scaled division avoids squaring the pivot magnitude. */
  if (fabsf(b.x) >= fabsf(b.y)) {
    const float ratio = b.y / b.x;
    return make_float2(a.x + a.y * ratio, a.y - a.x * ratio) / (b.x + b.y * ratio);
  }
  const float ratio = b.x / b.y;
  return make_float2(a.x * ratio + a.y, a.y * ratio - a.x) / (b.y + b.x * ratio);
}

/* Four float2 slots per port: r_TE, r_TM, (t_TE,t_TM), and the unit tangential
 * direction. T is zero for a nonpropagating exterior channel. The coefficient
 * construction is separate from the matching solve. The pointer type is deduced
 * so Metal can read either a device buffer or coefficients constructed in thread
 * memory without copying through a global buffer. */
template<typename BoundaryPointer>
ccl_device_inline float2 diffraction_reference_r(const BoundaryPointer boundary,
                                                 const int port,
                                                 const int row,
                                                 const int col)
{
  const float2 direction = boundary[4 * port + 3];
  const float a = row ? direction.y : direction.x;
  const float b = col ? direction.y : direction.x;
  const float projection = a * b;
  return (float(row == col) - projection) * boundary[4 * port] +
         projection * boundary[4 * port + 1];
}

template<typename BoundaryPointer>
ccl_device_inline float diffraction_reference_t(const BoundaryPointer boundary,
                                                const int port,
                                                const int row,
                                                const int col)
{
  const float2 direction = boundary[4 * port + 3];
  const float a = row ? direction.y : direction.x;
  const float b = col ? direction.y : direction.x;
  const float projection = a * b;
  const float2 t = boundary[4 * port + 2];
  return (float(row == col) - projection) * t.x + projection * t.y;
}

/* Gaussian elimination with partial pivoting, two simultaneous right-hand
 * sides. The caller owns scratch storage, allowing specialization by channel
 * count rather than allocating a maximum-size matrix at every shading point. */
template<int N>
ccl_device_inline bool diffraction_complex_solve(ccl_private float2 *matrix,
                                                 ccl_private float2 *rhs)
{
  for (int k = 0; k < N; k++) {
    int pivot = k;
    float magnitude = len_squared(matrix[k * N + k]);
    for (int row = k + 1; row < N; row++) {
      const float candidate = len_squared(matrix[row * N + k]);
      if (candidate > magnitude) {
        pivot = row;
        magnitude = candidate;
      }
    }
    if (!(magnitude > 0.0f) || !isfinite_safe(magnitude)) {
      return false;
    }
    if (pivot != k) {
      for (int col = k; col < N; col++) {
        const float2 value = matrix[k * N + col];
        matrix[k * N + col] = matrix[pivot * N + col];
        matrix[pivot * N + col] = value;
      }
      for (int col = 0; col < 2; col++) {
        const float2 value = rhs[2 * k + col];
        rhs[2 * k + col] = rhs[2 * pivot + col];
        rhs[2 * pivot + col] = value;
      }
    }
    for (int row = k + 1; row < N; row++) {
      const float2 factor = diffraction_complex_div(matrix[row * N + k], matrix[k * N + k]);
      for (int col = k + 1; col < N; col++) {
        matrix[row * N + col] = matrix[row * N + col] -
                                diffraction_complex_mul(factor, matrix[k * N + col]);
      }
      for (int col = 0; col < 2; col++) {
        rhs[2 * row + col] = rhs[2 * row + col] -
                             diffraction_complex_mul(factor, rhs[2 * k + col]);
      }
    }
  }
  for (int row = N - 1; row >= 0; row--) {
    for (int col = 0; col < 2; col++) {
      float2 value = rhs[2 * row + col];
      for (int j = row + 1; j < N; j++) {
        value = value - diffraction_complex_mul(matrix[row * N + j], rhs[2 * j + col]);
      }
      rhs[2 * row + col] = diffraction_complex_div(value, matrix[row * N + row]);
      if (!isfinite_safe(rhs[2 * row + col].x) || !isfinite_safe(rhs[2 * row + col].y)) {
        return false;
      }
    }
  }
  return true;
}

/* Match one incoming physical port to all outgoing ports. Output is an N x 2
 * complex Jones matrix in the physical Cartesian flux basis. Exterior R/T
 * coefficients must correspond to the same query as the interpolated chart.
 * No probability renormalization or clipping is applied. Numerical failure is
 * reported explicitly; integration must not silently discard such paths. */
template<int N, typename ChartPointer, typename BoundaryPointer>
ccl_device_inline bool diffraction_reference_match(const ChartPointer chart,
                                                   const BoundaryPointer boundary,
                                                   const float2 rotation,
                                                   const int incoming_port,
                                                   ccl_private float2 *jones)
{
  static_assert(N > 0 && N % 2 == 0, "Reference channels come in polarization pairs");
  for (int i = 0; i < 2 * N; i++) {
    jones[i] = zero_float2();
  }
  if (incoming_port < 0 || incoming_port >= N / 2 || !(boundary[4 * incoming_port + 2].x > 0.0f)) {
    return false;
  }
  float2 matrix[N * N], rhs[2 * N];
  const float2 inverse_rotation = make_float2(rotation.x, -rotation.y);
  for (int row = 0; row < N; row++) {
    const int port = row / 2, component = row % 2, peer = row ^ 1;
    const float2 diagonal = diffraction_reference_r(boundary, port, component, component);
    const float2 off_diagonal = diffraction_reference_r(boundary, port, component, 1 - component);
    for (int col = 0; col < N; col++) {
      const float2 value = chart[row * N + col];
      const float2 identity = make_float2(float(row == col), 0.0f);
      const float2 other_identity = make_float2(float(peer == col), 0.0f);
      const float2 reflected = diffraction_complex_mul(diagonal, identity - value) +
                               diffraction_complex_mul(off_diagonal,
                                                       other_identity - chart[peer * N + col]);
      matrix[row * N + col] = identity + value -
                              diffraction_complex_mul(inverse_rotation, reflected);
    }
    for (int col = 0; col < 2; col++) {
      rhs[2 * row + col] = make_float2(
          port == incoming_port ? diffraction_reference_t(boundary, port, component, col) : 0.0f,
          0.0f);
    }
  }
  if (!diffraction_complex_solve<N>(matrix, rhs)) {
    return false;
  }
  /* Reuse scratch for the reference outgoing field. */
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < 2; col++) {
      float2 value = rhs[2 * row + col];
      for (int j = 0; j < N; j++) {
        value = value - diffraction_complex_mul(chart[row * N + j], rhs[2 * j + col]);
      }
      matrix[2 * row + col] = diffraction_complex_mul(inverse_rotation, value);
    }
  }
  for (int row = 0; row < N; row++) {
    const int port = row / 2, component = row % 2, first = 2 * port;
    for (int col = 0; col < 2; col++) {
      const float2 value =
          diffraction_reference_t(boundary, port, component, 0) * matrix[2 * first + col] +
          diffraction_reference_t(boundary, port, component, 1) * matrix[2 * (first + 1) + col] -
          (port == incoming_port ? diffraction_reference_r(boundary, port, component, col) :
                                   zero_float2());
      if (!isfinite_safe(value.x) || !isfinite_safe(value.y)) {
        return false;
      }
      jones[2 * row + col] = value;
    }
  }
  return true;
}

/* Matrix-anchored chart S=U*(I-Y)*(I+Y)^-1. Unlike a scalar rotation,
 * U can align all channels to a local reference scattering operator. This
 * avoids chart poles caused solely by overlapping eigenphase ranges.
 * Match directly: V=U*(I-Y), A=(I+Y)-R*V, A*x=T, output=T*V*x-R.
 * The two incident polarizations share one factorization. The supplied anchor
 * and chart must have the same reference-port basis and phase planes.
 * This primitive does not certify interpolation accuracy or anchor passivity. */
template<int N, typename ChartPointer, typename AnchorPointer, typename BoundaryPointer>
/* Explicit outlining bounds Metal compilation of the fixed-channel dispatch.
 * Apple's generic ccl_device_noinline deliberately permits inlining. */
#ifdef __KERNEL_METAL__
ccl_device __attribute__((noinline))
#else
ccl_device_inline
#endif
bool diffraction_reference_match_anchor(const ChartPointer chart,
                                                          const AnchorPointer anchor,
                                                          const BoundaryPointer boundary,
                                                          const int incoming_port,
                                                          ccl_private float2 *jones)
{
  static_assert(N > 0 && N % 2 == 0);
  for (int i = 0; i < 2 * N; i++)
    jones[i] = zero_float2();
  if (incoming_port < 0 || incoming_port >= N / 2 ||
      !(boundary[4 * incoming_port + 2].x > 0.0f))
    return false;
  float2 outgoing[N * N], matrix[N * N], rhs[2 * N];
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      float2 value = anchor[row * N + col];
      for (int k = 0; k < N; k++)
        value = value - diffraction_complex_mul(anchor[row * N + k], chart[k * N + col]);
      outgoing[row * N + col] = value;
    }
  }
  for (int row = 0; row < N; row++) {
    const int port = row / 2, component = row % 2, peer = row ^ 1;
    const float2 diagonal = diffraction_reference_r(boundary, port, component, component);
    const float2 off_diagonal = diffraction_reference_r(boundary, port, component, 1 - component);
    for (int col = 0; col < N; col++) {
      matrix[row * N + col] = make_float2(float(row == col), 0.0f) + chart[row * N + col] -
                              diffraction_complex_mul(diagonal, outgoing[row * N + col]) -
                              diffraction_complex_mul(off_diagonal, outgoing[peer * N + col]);
    }
    for (int col = 0; col < 2; col++)
      rhs[2 * row + col] = make_float2(
          port == incoming_port ? diffraction_reference_t(boundary, port, component, col) : 0.0f,
          0.0f);
  }
  if (!diffraction_complex_solve<N>(matrix, rhs))
    return false;
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < 2; col++) {
      float2 value = zero_float2();
      for (int k = 0; k < N; k++)
        value = value + diffraction_complex_mul(outgoing[row * N + k], rhs[2 * k + col]);
      matrix[2 * row + col] = value;
    }
  }
  for (int row = 0; row < N; row++) {
    const int port = row / 2, component = row % 2, first = 2 * port;
    for (int col = 0; col < 2; col++) {
      const float2 value =
          diffraction_reference_t(boundary, port, component, 0) * matrix[2 * first + col] +
          diffraction_reference_t(boundary, port, component, 1) * matrix[2 * (first + 1) + col] -
          (port == incoming_port ? diffraction_reference_r(boundary, port, component, col) :
                                   zero_float2());
      if (!isfinite_safe(value.x) || !isfinite_safe(value.y))
        return false;
      jones[2 * row + col] = value;
    }
  }
  return true;
}

/* Destructive chart solve. With c=conj(rotation), A=(I+Y)-c*R*(I-Y),
 * A*x=T. Thus (I+c*R)*z=c*(2*x-T), avoiding retention of Y after
 * elimination. R and T share the TE/TM eigenspaces at each exterior port. */
template<int N, typename BoundaryPointer>
ccl_device_inline bool diffraction_reference_match_inplace(ccl_private float2 *chart,
                                                           const BoundaryPointer boundary,
                                                           const float2 rotation,
                                                           const int incoming_port,
                                                           ccl_private float2 *jones)
{
  static_assert(N > 0 && N % 2 == 0);
  for (int i = 0; i < 2 * N; i++)
    jones[i] = zero_float2();
  if (incoming_port < 0 || incoming_port >= N / 2 || !(boundary[4 * incoming_port + 2].x > 0))
    return false;
  const float2 c = make_float2(rotation.x, -rotation.y);
  /* The reconstruction divides by I+cR. At nearly grazing propagation and
   * nearly real rotations this can magnify cancellation in 2*x-T. Keep the
   * original formulation for those ports, before overwriting any chart data. */
  for (int port = 0; port < N / 2; port++) {
    if (!(boundary[4 * port + 2].x > 0))
      continue;
    for (int polarization = 0; polarization < 2; polarization++) {
      const float2 denominator = make_float2(1, 0) +
                                 diffraction_complex_mul(c, boundary[4 * port + polarization]);
      if (len_squared(denominator) < 0.0625f)
        return diffraction_reference_match<N>(chart, boundary, rotation, incoming_port, jones);
    }
  }
  float2 rhs[2 * N];
  for (int port = 0; port < N / 2; port++) {
    const int r = 2 * port;
    const float2 r00 = diffraction_reference_r(boundary, port, 0, 0);
    const float2 r01 = diffraction_reference_r(boundary, port, 0, 1);
    const float2 r11 = diffraction_reference_r(boundary, port, 1, 1);
    for (int col = 0; col < N; col++) {
      const float2 a = chart[r * N + col], b = chart[(r + 1) * N + col];
      const float2 i0 = make_float2(float(r == col), 0), i1 = make_float2(float(r + 1 == col), 0);
      chart[r * N + col] = i0 + a -
                           diffraction_complex_mul(c,
                                                   diffraction_complex_mul(r00, i0 - a) +
                                                       diffraction_complex_mul(r01, i1 - b));
      chart[(r + 1) * N + col] = i1 + b -
                                 diffraction_complex_mul(c,
                                                         diffraction_complex_mul(r01, i0 - a) +
                                                             diffraction_complex_mul(r11, i1 - b));
    }
    for (int row = 0; row < 2; row++)
      for (int col = 0; col < 2; col++)
        rhs[2 * (r + row) + col] = make_float2(
            port == incoming_port ? diffraction_reference_t(boundary, port, row, col) : 0, 0);
  }
  if (!diffraction_complex_solve<N>(chart, rhs))
    return false;
  for (int port = 0; port < N / 2; port++) {
    const float2 transmission = boundary[4 * port + 2], direction = boundary[4 * port + 3];
    if (!(transmission.x > 0))
      continue;
    const float2 te = diffraction_complex_div(
        transmission.x * c, make_float2(1, 0) + diffraction_complex_mul(c, boundary[4 * port]));
    const float2 tm = diffraction_complex_div(
        transmission.y * c,
        make_float2(1, 0) + diffraction_complex_mul(c, boundary[4 * port + 1]));
    for (int col = 0; col < 2; col++) {
      const float2 x = 2.0f * rhs[4 * port + col] -
                       make_float2(port == incoming_port ?
                                       diffraction_reference_t(boundary, port, 0, col) :
                                       0,
                                   0);
      const float2 y = 2.0f * rhs[4 * port + 2 + col] -
                       make_float2(port == incoming_port ?
                                       diffraction_reference_t(boundary, port, 1, col) :
                                       0,
                                   0);
      const float2 projection = direction.x * x + direction.y * y;
      const float2 values[2] = {x, y};
      for (int row = 0; row < 2; row++) {
        const float2 parallel = (row ? direction.y : direction.x) * projection;
        const float2 value = diffraction_complex_mul(te, values[row] - parallel) +
                             diffraction_complex_mul(tm, parallel) -
                             (port == incoming_port ?
                                  diffraction_reference_r(boundary, port, row, col) :
                                  zero_float2());
        if (!isfinite_safe(value.x) || !isfinite_safe(value.y))
          return false;
        jones[4 * port + 2 * row + col] = value;
      }
    }
  }
  return true;
}

/* Interpolate a common passive chart, then perform one physical matching solve.
 * The eight matrices share a rotation and hybrid port basis. Dense scratch is
 * specialized by channel count; intensity cells use the reduced solver below. */
template<int N, int Degree = 1, bool InPlace = false, typename BoundaryPointer>
ccl_device_inline bool diffraction_chart_cell_match(ccl_global const float2 *corners,
                                                    const BoundaryPointer boundary,
                                                    const float2 rotation,
                                                    const int incoming_port,
                                                    const float3 coordinate,
                                                    ccl_private float2 *jones)
{
  if (!isfinite_safe(coordinate.x) || !isfinite_safe(coordinate.y) ||
      !isfinite_safe(coordinate.z) || coordinate.x < 0 || coordinate.x > 1 || coordinate.y < 0 ||
      coordinate.y > 1 || coordinate.z < 0 || coordinate.z > 1)
    return false;
  static_assert(Degree == 1 || Degree == 2, "Unsupported chart polynomial degree");
  constexpr int Count = Degree == 1 ? 8 : 27;
  float weights[Count];
  if constexpr (Degree == 1) {
    for (int c = 0; c < Count; c++)
      weights[c] = (c & 1 ? coordinate.x : 1.0f - coordinate.x) *
                   (c & 2 ? coordinate.y : 1.0f - coordinate.y) *
                   (c & 4 ? coordinate.z : 1.0f - coordinate.z);
  }
  else {
    const float3 u = make_float3(1.0f) - coordinate;
    const float3 low = u * u, middle = 2.0f * coordinate * u, high = coordinate * coordinate;
    for (int c = 0; c < Count; c++) {
      const int x = c % 3, y = (c / 3) % 3, z = c / 9;
      weights[c] = (x == 0 ? low.x :
                    x == 1 ? middle.x :
                             high.x) *
                   (y == 0 ? low.y :
                    y == 1 ? middle.y :
                             high.y) *
                   (z == 0 ? low.z :
                    z == 1 ? middle.z :
                             high.z);
    }
  }
  float2 chart[N * N];
  for (int j = 0; j < N * N; j++) {
    float2 value = zero_float2();
    for (int c = 0; c < Count; c++)
      value += weights[c] * corners[c * N * N + j];
    chart[j] = value;
  }
  if constexpr (InPlace)
    return diffraction_reference_match_inplace<N>(chart, boundary, rotation, incoming_port, jones);
  return diffraction_reference_match<N>(chart, boundary, rotation, incoming_port, jones);
}

/* Hybrid matching uses only C feedback channels, in polarization pairs. The
 * active port indices identify ALL reference ports; every other port has R=0,
 * T=I. Operator validation and cutoff coverage belong to cache construction.
 * The dense source matrix can have a runtime channel count without allocating
 * an equally large private LU matrix at each shading point. */
template<int C, typename BoundaryPointer>
ccl_device_inline bool diffraction_hybrid_match(ccl_global const float2 *scattering,
                                                const BoundaryPointer boundary,
                                                ccl_global const int *active_ports,
                                                const int channels,
                                                const int incoming_port,
                                                ccl_private float2 *jones)
{
  static_assert(C >= 0 && C % 2 == 0, "Feedback channels come in polarization pairs");
  if (channels <= 0 || channels % 2 != 0 || incoming_port < 0 || incoming_port >= channels / 2 ||
      !(boundary[4 * incoming_port + 2].x > 0.0f))
  {
    return false;
  }
  for (int i = 0; i < C / 2; i++) {
    if (active_ports[i] < 0 || active_ports[i] >= channels / 2 ||
        (i > 0 && active_ports[i] <= active_ports[i - 1]))
    {
      return false;
    }
  }
  float2 matrix[C > 0 ? C * C : 1], rhs[C > 0 ? 2 * C : 1];
  for (int row = 0; row < C; row++) {
    const int source_row = 2 * active_ports[row / 2] + row % 2;
    for (int col = 0; col < C; col++) {
      const int port = active_ports[col / 2], component = col % 2;
      const float2 feedback =
          diffraction_complex_mul(scattering[source_row * channels + 2 * port],
                                  diffraction_reference_r(boundary, port, 0, component)) +
          diffraction_complex_mul(scattering[source_row * channels + 2 * port + 1],
                                  diffraction_reference_r(boundary, port, 1, component));
      matrix[row * C + col] = make_float2(float(row == col), 0.0f) - feedback;
    }
    for (int col = 0; col < 2; col++) {
      rhs[2 * row + col] = scattering[source_row * channels + 2 * incoming_port] *
                               diffraction_reference_t(boundary, incoming_port, 0, col) +
                           scattering[source_row * channels + 2 * incoming_port + 1] *
                               diffraction_reference_t(boundary, incoming_port, 1, col);
    }
  }
  if constexpr (C > 0) {
    if (!diffraction_complex_solve<C>(matrix, rhs)) {
      return false;
    }
  }
  for (int port = 0; port < channels / 2; port++) {
    float2 outgoing[4];
    for (int component = 0; component < 2; component++) {
      const int row = 2 * port + component;
      for (int col = 0; col < 2; col++) {
        float2 value = scattering[row * channels + 2 * incoming_port] *
                           diffraction_reference_t(boundary, incoming_port, 0, col) +
                       scattering[row * channels + 2 * incoming_port + 1] *
                           diffraction_reference_t(boundary, incoming_port, 1, col);
        for (int j = 0; j < C; j++) {
          const int active = active_ports[j / 2], polarization = j % 2;
          const float2 feedback =
              diffraction_complex_mul(scattering[row * channels + 2 * active],
                                      diffraction_reference_r(boundary, active, 0, polarization)) +
              diffraction_complex_mul(scattering[row * channels + 2 * active + 1],
                                      diffraction_reference_r(boundary, active, 1, polarization));
          value += diffraction_complex_mul(feedback, rhs[2 * j + col]);
        }
        outgoing[2 * component + col] = value;
      }
    }
    for (int component = 0; component < 2; component++) {
      for (int col = 0; col < 2; col++) {
        const float2 value =
            diffraction_reference_t(boundary, port, component, 0) * outgoing[col] +
            diffraction_reference_t(boundary, port, component, 1) * outgoing[2 + col] -
            (port == incoming_port ? diffraction_reference_r(boundary, port, component, col) :
                                     zero_float2());
        if (!isfinite_safe(value.x) || !isfinite_safe(value.y)) {
          return false;
        }
        jones[4 * port + 2 * component + col] = value;
      }
    }
  }
  return true;
}

/* Match all eight prepared hybrid corners to one query before interpolating
 * intensities. Interpolating the complex hybrid matrices directly would not
 * preserve lossless energy. This routine is for incoherent transport only.
 * Matrices are tightly packed corner-major with channels*channels float2 each.
 * The host guarantees identical port topology and feedback sets at all corners. */
template<int MaxChannels, int C, typename BoundaryPointer>
ccl_device_inline bool diffraction_hybrid_cell_power(ccl_global const float2 *matrices,
                                                     const BoundaryPointer boundary,
                                                     ccl_global const int *active_ports,
                                                     const int channels,
                                                     const int incoming_port,
                                                     const float3 coordinate,
                                                     ccl_private float *powers)
{
  static_assert(MaxChannels > 0 && MaxChannels % 2 == 0);
  if (channels <= 0 || channels > MaxChannels || channels % 2 != 0 ||
      !isfinite_safe(coordinate.x) || !isfinite_safe(coordinate.y) ||
      !isfinite_safe(coordinate.z) || coordinate.x < 0.0f || coordinate.x > 1.0f ||
      coordinate.y < 0.0f || coordinate.y > 1.0f || coordinate.z < 0.0f || coordinate.z > 1.0f)
  {
    return false;
  }
  for (int p = 0; p < channels / 2; p++) {
    powers[p] = 0.0f;
  }
  for (int corner = 0; corner < 8; corner++) {
    const float weight = (corner & 1 ? coordinate.x : 1.0f - coordinate.x) *
                         (corner & 2 ? coordinate.y : 1.0f - coordinate.y) *
                         (corner & 4 ? coordinate.z : 1.0f - coordinate.z);
    if (weight == 0.0f) {
      continue;
    }
    float2 jones[2 * MaxChannels];
    if (!diffraction_hybrid_match<C>(matrices + corner * channels * channels,
                                     boundary,
                                     active_ports,
                                     channels,
                                     incoming_port,
                                     jones))
    {
      return false;
    }
    for (int p = 0; p < channels / 2; p++) {
      const float power = 0.5f * (len_squared(jones[4 * p]) + len_squared(jones[4 * p + 1]) +
                                  len_squared(jones[4 * p + 2]) + len_squared(jones[4 * p + 3]));
      powers[p] += weight * power;
      if (!isfinite_safe(powers[p])) {
        return false;
      }
    }
  }
  return true;
}

CCL_NAMESPACE_END
