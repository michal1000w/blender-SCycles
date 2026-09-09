/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <metal_stdlib>
#define __KERNEL_GPU__
#define __KERNEL_METAL__
#define __KERNEL_METAL_APPLE__
#include "kernel/device/metal/compat.h"
#include "kernel/integrator/bidirectional_transport.h"
#include "kernel/integrator/bidirectional_volume.h"
#include "kernel/sample/guiding_distribution.h"
#include "kernel/sample/guiding_field.h"
#include "kernel/sample/guiding_resampling.h"
#include "kernel/sample/guiding_spherical_gaussian.h"
#include "kernel/sample/manifold.h"
#include "kernel/sample/sobol_burley.h"
#include "util/hash.h"

kernel void bdpt_mis_recurrence(device const float4 *events [[buffer(0)]],
                                device float4 *output [[buffer(1)]],
                                uint index [[thread_position_in_grid]])
{
  float2 balance = float2(1.25f, .75f);
  float2 power = balance * balance;
  for (uint i = 0; i < 16; ++i) {
    const float4 e = events[index * 16 + i];
    balance = BDPTMISRecurrence<1>::scatter(balance, e.x, e.y, e.z, e.w);
    power = BDPTMISRecurrence<2>::scatter(power, e.x, e.y, e.z, e.w);
  }
  output[index] = float4(balance.x, balance.y, power.x, power.y);
}

kernel void bdpt_mis_log_recurrence(device const float4 *events [[buffer(0)]],
                                    device float4 *output [[buffer(1)]],
                                    uint index [[thread_position_in_grid]])
{
  float2 balance = float2(log(1.25f), log(.75f));
  float2 power = 2.0f * balance;
  for (uint i = 0; i < 16; ++i) {
    const float4 e = events[index * 16 + i];
    balance = BDPTMISLogRecurrence<1>::scatter(balance, e.x, e.y, e.z, e.w);
    power = BDPTMISLogRecurrence<2>::scatter(power, e.x, e.y, e.z, e.w);
  }
  output[index] = float4(balance.x, balance.y, power.x, power.y);
}

kernel void bdpt_mis_connection_weights(device const float4 *input [[buffer(0)]],
                                        device float2 *output [[buffer(1)]],
                                        uint index [[thread_position_in_grid]])
{
  const float4 v = input[index];
  using Balance = BDPTMISWeightT<1>;
  using Power = BDPTMISWeightT<2>;
  const Balance a = Balance::from_encoded(v.x) + Balance::from_encoded(v.y);
  const Power b = Power::from_encoded(2 * v.x) + Power::from_encoded(2 * v.y);
  output[index] = float2(a.connection_weight(v.z), b.connection_weight(v.z));
}

kernel void bdpt_volume_homogeneous_events(device float4 *output [[buffer(0)]],
                                           uint index [[thread_position_in_grid]])
{
  const float majorants[5] = {1.4f, 3.0f, 7.0f, .6f, 1.0f};
  const float majorant = majorants[index % 5];
  const auto pure = BDPTVolumeEventDistribution::make(
      float3(majorant), float3(majorant), majorant);
  const auto vacuum = BDPTVolumeEventDistribution::make(float3(0), float3(0), majorant);
  if (!pure.valid || pure.real_probability != 1 || pure.null_probability != 0 ||
      metal::any(pure.real_weight != float3(1)) || !vacuum.valid || vacuum.real_probability != 0 ||
      metal::any(vacuum.null_weight != float3(1)))
  {
    output[index] = float4(0, 0, 0, -3);
    return;
  }
  const auto event = BDPTVolumeEventDistribution::make(
      float3(.3f, .8f, 1.4f), float3(.2f, .6f, .5f), majorant);
  if (!event.valid) {
    output[index] = float4(0, 0, 0, -1);
    return;
  }
  float distance = 0;
  float3 weight = float3(1);
  for (uint step = 0; step < 256; ++step) {
    const float u = uint_to_float_excl(hash_uint2(index, 2 * step + 937u));
    distance += -log(1 - u) / majorant;
    if (distance >= 2.0f) {
      output[index] = float4(weight, 0);
      return;
    }
    const float choice = uint_to_float_excl(hash_uint2(index, 2 * step + 938u));
    if (event.sample_real(choice)) {
      output[index] = float4(weight * event.real_weight, 1);
      return;
    }
    weight *= event.null_weight;
  }
  output[index] = float4(0, 0, 0, -2);  // Exceeding the test guard is a failure.
}

/* Three piecewise-constant slabs with forward delta scattering. Continuing
 * after every real collision admits independent analytic answers for zero,
 * one, and multiple collisions. This tests transport composition, not Cycles
 * shader/octree traversal or phase-function sampling. */
kernel void bdpt_volume_piecewise_events(device float4 *output [[buffer(0)]],
                                         uint index [[thread_position_in_grid]])
{
  const float lengths[3] = {.7f, .8f, .5f};
  const float3 extinction[3] = {
      float3(1.1f, .9f, 1.3f), float3(.4f, 1.4f, .8f), float3(1.6f, .6f, 1.0f)};
  const float3 scattering[3] = {
      float3(.7f, .8f, .8f), float3(.3f, 1.0f, .5f), float3(1.0f, .4f, .8f)};
  const uint mode = index % 4;
  const bool reverse = (mode & 1) != 0;
  const bool underestimated = mode >= 2;
  float3 weight(1);
  uint real_events = 0, dimension = 0;
  for (uint segment = 0; segment < 3; ++segment) {
    const uint slab = reverse ? 2 - segment : segment;
    const float rate = reduce_max(extinction[slab]) * (underestimated ? .7f : 1.2f);
    const auto event = BDPTVolumeEventDistribution::make(extinction[slab], scattering[slab], rate);
    if (!event.valid) {
      output[index] = float4(0, 0, 0, -1);
      return;
    }
    float position = 0;
    bool completed = false;
    for (uint step = 0; step < 256; ++step) {
      const float distance_sample = uint_to_float_excl(hash_uint2(index, 12719u + dimension++));
      position += -log(1 - distance_sample) / rate;
      if (position >= lengths[slab]) {
        completed = true;
        break;
      }
      const float event_sample = uint_to_float_excl(hash_uint2(index, 12719u + dimension++));
      if (event.sample_real(event_sample)) {
        weight *= event.real_weight;
        ++real_events;
      }
      else {
        weight *= event.null_weight;
      }
    }
    if (!completed) {
      output[index] = float4(0, 0, 0, -2);
      return;
    }
  }
  const float continuation = bdpt_light_path_continuation_probability(weight, float3(1));
  const float roulette = uint_to_float_excl(hash_uint2(index, 12719u + dimension));
  weight = roulette < continuation ? weight / continuation : float3(0);
  output[index] = float4(weight, float(real_events));
}

kernel void manifold_interface_samples(device float4 *result [[buffer(0)]],
                                       uint index [[thread_position_in_grid]])
{
  float4 values = float4(0.0f);
  for (uint interface_index = 0; interface_index < 3; ++interface_index) {
    values[interface_index] =
        sobol_burley_sample_2D(
            index, manifold_vertex_rng_dimension(interface_index), 0x392ff712u, ~0u)
            .x -
        0.5f;
  }
  result[index] = values;
}

kernel void guiding_resample(device const float2 *random [[buffer(0)]],
                             device float *result [[buffer(1)]],
                             uint index [[thread_position_in_grid]])
{
  const float p[4] = {0.0f, 0.2f, 0.3f, 0.5f};
  const float q[4] = {0.6f, 0.3f, 0.1f, 0.0f};
  const float target[4] = {2.0f, 0.0f, 8.0f, 0.3f};
  const float integrand[4] = {0.3f, 0.0f, 12.0f, 0.8f};
  const float mis[4] = {0.2f, 0.6f, 0.8f, 0.3f};
  uint candidate[2] = {3, 3};
  float2 value = random[index];
  for (uint i = 0; i < 3; ++i) {
    if (candidate[0] == 3 && value.x < p[i]) {
      candidate[0] = i;
    }
    if (candidate[1] == 3 && value.y < q[i]) {
      candidate[1] = i;
    }
    value -= float2(p[i], q[i]);
  }
  GuidingResamplingPair pair{
      {target[candidate[0]], target[candidate[1]]},
      {0.5f * (p[candidate[0]] + q[candidate[0]]), 0.5f * (p[candidate[1]] + q[candidate[1]])}};
  float pdf;
  const int selected = pair.sample(hash_uint_to_float(index ^ 0x72697330u), &pdf);
  result[index] = selected >= 0 ? integrand[candidate[selected]] * mis[candidate[selected]] / pdf :
                                  0.0f;
}

kernel void guiding_gaussian_sample(device const float2 *random [[buffer(0)]],
                                    device float4 *result [[buffer(1)]],
                                    constant uint &test [[buffer(2)]],
                                    device const float *mixture_storage [[buffer(3)]],
                                    uint index [[thread_position_in_grid]])
{
  if (test >= 7) {
    const GuidingGaussianProduct profile{
        {{normalize(float3(1, 2, 3)), test != 8 ? 5.0f : 16384.0f},
         {normalize(float3(-2, 1, -3)), 30.0f}},
        {test != 8 ? 0.3f : 1.0f, test != 8 ? 0.7f : 0.0f}};
    GuidingGaussianMixture mixture;
    const float3 position = test == 9  ? float3(.2f, .3f, .4f) :
                            test == 10 ? float3(.8f, .7f, .6f) :
                                         float3(0);
    float pdf;
    const float3 direction = mixture.sample_product(
        mixture_storage, profile, random[index], &pdf, position);
    result[2 * index] = float4(direction, pdf);
    result[2 * index + 1] = float4(
        mixture.pdf_product(mixture_storage, profile, direction, position));
    return;
  }
  const float concentrations[] = {0.0f, 1e-5f, 0.01f, 1.0f, 32.0f, 16384.0f};
  GuidingSphericalGaussian distribution{normalize(float3(1, -2, 3)),
                                        concentrations[min(test, 5u)]};
  if (test == 6) {
    const GuidingSphericalGaussian a{normalize(float3(1, 2, 3)), 5.0f};
    const GuidingSphericalGaussian b{normalize(float3(-2, 1, 3)), 12.0f};
    float integral;
    distribution = a.product(b, &integral);
  }
  float pdf;
  const float3 direction = distribution.sample(random[index], &pdf);
  result[2 * index] = float4(direction, pdf);
  result[2 * index + 1] = float4(distribution.pdf(direction));
}

kernel void guiding_build(device float *tree [[buffer(0)]],
                          device float *counts [[buffer(1)]],
                          constant uint &test [[buffer(2)]])
{
  GuidingDirectionalTree<5> distribution;
  if (test >= 9) {
    distribution.build_adaptive(tree, counts, 32.0f, 0.05f);
  }
  else {
    distribution.build(tree, 0.05f);
  }
}

kernel void guiding_sample(device const float *tree [[buffer(0)]],
                           device const float2 *random [[buffer(1)]],
                           device float4 *result [[buffer(2)]],
                           constant uint &test [[buffer(3)]],
                           uint index [[thread_position_in_grid]])
{
  GuidingDirectionalTree<5> distribution;
  float pdf;
  const float3 normal = normalize(float3(1, 2, -3));
  const GuidingDirectionalProduct product = {
      normal,
      test == 6 ? -0.65f : 0.65f,
      test >= 6 ? GuidingDirectionalProduct::PHASE :
      test == 5 ? GuidingDirectionalProduct::TWO_SIDED_COSINE :
                  GuidingDirectionalProduct::COSINE};
  float3 direction = test >= 3 ? distribution.sample_product(tree, random[index], product, &pdf) :
                                 distribution.sample(tree, random[index], &pdf);
  result[2 * index] = float4(direction, pdf);
  result[2 * index + 1] = float4(test >= 3 ? distribution.pdf_product(tree, direction, product) :
                                             distribution.pdf(tree, direction));
}

GuidingField test_field(device GuidingSpatialNode *nodes,
                        device float *accumulation,
                        device float *sampling,
                        device uint *counts,
                        uint capacity)
{
  return {nodes, accumulation, sampling, counts, capacity, float3(-4, -2, -1), float3(4, 2, 1)};
}

kernel void guiding_field_record(device GuidingSpatialNode *nodes [[buffer(0)]],
                                 device float *accumulation [[buffer(1)]],
                                 device float *sampling [[buffer(2)]],
                                 device uint *counts [[buffer(3)]],
                                 constant uint &capacity [[buffer(4)]],
                                 uint index [[thread_position_in_grid]])
{
  GuidingField field = test_field(nodes, accumulation, sampling, counts, capacity);
  const float3 P = float3(
      (float(index % 997) / 997 - 0.5f) * 8, (float((index / 997) % 991) / 991 - 0.5f) * 4, 0.1f);
  const uint node = field.find_leaf(P);
  const float3 direction = normalize(float3(1, 0.7f, P.x < 0 ? 3 : -3));
  field.record_visit(node);
  field.record(field.record_index(node, GUIDING_FIELD_SURFACE_RADIANCE, direction),
               1.0f,
               direction,
               field.normalized_position(P));
  field.record(field.record_index(node, GUIDING_FIELD_SURFACE_IMPORTANCE, -direction),
               1.0f,
               -direction,
               field.normalized_position(P));
}

kernel void guiding_field_begin(device uint *counts [[buffer(0)]])
{
  counts[1] = counts[0];
}

kernel void guiding_field_refine(device GuidingSpatialNode *nodes [[buffer(0)]],
                                 device float *accumulation [[buffer(1)]],
                                 device float *sampling [[buffer(2)]],
                                 device uint *counts [[buffer(3)]],
                                 constant uint &capacity [[buffer(4)]],
                                 uint index [[thread_position_in_grid]])
{
  GuidingField field = test_field(nodes, accumulation, sampling, counts, capacity);
  field.refine(index, 64);
}

kernel void guiding_field_publish(device GuidingSpatialNode *nodes [[buffer(0)]],
                                  device float *accumulation [[buffer(1)]],
                                  device float *sampling [[buffer(2)]],
                                  device uint *counts [[buffer(3)]],
                                  constant uint &capacity [[buffer(4)]],
                                  uint index [[thread_position_in_grid]])
{
  GuidingField field = test_field(nodes, accumulation, sampling, counts, capacity);
  field.publish(index, 0.05f);
}

#include "kernel/sample/guiding_history.h"

/* Separate dispatches model shadow histories surviving main state reuse. Half-depth heads are
 * captured before the later main records are appended; neither branch may observe another path. */
kernel void guiding_history_append(device GuidingHistoryRecord *records [[buffer(0)]],
                                   device uint *count [[buffer(1)]],
                                   device uint2 *heads [[buffer(2)]],
                                   constant uint &capacity [[buffer(3)]],
                                   uint path [[thread_position_in_grid]])
{
  const GuidingHistory history{records, count, capacity};
  uint head = ~0u;
  for (uint depth = 0; depth < 8; ++depth) {
    const uint next = history.append(head, path, depth, packed_float3(1, 2, 3));
    if (next != ~0u) {
      head = next;
    }
    if (depth == 3) {
      heads[path].x = head;
      history.append_observation(path, 99, packed_float3(0), 3.0f, 2.0f);
    }
  }
  heads[path].y = head;
  history.append_observation(path, 100, packed_float3(0), 5.0f);
  history.append_observation(path, 101, packed_float3(0), 0.0f);
}

kernel void guiding_history_accumulate(device GuidingHistoryRecord *records [[buffer(0)]],
                                       device const uint2 *heads [[buffer(1)]],
                                       uint branch_path [[thread_position_in_grid]])
{
  const GuidingHistory history{records, nullptr, 0};
  uint head = heads[branch_path / 2][branch_path % 2];
  while (head != ~0u) {
    history.accumulate(head, 1.0f);
    history.accumulate_source(head, 1.0f, 2.0f + float(branch_path % 2));
    head = records[head].parent;
  }
}

kernel void guiding_history_read(device const GuidingHistoryRecord *records [[buffer(0)]],
                                 device const uint2 *heads [[buffer(1)]],
                                 device uint4 *results [[buffer(2)]],
                                 uint path [[thread_position_in_grid]])
{
  uint4 result(0);
  for (uint branch = 0; branch < 2; ++branch) {
    uint head = heads[path][branch];
    uint depth = 0;
    while (head != ~0u && depth < 9) {
      const device GuidingHistoryRecord &record = records[head];
      result.z += record.field_index != path;
      result.z += metal::any(float3(record.inverse_weight) != float3(1, 2, 3));
      result.z += record.parent != ~0u && record.parent >= head;
      result.z += record.radiance != (record.direction < 4 ? 2.0f : 1.0f);
      result.z += record.source_weight != (record.direction < 4 ? 2.0f : 1.0f);
      result.z += record.distance_weight != (record.direction < 4 ? 5.0f : 3.0f);
      result.z += fabsf(record.inverse_distance_weight -
                        (record.direction < 4 ? 5.0f / 6.0f : 1.0f / 3.0f)) > 1e-6f;
      result.w += record.direction;
      head = record.parent;
      ++depth;
    }
    result[branch] = depth;
  }
  results[path] = result;
}

#include "kernel/sample/guiding_parallax.h"
#include "kernel/sample/guiding_position.h"

kernel void guiding_parallax_record(device float *moments [[buffer(0)]],
                                    uint index [[thread_position_in_grid]])
{
  const uint test = index / 4096, x = index % 64, y = (index / 64) % 64;
  const float3 p = float3((x + .5f) / 64 - .5f, (y + .5f) / 64 - .5f, 0);
  const float3 target = test == 1 ? float3(1 + .7f * ((x + .5f) / 64 - .5f),
                                           1 + .5f * ((y + .5f) / 64 - .5f),
                                           2) :
                                    float3(1, 1, 2);
  const float3 delta = target - p;
  const float weight = test == 3 && x == 0 && y == 0 ? 1e8f : 1;
  GuidingParallaxMoments model;
  model.record(moments + test * model.storage_size,
               weight,
               p,
               normalize(delta),
               test == 2 ? FLT_MAX : len(delta));
}

kernel void guiding_parallax_fit(device const float *moments [[buffer(0)]],
                                 device const float4 *positions [[buffer(1)]],
                                 device float4 *results [[buffer(2)]],
                                 uint index [[thread_position_in_grid]])
{
  const uint test = index / 4;
  GuidingParallaxMoments model;
  const GuidingParallaxFit fit = model.fit(
      moments + test * model.storage_size, test == 3 ? 1e8f + 4095 : 4096, 4096, float3(0, 0, 1));
  const float3 p = positions[index % 4].xyz;
  results[2 * index] = float4(fit.direction(p), fit.concentration(p));
  GuidingVirtualDistance distance;
  const float scale = distance.scatter_scale(true, 0, 1.5f, .8f, .6f);
  results[2 * index + 1] = float4(fit.valid ? 1 : 0,
                                  distance.propagate(2, 5 * (index % 4 + 1), scale),
                                  fit.target.x,
                                  fit.target.z);
}

kernel void guiding_position_fit(device const float *moments [[buffer(0)]],
                                 device const float4 *positions [[buffer(1)]],
                                 device float4 *results [[buffer(2)]],
                                 uint index [[thread_position_in_grid]])
{
  GuidingPositionMoments model;
  const GuidingPositionFit fit = model.fit(moments + (index / 4) * model.storage_size, 4096);
  results[2 * index] = float4(fit.direction(positions[index % 4].xyz), fit.valid ? 1.0f : 0.0f);
  results[2 * index + 1] = float4(fit.residual_variance);
}

/* Exercise publication, not only evaluation of a host-built distribution. */
kernel void guiding_hdr_build(device const float *tree [[buffer(0)]],
                              device const float *moments [[buffer(1)]],
                              device const float *counts [[buffer(2)]],
                              device float *storage [[buffer(3)]],
                              device float4 *result [[buffer(4)]])
{
  GuidingGaussianMixture mixture;
  mixture.build(storage, tree, moments, counts);
  for (int i = 0; i < mixture.components; ++i) {
    const auto component = mixture.component(storage, i);
    result[i] = float4(component.axis, component.pdf(normalize(float3(1, 2, 3))));
  }
}

#include "kernel/sample/guiding_mixture_fit.h"

template<int Capacity, bool Cooperative>
inline void guiding_mixture_fit_impl(device const float4 *observations,
                                     device float4 *result,
                                     uint count,
                                     device float *published,
                                     uint index,
                                     uint lane,
                                     uint width)
{
  GuidingDirectionalMixtureFit<Capacity> model;
  const device float4 *data = observations + index * count;
  bool valid = model.template initialize<Cooperative>(data, count, lane, width);
  for (int iteration = 0; iteration < 64 && valid; ++iteration) {
    valid = model.template iterate<Cooperative>(data, count, lane, width);
  }
  if (lane != 0) {
    return;
  }
  device float *storage = published + index * GuidingGaussianMixture::storage_size;
  model.publish(storage);
  GuidingGaussianMixture query;
  for (int component = 0; component < Capacity; ++component) {
    const int offset = 2 * Capacity * index + 2 * component;
    const bool active = component < model.active;
    const float3 axis = active ? model.lobes[component].axis : float3(0, 0, 1);
    result[offset] = float4(axis, active ? model.lobes[component].concentration : 0);
    result[offset + 1] = float4(active ? model.weights[component] : 0,
                                float(valid),
                                float(model.active),
                                query.pdf(storage, axis));
  }
}

kernel void guiding_mixture_fit(device const float4 *observations [[buffer(0)]],
                                device float4 *result [[buffer(1)]],
                                constant uint &count [[buffer(2)]],
                                device float *published [[buffer(3)]],
                                uint index [[thread_position_in_grid]])
{
  guiding_mixture_fit_impl<2, false>(observations, result, count, published, index, 0, 1);
}

kernel void guiding_mixture_fit_cooperative(device const float4 *observations [[buffer(0)]],
                                            device float4 *result [[buffer(1)]],
                                            constant uint &count [[buffer(2)]],
                                            device float *published [[buffer(3)]],
                                            uint index [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]],
                                            uint width [[threads_per_simdgroup]])
{
  guiding_mixture_fit_impl<2, true>(observations, result, count, published, index, lane, width);
}

kernel void guiding_mixture_fit_16(device const float4 *observations [[buffer(0)]],
                                   device float4 *result [[buffer(1)]],
                                   constant uint &count [[buffer(2)]],
                                   device float *published [[buffer(3)]],
                                   uint index [[thread_position_in_grid]])
{
  guiding_mixture_fit_impl<16, false>(observations, result, count, published, index, 0, 1);
}

kernel void guiding_mixture_fit_cooperative_16(device const float4 *observations [[buffer(0)]],
                                               device float4 *result [[buffer(1)]],
                                               constant uint &count [[buffer(2)]],
                                               device float *published [[buffer(3)]],
                                               uint index [[threadgroup_position_in_grid]],
                                               uint lane [[thread_index_in_simdgroup]],
                                               uint width [[threads_per_simdgroup]])
{
  guiding_mixture_fit_impl<16, true>(observations, result, count, published, index, lane, width);
}

#include "kernel/sample/guiding_mixture_statistics.h"

template<int Capacity>
inline void guiding_mixture_source_statistics_impl(
    device const float4 *directions,
    device const GuidingMixtureObservation *observations,
    device float *output,
    uint count,
    device float4 *fits,
    uint lane,
    uint width)
{
  GuidingDirectionalMixtureFit<Capacity> model;
  bool valid = model.template initialize<true>(directions, count, lane, width);
  for (int iteration = 0; iteration < 64 && valid; ++iteration) {
    valid = model.template iterate<true>(directions, count, lane, width);
  }
  GuidingMixtureStatistics statistics;
  for (uint i = 0; i < count && valid; ++i) {
    const auto observation = observations[i];
    const float3 direction = normalize(float3(observation.direction_weight));
    const float log_density = lane < uint(model.active) ? model.log_component(lane, direction) :
                                                          -FLT_MAX;
    const float maximum = metal::simd_max(log_density);
    const float density = lane < uint(model.active) ? expf(log_density - maximum) : 0;
    const float total = metal::simd_sum(density);
    if (lane < uint(model.active)) {
      statistics.record(observation, density / total, model.maximum_weight, float3(1, .5f, .25f));
    }
  }
  if (lane < Capacity) {
    device float *storage = output + lane * statistics.storage_size;
    statistics.publish(storage);
    GuidingParallaxMoments source;
    const auto source_fit = source.fit(
        storage + statistics.position_size, storage[0], count, model.lobes[lane].axis);
    GuidingPositionMoments position;
    const auto position_fit = position.fit(storage, count);
    fits[2 * lane] = float4(source_fit.target, float(source_fit.valid));
    fits[2 * lane + 1] = float4(position_fit.direction(float3(.02f, -.01f, .01f)),
                                float(position_fit.valid));
  }
}

kernel void guiding_mixture_source_statistics(device const float4 *directions [[buffer(0)]],
                                              device const GuidingMixtureObservation *observations
                                              [[buffer(1)]],
                                              device float *output [[buffer(2)]],
                                              constant uint &count [[buffer(3)]],
                                              device float4 *fits [[buffer(4)]],
                                              uint lane [[thread_index_in_simdgroup]],
                                              uint width [[threads_per_simdgroup]])
{
  guiding_mixture_source_statistics_impl<2>(
      directions, observations, output, count, fits, lane, width);
}

kernel void guiding_mixture_source_statistics_16(
    device const float4 *directions [[buffer(0)]],
    device const GuidingMixtureObservation *observations [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    device float4 *fits [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]])
{
  guiding_mixture_source_statistics_impl<16>(
      directions, observations, output, count, fits, lane, width);
}

#include "kernel/sample/guiding_mixture_conditional.h"

kernel void guiding_mixture_conditional_update(device const GuidingMixtureObservation *observations
                                               [[buffer(0)]],
                                               device const float *input [[buffer(1)]],
                                               device float *output [[buffer(2)]],
                                               device float *scratch [[buffer(3)]],
                                               constant uint &count [[buffer(4)]],
                                               uint lane [[thread_index_in_simdgroup]],
                                               uint width [[threads_per_simdgroup]])
{
  GuidingMixtureStatistics statistics;
  float scale = 0;
  GuidingConditionalMixture conditional;
  if (lane < GuidingGaussianMixture::components) {
    const uint base = lane * GuidingGaussianMixture::component_stride;
    statistics.direction_reference = float3(input[base + 2], input[base + 3], input[base + 4]);
  }
  /* Deliberately split the immutable observations at non-SIMD-aligned boundaries.
   * The host reference collects the entire interval in one pass. */
  for (uint begin = 0; begin < count; begin += 257) {
    const uint end = min(begin + 257, count);
    float maximum_weight = 0;
    for (uint i = begin + lane; i < end; i += width) {
      if (GuidingDirectionalMixtureFit<1>::valid(observations[i].direction_weight)) {
        maximum_weight = max(maximum_weight, observations[i].direction_weight.w);
      }
    }
    maximum_weight = metal::simd_max(maximum_weight);
    GuidingMixtureStatistics batch;
    batch.direction_reference = statistics.direction_reference;
    for (uint i = begin; i < end; ++i) {
      const auto observation = observations[i];
      if (!GuidingDirectionalMixtureFit<1>::valid(observation.direction_weight)) {
        continue;
      }
      const float3 direction = normalize(float3(observation.direction_weight));
      const float log_density = lane < GuidingGaussianMixture::components ?
                                    conditional.log_component(
                                        input, lane, direction, float3(observation.position)) :
                                    -FLT_MAX;
      const float maximum = metal::simd_max(log_density);
      const float density = log_density > -FLT_MAX ? expf(log_density - maximum) : 0;
      const float total = metal::simd_sum(density);
      if (lane < GuidingGaussianMixture::components && total > 0) {
        batch.record(observation, density / total, maximum_weight, float3(1, .5f, .25f));
      }
    }
    if (maximum_weight > 0) {
      statistics.merge(batch, maximum_weight, scale);
    }
  }
  const float total_weight = metal::simd_sum(statistics.values[0]);
  const float squared_weight = metal::simd_sum(
      statistics.values[GuidingMixtureStatistics::global_squared_weight]);
  if (lane < GuidingGaussianMixture::components) {
    device float *storage = scratch + lane * statistics.storage_size;
    statistics.recenter(statistics.direction_reference + float3(.001f, -.002f, .003f));
    statistics.publish(storage);
    conditional.publish_component(output,
                                  lane,
                                  storage,
                                  total_weight,
                                  count,
                                  statistics.directional_fit(),
                                  float3(1, .5f, .25f),
                                  total_weight * total_weight / squared_weight);
  }
}

#include "kernel/sample/guiding_observation_range.h"

kernel void guiding_observation_partition(device const GuidingHistoryRecord *records [[buffer(0)]],
                                          device uint *metadata [[buffer(1)]],
                                          device uint *indices [[buffer(2)]],
                                          constant uint &count [[buffer(3)]],
                                          constant uint &capacity [[buffer(4)]],
                                          constant uint &phase [[buffer(5)]],
                                          uint index [[thread_position_in_grid]])
{
  const GuidingObservationPartition partition{
      records, metadata, metadata + 16, metadata + 32, indices, metadata + 48, 16, 100, capacity};
  if (phase == 0 && index < count)
    partition.count(index);
  else if (phase == 1 && index == 0)
    partition.prefix();
  else if (phase == 2 && index < count)
    partition.scatter(index);
}

kernel void guiding_observation_indexed_fit(device const GuidingHistoryRecord *records
                                            [[buffer(0)]],
                                            device const uint *metadata [[buffer(1)]],
                                            device const uint *indices [[buffer(2)]],
                                            device float4 *result [[buffer(3)]],
                                            uint field [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]],
                                            uint width [[threads_per_simdgroup]])
{
  const GuidingHistoryObservationRange range{records, indices, metadata[16 + field], 2};
  GuidingDirectionalMixtureFit<1> fit;
  const uint count = metadata[field];
  const bool initialized = fit.initialize<true>(range, count, lane, width);
  const bool fitted = initialized && fit.iterate<true>(range, count, lane, width);
  if (lane == 0) {
    result[2 * field] = float4(fit.lobes[0].axis, fitted ? fit.lobes[0].concentration : -1);
    result[2 * field + 1] = range.observation(0).source;
  }
}

/* Use the renderer's resource context and actual BSDF implementations. The test only
 * selects dielectric Fresnel, which does not require lookup-table resources. */
/* The headers open and close a resource-context class; their order is semantic. */
// clang-format off
#include "kernel/device/metal/globals.h"
#include "kernel/tables.h"
#include "kernel/device/metal/context_begin.h"
#include "kernel/closure/bsdf_microfacet_manifold.h"
#include "kernel/device/metal/context_end.h"
// clang-format on

kernel void manifold_bsdf_parameter_size(device uint *output [[buffer(0)]])
{
  output[0] = sizeof(KernelParamsMetal);
}

kernel void manifold_bsdf_consistency(device float4 *output [[buffer(0)]],
                                      constant KernelParamsMetal &params [[buffer(1)]],
                                      uint index [[thread_position_in_grid]])
{
  MetalKernelContext context(params);
  const uint test = index % 24;
  const ClosureType types[4] = {CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,
                                CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
                                CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID,
                                CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID};
  const ClosureType type = types[test / 6];
  const bool beckmann = test / 6 >= 2;
  const float cosines[3] = {.3f, .7f, 1.0f};
  const float cosine = cosines[test % 3];
  MetalKernelContext::MicrofacetBsdf bsdf{};
  bsdf.type = type;
  bsdf.N = float3(0, 0, 1);
  bsdf.weight = float3(.3f, .5f, .8f);
  bsdf.alpha_x = bsdf.alpha_y = .35f;
  bsdf.ior = 1.45f;
  bsdf.energy_scale = (test / 3) % 2 ? 1.7f : 1.0f;
  bsdf.fresnel_type = MetalKernelContext::MicrofacetFresnel::DIELECTRIC;
  thread ShaderClosure *closure = reinterpret_cast<thread ShaderClosure *>(&bsdf);
  const float3 wi = float3(sqrt(1 - cosine * cosine), 0, cosine);
  const float3 h = normalize(float3(.08f, -.03f, 1));
  const float3 wo = refract(-wi, h, 1 / bsdf.ior);
  float pdf = 0;
  const Spectrum camera = beckmann ?
                              context.bsdf_microfacet_beckmann_eval(
                                  nullptr, closure, wi, wo, &pdf) :
                              context.bsdf_microfacet_ggx_eval(nullptr, closure, wi, wo, &pdf);
  const float D = beckmann ?
                      context.bsdf_D<MetalKernelContext::MicrofacetType::BECKMANN>(sqr(.35f),
                                                                                   h.z) :
                      context.bsdf_D<MetalKernelContext::MicrofacetType::GGX>(sqr(.35f), h.z);
  const float jacobian = sqr(dot(wi, h) + bsdf.ior * dot(wo, h)) /
                         (sqr(bsdf.ior) * fabs(dot(wo, h)));
  const Spectrum expected = bsdf.weight * camera * jacobian / (D * h.z * h.z);
  const Spectrum actual = context.mnee_eval_bsdf_contribution(nullptr, closure, wi, wo);
  const float2 sampled = context.mnee_sample_bsdf_dh(
      type, .3f, .3f, (float(index / 24) + .5f) / 10000, .317f);
  output[index] = float4(actual / expected, dot(sampled, sampled));
}
