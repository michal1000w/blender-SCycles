/* SPDX-FileCopyrightText: 2026 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/light/coherent_history.h"

#include <cstdio>

using namespace ccl;

int main()
{
  int checks = 0, failures = 0;
  auto expect = [&](const bool condition) {
    checks++;
    failures += !condition;
  };
  KernelCoherentCandidate direct{};
  direct.count = 0;
  KernelCoherentCandidate chain{};
  chain.count = 4;
  chain.patch[0] = 0; chain.event[0] = 0;
  chain.patch[1] = 63; chain.event[1] = 1;
  chain.patch[2] = 7; chain.event[2] = 0;
  chain.patch[3] = 31; chain.event[3] = 1;
  auto history = coherent_history_begin();
  expect(coherent_history_valid(history));
  expect(coherent_history_count(history) == 0);
  expect(coherent_history_matches(history, &direct));
  expect(!coherent_history_matches(history, &chain));
  for (int i = 0; i < 4; i++) {
    history = coherent_history_append(history, uint(chain.patch[i]), uint(chain.event[i]));
    expect(coherent_history_valid(history));
    expect(coherent_history_count(history) == uint(i + 1));
  }
  expect(coherent_history_matches(history, &chain));
  expect(coherent_history_matches_source(history, &chain, 17, 17));
  expect(!coherent_history_matches_source(history, &chain, 17, 18));
  KernelCoherentCandidate changed = chain;
  changed.event[3] = 0;
  expect(!coherent_history_matches(history, &changed));
  changed = chain;
  changed.patch[1] = 62;
  expect(!coherent_history_matches(history, &changed));
  changed = chain;
  changed.expected_incident_side[2] = 1;
  expect(!coherent_history_matches(history, &changed));
  auto sided = coherent_history_append(coherent_history_begin(), 7, 1, -1);
  KernelCoherentCandidate side_candidate{};
  side_candidate.count = 1;
  side_candidate.patch[0] = 7;
  side_candidate.event[0] = 1;
  side_candidate.expected_incident_side[0] = -1;
  expect(coherent_history_matches(sided, &side_candidate));
  side_candidate.expected_incident_side[0] = 1;
  expect(!coherent_history_matches(sided, &side_candidate));
  expect(!coherent_history_valid(coherent_history_append(history, 1, 0)));
  expect(!coherent_history_valid(coherent_history_append(coherent_history_begin(), 64, 0)));
  expect(!coherent_history_valid(coherent_history_append(coherent_history_begin(), 0, 4)));
  expect(!coherent_history_valid(coherent_history_append(coherent_history_begin(), 0, 0, 2)));
  expect(!coherent_history_matches(coherent_history_invalidate(history), &chain));
  const int mirror_delta = LABEL_REFLECT | LABEL_SINGULAR;
  const int glass_delta = LABEL_TRANSMIT | LABEL_SINGULAR;
  auto classified = coherent_history_after_scatter(coherent_history_begin(), 3, 1,
                                                    mirror_delta, 1);
  expect(coherent_history_valid(classified) && coherent_history_count(classified) == 1);
  expect(!coherent_history_valid(coherent_history_after_scatter(
      coherent_history_begin(), -1, 0, mirror_delta, 0)));
  expect(!coherent_history_valid(coherent_history_after_scatter(
      coherent_history_begin(), 3, 1, LABEL_REFLECT | LABEL_GLOSSY, 1)));
  expect(!coherent_history_valid(coherent_history_after_scatter(
      coherent_history_begin(), 3, 1, glass_delta, 1)));
  expect(!coherent_history_valid(coherent_history_after_scatter(
      classified, -1, 0, LABEL_REFLECT | LABEL_DIFFUSE, 0)));
  const auto glass_history = coherent_history_after_scatter(coherent_history_begin(), 2, 2,
                                                             glass_delta, -1);
  KernelCoherentCandidate glass_candidate{};
  glass_candidate.count = 1;
  glass_candidate.patch[0] = 2;
  glass_candidate.event[0] = 1;
  glass_candidate.expected_incident_side[0] = -1;
  expect(coherent_history_matches(glass_history, &glass_candidate));

  /* A stored light-side detector prefix belongs to the deterministic estimator
   * regardless of whether the camera-side connection vertex is also marked.
   * Unmarked light-side vertices store an invalid history and stay in BDPT. */
  for (const bool camera_vertex_marked : {false, true}) {
    (void)camera_vertex_marked;
    const auto marked_light_direct = coherent_history_begin();
    const auto unmarked_light_direct = coherent_history_invalidate(marked_light_direct);
    expect(coherent_history_matches_source(marked_light_direct, &direct, 17, 17));
    expect(!coherent_history_matches_source(unmarked_light_direct, &direct, 17, 17));
    expect(coherent_history_matches_source(history, &chain, 17, 17));
    expect(!coherent_history_matches_source(history, &chain, 18, 17));
  }
  direct.light = 4;
  expect(coherent_history_owns_direct_source(&direct, 4, true));
  expect(!coherent_history_owns_direct_source(&direct, 4, false));
  expect(!coherent_history_owns_direct_source(&direct, 5, true));
  chain.light = 4;
  expect(!coherent_history_owns_direct_source(&chain, 4, true));
  /* The first owned detector hit ends the light subpath, even though a
   * subsequent unmarked diffuse vertex could otherwise be cached. */
  expect(coherent_history_owns_detector_prefix(true, history, &chain, 17, 17));
  expect(!coherent_history_owns_detector_prefix(false, history, &chain, 17, 17));
  expect(!coherent_history_owns_detector_prefix(true, history, &chain, 18, 17));
  expect(!coherent_history_owns_detector_prefix(
      true, coherent_history_invalidate(history), &chain, 17, 17));
  KernelCoherentCandidate one_mirror{};
  one_mirror.count = 1;
  one_mirror.event[0] = 0;
  auto within_budget = [&](const KernelCoherentCandidate &candidate,
                           const int eye_bounce,
                           const int eye_glossy,
                           const int eye_transmission,
                           const int max_bounce,
                           const int max_glossy,
                           const int max_transmission,
                           const int bdpt_max,
                           const int light_max,
                           const bool bdpt_enabled = true) {
    return coherent_history_candidate_within_budget(&candidate,
                                                    eye_bounce,
                                                    eye_glossy,
                                                    eye_transmission,
                                                    max_bounce,
                                                    max_glossy,
                                                    max_transmission,
                                                    bdpt_enabled,
                                                    bdpt_max,
                                                    light_max);
  };
  expect(within_budget(direct, 7, 2, 1, 8, 3, 2, 2, 7));
  expect(!within_budget(direct, 8, 2, 1, 8, 3, 2, 2, 7));
  expect(within_budget(one_mirror, 0, 0, 0, 2, 2, 2, 2, 1));
  expect(!within_budget(one_mirror, 0, 0, 0, 1, 2, 2, 2, 1));
  expect(!within_budget(one_mirror, 1, 0, 0, 2, 2, 2, 2, 1));
  expect(!within_budget(one_mirror, 0, 1, 0, 2, 2, 2, 2, 1));
  expect(!within_budget(one_mirror, 0, 0, 0, 2, 2, 2, 1, 1));
  expect(within_budget(one_mirror, 0, 0, 0, 2, 2, 2, 1, 1, false));
  expect(!within_budget(one_mirror, 0, 0, 0, 2, 2, 2, 2, 0));
  KernelCoherentCandidate two_glass{};
  two_glass.count = 2;
  two_glass.event[0] = 1;
  two_glass.event[1] = 1;
  /* Scene upload stores each UI transmission limit plus one. A two-face slab
   * is allowed at UI=2 (kernel=3), rejected at UI=1 (kernel=2). */
  expect(within_budget(two_glass, 0, 0, 0, 3, 1, 3, 3, 2));
  expect(within_budget(two_glass, 0, 0, 0, 3, 1, 2 + 1, 3, 2));
  expect(!within_budget(two_glass, 0, 0, 0, 3, 1, 1 + 1, 3, 2));
  expect(!within_budget(two_glass, 0, 0, 1, 3, 1, 3, 3, 2));
  expect(!within_budget(two_glass, 0, 0, 0, 3, 1, 2, 3, 2));
  /* Repeated native Glass sphere primitive: exterior entry, internal
   * reflections, then interior-side exit. Each bounded order owns exactly
   * that history; opposite-side or missing internal events stay native. */
  for (int internal = 1; internal <= 2; internal++) {
    KernelCoherentCandidate sphere{}; sphere.count = internal + 2;
    auto sphere_history = coherent_history_begin();
    for (int event = 0; event < sphere.count; event++) {
      const bool transmission = event == 0 || event == sphere.count - 1;
      sphere.patch[event] = 9;
      sphere.event[event] = transmission ? 1 : 0;
      sphere.expected_incident_side[event] = event == 0 ? 1 : -1;
      sphere_history = coherent_history_after_scatter(sphere_history, 9, 2,
          LABEL_SINGULAR | (transmission ? LABEL_TRANSMIT : LABEL_REFLECT),
          sphere.expected_incident_side[event]);
    }
    expect(coherent_history_matches(sphere_history, &sphere));
    KernelCoherentCandidate wrong_side = sphere;
    wrong_side.expected_incident_side[1] = 1;
    expect(!coherent_history_matches(sphere_history, &wrong_side));
    KernelCoherentCandidate wrong_event = sphere;
    wrong_event.event[1] = 1;
    expect(!coherent_history_matches(sphere_history, &wrong_event));
  }
  KernelCoherentCandidate matched{};
  matched.count=1;matched.patch[0]=2;matched.event[0]=1;matched.expected_incident_side[0]=0;
  expect(coherent_history_matches(coherent_history_append(coherent_history_begin(),2,1,1),&matched));
  expect(coherent_history_matches(coherent_history_append(coherent_history_begin(),2,1,-1),&matched));
  matched.expected_incident_side[0]=1;
  expect(!coherent_history_matches(coherent_history_append(coherent_history_begin(),2,1,-1),&matched));
  std::printf("coherent_history checks=%d failures=%d bytes=%zu\n",
              checks, failures, sizeof(CoherentPathHistory));
  return failures != 0;
}
