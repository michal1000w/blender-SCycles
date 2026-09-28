# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Discrete path-strategy audit, not a geometric diffraction correctness test.

All geometric factors and endpoint densities are one. Each internal vertex
has specified forward/reverse probabilities for the selected event. Connections
adjacent to a delta vertex are inadmissible. Enumerated complete-path products
provide an independent reference for every admissible strategy's MIS weight.
This isolates unequal branch masses; it does not model grating constraint
Jacobians, shading normals, wavelength selection or polarization.
"""
import json
import math
import random


def partition_audit():
    rng = random.Random(1729)
    worst_error = 0.0
    cases = 0
    for exponent in (1, 2):
        for edges in range(3, 11):
            for trial in range(64):
                f = [1.0] + [rng.uniform(.05, .95) for _ in range(edges - 1)]
                r = [1.0] + [rng.uniform(.05, .95) for _ in range(edges - 1)]
                delta = [False] + [rng.random() < .4 for _ in range(edges - 1)] + [False]
                densities = [math.prod(r[1:])]
                for split in range(1, edges + 1):
                    densities.append(0.0 if delta[split - 1] or delta[split] else
                                     math.prod(f[:split - 1]) * math.prod(r[split + 1:]))
                total = sum(p ** exponent for p in densities)
                light, camera = {}, {}
                cm, vc = 1.0, 1.0
                for i in range(1, edges):
                    light[i] = cm, vc
                    if delta[i]:
                        cm, vc = 0.0, vc * (r[i] / f[i]) ** exponent
                    else:
                        cm, vc = (1 / f[i]) ** exponent, (cm + vc * r[i] ** exponent) / f[i] ** exponent
                cm, vc = 1.0, 0.0
                for i in range(edges - 1, -1, -1):
                    camera[i] = cm, vc
                    if i:
                        if delta[i]:
                            cm, vc = 0.0, vc * (f[i] / r[i]) ** exponent
                        else:
                            cm, vc = (1 / r[i]) ** exponent, (cm + vc * f[i] ** exponent) / r[i] ** exponent
                weights = []
                for split, density in enumerate(densities):
                    if density == 0:
                        weights.append(0.0)
                        continue
                    if split == 0:
                        alternatives = sum(camera[0])
                    elif split == 1:
                        cm, vc = camera[1]
                        alternatives = r[1] ** exponent + cm + vc * f[1] ** exponent
                    elif split == edges:
                        cm, vc = light[edges - 1]
                        alternatives = cm + vc * r[edges - 1] ** exponent
                    else:
                        lm, lv = light[split - 1]
                        cm, cv = camera[split]
                        alternatives = (r[split] ** exponent * (lm + lv * r[split - 1] ** exponent) +
                                        f[split - 1] ** exponent * (cm + cv * f[split] ** exponent))
                    weight = 1 / (1 + alternatives)
                    reference = density ** exponent / total
                    worst_error = max(worst_error, abs(weight - reference))
                    assert math.isclose(weight, reference, rel_tol=1e-11, abs_tol=1e-14)
                    weights.append(weight)
                assert math.isclose(sum(weights), 1.0, abs_tol=1e-12)
                cases += 1
    return dict(cases=cases, maximum_absolute_weight_error=worst_error,
                includes_multiple_adjacent_delta_events=True,
                production_kernel_exercised=False)


def audit():
    records = []
    for exponent in (1, 2):
        for forward, reverse in ((0.2, 0.7), (0.7, 0.2), (0.4, 0.4)):
            # Light -- diffuse -- delta -- diffuse -- camera.
            f = [1.0, 0.3, forward, 0.6]
            r = [1.0, 0.5, reverse, 0.4]
            delta = [False, False, True, False, False]
            densities = [math.prod(r[1:])]
            for s in range(1, 5):
                densities.append(0.0 if delta[s - 1] or delta[s] else
                                 math.prod(f[:s - 1]) * math.prod(r[s + 1:]))
            reference = densities[0] ** exponent / sum(p ** exponent for p in densities)
            weights = []
            for retain_branch_ratio in (False, True):
                cm, vc = 1.0, 0.0
                for i in (3, 2, 1):
                    if delta[i]:
                        cm = 0.0
                        if retain_branch_ratio:
                            vc *= (f[i] / r[i]) ** exponent
                    else:
                        cm, vc = (1 / r[i]) ** exponent, (vc * f[i] ** exponent + cm) / r[i] ** exponent
                weights.append(1 / (1 + cm + vc))
            assert math.isclose(weights[1], reference, rel_tol=1e-12)
            records.append(dict(exponent=exponent, forward_mass=forward, reverse_mass=reverse,
                                complete_path_weight=reference, unit_delta_recurrence=weights[0],
                                branch_ratio_recurrence=weights[1]))
    assert any(abs(x['unit_delta_recurrence'] - x['complete_path_weight']) > 0.01 for x in records)
    return dict(scope=__doc__, records=records, full_partition=partition_audit())


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
