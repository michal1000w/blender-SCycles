#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Analytic checks for a proposed bidirectional weighted-null density contract.

This is a mathematical model, not a renderer-support test. It checks both
shared event roulette and deterministic-null connection strategies, and the
conditions needed for their density cancellations. Actual Cycles integration,
majorant traversal, emission, guiding, retries, and spectral shaders are NOT
exercised here.
"""
import math
import unittest


def event_parameters(sigma_t, sigma_s, majorant):
    assert majorant > 0 and all(0 <= s <= t for s, t in zip(sigma_s, sigma_t))
    if max(sigma_t) > majorant:
        # Signed weighted tracking keeps the chosen rate and physical medium.
        # The event proposal uses positive coefficient magnitudes, independent
        # of incoming throughput. It does not promote or clamp the majorant.
        real_score = sum(sigma_s)
        null_score = sum(abs(majorant-t) for t in sigma_t)
        q_real = real_score / (real_score + null_score)
    else:
        q_real = sum(sigma_s) / len(sigma_s) / majorant
    rho = q_real * majorant
    q_null = 1 - q_real
    null_weight = [(1-t/majorant)/q_null if q_null else 0 for t in sigma_t]
    real_weight = [s/rho if rho else 0 for s in sigma_s]
    return rho, q_null, null_weight, real_weight


def enumerate_surviving_null_paths(sigma_t, sigma_s, majorant, length):
    rho, q_null, weights, _ = event_parameters(sigma_t, sigma_s, majorant)
    # Explicit Poisson counts and event choices, summed independently of the
    # exponential physical-transmittance reference. Inputs keep the tail tiny.
    poisson = math.exp(-majorant * length)
    total = [0.0] * len(sigma_t)
    for count in range(256):
        for channel, weight in enumerate(weights):
            total[channel] += poisson * q_null**count * weight**count
        poisson *= majorant * length / (count + 1)
    return total


class BidirectionalVolumeDensityModel(unittest.TestCase):
    def test_connection_expectation_matches_spectral_transmittance(self):
        for sigma_t, sigma_s in [([.2, .7, 1.2], [.1, .6, .3]),
                                 ([0, .4, 1.0], [0, .2, .8]),
                                 ([.3, .5, .9], [0, 0, 0]),
                                 ([1, 1, 1], [1, 1, 1])]:
            for scale in [.5, .75, 1, 2, 5]:
                majorant = max(sigma_t) * scale
                for length in [.1, 1, 3]:
                    actual = enumerate_surviving_null_paths(sigma_t, sigma_s, majorant, length)
                    for value, coefficient in zip(actual, sigma_t):
                        self.assertAlmostEqual(value, math.exp(-coefficient * length), delta=2e-13)

    def test_collision_expectation_includes_endpoint_density(self):
        sigma_t, sigma_s = [.3, .8, 1.4], [.2, .6, .5]
        for majorant in [.6, 1.0, 1.4, 3, 7]:
            rho, _, _, real_weight = event_parameters(sigma_t, sigma_s, majorant)
            for distance in [.1, .7, 2]:
                prefix = enumerate_surviving_null_paths(sigma_t, sigma_s, majorant, distance)
                for channel in range(3):
                    weighted_density = prefix[channel] * rho * real_weight[channel]
                    target = sigma_s[channel] * math.exp(-sigma_t[channel] * distance)
                    self.assertAlmostEqual(weighted_density, target, delta=2e-13)

    def test_piecewise_majorants_and_reversal_share_null_path_density(self):
        # Each tuple gives segment length, majorant, and null acceptance factors
        # at the same realized world-space candidate points, in path order.
        segments = [(.4, 2.0, [.7, .8]), (.6, 3.0, [.9]), (.3, 1.5, [.6, .95])]
        def null_density(sequence):
            density = 1.0
            for length, majorant, choices in sequence:
                density *= math.exp(-majorant * length)
                for q_null in choices:
                    density *= majorant * q_null
            return density
        forward = null_density(segments)
        reverse = null_density([(length, majorant, list(reversed(choices)))
                                for length, majorant, choices in reversed(segments)])
        self.assertAlmostEqual(forward, reverse, delta=1e-14)
        start_rho, end_rho = .25, .7
        # Endpoint factors differ; the shared null realization cancels. A
        # deterministic-null ratio-tracking shadow would not have this density.
        self.assertAlmostEqual((forward*end_rho)/(reverse*start_rho), end_rho/start_rho)
        deterministic = math.prod(math.exp(-length*majorant) * majorant**len(choices)
                                  for length, majorant, choices in segments)
        self.assertNotAlmostEqual(forward, deterministic)

    def test_common_real_event_density_cancels_for_internal_medium_vertices(self):
        # Emitter -> medium0 -> medium1 -> sensor; squared distances and emitter
        # cosine are one. All strategies sample both real medium vertices, so
        # the same direction-independent rho0*rho1 factors out of the partition.
        rho0, rho1 = .2, .8
        null_base = .31 * .17 * .42
        emission, camera = .4, .7
        l0, l1, c1, c2, sensor_count = .3, .6, .8, .2, 13
        complete = [camera*rho1*c2*rho0*c1,
                    emission*camera*rho1*c2*rho0,
                    emission*l0*rho0*camera*rho1,
                    sensor_count*emission*l0*rho0*l1*rho1]
        factored = [camera*c2*c1, emission*camera*c2,
                    emission*l0*camera, sensor_count*emission*l0*l1]
        for exponent in [1, 2]:
            a = [(p*null_base)**exponent for p in complete]
            b = [p**exponent for p in factored]
            for x, y in zip(a, b):
                self.assertAlmostEqual(x/sum(a), y/sum(b))

    def test_ratio_tracking_connection_requires_its_own_null_probability(self):
        # Three candidate connection edges. A random-walk segment samples null
        # choices q_n; ratio tracking makes those choices deterministically.
        # Build each full probability explicitly before comparing cancellation.
        q = [.3, .7, .2]
        ordinary = [.11, .23, .37, .29]  # emitter-hit, then each connection edge
        complete = [ordinary[0] * math.prod(q)]
        for selected in range(3):
            product = ordinary[selected+1]
            for edge in range(3):
                product *= 1 if edge == selected else q[edge]
            complete.append(product)
        corrected = [ordinary[0]] + [ordinary[i+1]/q[i] for i in range(3)]
        for exponent in [1, 2]:
            a = [p**exponent for p in complete]
            b = [p**exponent for p in corrected]
            wrong = [p**exponent for p in ordinary]
            for x, y in zip(a, b):
                self.assertAlmostEqual(x/sum(a), y/sum(b))
            self.assertGreater(max(abs(x/sum(a)-y/sum(wrong)) for x, y in zip(a, wrong)), .1)

    def test_direction_dependent_majorants_do_not_cancel(self):
        # Same realized null points, but bounds estimated independently along
        # the two ray orientations. Even constant real-event intensity does
        # not cancel the Poisson factors when the proposal majorants differ.
        length, rho = .7, .4
        forward_majorant, reverse_majorant = 2.0, 3.0
        count = 2

        def full_density(majorant):
            return (math.exp(-majorant * length) * majorant**count *
                    (1-rho/majorant)**count)

        actual = full_density(forward_majorant) / full_density(reverse_majorant)
        null_choices_only = ((1-rho/forward_majorant) /
                             (1-rho/reverse_majorant))**count
        poisson_ratio = (math.exp((reverse_majorant-forward_majorant)*length) *
                         (forward_majorant/reverse_majorant)**count)
        self.assertAlmostEqual(actual, poisson_ratio * null_choices_only, delta=1e-14)
        self.assertGreater(abs(actual-null_choices_only), .05)
        # The resulting MIS error persists for both supported heuristics.
        for exponent in [1, 2]:
            correct_weight = 1/(1+actual**exponent)
            incomplete_weight = 1/(1+null_choices_only**exponent)
            self.assertGreater(abs(correct_weight-incomplete_weight), .02)

    def test_four_sample_extrema_are_not_a_conservative_bound(self):
        # Mirrors the four equally spaced samples plus 1.5 safety multiplier
        # used for dynamic shader extrema. A narrow procedural feature can lie
        # between every probe. Increasing a sampled maximum is not a bound.
        def extinction(position):
            return 10.0 if .24 < position < .26 else .1

        probes = [(i+.5)/4 for i in range(4)]
        estimated_bound = max(.5, 1.5 * max(map(extinction, probes)))
        self.assertGreater(extinction(.25), estimated_bound)
        true_optical_depth = .98*.1 + .02*10
        clipped_optical_depth = .98*.1 + .02*estimated_bound
        self.assertGreater(math.exp(-clipped_optical_depth) -
                           math.exp(-true_optical_depth), .1)


if __name__ == '__main__':
    unittest.main()
