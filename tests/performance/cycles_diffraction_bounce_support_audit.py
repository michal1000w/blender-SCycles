# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive discrete audit of per-type termination, excluding null events.

Sequences list scattering events from camera to emitter. Cycles stores each
user bounce limit plus one, and terminates continuation immediately after an
event reaches that stored limit. Emission can still be evaluated afterwards.
This audit addresses support only, not PDFs, geometry, or rendering correctness.
"""
import itertools
import json


def sequential_support(events, limits):
    counts = [0] * len(limits)
    for index, event in enumerate(events):
        counts[event] += 1
        if counts[event] >= limits[event]:
            return index == len(events) - 1
    return True


def total_support(events, limits):
    return all(events.count(i) <= limit for i, limit in enumerate(limits))


def terminal_aware_support(events, limits):
    return all(events.count(i) <= limit - int(not events or events[-1] != i)
               for i, limit in enumerate(limits))


def audit():
    checked = 0
    count_only_false_positives = 0
    reverse_support_mismatches = 0
    for limits in itertools.product(range(1, 4), repeat=3):
        for length in range(8):
            for events in itertools.product(range(3), repeat=length):
                expected = sequential_support(events, limits)
                assert terminal_aware_support(events, limits) == expected
                # Every generated light prefix is a suffix of the complete
                # camera-ordered sequence. Test those suffixes in camera order,
                # rather than applying camera termination along the light ray.
                light_events = events[::-1]
                prefix_support = all(terminal_aware_support(light_events[:n][::-1], limits)
                                     for n in range(1, len(light_events) + 1))
                assert prefix_support == expected
                count_only_false_positives += total_support(events, limits) and not expected
                reverse_support_mismatches += expected != sequential_support(events[::-1], limits)
                checked += 1
    # D then G cannot continue when the first D reaches the stored D limit.
    # Reversing the order allows G then D and immediate emitter evaluation.
    assert not sequential_support((0, 1), (1, 13, 1))
    assert sequential_support((1, 0), (1, 13, 1))
    return dict(sequences_checked=checked,
                terminal_aware_matches_sequential=True,
                reversed_light_prefix_support_matches_camera_support=True,
                count_only_false_positives=count_only_false_positives,
                reverse_support_mismatches=reverse_support_mismatches,
                scope='Discrete scattering-order support only. Production integration and '
                      'MIS strategy support remain unimplemented.')


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
