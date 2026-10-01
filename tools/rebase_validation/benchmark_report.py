#!/usr/bin/env python3
"""Report the median render times measured by benchmark.sh.

  benchmark_report.py <output directory> [--scenes]

Feature suite times are the second render of a scene in the same process (no kernel loading or
scene export); parity suite times are complete renders. Prints the sum of the per-scene medians
for every suite and device, and with --scenes every scene.
"""
import glob
import json
import os
import statistics
import sys

out = sys.argv[1]
per_scene = "--scenes" in sys.argv


def medians(label, suite):
    times = {}
    for path in sorted(glob.glob(os.path.join(out, f"{label}_round[1-9]*", suite, "report.json"))):
        for scene, entry in json.load(open(path)).items():
            times.setdefault(scene, []).append(entry["render_seconds"])
    return {scene: statistics.median(v) for scene, v in times.items()}, times


for suite in ("feature_METAL", "parity_METAL", "feature_CPU", "parity_CPU"):
    before, before_all = medians("before", suite)
    after, after_all = medians("after", suite)
    scenes = [s for s in before if s in after]
    if not scenes:
        continue
    total_before = sum(before[s] for s in scenes)
    total_after = sum(after[s] for s in scenes)
    rounds = min(len(v) for v in before_all.values())
    print(f"{suite}: {len(scenes)} scenes, {rounds} rounds, sum of medians "
          f"{total_before:.2f}s -> {total_after:.2f}s ({(total_after / total_before - 1) * 100:+.1f}%)")
    if per_scene:
        for s in scenes:
            spread = (max(before_all[s]) - min(before_all[s])) / before[s] * 100
            print(f"   {s:30s} {before[s]:8.3f}s -> {after[s]:8.3f}s "
                  f"({(after[s] / before[s] - 1) * 100:+6.1f}%, before spread {spread:4.1f}%)")
