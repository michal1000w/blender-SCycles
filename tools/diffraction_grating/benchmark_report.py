# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Summarize tools/diffraction_grating/benchmark.sh: median second-render time per scene, device
and build over the rounds after round 0, and the ratio B / A."""

import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
times = {}
for timing in out.glob("*_round*/*/timings.json"):
    label, round_ = timing.parent.parent.name.split("_round")
    if round_ == "0":
        continue
    device = timing.parent.name
    for scene, entry in json.loads(timing.read_text()).items():
        times.setdefault((scene, device), {}).setdefault(label, []).append(entry["render_seconds"])
print(f"{'scene':16s} {'device':6s} {'A (s)':>8s} {'B (s)':>8s} {'B/A':>6s}  rounds")
ratios = {}
for (scene, device), by_label in sorted(times.items()):
    a = statistics.median(by_label.get("A", [float("nan")]))
    b = statistics.median(by_label.get("B", [float("nan")]))
    ratios.setdefault(device, []).append(b / a)
    print(f"{scene:16s} {device:6s} {a:8.3f} {b:8.3f} {b / a:6.3f}  {len(by_label.get('A', []))}")
for device, values in sorted(ratios.items()):
    print(f"geometric mean B/A {device}: {statistics.geometric_mean(values):.3f}")
