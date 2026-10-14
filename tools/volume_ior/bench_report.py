#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Report of the render times written by bench_matrix.py.

  tools/volume_ior/bench_report.py <directory> [<directory of the build before>]

For every scene and configuration: the time with the refraction, the time without, and the
overhead of the refraction. With a second directory also the times of that build and how many
times faster the first one renders the refraction.
"""

import json
import sys
from pathlib import Path


def load(directory):
    times = {}
    for line in open(Path(directory) / "times.jsonl"):
        entry = json.loads(line)
        key = (entry["scene"], entry["device"], entry["config"])
        # Later runs replace earlier ones.
        times.setdefault(key, {})[entry["mode"]] = entry["seconds"]
    return times


def main():
    after = load(sys.argv[1])
    before = load(sys.argv[2]) if len(sys.argv) > 2 else None

    header = "%-14s %-6s %-13s %9s %9s %9s" % ("scene", "device", "config", "on", "off", "overhead")
    if before:
        header += " | %9s %9s %9s %8s" % ("before on", "off", "overhead", "speedup")
    print(header)

    worst = 0.0
    for key in sorted(after):
        on, off = after[key].get("on"), after[key].get("off")
        if on is None or off is None:
            continue
        overhead = on / off - 1.0
        worst = max(worst, overhead)
        line = "%-14s %-6s %-13s %8.2fs %8.2fs %+8.0f%%" % (*key, on, off, 100.0 * overhead)
        if before and key in before and "on" in before[key] and "off" in before[key]:
            b_on, b_off = before[key]["on"], before[key]["off"]
            line += " | %8.2fs %8.2fs %+8.0f%% %7.1fx" % (
                b_on, b_off, 100.0 * (b_on / b_off - 1.0), b_on / on)
        print(line)
    print("largest overhead: %+.0f%%" % (100.0 * worst))


if __name__ == "__main__":
    main()
