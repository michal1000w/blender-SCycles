#!/usr/bin/env python3
"""Summarize tools/nested_dielectrics/benchmark.sh: median render seconds per scene.

  tools/nested_dielectrics/benchmark_report.py <output directory>
"""
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
times = {}
for report in sorted(out.glob("*_round*/*/report.json")):
    label, round_name = report.parts[-3].split("_round")
    if round_name == "0":
        continue
    config = report.parts[-2]
    for scene, data in json.loads(report.read_text()).items():
        times.setdefault((config, scene), {}).setdefault(label, []).append(data["render_seconds"])

def median(values):
    return statistics.median(values) if values else None

print("%-12s %-26s %9s %9s %8s" % ("config", "scene", "before s", "after s", "change"))
for (config, scene), data in sorted(times.items()):
    before = median(data.get("before", []))
    after = median(data.get("after", []))
    change = "%+.1f%%" % (100.0 * (after / before - 1.0)) if before and after else ""
    print("%-12s %-26s %9s %9s %8s" % (config, scene,
                                      "%.2f" % before if before else "-",
                                      "%.2f" % after if after else "-", change))

print()
print("Cost of using the feature (after build): nested scene against its counterpart")
pairs = [("drink_nested", "drink_overlap")]
pairs += [(name.replace("_reference", "_nested"), name)
          for (_, name) in times if name.endswith("_reference")]
for config in sorted({config for config, _ in times}):
    for nested, other in sorted(set(pairs)):
        a = median(times.get((config, nested), {}).get("after", []))
        b = median(times.get((config, other), {}).get("after", []))
        if a and b:
            print("%-12s %-22s %7.2f s vs %-24s %7.2f s  %+.1f%%" % (
                config, nested, a, other, b, 100.0 * (a / b - 1.0)))
