#!/usr/bin/env python3
"""Summarize the comparison files written by compare.sh.

  summarize.py <directory> [result names...]

Prints for every suite the number of bit-identical images and the differing ones, sorted by
relative RMSE, with the change of the mean and the maximum absolute difference.
"""
import json
import os
import sys

directory = sys.argv[1]
names = sys.argv[2:] or ["compare_" + s for s in
                         ("feature_METAL", "feature_CPU", "parity_METAL", "parity_CPU",
                          "nodes_METAL", "nodes_CPU")]
for name in names:
    path = os.path.join(directory, name + ".jsonl")
    if not os.path.exists(path):
        continue
    rows = [json.loads(line) for line in open(path)]
    identical = [r for r in rows if r["max_abs_diff"] == 0.0]
    print(f"== {name}: {len(rows)} images, {len(identical)} bit-identical")
    for r in sorted(rows, key=lambda r: -r["relative_rmse"]):
        if r["max_abs_diff"] == 0.0:
            continue
        scene = os.path.basename(r["candidate"])[:-4]
        mean = (r["candidate_mean"] - r["reference_mean"]) / max(r["reference_mean"], 1e-12) * 100
        print(f"   {scene:38s} relRMSE {r['relative_rmse']:.5f}  mean {mean:+.3f}%  "
              f"max {r['max_abs_diff']:.4g}  finite={r['finite']}")
