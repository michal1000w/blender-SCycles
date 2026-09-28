#!/usr/bin/env python3
"""Actual strict/fast Metal TT inventories against independent double geometry."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/python"))
from cycles_coherent_sphere_transmission_reference import inventory, spreading

out = ROOT / "build/tests/performance/coherent_sphere_tt_metal_precision_v39"
out.mkdir(parents=True, exist_ok=True)
cases = []
# Preserve the exact original failed GPU tuple, then vary both endpoints.
for y in [-1e-5, 1e-5]:
    for dy, dz in [(.012, -.018), (.015, .025), (-.025, .03)]:
        cases.append([-1, y, .05, .4, dy, dz, 0, 0, 0, .25, 1.5, -1, 0, 0])
for n, theta in [(1, 3), (1.5, 2.9), (1.5, 3.1)]:
    cases.append([2, 0, 0, 2 * math.cos(theta), 2 * math.sin(theta), 0,
                  0, 0, 0, 1, n, 1, 0, 0])
rotation, _ = np.linalg.qr(np.array([[.3, .7, .4], [.8, -.2, .3], [.1, .6, -.9]]))
for scale in [.01, 100]:
    c = np.array([1.7, -2.3, .8]) * scale
    s = rotation @ np.array([-1, -1e-5, .05]) * scale + c
    r = rotation @ np.array([.4, .012, -.018]) * scale + c
    normal = rotation @ np.array([-1, 0, 0])
    cases.append([*s, *r, *c, .25 * scale, 1.5, *normal])
cases = np.asarray(cases, dtype=np.float32).astype(float)
inputs = out / "inputs.txt"
inputs.write_text("\n".join(" ".join(map(str, c)) for c in cases) + "\n")
refs = [inventory(c[:3], c[3:6], c[6:9], c[9], c[10]) for c in cases]
runner = out / "probe"
cmd = ["xcrun", "-sdk", "macosx", "clang++", "-std=c++17",
       str(ROOT / "tests/metal/cycles_coherent_sphere_transmit_probe.mm"),
       "-framework", "Foundation", "-framework", "Metal", "-o", str(runner)]
subprocess.run(cmd, check=True, capture_output=True)
report = {"scope": "actual M5 isolated TT root/curvature/phase arithmetic, strict and production fast flags; no scene render",
          "original_failure_preserved": str(ROOT / "build/tests/performance/coherent_sphere_tt/metal_probe_fast.json"),
          "cases": [], "case_count": len(cases)}
for mode, flag in [("strict", "-fno-fast-math"), ("production_fast", "-ffast-math")]:
    air, lib = out / f"{mode}.air", out / f"{mode}.metallib"
    compile_cmd = ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", flag,
                   "-fmodules-cache-path=/tmp/coherent-tt-metal-module-cache",
                   "-I" + str(ROOT / "intern/cycles"), "-I" + str(ROOT / "intern/cycles/kernel"),
                   "-c", str(ROOT / "tests/metal/cycles_coherent_sphere_transmit_probe.metal"),
                   "-o", str(air)]
    subprocess.run(compile_cmd, check=True, capture_output=True)
    subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(lib)],
                   check=True, capture_output=True)
    result = subprocess.run([str(runner), str(lib), str(inputs)], check=True,
                            capture_output=True, text=True)
    (out / f"{mode}_raw.json").write_text(result.stdout)
    gpu = json.loads(result.stdout)
    errors, opls = [], []
    metrics = {"max_spreading_relative_error": 0., "max_opl_scaled_error": 0.,
               "max_point_scaled_error": 0., "max_pair_opd_error_m": 0.}
    checks = 0
    for i, (c, ref, actual) in enumerate(zip(cases, refs, gpu["cases"])):
        rows = np.asarray(actual["rows"], dtype=float)
        checks += 2
        if not np.isfinite(rows).all() or tuple(rows[0, :2]) != (0, len(ref)):
            errors.append(f"inventory {i}: {rows[0, :2]} expected 0/{len(ref)}")
            opls.append([])
            continue
        route_opl = []
        for k, path in enumerate(ref):
            h, spread, hi, lo = rows[1 + k * 3]
            points = rows[2 + k * 3:4 + k * 3, :3]
            er = abs(spread / spreading(path, c[11:]) - 1)
            ep = abs(hi + lo - path["opl"]) / c[9]
            ex = float(np.max(np.abs(points - path["points"]))) / c[9]
            metrics["max_spreading_relative_error"] = max(metrics["max_spreading_relative_error"], er)
            metrics["max_opl_scaled_error"] = max(metrics["max_opl_scaled_error"], ep)
            metrics["max_point_scaled_error"] = max(metrics["max_point_scaled_error"], ex)
            checks += 5
            checks += 1
            selected = rows[13 + k * 3:16 + k * 3]
            selected_opl = selected[0, 2] + selected[0, 3]
            selected_spread_error = abs(selected[0, 1] / spread - 1)
            selected_point_error = float(np.max(np.abs(selected[1:, :3] - points))) / c[9]
            if (selected_spread_error > 2e-6 or abs(selected_opl - hi - lo) > 5e-11 * c[9] or
                    selected_point_error > 2e-6 or selected[1, 3] != rows[2 + k * 3, 3] or
                    selected[2, 3] != rows[3 + k * 3, 3]):
                errors.append(f"selected/full mismatch {i}/{k}")
            if er > 3e-4 or ep > 2e-10 or ex > 1e-5 or int(rows[2 + k * 3, 3]) != path["morse"]:
                errors.append(f"branch {i}/{k}: spreading {er} OPL/R {ep} point/R {ex}")
            route_opl.append(hi + lo)
        opls.append(route_opl)
    for i in range(3):
        if len(opls[i]) != len(refs[i]) or len(opls[i + 3]) != len(refs[i + 3]):
            continue
        for a in range(len(refs[i])):
            for b in range(len(refs[i + 3])):
                error = abs((opls[i][a] - opls[i + 3][b]) - (refs[i][a]["opl"] - refs[i + 3][b]["opl"]))
                metrics["max_pair_opd_error_m"] = max(metrics["max_pair_opd_error_m"], error)
                checks += 1
    # Seventeen-digit serialization preserves each represented float split.
    if metrics["max_pair_opd_error_m"] > 5e-11:
        errors.append("paired OPD exceeds diagnostic precision gate")
    report["cases"].append({"mode": mode, "compile_command": compile_cmd,
                            "device": gpu["device"], "checks": checks,
                            "metrics": metrics, "errors": errors, "passed": not errors})
files = [Path(__file__), ROOT / "tests/metal/cycles_coherent_sphere_transmit_probe.metal",
         ROOT / "tests/metal/cycles_coherent_sphere_transmit_probe.mm",
         ROOT / "intern/cycles/kernel/light/coherent_sphere_transmit_geometry.h",
         ROOT / "tests/python/cycles_coherent_sphere_transmission_reference.py", inputs]
report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
report["passed"] = all(c["passed"] for c in report["cases"])
(out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["cases"], indent=2))
raise SystemExit(0 if report["passed"] else 1)
