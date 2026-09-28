#!/usr/bin/env python3
"""Run standalone completed-path field transport tests and record source provenance."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def run(command, root):
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "build/tests/performance/coherent_path_field_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [root / "intern/cycles/kernel/light/coherent_path_field.h",
               root / "intern/cycles/kernel/light/coherent_field.h",
               root / "intern/cycles/kernel/light/coherent_geometry.h",
               root / "tests/performance/cycles_coherent_path_field_test.cpp",
               Path(__file__)]
    report = {"scope": "Standalone completed-path field transport; no renderer scene",
              "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sources}, "cases": []}
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    with tempfile.TemporaryDirectory(prefix="coherent-path-field-") as temp_dir:
        temp = Path(temp_dir)
        for name, flags in (("cpu_strict", []), ("cpu_fast_math", ["-ffast-math"])):
            binary = temp / name
            command = [*compiler, "-std=c++20", "-O2", *flags,
                       "-DCCL_NAMESPACE_BEGIN=namespace ccl {",
                       "-DCCL_NAMESPACE_END=}",
                       "-I", str(root / "intern/cycles"),
                       "-I", str(root / "intern/cycles/kernel"),
                       str(root / "tests/performance/cycles_coherent_path_field_test.cpp"),
                       "-o", str(binary)]
            build = run(command, root)
            executed = run([str(binary)], root) if build["exit_code"] == 0 else None
            report["cases"].append({"name": name, "build": build, "run": executed})
            print(f"{name}: {executed['stdout'] if executed else build['stderr']}")
        for name, flags in (("metal_safe_compile", ["-fno-fast-math"]),
                            ("metal_fast_compile", ["-ffast-math"])):
            air = temp / f"{name}.air"
            command = ["xcrun", "-sdk", "macosx", "metal", "-c", "-std=metal3.1",
                       *flags, f"-fmodules-cache-path={temp / 'module_cache'}",
                       "-I", str(root / "intern/cycles"),
                       "-I", str(root / "intern/cycles/kernel"),
                       str(root / "tests/metal/cycles_coherent_path_field_compile.metal"),
                       "-o", str(air)]
            build = run(command, root)
            report["cases"].append({"name": name, "build": build})
            print(f"{name}: {'PASS' if build['exit_code'] == 0 else build['stderr']}")
    metal_source = root / "tests/metal/cycles_coherent_path_field_compile.metal"
    report["sha256"][str(metal_source.relative_to(root))] = hashlib.sha256(
        metal_source.read_bytes()).hexdigest()
    output = out_dir / "results.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results={output}")
    return 1 if any(case["build"]["exit_code"] != 0 or
                    ("run" in case and case["run"]["exit_code"] != 0)
                    for case in report["cases"]) else 0


if __name__ == "__main__":
    sys.exit(main())
