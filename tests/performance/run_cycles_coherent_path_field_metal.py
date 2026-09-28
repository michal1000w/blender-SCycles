#!/usr/bin/env python3
"""Compile and execute the tiny completed-path Jones probe on an Apple Metal GPU."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(command, root):
    process = subprocess.run(command, cwd=root, text=True, capture_output=True)
    return {"command": command, "exit_code": process.returncode,
            "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}


def main():
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "build/tests/performance/coherent_path_field_metal_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    metal_source = root / "tests/metal/cycles_coherent_path_field_probe.metal"
    host_source = root / "tests/metal/cycles_coherent_path_field_probe.mm"
    sources = [metal_source, host_source, Path(__file__),
               root / "intern/cycles/kernel/light/coherent_path_field.h",
               root / "intern/cycles/kernel/light/coherent_field.h",
               root / "intern/cycles/kernel/light/coherent_geometry.h"]
    report = {"scope": "One-thread M5 compute validation of completed-path Jones fields; no render",
              "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sources}, "cases": []}
    with tempfile.TemporaryDirectory(prefix="coherent-path-metal-") as temp_dir:
        temp = Path(temp_dir)
        host = temp / "path_field_host"
        report["host_build"] = run([
            "clang++", "-std=c++20", "-x", "objective-c++", "-O2", "-fobjc-arc",
            "-framework", "Foundation", "-framework", "Metal", str(host_source),
            "-o", str(host)], root)
        for name, flags in (("safe", ["-fno-fast-math"]), ("fast", ["-ffast-math"])):
            air = temp / f"path_field_{name}.air"
            library = temp / f"path_field_{name}.metallib"
            compiled = run(["xcrun", "-sdk", "macosx", "metal", "-c", "-std=metal3.1",
                            *flags, f"-fmodules-cache-path={temp / 'module_cache'}",
                            "-I", str(root / "intern/cycles"),
                            "-I", str(root / "intern/cycles/kernel"),
                            str(metal_source), "-o", str(air)], root)
            linked = run(["xcrun", "-sdk", "macosx", "metallib", str(air),
                          "-o", str(library)], root) if compiled["exit_code"] == 0 else None
            executed = run([str(host), str(library)], root) if (
                linked and linked["exit_code"] == 0 and
                report["host_build"]["exit_code"] == 0) else None
            report["cases"].append({"name": name, "compile": compiled,
                                    "link": linked, "run": executed})
            print(f"metal_{name}: {executed['stdout'] if executed else compiled['stderr']}")
    report["all_passed"] = all(case["run"] is not None and case["run"]["exit_code"] == 0
                                for case in report["cases"])
    output = output_dir / "results.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results={output}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
