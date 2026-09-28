#!/usr/bin/env python3
"""Run bounded CPU and optional Metal coherent geometry numerical probes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(command, root):
    run = subprocess.run(command, cwd=root, text=True, capture_output=True)
    return {"command": command, "exit_code": run.returncode,
            "stdout": run.stdout.strip(), "stderr": run.stderr.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal", action="store_true")
    parser.add_argument("--output-dir", default="build/tests/performance/coherent_geometry_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cpu_source = root / "tests/performance/cycles_coherent_geometry_test.cpp"
    mixed_source = root / "tests/performance/cycles_coherent_mixed_geometry_probe.cpp"
    geometry = root / "intern/cycles/kernel/light/coherent_geometry.h"
    metal_source = root / "tests/metal/cycles_coherent_geometry_probe.metal"
    metal_host = root / "tests/metal/cycles_coherent_geometry_probe.mm"
    files = [cpu_source, mixed_source, geometry, metal_source, metal_host, Path(__file__)]
    report = {"scope": "Standalone planar connector geometry; no renderer scene",
              "sha256": {str(p.relative_to(root)): digest(p) for p in files},
              "cases": []}
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler:
        print("CXX must name a compiler", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="cycles-coherent-geometry-") as temp:
        temp = Path(temp)
        for mode, flags in (("strict", []), ("fast_math", ["-ffast-math"])):
            for name, source in (("geometry", cpu_source), ("mixed", mixed_source)):
                executable = temp / f"coherent_{name}_{mode}"
                command = [*compiler, "-std=c++20", "-O2", *flags,
                           "-DCCL_NAMESPACE_BEGIN=namespace ccl {",
                           "-DCCL_NAMESPACE_END=}",
                           "-I", str(root / "intern/cycles"),
                           "-I", str(root / "intern/cycles/kernel"),
                           str(source), "-o", str(executable)]
                built = invoke(command, root)
                ran = invoke([str(executable)], root) if built["exit_code"] == 0 else None
                report["cases"].append({"name": f"cpu_{name}_{mode}",
                                        "build": built, "run": ran})
                print(f"cpu_{name}_{mode}: {ran['stdout'] if ran else built['stderr']}")
        if args.metal:
            module_cache = temp / "metal_module_cache"
            executable = temp / "coherent_geometry_metal"
            host_command = ["clang++", "-std=c++20", "-x", "objective-c++", "-O2",
                            "-fobjc-arc", "-framework", "Foundation", "-framework", "Metal",
                            str(metal_host), "-o", str(executable)]
            built_host = invoke(host_command, root)
            report["metal_host_build"] = built_host
            for mode, flags in (("safe", ["-fno-fast-math"]),
                                ("fast", ["-ffast-math"])):
                air = temp / f"geometry_{mode}.air"
                library = temp / f"geometry_{mode}.metallib"
                compile_command = ["xcrun", "-sdk", "macosx", "metal", "-c",
                                   "-std=metal3.1", *flags,
                                   f"-fmodules-cache-path={module_cache}",
                                   "-I", str(root / "intern/cycles"),
                                   "-I", str(root / "intern/cycles/kernel"),
                                   str(metal_source), "-o", str(air)]
                compiled = invoke(compile_command, root)
                linked = invoke(["xcrun", "-sdk", "macosx", "metallib", str(air),
                                 "-o", str(library)], root) if compiled["exit_code"] == 0 else None
                ran = invoke([str(executable), str(library)], root) if (
                    linked and linked["exit_code"] == 0 and built_host["exit_code"] == 0) else None
                report["cases"].append({"name": f"metal_{mode}", "compile": compiled,
                                        "link": linked, "run": ran})
                print(f"metal_{mode}: {ran['stdout'] if ran else compiled['stderr']}")
    output = output_dir / "results.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    failed = any(case["run"] is None or case["run"]["exit_code"] != 0
                 for case in report["cases"])
    print(f"results={output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
