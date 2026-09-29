#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build and run the streamed closed-convex Glass host/history test (strict and fast math)."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "build/tests/performance/coherent_streamed_glass_host_v46"
SOURCES = [ROOT / "tests/performance/cycles_coherent_streamed_glass_host_test.cpp",
           ROOT / "intern/cycles/scene/coherent_convex_mesh.h",
           ROOT / "intern/cycles/scene/coherent_planar_cluster.h",
           ROOT / "intern/cycles/kernel/light/coherent_history.h"]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {"source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in SOURCES}, "modes": {}}
    ok = True
    for mode, flags in (("strict", ["-O2"]), ("fast_math", ["-O3", "-ffast-math"])):
        executable = OUTPUT / f"cycles_coherent_streamed_glass_host_test_{mode}"
        command = ["c++", "-std=c++20", *flags, "-DCCL_NAMESPACE_BEGIN=namespace ccl {",
                   "-DCCL_NAMESPACE_END=}", "-Iintern/cycles", "-Iintern/glew-mx",
                   "-Iintern/guardedalloc", "-I.", str(SOURCES[0]), "-o", str(executable)]
        build = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        run = subprocess.run([str(executable)], text=True, capture_output=True) if build.returncode == 0 else None
        result["modes"][mode] = {"command": command, "build_exit_code": build.returncode,
                                 "build_stderr": build.stderr,
                                 "run_exit_code": run.returncode if run else None,
                                 "run_stdout": run.stdout if run else None}
        ok &= build.returncode == 0 and run is not None and run.returncode == 0
        print(mode, run.stdout.strip() if run else build.stderr[-400:])
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
