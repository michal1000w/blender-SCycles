#!/usr/bin/env python3
"""Build and run the standalone coherent-field numerical test with provenance."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "build/tests/performance/coherent_field_v1"
SOURCES = [
    ROOT / "intern/cycles/kernel/light/coherent_field.h",
    ROOT / "intern/cycles/kernel/light/coherent_util.h",
    ROOT / "tests/performance/cycles_coherent_field_test.cpp",
    ROOT / "tests/python/run_cycles_coherent_field_test.py",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    executable = OUTPUT / "cycles_coherent_field_test"
    command = [
        "c++", "-std=c++20", "-O2",
        "-DCCL_NAMESPACE_BEGIN=namespace ccl {", "-DCCL_NAMESPACE_END=}",
        "-Iintern/cycles", "-Iintern/glew-mx", "-Iintern/guardedalloc", "-I.",
        "tests/performance/cycles_coherent_field_test.cpp", "-o", str(executable),
    ]
    build = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    run = (subprocess.run([str(executable)], cwd=ROOT, text=True, capture_output=True)
           if build.returncode == 0 else None)
    result = {
        "scope": "Standalone kernel utility and independent analytic numerical tests; no renderer integration",
        "command": command,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in SOURCES},
        "build_exit_code": build.returncode,
        "build_stdout": build.stdout,
        "build_stderr": build.stderr,
        "run_exit_code": run.returncode if run else None,
        "run_stdout": run.stdout if run else None,
        "run_stderr": run.stderr if run else None,
        "executable_sha256": digest(executable) if run else None,
    }
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"build_exit_code": result["build_exit_code"],
                      "run_exit_code": result["run_exit_code"],
                      "run_stdout": result["run_stdout"]}))
    if build.returncode != 0 or run.returncode != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
