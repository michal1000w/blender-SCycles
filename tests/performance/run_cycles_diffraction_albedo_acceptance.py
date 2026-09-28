#!/usr/bin/env python3
"""Compile and run the standalone Cycles diffraction albedo acceptance test.

Usage: python3 tests/performance/run_cycles_diffraction_albedo_acceptance.py

Set CXX to a compiler command (including optional arguments) to select the
compiler. The command is split with shell-like quoting and passed directly to
subprocess without a shell. Results and the executable go under build/ by
default; use --output-dir to choose another location.
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="directory for the test executable and results.json",
    )
    parser.add_argument(
        "--results-json",
        type=pathlib.Path,
        help="override the results JSON path (default: <output-dir>/results.json)",
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[2]
    source = root / "tests/performance/cycles_diffraction_albedo_acceptance_test.cpp"
    output_dir = args.output_dir or root / "build/tests/performance/cycles_diffraction_albedo_acceptance"
    executable = output_dir / "cycles_diffraction_albedo_acceptance_test"
    results_path = args.results_json or output_dir / "results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler:
        parser.error("CXX must name a compiler")
    tbb_includes = sorted((root / "lib").glob("*/tbb/include"))
    if not tbb_includes:
        parser.error("could not find bundled TBB headers under lib/*/tbb/include")
    version = subprocess.run(
        [*compiler, "--version"], text=True, capture_output=True, check=False
    )
    compile_command = [
        *compiler,
        "-std=c++20",
        "-DCCL_NAMESPACE_BEGIN=namespace ccl {",
        "-DCCL_NAMESPACE_END=}",
        "-I",
        str(root / "intern/cycles"),
        "-I",
        str(root / "intern/cycles/kernel"),
        "-I",
        str(tbb_includes[0]),
        str(source),
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(
        compile_command, cwd=root, text=True, capture_output=True, check=False
    )

    result = {
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": str(source.relative_to(root)),
        "source_sha256": sha256(source),
        "compiler_command": compiler,
        "compiler_version": (version.stdout or version.stderr).strip(),
        "compile_command": compile_command,
        "compile_exit_code": compiled.returncode,
        "compile_stdout": compiled.stdout,
        "compile_stderr": compiled.stderr,
        "executable": str(executable),
        "status": "compile_failed",
        "test_exit_code": None,
        "test_stdout": "",
        "test_stderr": "",
        "executable_sha256": None,
    }
    exit_code = compiled.returncode
    if compiled.returncode == 0:
        result["executable_sha256"] = sha256(executable)
        tested = subprocess.run(
            [str(executable)], cwd=root, text=True, capture_output=True, check=False
        )
        result.update(
            status="passed" if tested.returncode == 0 else "test_failed",
            test_exit_code=tested.returncode,
            test_stdout=tested.stdout,
            test_stderr=tested.stderr,
        )
        exit_code = tested.returncode

    temporary_results = results_path.with_suffix(results_path.suffix + ".tmp")
    temporary_results.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_results.replace(results_path)

    if compiled.stdout:
        sys.stdout.write(compiled.stdout)
    if compiled.stderr:
        sys.stderr.write(compiled.stderr)
    if result["test_stdout"]:
        sys.stdout.write(result["test_stdout"])
    if result["test_stderr"]:
        sys.stderr.write(result["test_stderr"])
    print(f"status={result['status']} results={results_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
