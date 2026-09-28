#!/usr/bin/env python3
"""Compile and run native diffraction CPU acceptance tests.

Uses the repository build's Cycles device-test compile flags and link command,
but substitutes each standalone source and writes all evidence under build/.
No Blender render or GPU device is used.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEVICE_TEST_SOURCE = ROOT / "tests/performance/cycles_diffraction_device_test.cpp"
TEST_SOURCES = {
    "ashikhmin": ROOT / "tests/performance/cycles_diffraction_ashikhmin_test.cpp",
    "thin_wall": ROOT / "tests/performance/cycles_diffraction_thin_wall_native_test.cpp",
    "albedo": ROOT / "tests/performance/cycles_diffraction_albedo_test.cpp",
    "multiggx": ROOT / "tests/performance/cycles_diffraction_multiggx_closure_test.cpp",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_option_value(argv, option, replacement):
    try:
        index = argv.index(option)
    except ValueError as error:
        raise RuntimeError(f"command is missing {option}") from error
    if index + 1 >= len(argv):
        raise RuntimeError(f"command has no value after {option}")
    argv[index + 1] = str(replacement)


def compile_template(compile_db, build_dir):
    entries = json.loads(compile_db.read_text())
    for entry in entries:
        if Path(entry["file"]).resolve() == DEVICE_TEST_SOURCE.resolve():
            argv = shlex.split(entry["command"])
            cwd = Path(entry["directory"])
            if not cwd.is_absolute():
                cwd = ROOT / cwd
            return argv, cwd
    raise RuntimeError(f"no compile_commands.json entry for {DEVICE_TEST_SOURCE}")


def compile_command(template, source, output, link_executable):
    argv = list(template)
    if "-c" not in argv:
        raise RuntimeError("device-test compile template has no -c flag")
    if link_executable:
        argv.remove("-c")
    replace_option_value(argv, "-o", output)
    source_args = [i for i, value in enumerate(argv) if value == str(DEVICE_TEST_SOURCE)]
    if len(source_args) != 1:
        source_args = [i for i, value in enumerate(argv)
                       if Path(value).name == DEVICE_TEST_SOURCE.name]
    if len(source_args) != 1:
        raise RuntimeError("could not identify the device-test source argument")
    argv[source_args[0]] = str(source)
    return argv


def target_link_template(build_dir):
    result = subprocess.run(
        ["ninja", "-C", str(build_dir), "-t", "commands", "cycles_diffraction_device_test"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"ninja -t commands failed: {result.stderr}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("ninja returned no commands for cycles_diffraction_device_test")
    line = lines[-1].strip()
    if line.startswith(": &&"):
        line = line[len(": &&"):].strip()
    if line.endswith("&& :"):
        line = line[:-len("&& :")].strip()
    return shlex.split(line), result.stdout


def run_process(argv, cwd):
    process = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "argv": argv,
        "cwd": str(cwd),
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def execute_test(name, source, executable, object_file, compile_template_argv,
                 link_template_argv, cwd):
    record = {
        "source": str(source),
        "source_sha256": sha256(source),
        "compile": None,
        "link": None,
        "run": None,
        "executable": str(executable),
        "executable_sha256": None,
        "status": "compile_failed",
    }
    compile_argv = compile_command(
        compile_template_argv, source,
        executable if name == "ashikhmin" else object_file,
        link_executable=(name == "ashikhmin"),
    )
    record["compile"] = run_process(compile_argv, cwd)
    if record["compile"]["exit_code"] != 0:
        return record

    if name != "ashikhmin":
        link_argv = list(link_template_argv)
        old_object = "intern/cycles/test/CMakeFiles/cycles_diffraction_device_test.dir/__/__/__/tests/performance/cycles_diffraction_device_test.cpp.o"
        if old_object not in link_argv:
            raise RuntimeError("native target link command did not contain the device-test object")
        link_argv[link_argv.index(old_object)] = str(object_file)
        old_output = "bin/cycles_diffraction_device_test"
        if old_output not in link_argv:
            raise RuntimeError("native target link command did not contain the device-test output")
        link_argv[link_argv.index(old_output)] = str(executable)
        record["link"] = run_process(link_argv, cwd)
        if record["link"]["exit_code"] != 0:
            record["status"] = "link_failed"
            return record

    record["executable_sha256"] = sha256(executable)
    record["run"] = run_process([str(executable)], cwd)
    record["status"] = "passed" if record["run"]["exit_code"] == 0 else "test_failed"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/macos_arm64_Release")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "build/tests/performance/native_extensions_final_v3",
    )
    args = parser.parse_args()
    build_dir = args.build_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compile_db = build_dir / "compile_commands.json"
    if not compile_db.is_file():
        parser.error(f"compile database not found: {compile_db}")

    result = {
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "build_dir": str(build_dir),
        "compile_database": str(compile_db),
        "compile_database_sha256": sha256(compile_db),
        "target_link_command_source": "ninja -C <build-dir> -t commands cycles_diffraction_device_test (final line)",
        "target_link_command_raw": "",
        "tests": {},
    }
    try:
        template_argv, cwd = compile_template(compile_db, build_dir)
        link_argv, raw_link_commands = target_link_template(build_dir)
        result["compile_template_argv"] = template_argv
        result["compile_template_cwd"] = str(cwd)
        result["target_link_template_argv"] = link_argv
        result["target_link_command_raw"] = raw_link_commands
        for name, source in TEST_SOURCES.items():
            executable = output_dir / f"cycles_diffraction_{name}_native_test"
            object_file = output_dir / f"cycles_diffraction_{name}_native_test.o"
            result["tests"][name] = execute_test(
                name, source, executable, object_file,
                template_argv, link_argv, cwd,
            )
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        result["setup_error"] = f"{type(error).__name__}: {error}"

    result["passed"] = (
        len(result["tests"]) == len(TEST_SOURCES) and
        all(test["status"] == "passed" for test in result["tests"].values())
    )
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, test in result["tests"].items():
        print(f"{name}: {test['status']}")
        for stage in ("compile", "link", "run"):
            record = test.get(stage)
            if record:
                if record["stdout"]:
                    print(record["stdout"], end="")
                if record["stderr"]:
                    print(record["stderr"], end="", file=sys.stderr)
    if result.get("setup_error"):
        print(result["setup_error"], file=sys.stderr)
    print(f"results_json={result_path}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
