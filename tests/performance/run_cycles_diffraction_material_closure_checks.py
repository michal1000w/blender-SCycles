#!/usr/bin/env python3
"""Run selected closure/cache tests against current Blender build libraries.
GPU runs only for the explicit thin_sheet_gpu_cache selector.

Records compile/link commands, source hashes and numeric output under build/tests.
A cohesive full build must precede this runner after any scene/kernel ABI changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TESTS = ("multiggx_closure", "two_sided_closure", "principled_multiggx")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, cwd, env=None):
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    return {"argv": argv, "cwd": str(cwd), "exit": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/macos_arm64_Release")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build/tests/performance/principled_multiggx_v27")
    parser.add_argument("--test", nargs="+", choices=TESTS + ("conductor", "generalized_multiggx", "coated_generalized_multiggx", "coherent_detector", "coherent_patch_membership", "coherent_terminal_detector", "thin_wall_return_experiment", "thin_wall_reciprocal_envelope", "thin_sheet_cache", "thin_sheet_completed_closure", "thin_sheet_gpu_cache", "polarizer_payload"),
                        default=TESTS)
    parser.add_argument("--fresh-builder", action="store_true",
                        help="Compile the current cache builder beside tests before cohesive rebuild")
    args = parser.parse_args()
    build, output = args.build_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    database = build / "compile_commands.json"
    entries = json.loads(database.read_text())
    entry = next(item for item in entries if item["file"].endswith("/scene/diffraction_albedo.cpp"))
    template = shlex.split(entry["command"])
    commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "bin/Blender.app/Contents/MacOS/Blender"],
        text=True)
    link = shlex.split(commands.splitlines()[-1].split(" && ")[1])
    environment = dict(os.environ)
    environment["DYLD_LIBRARY_PATH"] = ":".join(sorted({
        str(path.parent) for path in (ROOT / "lib/macos_arm64").rglob("*.dylib")}))
    results = {"compile_database_sha256": sha256(database), "tests": {},
               "runtime_library_path": environment["DYLD_LIBRARY_PATH"],
               "kernel_source_hashes": {}}
    for name in ("bsdf.h", "bsdf_diffraction_conductor.h", "bsdf_diffraction_dielectric.h", "diffraction_thin_sheet_model.h"):
        path = ROOT / "intern/cycles/kernel/closure" / name
        results["kernel_source_hashes"][str(path)] = sha256(path)
    for path in (ROOT / "intern/cycles/kernel/svm/closure.h",
                 ROOT / "intern/cycles/kernel/osl/closures_setup.h"):
        results["kernel_source_hashes"][str(path)] = sha256(path)
    builder_obj = None
    if args.fresh_builder:
        builder_obj = output / "current_builder.o"
        command = template.copy()
        command[command.index("-o") + 1] = str(builder_obj)
        results["current_builder_compile"] = run(command, entry["directory"])
        results["builder_source_sha256"] = sha256(Path(entry["file"]))
        if results["current_builder_compile"]["exit"]:
            (output / "results.json").write_text(json.dumps(results, indent=2)+"\n")
            return 1
        results["note"] = "Current cache builder/test object with frozen other libs; fresh cohesive rerun remains required."
    for name in ("dielectric_dispersion.h", "dielectric_f0_cache.h",
                 "diffraction_two_sided_lookup.h", "diffraction_two_sided.h"):
        path = ROOT / "intern/cycles/kernel/util" / name
        results["kernel_source_hashes"][str(path)] = sha256(path)
    for relative in ("light/coherent_detector.h", "light/coherent_specular.h",
                     "integrator/surface_shader.h", "integrator/shade_surface.h",
                     "integrator/bidirectional.h"):
        path = ROOT / "intern/cycles/kernel" / relative
        results["kernel_source_hashes"][str(path)] = sha256(path)
    for name in args.test:
        source = ROOT / f"tests/performance/cycles_diffraction_{name}_test.cpp"
        obj, executable = output / f"{name}.o", output / name
        command = template.copy()
        command[command.index("-c") + 1] = str(source)
        command[command.index("-o") + 1] = str(obj)
        record = {"source": str(source), "source_sha256": sha256(source)}
        results["tests"][name] = record
        record["compile"] = run(command, entry["directory"])
        if record["compile"]["exit"]:
            break
        command = []
        for argument in link:
            if argument.endswith(".o"):
                continue
            if argument.endswith("libpython3.13.a"):
                argument = "-Wl,-force_load," + argument
            command.append(argument)
        command.insert(1, str(obj))
        if builder_obj:
            command.insert(2, str(builder_obj))
        command[command.index("-o") + 1] = str(executable)
        record["link"] = run(command, build)
        if record["link"]["exit"]:
            break
        if name == "thin_sheet_gpu_cache":
            # Device runtime source lookup expects SOURCE_ROOT/source; use the
            # current repository source, not a stale packaged resource tree.
            with tempfile.TemporaryDirectory(prefix="cycles-thin-cache-source-") as temporary:
                source_root = Path(temporary)
                (source_root / "source").symlink_to(ROOT / "intern/cycles", target_is_directory=True)
                comparison = output / "gpu_cache_comparison.json"
                record["runtime_source_tree"] = str(ROOT / "intern/cycles")
                record["run"] = run([str(executable), str(comparison), str(source_root)], build, environment)
                if comparison.exists():
                    record["gpu_cache_comparison"] = json.loads(comparison.read_text())
        else:
            record["run"] = run([str(executable)], build, environment)
        record["executable_sha256"] = sha256(executable)
        print(name, record["run"]["exit"], record["run"]["stdout"], flush=True)
        if record["run"]["exit"]:
            break
    results["passed"] = len(results["tests"]) == len(args.test) and all(
        test.get("run", {}).get("exit") == 0 for test in results["tests"].values())
    path = output / "results.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results_json={path}")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
