#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure a Blender command with a fresh Cycles cache on macOS.

Example:
  python3 tests/python/cycles_metal_compile_benchmark.py --output /tmp/metal-run -- \
    install/Blender.app/Contents/MacOS/Blender -b --factory-startup \
    --log cycles --log-level -1 --python-exit-code 1 --python scene_test.py

This does NOT clear Apple's system shader cache. Report library and pipeline
cache hits separately; an empty Cycles cache alone does not prove a cold compile.
Physical footprint includes memory charged to the process beyond resident pages.
All Metal compiler services owned by the current user are counted, so close other GPU workloads for a
controlled comparison. Sampling can miss short peaks; it is not a memory limit.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import subprocess
import time


class RUsageInfoV2(ctypes.Structure):
    # macOS SDK sys/resource.h, RUSAGE_INFO_V2.
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
            "pageins", "wired_size", "resident_size", "phys_footprint",
            "proc_start_abstime", "proc_exit_abstime", "child_user_time",
            "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
            "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
        )
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-compiler-footprint-gb", type=float, default=0)
    parser.add_argument("--max-combined-footprint-gb", type=float, default=0)
    parser.add_argument("--require-cold-app-cache", action="store_true",
                        help="Require an app identity with no existing Apple application cache")
    parser.add_argument("--require-first-sample", action="store_true",
                        help="Fail unless Cycles logs a completed positive sample batch")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if platform.system() != "Darwin" or not command or args.timeout <= 0:
        parser.error("Requires macOS, a command after --, and a positive timeout")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Use an empty output directory to preserve prior evidence and ensure a fresh Cycles cache")

    app_cache = None
    bundle_id = None
    plist = Path(command[0]).resolve().parents[1] / "Info.plist"
    if plist.is_file():
        bundle_id = plistlib.loads(plist.read_bytes()).get("CFBundleIdentifier")
        cache_root = subprocess.check_output(["getconf", "DARWIN_USER_CACHE_DIR"], text=True).strip()
        if bundle_id and cache_root:
            app_cache = Path(cache_root) / bundle_id
    cache_preexisted = app_cache.exists() if app_cache is not None else None
    if args.require_cold_app_cache and cache_preexisted is not False:
        parser.error("Cold application cache requires a temporary app copy with a new bundle identity")

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libproc.proc_pid_rusage.restype = ctypes.c_int
    env = dict(os.environ, XDG_CACHE_HOME=str(output / "cache"), CYCLES_METAL_DEBUG="1")
    (output / "manifest.json").write_text(json.dumps({
        "command": command, "macOS": platform.mac_ver()[0], "machine": platform.machine(),
        "cycles_cache": env["XDG_CACHE_HOME"], "system_shader_cache_cleared": False,
        "bundle_id": bundle_id, "apple_application_cache": str(app_cache),
        "apple_application_cache_preexisted": cache_preexisted,
        "sampling_interval_seconds": 0.25,
        "memory_scope": "Blender command plus current-user Metal compiler services; sampled peaks",
    }, indent=2) + "\n")

    peaks = {}
    footprint_failures = 0
    timed_out = False
    exceeded_limit = None
    started = time.monotonic()
    first_sample_seconds = None
    first_sample_peaks = None
    log_offset = 0
    pending_log = ""
    with (output / "run.log").open("w") as log, (output / "memory.jsonl").open("w") as samples:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while process.poll() is None:
                if time.monotonic() - started > args.timeout:
                    timed_out = True
                    break
                rows = subprocess.check_output(["ps", "-axo", "pid=,uid=,rss=,comm="], text=True)
                totals = {f"{group}_{metric}_bytes": 0 for group in ("blender", "compiler")
                          for metric in ("rss", "footprint")}
                sample_failures = 0
                for row in rows.splitlines():
                    pid, uid, rss, name = row.strip().split(None, 3)
                    if int(uid) != os.getuid():
                        continue
                    pid = int(pid)
                    group = "blender" if pid == process.pid else (
                        "compiler" if "MTLCompilerService" in name else None)
                    if group is None:
                        continue
                    totals[f"{group}_rss_bytes"] += int(rss) * 1024
                    usage = RUsageInfoV2()
                    if libproc.proc_pid_rusage(pid, 2, ctypes.byref(usage)) == 0:
                        totals[f"{group}_footprint_bytes"] += usage.phys_footprint
                    else:
                        sample_failures += 1
                for metric in ("rss", "footprint"):
                    totals[f"combined_{metric}_bytes"] = sum(
                        totals[f"{group}_{metric}_bytes"] for group in ("blender", "compiler"))
                footprint_failures += sample_failures
                for key, value in totals.items():
                    peaks[key] = max(peaks.get(key, 0), value)
                samples.write(json.dumps(dict(totals, seconds=time.monotonic() - started,
                                              footprint_failures=sample_failures)) + "\n")
                samples.flush()
                with (output / "run.log").open() as reader:
                    reader.seek(log_offset)
                    pending_log += reader.read()
                    log_offset = reader.tell()
                lines = pending_log.split("\n")
                pending_log = lines.pop()
                if first_sample_seconds is None and any(
                        re.search(r"Rendered [1-9][0-9]* samples in .*occupancy:", line)
                        for line in lines):
                    first_sample_seconds = time.monotonic() - started
                    first_sample_peaks = dict(peaks)
                for metric, limit in (("compiler_footprint_bytes", args.max_compiler_footprint_gb),
                                      ("combined_footprint_bytes", args.max_combined_footprint_gb)):
                    if limit > 0 and totals[metric] > limit * 1e9:
                        exceeded_limit = metric
                if exceeded_limit:
                    break
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    # A short render can complete and exit between polling intervals. Drain its final log too;
    # the observation time is a conservative upper bound, just like live polling above.
    if first_sample_seconds is None:
        with (output / "run.log").open() as reader:
            reader.seek(log_offset)
            final_log = pending_log + reader.read()
        if re.search(r"Rendered [1-9][0-9]* samples in .*occupancy:", final_log):
            first_sample_seconds = time.monotonic() - started
            first_sample_peaks = dict(peaks)
    report = dict(peaks, exit_code=process.returncode, timed_out=timed_out,
                  exceeded_limit=exceeded_limit,
                  elapsed_seconds=time.monotonic() - started,
                  footprint_read_failures=footprint_failures,
                  footprint_complete=footprint_failures == 0,
                  first_sample_seconds=first_sample_seconds,
                  peaks_through_first_sample=first_sample_peaks,
                  apple_application_cache_created=app_cache.exists() if app_cache else None)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(1 if timed_out or exceeded_limit or
                     (args.require_first_sample and first_sample_seconds is None)
                     else process.returncode)


if __name__ == "__main__":
    main()
