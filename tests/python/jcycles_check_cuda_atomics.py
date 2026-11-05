# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""
Check compiled CUDA kernels for a duplicated atomic reservation.

NVCC (seen with CUDA 11.8) rematerialized the warp-aggregated atomicAdd of the GPU guiding
refinement at each use of its result, which reserved every new pair of nodes four times and
corrupted the guiding tree (see atomic_reserve_uint32() in util/atomic.h). This disassembles a
cubin and checks that:

  * kernel_gpu_guiding_refine reserves its nodes with exactly one atomic that returns a value,
  * no other kernel returns-and-uses an atomic add on one address three times or more.

  python tests/python/jcycles_check_cuda_atomics.py BUILD/intern/cycles/kernel/device/cuda/\
kernel_transport_sm_86.cubin [--cuobjdump PATH]

Exits with a non-zero status when a check fails. Plain .cubin files only (not .zst).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

FUNCTION = re.compile(r"Function : (\S+)")
ATOMIC_RESULT = re.compile(r"ATOMG?\.E\.ADD(?!\.F)\S*\s+(?:PT,\s*)?R\d+,\s*(\[[^\]]+\])")


def find_cuobjdump(explicit):
    if explicit:
        return explicit
    found = shutil.which("cuobjdump")
    if found:
        return found
    for variable in sorted((k for k in os.environ if k.startswith("CUDA_PATH")), reverse=True):
        candidate = os.path.join(os.environ[variable], "bin", "cuobjdump.exe")
        if os.path.exists(candidate):
            return candidate
    sys.exit("cuobjdump not found, pass --cuobjdump")


def scan(cuobjdump, cubin):
    process = subprocess.Popen([cuobjdump, "-sass", cubin], stdout=subprocess.PIPE,
                               text=True, errors="replace")
    function = None
    counts = Counter()
    for line in process.stdout:
        match = FUNCTION.search(line)
        if match:
            if function is not None:
                yield function, counts
            function = match.group(1)
            counts = Counter()
            continue
        if "ATOM" in line and ".ADD" in line:
            atomic = ATOMIC_RESULT.search(line)
            if atomic:
                counts[atomic.group(1)] += 1
    if function is not None:
        yield function, counts
    if process.wait() != 0:
        sys.exit(f"cuobjdump failed on {cubin}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cubin")
    parser.add_argument("--cuobjdump", default="")
    args = parser.parse_args()
    cuobjdump = find_cuobjdump(args.cuobjdump)

    failures = 0
    refine_found = False
    for function, counts in scan(cuobjdump, args.cubin):
        if function == "kernel_gpu_guiding_refine":
            refine_found = True
            total = sum(counts.values())
            status = "ok" if total == 1 else "FAIL"
            failures += status != "ok"
            print(f"[atomics] {function}: {total} reservation atomic(s) {status}")
            continue
        repeated = {address: n for address, n in counts.items() if n >= 3}
        if repeated:
            failures += 1
            print(f"[atomics] {function}: repeated atomics {repeated} FAIL")
    if not refine_found:
        print("[atomics] kernel_gpu_guiding_refine not in this cubin (not a transport module?)")
    print(f"[atomics] {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
