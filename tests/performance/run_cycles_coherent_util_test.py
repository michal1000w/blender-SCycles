#!/usr/bin/env python3
"""Compile the coherent utility CPU test and check it against a float64 oracle."""

import math
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile


def f32(value):
    """Round an input coordinate as the C++ float3 constructor does."""
    import struct
    return struct.unpack("f", struct.pack("f", value))[0]


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(math.fsum(component * component for component in v))


def range_difference(receiver, source_a, source_b):
    to_a, to_b = sub(receiver, source_a), sub(receiver, source_b)
    delta_source = sub(source_b, source_a)
    numerator = math.fsum(delta_source[i] * (to_a[i] + to_b[i]) for i in range(3))
    return numerator / (norm(to_a) + norm(to_b))


def phase_cycles(path_difference, wavelength, offset):
    return (math.fmod(path_difference, wavelength) / wavelength + offset) % 1.0


def expected(receiver, source_a, source_b, wavelength):
    delta = range_difference(receiver, source_a, source_b)
    phase = phase_cycles(delta, f32(wavelength), f32(0.137))
    ra, rb = norm(sub(receiver, source_a)), norm(sub(receiver, source_b))
    ia, ib = 1.0 / (ra * ra), 0.7 / (rb * rb)
    gamma = math.exp(-0.5 * (delta / f32(0.002)) ** 2)
    intensity = ia + ib + 2.0 * gamma * math.sqrt(ia * ib) * math.cos(2 * math.pi * phase)
    return delta, phase, gamma, intensity


def circular_error(a, b):
    return abs((a - b + 0.5) % 1.0 - 0.5)


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    source = root / "tests/performance/cycles_coherent_util_test.cpp"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler:
        print("CXX must name a compiler", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="cycles-coherent-util-") as temp:
        executable = pathlib.Path(temp) / "cycles_coherent_util_test"
        command = [
            *compiler,
            "-std=c++20",
            "-DCCL_NAMESPACE_BEGIN=namespace ccl {",
            "-DCCL_NAMESPACE_END=}",
            "-I", str(root / "intern/cycles"),
            "-I", str(root / "intern/cycles/kernel"),
            str(source), "-o", str(executable),
        ]
        built = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        if built.stdout:
            print(built.stdout, end="")
        if built.stderr:
            print(built.stderr, end="", file=sys.stderr)
        if built.returncode:
            return built.returncode
        run = subprocess.run([str(executable)], cwd=root, text=True,
                             capture_output=True, check=False)
        if run.returncode:
            if run.stdout:
                print(run.stdout, end="")
            if run.stderr:
                print(run.stderr, end="", file=sys.stderr)
            return run.returncode

    source_a = (0.0, 0.0, 0.0)
    source_b = (f32(0.018), 0.0, f32(0.003))
    translation = (f32(1000.0), f32(-700.0), f32(350.0))
    wavelengths = (380e-9, 550e-9, 780e-9)
    rows = [line.split(",") for line in run.stdout.splitlines() if line.startswith("sample,")]
    max_range_error = max_phase_error = max_gamma_error = max_intensity_error = 0.0
    max_moved_phase_error = max_moved_intensity_error = 0.0
    failures = 0
    if len(rows) != 15:
        print(f"expected 15 C++ sample rows, got {len(rows)}", file=sys.stderr)
        return 1
    for row in rows:
        _, x_text, wavelength_text, range_text, phase_text, gamma_text, intensity_text, \
            moved_range_text, moved_phase_text, moved_gamma_text, moved_intensity_text = row
        x, wavelength = float(x_text), float(wavelength_text)
        receiver = (x, 0.0, f32(3.0))
        reference = expected(receiver, source_a, source_b, wavelength)
        moved_receiver = tuple(f32(receiver[i] + translation[i]) for i in range(3))
        moved_a = tuple(f32(source_a[i] + translation[i]) for i in range(3))
        moved_b = tuple(f32(source_b[i] + translation[i]) for i in range(3))
        moved_reference = expected(moved_receiver, moved_a, moved_b, wavelength)
        measured = (float(range_text), float(phase_text), float(gamma_text), float(intensity_text))
        moved = (float(moved_range_text), float(moved_phase_text),
                 float(moved_gamma_text), float(moved_intensity_text))
        max_range_error = max(max_range_error, abs(measured[0] - reference[0]))
        max_phase_error = max(max_phase_error, circular_error(measured[1], reference[1]))
        max_gamma_error = max(max_gamma_error, abs(measured[2] - reference[2]))
        max_intensity_error = max(max_intensity_error, abs(measured[3] - reference[3]))
        max_moved_phase_error = max(max_moved_phase_error,
                                    circular_error(moved[1], moved_reference[1]))
        max_moved_intensity_error = max(max_moved_intensity_error,
                                        abs(moved[3] - moved_reference[3]))

    # Fixed absolute tolerances keep near-zero cases meaningful. Input float
    # coordinate quantization is included in the oracle coordinates, not hidden.
    failures += max_range_error > 5e-10
    failures += max_phase_error > 0.001
    failures += max_gamma_error > 1e-6
    failures += max_intensity_error > 4e-4
    failures += max_moved_phase_error > 0.001
    failures += max_moved_intensity_error > 4e-4
    print("python_float64_oracle "
          f"range_error_m={max_range_error:.9g} "
          f"phase_error_cycles={max_phase_error:.9g} "
          f"mutual_coherence_error={max_gamma_error:.9g} "
          f"intensity_error={max_intensity_error:.9g} "
          f"translated_phase_error_cycles={max_moved_phase_error:.9g} "
          f"translated_intensity_error={max_moved_intensity_error:.9g} "
          f"failures={failures}")
    for line in run.stdout.splitlines():
        if line.startswith("checked="):
            print(line)
    if failures:
        if run.stderr:
            print(run.stderr, end="", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
