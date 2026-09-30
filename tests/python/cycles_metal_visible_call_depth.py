#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the Metal visible shading function contract against the kernel call graph.

The generic Metal library compiles the shader interpreter, closures and surface stages as
separately compiled visible functions (__KERNEL_METAL_VISIBLE_SHADING__). The host links them
only into kernels listed by metal_kernel_uses_shading_functions() and declares a maximum call
depth per pipeline. A kernel calling a function table without linked functions, or nesting
calls deeper than declared, corrupts GPU execution. This script compiles the kernel to LLVM IR
with xcrun, derives which function tables every kernel can reach and how deeply calls nest,
and checks both against the host code.

  python3 tests/python/cycles_metal_visible_call_depth.py
"""

import collections
import functools
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CYCLES = ROOT / "intern" / "cycles"


# Function name patterns implementing each visible function table, in `kernel.metal`.
TABLE_FUNCTIONS = {
    "vft_svm": r"cycles_metal_svm_\d+",
    "vft_svm_node": r"cycles_metal_svm_node",
    "vft_svm_closure": r"cycles_metal_svm_closure",
    "vft_bsdf_eval": r"cycles_metal_bsdf_eval",
    "vft_bsdf_eval_delta": r"cycles_metal_bsdf_eval_delta",
    "vft_bsdf_sample": r"cycles_metal_bsdf_sample",
    "vft_surface": r"cycles_metal_surface_\d+",
    "vft_polarization": r"cycles_metal_polarization_surface_transport",
    "vft_diffraction": r"cycles_metal_diffraction_power_column",
    "vft_mnee": r"cycles_metal_mnee_sample",
    "vft_pixel_displacement": r"cycles_metal_pixel_displacement_shader_setup",
}


def device_kernel_order():
    text = (CYCLES / "kernel" / "types.h").read_text()
    body = text[text.index("enum DeviceKernel"):]
    body = body[:body.index("DEVICE_KERNEL_NUM")]
    return re.findall(r"^\s*(DEVICE_KERNEL_\w+)", body, re.M)


def host_function_body(name):
    text = (CYCLES / "device" / "metal" / "kernel.h").read_text()
    body = text[text.index(f"{name}("):]
    return body[:body.index("\n}")]


def host_shading_kernels(displacement):
    """Evaluate metal_kernel_uses_shading_functions() from device/metal/kernel.h."""
    body = host_function_body("metal_kernel_uses_shading_functions")
    if not displacement:
        body = re.sub(r"\(pixel_displacement && [^)]*\)", "", body, flags=re.S)
    order = device_kernel_order()
    index = {name: i for i, name in enumerate(order)}
    kernels = set()
    for low, high in re.findall(r"kernel >= (DEVICE_KERNEL_\w+) &&\s*kernel <= (DEVICE_KERNEL_\w+)",
                                body):
        kernels.update(order[index[low]:index[high] + 1])
    kernels.update(re.findall(r"kernel == (DEVICE_KERNEL_\w+)", body))
    return {name[len("DEVICE_KERNEL_"):].lower() for name in kernels}


def host_call_depths(displacement):
    """Evaluate metal_kernel_shading_call_depth() from device/metal/kernel.h."""
    body = host_function_body("metal_kernel_shading_call_depth")
    deep, default = re.search(r"\?\s*(\d+)\s*:\s*(\d+)\)", body).groups()
    extra = int(re.search(r"pixel_displacement \? (\d+) : 0", body).group(1)) if displacement else 0
    kernels = {name[len("DEVICE_KERNEL_"):].lower()
               for name in re.findall(r"kernel == (DEVICE_KERNEL_\w+)", body)}
    return kernels, int(deep) + extra, int(default) + extra


def ancillary_tables():
    """Visible function table names in MetalAncillaries order, from compat.h."""
    text = (CYCLES / "kernel" / "device" / "metal" / "compat.h").read_text()
    struct = text[text.index("struct MetalAncillaries {"):]
    struct = struct[:struct.index("};")]
    return re.findall(r"visible_function_table<\w+> (vft_\w+);", struct)


def compile_ir(metalrt, displacement, output, motion=False):
    defines = [
        "-D__KERNEL_METAL_VISIBLE_SHADING__", "-D__KERNEL_LOCAL_ATOMIC_SORT__",
        "-D__KERNEL_METAL_APPLE__", "-D__METAL_FUNCTION_CONSTANTS_64BIT__",
        "-D__METAL_GLOBAL_BUILTINS__", "-D__KERNEL_METAL_TARGET_CPU_ARM64__",
        "-D__KERNEL_METAL_MACOS__=15", "-DWITH_NANOVDB",
    ]
    if metalrt:
        defines.append("-D__KERNEL_METALRT__")
    if motion:
        defines.append("-D__METALRT_MOTION__")
    if displacement:
        defines += ["-D__KERNEL_METAL_PIXEL_DISPLACEMENT__",
                    "-D__KERNEL_METAL_PIXEL_DISPLACEMENT_SHADE__"]
    subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-S", "-emit-llvm", "-std=metal3.2",
                    "-ffast-math", "-w", f"-I{CYCLES}", *defines, "-o", str(output),
                    str(CYCLES / "kernel" / "device" / "metal" / "kernel.metal")], check=True)


def analyze(ll_path, tables):
    calls = collections.defaultdict(set)
    fields = collections.defaultdict(set)
    # Visible function calls as (table field, entry index); index None when not constant.
    entries = collections.defaultdict(set)
    functions = []
    current = None
    struct_fields = None
    gep = re.compile(r"(%[\w.]+) = getelementptr inbounds %struct\.MetalAncillaries, "
                     r"%struct\.MetalAncillaries addrspace\(\d+\)\* [^,]+, i\d+ 0, i32 (\d+)")
    load = re.compile(r"(%[\w.]+) = load %struct\._visible_function_table_t.* (%[\w.]+), align")
    get_function = re.compile(r"@air\.get_function_pointer_visible_function_table\(.* (%[\w.]+), "
                              r"i32 (?:noundef )?(%?[\w.]+)\)")
    field_of_value = {}
    for line in open(ll_path):
        if line.startswith("%struct.MetalAncillaries = type {"):
            struct_fields = [f.strip() for f in
                             re.split(r",(?![^<]*>)", line.split("{", 1)[1].rsplit("}", 1)[0])]
            continue
        match = re.match(r'define [^@]*@("?[^"(]+"?)\(', line)
        if match:
            current = match.group(1).strip('"')
            functions.append(current)
            continue
        if line.startswith("}"):
            current = None
            continue
        if current:
            calls[current].update(c.strip('"') for c in re.findall(r'call [^@]*@("?[\w.$]+"?)\(', line))
            for value, field in gep.findall(line):
                fields[current].add(int(field))
                field_of_value[(current, value)] = int(field)
            match = load.search(line)
            if match and (current, match.group(2)) in field_of_value:
                field_of_value[(current, match.group(1))] = field_of_value[(current, match.group(2))]
            match = get_function.search(line)
            if match and (current, match.group(1)) in field_of_value:
                index = match.group(2)
                entries[current].add((field_of_value[(current, match.group(1))],
                                      None if index.startswith("%") else int(index)))

    visible_fields = [i for i, f in enumerate(struct_fields) if "visible_function_table" in f]
    intersection_fields = {i for i, f in enumerate(struct_fields) if "intersection_function_table" in f}
    if len(visible_fields) != len(tables):
        raise SystemExit(f"MetalAncillaries has {len(visible_fields)} visible tables, compat.h "
                         f"lists {len(tables)}")
    targets = {}
    for field, table in zip(visible_fields, tables):
        pattern = TABLE_FUNCTIONS[table]
        targets[field] = sorted((f for f in functions if re.fullmatch(pattern, f)),
                                key=lambda f: int(re.sub(r"\D", "", f[-3:]) or 0))
        # A table without implementation is fine as long as nothing calls it, checked below.

    for function in functions:
        if fields[function] & set(targets) and not entries[function]:
            raise SystemExit(f"FAILED: could not parse the table calls of {function}")

    def entry_targets(field, index):
        """Functions an entry can call: numbered tables are indexed by the name suffix."""
        if not targets[field]:
            raise SystemExit(f"FAILED: table field {field} is called but has no visible function")
        if index is None:
            return targets[field]
        numbered = [f for f in targets[field] if re.search(r"_\d+$", f)]
        if numbered:
            return [f for f in numbered if f.endswith(f"_{index}")]
        return targets[field]

    def reach(root):
        seen, stack = set(), [root]
        while stack:
            function = stack.pop()
            if function in seen or function not in calls and function not in functions:
                continue
            seen.add(function)
            stack.extend(calls[function])
        return seen

    # MetalRT calls intersection functions through the intersection function tables, and they
    # can call shading functions themselves (pixel displacement evaluates shaders).
    intersection_functions = [f for f in functions if f.startswith("__intersection__")]

    memo = {}
    chains = {}
    visiting = []

    def depth(root):
        if root in memo:
            return memo[root]
        if root in visiting:
            raise SystemExit("FAILED: recursive call chain " + " -> ".join(visiting + [root]))
        visiting.append(root)
        used = set()
        called = set()
        for function in reach(root):
            used |= fields[function]
            called |= entries[function]
        result = 0
        chain = []
        for field in used:
            if field in intersection_fields:
                for function in intersection_functions:
                    if 1 + depth(function) > result:
                        result, chain = 1 + depth(function), [function] + chains[function]
        for field, index in called:
            for target in entry_targets(field, index):
                if 1 + depth(target) > result:
                    result, chain = 1 + depth(target), [target] + chains[target]
        visiting.pop()
        memo[root] = result
        chains[root] = chain
        return result

    for function in intersection_functions:
        depth(function)

    def uses_tables(root):
        reached = reach(root)
        if any(fields[f] & intersection_fields for f in reached):
            # Intersection functions linked into the pipeline run in its context.
            for function in intersection_functions:
                reached |= reach(function)
        return any(fields[f] & set(targets) for f in reached)

    kernels = [f for f in functions if f.startswith("cycles_metal_") and
               not any(re.fullmatch(p, f) for p in TABLE_FUNCTIONS.values())]
    analyze.chains = chains
    return {k[len("cycles_metal_"):]: (uses_tables(k), depth(k)) for k in kernels}


def main():
    tables = ancillary_tables()
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        variants = [(False, False, False), (True, False, False), (True, True, False),
                    (False, False, True), (True, False, True), (True, True, True)]
        for metalrt, motion, displacement in variants:
            linked = host_shading_kernels(displacement)
            deep_kernels, deep_depth, default_depth = host_call_depths(displacement)
            ll_path = Path(tmp) / f"kernel_{int(metalrt)}{int(motion)}{int(displacement)}.ll"
            compile_ir(metalrt, displacement, ll_path, motion)
            for kernel, (uses, depth) in sorted(analyze(ll_path, tables).items()):
                variant = (("metalrt" + ("_motion" if motion else "")) if metalrt else "software") + \
                    ("+disp" if displacement else "")
                if uses and kernel not in linked:
                    errors.append(f"{variant}: {kernel} calls shading functions but does not link them")
                if kernel in linked:
                    declared = deep_depth if kernel in deep_kernels else default_depth
                    if depth > declared:
                        chain = " -> ".join(c.replace("cycles_metal_", "")[:40]
                                            for c in analyze.chains["cycles_metal_" + kernel])
                        errors.append(f"{variant}: {kernel} nests {depth} calls, declares "
                                      f"{declared}: {chain}")
                print(f"{variant:19s} {kernel:52s} uses={int(uses)} depth={depth} "
                      f"linked={int(kernel in linked)}")
    if errors:
        print("\n".join(["FAILED:"] + errors))
        return 1
    print("OK: visible shading call graph matches the host contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
