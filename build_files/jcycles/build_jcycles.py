#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Package this repository's Blender build as the JCycles add-on for official Blender.

Run through compileCycles.sh (macOS, Linux) or compileCycles.ps1 (Windows) after compile.sh /
compile.ps1 has built and installed Blender. Writes build/JCycles, a folder that can be zipped and
installed in an official Blender (Preferences > Add-ons > Install from Disk).

Steps:
1. Find the installed build of this repository (./install by default) and check it is current.
2. Ask that build for its Cycles features and the exact definitions of the fork's shader nodes.
3. Generate the add-on: the fork's Cycles add-on code renamed to JCycles (properties, UI,
   operators, presets, OSL), plus the JCycles bridge modules from build_files/jcycles/addon.
4. Copy the whole build into JCycles/runtime/<platform>/: renders run in it.
"""

import argparse
import ast
import datetime
import json
import os
import platform
import py_compile
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ADDON_TEMPLATE = os.path.join(HERE, "addon")
CYCLES_ADDON = os.path.join(ROOT, "intern", "cycles", "blender", "addon")
CYCLES_PRESETS = os.path.join(ROOT, "scripts", "presets", "cycles")
NODE_SOURCES = os.path.join(ROOT, "source", "blender", "nodes", "shader", "nodes")

sys.path.insert(0, HERE)
from fork_nodes import FORK_NODES  # noqa: E402

TRANSFORMED_FILES = ("properties.py", "ui.py", "operators.py", "presets.py", "osl.py")
COPIED_FILES = ("camera.py",)
MARKER_FILE = "jcycles_build.json"


class BuildError(Exception):
    pass


def log(text):
    print("\n==> " + text, flush=True)


# -----------------------------------------------------------------------------
# Platform and runtime


def platform_tag():
    system = platform.system()
    machine = platform.machine().lower()
    machine = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64"}.get(machine, machine)
    name = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(system)
    if name is None:
        raise BuildError("Unsupported platform: " + system)
    return "{:s}-{:s}".format(name, machine)


def find_runtime(path, tag):
    """Returns (directory to copy, path of the executable, name of the copy)."""
    if tag.startswith("macos"):
        app = path
        if not app.endswith(".app"):
            app = os.path.join(path, "Blender.app")
        exe = os.path.join(app, "Contents", "MacOS", "Blender")
        if not os.path.isfile(exe):
            raise BuildError("No Blender.app found at {:s}. Run ./compile.sh first.".format(path))
        return app, exe, "Blender.app"
    exe_name = "blender.exe" if tag.startswith("windows") else "blender"
    directory = os.path.dirname(path) if os.path.isfile(path) else path
    exe = os.path.join(directory, exe_name)
    if not os.path.isfile(exe):
        script = "compile.ps1" if tag.startswith("windows") else "./compile.sh"
        raise BuildError("No {:s} found in {:s}. Run {:s} first.".format(exe_name, directory, script))
    return directory, exe, None


def installed_addon_dir(runtime_dir):
    for dirpath, dirnames, _filenames in os.walk(runtime_dir):
        if dirpath.replace("\\", "/").endswith("scripts/addons_core") and "cycles" in dirnames:
            return os.path.join(dirpath, "cycles")
    return None


def check_runtime_is_current(runtime_dir, allow_stale):
    addon = installed_addon_dir(runtime_dir)
    if addon is None:
        raise BuildError("The build in {:s} has no Cycles add-on.".format(runtime_dir))
    stale = []
    for name in ("properties.py", "ui.py", "engine.py", "osl.py", "operators.py", "presets.py"):
        source = os.path.join(CYCLES_ADDON, name)
        installed = os.path.join(addon, name)
        if not os.path.isfile(installed) or open(source, "rb").read() != open(installed, "rb").read():
            stale.append(name)
    if stale:
        message = ("The installed build is older than the sources ({:s} differ). Rebuild with compile.sh / "
                   "compile.ps1, or pass --allow-stale.".format(", ".join(stale)))
        if not allow_stale:
            raise BuildError(message)
        print("warning: " + message)


# -----------------------------------------------------------------------------
# Node declarations in the C++ sources: panels and value ranges


def _statement_end(text, start):
    depth = 0
    i = start
    in_string = None
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == in_string:
                in_string = None
        elif c in "\"'":
            in_string = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                return i
        elif c == ";" and depth == 0:
            return i
        i += 1
    return len(text)


def _function_body(text, signature):
    start = text.find(signature)
    if start < 0:
        return None
    brace = text.find("{", start)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:i]
    return None


_NUMBER = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)f?"


def _number(statement, method):
    match = re.search(r"\." + method + r"\(\s*" + _NUMBER + r"\s*\)", statement)
    return float(match.group(1)) if match else None


def parse_node_source(idname):
    files = [f for f in os.listdir(NODE_SOURCES) if f.endswith(".cc")]
    for name in files:
        path = os.path.join(NODE_SOURCES, name)
        text = open(path, encoding="utf-8").read()
        if '"{:s}"'.format(idname) in text:
            break
    else:
        raise BuildError("No source file declares node {:s}".format(idname))
    body = _function_body(text, "static void node_declare(NodeDeclarationBuilder &b)")
    if body is None:
        raise BuildError("No node_declare() in " + path)
    variables = {"b": None}
    panels = {}
    panel_order = []
    panel_closed = {}
    ranges = {}
    token = re.compile(
        r"PanelDeclarationBuilder\s*&\s*(?P<pvar>\w+)\s*=\s*(?P<parent>\w+)\.add_panel\(\s*\"(?P<pname>[^\"]+)\""
        r"|\b(?P<var>\w+)\.add_(?P<io>input|output)<[^>]+>\(\s*\"(?P<sname>[^\"]+)\"")
    for match in token.finditer(body):
        end = _statement_end(body, match.start())
        statement = body[match.start():end]
        if match.group("pvar"):
            variables[match.group("pvar")] = match.group("pname")
            panel_order.append(match.group("pname"))
            panel_closed[match.group("pname")] = ".default_closed(true)" in statement
            continue
        if match.group("io") != "input":
            continue
        socket = match.group("sname")
        if match.group("var") in variables and variables[match.group("var")]:
            panels.setdefault(socket, variables[match.group("var")])
        low, high = _number(statement, "min"), _number(statement, "max")
        if low is not None or high is not None:
            ranges.setdefault(socket, [low, high])
    return {"panels": panels, "panel_order": panel_order, "panel_closed": panel_closed,
            "ranges": ranges, "source": os.path.relpath(path, ROOT)}


# -----------------------------------------------------------------------------
# Introspection of the build


def introspect(exe, tag):
    request = {"nodes": {}}
    sources = {}
    for idname, config in FORK_NODES.items():
        parsed = parse_node_source(idname)
        sources[idname] = parsed
        entry = dict(config)
        entry.update(panels=parsed["panels"], panel_order=parsed["panel_order"],
                     panel_closed=parsed["panel_closed"])
        request["nodes"][idname] = entry
    with tempfile.TemporaryDirectory(prefix="jcycles_build_") as tmp:
        request_path = os.path.join(tmp, "request.json")
        output_path = os.path.join(tmp, "output.json")
        with open(request_path, "w", encoding="utf-8") as fh:
            json.dump(request, fh)
        env = dict(os.environ)
        env["BLENDER_USER_RESOURCES"] = os.path.join(tmp, "user")
        command = [exe, "--background", "--factory-startup", "-noaudio", "--python-exit-code", "1",
                   "--python", os.path.join(HERE, "introspect.py"), "--", request_path, output_path]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                 text=True, errors="replace")
        if process.returncode != 0 or not os.path.isfile(output_path):
            raise BuildError("Introspecting the build failed:\n" + process.stdout[-4000:])
        with open(output_path, encoding="utf-8") as fh:
            result = json.load(fh)
    if result.get("errors"):
        raise BuildError("Fork node definitions failed:\n  " + "\n  ".join(result["errors"]))
    for idname, config in FORK_NODES.items():
        spec = result["nodes"][idname]
        names = [s["name"] for s in spec["inputs"]]
        missing = [n for n in config.get("extra_inputs", []) if n not in names]
        if missing:
            raise BuildError("{:s} has no inputs {:s}; update build_files/jcycles/fork_nodes.py".format(
                idname, ", ".join(missing)))
        for socket in spec["inputs"]:
            socket["range"] = sources[idname]["ranges"].get(socket["name"])
    return result


# -----------------------------------------------------------------------------
# Code transformation: Cycles add-on -> JCycles add-on


def transform_source(text):
    # Engine identifier, except where Cycles asks for the Cycles target of material outputs.
    text = re.sub(r"(?<!get_output_node\()(['\"])CYCLES\1", r"\1JCYCLES\1", text)
    # Classes, panel/menu/operator identifiers.
    text = re.sub(r"\bCYCLES_", "JCYCLES_", text)
    text = re.sub(r"\bCycles(?=[A-Z][A-Za-z0-9]*\b)", "JCycles", text)
    # Property groups registered on Blender data.
    text = re.sub(r"\.cycles\b", ".jcycles", text)
    text = re.sub(r"\bcycles_(curves|visibility|custom)\b", r"jcycles_\1", text)
    # Operator identifiers and preset directories.
    text = re.sub(r"(['\"])cycles\.", r"\1jcycles.", text)
    text = re.sub(r"(['\"])render\.cycles_", r"\1render.jcycles_", text)
    text = re.sub(r"(['\"])cycles/", r"\1jcycles/", text)
    # The C++ module.
    text = re.sub(r"^(\s*)import _cycles\s*$", r"\1from . import _cycles_shim as _cycles", text, flags=re.M)
    return text


def check_transformed(name, text):
    problems = []
    for match in re.finditer(r"(['\"])CYCLES\1", text):
        before = text[max(0, match.start() - 16):match.start()]
        if not before.endswith("get_output_node("):
            problems.append("engine literal at offset {:d}".format(match.start()))
    if re.search(r"\.cycles\b", text):
        problems.append("'.cycles' attribute access left")
    if re.search(r"^\s*import _cycles\b", text, flags=re.M) or re.search(r"\bfrom _cycles\b", text):
        problems.append("direct _cycles import left")
    if re.search(r"(['\"])cycles\.", text):
        problems.append("'cycles.' operator identifier left")
    if re.search(r"\bclass (Cycles|CYCLES_)", text):
        problems.append("Cycles class name left")
    if problems:
        raise BuildError("Transformation of {:s} is incomplete: {:s}".format(name, "; ".join(problems)))


def generated_passes_source():
    path = os.path.join(CYCLES_ADDON, "engine.py")
    text = open(path, encoding="utf-8").read()
    tree = ast.parse(text)
    parts = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"list_render_passes", "register_passes"}:
            parts.append(ast.get_source_segment(text, node))
    if len(parts) != 2:
        raise BuildError("list_render_passes/register_passes not found in " + path)
    return "# Taken from intern/cycles/blender/addon/engine.py at package time.\n\n" + \
        transform_source("\n\n\n".join(parts)) + "\n"


# -----------------------------------------------------------------------------
# Writing the package


def safe_reset_output(output):
    if os.path.exists(output):
        if os.path.isfile(output):
            raise BuildError("Output path is a file: " + output)
        entries = set(os.listdir(output))
        # A package, or what is left of one after an interrupted delete.
        if entries and MARKER_FILE not in entries and not entries <= {".DS_Store", "runtime"}:
            raise BuildError("Refusing to replace {:s}: it is not a JCycles package.".format(output))
        # Finder may write .DS_Store files while the folder is being deleted: retry.
        import time
        for attempt in range(10):
            try:
                shutil.rmtree(output)
                break
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(0.5)
    os.makedirs(output, exist_ok=True)


# Blender's "Install from Disk" treats a zip containing a blender_manifest.toml at ANY depth as an
# extension and installs that sub-folder. The Blender build ships one as a Text Editor template
# (scripts/templates_toml), so it must stay out of the package or it hijacks the installation.
EXTENSION_MANIFEST = "blender_manifest.toml"


def copy_tree(source, target):
    def ignore(directory, names):
        return [n for n in names if n == "__pycache__" or n.endswith((".pyc", ".pyo")) or n == ".DS_Store"
                or n == EXTENSION_MANIFEST]
    shutil.copytree(source, target, symlinks=False, ignore=ignore, dirs_exist_ok=True)


def check_installable(output):
    """The package must install as a regular add-on: bl_info in __init__.py and no manifest."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(output):
        if EXTENSION_MANIFEST in filenames:
            found.append(os.path.relpath(os.path.join(dirpath, EXTENSION_MANIFEST), output))
    if found:
        raise BuildError("The package contains extension manifests, Blender would install those instead "
                         "of JCycles: " + ", ".join(found))
    if "bl_info" not in open(os.path.join(output, "__init__.py"), encoding="utf-8").read():
        raise BuildError("__init__.py has no bl_info")


def git_info():
    def run(*args):
        try:
            return subprocess.run(["git"] + list(args), cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True).stdout.strip()
        except OSError:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def write_package(output, runtime, exe, app_name, tag, info, copy_runtime):
    log("Writing add-on to " + output)
    safe_reset_output(output)

    copy_tree(ADDON_TEMPLATE, output)
    shutil.copy2(os.path.join(HERE, "README.md"), os.path.join(output, "README.md"))

    for name in TRANSFORMED_FILES:
        text = open(os.path.join(CYCLES_ADDON, name), encoding="utf-8").read()
        text = transform_source(text)
        check_transformed(name, text)
        header = "# Generated by build_files/jcycles/build_jcycles.py from intern/cycles/blender/addon/{:s}.\n".format(name)
        with open(os.path.join(output, name), "w", encoding="utf-8") as fh:
            fh.write(header + text)
    for name in COPIED_FILES:
        shutil.copy2(os.path.join(CYCLES_ADDON, name), os.path.join(output, name))

    engine_path = os.path.join(output, "engine.py")
    text = open(engine_path, encoding="utf-8").read()
    marker = "# @@JCYCLES_GENERATED_PASSES@@"
    if marker not in text:
        raise BuildError("engine.py template has no pass marker")
    text = text.replace(marker, generated_passes_source())
    check_transformed("engine.py", text)
    with open(engine_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    preset_target = os.path.join(output, "presets", "jcycles")
    for dirpath, _dirnames, filenames in os.walk(CYCLES_PRESETS):
        relative = os.path.relpath(dirpath, CYCLES_PRESETS)
        for name in filenames:
            if not name.endswith(".py"):
                continue
            text = transform_source(open(os.path.join(dirpath, name), encoding="utf-8").read())
            check_transformed("preset " + name, text)
            destination = os.path.join(preset_target, relative)
            os.makedirs(destination, exist_ok=True)
            with open(os.path.join(destination, name), "w", encoding="utf-8") as fh:
                fh.write(text)

    fork = {
        "version": info["blender_version"],
        "version_string": info["blender_version_string"],
        "file_version": info["blender_file_version"],
        "build_hash": info["build_hash"],
        "build_branch": info["build_branch"],
        "build_date": info["build_date"],
    }
    build = {
        "jcycles_version": [1, 0, 0],
        "packaged": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "platform": tag,
        "fork": fork,
        "git": git_info(),
        "cycles": info["cycles"],
    }
    with open(os.path.join(output, MARKER_FILE), "w", encoding="utf-8") as fh:
        json.dump(build, fh, indent=1)
    with open(os.path.join(output, "node_specs.json"), "w", encoding="utf-8") as fh:
        json.dump({"nodes": info["nodes"]}, fh, indent=1)

    init_path = os.path.join(output, "__init__.py")
    text = open(init_path, encoding="utf-8").read()
    description = "Cycles of blender-SCycles {:s} ({:s}) as a separate engine".format(
        info["blender_version_string"], info["build_hash"][:10])
    text = text.replace("@JCYCLES_DESCRIPTION@", description.replace('"', "'"))
    with open(init_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    log("Checking generated Python")
    for dirpath, _dirnames, filenames in os.walk(output):
        for name in filenames:
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                try:
                    py_compile.compile(path, cfile=os.path.join(tempfile.gettempdir(), "jcycles_check.pyc"),
                                       doraise=True)
                except py_compile.PyCompileError as ex:
                    raise BuildError("Generated file does not compile: {:s}\n{:s}".format(path, str(ex)))

    if copy_runtime:
        target = os.path.join(output, "runtime", tag)
        log("Copying the Blender build into " + target + " (this takes a while)")
        if app_name:
            os.makedirs(target)
            copy_tree(runtime, os.path.join(target, app_name))
        else:
            copy_tree(runtime, target)
    else:
        print("warning: --no-runtime: the package cannot render until a runtime is added")
    check_installable(output)


def make_zip(output, zip_path):
    log("Creating " + zip_path)
    base = os.path.dirname(output)
    top = os.path.basename(output)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for dirpath, dirnames, filenames in os.walk(output):
            dirnames.sort()
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                arcname = os.path.relpath(full, base).replace(os.sep, "/")
                info = zipfile.ZipInfo.from_file(full, arcname)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (os.stat(full).st_mode & 0xFFFF) << 16
                with open(full, "rb") as src, archive.open(info, "w", force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
    print("Zip: {:s} ({:.0f} MB)".format(zip_path, os.path.getsize(zip_path) / 1e6))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime", default=os.path.join(ROOT, "install"),
                        help="Installed build of this repository (default: ./install)")
    parser.add_argument("--output", default=os.path.join(ROOT, "build", "JCycles"),
                        help="Add-on folder to write (default: ./build/JCycles)")
    parser.add_argument("--zip", action="store_true", help="Also write build/JCycles-<version>-<platform>.zip")
    parser.add_argument("--allow-stale", action="store_true",
                        help="Package even when the installed build is older than the sources")
    parser.add_argument("--no-runtime", action="store_true",
                        help="Do not copy the Blender build (for add-on development only)")
    args = parser.parse_args(argv)

    try:
        tag = platform_tag()
        runtime, exe, app_name = find_runtime(os.path.abspath(args.runtime), tag)
        log("Using build {:s} ({:s})".format(runtime, tag))
        check_runtime_is_current(runtime, args.allow_stale)
        log("Reading Cycles features and fork node definitions from the build")
        info = introspect(exe, tag)
        print("  Blender {:s} ({:s}), {:d} fork nodes".format(
            info["blender_version_string"], info["build_hash"][:12], len(info["nodes"])))
        output = os.path.abspath(args.output)
        write_package(output, runtime, exe, app_name, tag, info, not args.no_runtime)
        if args.zip:
            version = info["blender_version_string"].replace(" ", "-")
            make_zip(output, os.path.join(os.path.dirname(output),
                                          "JCycles-{:s}-{:s}.zip".format(version, tag)))
    except BuildError as ex:
        print("\nerror: " + str(ex), file=sys.stderr)
        return 1
    log("Done")
    print("Add-on folder: " + os.path.abspath(args.output))
    print("Install: zip the JCycles folder (or use --zip), then in an official Blender use")
    print("  Preferences > Add-ons > Install from Disk, and enable 'JCycles Render Engine'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
