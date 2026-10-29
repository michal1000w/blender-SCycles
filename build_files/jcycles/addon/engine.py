# SPDX-FileCopyrightText: 2011-2022 Blender Foundation, 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""
The `engine` module that the fork's Cycles UI and property code import.

In the Cycles add-on this wraps the C++ module; in JCycles the answers come from the bundled
fork build. The render pass list below is taken from the fork's engine.py at package time.
"""

from __future__ import annotations

from . import _cycles_shim, jcy_runtime


def with_osl():
    return bool(_cycles_shim.with_osl)


def osl_version():
    return tuple(_cycles_shim.osl_version)


def with_path_guiding():
    return bool(_cycles_shim.with_path_guiding)


def available_devices():
    return _cycles_shim.available_devices()


def system_info():
    return jcy_runtime.system_info()


# @@JCYCLES_GENERATED_PASSES@@
