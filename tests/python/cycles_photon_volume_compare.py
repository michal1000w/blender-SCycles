#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare raw RGB pixels from two render results inside Blender."""

import argparse
import math
import sys

import bpy


def main():
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("test")
    options = parser.parse_args(arguments)

    reference = list(bpy.data.images.load(options.reference, check_existing=False).pixels)
    test = list(bpy.data.images.load(options.test, check_existing=False).pixels)
    if len(reference) != len(test):
        raise RuntimeError("Image dimensions do not match")

    differences = [
        reference[index] - test[index] for index in range(len(reference)) if index % 4 != 3
    ]
    rmse = math.sqrt(sum(value * value for value in differences) / len(differences))
    print(
        "PHOTON_VOLUME_COMPARE "
        f"rmse={rmse:.9g} max_abs={max(abs(value) for value in differences):.9g}"
    )


if __name__ == "__main__":
    main()
