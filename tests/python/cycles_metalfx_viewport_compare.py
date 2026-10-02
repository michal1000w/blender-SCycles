# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare viewport captures of cycles_metalfx_viewport.py with a reference capture.

  python3 tests/python/cycles_metalfx_viewport_compare.py REFERENCE.png CAPTURE.png [...]

Needs numpy and Pillow, run with a regular Python. Reports the PSNR of the display values, the
ratio of mean brightness and the ratio of mean gradient magnitude ("edge", below 1.0 the capture
is blurrier than the reference, above it is noisier or has sharper edges).
"""

import sys

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def gradient(image):
    return np.abs(np.diff(image, axis=1)).mean() + np.abs(np.diff(image, axis=0)).mean()


def main():
    reference = load(sys.argv[1])
    # The corners of the viewport hold small interface buttons.
    margin = 48
    crop = (slice(margin, -margin), slice(margin, -margin))
    print("%-36s %8s %7s %6s" % ("capture", "PSNR", "mean", "edge"))
    for path in sys.argv[2:]:
        image = load(path)
        if image.shape != reference.shape:
            print("%-36s size differs: %s vs %s" % (path.split("/")[-1], image.shape, reference.shape))
            continue
        a = image[crop]
        b = reference[crop]
        mse = np.mean((a - b) ** 2)
        psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")
        print("%-36s %8.2f %7.4f %6.3f" % (
            path.split("/")[-1], psnr, a.mean() / b.mean(), gradient(a) / gradient(b)))


if __name__ == "__main__":
    main()
