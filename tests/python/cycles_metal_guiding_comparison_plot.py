#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plot raw render comparisons with one shared display transform and CPU reference.

Run with Python, NumPy and Matplotlib. Each --image takes a label and .npy path.
The display transform is x/(1+x) followed by sRGB encoding, at a common exposure.
Error panels use the unmodified linear inputs, transformed identically by log1p.
"""

import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=pathlib.Path, required=True)
    parser.add_argument('--image', nargs=2, action='append', required=True, metavar=('LABEL', 'NPY'))
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--exposure', type=float, default=0)
    parser.add_argument('--crop', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='Apply the same pixel crop to the reference and every comparison')
    args = parser.parse_args()
    reference = np.load(args.reference).astype(float)
    images = [reference] + [np.load(path).astype(float) for _, path in args.image]
    labels = ['CPU guided reference'] + [label for label, _ in args.image]
    assert all(image.shape == reference.shape and np.isfinite(image).all() for image in images)
    if args.crop:
        x0, y0, x1, y1 = args.crop
        if not (0 <= x0 < x1 <= reference.shape[1] and 0 <= y0 < y1 <= reference.shape[0]):
            parser.error('--crop must lie inside the reference image')
        images = [image[y0:y1, x0:x1] for image in images]
        reference = images[0]
    errors = [np.mean(np.abs(np.log1p(np.maximum(image, 0)) -
                             np.log1p(np.maximum(reference, 0))), axis=2) for image in images]
    error_max = max(float(error.max()) for error in errors)
    fig, axes = plt.subplots(2, len(images), figsize=(3.5 * len(images), 7), squeeze=False,
                             layout='constrained')
    for column, (label, image, error) in enumerate(zip(labels, images, errors)):
        rgb = np.maximum(image * 2 ** args.exposure, 0)
        rgb = rgb / (1 + rgb)
        display = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
        axes[0, column].imshow(display, interpolation='nearest')
        axes[0, column].set_title(label)
        heatmap = axes[1, column].imshow(error, cmap='magma', vmin=0, vmax=error_max,
                                        interpolation='nearest')
        axes[1, column].set_title('Mean absolute log1p RGB error')
        for row in range(2):
            axes[row, column].set_axis_off()
    fig.colorbar(heatmap, ax=list(axes[1]), shrink=.6)
    fig.suptitle('Experimental Metal guiding — common display transform; CPU guided reference')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
