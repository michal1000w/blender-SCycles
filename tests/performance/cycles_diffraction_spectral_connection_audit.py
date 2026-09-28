# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Quadrature audit of the existing finite-bandwidth spectral connection kernel.

This is an analytic spectral-response diagnostic, not a rendered grating test.
The light-wavelength sampling density cancels its inverse-density weight, so
integrating the normalized kernel against a response gives its expectation.
"""
import json
import numpy as np


def expectation(camera_nm, response, points=400001):
    wavelength = np.linspace(380.0, 780.0, points)
    x = (wavelength - camera_nm) / 20.0
    kernel = np.maximum(0.0, .75 * (1.0 - x*x)) / 20.0
    return float(np.trapezoid(kernel * response(wavelength), wavelength))


def main():
    records = []
    for camera in (380.0, 385.0, 400.0, 550.0, 760.0, 775.0, 780.0):
        records.append(dict(response='constant', camera_nm=camera, exact=1.0,
                            reconstructed=expectation(camera, lambda x: np.ones_like(x))))
    for sigma in (1.0, 5.0, 20.0):
        response = lambda x: np.exp(-.5 * ((x - 550.0) / sigma)**2)
        a = expectation(550, response)
        b = expectation(550, response, 200001)
        assert abs(a-b) < 1e-8
        records.append(dict(response='gaussian', sigma_nm=sigma, camera_nm=550,
                            exact=1.0, reconstructed=a))
    assert abs(records[0]['reconstructed']-.5) < 1e-8
    assert abs(records[3]['reconstructed']-1) < 1e-8
    print(json.dumps(dict(scope=__doc__, bandwidth_nm=20, records=records), indent=2))


if __name__ == '__main__':
    main()
