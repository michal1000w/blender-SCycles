#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Vector Fourier-modal validation solver for a binary lamellar grating.

Independent Maxwell transverse-field solve, including conical incidence and
Li inverse factorization for the electric field normal to the groove boundary.
This is a development reference, not production renderer code. Lengths are nm;
angles are radians. Passive complex indices have nonnegative imaginary parts.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cycles_diffraction_reference import outgoing_sqrt


def vector_grating(wavelength, pitch, depth, duty, n_ridge, n_groove=1.0,
                   n_incident=1.0, n_substrate=1.5, angle=0.0, azimuth=0.0, harmonics=24):
    assert 0 <= duty <= 1 and depth >= 0 and pitch > 0 and wavelength > 0
    assert np.isreal(n_incident) and n_incident > 0 and abs(angle) < np.pi / 2
    orders = np.arange(-harmonics, harmonics + 1)
    count = len(orders)
    identity = np.eye(count)
    kx = n_incident * np.sin(angle) * np.cos(azimuth) + orders * wavelength / pitch
    ky = n_incident * np.sin(angle) * np.sin(azimuth)
    K = np.diag(kx)
    difference = orders[:, None] - orders[None, :]
    indicator = duty * np.sinc(difference * duty)
    er, eg = complex(n_ridge) ** 2, complex(n_groove) ** 2
    E = (er - eg) * indicator + eg * identity
    inverse_E = np.linalg.inv(E)
    normal_E = np.linalg.inv((1 / er - 1 / eg) * indicator + identity / eg)
    # d_z [Ex,Ey] = i k0 P [Hx,Hy]; d_z H = i k0 Q E.
    P = np.block([[ky * K @ inverse_E, identity - K @ inverse_E @ K],
                  [ky ** 2 * inverse_E - identity, -ky * inverse_E @ K]])
    Q = np.block([[-ky * K, K @ K - E],
                  [normal_E - ky ** 2 * identity, ky * K]])
    eigenvalues, W = np.linalg.eig(P @ Q)
    q = outgoing_sqrt(eigenvalues)
    V = (Q @ W) / q[None, :]
    X = np.exp(2j * np.pi * q * depth / wavelength)

    def admittance(index):
        e = complex(index) ** 2
        z = outgoing_sqrt(e - kx ** 2 - ky ** 2)
        return np.block([[np.diag(-kx * ky / z), np.diag((kx ** 2 - e) / z)],
                         [np.diag((e - ky ** 2) / z), np.diag(kx * ky / z)]])

    upper, lower = admittance(n_incident), admittance(n_substrate)
    UW, LW = upper @ W, lower @ W
    matrix = np.block([[V + UW, (-V + UW) * X[None, :]],
                       [(V - LW) * X[None, :], -V - LW]])
    incident = np.zeros((2 * count, 2), dtype=complex)
    # Columns are unit electric field s and p incident polarizations.
    incident[harmonics, :] = [-np.sin(azimuth), np.cos(angle) * np.cos(azimuth)]
    incident[count + harmonics, :] = [np.cos(azimuth), np.cos(angle) * np.sin(azimuth)]
    rhs = np.concatenate((2 * upper @ incident, np.zeros_like(incident)), axis=0)
    A, B = np.split(np.linalg.solve(matrix, rhs), 2)
    reflected = W @ (A + X[:, None] * B) - incident
    transmitted = W @ (X[:, None] * A + B)

    def flux(field, Y):
        magnetic = Y @ field
        return np.real(field[:count] * magnetic[count:].conj() -
                       field[count:] * magnetic[:count].conj()) / (n_incident * np.cos(angle))

    R, T = flux(reflected, upper), flux(transmitted, lower)
    return dict(orders=orders, reflection=R, substrate_flux=T,
                reflected_field=reflected, transmitted_field=transmitted,
                absorption=1 - R.sum(axis=0) - T.sum(axis=0),
                boundary_residual=float(np.linalg.norm(matrix @ np.concatenate((A, B)) - rhs) /
                                        np.linalg.norm(rhs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from pySCATMECH.rcw import RCW_Model
    from pySCATMECH.mueller import Polarization, Sensitivity
    cases = []
    for substrate in [1.5, 0.9 + 6j]:
        for angle, azimuth in [(0, 0), (0.4, 0), (0.4, 0.7), (0.9, 1.1)]:
            for depth in [0.0, 150.0]:
                reference = vector_grating(550, 740, depth, 0.41, substrate,
                                           n_substrate=substrate, angle=angle, azimuth=azimuth,
                                           harmonics=24)
                material = f'({complex(substrate).real},{complex(substrate).imag})'
                model = RCW_Model({'order': 24, 'type': 0, 'lambda': 0.55,
                                   'thetai': np.degrees(angle), 'rotation': np.degrees(azimuth),
                                   'grating': {None: 'Single_Line_Grating', 'period': 0.74,
                                               'medium_i': 1, 'medium_t': material,
                                               'material': material, 'space': 1,
                                               'height': depth / 1000, 'topwidth': 0.74 * 0.41,
                                               'bottomwidth': 0.74 * 0.41, 'offset': 0, 'nlevels': 1}})
                nist = []
                for order in reference['orders']:
                    M = model.DiffractionEfficiency(int(order))
                    nist.append([float(Sensitivity('u') @ M @ Polarization(p)) for p in ['s', 'p']])
                nist = np.array(nist)
                # NIST uses the opposite outgoing diffraction-order sign.
                error = float(abs(nist[::-1] - reference['reflection']).max())
                case = dict(n_real=complex(substrate).real, n_imag=complex(substrate).imag,
                            angle=angle, azimuth=azimuth, depth_nm=depth, nist_max_error=error,
                            boundary_residual=reference['boundary_residual'],
                            absorption=reference['absorption'].tolist())
                cases.append(case)
                assert error < 1e-6, case
                assert reference['boundary_residual'] < 1e-10, case
                if np.isreal(substrate) or depth == 0:
                    assert abs(reference['absorption']).max() < 1e-10, case
    medium_errors = []
    for angle, azimuth in [(0, 0), (0.4, 0.7), (0.9, 1.1)]:
        index = 1.58
        material = 0.9 + 6j
        embedded = vector_grating(550, 740, 150, 0.41, material, n_groove=index,
                                  n_incident=index, n_substrate=material,
                                  angle=angle, azimuth=azimuth)
        relative = vector_grating(550 / index, 740, 150, 0.41, material / index,
                                  n_substrate=material / index, angle=angle, azimuth=azimuth)
        error = float(abs(embedded['reflection'] - relative['reflection']).max())
        assert error < 1e-10, (angle, azimuth, error)
        medium_errors.append(error)
    report = dict(cases=cases, incident_medium_similarity_errors=medium_errors,
                  numpy_version=np.__version__,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
