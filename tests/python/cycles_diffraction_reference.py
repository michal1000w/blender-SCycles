#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Independent TE Fourier-modal reference for a binary surface relief.

Solves the scalar TE Maxwell boundary problem at nonconical incidence, with a
single lamellar layer between homogeneous half spaces. This is a validation
reference, not a production BSDF or an unpolarized/conical solution. All length
units must match. Passive refractive indices use positive imaginary parts and
fields use exp(-i omega t). No implementation from external RCWA code is used.

The layer eigenproblem follows directly from
  d_z^2 E_y + (k0^2 epsilon(x) + d_x^2) E_y = 0.
E_y and d_z E_y are matched at both interfaces. Forward amplitudes are referenced
to the top and backward amplitudes to the bottom to avoid growing exponentials.
Background: Moharam et al., JOSA A 12, 1068 (1995), doi:10.1364/JOSAA.12.001068.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def outgoing_sqrt(value):
    result = np.sqrt(np.asarray(value, dtype=complex))
    return np.where(result.imag < 0, -result, result)


def te_grating(wavelength, pitch, depth, duty, n_ridge, n_groove=1.0,
               n_incident=1.0, n_substrate=1.5, angle=0.0, harmonics=24):
    """Return diffraction efficiencies for orders [-harmonics, harmonics].

    Ridge occupies fraction duty of the period and depth of the relief layer.
    The incident region must be lossless. Bottom flux is into the substrate;
    for absorbing substrates it is ultimately absorbed, not far-field light.
    """
    assert wavelength > 0 and pitch > 0 and depth >= 0 and 0 <= duty <= 1
    assert np.isreal(n_incident) and n_incident > 0 and abs(angle) < np.pi / 2
    orders = np.arange(-harmonics, harmonics + 1)
    kx = n_incident * np.sin(angle) + orders * wavelength / pitch
    difference = orders[:, None] - orders[None, :]
    # Center the ridge at x=0; Fourier coefficients are real before multiplying
    # by complex permittivity contrast. numpy.sinc(x) = sin(pi*x)/(pi*x).
    epsilon = ((complex(n_ridge) ** 2 - complex(n_groove) ** 2) *
               duty * np.sinc(difference * duty))
    epsilon += np.eye(len(orders)) * complex(n_groove) ** 2
    eigenvalues, W = np.linalg.eig(epsilon - np.diag(kx ** 2))
    q = outgoing_sqrt(eigenvalues)
    V = W * q[None, :]
    X = np.exp(2j * np.pi * q * depth / wavelength)
    upper = outgoing_sqrt(n_incident ** 2 - kx ** 2)
    lower = outgoing_sqrt(complex(n_substrate) ** 2 - kx ** 2)
    upper_W, lower_W = upper[:, None] * W, lower[:, None] * W
    matrix = np.block([[V + upper_W, (-V + upper_W) * X[None, :]],
                       [(V - lower_W) * X[None, :], -V - lower_W]])
    incident = np.zeros(len(orders), dtype=complex)
    incident[harmonics] = 1.0
    rhs = np.concatenate((2 * upper * incident, np.zeros_like(incident)))
    A, B = np.split(np.linalg.solve(matrix, rhs), 2)
    reflection = W @ (A + X * B) - incident
    transmission = W @ (X * A + B)
    incident_flux = upper[harmonics].real
    R = np.maximum(upper.real, 0) / incident_flux * abs(reflection) ** 2
    T = np.maximum(lower.real, 0) / incident_flux * abs(transmission) ** 2
    residual = np.linalg.norm(matrix @ np.concatenate((A, B)) - rhs) / np.linalg.norm(rhs)
    return dict(orders=orders, reflection=R, substrate_flux=T,
                absorption=float(1 - R.sum() - T.sum()), boundary_residual=float(residual))


def scalar_reflection(wavelength, pitch, depth, duty, angle, orders):
    """Double-precision independent evaluation of the current closure model.

    The tangential order sign is immaterial to the symmetric centered binary
    profile. This convention matches the RCWA incident wavevector convention.
    """
    sine = np.sin(angle) + orders * wavelength / pitch
    cosine = np.sqrt(np.maximum(1.0 - sine ** 2, 0))
    active = (abs(sine) < 1) & (orders != 0)
    result = np.zeros(len(orders))
    m = orders[active]
    phase = 2 * np.pi * depth / wavelength * (np.cos(angle) + cosine[active])
    result[active] = (2 * np.sin(phase / 2) * np.sin(np.pi * m * duty) / (np.pi * m)) ** 2
    result[orders == 0] = 1 - result.sum()
    return result


def validate():
    errors = []
    # Zero depth must reduce to the planar Fresnel interface regardless of
    # the unused layer profile, including for a lossy substrate.
    for substrate in [1.5, 0.9 + 6j]:
        for angle in [0, 0.45]:
            result = te_grating(550, 1600, 0, 0.37, 2.1, n_substrate=substrate, angle=angle)
            ci = np.cos(angle)
            ct = outgoing_sqrt(substrate ** 2 - np.sin(angle) ** 2)
            expected = abs((ci - ct) / (ci + ct)) ** 2
            error = abs(result['reflection'][24] - expected)
            errors.append(float(error))
            assert error < 1e-10, result
            assert abs(result['absorption']) < 1e-10, result
    # Nontrivial dielectric grating: energy and boundary residual, independent
    # of any scalar grating power formula.
    for angle in [0, 0.3, 0.7]:
        result = te_grating(550, 740, 180, 0.41, 1.8, angle=angle, harmonics=32)
        assert abs(result['absorption']) < 1e-10, result
        assert result['boundary_residual'] < 1e-10, result
    return dict(flat_fresnel_max_error=max(errors), lossless_energy_tests=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nist', action='store_true',
                        help='Cross-check with separately installed NIST pySCATMECH')
    args = parser.parse_args()
    checks = validate()
    cases = []
    # Synthetic conductor, not measured aluminum optical constants. The scalar
    # model is multiplied by the planar reflectance for an explicit comparison.
    metal = 0.9 + 6j
    for pitch in [1600.0, 740.0]:
        for wavelength in [450.0, 550.0, 650.0]:
            angle = 0.0
            convergence = []
            previous = None
            for truncation in [16, 32, 48, 64]:
                reference = te_grating(wavelength, pitch, 150, 0.5, metal,
                                       n_substrate=metal, angle=angle, harmonics=truncation)
                powers = dict(zip(reference['orders'].tolist(), reference['reflection'].tolist()))
                change = None if previous is None else sum(abs(powers.get(m, 0) - p)
                                                           for m, p in previous.items())
                convergence.append(dict(harmonics=truncation, l1_change=change,
                                        total_reflection=float(reference['reflection'].sum()),
                                        boundary_residual=reference['boundary_residual']))
                previous = powers
            scalar = scalar_reflection(wavelength, pitch, 150, 0.5, angle, reference['orders'])
            scalar *= abs((1 - metal) / (1 + metal)) ** 2
            propagating = abs(reference['orders'] * wavelength / pitch) < 1
            case = dict(pitch_nm=pitch, wavelength_nm=wavelength, depth_nm=150, duty=0.5,
                              metal_n=metal.real, metal_k=metal.imag, polarization='TE',
                              convergence=convergence,
                              orders=reference['orders'][propagating].tolist(),
                              reference=reference['reflection'][propagating].tolist(),
                              scalar=scalar[propagating].tolist(),
                              efficiency_l1_error=float(abs(scalar - reference['reflection']).sum()))
            if args.nist:
                from pySCATMECH.rcw import RCW_Model
                from pySCATMECH.mueller import Polarization, Sensitivity
                model = RCW_Model({
                    'order': 64, 'type': 0, 'lambda': wavelength / 1000,
                    'thetai': 0, 'rotation': 0,
                    'grating': {None: 'Single_Line_Grating', 'period': pitch / 1000,
                                'medium_i': 1, 'medium_t': '(0.9,6)', 'material': '(0.9,6)',
                                'space': 1, 'height': 0.15, 'topwidth': pitch / 2000,
                                'bottomwidth': pitch / 2000, 'offset': 0, 'nlevels': 1}})
                matrices = [model.DiffractionEfficiency(int(m)) for m in case['orders']]
                nist_te = np.array([float(Sensitivity('u') @ M @ Polarization('s'))
                                    for M in matrices])
                nist_tm = np.array([float(Sensitivity('u') @ M @ Polarization('p'))
                                    for M in matrices])
                error = float(max(abs(nist_te - np.array(case['reference']))))
                assert error < 1e-6, (pitch, wavelength, error)
                case.update(nist_te=nist_te.tolist(), nist_tm=nist_tm.tolist(),
                            nist_te_max_error=error)
            cases.append(case)
    report = dict(checks=checks, cases=cases, numpy_version=np.__version__,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
