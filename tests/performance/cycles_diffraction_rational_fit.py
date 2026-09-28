# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Offline shared-denominator AAA experiment; no production cache integration.

Reference: https://arxiv.org/abs/1612.00337 and matrix extension
https://arxiv.org/html/2602.18414v1 . Accuracy here concerns reference-operator
entries only, not passivity, physical power, angular coverage or modal accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def evaluate(z, nodes, values, weights):
    result = np.empty((len(z), values.shape[1]), dtype=complex)
    for i, x in enumerate(z):
        match = np.flatnonzero(x == nodes)
        if len(match):
            result[i] = values[match[0]]
        else:
            c = weights / (x - nodes)
            c /= np.max(np.abs(c))
            denominator = c.sum()
            if abs(denominator) < 1e-14:
                raise ValueError('Ill-conditioned rational denominator at evaluation point')
            result[i] = c @ values / denominator
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite rational evaluation')
    return result


def fit(z, values, tolerance=1e-8, maximum_support=24, real_weights=False):
    approximation = np.broadcast_to(values.mean(axis=0), values.shape).copy()
    support = []
    for _ in range(min(maximum_support, len(z) - 1)):
        errors = np.linalg.norm(values - approximation, axis=1)
        errors[support] = -1
        support.append(int(np.argmax(errors)))
        remaining = np.ones(len(z), dtype=bool)
        remaining[support] = False
        cauchy = 1 / (z[remaining, None] - z[support][None, :])
        loewner = ((values[remaining, :, None] - values[support].T[None, :, :]) *
                   cauchy[:, None, :]).reshape(-1, len(support))
        if real_weights:
            loewner = np.concatenate((loewner.real, loewner.imag), axis=0)
        _, _, vh = np.linalg.svd(loewner, full_matrices=False)
        weights = vh[-1].conj()
        approximation = evaluate(z, z[support], values[support], weights)
        if np.max(np.linalg.norm(values - approximation, axis=1)) <= tolerance:
            break
    return z[support], values[support], weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--precision', choices=['float64', 'float32'], default='float64')
    args = parser.parse_args()
    # Independent analytic shared-pole matrix-vector control, including support endpoints.
    z = np.linspace(-1, 1, 61)
    reference = lambda x: np.column_stack((1 / (x - .2j), 2 + .3 / (x - .2j)))
    model = fit(z, reference(z), tolerance=1e-10)
    assert np.max(np.abs(evaluate(z + .001, *model) - reference(z + .001))) < 1e-9
    dataset = json.loads(args.input.read_text())
    samples = dataset['samples']
    wave = np.array([s.get('coordinate', s['wavelength_nm']) for s in samples])
    # Affine normalization improves conditioning without changing rational degree.
    x = (wave - wave.mean()) / np.ptp(wave)
    raw = np.array([s['matrix'] for s in samples])
    values = raw[..., 0] + 1j * raw[..., 1]
    training = np.array([s['fitting_sample'] for s in samples])
    assert training.any() and (~training).any() and np.isfinite(values).all()
    model = fit(x[training], values[training])
    if args.precision == 'float32':
        model = (model[0].astype(np.float32), model[1].astype(np.complex64),
                 model[2].astype(np.complex64))
        evaluation_x = x.astype(np.float32)
    else:
        evaluation_x = x
    errors = np.linalg.norm(evaluate(evaluation_x, *model) - values, axis=1)
    degree = len(model[0]) - 1
    coefficients = np.polynomial.chebyshev.chebfit(x[training], values[training], degree)
    polynomial = np.polynomial.chebyshev.chebvander(x, degree) @ coefficients
    polynomial_errors = np.linalg.norm(polynomial - values, axis=1)
    report = dict(scope=__doc__, evaluation_precision=args.precision, polynomial_degree=degree,
                  polynomial_held_out_frobenius_error=float(polynomial_errors[~training].max()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  fitting_samples=int(training.sum()), held_out_samples=int((~training).sum()),
                  support_count=len(model[0]), maximum_fitting_frobenius_error=float(errors[training].max()),
                  maximum_held_out_frobenius_error=float(errors[~training].max()),
                  physical_power_validated=False, production_integrated=False)
    if args.predictions is not None:
        predicted = evaluate(evaluation_x[~training], *model)
        with args.predictions.open('w') as stream:
            stream.write(str(len(predicted)) + '\n')
            for sample, matrix in zip([s for s in samples if not s['fitting_sample']], predicted):
                b = sample['coordinate'] if dataset.get('axis') == 'bloch' else dataset['bloch']
                ky = sample['coordinate'] if dataset.get('axis') == 'ky' else dataset['ky']
                stream.write(f"{sample['wavelength_nm']:.17g} {b:.17g} {ky:.17g} {len(matrix)}\n")
                for value in matrix:
                    stream.write(f'{value.real:.17g} {value.imag:.17g}\n')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
