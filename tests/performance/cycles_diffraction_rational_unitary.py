# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Lossless-only Cayley/real-weight rational experiment, outside production."""
import argparse
import json
from pathlib import Path
import numpy as np
from cycles_diffraction_rational_fit import fit, evaluate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--predictions', type=Path, required=True)
parser.add_argument('--precision', choices=['float64', 'float32'], default='float64')
args = parser.parse_args()
data = json.loads(args.input.read_text())
samples = data['samples']
raw = np.asarray([s['matrix'] for s in samples])
values = raw[..., 0] + 1j * raw[..., 1]
n = int(np.sqrt(values.shape[1]))
matrices = values.reshape(-1, n, n)
identity = np.eye(n)
unitarity = max(np.linalg.norm(m.conj().T @ m - identity) for m in matrices)
if unitarity > 1e-8:
    raise ValueError('This parameterization requires lossless unitary reference operators')
training = np.asarray([s['fitting_sample'] for s in samples])
x = np.asarray([s['coordinate'] for s in samples])
x = (x - x.mean()) / np.ptp(x)
# Select a single chart from training data only. Never optimize on held-out data.
phases = np.exp(2j * np.pi * np.arange(32) / 32)
condition = [max(np.linalg.cond(identity + m / phase) for m in matrices[training])
             for phase in phases]
phase = phases[int(np.argmin(condition))]
if min(condition) > 1e6:
    raise ValueError('No well-conditioned Cayley chart found')
hermitian = []
for m in matrices[training]:
    u = m / phase
    hermitian.append(-1j * np.linalg.solve((identity + u).T, (identity - u).T).T)
hermitian = np.asarray(hermitian)
# Symmetrize only solver roundoff, recording its magnitude explicitly.
projection = (hermitian + hermitian.conj().transpose(0, 2, 1)) / 2
projection_error = float(np.max(np.linalg.norm(projection - hermitian, axis=(1, 2))))
if projection_error > 1e-8:
    raise ValueError('Non-Hermitian Cayley operator exceeds roundoff allowance')
model = fit(x[training], projection.reshape(-1, n*n), real_weights=True)
if args.precision == 'float32':
    model = (model[0].astype(np.float32), model[1].astype(np.complex64),
             model[2].astype(np.float32))
    coordinates = x[~training].astype(np.float32)
    identity = identity.astype(np.complex64)
    phase = np.complex64(phase)
    h = evaluate(coordinates, *model).astype(np.complex64).reshape(-1, n, n)
else:
    h = evaluate(x[~training], *model).reshape(-1, n, n)
predicted = np.asarray([phase * np.linalg.solve((identity + 1j*m).T,
                                               (identity - 1j*m).T).T for m in h])
errors = np.linalg.norm(predicted - matrices[~training], axis=(1, 2))
with args.predictions.open('w') as stream:
    stream.write(str(len(predicted)) + '\n')
    for sample, matrix in zip([s for s in samples if not s['fitting_sample']], predicted):
        b = sample['coordinate'] if data['axis'] == 'bloch' else data['bloch']
        ky = sample['coordinate'] if data['axis'] == 'ky' else data['ky']
        stream.write(f"{sample['wavelength_nm']:.17g} {b:.17g} {ky:.17g} {n*n}\n")
        for v in matrix.flat:
            stream.write(f'{v.real:.17g} {v.imag:.17g}\n')
report = dict(scope=__doc__, evaluation_precision=args.precision, support_count=len(model[0]), training_chart_condition=min(condition),
              input_unitarity_residual=unitarity, hermitian_roundoff_projection=projection_error,
              maximum_held_out_frobenius_error=float(errors.max()),
              maximum_predicted_unitarity_residual=max(float(np.linalg.norm(m.conj().T@m-identity))
                                                       for m in predicted))
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
