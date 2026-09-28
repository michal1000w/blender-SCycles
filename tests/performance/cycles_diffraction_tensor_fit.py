# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""3D lossless Cayley interpolation experiment, not a production cache.

Tensor Floater-Hormann weights follow Floater and Hormann (2007):
https://www.inf.usi.ch/hormann/papers/Floater.2007.BRI.pdf
Real interpolation coefficients preserve Hermitian structure in exact arithmetic.
This does not guarantee interpolation accuracy or float32 passivity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def weights(nodes, degree):
    result = np.zeros(len(nodes))
    for start in range(len(nodes) - degree):
        for k in range(start, start + degree + 1):
            others = [j for j in range(start, start + degree + 1) if j != k]
            result[k] += (-1.)**start / np.prod(nodes[k] - nodes[others])
    return result / np.max(np.abs(result))


def basis(x, nodes, w):
    exact = np.flatnonzero(x == nodes)
    if len(exact):
        result = np.zeros_like(nodes)
        result[exact[0]] = 1
        return result
    # Scaling all barycentric terms by the nearest distance leaves the ratio
    # unchanged and avoids overflow one representable float away from a knot.
    distances = x - nodes
    nearest = distances[np.argmin(np.abs(distances))]
    terms = w * (nearest / distances)
    return terms / np.sum(terms)


def pack_hermitian(matrices):
    """Real isometry: Euclidean packed norm equals matrix Frobenius norm."""
    n = matrices.shape[-1]
    upper = np.triu_indices(n, 1)
    return np.concatenate((matrices.diagonal(axis1=-2, axis2=-1).real,
                           np.sqrt(2) * matrices[..., upper[0], upper[1]].real,
                           np.sqrt(2) * matrices[..., upper[0], upper[1]].imag), axis=-1)


def unpack_hermitian(packed, n):
    upper = np.triu_indices(n, 1)
    count = len(upper[0])
    dtype = np.complex64 if packed.dtype == np.float32 else np.complex128
    result = np.zeros(packed.shape[:-1] + (n, n), dtype=dtype)
    result[..., np.arange(n), np.arange(n)] = packed[..., :n]
    result[..., upper[0], upper[1]] = (packed[..., n:n+count] +
                                      1j * packed[..., n+count:]) / np.sqrt(2)
    result[..., upper[1], upper[0]] = result[..., upper[0], upper[1]].conj()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='Export the compressed model for device experiments')
    parser.add_argument('--precision', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--compression-tolerance', type=float, default=0,
                        help='Maximum training Hermitian Frobenius residual; zero disables SVD')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    side = data['grid_side']
    if not 0 <= args.degree < side:
        raise ValueError('Degree must be smaller than grid side')
    samples = data['samples']
    mask = np.asarray([s['fitting_sample'] for s in samples], dtype=bool)
    raw = np.asarray([s['matrix'] for s in samples])
    n = int(np.sqrt(raw.shape[1]))
    matrices = (raw[..., 0] + 1j * raw[..., 1]).reshape(-1, n, n)
    identity = np.eye(n)
    unitary_error = float(np.max(np.linalg.norm(
        matrices.conj().transpose(0, 2, 1) @ matrices - identity, axis=(1, 2))))
    if unitary_error > 1e-8:
        raise ValueError('Requires unitary lossless reference operators')
    query = np.asarray([s['query'] for s in samples])
    normalized = (query - data['lower']) / (np.asarray(data['upper']) - data['lower'])
    nodes = np.linspace(0, 1, side)
    expected = np.asarray([[x, y, z] for z in nodes for y in nodes for x in nodes])
    if mask.sum() != side**3 or not np.allclose(normalized[mask], expected, atol=1e-13, rtol=0):
        raise ValueError('Training grid ordering or coordinates are invalid')
    phases = np.exp(2j * np.pi * np.arange(32) / 32)
    conditions = [max(np.linalg.cond(identity + m / phase) for m in matrices[mask])
                  for phase in phases]
    phase = phases[int(np.argmin(conditions))]
    if min(conditions) > 1e6:
        raise ValueError('Training domain requires multiple Cayley charts')
    u = matrices[mask] / phase
    h = -1j * np.linalg.solve((identity + u).transpose(0, 2, 1),
                              (identity - u).transpose(0, 2, 1)).transpose(0, 2, 1)
    projected = (h + h.conj().transpose(0, 2, 1)) / 2
    projection_error = float(np.max(np.linalg.norm(projected - h, axis=(1, 2))))
    if projection_error > 1e-8:
        raise ValueError('Hermitian projection exceeds roundoff allowance')
    dtype = np.float32 if args.precision == 'float32' else np.float64
    complex_dtype = np.complex64 if args.precision == 'float32' else np.complex128
    w = weights(nodes, args.degree).astype(dtype)
    nodes = nodes.astype(dtype)
    grid = projected.astype(complex_dtype).reshape(side, side, side, n, n)
    compression = None
    if args.compression_tolerance < 0 or not np.isfinite(args.compression_tolerance):
        raise ValueError('Compression tolerance must be finite and nonnegative')
    if args.model and not args.compression_tolerance:
        raise ValueError('Model export requires compression')
    if args.compression_tolerance:
        packed = pack_hermitian(projected)
        mean = packed.mean(axis=0)
        centered = packed - mean
        left, singular, right = np.linalg.svd(centered, full_matrices=False)
        coefficients = left * singular
        # Select rank from training residuals only, before consulting validation data.
        for rank in range(len(singular) + 1):
            residual = np.max(np.linalg.norm(
                centered - coefficients[:, :rank] @ right[:rank], axis=1))
            if residual <= args.compression_tolerance:
                break
        coefficients = coefficients[:, :rank].astype(dtype).reshape(side, side, side, rank)
        directions = right[:rank].astype(dtype)
        mean = mean.astype(dtype)
        compression = dict(rank=rank, training_tolerance=args.compression_tolerance,
                           maximum_training_residual=float(residual),
                           compressed_real_scalars=int(mean.size + directions.size + coefficients.size),
                           uncompressed_hermitian_real_scalars=int(packed.size))
    identity = identity.astype(complex_dtype)
    phase = complex_dtype(phase)
    if args.model:
        np.savez(args.model, coefficients=coefficients, directions=directions, mean=mean,
                 nodes=nodes, weights=w, phase=phase, matrix_size=n,
                 lower=np.asarray(data['lower'], dtype=dtype),
                 upper=np.asarray(data['upper'], dtype=dtype))
    predicted = []
    amplification = []
    for point in normalized[~mask].astype(dtype):
        bx, by, bz = [basis(point[a], nodes, w) for a in range(3)]
        amplification.append(float(np.abs(bx).sum() * np.abs(by).sum() * np.abs(bz).sum()))
        if compression is None:
            interpolated = np.einsum('i,j,k,ijkab->ab', bz, by, bx, grid, optimize=True)
        else:
            c = np.einsum('i,j,k,ijkr->r', bz, by, bx, coefficients, optimize=True)
            interpolated = unpack_hermitian(mean + c @ directions, n)
        predicted.append(phase * np.linalg.solve((identity + 1j*interpolated).T,
                                                 (identity - 1j*interpolated).T).T)
    predicted = np.asarray(predicted)
    if not np.isfinite(predicted).all():
        raise ValueError('Nonfinite prediction')
    errors = np.linalg.norm(predicted - matrices[~mask], axis=(1, 2))
    with args.predictions.open('w') as stream:
        stream.write(f'{len(predicted)}\n')
        for point, matrix in zip(query[~mask], predicted):
            stream.write(f'{point[2]:.17g} {point[0]:.17g} {point[1]:.17g} {n*n}\n')
            for v in matrix.flat:
                stream.write(f'{v.real:.17g} {v.imag:.17g}\n')
    report = dict(scope=__doc__, precision=args.precision, degree=args.degree,
                  domain_lower=data['lower'], domain_upper=data['upper'],
                  compression=compression,
                  training_count=int(mask.sum()), held_out_count=int((~mask).sum()),
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  training_chart_condition=min(conditions), input_unitarity_residual=unitary_error,
                  hermitian_roundoff_projection=projection_error,
                  maximum_held_out_frobenius_error=float(errors.max()),
                  median_held_out_frobenius_error=float(np.median(errors)),
                  maximum_coefficient_amplification=max(amplification),
                  maximum_predicted_unitarity_residual=float(np.max(np.linalg.norm(
                      predicted.conj().transpose(0, 2, 1) @ predicted - identity, axis=(1, 2)))))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
