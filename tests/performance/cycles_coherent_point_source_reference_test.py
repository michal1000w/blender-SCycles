#!/usr/bin/env python3
"""Independent scalar oracle for direct coherent point-source illumination.

This covers direct point-source interference only. It does not model general
coherent multipath transport, surfaces, polarization, or any renderer output.
Run with: python3 tests/performance/cycles_coherent_point_source_reference_test.py
"""

import math
import unittest

import numpy as np


def ranges(sources, detector):
    """Euclidean source-to-detector distances in a shared 2D coordinate frame."""
    point = np.asarray(detector, dtype=float)
    positions = np.asarray([source["position"] for source in sources], dtype=float)
    return np.linalg.norm(positions - point, axis=1)


def pairwise_intensity(sources, detector, wavelength, coherence_length=math.inf,
                       visibility=None):
    """Gaussian-coherence scalar irradiance with opaque per-source visibility.

    A source has ``position=(x,z)``, nonnegative ``power``, and ``phase`` in
    radians. Its field amplitude is sqrt(power)/r and its phase is
    2*pi*r/wavelength + phase. ``coherence_length`` is 1/sigma_k for a
    Gaussian distribution of wavenumbers. Zero denotes the exact incoherent
    limit; infinity denotes full coherence.
    """
    if wavelength <= 0 or coherence_length < 0:
        raise ValueError("wavelength must be positive and coherence length nonnegative")
    r = ranges(sources, detector)
    if np.any(r <= 0):
        raise ValueError("detector cannot coincide with a point source")
    powers = np.asarray([source["power"] for source in sources], dtype=float)
    phases = np.asarray([source["phase"] for source in sources], dtype=float)
    if np.any(powers < 0):
        raise ValueError("source powers must be nonnegative")
    visible = np.ones(len(sources), dtype=bool) if visibility is None else np.asarray(visibility, dtype=bool)
    if visible.shape != (len(sources),):
        raise ValueError("visibility must have one boolean per source")
    amplitudes = np.sqrt(powers) / r * visible
    optical_phases = 2 * np.pi * r / wavelength + phases
    if coherence_length == 0:
        return float(np.dot(amplitudes, amplitudes))

    path_delta = r[:, None] - r[None, :]
    if math.isinf(coherence_length):
        coherence = np.ones_like(path_delta)
    else:
        coherence = np.exp(-0.5 * (path_delta / coherence_length) ** 2)
    phase_delta = optical_phases[:, None] - optical_phases[None, :]
    return float(np.sum(amplitudes[:, None] * amplitudes[None, :] *
                        coherence * np.cos(phase_delta)))


def complex_field_intensity(sources, detector, wavelength, visibility=None):
    """Full-coherence oracle by direct complex field summation."""
    r = ranges(sources, detector)
    powers = np.asarray([source["power"] for source in sources], dtype=float)
    phases = np.asarray([source["phase"] for source in sources], dtype=float)
    visible = np.ones(len(sources), dtype=bool) if visibility is None else np.asarray(visibility, dtype=bool)
    field = np.sum((np.sqrt(powers) / r * visible) *
                   np.exp(1j * (2 * np.pi * r / wavelength + phases)))
    return float(abs(field) ** 2)


def gaussian_spectrum_intensity(sources, detector, wavelength, coherence_length,
                                visibility=None, quadrature_order=96):
    """Independent Hermite quadrature over Gaussian-distributed wavenumber."""
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    k_mean = 2 * np.pi / wavelength
    k_values = k_mean + np.sqrt(2) * nodes / coherence_length
    r = ranges(sources, detector)
    powers = np.asarray([source["power"] for source in sources], dtype=float)
    phases = np.asarray([source["phase"] for source in sources], dtype=float)
    visible = np.ones(len(sources), dtype=bool) if visibility is None else np.asarray(visibility, dtype=bool)
    source_amplitudes = np.sqrt(powers) / r * visible
    fields = np.sum(source_amplitudes[None, :] * np.exp(
        1j * (k_values[:, None] * r[None, :] + phases[None, :])), axis=1)
    return float(np.dot(weights, np.abs(fields) ** 2) / np.sqrt(np.pi))


def exact_direct_estimator_expectation(source_intensities, probabilities,
                                       mis_weights, correction_scale,
                                       multiply_correction_by_mis=False,
                                       clamp_negative_correction=False):
    """Enumerate a light-sampled baseline-MIS estimate and direct correction.

    ``source_intensities`` are the incoherent direct contributions c_i. The
    light-sampled part has baseline outcomes c_i * MIS_i / p_i; the matching
    BSDF-sampled direct part contributes its exact complement c_i*(1-MIS_i).
    Thus their sum is the ordinary incoherent direct estimate, independent of
    how the baseline is split by MIS. The signed coherence correction is
    c_i/p_i * (Icoh/Iinc - 1), whose expectation is Icoh-Iinc. It is deliberately
    outside the baseline MIS weighting. An arbitrary indirect term is added
    separately by the caller.

    The optional switches model two incorrect alternatives for counterexamples.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    mis_weights = np.asarray(mis_weights, dtype=float)
    source_intensities = np.asarray(source_intensities, dtype=float)
    if not (probabilities.shape == mis_weights.shape == source_intensities.shape):
        raise ValueError("probabilities, MIS weights, and intensities must align")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("source probabilities must sum to one")
    if np.any(probabilities <= 0) or np.any((mis_weights < 0) | (mis_weights > 1)):
        raise ValueError("probabilities must be positive and MIS weights in [0,1]")

    baseline_nee = np.sum(probabilities * source_intensities * mis_weights / probabilities)
    baseline_bsdf = np.sum(source_intensities * (1.0 - mis_weights))
    correction_factor = correction_scale
    if multiply_correction_by_mis:
        correction_factor = correction_factor * mis_weights
    correction = source_intensities / probabilities * correction_factor
    if clamp_negative_correction:
        correction = np.maximum(correction, 0.0)
    correction_expectation = np.sum(probabilities * correction)
    return float(baseline_nee + baseline_bsdf + correction_expectation)


class CoherentPointSourceReferenceTest(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"position": (-0.8, 2.1), "power": 1.2, "phase": 0.3},
            {"position": (0.15, 1.4), "power": 0.7, "phase": -0.8},
            {"position": (1.1, 2.5), "power": 1.8, "phase": 1.2},
        ]
        self.detector = (0.2, 0.0)
        self.wavelength = 0.37

    def test_pairwise_matches_complex_field_when_fully_coherent(self):
        for x in np.linspace(-0.7, 0.9, 17):
            detector = (float(x), self.detector[1])
            self.assertAlmostEqual(
                pairwise_intensity(self.sources, detector, self.wavelength),
                complex_field_intensity(self.sources, detector, self.wavelength),
                places=12,
            )

    def test_partial_coherence_matches_independent_spectral_quadrature(self):
        for length in (0.8, 1.7, 3.2):
            expected = pairwise_intensity(
                self.sources, self.detector, self.wavelength, length
            )
            measured = gaussian_spectrum_intensity(
                self.sources, self.detector, self.wavelength, length
            )
            self.assertAlmostEqual(expected, measured, delta=2e-12)

    def test_incoherent_limit_is_inverse_square_power_sum(self):
        expected = sum(source["power"] / distance**2 for source, distance in
                       zip(self.sources, ranges(self.sources, self.detector)))
        self.assertAlmostEqual(
            pairwise_intensity(self.sources, self.detector, self.wavelength, 0),
            expected,
            places=14,
        )
        # A small but nonzero Gaussian coherence length approaches the same
        # result for this fixture, whose source path lengths are all distinct.
        self.assertAlmostEqual(
            pairwise_intensity(self.sources, self.detector, self.wavelength, 1e-5),
            expected,
            places=14,
        )

    def test_visibility_is_an_opaque_per_source_mask(self):
        mask = (True, False, True)
        masked_sources = [self.sources[i] for i, visible in enumerate(mask) if visible]
        self.assertAlmostEqual(
            pairwise_intensity(self.sources, self.detector, self.wavelength,
                               visibility=mask),
            pairwise_intensity(masked_sources, self.detector, self.wavelength),
            places=14,
        )
        self.assertEqual(
            pairwise_intensity(self.sources, self.detector, self.wavelength,
                               visibility=(False, False, False)),
            0.0,
        )

    def test_source_permutation_does_not_change_result(self):
        expected = pairwise_intensity(
            self.sources, self.detector, self.wavelength, 1.3,
            visibility=(True, False, True),
        )
        order = (2, 0, 1)
        permuted = [self.sources[i] for i in order]
        mask = tuple((True, False, True)[i] for i in order)
        self.assertAlmostEqual(
            pairwise_intensity(permuted, self.detector, self.wavelength, 1.3,
                               visibility=mask),
            expected,
            places=14,
        )

    def test_gaussian_coherence_kernel_is_positive_semidefinite(self):
        path_lengths = np.array([0.2, 0.7, 1.9, 2.3, 5.1])
        for length in (0.05, 0.4, 2.0, 1e6):
            delta = path_lengths[:, None] - path_lengths[None, :]
            kernel = np.exp(-0.5 * (delta / length) ** 2)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(kernel).min()), -1e-12)
            # Phase modulation is a diagonal unitary congruence, preserving PSD.
            phase = np.exp(1j * np.array([0.1, -1.0, 0.7, 2.4, -0.5]))
            modulated = phase[:, None] * kernel * phase.conj()[None, :]
            self.assertGreaterEqual(float(np.linalg.eigvalsh(modulated).min()), -1e-12)

    def test_common_translation_and_unit_rescaling_are_stable(self):
        base = pairwise_intensity(self.sources, self.detector, self.wavelength, 1.7)
        shift = np.array([123.4, -56.7])
        moved_sources = [dict(source, position=tuple(np.asarray(source["position"]) + shift))
                         for source in self.sources]
        moved_detector = tuple(np.asarray(self.detector) + shift)
        self.assertAlmostEqual(
            pairwise_intensity(moved_sources, moved_detector, self.wavelength, 1.7),
            base,
            places=13,
        )
        scale = 1000.0
        scaled_sources = [dict(source, position=tuple(scale * np.asarray(source["position"])))
                          for source in self.sources]
        scaled_detector = tuple(scale * np.asarray(self.detector))
        scaled = pairwise_intensity(scaled_sources, scaled_detector,
                                    scale * self.wavelength, scale * 1.7)
        self.assertAlmostEqual(scaled * scale**2, base, places=12)

    def test_three_source_phase_cancellation(self):
        # Equal-distance sources have equal propagation phase. Source phases
        # spaced by 2*pi/3 sum to zero as complex amplitudes.
        phases = (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
        positions = ((1.0, 0.0), (-0.5, np.sqrt(3) / 2), (-0.5, -np.sqrt(3) / 2))
        sources = [dict(position=position, power=1.0, phase=float(phase))
                   for position, phase in zip(positions, phases)]
        result = pairwise_intensity(sources, (0.0, 0.0), self.wavelength)
        self.assertLess(abs(result), 1e-12)
        self.assertAlmostEqual(
            result,
            complex_field_intensity(sources, (0.0, 0.0), self.wavelength),
            delta=1e-12,
        )

    def test_intensity_is_nonnegative_for_partial_coherence(self):
        for length in (0.05, 0.5, 2.0, math.inf):
            for x in np.linspace(-2, 2, 51):
                value = pairwise_intensity(self.sources, (float(x), 0),
                                           self.wavelength, length)
                self.assertGreaterEqual(value, -1e-12)

    def test_exact_estimator_enumeration_restores_coherent_direct_term(self):
        # Equal-distance, equal-amplitude fields cancel exactly. The sampled
        # source probabilities and baseline MIS weights are intentionally
        # unequal, as is common when lights have different sampling weights.
        phases = (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
        positions = ((1.0, 0.0), (-0.5, np.sqrt(3) / 2), (-0.5, -np.sqrt(3) / 2))
        sources = [dict(position=position, power=1.0, phase=float(phase))
                   for position, phase in zip(positions, phases)]
        incoherent = np.asarray([
            source["power"] / ranges([source], (0.0, 0.0))[0] ** 2
            for source in sources
        ])
        incoherent_total = float(incoherent.sum())
        coherent_total = complex_field_intensity(sources, (0.0, 0.0), self.wavelength)
        probabilities = (0.7, 0.2, 0.1)
        mis_weights = (0.15, 0.55, 0.9)
        correction_scale = coherent_total / incoherent_total - 1.0
        self.assertLess(correction_scale, 0.0)

        # Enumerating every source draw exactly yields coherent direct
        # irradiance plus an arbitrary indirect baseline, with baseline MIS
        # unchanged and source probabilities canceled by inverse-PDF weights.
        indirect_baseline = 0.37
        estimate = exact_direct_estimator_expectation(
            incoherent, probabilities, mis_weights, correction_scale
        ) + indirect_baseline
        self.assertAlmostEqual(estimate, coherent_total + indirect_baseline, places=14)

        mis_weighted = exact_direct_estimator_expectation(
            incoherent, probabilities, mis_weights, correction_scale,
            multiply_correction_by_mis=True,
        )
        self.assertGreater(abs(mis_weighted - coherent_total), 0.1)

        clamped = exact_direct_estimator_expectation(
            incoherent, probabilities, mis_weights, correction_scale,
            clamp_negative_correction=True,
        )
        self.assertAlmostEqual(clamped, incoherent_total, places=14)
        self.assertGreater(abs(clamped - coherent_total), 0.1)

    def test_exact_estimator_single_source_and_incoherent_limit_are_baselines(self):
        indirect_baseline = 0.271
        for sources, length in (
            ([{"position": (0.4, 1.7), "power": 2.3, "phase": 1.1}], math.inf),
            (self.sources, 0.0),
        ):
            incoherent = np.asarray([
                source["power"] / distance**2
                for source, distance in zip(sources, ranges(sources, self.detector))
            ])
            target = pairwise_intensity(sources, self.detector, self.wavelength, length)
            scale = target / float(incoherent.sum()) - 1.0
            probabilities = np.linspace(1.0, 2.0, len(sources))
            probabilities /= probabilities.sum()
            estimate = exact_direct_estimator_expectation(
                incoherent, probabilities,
                np.linspace(0.2, 0.8, len(sources)), scale,
            ) + indirect_baseline
            self.assertAlmostEqual(estimate, target + indirect_baseline, places=13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
