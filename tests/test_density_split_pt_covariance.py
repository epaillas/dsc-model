"""Tests for the perturbative Gaussian-threshold covariance inputs."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.density_split_pt_covariance import (
    bin_periodic_multipoles,
    gaussian_quantile_correlation,
    gaussian_quantile_hermite_coefficients,
    linear_gaussian_threshold_rsd_spectra,
)
from scripts.validate_density_split_covariance import (
    smooth_calibration_spectra,
)


class DensitySplitPTCovarianceTest(unittest.TestCase):
    def test_hermite_coefficients_obey_partition_and_reflection(self) -> None:
        coefficients = gaussian_quantile_hermite_coefficients(5, 4)
        np.testing.assert_allclose(coefficients[:, 1:].sum(axis=0), 0.0, atol=1e-13)
        for order in range(1, 5):
            parity = -1.0 if order % 2 else 1.0
            np.testing.assert_allclose(
                coefficients[:, order],
                parity * coefficients[::-1, order],
                atol=1e-13,
            )

    def test_exact_threshold_correlation_endpoints_and_linear_limit(self) -> None:
        np.testing.assert_allclose(
            gaussian_quantile_correlation(0.0, 0, 0, ngrid=1001),
            0.0,
            atol=1e-14,
        )
        self.assertEqual(
            float(gaussian_quantile_correlation(1.0, 0, 0, ngrid=1001)),
            4.0,
        )
        self.assertEqual(
            float(gaussian_quantile_correlation(1.0, 0, 1, ngrid=1001)),
            -1.0,
        )
        self.assertEqual(
            float(gaussian_quantile_correlation(-1.0, 0, 4, ngrid=1001)),
            4.0,
        )
        coefficients = gaussian_quantile_hermite_coefficients(5, 1)
        rho = 1e-3
        correlation = gaussian_quantile_correlation(rho, 0, 1, ngrid=20001)
        expected = coefficients[0, 1] * coefficients[1, 1] * rho
        self.assertAlmostEqual(float(correlation), float(expected), delta=2e-6)

    def test_periodic_binning_checks_stored_mode_counts(self) -> None:
        modes = np.fft.fftfreq(8) * 8 * 2.0 * np.pi / 1000.0
        kx = modes[:, None, None]
        ky = modes[None, :, None]
        kz = modes[None, None, :]
        kmagnitude = np.sqrt(kx**2 + ky**2 + kz**2)
        mu = np.divide(
            kz,
            kmagnitude,
            out=np.zeros_like(kmagnitude),
            where=kmagnitude > 0.0,
        )
        edges = np.array([[0.005, 0.015], [0.015, 0.025]])
        counts = np.array(
            [
                np.count_nonzero((kmagnitude >= lower) & (kmagnitude < upper))
                for lower, upper in edges
            ]
        )
        poles = bin_periodic_multipoles(
            np.ones_like(kmagnitude), kmagnitude, mu, edges, counts, (0,)
        )
        np.testing.assert_allclose(poles, 1.0)
        with self.assertRaisesRegex(ValueError, "nmodes"):
            bin_periodic_multipoles(
                np.ones_like(kmagnitude),
                kmagnitude,
                mu,
                edges,
                counts + 1,
                (0,),
            )

    def test_small_lattice_prediction_shapes_symmetry_and_finiteness(self) -> None:
        meshsize = 16
        boxsize = 1000.0
        modes = np.fft.fftfreq(meshsize) * meshsize * 2.0 * np.pi / boxsize
        kmagnitude = np.sqrt(
            modes[:, None, None] ** 2
            + modes[None, :, None] ** 2
            + modes[None, None, :] ** 2
        )
        edges = np.array([[0.005, 0.015], [0.015, 0.025], [0.025, 0.035]])
        counts = np.array(
            [
                np.count_nonzero((kmagnitude >= lower) & (kmagnitude < upper))
                for lower, upper in edges
            ]
        )
        spectra = linear_gaussian_threshold_rsd_spectra(
            edges,
            counts,
            quantiles=(1, 2, 4, 5),
            ells=(0, 2),
            boxsize=boxsize,
            meshsize=meshsize,
            correlation_ngrid=1001,
        )
        self.assertEqual(spectra.matter_power_poles.shape, (2, 3))
        self.assertEqual(spectra.quantile_matter_power_poles.shape, (4, 2, 3))
        self.assertEqual(
            spectra.quantile_quantile_total_power_poles.shape,
            (4, 4, 2, 3),
        )
        self.assertGreater(spectra.smoothed_variance, 0.0)
        np.testing.assert_allclose(
            spectra.quantile_quantile_total_power_poles,
            spectra.quantile_quantile_total_power_poles.swapaxes(0, 1),
        )
        self.assertTrue(np.isfinite(spectra.quantile_quantile_total_power_poles).all())

    def test_small_mock_mean_smoothing_and_validation(self) -> None:
        x = np.arange(9.0)
        spectra = np.stack([x**2, 2.0 * x**2])
        smoothed = smooth_calibration_spectra(spectra, window=7, polynomial_order=2)
        np.testing.assert_allclose(smoothed, spectra, atol=1e-12)
        np.testing.assert_allclose(
            smooth_calibration_spectra(spectra, window=1), spectra
        )
        with self.assertRaisesRegex(ValueError, "odd"):
            smooth_calibration_spectra(spectra, window=4)


if __name__ == "__main__":
    unittest.main()
