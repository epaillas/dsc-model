"""Tests for the analytic Gaussian density-split covariance."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import integrate, special


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "density_split_gaussian_covariance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "density_split_gaussian_covariance", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _periodic_geometry(
    k_edges: np.ndarray, *, boxsize: float = 1000.0, meshsize: int = 16
) -> dict:
    modes = np.fft.fftfreq(meshsize) * meshsize * 2.0 * np.pi / boxsize
    magnitude = np.sqrt(
        modes[:, None, None] ** 2
        + modes[None, :, None] ** 2
        + modes[None, None, :] ** 2
    )
    nmodes = np.asarray(
        [
            np.count_nonzero((magnitude >= lower) & (magnitude < upper))
            for lower, upper in k_edges
        ],
        dtype="f8",
    )
    return {
        "k_edges": np.asarray(k_edges, dtype="f8"),
        "nmodes": nmodes,
        "boxsize": boxsize,
        "meshsize": meshsize,
        "los": "z",
    }


class DensitySplitGaussianCovarianceTest(unittest.TestCase):
    def test_gaussian_coefficients_obey_partition_and_reflection(self) -> None:
        coefficients = MODULE.gaussian_quantile_c1(5)
        self.assertEqual(tuple(coefficients), (1, 2, 3, 4, 5))
        self.assertAlmostEqual(sum(coefficients.values()), 0.0, places=14)
        self.assertAlmostEqual(coefficients[1], -coefficients[5], places=14)
        self.assertAlmostEqual(coefficients[2], -coefficients[4], places=14)
        self.assertEqual(coefficients[3], 0.0)

    def test_categorical_noise_has_partition_null_mode(self) -> None:
        probabilities = np.full(5, 0.2)
        noise = MODULE.categorical_lattice_noise(probabilities, query_density=2.0)
        np.testing.assert_allclose(noise @ probabilities, 0.0, atol=1e-14)
        np.testing.assert_allclose(np.diag(noise), 2.0)
        selected = noise[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])]
        self.assertGreater(np.linalg.eigvalsh(selected)[0], 0.0)

    def test_tree_spectra_and_quantile_major_power_covariance(self) -> None:
        k = np.array([0.05, 0.1])
        linear_power = np.array([100.0, 50.0])
        spectra = MODULE.tree_density_split_spectra(
            k,
            linear_power,
            {1: -2.0, 5: 2.0},
            smoothing_radius=0.0,
        )
        np.testing.assert_allclose(
            spectra.quantile_matter,
            [[-200.0, -100.0], [200.0, 100.0]],
        )
        covariance = MODULE.gaussian_cross_power_covariance(
            spectra.matter,
            spectra.quantile_matter,
            spectra.quantile_quantile,
            nmodes=[10.0, 20.0],
            matter_noise=1.0,
            quantile_noise=np.eye(2),
            nrealizations=2,
        )
        self.assertEqual(covariance.shape, (4, 4))
        self.assertEqual(covariance[0, 1], 0.0)
        self.assertNotEqual(covariance[0, 2], 0.0)
        np.testing.assert_allclose(covariance, covariance.T)
        self.assertGreater(np.linalg.eigvalsh(covariance)[0], 0.0)

    def test_standard_auto_limit_and_scaling(self) -> None:
        power = np.array([8.0, 3.0])
        noise = 2.0
        nmodes = np.array([10.0, 25.0])
        covariance = MODULE.gaussian_cross_power_covariance(
            power,
            power[None, :],
            power[None, None, :],
            nmodes=nmodes,
            matter_noise=noise,
            quantile_noise=[[noise]],
            cross_noise=[noise],
        )
        expected = 2.0 * (power + noise) ** 2 / nmodes
        np.testing.assert_allclose(np.diag(covariance), expected)
        covariance_mean = MODULE.gaussian_cross_power_covariance(
            power,
            power[None, :],
            power[None, None, :],
            nmodes=nmodes,
            matter_noise=noise,
            quantile_noise=[[noise]],
            cross_noise=[noise],
            nrealizations=4,
        )
        np.testing.assert_allclose(covariance_mean, covariance / 4.0)

    def test_multipole_covariance_reduces_to_monopole_result(self) -> None:
        matter = np.array([8.0, 3.0])
        cross = np.array([[2.0, -1.0], [1.0, 0.5]])
        quantile = np.empty((2, 2, 2), dtype="f8")
        quantile[:, :, 0] = [[5.0, 1.0], [1.0, 4.0]]
        quantile[:, :, 1] = [[6.0, 1.5], [1.5, 5.0]]
        geometry = _periodic_geometry(np.array([[0.0, 0.015], [0.015, 0.03]]))
        kwargs = {
            "nmodes": geometry["nmodes"],
            "matter_noise": 2.0,
            "quantile_noise": np.eye(2),
            "cross_noise": np.array([0.25, -0.5]),
            "nrealizations": 3,
        }
        monopole = MODULE.gaussian_cross_power_covariance(
            matter, cross, quantile, **kwargs
        )
        multipole = MODULE.gaussian_cross_power_multipole_covariance(
            matter[None, :],
            cross[:, None, :],
            quantile[:, :, None, :],
            ells=(0,),
            k_edges=geometry["k_edges"],
            boxsize=geometry["boxsize"],
            meshsize=geometry["meshsize"],
            los=geometry["los"],
            **kwargs,
        )
        np.testing.assert_allclose(multipole, monopole, rtol=1e-14)

    def test_multipole_normalization_ordering_noise_and_scaling(self) -> None:
        ells = np.array([0, 2, 4])
        matter = np.zeros((3, 2), dtype="f8")
        cross = np.zeros((1, 3, 2), dtype="f8")
        quantile = np.zeros((1, 1, 3, 2), dtype="f8")
        matter[0] = [4.0, 6.0]
        cross[0, 0] = [1.0, 2.0]
        quantile[0, 0, 0] = [5.0, 7.0]
        geometry = _periodic_geometry(np.array([[0.0, 0.015], [0.015, 0.03]]))
        nmodes = geometry["nmodes"]
        angular = MODULE.periodic_shell_angular_moments(
            geometry["k_edges"],
            nmodes,
            ells=ells,
            boxsize=geometry["boxsize"],
            meshsize=geometry["meshsize"],
            los=geometry["los"],
        )
        covariance = MODULE.gaussian_cross_power_multipole_covariance(
            matter,
            cross,
            quantile,
            ells=ells,
            nmodes=nmodes,
            k_edges=geometry["k_edges"],
            boxsize=geometry["boxsize"],
            meshsize=geometry["meshsize"],
            los=geometry["los"],
            angular_moments=angular,
            matter_noise=1.0,
            quantile_noise=[[2.0]],
            cross_noise=[0.5],
        )
        blocks = covariance.reshape(1, 3, 2, 1, 3, 2)
        kernel = np.array(
            [
                (4.0 + 1.0) * (5.0 + 2.0) + (1.0 + 0.5) ** 2,
                (6.0 + 1.0) * (7.0 + 2.0) + (2.0 + 0.5) ** 2,
            ]
        )
        for ia, ella in enumerate(ells):
            for ib, ellb in enumerate(ells):
                expected = (
                    (2 * ella + 1)
                    * (2 * ellb + 1)
                    * angular.moments[:, ia, ib, 0, 0]
                    * kernel
                    / nmodes
                )
                np.testing.assert_allclose(
                    blocks[0, ia, :, 0, ib, :].diagonal(), expected
                )
        np.testing.assert_allclose(blocks[:, :, 0, :, :, 1], 0.0)
        scaled = MODULE.gaussian_cross_power_multipole_covariance(
            matter,
            cross,
            quantile,
            ells=ells,
            nmodes=nmodes,
            k_edges=geometry["k_edges"],
            boxsize=geometry["boxsize"],
            meshsize=geometry["meshsize"],
            los=geometry["los"],
            angular_moments=angular,
            matter_noise=1.0,
            quantile_noise=[[2.0]],
            cross_noise=[0.5],
            nrealizations=5,
        )
        np.testing.assert_allclose(scaled, covariance / 5.0)

    def test_multipole_covariance_rejects_invalid_inputs(self) -> None:
        matter = np.ones((1, 2))
        cross = np.ones((1, 1, 2))
        quantile = np.ones((1, 1, 1, 2))
        geometry = _periodic_geometry(np.array([[0.0, 0.015], [0.015, 0.03]]))
        common = {
            "k_edges": geometry["k_edges"],
            "boxsize": geometry["boxsize"],
            "meshsize": geometry["meshsize"],
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.gaussian_cross_power_multipole_covariance(
                matter,
                cross,
                quantile,
                ells=(0, 0),
                nmodes=geometry["nmodes"],
                **common,
            )
        with self.assertRaisesRegex(ValueError, "ell=0"):
            MODULE.gaussian_cross_power_multipole_covariance(
                matter,
                cross,
                quantile,
                ells=(2,),
                nmodes=geometry["nmodes"],
                **common,
                matter_noise=1.0,
            )
        quantile[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            MODULE.gaussian_cross_power_multipole_covariance(
                matter,
                cross,
                quantile,
                ells=(0,),
                nmodes=geometry["nmodes"],
                **common,
            )

    def test_discrete_moments_match_jaxpower_hermitian_average(self) -> None:
        try:
            import jaxpower
        except ModuleNotFoundError:
            self.skipTest("jaxpower is not installed")
        edges = np.array([[0.0, 0.015], [0.015, 0.03]])
        geometry = _periodic_geometry(edges)
        exact = MODULE.periodic_shell_angular_moments(
            edges, geometry["nmodes"], ells=(0, 2, 4), boxsize=1000.0,
            meshsize=16, los="z"
        )
        attrs = jaxpower.MeshAttrs(
            meshsize=16, boxsize=1000.0, dtype=float, fft_backend="jax"
        )
        binner = jaxpower.BinMesh2SpectrumPoles(
            attrs, edges=np.array([0.0, 0.015, 0.03]), ells=(0,)
        )
        kvec = attrs.kcoords(sparse=True)
        norm = np.sqrt(sum(component**2 for component in kvec))
        mu = kvec[2] / np.where(norm == 0.0, 1.0, norm)
        for indices in ((0, 0, 0, 0), (0, 2, 1, 2), (2, 2, 2, 2)):
            polynomial = 1.0
            for index in indices:
                polynomial *= special.legendre((0, 2, 4)[index])
            expected = np.asarray(binner(polynomial(mu)))
            np.testing.assert_allclose(
                exact.moments[(slice(None), *indices)], expected, rtol=2e-6,
                atol=2e-7
            )
        power = np.array([[4.0, 6.0], [1.0, 2.0], [0.5, -0.2]])
        poles = jaxpower.Mesh2SpectrumPoles(
            [
                jaxpower.Mesh2SpectrumPole(
                    k=np.array([0.0075, 0.0225]),
                    k_edges=edges,
                    nmodes=geometry["nmodes"],
                    num_raw=value,
                    ell=ell,
                )
                for ell, value in zip((0, 2, 4), power)
            ]
        )
        jaxpower_covariance = np.asarray(
            jaxpower.compute_spectrum2_covariance(
                attrs, poles, flags=tuple()
            ).value()
        )
        covariance = MODULE.gaussian_cross_power_multipole_covariance(
            power,
            power[None, ...],
            power[None, None, ...],
            ells=(0, 2, 4),
            nmodes=geometry["nmodes"],
            k_edges=edges,
            boxsize=1000.0,
            meshsize=16,
            los="z",
            angular_moments=exact,
        )
        np.testing.assert_allclose(
            covariance, jaxpower_covariance, rtol=5e-7, atol=1e-6
        )

    def test_discrete_geometry_los_and_failures(self) -> None:
        edges = np.array([[0.0, 0.015], [0.015, 0.03]])
        geometry = _periodic_geometry(edges)
        along_x = MODULE.periodic_shell_angular_moments(
            edges, geometry["nmodes"], ells=(0, 2, 4), boxsize=1000.0,
            meshsize=16, los="x"
        )
        along_z = MODULE.periodic_shell_angular_moments(
            edges, geometry["nmodes"], ells=(0, 2, 4), boxsize=1000.0,
            meshsize=16, los="z"
        )
        np.testing.assert_allclose(along_x.moments, along_z.moments, atol=1e-14)
        fundamental = 2.0 * np.pi / 1000.0
        boundary_edges = np.array(
            [[0.0, fundamental], [fundamental, 2.0 * fundamental]]
        )
        boundary_geometry = _periodic_geometry(boundary_edges)
        self.assertEqual(boundary_geometry["nmodes"][0], 1.0)
        self.assertEqual(boundary_geometry["nmodes"][1], 26.0)
        MODULE.periodic_shell_angular_moments(
            boundary_edges,
            boundary_geometry["nmodes"],
            ells=(0, 2),
            boxsize=1000.0,
            meshsize=16,
        )
        with self.assertRaisesRegex(ValueError, "nmodes"):
            MODULE.periodic_shell_angular_moments(
                edges, geometry["nmodes"] + 1, ells=(0,), boxsize=1000.0,
                meshsize=16
            )
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            MODULE.periodic_shell_angular_moments(
                np.array([[0.0, 0.06]]), np.array([1.0]), ells=(0,),
                boxsize=1000.0, meshsize=16
            )

    def test_joint_matter_cross_power_covariance(self) -> None:
        matter = np.array([10.0, 20.0])
        cross = np.array([[2.0, 3.0], [-1.0, -2.0]])
        quantile = np.empty((2, 2, 2), dtype="f8")
        quantile[:, :, 0] = [[5.0, 1.0], [1.0, 4.0]]
        quantile[:, :, 1] = [[6.0, 1.5], [1.5, 5.0]]
        nmodes = np.array([100.0, 200.0])
        covariance = MODULE.gaussian_matter_cross_power_covariance(
            matter,
            cross,
            quantile,
            nmodes=nmodes,
            matter_noise=2.0,
            nrealizations=5,
        )
        blocks = covariance.reshape(3, 2, 3, 2)
        total_matter = matter + 2.0
        for ik in range(2):
            normalization = nmodes[ik] * 5
            self.assertAlmostEqual(
                blocks[0, ik, 0, ik],
                2.0 * total_matter[ik] ** 2 / normalization,
            )
            np.testing.assert_allclose(
                blocks[0, ik, 1:, ik],
                2.0 * total_matter[ik] * cross[:, ik] / normalization,
            )
            np.testing.assert_allclose(
                blocks[1:, ik, 1:, ik],
                (
                    quantile[:, :, ik] * total_matter[ik]
                    + np.outer(cross[:, ik], cross[:, ik])
                )
                / normalization,
            )
            off_k = [jk for jk in range(2) if jk != ik]
            np.testing.assert_allclose(blocks[:, ik, :, off_k], 0.0)
        np.testing.assert_allclose(covariance, covariance.T)
        self.assertGreater(np.linalg.eigvalsh(covariance)[0], 0.0)

    def test_correlation_matches_fine_power_covariance_transform(self) -> None:
        k = np.linspace(0.002, 0.8, 501)
        power = 1000.0 * np.exp(-((k / 0.2) ** 2))
        spectra = MODULE.tree_density_split_spectra(
            k, power, {1: -1.0, 5: 1.0}, smoothing_radius=5.0
        )
        volume = 2.0e9
        simpson_weights = integrate.simpson(np.eye(k.size), x=k, axis=-1)
        nmodes = volume * k**2 * simpson_weights / (2.0 * np.pi**2)
        power_covariance = MODULE.gaussian_cross_power_covariance(
            spectra.matter,
            spectra.quantile_matter,
            spectra.quantile_quantile,
            nmodes=nmodes,
        )
        separation_edges = np.array([20.0, 30.0, 40.0])
        bessel = MODULE.bin_averaged_spherical_j0(k, separation_edges)
        transform = bessel * (k**2 * simpson_weights / (2.0 * np.pi**2))[None, :]
        transform = np.kron(np.eye(2), transform)
        transformed = transform @ power_covariance @ transform.T
        direct = MODULE.gaussian_cross_correlation_covariance(
            k,
            spectra.matter,
            spectra.quantile_matter,
            spectra.quantile_quantile,
            separation_edges=separation_edges,
            volume=volume,
        )
        np.testing.assert_allclose(direct, transformed, rtol=2e-12, atol=1e-20)

    def test_white_noise_contact_is_exact(self) -> None:
        k = np.geomspace(1e-4, 1.0, 101)
        zeros = np.zeros_like(k)
        quantile_noise = np.array([[2.0, -1.0], [-1.0, 2.0]])
        matter_noise = 3.0
        volume = 1.0e8
        edges = np.array([10.0, 20.0, 30.0])
        covariance = MODULE.gaussian_cross_correlation_covariance(
            k,
            zeros,
            np.zeros((2, k.size)),
            np.zeros((2, 2, k.size)),
            separation_edges=edges,
            volume=volume,
            matter_noise=matter_noise,
            quantile_noise=quantile_noise,
        ).reshape(2, 2, 2, 2)
        shell_volume = 4.0 * np.pi / 3.0 * np.diff(edges**3)
        for ia in range(2):
            for ib in range(2):
                expected = quantile_noise[ia, ib] * matter_noise
                np.testing.assert_allclose(
                    np.diag(covariance[ia, :, ib, :]),
                    expected / (volume * shell_volume),
                )
                self.assertEqual(covariance[ia, 0, ib, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
