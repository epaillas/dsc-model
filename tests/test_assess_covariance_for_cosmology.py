from pathlib import Path

import numpy as np
import pytest

from scripts.assess_covariance_for_cosmology import (
    covariance_fold_metrics,
    load_derivative_archive,
    paired_chi2_metrics,
    project_covariance,
    scale_cut_indices,
    validate_covariance,
)


def test_scale_cut_and_covariance_validation():
    k = np.tile([0.05, 0.10, 0.15], 2)
    selected = scale_cut_indices(k, 0.10)
    np.testing.assert_array_equal(selected, [0, 1, 3, 4])
    validated = validate_covariance("identity", np.eye(3))
    np.testing.assert_array_equal(validated, np.eye(3))
    with pytest.raises(ValueError, match="symmetric"):
        validate_covariance("bad", np.array([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive definite"):
        validate_covariance("bad", np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_covariance_metrics_and_paired_chi2_are_finite():
    rng = np.random.default_rng(42)
    covariance = np.array(
        [[1.0, 0.25, 0.0], [0.25, 1.3, 0.1], [0.0, 0.1, 0.8]]
    )
    values = rng.multivariate_normal(np.zeros(3), covariance, size=5000)
    metrics, quadratic = covariance_fold_metrics(
        covariance,
        values,
        k_bins=np.arange(3),
        nbootstrap=20,
        rng=np.random.default_rng(12),
    )
    assert metrics["dimension"] == 3
    assert metrics["chi2_mean_ratio"] == pytest.approx(1.0, abs=0.04)
    assert metrics["chi2_width_ratio"] == pytest.approx(1.0, abs=0.05)
    assert metrics["generalized_log_eigenvalue_rms"] < 0.08
    paired = paired_chi2_metrics(
        quadratic,
        quadratic,
        dimension=3,
        nbootstrap=20,
        rng=np.random.default_rng(13),
    )
    assert paired["delta_chi2_per_element_mean"] == 0.0
    assert paired["fraction_diagonal_chi2_larger"] == 0.0


def test_projection_detects_alignment_with_corrected_mode():
    direction = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    orthogonal = np.array([0.0, 0.0, 1.0, -1.0]) / np.sqrt(2.0)
    diagonal = np.eye(4)
    corrected = diagonal + 2.0 * np.outer(direction, direction)
    residuals = np.zeros((20, 4))

    aligned_diagonal = project_covariance(
        diagonal, corrected, residuals, direction[:, None]
    )
    aligned_corrected = project_covariance(
        corrected, corrected, residuals, direction[:, None]
    )
    sigma_diagonal = np.sqrt(aligned_diagonal.posterior_covariance[0, 0])
    sigma_corrected = np.sqrt(aligned_corrected.posterior_covariance[0, 0])
    assert sigma_corrected / sigma_diagonal == pytest.approx(np.sqrt(3.0))

    orthogonal_diagonal = project_covariance(
        diagonal, corrected, residuals, orthogonal[:, None]
    )
    orthogonal_corrected = project_covariance(
        corrected, corrected, residuals, orthogonal[:, None]
    )
    np.testing.assert_allclose(
        orthogonal_diagonal.posterior_covariance,
        orthogonal_corrected.posterior_covariance,
        rtol=1e-12,
        atol=1e-12,
    )


def test_projection_matches_direct_generalized_least_squares():
    covariance = np.array(
        [[1.2, 0.2, 0.0], [0.2, 0.9, 0.1], [0.0, 0.1, 1.1]]
    )
    jacobian = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 0.7]])
    residuals = np.array([[0.2, -0.1, 0.4], [-0.3, 0.2, 0.1]])
    result = project_covariance(
        covariance,
        covariance,
        residuals,
        jacobian,
        cosmological_indices=(0, 1),
    )
    precision = np.linalg.inv(covariance)
    fisher = jacobian.T @ precision @ jacobian
    estimator = np.linalg.inv(fisher) @ jacobian.T @ precision
    np.testing.assert_allclose(result.estimates, residuals @ estimator.T)
    np.testing.assert_allclose(result.posterior_covariance, np.linalg.inv(fisher))
    np.testing.assert_allclose(result.sandwich_covariance, np.linalg.inv(fisher))


def _write_derivatives(path: Path, **updates):
    arrays = {
        "jacobian": np.arange(8, dtype="f8").reshape(4, 2),
        "parameter_names": np.array(["Omega_m", "bias"]),
        "cosmological_parameter_names": np.array(["Omega_m"]),
        "labels": np.array(["a", "b", "c", "d"]),
        "ells": np.array([0, 0, 2, 2]),
        "k_for_element": np.array([0.1, 0.2, 0.1, 0.2]),
    }
    arrays.update(updates)
    np.savez(path, **arrays)


def test_derivative_archive_validation(tmp_path):
    path = tmp_path / "derivatives.npz"
    labels = np.array(["a", "b", "c", "d"])
    ells = np.array([0, 0, 2, 2])
    k = np.array([0.1, 0.2, 0.1, 0.2])
    _write_derivatives(path)
    archive = load_derivative_archive(path, labels=labels, ells=ells, k_for_element=k)
    assert archive.jacobian.shape == (4, 2)
    assert archive.cosmological_parameter_names == ("Omega_m",)

    _write_derivatives(path, labels=np.array(["b", "a", "c", "d"]))
    with pytest.raises(ValueError, match="ordering"):
        load_derivative_archive(path, labels=labels, ells=ells, k_for_element=k)

    _write_derivatives(path, prior_precision=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="positive definite"):
        load_derivative_archive(path, labels=labels, ells=ells, k_for_element=k)


def test_projection_rejects_singular_fisher():
    with pytest.raises(ValueError, match="Fisher matrix"):
        project_covariance(
            np.eye(3),
            np.eye(3),
            np.zeros((10, 3)),
            np.ones((3, 2)),
        )
