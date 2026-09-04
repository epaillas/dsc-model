"""Assess whether a smooth diagonal covariance is sufficient for inference.

The covariance estimators are trained on each of three 500-realization folds
and evaluated on the complementary 1000 realizations.  Estimators are fitted
on the complete data vector before applying each requested scale cut, matching
how the production covariance is intended to be used.

An optional derivative archive enables a linearized cosmological projection.
It must contain ``jacobian``, ``parameter_names``,
``cosmological_parameter_names``, ``labels``, ``ells``, and
``k_for_element``.  A positive-definite ``prior_precision`` is optional.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.stats import chi2, kstest

try:
    from scripts import study_non_gaussian_covariance as study
except (ImportError, ModuleNotFoundError):
    import study_non_gaussian_covariance as study


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_DIR
    / "outputs"
    / "model_validation"
    / "non_gaussian_covariance_inference_assessment"
)
DEFAULT_KMAX_VALUES = (0.10, 0.12, 0.15, 0.20, 0.25)
EVALUATED_MODELS = ("gaussian", "diagonal_smooth", "spiked_diagonal")
MODEL_COLORS = {
    "gaussian": "#0072B2",
    "diagonal_smooth": "#E69F00",
    "spiked_diagonal": "#009E73",
}
MODEL_LABELS = {
    "gaussian": "Gaussian",
    "diagonal_smooth": "smooth diagonal",
    "spiked_diagonal": "diagonal + spikes",
}


@dataclass(frozen=True)
class AssessmentInputs:
    """Measured vectors, Gaussian target, and data-vector metadata."""

    vectors: np.ndarray
    gaussian: np.ndarray
    labels: np.ndarray
    pair_labels: np.ndarray
    ells: np.ndarray
    k_bins: np.ndarray
    k_for_element: np.ndarray
    k: np.ndarray
    nk: int
    nblocks: int


@dataclass(frozen=True)
class DerivativeArchive:
    """Validated tangent-space information for cosmological projection."""

    jacobian: np.ndarray
    parameter_names: tuple[str, ...]
    cosmological_parameter_names: tuple[str, ...]
    prior_precision: np.ndarray


@dataclass(frozen=True)
class ProjectionResult:
    """Linearized parameter inference for one covariance and test fold."""

    posterior_covariance: np.ndarray
    sandwich_covariance: np.ndarray
    estimates: np.ndarray
    marginal_coverage: np.ndarray
    joint_coverage_68: np.ndarray
    joint_coverage_95: np.ndarray


def _as_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(value) for value in np.asarray(values).tolist())


def validate_covariance(name: str, covariance: np.ndarray) -> np.ndarray:
    """Return a validated symmetric positive-definite covariance."""
    covariance = np.asarray(covariance, dtype="f8")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} is not symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} is not positive definite") from error
    return covariance


def _validate_positive_definite(name: str, matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="f8")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return 0.5 * (matrix + matrix.T)


def load_inputs(data_root: Path, *, kmin: float, kmax: float) -> AssessmentInputs:
    """Load the same redshift-space vector and Gaussian target as the study."""
    ensemble = study.load_rsd_ensemble(
        data_root,
        quantiles=study.QUANTILES,
        ells=study.INPUT_ELLS,
        data_vector=("pqm",),
        kmin=kmin,
        kmax=kmax,
    )
    nk = ensemble.k.size
    indices = study._observable_indices(
        len(study.QUANTILES), len(study.INPUT_ELLS), nk
    )
    vectors = np.asarray(ensemble.vectors(("pqm",))[:, indices], dtype="f8")
    geometry = study.periodic_shell_angular_moments(
        ensemble.k_edges,
        ensemble.nmodes,
        ells=study.INPUT_ELLS,
        boxsize=1000.0,
        meshsize=256,
        los=ensemble.los,
    )
    hybrid = study.build_hybrid_covariance(
        ensemble.spectra.pmm,
        ensemble.spectra.pqm,
        ensemble.spectra.pqq,
        quantiles=ensemble.spectra.quantiles,
        ells=ensemble.spectra.ells,
        k_edges=ensemble.k_edges,
        nmodes=ensemble.nmodes,
        boxsize=1000.0,
        meshsize=256,
        los=ensemble.los,
        data_vector=("pqm",),
        k=ensemble.k,
        angular_moments=geometry,
    )
    gaussian = validate_covariance(
        "Gaussian covariance", hybrid.covariance[np.ix_(indices, indices)]
    )
    nblocks = len(study.QUANTILES) * len(study.OUTPUT_ELLS)
    labels = np.asarray(hybrid.labels)[indices].astype("U")
    pair_labels = np.repeat(
        [f"Pq{quantile}m" for quantile in study.QUANTILES],
        len(study.OUTPUT_ELLS) * nk,
    ).astype("U")
    element_ells = np.tile(np.repeat(study.OUTPUT_ELLS, nk), len(study.QUANTILES))
    k_bins = np.tile(np.arange(nk, dtype="i8"), nblocks)
    k_for_element = np.tile(np.asarray(ensemble.k, dtype="f8"), nblocks)
    if vectors.shape[1] != gaussian.shape[0] or labels.size != vectors.shape[1]:
        raise ValueError("inconsistent data-vector dimensions")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("realization vectors contain non-finite values")
    return AssessmentInputs(
        vectors=vectors,
        gaussian=gaussian,
        labels=labels,
        pair_labels=pair_labels,
        ells=element_ells,
        k_bins=k_bins,
        k_for_element=k_for_element,
        k=np.asarray(ensemble.k, dtype="f8"),
        nk=nk,
        nblocks=nblocks,
    )


def crossfit_folds(nrealizations: int, *, seed: int) -> tuple[np.ndarray, ...]:
    """Return the three reproducible folds used by the covariance study."""
    if nrealizations < 6 or nrealizations % 3:
        raise ValueError("the three-fold assessment requires a multiple of three realizations")
    permutation = np.random.default_rng(seed).permutation(nrealizations)
    folds = tuple(np.asarray(fold, dtype="i8") for fold in np.array_split(permutation, 3))
    if any(fold.size * 3 != nrealizations for fold in folds):
        raise ValueError("cross-validation folds have inconsistent sizes")
    return folds


def scale_cut_indices(k_for_element: np.ndarray, kmax: float) -> np.ndarray:
    """Select all complete observable blocks through ``kmax``."""
    k_for_element = np.asarray(k_for_element, dtype="f8")
    if k_for_element.ndim != 1 or not np.all(np.isfinite(k_for_element)):
        raise ValueError("k coordinates must be a finite one-dimensional array")
    selected = np.flatnonzero(k_for_element <= float(kmax) + 1e-12)
    if not selected.size:
        raise ValueError(f"kmax={kmax:g} selects no data-vector elements")
    return selected


def _correlation(covariance: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.diag(covariance))
    return covariance / sigma[:, None] / sigma[None, :]


def _logdet(covariance: np.ndarray) -> float:
    cholesky = np.linalg.cholesky(covariance)
    return float(2.0 * np.sum(np.log(np.diag(cholesky))))


def _bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def _distribution_statistics(
    quadratic: np.ndarray,
    *,
    dimension: int,
    centering_scale: float,
) -> dict[str, float]:
    scaled = np.asarray(quadratic, dtype="f8") / centering_scale
    expected_width = np.sqrt(2.0 * dimension)
    central = chi2.ppf([0.16, 0.84], dimension)
    test = kstest(scaled, chi2(df=dimension).cdf)
    return {
        "chi2_mean_ratio": float(np.mean(scaled) / dimension),
        "chi2_width_ratio": float(np.std(scaled, ddof=1) / expected_width),
        "chi2_median_ratio": float(np.median(scaled) / chi2.ppf(0.5, dimension)),
        "chi2_q16": float(np.percentile(scaled, 16)),
        "chi2_q84": float(np.percentile(scaled, 84)),
        "chi2_central_68_coverage": float(
            np.mean((scaled >= central[0]) & (scaled <= central[1]))
        ),
        "chi2_cdf_68_coverage": float(np.mean(scaled <= chi2.ppf(0.68, dimension))),
        "chi2_cdf_95_coverage": float(np.mean(scaled <= chi2.ppf(0.95, dimension))),
        "chi2_ks_statistic": float(test.statistic),
        "chi2_ks_pvalue": float(test.pvalue),
    }


def _bootstrap_distribution_statistics(
    quadratic: np.ndarray,
    *,
    dimension: int,
    centering_scale: float,
    nbootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if nbootstrap <= 0:
        return {}
    quadratic = np.asarray(quadratic, dtype="f8")
    central = chi2.ppf([0.16, 0.84], dimension) * centering_scale
    threshold_68 = chi2.ppf(0.68, dimension) * centering_scale
    threshold_95 = chi2.ppf(0.95, dimension) * centering_scale
    means = np.empty(nbootstrap)
    widths = np.empty(nbootstrap)
    central_coverages = np.empty(nbootstrap)
    coverages_68 = np.empty(nbootstrap)
    coverages_95 = np.empty(nbootstrap)
    for ibootstrap in range(nbootstrap):
        sample = quadratic[rng.integers(0, quadratic.size, quadratic.size)]
        means[ibootstrap] = np.mean(sample) / centering_scale / dimension
        widths[ibootstrap] = (
            np.std(sample / centering_scale, ddof=1) / np.sqrt(2.0 * dimension)
        )
        central_coverages[ibootstrap] = np.mean(
            (sample >= central[0]) & (sample <= central[1])
        )
        coverages_68[ibootstrap] = np.mean(sample <= threshold_68)
        coverages_95[ibootstrap] = np.mean(sample <= threshold_95)
    result = {}
    for name, values in (
        ("chi2_mean_ratio", means),
        ("chi2_width_ratio", widths),
        ("chi2_central_68_coverage", central_coverages),
        ("chi2_cdf_68_coverage", coverages_68),
        ("chi2_cdf_95_coverage", coverages_95),
    ):
        lower, upper = _bootstrap_interval(values)
        result[f"{name}_bootstrap_p025"] = lower
        result[f"{name}_bootstrap_p975"] = upper
    return result


def covariance_fold_metrics(
    covariance: np.ndarray,
    test_values: np.ndarray,
    *,
    k_bins: np.ndarray,
    nbootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate one covariance on independent realization vectors."""
    covariance = validate_covariance("model covariance", covariance)
    test_values = np.asarray(test_values, dtype="f8")
    if test_values.ndim != 2 or test_values.shape[1] != covariance.shape[0]:
        raise ValueError("test vectors and model covariance have incompatible dimensions")
    if test_values.shape[0] <= covariance.shape[0]:
        raise ValueError("the held-out covariance must be full rank")
    residuals = test_values - np.mean(test_values, axis=0)
    reference = validate_covariance(
        "held-out covariance", study._sample_covariance(test_values)
    )
    factor = linalg.cho_factor(covariance, lower=True, check_finite=False)
    solved_residuals = linalg.cho_solve(factor, residuals.T, check_finite=False).T
    quadratic = np.einsum("ni,ni->n", residuals, solved_residuals)
    dimension = covariance.shape[0]
    centering_scale = (test_values.shape[0] - 1.0) / test_values.shape[0]
    metrics = _distribution_statistics(
        quadratic, dimension=dimension, centering_scale=centering_scale
    )
    metrics.update(
        _bootstrap_distribution_statistics(
            quadratic,
            dimension=dimension,
            centering_scale=centering_scale,
            nbootstrap=nbootstrap,
            rng=rng,
        )
    )
    solved_reference = linalg.cho_solve(factor, reference, check_finite=False)
    generalized = linalg.eigvalsh(reference, covariance, check_finite=False)
    if np.any(generalized <= 0.0):
        raise ValueError("generalized covariance eigenvalues must be positive")
    expected_width_ratio = float(
        np.sqrt(np.trace(solved_reference @ solved_reference) / dimension)
    )
    model_correlation = _correlation(covariance)
    reference_correlation = _correlation(reference)
    off_diagonal = ~np.eye(dimension, dtype=bool)
    cross_shell = np.asarray(k_bins)[:, None] != np.asarray(k_bins)[None, :]
    metrics.update(
        {
            "dimension": int(dimension),
            "ntest": int(test_values.shape[0]),
            "chi2_expected_width_ratio": expected_width_ratio,
            "kl_per_element": float(
                0.5
                * (
                    np.trace(solved_reference)
                    - dimension
                    + _logdet(covariance)
                    - _logdet(reference)
                )
                / dimension
            ),
            "correlation_rms": float(
                np.sqrt(
                    np.mean(
                        (
                            model_correlation[off_diagonal]
                            - reference_correlation[off_diagonal]
                        )
                        ** 2
                    )
                )
            ),
            "cross_shell_correlation_rms": float(
                np.sqrt(
                    np.mean(
                        (
                            model_correlation[cross_shell]
                            - reference_correlation[cross_shell]
                        )
                        ** 2
                    )
                )
            ),
            "generalized_eigenvalue_minimum": float(generalized[0]),
            "generalized_eigenvalue_p05": float(np.percentile(generalized, 5)),
            "generalized_eigenvalue_median": float(np.median(generalized)),
            "generalized_eigenvalue_p95": float(np.percentile(generalized, 95)),
            "generalized_eigenvalue_maximum": float(generalized[-1]),
            "generalized_log_eigenvalue_rms": float(
                np.sqrt(np.mean(np.log(generalized) ** 2))
            ),
        }
    )
    return metrics, quadratic


def paired_chi2_metrics(
    diagonal_quadratic: np.ndarray,
    spiked_quadratic: np.ndarray,
    *,
    dimension: int,
    nbootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Summarize the paired per-mock chi-square change."""
    diagonal_quadratic = np.asarray(diagonal_quadratic, dtype="f8")
    spiked_quadratic = np.asarray(spiked_quadratic, dtype="f8")
    if diagonal_quadratic.shape != spiked_quadratic.shape:
        raise ValueError("paired chi-square arrays must have the same shape")
    difference = (diagonal_quadratic - spiked_quadratic) / dimension
    result = {
        "delta_chi2_per_element_mean": float(np.mean(difference)),
        "delta_chi2_per_element_std": float(np.std(difference, ddof=1)),
        "delta_chi2_per_element_p16": float(np.percentile(difference, 16)),
        "delta_chi2_per_element_median": float(np.median(difference)),
        "delta_chi2_per_element_p84": float(np.percentile(difference, 84)),
        "fraction_diagonal_chi2_larger": float(np.mean(difference > 0.0)),
    }
    if nbootstrap > 0:
        means = np.empty(nbootstrap)
        for ibootstrap in range(nbootstrap):
            sample = difference[rng.integers(0, difference.size, difference.size)]
            means[ibootstrap] = np.mean(sample)
        lower, upper = _bootstrap_interval(means)
        result["delta_chi2_per_element_mean_bootstrap_p025"] = lower
        result["delta_chi2_per_element_mean_bootstrap_p975"] = upper
    return result


def load_derivative_archive(
    path: Path,
    *,
    labels: np.ndarray,
    ells: np.ndarray,
    k_for_element: np.ndarray,
) -> DerivativeArchive:
    """Load derivatives and require exact agreement with vector metadata."""
    required = {
        "jacobian",
        "parameter_names",
        "cosmological_parameter_names",
        "labels",
        "ells",
        "k_for_element",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"derivative archive is missing keys: {', '.join(missing)}")
        jacobian = np.asarray(archive["jacobian"], dtype="f8")
        parameter_names = _as_strings(archive["parameter_names"])
        cosmological_names = _as_strings(archive["cosmological_parameter_names"])
        archive_labels = np.asarray(archive["labels"]).astype("U")
        archive_ells = np.asarray(archive["ells"], dtype="i8")
        archive_k = np.asarray(archive["k_for_element"], dtype="f8")
        prior = (
            np.asarray(archive["prior_precision"], dtype="f8")
            if "prior_precision" in archive.files
            else np.zeros((len(parameter_names), len(parameter_names)), dtype="f8")
        )
        has_prior = "prior_precision" in archive.files
    size = np.asarray(labels).size
    if jacobian.ndim != 2 or jacobian.shape != (size, len(parameter_names)):
        raise ValueError(
            "jacobian must have shape (data-vector size, number of parameters)"
        )
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("jacobian contains non-finite values")
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter_names must be unique")
    if not cosmological_names or not set(cosmological_names).issubset(parameter_names):
        raise ValueError("cosmological_parameter_names must be a nonempty parameter subset")
    if not np.array_equal(archive_labels, np.asarray(labels).astype("U")):
        raise ValueError("derivative labels do not match the covariance ordering")
    if not np.array_equal(archive_ells, np.asarray(ells, dtype="i8")):
        raise ValueError("derivative multipoles do not match the covariance ordering")
    if not np.allclose(
        archive_k, np.asarray(k_for_element, dtype="f8"), rtol=1e-12, atol=1e-12
    ):
        raise ValueError("derivative k coordinates do not match the covariance ordering")
    if prior.shape != (len(parameter_names), len(parameter_names)):
        raise ValueError("prior_precision has inconsistent dimensions")
    if has_prior:
        prior = _validate_positive_definite("prior_precision", prior)
    elif not np.all(np.isfinite(prior)):
        raise ValueError("prior_precision contains non-finite values")
    return DerivativeArchive(
        jacobian=jacobian,
        parameter_names=parameter_names,
        cosmological_parameter_names=cosmological_names,
        prior_precision=prior,
    )


def project_covariance(
    covariance: np.ndarray,
    reference: np.ndarray,
    residuals: np.ndarray,
    jacobian: np.ndarray,
    *,
    prior_precision: np.ndarray | None = None,
    cosmological_indices: Sequence[int] | None = None,
) -> ProjectionResult:
    """Perform Fisher, sandwich, and mock-coverage calculations."""
    covariance = validate_covariance("projected model covariance", covariance)
    reference = validate_covariance("projected reference covariance", reference)
    residuals = np.asarray(residuals, dtype="f8")
    jacobian = np.asarray(jacobian, dtype="f8")
    dimension = covariance.shape[0]
    if reference.shape != covariance.shape:
        raise ValueError("projected covariances have incompatible dimensions")
    if jacobian.ndim != 2 or jacobian.shape[0] != dimension:
        raise ValueError("projected jacobian has incompatible dimensions")
    if residuals.ndim != 2 or residuals.shape[1] != dimension:
        raise ValueError("projected residuals have incompatible dimensions")
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(residuals)):
        raise ValueError("projected inputs contain non-finite values")
    nparameters = jacobian.shape[1]
    prior = (
        np.zeros((nparameters, nparameters), dtype="f8")
        if prior_precision is None
        else np.asarray(prior_precision, dtype="f8")
    )
    if prior.shape != (nparameters, nparameters):
        raise ValueError("projected prior precision has incompatible dimensions")
    if not np.all(np.isfinite(prior)) or not np.allclose(prior, prior.T):
        raise ValueError("projected prior precision must be finite and symmetric")
    factor = linalg.cho_factor(covariance, lower=True, check_finite=False)
    precision_jacobian = linalg.cho_solve(factor, jacobian, check_finite=False)
    fisher = jacobian.T @ precision_jacobian + prior
    fisher = 0.5 * (fisher + fisher.T)
    try:
        fisher_factor = linalg.cho_factor(fisher, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("Fisher matrix is singular or not positive definite") from error
    posterior = linalg.cho_solve(
        fisher_factor, np.eye(nparameters), check_finite=False
    )
    estimator = posterior @ precision_jacobian.T
    sandwich = estimator @ reference @ estimator.T
    sandwich = 0.5 * (sandwich + sandwich.T)
    estimates = residuals @ estimator.T
    sigma = np.sqrt(np.diag(posterior))
    marginal_coverage = np.abs(estimates) <= sigma[None, :]
    if cosmological_indices is None:
        cosmological_indices = tuple(range(nparameters))
    cosmological_indices = np.asarray(cosmological_indices, dtype="i8")
    if (
        cosmological_indices.ndim != 1
        or not cosmological_indices.size
        or np.any(cosmological_indices < 0)
        or np.any(cosmological_indices >= nparameters)
    ):
        raise ValueError("invalid cosmological parameter indices")
    cosmological_covariance = posterior[np.ix_(cosmological_indices, cosmological_indices)]
    cosmological_covariance = validate_covariance(
        "marginalized cosmological covariance", cosmological_covariance
    )
    cosmological_estimates = estimates[:, cosmological_indices]
    cosmological_precision = np.linalg.inv(cosmological_covariance)
    distance = np.einsum(
        "ni,ij,nj->n",
        cosmological_estimates,
        cosmological_precision,
        cosmological_estimates,
    )
    dof = cosmological_indices.size
    return ProjectionResult(
        posterior_covariance=posterior,
        sandwich_covariance=sandwich,
        estimates=estimates,
        marginal_coverage=marginal_coverage,
        joint_coverage_68=distance <= chi2.ppf(0.68, dof),
        joint_coverage_95=distance <= chi2.ppf(0.95, dof),
    )


def _coverage_interval(
    indicators: np.ndarray,
    *,
    nbootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indicators = np.asarray(indicators, dtype="f8")
    if nbootstrap <= 0:
        value = float(np.mean(indicators))
        return value, value
    values = np.empty(nbootstrap)
    for ibootstrap in range(nbootstrap):
        sample = indicators[rng.integers(0, indicators.shape[0], indicators.shape[0])]
        values[ibootstrap] = np.mean(sample)
    return _bootstrap_interval(values)


def _paired_coverage_interval(
    first: np.ndarray,
    second: np.ndarray,
    *,
    nbootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    difference = np.asarray(first, dtype="f8") - np.asarray(second, dtype="f8")
    return _coverage_interval(difference, nbootstrap=nbootstrap, rng=rng)


def projection_records(
    models: dict[str, np.ndarray],
    test_values: np.ndarray,
    reference: np.ndarray,
    derivatives: DerivativeArchive,
    selected: np.ndarray,
    *,
    fold: int,
    kmax: float,
    nbootstrap: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict]:
    """Project the diagonal and spiked covariances into parameter space."""
    names = derivatives.parameter_names
    cosmological = derivatives.cosmological_parameter_names
    cosmological_indices = tuple(names.index(name) for name in cosmological)
    residuals = test_values[:, selected] - np.mean(test_values[:, selected], axis=0)
    jacobian = derivatives.jacobian[selected]
    results = {}
    records = []
    for model_name in ("diagonal_smooth", "spiked_diagonal"):
        result = project_covariance(
            models[model_name][np.ix_(selected, selected)],
            reference,
            residuals,
            jacobian,
            prior_precision=derivatives.prior_precision,
            cosmological_indices=cosmological_indices,
        )
        results[model_name] = result
        predicted = np.sqrt(np.diag(result.posterior_covariance))
        sandwich = np.sqrt(np.diag(result.sandwich_covariance))
        for iparameter, name in enumerate(names):
            coverage = result.marginal_coverage[:, iparameter]
            lower, upper = _coverage_interval(
                coverage, nbootstrap=nbootstrap, rng=rng
            )
            records.append(
                {
                    "fold": fold,
                    "kmax": kmax,
                    "model": model_name,
                    "parameter": name,
                    "is_cosmological": name in cosmological,
                    "predicted_sigma": float(predicted[iparameter]),
                    "sandwich_sigma": float(sandwich[iparameter]),
                    "sandwich_to_predicted_sigma": float(
                        sandwich[iparameter] / predicted[iparameter]
                    ),
                    "marginal_68_coverage": float(np.mean(coverage)),
                    "marginal_68_coverage_bootstrap_p025": lower,
                    "marginal_68_coverage_bootstrap_p975": upper,
                }
            )
    joint_records = []
    for model_name, result in results.items():
        for level, indicators in (
            (68, result.joint_coverage_68),
            (95, result.joint_coverage_95),
        ):
            lower, upper = _coverage_interval(
                indicators, nbootstrap=nbootstrap, rng=rng
            )
            joint_records.append(
                {
                    "fold": fold,
                    "kmax": kmax,
                    "model": model_name,
                    "level": level,
                    "coverage": float(np.mean(indicators)),
                    "coverage_bootstrap_p025": lower,
                    "coverage_bootstrap_p975": upper,
                }
            )
    diagonal = results["diagonal_smooth"]
    spiked = results["spiked_diagonal"]
    comparisons = {"parameters": {}, "joint": {}}
    for iparameter, name in enumerate(names):
        if name not in cosmological:
            continue
        diagonal_sigma = np.sqrt(diagonal.posterior_covariance[iparameter, iparameter])
        spiked_sigma = np.sqrt(spiked.posterior_covariance[iparameter, iparameter])
        coverage_interval = _paired_coverage_interval(
            diagonal.marginal_coverage[:, iparameter],
            spiked.marginal_coverage[:, iparameter],
            nbootstrap=nbootstrap,
            rng=rng,
        )
        comparisons["parameters"][name] = {
            "fractional_sigma_difference": float(diagonal_sigma / spiked_sigma - 1.0),
            "paired_coverage_difference": float(
                np.mean(diagonal.marginal_coverage[:, iparameter])
                - np.mean(spiked.marginal_coverage[:, iparameter])
            ),
            "paired_coverage_difference_bootstrap_p025": coverage_interval[0],
            "paired_coverage_difference_bootstrap_p975": coverage_interval[1],
        }
    for level, diagonal_indicators, spiked_indicators in (
        (68, diagonal.joint_coverage_68, spiked.joint_coverage_68),
        (95, diagonal.joint_coverage_95, spiked.joint_coverage_95),
    ):
        interval = _paired_coverage_interval(
            diagonal_indicators,
            spiked_indicators,
            nbootstrap=nbootstrap,
            rng=rng,
        )
        comparisons["joint"][str(level)] = {
            "paired_coverage_difference": float(
                np.mean(diagonal_indicators) - np.mean(spiked_indicators)
            ),
            "paired_coverage_difference_bootstrap_p025": interval[0],
            "paired_coverage_difference_bootstrap_p975": interval[1],
        }
    return records, joint_records, comparisons


def _aggregate(records: Sequence[dict], keys: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        groups.setdefault(tuple(record[key] for key in keys), []).append(record)
    output = []
    ignored = set(keys) | {"fold", "model", "kmax"}
    for group, rows in sorted(groups.items()):
        aggregate = dict(zip(keys, group))
        aggregate["folds"] = len(rows)
        metric_names = sorted(set(rows[0]) - ignored)
        for metric in metric_names:
            values = [row[metric] for row in rows]
            if not values or isinstance(values[0], (str, bool)):
                continue
            numeric = np.asarray(values, dtype="f8")
            aggregate[f"{metric}_median"] = float(np.median(numeric))
            aggregate[f"{metric}_minimum"] = float(np.min(numeric))
            aggregate[f"{metric}_maximum"] = float(np.max(numeric))
        output.append(aggregate)
    return output


def _write_csv(path: Path, records: Sequence[dict]) -> None:
    if not records:
        return
    fields = list(records[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot_summary(path: Path, records: Sequence[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    specifications = (
        ("chi2_mean_ratio", r"$\langle\chi^2\rangle/p$", 1.0),
        ("chi2_width_ratio", r"$\mathrm{Std}(\chi^2)/\sqrt{2p}$", 1.0),
        ("generalized_log_eigenvalue_rms", r"RMS$(\log\lambda_a)$", 0.0),
    )
    for axis, (metric, ylabel, reference) in zip(axes, specifications):
        for model in EVALUATED_MODELS:
            rows = [row for row in records if row["model"] == model]
            kmax_values = sorted({float(row["kmax"]) for row in rows})
            medians = []
            minimum = []
            maximum = []
            for kmax in kmax_values:
                values = np.asarray(
                    [float(row[metric]) for row in rows if row["kmax"] == kmax]
                )
                medians.append(np.median(values))
                minimum.append(np.min(values))
                maximum.append(np.max(values))
                axis.scatter(
                    np.full(values.size, kmax),
                    values,
                    color=MODEL_COLORS[model],
                    alpha=0.28,
                    s=18,
                    zorder=2,
                )
            axis.plot(
                kmax_values,
                medians,
                marker="o",
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                zorder=3,
            )
            axis.fill_between(
                kmax_values,
                minimum,
                maximum,
                color=MODEL_COLORS[model],
                alpha=0.10,
                linewidth=0,
            )
        axis.axhline(reference, color="0.25", linewidth=1.0, linestyle="--")
        axis.set_xlabel(r"$k_{\max}\,[h\,\mathrm{Mpc}^{-1}]$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Held-out covariance diagnostics (500 calibration / 1000 test mocks)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _projection_verdict(
    parameter_records: Sequence[dict],
    joint_records: Sequence[dict],
    comparisons: Sequence[dict],
    *,
    kmax_values: Sequence[float],
    tolerance: float = 0.05,
) -> list[dict]:
    """Apply the preregistered 5% and coverage criteria per scale cut."""
    verdicts = []
    for kmax in kmax_values:
        reasons = []
        relevant_comparisons = [row for row in comparisons if row["kmax"] == kmax]
        for row in relevant_comparisons:
            for name, result in row["comparisons"]["parameters"].items():
                if abs(result["fractional_sigma_difference"]) >= tolerance:
                    reasons.append(
                        f"fold {row['fold']} {name} sigma changes by at least {100*tolerance:.0f}%"
                    )
                if not (
                    result["paired_coverage_difference_bootstrap_p025"]
                    <= 0.0
                    <= result["paired_coverage_difference_bootstrap_p975"]
                ):
                    reasons.append(f"fold {row['fold']} {name} coverage differs significantly")
            for level, result in row["comparisons"]["joint"].items():
                if not (
                    result["paired_coverage_difference_bootstrap_p025"]
                    <= 0.0
                    <= result["paired_coverage_difference_bootstrap_p975"]
                ):
                    reasons.append(
                        f"fold {row['fold']} joint {level}% coverage differs significantly"
                    )
        for row in parameter_records:
            if (
                row["kmax"] == kmax
                and row["model"] == "diagonal_smooth"
                and row["is_cosmological"]
            ):
                if abs(row["sandwich_to_predicted_sigma"] - 1.0) >= tolerance:
                    reasons.append(
                        f"fold {row['fold']} {row['parameter']} sandwich error misses 5% target"
                    )
                nominal = float(chi2.cdf(1.0, 1))
                if not (
                    row["marginal_68_coverage_bootstrap_p025"]
                    <= nominal
                    <= row["marginal_68_coverage_bootstrap_p975"]
                ):
                    reasons.append(
                        f"fold {row['fold']} {row['parameter']} marginal coverage is non-nominal"
                    )
        for row in joint_records:
            if row["kmax"] == kmax and row["model"] == "diagonal_smooth":
                nominal = row["level"] / 100.0
                if not (
                    row["coverage_bootstrap_p025"]
                    <= nominal
                    <= row["coverage_bootstrap_p975"]
                ):
                    reasons.append(
                        f"fold {row['fold']} joint {row['level']}% coverage is non-nominal"
                    )
        verdicts.append(
            {
                "kmax": float(kmax),
                "diagonal_only_sufficient": not reasons,
                "reasons": sorted(set(reasons)),
            }
        )
    return verdicts


def run_assessment(
    *,
    data_root: Path,
    output_root: Path,
    kmin: float,
    kmax_values: Sequence[float],
    seed: int,
    nbootstrap: int,
    derivative_archive: Path | None = None,
) -> dict:
    """Run covariance-only diagnostics and optional parameter projection."""
    kmax_values = tuple(sorted(set(float(value) for value in kmax_values)))
    if not kmax_values or kmin >= min(kmax_values):
        raise ValueError("k cuts must be nonempty and larger than kmin")
    if nbootstrap < 0:
        raise ValueError("nbootstrap must be nonnegative")
    inputs = load_inputs(data_root, kmin=kmin, kmax=max(kmax_values))
    folds = crossfit_folds(inputs.vectors.shape[0], seed=seed)
    if inputs.vectors.shape[0] == 1500 and any(fold.size != 500 for fold in folds):
        raise AssertionError("the 1500-mock study must use 500-realization folds")
    derivatives = (
        load_derivative_archive(
            derivative_archive,
            labels=inputs.labels,
            ells=inputs.ells,
            k_for_element=inputs.k_for_element,
        )
        if derivative_archive is not None
        else None
    )
    all_indices = np.arange(inputs.vectors.shape[0])
    records = []
    paired_records = []
    parameter_rows = []
    joint_rows = []
    comparison_rows = []
    for ifold, train_indices in enumerate(folds):
        test_indices = np.setdiff1d(all_indices, train_indices, assume_unique=False)
        fitted = study._fit_models(
            inputs.vectors[train_indices],
            inputs.gaussian,
            nblocks=inputs.nblocks,
            nk=inputs.nk,
        )
        models = {name: fitted[name].covariance for name in EVALUATED_MODELS}
        test_values = inputs.vectors[test_indices]
        for ikmax, kmax in enumerate(kmax_values):
            selected = scale_cut_indices(inputs.k_for_element, kmax)
            quadratic = {}
            selected_test = test_values[:, selected]
            selected_reference = validate_covariance(
                "selected held-out covariance", study._sample_covariance(selected_test)
            )
            for imodel, model_name in enumerate(EVALUATED_MODELS):
                covariance = models[model_name][np.ix_(selected, selected)]
                metrics, quadratic[model_name] = covariance_fold_metrics(
                    covariance,
                    selected_test,
                    k_bins=inputs.k_bins[selected],
                    nbootstrap=nbootstrap,
                    rng=np.random.default_rng(seed + 10000 * ifold + 100 * ikmax + imodel),
                )
                records.append(
                    {
                        "fold": ifold,
                        "ncalibration": int(train_indices.size),
                        "ntest": int(test_indices.size),
                        "kmax": kmax,
                        "model": model_name,
                        **metrics,
                    }
                )
            paired_records.append(
                {
                    "fold": ifold,
                    "ncalibration": int(train_indices.size),
                    "ntest": int(test_indices.size),
                    "kmax": kmax,
                    "dimension": int(selected.size),
                    **paired_chi2_metrics(
                        quadratic["diagonal_smooth"],
                        quadratic["spiked_diagonal"],
                        dimension=selected.size,
                        nbootstrap=nbootstrap,
                        rng=np.random.default_rng(seed + 50000 + 1000 * ifold + ikmax),
                    ),
                }
            )
            if derivatives is not None:
                projected, joint, comparisons = projection_records(
                    models,
                    test_values,
                    selected_reference,
                    derivatives,
                    selected,
                    fold=ifold,
                    kmax=kmax,
                    nbootstrap=nbootstrap,
                    rng=np.random.default_rng(seed + 90000 + 1000 * ifold + ikmax),
                )
                parameter_rows.extend(projected)
                joint_rows.extend(joint)
                comparison_rows.append(
                    {
                        "fold": ifold,
                        "kmax": kmax,
                        "comparisons": comparisons,
                    }
                )
    aggregates = _aggregate(records, ("kmax", "model"))
    verdicts = (
        _projection_verdict(
            parameter_rows,
            joint_rows,
            comparison_rows,
            kmax_values=kmax_values,
        )
        if derivatives is not None
        else []
    )
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "covariance_metrics.csv"
    paired_path = output_root / "paired_chi2.csv"
    parameter_path = output_root / "parameter_projection.csv"
    joint_path = output_root / "joint_coverage.csv"
    figure_path = output_root / "covariance_inference_assessment.png"
    summary_path = output_root / "summary.json"
    _write_csv(metrics_path, records)
    _write_csv(paired_path, paired_records)
    _write_csv(parameter_path, parameter_rows)
    _write_csv(joint_path, joint_rows)
    _plot_summary(figure_path, records)
    summary = {
        "configuration": {
            "data_root": str(Path(data_root).resolve()),
            "nrealizations": int(inputs.vectors.shape[0]),
            "fold_sizes": [int(fold.size) for fold in folds],
            "test_sizes": [int(inputs.vectors.shape[0] - fold.size) for fold in folds],
            "seed": int(seed),
            "nbootstrap": int(nbootstrap),
            "kmin_h_mpc": float(kmin),
            "kmax_values_h_mpc": list(kmax_values),
            "full_vector_size": int(inputs.vectors.shape[1]),
            "models": list(EVALUATED_MODELS),
            "derivative_archive": (
                str(Path(derivative_archive).resolve())
                if derivative_archive is not None
                else None
            ),
        },
        "covariance_fold_metrics": records,
        "covariance_aggregated_metrics": aggregates,
        "paired_chi2_metrics": paired_records,
        "cosmological_projection": {
            "status": "complete" if derivatives is not None else "pending_matched_derivatives",
            "parameter_records": parameter_rows,
            "joint_coverage_records": joint_rows,
            "comparisons": comparison_rows,
            "diagonal_sufficiency_verdicts": verdicts,
        },
        "outputs": {
            "covariance_metrics": str(metrics_path.resolve()),
            "paired_chi2": str(paired_path.resolve()),
            "parameter_projection": (
                str(parameter_path.resolve()) if parameter_rows else None
            ),
            "joint_coverage": str(joint_path.resolve()) if joint_rows else None,
            "figure": str(figure_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=study.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kmin", type=float, default=0.015)
    parser.add_argument(
        "--kmax-values", nargs="+", type=float, default=DEFAULT_KMAX_VALUES
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--nbootstrap", type=int, default=500)
    parser.add_argument("--derivative-archive", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    summary = run_assessment(**vars(build_parser().parse_args(argv)))
    projection = summary["cosmological_projection"]
    print(
        f"Wrote covariance assessment to {summary['outputs']['figure']}; "
        f"cosmological projection: {projection['status']}"
    )


if __name__ == "__main__":
    main()
