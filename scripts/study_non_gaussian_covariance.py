"""Study compressible non-Gaussian corrections to the density-split covariance.

The measured-input disconnected Gaussian covariance is treated as a fixed,
full-rank target.  Corrections are learned from subsets of a calibration pool
and evaluated against a disjoint test pool.  The primary data vector is
``(P_Q1m, P_Q2m, P_Q4m, P_Q5m)`` for ell=0,2 and the configured k range.

The candidate estimators are intentionally agnostic about the trispectrum:

``gaussian``
    The measured-spectrum disconnected Gaussian target.
``diagonal_smooth``
    Smooth blockwise variance rescaling of the Gaussian target.
``linear_shrink_gaussian``
    Parameter-free linear shrinkage of the empirical covariance toward the
    Gaussian target in whitened space.
``linear_shrink_diagonal``
    The same shrinkage after fitting the smooth diagonal correction.
``spiked_diagonal``
    Smooth diagonal correction plus population-spike estimates for sample
    eigenvalues outside the Marchenko--Pastur bulk.

The script writes a self-contained numerical summary, per-repeat convergence
records, final full-ensemble candidate covariances, and diagnostic figures.
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
from scipy.signal import savgol_filter

try:
    from scripts.density_split_gaussian_covariance import (
        periodic_shell_angular_moments,
    )
    from scripts.validate_density_split_covariance import (
        DEFAULT_DATA_ROOT,
        PowerMultipoles,
        build_hybrid_covariance,
        load_rsd_ensemble,
    )
except ModuleNotFoundError:
    from density_split_gaussian_covariance import periodic_shell_angular_moments
    from validate_density_split_covariance import (
        DEFAULT_DATA_ROOT,
        PowerMultipoles,
        build_hybrid_covariance,
        load_rsd_ensemble,
    )


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_DIR
    / "outputs"
    / "model_validation"
    / "non_gaussian_covariance_study"
)
INPUT_ELLS = (0, 2, 4)
OUTPUT_ELLS = (0, 2)
QUANTILES = (1, 2, 4, 5)
MODEL_NAMES = (
    "gaussian",
    "diagonal_smooth",
    "linear_shrink_gaussian",
    "linear_shrink_diagonal",
    "spiked_diagonal",
    "empirical",
)


@dataclass(frozen=True)
class MatrixRoot:
    """Symmetric square root and inverse square root of an SPD matrix."""

    root: np.ndarray
    inverse_root: np.ndarray


@dataclass(frozen=True)
class FittedCovariance:
    """A fitted covariance and compact estimator diagnostics."""

    covariance: np.ndarray
    shrinkage: float | None = None
    spike_count: int | None = None
    largest_population_spike: float | None = None


def _matrix_root(covariance: np.ndarray) -> MatrixRoot:
    covariance = np.asarray(covariance, dtype="f8")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = np.finfo("f8").eps * covariance.shape[0] * eigenvalues[-1]
    if eigenvalues[0] <= tolerance:
        raise ValueError(
            "covariance must be numerically positive definite; minimum "
            f"eigenvalue={eigenvalues[0]:.6e}, tolerance={tolerance:.6e}"
        )
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    return MatrixRoot(root=0.5 * (root + root.T), inverse_root=0.5 * (inverse_root + inverse_root.T))


def _sample_covariance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="f8")
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("values must contain at least two realization rows")
    return np.atleast_2d(np.cov(values, rowvar=False, ddof=1))


def _observable_indices(nquantiles: int, ninput_ells: int, nk: int) -> np.ndarray:
    indices = []
    for iq in range(nquantiles):
        for iell, ell in enumerate(INPUT_ELLS):
            if ell in OUTPUT_ELLS:
                start = (iq * ninput_ells + iell) * nk
                indices.extend(range(start, start + nk))
    return np.asarray(indices, dtype="i8")


def _smooth_diagonal_target(
    sample: np.ndarray,
    target: np.ndarray,
    *,
    nblocks: int,
    nk: int,
    window: int = 7,
    polynomial_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``D target D`` using smoothed log variance ratios."""
    sample_variance = np.diag(sample)
    target_variance = np.diag(target)
    if np.any(sample_variance <= 0.0) or np.any(target_variance <= 0.0):
        raise ValueError("sample and target variances must be positive")
    log_ratio = np.log(sample_variance / target_variance).reshape(nblocks, nk)
    smoothed = savgol_filter(
        log_ratio,
        window_length=window,
        polyorder=polynomial_order,
        axis=-1,
        mode="interp",
    )
    # Prevent isolated small-sample edge fits from generating pathological
    # targets while retaining corrections much larger than those observed.
    ratio = np.exp(np.clip(smoothed, np.log(0.25), np.log(4.0))).reshape(-1)
    scale = np.sqrt(ratio)
    covariance = target * scale[:, None] * scale[None, :]
    return 0.5 * (covariance + covariance.T), ratio


def _whitened_sample(values: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, MatrixRoot]:
    """Return an approximately unbiased sample covariance whitened by target."""
    values = np.asarray(values, dtype="f8")
    centered = values - values.mean(axis=0, keepdims=True)
    nrealizations = values.shape[0]
    root = _matrix_root(target)
    whitened = centered @ root.inverse_root
    covariance = whitened.T @ whitened / (nrealizations - 1)
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, root


def _linear_shrinkage_intensity(values: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, MatrixRoot]:
    """Estimate Frobenius-optimal shrinkage toward a known whitened identity.

    The variance of the sample covariance is estimated from the realization
    outer products.  Centered rows are rescaled so their maximum-likelihood
    covariance equals the usual ``ddof=1`` estimator.
    """
    values = np.asarray(values, dtype="f8")
    nrealizations = values.shape[0]
    centered = values - values.mean(axis=0, keepdims=True)
    root = _matrix_root(target)
    whitened = centered @ root.inverse_root
    whitened *= np.sqrt(nrealizations / (nrealizations - 1))
    sample = whitened.T @ whitened / nrealizations
    sample = 0.5 * (sample + sample.T)
    row_norm_squared = np.einsum("ij,ij->i", whitened, whitened)
    variance_error = (
        np.sum(row_norm_squared**2)
        - nrealizations * np.sum(sample * sample)
    ) / nrealizations**2
    distance = np.sum((sample - np.eye(sample.shape[0])) ** 2)
    shrinkage = 1.0 if distance <= 0.0 else float(np.clip(variance_error / distance, 0.0, 1.0))
    return shrinkage, sample, root


def _fit_linear_shrinkage(values: np.ndarray, target: np.ndarray) -> FittedCovariance:
    shrinkage, sample, root = _linear_shrinkage_intensity(values, target)
    whitened = (1.0 - shrinkage) * sample
    whitened[np.diag_indices_from(whitened)] += shrinkage
    covariance = root.root @ whitened @ root.root
    return FittedCovariance(
        covariance=0.5 * (covariance + covariance.T),
        shrinkage=shrinkage,
    )


def _population_spikes(sample_eigenvalues: np.ndarray, aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """Return indices and de-biased population spikes outside the MP bulk."""
    upper_edge = (1.0 + np.sqrt(aspect)) ** 2
    selected_indices: list[int] = []
    population_values: list[float] = []
    for index, value in enumerate(sample_eigenvalues):
        if value > upper_edge:
            center = value + 1.0 - aspect
            discriminant = center * center - 4.0 * value
            if discriminant > 0.0:
                selected_indices.append(index)
                population_values.append(0.5 * (center + np.sqrt(discriminant)))
    if aspect < 1.0:
        lower_edge = (1.0 - np.sqrt(aspect)) ** 2
        for index, value in enumerate(sample_eigenvalues):
            if 0.0 < value < lower_edge:
                center = value + 1.0 - aspect
                discriminant = center * center - 4.0 * value
                if discriminant > 0.0:
                    selected_indices.append(index)
                    population_values.append(0.5 * (center - np.sqrt(discriminant)))
    if not selected_indices:
        return np.empty(0, dtype="i8"), np.empty(0, dtype="f8")
    order = np.argsort(selected_indices)
    return (
        np.asarray(selected_indices, dtype="i8")[order],
        np.asarray(population_values, dtype="f8")[order],
    )


def _fit_spiked(
    values: np.ndarray,
    target: np.ndarray,
    *,
    max_modes: int | None = None,
) -> FittedCovariance:
    sample, root = _whitened_sample(values, target)
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    aspect = sample.shape[0] / (values.shape[0] - 1)
    indices, population = _population_spikes(eigenvalues, aspect)
    if indices.size and max_modes is not None:
        strength_order = np.argsort(np.abs(np.log(population)))[::-1][:max_modes]
        indices = indices[strength_order]
        population = population[strength_order]
    whitened = np.eye(sample.shape[0])
    if indices.size:
        modes = eigenvectors[:, indices]
        whitened += (modes * (population - 1.0)) @ modes.T
    covariance = root.root @ whitened @ root.root
    # The target diagonal was estimated separately with a smooth, blockwise
    # estimator.  Use the spikes only for correlation structure; otherwise a
    # broad positive mode double-counts the variance enhancement that is
    # already present in the diagonal target.
    scale = np.sqrt(np.diag(target) / np.diag(covariance))
    covariance *= scale[:, None] * scale[None, :]
    return FittedCovariance(
        covariance=0.5 * (covariance + covariance.T),
        spike_count=int(indices.size),
        largest_population_spike=(float(np.max(population)) if population.size else 1.0),
    )


def _spike_basis(values: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return MP-selected whitened eigenmodes ordered by spike strength."""
    sample, _ = _whitened_sample(values, target)
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    aspect = sample.shape[0] / (values.shape[0] - 1)
    indices, population = _population_spikes(eigenvalues, aspect)
    if not indices.size:
        return np.empty((sample.shape[0], 0), dtype="f8"), population
    strength_order = np.argsort(np.abs(np.log(population)))[::-1]
    return eigenvectors[:, indices[strength_order]], population[strength_order]


def _fit_models(
    values: np.ndarray,
    gaussian: np.ndarray,
    *,
    nblocks: int,
    nk: int,
) -> dict[str, FittedCovariance]:
    sample = _sample_covariance(values)
    diagonal, _ = _smooth_diagonal_target(
        sample,
        gaussian,
        nblocks=nblocks,
        nk=nk,
    )
    models = {
        "gaussian": FittedCovariance(gaussian),
        "diagonal_smooth": FittedCovariance(diagonal),
        "linear_shrink_gaussian": _fit_linear_shrinkage(values, gaussian),
        "linear_shrink_diagonal": _fit_linear_shrinkage(values, diagonal),
        "spiked_diagonal": _fit_spiked(values, diagonal),
    }
    if values.shape[0] > values.shape[1] + 2:
        models["empirical"] = FittedCovariance(sample)
    return models


def _correlation(covariance: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.diag(covariance))
    return covariance / sigma[:, None] / sigma[None, :]


def _logdet(covariance: np.ndarray) -> float:
    cholesky = np.linalg.cholesky(covariance)
    return float(2.0 * np.sum(np.log(np.diag(cholesky))))


def _evaluation_metrics(
    model: np.ndarray,
    reference: np.ndarray,
    *,
    k_bins: np.ndarray,
    q5_high_k: np.ndarray,
) -> dict[str, float]:
    size = model.shape[0]
    factor = linalg.cho_factor(model, lower=True, check_finite=False)
    solved = linalg.cho_solve(factor, reference, check_finite=False)
    chi2_ratio = float(np.trace(solved) / size)
    kl_per_element = 0.5 * (
        np.trace(solved)
        - size
        + _logdet(model)
        - _logdet(reference)
    ) / size
    sigma_ratio = np.sqrt(np.diag(model) / np.diag(reference))
    model_correlation = _correlation(model)
    reference_correlation = _correlation(reference)
    cross_shell = k_bins[:, None] != k_bins[None, :]
    off_diagonal = ~np.eye(size, dtype=bool)
    return {
        "kl_per_element": float(kl_per_element),
        "chi2_ratio": chi2_ratio,
        "sigma_log_rms": float(np.sqrt(np.mean(np.log(sigma_ratio) ** 2))),
        "sigma_ratio_median": float(np.median(sigma_ratio)),
        "correlation_rms": float(
            np.sqrt(np.mean((model_correlation[off_diagonal] - reference_correlation[off_diagonal]) ** 2))
        ),
        "cross_shell_correlation_rms": float(
            np.sqrt(np.mean((model_correlation[cross_shell] - reference_correlation[cross_shell]) ** 2))
        ),
        "q5_high_k_sigma_ratio_median": float(np.median(sigma_ratio[q5_high_k])),
        "q5_high_k_sigma_ratio_minimum": float(np.min(sigma_ratio[q5_high_k])),
        "q5_high_k_sigma_ratio_maximum": float(np.max(sigma_ratio[q5_high_k])),
    }


def _aggregate_records(records: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for record in records:
        groups.setdefault((int(record["ncalibration"]), str(record["model"])), []).append(record)
    aggregated = []
    nonmetric = {"ncalibration", "repeat", "model"}
    for (ncalibration, model), rows in sorted(groups.items()):
        result: dict[str, object] = {
            "ncalibration": ncalibration,
            "model": model,
            "repeats": len(rows),
        }
        metric_names = sorted(set(rows[0]) - nonmetric)
        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows if row[metric] is not None], dtype="f8")
            if not values.size:
                result[f"{metric}_median"] = None
                result[f"{metric}_p16"] = None
                result[f"{metric}_p84"] = None
            else:
                result[f"{metric}_median"] = float(np.median(values))
                result[f"{metric}_p16"] = float(np.percentile(values, 16))
                result[f"{metric}_p84"] = float(np.percentile(values, 84))
        aggregated.append(result)
    return aggregated


def _crossfit_study(
    vectors: np.ndarray,
    gaussian: np.ndarray,
    folds: Sequence[np.ndarray],
    *,
    nblocks: int,
    nk: int,
    k_bins: np.ndarray,
    q5_high_k: np.ndarray,
) -> tuple[list[dict], dict]:
    """Fit on each 1/3 fold and each complementary 2/3, then test disjointly."""
    records: list[dict] = []
    all_indices = np.arange(vectors.shape[0])
    for ifold, fold in enumerate(folds):
        configurations = (
            (fold, np.setdiff1d(all_indices, fold, assume_unique=False)),
            (np.setdiff1d(all_indices, fold, assume_unique=False), fold),
        )
        for train_indices, test_indices in configurations:
            reference = _sample_covariance(vectors[test_indices])
            train_sample = _sample_covariance(vectors[train_indices])
            train_diagonal, _ = _smooth_diagonal_target(
                train_sample,
                gaussian,
                nblocks=nblocks,
                nk=nk,
            )
            fitted = _fit_models(
                vectors[train_indices],
                gaussian,
                nblocks=nblocks,
                nk=nk,
            )
            for model_name, model in fitted.items():
                records.append(
                    {
                        "ncalibration": int(train_indices.size),
                        "repeat": ifold,
                        "model": model_name,
                        "shrinkage": model.shrinkage,
                        "spike_count": model.spike_count,
                        "largest_population_spike": model.largest_population_spike,
                        **_evaluation_metrics(
                            model.covariance,
                            reference,
                            k_bins=k_bins,
                            q5_high_k=q5_high_k,
                        ),
                    }
                )
            for rank in (1, 3, 5):
                model = _fit_spiked(
                    vectors[train_indices],
                    train_diagonal,
                    max_modes=rank,
                )
                records.append(
                    {
                        "ncalibration": int(train_indices.size),
                        "repeat": ifold,
                        "model": f"spiked_diagonal_rank_{rank}",
                        "shrinkage": model.shrinkage,
                        "spike_count": model.spike_count,
                        "largest_population_spike": model.largest_population_spike,
                        **_evaluation_metrics(
                            model.covariance,
                            reference,
                            k_bins=k_bins,
                            q5_high_k=q5_high_k,
                        ),
                    }
                )

    full_sample = _sample_covariance(vectors)
    common_diagonal, _ = _smooth_diagonal_target(
        full_sample,
        gaussian,
        nblocks=nblocks,
        nk=nk,
    )
    bases = []
    population_spikes = []
    for fold in folds:
        basis, population = _spike_basis(vectors[fold], common_diagonal)
        bases.append(basis)
        population_spikes.append(population)
    overlap_summary: dict[str, object] = {
        "spike_counts": [int(basis.shape[1]) for basis in bases],
        "largest_population_spikes": [
            float(np.max(values)) if values.size else 1.0
            for values in population_spikes
        ],
        "subspace_overlaps": {},
    }
    minimum_count = min(basis.shape[1] for basis in bases)
    requested_ranks = [rank for rank in (1, 3, 5, minimum_count) if 0 < rank <= minimum_count]
    for rank in sorted(set(requested_ranks)):
        pairwise = []
        for ia in range(len(bases)):
            for ib in range(ia + 1, len(bases)):
                singular_values = np.linalg.svd(
                    bases[ia][:, :rank].T @ bases[ib][:, :rank],
                    compute_uv=False,
                )
                pairwise.append(float(np.mean(singular_values**2)))
        overlap_summary["subspace_overlaps"][str(rank)] = {
            "pairwise": pairwise,
            "mean": float(np.mean(pairwise)),
            "minimum": float(np.min(pairwise)),
            "random_subspace_expectation": float(rank / vectors.shape[1]),
        }
    return records, overlap_summary


def _repeat_count(ncalibration: int, pool_size: int) -> int:
    if ncalibration == pool_size:
        return 1
    if ncalibration <= 100:
        return 40
    if ncalibration <= 250:
        return 30
    return 20


def _write_records(path: Path, records: Sequence[dict]) -> None:
    fieldnames = sorted({key for row in records for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _plot_results(
    path: Path,
    aggregated: Sequence[dict],
    *,
    k: np.ndarray,
    labels: np.ndarray,
    gaussian: np.ndarray,
    reference: np.ndarray,
    final_models: dict[str, FittedCovariance],
) -> None:
    colors = {
        "gaussian": "black",
        "diagonal_smooth": "#E69F00",
        "linear_shrink_gaussian": "#56B4E9",
        "linear_shrink_diagonal": "#0072B2",
        "spiked_diagonal": "#009E73",
        "empirical": "#CC79A7",
    }
    display_names = {
        "gaussian": "Gaussian",
        "diagonal_smooth": "smooth diagonal",
        "linear_shrink_gaussian": "shrink to Gaussian",
        "linear_shrink_diagonal": "shrink to diagonal",
        "spiked_diagonal": "spiked diagonal",
        "empirical": "raw empirical",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for model in MODEL_NAMES:
        rows = [row for row in aggregated if row["model"] == model]
        if not rows:
            continue
        x = np.asarray([row["ncalibration"] for row in rows])
        median = np.asarray([row["kl_per_element_median"] for row in rows])
        lower = np.asarray([row["kl_per_element_p16"] for row in rows])
        upper = np.asarray([row["kl_per_element_p84"] for row in rows])
        ax.plot(x, median, marker="o", color=colors[model], label=display_names[model])
        ax.fill_between(x, lower, upper, color=colors[model], alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("calibration mocks")
    ax.set_ylabel("held-out Gaussian KL / element")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for model in MODEL_NAMES:
        rows = [row for row in aggregated if row["model"] == model]
        if not rows:
            continue
        x = np.asarray([row["ncalibration"] for row in rows])
        median = np.asarray([row["q5_high_k_sigma_ratio_median_median"] for row in rows])
        lower = np.asarray([row["q5_high_k_sigma_ratio_median_p16"] for row in rows])
        upper = np.asarray([row["q5_high_k_sigma_ratio_median_p84"] for row in rows])
        ax.plot(x, median, marker="o", color=colors[model], label=display_names[model])
        ax.fill_between(x, lower, upper, color=colors[model], alpha=0.15)
    ax.axhline(1.0, color="0.5", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("calibration mocks")
    ax.set_ylabel(r"Q5 $\ell=0$, high-$k$ $\sigma_{\rm model}/\sigma_{\rm test}$")
    ax.grid(alpha=0.25)

    q5_ell0 = np.asarray([label.startswith("Pq5m_ell0") for label in labels])
    ax = axes[1, 0]
    reference_sigma = np.sqrt(np.diag(reference))[q5_ell0]
    for model in ("gaussian", "diagonal_smooth", "linear_shrink_diagonal", "spiked_diagonal"):
        covariance = gaussian if model == "gaussian" else final_models[model].covariance
        ratio = np.sqrt(np.diag(covariance))[q5_ell0] / reference_sigma
        ax.plot(k, ratio, marker="o", ms=3, color=colors[model], label=display_names[model])
    ax.axhline(1.0, color="0.5", lw=1)
    ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"Q5 $\ell=0$ $\sigma_{\rm model}/\sigma_{\rm test}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    model = final_models["spiked_diagonal"].covariance
    residual = _correlation(reference) - _correlation(model)
    image = ax.imshow(residual, origin="lower", cmap="RdBu_r", vmin=-0.12, vmax=0.12)
    ax.set_title("held-out minus spiked-diagonal correlation")
    ax.set_xlabel("data-vector index")
    ax.set_ylabel("data-vector index")
    figure.colorbar(image, ax=ax, label=r"$\Delta r_{ij}$")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_study(
    *,
    data_root: Path,
    output_root: Path,
    kmin: float,
    kmax: float,
    calibration_pool_size: int,
    seed: int,
    calibration_sizes: Sequence[int],
) -> dict:
    ensemble = load_rsd_ensemble(
        data_root,
        quantiles=QUANTILES,
        ells=INPUT_ELLS,
        data_vector=("pqm",),
        kmin=kmin,
        kmax=kmax,
    )
    nrealizations = len(ensemble.realization_ids)
    if not 2 <= calibration_pool_size < nrealizations:
        raise ValueError("calibration_pool_size must leave at least one held-out mock")
    sizes = tuple(sorted(set(int(value) for value in calibration_sizes)))
    if not sizes or sizes[0] < 2 or sizes[-1] > calibration_pool_size:
        raise ValueError("calibration sizes must lie between 2 and calibration_pool_size")

    nk = ensemble.k.size
    nquantiles = len(QUANTILES)
    indices = _observable_indices(nquantiles, len(INPUT_ELLS), nk)
    vectors = ensemble.vectors(("pqm",))[:, indices]
    nblocks = nquantiles * len(OUTPUT_ELLS)

    geometry = periodic_shell_angular_moments(
        ensemble.k_edges,
        ensemble.nmodes,
        ells=INPUT_ELLS,
        boxsize=1000.0,
        meshsize=256,
        los=ensemble.los,
    )
    hybrid = build_hybrid_covariance(
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
    gaussian = hybrid.covariance[np.ix_(indices, indices)]
    labels = np.asarray(hybrid.labels)[indices]
    pair_labels = np.repeat([f"Pq{quantile}m" for quantile in QUANTILES], len(OUTPUT_ELLS) * nk)
    element_ells = np.tile(np.repeat(OUTPUT_ELLS, nk), nquantiles)
    k_bins = np.tile(np.arange(nk, dtype="i8"), nblocks)
    element_k = np.tile(ensemble.k, nblocks)
    q5_high_k = (pair_labels == "Pq5m") & (element_ells == 0) & (element_k >= 0.19)
    if not np.any(q5_high_k):
        raise ValueError("the selected range contains no Q5 ell=0 bins above k=0.19")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(nrealizations)
    calibration_pool = permutation[:calibration_pool_size]
    test_indices = permutation[calibration_pool_size:]
    test_covariance = _sample_covariance(vectors[test_indices])
    full_covariance = _sample_covariance(vectors)

    records: list[dict] = []
    for ncalibration in sizes:
        repeats = _repeat_count(ncalibration, calibration_pool_size)
        for repeat in range(repeats):
            if ncalibration == calibration_pool_size:
                selected = calibration_pool
            else:
                selected = rng.choice(calibration_pool, size=ncalibration, replace=False)
            fitted = _fit_models(
                vectors[selected],
                gaussian,
                nblocks=nblocks,
                nk=nk,
            )
            for model_name, model in fitted.items():
                metrics = _evaluation_metrics(
                    model.covariance,
                    test_covariance,
                    k_bins=k_bins,
                    q5_high_k=q5_high_k,
                )
                records.append(
                    {
                        "ncalibration": ncalibration,
                        "repeat": repeat,
                        "model": model_name,
                        "shrinkage": model.shrinkage,
                        "spike_count": model.spike_count,
                        "largest_population_spike": model.largest_population_spike,
                        **metrics,
                    }
                )

    aggregated = _aggregate_records(records)
    crossfit_folds = tuple(np.asarray(fold, dtype="i8") for fold in np.array_split(permutation, 3))
    crossfit_records, mode_stability = _crossfit_study(
        vectors,
        gaussian,
        crossfit_folds,
        nblocks=nblocks,
        nk=nk,
        k_bins=k_bins,
        q5_high_k=q5_high_k,
    )
    crossfit_aggregated = _aggregate_records(crossfit_records)
    validation_models = _fit_models(
        vectors[calibration_pool],
        gaussian,
        nblocks=nblocks,
        nk=nk,
    )
    fixed_split_metrics = {
        name: _evaluation_metrics(
            fitted.covariance,
            test_covariance,
            k_bins=k_bins,
            q5_high_k=q5_high_k,
        )
        | {
            "shrinkage": fitted.shrinkage,
            "spike_count": fitted.spike_count,
            "largest_population_spike": fitted.largest_population_spike,
        }
        for name, fitted in validation_models.items()
    }
    final_models = _fit_models(
        vectors,
        gaussian,
        nblocks=nblocks,
        nk=nk,
    )
    final_diagonal, final_variance_ratio = _smooth_diagonal_target(
        full_covariance,
        gaussian,
        nblocks=nblocks,
        nk=nk,
    )
    final_spike_modes, final_population_spikes = _spike_basis(
        vectors,
        final_diagonal,
    )
    final_model_diagnostics = {
        name: {
            "shrinkage": fitted.shrinkage,
            "spike_count": fitted.spike_count,
            "largest_population_spike": fitted.largest_population_spike,
        }
        for name, fitted in final_models.items()
    }

    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "convergence_records.csv"
    crossfit_records_path = output_root / "crossfit_records.csv"
    summary_path = output_root / "summary.json"
    covariance_path = output_root / "covariances.npz"
    figure_path = output_root / "non_gaussian_covariance_study.png"
    _write_records(records_path, records)
    _write_records(crossfit_records_path, crossfit_records)
    np.savez_compressed(
        covariance_path,
        empirical_full=full_covariance,
        empirical_test=test_covariance,
        gaussian=gaussian,
        diagonal_smooth=final_models["diagonal_smooth"].covariance,
        linear_shrink_gaussian=final_models["linear_shrink_gaussian"].covariance,
        linear_shrink_diagonal=final_models["linear_shrink_diagonal"].covariance,
        spiked_diagonal=final_models["spiked_diagonal"].covariance,
        recommended=final_models["spiked_diagonal"].covariance,
        smooth_diagonal_target=final_diagonal,
        smooth_variance_ratio=final_variance_ratio,
        spike_modes_whitened=final_spike_modes,
        spike_population_eigenvalues=final_population_spikes,
        labels=labels,
        pair_labels=pair_labels,
        ells=element_ells,
        k_bins=k_bins,
        k_for_element=element_k,
        k=ensemble.k,
        realization_ids=np.asarray(ensemble.realization_ids, dtype="i8"),
        calibration_pool_indices=calibration_pool,
        test_indices=test_indices,
        crossfit_fold_0=crossfit_folds[0],
        crossfit_fold_1=crossfit_folds[1],
        crossfit_fold_2=crossfit_folds[2],
    )
    _plot_results(
        figure_path,
        aggregated,
        k=ensemble.k,
        labels=labels,
        gaussian=gaussian,
        reference=test_covariance,
        final_models=validation_models,
    )
    summary = {
        "configuration": {
            "data_root": str(ensemble.data_root),
            "nrealizations": nrealizations,
            "calibration_pool_size": calibration_pool_size,
            "test_size": int(test_indices.size),
            "seed": seed,
            "quantiles": list(QUANTILES),
            "input_ells": list(INPUT_ELLS),
            "output_ells": list(OUTPUT_ELLS),
            "nk": nk,
            "vector_size": int(vectors.shape[1]),
            "k_range_h_mpc": [float(ensemble.k[0]), float(ensemble.k[-1])],
            "q5_high_k_min_h_mpc": 0.19,
            "calibration_sizes": list(sizes),
        },
        "baseline_full_vs_test": _evaluation_metrics(
            gaussian,
            test_covariance,
            k_bins=k_bins,
            q5_high_k=q5_high_k,
        ),
        "aggregated_convergence": aggregated,
        "fixed_split_500_to_1000_metrics": fixed_split_metrics,
        "crossfit_aggregated": crossfit_aggregated,
        "crossfit_mode_stability": mode_stability,
        "full_ensemble_candidate_diagnostics": final_model_diagnostics,
        "recommended_template": {
            "model": "spiked_diagonal",
            "spike_count": int(final_population_spikes.size),
            "largest_population_spike": float(np.max(final_population_spikes)),
            "basis": "eigenmodes whitened by the smooth-diagonal Gaussian target",
            "diagonal_preserved_after_spike_update": True,
        },
        "outputs": {
            "records": str(records_path.resolve()),
            "crossfit_records": str(crossfit_records_path.resolve()),
            "covariances": str(covariance_path.resolve()),
            "figure": str(figure_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kmin", type=float, default=0.015)
    parser.add_argument("--kmax", type=float, default=0.25)
    parser.add_argument("--calibration-pool-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--calibration-sizes",
        type=int,
        nargs="+",
        default=(20, 30, 50, 75, 100, 150, 250, 350, 500),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    summary = run_study(**vars(build_parser().parse_args(argv)))
    baseline = summary["baseline_full_vs_test"]
    print(
        "Gaussian baseline: "
        f"KL/element={baseline['kl_per_element']:.6g}, "
        f"Q5 high-k sigma ratio={baseline['q5_high_k_sigma_ratio_median']:.4f}"
    )
    final = summary["fixed_split_500_to_1000_metrics"]
    for name in ("diagonal_smooth", "linear_shrink_gaussian", "linear_shrink_diagonal", "spiked_diagonal"):
        metrics = final[name]
        print(
            f"{name}: KL/element={metrics['kl_per_element']:.6g}, "
            f"Q5 high-k sigma ratio={metrics['q5_high_k_sigma_ratio_median']:.4f}"
        )
    print(f"summary: {Path(summary['outputs']['records']).parent / 'summary.json'}")


if __name__ == "__main__":
    main()
