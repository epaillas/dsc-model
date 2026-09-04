"""Build and validate density-split RSD multipole covariances.

There are two public construction paths:

``build_hybrid_covariance``
    Build the disconnected Gaussian covariance from supplied total ``Pmm``,
    ``Pqm``, and ``Pqq`` multipoles. The inputs may come from measurements,
    emulators, perturbation theory, or any mixture of those sources.

``build_pt_covariance``
    Predict all three sets of multipoles with the Gaussian-threshold RSD
    model, then assemble the same covariance without measured spectra.

The CLI loads every realization found below ``--data-root`` and compares both
models with the empirical covariance. ``--data-vector`` accepts one or more of
``pmm``, ``pqm``, and ``pqq`` in the requested data-vector order.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
from scipy.signal import savgol_filter

try:
    from scripts.density_split_gaussian_covariance import (
        PeriodicShellAngularMoments,
        periodic_shell_angular_moments,
        validate_covariance,
    )
    from scripts.density_split_pt_covariance import (
        GaussianThresholdRSDSpectra,
        linear_gaussian_threshold_rsd_spectra,
    )
except ModuleNotFoundError:
    from density_split_gaussian_covariance import (
        PeriodicShellAngularMoments,
        periodic_shell_angular_moments,
        validate_covariance,
    )
    from density_split_pt_covariance import (
        GaussianThresholdRSDSpectra,
        linear_gaussian_threshold_rsd_spectra,
    )


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_DIR / "data" / "clustering_measurements" / "z0.5" / "rsd_z"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_DIR / "outputs" / "model_validation" / "rsd_covariance_validation"
)
DEFAULT_QUANTILES = (1, 2, 4, 5)
DEFAULT_ELLS = (0, 2, 4)
DEFAULT_KMIN = 0.015
DEFAULT_KMAX = 0.25
STATISTICS = ("pmm", "pqm", "pqq")
DEFAULT_DATA_VECTOR = ("pmm", "pqm")
ARCHIVE_SCHEMA_VERSION = 2
MODEL_NAMES = ("hybrid", "pt")
MODEL_PROVENANCE = {
    "hybrid": "disconnected Gaussian covariance with all measured total multipoles",
    "pt": "disconnected Gaussian covariance with fully analytic PT multipoles",
}
_REALIZATION = re.compile(r"realization_(\d+)")


@dataclass(frozen=True)
class PowerMultipoles:
    """Total field multipoles used inside Gaussian contractions.

    Shapes are ``(nell, nk)`` for ``pmm``, ``(nq, nell, nk)`` for
    ``pqm``, and ``(nq, nq, nell, nk)`` for ``pqq``. White-noise power must
    already be included in the monopoles. In particular, the PT ``pqq``
    prediction already contains categorical lattice noise.
    """

    quantiles: tuple[int, ...]
    ells: tuple[int, ...]
    pmm: np.ndarray
    pqm: np.ndarray
    pqq: np.ndarray


@dataclass(frozen=True)
class CovarianceResult:
    """A covariance matrix and the ordering of its data vector."""

    covariance: np.ndarray
    labels: tuple[str, ...]
    pairs: tuple[tuple[int, int], ...]
    spectra: PowerMultipoles


@dataclass(frozen=True)
class CovarianceSelection:
    """A consistently selected empirical/model covariance data vector."""

    empirical: np.ndarray
    models: Mapping[str, np.ndarray]
    labels: np.ndarray
    statistics: np.ndarray
    pair_labels: np.ndarray
    quantile_a: np.ndarray
    quantile_b: np.ndarray
    ells: np.ndarray
    k_bins: np.ndarray
    k: np.ndarray
    nrealizations: int
    ordering: str
    model_provenance: Mapping[str, str]

    @property
    def size(self) -> int:
        return int(self.labels.size)

    @property
    def block_labels(self) -> np.ndarray:
        return np.asarray(
            [f"{pair}, ell={ell}" for pair, ell in zip(self.pair_labels, self.ells)]
        )


@dataclass(frozen=True)
class CovarianceArchive:
    """Self-describing covariance products loaded from ``covariances.npz``."""

    path: Path
    empirical: np.ndarray
    models: Mapping[str, np.ndarray]
    labels: np.ndarray
    statistics: np.ndarray
    pair_labels: np.ndarray
    quantile_a: np.ndarray
    quantile_b: np.ndarray
    element_ells: np.ndarray
    k_bins: np.ndarray
    element_k: np.ndarray
    quantiles: np.ndarray
    ells: np.ndarray
    k: np.ndarray
    k_edges: np.ndarray
    nmodes: np.ndarray
    los: np.ndarray
    realization_ids: np.ndarray
    data_vector: tuple[str, ...]
    ordering: str
    model_provenance: Mapping[str, str]

    @property
    def nrealizations(self) -> int:
        return int(self.realization_ids.size)

    def select(
        self,
        *,
        statistics: Sequence[str] | None = None,
        quantiles: Sequence[int] | None = None,
        pair_labels: Sequence[str] | None = None,
        ells: Sequence[int] | None = None,
        k_range: tuple[float, float] | None = None,
        models: Sequence[str] = MODEL_NAMES,
    ) -> CovarianceSelection:
        """Select covariance rows and columns using archive element metadata."""
        requested_models = tuple(str(model) for model in models)
        missing_models = sorted(set(requested_models) - set(self.models))
        if not requested_models or missing_models:
            raise ValueError(
                "models must be non-empty and available; missing "
                f"{missing_models or 'all requested models'}"
            )

        requested_statistics = (
            tuple(str(statistic) for statistic in statistics)
            if statistics is not None
            else tuple(dict.fromkeys(self.statistics.tolist()))
        )
        missing_statistics = sorted(
            set(requested_statistics) - set(self.statistics.tolist())
        )
        if not requested_statistics or missing_statistics:
            raise ValueError(
                "statistics must be non-empty and available; missing "
                f"{missing_statistics or 'all requested statistics'}"
            )
        mask = np.isin(self.statistics, requested_statistics)

        if quantiles is not None:
            requested_quantiles = tuple(int(quantile) for quantile in quantiles)
            missing_quantiles = sorted(set(requested_quantiles) - set(self.quantiles))
            if not requested_quantiles or missing_quantiles:
                raise ValueError(
                    "quantiles must be non-empty and available; missing "
                    f"{missing_quantiles or 'all requested quantiles'}"
                )
            selected_a = np.isin(self.quantile_a, requested_quantiles)
            selected_b = np.isin(self.quantile_b, requested_quantiles)
            quantile_mask = self.statistics == "pmm"
            quantile_mask |= (self.statistics == "pqm") & (selected_a | selected_b)
            quantile_mask |= (self.statistics == "pqq") & selected_a & selected_b
            mask &= quantile_mask

        if pair_labels is not None:
            requested_pairs = tuple(str(pair) for pair in pair_labels)
            missing_pairs = sorted(set(requested_pairs) - set(self.pair_labels))
            if not requested_pairs or missing_pairs:
                raise ValueError(
                    "pair_labels must be non-empty and available; missing "
                    f"{missing_pairs or 'all requested pairs'}"
                )
            mask &= np.isin(self.pair_labels, requested_pairs)

        if ells is not None:
            requested_ells = tuple(int(ell) for ell in ells)
            missing_ells = sorted(set(requested_ells) - set(self.ells))
            if not requested_ells or missing_ells:
                raise ValueError(
                    "ells must be non-empty and available; missing "
                    f"{missing_ells or 'all requested multipoles'}"
                )
            mask &= np.isin(self.element_ells, requested_ells)

        if k_range is not None:
            if len(k_range) != 2:
                raise ValueError("k_range must be a (kmin, kmax) pair")
            kmin, kmax = (float(value) for value in k_range)
            if not np.isfinite([kmin, kmax]).all() or kmin > kmax:
                raise ValueError("k_range must contain finite values with kmin <= kmax")
            mask &= (self.element_k >= kmin) & (self.element_k <= kmax)

        indices = np.flatnonzero(mask)
        if not indices.size:
            raise ValueError("the requested archive selection contains no elements")
        matrix_index = np.ix_(indices, indices)
        return CovarianceSelection(
            empirical=self.empirical[matrix_index],
            models={
                model: self.models[model][matrix_index] for model in requested_models
            },
            labels=self.labels[indices],
            statistics=self.statistics[indices],
            pair_labels=self.pair_labels[indices],
            quantile_a=self.quantile_a[indices],
            quantile_b=self.quantile_b[indices],
            ells=self.element_ells[indices],
            k_bins=self.k_bins[indices],
            k=self.element_k[indices],
            nrealizations=self.nrealizations,
            ordering=self.ordering,
            model_provenance={
                model: self.model_provenance[model] for model in requested_models
            },
        )


@dataclass(frozen=True)
class RSDEnsemble:
    """Measured RSD multipoles from every available realization."""

    data_root: Path
    realization_ids: tuple[int, ...]
    k: np.ndarray
    k_edges: np.ndarray
    nmodes: np.ndarray
    los: np.ndarray
    matter_noise: np.ndarray
    spectra: PowerMultipoles
    values: np.ndarray

    def vectors(
        self, data_vector: str | Sequence[str] = DEFAULT_DATA_VECTOR
    ) -> np.ndarray:
        """Return realization rows in the same ordering as a model covariance."""
        pairs = observable_pairs(data_vector, len(self.spectra.quantiles))
        blocks = [self.values[:, a, b] for a, b in pairs]
        return np.stack(blocks, axis=1).reshape(len(self.realization_ids), -1)


def _validated_multipoles(spectra: PowerMultipoles) -> PowerMultipoles:
    quantiles = tuple(int(q) for q in spectra.quantiles)
    ells = tuple(int(ell) for ell in spectra.ells)
    if not quantiles or len(set(quantiles)) != len(quantiles):
        raise ValueError("quantiles must be non-empty and unique")
    if not ells or len(set(ells)) != len(ells) or any(ell < 0 for ell in ells):
        raise ValueError("ells must be unique non-negative integers")
    pmm = np.asarray(spectra.pmm, dtype="f8")
    pqm = np.asarray(spectra.pqm, dtype="f8")
    pqq = np.asarray(spectra.pqq, dtype="f8")
    if pmm.ndim != 2 or pmm.shape[0] != len(ells):
        raise ValueError("pmm must have shape (nell, nk)")
    nell, nk = pmm.shape
    nq = len(quantiles)
    if pqm.shape != (nq, nell, nk):
        raise ValueError("pqm must have shape (nq, nell, nk)")
    if pqq.shape != (nq, nq, nell, nk):
        raise ValueError("pqq must have shape (nq, nq, nell, nk)")
    if not all(np.isfinite(array).all() for array in (pmm, pqm, pqq)):
        raise ValueError("all multipoles must be finite")
    if not np.allclose(pqq, pqq.swapaxes(0, 1), rtol=1e-10, atol=1e-12):
        raise ValueError("pqq must be symmetric in its quantile indices")
    return PowerMultipoles(quantiles, ells, pmm, pqm, pqq)


def _field_spectra(spectra: PowerMultipoles) -> np.ndarray:
    """Return ``P[field_a, field_b, ell, k]`` with matter at field zero."""
    spectra = _validated_multipoles(spectra)
    nq, nell, nk = spectra.pqm.shape
    fields = np.empty((nq + 1, nq + 1, nell, nk), dtype="f8")
    fields[0, 0] = spectra.pmm
    fields[1:, 0] = spectra.pqm
    fields[0, 1:] = spectra.pqm
    fields[1:, 1:] = spectra.pqq
    return fields


def _normalize_statistics(data_vector: str | Sequence[str]) -> tuple[str, ...]:
    """Validate a statistic selection while preserving its requested order."""
    statistics = (data_vector,) if isinstance(data_vector, str) else tuple(data_vector)
    invalid = [statistic for statistic in statistics if statistic not in STATISTICS]
    if not statistics or invalid:
        raise ValueError(
            f"data_vector must contain one or more of {STATISTICS}; invalid {invalid}"
        )
    if len(set(statistics)) != len(statistics):
        raise ValueError("data_vector statistics must be unique")
    return statistics


def observable_pairs(
    data_vector: str | Sequence[str] | Sequence[tuple[int, int]], nq: int
) -> tuple[tuple[int, int], ...]:
    """Map selected statistics to field pairs; field zero denotes matter."""
    values = (data_vector,) if isinstance(data_vector, str) else tuple(data_vector)
    if not values or all(isinstance(value, str) for value in values):
        statistics = _normalize_statistics(values)
        groups = {
            "pmm": ((0, 0),),
            "pqm": tuple((q, 0) for q in range(1, nq + 1)),
            "pqq": tuple((a, b) for a in range(1, nq + 1) for b in range(a, nq + 1)),
        }
        pairs = tuple(pair for statistic in statistics for pair in groups[statistic])
    else:
        try:
            pairs = tuple((int(a), int(b)) for a, b in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "data_vector must contain statistics or explicit field-index pairs"
            ) from exc
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("observable pairs must be non-empty and unique")
    if any(min(pair) < 0 or max(pair) > nq for pair in pairs):
        raise ValueError(f"observable field indices must lie between 0 and {nq}")
    return pairs


def required_measurement_statistics(
    data_vector: str | Sequence[str] | Sequence[tuple[int, int]], nq: int
) -> tuple[str, ...]:
    """Return the measured spectra needed by the vector and its covariance."""
    pairs = observable_pairs(data_vector, nq)

    def statistic(pair: tuple[int, int]) -> str:
        a, b = pair
        if a == b == 0:
            return "pmm"
        if a == 0 or b == 0:
            return "pqm"
        return "pqq"

    required = {statistic(pair) for pair in pairs}
    for a, b in pairs:
        for c, d in pairs:
            required.update(
                statistic(pair) for pair in ((a, c), (b, d), (a, d), (b, c))
            )
    return tuple(name for name in STATISTICS if name in required)


def _labels(
    pairs: Sequence[tuple[int, int]], spectra: PowerMultipoles, k: np.ndarray | None
) -> tuple[str, ...]:
    names = ("m",) + tuple(f"q{q}" for q in spectra.quantiles)
    nk = spectra.pmm.shape[-1]
    coordinates = np.arange(nk) if k is None else np.asarray(k, dtype="f8")
    if coordinates.shape != (nk,):
        raise ValueError("k must have shape (nk,)")
    return tuple(
        f"P{names[a]}{names[b]}_ell{ell}_k{coordinates[ik]:.8g}"
        for a, b in pairs
        for ell in spectra.ells
        for ik in range(nk)
    )


def _element_metadata(result: CovarianceResult, k: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-element metadata in the covariance result ordering."""
    k = np.asarray(k, dtype="f8")
    nk = result.spectra.pmm.shape[-1]
    if k.shape != (nk,):
        raise ValueError("k must have shape (nk,)")
    quantiles = result.spectra.quantiles
    pair_metadata = []
    for field_a, field_b in result.pairs:
        if field_a == field_b == 0:
            pair_metadata.append(("pmm", "Pmm", 0, 0))
        elif field_a == 0 or field_b == 0:
            quantile = quantiles[max(field_a, field_b) - 1]
            pair_metadata.append(("pqm", f"Pq{quantile}m", quantile, 0))
        else:
            quantile_a = quantiles[field_a - 1]
            quantile_b = quantiles[field_b - 1]
            pair_metadata.append(
                ("pqq", f"Pq{quantile_a}q{quantile_b}", quantile_a, quantile_b)
            )
    block_size = len(result.spectra.ells) * nk
    return {
        "statistic_for_element": np.repeat(
            [metadata[0] for metadata in pair_metadata], block_size
        ),
        "pair_label_for_element": np.repeat(
            [metadata[1] for metadata in pair_metadata], block_size
        ),
        "quantile_a_for_element": np.repeat(
            [metadata[2] for metadata in pair_metadata], block_size
        ),
        "quantile_b_for_element": np.repeat(
            [metadata[3] for metadata in pair_metadata], block_size
        ),
        "ell_for_element": np.tile(
            np.repeat(result.spectra.ells, nk), len(result.pairs)
        ),
        "k_bin_for_element": np.tile(
            np.arange(nk, dtype="i8"), len(result.pairs) * len(result.spectra.ells)
        ),
        "k_for_element": np.tile(k, len(result.pairs) * len(result.spectra.ells)),
    }


def load_covariance_archive(path: str | Path) -> CovarianceArchive:
    """Load and validate a self-describing covariance archive."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"covariance archive does not exist: {path}")
    required = {
        "archive_schema_version",
        "empirical_single_box",
        "hybrid_single_box",
        "pt_single_box",
        "labels",
        "statistic_for_element",
        "pair_label_for_element",
        "quantile_a_for_element",
        "quantile_b_for_element",
        "ell_for_element",
        "k_bin_for_element",
        "k_for_element",
        "quantiles",
        "ells",
        "k",
        "k_edges",
        "nmodes",
        "los",
        "realization_ids",
        "data_vector",
        "ordering",
        "model_names",
        "model_provenance",
    }
    with np.load(path, allow_pickle=False) as products:
        missing = sorted(required - set(products.files))
        if missing:
            raise ValueError(
                f"{path} is not a supported covariance archive; missing {missing}"
            )
        version = int(np.asarray(products["archive_schema_version"]).item())
        if version != ARCHIVE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported covariance archive schema {version}; "
                f"expected {ARCHIVE_SCHEMA_VERSION}"
            )
        arrays = {name: np.asarray(products[name]).copy() for name in required}

    labels = arrays["labels"].astype(str)
    size = labels.size
    matrices = {
        "empirical": np.asarray(arrays["empirical_single_box"], dtype="f8"),
        "hybrid": np.asarray(arrays["hybrid_single_box"], dtype="f8"),
        "pt": np.asarray(arrays["pt_single_box"], dtype="f8"),
    }
    for name, matrix in matrices.items():
        if matrix.shape != (size, size) or not np.isfinite(matrix).all():
            raise ValueError(f"archive {name} covariance must be finite and square")
    element_names = (
        "statistic_for_element",
        "pair_label_for_element",
        "quantile_a_for_element",
        "quantile_b_for_element",
        "ell_for_element",
        "k_bin_for_element",
        "k_for_element",
    )
    for name in element_names:
        if arrays[name].shape != (size,):
            raise ValueError(f"archive {name} must have one value per element")
    model_names = tuple(arrays["model_names"].astype(str).tolist())
    provenance_values = arrays["model_provenance"].astype(str).tolist()
    if model_names != MODEL_NAMES or len(provenance_values) != len(model_names):
        raise ValueError("archive model names or provenance are inconsistent")
    if arrays["data_vector"].ndim != 1:
        raise ValueError("archive data_vector must be a one-dimensional statistic list")
    try:
        data_vector = _normalize_statistics(arrays["data_vector"].astype(str).tolist())
    except ValueError as exc:
        raise ValueError("archive data_vector is invalid") from exc
    return CovarianceArchive(
        path=path.resolve(),
        empirical=matrices["empirical"],
        models={name: matrices[name] for name in model_names},
        labels=labels,
        statistics=arrays["statistic_for_element"].astype(str),
        pair_labels=arrays["pair_label_for_element"].astype(str),
        quantile_a=np.asarray(arrays["quantile_a_for_element"], dtype="i8"),
        quantile_b=np.asarray(arrays["quantile_b_for_element"], dtype="i8"),
        element_ells=np.asarray(arrays["ell_for_element"], dtype="i8"),
        k_bins=np.asarray(arrays["k_bin_for_element"], dtype="i8"),
        element_k=np.asarray(arrays["k_for_element"], dtype="f8"),
        quantiles=np.asarray(arrays["quantiles"], dtype="i8"),
        ells=np.asarray(arrays["ells"], dtype="i8"),
        k=np.asarray(arrays["k"], dtype="f8"),
        k_edges=np.asarray(arrays["k_edges"], dtype="f8"),
        nmodes=np.asarray(arrays["nmodes"], dtype="f8"),
        los=np.asarray(arrays["los"], dtype="f8"),
        realization_ids=np.asarray(arrays["realization_ids"], dtype="i8"),
        data_vector=data_vector,
        ordering=str(np.asarray(arrays["ordering"]).item()),
        model_provenance=dict(zip(model_names, provenance_values)),
    )


def gaussian_multipole_covariance(
    spectra: PowerMultipoles,
    *,
    k_edges: np.ndarray,
    nmodes: np.ndarray,
    boxsize: float | Sequence[float],
    meshsize: int | Sequence[int],
    los: str | Sequence[float] = "z",
    data_vector: str | Sequence[str] | Sequence[tuple[int, int]] = DEFAULT_DATA_VECTOR,
    k: np.ndarray | None = None,
    angular_moments: PeriodicShellAngularMoments | None = None,
    nrealizations: int = 1,
) -> CovarianceResult:
    """Assemble the disconnected Gaussian covariance of arbitrary field pairs.

    For observables ``P_ab`` and ``P_cd``, each shell uses
    ``P_ac P_bd + P_ad P_bc`` and the exact fixed-LOS lattice projection.
    The output ordering is pair-major, multipole-major, then k-major.
    """
    spectra = _validated_multipoles(spectra)
    fields = _field_spectra(spectra)
    pairs = observable_pairs(data_vector, len(spectra.quantiles))
    nmodes = np.asarray(nmodes, dtype="f8")
    nk = spectra.pmm.shape[-1]
    if nmodes.shape != (nk,) or np.any(nmodes <= 0) or not np.isfinite(nmodes).all():
        raise ValueError("nmodes must be finite, positive, and have shape (nk,)")
    raw_nrealizations = nrealizations
    nrealizations = int(raw_nrealizations)
    if nrealizations < 1 or nrealizations != raw_nrealizations:
        raise ValueError("nrealizations must be a positive integer")
    geometry = angular_moments
    if geometry is None:
        geometry = periodic_shell_angular_moments(
            k_edges,
            nmodes,
            ells=spectra.ells,
            boxsize=boxsize,
            meshsize=meshsize,
            los=los,
        )
    elif (
        not isinstance(geometry, PeriodicShellAngularMoments)
        or geometry.ells != spectra.ells
        or not np.array_equal(geometry.k_edges, np.asarray(k_edges))
        or not np.array_equal(geometry.nmodes, nmodes)
    ):
        raise ValueError("angular_moments do not match the requested spectra")
    nell, npair = len(spectra.ells), len(pairs)
    ell_factor = 2 * np.asarray(spectra.ells) + 1
    covariance = np.zeros((npair, nell, nk, npair, nell, nk), dtype="f8")
    for ipair, (a, b) in enumerate(pairs):
        for jpair, (c, d) in enumerate(pairs[ipair:], start=ipair):
            kernel = np.einsum("pk,sk->psk", fields[a, c], fields[b, d])
            kernel += np.einsum("pk,sk->psk", fields[a, d], fields[b, c])
            block = np.einsum("kabps,psk->abk", geometry.moments, kernel)
            block *= ell_factor[:, None, None] * ell_factor[None, :, None]
            block /= nmodes[None, None, :] * nrealizations
            for ik in range(nk):
                covariance[ipair, :, ik, jpair, :, ik] = block[..., ik]
                covariance[jpair, :, ik, ipair, :, ik] = block[..., ik].T
    covariance = covariance.reshape(npair * nell * nk, npair * nell * nk)
    covariance = 0.5 * (covariance + covariance.T)
    validate_covariance(covariance)
    return CovarianceResult(
        covariance=covariance,
        labels=_labels(pairs, spectra, k),
        pairs=pairs,
        spectra=spectra,
    )


def build_hybrid_covariance(
    pmm: np.ndarray,
    pqm: np.ndarray,
    pqq: np.ndarray,
    *,
    quantiles: Sequence[int],
    ells: Sequence[int],
    k_edges: np.ndarray,
    nmodes: np.ndarray,
    boxsize: float | Sequence[float],
    meshsize: int | Sequence[int],
    los: str | Sequence[float] = "z",
    data_vector: str | Sequence[str] | Sequence[tuple[int, int]] = DEFAULT_DATA_VECTOR,
    k: np.ndarray | None = None,
    angular_moments: PeriodicShellAngularMoments | None = None,
    nrealizations: int = 1,
) -> CovarianceResult:
    """Build a covariance from supplied total multipoles.

    The three arrays are independent inputs: callers can freely mix measured,
    emulated, and PT predictions. This is the only distinction between a
    hybrid and a fully analytic covariance; the Gaussian assembly is shared.
    """
    spectra = PowerMultipoles(
        tuple(int(q) for q in quantiles),
        tuple(int(ell) for ell in ells),
        np.asarray(pmm),
        np.asarray(pqm),
        np.asarray(pqq),
    )
    return gaussian_multipole_covariance(
        spectra,
        k_edges=k_edges,
        nmodes=nmodes,
        boxsize=boxsize,
        meshsize=meshsize,
        los=los,
        data_vector=data_vector,
        k=k,
        angular_moments=angular_moments,
        nrealizations=nrealizations,
    )


def build_pt_covariance(
    k_edges: np.ndarray,
    nmodes: np.ndarray,
    *,
    quantiles: Sequence[int] = DEFAULT_QUANTILES,
    ells: Sequence[int] = DEFAULT_ELLS,
    boxsize: float = 1000.0,
    meshsize: int = 256,
    los: str | Sequence[float] = "z",
    data_vector: str | Sequence[str] | Sequence[tuple[int, int]] = DEFAULT_DATA_VECTOR,
    k: np.ndarray | None = None,
    matter_noise: float | np.ndarray = 0.0,
    redshift: float = 0.5,
    smoothing_radius: float = 10.0,
    correlation_ngrid: int = 20001,
    angular_moments: PeriodicShellAngularMoments | None = None,
) -> tuple[CovarianceResult, GaussianThresholdRSDSpectra]:
    """Predict ``Pmm``, ``Pqm``, and total ``Pqq`` with PT and build covariance."""
    prediction = linear_gaussian_threshold_rsd_spectra(
        k_edges,
        nmodes,
        quantiles=quantiles,
        ells=ells,
        boxsize=boxsize,
        meshsize=meshsize,
        redshift=redshift,
        smoothing_radius=smoothing_radius,
        correlation_ngrid=correlation_ngrid,
    )
    pmm = prediction.matter_power_poles.copy()
    noise = np.asarray(matter_noise, dtype="f8")
    if noise.ndim == 0:
        noise = np.full(pmm.shape[-1], float(noise))
    if noise.shape != (pmm.shape[-1],) or not np.isfinite(noise).all():
        raise ValueError("matter_noise must be scalar or have shape (nk,)")
    try:
        i0 = tuple(prediction.ells).index(0)
    except ValueError as exc:
        if np.any(noise):
            raise ValueError("ell=0 is required when matter noise is non-zero") from exc
    else:
        pmm[i0] += noise
    result = build_hybrid_covariance(
        pmm,
        prediction.quantile_matter_power_poles,
        prediction.quantile_quantile_total_power_poles,
        quantiles=prediction.quantiles,
        ells=prediction.ells,
        k_edges=k_edges,
        nmodes=nmodes,
        boxsize=boxsize,
        meshsize=meshsize,
        los=los,
        data_vector=data_vector,
        k=k,
        angular_moments=angular_moments,
    )
    return result, prediction


def smooth_multipoles(
    multipoles: np.ndarray, window: int = 7, polynomial_order: int = 2
) -> np.ndarray:
    """Optionally smooth noisy input multipoles along their final (k) axis."""
    values = np.asarray(multipoles, dtype="f8")
    if values.ndim == 0 or not np.isfinite(values).all():
        raise ValueError("multipoles must be a finite array with a k axis")
    if window == 1:
        return values.copy()
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if polynomial_order < 0 or polynomial_order >= window:
        raise ValueError("polynomial_order must be smaller than window")
    if window > values.shape[-1]:
        raise ValueError("window cannot exceed the number of k bins")
    return savgol_filter(values, window, polynomial_order, axis=-1, mode="interp")


# Backward-compatible spelling retained for notebooks using the old PT validator.
smooth_calibration_spectra = smooth_multipoles


def _discover_realizations(data_root: Path) -> list[tuple[int, Path]]:
    found = []
    for path in Path(data_root).iterdir() if Path(data_root).is_dir() else ():
        match = _REALIZATION.fullmatch(path.name)
        if path.is_dir() and match:
            found.append((int(match.group(1)), path))
    found.sort()
    if len(found) < 2:
        raise ValueError(f"fewer than two realizations found in {data_root}")
    return found


def _read_leaf(
    handle: h5py.File,
    path: str,
    *,
    edges: np.ndarray | None = None,
    nmodes: np.ndarray | None = None,
    los: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path not in handle:
        raise ValueError(f"{handle.filename} is missing branch {path}")
    group = handle[path]
    required = ("value", "k", "k_edges", "nmodes", "num_shotnoise", "norm")
    missing = [name for name in required if name not in group]
    if missing or "los" not in group.attrs:
        raise ValueError(
            f"{handle.filename}:{path} is missing {missing or 'fixed LOS'}"
        )
    arrays = [np.asarray(group[name]) for name in required]
    if any(
        np.iscomplexobj(value) and not np.allclose(value.imag, 0.0) for value in arrays
    ):
        raise ValueError(f"{handle.filename}:{path} contains non-real values")
    value, k, current_edges, current_modes, shot, norm = (
        np.asarray(np.real(array), dtype="f8") for array in arrays
    )
    if not all(
        np.isfinite(array).all()
        for array in (value, k, current_edges, current_modes, shot, norm)
    ):
        raise ValueError(f"{handle.filename}:{path} contains non-finite values")
    if value.ndim != 1 or k.shape != value.shape or current_modes.shape != value.shape:
        raise ValueError(f"{handle.filename}:{path} has inconsistent vector shapes")
    if current_edges.shape != (value.size, 2) or np.any(
        current_edges[:, 1] <= current_edges[:, 0]
    ):
        raise ValueError(f"{handle.filename}:{path} has invalid k-bin edges")
    if np.any(current_modes <= 0) or np.any(norm == 0):
        raise ValueError(f"{handle.filename}:{path} has invalid modes or normalization")
    current_los = np.asarray(group.attrs["los"], dtype="f8")
    if (
        current_los.shape != (3,)
        or not np.isfinite(current_los).all()
        or not np.any(current_los)
    ):
        raise ValueError(f"{handle.filename}:{path} has invalid fixed LOS")
    current_los /= np.linalg.norm(current_los)
    if edges is not None and not np.array_equal(current_edges, edges):
        raise ValueError(f"{handle.filename}:{path} has inconsistent k-bin edges")
    if nmodes is not None and not np.array_equal(current_modes, nmodes):
        raise ValueError(f"{handle.filename}:{path} has inconsistent mode counts")
    if los is not None and not np.allclose(current_los, los, rtol=0.0, atol=1e-14):
        raise ValueError(f"{handle.filename}:{path} has inconsistent fixed LOS")
    return value, shot / norm, k, current_edges, current_modes, current_los


def load_rsd_ensemble(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    quantiles: Sequence[int] = DEFAULT_QUANTILES,
    ells: Sequence[int] = DEFAULT_ELLS,
    data_vector: str | Sequence[str] = DEFAULT_DATA_VECTOR,
    kmin: float = DEFAULT_KMIN,
    kmax: float = DEFAULT_KMAX,
) -> RSDEnsemble:
    """Load realizations containing every measurement needed by the covariance."""
    quantiles = tuple(int(q) for q in quantiles)
    ells = tuple(int(ell) for ell in ells)
    if not quantiles or len(set(quantiles)) != len(quantiles):
        raise ValueError("quantiles must be non-empty and unique")
    if any(q < 1 or q > 5 for q in quantiles):
        raise ValueError("quantiles must lie between 1 and 5")
    if set(quantiles) == set(range(1, 6)):
        raise ValueError("all five quantiles are linearly dependent; omit one")
    if not ells or len(set(ells)) != len(ells) or 0 not in ells:
        raise ValueError("ells must be unique and include zero")
    if not 0 <= kmin < kmax:
        raise ValueError("kmin and kmax must define a positive-width range")
    found = _discover_realizations(Path(data_root))
    required = required_measurement_statistics(data_vector, len(quantiles))
    required_files = tuple(f"{name}.h5" for name in required)
    complete = [
        (realization, directory)
        for realization, directory in found
        if all((directory / f"{name}.h5").is_file() for name in required)
    ]
    complete_ids = {realization for realization, _ in complete}
    skipped = [
        realization for realization, _ in found if realization not in complete_ids
    ]
    if skipped:
        preview = ", ".join(str(realization) for realization in skipped[:10])
        suffix = ", ..." if len(skipped) > 10 else ""
        warnings.warn(
            f"skipping {len(skipped)} realizations missing one or more required "
            f"files {required_files}: {preview}{suffix}",
            RuntimeWarning,
            stacklevel=2,
        )
    if len(complete) < 2:
        raise ValueError(
            f"fewer than two realizations contain all required files {required_files} "
            f"in {data_root}"
        )
    found = complete
    qindices = tuple(q - 1 for q in quantiles)
    reference_branches = {
        "pmm": str(ells[0]),
        "pqm": f"{qindices[0]}/{ells[0]}",
        "pqq": f"{qindices[0]}-{qindices[0]}/{ells[0]}",
    }
    reference = required[0]
    with h5py.File(found[0][1] / f"{reference}.h5", "r") as handle:
        _, _, k, edges, nmodes, los = _read_leaf(handle, reference_branches[reference])
    selection = (k >= kmin) & (k <= kmax)
    if not np.any(selection):
        raise ValueError("the requested scale cut selects no k bins")
    nq, nell, nk, nreal = len(quantiles), len(ells), len(k), len(found)
    values = np.zeros((nreal, nq + 1, nq + 1, nell, nk), dtype="f8")
    totals = np.zeros_like(values)
    for ireal, (_, directory) in enumerate(found):
        paths = {name: directory / f"{name}.h5" for name in required}
        if "pmm" in paths:
            with h5py.File(paths["pmm"], "r") as handle:
                for iell, ell in enumerate(ells):
                    value, noise, *_ = _read_leaf(
                        handle, str(ell), edges=edges, nmodes=nmodes, los=los
                    )
                    values[ireal, 0, 0, iell] = value
                    totals[ireal, 0, 0, iell] = value + noise
        if "pqm" in paths:
            with h5py.File(paths["pqm"], "r") as handle:
                for iq, qindex in enumerate(qindices, start=1):
                    for iell, ell in enumerate(ells):
                        value, noise, *_ = _read_leaf(
                            handle,
                            f"{qindex}/{ell}",
                            edges=edges,
                            nmodes=nmodes,
                            los=los,
                        )
                        values[ireal, iq, 0, iell] = values[ireal, 0, iq, iell] = value
                        totals[ireal, iq, 0, iell] = totals[ireal, 0, iq, iell] = (
                            value + noise
                        )
        if "pqq" in paths:
            with h5py.File(paths["pqq"], "r") as handle:
                for iq, q1 in enumerate(qindices, start=1):
                    for jq, q2 in enumerate(qindices[iq - 1 :], start=iq):
                        branch = f"{min(q1, q2)}-{max(q1, q2)}"
                        for iell, ell in enumerate(ells):
                            value, noise, *_ = _read_leaf(
                                handle,
                                f"{branch}/{ell}",
                                edges=edges,
                                nmodes=nmodes,
                                los=los,
                            )
                            values[ireal, iq, jq, iell] = values[
                                ireal, jq, iq, iell
                            ] = value
                            totals[ireal, iq, jq, iell] = totals[
                                ireal, jq, iq, iell
                            ] = value + noise
    mean = totals.mean(axis=0)[..., selection]
    spectra = PowerMultipoles(
        quantiles,
        ells,
        mean[0, 0],
        mean[1:, 0],
        mean[1:, 1:],
    )
    return RSDEnsemble(
        Path(data_root).resolve(),
        tuple(realization for realization, _ in found),
        k[selection],
        edges[selection],
        nmodes[selection],
        los,
        (totals[:, 0, 0, ells.index(0)] - values[:, 0, 0, ells.index(0)]).mean(axis=0)[
            selection
        ],
        _validated_multipoles(spectra),
        values[..., selection],
    )


def _correlation(covariance: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.diag(covariance))
    return covariance / sigma[:, None] / sigma[None, :]


def _comparison(empirical: np.ndarray, model: np.ndarray) -> dict:
    sigma_ratio = np.sqrt(np.diag(model) / np.diag(empirical))
    correlation_rms = np.sqrt(
        np.mean((_correlation(model) - _correlation(empirical)) ** 2)
    )
    residual = {
        "median_sigma_ratio": float(np.median(sigma_ratio)),
        "p16_sigma_ratio": float(np.percentile(sigma_ratio, 16)),
        "p84_sigma_ratio": float(np.percentile(sigma_ratio, 84)),
        "correlation_rms": float(correlation_rms),
    }
    residual.update(
        {f"matrix_{key}": value for key, value in validate_covariance(model).items()}
    )
    return residual


def run_validation(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    output_root: Path | None = None,
    data_vector: str | Sequence[str] = DEFAULT_DATA_VECTOR,
    quantiles: Sequence[int] = DEFAULT_QUANTILES,
    ells: Sequence[int] = DEFAULT_ELLS,
    kmin: float = DEFAULT_KMIN,
    kmax: float = DEFAULT_KMAX,
    boxsize: float = 1000.0,
    meshsize: int = 256,
    redshift: float = 0.5,
    smoothing_radius: float = 10.0,
    correlation_ngrid: int = 20001,
) -> tuple[dict, dict[str, Path]]:
    """Compare measured-input and fully PT covariances with all realizations."""
    statistics = _normalize_statistics(data_vector)
    ensemble = load_rsd_ensemble(
        data_root,
        quantiles=quantiles,
        ells=ells,
        data_vector=statistics,
        kmin=kmin,
        kmax=kmax,
    )
    geometry = periodic_shell_angular_moments(
        ensemble.k_edges,
        ensemble.nmodes,
        ells=ensemble.spectra.ells,
        boxsize=boxsize,
        meshsize=meshsize,
        los=ensemble.los,
    )
    common = dict(
        k_edges=ensemble.k_edges,
        nmodes=ensemble.nmodes,
        boxsize=boxsize,
        meshsize=meshsize,
        los=ensemble.los,
        data_vector=statistics,
        k=ensemble.k,
        angular_moments=geometry,
    )
    hybrid = build_hybrid_covariance(
        ensemble.spectra.pmm,
        ensemble.spectra.pqm,
        ensemble.spectra.pqq,
        quantiles=ensemble.spectra.quantiles,
        ells=ensemble.spectra.ells,
        **common,
    )
    pt, _ = build_pt_covariance(
        ensemble.k_edges,
        ensemble.nmodes,
        quantiles=ensemble.spectra.quantiles,
        ells=ensemble.spectra.ells,
        boxsize=boxsize,
        meshsize=meshsize,
        los=ensemble.los,
        data_vector=statistics,
        k=ensemble.k,
        matter_noise=ensemble.matter_noise,
        redshift=redshift,
        smoothing_radius=smoothing_radius,
        correlation_ngrid=correlation_ngrid,
        angular_moments=geometry,
    )
    vectors = ensemble.vectors(statistics)
    empirical = np.atleast_2d(np.cov(vectors, rowvar=False, ddof=1))
    summary = {
        "configuration": {
            "data_root": str(ensemble.data_root),
            "realizations": len(ensemble.realization_ids),
            "data_vector": list(statistics),
            "vector_size": vectors.shape[1],
            "quantiles": list(ensemble.spectra.quantiles),
            "ells": list(ensemble.spectra.ells),
            "k_range_h_mpc": [float(ensemble.k[0]), float(ensemble.k[-1])],
            "ordering": "pair-major, multipole-major, k-major",
            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        },
        "hybrid": {
            "provenance": MODEL_PROVENANCE["hybrid"],
            **_comparison(empirical, hybrid.covariance),
        },
        "pt": {
            "provenance": MODEL_PROVENANCE["pt"],
            **_comparison(empirical, pt.covariance),
        },
        "limitations": [
            "disconnected Gaussian covariance only",
            "no connected trispectrum, super-sample covariance, or survey window",
            "PT uses the Gaussian-threshold density-split model",
        ],
    }
    outputs: dict[str, Path] = {}
    if output_root is not None:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        outputs = {
            "summary": root / "summary.json",
            "covariances": root / "covariances.npz",
        }
        outputs["summary"].write_text(json.dumps(summary, indent=2) + "\n")
        element_metadata = _element_metadata(hybrid, ensemble.k)
        np.savez_compressed(
            outputs["covariances"],
            archive_schema_version=ARCHIVE_SCHEMA_VERSION,
            empirical_single_box=empirical,
            hybrid_single_box=hybrid.covariance,
            pt_single_box=pt.covariance,
            labels=np.asarray(hybrid.labels),
            **element_metadata,
            data_vector=np.asarray(statistics),
            ordering="pair-major, multipole-major, k-major",
            model_names=np.asarray(MODEL_NAMES),
            model_provenance=np.asarray(
                [MODEL_PROVENANCE[model] for model in MODEL_NAMES]
            ),
            quantiles=np.asarray(ensemble.spectra.quantiles, dtype="i8"),
            ells=np.asarray(ensemble.spectra.ells, dtype="i8"),
            k=ensemble.k,
            k_edges=ensemble.k_edges,
            nmodes=ensemble.nmodes,
            los=ensemble.los,
            realization_ids=np.asarray(ensemble.realization_ids, dtype="i8"),
            boxsize=boxsize,
            meshsize=meshsize,
            redshift=redshift,
            smoothing_radius=smoothing_radius,
            pmm_hybrid=ensemble.spectra.pmm,
            pqm_hybrid=ensemble.spectra.pqm,
            pqq_hybrid=ensemble.spectra.pqq,
            pmm_pt=pt.spectra.pmm,
            pqm_pt=pt.spectra.pqm,
            pqq_pt=pt.spectra.pqq,
        )
    return summary, outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--data-vector",
        choices=STATISTICS,
        nargs="+",
        default=list(DEFAULT_DATA_VECTOR),
        help="one or more statistics, in data-vector order (default: pmm pqm)",
    )
    parser.add_argument(
        "--quantiles", type=int, nargs="+", default=list(DEFAULT_QUANTILES)
    )
    parser.add_argument("--ells", type=int, nargs="+", default=list(DEFAULT_ELLS))
    parser.add_argument("--kmin", type=float, default=DEFAULT_KMIN)
    parser.add_argument("--kmax", type=float, default=DEFAULT_KMAX)
    parser.add_argument("--boxsize", type=float, default=1000.0)
    parser.add_argument("--meshsize", type=int, default=256)
    parser.add_argument("--redshift", type=float, default=0.5)
    parser.add_argument("--smoothing-radius", type=float, default=10.0)
    parser.add_argument("--correlation-ngrid", type=int, default=20001)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = vars(build_parser().parse_args(argv))
    summary, outputs = run_validation(**args)
    for name in ("hybrid", "pt"):
        result = summary[name]
        print(
            f"{name}: median sigma ratio={result['median_sigma_ratio']:.4f}, "
            f"correlation RMS={result['correlation_rms']:.4f}"
        )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
