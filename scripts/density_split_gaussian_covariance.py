"""Analytic Gaussian covariance for density-split two-point statistics.

The routines in this module use quantile-major ordering throughout.  They are
deliberately independent of desilike calculators so that the same covariance
kernel can be used for real-space monopoles, redshift-space multipoles, and
correlation-function observables.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
from scipy import integrate, special


@dataclass(frozen=True)
class TreeDensitySplitSpectra:
    """Tree-level matter, quantile-matter, and quantile-quantile spectra."""

    k: np.ndarray
    quantiles: tuple[int, ...]
    matter: np.ndarray
    quantile_matter: np.ndarray
    quantile_quantile: np.ndarray


@dataclass(frozen=True)
class PeriodicShellAngularMoments:
    """Exact fixed-LOS angular moments on a periodic ``rfftn`` lattice.

    ``moments[n, a, b, p, q]`` is the Hermitian-weighted shell average of
    ``L_a L_b L_p L_q``.  The conventions match JAXPower's unsmoothed
    ``BinMesh2SpectrumPoles`` covariance path.
    """

    ells: tuple[int, ...]
    k_edges: np.ndarray
    nmodes: np.ndarray
    boxsize: np.ndarray
    meshsize: np.ndarray
    los: np.ndarray
    moments: np.ndarray


def _as_strictly_increasing(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype="f8")
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must contain finite, strictly increasing values")
    return values


def _as_positive_integer(value, name: str) -> int:
    integer = int(value)
    if integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _validated_ells(ells: Sequence[int]) -> tuple[int, ...]:
    raw = np.asarray(ells)
    integer = raw.astype("i8")
    if (
        raw.ndim != 1
        or raw.size == 0
        or not np.array_equal(raw, integer)
        or np.any(integer < 0)
        or len(set(integer.tolist())) != integer.size
    ):
        raise ValueError("ells must contain unique, non-negative integers")
    return tuple(int(ell) for ell in integer)


def _as_three_vector(values, name: str, *, integer: bool = False) -> np.ndarray:
    raw = np.asarray(values)
    dtype = "i8" if integer else "f8"
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 0:
        array = np.repeat(array, 3)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite scalar or length-three vector")
    if np.any(array <= 0):
        raise ValueError(f"{name} must be positive")
    if integer:
        expanded_raw = np.repeat(raw, 3) if raw.ndim == 0 else raw
        if not np.array_equal(expanded_raw, array):
            raise ValueError(f"{name} must contain integers")
    return array


def _validated_los(los) -> np.ndarray:
    if isinstance(los, str):
        try:
            axis = {"x": 0, "y": 1, "z": 2}[los.lower()]
        except KeyError as exc:
            raise ValueError("los must be x, y, z, or a fixed three-vector") from exc
        vector = np.zeros(3, dtype="f8")
        vector[axis] = 1.0
        return vector
    vector = np.asarray(los, dtype="f8")
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("los must be x, y, z, or a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("los must be non-zero")
    return vector / norm


def _validated_periodic_geometry(
    k_edges,
    nmodes,
    *,
    boxsize,
    meshsize,
    los,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_edges = np.asarray(k_edges, dtype="f8")
    nmodes = np.asarray(nmodes, dtype="f8")
    if (
        k_edges.ndim != 2
        or k_edges.shape[1] != 2
        or k_edges.shape[0] == 0
        or not np.isfinite(k_edges).all()
        or np.any(k_edges[:, 0] < 0.0)
        or np.any(k_edges[:, 1] <= k_edges[:, 0])
    ):
        raise ValueError(
            "k_edges must have shape (nk, 2) with valid positive-width bins"
        )
    if not np.allclose(
        k_edges[1:, 0], k_edges[:-1, 1], rtol=0.0, atol=1e-12
    ):
        raise ValueError("k_edges must define contiguous shells")
    if (
        nmodes.shape != (len(k_edges),)
        or not np.isfinite(nmodes).all()
        or np.any(nmodes <= 0.0)
        or not np.array_equal(nmodes, np.rint(nmodes))
    ):
        raise ValueError("nmodes must contain positive integer shell counts")
    boxsize = _as_three_vector(boxsize, "boxsize")
    meshsize = _as_three_vector(meshsize, "meshsize", integer=True)
    los = _validated_los(los)
    nyquist = 2.0 * np.pi * (meshsize // 2) / boxsize
    tolerance = 32.0 * np.finfo("f8").eps * max(1.0, float(nyquist.min()))
    if k_edges[-1, 1] > float(nyquist.min()) + tolerance:
        raise ValueError(
            "the requested shells cross the smallest mesh Nyquist frequency"
        )
    return k_edges, nmodes, boxsize, meshsize, los


def _periodic_rfft_shell_modes(
    k_edges: np.ndarray,
    *,
    boxsize: np.ndarray,
    meshsize: np.ndarray,
    los: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shell indices, mu, and JAXPower Hermitian weights."""
    fundamental = 2.0 * np.pi / boxsize
    maximum = float(k_edges[-1, 1])
    frequencies = []
    for axis in range(3):
        transform = np.fft.rfftfreq if axis == 2 else np.fft.fftfreq
        values = transform(int(meshsize[axis])) * meshsize[axis] * fundamental[axis]
        frequencies.append(values[np.abs(values) < maximum])
    kx = frequencies[0][:, None, None]
    ky = frequencies[1][None, :, None]
    kz = frequencies[2][None, None, :]
    magnitude = np.sqrt(kx**2 + ky**2 + kz**2)
    projection = kx * los[0] + ky * los[1] + kz * los[2]
    mu = np.divide(
        projection,
        magnitude,
        out=np.zeros_like(magnitude, dtype="f8"),
        where=magnitude > 0.0,
    )
    weights = np.broadcast_to(1.0 + (kz > 0.0), magnitude.shape)
    magnitude = magnitude.ravel()
    mu = mu.ravel()
    weights = weights.ravel()
    indices = np.searchsorted(k_edges[:, 1], magnitude, side="right")
    selected = (indices < len(k_edges)) & (magnitude >= k_edges[0, 0])
    return indices[selected], mu[selected], weights[selected]


def periodic_shell_angular_moments(
    k_edges,
    nmodes,
    *,
    ells: Sequence[int],
    boxsize,
    meshsize,
    los="z",
) -> PeriodicShellAngularMoments:
    """Build exact JAXPower-compatible shell averages of four Legendre factors."""
    ells = _validated_ells(ells)
    k_edges, nmodes, boxsize, meshsize, los = _validated_periodic_geometry(
        k_edges,
        nmodes,
        boxsize=boxsize,
        meshsize=meshsize,
        los=los,
    )
    indices, mu, weights = _periodic_rfft_shell_modes(
        k_edges, boxsize=boxsize, meshsize=meshsize, los=los
    )
    counts = np.bincount(indices, weights=weights, minlength=len(k_edges))
    if not np.array_equal(counts, nmodes):
        mismatch = np.flatnonzero(counts != nmodes)
        details = ", ".join(
            f"bin {index}: lattice={counts[index]:g}, stored={nmodes[index]:g}"
            for index in mismatch[:5]
        )
        raise ValueError(
            "stored nmodes do not match the JAXPower periodic lattice; " + details
        )

    legendre = np.stack(
        [special.eval_legendre(ell, mu) for ell in ells], axis=0
    )
    pair = legendre[:, None, :] * legendre[None, :, :]
    nell = len(ells)
    moments = np.empty((len(k_edges), nell, nell, nell, nell), dtype="f8")
    for shell in range(len(k_edges)):
        selected = indices == shell
        moments[shell] = np.einsum(
            "apm,bqm,m->abpq",
            pair[..., selected],
            pair[..., selected],
            weights[selected],
            optimize=True,
        ) / nmodes[shell]
    return PeriodicShellAngularMoments(
        ells=ells,
        k_edges=k_edges.copy(),
        nmodes=nmodes.copy(),
        boxsize=boxsize.copy(),
        meshsize=meshsize.copy(),
        los=los.copy(),
        moments=moments,
    )


def _continuous_angular_moments(
    ells: Sequence[int], nmu: int = 64
) -> np.ndarray:
    """Continuum reference retained solely for regression tests."""
    ells = _validated_ells(ells)
    nmu = _as_positive_integer(nmu, "nmu")
    mu, weights = np.polynomial.legendre.leggauss(nmu)
    legendre = np.stack([special.eval_legendre(ell, mu) for ell in ells])
    pair = legendre[:, None, :] * legendre[None, :, :]
    return 0.5 * np.einsum(
        "apu,bqu,u->abpq", pair, pair, weights, optimize=True
    )


def gaussian_quantile_c1(nquantiles: int = 5) -> dict[int, float]:
    """Return first-order Gaussian-selection coefficients for equal bins."""
    nquantiles = _as_positive_integer(nquantiles, "nquantiles")
    normal = NormalDist()
    edges = [-np.inf]
    edges.extend(normal.inv_cdf(index / nquantiles) for index in range(1, nquantiles))
    edges.append(np.inf)
    probability = 1.0 / nquantiles

    def phi(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.exp(-0.5 * value**2) / np.sqrt(2.0 * np.pi))

    return {
        index: (phi(lower) - phi(upper)) / probability
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]), start=1)
    }


def _coefficient_vector(
    c1: Mapping[int, float] | Sequence[float],
    quantiles: Sequence[int] | None,
) -> tuple[tuple[int, ...], np.ndarray]:
    if isinstance(c1, Mapping):
        if quantiles is None:
            quantiles = tuple(int(quantile) for quantile in c1)
        else:
            quantiles = tuple(int(quantile) for quantile in quantiles)
        try:
            coefficients = np.asarray(
                [c1[quantile] for quantile in quantiles], dtype="f8"
            )
        except KeyError as exc:
            raise ValueError(
                f"missing c1 coefficient for quantile {exc.args[0]}"
            ) from exc
    else:
        coefficients = np.asarray(c1, dtype="f8")
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("c1 must be a non-empty one-dimensional sequence")
        if quantiles is None:
            quantiles = tuple(range(1, coefficients.size + 1))
        else:
            quantiles = tuple(int(quantile) for quantile in quantiles)
            if len(quantiles) != coefficients.size:
                raise ValueError("quantiles and c1 must have the same length")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("quantiles must be unique")
    if not np.isfinite(coefficients).all():
        raise ValueError("all c1 coefficients must be finite")
    return tuple(quantiles), coefficients


def tree_density_split_spectra(
    k,
    linear_power,
    c1: Mapping[int, float] | Sequence[float],
    smoothing_radius: float,
    *,
    quantiles: Sequence[int] | None = None,
) -> TreeDensitySplitSpectra:
    """Build tree-level density-split spectra on a common k grid."""
    k = _as_strictly_increasing(k, "k")
    linear_power = np.asarray(linear_power, dtype="f8")
    if linear_power.shape != k.shape:
        raise ValueError("linear_power must have the same shape as k")
    if not np.isfinite(linear_power).all() or np.any(linear_power < 0.0):
        raise ValueError("linear_power must be finite and non-negative")
    smoothing_radius = float(smoothing_radius)
    if not np.isfinite(smoothing_radius) or smoothing_radius < 0.0:
        raise ValueError("smoothing_radius must be finite and non-negative")
    quantiles, coefficients = _coefficient_vector(c1, quantiles)
    window = np.exp(-0.5 * (k * smoothing_radius) ** 2)
    transfer = coefficients[:, None] * window[None, :]
    quantile_matter = transfer * linear_power[None, :]
    quantile_quantile = (
        transfer[:, None, :] * transfer[None, :, :] * linear_power[None, None, :]
    )
    return TreeDensitySplitSpectra(
        k=k,
        quantiles=quantiles,
        matter=linear_power,
        quantile_matter=quantile_matter,
        quantile_quantile=quantile_quantile,
    )


def categorical_lattice_noise(
    probabilities: Sequence[float],
    query_density: float,
) -> np.ndarray:
    """Return the mutually exclusive categorical noise matrix."""
    probabilities = np.asarray(probabilities, dtype="f8")
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities <= 0.0)
        or not np.isclose(np.sum(probabilities), 1.0, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("probabilities must be positive and sum to one")
    query_density = float(query_density)
    if not np.isfinite(query_density) or query_density <= 0.0:
        raise ValueError("query_density must be finite and positive")
    noise = -np.ones((probabilities.size, probabilities.size), dtype="f8")
    noise[np.diag_indices_from(noise)] += 1.0 / probabilities
    return noise / query_density


def shell_mode_counts(k_edges, volume: float) -> np.ndarray:
    """Return continuum Fourier-mode counts for periodic spherical shells."""
    k_edges = np.asarray(k_edges, dtype="f8")
    if (
        k_edges.ndim != 2
        or k_edges.shape[1] != 2
        or not np.isfinite(k_edges).all()
        or np.any(k_edges[:, 0] < 0.0)
        or np.any(k_edges[:, 1] <= k_edges[:, 0])
    ):
        raise ValueError(
            "k_edges must have shape (nk, 2) with valid positive-width bins"
        )
    volume = float(volume)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("volume must be finite and positive")
    shell_volume = 4.0 * np.pi / 3.0 * (k_edges[:, 1] ** 3 - k_edges[:, 0] ** 3)
    return volume * shell_volume / (2.0 * np.pi) ** 3


def _broadcast_noise(
    noise,
    base_shape: tuple[int, ...],
    nk: int,
    name: str,
) -> np.ndarray:
    if noise is None:
        return np.zeros(base_shape + (nk,), dtype="f8")
    noise = np.asarray(noise, dtype="f8")
    if noise.shape == base_shape:
        noise = np.broadcast_to(noise[..., None], base_shape + (nk,))
    elif noise.shape != base_shape + (nk,):
        raise ValueError(f"{name} must have shape {base_shape} or {base_shape + (nk,)}")
    if not np.isfinite(noise).all():
        raise ValueError(f"{name} must be finite")
    return noise


def _gaussian_kernel(
    matter_power,
    quantile_matter_power,
    quantile_quantile_power,
    *,
    matter_noise: float | np.ndarray = 0.0,
    quantile_noise=None,
    cross_noise=None,
) -> tuple[np.ndarray, np.ndarray]:
    matter_power = np.asarray(matter_power, dtype="f8")
    quantile_matter_power = np.asarray(quantile_matter_power, dtype="f8")
    quantile_quantile_power = np.asarray(quantile_quantile_power, dtype="f8")
    if matter_power.ndim != 1 or matter_power.size == 0:
        raise ValueError("matter_power must be a non-empty one-dimensional array")
    nq, nk = quantile_matter_power.shape if quantile_matter_power.ndim == 2 else (0, 0)
    if nk != matter_power.size or nq == 0:
        raise ValueError("quantile_matter_power must have shape (nquantiles, nk)")
    if quantile_quantile_power.shape != (nq, nq, nk):
        raise ValueError(
            "quantile_quantile_power must have shape (nquantiles, nquantiles, nk)"
        )
    arrays = (matter_power, quantile_matter_power, quantile_quantile_power)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("all input spectra must be finite")

    matter_noise = np.asarray(matter_noise, dtype="f8")
    if matter_noise.ndim == 0:
        matter_noise = np.full(nk, float(matter_noise), dtype="f8")
    if matter_noise.shape != (nk,) or not np.isfinite(matter_noise).all():
        raise ValueError("matter_noise must be scalar or have shape (nk,)")
    qnoise = _broadcast_noise(quantile_noise, (nq, nq), nk, "quantile_noise")
    xnoise = _broadcast_noise(cross_noise, (nq,), nk, "cross_noise")

    total_mm = matter_power + matter_noise
    total_qm = quantile_matter_power + xnoise
    total_qq = quantile_quantile_power + qnoise
    kernel = total_qq * total_mm[None, None, :]
    kernel += total_qm[:, None, :] * total_qm[None, :, :]
    contact = qnoise * matter_noise[None, None, :]
    contact += xnoise[:, None, :] * xnoise[None, :, :]
    return kernel, contact


def validate_covariance(covariance, *, positive_tolerance: float = 1e-10) -> dict:
    """Validate a covariance matrix and return numerical diagnostics."""
    covariance = np.asarray(covariance, dtype="f8")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must be finite")
    scale = max(float(np.max(np.abs(np.diag(covariance)))), 1.0)
    asymmetry = float(np.max(np.abs(covariance - covariance.T)))
    if asymmetry > 1e-10 * scale:
        raise ValueError(
            f"covariance is not symmetric; max asymmetry is {asymmetry:.3e}"
        )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] < -float(positive_tolerance) * scale:
        raise ValueError(
            "covariance is not positive semidefinite; minimum eigenvalue is "
            f"{eigenvalues[0]:.6e}"
        )
    rank = int(np.linalg.matrix_rank(covariance))
    positive = eigenvalues[eigenvalues > positive_tolerance * scale]
    condition = float(eigenvalues[-1] / positive[0]) if positive.size else np.inf
    return {
        "rank": rank,
        "size": int(covariance.shape[0]),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": condition,
        "maximum_asymmetry": asymmetry,
    }


def gaussian_cross_power_covariance(
    matter_power,
    quantile_matter_power,
    quantile_quantile_power,
    *,
    nmodes,
    matter_noise: float | np.ndarray = 0.0,
    quantile_noise=None,
    cross_noise=None,
    nrealizations: int = 1,
) -> np.ndarray:
    """Return the Gaussian covariance of quantile-matter power monopoles."""
    kernel, _ = _gaussian_kernel(
        matter_power,
        quantile_matter_power,
        quantile_quantile_power,
        matter_noise=matter_noise,
        quantile_noise=quantile_noise,
        cross_noise=cross_noise,
    )
    nq, _, nk = kernel.shape
    nmodes = np.asarray(nmodes, dtype="f8")
    if nmodes.shape != (nk,) or not np.isfinite(nmodes).all() or np.any(nmodes <= 0.0):
        raise ValueError("nmodes must be finite, positive, and have shape (nk,)")
    nrealizations = _as_positive_integer(nrealizations, "nrealizations")
    covariance = np.zeros((nq, nk, nq, nk), dtype="f8")
    diagonal = np.arange(nk)
    covariance[:, diagonal, :, diagonal] = np.moveaxis(
        kernel / nmodes[None, None, :] / nrealizations, -1, 0
    )
    covariance = covariance.reshape(nq * nk, nq * nk)
    covariance = 0.5 * (covariance + covariance.T)
    validate_covariance(covariance)
    return covariance


def gaussian_cross_power_multipole_covariance(
    matter_power_poles,
    quantile_matter_power_poles,
    quantile_quantile_power_poles,
    *,
    ells,
    nmodes,
    k_edges,
    boxsize,
    meshsize,
    los="z",
    angular_moments: PeriodicShellAngularMoments | None = None,
    matter_noise: float | np.ndarray = 0.0,
    quantile_noise=None,
    cross_noise=None,
    nrealizations: int = 1,
) -> np.ndarray:
    """Return the Gaussian covariance of quantile--matter multipoles.

    Spectra are expanded as ``P(k, mu) = sum_ell P_ell(k) L_ell(mu)`` and
    projected over the exact fixed-LOS Fourier modes in each shell, with the
    same Hermitian weights as JAXPower's ``rfftn`` estimator.
    The input shapes are ``(nell, nk)``, ``(nq, nell, nk)``, and
    ``(nq, nq, nell, nk)`` for matter, quantile--matter, and
    quantile--quantile poles.  Noise arguments describe white-noise power and
    are added to the monopoles only.  The flattened output ordering is
    ``(quantile, ell, k)``; Gaussian periodic-box covariance has zero
    cross-``k`` blocks.
    """
    ells = _validated_ells(ells)

    matter = np.asarray(matter_power_poles, dtype="f8")
    quantile_matter = np.asarray(quantile_matter_power_poles, dtype="f8")
    quantile_quantile = np.asarray(quantile_quantile_power_poles, dtype="f8")
    if matter.ndim != 2 or matter.shape[0] != len(ells):
        raise ValueError("matter_power_poles must have shape (nell, nk)")
    nell, nk = matter.shape
    if quantile_matter.ndim != 3:
        raise ValueError(
            "quantile_matter_power_poles must have shape (nq, nell, nk)"
        )
    nq = quantile_matter.shape[0]
    if nq == 0 or quantile_matter.shape[1:] != (nell, nk):
        raise ValueError(
            "quantile_matter_power_poles must have shape (nq, nell, nk)"
        )
    if quantile_quantile.shape != (nq, nq, nell, nk):
        raise ValueError(
            "quantile_quantile_power_poles must have shape "
            "(nq, nq, nell, nk)"
        )
    if not all(
        np.isfinite(array).all()
        for array in (matter, quantile_matter, quantile_quantile)
    ):
        raise ValueError("all input spectra must be finite")

    nmodes = np.asarray(nmodes, dtype="f8")
    if (
        nmodes.shape != (nk,)
        or not np.isfinite(nmodes).all()
        or np.any(nmodes <= 0.0)
    ):
        raise ValueError("nmodes must be finite, positive, and have shape (nk,)")
    nrealizations = _as_positive_integer(nrealizations, "nrealizations")
    geometry = (
        periodic_shell_angular_moments(
            k_edges,
            nmodes,
            ells=ells,
            boxsize=boxsize,
            meshsize=meshsize,
            los=los,
        )
        if angular_moments is None
        else angular_moments
    )
    if not isinstance(geometry, PeriodicShellAngularMoments):
        raise TypeError("angular_moments must be PeriodicShellAngularMoments")
    expected_edges, expected_modes, expected_box, expected_mesh, expected_los = (
        _validated_periodic_geometry(
            k_edges,
            nmodes,
            boxsize=boxsize,
            meshsize=meshsize,
            los=los,
        )
    )
    if (
        geometry.ells != ells
        or not np.array_equal(geometry.k_edges, expected_edges)
        or not np.array_equal(geometry.nmodes, expected_modes)
        or not np.array_equal(geometry.boxsize, expected_box)
        or not np.array_equal(geometry.meshsize, expected_mesh)
        or not np.allclose(geometry.los, expected_los, rtol=0.0, atol=1e-14)
    ):
        raise ValueError("angular_moments do not match the requested geometry")

    matter_noise = np.asarray(matter_noise, dtype="f8")
    if matter_noise.ndim == 0:
        matter_noise = np.full(nk, float(matter_noise), dtype="f8")
    if matter_noise.shape != (nk,) or not np.isfinite(matter_noise).all():
        raise ValueError("matter_noise must be scalar or have shape (nk,)")
    qnoise = _broadcast_noise(
        quantile_noise, (nq, nq), nk, "quantile_noise"
    )
    xnoise = _broadcast_noise(cross_noise, (nq,), nk, "cross_noise")
    if np.any(matter_noise) or np.any(qnoise) or np.any(xnoise):
        monopoles = [index for index, ell in enumerate(ells) if ell == 0]
        if len(monopoles) != 1:
            raise ValueError("ell=0 is required when adding white noise")
        i0 = monopoles[0]
        matter = matter.copy()
        quantile_matter = quantile_matter.copy()
        quantile_quantile = quantile_quantile.copy()
        matter[i0] += matter_noise
        quantile_matter[:, i0] += xnoise
        quantile_quantile[:, :, i0] += qnoise

    kernel_poles = np.einsum(
        "qrpk,sk->qrpsk", quantile_quantile, matter
    )
    kernel_poles += np.einsum(
        "qpk,rsk->qrpsk", quantile_matter, quantile_matter
    )
    blocks = np.einsum(
        "kabps,qrpsk->qarbk",
        geometry.moments,
        kernel_poles,
        optimize=True,
    )
    ell_array = np.asarray(ells, dtype="i8")
    ell_factor = (2 * ell_array + 1)[:, None] * (2 * ell_array + 1)[None, :]
    blocks *= ell_factor[None, :, None, :, None]
    blocks /= nmodes[None, None, None, None, :] * nrealizations

    covariance = np.zeros((nq, nell, nk, nq, nell, nk), dtype="f8")
    for ik in range(nk):
        covariance[:, :, ik, :, :, ik] = blocks[..., ik]
    covariance = covariance.reshape(nq * nell * nk, nq * nell * nk)
    covariance = 0.5 * (covariance + covariance.T)
    validate_covariance(covariance)
    return covariance


def gaussian_matter_cross_power_covariance(
    matter_power,
    quantile_matter_power,
    quantile_quantile_power,
    *,
    nmodes,
    matter_noise: float | np.ndarray = 0.0,
    quantile_noise=None,
    cross_noise=None,
    nrealizations: int = 1,
) -> np.ndarray:
    """Return the joint covariance of matter and quantile-matter powers.

    The flattened ordering is ``(matter, quantile_1, ..., quantile_n)``, with
    all Fourier bins for one field contiguous.  Input spectra are understood
    as noise-subtracted; the noise arguments are added only inside Gaussian
    contractions.
    """
    kernel, _ = _gaussian_kernel(
        matter_power,
        quantile_matter_power,
        quantile_quantile_power,
        matter_noise=matter_noise,
        quantile_noise=quantile_noise,
        cross_noise=cross_noise,
    )
    matter_power = np.asarray(matter_power, dtype="f8")
    quantile_matter_power = np.asarray(quantile_matter_power, dtype="f8")
    nq, nk = quantile_matter_power.shape
    nmodes = np.asarray(nmodes, dtype="f8")
    if nmodes.shape != (nk,) or not np.isfinite(nmodes).all() or np.any(nmodes <= 0.0):
        raise ValueError("nmodes must be finite, positive, and have shape (nk,)")
    nrealizations = _as_positive_integer(nrealizations, "nrealizations")

    matter_noise = np.asarray(matter_noise, dtype="f8")
    if matter_noise.ndim == 0:
        matter_noise = np.full(nk, float(matter_noise), dtype="f8")
    if matter_noise.shape != (nk,) or not np.isfinite(matter_noise).all():
        raise ValueError("matter_noise must be scalar or have shape (nk,)")
    total_mm = matter_power + matter_noise
    total_qm = quantile_matter_power + _broadcast_noise(
        cross_noise, (nq,), nk, "cross_noise"
    )

    covariance = np.zeros((nq + 1, nk, nq + 1, nk), dtype="f8")
    normalization = nmodes * nrealizations
    matter_cross = 2.0 * total_mm[None, :] * total_qm / normalization[None, :]
    for ik in range(nk):
        covariance[0, ik, 0, ik] = 2.0 * total_mm[ik] ** 2 / normalization[ik]
        covariance[0, ik, 1:, ik] = matter_cross[:, ik]
        covariance[1:, ik, 0, ik] = matter_cross[:, ik]
        covariance[1:, ik, 1:, ik] = kernel[:, :, ik] / normalization[ik]
    covariance = covariance.reshape((nq + 1) * nk, (nq + 1) * nk)
    covariance = 0.5 * (covariance + covariance.T)
    validate_covariance(covariance)
    return covariance


def bin_averaged_spherical_j0(k, separation_edges) -> np.ndarray:
    """Return volume-averaged spherical j0 for radial shell bins."""
    k = _as_strictly_increasing(k, "k")
    separation_edges = _as_strictly_increasing(separation_edges, "separation_edges")
    if separation_edges[0] < 0.0 or separation_edges.size < 2:
        raise ValueError("separation_edges must define at least one non-negative bin")
    lower, upper = separation_edges[:-1], separation_edges[1:]
    denominator = upper**3 - lower**3
    result = np.empty((lower.size, k.size), dtype="f8")
    for ik, kval in enumerate(k):
        if kval == 0.0:
            result[:, ik] = 1.0
        else:
            result[:, ik] = (
                3.0
                * (
                    upper**2 * special.spherical_jn(1, kval * upper)
                    - lower**2 * special.spherical_jn(1, kval * lower)
                )
                / (kval * denominator)
            )
    return result


def _shell_overlap_matrix(separation_edges: np.ndarray) -> np.ndarray:
    lower, upper = separation_edges[:-1], separation_edges[1:]
    overlap_lower = np.maximum(lower[:, None], lower[None, :])
    overlap_upper = np.minimum(upper[:, None], upper[None, :])
    return 4.0 * np.pi / 3.0 * np.maximum(overlap_upper**3 - overlap_lower**3, 0.0)


def gaussian_cross_correlation_covariance(
    k,
    matter_power,
    quantile_matter_power,
    quantile_quantile_power,
    *,
    separation_edges,
    volume: float,
    matter_noise: float | np.ndarray = 0.0,
    quantile_noise=None,
    cross_noise=None,
    nrealizations: int = 1,
) -> np.ndarray:
    """Return the Gaussian covariance of binned quantile-matter correlations."""
    k = _as_strictly_increasing(k, "k")
    volume = float(volume)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("volume must be finite and positive")
    nrealizations = _as_positive_integer(nrealizations, "nrealizations")
    kernel, contact = _gaussian_kernel(
        matter_power,
        quantile_matter_power,
        quantile_quantile_power,
        matter_noise=matter_noise,
        quantile_noise=quantile_noise,
        cross_noise=cross_noise,
    )
    if kernel.shape[-1] != k.size:
        raise ValueError("all spectra must be tabulated on k")
    nq = kernel.shape[0]
    separation_edges = _as_strictly_increasing(separation_edges, "separation_edges")
    bessel = bin_averaged_spherical_j0(k, separation_edges)
    ns = bessel.shape[0]
    covariance = np.empty((nq, ns, nq, ns), dtype="f8")

    contact_is_constant = np.allclose(contact, contact[..., :1], rtol=1e-12, atol=1e-14)
    if not contact_is_constant:
        raise ValueError(
            "configuration-space contact treatment requires scale-independent noise"
        )
    noncontact = kernel - contact
    prefactor = 1.0 / (2.0 * np.pi**2 * volume * nrealizations)
    for ia in range(nq):
        for ib in range(nq):
            integrand = (
                k[None, None, :] ** 2
                * noncontact[ia, ib][None, None, :]
                * bessel[:, None, :]
                * bessel[None, :, :]
            )
            covariance[ia, :, ib, :] = prefactor * integrate.simpson(
                integrand, x=k, axis=-1
            )

    shell_volume = 4.0 * np.pi / 3.0 * np.diff(separation_edges**3)
    overlap = _shell_overlap_matrix(separation_edges)
    contact_geometry = overlap / (shell_volume[:, None] * shell_volume[None, :])
    covariance += (
        contact[..., 0][:, None, :, None]
        * contact_geometry[None, :, None, :]
        / (volume * nrealizations)
    )
    covariance = covariance.reshape(nq * ns, nq * ns)
    covariance = 0.5 * (covariance + covariance.T)
    validate_covariance(covariance)
    return covariance
