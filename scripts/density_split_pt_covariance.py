"""Perturbative spectra for the density-split Gaussian covariance.

The quantile-pair prediction evaluates the exact two-point function of
equal-probability thresholds of a Gaussian, linearly evolved, smoothed RSD
density field on the periodic query lattice.  This resums the full Hermite
selection expansion and therefore predicts both clustering and the mutually
exclusive categorical white term without density-split measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import integrate, special
from scipy.stats import norm

try:
    from scripts.density_split_cosmology import QUIJOTE_COSMOLOGY
except ModuleNotFoundError:
    from density_split_cosmology import QUIJOTE_COSMOLOGY


@dataclass(frozen=True)
class GaussianThresholdRSDSpectra:
    """Tree-level RSD spectra in quantile-major multipole ordering."""

    quantiles: tuple[int, ...]
    ells: tuple[int, ...]
    matter_power_poles: np.ndarray
    quantile_matter_power_poles: np.ndarray
    quantile_quantile_total_power_poles: np.ndarray
    smoothed_variance: float
    meshsize: int
    boxsize: float
    redshift: float
    smoothing_radius: float
    growth_rate: float


def gaussian_quantile_hermite_coefficients(
    nquantiles: int = 5, max_order: int = 3
) -> np.ndarray:
    """Return Gaussian indicator-field Hermite coefficients through an order.

    The result has shape ``(nquantiles, max_order + 1)``.  Column zero is one;
    the remaining columns are the coefficients ``a_n`` in
    ``delta_Q = sum_n a_n He_n(nu) / n!``.
    """
    nquantiles = int(nquantiles)
    max_order = int(max_order)
    if nquantiles < 2:
        raise ValueError("nquantiles must be at least two")
    if max_order < 1:
        raise ValueError("max_order must be positive")
    edges = norm.ppf(np.arange(nquantiles + 1) / nquantiles)
    probability = 1.0 / nquantiles
    coefficients = np.zeros((nquantiles, max_order + 1), dtype="f8")
    coefficients[:, 0] = 1.0
    for quantile, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        for order in range(1, max_order + 1):
            low = (
                0.0
                if not np.isfinite(lower)
                else norm.pdf(lower)
                * special.eval_hermitenorm(order - 1, lower)
            )
            high = (
                0.0
                if not np.isfinite(upper)
                else norm.pdf(upper)
                * special.eval_hermitenorm(order - 1, upper)
            )
            coefficients[quantile, order] = (low - high) / probability
    return coefficients


def _bivariate_normal_pdf(x: float, y: float, rho: np.ndarray) -> np.ndarray:
    if not np.isfinite(x) or not np.isfinite(y):
        return np.zeros_like(rho)
    variance = 1.0 - rho**2
    exponent = -(x**2 - 2.0 * rho * x * y + y**2) / (2.0 * variance)
    return np.exp(exponent) / (2.0 * np.pi * np.sqrt(variance))


def gaussian_quantile_correlation_table(
    quantile1: int,
    quantile2: int,
    *,
    nquantiles: int = 5,
    ngrid: int = 20001,
    endpoint: float = 0.999999,
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the exact threshold correlation as a function of ``rho``.

    Plackett's identity turns the bivariate-normal rectangle probability into
    a stable one-dimensional integral starting from the independent value at
    ``rho = 0``.  Quantile indices are zero based.
    """
    nquantiles = int(nquantiles)
    quantile1, quantile2 = int(quantile1), int(quantile2)
    ngrid = int(ngrid)
    endpoint = float(endpoint)
    if nquantiles < 2:
        raise ValueError("nquantiles must be at least two")
    if not 0 <= quantile1 < nquantiles or not 0 <= quantile2 < nquantiles:
        raise ValueError("quantile indices are outside the available bins")
    if ngrid < 101 or ngrid % 2 == 0:
        raise ValueError("ngrid must be an odd integer of at least 101")
    if not 0.9 <= endpoint < 1.0:
        raise ValueError("endpoint must satisfy 0.9 <= endpoint < 1")

    edges = norm.ppf(np.arange(nquantiles + 1) / nquantiles)
    lower1, upper1 = edges[quantile1 : quantile1 + 2]
    lower2, upper2 = edges[quantile2 : quantile2 + 2]
    rho = np.linspace(-endpoint, endpoint, ngrid)
    derivative = (
        _bivariate_normal_pdf(upper1, upper2, rho)
        - _bivariate_normal_pdf(lower1, upper2, rho)
        - _bivariate_normal_pdf(upper1, lower2, rho)
        + _bivariate_normal_pdf(lower1, lower2, rho)
    )
    zero = ngrid // 2
    independent = 1.0 / nquantiles**2
    probability = np.empty_like(rho)
    probability[zero:] = independent + integrate.cumulative_trapezoid(
        derivative[zero:], rho[zero:], initial=0.0
    )
    reverse_integral = integrate.cumulative_trapezoid(
        derivative[: zero + 1][::-1], rho[: zero + 1][::-1], initial=0.0
    )
    probability[: zero + 1] = (independent + reverse_integral)[::-1]
    return rho, probability * nquantiles**2 - 1.0


def gaussian_quantile_correlation(
    rho,
    quantile1: int,
    quantile2: int,
    *,
    nquantiles: int = 5,
    ngrid: int = 20001,
) -> np.ndarray:
    """Evaluate an exact Gaussian-threshold quantile correlation."""
    values = np.asarray(rho, dtype="f8")
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1.0 + 1e-12):
        raise ValueError("rho must be finite and lie in [-1, 1]")
    grid, table = gaussian_quantile_correlation_table(
        quantile1, quantile2, nquantiles=nquantiles, ngrid=ngrid
    )
    result = np.interp(np.clip(values, grid[0], grid[-1]), grid, table)
    same = quantile1 == quantile2
    reverse = quantile1 + quantile2 == nquantiles - 1
    result = np.where(np.isclose(values, 1.0, rtol=0.0, atol=1e-13),
                      nquantiles - 1.0 if same else -1.0, result)
    result = np.where(np.isclose(values, -1.0, rtol=0.0, atol=1e-13),
                      nquantiles - 1.0 if reverse else -1.0, result)
    return result


def _validated_integer_sequence(
    values: Sequence[int], name: str, *, minimum: int = 0
) -> tuple[int, ...]:
    raw = np.asarray(values)
    integer = raw.astype("i8")
    if (
        raw.ndim != 1
        or raw.size == 0
        or not np.array_equal(raw, integer)
        or np.any(integer < minimum)
        or len(set(integer.tolist())) != integer.size
    ):
        raise ValueError(f"{name} must contain unique integers >= {minimum}")
    return tuple(int(value) for value in integer)


def bin_periodic_multipoles(
    field,
    kmagnitude,
    mu,
    k_edges,
    nmodes,
    ells: Sequence[int],
) -> np.ndarray:
    """Bin a full FFT field with JAXPower's Hermitian ``rfftn`` convention."""
    field = np.asarray(field, dtype="f8")
    kmagnitude = np.asarray(kmagnitude, dtype="f8")
    mu = np.asarray(mu, dtype="f8")
    k_edges = np.asarray(k_edges, dtype="f8")
    nmodes = np.asarray(nmodes, dtype="f8")
    ells = _validated_integer_sequence(ells, "ells")
    if field.shape != kmagnitude.shape or field.shape != mu.shape:
        raise ValueError("field, kmagnitude, and mu must have matching shapes")
    if not all(np.isfinite(array).all() for array in (field, kmagnitude, mu)):
        raise ValueError("lattice fields must be finite")
    if k_edges.ndim != 2 or k_edges.shape[1] != 2:
        raise ValueError("k_edges must have shape (nk, 2)")
    if nmodes.shape != (len(k_edges),) or np.any(nmodes <= 0.0):
        raise ValueError("nmodes must be positive and have shape (nk,)")
    if field.ndim != 3:
        raise ValueError("periodic multipole binning requires a three-dimensional mesh")
    hermitian_stop = field.shape[-1] // 2 + 1
    hermitian = np.s_[..., :hermitian_stop]
    field = field[hermitian]
    kmagnitude = kmagnitude[hermitian]
    mu = mu[hermitian]
    hermitian_weights = np.ones((1, 1, hermitian_stop), dtype="f8")
    hermitian_weights[..., 1:] = 2.0
    hermitian_weights = np.broadcast_to(hermitian_weights, field.shape)
    selected = (kmagnitude >= k_edges[0, 0]) & (
        kmagnitude < k_edges[-1, 1]
    )
    indices = np.searchsorted(k_edges[:, 1], kmagnitude[selected], side="right")
    mode_weights = hermitian_weights[selected]
    counts = np.bincount(
        indices, weights=mode_weights, minlength=len(k_edges)
    )
    if not np.array_equal(counts, nmodes):
        raise ValueError("stored nmodes do not match the periodic prediction lattice")
    result = np.empty((len(ells), len(k_edges)), dtype="f8")
    for iell, ell in enumerate(ells):
        weights = (
            mode_weights
            * field[selected]
            * special.eval_legendre(ell, mu[selected])
        )
        result[iell] = (2 * ell + 1) * np.bincount(
            indices, weights=weights, minlength=len(k_edges)
        ) / nmodes
    return result


def linear_gaussian_threshold_rsd_spectra(
    k_edges,
    nmodes,
    *,
    quantiles: Sequence[int] = (1, 2, 4, 5),
    ells: Sequence[int] = (0, 2, 4),
    nquantiles: int = 5,
    boxsize: float = 1000.0,
    meshsize: int = 256,
    redshift: float = 0.5,
    smoothing_radius: float = 10.0,
    cosmology: Mapping[str, float] | None = None,
    correlation_ngrid: int = 20001,
) -> GaussianThresholdRSDSpectra:
    """Predict RSD matter, quantile--matter, and total quantile-pair poles.

    ``meshsize`` must describe the actual query lattice.  Consequently the
    returned quantile-pair spectra already contain the categorical white term;
    callers must not add a second quantile-noise contribution.
    """
    k_edges = np.asarray(k_edges, dtype="f8")
    nmodes = np.asarray(nmodes, dtype="f8")
    quantiles = _validated_integer_sequence(quantiles, "quantiles", minimum=1)
    ells = _validated_integer_sequence(ells, "ells")
    nquantiles = int(nquantiles)
    meshsize = int(meshsize)
    boxsize = float(boxsize)
    redshift = float(redshift)
    smoothing_radius = float(smoothing_radius)
    if any(quantile > nquantiles for quantile in quantiles):
        raise ValueError("requested quantiles exceed nquantiles")
    if meshsize < 8:
        raise ValueError("meshsize must be at least eight")
    if not np.isfinite([boxsize, redshift, smoothing_radius]).all():
        raise ValueError("boxsize, redshift, and smoothing_radius must be finite")
    if boxsize <= 0.0 or redshift < 0.0 or smoothing_radius <= 0.0:
        raise ValueError("boxsize and smoothing_radius must be positive")

    from cosmoprimo import Cosmology

    volume = boxsize**3
    fundamental = 2.0 * np.pi / boxsize
    modes = np.fft.fftfreq(meshsize) * meshsize * fundamental
    kx = modes[:, None, None]
    ky = modes[None, :, None]
    kz = modes[None, None, :]
    kmagnitude = np.sqrt(kx**2 + ky**2 + kz**2)
    mu = np.divide(
        kz, kmagnitude, out=np.zeros_like(kmagnitude), where=kmagnitude > 0.0
    )

    cosmology = dict(QUIJOTE_COSMOLOGY if cosmology is None else cosmology)
    cosmo = Cosmology(**cosmology, engine="class")
    linear = cosmo.get_fourier().pk_interpolator().to_1d(z=redshift)
    growth_rate = float(cosmo.growth_rate(redshift))
    matter = np.zeros_like(kmagnitude)
    nonzero = kmagnitude > 0.0
    matter[nonzero] = linear(kmagnitude[nonzero]) * (
        1.0 + growth_rate * mu[nonzero] ** 2
    ) ** 2
    window = np.exp(-0.5 * (kmagnitude * smoothing_radius) ** 2)
    smoothed_power = window**2 * matter
    variance = float(np.sum(smoothed_power) / volume)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("the predicted smoothed variance is not positive")
    normalized_power = smoothed_power / variance
    correlation_coefficient = (
        np.fft.ifftn(normalized_power).real * meshsize**3 / volume
    )
    del normalized_power, smoothed_power

    matter_poles = bin_periodic_multipoles(
        matter, kmagnitude, mu, k_edges, nmodes, ells
    )
    coefficients = gaussian_quantile_hermite_coefficients(
        nquantiles=nquantiles, max_order=1
    )
    selected = np.asarray(quantiles, dtype="i8") - 1
    cross_basis = window * matter / np.sqrt(variance)
    cross_basis_poles = bin_periodic_multipoles(
        cross_basis, kmagnitude, mu, k_edges, nmodes, ells
    )
    cross_poles = coefficients[selected, 1, None, None] * cross_basis_poles

    nq = len(quantiles)
    pair_poles = np.empty((nq, nq, len(ells), len(k_edges)), dtype="f8")
    cell_volume = volume / meshsize**3
    for index1, quantile1 in enumerate(selected):
        for index2 in range(index1, nq):
            quantile2 = selected[index2]
            correlation = gaussian_quantile_correlation(
                correlation_coefficient,
                int(quantile1),
                int(quantile2),
                nquantiles=nquantiles,
                ngrid=correlation_ngrid,
            )
            power = cell_volume * np.fft.fftn(correlation).real
            poles = bin_periodic_multipoles(
                power, kmagnitude, mu, k_edges, nmodes, ells
            )
            pair_poles[index1, index2] = poles
            pair_poles[index2, index1] = poles

    return GaussianThresholdRSDSpectra(
        quantiles=quantiles,
        ells=ells,
        matter_power_poles=matter_poles,
        quantile_matter_power_poles=cross_poles,
        quantile_quantile_total_power_poles=pair_poles,
        smoothed_variance=variance,
        meshsize=meshsize,
        boxsize=boxsize,
        redshift=redshift,
        smoothing_radius=smoothing_radius,
        growth_rate=growth_rate,
    )
