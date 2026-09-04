"""Periodic-box window construction and cache plumbing for power fits."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PMM_WINDOW_CACHE_VERSION = 1
PQM_WINDOW_CACHE_VERSION = 1
PMM_WINDOW_BOXSIZE = 1000.0
PMM_WINDOW_MESHSIZE = 256
PMM_WINDOW_K_CENTER_ATOL = 1.0e-5
KAISER_ELLS = (0, 2, 4)


def _pmm_theory_ells(*, rsd: bool) -> tuple[int, ...]:
    """Return all theory multipoles that can leak into the measurement."""
    return KAISER_ELLS if rsd else (0,)


def _periodic_lattice_theory_grid(
    k_edges: np.ndarray, *, boxsize: float, meshsize: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return one input bin and exact theory coordinate per lattice radius."""
    k_edges = np.asarray(k_edges, dtype="f8")
    if (
        k_edges.ndim != 2
        or k_edges.shape[1] != 2
        or k_edges.shape[0] == 0
        or not np.isfinite(k_edges).all()
        or np.any(k_edges[:, 1] <= k_edges[:, 0])
        or not np.allclose(k_edges[1:, 0], k_edges[:-1, 1], rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("k_edges must contain finite, contiguous bin pairs")
    boxsize = float(boxsize)
    meshsize = int(meshsize)
    if not np.isfinite(boxsize) or boxsize <= 0.0:
        raise ValueError("boxsize must be positive")
    if meshsize <= 0:
        raise ValueError("meshsize must be positive")

    fundamental = 2.0 * np.pi / boxsize
    nyquist = fundamental * (meshsize // 2)
    if k_edges[-1, 1] > nyquist + 32.0 * np.finfo("f8").eps:
        raise ValueError("selected k bins extend beyond the mesh Nyquist frequency")

    integer_frequencies = []
    for axis in range(3):
        transform = np.fft.rfftfreq if axis == 2 else np.fft.fftfreq
        values = np.rint(transform(meshsize) * meshsize).astype("i8")
        values = values[np.abs(values * fundamental) < k_edges[-1, 1]]
        integer_frequencies.append(values)
    nx = integer_frequencies[0][:, None, None]
    ny = integer_frequencies[1][None, :, None]
    nz = integer_frequencies[2][None, None, :]
    squared_radius = np.unique((nx**2 + ny**2 + nz**2).ravel())
    radii = fundamental * np.sqrt(squared_radius.astype("f8"))
    radii = radii[(radii > 0.0) & (radii >= k_edges[0, 0]) & (radii < k_edges[-1, 1])]
    if radii.size == 0:
        raise ValueError("selected k bins contain no non-zero Fourier modes")

    boundaries = np.concatenate(
        (
            [k_edges[0, 0]],
            0.5 * (radii[:-1] + radii[1:]),
            [k_edges[-1, 1]],
        )
    )
    theory_edges = np.column_stack((boundaries[:-1], boundaries[1:]))
    return radii, theory_edges


def validate_pmm_window(window, measurements, *, rsd: bool):
    """Validate a window against the selected data and theory model."""
    import lsstypes

    if not isinstance(window, lsstypes.WindowMatrix):
        raise TypeError("pmm window must be an lsstypes.WindowMatrix")
    if tuple(window.observable.ells) != measurements.ells:
        raise ValueError(
            "window observable multipoles do not match measurement multipoles"
        )
    expected_theory_ells = _pmm_theory_ells(rsd=rsd)
    if tuple(window.theory.ells) != expected_theory_ells:
        raise ValueError(
            "window theory multipoles do not match the required model "
            f"multipoles {expected_theory_ells}"
        )

    for ell in measurements.ells:
        pole = window.observable.get(ells=ell)
        if not np.allclose(
            pole.coords("k"),
            measurements.k,
            rtol=1.0e-5,
            atol=PMM_WINDOW_K_CENTER_ATOL,
        ):
            raise ValueError("window observable k bins do not match measurements")
        if measurements.k_edges is not None and not np.allclose(
            pole.edges("k"), measurements.k_edges, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("window observable k edges do not match measurements")
        if measurements.nmodes is not None and not np.array_equal(
            pole.values("nmodes"), measurements.nmodes
        ):
            raise ValueError("window observable mode counts do not match measurements")

    theory_k = None
    for ell in expected_theory_ells:
        current = np.asarray(window.theory.get(ells=ell).coords("k"), dtype="f8")
        if current.ndim != 1 or current.size == 0:
            raise ValueError(
                "window theory k grid must be non-empty and one-dimensional"
            )
        if not np.isfinite(current).all() or np.any(current <= 0.0):
            raise ValueError("window theory k grid must be finite and positive")
        if theory_k is None:
            theory_k = current
        elif not np.array_equal(current, theory_k):
            raise ValueError("window theory multipoles use inconsistent k grids")

    value = np.asarray(window.value(), dtype="f8")
    expected_shape = (
        measurements.data.size,
        len(expected_theory_ells) * theory_k.size,
    )
    if value.shape != expected_shape:
        raise ValueError(f"window shape is {value.shape}, expected {expected_shape}")
    if not np.isfinite(value).all():
        raise ValueError("window matrix contains non-finite values")
    return window


def _validate_cached_pmm_window(
    window,
    measurements,
    *,
    boxsize: float,
    meshsize: int,
    rsd: bool,
):
    """Validate implementation-specific metadata on an automatic cache."""
    window = validate_pmm_window(window, measurements, rsd=rsd)
    expected_attrs = {
        "cache_version": PMM_WINDOW_CACHE_VERSION,
        "kind": "periodic-box-exact-radii",
        "boxsize": float(boxsize),
        "meshsize": int(meshsize),
    }
    for name, expected in expected_attrs.items():
        if window.attrs.get(name) != expected:
            raise ValueError(
                f"cached window attribute {name!r} is "
                f"{window.attrs.get(name)!r}, expected {expected!r}"
            )
    if not np.allclose(
        window.attrs.get("los"), measurements.los, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("cached window line of sight does not match measurements")

    expected_k, _ = _periodic_lattice_theory_grid(
        measurements.k_edges,
        boxsize=boxsize,
        meshsize=meshsize,
    )
    for ell in _pmm_theory_ells(rsd=rsd):
        if not np.array_equal(window.theory.get(ells=ell).coords("k"), expected_k):
            raise ValueError(
                "cached window theory grid does not match the periodic lattice"
            )
    return window


def build_pmm_window(
    measurements,
    *,
    boxsize: float = PMM_WINDOW_BOXSIZE,
    meshsize: int = PMM_WINDOW_MESHSIZE,
    rsd: bool = False,
):
    """Build the exact periodic-box binning and multipole-mixing matrix."""
    from jaxpower import (
        BinMesh2SpectrumPoles,
        MeshAttrs,
        compute_mesh2_spectrum_window,
    )

    if measurements.k_edges is None or measurements.nmodes is None:
        raise ValueError("measurements do not contain Fourier-bin metadata")
    if measurements.los is None:
        raise ValueError("measurements do not contain a line of sight")

    theory_k, theory_edges = _periodic_lattice_theory_grid(
        measurements.k_edges,
        boxsize=boxsize,
        meshsize=meshsize,
    )
    mesh = MeshAttrs(boxsize=float(boxsize), boxcenter=0.0, meshsize=int(meshsize))
    binning = BinMesh2SpectrumPoles(
        mesh, edges=measurements.k_edges, ells=measurements.ells
    )
    window = compute_mesh2_spectrum_window(
        mesh,
        edgesin=theory_edges,
        ellsin=_pmm_theory_ells(rsd=rsd),
        los=measurements.los,
        bin=binning,
    )
    theory_observable = window.theory.map(lambda pole: pole.clone(k=theory_k), level=1)
    window = window.clone(
        theory=theory_observable,
        attrs={
            "cache_version": PMM_WINDOW_CACHE_VERSION,
            "kind": "periodic-box-exact-radii",
            "boxsize": float(boxsize),
            "meshsize": int(meshsize),
            "los": np.asarray(measurements.los, dtype="f8"),
        },
    )
    return validate_pmm_window(window, measurements, rsd=rsd)


def _pmm_window_cache_options(
    measurements,
    *,
    boxsize: float,
    meshsize: int,
    rsd: bool,
) -> dict:
    """Return the complete configuration defining a periodic-box window."""
    return {
        "cache_version": PMM_WINDOW_CACHE_VERSION,
        "kind": "periodic-box-exact-radii",
        "boxsize": float(boxsize),
        "meshsize": int(meshsize),
        "los": np.asarray(measurements.los, dtype="f8").tolist(),
        "k_edges": np.asarray(measurements.k_edges, dtype="f8").tolist(),
        "nmodes": np.asarray(measurements.nmodes, dtype="f8").tolist(),
        "observable_ells": [int(ell) for ell in measurements.ells],
        "theory_ells": list(_pmm_theory_ells(rsd=rsd)),
    }


def _pmm_window_path(
    output_dir: Path,
    measurements,
    *,
    boxsize: float,
    meshsize: int,
    rsd: bool,
) -> Path:
    options = _pmm_window_cache_options(
        measurements,
        boxsize=boxsize,
        meshsize=meshsize,
        rsd=rsd,
    )
    serialized = json.dumps(
        options, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]
    return Path(output_dir) / f"pmm_window_{digest}.h5"


def get_or_build_pmm_window(
    output_dir: Path,
    measurements,
    *,
    boxsize: float = PMM_WINDOW_BOXSIZE,
    meshsize: int = PMM_WINDOW_MESHSIZE,
    rsd: bool = False,
    window_path: Path | None = None,
):
    """Load a supplied/cached window or build and atomically cache one."""
    import lsstypes

    if window_path is not None:
        path = Path(window_path)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = _pmm_window_path(
            output_dir,
            measurements,
            boxsize=boxsize,
            meshsize=meshsize,
            rsd=rsd,
        )
    if path.is_file():
        try:
            window = lsstypes.read(path)
            if window_path is None:
                return _validate_cached_pmm_window(
                    window,
                    measurements,
                    boxsize=boxsize,
                    meshsize=meshsize,
                    rsd=rsd,
                )
            return validate_pmm_window(window, measurements, rsd=rsd)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load compatible pmm window {path}; "
                "remove or replace this artifact and rerun"
            ) from exc
    if window_path is not None:
        raise FileNotFoundError(f"pmm window does not exist: {path}")

    window = build_pmm_window(
        measurements,
        boxsize=boxsize,
        meshsize=meshsize,
        rsd=rsd,
    )
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=".tmp.h5",
        dir=path.parent,
        delete=False,
    )
    handle.close()
    temporary = Path(handle.name)
    try:
        window.write(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return window


def _pqm_quantile_labels(measurements) -> tuple[int, ...]:
    """Return the zero-based lsstypes labels for one-based model quantiles."""
    quantiles = tuple(int(quantile) for quantile in measurements.quantiles)
    if not quantiles:
        raise ValueError("pqm measurements must select at least one quantile")
    return tuple(quantile - 1 for quantile in quantiles)


def validate_pqm_window(window, measurements, *, rsd: bool):
    """Validate a quantile-major block window against pQm measurements."""
    import lsstypes

    if not isinstance(window, lsstypes.WindowMatrix):
        raise TypeError("pqm window must be an lsstypes.WindowMatrix")
    expected_quantiles = _pqm_quantile_labels(measurements)
    if tuple(window.observable.quantiles) != expected_quantiles:
        raise ValueError("window observable quantiles do not match measurements")
    if tuple(window.theory.quantiles) != expected_quantiles:
        raise ValueError("window theory quantiles do not match the model")

    expected_theory_ells = _pmm_theory_ells(rsd=rsd)
    theory_k = None
    for quantile in expected_quantiles:
        observable = window.observable.get(quantiles=quantile)
        theory = window.theory.get(quantiles=quantile)
        if tuple(observable.ells) != measurements.ells:
            raise ValueError(
                "window observable multipoles do not match measurement multipoles"
            )
        if tuple(theory.ells) != expected_theory_ells:
            raise ValueError(
                "window theory multipoles do not match the required model "
                f"multipoles {expected_theory_ells}"
            )
        for ell in measurements.ells:
            pole = observable.get(ells=ell)
            if not np.allclose(
                pole.coords("k"),
                measurements.k,
                rtol=1.0e-5,
                atol=PMM_WINDOW_K_CENTER_ATOL,
            ):
                raise ValueError("window observable k bins do not match measurements")
            if measurements.k_edges is not None and not np.allclose(
                pole.edges("k"), measurements.k_edges, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError("window observable k edges do not match measurements")
            if measurements.nmodes is not None and not np.array_equal(
                pole.values("nmodes"), measurements.nmodes
            ):
                raise ValueError(
                    "window observable mode counts do not match measurements"
                )
        for ell in expected_theory_ells:
            current = np.asarray(theory.get(ells=ell).coords("k"), dtype="f8")
            if current.ndim != 1 or current.size == 0:
                raise ValueError(
                    "window theory k grid must be non-empty and one-dimensional"
                )
            if not np.isfinite(current).all() or np.any(current <= 0.0):
                raise ValueError("window theory k grid must be finite and positive")
            if theory_k is None:
                theory_k = current
            elif not np.array_equal(current, theory_k):
                raise ValueError(
                    "window theory quantiles or multipoles use inconsistent k grids"
                )

    value = np.asarray(window.value(), dtype="f8")
    expected_shape = (
        measurements.data.size,
        len(expected_quantiles) * len(expected_theory_ells) * theory_k.size,
    )
    if value.shape != expected_shape:
        raise ValueError(f"window shape is {value.shape}, expected {expected_shape}")
    if not np.isfinite(value).all():
        raise ValueError("window matrix contains non-finite values")
    return window


def build_pqm_window(
    measurements,
    *,
    boxsize: float = PMM_WINDOW_BOXSIZE,
    meshsize: int = PMM_WINDOW_MESHSIZE,
    rsd: bool = False,
):
    """Build the pQm response by repeating the exact single-field window."""
    import lsstypes

    quantile_labels = _pqm_quantile_labels(measurements)
    single_size = len(measurements.ells) * len(measurements.k)
    single_measurements = SimpleNamespace(
        k=measurements.k,
        data=np.zeros(single_size, dtype="f8"),
        ells=measurements.ells,
        k_edges=measurements.k_edges,
        nmodes=measurements.nmodes,
        los=measurements.los,
    )
    single = build_pmm_window(
        single_measurements,
        boxsize=boxsize,
        meshsize=meshsize,
        rsd=rsd,
    )
    observable = lsstypes.ObservableTree(
        [single.observable.copy() for _ in quantile_labels],
        quantiles=quantile_labels,
    )
    theory = lsstypes.ObservableTree(
        [single.theory.copy() for _ in quantile_labels],
        quantiles=quantile_labels,
    )
    value = np.kron(np.eye(len(quantile_labels)), np.asarray(single.value()))
    window = lsstypes.WindowMatrix(
        value,
        observable=observable,
        theory=theory,
        attrs={
            "cache_version": PQM_WINDOW_CACHE_VERSION,
            "kind": "periodic-box-exact-radii-quantile-block",
            "boxsize": float(boxsize),
            "meshsize": int(meshsize),
            "los": np.asarray(measurements.los, dtype="f8"),
            "quantiles": np.asarray(measurements.quantiles, dtype="i8"),
        },
    )
    return validate_pqm_window(window, measurements, rsd=rsd)


def _pqm_window_cache_options(
    measurements,
    *,
    boxsize: float,
    meshsize: int,
    rsd: bool,
) -> dict:
    return {
        "cache_version": PQM_WINDOW_CACHE_VERSION,
        "kind": "periodic-box-exact-radii-quantile-block",
        "boxsize": float(boxsize),
        "meshsize": int(meshsize),
        "los": np.asarray(measurements.los, dtype="f8").tolist(),
        "k_edges": np.asarray(measurements.k_edges, dtype="f8").tolist(),
        "nmodes": np.asarray(measurements.nmodes, dtype="f8").tolist(),
        "quantiles": [int(quantile) for quantile in measurements.quantiles],
        "observable_ells": [int(ell) for ell in measurements.ells],
        "theory_ells": list(_pmm_theory_ells(rsd=rsd)),
    }


def _pqm_window_path(
    output_dir: Path,
    measurements,
    *,
    boxsize: float,
    meshsize: int,
    rsd: bool,
) -> Path:
    serialized = json.dumps(
        _pqm_window_cache_options(
            measurements,
            boxsize=boxsize,
            meshsize=meshsize,
            rsd=rsd,
        ),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]
    return Path(output_dir) / f"pqm_window_{digest}.h5"


def _validate_cached_pqm_window(
    window,
    measurements,
    *,
    boxsize: float,
    meshsize: int,
    rsd: bool,
):
    window = validate_pqm_window(window, measurements, rsd=rsd)
    expected_attrs = {
        "cache_version": PQM_WINDOW_CACHE_VERSION,
        "kind": "periodic-box-exact-radii-quantile-block",
        "boxsize": float(boxsize),
        "meshsize": int(meshsize),
    }
    for name, expected in expected_attrs.items():
        if window.attrs.get(name) != expected:
            raise ValueError(
                f"cached window attribute {name!r} is "
                f"{window.attrs.get(name)!r}, expected {expected!r}"
            )
    if not np.allclose(
        window.attrs.get("los"), measurements.los, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("cached window line of sight does not match measurements")
    if not np.array_equal(window.attrs.get("quantiles"), measurements.quantiles):
        raise ValueError("cached window quantiles do not match measurements")
    expected_k, _ = _periodic_lattice_theory_grid(
        measurements.k_edges, boxsize=boxsize, meshsize=meshsize
    )
    for quantile in _pqm_quantile_labels(measurements):
        theory = window.theory.get(quantiles=quantile)
        for ell in _pmm_theory_ells(rsd=rsd):
            if not np.array_equal(theory.get(ells=ell).coords("k"), expected_k):
                raise ValueError(
                    "cached window theory grid does not match the periodic lattice"
                )
    return window


def get_or_build_pqm_window(
    output_dir: Path,
    measurements,
    *,
    boxsize: float = PMM_WINDOW_BOXSIZE,
    meshsize: int = PMM_WINDOW_MESHSIZE,
    rsd: bool = False,
    window_path: Path | None = None,
):
    """Load, validate, or atomically cache a pQm block window."""
    import lsstypes

    if window_path is not None:
        path = Path(window_path)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = _pqm_window_path(
            output_dir,
            measurements,
            boxsize=boxsize,
            meshsize=meshsize,
            rsd=rsd,
        )
    if path.is_file():
        try:
            window = lsstypes.read(path)
            if window_path is None:
                return _validate_cached_pqm_window(
                    window,
                    measurements,
                    boxsize=boxsize,
                    meshsize=meshsize,
                    rsd=rsd,
                )
            return validate_pqm_window(window, measurements, rsd=rsd)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load compatible pqm window {path}; "
                "remove or replace this artifact and rerun"
            ) from exc
    if window_path is not None:
        raise FileNotFoundError(f"pqm window does not exist: {path}")

    window = build_pqm_window(
        measurements,
        boxsize=boxsize,
        meshsize=meshsize,
        rsd=rsd,
    )
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".tmp.h5", dir=path.parent, delete=False
    )
    handle.close()
    temporary = Path(handle.name)
    try:
        window.write(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return window
