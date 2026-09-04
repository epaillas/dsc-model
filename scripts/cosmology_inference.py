"""Run minimal Kaiser inference for pmm, pQm, or their joint data vector."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from desilike import Calculator
from desilike.theories import ACECosmology, CosmoprimoCosmology
from desilike.theories.galaxy_clustering import (
    DensitySplitMatterPowerSpectrumMultipoles,
)

if __package__:
    from scripts import emulation
    from scripts.window import (
        PMM_WINDOW_BOXSIZE,
        PMM_WINDOW_MESHSIZE,
        get_or_build_pmm_window,
        get_or_build_pqm_window,
        validate_pmm_window,
        validate_pqm_window,
    )
else:
    import emulation
    from window import (
        PMM_WINDOW_BOXSIZE,
        PMM_WINDOW_MESHSIZE,
        get_or_build_pmm_window,
        get_or_build_pqm_window,
        validate_pmm_window,
        validate_pqm_window,
    )


PMM_EMULATOR_CACHE_VERSION = 2
PMM_EMULATOR_ORDER = 3
PMM_EMULATOR_ACCURACY = 2
PMM_EMULATOR_METHOD = "finite"
PMM_M_NCDM = 0.0
PMM_AP_MODEL_VERSION = 1
PMM_AP_MU = 64
PQM_EMULATOR_CACHE_VERSION = 2
PQM_MODEL_VERSION = "factored-density-split-matter-kaiser-v2"
PQM_AP_MODEL_VERSION = "factored-density-split-matter-ap-v1"

# Physical densities omega_i = Omega_i h^2, not density fractions Omega_i.
# logA = ln(10^10 A_s), normalized to massless Quijote sigma8 = 0.834.
QUIJOTE_PARAMETER_VALUES = {
    "h": 0.6711,
    "omega_b": 0.02206838529,
    "omega_cdm": 0.120925743885,
    "n_s": 0.9624,
    "logA": 3.061143632659324,
}


def fix_quijote_cosmology(calculator, names=()) -> dict[str, float]:
    """Fix selected shared parameters in memory, without changing saved caches.

    Apply to the complete likelihood before constructing a profiler or sampler.
    Physical densities remain constant even if h is subsequently varied.
    """
    names = tuple(names)
    if len(set(names)) != len(names):
        raise ValueError("fixed cosmological parameter names must be unique")
    unknown = set(names) - QUIJOTE_PARAMETER_VALUES.keys()
    if unknown:
        raise ValueError(
            f"unsupported fixed cosmological parameters: {sorted(unknown)}"
        )
    if not names:
        return {}
    from desilike import get_params
    params = get_params(calculator)
    missing = [name for name in names if name not in params]
    if missing:
        raise ValueError(f"cosmological parameters not found in calculator: {missing}")
    values = {name: QUIJOTE_PARAMETER_VALUES[name] for name in names}
    for name, value in values.items():
        params[name].update(value=value, fixed=True)
    return values


@dataclass(frozen=True)
class ClusteringMeasurements:
    """Mean data vector and the realization vectors used to estimate it."""

    k: np.ndarray
    data: np.ndarray
    realizations: np.ndarray
    files: tuple[Path, ...]
    ells: tuple[int, ...] = (0,)
    k_edges: np.ndarray | None = None
    nmodes: np.ndarray | None = None
    los: np.ndarray | None = None
    quantiles: tuple[int, ...] = ()


@dataclass(frozen=True)
class MeasurementBundle:
    """Matched realization vectors in the user's requested statistic order."""

    statistics: tuple[str, ...]
    by_stat: dict[str, ClusteringMeasurements]
    data: np.ndarray
    realizations: np.ndarray
    stat_slices: dict[str, slice]


def _read_pmm(path: Path, *, ell: int, kmin: float, kmax: float):
    import lsstypes

    leaf = lsstypes.read(path).get(ells=ell)
    k = np.asarray(leaf.coords("k"), dtype="f8")
    mask = (k >= kmin) & (k <= kmax)
    if not np.any(mask):
        raise ValueError(f"k cuts select no bins in {path}")
    try:
        k_edges = np.asarray(leaf.edges("k"), dtype="f8")[mask]
        nmodes = np.asarray(leaf.values("nmodes"), dtype="f8")[mask]
        los = np.asarray(leaf.attrs["los"], dtype="f8")
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"missing Fourier-bin metadata in {path}") from exc
    return (
        k[mask],
        np.asarray(leaf.value(), dtype="f8")[mask],
        k_edges,
        nmodes,
        los,
    )


def _measurement_files(root: Path, stat: str) -> tuple[Path, ...]:
    return tuple(
        sorted(
            Path(root).glob(f"realization_*/{stat}.h5"),
            key=lambda path: int(path.parent.name.removeprefix("realization_")),
        )
    )


def _validate_measurement_metadata(
    path: Path,
    current,
    reference,
) -> None:
    k, _, k_edges, nmodes, los = current
    reference_k, reference_edges, reference_nmodes, reference_los = reference
    if not np.allclose(k, reference_k, rtol=1.0e-5, atol=1.0e-8):
        raise ValueError(f"inconsistent k bins in {path}")
    if not np.allclose(k_edges, reference_edges, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"inconsistent k edges in {path}")
    if not np.array_equal(nmodes, reference_nmodes):
        raise ValueError(f"inconsistent mode counts in {path}")
    if not np.allclose(los, reference_los, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"inconsistent line of sight in {path}")


def read_clustering_measurements(
    root: Path,
    *,
    stat: str = "pmm",
    ells: tuple[int, ...] | list[int] = (0,),
    kmin: float = 0.01,
    kmax: float = 0.2,
    _files: tuple[Path, ...] | None = None,
) -> ClusteringMeasurements:
    """Read matching realization measurements and return their ensemble mean."""
    if stat != "pmm":
        raise NotImplementedError("only pmm is implemented")
    selected_ells = tuple(ells)

    files = _measurement_files(root, stat) if _files is None else tuple(_files)
    if not files:
        raise FileNotFoundError(f"no realization_*/pmm.h5 files under {root}")

    reference_k = reference_edges = reference_nmodes = reference_los = None
    rows = []
    for path in files:
        poles = []
        for selected_ell in selected_ells:
            k, power, k_edges, nmodes, los = _read_pmm(
                path, ell=selected_ell, kmin=kmin, kmax=kmax
            )
            current = (k, power, k_edges, nmodes, los)
            if reference_k is None:
                reference_k = k
                reference_edges = k_edges
                reference_nmodes = nmodes
                reference_los = los
            else:
                _validate_measurement_metadata(
                    path,
                    current,
                    (reference_k, reference_edges, reference_nmodes, reference_los),
                )
            poles.append(power)
        rows.append(np.concatenate(poles))

    realizations = np.asarray(rows, dtype="f8")
    return ClusteringMeasurements(
        k=reference_k,
        data=np.mean(realizations, axis=0),
        realizations=realizations,
        files=files,
        ells=selected_ells,
        k_edges=reference_edges,
        nmodes=reference_nmodes,
        los=reference_los,
    )


def read_density_split_measurements(
    root: Path,
    *,
    quantiles: tuple[int, ...] | list[int] = (1, 2, 4, 5),
    ells: tuple[int, ...] | list[int] = (0,),
    kmin: float = 0.01,
    kmax: float = 0.2,
    _files: tuple[Path, ...] | None = None,
) -> ClusteringMeasurements:
    """Read pQm branches, converting one-based quantiles to HDF5 labels."""
    import lsstypes

    selected_quantiles = tuple(int(quantile) for quantile in quantiles)
    if not selected_quantiles:
        raise ValueError("at least one density quantile is required")
    if len(set(selected_quantiles)) != len(selected_quantiles):
        raise ValueError("density quantiles must be unique")
    if any(quantile not in range(1, 6) for quantile in selected_quantiles):
        raise ValueError("density quantiles must be between 1 and 5")
    if set(selected_quantiles) == set(range(1, 6)):
        raise ValueError("all five pQm quantiles form a singular partition data vector")
    selected_ells = tuple(int(ell) for ell in ells)
    files = _measurement_files(root, "pqm") if _files is None else tuple(_files)
    if not files:
        raise FileNotFoundError(f"no realization_*/pqm.h5 files under {root}")

    reference_k = reference_edges = reference_nmodes = reference_los = None
    rows = []
    for path in files:
        tree = lsstypes.read(path)
        branches = []
        for quantile in selected_quantiles:
            try:
                poles = tree.get(quantiles=quantile - 1)
            except Exception as exc:
                raise ValueError(
                    f"missing zero-based quantile branch {quantile - 1} in {path}"
                ) from exc
            for ell in selected_ells:
                try:
                    leaf = poles.get(ells=ell)
                except Exception as exc:
                    raise ValueError(
                        f"missing ell={ell} for Q{quantile} in {path}"
                    ) from exc
                k = np.asarray(leaf.coords("k"), dtype="f8")
                mask = (k >= kmin) & (k <= kmax)
                if not np.any(mask):
                    raise ValueError(f"k cuts select no bins in {path}")
                try:
                    current = (
                        k[mask],
                        np.asarray(leaf.value(), dtype="f8")[mask],
                        np.asarray(leaf.edges("k"), dtype="f8")[mask],
                        np.asarray(leaf.values("nmodes"), dtype="f8")[mask],
                        np.asarray(leaf.attrs["los"], dtype="f8"),
                    )
                except (AttributeError, KeyError) as exc:
                    raise ValueError(f"missing Fourier-bin metadata in {path}") from exc
                if reference_k is None:
                    reference_k, _, reference_edges, reference_nmodes, reference_los = (
                        current
                    )
                else:
                    _validate_measurement_metadata(
                        path,
                        current,
                        (
                            reference_k,
                            reference_edges,
                            reference_nmodes,
                            reference_los,
                        ),
                    )
                branches.append(current[1])
        rows.append(np.concatenate(branches))

    realizations = np.asarray(rows, dtype="f8")
    return ClusteringMeasurements(
        k=reference_k,
        data=np.mean(realizations, axis=0),
        realizations=realizations,
        files=files,
        ells=selected_ells,
        k_edges=reference_edges,
        nmodes=reference_nmodes,
        los=reference_los,
        quantiles=selected_quantiles,
    )


def _realization_id(path: Path) -> int:
    return int(path.parent.name.removeprefix("realization_"))


def read_inference_measurements(
    root: Path,
    *,
    stats: tuple[str, ...] | list[str] = ("pmm",),
    quantiles: tuple[int, ...] | list[int] = (1, 2, 4, 5),
    ells: tuple[int, ...] | list[int] = (0,),
    kmin: float = 0.01,
    kmax: float = 0.2,
) -> MeasurementBundle:
    """Read only realizations shared by every requested statistic."""
    statistics = tuple(stats)
    if not statistics or len(set(statistics)) != len(statistics):
        raise ValueError("statistics must be non-empty and unique")
    if set(statistics) - {"pmm", "pqm"}:
        raise ValueError("statistics must contain only pmm and pqm")
    files_by_stat = {stat: _measurement_files(root, stat) for stat in statistics}
    missing = [stat for stat, files in files_by_stat.items() if not files]
    if missing:
        raise FileNotFoundError(
            f"no realization_* files for {', '.join(missing)} under {root}"
        )
    ids = set.intersection(
        *[{_realization_id(path) for path in files} for files in files_by_stat.values()]
    )
    if not ids:
        raise ValueError("no realization IDs contain every requested statistic")
    ordered_ids = sorted(ids)
    matched = {
        stat: tuple(
            {_realization_id(path): path for path in files_by_stat[stat]}[realization]
            for realization in ordered_ids
        )
        for stat in statistics
    }
    by_stat = {}
    for stat in statistics:
        if stat == "pmm":
            by_stat[stat] = read_clustering_measurements(
                root, ells=ells, kmin=kmin, kmax=kmax, _files=matched[stat]
            )
        elif stat == "pqm":
            by_stat[stat] = read_density_split_measurements(
                root,
                quantiles=quantiles,
                ells=ells,
                kmin=kmin,
                kmax=kmax,
                _files=matched[stat],
            )
        else:
            raise ValueError(f"unsupported statistic {stat!r}")

    reference = by_stat[statistics[0]]
    for stat in statistics[1:]:
        current = by_stat[stat]
        _validate_measurement_metadata(
            current.files[0],
            (
                current.k,
                current.data[: current.k.size],
                current.k_edges,
                current.nmodes,
                current.los,
            ),
            (reference.k, reference.k_edges, reference.nmodes, reference.los),
        )
    data_blocks = [by_stat[stat].data for stat in statistics]
    realization_blocks = [by_stat[stat].realizations for stat in statistics]
    start = 0
    stat_slices = {}
    for stat, block in zip(statistics, data_blocks):
        stat_slices[stat] = slice(start, start + block.size)
        start += block.size
    return MeasurementBundle(
        statistics=statistics,
        by_stat=by_stat,
        data=np.concatenate(data_blocks),
        realizations=np.concatenate(realization_blocks, axis=1),
        stat_slices=stat_slices,
    )


def build_covariance_matrix(
    measurements: ClusteringMeasurements | MeasurementBundle,
    *,
    covariance_of_mean: bool = False,
    covariance_scale: float = 1.0,
) -> np.ndarray:
    """Estimate the covariance from realization scatter.

    By default this returns the covariance of one realization. Set
    ``covariance_of_mean`` to divide by the number of realizations and obtain
    the covariance of the ensemble mean stored in ``measurements.data``.
    ``covariance_scale`` must be finite and strictly positive and multiplies
    the entire covariance in either mode (error bars scale by its square root).
    """
    if not np.isfinite(covariance_scale) or covariance_scale <= 0:
        raise ValueError("covariance_scale must be finite and strictly positive")
    realizations = measurements.realizations
    if realizations.shape[0] < 2:
        raise ValueError("at least two realizations are required")
    if realizations.shape[0] <= realizations.shape[1]:
        raise ValueError(
            "the empirical covariance is necessarily singular: "
            f"got {realizations.shape[0]} realizations for "
            f"{realizations.shape[1]} data points"
        )
    covariance = np.atleast_2d(np.cov(realizations, rowvar=False, ddof=1))
    covariance *= covariance_scale
    if covariance_of_mean:
        covariance /= realizations.shape[0]
    return covariance


def _pmm_ap_options() -> dict:
    """Configuration of the optional geometry mapping, also used in cache keys."""
    return {
        "model_version": PMM_AP_MODEL_VERSION,
        "mode": "geometry",
        "fiducial": {**QUIJOTE_PARAMETER_VALUES, "m_ncdm": PMM_M_NCDM,
                     "engine": "class"},
        "mu": PMM_AP_MU,
        "method": "leggauss",
    }


def _validate_pmm_ap(*, ap, rsd, emulator="taylor"):
    if ap and not rsd:
        raise ValueError("AP distortions require RSD (rsd=True)")
    if ap and emulator != "taylor":
        raise ValueError("AP distortions support only exact or Taylor evaluation")


def _build_cosmology(engine):
    """Build a refactored cosmology node; ``ace`` selects MAPSE for linear P(k)."""
    cls = ACECosmology if engine == "ace" else CosmoprimoCosmology
    params = cls.propose_params(engine=engine, fiducial="DESI")
    params["tau_reio"].update(fixed=True)
    params["m_ncdm"].update(value=PMM_M_NCDM, fixed=True)
    return cls(engine=engine, fiducial="DESI", params=params)


class MatterPowerTheory(Calculator):
    """Linear matter power in the refactored JAX graph format."""

    def __init__(self, k=None, z=0.0, ells=(0,), rsd=False, ap=False,
                 engine="class", smoothing_radius=0.):
        _validate_pmm_ap(ap=ap, rsd=rsd)
        self.k = np.asarray(k, dtype="f8")
        self.z = float(z)
        self.ells = tuple(ells)
        self.rsd = bool(rsd)
        self.ap = bool(ap)
        self.engine = str(engine)
        self.smoothing_radius = float(smoothing_radius)
        self.cosmo = _build_cosmology(self.engine)

    def __post_init__(self, **kwargs):
        if self.k.ndim != 1 or not self.k.size:
            raise ValueError("k must be a non-empty one-dimensional array")
        self._pk_k = (np.geomspace(max(1.e-5, self.k.min() / 2.), self.k.max() * 2., 512)
                      if self.ap else self.k)
        requirements = {
            "fourier.pk": [{"of": "delta_cb", "z": self.z, "k": self._pk_k}],
            "fourier.sigma8_z": [
                {"of": "delta_cb", "z": self.z},
                {"of": "theta_cb", "z": self.z},
            ],
        }
        if self.ap:
            requirements.update({
                "background.efunc": [{"z": self.z}],
                "background.comoving_transverse_distance": [{"z": self.z}],
            })
            options = _pmm_ap_options()
            fiducial = dict(options["fiducial"])
            fiducial.pop("engine", None)
            import cosmoprimo
            from scipy import special
            reference = cosmoprimo.Cosmology(engine="class", **fiducial)
            self._DH_fid = float(299792.458 / (100. * reference.efunc(self.z)))
            self._DM_fid = float(reference.comoving_angular_distance(self.z))
            mu, weights = np.polynomial.legendre.leggauss(2 * options["mu"])
            self._mu = mu[options["mu"]:]
            weights = weights[options["mu"]:]
            self._legendre_weights = np.asarray([
                2. * weights * (2 * ell + 1) / 2. * special.eval_legendre(ell, self._mu)
                for ell in self.ells
            ])
        self.cosmo.add_requirements(requirements)

    def __call__(self):
        fourier = self.cosmo.get_fourier()
        pk = fourier.pk(of="delta_cb", z=self.z, k=self._pk_k)
        f = (fourier.sigma8_z(of="theta_cb", z=self.z)
             / fourier.sigma8_z(of="delta_cb", z=self.z))
        if self.ap:
            background = self.cosmo.get_background()
            qpar = (299792.458 / (100. * background.efunc(z=self.z))) / self._DH_fid
            qper = background.comoving_transverse_distance(z=self.z) / self._DM_fid
            qap = qpar / qper
            mu, k = jnp.asarray(self._mu)[None, :], jnp.asarray(self.k)[:, None]
            factor = jnp.sqrt(1. + mu**2 * (1. / qap**2 - 1.))
            kap, muap = k / qper * factor, mu / qap / factor
            anisotropic = jnp.exp(jnp.interp(jnp.log(kap), jnp.log(self._pk_k), jnp.log(pk)))
            anisotropic /= qpar * qper**2
            if self.smoothing_radius:
                anisotropic *= jnp.exp(-0.5 * (kap * self.smoothing_radius)**2)
            anisotropic *= (1. + f * muap**2)**2
            self.power = jnp.asarray(self._legendre_weights) @ anisotropic.T
        else:
            if self.smoothing_radius:
                pk *= jnp.exp(-0.5 * (jnp.asarray(self.k) * self.smoothing_radius)**2)
            if self.rsd:
                factors = {0: 1. + 2. * f / 3. + f**2 / 5.,
                           2: 4. * f / 3. + 4. * f**2 / 7.,
                           4: 8. * f**2 / 35.}
                self.power = jnp.stack([factors[ell] * pk for ell in self.ells])
            else:
                self.power = pk[None, :]
        self.poles = self.power
        return self.power

    def tree_flatten(self):
        return [self.power], dict(k=self.k, z=self.z, ells=self.ells, rsd=self.rsd,
                                  ap=self.ap, smoothing_radius=self.smoothing_radius)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.power = obj.poles = children[0]
        for name, value in aux.items():
            setattr(obj, name, value)
        return obj


def _pqm_ap_smoothing(smoothing_radius):
    radius = float(smoothing_radius)
    if not np.isfinite(radius) or radius < 0.:
        raise ValueError("smoothing_radius must be finite and non-negative")
    return dict(model_version=1, kernel="gaussian", radius=radius, coordinates="true")


class SmoothedAPMatterPowerTheory(MatterPowerTheory):
    """AP matter kernel with Gaussian smoothing in true coordinates."""

    def __init__(self, k=None, z=0., ells=(0, 2, 4), rsd=True, ap=True,
                 smoothing_radius=10., engine="class"):
        if not ap:
            raise ValueError("the smoothed AP kernel requires ap=True")
        super().__init__(k=k, z=z, ells=ells, rsd=rsd, ap=ap, engine=engine,
                         smoothing_radius=_pqm_ap_smoothing(smoothing_radius)["radius"])


class APDensitySplitMatterPowerSpectrumMultipoles(DensitySplitMatterPowerSpectrumMultipoles):
    """Apply exact quantile responses to an already smoothed AP kernel."""

    def __init__(self, k=None, z=0., ells=(0, 2, 4), quantiles=(1, 2, 3, 4, 5),
                 smoothing_radius=10., rsd=True, matter=None):
        if matter is None:
            raise ValueError("provide a smoothed AP matter kernel")
        self.ap_smoothing_radius = float(smoothing_radius)
        super().__init__(k=k, z=z, ells=ells, quantiles=quantiles,
                         smoothing_radius=0., rsd=rsd, matter=matter)
        if not self.rsd or not getattr(matter, "ap", False):
            raise ValueError("the pQm AP wrapper requires an RSD AP kernel")
        if self.ap_smoothing_radius != float(getattr(matter, "smoothing_radius", 0.)):
            raise ValueError("matter kernel smoothing radius must match pQm")
        self.ap = True

    def __call__(self):
        power = jnp.asarray(self.matter.power)
        shape = (len(self.ells), self.k.size)
        if power.shape not in (shape, (shape[0] * shape[1],)):
            raise ValueError("matter power must have shape (nells, nk) or be ell-major flattened")
        responses = jnp.stack([param.value for param in self.response_params])
        self.power = responses[:, None, None] * power.reshape(shape)[None, :, :]
        self.poles = self.power
        return self.power


def _build_pqm_ap_kernel(k, *, redshift, ells, smoothing_radius, rsd,
                         cosmo_engine="class"):
    radius = _pqm_ap_smoothing(smoothing_radius)["radius"]
    if radius == 0.:
        return build_pmm_theory(k, redshift=redshift, ells=ells, rsd=rsd, ap=True,
                                cosmo_engine=cosmo_engine)
    return SmoothedAPMatterPowerTheory(
        k=k, z=redshift, ells=ells, rsd=rsd, smoothing_radius=radius,
        engine=cosmo_engine,
    )


def build_pmm_theory(
    k,
    *,
    redshift: float = 0.0,
    ells: tuple[int, ...] | list[int] = (0,),
    rsd: bool = False,
    ap: bool = False,
    cosmo_engine: str = "class",
):
    """Build the exact calculator used to train or check the pmm emulator."""
    _validate_pmm_ap(ap=ap, rsd=rsd)
    selected_ells = tuple(ells)
    if not rsd and selected_ells != (0,):
        raise ValueError("real-space pmm supports only ell=0")
    return MatterPowerTheory(
        k=np.asarray(k, dtype="f8"),
        z=float(redshift),
        ells=selected_ells,
        rsd=bool(rsd),
        ap=bool(ap),
        engine=cosmo_engine,
    )


def _pmm_emulator_cache_options(
    k,
    *,
    redshift: float,
    ells: tuple[int, ...] | list[int],
    rsd: bool,
    ap: bool = False,
    emulator: str = "taylor",
    mlp_config=None,
) -> dict:
    """Return the complete configuration defining a reusable emulator."""
    _validate_pmm_ap(ap=ap, rsd=rsd, emulator=emulator)
    options = emulation.matter_cache_options(
        k, redshift=redshift, ells=ells, rsd=rsd,
        cache_version=PMM_EMULATOR_CACHE_VERSION, m_ncdm=PMM_M_NCDM,
        order=PMM_EMULATOR_ORDER, accuracy=PMM_EMULATOR_ACCURACY,
        method=PMM_EMULATOR_METHOD, emulator=emulator, mlp_config=mlp_config,
    )
    if ap:
        options["ap"] = _pmm_ap_options()
    return options


def _pmm_emulator_path(
    output_dir: Path,
    k,
    *,
    redshift: float,
    ells: tuple[int, ...] | list[int],
    rsd: bool,
    ap: bool = False,
    emulator: str = "taylor",
    mlp_config=None,
) -> Path:
    options = _pmm_emulator_cache_options(
        k, redshift=redshift, ells=ells, rsd=rsd, ap=ap,
        emulator=emulator, mlp_config=mlp_config,
    )
    return emulation.emulator_path(output_dir, options)


def get_or_train_pmm_emulator(
    output_dir: Path,
    k,
    *,
    redshift: float = 0.0,
    ells: tuple[int, ...] | list[int] = (0,),
    rsd: bool = False,
    ap: bool = False,
    emulator: str = "taylor",
    mlp_config=None,
):
    """Load a matching cached emulator, or train and save it if absent."""
    options = _pmm_emulator_cache_options(
        k, redshift=redshift, ells=ells, rsd=rsd, ap=ap,
        emulator=emulator, mlp_config=mlp_config,
    )
    return emulation.get_or_train_emulator(
        output_dir, options,
        lambda: build_pmm_theory(k, redshift=redshift, ells=ells, rsd=rsd, ap=ap),
    )


def build_pqm_theory(
    k,
    *,
    redshift: float = 0.0,
    ells: tuple[int, ...] | list[int] = (0,),
    quantiles: tuple[int, ...] | list[int] = (1, 2, 4, 5),
    smoothing_radius: float = 10.0,
    rsd: bool = False,
    ap: bool = False,
    cosmo_engine: str = "class",
):
    """Build the minimal massless-neutrino density-split matter model."""
    _validate_pmm_ap(ap=ap, rsd=rsd)
    selected_ells = tuple(ells)
    if not rsd and selected_ells != (0,):
        raise ValueError("real-space pQm supports only ell=0")
    if ap:
        matter = _build_pqm_ap_kernel(
            k, redshift=redshift, ells=selected_ells,
            smoothing_radius=smoothing_radius, rsd=rsd, cosmo_engine=cosmo_engine,
        )
        return APDensitySplitMatterPowerSpectrumMultipoles(
            k=k, z=redshift, ells=selected_ells, quantiles=quantiles,
            smoothing_radius=smoothing_radius, rsd=rsd, matter=matter,
        )
    return DensitySplitMatterPowerSpectrumMultipoles(
        k=np.asarray(k, dtype="f8"),
        z=float(redshift),
        ells=selected_ells,
        quantiles=tuple(quantiles),
        smoothing_radius=float(smoothing_radius),
        rsd=bool(rsd),
        cosmo=_build_cosmology(cosmo_engine),
    )


def _pqm_emulator_cache_options(
    k,
    *,
    redshift: float,
    ells: tuple[int, ...] | list[int],
    quantiles: tuple[int, ...] | list[int],
    smoothing_radius: float,
    rsd: bool,
    ap: bool = False,
    emulator: str = "taylor",
    mlp_config=None,
) -> dict:
    kernel = _pmm_emulator_cache_options(
        k, redshift=redshift, ells=ells, rsd=rsd, ap=ap,
        emulator=emulator, mlp_config=mlp_config,
    )
    if ap:
        smoothing = _pqm_ap_smoothing(smoothing_radius)
        if smoothing["radius"] != 0.:
            kernel["smoothing"] = smoothing
    return {
        "cache_version": PQM_EMULATOR_CACHE_VERSION,
        "model": PQM_AP_MODEL_VERSION if ap else PQM_MODEL_VERSION,
        "quantiles": [int(quantile) for quantile in quantiles],
        "smoothing_radius": float(smoothing_radius),
        "kernel": kernel,
    }


def get_or_train_pqm_emulator(
    output_dir: Path,
    k,
    *,
    redshift: float = 0.0,
    ells: tuple[int, ...] | list[int] = (0,),
    quantiles: tuple[int, ...] | list[int] = (1, 2, 4, 5),
    smoothing_radius: float = 10.0,
    rsd: bool = False,
    ap: bool = False,
    emulator: str = "taylor",
    mlp_config=None,
):
    """Emulate the matter kernel, retaining exact quantile responses.

    With AP, smoothing is inside the radius-dependent emulated kernel.
    Legacy full-pQm emulator files are intentionally neither read nor modified.
    """
    _validate_pmm_ap(ap=ap, rsd=rsd, emulator=emulator)
    if ap:
        options = _pqm_emulator_cache_options(
            k, redshift=redshift, ells=ells, quantiles=quantiles,
            smoothing_radius=smoothing_radius, rsd=rsd, ap=True,
            emulator=emulator, mlp_config=mlp_config,
        )
        matter = emulation.get_or_train_emulator(
            output_dir, options["kernel"],
            lambda: _build_pqm_ap_kernel(k, redshift=redshift, ells=ells,
                                         smoothing_radius=smoothing_radius, rsd=rsd),
        )
        return APDensitySplitMatterPowerSpectrumMultipoles(
            k=k, z=redshift, ells=ells, quantiles=quantiles,
            smoothing_radius=smoothing_radius, rsd=rsd, matter=matter,
        )
    matter = get_or_train_pmm_emulator(
        output_dir,
        k,
        redshift=redshift,
        ells=ells,
        rsd=rsd,
        ap=False,
        **(dict(emulator=emulator, mlp_config=mlp_config)
           if emulator != "taylor" or mlp_config is not None else {}),
    )
    return DensitySplitMatterPowerSpectrumMultipoles(
        matter=matter,
        k=k,
        z=redshift,
        ells=ells,
        quantiles=quantiles,
        smoothing_radius=smoothing_radius,
        rsd=rsd,
    )


def build_pmm_likelihood(
    measurements: ClusteringMeasurements,
    covariance: np.ndarray,
    *,
    window,
    redshift: float = 0.0,
    rsd: bool = False,
    ap: bool = False,
    theory=None,
):
    """Build a window-convolved Gaussian matter-power likelihood."""
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.observables.galaxy_clustering import (
        Spectrum2PolesObservable,
    )

    _validate_pmm_ap(ap=ap, rsd=rsd)
    selected_ells = measurements.ells
    if not rsd and selected_ells != (0,):
        raise ValueError("real-space pmm supports only ell=0")
    window = validate_pmm_window(window, measurements, rsd=rsd)

    if theory is None:
        theory_ells = tuple(window.theory.ells)
        theory_k = window.theory.get(ells=theory_ells[0]).coords("k")
        theory = build_pmm_theory(
            theory_k,
            redshift=redshift,
            ells=theory_ells,
            rsd=rsd,
            ap=ap,
        )
    data = window.observable.clone(value=measurements.data)
    observable = Spectrum2PolesObservable(
        name="pmm",
        theory=theory,
        data=data,
        window=window,
        covariance=covariance,
    )
    return ObservablesGaussianLikelihood(observables=[observable])


class WindowedDensitySplitPowerSpectrumMultipolesObservable(Calculator):
    """Repository-local adapter from quantile theory to an lsstypes window."""

    def __init__(self, data=None, theory=None, window=None, name="pqm"):
        self.name = str(name)
        self.data = data
        self.theory = theory
        self.window = window
        self.flatdata = np.asarray(data.value(), dtype="f8").ravel()
        value = np.asarray(window.value(), dtype="f8")
        if value.shape[0] != self.flatdata.size:
            raise ValueError("pqm data and window observable sizes do not match")

    def __call__(self):
        matrix = jnp.asarray(self.window.value())
        flatpower = jnp.ravel(self.theory.power)
        if matrix.shape[1] != flatpower.size:
            raise ValueError("pqm theory and window theory sizes do not match")
        self.flattheory = matrix @ flatpower
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


def _build_pmm_observable(measurements, *, window, theory, covariance=None):
    from desilike.observables.galaxy_clustering import (
        Spectrum2PolesObservable,
    )

    data = window.observable.clone(value=measurements.data)
    kwargs = {}
    if covariance is not None:
        kwargs["covariance"] = covariance
    return Spectrum2PolesObservable(
        name="pmm", theory=theory, data=data, window=window, **kwargs
    )


def _kaiser_multipoles(
    linear_power: np.ndarray,
    growth_rate: float,
    ells: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Return linear matter Kaiser multipoles flattened in ell-major order."""
    f = growth_rate
    factors = {
        0: 1.0 + 2.0 * f / 3.0 + f**2 / 5.0,
        2: 4.0 * f / 3.0 + 4.0 * f**2 / 7.0,
        4: 8.0 * f**2 / 35.0,
    }
    power = np.asarray(linear_power)
    return np.concatenate([factors[ell] * power for ell in ells])


def _build_pqm_observable(measurements, *, window, theory):
    data = window.observable.clone(value=measurements.data)
    return WindowedDensitySplitPowerSpectrumMultipolesObservable(
        name="pqm", theory=theory, data=data, window=window
    )


def build_density_split_likelihood(
    measurements: ClusteringMeasurements,
    covariance: np.ndarray,
    *,
    window,
    redshift: float = 0.0,
    rsd: bool = False,
    ap: bool = False,
    smoothing_radius: float = 10.0,
    theory=None,
):
    """Build a standalone window-convolved pQm Gaussian likelihood."""
    from desilike.likelihoods import ObservablesGaussianLikelihood

    _validate_pmm_ap(ap=ap, rsd=rsd)
    window = validate_pqm_window(window, measurements, rsd=rsd)
    if theory is None:
        first_quantile = window.theory.quantiles[0]
        theory_branch = window.theory.get(quantiles=first_quantile)
        theory_ells = tuple(theory_branch.ells)
        theory_k = theory_branch.get(ells=theory_ells[0]).coords("k")
        theory = build_pqm_theory(
            theory_k,
            redshift=redshift,
            ells=theory_ells,
            quantiles=measurements.quantiles,
            smoothing_radius=smoothing_radius,
            rsd=rsd,
            ap=ap,
        )
    observable = _build_pqm_observable(measurements, window=window, theory=theory)
    return ObservablesGaussianLikelihood(
        observables=[observable], covariance=covariance
    )


def build_joint_likelihood(
    measurements: MeasurementBundle,
    covariance: np.ndarray,
    *,
    windows: dict[str, object],
    theories: dict[str, Calculator],
    rsd: bool = False,
):
    """Build a global-covariance pmm+pQm likelihood in requested order."""
    from desilike.likelihoods import ObservablesGaussianLikelihood

    observables = []
    for stat in measurements.statistics:
        selected = measurements.by_stat[stat]
        if stat == "pmm":
            validate_pmm_window(windows[stat], selected, rsd=rsd)
            observables.append(
                _build_pmm_observable(
                    selected, window=windows[stat], theory=theories[stat]
                )
            )
        elif stat == "pqm":
            validate_pqm_window(windows[stat], selected, rsd=rsd)
            observables.append(
                _build_pqm_observable(
                    selected, window=windows[stat], theory=theories[stat]
                )
            )
        else:
            raise ValueError(f"unsupported statistic {stat!r}")
    return ObservablesGaussianLikelihood(observables=observables, covariance=covariance)


def _bestfit_values(profiles) -> dict[str, float]:
    """Return the varied parameter values at the best profile point."""
    chosen = profiles.choice(squeeze=True)
    return {
        str(name): float(np.asarray(value).reshape(-1)[-1])
        for name, value in chosen.best.items()
    }


def plot_pmm_fit(
    path: Path,
    measurements: ClusteringMeasurements,
    theory: np.ndarray,
    covariance: np.ndarray,
    *,
    rsd: bool,
) -> None:
    """Plot best-fit pmm multipoles and diagonal-normalized residuals."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    k = np.asarray(measurements.k, dtype="f8")
    nells, nk = len(measurements.ells), k.size
    data = np.asarray(measurements.data, dtype="f8").reshape(nells, nk)
    prediction = np.asarray(theory, dtype="f8").reshape(nells, nk)
    error = np.sqrt(np.diag(np.asarray(covariance, dtype="f8"))).reshape(nells, nk)

    figure, axes = plt.subplots(
        2,
        nells,
        figsize=(5.0 * nells, 6.0),
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    for index, ell in enumerate(measurements.ells):
        upper, lower = axes[:, index]
        upper.errorbar(
            k,
            k * data[index],
            yerr=k * error[index],
            fmt="o",
            ms=4,
            label="ensemble mean",
        )
        upper.plot(
            k,
            k * prediction[index],
            lw=1.8,
            label="profile best fit",
        )
        upper.set_title(rf"$\ell = {ell}$")
        upper.set_ylabel(rf"$kP_{{mm,{ell}}}(k)$")
        upper.legend(frameon=False)

        lower.plot(
            k,
            (data[index] - prediction[index]) / error[index],
            marker="o",
            ms=3,
        )
        lower.axhline(0.0, color="0.4", lw=0.8)
        lower.axhline(2.0, color="0.6", ls="--", lw=0.8)
        lower.axhline(-2.0, color="0.6", ls="--", lw=0.8)
        lower.set_ylabel(r"$\Delta P/\sigma$")
        lower.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        for axis in (upper, lower):
            axis.grid(alpha=0.2)

    space = "Redshift-space" if rsd else "Real-space"
    figure.suptitle(f"{space} matter power spectrum")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_pqm_fit(
    path: Path,
    measurements: ClusteringMeasurements,
    theory: np.ndarray,
    covariance: np.ndarray,
    *,
    rsd: bool,
) -> None:
    """Plot every selected pQm quantile and multipole."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    k = np.asarray(measurements.k, dtype="f8")
    shape = (len(measurements.quantiles), len(measurements.ells), k.size)
    data = np.asarray(measurements.data, dtype="f8").reshape(shape)
    prediction = np.asarray(theory, dtype="f8").reshape(shape)
    error = np.sqrt(np.diag(np.asarray(covariance, dtype="f8"))).reshape(shape)
    figure, axes = plt.subplots(
        2,
        len(measurements.ells),
        figsize=(5.0 * len(measurements.ells), 6.0),
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(measurements.quantiles)))
    for iell, ell in enumerate(measurements.ells):
        upper, lower = axes[:, iell]
        for iq, (quantile, color) in enumerate(zip(measurements.quantiles, colors)):
            upper.errorbar(
                k,
                k * data[iq, iell],
                yerr=k * error[iq, iell],
                fmt="o",
                ms=3,
                color=color,
                label=rf"$Q_{quantile}$",
            )
            upper.plot(k, k * prediction[iq, iell], lw=1.6, color=color)
            lower.plot(
                k,
                (data[iq, iell] - prediction[iq, iell]) / error[iq, iell],
                marker="o",
                ms=2.5,
                color=color,
            )
        upper.set_title(rf"$\ell = {ell}$")
        upper.set_ylabel(rf"$kP_{{Qm,{ell}}}(k)$")
        upper.legend(frameon=False, ncol=2)
        lower.axhline(0.0, color="0.4", lw=0.8)
        lower.axhline(2.0, color="0.6", ls="--", lw=0.8)
        lower.axhline(-2.0, color="0.6", ls="--", lw=0.8)
        lower.set_ylabel(r"$\Delta P/\sigma$")
        lower.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        for axis in (upper, lower):
            axis.grid(alpha=0.2)
    space = "Redshift-space" if rsd else "Real-space"
    figure.suptitle(f"{space} density-split matter power spectrum")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def profile(likelihood, output: Path, *, seed: int = 42, niterations: int = 5):
    """Maximize the posterior and save the resulting profiles."""
    from desilike import compile
    from desilike.profilers import Minuit, Profiler

    profiles = Profiler(compile(likelihood), kernel=Minuit(), rng=seed,
                        output_fn=output).maximize(niterations=niterations)
    profiles.write(output)
    print(profiles.to_stats(tablefmt="pretty"))
    return profiles


def fit_output_hash(args, measurements, covariance, windows) -> str:
    """Identify a fit independently of its output directory or stopping budget.

    The target, effective windows, theory configuration, and random seed define
    the hash. Profile/sample/both and max_iterations do not change the target.
    """

    def array_digest(value):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest = hashlib.sha256()
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        return digest.hexdigest()

    statistics = []
    for stat in measurements.statistics:
        selected = measurements.by_stat[stat]
        window = windows[stat]
        branch = window.theory
        if stat == "pqm":
            branch = branch.get(quantiles=selected.quantiles[0] - 1)
        ells = tuple(branch.ells)
        k = branch.get(ells=ells[0]).coords("k")
        options = dict(redshift=args.redshift, ells=ells, rsd=args.rsd)
        if getattr(args, "emulator", "taylor") == "mlp":
            options.update(emulator="mlp", mlp_config=args.mlp_options)
        if getattr(args, "direct_cosmology", args.no_emulator):
            theory_config = {
                "evaluation": "exact",
                "cosmo_engine": getattr(args, "cosmo_engine", "class") or "class",
                "model_version": 1,
                "model": "linear-delta-cb-kaiser",
                "k": np.asarray(k, dtype="f8").tolist(),
                **options,
                "fixed_cosmology": {"m_ncdm": PMM_M_NCDM},
            }
            if stat == "pqm":
                theory_config.update(
                    model="minimal-density-split-matter-kaiser",
                    quantiles=list(selected.quantiles),
                    smoothing_radius=args.smoothing_radius,
                )
                if getattr(args, "ap", False):
                    theory_config.update(
                        model=PQM_AP_MODEL_VERSION,
                        ap=_pmm_ap_options(),
                        smoothing=_pqm_ap_smoothing(args.smoothing_radius),
                    )
        elif stat == "pqm":
            theory_config = _pqm_emulator_cache_options(
                k,
                quantiles=selected.quantiles,
                smoothing_radius=args.smoothing_radius,
                ap=getattr(args, "ap", False),
                **options,
            )
        else:
            theory_config = _pmm_emulator_cache_options(k, **options)
        if stat == "pmm" and getattr(args, "ap", False):
            theory_config["ap"] = _pmm_ap_options()
        statistics.append(
            {
                "stat": stat,
                "ells": list(selected.ells),
                "quantiles": list(selected.quantiles),
                "k": array_digest(selected.k),
                "k_edges": array_digest(selected.k_edges),
                "nmodes": array_digest(selected.nmodes),
                "los": array_digest(selected.los),
                "window": array_digest(window.value()),
                "theory" if getattr(args, "direct_cosmology", args.no_emulator) else "emulator": theory_config,
            }
        )
    options = {
        "version": 1,
        "input_root": str(args.input_root.resolve()),
        "statistics": statistics,
        "data": array_digest(measurements.data),
        "covariance": array_digest(covariance),
        "covariance_of_mean": args.covariance_of_mean,
        "fixed_cosmology": {
            name: QUIJOTE_PARAMETER_VALUES[name] for name in args.fix_cosmo
        },
        "seed": args.seed,
        "chains": args.chains,
    }
    serialized = json.dumps(
        options, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]


def sample(
    likelihood,
    output_dir: Path,
    *,
    chains: int = 4,
    seed: int = 42,
    max_iterations: int = 10_000,
    proposal=None,
    start_from_bestfit: bool = False,
    output_hash: str | None = None,
):
    """Run MCMC and save one file per chain."""
    from desilike import compile, get_params
    from desilike.samplers import MH, Sampler

    graph = compile(likelihood)
    start = None
    if start_from_bestfit:
        if proposal is None:
            raise ValueError("starting from the best fit requires profile results")
        bestfit = _bestfit_values(proposal)
        names = get_params(likelihood).select(varied=True, derived=False).names()
        missing = sorted(set(names) - bestfit.keys())
        if missing:
            raise ValueError(f"profile best fit is missing varied parameters: {missing}")
        start = {name: bestfit[name] for name in names}
        if not np.isfinite(list(start.values())).all() or not np.isfinite(graph(start)):
            raise ValueError("profile best fit must have finite values and posterior")

    if start is not None:
        for param in get_params(likelihood).select(varied=True, derived=False):
            param.update(value=start[param.name], ref={})
    chain_dir = output_dir / (f"chains_{output_hash}" if output_hash else "chains")
    sampler = Sampler(
        graph, kernel=MH(), nparallel=chains, rng=seed,
        output_dir=chain_dir, proposal=proposal,
    )
    return sampler.run(
        gelman_rubin=1.03,
        check_every=500,
        max_steps=max_iterations,
    )


def parse_args(argv=None):
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stats", nargs="+", choices=("pmm", "pqm", "pqq"), default=["pmm"]
    )
    parser.add_argument(
        "--ells",
        type=int,
        nargs="+",
        choices=(0, 2, 4),
        default=[0],
        help="One or more output multipoles for every selected statistic.",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        nargs="+",
        choices=(1, 2, 3, 4, 5),
        default=[1, 2, 4, 5],
        help="One-based pQm quantiles (all five are intentionally disallowed).",
    )
    parser.add_argument(
        "--smoothing-radius",
        type=float,
        default=10.0,
        help="Gaussian density smoothing radius in Mpc/h for the pQm model.",
    )
    parser.add_argument(
        "--rsd",
        action="store_true",
        help="Fit the selected multipoles with linear Kaiser RSD theory.",
    )
    parser.add_argument(
        "--ap",
        action="store_true",
        help="Apply cosmology-derived AP distortions relative to Quijote; "
             "requires RSD with exact or Taylor evaluation (pmm, pqm, or joint).",
    )
    parser.add_argument("--kmin", type=float, default=0.01)
    parser.add_argument("--kmax", type=float, default=0.2)
    parser.add_argument("--redshift", type=float, default=0.0)
    parser.add_argument(
        "--window",
        type=Path,
        help=(
            "precomputed lsstypes window matrix; by default an exact "
            "periodic-box window is cached under --output-dir"
        ),
    )
    parser.add_argument(
        "--pmm-window", type=Path, help="precomputed pmm lsstypes window matrix"
    )
    parser.add_argument(
        "--pqm-window", type=Path, help="precomputed pQm lsstypes window matrix"
    )
    parser.add_argument(
        "--boxsize",
        type=float,
        default=PMM_WINDOW_BOXSIZE,
        help="periodic box side length in Mpc/h used to build the window",
    )
    parser.add_argument(
        "--meshsize",
        type=int,
        default=PMM_WINDOW_MESHSIZE,
        help="FFT mesh size used to build the periodic-box window",
    )
    parser.add_argument(
        "--covariance-scale",
        type=float,
        default=1.0,
        help=(
            "multiply the entire single-box covariance by this finite, strictly "
            "positive factor (default: 1.0); with --covariance-of-mean, also "
            "divide by the number of realizations"
        ),
    )
    parser.add_argument(
        "--covariance-of-mean",
        action="store_true",
        help=(
            "divide the realization covariance by the number of realizations; "
            "the default uses the covariance of one realization"
        ),
    )
    parser.add_argument(
        "--method", choices=("profile", "sample", "both"), default="profile"
    )
    parser.add_argument(
        "--start-from-bestfit", action="store_true",
        help="start every chain at the profile best fit; with --method sample, "
             "load profiles_<fit-hash>.h5 from --output-dir; with both, use the fresh profile",
    )
    parser.add_argument(
        "--no-emulator",
        action="store_true",
        help="evaluate exact Cosmoprimo/CLASS theories; bypass emulator loading and training",
    )
    parser.add_argument(
        "--cosmo-engine", choices=("ace", "class", "eisenstein_hu"),
        help="use the refactored cosmology graph directly; ace selects JAX/MAPSE",
    )
    parser.add_argument("--emulator", choices=("taylor", "mlp"), default="taylor")
    parser.add_argument("--mlp-config", type=Path,
                        help="YAML with five cosmology bounds, optional ntrain and training seed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument(
        "--fix-cosmo",
        nargs="+",
        choices=tuple(QUIJOTE_PARAMETER_VALUES),
        default=[],
        help="fix selected cosmological parameters to Quijote values; reuse existing emulators",
    )
    args = parser.parse_args(argv)
    if args.emulator == "mlp":
        parser.error("MLP emulation is not yet ported to the refactor-jax desilike API")
    args.direct_cosmology = args.no_emulator or args.cosmo_engine is not None
    if args.no_emulator:
        if args.cosmo_engine not in (None, "class"):
            parser.error("--no-emulator is an alias for --cosmo-engine class")
        args.cosmo_engine = "class"

    if not np.isfinite(args.covariance_scale) or args.covariance_scale <= 0:
        parser.error("--covariance-scale must be finite and strictly positive")

    if args.start_from_bestfit and args.method == "profile":
        parser.error("--start-from-bestfit requires --method sample or both")
    if args.ap:
        if not args.rsd:
            parser.error("--ap requires --rsd")
        if not args.direct_cosmology and args.emulator != "taylor":
            parser.error("--ap supports only exact or Taylor evaluation")

    args.mlp_options = None
    if args.emulator == "mlp":
        if args.direct_cosmology:
            parser.error("--cosmo-engine/--no-emulator cannot be combined with --emulator mlp")
        if args.mlp_config is None:
            parser.error("--emulator mlp requires --mlp-config")
        try:
            args.mlp_options = emulation.read_mlp_config(args.mlp_config)
            emulation.check_fixed_bounds(args.mlp_options, {
                name: QUIJOTE_PARAMETER_VALUES[name] for name in args.fix_cosmo})
        except ValueError as exc:
            parser.error(str(exc))
    elif args.mlp_config is not None:
        parser.error("--mlp-config requires --emulator mlp")
    if len(set(args.fix_cosmo)) != len(args.fix_cosmo):
        parser.error("--fix-cosmo entries must be unique")
    if "pqq" in args.stats:
        parser.error("--stats pqq is not implemented")
    if len(set(args.stats)) != len(args.stats):
        parser.error("--stats entries must be unique")
    if not args.stats or len(args.stats) > 2 or set(args.stats) - {"pmm", "pqm"}:
        parser.error("--stats must be pmm, pqm, or both")
    if not args.rsd and args.ells != [0]:
        parser.error("real-space fits support only --ells 0")
    if "pqm" in args.stats:
        if len(set(args.quantiles)) != len(args.quantiles):
            parser.error("--quantiles entries must be unique")
        if set(args.quantiles) == set(range(1, 6)):
            parser.error(
                "all five pQm quantiles make the empirical covariance singular"
            )
        if not np.isfinite(args.smoothing_radius) or args.smoothing_radius < 0.0:
            parser.error("--smoothing-radius must be finite and non-negative")
    if args.window is not None:
        if len(args.stats) != 1:
            parser.error(
                "--window is ambiguous for joint fits; use statistic-specific windows"
            )
        destination = f"{args.stats[0]}_window"
        if getattr(args, destination) is not None:
            parser.error(
                f"--window cannot be combined with --{destination.replace('_', '-')}"
            )
        setattr(args, destination, args.window)
    if "pmm" not in args.stats and args.pmm_window is not None:
        parser.error("--pmm-window was supplied without --stats pmm")
    if "pqm" not in args.stats and args.pqm_window is not None:
        parser.error("--pqm-window was supplied without --stats pqm")
    if args.kmin >= args.kmax:
        parser.error("--kmin must be smaller than --kmax")
    if args.boxsize <= 0.0:
        parser.error("--boxsize must be positive")
    if args.meshsize <= 0:
        parser.error("--meshsize must be positive")
    return args


def main(argv=None) -> None:
    from desilike import setup_logging

    setup_logging()
    args = parse_args(argv)
    if args.ap:
        print("AP distortions: cosmology-derived geometry relative to massless Quijote")
    if args.direct_cosmology:
        label = "JAX/ACE+MAPSE" if args.cosmo_engine == "ace" else args.cosmo_engine
        print(f"Theory evaluation: direct refactored cosmology graph ({label})")
    emulator_options = {}
    if args.emulator == "mlp":
        emulator_options = dict(emulator="mlp", mlp_config=args.mlp_options)
        print(f"Theory evaluation: MLP; uniform cosmology priors: {args.mlp_options['bounds']}")
    if args.fix_cosmo:
        values = ", ".join(
            f"{name}={QUIJOTE_PARAMETER_VALUES[name]:.15g}" for name in args.fix_cosmo
        )
        print(f"Fixing cosmological parameters to Quijote values: {values}")

    measurements = read_inference_measurements(
        args.input_root,
        stats=args.stats,
        quantiles=args.quantiles,
        ells=args.ells,
        kmin=args.kmin,
        kmax=args.kmax,
    )
    covariance = build_covariance_matrix(
        measurements,
        covariance_of_mean=args.covariance_of_mean,
        covariance_scale=args.covariance_scale,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = {}
    theories = {}
    for stat in measurements.statistics:
        selected = measurements.by_stat[stat]
        if stat == "pmm":
            window = get_or_build_pmm_window(
                args.output_dir,
                selected,
                boxsize=args.boxsize,
                meshsize=args.meshsize,
                rsd=args.rsd,
                window_path=args.pmm_window,
            )
            theory_ells = tuple(window.theory.ells)
            theory_k = window.theory.get(ells=theory_ells[0]).coords("k")
            builder = build_pmm_theory if args.direct_cosmology else get_or_train_pmm_emulator
            theory = builder(
                *(() if args.direct_cosmology else (args.output_dir,)),
                theory_k,
                redshift=args.redshift,
                ells=theory_ells,
                rsd=args.rsd,
                ap=args.ap,
                **(dict(cosmo_engine=args.cosmo_engine) if args.direct_cosmology else {}),
                **emulator_options,
            )
        else:
            window = get_or_build_pqm_window(
                args.output_dir,
                selected,
                boxsize=args.boxsize,
                meshsize=args.meshsize,
                rsd=args.rsd,
                window_path=args.pqm_window,
            )
            first_quantile = window.theory.quantiles[0]
            theory_branch = window.theory.get(quantiles=first_quantile)
            theory_ells = tuple(theory_branch.ells)
            theory_k = theory_branch.get(ells=theory_ells[0]).coords("k")
            builder = build_pqm_theory if args.direct_cosmology else get_or_train_pqm_emulator
            theory = builder(
                *(() if args.direct_cosmology else (args.output_dir,)),
                theory_k,
                redshift=args.redshift,
                ells=theory_ells,
                quantiles=selected.quantiles,
                smoothing_radius=args.smoothing_radius,
                rsd=args.rsd,
                ap=args.ap,
                **(dict(cosmo_engine=args.cosmo_engine) if args.direct_cosmology else {}),
                **emulator_options,
            )
        windows[stat] = window
        theories[stat] = theory

    if measurements.statistics == ("pmm",):
        likelihood = build_pmm_likelihood(
            measurements.by_stat["pmm"],
            covariance,
            window=windows["pmm"],
            redshift=args.redshift,
            rsd=args.rsd,
            ap=args.ap,
            theory=theories["pmm"],
        )
    elif measurements.statistics == ("pqm",):
        likelihood = build_density_split_likelihood(
            measurements.by_stat["pqm"],
            covariance,
            window=windows["pqm"],
            redshift=args.redshift,
            rsd=args.rsd,
            smoothing_radius=args.smoothing_radius,
            ap=args.ap,
            theory=theories["pqm"],
        )
    else:
        likelihood = build_joint_likelihood(
            measurements,
            covariance,
            windows=windows,
            theories=theories,
            rsd=args.rsd,
        )

    if args.emulator == "mlp":
        emulation.apply_mlp_priors(likelihood, args.mlp_options)
    fix_quijote_cosmology(likelihood, args.fix_cosmo)
    output_hash = fit_output_hash(args, measurements, covariance, windows)
    print(f"Fit output hash: {output_hash}; directory: {args.output_dir.resolve()}")

    profiles = None
    if args.method in ("profile", "both"):
        profiles = profile(
            likelihood, args.output_dir / f"profiles_{output_hash}.h5", seed=args.seed
        )
        bestfit = _bestfit_values(profiles)
        from desilike import compile
        compile(likelihood)(bestfit)
        for stat, observable in zip(measurements.statistics, likelihood.observables):
            selected = measurements.by_stat[stat]
            selected_covariance = covariance[
                measurements.stat_slices[stat], measurements.stat_slices[stat]
            ]
            output = (
                args.output_dir / f"fit_{output_hash}.png"
                if len(measurements.statistics) == 1
                else args.output_dir / f"fit_{stat}_{output_hash}.png"
            )
            plotter = plot_pmm_fit if stat == "pmm" else plot_pqm_fit
            plotter(
                output,
                selected,
                observable.flattheory,
                selected_covariance,
                rsd=args.rsd,
            )
    if args.method in ("sample", "both"):
        if args.start_from_bestfit and profiles is None:
            from desilike.samples import Profiles

            profile_path = args.output_dir / f"profiles_{output_hash}.h5"
            if not profile_path.is_file():
                raise FileNotFoundError(
                    f"no profile best fit found at {profile_path}; "
                    "run --method both --start-from-bestfit to profile first"
                )
            profiles = Profiles.read(str(profile_path))
        if args.start_from_bestfit:
            print(f"Starting all {args.chains} chains at the profile best fit")
        sample(
            likelihood,
            args.output_dir,
            chains=args.chains,
            seed=args.seed,
            max_iterations=args.max_iterations,
            proposal=profiles,
            start_from_bestfit=args.start_from_bestfit,
            output_hash=output_hash,
        )


if __name__ == "__main__":
    main()
