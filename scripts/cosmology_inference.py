"""Run minimal cosmological inference from matter power measurements.

Only the real-space matter monopole (``pmm``) is supported for now. Density-
split hooks are included as placeholders for a later iteration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClusteringMeasurements:
    """Mean data vector and the realization vectors used to estimate it."""

    k: np.ndarray
    data: np.ndarray
    realizations: np.ndarray
    files: tuple[Path, ...]


def _read_pmm(path: Path, *, ell: int, kmin: float, kmax: float):
    import lsstypes

    leaf = lsstypes.read(path).get(ells=ell)
    k = np.asarray(leaf.coords("k"), dtype="f8")
    mask = (k >= kmin) & (k <= kmax)
    if not np.any(mask):
        raise ValueError(f"k cuts select no bins in {path}")
    return k[mask], np.asarray(leaf.value(), dtype="f8")[mask]


def read_clustering_measurements(
    root: Path,
    *,
    stat: str = "pmm",
    ell: int = 0,
    kmin: float = 0.01,
    kmax: float = 0.2,
) -> ClusteringMeasurements:
    """Read matching realization measurements and return their ensemble mean."""
    if stat != "pmm":
        raise NotImplementedError("only pmm is implemented")

    files = tuple(
        sorted(
            Path(root).glob("realization_*/pmm.h5"),
            key=lambda path: int(path.parent.name.removeprefix("realization_")),
        )
    )
    if not files:
        raise FileNotFoundError(f"no realization_*/pmm.h5 files under {root}")

    reference_k = None
    rows = []
    for path in files:
        k, power = _read_pmm(path, ell=ell, kmin=kmin, kmax=kmax)
        if reference_k is None:
            reference_k = k
        elif not np.array_equal(k, reference_k):
            raise ValueError(f"inconsistent k bins in {path}")
        rows.append(power)

    realizations = np.asarray(rows, dtype="f8")
    return ClusteringMeasurements(
        k=reference_k,
        data=np.mean(realizations, axis=0),
        realizations=realizations,
        files=files,
    )


def read_density_split_measurements(*args, **kwargs):
    """Read density-split measurements in a future iteration."""


def build_covariance_matrix(
    measurements: ClusteringMeasurements, *, covariance_of_mean: bool = True
) -> np.ndarray:
    """Estimate the covariance from realization scatter.

    By default the sample covariance is divided by the number of realizations,
    matching the ensemble mean stored in ``measurements.data``.
    """
    realizations = measurements.realizations
    if realizations.shape[0] < 2:
        raise ValueError("at least two realizations are required")
    covariance = np.atleast_2d(np.cov(realizations, rowvar=False, ddof=1))
    if covariance_of_mean:
        covariance /= realizations.shape[0]
    return covariance


def build_pmm_likelihood(
    measurements: ClusteringMeasurements,
    covariance: np.ndarray,
    *,
    redshift: float = 0.0,
):
    """Build a Gaussian likelihood for the linear real-space matter monopole."""
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood
    from desilike.theories import Cosmoprimo

    class MatterPowerTheory(BaseCalculator):
        def initialize(self, k=None, z=0.0):
            self.k = np.asarray(k, dtype="f8")
            self.z = float(z)
            self.cosmo = Cosmoprimo()
            self.cosmo.init.params["tau_reio"].update(fixed=True)

        def calculate(self):
            fourier = self.cosmo.cosmo.get_fourier()
            self.power = fourier.pk_interpolator(
                of="delta_cb"
            ).to_1d(z=self.z)(self.k)

        def get(self):
            return self.power

    class MatterPowerLikelihood(BaseGaussianLikelihood):
        def initialize(self, theory=None, data=None, covariance=None):
            self.theory = theory
            super().initialize(data=data, covariance=covariance)

        def calculate(self):
            self.flattheory = self.theory.power
            super().calculate()

    theory = MatterPowerTheory(k=measurements.k, z=redshift)
    return MatterPowerLikelihood(
        theory=theory,
        data=measurements.data,
        covariance=covariance,
    )


def build_density_split_likelihood(*args, **kwargs):
    """Build the density-split likelihood in a future iteration."""


def profile(likelihood, output: Path, *, seed: int = 42, niterations: int = 5):
    """Maximize the posterior and save the resulting profiles."""
    from desilike.profilers import MinuitProfiler

    profiles = MinuitProfiler(likelihood, seed=seed).maximize(
        niterations=niterations
    )
    profiles.save(output)
    return profiles


def sample(
    likelihood,
    output_dir: Path,
    *,
    chains: int = 4,
    seed: int = 42,
    max_iterations: int = 10_000,
    proposal=None,
):
    """Run MCMC and save one file per chain."""
    from desilike.samplers import MCMCSampler

    sampler = MCMCSampler(
        likelihood,
        chains=chains,
        seed=seed,
        covariance=proposal,
        save_fn=str(output_dir / "chain_*.npy"),
    )
    return sampler.run(
        check={"max_eigen_gr": 0.03},
        check_every=500,
        max_iterations=max_iterations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stats", nargs="+", choices=("pmm", "pqm", "pqq"), default=["pmm"]
    )
    parser.add_argument("--ell", type=int, default=0)
    parser.add_argument("--kmin", type=float, default=0.01)
    parser.add_argument("--kmax", type=float, default=0.2)
    parser.add_argument("--redshift", type=float, default=0.0)
    parser.add_argument(
        "--method", choices=("profile", "sample", "both"), default="profile"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    args = parser.parse_args()

    if args.stats != ["pmm"]:
        parser.error("only --stats pmm is implemented")
    if args.ell != 0:
        parser.error("the current real-space pmm model supports only --ell 0")
    if args.kmin >= args.kmax:
        parser.error("--kmin must be smaller than --kmax")

    measurements = read_clustering_measurements(
        args.input_root,
        ell=args.ell,
        kmin=args.kmin,
        kmax=args.kmax,
    )
    covariance = build_covariance_matrix(measurements)
    likelihood = build_pmm_likelihood(
        measurements, covariance, redshift=args.redshift
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles = None
    if args.method in ("profile", "both"):
        profiles = profile(
            likelihood, args.output_dir / "profiles.npy", seed=args.seed
        )
    if args.method in ("sample", "both"):
        sample(
            likelihood,
            args.output_dir,
            chains=args.chains,
            seed=args.seed,
            max_iterations=args.max_iterations,
            proposal=profiles,
        )


if __name__ == "__main__":
    main()
