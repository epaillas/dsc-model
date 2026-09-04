"""Cosmology-dependent kernels for real-space matter density-split fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np


QUANTILES = (1, 2, 4, 5)
KERNEL_NAMES = ("matter_base", "matter_alpha0", "quadratic_response")
QUIJOTE_COSMOLOGY = {
    "Omega_m": 0.3175,
    "Omega_b": 0.049,
    "h": 0.6711,
    "n_s": 0.9624,
    "sigma8": 0.834,
    "m_ncdm": 0.0,
    "N_eff": 3.046,
    "w0_fld": -1.0,
}
COSMOLOGY_PRIORS = {
    "Omega_m": (0.2, 0.4),
    "sigma8": (0.7, 0.95),
}
COSMOLOGY_DELTAS = {"Omega_m": 0.02, "sigma8": 0.025}
FIDUCIAL_ALPHA0 = -2.691269896234135
FIDUCIAL_NUISANCE = {
    "c1q1": -3.667571544313163,
    "c1q2": -1.5353773046205035,
    "c1q4": 1.3658970178905516,
    "c1q5": 3.9607439476777917,
    "c2q1": 10.085972329275345,
    "c2q2": -0.9562165376311159,
    "c2q4": -5.830672489354029,
    "c2q5": 1.4181194200237999,
}
FIDUCIAL_NUISANCE_STD = {
    "c1q1": 0.021724570717728535,
    "c1q2": 0.010606730215870561,
    "c1q4": 0.010021461853508344,
    "c1q5": 0.022536358629479405,
    "c2q1": 0.2504645474931798,
    "c2q2": 0.13997095197465692,
    "c2q4": 0.128761305480317,
    "c2q5": 0.27113470960457575,
}


def _uniform_parameter(value: float, limits: tuple[float, float], delta: float):
    return {
        "value": float(value),
        "prior": {"limits": list(limits)},
        "ref": {
            "dist": "norm",
            "loc": float(value),
            "scale": float(delta),
        },
        "delta": float(delta),
        "fixed": False,
    }


def build_quijote_cosmology():
    """Return a desilike cosmology varying only Omega_m and sigma8."""
    from cosmoprimo import Cosmology
    from desilike.theories import Cosmoprimo

    fiducial = Cosmology(**QUIJOTE_COSMOLOGY, engine="class")
    cosmology = Cosmoprimo(fiducial=fiducial)
    cosmology.init.params = {
        name: _uniform_parameter(
            QUIJOTE_COSMOLOGY[name], COSMOLOGY_PRIORS[name], COSMOLOGY_DELTAS[name]
        )
        for name in ("Omega_m", "sigma8")
    }
    return cosmology


class NoAPDirectPowerSpectrumTemplate:
    """Factory for a direct template with unit AP scaling.

    A factory keeps the desilike subclass definition at module scope while
    avoiding an eager desilike import when data-only utilities are imported.
    """

    def __new__(cls, *args, **kwargs):
        return _no_ap_template_class()(*args, **kwargs)


def _no_ap_template_class():
    from desilike.parameter import Parameter, ParameterCollection
    from desilike.theories.galaxy_clustering.power_template import (
        BasePowerSpectrumExtractor,
        BasePowerSpectrumTemplate,
        DirectPowerSpectrumTemplate,
    )

    class NoAPTemplate(DirectPowerSpectrumTemplate):
        config_fn = DirectPowerSpectrumTemplate

        def initialize(self, *args, cosmo=None, **kwargs):
            engine = kwargs.pop("engine", "class")
            BasePowerSpectrumTemplate.initialize(
                self, *args, apmode="qparqper", cosmo=cosmo, **kwargs
            )
            self.apeffect.init.params = ParameterCollection(
                [
                    Parameter("qpar", value=1.0, fixed=True),
                    Parameter("qper", value=1.0, fixed=True),
                ]
            )
            self.cosmo_requires = {}
            self.cosmo = cosmo
            params = self.init.params.select(derived=True)
            if cosmo is None:
                from desilike.theories import Cosmoprimo

                self.cosmo = Cosmoprimo(fiducial=self.fiducial, engine=engine)
                self.cosmo.init.params = [
                    parameter for parameter in self.params if parameter not in params
                ]
            self.init.params = params
            self.apeffect.init.update(cosmo=self.cosmo)
            state = BasePowerSpectrumExtractor._calculate(self, fiducial=True)
            self.__dict__.update(
                {f"{name}_fid": value for name, value in state.items()}
            )

    NoAPTemplate.__name__ = "_NoAPDirectPowerSpectrumTemplate"
    NoAPTemplate.__qualname__ = "_NoAPDirectPowerSpectrumTemplate"
    NoAPTemplate.__module__ = __name__
    return NoAPTemplate


def build_no_ap_template(*, k=None, z: float = 0.0):
    """Return a CLASS direct template with fixed unit AP parameters."""
    from cosmoprimo import Cosmology

    kwargs = {
        "z": float(z),
        "fiducial": Cosmology(**QUIJOTE_COSMOLOGY, engine="class"),
        "cosmo": build_quijote_cosmology(),
    }
    if k is not None:
        kwargs["k"] = np.asarray(k, dtype="f8")
    return NoAPDirectPowerSpectrumTemplate(**kwargs)


def _kernel_calculator_class():
    from desilike.jax import numpy as jnp
    from desilike.parameter import ParameterCollection
    from desilike.theories.galaxy_clustering.density_split import (
        DensitySplitTracerPowerSpectrumMultipoles,
        contract_p2_moments,
    )

    class KernelCalculator(DensitySplitTracerPowerSpectrumMultipoles):
        """Return matter, alpha0-response, and quadratic-response kernels."""

        config_fn = DensitySplitTracerPowerSpectrumMultipoles

        def initialize(self, *args, **kwargs):
            kwargs.update(
                model="1-loop",
                rsd=False,
                prior_basis="standard",
                quantiles=(5,),
                ells=(0,),
                smoothing_apmode="physical",
                backend="jax",
            )
            super().initialize(*args, **kwargs)
            self.init.params = ParameterCollection()
            self.kernel_names = KERNEL_NAMES

        def calculate(self, **params):
            self._set_from_pt()
            jac, kap, muap = self.pt.pt.jac, self.pt.pt.kap, self.pt.pt.muap
            mu = jnp.zeros_like(muap)
            base = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            shifted = list(base)
            shifted[4] = 1.0
            matter_base = self.to_poles(
                jac * self._folps_pkmu(kap, mu, base, shotnoise=0.0)
            )[0]
            matter_shifted = self.to_poles(
                jac * self._folps_pkmu(kap, mu, shifted, shotnoise=0.0)
            )[0]
            moments = self._composite_p2_moments(kap, mu, 0.0)
            quadratic = self.to_poles(
                jac * contract_p2_moments(moments, 1.0, 0.0, 0.0)
            )[0]
            self.kernels = jnp.stack(
                [matter_base, matter_shifted - matter_base, quadratic], axis=0
            )

        def get(self):
            return self.kernels

        def __getstate__(self):
            state = super().__getstate__()
            state["kernels"] = self.kernels
            state["kernel_names"] = self.kernel_names
            return state

    KernelCalculator.__name__ = "DensitySplitCosmologyKernels"
    KernelCalculator.__qualname__ = "DensitySplitCosmologyKernels"
    KernelCalculator.__module__ = __name__
    return KernelCalculator


class DensitySplitCosmologyKernels:
    """Lazy compatibility factory for the legacy exact-kernel calculator.

    Keeping the legacy calculator import lazy lets data-only covariance tools
    use this module under the refactored desilike API.  The exact one-loop
    tracer calculator itself will be ported separately if it is needed again.
    """

    def __new__(cls, *args, **kwargs):
        return _kernel_calculator_class()(*args, **kwargs)


def build_exact_kernels(
    k,
    *,
    z: float = 0.0,
    smoothing_radius: float = 10.0,
    loop_options: dict | None = None,
):
    """Build the exact cosmology-dependent kernel calculator."""
    options = dict(loop_options or {})
    return DensitySplitCosmologyKernels(
        k=np.asarray(k, dtype="f8"),
        z=float(z),
        template=build_no_ap_template(k=k, z=z),
        smoothing_radius=float(smoothing_radius),
        smoothing_kernel="gaussian",
        shotnoise=1.0,
        **options,
    )


def reconstruct_power(
    kernels,
    k,
    *,
    c1,
    c2,
    alpha0: float,
    smoothing_radius: float = 10.0,
    joint: bool = False,
):
    """Reconstruct matter and density-split spectra from the three kernels."""
    from desilike.jax import numpy as jnp

    kernels = jnp.asarray(kernels)
    if kernels.ndim != 2 or kernels.shape[0] != len(KERNEL_NAMES):
        raise ValueError("kernels must have shape (3, nk)")
    k = jnp.asarray(k)
    c1 = jnp.asarray(c1)
    c2 = jnp.asarray(c2)
    if c1.shape != (len(QUANTILES),) or c2.shape != (len(QUANTILES),):
        raise ValueError("c1 and c2 must contain Q1, Q2, Q4, and Q5")
    matter = kernels[0] + alpha0 * kernels[1]
    window = jnp.exp(-0.5 * (k * float(smoothing_radius)) ** 2)
    cross = c1[:, None] * window[None, :] * matter[None, :]
    cross = cross + c2[:, None] * kernels[2][None, :]
    if joint:
        return jnp.concatenate([matter[None, :], cross], axis=0)
    return cross


def load_emulated_kernels(path: str | Path):
    """Load a saved desilike kernel emulator as a calculator."""
    from desilike.emulators import EmulatedCalculator

    return EmulatedCalculator.load(str(path))


def measured_data_and_covariance(inputs, *, joint: bool):
    """Return the selected measured mean vector and Gaussian covariance."""
    try:
        from scripts.density_split_gaussian_covariance import (
            gaussian_cross_power_covariance,
            gaussian_matter_cross_power_covariance,
        )
    except ModuleNotFoundError:
        from density_split_gaussian_covariance import (
            gaussian_cross_power_covariance,
            gaussian_matter_cross_power_covariance,
        )

    nq = len(QUANTILES)
    cross = np.asarray(inputs.flatdata, dtype="f8").reshape(nq, inputs.k.size)
    matter_noise = float(inputs.volume / inputs.particle_count)
    kwargs = {
        "nmodes": inputs.nmodes,
        "matter_noise": matter_noise,
        "quantile_noise": np.zeros((nq, nq), dtype="f8"),
        "cross_noise": np.zeros(nq, dtype="f8"),
        "nrealizations": inputs.ensemble_nrealizations,
    }
    if joint:
        covariance = gaussian_matter_cross_power_covariance(
            inputs.matter_power,
            cross,
            inputs.quantile_quantile_total_power,
            **kwargs,
        )
        data = np.concatenate([inputs.matter_power, np.ravel(cross)])
    else:
        covariance = gaussian_cross_power_covariance(
            inputs.matter_power,
            cross,
            inputs.quantile_quantile_total_power,
            **kwargs,
        )
        data = np.ravel(cross)
    return data, covariance


def realization_data(inputs, *, joint: bool) -> np.ndarray:
    """Return realization vectors in the same ordering as the mean data."""
    cross = np.asarray(inputs.realizations, dtype="f8")
    if not joint:
        return cross
    matter = np.asarray(inputs.matter_realizations, dtype="f8")
    return np.concatenate([matter, cross], axis=1)


def _cosmology_theory_class():
    from desilike.base import BaseCalculator
    from desilike.jax import numpy as jnp
    from desilike.parameter import Parameter

    class CosmologyTheory(BaseCalculator):
        def initialize(
            self,
            kernels=None,
            k=None,
            joint=False,
            alpha0=FIDUCIAL_ALPHA0,
            smoothing_radius=10.0,
        ):
            self.kernels_calculator = kernels
            self.k = np.asarray(k, dtype="f8")
            self.joint = bool(joint)
            self.fixed_alpha0 = float(alpha0)
            self.smoothing_radius = float(smoothing_radius)
            for quantile in QUANTILES:
                self.params.set(
                    Parameter(
                        f"c1q{quantile}",
                        value=FIDUCIAL_NUISANCE[f"c1q{quantile}"],
                        prior={"limits": [-20.0, 20.0]},
                        ref={
                            "dist": "norm",
                            "loc": FIDUCIAL_NUISANCE[f"c1q{quantile}"],
                            "scale": 0.1,
                        },
                    )
                )
                self.params.set(
                    Parameter(
                        f"c2q{quantile}",
                        value=FIDUCIAL_NUISANCE[f"c2q{quantile}"],
                        prior={"limits": [-50.0, 50.0]},
                        ref={
                            "dist": "norm",
                            "loc": FIDUCIAL_NUISANCE[f"c2q{quantile}"],
                            "scale": 0.5,
                        },
                    )
                )
            self.params.set(
                Parameter(
                    "alpha0",
                    value=self.fixed_alpha0,
                    prior={"limits": [-20.0, 20.0]},
                    ref={"dist": "norm", "loc": self.fixed_alpha0, "scale": 1.0},
                    fixed=not self.joint,
                )
            )

        def calculate(self, **params):
            alpha0 = params.get("alpha0", self.fixed_alpha0)
            c1 = [params[f"c1q{quantile}"] for quantile in QUANTILES]
            c2 = [params[f"c2q{quantile}"] for quantile in QUANTILES]
            self.power = reconstruct_power(
                self.kernels_calculator.kernels,
                self.k,
                c1=c1,
                c2=c2,
                alpha0=alpha0,
                smoothing_radius=self.smoothing_radius,
                joint=self.joint,
            )
            self.flatpower = jnp.ravel(self.power)

        def get(self):
            return self.flatpower

    CosmologyTheory.__name__ = "DensitySplitCosmologyTheory"
    CosmologyTheory.__qualname__ = "DensitySplitCosmologyTheory"
    CosmologyTheory.__module__ = __name__
    return CosmologyTheory


class DensitySplitCosmologyTheory:
    """Lazy compatibility factory for the legacy reconstructed theory."""

    def __new__(cls, *args, **kwargs):
        return _cosmology_theory_class()(*args, **kwargs)
