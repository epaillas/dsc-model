"""Tests for the consolidated RSD density-split covariance workflow."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.density_split_gaussian_covariance import (
    gaussian_cross_power_multipole_covariance,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_density_split_covariance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_density_split_covariance_rsd_test", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_leaf(
    handle: h5py.File,
    path: str,
    value: np.ndarray,
    *,
    k: np.ndarray,
    k_edges: np.ndarray,
    nmodes: np.ndarray,
    noise: float = 0.0,
) -> None:
    group = handle.require_group(path)
    group.attrs["los"] = np.array([0.0, 0.0, 1.0])
    group.create_dataset("value", data=np.asarray(value, dtype="f8"))
    group.create_dataset("k", data=k)
    group.create_dataset("k_edges", data=k_edges)
    group.create_dataset("nmodes", data=nmodes)
    group.create_dataset("num_shotnoise", data=np.full(k.size, noise, dtype="f8"))
    group.create_dataset("norm", data=np.ones(k.size, dtype="f8"))


def _write_mock_ensemble(root: Path, nrealizations: int = 12) -> None:
    rng = np.random.default_rng(42)
    k = np.array([0.01, 0.03])
    k_edges = np.array([[0.0, 0.02], [0.02, 0.04]])
    modes = np.fft.fftfreq(16) * 16 * 2.0 * np.pi / 1000.0
    magnitude = np.sqrt(
        modes[:, None, None] ** 2
        + modes[None, :, None] ** 2
        + modes[None, None, :] ** 2
    )
    nmodes = np.asarray(
        [
            np.count_nonzero((magnitude >= lower) & (magnitude < upper))
            for lower, upper in k_edges
        ],
        dtype="f8",
    )
    quantile_mean = np.array(
        [[-4.0, -3.0], [-2.0, -1.5], [0.0, 0.0], [2.0, 1.5], [4.0, 3.0]]
    )
    variations = rng.normal(size=(nrealizations, 5, 2))
    for realization in range(nrealizations):
        directory = root / f"realization_{realization:05d}"
        directory.mkdir(parents=True)
        with h5py.File(directory / "pmm.h5", "w") as handle:
            for ell in (0, 2):
                value = (
                    np.array([50.0, 30.0]) + 0.1 * variations[realization, 0]
                    if ell == 0
                    else 0.02 * variations[realization, 0]
                )
                _write_leaf(
                    handle,
                    str(ell),
                    value,
                    k=k,
                    k_edges=k_edges,
                    nmodes=nmodes,
                    noise=1.0 if ell == 0 else 0.0,
                )
        with h5py.File(directory / "pqm.h5", "w") as handle:
            for quantile in range(5):
                for ell in (0, 2):
                    value = (
                        quantile_mean[quantile] + variations[realization, quantile]
                        if ell == 0
                        else 0.02 * variations[realization, quantile]
                    )
                    _write_leaf(
                        handle,
                        f"{quantile}/{ell}",
                        value,
                        k=k,
                        k_edges=k_edges,
                        nmodes=nmodes,
                    )
        with h5py.File(directory / "pqq.h5", "w") as handle:
            handle.attrs["nquantiles"] = 5
            handle.attrs["pair_convention"] = "upper_triangle_including_diagonal"
            handle.attrs["total_convention"] = "value + shotnoise"
            handle.attrs["value_convention"] = (
                "diagonal Poisson-subtracted; off-diagonal raw cross-spectrum"
            )
            for quantile1 in range(5):
                for quantile2 in range(quantile1, 5):
                    monopole = (
                        np.array([10.0, 8.0])
                        if quantile1 == quantile2
                        else np.array([0.2, 0.1])
                    )
                    variation = 0.05 * (
                        variations[realization, quantile1]
                        + variations[realization, quantile2]
                    )
                    for ell in (0, 2):
                        _write_leaf(
                            handle,
                            f"{quantile1}-{quantile2}/{ell}",
                            monopole + variation if ell == 0 else 0.02 * variation,
                            k=k,
                            k_edges=k_edges,
                            nmodes=nmodes,
                            noise=(2.0 if quantile1 == quantile2 and ell == 0 else 0.0),
                        )


class RSDCovarianceValidationTest(unittest.TestCase):
    def test_small_ensemble_comparison_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "measurements"
            output = Path(temporary) / "outputs"
            _write_mock_ensemble(root)
            summary, outputs = MODULE.run_validation(
                root,
                output_root=output,
                data_vector=("pmm", "pqm"),
                ells=(0,),
                kmin=0.0,
                kmax=0.04,
                boxsize=1000.0,
                meshsize=16,
                correlation_ngrid=1001,
            )
            self.assertEqual(summary["configuration"]["realizations"], 12)
            self.assertEqual(summary["configuration"]["data_vector"], ["pmm", "pqm"])
            self.assertEqual(summary["configuration"]["vector_size"], 10)
            self.assertEqual(summary["hybrid"]["matrix_rank"], 10)
            self.assertEqual(summary["pt"]["matrix_rank"], 10)
            assert outputs is not None
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            with np.load(outputs["covariances"]) as products:
                self.assertEqual(products["labels"].shape, (10,))
                self.assertEqual(products["hybrid_single_box"].shape, (10, 10))
                self.assertEqual(products["pt_single_box"].shape, (10, 10))

    def test_archive_round_trip_and_flexible_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "measurements"
            output = Path(temporary) / "outputs"
            _write_mock_ensemble(root)
            _, outputs = MODULE.run_validation(
                root,
                output_root=output,
                data_vector=("pmm", "pqm", "pqq"),
                ells=(0, 2),
                kmin=0.0,
                kmax=0.04,
                boxsize=1000.0,
                meshsize=16,
                correlation_ngrid=1001,
            )
            archive = MODULE.load_covariance_archive(outputs["covariances"])
            self.assertEqual(MODULE.ARCHIVE_SCHEMA_VERSION, 2)
            self.assertEqual(archive.data_vector, ("pmm", "pqm", "pqq"))
            self.assertEqual(archive.nrealizations, 12)
            self.assertEqual(archive.empirical.shape, (60, 60))
            self.assertEqual(archive.labels.shape, (60,))
            self.assertEqual(set(archive.statistics), {"pmm", "pqm", "pqq"})
            self.assertEqual(tuple(archive.models), ("hybrid", "pt"))
            self.assertIn("all measured", archive.model_provenance["hybrid"])
            self.assertIn("fully analytic", archive.model_provenance["pt"])

            selections = {
                "pmm": (dict(statistics=("pmm",), quantiles=(1,)), 4),
                "pqm": (dict(statistics=("pqm",), quantiles=(1, 5)), 8),
                "pqq": (dict(statistics=("pqq",), quantiles=(1, 5)), 12),
                "mixed": (
                    dict(
                        statistics=("pmm", "pqm"),
                        quantiles=(1, 5),
                        ells=(2,),
                        k_range=(0.02, 0.04),
                    ),
                    3,
                ),
                "pair": (
                    dict(
                        statistics=("pqq",),
                        pair_labels=("Pq1q5",),
                        ells=(0,),
                        models=("pt",),
                    ),
                    2,
                ),
            }
            for name, (kwargs, size) in selections.items():
                with self.subTest(selection=name):
                    selected = archive.select(**kwargs)
                    self.assertEqual(selected.size, size)
                    self.assertEqual(selected.empirical.shape, (size, size))
                    self.assertTrue(
                        all(
                            matrix.shape == (size, size)
                            for matrix in selected.models.values()
                        )
                    )
                    self.assertEqual(selected.block_labels.shape, (size,))

            pqm = archive.select(statistics=("pqm",), quantiles=(1, 5), ells=(0, 2))
            changes = np.r_[
                0,
                np.flatnonzero(pqm.block_labels[1:] != pqm.block_labels[:-1]) + 1,
                pqm.size,
            ]
            np.testing.assert_array_equal(changes, [0, 2, 4, 6, 8])

            with self.assertRaisesRegex(ValueError, "statistics.*missing"):
                archive.select(statistics=("not-a-statistic",))
            with self.assertRaisesRegex(ValueError, "quantiles.*missing"):
                archive.select(quantiles=(3,))
            with self.assertRaisesRegex(ValueError, "ells.*missing"):
                archive.select(ells=(4,))
            with self.assertRaisesRegex(ValueError, "pair_labels.*missing"):
                archive.select(pair_labels=("Pq2q5-missing",))
            with self.assertRaisesRegex(ValueError, "no elements"):
                archive.select(k_range=(0.5, 0.6))

            legacy = Path(temporary) / "legacy.npz"
            np.savez(legacy, empirical_single_box=np.eye(2))
            with self.assertRaisesRegex(
                ValueError, "not a supported covariance archive"
            ):
                MODULE.load_covariance_archive(legacy)

    def test_hybrid_supports_every_pair_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_mock_ensemble(root)
            ensemble = MODULE.load_rsd_ensemble(root, ells=(0,), kmin=0.0, kmax=0.04)
            common = dict(
                quantiles=ensemble.spectra.quantiles,
                ells=ensemble.spectra.ells,
                k_edges=ensemble.k_edges,
                nmodes=ensemble.nmodes,
                boxsize=1000.0,
                meshsize=16,
                los=ensemble.los,
            )
            expected_pairs = {
                "pmm": 1,
                "pqm": 4,
                "pqq": 10,
                ("pmm", "pqm"): 5,
                ("pmm", "pqm", "pqq"): 15,
            }
            for data_vector, npairs in expected_pairs.items():
                with self.subTest(data_vector=data_vector):
                    result = MODULE.build_hybrid_covariance(
                        ensemble.spectra.pmm,
                        ensemble.spectra.pqm,
                        ensemble.spectra.pqq,
                        data_vector=data_vector,
                        **common,
                    )
                    size = npairs * len(ensemble.k)
                    self.assertEqual(result.covariance.shape, (size, size))
                    self.assertEqual(len(result.labels), size)

            reordered = MODULE.build_hybrid_covariance(
                ensemble.spectra.pmm,
                ensemble.spectra.pqm,
                ensemble.spectra.pqq,
                data_vector=("pqq", "pmm"),
                **common,
            )
            self.assertEqual(reordered.pairs[0], (1, 1))
            self.assertEqual(reordered.pairs[-1], (0, 0))

            generic = MODULE.build_hybrid_covariance(
                ensemble.spectra.pmm,
                ensemble.spectra.pqm,
                ensemble.spectra.pqq,
                data_vector="pqm",
                **common,
            )
            specialized = gaussian_cross_power_multipole_covariance(
                ensemble.spectra.pmm,
                ensemble.spectra.pqm,
                ensemble.spectra.pqq,
                ells=ensemble.spectra.ells,
                k_edges=ensemble.k_edges,
                nmodes=ensemble.nmodes,
                boxsize=1000.0,
                meshsize=16,
                los=ensemble.los,
            )
            np.testing.assert_allclose(generic.covariance, specialized)

    def test_cumulative_data_vector_cli(self) -> None:
        parser = MODULE.build_parser()
        self.assertEqual(parser.parse_args([]).data_vector, ["pmm", "pqm"])
        self.assertEqual(
            parser.parse_args(["--data-vector", "pmm", "pqm", "pqq"]).data_vector,
            ["pmm", "pqm", "pqq"],
        )
        self.assertEqual(
            parser.parse_args(["--data-vector", "pqq"]).data_vector, ["pqq"]
        )
        for obsolete in ("all", "joint"):
            with self.subTest(obsolete=obsolete):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(["--data-vector", obsolete])
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.observable_pairs(("pmm", "pmm"), nq=4)

    def test_skips_realizations_missing_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_mock_ensemble(root)
            incomplete = root / "realization_00003"
            (incomplete / "pqq.h5").unlink()
            with self.assertWarnsRegex(RuntimeWarning, "skipping 1 realizations"):
                ensemble = MODULE.load_rsd_ensemble(
                    root,
                    data_vector=("pmm", "pqm"),
                    ells=(0,),
                    kmin=0.0,
                    kmax=0.04,
                )
            self.assertEqual(len(ensemble.realization_ids), 11)
            self.assertNotIn(3, ensemble.realization_ids)

    def test_file_requirements_follow_covariance_contractions(self) -> None:
        self.assertEqual(MODULE.required_measurement_statistics(("pmm",), 4), ("pmm",))
        self.assertEqual(MODULE.required_measurement_statistics(("pqq",), 4), ("pqq",))
        self.assertEqual(
            MODULE.required_measurement_statistics(("pqm",), 4),
            ("pmm", "pqm", "pqq"),
        )
        self.assertEqual(
            MODULE.required_measurement_statistics(("pmm", "pqq"), 4),
            ("pmm", "pqm", "pqq"),
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_mock_ensemble(root)
            for directory in root.glob("realization_*"):
                (directory / "pqm.h5").unlink()
                (directory / "pqq.h5").unlink()
            ensemble = MODULE.load_rsd_ensemble(
                root,
                data_vector=("pmm",),
                ells=(0,),
                kmin=0.0,
                kmax=0.04,
            )
            self.assertEqual(len(ensemble.realization_ids), 12)

    def test_rejects_dependent_five_quantile_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "linearly dependent"):
            MODULE.load_rsd_ensemble(Path("unused"), quantiles=(1, 2, 3, 4, 5))

    def test_rejects_inconsistent_edges_and_mode_counts(self) -> None:
        for dataset, replacement, message in (
            ("k_edges", np.array([[0.0, 0.021], [0.02, 0.04]]), "k-bin edges"),
            ("nmodes", np.array([148.0, 898.0]), "mode counts"),
        ):
            with (
                self.subTest(dataset=dataset),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                _write_mock_ensemble(root)
                path = root / "realization_00001" / "pmm.h5"
                with h5py.File(path, "r+") as handle:
                    handle[f"0/{dataset}"][:] = replacement
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.load_rsd_ensemble(
                        root,
                        ells=(0,),
                        kmin=0.0,
                        kmax=0.04,
                    )

    def test_rejects_missing_pair_branch_and_nonfinite_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_mock_ensemble(root)
            path = root / "realization_00003" / "pqq.h5"
            with h5py.File(path, "r+") as handle:
                del handle["0-4"]
            with self.assertRaisesRegex(ValueError, "missing branch"):
                MODULE.load_rsd_ensemble(
                    root,
                    ells=(0,),
                    kmin=0.0,
                    kmax=0.04,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_mock_ensemble(root)
            path = root / "realization_00003" / "pqm.h5"
            with h5py.File(path, "r+") as handle:
                handle["0/0/value"][0] = np.nan
            with self.assertRaisesRegex(ValueError, "non-finite"):
                MODULE.load_rsd_ensemble(
                    root,
                    ells=(0,),
                    kmin=0.0,
                    kmax=0.04,
                )

    def test_rejects_inconsistent_fixed_los(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_mock_ensemble(root)
            path = root / "realization_00001" / "pqm.h5"
            with h5py.File(path, "r+") as handle:
                handle["0/0"].attrs["los"] = np.array([1.0, 0.0, 0.0])
            with self.assertRaisesRegex(ValueError, "fixed LOS"):
                MODULE.load_rsd_ensemble(
                    root,
                    ells=(0,),
                    kmin=0.0,
                    kmax=0.04,
                )


if __name__ == "__main__":
    unittest.main()
