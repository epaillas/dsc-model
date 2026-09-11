"""Tests for Quijote realization discovery, loading, and averaging."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import h5py
import lsstypes
import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "measure_quijote_acm.py"
SPEC = importlib.util.spec_from_file_location("measure_quijote_acm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_part(
    path: Path,
    coordinates: np.ndarray,
    *,
    part_count: int,
    total: int,
    num_files: int = 2,
    boxsize: float = 1000.0,
    velocities: np.ndarray | None = None,
) -> None:
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        this_file = np.zeros(6, dtype=np.uint32)
        this_file[1] = part_count
        all_particles = np.zeros(6, dtype=np.uint32)
        all_particles[1] = total
        header.attrs["NumFilesPerSnapshot"] = num_files
        header.attrs["NumPart_ThisFile"] = this_file
        header.attrs["NumPart_Total"] = all_particles
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        header.attrs["BoxSize"] = boxsize
        header.attrs["Redshift"] = 0.0
        header.attrs["HubbleParam"] = 0.6711
        header.attrs["Omega0"] = 0.3175
        header.attrs["OmegaLambda"] = 0.6825
        masses = np.zeros(6)
        masses[1] = 65.656
        header.attrs["MassTable"] = masses
        group = handle.create_group("PartType1")
        group.create_dataset("Coordinates", data=coordinates.astype("f4"))
        if velocities is None:
            velocities = np.zeros_like(coordinates)
        group.create_dataset("Velocities", data=velocities.astype("f4"))


def write_snapshot(snapshot_dir: Path, second_boxsize: float = 1000.0) -> None:
    snapshot_dir.mkdir(parents=True)
    write_part(
        snapshot_dir / "snap_004.1.hdf5",
        np.array([[100.0, 600.0, 900.0]]),
        part_count=1,
        total=3,
        boxsize=second_boxsize,
        velocities=np.array([[0.0, 0.0, 10.0]]),
    )
    write_part(
        snapshot_dir / "snap_004.0.hdf5",
        np.array([[0.0, 250.0, 500.0], [750.0, 999.0, 1000.0]]),
        part_count=2,
        total=3,
    )


class FakeObservable:
    written = []

    def __init__(self, value, attrs=None):
        self._value = np.asarray(value)
        self.attrs = attrs or {}

    def value(self):
        return self._value

    def clone(self, value, attrs):
        return FakeObservable(value, attrs)

    def write(self, filename):
        self.written.append(self)
        Path(filename).write_bytes(b"fake hdf5")


def make_spectrum(value: float, shotnoise: float = 0.0):
    pole = lsstypes.Mesh2SpectrumPole(
        k=np.asarray([0.1]),
        k_edges=np.asarray([[0.05, 0.15]]),
        num_raw=np.asarray([value + shotnoise]),
        num_shotnoise=np.asarray([shotnoise]),
        norm=np.asarray([1.0]),
        nmodes=np.asarray([1]),
        ell=0,
    )
    return lsstypes.Mesh2SpectrumPoles([pole])


def make_pair_tree(nquantiles: int, pairs=None):
    if pairs is None:
        pairs = [
            (quantile1, quantile2)
            for quantile1 in range(nquantiles)
            for quantile2 in range(quantile1, nquantiles)
        ]
    counts = np.full(nquantiles, 10, dtype=np.int64)
    return lsstypes.ObservableTree(
        [
            make_spectrum(
                10 * quantile1 + quantile2,
                shotnoise=2.0 if quantile1 == quantile2 else 0.0,
            )
            for quantile1, quantile2 in pairs
        ],
        quantiles1=[pair[0] for pair in pairs],
        quantiles2=[pair[1] for pair in pairs],
        attrs={
            "nquantiles": nquantiles,
            "query_method": "lattice",
            "query_count": int(counts.sum()),
            "quantile_counts": counts,
            "quantile_fractions": counts / counts.sum(),
            "boxsize": np.full(3, 1000.0),
            "meshsize": np.full(3, 16),
            "pair_convention": "upper_triangle_including_diagonal",
            "value_convention": (
                "diagonal Poisson-subtracted; off-diagonal raw cross-spectrum"
            ),
            "total_convention": "value + shotnoise",
        },
    )


class MeasureQuijoteAcmTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plugin = mock.patch.dict(
            sys.modules, {"hdf5plugin": types.ModuleType("hdf5plugin")}
        )
        self.plugin.start()

    def tearDown(self) -> None:
        self.plugin.stop()

    def test_numeric_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("100", "1", "10", "notes", "0"):
                (root / name).mkdir()
            found = MODULE.discover_realizations(root)
            self.assertEqual([path.name for path in found], ["0", "1", "10", "100"])
            selected = MODULE.discover_realizations(root, {1, 100})
            self.assertEqual([path.name for path in selected], ["1", "100"])

    def test_inspect_load_rsd_and_leave_source_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_dir = Path(temporary) / "0" / "snapdir_004"
            write_snapshot(snapshot_dir)
            before = {
                path: (path.stat().st_size, path.stat().st_mtime_ns)
                for path in snapshot_dir.iterdir()
            }
            files, header = MODULE.inspect_snapshot(snapshot_dir)
            positions = MODULE.read_snapshot(files, header, los="z", rsd=True)
            after = {
                path: (path.stat().st_size, path.stat().st_mtime_ns)
                for path in snapshot_dir.iterdir()
            }

            self.assertEqual([path.name for path in files], [
                "snap_004.0.hdf5",
                "snap_004.1.hdf5",
            ])
            self.assertEqual(header["total_particles"], 3)
            self.assertEqual(before, after)
            np.testing.assert_allclose(
                positions,
                np.array(
                    [
                        [0.0, 0.25, -0.5],
                        [-0.25, -0.001, 0.0],
                        [0.1, -0.4, 0.0],
                    ],
                    dtype="f4",
                ),
                atol=2e-7,
            )

    def test_gadget_velocity_conversion_at_nonzero_redshift(self) -> None:
        from scripts.measure_clustering import read_snapshot as clustering_reader

        coordinates = np.array([[499800., 499800., 499800.],
                                [500200., 500200., 500200.]], dtype='f4')
        velocities = np.array([[100., 200., 300.], [-100., -200., -300.]], dtype='f4')
        hubble = 100 * np.sqrt(.3175 * 1.5**3 + .6825)
        header = dict(total_particles=2, redshift=.5,
                      hubble_z_km_s_mpc_h=hubble, boxsize_mpc_h=1000.)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'snapshot.hdf5'
            write_part(path, coordinates, part_count=2, total=2,
                       num_files=1, boxsize=1e6, velocities=velocities)
            before = path.read_bytes()
            for reader in (MODULE.read_snapshot, clustering_reader):
                for axis, los in enumerate('xyz'):
                    for rsd in (False, True):
                        with self.subTest(reader=reader.__module__, los=los, rsd=rsd):
                            expected = coordinates.astype('f8') / 1000.
                            if rsd:
                                peculiar = velocities.astype('f8') * np.sqrt(2./3.)
                                expected[:, axis] += peculiar[:, axis] / ((2./3.) * hubble)
                            expected = (expected + 500.) % 1000. - 500.
                            actual = reader([path], header, los=los, rsd=rsd)
                            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
            self.assertEqual(path.read_bytes(), before)

    def test_rejects_incomplete_unreadable_and_inconsistent_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete = root / "incomplete"
            incomplete.mkdir()
            write_part(
                incomplete / "snap_004.0.hdf5",
                np.zeros((1, 3)),
                part_count=1,
                total=2,
            )
            with self.assertRaisesRegex(ValueError, "incomplete parts"):
                MODULE.inspect_snapshot(incomplete)

            unreadable = root / "unreadable"
            unreadable.mkdir()
            (unreadable / "snap_004.0.hdf5").write_bytes(b"not hdf5")
            with self.assertRaisesRegex(ValueError, "cannot read"):
                MODULE.inspect_snapshot(unreadable)

            inconsistent = root / "inconsistent"
            write_snapshot(inconsistent, second_boxsize=2000.0)
            with self.assertRaisesRegex(ValueError, "inconsistent header"):
                MODULE.inspect_snapshot(inconsistent)

    def test_writes_numeric_ensemble_mean_and_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            successful = {}
            values = {1: 1.0, 10: 3.0}
            lookup = {}
            for realization, value in values.items():
                realization_dir = root / str(realization)
                realization_dir.mkdir()
                paths = {}
                for key in MODULE.SPECTRUM_KEYS:
                    path = realization_dir / f"{key}.h5"
                    path.touch()
                    paths[key] = path
                    lookup[path] = FakeObservable([value, value + 2])
                metadata = realization_dir / "metadata.json"
                metadata.write_text(json.dumps({"redshift": 0.0}))
                paths["metadata"] = metadata
                successful[realization] = paths

            fake_lsstypes = types.ModuleType("lsstypes")
            fake_lsstypes.read = lambda path: lookup[Path(path)]
            FakeObservable.written.clear()
            with mock.patch.dict(sys.modules, {"lsstypes": fake_lsstypes}):
                outputs = MODULE.write_ensemble(
                    successful,
                    {100: "incomplete parts"},
                    root / "ensemble",
                    model="fiducial",
                    snapshot=4,
                    settings={"rsd": True},
                )

            self.assertEqual(len(FakeObservable.written), 4)
            for observable in FakeObservable.written:
                np.testing.assert_allclose(observable.value(), [2.0, 4.0])
                self.assertEqual(observable.attrs["nrealizations"], 2)
            metadata = json.loads(outputs["metadata"].read_text())
            self.assertEqual(metadata["included_realizations"], [1, 10])
            self.assertEqual(
                metadata["skipped_realizations"], {"100": "incomplete parts"}
            )

    def test_pair_product_fresh_backfill_and_auto_recovery(self) -> None:
        class FakeMatterPower:
            compute_calls = 0

            def __init__(self, **kwargs):
                pass

            def compute_spectrum(self, *, save_fn, **kwargs):
                self.__class__.compute_calls += 1
                Path(save_fn).write_bytes(b"matter")

        class FakeDensitySplit:
            pair_calls = []
            quantile_matter_calls = 0

            def __init__(self, **kwargs):
                pass

            def set_density_contrast(self, **kwargs):
                pass

            def set_quantiles(self, **kwargs):
                self.nquantiles = kwargs["nquantiles"]

            def quantile_data_power(self, positions, *, save_fn, **kwargs):
                self.__class__.quantile_matter_calls += 1
                Path(save_fn).write_bytes(b"quantile matter")

            def quantile_pair_power(self, *, pairs=None, save_fn=None, **kwargs):
                canonical = None if pairs is None else list(pairs)
                self.__class__.pair_calls.append(canonical)
                tree = make_pair_tree(self.nquantiles, pairs=canonical)
                if save_fn is not None:
                    MODULE._write_observable(tree, save_fn)
                return tree

        density_module = types.ModuleType(
            "acm.estimators.galaxy_clustering.density_split"
        )
        density_module.DensitySplit = FakeDensitySplit
        spectrum_module = types.ModuleType(
            "acm.estimators.galaxy_clustering.spectrum"
        )
        spectrum_module.PowerSpectrumMultipoles = FakeMatterPower
        header = {
            "redshift": 0.0,
            "source_files": [],
            "boxsize_mpc_h": 1000.0,
            "omega_m": 0.3175,
            "omega_lambda": 0.6825,
            "hubble_param": 0.6711,
            "total_particles": 10,
            "particle_mass_msun_h": 1.0,
            "hubble_z_km_s_mpc_h": 100.0,
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "output"
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        density_module.__name__: density_module,
                        spectrum_module.__name__: spectrum_module,
                    },
                ),
                mock.patch.object(
                    MODULE, "inspect_snapshot", return_value=([], header)
                ),
                mock.patch.object(
                    MODULE,
                    "read_snapshot",
                    return_value=np.zeros((10, 3), dtype="f4"),
                ) as read_snapshot,
            ):
                kwargs = {
                    "model": "fiducial",
                    "realization": 0,
                    "snapshot": 4,
                    "meshsize": 8,
                    "cellsize": 3.9,
                    "smoothing_radius": 10.0,
                    "nquantiles": 5,
                    "los": "z",
                    "ells": (0, 2, 4),
                    "k_step": 0.01,
                    "rsd": False,
                    "overwrite": False,
                }
                paths = MODULE.measure_realization(
                    root / "snapshot", output_dir, **kwargs
                )
                self.assertTrue(
                    all(paths[key].exists() for key in MODULE.SPECTRUM_KEYS)
                )
                self.assertIsNone(FakeDensitySplit.pair_calls[0])
                self.assertEqual(FakeMatterPower.compute_calls, 1)
                self.assertEqual(FakeDensitySplit.quantile_matter_calls, 1)

                paths["quantile_pair"].unlink()
                MODULE.measure_realization(
                    root / "snapshot", output_dir, **kwargs
                )
                backfill_pairs = FakeDensitySplit.pair_calls[1]
                self.assertEqual(len(backfill_pairs), 10)
                self.assertTrue(
                    all(first < second for first, second in backfill_pairs)
                )
                self.assertEqual(FakeMatterPower.compute_calls, 1)
                self.assertEqual(FakeDensitySplit.quantile_matter_calls, 1)
                pair_tree = lsstypes.read(paths["quantile_pair"])
                self.assertEqual(len(pair_tree.labels()), 15)

                reads_before_recovery = read_snapshot.call_count
                paths["quantile_auto"].unlink()
                MODULE.measure_realization(
                    root / "snapshot", output_dir, **kwargs
                )
                self.assertEqual(
                    read_snapshot.call_count, reads_before_recovery
                )
                auto_tree = lsstypes.read(paths["quantile_auto"])
                self.assertEqual(
                    auto_tree.labels(),
                    [{"quantiles": quantile} for quantile in range(5)],
                )

    def test_pair_ensemble_preserves_all_labels_and_averages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            successful = {}
            for realization, offset in ((1, 0.0), (2, 2.0)):
                realization_dir = root / str(realization)
                realization_dir.mkdir()
                paths = {}
                for key in (
                    "matter_power",
                    "quantile_matter",
                    "quantile_auto",
                ):
                    path = realization_dir / f"{key}.h5"
                    make_spectrum(offset).write(path)
                    paths[key] = path
                pair_path = realization_dir / "quantile_pair.h5"
                pair_tree = make_pair_tree(5)
                pair_tree.clone(
                    value=pair_tree.value() + offset
                ).write(pair_path)
                paths["quantile_pair"] = pair_path
                metadata = realization_dir / "metadata.json"
                metadata.write_text(json.dumps({"redshift": 0.0}))
                paths["metadata"] = metadata
                successful[realization] = paths

            outputs = MODULE.write_ensemble(
                successful,
                {},
                root / "ensemble",
                model="fiducial",
                snapshot=4,
                settings={"rsd": False},
            )
            pair_mean = lsstypes.read(outputs["quantile_pair"])
            self.assertEqual(len(pair_mean.labels()), 15)
            np.testing.assert_allclose(
                pair_mean.get(quantiles1=0, quantiles2=1).value(),
                [2.0],
            )
            metadata = json.loads(outputs["metadata"].read_text())
            self.assertIn("quantile_pair", metadata["ensemble_outputs"])


if __name__ == "__main__":
    unittest.main()
