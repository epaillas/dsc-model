"""Measure and average ACM statistics across Quijote realizations."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_DIR / "data"
SPECTRUM_KEYS = (
    "matter_power",
    "quantile_matter",
    "quantile_auto",
    "quantile_pair",
)


def _write_observable(observable, path: Path) -> None:
    """Atomically write an lsstypes observable."""
    path = Path(path)
    temporary = path.with_suffix(".tmp.h5")
    observable.write(temporary)
    temporary.replace(path)


def quantile_auto_from_pairs(pair_power):
    """Return the legacy one-label auto tree from a full quantile-pair tree."""
    import lsstypes

    nquantiles = int(pair_power.attrs["nquantiles"])
    attrs = {
        key: pair_power.attrs[key]
        for key in ("nquantiles", "query_method", "boxsize", "meshsize")
        if key in pair_power.attrs
    }
    return lsstypes.ObservableTree(
        [
            pair_power.get(quantiles1=quantile, quantiles2=quantile)
            for quantile in range(nquantiles)
        ],
        quantiles=list(range(nquantiles)),
        attrs=attrs,
    )


def quantile_pairs_from_auto_and_cross(auto_power, cross_power):
    """Combine legacy autos and off-diagonal crosses into a full pair tree."""
    import lsstypes

    nquantiles = int(auto_power.attrs["nquantiles"])
    pairs = [
        (quantile1, quantile2)
        for quantile1 in range(nquantiles)
        for quantile2 in range(quantile1, nquantiles)
    ]
    branches = [
        (
            auto_power.get(quantiles=quantile1)
            if quantile1 == quantile2
            else cross_power.get(
                quantiles1=quantile1, quantiles2=quantile2
            )
        )
        for quantile1, quantile2 in pairs
    ]
    attrs = dict(cross_power.attrs)
    return lsstypes.ObservableTree(
        branches,
        quantiles1=[pair[0] for pair in pairs],
        quantiles2=[pair[1] for pair in pairs],
        attrs=attrs,
    )


def discover_realizations(
    model_dir: Path, requested: set[int] | None = None
) -> list[Path]:
    """Return numeric realization directories in numeric order."""
    return sorted(
        (
            path
            for path in Path(model_dir).iterdir()
            if path.is_dir()
            and path.name.isdigit()
            and (requested is None or int(path.name) in requested)
        ),
        key=lambda path: int(path.name),
    )


def inspect_snapshot(snapshot_dir: Path) -> tuple[list[Path], dict]:
    """Validate a multipart Quijote snapshot without loading all particles."""
    try:
        import hdf5plugin  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Install the Quijote compression filter with "
            "`python -m pip install hdf5plugin`."
        ) from exc
    import h5py

    snapshot_dir = Path(snapshot_dir)
    files = sorted(
        snapshot_dir.glob("snap_*.hdf5"),
        key=lambda path: int(path.stem.rsplit(".", 1)[1]),
    )
    if not files:
        raise ValueError("no snapshot files")

    reference = None
    particle_count = 0
    part_indices = []
    for filename in files:
        part_indices.append(int(filename.stem.rsplit(".", 1)[1]))
        try:
            with h5py.File(filename, "r") as handle:
                attrs = handle["Header"].attrs
                low = np.asarray(attrs["NumPart_Total"], dtype=np.uint64)
                high = np.asarray(
                    attrs.get("NumPart_Total_HighWord", np.zeros(6)),
                    dtype=np.uint64,
                )
                current = {
                    "num_files": int(attrs["NumFilesPerSnapshot"]),
                    "total_particles": int(
                        low[1] + (high[1] << np.uint64(32))
                    ),
                    "boxsize_mpc_h": float(attrs["BoxSize"]) / 1e3,
                    "redshift": float(attrs["Redshift"]),
                    "omega_m": float(attrs["Omega0"]),
                    "omega_lambda": float(attrs["OmegaLambda"]),
                    "hubble_param": float(attrs["HubbleParam"]),
                    "particle_mass_msun_h": float(attrs["MassTable"][1]) * 1e10,
                }
                count = int(attrs["NumPart_ThisFile"][1])
                coordinates = handle["PartType1/Coordinates"]
                velocities = handle["PartType1/Velocities"]
                if coordinates.shape != (count, 3):
                    raise ValueError("invalid coordinate shape")
                if velocities.shape != (count, 3):
                    raise ValueError("invalid velocity shape")
                if count:
                    coordinates[:1]
                    velocities[:1]
        except Exception as exc:
            raise ValueError(f"cannot read {filename.name}: {exc}") from exc
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError(f"inconsistent header in {filename.name}")
        particle_count += count

    assert reference is not None
    expected = reference["num_files"]
    if len(files) != expected or part_indices != list(range(expected)):
        raise ValueError(
            f"incomplete parts: found {part_indices}, expected {list(range(expected))}"
        )
    if particle_count != reference["total_particles"]:
        raise ValueError(
            f"particle count is {particle_count}, "
            f"expected {reference['total_particles']}"
        )
    reference["hubble_z_km_s_mpc_h"] = 100 * np.sqrt(
        reference["omega_m"] * (1 + reference["redshift"]) ** 3
        + reference["omega_lambda"]
    )
    reference["source_files"] = [
        {"path": str(path), "size": path.stat().st_size} for path in files
    ]
    return files, reference


def read_snapshot(
    files: list[Path], header: dict, los: str, rsd: bool
) -> np.ndarray:
    """Load CDM positions and optionally apply the plane-parallel RSD shift."""
    import h5py

    positions = np.empty((header["total_particles"], 3), dtype="f4")
    los_axis = {"x": 0, "y": 1, "z": 2}[los]
    scale_factor = 1.0 / (1.0 + header["redshift"])
    rsd_factor = np.float32(
        np.sqrt(scale_factor)
        * (1.0 + header["redshift"])
        / header["hubble_z_km_s_mpc_h"]
    )
    offset = 0
    for filename in files:
        with h5py.File(filename, "r") as handle:
            coordinates = handle["PartType1/Coordinates"]
            end = offset + len(coordinates)
            coordinates.read_direct(positions, dest_sel=np.s_[offset:end])
            positions[offset:end] /= np.float32(1e3)
            if rsd:
                # Gadget stores v_pec / sqrt(a); convert before the RSD shift.
                velocities = handle["PartType1/Velocities"][:, los_axis]
                positions[offset:end, los_axis] += velocities * rsd_factor
        offset = end

    boxsize = np.float32(header["boxsize_mpc_h"])
    positions += boxsize / 2
    np.remainder(positions, boxsize, out=positions)
    positions -= boxsize / 2
    return positions


def measure_realization(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    model: str,
    realization: int,
    snapshot: int,
    meshsize: int | np.ndarray,
    cellsize: float,
    smoothing_radius: float,
    nquantiles: int,
    los: str,
    ells: tuple[int, ...],
    k_step: float,
    rsd: bool,
    overwrite: bool,
) -> dict[str, Path]:
    """Measure one realization, recovering incomplete derived-output caches."""
    from acm.estimators.galaxy_clustering.density_split import DensitySplit
    from acm.estimators.galaxy_clustering.spectrum import PowerSpectrumMultipoles

    files, header = inspect_snapshot(snapshot_dir)
    paths = {
        "matter_power": output_dir / "matter_power.h5",
        "quantile_matter": output_dir / "density_split_quantile_matter.h5",
        "quantile_auto": output_dir / "density_split_quantile_auto.h5",
        "quantile_pair": output_dir / "density_split_quantile_pair.h5",
        "metadata": output_dir / "metadata.json",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_meshsize = (
        max(int(header["boxsize_mpc_h"] // cellsize), 1)
        if cellsize > 0
        else meshsize
    )
    measurement = {
        "rsd": rsd,
        "los": los,
        "meshsize": (
            meshsize.tolist() if isinstance(meshsize, np.ndarray) else meshsize
        ),
        "density_split_meshsize": (
            ds_meshsize.tolist()
            if isinstance(ds_meshsize, np.ndarray)
            else ds_meshsize
        ),
        "cellsize_mpc_h": cellsize,
        "smoothing_radius_mpc_h": smoothing_radius,
        "nquantiles": nquantiles,
        "query_method": "lattice",
        "ells": list(ells),
        "k_step_h_mpc": k_step,
    }
    identity = {
        "model": model,
        "realization": realization,
        "snapshot": snapshot,
        "redshift": header["redshift"],
        "source_files": header["source_files"],
        "measurement": measurement,
    }
    cache_valid = False
    if not overwrite and paths["metadata"].exists():
        cached = json.loads(paths["metadata"].read_text())
        cache_valid = all(
            cached.get(key) == value for key, value in identity.items()
        )
    available = {
        key: cache_valid and paths[key].exists() for key in SPECTRUM_KEYS
    }

    if available["quantile_pair"] and not available["quantile_auto"]:
        import lsstypes

        pair_power = lsstypes.read(paths["quantile_pair"])
        _write_observable(
            quantile_auto_from_pairs(pair_power), paths["quantile_auto"]
        )
        available["quantile_auto"] = True
    if all(available.values()):
        return paths

    positions = read_snapshot(files, header, los=los, rsd=rsd)
    spectrum_args = {"edges": {"step": k_step}, "ells": ells, "los": los}

    if not available["matter_power"]:
        matter_power = PowerSpectrumMultipoles(
            data_positions=positions,
            boxsize=header["boxsize_mpc_h"],
            boxcenter=0.0,
            meshsize=meshsize,
        )
        matter_power.compute_spectrum(
            **spectrum_args, save_fn=paths["matter_power"]
        )

    if not available["quantile_matter"] or not available["quantile_pair"]:
        density_split = DensitySplit(
            data_positions=positions,
            boxsize=header["boxsize_mpc_h"],
            boxcenter=0.0,
            meshsize=ds_meshsize,
        )
        density_split.set_density_contrast(
            smoothing_radius=smoothing_radius
        )
        density_split.set_quantiles(
            nquantiles=nquantiles, query_method="lattice"
        )
        if not available["quantile_matter"]:
            density_split.quantile_data_power(
                positions,
                **spectrum_args,
                save_fn=paths["quantile_matter"],
            )
        if not available["quantile_pair"]:
            if available["quantile_auto"]:
                import lsstypes

                off_diagonal = [
                    (quantile1, quantile2)
                    for quantile1 in range(nquantiles)
                    for quantile2 in range(quantile1 + 1, nquantiles)
                ]
                cross_power = density_split.quantile_pair_power(
                    pairs=off_diagonal, **spectrum_args
                )
                auto_power = lsstypes.read(paths["quantile_auto"])
                pair_power = quantile_pairs_from_auto_and_cross(
                    auto_power, cross_power
                )
                _write_observable(pair_power, paths["quantile_pair"])
            else:
                pair_power = density_split.quantile_pair_power(
                    **spectrum_args, save_fn=paths["quantile_pair"]
                )
            if not available["quantile_auto"]:
                _write_observable(
                    quantile_auto_from_pairs(pair_power),
                    paths["quantile_auto"],
                )

    metadata = {
        **identity,
        "source_directory": str(snapshot_dir),
        "cosmology": {
            key: header[key]
            for key in ("omega_m", "omega_lambda", "hubble_param")
        },
        "boxsize_mpc_h": header["boxsize_mpc_h"],
        "total_particles": header["total_particles"],
        "particle_mass_msun_h": header["particle_mass_msun_h"],
        "hubble_z_km_s_mpc_h": header["hubble_z_km_s_mpc_h"],
        "particle_type": "PartType1 (CDM)",
        "position_unit": "Mpc/h",
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    temporary = paths["metadata"].with_suffix(".tmp.json")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(paths["metadata"])
    return paths


def write_ensemble(
    successful: dict[int, dict[str, Path]],
    skipped: dict[int, str],
    ensemble_dir: Path,
    *,
    model: str,
    snapshot: int,
    settings: dict,
) -> dict[str, Path]:
    """Average compatible realization products and write ensemble metadata."""
    import lsstypes

    ensemble_dir.mkdir(parents=True, exist_ok=True)
    output_names = {
        "matter_power": "matter_power_mean.h5",
        "quantile_matter": "density_split_quantile_matter_mean.h5",
        "quantile_auto": "density_split_quantile_auto_mean.h5",
        "quantile_pair": "density_split_quantile_pair_mean.h5",
    }
    outputs = {
        key: ensemble_dir / filename for key, filename in output_names.items()
    }
    realization_ids = sorted(successful)
    if not realization_ids:
        raise RuntimeError("No complete realization measurements are available")

    for key, output in outputs.items():
        observables = [
            lsstypes.read(successful[realization][key])
            for realization in realization_ids
        ]
        shapes = {observable.value().shape for observable in observables}
        if len(shapes) != 1:
            raise ValueError(f"Incompatible {key} observable shapes: {shapes}")
        mean = np.mean(
            np.stack([observable.value() for observable in observables]), axis=0
        )
        attrs = dict(observables[0].attrs)
        attrs.update(
            {
                "statistic": "ensemble_mean",
                "model": model,
                "snapshot": snapshot,
                "nrealizations": len(realization_ids),
                "realizations": np.asarray(realization_ids, dtype=np.int64),
            }
        )
        averaged = observables[0].clone(value=mean, attrs=attrs)
        temporary = output.with_suffix(".tmp.h5")
        averaged.write(temporary)
        temporary.replace(output)

    metadata_path = ensemble_dir / "metadata.json"
    metadata = {
        "model": model,
        "snapshot": snapshot,
        "redshift": json.loads(
            successful[realization_ids[0]]["metadata"].read_text()
        )["redshift"],
        "nrealizations": len(realization_ids),
        "included_realizations": realization_ids,
        "skipped_realizations": {
            str(realization): reason
            for realization, reason in sorted(skipped.items())
        },
        "settings": settings,
        "realization_outputs": {
            str(realization): {
                key: str(path) for key, path in successful[realization].items()
            }
            for realization in realization_ids
        },
        "ensemble_outputs": {key: str(path) for key, path in outputs.items()},
    }
    temporary = metadata_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)
    outputs["metadata"] = metadata_path
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure and average Quijote realization statistics."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model", default="fiducial")
    parser.add_argument("--snapshot", type=int, default=4)
    parser.add_argument("--realizations", type=int, nargs="+")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--meshsize", type=int, nargs="+", default=[128])
    parser.add_argument("--cellsize", type=float, default=3.9)
    parser.add_argument("--smoothing-radius", type=float, default=10.0)
    parser.add_argument("--nquantiles", type=int, default=5)
    parser.add_argument("--los", choices=("x", "y", "z"), default="z")
    parser.add_argument("--ells", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--k-step", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-rsd", action="store_true")
    args = parser.parse_args(argv)

    if len(args.meshsize) not in (1, 3):
        parser.error("--meshsize expects one or three integers")
    meshsize = (
        args.meshsize[0]
        if len(args.meshsize) == 1
        else np.asarray(args.meshsize, dtype=int)
    )
    ells = tuple(args.ells)
    rsd = not args.no_rsd
    model_dir = args.input_root / args.model
    requested = set(args.realizations) if args.realizations else None
    realization_dirs = discover_realizations(model_dir, requested)
    geometry = f"rsd_{args.los}" if rsd else "real"
    base_dir = (
        args.output_root
        / args.model
        / f"snapshot_{args.snapshot:03d}"
        / geometry
    )
    settings = {
        "rsd": rsd,
        "los": args.los,
        "meshsize": (
            meshsize.tolist() if isinstance(meshsize, np.ndarray) else meshsize
        ),
        "cellsize_mpc_h": args.cellsize,
        "smoothing_radius_mpc_h": args.smoothing_radius,
        "nquantiles": args.nquantiles,
        "query_method": "lattice",
        "ells": list(ells),
        "k_step_h_mpc": args.k_step,
    }

    from jax import config

    from acm.utils.logging import setup_logging

    config.update("jax_enable_x64", True)
    setup_logging()
    logger = logging.getLogger(__name__)
    successful = {}
    skipped = {}
    for realization_dir in realization_dirs:
        realization = int(realization_dir.name)
        snapshot_dir = realization_dir / f"snapdir_{args.snapshot:03d}"
        output_dir = (
            base_dir
            / "realizations"
            / f"realization_{realization:05d}"
        )
        try:
            inspect_snapshot(snapshot_dir)
        except Exception as exc:
            skipped[realization] = str(exc)
            logger.warning("Skipping realization %d: %s", realization, exc)
            continue
        try:
            logger.info("Processing realization %d.", realization)
            successful[realization] = measure_realization(
                snapshot_dir,
                output_dir,
                model=args.model,
                realization=realization,
                snapshot=args.snapshot,
                meshsize=meshsize,
                cellsize=args.cellsize,
                smoothing_radius=args.smoothing_radius,
                nquantiles=args.nquantiles,
                los=args.los,
                ells=ells,
                k_step=args.k_step,
                rsd=rsd,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            skipped[realization] = str(exc)
            logger.exception("Failed realization %d.", realization)
        finally:
            gc.collect()

    ensemble = write_ensemble(
        successful,
        skipped,
        base_dir / "ensemble",
        model=args.model,
        snapshot=args.snapshot,
        settings=settings,
    )
    logger.info(
        "Wrote ensemble of %d realizations to %s.",
        len(successful),
        ensemble["metadata"].parent,
    )


if __name__ == "__main__":
    main()
