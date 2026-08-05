"""
Measure the matter power-spectrum multipoles from periodic-box positions.
"""

import argparse
from pathlib import Path

import numpy as np


def discover_realizations(model_dir: Path) -> list[int]:
    """Return available numeric realization directory names in numeric order."""
    return sorted(
        int(path.name)
        for path in model_dir.glob("*")
        if path.is_dir() and path.name.isdigit()
    )


def read_snapshot(
    files: list[Path], header: dict, los: str, rsd: bool,
) -> np.ndarray:
    """Load CDM positions and optionally apply the plane-parallel RSD shift."""
    import h5py

    positions = np.empty((header["total_particles"], 3), dtype="f4")
    los_axis = {"x": 0, "y": 1, "z": 2}[los]
    rsd_factor = np.float32(
        (1 + header["redshift"]) / header["hubble_z_km_s_mpc_h"]
    )
    offset = 0
    for filename in files:
        with h5py.File(filename, "r") as handle:
            coordinates = handle["PartType1/Coordinates"]
            end = offset + len(coordinates)
            coordinates.read_direct(positions, dest_sel=np.s_[offset:end])
            positions[offset:end] /= np.float32(1e3)
            if rsd:
                velocities = handle["PartType1/Velocities"][:, los_axis]
                positions[offset:end, los_axis] += velocities * rsd_factor
        offset = end

    boxsize = np.float32(header["boxsize_mpc_h"])
    positions += boxsize / 2
    np.remainder(positions, boxsize, out=positions)
    positions -= boxsize / 2
    return positions


def measure_pmm(
    positions: np.ndarray,
    boxsize: float,
    meshsize: int,
    los: str = "z",
    ells: tuple[int, ...] = (0,),
    k_step: float = 0.001,
    save_fn: Path | None = None,
):
    """Measure matter power-spectrum multipoles from periodic-box positions."""
    from acm.estimators.galaxy_clustering.spectrum import PowerSpectrumMultipoles

    estimator = PowerSpectrumMultipoles(
        data_positions=positions,
        boxsize=boxsize,
        boxcenter=0.0,
        meshsize=meshsize,
    )
    return estimator.compute_spectrum(
        edges={"step": k_step},
        ells=ells,
        los=los,
        save_fn=save_fn,
    )


def measure_pqm(
    positions: np.ndarray,
    boxsize: float,
    meshsize: int,
    smoothing_radius: float = 10.0,
    nquantiles: int = 5,
    los: str = "z",
    ells: tuple[int, ...] = (0,),
    k_step: float = 0.001,
    save_fn: Path | None = None,
):
    """Measure density-split–matter cross-power multipoles."""
    from acm.estimators.galaxy_clustering.density_split import DensitySplit

    density_split = DensitySplit(
        data_positions=positions,
        boxsize=boxsize,
        boxcenter=0.0,
        meshsize=meshsize,
    )
    density_split.set_density_contrast(smoothing_radius=smoothing_radius)
    density_split.set_quantiles(
        nquantiles=nquantiles,
        query_method="lattice",
    )
    return density_split.quantile_data_power(
        positions,
        edges={"step": k_step},
        ells=ells,
        los=los,
        save_fn=save_fn,
    )


def measure_pqq(
    positions: np.ndarray,
    boxsize: float,
    meshsize: int,
    smoothing_radius: float = 10.0,
    nquantiles: int = 5,
    los: str = "z",
    ells: tuple[int, ...] = (0,),
    k_step: float = 0.001,
    save_fn: Path | None = None,
):
    """Measure density-split quantile-pair power multipoles."""
    from acm.estimators.galaxy_clustering.density_split import DensitySplit

    density_split = DensitySplit(
        data_positions=positions,
        boxsize=boxsize,
        boxcenter=0.0,
        meshsize=meshsize,
    )
    density_split.set_density_contrast(smoothing_radius=smoothing_radius)
    density_split.set_quantiles(
        nquantiles=nquantiles,
        query_method="lattice",
    )
    return density_split.quantile_pair_power(
        edges={"step": k_step},
        ells=ells,
        los=los,
        save_fn=save_fn,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--realizations",
        type=int,
        nargs="+",
        help=(
            "Realization IDs to process. If omitted, process every numeric "
            "directory under <input-root>/<model>."
        ),
    )
    parser.add_argument(
        "--todo",
        choices=("pmm", "pqm", "pqq"),
        nargs="+",
        default=["pmm", "pqm", "pqq"],
    )
    parser.add_argument("--model", default="fiducial")
    parser.add_argument("--snapshot", type=int, default=4)
    parser.add_argument("--meshsize", type=int, default=256)
    parser.add_argument("--smoothing-radius", type=float, default=10.0)
    parser.add_argument("--nquantiles", type=int, default=5)
    parser.add_argument("--los", choices=("x", "y", "z"), default="z")
    parser.add_argument("--ells", type=int, nargs="+", default=[0])
    parser.add_argument("--k-step", type=float, default=0.01)
    parser.add_argument("--rsd", action="store_true")
    args = parser.parse_args()

    import hdf5plugin  # noqa: F401
    import h5py

    model_dir = args.input_root / args.model
    realizations = args.realizations
    if realizations is None:
        realizations = discover_realizations(model_dir)
        if not realizations:
            raise FileNotFoundError(
                f"no numeric realization directories in {model_dir}"
            )
        print(
            f"Discovered {len(realizations)} realizations in {model_dir}."
        )

    for realization in realizations:
        snapshot_dir = (
            model_dir
            / str(realization)
            / f"snapdir_{args.snapshot:03d}"
        )
        files = sorted(
            snapshot_dir.glob(f"snap_{args.snapshot:03d}.*.hdf5"),
            key=lambda path: int(path.stem.rsplit(".", 1)[1]),
        )
        if not files:
            raise FileNotFoundError(f"no snapshot files in {snapshot_dir}")

        with h5py.File(files[0], "r") as handle:
            attrs = handle["Header"].attrs
            low = np.asarray(attrs["NumPart_Total"], dtype=np.uint64)
            high = np.asarray(
                attrs.get("NumPart_Total_HighWord", np.zeros(6)),
                dtype=np.uint64,
            )
            omega_m = float(attrs["Omega0"])
            omega_lambda = float(attrs["OmegaLambda"])
            redshift = float(attrs["Redshift"])
            header = {
                "total_particles": int(
                    low[1] + (high[1] << np.uint64(32))
                ),
                "boxsize_mpc_h": float(attrs["BoxSize"]) / 1e3,
                "redshift": redshift,
                "hubble_z_km_s_mpc_h": 100
                * np.sqrt(
                    omega_m * (1 + redshift) ** 3 + omega_lambda
                ),
            }

        positions = read_snapshot(files, header, los=args.los, rsd=args.rsd)
        output_dir = args.output_root / f"realization_{realization:05d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        common = {
            "positions": positions,
            "boxsize": header["boxsize_mpc_h"],
            "meshsize": args.meshsize,
            "los": args.los,
            "ells": tuple(args.ells),
            "k_step": args.k_step,
        }
        if "pmm" in args.todo:
            measure_pmm(**common, save_fn=output_dir / "pmm.h5")
        density_split = {
            **common,
            "smoothing_radius": args.smoothing_radius,
            "nquantiles": args.nquantiles,
        }
        if "pqm" in args.todo:
            measure_pqm(**density_split, save_fn=output_dir / "pqm.h5")
        if "pqq" in args.todo:
            measure_pqq(**density_split, save_fn=output_dir / "pqq.h5")


if __name__ == "__main__":
    main()
