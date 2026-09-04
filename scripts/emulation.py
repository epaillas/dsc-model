"""Shared matter-emulator training and caches; no model physics lives here."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np


COSMOLOGY_NAMES = ("h", "omega_b", "omega_cdm", "n_s", "logA")
MLP_VERSION = 1
MLP_VALIDATION_SIZE = 256
MLP_ERROR_THRESHOLD = 0.001
# Pin desilike's current default schedule so it is part of the cache identity.
MLP_TRAINING = dict(validation_frac=0.1, optimizer="adam", loss=None,
                    batch_frac=[0.1, 0.3, 1.0], epochs=1000,
                    learning_rate=[1e-2, 1e-3, 1e-5],
                    learning_rate_scheduling=False, batch_norm=False, patience=100)


def validate_mlp_config(config):
    """Normalize the deliberately small YAML schema."""
    if not isinstance(config, dict) or set(config) - {"bounds", "ntrain", "seed"}:
        raise ValueError("MLP config must contain bounds and optional ntrain/seed only")
    bounds = config.get("bounds")
    if not isinstance(bounds, dict) or set(bounds) != set(COSMOLOGY_NAMES):
        raise ValueError(f"MLP bounds must specify exactly {COSMOLOGY_NAMES}")
    normalized = {}
    for name in COSMOLOGY_NAMES:
        pair = bounds[name]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"MLP bounds for {name} must be [lower, upper]")
        try:
            low, high = map(float, pair)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid MLP bounds for {name}: {pair}") from exc
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"MLP bounds for {name} must be finite and increasing")
        normalized[name] = [low, high]
    ntrain, seed = config.get("ntrain", 4096), config.get("seed", 42)
    if type(ntrain) is not int or ntrain < 10:
        raise ValueError("MLP ntrain must be an integer >= 10 (for the validation split)")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("MLP seed must be an integer in [0, 2**32)")
    return dict(bounds=normalized, ntrain=ntrain, seed=seed)


def read_mlp_config(path):
    import yaml

    try:
        return validate_mlp_config(yaml.safe_load(Path(path).read_text()))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid MLP configuration {path}: {exc}") from exc


def check_fixed_bounds(config, fixed_values):
    for name, value in fixed_values.items():
        low, high = config["bounds"][name]
        if not low <= value <= high:
            raise ValueError(f"fixed {name}={value} is outside MLP bounds [{low}, {high}]")


def apply_mlp_priors(calculator, config):
    """Bound the complete likelihood before fixing parameters; never edit caches."""
    for name, (low, high) in config["bounds"].items():
        param = calculator.all_params[name]
        value = param.value if low < param.value < high else (low + high) / 2
        ref = param.ref.__getstate__()
        ref["limits"] = (low, high)
        if "loc" in ref and not low < ref["loc"] < high:
            ref["loc"] = value
        param.update(value=value, prior=dict(dist="uniform", limits=(low, high)), ref=ref)


def matter_cache_options(k, *, redshift, ells, rsd, cache_version=2,
                         order=3, accuracy=2, method="finite", m_ncdm=0.,
                         emulator="taylor", mlp_config=None):
    settings = dict(engine="TaylorEmulatorEngine", order=order, accuracy=accuracy, method=method)
    if emulator == "mlp":
        config = validate_mlp_config(mlp_config)
        settings = dict(engine="MLPEmulatorEngine", version=MLP_VERSION, **config,
                        nhidden=[64, 64, 64], activation="silu",
                        dtype="float64",
                        xoperation="default-scale", yoperation=["log10", "default-scale"],
                        sampling="scrambled-sobol-base2-prefix-v1",
                        training=dict(MLP_TRAINING, seed=config["seed"]),
                        validation=dict(nrandom=MLP_VALIDATION_SIZE, faces=True,
                                        seed=(config["seed"] + 1) % 2**32,
                                        threshold=MLP_ERROR_THRESHOLD))
    elif emulator != "taylor" or mlp_config is not None:
        raise ValueError("mlp_config is only supported with emulator='mlp'")
    return dict(cache_version=cache_version, model="linear-delta-cb-kaiser",
                k=np.asarray(k, dtype="f8").tolist(), redshift=float(redshift),
                ells=[int(ell) for ell in ells], rsd=bool(rsd),
                fixed_cosmology={"m_ncdm": m_ncdm}, emulator=settings)


def emulator_path(output_dir, options):
    serialized = json.dumps(options, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(serialized.encode()).hexdigest()[:8]
    return Path(output_dir) / f"pmm_emulator_{digest}.npy"


def sobol_points(config):
    from scipy.stats import qmc

    ntrain = config["ntrain"]
    unit = qmc.Sobol(d=5, scramble=True, seed=config["seed"]).random_base2(
        int(np.ceil(np.log2(ntrain))))[:ntrain]
    bounds = np.array([config["bounds"][name] for name in COSMOLOGY_NAMES])
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])


def validation_points(settings):
    bounds = np.array([settings["bounds"][name] for name in COSMOLOGY_NAMES])
    validation = settings["validation"]
    rng = np.random.default_rng(validation["seed"])
    unit = rng.uniform(size=(validation["nrandom"], 5))
    faces = np.full((10, 5), 0.5)
    for axis in range(5):
        faces[2 * axis, axis], faces[2 * axis + 1, axis] = 0., 1.
    return bounds[:, 0] + np.concatenate([unit, faces]) * np.diff(bounds, axis=1).ravel()


def evaluate_samples(calculator, points):
    """Evaluate explicit cosmologies using desilike's MPI-aware sampling machinery."""
    from desilike.base import vmap
    from desilike.samples import Samples

    comm = calculator.mpicomm
    samples = Samples(points.T, params=[calculator.all_params[name] for name in COSMOLOGY_NAMES])
    power = vmap(calculator, backend="mpi", errors="raise")(
        samples.to_dict() if comm.rank == 0 else {}, mpicomm=comm)
    if comm.rank == 0:
        from desilike.parameter import Parameter
        power = np.asarray(power)
        samples[Parameter("power", derived=True, shape=power.shape[1:])] = power
        valid = np.isfinite(power).all() and (power > 0).all()
    else:
        valid = None
    if not comm.bcast(valid, root=0):
        raise ValueError("MLP requires finite, strictly positive matter spectra")
    return samples if comm.rank == 0 else None


def validate_predictions(truth, prediction, ells):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid shape or non-finite MLP validation predictions")
    error = np.abs(prediction / truth - 1).reshape(len(truth), len(ells), -1)
    if not np.isfinite(error).all():
        raise ValueError("non-finite MLP fractional validation errors")
    def metrics(values):
        return dict(max=float(np.max(values)), rms=float(np.sqrt(np.mean(values**2))),
                    p99=float(np.quantile(values, 0.99)))
    return dict(nvalidation=len(truth), overall=metrics(error),
                by_ell={str(ell): metrics(error[:, i]) for i, ell in enumerate(ells)})


def report_accuracy(report, path):
    metrics = report["validation"]
    maximum = metrics["overall"]["max"]
    by_ell = "; ".join(f"ell={ell}: max={group['max']:.3%}, RMS={group['rms']:.3%}"
                       for ell, group in metrics["by_ell"].items())
    print(f"MLP validation {path}: max={maximum:.3%}, "
          f"RMS={metrics['overall']['rms']:.3%}; {by_ell}", flush=True)
    if maximum > report["configuration"]["emulator"]["validation"]["threshold"]:
        warnings.warn(f"MLP ACCURACY WARNING: {path}: maximum fractional spectrum error "
                      f"{maximum:.3%} exceeds 0.1%. Continuing; posterior accuracy is not assured.",
                      RuntimeWarning, stacklevel=2)


def get_or_train_emulator(output_dir, options, build_exact):
    from desilike.emulators import Emulator, EmulatedCalculator, TaylorEmulatorEngine

    path = emulator_path(output_dir, options)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = options["emulator"]
    is_mlp = settings["engine"] == "MLPEmulatorEngine"
    report_path = path.with_suffix(".json")
    if path.exists() or (is_mlp and report_path.exists()):
        try:
            if is_mlp:
                report = json.loads(report_path.read_text())
                if report["configuration"] != options:
                    raise ValueError(f"incompatible configuration in {report_path}")
                if report["model_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError(f"model checksum does not match {report_path}")
                metrics = report["validation"]
                if (metrics["nvalidation"] != settings["validation"]["nrandom"] + 10
                        or set(metrics["by_ell"]) != set(map(str, options["ells"]))):
                    raise ValueError(f"incompatible validation in {report_path}")
                values = [v for group in [metrics["overall"], *metrics["by_ell"].values()]
                          for v in (group["max"], group["rms"], group["p99"])]
                if not np.isfinite(values).all() or min(values) < 0:
                    raise ValueError(f"invalid validation metrics in {report_path}")
            calculator = EmulatedCalculator.load(str(path))
            if is_mlp:
                power = np.asarray(calculator())
                if not np.isfinite(power).all() or not (power > 0).all():
                    raise ValueError("non-finite or non-positive loaded MLP prediction")
                report_accuracy(report, path)
            return calculator
        except Exception as exc:
            raise RuntimeError(f"failed to load cached pmm emulator {path}"
                               f"{(' / ' + str(report_path)) if is_mlp else ''}; "
                               "remove this artifact and rerun to regenerate it") from exc

    exact = build_exact()
    if not is_mlp:
        emulator = Emulator(exact, engine=TaylorEmulatorEngine(
            **{name: settings[name] for name in ("order", "accuracy", "method")}))
        emulator.set_samples()
        emulator.fit()
        emulator.save(str(path))
        return emulator.to_calculator()

    from desilike.emulators import MLPEmulatorEngine, Log10Operation

    # Initialize inside the training domain even if it excludes the model defaults.
    apply_mlp_priors(exact, settings)
    if set(exact.varied_params.names()) != set(COSMOLOGY_NAMES):
        raise ValueError("MLP matter training requires all five cosmological parameters")
    engine = MLPEmulatorEngine(
        nhidden=tuple(settings["nhidden"]), activation=settings["activation"],
        yoperation=Log10Operation())
    emulator = Emulator(exact, engine=engine)
    print(f"Training matter MLP: {settings['ntrain']} Sobol cosmologies; cache {path}", flush=True)
    try:
        samples = evaluate_samples(exact, sobol_points(settings))
        # This desilike version's explicit-samples branch expects cosmoprimo
        # columns(), but its conversion hook expects desilike Samples. Supply
        # our already-evaluated samples through the engine's sampling hook;
        # never invoke the default 100,000-point proposal-based sampler.
        engine.get_default_samples = lambda calculator, **kwargs: samples
        try:
            emulator.set_samples()
        finally:
            del engine.get_default_samples
        emulator.fit(**settings["training"])
        calculator = emulator.to_calculator()
        points = validation_points(settings)
        print(f"Validating matter MLP against {len(points)} independent exact cosmologies", flush=True)
        truth = evaluate_samples(exact, points)
        prediction = evaluate_samples(calculator, points)
        publication_error = None
        if exact.mpicomm.rank == 0:
            try:
                validation = validate_predictions(truth["power"], prediction["power"], options["ells"])
                # Publish the report last, tying it to the actual serialized model.
                with tempfile.TemporaryDirectory(dir=path.parent, prefix=".mlp-") as staging:
                    staged = Path(staging) / path.name
                    emulator.save(str(staged), yaml=False)
                    report = dict(configuration=options, validation=validation,
                                  model_sha256=hashlib.sha256(staged.read_bytes()).hexdigest())
                    staged.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False))
                    for suffix in (".npy", ".json"):
                        os.replace(staged.with_suffix(suffix), path.with_suffix(suffix))
                report_accuracy(report, path)
            except Exception as exc:
                publication_error = str(exc)
        publication_error = exact.mpicomm.bcast(publication_error, root=0)
        if publication_error:
            raise RuntimeError(publication_error)
    except Exception as exc:
        raise RuntimeError(f"MLP training/validation failed for {path}: {exc}") from exc
    return calculator
