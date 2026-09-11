# Measure density-split operators on the cluster

This entry point measures ingredients for the quadratic-selection diagnostic.
It does not run fits, load desilike, or require FOLPS. Run it from the repository
root with the cluster's installed ACM/JAXPower environment.

## Install and verify the environment

Required packages are ACM, JAXPower, JAX/jaxlib, NumPy, SciPy, h5py,
hdf5plugin, and ACM's estimator import dependencies (including pandas,
lsstypes, matplotlib and pycorr). Use the cluster's appropriate JAX build;
the locally tested versions below are provenance, not a portable binary lockfile.

| Package | Locally tested version |
|---|---|
| Python | 3.13.13 |
| ACM / JAXPower | 0.2.0 / 0.2.0 |
| JAX / jaxlib | 0.10.0 / 0.10.0 |
| NumPy / SciPy | 2.4.4 / 1.17.1 |
| h5py / hdf5plugin | 3.16.0 / 7.0.0 |
| lsstypes / pandas | 1.0.0 / 3.0.2 |

Tested source revisions: ACM `cc953ec0a29d669135db275a4bfad65e9f5ed98a`,
JAXPower `31f70b7995a3639a68f67d3ac6930566b667fe73`.
The local ACM checkout has modifications, but its added `quantile_pair_power`
method is not used by this entry point. It measures pairs itself.

```bash
python -m pytest tests/test_measure_density_split_operators.py tests/test_measure_quijote_acm.py -q
python -m scripts.measure_density_split_operators --help
```

## Run one realization first

Supply the directory immediately containing numeric realization directories:

```bash
python -m scripts.measure_density_split_operators \
  --snapshot-root /cluster/path/to/snapshots/fiducial \
  --output-dir /cluster/path/to/operator_spectra \
  --realizations 0
```

The input is `0/snapdir_003/snap_003.*.hdf5`, validated as a complete snapshot
at z=0.5, box size 1000 Mpc/h, fiducial Quijote cosmology. Positions and velocities
are read by `scripts/measure_quijote_acm.py`. Its RSD convention uses the stored
velocity component times `sqrt(a) * (1+z)/[100 sqrt(Omega_m (1+z)^3 + Omega_Lambda)]`,
where `a=1/(1+z)`. Gadget stores `v_pec/sqrt(a)`, so the sqrt(a) factor converts
to peculiar velocities. This matches the cluster's locally preserved
`stash@{0}` version of `scripts/measure_clustering.py` at
`ae3f2cf9b7817d54047f1f559a97adc2ad8c11ab`. Operator cache identity version 4
records this conversion. Use a separate output directory when remeasuring;
version-3 outputs used the missing-factor reader and must be preserved for comparison.

The selection is always CIC, uncompensated, without interlacing, on a 256^3
mesh, with Gaussian smoothing R=10 Mpc/h and CIC readout at lattice queries.
O1=s and O2=(s²-mean(s²))/2 use the actual quantile-assignment values. Signed
weights are painted at those positions and normalized by the parent query
lattice density. Spectrum painting is compensated TSC without interlacing;
the matter leg uses the same settings. LOS is z.

The optional `--analysis-mesh 512` and `--precision float32` select separate
numerical variants, defaulting to 256 and float64. They retain the original
selection mesh and quantile assignment. The precision option changes operator
arithmetic and analysis arrays; it does not change the snapshot reader or force
all internal ACM painting arithmetic to that precision.

Multiple explicit IDs run sequentially in the supplied order. For scheduler
arrays, supply one ID per task using the cluster's actual realization list;
do not assume that 1,500 IDs necessarily means 0 through 1499. Do not launch
two jobs for the same ID/settings/output path. Exceptions stop the invocation
with a nonzero exit status; completed realization caches remain available.

## Output and cache contract

Each `operators_<ID:05d>_<mesh>_<precision>.npz` contains:

- `power`: shape `(7,7,3,15)`, symmetric, in field order
  `m,O1,O2,q1,q2,q4,q5`, multipoles `(0,2,4)`, and k bins of width 0.01
  through 0.15 h/Mpc. These are total powers in (Mpc/h)^3.
- `k`, `nmodes`: length 15, mean lattice radius and full-lattice mode count.
  The first bin includes the zero mode in its count, following JAXPower.
- `fractions`: length 5, Q1 through Q5. Q3 is checked through the weighted
  partition identity, but is not a separate saved spectral-matrix row.
- `identity`: JSON containing bin edges, geometry, source headers/files,
  settings, software versions, available Git revisions, tracked-change flags
  and SHA256 hashes of the two measurement scripts.
- `checks`: JSON with the O2 mean, partition residual and minimum monopole
  spectral-matrix eigenvalue. `seconds` records original measurement runtime.

```python
import json
import numpy as np
with np.load('operators_00000_256_float64.npz', allow_pickle=False) as result:
    power = result['power']
    basis = power[[1, 2, 1, 1, 2], [0, 0, 1, 2, 2]]
    identity = json.loads(str(result['identity']))
# basis order: P1m, P2m, P11, P12, P22; shape (5,3,15)
```

These measured spectra already contain smoothing, painting and angular mixing.
Do not apply another window. No operator shot-noise subtraction is invented.
The matrix includes matching total Pqm and all ten independent Pqq pairs.
Existing estimator noise arrays must be read separately from the saved DS files
for comparison; this measurement command does not read those files.

Caches are reused only when source metadata, software provenance and settings
match. Old cache schemas are recomputed. Completed files are written atomically;
nonfinite or malformed matching caches raise an error. Only spectra and metadata
are saved, not particle catalogs or 3D meshes. Transfer these NPZ files and the
run logs, along with the exact repository commit used.

## Required measurement-reproduction gate

The local realization-0 remeasurement did **not** reproduce the existing saved
DS spectra after exact estimator noise add-back. For bins 0.01–0.12 h/Mpc,
using the empirical single-realization covariance from 1,500 realizations:

| Statistic | Delta chi2 |
|---|---:|
| Pqm | 67.226 |
| Pqq | 42.891 |
| Joint | 197.716 |

A native ACM Pqm calculation agreed with the helper to delta chi2=5.4e-7.
The published readers at `bdc35d3` omitted the sqrt(a) conversion present in the
preserved cluster script. The readers now include it, but the corrected spectra
have not yet passed the numerical reproduction gate. These discrepancy numbers
describe the uncorrected measurements and are not evidence against the quadratic
selection model.

The cluster session must first establish snapshot identity, velocity units/RSD
conversion, smoothing, painting and quantile conventions from provenance.
Compare a few matching realizations to saved `pqm.h5` and `pqq.h5` using their
exact `value + shotnoise` total spectra. Require delta chi2 < 0.01 against the
existing covariance before launching all 1,500. Do not change a convention
merely to improve agreement. If provenance remains unresolved, stop and report
it. This CLI performs field checks but does not automate that covariance gate.

Passing code tests or this gate does not certify cosmological recovery or a
new scale cut. The spectra will be analyzed separately after transfer.
