# Measure density-split operators on the cluster

This entry point measures ingredients for quadratic and cubic selection diagnostics.
It does not run fits, load desilike, or require FOLPS. Run it from the repository
root with the cluster's installed ACM/JAXPower environment. Cubic fitting and
analytic cubic predictions are separate work.

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

This defaults to `--operator-order 2`. To include the cubic operator:

```bash
python -m scripts.measure_density_split_operators \
  --snapshot-root /cluster/path/to/snapshots/fiducial \
  --output-dir /cluster/path/to/operator_spectra \
  --realizations 0 \
  --operator-order 3
```

The Python entry point also accepts `measure(..., operator_order=3)`.

The input is `0/snapdir_003/snap_003.*.hdf5`, validated as a complete snapshot
at z=0.5, box size 1000 Mpc/h, fiducial Quijote cosmology. Positions and velocities
are read by `scripts/measure_quijote_acm.py`. Its RSD convention uses the stored
velocity component times `sqrt(a) * (1+z)/[100 sqrt(Omega_m (1+z)^3 + Omega_Lambda)]`,
where `a=1/(1+z)`. Gadget stores `v_pec/sqrt(a)`, so the sqrt(a) factor converts
to peculiar velocities. This matches the cluster's locally preserved
`stash@{0}` version of `scripts/measure_clustering.py` at
`ae3f2cf9b7817d54047f1f559a97adc2ad8c11ab`. Commit `b469ae8` restored this
conversion; quadratic schema 4 and cubic schema 5 record it. The corrected
measurements passed the reproduction checks described below.
Use a separate output directory when remeasuring old files;
version-3 outputs used the missing-factor reader and must be preserved for comparison.

The selection is always CIC, uncompensated, without interlacing, on a 256^3
mesh, with Gaussian smoothing R=10 Mpc/h and CIC readout at lattice queries.
O1=s and O2=(s²-mean(s²))/2 use the actual quantile-assignment values. Signed
weights are painted at those positions and normalized by the parent query
lattice density. Spectrum painting is compensated TSC without interlacing;
the matter leg uses the same settings. LOS is z.

Order 3 appends `O3=(s³-mean(s³))/6`. Only the cubic mean is subtracted:
there is no Hermite subtraction, removal of its linear response, or variance
normalization. It uses the same signed painting and parent-lattice normalization
as O1 and O2, never normalization by its own mean weight. Snapshot loading and
selection run once; only the additional low-k Fourier modes are retained after
painting O3. The original seven fields retain their order and definitions.

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

Quadratic files retain the name `operators_<ID:05d>_<mesh>_<precision>.npz`
and schema version 4. Cubic files use
`operators3_<ID:05d>_<mesh>_<precision>.npz` and schema version 5. They can coexist;
the existing quadratic-analysis glob `operators_*_256_float64.npz` excludes
cubic files. Each file contains:

- `power`: symmetric, with multipoles `(0,2,4)` and k bins of width 0.01
  through 0.15 h/Mpc. These are total powers in (Mpc/h)^3. Quadratic shape
  is `(7,7,3,15)`, field order `m,O1,O2,q1,q2,q4,q5` (28 independent spectra).
  Cubic shape is `(8,8,3,15)`, field order `m,O1,O2,q1,q2,q4,q5,O3`
  (36 independent spectra).
- `k`, `nmodes`: length 15, mean lattice radius and full-lattice mode count.
  The first bin includes the zero mode in its count, following JAXPower.
- `fractions`: length 5, Q1 through Q5. Q3 is checked through the weighted
  partition identity, but is not a separate saved spectral-matrix row.
- `identity`: JSON containing bin edges, geometry, source headers/files,
  operator order, field order, exact operator definitions, settings, software
  versions, available Git revisions, tracked-change flags and SHA256 hashes
  of the two measurement scripts.
- `checks`: JSON with the O2 mean, partition residual and minimum monopole
  spectral-matrix eigenvalue, plus `o3_mean` for cubic files. `seconds` records
  original measurement runtime.

Zero-based cubic field indices are:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Field | m | O1 | O2 | q1 | q2 | q4 | q5 | O3 |

The eight new independent spectra are `power[7,0]` (P3m), `power[1,7]`
(P13), `power[2,7]` (P23), `power[7,7]` (P33), and `power[7,3:7]`
(the four P3q cross-spectra).

```python
import json
import numpy as np
with np.load('operators_00000_256_float64.npz', allow_pickle=False) as result:
    power = result['power']
    basis = power[[1, 2, 1, 1, 2], [0, 0, 1, 2, 2]]
    identity = json.loads(str(result['identity']))
# basis order: P1m, P2m, P11, P12, P22; shape (5,3,15)
```

For cubic files, extract the original matrix with `power[:7, :7]` before using
`basis_from_matrix`, `data_from_matrix` or `residual_matrix`. These helpers
explicitly require seven-field matrices; they do not implement cubic fitting.

These measured spectra already contain smoothing, painting and angular mixing.
Do not apply another window. No operator shot-noise subtraction is invented.
The matrix includes matching total Pqm and all ten independent Pqq pairs.
Existing estimator noise arrays must be read separately from the saved DS files
for comparison; this measurement command does not read those files.

Caches are reused only when source metadata, software provenance and settings
match, including the operator order. Dimensions and the recorded field list
are validated; a quadratic cache is never used for a cubic request. Stale
identities are recomputed at the requested output path. Completed files are
written atomically;
nonfinite or malformed matching caches raise an error. Only spectra and metadata
are saved, not particle catalogs or 3D meshes. Transfer these NPZ files and the
run logs, along with the exact repository commit used. Download both quadratic
and cubic NPZ files into `data/operators/z0.5/rsd_z/` under the local `dsc-model`
checkout, keeping their original filenames. The cubic prefix preserves the
current quadratic analysis's input selection.

## Required measurement-reproduction gate

The former saved-spectrum mismatch was caused by the omitted Gadget `sqrt(a)`
velocity conversion. With the corrected reader, realization 0 passed the gate
after exact estimator noise add-back. For bins 0.01–0.12 h/Mpc, using the
empirical single-realization covariance from 1,500 realizations:

| Statistic | Delta chi2 |
|---|---:|
| Pqm | 0.00002364 |
| Pqq | 0.00014960 |
| Joint | 0.00040905 |

The subsequent 85-file comparison also passed: its largest individual joint
delta chi2 was 0.00073011. These checks resolve the measurement convention
mismatch; they do not establish that the quadratic or cubic selection model
describes the measured fields.

Fresh cubic measurements with the default numerical settings also passed for
realizations 0, 1 and 10. Their original DS blocks give the following delta chi2
against the saved total spectra through k=0.12 h/Mpc:

| Realization | Pqm | Pqq | Joint |
|---|---:|---:|---:|
| 0 | 0.00002292 | 0.00014673 | 0.00039912 |
| 1 | 0.00002691 | 0.00018580 | 0.00084984 |
| 10 | 0.00002994 | 0.00014482 | 0.00046143 |

Fresh quadratic runs preserve the original seven-field block: realizations 0
and 10 are bitwise identical, and realization 1 differs by at most 1.8e-8
relative in O1/O2 entries. Repeating the quadratic run reproduces that same
variation and agrees exactly with the cubic file's original block. The DS
entries are bitwise identical in every comparison. Cubic measurements took
about 24 seconds per realization locally; cluster runtimes will depend on
the allocated hardware and snapshot I/O.

All three statistics also pass at kmax=0.05, 0.07 and 0.10 h/Mpc. The focused
tests check the cubic harmonic identity `O3=cos(x)/8+cos(3x)/24` for `s=cos(x)`,
signed painting, planted cubic fields, native JAXPower cross spectra, symmetry,
monopole positive semidefiniteness, cache separation and CLI handling. They run
from a clean export with desilike and FOLPS imports blocked.

For a new cluster environment, first verify snapshot identity, velocity units/RSD
conversion, smoothing, painting and quantile conventions from provenance.
Compare a few matching realizations to saved `pqm.h5` and `pqq.h5` using their
exact `value + shotnoise` total spectra. Require delta chi2 < 0.01 against the
existing covariance before launching all 1,500. Do not change a convention
merely to improve agreement. If provenance remains unresolved, stop and report
it. This CLI performs field checks but does not automate that covariance gate.

Passing code tests or this gate does not certify cosmological recovery or a
new scale cut. The spectra will be analyzed separately after transfer.
