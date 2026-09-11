"""Measure finite-smoothing operators with the existing ACM selection convention."""
from __future__ import annotations

import argparse
import gc
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.special import eval_legendre
from scripts.measure_clustering import discover_realizations
from scripts.measure_quijote_acm import inspect_snapshot, read_snapshot

FIELDS = ('m', 'O1', 'O2', 'q1', 'q2', 'q4', 'q5')
CUBIC_FIELDS = FIELDS + ('O3',)
OPERATOR_DEFINITIONS = dict(O1='s', O2='(s**2 - mean(s**2)) / 2',
                            O3='(s**3 - mean(s**3)) / 6')
REALIZATIONS = (0, 1, 10, 100, 1000, 10000, 10001, 10002, 10003, 10004)


class FourierBins:
    """Low-k rFFT selection and full-lattice, even-multipole bin averages."""

    def __init__(self, meshsize, boxsize=1000., edges=None):
        self.n, self.volume = meshsize, boxsize**3
        self.edges = np.arange(16)*.01 if edges is None else np.asarray(edges)
        f = np.fft.fftfreq(meshsize)*meshsize*2*np.pi/boxsize
        z = np.fft.rfftfreq(meshsize)*meshsize*2*np.pi/boxsize
        radius = np.sqrt(f[:, None, None]**2+f[None, :, None]**2+z[None, None, :]**2)
        self.mask = (radius < self.edges[-1])
        k = radius[self.mask]
        self.index = np.searchsorted(self.edges, k, side='right')-1
        kz = np.broadcast_to(z[None, None, :], radius.shape)[self.mask]
        self.weights = np.where((kz == 0.) | (kz == np.pi*meshsize/boxsize), 1., 2.)
        self.counts = self.bin(self.weights)
        if np.any(self.counts == 0):
            raise ValueError('empty Fourier bins')
        self.k = self.bin(self.weights*k)/self.counts
        self.angular = np.array([(2*ell+1)*eval_legendre(ell, np.divide(kz, k, out=np.zeros_like(k), where=k != 0.)) for ell in (0, 2, 4)])

    def bin(self, values):
        return np.bincount(self.index, weights=values, minlength=len(self.edges)-1)

    def transform(self, mesh):
        from scipy.fft import rfftn
        return rfftn(np.asarray(mesh))[self.mask]/self.n**3

    def matrix(self, modes):
        modes = np.asarray(modes)
        result = np.empty((len(modes), len(modes), 3, len(self.k)))
        for a in range(len(modes)):
            for b in range(a, len(modes)):
                power = (modes[a].conj()*modes[b]).real*self.volume*self.weights
                result[a, b] = result[b, a] = [self.bin(power*w)/self.counts for w in self.angular]
        return result


def atomic_save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def software_metadata():
    """Record installed versions and source identities without importing theory code."""
    versions = {'python': platform.python_version()}
    for package in ('acm', 'jax-power', 'jax', 'jaxlib', 'numpy', 'scipy',
                    'h5py', 'hdf5plugin', 'lsstypes', 'pandas'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    sources = {}
    locations = {'dsc-model': Path(__file__).resolve().parent.parent}
    for module in ('acm', 'jaxpower'):
        spec = util.find_spec(module)
        if spec is not None and spec.origin:
            locations[module] = Path(spec.origin).resolve().parent
    for name, location in locations.items():
        try:
            revision = subprocess.check_output(['git', '-C', str(location), 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL, timeout=5, text=True).strip()
            dirty = subprocess.check_output(['git', '-C', str(location), 'status',
                '--porcelain', '--untracked-files=no'], stderr=subprocess.DEVNULL,
                timeout=5, text=True).strip()
            sources[name] = dict(revision=revision, tracked_changes=bool(dirty))
        except (OSError, subprocess.SubprocessError):
            sources[name] = dict(revision=None, tracked_changes=None)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (Path(__file__), Path(__file__).with_name('measure_quijote_acm.py'))}
    return dict(versions=versions, sources=sources, script_sha256=hashes)


def load_cache(path, identity):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cached:
        if json.loads(str(cached['identity'])) != identity:
            return None
        result = {key: cached[key] for key in cached.files}
    fields = tuple(identity.get('fields', ()))
    order = identity.get('operator_order', 2)
    expected_fields = {2: FIELDS, 3: CUBIC_FIELDS}.get(order)
    if (fields != expected_fields or identity.get('version') != {2: 4, 3: 5}.get(order)
            or result['power'].shape != (len(fields), len(fields), 3, 15)
            or result['k'].shape != (15,) or result['nmodes'].shape != (15,)
            or result['fractions'].shape != (5,)) or not all(
            np.isfinite(result[key]).all() for key in ('power', 'k', 'nmodes', 'fractions')):
        raise ValueError('invalid cached spectral matrix or geometry')
    return result


def cubic_operator(s):
    """Finite-smoothing O3: subtract only the cubic mean, with a factor 1/6."""
    cube = np.asarray(s)**3
    return (cube-cube.mean())/6.


def measure(snapshot_root, realization, output_dir, analysis_mesh=256, precision='float64',
            operator_order=2):
    """Cache spectra only; selection is always the original 256^3 field."""
    if (realization < 0 or analysis_mesh not in (256, 512)
            or precision not in ('float64', 'float32') or operator_order not in (2, 3)):
        raise ValueError('invalid realization, analysis mesh, precision or operator order')
    import jax
    jax.config.update('jax_enable_x64', True)
    from jaxpower import MeshAttrs, ParticleField
    from acm.estimators.galaxy_clustering.density_split import DensitySplit

    files, header = inspect_snapshot(Path(snapshot_root)/str(realization)/'snapdir_003')
    if not np.isclose(header['redshift'], .5) or header['boxsize_mpc_h'] != 1000.:
        raise ValueError('expected z=0.5, L=1000 snapshot')
    if not np.allclose([header['omega_m'], header['hubble_param']], [.3175, .6711]):
        raise ValueError('expected fiducial Quijote cosmology')
    fields = FIELDS if operator_order == 2 else CUBIC_FIELDS
    identity = dict(version=4 if operator_order == 2 else 5, operator_order=operator_order,
        operator_definitions={name: OPERATOR_DEFINITIONS[name] for name in fields if name.startswith('O')},
        realization=realization, header=header, software=software_metadata(),
        sources=[dict(path=str(p.resolve()), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files],
        selection_mesh=256, radius=10., selection='cic, uncompensated, no interlacing; CIC lattice readout',
        analysis_mesh=analysis_mesh, precision=precision, painting='tsc compensated, interlacing=0',
        los='z', rsd_velocity_conversion='v_pec = sqrt(a) * stored_velocity',
        ells=[0, 2, 4], edges=(np.arange(16)*.01).tolist(), fields=fields)
    # JSON round trip makes tuple/list comparisons stable.
    identity = json.loads(json.dumps(identity))
    prefix = 'operators' if operator_order == 2 else 'operators3'
    path = Path(output_dir)/f'{prefix}_{realization:05d}_{analysis_mesh}_{precision}.npz'
    cached = load_cache(path, identity)
    if cached is not None:
        print(f'Realization {realization}: using {path}', flush=True)
        return cached
    started = time.perf_counter()
    print(f'Realization {realization}: loading snapshot (order {operator_order}, {analysis_mesh}, {precision})', flush=True)
    positions = read_snapshot(files, header, los='z', rsd=True)
    ds = DensitySplit(data_positions=positions, boxsize=1000., boxcenter=0., meshsize=256)
    ds.set_density_contrast(resampler='cic', interlacing=False, compensate=False, smoothing_radius=10.)
    ds.set_quantiles(query_method='lattice', nquantiles=5)
    query = np.asarray(ds.query_positions)
    s = np.asarray(ds.delta_query, dtype=precision)
    labels = np.asarray(ds.quantiles_idx)
    fractions = np.bincount(labels, minlength=5)/len(labels)
    del ds
    gc.collect()
    attrs = MeshAttrs(meshsize=analysis_mesh, boxsize=1000., boxcenter=0.)
    bins = FourierBins(analysis_mesh)

    def paint(points, weights=None, normalization=None):
        mesh = ParticleField(points, weights=weights, attrs=attrs, backend='jax', exchange=True).paint(
            resampler='tsc', compensate=True, interlacing=0, out='real')
        values = np.asarray(mesh).astype(precision, copy=False)
        values = values/(values.mean() if normalization is None else normalization)
        return bins.transform(values-values.mean())

    modes = [paint(positions)]
    del positions
    parent_mean = len(query)/analysis_mesh**3
    modes.append(paint(query, s, parent_mean))
    o2 = .5*(s*s-np.mean(s*s))
    zero = float(np.mean(o2))
    modes.append(paint(query, o2, parent_mean))
    del o2
    if operator_order == 3:
        o3 = cubic_operator(s)
        cubic_mean = float(o3.mean())
        cubic_modes = paint(query, o3, parent_mean)
        del o3
    del s
    partition = np.zeros_like(modes[0])
    for q in range(5):
        mode = paint(query[labels == q])
        partition += fractions[q]*mode
        if q != 2:
            modes.append(mode)
    if operator_order == 3:
        modes.append(cubic_modes)
    power = bins.matrix(modes)
    minimum = min(np.linalg.eigvalsh(power[:, :, 0, i]).min() for i in range(len(bins.k)))
    checks = dict(o2_mean=zero, partition_mode_max=float(np.abs(partition).max()),
                  monopole_min_eigenvalue=float(minimum))
    if operator_order == 3:
        checks['o3_mean'] = cubic_mean
    if not np.isfinite(power).all() or not all(np.isfinite(value) for value in checks.values()):
        raise RuntimeError('nonfinite spectra or field checks')
    if checks['partition_mode_max'] > 1.e-6 or minimum < -1.e-7*np.max(np.abs(power)):
        raise RuntimeError(f'field identity failed: {checks}')
    result = dict(power=power, k=bins.k, nmodes=bins.counts, fractions=fractions,
                  identity=json.dumps(identity), checks=json.dumps(checks), seconds=time.perf_counter()-started)
    atomic_save(path, **result)
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['power'], power)
    jax.clear_caches()
    gc.collect()
    return result


def _quadratic_matrix(power):
    power = np.asarray(power)
    if power.ndim != 4 or power.shape[:2] != (7, 7):
        raise ValueError('quadratic helpers require seven fields; use power[:7, :7] for cubic files')
    return power


def basis_from_matrix(power):
    power = _quadratic_matrix(power)
    return np.array([power[1, 0], power[2, 0], power[1, 1], power[1, 2], power[2, 2]])


def data_from_matrix(power):
    power = _quadratic_matrix(power)
    return np.concatenate([power[3:, 0].ravel(),
        np.array([power[a, b] for a in range(3, 7) for b in range(a, 7)]).ravel()])


def residual_matrix(power, c1, c2):
    """Rows are m,O1,O2,epsilon1,epsilon2,epsilon4,epsilon5."""
    power = _quadratic_matrix(power)
    transform = np.eye(7)
    transform[3:, 1] = -np.asarray(c1)
    transform[3:, 2] = -np.asarray(c2)
    return np.einsum('ai,ijlk,bj->ablk', transform, power, transform)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-root', type=Path, required=True,
                        help='Directory containing numeric fiducial realization directories')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--realizations', type=int, nargs='+',
                        help='Realization IDs to process. If omitted, process every numeric '
                             'directory under --snapshot-root.')
    parser.add_argument('--analysis-mesh', type=int, choices=(256, 512), default=256)
    parser.add_argument('--precision', choices=('float64', 'float32'), default='float64')
    parser.add_argument('--operator-order', type=int, choices=(2, 3), default=2,
                        help='Highest selection-operator order (default: 2)')
    args = parser.parse_args(argv)
    realizations = args.realizations
    if realizations is None:
        realizations = discover_realizations(args.snapshot_root)
        if not realizations:
            raise FileNotFoundError(
                f'no numeric realization directories in {args.snapshot_root}'
            )
        print(f'Discovered {len(realizations)} realizations in {args.snapshot_root}.', flush=True)
    if any(i < 0 for i in realizations) or len(set(realizations)) != len(realizations):
        parser.error('realizations must be distinct nonnegative integers')
    for realization in realizations:
        measure(args.snapshot_root, realization, args.output_dir,
                analysis_mesh=args.analysis_mesh, precision=args.precision,
                operator_order=args.operator_order)


if __name__ == '__main__':
    main()
