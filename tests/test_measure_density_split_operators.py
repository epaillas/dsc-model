"""Measurement tests independent of local fitting code and desilike."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scripts import measure_density_split_operators as measurement
from scripts.measure_density_split_operators import (
    FourierBins, residual_matrix, basis_from_matrix, data_from_matrix, atomic_save,
)


def test_plane_wave_and_quadratic_normalization():
    n, length = 16, 100.
    bins = FourierBins(n, length, edges=np.array([0., .09, .15]))
    x = np.arange(n)*2*np.pi/n
    s = np.broadcast_to(np.cos(x)[:, None, None], (n, n, n))
    o2 = .5*(s*s-np.mean(s*s))
    assert abs(o2.mean()) < 1.e-16
    f = np.fft.fftn(o2)/n**3
    np.testing.assert_allclose(f[2, 0, 0], .125, atol=1.e-16)
    modes = [bins.transform(s), bins.transform(o2)]
    matrix = bins.matrix(modes)
    np.testing.assert_allclose(matrix[0, 0, 0, 0]*bins.counts[0], length**3/2)
    np.testing.assert_allclose(matrix[1, 1, 0, 1]*bins.counts[1], length**3/32)
    np.testing.assert_array_equal(matrix, matrix.swapaxes(0, 1))
    full = np.fft.fftfreq(n)*n*2*np.pi/length
    radius = np.sqrt(sum(v*v for v in np.meshgrid(full, full, full, indexing='ij')))
    np.testing.assert_array_equal(bins.counts, np.histogram(radius, bins.edges)[0])


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_cubic_mean_and_sinusoid_normalization(dtype):
    n, length = 32, 100.
    x = np.arange(n)*2*np.pi/n
    s = np.broadcast_to(np.cos(x)[:, None, None], (n, n, n)).astype(dtype)
    o3 = measurement.cubic_operator(s)
    tolerance = 5.e-8 if dtype == 'float32' else 1.e-15
    assert o3.dtype == s.dtype
    assert abs(o3.mean()) < tolerance
    expected = np.broadcast_to((np.cos(x)/8+np.cos(3*x)/24)[:, None, None], s.shape)
    np.testing.assert_allclose(o3, expected, atol=tolerance)
    f = np.fft.fftn(o3)/n**3
    np.testing.assert_allclose(f[[1, 3], 0, 0], [1/16, 1/48], atol=tolerance)
    bins = FourierBins(n, length, edges=[0., .09, .15, .21])
    matrix = bins.matrix([bins.transform(s), bins.transform(o3)])
    np.testing.assert_allclose(matrix[0, 1, 0, 0]*bins.counts[0], length**3/16, rtol=1.e-6)
    np.testing.assert_allclose(matrix[1, 1, 0, [0, 2]]*bins.counts[[0, 2]],
                               length**3*np.array([1/128, 1/1152]), rtol=1.e-6)
    # A nonzero third moment exercises mean subtraction, which a pure cosine cannot.
    shifted = s + .7
    cube = shifted.astype('float64')**3
    np.testing.assert_allclose(measurement.cubic_operator(shifted),
                               (cube-cube.mean())/6, atol=5*tolerance)


@pytest.mark.parametrize('cross', ['random', 'm', 'O1', 'O2', 'O3'])
def test_native_jaxpower_normalization(cross):
    import jax
    jax.config.update('jax_enable_x64', True)
    from jaxpower import MeshAttrs, RealMeshField, BinMesh2SpectrumPoles, compute_mesh2_spectrum
    n, length = 16, 100.
    edges = np.array([0., .09, .15])
    attrs = MeshAttrs(meshsize=n, boxsize=length)
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(2, n, n, n))
    if cross != 'random':
        s = b-b.mean()
        o3 = measurement.cubic_operator(s)
        a = {'m': a, 'O1': s, 'O2': .5*(s*s-np.mean(s*s)), 'O3': o3}[cross]
        b = o3
    a -= a.mean()
    b -= b.mean()
    bins = FourierBins(n, length, edges)
    ours = bins.matrix([bins.transform(a), bins.transform(b)])[0, 1]
    native = compute_mesh2_spectrum(RealMeshField(a, attrs=attrs), RealMeshField(b, attrs=attrs),
        bin=BinMesh2SpectrumPoles(attrs, edges=edges, ells=(0,2,4)), los='z')
    np.testing.assert_allclose(ours, np.array([native.get(ells=ell).value() for ell in (0,2,4)]), atol=1.e-10)


def test_planted_fields_and_residual_identity():
    rng = np.random.default_rng(12)
    modes = rng.normal(size=(3, 70))+1j*rng.normal(size=(3, 70))
    c1, c2 = np.array([-2., -1., 1., 2.]), np.array([.5, -.7, .2, 1.])
    fields = np.vstack([modes, c1[:, None]*modes[1]+c2[:, None]*modes[2]])
    power = np.real(np.einsum('ik,jk->ijk', fields.conj(), fields))[:, :, None, :]
    residual = residual_matrix(power, c1, c2)
    np.testing.assert_allclose(residual[3:], 0., atol=1.e-13)
    basis = basis_from_matrix(power)
    np.testing.assert_allclose(power[3:,0], c1[:,None,None]*basis[0]+c2[:,None,None]*basis[1])
    assert data_from_matrix(power).size == 14*70
    eigenvalues = np.linalg.eigvalsh(power[:,:,0].transpose(2,0,1))
    assert eigenvalues.min() > -1.e-12


@pytest.mark.parametrize('cubic_response', [0., 1.])
def test_planted_cubic_fields_and_original_block(cubic_response):
    rng = np.random.default_rng(17)
    m, s = rng.normal(size=(2, 16, 16, 16))
    s -= s.mean()
    o2, o3 = .5*(s*s-np.mean(s*s)), measurement.cubic_operator(s)
    c1, c2 = np.array([-2., -1., 1., 2.]), np.array([.5, -.7, .2, 1.])
    c3 = cubic_response*np.array([.3, -.6, .4, 1.5])
    q = c1[:, None, None, None]*s + c2[:, None, None, None]*o2 + c3[:, None, None, None]*o3
    bins = FourierBins(16, 500.)
    modes = [bins.transform(field) for field in (m, s, o2, *q, o3)]
    power = bins.matrix(modes)
    np.testing.assert_array_equal(power[:7, :7], bins.matrix(modes[:7]))
    np.testing.assert_array_equal(power, power.swapaxes(0, 1))
    transform = np.eye(8)
    transform[3:7, 1], transform[3:7, 2], transform[3:7, 7] = -c1, -c2, -c3
    residual = np.einsum('ai,ijlk,bj->ablk', transform, power, transform)
    np.testing.assert_allclose(residual[3:7], 0., atol=1.e-9)
    assert np.linalg.eigvalsh(power[:, :, 0].transpose(2, 0, 1)).min() > -1.e-9
    if not cubic_response:
        np.testing.assert_allclose(residual_matrix(power[:7, :7], c1, c2)[3:], 0., atol=1.e-9)
    for helper, args in [(basis_from_matrix, ()), (data_from_matrix, ()),
                          (residual_matrix, (c1, c2))]:
        with pytest.raises(ValueError, match='seven fields'):
            helper(power, *args)


def test_signed_painting_and_partition():
    import jax
    jax.config.update('jax_enable_x64', True)
    from jaxpower import MeshAttrs, ParticleField
    n = 8
    attrs = MeshAttrs(meshsize=n, boxsize=80.)
    coords = attrs.rcoords()
    positions = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1).reshape(-1, 3)
    positions = np.repeat(positions, 2, axis=0)
    rng = np.random.default_rng(7)
    positions = positions[rng.permutation(len(positions))]
    s = np.cos(2*np.pi*positions[:, 0]/80.)
    o2 = .5*(s*s-np.mean(s*s))
    o3 = measurement.cubic_operator(s)
    parent_mean = len(positions)/n**3

    def paint(weights):
        return np.asarray(ParticleField(positions, weights=weights, attrs=attrs, backend='jax').paint(
            resampler='tsc', compensate=True, interlacing=0, out='real'))/parent_mean

    parent = paint(np.ones(len(s)))
    np.testing.assert_allclose(parent, parent.mean(), atol=1.e-12)
    np.testing.assert_allclose(paint(1.3*s-.7*o2), 1.3*paint(s)-.7*paint(o2), atol=1.e-12)
    np.testing.assert_allclose(paint(1.3*s-.7*o2+.8*o3),
                               1.3*paint(s)-.7*paint(o2)+.8*paint(o3), atol=1.e-12)
    assert abs(paint(o3).mean()) < 1.e-14
    assert np.isfinite(paint(o3)).all()
    labels = np.arange(len(s)) % 5
    fractions = np.bincount(labels)/len(labels)
    partition = sum(fractions[q]*(paint((labels==q).astype(float))/fractions[q]/parent.mean()-1.) for q in range(5))
    np.testing.assert_allclose(partition, 0., atol=1.e-12)


@pytest.mark.parametrize('order', [2, 3])
def test_cache_identity_round_trip_and_nonfinite_rejection(tmp_path, order):
    path = tmp_path/'operators.npz'
    fields = measurement.FIELDS if order == 2 else measurement.CUBIC_FIELDS
    identity = dict(version=order+2, operator_order=order, fields=list(fields),
                    software={'revision': 'example'})
    arrays = dict(power=np.zeros((len(fields),len(fields),3,15)), k=np.arange(15.), nmodes=np.ones(15),
                  fractions=np.ones(5)/5, identity=json.dumps(identity))
    assert measurement.load_cache(path, identity) is None
    atomic_save(path, **arrays)
    first_mtime = path.stat().st_mtime_ns
    result = measurement.load_cache(path, identity)
    np.testing.assert_array_equal(result['power'], arrays['power'])
    assert path.stat().st_mtime_ns == first_mtime
    assert measurement.load_cache(path, {**identity, 'operator_order': 5-order}) is None
    arrays['power'][0,0,0,0] = np.nan
    atomic_save(path, **arrays)
    with pytest.raises(ValueError, match='invalid cached'):
        measurement.load_cache(path, identity)
    assert not list(tmp_path.glob('*.tmp.npz'))


@pytest.mark.parametrize('field_count', [7, 8])
def test_cache_rejects_wrong_field_dimensions(tmp_path, field_count):
    path = tmp_path/'bad.npz'
    fields = measurement.FIELDS if field_count == 7 else measurement.CUBIC_FIELDS
    identity = dict(version=field_count-3, operator_order=field_count-5, fields=list(fields))
    arrays = dict(power=np.zeros((15-field_count,15-field_count,3,15)), k=np.arange(15.),
                  nmodes=np.ones(15), fractions=np.ones(5)/5, identity=json.dumps(identity))
    atomic_save(path, **arrays)
    with pytest.raises(ValueError, match='invalid cached'):
        measurement.load_cache(path, identity)
    arrays['power'] = np.zeros((field_count,field_count,3,15))
    identity['fields'][0], identity['fields'][1] = identity['fields'][1], identity['fields'][0]
    arrays['identity'] = json.dumps(identity)
    atomic_save(path, **arrays)
    with pytest.raises(ValueError, match='invalid cached'):
        measurement.load_cache(path, identity)


@pytest.mark.parametrize('mesh, precision', [(256, 'float64'), (512, 'float32')])
def test_measure_cache_separation_and_reuse(monkeypatch, tmp_path, mesh, precision):
    source = tmp_path/'snapshot.hdf5'
    source.write_bytes(b'fake source for cache routing')
    header = dict(redshift=.5, boxsize_mpc_h=1000., omega_m=.3175, hubble_param=.6711)
    monkeypatch.setattr(measurement, 'inspect_snapshot', lambda path: ([source], header))
    monkeypatch.setattr(measurement, 'software_metadata', lambda: {'revision': 'test'})
    def no_snapshot(*args, **kwargs):
        pytest.fail('a valid cache should skip snapshot loading')
    monkeypatch.setattr(measurement, 'read_snapshot', no_snapshot)
    cache_loader = measurement.load_cache
    calls = []
    def populate_cache(path, identity):
        calls.append((path, identity))
        n = len(identity['fields'])
        atomic_save(path, power=np.zeros((n,n,3,15)), k=np.arange(15.), nmodes=np.ones(15),
                    fractions=np.ones(5)/5, identity=json.dumps(identity))
        return cache_loader(path, identity)
    monkeypatch.setattr(measurement, 'load_cache', populate_cache)
    for order in (2, 3):
        result = measurement.measure(tmp_path, 10, tmp_path, analysis_mesh=mesh,
                                     precision=precision, operator_order=order)
        assert result['power'].shape == (order+5, order+5, 3, 15)
    assert [path.name for path, _ in calls] == [f'operators_00010_{mesh}_{precision}.npz',
                                               f'operators3_00010_{mesh}_{precision}.npz']
    for path, identity in calls:
        assert identity['version'] == identity['operator_order']+2
        assert identity['selection_mesh'] == 256
        assert identity['analysis_mesh'] == mesh and identity['precision'] == precision
        assert identity['rsd_velocity_conversion'] == 'v_pec = sqrt(a) * stored_velocity'
        assert ('O3' in identity['operator_definitions']) == (identity['operator_order'] == 3)
        assert identity['operator_definitions']['O2'] == '(s**2 - mean(s**2)) / 2'
    monkeypatch.setattr(measurement, 'load_cache', cache_loader)
    mtimes = [path.stat().st_mtime_ns for path, _ in calls]
    for order in (2, 3):
        measurement.measure(tmp_path, 10, tmp_path, analysis_mesh=mesh,
                            precision=precision, operator_order=order)
    assert mtimes == [path.stat().st_mtime_ns for path, _ in calls]
    # Existing quadratic-analysis discovery excludes the cubic file.
    assert list(tmp_path.glob(f'operators_*_{mesh}_{precision}.npz')) == [calls[0][0]]


def test_measure_rejects_invalid_order(tmp_path):
    with pytest.raises(ValueError, match='operator order'):
        measurement.measure(tmp_path, 0, tmp_path, operator_order=4)


def test_cli_defaults_sequential_ids_and_variants(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(measurement, 'measure', lambda *args, **kwargs: calls.append((args, kwargs)))
    common = ['--snapshot-root', str(tmp_path/'snapshots'), '--output-dir', str(tmp_path/'output')]
    measurement.main(common+['--realizations', '0'])
    assert calls == [((tmp_path/'snapshots', 0, tmp_path/'output'),
                      dict(analysis_mesh=256, precision='float64', operator_order=2))]
    calls.clear()
    measurement.main(common+['--realizations','10','1','--analysis-mesh','512',
                             '--precision','float32','--operator-order','3'])
    assert [args[1] for args, _ in calls] == [10, 1]
    assert all(kwargs == dict(analysis_mesh=512, precision='float32', operator_order=3) for _,kwargs in calls)


@pytest.mark.parametrize('arguments', [[], ['--realizations','0'],
    ['--snapshot-root','s','--output-dir','o','--realizations','-1'],
    ['--snapshot-root','s','--output-dir','o','--realizations','0','0'],
    ['--snapshot-root','s','--output-dir','o','--realizations','0','--analysis-mesh','128'],
    ['--snapshot-root','s','--output-dir','o','--realizations','0','--precision','float16'],
    ['--snapshot-root','s','--output-dir','o','--realizations','0','--operator-order','4']])
def test_cli_rejects_invalid_arguments(arguments):
    with pytest.raises(SystemExit) as error:
        measurement.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize('order', [2, 3])
def test_cli_discovers_numeric_directories(monkeypatch, tmp_path, capsys, order):
    for name in ('10', '2', '0', 'notes'):
        (tmp_path/name).mkdir()
    (tmp_path/'3').touch()
    calls = []
    monkeypatch.setattr(measurement, 'measure', lambda *args, **kwargs: calls.append((args, kwargs)))
    measurement.main(['--snapshot-root', str(tmp_path), '--output-dir', str(tmp_path/'output'),
                      '--analysis-mesh', '512', '--precision', 'float32',
                      '--operator-order', str(order)])
    assert [args[1] for args, _ in calls] == [0, 2, 10]
    assert all(args[0] == tmp_path and args[2] == tmp_path/'output' for args, _ in calls)
    assert all(kwargs == dict(analysis_mesh=512, precision='float32', operator_order=order)
               for _, kwargs in calls)
    assert 'Discovered 3 realizations' in capsys.readouterr().out


@pytest.mark.parametrize('missing', [False, True])
def test_cli_discovery_requires_realizations(monkeypatch, tmp_path, missing):
    root = tmp_path/'missing' if missing else tmp_path
    def unexpected_measure(*args, **kwargs):
        pytest.fail('measurement should not run without realizations')
    monkeypatch.setattr(measurement, 'measure', unexpected_measure)
    with pytest.raises(FileNotFoundError, match='no numeric realization directories'):
        measurement.main(['--snapshot-root', str(root), '--output-dir', str(tmp_path/'output')])


@pytest.mark.parametrize('order', [2, 3])
def test_cli_failure_exit_and_no_fitting_imports(tmp_path, order):
    # Block theory dependencies even if the local environment has them installed.
    code = '''
import importlib.abc, runpy, sys
class BlockTheory(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('desilike', 'folps', 'FOLPS', 'scripts.cosmology_inference', 'scripts.validate_')):
            raise RuntimeError('unexpected theory import: '+fullname)
sys.meta_path.insert(0, BlockTheory())
from scripts import measure_density_split_operators as m
m.software_metadata()
def fail(*args, **kwargs):
    raise ValueError('snapshot failure')
m.measure = fail
m.main(['--snapshot-root','missing','--output-dir','unused','--realizations','0',
        '--operator-order',sys.argv[1]])
'''
    result = subprocess.run([sys.executable, '-c', code, str(order)], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'snapshot failure' in result.stderr
    assert 'unexpected theory import' not in result.stderr


def test_software_metadata():
    recorded = measurement.software_metadata()
    assert recorded['versions']['numpy'] == np.__version__
    assert set(recorded['script_sha256']) == {'measure_density_split_operators.py', 'measure_quijote_acm.py'}
    assert all(len(value) == 64 for value in recorded['script_sha256'].values())
