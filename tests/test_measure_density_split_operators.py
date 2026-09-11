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


def test_native_jaxpower_normalization():
    import jax
    jax.config.update('jax_enable_x64', True)
    from jaxpower import MeshAttrs, RealMeshField, BinMesh2SpectrumPoles, compute_mesh2_spectrum
    n, length = 16, 100.
    edges = np.array([0., .09, .15])
    attrs = MeshAttrs(meshsize=n, boxsize=length)
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(2, n, n, n))
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


def test_signed_painting_and_partition():
    import jax
    jax.config.update('jax_enable_x64', True)
    from jaxpower import MeshAttrs, ParticleField
    n = 8
    attrs = MeshAttrs(meshsize=n, boxsize=80.)
    coords = attrs.rcoords()
    positions = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(7)
    positions = positions[rng.permutation(len(positions))]
    s = np.cos(2*np.pi*positions[:, 0]/80.)
    o2 = .5*(s*s-np.mean(s*s))

    def paint(weights):
        return np.asarray(ParticleField(positions, weights=weights, attrs=attrs, backend='jax').paint(
            resampler='tsc', compensate=True, interlacing=0, out='real'))

    parent = paint(np.ones(len(s)))
    np.testing.assert_allclose(parent, parent.mean(), atol=1.e-12)
    np.testing.assert_allclose(paint(1.3*s-.7*o2), 1.3*paint(s)-.7*paint(o2), atol=1.e-12)
    labels = np.arange(len(s)) % 5
    fractions = np.bincount(labels)/len(labels)
    partition = sum(fractions[q]*(paint((labels==q).astype(float))/fractions[q]/parent.mean()-1.) for q in range(5))
    np.testing.assert_allclose(partition, 0., atol=1.e-12)


def test_cache_identity_round_trip_and_nonfinite_rejection(tmp_path):
    path = tmp_path/'operators.npz'
    identity = {'version': 3, 'software': {'revision': 'example'}}
    arrays = dict(power=np.zeros((7,7,3,15)), k=np.arange(15.), nmodes=np.ones(15),
                  fractions=np.ones(5)/5, identity=json.dumps(identity))
    assert measurement.load_cache(path, identity) is None
    atomic_save(path, **arrays)
    first_mtime = path.stat().st_mtime_ns
    result = measurement.load_cache(path, identity)
    np.testing.assert_array_equal(result['power'], arrays['power'])
    assert path.stat().st_mtime_ns == first_mtime
    assert measurement.load_cache(path, {**identity, 'version': 4}) is None
    arrays['power'][0,0,0,0] = np.nan
    atomic_save(path, **arrays)
    with pytest.raises(ValueError, match='invalid cached'):
        measurement.load_cache(path, identity)
    assert not list(tmp_path.glob('*.tmp.npz'))


def test_cli_defaults_sequential_ids_and_variants(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(measurement, 'measure', lambda *args, **kwargs: calls.append((args, kwargs)))
    common = ['--snapshot-root', str(tmp_path/'snapshots'), '--output-dir', str(tmp_path/'output')]
    measurement.main(common+['--realizations', '0'])
    assert calls == [((tmp_path/'snapshots', 0, tmp_path/'output'),
                      dict(analysis_mesh=256, precision='float64'))]
    calls.clear()
    measurement.main(common+['--realizations','10','1','--analysis-mesh','512','--precision','float32'])
    assert [args[1] for args, _ in calls] == [10, 1]
    assert all(kwargs == dict(analysis_mesh=512, precision='float32') for _,kwargs in calls)


@pytest.mark.parametrize('arguments', [[], ['--realizations','0'],
    ['--snapshot-root','s','--output-dir','o'],
    ['--snapshot-root','s','--output-dir','o','--realizations','-1'],
    ['--snapshot-root','s','--output-dir','o','--realizations','0','0'],
    ['--snapshot-root','s','--output-dir','o','--realizations','0','--analysis-mesh','128'],
    ['--snapshot-root','s','--output-dir','o','--realizations','0','--precision','float16']])
def test_cli_rejects_invalid_arguments(arguments):
    with pytest.raises(SystemExit) as error:
        measurement.main(arguments)
    assert error.value.code == 2


def test_cli_failure_exit_and_no_fitting_imports(tmp_path):
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
m.main(['--snapshot-root','missing','--output-dir','unused','--realizations','0'])
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'snapshot failure' in result.stderr
    assert 'unexpected theory import' not in result.stderr


def test_software_metadata():
    recorded = measurement.software_metadata()
    assert recorded['versions']['numpy'] == np.__version__
    assert set(recorded['script_sha256']) == {'measure_density_split_operators.py', 'measure_quijote_acm.py'}
    assert all(len(value) == 64 for value in recorded['script_sha256'].values())
