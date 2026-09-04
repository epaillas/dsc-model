import numpy as np
import jax.numpy as jnp
import pytest

from desilike import Calculator, Parameter, compile

from scripts import cosmology_inference as inference
from scripts import emulation


class ToyMatter(Calculator):

    def __init__(self):
        self.amplitude = Parameter('amplitude', value=1., fixed=False)
        self.k = np.array([0.02, 0.04])
        self.ells = (0,)
        self.z = 0.5
        self.rsd = False

    def __call__(self):
        self.power = self.amplitude.value**2 * jnp.asarray([[1., 2.]])
        self.poles = self.power
        return self.power

    def tree_flatten(self):
        return [self.power], dict(k=self.k, ells=self.ells, z=self.z, rsd=self.rsd)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.power = obj.poles = children[0]
        for name, value in aux.items():
            setattr(obj, name, value)
        return obj


def test_direct_matter_and_density_split_graphs():
    k = np.array([0.02, 0.04, 0.08])
    pmm = inference.build_pmm_theory(
        k, redshift=0.5, ells=(0, 2), rsd=True,
        cosmo_engine='eisenstein_hu')
    pqm = inference.build_pqm_theory(
        k, redshift=0.5, ells=(0, 2), quantiles=(1, 5), rsd=True,
        cosmo_engine='eisenstein_hu')

    matter = np.asarray(compile(pmm)())
    split = np.asarray(compile(pqm)({'c1q1': -1., 'c1q5': 2.}))
    assert matter.shape == (2, 3)
    assert split.shape == (2, 2, 3)
    assert np.isfinite(matter).all()
    assert np.isfinite(split).all()


def test_direct_ap_graphs():
    k = np.array([0.02, 0.04, 0.08])
    pmm = inference.build_pmm_theory(
        k, redshift=0.5, ells=(0, 2), rsd=True, ap=True,
        cosmo_engine='eisenstein_hu')
    pqm = inference.build_pqm_theory(
        k, redshift=0.5, ells=(0, 2), quantiles=(1, 5), rsd=True, ap=True,
        cosmo_engine='eisenstein_hu')

    assert np.asarray(compile(pmm)()).shape == (2, 3)
    assert np.asarray(compile(pqm)()).shape == (2, 2, 3)


def test_ace_cli_selects_direct_cosmology():
    args = inference.parse_args([
        '--input-root', '/tmp/input', '--output-dir', '/tmp/output',
        '--cosmo-engine', 'ace',
    ])
    assert args.cosmo_engine == 'ace'
    assert args.direct_cosmology


def test_mlp_cli_fails_clearly(tmp_path, capsys):
    with pytest.raises(SystemExit):
        inference.parse_args([
            '--input-root', '/tmp/input', '--output-dir', str(tmp_path),
            '--emulator', 'mlp',
        ])
    assert 'MLP emulation is not yet ported' in capsys.readouterr().err


def test_taylor_hdf5_cache_round_trip_and_legacy_replacement(tmp_path):
    options = emulation.matter_cache_options(
        [0.02, 0.04], redshift=0.5, ells=(0,), rsd=False,
        order=2, accuracy=2, method='finite')
    path = emulation.emulator_path(tmp_path, options)
    legacy = path.with_suffix('.npy')
    legacy.write_bytes(b'legacy cache')

    builds = 0

    def build():
        nonlocal builds
        builds += 1
        return ToyMatter()

    first = emulation.get_or_train_emulator(tmp_path, options, build)
    first_run = compile(first)
    expected = np.asarray(first_run({'amplitude': 1.2}))
    assert path.suffix == '.h5'
    assert path.is_file()
    assert not legacy.exists()

    second = emulation.get_or_train_emulator(
        tmp_path, options, lambda: pytest.fail('cache should be reused'))
    actual = np.asarray(compile(second)({'amplitude': 1.2}))
    np.testing.assert_allclose(actual, expected)
    assert builds == 1


def test_corrupt_taylor_cache_reports_path(tmp_path):
    options = emulation.matter_cache_options(
        [0.02], redshift=0.5, ells=(0,), rsd=False)
    path = emulation.emulator_path(tmp_path, options)
    path.write_bytes(b'not hdf5')
    with pytest.raises(RuntimeError, match=str(path)):
        emulation.get_or_train_emulator(
            tmp_path, options, lambda: pytest.fail('must not retrain corrupt cache'))
