import numpy as np

from desilike import compile

from scripts import cosmology_inference as inference


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
