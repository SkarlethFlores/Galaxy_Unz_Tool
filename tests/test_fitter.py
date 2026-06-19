"""
Tests for galaxy_unz.fitter.BayesianFitter
       and galaxy_unz.results.FitResult
"""
import numpy as np
import pandas as pd
import pytest

from galaxy_unz.io      import DataLoader, RATIO_NAMES
from galaxy_unz.models  import ModelGrid, _DEFAULT_Z_POINTS
from galaxy_unz.fitter  import BayesianFitter
from galaxy_unz.results import FitResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_loader(flux=100.0) -> DataLoader:
    """One-galaxy DataLoader with known flux values."""
    row = dict(
        id=1, Galaxy="NGC_test", date="2020",
        OIII52=flux,       OIII52_unc=flux * 0.05,
        OIII88=flux * 2,   OIII88_unc=flux * 0.04,
        NIII57=flux * 0.5, NIII57_unc=flux * 0.06,
        NII122=flux * 0.3, NII122_unc=flux * 0.07,
        reference="test",
    )
    df = pd.DataFrame([row])
    df = DataLoader._add_calibration_error(df)
    df = DataLoader._compute_ratios(df)
    return DataLoader(df)


def _make_grid(n_u=6, n_n=4, n_z=5) -> ModelGrid:
    """
    Synthetic ModelGrid with a known cube.
    cube[i, j, k, r] = u_vals[i] + n_vals[j] + z_vals[k] + r*0.1
    (arbitrary, but deterministic — lets us check that the fitter
    finds sensible best-fit indices).
    """
    u_vals = np.linspace(-4.0, -2.0, n_u)
    n_vals = np.linspace(1.0,   3.0, n_n)
    z_vals = np.linspace(0.05,  2.0, n_z)
    n_r    = len(RATIO_NAMES)

    cube = np.zeros((n_u, n_n, n_z, n_r))
    for i, u in enumerate(u_vals):
        for j, n in enumerate(n_vals):
            for k, z in enumerate(z_vals):
                for r in range(n_r):
                    cube[i, j, k, r] = u + n + z + r * 0.1

    grid         = ModelGrid.__new__(ModelGrid)
    grid.models  = pd.DataFrame()
    grid.cube    = cube
    grid.u_grid  = u_vals
    grid.n_grid  = n_vals
    grid.z_grid  = z_vals
    grid._raw_cube = None
    return grid


# ---------------------------------------------------------------------------
# BayesianFitter construction
# ---------------------------------------------------------------------------

def test_fitter_requires_refined_grid():
    loader = _make_loader()
    grid   = ModelGrid.__new__(ModelGrid)
    grid.cube = None
    with pytest.raises(ValueError, match="not been refined"):
        BayesianFitter(loader, grid)


def test_fitter_repr():
    loader = _make_loader()
    grid   = _make_grid()
    fitter = BayesianFitter(loader, grid)
    assert "BayesianFitter" in repr(fitter)
    assert "n_galaxies=1"   in repr(fitter)


# ---------------------------------------------------------------------------
# _compute_likelihood
# ---------------------------------------------------------------------------

def test_likelihood_prob_sums_to_one():
    loader = _make_loader()
    grid   = _make_grid()
    fitter = BayesianFitter(loader, grid)

    y_obs, y_sig = loader.get_observations(0)
    _, prob = fitter._compute_likelihood(y_obs, y_sig)

    assert abs(prob.sum() - 1.0) < 1e-10


def test_likelihood_prob_shape():
    loader = _make_loader()
    grid   = _make_grid(n_u=6, n_n=4, n_z=5)
    fitter = BayesianFitter(loader, grid)

    y_obs, y_sig = loader.get_observations(0)
    chi2, prob   = fitter._compute_likelihood(y_obs, y_sig)

    assert chi2.shape == (6, 4, 5)
    assert prob.shape == (6, 4, 5)


def test_likelihood_chi2_non_negative():
    loader = _make_loader()
    grid   = _make_grid()
    fitter = BayesianFitter(loader, grid)

    y_obs, y_sig = loader.get_observations(0)
    chi2, _      = fitter._compute_likelihood(y_obs, y_sig)
    assert np.all(chi2 >= 0)


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------

def test_percentile_median_uniform():
    """Median of a uniform distribution on [0,1] should be ~0.5."""
    x  = np.linspace(0, 1, 100)
    px = np.ones(100) / 100
    result = BayesianFitter._percentile(x, px, 50)
    assert abs(result - 0.5) < 0.02


def test_percentile_boundaries():
    x  = np.linspace(0, 1, 10)
    px = np.ones(10) / 10
    assert BayesianFitter._percentile(x, px, 0)   == pytest.approx(x[0],  abs=0.01)
    assert BayesianFitter._percentile(x, px, 100) == pytest.approx(x[-1], abs=0.01)


def test_percentile_16_84_span():
    """16th and 84th percentiles should bracket most of the mass."""
    x  = np.linspace(-3, 3, 200)
    px = np.exp(-x**2 / 2)
    px /= px.sum()
    p16 = BayesianFitter._percentile(x, px, 16)
    p84 = BayesianFitter._percentile(x, px, 84)
    assert p16 < 0
    assert p84 > 0
    assert p84 > p16


# ---------------------------------------------------------------------------
# _correlation
# ---------------------------------------------------------------------------

def test_correlation_perfect():
    """
    If prob is concentrated on the diagonal of a 2-D grid (U == Z),
    the U-Z correlation should be close to +1.
    """
    n = 10
    u = np.linspace(-4, -2, n)
    nn = np.array([2.0])
    z  = np.linspace(0.1, 2.0, n)

    prob = np.zeros((n, 1, n))
    for i in range(n):
        prob[i, 0, i] = 1.0
    prob /= prob.sum()

    means = (float(np.sum(u * prob.sum(axis=(1,2)))),
             2.0,
             float(np.sum(z * prob.sum(axis=(0,1)))))
    p_u = prob.sum(axis=(1,2))
    p_z = prob.sum(axis=(0,1))
    stds  = (float(np.sqrt(np.sum((u - means[0])**2 * p_u))),
             0.0,
             float(np.sqrt(np.sum((z - means[2])**2 * p_z))))

    corr = BayesianFitter._correlation(0, 2, prob, means, stds, u, nn, z)
    assert abs(corr - 1.0) < 0.05


def test_correlation_zero_std_returns_zero():
    """If one parameter has zero std, correlation is undefined → return 0."""
    n = 5
    u = np.linspace(-4, -2, n)
    nn = np.array([2.0])
    z  = np.linspace(0.1, 2.0, n)
    prob = np.ones((n, 1, n)) / (n * n)

    means = (float(u.mean()), 2.0, float(z.mean()))
    stds  = (0.0, 0.0, float(z.std()))   # u_std = 0 → should return 0

    corr = BayesianFitter._correlation(0, 2, prob, means, stds, u, nn, z)
    assert corr == 0.0


# ---------------------------------------------------------------------------
# Full fit() — integration test
# ---------------------------------------------------------------------------

def test_fit_returns_fitresult():
    loader = _make_loader()
    grid   = _make_grid()
    result = BayesianFitter(loader, grid).fit(0)
    assert isinstance(result, FitResult)


def test_fit_galaxy_name():
    loader = _make_loader()
    grid   = _make_grid()
    result = BayesianFitter(loader, grid).fit(0)
    assert result.galaxy_name == "NGC_test"


def test_fit_posteriors_sum_to_one():
    loader = _make_loader()
    grid   = _make_grid()
    result = BayesianFitter(loader, grid).fit(0)
    assert abs(result.prob.sum() - 1.0) < 1e-10
    assert abs(result.p_u.sum()  - 1.0) < 1e-10
    assert abs(result.p_n.sum()  - 1.0) < 1e-10
    assert abs(result.p_z.sum()  - 1.0) < 1e-10


def test_fit_percentile_ordering():
    loader = _make_loader()
    grid   = _make_grid()
    result = BayesianFitter(loader, grid).fit(0)
    assert result.u_16 <= result.u_50 <= result.u_84
    assert result.n_16 <= result.n_50 <= result.n_84
    assert result.z_16 <= result.z_50 <= result.z_84


def test_fit_use_lines_mask():
    """Masking all but one line should still produce a valid result."""
    loader   = _make_loader()
    grid     = _make_grid()
    use_only = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    result   = BayesianFitter(loader, grid).fit(0, use_lines=use_only)
    assert abs(result.prob.sum() - 1.0) < 1e-10


def test_fit_all_length():
    loader = _make_loader()
    grid   = _make_grid()
    results = BayesianFitter(loader, grid).fit_all()
    assert len(results) == 1


# ---------------------------------------------------------------------------
# FitResult helpers
# ---------------------------------------------------------------------------

def test_fitresult_err_properties():
    loader = _make_loader()
    grid   = _make_grid()
    result = BayesianFitter(loader, grid).fit(0)
    lo_u, hi_u = result.u_err
    assert lo_u >= 0
    assert hi_u >= 0


def test_fitresult_summary_runs(capsys):
    loader = _make_loader()
    grid   = _make_grid()
    result = BayesianFitter(loader, grid).fit(0)
    result.summary()
    out = capsys.readouterr().out
    assert "NGC_test" in out
    assert "log(U)"   in out


def test_fitresult_repr():
    loader = _make_loader()
    grid   = _make_grid()
    result = BayesianFitter(loader, grid).fit(0)
    assert "FitResult" in repr(result)
    assert "NGC_test"  in repr(result)
