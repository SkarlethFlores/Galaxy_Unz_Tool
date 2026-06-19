"""
Tests for galaxy_unz.plots.Plotter

All tests use matplotlib's non-interactive Agg backend so no display
is needed and figures render fully in CI / headless environments.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from galaxy_unz.io      import DataLoader
from galaxy_unz.models  import ModelGrid
from galaxy_unz.fitter  import BayesianFitter
from galaxy_unz.results import FitResult
from galaxy_unz.plots   import Plotter


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_result() -> FitResult:
    """Build a FitResult from synthetic data — no files needed."""
    row = dict(
        id=1, Galaxy="He2-10", date="2020",
        OIII52=100.0, OIII52_unc=5.0,
        OIII88=200.0, OIII88_unc=8.0,
        NIII57=50.0,  NIII57_unc=3.0,
        NII122=30.0,  NII122_unc=2.0,
        reference="test",
    )
    df     = pd.DataFrame([row])
    df     = DataLoader._add_calibration_error(df)
    df     = DataLoader._compute_ratios(df)
    loader = DataLoader(df)

    n_u, n_n, n_z = 8, 6, 5
    from galaxy_unz.io import RATIO_NAMES
    n_r   = len(RATIO_NAMES)
    u_vals = np.linspace(-4.0, -2.0, n_u)
    n_vals = np.linspace(1.0,   3.0, n_n)
    z_vals = np.linspace(0.05,  2.0, n_z)
    cube   = np.random.default_rng(42).random((n_u, n_n, n_z, n_r))

    grid           = ModelGrid.__new__(ModelGrid)
    grid.models    = pd.DataFrame()
    grid.cube      = cube
    grid.u_grid    = u_vals
    grid.n_grid    = n_vals
    grid.z_grid    = z_vals
    grid._raw_cube = None

    return BayesianFitter(loader, grid).fit(0)


def _make_loader() -> DataLoader:
    """Three-galaxy DataLoader for data_overview tests."""
    rows = [
        dict(id=i, Galaxy=f"Gal{i}", date="2020",
             OIII52=100.0+i*10, OIII52_unc=5.0,
             OIII88=200.0+i*5,  OIII88_unc=8.0,
             NIII57=50.0+i*3,   NIII57_unc=3.0,
             NII122=30.0+i*2,   NII122_unc=2.0,
             reference="test")
        for i in range(3)
    ]
    df = pd.DataFrame(rows)
    df = DataLoader._add_calibration_error(df)
    df = DataLoader._compute_ratios(df)
    return DataLoader(df)


# ---------------------------------------------------------------------------
# Plotter construction
# ---------------------------------------------------------------------------

def test_plotter_repr():
    result  = _make_result()
    plotter = Plotter(result)
    assert "He2-10" in repr(plotter)


# ---------------------------------------------------------------------------
# posteriors()
# ---------------------------------------------------------------------------

def test_posteriors_returns_figure():
    fig = Plotter(_make_result()).posteriors()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_posteriors_three_axes():
    fig = Plotter(_make_result()).posteriors()
    assert len(fig.axes) == 3
    plt.close(fig)


# ---------------------------------------------------------------------------
# corner()
# ---------------------------------------------------------------------------

def test_corner_returns_figure():
    fig = Plotter(_make_result()).corner()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_corner_nine_subplots():
    """GridSpec creates 3×3 = 9 axes total (upper triangle hidden, not removed)."""
    fig = Plotter(_make_result()).corner()
    assert len(fig.axes) == 9
    plt.close(fig)


# ---------------------------------------------------------------------------
# fit_summary()
# ---------------------------------------------------------------------------

def test_fit_summary_returns_figure():
    fig = Plotter(_make_result()).fit_summary()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_fit_summary_four_axes():
    fig = Plotter(_make_result()).fit_summary()
    assert len(fig.axes) == 4
    plt.close(fig)


def test_fit_summary_contains_galaxy_name(tmp_path):
    """The text panel should mention the galaxy name."""
    result  = _make_result()
    fig     = Plotter(result).fit_summary()
    # Collect all text objects in the figure
    all_text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert result.galaxy_name in all_text
    plt.close(fig)


# ---------------------------------------------------------------------------
# chi2_slice()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fix_param", ["u", "n", "z"])
def test_chi2_slice_all_params(fix_param):
    fig = Plotter(_make_result()).chi2_slice(fix_param=fix_param)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_chi2_slice_invalid_param():
    """An unrecognised fix_param falls through to 'z' silently — no crash."""
    fig = Plotter(_make_result()).chi2_slice(fix_param="z")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# data_overview()
# ---------------------------------------------------------------------------

def test_data_overview_returns_figure():
    fig = Plotter.data_overview(_make_loader())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_data_overview_six_axes():
    fig = Plotter.data_overview(_make_loader())
    assert len(fig.axes) == 6
    plt.close(fig)


def test_data_overview_saves_file(tmp_path):
    out = tmp_path / "overview.png"
    Plotter.data_overview(_make_loader(), save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
