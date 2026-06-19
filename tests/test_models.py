"""
Tests for galaxy_unz.models.ModelGrid

These tests use a small synthetic model table so no real data file is needed.
"""
import numpy as np
import pandas as pd
import pytest
from galaxy_unz.models import ModelGrid, _DEFAULT_Z_POINTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_models(n_u=3, n_n=2) -> pd.DataFrame:
    """
    Build a minimal synthetic Cloudy model table that ModelGrid can digest.
    Values are arbitrary but self-consistent (linear, not log).
    """
    rows = []
    z_vals = _DEFAULT_Z_POINTS
    u_vals = np.linspace(-4.0, -2.0, n_u)
    n_vals = np.linspace(1.0, 3.0, n_n)

    for u in u_vals:
        for n in n_vals:
            for z in z_vals:
                rows.append({
                    "log(U)":  u,
                    "log(n)":  n,
                    "Z":       z,
                    # All log-ratio columns (will be converted to linear in from_file)
                    "SIII19_SIII33":  0.1,
                    "OIII52_OIII88":  0.5 + 0.1 * u,
                    "NII122_NII205":  0.3,
                    "SIV11_NeII16":   0.4,
                    "NeII13_NeIII16": 0.2,
                    "OIII52_NIII57":  0.6 + 0.05 * u,
                    "OIII88_NIII57":  0.7,
                    "OIII52_NII122":  0.8,
                    "OIII52_NII205":  0.9,
                    "OIII88_NII122":  1.0,
                    "OIII88_NII205":  1.1,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ModelGrid construction from a DataFrame (bypassing from_file CSV parsing)
# ---------------------------------------------------------------------------

def test_modelgrid_init_builds_raw_cube():
    df = _make_synthetic_models()
    # Simulate the linear conversion that from_file does
    from galaxy_unz.models import _MODEL_LOG_COLS
    for col in _MODEL_LOG_COLS:
        df[col] = np.power(10.0, df[col])
    df["(2.2XOIII88+OIII52)/NIII57"] = 2.2 * df["OIII88_NIII57"] + df["OIII52_NIII57"]
    df["NIII57_NII122"] = df["OIII88_NII122"] / df["OIII88_NIII57"]
    df["U"]             = np.power(10.0, df["log(U)"])
    df["NIII57_OIII88"] = 1.0 / df["OIII88_NIII57"]
    df["NII122_OIII88"] = 1.0 / df["OIII88_NII122"]

    grid = ModelGrid(df)
    assert grid._raw_cube is not None


def test_modelgrid_n_ratios():
    df = _make_synthetic_models()
    grid = ModelGrid.__new__(ModelGrid)
    grid.models     = df
    grid.cube       = None
    grid.u_grid     = np.array([])
    grid.n_grid     = np.array([])
    grid.z_grid     = np.array([])
    grid._raw_cube  = None
    # n_ratios is a fixed property
    assert grid.n_ratios == 9


def test_modelgrid_repr_no_cube():
    df = _make_synthetic_models()
    grid = ModelGrid.__new__(ModelGrid)
    grid.models     = df
    grid.cube       = None
    grid.u_grid     = np.array([])
    grid.n_grid     = np.array([])
    grid.z_grid     = np.array([])
    grid._raw_cube  = None
    assert "not refined" in repr(grid)


# ---------------------------------------------------------------------------
# Smoke test: shape is None before refine
# ---------------------------------------------------------------------------

def test_shape_none_before_refine():
    df = _make_synthetic_models()
    grid = ModelGrid.__new__(ModelGrid)
    grid.models     = df
    grid.cube       = None
    grid.u_grid     = np.array([])
    grid.n_grid     = np.array([])
    grid.z_grid     = np.array([])
    grid._raw_cube  = None
    assert grid.shape is None
