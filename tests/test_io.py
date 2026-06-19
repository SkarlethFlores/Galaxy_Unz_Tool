"""
Tests for galaxy_unz.io.DataLoader
"""
import io
import numpy as np
import pandas as pd
import pytest
from galaxy_unz.io import DataLoader, RATIO_NAMES, _CAL_FRAC, _MISSING


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_flux_df(**overrides) -> pd.DataFrame:
    """Return a minimal one-row flux DataFrame for testing."""
    base = dict(
        id=1, Galaxy="TestGal", date="2020",
        OIII52=100.0, OIII52_unc=5.0,
        OIII88=200.0, OIII88_unc=8.0,
        NIII57=50.0,  NIII57_unc=3.0,
        NII122=30.0,  NII122_unc=2.0,
        reference="test",
    )
    base.update(overrides)
    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# DataLoader._drop_empty_galaxy_rows
# ---------------------------------------------------------------------------

def test_drop_empty_galaxy_rows_removes_nan():
    df = pd.DataFrame({"Galaxy": ["GalA", float("nan"), "GalB"]})
    result = DataLoader._drop_empty_galaxy_rows(df)
    assert len(result) == 2
    assert "GalA" in result["Galaxy"].values


# ---------------------------------------------------------------------------
# DataLoader._coerce_flux_columns
# ---------------------------------------------------------------------------

def test_coerce_flux_columns_replaces_bad_strings():
    df = pd.DataFrame({
        "OIII52":     ["100.0", "nan",      "#DIV/0!", "0"],
        "OIII52_unc": ["5.0",   "#DIV/0!",  "3.0",    "nan"],
    })
    result = DataLoader._coerce_flux_columns(df, ("OIII52", "OIII52_unc"))
    assert result["OIII52"].iloc[0] == 100.0
    assert result["OIII52"].iloc[1] == _MISSING
    assert result["OIII52"].iloc[2] == _MISSING
    assert result["OIII52"].iloc[3] == _MISSING


# ---------------------------------------------------------------------------
# DataLoader._add_calibration_error
# ---------------------------------------------------------------------------

def test_add_calibration_error_quadrature():
    df = _make_flux_df()
    # Record original values
    flux    = df["OIII52"].iloc[0]
    stat_uc = df["OIII52_unc"].iloc[0]
    expected = np.sqrt(stat_uc**2 + (flux * _CAL_FRAC)**2)

    result = DataLoader._add_calibration_error(df.copy())
    assert abs(result["OIII52_unc"].iloc[0] - expected) < 1e-10


# ---------------------------------------------------------------------------
# DataLoader._compute_ratios — values
# ---------------------------------------------------------------------------

def test_compute_ratios_basic_values():
    df = _make_flux_df()
    df = DataLoader._add_calibration_error(df)
    df = DataLoader._compute_ratios(df)

    expected_ratio = df["OIII52"].iloc[0] / df["OIII88"].iloc[0]
    assert abs(df["OIII52/OIII88"].iloc[0] - expected_ratio) < 1e-10


def test_compute_ratios_all_ratio_names_present():
    df = _make_flux_df()
    df = DataLoader._add_calibration_error(df)
    df = DataLoader._compute_ratios(df)
    for name in RATIO_NAMES:
        assert name in df.columns, f"Missing ratio column: {name}"
    for name in RATIO_NAMES:
        assert (name + "_ERR") in df.columns, f"Missing error column: {name}_ERR"


# ---------------------------------------------------------------------------
# DataLoader.get_observations
# ---------------------------------------------------------------------------

def test_get_observations_shape():
    df = _make_flux_df()
    df = DataLoader._drop_empty_galaxy_rows(df)
    df = DataLoader._coerce_flux_columns(df, (
        "OIII52", "OIII52_unc", "OIII88", "OIII88_unc",
        "NIII57", "NIII57_unc", "NII122", "NII122_unc",
    ))
    df = DataLoader._add_calibration_error(df)
    df = DataLoader._compute_ratios(df)
    loader = DataLoader(df)

    y_obs, y_sig = loader.get_observations(0)
    assert y_obs.shape == (len(RATIO_NAMES),)
    assert y_sig.shape == (len(RATIO_NAMES),)


def test_get_observations_use_lines_masks():
    df = _make_flux_df()
    df = DataLoader._add_calibration_error(df)
    df = DataLoader._compute_ratios(df)
    loader = DataLoader(df)

    use = [1, 0, 1, 0, 1, 0, 1, 0, 1]
    _, y_sig = loader.get_observations(0, use_lines=use)
    # Lines flagged 0 should have their sigma set to sentinel
    for i, flag in enumerate(use):
        if flag == 0:
            assert y_sig[i] == 999_999.0


# ---------------------------------------------------------------------------
# DataLoader.__len__ and __repr__
# ---------------------------------------------------------------------------

def test_len():
    df = _make_flux_df()
    df = DataLoader._add_calibration_error(df)
    df = DataLoader._compute_ratios(df)
    loader = DataLoader(df)
    assert len(loader) == 1


def test_repr_contains_n_galaxies():
    df = _make_flux_df()
    df = DataLoader._add_calibration_error(df)
    df = DataLoader._compute_ratios(df)
    loader = DataLoader(df)
    assert "n_galaxies=1" in repr(loader)
