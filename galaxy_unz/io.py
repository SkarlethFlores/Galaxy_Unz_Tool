"""
galaxy_unz.io
=============
DataLoader — reads galaxy observation tables, cleans missing values,
adds calibration uncertainties, and computes all line ratios with
propagated errors.

Three classmethods cover the three input formats found in the original
codebase:
  - DataLoader.from_ratios(path)   ← pre-computed ratio CSV
  - DataLoader.from_fluxes(path)   ← raw flux CSV
  - DataLoader.from_fluxes_tf(path)← Thuban-format whitespace file
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sentinel value used to flag missing / unusable measurements
_MISSING = -100.0

# Calibration uncertainty fraction (15 %)
_CAL_FRAC = 0.15

# Flux column names expected in the standard flux-format CSV
_FLUX_COLS = ("OIII52", "OIII52_unc", "OIII88", "OIII88_unc",
              "NIII57", "NIII57_unc", "NII122", "NII122_unc")

# Ratio names and their display labels (used by Plotter)
RATIO_NAMES = [
    "OIII52/OIII88",
    "NIII57/OIII88",
    "NII122/OIII88",
    "OIII52/NIII57",
    "OIII52/NII122",
    "NIII57/NII122",
    "OIII88/NII122",
    "OIII88/NIII57",
    "(2.2OIII88+OIII52)/NIII57",
]

RATIO_LABELS = [
    "[OIII] 52/[OIII] 88",
    "[NIII] 57/[OIII] 88",
    "[NII] 122/[OIII] 88",
    "[OIII] 52/[NIII] 57",
    "[OIII] 52/[NII] 122",
    "[NIII] 57/[NII] 122",
    "[OIII] 88/[NII] 122",
    "[OIII] 88/[NIII] 57",
    "(2.2×[OIII] 88+[OIII] 52)/[NIII] 57",
]


class DataLoader:
    """
    Container for a cleaned galaxy observation table.

    Attributes
    ----------
    galaxies : pd.DataFrame
        Cleaned table with flux columns, ratio columns, and error columns.
    n_galaxies : int
        Number of valid galaxy rows.

    Notes
    -----
    Do not call ``__init__`` directly. Use one of the classmethods:
    ``from_fluxes``, ``from_fluxes_tf``, or ``from_ratios``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.galaxies: pd.DataFrame = df
        self.n_galaxies: int = len(df)

    # ------------------------------------------------------------------
    # Public constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_fluxes(cls, path: str) -> "DataLoader":
        """
        Load from a standard flux CSV (comma-separated, header row 0).

        Expected columns: id, Galaxy, date, OIII52, OIII52_unc,
        OIII88, OIII88_unc, NIII57, NIII57_unc, NII122, NII122_unc,
        reference.

        Parameters
        ----------
        path : str
            Path to the CSV file.

        Returns
        -------
        DataLoader
        """
        df = pd.read_csv(path, header=0, engine="python")
        df = cls._drop_empty_galaxy_rows(df)
        df = cls._coerce_flux_columns(df, _FLUX_COLS)
        df = cls._add_calibration_error(df)
        df = cls._compute_ratios(df)
        return cls(df)

    @classmethod
    def from_fluxes_tf(cls, path: str) -> "DataLoader":
        """
        Load from a Thuban-format whitespace-separated file
        (13 header lines, columns: id Galaxy OIII52 OIII52_unc
        NIII57 NIII57_unc OIII88 OIII88_unc NII122 NII122_unc
        12+log(O/H)_PT2005  12+log(O/H)_I2006).

        Parameters
        ----------
        path : str
            Path to the file.

        Returns
        -------
        DataLoader
        """
        col_names = [
            "id", "Galaxy",
            "OIII52", "OIII52_unc",
            "NIII57", "NIII57_unc",
            "OIII88", "OIII88_unc",
            "NII122", "NII122_unc",
            "12+log(O/H) PT2005",
            "12+log(O/H) I2006",
        ]
        df = pd.read_csv(path, header=13, sep=r"\s+", engine="python",
                         names=col_names)
        df = cls._drop_empty_galaxy_rows(df)
        df = cls._coerce_flux_columns(df, _FLUX_COLS)
        df = cls._add_calibration_error(df)
        df = cls._compute_ratios(df)
        return cls(df)

    @classmethod
    def from_ratios(cls, path: str) -> "DataLoader":
        """
        Load from a pre-computed ratio CSV (columns already contain
        OIII52/OIII88, OIII52/OIII88_ERR, etc.).

        Parameters
        ----------
        path : str
            Path to the CSV file.

        Returns
        -------
        DataLoader
        """
        df = pd.read_csv(path, header=0, engine="python")
        df = cls._drop_empty_galaxy_rows(df)

        # Drop unused line-level columns that appear in the ratio format
        _drop = ["Line", "Wavelength(um)", "Flux1", "Flux1_Error",
                 "SNR1", "Flux2", "Flux2_Error", "SNR2", "Aperture"]
        df = df.drop(columns=[c for c in _drop if c in df.columns])
        df = df.reset_index(drop=True)

        # Coerce numeric columns
        numeric_cols = [c for c in df.columns if c not in ("Galaxy",)]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(_MISSING)

        return cls(df)

    # ------------------------------------------------------------------
    # Per-galaxy accessors
    # ------------------------------------------------------------------

    def get_observations(
        self,
        galaxy_index: int,
        use_lines: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return observed ratios and their uncertainties for one galaxy.

        Parameters
        ----------
        galaxy_index : int
            Row index into ``self.galaxies``.
        use_lines : list of int, optional
            Boolean mask (1 = use, 0 = exclude) of length
            ``len(RATIO_NAMES)``.  Lines flagged 0 get their sigma
            set to a large sentinel so they are effectively ignored
            in the chi-squared calculation.  Defaults to all-ones
            (use every ratio).

        Returns
        -------
        y_obs : np.ndarray, shape (n_ratios,)
        y_sig : np.ndarray, shape (n_ratios,)
        """
        row = self.galaxies.iloc[galaxy_index]

        y_obs = np.array([row[r] for r in RATIO_NAMES], dtype=float)
        y_sig = np.array([row[r + "_ERR"] for r in RATIO_NAMES], dtype=float)

        if use_lines is not None:
            for i, flag in enumerate(use_lines):
                if not flag:
                    y_sig[i] = 999_999.0

        return y_obs, y_sig

    def galaxy_name(self, galaxy_index: int) -> str:
        """Return the Galaxy name string for a given row index."""
        return str(self.galaxies.iloc[galaxy_index]["Galaxy"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_empty_galaxy_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where the Galaxy name is NaN or an empty string."""
        mask = df["Galaxy"].isna() | (df["Galaxy"].astype(str).str.strip() == "nan")
        return df[~mask].reset_index(drop=True)

    @staticmethod
    def _coerce_flux_columns(
        df: pd.DataFrame, flux_cols: tuple[str, ...]
    ) -> pd.DataFrame:
        """
        Convert flux columns to float, replacing NaN / '#DIV/0!' / '0'
        with the missing sentinel value.
        """
        bad_strings = {"nan", "#DIV/0!", "0"}
        for col in flux_cols:
            if col not in df.columns:
                continue
            coerced = []
            for val in df[col]:
                s = str(val).strip()
                if s in bad_strings:
                    coerced.append(_MISSING)
                else:
                    try:
                        coerced.append(float(s))
                    except ValueError:
                        coerced.append(_MISSING)
            df[col] = coerced
        return df

    @staticmethod
    def _add_calibration_error(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 15 % calibration uncertainty in quadrature to each flux
        uncertainty column.

        σ_total = √( σ_stat² + (flux × 0.15)² )
        """
        pairs = [
            ("OIII52", "OIII52_unc"),
            ("OIII88", "OIII88_unc"),
            ("NIII57", "NIII57_unc"),
            ("NII122", "NII122_unc"),
        ]
        for flux_col, unc_col in pairs:
            if flux_col in df.columns and unc_col in df.columns:
                df[unc_col] = np.sqrt(
                    df[unc_col] ** 2 + (df[flux_col] * _CAL_FRAC) ** 2
                )
        return df

    @staticmethod
    def _compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all line ratios and propagate uncertainties.

        Propagation rule for A/B:
            σ(A/B) = (A/B) × √[ (σ_A/A)² + (σ_B/B)² ]
        """
        # --- Ratios ---
        df["OIII52/OIII88"]  = df["OIII52"] / df["OIII88"]
        df["NIII57/OIII88"]  = df["NIII57"] / df["OIII88"]
        df["NII122/OIII88"]  = df["NII122"] / df["OIII88"]
        df["OIII52/NIII57"]  = df["OIII52"] / df["NIII57"]
        df["OIII52/NII122"]  = df["OIII52"] / df["NII122"]
        df["NIII57/NII122"]  = df["NIII57"] / df["NII122"]
        df["OIII88/NII122"]  = df["OIII88"] / df["NII122"]
        df["OIII88/NIII57"]  = df["OIII88"] / df["NIII57"]
        df["(2.2OIII88+OIII52)/NIII57"] = (
            2.2 * df["OIII88"] + df["OIII52"]
        ) / df["NIII57"]

        # --- Propagated uncertainties ---
        def _rel2(col, unc):
            """Squared relative uncertainty, safe against zero."""
            return np.where(
                df[col] != 0,
                (df[unc] / df[col]) ** 2,
                0.0,
            )

        o52_r2  = _rel2("OIII52",  "OIII52_unc")
        o88_r2  = _rel2("OIII88",  "OIII88_unc")
        n57_r2  = _rel2("NIII57",  "NIII57_unc")
        n122_r2 = _rel2("NII122",  "NII122_unc")

        df["OIII52/OIII88_ERR"]  = df["OIII52/OIII88"]  * np.sqrt(o52_r2 + o88_r2)
        df["NIII57/OIII88_ERR"]  = df["NIII57/OIII88"]  * np.sqrt(n57_r2 + o88_r2)
        df["NII122/OIII88_ERR"]  = df["NII122/OIII88"]  * np.sqrt(n122_r2 + o88_r2)
        df["OIII52/NIII57_ERR"]  = df["OIII52/NIII57"]  * np.sqrt(o52_r2 + n57_r2)
        df["OIII52/NII122_ERR"]  = df["OIII52/NII122"]  * np.sqrt(o52_r2 + n122_r2)
        df["NIII57/NII122_ERR"]  = df["NIII57/NII122"]  * np.sqrt(n57_r2 + n122_r2)
        df["OIII88/NII122_ERR"]  = df["OIII88/NII122"]  * np.sqrt(o88_r2 + n122_r2)
        df["OIII88/NIII57_ERR"]  = df["OIII88/NIII57"]  * np.sqrt(o88_r2 + n57_r2)
        df["(2.2OIII88+OIII52)/NIII57_ERR"] = (
            df["(2.2OIII88+OIII52)/NIII57"]
            * np.sqrt(o88_r2 + o52_r2 + n57_r2)
        )

        return df

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_galaxies

    def __repr__(self) -> str:
        return (
            f"DataLoader(n_galaxies={self.n_galaxies}, "
            f"columns={list(self.galaxies.columns[:6])} ...)"
        )
