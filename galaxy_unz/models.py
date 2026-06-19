"""
galaxy_unz.models
=================
ModelGrid — loads the Cloudy photoionisation model table, builds an
interpolated 4-D data cube over (log U, log n, Z, line_ratio), and
exposes refined grids for the Bayesian fitter.

The original code scattered grid construction across multiple notebook
cells and global variables (ModelsCube, ModelsCI, u_grid, n_grid,
z_grid).  All of that lives here as instance attributes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# Metallicity grid points in the Cloudy model table (in Z_sun units)
_DEFAULT_Z_POINTS = np.array([0.05, 0.20, 0.40, 1.00, 2.00])

# Column names as they appear in the model CSV after parsing
_MODEL_LOG_COLS = [
    "SIII19_SIII33",
    "OIII52_OIII88",
    "NII122_NII205",
    "SIV11_NeII16",
    "NeII13_NeIII16",
    "OIII52_NIII57",
    "OIII88_NIII57",
    "OIII52_NII122",
    "OIII52_NII205",
    "OIII88_NII122",
    "OIII88_NII205",
]

# Which model columns correspond to each observed ratio (RATIO_NAMES order)
_RATIO_TO_MODEL = [
    "OIII52_OIII88",               # OIII52/OIII88
    "NIII57_OIII88",               # NIII57/OIII88  (derived: 1/OIII88_NIII57)
    "NII122_OIII88",               # NII122/OIII88  (derived: 1/OIII88_NII122)
    "OIII52_NIII57",               # OIII52/NIII57
    "OIII52_NII122",               # OIII52/NII122
    "NIII57_NII122",               # NIII57/NII122  (derived)
    "OIII88_NII122",               # OIII88/NII122
    "OIII88_NIII57",               # OIII88/NIII57
    "(2.2XOIII88+OIII52)/NIII57",  # composite
]

# Map of known column name variants → internal standard names
_COL_ALIASES = {
    "Z/Zo":         "Z",
    "Z/Z_sun":      "Z",
    "z":            "Z",
    "log(n_Hcm-3)": "log(n)",
    "log(nH)":      "log(n)",
    "log(n_H)":     "log(n)",
}


class ModelGrid:
    """
    Photoionisation model grid with optional interpolation.

    Attributes
    ----------
    models : pd.DataFrame
        Raw model table after unit conversion (linear ratios, not log).
    u_grid : np.ndarray
        log(U) axis values used in the refined cube.
    n_grid : np.ndarray
        log(n_e) axis values used in the refined cube.
    z_grid : np.ndarray
        Metallicity (Z/Z_sun) axis values used in the refined cube.
    cube : np.ndarray, shape (n_u, n_n, n_z, n_ratios)
        Interpolated model cube. ``None`` until ``refine()`` is called.

    Notes
    -----
    Call ``ModelGrid.from_file(path)`` to construct.  Then call
    ``.refine(new_shape)`` or ``.refine_by_range(...)`` to build the
    interpolated cube before passing to ``BayesianFitter``.
    """

    def __init__(self, models: pd.DataFrame) -> None:
        self.models: pd.DataFrame = models
        self.cube: np.ndarray | None = None

        # These are populated by _build_raw_cube / refine
        self.u_grid: np.ndarray = np.array([])
        self.n_grid: np.ndarray = np.array([])
        self.z_grid: np.ndarray = np.array([])

        # Build the raw (coarse) cube from the loaded models
        self._raw_cube: np.ndarray | None = None
        self._build_raw_cube()

    # ------------------------------------------------------------------
    # Public constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "ModelGrid":
        """
        Load Cloudy model table from a space-and-ampersand delimited file.

        Handles column name variants automatically:
          'Z/Zo'  or 'Z/Z_sun'  → 'Z'
          'log(n_Hcm-3)'        → 'log(n)'

        Also handles '--' as the negative sign (e.g. '--4.0' → -4.0),
        which is the format used in the Pereira et al. model tables.

        Parameters
        ----------
        path : str
            Path to the model file (e.g. ``data/PereiraModelsSB.txt``).

        Returns
        -------
        ModelGrid
        """
        df = pd.read_csv(path, sep=" & ", header=0, engine="python")

        # Normalise column names to internal standard
        df = df.rename(columns=_COL_ALIASES)

        # Strip trailing LaTeX backslashes from the last column
        last_col = "OIII88_NII205"
        if last_col in df.columns:
            df[last_col] = df[last_col].astype(str).str.replace("\\\\", "", regex=False)

        # Parse all numeric columns.
        # '--' encodes negative values in this file format (e.g. '--4.0' = -4.0)
        all_numeric = ["log(U)", "log(n)", "Z"] + _MODEL_LOG_COLS
        for col in all_numeric:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace("--", "-", regex=False)
                    .pipe(pd.to_numeric, errors="coerce")
                )

        # Convert log-ratios to linear
        for col in _MODEL_LOG_COLS:
            if col in df.columns:
                df[col] = np.power(10.0, df[col])

        # Derived ratios used by the fitter
        df["(2.2XOIII88+OIII52)/NIII57"] = (
            2.2 * df["OIII88_NIII57"] + df["OIII52_NIII57"]
        )
        df["NIII57_NII122"] = df["OIII88_NII122"] / df["OIII88_NIII57"]
        df["U"]             = np.power(10.0, df["log(U)"])
        df["NIII57_OIII88"] = 1.0 / df["OIII88_NIII57"]
        df["NII122_OIII88"] = 1.0 / df["OIII88_NII122"]

        # Print the actual grid ranges so the caller knows what to pass
        # to refine_by_range()
        u_vals = np.sort(df["log(U)"].dropna().unique())
        n_vals = np.sort(df["log(n)"].dropna().unique()) if "log(n)" in df.columns else []
        z_vals = np.sort(df["Z"].dropna().unique()) if "Z" in df.columns else []
        print("ModelGrid loaded:")
        print(f"  log(U)  : {u_vals.min():.2f} → {u_vals.max():.2f}  ({len(u_vals)} values)")
        if len(n_vals):
            print(f"  log(n)  : {n_vals.min():.2f} → {n_vals.max():.2f}  ({len(n_vals)} values: {list(n_vals)})")
        if len(z_vals):
            print(f"  Z/Zsun  : {z_vals.min():.2f} → {z_vals.max():.2f}  ({len(z_vals)} values: {list(z_vals)})")

        return cls(df)

    # ------------------------------------------------------------------
    # Grid refinement (interpolation)
    # ------------------------------------------------------------------

    def refine(self, new_shape: tuple[int, int, int, int]) -> "ModelGrid":
        """
        Interpolate the raw coarse cube to a finer grid.

        Parameters
        ----------
        new_shape : tuple of 4 ints
            Desired output shape (n_u, n_n, n_z, n_ratios).
            The n_ratios dimension must match the number of model
            ratio columns (9).

        Returns
        -------
        self  (allows chaining: ``grid.refine(...).refine_by_range(...)``)
        """
        if self._raw_cube is None:
            raise RuntimeError("Raw cube not built; call from_file() first.")

        dim1, dim2, dim3, dim4 = self._raw_cube.shape

        x = np.linspace(0, dim1 - 1, dim1)
        y = np.linspace(0, dim2 - 1, dim2)
        z = _DEFAULT_Z_POINTS
        n = np.linspace(0, dim4 - 1, dim4)

        interp = RegularGridInterpolator((x, y, z, n), self._raw_cube)

        new_x = np.linspace(0, dim1 - 1, new_shape[0])
        new_y = np.linspace(0, dim2 - 1, new_shape[1])
        new_z = np.round(np.arange(0.05, 2.01, 0.01), decimals=3)
        new_n = np.linspace(0, dim4 - 1, new_shape[3])

        grid_pts = np.meshgrid(new_x, new_y, new_z, new_n, indexing="ij")
        points   = np.column_stack([g.ravel() for g in grid_pts])

        self.cube   = interp(points).reshape(new_shape)
        self.z_grid = new_z
        return self

    def refine_by_range(
        self,
        range_u: tuple[float, float],
        range_n: tuple[float, float],
        range_z: tuple[float, float],
        new_shape: tuple[int, int, int, int],
    ) -> "ModelGrid":
        """
        Interpolate the raw cube onto a user-specified parameter range
        (useful for zooming in after a first-pass fit).

        The ranges must be within the bounds printed by from_file().

        Parameters
        ----------
        range_u : (min, max) for log(U)
        range_n : (min, max) for log(n_e)
        range_z : (min, max) for Z/Z_sun
        new_shape : (n_u, n_n, n_z, n_ratios)

        Returns
        -------
        self
        """
        if self._raw_cube is None:
            raise RuntimeError("Raw cube not built; call from_file() first.")

        dim1, dim2, dim3, dim4 = self._raw_cube.shape

        # Build coordinate axes from the actual grid values
        x_orig = np.linspace(self.u_grid.min(), self.u_grid.max(), dim1)
        y_orig = np.linspace(self.n_grid.min(), self.n_grid.max(), dim2)
        z_orig = np.linspace(self.z_grid.min(), self.z_grid.max(), dim3)
        n_orig = np.linspace(0, dim4 - 1, dim4)

        interp = RegularGridInterpolator(
            (x_orig, y_orig, z_orig, n_orig), self._raw_cube
        )

        new_x = np.round(np.linspace(range_u[0], range_u[1], new_shape[0]), 2)
        new_y = np.round(np.linspace(range_n[0], range_n[1], new_shape[1]), 2)
        new_z = np.round(np.linspace(range_z[0], range_z[1], new_shape[2]), 2)
        new_n = np.round(np.linspace(0, dim4 - 1, new_shape[3]), 2)

        grid_pts = np.meshgrid(new_x, new_y, new_z, new_n, indexing="ij")
        points   = np.column_stack([g.ravel() for g in grid_pts])

        self.cube   = interp(points).reshape(new_shape)
        self.u_grid = new_x
        self.n_grid = new_y
        self.z_grid = new_z
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Shape of the interpolated cube, or None if not yet refined."""
        return self.cube.shape if self.cube is not None else None

    @property
    def n_ratios(self) -> int:
        """Number of model line ratios available."""
        return len(_RATIO_TO_MODEL)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_raw_cube(self) -> None:
        """
        Organise the flat model DataFrame into a 4-D numpy array
        (n_logU, n_logN, n_Z, n_ratios) using the unique parameter
        combinations found in the table.
        """
        df = self.models

        u_vals = np.sort(df["log(U)"].dropna().unique())
        n_vals = np.sort(df["log(n)"].dropna().unique()) if "log(n)" in df.columns else np.array([0.0])
        z_vals = np.sort(df["Z"].dropna().unique()) if "Z" in df.columns else _DEFAULT_Z_POINTS

        ratio_cols = [c for c in _RATIO_TO_MODEL if c in df.columns]

        n_u = len(u_vals)
        n_n = len(n_vals)
        n_z = len(z_vals)
        n_r = len(ratio_cols)

        cube  = np.full((n_u, n_n, n_z, n_r), np.nan)
        u_idx = {v: i for i, v in enumerate(u_vals)}
        n_idx = {v: i for i, v in enumerate(n_vals)}
        z_idx = {round(float(v), 6): i for i, v in enumerate(z_vals)}

        for _, row in df.iterrows():
            ui = u_idx.get(row["log(U)"])
            ni = n_idx.get(row.get("log(n)", 0.0))
            zi = z_idx.get(round(float(row.get("Z", 0.05)), 6))
            if ui is None or ni is None or zi is None:
                continue
            for ri, col in enumerate(ratio_cols):
                cube[ui, ni, zi, ri] = row[col]

        self._raw_cube = cube
        self.u_grid    = u_vals
        self.n_grid    = n_vals
        self.z_grid    = z_vals

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cube_info = (
            f"cube shape={self.shape}" if self.cube is not None
            else "cube=not refined"
        )
        return (
            f"ModelGrid(n_models={len(self.models)}, {cube_info}, "
            f"n_ratios={self.n_ratios})"
        )