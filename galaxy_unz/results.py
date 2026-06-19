"""
galaxy_unz.results
==================
FitResult — dataclass holding the full posterior distributions and all
derived statistics for one galaxy fit.

Replaces the scattered output variables from GetSolutions():
  u_50, n_50, z_50, u_ave, n_ave, z_ave, STD, etc.
Everything lives on one object that can be inspected, printed, or passed
to Plotter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FitResult:
    """
    Results of a Bayesian fit for a single galaxy.

    Attributes
    ----------
    galaxy_name : str
        Name of the fitted galaxy.
    galaxy_index : int
        Row index in the DataLoader table.

    -- Grids (1-D axes) --
    u_grid : np.ndarray   log(U) values
    n_grid : np.ndarray   log(n_e) values
    z_grid : np.ndarray   Z/Z_sun values

    -- Full posteriors (3-D cubes, summing to 1) --
    prob : np.ndarray, shape (n_u, n_n, n_z)
        Joint normalised posterior P(U, n, Z | data).
    chi2 : np.ndarray, shape (n_u, n_n, n_z)
        Chi-squared surface.

    -- Marginalised 1-D posteriors --
    p_u : np.ndarray   P(U)   marginalised over n, Z
    p_n : np.ndarray   P(n)   marginalised over U, Z
    p_z : np.ndarray   P(Z)   marginalised over U, n

    -- Best-fit (minimum chi-squared) --
    u_best : float
    n_best : float
    z_best : float

    -- Posterior mean --
    u_mean : float
    n_mean : float
    z_mean : float

    -- Posterior std dev --
    u_std : float
    n_std : float
    z_std : float

    -- 16th / 50th / 84th percentiles --
    u_16, u_50, u_84 : float
    n_16, n_50, n_84 : float
    z_16, z_50, z_84 : float

    -- Pairwise correlation coefficients --
    corr_un : float   ρ(U, n)
    corr_uz : float   ρ(U, Z)
    corr_nz : float   ρ(n, Z)

    -- Inputs used --
    y_obs : np.ndarray   observed ratios
    y_sig : np.ndarray   observed uncertainties (with masked lines = 999999)
    use_lines : list[int] | None
    """

    # Identity
    galaxy_name:  str
    galaxy_index: int

    # Grids
    u_grid: np.ndarray
    n_grid: np.ndarray
    z_grid: np.ndarray

    # Posteriors
    prob: np.ndarray
    chi2: np.ndarray
    p_u:  np.ndarray
    p_n:  np.ndarray
    p_z:  np.ndarray

    # Best-fit
    u_best: float
    n_best: float
    z_best: float

    # Mean
    u_mean: float
    n_mean: float
    z_mean: float

    # Std
    u_std: float
    n_std: float
    z_std: float

    # Percentiles
    u_16: float; u_50: float; u_84: float
    n_16: float; n_50: float; n_84: float
    z_16: float; z_50: float; z_84: float

    # Correlations
    corr_un: float
    corr_uz: float
    corr_nz: float

    # Inputs
    y_obs:     np.ndarray
    y_sig:     np.ndarray
    use_lines: list[int] | None = field(default=None)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def u_err(self) -> tuple[float, float]:
        """Asymmetric 1-sigma uncertainty on log(U): (lower, upper)."""
        return (self.u_50 - self.u_16, self.u_84 - self.u_50)

    @property
    def n_err(self) -> tuple[float, float]:
        """Asymmetric 1-sigma uncertainty on log(n_e): (lower, upper)."""
        return (self.n_50 - self.n_16, self.n_84 - self.n_50)

    @property
    def z_err(self) -> tuple[float, float]:
        """Asymmetric 1-sigma uncertainty on Z/Z_sun: (lower, upper)."""
        return (self.z_50 - self.z_16, self.z_84 - self.z_50)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a formatted summary of the fit results."""
        lo_u, hi_u = self.u_err
        lo_n, hi_n = self.n_err
        lo_z, hi_z = self.z_err
        print(
            f"\n{'='*52}\n"
            f"  Galaxy : {self.galaxy_name}  (index {self.galaxy_index})\n"
            f"{'='*52}\n"
            f"  Parameter     Best-fit    Median     Mean     Std\n"
            f"  {'─'*48}\n"
            f"  log(U)        {self.u_best:+.3f}      {self.u_50:+.3f}    {self.u_mean:+.3f}   {self.u_std:.3f}\n"
            f"  log(n_e)      {self.n_best:+.3f}      {self.n_50:+.3f}    {self.n_mean:+.3f}   {self.n_std:.3f}\n"
            f"  Z/Z_sun       {self.z_best:+.3f}      {self.z_50:+.3f}    {self.z_mean:+.3f}   {self.z_std:.3f}\n"
            f"\n"
            f"  16th / 50th / 84th percentiles\n"
            f"  log(U)   : {self.u_16:.3f}  {self.u_50:.3f}  {self.u_84:.3f}  "
            f"(-{lo_u:.3f} / +{hi_u:.3f})\n"
            f"  log(n_e) : {self.n_16:.3f}  {self.n_50:.3f}  {self.n_84:.3f}  "
            f"(-{lo_n:.3f} / +{hi_n:.3f})\n"
            f"  Z/Z_sun  : {self.z_16:.3f}  {self.z_50:.3f}  {self.z_84:.3f}  "
            f"(-{lo_z:.3f} / +{hi_z:.3f})\n"
            f"\n"
            f"  Correlations  ρ(U,n)={self.corr_un:.3f}  "
            f"ρ(U,Z)={self.corr_uz:.3f}  ρ(n,Z)={self.corr_nz:.3f}\n"
            f"{'='*52}"
        )

    def __repr__(self) -> str:
        return (
            f"FitResult(galaxy='{self.galaxy_name}', "
            f"log(U)={self.u_50:.2f}, "
            f"log(n)={self.n_50:.2f}, "
            f"Z={self.z_50:.2f})"
        )
