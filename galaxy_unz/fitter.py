"""
galaxy_unz.fitter
=================
BayesianFitter — computes the chi-squared likelihood over the full
(log U, log n, Z) model cube and returns a FitResult.

This replaces and consolidates:
  CalcChi2 / CalcChi2_N  →  _compute_likelihood()
  PercentileXPx           →  _percentile()
  CorrCoef                →  _correlation()
  GetSolutions            →  _build_result()

No global variables.  Every quantity that was previously a notebook-level
global (Y_obs, Y_sig, ModelsCI, prob, u_grid, …) is now a local variable
inside fit() or a private helper method.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from .io      import DataLoader
from .models  import ModelGrid
from .results import FitResult


class BayesianFitter:
    """
    Bayesian parameter fitter over a photoionisation model grid.

    Parameters
    ----------
    data : DataLoader
        Cleaned observation table produced by DataLoader.from_fluxes() etc.
    grid : ModelGrid
        Refined model cube produced by ModelGrid.from_file().refine_by_range().

    Example
    -------
    >>> fitter = BayesianFitter(data, grid)
    >>> result = fitter.fit(galaxy_index=0)
    >>> result.summary()
    """

    def __init__(self, data: DataLoader, grid: ModelGrid) -> None:
        if grid.cube is None:
            raise ValueError(
                "ModelGrid has not been refined yet. "
                "Call grid.refine() or grid.refine_by_range() first."
            )
        self.data = data
        self.grid = grid

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        galaxy_index: int,
        use_lines: list[int] | None = None,
    ) -> FitResult:
        """
        Fit one galaxy and return a FitResult.

        Parameters
        ----------
        galaxy_index : int
            Row index into DataLoader.galaxies.
        use_lines : list of int, optional
            Boolean mask (1 = use, 0 = ignore) of length n_ratios.
            Defaults to all ones (use every ratio).

        Returns
        -------
        FitResult
        """
        # --- 1. Observations ---
        y_obs, y_sig = self.data.get_observations(galaxy_index, use_lines)
        name         = self.data.galaxy_name(galaxy_index)

        # --- 2. Likelihood ---
        chi2, prob = self._compute_likelihood(y_obs, y_sig)

        # --- 3. Marginalise ---
        p_u = prob.sum(axis=(1, 2))   # sum over n, Z  → shape (n_u,)
        p_n = prob.sum(axis=(0, 2))   # sum over U, Z  → shape (n_n,)
        p_z = prob.sum(axis=(0, 1))   # sum over U, n  → shape (n_z,)

        u_grid = self.grid.u_grid
        n_grid = self.grid.n_grid
        z_grid = self.grid.z_grid

        # --- 4. Best-fit (minimum chi-squared) ---
        idx_best      = np.unravel_index(chi2.argmin(), chi2.shape)
        u_best = float(u_grid[idx_best[0]])
        n_best = float(n_grid[idx_best[1]])
        z_best = float(z_grid[idx_best[2]])

        # --- 5. Posterior mean and std ---
        u_mean = float(np.sum(u_grid * p_u))
        n_mean = float(np.sum(n_grid * p_n))
        z_mean = float(np.sum(z_grid * p_z))

        u_std = float(np.sqrt(np.sum((u_grid - u_mean) ** 2 * p_u)))
        n_std = float(np.sqrt(np.sum((n_grid - n_mean) ** 2 * p_n)))
        z_std = float(np.sqrt(np.sum((z_grid - z_mean) ** 2 * p_z)))

        # --- 6. Percentiles ---
        u_16 = self._percentile(u_grid, p_u, 16)
        u_50 = self._percentile(u_grid, p_u, 50)
        u_84 = self._percentile(u_grid, p_u, 84)

        n_16 = self._percentile(n_grid, p_n, 16)
        n_50 = self._percentile(n_grid, p_n, 50)
        n_84 = self._percentile(n_grid, p_n, 84)

        z_16 = self._percentile(z_grid, p_z, 16)
        z_50 = self._percentile(z_grid, p_z, 50)
        z_84 = self._percentile(z_grid, p_z, 84)

        # --- 7. Correlations ---
        means = (u_mean, n_mean, z_mean)
        stds  = (u_std,  n_std,  z_std)
        corr_un = self._correlation(0, 1, prob, means, stds, u_grid, n_grid, z_grid)
        corr_uz = self._correlation(0, 2, prob, means, stds, u_grid, n_grid, z_grid)
        corr_nz = self._correlation(1, 2, prob, means, stds, u_grid, n_grid, z_grid)

        # --- 8. Pack into FitResult ---
        return FitResult(
            galaxy_name=name,
            galaxy_index=galaxy_index,
            u_grid=u_grid, n_grid=n_grid, z_grid=z_grid,
            prob=prob, chi2=chi2,
            p_u=p_u, p_n=p_n, p_z=p_z,
            u_best=u_best, n_best=n_best, z_best=z_best,
            u_mean=u_mean, n_mean=n_mean, z_mean=z_mean,
            u_std=u_std,   n_std=n_std,   z_std=z_std,
            u_16=u_16, u_50=u_50, u_84=u_84,
            n_16=n_16, n_50=n_50, n_84=n_84,
            z_16=z_16, z_50=z_50, z_84=z_84,
            corr_un=corr_un, corr_uz=corr_uz, corr_nz=corr_nz,
            y_obs=y_obs, y_sig=y_sig,
            use_lines=use_lines,
        )

    def fit_all(
        self,
        use_lines: list[int] | None = None,
    ) -> list[FitResult]:
        """
        Fit every galaxy in DataLoader and return a list of FitResults.

        Parameters
        ----------
        use_lines : list of int, optional
            Same mask applied to every galaxy.

        Returns
        -------
        list[FitResult]
        """
        return [
            self.fit(i, use_lines=use_lines)
            for i in range(self.data.n_galaxies)
        ]

    # ------------------------------------------------------------------
    # Private: likelihood
    # ------------------------------------------------------------------

    def _compute_likelihood(
        self,
        y_obs: np.ndarray,
        y_sig: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute chi-squared and the normalised likelihood over the full grid.

        chi²(U,n,Z) = Σ_i [ (model_i(U,n,Z) - y_obs_i) / y_sig_i ]²

        prob(U,n,Z) = exp(-chi²/2), normalised so Σ prob = 1.

        Parameters
        ----------
        y_obs : np.ndarray, shape (n_ratios,)
        y_sig : np.ndarray, shape (n_ratios,)
            Lines to ignore have y_sig set to 999_999 by get_observations().

        Returns
        -------
        chi2 : np.ndarray, shape (n_u, n_n, n_z)
        prob : np.ndarray, shape (n_u, n_n, n_z)
        """
        cube   = self.grid.cube   # (n_u, n_n, n_z, n_ratios)
        n_u, n_n, n_z, n_r = cube.shape

        # Vectorised chi-squared over all grid points at once
        # cube shape: (n_u, n_n, n_z, n_r)
        # y_obs / y_sig broadcast as (n_r,)
        residuals = (cube - y_obs) / y_sig          # (n_u, n_n, n_z, n_r)
        chi2      = np.sum(residuals ** 2, axis=3)  # (n_u, n_n, n_z)

        # Likelihood — use float64 (float128 is non-portable and rarely needed
        # at grid resolutions typical for this tool)
        raw_prob = np.exp(-chi2 / 2.0)
        prob     = raw_prob / raw_prob.sum()        # normalise

        return chi2, prob

    # ------------------------------------------------------------------
    # Private: percentile from a discrete 1-D posterior
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile(x: np.ndarray, px: np.ndarray, q: float) -> float:
        """
        Compute the q-th percentile of a discrete distribution (x, P(x)).

        Matches the original PercentileXPx() logic:
        builds a piecewise-linear CDF from the midpoint probabilities,
        then interpolates.

        Parameters
        ----------
        x  : np.ndarray   parameter values (sorted)
        px : np.ndarray   probability at each x (normalised)
        q  : float        percentile in [0, 100]

        Returns
        -------
        float
        """
        # Midpoint rule for CDF (mirrors the original implementation)
        a      = np.concatenate([[0.0], px[:-1]])
        b      = np.concatenate([[0.0], px[1:]])
        px_mid = (a + b) / 2.0
        px_mid = px_mid / px_mid.sum()
        cdf    = np.cumsum(px_mid)

        # Guard: clamp CDF to [0, 1] to avoid interp extrapolation errors
        cdf = np.clip(cdf, 0.0, 1.0)

        target = q / 100.0
        # If target is outside the CDF range, return the boundary
        if target <= cdf[0]:
            return float(x[0])
        if target >= cdf[-1]:
            return float(x[-1])

        f = interp1d(cdf, x)
        return float(f(target))

    # ------------------------------------------------------------------
    # Private: pairwise correlation coefficient
    # ------------------------------------------------------------------

    @staticmethod
    def _correlation(
        axis1: int,
        axis2: int,
        prob:  np.ndarray,
        means: tuple[float, float, float],
        stds:  tuple[float, float, float],
        u_grid: np.ndarray,
        n_grid: np.ndarray,
        z_grid: np.ndarray,
    ) -> float:
        """
        Compute the posterior correlation coefficient between two parameters.

        ρ(A, B) = Cov(A,B) / (σ_A · σ_B)

        Cov(A,B) = Σ_{i,j} (A_i - <A>)(B_j - <B>) · P_marginal(A_i, B_j)

        Parameters
        ----------
        axis1, axis2 : int   indices into (U=0, n=1, Z=2)
        prob         : 3-D joint posterior
        means        : (u_mean, n_mean, z_mean)
        stds         : (u_std,  n_std,  z_std)
        u_grid, n_grid, z_grid : 1-D parameter axes

        Returns
        -------
        float   correlation coefficient in [-1, 1]
        """
        grids = (u_grid, n_grid, z_grid)
        arr1  = grids[axis1]
        arr2  = grids[axis2]

        # The marginal axis to sum over is the one that is neither axis1 nor axis2
        margin_axis = ({0, 1, 2} - {axis1, axis2}).pop()
        p2d = prob.sum(axis=margin_axis)   # shape (len_axis1, len_axis2)

        # Make sure axis ordering matches: we want p2d[i, j] = P(arr1[i], arr2[j])
        # After summing over margin_axis, the remaining axes are in original order.
        # We may need to transpose if axis1 > axis2.
        if axis1 > axis2:
            p2d = p2d.T

        mean1, mean2 = means[axis1], means[axis2]
        std1,  std2  = stds[axis1],  stds[axis2]

        if std1 == 0 or std2 == 0:
            return 0.0

        # Vectorised covariance
        d1   = arr1 - mean1                      # (n1,)
        d2   = arr2 - mean2                      # (n2,)
        cov  = float(np.sum(np.outer(d1, d2) * p2d))
        return cov / (std1 * std2)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BayesianFitter("
            f"n_galaxies={self.data.n_galaxies}, "
            f"grid_shape={self.grid.shape})"
        )
