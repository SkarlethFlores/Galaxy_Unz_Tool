"""
galaxy_unz.plots
================
Plotter — all visualisation for a completed fit.

Replaces and consolidates from gism_Functions.py:
  Doplot        → Plotter.data_overview()
  plot_probs    → Plotter.posteriors()      (1-D marginal posteriors)
  plot_grids    → Plotter.corner()          (2-D joint posteriors)
  GetSolutions  → split: FitResult.summary() handles printing,
                         Plotter.fit_summary() handles the figure

Every method returns the matplotlib Figure so the caller can save,
display, or further customise it.  Nothing is saved or shown
automatically — the caller decides:

    fig = Plotter(result).posteriors()
    fig.savefig("output/NGC1569_posteriors.pdf")
    plt.show()
"""

from __future__ import annotations

from pathlib import Path
from typing  import Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import numpy as np

from .results import FitResult
from .io      import DataLoader, RATIO_NAMES, RATIO_LABELS


# Tick label formatter shared across panels
_fmt_tick = FuncFormatter(lambda x, _: f"{x:.1f}")


class Plotter:
    """
    Visualisation for a single galaxy FitResult.

    Parameters
    ----------
    result : FitResult
        Output of BayesianFitter.fit().
    style : str, optional
        Matplotlib style name (default "default").  Pass "seaborn-v0_8"
        or any installed style for a different look.

    Example
    -------
    >>> plotter = Plotter(result)
    >>> fig = plotter.posteriors()
    >>> fig.savefig("posteriors.pdf")
    >>> fig = plotter.corner()
    >>> fig.savefig("corner.pdf")
    """

    def __init__(self, result: FitResult, style: str = "default") -> None:
        self.result = result
        self.style  = style

    # ------------------------------------------------------------------
    # Public: 1-D marginal posteriors
    # ------------------------------------------------------------------

    def posteriors(
        self,
        figsize: tuple[float, float] = (14, 4),
        color_best: str = "limegreen",
        color_mean: str = "red",
    ) -> plt.Figure:
        """
        Three-panel figure showing the marginal 1-D posterior for each
        parameter (log U, log n_e, Z/Z_sun).

        Vertical lines mark the best-fit (green dashed) and posterior
        mean (red dashed).  The 16–84th percentile range is shaded.

        Parameters
        ----------
        figsize       : figure size in inches
        color_best    : colour for the best-fit line
        color_mean    : colour for the mean line

        Returns
        -------
        matplotlib.figure.Figure
        """
        r   = self.result
        fig = plt.figure(figsize=figsize)

        specs = [
            (r.u_grid, r.p_u, r.u_best, r.u_mean, r.u_16, r.u_84,
             r"$\log(U)$",    r.u_grid.min(), r.u_grid.max()),
            (r.n_grid, r.p_n, r.n_best, r.n_mean, r.n_16, r.n_84,
             r"$\log(n_e)$",  r.n_grid.min(), r.n_grid.max()),
            (r.z_grid, r.p_z, r.z_best, r.z_mean, r.z_16, r.z_84,
             r"$Z/Z_\odot$",  r.z_grid.min(), r.z_grid.max()),
        ]

        for idx, (grid, prob, best, mean, p16, p84, xlabel, xmin, xmax) in enumerate(specs):
            ax = fig.add_subplot(1, 3, idx + 1)
            ax.plot(grid, prob, color="steelblue", lw=1.5)
            ax.axvline(best, linestyle="--", lw=2, color=color_best,
                       label="Best fit")
            ax.axvline(mean, linestyle="--", lw=2, color=color_mean,
                       label="Mean")
            ax.axvspan(p16, p84, color=color_mean, alpha=0.10)

            ax.set_xlim(xmin * 0.97 if xmin < 0 else xmin * 0.95,
                        xmax * 1.03 if xmax > 0 else xmax * 0.95)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(xlabel, fontsize=14)
            ax.xaxis.set_major_formatter(_fmt_tick)
            ax.tick_params(axis="both", direction="in", which="both",
                           length=6, labelsize=12, top=True, right=True)
            ax.set_yticklabels([])

            if idx == 0:
                ax.legend(loc="upper right", fontsize=11)

        fig.suptitle(f"{r.galaxy_name} — marginal posteriors", fontsize=13)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Public: 2-D corner / joint posteriors
    # ------------------------------------------------------------------

    def corner(
        self,
        figsize: tuple[float, float] = (10, 10),
        cmap: str = "Blues",
        n_levels: int = 6,
    ) -> plt.Figure:
        """
        Corner plot: 3×3 grid showing all pairwise 2-D joint posteriors
        on the lower triangle and 1-D marginals on the diagonal.

        Parameters
        ----------
        figsize  : figure size in inches
        cmap     : matplotlib colormap for the 2-D panels
        n_levels : number of contour levels

        Returns
        -------
        matplotlib.figure.Figure
        """
        r      = self.result
        params = [
            (r.u_grid, r.p_u, r.u_best, r.u_mean, r"$\log(U)$"),
            (r.n_grid, r.p_n, r.n_best, r.n_mean, r"$\log(n_e)$"),
            (r.z_grid, r.p_z, r.z_best, r.z_mean, r"$Z/Z_\odot$"),
        ]
        n = len(params)

        fig, axes = plt.subplots(n, n, figsize=figsize)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        # Marginalised 2-D joints
        joints = {
            (0, 1): r.prob.sum(axis=2),   # U vs n  (sum over Z)
            (0, 2): r.prob.sum(axis=1),   # U vs Z  (sum over n)
            (1, 2): r.prob.sum(axis=0),   # n vs Z  (sum over U)
        }

        for row in range(n):
            for col in range(n):
                ax = axes[row, col]

                if col > row:
                    # Upper triangle — hide
                    ax.set_visible(False)
                    continue

                if row == col:
                    # Diagonal — 1-D marginal
                    grid, prob, best, mean, xlabel = params[row]
                    ax.plot(grid, prob, color="steelblue", lw=1.5)
                    ax.axvline(best, linestyle="--", lw=1.5, color="limegreen")
                    ax.axvline(mean, linestyle="--", lw=1.5, color="red")
                    ax.set_yticklabels([])
                    ax.tick_params(direction="in", which="both",
                                   length=5, labelsize=10, top=True, right=True)
                    if row == n - 1:
                        ax.set_xlabel(xlabel, fontsize=12)
                    else:
                        ax.set_xticklabels([])
                    ax.xaxis.set_major_formatter(_fmt_tick)

                else:
                    # Lower triangle — 2-D joint contours
                    # x-axis = col parameter, y-axis = row parameter
                    x_grid, _, x_best, x_mean, xlabel = params[col]
                    y_grid, _, y_best, y_mean, ylabel = params[row]

                    key = (col, row) if (col, row) in joints else (row, col)
                    p2d = joints[key]
                    # Ensure shape is (len_x, len_y) = (len_col, len_row)
                    if key == (row, col):
                        p2d = p2d.T

                    ax.contourf(x_grid, y_grid, p2d.T,
                                levels=n_levels, cmap=cmap)
                    ax.contour(x_grid, y_grid, p2d.T,
                               levels=n_levels, colors="white",
                               linewidths=0.5, alpha=0.4)
                    ax.plot(x_best, y_best, "o", color="limegreen",
                            ms=7, zorder=5, label="Best fit")
                    ax.plot(x_mean, y_mean, "o", color="red",
                            ms=7, zorder=5, label="Mean")

                    ax.tick_params(direction="in", which="both",
                                   length=5, labelsize=10, top=True, right=True)
                    ax.xaxis.set_major_formatter(_fmt_tick)
                    ax.yaxis.set_major_formatter(_fmt_tick)

                    if row == n - 1:
                        ax.set_xlabel(xlabel, fontsize=12)
                    else:
                        ax.set_xticklabels([])

                    if col == 0:
                        ax.set_ylabel(ylabel, fontsize=12)
                    else:
                        ax.set_yticklabels([])

        fig.suptitle(f"{r.galaxy_name} — corner plot", fontsize=13, y=1.01)
        return fig

    # ------------------------------------------------------------------
    # Public: combined fit summary (posteriors + text box)
    # ------------------------------------------------------------------

    def fit_summary(
        self,
        figsize: tuple[float, float] = (16, 5),
    ) -> plt.Figure:
        """
        Four-panel figure: three 1-D posteriors + a text panel with the
        numerical results.  Equivalent to the original GetSolutions figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        r   = self.result
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.08)

        # --- Three posterior panels (reuse the same drawing logic) ---
        specs = [
            (r.u_grid, r.p_u, r.u_best, r.u_mean, r.u_16, r.u_84,
             r"$\log(U)$"),
            (r.n_grid, r.p_n, r.n_best, r.n_mean, r.n_16, r.n_84,
             r"$\log(n_e)$"),
            (r.z_grid, r.p_z, r.z_best, r.z_mean, r.z_16, r.z_84,
             r"$Z/Z_\odot$"),
        ]

        for idx, (grid, prob, best, mean, p16, p84, xlabel) in enumerate(specs):
            ax = fig.add_subplot(gs[idx])
            ax.plot(grid, prob, color="steelblue", lw=1.5)
            ax.axvline(best, linestyle="--", lw=2, color="limegreen",
                       label="Best fit" if idx == 0 else "")
            ax.axvline(mean, linestyle="--", lw=2, color="red",
                       label="Mean" if idx == 0 else "")
            ax.axvspan(p16, p84, color="red", alpha=0.10)
            ax.set_xlabel(xlabel, fontsize=13)
            ax.xaxis.set_major_formatter(_fmt_tick)
            ax.tick_params(direction="in", which="both", length=6,
                           labelsize=11, top=True, right=True)
            ax.set_yticklabels([])
            ax.set_ylim(bottom=0)
            if idx == 0:
                ax.legend(fontsize=10)

        # --- Text panel ---
        ax_txt = fig.add_subplot(gs[3])
        ax_txt.axis("off")

        lo_u, hi_u = r.u_err
        lo_n, hi_n = r.n_err
        lo_z, hi_z = r.z_err

        lines = [
            r.galaxy_name,
            "",
            f"log(U)  = {r.u_50:.2f}  $-${lo_u:.2f}  $+${hi_u:.2f}",
            f"log(n)  = {r.n_50:.2f}  $-${lo_n:.2f}  $+${hi_n:.2f}",
            f"Z/Z☉    = {r.z_50:.2f}  $-${lo_z:.2f}  $+${hi_z:.2f}",
            "",
            f"ρ(U,n) = {r.corr_un:.2f}",
            f"ρ(U,Z) = {r.corr_uz:.2f}",
            f"ρ(n,Z) = {r.corr_nz:.2f}",
        ]
        ax_txt.text(
            0.05, 0.95, "\n".join(lines),
            transform=ax_txt.transAxes,
            fontsize=11, va="top", family="monospace",
        )

        fig.suptitle("Fit results", fontsize=13)
        return fig

    # ------------------------------------------------------------------
    # Public: galaxy data overview  (replaces Doplot)
    # ------------------------------------------------------------------

    @staticmethod
    def data_overview(
        data: DataLoader,
        figsize: tuple[float, float] = (24, 12),
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """
        Six-panel overview of all line ratios across the galaxy sample.
        Replaces the original Doplot() + figure assembly in GetDatatable.

        Parameters
        ----------
        data      : DataLoader   — the loaded galaxy table
        figsize   : figure size in inches
        save_path : if given, saves the figure to this path

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = data.galaxies

        # Six ratios shown in the original Doplot calls
        plot_specs = [
            ("OIII52/OIII88",              "[OIII] 52/[OIII] 88",              "OIII52/OIII88_ERR"),
            ("(2.2OIII88+OIII52)/NIII57",  "(2.2×[OIII] 88+[OIII] 52)/[NIII] 57", "(2.2OIII88+OIII52)/NIII57_ERR"),
            ("OIII88/NIII57",              "[OIII] 88/[NIII] 57",              "OIII88/NIII57_ERR"),
            ("OIII52/NIII57",              "[OIII] 52/[NIII] 57",              "OIII52/NIII57_ERR"),
            ("OIII88/NII122",              "[OIII] 88/[NII] 122",              "OIII88/NII122_ERR"),
            ("OIII52/NII122",              "[OIII] 52/[NII] 122",              "OIII52/NII122_ERR"),
        ]

        x      = np.arange(len(df))
        labels = df["Galaxy"].tolist()

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for ax, (ratio_col, title, err_col) in zip(axes, plot_specs):
            y   = df[ratio_col].to_numpy(dtype=float)
            err = df[err_col].to_numpy(dtype=float)

            # Mask sentinel values
            valid = (y > -90) & (err > 0)

            ax.errorbar(
                x[valid], y[valid], yerr=err[valid],
                fmt="o", capsize=4, lw=1.2,
                color="steelblue", ecolor="steelblue",
            )
            # Label each point with the galaxy name
            for xi, yi, name in zip(x[valid], y[valid], np.array(labels)[valid]):
                ax.annotate(
                    name, (xi, yi),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=8, color="dimgray",
                )

            ax.set_title(title, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Line ratio", fontsize=10)
            ax.tick_params(direction="in", which="both", top=True, right=True)

        fig.suptitle("Galaxy line ratios overview", fontsize=14)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Public: chi-squared map (2-D slice through the cube)
    # ------------------------------------------------------------------

    def chi2_slice(
        self,
        fix_param: str = "z",
        fix_index: int | None = None,
        figsize: tuple[float, float] = (7, 5),
        cmap: str = "viridis_r",
    ) -> plt.Figure:
        """
        2-D chi-squared map with one parameter fixed at its best-fit value.

        Parameters
        ----------
        fix_param : "u", "n", or "z" — which parameter to fix
        fix_index : grid index to fix at; defaults to the best-fit index
        figsize   : figure size in inches
        cmap      : matplotlib colormap

        Returns
        -------
        matplotlib.figure.Figure
        """
        r    = self.result
        chi2 = r.chi2

        # Determine the slice
        best_idx = np.unravel_index(chi2.argmin(), chi2.shape)

        if fix_param == "u":
            fi    = fix_index if fix_index is not None else best_idx[0]
            data  = chi2[fi, :, :]
            xlabel, ylabel = r"$\log(n_e)$", r"$Z/Z_\odot$"
            xgrid, ygrid   = r.n_grid, r.z_grid
            bx, by         = r.n_best, r.z_best
            mx, my         = r.n_mean, r.z_mean
            title_fix      = f"log(U) = {r.u_grid[fi]:.2f}"
        elif fix_param == "n":
            fi    = fix_index if fix_index is not None else best_idx[1]
            data  = chi2[:, fi, :]
            xlabel, ylabel = r"$\log(U)$", r"$Z/Z_\odot$"
            xgrid, ygrid   = r.u_grid, r.z_grid
            bx, by         = r.u_best, r.z_best
            mx, my         = r.u_mean, r.z_mean
            title_fix      = f"log(n) = {r.n_grid[fi]:.2f}"
        else:  # "z"
            fi    = fix_index if fix_index is not None else best_idx[2]
            data  = chi2[:, :, fi]
            xlabel, ylabel = r"$\log(U)$", r"$\log(n_e)$"
            xgrid, ygrid   = r.u_grid, r.n_grid
            bx, by         = r.u_best, r.n_best
            mx, my         = r.u_mean, r.n_mean
            title_fix      = f"Z/Z☉ = {r.z_grid[fi]:.2f}"

        fig, ax = plt.subplots(figsize=figsize)
        cf = ax.contourf(xgrid, ygrid, data.T, levels=12, cmap=cmap)
        ax.contour(xgrid, ygrid, data.T, levels=12,
                   colors="white", linewidths=0.4, alpha=0.5)
        fig.colorbar(cf, ax=ax, label=r"$\chi^2$")

        ax.plot(bx, by, "o", color="limegreen", ms=9, zorder=5,
                label="Best fit")
        ax.plot(mx, my, "o", color="red",       ms=9, zorder=5,
                label="Mean")

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(
            f"{r.galaxy_name} — χ² map  ({title_fix})", fontsize=12
        )
        ax.legend(fontsize=10)
        ax.tick_params(direction="in", which="both", length=6,
                       labelsize=11, top=True, right=True)
        ax.xaxis.set_major_formatter(_fmt_tick)
        ax.yaxis.set_major_formatter(_fmt_tick)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Plotter(galaxy='{self.result.galaxy_name}')"
