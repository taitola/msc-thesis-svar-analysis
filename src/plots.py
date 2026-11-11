"""Plotting functions for the exploratory figures, IRF grid and counterfactual figures.

Every public function accepts an optional `save_path`. When given, the figure is written
to that path at 300 dpi before it is displayed.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src import config
from src.var_irf import IrfBand

SavePath = str | PathLike | None


@contextmanager
def _seaborn_theme() -> Iterator[None]:
    """Apply the exploratory plot style without affecting later figures."""
    with plt.rc_context():
        sns.set_theme(style="whitegrid")
        yield


def _save_and_show(fig: Figure, save_path: SavePath = None) -> None:
    """Optionally save a figure at 300 dpi, then display it."""
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_series(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = 'Year',
    xlim_right: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (12, 6),
    palette: str | list[str] | None = None,
    linewidth: float = 2.5,
    save_path: SavePath = None,
) -> None:
    """Generic plotting function for time series."""
    with _seaborn_theme():
        fig = plt.figure(figsize=figsize)
        sns.lineplot(data=data, x=x, y=y, palette=palette, linewidth=linewidth)
        if title:
            plt.title(title)
        if ylabel:
            plt.ylabel(ylabel)
        if xlabel:
            plt.xlabel(xlabel)
        if xlim_right:
            plt.xlim(right=pd.Timestamp(xlim_right))
        if ylim:
            plt.ylim(ylim)
        _save_and_show(fig, save_path)


def plot_national_permits_per_capita(
    df_per_capita: pd.DataFrame,
    save_path: SavePath = None,
) -> None:
    """Plot national building permits per 1,000 residents."""
    plot_series(df_per_capita[['National']],
                title='National Building Permits per Capita (Monthly Rate)',
                ylabel='Permits per 1,000 Residents', palette=['black'],
                save_path=save_path)


def plot_metro_permits_per_capita(
    df_per_capita: pd.DataFrame,
    save_path: SavePath = None,
) -> None:
    """Plot metro building permits per 1,000 residents, with the legend sorted by peak value."""
    df_metro = df_per_capita.drop(columns='National', errors='ignore')
    sorted_cols = df_metro.max().sort_values(ascending=False).index.tolist()
    plot_series(df_metro[sorted_cols], title='Metro Building Permits per Capita',
                ylabel='Permits per 1,000 Residents', palette='Set2', save_path=save_path)


def plot_house_price_indices(df_var: pd.DataFrame, save_path: SavePath = None) -> None:
    """Plot Case-Shiller indices, with the legend sorted by the last value."""
    cols = [col for col in config.HPI_COLS if col in df_var.columns]
    if not cols:
        print("No HPI columns found in the dataset.")
        return

    sorted_cols = df_var[cols].iloc[-1].sort_values(ascending=False).index.tolist()
    plot_series(df_var[sorted_cols],
                title='Case-Shiller House Price Indices (1987/01-2007/07), index at 2000/01 = 100',
                ylabel='Index Value', xlim_right='2008-01-01', palette='tab10',
                save_path=save_path)


def plot_economic_indicators(df_var: pd.DataFrame, save_path: SavePath = None) -> None:
    """Plot the macroeconomic indicators used in the VAR on one percentage axis."""
    cols = [col for col in config.ECONOMIC_INDICATORS if col in df_var.columns]
    plot_series(df_var[cols].rename(columns=config.ECONOMIC_INDICATORS),
                title='Economic Indicators (1987/01-2007/07)',
                ylabel='Percent', xlim_right='2008-01-01', ylim=(0, 10), save_path=save_path)


def plot_housing_starts(df_visuals: pd.DataFrame, save_path: SavePath = None) -> None:
    """Plot national housing starts with recession periods shaded."""
    with _seaborn_theme():
        fig = plt.figure(figsize=(14, 7))
        sns.lineplot(data=df_visuals, x=df_visuals.index, y='National_housing starts',
                     linewidth=2.5, label='Housing Starts')

        # Shade recessions
        y_min, y_max = plt.ylim()
        plt.fill_between(df_visuals.index, y_min, y_max,
                         where=df_visuals['National_recessions'] == 1,
                         color='grey', alpha=0.3, label='Recession')

        plt.title('Housing Starts with Recession Shading (1959 - 2025)')
        plt.ylabel('Housing Starts (Thousands of Units)')
        plt.xlabel('Year')
        plt.xlim(right=pd.Timestamp('2026-01-01'))
        plt.legend()
        _save_and_show(fig, save_path)


def plot_irf_grid(
    irf_results: Mapping[str, IrfBand],
    cs_names: Mapping[str, str],
    ylim: tuple[float, float] = (-0.042, 0.006),
    save_path: SavePath = None,
) -> None:
    """
    Plots theta_{43,i}, the response of log(hpi_{t+i}) to a 1-SD structural FFR shock,
    for the 7 indices in a 4x2 grid matching Figure 22 in the thesis.
    """
    if not irf_results:
        print("No results to plot.")
        return

    with plt.style.context('seaborn-v0_8-whitegrid'):
        n_rows, n_cols = 4, 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12), sharex=True, sharey=True)
        axes_flat = axes.flatten()
        n_panels = len(irf_results)

        for i, (cs_col, (response, lower, upper)) in enumerate(irf_results.items()):
            ax = axes_flat[i]
            steps = np.arange(len(response))

            # Mean OIRF and symmetric 95% bootstrapped confidence band
            ax.plot(steps, response, color='#1f77b4', linewidth=1.8)
            ax.fill_between(steps, lower, upper, color='#1f77b4', alpha=0.22, edgecolor='none')
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)

            title_name = cs_names.get(cs_col, cs_col)
            ax.set_title(title_name, fontsize=13, fontweight='bold', pad=8)
            ax.set_ylim(ylim)
            ax.set_yticks([0.00, -0.01, -0.02, -0.03, -0.04])
            ax.set_xticks([0, 20, 40, 60])

            # Label left column Y-axes matching Figure 22
            if i % n_cols == 0:
                ax.set_ylabel('Response', fontsize=11)

        # The last panel of each column needs its own X-axis, because sharex=True hides
        # tick labels on any panel above a subplot that is later removed
        for ax in axes_flat[max(n_panels - n_cols, 0):n_panels]:
            ax.set_xlabel('Months', fontsize=11)
            ax.tick_params(labelbottom=True)

        # Remove unused bottom-right subplot
        for j in range(n_panels, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.suptitle(
            'Orthogonalized IRF: Response of House Prices to FFR Shock\n(95 % Bootstrapped CI)',
            fontsize=15,
            fontweight='bold',
            y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        _save_and_show(fig, save_path)


def plot_ffr_counterfactual_comparison(
    cf_results: Mapping[str, pd.DataFrame],
    df_3var: pd.DataFrame,
    cf_start_date: str = config.COUNTERFACTUAL_START_DATE,
    save_path: SavePath = None,
) -> None:
    """
    Replicates Figure 21 in the thesis: Actual FFR vs. 3-VAR Counterfactual
    and National Case-Shiller VAR Counterfactual, with w_3t^{cf} = 0 from T*.
    """
    if 'cs' not in cf_results:
        print("National ('cs') results required for Figure 21 comparison.")
        return

    df_nat = cf_results['cs']

    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, ax = plt.subplots(figsize=(12, 5.5))

        ax.plot(df_nat.index, df_nat['actual_ffr'], color='#2b5c8f',
                linewidth=1.8, label='Actual FFR')
        ax.plot(df_3var.index, df_3var['cf_ffr_3var'], color='#d62728',
                linestyle='--', linewidth=1.6, label='3-VAR Counterfactual')
        ax.plot(df_nat.index, df_nat['cf_ffr'], color='#2ca02c',
                linestyle='-.', linewidth=1.6, label='National CS Counterfactual')

        ax.axvline(pd.Timestamp(cf_start_date), color='grey', linestyle='--',
                   linewidth=1.0, alpha=0.6)

        ax.set_title('Federal Funds Rate Counterfactual: Structural FFR Shock = 0 from 2001/1',
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel('FFR (%)', fontsize=11)
        ax.set_ylim(0, 10.5)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.legend(frameon=True, fontsize=10.5, loc='upper right')

        plt.tight_layout()
        _save_and_show(fig, save_path)


def plot_counterfactual_grid(
    cf_results: Mapping[str, pd.DataFrame],
    cs_names: Mapping[str, str] = config.CS_NAMES,
    cf_start_date: str = config.COUNTERFACTUAL_START_DATE,
    save_path: SavePath = None,
) -> None:
    """
    Replicates Figure 23 in the thesis: 7x2 grid showing Actual vs. Counterfactual
    Federal Funds Rate (left column) and Case-Shiller Index in levels (right column).
    Rows follow the order of `cf_results`.
    """
    if not cf_results:
        print("No counterfactual results to plot.")
        return

    n_rows = len(cf_results)
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.0 * n_rows), sharex=True)

        if n_rows == 1:
            axes = np.array([axes])

        for i, (cs_col, df_cf) in enumerate(cf_results.items()):
            ax_ffr = axes[i, 0]
            ax_hpi = axes[i, 1]
            city_title = cs_names.get(cs_col, cs_col)

            # Left Panel: Federal Funds Rate
            ax_ffr.plot(df_cf.index, df_cf['actual_ffr'], color='#2b5c8f',
                        linewidth=1.5, label='Actual')
            ax_ffr.plot(df_cf.index, df_cf['cf_ffr'], color='#d62728',
                        linestyle='--', linewidth=1.5, label='Counterfactual')
            ax_ffr.axvline(pd.Timestamp(cf_start_date), color='grey',
                           linestyle='--', linewidth=0.9, alpha=0.5)
            ax_ffr.set_title('Federal Funds Rate', fontsize=11.5, fontweight='bold')
            ax_ffr.set_ylabel('(%)', fontsize=10)
            ax_ffr.set_ylim(0, 10.5)
            ax_ffr.set_yticks([0, 2, 4, 6, 8, 10])
            ax_ffr.legend(frameon=True, fontsize=9, loc='upper right')

            # Right Panel: Case-Shiller Index (Levels)
            ax_hpi.plot(df_cf.index, df_cf['actual_hpi'], color='#2b5c8f',
                        linewidth=1.5, label='Actual')
            ax_hpi.plot(df_cf.index, df_cf['cf_hpi'], color='#d62728',
                        linestyle='--', linewidth=1.5, label='Counterfactual')
            ax_hpi.axvline(pd.Timestamp(cf_start_date), color='grey',
                           linestyle='--', linewidth=0.9, alpha=0.5)
            ax_hpi.set_title(city_title, fontsize=11.5, fontweight='bold')
            ax_hpi.set_ylabel('Case-Shiller Index', fontsize=10)
            ax_hpi.set_ylim(0, 250)
            ax_hpi.set_yticks([0, 50, 100, 150, 200, 250])
            ax_hpi.legend(frameon=True, fontsize=9, loc='upper left')

        fig.suptitle(
            'Counterfactual: Structural FFR Shock = 0 from 2001/1',
            fontsize=15,
            fontweight='bold',
            y=0.995,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.985])
        _save_and_show(fig, save_path)
