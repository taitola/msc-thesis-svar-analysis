"""Plotting functions for the exploratory figures and the IRF grid."""

from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config


@contextmanager
def _seaborn_theme():
    """Apply the exploratory plot style without affecting later figures."""
    with plt.rc_context():
        sns.set_theme(style="whitegrid")
        yield


def plot_series(data, x=None, y=None, title=None, ylabel=None, xlabel='Year',
                xlim_right=None, ylim=None, figsize=(12, 6), palette=None, linewidth=2.5):
    """Generic plotting function for time series."""
    with _seaborn_theme():
        plt.figure(figsize=figsize)
        sns.lineplot(data=data, x=x, y=y, palette=palette, linewidth=linewidth)
        if title: plt.title(title)
        if ylabel: plt.ylabel(ylabel)
        if xlabel: plt.xlabel(xlabel)
        if xlim_right: plt.xlim(right=pd.Timestamp(xlim_right))
        if ylim: plt.ylim(ylim)
        plt.show()


def plot_national_permits_per_capita(df_per_capita):
    """Plot national building permits per capita."""
    plot_series(df_per_capita[['National']],
                title='National Building Permits per Capita (Monthly Rate)',
                ylabel='Ratio', palette=['black'])


def plot_metro_permits_per_capita(df_per_capita):
    """Plot metro building permits per capita, with the legend sorted by peak value."""
    df_metro = df_per_capita.drop(columns='National', errors='ignore')
    sorted_cols = df_metro.max().sort_values(ascending=False).index.tolist()
    plot_series(df_metro[sorted_cols], title='Metro Building Permits per Capita',
                ylabel='Permits / Capita', palette='Set2')


def plot_house_price_indices(df_var):
    """Plot Case-Shiller indices, with the legend sorted by the last value."""
    cols = [col for col in config.HPI_COLS if col in df_var.columns]
    if not cols:
        print("No HPI columns found in the dataset.")
        return

    sorted_cols = df_var[cols].iloc[-1].sort_values(ascending=False).index.tolist()
    plot_series(df_var[sorted_cols],
                title='Case-Shiller House Price Indices (1987/01-2007/07), index at 2000/01 = 100',
                ylabel='Index Value', xlim_right='2008-01-01', palette='tab10')


def plot_economic_indicators(df_var):
    """Plot the macroeconomic indicators used in the VAR on one percentage axis."""
    cols = [col for col in config.ECONOMIC_INDICATORS if col in df_var.columns]
    plot_series(df_var[cols].rename(columns=config.ECONOMIC_INDICATORS),
                title='Economic Indicators (1987/01-2007/07)',
                ylabel='Percent', xlim_right='2008-01-01', ylim=(0, 10))


def plot_housing_starts(df_visuals):
    """Plot national housing starts with recession periods shaded."""
    with _seaborn_theme():
        plt.figure(figsize=(14, 7))
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
        plt.show()


def plot_irf_grid(irf_results, cs_names, ylim=(-0.042, 0.006)):
    """
    Plots the 7 IRF results in a 4x2 grid matching Figure 22 in the thesis.
    """
    if not irf_results:
        print("No results to plot.")
        return

    with plt.style.context('seaborn-v0_8-whitegrid'):
        n_rows, n_cols = 4, 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12), sharex=True, sharey=True)
        axes_flat = axes.flatten()

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

        # Set X-axis label and ticks on the bottom panel(s)
        axes_flat[len(irf_results) - 1].set_xlabel('Months', fontsize=11)
        axes_flat[len(irf_results) - 1].tick_params(labelbottom=True)

        # Remove unused bottom-right subplot
        for j in range(len(irf_results), len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.suptitle(
            'Orthogonalized IRF: Response of House Prices to FFR Shock\n(95 % Bootstrapped CI)',
            fontsize=15,
            fontweight='bold',
            y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()
