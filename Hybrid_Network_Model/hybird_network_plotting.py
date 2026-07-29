"""
Hybrid Network Model Plotting
=============================

Visualizes parameter sweep results for the hybrid model combining:
- Reporting delay (δ)
- Incomplete information (p, q)
- Intervention time (τ)

Creates heatmaps showing how final epidemic fraction varies across
the (p, q) parameter space for different combinations of τ and δ.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from plotting_style import set_plot_style, heatmap_cmap

set_plot_style()


def load_sweep_results(csv_path):
    '''
    Load sweep data from CSV.

    Args:
        csv_path: path to CSV file with sweep results

    Returns:
        DataFrame with columns: tau, delay, p, q, mean_final_epidemic_fraction, std_final_epidemic_fraction
    '''
    return pd.read_csv(csv_path)


def plot_pq_heatmap_single(sweep_df, tau, delay, val_col='mean_final_epidemic_fraction',
                           figsize=(8, 6), save_path=None):
    '''
    Plot 2D heatmap of mean R_inf over p,q space for given tau and delay values.

    Args:
        sweep_df: DataFrame with columns tau, delay, p, q, and val_col
        tau: intervention time to slice
        delay: reporting delay to slice
        val_col: which column to use for heatmap color (default: mean_final_epidemic_fraction)
        figsize: figure size tuple
        save_path: optional path to save figure

    Returns:
        fig, ax: matplotlib figure and axes objects
    '''
    # Filter for specific tau and delay
    filtered_data = sweep_df[(sweep_df['tau'] == tau) & (sweep_df['delay'] == delay)]

    if filtered_data.empty:
        print(f"No data found for tau={tau}, delay={delay}")
        return None, None

    # Pivot to create grid
    pq_grid = filtered_data.pivot(index='q', columns='p', values=val_col)
    pq_grid = pq_grid.sort_index(ascending=False)  # q increasing upward
    pq_grid = pq_grid.sort_index(axis=1, ascending=True)  # p increasing rightward

    fig, ax = plt.subplots(figsize=figsize)

    im = sb.heatmap(
        pq_grid,
        annot=False,
        cmap=heatmap_cmap,
        cbar_kws={"label": r"Mean $R_\infty$ (fraction)"},
        ax=ax,
        vmin=0,
        vmax=1
    )

    ax.set_xlabel(r"$p$ (informed edge removal probability)")
    ax.set_ylabel(r"$q$ (proportion of edges removed)")

    # Set ticks and labels properly
    ax.set_xticks(np.arange(len(pq_grid.columns)) + 0.5)
    ax.set_yticks(np.arange(len(pq_grid.index)) + 0.5)
    ax.set_xticklabels([f"{x:.2f}" for x in pq_grid.columns], rotation=45, ha="right")
    ax.set_yticklabels([f"{y:.2f}" for y in pq_grid.index], rotation=0)

    ax.set_title(rf"Mean $R_\infty$ for $\tau = {tau}$, $\delta = {delay}$")
    ax.grid(False)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

    return fig, ax


def plot_pq_heatmap_grid(sweep_df, tau_vals, delay_vals, val_col='mean_final_epidemic_fraction',
                         save_path=None):
    '''
    Create a grid of heatmaps showing p,q parameter space for all tau and delay combinations.
    Organized as a tau vs delta grid (tau increases left-to-right, delta increases bottom-to-top).
    '''
    n_tau = len(tau_vals)
    n_delay = len(delay_vals)

    # Create figure with space for colorbar on the right
    fig = plt.figure(figsize=(3 * n_tau + 1, 3 * n_delay))
    gs = fig.add_gridspec(n_delay, n_tau, right=0.92, wspace=0.3, hspace=0.35)

    # Find global min/max for consistent colorbar scaling
    vmin = sweep_df[val_col].min()
    vmax = sweep_df[val_col].max()

    # Sort tau and delay values
    tau_vals_sorted = sorted(tau_vals)
    delay_vals_sorted = sorted(delay_vals, reverse=True)  # Reverse so higher delays are at top

    first_im = None
    for i, delay in enumerate(delay_vals_sorted):
        for j, tau in enumerate(tau_vals_sorted):
            ax = fig.add_subplot(gs[i, j])

            filtered_data = sweep_df[(sweep_df['tau'] == tau) & (sweep_df['delay'] == delay)]

            if filtered_data.empty:
                ax.text(0.5, 0.5, f'No data\nτ={tau}, δ={delay}',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Pivot to create grid
            pq_grid = filtered_data.pivot(index='q', columns='p', values=val_col)
            pq_grid = pq_grid.sort_index(ascending=False)
            pq_grid = pq_grid.sort_index(axis=1, ascending=True)

            im = sb.heatmap(
                pq_grid,
                annot=False,
                cmap=heatmap_cmap,
                ax=ax,
                cbar=False,
                vmin=vmin,
                vmax=vmax
            )

            # Store first image for colorbar reference
            if first_im is None:
                first_im = im

            # Set ticks and labels properly to avoid mismatch error
            ax.set_xticks(np.arange(len(pq_grid.columns)) + 0.5)
            ax.set_yticks(np.arange(len(pq_grid.index)) + 0.5)
            ax.set_xticklabels([f"{x:.2f}" for x in pq_grid.columns],
                               rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels([f"{y:.2f}" for y in pq_grid.index],
                               rotation=0, fontsize=8)

            # Only show axis labels on edges
            if j == 0:
                ax.set_ylabel(r"$q$", fontsize=10)
            else:
                ax.set_ylabel("")

            if i == n_delay - 1:
                ax.set_xlabel(r"$p$", fontsize=10)
            else:
                ax.set_xlabel("")

            # Title with tau and delta
            ax.set_title(rf"$\tau = {tau}$, $\delta = {delay}$", fontsize=11, pad=8)
            ax.grid(False)

    # Add colorbar on the right side
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(first_im.collections[0], cax=cbar_ax)
    cbar.set_label(r"Mean $R_\infty$ (fraction)", fontsize=11)

    fig.suptitle(
        r"Hybrid Model: Effect of Reporting Delay ($\delta$, vertical) and Intervention Time ($\tau$, horizontal) on $(p, q)$ Space",
        fontsize=13, y=0.98)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

    return fig, gs


def plot_delay_effect(sweep_df, p_val, q_val, tau_vals=None,
                      figsize=(10, 6), save_path=None):
    '''
    Plot how final epidemic size varies with reporting delay for fixed (p, q).

    Args:
        sweep_df: DataFrame with sweep results
        p_val: fixed probability value
        q_val: fixed proportion value
        tau_vals: list of tau values to plot (if None, uses all in data)
        figsize: figure size tuple
        save_path: optional path to save figure

    Returns:
        fig, ax: matplotlib figure and axes objects
    '''
    if tau_vals is None:
        tau_vals = sorted(sweep_df['tau'].unique())

    fig, ax = plt.subplots(figsize=figsize)

    for tau in tau_vals:
        data_slice = sweep_df[(sweep_df['p'] == p_val) &
                              (sweep_df['q'] == q_val) &
                              (sweep_df['tau'] == tau)]

        if not data_slice.empty:
            delay_vals = data_slice['delay'].values
            means = data_slice['mean_final_epidemic_fraction'].values
            stds = data_slice['std_final_epidemic_fraction'].values

            ax.errorbar(delay_vals, means, yerr=stds, marker='o',
                        label=rf"$\tau = {tau}$", linewidth=2, markersize=8, capsize=5)

    ax.set_xlabel(r"Reporting delay $\delta$")
    ax.set_ylabel(r"Mean $R_\infty$ (fraction)")
    ax.set_title(rf"Effect of Reporting Delay ($p={p_val}$, $q={q_val}$)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

    return fig, ax


def plot_intervention_strength(sweep_df, tau_vals=None, delay_vals=None,
                               figsize=(10, 6), save_path=None):
    '''
    Plot effect of intervention strength (p, q combination) across different
    delays and intervention times.

    Args:
        sweep_df: DataFrame with sweep results
        tau_vals: list of tau values to plot (if None, uses all in data)
        delay_vals: list of delay values to plot (if None, uses all in data)
        figsize: figure size tuple
        save_path: optional path to save figure

    Returns:
        fig, ax: matplotlib figure and axes objects
    '''
    if tau_vals is None:
        tau_vals = sorted(sweep_df['tau'].unique())
    if delay_vals is None:
        delay_vals = sorted(sweep_df['delay'].unique())

    fig, axes = plt.subplots(1, len(tau_vals), figsize=(6 * len(tau_vals), 5), sharey=True)

    if len(tau_vals) == 1:
        axes = [axes]

    for ax_idx, tau in enumerate(tau_vals):
        ax = axes[ax_idx]

        for delay in delay_vals:
            data_slice = sweep_df[(sweep_df['tau'] == tau) & (sweep_df['delay'] == delay)]

            if not data_slice.empty:
                # Average over p,q to get a single number per delay
                mean_r_inf = data_slice['mean_final_epidemic_fraction'].mean()
                std_r_inf = data_slice['mean_final_epidemic_fraction'].std()

                ax.errorbar(delay, mean_r_inf, yerr=std_r_inf, marker='o',
                            markersize=10, capsize=5, label=f"$\delta={delay}$")

        ax.set_xlabel(r"Reporting delay $\delta$")
        if ax_idx == 0:
            ax.set_ylabel(r"Mean $R_\infty$ (fraction)")
        ax.set_title(rf"$\tau = {tau}$")
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

    fig.suptitle("Effect of Intervention Delay on Final Epidemic Size", fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

    return fig, axes


def create_summary_table(sweep_df, metric='mean_final_epidemic_fraction'):
    '''
    Create a summary table of results for key parameter combinations.

    Args:
        sweep_df: DataFrame with sweep results
        metric: which metric to summarize

    Returns:
        summary_df: summary table
    '''
    summary_df = sweep_df.groupby(['tau', 'delay'])[metric].agg(['mean', 'min', 'max', 'std'])
    return summary_df.round(4)


# Example usage and main plotting routine
if __name__ == "__main__":

    # Load the sweep results
    csv_path = "hybrid_sweep_results_tau34.csv"
    sweep_df = load_sweep_results(csv_path)

    # Get unique values for plotting
    tau_vals = sorted(sweep_df['tau'].unique())
    delay_vals = sorted(sweep_df['delay'].unique())

    # Grid of heatmaps for all tau and delay combinations
    print("Creating grid of heatmaps...")
    fig_grid, gs_grid = plot_pq_heatmap_grid(
        sweep_df, tau_vals, delay_vals, save_path="hybrid_heatmap_grid_tau34.pdf"
    )
    plt.show()

    # Individual heatmaps for each tau-delay combination
    print("Creating individual heatmaps...")
    for tau in tau_vals:
        for delay in delay_vals:
            fig, ax = plot_pq_heatmap_single(
                sweep_df, tau, delay,
                save_path=f"hybrid_heatmap_tau{tau}_delay{delay}.pdf"
            )
            if fig is not None:
                plt.show()

    # Intervention strength comparison
    print("Creating intervention strength plots...")
    fig, axes = plot_intervention_strength(
        sweep_df, tau_vals, delay_vals,
        save_path="hybrid_intervention_strength_tau34.pdf")
    plt.show()

    # Print summary table
    print("\nSummary statistics:")
    summary = create_summary_table(sweep_df)
    print(summary)



