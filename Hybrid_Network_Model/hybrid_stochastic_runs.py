"""
hybrid_stochastic_runs.py

Run repeated stochastic hybrid model simulations and plot histograms
for multiple (p, q, tau, delta) parameter combinations.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Import simulation functions from existing module ----
MODULE_DIR = "/Users/joycewilliamslt/Documents/GitHub/Oxford_Maths01/Hybrid_Network_Model"
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)

try:
    from hybrid_network_model_v1 import run_repeated_sims
    print("Successfully imported run_repeated_sims from hybrid_network_model_v1")
except ImportError as e:
    print(f"Error importing: {e}")
    print(f"Make sure the file exists at: {MODULE_DIR}/hybrid_network_model_v1.py")
    sys.exit(1)

# set plotting style
try:
    from Oxford_Maths01.plotting_style import set_plot_style
    set_plot_style()
except ImportError:
    plt.style.use('seaborn-v0_8-whitegrid')


def plot_histogram_grid_for_combos(
    param_combos,
    N, k, beta, gamma, rho,
    tmax=100, n_runs=1000,
    metric='final_epidemic_fraction',
    normalize=True,
    bins=50,
    save_fig_path=None,
    save_csv_path=None,
    figsize_factor=5  # width and height per subplot
):
    """
    Run repeated simulations for multiple (p, q, tau, delay) combinations
    and plot histograms in a grid.

    Parameters
    ----------
    param_combos : list of tuples
        Each tuple: (p, q, tau, delay)
    N, k, beta, gamma, rho : float
        Network and epidemic parameters
    tmax : float
        Simulation end time
    n_runs : int
        Number of stochastic runs per combination
    metric : str
        Column name in the results DataFrame to plot (default 'final_epidemic_fraction')
    normalize : bool
        If True, divide by N (plot fraction). If False, plot counts.
    bins : int or sequence
        Number of bins for histograms
    save_fig_path : str or None
        Path to save the figure
    save_csv_path : str or None
        Path to save the summary CSV
    figsize_factor : int
        Base size for each subplot (width and height)

    Returns
    -------
    summary_df : pd.DataFrame
        Summary statistics for each combination.
    """
    if not param_combos:
        print("No combinations provided.")
        return pd.DataFrame()

    n_combos = len(param_combos)
    results = []   # list of (p, q, tau, delay, values_array)
    summary_records = []

    for idx, (p, q, tau, delay) in enumerate(param_combos, 1):
        print(f"Running combo {idx}/{n_combos}: p={p:.2f}, q={q:.3f}, tau={tau:.1f}, delay={delay:.1f}")

        df = run_repeated_sims(
            N=N, k=k, beta=beta, gamma=gamma, rho=rho,
            tau=tau, delay=delay, p=p, q=q, tmax=tmax,
            n_runs=n_runs
        )

        # Extract the values to plot
        values = df[metric].values
        if normalize and metric == 'final_epidemic_fraction':
            # Already a fraction, but if not, we can normalize
            pass  # final_epidemic_fraction is already normalized
        elif normalize and metric != 'final_epidemic_fraction':
            # If they use e.g. 'final_epidemic_size', divide by N
            values = values / N

        # Store summary statistics
        rec = {
            'p': p, 'q': q, 'tau': tau, 'delay': delay,
            'mean': values.mean(),
            'std': values.std(ddof=1),
            'median': np.median(values),
            'min': values.min(),
            'max': values.max(),
            'q5': np.percentile(values, 5),
            'q95': np.percentile(values, 95)
        }
        summary_records.append(rec)
        results.append((p, q, tau, delay, values))

    summary_df = pd.DataFrame(summary_records)

    # Save CSV if requested
    if save_csv_path is not None:
        summary_df.to_csv(save_csv_path, index=False)
        print(f"Summary CSV saved to {save_csv_path}")

    # ---- Create the grid of histograms ----
    # Determine grid layout
    ncols = int(np.ceil(np.sqrt(n_combos)))
    nrows = int(np.ceil(n_combos / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_factor * ncols, figsize_factor * nrows))
    axes_flat = axes.flatten() if n_combos > 1 else [axes]

    # Determine x-axis label
    if normalize:
        xlabel = r'Final epidemic fraction $R_\infty / N$'
    else:
        xlabel = r'Final epidemic size $R_\infty$ (count)'

    for ax, (p, q, tau, delay, vals) in zip(axes_flat, results):
        ax.hist(vals, bins=bins, edgecolor='black', alpha=0.7, color='teal')
        ax.set_title(rf'$p={p:.2f}, q={q:.2f}, \tau={tau:.1f}, \delta={delay:.1f}$', fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes_flat[n_combos:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_fig_path is not None:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_fig_path}")
    plt.show()

    return summary_df

def plot_histogram_grid_beta_gamma(
    p, q, tau, delay,
    beta_values, gamma_values,
    N=1000, k=4, rho=0.02,
    tmax=100, n_runs=1000,
    metric='final_epidemic_fraction',
    normalize=True,
    bins=30,
    save_fig_path=None,
    save_csv_path=None,
    figsize_factor=4.5  # size per subplot
):
    """
    Run repeated simulations for a grid of (beta, gamma) values,
    keeping p, q, tau, and delay fixed. Plot histograms in a grid.

    Parameters
    ----------
    p, q, tau, delay : float
        Fixed intervention parameters.
    beta_values : list of float
        Transmission rates to vary.
    gamma_values : list of float
        Recovery rates to vary.
    N, k, rho : float
        Network and epidemic parameters (beta and gamma are varied).
    tmax : float
        Simulation end time.
    n_runs : int
        Number of stochastic runs per combination.
    metric : str
        Column name in results DataFrame to plot.
    normalize : bool
        If True, plot fraction (R∞/N); else counts.
    bins : int or sequence
        Number of bins for histograms.
    save_fig_path : str or None
        Path to save the figure.
    save_csv_path : str or None
        Path to save the summary CSV.
    figsize_factor : float
        Base size per subplot.

    Returns
    -------
    summary_df : pd.DataFrame
        Summary statistics for each beta-gamma combination.
    """
    n_beta = len(beta_values)
    n_gamma = len(gamma_values)
    n_combos = n_beta * n_gamma

    if n_combos == 0:
        print("No combinations provided.")
        return pd.DataFrame()

    results = []   # list of (beta, gamma, values_array)
    summary_records = []

    # Loop over beta and gamma
    for i, beta in enumerate(beta_values, 1):
        for j, gamma in enumerate(gamma_values, 1):
            combo_idx = (i - 1) * n_gamma + j
            print(f"Running combo {combo_idx}/{n_combos}: beta={beta:.2f}, gamma={gamma:.2f}, "
                  f"p={p:.2f}, q={q:.3f}, tau={tau:.1f}, delay={delay:.1f}")

            df = run_repeated_sims(
                N=N, k=k, beta=beta, gamma=gamma, rho=rho,
                tau=tau, delay=delay, p=p, q=q, tmax=tmax,
                n_runs=n_runs
            )

            values = df[metric].values
            if normalize and metric == 'final_epidemic_fraction':
                pass  # already fraction
            elif normalize:
                values = values / N

            # Summary stats
            rec = {
                'beta': beta, 'gamma': gamma,
                'mean': values.mean(),
                'std': values.std(ddof=1),
                'median': np.median(values),
                'min': values.min(),
                'max': values.max(),
                'q5': np.percentile(values, 5),
                'q95': np.percentile(values, 95)
            }
            summary_records.append(rec)
            results.append((beta, gamma, values))

    summary_df = pd.DataFrame(summary_records)

    # Save CSV if requested
    if save_csv_path is not None:
        summary_df.to_csv(save_csv_path, index=False)
        print(f"Summary CSV saved to {save_csv_path}")

    # ---- Create grid of histograms: rows = beta, cols = gamma ----
    fig, axes = plt.subplots(
        n_beta, n_gamma,
        figsize=(figsize_factor * n_gamma, figsize_factor * n_beta),
        sharex=True, sharey=True
    )
    # Handle single row/col cases
    if n_beta == 1 and n_gamma == 1:
        axes = np.array([[axes]])
    elif n_beta == 1:
        axes = axes.reshape(1, -1)
    elif n_gamma == 1:
        axes = axes.reshape(-1, 1)

    xlabel = r'Final epidemic fraction $R_\infty / N$' if normalize else r'Final epidemic size $R_\infty$ (count)'

    for i, beta in enumerate(beta_values):
        for j, gamma in enumerate(gamma_values):
            ax = axes[i, j]
            idx = i * n_gamma + j
            vals = results[idx][2]
            ax.hist(vals, bins=bins, edgecolor='black', alpha=0.7, color='teal')
            ax.set_title(rf'$\beta={beta:.2f}, \gamma={gamma:.2f}$', fontsize=12)
            if i == n_beta - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

    # Figure-level title with fixed parameters
    fig.suptitle(
        rf'Fixed: $p={p:.2f}, q={q:.2f}, \tau={tau:.1f}, \delta={delay:.1f}$',
        fontsize=14, y=0.98
    )

    plt.tight_layout()
    if save_fig_path is not None:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_fig_path}")
    plt.show()

    return summary_df
# Example usage
if __name__ == "__main__":
    # Define a set of parameter combinations (p, q, tau, delay)
    combos = [
        (0.9, 0.22, 2.0, 0),]

        # tested combos for gamma = 0.2: (0.9, 0.22, 2.0, 0)
    # Run and plot
    # summary = plot_histogram_grid_for_combos(
    #     param_combos=combos,
    #     N=1000, k=3, beta=0.4, gamma=0.2, rho=0.05,
    #     tmax=100, n_runs=1000,
    #     bins=50,
    #     save_fig_path="histogram_grid_0.5.pdf",
    #     save_csv_path="histogram_summary_0.5.csv"
    # )
    #
    # print("\nSummary table:")
    # print(summary)

    # vary beta and gamma, keep p, q, tau, delta fixed
    summary_beta_gamma = plot_histogram_grid_beta_gamma(
        p=0.52, q=0.085, tau=2.0, delay=0.0,
        beta_values=[0.4],
        gamma_values=[0.1, 0.2, 0.3],
        N=1000, k=3, rho=0.05,
        tmax=100, n_runs=1000,
        save_fig_path="histogram_p0.52_q0.085_t2_d0.pdf",
        save_csv_path="histogram_summary_p0.52_q0.085_t2_d0.csv"
    )

