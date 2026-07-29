"""
Generate the intervention-strength/delay effect plot across tau = 3..6,
using plot_intervention_strength().
"""
import glob
import pandas as pd
import matplotlib.pyplot as plt

from plotting_style import set_plot_style, heatmap_cmap

set_plot_style()


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


# ---- load all sweep files (tau 3-6) and plot ----
paths = sorted(glob.glob("hybrid_sweep_results_tau*.csv"))
sweep_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

fig, axes = plot_intervention_strength(
    sweep_df,
    tau_vals=[3, 4, 5, 6],
    save_path="../intervention_strength_tau3-6.pdf",)