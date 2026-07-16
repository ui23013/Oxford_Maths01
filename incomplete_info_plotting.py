import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

plt.style.use('bmh')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "black",
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.7
})

def load_sweep_results(csv_path):
    '''
    load sweep data
    Args:
        csv_path: to data on device
    '''
    return pd.read_csv(csv_path)


def plot_pq_heatmap(sweep_df, tau, val_col='mean_r_inf'):
    '''
    plots 2d heatmap of mean r_inf over p,q space for given tau value
    Args:
        sweep_df: df with columns p, q, tau, mean_inf
        tau: which tau slice to plot
        val_col: which col ot plot as the heatmap colour
    '''

    tau_data = sweep_df[sweep_df['tau']==tau]
    pq_grid = tau_data.pivot(index='q', columns='p', values=val_col)
    pq_grid = pq_grid.sort_index(ascending=False)          # q increasing upward
    pq_grid = pq_grid.sort_index(axis=1, ascending=True)  # p increasing rightward

    fig, ax = plt.subplots(figsize=(6, 5))
    sb.heatmap(pq_grid, annot=False, cmap= 'viridis', cbar_kws={"label": r"Mean $R_\infty (\%)$"}, ax=ax)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$q$")
    ax.set_xticklabels([f"{x:.3f}" for x in pq_grid.columns], rotation=45, ha="right")
    ax.set_yticklabels([f"{y:.3f}" for y in pq_grid.index], rotation=0)
    ax.set_title(rf"Mean $R_\infty$ for $\tau = {tau}$")
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    return fig,ax

trial_csv = load_sweep_results('/Users/joycewilliamslt/Documents/GitHub/Oxford_Maths01/trial4_sweep_results.csv')

tau_vals = [5, 10]

for tau in tau_vals:
    trial_heatmap = plot_pq_heatmap(trial_csv, tau, val_col='mean_r_inf')
    print(trial_heatmap)