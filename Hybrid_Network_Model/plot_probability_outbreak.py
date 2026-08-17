"""
Plot probability of large epidemic branch.

P_large = P(R_inf > 0.35)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODULE_DIR = "/Users/joycewilliamslt/Documents/GitHub/Oxford_Maths01"

if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)


try:
    from plotting_style import set_plot_style, heatmap_cmap
    print("Successfully imported plotting style")
except ImportError as e:
    print(f"Error importing plotting style: {e}")
    print(f"Check file exists at: {MODULE_DIR}/plotting_style.py")
    sys.exit(1)

MODULE_DIR2 = "/Users/joycewilliamslt/Documents/GitHub/Oxford_Maths01/Hybrid_Network_Model"

if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)


try:
    from hybrid_network_model_v1 import run_repeated_sims
    print("Successfully imported run_repeated_sims")
except ImportError as e:
    print(f"Error importing plotting style: {e}")
    print(f"Check file exists at: {MODULE_DIR2}/plotting_style.py")
    sys.exit(1)

set_plot_style()


def plot_probability_heatmap_grid(df, tau_vals, delay_vals,
                                  val_col="prob_large",
                                  save_path=None):

    fig = plt.figure(figsize=(3*len(tau_vals)+1, 3*len(delay_vals)))
    gs = fig.add_gridspec(len(delay_vals), len(tau_vals),
                          right=0.92, wspace=0.3, hspace=0.35)

    first_im = None

    for i, delay in enumerate(sorted(delay_vals, reverse=True)):
        for j, tau in enumerate(sorted(tau_vals)):

            ax = fig.add_subplot(gs[i, j])

            data = df[(df.tau == tau) & (df.delay == delay)]

            grid = data.pivot(index="q", columns="p", values=val_col)
            grid = grid.sort_index(ascending=False).sort_index(axis=1)

            im = sns.heatmap(grid, cmap=heatmap_cmap, ax=ax,
                             cbar=False, vmin=0, vmax=1)

            if first_im is None:
                first_im = im

                # ---- Format p and q labels to 3 decimal places ----
                # Get the actual tick positions (0.5 offset for heatmap)
                # Seaborn places ticks at 0.5, 1.5, ...; we need to set labels accordingly.
                # But we can simply set the labels with the index values.
                # We need to set the ticks to the centers of the cells.
            n_p = len(grid.columns)
            n_q = len(grid.index)
            ax.set_xticks(np.arange(n_p) + 0.5)
            ax.set_yticks(np.arange(n_q) + 0.5)
            ax.set_xticklabels([f"{x:.3f}" for x in grid.columns],
                               rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels([f"{y:.3f}" for y in grid.index],
                               rotation=0, fontsize=8)

            # Axis labels
            if j == 0:
                ax.set_ylabel(r"$q$", fontsize=10)
            else:
                ax.set_ylabel("")

            if i == len(delay_vals) - 1:
                ax.set_xlabel(r"$p$", fontsize=10)
            else:
                ax.set_xlabel("")

            ax.set_title(rf"$\tau = {tau}$, $\delta = {delay}$", fontsize=11)
            ax.grid(False)

    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(first_im.collections[0], cax=cbar_ax).set_label(
        r"$P(R_\infty>0.35)$")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


df = pd.read_csv("probability_large_outbreak_results.csv")

plot_probability_heatmap_grid(
    df,
    tau_vals=[2, 4, 6],
    delay_vals=[0, 0.5, 1],
    save_path="probability_large_branch_heatmaps.png")

plt.show()