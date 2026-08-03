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

            if j == 0:
                ax.set_ylabel(r"$q$")
            else:
                ax.set_ylabel("")

            if i == len(delay_vals)-1:
                ax.set_xlabel(r"$p$")
            else:
                ax.set_xlabel("")

            ax.set_title(rf"$\tau={tau},\delta={delay}$", fontsize=11)
            ax.grid(False)

    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(first_im.collections[0], cax=cbar_ax).set_label(
        r"$P(R_\infty>0.35)$"
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


df = pd.read_csv("probability_large_outbreak_results.csv")

plot_probability_heatmap_grid(
    df,
    tau_vals=[2, 3, 4, 5],
    delay_vals=[0, 0.5, 1],
    save_path="probability_large_branch_heatmaps.png"
)

plt.show()