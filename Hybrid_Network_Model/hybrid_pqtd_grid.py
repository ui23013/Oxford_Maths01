"""
Combine the per-(tau, delay) hybrid sweep heatmaps into a single grid
figure: tau increasing left -> right across columns, delay ("delta")
increasing bottom -> top across rows. Each cell is itself a heatmap
of mean_final_epidemic_fraction over the (p, q) sweep.

Expects one or more CSVs with columns:
    tau, delay, p, q, mean_final_epidemic_fraction, std_final_epidemic_fraction
Glob pattern below picks up all matching sweep files in INPUT_DIR.
"""
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotting_style import set_plot_style, heatmap_cmap

set_plot_style()

# ---- config ----
INPUT_DIR = ".."
FILE_PATTERN = "hybrid_sweep_results_tau*.csv"
VALUE_COL = "mean_final_epidemic_fraction"
OUTPUT_PATH = "../Tau_Delta_Heatmap_Plots/tau_delay_grid.png"

# ---- load + combine all sweep files ----
paths = sorted(glob.glob(f"{INPUT_DIR}/{FILE_PATTERN}"))
if not paths:
    raise FileNotFoundError(f"No files matched {FILE_PATTERN} in {INPUT_DIR}")
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

taus = sorted(df["tau"].unique())          # columns, left -> right
delays = sorted(df["delay"].unique())      # rows, bottom -> top
p_vals = sorted(df["p"].unique())
q_vals = sorted(df["q"].unique())

n_cols = len(taus)
n_rows = len(delays)

# Shared color scale across every panel so cells are comparable
vmin, vmax = df[VALUE_COL].min(), df[VALUE_COL].max()

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(2.2 * n_cols, 2.2 * n_rows),
    squeeze=False,
    sharex=True, sharey=True,
)

im = None
for row_idx, delay in enumerate(delays):
    # row 0 of the axes grid is the TOP of the figure, and we want
    # delay increasing upward, so the largest delay goes in row 0.
    plot_row = n_rows - 1 - row_idx
    for col_idx, tau in enumerate(taus):
        ax = axes[plot_row, col_idx]
        cell = df[(df["tau"] == tau) & (df["delay"] == delay)]
        grid = cell.pivot(index="q", columns="p", values=VALUE_COL)
        grid = grid.reindex(index=q_vals, columns=p_vals)

        im = ax.imshow(
            grid.values,
            origin="lower",
            aspect="auto",
            cmap=heatmap_cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[min(p_vals), max(p_vals), min(q_vals), max(q_vals)],
        )

        # Row labels (delay) along the left column only
        if col_idx == 0:
            ax.set_ylabel(f"$\\delta$ = {delay}\n$q$")
        else:
            ax.set_ylabel("")
        # Bottom row gets the p-axis label, with tau underneath it
        if plot_row == n_rows - 1:
            ax.set_xlabel(f"$p$\n$\\tau$ = {tau}")

# fig.suptitle("Final epidemic fraction across ($p$, $q$) sweeps, by $\\tau$ and $\\delta$", y=1.02)

# Single shared colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
cbar.set_label(r"Mean $R_\infty$ (proportion of population)")

fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved {OUTPUT_PATH}")
print(f"Grid: {n_rows} rows (delay: {delays}) x {n_cols} cols (tau: {taus})")

