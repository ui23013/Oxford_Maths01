"""
Plot critical delay results

x-axis: R0
y-axis: tau
colour: critical delay delta_c
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


MODULE_DIR2 = "/Users/joycewilliamslt/Documents/GitHub/Oxford_Maths01"
if MODULE_DIR2 not in sys.path:
    sys.path.append(MODULE_DIR2)
from Oxford_Maths01.plotting_style import set_plot_style, heatmap_cmap

set_plot_style()

df = pd.read_csv("Critical_delay_estimation/critical_delay_Rinf_results.csv")
# don't dropna here — NaN is meaningful (no crossing in tested delay range)

pqs = df[["p", "q"]].drop_duplicates().values

fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)

all_taus = np.round(np.arange(2.5, 6.5, 0.5), 1)
all_R0s = sorted(df["R0"].unique())

for ax, (p, q) in zip(axes.flat, pqs):
    subset = df[(df["p"] == p) & (df["q"] == q)]

    heatmap = (
        subset.pivot(index="tau", columns="R0", values="critical_delay")
        .reindex(index=all_taus, columns=all_R0s)
    )

    sns.heatmap(
        heatmap,
        ax=ax,
        annot=True,
        vmin=0, vmax=df["critical_delay"].max(),
        fmt=".2f",
        cmap=heatmap_cmap,
        cbar=True,
        linewidths=0.5,
        mask=heatmap.isna(),
    )

    ax.invert_yaxis()
    ax.set_yticks([i + 0.5 for i in range(len(all_taus))])
    ax.set_yticklabels([f"{t:.1f}" for t in all_taus], rotation=0)

    ax.set_title(rf"$p={p:.2f},\ q={q:.2f}$")
    ax.set_xlabel(r"$R_0$")
    ax.set_ylabel(r"$\tau$")

fig.suptitle(r"$\delta_c$ ($R_\infty > 0.5$ crossing)")
plt.savefig("critical_delay_Rinf_heatmaps.png", dpi=300, bbox_inches="tight")
plt.show()