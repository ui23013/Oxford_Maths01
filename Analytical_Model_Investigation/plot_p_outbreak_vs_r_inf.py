"""
plot_Poutbreak_vs_Rinf.py

For each (p, q) combination, produces ONE figure with a grid of
subplots -- one subplot per R0 -- plotting P_outbreak (y) against mean
R_inf (x), one parametric curve per tau, traced out as delay increases
from 0 to max(delay).

Critical delay definition
--------------------------
delta_c is the first (smallest) delay at which mean_R_inf crosses
above 0.5, i.e. the delay beyond which the expected final size of the
epidemic exceeds half the population. This is marked with an "x" on
each curve, at the point where it crosses R_inf = 0.5.

(This is a threshold on the OUTBREAK SIZE, not on the outbreak
probability -- it does not use the P_outbreak=0.5 crossing.)

The delay=0 point on each curve is marked with a square, and the
largest-delay point with a triangle, so the direction of increasing
delay along the curve is readable without needing delay as an axis.

Input
-----
extinction_probability_234.csv, with columns:
    R0, tau, delay, p, q, prob_dieout, mean_R_inf
P_outbreak is computed as 1 - prob_dieout.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys

MODULE_DIR = "/Users/joycewilliamslt/Documents/GitHub/Oxford_Maths01"
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)
from Oxford_Maths01.plotting_style import set_plot_style

set_plot_style()

DATA_PATH = "Critical_delay_estimation/extinction_probability_234.csv"
CROSS_RINF = 0.5   # threshold on mean_R_inf used to define delta_c


def find_critical_delay(df, value_col="mean_R_inf", cross_value=CROSS_RINF, interpolate=True):
    """
    For each (p, q, tau, R0), find delta_c: the delay at which
    df[value_col] first crosses above cross_value, interpolated
    between the bracketing grid points.

    status:
      "crossed"         - a genuine crossing was found and (optionally) interpolated
      "always_outbreak" - already past the threshold at the smallest delay tested
      "always_dieout"   - value never crosses the threshold within the tested range
    """
    critical = []

    for keys, group in df.groupby(["p", "q", "tau", "R0"]):
        group = group.sort_values("delay")
        delays = group["delay"].to_numpy()
        values = group[value_col].to_numpy()

        hits = np.where(values >= cross_value)[0]

        if len(hits) == 0:
            delta_c, status = np.nan, "always_dieout"
        elif hits[0] == 0:
            delta_c, status = delays[0], "always_outbreak"
        else:
            i = hits[0]
            d_lo, d_hi = delays[i - 1], delays[i]
            v_lo, v_hi = values[i - 1], values[i]
            if interpolate and v_hi != v_lo:
                delta_c = d_lo + (cross_value - v_lo) * (d_hi - d_lo) / (v_hi - v_lo)
            else:
                delta_c = d_hi
            status = "crossed"

        critical.append({
            "p": keys[0], "q": keys[1], "tau": keys[2], "R0": keys[3],
            "critical_delay": delta_c, "status": status,
        })
    return pd.DataFrame(critical)


def main():
    results = pd.read_csv(DATA_PATH)
    results["prob_outbreak"] = 1.0 - results["prob_dieout"]

    critical = find_critical_delay(results)
    critical.to_csv("critical_delay_Rinf_results.csv", index=False)

    pqs = results[["p", "q"]].drop_duplicates().values
    R0_vals = sorted(results["R0"].unique())
    tau_vals = sorted(results["tau"].unique())

    style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {tau: style_colors[i % len(style_colors)] for i, tau in enumerate(tau_vals)}

    legend_handles = [
        mlines.Line2D([], [], color=colors[tau], marker="o", label=rf"$\tau={tau}$")
        for tau in tau_vals
    ]
    legend_handles.append(
        mlines.Line2D([], [], color="grey", marker="s", linestyle="None", label=r"$\delta=0$")
    )
    legend_handles.append(
        mlines.Line2D([], [], color="grey", marker="^", linestyle="None",
                       label=rf"$\delta={results['delay'].max()}$")
    )
    legend_handles.append(
        mlines.Line2D([], [], color="grey", marker="x", linestyle=":",
                       label=r"$\delta_c$ ($R_\infty=0.5$ crossing)")
    )

    for p, q in pqs:

        n_cols = int(np.ceil(len(R0_vals) / 2))
        fig, axes = plt.subplots(
            2, n_cols,
            figsize=(4 * n_cols, 4 * 2),
            sharex=True, sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).flatten()
        for extra_ax in axes[len(R0_vals):]:
            extra_ax.axis("off")

        pq_subset = results[(results["p"] == p) & (results["q"] == q)]

        for ax, R0 in zip(axes, R0_vals):

            subset = pq_subset[pq_subset["R0"] == R0]

            for tau in tau_vals:
                curve = subset[subset["tau"] == tau].sort_values("delay")
                if curve.empty:
                    continue

                r_inf = curve["mean_R_inf"].to_numpy()
                prob = curve["prob_outbreak"].to_numpy()

                ax.plot(r_inf, prob, color=colors[tau], marker="o", markersize=5)
                ax.plot(r_inf[0], prob[0], color=colors[tau], marker="s", markersize=8)
                ax.plot(r_inf[-1], prob[-1], color=colors[tau], marker="^", markersize=8)

                crit_row = critical[
                    (critical["p"] == p) & (critical["q"] == q) &
                    (critical["tau"] == tau) & (critical["R0"] == R0)
                ]
                if not crit_row.empty and crit_row["status"].iloc[0] == "crossed":
                    delta_c = crit_row["critical_delay"].iloc[0]
                    # by construction R_inf = CROSS_RINF at delta_c; interpolate
                    # P_outbreak at that same delay for the y-position
                    prob_at_delta_c = np.interp(delta_c, curve["delay"], prob)
                    ax.plot(
                        CROSS_RINF, prob_at_delta_c,
                        marker="x", color=colors[tau], markersize=9, markeredgewidth=2,
                    )

            ax.axvline(CROSS_RINF, color="grey", linestyle="--", linewidth=1)
            ax.set_title(rf"$R_0={R0}$")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(-0.02, 1.02)

        for ax in axes[:len(R0_vals)]:
            ax.set_xlabel(r"$R_\infty$")
        axes[0].set_ylabel(r"$P_{\mathrm{outbreak}}$")

        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )

        fig.suptitle(rf"$p={p:.2f},\ q={q:.2f}$")

        fig.savefig(
            f"P_outbreak_vs_R_inf_p{p:.2f}_q{q:.2f}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()