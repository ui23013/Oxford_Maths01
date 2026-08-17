"""
estimate_i_critv1.py

Estimate the critical infection count I_crit at intervention time tau,
across a grid of (R0, tau, delay) combinations, using the same
first-crossing logic already used for delta_c elsewhere in this repo.

Method
------
For a fixed (p, q, R0, tau, delay):
  1. Run n_runs repeated stochastic sims (run_repeated_sims), giving a
     scatter of (I(tau), R_inf) pairs across runs -- this is exactly the
     scatter in rinf_intervation_scatter.py.
  2. Sort the individual runs by I(tau) ascending.
  3. I_crit is the I(tau) value of the first individual run (in that
     ascending order) whose R_inf exceeds `THRESHOLD` (0.5).

No binning/averaging is used -- this is the literal "first I value
where R_inf crosses the threshold" definition, read straight off the
raw scatter of individual runs. NaN means no run in the tested range
ever crossed the threshold -- distinct from "I_crit = 0". Because this
is a single-run first-crossing rather than a smoothed statistic, it
will be more sensitive to individual noisy runs than a binned/averaged
estimate would be; a logistic-regression-based estimate (mentioned
separately as the preferred next step for delta_c) would be the natural
way to make this more robust later, but this version implements the
literal first-crossing method as specified.

ASSUMPTIONS FLAGGED FOR REVIEW
-------------------------------
- GAMMA is fixed at 1.0, so beta = R0 * GAMMA = R0 directly.
- PQ_COMBOS is the fixed list of six (p, q) pairs you gave:
  (0.945, 0.25), (0.84, 0.44), (0.615, 0.415), (0.275, 0.47),
  (0.055, 0.36), (0.5, 0.305). Heatmaps are laid out as a 3x2 grid over
  these six combos (one figure per delay value), matching the
  plot_delta_crit.py convention of 3x2 heatmap grids across (tau, R0).

Usage
-----
    python estimate_i_critv1.py

Leave SMOKE_TEST = True the first time you run this on a new machine --
it runs a tiny slice of the grid with few runs/combo so you can check
the pipeline (and the plots) before committing to the full grid, which
is a fairly large amount of compute (len(P_VALS) * len(Q_VALS) *
len(R0_vals) * len(tau_vals) * len(delay_vals) * n_runs simulations).
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

try:
    from Oxford_Maths01.plotting_style import set_plot_style, heatmap_cmap
    set_plot_style()
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")


# ============================== CONFIG ==============================

THRESHOLD = 0.5  # R_inf threshold that defines "outbreak"

R0_vals = [1.125, 1.25, 1.3125, 1.375, 1.4375, 1.5, 1.75]
tau_vals = [2.5, 3, 3.5, 4]
delay_vals = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]

GAMMA = 1.0

# Fixed (p, q) pairs -- see module docstring
PQ_COMBOS = [
    (0.945, 0.25),
    (0.84, 0.44),
    (0.615, 0.415),
    (0.275, 0.47),
    (0.055, 0.36),
    (0.5, 0.305),
]

N = 1000
k = 3
rho = 0.05
tmax = 100
n_runs = 500

USE_CACHE = True
RESULTS_CSV = "i_crit_results.csv"
RAW_CACHE_PATH = "i_crit_raw_cache.pkl"

SMOKE_TEST = True
if SMOKE_TEST:
    print("SMOKE_TEST = True -- running a small slice of the grid with fewer runs. "
          "Set SMOKE_TEST = False for the full sweep.")
    R0_vals = R0_vals[:2]
    tau_vals = tau_vals[:1]
    delay_vals = delay_vals[:3]
    PQ_COMBOS = PQ_COMBOS[:2]
    n_runs = 50
    USE_CACHE = False  # never reuse a smoke-test cache for the real run

# ======================================================================


def beta_from_R0(R0, gamma=GAMMA):
    """beta such that beta / gamma = R0, at fixed gamma."""
    return R0 * gamma


def get_i_tau_and_r_inf(df, N):
    """
    Extract I(tau) and R_inf as fractions of N from a run_repeated_sims
    output DataFrame. Same convention as rinf_intervation_scatter.py.
    """
    i_tau_frac = df["infected_at_intervention"].to_numpy() / N
    r_inf_frac = df["final_epidemic_fraction"].to_numpy()
    return i_tau_frac, r_inf_frac


def estimate_i_crit(i_tau_frac, r_inf_frac, threshold=THRESHOLD):
    """
    Literal first-crossing estimate of I_crit from a scatter of
    (I(tau), R_inf) pairs across repeated runs at one fixed parameter
    combination -- no binning or averaging.

    Sorts individual runs by I(tau) ascending and returns the I(tau) of
    the first run whose R_inf exceeds `threshold`.

    Returns a dict with:
      i_crit  : float or np.nan. NaN means no individual run crossed
                `threshold` anywhere in the observed range of I(tau) --
                distinct from "I_crit = 0".
      reason  : "ok" | "no_crossing"
    """
    i_tau_frac = np.asarray(i_tau_frac, dtype=float)
    r_inf_frac = np.asarray(r_inf_frac, dtype=float)

    order = np.argsort(i_tau_frac)
    i_sorted = i_tau_frac[order]
    r_sorted = r_inf_frac[order]

    crossing = np.where(r_sorted > threshold)[0]
    if len(crossing) == 0:
        i_crit = np.nan
        reason = "no_crossing"
    else:
        i_crit = float(i_sorted[crossing[0]])
        reason = "ok"

    return dict(i_crit=i_crit, reason=reason)


def run_all_combos(force_recompute=not USE_CACHE):
    """
    Run (or load cached) simulations for every (p, q, R0, tau, delay)
    combination, estimate I_crit for each, and return:
      results_df : one row per combo, with I_crit and diagnostics
      raw_data   : dict keyed by (p, q, R0, tau, delay) with full
                   per-run arrays + the estimate_i_crit() output, for
                   diagnostic scatter plots
    """
    if not force_recompute and os.path.exists(RESULTS_CSV) and os.path.exists(RAW_CACHE_PATH):
        print(f"Loading cached results from {RESULTS_CSV} and {RAW_CACHE_PATH}")
        results_df = pd.read_csv(RESULTS_CSV)
        with open(RAW_CACHE_PATH, "rb") as f:
            raw_data = pickle.load(f)
        return results_df, raw_data

    combos = [
        (p, q, R0, tau, delay)
        for (p, q) in PQ_COMBOS
        for R0 in R0_vals
        for tau in tau_vals
        for delay in delay_vals
    ]
    total = len(combos)
    print(f"Running {total} parameter combinations x n_runs={n_runs} simulations each...")

    records = []
    raw_data = {}

    for idx, (p, q, R0, tau, delay) in enumerate(combos, start=1):
        beta = beta_from_R0(R0)
        gamma = GAMMA

        df = run_repeated_sims(
            N=N, k=k, beta=beta, gamma=gamma, rho=rho,
            tau=tau, delay=delay, p=p, q=q, tmax=tmax, n_runs=n_runs,
        )
        i_tau_frac, r_inf_frac = get_i_tau_and_r_inf(df, N)
        est = estimate_i_crit(i_tau_frac, r_inf_frac, threshold=THRESHOLD)

        records.append({
            "p": p, "q": q, "R0": R0, "beta": beta, "gamma": gamma,
            "tau": tau, "delay": delay,
            "I_crit": est["i_crit"], "reason": est["reason"], "n_runs": len(df),
        })
        raw_data[(p, q, R0, tau, delay)] = {
            "i_tau_frac": i_tau_frac, "r_inf_frac": r_inf_frac, **est,
        }

        status = f"I_crit={est['i_crit']:.4f}" if np.isfinite(est["i_crit"]) else f"I_crit=NaN ({est['reason']})"
        print(f"[{idx}/{total}] p={p} q={q} R0={R0:.4f} tau={tau} delay={delay} -> {status}")

    results_df = pd.DataFrame(records)
    results_df.to_csv(RESULTS_CSV, index=False)
    with open(RAW_CACHE_PATH, "wb") as f:
        pickle.dump(raw_data, f)
    print(f"Saved results to {RESULTS_CSV} and raw scatter data to {RAW_CACHE_PATH}")

    return results_df, raw_data


def plot_i_crit_heatmaps_for_delay(results_df, delay, pq_combos, tau_vals, R0_vals, save_path=None):
    """
    One figure per delay value: a 3x2 grid of heatmaps of I_crit across
    (tau, R0), one panel per (p, q) combo -- matches the plot_delta_crit.py
    convention of 3x2 heatmap grids across (tau, R0), with NaN cells
    masked rather than dropped.
    """
    sub = results_df[results_df["delay"] == delay]

    nrows, ncols = 3, 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True,
    )
    axes = axes.flatten()

    for idx, (p, q) in enumerate(pq_combos):
        ax = axes[idx]
        cell = sub[(sub["p"] == p) & (sub["q"] == q)]
        pivot = cell.pivot(index="tau", columns="R0", values="I_crit")
        pivot = pivot.reindex(index=tau_vals, columns=R0_vals)

        values = pivot.to_numpy(dtype=float)
        vmax = 0.5 if np.all(np.isnan(values)) else max(0.5, np.nanmax(values))

        sns.heatmap(
            pivot,
            ax=ax,
            mask=pivot.isna(),
            cmap=heatmap_cmap,
            vmin=0,
            vmax=vmax,
            cbar_kws={"label": r"$I_{crit}/N$"},
        )
        ax.set_title(rf"$p={p}$, $q={q}$", fontsize=10)
        ax.set_xlabel(r"$R_0$")
        ax.set_ylabel(r"$\tau$")
        ax.set_xticklabels([f"{v:.3f}" for v in R0_vals], rotation=45, ha="right")
        ax.set_yticklabels([f"{v:.1f}" for v in tau_vals], rotation=0)

    for idx in range(len(pq_combos), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(rf"$I_{{crit}}$ across $(\tau, R_0)$ for each $(p,q)$ combo   [$\delta={delay}$]")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")

    return fig, axes


def plot_diagnostic_scatter(i_tau_frac, r_inf_frac, i_crit, params, save_path=None):
    """
    Per-combo diagnostic: the raw I(tau) vs R_inf scatter (as in
    rinf_intervation_scatter.py), with a single vertical line marking
    I_crit and its value annotated directly on the plot. No binning --
    this shows exactly the runs the estimate was computed from.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(i_tau_frac, r_inf_frac, s=25, alpha=0.5, edgecolor="none", label="individual runs")
    ax.axhline(THRESHOLD, color="grey", linestyle="--", linewidth=1, alpha=0.7)

    if np.isfinite(i_crit):
        ax.axvline(i_crit, color="crimson", linestyle="--", linewidth=1.5)
        ax.annotate(
            rf"$I_{{crit}}$ = {i_crit:.4f}",
            xy=(i_crit, THRESHOLD),
            xytext=(8, 10),
            textcoords="offset points",
            color="crimson",
            fontsize=10,
            fontweight="bold",
        )
    else:
        ax.text(
            0.97, 0.95, r"$I_{crit}$: no crossing observed",
            transform=ax.transAxes, ha="right", va="top",
            color="crimson", fontsize=10,
        )

    ax.set_xlabel(r"$I(\tau)/N$")
    ax.set_ylabel(r"$R_\infty/N$")
    ax.set_title(
        rf"$R_0$={params['R0']:.3f}, $\tau$={params['tau']}, $\delta$={params['delay']}, "
        rf"$p$={params['p']}, $q$={params['q']}"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"Saved diagnostic figure to {save_path}")

    return fig, ax


if __name__ == "__main__":

    results_df, raw_data = run_all_combos()

    for delay in delay_vals:
        plot_i_crit_heatmaps_for_delay(
            results_df, delay, PQ_COMBOS, tau_vals, R0_vals,
            save_path=f"i_crit_heatmap_delay{delay}.png",
        )

    # One diagnostic scatter, for a representative combo, so the
    # crossing logic can be sanity-checked by eye.
    demo_p, demo_q = PQ_COMBOS[0]
    demo_R0 = R0_vals[len(R0_vals) // 2]
    demo_tau = tau_vals[0]
    demo_delay = delay_vals[len(delay_vals) // 2]
    demo_key = (demo_p, demo_q, demo_R0, demo_tau, demo_delay)

    if demo_key in raw_data:
        r = raw_data[demo_key]
        plot_diagnostic_scatter(
            r["i_tau_frac"], r["r_inf_frac"], r["i_crit"],
            params=dict(R0=demo_R0, tau=demo_tau, delay=demo_delay, p=demo_p, q=demo_q),
            save_path="i_crit_diagnostic_example.png",
        )

    plt.show()