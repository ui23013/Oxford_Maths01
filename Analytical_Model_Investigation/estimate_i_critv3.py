"""
estimate_i_critv3.py

Estimate the critical infection count I_crit at intervention time tau,
across a grid of (R0, tau, delay) combinations, by bootstrapping the
exact first-crossing method from v1.

Method
------
For a fixed (p, q, R0, tau, delay):
  1. Run n_runs repeated stochastic sims, giving (I(tau), R_inf) pairs
     across runs -- same scatter as v1 and rinf_intervation_scatter.py.
  2. Resample the n_runs runs with replacement, n_boot times.
  3. For each resample, compute I_crit exactly as in v1: sort by I(tau)
     ascending, take the I(tau) of the first run whose R_inf exceeds
     THRESHOLD (NaN if no run in that resample crosses).
  4. Summarise the n_boot bootstrap estimates: the point estimate is
     their median, with a 90% interval from the 5th/95th percentiles.

This keeps v1's exact literal definition of I_crit unchanged (no
averaging or fitting is applied to the underlying I(tau)/R_inf data --
only resampling), and instead reports how much that literal estimate
moves around under resampling, which is a direct, model-free way to
see whether the single-run first-crossing is being driven by one
borderline run or is actually stable. NaN is still "no bootstrap
resample found a crossing", not "I_crit = 0".

ASSUMPTIONS FLAGGED FOR REVIEW -- identical to v1, see that file:
  GAMMA = 1.0; PQ_COMBOS = the six (p, q) pairs you gave.

Caching
-------
Caching is incremental, not all-or-nothing: run_all_combos() only
(re-)simulates combos from the current CONFIG that aren't already in
the cached raw_data, and merges new results in with whatever was
already there. This matters because CONFIG (R0_vals, tau_vals,
delay_vals, PQ_COMBOS) is expected to grow over the course of the
project -- with the old all-or-nothing caching, growing any of those
lists after a cache already existed on disk meant the script silently
loaded the stale cache and never simulated the new combos, which is
exactly what produced the KeyError when the diagnostic-plotting loop
in __main__ went looking for them. Note this does NOT protect against
changing n_runs, N_BOOT, GAMMA, or THRESHOLD after a cache already
exists -- those aren't part of a combo's key, so a change there won't
be detected and old results will silently be reused under the new
settings. Delete RESULTS_CSV / RAW_CACHE_PATH by hand if you change
any of those.

Usage
-----
    python estimate_i_critv3.py

Leave SMOKE_TEST = True the first time you run this -- it hasn't been
checked against the real simulation code yet, unlike v1.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

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
    heatmap_cmap = "viridis"  # fallback only used if the house style module isn't found


# ============================== CONFIG ==============================

THRESHOLD = 0.5  # R_inf threshold that defines "outbreak"
N_BOOT = 1000     # bootstrap resamples per combo (cheap -- resampling only, no re-simulation)
CI_PERCENTILES = (5, 95)  # 90% interval
BOOT_SEED = 0     # fixed seed for reproducibility across runs

# House colours only (Oxford_Maths01.plotting_style: darkgoldenrod, teal,
# firebrick, gold, olive, darkkhaki, sienna). Scatter points use whatever
# colour set_plot_style() puts first in axes.prop_cycle automatically, so
# they don't need to be set explicitly here.
COLOR_ACTUAL = "firebrick"  # I_crit value actually used (bootstrap median)
COLOR_THRESHOLD = "sienna"  # muted reference line at R_inf = THRESHOLD
COLOR_LOW = "teal"          # 5th percentile bound
COLOR_HIGH = "olive"        # 95th percentile bound
COLOR_HIST = "darkkhaki"    # bootstrap histogram bars

R0_vals = [1.125, 1.25, 1.3125, 1.375, 1.4375, 1.5, 1.75]
tau_vals = [2.5, 3, 3.5, 4]
delay_vals = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]

GAMMA = 1.0  # ASSUMPTION -- see module docstring

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
RESULTS_CSV = "i_crit_results_v3.csv"
RAW_CACHE_PATH = "i_crit_raw_cache_v3.pkl"

SMOKE_TEST = False
if SMOKE_TEST:
    print("SMOKE_TEST = True -- running a small slice of the grid with fewer runs. "
          "Set SMOKE_TEST = False for the full sweep.")
    R0_vals = R0_vals[:2]
    tau_vals = tau_vals[:1]
    delay_vals = delay_vals[:3]
    PQ_COMBOS = PQ_COMBOS[:2]
    n_runs = 50
    N_BOOT = 200
    USE_CACHE = False

# ======================================================================

_rng = np.random.default_rng(BOOT_SEED)


def beta_from_R0(R0, gamma=GAMMA):
    """beta such that beta / gamma = R0, at fixed gamma."""
    return R0 * gamma


def get_i_tau_and_r_inf(df, N):
    """Extract I(tau) and R_inf as fractions of N. Same convention as v1."""
    i_tau_frac = df["infected_at_intervention"].to_numpy() / N
    r_inf_frac = df["final_epidemic_fraction"].to_numpy()
    return i_tau_frac, r_inf_frac


def _first_crossing(i_tau_frac, r_inf_frac, threshold=THRESHOLD):
    """Exactly v1's estimate_i_crit, factored out so it can be bootstrapped."""
    order = np.argsort(i_tau_frac)
    r_sorted = r_inf_frac[order]
    i_sorted = i_tau_frac[order]
    crossing = np.where(r_sorted > threshold)[0]
    if len(crossing) == 0:
        return np.nan
    return i_sorted[crossing[0]]


def estimate_i_crit(i_tau_frac, r_inf_frac, threshold=THRESHOLD, n_boot=N_BOOT, rng=_rng):
    """
    Bootstrap the v1 first-crossing estimate.

    Returns a dict with:
      i_crit           : float or np.nan -- point estimate, the median
                          of the bootstrap distribution
      i_crit_ci_low,
      i_crit_ci_high    : the CI_PERCENTILES interval of the bootstrap
                          distribution (NaN if it has no finite draws)
      frac_no_crossing  : fraction of bootstrap resamples with no
                          crossing at all
      boot_estimates    : the full array of n_boot bootstrap estimates
                          (NaN where that resample had no crossing),
                          kept for diagnostic histograms
      reason            : "ok" | "no_crossing"
    """
    i_tau_frac = np.asarray(i_tau_frac, dtype=float)
    r_inf_frac = np.asarray(r_inf_frac, dtype=float)
    n = len(i_tau_frac)

    boot_estimates = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_estimates[b] = _first_crossing(i_tau_frac[idx], r_inf_frac[idx], threshold)

    finite = boot_estimates[np.isfinite(boot_estimates)]
    frac_no_crossing = 1.0 - len(finite) / n_boot

    if len(finite) == 0:
        return dict(
            i_crit=np.nan, i_crit_ci_low=np.nan, i_crit_ci_high=np.nan,
            frac_no_crossing=1.0, boot_estimates=boot_estimates, reason="no_crossing",
        )

    ci_low, ci_high = np.percentile(finite, CI_PERCENTILES)
    return dict(
        i_crit=float(np.median(finite)),
        i_crit_ci_low=float(ci_low), i_crit_ci_high=float(ci_high),
        frac_no_crossing=frac_no_crossing, boot_estimates=boot_estimates, reason="ok",
    )


def run_all_combos(force_recompute=not USE_CACHE):
    """
    Run simulations for every (p, q, R0, tau, delay) combination in the
    current CONFIG, bootstrap the I_crit estimate for each, and return:
      results_df : one row per combo, with I_crit, its 90% CI, and the
                   no-crossing fraction
      raw_data   : dict keyed by (p, q, R0, tau, delay) with full
                   per-run arrays + the estimate_i_crit() output
                   (including the full bootstrap distribution)

    Caching is incremental (see module docstring): only combos from the
    current CONFIG that are missing from the cached raw_data get
    (re-)simulated; anything already cached is kept and merged in
    rather than the whole cache being used or discarded wholesale.
    """
    combos = [
        (p, q, R0, tau, delay)
        for (p, q) in PQ_COMBOS
        for R0 in R0_vals
        for tau in tau_vals
        for delay in delay_vals
    ]

    results_df = pd.DataFrame()
    raw_data = {}

    if not force_recompute and os.path.exists(RESULTS_CSV) and os.path.exists(RAW_CACHE_PATH):
        print(f"Loading cached results from {RESULTS_CSV} and {RAW_CACHE_PATH}")
        results_df = pd.read_csv(RESULTS_CSV)
        with open(RAW_CACHE_PATH, "rb") as f:
            raw_data = pickle.load(f)

    missing = [c for c in combos if c not in raw_data]

    if not missing:
        print(f"All {len(combos)} combos already cached -- nothing to (re-)run.")
        return results_df, raw_data

    if raw_data:
        print(f"{len(combos) - len(missing)}/{len(combos)} combos already cached; "
              f"running the remaining {len(missing)} x n_runs={n_runs} simulations each "
              f"(x n_boot={N_BOOT} bootstrap resamples)...")
    else:
        print(f"Running {len(missing)} parameter combinations x n_runs={n_runs} simulations each "
              f"(x n_boot={N_BOOT} bootstrap resamples)...")

    records = []
    total = len(missing)

    for idx, (p, q, R0, tau, delay) in enumerate(missing, start=1):
        beta = beta_from_R0(R0)
        gamma = GAMMA

        df = run_repeated_sims(
            N=N, k=k, beta=beta, gamma=gamma, rho=rho,
            tau=tau, delay=delay, p=p, q=q, tmax=tmax, n_runs=n_runs,
        )
        i_tau_frac, r_inf_frac = get_i_tau_and_r_inf(df, N)
        est = estimate_i_crit(i_tau_frac, r_inf_frac, threshold=THRESHOLD, n_boot=N_BOOT)

        records.append({
            "p": p, "q": q, "R0": R0, "beta": beta, "gamma": gamma,
            "tau": tau, "delay": delay,
            "I_crit": est["i_crit"],
            "I_crit_ci_low": est["i_crit_ci_low"], "I_crit_ci_high": est["i_crit_ci_high"],
            "frac_no_crossing": est["frac_no_crossing"],
            "reason": est["reason"], "n_runs": len(df),
        })
        raw_data[(p, q, R0, tau, delay)] = {
            "i_tau_frac": i_tau_frac, "r_inf_frac": r_inf_frac, **est,
        }

        status = (
            f"I_crit={est['i_crit']:.4f} [{est['i_crit_ci_low']:.4f}, {est['i_crit_ci_high']:.4f}]"
            if np.isfinite(est["i_crit"]) else f"I_crit=NaN ({est['reason']})"
        )
        print(f"[{idx}/{total}] p={p} q={q} R0={R0:.4f} tau={tau} delay={delay} -> {status}")

    new_results_df = pd.DataFrame(records)
    results_df = (
        pd.concat([results_df, new_results_df], ignore_index=True)
        if len(results_df) else new_results_df
    )

    results_df.to_csv(RESULTS_CSV, index=False)
    with open(RAW_CACHE_PATH, "wb") as f:
        pickle.dump(raw_data, f)
    print(f"Saved results to {RESULTS_CSV} and raw scatter data to {RAW_CACHE_PATH}")

    return results_df, raw_data


def plot_i_crit_heatmaps_for_delay(results_df, delay, pq_combos, tau_vals, R0_vals, save_path=None):
    """
    One figure per delay value: a 3x2 grid of heatmaps of I_crit
    (bootstrap median) across (tau, R0), one panel per (p, q) combo.
    Same layout as v1/v2.
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
        if np.all(np.isnan(values)) or np.nanmax(values) <= 0:
            # Nothing to scale by -- either every cell is masked, or every
            # I_crit in this panel is 0. Fall back to a small fixed range
            # rather than forcing vmax=0.5, which was previously washing
            # out real (and usually much smaller) I_crit values into a
            # single indistinguishable colour across the whole heatmap.
            vmax = 1e-3
        else:
            vmax = np.nanmax(values)

        sns.heatmap(
            pivot,
            ax=ax,
            mask=pivot.isna(),
            cmap=heatmap_cmap,
            vmin=0,
            vmax=vmax,
            cbar_kws={"label": r"$I_{crit}/N$ (bootstrap median)"},
        )
        ax.set_title(rf"$p={p}$, $q={q}$", fontsize=10)
        ax.set_xlabel(r"$R_0$")
        ax.set_ylabel(r"$\tau$")
        ax.set_xticklabels([f"{v:.3f}" for v in R0_vals], rotation=45, ha="right")
        ax.set_yticklabels([f"{v:.1f}" for v in tau_vals], rotation=0)
        # sharex/sharey hide interior tick labels by default -- force every
        # subplot to show its own R0/tau values, not just the edge row/column
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)

    for idx in range(len(pq_combos), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        rf"$I_{{crit}}$ (bootstrap median) across $(\tau, R_0)$ for each $(p,q)$ combo   [$\delta={delay}$]"
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")

    return fig, axes


def plot_diagnostic_bootstrap(i_tau_frac, r_inf_frac, boot, params, save_path=None):
    """
    Two-panel diagnostic for one combo:
      left  : raw (I(tau), R_inf) scatter with three things marked --
              the 5th percentile bound and 95th percentile bound
              labelled directly on the graph, and the actual I_crit
              value used (the bootstrap median) given as a legend
              entry rather than an on-graph label, so it doesn't
              clutter the scatter.
      right : histogram of the n_boot bootstrap I_crit estimates, with
              the same three positions marked for direct comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.scatter(i_tau_frac, r_inf_frac, s=25, alpha=0.5, edgecolor="none")
    ax.axhline(THRESHOLD, color=COLOR_THRESHOLD, linestyle="--", linewidth=1, alpha=0.7)

    if np.isfinite(boot["i_crit"]):
        ci_low, ci_high, i_crit = boot["i_crit_ci_low"], boot["i_crit_ci_high"], boot["i_crit"]
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        ax.axvspan(ci_low, ci_high, color="0.85", alpha=0.6, zorder=0)

        ax.axvline(ci_low, color=COLOR_LOW, linestyle=":", linewidth=1.6)
        ax.text(ci_low, 0.98, f"5th pct = {ci_low:.4f}", transform=trans,
                rotation=90, ha="right", va="top", color=COLOR_LOW, fontsize=8, fontweight="bold")

        ax.axvline(ci_high, color=COLOR_HIGH, linestyle=":", linewidth=1.6)
        ax.text(ci_high, 0.98, f"95th pct = {ci_high:.4f}", transform=trans,
                rotation=90, ha="left", va="top", color=COLOR_HIGH, fontsize=8, fontweight="bold")

        ax.axvline(i_crit, color=COLOR_ACTUAL, linestyle="--", linewidth=2,
                   label=rf"$I_{{crit}}$ used = {i_crit:.4f}")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    else:
        ax.text(0.97, 0.95, r"$I_{crit}$: no crossing in any resample",
                transform=ax.transAxes, ha="right", va="top", color=COLOR_ACTUAL, fontsize=9)

    ax.set_xlabel(r"$I(\tau)/N$")
    ax.set_ylabel(r"$R_\infty/N$")
    ax.set_ylim(-0.02, 1.06)
    ax.grid(alpha=0.3)
    ax.set_title("Raw scatter: 5th/95th pct bounds and value used")

    ax = axes[1]
    finite = boot["boot_estimates"][np.isfinite(boot["boot_estimates"])]
    if len(finite) > 0:
        ax.hist(finite, bins=30, color=COLOR_HIST, alpha=0.85)
        if np.isfinite(boot["i_crit"]):
            ax.axvline(boot["i_crit_ci_low"], color=COLOR_LOW, linestyle=":", linewidth=1.6)
            ax.axvline(boot["i_crit_ci_high"], color=COLOR_HIGH, linestyle=":", linewidth=1.6)
            ax.axvline(boot["i_crit"], color=COLOR_ACTUAL, linestyle="--", linewidth=2)
    ax.set_xlabel(r"bootstrap $I_{crit}$ estimate ($I(\tau)/N$)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Bootstrap distribution (n boot={len(boot['boot_estimates'])}, "
        f"no-crossing frac={boot['frac_no_crossing']:.2f})"
    )
    ax.grid(alpha=0.3)

    fig.suptitle(
        rf"$R_0$={params['R0']:.3f}, $\tau$={params['tau']}, $\delta$={params['delay']}, "
        rf"$p$={params['p']}, $q$={params['q']}"
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"Saved diagnostic figure to {save_path}")

    return fig, axes


if __name__ == "__main__":

    results_df, raw_data = run_all_combos()

    for delay in delay_vals:
        fig, _ = plot_i_crit_heatmaps_for_delay(
            results_df, delay, PQ_COMBOS, tau_vals, R0_vals,
            save_path=f"i_crit_heatmap_v3_delay{delay}.png",
        )
        plt.close(fig)

    # Save diagnostics as one PDF per (p, q), with one page per R0
    diagnostic_dir = "i_crit_diagnostics_v3"
    os.makedirs(diagnostic_dir, exist_ok=True)

    for p, q in PQ_COMBOS:
        pdf_path = f"{diagnostic_dir}/p{p}_q{q}.pdf"

        with PdfPages(pdf_path) as pdf:
            for R0 in R0_vals:
                fig, axes = plt.subplots(
                    len(tau_vals), len(delay_vals),
                    figsize=(24, 12)
                )
                axes = np.asarray(axes).reshape(len(tau_vals), len(delay_vals))

                for i, tau in enumerate(tau_vals):
                    for j, delay in enumerate(delay_vals):
                        r = raw_data[(p, q, R0, tau, delay)]

                        diag_fig, _ = plot_diagnostic_bootstrap(
                            r["i_tau_frac"], r["r_inf_frac"], r,
                            params=dict(R0=R0, tau=tau, delay=delay, p=p, q=q)
                        )

                        diag_fig.canvas.draw()
                        img = np.asarray(diag_fig.canvas.buffer_rgba())
                        axes[i, j].imshow(img)
                        axes[i, j].axis("off")
                        diag_fig.clf()
                        plt.close(diag_fig)

                fig.suptitle(
                    rf"$p={p}$, $q={q}$, $R_0={R0}$",
                    fontsize=18
                )
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"Saved {pdf_path}")