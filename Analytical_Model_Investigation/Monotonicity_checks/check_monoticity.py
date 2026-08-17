"""
check_Rinf_monotonicity.py

Standalone diagnostic: checks whether mean_R_inf is monotonically
non-decreasing in delay, for every (p, q, tau, R0) group in the sweep
CSV.

This is the assumption estimate_delta_crit_Rinf.py depends on: delta_c
there is defined as the *first* delay at which mean_R_inf crosses above
0.5, which only identifies a well-defined "critical delay" if R_inf
doesn't dip back down below 0.5 again later along the curve. This
script checks that assumption directly, rather than just visually
dashing suspect segments inside a full results plot.

Reuses load_results() from estimate_delta_critv2.py's pipeline
autodetection logic, and a stricter (default tol=0) version of its
check_monotonicity() function -- here also reporting how many
individual delay-steps violate monotonicity, not just the worst one,
plus a histogram of drop sizes across all groups to help calibrate a
sensible tolerance.

Outputs:
  - Rinf_monotonicity_flags.csv       one row per (p,q,tau,R0), with
                                       is_monotonic, max_drop,
                                       n_violating_steps
  - Rinf_monotonicity_drops_hist.png  histogram of max_drop across all
                                       groups, to help judge whether any
                                       violations are noise-sized or real
  - console summary, plus the full delay -> mean_R_inf trace printed
    for the single worst-offending group

Does NOT re-run simulations -- reads whichever raw sweep CSV is present
in the working directory:

  - outbreak_probability_results.csv  (columns include prob_outbreak)
    from outbreak_probability_sweep.py, or
  - extinction_probability_results.csv  (columns include prob_dieout)
    from the extinction/dieout-style sweep script.

Whichever is found is printed at runtime, so a stale file from a
different parameter grid can't get used silently. If BOTH files are
present, outbreak_probability_results.csv takes priority -- delete or
rename whichever one you don't want picked up.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

MODULE_DIR = "/Oxford_Maths01"
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)
from Oxford_Maths01.plotting_style import set_plot_style

set_plot_style()

VALUE_COL = "mean_R_inf"
GROUP_COLS = ["p", "q", "tau", "R0"]

# Tolerance for the strict pass/fail flag: a drop smaller than this (in
# mean_R_inf units) between consecutive delay points is treated as
# sampling noise rather than a genuine non-monotonic trend. Default 0.0
# means ANY decrease counts as a violation -- tighten/loosen this after
# inspecting the histogram this script produces (Rinf_monotonicity_drops_hist.png).
MONOTONICITY_TOL = 0.005


def load_results():
    """
    Autodetects which sweep pipeline produced the data sitting in the
    working directory, and returns (df, prob_col, direction). Identical
    to estimate_delta_critv2.load_results -- kept here so this script
    runs standalone without importing v2.
    """
    if os.path.exists("outbreak_probability_results.csv"):
        path, prob_col, direction = "outbreak_probability_results234.csv", "prob_outbreak", "above"
    elif os.path.exists("../Critical_delay_estimation/extinction_probability_results.csv"):
        path, prob_col, direction = "extinction_probability_234.csv", "prob_dieout", "below"
    else:
        raise FileNotFoundError(
            "Neither outbreak_probability_results.csv nor "
            "extinction_probability_results.csv found in the working "
            "directory -- run the matching sweep script first."
        )

    df = pd.read_csv(path)
    print(f"Using {path} (probability column: {prob_col}, direction: {direction})")
    return df, prob_col, direction


def check_monotonicity(df, value_col=VALUE_COL, group_cols=GROUP_COLS, tol=MONOTONICITY_TOL):
    """
    For each group (sorted by delay), flag whether value_col is
    monotonically non-decreasing, within tolerance tol.

    max_drop:          largest single-step decrease observed (<=0; 0
                        means no decrease anywhere) -- magnitude
                        indicates how severe the worst dip is.
    n_violating_steps: how many individual delay-to-delay steps dropped
                        by more than tol -- distinguishes a single bad
                        step from a genuinely wobbly curve, which
                        max_drop alone can't.
    """
    flags = []
    for keys, group in df.groupby(group_cols):
        group = group.sort_values("delay")
        vals = group[value_col].to_numpy()
        diffs = np.diff(vals)
        max_drop = diffs.min() if len(diffs) else 0.0
        n_violating_steps = int(np.sum(diffs < -tol))
        is_monotonic = bool(max_drop >= -tol)
        flags.append({
            **dict(zip(group_cols, keys)),
            "is_monotonic": is_monotonic,
            "max_drop": max_drop,
            "n_violating_steps": n_violating_steps,
            "n_delay_points": len(vals),
        })
    return pd.DataFrame(flags)


def main():
    results, prob_col, direction = load_results()

    flags = check_monotonicity(results)
    flags.to_csv("Rinf_monotonicity_flags.csv", index=False)

    n_groups = len(flags)
    n_violations = int((~flags["is_monotonic"]).sum())
    print(
        f"\n{n_violations} / {n_groups} (p,q,tau,R0) groups show a mean "
        f"R_inf decrease > {MONOTONICITY_TOL} between consecutive delay "
        f"points (tol={MONOTONICITY_TOL})."
    )

    print("\nmax_drop distribution across all groups:")
    print(flags["max_drop"].describe().to_string())

    if n_violations:
        n_show = min(10, n_violations)
        print(f"\nWorst {n_show} offending groups (by max_drop):")
        worst = flags[~flags["is_monotonic"]].sort_values("max_drop").head(n_show)
        print(worst.to_string(index=False))

        print("\nFull delay -> mean_R_inf trace for the single worst-offending group:")
        p, q, tau, R0 = worst.iloc[0][GROUP_COLS]
        trace = results[
            (results["p"] == p) & (results["q"] == q) &
            (results["tau"] == tau) & (results["R0"] == R0)
        ].sort_values("delay")[["delay", VALUE_COL]]
        print(trace.to_string(index=False))
    else:
        print(
            "\nNo violations found -- mean_R_inf is monotonic non-decreasing "
            "in delay for every (p,q,tau,R0) group at this tolerance."
        )

    # histogram of drop sizes, to help calibrate a sensible tolerance for
    # both this script and MONOTONICITY_TOL in estimate_delta_critv2.py
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(flags["max_drop"], bins=30, color="goldenrod", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="no decrease")
    ax.set_xlabel(r"largest single-step drop in $R_\infty$ (per group)")
    ax.set_ylabel(r"number of ($p, q, \tau, R_0$) groups")
    ax.set_title(r"Distribution of the worst $R_\infty$ decrease per curve")
    ax.legend()
    fig.savefig("Rinf_monotonicity_drops_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        "\nWrote Rinf_monotonicity_flags.csv and "
        "Rinf_monotonicity_drops_hist.png to the working directory.")


if __name__ == "__main__":
    main()