"""
Compare outcome distributions across several reporting-delay (delta) values,
for fixed p, q, tau, beta, gamma.

Reuses run_repeated_sims from hybrid_network_model_v1.py.

Note on what actually varies with delta in this model:
    - `infected_at_intervention` (I(tau)) is determined purely by the
      baseline trajectory up to tau, which does NOT depend on delta ---
      delta only changes which edges get cut (via reported_infected,
      drawn from the state at tau - delta). So its distribution is
      essentially delta-invariant (up to simulation noise).
    - `final_epidemic_fraction` (or final_epidemic_size) DOES vary with
      delta, since a larger delta means staler information is used to
      target the edge removal, generally making the intervention less
      effective and shifting/widening the final-size distribution.
    For that reason the default metric below is final_epidemic_fraction.
    Change METRIC to "infected_at_intervention" if you want to confirm
    that behaviour for yourself.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from hybrid_network_model_v1 import run_repeated_sims

# set plotting style
try:
    from Oxford_Maths01.plotting_style import set_plot_style
    set_plot_style()
except ImportError:
    plt.style.use('seaborn-v0_8-whitegrid')


def sweep_over_delays(N, k, beta, gamma, rho, tau, p, q, tmax,
                      delta_values, n_runs, metric):
    """Run repeated simulations at each delta and collect the metric column."""
    results_by_delta = {}
    for delta in delta_values:
        print(f"Running delta={delta} ({n_runs} runs)...")
        df = run_repeated_sims(
            N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
            delay=delta, p=p, q=q, tmax=tmax, n_runs=n_runs,
        )
        results_by_delta[delta] = df[metric].values
    return results_by_delta


def plot_delay_histograms(results_by_delta, metric, tau, p, q, beta, gamma,
                          bins=25, save_path="delay_histograms.png"):
    """Overlay histograms for each delta value on a single set of axes."""
    fig, ax = plt.subplots(figsize=(9, 6))

    all_values = np.concatenate(list(results_by_delta.values()))
    bin_edges = np.linspace(all_values.min(), all_values.max(), bins + 1)

    # Get the current color cycle from rcParams (set by your style)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (delta, values) in enumerate(results_by_delta.items()):
        ax.hist(
            values,
            bins=bin_edges,
            alpha=0.5,
            label=fr"$\delta={delta:g}$",
            color=colors[i % len(colors)],   # explicitly use style colours
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel(metric.replace("_", " ").capitalize())
    ax.set_ylabel("Count")
    ax.set_title(
        fr"Distribution of {metric.replace('_', ' ')} vs $\delta$"
        fr"  ($\tau={tau}$, $p={p}$, $q={q}$, $\beta={beta}$, $\gamma={gamma}$)"
    )
    ax.legend(title=r"Reporting delay $\delta$")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    return fig


# =============================================================================
# NEW FUNCTION: run a grid of parameter combinations
# =============================================================================
def run_parameter_sweep(
    N, k, rho, tmax,
    delta_values,
    beta_values, gamma_values, tau_values, p_values, q_values,
    n_runs=50,
    metric="final_epidemic_fraction",
    bins=25,
    base_save_path="delay_histograms",
    show_plots=False
):
    """
    Run the delay‑sweep over a grid of parameter combinations and save
    a separate histogram plot for each combination.

    Parameters
    ----------
    N, k, rho, tmax : fixed model parameters (same for all runs)
    delta_values : list of delays to compare
    beta_values, gamma_values, tau_values, p_values, q_values : lists
        of parameter values to sweep over.
    n_runs : number of repeated simulations per (delta, parameter combo)
    metric : which column to histogram (e.g., 'final_epidemic_fraction')
    bins : number of histogram bins
    base_save_path : prefix for saved plot filenames
    show_plots : if True, display each figure interactively (blocks until closed)
    """
    # Generate all combinations
    param_combos = list(product(beta_values, gamma_values, tau_values,
                                p_values, q_values))

    total_combos = len(param_combos)
    print(f"Total parameter combinations to process: {total_combos}")

    for idx, (beta, gamma, tau, p, q) in enumerate(param_combos, 1):
        print(f"\n--- Combination {idx}/{total_combos}: "
              f"beta={beta}, gamma={gamma}, tau={tau}, p={p}, q={q} ---")

        # Run the sweep for this parameter set
        results = sweep_over_delays(
            N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
            p=p, q=q, tmax=tmax,
            delta_values=delta_values, n_runs=n_runs, metric=metric
        )

        # Create a unique filename
        save_path = (f"{base_save_path}_b{beta}_g{gamma}_t{tau}"
                     f"_p{p}_q{q}.png")
        fig = plot_delay_histograms(
            results, metric, tau, p, q, beta, gamma,
            bins=bins, save_path=save_path
        )

        if show_plots:
            plt.show()
        else:
            plt.close(fig)   # free memory if not showing


# =============================================================================
# Example usage (commented out – uncomment and edit to run your own sweeps)
# =============================================================================
if __name__ == "__main__":
    # ---- Example: single parameter set (original behaviour) ----
    N = 1000
    k = 3
    beta = 0.4
    gamma = 0.2
    rho = 0.05
    tau = 2.0
    p = 0.9
    q = 0.25
    tmax = 100
    n_runs = 10
    delta_values = [0.0, 0.5, 1.0, 2.0]
    METRIC = "final_epidemic_fraction"

    # Uncomment to run the original single sweep:
    # results = sweep_over_delays(
    #     N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau, p=p, q=q,
    #     tmax=tmax, delta_values=delta_values, n_runs=n_runs, metric=METRIC,
    # )
    # plot_delay_histograms(results, METRIC, tau, p, q, beta, gamma)
    # plt.show()

    # run a parameter sweep over multiple values
    # Example: vary beta and tau, keep others fixed
    beta_values = [0.4]
    gamma_values = [0.1, 0.2, 0.3]
    tau_values = [2.0, 4.0, 6.0]
    p_values = [0.9, 0.84, 0.52, 0.28]              # single value
    q_values = [0.22, 0.44, 0.31, 0.9]             # single value

    run_parameter_sweep(
        N=N, k=k, rho=rho, tmax=tmax,
        delta_values=delta_values,
        beta_values=beta_values,
        gamma_values=gamma_values,
        tau_values=tau_values,
        p_values=p_values,
        q_values=q_values,
        n_runs=n_runs,
        metric=METRIC,
        bins=25,
        base_save_path="delay_histograms_sweep",
        show_plots=False          # set True to see each plot as it is generated
    )