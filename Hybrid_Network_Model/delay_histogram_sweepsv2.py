"""
Compare outcome distributions across several reporting-delay (delta) values,
for fixed p, q, tau, beta, gamma.

Reuses run_repeated_sims from hybrid_network_model_v1.py.
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

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (delta, values) in enumerate(results_by_delta.items()):
        ax.hist(
            values,
            bins=bin_edges,
            alpha=0.5,
            label=fr"$\delta={delta:g}$",
            color=colors[i % len(colors)],
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
# NEW FUNCTION: run a custom list of parameter combinations
# =============================================================================
def run_custom_combinations(
    combinations,
    N, k, rho, tmax,
    delta_values,
    n_runs=50,
    metric="final_epidemic_fraction",
    bins=25,
    base_save_path="delay_histograms",
    show_plots=False
):
    """
    Run the delay‑sweep for each parameter combination in a user‑provided list.

    Parameters
    ----------
    combinations : list of dicts
        Each dict must contain keys: 'beta', 'gamma', 'tau', 'p', 'q'.
        Example: [{'beta':0.4, 'gamma':0.2, 'tau':2, 'p':0.9, 'q':0.25}, ...]
    N, k, rho, tmax : fixed model parameters
    delta_values : list of delays to compare
    n_runs : number of repeated simulations per (delta, parameter combo)
    metric : which column to histogram
    bins : number of histogram bins
    base_save_path : prefix for saved plot filenames
    show_plots : if True, display each figure interactively
    """
    total = len(combinations)
    print(f"Total custom combinations to process: {total}")

    for idx, params in enumerate(combinations, 1):
        beta = params['beta']
        gamma = params['gamma']
        tau = params['tau']
        p = params['p']
        q = params['q']

        print(f"\n--- Combination {idx}/{total}: "
              f"beta={beta}, gamma={gamma}, tau={tau}, p={p}, q={q} ---")

        results = sweep_over_delays(
            N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
            p=p, q=q, tmax=tmax,
            delta_values=delta_values, n_runs=n_runs, metric=metric
        )

        save_path = (f"{base_save_path}_b{beta}_g{gamma}_t{tau}"
                     f"_p{p}_q{q}.png")
        fig = plot_delay_histograms(
            results, metric, tau, p, q, beta, gamma,
            bins=bins, save_path=save_path
        )

        if show_plots:
            plt.show()
        else:
            plt.close(fig)


# =============================================================================
# Example usage with your specific combinations
# =============================================================================
if __name__ == "__main__":
    # ---- Fixed parameters (edit as needed) ----
    N = 1000
    k = 3
    rho = 0.05
    tmax = 100
    n_runs = 1000

    # ---- Delay values to compare ----
    delta_values = [0.0, 0.5, 1.0, 1.5, 2.0]

    # ---- Metric to histogram ----
    METRIC = "final_epidemic_fraction"

    # ---- Fixed beta and gamma (you can change if needed) ----
    fixed_beta = 0.4
    fixed_gamma = 0.2

    # ---- Your custom (p, q) pairs and tau values ----
    p_q_pairs = [(0.9, 0.22), (0.84, 0.44), (0.52, 0.09), (0.28, 0.31)]
    tau_values = [2, 4, 6]

    # Build the list of combinations
    combinations = []
    for tau in tau_values:
        for p, q in p_q_pairs:
            combinations.append({
                'beta': fixed_beta,
                'gamma': fixed_gamma,
                'tau': tau,
                'p': p,
                'q': q
            })

    # Run the custom sweep
    run_custom_combinations(
        combinations=combinations,
        N=N, k=k, rho=rho, tmax=tmax,
        delta_values=delta_values,
        n_runs=n_runs,
        metric=METRIC,
        bins=25,
        base_save_path="delay_histograms_custom",
        show_plots=False   # set True to see plots as they are generated
    )