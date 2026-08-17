"""
Plot epidemic size at intervention time, R(tau), against final epidemic
size, R_inf, for repeated stochastic runs at a fixed parameter combination.

Motivation
----------
If there is a threshold in the early growth trajectory (i.e. the process
behaves like a branching process near tau, with a nonzero extinction
probability), then runs should separate into two distinct branches:

  - a "fade-out" branch: R(tau) stays small AND R_inf stays small
  - an "escape" branch:  R(tau) is large enough that the epidemic has
                          already escaped stochastic extinction, and
                          R_inf converges towards the large deterministic
                          outbreak size

A smooth/unimodal cloud with no gap would suggest no such threshold at
this (p, q, tau, delay) combination; a visible gap or two clusters
supports the branching-process / extinction-probability interpretation.

Usage
-----
Edit the PARAMETER COMBINATION block below, then run:
    python plot_r_tau_vs_r_inf.py
"""

import sys

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

# set plotting style
try:
    from Oxford_Maths01.plotting_style import set_plot_style
    set_plot_style()
except ImportError:
    plt.style.use('seaborn-v0_8-whitegrid')

import matplotlib.pyplot as plt
import numpy as np


def get_r_tau_and_r_inf(df, N):
    """
    Extract R(tau) and R_inf as fractions of N from a run_repeated_sims
    output DataFrame.

    R(tau) here is taken as the number of nodes infected at the moment
    intervention is applied (infected_at_intervention), which is the
    quantity the branching-process argument concerns — i.e. how far the
    early outbreak has grown by the time control acts.
    """
    r_tau_frac = df["infected_at_intervention"].to_numpy() / N
    r_inf_frac = df["final_epidemic_fraction"].to_numpy()
    return r_tau_frac, r_inf_frac


def plot_scatter(r_tau_frac, r_inf_frac, params, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        r_tau_frac,
        r_inf_frac,
        s=25,
        alpha=0.5,
        edgecolor="none",
    )

    ax.set_xlabel(r"Epidemic size at intervention, $R(\tau)/N$")
    ax.set_ylabel(r"Final epidemic size, $R_\infty/N$")
    ax.set_title(
        r"$R(\tau)$ vs $R_\infty$"
        f"\n"
        rf"$\beta$={params['beta']}, $\gamma$={params['gamma']}, "
        rf"$\tau$={params['tau']}, $\delta$={params['delay']}, "
        rf"$p$={params['p']}, $q$={params['q']}, n={params['n_runs']}"
    )
    ax.set_xlim(-0.02, max(0.05, r_tau_frac.max() * 1.1))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")

    return fig, ax


if __name__ == "__main__":

    # Fixed parameters
    N = 1000
    k = 3
    rho = 0.05
    tmax = 100
    n_runs = 500

    # Different R0 regimes
    beta_gamma_pairs = [
        (0.20, 0.4),    # R0 = 0.5
        (0.20, 0.20),  # R0 = 1
        (0.40, 0.20),  # R0 = 2
        (0.60, 0.20),  # R0 = 3
    ]

    # (p, q, tau, delay)
    parameter_sets = [
        (1.0, 0.10, 2.0, 0.0),  # Weak, very early intervention
        (1.0, 0.50, 2.0, 0.0),  # Strong, very early intervention
        (1.0, 0.30, 3.0, 0.0),  # Intermediate timing (likely near threshold)
        (1.0, 0.30, 4.0, 0.0),  # Slightly later intervention
        (1.0, 0.30, 5.0, 0.0),  # Late intervention
        (1.0, 0.30, 3.0, 0.1),  # Small reporting delay
        (1.0, 0.30, 3.0, 0.25),  # Moderate reporting delay
        (1.0, 0.30, 3.0, 0.5),  # Large reporting delay
        (0.5, 0.30, 3.0, 0.0),  # Imperfect targeting
        (0.0, 0.30, 3.0, 0.0),  # Completely random intervention
    ]

    fig, axes = plt.subplots(
        len(parameter_sets),
        len(beta_gamma_pairs),
        figsize=(5 * len(beta_gamma_pairs), 4 * len(parameter_sets)),
        sharex=True,
        sharey=True,
    )

    if len(parameter_sets) == 1:
        axes = np.array([axes])

    if len(beta_gamma_pairs) == 1:
        axes = axes[:, np.newaxis]

    for i, (p, q, tau, delay) in enumerate(parameter_sets):

        for j, (beta, gamma) in enumerate(beta_gamma_pairs):

            df = run_repeated_sims(
                N=N,
                k=k,
                beta=beta,
                gamma=gamma,
                rho=rho,
                tau=tau,
                delay=delay,
                p=p,
                q=q,
                tmax=tmax,
                n_runs=n_runs,
            )

            print(
                beta, gamma,
                "rows =", len(df),
                "missing R(tau) =", df["infected_at_intervention"].isna().sum(),
            )

            print(df[["infected_at_intervention", "final_epidemic_fraction"]].describe())

            r_tau_frac, r_inf_frac = get_r_tau_and_r_inf(df, N)

            ax = axes[i, j]

            ax.scatter(
                r_tau_frac,
                r_inf_frac,
                s=20,
                alpha=0.5,
                edgecolor="none",
            )

            R0 = beta / gamma

            ax.set_title(
                rf"$R_0={R0:.1f}$"
                "\n"
                rf"$\beta={beta},\ \gamma={gamma}$",
                fontsize=10,
            )

            if j == 0:
                ax.set_ylabel(
                    rf"$p={p}$" "\n"
                    rf"$q={q}$" "\n"
                    rf"$\tau={tau},\ \delta={delay}$"
                )

            if i == len(parameter_sets) - 1:
                ax.set_xlabel(r"$R(\tau)/N$")

            ax.set_xlim(-0.02, 0.55)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)

            # Print summaries
            print(
                f"(p={p}, q={q}, tau={tau}, delay={delay}, "
                f"beta={beta}, gamma={gamma}, R0={R0:.1f})"
            )
            print(
                f"R(tau): mean={r_tau_frac.mean():.4f}, "
                f"R_inf: mean={r_inf_frac.mean():.4f}"
            )

    fig.supylabel(r"$R_\infty/N$")
    fig.supxlabel(r"$R(\tau)/N$")
    fig.tight_layout()

    plt.savefig("r_tau_vs_r_inf_grid.png", dpi=300)
    plt.show()

