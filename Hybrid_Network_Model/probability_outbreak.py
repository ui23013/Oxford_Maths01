"""
Calculate probability of a large epidemic branch.

P_large = P(R_inf > 0.35)

Only small reporting delays are considered.
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

import sys

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


set_plot_style()


threshold = 0.35


def worker(args):

    N, k, beta, gamma, rho, tau, delay, p, q, tmax, n_runs, count, total = args

    df = run_repeated_sims(N, k, beta, gamma, rho, tau, delay, p, q, tmax, n_runs)

    prob_large = (df["final_epidemic_fraction"] > threshold).mean()

    print(f"[{count}/{total}] tau={tau}, δ={delay}, p={p:.2f}, q={q:.2f}, P={prob_large:.3f}")

    return {
        "tau": tau,
        "delay": delay,
        "p": p,
        "q": q,
        "prob_large": prob_large,
        "mean_R_inf": df["final_epidemic_fraction"].mean(),
        "std_R_inf": df["final_epidemic_fraction"].std()
    }


def probability_sweep(N, k, beta, gamma, rho, tau_vals, delay_vals,
                      p_vals, q_vals, tmax, n_runs=500, workers=6):

    combos = [
        (N, k, beta, gamma, rho, tau, delay, p, q, tmax, n_runs, i+1,
         len(tau_vals)*len(delay_vals)*len(p_vals)*len(q_vals))
        for i, (tau, delay, p, q) in enumerate(
            (x for x in __import__("itertools").product(
                tau_vals, delay_vals, p_vals, q_vals)))]

    with Pool(workers) as pool:
        results = pool.map(worker, combos)

    return pd.DataFrame(results)


if __name__ == "__main__":

    results = probability_sweep(
        N=1000,
        k=3,
        beta=0.4,
        gamma=0.2,
        rho=0.05,
        tau_vals=[2, 4, 6],
        delay_vals=[0, 0.5, 1],   # small delays only
        p_vals=np.linspace(0, 1, 10),
        q_vals=np.linspace(0, 0.5, 10),
        tmax=100,
        n_runs=250,
        workers=6
    )

    results.to_csv("probability_large_outbreak_results.csv", index=False)

    print("\nFinished")
    print(results.head())