"""
Estimate critical delay δ where P(R_inf < 0.5)
Critical delay:
largest δ where P(R_inf < 0.5) >= 0.5
"""
import numpy as np
import pandas as pd
from multiprocessing import Pool
import itertools
import sys
MODULE_DIR = "/Users/joycewilliamslt/Documents/GitHub/Oxford_Maths01"
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)
from Oxford_Maths01.Hybrid_Network_Model.hybrid_network_model_v1 import run_repeated_sims
MODULE_DIR2 = "/Users/joycewilliamslt/Documents/GitHub/Oxford_Maths01"
if MODULE_DIR2 not in sys.path:
    sys.path.append(MODULE_DIR2)
from Oxford_Maths01.plotting_style import set_plot_style
set_plot_style()
threshold = 0.5
def worker(args):
    N,k,gamma,rho,R0,tau,delay,p,q,tmax,n_runs,count,total = args
    beta = R0*gamma
    df = run_repeated_sims(N,k,beta,gamma,rho,tau,delay,p,q,tmax,n_runs)
    prob_dieout = (df["final_epidemic_fraction"] < threshold).mean()
    print(
        f"[{count}/{total}] "
        f"R0={R0}, tau={tau}, δ={delay}, "
        f"p={p}, q={q}, P={prob_dieout:.3f}")
    return {"R0":R0, "tau":tau, "delay":delay, "p":p, "q":q, "prob_dieout":prob_dieout,
        "mean_R_inf":df["final_epidemic_fraction"].mean()}
def extinction_sweep(N,k,gamma,rho,R0_vals,tau_vals,delay_vals,pqs,tmax,n_runs,workers=6):
    combos = []
    total = len(R0_vals)*len(tau_vals)*len(delay_vals)*len(pqs)
    count = 1
    for R0,tau,delay,(p,q) in itertools.product(R0_vals,tau_vals,delay_vals,pqs):
        combos.append((N,k,gamma,rho, R0,tau,delay,p,q,tmax,n_runs,count,total))
        count += 1
    with Pool(workers) as pool:
        results = pool.map(worker,combos)
    return pd.DataFrame(results)
def find_critical_delay(df, threshold=0.5, interpolate=True):
    critical = []
    for keys, group in df.groupby(["p", "q", "tau", "R0"]):
        group = group.sort_values("delay")
        delays = group["delay"].to_numpy()
        probs = group["prob_dieout"].to_numpy()
        below = np.where(probs < threshold)[0]
        if len(below) == 0:
            # never crosses within the tested delay range
            delta_c, status = np.nan, "always_dieout"
        elif below[0] == 0:
            # already below threshold at the smallest delay tested
            delta_c, status = delays[0], "always_outbreak"
        else:
            i = below[0]
            d_lo, d_hi = delays[i - 1], delays[i]
            p_lo, p_hi = probs[i - 1], probs[i]
            if interpolate and p_lo != p_hi:
                delta_c = d_lo + (threshold - p_lo) * (d_hi - d_lo) / (p_hi - p_lo)
            else:
                delta_c = d_hi
            status = "crossed"
        critical.append({
            "p": keys[0], "q": keys[1], "tau": keys[2], "R0": keys[3],
            "critical_delay": delta_c, "status": status,
        })
    return pd.DataFrame(critical)


if __name__ == "__main__":
    # params
    # (p,q) pairs corresponding to regions on heatmaps
    # pqs = [(0.90, 0.22), (0.84, 0.44), (0.52, 0.09), (0.28, 0.31)] # original pq values from heatmap
    pqs = [(0.945, 0.25), (0.84, 0.44), (0.615,0.415), (0.275, 0.47), (0.055, 0.36), (0.5, 0.305)]

    R0_vals = [1.125, 1.25, 1.3125, 1.375, 1.4375, 1.5, 1.75]
    tau_vals = [2.5, 3, 3.5, 4]
    delay_vals = [0,0.25,0.5,0.75,1,1.25,1.5,2]
    threshold = 0.5
    results = extinction_sweep(
        N=1000, k=3, gamma=1, rho=0.05, R0_vals=R0_vals,tau_vals=tau_vals, delay_vals=delay_vals,
        pqs=pqs, tmax=100, n_runs=1000, workers=6)
    results.to_csv("extinction_probability_results6"
                   ".csv", index=False)
    critical = find_critical_delay(results)
    critical.to_csv("critical_delay_results6.csv", index=False)
    print("\nFinished")
    print(critical)