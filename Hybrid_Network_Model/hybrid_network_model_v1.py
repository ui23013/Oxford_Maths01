# Hybrid SIR network model with reporting delay and incomplete information
#
# Combines:
# 1. Reporting delay (δ): Intervention happens at time τ, but available infection
#                         data come from τ - δ
# 2. Incomplete information (p, q): Edges are removed with probability p of being
#                                   adjacent to an infected node, and q is the
#                                   proportion of total edges removed.
#
# At time τ - δ, we identify infected nodes from the baseline trajectory.
# At time τ, we intervene by removing edges:
#   - With probability p, preferentially choose edges adjacent to these reported nodes
#   - Otherwise (1-p), choose random edges
#   - Total edges removed = q × total_edges

import random
from multiprocessing import Pool
from itertools import product

import EoN
import networkx as nx
import numpy as np
import pandas as pd


def create_rrg(k, N, seed):
    """Create a random regular graph with k-regular degree."""
    return nx.random_regular_graph(k, N, seed=seed)


def run_first_sim(graph, beta, gamma, rho, tau, seed):
    """
    Run simulation until first intervention time and return infected
    and recovered nodes at time tau.
    """
    random.seed(seed)
    np.random.seed(seed)

    sim = EoN.Gillespie_SIR(
        graph, beta, gamma, rho=rho, tmax=tau, return_full_data=True
    )

    # Extract states at intervention time
    statuses = sim.get_statuses(time=tau)

    infected_nodes = [node for node, status in statuses.items() if status == "I"]
    recovered_nodes = [node for node, status in statuses.items() if status == "R"]

    return infected_nodes, recovered_nodes, statuses, sim


def edge_removal_with_delay(graph, beta, gamma, p, q, reported_infected,
                            infected_at_tau, tau, seed):
    """
    Remove edges from the graph based on information from reported infected nodes.

    Args:
        graph: NetworkX graph
        beta: transmission rate
        gamma: recovery rate
        p: probability of choosing edges adjacent to reported infected nodes
        q: proportion of total edges to remove
        reported_infected: list of nodes reported as infected at time τ - δ
        infected_at_tau: list of nodes actually infected at time τ
        tau: intervention time
        seed: random seed

    Returns:
        modified_graph: graph after edge removals
        removed_edges_count: number of edges removed
        correct_count: nodes correctly identified as infected
        stale_count: nodes reported but no longer infected
        missed_count: nodes infected but not reported
    """
    randgen = random.Random(seed)

    # Total edges in original graph
    total_edges = graph.number_of_edges()
    removal_count = round(q * total_edges)

    if removal_count == 0:
        return graph, 0, 0, 0, 0

    # Sets for overlap analysis
    reported_set = set(reported_infected)
    infected_set = set(infected_at_tau)

    correct = reported_set & infected_set
    stale = reported_set - infected_set
    missed = infected_set - reported_set

    # Perform edge removal
    G_new = graph.copy()
    edges_removed = 0

    for _ in range(removal_count):
        current_edges = list(G_new.edges())

        if not current_edges:
            break

        # Edges adjacent to reported nodes
        informed_edges = [
            edge for edge in current_edges
            if edge[0] in reported_set or edge[1] in reported_set
        ]

        # Choose edge to remove
        if informed_edges and randgen.random() < p:
            removed_edge = randgen.choice(informed_edges)
        else:
            removed_edge = randgen.choice(current_edges)

        G_new.remove_edge(*removed_edge)
        edges_removed += 1

    return G_new, edges_removed, len(correct), len(stale), len(missed)


def run_second_sim(graph, beta, gamma, infected_nodes, recovered_nodes,
                   tau_start, tmax, seed):
    """
    Run simulation from tau to tmax with modified graph.
    """
    random.seed(seed)
    np.random.seed(seed)

    sim = EoN.Gillespie_SIR(
        graph,
        beta,
        gamma,
        initial_infecteds=infected_nodes,
        initial_recovereds=recovered_nodes,
        tmin=tau_start,
        tmax=tmax,
        return_full_data=True
    )

    return sim


def join_trajectories(first_sim, second_sim, tau):
    """Join two simulation trajectories at time tau."""
    t1, data1 = first_sim.summary()
    t2, data2 = second_sim.summary()

    keep = t1 < tau

    t = np.concatenate((t1[keep], t2))
    S = np.concatenate((data1["S"][keep], data2["S"]))
    I = np.concatenate((data1["I"][keep], data2["I"]))
    R = np.concatenate((data1["R"][keep], data2["R"]))

    return t, S, I, R


def run_hybrid_intervention(N, k, beta, gamma, rho, tau, delay, p, q,
                           tmax, run_number, seed):
    """
    Run a single hybrid intervention simulation.
    """
    # Create graph
    G = create_rrg(k, N, seed=seed)
    initial_edges = G.number_of_edges()

    # Run baseline simulation to tau
    infected_tau, recovered_tau, statuses_tau, baseline_sim = run_first_sim(
        G, beta, gamma, rho, tau, seed=seed + 1000
    )

    # Get reported infected nodes (from time τ - δ)
    report_time = max(0.0, tau - delay)
    report_statuses = baseline_sim.get_statuses(time=report_time)
    reported_infected = [
        node for node, status in report_statuses.items() if status == "I"
    ]

    # Apply edge removal based on reported data
    G_intervened, edges_removed, correct, stale, missed = edge_removal_with_delay(
        G, beta, gamma, p, q, reported_infected, infected_tau, tau,
        seed=seed + 2000
    )

    # Run continuation simulation from tau to tmax
    if infected_tau:
        continuation = run_second_sim(
            G_intervened, beta, gamma, infected_tau, recovered_tau,
            tau_start=tau, tmax=tmax, seed=seed + 3000
        )
        t, S, I, R = join_trajectories(baseline_sim, continuation, tau)
    else:
        t, data = baseline_sim.summary()
        S, I, R = data["S"], data["I"], data["R"]

    final_size = int(N - S[-1])

    result = {
        "run": run_number,
        "seed": seed,
        "tau": tau,
        "delay": delay,
        "p": p,
        "q": q,
        "report_time": report_time,
        "intervention_time": tau,
        "reported_infected": len(reported_infected),
        "infected_at_intervention": len(infected_tau),
        "correctly_reported": correct,
        "stale_reports": stale,
        "missed_cases": missed,
        "edges_removed": edges_removed,
        "final_epidemic_size": final_size,
        "final_epidemic_fraction": final_size / N,
    }

    return result


def run_repeated_sims(N, k, beta, gamma, rho, tau, delay, p, q, tmax,
                     n_runs=10):
    """
    Run hybrid intervention simulation multiple times.
    """
    records = []

    for run_num in range(1, n_runs + 1):
        seed = 1000 * run_num

        result = run_hybrid_intervention(
            N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau, delay=delay,
            p=p, q=q, tmax=tmax, run_number=run_num, seed=seed
        )

        records.append(result)

    return pd.DataFrame(records)


def _worker_combo(args):
    """
    Worker function for parallel processing of parameter combinations.

    Args:
        args: tuple of (N, k, beta, gamma, rho, tau, delay, p, q, tmax, n_runs,
              combo_num, total_combos)

    Returns:
        dict with aggregated results for this parameter combination
    """
    (N, k, beta, gamma, rho, tau, delay, p, q, tmax, n_runs,
     combo_num, total_combos) = args

    # Run repeated simulations for this parameter combo
    df = run_repeated_sims(
        N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau, delay=delay,
        p=p, q=q, tmax=tmax, n_runs=n_runs
    )

    mean_metric = df['final_epidemic_fraction'].mean()
    std_metric = df['final_epidemic_fraction'].std()

    print(f"[{combo_num}/{total_combos}] tau={tau}, delay={delay}, "
          f"p={p:.2f}, q={q:.2f} "
          f"mean={mean_metric:.4f} ± {std_metric:.4f}")

    return {
        "tau": tau,
        "delay": delay,
        "p": p,
        "q": q,
        "mean_final_epidemic_fraction": mean_metric,
        "std_final_epidemic_fraction": std_metric,
    }


def parameter_sweep(N, k, beta, gamma, rho, tau_vals, delay_vals, p_vals,
                    q_vals, tmax, n_runs=10, metric="final_epidemic_fraction",
                    save_path=None, n_workers=None, use_multiprocessing=True):
    """
    Optimized parameter sweep across (tau, delay, p, q) with multiprocessing support.

    Args:
        N: population size
        k: regular graph degree
        beta: transmission rate
        gamma: recovery rate
        rho: initial proportion infected
        tau_vals: list of intervention times
        delay_vals: list of reporting delays
        p_vals: list of probabilities for informed edge removal
        q_vals: list of proportions of edges to remove
        tmax: simulation end time
        n_runs: number of runs per parameter combination
        metric: which metric to average (default: final_epidemic_fraction)
        save_path: optional path to save results CSV
        n_workers: number of worker processes (default: CPU count)
        use_multiprocessing: whether to use multiprocessing (default: True)

    Returns:
        DataFrame with sweep results
    """
    total_combos = len(tau_vals) * len(delay_vals) * len(p_vals) * len(q_vals)

    # Create list of all parameter combinations with indices
    combo_list = []
    combo_num = 0

    for tau in tau_vals:
        for delay in delay_vals:
            for p_val in p_vals:
                for q_val in q_vals:
                    combo_num += 1
                    combo_list.append((
                        N, k, beta, gamma, rho, tau, delay, p_val, q_val,
                        tmax, n_runs, combo_num, total_combos
                    ))

    print(f"Total parameter combinations: {total_combos}")
    print(f"Using multiprocessing: {use_multiprocessing}")

    if use_multiprocessing:
        # Use multiprocessing for parallel execution
        with Pool(processes=n_workers) as pool:
            records = pool.map(_worker_combo, combo_list)
    else:
        # Sequential execution
        records = [_worker_combo(args) for args in combo_list]

    sweep_df = pd.DataFrame(records)

    if save_path is not None:
        sweep_df.to_csv(save_path, index=False)
        print(f"\nSaved sweep results to {save_path}")

    return sweep_df


if __name__ == "__main__":
    # Example usage
    p_values = list(np.linspace(0, 1, 10))
    q_values = list(np.linspace(0, 0.5, 10))
    tau_values = [2, 2.5, 3, 3.5, 4, 4.5, 5]
    delay_values = [0.0, 0.5, 1, 1.5, 2, 2.5, 3]

    hybrid_sweep = parameter_sweep(
        N=1000, k=3, beta=0.4, gamma=0.3, rho=0.05, tau_vals=tau_values,
        delay_vals=delay_values, p_vals=p_values, q_vals=q_values, tmax=100,
        n_runs=500, metric="final_epidemic_fraction", save_path="hybrid_sweep_results_0.25.csv", n_workers=6,
        use_multiprocessing=True)

    print(hybrid_sweep.head(20))

