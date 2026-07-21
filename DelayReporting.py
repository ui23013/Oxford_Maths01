# SIR network model with reporting delay
#
# At time tau, edges around reported infected nodes are removed.
# With reporting delay delta, the available infection data come from tau - delta.

import random

import EoN
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from plotting_style import set_plot_style


set_plot_style()
plt.rcParams["text.usetex"] = False


# Parameters

N = 2500
DEGREE = 4

BETA = 0.5
GAMMA = 0.2
INITIAL_INFECTED = 125

TAU = 5.0
DELAYS = [0.0, 0.5, 1.0, 2.0, 3.0]

N_RUNS = 1
TMAX = 100.0
BASE_SEED = 50

RESULTS_FILE = "reporting_delay_results.csv"
SUMMARY_FILE = "reporting_delay_summary.csv"


def make_network(seed):
    return nx.random_regular_graph(DEGREE, N, seed=seed)


def choose_initial_cases(G, seed):
    rng = random.Random(seed)
    return rng.sample(list(G.nodes()), INITIAL_INFECTED)


def run_sir(G, initial_infected, seed, initial_recovered=None,
            tmin=0.0, tmax=TMAX):
    random.seed(seed)
    np.random.seed(seed)

    if initial_recovered is None:
        initial_recovered = []

    return EoN.Gillespie_SIR(
        G,
        tau=BETA,
        gamma=GAMMA,
        initial_infecteds=initial_infected,
        initial_recovereds=initial_recovered,
        tmin=tmin,
        tmax=tmax,
        return_full_data=True
    )


def remove_reported_edges(G, reported_infected):
    G_new = G.copy()
    edges = set()

    for u in reported_infected:
        for v in G.neighbors(u):
            edges.add(tuple(sorted((u, v))))

    G_new.remove_edges_from(edges)
    return G_new, edges


def join_trajectories(first_sim, second_sim):
    t1, data1 = first_sim.summary()
    t2, data2 = second_sim.summary()

    keep = t1 < TAU

    t = np.concatenate((t1[keep], t2))
    S = np.concatenate((data1["S"][keep], data2["S"]))
    I = np.concatenate((data1["I"][keep], data2["I"]))
    R = np.concatenate((data1["R"][keep], data2["R"]))

    return t, S, I, R


def run_delay(G, baseline, delay, run_number, seed):
    report_time = max(0.0, TAU - delay)

    report_status = baseline.get_statuses(time=report_time)
    tau_status = baseline.get_statuses(time=TAU)

    reported = [n for n, state in report_status.items() if state == "I"]
    infected_tau = [n for n, state in tau_status.items() if state == "I"]
    recovered_tau = [n for n, state in tau_status.items() if state == "R"]

    reported_set = set(reported)
    infected_set = set(infected_tau)

    correct = reported_set & infected_set
    stale = reported_set - infected_set
    missed = infected_set - reported_set

    G_intervened, removed_edges = remove_reported_edges(G, reported)

    if infected_tau:
        continuation = run_sir(
            G_intervened,
            infected_tau,
            seed=seed,
            initial_recovered=recovered_tau,
            tmin=TAU
        )
        t, S, I, R = join_trajectories(baseline, continuation)
    else:
        t, data = baseline.summary()
        S, I, R = data["S"], data["I"], data["R"]

    final_size = int(N - S[-1])

    result = {
        "run": run_number,
        "seed": seed,
        "delay": delay,
        "report_time": report_time,
        "intervention_time": TAU,
        "reported_infected": len(reported),
        "infected_at_intervention": len(infected_tau),
        "correctly_reported": len(correct),
        "stale_reports": len(stale),
        "missed_cases": len(missed),
        "edges_removed": len(removed_edges),
        "final_epidemic_size": final_size,
        "final_epidemic_fraction": final_size / N
    }

    trajectory = {
        "time": t,
        "S": S,
        "I": I,
        "R": R,
        "delay": delay,
        "report_time": report_time
    }

    return result, trajectory


def run_simulations():
    results = []
    examples = {}

    for run in range(1, N_RUNS + 1):
        seed = BASE_SEED + run

        G = make_network(seed)
        initial_cases = choose_initial_cases(G, seed + 1000)
        baseline = run_sir(G, initial_cases, seed + 2000)

        for j, delay in enumerate(DELAYS):
            result, trajectory = run_delay(
                G,
                baseline,
                delay,
                run_number=run,
                seed=seed + 10000 + j
            )

            results.append(result)

            if run == 1:
                examples[delay] = trajectory

        if run % 10 == 0:
            print(f"Completed {run} of {N_RUNS} runs")

    return pd.DataFrame(results), examples


def make_summary(results):
    return (
        results.groupby("delay")
        .agg(
            mean_final_size=("final_epidemic_size", "mean"),
            mean_final_fraction=("final_epidemic_fraction", "mean"),
            standard_deviation=("final_epidemic_fraction", "std"),
            mean_reported_infected=("reported_infected", "mean"),
            mean_infected_at_intervention=("infected_at_intervention", "mean"),
            mean_edges_removed=("edges_removed", "mean"),
            mean_stale_reports=("stale_reports", "mean"),
            mean_missed_cases=("missed_cases", "mean")
        )
        .reset_index()
    )


def plot_trajectory(traj):
    delay = traj["delay"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(traj["time"], traj["S"], linewidth=2, label="Susceptible")
    ax.plot(traj["time"], traj["I"], linewidth=2, label="Infected")
    ax.plot(traj["time"], traj["R"], linewidth=2, label="Recovered")

    ax.axvline(
        traj["report_time"],
        linestyle=":",
        linewidth=1.8,
        label=r"Report time $\tau-\delta$"
    )
    ax.axvline(
        TAU,
        linestyle="--",
        linewidth=1.8,
        label=r"Intervention time $\tau$"
    )

    ax.set(
        xlabel="Time",
        ylabel="Number of nodes",
        title=fr"SIR epidemic with reporting delay $\delta={delay}$",
        xlim=(0, None),
        ylim=(0, N)
    )

    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    delay_name = str(delay).replace(".", "_")
    filename = f"trajectory_delay_{delay_name}.pdf"

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved figure: {filename}")


def main():
    results, examples = run_simulations()

    results.to_csv(RESULTS_FILE, index=False)

    summary = make_summary(results)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nMean results by reporting delay:\n")
    print(
        summary[
            [
                "delay",
                "mean_final_fraction",
                "mean_edges_removed",
                "mean_stale_reports",
                "mean_missed_cases"
            ]
        ].to_string(index=False)
    )

    print(f"\nDetailed results saved to {RESULTS_FILE}")
    print(f"Summary saved to {SUMMARY_FILE}\n")

    for delay in DELAYS:
        plot_trajectory(examples[delay])


if __name__ == "__main__":
    main()
