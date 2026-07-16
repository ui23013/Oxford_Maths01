import random
import EoN as eon
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ------ parameters ----------

N, k = 500, 4  # number of nodes, node degree
beta, gamma = 0.4, 0.1  # transmission, recovery rates respectively
# not setting initial infected as it raises error if both this and rho are set
rho = 0.05 # fraction of initially infected
tmin, tmax = 0, 100 # start and end time of sim
INTERVENTION_TIME = 5
REPORT_DELAY = 2
SEED = 1

# ----- network ----------------

def nodes_in_state(statuses, state):
    return [
        node
        for node, value in statuses.items()
        if value == state
    ]


random.seed(SEED)
np.random.seed(SEED)

g = nx.random_regular_graph(k, N, seed=SEED)

plt.figure(figsize=(6, 6))
nx.draw(g, node_size=20)
plt.title("Random Regular Graph")
plt.show()


# ------ epidemic before intervention ------

sim_before = eon.Gillespie_SIR(
    g,
    tau=beta,
    gamma=gamma,
    rho=rho,
    tmin=tmin,
    tmax=INTERVENTION_TIME,
    return_full_data=True,
)

t_before, data_before = sim_before.summary()

report_time = max(tmin, INTERVENTION_TIME - REPORT_DELAY)

# Delayed information/
reported_statuses = sim_before.get_statuses(time=report_time)
reported_cases = nodes_in_state(reported_statuses, "I")

# state when the intervention actually happens
current_statuses = sim_before.get_statuses(time=INTERVENTION_TIME)
current_infected = nodes_in_state(current_statuses, "I")
current_recovered = nodes_in_state(current_statuses, "R")


# ------- remove edges around delayed reported cases -----------

g_after = g.copy()

edges_to_remove = {
    tuple(sorted((case, neighbour)))
    for case in reported_cases
    for neighbour in g.neighbors(case)
}

g_after.remove_edges_from(edges_to_remove)


# ------- continuation of epidemic -----

if current_infected:
    sim_after = eon.Gillespie_SIR(
        g_after,
        tau=beta,
        gamma=gamma,
        initial_infecteds=current_infected,
        initial_recovereds=current_recovered,
        tmin=INTERVENTION_TIME,
        tmax=tmax,
        return_full_data=True,
    )

    t_after, data_after = sim_after.summary()

    mask = t_before < INTERVENTION_TIME

    t = np.concatenate((t_before[mask], t_after))
    S = np.concatenate((data_before["S"][mask], data_after["S"]))
    I = np.concatenate((data_before["I"][mask], data_after["I"]))
    R = np.concatenate((data_before["R"][mask], data_after["R"]))

else:
    # The epidemic has already ended by the intervention time.
    t, S, I, R = (
        t_before,
        data_before["S"],
        data_before["I"],
        data_before["R"],
    )


# -------- plot epidemic dynamics ------------

plt.figure(figsize=(9, 5))
plt.plot(t, S, label="Susceptible")
plt.plot(t, I, label="Infected")
plt.plot(t, R, label="Recovered")
plt.axvline(
    INTERVENTION_TIME,
    linestyle="--",
    label="Intervention",
)

plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Network SIR with reporting delay")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------- delay stuff ----------------

reported_set = set(reported_cases)
current_set = set(current_infected)

stale_reports = reported_set - current_set
missed_cases = current_set - reported_set

print(f"Intervention time: {INTERVENTION_TIME}")
print(f"Reporting delay: {REPORT_DELAY}")
print(f"Report uses infection data from time: {report_time}")
print(f"Reported infected cases: {len(reported_cases)}")
print(f"Actually infected at intervention: {len(current_infected)}")
print(f"Stale reports: {len(stale_reports)}")
print(f"Missed current cases: {len(missed_cases)}")
print(f"Edges removed: {len(edges_to_remove)}")
print(f"Final epidemic size: {N - S[-1]} ({(N - S[-1]) / N:.3f})")