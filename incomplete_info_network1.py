import networkx as nx
import matplotlib.pyplot as plt
import EoN as eon
import random
import pandas as pd


def create_rrg(k, N, seed):
    '''
    creates RRG
    Args:
        k: node degree
        N: no. of nodes
    '''
    graph = nx.random_regular_graph(k, N, seed=seed)
    return graph


def run_first_sim(graph, beta, gamma, rho, tau):
    '''
    runs simulation until first intervention time and returns the infected
    and recovered nodes at time tau
    Args:
        beta: transmission rate
        gamma: recovery rate
        rho: proportion of population that's initially infected
        tau: intervention time
    '''
    # simulate until first intervention
    sim = eon.Gillespie_SIR(graph, beta, gamma, rho=rho, tmax=tau, return_full_data=True)

    # and then extract states at intervention time, tau
    first_states = sim.get_statuses(time=tau)

    # retrieve infected and recovered nodes at time tau
    infected_nodes = [node for node, status in first_states.items() if status == 'I']

    recovered_nodes = [node for node, status in first_states.items() if status == 'R']

    return infected_nodes, recovered_nodes, first_states


def edge_removal(graph, p, q,  i_nodes):
    '''
    removes a proportion of edges from the graph
    with probability p, each removal preferentially chooses a edge adjacent to an infected node,
    otherwise (so with probabilty 1-p OR if there are no such edges) a random edge is removed
    (in the second instance this alters p?)
    Args:
        graph: simualtion graph
        i_nodes: infected nodes from simulation
    '''
    randgen = random.Random()

    # define total edges in the graph and thus how many should be removed
    total_edges = graph.number_of_edges()
    removal_count = round(q * total_edges)

    if removal_count == 0: # remove nothing case
        return graph

    # define set of all infected nodes
    infected_nodes = set(i_nodes)

    for e in range(removal_count):
        # and list of all edges in the graph
        current_edges = list(graph.edges)

        # create set and list of informed edges
        informed_edges = [edge for edge in current_edges
                        if edge[0] in infected_nodes or edge[1] in infected_nodes]

        # preferentially choose edge adjacent to infected node
        if informed_edges and randgen.random() < p:
            removed_edge = randgen.choice(informed_edges)
        else:
            removed_edge = randgen.choice(current_edges)

        graph.remove_edge(*removed_edge)

        # print('Informed edges:', len(informed_edges)) # debug
    return graph


def run_second_sim(graph, beta, gamma, infected_nodes, recovered_nodes,tmin, tmax):

    sim = eon.Gillespie_SIR(graph, beta, gamma, initial_infecteds=infected_nodes, initial_recovereds=recovered_nodes,
                            tmin=tmin, tmax=tmax, return_full_data=True)

    return sim


def sim_single_intervention(N, k, beta, gamma, rho, tau, p, q, sim_duration, seed):
    '''
    simulate the SIR model which models having incomplete information
    '''

    # create rrg graph
    sim_g = create_rrg(k, N, seed)
    initial_edges = sim_g.number_of_edges()

    # simulate until time tau and extract the infected nodes
    infected_nodes, recovered_nodes, first_states = run_first_sim(sim_g, beta, gamma, rho, tau)

    # debug
    # print('Infected nodes', len(infected_nodes))

    # obtain graph after edge removal
    modified_graph = edge_removal(sim_g, p, q, infected_nodes)
    edges_removed = initial_edges - modified_graph.number_of_edges()

    # and run a second sum to see results of intervention
    second_sim = run_second_sim(modified_graph, beta, gamma, infected_nodes, recovered_nodes, tmin=tau, tmax=sim_duration)
    final_states = second_sim.get_statuses(time=sim_duration) # states at final time

    # outcomes
    final_infected = sum(1 for status in final_states.values() if status == 'I')
    final_recovered = sum(1 for status in final_states.values() if status == 'R')
    total_infected = final_infected + final_recovered

    return modified_graph, edges_removed, final_infected, final_recovered, total_infected, len(infected_nodes)


# # set graph seed for reproducibility
# graph_seed = 120
# no_intervention = sim_single_intervention(N=1000, k=3, beta=0.4, gamma=0.1, rho=0.05, tau=20, p=1, q=0, sim_duration=500
#                                           , seed=graph_seed)
# print('No intervention stats', no_intervention)
#
# # random intervention
# test_run_p0= sim_single_intervention(N=1000, k=3, beta=0.4, gamma=0.1, rho=0.05, tau=20, p=0, q=0.1, sim_duration=500
#                                      , seed=graph_seed)
# print('Random intervention stats',test_run_p0)
#
# # perfect information
# test_run_p1 = sim_single_intervention(N=1000, k=3, beta=0.4, gamma=0.1, rho=0.05, tau=20, p=1, q=0.1, sim_duration=500
#                                       , seed=graph_seed)
# print('Perfect intervention stats',test_run_p1)


def run_repeated_sims(N, k, beta, gamma, rho, tau, p, q, sim_duration, seed, n_runs=25):
    records = []
    for _ in range(n_runs):
        (modified_graph, edges_removed, final_infected, final_recovered, total_infected, seed_infected) = (
            sim_single_intervention(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau, p=p, q=q,
                                    sim_duration=sim_duration, seed=seed))

        records.append({'seed_infected': seed_infected, 'edges_removed': edges_removed, 'final_infected':final_infected,
                        'final_recovered': final_recovered, 'total_infected': total_infected,})

    return pd.DataFrame(records)


# run sim multiple times to get data
# df = run_repeated_sims(N=1000, k=3, beta=0.4, gamma=0.1, rho=0.05, tau=20, p=1, q=0.1, sim_duration=500, seed=120,
#                        n_runs=25)
# df['seed_infected'].describe()
# df['total_infected'].describe()

def compare_interventions(N, k, beta, gamma, rho, tau, sim_duration, seed, n_runs):
    df_no = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
                               p=1, q=0.0, sim_duration=sim_duration, seed=seed, n_runs=n_runs)
    df_random = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
                                   p=0, q=0.1, sim_duration=sim_duration, seed=seed, n_runs=n_runs)
    df_perfect = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
                                    p=1, q=0.1, sim_duration=sim_duration, seed=seed, n_runs=n_runs)

    return pd.Series({
        'no_intervention': (df_no['final_recovered'] / N).mean(),
        'random_intervention': (df_random['final_recovered'] / N).mean(),
        'perfect_intervention': (df_perfect['final_recovered'] / N).mean(),
        }, name='mean_r_inf')


r_inf_summary = compare_interventions(N=1000, k=3, beta=0.5, gamma=0.3, rho=0.05, tau=10, sim_duration=1000, seed=120,
                                      n_runs=100)
print(r_inf_summary)