import networkx as nx
import matplotlib.pyplot as plt
import EoN as eon
import random
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial


def create_rrg(k, N):
    graph = nx.random_regular_graph(k, N)
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
        current_edges = list(graph.edges())

        if not current_edges:
            break

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
    '''
    run simultation from time tau to tmax
    '''
    sim = eon.Gillespie_SIR(graph, beta, gamma, initial_infecteds=infected_nodes, initial_recovereds=recovered_nodes,
                            tmin=tmin, tmax=tmax, return_full_data=True)

    return sim


def sim_single_intervention(N, k, beta, gamma, rho, tau, p, q, sim_duration):
    '''
    simulate the SIR model which models having incomplete information
    '''

    # create rrg graph
    sim_g = create_rrg(k, N)
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

    # print(final_infected) # to ensure epidemic is in fact over

    total_infected = final_infected + final_recovered

    return modified_graph, edges_removed, final_infected, final_recovered, total_infected, len(infected_nodes)


# github copilot
def _run_single_sim(_, N, k, beta, gamma, rho, tau, p, q, sim_duration):
    '''Helper function for parallelization'''
    return sim_single_intervention(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
                                   p=p, q=q, sim_duration=sim_duration)


def run_repeated_sims(N, k, beta, gamma, rho, tau, p, q, sim_duration, n_runs, n_workers=None):
    '''
    runs same simulation multiple times
    Args:
        sim_duration: how long the sim should be run from
        seed: graph no.
        n_runs: how many repeats
        n_works: number of parallel works
    '''

    run_sim = partial(_run_single_sim, N=N, k=k, beta=beta, gamma=gamma, rho=rho,
                      tau=tau, p=p, q=q, sim_duration=sim_duration)

    with Pool(n_workers) as pool:
        results = pool.map(run_sim, range(n_runs))

    records = [
        {
            'seed_infected': seed_infected,
            'edges_removed': edges_removed,
            'final_infected': final_infected,
            'final_recovered': final_recovered,
            'total_infected': total_infected
        }
        for modified_graph, edges_removed, final_infected, final_recovered, total_infected, seed_infected in results
    ]

    # for _ in range(n_runs):
    #     (modified_graph, edges_removed, final_infected, final_recovered, total_infected, seed_infected) = (
    #         sim_single_intervention(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau, p=p, q=q,
    #                                 sim_duration=sim_duration))
    #
    #     records.append({'seed_infected': seed_infected, 'edges_removed': edges_removed, 'final_infected':final_infected,
    #                     'final_recovered': final_recovered, 'total_infected': total_infected,})

    return pd.DataFrame(records)


def compare_interventions(N, k, beta, gamma, rho, tau, sim_duration, n_runs):
    '''
    compare NI, RI and PI on a graph
    Args:
        n_runs: how many comparisons should be done
    '''

    df_no = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
                               p=1, q=0.0, sim_duration=sim_duration,n_runs=n_runs, n_workers=4)
    df_random = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
                                   p=0, q=0.1, sim_duration=sim_duration, n_runs=n_runs, n_workers=4)
    df_perfect = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
                                    p=1, q=0.1, sim_duration=sim_duration, n_runs=n_runs, n_workers=4)

    return pd.Series({
        'no_intervention': (df_no['final_recovered'] / N).mean(),
        'random_intervention': (df_random['final_recovered'] / N).mean(),
        'perfect_intervention': (df_perfect['final_recovered'] / N).mean(),
        }, name='mean_r_inf')


# r_inf_summary = compare_interventions(N=1000, k=3, beta=0.5, gamma=0.3, rho=0.05, tau=10, sim_duration=1000, n_runs=5)
# print(r_inf_summary)


def parameter_sweep(N, k_vals, beta_range, gamma_range, rho, sim_duration, n_runs, p_vals, q_vals, tau_vals,
                    metric='final_recovered', save_path=None, n_workers=None):
        '''
        does a parameter sweep across (p, q, tau) and records mean r_inf for a given parameter combination
        Args:
            k_vals: list of possible degree values for graph
            beta_range: min and max beta values for sampling of beta
            gamma_range: same as above but for gamma
            p_vals: list of values for probability of informed edge removal, p
            q_vals: list of values for proportion of edges removed at intervention, q
            tau_vals: list of values for intervention time, tau
            metric: the metric that we find a mean value for, in this instance r_inf
            save_path: to save to device
        '''
        # to optimise, precompute all param combos
        param_combos = [(tau, p_val, q_val) for tau in tau_vals for p_val in p_vals for q_val in q_vals]
        total_combos = len(param_combos
                           )
        records = [] # initialise empty list to store data
        combo_num = 0

        for combo_num, (tau, p_val, q_val) in enumerate(param_combos, 1):

            # randomly sample beta and gamma
            beta = np.random.uniform(beta_range[0], beta_range[-1])
            gamma = np.random.uniform(gamma_range[0], gamma_range[-1])
            k = np.random.choice(k_vals)

            # for given combination, repeatedly run an sir sim
            df = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau, p=p_val, q=q_val
                                    ,sim_duration=sim_duration, n_runs=n_runs, n_workers=n_workers)

            # and get average r_inf value (as % of population) vectorised
            mean_r_inf = (df[metric]/N).mean()
            # append param combo and corresponding mean r_inf value plus beta, gamma and k used
            records.append({'p': p_val, 'q': q_val, 'tau': tau,  'k': k, 'beta': beta, 'gamma': gamma,
                            'mean_r_inf': mean_r_inf})

            print(f'[{combo_num}/{total_combos}] p={p_val}, q={q_val}, tau={tau}, k={k}, beta={beta:.4f}, gamma={gamma:.4f} '
                    f'mean_r_inf={mean_r_inf:.4f}')

        sweep_df = pd.DataFrame(records)

        if save_path is not None:
            sweep_df.to_csv(save_path, index=False)
            print(f'Saved sweep results to {save_path}')

        return sweep_df


if __name__ == '__main__':
    p_values = list(np.linspace(0, 1, 10))
    q_values = list(np.linspace(0, 0.5, 10))
    tau_values = list(range(10, 21, 10))
    # print(tau_values)

    # min and max values for beta and gamma
    beta_range = (0.25, 0.55)
    gamma_range = (0.1, 0.4)
    k_range = [3, 4, 5]
    num_workers = 4

    trial_sweep = parameter_sweep(N=2500, k_vals=k_range, beta_range=beta_range, gamma_range=gamma_range, rho=0.05,
                                  sim_duration=500,
                                  n_runs=10, p_vals=p_values, q_vals=q_values, tau_vals=tau_values,
                                  save_path='trial5_sweep_results.csv', n_workers=num_workers)