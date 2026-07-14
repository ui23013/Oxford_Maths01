import networkx as nx
import matplotlib.pyplot as plt
import EoN as eon
import random

random.seed(0)

def create_rrg(k, N):
    '''
    creates RRG
    Args:
        k: node degree
        N: no. of nodes
    '''
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
    sim = eon.fast_SIR(graph, beta, gamma, rho=rho, tmax=tau, return_full_data=True)

    # and then extract states at intervention time, tau
    first_states = sim.get_statuses(time=tau)

    # retrieve infected and recovered nodes at time tau
    infected_nodes = [node for node, status in first_states.items() if status == 'I']

    # don't know if this is needed
    # recovered_nodes = [node for node, status in first_states.items() if status == 'R']


    return infected_nodes, first_states


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

    return graph

def run_second_sim(graph, beta, gamma, initial_states,tmin, tmax):
    # doesnt work, cant start fast_sir mid sim
    # sim = eon.fast_SIR(graph, beta, gamma, status=initial_states,
                       #tmin=tmin, tmax=tmax, return_full_data=True)

    # extract which nodes were infected/recovered at time tmin
    I_nodes = [node for node, status in initial_states.items() if status == 'I']
    R_nodes = [node for node, status in initial_states.items() if status == 'R']

    sim = eon.fast_SIR(graph, beta, gamma, initial_infecteds=I_nodes, tmax = tmax, return_full_data = True)
    # this isnt considering an 'initial_recoverds' so need to fix that
    return sim


def sim_single_intervention(N, k, beta, gamma, rho, tau, p, q, sim_duration):
    '''
    simulate the SIR model which models having incomplete information
    '''

    # create rrg graph
    sim_g = create_rrg(k, N)
    initial_edges = sim_g.number_of_edges()

    # simulate until time tau and extract the infected nodes
    infected_nodes, first_states = run_first_sim(sim_g, beta, gamma, rho, tau)

    # obtain graph after edge removal
    modified_graph = edge_removal(sim_g, p, q, infected_nodes)
    edges_removed = initial_edges - modified_graph.number_of_edges()

    # and run a second sum to see results of intervention
    second_sim = run_second_sim(modified_graph, beta, gamma, first_states, tmin=tau, tmax=sim_duration)
    final_states = second_sim.get_statuses(time=sim_duration) # states at final time

    # outcomes
    final_infected = sum(1 for status in final_states.values() if status == 'I')
    final_recovered = sum(1 for status in final_states.values() if status == 'R')
    total_infected = final_infected + final_recovered

    return modified_graph, edges_removed, final_infected, final_recovered, total_infected


# !!! the recovered people at time tau are being neglected
no_intervention = sim_single_intervention(N=1000, k=3, beta=0.4, gamma=0.1, rho=0.05, tau=20, p=1, q=0, sim_duration=100)
print(no_intervention) # shows edges removed, final_infected = 0 and final_recovered, final_infected = n
# perfect information
test_run_p1 = sim_single_intervention(N=1000, k=3, beta=0.4, gamma=0.1, rho=0.05, tau=20, p=1, q=0.1, sim_duration=100)
print(test_run_p1)
# random intervention
test_run_p0= sim_single_intervention(N=1000, k=3, beta=0.4, gamma=0.1, rho=0.05, tau=20, p=0, q=0.1, sim_duration=100)
print(test_run_p0)