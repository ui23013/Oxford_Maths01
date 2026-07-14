import networkx as nx
import matplotlib.pyplot as plt
import EoN as eon
import random

N, k= 1000, 3 # number of nodes, node degree
beta, gamma = 0.4, 0.1 # transmission, recovery rates respectively

rho = 0.05 # fraction of initially infected
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


    return infected_nodes


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
        return

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



def simulate_incomp_info_SIR(N, k, beta, gamma, rho, tau, p, q):
    '''
    Simulate the SIR model which models having incomplete information
    Args:
        p: probability of informed removal
        q: proportion of nodes to be removed

    Returns: SIR simulation for incomplete information case
    '''

    # create rrg graph
    sim_g = create_rrg(k, N)

    # simulate until time tau and extract the infected nodes
    infected_nodes = run_first_sim(sim_g, beta, gamma, rho, tau)

    first_edge_removal = edge_removal(sim_g, p, q, infected_nodes)




