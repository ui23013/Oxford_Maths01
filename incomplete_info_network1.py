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

    recovered_nodes = [node for node, status in first_states.items() if status == 'R']
    return infected_nodes, recovered_nodes

def simulate_incomp_info_SIR(N, k, beta, gamma, rho, tau, p, q):
    '''
    Simulate the SIR model which models having incomplete information
    Args:
        N: no. of nodes
        k: degree of node
        beta: transmission rate
        gamma: recovery rate
        rho: proportion of population initially infected
        tau: info time
        p: probability of informed removal
        q: proportion of nodes to be removed

    Returns: SIR simulation for incomplete information case
    '''

    # create rrg graph
    sim_g = create_rrg(k, N)

    # simulate until time tau and extrac
    first_sim = eon.fast_SIR(sim_g, beta, gamma, rho=rho, tmax=tau, return_full_data=True)




